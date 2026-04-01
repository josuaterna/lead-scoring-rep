import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from src.data_loader import subir_archivo
from scripts.train import train_big_data
from scripts.mlfunc import promote_if_better
from scripts.mlfunc import batch_predict_to_disk
from pathlib import Path
from PIL import Image

FOLDER_DATA = Path(__file__).parent
client = MlflowClient()
fav = Image.open(FOLDER_DATA / "img" / "favicon.png" )
logo = Image.open(FOLDER_DATA / "img" / "logo.png" )

#st.set_page_config(layout="wide", page_icon=fav, page_title="Lead Scoring")
st.set_page_config(page_icon=fav, page_title="Lead Scoring")

def load_file(file_name, exp_id):
    #st.subheader(f"Cargando datos para: {file_name}")
    # En lugar de otro formulario, usa un botón simple para disparar el diálogo de archivos
    if st.button("Seleccionar Archivo CSV", key="btn_explore"):
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        ruta = filedialog.askopenfilename(master=root)
        root.destroy() # Importante cerrar la instancia de tk
        
        if ruta:
            st.session_state.ruta_manual = ruta

    # Si ya tenemos la ruta, procedemos al procesamiento
    if 'ruta_manual' in st.session_state:
        ruta_origen = st.text_input("Ruta seleccionada:", value=st.session_state.ruta_manual)
        final_path = FOLDER_DATA.parent / "mlruns" / exp_id / f"{file_name}.csv"
        
        if st.button("Confirmar y Procesar",key="btn_load"):
            status_text = st.empty()
            def mi_progreso(n_chunk, n_filas):
                status_text.info(f"📊 Procesando fragmento #{n_chunk} | Filas leídas: {n_filas:,}")
            
            try:
                total = subir_archivo(ruta_origen, final_path, progress_callback=mi_progreso)
                st.success(f"¡Archivo guardado: {total} filas!")
                return True
            except Exception as e:
                st.error(f"Error: {e}")
                return False

def train_ui(exp_id, exp_name, mod_name, file_name):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 3. Definir la función callback que se enviará al módulo externo
        def actualizar_ui(valor, texto):
            progress_bar.progress(valor)
            status_text.text(texto)
            
        # 4. Llamar a la función del otro módulo pasando el callback
        with st.spinner("Ejecutando pipeline..."):
            file_path = FOLDER_DATA.parent / "mlruns" / exp_id / f"{file_name}.csv"
            run_id, new_auc = train_big_data(file_path, exp_name, mod_name, "contacto_positivo", progress_callback=actualizar_ui)
            
        st.success(f"Modelo entrenado. Run ID: {run_id} - AUC: {new_auc:.4f}")
        # run_id, new_auc = train(file_path, exp_names.name, texto_m, "contacto_positivo")
        # st.success(f"Entrenamiento finalizado.")
        promoted, old_auc = promote_if_better(exp_name, mod_name, run_id, new_auc)
        st.success(f"Entrenamiento finalizado. Promovido: {promoted} - AUC anterior/nuevo: {old_auc} / {new_auc}")

    except Exception as e:
        st.warning(f"Error entrenando modelo {e}")

def score_ui(run_id, path_in, path_out):
    print("Inicio score_ui")
    print(run_id)
    print(path_in)
    print(path_out)
    if path_in.exists():
        if not run_id:
            st.error("Error: No hay un modelo seleccionado.")
        else:
            print(" Eliminar el archivo de salida previo si existe para empezar de cero")
            if path_out.exists():
                path_out.unlink(missing_ok=True)
            with st.spinner("Scoring..."):
                try:
                    print(" Ejecutar la predicción pesada")
                    batch_predict_to_disk(
                        run_id=run_id,
                        input_csv_path=path_in,
                        output_csv_path=path_out
                    )
                    st.success(f"Procesamiento finalizado.")
                    
                    # Botón de descarga: Lee del archivo físico
                    if path_out.exists():
                        try:
                            with open(path_out, "rb") as file:
                                st.download_button(
                                    label="📥 Descargar Resultado Final",
                                    data=file,
                                    file_name=path_out.name,
                                    mime="text/csv",
                                    key="btn_dl"
                                )
                        except Exception as e:
                            st.error(f"Error al descargar: {e}")
                except Exception as e:
                    st.error(f"Ocurrió un error: {e}")

st.divider()
with st.container():
    col_logo, col_text = st.columns([5 , 10])
    with col_logo:
        if logo:
            st.image(logo)
        else:
            st.error(f"No se encontró el logo.")
    with col_text:
        st.title("Lead Scoring")
        st.caption("Soluciones Inteligentes de Contact Center & BPO")


# --- LÓGICA DE EXPERIMENTOS ---
with st.container(border=True):
    col_t1, col_b1 = st.columns([0.8, 0.2])
    col_t1.subheader("EXPERIMENTS")
    
    # 1. Omitir el experimento "Default" (ID '0')
    all_exps = [e for e in client.search_experiments() if e.name != "Default"]
    exp_options = {e.name: e.experiment_id for e in all_exps}
    
    col_exp, col_nuevo = st.columns(2)
    with col_exp:
        # Al cambiar este selectbox, Streamlit recarga automáticamente la sección inferior
        seleccion_exp_name = st.selectbox(
            "Selecciona un experiment:", 
            ["Choose an option"] + list(exp_options.keys()),
            key="main_exp_selector" 
        )

    with col_nuevo:
        if col_b1.button("Nuevo", key="btn_n_exp"):
            st.session_state.crear_exp = True
            
        if st.session_state.get('crear_exp', False):
            with st.form("form_exp"):
                n_exp = st.text_input("Nombre experiment:")
                if st.form_submit_button("Crear", key= "btn_n_exp_conf"):
                    mlflow.create_experiment(n_exp)
                    st.session_state.crear_exp = False
                    st.rerun()

# --- LÓGICA DE MODELS (FILTRADO POR EXPERIMENTO) ---
if seleccion_exp_name != "Choose an option":
    target_exp_id = exp_options[seleccion_exp_name]
    
    with st.container(border=True):
        col_t2, col_b2 = st.columns([0.8, 0.2])
        col_t2.subheader("MODELS")
        
        # 2. Filtrar modelos asociados al experimento seleccionado
        # Buscamos runs del experimento que tengan modelos registrados
        runs = client.search_runs(experiment_ids=[target_exp_id])
        models_dict = {}
        for run in runs:
            current_run_id = run.info.run_id
            # Buscamos si el run tiene modelos registrados con alias 'production'
            # Nota: Esto asume que el modelo se registró desde un run de este experimento
            filter_string = f"run_id = '{run.info.run_id}'"
            mv_list = client.search_model_versions(filter_string)
            for mv in mv_list:
                try:
                    # Verificamos si esa versión específica tiene el alias 'production'
                    aliases = client.get_model_version(mv.name, mv.version).aliases
                    if mv.name not in models_dict:
                        models_dict[mv.name] = current_run_id
                except:
                    continue
            # 1. Inicializar estados
        if 'form_lvl' not in st.session_state:
            st.session_state.form_lvl = 0                
        col_left, col_right = st.columns(2)
        nombre_m = None
        with col_left:
            # Lista desplegable (se actualiza al cambiar el experimento)
            if models_dict:
                nombres_modelos = list(models_dict.keys())
                nombre_m = st.selectbox("Selecciona un modelo:", ["Choose an option"] + nombres_modelos)
            # 3. Lógica de advertencia y file loader
            else:
                st.warning(f"⚠️ No hay modelos en 'production' para: {seleccion_exp_name}")

            if nombre_m and nombre_m != "Choose an option":
                st.session_state.run_id_modelo = models_dict[nombre_m]
                st.session_state.nombre_m = nombre_m
                col_left_l, col_left_r = st.columns(2)
                with col_left_l:
                    if col_left_l.button("Train", key="btn_train"):
                        st.session_state.nombre_archivo = nombre_m.replace("_model", "_file")
                        st.session_state.form_lvl = 2 # Pasamos al siguiente nivel
                        st.rerun()
                with col_left_r:
                    if col_left_r.button("Score", key="btn_score"):
                        st.session_state.nombre_archivo = nombre_m.replace("_model", "_score_in")
                        st.session_state.nombre_archivo_out = nombre_m.replace("_model", "_score_out")
                        path_in = FOLDER_DATA.parent / "mlruns" / target_exp_id / f"{st.session_state.nombre_archivo}.csv"
                        path_out = FOLDER_DATA.parent / "mlruns" / target_exp_id / f"{st.session_state.nombre_archivo_out}.csv"
                        st.session_state.path_in = path_in
                        st.session_state.path_out = path_out
                        st.session_state.form_lvl = 4 # Pasamos al siguiente nivel
                        st.rerun()

        with col_right:
            if col_b2.button("Nuevo", key="btn_n_mod"):
                st.session_state.form_lvl = 1
                st.rerun()

            # --- CONTROL DE NIVELES ---

            # NIVEL 1: Formulario de Nombre
        if st.session_state.form_lvl == 1:
            with col_right:
                with st.form("form_nuevo_mod", clear_on_submit=True):
                    nombre_m = st.text_input("Nombre model:")
                    enviado_m = st.form_submit_button("Enviar", key="btn_nu_mod")
                    if enviado_m:
                        if nombre_m:
                            st.session_state.nombre_m = f"{nombre_m}_model"
                            st.session_state.nombre_archivo = f"{nombre_m}_file"
                            st.session_state.form_lvl = 2 # Pasamos al siguiente nivel
                            st.rerun() # Ahora sí, recargamos para mostrar el nivel 2
                        else:
                            st.error("Por favor, ingresa un nombre.")
                            st.rerun()

                # NIVEL 2: Carga de Archivo (Llamada a la función)
        elif st.session_state.form_lvl == 2:
            with col_right:
                if load_file(st.session_state.nombre_archivo, target_exp_id):
                    st.session_state.form_lvl = 3
                    st.rerun()
        elif st.session_state.form_lvl == 3:    
            train_ui(target_exp_id, seleccion_exp_name, st.session_state.nombre_m, st.session_state.nombre_archivo)
            st.session_state.form_lvl = 0
            #st.rerun()
                        
        elif st.session_state.form_lvl == 4:
            if load_file(st.session_state.nombre_archivo, target_exp_id):
                st.session_state.form_lvl = 5
                st.rerun()
        elif st.session_state.form_lvl == 5:
            score_ui(st.session_state.run_id_modelo, st.session_state.path_in, st.session_state.path_out)
            st.session_state.form_lvl = 0
            #st.rerun()