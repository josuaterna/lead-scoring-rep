import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from src.data_loader import subir_archivo
from pathlib import Path

FOLDER_DATA = Path(__file__).parent
client = MlflowClient()

st.set_page_config(layout="wide")
st.title("Lead Scoring")

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
                if st.form_submit_button("Crear"):
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
        models_in_exp = []
        
        for run in runs:
            # Buscamos si el run tiene modelos registrados con alias 'production'
            # Nota: Esto asume que el modelo se registró desde un run de este experimento
            filter_string = f"run_id = '{run.info.run_id}'"
            mv_list = client.search_model_versions(filter_string)
            for mv in mv_list:
                try:
                    # Verificamos si esa versión específica tiene el alias 'production'
                    aliases = client.get_model_version(mv.name, mv.version).aliases
                    if "production" in aliases:
                        if mv.name not in models_in_exp:
                            models_in_exp.append(mv.name)
                except:
                    continue

        col_left, col_right = st.columns(2)
        
        with col_left:
            # Lista desplegable (se actualiza al cambiar el experimento)
            st.selectbox("Selecciona un modelo:", ["Choose an option"] + models_in_exp)
            
            # 3. Lógica de advertencia y file loader
            if not models_in_exp:
                st.warning(f"⚠️ No hay modelos en 'production' para: {seleccion_exp_name}")
                #st.file_uploader("Cargar archivo para entrenar nuevo modelo:", type=['csv', 'xlsx'])
                # Inicializar Tkinter oculto
                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True) # Asegura que aparezca al frente
                if st.button('Abrir Explorador de Archivos'):
                    # Abre la ventana de búsqueda y captura el string de la ruta
                    ruta = filedialog.askopenfilename(master=root)
                    if ruta:
                        st.session_state.ruta_manual = ruta

                # Campo de texto editable vinculado a la selección
                if 'ruta_manual' in st.session_state:
                    ruta_origen = st.text_input("Ruta seleccionada:", value=st.session_state.ruta_manual)
                    orig_path = Path(ruta_origen)
                    final_path = FOLDER_DATA.parent / "mlruns" / target_exp_id / f"{target_exp_id}.csv"
                    status_text = st.empty()
                    def mi_progreso(n_chunk, n_filas):
                        status_text.info(f"📊 Procesando fragmento #{n_chunk} | Filas leídas: {n_filas:,}")
                    try:
                        total = subir_archivo(ruta_origen, final_path, progress_callback=mi_progreso)
                        st.success(f"Archivo guardado: {total} filas")
                    except Exception as e:
                        st.error(f"Error: {e}")
                
                with col_right:
                    if col_b2.button("Nuevo", key="btn_n_mod"):
                        st.session_state.mostrar_form_mod = True

                    if st.session_state.get('mostrar_form_mod', False):
                        with st.form("form_nuevo_mod"):
                            nombre_m = st.text_input("Nombre model:")
                            if st.form_submit_button("Enviar"):
                                client.create_registered_model(nombre_m)
                                st.success(f"Modelo {nombre_m} creado.")
                                st.session_state.mostrar_form_mod = False
