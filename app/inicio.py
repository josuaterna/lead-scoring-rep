import streamlit as st
import pandas as pd
import mlflow
from pathlib import Path
from PIL import Image
from pathlib import Path
from mlflow.tracking import MlflowClient
from scripts.user import crear_tabla
from scripts.user import registrar_usuario
from scripts.user import autenticar_usuario
from scripts.user import cambiar_contrasena
from scripts.mlfunc import train_big_data
from scripts.mlfunc import promote_if_better
from scripts.mlfunc import batch_predict_to_disk
from src.preprocessing import procesar_csv_multiple_filtros
from src.evaluation import graficar_ganancia
from src.evaluation import graficar_ganancia_comparativa


FOLDER_DATA = Path(__file__).parent

fav = Image.open(FOLDER_DATA / "img" / "favicon.png" )
logo = Image.open(FOLDER_DATA / "img" / "logo.png" )
st.set_page_config(page_icon=fav, page_title="Lead Scoring", layout="wide")

def modulo_login():
    st.title("Lead Scoring")
    st.caption("Soluciones Inteligentes de Contact Center & BPO")
    crear_tabla()
    if 'autenticado' not in st.session_state:
        st.session_state.autenticado = True
    if 'username' not in st.session_state:
        st.session_state.username = "josuaterna"

    if not st.session_state.autenticado:
        tab1, tab2, tab3 = st.tabs(["Login", "Registro", "Cambiar contraseña"])
        
        with tab1:
            user = st.text_input("Usuario", key="user_n")
            pwd = st.text_input("Contraseña", type="password")
            if st.button("Ingresar"):
                if autenticar_usuario(user, pwd):
                    st.session_state.autenticado = True
                    st.session_state.username = user
                    st.rerun()
                else:
                    st.error("Credenciales inválidas")
                    
        with tab2:
            new_user = st.text_input("Nuevo Usuario")
            new_pwd = st.text_input("Nueva Contraseña", type="password")
            if st.button("Registrarse"):
                if registrar_usuario(new_user, new_pwd):
                    st.success("Usuario registrado")
                else:
                    st.error("El usuario ya existe")
        
        with tab3:
            user = st.text_input("Usuario")
            old_pwd = st.text_input("Contraseña", type="password", key="old_p")
            new_pwd = st.text_input("Nueva Contraseña", type="password", key="new_p")
            conf_pwd = st.text_input("Confirmar Contraseña", type="password", key="conf_p")
            if st.button("Cambiar", key="camb_c"):
                if new_pwd != conf_pwd:
                    st.error("Las contraseñas nuevas no coinciden.")
                elif len(new_pwd) < 8:
                    st.warning("La contraseña debe tener al menos 8 caracteres.")
                else:                    
                    cambio, descripcion = cambiar_contrasena(user, old_pwd, new_pwd)
                    if cambio:
                        st.success(descripcion)
                    else:
                        st.error(descripcion)
    else:
        st.write(f"Bienvenido al sistema de Outsourcing SAS BIC")
        if st.button("Cerrar sesión", key="close_s"):
            st.session_state.autenticado = False
            st.rerun()
        with st.expander("Cambiar Contraseña"):
            # Asegúrate de tener guardado el usuario en sesión al loguearse
            usuario_actual = st.session_state.username 
            
            old_pass = st.text_input("Contraseña actual", type="password", key="old_p")
            new_pass = st.text_input("Nueva contraseña", type="password", key="new_p")
            conf_pass = st.text_input("Confirmar nueva contraseña", type="password", key="conf_p")
            
            if st.button("Actualizar"):
                if new_pass != conf_pass:
                    st.error("Las contraseñas nuevas no coinciden.")
                elif len(new_pass) < 8: # Argon2 prefiere contraseñas más largas
                    st.warning("La contraseña debe tener al menos 8 caracteres.")
                else:
                    success, msg = cambiar_contrasena(usuario_actual, old_pass, new_pass)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
            

def modulo_interfaz_principal():
    if st.session_state.autenticado:
        client = MlflowClient()

        def load_file_remote(file_name, exp_id, tipo_accion):
            file_name_original = None
            uploaded_file = st.file_uploader("Seleccionar Archivo CSV", type=["csv"], key="uploader_csv")
            if uploaded_file is not None:
                file_name_original = uploaded_file.name
                st.session_state.ruta_manual = file_name_original 
                st.text_input("Archivo seleccionado:", value=st.session_state.ruta_manual, disabled=True)
                final_path = FOLDER_DATA.parent / "mlruns" / exp_id / f"{file_name}.csv"
                if tipo_accion == "entrenar":
                    try:
                        st.header("Módulo de Depuración y Etiquetado")
                        if 'procesamiento_exitoso' not in st.session_state:
                            st.session_state.procesamiento_exitoso = False
                        @st.cache_data
                        def cargar_datos(file):
                            # Carga optimizada
                            return pd.read_csv(file, sep=';', dtype=str, encoding='utf-8')
                        df = cargar_datos(uploaded_file)
                        print(f"Modulo depuracion df creado")
                        columnas = list(df.columns)
                        print(f" --- SECCIÓN 1: UI de Filtros Dinámicos ---")
                        st.subheader("1. Configurar Reglas de Filtrado (Target)")
                        col_check, col_name, col_op, col_val = st.columns([0.5, 2, 2, 2])
                        col_name.write("**Campo**")
                        col_op.write("**Validación**")
                        col_val.write("**Valor**")
                        columnas_seleccionadas = st.multiselect(
                            "Seleccione los campos a los que desea aplicar filtros/validaciones:",
                            options=columnas,
                            help="Seleccione una o varias columnas para desplegar sus opciones de validación."
                        )
                        st.divider()
                        modulo_depuracion_leads(df, uploaded_file, final_path, columnas, columnas_seleccionadas)
                        return file_name_original
                    except Exception as e:
                        st.error(f"Error en el servidor: {e}")
                        return None
                else:
                    if st.button(f"Confirmar y {tipo_accion}", key="btn_load"):
                        status_text = st.empty()
                        def mi_progreso(n_chunk, n_filas):
                            status_text.info(f"Procesando fragmento #{n_chunk} | Filas leídas: {n_filas:,}")
                        try:
                            total_filas = 0
                            n_chunk = 1
                            final_path.parent.mkdir(parents=True, exist_ok=True)
                            reader = pd.read_csv(uploaded_file, sep=";", chunksize=50000)
                            for chunk in reader:
                                modo = 'w' if n_chunk == 1 else 'a'
                                header = True if n_chunk == 1 else False
                                chunk.to_csv(final_path, mode=modo, index=False, sep=";", header=header)
                                total_filas += len(chunk)
                                mi_progreso(n_chunk, total_filas)
                                n_chunk += 1
                            st.success(f"¡Archivo guardado en servidor: {total_filas} filas!")
                            return file_name_original
                        except Exception as e:
                            st.error(f"Error en el servidor: {e}")
                            return None
            return None

        def train_ui(exp_id, exp_name, mod_name, file_name, file_name_or):
            print("Inicio")
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                print("# 3. Definir la función callback que se enviará al módulo externo")
                def actualizar_ui(valor, texto):
                    progress_bar.progress(valor)
                    status_text.text(texto)
                    
                print("# 4. Llamar a la función del otro módulo pasando el callback")
                with st.spinner("Ejecutando pipeline..."):
                    file_path = FOLDER_DATA.parent / "mlruns" / exp_id / f"{file_name}.csv"
                    run_id, new_auc = train_big_data(file_path, exp_name, mod_name, "contacto_positivo", progress_callback=actualizar_ui, file_name_or=file_name_or)
                    
                st.success(f"Modelo entrenado. Run ID: {run_id} - AUC: {new_auc:.4f}")
                promoted, old_auc = promote_if_better(exp_name, mod_name, run_id, new_auc)
                st.success(f"Entrenamiento finalizado. Promovido: {promoted} - AUC anterior/nuevo: {old_auc} / {new_auc}")

            except Exception as e:
                st.warning(f"Error entrenando modelo {e}")

        def score_ui(run_id, path_in, path_out):
            print("Inicio score_ui")
            fig = None
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
                            fig = batch_predict_to_disk(
                                run_id=run_id,
                                input_csv_path=path_in,
                                output_csv_path=path_out
                            )
                            st.pyplot(fig)
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

        def modulo_depuracion_leads(df, archivo_cargado, path_out, columnas, columnas_seleccionadas):
            filtros_seleccionados = []
            with st.form(key="config_depuracion"):
                for columna in columnas_seleccionadas:
                    c1, c2, c3, c4 = st.columns([0.5, 2, 2, 2])
                    with c1: 
                        # Como el usuario ya lo seleccionó arriba, lo marcamos True por defecto
                        activar_filtro = st.checkbox(
                            f"Filtrar por {columna}", 
                            value=True, # <-- Mejora de usabilidad
                            key=f"chk_{columna}", 
                            label_visibility="collapsed"
                        )
                        
                    with c2: 
                        st.write(columna)
                    
                    # Detección de tipo para sugerir el operador correcto
                    es_num = pd.api.types.is_numeric_dtype(pd.to_numeric(df[columna], errors='ignore'))
                    
                    with c3:
                        op = st.selectbox(
                            "Op", 
                            ["mayor que", "menor que", "igual"] if es_num else ["es igual a", "contiene"], 
                            key=f"op_{columna}", 
                            label_visibility="collapsed"
                        )
                        
                    with c4:
                        val = st.text_input(
                            "Valor", 
                            key=f"val_{columna}", 
                            label_visibility="collapsed", 
                            placeholder="ej: VENTA|DESISTE"
                        )
                    
                    # 3. Solo agregamos a la lista final si el checkbox se mantiene activo
                    if activar_filtro:
                        filtros_seleccionados.append({"campo": columna, "operador": op, "valor": val})            
            
                st.subheader("2. Selección de Columnas para el Archivo Final")
                campos_a_mantener = st.multiselect(
                    "Seleccione los campos que desea mantener en el CSV de salida:",
                    options=columnas,
                    help="Puede eliminar los campos que no desea usar para el modelo de ML."
                )

                target_name = st.text_input("Nombre de columna Target", value="contacto_positivo")
                #print(f" --- SECCIÓN 3: Procesamiento y Descarga ---")
                boton_procesar = st.form_submit_button(label="Procesar archivo", type="primary")
            
            if boton_procesar:
                if not campos_a_mantener:
                    st.warning("Debe seleccionar al menos un campo para exportar.")
                    st.stop()
                st.success(f"Procesando {len(filtros_seleccionados)} reglas de negocio...")
                try:
                    procesar_csv_multiple_filtros(archivo_cargado, path_out, target_name, campos_a_mantener, filtros_seleccionados)
                    st.session_state.procesamiento_exitoso = True
                except Exception as e:
                    st.error(f"Se produjo un error al procesar o descargar el archivo: {e}")
                    return e
            if not st.session_state.procesamiento_exitoso:
                st.info("Esperando procesamiento para continuar...")
                st.stop()
                
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
                loaded = False
                file_name_or = None
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
                    print("# Lista desplegable (se actualiza al cambiar el experimento)")
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
                                print(st.session_state.form_lvl)
                                st.rerun()
                        with col_left_r:
                            if col_left_r.button("Score", key="btn_score"):
                                st.session_state.nombre_archivo = nombre_m.replace("_model", "_score_in")
                                print(st.session_state.nombre_archivo)
                                st.session_state.nombre_archivo_out = nombre_m.replace("_model", "_score_out")
                                print(st.session_state.nombre_archivo_out)
                                path_in = FOLDER_DATA.parent / "mlruns" / target_exp_id / f"{st.session_state.nombre_archivo}.csv"
                                path_out = FOLDER_DATA.parent / "mlruns" / target_exp_id / f"{st.session_state.nombre_archivo_out}.csv"
                                st.session_state.path_in = path_in
                                print(st.session_state.path_in)
                                st.session_state.path_out = path_out
                                print(st.session_state.path_out)
                                st.session_state.form_lvl = 4 # Pasamos al siguiente nivel
                                print(st.session_state.form_lvl)
                                st.rerun()

                with col_right:
                    if col_b2.button("Nuevo", key="btn_n_mod"):
                        st.session_state.form_lvl = 1
                        print(st.session_state.form_lvl)
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
                                    print(f"st.session_state.nombre_m={st.session_state.nombre_m}")
                                    st.session_state.nombre_archivo = f"{nombre_m}_file"
                                    print(f"st.session_state.nombre_archivo={st.session_state.nombre_archivo}")
                                    st.session_state.form_lvl = 2 # Pasamos al siguiente nivel
                                    print(st.session_state.form_lvl)
                                    st.rerun() # Ahora sí, recargamos para mostrar el nivel 2
                                else:
                                    st.error("Por favor, ingresa un nombre.")
                                    st.rerun()

                        # NIVEL 2: Carga de Archivo (Llamada a la función)
                elif st.session_state.form_lvl == 2:
                    try:
                        print("Pre load_file")
                        print(f"st.session_state.nombre_archivo={st.session_state.nombre_archivo}")
                        file_name_or = load_file_remote(st.session_state.nombre_archivo, target_exp_id, "entrenar")
                        print(f"file_name_or={file_name_or}")
                        st.session_state.nombre_archivo_original = file_name_or
                        print(f"st.session_state.nombre_archivo_original={st.session_state.nombre_archivo_original}")
                    except Exception as e:
                        st.warning(e)
                    if file_name_or:
                        st.session_state.form_lvl = 3
                        print(f"st.session_state.form_lvl={st.session_state.form_lvl}")
                        st.rerun()
                elif st.session_state.form_lvl == 3:    
                    train_ui(target_exp_id, seleccion_exp_name, st.session_state.nombre_m, st.session_state.nombre_archivo, st.session_state.nombre_archivo_original)
                    st.session_state.form_lvl = 0
                    print(st.session_state.form_lvl)
                    #st.rerun()
                                
                elif st.session_state.form_lvl == 4:
                    try:
                        file_name_or = load_file_remote(st.session_state.nombre_archivo, target_exp_id, "ejecutar scoring")
                    except Exception as e:
                            st.warning(e)
                    if file_name_or:
                        st.session_state.form_lvl = 5
                        print(st.session_state.form_lvl)
                        st.rerun()
                elif st.session_state.form_lvl == 5:
                    score_ui(st.session_state.run_id_modelo, st.session_state.path_in, st.session_state.path_out)
                    st.session_state.form_lvl = 0
                    print(st.session_state.form_lvl)


def modulo_depuracion_leads():
    """Módulo de depuración y filtrado dinámico."""
    st.header("Módulo de Depuración y Etiquetado")
    st.markdown("Configuración de criterios lógicos para definir el Target de la campaña.")

    archivo_cargado = st.file_uploader("Cargar registros de campaña (CSV ;)", type=['csv'], key="depurador")

    if archivo_cargado is not None:
        # Usamos caché para no recargar el archivo en cada interacción de los filtros
        @st.cache_data
        def cargar_datos(file):
            # Carga optimizada
            return pd.read_csv(file, sep=';', dtype=str, encoding='utf-8')
        
        df = cargar_datos(archivo_cargado)
        columnas = list(df.columns)
        
        # --- SECCIÓN 1: UI de Filtros Dinámicos ---
        st.subheader("1. Configurar Reglas de Filtrado (Target)")
        filtros_seleccionados = []

        col_check, col_name, col_op, col_val = st.columns([0.5, 2, 2, 2])
        col_name.write("**Campo**")
        col_op.write("**Validación**")
        col_val.write("**Valor**")
        st.divider()

        for columna in columnas:
            c1, c2, c3, c4 = st.columns([0.5, 2, 2, 2])
            with c1: 
                activar_filtro = st.checkbox(
                    f"Filtrar por {columna}", 
                    key=f"chk_{columna}", 
                    label_visibility="collapsed"
                )
            with c2: 
                st.write(columna)
            
            if activar_filtro:
                # Detección de tipo para sugerir el operador correcto
                es_num = pd.api.types.is_numeric_dtype(pd.to_numeric(df[columna], errors='ignore'))
                
                with c3:
                    op = st.selectbox("Op", ["mayor que", "menor que", "igual"] if es_num else ["es igual a", "contiene"], 
                                      key=f"op_{columna}", label_visibility="collapsed")
                with c4:
                    val = st.text_input("Valor", key=f"val_{columna}", label_visibility="collapsed", placeholder="ej: VENTA|DESISTE")
                
                filtros_seleccionados.append({"campo": columna, "operador": op, "valor": val})

        st.divider()
        
        # --- SECCIÓN 2: Selección de columnas a exportar ---
        st.subheader("2. Selección de Columnas para el Archivo Final")
        campos_a_mantener = st.multiselect(
            "Seleccione los campos que desea mantener en el CSV de salida:",
            options=columnas,
            default=columnas, # Por defecto pre-selecciona todos
            help="Puede eliminar los campos que no desea usar para el modelo de ML (ej. NIVEL3)"
        )

        target_name = st.text_input("Nombre de columna Target", value="TARGET_LEAD")

        # --- SECCIÓN 3: Procesamiento y Descarga ---
        if st.button("Procesar Archivo", type="primary"):
            if not campos_a_mantener:
                st.warning("Debe seleccionar al menos un campo para exportar.")
            else:
                st.success(f"Procesando {len(filtros_seleccionados)} reglas de negocio...")
                
                # Crear el directorio temporal si no existe
                tmp_dir = FOLDER_DATA.parent / "src" / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                
                path_out = tmp_dir / f"procesado_{archivo_cargado.name}"
                
                try:
                    # Invocación de la lógica de pandas (usando el script preprocessing.py)
                    procesar_csv_multiple_filtros(archivo_cargado, path_out, target_name, campos_a_mantener, filtros_seleccionados)
                    
                    if path_out.exists():
                        with open(path_out, "rb") as file:
                            st.download_button(
                                label="📥 Descargar Resultado Final",
                                data=file,
                                file_name=path_out.name,
                                mime="text/csv",
                                key="btn_dl"
                            )
                except Exception as e:
                    st.error(f"Se produjo un error al procesar o descargar el archivo: {e}")

def modulo_gráfica():
    def validar_campos_existentes(df, campos):
        """
        Función que verifica si una lista de campos existe en un DataFrame.
        """
        # Se usa el método issubset() combinado con set() para una validación vectorial rápida
        if set(campos).issubset(df.columns):
            return True
        else:
            return False
    fig = None
    campos = ["prediccion", "contacto_positivo"]
    campos1 = ["prediccion"]
    archivo_cargado = st.file_uploader("Cargar registros de campaña (CSV ;)", type=['csv'], key="depurador")
    if archivo_cargado:
        df = pd.read_csv(archivo_cargado, sep=';', encoding='utf-8')
        if validar_campos_existentes(df, campos):
            fig = graficar_ganancia_comparativa(df, "prediccion", "contacto_positivo")    
        elif validar_campos_existentes(df, campos1):
            fig = graficar_ganancia(df, "prediccion")
        if fig:
            st.pyplot(fig)
      
# --- 3. BARRA LATERAL (SIDEBAR) PARA NAVEGACIÓN ---

with st.sidebar:
    st.image(logo)
    st.markdown("---")
    
    # Selección de Módulo
    opcion = st.radio(
        #"Seleccione el módulo:",
        "",
        ("Login","Modelos", "Gráfica", "Depuración"),
        index=0,
        #help="Cambie entre la operación normal y la configuración de filtros de depuración."
    )
    
    st.markdown("---")
    st.caption("Desarrollado para: Outsourcing SAS BIC")

# --- 4. LÓGICA DE RENDERIZADO ---

if opcion == "Modelos":
    modulo_interfaz_principal()
elif opcion == "Login":
    modulo_login()    
elif opcion == "Depuración":
    modulo_depuracion_leads()
elif opcion == "Gráfica":
    modulo_gráfica()    