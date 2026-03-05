import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
from src.data_loader import get_path
from scripts.mlfunc import create_exp
from scripts.mlfunc import promote_if_better
from scripts.mlfunc import list_experiments
from scripts.mlfunc import list_models
from scripts.mlfunc import limpiar_basura_mlflow
from scripts.mlfunc import get_experiment
from scripts.train import train_big_data
from scripts.scoring import score_model

FOLDER_DATA = Path(__file__).parent
logo_path = FOLDER_DATA / "img" / "logo.png"

def limpiar_button():
    with st.spinner("Limpiando base de datos de MLflow..."):
        mensaje = limpiar_basura_mlflow()
        st.write(mensaje)

st.set_page_config(page_title="Lead Scoring - Outsourcing", layout="wide")
st.divider()
with st.container():
    col_logo, col_text = st.columns([1, 4])
    with col_logo:
        # URL del logo oficial de Outsourcing S.A.S. BIC
        if logo_path.exists:
            st.image(logo_path, width=80)
        else:
            st.error(f"No se encontró el logo en: {logo_path}")
    with col_text:
        st.title("Lead Scoring")
        st.caption("Soluciones Inteligentes de Contact Center & BPO")

st.divider()
with st.container():
    st.subheader("Creación de experiment")
    with st.form("Crear experiement"):
        texto = st.text_input("Nombre:")
        enviado = st.form_submit_button("Enviar")
    if enviado:
        if texto:
            crea = create_exp(texto)
            if crea == "existe":
                estado, experiment_id = get_experiment(texto)
                st.warning(f"⚠️ Experimento '{texto}' existe. Estado: {estado}, ID: {experiment_id}.")
                if st.button("Limpiar Borrados", on_click=limpiar_button, help="Elimina permanentemente los experimentos en estado 'DELETED'"):
                    st.write("Eliminados")
            elif "Error" in crea:
                st.error(f"Error al crear experimento {texto}. {crea}")
            else:
                st.success(f"✅ ¡Experimento creado con éxito! ID: `{crea}`")
        else:
            st.error("Por favor, introduce un nombre válido.")

with st.container():
    # Los datos pueden ser una lista, tupla o Series de Pandas
    st.subheader("Predicción y entrenamiento")
    exps = list_experiments()
    exp_names = st.selectbox(
        "Selecciona un experiment:",
        options=exps,
        #format_func=lambda x: f"[{x.experiment_id}] {x.name}", # Lambda es una función corta
        format_func=lambda x: f"{x.name}", # Lambda es una función corta
        index=None  # Define cuál aparece seleccionada por defecto (0 es la primera)
    )
    
    if 'exp_names' in locals() and not exp_names == None:
        try:
            exp_id = exp_names.experiment_id
            mods = list_models(exp_id)
        except Exception as e:
            st.warning(str(e))

        mod_names = st.selectbox(
            "Selecciona un modelo:",
            options=mods,
            format_func=lambda x: f"{x.name}", # Lambda es una función corta
            index=0  # Define cuál aparece seleccionada por defecto (0 es la primera)
        )
        try:
            st.success(f"✅ Modelo en Producción: **{mod_names.name}**")
            if st.button('Cargar modelo'):
                @st.cache_resource
                def load_model():
                    model_uri=f"models:/{mod_names.name}@production"
                    return mlflow.sklearn.load_model(
                        #model_uri="models:/lead_scoring_model@production"
                        model_uri=model_uri
                    )

                model = load_model()

                uploaded_file = st.file_uploader(
                    "Carga archivo CSV de leads",
                    type=["csv"]
                )

                if uploaded_file:
                    df = pd.read_csv(
                    uploaded_file,
                    sep=";",
                    encoding="utf-8",
                    low_memory=False
                    )
                    df.columns = (
                        df.columns
                        .str.strip()
                        .str.lower()
                        .str.replace(" ", "_")
                    )

                    st.subheader("Vista previa")
                    st.dataframe(df.head())

                    if st.button("🔮 Calcular scoring"):
                        st.write("Shape del input:", df.shape)
                        st.write("Columnas:", df.columns.tolist())
                        scores = score_model(model, df)
                        df["score"] = scores

                        st.subheader("Resultados")
                        st.dataframe(df.sort_values("score", ascending=False))

                        st.download_button(
                            "⬇️ Descargar resultados",
                            df.to_csv(index=False),
                            "leads_scored.csv",
                            "text/csv"
                        )
        except Exception as e:
            print(str(e))
            st.warning(f"⚠️ No se encontró ningún modelo con el alias 'production' en {exp_names.name}.")
            exp_route = FOLDER_DATA.parent / "mlruns" / exp_names.experiment_id
            try:
                train_file = st.file_uploader("Cargar achivo para entrenamiento", type=["csv"])
                if train_file is not None:
                    file_path = exp_route / f"{exp_names.name}.csv"  
                    if st.button("Subir archivo"):
                        with st.spinner("Guardando..."):
                            # Aquí obtenemos la RUTA REAL en el servidor
                            ruta_final_servidor = get_path(train_file, file_path)
                        st.success(f"Archivo guardado: {ruta_final_servidor.name}")
                    with st.form("Crear modelo"):
                        texto_m = st.text_input("Nombre modelo:")
                        enviado_m = st.form_submit_button("Enviar")
                    if enviado_m:
                        if texto_m:
                            texto_m = f"{texto_m}_model"
                            try:
                                print(f"INICIO TRY - ANTES DE BAR")
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                # 3. Definir la función callback que se enviará al módulo externo
                                def actualizar_ui(valor, texto):
                                    progress_bar.progress(valor)
                                    status_text.text(texto)
                                    
                                # 4. Llamar a la función del otro módulo pasando el callback
                                with st.spinner("Ejecutando pipeline..."):
                                    run_id, new_auc = train_big_data(file_path, exp_names.name, texto_m, "contacto_positivo", progress_callback=actualizar_ui)
                                    
                                st.success(f"Modelo entrenado. Run ID: {run_id} - AUC: {new_auc:.4f}")
                                # run_id, new_auc = train(file_path, exp_names.name, texto_m, "contacto_positivo")
                                # st.success(f"Entrenamiento finalizado.")
                                promoted, old_auc = promote_if_better(exp_names.name, texto_m, run_id, new_auc)
                            except Exception as e:
                                st.warning(f"Error entrenando modelo {e}")
            except Exception as e:
                st.warning(str(e))
            
st.divider()
if not exp_names == None:
    st.subheader("Entrenamiento y promoción")
    if st.button("🔥 Reentrenar modelo"):
        with st.spinner("Entrenando nuevo modelo..."):
            from scripts.train import train

            run_id, new_auc = train("data/raw/leads.csv", "contacto_positivo")

            print(f"run_id={run_id}")
            print(f"run_id={new_auc}")

            promoted, old_auc = promote_if_better(
                "lead_scoring_model", run_id, new_auc
            )
        
        st.subheader("⚔️ Champion vs Challenger")

        col1, col2 = st.columns(2)
        col1.metric("ROC-AUC Champion", f"{old_auc:.3f}")
        col2.metric("ROC-AUC Challenger", f"{new_auc:.3f}")

        if promoted:
            st.success("Nuevo modelo promovido a producción")
        else:
            st.warning("El modelo challenger no superó al champion")
