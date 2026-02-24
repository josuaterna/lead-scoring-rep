import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from scripts.promotion import promote_if_better
from scripts.train import train
from scripts.scoring import score_model


st.set_page_config(page_title="Lead Scoring", layout="wide")

st.title("📊 Lead Scoring – Producción")

@st.cache_resource
def load_model():
    return mlflow.sklearn.load_model(
        model_uri="models:/lead_scoring_model@production"
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
        #proba = model.predict_proba(df)[:, 1]
        #df["score"] = proba
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
st.divider()
st.subheader("MLOps - Reentrenamiento y Gobierno del Modelo")        
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
