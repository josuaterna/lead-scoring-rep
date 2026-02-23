import mlflow
import mlflow.sklearn
from pipelines.predict_pipeline import predict

if __name__ == "__main__":
    model = mlflow.sklearn.load_model(
        model_uri="models:/lead_scoring_model@production"
    )

    predict(
        model=model,
        csv_path="data/raw/nuevos_leads.csv",
        output_path="data/predictions/leads_scored.csv"
    )