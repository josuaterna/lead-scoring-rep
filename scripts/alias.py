from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_registered_model_alias(
    name="lead_scoring_model",
    alias="production",
    version="1"
)