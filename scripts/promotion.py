import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()

def _get_model_version_by_run_id(model_name, run_id):
    versions = client.search_model_versions(
        f"name='{model_name}'"
    )
    for v in versions:
        if v.run_id == run_id:
            return v.version
    raise ValueError(
        f"No model version found for run_id={run_id}"
    )


def promote_if_better(
    model_name,
    new_run_id,
    new_auc,
    experiment_name="lead_scoring_training"
):
    """
    Promueve un modelo Challenger a Production si supera al Champion.
    """

    # Obtener AUC del Champion desde runs históricos
    experiment = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.roc_auc_cv DESC"],
        max_results=100
    )

    champion_runs = (
        runs
        .dropna(subset=["metrics.roc_auc_cv"])
        .sort_values("metrics.roc_auc_cv", ascending=False)
    )
    

    old_auc = (
        float(champion_runs.iloc[0]["metrics.roc_auc_cv"])
        if not champion_runs.empty
        else None
    )

    # Decisión de promoción
    if old_auc is None or new_auc >= old_auc:
        new_version = _get_model_version_by_run_id(
            model_name, new_run_id
        )

        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=new_version
        )

        return True, old_auc

    return False, old_auc
