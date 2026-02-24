import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.data_loader import load_csv
from src.preprocessing import build_preprocessor
from src.preprocessing import col_filter
from src.model_selection import get_models


def train(csv_path, target_col):

    df = load_csv(csv_path)
    df = col_filter(df)

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor = build_preprocessor(df, target_col)
    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall"
    }

    mlflow.set_experiment("lead_scoring_training")

    best_auc = 0
    best_model = None

    # 1. Iniciar el Run PADRE (el que registrará el modelo al final)
    #with mlflow.start_run(run_name="Pipeline_Comparacion") as parent_run:
    with mlflow.start_run() as parent_run:
    #with mlflow.start_run():
        for name, model in models.items():

            pipeline = Pipeline([
                ("preprocess", preprocessor),
                ("model", model)
            ])

            with mlflow.start_run(run_name=name, nested=True):

                cv_results = cross_validate(
                    pipeline,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1
                )

                roc_auc_mean = np.mean(cv_results["test_roc_auc"])
                recall_mean = np.mean(cv_results["test_recall"])

                mlflow.log_param("model_name", name)
                mlflow.log_param("cv_folds", 5)

                mlflow.log_metric("roc_auc_cv", roc_auc_mean)
                mlflow.log_metric("recall_cv", recall_mean)

                pipeline.fit(X, y)

                mlflow.sklearn.log_model(
                    pipeline,
                    artifact_path="model"
                )

                if roc_auc_mean > best_auc:
                    best_auc = roc_auc_mean
                    best_model = pipeline

                print(
                    f"{name} | ROC-AUC CV: {roc_auc_mean:.4f} | "
                    f"Recall CV: {recall_mean:.4f}"
                )
        registered_model_name = "lead_scoring_model"

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            registered_model_name=registered_model_name
        )
        
        # Opcional: Registrar en el Model Registry
        # run_id = parent_run.info.run_id
        # model_uri = f"runs:/{run_id}/model"
        # mlflow.register_model(model_uri, "nombre_de_modelo")
        run_id = parent_run.info.run_id

    #return best_model
    return run_id, best_auc
