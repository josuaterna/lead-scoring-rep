import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import hashlib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score, recall_score
from src.preprocessing import col_filter
from src.preprocessing import build_preprocessor
from src.preprocessing import build_preprocessor_big_data
from src.preprocessing import verificar_cobertura_categorias
from src.model_selection import get_models

def pertenece_a_test(row, test_size_percent=15):
    """
    Determina si una fila pertenece al set de test basado en su contenido.
    """
    # Convertimos la fila a un string único
    row_str = "".join(row.astype(str))
    # Generamos un hash MD5
    hash_object = hashlib.md5(row_str.encode())
    # Convertimos el hash a un número entre 0 y 99
    hash_num = int(hash_object.hexdigest(), 16) % 100
    return hash_num < test_size_percent

def train(csv_path, exp_name, model_name, target_col):
    df = load_csv(csv_path)
    df = col_filter(df)
    
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

    mlflow.set_experiment(exp_name)

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
        registered_model_name = model_name

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

    return run_id, best_auc

def train_big_data(csv_path, exp_name, model_name, target_col, progress_callback=None):
    # Calculamos el número total de filas aproximado para la barra
    # (Hacer un count rápido inicial o estimar por tamaño de archivo)
    
    print(" 1. Configuración Inicial")
    dict_categorias = verificar_cobertura_categorias(csv_path, target_col)
    chunksize = 100000
    print(" Muestra pequeña para el preprocesador (usamos las primeras filas por velocidad)")
    sample_df = pd.read_csv(csv_path, nrows=10000)
    preprocessor = build_preprocessor_big_data(sample_df, dict_categorias, target_col)
    
    mlflow.set_experiment(exp_name)
    best_auc = 0
    
    print(" Listas para acumular métricas de evaluación final")
    test_chunks_x = []
    test_chunks_y = []

    with mlflow.start_run() as parent_run:
        models = get_models()
        
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                trained_booster = None
                
                # 2. PROCESAMIENTO POR CHUNKS
                for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
                    # Aplicamos el filtrado por Hash a cada fila del chunk
                    # Nota: Excluimos el target para que el hash dependa solo de las features
                    es_test = chunk.drop(columns=[target_col]).apply(pertenece_a_test, axis=1)
                    
                    train_chunk = chunk[~es_test]
                    val_chunk = chunk[es_test]
                    
                    # Acumulamos una parte del test para la evaluación final (solo en el primer modelo)
                    if len(test_chunks_x) < 5 and name == list(models.keys())[0]:
                        test_chunks_x.append(preprocessor.transform(val_chunk.drop(columns=[target_col])))
                        test_chunks_y.append(val_chunk[target_col])

                    # Entrenamiento incremental
                    X_train_trans = preprocessor.transform(train_chunk.drop(columns=[target_col]))
                    y_train = train_chunk[target_col]
                    
                    if name == "xgboost":
                        dtrain = xgb.DMatrix(X_train_trans, label=y_train)
                        trained_booster = xgb.train(
                            {k: v for k, v in model.get_params().items() if k not in ['n_estimators', 'missing']},
                            dtrain, num_boost_round=10, xgb_model=trained_booster
                        )
                    elif name == "lightgbm":
                        dtrain = lgb.Dataset(X_train_trans, label=y_train, free_raw_data=False)
                        trained_booster = lgb.train(
                            {k: v for k, v in model.get_params().items() if k not in ['n_estimators']},
                            dtrain, num_boost_round=10, init_model=trained_booster, keep_training_booster=True
                        )
                    if progress_callback:
                        # Estimamos progreso basado en el tamaño del archivo procesado
                        # Cada fila tiene un peso aprox, o simplemente por número de chunks
                        # Si conoces el total de filas, usa: (i * chunk_size) / total_filas
                        progreso_estimado = min((i + 1) * chunksize / 1000000, 0.99) # Ejemplo para 1M filas
                        progress_callback(progreso_estimado, f"Procesando bloque {i+1}...")
                if progress_callback:
                    progress_callback(1.0, f"Entrenamiento {name} completado con éxito.")                        
                # 3. EVALUACIÓN FINAL CON EL TEST SET ACUMULADO
                X_test = np.vstack(test_chunks_x)
                y_test = np.concatenate(test_chunks_y)
                
                model._Booster = trained_booster
                pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
                
                # El preprocessor ya se aplicó al acumular, así que usamos el modelo directo
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_proba)
                
                mlflow.log_metric("auc_hash_test", auc_score)
                mlflow.sklearn.log_model(pipeline, "model")
                
                if auc_score > best_auc:
                    best_auc = auc_score
                    best_pipeline = pipeline

        # Registro del mejor
        mlflow.sklearn.log_model(best_pipeline, "model", registered_model_name=model_name)

    return parent_run.info.run_id, best_auc
