import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import hashlib
import mlflow
import mlflow.sklearn
import time
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
    columnas_a_forzar = ["codigo_ciudad"]
    dict_categorias = verificar_cobertura_categorias(csv_path, target_col, forced_cat_cols=columnas_a_forzar)
    chunksize = 100000
    sample_df = pd.read_csv(csv_path, nrows=100000, sep=";")
    preprocessor = build_preprocessor_big_data(sample_df, dict_categorias, target_col)
    mlflow.set_experiment(exp_name)
    best_auc = 0
    
    # Listas para acumular métricas de evaluación final"
    test_chunks_x = []
    test_chunks_y = []

    with mlflow.start_run() as parent_run:
        models = get_models()
        id_tag = None
        id_parent = parent_run.info.run_id
        print(f"ID parent: {id_parent}")
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True) as child_run:
                trained_booster = None
                id_child = child_run.info.run_id
                print(f"ID child: {id_child}")
                if id_tag == None:
                    id_tag = id_child
                print(f"ID tag: {id_tag}")
                print(" 2. PROCESAMIENTO POR CHUNKS")
                total_positivos = 0
                total_negativos = 0
                for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize, sep=";")):
                    try:
                        conteo = chunk[target_col].value_counts()
                        total_positivos += conteo.get(1, 0)
                        total_negativos += conteo.get(0, 0)
                        pos_weight = total_negativos / max(total_positivos, 1)
                    except Exception as e:
                        pos_weight = 0.05
                        print(f"Error conteo: {e}")

                    # print(" Aplicamos el filtrado por Hash a cada fila del chunk")
                    # print(" Nota: Excluimos el target para que el hash dependa solo de las features")
                    es_test = chunk.drop(columns=[target_col]).apply(pertenece_a_test, axis=1)
                    
                    train_chunk = chunk[~es_test].copy() # Usamos .copy() para evitar SettingWithCopyWarning
                    # print(" LIMPIEZA CRÍTICA: Eliminar filas donde el target sea NaN o Infinito")
                    # print(" Esto evita que XGBoost rompa")
                    train_chunk = train_chunk[
                        train_chunk[target_col].notna() & 
                        np.isfinite(train_chunk[target_col])
                    ]
                    val_chunk = chunk[es_test]
                    
                    # print(" Acumulación segura para evaluación (solo en el primer modelo)")
                    # print(" Importante: Solo si el chunk de validación no está vacío")
                    if not val_chunk.empty and len(test_chunks_x) < 5 and name == list(models.keys())[0]:
                        val_x_trans = preprocessor.transform(val_chunk.drop(columns=[target_col]))
                        if hasattr(val_x_trans, "toarray"):
                            val_x_trans = val_x_trans.toarray()
                        test_chunks_x.append(val_x_trans)
                        test_chunks_y.append(val_chunk[target_col].values)

                    # print(" Entrenamiento incremental")
                    if not train_chunk.empty:
                        X_train_trans = preprocessor.transform(train_chunk.drop(columns=[target_col]))
                        # Aseguramos formato denso para evitar "setting an array element with a sequence"
                        if hasattr(X_train_trans, "toarray"):
                            X_train_trans = X_train_trans.toarray()
                        y_train = train_chunk[target_col].values.astype(np.float32)
                    if name == "xgboost":
                        print(" 1. Filtramos los parámetros que causan ruido o no aplican a xgb.train")
                        params = {
                            k: v for k, v in model.get_params().items() 
                            if k not in ['n_estimators', 'missing', 'enable_categorical']
                        }
                        
                        print(" 2. Opcional: Silenciar warnings internos de XGBoost")
                        params['verbosity'] = 0
                        params['tree_method']='hist'
                        params['max_depth'] = 8
                        params['scale_pos_weight'] = pos_weight

                        dtrain = xgb.DMatrix(X_train_trans, label=y_train)
                        trained_booster = xgb.train(
                            params, dtrain, num_boost_round=10, xgb_model=trained_booster
                        )
                    elif name == "lightgbm":
                        #print(" Extraemos parámetros y añadimos la optimización")
                        params = {k: v for k, v in model.get_params().items() if k not in ['n_estimators']}
                        #print(" 1. Aplicamos la recomendación para mejorar la ejecución")
                        params['force_row_wise'] = True  # O True/False según tus datos
                        #print(" 2. Para silenciar otras advertencias menos importantes:")
                        params['verbosity'] = -1
                        params['max_depth'] = 8
                        params['scale_pos_weight'] = pos_weight
                        #print(" En LGBM es vital ajustar num_leaves si subes max_depth (2^max_depth - 1)")
                        params['num_leaves'] = 255
                        dtrain = lgb.Dataset(X_train_trans, label=y_train, free_raw_data=False)
                        trained_booster = lgb.train(
                            params, dtrain, num_boost_round=10, init_model=trained_booster, keep_training_booster=True
                        )
                    if progress_callback:
                        #print(" Estimamos progreso basado en el tamaño del archivo procesado")
                        # Cada fila tiene un peso aprox, o simplemente por número de chunks
                        # Si conoces el total de filas, usa: (i * chunk_size) / total_filas
                        progreso_estimado = min((i + 1) * chunksize / 1000000, 0.99) # Ejemplo para 1M filas
                        progress_callback(progreso_estimado, f"Procesando bloque {i+1}...")
                if progress_callback:
                    progress_callback(1.0, f"Entrenamiento {name} completado con éxito.")                        
                print("3. EVALUACIÓN FINAL CON EL TEST SET ACUMULADO")
                X_test = np.concatenate(test_chunks_x, axis=0)
                y_test = np.concatenate(test_chunks_y, axis=0)

                print(" Predicción usando el sabor nativo del booster")
                if name == "xgboost":
                    dtest = xgb.DMatrix(X_test)
                    y_proba = trained_booster.predict(dtest)
                    print(" Logueo específico para XGBoost")
                    mlflow.xgboost.log_model(trained_booster, artifact_path="model")
                elif name == "lightgbm":
                    y_proba = trained_booster.predict(X_test)
                    print(" Logueo específico para LightGBM")
                    mlflow.lightgbm.log_model(trained_booster, artifact_path="model")

                auc_score = roc_auc_score(y_test, y_proba)
                if auc_score == 1:
                    auc_score = -1
                    print(f"Error: AUC == 1")
                mlflow.log_metric("auc_hash_test", auc_score)

                #print(" También es vital guardar el preprocessor, ya qu
                # 
                # 
                # e el Booster no lo incluye")
                mlflow.sklearn.log_model(preprocessor, "preprocessor")

                if auc_score > best_auc and auc_score != 1:
                    best_auc = auc_score
                    id_tag = id_child
                    print(f"ID tag2: {id_tag}")
                    print(" Guardamos el booster y el nombre para el registro final")
                    best_booster = trained_booster
                    best_model_type = name
            time.sleep(0.05)
        print(" Registro del mejor")
        if best_model_type == "xgboost":
            mlflow.xgboost.log_model(best_booster, "best_model", registered_model_name=model_name)
        else:
            mlflow.lightgbm.log_model(best_booster, "best_model", registered_model_name=model_name)
        
        mlflow.set_tag("id_run_mod", id_tag)
    if best_auc ==1:
        best_auc = -1
    return parent_run.info.run_id, best_auc
