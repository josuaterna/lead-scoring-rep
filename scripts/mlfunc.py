import mlflow
import shutil
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

FOLDER_DATA = Path(__file__).parent

client = MlflowClient()

def create_exp(name):
    try:
        # Intenta crear el experimento
        exp_id = mlflow.create_experiment(name=name)
        return exp_id
    except MlflowException as e:
        # Si el nombre ya existe, MLflow lanza un error
        error_msg = str(e)
        if "already exists" in error_msg:
            return "existe"
        return f"Error de MLflow: {error_msg}"
    except Exception as e:
        # Captura cualquier otro error inesperado (permisos, carpetas, etc.)
        return f"Error inesperado: {str(e)}"

def promote_action(model_name, new_run_id):
    all_versions = client.search_model_versions(f"name='{model_name}'")
    new_v = [v for v in all_versions if v.run_id == new_run_id]
    
    if new_v:
        # Ordenamos por número de versión para asegurar que tomamos la correcta
        version_to_promote = max(new_v, key=lambda x: int(x.version))
        
        client.set_registered_model_alias(
            name=model_name,
            alias="production",
            version=version_to_promote.version
        )
        return f"--- ¡ÉXITO! Modelo {model_name} versión {version_to_promote.version} promovido a Production ---"
        
    else:
        return "Error: No se encontró una versión de modelo asociada al run_id enviado."

def promote_if_better(
    exp_name,
    model_name,
    new_run_id,
    new_auc,
):
    """
    Promueve un modelo Challenger a Production si supera al Champion.
    """
    promote = False
    current_best_auc = -1.0
    mlflow.set_experiment(exp_name)

    try:
        # 1. Buscar la versión que tenga el alias "production"
        try:
            prod_version = client.get_model_version_by_alias(model_name, "production")
        except MlflowException as e:
            promote = True
            promote_action(model_name, new_run_id)
            return promote, new_auc
        except Exception as e:
            promote = True
            promote_action(model_name, new_run_id)
            return promote, new_auc

        prod_run_id = prod_version.run_id
        
        # 2. Buscar child-runs de esa versión
        # Buscamos en el mismo experimento que el run de producción
        run_info = client.get_run(prod_run_id)
        experiment_id = run_info.info.experiment_id
        
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.`mlflow.parentRunId` = '{prod_run_id}'"
        )

        # 3. Seleccionar la child-run con la mejor métrica "roc_auc_cv"
        aucs = []
        for run in child_runs:
            val = run.data.metrics.get("roc_auc_cv")
            if val is not None:
                aucs.append(val)
        
        if aucs:
            current_best_auc = max(aucs)
            print(f"Mejor AUC actual en producción: {current_best_auc:.4f}")
        else:
            print("No se encontraron métricas 'roc_auc_cv' en los child-runs de producción.")

    except MlflowException:
        # Si no existe versión con el alias "production", se asume que el nuevo es el primero
        print(f"No se encontró una versión con alias 'production' para el modelo {model_name}.")
        promote = True

    # 4. Comparar el nuevo valor con el encontrado
    if not promote and new_auc > current_best_auc:
        print(f"Nuevo AUC ({new_auc:.4f}) es mejor que el actual ({current_best_auc:.4f}).")
        promote = True
    elif not promote:
        print(f"Nuevo AUC ({new_auc:.4f}) NO supera al actual. No se promueve.")

    # 5. Si es mejor, buscar la versión correspondiente al new_run_id y asignar alias
    if promote:
        # Buscamos qué versión del modelo corresponde a ese new_run_id
        # (Un run_id puede tener varias versiones si se registró varias veces)
        print(promote_action(model_name, new_run_id))

    return promote, current_best_auc

def list_experiments():
    exps = client.search_experiments()
    return [e for e in exps if e.name != "Default"]

def list_models(experiment_id):
    try:
        modelos_reg = client.search_registered_models()
        modelos_prod = []
        for model in modelos_reg:
            try:
                version_prod = client.get_model_version_by_alias(model.name, "production")
            except Exception as e:
                continue
            run_info = client.get_run(version_prod.run_id)
            if run_info.info.experiment_id == experiment_id:
                modelos_prod.append(model)
    except Exception as e:
        print(str(e))                
    return modelos_prod

# def list_models(experiment_id):
#     try:
#         modelos_prod = []
#         runs = mlflow.search_runs(experiment_ids=[experiment_id])
#         # 2. Iteras sobre los run_id para buscar si tienen modelos asociados en el registro
#         for run_id in runs['run_id']:
#             # Buscamos versiones de modelos que coincidan con este run_id
#             filter_string = f"run_id = '{run_id}'"
#             registered_models = client.search_model_versions(filter_string)
            
#             for model in registered_models:
#                 if model:
#                     if model.tags.get() == "production":
#                         modelos_prod.append(model)
#                         print(f"Run ID: {run_id} -> Modelo: {model.name}, Versión: {model.version}, Etapa: {model.current_stage}")
#     except Exception as e:
#         print(str(e))
    
#     return modelos_prod
    
def limpiar_basura_mlflow():
    experimentos = client.search_experiments(view_type=mlflow.entities.ViewType.DELETED_ONLY)
    
    if not experimentos:
        return print("No hay experimentos marcados como 'deleted' para limpiar.")

    try:
        # Nota: La API de MLflow no permite 'hard delete' directo de experimentos.
        # Si usas FileStore (local), puedes limpiar la carpeta .trash
        trash_path = FOLDER_DATA.parent / "mlruns" / ".trash"
        if trash_path.exists and trash_path.is_dir():
            shutil.rmtree(trash_path)
            trash_path.mkdir(parents=True, exist_ok=True)
            return f"Se han eliminado permanentemente {len(experimentos)} experimentos del storage local."
        else:
            return "No se encontró la ruta."
            
    except Exception as e:
        return f"Error al limpiar: {e}"

def load_data(exp_id):
    exp_route = FOLDER_DATA.parent / "mlruns" / exp_id
    if not exp_route.exists:
        try:
            train_file = st.file_uploader("Cargar achivo para entrenamiento", type=["csv"])
            if train_file:
                file_path = exp_route / f"{texto}.csv"  
                with open(file_path, "wb") as f:
                    f.write(train_file.getbuffer())
                    return f"Carga OK"
        except Exception as e:
            return e

def get_experiment(exp_name):

    exp = mlflow.get_experiment_by_name(exp_name)

    if exp:
        nombre = exp.name
        estado = exp.lifecycle_stage  # Retorna 'active' o 'deleted'
        experiment_id = exp.experiment_id
        
        return estado, experiment_id
    else:
        return None

def predict_from_mlflow(run_id, data_to_predict):
    """
    run_id: El ID del experimento de MLflow donde se guardó el modelo.
    data_to_predict: DataFrame con los datos nuevos (crudos).
    """
    # 1. Cargar el Preprocesador (Sklearn)
    preprocessor_uri = f"runs:/{run_id}/preprocessor"
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)
    
    # 2. Cargar el Modelo (Booster nativo)
    model_uri = f"runs:/{run_id}/model"
    
    # Intentamos cargar como LightGBM, si falla probamos XGBoost
    try:
        model = mlflow.lightgbm.load_model(model_uri)
        is_lgb = True
    except:
        model = mlflow.xgboost.load_model(model_uri)
        is_lgb = False

    # 3. Transformar los datos nuevos
    X_processed = preprocessor.transform(data_to_predict)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    # 4. Predicción
    if is_lgb:
        # LightGBM Booster usa predict directamente
        y_proba = model.predict(X_processed)
    else:
        # XGBoost Booster requiere DMatrix
        import xgboost as xgb
        dtest = xgb.DMatrix(X_processed)
        y_proba = model.predict(dtest)

    return y_proba