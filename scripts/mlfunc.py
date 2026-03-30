import mlflow
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

FOLDER_DATA = Path(__file__).parent

client = MlflowClient()


def sigmoid(x):
    """Convierte scores brutos en probabilidades [0, 1]"""
    return 1 / (1 + np.exp(-x))

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
            if max(aucs) != 1:
                current_best_auc = max(aucs)
            else:
                print("Current best AUC fail == 1")
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

def get_experiment(exp_name):

    exp = mlflow.get_experiment_by_name(exp_name)

    if exp:
        nombre = exp.name
        estado = exp.lifecycle_stage  # Retorna 'active' o 'deleted'
        experiment_id = exp.experiment_id
        
        return estado, experiment_id
    else:
        return None

def batch_predict_to_disk(run_id, input_csv_path, output_csv_path, chunksize=50000):
    """
    Lee un CSV grande, predice por chunks y guarda el resultado en disco.
    """
    Path(output_csv_path).unlink(missing_ok=True)
    try:
        run = mlflow.get_run(run_id)
        tags = run.data.tags
        run_id_model = tags.get("id_run_mod", "No encontrado")
    except:
        return
    # 1. Cargar artefactos de MLflow
    print(f"runs:/{run_id_model}/preprocessor")
    preprocessor = mlflow.sklearn.load_model(f"runs:/{run_id_model}/preprocessor")
    model_uri = f"runs:/{run_id_model}/model"
 
    print(" Determinar si es LightGBM o XGBoost")
    try:
        model = mlflow.lightgbm.load_model(model_uri)
        is_lgb = True
    except:
        model = mlflow.xgboost.load_model(model_uri)
        is_lgb = False

    print(" 2. Procesar por Chunks")
    first_chunk = True
    # Asumimos separador ';' según tus ejemplos anteriores
    for chunk in pd.read_csv(input_csv_path, sep=";", chunksize=chunksize):
        # ... dentro del bucle de chunks ...
        print(f"DEBUG: Columnas en el CSV: {list(chunk.columns)}")
        print(f"DEBUG: Columnas esperadas: {list(preprocessor.feature_names_in_)}")

        columnas_esperadas = list(preprocessor.feature_names_in_)
        for col in columnas_esperadas:
            if col not in chunk.columns:
                chunk[col] = 0 # Creamos columnas faltantes (incluyendo el target)
        
        chunk_features = chunk[columnas_esperadas].copy()
        
        print(" Transformación")
        X_trans = preprocessor.transform(chunk_features)
        # 3. Filtrado con copia explícita
        print("DEBUG: Intentando filtrar columnas...")
        chunk_features = chunk[list(preprocessor.feature_names_in_)].copy()
        print("✅ Filtrado exitoso.")
        
        print(" Validación/Reordenamiento de columnas (usando la lógica que definimos antes")
        if hasattr(preprocessor, "feature_names_in_"):
            print(f"DEBUG: Filtrando {len(preprocessor.feature_names_in_)} columnas")
            chunk_features = chunk[preprocessor.feature_names_in_].copy()
        else:
            print("DEBUG: Usando chunk completo")
            chunk_features = chunk
            
        print("DEBUG: Iniciando Transformación...")
        try:
            X_trans = preprocessor.transform(chunk_features)
            print(f"DEBUG: Transformación exitosa. Shape: {X_trans.shape}")
        except Exception as e:
            print(f"❌ ERROR en Transformación: {str(e)}")
            raise e # Forzar el error para ver el traceback completo
        
        if hasattr(X_trans, "toarray"):
            print("DEBUG: Convirtiendo matriz dispersa a densa...")
            X_trans = X_trans.toarray()
            
        print(" Predicción")
        if is_lgb:
            raw_preds = model.predict(X_trans, raw_score=True)
        else:
            raw_preds = model.predict(xgb.DMatrix(X_trans), output_margin= True )

        probs = sigmoid(raw_preds)

        print(" Añadir resultados al chunk actual")
        chunk['probabilidad'] = probs
        chunk['prediccion'] = (probs > 0.5).astype(int)
        
        print(
            
        )
        # Si es el primer chunk, escribimos el header. Si no, lo omitimos.
        chunk.to_csv(output_csv_path, 
                     mode='a', 
                     index=False, 
                     sep=";", 
                     header=first_chunk)
        first_chunk = False
        
    return output_csv_path