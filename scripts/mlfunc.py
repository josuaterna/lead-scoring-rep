import shutil
import hashlib
import time
import shutil
import hashlib
import time
import mlflow
import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import lightgbm as lgb
from pathlib import Path
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import roc_auc_score
from src.preprocessing import build_preprocessor_big_data
from src.preprocessing import verificar_cobertura_categorias
from src.model_selection import get_models
from src.evaluation import graficar_ganancia
from src.evaluation import graficar_ganancia_comparativa

FOLDER_DATA = Path(__file__).parent

client = MlflowClient()

def grafica(df, campos1, campos2):
    def validar_campos_existentes(df, campos):
        """
        Función que verifica si una lista de campos existe en un DataFrame.
        """
        # Se usa el método issubset() combinado con set() para una validación vectorial rápida
        if set(campos).issubset(df.columns):
            return True
        else:
            return False
    fig = None
    if validar_campos_existentes(df, campos1):
        fig = graficar_ganancia_comparativa(df, "prediccion", "contacto_positivo")    
    elif validar_campos_existentes(df, campos1):
        fig = graficar_ganancia(df, "prediccion")
    return fig

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

def batch_predict_to_disk(run_id, input_csv_path, output_csv_path, chunksize=50000, pred=0, fact=2000):
    """
    Lee un CSV grande, predice por chunks y guarda el resultado en disco.
    """
    chunk_max = 0.00
    chunk_min = 0.00

    chunk_max = 0.00
    chunk_min = 0.00

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

    chunk = pd.read_csv(input_csv_path, sep=";")
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
    
    print(" Validación/Reordenamiento de columnas")
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
    chunk = pd.read_csv(input_csv_path, sep=";")
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
    
    print(" Validación/Reordenamiento de columnas")
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
    probs = sigmoid(raw_preds)

    print(" Añadir resultados al chunk actual")
    chunk['probabilidad'] = probs
    chunk_max = chunk['probabilidad'].max()
    chunk_min = chunk['probabilidad'].min()
    chunk_rango = chunk_max - chunk_min
    chunk['prediccion'] = ((chunk['probabilidad'] - chunk_min)/chunk_rango)
    fig = None
    campos = ["prediccion", "contacto_positivo"]
    campos1 = ["prediccion"]
    fig = grafica (chunk, campos, campos1)

    chunk.to_csv(output_csv_path, 
                    mode='a', 
                    index=False, 
                    sep=";", 
#                    header=first_chunk
                    )

    return fig

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

def train_big_data(csv_path, exp_name, model_name, target_col, progress_callback=None, file_name_or=None):
    print("# Calculamos el número total de filas aproximado para la barra")
    # (Hacer un count rápido inicial o estimar por tamaño de archivo)
    columnas_a_forzar = ["codigo_ciudad"]
    dict_categorias = verificar_cobertura_categorias(csv_path, target_col, forced_cat_cols=columnas_a_forzar)
    chunksize = 100000
    sample_df = pd.read_csv(csv_path, nrows=100000, sep=";")
    dataset_mlflow = mlflow.data.from_pandas(sample_df, name=file_name_or)
    preprocessor = build_preprocessor_big_data(sample_df, dict_categorias, target_col)
    mlflow.set_experiment(exp_name)
    best_auc = 0
    
    print("# Listas para acumular métricas de evaluación final")
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
                mlflow.log_metric("roc_auc", auc_score)
                mlflow.log_input(dataset_mlflow, context="training")
                #print(" También es vital guardar el preprocessor, ya que el Booster no lo incluye")
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