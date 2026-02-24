import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

def promote_if_better(
    model_name,
    new_run_id,
    new_auc,
#    experiment_name="lead_scoring_training"
):
    """
    Promueve un modelo Challenger a Production si supera al Champion.
    """
    promote = False
    current_best_auc = -1.0

    try:
        # 1. Buscar la versión que tenga el alias "production"
        prod_version = client.get_model_version_by_alias(model_name, "production")
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

    except mlflow.exceptions.MlflowException:
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
            print(f"--- ¡ÉXITO! Modelo {model_name} versión {version_to_promote.version} promovido a Production ---")
            
        else:
            print("Error: No se encontró una versión de modelo asociada al run_id enviado.")
            
    return promote, current_best_auc


    
