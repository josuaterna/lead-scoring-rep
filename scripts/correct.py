# import mlflow
# from scripts.promotion import promote_if_better

# #new_run_id = "<RUN_ID_DE_LA_V2>"
# new_run_id = "b7290c1e617f46509b02bf4a7280bb54"
# new_run_id2 = "dee6d2cb09ee4e1bb1bf721fd4fbcee7"
# run = mlflow.get_run(new_run_id)
# #print(run.data.metrics)
# #new_auc = 0.78  # el valor real que viste en MLflow
# new_auc = run.data.metrics["roc_auc_cv"]

# promoted, old_auc = promote_if_better(
#      "lead_scoring_model",
#      new_run_id2,
#      new_auc
#  )

# print(promoted, old_auc, new_auc)

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import RunStatus

client = MlflowClient()

#run_id = "PEGA_AQUI_EL_RUN_ID"
run_id = "896bf4761e17497e8263adb9f47d4d64"

client.set_terminated(
    run_id=run_id,
    status='FINISHED'
)