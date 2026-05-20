from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        # "logistic": LogisticRegression(
        #     max_iter=1000,
        #     class_weight="balanced",
        #     n_jobs=-1,
        #     solver="lbfgs"
        # ),

        # "random_forest": RandomForestClassifier(
        #     n_estimators=300,
        #     max_depth=None,
        #     n_jobs=-1,
        #     class_weight="balanced",
        #     random_state=42
        # ),

        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            data_random_seed=42
        ),

        "xgboost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            n_jobs=-1,
            random_state=42
        )
    }