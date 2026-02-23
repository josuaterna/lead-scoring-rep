import numpy as np

def score_model(model, X):
    """
    Devuelve un score continuo en [0, 1] siempre.
    Funciona con y sin predict_proba.
    """

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # normalización min-max
        return (scores - scores.min()) / (scores.max() - scores.min())

    # fallback (último recurso)
    preds = model.predict(X)
    return preds.astype(float)