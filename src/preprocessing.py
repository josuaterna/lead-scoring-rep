import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def col_filter(df):
    DROP_COLS = [
        "Acumulados_Intentos",
        "Año_mes_Nomenclatura",
        "Cargados",
        "Clave",
        "CONTACTO EFECTIVOS",
        "Contacto Titular",
        "EFECTIVOS CLIENTE",
        "Fecha",
        "Fecha Cargue",
        "Franquicia TC",
        "Gestion",
        "IdProductoOfrecido",
        "Llave Base",
        "Llave s",
        "LOADNAME",
        "LOGIN",
        "Mes",
        "NIVEL 1",
        "NIVEL 2",
        "NIVEL 3",
        "NO CONTACTADOS",
        "NombrePlanOfrecido",
        "Nombreproductoofrecido",
        "Numero mes",
        "oferta3",
        "oferta2",
        "Phone1",
        "Phone2",
        "Phone3",
        "Phone4",
        "Phone5",
        "TALKTIME",
        "Tipo Gestion",
        "Tipo Tarjeta",
        "Ventas_Efectivas"
    ]
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)
    

def build_preprocessor(df, target_col):
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.drop(target_col)
    categorical_cols = df.select_dtypes(include=["object"]).columns

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols)
        ]
    )

    return preprocessor