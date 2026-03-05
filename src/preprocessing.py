import pandas as pd
from src.data_loader import drop_fields
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def col_filter(df):
    DROP_COLS = drop_fields(r"D:\Backup2026\UNIR\TFM 2025\lead_scoring_app\data\experiments\lead_scoring_training\fields.txt")
    fields_to_drop = [i.strip().lower().replace(" ", "_") for i in DROP_COLS]
    cols_to_drop = [c for c in fields_to_drop if c in df.columns]
    df.drop(columns=cols_to_drop)
    return df


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

def verificar_cobertura_categorias(csv_path, target_col, threshold=0.05, max_unique=100):
    """
    Detecta columnas categóricas reales y numéricas que actúan como categorías.
    threshold: % máximo de valores únicos respecto al total para considerarlo categoría.
    max_unique: número máximo absoluto de valores únicos para ser categoría.
    """
    # 1. Muestra para análisis inicial
    df_sample = pd.read_csv(csv_path, nrows=50000, sep=';')
    
    # Detectar candidatas:
    # - Columnas tipo 'object' (texto)
    # - Columnas numéricas con baja cardinalidad (pocos valores únicos)
    candidatas = []
    for col in df_sample.columns:
        if col == target_col: continue
        unique_count = df_sample[col].nunique()
        if unique_count < 100: 
            is_object = df_sample[col].dtype == "object"
            is_low_cardinality_num = (df_sample[col].dtype in ["int64", "float64"]) and (unique_count < max_unique)
            candidatas.append(col)
        else:
            print(f"Columna {col} ignorada: demasiada cardinalidad ({unique_count})")
                
        if is_object or is_low_cardinality_num:
            candidatas.append(col)

    # 2. Escaneo de categorías totales (Igual que antes pero con la lista filtrada)
    categorias_totales = {col: set() for col in candidatas}
    
    for chunk in pd.read_csv(csv_path, usecols=candidatas, chunksize=200000, sep=';'):
        for col in candidatas:
            # Forzamos conversión a string para que el OneHotEncoder sea consistente
            categorias_totales[col].update(chunk[col].dropna().astype(str).unique())
            
    return {col: sorted(list(values)) for col, values in categorias_totales.items()}

def build_preprocessor_big_data(df_sample, dict_categorias, target_col):
    # 1. Extraemos nombres y valores del diccionario de forma SINCRONIZADA
    categorical_cols = list(dict_categorias.keys())
    lista_categorias = [dict_categorias[col] for col in categorical_cols]
    
    # 2. Definimos numéricas (solo las que son números y NO están en categorías ni es el target)
    numeric_cols = [
        c for c in df_sample.select_dtypes(include=["int64", "float64"]).columns 
        if c != target_col and c not in categorical_cols
    ]

    # 3. Pipeline Categórico (Seguro y Eficiente)
    categorical_pipeline = Pipeline(steps=[
        # Convertimos a string para evitar errores de tipos mixtos
        ("to_str", FunctionTransformer(lambda x: x.astype(str))),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            categories=lista_categorias, 
            handle_unknown="ignore", 
            sparse_output=True # <--- OBLIGATORIO para 1GB de datos
        ))
    ])

    # 4. Pipeline Numérico
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # 5. Unión en ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numeric_pipeline, numeric_cols)
        ],
        remainder='drop' # Ignora las columnas de alta cardinalidad descartadas
    )
    
    # Ajustamos con la muestra
    preprocessor.fit(df_sample)
    
    return preprocessor
