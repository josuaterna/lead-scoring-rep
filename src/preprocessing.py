import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def procesar_csv_multiple_filtros(ruta_entrada, ruta_salida, target_col, campos_mantener, filtros):
    df = pd.read_csv(ruta_entrada, sep=';', dtype=str, encoding='utf-8')
    
    # Máscara inicial: Todo verdadero
    mascara = pd.Series(True, index=df.index)

    for f in filtros:
        campo = f['campo']
        op = f['operador']
        valor = str(f['valor']).strip()

        if not valor: continue  # Saltar si no hay valor para comparar

        try:
            if op == "es igual a":
                mascara &= (df[campo].astype(str) == valor)
            elif op == "contiene":
                # Soporta 'VENTA|COBRO|DESISTE' gracias a Regex
                mascara &= (df[campo].astype(str).str.contains(valor, case=False, na=False))
            
            elif op in ["mayor que", "menor que", "igual"]:
                # Convertir columna y valor a numérico de forma segura
                col_num = pd.to_numeric(df[campo], errors='coerce')
                val_num = float(valor)
                
                if op == "mayor que": mascara &= (col_num > val_num)
                elif op == "menor que": mascara &= (col_num < val_num)
                elif op == "igual": mascara &= (col_num == val_num)
        except Exception as e:
            print(f"Error procesando campo {campo}: {e}")
            continue

    # Crear columna Target
    df[target_col] = np.where(mascara, "1", "0")

    # Asegurar que el target esté en la salida aunque no se pida explícitamente
    columnas_finales = [c for c in campos_mantener if c in df.columns]
    if target_col not in columnas_finales:
        columnas_finales.append(target_col)
    print(f"procesar_csv_multiple_filtro - ESCRIBIR CSV -- {ruta_salida}")
    df[columnas_finales].to_csv(ruta_salida, sep=';', index=False, encoding='utf-8')
    return True

def verificar_cobertura_categorias(csv_path, target_col, max_unique=150, forced_cat_cols=None):
    if forced_cat_cols is None:
        forced_cat_cols = []
    """
    Detecta columnas categóricas reales y numéricas que actúan como categorías.
    max_unique: número máximo absoluto de valores únicos para ser categoría.
    """
    df_sample = pd.read_csv(csv_path, nrows=100000, sep=';')
    
    # Detectar candidatas:
    # - Columnas tipo 'object' (texto)
    # - Columnas numéricas con baja cardinalidad (pocos valores únicos)
    candidatas = []
    for col in df_sample.columns:
        # 1. Ignorar la columna objetivo
        if col == target_col:
            continue
        if col in forced_cat_cols:
            candidatas.append(col)
            continue
        unique_count = df_sample[col].nunique()
        dtype = df_sample[col].dtype
        
        # 2. Definir condiciones de filtrado
        is_object = (dtype == "object")
        is_low_cardinality = (unique_count < 100) # Ajusta según tu lógica
        is_low_cardinality_num = (dtype in ["int64", "float64"]) and (unique_count < max_unique)

        # 3. Decidir si se agrega
        if is_low_cardinality and (is_object or is_low_cardinality_num):
            candidatas.append(col)

    # 2. Escaneo de categorías totales (Igual que antes pero con la lista filtrada)
    categorias_totales = {col: set() for col in candidatas}
    
    for chunk in pd.read_csv(csv_path, usecols=candidatas, chunksize=200000, sep=';'):
        for col in candidatas:
            # Forzamos conversión a string para que el OneHotEncoder sea consistente
            categorias_totales[col].update(chunk[col].dropna().astype(str).unique())
    dict_final = {col: sorted(list(values)) for col, values in categorias_totales.items() if len(values) > 0}

    return dict_final

def build_preprocessor_big_data(df_sample, dict_categorias, target_col):
    # 1. Extraemos nombres y valores del diccionario de forma SINCRONIZADA
    categorical_cols = list(dict_categorias.keys())
    lista_categorias = [dict_categorias[col] for col in categorical_cols]
    
    # 2. Definimos numéricas (solo las que son números y NO están en categorías ni es el target)
    numeric_cols = [
        c for c in df_sample.select_dtypes(include=["int64", "float64"]).columns 
        if c != target_col and c not in categorical_cols
    ]
    #print(f"Numeric cols: {numeric_cols}")

    # 3. Pipeline Categórico (Seguro y Eficiente)
    categorical_pipeline = Pipeline(steps=[
    # Cambiamos .astype(str) por .astype(object) para evitar el tipo <U54
    ("to_str", FunctionTransformer(lambda x: x.astype(str).astype(object))),
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        categories=lista_categorias, 
        handle_unknown="ignore", 
        sparse_output=False 
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
    for col in categorical_cols:
        df_sample[col] = df_sample[col].astype(str).astype(object)
    
    df_sample=df_sample.drop(columns=[target_col])
    preprocessor.fit(df_sample)
    
    return preprocessor
