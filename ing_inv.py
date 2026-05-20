import mlflow.xgboost
import matplotlib.pyplot as plt
import xgboost as xgb
import pandas as pd

#VENTA CANTADA|VENTA NO CONFIRMADA|VENTA EFECTIVA|CLIENTE  DESISTE

# 1. Cargar el booster desde MLflow
path = f"C:\\Users\\gitol\\Downloads\\JUL_AGO_SEP_2025_limpio.csv"
run_id = "7eb469a37e4e482faf242514351302bd"
model_uri = f"runs:/{run_id}/model"
preprocessor = mlflow.sklearn.load_model(f"runs:/{run_id}/preprocessor")
#trained_booster = mlflow.xgboost.load_model(model_uri)
trained_booster = mlflow.lightgbm.load_model(model_uri)
target_col = "contacto_positivo"

# 2. Graficar importancia
# 'weight' es el default, pero 'gain' suele ser más informativo para detectar fugas
#nombres_features = preprocessor.get_feature_names_out()
#nombres_features = preprocessor[:-1].get_feature_names_out()
try:
     nombres_features = preprocessor.get_feature_names_out()
     print("Logrado con método directo")
except Exception as e:
     print(f"Error directo: {e}")
    
     # 2. Si falla, vamos a inspeccionar qué hay dentro del ColumnTransformer
     # Esto te mostrará los nombres de las columnas antes de f2065
     nombres_features = []
     for name, transformer, columns in preprocessor.transformers_:
         if name == 'remainder' and transformer == 'drop':
             continue
        
         try:
             # Intentamos obtener nombres de cada transformador individual
             names = transformer.get_feature_names_out()
             nombres_features.extend(names)
         except:
             # Si el transformador (como tu 'to_str') no tiene el método,
             # usamos los nombres de las columnas originales que recibió
             nombres_features.extend(columns)

 # Ahora ya puedes buscar a los culpables
# print(f"Culpable 1 (f2065): {nombres_features[2065]}")
# print(f"Culpable 2 (f2066): {nombres_features[2066]}")

# 1. Toma una muestra pequeña de tus datos originales (1 sola fila)
sample_df = pd.read_csv(path, sep=";", nrows=1) 
# (Asegúrate de quitar el target si está presente)
if target_col in sample_df.columns:
    sample_df = sample_df.drop(columns=[target_col])

# 2. Transforma esa fila con el preprocesador cargado de MLflow
X_transformed = preprocessor.transform(sample_df)

# 3. Si el resultado es una matriz dispersa (sparse), conviértela a densa
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

# 4. Obtener los nombres de columnas REALES
# Si get_feature_names_out falla, usaremos la inspección por pasos:
try:
    nombres_reales = preprocessor.get_feature_names_out()
except:
    # Si falla, miramos dentro de los transformadores para ver el orden
    print("Detectando nombres por inspección de transformadores...")
    nombres_reales = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder' and trans == 'drop': continue
        try:
            nombres_reales.extend(trans.get_feature_names_out())
        except:
            # Si es tu transformador 'to_str', el nombre suele ser el mismo de la columna original
            nombres_reales.extend(cols)

# # 5. Imprimir con validación de rango
print(f"Total de columnas detectadas: {len(nombres_reales)}")
if len(nombres_reales) > 2066:
    print(f"Culpable 1 (f2065): {nombres_reales[2065]}")
    print(f"Culpable 2 (f2066): {nombres_reales[2066]}")
else:
    print("La lista sigue siendo corta. Es probable que el One-Hot Encoder esté creando miles de columnas.")
# 1. Mira la forma de la matriz transformada
# Si X_transformed tiene más de 2000 columnas, aquí lo veremos
print(f"Forma de la matriz transformada: {X_transformed.shape}")
# 2. Vamos a extraer los nombres directos del ColumnTransformer de forma atómica
# Scikit-learn guarda los nombres generados en el atributo 'get_feature_names_out'
# pero hay que asegurarse de llamar al método del objeto principal
try:
    all_names = preprocessor.get_feature_names_out()
    print(f"Total nombres recuperados: {len(all_names)}")
    print(f"Culpable f2065: {all_names[2065]}")
    print(f"Culpable f2066: {all_names[2066]}")
except Exception as e:
    print(f"No se pudo extraer automáticamente: {e}")
nombres_finales = []
# Accedemos a los transformadores internos del ColumnTransformer
for name, transformer, columns in preprocessor.transformers_:
    if name == 'remainder' and transformer == 'drop':
        continue

    try:
        # Intentamos obtener nombres (esto funcionará para OneHotEncoder, Scalers, etc.)
        names = transformer.get_feature_names_out()
        nombres_finales.extend(names)
    except:
        # Si falla (como en to_str), usamos los nombres de las columnas originales
        # que ese transformador recibió.
        nombres_finales.extend(columns)
print(f"Lista reconstruida: {len(nombres_finales)} columnas")
if len(nombres_finales) >= 2066:
    print(f"🔍 CULPABLE f2065: {nombres_finales[2065]}")
    print(f"🔍 CULPABLE f2066: {nombres_finales[2066]}")

    # 3. SI FALLA: Mapeo por descarte
    # Mira tus columnas originales. ¿Cuál de ellas es categórica?
    # El OneHotEncoder suele poner los nombres así: "cat__NOMBRE_VALOR"
    # Si f2065 existe, es porque una de tus columnas categóricas tiene MILES de valores.
acumulado = 0
print(f"Buscando el origen de la columna 2065...")

for name, trans, cols in preprocessor.transformers_:
    if name == 'remainder' and trans == 'drop':
        continue
    
    # Creamos un dummy para ver cuántas columnas genera este transformador
    X_dummy = trans.transform(sample_df[cols])
    if hasattr(X_dummy, "toarray"):
        X_dummy = X_dummy.toarray()
    
    cantidad_columnas = X_dummy.shape[1]
    rango_final = acumulado + cantidad_columnas
    
    print(f"Transformador '{name}' procesa: {cols} -> Genera {cantidad_columnas} columnas (Rango: {acumulado} a {rango_final})")
    
    if acumulado <= 2065 < rango_final:
        print(f"--- 🎯 ¡ENCONTRADO! ---")
        print(f"El culpable es el transformador '{name}' que procesa las columnas: {cols}")
        
        # Si es un OneHotEncoder, intentamos ver el valor específico
        if hasattr(trans, 'get_feature_names_out'):
            nombres = trans.get_feature_names_out()
            indice_relativo = 2065 - acumulado
            print(f"La variable exacta f2065 es: {nombres[indice_relativo]}")
            
    acumulado = rango_final
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(trained_booster, importance_type='gain', ax=ax)
plt.title("Importancia de Variables (XGBoost)")
plt.show()

# En Streamlit, usa: st.pyplot(fig)