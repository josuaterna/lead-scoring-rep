import numpy as np
import pandas as pd

def procesar_csv_anonim(ruta_entrada, ruta_salida, target_col, eliminar=True):
    # 1. Definir los campos que queremos conservar
    campos_mantener = [
        "PROFESION", "Codigo_ciudad", "Estado Civil", "Nombre Departamento", 
        "Genero", "edad", "IdProductoOfrecido", target_col
        #, "CANTIDAD_PRODUCTO"
    ]

    # 2. Cargar el archivo CSV
    df = pd.read_csv(ruta_entrada, sep=';', dtype=str, encoding='utf-8')
    if target_col in df.columns:
        df.drop(columns=[target_col])

    # Buscamos las frases en la columna 'NIVEL3'
    # .str.contains permite buscar múltiples términos usando el operador '|' (OR)
    patron = "VENTA CANTADA|VENTA NO CONFIRMADA|VENTA EFECTIVA|CLIENTE DESISTE"
    
    # Creamos la columna: 1 si coincide, 0 si no
    # na=False asegura que si hay celdas vacías, no de error y ponga 0
    df['contacto_positivo'] = np.where(
        df['NIVEL 3'].str.contains(patron, case=False, na=False), 
        "1", 
        "0"
    )

    # 4. Eliminar columnas que no están en la lista 'campos_mantener'
        # Esto reemplaza el Loop de VBA que borraba columnas una a una
    if eliminar:
        columnas_existentes = [col for col in campos_mantener if col in df.columns]
        df_final = df[columnas_existentes]

        # 5. Guardar como CSV separado por ';' y en UTF-8
        df_final.to_csv(ruta_salida, sep=';', index=False, encoding='utf-8')
        print(f"Archivo procesado con éxito. Guardado en: {ruta_salida}")
    else:
        df.to_csv(ruta_salida, sep=';', index=False, encoding='utf-8')
        print(f"Archivo procesado con éxito. Guardado en: {ruta_salida}")

# --- Uso del script ---
# Reemplaza con tus rutas de archivo
procesar_csv_anonim(f'C:\\Users\\gitol\\Downloads\\ENERO_FEBRERO_MARZO.csv', f'C:\\Users\\gitol\\Downloads\\ENERO_FEBRERO_MARZO_entrenamiento.csv', 'contacto_positivo', True)