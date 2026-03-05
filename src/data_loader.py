import shutil
import gc
import re
import pandas as pd
from pathlib import Path


FOLDER_DATA = Path(__file__).parent

def load_csv(path: str, nrows=None, usecols=None, chunksize=None) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        nrows=nrows,
        usecols=usecols,
        chunksize=chunksize
    )
    # df.columns = (
    #     df.columns
    #     .str.strip()
    #     .str.lower()
    #     .str.replace(" ", "_")
    #     .str.replace(r"[^\w]", "_", regex=True)
    # )
    return df   

def drop_fields(path):
    with open(path, 'r') as archivo:
        lista = archivo.readlines()
    return lista

def get_path(uploaded_file, final_path):
    # 1. Crear una carpeta temporal si no existe
    temp_dir = FOLDER_DATA / "tmp"
    if not temp_dir.exists():
        temp_dir.mkdir(parents=True, exist_ok=True)
    
    ruta_destino = temp_dir / uploaded_file.name
    
    # 2. Escritura por Chunks (Streaming)
    # Aquí está el truco: uploaded_file.read(bytes) lee solo un trozo
    with open(ruta_destino, "wb") as f:
        while True:
            chunk = uploaded_file.read(10 * 1024 * 1024)  # Lee bloques de 10MB
            if not chunk:
                break
            f.write(chunk)
    
    # 3. Importante: Resetear el puntero del archivo de Streamlit 
    # por si necesitas volver a leerlo, aunque aquí ya lo tenemos en disco.
    uploaded_file.seek(0)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    ruta_destino.replace(final_path)
    if limpiar_y_sobrescribir_csv(final_path):
        if final_path.exists():
            return final_path
        else:
            raise f"Error al subir archivo"
    else:
        raise f"Error al limpiar archivo"

def subir_archivo(ruta_origen_: Path, final_path: Path, progress_callback=None):
    ruta_origen = Path(ruta_origen_.strip().strip('"'))

    # if not ruta_origen.exists():
    #     raise FileNotFoundError(f"No se encontró el archivo en: {ruta_origen}")
    try:
        temp_dir = FOLDER_DATA / "tmp"
        if not temp_dir.exists():
            temp_dir.mkdir(parents=True, exist_ok=True)
        destino = temp_dir / ruta_origen.name
        chunksize = 10000
        # 2. Leer desde el disco SIN cargar todo en RAM
        # Al pasar la ruta como string, Pandas abre un puntero al archivo
        lector = pd.read_csv(ruta_origen, chunksize=chunksize)
        total_filas = 0
        for i, chunk in enumerate(lector):
            chunk.to_csv(destino, mode='a', index=False, header=(i == 0))
            total_filas += len(chunk)
            if progress_callback:
                progress_callback(i + 1, total_filas)
                del chunk
                gc.collect() # Fuerza al sistema a limpiar la RAM AHORA
        #destino.seek(0)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        destino.replace(final_path)
        if limpiar_y_sobrescribir_csv(final_path):
            if final_path.exists():
                return total_filas
            else:
                raise f"Error al subir archivo"
        else:
            raise f"Error al limpiar archivo"                
    except Exception as e:
        raise e

    
def limpiar_temporales(ruta_archivo):
    path = Path(ruta_archivo)
    
    # Borrar el archivo específico
    if path.exists():
        path.unlink() # Borra el archivo
        print(f"Archivo {path.name} eliminado.")
    
    # Opcional: Borrar toda la carpeta si está vacía
    folder = path.parent
    if folder.exists() and not any(folder.iterdir()):
        folder.rmdir()
        print("Carpeta temporal eliminada por estar vacía.")

def limpiar_y_sobrescribir_csv(ruta_original: str, separador=';'):
    # Convertimos el string a un objeto Path
    path_orig = Path(ruta_original)
    # Creamos una ruta para el archivo temporal en la misma carpeta
    path_temp = path_orig.with_suffix(".tmp")
    
    try:
        # 1. Leer solo la cabecera (fila 0) para preparar los nuevos nombres
        df_header = pd.read_csv(path_orig, sep=separador, nrows=0)
        
        def limpiar_texto(texto):
            texto = texto.strip().lower()
            texto = texto.replace(" ", "_")
            # Mantiene solo caracteres alfanuméricos y guiones bajos
            texto = re.sub(r"[^\w]", "", texto)
            return texto

        nuevos_encabezados = [limpiar_texto(col) for col in df_header.columns]
        
        # 2. Procesar el archivo por chunks y escribir en el temporal
        first_chunk = True
        # Usamos el path original para leer
        for chunk in pd.read_csv(path_orig, sep=separador, chunksize=100000, low_memory=False):
            chunk.columns = nuevos_encabezados
            
            # Escribir en el path temporal
            chunk.to_csv(
                path_temp, 
                mode='a', 
                index=False, 
                sep=separador, 
                header=first_chunk, 
                encoding='utf-8'
            )
            first_chunk = False
        
        # 3. INTERCAMBIO DE ARCHIVOS CON PATHLIB
        # unlink() elimina el archivo original
        path_orig.unlink()
        # rename() mueve el temporal a la ubicación del original
        path_temp.rename(path_orig)
        
        print(f"✅ Archivo normalizado con éxito: {path_orig.name}")
        return True

    except Exception as e:
        # Si algo falla, intentamos limpiar el temporal si se alcanzó a crear
        if path_temp.exists():
            path_temp.unlink()
        print(f"❌ Error durante la limpieza: {e}")
        return False        