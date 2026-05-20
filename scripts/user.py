import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

ph = PasswordHasher()

# Conexión a la base de datos local
def get_db():
    conn = sqlite3.connect('usuarios_tfm.db')
    return conn

# Crear tabla de usuarios
def crear_tabla():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Registrar usuario
def registrar_usuario(username, password):
    hashed = ph.hash(password) # Genera el hash
    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO usuarios (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Usuario ya existe
    finally:
        conn.close()

# Validar credenciales
def autenticar_usuario(username, password):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT password FROM usuarios WHERE username = ?", (username,))
    row = cursor.fetchone()
    conn.close()
    try:
        return ph.verify(row[0], password) # Verifica
    except:
        return False
    
def cambiar_contrasena(username, old_password, new_password):
    conn = get_db()
    cursor = conn.cursor()
    
    # 1. Obtener el hash actual
    cursor.execute("SELECT password FROM usuarios WHERE username = ?", (username,))
    row = cursor.fetchone()
    
    if row:
        try:
            ph.verify(row[0], old_password)
            new_hashed = ph.hash(new_password)
            cursor.execute("UPDATE usuarios SET password = ? WHERE username = ?", (new_hashed, username))
            conn.commit()
            conn.close()
            return True, "Contraseña actualizada correctamente."
            
        except VerifyMismatchError:
            conn.close()
            return False, "Error: La contraseña actual es incorrecta."
    
    conn.close()
    return False, "Error: Usuario no encontrado."