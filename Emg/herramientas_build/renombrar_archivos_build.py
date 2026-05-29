import os

def renombrar_archivos_py():
    # Al moverse a 'herramientas_build', el directorio raíz del proyecto es el padre
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")

    if not os.path.exists(build_dir):
        print(f"La carpeta no existe: {build_dir}")
        return

    # Recorrer todos los archivos y carpetas dentro de EMG_Ejecutable_Build
    for root, dirs, files in os.walk(build_dir):
        # Evitamos renombrar archivos dentro del entorno virtual si se copió por error
        if 'venv' in root:
            continue
            
        for file in files:
            if file.endswith(".py") and not file.endswith("_build.py"):
                nombre_base = file[:-3]
                nuevo_nombre = f"{nombre_base}_build.py"
                
                ruta_vieja = os.path.join(root, file)
                ruta_nueva = os.path.join(root, nuevo_nombre)
                
                os.rename(ruta_vieja, ruta_nueva)
                print(f"Renombrado: {file} -> {nuevo_nombre}")

if __name__ == "__main__":
    renombrar_archivos_py()