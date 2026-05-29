import os
import shutil

def crear_entorno_seguro():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dest_dir = os.path.join(src_dir, "EMG_Ejecutable_Build")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"📁 Carpeta creada: {dest_dir}")
    else:
        print(f"⚠️ La carpeta ya existe: {dest_dir}")

    # Ignoramos bases de datos pesadas y entornos virtuales
    ignorar_carpetas = {'base_de_datos_electrodos', 'base_de_datos_letras', 'analisis_comparativos', 'venv', '__pycache__', '.git', 'EMG_Ejecutable_Build'}

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_path):
            if item not in ignorar_carpetas:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path)
                print(f"📂 Copiada carpeta: {item}")
        else:
            # Copiamos todos los archivos (excepto este mismo script)
            if item != "crear_entorno_ejecutable.py":
                shutil.copy2(src_path, dest_path)
                print(f"📄 Copiado archivo: {item}")

    print("\n✅ ¡Copia de seguridad y entorno de trabajo listos!")
    print("A partir de ahora, haremos todos los cambios de código dentro de la nueva carpeta 'EMG_Ejecutable_Build'.")

if __name__ == "__main__":
    crear_entorno_seguro()