# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Prepara el entorno y dependencias para generar el ejecutable v6.2
#              (Sanitizado sin configuraciones personales ni bases de datos).
# ==============================================================================

import os
import shutil

def crear_entorno_seguro():
    # Al moverse a 'herramientas_build', el directorio raíz del proyecto es el padre
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest_dir = os.path.join(src_dir, "EMG_Ejecutable_Build")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Carpeta creada: {dest_dir}")
    else:
        print(f"La carpeta ya existe: {dest_dir}")

    # Ignoramos bases de datos, entornos, cachés locales, logs, resultados pesados y herramientas
    ignorar_carpetas = {
        'base_de_datos_electrodos', 'base_de_datos_letras', 'analisis_comparativos',
        'analisis_de_sesiones', 'resultados', 'logs', 'venv', '.venv', '__pycache__', '.git',
        'EMG_Ejecutable_Build', 'herramientas_build', 'tests', 'build_linux',
        'build_windows', 'dist', 'build', '.pytest_cache', 'codigos_de_prueba',
        'cache_datos', 'carpeta_para_vm_windows'
    }

    ignorar_archivos_exactos = {
        'crear_entorno_ejecutable.py',
        'argv_log.txt',
        'multiplexer_error.log',
        'config_local.json',
        '_sujeto_explorer_cache.json',
        'autoforge_daq_experimental_pc2_referencia.py',
    }

    def ignorar_rutas(directorio, archivos):
        ignorar = set()
        for a in archivos:
            if a in (
                '__pycache__', 'resultados_pca_umap', 'resultados_umap_supervisado',
                'resultados_autoencoder', '.pytest_cache', '.git', 'codigos_de_prueba',
                'autoforge_daq_experimental_pc2_referencia.py', 'config_local.json',
                '_sujeto_explorer_cache.json', 'cache_datos'
            ):
                ignorar.add(a)
            elif a.endswith(('.pyc', '.tar', '.zip', '.tmp', '.log')):
                ignorar.add(a)
        return ignorar

    for item in os.listdir(src_dir):
        src_path = os.path.join(src_dir, item)
        dest_path = os.path.join(dest_dir, item)

        if os.path.isdir(src_path):
            if item not in ignorar_carpetas:
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(src_path, dest_path, ignore=ignorar_rutas)
                print(f"[OK] Copiada carpeta: {item}")
        else:
            if item in ignorar_archivos_exactos or item.endswith(('.pyc', '.tmp', '.log', '.docx', '.aux', '.out', '.toc')):
                continue
            if item.startswith(("temp_", "test_")):
                continue
            shutil.copy2(src_path, dest_path)
            print(f"[OK] Copiado archivo: {item}")

    print("\n[OK] Copia de seguridad y entorno de compilación v6.2 listos!")
    print("El directorio 'EMG_Ejecutable_Build' está limpio de datos personales y listo para PyInstaller.")

if __name__ == "__main__":
    crear_entorno_seguro()
