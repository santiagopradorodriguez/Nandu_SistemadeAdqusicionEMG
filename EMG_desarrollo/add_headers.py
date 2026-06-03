import os
import glob
import re

header_template = """# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: {desc}
# ==============================================================================
"""

descriptions = {
    'autoforge_daq.py': 'Adquisición de datos automatizada y comunicación con hardware EMG.',
    'manual_daq.py': 'Módulo de adquisición manual para pruebas de hardware.',
    'metronomo_visual.py': 'Metrónomo visual y sonoro para guiar las pruebas de adquisición.',
    'ventana_palabras.py': 'Interfaz de ventana para mostrar palabras clave durante la adquisición.',
    'analisis_estadistico_pulsos.py': 'Realiza análisis estadístico sobre pulsos extraídos y genera métricas.',
    'analisis_por_track_integrado.py': 'Procesamiento, filtrado y análisis de señales por track integrado.',
    'correlaciondeseñales.py': 'Cálculo y visualización de correlación cruzada entre diferentes señales.',
    'dl_data_pipeline.py': 'Pipeline Batch Processing de señales sEMG para Deep Learning.',
    'electrode_viewer_4.py': 'Visor interactivo para datos por electrodo/medición.',
    'feature_extractor.py': 'Extracción de características y pulsos desde mediciones procesadas.',
    'plotter_calibrado.py': 'Visualización gráfica de señales EMG calibradas (multi-archivo).',
    'Sistema_de_Adquisicion_Emg.py': 'Launcher antiguo del sistema de adquisición EMG.',
    'visor_csv_interactivo.py': 'Visor interactivo antiguo para explorar datos en formato CSV.',
    'analisis_por_track_integrado_experimental.py': 'Versión experimental del análisis por track integrado.',
    'main_app.py': 'Punto de entrada principal para la aplicación gráfica (GUI).',
    'temp_comparativo.py': 'Script temporal/de soporte para la vista comparativa.',
    'temp_procesar.py': 'Script temporal/de soporte para procesamiento de datos.',
    'threads.py': 'Manejo de hilos para procesos en segundo plano en la GUI.',
    'comparative_explorer_widget.py': 'Widget de interfaz para exploración comparativa de sesiones.',
    'config_dialog.py': 'Diálogo de configuración de hardware y parámetros del sistema.',
    'csv_viewer_widget.py': 'Widget de interfaz para visualizar datos crudos desde archivos CSV.',
    'electrode_viewer_widget.py': 'Widget de interfaz para visualizar señales por electrodo.',
    'session_explorer.py': 'Widget de interfaz para navegar y explorar sesiones de medición.',
    'ui_analysis.py': 'Definiciones de interfaz de usuario para módulos de análisis.',
    'aplicar_parches_ejecutable.py': 'Inyecta parches en scripts durante la compilación a ejecutable.',
    'crear_entorno_ejecutable.py': 'Prepara el entorno y dependencias para generar el ejecutable.',
    'crear_spec_ejecutable.py': 'Genera el archivo spec de PyInstaller para la construcción.',
    'actualizar_metadata.py': 'Utilidad para actualizar o corregir metadatos de las sesiones grabadas.',
    'config_manager.py': 'Gestor de carga y guardado de configuraciones globales del sistema.',
    'editor_mediciones.py': 'Herramienta de utilidad para editar detalles de mediciones existentes.',
    'migrar_mediciones_por_fecha.py': 'Script de migración para organizar mediciones en carpetas por fecha.',
    'instrucciones_uso.py': 'Módulo de interfaz gráfica que muestra las instrucciones de uso del sistema.'
}

def get_description(file_path):
    name = os.path.basename(file_path)
    return descriptions.get(name, f'Módulo {name} del sistema NANDU LSD.')

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Try to find an existing block that matches our header signature
    # Signature: Starts with # ======, has "Proyecto:" or "NANDU LSD", ends with # ======
    header_pattern = re.compile(r'^# =+(?:\n#.*)*?(?:Proyecto:|NANDU LSD|Autores:)(?:\n#.*)*?\n# =+\n', re.MULTILINE)
    
    # If found, remove it
    content = header_pattern.sub('', content, count=1)
    
    # We also want to put the header AFTER #!/usr/bin/env... and # -*- coding...
    shebang = ""
    coding = ""
    
    lines = content.splitlines(keepends=True)
    start_idx = 0
    if len(lines) > 0 and lines[0].startswith('#!'):
        shebang = lines[0]
        start_idx = 1
    if len(lines) > start_idx and lines[start_idx].startswith('# -*- coding'):
        coding = lines[start_idx]
        start_idx += 1
        
    rest_of_file = "".join(lines[start_idx:])
    
    # Some files have #%% at the beginning which we should keep or push down. Let's just push them down.
    # Also strip leading newlines from rest_of_file so we don't accumulate empty lines
    rest_of_file = rest_of_file.lstrip('\n')
    
    desc = get_description(file_path)
    new_header = header_template.format(desc=desc)
    
    final_content = shebang + coding + new_header + "\n" + rest_of_file
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

files = glob.glob('**/*.py', recursive=True)
updated = []
for f in files:
    # Skip build dirs, virtual envs, and this script itself
    if 'EMG_Ejecutable_Build' in f or 'env' in f or 'venv' in f or 'add_headers.py' in f:
        continue
    process_file(f)
    updated.append(f)
    
with open('updated_files.txt', 'w', encoding='utf-8') as out:
    out.write('\\n'.join(updated))
