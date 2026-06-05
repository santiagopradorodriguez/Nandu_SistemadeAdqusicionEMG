# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Inyecta parches en scripts durante la compilación a ejecutable.
# ==============================================================================

import os
import re

def aplicar_parches():
    # Al moverse a 'herramientas_build', el directorio raíz del proyecto es el padre
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")

    if not os.path.exists(build_dir):
        print("❌ Carpeta 'EMG_Ejecutable_Build' no encontrada. Ejecuta primero 'crear_entorno_ejecutable.py'.")
        return

    # Este es el bloque de código que inyectaremos en la cabecera de los archivos
    inyeccion_base = '''
# ==============================================================================
# 🛑 ⚠️ ¡ATENCIÓN! ESTE ES UN ARCHIVO TEMPORAL DE BUILD ⚠️ 🛑
# 
# 🚫 NO EDITES ESTE ARCHIVO 🚫
# Cualquier cambio que hagas aquí se PERDERÁ al volver a compilar.
# Por favor, realiza tus cambios en los archivos de la carpeta principal.
# ==============================================================================
# --- INYECCIÓN AUTOMÁTICA PARA PYINSTALLER ---
import sys
import os
import subprocess

def resource_path(relative_path):
    """ Obtiene la ruta al recurso empaquetado (solo lectura) """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(base_path) == "gui_app":
            base_path = os.path.dirname(base_path)
        if os.path.basename(base_path) == "EMG_Ejecutable_Build":
            base_path = os.path.dirname(base_path)
    return os.path.join(base_path, relative_path)

def user_data_path(relative_path):
    """ Obtiene la ruta a los datos del usuario (lectura/escritura) """
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(base_path) == "gui_app":
            base_path = os.path.dirname(base_path)
        if os.path.basename(base_path) == "EMG_Ejecutable_Build":
            base_path = os.path.dirname(base_path)
    return os.path.join(base_path, relative_path)

def lanzar_script(script_name, args=[]):
    """ Genera el comando de subprocess compatible tanto en Python como en .exe """
    if getattr(sys, 'frozen', False):
        exe_name = script_name.replace('.py', '.exe')
        exe_path = os.path.join(os.path.dirname(sys.executable), exe_name)
        return [exe_path] + args
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(base_path) == "gui_app":
            base_path = os.path.dirname(base_path)
        return [sys.executable, os.path.join(base_path, script_name)] + args
# ---------------------------------------------
'''

    def parchear_archivo(ruta_absoluta, reemplazos_texto, reemplazos_regex=None):
        if reemplazos_regex is None:
            reemplazos_regex = []
        with open(ruta_absoluta, 'r', encoding='utf-8') as f:
            contenido = f.read()

        # Inyectar las funciones al inicio del archivo si no existen
        if "def resource_path" not in contenido:
            if contenido.startswith("# -*- coding: utf-8 -*-"):
                contenido = contenido.replace("# -*- coding: utf-8 -*-", "# -*- coding: utf-8 -*-\n" + inyeccion_base, 1)
            else:
                contenido = inyeccion_base + contenido

        # Aplicar reemplazos exactos
        for viejo, nuevo in reemplazos_texto:
            contenido = contenido.replace(viejo, nuevo)

        # Aplicar reemplazos por expresiones regulares
        for patron, nuevo in reemplazos_regex:
            contenido = re.sub(patron, nuevo, contenido)

        with open(ruta_absoluta, 'w', encoding='utf-8') as f:
            f.write(contenido)

    # --- 1. INYECTAR CABECERA EN TODOS LOS ARCHIVOS .PY ---
    print("Inyectando funciones base en todos los archivos .py...")
    for root, dirs, files in os.walk(build_dir):
        if 'venv' in root: continue
        for file in files:
            if file.endswith('.py') and file not in ["crear_entorno_ejecutable.py", "aplicar_parches_ejecutable.py", "refactor_experimental.py", "refactor_experimental2.py", "autoforge_patcher.py"]:
                ruta = os.path.join(root, file)
                parchear_archivo(ruta, [])

    # --- 2. REEMPLAZOS ESPECÍFICOS PARA ADQUISICIÓN ---
    reemplazos_daq = [
        ("os.path.exists('metronome_config.json')", "os.path.exists(user_data_path('metronome_config.json'))"),
        ("open('metronome_config.json'", "open(user_data_path('metronome_config.json')"),
        ("base_dir = \"base_de_datos_electrodos\"", "base_dir = user_data_path(\"base_de_datos_electrodos\")"),
        ('test_dir = os.path.join(script_dir, "base_de_datos_electrodos"', 'test_dir = os.path.join(user_data_path("base_de_datos_electrodos")'),
        ('ruta_palabras = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palabras.txt")', 'ruta_palabras = user_data_path("palabras.txt")'),
        ("python_executable = sys.executable", ""),
        ("script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')", ""),
        ("word_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ventana_palabras.py')", ""),
    ]
    reemplazos_regex_daq = [
        (r"\[\s*python_executable,\s*script_path,\s*(.*?)\]", r"lanzar_script('metronomo_visual.py', [\1])"),
        (r"\[\s*python_executable,\s*word_script_path,\s*(.*?)\]", r"lanzar_script('ventana_palabras.py', [\1])")
    ]
    for archivo in ["CodigoUnificador_integrado.py", "Nandu_AutoForge_DAQ.py"]:
        ruta = os.path.join(build_dir, archivo)
        if os.path.exists(ruta):
            parchear_archivo(ruta, reemplazos_daq, reemplazos_regex_daq)

    # --- 3. REEMPLAZOS PARA GUI_APP Y DOCKERS ---
    viejo_launch = '''    def _launch_external(self, script_name):
        import subprocess
        import os
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script_path = os.path.join(root_dir, script_name)
        
        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Error", f"No se encontró el script: {script_name}")
            return
            
        try:
            if sys.platform == "win32":
                subprocess.Popen([sys.executable, script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen([sys.executable, script_path])'''

    nuevo_launch = '''    def _launch_external(self, script_name):
        import subprocess
        import os
        from PySide6.QtWidgets import QMessageBox
        
        comando = lanzar_script(script_name)
        
        try:
            if sys.platform == "win32":
                subprocess.Popen(comando, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(comando)'''

    reemplazos_main = [
        ("assets_dir = gui_dir / \"assets\"", "assets_dir = Path(resource_path(os.path.join('gui_app', 'assets')))"),
        ("search_dirs = [assets_dir, gui_dir, root_dir, pictures_dir]", "search_dirs = [assets_dir, pictures_dir]"),
        ("md_path = os.path.abspath(os.path.join(os.path.dirname(__file__), \"..\", \"justificacion_matematica.md\"))", "md_path = resource_path('justificacion_matematica.md')"),
        ('comparative_path = os.path.join(root_dir, "analisis_comparativos")', 'comparative_path = user_data_path("analisis_comparativos")'),
        ('db_path = os.path.join(root_dir, "base_de_datos_electrodos")', 'db_path = user_data_path("base_de_datos_electrodos")'),
        (viejo_launch, nuevo_launch)
    ]
    ruta_main = os.path.join(build_dir, "gui_app", "main_app.py")
    if os.path.exists(ruta_main):
        parchear_archivo(ruta_main, reemplazos_main)

    # --- 4. REEMPLAZOS PARA EL LANZADOR Y VISORES AUXILIARES ---
    reemplazos_bases = []
    reemplazos_regex_bases = [
        (r"self\.BASE_DIR\s*=\s*os\.path\.join\(.*?,?\s*[\"']base_de_datos_electrodos[\"']\)", r"self.BASE_DIR = user_data_path('base_de_datos_electrodos')"),
        (r"self\.BASE_DIR\s*=\s*[\"']base_de_datos_electrodos[\"']", r"self.BASE_DIR = user_data_path('base_de_datos_electrodos')"),
        (r"BASE_DIR\s*=\s*os\.path\.join\(.*?,?\s*[\"']base_de_datos_electrodos[\"']\)", r"BASE_DIR = user_data_path('base_de_datos_electrodos')"),
        (r"BASE_DIR\s*=\s*[\"']base_de_datos_electrodos[\"']", r"BASE_DIR = user_data_path('base_de_datos_electrodos')"),
        (r"base_dir\s*=\s*[\"']base_de_datos_electrodos[\"']", r"base_dir = user_data_path('base_de_datos_electrodos')"),
        (r"self\.base_folder\s*=\s*[\"']base_de_datos_electrodos[\"']", r"self.base_folder = user_data_path('base_de_datos_electrodos')"),
        (r"self\.fuente_dir\s*=\s*[\"']base_de_datos_electrodos[\"']", r"self.fuente_dir = user_data_path('base_de_datos_electrodos')"),
        (r"comp_dir\s*=\s*[\"']analisis_comparativos[\"']", r"comp_dir = user_data_path('analisis_comparativos')"),
        (r"destino_dir\s*=\s*[\"']base_de_datos_letras[\"']", r"destino_dir = user_data_path('base_de_datos_letras')"),
        (r"self\.destino_dir\s*=\s*[\"']base_de_datos_letras[\"']", r"self.destino_dir = user_data_path('base_de_datos_letras')")
    ]
    archivos_auxiliares = [
        "visor_csv_interactivo.py", "plotter_calibrado.py", "extractor_de_datos_procesados.py", 
        "electrode_viewer_4.py", "editor_mediciones.py", "analisis_por_track_integrado.py", 
        "analisis_por_track_integrado_experimental.py", "correlaciondeseñales.py", 
        "actualizar_metadata.py", "migrar_mediciones_por_fecha.py"
    ]
    for archivo in archivos_auxiliares:
        ruta = os.path.join(build_dir, archivo)
        if os.path.exists(ruta):
            parchear_archivo(ruta, reemplazos_bases, reemplazos_regex_bases)

    # Parche especial avanzado para Sistema_de_Adquisicion_Emg.py (El Lanzador)
    reemplazo_launcher = """def launch_script(script_path_rel):
    comando = lanzar_script(script_path_rel)
    try:
        if sys.platform == 'win32':
            subprocess.Popen(comando, creationflags=0x08000000)
        else:
            subprocess.Popen(comando)
    except Exception as e:
        messagebox.showerror('Error de Lanzamiento', str(e))
    return
def launch_script_original(script_path_rel):"""
    
    ruta_sistema = os.path.join(build_dir, "Sistema_de_Adquisicion_Emg.py")
    if os.path.exists(ruta_sistema):
        parchear_archivo(ruta_sistema, [
            ("BASE_DIR = Path(__file__).resolve().parent", "BASE_DIR = Path(user_data_path(''))"),
            ("ICON_DIR = BASE_DIR / \"icons\"", "ICON_DIR = Path(resource_path('icons'))"),
            ('pictures_dir = BASE_DIR / "DataConfig" / "Pictures"', 'pictures_dir = Path(resource_path("DataConfig/Pictures"))'),
            ("def launch_script(script_path_rel):", reemplazo_launcher)
        ])

    print("\n🚀 ¡Parches aplicados a TODA la suite de códigos! Entorno listo para la compilación (Fases 1.1 y 1.2 superadas).")

if __name__ == "__main__":
    aplicar_parches()