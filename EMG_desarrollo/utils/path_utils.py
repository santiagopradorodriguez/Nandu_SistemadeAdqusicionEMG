# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo centralizado de resolución de rutas y directorios de usuario.
# ==============================================================================

import os
import sys

def get_project_root():
    """
    Retorna el directorio raíz del proyecto:
    - En modo ejecutable congelado (PyInstaller): carpeta contenedora del ejecutable
      (fuera de '_internal' para no mezclar datos de usuario con librerías).
    - En modo desarrollo: carpeta raíz 'EMG_desarrollo'.
    """
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(exe_dir) == '_internal':
            exe_dir = os.path.dirname(exe_dir)
        return exe_dir
    else:
        cur = os.path.abspath(os.path.dirname(__file__))
        while cur:
            if os.path.basename(cur) in ('EMG_desarrollo', 'EMG_Ejecutable_Build'):
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_database_path():
    """Ruta a la carpeta base_de_datos_electrodos (fuera de _internal)."""
    return os.path.join(get_project_root(), 'base_de_datos_electrodos')

def get_comparative_path():
    """Ruta a la carpeta analisis_comparativos."""
    return os.path.join(get_project_root(), 'analisis_comparativos')

def get_session_analysis_path():
    """Ruta a la carpeta analisis_de_sesiones."""
    return os.path.join(get_project_root(), 'analisis_de_sesiones')

def get_resource_path(relative_path):
    """
    Obtiene la ruta a un recurso empaquetado (solo lectura) o local.
    Prioriza sys._MEIPASS si existe, luego la raíz del proyecto.
    """
    if hasattr(sys, '_MEIPASS'):
        res_path = os.path.join(sys._MEIPASS, relative_path)
        if os.path.exists(res_path):
            return res_path
    return os.path.join(get_project_root(), relative_path)

def user_data_path(relative_path):
    """Alias para compatibilidad con código existente."""
    return os.path.join(get_project_root(), relative_path)
