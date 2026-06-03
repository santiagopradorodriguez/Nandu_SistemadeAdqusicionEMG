# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo logger.py del sistema NANDU LSD.
# ==============================================================================

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

def setup_logger(name="EMG_Studio", log_dir="logs", level=logging.DEBUG):
    """
    Configura y devuelve un logger estándar para la aplicación.
    Guarda los logs en archivos rotativos y los muestra en consola.
    """
    logger = logging.getLogger(name)
    
    # Evitar añadir múltiples handlers si ya se instanció
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Crear directorio de logs si no existe
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_log_dir = os.path.join(base_dir, log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    # Formato del log
    log_format = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Handler para Consola (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    # Por defecto en consola mostramos INFO para arriba (para no saturar)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # Handler para Archivo (Rotativo)
    log_file = os.path.join(full_log_dir, f"{name.lower()}.log")
    # MaxBytes = 5 MB, backupCount = 3 archivos
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG) # En el archivo guardamos TODO (Debug)
    logger.addHandler(file_handler)

    return logger

# Instancia global por defecto
logger = setup_logger()
