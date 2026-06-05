# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Gestor de carga y guardado de configuraciones globales del sistema.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Gestor de carga y guardado de configuraciones globales del sistema.
# ==============================================================================

import os
import json

CONFIG_FILE_NAME = "config_general.json"

# Valores por defecto para toda la aplicación
DEFAULT_CONFIG = {
    "estetica_global": {
        "tema_oscuro": True,
        "color_fondo": "#050505",
        "color_acento": "#00ffcc"
    },
    "adquisicion": {
        "frecuencia_muestreo": 2000.0,
        "ruido_segundos": 3.0,
        "nidaq_channels": ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]
    },
    "analisis_extractor": {
        "frecuencia_remuestreo": 500.0
    },
    "canales": {
        "Canal 0": {
            "musculo": "Depresor Anguli Oris",
            "color_hex": "#00ffcc",
            "factor_calibracion": 495.0
        },
        "Canal 1": {
            "musculo": "Orbicularis Oris",
            "color_hex": "#ff00ff",
            "factor_calibracion": 495.0
        },
        "Canal 2": {
            "musculo": "Mylohyoid",
            "color_hex": "#ffff00",
            "factor_calibracion": 495.0
        },
        "Canal 3": {
            "musculo": "Canal 3",
            "color_hex": "#ff5500",
            "factor_calibracion": 495.0
        }
    }
}

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # El archivo de configuración se guardará en la raíz del proyecto
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.config_path = os.path.join(base_dir, CONFIG_FILE_NAME)
        self.config = self._load_config()

    def _load_config(self):
        if not os.path.exists(self.config_path):
            self._save_default_config()
            return DEFAULT_CONFIG.copy()
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Merge con default en caso de que falten claves nuevas
                return self._merge_dicts(DEFAULT_CONFIG.copy(), data)
        except Exception as e:
            print(f"Error cargando config: {e}. Usando defaults.")
            return DEFAULT_CONFIG.copy()

    def _merge_dicts(self, default_dict, user_dict):
        merged = default_dict.copy()
        for k, v in user_dict.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._merge_dicts(merged[k], v)
            else:
                merged[k] = v
        return merged

    def _save_default_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando config por defecto: {e}")

    def save_config(self):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando config: {e}")

    def get(self, section, key=None):
        """
        Obtiene un valor de configuración.
        Si key es None, devuelve la sección entera.
        """
        if section not in self.config:
            return None
        if key is None:
            return self.config[section]
        return self.config[section].get(key)

    def set(self, section, key, value):
        """
        Actualiza un valor y guarda automáticamente.
        """
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config()

    def get_channel_config(self, index):
        """Devuelve dict con la config del canal Ej: index=0 -> 'Canal 0'"""
        ch_key = f"Canal {index}"
        canales = self.get("canales")
        if ch_key in canales:
            return canales[ch_key]
        # Fallback genérico si no existe
        return {"musculo": ch_key, "color_hex": "#ffffff", "factor_calibracion": 1.0}
