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

# Paleta canónica de colores por músculo (y rojo estricto para micrófono / canal 3)
MUSCLE_COLORS = {
    "depresor anguli oris": "#8a2be2",     # Violeta / Púrpura
    "orbicularis oris": "#ffff00",         # Amarillo
    "mylohyoid": "#39ff14",                # Verde
    "milohyoid": "#39ff14",
    "milohioideo": "#39ff14",
    "milohioide": "#39ff14",
    "milohiode": "#39ff14",
    "levatori oris": "#00ffcc",            # Cyan / Celeste brillante
    "levator labii superioris": "#00ffcc",
    "anterior belly": "#ffaa00",           # Naranja
    "zygomaticus major": "#ff00ff",        # Magenta
    "micrófono": "#ff0000",                # Rojo
    "microfono": "#ff0000",
    "mic": "#ff0000",
    "canal 3": "#ff0000",
    "canal_3": "#ff0000"
}

def get_muscle_color(name, default="#00ffcc"):
    """Devuelve el color hex estandarizado para un músculo dado. El canal 3 / micrófono siempre es rojo."""
    if not name:
        return default
    name_str = str(name).strip().lower()
    if "mic" in name_str or name_str in ("canal 3", "canal_3", "ch3", "dev1/ai3", "dev2/ai3"):
        return "#ff0000"
    for k, v in MUSCLE_COLORS.items():
        if k in name_str or name_str in k:
            return v
    return default

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
        "bpm": 60,
        "tiempo_descanso": 10.0,
        "nidaq_channels": ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]
    },
    "analisis_extractor": {
        "frecuencia_remuestreo": 500.0
    },
    "canales": {
        "Canal 0": {
            "musculo": "Depresor Anguli Oris",
            "color_hex": "#8a2be2",
            "factor_calibracion": 495.0
        },
        "Canal 1": {
            "musculo": "Orbicularis Oris",
            "color_hex": "#ffff00",
            "factor_calibracion": 495.0
        },
        "Canal 2": {
            "musculo": "Mylohyoid",
            "color_hex": "#39ff14",
            "factor_calibracion": 495.0
        },
        "Canal 3": {
            "musculo": "Micrófono",
            "color_hex": "#ff0000",
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
        from .path_utils import get_project_root
        base_dir = get_project_root()
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
        canales = self.get("canales") or {}
        if ch_key in canales:
            ch_data = canales[ch_key].copy()
            musculo_nom = ch_data.get("musculo", ch_key)
            if index == 3 or "mic" in musculo_nom.lower():
                ch_data["color_hex"] = "#ff0000"
            else:
                ch_data["color_hex"] = get_muscle_color(musculo_nom, ch_data.get("color_hex", "#00ffcc"))
            return ch_data
            
        # Fallback genérico si no existe
        if index == 3:
            return {"musculo": "Micrófono", "color_hex": "#ff0000", "factor_calibracion": 1.0}
        return {"musculo": ch_key, "color_hex": get_muscle_color(ch_key, "#00ffcc"), "factor_calibracion": 1.0}
