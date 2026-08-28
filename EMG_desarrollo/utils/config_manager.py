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
    # Depresor del ángulo de la boca / DAO
    "depresor anguli oris": "#8a2be2",     # Violeta / Púrpura
    "depresor": "#8a2be2",
    "depressor": "#8a2be2",
    "dao": "#8a2be2",
    
    # Orbicular de los labios / ojos
    "orbicularis oris": "#ffff00",         # Amarillo
    "orbicularis": "#ffff00",
    "orbicular": "#ffff00",
    "orbicularis sup": "#ffff00",
    "orbicularis inf": "#ffff00",
    "orbicularis oculi": "#ffff00",
    
    # Milohioideo
    "mylohyoid": "#39ff14",                # Verde Neón
    "milohyoid": "#39ff14",
    "milohioideo": "#39ff14",
    "milohioide": "#39ff14",
    "milohiode": "#39ff14",
    "milo": "#39ff14",
    
    # Digástrico / Vientre anterior
    "digastrico": "#ffaa00",               # Naranja brillante
    "digastric": "#ffaa00",
    "digastrio": "#ffaa00",
    "digastrioo": "#ffaa00",
    "anterior belly": "#ffaa00",
    "vientre anterior": "#ffaa00",
    
    # Cigomático mayor y menor
    "zygomaticus major": "#ff00ff",        # Magenta
    "zygomaticus": "#ff00ff",
    "cigomatico": "#ff00ff",
    "cigomatico mayor": "#ff00ff",
    "cigomatico menor": "#ff00ff",
    
    # Masetero
    "masseter": "#00ffcc",                 # Cyan brillante
    "masetero": "#00ffcc",
    "masetero superficial": "#00ffcc",
    "masetero profundo": "#00ffcc",
    
    # Elevador del labio superior
    "levatori oris": "#00bfff",            # Azul eléctrico / DeepSkyBlue
    "levator labii superioris": "#00bfff",
    "elevador": "#00bfff",
    "elevador del labio": "#00bfff",
    
    # Temporal
    "temporal": "#ff69b4",                 # Rosa Neón (HotPink)
    "temporalis": "#ff69b4",
    
    # Buccinador
    "buccinador": "#ff7f50",               # Coral
    "buccinator": "#ff7f50",
    
    # Mentoniano
    "mentalis": "#adff2f",                 # Verde Lima
    "mentoniano": "#adff2f",
    
    # Pterigoideo
    "pterigoideo": "#00e5ff",              # Aqua
    "pterygoid": "#00e5ff",
    
    # Risorio
    "risorio": "#e040fb",                  # Púrpura neón
    "risorius": "#e040fb",
    
    # Esternocleidomastoideo / Trapecio
    "esternocleidomastoideo": "#76ff03",    # Verde claro
    "scm": "#76ff03",
    "trapecio": "#ffd600",                 # Oro
    "trapezius": "#ffd600",
    
    # Micrófono / Canal 3 (Estrictamente Rojo)
    "micrófono": "#ff0000",
    "microfono": "#ff0000",
    "mic": "#ff0000",
    "canal 3": "#ff0000",
    "canal_3": "#ff0000"
}

# Paleta estética de 16 colores de alto contraste para resolver colisiones
DISTINCT_PALETTE = [
    "#8a2be2",  # Violeta
    "#ffff00",  # Amarillo
    "#39ff14",  # Verde neón
    "#ffaa00",  # Naranja
    "#00ffcc",  # Cyan brillante
    "#ff00ff",  # Magenta
    "#00bfff",  # Azul eléctrico
    "#ff69b4",  # Rosa neón
    "#adff2f",  # Verde lima
    "#ff7f50",  # Coral
    "#00e5ff",  # Aqua
    "#e040fb",  # Púrpura neón
    "#ffd600",  # Oro
    "#00e676",  # Verde esmeralda
    "#ff3d00",  # Naranja rojizo
    "#651fff"   # Azul índigo
]

def get_muscle_color(name, default=None):
    """
    Devuelve el color hex estandarizado o personalizado para un músculo dado.
    El canal 3 / micrófono siempre se asigna estrictamente a rojo (#ff0000).
    """
    if not name:
        return default or "#00ffcc"
        
    name_str = str(name).strip().lower()
    
    # 1. Comprobar si es micrófono o canal 3
    if "mic" in name_str or name_str in ("canal 3", "canal_3", "ch3", "dev1/ai3", "dev2/ai3"):
        return "#ff0000"
        
    # 2. Comprobar si el usuario definió un color personalizado en config_general.json
    try:
        mgr = ConfigManager()
        custom_colors = mgr.get("colores_musculos") or {}
        for k, v in custom_colors.items():
            if k.strip().lower() == name_str:
                return v
    except Exception:
        pass
        
    # 3. Comprobar coincidencias exactas o parciales en el diccionario canónico
    for k, v in MUSCLE_COLORS.items():
        if k in name_str or name_str in k:
            return v
            
    return default or "#00ffcc"

def get_unique_channel_colors(channels_info):
    """
    Recibe una lista de canales (nombres, diccionarios o tuplas) y devuelve
    una lista de colores hexadecimales garantizando que NINGÚN canal repita color.
    
    El micrófono / canal 3 siempre recibe #ff0000 de forma exclusiva.
    """
    used_colors = set()
    result_colors = []
    palette_idx = 0

    for item in channels_info:
        # Normalizar entrada
        if isinstance(item, dict):
            musc = item.get("musculo", "")
            preferred = item.get("color_hex", None)
            is_mic = item.get("is_mic", False) or ("mic" in str(musc).lower()) or (item.get("idx") == 3)
        elif isinstance(item, (tuple, list)):
            musc = item[1] if len(item) > 1 else str(item[0])
            preferred = item[2] if len(item) > 2 else None
            is_mic = ("mic" in str(musc).lower()) or (str(item[0]).lower() in ("canal 3", "canal_3", "3"))
        else:
            musc = str(item)
            preferred = None
            is_mic = ("mic" in musc.lower()) or (musc.lower() in ("canal 3", "canal_3", "3"))

        # El micrófono siempre es rojo
        if is_mic:
            result_colors.append("#ff0000")
            used_colors.add("#ff0000")
            continue

        # Color sugerido por preferencia del usuario o por nombre de músculo
        color = preferred or get_muscle_color(musc, default=None)
        
        # Si el color ya fue usado o coincide con rojo (#ff0000), buscar el siguiente de la paleta
        if color is None or color.lower() in [c.lower() for c in used_colors] or color.lower() == "#ff0000":
            while palette_idx < len(DISTINCT_PALETTE):
                candidate = DISTINCT_PALETTE[palette_idx]
                palette_idx += 1
                if candidate.lower() not in [c.lower() for c in used_colors] and candidate.lower() != "#ff0000":
                    color = candidate
                    break
            else:
                # Si se agotaron los colores predefinidos, generar color contrastante determinista
                color = f"#{abs(hash(musc + str(len(result_colors)))) % 0xFFFFFF:06x}"

        used_colors.add(color)
        result_colors.append(color)

    return result_colors

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
    "colores_musculos": {},
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

    def save(self):
        """Alias compatible para save_config."""
        self.save_config()

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

    def set(self, section, key, value=None):
        """
        Actualiza un valor y guarda automáticamente.
        Soporta:
        - set('canales', dict_canales)
        - set('adquisicion', 'bpm', 60)
        """
        if value is None:
            self.config[section] = key
        else:
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
