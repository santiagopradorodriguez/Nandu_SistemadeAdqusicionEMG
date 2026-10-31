# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Gestor de carga y guardado de configuraciones globales del sistema.
# ==============================================================================

import os
import json

CONFIG_FILE_NAME = "config_general.json"

# Paleta canónica oficial de colores por músculo establecida para todo el sistema
MUSCLE_COLORS = {
    # Anterior belly del digástrico / Digástrico: NARANJA INTENSO (#ff754b)
    "anterior belly": "#ff754b",
    "anterior belly of digastric": "#ff754b",
    "vientre anterior": "#ff754b",
    "vientre anterior del digastrico": "#ff754b",
    "vientre anterior del digástrico": "#ff754b",
    "digastrico": "#ff754b",
    "digástrico": "#ff754b",
    "digastric": "#ff754b",
    "digastrio": "#ff754b",
    "digastrioo": "#ff754b",

    # Zigomático: VERDE AZULADO
    "zygomaticus major": "#00a896",
    "zygomaticus mayor": "#00a896",
    "zygomaticus": "#00a896",
    "zygomatic": "#00a896",
    "cigomatico": "#00a896",
    "cigomático": "#00a896",
    "cigomatico mayor": "#00a896",
    "cigomático mayor": "#00a896",
    "zigomatico": "#00a896",
    "zigomático": "#00a896",
    "zigomatico mayor": "#00a896",
    "zigomático mayor": "#00a896",
    "cigomatico menor": "#00a896",
    "cigomático menor": "#00a896",
    "zigomatico menor": "#00a896",
    "zigomático menor": "#00a896",

    # Risorio: VERDE
    "risorio": "#00cc44",
    "risorius": "#00cc44",

    # Modíolo: VERDE AMARILLENTO
    "modiolo": "#a3e635",
    "modíolo": "#a3e635",
    "modiolus": "#a3e635",
    "nudo sonrisa": "#a3e635",
    "nudo de la sonrisa": "#a3e635",

    # Orbicularis: AMARILLO
    "orbicularis oris": "#ffff00",
    "orbicularis": "#ffff00",
    "orbicular": "#ffff00",
    "orbicular de los labios": "#ffff00",
    "orbicularis sup": "#ffff00",
    "orbicularis inf": "#ffff00",
    "orbicularis oculi": "#ffff00",

    # Depresor: VERDE MÁS OSCURO
    "depresor anguli oris": "#15803d",
    "depresor": "#15803d",
    "depressor": "#15803d",
    "dao": "#15803d",
    "depresor del angulo": "#15803d",
    "depresor del ángulo": "#15803d",
    "depresor del angulo de la boca": "#15803d",
    "depresor del ángulo de la boca": "#15803d",
    "depressor anguli oris": "#15803d",

    # Milohioideo: NARANJA
    "mylohyoid": "#ff7700",
    "milohyoid": "#ff7700",
    "milohioideo": "#ff7700",
    "milohioide": "#ff7700",
    "milohiode": "#ff7700",
    "milo": "#ff7700",

    # Masetero
    "masseter": "#00ffcc",
    "masetero": "#00ffcc",
    "masetero superficial": "#00ffcc",
    "masetero profundo": "#00ffcc",

    # Elevador del labio superior / ángulo oral
    "levatori oris": "#00bfff",
    "levator anguli oris": "#00bfff",
    "levator labii superioris": "#00bfff",
    "elevador": "#00bfff",
    "elevador del labio": "#00bfff",

    # Platisma
    "platysma": "#00e676",
    "platisma": "#00e676",

    # Temporal
    "temporal": "#ff69b4",
    "temporalis": "#ff69b4",

    # Buccinador
    "buccinador": "#ff7f50",
    "buccinator": "#ff7f50",

    # Mentoniano
    "mentalis": "#adff2f",
    "mentoniano": "#adff2f",

    # Pterigoideo
    "pterigoideo": "#00e5ff",
    "pterygoid": "#00e5ff",

    # Esternocleidomastoideo / Trapecio
    "esternocleidomastoideo": "#76ff03",
    "scm": "#76ff03",
    "trapecio": "#ffd600",
    "trapezius": "#ffd600",

    # Micrófono / Canal 3 (Rojo estándar)
    "micrófono": "#ff0000",
    "microfono": "#ff0000",
    "mic": "#ff0000",
    "canal 3": "#ff0000",
    "canal_3": "#ff0000",
    "ch3": "#ff0000"
}

# Paleta estética de 16 colores de alto contraste para resolver colisiones
DISTINCT_PALETTE = [
    "#ff754b",  # Naranja intenso (Anterior Belly)
    "#00a896",  # Verde azulado (Zigomático)
    "#00cc44",  # Verde (Risorio)
    "#a3e635",  # Verde amarillento (Modíolo)
    "#ffff00",  # Amarillo (Orbicularis)
    "#15803d",  # Verde más oscuro (Depresor)
    "#ff7700",  # Naranja (Milohioideo)
    "#ff0000",  # Rojo (Micrófono)
    "#00bfff",  # Azul eléctrico
    "#ff00ff",  # Magenta
    "#00ffcc",  # Cyan brillante
    "#ff69b4",  # Rosa neón
    "#8a2be2",  # Violeta
    "#ff7f50",  # Coral
    "#00e5ff",  # Aqua
    "#ffd600",  # Oro
    "#651fff"   # Azul índigo
]

# Lista anatómica estándar de músculos faciales para selección rápida en diálogos
STANDARD_MUSCLES_LIST = [
    "Anterior Belly of Digastric",
    "Zygomaticus Major",
    "Orbicularis Oris",
    "Risorio",
    "Modíolo",
    "Depresor Anguli Oris",
    "Mentoniano",
    "Milohioideo",
    "Masetero",
    "Micrófono"
]

def _is_generic_or_mic_channel(name, idx=None):
    """
    Determina si un canal corresponde a un micrófono o si es el Canal 3 sin un músculo real asignado.
    Si el usuario asignó un músculo real (ej. 'Risorio', 'Orbicularis Oris') al Canal 3, devuelve False.
    """
    name_str = str(name or "").strip().lower()
    if "mic" in name_str or "audio" in name_str:
        return True
    generic_ch3_names = ("", "canal 3", "canal_3", "ch3", "ai3", "3", "dev1/ai3", "dev2/ai3", "dev3/ai3")
    if idx is not None:
        idx_str = str(idx).strip().lower()
        is_idx_3 = (idx == 3) or (idx_str in ("3", "canal 3", "canal_3", "ch3", "ai3", "dev1/ai3", "dev2/ai3"))
        if is_idx_3 and name_str in generic_ch3_names:
            return True
        return False
    return name_str in ("canal 3", "canal_3", "ch3", "dev1/ai3", "dev2/ai3")


def get_muscle_color(name, default=None):
    """
    Devuelve el color hex estandarizado o personalizado para un músculo dado.
    El micrófono (o Canal 3 genérico sin músculo asignado) se asigna a rojo (#ff0000).
    """
    if not name:
        return default or "#ff0000"
        
    name_str = str(name).strip().lower()
    
    # 1. Comprobar si es micrófono o canal 3 genérico
    if _is_generic_or_mic_channel(name_str):
        return "#ff0000"
        
    # 2. Comprobar si el usuario definió un color personalizado en config_general.json
    try:
        mgr = ConfigManager()
        custom_colors = mgr.get("colores_musculos") or {}
        # Búsqueda exacta primero
        for k, v in custom_colors.items():
            if k.strip().lower() == name_str:
                return v
        # Búsqueda por coincidencia parcial de sinónimos en custom_colors
        for k, v in custom_colors.items():
            k_clean = k.strip().lower()
            if k_clean and (k_clean in name_str or name_str in k_clean):
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
    una lista de colores hexadecimales garantizando que cada canal reciba su color canónico
    y que NINGÚN canal repita color innecesariamente.
    """
    used_colors = set()
    result_colors = []
    palette_idx = 0

    for item in channels_info:
        # Normalizar entrada
        if isinstance(item, dict):
            musc = item.get("musculo", "")
            preferred = item.get("color_hex", None)
            idx_val = item.get("idx", None)
            explicit_mic = bool(item.get("is_mic", False))
            # Solo forzar micrófono si el nombre realmente es de micrófono o genérico de Canal 3
            is_mic = _is_generic_or_mic_channel(musc, idx=idx_val) or (explicit_mic and _is_generic_or_mic_channel(musc, idx=3))
        elif isinstance(item, (tuple, list)):
            musc = item[1] if len(item) > 1 else str(item[0])
            preferred = item[2] if len(item) > 2 else None
            is_mic = _is_generic_or_mic_channel(musc, idx=item[0])
        else:
            musc = str(item)
            preferred = None
            is_mic = _is_generic_or_mic_channel(musc)

        # El micrófono siempre recibe su color de micrófono (#ff0000)
        if is_mic:
            mic_col = get_muscle_color("micrófono", default="#ff0000")
            result_colors.append(mic_col)
            used_colors.add(mic_col.lower())
            continue

        # Si el músculo tiene un color canónico y preferred era #ff0000 heredado de un Micrófono previo,
        # priorizar el color canónico del músculo nuevo
        canonical_col = get_muscle_color(musc, default=None)
        if canonical_col and preferred and str(preferred).lower() == "#ff0000" and canonical_col.lower() != "#ff0000":
            color = canonical_col
        elif preferred and str(preferred).startswith("#") and str(preferred).lower() not in [c.lower() for c in used_colors]:
            color = preferred
        else:
            color = canonical_col or preferred
        
        # Si el color ya fue usado o no está definido, buscar de la paleta
        if color is None or color.lower() in [c.lower() for c in used_colors]:
            while palette_idx < len(DISTINCT_PALETTE):
                candidate = DISTINCT_PALETTE[palette_idx]
                palette_idx += 1
                if candidate.lower() not in [c.lower() for c in used_colors]:
                    color = candidate
                    break
            else:
                color = f"#{abs(hash(str(musc) + str(len(result_colors)))) % 0xFFFFFF:06x}"

        used_colors.add(color.lower())
        result_colors.append(color)

    return result_colors

# Valores por defecto para toda la aplicación (Protocolo v6.0)
DEFAULT_CONFIG = {
    "estetica_global": {
        "tema_oscuro": True,
        "color_fondo": "#050505",
        "color_acento": "#00ffcc"
    },
    "adquisicion": {
        "frecuencia_muestreo": 2000.0,
        "ruido_segundos": 3.0,
        "bpm": 30,
        "subdivisiones": 4,
        "notch_50hz_activo": False,
        "modo_prueba_activo": False,
        "tiempo_descanso": 10.0,
        "nidaq_channels": ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]
    },
    "protocolo_secuencia": {
        "usar_secuencia_debruijn": True,
        "repeticiones_continua_default": 25,
        "repeticiones_discreta_default": 5,
        "abrir_fotometria_al_iniciar": False,
        "calibrar_cvm_por_defecto": False,
        "cvm_modo": "guiado_sin_metronomo",
        "cvm_duracion_contraccion_s": 2.5,
        "cvm_duracion_descanso_s": 2.5,
        "cvm_tomas_por_vocal": 2,
        "cvm_vocales": ["A", "I", "U"]
    },
    "jitter_metronomo": {
        "habilitado": False,
        "porcentaje": 25.0,
        "modo": "exponential_hazard",
        "guardar_latencia_sincronizacion": True
    },
    "audio_speaking": {
        "mute_speaking": False,
        "modo_locucion": "fases_y_continua",
        "adelantar_voz_subpulso": True,
        "blanking_acustico_mic_s": 0.55
    },
    "medidor_pca_vivo": {
        "habilitado": True,
        "target_length": 20,
        "min_pulsos_clustering": 4,
        "algoritmo_clustering": "K-Means",
        "mostrar_centroides": True
    },
    "display_remoto": {
        "habilitado": True,
        "modo_transmision": "tailscale_lan",
        "puerto": 8765,
        "token": "",
        "tailscale_ip_pc": "auto",
        "tailscale_ip_celu": "auto",
        "auto_adb_reverse": True,
        "modo_sujeto_ciego_ocultar_visor_local": False
    },
    "analisis_extractor": {
        "frecuencia_remuestreo": 500.0,
        "target_length_pca": 20,
        "soportar_secuencia_continua": True,
        "soportar_pulsos_jitter": True,
        "excluir_pulsos_invalidos_manuales": True,
        "alineacion_default": "Pico Volumen Micrófono"
    },
    "colores_musculos": {},
    "canales": {
        "Canal 0": {
            "musculo": "Anterior Belly",
            "color_hex": "#ff754b",
            "factor_calibracion": 495.0
        },
        "Canal 1": {
            "musculo": "Zygomaticus Major",
            "color_hex": "#00a896",
            "factor_calibracion": 495.0
        },
        "Canal 2": {
            "musculo": "Orbicularis Oris",
            "color_hex": "#ffff00",
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
        try:
            from .path_utils import get_project_root
            base_dir = get_project_root()
        except Exception:
            try:
                from utils.path_utils import get_project_root
                base_dir = get_project_root()
            except Exception:
                base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(base_dir, CONFIG_FILE_NAME)
        self.config = self._load_config()

    def _load_config(self):
        base_merged = DEFAULT_CONFIG.copy()
        if not os.path.exists(self.config_path):
            self._save_default_config()
        else:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    base_merged = self._merge_dicts(base_merged, data)
            except Exception as e:
                print(f"Error cargando config: {e}. Usando defaults.")
        # Superponer config_local.json si existe (para IPs de Tailscale/VM específicas de cada PC sin subirlas a GitHub)
        local_path = os.path.join(os.path.dirname(self.config_path), "config_local.json")
        if os.path.exists(local_path):
            try:
                with open(local_path, 'r', encoding='utf-8') as f_loc:
                    local_data = json.load(f_loc)
                    base_merged = self._merge_dicts(base_merged, local_data)
            except Exception:
                pass
        return base_merged

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
        idx_str = str(index).strip()
        if idx_str.lower().startswith("canal"):
            num_part = idx_str[5:].strip()
            ch_key = f"Canal {num_part}"
            idx_num = int(num_part) if num_part.isdigit() else -1
        elif idx_str.isdigit():
            ch_key = f"Canal {idx_str}"
            idx_num = int(idx_str)
        else:
            ch_key = idx_str
            idx_num = -1

        canales = self.get("canales") or {}
        if ch_key in canales:
            ch_data = canales[ch_key].copy()
            musculo_nom = ch_data.get("musculo", ch_key)
            if _is_generic_or_mic_channel(musculo_nom, idx=idx_num):
                ch_data["color_hex"] = "#ff0000"
            else:
                ch_data["color_hex"] = get_muscle_color(musculo_nom, ch_data.get("color_hex", "#00ffcc"))
            return ch_data
            
        # Fallback genérico si no existe
        if idx_num == 3:
            return {"musculo": "Micrófono", "color_hex": "#ff0000", "factor_calibracion": 495.0}
        return {"musculo": ch_key, "color_hex": get_muscle_color(ch_key, "#00ffcc"), "factor_calibracion": 495.0}

    def update_channel_muscle(self, index, muscle_name, save=True):
        """Actualiza el músculo y color canónico de un canal físico (ej. index=3 -> 'Canal 3')."""
        idx_str = str(index).strip()
        if idx_str.lower().startswith("canal"):
            num_part = idx_str[5:].strip()
            ch_key = f"Canal {num_part}"
            idx_num = int(num_part) if num_part.isdigit() else -1
        elif idx_str.isdigit():
            ch_key = f"Canal {idx_str}"
            idx_num = int(idx_str)
        else:
            ch_key = idx_str
            idx_num = -1

        if "canales" not in self.config or not isinstance(self.config["canales"], dict):
            self.config["canales"] = {}
        if ch_key not in self.config["canales"]:
            self.config["canales"][ch_key] = {"factor_calibracion": 495.0}
        musc_clean = str(muscle_name or "").strip() or ch_key
        self.config["canales"][ch_key]["musculo"] = musc_clean
        if _is_generic_or_mic_channel(musc_clean, idx=idx_num):
            self.config["canales"][ch_key]["color_hex"] = "#ff0000"
        else:
            self.config["canales"][ch_key]["color_hex"] = get_muscle_color(musc_clean, "#00ffcc")
        if save:
            self.save_config()
        return self.config["canales"][ch_key]

    def set_active_channels_default(self, active_hw_indices, save=True):
        """Persiste 'activo_por_defecto' en cada 'Canal {hw}' en config_general.json."""
        active_set = set(active_hw_indices or [])
        if "canales" not in self.config or not isinstance(self.config["canales"], dict):
            self.config["canales"] = {}
        for hw_idx in range(16):
            ch_key = f"Canal {hw_idx}"
            if ch_key in self.config["canales"] or hw_idx in active_set:
                if ch_key not in self.config["canales"]:
                    self.config["canales"][ch_key] = {
                        "musculo": "Micrófono" if hw_idx == 3 else ch_key,
                        "color_hex": "#ff0000" if hw_idx == 3 else get_muscle_color(ch_key, "#00ffcc"),
                        "factor_calibracion": 495.0
                    }
                self.config["canales"][ch_key]["activo_por_defecto"] = (hw_idx in active_set)
        if save:
            self.save_config()

    def get_protocol_config(self):
        """Devuelve la configuración de protocolo y CVM combinada con defaults v6.0."""
        return self._merge_dicts(
            DEFAULT_CONFIG["protocolo_secuencia"].copy(),
            self.get("protocolo_secuencia") or {}
        )

    def get_jitter_config(self):
        """Devuelve la configuración del modo Jitter anti-sincronización."""
        return self._merge_dicts(
            DEFAULT_CONFIG["jitter_metronomo"].copy(),
            self.get("jitter_metronomo") or {}
        )

    def get_speaking_config(self):
        """Devuelve la configuración de comandos de voz (Speaking) y blanking acústico."""
        return self._merge_dicts(
            DEFAULT_CONFIG["audio_speaking"].copy(),
            self.get("audio_speaking") or {}
        )

    def get_pca_live_config(self):
        """Devuelve la configuración del medidor PCA / Separabilidad en tiempo real."""
        return self._merge_dicts(
            DEFAULT_CONFIG["medidor_pca_vivo"].copy(),
            self.get("medidor_pca_vivo") or {}
        )

    def get_remote_display_config(self):
        """Devuelve la configuración del servidor de estímulos y pantalla externa (sujeto ciego)."""
        return self._merge_dicts(
            DEFAULT_CONFIG["display_remoto"].copy(),
            self.get("display_remoto") or {}
        )

    def get_extractor_config(self):
        """Devuelve la configuración de extracción para generador_pca_umap y analisis_trevisan."""
        return self._merge_dicts(
            DEFAULT_CONFIG["analisis_extractor"].copy(),
            self.get("analisis_extractor") or {}
        )

    def get_visualization_config(self) -> dict:
        """Retorna configuración de visualización, umbrales anti-picos y analizador espectral."""
        vis = self.get("visualizacion") or {}
        adq = self.get("adquisicion") or {}
        return {
            "limitar_rango_500uv": bool(vis.get("limitar_rango_500uv", adq.get("limitar_rango_500uv", True))),
            "techo_fisiologico_uv": float(vis.get("techo_fisiologico_uv", 500.0)),
            "umbral_picos_snr_uv": float(vis.get("umbral_picos_snr_uv", 200.0)),
            "duracion_plot_s": float(vis.get("duracion_plot_s", 5.0)),
            "mostrar_envolvente_rms": bool(vis.get("mostrar_envolvente_rms", True)),
            "analizador_formantes_ch3": bool(vis.get("analizador_formantes_ch3", True)),
        }

    def persistir_estado_autoforge(self, estado_dict: dict) -> bool:
        """
        Guarda bidireccionalmente en config_local.json los parámetros personalizados
        desde AutoForge (BPM, canales, ganancias, filtros, anti-spikes, rutas) sin
        ensuciar config_general.json (listo para GitHub).
        """
        if not isinstance(estado_dict, dict):
            return False
        local_data = {}
        if os.path.exists(self.local_config_path):
            try:
                with open(self.local_config_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        local_data = loaded
            except Exception:
                local_data = {}
        self._deep_update(local_data, estado_dict)
        self._deep_update(self.config_data, estado_dict)
        try:
            with open(self.local_config_path, "w", encoding="utf-8") as f:
                json.dump(local_data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"[ConfigManager] Error guardando config_local.json: {e}")
            return False

