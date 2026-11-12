# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Adquisidor y decodificador bioeléctrico sEMG en tiempo real con
#              visualizador AutoForge, tecnología de desfasaje EMD respecto al mic,
#              mapa latente sin puntos de Lucas y modelo récord de media armónica 87%.
# ==============================================================================

__version__ = "1.2.0"

import os
import sys
import time
import json
import queue
import threading
import numpy as np
from datetime import datetime
from scipy import signal
import scipy.io.wavfile as wavfile

# Asegurar path del proyecto dinámicamente
if getattr(sys, 'frozen', False):
    project_root = os.path.dirname(sys.executable)
    if os.path.basename(project_root) == "_internal":
        project_root = os.path.dirname(project_root)
else:
    _this_file_dir = os.path.dirname(os.path.abspath(__file__))
    _emg_dir = os.path.dirname(_this_file_dir)
    project_root = os.path.dirname(_emg_dir)
    if _emg_dir not in sys.path:
        sys.path.insert(0, _emg_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QSplitter, QFileDialog, QFrame,
    QStackedWidget, QButtonGroup, QScrollArea
)
import pyqtgraph as pg
from pyqtgraph import InfiniteLine

# OpenCV y AntigravityAgent para tracking facial en tiempo real
import cv2
try:
    import sounddevice as sd
    SOUNDDEVICE_DISPONIBLE = True
except Exception:
    SOUNDDEVICE_DISPONIBLE = False

from EMG_desarrollo.acquisition.calibracion_espacial_electrodos import AntigravityAgent

# Visualización 3D interactiva con Matplotlib Qt
import matplotlib
from matplotlib.figure import Figure
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    try:
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    except ImportError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

# Intentar importar nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    NIDAQMX_DISPONIBLE = True
except Exception:
    NIDAQMX_DISPONIBLE = False

# Importar motor de decodificación en tiempo real con fallbacks (prioridad facial de 6 canales)
try:
    from EMG_desarrollo.deep_learning.realtime_facial_decoder_engine import (
        RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA,
        calcular_canales_relativos_invariantes, OneEuroLandmarkFilter3D, ProcrustesKabschHeadFrame
    )
except ImportError:
    try:
        from deep_learning.realtime_facial_decoder_engine import (
            RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA,
            calcular_canales_relativos_invariantes, OneEuroLandmarkFilter3D, ProcrustesKabschHeadFrame
        )
    except ImportError:
        try:
            from realtime_facial_decoder_engine import (
                RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA,
                calcular_canales_relativos_invariantes, OneEuroLandmarkFilter3D, ProcrustesKabschHeadFrame
            )
        except ImportError:
            try:
                from EMG_desarrollo.deep_learning.realtime_decoder_engine import (
                    RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA
                )
                OneEuroLandmarkFilter3D = None
                ProcrustesKabschHeadFrame = None
            except ImportError:
                from realtime_decoder_engine import (
                    RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA
                )
                OneEuroLandmarkFilter3D = None
                ProcrustesKabschHeadFrame = None

try:
    from numba import jit
    NUMBA_DISPONIBLE = True
except ImportError:
    NUMBA_DISPONIBLE = False
    def jit(*args, **kwargs):
        def decorator(f): return f
        return decorator

@jit(nopython=True, nogil=True)
def calculate_rms_envelope(buffer: np.ndarray, window_size: int) -> np.ndarray:
    """Calcula la envolvente RMS sobre un buffer continuo usando una ventana deslizante."""
    n = len(buffer)
    out = np.zeros(n, dtype=np.float64)
    if window_size <= 0:
        return out

    sum_sq = 0.0
    for i in range(min(window_size, n)):
        sum_sq += buffer[i] ** 2
        out[i] = np.sqrt(sum_sq / (i + 1))

    for i in range(window_size, n):
        val_add = buffer[i]
        val_sub = buffer[i - window_size]

        if np.isnan(val_add): val_add = 0.0
        if np.isnan(val_sub): val_sub = 0.0

        sum_sq += val_add ** 2 - val_sub ** 2
        if sum_sq < 0.0 or np.isnan(sum_sq):
            sum_sq = 0.0
        out[i] = np.sqrt(sum_sq / window_size)

    return out

# Pre-compilar Numba
_dummy = calculate_rms_envelope(np.zeros(10, dtype=np.float64), 5)

def decimate_min_max(data_x, data_y, max_points=2000):
    """
    Decimación Dinámica usando Min/Max Peaking (Downsampling).
    Divide los datos en fragmentos y extrae el mínimo y el máximo de cada fragmento.
    Preserva las envolventes de alta frecuencia del sEMG sin aliasing visual.
    """
    n = len(data_y)
    if n <= max_points:
        return data_x, data_y

    num_chunks = max_points // 2
    chunk_size = n // num_chunks
    trunc_length = chunk_size * num_chunks
    y_trunc = data_y[:trunc_length].reshape((num_chunks, chunk_size))
    x_trunc = data_x[:trunc_length].reshape((num_chunks, chunk_size))

    min_indices = np.argmin(y_trunc, axis=1)
    max_indices = np.argmax(y_trunc, axis=1)
    idx_range = np.arange(num_chunks)

    out_x = np.empty(num_chunks * 2, dtype=data_x.dtype)
    out_y = np.empty(num_chunks * 2, dtype=data_y.dtype)

    take_min_first = min_indices <= max_indices
    first_indices = np.where(take_min_first, min_indices, max_indices)
    second_indices = np.where(take_min_first, max_indices, min_indices)

    out_x[0::2] = x_trunc[idx_range, first_indices]
    out_y[0::2] = y_trunc[idx_range, first_indices]
    out_x[1::2] = x_trunc[idx_range, second_indices]
    out_y[1::2] = y_trunc[idx_range, second_indices]

    return out_x, out_y

def resolver_ruta_modelo(rel_path):
    if not rel_path:
        return ""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    
    candidatos = [
        os.path.join(project_root, rel_path),
        os.path.join(project_root, "EMG_desarrollo", rel_path),
        os.path.join(project_root, "modelos", os.path.basename(rel_path)),
        os.path.join(project_root, "EMG_desarrollo", "resultados", os.path.basename(rel_path)),
    ]
    if getattr(sys, 'frozen', False):
        try:
            candidatos.append(os.path.join(sys._MEIPASS, rel_path))
            candidatos.append(os.path.join(sys._MEIPASS, "modelos", os.path.basename(rel_path)))
        except Exception:
            pass
        exe_dir = os.path.dirname(sys.executable)
        candidatos.append(os.path.join(exe_dir, "modelos", os.path.basename(rel_path)))
        candidatos.append(os.path.join(exe_dir, rel_path))

    for c in candidatos:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.join(project_root, rel_path)

# Modelos disponibles en el repositorio
MODELOS_PRESETS = [
    {
        'nombre': "Campeón Facial 6Ch GeoSupConvOrtho 4D (Pose-Ortogonal + Margen Focal) - 97.2% (243/250)",
        'subtitulo': "Espacio 4D Ortogonal a Yaw/Pitch/Roll (Corr=0.000): Desacople /E/ (87.0%) vs /O/ (82.6%) + Centroide Reposo '---'",
        'path': resolver_ruta_modelo("modelos/campeon_geosup_conv_ortho_4d.pt")
    },
    {
        'nombre': "Campeón Facial 6Ch GeoSupConvOrtho 3D (Pose-Ortogonal + Margen Focal) - 95.6% (239/250)",
        'subtitulo': "Espacio 3D Ortogonal a Postura Craneal: /A/ 82.6%, /E/ 78.3%, /I/ 71.7%, /O/ 82.6%, /U/ 78.3%",
        'path': resolver_ruta_modelo("modelos/campeon_geosup_conv_ortho_3d.pt")
    },
    {
        'nombre': "Campeón Facial 6Ch GeoSupConvOrtho 2D (Pose-Ortogonal + Margen Focal) - 86.4% (216/250)",
        'subtitulo': "Fronteras Lineales 2D Ortogonales a Postura Craneal + Centroide de Silencio/Reposo",
        'path': resolver_ruta_modelo("modelos/campeon_geosup_conv_ortho_2d.pt")
    }
]

NOMBRES_CANALES = [
    "CH0: Apertura Mandibular (Digástrico)",
    "CH1: Tracción Sonrisa (Cigomático Mayor)",
    "CH2: Constricción Esfínter (Orbicular)",
    "CH3: Retracción Bucal (Buccinador/Risorio)",
    "CH4: Ascenso Malar (Cigomático Menor)",
    "CH5: Compresión Maseterina (Masetero)",
    "CH6: Micrófono (Master Clock Acústico)"
]
COLORES_CANALES = ["#FF5555", "#39FF14", "#FFAA00", "#00E5FF", "#D884FF", "#FFFF00", "#00FFFF"]


# ==============================================================================
# 1. HILOS DE ADQUISICIÓN Y REPRODUCCIÓN (CÁMARA + MIC / CSV / SIM)
# ==============================================================================
class WebcamAudioLiveThread(QtCore.QThread):
    """
    Hilo de adquisición en tiempo real mediante Cámara Web (OpenCV) + Micrófono (SoundDevice).
    - Extrae la malla facial 3D fiduciaria mediante AntigravityAgent a ~30 FPS.
    - Calcula en milímetros reales los 6 canales cinemáticos articulatorios (Apertura, Sonrisa,
      Constricción, Retracción, Malar y Masetero) normalizados por la Distancia Interpupilar (DIP).
    - Captura el audio del micrófono a 44.1 kHz, calcula la envolvente RMS (90 ms) como Canal 6.
    - Interpola/emite paquetes continuos a 100 Hz directamente a la cola del decodificador.
    - Emite fotogramas anotados para visualización en vivo con bounding box y marcas anatómicas.
    """
    frame_ready = QtCore.Signal(QtGui.QImage, bool, float)  # frame, face_ok, mic_level

    def __init__(self, data_queue, stop_event, sample_rate=100, camera_index=0, audio_device=1, baseline_stats=None, enable_mic=False):
        super().__init__()
        self.data_queue = data_queue
        self.stop_event = stop_event
        self.sample_rate = sample_rate
        self.camera_index = camera_index
        self.audio_device = audio_device
        self.enable_mic = bool(enable_mic)
        # NUNCA heredar un baseline rígido ajeno al iniciar cámara en vivo; calibrar siempre sobre el usuario actual
        self.baseline_stats = {}
        self.agent = None
        self._calibrating = True
        self._calib_frames = []
        self._calib_count_target = 35  # ~1.2 segundos para asimilar la cara real del usuario
        self._dist_history = []
        self._smooth_win = 3
        self._ema_c_state = np.zeros(6, dtype=np.float32)
        self._one_euro_2d = OneEuroLandmarkFilter3D(f_min=0.85, beta=0.055, f_d=1.0)
        self._one_euro_3d = OneEuroLandmarkFilter3D(f_min=0.85, beta=0.055, f_d=1.0)
        self._procrustes_hf = ProcrustesKabschHeadFrame()

    def set_mic_enabled(self, enabled: bool):
        """Habilita o desconecta totalmente el micrófono en vivo (Tarea E.1)."""
        self.enable_mic = bool(enabled)

    def forzar_recalibracion_facial(self):
        """Reinicia la calibración del reposo facial y los filtros Procrustes / One-Euro."""
        self._calib_frames = []
        self._dist_history = []
        self._ema_c_state = np.zeros(6, dtype=np.float32)
        if hasattr(self, '_one_euro_2d'):
            self._one_euro_2d.reset()
            self._one_euro_3d.reset()
            self._procrustes_hf.reset()
        self.baseline_stats = {}
        self._calibrating = True

    def run(self):
        print(f"[CAM+MIC LIVE] Iniciando captura facial (Cam={self.camera_index}, Audio={self.audio_device})...")
        try:
            self.agent = AntigravityAgent(refine_landmarks=True)
        except Exception as e:
            print(f"[CAM LIVE ERROR] Error inicializando AntigravityAgent: {e}")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[CAM LIVE ERROR] No se pudo abrir la cámara índice {self.camera_index}.")
            return
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 60.0)
        except Exception:
            pass

        # Configurar micrófono con sounddevice SOLO si está explícitamente habilitado (Tarea E.1)
        audio_stream = None
        current_mic_rms = [0.0]
        if SOUNDDEVICE_DISPONIBLE and self.enable_mic:
            try:
                def audio_callback(indata, frames, time_info, status):
                    if not self.enable_mic:
                        current_mic_rms[0] = 0.0
                        return
                    if indata is not None and len(indata) > 0:
                        rms_val = float(np.sqrt(np.mean(indata**2)))
                        current_mic_rms[0] = rms_val

                # Probar dispositivo configurado o por defecto
                dev_to_use = self.audio_device
                try:
                    audio_stream = sd.InputStream(
                        samplerate=44100,
                        channels=1,
                        device=dev_to_use,
                        blocksize=2048,
                        callback=audio_callback
                    )
                    audio_stream.start()
                    print(f"[AUDIO LIVE] Stream de micrófono iniciado en dispositivo {dev_to_use}.")
                except Exception as _dev_e:
                    print(f"[AUDIO LIVE] Aviso: Falló dispositivo {dev_to_use}, probando default: {_dev_e}")
                    audio_stream = sd.InputStream(
                        samplerate=44100,
                        channels=1,
                        blocksize=2048,
                        callback=audio_callback
                    )
                    audio_stream.start()
                    print("[AUDIO LIVE] Stream de micrófono iniciado con dispositivo default.")
            except Exception as e_sd:
                print(f"[AUDIO LIVE] Advertencia: No se pudo abrir micrófono: {e_sd}")
                audio_stream = None

        t0_ref = time.perf_counter()
        t_last_sample = t0_ref
        dt_target = 1.0 / self.sample_rate

        # Buffer temporal de canales cinemáticos
        c_state = np.zeros(6, dtype=np.float32)

        def dist_2d(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        try:
            while not self.stop_event.is_set():
                t_loop_start = time.perf_counter()
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                face_ok = False
                res = None
                try:
                    res = self.agent.extraer_malla_facial_3d(frame)
                except Exception:
                    res = None

                h_f, w_f, _ = frame.shape

                if res is not None and res.get('puntos_2d_px') is not None:
                    face_ok = True
                    t_frame_now = time.perf_counter()
                    p2d_raw = res['puntos_2d_px']
                    p3d_raw = res.get('puntos_3d')
                    # Estabilizar coordenadas crudas con One-Euro Filter ANTES de proyectar y dibujar
                    p2d = self._one_euro_2d.filter(p2d_raw, t_now=t_frame_now)
                    p3d = self._one_euro_3d.filter(p3d_raw, t_now=t_frame_now) if p3d_raw is not None else None
                    info_hf = self._procrustes_hf.project(p2d, p3d, d_ip_nominal_mm=63.0, ancho_img=float(w_f), t_now=t_frame_now)
                    r_loc = info_hf['r_local_mm']
                    escala = info_hf['escala_mm_px']
                    prot_labial_mm = info_hf.get('protrusion_labial_mm', 0.0)

                    # Puntos articulatorios anatómicos
                    p_sup = p2d[13]
                    p_inf = p2d[14]
                    p_menton = p2d[152] if len(p2d) > 152 else p2d[14]
                    p_cizq = p2d[61]
                    p_cder = p2d[291]
                    p_mej_izq = p2d[207]
                    p_mej_der = p2d[427]
                    p_mal_izq = p2d[123]
                    p_mal_der = p2d[352]
                    p_mas_izq = p2d[58]
                    p_mas_der = p2d[288]

                    # Proyección local Procrustes Kabsch ya calculada en r_loc (invariante 3D completa)
                    if r_loc is not None and len(r_loc) > 427:
                        def _d3(ia, ib):
                            return float(np.linalg.norm(r_loc[ia] - r_loc[ib]))
                        # Apertura mandibular verdadera: distancia vertical local al puente nasal (eje Y craneal)
                        # más separación labial euclídea 3D (13 a 14)
                        ap_labial = _d3(13, 14)
                        # Distancia vertical anatómica del mentón respecto a la cresta nasal (r_loc[:, 1] apunta hacia arriba)
                        dist_vert_menton = float(r_loc[1, 1] - r_loc[152, 1]) if len(r_loc) > 152 else ap_labial
                        
                        # Almacenar o referenciar la línea base del mentón durante calibración
                        base_menton_ref = getattr(self, '_base_menton_ref', None)
                        if self._calibrating or base_menton_ref is None:
                            if not hasattr(self, '_calib_menton_list'):
                                self._calib_menton_list = []
                            self._calib_menton_list.append(dist_vert_menton)
                            excursion_menton = 0.0
                        else:
                            excursion_menton = max(0.0, dist_vert_menton - base_menton_ref)

                        # Ponderación fisiológica de apertura: 60% separación labial + 40% excursión mentoniana
                        ap_raw = 0.60 * ap_labial + 0.40 * excursion_menton
                        com_raw = _d3(61, 291)
                        mej_raw = 0.5 * (_d3(61, 207) + _d3(291, 427))
                        mal_raw = 0.5 * (_d3(61, 123) + _d3(291, 352))
                        mas_raw = 0.5 * (_d3(61, 58) + _d3(291, 288))
                    else:
                        ap_raw = dist_2d(p_sup, p_inf) * escala
                        com_raw = dist_2d(p_cizq, p_cder) * escala
                        mej_raw = 0.5 * (dist_2d(p_cizq, p_mej_izq) + dist_2d(p_cder, p_mej_der)) * escala
                        mal_raw = 0.5 * (dist_2d(p_cizq, p_mal_izq) + dist_2d(p_cder, p_mal_der)) * escala
                        mas_raw = 0.5 * (dist_2d(p_cizq, p_mas_izq) + dist_2d(p_cder, p_mas_der)) * escala

                    # Tarea B: Filtro causal ligero EMA (alpha = 0.72 >= 0.65) para capturar transitorios rápidos (60-100 ms)
                    cur_vec = np.array([ap_raw, com_raw, mej_raw, mal_raw, mas_raw], dtype=np.float64)
                    if not self._dist_history:
                        self._dist_history = [cur_vec]
                    else:
                        self._dist_history[0] = 0.72 * cur_vec + 0.28 * self._dist_history[0]
                    ap_mm, com_mm, d_mej, d_mal, d_mas = self._dist_history[0]

                    # Calibración dinámica adaptativa del reposo del usuario actual (sin heredar baselines ajenos)
                    if self._calibrating:
                        self._calib_frames.append([ap_mm, com_mm, d_mej, d_mal, d_mas])
                        if len(self._calib_frames) >= self._calib_count_target:
                            arr_c = np.array(self._calib_frames)
                            if hasattr(self, '_calib_menton_list') and self._calib_menton_list:
                                self._base_menton_ref = float(np.percentile(self._calib_menton_list, 20))
                                self._calib_menton_list = []
                            self.baseline_stats = {
                                'base_ap': float(np.percentile(arr_c[:, 0], 25)),
                                'base_com': float(np.median(arr_c[:, 1])),
                                'base_mej': float(np.median(arr_c[:, 2])),
                                'base_mal': float(np.median(arr_c[:, 3])),
                                'base_mas': float(np.median(arr_c[:, 4])),
                                'escala': float(escala)
                            }
                            self._calibrating = False
                            print(f"[CALIB FACIAL OK] Reposo calibrado (invariante intersujeto): ap={self.baseline_stats['base_ap']:.2f}mm, com={self.baseline_stats['base_com']:.2f}mm, menton_ref={getattr(self, '_base_menton_ref', 0.0):.2f}mm")

                    # Si aún está calibrando el reposo, las desviaciones se fijan estrictamente en cero
                    if self._calibrating or not self.baseline_stats:
                        c_state = np.zeros(6, dtype=np.float32)
                        self._ema_c_state = np.zeros(6, dtype=np.float32)
                    else:
                        # Normalización intersujeto por ratios relativos al reposo (\Delta d / d_{reposo})
                        # con banda muerta (deadband) anti-micro-oscilaciones
                        c_raw_inv = calcular_canales_relativos_invariantes(
                            ap_mm, com_mm, d_mej, d_mal, d_mas,
                            baseline_stats=self.baseline_stats,
                            aplicar_deadband=True,
                            devolver_en_mm_canonicos=True,
                            protrusion_labial_mm=prot_labial_mm
                        )
                        # Filtro EMA adicional para suavidad de la trayectoria latente
                        self._ema_c_state = 0.65 * c_raw_inv + 0.35 * self._ema_c_state
                        c_state = np.where(self._ema_c_state < 0.15, 0.0, self._ema_c_state).astype(np.float32)

                        # Adaptación lenta del reposo si el usuario permanece en reposo absoluto
                        if float(np.max(c_raw_inv)) == 0.0:
                            alpha_rest = 0.015
                            self.baseline_stats['base_ap'] = (1.0 - alpha_rest) * self.baseline_stats['base_ap'] + alpha_rest * float(ap_mm)
                            self.baseline_stats['base_com'] = (1.0 - alpha_rest) * self.baseline_stats['base_com'] + alpha_rest * float(com_mm)
                            self.baseline_stats['base_mej'] = (1.0 - alpha_rest) * self.baseline_stats['base_mej'] + alpha_rest * float(d_mej)
                            self.baseline_stats['base_mal'] = (1.0 - alpha_rest) * self.baseline_stats['base_mal'] + alpha_rest * float(d_mal)
                            self.baseline_stats['base_mas'] = (1.0 - alpha_rest) * self.baseline_stats['base_mas'] + alpha_rest * float(d_mas)

                    # Dibujar landmarks articulatorios, mentón (152) y puntos óseos del Head Frame (1, 10, 33, 263, 234, 454)
                    pts_dibujo = [
                        (p_sup, (0, 0, 255), "13"), (p_inf, (0, 0, 255), "14"),
                        (p_menton, (0, 255, 255), "152 (Mentón)"),
                        (p_cizq, (0, 255, 0), "61"), (p_cder, (0, 255, 0), "291"),
                        (p_mej_izq, (255, 200, 0), "207"), (p_mej_der, (255, 200, 0), "427"),
                        (p_mal_izq, (255, 0, 255), "123"), (p_mal_der, (255, 0, 255), "352"),
                        (p_mas_izq, (0, 255, 255), "58"), (p_mas_der, (0, 255, 255), "288"),
                        (p2d[1], (0, 255, 255), "1"), (p2d[10], (255, 255, 0), "10"),
                        (p2d[33], (255, 200, 0), "33"), (p2d[263], (255, 200, 0), "263"),
                        (p2d[234], (200, 200, 255), "234"), (p2d[454], (200, 200, 255), "454"),
                    ]
                    for pt, col, _ in pts_dibujo:
                        cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, col, -1)

                    # Líneas articulatorias (apertura labial, mentón y comisuras)
                    cv2.line(frame, (int(p_sup[0]), int(p_sup[1])), (int(p_inf[0]), int(p_inf[1])), (0, 0, 255), 2)
                    cv2.line(frame, (int(p_inf[0]), int(p_inf[1])), (int(p_menton[0]), int(p_menton[1])), (0, 255, 255), 2)
                    cv2.line(frame, (int(p_cizq[0]), int(p_cizq[1])), (int(p_cder[0]), int(p_cder[1])), (0, 255, 0), 2)
                    cv2.line(frame, (int(p_cizq[0]), int(p_cizq[1])), (int(p_mej_izq[0]), int(p_mej_izq[1])), (255, 200, 0), 1)
                    cv2.line(frame, (int(p_cder[0]), int(p_cder[1])), (int(p_mej_der[0]), int(p_mej_der[1])), (255, 200, 0), 1)

                    # Dibujar cruz del Centro de Masa Óseo Craneal (Origen rígido desacoplado sobre la cresta nasal)
                    # Media ponderada de anclaje óseo en 2D (inmune a la mandíbula y a las comisuras)
                    idx_craneo = [1, 4, 6, 168, 10, 33, 133, 362, 263]
                    w_craneo = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.35, 0.45, 0.45, 0.35], dtype=np.float64)
                    valid_k = [k for k in idx_craneo if k < len(p2d)]
                    if len(valid_k) >= 5:
                        w_sub = w_craneo[:len(valid_k)]
                        w_sub = w_sub / np.sum(w_sub)
                        c_craneo_px = np.sum(p2d[valid_k] * w_sub[:, None], axis=0)
                    else:
                        c_craneo_px = 0.5 * (p2d[33] + p2d[263]) if len(p2d) > 263 else p2d[1]
                    cx_i, cy_i = int(c_craneo_px[0]), int(c_craneo_px[1])
                    cv2.drawMarker(frame, (cx_i, cy_i), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                    cv2.putText(frame, "C_craneo", (cx_i + 8, cy_i - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # Convertir frame a QImage para la interfaz
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_img = QtGui.QImage(frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()

                # Tarea E.1: Si el micrófono está deshabilitado, forzar estrictamente a 0.0
                if not self.enable_mic:
                    current_mic_rms[0] = 0.0
                mic_amp = float(current_mic_rms[0] * 50.0) if self.enable_mic else 0.0
                self.frame_ready.emit(qt_img, face_ok, mic_amp)

                # Generar muestras a 100 Hz y enviar a cola
                t_now = time.perf_counter()
                elapsed_since_sample = t_now - t_last_sample
                n_samples_to_push = max(1, int(round(elapsed_since_sample * self.sample_rate)))
                if n_samples_to_push > 10:
                    n_samples_to_push = 10

                chunk = np.zeros((7, n_samples_to_push), dtype=np.float32)
                for ci in range(6):
                    chunk[ci, :] = c_state[ci]
                chunk[6, :] = mic_amp if self.enable_mic else 0.0

                self.data_queue.put(chunk)
                t_last_sample = t_now

                # Dormir para mantener ~30-40 FPS de cámara sin sofocar CPU
                t_elapsed_loop = time.perf_counter() - t_loop_start
                t_sleep = max(0.003, 0.016 - t_elapsed_loop)
                time.sleep(t_sleep)

        except Exception as e_run:
            print(f"[CAM LIVE RUN ERROR] {e_run}")
        finally:
            cap.release()
            if audio_stream is not None:
                try:
                    audio_stream.stop()
                    audio_stream.close()
                except Exception:
                    pass
            print("[CAM+MIC LIVE] Captura finalizada.")
            if not self.stop_event.is_set():
                self.stop_event.set()


def nidaq_acquisition_thread(device_channels, sample_rate, chunk_samples, num_canales, data_queue, stop_event, terminal_config_val=None):
    """Hilo de adquisición de hardware físico con NI-DAQmx (Modo seguro sEMG)."""
    print(f"[DAQ] Iniciando adquisición física (SR={sample_rate} Hz)...")
    try:
        with nidaqmx.Task() as task:
            for canal in device_channels:
                if terminal_config_val is not None:
                    task.ai_channels.add_ai_voltage_chan(canal, terminal_config=terminal_config_val, min_val=-10.0, max_val=10.0)
                else:
                    task.ai_channels.add_ai_voltage_chan(canal, terminal_config=TerminalConfiguration.RSE, min_val=-10.0, max_val=10.0)

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=sample_rate
            )
            reader = AnalogMultiChannelReader(task.in_stream)
            buffer_daq = np.zeros((num_canales, chunk_samples), dtype=np.float64)
            task.start()
            print("[DAQ] Tarea iniciada.")

            while not stop_event.is_set():
                reader.read_many_sample(
                    buffer_daq,
                    number_of_samples_per_channel=chunk_samples,
                    timeout=(chunk_samples / sample_rate) * 5
                )
                data_queue.put(buffer_daq.copy())

    except Exception as e:
        print(f"[DAQ ERROR] {e}")
    finally:
        if not stop_event.is_set():
            stop_event.set()



def csv_6ch_playback_thread(session_dir, sample_rate, chunk_samples, data_queue, stop_event, ckpt=None):
    """Hilo de reproducción en tiempo real (100 Hz) para sesiones cinemáticas de 6 canales (trayectoria_puntos.csv)."""
    try:
        from deep_learning.decodificador_continuo_facial_6canales import extraer_canales_y_pulsos_sesion
    except ImportError:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deep_learning"))
        from decodificador_continuo_facial_6canales import extraer_canales_y_pulsos_sesion

    datos = extraer_canales_y_pulsos_sesion(session_dir, baseline_stats=None, usar_baseline_local=True, aplicar_deadband=False)
    t_cam = datos['t_cam']
    canales = datos['canales']
    t_wav = datos['t_wav_env']
    env_mic = datos['env_mic_raw']

    t_min, t_max = float(t_cam[0]), float(t_cam[-1])
    t_grid = np.arange(t_min, t_max, 1.0 / sample_rate)
    sig_all = np.zeros((7, len(t_grid)), dtype=np.float32)
    for c in range(6):
        sig_all[c] = np.interp(t_grid, t_cam, canales[c], left=0.0, right=0.0)
    if len(t_wav) > 0:
        df_p_sub = datos.get('df_princ', None)
        off_wav = float(np.median(df_p_sub['t_wav_s'].values - df_p_sub['t_daq_s'].values)) if (df_p_sub is not None and len(df_p_sub) > 0 and 't_wav_s' in df_p_sub.columns) else 4.950
        sig_all[6] = np.interp(t_grid, t_wav - off_wav, env_mic, left=0.0, right=0.0)

    idx = 0
    total_muestras = sig_all.shape[1]
    t_chunk_s = chunk_samples / sample_rate
    while not stop_event.is_set() and idx < total_muestras:
        t0 = time.perf_counter()
        end_i = min(total_muestras, idx + chunk_samples)
        chunk = sig_all[:, idx:end_i]
        idx = end_i
        data_queue.put(chunk)
        el = time.perf_counter() - t0
        if t_chunk_s - el > 0:
            time.sleep(t_chunk_s - el)
    stop_event.set()

def wav_playback_thread(wav_paths, sample_rate, chunk_samples, data_queue, stop_event, loop=False):
    """
    Hilo de reproducción de tomas grabadas (WAVs) a velocidad de tiempo real (2000 Hz).
    Carga los 4 canales completos (3 sEMG + 1 micrófono) para visualización simultánea.
    """
    print(f"[WAV PLAYBACK] Cargando 4 canales desde disco...")
    signals = []
    min_len = float('inf')
    for p in wav_paths:
        if os.path.exists(p):
            _, d = wavfile.read(p)
            signals.append(d.astype(np.float32))
            if len(d) < min_len:
                min_len = len(d)
        else:
            print(f"[WAV PLAYBACK] Archivo no encontrado: {p}")

    if not signals:
        print("[WAV PLAYBACK] Error: no se pudieron cargar las señales.")
        stop_event.set()
        return

    sig_all = np.stack([s[:min_len] for s in signals], axis=0)
    num_canales = sig_all.shape[0]
    total_muestras = sig_all.shape[1]
    print(f"[WAV PLAYBACK] Señal lista: {num_canales} canales, {total_muestras} muestras ({total_muestras/sample_rate:.1f} s).")

    idx = 0
    t_chunk_s = chunk_samples / sample_rate

    while not stop_event.is_set():
        t_start = time.perf_counter()
        if idx + chunk_samples <= total_muestras:
            chunk = sig_all[:, idx:idx+chunk_samples]
            idx += chunk_samples
        else:
            if loop:
                idx = 0
                continue
            else:
                rem = total_muestras - idx
                if rem > 0:
                    chunk = np.zeros((num_canales, chunk_samples), dtype=np.float32)
                    chunk[:, :rem] = sig_all[:, idx:]
                    data_queue.put(chunk)
                print("[WAV PLAYBACK] Fin de la grabación.")
                break

        data_queue.put(chunk)
        elapsed = time.perf_counter() - t_start
        sleep_time = max(0.0, t_chunk_s - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    if not stop_event.is_set():
        stop_event.set()


def simulador_thread(chunk_samples, sample_rate, num_canales, data_queue, stop_event, test_freq=50):
    """Hilo simulador de ondas sintéticas para depuración rápida."""
    t_global = 0.0
    dt = 1.0 / sample_rate
    t_chunk = chunk_samples / sample_rate
    while not stop_event.is_set():
        t0 = time.perf_counter()
        t_vec = t_global + np.arange(chunk_samples) * dt
        chunk = np.zeros((num_canales, chunk_samples), dtype=np.float32)
        for c in range(min(num_canales, 3)):
            ruido = np.random.normal(0, 15.0, chunk_samples)
            burst = np.sin(2 * np.pi * test_freq * t_vec) * 400.0 * (np.sin(2 * np.pi * 0.5 * t_vec) > 0.7)
            chunk[c] = burst + ruido

        # Canal 3: micrófono acústico con retardo fisiológico EMD de 350 ms
        if num_canales >= 4:
            ruido_mic = np.random.normal(0, 5.0, chunk_samples)
            burst_mic = np.sin(2 * np.pi * 220.0 * t_vec) * 1200.0 * (np.sin(2 * np.pi * 0.5 * (t_vec - 0.35)) > 0.75)
            chunk[3] = burst_mic + ruido_mic

        t_global += chunk_samples * dt
        data_queue.put(chunk)
        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, t_chunk - elapsed))


# ==============================================================================
# 2. VENTANA PRINCIPAL DE ADQUISICIÓN Y DECODIFICACIÓN
# ==============================================================================
class RealTimeDecoderPlotter(QtWidgets.QWidget):
    """
    Estación integrada de adquisición sEMG y decodificación bioeléctrica en tiempo real:
    - Panel Izquierdo: Osciloscopio multicanal idéntico al de AutoForge experimental
      (GraphicsLayoutWidget, viz_control_panel, checkboxes CH0-CH3, trigger y SNR).
    - Panel Derecho: Decodificador con tecnología de retardo EMD, display gigante,
      mapa latente 2D sin puntos de Lucas y selector de arquitectura.
    """

    def __init__(self):
        super().__init__()

        # Estilos visuales de botones de control
        self.BTN_START_STYLE = "background-color: #050505; color: #00FF00; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px; border: 2px solid #00FF00; border-radius: 4px;"
        self.BTN_STOP_STYLE = "background-color: #050505; color: #FF0000; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px; border: 2px solid #FF0000; border-radius: 4px;"
        self.BTN_PLAY_STYLE = "background-color: #0A192F; color: #64FFDA; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px; border: 2px solid #64FFDA; border-radius: 4px;"

        self.setWindowTitle(f"Ñandú LSD - Decodificador Bioeléctrico en Tiempo Real v{__version__}")
        self.resize(1560, 920)

        # Estado DAQ
        self.is_acquiring = False
        self.is_playback = False
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.acquisition_thread = None

        self.SAMPLE_RATE = 100
        self.NUM_CANALES = 7  # CH0, CH1, CH2 (sEMG) + CH3 (Micrófono)
        self.PLOT_DURATION_S = 8
        self.PLOT_SAMPLES = self.SAMPLE_RATE * self.PLOT_DURATION_S
        self.plot_buffer = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES), dtype=np.float32)

        # Envolvente RMS en tiempo real (idéntico a AutoForge DAQ)
        self.env_buffer = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES), dtype=np.float64)
        self._held_plot_ymax = 500.0

        # Historial de puntos decodificados en el mapa latente (hasta 150 para cubrir sesiones completas)
        self.historial_puntos = []  # Tuplas (z1, z2, vocal, timestamp)
        self.MAX_HISTORIAL = 150
        self.ultimo_evento_decodificado = None
        self._last_event_sample_span = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

        # Marcador y conteo de aciertos para grabación de prueba
        self.test_ground_truth = []
        self.test_event_count = 0
        self.test_correct_count = 0
        self.test_vocal_stats = {v: {'correct': 0, 'total': 0} for v in VOCALES}

        # Motor de decodificación: buscar primer modelo disponible entre los presets
        pt_model_default = None
        for p in MODELOS_PRESETS:
            if os.path.exists(p['path']):
                pt_model_default = p['path']
                break

        self.engine = RealtimeDecoderEngine(
            model_path=pt_model_default,
            sample_rate=self.SAMPLE_RATE,
            emd_delay_ms=71, trigger_mode='GATE_HYBRID'
        )

        # Sesión WAV de prueba por defecto (P5)
        self.ruta_sesion_wav = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sesiones_cortadas/medicion_2_test_tiempo_real_pulsos_131_200"
        )

        # Construir interfaz
        self._init_ui()

        # Cargar metadata de la sesión inicial si existe
        self._cargar_metadata_sesion(self.ruta_sesion_wav)

        # Timer de actualización continua GUI (30 ms ~ 33 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.actualizar_ciclo)
        self.timer.start(30)

        # Timer para ocultar el overlay flotante
        self.overlay_timer = QtCore.QTimer()
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.timeout.connect(self._ocultar_overlay)

        # Cargar mapa de decisión 2D
        self._cargar_mapa_fronteras_decision()

    def _init_ui(self):
        """Construye la distribución general de la ventana con soporte responsivo y scrollable."""
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(4)

        # Splitter horizontal: Panel DAQ (Izquierda) vs Panel Decodificador (Derecha)
        self.splitter_principal = QSplitter(QtCore.Qt.Horizontal)

        # 1. Panel DAQ (Cámara + Osciloscopio)
        self.widget_daq = QWidget()
        self.layout_daq = QVBoxLayout(self.widget_daq)
        self.layout_daq.setContentsMargins(4, 4, 4, 4)
        self.layout_daq.setSpacing(4)
        self._build_daq_panel()

        # 2. Panel Decodificador en Tiempo Real
        self.widget_decodificador = QWidget()
        self.layout_decodificador = QVBoxLayout(self.widget_decodificador)
        self.layout_decodificador.setContentsMargins(4, 4, 4, 4)
        self.layout_decodificador.setSpacing(4)
        self._build_decoder_panel()

        # Envolver el decodificador en un QScrollArea para pantallas pequeñas
        self.scroll_decodificador = QScrollArea()
        self.scroll_decodificador.setWidgetResizable(True)
        self.scroll_decodificador.setFrameShape(QFrame.NoFrame)
        self.scroll_decodificador.setStyleSheet("background: transparent; border: none;")
        self.scroll_decodificador.setWidget(self.widget_decodificador)

        # Envolver también el panel DAQ en un QScrollArea para que nunca empuje fuera de pantalla
        self.scroll_daq = QScrollArea()
        self.scroll_daq.setWidgetResizable(True)
        self.scroll_daq.setFrameShape(QFrame.NoFrame)
        self.scroll_daq.setStyleSheet("background: transparent; border: none;")
        self.scroll_daq.setWidget(self.widget_daq)

        # Agregar al splitter con reparto equitativo
        self.splitter_principal.addWidget(self.scroll_daq)
        self.splitter_principal.addWidget(self.scroll_decodificador)
        self.splitter_principal.setStretchFactor(0, 1)
        self.splitter_principal.setStretchFactor(1, 1)
        self.splitter_principal.setSizes([960, 960])

        self.main_layout.addWidget(self.splitter_principal)

    def _build_daq_panel(self):
        """Construye el panel de cámara en vivo, osciloscopio cinemático y micrófono."""
        # Grupo de Adquisición y Cámara Facial
        grp_ctrl = QGroupBox("Adquisición Facial y Control Biocinemático")
        lay_ctrl = QGridLayout(grp_ctrl)
        lay_ctrl.setContentsMargins(6, 6, 6, 6)
        lay_ctrl.setSpacing(6)

        self.btn_iniciar_daq = QPushButton("INICIAR CÁMARA FACIAL EN VIVO")
        self.btn_iniciar_daq.setStyleSheet(self.BTN_START_STYLE)
        self.btn_iniciar_daq.clicked.connect(self.toggle_daq)

        self.btn_play_wav = QPushButton("Reproducir Medición CSV")
        self.btn_play_wav.setStyleSheet(self.BTN_PLAY_STYLE)
        self.btn_play_wav.clicked.connect(self.toggle_wav_playback)

        self.btn_cargar_wav = QPushButton("Cargar Sesión CSV...")
        self.btn_cargar_wav.clicked.connect(self.seleccionar_sesion_wav)

        self.lbl_sesion_actual = QLabel(f"Sesión: {os.path.basename(self.ruta_sesion_wav)}")
        self.lbl_sesion_actual.setStyleSheet("color: #AAAAAA; font-size: 11px;")

        self.chk_modo_simulado = QCheckBox("Modo Señal Sintética")

        self.btn_recalibrar_reposo = QPushButton("Grabar Ruido / Calibrar Reposo (3s)")
        self.btn_recalibrar_reposo.setStyleSheet("""
            QPushButton {
                background-color: #2B3A42; color: #66FCF1; font-weight: bold; font-size: 11px; padding: 5px; border-radius: 4px; border: 1px solid #45A29E;
            }
            QPushButton:hover {
                background-color: #3A4F59; color: #FFFFFF; border-color: #66FCF1;
            }
            QPushButton:pressed {
                background-color: #00ADB5; color: #0B0C10; font-weight: 900;
            }
        """)
        self.btn_recalibrar_reposo.clicked.connect(self._forzar_recalibracion_ruido)

        lay_ctrl.addWidget(self.btn_iniciar_daq, 0, 0)
        lay_ctrl.addWidget(self.btn_play_wav, 0, 1)
        lay_ctrl.addWidget(self.btn_cargar_wav, 0, 2)
        lay_ctrl.addWidget(self.btn_recalibrar_reposo, 1, 0)
        lay_ctrl.addWidget(self.chk_modo_simulado, 1, 1)
        lay_ctrl.addWidget(self.lbl_sesion_actual, 1, 2)

        # Panel de visualización de video y telemetría de detección
        box_cam = QWidget()
        lay_cam = QHBoxLayout(box_cam)
        lay_cam.setContentsMargins(0, 2, 0, 2)
        lay_cam.setSpacing(8)

        # Vista de video webcam
        self.lbl_video_preview = QLabel("CÁMARA DETENIDA")
        self.lbl_video_preview.setFixedSize(300, 210)
        self.lbl_video_preview.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_video_preview.setStyleSheet(
            "background-color: #050505; color: #66FCF1; border: 2px dashed #333333; font-family: monospace; font-size: 11px; font-weight: bold; border-radius: 4px;"
        )
        lay_cam.addWidget(self.lbl_video_preview)

        # Badges de estado de rostro y nivel de micrófono
        box_telemetria = QWidget()
        lay_telemetria = QVBoxLayout(box_telemetria)
        lay_telemetria.setContentsMargins(0, 0, 0, 0)
        lay_telemetria.setSpacing(6)

        self.lbl_badge_rostro = QLabel("ROSTRO: ESPERANDO CÁMARA")
        self.lbl_badge_rostro.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_badge_rostro.setFixedHeight(28)
        self.lbl_badge_rostro.setStyleSheet(
            "background-color: #1F2833; color: #FFAA00; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px;"
        )
        lay_telemetria.addWidget(self.lbl_badge_rostro)

        self.chk_habilitar_microfono = QtWidgets.QCheckBox("Habilitar Micrófono (Audio)")
        self.chk_habilitar_microfono.setChecked(False)
        self.chk_habilitar_microfono.setStyleSheet("color: #FFCC00; font-size: 11px; font-weight: bold;")
        self.chk_habilitar_microfono.setToolTip(
            "Tarea E.1: En modo Gate Cinemático el micrófono permanece desconectado (0.0) para evitar disparos por ruido acústico."
        )
        self.chk_habilitar_microfono.toggled.connect(self._on_toggle_microfono)
        lay_telemetria.addWidget(self.chk_habilitar_microfono)

        lbl_mic_title = QLabel("Nivel Micrófono (RMS):")
        lbl_mic_title.setStyleSheet("color: #66FCF1; font-size: 11px; font-weight: bold;")
        lay_telemetria.addWidget(lbl_mic_title)

        self.bar_mic_level = QProgressBar()
        self.bar_mic_level.setRange(0, 100)
        self.bar_mic_level.setValue(0)
        self.bar_mic_level.setFixedHeight(16)
        self.bar_mic_level.setTextVisible(False)
        self.bar_mic_level.setStyleSheet(
            "QProgressBar { background-color: #050505; border: 1px solid #444; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #00FFFF; border-radius: 2px; }"
        )
        lay_telemetria.addWidget(self.bar_mic_level)

        self.lbl_mic_val = QLabel("0.0 µV / Energía Acústica")
        self.lbl_mic_val.setStyleSheet("color: #888888; font-size: 10px; font-family: monospace;")
        lay_telemetria.addWidget(self.lbl_mic_val)
        lay_telemetria.addStretch()

        lay_cam.addWidget(box_telemetria)
        lay_ctrl.addWidget(box_cam, 2, 0, 1, 3)

        self.layout_daq.addWidget(grp_ctrl)

        # Contenedor del Osciloscopio con la barra superior AutoForge
        self.plot_container = QtWidgets.QWidget()
        self.plot_container_layout = QtWidgets.QVBoxLayout(self.plot_container)
        self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_container_layout.setSpacing(2)

        # Panel de control de visualización superior (AutoForge viz_control_panel)
        self.viz_control_panel = QtWidgets.QWidget()
        self.viz_control_panel.setStyleSheet("background-color: #050505; border-bottom: 1px solid #333;")
        self.viz_control_layout = QtWidgets.QHBoxLayout(self.viz_control_panel)
        self.viz_control_layout.setContentsMargins(5, 3, 5, 3)

        lbl_viz = QtWidgets.QLabel("Visualización de Canales:")
        lbl_viz.setStyleSheet("color: #00FFFF; font-weight: bold; border: none; font-size: 11px;")
        self.viz_control_layout.addWidget(lbl_viz)

        self.btn_hide_all_viz = QtWidgets.QPushButton("Ocultar Todos")
        self.btn_hide_all_viz.setStyleSheet(
            "background-color: #330000; color: #ff5555; border: 1px solid #ff5555; padding: 2px 8px; font-size: 11px; font-weight: bold; border-radius: 3px;"
        )
        self.btn_hide_all_viz.clicked.connect(self.hide_all_viz_channels)
        self.viz_control_layout.addWidget(self.btn_hide_all_viz)

        # Control de Envolvente RMS en tiempo real (idéntico a AutoForge DAQ)
        self.chk_rms_env = QtWidgets.QCheckBox("Envolvente RMS (Realtime)")
        self.chk_rms_env.setChecked(True)
        self.chk_rms_env.setStyleSheet("color: #00FFCC; font-weight: bold; border: none; font-size: 11px;")
        self.chk_rms_env.setToolTip("Calcula y grafica la envolvente RMS en tiempo real idéntico a AutoForge DAQ.")
        self.viz_control_layout.addWidget(self.chk_rms_env)

        # Control de Auto-escala dinámico del osciloscopio
        self.chk_autoescala = QtWidgets.QCheckBox("Auto-escala")
        self.chk_autoescala.setChecked(True)
        self.chk_autoescala.setStyleSheet("color: #66FCF1; font-weight: bold; border: none; font-size: 11px;")
        self.chk_autoescala.setToolTip("Ajusta dinámicamente la escala del osciloscopio al pico máximo de las señales visibles sin techos artificiales.")
        self.viz_control_layout.addWidget(self.chk_autoescala)

        lbl_win = QtWidgets.QLabel("Ventana RMS:")
        lbl_win.setStyleSheet("color: #C5C6C7; font-weight: bold; border: none; font-size: 11px;")
        self.viz_control_layout.addWidget(lbl_win)

        self.spin_rms_window = QSpinBox()
        self.spin_rms_window.setRange(10, 1000)
        self.spin_rms_window.setValue(90)
        self.spin_rms_window.setSingleStep(10)
        self.spin_rms_window.setSuffix(" ms")
        self.spin_rms_window.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 2px;")
        self.spin_rms_window.setToolTip("Tamaño de ventana de integración en milisegundos (90 ms por defecto).")
        self.spin_rms_window.valueChanged.connect(self._on_rms_window_changed)
        self.viz_control_layout.addWidget(self.spin_rms_window)

        # Contenedor de checkboxes con scroll horizontal para no desbordar el ancho de pantalla
        self.scroll_checkboxes = QScrollArea()
        self.scroll_checkboxes.setWidgetResizable(True)
        self.scroll_checkboxes.setFixedHeight(30)
        self.scroll_checkboxes.setFrameShape(QFrame.NoFrame)
        self.scroll_checkboxes.setStyleSheet("background: transparent; border: none;")
        self.widget_checkboxes = QWidget()
        self.viz_checkbox_layout = QtWidgets.QHBoxLayout(self.widget_checkboxes)
        self.viz_checkbox_layout.setContentsMargins(0, 0, 0, 0)
        self.viz_checkbox_layout.setSpacing(8)
        self.scroll_checkboxes.setWidget(self.widget_checkboxes)
        self.viz_control_layout.addWidget(self.scroll_checkboxes)

        self.viz_checkboxes = []

        self.plot_container_layout.addWidget(self.viz_control_panel)

        # GraphicsLayoutWidget de AutoForge
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('k')  # Fondo Negro AutoForge
        self.plot_widget.ci.layout.setContentsMargins(10, 10, 20, 10)

        # Overlay Flotante AutoForge
        self.autoforge_overlay = QtWidgets.QLabel(self.plot_widget)
        self.autoforge_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.autoforge_overlay.setStyleSheet(
            "background-color: rgba(10, 5, 20, 190); color: #FF0055; font-family: 'Courier New', monospace; "
            "font-size: 55px; font-weight: 900; border: 3px solid #00FFFF; border-radius: 8px; padding: 10px;"
        )
        self.autoforge_overlay.hide()
        self.plot_widget.installEventFilter(self)

        self.plot_container_layout.addWidget(self.plot_widget)

        # Plot principal de adquisición
        self.plot = self.plot_widget.addPlot(title="Canales de Adquisición - Señales sEMG y Micrófono")
        self.plot.setLabel('bottom', "Tiempo (s)")
        self.plot.setLabel('left', "Amplitud (µV)")
        self.plot.getAxis('left').setWidth(60)
        self.plot.addLegend(offset=(10, 10))
        self.plot.showGrid(x=True, y=True, alpha=0.65)
        self.plot.setClipToView(True)
        self.plot.setDownsampling(auto=True, mode='peak')
        self.plot.autoBtn.setVisible(True)
        self.plot.setYRange(-2000, 2000)

        # Línea de Trigger AutoForge (deslizable, roja punteada)
        self.trigger_line = InfiniteLine(
            pos=1000.0, angle=0, movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.trigger_line)

        # Líneas de umbral SNR AutoForge (cian punteadas)
        self.peak_th_line_pos = InfiniteLine(
            pos=200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine)
        )
        self.peak_th_line_neg = InfiniteLine(
            pos=-200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.peak_th_line_pos)
        self.plot.addItem(self.peak_th_line_neg)

        # Sombreado vertical translúcido (LinearRegionItem) para auditar la ventana exacta [t_inicio, t_fin]
        self.region_ventana = pg.LinearRegionItem(
            values=(-10.0, -9.5),
            orientation='vertical',
            brush=pg.mkBrush(0, 255, 255, 48),
            pen=pg.mkPen('#00FFFF', width=2.0),
            movable=False
        )
        self.plot.addItem(self.region_ventana)

        # Marcadores verticales de Onset y Centro EMD proyectado
        self.marker_onset = InfiniteLine(
            pos=-10.0, angle=90, movable=False, pen=pg.mkPen('#FF8800', width=2, style=QtCore.Qt.DotLine)
        )
        self.marker_centro = InfiniteLine(
            pos=-10.0, angle=90, movable=False, pen=pg.mkPen('#00FFFF', width=2, style=QtCore.Qt.DashLine)
        )
        self.plot.addItem(self.marker_onset)
        self.plot.addItem(self.marker_centro)

        # Crear curvas y checkboxes para los 4 canales
        self.curvas_emg = []
        for i in range(self.NUM_CANALES):
            nombre = NOMBRES_CANALES[i]
            col = COLORES_CANALES[i]

            curva = self.plot.plot(pen=pg.mkPen(col, width=1.5), name=nombre)
            curva.setVisible(i != 6)
            self.curvas_emg.append(curva)

            chk = QtWidgets.QCheckBox(nombre)
            chk.setChecked(i != 6)
            chk.setStyleSheet(f"color: {col}; font-weight: bold; border: none; font-size: 11px;")

            def toggle_viz(state, idx=i):
                if idx < len(self.curvas_emg):
                    self.curvas_emg[idx].setVisible(state)

            chk.toggled.connect(toggle_viz)
            self.viz_checkbox_layout.addWidget(chk)
            self.viz_checkboxes.append(chk)

        # Curva 5: Norma Tricanal Combinada S_emg (Gate Doble)
        self.curva_semg_norma = self.plot.plot(pen=pg.mkPen('#FF00FF', width=2.6), name="S_emg: Norma")
        self.curvas_emg.append(self.curva_semg_norma)

        chk_semg = QtWidgets.QCheckBox("S_emg: Norma")
        chk_semg.setChecked(True)
        chk_semg.setStyleSheet("color: #FF00FF; font-weight: bold; border: none; font-size: 11px;")
        def toggle_semg(state):
            self.curva_semg_norma.setVisible(state)
        chk_semg.toggled.connect(toggle_semg)
        self.viz_checkbox_layout.addWidget(chk_semg)
        self.viz_checkboxes.append(chk_semg)

        self.layout_daq.addWidget(self.plot_container)

    def _build_decoder_panel(self):
        """Construye el panel de decodificación bioeléctrica, cartel y mapa latente."""
        # 1. Selector de Modelo PyTorch / Arquitectura
        grp_modelo = QGroupBox("Modelo PyTorch y Arquitectura Latente")
        lay_modelo = QVBoxLayout(grp_modelo)
        lay_modelo.setContentsMargins(6, 6, 6, 6)
        lay_modelo.setSpacing(4)

        self.cmb_modelo = QComboBox()
        for preset in MODELOS_PRESETS:
            self.cmb_modelo.addItem(preset['nombre'], preset['path'])
        self.cmb_modelo.currentIndexChanged.connect(self._on_model_changed)
        lay_modelo.addWidget(self.cmb_modelo)

        self.btn_cargar_modelo_custom = QPushButton("📂 Cargar Modelo Propio (.pt)...")
        self.btn_cargar_modelo_custom.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_cargar_modelo_custom.setStyleSheet("""
            QPushButton {
                background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E;
                border-radius: 4px; padding: 4px 8px; font-weight: bold; font-size: 11px;
            }
            QPushButton:hover { background-color: #45A29E; color: #0B0C10; }
        """)
        self.btn_cargar_modelo_custom.clicked.connect(self._cargar_modelo_personalizado_dialogo)
        lay_modelo.addWidget(self.btn_cargar_modelo_custom)

        if self.engine.model is not None:
            self.lbl_modelo_info = QLabel(MODELOS_PRESETS[0]['subtitulo'])
            self.lbl_modelo_info.setStyleSheet("color: #66FCF1; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_modelo_info = QLabel("⚠ Sin modelo cargado. Utilice 'Cargar Modelo Propio (.pt)' para decodificar.")
            self.lbl_modelo_info.setStyleSheet("color: #FFB703; font-size: 11px; font-weight: bold;")
        lay_modelo.addWidget(self.lbl_modelo_info)

        self.layout_decodificador.addWidget(grp_modelo)

        # 2. Cartel Gigante de Fonema Decodificado
        self.grp_vocal = QGroupBox("Fonema Decodificado en Tiempo Real")
        lay_vocal = QVBoxLayout(self.grp_vocal)
        lay_vocal.setContentsMargins(6, 6, 6, 6)
        lay_vocal.setSpacing(6)

        self.lbl_vocal_gigante = QLabel("---")
        self.lbl_vocal_gigante.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_vocal_gigante.setStyleSheet("""
            font-size: 78px;
            font-weight: 900;
            color: #66FCF1;
            background-color: #0B0C10;
            border: 3px solid #66FCF1;
            border-radius: 10px;
            padding: 8px;
        """)
        lay_vocal.addWidget(self.lbl_vocal_gigante)

        # Barra y porcentaje de confianza
        self.lbl_confianza = QLabel("Confianza: 0.0% | Latencia de Inferencia: 0.00 ms")
        self.lbl_confianza.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_confianza.setStyleSheet("color: #E0E0E0; font-weight: bold; font-size: 12px;")
        lay_vocal.addWidget(self.lbl_confianza)

        self.bar_confianza = QProgressBar()
        self.bar_confianza.setRange(0, 100)
        self.bar_confianza.setValue(0)
        self.bar_confianza.setTextVisible(True)
        self.bar_confianza.setStyleSheet("""
            QProgressBar {
                border: 1px solid #45A29E;
                border-radius: 4px;
                text-align: center;
                background-color: #1F2833;
                color: white;
                font-weight: bold;
                height: 18px;
            }
            QProgressBar::chunk {
                background-color: #66FCF1;
            }
        """)
        lay_vocal.addWidget(self.bar_confianza)

        # Badges de desglose de probabilidades (/a/, /e/, /i/, /o/, /u/)
        lay_probs = QHBoxLayout()
        self.lbl_probs = {}
        for v in VOCALES:
            col = VOCAL_A_COLOR_HEX[v]
            lbl = QLabel(f"/{v}/: 0%")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px; border: 1px solid {col}; border-radius: 4px; padding: 2px;")
            self.lbl_probs[v] = lbl
            lay_probs.addWidget(lbl)
        lay_vocal.addLayout(lay_probs)

        self.layout_decodificador.addWidget(self.grp_vocal)

        # 3. Gráfico del Espacio Latente: Fronteras 2D y Nube 3D Interactiva
        self.grp_latente = QGroupBox("Espacio Latente: Fronteras 2D y Nube 3D Interactiva")
        lay_latente = QVBoxLayout(self.grp_latente)
        lay_latente.setContentsMargins(6, 6, 6, 6)
        lay_latente.setSpacing(4)

        # Barra de selección de vista: 2D vs 3D
        lay_switch_vista = QHBoxLayout()
        lay_switch_vista.setContentsMargins(0, 0, 0, 0)
        lay_switch_vista.setSpacing(6)

        self.btn_vista_2d = QPushButton("Vista 2D: Fronteras de Decisión")
        self.btn_vista_3d = QPushButton("Vista 3D: Nube Interactiva")
        for btn in (self.btn_vista_2d, self.btn_vista_3d):
            btn.setCheckable(True)
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            btn.setFixedHeight(26)

        self.btn_vista_2d.setChecked(True)
        self.btn_vista_2d.clicked.connect(lambda: self._conmutar_vista_latente('2D'))
        self.btn_vista_3d.clicked.connect(lambda: self._conmutar_vista_latente('3D'))

        lay_switch_vista.addWidget(self.btn_vista_2d)
        lay_switch_vista.addWidget(self.btn_vista_3d)

        # Selector de proyección de plano 2D
        self.lbl_proy_2d = QLabel("Plano 2D:")
        self.lbl_proy_2d.setStyleSheet("color: #66FCF1; font-weight: bold; font-size: 11px;")
        self.cmb_proyeccion_2d = QComboBox()
        self.cmb_proyeccion_2d.addItems(["Z1 - Z2", "Z1 - Z3", "Z2 - Z3", "Z1 - Z4 (Desacople /E/-/O/)", "Z2 - Z4 (Desacople /E/-/O/)", "Z3 - Z4"])
        self.cmb_proyeccion_2d.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 2px;")
        self.cmb_proyeccion_2d.currentIndexChanged.connect(self._on_proyeccion_2d_cambiada)
        self.ejes_proyeccion_2d = (0, 1)

        lay_switch_vista.addWidget(self.lbl_proy_2d)
        lay_switch_vista.addWidget(self.cmb_proyeccion_2d)
        lay_latente.addLayout(lay_switch_vista)

        # Contenedor apilado para alternar entre 2D y 3D
        self.stack_latente = QStackedWidget()

        # --- Página 0: Vista 2D con PyQtGraph ---
        self.plot_latente = pg.PlotWidget()
        self.plot_latente.showGrid(x=True, y=True, alpha=0.65)
        self.plot_latente.setLabel('bottom', "Coordenada Z1")
        self.plot_latente.setLabel('left', "Coordenada Z2")
        self.plot_latente.setAspectLocked(False)
        self.plot_latente.setMinimumHeight(320)

        # Fondo con imagen RGBA de las fronteras continuas
        self.img_fronteras = pg.ImageItem()
        self.plot_latente.addItem(self.img_fronteras)

        # Centroides GMM 2D como diamantes
        self.scatter_centroides_2d = pg.ScatterPlotItem()
        self.plot_latente.addItem(self.scatter_centroides_2d)

        # Origen (0,0) como cruz
        self.scatter_origen_2d = pg.ScatterPlotItem()
        self.plot_latente.addItem(self.scatter_origen_2d)

        # Rastro de puntos decodificados en vivo
        self.scatter_trail = pg.ScatterPlotItem()
        self.plot_latente.addItem(self.scatter_trail)

        # Estrella destacada en el fonema actual
        self.scatter_actual = pg.ScatterPlotItem()
        self.plot_latente.addItem(self.scatter_actual)

        self.stack_latente.addWidget(self.plot_latente)

        # --- Página 1: Vista 3D Interactiva con Matplotlib ---
        self.widget_3d = QWidget()
        lay_w3d = QVBoxLayout(self.widget_3d)
        lay_w3d.setContentsMargins(0, 0, 0, 0)

        self.fig_3d = Figure(figsize=(4.5, 3.2), facecolor='#0B0C10', dpi=100)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setMinimumHeight(320)
        self.ax_3d = self.fig_3d.add_subplot(111, projection='3d')
        self._configurar_ejes_3d()

        lay_w3d.addWidget(self.canvas_3d)
        self.stack_latente.addWidget(self.widget_3d)

        lay_latente.addWidget(self.stack_latente)
        self.layout_decodificador.addWidget(self.grp_latente, stretch=2)

        self._actualizar_estilo_botones_vista()

        # 4. Telemetría y Controles del Gate Doble con Desfasaje EMD
        self.grp_gate = QGroupBox("Detector Gate Doble y Desfasaje Fisiológico EMD respecto al Micrófono")
        lay_gate = QGridLayout(self.grp_gate)
        lay_gate.setContentsMargins(6, 6, 6, 6)
        lay_gate.setSpacing(6)

        self.lbl_badge_estado = QLabel("GATE: CALIBRANDO")
        self.lbl_badge_estado.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_badge_estado.setStyleSheet("background-color: #222; color: #FFFF00; font-weight: bold; padding: 4px; border-radius: 4px;")

        self.lbl_u_bajo = QLabel("U_bajo: 0.0")
        self.lbl_u_alto = QLabel("U_alto: 0.0")

        # Control interactivo de desfasaje EMD
        lbl_spin_emd = QLabel("Retardo EMD respecto al Micrófono:")
        lbl_spin_emd.setStyleSheet("color: #C5C6C7; font-weight: bold; font-size: 11px;")

        self.spin_emd_delay = QSpinBox()
        self.spin_emd_delay.setRange(0, 1000)
        self.spin_emd_delay.setValue(71)
        self.spin_emd_delay.setSuffix(" ms")
        self.spin_emd_delay.setSingleStep(10)
        self.spin_emd_delay.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 3px;")
        self.spin_emd_delay.setToolTip(
            "Retardo electromecánico fisiológico del músculo respecto al audio: proyecta el centro de la ventana a n_onset + EMD_lag."
        )
        self.spin_emd_delay.valueChanged.connect(self._on_emd_delay_changed)

        self.lbl_emd_info = QLabel("Corrimiento Gate->Mic: +71 ms (Híbrido)")
        self.lbl_emd_info.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 11px;")

        self.btn_recalibrar = QPushButton("Recalibrar Silencio")
        self.btn_recalibrar.clicked.connect(self.recalibrar_silencio)

        lay_gate.addWidget(self.lbl_badge_estado, 0, 0)
        lay_gate.addWidget(self.lbl_u_bajo, 0, 1)
        lay_gate.addWidget(self.lbl_u_alto, 0, 2)

        lay_gate.addWidget(lbl_spin_emd, 1, 0)
        lay_gate.addWidget(self.spin_emd_delay, 1, 1)
        lay_gate.addWidget(self.lbl_emd_info, 1, 2)

        self.lbl_ventana_info = QLabel("Ventana [t_ini, t_fin]: Esperando disparo... | Duración: -- ms")
        self.lbl_ventana_info.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_ventana_info.setStyleSheet("background-color: #111827; color: #66FCF1; font-weight: bold; font-size: 11px; padding: 4px; border: 1px solid #374151; border-radius: 4px;")
        lay_gate.addWidget(self.lbl_ventana_info, 2, 0, 1, 3)

        lay_gate.addWidget(self.btn_recalibrar, 3, 0, 1, 3)

        self.layout_decodificador.addWidget(self.grp_gate)

        # 4B. Panel de Auditoría y Feedback en Tiempo Real por Teclado (Tarea D)
        self.grp_auditoria = QGroupBox("Auditoría en Vivo (Teclado: [Espacio/V]=Acierto | [C/M]=Confusión | [X/F]=Falsa Ventana)")
        lay_aud = QVBoxLayout(self.grp_auditoria)
        lay_aud.setContentsMargins(6, 6, 6, 6)
        lay_aud.setSpacing(4)

        row_aud_btns = QHBoxLayout()
        self.btn_aud_acierto = QPushButton("✓ ACIERTO [Espacio / V]")
        self.btn_aud_acierto.setStyleSheet("background-color: #064E3B; color: #34D399; font-weight: bold; border: 1px solid #10B981; padding: 5px; border-radius: 4px;")
        self.btn_aud_acierto.clicked.connect(lambda: self.registrar_auditoria_en_vivo("acierto"))

        self.btn_aud_confusion = QPushButton("⚠ VOCAL ERRÓNEA [C / M]")
        self.btn_aud_confusion.setStyleSheet("background-color: #78350F; color: #FBBF24; font-weight: bold; border: 1px solid #F59E0B; padding: 5px; border-radius: 4px;")
        self.btn_aud_confusion.clicked.connect(lambda: self.registrar_auditoria_en_vivo("confusion"))

        self.btn_aud_falsa = QPushButton("✖ FALSA VENTANA [X / F]")
        self.btn_aud_falsa.setStyleSheet("background-color: #7F1D1D; color: #F87171; font-weight: bold; border: 1px solid #EF4444; padding: 5px; border-radius: 4px;")
        self.btn_aud_falsa.clicked.connect(lambda: self.registrar_auditoria_en_vivo("falsa_ventana"))

        row_aud_btns.addWidget(self.btn_aud_acierto)
        row_aud_btns.addWidget(self.btn_aud_confusion)
        row_aud_btns.addWidget(self.btn_aud_falsa)
        lay_aud.addLayout(row_aud_btns)

        self.lbl_auditoria_status = QLabel("Auditoría lista: presione Espacio/V (Acierto), C/M (Confusión) o X/F (Falsa Ventana).")
        self.lbl_auditoria_status.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_auditoria_status.setStyleSheet("color: #9CA3AF; font-size: 11px; font-weight: bold; padding: 3px; border-radius: 3px;")
        lay_aud.addWidget(self.lbl_auditoria_status)

        self.layout_decodificador.addWidget(self.grp_auditoria)

        # 5. Marcador y Contador de Frecuencia de Aciertos para Grabaciones de Prueba
        self.grp_score = QGroupBox("Rendimiento en Vivo - Grabación de Prueba")
        lay_score = QVBoxLayout(self.grp_score)
        lay_score.setContentsMargins(6, 6, 6, 6)
        lay_score.setSpacing(4)

        self.lbl_score_total = QLabel("Aciertos: 0 / 0 (0.0%)")
        self.lbl_score_total.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_score_total.setStyleSheet("color: #00FFCC; font-size: 18px; font-weight: 900; padding: 4px;")
        lay_score.addWidget(self.lbl_score_total)

        self.bar_score = QProgressBar()
        self.bar_score.setRange(0, 100)
        self.bar_score.setValue(0)
        self.bar_score.setTextVisible(True)
        self.bar_score.setStyleSheet("""
            QProgressBar {
                border: 1px solid #00FFCC;
                border-radius: 4px;
                text-align: center;
                background-color: #1F2833;
                color: white;
                font-weight: bold;
                height: 16px;
            }
            QProgressBar::chunk {
                background-color: #00FFCC;
            }
        """)
        lay_score.addWidget(self.bar_score)

        lay_vocal_stats = QHBoxLayout()
        self.lbl_vocal_scores = {}
        for v in VOCALES:
            col = VOCAL_A_COLOR_HEX[v]
            lbl = QLabel(f"/{v}/: 0/0")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px; border: 1px solid {col}; border-radius: 3px; padding: 2px;")
            self.lbl_vocal_scores[v] = lbl
            lay_vocal_stats.addWidget(lbl)
        lay_score.addLayout(lay_vocal_stats)

        self.lbl_last_comp = QLabel("Último pulso: Esperando eventos...")
        self.lbl_last_comp.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_last_comp.setStyleSheet("color: #C5C6C7; font-size: 11px; font-style: italic; padding: 2px;")
        lay_score.addWidget(self.lbl_last_comp)

        btn_reset_score = QPushButton("Reiniciar Marcador")
        btn_reset_score.setStyleSheet("background-color: #21262d; color: #8b949e; border: 1px solid #30363d; padding: 3px; border-radius: 3px; font-size: 11px;")
        btn_reset_score.clicked.connect(self.reset_test_score)
        lay_score.addWidget(btn_reset_score)

        self.layout_decodificador.addWidget(self.grp_score)

    def eventFilter(self, obj, event):
        """Redimensiona el overlay flotante AutoForge junto con el plot_widget."""
        if obj == getattr(self, 'plot_widget', None) and event.type() == QtCore.QEvent.Type.Resize:
            if hasattr(self, 'autoforge_overlay'):
                self.autoforge_overlay.resize(event.size())
        return super().eventFilter(obj, event)

    def hide_all_viz_channels(self):
        """Oculta o muestra todos los canales en el osciloscopio idéntico a AutoForge."""
        all_unchecked = all(not chk.isChecked() for chk in self.viz_checkboxes)
        if all_unchecked:
            for chk in self.viz_checkboxes:
                chk.setChecked(True)
            self.btn_hide_all_viz.setText("Ocultar Todos")
        else:
            for chk in self.viz_checkboxes:
                chk.setChecked(False)
            self.btn_hide_all_viz.setText("Mostrar Todos")

    def _on_emd_delay_changed(self, val):
        """Actualiza el retraso electromecánico EMD en el motor en tiempo real."""
        self.engine.set_emd_delay_ms(val)
        self.lbl_emd_info.setText(f"Lag Fisiológico: +{val} ms")

    def _on_rms_window_changed(self, val):
        """Sincroniza la ventana de suavizado RMS tanto en el motor como en el osciloscopio."""
        if hasattr(self.engine, 'set_rms_window_ms'):
            self.engine.set_rms_window_ms(val)

    def _actualizar_estilo_botones_vista(self):
        """Aplica el estilo visual a los botones de conmutación 2D / 3D."""
        style_active = "background-color: #1F2833; color: #66FCF1; font-weight: bold; border: 2px solid #66FCF1; border-radius: 4px;"
        style_inactive = "background-color: #0B0C10; color: #888888; font-weight: normal; border: 1px solid #444444; border-radius: 4px;"
        if self.stack_latente.currentIndex() == 0:
            self.btn_vista_2d.setChecked(True)
            self.btn_vista_3d.setChecked(False)
            self.btn_vista_2d.setStyleSheet(style_active)
            self.btn_vista_3d.setStyleSheet(style_inactive)
        else:
            self.btn_vista_2d.setChecked(False)
            self.btn_vista_3d.setChecked(True)
            self.btn_vista_2d.setStyleSheet(style_inactive)
            self.btn_vista_3d.setStyleSheet(style_active)

    def _conmutar_vista_latente(self, modo: str):
        """Conmuta entre la vista 2D (PyQtGraph) y la vista 3D interactiva (Matplotlib)."""
        if modo == '3D':
            self.stack_latente.setCurrentIndex(1)
            self._actualizar_grafico_3d()
        else:
            self.stack_latente.setCurrentIndex(0)
        self._actualizar_estilo_botones_vista()

    def _configurar_ejes_3d(self):
        """Configura los ejes 3D oscuros con tipografía limpia y sin brillos invasivos."""
        self.ax_3d.set_facecolor('#0B0C10')
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('#222222')
        self.ax_3d.yaxis.pane.set_edgecolor('#222222')
        self.ax_3d.zaxis.pane.set_edgecolor('#222222')
        self.ax_3d.grid(True, color='#333333', linestyle=':', alpha=0.6)
        self.ax_3d.tick_params(colors='#AAAAAA', labelsize=8)
        self.ax_3d.set_xlabel("Z1", color='#66FCF1', fontsize=9, fontweight='bold', labelpad=2)
        self.ax_3d.set_ylabel("Z2", color='#66FCF1', fontsize=9, fontweight='bold', labelpad=2)
        self.ax_3d.set_zlabel("Z3", color='#66FCF1', fontsize=9, fontweight='bold', labelpad=2)
        self.fig_3d.tight_layout(pad=1.2)

    def _actualizar_grafico_3d(self):
        """Actualiza la nube de puntos 3D interactiva con rotación por mouse, centroides y fonemas."""
        if not hasattr(self, 'ax_3d') or self.ax_3d is None:
            return

        elev = self.ax_3d.elev
        azim = self.ax_3d.azim

        self.ax_3d.cla()
        self._configurar_ejes_3d()

        # 1. Origen 3D (0, 0, 0)
        self.ax_3d.scatter([0], [0], [0], color='#FFFFFF', marker='+', s=140, linewidth=2.2, zorder=10)

        # 2. Centroides GMM 3D en diamantes
        if self.engine.gmm is not None and hasattr(self.engine.gmm, 'means_'):
            means = self.engine.gmm.means_
            for c_idx, mean_pt in enumerate(means):
                v_idx = self.engine.cluster_to_vocal.get(c_idx, c_idx)
                v_name = VOCALES[v_idx] if 0 <= v_idx < len(VOCALES) else 'A'
                c_hex = VOCAL_A_COLOR_HEX.get(v_name, '#CCCCCC')
                m_z1 = float(mean_pt[0])
                m_z2 = float(mean_pt[1]) if len(mean_pt) > 1 else 0.0
                m_z3 = float(mean_pt[2]) if len(mean_pt) > 2 else 0.0
                self.ax_3d.scatter(
                    [m_z1], [m_z2], [m_z3],
                    color=c_hex, marker='D', s=160, edgecolor='white', linewidth=1.5, zorder=8
                )
                self.ax_3d.text(
                    m_z1, m_z2, m_z3 + 0.12, f"/{v_name}/",
                    color=c_hex, fontsize=9, fontweight='bold', zorder=9
                )

        # 3. Puntos fonatorios decodificados del historial
        if self.historial_puntos:
            pts_by_v = {v: {'z1': [], 'z2': [], 'z3': []} for v in VOCALES}
            for pt in self.historial_puntos:
                hz1 = pt[0]
                hz2 = pt[1]
                hz3 = pt[2] if len(pt) > 4 else 0.0
                hv = pt[4] if len(pt) >= 6 else (pt[3] if len(pt) > 4 else pt[2])
                if hv in pts_by_v:
                    pts_by_v[hv]['z1'].append(hz1)
                    pts_by_v[hv]['z2'].append(hz2)
                    pts_by_v[hv]['z3'].append(hz3)

            for v in VOCALES:
                xs = pts_by_v[v]['z1']
                ys = pts_by_v[v]['z2']
                zs = pts_by_v[v]['z3']
                if xs:
                    self.ax_3d.scatter(
                        xs, ys, zs,
                        color=VOCAL_A_COLOR_HEX[v],
                        edgecolor='#000000',
                        linewidth=0.7,
                        s=45,
                        alpha=0.85,
                        zorder=6
                    )

            # Destacar fonema actual
            ultimo = self.historial_puntos[-1]
            uz1 = ultimo[0]
            uz2 = ultimo[1]
            uz3 = ultimo[2] if len(ultimo) > 4 else 0.0
            uv = ultimo[4] if len(ultimo) >= 6 else (ultimo[3] if len(ultimo) > 4 else ultimo[2])
            self.ax_3d.scatter(
                [uz1], [uz2], [uz3],
                color='#FFFFFF',
                edgecolor=VOCAL_A_COLOR_HEX.get(uv, '#00FFFF'),
                linewidth=2.5,
                marker='*',
                s=240,
                zorder=12
            )

        if elev is not None and azim is not None:
            self.ax_3d.view_init(elev=elev, azim=azim)

        self.canvas_3d.draw_idle()

    def _on_proyeccion_2d_cambiada(self, idx):
        """Conmuta los ejes de proyección cartesianos para la visualización del plano 2D (soporta 3D y 4D)."""
        max_ax = max(1, int(getattr(self.engine, 'latent_dim', 3)) - 1)
        mapa_ejes = {
            0: ((0, 1), "Coordenada Z1", "Coordenada Z2"),
            1: ((0, min(2, max_ax)), "Coordenada Z1", f"Coordenada Z{min(2, max_ax)+1}"),
            2: ((1, min(2, max_ax)), "Coordenada Z2", f"Coordenada Z{min(2, max_ax)+1}"),
            3: ((0, min(3, max_ax)), "Coordenada Z1", f"Coordenada Z{min(3, max_ax)+1} (Desacople /E/-/O/)"),
            4: ((1, min(3, max_ax)), "Coordenada Z2", f"Coordenada Z{min(3, max_ax)+1} (Desacople /E/-/O/)"),
            5: ((min(2, max_ax), min(3, max_ax)), f"Coordenada Z{min(2, max_ax)+1}", f"Coordenada Z{min(3, max_ax)+1}")
        }
        self.ejes_proyeccion_2d, lbl_x, lbl_y = mapa_ejes.get(idx, ((0, 1), "Coordenada Z1", "Coordenada Z2"))

        self.plot_latente.setLabel('bottom', lbl_x)
        self.plot_latente.setLabel('left', lbl_y)
        self._cargar_mapa_fronteras_decision()
        self._redibujar_puntos_2d()

    def _redibujar_puntos_2d(self):
        """Redibuja el rastro de puntos fonatorios en el plano 2D seleccionado."""
        ax_x, ax_y = getattr(self, 'ejes_proyeccion_2d', (0, 1))
        trail_spots = []
        for i, pt in enumerate(self.historial_puntos[:-1]):
            coords_pt = [pt[0], pt[1], pt[2] if len(pt) > 4 else 0.0, pt[3] if len(pt) >= 6 else 0.0]
            hz_x = coords_pt[ax_x] if ax_x < len(coords_pt) else coords_pt[0]
            hz_y = coords_pt[ax_y] if ax_y < len(coords_pt) else coords_pt[1]
            hv = pt[4] if len(pt) >= 6 else (pt[3] if len(pt) > 4 else pt[2])
            alpha_ratio = (i + 1) / len(self.historial_puntos)
            c = VOCAL_A_COLOR_HEX.get(hv, '#FFFFFF')
            trail_spots.append({
                'pos': (hz_x, hz_y),
                'size': int(11 + 6 * alpha_ratio),
                'brush': pg.mkBrush(c),
                'pen': pg.mkPen('#000000', width=1.0)
            })
        self.scatter_trail.setData(trail_spots)

        if self.historial_puntos:
            ultimo = self.historial_puntos[-1]
            coords_u = [ultimo[0], ultimo[1], ultimo[2] if len(ultimo) > 4 else 0.0, ultimo[3] if len(ultimo) >= 6 else 0.0]
            uz_x = coords_u[ax_x] if ax_x < len(coords_u) else coords_u[0]
            uz_y = coords_u[ax_y] if ax_y < len(coords_u) else coords_u[1]
            uv = ultimo[4] if len(ultimo) >= 6 else (ultimo[3] if len(ultimo) > 4 else ultimo[2])
            color_hex = VOCAL_A_COLOR_HEX.get(uv, '#FFFFFF')
            self.scatter_actual.setData([{
                'pos': (uz_x, uz_y),
                'size': 26,
                'symbol': 'star',
                'brush': pg.mkBrush('#FFFFFF'),
                'pen': pg.mkPen(color_hex, width=3)
            }])

    def _cargar_modelo_personalizado_dialogo(self):
        """Abre un diálogo de selección de archivo para cargar un checkpoint PyTorch propio."""
        f, _ = QFileDialog.getOpenFileName(self, "Seleccionar Modelo PyTorch (.pt / .pth)", project_root, "Modelos PyTorch (*.pt *.pth);;Todos (*.*)")
        if f and os.path.exists(f):
            nombre = os.path.basename(f)
            if self.engine.load_model(f):
                self.lbl_modelo_info.setText(f"Modelo cargado: {nombre}")
                self.lbl_modelo_info.setStyleSheet("color: #39FF14; font-size: 11px; font-weight: bold;")
                self._cargar_mapa_fronteras_decision()
                self.historial_puntos.clear()
                self.scatter_trail.clear()
                self.scatter_actual.clear()
                if self.engine.latent_dim >= 3:
                    self.cmb_proyeccion_2d.setEnabled(True)
                    self.lbl_proy_2d.setText("Plano 2D:")
                    self._conmutar_vista_latente('3D')
                else:
                    self.cmb_proyeccion_2d.setEnabled(False)
                    self.lbl_proy_2d.setText("Plano (2D Nativo):")
                    self._conmutar_vista_latente('2D')
            else:
                self.lbl_modelo_info.setText(f"⚠ Error al procesar checkpoint: {nombre}")
                self.lbl_modelo_info.setStyleSheet("color: #FF3300; font-size: 11px; font-weight: bold;")

    def _on_model_changed(self, idx):
        """Carga el modelo seleccionado en el desplegable y regenera las fronteras de decisión."""
        if idx < 0 or idx >= len(MODELOS_PRESETS):
            return
        preset = MODELOS_PRESETS[idx]
        p_pt = preset['path']
        if os.path.exists(p_pt):
            print(f"[Modelo] Conmutando a: {preset['nombre']}")
            if self.engine.load_model(p_pt):
                self.lbl_modelo_info.setText(preset['subtitulo'])
                self.lbl_modelo_info.setStyleSheet("color: #66FCF1; font-size: 11px; font-weight: bold;")
                self._cargar_mapa_fronteras_decision()
                self.historial_puntos.clear()
                self.scatter_trail.clear()
                self.scatter_actual.clear()
                if self.engine.latent_dim >= 3:
                    self.cmb_proyeccion_2d.setEnabled(True)
                    self.lbl_proy_2d.setText("Plano 2D:")
                    self._conmutar_vista_latente('3D')
                else:
                    self.cmb_proyeccion_2d.setEnabled(False)
                    self.cmb_proyeccion_2d.blockSignals(True)
                    self.cmb_proyeccion_2d.setCurrentIndex(0)
                    self.cmb_proyeccion_2d.blockSignals(False)
                    self.ejes_proyeccion_2d = (0, 1)
                    self.lbl_proy_2d.setText("Plano (2D Nativo):")
                    self.plot_latente.setLabel('bottom', "Coordenada Z1")
                    self.plot_latente.setLabel('left', "Coordenada Z2")
                    self._conmutar_vista_latente('2D')
            else:
                self.lbl_modelo_info.setText(f"⚠ Error al leer checkpoint de {preset['nombre']}")
                self.lbl_modelo_info.setStyleSheet("color: #FFB703; font-size: 11px; font-weight: bold;")
        else:
            print(f"[Modelo AVISO] Archivo preset no encontrado: {p_pt}")
            self.lbl_modelo_info.setText(f"⚠ Preset no encontrado: {os.path.basename(p_pt)} (cargue uno manualmente)")
            self.lbl_modelo_info.setStyleSheet("color: #FFB703; font-size: 11px; font-weight: bold;")

    def _cargar_mapa_fronteras_decision(self):
        """Calcula y dibuja las regiones de decisión del GMM con zoom óptimo sobre los fonemas activos."""
        if self.engine.model is None or self.engine.gmm is None:
            self.img_fronteras.clear()
            self.scatter_centroides_2d.setData([])
            return

        ax_x, ax_y = getattr(self, 'ejes_proyeccion_2d', (0, 1))

        if self.engine.latent_dim == 2:
            x_range = (-1.8, 2.5)
            y_range = (-1.7, 3.4)
        else:
            if self.engine.gmm is not None and hasattr(self.engine.gmm, 'means_'):
                means = self.engine.gmm.means_
                mx_min, mx_max = float(means[:, ax_x].min()), float(means[:, ax_x].max())
                my_min, my_max = float(means[:, ax_y].min()), float(means[:, ax_y].max())
                pad_x = max(0.8, (mx_max - mx_min) * 0.45)
                pad_y = max(0.8, (my_max - my_min) * 0.45)
                x_range = (mx_min - pad_x, mx_max + pad_x)
                y_range = (my_min - pad_y, my_max + pad_y)
            else:
                x_range = (-2.5, 2.5)
                y_range = (-2.5, 2.5)

        rgba_img, x_rng, y_rng = self.engine.compute_decision_grid(
            x_range=x_range,
            y_range=y_range,
            resolution=180,
            proj_axes=(ax_x, ax_y)
        )
        if rgba_img is not None:
            self.img_fronteras.setImage(np.transpose(rgba_img, (1, 0, 2)))
            self.img_fronteras.setRect(QtCore.QRectF(x_rng[0], y_rng[0], x_rng[1] - x_rng[0], y_rng[1] - y_rng[0]))
            self.plot_latente.setXRange(x_rng[0], x_rng[1], padding=0.0)
            self.plot_latente.setYRange(y_rng[0], y_rng[1], padding=0.0)
            self.plot_latente.getViewBox().setLimits(
                xMin=x_rng[0] - 0.5, xMax=x_rng[1] + 0.5,
                yMin=y_rng[0] - 0.5, yMax=y_rng[1] + 0.5
            )

        # Centroides 2D en diamante en el plano seleccionado
        centroides_spots = []
        if self.engine.gmm is not None and hasattr(self.engine.gmm, 'means_'):
            means = self.engine.gmm.means_
            for c_idx, cen in enumerate(means):
                v_idx = self.engine.cluster_to_vocal.get(c_idx, c_idx)
                v_name = VOCALES[v_idx] if 0 <= v_idx < len(VOCALES) else 'A'
                c_hex = VOCAL_A_COLOR_HEX.get(v_name, '#FFFFFF')
                cx = float(cen[ax_x])
                cy = float(cen[ax_y]) if len(cen) > ax_y else 0.0
                centroides_spots.append({
                    'pos': (cx, cy),
                    'size': 18,
                    'symbol': 'd',
                    'brush': pg.mkBrush(c_hex),
                    'pen': pg.mkPen('#000000', width=2.0)
                })
        self.scatter_centroides_2d.setData(centroides_spots)

        # Origen (0, 0) como cruz
        self.scatter_origen_2d.setData([{
            'pos': (0.0, 0.0),
            'size': 16,
            'symbol': '+',
            'brush': pg.mkBrush('#FFFFFF'),
            'pen': pg.mkPen('#FFFFFF', width=2.5)
        }])

        self._actualizar_grafico_3d()

    def _on_toggle_microfono(self, checked):
        """Conecta o desconecta totalmente el micrófono en modo Gate Cinemático (Tarea E.1)."""
        mic_on = bool(checked)
        self.engine.allow_mic_monitor = mic_on
        if hasattr(self, "acquisition_thread") and self.acquisition_thread is not None:
            if hasattr(self.acquisition_thread, "set_mic_enabled"):
                self.acquisition_thread.set_mic_enabled(mic_on)
        if not mic_on:
            self.engine.mic_buffer[:] = 0.0
            if hasattr(self, "bar_mic_level") and self.bar_mic_level is not None:
                self.bar_mic_level.setValue(0)
                self.lbl_mic_val.setText("DESCONECTADO (Modo Gate Cinemático)")
        if len(getattr(self, "viz_checkboxes", [])) >= 7:
            self.viz_checkboxes[6].setChecked(mic_on)

    def _on_live_frame_ready(self, qt_img, face_ok, mic_amp):
        """Recibe el fotograma de la cámara con landmarks y telemetría de audio."""
        if hasattr(self, 'lbl_video_preview') and self.lbl_video_preview is not None:
            pixmap = QtGui.QPixmap.fromImage(qt_img)
            scaled = pixmap.scaled(
                self.lbl_video_preview.width(), self.lbl_video_preview.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.lbl_video_preview.setPixmap(scaled)

        if hasattr(self, 'lbl_badge_rostro') and self.lbl_badge_rostro is not None:
            if face_ok:
                self.lbl_badge_rostro.setText("ROSTRO: DETECTADO (OK)")
                self.lbl_badge_rostro.setStyleSheet(
                    "background-color: #064E3B; color: #34D399; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px;"
                )
            else:
                self.lbl_badge_rostro.setText("ROSTRO: BUSCANDO...")
                self.lbl_badge_rostro.setStyleSheet(
                    "background-color: #450A0A; color: #F87171; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px;"
                )

        if hasattr(self, 'bar_mic_level') and self.bar_mic_level is not None:
            lvl_pct = min(100, int(mic_amp * 2.0))
            self.bar_mic_level.setValue(lvl_pct)
            self.lbl_mic_val.setText(f"{mic_amp:.2f} µV / RMS Acústico")

    def toggle_daq(self):
        """Inicia o detiene la adquisición en vivo por Cámara Web y Micrófono."""
        if not self.is_acquiring:
            if self.is_playback:
                self.toggle_wav_playback()

            self.stop_event.clear()
            chunk_samples = int(self.SAMPLE_RATE * 0.05)

            if self.chk_modo_simulado.isChecked():
                self.acquisition_thread = threading.Thread(
                    target=simulador_thread,
                    args=(chunk_samples, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event),
                    daemon=True
                )
                self.acquisition_thread.start()
            else:
                mic_on = bool(getattr(self, "chk_habilitar_microfono", None) and self.chk_habilitar_microfono.isChecked())
                self.engine.allow_mic_monitor = mic_on
                self.acquisition_thread = WebcamAudioLiveThread(
                    data_queue=self.data_queue,
                    stop_event=self.stop_event,
                    sample_rate=self.SAMPLE_RATE,
                    camera_index=0,
                    audio_device=1,
                    baseline_stats=None,
                    enable_mic=mic_on
                )
                self.acquisition_thread.frame_ready.connect(self._on_live_frame_ready)
                self.acquisition_thread.start()

            self.is_acquiring = True
            self.btn_iniciar_daq.setText("DETENER CÁMARA FACIAL")
            self.btn_iniciar_daq.setStyleSheet(self.BTN_STOP_STYLE)
            self.engine.reset_state()
            self.historial_puntos.clear()
            self.scatter_trail.clear()
            self.scatter_actual.clear()
        else:
            self.stop_event.set()
            if self.acquisition_thread:
                if hasattr(self.acquisition_thread, 'wait'):
                    self.acquisition_thread.wait(1500)
                elif hasattr(self.acquisition_thread, 'join'):
                    self.acquisition_thread.join(timeout=1.0)
            self.is_acquiring = False
            self.btn_iniciar_daq.setText("INICIAR CÁMARA FACIAL EN VIVO")
            self.btn_iniciar_daq.setStyleSheet(self.BTN_START_STYLE)
            if hasattr(self, 'lbl_badge_rostro'):
                self.lbl_badge_rostro.setText("ROSTRO: ESPERANDO CÁMARA")
                self.lbl_badge_rostro.setStyleSheet(
                    "background-color: #1F2833; color: #FFAA00; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px;"
                )
            if hasattr(self, 'lbl_video_preview'):
                self.lbl_video_preview.clear()
                self.lbl_video_preview.setText("CÁMARA DETENIDA")

    def toggle_wav_playback(self):
        """Inicia o detiene la reproducción en tiempo real de los 4 canales de una sesión grabada."""
        if not self.is_playback:
            if self.is_acquiring:
                self.toggle_daq()

            tray_csv = os.path.join(self.ruta_sesion_wav, "trayectoria_puntos.csv")
            wav_paths = [os.path.join(self.ruta_sesion_wav, f"canal_{ch}/grabacion.wav") for ch in range(self.NUM_CANALES)]
            if os.path.exists(tray_csv):
                self.stop_event.clear()
                chunk_samples = max(1, int(self.SAMPLE_RATE * 0.05))
                self.acquisition_thread = threading.Thread(
                    target=csv_6ch_playback_thread,
                    args=(self.ruta_sesion_wav, self.SAMPLE_RATE, chunk_samples, self.data_queue, self.stop_event, self.engine.ckpt),
                    daemon=True
                )
                self.acquisition_thread.start()
                self.is_playback = True
                self.btn_play_wav.setText("Detener Medición")
                self.btn_play_wav.setStyleSheet(self.BTN_STOP_STYLE)
                self.engine.reset_state()
                self.reset_test_score()
                self.historial_puntos.clear()
                self.scatter_trail.clear()
                self.scatter_actual.clear()
                return
            if not all(os.path.exists(p) for p in wav_paths):
                print(f"[SESION ERROR] Faltan archivos en {self.ruta_sesion_wav}")
                return

            self.stop_event.clear()
            chunk_samples = int(self.SAMPLE_RATE * 0.05)
            self.acquisition_thread = threading.Thread(
                target=wav_playback_thread,
                args=(wav_paths, self.SAMPLE_RATE, chunk_samples, self.data_queue, self.stop_event, False),
                daemon=True
            )
            self.acquisition_thread.start()
            self.is_playback = True
            self.btn_play_wav.setText("Detener WAV")
            self.btn_play_wav.setStyleSheet(self.BTN_STOP_STYLE)
            self.engine.reset_state()
            self.reset_test_score()
            self.historial_puntos.clear()
            self.scatter_trail.clear()
            self.scatter_actual.clear()
        else:
            self.stop_event.set()
            if self.acquisition_thread:
                self.acquisition_thread.join(timeout=1.0)
            self.is_playback = False
            self.btn_play_wav.setText("Reproducir WAV en Vivo")
            self.btn_play_wav.setStyleSheet(self.BTN_PLAY_STYLE)
            # Decodificar cualquier pulso pendiente al final de la grabación
            flushed = self.engine.flush_pending_events()
            if flushed:
                for ev in flushed:
                    self._mostrar_evento_decodificado(ev)

    def reset_test_score(self):
        """Reinicia el contador de aciertos y estadísticas por vocal para la prueba."""
        self.test_event_count = 0
        self.test_correct_count = 0
        self.test_vocal_stats = {v: {'correct': 0, 'total': 0} for v in VOCALES}
        self.lbl_score_total.setText("Aciertos: 0 / 0 (0.0%)")
        self.bar_score.setValue(0)
        for v in VOCALES:
            self.lbl_vocal_scores[v].setText(f"/{v}/: 0/0")
        self.lbl_last_comp.setText("Último pulso: Esperando eventos...")

    def _cargar_metadata_sesion(self, folder):
        """Lee metadata.json de canal_0 para extraer valid_words de prueba y calibrar sesión."""
        p_meta = os.path.join(folder, "canal_0/metadata.json")
        if os.path.exists(p_meta):
            try:
                with open(p_meta, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                self.test_ground_truth = meta.get("valid_words", [])
                print(f"[Sesión] Ground truth cargado: {len(self.test_ground_truth)} fonemas de prueba.")
            except Exception as e:
                print(f"[Sesión Error] No se pudo leer metadata: {e}")
                self.test_ground_truth = []
        else:
            self.test_ground_truth = []

        # Calibración automática según la sesión (manteniendo modo cinemático autónomo GATE_HYBRID)
        nom = os.path.basename(folder).upper()
        if "P5" in nom or "PRUEBA5" in nom:
            if hasattr(self.engine, 'set_calibration_preset'):
                self.engine.set_calibration_preset('P5')
            self.engine.set_trigger_mode('GATE_HYBRID', auto_update_shift=True)
            self.spin_emd_delay.setValue(71)
            print("[Sesión] Perfil P5 (Secuencia Continua) activado en modo cinemático autónomo (GATE_HYBRID +71ms).")
        elif "LUCAS" in nom:
            if hasattr(self.engine, 'set_calibration_preset'):
                self.engine.set_calibration_preset('LUCAS')
            self.engine.set_trigger_mode('GATE_HYBRID', auto_update_shift=True)
            print("[Sesión] Perfil Lucas activado en modo cinemático autónomo (GATE_HYBRID).")
        self.reset_test_score()

    def seleccionar_sesion_wav(self):
        """Permite al usuario elegir otra carpeta de sesión grabada."""
        base_dir = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos")
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta de Sesión", base_dir)
        if folder:
            self.ruta_sesion_wav = folder
            self.lbl_sesion_actual.setText(f"Sesión: {os.path.basename(folder)}")
            self._cargar_metadata_sesion(folder)

    def recalibrar_silencio(self):
        """Fuerza la recalibración del reposo facial del usuario y del piso de ruido del Gate Doble."""
        self._forzar_recalibracion_ruido()
        res = self.engine.recalibrar_ruido(duracion_sec=5.0)
        if res:
            t = self.engine.get_telemetry()
            self.lbl_u_bajo.setText(f"U_bajo: {t['u_bajo']:.1f}")
            self.lbl_u_alto.setText(f"U_alto: {t['u_alto']:.1f}")
            self.lbl_badge_estado.setText(f"GATE: {t['estado']}")

    def _forzar_recalibracion_ruido(self):
        """Reinicia la ventana de 3 segundos para grabar el ruido basal del rostro."""
        if hasattr(self, "btn_recalibrar_reposo") and self.btn_recalibrar_reposo is not None:
            self.btn_recalibrar_reposo.setText("CALIBRANDO... (3s)")
            self.btn_recalibrar_reposo.setStyleSheet("background-color: #00ADB5; color: #0B0C10; font-weight: 900; font-size: 11px; padding: 5px; border-radius: 4px;")
            QtCore.QTimer.singleShot(3200, self._restaurar_boton_calibrar)

        if hasattr(self, "acquisition_thread") and self.acquisition_thread is not None:
            if hasattr(self.acquisition_thread, "forzar_recalibracion_facial"):
                self.acquisition_thread.forzar_recalibracion_facial()
                print("[RUIDO] Iniciando nueva grabación de ruido basal (3 segundos, quédate quieto)...")
        if hasattr(self, "lbl_badge_rostro") and self.lbl_badge_rostro is not None:
            self.lbl_badge_rostro.setText("CALIBRANDO RUIDO BASAL (3s)...")
            self.lbl_badge_rostro.setStyleSheet(
                "background-color: #3730A3; color: #A5B4FC; font-weight: bold; font-size: 11px; border-radius: 4px; padding: 4px;"
            )

    def _restaurar_boton_calibrar(self):
        """Restaura el texto y estilo del botón de calibración tras concluir los 3 segundos."""
        if hasattr(self, "btn_recalibrar_reposo") and self.btn_recalibrar_reposo is not None:
            self.btn_recalibrar_reposo.setText("Grabar Ruido / Calibrar Reposo (3s)")
            self.btn_recalibrar_reposo.setStyleSheet("""
                QPushButton {
                    background-color: #2B3A42; color: #66FCF1; font-weight: bold; font-size: 11px; padding: 5px; border-radius: 4px; border: 1px solid #45A29E;
                }
                QPushButton:hover {
                    background-color: #3A4F59; color: #FFFFFF; border-color: #66FCF1;
                }
                QPushButton:pressed {
                    background-color: #00ADB5; color: #0B0C10; font-weight: 900;
                }
            """)

    def actualizar_ciclo(self):
        """Bucle principal de lectura de cola, actualización de gráficos e inferencia."""
        chunks = []
        while not self.data_queue.empty():
            try:
                chunks.append(self.data_queue.get_nowait())
            except queue.Empty:
                break

        if not chunks:
            return

        # Concatenar lote recibido: forma (4, n_muestras)
        data_block = np.concatenate(chunks, axis=1)
        n_canales_recibidos, n_muestras = data_block.shape
        ch_to_plot = min(n_canales_recibidos, self.NUM_CANALES)

        # 1. Ingesta estricta de los 6 canales cinemáticos en el motor
        # Tarea E.1: En modo Gate Doble sin micrófono habilitado, forzar Canal 6 a 0.0 absoluto
        semg_block = data_block[:6]
        mic_enabled = bool(
            self.engine.mic_sync_enabled
            or (getattr(self, "chk_habilitar_microfono", None) and self.chk_habilitar_microfono.isChecked())
        )
        if not mic_enabled and n_canales_recibidos >= 7:
            data_block[6, :] = 0.0
        mic_block = data_block[6] if (n_canales_recibidos >= 7 and mic_enabled) else None
        eventos, s_tail, telemetria = self.engine.push_chunk(semg_block, mic_chunk=mic_block)

        # 2. Renderizado en el osciloscopio de señales con Notch 50 Hz, envolventes y norma S_emg
        is_rms = self.chk_rms_env.isChecked()
        win_size_ms = self.spin_rms_window.value()
        win_size = max(1, int((win_size_ms / 1000.0) * self.SAMPLE_RATE))
        t_vec = np.linspace(-self.PLOT_DURATION_S, 0, self.PLOT_SAMPLES)

        # Canales 0, 1, 2: Músculos sEMG pasados por filtro Notch 50 Hz en tiempo real
        for i in range(6):
            if is_rms:
                valid_data = np.nan_to_num(self.engine.env_buffer[i])
            else:
                valid_data = np.nan_to_num(self.engine.filt_buffer[i])
            x_dec, y_dec = decimate_min_max(t_vec, valid_data, max_points=3000)
            self.curvas_emg[i].setData(x_dec, y_dec)

        # Canal 3: Micrófono (Canal acústico)
        valid_mic = np.zeros(self.PLOT_SAMPLES, dtype=np.float32)
        if ch_to_plot >= 7:
            valid_mic = np.nan_to_num(self.engine.mic_buffer)
            if is_rms:
                valid_mic = calculate_rms_envelope(valid_mic, win_size)
            x_dec, y_dec = decimate_min_max(t_vec, valid_mic, max_points=3000)
            self.curvas_emg[6].setData(x_dec, y_dec)

        # Curva 4: S_emg (Norma Tricanal Combinada del Gate Doble)
        valid_s = np.nan_to_num(self.engine.s_emg_buffer)
        x_dec, y_dec = decimate_min_max(t_vec, valid_s, max_points=3000)
        self.curva_semg_norma.setData(x_dec, y_dec)

        # Actualizar líneas de umbrales Gate Doble sobre la escala de la norma S_emg
        u_bajo = telemetria.get('u_bajo', 0.0)
        u_alto = telemetria.get('u_alto', 0.0)
        if u_bajo > 0:
            self.peak_th_line_pos.setValue(u_bajo)
        if u_alto > 0:
            self.trigger_line.setValue(u_alto)

        # 3. Auto-escala dinámica sobre canales visibles (incluyendo S_emg)
        if getattr(self, 'chk_autoescala', None) and self.chk_autoescala.isChecked():
            cands = []
            for idx_ch, chk in enumerate(self.viz_checkboxes):
                if chk.isChecked():
                    if idx_ch < 6:
                        d = self.engine.env_buffer[idx_ch] if is_rms else np.abs(self.engine.filt_buffer[idx_ch])
                        cands.append(float(np.max(d)))
                    elif idx_ch == 6 and ch_to_plot >= 7:
                        d = valid_mic if is_rms else np.abs(self.engine.mic_buffer)
                        cands.append(float(np.max(d)))
                    elif idx_ch == 7:
                        cands.append(float(np.max(valid_s)))

            if cands:
                current_max = max(cands)
                target_max = current_max * 1.25 if current_max > 0.5 else 15.0
                if target_max > self._held_plot_ymax:
                    self._held_plot_ymax = target_max
                else:
                    self._held_plot_ymax = max(target_max, self._held_plot_ymax * 0.96)
                self._held_plot_ymax = max(3.0, self._held_plot_ymax)

                if is_rms:
                    self.plot.setYRange(0.0, self._held_plot_ymax, padding=0.02)
                else:
                    self.plot.setYRange(-self._held_plot_ymax, self._held_plot_ymax, padding=0.02)

        # Actualizar telemetría visual de Gate Doble
        estado = telemetria['estado']
        if estado == 'ARMADO':
            col_badge = "#00FF66"
            txt_col = "#000"
        elif estado == 'ESPERANDO_FIN_VENTANA':
            col_badge = "#00FFFF"
            txt_col = "#000"
        elif estado == 'REFRACTARIO':
            col_badge = "#FF9900"
            txt_col = "#000"
        else:
            col_badge = "#FFDD00"
            txt_col = "#000"

        self.lbl_badge_estado.setText(f"GATE: {estado}")
        self.lbl_badge_estado.setStyleSheet(f"background-color: {col_badge}; color: {txt_col}; font-weight: bold; padding: 4px; border-radius: 4px;")
        self.lbl_u_bajo.setText(f"U_bajo: {telemetria['u_bajo']:.1f}")
        self.lbl_u_alto.setText(f"U_alto: {telemetria['u_alto']:.1f}")

        # Desplazar solidariamente en el osciloscopio la ventana sombreada [t_inicio, t_fin] del último evento
        if getattr(self, '_last_event_sample_span', None) is not None:
            on_s, cen_s, w_ini_s, w_fin_s, _ = self._last_event_sample_span
            tot_s = self.engine.total_samples_received
            t_on_p = -(tot_s - on_s) / self.SAMPLE_RATE
            t_cen_p = -(tot_s - cen_s) / self.SAMPLE_RATE
            t_wi_p = -(tot_s - w_ini_s) / self.SAMPLE_RATE
            t_wf_p = -(tot_s - w_fin_s) / self.SAMPLE_RATE
            if t_wf_p >= -self.PLOT_DURATION_S:
                self.marker_onset.setValue(max(-self.PLOT_DURATION_S, t_on_p))
                self.marker_centro.setValue(max(-self.PLOT_DURATION_S, t_cen_p))
                if hasattr(self, 'region_ventana'):
                    self.region_ventana.setRegion((max(-self.PLOT_DURATION_S, t_wi_p), min(0.0, t_wf_p)))
            else:
                if hasattr(self, 'region_ventana'):
                    self.region_ventana.setRegion((-10.0, -9.8))

        # 3. Procesar eventos decodificados si hubo fonación confirmada
        if eventos:
            for ev in eventos:
                self._mostrar_evento_decodificado(ev)

    def _mostrar_evento_decodificado(self, ev):
        """Actualiza el cartel gigante, marcas temporales EMD, marcador en vivo y mapa latente."""
        vocal = ev['vocal']
        color_hex = ev['color']
        confianza = ev['confianza']
        z1 = ev['z1']
        z2 = ev['z2']
        z3 = float(ev.get('z3', 0.0))
        z4 = float(ev.get('z4', 0.0))
        latencia_infer = ev['latencia_infer_ms']
        latencia_total = ev['latencia_total_ms']
        onset_s = ev.get('onset_sample', 0)
        center_s = ev.get('center_sample', 0)

        # Cartel gigante
        if vocal == '---':
            self.lbl_vocal_gigante.setText("---")
            self.lbl_vocal_gigante.setStyleSheet("""
                font-size: 78px;
                font-weight: 900;
                color: #66FCF1;
                background-color: #0B0C10;
                border: 3px solid #66FCF1;
                border-radius: 10px;
                padding: 8px;
            """)
            self.lbl_confianza.setText("Esperando contracción fonatoria o modelo no cargado...")
            self.bar_confianza.setValue(0)
            return

        self.lbl_vocal_gigante.setText(f"/{vocal}/")
        self.lbl_vocal_gigante.setStyleSheet(f"""
            font-size: 82px;
            font-weight: 900;
            color: {color_hex};
            background-color: #0B0C10;
            border: 4px solid {color_hex};
            border-radius: 12px;
            padding: 8px;
        """)

        # Confianza y latencia
        self.lbl_confianza.setText(f"Vocal /{vocal}/ | Confianza: {confianza:.1f}% | Inferencia: {latencia_infer:.2f} ms | Total: {latencia_total:.1f} ms")
        self.bar_confianza.setValue(int(confianza))
        self.bar_confianza.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {color_hex};
                border-radius: 4px;
                text-align: center;
                background-color: #1F2833;
                color: white;
                font-weight: bold;
                height: 18px;
            }}
            QProgressBar::chunk {{
                background-color: {color_hex};
            }}
        """)

        # Desglose por vocal
        for v in VOCALES:
            p = ev['probs'].get(v, 0.0)
            self.lbl_probs[v].setText(f"/{v}/: {p:.0f}%")
            if v == vocal:
                self.lbl_probs[v].setStyleSheet(f"color: white; background-color: {VOCAL_A_COLOR_HEX[v]}; font-weight: bold; font-size: 12px; border-radius: 4px; padding: 2px;")
            else:
                self.lbl_probs[v].setStyleSheet(f"color: {VOCAL_A_COLOR_HEX[v]}; font-weight: bold; font-size: 11px; border: 1px solid {VOCAL_A_COLOR_HEX[v]}; border-radius: 4px; padding: 2px;")

        # Actualizar Marcador en Vivo para grabación de prueba
        if self.test_ground_truth and self.test_event_count < len(self.test_ground_truth):
            ground_v = self.test_ground_truth[self.test_event_count].upper()
            pred_v = vocal.upper()
            is_ok = (pred_v == ground_v)
            self.test_event_count += 1
            if is_ok:
                self.test_correct_count += 1
            if ground_v in self.test_vocal_stats:
                self.test_vocal_stats[ground_v]['total'] += 1
                if is_ok:
                    self.test_vocal_stats[ground_v]['correct'] += 1

            acc = (self.test_correct_count / self.test_event_count) * 100.0
            self.lbl_score_total.setText(f"Aciertos: {self.test_correct_count} / {self.test_event_count} ({acc:.1f}%)")
            self.bar_score.setValue(int(acc))

            for v_k in VOCALES:
                st = self.test_vocal_stats[v_k]
                v_acc = (st['correct'] / st['total'] * 100) if st['total'] > 0 else 0
                self.lbl_vocal_scores[v_k].setText(f"/{v_k}/: {st['correct']}/{st['total']} ({v_acc:.0f}%)")

            status_str = "CORRECTO" if is_ok else "ERROR"
            col_status = "#39FF14" if is_ok else "#FF0055"
            self.lbl_last_comp.setText(
                f"[#{self.test_event_count:02d}] Pred: /{pred_v}/ | Real: /{ground_v}/ -> <span style='color:{col_status}; font-weight:bold;'>{status_str}</span> (conf: {confianza:.1f}%)"
            )

        # Guardar último evento para auditoría por teclado y actualizar marcas y región [t_inicio, t_fin]
        self.ultimo_evento_decodificado = ev
        tot_samples = self.engine.total_samples_received
        win_start_s = int(ev.get('win_start_sample', center_s - int(0.25 * self.SAMPLE_RATE)))
        win_end_s = int(ev.get('win_end_sample', center_s + int(0.35 * self.SAMPLE_RATE)))
        dur_ms = float(ev.get('duracion_ventana_ms', (win_end_s - win_start_s) / self.SAMPLE_RATE * 1000.0))
        self._last_event_sample_span = (onset_s, center_s, win_start_s, win_end_s, dur_ms)

        t_onset_plot = -(tot_samples - onset_s) / self.SAMPLE_RATE
        t_centro_plot = -(tot_samples - center_s) / self.SAMPLE_RATE
        t_win_ini_plot = -(tot_samples - win_start_s) / self.SAMPLE_RATE
        t_win_fin_plot = -(tot_samples - win_end_s) / self.SAMPLE_RATE

        if t_onset_plot >= -self.PLOT_DURATION_S:
            self.marker_onset.setValue(t_onset_plot)
        if t_centro_plot >= -self.PLOT_DURATION_S:
            self.marker_centro.setValue(t_centro_plot)
        if hasattr(self, 'region_ventana'):
            self.region_ventana.setRegion((max(-self.PLOT_DURATION_S, t_win_ini_plot), min(0.0, t_win_fin_plot)))
        if hasattr(self, 'lbl_ventana_info'):
            t_ini_abs = win_start_s / self.SAMPLE_RATE
            t_fin_abs = win_end_s / self.SAMPLE_RATE
            self.lbl_ventana_info.setText(
                f"Ventana [{t_ini_abs:.2f}s, {t_fin_abs:.2f}s] (plot: [{t_win_ini_plot:.2f}s, {t_win_fin_plot:.2f}s]) | Duración: {dur_ms:.0f} ms"
            )

        # Mostrar brevemente el overlay flotante AutoForge
        self.autoforge_overlay.setText(f"/{vocal}/ ({confianza:.0f}%)")
        self.autoforge_overlay.setStyleSheet(
            f"background-color: rgba(10, 5, 20, 190); color: {color_hex}; font-family: 'Courier New', monospace; "
            f"font-size: 55px; font-weight: 900; border: 3px solid {color_hex}; border-radius: 8px; padding: 10px;"
        )
        self.autoforge_overlay.show()
        self.overlay_timer.start(750)  # Ocultar a los 750 ms

        # Actualizar mapa latente 2D y nube 3D interactiva
        self.historial_puntos.append((z1, z2, z3, z4, vocal, time.time()))
        if len(self.historial_puntos) > self.MAX_HISTORIAL:
            self.historial_puntos.pop(0)

        # Dibujar rastro animado y punto actual en 2D respetando el plano activo
        self._redibujar_puntos_2d()

        # Actualizar nube 3D interactiva
        self._actualizar_grafico_3d()

    def _ocultar_overlay(self):
        """Oculta el cartel flotante AutoForge al concluir el pulso."""
        self.autoforge_overlay.hide()

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        """
        Atajos de teclado para validación y auditoría en vivo (Tarea D):
          - Espacio o V : Acierto (True Positive)
          - C o M       : Vocal Incorrecta / Confusión
          - X o F       : Falso Disparo de Ventana (False Positive / Espurio en silencio)
        """
        key = event.key()
        if key in (QtCore.Qt.Key_Space, QtCore.Qt.Key_V):
            self.registrar_auditoria_en_vivo("acierto")
            event.accept()
            return
        elif key in (QtCore.Qt.Key_C, QtCore.Qt.Key_M):
            self.registrar_auditoria_en_vivo("confusion")
            event.accept()
            return
        elif key in (QtCore.Qt.Key_X, QtCore.Qt.Key_F):
            self.registrar_auditoria_en_vivo("falsa_ventana")
            event.accept()
            return
        super().keyPressEvent(event)

    def registrar_auditoria_en_vivo(self, tipo_evento: str):
        """
        Registra de inmediato en auditoria_detecciones_tiempo_real.csv la validación del usuario
        ('acierto', 'confusion' o 'falsa_ventana'). Si es 'falsa_ventana', guarda el buffer crudo
        en la carpeta falsos_positivos_gate/ y ajusta el umbral del Gate Doble para purgar disparos espurios.
        """
        import csv as _csv
        ev = self.ultimo_evento_decodificado
        now_s = time.time()
        dt_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        if ev is not None:
            v_pred = str(ev.get('vocal', '---'))
            prob_lat = float(ev.get('confianza', 0.0))
            z1, z2, z3 = float(ev.get('z1', 0.0)), float(ev.get('z2', 0.0)), float(ev.get('z3', 0.0))
            dur_ms = float(ev.get('duracion_ventana_ms', 0.0))
            w_ini_s = float(ev.get('win_start_sample', 0) / self.SAMPLE_RATE)
            w_fin_s = float(ev.get('win_end_sample', 0) / self.SAMPLE_RATE)
            raw_6ch = ev.get('raw_window_6ch', self.engine.env_buffer[:, -100:].copy())
            x_150 = ev.get('x_150', np.zeros(150, dtype=np.float32))
        else:
            v_pred = "---"
            prob_lat = 0.0
            z1, z2, z3 = 0.0, 0.0, 0.0
            dur_ms = 600.0
            tot_s = self.engine.total_samples_received / self.SAMPLE_RATE
            w_ini_s = max(0.0, tot_s - 0.6)
            w_fin_s = tot_s
            raw_6ch = self.engine.env_buffer[:, -100:].copy()
            x_150 = np.zeros(150, dtype=np.float32)

        # Directorio de auditoría (en la sesión actual si existe, o en project_root/auditoria_tiempo_real)
        out_base = self.ruta_sesion_wav if (self.ruta_sesion_wav and os.path.isdir(self.ruta_sesion_wav)) else os.path.join(project_root, "auditoria_tiempo_real")
        os.makedirs(out_base, exist_ok=True)
        csv_aud_path = os.path.join(out_base, "auditoria_detecciones_tiempo_real.csv")
        es_nuevo = not os.path.exists(csv_aud_path)

        archivo_fp_rel = ""
        if tipo_evento == "falsa_ventana":
            fp_dir = os.path.join(out_base, "falsos_positivos_gate")
            os.makedirs(fp_dir, exist_ok=True)
            ts_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fp_filename = f"falso_positivo_{ts_id}_pred_{v_pred}.npz"
            fp_path = os.path.join(fp_dir, fp_filename)
            try:
                np.savez(
                    fp_path,
                    timestamp_s=now_s,
                    vocal_predicha=v_pred,
                    probabilidad_latente=prob_lat,
                    raw_window_6ch=raw_6ch,
                    x_150=x_150,
                    s_emg_recent=self.engine.s_emg_buffer[-200:].copy(),
                    u_bajo=self.engine.u_bajo,
                    u_alto=self.engine.u_alto
                )
                archivo_fp_rel = f"falsos_positivos_gate/{fp_filename}"
            except Exception as e_fp:
                print(f"[Auditoría Error] No se pudo guardar buffer de falso positivo: {e_fp}")

            # Purgar/elevar suavemente los umbrales del Gate Doble para evitar que se repita el disparo espurio
            u_b, u_a = self.engine.ajustar_umbral_por_falso_positivo(1.08)
            self.lbl_u_bajo.setText(f"U_bajo: {u_b:.2f}")
            self.lbl_u_alto.setText(f"U_alto: {u_a:.2f}")

        try:
            with open(csv_aud_path, "a", encoding="utf-8", newline="") as fh:
                writer = _csv.writer(fh)
                if es_nuevo:
                    writer.writerow([
                        "timestamp_s", "fecha_hora", "vocal_predicha", "probabilidad_latente",
                        "tipo_evento", "t_inicio_s", "t_fin_s", "duracion_ventana_ms",
                        "z1", "z2", "z3", "u_bajo", "u_alto", "archivo_buffer_crudo"
                    ])
                writer.writerow([
                    f"{now_s:.3f}", dt_iso, v_pred, f"{prob_lat:.2f}",
                    tipo_evento, f"{w_ini_s:.3f}", f"{w_fin_s:.3f}", f"{dur_ms:.1f}",
                    f"{z1:.4f}", f"{z2:.4f}", f"{z3:.4f}",
                    f"{self.engine.u_bajo:.3f}", f"{self.engine.u_alto:.3f}", archivo_fp_rel
                ])
        except Exception as e_csv:
            print(f"[Auditoría Error] Falló escritura en {csv_aud_path}: {e_csv}")

        # Feedback visual en el video / overlay y en el badge de auditoría
        if tipo_evento == "acierto":
            msg_ui = f"✓ ACIERTO REGISTRADO: /{v_pred}/ ({prob_lat:.0f}%)"
            col_bg, col_fg, col_bd = "#064E3B", "#34D399", "#10B981"
            ov_txt = f"✓ ACIERTO /{v_pred}/"
        elif tipo_evento == "confusion":
            msg_ui = f"⚠ CONFUSIÓN REGISTRADA: Predijo /{v_pred}/ ({prob_lat:.0f}%)"
            col_bg, col_fg, col_bd = "#78350F", "#FBBF24", "#F59E0B"
            ov_txt = f"⚠ VOCAL ERRÓNEA (/{v_pred}/)"
        else:
            msg_ui = f"✖ FALSA VENTANA REGISTRADA (Buffer guardado | Nuevo U_alto={self.engine.u_alto:.2f})"
            col_bg, col_fg, col_bd = "#7F1D1D", "#FCA5A5", "#EF4444"
            ov_txt = "✖ FALSA VENTANA"
            if hasattr(self, 'lbl_badge_rostro') and self.lbl_badge_rostro is not None:
                self.lbl_badge_rostro.setText("ALERTA: FALSA VENTANA REPORTADA")
                self.lbl_badge_rostro.setStyleSheet(
                    "background-color: #7F1D1D; color: #FFFFFF; font-weight: bold; font-size: 11px; border: 2px solid #EF4444; border-radius: 4px; padding: 4px;"
                )

        if hasattr(self, 'lbl_auditoria_status'):
            self.lbl_auditoria_status.setText(msg_ui)
            self.lbl_auditoria_status.setStyleSheet(
                f"background-color: {col_bg}; color: {col_fg}; border: 1px solid {col_bd}; font-size: 11px; font-weight: bold; padding: 3px; border-radius: 3px;"
            )
        self.autoforge_overlay.setText(ov_txt)
        self.autoforge_overlay.setStyleSheet(
            f"background-color: rgba(15, 10, 20, 210); color: {col_fg}; font-family: 'Courier New', monospace; "
            f"font-size: 44px; font-weight: 900; border: 3px solid {col_bd}; border-radius: 8px; padding: 10px;"
        )
        self.autoforge_overlay.show()
        self.overlay_timer.start(950)
        print(f"[AUDITORÍA EN VIVO] {msg_ui} -> {csv_aud_path}")

    def closeEvent(self, event):
        """Cierre ordenado de los hilos."""
        self.stop_event.set()
        if self.acquisition_thread:
            if hasattr(self.acquisition_thread, 'wait'):
                self.acquisition_thread.wait(1500)
            elif hasattr(self.acquisition_thread, 'is_alive') and self.acquisition_thread.is_alive():
                self.acquisition_thread.join(timeout=1.0)
        event.accept()


# ==============================================================================
# 3. PUNTO DE ENTRADA PRINCIPAL
# ==============================================================================
def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor('#0B0C10'))
    palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor('#E0E0E0'))
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor('#1F2833'))
    palette.setColor(QtGui.QPalette.Text, QtGui.QColor('#FFFFFF'))
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor('#1F2833'))
    palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor('#FFFFFF'))
    app.setPalette(palette)

    gui = RealTimeDecoderPlotter()
    gui.showMaximized()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
