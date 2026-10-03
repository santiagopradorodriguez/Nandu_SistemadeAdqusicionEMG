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

# Asegurar path del proyecto
project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QComboBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QProgressBar, QSplitter, QFileDialog, QFrame,
    QStackedWidget, QButtonGroup
)
import pyqtgraph as pg
from pyqtgraph import InfiniteLine

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

# Importar motor de decodificación en tiempo real
from EMG_desarrollo.deep_learning.realtime_decoder_engine import (
    RealtimeDecoderEngine, VOCALES, VOCAL_A_COLOR_HEX, VOCAL_A_RGBA
)

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

# Modelos disponibles en el repositorio
MODELOS_PRESETS = [
    {
        'nombre': "Campeón MLP 3D sin SO2 - 87.21% Media Armónica",
        'subtitulo': "Récord: 87.4% Lucas | 87.0% P5 (107/123) | Ventana 40/60",
        'path': os.path.join(
            project_root,
            "EMG_desarrollo/resultados/grid_search_lucas_ventana4060/mlp_3d_sin_so2/campeon_mlp_3d_sin_so2_zona_dulce_completo.pt"
        )
    },
    {
        'nombre': "Campeón Convolucional Ortogonal 2D - 85.32% Armónica",
        'subtitulo': "87.8% Lucas | 82.9% P5 | Ventana 50/50",
        'path': os.path.join(
            project_root,
            "EMG_desarrollo/resultados/grid_search_conv_ortogonal/modelo_campeon_conv_ortogonal.pt"
        )
    },
    {
        'nombre': "Autoencoder Ortogonal 2D Nativo - modelo_optimo.pt",
        'subtitulo': "87.85% Lucas | 81.30% P5 con EMD | Densa Lineal",
        'path': os.path.join(
            project_root,
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/modelo_optimo.pt"
        )
    },
    {
        'nombre': "Modelo Semisupervisado 2D - Apunte LaTeX",
        'subtitulo': "Entropía Cruzada lambda=0.5 | Hito 115",
        'path': os.path.join(
            project_root,
            "EMG_desarrollo/resultados/grid_search_conv_ortogonal/modelo_campeon_semisupervisado_87.pt"
        )
    }
]

NOMBRES_CANALES = [
    "CH0: Vientre anterior",
    "CH1: Depresor Anguli Oris",
    "CH2: Orbicularis Oris",
    "CH3: Micrófono"
]
COLORES_CANALES = ["#FFAA00", "#39FF14", "#FFFF00", "#00FFFF"]


# ==============================================================================
# 1. HILOS DE ADQUISICIÓN Y REPRODUCCIÓN (DAQ / SIM / WAV)
# ==============================================================================
def nidaq_acquisition_thread(device_channels, sample_rate, chunk_samples, num_canales, data_queue, stop_event, terminal_config_val=None):
    """Hilo de adquisición de hardware físico con NI-DAQmx."""
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

        self.SAMPLE_RATE = 2000
        self.NUM_CANALES = 4  # CH0, CH1, CH2 (sEMG) + CH3 (Micrófono)
        self.PLOT_DURATION_S = 8
        self.PLOT_SAMPLES = self.SAMPLE_RATE * self.PLOT_DURATION_S
        self.plot_buffer = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES), dtype=np.float32)

        # Envolvente RMS en tiempo real (idéntico a AutoForge DAQ)
        self.env_buffer = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES), dtype=np.float64)
        self._held_plot_ymax = 500.0

        # Historial de puntos decodificados en el mapa latente (hasta 150 para cubrir sesiones completas)
        self.historial_puntos = []  # Tuplas (z1, z2, vocal, timestamp)
        self.MAX_HISTORIAL = 150

        # Marcador y conteo de aciertos para grabación de prueba
        self.test_ground_truth = []
        self.test_event_count = 0
        self.test_correct_count = 0
        self.test_vocal_stats = {v: {'correct': 0, 'total': 0} for v in VOCALES}

        # Motor de decodificación inicializado con el modelo campeón (87.21% armónica)
        pt_model_default = MODELOS_PRESETS[0]['path']
        if not os.path.exists(pt_model_default):
            pt_model_default = MODELOS_PRESETS[1]['path']

        self.engine = RealtimeDecoderEngine(
            model_path=pt_model_default,
            sample_rate=self.SAMPLE_RATE,
            emd_delay_ms=350
        )

        # Sesión WAV de prueba por defecto (P5)
        self.ruta_sesion_wav = os.path.join(
            project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-06-10/SecuenciaContinua_Prueba5_Sujeto1"
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
        """Construye la distribución general de la ventana."""
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_layout.setSpacing(6)

        # Splitter horizontal: Panel DAQ (Izquierda) vs Panel Decodificador (Derecha)
        self.splitter_principal = QSplitter(QtCore.Qt.Horizontal)

        # 1. Panel DAQ (AutoForge)
        self.widget_daq = QWidget()
        self.layout_daq = QVBoxLayout(self.widget_daq)
        self.layout_daq.setContentsMargins(4, 4, 4, 4)
        self.layout_daq.setSpacing(4)
        self._build_daq_panel()

        # 2. Panel Decodificador en Tiempo Real
        self.widget_decodificador = QWidget()
        self.layout_decodificador = QVBoxLayout(self.widget_decodificador)
        self.layout_decodificador.setContentsMargins(4, 4, 4, 4)
        self.layout_decodificador.setSpacing(6)
        self._build_decoder_panel()

        # Agregar al splitter
        self.splitter_principal.addWidget(self.widget_daq)
        self.splitter_principal.addWidget(self.widget_decodificador)
        self.splitter_principal.setSizes([880, 680])

        self.main_layout.addWidget(self.splitter_principal)

    def _build_daq_panel(self):
        """Construye el osciloscopio multicanal y controles idénticos a autoforge_daq_experimental."""
        grp_ctrl = QGroupBox("Control de Adquisición sEMG")
        lay_ctrl = QGridLayout(grp_ctrl)
        lay_ctrl.setContentsMargins(6, 6, 6, 6)
        lay_ctrl.setSpacing(6)

        self.btn_iniciar_daq = QPushButton("Iniciar Hardware DAQ")
        self.btn_iniciar_daq.setStyleSheet(self.BTN_START_STYLE)
        self.btn_iniciar_daq.clicked.connect(self.toggle_daq)

        self.btn_play_wav = QPushButton("Reproducir WAV en Vivo")
        self.btn_play_wav.setStyleSheet(self.BTN_PLAY_STYLE)
        self.btn_play_wav.clicked.connect(self.toggle_wav_playback)

        self.btn_cargar_wav = QPushButton("Cargar Sesión WAV...")
        self.btn_cargar_wav.clicked.connect(self.seleccionar_sesion_wav)

        self.lbl_sesion_actual = QLabel(f"Sesión: {os.path.basename(self.ruta_sesion_wav)}")
        self.lbl_sesion_actual.setStyleSheet("color: #AAAAAA; font-size: 11px;")

        self.chk_modo_simulado = QCheckBox("Modo Señal Sintética")

        lay_ctrl.addWidget(self.btn_iniciar_daq, 0, 0)
        lay_ctrl.addWidget(self.btn_play_wav, 0, 1)
        lay_ctrl.addWidget(self.btn_cargar_wav, 0, 2)
        lay_ctrl.addWidget(self.chk_modo_simulado, 1, 0)
        lay_ctrl.addWidget(self.lbl_sesion_actual, 1, 1, 1, 2)

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

        # Checkboxes dinámicos por canal
        self.viz_checkbox_layout = QtWidgets.QHBoxLayout()
        self.viz_checkbox_layout.setSpacing(8)
        self.viz_control_layout.addLayout(self.viz_checkbox_layout)
        self.viz_control_layout.addStretch()

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
        self.plot.showGrid(x=True, y=True, alpha=0.3)
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
            self.curvas_emg.append(curva)

            chk = QtWidgets.QCheckBox(nombre)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {col}; font-weight: bold; border: none; font-size: 11px;")

            def toggle_viz(state, idx=i):
                if idx < len(self.curvas_emg):
                    self.curvas_emg[idx].setVisible(state)

            chk.toggled.connect(toggle_viz)
            self.viz_checkbox_layout.addWidget(chk)
            self.viz_checkboxes.append(chk)

        # Curva 5: Norma Tricanal Combinada S_emg (Gate Doble)
        self.curva_semg_norma = self.plot.plot(pen=pg.mkPen('#FF00FF', width=2.0), name="S_emg: Norma")
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

        self.lbl_modelo_info = QLabel(MODELOS_PRESETS[0]['subtitulo'])
        self.lbl_modelo_info.setStyleSheet("color: #66FCF1; font-size: 11px; font-weight: bold;")
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
        self.cmb_proyeccion_2d.addItems(["Z1 - Z2", "Z1 - Z3", "Z2 - Z3"])
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
        self.plot_latente.showGrid(x=True, y=True, alpha=0.3)
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
        self.spin_emd_delay.setValue(350)
        self.spin_emd_delay.setSuffix(" ms")
        self.spin_emd_delay.setSingleStep(10)
        self.spin_emd_delay.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 3px;")
        self.spin_emd_delay.setToolTip(
            "Retardo electromecánico fisiológico del músculo respecto al audio: proyecta el centro de la ventana a n_onset + EMD_lag."
        )
        self.spin_emd_delay.valueChanged.connect(self._on_emd_delay_changed)

        self.lbl_emd_info = QLabel("Lag Fisiológico: +350 ms")
        self.lbl_emd_info.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 11px;")

        self.btn_recalibrar = QPushButton("Recalibrar Silencio")
        self.btn_recalibrar.clicked.connect(self.recalibrar_silencio)

        lay_gate.addWidget(self.lbl_badge_estado, 0, 0)
        lay_gate.addWidget(self.lbl_u_bajo, 0, 1)
        lay_gate.addWidget(self.lbl_u_alto, 0, 2)

        lay_gate.addWidget(lbl_spin_emd, 1, 0)
        lay_gate.addWidget(self.spin_emd_delay, 1, 1)
        lay_gate.addWidget(self.lbl_emd_info, 1, 2)

        lay_gate.addWidget(self.btn_recalibrar, 2, 0, 1, 3)

        self.layout_decodificador.addWidget(self.grp_gate)

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
                hv = pt[3] if len(pt) > 4 else pt[2]
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
            uv = ultimo[3] if len(ultimo) > 4 else ultimo[2]
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
        """Conmuta los ejes de proyección cartesianos para la visualización del plano 2D."""
        if idx == 1:
            self.ejes_proyeccion_2d = (0, 2)
            lbl_x, lbl_y = "Coordenada Z1", "Coordenada Z3"
        elif idx == 2:
            self.ejes_proyeccion_2d = (1, 2)
            lbl_x, lbl_y = "Coordenada Z2", "Coordenada Z3"
        else:
            self.ejes_proyeccion_2d = (0, 1)
            lbl_x, lbl_y = "Coordenada Z1", "Coordenada Z2"

        self.plot_latente.setLabel('bottom', lbl_x)
        self.plot_latente.setLabel('left', lbl_y)
        self._cargar_mapa_fronteras_decision()
        self._redibujar_puntos_2d()

    def _redibujar_puntos_2d(self):
        """Redibuja el rastro de puntos fonatorios en el plano 2D seleccionado."""
        ax_x, ax_y = getattr(self, 'ejes_proyeccion_2d', (0, 1))
        trail_spots = []
        for i, pt in enumerate(self.historial_puntos[:-1]):
            hz_x = pt[ax_x] if len(pt) > ax_x else pt[0]
            hz_y = pt[ax_y] if len(pt) > ax_y else pt[1]
            hv = pt[3] if len(pt) > 4 else pt[2]
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
            uz_x = ultimo[ax_x] if len(ultimo) > ax_x else ultimo[0]
            uz_y = ultimo[ax_y] if len(ultimo) > ax_y else ultimo[1]
            uv = ultimo[3] if len(ultimo) > 4 else ultimo[2]
            color_hex = VOCAL_A_COLOR_HEX.get(uv, '#FFFFFF')
            self.scatter_actual.setData([{
                'pos': (uz_x, uz_y),
                'size': 26,
                'symbol': 'star',
                'brush': pg.mkBrush('#FFFFFF'),
                'pen': pg.mkPen(color_hex, width=3)
            }])

    def _on_model_changed(self, idx):
        """Carga el modelo seleccionado en el desplegable y regenera las fronteras de decisión."""
        if idx < 0 or idx >= len(MODELOS_PRESETS):
            return
        preset = MODELOS_PRESETS[idx]
        p_pt = preset['path']
        if os.path.exists(p_pt):
            print(f"[Modelo] Conmutando a: {preset['nombre']}")
            self.engine.load_model(p_pt)
            self.lbl_modelo_info.setText(preset['subtitulo'])
            self._cargar_mapa_fronteras_decision()
            self.historial_puntos.clear()
            self.scatter_trail.clear()
            self.scatter_actual.clear()
            if self.engine.latent_dim == 3:
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
            print(f"[Modelo ERROR] Archivo no encontrado: {p_pt}")

    def _cargar_mapa_fronteras_decision(self):
        """Calcula y dibuja las regiones de decisión del GMM con zoom óptimo sobre los fonemas activos."""
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

    def toggle_daq(self):
        """Inicia o detiene la adquisición por hardware físico."""
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
            else:
                if not NIDAQMX_DISPONIBLE:
                    print("[DAQ ERROR] NI-DAQmx no disponible en este entorno.")
                    return
                device_channels = [f"Dev1/ai{i}" for i in range(self.NUM_CANALES)]
                self.acquisition_thread = threading.Thread(
                    target=nidaq_acquisition_thread,
                    args=(device_channels, self.SAMPLE_RATE, chunk_samples, self.NUM_CANALES, self.data_queue, self.stop_event),
                    daemon=True
                )

            self.acquisition_thread.start()
            self.is_acquiring = True
            self.btn_iniciar_daq.setText("Detener Hardware DAQ")
            self.btn_iniciar_daq.setStyleSheet(self.BTN_STOP_STYLE)
            self.engine.reset_state()
            self.historial_puntos.clear()
            self.scatter_trail.clear()
            self.scatter_actual.clear()
        else:
            self.stop_event.set()
            if self.acquisition_thread:
                self.acquisition_thread.join(timeout=1.0)
            self.is_acquiring = False
            self.btn_iniciar_daq.setText("Iniciar Hardware DAQ")
            self.btn_iniciar_daq.setStyleSheet(self.BTN_START_STYLE)

    def toggle_wav_playback(self):
        """Inicia o detiene la reproducción en tiempo real de los 4 canales de una sesión grabada."""
        if not self.is_playback:
            if self.is_acquiring:
                self.toggle_daq()

            wav_paths = [os.path.join(self.ruta_sesion_wav, f"canal_{ch}/grabacion.wav") for ch in range(self.NUM_CANALES)]
            if not all(os.path.exists(p) for p in wav_paths):
                print(f"[WAV ERROR] Faltan archivos en {self.ruta_sesion_wav}")
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

        # Calibración automática según la sesión
        nom = os.path.basename(folder).upper()
        if "P5" in nom or "PRUEBA5" in nom:
            self.engine.set_calibration_preset('P5')
            self.engine.set_mic_sync_enabled(True)
            self.spin_emd_delay.setValue(350)
            print("[Sesión] Perfil P5 (Secuencia Continua) activado con Mic-Sync y retardo EMD.")
        elif "LUCAS" in nom:
            self.engine.set_calibration_preset('LUCAS')
            self.engine.set_mic_sync_enabled(True)
            print("[Sesión] Perfil Lucas activado.")
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
        """Fuerza la recalibración del piso de ruido del Gate Doble con el buffer actual."""
        res = self.engine.recalibrar_ruido(duracion_sec=5.0)
        if res:
            t = self.engine.get_telemetry()
            self.lbl_u_bajo.setText(f"U_bajo: {t['u_bajo']:.1f}")
            self.lbl_u_alto.setText(f"U_alto: {t['u_alto']:.1f}")
            self.lbl_badge_estado.setText(f"GATE: {t['estado']}")

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

        # 1. Ingesta estricta de sEMG + Canal 3 (Micrófono) en el motor bioeléctrico
        # El motor procesa el lote con filtro Notch 50 Hz IIR causal y pasa-banda continuo
        semg_block = data_block[:3]
        mic_block = data_block[3] if n_canales_recibidos >= 4 else None
        eventos, s_tail, telemetria = self.engine.push_chunk(semg_block, mic_chunk=mic_block)

        # 2. Renderizado en el osciloscopio de señales con Notch 50 Hz, envolventes y norma S_emg
        is_rms = self.chk_rms_env.isChecked()
        win_size_ms = self.spin_rms_window.value()
        win_size = max(1, int((win_size_ms / 1000.0) * self.SAMPLE_RATE))
        t_vec = np.linspace(-self.PLOT_DURATION_S, 0, self.PLOT_SAMPLES)

        # Canales 0, 1, 2: Músculos sEMG pasados por filtro Notch 50 Hz en tiempo real
        for i in range(3):
            if is_rms:
                valid_data = np.nan_to_num(self.engine.env_buffer[i])
            else:
                valid_data = np.nan_to_num(self.engine.filt_buffer[i])
            x_dec, y_dec = decimate_min_max(t_vec, valid_data, max_points=3000)
            self.curvas_emg[i].setData(x_dec, y_dec)

        # Canal 3: Micrófono (Canal acústico)
        valid_mic = np.zeros(self.PLOT_SAMPLES, dtype=np.float32)
        if ch_to_plot >= 4:
            valid_mic = np.nan_to_num(self.engine.mic_buffer)
            if is_rms:
                valid_mic = calculate_rms_envelope(valid_mic, win_size)
            x_dec, y_dec = decimate_min_max(t_vec, valid_mic, max_points=3000)
            self.curvas_emg[3].setData(x_dec, y_dec)

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
                    if idx_ch < 3:
                        d = self.engine.env_buffer[idx_ch] if is_rms else np.abs(self.engine.filt_buffer[idx_ch])
                        cands.append(float(np.max(d)))
                    elif idx_ch == 3 and ch_to_plot >= 4:
                        d = valid_mic if is_rms else np.abs(self.engine.mic_buffer)
                        cands.append(float(np.max(d)))
                    elif idx_ch == 4:
                        cands.append(float(np.max(valid_s)))

            if cands:
                current_max = max(cands)
                target_max = current_max * 1.25 if current_max > 1.0 else 500.0
                if target_max > self._held_plot_ymax:
                    self._held_plot_ymax = target_max
                else:
                    self._held_plot_ymax = max(target_max, self._held_plot_ymax * 0.98)
                self._held_plot_ymax = max(50.0, self._held_plot_ymax)

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
        latencia_infer = ev['latencia_infer_ms']
        latencia_total = ev['latencia_total_ms']
        onset_s = ev.get('onset_sample', 0)
        center_s = ev.get('center_sample', 0)

        # Cartel gigante
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

        # Actualizar marcas temporales EMD en el osciloscopio
        tot_samples = self.engine.total_samples_received
        dist_onset = (tot_samples - onset_s) / self.SAMPLE_RATE
        dist_centro = (tot_samples - center_s) / self.SAMPLE_RATE
        t_onset_plot = -dist_onset
        t_centro_plot = -dist_centro

        if t_onset_plot >= -self.PLOT_DURATION_S:
            self.marker_onset.setValue(t_onset_plot)
        if t_centro_plot >= -self.PLOT_DURATION_S:
            self.marker_centro.setValue(t_centro_plot)

        # Mostrar brevemente el overlay flotante AutoForge
        self.autoforge_overlay.setText(f"/{vocal}/ ({confianza:.0f}%)")
        self.autoforge_overlay.setStyleSheet(
            f"background-color: rgba(10, 5, 20, 190); color: {color_hex}; font-family: 'Courier New', monospace; "
            f"font-size: 55px; font-weight: 900; border: 3px solid {color_hex}; border-radius: 8px; padding: 10px;"
        )
        self.autoforge_overlay.show()
        self.overlay_timer.start(750)  # Ocultar a los 750 ms

        # Actualizar mapa latente 2D y nube 3D interactiva
        self.historial_puntos.append((z1, z2, z3, vocal, time.time()))
        if len(self.historial_puntos) > self.MAX_HISTORIAL:
            self.historial_puntos.pop(0)

        # Dibujar rastro animado y punto actual en 2D respetando el plano activo
        self._redibujar_puntos_2d()

        # Actualizar nube 3D interactiva
        self._actualizar_grafico_3d()

    def _ocultar_overlay(self):
        """Oculta el cartel flotante AutoForge al concluir el pulso."""
        self.autoforge_overlay.hide()

    def closeEvent(self, event):
        """Cierre ordenado de los hilos."""
        self.stop_event.set()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
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
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
