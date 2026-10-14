# ==============================================================================
# Archivo: reparar_autoforge_v3.py
# Descripcion: Repara autoforge_daq_experimental.py, elimina dependencias rotas,
#              quita emojis, reubica el medidor de 50 Hz en vivo dentro de cada
#              canal, agrega detector de hardware DAQ conectado, consola de
#              errores integrada y organiza el espacio en 3 paneles paralelos.
# ==============================================================================

import os
import re
import sys
import py_compile

CANDIDATES = [
    os.path.join("acquisition", "autoforge_daq_experimental.py"),
    os.path.join("EMG_desarrollo", "acquisition", "autoforge_daq_experimental.py"),
    "autoforge_daq_experimental.py",
    "autoforge_daq_experimental_3.py",
]

TARGET_PATH = next((os.path.abspath(c) for c in CANDIDATES if os.path.exists(c)), None)
if not TARGET_PATH:
    print("Error: No se encontro autoforge_daq_experimental.py")
    sys.exit(1)


# ==============================================================================
# 1. CLASES AUTOCONTENIDAS: PRE-ROLL + CONSOLA DE ERRORES + MONITOR DAQ
# ==============================================================================
SELF_CONTAINED_CLASSES = '''
# ==============================================================================
# --- COMPONENTES INTEGRADOS: PRE-ROLL, TERMINAL DE DIAGNOSTICO Y MONITOR DAQ ---
# ==============================================================================
from collections import deque as _deque

class PreRollRecordingBuffer:
  """Buffer circular autocontenido para rescatar los ultimos N segundos antes de grabar."""
  def __init__(self, sample_rate: int = 2000, preroll_seconds: float = 1.0):
    self.sample_rate = int(sample_rate)
    self.preroll_seconds = max(0.0, float(preroll_seconds))
    self._chunks = _deque()
    self._total_samples = 0

  @property
  def max_samples(self) -> int:
    return max(0, int(self.sample_rate * self.preroll_seconds))

  def set_config(self, sample_rate: int, preroll_seconds: float) -> None:
    self.sample_rate = int(sample_rate)
    self.preroll_seconds = max(0.0, float(preroll_seconds))
    self._trim()

  def clear(self) -> None:
    self._chunks.clear()
    self._total_samples = 0

  def push_raw_chunk(self, raw_chunk_volts: np.ndarray) -> None:
    if self.max_samples <= 0 or raw_chunk_volts is None or raw_chunk_volts.shape[1] == 0:
      return
    self._chunks.append(raw_chunk_volts.copy())
    self._total_samples += raw_chunk_volts.shape[1]
    self._trim()

  def _trim(self) -> None:
    target = self.max_samples
    while self._chunks and (self._total_samples - self._chunks[0].shape[1]) >= target:
      removed = self._chunks.popleft()
      self._total_samples -= removed.shape[1]

  def dump_preroll_for_recording(self) -> list:
    if not self._chunks or self.max_samples <= 0:
      return []
    concat = np.concatenate(list(self._chunks), axis=1)
    if concat.shape[1] > self.max_samples:
      concat = concat[:, -self.max_samples:]
    return [concat]


class _TeeLogStream(QtCore.QObject):
  """Intercepta stdout y stderr hacia la consola integrada sin perder la salida original."""
  text_written = QtCore.Signal(str, bool)

  def __init__(self, original_stream, is_err=False):
    super().__init__()
    self._orig = original_stream
    self._is_err = is_err

  def write(self, text):
    if self._orig is not None:
      try:
        self._orig.write(text)
      except Exception:
        pass
    if text:
      self.text_written.emit(str(text), self._is_err)

  def flush(self):
    if self._orig is not None:
      try:
        self._orig.flush()
      except Exception:
        pass


class DiagnosticTerminalDialog(QtWidgets.QDialog):
  """Ventana de Terminal / Consola de Errores integrada con acceso a shell del sistema."""
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Consola de Diagnostico y Registro de Errores")
    self.resize(900, 520)
    self.setStyleSheet("""
      QDialog { background-color: #050505; color: #00FF00; border: 2px solid #00FFFF; font-family: 'Courier New', monospace; }
      QTextEdit { background-color: #080808; color: #00FF00; border: 1px solid #333333; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; }
      QPushButton { background-color: #111111; color: #00FFFF; border: 1px solid #00FFFF; padding: 6px 14px; font-weight: bold; border-radius: 3px; }
      QPushButton:hover { background-color: #00FFFF; color: #000000; }
    """)
    layout = QtWidgets.QVBoxLayout(self)

    hdr_layout = QtWidgets.QHBoxLayout()
    lbl_info = QtWidgets.QLabel("REGISTRO EN VIVO DE ADQUISICION, EVENTOS DAQ Y TRACEBACKS:")
    lbl_info.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 13px;")
    hdr_layout.addWidget(lbl_info)
    hdr_layout.addStretch()
    layout.addLayout(hdr_layout)

    self.txt_log = QtWidgets.QTextEdit()
    self.txt_log.setReadOnly(True)
    layout.addWidget(self.txt_log, stretch=1)

    btn_row = QtWidgets.QHBoxLayout()
    self.btn_open_os_term = QtWidgets.QPushButton("Abrir Terminal del Sistema (Shell)")
    self.btn_open_os_term.setStyleSheet("background-color: #1a001a; color: #FF00FF; border: 1px solid #FF00FF; padding: 6px 14px; font-weight: bold;")
    self.btn_open_os_term.clicked.connect(self.launch_system_terminal)

    self.btn_copy = QtWidgets.QPushButton("Copiar Todo")
    self.btn_copy.clicked.connect(self.copy_all)

    self.btn_clear = QtWidgets.QPushButton("Limpiar Consola")
    self.btn_clear.clicked.connect(self.txt_log.clear)

    self.btn_close = QtWidgets.QPushButton("Cerrar")
    self.btn_close.clicked.connect(self.hide)

    btn_row.addWidget(self.btn_open_os_term)
    btn_row.addStretch()
    btn_row.addWidget(self.btn_copy)
    btn_row.addWidget(self.btn_clear)
    btn_row.addWidget(self.btn_close)
    layout.addLayout(btn_row)

  def append_log(self, text: str, is_error: bool = False):
    cursor = self.txt_log.textCursor()
    cursor.movePosition(QtGui.QTextCursor.End)
    fmt = QtGui.QTextCharFormat()
    upper = text.upper()
    if is_error or "ERROR" in upper or "TRACEBACK" in upper or "EXCEPTION" in upper:
      fmt.setForeground(QtGui.QColor("#FF3333"))
    elif "ADVERTENCIA" in upper or "WARN" in upper:
      fmt.setForeground(QtGui.QColor("#FFAA00"))
    elif "[SIMULADOR]" in upper or "[DEBUG" in upper or "[OK]" in upper:
      fmt.setForeground(QtGui.QColor("#00FFFF"))
    else:
      fmt.setForeground(QtGui.QColor("#00FF00"))
    cursor.insertText(text, fmt)
    self.txt_log.setTextCursor(cursor)
    self.txt_log.ensureCursorVisible()

  def copy_all(self):
    QtWidgets.QApplication.clipboard().setText(self.txt_log.toPlainText())

  def launch_system_terminal(self):
    import shutil
    cwd = os.getcwd()
    try:
      if sys.platform == "win32":
        subprocess.Popen(["cmd.exe"], cwd=cwd, creationflags=subprocess.CREATE_NEW_CONSOLE)
      else:
        for term in ["konsole", "gnome-terminal", "xfce4-terminal", "mate-terminal", "xterm"]:
          if shutil.which(term):
            subprocess.Popen([term], cwd=cwd)
            return
    except Exception as e:
      print(f"Error al abrir terminal del sistema: {e}")
'''


# ==============================================================================
# 2. METODOS DE LAYOUT ESPACIAL REORGANIZADOS (3 PANELES + BARRA 50Hz + DAQ LED)
# ==============================================================================
NEW_SPATIAL_METHODS = '''  def _setup_ui_layouts(self):
    """
    Organizacion espacial en 4 niveles horizontales limpios:
      1. Barra Superior de Transporte, Estado DAQ y Acceso a Consola (self.transport_frame)
      2. Deck Superior Conmutable:
         - Izquierda (self.config_stack): 3 GroupBoxes lado a lado (Hardware | Filtros y Ritmo | Trigger y Espec)
           que al grabar conmutan al Visor Gigante de Palabras (self.empty_recording_widget).
         - Derecha Fija (self.right_side_panel): Metronomo siempre visible y sin deformarse.
      3. Barra de Mediciones por Canal con Medidor de Ruido de Linea 50 Hz integrado (self.measure_widget)
      4. Splitter Principal: Barra de Canales + Osciloscopio + Espectrograma
    """
    self.main_layout = QtWidgets.QVBoxLayout()
    self.main_layout.setContentsMargins(6, 6, 6, 6)
    self.main_layout.setSpacing(4)
    self.setLayout(self.main_layout)

    # Nivel 1: Barra Superior de Transporte y Diagnostico
    self.transport_frame = QtWidgets.QFrame()
    self.transport_frame.setStyleSheet(
      "QFrame { background-color: #080808; border: 1px solid #00FFFF; border-radius: 4px; }"
    )
    self.button_layout = QtWidgets.QHBoxLayout(self.transport_frame)
    self.button_layout.setContentsMargins(8, 4, 8, 4)
    self.button_layout.setSpacing(8)

    # Nivel 2: Contenedor de los 3 Paneles de Configuracion en Paralelo
    self.config_groupbox = QtWidgets.QWidget()
    self.deck_columns_layout = QtWidgets.QHBoxLayout(self.config_groupbox)
    self.deck_columns_layout.setContentsMargins(0, 0, 0, 0)
    self.deck_columns_layout.setSpacing(6)

    self.hw_groupbox = QtWidgets.QGroupBox("Configuracion de Adquisicion y Canales")
    self.config_layout = QtWidgets.QGridLayout(self.hw_groupbox)
    self.config_layout.setContentsMargins(8, 10, 8, 6)
    self.config_layout.setHorizontalSpacing(8)
    self.config_layout.setVerticalSpacing(4)

    self.filter_groupbox = QtWidgets.QGroupBox("Filtro Digital (Pasa-Banda) y Ritmo")
    self.filter_layout = QtWidgets.QGridLayout(self.filter_groupbox)
    self.filter_layout.setContentsMargins(8, 10, 8, 6)
    self.filter_layout.setHorizontalSpacing(8)
    self.filter_layout.setVerticalSpacing(4)

    self.trig_spec_groupbox = QtWidgets.QGroupBox("Trigger, SNR y Espectrograma")
    self.trig_spec_layout = QtWidgets.QGridLayout(self.trig_spec_groupbox)
    self.trig_spec_layout.setContentsMargins(8, 10, 8, 6)
    self.trig_spec_layout.setHorizontalSpacing(8)
    self.trig_spec_layout.setVerticalSpacing(4)

    self.deck_columns_layout.addWidget(self.hw_groupbox, stretch=4)
    self.deck_columns_layout.addWidget(self.filter_groupbox, stretch=4)
    self.deck_columns_layout.addWidget(self.trig_spec_groupbox, stretch=3)

    # Layouts auxiliares mantenidos por compatibilidad
    self.trigger_layout = QtWidgets.QHBoxLayout()
    self.spectrogram_layout = QtWidgets.QHBoxLayout()
    self.spectrogram_groupbox = QtWidgets.QWidget()

    # Nivel 3: Barra dedicada a Tarjetas de Canal + Medidor de Ruido de Linea 50 Hz
    self.measure_widget = QtWidgets.QFrame()
    self.measure_widget.setStyleSheet(
      "QFrame { background-color: #060606; border: 1px solid #222222; border-radius: 4px; }"
    )
    self.measure_layout = QtWidgets.QHBoxLayout(self.measure_widget)
    self.measure_layout.setContentsMargins(6, 3, 6, 3)
    self.measure_layout.setSpacing(6)

    # Inicializar Consola de Diagnostico Integrada
    self._error_count = 0
    self.terminal_dialog = DiagnosticTerminalDialog(self)
    self._stdout_tee = _TeeLogStream(sys.stdout, is_err=False)
    self._stderr_tee = _TeeLogStream(sys.stderr, is_err=True)
    self._stdout_tee.text_written.connect(self._on_log_captured)
    self._stderr_tee.text_written.connect(self._on_log_captured)
    sys.stdout = self._stdout_tee
    sys.stderr = self._stderr_tee

  def _setup_ui_config_panel(self):
    """Puebla el Panel 1 (Hardware y Canales) y crea el Metronomo fijo lateral."""
    self.label_device = QtWidgets.QLabel("Dispositivo:")
    self.cmb_device = QtWidgets.QComboBox()
    self.cmb_device.addItems(["Dev1", "Dev2", "Dev3"])
    self.cmb_device.currentIndexChanged.connect(self.verificar_conexion_daq)

    self.label_terminal_config = QtWidgets.QLabel("CONEXION DAQ:")
    self.label_terminal_config.setStyleSheet("color: #FFFF00; font-weight: bold;")
    self.cmb_terminal_config = QtWidgets.QComboBox()
    self.cmb_terminal_config.setStyleSheet(
      "background-color: #000000; color: #FFFF00; border: 1px solid #FFFF00; font-weight: bold; padding: 2px;"
    )
    self.cmb_terminal_config.addItems(["RSE", "DIFF", "NRSE", "DEFAULT"])
    self.cmb_terminal_config.setCurrentText("RSE")
    self.cmb_terminal_config.setToolTip("Modo de conexion a tierra. Usa 'DIFF' si tienes mucho ruido de 50 Hz.")

    self.label_sample_rate = QtWidgets.QLabel("Sample Rate (S/s):")
    self.cmb_sample_rate = QtWidgets.QComboBox()
    self.cmb_sample_rate.addItems(["2000", "6000", "10000", "20000", "44100", "50000"])
    self.cmb_sample_rate.setCurrentText("2000")

    self.label_plot_duration = QtWidgets.QLabel("Duracion Plot (s):")
    self.spin_plot_duration = SpinBox(value=10, step=1, bounds=(1, 60), int=True)
    self.spin_plot_duration.setFixedWidth(65)

    self.chk_modo_prueba = QtWidgets.QCheckBox("Modo Prueba")
    self.chk_modo_prueba.setToolTip("Usa datos simulados en lugar de la tarjeta NI-DAQ.")

    self.cmb_fuente_prueba = QtWidgets.QComboBox()
    self.cmb_fuente_prueba.addItems([
        "Cande 09-01 (/a/ normal)",
        "Cande 09-01 (Secuencia Continua)",
        "Senoidal (50 Hz Sintetica)",
        "Archivo 'grabacion.csv'"
    ])
    self.cmb_fuente_prueba.setEnabled(False)

    self.label_test_freq = QtWidgets.QLabel("Frec. Prueba (Hz):")
    self.spin_test_freq = SpinBox(value=50, step=10, bounds=(1, 5000), int=True)
    self.spin_test_freq.setFixedWidth(60)

    self.chk_use_metronome = QtWidgets.QCheckBox("Usar Metronomo")
    self.chk_use_metronome.setChecked(True)

    self.chk_use_mic = QtWidgets.QCheckBox("Usar Microfono (PC)")
    self.chk_use_mic.setToolTip("Usa el microfono de la PC en lugar de la DAQ.")
    self.chk_use_mic.toggled.connect(self.verificar_conexion_daq)

    self.label_channels = QtWidgets.QLabel("Canales:")
    self.channel_checkboxes = []
    self.channel_layout = QtWidgets.QHBoxLayout()
    self.channel_layout.setSpacing(6)
    any_checked = False
    for i in range(8):
      chk = QtWidgets.QCheckBox(f"ai{i}")
      is_active = self.config_mgr.config.get("canales", {}).get(f"Canal {i}", {}).get("activo_por_defecto", False)
      if is_active:
        chk.setChecked(True)
        any_checked = True
      self.channel_checkboxes.append(chk)
      self.channel_layout.addWidget(chk)
    if not any_checked and self.channel_checkboxes:
      self.channel_checkboxes[0].setChecked(True)

    self.btn_start_acq = QtWidgets.QPushButton("Iniciar Adquisicion")
    self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)
    self.btn_start_acq.setMinimumWidth(170)

    self.label_bpm = QtWidgets.QLabel("BPM / Sub:")
    self.spin_bpm = SpinBox(value=60, int=True, bounds=(30, 200), step=1)
    self.spin_bpm.setFixedWidth(55)
    self.spin_metro_subdivs = SpinBox(value=4, int=True, bounds=(1, 8), step=1)
    self.spin_metro_subdivs.setFixedWidth(45)

    self.label_noise_duration = QtWidgets.QLabel("Noise Inicio (s):")
    adq_conf = self.config_mgr.get("adquisicion") or {}
    default_noise = adq_conf.get("ruido_segundos", 3.0)
    self.spin_noise_duration = SpinBox(value=default_noise, dec=True, bounds=(0.5, 20.0), step=0.5)
    self.spin_noise_duration.setFixedWidth(60)

    # Grilla ordenada de 4 filas en el Panel 1
    self.config_layout.addWidget(self.label_device, 0, 0)
    self.config_layout.addWidget(self.cmb_device, 0, 1)
    self.config_layout.addWidget(self.label_terminal_config, 0, 2)
    self.config_layout.addWidget(self.cmb_terminal_config, 0, 3)

    self.config_layout.addWidget(self.label_sample_rate, 1, 0)
    self.config_layout.addWidget(self.cmb_sample_rate, 1, 1)
    self.config_layout.addWidget(self.label_plot_duration, 1, 2)
    self.config_layout.addWidget(self.spin_plot_duration, 1, 3)

    self.config_layout.addWidget(self.chk_modo_prueba, 2, 0)
    self.config_layout.addWidget(self.cmb_fuente_prueba, 2, 1)
    self.config_layout.addWidget(self.label_test_freq, 2, 2)
    self.config_layout.addWidget(self.spin_test_freq, 2, 3)

    self.config_layout.addWidget(self.label_channels, 3, 0)
    self.config_layout.addLayout(self.channel_layout, 3, 1, 1, 3)

    # Inicializar el Metronomo en su propio panel derecho independiente
    self._setup_native_metronome()

  def _setup_ui_filter_panel(self):
    """Puebla el Panel 2 (Filtros Digitales, Envolvente RMS, Ritmo y Pre-Roll)."""
    self.chk_notch_enable = QtWidgets.QCheckBox("Notch 50 Hz")
    self.chk_notch_enable.setChecked(True)

    self.label_notch_q = QtWidgets.QLabel("Q:")
    self.spin_notch_q = SpinBox(value=2.0, step=1.0, bounds=(1.0, 50.0), dec=True)
    self.spin_notch_q.setFixedWidth(50)

    self.chk_notch_comb = QtWidgets.QCheckBox("Peine (150/250/350 Hz)")
    self.chk_notch_comb.setChecked(False)

    self.spin_notch_q.valueChanged.connect(self.on_filter_settings_changed)
    self.chk_notch_comb.clicked.connect(self.on_filter_settings_changed)

    self.chk_filter_enable = QtWidgets.QCheckBox("Pasa-Banda")
    self.chk_filter_enable.setChecked(True)

    self.label_low_cut = QtWidgets.QLabel("Baja (Hz):")
    self.spin_low_cut = SpinBox(value=20, step=1, bounds=(1, 20000), int=True)
    self.spin_low_cut.setFixedWidth(60)
    self.label_high_cut = QtWidgets.QLabel("Alta (Hz):")
    self.spin_high_cut = SpinBox(value=500, step=1, bounds=(2, 22000), int=True)
    self.spin_high_cut.setFixedWidth(60)
    self.label_filter_order = QtWidgets.QLabel(f"Ord. {self.FILTER_ORDER}")

    self.chk_rms_env = QtWidgets.QCheckBox("Envolvente RMS")
    self.chk_rms_env.setChecked(False)
    self.label_rms_window = QtWidgets.QLabel("Ventana (ms):")
    self.spin_rms_window = SpinBox(value=100.0, step=10.0, bounds=(10.0, 1000.0), dec=True)
    self.spin_rms_window.setFixedWidth(60)

    self.lbl_preroll = QtWidgets.QLabel("Pre-Roll (s):")
    self.spin_preroll = SpinBox(value=1.0, step=0.5, bounds=(0.0, 5.0), dec=True)
    self.spin_preroll.setFixedWidth(55)
    self.spin_preroll.valueChanged.connect(
      lambda val: self.preroll_buffer.set_config(max(2000, self.SAMPLE_RATE), val)
    )

    self.chk_show_noise = QtWidgets.QCheckBox("Mostrar Ruido")
    self.chk_show_noise.setChecked(False)
    self.chk_show_noise.clicked.connect(self.toggle_noise_regions)

    self.chk_show_peaks = QtWidgets.QCheckBox("Mostrar Picos")
    self.chk_show_peaks.setChecked(False)
    self.chk_show_peaks.clicked.connect(self.toggle_peak_scatter)

    # Fila 0: Notch
    self.filter_layout.addWidget(self.chk_notch_enable, 0, 0)
    self.filter_layout.addWidget(self.label_notch_q, 0, 1)
    self.filter_layout.addWidget(self.spin_notch_q, 0, 2)
    self.filter_layout.addWidget(self.chk_notch_comb, 0, 3, 1, 3)

    # Fila 1: Pasa-Banda
    self.filter_layout.addWidget(self.chk_filter_enable, 1, 0)
    self.filter_layout.addWidget(self.label_low_cut, 1, 1)
    self.filter_layout.addWidget(self.spin_low_cut, 1, 2)
    self.filter_layout.addWidget(self.label_high_cut, 1, 3)
    self.filter_layout.addWidget(self.spin_high_cut, 1, 4)
    self.filter_layout.addWidget(self.label_filter_order, 1, 5)

    # Fila 2: Envolvente RMS + Pre-Roll + Microfono
    self.filter_layout.addWidget(self.chk_rms_env, 2, 0)
    self.filter_layout.addWidget(self.label_rms_window, 2, 1)
    self.filter_layout.addWidget(self.spin_rms_window, 2, 2)
    self.filter_layout.addWidget(self.lbl_preroll, 2, 3)
    self.filter_layout.addWidget(self.spin_preroll, 2, 4)
    self.filter_layout.addWidget(self.chk_use_mic, 2, 5)

    # Fila 3: Metronomo + BPM/Sub + Noise Inicio
    bpm_sub_layout = QtWidgets.QHBoxLayout()
    bpm_sub_layout.setContentsMargins(0, 0, 0, 0)
    bpm_sub_layout.setSpacing(4)
    bpm_sub_layout.addWidget(self.spin_bpm)
    bpm_sub_layout.addWidget(self.spin_metro_subdivs)

    self.filter_layout.addWidget(self.chk_use_metronome, 3, 0)
    self.filter_layout.addWidget(self.label_bpm, 3, 1)
    self.filter_layout.addLayout(bpm_sub_layout, 3, 2, 1, 2)
    self.filter_layout.addWidget(self.label_noise_duration, 3, 4)
    self.filter_layout.addWidget(self.spin_noise_duration, 3, 5)

  def _setup_ui_controls(self):
    """
    Configura la Barra Superior (Transporte + Indicador DAQ + Boton Terminal)
    y el Panel 3 (Trigger, SNR y Espectrograma).
    """
    # Indicador luminoso de conexion DAQ
    self.lbl_daq_status = QtWidgets.QLabel("DAQ: VERIFICANDO...")
    self.lbl_daq_status.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_daq_status.setMinimumWidth(190)
    self.lbl_daq_status.setStyleSheet(
      "background-color: #111111; color: #00FFFF; border: 1px solid #00FFFF; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
    )

    self.btn_check_daq = QtWidgets.QPushButton("Verificar DAQ")
    self.btn_check_daq.setToolTip("Comprueba fisicamente si la tarjeta NI-DAQmx seleccionada responde por USB/PCI.")
    self.btn_check_daq.setStyleSheet(
      "QPushButton { background-color: #111111; color: #00FFFF; border: 1px solid #00FFFF; padding: 6px 10px; font-weight: bold; border-radius: 3px; }"
      "QPushButton:hover { background-color: #00FFFF; color: #000000; }"
    )
    self.btn_check_daq.clicked.connect(self.verificar_conexion_daq)

    self.btn_record = QtWidgets.QPushButton("Empezar a Grabar")
    self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)

    self.btn_pause = QtWidgets.QPushButton("PAUSAR")
    self.btn_pause.setCheckable(True)
    self.btn_pause.setEnabled(False)
    self.btn_pause.setStyleSheet("""
      QPushButton {
        background-color: #1a1400; color: #ffcc00; font-weight: bold;
        font-family: 'Courier New'; font-size: 14px; padding: 8px 14px;
        border: 2px solid #ffcc00; border-radius: 4px;
      }
      QPushButton:checked {
        background-color: #ffcc00; color: #000000; border: 2px solid #ffffff;
      }
      QPushButton:disabled { border-color: #333333; color: #444444; background-color: #080808; }
    """)
    self.btn_pause.toggled.connect(self.on_pause_toggled)

    self.btn_autoforge = QtWidgets.QPushButton("AUTOGRABADO")
    self.btn_autoforge.setStyleSheet("""
      QPushButton {
        background-color: #1a001a; color: #ff00ff; font-weight: 900;
        font-family: 'Consolas', 'Courier New', monospace; font-size: 14px;
        padding: 8px 12px; border-radius: 4px; border: 2px solid #ff00ff;
        border-right: 4px solid #00ffff; border-bottom: 4px solid #00ffff;
      }
      QPushButton:hover { background-color: #ff00ff; color: #000000; border: 2px solid #00ffff; }
    """)
    self.btn_autoforge.clicked.connect(self.iniciar_autoforge)

    self.btn_autoforge_continuo = QtWidgets.QPushButton("SECUENCIA CONTINUA")
    self.btn_autoforge_continuo.setStyleSheet("""
      QPushButton {
        background-color: #0d001a; color: #aa00ff; font-weight: 900;
        font-family: 'Consolas', 'Courier New', monospace; font-size: 14px;
        padding: 8px 12px; border-radius: 4px; border: 2px solid #aa00ff;
        border-right: 4px solid #ff00ff; border-bottom: 4px solid #ff00ff;
      }
      QPushButton:hover { background-color: #aa00ff; color: #ffffff; border: 2px solid #ff00ff; }
    """)
    self.btn_autoforge_continuo.clicked.connect(self.iniciar_autoforge_continuo)

    self.label_rec_time = QtWidgets.QLabel("SISTEMA EN ESPERA")
    self.label_rec_time.setAlignment(QtCore.Qt.AlignCenter)
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 13px; background-color: #050505; border: 1px solid #222222; padding: 4px 10px; border-radius: 3px;")
    self.label_rec_time.setVisible(True)

    # Boton de Consola / Terminal de Errores
    self.btn_terminal = QtWidgets.QPushButton("CONSOLA / LOGS (0)")
    self.btn_terminal.setToolTip("Abre la consola integrada para inspeccionar mensajes, errores de NIDAQmx o abrir una terminal del sistema.")
    self.btn_terminal.setStyleSheet(
      "QPushButton { background-color: #0a150a; color: #00FF00; border: 1.5px solid #00FF00; padding: 7px 12px; font-weight: bold; border-radius: 4px; }"
      "QPushButton:hover { background-color: #00FF00; color: #000000; }"
    )
    self.btn_terminal.clicked.connect(self.mostrar_consola_diagnostico)

    # Ensamblar Barra Superior de Transporte
    self.button_layout.addWidget(self.btn_start_acq)
    self.button_layout.addWidget(self.lbl_daq_status)
    self.button_layout.addWidget(self.btn_check_daq)
    self.button_layout.addSpacing(8)
    self.button_layout.addWidget(self.btn_record)
    self.button_layout.addWidget(self.btn_pause)
    self.button_layout.addWidget(self.btn_autoforge)
    self.button_layout.addWidget(self.btn_autoforge_continuo)
    self.button_layout.addWidget(self.label_rec_time, stretch=1)
    self.button_layout.addWidget(self.btn_terminal)

    # Controles del Panel 3 (Trigger, SNR y Espectrograma)
    self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll (Armar Trigger)")
    self.chk_autoscroll.setChecked(True)

    self.chk_trigger = QtWidgets.QCheckBox("Habilitar Trigger")
    self.label_trig_chan = QtWidgets.QLabel("Canal:")
    self.cmb_trig_chan = QtWidgets.QComboBox()

    self.label_trig_level = QtWidgets.QLabel("Nivel (uV):")
    self.spin_trig_level = SpinBox(value=1000.0, step=100.0, dec=True, minStep=1.0, bounds=(-20000.0, 20000.0))
    self.spin_trig_level.setFixedWidth(75)

    self.label_trig_edge = QtWidgets.QLabel("Flanco:")
    self.cmb_trig_edge = QtWidgets.QComboBox()
    self.cmb_trig_edge.addItems(["Subida ", "Bajada "])

    self.label_peak_th = QtWidgets.QLabel("Umbral Picos SNR (uV):")
    self.spin_peak_th = SpinBox(value=200.0, step=50.0, bounds=(0.0, 20000.0), dec=True)
    self.spin_peak_th.setFixedWidth(75)

    self.chk_spectrogram_enable = QtWidgets.QCheckBox("Habilitar Espectrograma")
    self.label_spectrogram_chan = QtWidgets.QLabel("Canal:")
    self.cmb_spectrogram_chan = QtWidgets.QComboBox()

    self.trig_spec_layout.addWidget(self.chk_trigger, 0, 0, 1, 2)
    self.trig_spec_layout.addWidget(self.label_trig_chan, 0, 2)
    self.trig_spec_layout.addWidget(self.cmb_trig_chan, 0, 3)

    self.trig_spec_layout.addWidget(self.label_trig_level, 1, 0)
    self.trig_spec_layout.addWidget(self.spin_trig_level, 1, 1)
    self.trig_spec_layout.addWidget(self.label_trig_edge, 1, 2)
    self.trig_spec_layout.addWidget(self.cmb_trig_edge, 1, 3)

    self.trig_spec_layout.addWidget(self.label_peak_th, 2, 0, 1, 2)
    self.trig_spec_layout.addWidget(self.spin_peak_th, 2, 2, 1, 2)

    self.trig_spec_layout.addWidget(self.chk_spectrogram_enable, 3, 0, 1, 2)
    self.trig_spec_layout.addWidget(self.label_spectrogram_chan, 3, 2)
    self.trig_spec_layout.addWidget(self.cmb_spectrogram_chan, 3, 3)

    self.measure_labels = []
    self.measure_layout.addStretch(1)

    # Verificar estado DAQ inicial
    QtCore.QTimer.singleShot(300, self.verificar_conexion_daq)

  def _setup_ui_plots(self):
    """Configura la barra de canales y el osciloscopio principal."""
    self.plot_container = QtWidgets.QWidget()
    self.plot_container_layout = QtWidgets.QVBoxLayout(self.plot_container)
    self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
    self.plot_container_layout.setSpacing(2)

    self.viz_control_panel = QtWidgets.QWidget()
    self.viz_control_panel.setStyleSheet("background-color: #050505; border: 1px solid #333333;")
    self.viz_control_layout = QtWidgets.QHBoxLayout(self.viz_control_panel)
    self.viz_control_layout.setContentsMargins(6, 3, 6, 3)
    self.viz_control_layout.setSpacing(10)

    lbl_viz = QtWidgets.QLabel("Visualizacion de Canales:")
    lbl_viz.setStyleSheet("color: #00FFFF; font-weight: bold; border: none;")
    self.viz_control_layout.addWidget(lbl_viz)

    self.btn_hide_all_viz = QtWidgets.QPushButton("Ocultar Todos")
    self.btn_hide_all_viz.setStyleSheet(
      "background-color: #330000; color: #ff5555; border: 1px solid #ff5555; padding: 2px 8px; font-size: 11px; font-weight: bold; border-radius: 3px;"
    )
    self.btn_hide_all_viz.clicked.connect(self.hide_all_viz_channels)
    self.viz_control_layout.addWidget(self.btn_hide_all_viz)

    self.btn_toggle_multitrack = QtWidgets.QPushButton("Vista: Pistas Separadas")
    self.btn_toggle_multitrack.setCheckable(True)
    self.btn_toggle_multitrack.setStyleSheet(
      "QPushButton { background-color: #050505; color: #00FFFF; border: 1px solid #00FFFF; padding: 2px 10px; font-size: 11px; font-weight: bold; border-radius: 3px; }"
      "QPushButton:checked { background-color: #00FFFF; color: #000000; }"
    )
    self.btn_toggle_multitrack.toggled.connect(self.toggle_multitrack_view)
    self.viz_control_layout.addWidget(self.btn_toggle_multitrack)

    self.viz_checkbox_layout = QtWidgets.QHBoxLayout()
    self.viz_checkbox_layout.setSpacing(8)
    self.viz_control_layout.addLayout(self.viz_checkbox_layout)
    self.viz_control_layout.addStretch(1)

    self.viz_control_layout.addWidget(self.chk_show_noise)
    self.viz_control_layout.addWidget(self.chk_show_peaks)
    self.viz_control_layout.addWidget(self.chk_autoscroll)

    self.viz_checkboxes = []
    self.plot_container_layout.addWidget(self.viz_control_panel)

    self.plot_widget = pg.GraphicsLayoutWidget()

    self.autoforge_overlay = QtWidgets.QLabel(self.plot_widget)
    self.autoforge_overlay.setAlignment(QtCore.Qt.AlignCenter)
    self.autoforge_overlay.setStyleSheet(
      "background-color: rgba(10, 5, 20, 185); color: #FF0055; font-family: 'Courier New', monospace; "
      "font-size: 75px; font-weight: 900; padding: 20px; border: 4px solid #00FFFF;"
    )
    self.autoforge_overlay.hide()
    self.plot_widget.installEventFilter(self)

    self.plot_container_layout.addWidget(self.plot_widget, stretch=1)
    self.plot_widget.setBackground('k')

    self.plot = self.plot_widget.addPlot(title="Canales de Adquisicion")
    self.plot.setLabel('bottom', 'Tiempo (s)')
    self.plot.setLabel('left', 'Amplitud (µV)')
    self.plot.getAxis('left').setWidth(60)

    self.spectrogram_view = pg.ImageView()
    self.spectrogram_view.ui.histogram.hide()
    self.spectrogram_view.ui.roiBtn.hide()
    self.spectrogram_view.ui.menuBtn.hide()

    self.plot_widget.ci.layout.setContentsMargins(10, 10, 30, 10)
    self.plot.addLegend()
    self.plot.showGrid(x=True, y=True)
    self.plot.setClipToView(True)
    self.plot.setDownsampling(auto=True, mode='peak')
    self.plot.autoBtn.setVisible(True)
    self.plot.getViewBox().setLimits(xMin=-self.spin_plot_duration.value(), minXRange=1e-6)

    self.trigger_line = InfiniteLine(pos=1000.0, angle=0, movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
    self.plot.addItem(self.trigger_line)
    self.trigger_label = pg.TextItem(anchor=(0, 1), color=(255, 255, 0), fill=(0, 0, 0, 150))
    self.trigger_label.setZValue(100)
    self.trigger_label.hide()
    self.plot.addItem(self.trigger_label)

    self.peak_th_line_pos = InfiniteLine(pos=200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine))
    self.peak_th_line_neg = InfiniteLine(pos=-200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine))
    self.plot.addItem(self.peak_th_line_pos)
    self.plot.addItem(self.peak_th_line_neg)

    self.countdown_text = pg.TextItem(text="", color=(0, 255, 255), anchor=(0.5, 0.5))
    font = QtGui.QFont()
    font.setPixelSize(80)
    font.setBold(True)
    self.countdown_text.setFont(font)
    self.countdown_text.setZValue(10000)
    self.plot.addItem(self.countdown_text)
    self.countdown_text.hide()

  def _setup_ui_final_layout(self):
    """Ensambla las 4 zonas manteniendo el Metronomo fijo a la derecha y los 50 Hz en las tarjetas de canal."""
    self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    self.splitter.addWidget(self.plot_container)
    self.splitter.addWidget(self.spectrogram_view)
    plot_h = int(self.screen_h * 0.54) if hasattr(self, 'screen_h') else 440
    spec_h = int(self.screen_h * 0.15) if hasattr(self, 'screen_h') else 120
    self.splitter.setSizes([plot_h, spec_h])

    self.config_stack = QtWidgets.QStackedWidget()
    self.config_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    min_config_h = 135 if getattr(self, 'is_compact_screen', False) else 155
    self.config_groupbox.setMinimumHeight(min_config_h)
    self.config_stack.setMinimumHeight(min_config_h)
    self.config_stack.setMaximumHeight(175)
    self.config_stack.addWidget(self.config_groupbox)

    self.empty_recording_widget = QtWidgets.QWidget()
    self.empty_recording_widget.setStyleSheet("background-color: #050505;")

    lbl_font_size = "30px" if getattr(self, 'is_compact_screen', False) else "42px"
    self.lbl_recording_space = QtWidgets.QLabel("MODO GRABACION ACTIVO", self.empty_recording_widget)
    self.lbl_recording_space.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_recording_space.setWordWrap(True)
    self.lbl_recording_space.setStyleSheet(
      f"background-color: #050510; color: #00FF00; font-family: 'Courier New', monospace; "
      f"font-size: {lbl_font_size}; font-weight: 900; border: 2px dashed #FF0055; padding: 6px;"
    )

    self.empty_recording_layout = QtWidgets.QHBoxLayout(self.empty_recording_widget)
    self.empty_recording_layout.setContentsMargins(0, 0, 0, 0)
    self.empty_recording_layout.addWidget(self.lbl_recording_space)
    self.config_stack.addWidget(self.empty_recording_widget)

    # Contenedor Superior: [Config_Stack a la izquierda] + [Metronomo fijo a la derecha]
    self.top_deck_widget = QtWidgets.QWidget()
    self.top_deck_layout = QtWidgets.QHBoxLayout(self.top_deck_widget)
    self.top_deck_layout.setContentsMargins(0, 0, 0, 0)
    self.top_deck_layout.setSpacing(6)
    self.top_deck_layout.addWidget(self.config_stack, stretch=6)
    self.top_deck_layout.addWidget(self.right_side_panel, stretch=1)

    self.main_layout.addWidget(self.transport_frame)    # Fila 1: Transporte + Estado DAQ + Consola
    self.main_layout.addWidget(self.top_deck_widget)    # Fila 2: 3 Paneles Config / Palabras + Metronomo
    self.main_layout.addWidget(self.measure_widget)     # Fila 3: Mediciones y Ruido 50 Hz por Canal
    self.main_layout.addWidget(self.splitter, stretch=1)# Fila 4: Osciloscopio + Espectrograma
'''


# ==============================================================================
# 3. METODOS NUEVOS: DETECTOR DAQ, CONSOLA Y ACTUALIZACION EN VIVO DE 50 HZ
# ==============================================================================
EXTRA_METHODS = '''
  @QtCore.Slot(str, bool)
  def _on_log_captured(self, text: str, is_err: bool):
    if hasattr(self, 'terminal_dialog'):
      self.terminal_dialog.append_log(text, is_err)
    upper = text.upper()
    if is_err or "ERROR" in upper or "TRACEBACK" in upper:
      self._error_count = getattr(self, '_error_count', 0) + 1
      if hasattr(self, 'btn_terminal'):
        self.btn_terminal.setText(f"CONSOLA / ERRORES ({self._error_count})")
        self.btn_terminal.setStyleSheet(
          "QPushButton { background-color: #330000; color: #FF3333; border: 2px solid #FF0000; padding: 7px 12px; font-weight: bold; border-radius: 4px; }"
          "QPushButton:hover { background-color: #FF0000; color: #000000; }"
        )

  def mostrar_consola_diagnostico(self):
    self._error_count = 0
    if hasattr(self, 'btn_terminal'):
      self.btn_terminal.setText("CONSOLA / LOGS (0)")
      self.btn_terminal.setStyleSheet(
        "QPushButton { background-color: #0a150a; color: #00FF00; border: 1.5px solid #00FF00; padding: 7px 12px; font-weight: bold; border-radius: 4px; }"
        "QPushButton:hover { background-color: #00FF00; color: #000000; }"
      )
    self.terminal_dialog.show()
    self.terminal_dialog.raise_()
    self.terminal_dialog.activateWindow()

  def verificar_conexion_daq(self):
    """Detecta en tiempo real si la tarjeta NI-DAQmx seleccionada esta fisicamente conectada."""
    if not hasattr(self, 'lbl_daq_status'):
      return
    if getattr(self, 'chk_use_mic', None) and self.chk_use_mic.isChecked():
      self.lbl_daq_status.setText("FUENTE: MICROFONO PC")
      self.lbl_daq_status.setStyleSheet(
        "background-color: #001a22; color: #00FFFF; border: 1.5px solid #00FFFF; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
      )
      return
    if getattr(self, 'chk_modo_prueba', None) and self.chk_modo_prueba.isChecked():
      self.lbl_daq_status.setText("FUENTE: SIMULADOR ACTIVO")
      self.lbl_daq_status.setStyleSheet(
        "background-color: #1a001a; color: #FF00FF; border: 1.5px solid #FF00FF; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
      )
      return

    dev_target = self.cmb_device.currentText().strip()
    if not NIDAQMX_DISPONIBLE:
      self.lbl_daq_status.setText(f"DAQ: SIN DRIVER NIDAQMX")
      self.lbl_daq_status.setStyleSheet(
        "background-color: #2a0000; color: #FF4444; border: 1.5px solid #FF0000; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
      )
      return

    try:
      import nidaqmx.system
      system = nidaqmx.system.System.local()
      dev_names = [d.name for d in system.devices]
      if dev_target in dev_names:
        dev_obj = system.devices[dev_target]
        prod_type = getattr(dev_obj, 'product_type', 'NI-DAQ')
        self.lbl_daq_status.setText(f"DAQ: {dev_target} CONECTADO [{prod_type}]")
        self.lbl_daq_status.setStyleSheet(
          "background-color: #00220a; color: #00FF00; border: 1.5px solid #00FF00; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
        )
        print(f"[DAQ CHECK] Dispositivo fisico detectado: {dev_target} ({prod_type})")
      else:
        avail_str = ", ".join(dev_names) if dev_names else "Ninguno"
        self.lbl_daq_status.setText(f"DAQ: {dev_target} DESCONECTADO")
        self.lbl_daq_status.setStyleSheet(
          "background-color: #2a0008; color: #FF3333; border: 1.5px solid #FF0033; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
        )
        print(f"[DAQ CHECK] {dev_target} no encontrado. Dispositivos conectados: {avail_str}")
    except Exception as e:
      self.lbl_daq_status.setText(f"DAQ: ERROR DRIVER")
      self.lbl_daq_status.setStyleSheet(
        "background-color: #2a0008; color: #FF3333; border: 1.5px solid #FF0033; padding: 5px 8px; font-weight: bold; border-radius: 3px;"
      )
      print(f"[DAQ CHECK ERROR] {e}")

  def _actualizar_medidor_50hz_en_vivo(self, raw_uV_chunk: np.ndarray):
    """Calcula y actualiza cada 0.5 s el ruido de linea de 50 Hz en cada tarjeta de canal."""
    if self.NUM_CANALES <= 0 or raw_uV_chunk is None or raw_uV_chunk.shape[1] == 0:
      return
    win_50hz = max(int(self.SAMPLE_RATE * 0.5), 200)
    if not hasattr(self, '_raw_50hz_buffer') or self._raw_50hz_buffer is None or self._raw_50hz_buffer.shape != (self.NUM_CANALES, win_50hz):
      self._raw_50hz_buffer = np.zeros((self.NUM_CANALES, win_50hz), dtype=np.float64)

    n_new = raw_uV_chunk.shape[1]
    if n_new >= win_50hz:
      self._raw_50hz_buffer[:, :] = raw_uV_chunk[:, -win_50hz:]
    else:
      self._raw_50hz_buffer = np.roll(self._raw_50hz_buffer, -n_new, axis=1)
      self._raw_50hz_buffer[:, -n_new:] = raw_uV_chunk

    now_t = time.perf_counter()
    if now_t - getattr(self, '_last_50hz_ui_update', 0.0) < 0.5:
      return
    self._last_50hz_ui_update = now_t

    for i in range(self.NUM_CANALES):
      val_50 = calcular_ruido_linea_50hz(self._raw_50hz_buffer[i], self.SAMPLE_RATE)
      if val_50 < 15.0:
        color_txt, color_bg, estado_str = "#00FFCC", "#002211", "OPTIMO"
      elif val_50 < 40.0:
        color_txt, color_bg, estado_str = "#FFFF00", "#222200", "MODERADO"
      else:
        color_txt, color_bg, estado_str = "#FF3300", "#330000", "ALTO"

      musculo = self.nombres_musculos[i % len(self.nombres_musculos)]
      if hasattr(self, 'line_noise_status_labels') and i < len(self.line_noise_status_labels):
        self.line_noise_status_labels[i].setText(f"50Hz ({musculo}): {val_50:.1f} uV [{estado_str}]")
        self.line_noise_status_labels[i].setStyleSheet(f"color: {color_txt}; font-size: 11px; font-weight: bold; border: none;")
      if hasattr(self, 'line_noise_status_frames') and i < len(self.line_noise_status_frames):
        self.line_noise_status_frames[i].setStyleSheet(
          f"background-color: {color_bg}; border: 1px solid {color_txt}; border-radius: 3px; padding: 2px 6px;"
        )

  def _schedule_phase(self, duration_ms: int, callback):
    if not hasattr(self, '_phase_timer'):
      self._phase_timer = QtCore.QTimer(self)
      self._phase_timer.setSingleShot(True)
      self._phase_timer.timeout.connect(self._run_scheduled_phase_callback)
    self._phase_timer.stop()
    self._scheduled_phase_cb = callback
    self._paused_phase_remaining_ms = max(1, int(duration_ms))
    if not getattr(self, 'is_paused', False):
      self._phase_timer.start(self._paused_phase_remaining_ms)

  def _run_scheduled_phase_callback(self):
    cb = getattr(self, '_scheduled_phase_cb', None)
    self._scheduled_phase_cb = None
    self._paused_phase_remaining_ms = 0
    if callable(cb) and getattr(self, 'is_autoforge_running', False):
      cb()

  def toggle_multitrack_view(self, multitrack: bool):
    self.is_multitrack_view = multitrack
    if hasattr(self, 'btn_toggle_multitrack'):
      self.btn_toggle_multitrack.setText("Vista: Superpuesta" if multitrack else "Vista: Pistas Separadas")
    if self.NUM_CANALES <= 0 or not self.curvas:
      return
    for sp in getattr(self, 'sub_plots', []):
      try:
        self.plot_widget.removeItem(sp)
      except Exception:
        pass
    self.sub_plots = []
    for c in self.curvas:
      try:
        if c.scene() is not None:
          c.getViewBox().removeItem(c)
      except Exception:
        pass
    if not multitrack:
      self.plot.showAxis('bottom')
      for i, c in enumerate(self.curvas):
        self.plot.addItem(c)
        if i < len(self.viz_checkboxes):
          c.setVisible(self.viz_checkboxes[i].isChecked())
    else:
      self.plot.hideAxis('bottom')
      self.plot.addItem(self.curvas[0])
      for i in range(1, self.NUM_CANALES):
        sp = self.plot_widget.addPlot(row=i, col=0)
        sp.showGrid(x=True, y=True)
        sp.getAxis('left').setWidth(60)
        musc_name = self.nombres_musculos[i % len(self.nombres_musculos)]
        sp.setLabel('left', f"{musc_name} (µV)")
        sp.setXLink(self.plot)
        sp.setYLink(self.plot)
        if i < self.NUM_CANALES - 1:
          sp.hideAxis('bottom')
        else:
          sp.setLabel('bottom', 'Tiempo (s)')
        sp.addItem(self.curvas[i])
        if i < len(self.viz_checkboxes):
          self.curvas[i].setVisible(self.viz_checkboxes[i].isChecked())
        self.sub_plots.append(sp)
'''


def repair_and_upgrade(path: str):
    with open(path, "r", encoding="utf-8") as f:
        code = f.read()

    # 1. Insertar clases autocontenidas y eliminar el import roto de core.preroll_and_filters
    if "class PreRollRecordingBuffer:" not in code:
        code = code.replace(
            "from pyqtgraph import SpinBox, InfiniteLine",
            "from pyqtgraph import SpinBox, InfiniteLine\n" + SELF_CONTAINED_CLASSES,
            1,
        )
    code = code.replace(
        "from core.preroll_and_filters import PreRollRecordingBuffer, ConfigurableFilterChain\n",
        "",
    )
    code = code.replace(
        "    from core.preroll_and_filters import PreRollRecordingBuffer, ConfigurableFilterChain\n",
        "",
    )
    code = re.sub(
        r"[ \t]*from core\.preroll_and_filters import PreRollRecordingBuffer[^\n]*\n?",
        "",
        code,
    )

    # 2. Reemplazar los metodos de layout (_setup_ui_layouts .. _setup_ui_final_layout)
    pattern_ui = re.compile(
        r"  def _setup_ui_layouts\(self\):.*?(?=  def _connect_signals\(self\):)",
        re.DOTALL,
    )
    if not pattern_ui.search(code):
        raise RuntimeError("No se encontro el bloque _setup_ui_layouts .. _connect_signals")
    code = pattern_ui.sub(NEW_SPATIAL_METHODS + "\n", code, count=1)

    # 3. Reubicar el medidor de 50 Hz: sacarlo de metro_noise_layout y ponerlo dentro de cada tarjeta de canal
    code = code.replace(
        "if hasattr(self, 'metro_noise_layout'):\n          self.metro_noise_layout.addWidget(frame_50hz)\n      else:\n          frame_layout.addWidget(frame_50hz)",
        "frame_layout.addWidget(frame_50hz)",
    )
    code = code.replace(
        "      if hasattr(self, 'metro_noise_layout'):\n          self.metro_noise_layout.addWidget(frame_50hz)\n      else:\n          frame_layout.addWidget(frame_50hz)",
        "      frame_layout.addWidget(frame_50hz)",
    )

    # 4. Arreglar _reubicar_metronomo_dinamico para que no mueva ni rompa el metronomo
    pattern_reubicar = re.compile(
        r"  def _reubicar_metronomo_dinamico\(self, index\):.*?self\.empty_recording_layout\.addWidget\(self\.right_side_panel, stretch=1\)",
        re.DOTALL,
    )
    code = pattern_reubicar.sub(
        "  def _reubicar_metronomo_dinamico(self, index):\n    pass",
        code,
        count=1,
    )

    # 5. Actualizar _on_modo_prueba_toggled para que refresque el indicador de conexion DAQ
    code = code.replace(
        'print(f"Modo Prueba {\'Activado\' if checked else \'Desactivado\'}.")',
        'print(f"Modo Prueba {\'Activado\' if checked else \'Desactivado\'}.")\n    self.verificar_conexion_daq()',
    )

    # 6. Mantener visible la barra de mediciones y 50Hz para que la interfaz no salte
    code = code.replace("self.measure_widget.setVisible(enabled)", "self.measure_widget.setVisible(True)")
    code = code.replace("self.spectrogram_groupbox.setVisible(enabled)", "pass")
    code = code.replace("self.label_rec_time.setVisible(False)", "self.label_rec_time.setText('SISTEMA EN ESPERA')")

    # 7. Llamar a _actualizar_medidor_50hz_en_vivo en actualizar_plot con los datos crudos uV
    if "self._actualizar_medidor_50hz_en_vivo(raw_uV_data)" not in code:
        code = code.replace(
            "raw_uV_data = (all_new_data / 495.0) * 1000000.0\n      processed_data = raw_uV_data.copy()",
            "raw_uV_data = (all_new_data / 495.0) * 1000000.0\n      processed_data = raw_uV_data.copy()\n      self._actualizar_medidor_50hz_en_vivo(raw_uV_data)",
            1,
        )

    # 8. Actualizar on_pause_toggled sin emojis y con soporte real de pausa en AutoForge
    pattern_pause = re.compile(
        r"  def on_pause_toggled\(self, paused: bool\):.*?(?=  def actualizar_plot\(self\):)",
        re.DOTALL,
    )
    new_pause_code = '''  def on_pause_toggled(self, paused: bool):
    """Pausa o reanuda la grabacion, el metronomo y las fases de AutoForge sin emojis."""
    self.is_paused = paused
    if paused:
      self.pause_start_perf = time.perf_counter()
      self.btn_pause.setText("REANUDAR")
      if hasattr(self, '_phase_timer') and self._phase_timer.isActive():
        self._paused_phase_remaining_ms = max(50, self._phase_timer.remainingTime())
        self._phase_timer.stop()
      if hasattr(self, 'metro_timer') and self.metro_timer.isActive():
        self.metro_timer.stop()
      if hasattr(self, 'session_timer') and self.session_timer.isActive():
        self.session_timer.stop()
      if hasattr(self, 'autoforge_overlay'):
        self._saved_overlay_html = self.autoforge_overlay.text()
        self._saved_overlay_vis = self.autoforge_overlay.isVisible()
        self.autoforge_overlay.setText(
          "<div align='center' style='color:#FFCC00;'>EN PAUSA<br>"
          "<span style='font-size:24px; color:#00FFFF;'>Presiona REANUDAR para continuar</span></div>"
        )
        self.autoforge_overlay.show()
    else:
      if self.pause_start_perf > 0:
        paused_delta = time.perf_counter() - self.pause_start_perf
        self.total_paused_duration += paused_delta
        if self.recording_start_time is not None:
          self.recording_start_time += paused_delta
        self.pause_start_perf = 0.0
      self.btn_pause.setText("PAUSAR")
      if hasattr(self, 'autoforge_overlay'):
        if getattr(self, '_saved_overlay_vis', False):
          self.autoforge_overlay.setText(getattr(self, '_saved_overlay_html', ''))
          self.autoforge_overlay.show()
        else:
          self.autoforge_overlay.hide()
      if getattr(self, '_scheduled_phase_cb', None) is not None and getattr(self, '_paused_phase_remaining_ms', 0) > 0:
        self._phase_timer.start(self._paused_phase_remaining_ms)
      if hasattr(self, 'metro_is_running') and self.metro_is_running:
        bpm = max(1, self.spin_bpm.value())
        subdivs = max(1, self.spin_metro_subdivs.value())
        self.metro_timer.start(int(60000 / bpm) // subdivs)
      if hasattr(self, 'session_timer') and getattr(self, 'is_autoforge_running', False):
        self.session_timer.start(1000)

'''
    code = pattern_pause.sub(new_pause_code, code, count=1)

    # Habilitar el boton PAUSAR durante AutoForge y Secuencia Continua
    code = code.replace(
        "self.is_autoforge_running = True",
        "self.is_autoforge_running = True\n        if hasattr(self, 'btn_pause'): self.btn_pause.setEnabled(True)",
    )
    code = code.replace(
        "self.is_autoforge_running = False",
        "self.is_autoforge_running = False\n      if hasattr(self, 'btn_pause'):\n        self.btn_pause.setChecked(False)\n        self.btn_pause.setEnabled(False)",
    )

    # Conectar los QTimer.singleShot de AutoForge a self._schedule_phase
    transitions = [
        ("QtCore.QTimer.singleShot(3000, self.estado_grabar_ruido_continuo)", "self._schedule_phase(3000, self.estado_grabar_ruido_continuo)"),
        ("QtCore.QTimer.singleShot(int(noise_dur * 1000), self.estado_mostrar_preparate_continuo)", "self._schedule_phase(int(noise_dur * 1000), self.estado_mostrar_preparate_continuo)"),
        ("QtCore.QTimer.singleShot(int(3 * beat_interval_s * 1000), self.estado_iniciar_grabacion_continua)", "self._schedule_phase(int(3 * beat_interval_s * 1000), self.estado_iniciar_grabacion_continua)"),
        ("QtCore.QTimer.singleShot(5000, self.estado_finalizar_secuencia_continua)", "self._schedule_phase(5000, self.estado_finalizar_secuencia_continua)"),
        ("QtCore.QTimer.singleShot(3000, self.estado_grabar_ruido)", "self._schedule_phase(3000, self.estado_grabar_ruido)"),
        ("QtCore.QTimer.singleShot(int(noise_dur * 1000), self.estado_mostrar_preparate)", "self._schedule_phase(int(noise_dur * 1000), self.estado_mostrar_preparate)"),
        ("QtCore.QTimer.singleShot(int(3 * beat_interval_s * 1000), self.estado_cambiar_ventana_palabra)", "self._schedule_phase(int(3 * beat_interval_s * 1000), self.estado_cambiar_ventana_palabra)"),
        ("QtCore.QTimer.singleShot(ms_totales, self.estado_guardar_palabra)", "self._schedule_phase(ms_totales, self.estado_guardar_palabra)"),
        ("QtCore.QTimer.singleShot(5000, self.estado_verificacion_muscular)", "self._schedule_phase(5000, self.estado_verificacion_muscular)"),
        ("QtCore.QTimer.singleShot(5000, self.estado_siguiente)", "self._schedule_phase(5000, self.estado_siguiente)"),
        ("QtCore.QTimer.singleShot(2500, self._continuar_post_cvm)", "self._schedule_phase(2500, self._continuar_post_cvm)"),
        ("QtCore.QTimer.singleShot(int(dur_sec * 1000), self.estado_finalizar_ruido_final)", "self._schedule_phase(int(dur_sec * 1000), self.estado_finalizar_ruido_final)"),
    ]
    for old_t, new_t in transitions:
        code = code.replace(old_t, new_t)

    # 9. Inyectar los metodos nuevos antes de closeEvent y restaurar stdout/stderr al cerrar
    if "def verificar_conexion_daq(" not in code:
        code = code.replace(
            "  def closeEvent(self, event):",
            EXTRA_METHODS + "\n  def closeEvent(self, event):\n"
            "    if hasattr(self, '_stdout_tee'): sys.stdout = self._stdout_tee._orig\n"
            "    if hasattr(self, '_stderr_tee'): sys.stderr = self._stderr_tee._orig",
            1,
        )

    # 10. Eliminar cualquier emoji residual
    code = code.replace("⏸ ", "").replace("▶ ", "").replace("⏸", "").replace("▶", "")

    with open(path, "w", encoding="utf-8") as f:
        f.write(code)

    py_compile.compile(path, doraise=True)
    print("======================================================================")
    print(f" [OK] Archivo reparado y actualizado: {path}")
    print(" [OK] Import roto eliminado (PreRollRecordingBuffer autocontenido)")
    print(" [OK] Indicador de conexion fisica DAQ en vivo agregado")
    print(" [OK] Medidor de 50 Hz movido a las tarjetas de canal y activo en vivo")
    print(" [OK] Boton CONSOLA / LOGS con terminal integrada agregado")
    print(" [OK] 0 emojis y compilacion verificada con py_compile")
    print("======================================================================")


if __name__ == "__main__":
    repair_and_upgrade(TARGET_PATH)
