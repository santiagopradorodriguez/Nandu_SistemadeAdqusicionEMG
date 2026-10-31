# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Adquisición de datos automatizada y comunicación con hardware EMG.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Adquisición de datos automatizada y comunicación con hardware EMG.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
# =============================================================================
# --- ESTA ES LA ÚLTIMA VERSIÓN FUNCIONAL CONOCIDA (v4.3) ---
- La aplicación ya no inicia la adquisición automáticamente.
- La aplicación ya no inicia la adquisición automáticamente.
- Se debe configurar el dispositivo y los canales, y luego presionar "Iniciar Adquisición".
- La interfaz (gráficos, mediciones) se adapta dinámicamente al número de canales seleccionados.
- El hilo de adquisición se crea y destruye con los botones de Iniciar/Detener.
- Mantiene todas las funcionalidades anteriores (grabación, exportación, trigger, etc.).
- NUEVO: Se puede configurar el Sample Rate y la Duración del Ploteo desde la GUI.
"""

# --- Versión del script ---
__version__ = "6.0.0"

import os as _os_env
_os_env.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
_os_env.environ.setdefault("OPENCV_VIDEOIO_DEBUG", "0")

import numpy as np
from scipy.io.wavfile import write as write_wav
from datetime import datetime
import time
import queue
import threading
import sys
import os
import re

# --- SPLASH SCREEN LOGIC (CARGA DE MÓDULOS PESADOS) ---
from PySide6.QtWidgets import QApplication, QSplashScreen, QProgressBar
from PySide6.QtGui import QPixmap, QColor, QIcon
from PySide6.QtCore import Qt
import time

app = QApplication.instance()
if not app:
  app = QApplication(sys.argv)

logo_path = None
try:
  from pathlib import Path
  root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  search_dirs = [Path(os.path.dirname(os.path.abspath(__file__))) / "assets", Path(os.path.dirname(os.path.abspath(__file__))), root_dir, Path.home() / "Pictures"]
  for search_dir in search_dirs:
    if search_dir.exists():
      for filename in os.listdir(search_dir):
        if filename.lower().startswith("logo") and filename.lower().endswith((".png", ".jpg", ".jpeg")):
          logo_path = str(search_dir / filename)
          break
    if logo_path: break
except Exception: pass

global_splash = None
global_progress = None
if logo_path and os.path.exists(logo_path):
  pixmap = QPixmap(logo_path)
  if pixmap.width() > 800: pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
  global_splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
  
  global_progress = QProgressBar(global_splash)
  global_progress.setGeometry(10, pixmap.height() - 30, pixmap.width() - 20, 20)
  global_progress.setStyleSheet("QProgressBar { border: 1px solid white; border-radius: 5px; text-align: center; color: white; font-weight: bold; } QProgressBar::chunk { background-color: #00ffcc; width: 10px; }")
  global_progress.setValue(10)
  
  global_splash.show()
  global_splash.showMessage("Cargando entorno base...", Qt.AlignBottom | Qt.AlignCenter, QColor("white"))
  app.processEvents()

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
  sys.path.insert(0, _curr_dir)
_parent_dir = os.path.dirname(_curr_dir)
if _parent_dir not in sys.path:
  sys.path.insert(0, _parent_dir)

try:
  from utils.config_manager import ConfigManager, STANDARD_MUSCLES_LIST
  from utils.path_utils import get_project_root
except ModuleNotFoundError:
  try:
    from config_manager import ConfigManager, STANDARD_MUSCLES_LIST
  except ImportError:
    ConfigManager = None
    STANDARD_MUSCLES_LIST = []
  try:
    from path_utils import get_project_root
  except ImportError:
    def get_project_root(): return Path(__file__).resolve().parent

METRONOME_CONFIG_PATH = os.path.join(get_project_root(), 'metronome_config.json')
PALABRAS_TXT_PATH = os.path.join(get_project_root(), 'palabras.txt')

import json
import subprocess

_PORTAUDIO_LOCK = threading.RLock()

try:
  from utils.sound_utils import play_beep as _raw_play_beep
except ImportError:
  _raw_play_beep = None

def play_beep(freq=660, duration_ms=80, async_play=True):
  # Generador neuro-ergonomico de pulsos acusticos suaves
  # Evita el reflejo de sobresalto (startle reflex)
  with _PORTAUDIO_LOCK:
    try:
      import sounddevice as _sd
      sr = 44100
      dur_s = max(0.025, min(float(duration_ms) / 1000.0, 0.25))
      n_samples = int(sr * dur_s)
      t = np.linspace(0.0, dur_s, n_samples, endpoint=False)
      # Mapear frecuencias agudas clásicas (1200 Hz / 800 Hz) a tonos cálidos de percusión tonal
      f_base = float(freq)
      if f_base >= 1100:
        f0 = 659.25  # Mi5 cálido (pulso principal de fonación)
        amp = 0.28
        tau_decay = dur_s * 0.42
      elif f_base >= 750:
        f0 = 523.25  # Do5 suave (pre-aviso / count-in)
        amp = 0.20
        tau_decay = dur_s * 0.38
      else:
        f0 = max(260.0, f_base * 0.75)  # Sub-pulso suave de aproximación
        amp = 0.12
        tau_decay = dur_s * 0.32

      # Onda fundamental + segundo armónico suave tipo marimba
      wave = np.sin(2.0 * np.pi * f0 * t) + 0.25 * np.sin(2.0 * np.pi * (f0 * 2.0) * t)
      # Envolvente: fade-in coseno elevado (10 ms) + decaimiento exponencial suave
      attack_samples = max(1, min(int(0.010 * sr), n_samples // 3))
      env = np.exp(-t / max(0.015, tau_decay))
      env[:attack_samples] *= 0.5 * (1.0 - np.cos(np.pi * np.arange(attack_samples) / float(attack_samples)))
      env[-min(64, n_samples):] *= np.linspace(1.0, 0.0, min(64, n_samples))
      audio_out = (amp * wave * env).astype(np.float32)
      _sd.play(audio_out, samplerate=sr, blocking=not async_play)
      return
    except Exception:
      if _raw_play_beep is not None:
        return _raw_play_beep(freq=min(int(freq), 880), duration_ms=min(int(duration_ms), 120), async_play=async_play)

try:
  from display_remoto_servidor import ServidorEstimulos, _ip_local, _ip_tailscale, configurar_adb_reverse, configurar_red_tailscale
except ImportError:
  try:
    from acquisition.display_remoto_servidor import ServidorEstimulos, _ip_local, _ip_tailscale, configurar_adb_reverse, configurar_red_tailscale
  except ImportError:
    ServidorEstimulos = None
    def _ip_local(): return "127.0.0.1"
    def _ip_tailscale(default_ip="100.118.12.94"): return default_ip
    def configurar_adb_reverse(puerto=8765): return False, "display_remoto_servidor no disponible"
    def configurar_red_tailscale(puerto=8765, ip_pc="100.118.12.94", ip_celu="100.102.47.75"): return False, "display_remoto_servidor no disponible"


try:
  from acquisition.fotometria_webcam import WebcamPhotometryManager
except ImportError:
  try:
    from fotometria_webcam import WebcamPhotometryManager
  except ImportError:
    WebcamPhotometryManager = None

if global_progress: global_progress.setValue(30); global_splash.showMessage("Cargando PyQtGraph...", Qt.AlignBottom | Qt.AlignCenter, QColor("white")); app.processEvents()
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui 

# NUEVO: Import para SpinBox y Línea
from pyqtgraph import SpinBox, InfiniteLine

# ==============================================================================
# --- ESTILOS VISUALES CYBERPUNK PRO (FABFILTER / TEKTRONIX / REAPER) ---
# ==============================================================================

CYBERPUNK_QSS = """
QWidget {
    background-color: #050508;
    color: #00FFCC;
    font-family: 'Courier New', Courier, monospace;
    font-size: 12px;
}

/* Agrupadores tipo Rack de Laboratorio */
QGroupBox {
    background-color: #080C14;
    border: 1px solid #00FFFF;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #00FFFF;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background-color: #080C14;
    color: #00FFFF;
}

/* Etiquetas y Checkboxes */
QLabel {
    color: #00FFCC;
    font-weight: bold;
    background-color: transparent;
}
QCheckBox {
    color: #00FFCC;
    font-weight: bold;
    spacing: 6px;
    background-color: transparent;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border-radius: 3px;
    border: 1px solid #00FFFF;
    background-color: #0B0F19;
}
QCheckBox::indicator:hover {
    border: 1px solid #00FFCC;
    background-color: #112233;
}
QCheckBox::indicator:checked {
    background-color: #00FFFF;
    border: 1px solid #00FFFF;
}

/* PyQtGraph SpinBox: borde y fondo exterior compacto con separación limpia */
SpinBox {
    background-color: #0B0F19;
    border: 1.5px solid #00FFFF;
    border-radius: 4px;
    min-height: 22px;
    max-height: 24px;
    padding: 0px 2px;
    margin: 1px 0px;
}
SpinBox:hover {
    border: 1.5px solid #39FF14;
}
SpinBox QLineEdit {
    border: none;
    background: transparent;
    color: #00FFCC;
    padding: 0px 3px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    font-weight: bold;
    min-height: 18px;
}
SpinBox:hover QLineEdit {
    color: #39FF14;
}

/* Campos de entrada, Combobox y LineEdit analógicos */
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 4px;
    padding: 1px 5px;
    margin: 1px 0px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    font-weight: bold;
    min-height: 22px;
    max-height: 24px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1.5px solid #39FF14;
    color: #39FF14;
}
QComboBox QAbstractItemView {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1px solid #00FFFF;
    selection-background-color: #003344;
    selection-color: #39FF14;
}

/* Botones Estándar Cyberpunk Pro */
QPushButton {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 5px;
    padding: 6px 14px;
    font-weight: bold;
    font-family: 'Courier New', Courier, monospace;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
    border: 1.5px solid #FFFFFF;
}
QPushButton:pressed {
    background-color: #0088AA;
    color: #FFFFFF;
}
QPushButton:disabled {
    background-color: #06080C;
    color: #334455;
    border: 1.5px solid #1A2233;
}
"""

# Botones de Transporte y Comandos Maestros
BTN_START_CYBER = """
QPushButton {
    background-color: rgba(0, 255, 128, 0.12);
    color: #39FF14;
    border: 2px solid #39FF14;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 900;
    padding: 8px 16px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #39FF14;
    color: #050508;
}
"""

BTN_STOP_CYBER = """
QPushButton {
    background-color: rgba(255, 0, 85, 0.15);
    color: #FF0055;
    border: 2px solid #FF0055;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 900;
    padding: 8px 16px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #FF0055;
    color: #FFFFFF;
}
"""

BTN_REC_START_CYBER = """
QPushButton {
    background-color: #0B0F19;
    color: #00FFFF;
    border: 2px solid #00FFFF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
}
"""

BTN_REC_STOP_CYBER = """
QPushButton {
    background-color: #FF0055;
    color: #FFFFFF;
    border: 2px solid #FFAACC;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
}
QPushButton:hover {
    background-color: #CC0044;
}
"""

BTN_PAUSE_CYBER = """
QPushButton {
    background-color: rgba(255, 204, 0, 0.12);
    color: #FFCC00;
    border: 2px solid #FFCC00;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: rgba(255, 204, 0, 0.25);
}
QPushButton:checked {
    background-color: #FFCC00;
    color: #050508;
    border: 2px solid #FFFFFF;
}
QPushButton:disabled {
    background-color: #06080C;
    color: #333333;
    border: 1.5px solid #1A2233;
}
"""

BTN_AUTOFORGE_DISC_CYBER = """
QPushButton {
    background-color: rgba(255, 0, 255, 0.12);
    color: #FF00FF;
    border: 2px solid #FF00FF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #FF00FF;
    color: #050508;
}
"""

BTN_AUTOFORGE_CONT_CYBER = """
QPushButton {
    background-color: rgba(170, 0, 255, 0.14);
    color: #CC44FF;
    border: 2px solid #AA00FF;
    border-radius: 5px;
    font-size: 12px;
    font-weight: 900;
    padding: 7px 14px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #AA00FF;
    color: #FFFFFF;
}
"""

BTN_CYBER_TOOL = """
QPushButton {
    background-color: #0B0F19;
    color: #00FFCC;
    border: 1.5px solid #00FFFF;
    border-radius: 5px;
    font-size: 11px;
    font-weight: 900;
    padding: 7px 10px;
    font-family: 'Courier New', monospace;
}
QPushButton:hover {
    background-color: #00FFFF;
    color: #050508;
    border: 1.5px solid #FFFFFF;
}
"""


# ==============================================================================
# --- COMPONENTES INTEGRADOS: PRE-ROLL, TERMINAL DE DIAGNOSTICO Y MONITOR DAQ ---
# ==============================================================================
from collections import deque as _deque


class VoiceCuePlayer:
  """
  Reproductor asíncrono de comandos de voz ("haz_silencio", "preparate",
  "descanso", "di_a", "di_e", "di_i", "di_o", "di_u") sin repetir jamás "A, A, A, A"
  en pulsos consecutivos y con registro de auditoría completo para el experimento.
  Modos soportados (self.speaking_mode):
    - "fases_y_continua": Habla en Ruido Basal, Preparación y Descanso; en Secuencia Continua dice "Di A", "Di E"... sólo al cambiar de vocal (nunca repite en pulsos iguales).
    - "solo_fases": Habla únicamente en Ruido Basal ("Haz silencio"), Preparación inicial ("Prepárate, di X") y Descanso (nunca durante los pulsos del metrónomo).
    - "mute": Silenciado total (equivalente a SPEAKING: MUTE).
  """
  def __init__(self, parent_plotter=None):
    self.enabled = True
    self.muted = False
    self.speaking_mode = "fases_y_continua"
    self.parent_plotter = parent_plotter
    self.history = []
    self.speech_blanking_windows = []
    self._cache = {}
    self._queue = queue.Queue()
    self._search_dirs = [
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice_cues"),
      os.path.join(get_project_root(), "assets", "voice_cues"),
      os.path.join(get_project_root(), "voice_cues"),
      os.path.join(get_project_root(), "assets", "sounds"),
      os.path.join(get_project_root(), "sounds"),
    ]
    os.environ["NANDU_MUTE_SPEAKING"] = "0"
    self._worker = threading.Thread(target=self._run_worker, daemon=True)
    self._worker.start()

  def reset_session_history(self):
    self.history = []
    self.speech_blanking_windows = []

  def set_muted(self, muted: bool):
    """Activa o desactiva el modo Mute Speaking en toda la aplicación y módulo de cámara."""
    self.muted = bool(muted)
    os.environ["NANDU_MUTE_SPEAKING"] = "1" if self.muted else "0"
    if self.muted:
      while not self._queue.empty():
        try:
          self._queue.get_nowait()
        except queue.Empty:
          break
      try:
        if SD_AVAILABLE:
          with _PORTAUDIO_LOCK:
            sd.stop()
      except Exception:
        pass
      if sys.platform == "win32":
        try:
          import winsound
          winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
          pass
    pm = getattr(self.parent_plotter, 'photometry_mgr', None) if self.parent_plotter else None
    if pm is not None:
      for attr_true_when_muted in ("mute_speaking", "mute_audio", "is_muted", "audio_muted"):
        if hasattr(pm, attr_true_when_muted):
          try:
            setattr(pm, attr_true_when_muted, self.muted)
          except Exception:
            pass
      for attr_false_when_muted in ("voice_enabled", "tts_enabled", "speaking_enabled", "audio_enabled"):
        if hasattr(pm, attr_false_when_muted):
          try:
            setattr(pm, attr_false_when_muted, not self.muted)
          except Exception:
            pass
      if hasattr(pm, "set_mute_speaking") and callable(pm.set_mute_speaking):
        try:
          pm.set_mute_speaking(self.muted)
        except Exception:
          pass

  def say(self, cue_key: str, fallback_text: str = "", fase: str = "fase"):
    """
    Encola un mensaje hablado y lo registra en self.history para trazabilidad del protocolo.
    Si fase == 'pulso_continua' y speaking_mode == 'solo_fases', no emite voz pero deja constancia.
    """
    now_perf = time.perf_counter()
    t0 = getattr(self.parent_plotter, 'recording_start_time', None) if self.parent_plotter else None
    t_rel = round(now_perf - t0, 4) if t0 is not None else 0.0
    is_Allowed = (
      self.enabled
      and not self.muted
      and self.speaking_mode != "mute"
      and not (fase == "pulso_continua" and self.speaking_mode == "solo_fases")
    )
    entry = {
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
      "t_perf_s": round(now_perf, 4),
      "t_rel_s": t_rel,
      "fase": str(fase),
      "cue_key": str(cue_key).strip().lower(),
      "texto": str(fallback_text or cue_key),
      "emitido": bool(is_Allowed),
      "muted": bool(self.muted or not self.enabled or self.speaking_mode == "mute"),
      "speaking_mode": str(self.speaking_mode),
    }
    self.history.append(entry)
    if self.parent_plotter is not None and hasattr(self.parent_plotter, 'spoken_audio_log'):
      self.parent_plotter.spoken_audio_log.append(entry)

    if not is_Allowed:
      return
    try:
      while not self._queue.empty():
        try:
          self._queue.get_nowait()
        except queue.Empty:
          break
      # Registrar ventana de blanking acústico (~0.50 s) para que el detector de micrófono
      # no confunda la voz emitida por el parlante con la voz del sujeto.
      self.speech_blanking_windows.append((now_perf - 0.02, now_perf + 0.55))
      if len(self.speech_blanking_windows) > 100:
        self.speech_blanking_windows = self.speech_blanking_windows[-50:]
      self._queue.put_nowait((str(cue_key).strip().lower(), str(fallback_text or cue_key)))
    except Exception:
      pass

  def _find_wav(self, key: str):
    if key in self._cache:
      return self._cache[key]
    slug = re.sub(r"[^a-z0-9_]", "_", key.lower())
    # Si la clave es "di_a", buscar primero "di_a.wav", luego "cue_di_a.wav", y como último recurso "a.wav"
    candidates = [f"{slug}.wav", f"vocal_{slug}.wav", f"cue_{slug}.wav"]
    if slug.startswith("di_") and len(slug) > 3:
      base_v = slug[3:]
      candidates.extend([f"di_{base_v}.wav", f"{base_v}.wav"])
    for d in self._search_dirs:
      if os.path.isdir(d):
        for cname in candidates:
          cand = os.path.join(d, cname)
          if os.path.isfile(cand):
            self._cache[key] = cand
            return cand
    self._cache[key] = None
    return None

  def _run_worker(self):
    while True:
      item = self._queue.get()
      if item is None:
        break
      key, text = item
      if not self.enabled or self.muted:
        continue
      # 1. Intentar reproducir WAV pre-generado en disco (latencia mínima)
      wav_path = self._find_wav(key)
      if wav_path and os.path.isfile(wav_path):
        try:
          if SD_AVAILABLE:
            from scipy.io import wavfile
            sr_w, data_w = wavfile.read(wav_path)
            if not self.muted:
              with _PORTAUDIO_LOCK:
                sd.play(data_w, sr_w, blocking=False)
            continue
          elif sys.platform == "win32" and not self.muted:
            import winsound
            winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            continue
        except Exception:
          pass
      # 2. Intentar usar helper de voz existente en utils.sound_utils si está definido
      try:
        import utils.sound_utils as _su
        for fn_name in ("play_voice_cue", "speak_async", "say_text", "speak_text"):
          fn = getattr(_su, fn_name, None)
          if callable(fn) and not self.muted:
            fn(text)
            break
      except Exception:
        pass

class PreRollRecordingBuffer:
  """Buffer circular autocontenido para rescatar los ultimos N segundos antes de grabar."""
  def __init__(self, sample_rate: int = 2000, preroll_seconds: float = 1.0, num_channels: int = 0):
    self.sample_rate = int(sample_rate)
    self.preroll_seconds = max(0.0, float(preroll_seconds))
    self._num_channels = int(num_channels)
    self._chunks = _deque()
    self._total_samples = 0

  @property
  def max_samples(self) -> int:
    return max(0, int(self.sample_rate * self.preroll_seconds))

  def set_config(self, sample_rate: int, preroll_seconds: float, num_channels: int = None) -> None:
    self.sample_rate = int(sample_rate)
    self.preroll_seconds = max(0.0, float(preroll_seconds))
    if num_channels is not None and int(num_channels) != self._num_channels:
      self._num_channels = int(num_channels)
      self.clear()
    else:
      self._trim()

  def clear(self) -> None:
    self._chunks.clear()
    self._total_samples = 0

  def push_raw_chunk(self, raw_chunk_volts: np.ndarray) -> None:
    if self.max_samples <= 0 or raw_chunk_volts is None or raw_chunk_volts.shape[1] == 0:
      return
    if self._num_channels > 0 and raw_chunk_volts.shape[0] != self._num_channels:
      self._num_channels = raw_chunk_volts.shape[0]
      self.clear()
    elif self._num_channels == 0:
      self._num_channels = raw_chunk_volts.shape[0]
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



# --- INYECCION REAPER ARCHITECTURE ---
from PySide6.QtCore import QThread, Signal

class RingBuffer:
    """
    Buffer circular basado en NumPy estático para evitar recolección de basura.
    Almacena datos en la forma (canales, muestras) para coincidir de forma nativa
    con el formato de salida de NI-DAQmx.
    """
    def __init__(self, capacity: int, num_channels: int):
        self.capacity = capacity
        self.num_channels = num_channels
        # Array estático gigante pre-asignado
        self.buffer = np.zeros((num_channels, capacity), dtype=np.float64)
        self.head = 0
        self.is_full = False
        self._lock = threading.Lock()
        
    def push(self, data: np.ndarray):
        """
        Inserta datos en el buffer circular.
        Se asume que data tiene la forma (num_channels, samples).
        """
        samples = data.shape[1]
        if samples == 0:
            return

        with self._lock:
            if self.head + samples <= self.capacity:
                self.buffer[:, self.head:self.head+samples] = data
                self.head += samples
            else:
                first_part = self.capacity - self.head
                self.buffer[:, self.head:self.capacity] = data[:, :first_part]
                second_part = samples - first_part
                self.buffer[:, 0:second_part] = data[:, first_part:]
                self.head = second_part
                self.is_full = True

    def get_latest(self, num_samples: int) -> np.ndarray:
        """
        Retorna las últimas num_samples muestras añadidas.
        """
        with self._lock:
            if num_samples > self.capacity:
                num_samples = self.capacity
            
            if self.head >= num_samples:
                return self.buffer[:, self.head-num_samples:self.head].copy()
            else:
                if not self.is_full:
                    return self.buffer[:, :self.head].copy()
                else:
                    first_part = num_samples - self.head
                    res = np.empty((self.num_channels, num_samples), dtype=np.float64)
                    res[:, :first_part] = self.buffer[:, self.capacity-first_part:self.capacity]
                    res[:, first_part:] = self.buffer[:, :self.head]
                    return res


class DAQThread(QThread):
    """
    Hilo dedicado exclusivamente a la adquisición de datos de NI-DAQmx.
    Escribe directamente en el RingBuffer de forma optimizada.
    """
    data_acquired = Signal(int)
    error_occurred = Signal(str)

    def __init__(self, ring_buffer: RingBuffer, channels: list, sample_rate: float, read_chunk_size: int = 1000, parent=None):
        super().__init__(parent)
        self.ring_buffer = ring_buffer
        self.channels = channels
        self.sample_rate = sample_rate
        self.read_chunk_size = read_chunk_size
        self._is_running = False
        # Cola lock-free para eventos de control (ej. detener el hilo) desde el hilo principal
        self.event_queue = queue.Queue()

    def stop(self):
        """
        Señaliza al hilo para que se detenga de manera segura.
        """
        self._is_running = False
        self.event_queue.put("STOP")
        self.wait()

    def run(self):
        self._is_running = True
        num_channels = len(self.channels)
        
        # Buffer de lectura estático pre-asignado para evitar recolección de basura durante la adquisición
        read_buffer = np.zeros((num_channels, self.read_chunk_size), dtype=np.float64)

        try:
            with nidaqmx.Task() as task:
                # Configurar canales
                for ch in self.channels:
                    task.ai_channels.add_ai_voltage_chan(ch)
                
                # Configurar reloj y buffer interno de la tarjeta
                task.timing.cfg_samp_clk_timing(
                    rate=self.sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.read_chunk_size * 10
                )
                
                # Usar stream_readers.AnalogMultiChannelReader es más eficiente y permite volcar en arrays pre-asignados
                reader = AnalogMultiChannelReader(task.in_stream)
                task.start()
                
                while self._is_running:
                    # Verificar eventos externos de forma asíncrona y sin bloqueos
                    try:
                        event = self.event_queue.get_nowait()
                        if event == "STOP":
                            break
                    except queue.Empty:
                        pass
                    
                    try:
                        # Leer fragmentos directamente al buffer en memoria estática (sin crear nuevos objetos)
                        reader.read_many_sample(
                            read_buffer, 
                            number_of_samples_per_channel=self.read_chunk_size, 
                            timeout=0.1
                        )
                        
                        # Guardar al RingBuffer
                        self.ring_buffer.push(read_buffer)
                        
                        # Notificar al sistema principal, enviando solo el número para no pasar arrays pesados
                        self.data_acquired.emit(self.read_chunk_size)
                        
                    except nidaqmx.errors.DaqError as e:
                        # El código -200284 corresponde a un timeout en la lectura
                        # Se ignora para permitir revisar la cola de eventos y volver a intentar
                        if e.error_code == -200284:
                            continue
                        else:
                            raise e

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._is_running = False

from numba import jit

@jit(nopython=True, nogil=True)
def calculate_rms_envelope(buffer: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calcula la envolvente RMS sobre un buffer continuo usando una ventana deslizante.
    """
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
        
        # Ignorar NaNs que puedan venir del filtro
        if np.isnan(val_add): val_add = 0.0
        if np.isnan(val_sub): val_sub = 0.0
        
        sum_sq += val_add ** 2 - val_sub ** 2
        if sum_sq < 0.0 or np.isnan(sum_sq):
            sum_sq = 0.0
        out[i] = np.sqrt(sum_sq / window_size)
        
    return out

# --- NUEVO: Pre-compilar Numba para evitar lag al activarlo por primera vez ---
_dummy = calculate_rms_envelope(np.zeros(10, dtype=np.float64), 5)

@jit(nopython=True, nogil=True)
def moving_average(buffer: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calcula la media movil de un arreglo utilizando suma acumulativa.
    """
    n = len(buffer)
    out = np.zeros(n, dtype=np.float64)
    if window_size <= 0:
        return out
        
    current_sum = 0.0
    for i in range(min(window_size, n)):
        current_sum += buffer[i]
        out[i] = current_sum / (i + 1)
        
    for i in range(window_size, n):
        current_sum += buffer[i] - buffer[i - window_size]
        out[i] = current_sum / window_size
        
    return out

def generar_secuencia_balanceada_debruijn(
    vocales_base: list,
    num_ciclos: int = 5,
    seed: int = None
) -> list:
  """
  Genera una secuencia balanceada de longitud len(vocales_base) * num_ciclos
  basada en bloques balanceados de Williams / grafo euleriano de De Bruijn:
    1. Cada vocal aparece exactamente `num_ciclos` veces.
    2. Todas las transiciones dirigidas (v_i -> v_j, con v_i != v_j) ocurren de forma
       equitativa, eliminando el sesgo de coarticulación para el modelo de Deep Learning.
    3. Jamás se repite la misma vocal dos veces seguidas en la frontera entre ciclos.
  """
  items = [str(v).strip().upper() for v in (vocales_base or []) if str(v).strip()]
  if not items:
    items = ["A", "E", "I", "O", "U"]
  n = len(items)
  if n == 1 or num_ciclos <= 0:
    return items * max(1, int(num_ciclos))

  import random as _rnd
  rng = _rnd.Random(seed)
  trans_counts = [[0] * n for _ in range(n)]
  secuencia_indices = []

  for c in range(int(num_ciclos)):
    if c == 0:
      perm = list(range(n))
      rng.shuffle(perm)
    else:
      prev_last = secuencia_indices[-1]
      disponibles = list(range(n))
      candidatas_inicio = [idx for idx in disponibles if idx != prev_last]
      candidatas_inicio.sort(key=lambda idx: (trans_counts[prev_last][idx], rng.random()))
      primero = candidatas_inicio[0]
      perm = [primero]
      disponibles.remove(primero)
      while disponibles:
        actual = perm[-1]
        disponibles.sort(key=lambda idx: (trans_counts[actual][idx], rng.random()))
        sig = disponibles.pop(0)
        perm.append(sig)

    for idx in perm:
      if secuencia_indices:
        ant = secuencia_indices[-1]
        trans_counts[ant][idx] += 1
      secuencia_indices.append(idx)

  return [items[i] for i in secuencia_indices]


def calcular_metricas_espectrales_fatiga(sig_uV: np.ndarray, sample_rate: float) -> dict:
  """
  Calcula sobre un segmento de señal EMG cruda calibrada en uV:
    - rms_uV: valor eficaz en microvoltios
    - mdf_hz: Frecuencia Mediana del espectro de potencia (20-450 Hz)
    - mnf_hz: Frecuencia Media del espectro de potencia (20-450 Hz)
  La caída progresiva de mdf_hz a lo largo de los ciclos cuantifica la fatiga muscular.
  """
  res = {"rms_uV": 0.0, "mdf_hz": 0.0, "mnf_hz": 0.0}
  try:
    arr = np.asarray(sig_uV, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size < 16 or sample_rate <= 0:
      return res
    arr = arr - np.median(arr)
    res["rms_uV"] = round(float(np.sqrt(np.mean(arr ** 2))), 3)
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / float(sample_rate))
    psd = np.abs(np.fft.rfft(arr * np.hanning(arr.size))) ** 2
    mask = (freqs >= 20.0) & (freqs <= min(450.0, float(sample_rate) * 0.45))
    if not np.any(mask):
      return res
    f_band = freqs[mask]
    p_band = psd[mask]
    total_power = float(np.sum(p_band))
    if total_power <= 1e-12:
      return res
    cum_power = np.cumsum(p_band)
    idx_mdf = int(np.searchsorted(cum_power, 0.5 * total_power))
    idx_mdf = min(max(0, idx_mdf), len(f_band) - 1)
    res["mdf_hz"] = round(float(f_band[idx_mdf]), 2)
    res["mnf_hz"] = round(float(np.sum(f_band * p_band) / total_power), 2)
  except Exception:
    pass
  return res


def exportar_segmentos_dataset_y_fatiga(
    datos_recording: list,
    sample_rate: float,
    output_dir: str,
    num_canales: int,
    vocal_cue_log: list,
    sync_evaluation_log: list,
    invalid_pulses_set: set,
    factores_calibracion: list,
    emg_channel_indices: list,
    initial_noise_mean: list,
    initial_noise_std: list,
) -> dict:
  """
  Exporta automáticamente `segmentos_dataset.csv` con una fila por pulso de vocal
  lista para entrenamiento de Deep Learning, evaluando SNR, sincronización,
  frecuencia mediana (MDF), frecuencia media (MNF), índice de fatiga muscular y
  validez de cada pulso (descartando pulsos marcados manualmente por tos/error o sin activación).
  """
  resumen = {
    "archivo_csv": None,
    "total_pulsos": 0,
    "pulsos_validos_entrenamiento": 0,
    "pulsos_rechazados": 0,
    "pulsos_invalidados_manualmente": sorted(list(invalid_pulses_set or [])),
    "fatiga_muscular_por_canal": {},
    "segmentos": [],
  }
  if not vocal_cue_log or not datos_recording or sample_rate <= 0:
    return resumen

  try:
    import csv as _csv
    raw_concat = np.concatenate(datos_recording, axis=1)
    total_samples = raw_concat.shape[1]
    uV_matrix = np.zeros_like(raw_concat, dtype=np.float64)
    for ch in range(min(num_canales, raw_concat.shape[0])):
      gain = float(factores_calibracion[ch]) if (factores_calibracion and ch < len(factores_calibracion) and factores_calibracion[ch] > 0) else 495.0
      uV_matrix[ch, :] = (raw_concat[ch, :] / gain) * 1e6

    # Indexar sincronización por número de pulso
    sync_by_pulse = {}
    for s_ev in (sync_evaluation_log or []):
      p_num = int(s_ev.get("pulso_asociado", 0) or 0)
      if p_num > 0:
        sync_by_pulse[p_num] = s_ev

    emg_chs = list(emg_channel_indices) if emg_channel_indices else list(range(min(3, num_canales)))
    mdf_history_per_ch = {ch: [] for ch in emg_chs}
    rms_history_per_ch = {ch: [] for ch in emg_chs}

    rows = []
    for ev in vocal_cue_log:
      pulso_id = int(ev.get("pulso", 0) or 0)
      ciclo = int(ev.get("ciclo", 1) or 1)
      vocal = str(ev.get("vocal", "")).upper()
      sample_beat = int(ev.get("sample_idx", 0) or 0)
      tau_s = float(ev.get("tau_s", 2.0) or 2.0)
      t_beat_s = float(ev.get("t_rel_s", 0.0) or 0.0)

      s_info = sync_by_pulse.get(pulso_id, {})
      t_onset_s = s_info.get("t_onset_s", None)
      delta_sync_ms = s_info.get("delta_sync_ms", None)
      clasif_sync = str(s_info.get("clasificacion", "SIN_DETECCION_ONSET"))

      if t_onset_s is not None:
        sample_onset = int(round((float(t_onset_s) - t_beat_s) * sample_rate)) + sample_beat
      else:
        sample_onset = sample_beat

      win_ini = max(0, int(round(sample_onset - 0.10 * tau_s * sample_rate)))
      win_fin = min(total_samples, int(round(sample_onset + 0.45 * tau_s * sample_rate)))
      if win_fin <= win_ini:
        win_ini = max(0, sample_beat)
        win_fin = min(total_samples, sample_beat + int(round(0.50 * tau_s * sample_rate)))

      snr_list = []
      rms_list = []
      mdf_list = []
      mnf_list = []
      for ch in emg_chs:
        if ch < uV_matrix.shape[0] and win_fin > win_ini:
          seg_ch = uV_matrix[ch, win_ini:win_fin]
          met = calcular_metricas_espectrales_fatiga(seg_ch, sample_rate)
          rms_val = met["rms_uV"]
          mdf_val = met["mdf_hz"]
          mnf_val = met["mnf_hz"]
          base_std = float(initial_noise_std[ch]) if (initial_noise_std and ch < len(initial_noise_std) and initial_noise_std[ch] > 0) else 2.5
          snr_db = 20.0 * np.log10(max(rms_val, 0.1) / max(base_std, 0.5))
          snr_list.append(round(float(snr_db), 2))
          rms_list.append(rms_val)
          if mdf_val > 0:
            mdf_list.append(mdf_val)
            mdf_history_per_ch[ch].append(mdf_val)
          if mnf_val > 0:
            mnf_list.append(mnf_val)
          if rms_val > 0:
            rms_history_per_ch[ch].append(rms_val)

      snr_max_db = round(max(snr_list), 2) if snr_list else 0.0
      rms_max_uV = round(max(rms_list), 2) if rms_list else 0.0
      mdf_media_hz = round(float(np.mean(mdf_list)), 2) if mdf_list else 0.0
      mnf_media_hz = round(float(np.mean(mnf_list)), 2) if mnf_list else 0.0

      invalido_manual = bool(pulso_id in (invalid_pulses_set or set()))
      if invalido_manual:
        valido_train = False
        motivo = "marcado_invalido_manual_operador"
      elif (win_fin - win_ini) < int(0.10 * sample_rate):
        valido_train = False
        motivo = "ventana_incompleta"
      elif snr_max_db < 3.0 and t_onset_s is None:
        valido_train = False
        motivo = "baja_snr_sin_onset"
      else:
        valido_train = True
        motivo = "ok"

      row = {
        "pulso_id": pulso_id,
        "ciclo": ciclo,
        "vocal": vocal,
        "sample_beat": sample_beat,
        "sample_onset_real": sample_onset,
        "sample_ventana_ini": win_ini,
        "sample_ventana_fin": win_fin,
        "t_beat_s": round(t_beat_s, 4),
        "t_onset_s": round(float(t_onset_s), 4) if t_onset_s is not None else "",
        "delta_sync_ms": round(float(delta_sync_ms), 2) if delta_sync_ms is not None else "",
        "clasificacion_sync": clasif_sync,
        "snr_max_db": snr_max_db,
        "rms_max_uV": rms_max_uV,
        "mdf_media_hz": mdf_media_hz,
        "mnf_media_hz": mnf_media_hz,
        "invalido_manual": int(invalido_manual),
        "valido_para_entrenamiento": int(valido_train),
        "motivo_calidad": motivo,
      }
      rows.append(row)

    # Calcular evolución de fatiga por canal (primer tercio vs último tercio)
    for ch in emg_chs:
      mdfs = mdf_history_per_ch.get(ch, [])
      rmss = rms_history_per_ch.get(ch, [])
      if len(mdfs) >= 3:
        k = max(1, len(mdfs) // 3)
        mdf_ini = float(np.mean(mdfs[:k]))
        mdf_fin = float(np.mean(mdfs[-k:]))
        cambio_pct = ((mdf_fin - mdf_ini) / mdf_ini * 100.0) if mdf_ini > 0 else 0.0
        estado_fatiga = "FATIGA_DETECTADA" if cambio_pct <= -12.0 else "ESTABLE"
      else:
        mdf_ini = float(np.mean(mdfs)) if mdfs else 0.0
        mdf_fin = mdf_ini
        cambio_pct = 0.0
        estado_fatiga = "MUESTRAS_INSUFICIENTES"
      resumen["fatiga_muscular_por_canal"][f"canal_{ch}"] = {
        "mdf_inicial_hz": round(mdf_ini, 2),
        "mdf_final_hz": round(mdf_fin, 2),
        "variacion_mdf_pct": round(cambio_pct, 2),
        "rms_medio_uV": round(float(np.mean(rmss)), 2) if rmss else 0.0,
        "estado_fatiga": estado_fatiga,
      }

    os.makedirs(str(output_dir), exist_ok=True)
    csv_path = os.path.join(str(output_dir), "segmentos_dataset.csv")
    fieldnames = [
      "pulso_id", "ciclo", "vocal", "sample_beat", "sample_onset_real",
      "sample_ventana_ini", "sample_ventana_fin", "t_beat_s", "t_onset_s",
      "delta_sync_ms", "clasificacion_sync", "snr_max_db", "rms_max_uV",
      "mdf_media_hz", "mnf_media_hz", "invalido_manual",
      "valido_para_entrenamiento", "motivo_calidad"
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
      writer = _csv.DictWriter(fh, fieldnames=fieldnames)
      writer.writeheader()
      for r in rows:
        writer.writerow(r)

    resumen["archivo_csv"] = "segmentos_dataset.csv"
    resumen["total_pulsos"] = len(rows)
    resumen["pulsos_validos_entrenamiento"] = sum(1 for r in rows if r["valido_para_entrenamiento"] == 1)
    resumen["pulsos_rechazados"] = sum(1 for r in rows if r["valido_para_entrenamiento"] == 0)
    resumen["segmentos"] = rows
    print(
      f"[Dataset] segmentos_dataset.csv exportado: {resumen['pulsos_validos_entrenamiento']}/{resumen['total_pulsos']} "
      f"pulsos válidos para entrenamiento en {csv_path}"
    )
  except Exception as e:
    print(f"[Aviso Dataset] Error generando segmentos_dataset.csv: {e}")
  return resumen


def generate_jittered_intervals(bpm: float, n_pulses: int, jitter_pct: float, mode: str = "exponential_hazard", seed: int = None) -> list:
    """
    Genera una lista de intervalos inter-pulso (ISI) en segundos con Jitter controlado
    para evitar que el sujeto se sincronice o anticipe al metrónomo (anti-entrainment).
    Garantiza suma cero (duración media igual a 60/bpm) y piso fisiológico >= 0.75 s.
    """
    tau_0 = 60.0 / max(1.0, float(bpm))
    if jitter_pct <= 0.0 or n_pulses <= 0:
        return [tau_0] * max(1, int(n_pulses) if n_pulses > 0 else 1)
    rng = np.random.default_rng(seed)
    tau_min = max(0.75, tau_0 * (1.0 - float(jitter_pct)))
    tau_max = max(tau_min + 0.05, tau_0 * (1.0 + float(jitter_pct)))
    if mode == "gaussian":
        raw = rng.normal(loc=tau_0, scale=(float(jitter_pct) * tau_0) / 2.0, size=n_pulses)
    elif mode == "exponential_hazard":
        # Tasa de riesgo (Hazard Rate) constante tras tau_min: elimina predicción temporal
        scale = max(0.05, tau_0 - tau_min)
        raw = tau_min + rng.exponential(scale=scale, size=n_pulses)
    else:
        raw = rng.uniform(low=tau_min, high=tau_max, size=n_pulses)
    clipped = np.clip(raw, tau_min, tau_max)
    centered = clipped - np.mean(clipped) + tau_0
    return np.clip(centered, tau_min, tau_max).tolist()

def calcular_ruido_linea_50hz(noise_signal_uV: np.ndarray, fs: float) -> float:
    """
    Calcula el voltaje RMS en microvoltios (uV) de la componente de 50 Hz de la red eléctrica
    en un segmento de señal en reposo.
    """
    if noise_signal_uV is None or fs is None or fs <= 0 or len(noise_signal_uV) < int(fs * 0.2):
        return 0.0
    try:
        N = len(noise_signal_uV)
        fft_vals = np.fft.rfft(noise_signal_uV)
        freqs = np.fft.rfftfreq(N, 1.0 / fs)
        idx_50 = np.where((freqs >= 48.0) & (freqs <= 52.0))[0]
        if len(idx_50) == 0:
            return 0.0
        power_50 = np.sum(np.abs(fft_vals[idx_50]) ** 2) / (N ** 2)
        rms_50 = float(np.sqrt(2.0 * power_50))
        return rms_50
    except Exception as e:
        print(f"[Error 50Hz] {e}")
        return 0.0

@jit(nopython=True, nogil=True)
def apply_iir_filter(data: np.ndarray, b: np.ndarray, a: np.ndarray, zi: np.ndarray) -> tuple:
    """
    Aplica un filtro IIR a un bloque de datos usando coeficientes b, a y estado zi.
    Retorna la data filtrada y el nuevo estado zf.
    """
    n = len(data)
    m = len(a)
    out = np.zeros(n, dtype=np.float64)
    zf = np.copy(zi)
    
    for i in range(n):
        x_val = data[i]
        y_val = b[0] * x_val + zf[0]
        out[i] = y_val
        
        for j in range(m - 2):
            zf[j] = b[j + 1] * x_val - a[j + 1] * y_val + zf[j + 1]
            
        zf[m - 2] = b[m - 1] * x_val - a[m - 1] * y_val
        
    return out, zf

@jit(nopython=True, nogil=True)
def estimate_instantaneous_snr(signal_buffer: np.ndarray, noise_power: float) -> float:
    """
    Estima el SNR (Signal-to-Noise Ratio) instantaneo basado en la varianza
    de la senal respecto a una potencia de ruido conocida de fondo (50Hz u otro).
    Penaliza fuertemente niveles bajos.
    """
    n = len(signal_buffer)
    if n == 0:
        return 0.0
        
    mean = 0.0
    for i in range(n):
        mean += signal_buffer[i]
    mean /= n
    
    variance = 0.0
    for i in range(n):
        variance += (signal_buffer[i] - mean) ** 2
    variance /= n
    
    if variance <= noise_power or noise_power <= 1e-12:
        return 0.0
        
    signal_power = variance - noise_power
    if signal_power <= 0:
        return 0.0
        
    # En decibelios
    return 10.0 * np.log10(signal_power / noise_power)

def decimate_min_max(data_x, data_y, max_points=2000):
    """
    Decimación Dinámica usando Min/Max Peaking (Downsampling).
    Divide los datos en fragmentos y extrae el mínimo y el máximo de cada fragmentos.
    Esto preserva la envolvente visual de señales de alta frecuencia como el EMG,
    evitando el aliasing visual y reduciendo drásticamente la carga de renderizado.
    
    Args:
        data_x: Array Numpy de coordenadas X (tiempo).
        data_y: Array Numpy de coordenadas Y (amplitud).
        max_points: Número máximo de puntos a renderizar en pantalla.
        
    Returns:
        decimated_x, decimated_y: Arrays reducidos listos para graficar.
    """
    n = len(data_y)
    if n <= max_points:
        return data_x, data_y
        
    # Necesitamos max_points en la salida. Cada bloque aporta 2 puntos (min y max).
    num_chunks = max_points // 2
    chunk_size = n // num_chunks
    
    # Truncar arrays para que sean múltiplos exactos de chunk_size
    trunc_length = chunk_size * num_chunks
    y_trunc = data_y[:trunc_length].reshape((num_chunks, chunk_size))
    x_trunc = data_x[:trunc_length].reshape((num_chunks, chunk_size))
    
    # Encontrar los índices del mínimo y máximo dentro de cada fragmento
    min_indices = np.argmin(y_trunc, axis=1)
    max_indices = np.argmax(y_trunc, axis=1)
    
    # Extraer los valores correspondientes
    y_mins = y_trunc[np.arange(num_chunks), min_indices]
    x_mins = x_trunc[np.arange(num_chunks), min_indices]
    
    y_maxs = y_trunc[np.arange(num_chunks), max_indices]
    x_maxs = x_trunc[np.arange(num_chunks), max_indices]
    
    # Combinar los arrays intercalando mínimos y máximos
    x_combined = np.column_stack((x_mins, x_maxs))
    y_combined = np.column_stack((y_mins, y_maxs))
    
    # Ordenar cada par para asegurar que el eje X siga siendo monótonamente creciente
    sort_mask = x_combined[:, 0] > x_combined[:, 1]
    
    if np.any(sort_mask):
        # Intercambiar elementos donde el mínimo ocurrió después del máximo en el tiempo
        x_combined[sort_mask, 0], x_combined[sort_mask, 1] = x_combined[sort_mask, 1], x_combined[sort_mask, 0].copy()
        y_combined[sort_mask, 0], y_combined[sort_mask, 1] = y_combined[sort_mask, 1], y_combined[sort_mask, 0].copy()
    
    return x_combined.flatten(), y_combined.flatten()


# --- FIN INYECCION ---


# NUEVO: Import para el espectrograma
try:
  from scipy import signal
  SCIPY_DISPONIBLE = True
except ImportError:
  SCIPY_DISPONIBLE = False
  print("Advertencia: La librería 'scipy' no está instalada. El espectrograma no funcionará.")

# NUEVO: Import para generar el gráfico
# matplotlib.pyplot se importa de forma diferida en las funciones para que la GUI cargue más rápido.

if global_progress: global_progress.setValue(70); global_splash.showMessage("Cargando NIDAQmx...", Qt.AlignBottom | Qt.AlignCenter, QColor("white")); app.processEvents()

# Configuración de PyQtGraph para MÁXIMO rendimiento
pg.setConfigOptions(antialias=False) # Optimización

# Intenta importar nidaqmx
try:
  try:
    import sounddevice as sd
    SD_AVAILABLE = True
  except ImportError:
    SD_AVAILABLE = False

  import nidaqmx
  from nidaqmx.constants import AcquisitionType, TerminalConfiguration
  from nidaqmx.stream_readers import AnalogMultiChannelReader 
  NIDAQMX_DISPONIBLE = True
except Exception as e:
  import traceback
  print("--- TRACEBACK DE NIDAQMX ---")
  traceback.print_exc()
  NIDAQMX_DISPONIBLE = False
  print(f"Advertencia: La librería 'nidaqmx' falló al cargar: {e}")

if global_progress: global_progress.setValue(100); global_splash.showMessage("Iniciando AutoForge...", Qt.AlignBottom | Qt.AlignCenter, QColor("white")); app.processEvents()

# =============================================================================
# --- CONFIGURACIÓN PRINCIPAL ---
# =============================================================================
# CANALES_DAQ = [f"{DEVICE_NAME}/ai0", f"{DEVICE_NAME}/ai1", f"{DEVICE_NAME}/ai2"]
# NUM_CANALES = len(CANALES_DAQ)

# --- ESTOS VALORES AHORA SE CONFIGURAN DESDE LA GUI ---
# SAMPLE_RATE = 6000
# PLOT_DURATION_S = 30

# CHUNK_SAMPLES ahora se calcula dinámicamente al iniciar la adquisición
# CHUNK_DURATION_S y PLOT_SAMPLES se calculan dinámicamente
# =============================================================================
# BLOQUE 1: HILO DE ADQUISICIÓN (Sin cambios)
# =============================================================================
def acquisition_thread(device_channels, sample_rate, chunk_samples, num_canales, data_queue, stop_event, terminal_config_val=None):
  """
  Ejecuta la funcionalidad de acquisition_thread.

  Args:
    device_channels (Any): Argumento posicional device_channels.
    sample_rate (Any): Argumento posicional sample_rate.
    chunk_samples (Any): Argumento posicional chunk_samples.
    num_canales (Any): Argumento posicional num_canales.
    data_queue (Any): Argumento posicional data_queue.
    stop_event (Any): Argumento posicional stop_event.
    terminal_config_val (Any): Argumento posicional terminal_config_val.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  print(f"Iniciando hilo de adquisición con SR={sample_rate} Hz...")
  try:
    with nidaqmx.Task() as task:
      for canal in device_channels:
        # --- CORRECCIÓN: Forzar rango de voltaje explícito para evitar auto-escalado de la DAQ ---
        if terminal_config_val is not None:
          task.ai_channels.add_ai_voltage_chan(canal, terminal_config=terminal_config_val, min_val=-10.0, max_val=10.0)
        else:
          task.ai_channels.add_ai_voltage_chan(canal, terminal_config=TerminalConfiguration.RSE, min_val=-10.0, max_val=10.0)
      
      task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=AcquisitionType.CONTINUOUS,
        # --- MEJORA: Aumentar el tamaño del búfer de la DAQ ---
        # Un búfer pequeño (como chunk_samples * 2) puede causar un desbordamiento (overflow)
        # a altas frecuencias de muestreo si el hilo de Python no lo vacía a tiempo.
        # Al establecer un búfer grande (ej: 1 segundo de datos), le damos al sistema
        # mucho más margen, evitando la pérdida de muestras y asegurando una adquisición continua.
        # Esto soluciona el problema de que las señales reales se vean "triangulares" o mal definidas.
        samps_per_chan=sample_rate 
      )
      
      reader = AnalogMultiChannelReader(task.in_stream)
      buffer_daq = np.zeros((num_canales, chunk_samples))
      task.start()
      print("Hilo DAQ iniciado y listo.")
      
      while not stop_event.is_set():
        reader.read_many_sample(
          buffer_daq,
          number_of_samples_per_channel=chunk_samples,
          timeout=(chunk_samples / sample_rate) * 5
        )
        data_queue.put(buffer_daq.copy())

  except nidaqmx.errors.DaqError as e:
    print(f"\n--- ERROR FATAL EN HILO DAQ --- \n{e}")
  finally:
    print("Cerrando hilo de adquisición.")
    if not stop_event.is_set():
      stop_event.set()


def zmq_bridge_acquisition_thread(
  vm_endpoint_str,
  device_name,
  device_channels,
  sample_rate,
  chunk_samples,
  num_canales,
  data_queue,
  stop_event,
  terminal_config_str="RSE"
):
  """
  Hilo de adquisición en tiempo real a través del Puente ZeroMQ (Pop!_OS <-> VM Windows con NI USB-6259).
  Envía la configuración deseada al puerto de control (puerto+1) y consume los bloques
  multipart (Frame 1: JSON metadata, Frame 2: buffer binario NumPy) mediante `read_block()`.
  """
  print(f"[PUENTE ZMQ] Iniciando hilo cliente ZeroMQ hacia VM ({vm_endpoint_str}) con SR={sample_rate} Hz, Canales={device_channels}, Modo={terminal_config_str}...")
  client = None
  try:
    from daq_bridge_client import DAQBridgeClient, parse_host_port
    host, port = parse_host_port(vm_endpoint_str, default_port=5555)
    client = DAQBridgeClient(
      host=host,
      port=port,
      socket_mode="SUB",
      recv_timeout_ms=1500,
      reconnect_Wait_s=1.0
    )
    # Intentar sincronizar parámetros de adquisición en caliente con la VM Windows
    cfg_reply = client.configure_remote_server(
      device=device_name,
      channels=device_channels,
      terminal_config=terminal_config_str,
      sample_rate_hz=float(sample_rate),
      chunk_size=int(chunk_samples),
      min_val=-10.0,
      max_val=10.0,
      dtype="float64",
      timeout_ms=700
    )
    if cfg_reply and cfg_reply.get("status") == "reconfigured":
      print(f"[PUENTE ZMQ] Servidor VM Windows reconfigurado en vivo: {cfg_reply}")
    else:
      print(f"[PUENTE ZMQ] Conectado a stream de datos en {client.endpoint} (modo escucha directa).")

    for block_arr, meta in client.read_block(stop_event=stop_event):
      if stop_event.is_set():
        break
      arr_f64 = np.asarray(block_arr, dtype=np.float64)
      if arr_f64.ndim == 1:
        arr_f64 = arr_f64.reshape(1, -1)
      # Ajustar número de canales si la VM emite más o menos canales que los seleccionados en la GUI
      if arr_f64.shape[0] > num_canales:
        arr_f64 = arr_f64[:num_canales, :]
      elif arr_f64.shape[0] < num_canales:
        pad_ch = np.zeros((num_canales - arr_f64.shape[0], arr_f64.shape[1]), dtype=np.float64)
        arr_f64 = np.vstack([arr_f64, pad_ch])
      data_queue.put(arr_f64.copy())

  except Exception as e:
    print(f"\n--- ERROR EN HILO PUENTE ZMQ --- \n{e}")
  finally:
    if client is not None:
      try:
        client.close()
      except Exception:
        pass
    print("[PUENTE ZMQ] Cerrando hilo de adquisición ZeroMQ.")
    if not stop_event.is_set():
      stop_event.set()


def microphone_thread(chunk_samples, sample_rate, num_canales, data_queue, stop_event):
  """
  Ejecuta la funcionalidad de microphone_thread.

  Args:
    chunk_samples (Any): Argumento posicional chunk_samples.
    sample_rate (Any): Argumento posicional sample_rate.
    num_canales (Any): Argumento posicional num_canales.
    data_queue (Any): Argumento posicional data_queue.
    stop_event (Any): Argumento posicional stop_event.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  print(f"Iniciando hilo de MICRÓFONO con SR={sample_rate} Hz...")
  try:
    import sounddevice as sd
    def audio_callback(indata, frames, time_info, status):
      """
      Ejecuta la funcionalidad de audio_callback.

      Args:
        indata (Any): Argumento posicional indata.
        frames (Any): Argumento posicional frames.
        time_info (Any): Argumento posicional time_info.
        status (Any): Argumento posicional status.

      Returns:
        Any: Resultado de la ejecución de la función.
      """
      if not stop_event.is_set():
        import numpy as np
        data = indata[:, 0]
        buffer = np.zeros((num_canales, len(data)))
        for i in range(num_canales):
          buffer[i, :] = data
        data_queue.put(buffer)

    with sd.InputStream(samplerate=sample_rate, channels=1, callback=audio_callback, blocksize=chunk_samples):
      while not stop_event.is_set():
        import time
        time.sleep(0.1)
  except Exception as e:
    print(f"\n--- ERROR EN HILO MICRÓFONO --- \n{e}")
  finally:
    print("Cerrando hilo de micrófono.")

def simulador_thread(chunk_samples, sample_rate, num_canales, data_queue, stop_event, test_freq=50, tipo_prueba="Senoidal", active_channel_indices=None):
  """
  Ejecuta la funcionalidad de simulador_thread.

  Args:
    chunk_samples (Any): Argumento posicional chunk_samples.
    sample_rate (Any): Argumento posicional sample_rate.
    num_canales (Any): Argumento posicional num_canales.
    data_queue (Any): Argumento posicional data_queue.
    stop_event (Any): Argumento posicional stop_event.
    test_freq (Any): Argumento posicional test_freq.
    tipo_prueba (Any): Argumento posicional tipo_prueba.
    active_channel_indices (list, optional): Indices fisicos activos (ej. [0, 2]).

  Returns:
    Any: Resultado de la ejecucion de la funcion.
  """
  print(f"Iniciando hilo de simulacion con SR={sample_rate} Hz, Tipo={tipo_prueba}, Canales fisicos={active_channel_indices}...")
  
  datos_archivo = None
  total_samples_archivo = 0
  
  if "Cande" in tipo_prueba or "Archivo" in tipo_prueba:
    try:
      import pandas as pd
      from utils.path_utils import get_database_path, get_project_root
      
      db_path = get_database_path()
      csv_path = None

      if "Secuencia Continua" in tipo_prueba:
        csv_path = os.path.join(db_path, "2026-09-01", "SecuenciaContinua_Prueba1_Candela", "grabacion.csv")
      elif "Cande" in tipo_prueba:
        # Medicion normal estandar de Cande (A_Prueba1_Candela: 4 canales, 12 pulsos de /a/)
        csv_path = os.path.join(db_path, "2026-09-01", "A_Prueba1_Candela", "grabacion.csv")
      else:
        candidates = [
            os.path.join(db_path, "2026-09-01", "A_Prueba1_Candela", "grabacion.csv"),
            os.path.join(get_project_root(), "grabacion.csv"),
        ]
        for c in candidates:
          if os.path.exists(c):
            csv_path = c
            break

      if csv_path and os.path.exists(csv_path):
        print(f"[SIMULADOR] Cargando medicion de prueba real: {csv_path}")
        df = pd.read_csv(csv_path)
        canales_cols = [c for c in df.columns if "Canal" in c]
        if canales_cols:
          raw_all = df[canales_cols].values.T
          if active_channel_indices and len(active_channel_indices) == num_canales:
            selected_rows = []
            for hw_idx in active_channel_indices:
              col_found = None
              for c in df.columns:
                m_c = re.search(r'(?:Canal|ai)[_\s]*(\d+)', c, re.IGNORECASE)
                if m_c and int(m_c.group(1)) == hw_idx:
                  col_found = c
                  break
              if col_found:
                selected_rows.append(df[col_found].values)
              elif 0 <= hw_idx < raw_all.shape[0]:
                selected_rows.append(raw_all[hw_idx, :])
              else:
                selected_rows.append(np.zeros(raw_all.shape[1]))
            datos_archivo = np.vstack(selected_rows)
          else:
            if raw_all.shape[0] < num_canales:
              pad = np.zeros((num_canales - raw_all.shape[0], raw_all.shape[1]))
              datos_archivo = np.vstack((raw_all, pad))
            else:
              datos_archivo = raw_all[:num_canales, :]
          total_samples_archivo = datos_archivo.shape[1]
          dur_s = total_samples_archivo / sample_rate
          print(f"[SIMULADOR] Medicion de prueba cargada ({total_samples_archivo} muestras, {dur_s:.1f} s, {num_canales} canales).")
        else:
          print("[SIMULADOR] El CSV no tiene columnas de 'Canal'. Usando Senoidal.")
      else:
        print(f"[SIMULADOR] Archivo no encontrado en {csv_path}. Usando Senoidal.")
    except Exception as e:
      print(f"[SIMULADOR] Error al cargar el archivo de prueba: {e}. Usando Senoidal.")

  # --- MEJORA: Usar el tiempo real para una simulación más fluida ---
  # En lugar de un tiempo_acumulado propenso a errores por la imprecisión de time.sleep(),
  # usamos el tiempo real del sistema para generar la señal.
  start_time = time.perf_counter()
  samples_generados = 0

  while not stop_event.is_set():
    if datos_archivo is not None:
      idx_start = samples_generados % total_samples_archivo
      idx_end = idx_start + chunk_samples
      if idx_end <= total_samples_archivo:
        datos_leidos = datos_archivo[:, idx_start:idx_end]
      else:
        part1 = datos_archivo[:, idx_start:]
        part2 = datos_archivo[:, :idx_end - total_samples_archivo]
        datos_leidos = np.hstack((part1, part2))
    else:
      tiempo_actual_bloque = samples_generados / sample_rate
      datos_leidos = generar_senales_prueba(tiempo_actual_bloque, chunk_samples, sample_rate, num_canales, test_freq)
      
    data_queue.put(datos_leidos)
    samples_generados += chunk_samples

    # Esperamos el tiempo correcto para el siguiente bloque.
    next_time = start_time + (samples_generados / sample_rate)
    sleep_duration = next_time - time.perf_counter()
    if sleep_duration > 0:
      time.sleep(sleep_duration)
    elif sleep_duration < -0.2:
      # Sincronizar referencia temporal para evitar desbordes por lag del sistema
      start_time = time.perf_counter() - (samples_generados / sample_rate)
  print("Cerrando hilo de simulación.")

# =============================================================================
# FUNCIÓN EXTRA: GENERADOR DE SEÑALES (Sin cambios)
# =============================================================================
def generar_senales_prueba(tiempo_actual, samples_por_canal, sample_rate, num_canales, test_freq=50):
  """
  Ejecuta la funcionalidad de generar_senales_prueba.

  Args:
    tiempo_actual (Any): Argumento posicional tiempo_actual.
    samples_por_canal (Any): Argumento posicional samples_por_canal.
    sample_rate (Any): Argumento posicional sample_rate.
    num_canales (Any): Argumento posicional num_canales.
    test_freq (Any): Argumento posicional test_freq.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  t = np.linspace(
    tiempo_actual, 
    tiempo_actual + samples_por_canal / sample_rate,
    samples_por_canal,
    endpoint=False
  )
  # Canal 0: Usa la frecuencia de prueba de la GUI
  senal1 = 1.0 * np.sin(2 * np.pi * test_freq * t) 
  
  senales = [senal1]
  # Generar señales de prueba distintas para los siguientes canales
  if num_canales > 1:
    # Canal 1: Frecuencia fija diferente
    senal2 = 0.8 * np.sin(2 * np.pi * 120.0 * t)
    senales.append(senal2)
  if num_canales > 2:
    # Canal 2: Señal compuesta
    senal3 = 0.6 * np.sin(2 * np.pi * 50.0 * t) + 0.4 * np.sin(2 * np.pi * 250.0 * t)
    senales.append(senal3)
  # Canales restantes con frecuencias aleatorias
  for i in range(3, num_canales):
    f_rand = 100 + i * 110
    senales.append(0.7 * np.sin(2 * np.pi * f_rand * t))

  # ruido = 0.1 * np.random.randn(num_canales, samples_por_canal)
  ruido=0
  return np.array(senales) + ruido

# =============================================================================
# BLOQUE 3.1: GUARDADO DE DATOS EN .WAV (MODIFICADO para carpetas)
# =============================================================================
def guardar_grabacion_wav(datos_completos, sample_rate, output_dir, num_canales, base_name="grabacion"):
  """
  Ejecuta la funcionalidad de guardar_grabacion_wav.

  Args:
    datos_completos (Any): Argumento posicional datos_completos.
    sample_rate (Any): Argumento posicional sample_rate.
    output_dir (Any): Argumento posicional output_dir.
    num_canales (Any): Argumento posicional num_canales.
    base_name (Any): Argumento posicional base_name.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  if not datos_completos:
    print("No hay datos para guardar.")
    return False
  print("\nConcatenando datos para guardado WAV...")
  try:
    grabacion = np.concatenate(datos_completos, axis=1)
  except ValueError:
    print("Error: No se grabaron datos. El buffer está vacío.")
    return False
  
  exito_total = True
  for i in range(num_canales):
    datos_canal = grabacion[i]
    max_val = np.max(np.abs(datos_canal))
    if max_val == 0:
      print(f"Canal {i} sin señal, no se puede guardar.")
      continue
    
    normalizado = datos_canal / max_val
    datos_int16 = (normalizado * 32767).astype(np.int16)
    
    # --- NUEVO: Crear subcarpeta para el canal ---
    channel_output_dir = os.path.join(output_dir, f"canal_{i}")
    os.makedirs(channel_output_dir, exist_ok=True)
    nombre_archivo_canal = os.path.join(channel_output_dir, f"{base_name}.wav")
    
    try:
      write_wav(nombre_archivo_canal, sample_rate, datos_int16)
      print(f"  Canal {i} guardado como: {nombre_archivo_canal}")
    except Exception as e:
      print(f"  Error al guardar el .wav del canal {i}: {e}")
      exito_total = False
  return exito_total

# =============================================================================
# BLOQUE 3.2: GUARDADO DE DATOS EN .CSV (Sin cambios)
# =============================================================================
def guardar_grabacion_csv(datos_completos, sample_rate, output_dir, num_canales, base_name="grabacion", canales_daq=None):
  """
  Ejecuta la funcionalidad de guardar_grabacion_csv.

  Args:
    datos_completos (Any): Argumento posicional datos_completos.
    sample_rate (Any): Argumento posicional sample_rate.
    output_dir (Any): Argumento posicional output_dir.
    num_canales (Any): Argumento posicional num_canales.
    base_name (Any): Argumento posicional base_name.
    canales_daq (list, optional): Nombres de canales DAQ (ej: ['Dev1/ai0', 'Dev1/ai2']).

  Returns:
    Any: Resultado de la ejecucion de la funcion.
  """
  if not datos_completos:
    return False

  print("Concatenando datos para CSV...")
  try:
    grabacion = np.concatenate(datos_completos, axis=1) # (num_canales, N)
  except ValueError:
    return False
  
  # Crear vector de tiempo
  num_muestras = grabacion.shape[1]
  tiempo = np.arange(num_muestras) / float(sample_rate) # (N,)
  
  # Transponer datos de (num_canales, N) a (N, num_canales)
  datos_t = grabacion.T
  
  # Apilar tiempo y datos: (N, 1) + (N, num_canales) -> (N, num_canales + 1)
  datos_para_csv = np.hstack((tiempo.reshape(-1, 1), datos_t))
  
  # Definir nombre de archivo
  nombre_archivo_csv = os.path.join(output_dir, f"{base_name}.csv")
  
  # Guardar
  try:
    print(f"Guardando CSV en: {nombre_archivo_csv}...")
    col_names = []
    for i in range(num_canales):
      if canales_daq and i < len(canales_daq):
        m_ai = re.search(r"ai(\d+)", str(canales_daq[i]), re.IGNORECASE)
        hw_idx = int(m_ai.group(1)) if m_ai else i
        col_names.append(f"Canal {hw_idx}")
      else:
        col_names.append(f"Canal {i}")
    headers = "Tiempo (s)," + ",".join(col_names)
    np.savetxt(
      nombre_archivo_csv, 
      datos_para_csv, 
      delimiter=",", 
      header=headers, 
      comments="" # Evita el '#' en el header
    )
    print(f"  CSV guardado exitosamente.")
    return True
  except Exception as e:
    print(f"  Error al guardar el CSV: {e}")
    return False

# =============================================================================
# BLOQUE 3.3: GENERADOR DE GRÁFICO (SEÑAL CRUDA NORMALIZADA / ESTÉTICA CALIBRADA)
# =============================================================================
def escribir_pulsos_metronomo_csv(pulsos, output_dir, base_name="pulsos_metronomo"):
  """
  Escribe el registro de pulsos del metronomo (tiempos exactos de disparo) en la
  carpeta de la sesion. Cada fila incluye epoch, perf_counter, tiempo relativo a
  la DAQ y el indice de muestra DAQ calculado con el ancla de grabacion.
  """
  if not pulsos:
    return None
  try:
    import csv as _csv
    os.makedirs(str(output_dir), exist_ok=True)
    path = os.path.join(str(output_dir), f"{base_name}.csv")
    campos = ["seq", "idx", "sub", "principal", "count_in", "t_epoch", "t_perf",
              "t_daq_s", "t_wav_s", "sample_daq", "offset_muestras", "bpm", "subdiv",
              "intervalo_s", "jitter_activo", "jitter_mode", "palabra"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
      writer = _csv.DictWriter(fh, fieldnames=campos, extrasaction="ignore")
      writer.writeheader()
      for p in pulsos:
        writer.writerow({k: p.get(k, "") for k in campos})
    print(f"[Metronomo] Pulsos guardados en: {path} ({len(pulsos)} filas)")
    return path
  except Exception as e:
    print(f"[Metronomo] No se pudo escribir {base_name}.csv: {e}")
    return None


def generar_grafico_grabacion(datos_completos, sample_rate, output_dir, num_canales, canales_daq, base_name="photo"):
  """
  Genera el gráfico de señal cruda normalizada y calibrada al finalizar la grabación,
  replicando la estética, código de colores y layout de plotter_calibrado.
  Guarda la imagen resultante exclusivamente como photo.png en output_dir.
  """
  if datos_completos is None or len(datos_completos) == 0:
    return False

  print("Generando gráfico de señal cruda calibrada (estética plotter calibrado)...")
  try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  except ImportError:
    print("Error: matplotlib no está instalado.")
    return False

  try:
    if isinstance(datos_completos, list):
      grabacion = np.concatenate(datos_completos, axis=1)
    else:
      grabacion = np.array(datos_completos)
  except ValueError:
    return False

  num_muestras = grabacion.shape[1]
  if num_muestras == 0:
    return False

  sample_rate = float(sample_rate) if sample_rate and sample_rate > 0 else 2000.0
  tiempo = np.arange(num_muestras) / sample_rate
  t_min = float(tiempo[0])
  t_max = float(tiempo[-1])

  # 1. Cargar metadatos desde metadata.json si existe
  bpm = None
  noise_seconds = 5.0
  muscles_map = {}
  muscles_list = []
  
  meta_path_ch0 = os.path.join(output_dir, "canal_0", "metadata.json")
  meta_path_root = os.path.join(output_dir, "metadata.json")
  meta_file = meta_path_ch0 if os.path.exists(meta_path_ch0) else (meta_path_root if os.path.exists(meta_path_root) else None)
  
  beat_timestamps = []
  if meta_file:
    try:
      with open(meta_file, 'r', encoding='utf-8') as f:
        md = json.load(f)
        bpm = md.get("bpm")
        noise_seconds = md.get("noise_seconds", 5.0)
        muscles_map = md.get("muscles_map", {})
        muscles_list = md.get("muscles", [])
        beat_timestamps = md.get("beat_timestamps_s", []) or []
    except Exception as e_meta:
      print(f"  [Aviso] Error leyendo metadatos: {e_meta}")

  # Si no estaban en metadata.json pero existe pulsos_metronomo.csv, extraer t_wav_s de pulsos principales post-count-in
  if not beat_timestamps:
    pulsos_csv_file = os.path.join(output_dir, "pulsos_metronomo.csv")
    if os.path.exists(pulsos_csv_file):
      try:
        import csv as _csv
        with open(pulsos_csv_file, 'r', encoding='utf-8') as f_p:
          reader = _csv.DictReader(f_p)
          for row in reader:
            if str(row.get("principal", "0")) == "1" and int(row.get("idx", "0") or 0) >= 1 and str(row.get("t_wav_s", "")).strip() not in ("", "None"):
              beat_timestamps.append(float(row["t_wav_s"]))
      except Exception:
        pass

  # 2. Cargar configuraciones de canales y colores
  try:
    from utils.config_manager import ConfigManager, get_unique_channel_colors
    config_mgr = ConfigManager()
    canales_config = config_mgr.get("canales") or {}
  except Exception:
    get_unique_channel_colors = None
    canales_config = {}

  ch_info_list = []
  for i in range(num_canales):
    daq_name = canales_daq[i] if (canales_daq and i < len(canales_daq)) else f"Canal {i}"
    m_ai = re.search(r"ai(\d+)", str(daq_name), re.IGNORECASE)
    hw_idx = int(m_ai.group(1)) if m_ai else i
    nom_limpio = f"Canal {hw_idx}"
    ch_conf = canales_config.get(nom_limpio, canales_config.get(f"Canal {i}", {}))
    musculo = ch_conf.get("musculo", nom_limpio)
    ganancia = ch_conf.get("factor_calibracion", 495.0)
    color_hex = ch_conf.get("color_hex")

    if muscles_map and f"canal_{i}" in muscles_map:
      musculo = muscles_map[f"canal_{i}"]
    elif i < len(muscles_list):
      musculo = muscles_list[i]
      
    ch_meta_path = os.path.join(output_dir, f"canal_{i}", "metadata.json")
    if os.path.exists(ch_meta_path):
      try:
        with open(ch_meta_path, 'r', encoding='utf-8') as f_ch:
          md_ch = json.load(f_ch)
          if 'musculo' in md_ch and md_ch['musculo']:
            musculo = md_ch['musculo']
          if 'resistencia_ohm' in md_ch:
            res_ohm = float(md_ch['resistencia_ohm'])
            ganancia = 1.0 + (49400.0 / res_ohm)
      except Exception:
        pass

    m_low = str(musculo or "").strip().lower()
    is_generic_ch3 = (hw_idx == 3 and m_low in ("", "canal 3", "canal_3", "ch3", "ai3", "dev1/ai3", "dev2/ai3"))
    is_mic = ("mic" in m_low or "audio" in m_low or "mic" in str(daq_name).lower() or is_generic_ch3)
    
    ch_info_list.append({
      "idx": hw_idx,
      "col_name": nom_limpio,
      "daq_name": daq_name,
      "musculo": musculo,
      "ganancia": ganancia,
      "color_hex": color_hex,
      "is_mic": is_mic
    })

  if get_unique_channel_colors:
    colores_canales = get_unique_channel_colors(ch_info_list)
  else:
    colores_canales = ["#ff4500", "#00a896", "#ffff00", "#ff0000"]

  # 3. Procesar señales 100% crudas calibradas en microvoltios
  processed_channels = []
  for i, ch_meta in enumerate(ch_info_list):
    raw = grabacion[i]
    ganancia = ch_meta["ganancia"]
    color_hex = colores_canales[i]
    is_mic = ch_meta["is_mic"]

    # Conversión física a microvoltios (µV)
    sig = (raw / ganancia) * 1e6

    # Restar mediana basal para centrar la oscilación cruda perfectamente en 0
    if not is_mic:
      if noise_seconds is not None and noise_seconds > 0 and noise_seconds >= t_min:
        n_idx = np.searchsorted(tiempo, min(noise_seconds, t_max), side='right')
        base_med = np.median(sig[:n_idx]) if n_idx > 0 else np.median(sig)
      else:
        base_med = np.median(sig)
      sig = sig - base_med

    # Evaluar amplitud máxima post-ruido para escala compartida
    if noise_seconds is not None and noise_seconds > 0:
      mask_post = (tiempo >= noise_seconds)
      y_eval = sig[mask_post] if np.any(mask_post) else sig
    else:
      y_eval = sig

    if not is_mic and len(y_eval) > 0:
      # Suprimir picos eléctricos (> 500 uV) para que artefactos puntuales no aplasten la escala fisiológica (0-500 uV)
      sig = np.clip(sig, -500.0, 500.0)
      y_eval = np.clip(y_eval, -500.0, 500.0)
      p99_abs = float(np.nanpercentile(np.abs(y_eval), 99.2))
      max_abs = min(500.0, max(25.0, p99_abs * 1.12))
    else:
      max_abs = float(np.nanmax(np.abs(y_eval))) if len(y_eval) > 0 else 1.0
    min_val = float(np.nanmin(y_eval)) if len(y_eval) > 0 else 0.0
    max_val = float(np.nanmax(y_eval)) if len(y_eval) > 0 else 1.0

    processed_channels.append({
      "idx": i,
      "hw_idx": ch_meta["idx"],
      "musculo": ch_meta["musculo"],
      "daq_name": ch_meta["daq_name"],
      "color_hex": color_hex,
      "sig": sig,
      "max_abs": max_abs,
      "min_val": min_val,
      "max_val": max_val,
      "is_mic": is_mic
    })

  # 4. Normalización Tricanal: Escala vertical simétrica compartida entre los canales musculares
  muscle_channels = [ch for ch in processed_channels if not ch["is_mic"]]
  shared_muscle_ylim = None
  if muscle_channels:
    m_supremo = max(ch["max_abs"] for ch in muscle_channels)
    if m_supremo > 0:
      margin = m_supremo * 0.08
      shared_muscle_ylim = (-m_supremo - margin, m_supremo + margin)
    else:
      shared_muscle_ylim = (-50.0, 50.0)

  # 5. Configurar figura Matplotlib thread-safe con tema oscuro puro
  ancho_fig = 20
  alto_fig = max(8.0, 3.2 * num_canales)
  fig = Figure(figsize=(ancho_fig, alto_fig), facecolor='#000000')
  canvas = FigureCanvas(fig)
  axs = fig.subplots(num_canales, 1, sharex=True)
  if num_canales == 1:
    axs = [axs]

  # 6. Graficar canales
  for i, ch in enumerate(processed_channels):
    ax = axs[i]
    ax.set_facecolor('#000000')
    color_hex = ch["color_hex"]
    sig = ch["sig"]
    is_mic = ch["is_mic"]
    musculo = ch["musculo"]

    ax.plot(tiempo, sig, color=color_hex, lw=0.8)

    # Asignar escala vertical (normalizada compartida para músculos, autoescala para micrófono)
    if not is_mic and shared_muscle_ylim is not None:
      ax.set_ylim(shared_muscle_ylim)
    else:
      min_v, max_v = ch["min_val"], ch["max_val"]
      m_diff = max_v - min_v
      m_margin = m_diff * 0.08 if m_diff > 0 else 1.0
      ax.set_ylim(min_v - m_margin, max_v + m_margin)

    # Franja y línea indicadora de Ruido Basal
    if noise_seconds is not None and noise_seconds > 0 and noise_seconds >= t_min:
      span_color = '#00e5ff'
      ax.axvspan(max(0.0, t_min), min(noise_seconds, t_max), color=span_color, alpha=0.12)
      ax.axvline(x=noise_seconds, color=span_color, ls='--', lw=1.5, alpha=0.75)
      y_bounds = ax.get_ylim()
      y_text = y_bounds[1] - 0.08 * (y_bounds[1] - y_bounds[0])
      ax.text(noise_seconds / 2.0, y_text, "Ruido Basal", color=span_color,
              fontsize=13, ha='center', va='top', fontweight='bold', alpha=0.9)

    # Guías del metrónomo (usa timestamps reales si existen; si no, grilla teórica)
    win_color = '#ffffff'
    beat_color = '#ffaa00'
    t_limite_ruido = float(noise_seconds) if (noise_seconds is not None and noise_seconds > 0) else 0.0
    valid_beats = sorted([float(b) for b in beat_timestamps if max(t_min, t_limite_ruido - 0.2) <= float(b) <= t_max])
    beat_windows = []

    if valid_beats:
      tau_ref = (60.0 / bpm) if (bpm and bpm > 0) else 1.5
      bounds = [max(t_min, valid_beats[0] - 0.5 * tau_ref)]
      for idx_b in range(len(valid_beats) - 1):
        bounds.append(0.5 * (valid_beats[idx_b] + valid_beats[idx_b + 1]))
      bounds.append(min(t_max, valid_beats[-1] + 0.5 * tau_ref))
      for t_b in valid_beats:
        ax.axvline(x=t_b, color=beat_color, ls=':', lw=0.9, alpha=0.35)
      for t_bnd in bounds:
        if t_min <= t_bnd <= t_max:
          ax.axvline(x=t_bnd, color=win_color, ls='--', lw=1.0, alpha=0.4)
      for idx_b in range(len(valid_beats)):
        beat_windows.append((bounds[idx_b], bounds[idx_b + 1]))
    elif bpm and noise_seconds is not None and noise_seconds > 0:
      tau = 60.0 / bpm
      first_bound = noise_seconds - tau / 2.0
      if t_min <= first_bound <= t_max:
        ax.axvline(x=first_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
      k = 0
      while True:
        t_beat = noise_seconds + k * tau
        t_bound = t_beat + tau / 2.0
        if t_min <= t_beat <= t_max:
          ax.axvline(x=t_beat, color=beat_color, ls=':', lw=0.9, alpha=0.35)
        if k >= 1 and (t_beat - tau / 2.0) <= t_max:
          beat_windows.append((t_beat - tau / 2.0, t_bound))
        if t_bound > t_max:
          break
        if t_bound >= t_min:
          ax.axvline(x=t_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
        k += 1

    # Detección y marcado de picos de contracción en señal cruda
    if not is_mic:
      picos_t, picos_y = [], []
      if beat_windows:
        for t_w_start, t_w_end in beat_windows:
          t_start_val = max(t_min, max(t_limite_ruido, t_w_start))
          t_end_val = min(t_max, t_w_end)
          if t_start_val < t_end_val:
            mask_win = (tiempo >= t_start_val) & (tiempo < t_end_val)
            if np.any(mask_win):
              sub_t = tiempo[mask_win]
              sub_y = sig[mask_win]
              if len(sub_y) > 0:
                idx_max = np.argmax(sub_y)
                p_val = sub_y[idx_max]
                p_t = sub_t[idx_max]
                if p_val > 0 and p_t >= t_limite_ruido:
                  picos_t.append(p_t)
                  picos_y.append(p_val)
      else:
        try:
          from scipy import signal
          mask_post = (tiempo >= t_limite_ruido)
          if np.any(mask_post):
            y_sub = sig[mask_post]
            t_sub = tiempo[mask_post]
            min_dist = max(1, int(sample_rate * 0.3))
            h_thresh = max(0.0, float(np.mean(y_sub)))
            p_idx, _ = signal.find_peaks(y_sub, distance=min_dist, height=h_thresh)
            if len(p_idx) > 0:
              picos_t = t_sub[p_idx].tolist()
              picos_y = y_sub[p_idx].tolist()
        except Exception:
          pass

      if len(picos_y) > 0:
        media_picos = float(np.mean(picos_y))
        std_picos = float(np.std(picos_y))
        if std_picos > 0:
          ax.axhspan(max(0.0, media_picos - std_picos), media_picos + std_picos,
                     color=color_hex, alpha=0.18, zorder=3)
        ax.axhline(y=media_picos, color=color_hex, ls=':', lw=1.5, alpha=0.85, zorder=4)
        ax.scatter(picos_t, picos_y, color=color_hex, s=26, alpha=0.85, zorder=5,
                   edgecolors='white', linewidths=0.6)
        ax.text(t_max, media_picos, f"  μ = {media_picos:.1f} ± {std_picos:.1f} µV",
                color=color_hex, fontsize=12, va='center', ha='left',
                fontweight='bold', alpha=0.95, zorder=6)

    # Títulos y ejes
    hw_ch = ch.get("hw_idx", i)
    tit = f"ai{hw_ch} - {musculo} - Cruda" if not is_mic else f"ai{hw_ch} - {musculo}"
    ax.set_title(tit, fontsize=21, color='white', pad=6)
    ax.set_ylabel("Amplitud (µV)" if not is_mic else "Micrófono", fontsize=21, color='white')
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=16, colors='white')
    for spine in ax.spines.values():
      spine.set_color('#333333')

  for ax in axs[:-1]:
    ax.tick_params(labelbottom=False)
  axs[-1].set_xlabel("Tiempo (s)", fontsize=21, color='white')

  nombre_archivo_grafico = os.path.join(output_dir, f"{base_name}.png")

  try:
    fig.tight_layout(rect=[0, 0.02, 1, 0.96], h_pad=1.2)
    fig.savefig(nombre_archivo_grafico, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print(f"  Gráfico de señal cruda guardado como: {nombre_archivo_grafico}")
    return True
  except Exception as e:
    print(f"  Error al guardar el gráfico: {e}")
    return False
  finally:
    fig.clf()


# =============================================================================
# BLOQUE 3.3B: GENERADOR DE GRÁFICOS ESTADÍSTICOS
# =============================================================================
def generar_grafico_estadisticas(stats_time, stats_snr, stats_noise_mean, stats_noise_std, output_dir, num_canales, canales_daq, base_name="estadisticas_evolucion"):
  """
  Genera el gráfico de evolución temporal de estadísticas (SNR y Ruido inter-pulso) de manera thread-safe.
  """
  hay_datos = any(len(t) > 0 for t in stats_time) if stats_time else False
  if not hay_datos:
    return False

  try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  except ImportError:
    return False

  fig = Figure(figsize=(15, max(4, 4 * num_canales)))
  canvas = FigureCanvas(fig)
  axs = fig.subplots(num_canales, 2, squeeze=False)
  fig.suptitle("Evolución Temporal: SNR y Ruido Inter-pulso", fontsize=16)

  for i in range(num_canales):
    if i >= len(stats_time):
      continue
    t = stats_time[i]
    if not t:
      continue

    t_arr = np.array(t)
    mask = t_arr >= 20.0
    if not np.any(mask):
      mask = np.ones_like(t_arr, dtype=bool)

    t_plot = t_arr[mask]
    snr_plot = np.array(stats_snr[i])[mask]
    noise_mean_plot = np.array(stats_noise_mean[i])[mask]
    noise_std_plot = np.array(stats_noise_std[i])[mask]

    daq_ch_name = canales_daq[i] if (canales_daq and i < len(canales_daq)) else f"Canal {i}"
    m_ai = re.search(r"ai(\d+)", str(daq_ch_name), re.IGNORECASE)
    hw_idx = int(m_ai.group(1)) if m_ai else i

    # Subplot Izquierdo: SNR
    axs[i, 0].plot(t_plot, snr_plot, marker='o', linestyle='-', color='#00ffcc', linewidth=1.5, markersize=3)
    axs[i, 0].set_title(f"ai{hw_idx} - {daq_ch_name}: Evolución SNR Promedio", fontsize=12, fontweight="bold")
    axs[i, 0].set_ylabel("SNR Promedio Acumulado")
    axs[i, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot Derecho: Ruido
    axs[i, 1].plot(t_plot, noise_mean_plot, marker='o', linestyle='-', color='#ff754b', label='Promedio', linewidth=1.5, markersize=3)
    axs[i, 1].plot(t_plot, noise_std_plot, marker='x', linestyle='--', color='#ffaa00', label='Desviación', linewidth=1.5, markersize=3)
    axs[i, 1].set_title(f"ai{hw_idx} - {daq_ch_name}: Evolución Ruido Inter-pulso", fontsize=12, fontweight="bold")
    axs[i, 1].set_ylabel("Amplitud (µV)")
    axs[i, 1].legend()
    axs[i, 1].grid(True, linestyle="--", alpha=0.5)

  for ax in axs[-1, :]:
    ax.set_xlabel("Tiempo de Señal (s)")

  nombre_archivo = os.path.join(output_dir, f"{base_name}.png")
  try:
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(nombre_archivo, dpi=120)
    print(f"  Gráfico de estadísticas guardado como: {nombre_archivo}")
    return True
  except Exception as e:
    print(f"  Error al guardar el gráfico de estadísticas: {e}")
    return False
  finally:
    fig.clf()


# =============================================================================
# BLOQUE 3.4: DIÁLOGO DE VISTA PREVIA POST-GRABACIÓN
# =============================================================================
class PreviewPlotDialog(QtWidgets.QDialog):
  """
  Diálogo para mostrar la vista previa de la grabación recién finalizada (photo.png)
  durante unos segundos (con cierre automático o manual).
  """
  def __init__(self, image_path, parent=None, duration_ms=5000):
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Vista Previa de Grabación")
    self.setStyleSheet("""
      QDialog {
        background-color: #050505;
        color: #ffffff;
        border: 2px solid #00ffcc;
      }
      QLabel {
        color: #00ffcc;
        font-family: 'Consolas', 'Courier New', monospace;
      }
      QPushButton {
        background-color: #111111;
        color: #00ffcc;
        border: 1px solid #00ffcc;
        padding: 6px 16px;
        border-radius: 4px;
        font-weight: bold;
      }
      QPushButton:hover {
        background-color: #00ffcc;
        color: #000000;
      }
    """)
    self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)
    self.resize(1100, 720)
    
    layout = QtWidgets.QVBoxLayout(self)
    layout.setContentsMargins(12, 10, 12, 10)
    layout.setSpacing(8)
    
    top_layout = QtWidgets.QHBoxLayout()
    self.lbl_info = QtWidgets.QLabel("VISTA PREVIA DE SEÑAL CRUDA (NORMALIZADA)")
    self.lbl_info.setStyleSheet("font-size: 15px; font-weight: bold; color: #00ffcc;")
    
    self.segundos_restantes = max(1, int(duration_ms / 1000))
    self.lbl_timer = QtWidgets.QLabel(f"Cerrando en {self.segundos_restantes}s...")
    self.lbl_timer.setStyleSheet("font-size: 13px; color: #ffaa00;")
    
    btn_cerrar = QtWidgets.QPushButton("Cerrar (Esc)")
    btn_cerrar.clicked.connect(self.accept)
    
    top_layout.addWidget(self.lbl_info)
    top_layout.addStretch()
    top_layout.addWidget(self.lbl_timer)
    top_layout.addSpacing(10)
    top_layout.addWidget(btn_cerrar)
    layout.addLayout(top_layout)
    
    self.lbl_imagen = QtWidgets.QLabel()
    self.lbl_imagen.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_imagen.setStyleSheet("background-color: #000000; border: 1px solid #222222;")
    layout.addWidget(self.lbl_imagen, stretch=1)
    
    if os.path.exists(image_path):
      pix = QtGui.QPixmap(image_path)
      if not pix.isNull():
        self.lbl_imagen.setPixmap(pix.scaled(1080, 640, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
      else:
        self.lbl_imagen.setText("No se pudo cargar la imagen del gráfico.")
    else:
      self.lbl_imagen.setText("Archivo de gráfico no encontrado.")
        
    self.timer = QtCore.QTimer(self)
    self.timer.setInterval(1000)
    self.timer.timeout.connect(self._on_tick)
    self.timer.start()

  def _on_tick(self):
    self.segundos_restantes -= 1
    if self.segundos_restantes <= 0:
      self.timer.stop()
      self.accept()
    else:
      self.lbl_timer.setText(f"Cerrando en {self.segundos_restantes}s...")

  def keyPressEvent(self, event):
    if event.key() in (QtCore.Qt.Key_Escape, QtCore.Qt.Key_Return, QtCore.Qt.Key_Space):
      self.timer.stop()
      self.accept()
    else:
      super().keyPressEvent(event)


# =============================================================================
# BLOQUE 4: INTERFAZ GRÁFICA (GUI) (MODIFICADO v3.12)
# =============================================================================

class SaveMeasurementDialog(QtWidgets.QDialog):
  """
  Un diálogo personalizado para guardar mediciones con dos formatos:
  1. Medición de Prueba (nombre aleatorio).
  2. Medición Formal (formato estructurado: Letra_Prueba_Sujeto).
  """
  def __init__(self, parent=None):
    """
    Ejecuta la funcionalidad de __init__.

    Args:
      parent (Any): Argumento posicional parent.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Guardar Medición")
    self.setMinimumWidth(350)
    self.layout = QtWidgets.QVBoxLayout(self)
    self.measurement_name = ""
    # --- NUEVO: Guardar los componentes del nombre por separado ---
    self.sujeto = ""
    self.letra = ""
    self.prueba = ""
    self.es_formal = False
    # --- NUEVO: Campo para el comentario ---
    self.comentario = ""

    # Grupo de Radio Buttons para seleccionar el tipo
    self.radio_group = QtWidgets.QGroupBox("Tipo de Medición")
    self.radio_layout = QtWidgets.QVBoxLayout()
    self.radio_random = QtWidgets.QRadioButton("Medición de Prueba (ej: prueba_...)")
    self.radio_formal = QtWidgets.QRadioButton("Medición Formal (ej: A_Prueba1_Sujeto1)")
    self.radio_random.setChecked(True)
    self.radio_layout.addWidget(self.radio_random)
    self.radio_layout.addWidget(self.radio_formal)
    self.radio_group.setLayout(self.radio_layout)
    self.layout.addWidget(self.radio_group)

    # Grupo para los campos del formato formal (Letra - Prueba - Sujeto)
    self.formal_group = QtWidgets.QGroupBox("Formato Formal")
    self.formal_layout = QtWidgets.QFormLayout()
    self.edit_letra = QtWidgets.QLineEdit("A")
    self.edit_prueba = QtWidgets.QLineEdit("Prueba1")
    self.edit_sujeto = QtWidgets.QLineEdit("Sujeto1")
    self.formal_layout.addRow("Letra:", self.edit_letra)
    self.formal_layout.addRow("Prueba:", self.edit_prueba)
    self.formal_layout.addRow("Sujeto:", self.edit_sujeto)
    # --- NUEVO: Añadir campo de comentario al layout formal ---
    self.edit_comentario = QtWidgets.QLineEdit()
    self.formal_layout.addRow("Comentario:", self.edit_comentario)
    self.formal_group.setLayout(self.formal_layout)
    self.layout.addWidget(self.formal_group)

    # Botones de OK y Cancelar
    self.button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    self.layout.addWidget(self.button_box)

    # Conexiones de señales
    self.button_box.accepted.connect(self.on_accept)
    self.button_box.rejected.connect(self.reject)
    self.radio_formal.toggled.connect(self.update_ui_state)

    self.update_ui_state() # Estado inicial

  def update_ui_state(self):
    """Habilita o deshabilita los campos del formato formal según la selección."""
    is_formal = self.radio_formal.isChecked()
    self.formal_group.setEnabled(is_formal)

  def on_accept(self):
    """Construye el nombre del archivo al presionar OK."""
    if self.radio_formal.isChecked():
      self.es_formal = True
      self.letra = self.edit_letra.text()
      self.prueba = self.edit_prueba.text()
      self.sujeto = self.edit_sujeto.text()
      self.comentario = self.edit_comentario.text() # Guardar el comentario
      self.measurement_name = f"{self.letra}_{self.prueba}_{self.sujeto}"
    else:
      self.es_formal = False
      self.measurement_name = f"prueba_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      self.sujeto = "prueba"
      self.letra = ""
      self.prueba = ""
      self.comentario = "" # Sin comentario para mediciones de prueba
    self.accept()



class VentanaEstimulosExterna(QtWidgets.QWidget):
  """
  Ventana dedicada de estímulos para un 2do Monitor HDMI/VGA (Modo Sujeto Ciego sin red).
  Replica la estética de cliente_estimulos.html (#00FFFF sobre #050505) para que la persona
  vea únicamente la vocal actual, la próxima vocal, el contador y el destello del pulso,
  sin ver las señales EMG ni el PCA en la pantalla del operador.
  """
  def __init__(self, parent=None):
    super().__init__(None)
    self.setWindowTitle("Ñandú LSD - Pantalla Externa de Estímulos (Sujeto)")
    self.resize(960, 640)
    self.setStyleSheet("background-color: #050505; color: #00FFFF; font-family: 'Helvetica Neue', Arial, sans-serif;")
    lyt = QtWidgets.QVBoxLayout(self)
    lyt.setContentsMargins(24, 24, 24, 18)
    lyt.setSpacing(10)

    top_row = QtWidgets.QHBoxLayout()
    self.lbl_countin = QtWidgets.QLabel("")
    self.lbl_countin.setStyleSheet("color: #FF3355; font-size: 38px; font-weight: 900;")
    self.btn_fs = QtWidgets.QPushButton("PANTALLA COMPLETA (F11)")
    self.btn_fs.setStyleSheet(
      "QPushButton { background-color: #0d2626; color: #00FFFF; border: 1px solid #00FFFF; border-radius: 6px; padding: 6px 14px; font-weight: bold; }"
    )
    self.btn_fs.clicked.connect(self.toggle_fullscreen)
    top_row.addWidget(self.lbl_countin)
    top_row.addStretch(1)
    top_row.addWidget(self.btn_fs)
    lyt.addLayout(top_row)

    lyt.addStretch(1)
    self.lbl_palabra = QtWidgets.QLabel("ESPERANDO...")
    self.lbl_palabra.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_palabra.setStyleSheet("color: #00FFFF; font-size: 150px; font-weight: 900;")
    lyt.addWidget(self.lbl_palabra)

    self.lbl_proxima = QtWidgets.QLabel("")
    self.lbl_proxima.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_proxima.setStyleSheet("color: #FFD166; font-size: 28px; font-weight: bold;")
    lyt.addWidget(self.lbl_proxima)

    self.lbl_contador = QtWidgets.QLabel("0")
    self.lbl_contador.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_contador.setStyleSheet("color: #00b3b3; font-size: 64px; font-weight: bold;")
    lyt.addWidget(self.lbl_contador)

    pulse_row = QtWidgets.QHBoxLayout()
    pulse_row.addStretch(1)
    self.lbl_ring_ext = QtWidgets.QLabel()
    self.lbl_ring_ext.setAlignment(QtCore.Qt.AlignCenter)
    self.frame_pulso = QtWidgets.QFrame()
    self.frame_pulso.setFixedSize(88, 88)
    self.frame_pulso.setStyleSheet("background-color: #0d1414; border: 2px solid #004444; border-radius: 10px;")
    pulse_row.addWidget(self.lbl_ring_ext)
    pulse_row.addWidget(self.frame_pulso)
    pulse_row.addStretch(1)
    lyt.addLayout(pulse_row)
    lyt.addStretch(1)

    self.lbl_barra = QtWidgets.QLabel("MODO PANTALLA EXTERNA ACTIVO")
    self.lbl_barra.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_barra.setStyleSheet("color: #4f6f6f; font-size: 14px;")
    lyt.addWidget(self.lbl_barra)

  def toggle_fullscreen(self):
    if self.isFullScreen():
      self.showNormal()
    else:
      self.showFullScreen()

  def keyPressEvent(self, event):
    if event.key() == QtCore.Qt.Key_F11 or event.key() == QtCore.Qt.Key_Escape:
      self.toggle_fullscreen()
      return
    super().keyPressEvent(event)

  def actualizar_estimulo(self, palabra="", palabra_siguiente="", pulso_actual=0, total_pulsos=0,
                          flash=False, count_in=None, sesion="", bpm=30, instruccion_fase="",
                          subtitulo_fase="", tira_html="", ciclo_actual=1, total_ciclos=1,
                          sub_actual=1, subdiv=4, pre_cambio=False):
    p_txt = str(palabra or "ESPERANDO...").upper()
    sig_txt = str(palabra_siguiente or "").upper()
    vocal_colores = {'A': '#FF3355', 'E': '#00B4D8', 'I': '#39FF14', 'O': '#C77DFF', 'U': '#FFD166'}
    v_col = vocal_colores.get(p_txt, '#00FFFF')
    if count_in is not None and int(count_in) > 0:
      self.lbl_countin.setText(f"CUENTA REGRESIVA: {int(count_in)} — {instruccion_fase or 'PREPÁRATE'}")
      self.lbl_countin.setStyleSheet("color: #FF8800; font-size: 28px; font-weight: 900;")
    elif pre_cambio and sig_txt:
      self.lbl_countin.setText(f"ATENCIÓN: PRÓXIMO PULSO CAMBIA A /{sig_txt}/")
      self.lbl_countin.setStyleSheet("color: #FFFF00; font-size: 26px; font-weight: 900;")
    else:
      self.lbl_countin.setText(str(instruccion_fase or "SIGUE EL RITMO DEL METRÓNOMO"))
      self.lbl_countin.setStyleSheet("color: #00FFCC; font-size: 24px; font-weight: 900;")

    self.lbl_palabra.setText(f"/{p_txt}/" if len(p_txt) <= 3 else p_txt)
    if flash:
      self.lbl_palabra.setStyleSheet("color: #FFFFFF; background-color: rgba(0, 255, 255, 0.22); border: 3px solid #FFFFFF; border-radius: 12px; font-size: 150px; font-weight: 900;")
      self.frame_pulso.setStyleSheet("background-color: #96FFFF; border: 3px solid #FFFFFF; border-radius: 10px;")
    else:
      self.lbl_palabra.setStyleSheet(f"color: {v_col}; background-color: transparent; border: none; font-size: 140px; font-weight: 900;")
      self.frame_pulso.setStyleSheet("background-color: #0d1414; border: 2px solid #004444; border-radius: 10px;")
    prox_line = f"PRÓXIMA VOCAL: /{sig_txt}/" if sig_txt else ""
    if subtitulo_fase:
      prox_line = f"{prox_line}  ·  {subtitulo_fase}" if prox_line else subtitulo_fase
    if tira_html:
      self.lbl_proxima.setText(f"<div align='center'>{prox_line}<br>{tira_html}</div>")
    else:
      self.lbl_proxima.setText(prox_line)
    self.lbl_contador.setText(
      f"Pulso {pulso_actual}/{total_pulsos}  (Ciclo {ciclo_actual}/{total_ciclos})"
      if total_pulsos > 0 else f"Pulso {pulso_actual}"
    )
    self.lbl_barra.setText(f"SESIÓN: {sesion or 'EN CURSO'} · {int(bpm)} BPM · Compás {sub_actual}/{subdiv}")

    # Renderizar anillo circular radial en la pantalla externa
    r_col = "#FFFFFF" if flash else "#00FFFF"
    circ_ext = 226.2
    prog_ext = 1.0 if flash else 0.5
    off_ext = circ_ext * (1.0 - prog_ext)
    self.lbl_ring_ext.setText(
      f"<svg width='88' height='88' viewBox='0 0 80 80'>"
      f"<circle stroke='#122323' stroke-width='7' fill='transparent' r='36' cx='40' cy='40'/>"
      f"<circle stroke='{r_col}' stroke-width='7' stroke-dasharray='{circ_ext}' stroke-dashoffset='{off_ext:.1f}' stroke-linecap='round' fill='transparent' r='36' cx='40' cy='40' transform='rotate(-90 40 40)'/>"
      f"</svg>"
    )


class DialogoPantallaExterna(QtWidgets.QDialog):
  """
  Diálogo de control para transmitir la ventana de palabras a una pantalla externa
  (Tailscale 100.118.12.94 <-> Celu 100.102.47.75, Cable USB adb reverse o 2do Monitor HDMI).
  """
  def __init__(self, parent_plotter):
    super().__init__(parent_plotter)
    self.plotter = parent_plotter
    self.setWindowTitle("Ñandú LSD - Pantalla Externa (Tailscale / USB / 2do Monitor) y Sujeto Ciego")
    self.setMinimumWidth(640)
    self.setStyleSheet("""
      QDialog { background-color: #0b0e14; color: #00ffff; font-family: 'Courier New', monospace; }
      QLabel { color: #e0f7fa; }
      QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #00ffcc; }
      QPushButton { background-color: #12263a; color: #00ffcc; border: 1.5px solid #00ffcc; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
      QPushButton:hover { background-color: #00ffcc; color: #050508; }
      QLineEdit, QSpinBox, QComboBox { background-color: #161b22; color: #00ffcc; border: 1px solid #00ffcc; padding: 4px; }
    """)
    lyt = QtWidgets.QVBoxLayout(self)

    rem_cfg = self.plotter.config_mgr.get_remote_display_config() if hasattr(self.plotter.config_mgr, 'get_remote_display_config') else {}
    puerto_init = int(rem_cfg.get("puerto", 8765))
    token_init = str(rem_cfg.get("token", ""))
    ts_pc_init = str(rem_cfg.get("tailscale_ip_pc", _ip_tailscale("100.118.12.94")))
    ts_celu_init = str(rem_cfg.get("tailscale_ip_celu", "100.102.47.75"))

    # Auto-iniciar el servidor al abrir el diálogo si todavía no estaba corriendo
    srv = getattr(self.plotter, 'remote_display_server', None)
    if srv is None or not srv.activo():
      self.plotter.iniciar_servidor_display_remoto(puerto=puerto_init, token=token_init)

    grp_srv = QtWidgets.QGroupBox("1. Red Tailscale / Servidor Web de Estímulos (Celular / Tablet)")
    fl = QtWidgets.QFormLayout(grp_srv)
    self.spin_puerto = QtWidgets.QSpinBox()
    self.spin_puerto.setRange(1024, 65535)
    self.spin_puerto.setValue(puerto_init)

    ts_row = QtWidgets.QHBoxLayout()
    self.edit_ts_pc = QtWidgets.QLineEdit(ts_pc_init)
    self.edit_ts_pc.setToolTip("IP Tailscale de esta PC (ej. 100.118.12.94)")
    self.edit_ts_celu = QtWidgets.QLineEdit(ts_celu_init)
    self.edit_ts_celu.setToolTip("IP Tailscale de tu celular (ej. 100.102.47.75)")
    ts_row.addWidget(QtWidgets.QLabel("PC:"))
    ts_row.addWidget(self.edit_ts_pc)
    ts_row.addWidget(QtWidgets.QLabel("Celu:"))
    ts_row.addWidget(self.edit_ts_celu)
    fl.addRow("IPs Tailscale:", ts_row)

    self.edit_token = QtWidgets.QLineEdit(token_init)
    self.edit_token.setPlaceholderText("Opcional (vacío = acceso directo sin token)")
    fl.addRow("Puerto TCP:", self.spin_puerto)
    fl.addRow("Token opcional:", self.edit_token)

    self.lbl_estado_srv = QtWidgets.QLabel()
    self.lbl_estado_srv.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    fl.addRow("Estado y URLs:", self.lbl_estado_srv)

    btn_row = QtWidgets.QHBoxLayout()
    self.btn_toggle_srv = QtWidgets.QPushButton("Detener Servidor Remoto")
    self.btn_toggle_srv.clicked.connect(self._on_toggle_srv)
    self.btn_tailscale = QtWidgets.QPushButton("Abrir / Verificar Tailscale (100.118.12.94 <-> Celu)")
    self.btn_tailscale.setStyleSheet("QPushButton { background-color: #0d2b1d; color: #39ff14; border: 1.5px solid #39ff14; padding: 6px 12px; font-weight: bold; } QPushButton:hover { background-color: #39ff14; color: #050508; }")
    self.btn_tailscale.clicked.connect(self._on_tailscale_check)
    self.btn_usb_adb = QtWidgets.QPushButton("Túnel USB (adb reverse)")
    self.btn_usb_adb.clicked.connect(self._on_adb_reverse)
    btn_row.addWidget(self.btn_toggle_srv)
    btn_row.addWidget(self.btn_tailscale)
    btn_row.addWidget(self.btn_usb_adb)
    fl.addRow(btn_row)
    lyt.addWidget(grp_srv)

    grp_mon = QtWidgets.QGroupBox("2. Ventana Nativa para 2do Monitor HDMI/VGA y Modo Sujeto Ciego")
    ml = QtWidgets.QVBoxLayout(grp_mon)
    self.btn_abrir_ventana_ext = QtWidgets.QPushButton("Abrir / Mostrar Ventana Externa de Estímulos (2do Monitor)")
    self.btn_abrir_ventana_ext.clicked.connect(self._on_open_ext_window)
    ml.addWidget(self.btn_abrir_ventana_ext)

    self.chk_ocultar_local = QtWidgets.QCheckBox("Modo Sujeto Ciego en Monitor Principal (Ocultar vocal gigante al operador)")
    self.chk_ocultar_local.setChecked(bool(getattr(self.plotter, 'blind_subject_hide_local_vowel', False)))
    self.chk_ocultar_local.setStyleSheet("color: #ffcc00; font-weight: bold;")
    self.chk_ocultar_local.toggled.connect(self._on_toggle_blind_local)
    ml.addWidget(self.chk_ocultar_local)
    lyt.addWidget(grp_mon)

    grp_help = QtWidgets.QGroupBox("Guía de Conexión Rápida (Tailscale / USB / 2do Monitor)")
    hl = QtWidgets.QVBoxLayout(grp_help)
    lbl_h = QtWidgets.QLabel(
      "• <b>Opción 1 (Tailscale — Recomendada con Wi-Fi de Física):</b> Con Tailscale activo en la PC (<b>100.118.12.94</b>) "
      "y en tu celular (<b>100.102.47.75</b>), entra desde el navegador del celular a <b>http://100.118.12.94:8765/</b>.<br>"
      "• <b>Opción 2 (Cable USB):</b> Conecta el celular por USB y pulsa 'Túnel USB' para abrir <b>http://127.0.0.1:8765/</b>.<br>"
      "• <b>Opción 3 (2do Monitor HDMI):</b> Pulsa 'Abrir Ventana Externa', arrástrala al segundo monitor y toca F11."
    )
    lbl_h.setWordWrap(True)
    lbl_h.setStyleSheet("color: #aaddff; font-size: 11px;")
    hl.addWidget(lbl_h)
    lyt.addWidget(grp_help)

    close_btn = QtWidgets.QPushButton("Cerrar")
    close_btn.clicked.connect(self.accept)
    lyt.addWidget(close_btn)
    self._refresh_status_label()

  def _refresh_status_label(self):
    srv = getattr(self.plotter, 'remote_display_server', None)
    ip_ts = self.edit_ts_pc.text().strip() or _ip_tailscale("100.118.12.94")
    ip_celu = self.edit_ts_celu.text().strip() or "100.102.47.75"
    if srv is not None and srv.activo():
      ip_lan = _ip_local()
      p = srv.puerto
      tok = f"?t={srv.token}" if srv.token else ""
      self.lbl_estado_srv.setText(
        f"<span style='color:#39ff14; font-weight:bold;'>ACTIVO (0.0.0.0:{p})</span><br>"
        f"• <b>Tailscale (para Celu {ip_celu}):</b> <span style='color:#39ff14; font-size:14px; font-weight:bold;'>http://{ip_ts}:{p}/{tok}</span><br>"
        f"• USB Local: <b>http://127.0.0.1:{p}/{tok}</b> | LAN Física: http://{ip_lan}:{p}/{tok}"
      )
      self.btn_toggle_srv.setText("Detener Servidor Remoto")
    else:
      self.lbl_estado_srv.setText("<span style='color:#ff5555; font-weight:bold;'>DETENIDO</span>")
      self.btn_toggle_srv.setText("Iniciar Servidor Remoto")

  def _on_toggle_srv(self):
    srv = getattr(self.plotter, 'remote_display_server', None)
    if srv is not None and srv.activo():
      self.plotter.detener_servidor_display_remoto()
    else:
      self.plotter.iniciar_servidor_display_remoto(
        puerto=self.spin_puerto.value(),
        token=self.edit_token.text().strip()
      )
    self._refresh_status_label()

  def _on_tailscale_check(self):
    ip_pc = self.edit_ts_pc.text().strip() or "100.118.12.94"
    ip_celu = self.edit_ts_celu.text().strip() or "100.102.47.75"
    puerto = self.spin_puerto.value()
    self.plotter.config_mgr.set("display_remoto", "tailscale_ip_pc", ip_pc)
    self.plotter.config_mgr.set("display_remoto", "tailscale_ip_celu", ip_celu)
    srv = getattr(self.plotter, 'remote_display_server', None)
    if srv is None or not srv.activo():
      self.plotter.iniciar_servidor_display_remoto(
        puerto=puerto,
        token=self.edit_token.text().strip()
      )
    self._refresh_status_label()
    ok, msg = configurar_red_tailscale(puerto=puerto, ip_pc=ip_pc, ip_celu=ip_celu)
    QtWidgets.QMessageBox.information(self, "Red Tailscale Configurada", msg)

  def _on_adb_reverse(self):
    srv = getattr(self.plotter, 'remote_display_server', None)
    if srv is None or not srv.activo():
      self.plotter.iniciar_servidor_display_remoto(
        puerto=self.spin_puerto.value(),
        token=self.edit_token.text().strip()
      )
      self._refresh_status_label()
    ok, msg = configurar_adb_reverse(self.spin_puerto.value())
    QtWidgets.QMessageBox.information(self, "Túnel USB (ADB Reverse)", msg)

  def _on_open_ext_window(self):
    self.plotter.abrir_ventana_estimulos_segundo_monitor()

  def _on_toggle_blind_local(self, checked):
    self.plotter.blind_subject_hide_local_vowel = bool(checked)
    self.plotter.config_mgr.set("display_remoto", "modo_sujeto_ciego_ocultar_visor_local", bool(checked))


class AutoForgeDialog(QtWidgets.QDialog):
  """
  Clase AutoForgeDialog.

  Representa y gestiona las operaciones relacionadas con AutoForgeDialog.
  """
  def __init__(self, parent=None):
    """
    Ejecuta la funcionalidad de __init__.

    Args:
      parent (Any): Argumento posicional parent.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Configuración de Autograbado")
    self.setMinimumWidth(380)
    self.setStyleSheet("""
      QDialog { background-color: #0d0d0d; color: #00ffff; font-family: 'Courier New', monospace; }
      QLabel { color: #00ffcc; font-weight: bold; }
      QLineEdit, QSpinBox, QComboBox { background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc; padding: 4px; border-radius: 3px; }
      QGroupBox { border: 1px solid #ff00ff; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #ff00ff; }
      QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
    """)
    
    self.layout = QtWidgets.QFormLayout(self)
    
    self.edit_prueba = QtWidgets.QLineEdit("Prueba1")
    self.edit_sujeto = QtWidgets.QLineEdit("Sujeto1")
    self.spin_reps = QtWidgets.QSpinBox()
    self.spin_reps.setRange(1, 1000)
    self.spin_reps.setValue(5)
    self.spin_bpm = QtWidgets.QSpinBox()
    self.spin_bpm.setRange(10, 240)
    bpm_init = getattr(parent, 'spin_bpm', None).value() if hasattr(parent, 'spin_bpm') else 30
    self.spin_bpm.setValue(int(bpm_init))
    self.spin_subdivs = QtWidgets.QSpinBox()
    self.spin_subdivs.setRange(1, 8)
    subdiv_init = getattr(parent, 'spin_metro_subdivs', None).value() if hasattr(parent, 'spin_metro_subdivs') else 4
    self.spin_subdivs.setValue(int(subdiv_init))
    
    self.layout.addRow("Prueba:", self.edit_prueba)
    self.layout.addRow("Sujeto:", self.edit_sujeto)
    self.layout.addRow("Repeticiones:", self.spin_reps)
    self.layout.addRow("BPM Metrónomo:", self.spin_bpm)
    self.layout.addRow("Subdivisiones / Compás:", self.spin_subdivs)
    
    prot_cfg = parent.config_mgr.get_protocol_config() if (parent and hasattr(parent, 'config_mgr') and hasattr(parent.config_mgr, 'get_protocol_config')) else {}
    self.chk_calibrar_cvm = QtWidgets.QCheckBox("Calibrar Contracción Máxima (A, I, U) [Opcional]")
    self.chk_calibrar_cvm.setChecked(bool(prot_cfg.get("calibrar_cvm_por_defecto", False)))
    self.chk_calibrar_cvm.setToolTip("Desmarcado por defecto para no fatigar al sujeto antes de la Secuencia Continua.")
    self.chk_calibrar_cvm.setStyleSheet("color: #00ffcc; font-weight: bold;")
    self.cmb_cvm_modo = QtWidgets.QComboBox()
    self.cmb_cvm_modo.addItems([
      "Guiado Sostenido SIN Metrónomo (2.5s sostén + 2.5s descanso) [Recomendado]",
      "Rítmico CON Metrónomo (Clásico)"
    ])
    if str(prot_cfg.get("cvm_modo", "guiado_sin_metronomo")) == "con_metronomo":
      self.cmb_cvm_modo.setCurrentIndex(1)
    else:
      self.cmb_cvm_modo.setCurrentIndex(0)
    self.layout.addRow("Calibración CVM:", self.chk_calibrar_cvm)
    self.layout.addRow("Modo de CVM:", self.cmb_cvm_modo)

    self.chk_abrir_fotometria = QtWidgets.QCheckBox("Abrir Asistente de Cámara 3D al iniciar")
    self.chk_abrir_fotometria.setChecked(bool(prot_cfg.get("abrir_fotometria_al_iniciar", False)))
    self.chk_abrir_fotometria.setToolTip("Si está desmarcado, inicia la secuencia directamente sin sondear cámaras web (evita bloqueos si no hay webcam).")
    self.chk_abrir_fotometria.setStyleSheet("color: #00ffff; font-weight: bold;")
    self.layout.addRow("Fotometría 3D:", self.chk_abrir_fotometria)

    self.chk_jitter = QtWidgets.QCheckBox("Activar Jitter Anti-Sincronización")
    jit_init = getattr(parent, 'chk_metro_jitter', None).isChecked() if hasattr(parent, 'chk_metro_jitter') else False
    self.chk_jitter.setChecked(jit_init)
    self.chk_jitter.setStyleSheet("color: #ff55ff; font-weight: bold;")
    self.spin_jitter_pct = QtWidgets.QSpinBox()
    self.spin_jitter_pct.setRange(5, 50)
    self.spin_jitter_pct.setSingleStep(5)
    pct_init = getattr(parent, 'spin_metro_jitter_pct', None).value() if hasattr(parent, 'spin_metro_jitter_pct') else 25
    self.spin_jitter_pct.setValue(int(pct_init))
    self.cmb_jitter_mode = QtWidgets.QComboBox()
    self.cmb_jitter_mode.addItems(["exponential_hazard", "uniform", "gaussian"])
    if hasattr(parent, 'cmb_metro_jitter_mode'):
      self.cmb_jitter_mode.setCurrentText(parent.cmb_metro_jitter_mode.currentText())
    self.layout.addRow("Modo Jitter:", self.chk_jitter)
    self.layout.addRow("Jitter (+/- %):", self.spin_jitter_pct)
    self.layout.addRow("Perfil Jitter:", self.cmb_jitter_mode)

    self.chk_hablar_vocales = QtWidgets.QCheckBox("Pronunciar 'Di Vocal' en cada cambio (con auriculares)")
    self.chk_hablar_vocales.setChecked(False) # DESMARCADO POR DEFECTO para evitar interferencias
    self.chk_hablar_vocales.setStyleSheet("color: #ffcc00; font-weight: bold;")
    self.chk_hablar_vocales.setToolTip("Desmarcado por defecto. Si se activa, se recomienda usar auriculares para evitar crosstalk acústico.")

    self.cmb_speaking_mode = QtWidgets.QComboBox()
    self.cmb_speaking_mode.addItems([
      "Solo fases de transición (Ruido Basal, Preparación, Descanso) [Por Defecto]",
      "Completo: Fases + pronunciar 'Di Vocal' en cada cambio (con auriculares)",
      "Silenciado Total (Sin audio de voz)"
    ])
    self.cmb_speaking_mode.setCurrentIndex(0) # Por defecto: solo frases de transición

    self.chk_hablar_fases = QtWidgets.QCheckBox("Activar audio en fases (Ruido/Prep/Descanso)")
    self.chk_hablar_fases.setChecked(True)
    self.chk_hablar_fases.setStyleSheet("color: #00ffcc; font-weight: bold;")

    self.layout.addRow("Locución Vocales:", self.chk_hablar_vocales)
    self.layout.addRow("Locución Fases:", self.chk_hablar_fases)
    self.layout.addRow("Modo de Audio:", self.cmb_speaking_mode)
    
    # Fila de control de palabras y aleatorización
    self.words_row_layout = QtWidgets.QHBoxLayout()
    self.words_row_layout.setSpacing(6)

    self.btn_edit_words = QtWidgets.QPushButton(" Editar Palabras")
    self.btn_edit_words.setStyleSheet("background-color: #121f2d; color: #00ffff; font-weight: bold; font-family: 'Courier New'; font-size: 13px; padding: 6px; border: 1.5px solid #00ffff; border-radius: 4px;")
    self.btn_edit_words.clicked.connect(self.abrir_editor_palabras)

    self.btn_randomize_direct = QtWidgets.QPushButton(" Aleatorizar (A E I O U...)")
    self.btn_randomize_direct.setStyleSheet("background-color: #2b1238; color: #ff55ff; font-weight: bold; font-family: 'Courier New'; font-size: 13px; padding: 6px; border: 1.5px solid #ff55ff; border-radius: 4px;")
    self.btn_randomize_direct.setToolTip("Mezcla aleatoriamente las palabras/vocales guardadas")
    self.btn_randomize_direct.clicked.connect(self.aleatorizar_palabras_directo)

    self.btn_debruijn_direct = QtWidgets.QPushButton(" Balancear De Bruijn")
    self.btn_debruijn_direct.setStyleSheet("background-color: #0d2b22; color: #39ff14; font-weight: bold; font-family: 'Courier New'; font-size: 13px; padding: 6px; border: 1.5px solid #39ff14; border-radius: 4px;")
    self.btn_debruijn_direct.setToolTip("Genera un bloque balanceado De Bruijn donde todas las transiciones entre vocales son equitativas")
    self.btn_debruijn_direct.clicked.connect(self.balancear_palabras_debruijn_directo)

    self.words_row_layout.addWidget(self.btn_edit_words)
    self.words_row_layout.addWidget(self.btn_randomize_direct)
    self.words_row_layout.addWidget(self.btn_debruijn_direct)
    self.layout.addRow("Vocales / Palabras:", self.words_row_layout)

    self.chk_debruijn = QtWidgets.QCheckBox("Balancear Transiciones De Bruijn en Todos los Ciclos")
    self.chk_debruijn.setChecked(True)
    self.chk_debruijn.setToolTip("En Secuencia Continua, permuta cada ciclo sin repetir vocal en los bordes y equilibrando todos los pares (v_i -> v_j)")
    self.chk_debruijn.setStyleSheet("color: #39ff14; font-weight: bold;")
    self.layout.addRow("Secuencia IA:", self.chk_debruijn)

    # --- Asignación de Músculos ---
    self.group_muscles = QtWidgets.QGroupBox("Asignación de Músculos")
    self.layout_muscles = QtWidgets.QFormLayout(self.group_muscles)
    self.muscle_inputs = []
    
    num_chans = getattr(parent, 'NUM_CANALES', 4)
    parent_muscles = getattr(parent, 'nombres_musculos', [])
    
    musculos_sugeridos = [
      "Masetero", "Risorio", "Depresor", "Micrófono", 
      "Vientre Anterior del Digástrico", "Orbicular de los Labios", 
      "Cigomático Mayor", "Temporal"
    ]
    
    for i in range(num_chans):
      default_val = parent_muscles[i] if i < len(parent_muscles) else f"Canal {i}"
      cmb = QtWidgets.QComboBox()
      cmb.setEditable(True)
      for m in musculos_sugeridos:
        cmb.addItem(m)
      if default_val not in musculos_sugeridos:
        cmb.insertItem(0, default_val)
      cmb.setCurrentText(default_val)
      
      lbl = QtWidgets.QLabel(f"Canal {i}:")
      self.layout_muscles.addRow(lbl, cmb)
      self.muscle_inputs.append(cmb)
      
    self.layout.addRow(self.group_muscles)
    
    self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    self.buttonBox.accepted.connect(self.accept)
    self.buttonBox.rejected.connect(self.reject)
    self.layout.addWidget(self.buttonBox)

  def get_muscle_names(self):
    return [inp.currentText().strip() for inp in self.muscle_inputs]

  def balancear_palabras_debruijn_directo(self):
    """Genera una secuencia balanceada De Bruijn sobre las vocales de palabras.txt."""
    ruta_palabras = PALABRAS_TXT_PATH
    if not os.path.exists(ruta_palabras):
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("A\nE\nI\nO\nU\n")
    with open(ruta_palabras, 'r', encoding='utf-8') as f:
      words = [line.strip().upper() for line in f if line.strip()]
    if words:
      # Eliminar duplicados preservando elementos únicos para armar el bloque base
      unicos = list(dict.fromkeys(words))
      seq_bal = generar_secuencia_balanceada_debruijn(unicos, num_ciclos=1)
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("\n".join(seq_bal) + "\n")
      orden_str = " -> ".join(seq_bal)
      QtWidgets.QMessageBox.information(
        self, "Secuencia Balanceada De Bruijn",
        f"Bloque base balanceado con éxito ({len(seq_bal)} vocales):\n\n{orden_str}\n\n"
        f"Además, al mantener activa la casilla 'Balancear Transiciones De Bruijn', "
        f"los {self.spin_reps.value()} ciclos tendrán todas las transiciones (v_i -> v_j) equilibradas."
      )

  def aleatorizar_palabras_directo(self):
    """Mezcla aleatoriamente las palabras en palabras.txt directamente desde la ventana principal."""
    import random
    ruta_palabras = PALABRAS_TXT_PATH
    if not os.path.exists(ruta_palabras):
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("A\nE\nI\nO\nU\n")
    with open(ruta_palabras, 'r', encoding='utf-8') as f:
      words = [line.strip().upper() for line in f if line.strip()]
    if words:
      random.shuffle(words)
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("\n".join(words) + "\n")
      orden_str = " -> ".join(words)
      QtWidgets.QMessageBox.information(
        self, "Lista Aleatorizada",
        f"Palabras aleatorizadas con exito ({len(words)} items):\n\n{orden_str}"
      )

  def abrir_editor_palabras(self):
    """
    Abre la ventana para editar la lista de palabras/vocales, permitiendo
    escribir la lista, aleatorizar el orden y previsualizar la secuencia.
    """
    ruta_palabras = PALABRAS_TXT_PATH
    if not os.path.exists(ruta_palabras):
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("A\nE\nI\nO\nU\n")
    
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle("Editor de Palabras y Vocales - AutoForge")
    dialog.resize(380, 480)
    dialog.setStyleSheet("""
      QDialog { background-color: #0b0f19; color: #00ffff; font-family: 'Courier New', monospace; }
      QLabel { color: #00ffcc; font-weight: bold; }
      QTextEdit { background-color: #060910; color: #00ffcc; border: 1.5px solid #00ffff; font-size: 15px; font-weight: bold; padding: 8px; border-radius: 4px; }
      QPushButton { background-color: #111a2e; color: #00ffcc; border: 1.5px solid #00ffff; border-radius: 4px; padding: 6px 12px; font-weight: bold; }
      QPushButton:hover { background-color: #00ffff; color: #050508; }
    """)
    dialog_layout = QtWidgets.QVBoxLayout(dialog)
    dialog_layout.setContentsMargins(12, 12, 12, 12)
    dialog_layout.setSpacing(8)
    
    lbl = QtWidgets.QLabel("Escribe una vocal o palabra por línea:")
    dialog_layout.addWidget(lbl)
    
    text_edit = QtWidgets.QTextEdit()
    with open(ruta_palabras, 'r', encoding='utf-8') as f:
      text_edit.setPlainText(f.read())
    dialog_layout.addWidget(text_edit, stretch=1)
    
    # Botonera de herramientas rápidas: Aleatorizar y Restablecer A E I O U
    tools_layout = QtWidgets.QHBoxLayout()
    tools_layout.setSpacing(6)
    
    btn_randomize = QtWidgets.QPushButton("Aleatorizar Lista")
    btn_randomize.setStyleSheet("QPushButton { background-color: #2b1238; color: #ff55ff; border: 1.5px solid #ff55ff; font-weight: bold; padding: 6px; } QPushButton:hover { background-color: #ff55ff; color: #000000; }")
    btn_randomize.setToolTip("Mezcla aleatoriamente las palabras/vocales escritas (ej: A E I O U)")
    
    btn_load_aeiou = QtWidgets.QPushButton("Cargar A E I O U")
    btn_load_aeiou.setStyleSheet("QPushButton { background-color: #122238; color: #00ffff; border: 1.5px solid #00ffff; font-weight: bold; padding: 6px; } QPushButton:hover { background-color: #00ffff; color: #000000; }")
    btn_load_aeiou.setToolTip("Carga las cinco vocales canonicas en orden estandar")
    
    tools_layout.addWidget(btn_randomize)
    tools_layout.addWidget(btn_load_aeiou)
    dialog_layout.addLayout(tools_layout)
    
    lbl_preview = QtWidgets.QLabel("Orden actual: --")
    lbl_preview.setStyleSheet("color: #ffff00; font-size: 11px; font-weight: bold;")
    lbl_preview.setWordWrap(True)
    dialog_layout.addWidget(lbl_preview)
    
    def actualizar_preview():
      lineas = [l.strip().upper() for l in text_edit.toPlainText().splitlines() if l.strip()]
      if lineas:
        lbl_preview.setText(f"Orden ({len(lineas)} items): {' -> '.join(lineas)}")
      else:
        lbl_preview.setText("Lista vacía")
        
    actualizar_preview()
    text_edit.textChanged.connect(actualizar_preview)
    
    def on_randomize():
      import random
      lineas = [l.strip().upper() for l in text_edit.toPlainText().splitlines() if l.strip()]
      if lineas:
        random.shuffle(lineas)
        text_edit.setPlainText("\n".join(lineas) + "\n")
        actualizar_preview()
        
    def on_load_aeiou():
      text_edit.setPlainText("A\nE\nI\nO\nU\n")
      actualizar_preview()
      
    btn_randomize.clicked.connect(on_randomize)
    btn_load_aeiou.clicked.connect(on_load_aeiou)
    
    btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
    btn_box.accepted.connect(dialog.accept)
    btn_box.rejected.connect(dialog.reject)
    dialog_layout.addWidget(btn_box)
    
    if dialog.exec() == QtWidgets.QDialog.Accepted:
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        texto_guardar = text_edit.toPlainText().strip() + "\n"
        f.write(texto_guardar)
      QtWidgets.QMessageBox.information(self, "Guardado", "Lista de palabras guardada con éxito.")

class RealTimePlotter(QtWidgets.QWidget):
  """
  Clase RealTimePlotter.

  Representa y gestiona las operaciones relacionadas con RealTimePlotter.
  """
  mostrar_preview_signal = QtCore.Signal(str)

  def __init__(self):
    """
    Ejecuta la funcionalidad de __init__.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    super().__init__()

    # --- Constantes de la GUI ---
    self.BTN_START_STYLE = "background-color: #050505; color: #00FF00; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px 12px; border: 2px solid #00FF00; border-radius: 4px; min-width: 175px;"
    self.BTN_STOP_STYLE = "background-color: #050505; color: #FF0000; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px 12px; border: 2px solid #FF0000; border-radius: 4px; min-width: 175px;"
    self.BTN_REC_START_STYLE = "background-color: #050505; color: #00FFFF; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px 10px; border: 2px solid #00FFFF; border-radius: 4px;"
    self.BTN_REC_STOP_STYLE = "background-color: #FF0000; color: #000000; font-weight: bold; font-family: 'Courier New'; font-size: 12px; padding: 6px 10px; border: 2px solid #FF0000; border-radius: 4px;"

    # --- Estado interno ---
    self.preroll_buffer = PreRollRecordingBuffer(sample_rate=2000, preroll_seconds=1.0)
    self.spoken_audio_log = []
    self._last_spoken_continuo_word = None
    self._pre_announced_continuo_beat = -1
    self.invalid_pulses_set = set()
    self.autoforge_full_sequence = []
    self.autoforge_use_debruijn = True
    self.diagnostico_contacto_electrodos = {}
    self.last_dataset_summary = {}
    self.voice_cue_player = VoiceCuePlayer(parent_plotter=self)
    self.vocal_cue_log = []
    self.sync_evaluation_log = []
    self.is_paused = False
    self.pause_start_perf = 0.0
    self.total_paused_duration = 0.0

    self.last_separability_metrics = None
    self.txt_notas_live = QtWidgets.QTextEdit()

    self.is_recording = False
    self.counting_started = False # NUEVO: Bandera para controlar el inicio del conteo del metrónomo
    self.current_recording = []
    self.recording_start_time = None
    # Muestras que el .wav ya contiene al fijar el ancla (pre-roll o ruido basal)
    self._offset_muestras_ancla = 0
    self.acquisition_thread = None
    # --- REFACTOR: Encapsular cola y evento ---
    self.data_queue = queue.Queue()
    self.stop_event = threading.Event()

    self.is_acquiring = False
    
    from utils.config_manager import ConfigManager
    self.config_mgr = ConfigManager()

    # Canales fisicos y hardware
    self.nombres_musculos_hw = [None] * 16
    self.factores_calibracion_hw = [495.0] * 16
    self.active_channel_indices = []
    self._autoforge_token = 0
    self.autoforge_session_dirs = []
    self.cvm_completed = False
    self.final_noise_completed = False
    self.is_cvm_active = False
    self.is_recording_final_noise = False
    
    # --- Propiedades iniciales seguras (evita SAMPLE_RATE=0 antes de iniciar adquisición) ---
    self.NUM_CANALES = 0
    self.CANALES_DAQ = []
    self.SAMPLE_RATE = 2000
    self.PLOT_DURATION_S = 10
    self.PLOT_SAMPLES = 20000
    self.CHUNK_DURATION_S = 0.05
    self._photometry_session_active = False
    
    # --- Buffers de ploteo ---
    # Se inicializarán al empezar la adquisición
    self.plot_buffer_datos = None
    self.plot_vector_tiempo = None
    # self.plot_vector_tiempo = np.linspace(-PLOT_DURATION_S, 0, PLOT_SAMPLES)

    # --- NUEVO: Variables para piso de ruido ---
    self.noise_data_accumulated = []
    self.noise_levels = []
    self.noise_lines = []
    self.noise_lines_neg = []
    self.noise_regions = []
    self.dynamic_noise_regions = []
    self.noise_calculated = False

    # --- NUEVO: Buffers y configuración del espectrograma ---
    self.spectrogram_buffer = None
    self.SPECTROGRAM_HISTORY_LEN = 200 # Número de FFTs en el historial del espectrograma
    self.SPECTROGRAM_FFT_LEN = 256   # Puntos para la FFT (mejor si es potencia de 2)
    self.spectrogram_channel_index = 0
    

    # --- Estado del Trigger ---
    self.trigger_last_values = None
    self.is_trigger_line_moving = False # Para evitar bucles de señales

    # --- NUEVO: Estado del Filtro ---
    self.filter_sos = None # Coeficientes del filtro
    self.filter_zi = None

    self.autoforge_word_idx = 0
    self.autoforge_words = []
    self.autoforge_base_name = ""
    self.autoforge_target_reps = 25
    self.autoforge_current_reps = 0

    # Instanciacion de Fotometria 3D y Triangulacion Espacial
    if WebcamPhotometryManager is not None:
      try:
        self.photometry_mgr = WebcamPhotometryManager(parent=self, camera_index=0)
      except Exception as e:
        print(f"[FOTOMETRIA] Advertencia al instanciar WebcamPhotometryManager: {e}")
        self.photometry_mgr = None
    else:
      self.photometry_mgr = None

    # --- NUEVO: Estado del Filtro Notch ---
    self.notch_sos = None
    self.notch_zi = None

    self.FILTER_ORDER = 4 # Orden del filtro Butterworth

    # --- Configurar la ventana principal ---
    self.setWindowTitle(f"Ñandú LSD - Visor y Grabador de Señales Emg v{__version__}")
    
    # --- Detección y Adaptación de Escala de Pantalla (Responsive) ---
    screen = QtGui.QGuiApplication.primaryScreen()
    if screen:
      avail = screen.availableGeometry()
      self.screen_w = avail.width()
      self.screen_h = avail.height()
    else:
      self.screen_w, self.screen_h = 1920, 1080
    
    self.is_compact_screen = (self.screen_h <= 850 or self.screen_w <= 1400)
    init_w = min(1200, int(self.screen_w * 0.95))
    init_h = min(750, int(self.screen_h * 0.92))
    self.setMinimumSize(850, 550)
    self.resize(init_w, init_h)
    
    # --- REFACTOR: Dividir la configuración de la UI en métodos ---
    self._setup_ui_layouts()
    self._setup_ui_config_panel()
    self._setup_ui_filter_panel()
    self._setup_ui_controls()
    self._setup_notas_panel()
    self._setup_ui_plots()
    self._setup_ui_final_layout()
    self._connect_signals()
    self._load_protocol_config()
    
    # --- Configuración inicial ---
    self.on_autoscroll_toggle() 
    self.set_controls_enabled(False) # Deshabilitar controles al inicio
    
    # --- Detector de Pulsos Fonatorios (Micrófono + Gate Doble) + Medidor PCA en Vivo ---
    self.detected_pulses_history = []
    self.latest_pulse_event = None
    self.live_pca_features = []
    self.live_pca_labels = []
    self.remote_display_server = None
    self.external_stimulus_window = None
    self.blind_subject_hide_local_vowel = False
    self.session_timer = QtCore.QTimer(self)
    self.session_timer.timeout.connect(self._tick_session_timer)
    self.pulse_detector = PhonatoryPulseDetector(plotter=self)

    # --- Timer para actualizar el plot ---
    self.timer = QtCore.QTimer()
    self.timer.timeout.connect(self.actualizar_plot)
    self.timer.start(30) # Actualiza la GUI cada 30 ms
    
    # Auto-iniciar servidor de estímulos (Tailscale 100.118.12.94:8765 / USB) en segundo plano
    try:
      rem_cfg = self.config_mgr.get_remote_display_config() if hasattr(self.config_mgr, 'get_remote_display_config') else {}
      if rem_cfg.get("habilitado", True):
        self.iniciar_servidor_display_remoto(
          puerto=int(rem_cfg.get("puerto", 8765)),
          token=str(rem_cfg.get("token", ""))
        )
    except Exception:
      pass

    # --- NUEVO: Maximizar ventana por defecto ---
    self.showMaximized()

  def _load_protocol_config(self):
    """
    Carga la configuración inicial desde config_general.json (y metronome_config.json si existe),
    permitiendo cambiar libremente BPM, subdivisiones, tiempo de ruido y filtros en cualquier momento.
    """
    adq_cfg = self.config_mgr.get("adquisicion") or {}
    bpm = int(adq_cfg.get("bpm", 25))
    subdivs = int(adq_cfg.get("subdivisiones", 4))
    notch_enabled = bool(adq_cfg.get("notch_50hz_activo", False))
    noise_s = float(adq_cfg.get("ruido_segundos", 3.0))
    if os.path.exists(METRONOME_CONFIG_PATH):
      try:
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          m_cfg = json.load(f)
          if 'last_bpm' in m_cfg:
            bpm = int(m_cfg['last_bpm'])
          if 'subdivisions' in m_cfg:
            subdivs = int(m_cfg['subdivisions'])
      except Exception:
        pass
    self.spin_bpm.setValue(max(10, min(240, bpm)))
    if hasattr(self, 'spin_metro_subdivs'):
      self.spin_metro_subdivs.setValue(max(1, min(8, subdivs)))
    if hasattr(self, 'spin_noise_duration'):
      self.spin_noise_duration.setValue(max(0.5, min(20.0, noise_s)))
    self.chk_notch_enable.setChecked(notch_enabled)

  def _setup_ui_layouts(self):
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

    # Nivel 1: Barra Superior de Transporte y Diagnostico en 2 filas limpias (cero botones cortados)
    self.transport_frame = QtWidgets.QFrame()
    self.transport_frame.setStyleSheet(
      "QFrame { background-color: #080808; border: 1px solid #00FFFF; border-radius: 4px; }"
    )
    self.transport_rows_layout = QtWidgets.QVBoxLayout(self.transport_frame)
    self.transport_rows_layout.setContentsMargins(8, 4, 8, 4)
    self.transport_rows_layout.setSpacing(4)
    self.button_layout = QtWidgets.QHBoxLayout()
    self.button_layout.setContentsMargins(0, 0, 0, 0)
    self.button_layout.setSpacing(6)
    self.tools_bar_layout = QtWidgets.QHBoxLayout()
    self.tools_bar_layout.setContentsMargins(0, 0, 0, 0)
    self.tools_bar_layout.setSpacing(6)
    self.transport_rows_layout.addLayout(self.button_layout)
    self.transport_rows_layout.addLayout(self.tools_bar_layout)

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

    self.filter_groupbox = QtWidgets.QGroupBox("Filtro Digital: Pasa-Banda y Ritmo")
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
    self.spin_plot_duration.setFixedHeight(26)

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
    self.cmb_fuente_prueba.currentIndexChanged.connect(self._on_fuente_prueba_changed)

    self.label_test_freq = QtWidgets.QLabel("Frec. Prueba (Hz):")
    self.spin_test_freq = SpinBox(value=50, step=10, bounds=(1, 5000), int=True)
    self.spin_test_freq.setFixedWidth(65)
    self.spin_test_freq.setFixedHeight(26)

    self.chk_use_metronome = QtWidgets.QCheckBox("Usar Metronomo")
    self.chk_use_metronome.setChecked(True)

    self.chk_use_mic = QtWidgets.QCheckBox("Usar Microfono (PC)")
    self.chk_use_mic.setToolTip("Usa el microfono de la PC en lugar de la DAQ.")
    self.chk_use_mic.setStyleSheet("QCheckBox { color: #00FFFF; font-weight: bold; }")
    self.chk_use_mic.toggled.connect(self._on_modo_mic_toggled)

    self.cmb_fuente_prueba.setMinimumWidth(135)
    self.label_channels = QtWidgets.QLabel("Canales:")
    self.channel_checkboxes = []
    self.channel_layout = QtWidgets.QGridLayout()
    self.channel_layout.setContentsMargins(0, 0, 0, 0)
    self.channel_layout.setHorizontalSpacing(8)
    self.channel_layout.setVerticalSpacing(2)
    any_checked = False
    for i in range(8):
      chk = QtWidgets.QCheckBox(f"ai{i}")
      chk.setMinimumWidth(46)
      is_active = self.config_mgr.config.get("canales", {}).get(f"Canal {i}", {}).get("activo_por_defecto", i < 4)
      if is_active:
        chk.setChecked(True)
        any_checked = True
      chk.toggled.connect(self._on_channel_checkbox_toggled)
      self.channel_checkboxes.append(chk)
      self.channel_layout.addWidget(chk, i // 4, i % 4)
    if not any_checked and self.channel_checkboxes:
      self.channel_checkboxes[0].setChecked(True)

    self.btn_config_musculos_panel = QtWidgets.QPushButton("Configurar Musculos")
    self.btn_config_musculos_panel.setStyleSheet("""
      QPushButton {
        background-color: #141424;
        color: #00ffff;
        font-weight: bold;
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 4px 10px;
        border: 1px solid #00ffff;
        border-radius: 4px;
        font-size: 11px;
      }
      QPushButton:hover {
        background-color: #00ffff;
        color: #000000;
      }
    """)
    self.btn_config_musculos_panel.setToolTip("Configurar nombres de musculos asignados a los canales (detiene la adquisicion si esta corriendo)")
    self.btn_config_musculos_panel.clicked.connect(self.abrir_dialogo_musculos)
    self.btn_config_musculos_panel.hide() # Ya existe CONFIG MUSCULOS en la barra superior; evita apeñuscar ai0..ai7

    self.btn_start_acq = QtWidgets.QPushButton("Iniciar Adquisicion")
    self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)
    self.btn_start_acq.setMinimumWidth(160)

    self.label_bpm = QtWidgets.QLabel("BPM / Sub:")
    self.spin_bpm = SpinBox(value=25, int=True, bounds=(10, 240), step=1)
    self.spin_bpm.setFixedWidth(54)
    self.spin_bpm.setFixedHeight(22)
    self.spin_bpm.setToolTip("BPM del metrónomo (editable en cualquier momento, incluso durante Autograbado o Secuencia Continua)")
    self.spin_metro_subdivs = SpinBox(value=4, int=True, bounds=(1, 8), step=1)
    self.spin_metro_subdivs.setFixedWidth(44)
    self.spin_metro_subdivs.setFixedHeight(22)
    self.spin_bpm.valueChanged.connect(self._on_live_bpm_or_subdiv_changed)
    self.spin_metro_subdivs.valueChanged.connect(self._on_live_bpm_or_subdiv_changed)

    self.label_noise_duration = QtWidgets.QLabel("Noise (s):")
    adq_conf = self.config_mgr.get("adquisicion") or {}
    default_noise = adq_conf.get("ruido_segundos", 3.0)
    self.spin_noise_duration = SpinBox(value=default_noise, dec=True, bounds=(0.5, 20.0), step=0.5)
    self.spin_noise_duration.setFixedWidth(60)
    self.spin_noise_duration.setFixedHeight(22)

    # Grilla ordenada de 4 filas en el Panel 1
    self.config_layout.addWidget(self.label_device, 0, 0)
    self.config_layout.addWidget(self.cmb_device, 0, 1)
    self.config_layout.addWidget(self.label_terminal_config, 0, 2)
    self.config_layout.addWidget(self.cmb_terminal_config, 0, 3)

    self.config_layout.addWidget(self.label_sample_rate, 1, 0)
    self.config_layout.addWidget(self.cmb_sample_rate, 1, 1)
    self.config_layout.addWidget(self.label_plot_duration, 1, 2)
    self.config_layout.addWidget(self.spin_plot_duration, 1, 3)

    test_mode_layout = QtWidgets.QHBoxLayout()
    test_mode_layout.setContentsMargins(0, 0, 0, 0)
    test_mode_layout.setSpacing(4)
    test_mode_layout.addWidget(self.cmb_fuente_prueba)
    test_mode_layout.addWidget(self.spin_test_freq)

    self.config_layout.addWidget(self.chk_modo_prueba, 2, 0)
    self.config_layout.addLayout(test_mode_layout, 2, 1)
    self.config_layout.addWidget(self.chk_use_mic, 2, 2, 1, 2)

    self.config_layout.addWidget(self.label_channels, 3, 0)
    self.config_layout.addLayout(self.channel_layout, 3, 1, 1, 3)

    # --- NUEVO: Puente VM Windows (ZeroMQ) para NI USB-6259 en Pop!_OS / Linux ---
    self.chk_zmq_bridge = QtWidgets.QCheckBox("Puente ZMQ")
    self.chk_zmq_bridge.setToolTip(
      "Recibe datos en tiempo real de la NI USB-6259 desde la VM con Windows vía ZeroMQ (tcp://192.168.122.x:5555)."
    )
    self.chk_zmq_bridge.setStyleSheet("QCheckBox { color: #FFAA00; font-weight: bold; }")
    adq_cfg_init = self.config_mgr.get("adquisicion") or {}
    default_zmq_on = adq_cfg_init.get("usar_puente_zmq", (not NIDAQMX_DISPONIBLE))
    self.chk_zmq_bridge.setChecked(bool(default_zmq_on))
    self.chk_zmq_bridge.toggled.connect(self._on_zmq_bridge_toggled)

    self.edit_zmq_host = QtWidgets.QLineEdit(str(adq_cfg_init.get("zmq_endpoint", "auto:5555")))
    self.edit_zmq_host.setPlaceholderText("auto:5555 o 192.168.122.100:5555")
    self.edit_zmq_host.setToolTip(
      "Endpoint ZeroMQ de la VM Windows (virbr0). Usa 'auto:5555' para autodetección o '192.168.122.x:5555'."
    )
    self.edit_zmq_host.setStyleSheet(
      "QLineEdit { background-color: #0d1117; color: #FFAA00; border: 1px solid #FFAA00; padding: 2px 6px; font-family: Consolas, monospace; font-weight: bold; }"
    )
    self.edit_zmq_host.editingFinished.connect(self.verificar_conexion_daq)

    self.btn_ping_zmq = QtWidgets.QPushButton("Probar VM")
    self.btn_ping_zmq.setFixedWidth(85)
    self.btn_ping_zmq.setToolTip("Verifica la conexión con daq_bridge_server.py en la VM Windows")
    self.btn_ping_zmq.setStyleSheet(
      "QPushButton { background-color: #1f1600; color: #FFAA00; border: 1px solid #FFAA00; border-radius: 3px; font-weight: bold; padding: 2px 6px; } "
      "QPushButton:hover { background-color: #FFAA00; color: #000000; }"
    )
    self.btn_ping_zmq.clicked.connect(self.verificar_conexion_daq)

    zmq_row_layout = QtWidgets.QHBoxLayout()
    zmq_row_layout.setContentsMargins(0, 0, 0, 0)
    zmq_row_layout.setSpacing(6)
    zmq_row_layout.addWidget(QtWidgets.QLabel("VM IP:Puerto:"))
    zmq_row_layout.addWidget(self.edit_zmq_host, 1)
    zmq_row_layout.addWidget(self.btn_ping_zmq)

    self.config_layout.addWidget(self.chk_zmq_bridge, 4, 0)
    self.config_layout.addLayout(zmq_row_layout, 4, 1, 1, 3)

    # Inicializar el Metronomo en su propio panel derecho independiente
    self._setup_native_metronome()

  def _setup_ui_filter_panel(self):
    """Puebla el Panel 2 (Filtros Digitales, Envolvente RMS, Ritmo, Jitter y Voz Guía) con grilla limpia sin solapamientos."""
    self.filter_layout.setContentsMargins(10, 14, 10, 8)
    self.filter_layout.setHorizontalSpacing(8)
    self.filter_layout.setVerticalSpacing(6)

    self.chk_notch_enable = QtWidgets.QCheckBox("Notch 50 Hz")
    self.chk_notch_enable.setChecked(False) # Desactivado por defecto (v6.0)

    self.label_notch_q = QtWidgets.QLabel("Factor Q:")
    self.spin_notch_q = SpinBox(value=2.0, step=1.0, bounds=(1.0, 50.0), dec=True)
    self.spin_notch_q.setFixedWidth(60)
    self.spin_notch_q.setFixedHeight(22)

    self.chk_notch_comb = QtWidgets.QCheckBox("Peine (150/250/350 Hz)")
    self.chk_notch_comb.setChecked(False)

    self.spin_notch_q.valueChanged.connect(self.on_filter_settings_changed)
    self.chk_notch_comb.clicked.connect(self.on_filter_settings_changed)

    self.chk_filter_enable = QtWidgets.QCheckBox("Pasa-Banda")
    self.chk_filter_enable.setChecked(True)

    self.label_low_cut = QtWidgets.QLabel("Baja (Hz):")
    self.spin_low_cut = SpinBox(value=20, step=1, bounds=(1, 20000), int=True)
    self.spin_low_cut.setFixedWidth(60)
    self.spin_low_cut.setFixedHeight(22)
    self.label_high_cut = QtWidgets.QLabel("Alta (Hz):")
    self.spin_high_cut = SpinBox(value=500, step=1, bounds=(2, 22000), int=True)
    self.spin_high_cut.setFixedWidth(60)
    self.spin_high_cut.setFixedHeight(22)
    self.label_filter_order = QtWidgets.QLabel(f"Ord. {self.FILTER_ORDER}")

    self.chk_rms_env = QtWidgets.QCheckBox("Envolvente RMS")
    self.chk_rms_env.setChecked(False)
    self.label_rms_window = QtWidgets.QLabel("Ventana (ms):")
    self.spin_rms_window = SpinBox(value=100.0, step=10.0, bounds=(10.0, 1000.0), dec=True)
    self.spin_rms_window.setFixedWidth(60)
    self.spin_rms_window.setFixedHeight(22)

    self.lbl_preroll = QtWidgets.QLabel("Pre-Roll (s):")
    self.spin_preroll = SpinBox(value=1.0, step=0.5, bounds=(0.0, 5.0), dec=True)
    self.spin_preroll.setFixedWidth(60)
    self.spin_preroll.setFixedHeight(22)
    self.spin_preroll.valueChanged.connect(
      lambda val: self.preroll_buffer.set_config(max(2000, self.SAMPLE_RATE), val)
    )

    self.chk_show_noise = QtWidgets.QCheckBox("Mostrar Ruido")
    self.chk_show_noise.setChecked(False)
    self.chk_show_noise.clicked.connect(self.toggle_noise_regions)

    self.chk_show_peaks = QtWidgets.QCheckBox("Mostrar Picos")
    self.chk_show_peaks.setChecked(False)
    self.chk_show_peaks.clicked.connect(self.toggle_peak_scatter)

    self.chk_show_detection_window = QtWidgets.QCheckBox("Ventana Deteccion")
    self.chk_show_detection_window.setChecked(True)
    self.chk_show_detection_window.setStyleSheet("color: #00FFFF; font-weight: bold;")
    self.chk_show_detection_window.clicked.connect(self.toggle_detection_window)

    # Controles de Modo Jittering y Voz Guía
    self.chk_metro_jitter = QtWidgets.QCheckBox("Jitter Anti-Sync")
    self.chk_metro_jitter.setChecked(False)
    self.chk_metro_jitter.setToolTip("Varía pseudoaleatoriamente el intervalo del metrónomo para evitar anticipación motora.")
    self.label_jitter_pct = QtWidgets.QLabel("Jitter (%):")
    self.spin_metro_jitter_pct = SpinBox(value=25, int=True, bounds=(5, 50), step=5)
    self.spin_metro_jitter_pct.setFixedWidth(60)
    self.spin_metro_jitter_pct.setFixedHeight(22)
    self.label_jitter_mode = QtWidgets.QLabel("Perfil:")
    self.cmb_metro_jitter_mode = QtWidgets.QComboBox()
    self.cmb_metro_jitter_mode.addItems(["exponential_hazard", "uniform", "gaussian"])
    self.cmb_metro_jitter_mode.setFixedHeight(22)
    self.cmb_metro_jitter_mode.setToolTip("exponential_hazard: Tasa de riesgo constante (no predecible). uniform: Uniforme. gaussian: Gaussiano.")

    self.chk_mute_speaking = QtWidgets.QCheckBox("Mutear Speaking")
    self.chk_mute_speaking.setChecked(False)
    self.chk_mute_speaking.setToolTip("Silencia inmediatamente todas las indicaciones habladas (Speaking) de la interfaz y de la cámara sin apagar el metrónomo.")
    self.chk_mute_speaking.toggled.connect(self.on_mute_speaking_toggled)
    self.chk_voice_cues = self.chk_mute_speaking
    self.voice_cue_player.enabled = True

    # Grilla uniforme de 6 columnas:
    # Col 0: Checkbox | Col 1: Label Izq | Col 2: Control Izq | Col 3: Label Der | Col 4: Control Der | Col 5: Extra
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

    # Fila 2: Envolvente RMS + Pre-Roll + Voz Guía
    self.filter_layout.addWidget(self.chk_rms_env, 2, 0)
    self.filter_layout.addWidget(self.label_rms_window, 2, 1)
    self.filter_layout.addWidget(self.spin_rms_window, 2, 2)
    self.filter_layout.addWidget(self.lbl_preroll, 2, 3)
    self.filter_layout.addWidget(self.spin_preroll, 2, 4)
    self.filter_layout.addWidget(self.chk_voice_cues, 2, 5)

    # Fila 3: Metronomo + BPM/Sub + Noise Inicio
    bpm_sub_layout = QtWidgets.QHBoxLayout()
    bpm_sub_layout.setContentsMargins(0, 0, 0, 0)
    bpm_sub_layout.setSpacing(4)
    bpm_sub_layout.addWidget(self.spin_bpm)
    bpm_sub_layout.addWidget(self.spin_metro_subdivs)

    self.filter_layout.addWidget(self.chk_use_metronome, 3, 0)
    self.filter_layout.addWidget(self.label_bpm, 3, 1)
    self.filter_layout.addLayout(bpm_sub_layout, 3, 2)
    self.filter_layout.addWidget(self.label_noise_duration, 3, 3)
    self.filter_layout.addWidget(self.spin_noise_duration, 3, 4, 1, 2)

    # Fila 4: Modo Jittering Anti-Sincronizacion (con su etiqueta Perfil Jitter clara)
    self.label_jitter_mode.setText("Perfil Jitter:")
    self.filter_layout.addWidget(self.chk_metro_jitter, 4, 0)
    self.filter_layout.addWidget(self.label_jitter_pct, 4, 1)
    self.filter_layout.addWidget(self.spin_metro_jitter_pct, 4, 2)
    self.filter_layout.addWidget(self.label_jitter_mode, 4, 3)
    self.filter_layout.addWidget(self.cmb_metro_jitter_mode, 4, 4, 1, 2)

    self.chk_escala_cvm = QtWidgets.QCheckBox("Normalizar Eje Y en % CVM")
    self.chk_escala_cvm.setToolTip("Normaliza las señales EMG en porcentaje de la Contracción Voluntaria Máxima (% CVM) en tiempo real.")
    self.chk_escala_cvm.setChecked(False)
    self.chk_escala_cvm.setStyleSheet("color: #ffcc00; font-weight: bold;")
    self.chk_escala_cvm.toggled.connect(self._on_escala_cvm_toggled)

    self.chk_clamp_500uv = QtWidgets.QCheckBox("Limitar Rango Fisiológico (0–500 µV / Anti-Picos)")
    self.chk_clamp_500uv.setToolTip("Recorta picos eléctricos (> 500 µV) en los canales EMG y mantiene el eje Y entre 0 y 500 µV para evitar saltos bruscos de escala.")
    self.chk_clamp_500uv.setChecked(True)
    self.chk_clamp_500uv.setStyleSheet("color: #00ffcc; font-weight: bold;")

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

    self.btn_record = QtWidgets.QPushButton("Empezar a Grabar")
    self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)

    self.btn_pause = QtWidgets.QPushButton("PAUSAR")
    self.btn_pause.setCheckable(True)
    self.btn_pause.setEnabled(False)
    self.btn_pause.setStyleSheet(BTN_PAUSE_CYBER)
    self.btn_pause.toggled.connect(self.on_pause_toggled)

    self.btn_autoforge = QtWidgets.QPushButton("AUTOGRABADO")
    self.btn_autoforge.setStyleSheet(BTN_AUTOFORGE_DISC_CYBER)
    self.btn_autoforge.clicked.connect(self.iniciar_autoforge)

    self.btn_autoforge_continuo = QtWidgets.QPushButton("SECUENCIA CONTINUA")
    self.btn_autoforge_continuo.setStyleSheet(BTN_AUTOFORGE_CONT_CYBER)
    self.btn_autoforge_continuo.clicked.connect(self.iniciar_autoforge_continuo)

    self.btn_config_musculos = QtWidgets.QPushButton("CONFIG MUSCULOS")
    self.btn_config_musculos.setStyleSheet(BTN_CYBER_TOOL)
    self.btn_config_musculos.setToolTip("Configurar asignacion de musculos por canal")
    self.btn_config_musculos.clicked.connect(self.abrir_dialogo_musculos)

    self.btn_notas = QtWidgets.QPushButton("NOTAS")
    self.btn_notas.setStyleSheet(BTN_CYBER_TOOL)
    self.btn_notas.setToolTip("Mostrar u ocultar el panel de notas del experimento")
    self.btn_notas.clicked.connect(self.toggle_notas_panel)

    self.btn_fotometria = QtWidgets.QPushButton("FOTOMETRIA 3D")
    self.btn_fotometria.setStyleSheet(
      "QPushButton { background-color: #00332a; color: #00FFCC; border: 1.5px solid #00FFCC; padding: 7px 12px; font-weight: bold; border-radius: 4px; } "
      "QPushButton:hover { background-color: #00FFCC; color: #050508; }"
    )
    self.btn_fotometria.setToolTip("Abrir Asistente de Fotometria Facial 3D y Triangulacion de Electrodos")
    self.btn_fotometria.clicked.connect(self.abrir_asistente_fotometria)

    self.btn_mute_speaking = QtWidgets.QPushButton("SPEAKING: ON")
    self.btn_mute_speaking.setCheckable(True)
    self.btn_mute_speaking.setChecked(False)
    self.btn_mute_speaking.setToolTip("Mutear o activar el audio de voz hablada (Speaking) en todo el sistema")
    self.btn_mute_speaking.setStyleSheet(
      "QPushButton { background-color: #0f2419; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 10px; font-weight: bold; border-radius: 4px; } "
      "QPushButton:checked { background-color: #3a0d16; color: #FF4455; border: 1.5px solid #FF4455; }"
    )
    self.btn_mute_speaking.toggled.connect(self.on_mute_speaking_toggled)

    self.label_rec_time = QtWidgets.QLabel("SISTEMA EN ESPERA")
    self.label_rec_time.setAlignment(QtCore.Qt.AlignCenter)
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #00FFFF; font-size: 13px; background-color: #050505; border: 1px solid #222222; padding: 4px 10px; border-radius: 3px;")
    self.label_rec_time.setVisible(True)

    # Boton de Consola / Terminal de Errores
    self.btn_terminal = QtWidgets.QPushButton("CONSOLA / LOGS (0)")
    self.btn_terminal.setToolTip("Abre la consola integrada para inspeccionar mensajes, errores de NIDAQmx o abrir una terminal del sistema.")
    self.btn_terminal.setStyleSheet("QPushButton { background-color: #0B0F19; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 12px; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #39FF14; color: #050508; }")
    self.btn_terminal.clicked.connect(self.mostrar_consola_diagnostico)

    self.btn_start_acq.setMinimumWidth(185)
    self.btn_record.setText("GRABAR LIBRE (Manual)")
    self.btn_record.setToolTip("Inicia o detiene una grabación libre manual (sin protocolo guiado de vocales)")
    self.label_rec_time.setMinimumWidth(290)

    self.btn_marcar_invalido = QtWidgets.QPushButton("PULSO INVÁLIDO (Supr)")
    self.btn_marcar_invalido.setToolTip("Marca el pulso actual como inválido (tos, deglución o error del sujeto) [Atajo: Tecla Backspace / Supr]")
    self.btn_marcar_invalido.setStyleSheet(
      "QPushButton { background-color: #2b1d00; color: #FFCC00; border: 1.5px solid #FFCC00; padding: 5px 10px; font-weight: bold; border-radius: 4px; font-size: 11px; } "
      "QPushButton:hover { background-color: #FFCC00; color: #050508; }"
    )
    self.btn_marcar_invalido.clicked.connect(self.marcar_pulso_actual_invalido)

    self.btn_pantalla_externa = QtWidgets.QPushButton("PANTALLA EXTERNA (Tailscale/USB)")
    self.btn_pantalla_externa.setToolTip("Transmitir vocales por Tailscale (100.118.12.94), USB o 2do Monitor para Modo Sujeto Ciego")
    self.btn_pantalla_externa.setStyleSheet(
      "QPushButton { background-color: #1a0f2e; color: #c77dff; border: 1.5px solid #c77dff; padding: 5px 10px; font-weight: bold; border-radius: 4px; font-size: 11px; } "
      "QPushButton:hover { background-color: #c77dff; color: #050508; }"
    )
    self.btn_pantalla_externa.clicked.connect(self.abrir_dialogo_pantalla_externa)

    # Fila 1: Transporte Principal + Modos de Grabación + Estado / Cuenta Regresiva Amplia
    self.button_layout.addWidget(self.btn_start_acq)
    self.button_layout.addWidget(self.btn_record)
    self.button_layout.addWidget(self.btn_pause)
    self.button_layout.addWidget(self.btn_autoforge)
    self.button_layout.addWidget(self.btn_autoforge_continuo)
    self.button_layout.addWidget(self.label_rec_time, stretch=1)

    # Fila 2: Estado DAQ + Herramientas de Configuración + Pantalla Externa + Audio + Consola
    self.tools_bar_layout.addWidget(self.lbl_daq_status)
    self.tools_bar_layout.addWidget(self.btn_config_musculos)
    self.tools_bar_layout.addWidget(self.btn_notas)
    self.tools_bar_layout.addWidget(self.btn_fotometria)
    self.tools_bar_layout.addWidget(self.btn_pantalla_externa)
    self.tools_bar_layout.addWidget(self.btn_mute_speaking)
    self.tools_bar_layout.addWidget(self.btn_marcar_invalido)
    self.tools_bar_layout.addStretch(1)
    self.tools_bar_layout.addWidget(self.btn_terminal)

    # Controles del Panel 3 (Trigger, SNR y Espectrograma)
    self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll (Armar Trigger)")
    self.chk_autoscroll.setChecked(True)

    self.chk_trigger = QtWidgets.QCheckBox("Habilitar Trigger")
    self.label_trig_chan = QtWidgets.QLabel("Canal:")
    self.cmb_trig_chan = QtWidgets.QComboBox()

    self.label_trig_level = QtWidgets.QLabel("Nivel (uV):")
    self.spin_trig_level = SpinBox(value=1000.0, step=100.0, dec=True, minStep=1.0, bounds=(-20000.0, 20000.0))
    self.spin_trig_level.setFixedWidth(75)
    self.spin_trig_level.setFixedHeight(26)

    self.label_trig_edge = QtWidgets.QLabel("Flanco:")
    self.cmb_trig_edge = QtWidgets.QComboBox()
    self.cmb_trig_edge.addItems(["Subida ", "Bajada "])

    self.label_peak_th = QtWidgets.QLabel("Umbral Picos SNR (uV):")
    self.spin_peak_th = SpinBox(value=200.0, step=50.0, bounds=(0.0, 20000.0), dec=True)
    self.spin_peak_th.setFixedWidth(75)
    self.spin_peak_th.setFixedHeight(26)

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
    self.trig_spec_layout.addWidget(self.chk_escala_cvm, 4, 0, 1, 2)
    self.trig_spec_layout.addWidget(self.chk_clamp_500uv, 4, 2, 1, 2)

    self.measure_labels = []
    self.measure_layout.addStretch(1)

    # Iniciar chequeo periodico de estado de hardware DAQ (cada 2.5s)
    self.daq_check_timer = QtCore.QTimer(self)
    self.daq_check_timer.timeout.connect(self.verificar_conexion_daq)
    self.daq_check_timer.start(2500)
    QtCore.QTimer.singleShot(300, self.verificar_conexion_daq)

  def _setup_notas_panel(self):
    """Configura el panel Cyberpunk Pro de notas del experimento y trazabilidad."""
    self.notas_groupbox = QtWidgets.QGroupBox("Notas del Experimento y Trazabilidad")
    self.notas_groupbox.setStyleSheet("""
      QGroupBox {
        border: 1.5px solid #00FFFF;
        border-radius: 5px;
        margin-top: 1.2ex;
        font-family: 'Courier New', monospace;
        color: #00FFFF;
        background-color: #060911;
        font-weight: bold;
      }
      QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: #00FFFF;
      }
    """)
    lyt_notas = QtWidgets.QVBoxLayout(self.notas_groupbox)
    lyt_notas.setContentsMargins(8, 8, 8, 6)
    lyt_notas.setSpacing(5)

    self.txt_notas_live = QtWidgets.QPlainTextEdit()
    self.txt_notas_live.setFixedHeight(68)
    self.txt_notas_live.setPlaceholderText("Registre anotaciones del experimento, estado de electrodos, gel o eventos...")
    self.txt_notas_live.setStyleSheet("""
      QPlainTextEdit {
        background-color: #0B0F19;
        color: #00FFCC;
        border: 1px solid #00FFFF;
        border-radius: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        padding: 4px;
      }
      QPlainTextEdit:focus {
        border: 1.5px solid #39FF14;
      }
    """)
    lyt_notas.addWidget(self.txt_notas_live)

    btn_bar = QtWidgets.QHBoxLayout()
    btn_bar.setContentsMargins(0, 0, 0, 0)
    btn_bar.setSpacing(6)

    btn_style_tmpl = """
      QPushButton {
        background-color: #0B0F19;
        color: #39FF14;
        border: 1px solid #39FF14;
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
      }
      QPushButton:hover {
        background-color: #39FF14;
        color: #050508;
      }
    """
    btn_style_save = """
      QPushButton {
        background-color: #002B20;
        color: #00FFCC;
        border: 1.5px solid #00FFCC;
        border-radius: 3px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: bold;
        font-family: 'Courier New', monospace;
      }
      QPushButton:hover {
        background-color: #00FFCC;
        color: #050508;
      }
    """

    self.btn_tmpl_electrodos = QtWidgets.QPushButton("+ Electrodos")
    self.btn_tmpl_electrodos.setStyleSheet(btn_style_tmpl)
    self.btn_tmpl_electrodos.clicked.connect(
      lambda: self._insertar_plantilla_nota("[ELECTRODOS] Colocación: Ch0=Digástrico, Ch1=Cigomático, Ch2=Orbicular. Tierra=Oreja/Mastoides.")
    )

    self.btn_tmpl_piel = QtWidgets.QPushButton("+ Piel/Gel")
    self.btn_tmpl_piel.setStyleSheet(btn_style_tmpl)
    self.btn_tmpl_piel.clicked.connect(
      lambda: self._insertar_plantilla_nota("[PIEL/GEL] Preparación: Limpieza con alcohol isopropílico. Gel conductor Ten20 aplicado. Impedancia baja.")
    )

    self.btn_tmpl_confusion = QtWidgets.QPushButton("+ Confusion Vocales")
    self.btn_tmpl_confusion.setStyleSheet(btn_style_tmpl)
    self.btn_tmpl_confusion.clicked.connect(
      lambda: self._insertar_plantilla_nota("[CONFUSION VOCALES] Fonación: Monitorear posible confusión entre vocales cerradas /o/ vs /u/ o sonrisas /e/ vs /i/.")
    )

    self.btn_tmpl_ruido = QtWidgets.QPushButton("+ Ruido 50Hz")
    self.btn_tmpl_ruido.setStyleSheet(btn_style_tmpl)
    self.btn_tmpl_ruido.clicked.connect(
      lambda: self._insertar_plantilla_nota("[RUIDO 50Hz] Monitoreo: Notch activo. Verificar contacto de tierra y cables apantallados.")
    )

    self.btn_actualizar_notas = QtWidgets.QPushButton("Actualizar Notas en Ultima Medicion")
    self.btn_actualizar_notas.setStyleSheet(btn_style_save)
    self.btn_actualizar_notas.clicked.connect(self.actualizar_notas_ultima_medicion)

    btn_bar.addWidget(self.btn_tmpl_electrodos)
    btn_bar.addWidget(self.btn_tmpl_piel)
    btn_bar.addWidget(self.btn_tmpl_confusion)
    btn_bar.addWidget(self.btn_tmpl_ruido)
    btn_bar.addStretch(1)
    btn_bar.addWidget(self.btn_actualizar_notas)

    lyt_notas.addLayout(btn_bar)
    self.notas_groupbox.setVisible(False)

  def _insertar_plantilla_nota(self, texto_plantilla: str):
    if hasattr(self, 'txt_notas_live'):
      cursor = self.txt_notas_live.textCursor()
      cursor.movePosition(QtGui.QTextCursor.End)
      actual = self.txt_notas_live.toPlainText()
      if actual and not actual.endswith("\n"):
        cursor.insertText("\n")
      cursor.insertText(texto_plantilla + "\n")
      self.txt_notas_live.setTextCursor(cursor)
      self.txt_notas_live.ensureCursorVisible()

  def toggle_notas_panel(self):
    if hasattr(self, 'notas_groupbox'):
      vis = not self.notas_groupbox.isVisible()
      self.notas_groupbox.setVisible(vis)
      if vis:
        self.btn_notas.setStyleSheet(
          "QPushButton { background-color: #00FFFF; color: #050508; border: 1.5px solid #FFFFFF; border-radius: 5px; font-size: 11px; font-weight: 900; padding: 7px 10px; font-family: 'Courier New', monospace; }"
        )
      else:
        self.btn_notas.setStyleSheet(BTN_CYBER_TOOL)

  def _reload_channel_muscles(self):
    self.nombres_musculos = []
    canales_conf = self.config_mgr.get("canales") or {}
    for i in range(16):
      key = f"Canal {i}"
      musc = canales_conf.get(key, {}).get("musculo", f"Canal {i}")
      self.nombres_musculos.append(musc)
    if self.is_acquiring:
      self.setup_gui_for_channels()

  def abrir_modulo_fotometria(self):
    import subprocess
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibracion_espacial_electrodos.py")
    if not os.path.exists(target):
      target = os.path.join(get_project_root(), "acquisition", "calibracion_espacial_electrodos.py")
    try:
      if getattr(sys, 'frozen', False):
        subprocess.Popen([sys.executable, "--run-script", target])
      else:
        subprocess.Popen([sys.executable, target])
      print(f"[Fotometria] Modulo lanzado: {target}")
    except Exception as e:
      print(f"[Error] No se pudo lanzar calibracion espacial: {e}")

  def abrir_asistente_fotometria(self):
    """Abre el Asistente de Fotometria Facial 3D y Triangulacion de Electrodos."""
    if getattr(self, 'photometry_mgr', None) is None:
      QtWidgets.QMessageBox.warning(
        self,
        "Fotometria No Disponible",
        "El modulo de fotometria o la camara web no se encuentran disponibles.",
      )
      return
    sujeto = getattr(self, "autoforge_sujeto", "") or getattr(self, "autoforge_base_name", "") or "Sujeto_0"
    prueba = getattr(self, "autoforge_prueba", "") or "Autoforge"
    canales_activos = [i for i, chk in enumerate(getattr(self, "channel_checkboxes", [])) if chk.isChecked()]
    if not canales_activos:
      canales_activos = [i for i in range(min(4, self.NUM_CANALES))]
    nombres_canales = [
      self.nombres_musculos[i % len(self.nombres_musculos)] if hasattr(self, 'nombres_musculos') and self.nombres_musculos else f"Canal {i}"
      for i in canales_activos
    ]
    try:
      self.photometry_mgr.open_alignment_dialog(
        parent_widget=self,
        nombres_musculos=nombres_canales,
        active_channel_indices=canales_activos,
        sujeto=sujeto,
        prueba=prueba,
      )
    except Exception as exc:
      print(f"[FOTOMETRIA-AVISO] No se pudo abrir el asistente de fotometría 3D: {exc}")

  def _construir_resumen_protocolo_dict(self, modo_ejecucion: str = None) -> dict:
    """
    Recopila ABSOLUTAMENTE TODA la configuración e información del protocolo y del experimento:
    hardware, fuente real (DAQ / Mic / Simulador), canales y ganancias, filtros (Notch, Pasa-Banda, RMS),
    metrónomo, modo Jitter, si se usó audio/speaking o estuvo muteado, historial de locuciones,
    fotometría/cámara, calibración CVM, ruido basal inicial/final y sesiones guardadas.
    """
    if modo_ejecucion is None:
      if getattr(self, 'is_autoforge_continuo', False):
        modo_ejecucion = "Secuencia Continua"
      elif getattr(self, 'is_autoforge_running', False):
        modo_ejecucion = "Autograbado Discreto"
      else:
        modo_ejecucion = "Grabación Manual"

    es_modo_prueba = bool(getattr(self, 'chk_modo_prueba', None) and self.chk_modo_prueba.isChecked())
    es_modo_mic = bool(getattr(self, 'chk_use_mic', None) and self.chk_use_mic.isChecked())
    if es_modo_mic:
      fuente_adq = "Micrófono PC"
    elif es_modo_prueba:
      fuente_adq = f"Modo Prueba / Simulador ({self.cmb_fuente_prueba.currentText() if hasattr(self, 'cmb_fuente_prueba') else 'Simulado'})"
    elif getattr(self, 'chk_zmq_bridge', None) and self.chk_zmq_bridge.isChecked():
      ep_txt = self.edit_zmq_host.text().strip() if hasattr(self, 'edit_zmq_host') else 'auto:5555'
      fuente_adq = f"Puente VM ZeroMQ NI-DAQ ({self.cmb_device.currentText() if hasattr(self, 'cmb_device') else 'Dev1'} @ {ep_txt})"
    else:
      fuente_adq = f"Tarjeta Física NI-DAQ ({self.cmb_device.currentText() if hasattr(self, 'cmb_device') else 'Dev1'})"

    vcp = getattr(self, 'voice_cue_player', None)
    speaking_muted = bool(
      (getattr(self, 'chk_mute_speaking', None) and self.chk_mute_speaking.isChecked())
      or (vcp is not None and (vcp.muted or not vcp.enabled or vcp.speaking_mode == "mute"))
    )
    speaking_mode = getattr(vcp, 'speaking_mode', 'fases_y_continua') if vcp is not None else 'desconocido'
    spoken_log = list(getattr(self, 'spoken_audio_log', []) or (getattr(vcp, 'history', []) if vcp is not None else []))
    cues_emitidos = [c for c in spoken_log if c.get("emitido", False)]

    canales_detalle = []
    for i in range(int(getattr(self, 'NUM_CANALES', 0) or 0)):
      hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and i < len(self.active_channel_indices)) else i
      daq_ch = self.CANALES_DAQ[i] if (hasattr(self, 'CANALES_DAQ') and i < len(self.CANALES_DAQ)) else f"ai{hw_idx}"
      musc = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
      gain = float(self.factores_calibracion_hw[hw_idx]) if (hasattr(self, 'factores_calibracion_hw') and hw_idx < len(self.factores_calibracion_hw)) else 495.0
      canales_detalle.append({
        "indice_logico": i,
        "indice_fisico_hw": hw_idx,
        "canal_daq": daq_ch,
        "musculo": musc,
        "ganancia_calibracion": gain,
        "es_microfono": bool(self._is_mic_channel(i)),
      })

    bpm_val = int(self.spin_bpm.value()) if hasattr(self, 'spin_bpm') else 30
    subdiv_val = int(self.spin_metro_subdivs.value()) if hasattr(self, 'spin_metro_subdivs') else 4
    jitter_on = bool(hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked())
    jitter_pct = float(self.spin_metro_jitter_pct.value()) if hasattr(self, 'spin_metro_jitter_pct') else 0.0
    jitter_mode = str(self.cmb_metro_jitter_mode.currentText()) if hasattr(self, 'cmb_metro_jitter_mode') else "exponential_hazard"

    return {
      "version_software": str(__version__),
      "modo_ejecucion": str(modo_ejecucion),
      "sujeto": str(getattr(self, 'autoforge_sujeto', '') or 'Sujeto'),
      "prueba": str(getattr(self, 'autoforge_prueba', '') or 'Prueba'),
      "hardware_adquisicion": {
        "fuente_datos": fuente_adq,
        "modo_prueba_activo": es_modo_prueba,
        "fuente_simulador": str(self.cmb_fuente_prueba.currentText()) if (es_modo_prueba and hasattr(self, 'cmb_fuente_prueba')) else None,
        "frecuencia_simulador_hz": int(self.spin_test_freq.value()) if (es_modo_prueba and hasattr(self, 'spin_test_freq')) else None,
        "microfono_pc_activo": es_modo_mic,
        "dispositivo_daq": str(self.cmb_device.currentText()) if hasattr(self, 'cmb_device') else "Dev1",
        "configuracion_terminal": str(self.cmb_terminal_config.currentText()) if hasattr(self, 'cmb_terminal_config') else "RSE",
        "sample_rate_hz": int(getattr(self, 'SAMPLE_RATE', 2000) or 2000),
        "plot_duration_s": float(getattr(self, 'PLOT_DURATION_S', 10) or 10),
        "preroll_s": float(self.spin_preroll.value()) if hasattr(self, 'spin_preroll') else 0.0,
        "offset_muestras_ancla": int(getattr(self, '_offset_muestras_ancla', 0) or 0),
        "num_canales_activos": int(getattr(self, 'NUM_CANALES', 0) or 0),
        "canales": canales_detalle,
      },
      "filtros_y_procesamiento": {
        "notch_50hz_activo": bool(hasattr(self, 'chk_notch_enable') and self.chk_notch_enable.isChecked()),
        "notch_q": float(self.spin_notch_q.value()) if hasattr(self, 'spin_notch_q') else 30.0,
        "notch_armonicos_comb": bool(hasattr(self, 'chk_notch_comb') and self.chk_notch_comb.isChecked()),
        "pasabanda_activo": bool(hasattr(self, 'chk_filter_enable') and self.chk_filter_enable.isChecked()),
        "pasabanda_corte_inferior_hz": float(self.spin_low_cut.value()) if hasattr(self, 'spin_low_cut') else 20.0,
        "pasabanda_corte_superior_hz": float(self.spin_high_cut.value()) if hasattr(self, 'spin_high_cut') else 500.0,
        "pasabanda_orden_butterworth": int(getattr(self, 'FILTER_ORDER', 4)),
        "envolvente_rms_activa": bool(hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()),
        "ventana_rms_ms": float(self.spin_rms_window.value()) if hasattr(self, 'spin_rms_window') else 50.0,
        "umbral_picos_snr_uV": float(self.spin_peak_th.value()) if hasattr(self, 'spin_peak_th') else 200.0,
        "trigger_activo": bool(hasattr(self, 'chk_trigger') and self.chk_trigger.isChecked()),
      },
      "metronomo_y_ritmo": {
        "metronomo_activo": bool(hasattr(self, 'chk_use_metronome') and self.chk_use_metronome.isChecked()),
        "bpm_nominal": bpm_val,
        "intervalo_base_tau0_s": round(60.0 / max(1, bpm_val), 4),
        "subdivisiones_compas": subdiv_val,
        "subdivisiones_efectivas_en_fonacion": 1 if jitter_on else subdiv_val,
        "jitter_activo": jitter_on,
        "jitter_porcentaje": jitter_pct if jitter_on else 0.0,
        "jitter_modo": jitter_mode if jitter_on else "ninguno",
        "duracion_ruido_basal_configurada_s": float(self.spin_noise_duration.value()) if hasattr(self, 'spin_noise_duration') else 3.0,
      },
      "audio_y_speaking": {
        "se_uso_audio_speaking": bool(not speaking_muted and len(cues_emitidos) > 0),
        "speaking_habilitado_en_config": bool(not speaking_muted),
        "speaking_muteado": speaking_muted,
        "modo_locucion": speaking_mode,
        "beeps_metronomo_activos": bool(hasattr(self, 'chk_use_metronome') and self.chk_use_metronome.isChecked()),
        "total_comandos_voz_emitidos": len(cues_emitidos),
        "total_comandos_voz_registrados": len(spoken_log),
        "historial_comandos_voz": spoken_log,
      },
      "fotometria_y_camara": {
        "modulo_disponible": bool(getattr(self, 'photometry_mgr', None) is not None),
        "asistente_confirmado_en_sesion": bool(getattr(self, '_photometry_session_active', False)),
      },
      "protocolo_secuencia": {
        "orden_vocales_base": list(getattr(self, 'autoforge_words', []) or []),
        "secuencia_balanceada_debruijn_activa": bool(getattr(self, 'autoforge_use_debruijn', False)),
        "secuencia_completa_ejecutada": list(getattr(self, 'autoforge_full_sequence', []) or []),
        "repeticiones_configuradas": int(getattr(self, 'autoforge_target_reps', 1) or 1),
        "calibracion_cvm_solicitada": bool(getattr(self, 'autoforge_do_cvm', False)),
        "calibracion_cvm_completada": bool(getattr(self, 'cvm_completed', False)),
        "ruido_final_completado": bool(getattr(self, 'final_noise_completed', False)),
        "pulsos_marcados_invalidos_manual": sorted(list(getattr(self, 'invalid_pulses_set', set()) or set())),
        "semaforo_contacto_electrodos": getattr(self, 'diagnostico_contacto_electrodos', {}),
        "resumen_dataset_y_fatiga": {
          k: v for k, v in (getattr(self, 'last_dataset_summary', {}) or {}).items() if k != "segmentos"
        },
        "directorios_sesion": list(getattr(self, 'autoforge_session_dirs', []) or []),
      },
    }

  def _formatear_informe_protocolo_completo_txt(self, metadata_dict: dict = None, modo_ejecucion: str = None) -> str:
    """
    Construye un informe en texto plano (.txt) exhaustivo y legible con ABSOLUTAMENTE TODA
    la información del experimento, protocolo, audio/speaking, hardware, filtros, ruido,
    CVM, pulsos del metrónomo, sincronización y notas.
    """
    meta = metadata_dict or {}
    proto = meta.get("protocolo_experimento_completo") or self._construir_resumen_protocolo_dict(modo_ejecucion=modo_ejecucion)
    hw = proto.get("hardware_adquisicion", {})
    flt = proto.get("filtros_y_procesamiento", {})
    met = proto.get("metronomo_y_ritmo", {})
    aud = proto.get("audio_y_speaking", {})
    fot = proto.get("fotometria_y_camara", {})
    sec = proto.get("protocolo_secuencia", {})

    lines = [
      "=" * 88,
      f"INFORME COMPLETO DEL PROTOCOLO Y EXPERIMENTO — ÑANDÚ LSD v{proto.get('version_software', __version__)}",
      "=" * 88,
      f"Fecha y Hora Guardado : {meta.get('measurement_date', datetime.now().isoformat())}",
      f"Modo de Ejecución     : {proto.get('modo_ejecucion', 'N/A')}",
      f"Sujeto                : {meta.get('sujeto', proto.get('sujeto', 'N/A'))}",
      f"Prueba                : {meta.get('prueba', proto.get('prueba', 'N/A'))}",
      f"Letra / Etiqueta      : {meta.get('letra', 'N/A')}",
      f"Ancla t0_epoch        : {meta.get('t0_epoch', 'N/A')}",
      f"Ancla t0_perf_counter : {meta.get('t0_perf_counter', 'N/A')}",
      "",
      "-" * 88,
      "1. CONFIGURACIÓN DE AUDIO / SPEAKING Y SONIDO",
      "-" * 88,
      f"¿Se usó Audio Speaking?      : {'SÍ (Voz activa)' if aud.get('se_uso_audio_speaking') else 'NO (Muteado o sin locución)'}",
      f"Estado Botón Speaking        : {'MUTEADO (SPEAKING: MUTE)' if aud.get('speaking_muteado') else 'ACTIVO (SPEAKING: ON)'}",
      f"Modo de Locución             : {aud.get('modo_locucion', 'N/A')}",
      f"Beeps del Metrónomo Activos  : {'SÍ' if aud.get('beeps_metronomo_activos') else 'NO'}",
      f"Comandos de Voz Emitidos     : {aud.get('total_comandos_voz_emitidos', 0)} de {aud.get('total_comandos_voz_registrados', 0)} registrados",
    ]
    hist_voz = aud.get("historial_comandos_voz", []) or []
    if hist_voz:
      lines.append("Detalle cronológico de locuciones (Speaking):")
      for idx_v, hv in enumerate(hist_voz, 1):
        est_v = "EMITIDO" if hv.get("emitido") else "MUTEADO/OMITIDO"
        lines.append(
          f"  [{idx_v:02d}] t_rel={hv.get('t_rel_s', 0.0):+7.3f}s | Fase={hv.get('fase', '-'):26s} | "
          f"Clave={hv.get('cue_key', '-'):14s} | Texto='{hv.get('texto', '')}' ({est_v})"
        )

    lines.extend([
      "",
      "-" * 88,
      "2. HARDWARE, FUENTE DE DATOS Y CANALES ACTIVOS",
      "-" * 88,
      f"Fuente de Datos              : {hw.get('fuente_datos', 'N/A')}",
      f"Modo Prueba Activo           : {'SÍ' if hw.get('modo_prueba_activo') else 'NO'}",
      f"Micrófono PC Activo          : {'SÍ' if hw.get('microfono_pc_activo') else 'NO'}",
      f"Dispositivo DAQ / Terminal   : {hw.get('dispositivo_daq', 'Dev1')} | Terminal: {hw.get('configuracion_terminal', 'RSE')}",
      f"Frecuencia de Muestreo (SR)  : {hw.get('sample_rate_hz', 2000)} Hz",
      f"Pre-Roll / Offset Ancla      : {hw.get('preroll_s', 0.0)} s ({hw.get('offset_muestras_ancla', 0)} muestras previas en WAV)",
      f"Canales Activos ({hw.get('num_canales_activos', 0)})         :",
    ])
    for ch_d in hw.get("canales", []):
      tipo_ch = "MICRÓFONO" if ch_d.get("es_microfono") else "EMG MUSCULAR"
      lines.append(
        f"  - Canal lógico {ch_d.get('indice_logico')}: {ch_d.get('canal_daq')} (ai{ch_d.get('indice_fisico_hw')}) | "
        f"Músculo: {ch_d.get('musculo')} | Ganancia: {ch_d.get('ganancia_calibracion')} | Tipo: {tipo_ch}"
      )

    lines.extend([
      "",
      "-" * 88,
      "3. FILTROS DIGITALES Y PROCESAMIENTO EN TIEMPO REAL",
      "-" * 88,
      f"Filtro Notch 50 Hz           : {'ACTIVO' if flt.get('notch_50hz_activo') else 'DESACTIVADO (Señal cruda sin Notch)'} (Q={flt.get('notch_q')}, Comb={'SÍ' if flt.get('notch_armonicos_comb') else 'NO'})",
      f"Filtro Pasa-Banda Butterworth: {'ACTIVO' if flt.get('pasabanda_activo') else 'DESACTIVADO'} ({flt.get('pasabanda_corte_inferior_hz')} Hz - {flt.get('pasabanda_corte_superior_hz')} Hz, Orden {flt.get('pasabanda_orden_butterworth')})",
      f"Envolvente RMS               : {'ACTIVA' if flt.get('envolvente_rms_activa') else 'DESACTIVADA'} (Ventana={flt.get('ventana_rms_ms')} ms)",
      f"Umbral Picos SNR             : {flt.get('umbral_picos_snr_uV')} µV | Trigger Activo: {'SÍ' if flt.get('trigger_activo') else 'NO'}",
      "",
      "-" * 88,
      "4. METRÓNOMO, RITMO Y MODO JITTER ANTI-SINCRONIZACIÓN",
      "-" * 88,
      f"Metrónomo Activo             : {'SÍ' if met.get('metronomo_activo') else 'NO'}",
      f"BPM Nominal / Intervalo Base : {met.get('bpm_nominal')} BPM (tau_0 = {met.get('intervalo_base_tau0_s')} s)",
      f"Subdivisiones Configuradas   : {met.get('subdivisiones_compas')} (Efectivas en fonación: {met.get('subdivisiones_efectivas_en_fonacion')})",
      f"Modo Jitter Anti-Sync        : {'ACTIVO' if met.get('jitter_activo') else 'DESACTIVADO (Ritmo Iso-rítmico Fijo)'}",
      f"Parámetros Jitter            : +/- {met.get('jitter_porcentaje')}% | Perfil: {met.get('jitter_modo')}",
      f"Duración Ruido Basal Config. : {met.get('duracion_ruido_basal_configurada_s')} s",
      "",
      "-" * 88,
      "5. CÁMARA Y FOTOMETRÍA FACIAL 3D",
      "-" * 88,
      f"Módulo de Cámara Disponible  : {'SÍ' if fot.get('modulo_disponible') else 'NO'}",
      f"Montaje Fotométrico en Sesión: {'CONFIRMADO Y ACTIVO' if fot.get('asistente_confirmado_en_sesion') else 'OMITIDO / SIN CÁMARA'}",
    ])

    rb = meta.get("ruido_basal", {}) or {}
    rb_ini = rb.get("inicial", {}) or {}
    rb_fin = rb.get("final", {}) or {}
    lines.extend([
      "",
      "-" * 88,
      "6. RUIDO BASAL (INICIAL Y FINAL) Y RUIDO DE LÍNEA 50 HZ",
      "-" * 88,
      f"Ruido Inicial — Media (µV)   : {rb_ini.get('mean_uV', [])}",
      f"Ruido Inicial — Desv.Std (µV): {rb_ini.get('std_uV', [])}",
      f"Ruido Inicial — 50 Hz (µV)   : {rb_ini.get('line_noise_50hz_uV', [])}",
      f"Ruido Final   — Media (µV)   : {rb_fin.get('mean_uV', [])}",
      f"Ruido Final   — Desv.Std (µV): {rb_fin.get('std_uV', [])}",
      f"Ruido Final   — 50 Hz (µV)   : {rb_fin.get('line_noise_50hz_uV', [])}",
    ])

    cvm = meta.get("calibracion_cvm", {}) or {}
    lines.extend([
      "",
      "-" * 88,
      "7. CALIBRACIÓN DE CONTRACCIÓN MUSCULAR MÁXIMA (CVM)",
      "-" * 88,
      f"CVM Calibrado en Sesión      : {'SÍ' if cvm.get('calibrado') else 'NO'}",
      f"Supremo Global Pico / P95    : {cvm.get('supremo_global_uV', 0.0):.2f} µV | P95: {cvm.get('supremo_global_p95_uV', 0.0):.2f} µV | RMS: {cvm.get('supremo_global_rms_uV', 0.0):.2f} µV",
    ])

    lines.extend([
      "",
      "-" * 88,
      "8. SECUENCIA DE VOCALES (DE BRUIJN), FATIGA MUSCULAR (MDF) Y SINCRONIZACIÓN",
      "-" * 88,
      f"Orden Base de Vocales        : {' -> '.join(sec.get('orden_vocales_base', []) or ['N/A'])}",
      f"Balanceo De Bruijn Activo    : {'SÍ (Transiciones v_i -> v_j equilibradas)' if sec.get('secuencia_balanceada_debruijn_activa') else 'NO (Ciclo fijo)'}",
      f"Repeticiones / Ciclos        : {sec.get('repeticiones_configuradas', 1)}",
      f"Pulsos Inválidos (Manual)    : {sec.get('pulsos_marcados_invalidos_manual', [])}",
      f"Semáforo Contacto Electrodos : {'ALERTA: ' + ', '.join(sec.get('semaforo_contacto_electrodos', {}).get('canales_con_alerta', [])) if sec.get('semaforo_contacto_electrodos', {}).get('canales_con_alerta') else 'ÓPTIMO'}",
      f"Resumen Fatiga Muscular (MDF): {sec.get('resumen_dataset_y_fatiga', {}).get('fatiga_muscular_por_canal', {})}",
      f"Total Eventos Vocales        : {len(meta.get('eventos_vocales_metronomo', []) or [])}",
    ])
    ev_vocales = meta.get("eventos_vocales_metronomo", []) or []
    for ev in ev_vocales:
      lines.append(
        f"  Pulso {int(ev.get('pulso', 0)):03d} | Ciclo {int(ev.get('ciclo', 1)):02d} | "
        f"Vocal: /{ev.get('vocal', '')}/ | t_rel={float(ev.get('t_rel_s', 0.0)):8.3f}s | "
        f"sample={ev.get('sample_idx', 0)} | tau={float(ev.get('tau_s', 0.0)):.3f}s | "
        f"jitter={'SÍ' if ev.get('jitter_enabled') else 'NO'}"
      )

    sync_list = meta.get("sincronizacion_sujeto_vs_metronomo", []) or []
    if sync_list:
      lines.append("\nEvaluación de Sincronización Fonatoria (Onset Sujeto vs. Pulso Metrónomo):")
      for s_ev in sync_list:
        lines.append(
          f"  Pulso {int(s_ev.get('pulso_asociado', 0)):03d} (/{s_ev.get('vocal', '')}/) | "
          f"t_beat={float(s_ev.get('t_beat_s', 0.0)):7.3f}s | t_onset={float(s_ev.get('t_onset_s', 0.0)):7.3f}s | "
          f"Delta={float(s_ev.get('delta_sync_ms', 0.0)):+7.1f} ms [{s_ev.get('clasificacion', '')}]"
        )

    if sec.get("directorios_sesion"):
      lines.append("\nCarpetas generadas en la sesión:")
      for d_s in sec.get("directorios_sesion", []):
        lines.append(f"  - {d_s}")

    notas_txt = meta.get("notas_experimento") or (self.txt_notas_live.toPlainText().strip() if hasattr(self, 'txt_notas_live') else "")
    lines.extend([
      "",
      "-" * 88,
      "9. NOTAS DEL EXPERIMENTO",
      "-" * 88,
      notas_txt if notas_txt else "(Sin notas adicionales escritas por el operador)",
      "=" * 88,
    ])
    return "\n".join(lines) + "\n"

  def persistir_notas_experimento(self, target_dir: str, metadata_base: dict = None, texto_notas: str = None):
    """
    Persiste las notas del experimento y ABSOLUTAMENTE TODA la información del protocolo
    tanto en metadata.json raiz como en los metadata.json de cada canal (canal_*/metadata.json),
    y genera notas_experimento.txt y protocolo_experimento_completo.txt.
    """
    if not target_dir or not os.path.exists(target_dir):
      return
    self.last_saved_measurement_dir = str(target_dir)

    if texto_notas is None:
      if hasattr(self, 'txt_notas_live'):
        texto_notas = self.txt_notas_live.toPlainText().strip()
      else:
        texto_notas = ""

    # 1. metadata.json raiz
    root_meta_path = os.path.join(target_dir, "metadata.json")
    root_meta = {}
    if os.path.isfile(root_meta_path):
      try:
        with open(root_meta_path, 'r', encoding='utf-8') as f:
          loaded = json.load(f)
          if isinstance(loaded, dict):
            root_meta = loaded
      except Exception:
        root_meta = {}
    elif metadata_base is not None and isinstance(metadata_base, dict):
      root_meta = metadata_base.copy()

    proto_completo = self._construir_resumen_protocolo_dict()
    root_meta["notas_experimento"] = texto_notas
    if texto_notas:
      root_meta["comentario"] = texto_notas
    root_meta["protocolo_experimento_completo"] = proto_completo
    root_meta["audio_y_speaking"] = proto_completo.get("audio_y_speaking", {})
    if hasattr(self, 'last_separability_metrics') and self.last_separability_metrics is not None:
      root_meta["diagnostico_separabilidad_vocales"] = self.last_separability_metrics

    try:
      with open(root_meta_path, 'w', encoding='utf-8') as f:
        json.dump(root_meta, f, indent=4, ensure_ascii=False)
      print(f"[Notas] metadata.json raiz actualizado en: {root_meta_path}")
    except Exception as e:
      print(f"[Aviso] No se pudo escribir metadata.json raiz: {e}")

    # 2. canal_*/metadata.json
    try:
      for item in os.listdir(target_dir):
        sub_dir = os.path.join(target_dir, item)
        if os.path.isdir(sub_dir) and item.startswith("canal_"):
          ch_meta_path = os.path.join(sub_dir, "metadata.json")
          ch_meta = {}
          if os.path.isfile(ch_meta_path):
            try:
              with open(ch_meta_path, 'r', encoding='utf-8') as f:
                ch_loaded = json.load(f)
                if isinstance(ch_loaded, dict):
                  ch_meta = ch_loaded
            except Exception:
              ch_meta = {}
          ch_meta["notas_experimento"] = texto_notas
          if texto_notas:
            ch_meta["comentario"] = texto_notas
          ch_meta["protocolo_experimento_completo"] = proto_completo
          ch_meta["audio_y_speaking"] = proto_completo.get("audio_y_speaking", {})
          if hasattr(self, 'last_separability_metrics') and self.last_separability_metrics is not None:
            ch_meta["diagnostico_separabilidad_vocales"] = self.last_separability_metrics
          try:
            with open(ch_meta_path, 'w', encoding='utf-8') as f:
              json.dump(ch_meta, f, indent=4, ensure_ascii=False)
          except Exception:
            pass
    except Exception:
      pass

    # 3. Generar informe completo en protocolo_experimento_completo.txt y notas_experimento.txt
    informe_txt = self._formatear_informe_protocolo_completo_txt(metadata_dict=root_meta)
    proto_txt_path = os.path.join(target_dir, "protocolo_experimento_completo.txt")
    try:
      with open(proto_txt_path, 'w', encoding='utf-8') as f:
        f.write(informe_txt)
      print(f"[Protocolo] protocolo_experimento_completo.txt generado en: {proto_txt_path}")
    except Exception as e:
      print(f"[Aviso] No se pudo escribir protocolo_experimento_completo.txt: {e}")

    txt_path = os.path.join(target_dir, "notas_experimento.txt")
    try:
      with open(txt_path, 'w', encoding='utf-8') as f:
        if texto_notas:
          f.write("NOTAS DEL OPERADOR:\n" + texto_notas + "\n\n")
        f.write(informe_txt)
      print(f"[Notas] notas_experimento.txt generado en: {txt_path}")
    except Exception as e:
      print(f"[Aviso] No se pudo escribir notas_experimento.txt: {e}")

  def actualizar_notas_ultima_medicion(self):
    target = getattr(self, 'last_saved_measurement_dir', None)
    if not target or not os.path.isdir(target):
      try:
        if getattr(sys, 'frozen', False):
          root_dir = os.path.dirname(os.path.abspath(sys.executable))
          if os.path.basename(root_dir) == "_internal":
            root_dir = os.path.dirname(root_dir)
        else:
          root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_dir = os.path.join(root_dir, "base_de_datos_electrodos")
        if os.path.isdir(db_dir):
          fechas = sorted([d for d in os.listdir(db_dir) if os.path.isdir(os.path.join(db_dir, d))], reverse=True)
          if fechas:
            fecha_dir = os.path.join(db_dir, fechas[0])
            meds = sorted([d for d in os.listdir(fecha_dir) if os.path.isdir(os.path.join(fecha_dir, d))],
                          key=lambda m: os.path.getmtime(os.path.join(fecha_dir, m)), reverse=True)
            if meds:
              target = os.path.join(fecha_dir, meds[0])
      except Exception:
        target = None

    if target and os.path.isdir(target):
      self.persistir_notas_experimento(target)
      if hasattr(self, 'terminal_dialog'):
        self.terminal_dialog.append_log(f"[OK] Notas actualizadas en ultima medicion: {os.path.basename(target)}\n")
      print(f"[OK] Notas actualizadas en ultima medicion: {target}")
    else:
      print("[Aviso] No se encontro ninguna medicion previa guardada para actualizar notas.")

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

    self.viz_control_layout.addWidget(self.lbl_daq_status)
    self.viz_control_layout.addWidget(self.chk_show_noise)
    self.viz_control_layout.addWidget(self.chk_show_peaks)
    self.viz_control_layout.addWidget(self.chk_show_detection_window)
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

    # Region sombreada de la ventana de deteccion fonatoria 0.5 tau
    brush_det = pg.mkBrush(0, 255, 255, 38)
    pen_det = pg.mkPen(color='#00FFFF', width=1.5, style=QtCore.Qt.DashLine)
    self.detection_window_region = pg.LinearRegionItem(orientation='vertical', brush=brush_det, pen=pen_det, movable=False)
    self.detection_window_region.setZValue(-30)
    self.detection_window_region.hide()
    self.plot.addItem(self.detection_window_region)

    self.detection_window_label = pg.TextItem(anchor=(0.5, 1.0), color=(0, 255, 255), fill=(5, 5, 15, 200))
    self.detection_window_label.setZValue(50)
    self.detection_window_label.hide()
    self.plot.addItem(self.detection_window_label)

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
    min_config_h = 180 if getattr(self, 'is_compact_screen', False) else 195
    self.config_groupbox.setMinimumHeight(min_config_h)
    self.config_stack.setMinimumHeight(min_config_h)
    self.config_stack.setMaximumHeight(220)
    self.filter_groupbox.setMinimumHeight(min_config_h)
    self.filter_groupbox.setMaximumHeight(220)
    self.config_stack.addWidget(self.config_groupbox)

    self.empty_recording_widget = QtWidgets.QWidget()
    self.empty_recording_widget.setStyleSheet("background-color: #050505;")

    self.lbl_recording_space = QtWidgets.QLabel("MODO GRABACION ACTIVO", self.empty_recording_widget)
    self.lbl_recording_space.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_recording_space.setWordWrap(True)
    self.lbl_recording_space.setStyleSheet(
      "background-color: #050510; color: #00FFCC; font-family: 'Courier New', monospace; "
      "border: 2px solid #00FFFF; border-radius: 6px; padding: 4px;"
    )

    self.empty_recording_layout = QtWidgets.QHBoxLayout(self.empty_recording_widget)
    self.empty_recording_layout.setContentsMargins(0, 0, 0, 0)
    self.empty_recording_layout.addWidget(self.lbl_recording_space)
    self.config_stack.addWidget(self.empty_recording_widget)

    # --- NUEVO: Medidor PCA / Separabilidad en Tiempo Real (ocupa el espacio libre del deck superior) ---
    self.pca_live_groupbox = QtWidgets.QGroupBox("PCA 2D en Vivo (Vocales)")
    self.pca_live_groupbox.setMinimumHeight(min_config_h)
    self.pca_live_groupbox.setMaximumHeight(220)
    self.pca_live_groupbox.setMinimumWidth(255)
    self.pca_live_groupbox.setStyleSheet("""
      QGroupBox {
        border: 1.5px solid #c77dff;
        border-radius: 6px;
        margin-top: 1.5ex;
        font-family: 'Courier New', monospace;
        color: #c77dff;
        background-color: #050508;
        font-weight: bold;
      }
      QGroupBox::title {
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: #c77dff;
      }
    """)
    pca_lyt = QtWidgets.QVBoxLayout(self.pca_live_groupbox)
    pca_lyt.setContentsMargins(5, 10, 5, 4)
    pca_lyt.setSpacing(2)

    self.pca_live_plot = pg.PlotWidget()
    self.pca_live_plot.setBackground('#050508')
    self.pca_live_plot.showGrid(x=True, y=True, alpha=0.22)
    self.pca_live_plot.setMouseEnabled(x=False, y=False)
    self.pca_live_plot.hideButtons()
    self.pca_live_plot.getAxis('bottom').setStyle(showValues=False)
    self.pca_live_plot.getAxis('left').setStyle(showValues=False)
    pca_lyt.addWidget(self.pca_live_plot, stretch=1)

    self.lbl_pca_legend = QtWidgets.QLabel(
      "<span style='color:#FF3355;'>●A</span> "
      "<span style='color:#00B4D8;'>●E</span> "
      "<span style='color:#39FF14;'>●I</span> "
      "<span style='color:#C77DFF;'>●O</span> "
      "<span style='color:#FFD166;'>●U</span>"
    )
    self.lbl_pca_legend.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_pca_legend.setStyleSheet("font-size: 10px; font-weight: bold; background: transparent;")
    pca_lyt.addWidget(self.lbl_pca_legend)

    pca_bottom_row = QtWidgets.QHBoxLayout()
    pca_bottom_row.setSpacing(3)
    self.lbl_pca_live_metrics = QtWidgets.QLabel("Acc:--% Sil:-- Sep:--% N=0")
    self.lbl_pca_live_metrics.setStyleSheet("color: #00ffcc; font-size: 9.5px; font-weight: bold;")
    self.btn_reset_pca_live = QtWidgets.QPushButton("R")
    self.btn_reset_pca_live.setFixedWidth(20)
    self.btn_reset_pca_live.setFixedHeight(17)
    self.btn_reset_pca_live.setToolTip("Reiniciar acumulador del medidor PCA en vivo")
    self.btn_reset_pca_live.setStyleSheet("background-color:#1a1a2e; color:#c77dff; border:1px solid #c77dff; font-size:9px; padding:0px;")
    self.btn_reset_pca_live.clicked.connect(self.reset_medidor_pca_vivo)
    pca_bottom_row.addWidget(self.lbl_pca_live_metrics, stretch=1)
    pca_bottom_row.addWidget(self.btn_reset_pca_live)
    pca_lyt.addLayout(pca_bottom_row)

    # Contenedor Superior: [Config_Stack (Hardware/Trigger o Vocal)] + [Filtros fijos] + [PCA Vivo] + [Metronomo fijo]
    self.top_deck_widget = QtWidgets.QWidget()
    self.top_deck_layout = QtWidgets.QHBoxLayout(self.top_deck_widget)
    self.top_deck_layout.setContentsMargins(0, 0, 0, 0)
    self.top_deck_layout.setSpacing(6)
    self.top_deck_layout.addWidget(self.config_stack, stretch=6)
    self.top_deck_layout.addWidget(self.filter_groupbox, stretch=4)
    self.top_deck_layout.addWidget(self.pca_live_groupbox, stretch=2)
    self.top_deck_layout.addWidget(self.right_side_panel, stretch=1)

    self.main_layout.addWidget(self.transport_frame)    # Fila 1: Transporte + Estado DAQ + Consola
    self.main_layout.addWidget(self.top_deck_widget)    # Fila 2: 3 Paneles Config / Palabras + Metronomo
    if hasattr(self, 'notas_groupbox'):
      self.main_layout.addWidget(self.notas_groupbox)   # Fila 2B: Panel Notas y Trazabilidad (conmutable)
    self.main_layout.addWidget(self.measure_widget)     # Fila 3: Mediciones y Ruido 50 Hz por Canal
    self.main_layout.addWidget(self.splitter, stretch=1)# Fila 4: Osciloscopio + Espectrograma

  def _connect_signals(self):
    
    # --- Conectar Señales (Botones) ---
    """
    Ejecuta la funcionalidad de _connect_signals.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.btn_start_acq.clicked.connect(self.on_start_acq_click)
    self.btn_record.clicked.connect(self.on_record_click)
    self.chk_autoscroll.clicked.connect(self.on_autoscroll_toggle)
    
    # --- Conectar Widgets del Trigger ---
    self.chk_trigger.clicked.connect(self.on_trigger_enable_toggle)
    self.trigger_line.sigPositionChanged.connect(self.on_trigger_line_moved)
    # --- CORREGIDO: Conectar aquí la señal para ocultar la etiqueta ---
    self.trigger_line.sigPositionChangeFinished.connect(lambda: self.trigger_label.hide())
    self.spin_trig_level.valueChanged.connect(self.on_trigger_level_changed)
    
    # --- NUEVO: Conectar Señales de Umbral SNR ---
    self.peak_th_line_pos.sigPositionChanged.connect(self.on_peak_th_line_pos_moved)
    self.peak_th_line_neg.sigPositionChanged.connect(self.on_peak_th_line_neg_moved)
    self.spin_peak_th.valueChanged.connect(self.on_peak_th_changed)

    # --- NUEVO: Conectar Checkbox de Modo Prueba ---
    self.chk_modo_prueba.toggled.connect(self._on_modo_prueba_toggled)
    self.cmb_terminal_config.currentIndexChanged.connect(self.on_terminal_mode_changed) # <-- NUEVO

    # --- NUEVO: Conectar Widgets del Espectrograma ---
    self.chk_spectrogram_enable.clicked.connect(self.on_spectrogram_enable_toggle)
    self.cmb_spectrogram_chan.currentIndexChanged.connect(self.on_spectrogram_channel_change)

    # --- NUEVO: Conectar Widgets del Filtro ---
    self.chk_notch_enable.clicked.connect(self.on_filter_settings_changed)
    self.chk_filter_enable.clicked.connect(self.on_filter_settings_changed)
    self.spin_low_cut.valueChanged.connect(self.on_filter_settings_changed)
    self.spin_high_cut.valueChanged.connect(self.on_filter_settings_changed)

    # --- Curvas del Plot ---
    self.curvas = []
    self.nombres_musculos = []
    canales_conf = self.config_mgr.get("canales") or {}
    
    canales_info = []
    for i in range(16):
      key = f"Canal {i}"
      musc = canales_conf.get(key, {}).get("musculo", f"Canal {i}")
      self.nombres_musculos.append(musc)
      canales_info.append({
          "idx": i,
          "musculo": musc,
          "color_hex": canales_conf.get(key, {}).get("color_hex"),
          "is_mic": (i == 3 or "mic" in musc.lower())
      })
      
    try:
      from utils.config_manager import get_unique_channel_colors
      self.colores_curvas = get_unique_channel_colors(canales_info)
    except Exception:
      self.colores_curvas = ["#ffaa00", "#39ff14", "#ffff00", "#ff0000"] + ["#00ffcc"] * 12

    # Conectar señal de vista previa de gráfico
    self.mostrar_preview_signal.connect(self.mostrar_preview_plot)
    self._actualizar_canales_activos(update_gui=False)

  def _safe_autoforge_single_shot(self, delay_ms, callback):
    """
    Ejecuta callback tras delay_ms sólo si la sesión actual de AutoForge sigue activa
    y coincide con el token de sesión _autoforge_token. Si AutoForge fue cancelado o reiniciado,
    invalida limpiamente la ejecución del timer pendiente.
    Usa _schedule_phase para sincronizarse con el botón PAUSAR.
    """
    current_token = getattr(self, '_autoforge_token', 0)
    def wrapper():
      if getattr(self, '_autoforge_token', 0) == current_token and getattr(self, 'is_autoforge_running', False):
        callback()
    self._schedule_phase(delay_ms, wrapper)

  def _is_mic_channel(self, ch_idx):
    """
    Determina si el canal activo ch_idx (0..NUM_CANALES-1) corresponde a un micrófono
    o si es un canal muscular EMG.
    """
    if ch_idx is None:
      return False
    from utils.config_manager import _is_generic_or_mic_channel
    if isinstance(ch_idx, int):
      hw_idx = self.active_channel_indices[ch_idx] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and ch_idx < len(self.active_channel_indices)) else ch_idx
      if (
        hw_idx == 3
        and getattr(self, 'chk_modo_prueba', None) is not None
        and self.chk_modo_prueba.isChecked()
        and hasattr(self, 'cmb_fuente_prueba')
        and "cande" in str(self.cmb_fuente_prueba.currentText()).lower()
      ):
        return True
      musc = ""
      if hasattr(self, 'nombres_musculos') and ch_idx < len(self.nombres_musculos):
        musc = self.nombres_musculos[ch_idx]
      elif hasattr(self, 'nombres_musculos_hw') and hw_idx < len(self.nombres_musculos_hw):
        musc = self.nombres_musculos_hw[hw_idx]
      daq_name = self.CANALES_DAQ[ch_idx] if (hasattr(self, 'CANALES_DAQ') and ch_idx < len(self.CANALES_DAQ)) else ""
      if "mic" in str(daq_name).lower() or "audio" in str(daq_name).lower():
        return True
      return _is_generic_or_mic_channel(musc, idx=hw_idx)
    else:
      return _is_generic_or_mic_channel(str(ch_idx))

  def _actualizar_canales_activos(self, update_gui=True):
    """
    Actualiza active_channel_indices, NUM_CANALES, CANALES_DAQ, nombres_musculos y colores_curvas
    basándose en los checkboxes seleccionados y la configuración de canales físicos en config_general.json.
    """
    device = self.cmb_device.currentText() if hasattr(self, 'cmb_device') else "Dev1"
    selected_channels = [chk.text() for chk in getattr(self, 'channel_checkboxes', []) if chk.isChecked()]

    self.CANALES_DAQ = [f"{device}/{ch}" for ch in selected_channels]
    self.active_channel_indices = []
    for ch in selected_channels:
      m = re.search(r'\d+', ch)
      self.active_channel_indices.append(int(m.group()) if m else 0)
    self.NUM_CANALES = len(self.active_channel_indices)

    # Recargar hardware channels
    if not hasattr(self, 'nombres_musculos_hw') or not self.nombres_musculos_hw:
      self.nombres_musculos_hw = [None] * 16
      self.factores_calibracion_hw = [495.0] * 16
    for hw in range(16):
      cfg = self.config_mgr.get_channel_config(hw)
      self.nombres_musculos_hw[hw] = cfg.get("musculo", f"Canal {hw}")
      self.factores_calibracion_hw[hw] = float(cfg.get("factor_calibracion", 495.0))

    # Mapear canales activos
    from utils.config_manager import get_unique_channel_colors
    active_info = []
    self.nombres_musculos = []
    for i, hw_idx in enumerate(self.active_channel_indices):
      musc = self.nombres_musculos_hw[hw_idx] if hw_idx < len(self.nombres_musculos_hw) else f"Canal {hw_idx}"
      self.nombres_musculos.append(musc)
      ch_cfg = self.config_mgr.get_channel_config(hw_idx)
      active_info.append({
        "idx": hw_idx,
        "musculo": musc,
        "color_hex": ch_cfg.get("color_hex"),
        "is_mic": self._is_mic_channel(i)
      })

    self.colores_curvas = get_unique_channel_colors(active_info)

    if update_gui and hasattr(self, 'plot'):
      self.setup_gui_for_channels()

  def _on_channel_checkbox_toggled(self, checked):
    if not getattr(self, 'is_acquiring', False):
      self._actualizar_canales_activos(update_gui=True)

  def abrir_dialogo_musculos(self):
    if getattr(self, 'is_acquiring', False):
      resp = QtWidgets.QMessageBox.question(
        self,
        "Detener Adquisición",
        "La configuración de músculos requiere detener la adquisición en curso.\n¿Desea detener la adquisición y continuar?",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        QtWidgets.QMessageBox.Yes
      )
      if resp != QtWidgets.QMessageBox.Yes:
        return
      self.on_start_acq_click() # Detiene la adquisición

    dialog = MuscleSelectionDialog(self)
    if dialog.exec() == QtWidgets.QDialog.Accepted:
      self.refrescar_interfaz_canales()

  def refrescar_interfaz_canales(self):
    """
    Recarga configuraciones de canales y músculos, actualiza active_channel_indices,
    nombres_musculos_hw, factores_calibracion_hw, colores_curvas y refresca
    la interfaz gráfica (etiquetas, leyendas, curvas) sin reiniciar la app.
    """
    for hw in range(16):
      cfg = self.config_mgr.get_channel_config(hw)
      if hw < len(self.nombres_musculos_hw):
        self.nombres_musculos_hw[hw] = cfg.get("musculo", f"Canal {hw}")
        self.factores_calibracion_hw[hw] = float(cfg.get("factor_calibracion", 495.0))

    self._actualizar_canales_activos(update_gui=True)

  @QtCore.Slot(str)
  def mostrar_preview_plot(self, ruta_img):
    """
    Muestra la ventana emergente de vista previa del gráfico durante unos segundos.
    """
    if not ruta_img or not os.path.exists(ruta_img):
      return
    # Si ya comenzó la verificación muscular en vivo, no tapar el osciloscopio
    if hasattr(self, 'autoforge_estado_actual_str') and "VERIFICACIÓN MUSCULAR" in self.autoforge_estado_actual_str:
      return
    try:
      dialog = PreviewPlotDialog(ruta_img, self, duration_ms=4000)
      dialog.show()
      self._preview_dialog = dialog
    except Exception as e:
      print(f"Error al mostrar vista previa de gráfico: {e}")

  def _on_zmq_bridge_toggled(self, checked):
    """Maneja la activación/desactivación del Puente VM ZeroMQ."""
    if checked:
      if hasattr(self, 'chk_modo_prueba') and self.chk_modo_prueba.isChecked():
        self.chk_modo_prueba.setChecked(False)
      if hasattr(self, 'chk_use_mic') and self.chk_use_mic.isChecked():
        self.chk_use_mic.setChecked(False)
    if hasattr(self, 'edit_zmq_host'):
      self.edit_zmq_host.setEnabled(bool(checked))
    if hasattr(self, 'btn_ping_zmq'):
      self.btn_ping_zmq.setEnabled(bool(checked))
    print(f"[PUENTE ZMQ] Modo Puente VM Windows {'Activado' if checked else 'Desactivado'}.")
    self.verificar_conexion_daq()

  # --- NUEVO: Cambio de modo de conexión en tiempo real ---
  def on_terminal_mode_changed(self):
    """
    Ejecuta la funcionalidad de on_terminal_mode_changed.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if self.is_acquiring and not self.chk_modo_prueba.isChecked() and getattr(self, 'chk_zmq_bridge', None) and self.chk_zmq_bridge.isChecked():
      new_mode = self.cmb_terminal_config.currentText()
      vm_ep = self.edit_zmq_host.text().strip() if hasattr(self, 'edit_zmq_host') else "auto:5555"
      print(f"[PUENTE ZMQ] Solicitando cambio de modo terminal a {new_mode} en la VM Windows ({vm_ep})...")
      try:
        from daq_bridge_client import DAQBridgeClient, parse_host_port
        host, port = parse_host_port(vm_ep, default_port=5555)
        client = DAQBridgeClient(host=host, port=port)
        resp = client.configure_remote_server(terminal_config=new_mode, timeout_ms=600)
        client.close()
        if resp and resp.get("status") == "reconfigured":
          print(f"[PUENTE ZMQ] Modo terminal actualizado en caliente en la VM: {resp}")
      except Exception as e:
        print(f"[PUENTE ZMQ WARN] No se pudo reconfigurar modo terminal remoto: {e}")
      return

    if self.is_acquiring and not self.chk_modo_prueba.isChecked() and NIDAQMX_DISPONIBLE:
      print(f"Cambiando modo terminal a {self.cmb_terminal_config.currentText()} en tiempo real...")
      
      # 1. Detener hilo actual DAQ temporalmente
      self.stop_event.set()
      if self.acquisition_thread:
        self.acquisition_thread.join(timeout=2.0)
      
      # 2. Vaciar la cola de datos para evitar un desborde con el salto de tiempo
      while not self.data_queue.empty():
        try: self.data_queue.get_nowait()
        except queue.Empty: break
      
      # 3. Obtener el nuevo modo seleccionado
      terminal_config_str = self.cmb_terminal_config.currentText()
      terminal_modes = {
        "RSE": TerminalConfiguration.RSE, "DIFF": TerminalConfiguration.DIFF,
        "NRSE": TerminalConfiguration.NRSE, "DEFAULT": TerminalConfiguration.DEFAULT
      }
      terminal_config_val = terminal_modes.get(terminal_config_str, TerminalConfiguration.DEFAULT)
      
      # 4. Reiniciar el hilo inmediatamente con la nueva configuración
      self.stop_event.clear()
      chunk_samples_dinamico = int(self.SAMPLE_RATE * 0.05)
      self.acquisition_thread = threading.Thread(
        target=acquisition_thread, 
        args=(self.CANALES_DAQ, self.SAMPLE_RATE, chunk_samples_dinamico, self.NUM_CANALES, self.data_queue, self.stop_event, terminal_config_val), 
        daemon=True
      )
      self.acquisition_thread.start()

  def _on_modo_mic_toggled(self, checked):
    """Maneja la seleccion del microfono de PC."""
    if checked and hasattr(self, 'chk_modo_prueba') and self.chk_modo_prueba.isChecked():
      self.chk_modo_prueba.setChecked(False)
    has_test = getattr(self.chk_modo_prueba, 'isChecked', lambda: False)()
    self.cmb_device.setEnabled(not checked and not has_test)
    self.cmb_terminal_config.setEnabled(not checked and not has_test)
    print(f"Modo Microfono PC {'Activado' if checked else 'Desactivado'}.")
    self.verificar_conexion_daq()

  def _on_modo_prueba_toggled(self, checked):
    """Se llama cuando el checkbox de Modo Prueba cambia."""
    if checked and hasattr(self, 'chk_use_mic') and self.chk_use_mic.isChecked():
      self.chk_use_mic.setChecked(False)
    has_mic = getattr(self.chk_use_mic, 'isChecked', lambda: False)()
    self.cmb_device.setEnabled(not checked and not has_mic)
    self.cmb_terminal_config.setEnabled(not checked and not has_mic)
    self.cmb_fuente_prueba.setEnabled(checked)
    self.label_test_freq.setEnabled(checked)
    self.spin_test_freq.setEnabled(checked)
    if checked:
      self._on_fuente_prueba_changed(self.cmb_fuente_prueba.currentIndex())
    print(f"Modo Prueba {'Activado' if checked else 'Desactivado'}.")
    self.verificar_conexion_daq()

  def on_mute_speaking_toggled(self, muted: bool):
    """Sincroniza el estado de Mute Speaking entre el botón de barra superior, el panel de filtros y VoiceCuePlayer."""
    muted_bool = bool(muted)
    if hasattr(self, 'chk_mute_speaking') and self.chk_mute_speaking.isChecked() != muted_bool:
      self.chk_mute_speaking.blockSignals(True)
      self.chk_mute_speaking.setChecked(muted_bool)
      self.chk_mute_speaking.blockSignals(False)
    if hasattr(self, 'btn_mute_speaking'):
      if self.btn_mute_speaking.isChecked() != muted_bool:
        self.btn_mute_speaking.blockSignals(True)
        self.btn_mute_speaking.setChecked(muted_bool)
        self.btn_mute_speaking.blockSignals(False)
      self.btn_mute_speaking.setText("SPEAKING: MUTE" if muted_bool else "SPEAKING: ON")
    if hasattr(self, 'voice_cue_player') and self.voice_cue_player is not None:
      self.voice_cue_player.set_muted(muted_bool)
    print(f"[Audio] Speaking {'MUTEADO (Silenciado)' if muted_bool else 'ACTIVADO'}.")

  def marcar_pulso_actual_invalido(self):
    """Marca en vivo el pulso actual del metrónomo como inválido (tos/deglución/error) para excluirlo en segmentos_dataset.csv."""
    pulso_act = int(getattr(self, 'metro_beat_count', 0) or 0)
    if pulso_act <= 0:
      print("[Control Calidad] Aún no hay un pulso activo para marcar como inválido.")
      return
    if not hasattr(self, 'invalid_pulses_set') or self.invalid_pulses_set is None:
      self.invalid_pulses_set = set()
    self.invalid_pulses_set.add(pulso_act)
    print(f"[Control Calidad] Pulso #{pulso_act} marcado manualmente como INVÁLIDO (tos / artefacto / error del sujeto).")
    if hasattr(self, 'label_rec_time'):
      self.label_rec_time.setText(f"PULSO #{pulso_act} MARCADO INVÁLIDO (EXCLUIDO DE DATASET)")
      self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFCC00; background-color: #2b1d00; border: 1px solid #FFCC00; padding: 4px 10px;")

  def keyPressEvent(self, event):
    """Atajo en vivo: Backspace o Delete marca el pulso actual como inválido sin detener la grabación."""
    if event.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
      if getattr(self, 'is_recording', False) and getattr(self, 'metro_beat_count', 0) > 0:
        self.marcar_pulso_actual_invalido()
        return
    super().keyPressEvent(event)

  def _on_fuente_prueba_changed(self, index=0):
    if not (getattr(self, 'chk_modo_prueba', None) is not None and self.chk_modo_prueba.isChecked()):
      return
    if hasattr(self, 'cmb_fuente_prueba'):
      txt = str(self.cmb_fuente_prueba.currentText()).lower()
      if "cande" in txt:
        if hasattr(self, 'spin_bpm'):
          self.spin_bpm.setValue(30)
        # Activar automaticamente los 4 canales de la sesion de Cande (ai0, ai1, ai2, ai3) solo en Modo Prueba
        if hasattr(self, 'channel_checkboxes') and not getattr(self, 'is_acquiring', False):
          for i, chk in enumerate(self.channel_checkboxes):
            chk.blockSignals(True)
            chk.setChecked(i < 4)
            chk.blockSignals(False)
          self._actualizar_canales_activos(update_gui=True)

  def hide_all_viz_channels(self):
    all_unchecked = all(not chk.isChecked() for chk in self.viz_checkboxes)
    if all_unchecked:
      for chk in self.viz_checkboxes:
        chk.setChecked(True)
      self.btn_hide_all_viz.setText("Ocultar Todos")
    else:
      for chk in self.viz_checkboxes:
        chk.setChecked(False)
      self.btn_hide_all_viz.setText("Mostrar Todos")

  def set_controls_enabled(self, enabled):
    """Habilita o deshabilita todos los controles excepto el de Start/Stop Acq."""
    self.btn_record.setEnabled(enabled)
    # btn_export se maneja por separado
    self.chk_autoscroll.setEnabled(enabled)
    self.chk_trigger.setEnabled(enabled)
    self.cmb_trig_chan.setEnabled(enabled and self.chk_trigger.isChecked())
    self.spin_trig_level.setEnabled(enabled and self.chk_trigger.isChecked())
    self.cmb_trig_edge.setEnabled(enabled and self.chk_trigger.isChecked())
    self.measure_widget.setVisible(True)
    self.trigger_line.setVisible(enabled and self.chk_trigger.isChecked())
    pass
    self.chk_spectrogram_enable.setEnabled(SCIPY_DISPONIBLE)
    self.cmb_spectrogram_chan.setEnabled(self.chk_spectrogram_enable.isChecked())
    self.spectrogram_view.setVisible(enabled and self.chk_spectrogram_enable.isChecked())
    
    # Filtros digitales: SIEMPRE habilitados y desbloqueados para interactuar en cualquier momento
    self.filter_groupbox.setEnabled(True)
    self.chk_filter_enable.setEnabled(SCIPY_DISPONIBLE)
    self.chk_notch_enable.setEnabled(SCIPY_DISPONIBLE)
    is_filter_enabled = self.chk_filter_enable.isChecked()
    self.spin_low_cut.setEnabled(is_filter_enabled)
    self.spin_high_cut.setEnabled(is_filter_enabled)

  def on_start_acq_click(self, checked=False, start_metronome=True):
    """
    Ejecuta la funcionalidad de on_start_acq_click.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if self.is_acquiring:
      # --- DETENER ADQUISICIÓN ---
      print("Deteniendo adquisición...")
      self.stop_event.set()
      
      # --- Abortar AutoForge o grabacion activa ---
      if getattr(self, 'is_autoforge_running', False):
        self.detener_autoforge()
      elif getattr(self, 'is_recording', False):
        self.on_record_click()
      else:
        self.stop_native_metronome()

      if hasattr(self, 'word_window_process') and self.word_window_process:
        try:
          self.word_window_process.kill()
          self.word_window_process.wait(timeout=1)
        except:
          pass
        self.word_window_process = None

      self._on_modo_prueba_toggled(self.chk_modo_prueba.isChecked()) # Restaurar estado de habilitación
      if self.acquisition_thread:
        self.acquisition_thread.join(timeout=2.0)
      self.acquisition_thread = None
      self.is_acquiring = False
      
      self.btn_start_acq.setText("Iniciar Adquisición")
      self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)
      self.cmb_device.setEnabled(True)
      self.cmb_terminal_config.setEnabled(True)
      self.cmb_sample_rate.setEnabled(True)
      self.chk_modo_prueba.setEnabled(True)
      self.cmb_fuente_prueba.setEnabled(self.chk_modo_prueba.isChecked())
      if hasattr(self, 'chk_zmq_bridge'):
        self.chk_zmq_bridge.setEnabled(True)
      if hasattr(self, 'edit_zmq_host'):
        self.edit_zmq_host.setEnabled(bool(self.chk_zmq_bridge.isChecked()))
      self.spin_plot_duration.setEnabled(True)
      self.chk_use_metronome.setEnabled(True)
      for chk in self.channel_checkboxes: chk.setEnabled(True)
      
      self.set_controls_enabled(False)
      if self.is_recording: # Si estaba grabando, detenerla
        self.on_record_click()

      # --- NUEVO: Desbloquear el mouse para permitir explorar el eje temporal al detener ---
      self.plot.setMouseEnabled(x=True, y=True)

    else:
      # --- INICIAR ADQUISICIÓN ---
      device = self.cmb_device.currentText()
      
      # Leer nuevos parámetros de la GUI
      self.SAMPLE_RATE = int(self.cmb_sample_rate.currentText())
      
      # --- NUEVO: Si se usa el micrófono, forzar el sample rate correcto ANTES de crear el búfer visual ---
      if getattr(self, 'chk_use_mic', None) and self.chk_use_mic.isChecked():
        self.SAMPLE_RATE = 44100
        
      self.PLOT_DURATION_S = self.spin_plot_duration.value()
      self.PLOT_SAMPLES = int(self.PLOT_DURATION_S * self.SAMPLE_RATE)
      
      # --- MEJORA: Calcular el búfer dinámicamente ---
      # Un chunk de 50ms garantiza actualizaciones de pantalla fluidas (20 FPS)
      chunk_samples_dinamico = int(self.SAMPLE_RATE * 0.05)
      self.CHUNK_DURATION_S = chunk_samples_dinamico / self.SAMPLE_RATE

      selected_channels = [chk.text() for chk in self.channel_checkboxes if chk.isChecked()]
      
      if not selected_channels:
        print("Error: Debes seleccionar al menos un canal.")
        return

      self.CANALES_DAQ = [f"{device}/{ch}" for ch in selected_channels]
      self.NUM_CANALES = len(self.CANALES_DAQ)
      
      # --- NUEVO: Obtener modo terminal ---
      terminal_config_str = self.cmb_terminal_config.currentText()
      terminal_config_val = None
      if NIDAQMX_DISPONIBLE:
        terminal_modes = {
          "RSE": TerminalConfiguration.RSE,
          "DIFF": TerminalConfiguration.DIFF,
          "NRSE": TerminalConfiguration.NRSE,
          "DEFAULT": TerminalConfiguration.DEFAULT
        }
        terminal_config_val = terminal_modes.get(terminal_config_str, TerminalConfiguration.DEFAULT)

      print(f"Iniciando con SR={self.SAMPLE_RATE}, Plot={self.PLOT_DURATION_S}s, Canales={self.CANALES_DAQ}, Terminal={terminal_config_str}")

      # --- NUEVO: Vaciar la cola de datos antes de empezar ---
      # Esto previene que datos de una adquisición anterior (con diferente N de canales)
      # se procesen en la nueva adquisición, causando un ValueError.
      while not self.data_queue.empty(): self.data_queue.get_nowait()

      # Limpiar y re-crear elementos de la GUI
      self.setup_gui_for_channels()

      usar_puente_zmq = bool(getattr(self, 'chk_zmq_bridge', None) and self.chk_zmq_bridge.isChecked())

      # --- Verificar si el dispositivo físico NI-DAQ está conectado SIN activar Modo Prueba a espaldas del usuario ---
      if not self.chk_use_mic.isChecked() and not self.chk_modo_prueba.isChecked() and not usar_puente_zmq:
        daq_conectado = False
        if NIDAQMX_DISPONIBLE:
          try:
            import nidaqmx.system
            dev_names = [d.name for d in nidaqmx.system.System.local().devices]
            daq_conectado = (device in dev_names)
          except Exception:
            daq_conectado = False
        if not daq_conectado:
          msg_err = (
            f"El dispositivo físico NI-DAQ '{device}' no está disponible localmente en este sistema.\n\n"
            f"• Si la placa USB-6259 está en la VM con Windows, active la casilla 'Puente VM (ZeroMQ)'.\n"
            f"• O active manualmente la casilla 'Modo Prueba' si desea usar datos simulados."
          )
          print(f"[ERROR DAQ] {msg_err.replace(chr(10), ' ')}")
          QtWidgets.QMessageBox.warning(self, "Dispositivo NI-DAQ No Detectado Localmente", msg_err)
          return

      # --- Lanzar el metrónomo solo si no estamos en pre-arranque de AutoForge ---
      if self.chk_use_metronome.isChecked() and start_metronome:
        self._save_metronome_config()
        self.start_native_metronome(count_in=0, force_start=True)

      # Iniciar hilo
      self.stop_event.clear()

      if self.chk_use_mic.isChecked():
        chunk_samples_dinamico = int(self.SAMPLE_RATE * 0.05)
        self.acquisition_thread = threading.Thread(target=microphone_thread, args=(chunk_samples_dinamico, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event), daemon=True)
      elif self.chk_modo_prueba.isChecked():
        test_freq = self.spin_test_freq.value()
        tipo_prueba = self.cmb_fuente_prueba.currentText()
        self.acquisition_thread = threading.Thread(target=simulador_thread, args=(chunk_samples_dinamico, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event, test_freq, tipo_prueba, self.active_channel_indices), daemon=True)
      elif usar_puente_zmq:
        vm_endpoint_str = self.edit_zmq_host.text().strip() if hasattr(self, 'edit_zmq_host') else "auto:5555"
        self.acquisition_thread = threading.Thread(
          target=zmq_bridge_acquisition_thread,
          args=(vm_endpoint_str, device, self.CANALES_DAQ, self.SAMPLE_RATE, chunk_samples_dinamico, self.NUM_CANALES, self.data_queue, self.stop_event, terminal_config_str),
          daemon=True
        )
      else:
        self.acquisition_thread = threading.Thread(target=acquisition_thread, args=(self.CANALES_DAQ, self.SAMPLE_RATE, chunk_samples_dinamico, self.NUM_CANALES, self.data_queue, self.stop_event, terminal_config_val), daemon=True)
      
      self.acquisition_thread.start()
      self.is_acquiring = True
      
      # --- NUEVO: Preparar el auto-ajuste de barras al iniciar ---
      self.needs_auto_threshold = True
      self.acq_start_time = time.perf_counter()
      self._raw_print_done = False # Para imprimir el voltaje RAW de depuración
      
      # --- CORRECCIÓN: Inicializar filtros ahora que is_acquiring es True ---
      self.on_filter_settings_changed()
      
      self.btn_start_acq.setText("Detener Adquisición")
      self.btn_start_acq.setStyleSheet(self.BTN_STOP_STYLE)
      self.cmb_device.setEnabled(False)
      # self.cmb_terminal_config.setEnabled(False) # <-- ELIMINADO para poder cambiar en vivo
      self.cmb_sample_rate.setEnabled(False)
      self.chk_modo_prueba.setEnabled(False)
      self.cmb_fuente_prueba.setEnabled(False)
      if hasattr(self, 'chk_zmq_bridge'):
        self.chk_zmq_bridge.setEnabled(False)
      if hasattr(self, 'edit_zmq_host'):
        self.edit_zmq_host.setEnabled(False)
      self.spin_plot_duration.setEnabled(False)
      self.chk_use_metronome.setEnabled(True)
      for chk in self.channel_checkboxes: chk.setEnabled(False)
      
      self.set_controls_enabled(True)
      self.on_autoscroll_toggle() # Restaura el estado del mouse según el Auto-scroll

  def _persistir_preferencias_en_config_local(self):
    """Persiste bidireccionalmente los ajustes de la interfaz en config_local.json."""
    try:
      if not hasattr(self, 'config_mgr') or self.config_mgr is None:
        return
      estado = {
        "adquisicion": {
          "bpm": int(self.spin_bpm.value()) if hasattr(self, 'spin_bpm') else 25,
          "subdivisiones": int(self.spin_metro_subdivs.value()) if hasattr(self, 'spin_metro_subdivs') else 4,
          "ruido_segundos": float(self.spin_noise_duration.value()) if hasattr(self, 'spin_noise_duration') else 3.0,
          "notch_50hz_activo": bool(self.chk_notch_enable.isChecked()) if hasattr(self, 'chk_notch_enable') else False,
          "pasabanda_low_hz": float(self.spin_low_cut.value()) if hasattr(self, 'spin_low_cut') else 20.0,
          "pasabanda_high_hz": float(self.spin_high_cut.value()) if hasattr(self, 'spin_high_cut') else 500.0,
          "limitar_rango_500uv": bool(self.chk_clamp_500uv.isChecked()) if hasattr(self, 'chk_clamp_500uv') else True,
        },
        "visualizacion": {
          "limitar_rango_500uv": bool(self.chk_clamp_500uv.isChecked()) if hasattr(self, 'chk_clamp_500uv') else True,
          "umbral_picos_snr_uv": float(self.spin_peak_th.value()) if hasattr(self, 'spin_peak_th') else 200.0,
        },
      }
      if hasattr(self.config_mgr, 'persistir_estado_autoforge'):
        self.config_mgr.persistir_estado_autoforge(estado)
      else:
        self.config_mgr.save_local_config()
    except Exception:
      pass

  def _on_live_bpm_or_subdiv_changed(self, value=None):
    self._persistir_preferencias_en_config_local()
    '''Permite cambiar el BPM o las subdivisiones en vivo (incluso durante Autograbado o Secuencia Continua).'''
    bpm_val = max(10, int(self.spin_bpm.value())) if hasattr(self, 'spin_bpm') else 30
    sub_val = max(1, int(self.spin_metro_subdivs.value())) if hasattr(self, 'spin_metro_subdivs') else 4
    self.current_beat_tau_s = 60.0 / float(bpm_val)
    if hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked() and not getattr(self, 'is_cvm_active', False):
      n_sched = max(60, int(getattr(self, 'autoforge_target_reps', 30)) * max(1, len(getattr(self, 'autoforge_words', [1]))))
      self.metro_jitter_schedule = generate_jittered_intervals(
        bpm=bpm_val,
        n_pulses=n_sched,
        jitter_pct=float(self.spin_metro_jitter_pct.value()) / 100.0,
        mode=self.cmb_metro_jitter_mode.currentText()
      )
    try:
      self.config_mgr.set("adquisicion", "bpm", bpm_val)
      self.config_mgr.set("adquisicion", "subdivisiones", sub_val)
    except Exception:
      pass
    self._save_metronome_config()

  def _save_metronome_config(self):
    """Guarda el valor actual de BPM y subdivisiones en el archivo de configuración del metrónomo."""
    try:
      current_bpm = self.spin_bpm.value()
      config_data = {}
      if os.path.exists(METRONOME_CONFIG_PATH):
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          config_data = json.load(f) # Asegurar que se lee como UTF-8
      
      config_data['last_bpm'] = current_bpm
      if hasattr(self, 'spin_metro_subdivs'):
        config_data['subdivisions'] = self.spin_metro_subdivs.value()
      with open(METRONOME_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
      print(f"Actualizado BPM ({current_bpm}) y Subdivisiones ({config_data.get('subdivisions', 4)}) en metronome_config.json")
    except Exception as e:
      print(f"Advertencia: No se pudo guardar config para el metrónomo. Error: {e}")

  def setup_gui_for_channels(self):
    """Limpia y re-crea los elementos de la GUI que dependen del número de canales."""
    # Limpiar curvas, labels de medida y combo de trigger
    for curva in self.curvas: self.plot.removeItem(curva)
    for line in getattr(self, 'noise_lines', []): self.plot.removeItem(line)
    for line in getattr(self, 'noise_lines_neg', []): self.plot.removeItem(line)
    for reg in getattr(self, 'noise_regions', []): self.plot.removeItem(reg)
    for region in getattr(self, 'dynamic_noise_regions', []): self.plot.removeItem(region)
    for scatter in getattr(self, 'peak_scatters', []): self.plot.removeItem(scatter)
    for label in self.measure_labels: label.deleteLater()
    self.curvas.clear()
    self.noise_lines.clear()
    self.noise_lines_neg.clear()
    self.noise_regions.clear()
    self.dynamic_noise_regions.clear()
    self.peak_scatters = []
    self.measure_labels.clear()
    if hasattr(self, 'noise_status_labels'):
      self.noise_status_labels.clear()
    else:
      self.noise_status_labels = []
    if hasattr(self, 'line_noise_status_frames'):
      self.line_noise_status_frames.clear()
      self.line_noise_status_labels.clear()
    else:
      self.line_noise_status_frames = []
      self.line_noise_status_labels = []
    self.cmb_spectrogram_chan.clear()
    self.cmb_trig_chan.clear()
    
    if hasattr(self.plot, 'legend') and self.plot.legend is not None:
      self.plot.scene().removeItem(self.plot.legend)
      self.plot.legend = None
      
    # Limpiar checkboxes de visualizacion
    while self.viz_checkbox_layout.count():
      item = self.viz_checkbox_layout.takeAt(0)
      if item.widget():
        item.widget().deleteLater()
    self.viz_checkboxes.clear()
      
    plot_dur_safe = float(self.PLOT_DURATION_S) if getattr(self, 'PLOT_DURATION_S', 0) > 0 else float(self.spin_plot_duration.value() if hasattr(self, 'spin_plot_duration') else 10.0)
    self.plot.getViewBox().setLimits(xMin=-plot_dur_safe) # Actualizar límite de zoom
    self.plot.addLegend()
    
    # Re-crear elementos
    # Limpiar el layout de mediciones antes de añadir nuevos elementos
    while self.measure_layout.count():
      item = self.measure_layout.takeAt(0)
      if item.widget():
        item.widget().deleteLater()
      elif item.layout():
        while item.layout().count():
          sub_item = item.layout().takeAt(0)
          if sub_item.widget():
            sub_item.widget().deleteLater()
            
    # Limpiar layout de ruido del metronomo
    if hasattr(self, 'metro_noise_layout'):
      while self.metro_noise_layout.count():
        item = self.metro_noise_layout.takeAt(0)
        widget = item.widget()
        if widget:
          widget.deleteLater()
    
    title_label = QtWidgets.QLabel("<b>Mediciones (chunk):</b>")
    title_label.setStyleSheet("color: white;")
    self.measure_layout.addWidget(title_label)
    for i in range(self.NUM_CANALES):
      hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and i < len(self.active_channel_indices)) else i
      color = self.colores_curvas[i % len(self.colores_curvas)]
      musculo = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
      curva_nombre = f"ai{hw_idx} - {musculo}"
      curva_item = self.plot.plot(pen=color, name=curva_nombre)
      curva_item.setZValue(10) # Señales siempre en primer plano sobre el sombreado
      self.curvas.append(curva_item)
      
      # --- NUEVO: Checkbox de visualización ---
      chk_viz = QtWidgets.QCheckBox(curva_nombre)
      chk_viz.setChecked(True)
      chk_viz.setStyleSheet(f"color: {pg.mkColor(color).name()}; font-weight: bold; border: none;")
      
      def toggle_viz(state, idx=i):
        if not getattr(self, 'is_multitrack_view', False):
          if idx < len(self.curvas):
            self.curvas[idx].setVisible(state)
        else:
          if idx == 0 and len(self.curvas) > 0:
            self.curvas[0].setVisible(state)
          elif idx > 0:
            sub_idx = idx - 1
            multitrack_curvas = getattr(self, 'curvas_multitrack', [])
            if sub_idx < len(multitrack_curvas):
              multitrack_curvas[sub_idx].setVisible(state)
          
      chk_viz.toggled.connect(toggle_viz)
      self.viz_checkbox_layout.addWidget(chk_viz)
      self.viz_checkboxes.append(chk_viz)
      
      # --- NUEVO: Línea fina y distinguible por canal para piso de ruido ---
      chan_qcolor = pg.mkColor(color)
      pen_ruido = pg.mkPen(color=chan_qcolor, width=1.2, style=QtCore.Qt.DashLine)
      
      line_ruido = pg.InfiniteLine(angle=0, pen=pen_ruido)
      line_ruido.setZValue(-10) # Visible pero sin opacar la señal
      line_ruido.hide()
      self.plot.addItem(line_ruido)
      self.noise_lines.append(line_ruido)
      
      line_ruido_n = pg.InfiniteLine(angle=0, pen=pen_ruido)
      line_ruido_n.setZValue(-10)
      line_ruido_n.hide()
      self.plot.addItem(line_ruido_n)
      self.noise_lines_neg.append(line_ruido_n)
      
      # Sombreado translúcido y suave estrictamente por debajo de las señales (ZValue=-100)
      brush_ruido = pg.mkBrush(chan_qcolor.red(), chan_qcolor.green(), chan_qcolor.blue(), 18)
      region_ruido = pg.LinearRegionItem(orientation='horizontal', brush=brush_ruido, movable=False)
      region_ruido.lines[0].setPen(pg.mkPen(None)) # Ocultar bordes propios
      region_ruido.lines[1].setPen(pg.mkPen(None))
      region_ruido.setZValue(-100) # Por debajo de las señales
      region_ruido.hide()
      self.plot.addItem(region_ruido)
      self.noise_regions.append(region_ruido)
      
      # --- NUEVO: Marcador visual para el pico máximo (SNR) ---
      scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=1.5), brush=pg.mkBrush(color))
      scatter.setZValue(20)
      scatter.setVisible(self.chk_show_peaks.isChecked())
      self.plot.addItem(scatter)
      self.peak_scatters.append(scatter)
      
      # --- NUEVO: Región dinámica de ruido inter-pulso translúcida por debajo de la señal ---
      dyn_region = pg.LinearRegionItem(orientation='horizontal', brush=pg.mkBrush(0, 255, 128, 18), movable=False)
      dyn_region.lines[0].setPen(pg.mkPen(None))
      dyn_region.lines[1].setPen(pg.mkPen(None))
      dyn_region.setZValue(-90) # Por debajo de las señales
      dyn_region.setVisible(self.chk_show_noise.isChecked())
      self.plot.addItem(dyn_region)
      self.dynamic_noise_regions.append(dyn_region)

      # --- NUEVO: Crear un QFrame para cada medición ---
      measurement_frame = QtWidgets.QFrame()
      measurement_frame.setStyleSheet("background-color: #050505; border: 2px solid #00FFFF; border-radius: 5px;")
      measurement_frame.setMinimumWidth(100)
      
      frame_layout = QtWidgets.QVBoxLayout(measurement_frame)
      frame_layout.setContentsMargins(5, 5, 5, 5) # Padding interno
      
      label = QtWidgets.QLabel(f"<b>ai{hw_idx} - {musculo}:</b> -- µVp-p, -- µVrms")
      label.setStyleSheet(f"color: {pg.mkColor(color).name()}; font-size: 14px; font-weight: bold; background-color: transparent; border: none;")
      label.setWordWrap(True)
      label.setMinimumWidth(0)
      frame_layout.addWidget(label)
      
      # --- NUEVO: Etiqueta para el tester de ruido ---
      label_ruido = QtWidgets.QLabel("Ruido inter-pulso: Esperando grabación...")
      label_ruido.setStyleSheet("color: gray; font-size: 11px; background-color: transparent;")
      label_ruido.setWordWrap(True)
      label_ruido.setMinimumWidth(0)
      label_ruido.setVisible(self.chk_show_noise.isChecked())
      frame_layout.addWidget(label_ruido)
      self.noise_status_labels.append(label_ruido)
      
      # --- NUEVO: Recuadro destacado con color para Ruido de Línea 50 Hz ---
      frame_50hz = QtWidgets.QFrame()
      frame_50hz.setStyleSheet("background-color: #111111; border: 1.5px solid #555555; border-radius: 3px; padding: 2px;")
      layout_50hz = QtWidgets.QVBoxLayout(frame_50hz)
      layout_50hz.setContentsMargins(3, 2, 3, 2)
      label_50hz = QtWidgets.QLabel(f"50Hz ai{hw_idx} - {musculo}: -- µV")
      label_50hz.setStyleSheet("color: #AAAAAA; font-size: 11px; font-weight: bold;")
      label_50hz.setWordWrap(True)
      layout_50hz.addWidget(label_50hz)
      
      frame_layout.addWidget(frame_50hz)
          
      self.line_noise_status_frames.append(frame_50hz)
      self.line_noise_status_labels.append(label_50hz)
      
      self.measure_layout.addWidget(measurement_frame)
      self.measure_labels.append(label) # Guardar solo el label para actualizar su texto

    self.measure_layout.addStretch(1) # Asegurar que los recuadros se alineen a la izquierda

    for i in range(self.NUM_CANALES):
      hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and i < len(self.active_channel_indices)) else i
      musc = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
      self.cmb_trig_chan.addItem(f"ai{hw_idx} - {musc}")
      self.cmb_spectrogram_chan.addItem(f"ai{hw_idx} - {musc}")
    
    # Inicializar buffers con el tamaño correcto
    self.plot_buffer_datos = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES))
    self.env_buffer_datos = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES))
    self.plot_vector_tiempo = np.linspace(-self.PLOT_DURATION_S, 0, self.PLOT_SAMPLES)
    self.trigger_last_values = np.zeros(self.NUM_CANALES)

    # Inicializar buffer del espectrograma
    chunk_samples_dinamico = int(self.SAMPLE_RATE * 0.05)
    self.CURRENT_FFT_LEN = min(self.SPECTROGRAM_FFT_LEN, chunk_samples_dinamico)
    if self.CURRENT_FFT_LEN % 2 != 0: self.CURRENT_FFT_LEN -= 1
    self.spectrogram_buffer = np.zeros((self.SPECTROGRAM_HISTORY_LEN, self.CURRENT_FFT_LEN // 2 + 1))
    self.on_spectrogram_enable_toggle() # Para mostrar/ocultar la vista

    # --- CORRECCIÓN v3: Configurar la transformación del espectrograma una sola vez ---
    # En lugar de pasar 'scale' en cada llamada a setImage, lo que puede ser propenso a errores,
    # configuramos la transformación directamente en el ViewBox del ImageView.
    # Esto es más eficiente y robusto.
    view = self.spectrogram_view.getView()
    
    # Eje Y (Frecuencia): Va de 0 a Frecuencia de Nyquist.
    fs_val = float(self.SAMPLE_RATE) if (hasattr(self, 'SAMPLE_RATE') and self.SAMPLE_RATE > 0) else 2000.0
    nyquist = fs_val / 2.0
    freq_scale = nyquist / (self.CURRENT_FFT_LEN / 2 + 1)

    # Eje X (Tiempo): El ancho total del historial del espectrograma en segundos.
    # Cada columna representa un segmento de tiempo (nperseg - noverlap) / fs.
    time_per_column = (self.CURRENT_FFT_LEN - (self.CURRENT_FFT_LEN // 2)) / fs_val
    time_scale = time_per_column

    view.setAspectLocked(False) # Desbloquear la relación de aspecto para escalar ejes independientemente
    view.setRange(xRange=(0, self.SPECTROGRAM_HISTORY_LEN * time_scale), yRange=(0, nyquist), padding=0)


  # --- NUEVO: FUNCIONES DEL ESPECTROGRAMA ---
  def on_spectrogram_enable_toggle(self):
    """Muestra u oculta la vista del espectrograma."""
    enabled = self.chk_spectrogram_enable.isChecked() and SCIPY_DISPONIBLE
    self.spectrogram_view.setVisible(enabled)
    self.cmb_spectrogram_chan.setEnabled(enabled)
    if not enabled:
      # Limpia el buffer si se deshabilita para no mostrar datos viejos
      self.spectrogram_buffer.fill(0)
      self.spectrogram_view.setImage(self.spectrogram_buffer.T, autoLevels=False, levels=(0, 1))

  def on_spectrogram_channel_change(self, index):
    """Cambia el canal que se usa para el espectrograma y limpia el buffer."""
    # --- CORRECCIÓN: Asegurarse de que el buffer existe antes de limpiarlo ---
    if self.spectrogram_buffer is None:
      return
    self.spectrogram_channel_index = index
    self.spectrogram_buffer.fill(0) # Limpiar historial al cambiar de canal

  def toggle_noise_regions(self, checked=None):
    if checked is None:
      show = self.chk_show_noise.isChecked()
    else:
      show = checked
    is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
    for reg in getattr(self, 'noise_regions', []): reg.setVisible(show)
    for reg in getattr(self, 'dynamic_noise_regions', []): reg.setVisible(show)
    for line in getattr(self, 'noise_lines', []): line.setVisible(show)
    for line in getattr(self, 'noise_lines_neg', []): line.setVisible(show if not is_rms else False)
    for lbl in getattr(self, 'noise_status_labels', []): lbl.setVisible(show)

  def toggle_peak_scatter(self):
    show = self.chk_show_peaks.isChecked()
    for scatter in getattr(self, 'peak_scatters', []): scatter.setVisible(show)

  def toggle_detection_window(self):
    show = self.chk_show_detection_window.isChecked()
    if not show:
      if hasattr(self, 'detection_window_region'):
        self.detection_window_region.hide()
      if hasattr(self, 'detection_window_label'):
        self.detection_window_label.hide()
    else:
      self._actualizar_posicion_ventana_deteccion()

  def _actualizar_posicion_ventana_deteccion(self):
    if not hasattr(self, 'detection_window_region') or not hasattr(self, 'chk_show_detection_window'):
      return
    if not self.chk_show_detection_window.isChecked():
      self.detection_window_region.hide()
      self.detection_window_label.hide()
      return

    ev = getattr(self, 'latest_pulse_event', None)
    if ev is None:
      self.detection_window_region.hide()
      self.detection_window_label.hide()
      return

    tau = float(ev.get('tau', 0.0))
    if tau <= 0.0:
      self.detection_window_region.hide()
      self.detection_window_label.hide()
      return

    fs = float(getattr(self, 'SAMPLE_RATE', 0.0) or 0.0)
    event_abs_sample = ev.get('event_abs_sample', None)
    if fs > 0.0 and event_abs_sample is not None:
      # Anclaje absoluto: la ventana se recalcula en cada refresco restando el
      # contador de muestras procesadas. Se desplaza exactamente a la velocidad
      # del flujo de datos, sin acumular deriva en los valles de silencio.
      t_anchor = (float(event_abs_sample) - float(getattr(self, 'total_muestras_procesadas', 0))) / fs
    else:
      # Compatibilidad con eventos antiguos que no guardaban event_abs_sample.
      detect_sample_idx = ev.get('detect_sample_idx', None)
      if detect_sample_idx is not None and fs > 0.0:
        samples_elapsed = getattr(self, 'total_muestras_procesadas', 0) - detect_sample_idx
        dt = samples_elapsed / fs
      else:
        t_detect_perf = ev.get('t_detect_perf', 0.0)
        if t_detect_perf <= 0.0:
          return
        dt = time.perf_counter() - t_detect_perf
      t_anchor = float(ev.get('t_anchor_rel', ev.get('t_center', 0.0) - 0.10 * tau)) - dt

    w_start = t_anchor - 0.10 * tau
    w_end = t_anchor + 0.40 * tau

    plot_dur = float(self.PLOT_DURATION_S) if getattr(self, 'PLOT_DURATION_S', 0) > 0 else 30.0
    if w_end < -plot_dur or w_start > 0.5:
      self.detection_window_region.hide()
      self.detection_window_label.hide()
      return

    self.detection_window_region.setRegion([w_start, w_end])
    self.detection_window_region.show()

    method_tag = "GATE" if ev.get('method') == 'GATE_DOBLE' else "MIC"
    vocal = ev.get('vocal', '')
    vocal_tag = f" [/{vocal}/]" if vocal else ""
    sup = ev.get('supremo_uV', 0.0)
    sep = ev.get('separability_pct', 0.0)

    vb = self.plot.getViewBox()
    view_range_y = vb.viewRange()[1]
    y_pos = view_range_y[1] * 0.88 if view_range_y[1] > 0 else 0.0

    label_text = f"Ventana 0.5τ: {method_tag}{vocal_tag} | Sup: {sup:.0f}µV | Sep: {sep:.0f}%"
    self.detection_window_label.setText(label_text)
    self.detection_window_label.setPos((w_start + w_end) / 2.0, y_pos)
    self.detection_window_label.show()

  def _reubicar_metronomo_dinamico(self, index):
    pass

  def _setup_native_metronome(self):
    self.right_side_panel = QtWidgets.QWidget()
    self.right_side_layout = QtWidgets.QVBoxLayout(self.right_side_panel)
    self.right_side_layout.setContentsMargins(0, 0, 0, 0)
    
    self.metronome_container = QtWidgets.QGroupBox("Metrónomo")
    self.metronome_container.setStyleSheet("""
      QGroupBox {
          border: 2px solid #00FFFF;
          border-radius: 6px;
          margin-top: 1.5ex;
          font-family: 'Courier New', monospace;
          color: #00FFFF;
          background-color: #050505;
      }
      QGroupBox::title {
          subcontrol-origin: margin;
          left: 10px;
          padding: 0 3px 0 3px;
          color: #00FFFF;
          font-weight: bold;
      }
    """)
    self.metronome_container.setMinimumWidth(165)
    metro_layout = QtWidgets.QVBoxLayout(self.metronome_container)
    metro_layout.setContentsMargins(6, 6, 6, 4)
    metro_layout.setSpacing(2)

    self.metro_pulse_frame = QtWidgets.QFrame()
    self.metro_pulse_frame.setFixedHeight(34)
    self.metro_pulse_frame.setStyleSheet("background-color: #111111; border: 2px solid #00FFFF;")
    metro_layout.addWidget(self.metro_pulse_frame)

    self.metro_lbl_title = QtWidgets.QLabel("PULSO")
    self.metro_lbl_title.setAlignment(QtCore.Qt.AlignCenter)
    self.metro_lbl_title.setStyleSheet("font-size: 12px; color: #00FFFF; font-family: 'Courier New', monospace; font-weight: bold;")
    metro_layout.addWidget(self.metro_lbl_title)

    self.metro_lbl_count = QtWidgets.QLabel("0")
    self.metro_lbl_count.setAlignment(QtCore.Qt.AlignCenter)
    self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #00FFFF; font-weight: bold; font-family: 'Courier New', monospace;")
    metro_layout.addWidget(self.metro_lbl_count)

    self.right_side_layout.addWidget(self.metronome_container)

    # --- Semáforo de Fatiga Mioeléctrica (MDF) ubicado debajo del Metrónomo a la derecha ---
    self.frame_fatiga = QtWidgets.QFrame()
    self.frame_fatiga.setMinimumWidth(165)
    self.frame_fatiga.setStyleSheet("background-color: #0b1a14; border: 1.5px solid #39ff14; border-radius: 5px; padding: 3px 6px;")
    lyt_fat = QtWidgets.QHBoxLayout(self.frame_fatiga)
    lyt_fat.setContentsMargins(4, 2, 4, 2)
    lyt_fat.setSpacing(5)
    self.lbl_fatiga_dot = QtWidgets.QLabel("●")
    self.lbl_fatiga_dot.setStyleSheet("color: #39ff14; font-size: 14px; font-weight: bold;")
    self.lbl_fatiga_text = QtWidgets.QLabel("FATIGA: ESTABLE\nMDF 100%")
    self.lbl_fatiga_text.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_fatiga_text.setStyleSheet("color: #39ff14; font-size: 10px; font-weight: bold; font-family: 'Courier New', monospace;")
    lyt_fat.addWidget(self.lbl_fatiga_dot)
    lyt_fat.addWidget(self.lbl_fatiga_text, stretch=1)
    self.right_side_layout.addWidget(self.frame_fatiga)

    self.noise_status_group = QtWidgets.QWidget()
    self.metro_noise_layout = QtWidgets.QVBoxLayout(self.noise_status_group)
    self.metro_noise_layout.setContentsMargins(0, 2, 0, 0)
    self.right_side_layout.addWidget(self.noise_status_group)
    self.right_side_layout.addStretch()

    # --- TREADING PERSISTENTE PARA BEEPS (ELIMINA EL LAG) ---
    import queue
    self.beep_queue = queue.Queue()
    def beep_worker():
        while True:
            item = self.beep_queue.get()
            if item is None: break
            freq, duration = item
            play_beep(freq, duration, async_play=False)
    self.beep_thread = threading.Thread(target=beep_worker, daemon=True)
    self.beep_thread.start()

    self.metro_timer = QtCore.QTimer()
    # CoarseTimer (default) tiene +/-5% de tolerancia: a 250 ms son +/-12 ms y
    # el pulso se adelanta. Para medir tiempos exactos hace falta PreciseTimer.
    self.metro_timer.setTimerType(QtCore.Qt.PreciseTimer)
    self.metro_timer.timeout.connect(self.on_metro_beat)
    
    # Hacer que el metrónomo sea visible por defecto
    self.metronome_container.show()
    self.metro_is_running = False
    self.metro_count_in_remaining = 0
    self.metro_beat_count = 0
    # Registro de tiempos exactos de los pulsos (impresion + CSV de sesion + metadata)
    self._metro_pulsos = []
    self._metro_pulso_seq = 0
    self.metro_beat_timestamps_s = []
    self.metro_beat_sample_indices = []
    self.metro_applied_intervals_s = []
    self.metro_jitter_schedule = []
    self.current_beat_tau_s = 1.0
    # Reloj monotono absoluto del proximo pulso (evita deriva acumulada)
    self._metro_next_perf = None
    self._paused_metro_remaining_ms = 0
    self._on_count_in_finished_cb = None
    
    self.COLOR_IDLE = "#111111"
    self.COLOR_BEAT = "#00FFFF"
    self.COLOR_PREP = "#FF0000"

    self.metronome_container.setVisible(self.chk_use_metronome.isChecked())
    self.chk_use_metronome.toggled.connect(self._on_metronome_toggled)
    self.right_side_panel.show()

  def _on_metronome_toggled(self, checked):
    self.metronome_container.setVisible(checked)
    if checked:
      if self.is_acquiring and not getattr(self, 'metro_is_running', False):
        self.start_native_metronome(count_in=0, force_start=True)
    else:
      if getattr(self, 'metro_is_running', False):
        self.stop_native_metronome()

  def start_native_metronome(self, count_in=0, force_start=False):
    if not force_start and not self.chk_use_metronome.isChecked():
        return
    self.chk_use_metronome.setChecked(True)
    self.metronome_container.setVisible(True)
    self.metronome_container.show()
    self.right_side_panel.show()
    self.metro_count_in_remaining = count_in
    self.metro_is_running = True
    self.metro_beat_count = 0
    self.metro_subdiv_index = 0
    self._metro_pulsos = []
    self._metro_pulso_seq = 0
    self.metro_beat_timestamps_s = []
    self.metro_beat_sample_indices = []
    self.metro_applied_intervals_s = []
    self._metro_next_perf = None
    self._paused_metro_remaining_ms = 0

    bpm_val = max(1, self.spin_bpm.value())
    self.current_beat_tau_s = 60.0 / float(bpm_val)
    if hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked() and not getattr(self, 'is_cvm_active', False):
        n_sched = max(60, int(getattr(self, 'autoforge_target_reps', 30)) * max(1, len(getattr(self, 'autoforge_words', [1]))))
        self.metro_jitter_schedule = generate_jittered_intervals(
            bpm=bpm_val,
            n_pulses=n_sched,
            jitter_pct=float(self.spin_metro_jitter_pct.value()) / 100.0,
            mode=self.cmb_metro_jitter_mode.currentText()
        )
    else:
        self.metro_jitter_schedule = []

    self.on_metro_beat()

  def save_metronome_config(self):
    self._save_metronome_config()

  def stop_native_metronome(self):
    self.metro_is_running = False
    self.metro_timer.stop()
    self._metro_next_perf = None
    self._paused_metro_remaining_ms = 0
    self._on_count_in_finished_cb = None
    self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_IDLE}; border: 2px solid #00FFFF;")
    self.metro_lbl_count.setText("0")
    self.metro_lbl_title.setText("PULSO")
    self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #00FFFF;")
    self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #00FFFF; font-weight: bold;")
    self.save_metronome_config()

  def reset_metro_color(self):
    self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_IDLE}; border: 2px solid #00FFFF;")

  def _construir_tira_secuencia_html(self, words_list, active_idx):
    """Construye la barra horizontal de vocales resaltando la vocal actual y la siguiente."""
    if not words_list:
      return ""
    vocal_colores = {'A': '#FF3355', 'E': '#00B4D8', 'I': '#39FF14', 'O': '#C77DFF', 'U': '#FFD166'}
    items_html = []
    n_w = len(words_list)
    # Si la lista es muy larga (>8), mostrar ventana alrededor de active_idx
    if n_w <= 8:
      indices = list(range(n_w))
    else:
      indices = [(active_idx + offset) % n_w for offset in range(-1, 6)]
    for i_w in indices:
      w_txt = str(words_list[i_w]).upper()
      c_w = vocal_colores.get(w_txt, '#00FFCC')
      if i_w == (active_idx % n_w):
        items_html.append(
          f"<span style='background-color:{c_w}; color:#050508; font-weight:900; font-size:18px; padding:2px 8px; border-radius:4px; border:2px solid #FFFFFF;'>/{w_txt}/</span>"
        )
      elif i_w == ((active_idx + 1) % n_w):
        items_html.append(
          f"<span style='color:#FFFFFF; font-weight:bold; font-size:16px; border:1px dashed {c_w}; padding:1px 6px; border-radius:3px;'>/{w_txt}/</span>"
        )
      else:
        items_html.append(
          f"<span style='color:#667788; font-size:15px;'>/{w_txt}/</span>"
        )
    return " &rarr; ".join(items_html)

  def _renderizar_visor_vocal(self, palabra_actual: str, pulso_actual: int, total_pulsos: int,
                              palabra_siguiente: str = "", ciclo_actual: int = 1, total_ciclos: int = 1,
                              flash: bool = False, pre_cambio: bool = False):
    """
    Actualiza el visor de vocales (lbl_recording_space) con:
    - Destello visual (flash=True) sincronizado al golpe del metrónomo para que el sujeto perciba el pulso en la propia letra.
    - Tira de secuencia y cartel de PRÓXIMA VOCAL para que nunca se pierda en Secuencia Continua ni Discreta.
    """
    if not hasattr(self, 'lbl_recording_space'):
      return
    p_act = str(palabra_actual or "").upper()
    p_sig = str(palabra_siguiente or "").upper()
    vocal_colores = {'A': '#FF3355', 'E': '#00B4D8', 'I': '#39FF14', 'O': '#C77DFF', 'U': '#FFD166'}
    vocal_rgb = {'A': '255, 51, 85', 'E': '0, 180, 216', 'I': '57, 255, 20', 'O': '199, 125, 255', 'U': '255, 209, 102'}
    v_col = vocal_colores.get(p_act, '#00FFCC')
    v_rgb = vocal_rgb.get(p_act, '0, 255, 204')
    sig_col = vocal_colores.get(p_sig, '#FFFF00')

    words = getattr(self, 'autoforge_words', []) or [p_act]
    if getattr(self, 'is_autoforge_continuo', False):
      full_seq = getattr(self, 'autoforge_full_sequence', None) or (words * max(1, getattr(self, 'autoforge_target_reps', 1)))
      active_seq_idx = max(0, pulso_actual - 1) % max(1, len(full_seq))
      tira_html = self._construir_tira_secuencia_html(full_seq, active_seq_idx)
    else:
      active_seq_idx = int(getattr(self, 'autoforge_word_idx', 0))
      tira_html = self._construir_tira_secuencia_html(words, active_seq_idx)

    if flash:
      self.lbl_recording_space.setStyleSheet(
        f"background-color: rgba({v_rgb}, 0.38); color: #FFFFFF; font-family: 'Courier New', monospace; "
        f"border: 4px solid #FFFFFF; border-radius: 8px; padding: 4px;"
      )
      badge_top = f"<span style='background-color:#FFFFFF; color:#050508; font-size:15px; font-weight:900; padding:1px 10px; border-radius:3px;'>&#9733; ¡PRONUNCIA AHORA! &mdash; PULSO {pulso_actual}/{total_pulsos} &#9733;</span>"
      letra_style = f"font-size: 76px; font-weight: 900; color: #FFFFFF;"
    else:
      b_col = "#FFFF00" if pre_cambio else v_col
      self.lbl_recording_space.setStyleSheet(
        f"background-color: #050510; color: {v_col}; font-family: 'Courier New', monospace; "
        f"border: 2px solid {b_col}; border-radius: 6px; padding: 4px;"
      )
      if pre_cambio and p_sig:
        badge_top = f"<span style='color:#FFFF00; font-size:14px; font-weight:bold;'>ATENCIÓN: PRÓXIMO PULSO CAMBIA A /{p_sig}/ &mdash; PULSO {pulso_actual}/{total_pulsos}</span>"
      else:
        badge_top = f"<span style='color:#AAAAAA; font-size:14px; font-weight:bold;'>MANTÉN LA VOCAL &mdash; PULSO {pulso_actual}/{total_pulsos} (CICLO {ciclo_actual}/{total_ciclos})</span>"
      letra_style = f"font-size: 68px; font-weight: 900; color: {v_col};"

    sig_html = (
      f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#CCCCCC; font-size:14px;'>SIGUIENTE: <b style='color:{sig_col}; font-size:18px;'>/{p_sig}/</b></span>"
      if p_sig else ""
    )

    if getattr(self, 'blind_subject_hide_local_vowel', False):
      html = (
        f"<div align='center' style='line-height:1.25;'>"
        f"<span style='color:#c77dff; font-size:16px; font-weight:bold;'>[MODO SUJETO CIEGO ACTIVO EN PANTALLA EXTERNA]</span><br>"
        f"<span style='color:#00ffcc; font-size:15px;'>Pulso {pulso_actual}/{total_pulsos} (Ciclo {ciclo_actual}/{total_ciclos}) — Vocal oculta en monitor local</span>"
        f"</div>"
      )
    else:
      # Anillo circular radial de ritmo / cuenta regresiva
      sub_p = getattr(self, 'metro_subdiv_index', 0)
      n_sub = max(1, getattr(self, 'spin_metro_subdivs', None).value() if hasattr(self, 'spin_metro_subdivs') else 4)
      prog_compas = max(0.0, min(1.0, (sub_p + 1) / float(n_sub)))
      circ = 188.5
      offset_circ = circ * (1.0 - prog_compas)
      ring_col = "#FFFFFF" if flash else "#00FFFF"
      ring_svg = (
        f"<svg width='64' height='64' viewBox='0 0 70 70' style='vertical-align:middle; margin-right:12px;'>"
        f"<circle stroke='#122323' stroke-width='6' fill='transparent' r='30' cx='35' cy='35'/>"
        f"<circle stroke='{ring_col}' stroke-width='6' stroke-dasharray='{circ}' stroke-dashoffset='{offset_circ:.1f}' stroke-linecap='round' fill='transparent' r='30' cx='35' cy='35' transform='rotate(-90 35 35)'/>"
        f"<text x='35' y='41' text-anchor='middle' fill='{ring_col}' font-size='18' font-weight='bold'>{sub_p+1}</text>"
        f"</svg>"
      )

      html = (
        f"<div align='center' style='line-height:1.05;'>"
        f"{badge_top}<br>"
        f"<table align='center' style='margin:0 auto;'><tr>"
        f"<td>{ring_svg}</td>"
        f"<td><span style='{letra_style}'>/{p_act}/</span></td>"
        f"</tr></table>"
        f"<span>{tira_html}{sig_html}</span>"
        f"</div>"
      )
    self.lbl_recording_space.setText(html)
    inst_txt = (
      f"¡PRONUNCIA /{p_act}/ AHORA!" if flash
      else (f"ATENCIÓN: PRÓXIMO PULSO CAMBIA A /{p_sig}/" if (pre_cambio and p_sig) else f"MANTÉN LA VOCAL /{p_act}/ AL RITMO DEL METRÓNOMO")
    )
    sub_txt = f"Pulso {pulso_actual} de {total_pulsos} — Ciclo {ciclo_actual} de {total_ciclos}"
    self._publicar_estado_display_remoto(
      palabra=p_act,
      palabra_siguiente=p_sig,
      pulso_actual=pulso_actual,
      total_pulsos=total_pulsos,
      flash=flash,
      grabando=True,
      fase="FONACIÓN ACTIVA",
      instruccion_fase=inst_txt,
      subtitulo_fase=sub_txt,
      ciclo_actual=ciclo_actual,
      total_ciclos=total_ciclos,
      pre_cambio=pre_cambio,
      tira_html=tira_html,
    )

  def _disparar_destello_vocal(self, palabra_actual: str, pulso_actual: int, total_pulsos: int,
                               palabra_siguiente: str = "", ciclo_actual: int = 1, total_ciclos: int = 1):
    """Activa el destello intenso durante 200 ms cuando suena el pulso del metrónomo y luego vuelve al estado normal."""
    self._ultimo_estado_visor = (palabra_actual, pulso_actual, total_pulsos, palabra_siguiente, ciclo_actual, total_ciclos)
    self._renderizar_visor_vocal(
      palabra_actual, pulso_actual, total_pulsos,
      palabra_siguiente=palabra_siguiente, ciclo_actual=ciclo_actual, total_ciclos=total_ciclos,
      flash=True, pre_cambio=False
    )
    def _apagar_destello():
      if not getattr(self, 'is_recording', False):
        return
      st = getattr(self, '_ultimo_estado_visor', None)
      if st and st[1] == pulso_actual:
        self._renderizar_visor_vocal(
          st[0], st[1], st[2],
          palabra_siguiente=st[3], ciclo_actual=st[4], total_ciclos=st[5],
          flash=False, pre_cambio=False
        )
    QtCore.QTimer.singleShot(200, _apagar_destello)

  def on_metro_beat(self):
    if not self.metro_is_running:
        return

    # Instante exacto del disparo del pulso: se captura antes de cualquier
    # trabajo de UI o de encolar el beep, y se registra al final del beat.
    t_beat_epoch = time.time()
    t_beat_perf = time.perf_counter()
    sub_idx_actual = int(self.metro_subdiv_index)
    # Se captura antes del decremento: valor 1 = este es el ultimo tick de
    # count-in (el que arranca la toma), 0 = pulso real de la secuencia.
    count_in_actual = int(getattr(self, "metro_count_in_remaining", 0))

    bpm = max(1, self.spin_bpm.value())
    jitter_active = (
        hasattr(self, 'chk_metro_jitter')
        and self.chk_metro_jitter.isChecked()
        and not getattr(self, 'is_cvm_active', False)
        and self.metro_count_in_remaining <= 1
    )
    # En Modo Jitter activo durante fonación, suprimir sub-pulsos para que el sujeto
    # no pueda anticipar cuándo caerá el siguiente pulso principal.
    num_subdivs = 1 if jitter_active else max(1, self.spin_metro_subdivs.value())
    is_main_beat = (self.metro_subdiv_index == 0)

    if jitter_active and getattr(self, 'metro_jitter_schedule', None) and is_main_beat:
        sched_idx = max(0, self.metro_beat_count) % len(self.metro_jitter_schedule)
        self.current_beat_tau_s = float(self.metro_jitter_schedule[sched_idx])
    elif not jitter_active:
        self.current_beat_tau_s = 60.0 / float(bpm)

    sub_interval_s = self.current_beat_tau_s / float(num_subdivs)
    self._programar_siguiente_pulso(sub_interval_s)

    if is_main_beat:
        if self.metro_count_in_remaining > 1:
            cnt_val = self.metro_count_in_remaining - 1
            self.metro_lbl_title.setText("INICIANDO")
            self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #FF0000; font-weight: bold;")
            self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #FF0000; font-weight: bold;")
            self.metro_lbl_count.setText(str(cnt_val))
            self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_PREP}; border: 2px solid #FF0000;")
            if hasattr(self, 'countdown_text') and self.countdown_text.isVisible():
                self.countdown_text.setText(f"PREPÁRATE...\n{cnt_val}")
            # Mostrar también en el visor superior la cuenta regresiva y la primera vocal que viene
            if getattr(self, 'is_autoforge_running', False) and not getattr(self, 'is_cvm_active', False):
                words = getattr(self, 'autoforge_words', []) or ["A"]
                w_idx_prep = 0 if getattr(self, 'is_autoforge_continuo', False) else int(getattr(self, 'autoforge_word_idx', 0))
                primera_v = words[w_idx_prep % len(words)].upper()
                tira_prep = self._construir_tira_secuencia_html(words, w_idx_prep)
                self.lbl_recording_space.setStyleSheet(
                  "background-color: #1A0808; color: #FFCC00; font-family: 'Courier New', monospace; "
                  "border: 3px solid #FF4400; border-radius: 6px; padding: 4px;"
                )
                self.lbl_recording_space.setText(
                  f"<div align='center' style='line-height:1.1;'>"
                  f"<span style='color:#FF8800; font-size:16px; font-weight:900;'>CUENTA REGRESIVA DEL METRÓNOMO: {cnt_val}</span><br>"
                  f"<span style='color:#FFFFFF; font-size:48px; font-weight:900;'>PREPÁRATE PARA: /{primera_v}/</span><br>"
                  f"<span>{tira_prep}</span>"
                  f"</div>"
                )
                self._publicar_estado_display_remoto(
                  palabra=primera_v,
                  palabra_siguiente=primera_v,
                  pulso_actual=0,
                  total_pulsos=len(getattr(self, 'autoforge_full_sequence', words)),
                  count_in=cnt_val,
                  grabando=False,
                  fase=f"CUENTA REGRESIVA: {cnt_val}",
                  instruccion_fase=f"PREPÁRATE PARA PRONUNCIAR /{primera_v}/",
                  subtitulo_fase="Escucha los pulsos de aproximación y entra en el 1",
                  tira_html=tira_prep,
                )
            self.metro_count_in_remaining -= 1
            self.beep_queue.put((800, 200))
        else:
            if self.metro_count_in_remaining == 1:
                self.metro_beat_count = 1
                self.metro_count_in_remaining = 0
                self.beep_queue.put((1200, 400))
                cb_start = getattr(self, '_on_count_in_finished_cb', None)
                self._on_count_in_finished_cb = None
                if callable(cb_start):
                    cb_start(anchor_perf=t_beat_perf)
            else:
                self.metro_beat_count += 1
                # 1) Verificar cierre de Calibración CVM (18 pulsos) ANTES de emitir beep #19
                if getattr(self, 'is_cvm_active', False) and self.metro_beat_count > 18:
                    if getattr(self, 'is_cvm_recording_take', False):
                        self.estado_cvm_procesar_toma()
                    self.is_cvm_active = False
                    self.stop_native_metronome()
                    self.estado_finalizar_calibracion_cvm()
                    return
                # 2) Verificar cierre de Secuencia Continua ANTES de emitir beep extra
                if (
                    getattr(self, 'is_autoforge_running', False)
                    and getattr(self, 'is_autoforge_continuo', False)
                    and not getattr(self, 'is_cvm_active', False)
                    and getattr(self, 'is_recording', False)
                ):
                    total_pulsos_cont = len(self.autoforge_words) * self.autoforge_target_reps
                    if self.metro_beat_count > total_pulsos_cont:
                        if not getattr(self, 'is_finishing_sequence', False):
                            self.is_finishing_sequence = True
                            self.estado_guardar_secuencia_continua()
                        return
                # 3) Verificar cierre de Toma Discreta ANTES de emitir beep extra
                if (
                    getattr(self, 'is_autoforge_running', False)
                    and not getattr(self, 'is_autoforge_continuo', False)
                    and not getattr(self, 'is_cvm_active', False)
                    and getattr(self, 'is_recording', False)
                ):
                    if self.metro_beat_count > self.autoforge_target_reps:
                        self.estado_guardar_palabra()
                        return
                # Pulso Principal (1): Beep brillante y definido de fonación (1200 Hz, 110 ms)
                self.beep_queue.put((1200, 110))

            palabra_actual = ""
            if getattr(self, 'is_autoforge_running', False):
                words = getattr(self, 'autoforge_words', [])
                if getattr(self, 'is_autoforge_continuo', False):
                    full_seq = getattr(self, 'autoforge_full_sequence', None) or (words * max(1, self.autoforge_target_reps))
                    idx = (self.metro_beat_count - 1) % len(full_seq) if full_seq else 0
                    self.last_continuo_beat = idx
                    palabra_actual = full_seq[idx].upper() if full_seq else ""
                else:
                    w_idx = getattr(self, 'autoforge_word_idx', 0)
                    palabra_actual = words[w_idx].upper() if words and w_idx < len(words) else ""
                
                self.metro_lbl_title.setText(f"VOCAL /{palabra_actual}/ (P{self.metro_beat_count})")
                self.metro_lbl_title.setStyleSheet("font-size: 15px; color: #00FFCC; font-weight: bold;")
            else:
                self.metro_lbl_title.setText(f"PULSO {self.metro_beat_count}")
                self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #00FFFF;")
            
            display_num = "1" if num_subdivs > 1 else str(self.metro_beat_count)
            self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #00FFFF; font-weight: bold;")
            self.metro_lbl_count.setText(display_num)
            self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_BEAT}; border: 2px solid #00FFFF;")

            # Hook fotometria: captura segura sincronizada con pulso de metronomo
            if getattr(self, 'is_recording', False):
                self._safe_photometry_snapshot(
                    fase="vocal_metronomo",
                    palabra=str(palabra_actual if getattr(self, 'is_autoforge_running', False) else ""),
                    pulso=int(self.metro_beat_count),
                )
            
            # --- Integración de Calibración CVM con Metrónomo (18 pulsos: 9 preparación, 9 contracción) ---
            if getattr(self, 'is_cvm_active', False):
                if self.metro_beat_count % 2 != 0:
                    # IMPAR: Fase de Preparación
                    if getattr(self, 'is_cvm_recording_take', False):
                        self.estado_cvm_procesar_toma()
                    
                    self.cvm_step_idx = (self.metro_beat_count - 1) // 2
                    vocal, toma = self.cvm_sequence[self.cvm_step_idx]
                    instrucciones = {
                        'A': "Abre la boca y pronuncia /A/ con fuerza",
                        'I': "Sonríe y pronuncia /I/ con fuerza",
                        'U': "Redondea y protruye los labios con /U/ con fuerza"
                    }
                    desc = instrucciones.get(vocal, "")
                    
                    self.autoforge_overlay.setText(
                        f"<div align='center'>"
                        f"<span style='color:#FFCC00; font-size:32px; font-weight:bold;'>CONTRACCIÓN MUSCULAR MÁXIMA</span><br>"
                        f"<span style='color:#CCCCCC; font-size:18px;'>Toma {self.cvm_step_idx+1}/9</span><br><br>"
                        f"<span style='color:#FFFF00; font-size:40px; font-weight:bold;'>PREPÁRATE PARA /{vocal}/</span><br>"
                        f"<span style='font-size:22px; color:#AAFFAA; font-weight:bold;'>{desc}</span>"
                        f"</div>"
                    )
                    self.lbl_recording_space.setText(f"<div align='center'><span style='color:#FFFF00; font-size:40px; font-weight:bold;'>PREPÁRATE: /{vocal}/</span></div>")
                    self.autoforge_overlay.show()
                    # En CVM sólo anunciar en la preparación de la Toma 1 de cada vocal nueva (evita repetir "A, A, A")
                    if int(toma) == 1 and hasattr(self, 'voice_cue_player'):
                        self.voice_cue_player.say(f"di_{vocal.lower()}", f"Prepárate, di {vocal} con fuerza", fase="cvm_preparacion")
                else:
                    # PAR: Fase de Grabación (Toma)
                    vocal, toma = self.cvm_sequence[self.cvm_step_idx]
                    self.autoforge_overlay.setText(
                        f"<div align='center'>"
                        f"<span style='color:#FF0000; font-size:32px; font-weight:bold;'>¡CONTRAE AHORA!</span><br>"
                        f"<span style='color:#00FFFF; font-size:90px; font-weight:bold;'>/{vocal}/</span><br>"
                        f"</div>"
                    )
                    self.lbl_recording_space.setText(f"<div align='center'><span style='color:#FFFFFF; background-color:#FF0055; font-size:70px; font-weight:bold; padding:4px 20px; border:3px solid #FFFFFF;'>/{vocal}/</span></div>")
                    self.autoforge_overlay.show()
                    
                    self.cvm_current_take_chunks = []
                    self.is_cvm_recording_take = True

                    self._safe_photometry_snapshot(fase="cvm", palabra=vocal, pulso=toma)
            
            # --- Destello Visual y Guía de Secuencia en Fonación (Secuencia Continua y Discreta) ---
            elif getattr(self, 'is_autoforge_running', False) and getattr(self, 'is_recording', False):
                words = getattr(self, 'autoforge_words', []) or ["A"]
                n_w = len(words)
                if getattr(self, 'is_autoforge_continuo', False):
                    full_seq = getattr(self, 'autoforge_full_sequence', None) or (words * max(1, self.autoforge_target_reps))
                    total_pulsos = len(full_seq)
                    idx = (self.metro_beat_count - 1) % total_pulsos
                    palabra_actual = full_seq[idx].upper()
                    palabra_sig = full_seq[idx + 1].upper() if (idx + 1) < total_pulsos else "FIN"
                    ciclo_act = ((self.metro_beat_count - 1) // n_w) + 1
                    total_ciclos = self.autoforge_target_reps
                else:
                    total_pulsos = self.autoforge_target_reps
                    w_idx = int(getattr(self, 'autoforge_word_idx', 0))
                    palabra_actual = words[w_idx % n_w].upper()
                    if self.metro_beat_count < total_pulsos:
                        palabra_sig = palabra_actual
                    elif (w_idx + 1) < n_w:
                        palabra_sig = words[w_idx + 1].upper()
                    else:
                        palabra_sig = "FIN"
                    ciclo_act = w_idx + 1
                    total_ciclos = n_w

                self._disparar_destello_vocal(
                    palabra_actual=palabra_actual,
                    pulso_actual=self.metro_beat_count,
                    total_pulsos=total_pulsos,
                    palabra_siguiente=palabra_sig,
                    ciclo_actual=ciclo_act,
                    total_ciclos=total_ciclos
                )
                # Locución inteligente:
                # - En Autograbado Discreto NO se repite "A, A, A, A" en cada pulso (ya se anunció en la fase PREPÁRATE).
                # - En Secuencia Continua, si num_subdivs > 1, el "Di <Vocal>" ya se adelantó en el último sub-pulso previo
                #   para no pisar la voz del sujeto; si num_subdivs == 1 (ej. Jitter activo), se dice aquí cuando cambia la vocal.
                if getattr(self, 'is_autoforge_continuo', False) and hasattr(self, 'voice_cue_player'):
                    if getattr(self, '_pre_announced_continuo_beat', -1) != self.metro_beat_count:
                        if palabra_actual != getattr(self, '_last_spoken_continuo_word', None):
                            self._last_spoken_continuo_word = palabra_actual
                            self.voice_cue_player.say(f"di_{palabra_actual.lower()}", f"Di {palabra_actual}", fase="pulso_continua")
                if hasattr(self, 'countdown_text'):
                    self.countdown_text.hide()
        
        QtCore.QTimer.singleShot(60, self.reset_metro_color)
    else:
        # --- SUB-PULSOS MUSICALES CON RAMPA ACÚSTICA DE APROXIMACIÓN (2, 3, 4...) ---
        # Cada sub-pulso sube de frecuencia (ej. 520 Hz -> 680 Hz -> 880 Hz) para que el sujeto
        # sienta auditivamente cómo se acerca el golpe principal (1200 Hz) y articule a tiempo.
        sub_num = self.metro_subdiv_index + 1
        frac_sub = float(sub_num - 1) / float(max(1, num_subdivs - 1))
        freq_sub = int(round(520.0 + 360.0 * frac_sub))  # 520 Hz -> 680 Hz -> 880 Hz
        dur_sub = int(round(35.0 + 20.0 * frac_sub))     # 35 ms -> 45 ms -> 55 ms
        es_ultimo_sub = (sub_num == num_subdivs)

        if self.metro_count_in_remaining > 0:
            self.metro_pulse_frame.setStyleSheet("background-color: #660000; border: 2px solid #FF0000;")
            self.beep_queue.put((freq_sub, dur_sub))
        else:
            if es_ultimo_sub:
                # Último sub-pulso antes del golpe principal: color ámbar de pre-aviso
                self.metro_pulse_frame.setStyleSheet("background-color: #FFAA00; border: 2px solid #FFFF00;")
                self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #FFCC00; font-weight: bold;")
            else:
                self.metro_pulse_frame.setStyleSheet("background-color: #006677; border: 2px solid #00FFFF;")
                self.metro_lbl_count.setStyleSheet("font-size: 28px; color: #00AAAA; font-weight: bold;")
            self.metro_lbl_count.setText(str(sub_num))
            self.beep_queue.put((freq_sub, dur_sub))
            # Actualizar el anillo circular radial en cada sub-pulso
            st_ring = getattr(self, '_ultimo_estado_visor', None)
            if st_ring and getattr(self, 'is_recording', False) and not es_ultimo_sub:
                self._renderizar_visor_vocal(
                    st_ring[0], st_ring[1], st_ring[2],
                    palabra_siguiente=st_ring[3], ciclo_actual=st_ring[4], total_ciclos=st_ring[5],
                    flash=False, pre_cambio=False
                )
            # En el último sub-pulso de Secuencia Continua, pre-avisar visualmente Y adelantar el "Di <Vocal>"
            # para que la voz de la PC termine antes del golpe principal y no pise el inicio de fonación del sujeto
            if (
                sub_num == num_subdivs
                and getattr(self, 'is_autoforge_running', False)
                and getattr(self, 'is_autoforge_continuo', False)
                and not getattr(self, 'is_cvm_active', False)
                and getattr(self, 'is_recording', False)
            ):
                st = getattr(self, '_ultimo_estado_visor', None)
                if st:
                    self._renderizar_visor_vocal(
                        st[0], st[1], st[2],
                        palabra_siguiente=st[3], ciclo_actual=st[4], total_ciclos=st[5],
                        flash=False, pre_cambio=True
                    )
                    next_word = str(st[3] or "").upper()
                    next_beat = int(self.metro_beat_count) + 1
                    if (
                        next_word
                        and next_word != "FIN"
                        and hasattr(self, 'voice_cue_player')
                        and next_word != getattr(self, '_last_spoken_continuo_word', None)
                    ):
                        self._last_spoken_continuo_word = next_word
                        self._pre_announced_continuo_beat = next_beat
                        self.voice_cue_player.say(f"di_{next_word.lower()}", f"Di {next_word}", fase="pulso_continua")
            
        QtCore.QTimer.singleShot(60, self.reset_metro_color)
        
    self._registrar_pulso_metronomo(t_beat_epoch, t_beat_perf, sub_idx_actual, bpm, num_subdivs, count_in_actual)
    self.metro_subdiv_index = (self.metro_subdiv_index + 1) % num_subdivs

  def _programar_siguiente_pulso(self, intervalo_s):
    """Programa el proximo pulso sobre un reloj monotono absoluto.

    Reiniciar el QTimer dentro del callback acumula el costo de proceso de cada
    beat (se observaron ~14 ms por pulso a 120 BPM, ~1,7 s de deriva por minuto).
    Aca el objetivo es siempre `anterior + intervalo`, asi el error no se suma.
    """
    try:
      ahora = time.perf_counter()
      intervalo = max(0.005, float(intervalo_s))
      siguiente = getattr(self, "_metro_next_perf", None)
      if siguiente is None:
        siguiente = ahora
      siguiente += intervalo
      restante = siguiente - ahora
      if restante < 0:
        # Atraso grande (ventana bloqueada): re-anclar en vez de correr detras.
        siguiente = ahora
        restante = 0.0
      self._metro_next_perf = siguiente
      self.metro_timer.setSingleShot(True)
      self.metro_timer.start(max(0, int(round(restante * 1000.0))))
    except Exception as e:
      print(f"[Metronomo] Error al programar el proximo pulso: {e}")

  def _contar_muestras_recording(self):
    """Muestras ya presentes en current_recording (pre-roll / ruido previo).

    El .wav final concatena esos datos ANTES del instante ancla, asi que el
    indice de muestra de un pulso dentro del archivo lleva este desplazamiento.
    """
    total = 0
    for ch in (getattr(self, "current_recording", None) or []):
      try:
        total += int(ch.shape[1])
      except Exception:
        continue
    return total

  def _registrar_pulso_metronomo(self, t_epoch, t_perf, sub_idx, bpm, num_subdivs, count_in=None):
    """Imprime y acumula el tiempo exacto de cada pulso del metronomo.

    Emite una linea por pulso con epoch, perf_counter (el mismo reloj que las
    anclas t0_perf_counter de metadata), el tiempo relativo al ancla de
    grabacion y el indice de muestra dentro del .wav final. Al guardar la
    sesion se escribe pulsos_metronomo.csv en la carpeta de la medicion.
    """
    try:
      es_principal = (int(sub_idx) == 0)
      self._metro_pulso_seq = int(getattr(self, "_metro_pulso_seq", 0)) + 1
      t_daq_s = None
      t_wav_s = None
      sample_daq = None
      offset = int(getattr(self, "_offset_muestras_ancla", 0) or 0)
      sr = float(getattr(self, "SAMPLE_RATE", 0) or 0)
      if getattr(self, "is_recording", False) and getattr(self, "recording_start_time", None) is not None:
        t_daq_s = float(t_perf - self.recording_start_time)
        if sr > 0:
          sample_daq = int(round(t_daq_s * sr)) + offset
          t_wav_s = sample_daq / sr
        if es_principal and int(getattr(self, "metro_beat_count", 0)) >= 1:
          if sample_daq is not None:
            self.metro_beat_sample_indices.append(int(sample_daq))
          if t_wav_s is not None:
            self.metro_beat_timestamps_s.append(round(float(t_wav_s), 6))
          self.metro_applied_intervals_s.append(round(float(getattr(self, "current_beat_tau_s", 60.0 / max(1, bpm))), 6))
      palabra = ""
      if getattr(self, "is_autoforge_running", False):
        words = getattr(self, "autoforge_words", []) or []
        if getattr(self, "is_autoforge_continuo", False):
          full_seq = getattr(self, "autoforge_full_sequence", None) or words
          idx_w = (self.metro_beat_count - 1) % len(full_seq) if full_seq else 0
          palabra = full_seq[idx_w].upper() if full_seq and 0 <= idx_w < len(full_seq) else ""
        else:
          idx_w = int(getattr(self, "autoforge_word_idx", 0))
          palabra = words[idx_w].upper() if words and 0 <= idx_w < len(words) else ""
      jit_enabled = bool(hasattr(self, "chk_metro_jitter") and self.chk_metro_jitter.isChecked())
      jit_mode = self.cmb_metro_jitter_mode.currentText() if (jit_enabled and hasattr(self, "cmb_metro_jitter_mode")) else "off"
      intervalo_actual = round(float(getattr(self, "current_beat_tau_s", 60.0 / max(1, bpm))), 6)
      registro = {
        "seq": self._metro_pulso_seq,
        "idx": int(getattr(self, "metro_beat_count", 0)),
        "sub": int(sub_idx),
        "principal": int(es_principal),
        "count_in": int(count_in) if count_in is not None else int(getattr(self, "metro_count_in_remaining", 0)),
        "t_epoch": round(float(t_epoch), 6),
        "t_perf": round(float(t_perf), 6),
        "t_daq_s": None if t_daq_s is None else round(t_daq_s, 6),
        "t_wav_s": None if t_wav_s is None else round(t_wav_s, 6),
        "sample_daq": sample_daq,
        "offset_muestras": offset if sample_daq is not None else None,
        "bpm": int(bpm),
        "subdiv": int(num_subdivs),
        "intervalo_s": intervalo_actual,
        "jitter_activo": int(jit_enabled),
        "jitter_mode": jit_mode,
        "palabra": palabra,
      }
      # Si es un pulso principal donde suena una vocal durante grabación (con o sin Jitter),
      # guardarlo en vocal_cue_log para medir exactamente cuán sincronizado le pegó el sujeto.
      if es_principal and getattr(self, "is_recording", False) and int(getattr(self, "metro_beat_count", 0)) >= 1:
        if not hasattr(self, "vocal_cue_log") or self.vocal_cue_log is None:
          self.vocal_cue_log = []
        self.vocal_cue_log.append({
          "pulso_vocal_idx": int(self.metro_beat_count),
          "palabra": palabra,
          "t_epoch": registro["t_epoch"],
          "t_perf": registro["t_perf"],
          "t_daq_s": registro["t_daq_s"],
          "t_wav_s": registro["t_wav_s"],
          "sample_daq": sample_daq,
          "intervalo_s": intervalo_actual,
          "jitter_activo": bool(jit_enabled),
          "jitter_mode": jit_mode,
        })
      pulsos = getattr(self, "_metro_pulsos", None)
      if pulsos is None:
        pulsos = []
        self._metro_pulsos = pulsos
      pulsos.append(registro)
      t_daq_txt = "-" if t_daq_s is None else f"{t_daq_s:.6f}"
      sample_txt = "-" if sample_daq is None else str(sample_daq)
      print(
        f"[METRONOMO] #{registro['seq']} principal={registro['principal']} idx={registro['idx']} "
        f"sub={registro['sub']} t_epoch={registro['t_epoch']:.6f} t_perf={registro['t_perf']:.6f} "
        f"t_daq_s={t_daq_txt} sample_daq={sample_txt} offset={registro['offset_muestras']} "
        f"bpm={registro['bpm']} subdiv={registro['subdiv']} tau={registro['intervalo_s']:.3f}s"
        + (f" palabra={palabra}" if palabra else ""),
        flush=True,
      )
    except Exception as e:
      print(f"[Metronomo] Error al registrar pulso: {e}")

  def _get_plot_y_center(self):
    """Calcula dinámicamente el centro vertical visible del osciloscopio."""
    try:
      if hasattr(self, 'plot') and self.plot is not None:
        vb = self.plot.getViewBox()
        if vb is not None:
          vr = vb.viewRange()
          if vr and len(vr) > 1 and len(vr[1]) >= 2:
            return float(0.5 * (vr[1][0] + vr[1][1]))
    except Exception:
      pass
    if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
      return float(getattr(self, '_held_plot_ymax', 100.0) * 0.5)
    return 0.0

  # --- NUEVO: FUNCIONES DEL FILTRO ---
  def on_filter_settings_changed(self):
    """Se llama al cambiar cualquier ajuste del filtro. Rediseña el filtro y resetea su estado."""
    is_filter_enabled = self.chk_filter_enable.isChecked()
    self.spin_low_cut.setEnabled(is_filter_enabled)
    self.spin_high_cut.setEnabled(is_filter_enabled)

    # --- Lógica para el filtro Pasa-Banda ---
    if is_filter_enabled and self.is_acquiring and SCIPY_DISPONIBLE:
      low_cut = self.spin_low_cut.value()
      high_cut = self.spin_high_cut.value()

      # Validación simple de frecuencias
      if low_cut >= high_cut:
        print(f"Advertencia de filtro: Frec. Baja ({low_cut} Hz) debe ser menor que Frec. Alta ({high_cut} Hz).")
        self.filter_sos = None
        self.filter_zi = None
      else:
        nyquist = 0.5 * self.SAMPLE_RATE
        low = low_cut / nyquist
        
        # Asegurar que la frecuencia alta sea estrictamente menor a Nyquist (Wn < 1)
        if high_cut >= nyquist:
          high_cut = nyquist * 0.99
        high = high_cut / nyquist
        
        self.filter_sos = signal.butter(self.FILTER_ORDER, [low, high], btype='band', output='sos')
        # --- MEJORA: Crear un estado inicial 'zi' para CADA canal ---
        # El estado inicial para un canal tiene forma (n_sections, 2).
        # Para N canales, la forma correcta del array de estados es (n_sections, 2, N_CANALES).
        zi_single_channel = signal.sosfilt_zi(self.filter_sos)
        # Replicamos el estado para cada canal en la dimensión correcta.
        self.filter_zi = np.stack([zi_single_channel] * self.NUM_CANALES, axis=-1)
    else:
      # --- Deshabilitar filtro pasa-banda ---
      self.filter_sos = None
      self.filter_zi = None

    # --- NUEVO: Lógica para el filtro Notch ---
    is_notch_enabled = self.chk_notch_enable.isChecked()
    if is_notch_enabled and self.is_acquiring and SCIPY_DISPONIBLE:
      # Diseñar filtro Notch para 50 Hz
      f0 = 50.0 # Frecuencia a remover
      Q = float(self.spin_notch_q.value()) if hasattr(self, 'spin_notch_q') else 2.0
      
      # --- CORRECCIÓN DE COMPATIBILIDAD ---
      # La versión de SciPy del usuario no soporta `output='sos'`.
      # Se genera el filtro en formato (b, a) y se convierte a SOS.
      b, a = signal.iirnotch(f0, Q, fs=self.SAMPLE_RATE)
      sos_list = [signal.tf2sos(b, a)]

      if hasattr(self, 'chk_notch_comb') and self.chk_notch_comb.isChecked():
        nyquist = 0.5 * self.SAMPLE_RATE
        for harmonic in [150.0, 250.0, 350.0]:
          if harmonic < nyquist:
            b_h, a_h = signal.iirnotch(harmonic, Q, fs=self.SAMPLE_RATE)
            sos_list.append(signal.tf2sos(b_h, a_h))

      self.notch_sos = np.vstack(sos_list)
      zi_notch_single = signal.sosfilt_zi(self.notch_sos)
      self.notch_zi = np.stack([zi_notch_single] * self.NUM_CANALES, axis=-1)
    else:
      # --- Deshabilitar filtro Notch ---
      self.notch_sos = None
      self.notch_zi = None
      
    # --- NUEVO: Lógica para el filtro Envolvente RMS (Fluidez extrema) ---
    is_rms_enabled = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
    if is_rms_enabled and self.is_acquiring and SCIPY_DISPONIBLE:
      # Filtro pasa-bajos simple (2do orden, 5Hz) para envolvente visual fluida sin sobrecarga
      self.env_sos = signal.butter(2, 5.0, btype='low', fs=self.SAMPLE_RATE, output='sos')
      zi_single_env = signal.sosfilt_zi(self.env_sos)
      self.env_zi = np.stack([zi_single_env] * self.NUM_CANALES, axis=-1)
    else:
      self.env_sos = None
      self.env_zi = None

    self._init_filter_state = True # <-- NUEVO: Bandera para eliminar el pico transitorio inicial

  # --- FUNCIONES v3.7 (Trigger) ---
  def on_trigger_enable_toggle(self):
    """Habilita o deshabilita la lógica del trigger y la UI."""
    enabled = self.chk_trigger.isChecked()
    self.trigger_line.setVisible(enabled)
    self.cmb_trig_chan.setEnabled(enabled)
    self.spin_trig_level.setEnabled(enabled)
    self.cmb_trig_edge.setEnabled(enabled)

  def on_trigger_line_moved(self, line):
    """Se llama cuando el usuario arrastra la línea roja."""
    # --- NUEVO: Mostrar y actualizar la etiqueta de valor ---
    y_pos = line.value()
    self.trigger_label.setText(f"Nivel: {y_pos:.1f} µV")
    # Colocar la etiqueta cerca del cursor, en el borde derecho del plot
    x_range = self.plot.getViewBox().viewRange()[0]
    self.trigger_label.setPos(x_range[1] * 0.95, y_pos) # 95% a la derecha
    self.trigger_label.show()

    self.is_trigger_line_moving = True
    self.spin_trig_level.setValue(y_pos)
    self.is_trigger_line_moving = False

  def on_trigger_level_changed(self, value):
    """Se llama cuando el usuario cambia el SpinBox."""
    if not self.is_trigger_line_moving:
      self.trigger_line.setPos(value)
      
  def on_peak_th_line_pos_moved(self, line):
    """Se llama al arrastrar la línea de umbral superior (celeste)."""
    val = max(0.001, line.value())
    self.spin_peak_th.blockSignals(True)
    self.spin_peak_th.setValue(val)
    self.spin_peak_th.blockSignals(False)
    self.peak_th_line_neg.blockSignals(True)
    self.peak_th_line_neg.setPos(-val)
    self.peak_th_line_neg.blockSignals(False)

  def on_peak_th_line_neg_moved(self, line):
    """Se llama al arrastrar la línea de umbral inferior (celeste)."""
    val = max(0.001, -line.value())
    self.spin_peak_th.blockSignals(True)
    self.spin_peak_th.setValue(val)
    self.spin_peak_th.blockSignals(False)
    self.peak_th_line_pos.blockSignals(True)
    self.peak_th_line_pos.setPos(val)
    self.peak_th_line_pos.blockSignals(False)

  def on_peak_th_changed(self, value):
    """Se llama al modificar el SpinBox de Umbral Picos SNR."""
    self.peak_th_line_pos.blockSignals(True)
    self.peak_th_line_pos.setPos(value)
    self.peak_th_line_pos.blockSignals(False)
    self.peak_th_line_neg.blockSignals(True)
    self.peak_th_line_neg.setPos(-value)
    self.peak_th_line_neg.blockSignals(False)

  def check_for_trigger(self, new_data, num_new_samples):
    """Escanea el último chunk de datos en busca de un evento de trigger."""
    
    if not self.is_acquiring or self.NUM_CANALES == 0:
      return

    if getattr(self, 'is_autoforge_running', False):
      self.trigger_last_values = self.plot_buffer_datos[:, -1]
      return # Jamás congelar pantalla por trigger durante secuencia de autograbado

    if not self.chk_trigger.isChecked() or not self.chk_autoscroll.isChecked():
      self.trigger_last_values = self.plot_buffer_datos[:, -1]
      return

    level = self.spin_trig_level.value()
    chan_idx = self.cmb_trig_chan.currentIndex() # Puede ser -1 si no hay canales
    edge_is_rising = (self.cmb_trig_edge.currentIndex() == 0)
    
    prev_val = self.trigger_last_values[chan_idx]
    signal_chunk = new_data[chan_idx]
    
    all_vals = np.insert(signal_chunk, 0, prev_val)
    prev_samples = all_vals[:-1]
    curr_samples = all_vals[1:]

    if edge_is_rising: # Flanco de Subida
      crossings = (prev_samples < level) & (curr_samples >= level)
    else: # Flanco de Bajada
      crossings = (prev_samples > level) & (curr_samples <= level)
    
    if np.any(crossings):
      self.trigger_fired()
      
    self.trigger_last_values = self.plot_buffer_datos[:, -1]

  def trigger_fired(self):
    """¡Se detectó un trigger! Congela y centra el gráfico."""
    print(f"¡TRIGGER DETECTADO! (t=0)")

    # Hook fotometria: captura segura al detectar onset por trigger
    self._safe_photometry_snapshot(fase="onset", palabra="trigger", pulso=0)
    
    # 1. Congela el gráfico
    self.chk_autoscroll.setChecked(False)
    self.on_autoscroll_toggle()
    
    # 2. Centra el gráfico en una ventana de 2 segundos ANTES del trigger
    self.plot.setXRange(-2.0, 0.0, padding=0)
    
    # 3. Feedback visual
    self.plot_widget.setBackground('#400000') # Rojo oscuro

  # --- FUNCIONES MODIFICADAS ---
  
  def on_autoscroll_toggle(self):
    """
    Restaura el fondo a negro ('k') en lugar de 'None'.
    """
    if self.chk_autoscroll.isChecked():
      # MODO ROLL (Armado)
      self.plot.setMouseEnabled(x=False, y=True) # Solo zoom Y
      self.plot.getViewBox().disableAutoRange(pg.ViewBox.XAxis) # Desactiva auto-range en X
      self.plot.getViewBox().disableAutoRange(pg.ViewBox.YAxis) # Desactiva auto-range en Y para el Peak-Hold
      self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)
      for sp in getattr(self, 'sub_plots', []):
        sp.setMouseEnabled(x=False, y=True)
        sp.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
        sp.getViewBox().disableAutoRange(pg.ViewBox.YAxis)
      self.plot_widget.setBackground('k') # Color de fondo negro
    else:
      # MODO ANÁLISIS (Congelado)
      self.plot.setMouseEnabled(x=True, y=True) # Zoom X e Y
      self.plot.getViewBox().enableAutoRange(pg.ViewBox.XAxis)
      self.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis) # Auto-escala todo al congelar
      for sp in getattr(self, 'sub_plots', []):
        sp.setMouseEnabled(x=True, y=True)
        sp.getViewBox().enableAutoRange(pg.ViewBox.XAxis)
        sp.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)

  def on_record_click(self):
    """
    Ejecuta la funcionalidad de on_record_click.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if not self.is_recording:
      # --- HOMOLOGACIÓN CON MANUAL_DAQ: Si no está adquiriendo, iniciar adquisición automáticamente ---
      if not getattr(self, 'is_acquiring', False):
        self.start_acquisition()
        QtWidgets.QApplication.processEvents()
        if not getattr(self, 'is_acquiring', False):
          return

      # --- EMPEZAR A GRABAR MANUALMENTE ---
      self.is_recording = True
      self.recording_start_time = time.perf_counter()
      self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.final_raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self._metro_pulsos = []
      self.vocal_cue_log = []
      self._safe_photometry_start_session("grabacion_manual")

      if hasattr(self, 'btn_record'):
        self.btn_record.setText("■ DETENER Y GUARDAR (Manual)")
        self.btn_record.setStyleSheet(self.BTN_REC_STOP_STYLE)

      # --- Resetear el zoom en Y a rango base calibrado (500 µV) ---
      if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
        self.plot.setYRange(0.0, 500.0, padding=0)
      else:
        self.plot.setYRange(-500.0, 500.0, padding=0)
      self._held_plot_ymax = 500.0

      self.counting_started = False
      # Homologado con manual_daq.py: vaciar buffers previos para que la toma manual empiece en t=0.0s exacto
      self.current_recording.clear()
      if hasattr(self, 'preroll_buffer') and hasattr(self.preroll_buffer, 'clear'):
        try:
          self.preroll_buffer.clear()
        except Exception:
          pass
      # El .wav empieza con este pre-roll: desplazamiento de muestras del ancla.
      self._offset_muestras_ancla = self._contar_muestras_recording()

      if hasattr(self, 'btn_pause'):
        self.btn_pause.setEnabled(True)
      self.cmb_terminal_config.setEnabled(False) # <-- NUEVO: Bloquear cambio en plena grabación
      
      # --- NUEVO: Reiniciar variables de ruido y buffers ---
      self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.noise_levels = [0.0] * self.NUM_CANALES
      self.noise_calculated = False
      self.noise_initialized = False # <-- NUEVO: Resetear estado
      self.countdown_active = False # Permitir lanzar el countdown de nuevo
      self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
      self.rt_last_noise_eval = [0.0] * self.NUM_CANALES
      for line in getattr(self, 'noise_lines', []): line.hide()
      for line in getattr(self, 'noise_lines_neg', []): line.hide()
      for reg in getattr(self, 'noise_regions', []): reg.hide()
      for region in getattr(self, 'dynamic_noise_regions', []): region.hide()
      
      # --- NUEVO: Reiniciar tester de ruido ---
      self.initial_noise_mean = [0.0] * self.NUM_CANALES
      self.initial_noise_std = [0.0] * self.NUM_CANALES
      self.last_phase = 0.0
      
      # --- NUEVO: Variables para gráfico final de estadísticas ---
      self.stats_time = [[] for _ in range(self.NUM_CANALES)]
      self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
      self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
      self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
      
      # --- NUEVO: Acumulador Global para el UI ---
      self.global_snr_acumulado = [0.0] * self.NUM_CANALES
      self.global_snr_count = [0] * self.NUM_CANALES
      
      if hasattr(self, 'noise_status_labels'):
        for label in self.noise_status_labels:
          label.setText("Ruido inter-pulso: Evaluando base...")
          label.setStyleSheet("color: gray; font-size: 11px; background-color: transparent;")
      
      # Mantener paneles de configuracion (filtros, envolvente, trigger) siempre visibles
      self.config_stack.setCurrentIndex(0)
      
      # --- NUEVO: Re-lanzar el metrónomo con count-in (mismo beep que autograbado) al inicio de grabar ruido ---
      if self.chk_use_metronome.isChecked():
        self.stop_native_metronome()
        self.start_native_metronome(count_in=4, force_start=True)

    else:
      # --- DETENER GRABAR MANUALMENTE ---
      self.is_recording = False
      if hasattr(self, 'btn_record'):
        self.btn_record.setText("GRABAR LIBRE (Manual)")
        self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)
      if hasattr(self, 'btn_pause'):
        self.btn_pause.setEnabled(False)
        if self.btn_pause.isChecked():
          self.btn_pause.setChecked(False)
      self.cmb_terminal_config.setEnabled(True)
      self.stop_native_metronome()
      self.config_stack.setCurrentIndex(0)
      self._persistir_preferencias_en_config_local()

      if self.current_recording:
        self.on_export_click() # <-- LLAMADA AUTOMÁTICA A EXPORTAR (AHORA DESPUÉS DE CERRAR EL METRÓNOMO)

  # --- MODIFICADO v3.12: Manejo de errores de exportación individual ---
  def on_export_click(self):
    # --- MODIFICADO: Usar el nuevo diálogo personalizado ---
    """
    Ejecuta la funcionalidad de on_export_click.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    dialog = SaveMeasurementDialog(self)
    result = dialog.exec() # Esto muestra el diálogo y espera

    if result == QtWidgets.QDialog.Accepted:
      measurement_name = dialog.measurement_name
      # --- NUEVO: Capturar los detalles del diálogo ---
      is_formal = dialog.es_formal
      details = {"sujeto": dialog.sujeto, "letra": dialog.letra, "prueba": dialog.prueba}
      comentario = dialog.comentario
    else:
      measurement_name = None # El usuario canceló

    if not measurement_name:
      return # El usuario canceló

    # Crear la estructura de directorios
    import os
    if getattr(sys, 'frozen', False):
      root_dir = os.path.dirname(os.path.abspath(sys.executable))
      if os.path.basename(root_dir) == "_internal":
        root_dir = os.path.dirname(root_dir)
    else:
      root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.join(root_dir, "base_de_datos_electrodos")
    # --- NUEVO: Crear carpeta de fecha ---
    today_str = datetime.now().strftime('%Y-%m-%d')
    date_dir = os.path.join(base_dir, today_str)
    output_dir = os.path.join(date_dir, measurement_name)
    try:
      os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
      print(f"--- ERROR FATAL AL CREAR DIRECTORIO ---\n{e}")
      return

    # --- NUEVO: Crear las carpetas de los canales desde el principio ---
    for i in range(self.NUM_CANALES):
      channel_output_dir = os.path.join(output_dir, f"canal_{i}")
      os.makedirs(channel_output_dir, exist_ok=True)
    print(f"  Creadas {self.NUM_CANALES} carpetas de canal en '{output_dir}'")

    print(f"\n--- INICIANDO EXPORTACIÓN a la carpeta '{output_dir}' ---")
    
    if not self.current_recording:
      print("Error: No hay datos grabados para exportar.")
      return
    
    # --- NUEVO: Leer el conteo de pulsos desde el metrónomo nativo ---
    pulse_count_from_metronome = None
    if self.chk_use_metronome.isChecked():
      pulse_count_from_metronome = self.metro_beat_count
      print(f" Guardados {pulse_count_from_metronome} pulsos en el metadata.")
    else:
      pulse_count_from_metronome = 0

    now = datetime.now()
    muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
    muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}

    calib_cvm_dict = {
        "calibrado": getattr(self, 'cvm_completed', False),
        "metodo": "envolvente_rms_multiparametrico",
        "tomas_por_vocal": 3,
        "canales": {
            f"canal_{i}": {
                "pico_envolvente_uV": self.cvm_env_max_per_channel[i] if hasattr(self, 'cvm_env_max_per_channel') and i < len(self.cvm_env_max_per_channel) else 0.0,
                "percentil_95_uV": self.cvm_env_p95_per_channel[i] if hasattr(self, 'cvm_env_p95_per_channel') and i < len(self.cvm_env_p95_per_channel) else 0.0,
                "percentil_90_uV": self.cvm_env_p90_per_channel[i] if hasattr(self, 'cvm_env_p90_per_channel') and i < len(self.cvm_env_p90_per_channel) else 0.0,
                "rms_envolvente_uV": self.cvm_env_rms_per_channel[i] if hasattr(self, 'cvm_env_rms_per_channel') and i < len(self.cvm_env_rms_per_channel) else 0.0,
                "energia_uV2_s": self.cvm_energia_per_channel[i] if hasattr(self, 'cvm_energia_per_channel') and i < len(self.cvm_energia_per_channel) else 0.0,
                "auc_uV_s": self.cvm_auc_per_channel[i] if hasattr(self, 'cvm_auc_per_channel') and i < len(self.cvm_auc_per_channel) else 0.0
            } for i in range(self.NUM_CANALES)
        },
        "supremo_global_uV": getattr(self, 'supremo_global_uV', 0.0),
        "supremo_global_p95_uV": getattr(self, 'supremo_global_p95_uV', 0.0),
        "supremo_global_p90_uV": getattr(self, 'supremo_global_p90_uV', 0.0),
        "supremo_global_rms_uV": getattr(self, 'supremo_global_rms_uV', 0.0),
        "supremo_global_energia_uV2_s": getattr(self, 'supremo_global_energia', 0.0),
        "supremo_global_auc_uV_s": getattr(self, 'supremo_global_auc', 0.0)
    }

    # --- NUEVO: Guardar metadata.json con la fecha, hora, timestamp y músculos ---
    # Ancla temporal real de la grabacion (inicio de adquisicion, no el guardado)
    _t0_perf = getattr(self, 'recording_start_time', None) or time.perf_counter()
    _t0_epoch = time.time() - (time.perf_counter() - _t0_perf)
    metadata = {
      "measurement_date": now.isoformat(),
      "t0_epoch": _t0_epoch,
      "t0_perf_counter": _t0_perf,
      "daq_sample_rate": self.SAMPLE_RATE,
      "metronomo_pulsos": len(getattr(self, '_metro_pulsos', []) or []),
      "metronomo_pulsos_csv": "pulsos_metronomo.csv" if getattr(self, '_metro_pulsos', None) else None,
      "beat_timestamps_s": list(getattr(self, 'metro_beat_timestamps_s', [])),
      "beat_sample_indices": list(getattr(self, 'metro_beat_sample_indices', [])),
      "beat_intervals_s": list(getattr(self, 'metro_applied_intervals_s', [])),
      "jitter_config": {
          "enabled": bool(hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked()),
          "pct": float(self.spin_metro_jitter_pct.value()) if hasattr(self, 'spin_metro_jitter_pct') else 0.0,
          "mode": self.cmb_metro_jitter_mode.currentText() if hasattr(self, 'cmb_metro_jitter_mode') else "exponential_hazard"
      },
      "eventos_vocales_metronomo": list(getattr(self, 'vocal_cue_log', []) or []),
      "sincronizacion_sujeto_vs_metronomo": list(getattr(self, 'sync_evaluation_log', []) or []),
      "eventos_vocales_metronomo": list(getattr(self, 'vocal_cue_log', []) or []),
      "sincronizacion_sujeto_vs_metronomo": list(getattr(self, 'sync_evaluation_log', []) or []),
      "timestamp": int(now.timestamp()),
      "sample_rate": self.SAMPLE_RATE,
      "channels": self.CANALES_DAQ,
      "muscles": muscles_list,
      "muscles_map": muscles_map,
      "bpm": self.spin_bpm.value(), # BPM se sigue tomando de la GUI del adquisidor
      "noise_seconds": self.spin_noise_duration.value(),
      "pulse_count": pulse_count_from_metronome,
      # --- NUEVO: Añadir los detalles del nombre al metadata ---
      "is_formal": is_formal,
      "sujeto": details["sujeto"],
      "letra": details["letra"],
      "prueba": details["prueba"],
      "comentario": comentario,
      "ruido_basal": {
          "inicial": {
              "mean_uV": getattr(self, 'initial_noise_mean', [0.0]*self.NUM_CANALES),
              "std_uV": getattr(self, 'initial_noise_std', [0.0]*self.NUM_CANALES),
              "line_noise_50hz_uV": getattr(self, 'line_noise_50hz_uV', [0.0]*self.NUM_CANALES)
          },
          "final": {
              "mean_uV": getattr(self, 'final_noise_mean', [0.0]*self.NUM_CANALES),
              "std_uV": getattr(self, 'final_noise_std', [0.0]*self.NUM_CANALES),
              "line_noise_50hz_uV": getattr(self, 'final_noise_50hz', [0.0]*self.NUM_CANALES)
          }
      },
      "calibracion_cvm": calib_cvm_dict
    }
    photo_meta = self._safe_photometry_export(str(output_dir))
    if photo_meta is not None:
      metadata["fotometria_espacial"] = photo_meta
    if hasattr(self, 'last_separability_metrics') and self.last_separability_metrics is not None:
      metadata["diagnostico_separabilidad_vocales"] = self.last_separability_metrics
    # --- CORRECCIÓN: Guardar metadata ÚNICAMENTE en la carpeta de cada canal con su respectivo músculo ---
    for i in range(self.NUM_CANALES):
      current_dir = os.path.join(output_dir, f"canal_{i}")
      metadata_path = os.path.join(current_dir, "metadata.json")
      ch_metadata = metadata.copy()
      ch_metadata["canal"] = f"canal_{i}"
      ch_metadata["musculo"] = self.nombres_musculos[i % len(self.nombres_musculos)]
      ch_metadata["physical_channel"] = self.CANALES_DAQ[i] if i < len(self.CANALES_DAQ) else f"ai{i}"
      try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
          json.dump(ch_metadata, f, indent=4)
        print(f"  Metadata guardado en: {metadata_path}")
      except Exception as e:
        print(f"--- ERROR AL GUARDAR METADATA.JSON en '{current_dir}' ---\n{e}")

    try:
      root_meta_path = os.path.join(output_dir, "metadata.json")
      with open(root_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
      print(f"  Metadata raíz guardado en: {root_meta_path}")
    except Exception as e:
      print(f"--- ERROR AL GUARDAR METADATA.JSON RAIZ ---: {e}")

    # 1. Guarda los pulsos del metrónomo y el .wav
    try:
      escribir_pulsos_metronomo_csv(getattr(self, '_metro_pulsos', None), output_dir)
    except Exception as e:
      print(f"[Aviso] No se pudieron guardar los pulsos del metronomo: {e}")
    try:
      guardar_grabacion_wav(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR ARCHIVOS .WAV ---\n{e}")

    # 2. Guarda el .csv
    try:
      guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES, canales_daq=self.CANALES_DAQ)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR .CSV ---\n{e}")

    # 3. Genera el gráfico .png
    try:
      generar_grafico_grabacion(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES, self.CANALES_DAQ)
      ruta_photo = os.path.join(output_dir, "photo.png")
      if os.path.exists(ruta_photo):
        self.mostrar_preview_signal.emit(ruta_photo)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR .PNG ---\n{e}")
      print("  (¿Estás seguro de que 'matplotlib' está instalado? -> pip install matplotlib)")

    # 4. Genera el gráfico de estadísticas (Evolución de SNR y Ruido)
    try:
      generar_grafico_estadisticas(self.stats_time, self.stats_snr, self.stats_noise_mean, self.stats_noise_std, output_dir, self.NUM_CANALES, self.CANALES_DAQ)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR GRÁFICO ESTADÍSTICO ---\n{e}")

    # 5. Persistir notas del experimento en metadata y archivo de texto
    try:
      self.persistir_notas_experimento(output_dir, metadata_base=metadata, texto_notas=comentario)
    except Exception as e:
      print(f"[Aviso] No se pudieron persistir notas: {e}")

    print("--- EXPORTACIÓN FINALIZADA ---")

  def on_pause_toggled(self, paused: bool):
    """Pausa o reanuda la grabacion, el metronomo y las fases de AutoForge sin emojis."""
    self.is_paused = paused
    if paused:
      self.pause_start_perf = time.perf_counter()
      self.btn_pause.setText("REANUDAR")
      if hasattr(self, '_phase_timer') and self._phase_timer.isActive():
        self._paused_phase_remaining_ms = max(50, self._phase_timer.remainingTime())
        self._phase_timer.stop()
      if hasattr(self, 'metro_timer') and self.metro_timer.isActive():
        self._paused_metro_remaining_ms = max(10, self.metro_timer.remainingTime())
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
        rem_ms = getattr(self, '_paused_metro_remaining_ms', 0)
        if rem_ms <= 0:
          bpm = max(1, self.spin_bpm.value())
          subdivs = max(1, self.spin_metro_subdivs.value())
          rem_ms = max(10, int(round((60000.0 / bpm) / subdivs)))
        self._metro_next_perf = time.perf_counter() + (rem_ms / 1000.0)
        self.metro_timer.setSingleShot(True)
        self.metro_timer.start(rem_ms)
      if hasattr(self, 'session_timer') and getattr(self, 'is_autoforge_running', False):
        self.session_timer.start(1000)

  def actualizar_plot(self):
    """
    Ejecuta la funcionalidad de actualizar_plot.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if not self.is_acquiring:
      return # No hacer nada si la adquisición no está activa

    # Si el hilo de adquisición abortó por error fatal de hardware (stop_event.is_set()),
    # detener limpiamente el metrónomo y la adquisición para no dejar la GUI congelada.
    if self.stop_event.is_set() and self.data_queue.empty():
      if getattr(self, 'acquisition_thread', None) is not None and not self.acquisition_thread.is_alive():
        print("[AVISO DAQ] El hilo de adquisición se detuvo inesperadamente. Cerrando sesión de captura limpiamente.")
        if getattr(self, 'is_autoforge_running', False):
          self.detener_autoforge()
        self.on_start_acq_click()
        return



    try:
      # --- REFACTOR: Procesamiento por lotes ---
      # 1. Drenar toda la cola en una lista
      chunks = []
      while not self.data_queue.empty():
        chunks.append(self.data_queue.get_nowait())

      if not chunks:
        return # No hay datos, no hacer nada

      # 2. Concatenar todos los chunks en un solo bloque de datos
      all_new_data = np.concatenate(chunks, axis=1)
      total_muestras_leidas = all_new_data.shape[1]

      # Dentro de actualizar_plot, justo después de all_new_data = np.concatenate(chunks, axis=1):
      if self.is_paused:
        return  # La cola ya fue drenada (evita overflow en NIDAQ), pero congelamos grabación y pantalla

      if self.is_recording or getattr(self, 'is_recording_final_noise', False):
        self.current_recording.append(all_new_data.copy())
      else:
        self.preroll_buffer.push_raw_chunk(all_new_data)

      # 4. Calibrar para la visualización y análisis en tiempo real
      
      # --- NUEVO: Imprimir el voltaje RAW de la placa 1 vez por medición para depurar ---
      if not getattr(self, '_raw_print_done', False) and not self.chk_modo_prueba.isChecked():
        print(f"\n[DEBUG DAQ] Voltaje RAW puro entregado por la placa (Canal 0): {all_new_data[0, 0]:.6f} V")
        self._raw_print_done = True

      # Calibración a microvoltios usando el factor por canal físico
      raw_uV_data = np.zeros_like(all_new_data, dtype=np.float64)
      for i in range(self.NUM_CANALES):
        hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and i < len(self.active_channel_indices)) else i
        gain = 495.0
        if hasattr(self, 'factores_calibracion_hw') and hw_idx < len(self.factores_calibracion_hw):
          gain = float(self.factores_calibracion_hw[hw_idx])
        gain = gain if gain > 0 else 495.0
        raw_uV_data[i, :] = (all_new_data[i, :] / gain) * 1000000.0
      processed_data = raw_uV_data.copy()
      if getattr(self, "chk_clamp_500uv", None) and self.chk_clamp_500uv.isChecked():
        for ch_i in range(self.NUM_CANALES):
          if not self._is_mic_channel(ch_i):
            ch_med = float(np.median(processed_data[ch_i, :]))
            processed_data[ch_i, :] = np.clip(processed_data[ch_i, :], ch_med - 500.0, ch_med + 500.0)
      self._actualizar_medidor_50hz_en_vivo(raw_uV_data)

      # --- CVM RECORDING HOOK ---
      if getattr(self, 'is_cvm_recording_take', False):
        if not hasattr(self, 'cvm_current_take_chunks'):
          self.cvm_current_take_chunks = []
        self.cvm_current_take_chunks.append(all_new_data.copy())

      # --- RUIDO FINAL HOOK ---
      if getattr(self, 'is_recording_final_noise', False):
        if not hasattr(self, 'final_raw_noise_accumulated'):
          self.final_raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
        for i in range(self.NUM_CANALES):
          self.final_raw_noise_accumulated[i].append(raw_uV_data[i, :])

      # --- NUEVO: Matar el pico transitorio al iniciar/cambiar el filtro ---
      if getattr(self, '_init_filter_state', False):
        initial_vals = processed_data[:, 0]
        if self.notch_zi is not None:
          for i in range(self.NUM_CANALES):
            self.notch_zi[:, :, i] = self.notch_zi[:, :, i] * initial_vals[i]
        if self.filter_zi is not None:
          for i in range(self.NUM_CANALES):
            self.filter_zi[:, :, i] = self.filter_zi[:, :, i] * initial_vals[i]
        if getattr(self, 'env_zi', None) is not None:
          initial_rect = np.abs(initial_vals)
          for i in range(self.NUM_CANALES):
            self.env_zi[:, :, i] = self.env_zi[:, :, i] * initial_rect[i]
        self._init_filter_state = False

      # Aplicar filtros (si están habilitados) al bloque calibrado
      # --- NUEVO: Aplicar filtro Notch primero ---
      if self.chk_notch_enable.isChecked() and self.notch_sos is not None and self.notch_zi is not None:
        notch_filtered_data = np.zeros_like(processed_data)
        for i in range(self.NUM_CANALES):
          notch_filtered_data[i, :], self.notch_zi[:, :, i] = signal.sosfilt(self.notch_sos, processed_data[i, :], zi=self.notch_zi[:, :, i])
        processed_data = notch_filtered_data

      # --- Aplicar filtro Pasa-Banda después ---
      if self.chk_filter_enable.isChecked() and self.filter_sos is not None and self.filter_zi is not None:
        bandpass_filtered_data = np.zeros_like(processed_data)
        for i in range(self.NUM_CANALES):
          if self._is_mic_channel(i):
            # Canal de micrófono/audio: preservar audio sin filtro pasa-banda muscular 20-500 Hz
            bandpass_filtered_data[i, :] = processed_data[i, :]
            continue
          # --- CORRECCIÓN: Indexar correctamente el array de estado del filtro 'zi' ---
          # La forma de self.filter_zi es (n_sections, 2, NUM_CANALES).
          bandpass_filtered_data[i, :], self.filter_zi[:, :, i] = signal.sosfilt(self.filter_sos, processed_data[i, :], zi=self.filter_zi[:, :, i])
        processed_data = bandpass_filtered_data
      
      # 5. Actualizar el buffer de ploteo
      if total_muestras_leidas >= self.PLOT_SAMPLES:
        # Si llegan más datos que el tamaño del buffer, tomar solo los más recientes
        self.plot_buffer_datos[:, :] = processed_data[:, -self.PLOT_SAMPLES:]
      else:
        self.plot_buffer_datos = np.roll(self.plot_buffer_datos, -total_muestras_leidas, axis=1)
        self.plot_buffer_datos[:, -total_muestras_leidas:] = processed_data
      self.total_muestras_procesadas = getattr(self, 'total_muestras_procesadas', 0) + total_muestras_leidas

      # --- NUEVO: Auto-ajuste de umbral ANTES de grabar ---
      if getattr(self, 'needs_auto_threshold', False):
        # Esperar 1.5 segundos desde que inició para que la señal se estabilice
        if time.perf_counter() - getattr(self, 'acq_start_time', 0) > 1.5:
          samples_1s = int(self.SAMPLE_RATE * 1.0) # Tomar el último segundo de datos
          if samples_1s <= self.PLOT_SAMPLES and self.NUM_CANALES > 0:
            last_1s_data = self.plot_buffer_datos[:, -samples_1s:]
            # Calcular la desviación estándar de cada canal
            stds = [np.std(last_1s_data[i, :]) for i in range(self.NUM_CANALES)]
            max_std = max(stds)
            if max_std > 0:
              self.spin_peak_th.setValue(max_std * 5.0)
              print(f"[Auto-Ajuste] Umbral de picos fijado en {max_std * 5.0:.1f} µV (basado en el ruido inicial en reposo).")
          self.needs_auto_threshold = False # Solo se hace una vez por adquisición

      # --- NUEVO: Acumular datos de ruido y establecer promedio automáticamente ---
      if self.is_recording and self.recording_start_time is not None:
        elapsed_time = time.perf_counter() - self.recording_start_time
        
        # --- FIX: Calcular duration según si se usa metrónomo en modo manual ---
        if self.chk_use_metronome.isChecked() and not getattr(self, 'is_autoforge_running', False):
          bpm = self.spin_bpm.value()
          if bpm <= 0: bpm = 60
          beat_interval = 60.0 / bpm
          noise_dur = 3.0 * beat_interval
        else:
          noise_dur = self.spin_noise_duration.value()

        # CRITICAL FIX: Verificamos si el ruido ya fue calculado para evitar re-entrar aquí
        # cuando AutoForge resetea el elapsed_time a 0 para grabar la señal.
        if not getattr(self, 'noise_calculated', False) and elapsed_time < noise_dur:
          if not hasattr(self, 'raw_noise_accumulated') or not self.raw_noise_accumulated:
            self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
          for i in range(self.NUM_CANALES):
            self.noise_data_accumulated[i].append(processed_data[i, :])
            if i < len(self.raw_noise_accumulated):
              self.raw_noise_accumulated[i].append(raw_uV_data[i, :])
            
          # --- NUEVO: Lógica de Cuenta Regresiva dentro del ruido (Solo si no es AutoForge) ---
          if not getattr(self, 'is_autoforge_running', False):
            if self.chk_use_metronome.isChecked():
              time_remaining = noise_dur - elapsed_time
              import math
              current_countdown = int(math.ceil(time_remaining / beat_interval))
              if current_countdown > 3: current_countdown = 3
              if current_countdown < 1: current_countdown = 1
              
              self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, self._get_plot_y_center())
              self.countdown_text.setText(f"PREPÁRATE...\n{current_countdown}")
              self.countdown_text.show()
              self.countdown_active = True
            else:
              time_remaining = noise_dur - elapsed_time
              if time_remaining <= 3.0:
                import math
                current_countdown = int(math.ceil(time_remaining))
                if not hasattr(self, 'last_countdown') or self.last_countdown != current_countdown:
                  self.last_countdown = current_countdown
                  self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, self._get_plot_y_center())
                  play_beep(800, 200)
                
                self.countdown_text.setText(f"PREPÁRATE...\n{current_countdown}")
                self.countdown_text.show()
                self.countdown_active = True
              else:
                self.countdown_text.hide()
          else:
            pass # AutoForge maneja su propio countdown
        else:
          if not getattr(self, 'noise_calculated', False):
            if not getattr(self, 'noise_initialized', False):
              # --- GO! ---
              if getattr(self, 'countdown_active', False) and not getattr(self, 'is_autoforge_running', False) and not self.chk_use_metronome.isChecked():
                play_beep(1200, 500)
              self.countdown_text.setText("¡GO!")
              QtCore.QTimer.singleShot(1000, self.countdown_text.hide)
              self.countdown_active = False
            
              for i in range(self.NUM_CANALES):
                if self.noise_data_accumulated[i]:
                  all_noise = np.concatenate(self.noise_data_accumulated[i])
                  window_size_ms = self.spin_rms_window.value()
                  window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
                  if window_size < 1: window_size = 1
                  all_noise_env = calculate_rms_envelope(all_noise, window_size)
                  
                  self.noise_levels[i] = np.mean(all_noise_env)
                  self.initial_noise_mean[i] = self.noise_levels[i]
                  self.initial_noise_std[i] = np.std(all_noise_env)
                  
                  is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
                  show_noise = self.chk_show_noise.isChecked()
                  self.noise_lines[i].setPos(self.noise_levels[i])
                  if is_rms:
                    self.noise_lines_neg[i].setVisible(False)
                    self.noise_regions[i].setRegion([0.0, self.noise_levels[i]])
                  else:
                    self.noise_lines_neg[i].setPos(-self.noise_levels[i])
                    self.noise_lines_neg[i].setVisible(show_noise)
                    self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
                  self.noise_lines[i].setVisible(show_noise)
                  self.noise_regions[i].setVisible(show_noise)
                  
                  if hasattr(self, 'noise_status_labels'):
                    self.noise_status_labels[i].setText(f"Ruido Base: x̄={self.initial_noise_mean[i]:.1f}µV, σ={self.initial_noise_std[i]:.1f}µV")
                    self.noise_status_labels[i].setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold; background-color: #111111; border: 1px solid #00FFFF;")
              
              if self.NUM_CANALES > 0:
                ruido_maximo_std = max(getattr(self, 'initial_noise_std', [0]))
                if ruido_maximo_std > 0:
                  self.spin_peak_th.setValue(ruido_maximo_std * 5.0)

              self.noise_initialized = True
              self.noise_calculated = True

      # 6. Actualizar la GUI
      is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
      
      # --- NUEVO: Envolvente RMS Numba (Rendimiento Extremo) ---
      # Siempre se calcula para asegurar cálculos de SNR robustos
      window_size_ms = self.spin_rms_window.value()
      window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
      if window_size < 1: window_size = 1
      for i in range(self.NUM_CANALES):
        self.env_buffer_datos[i, :] = calculate_rms_envelope(self.plot_buffer_datos[i, :], window_size)
          
      escala_cvm_activa = bool(getattr(self, 'chk_escala_cvm', None) and self.chk_escala_cvm.isChecked())
      for i in range(self.NUM_CANALES):
        try:
          if is_rms:
            valid_data = np.nan_to_num(self.env_buffer_datos[i]).copy()
          else:
            valid_data = np.nan_to_num(self.plot_buffer_datos[i]).copy()

          # Escalado en tiempo real en % CVM para canales EMG
          if escala_cvm_activa and not self._is_mic_channel(i):
            ref_cvm = 500.0
            if hasattr(self, 'cvm_env_p95_per_channel') and i < len(self.cvm_env_p95_per_channel) and self.cvm_env_p95_per_channel[i] > 10.0:
              ref_cvm = float(self.cvm_env_p95_per_channel[i])
            elif getattr(self, 'supremo_global_p95_uV', 0.0) > 10.0:
              ref_cvm = float(self.supremo_global_p95_uV)
            valid_data = (valid_data / (ref_cvm + 1e-6)) * 100.0

          x_dec, y_dec = decimate_min_max(self.plot_vector_tiempo, valid_data, max_points=3000)

          if not getattr(self, 'is_multitrack_view', False):
            if i < len(self.curvas) and self.curvas[i].scene() is not None:
              self.curvas[i].setData(x_dec, y_dec)
          else:
            if i == 0:
              if len(self.curvas) > 0 and self.curvas[0].scene() is not None:
                self.curvas[0].setData(x_dec, y_dec)
            else:
              sub_idx = i - 1
              multitrack_curvas = getattr(self, 'curvas_multitrack', [])
              if sub_idx < len(multitrack_curvas) and multitrack_curvas[sub_idx].scene() is not None:
                multitrack_curvas[sub_idx].setData(x_dec, y_dec)
        except Exception:
          pass # Evita que se congele toda la GUI si hay un error matematico o de transicion de vista
      
      if self.chk_autoscroll.isChecked():
        self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)
        
        # --- Auto-Escala Dinámica Estable con Peak-Hold, Hold Timer e Histéresis ---
        if self.NUM_CANALES > 0 and self.plot_buffer_datos.size > 0:
          # Evaluar ventana reciente (~2.5s o 2 ciclos de metrónomo) para adaptarse rápido a nuevas vocales
          bpm = max(1, self.spin_bpm.value())
          eval_seconds = min(self.PLOT_DURATION_S, max(2.0, (60.0 / bpm) * 2.0))
          eval_samples = min(self.PLOT_SAMPLES, int(eval_seconds * self.SAMPLE_RATE))
          
          emg_idxs = [ch_i for ch_i in range(self.NUM_CANALES) if not self._is_mic_channel(ch_i)]
          if not emg_idxs:
            emg_idxs = list(range(self.NUM_CANALES))
          clamp_500 = bool(getattr(self, "chk_clamp_500uv", None) and self.chk_clamp_500uv.isChecked())
          if is_rms:
            sub_buf = self.env_buffer_datos[emg_idxs, -eval_samples:]
            recent_max = float(np.percentile(sub_buf, 98.5)) if sub_buf.size > 0 else 50.0
          else:
            sub_buf = np.abs(self.plot_buffer_datos[emg_idxs, -eval_samples:])
            recent_max = float(np.percentile(sub_buf, 98.5)) if sub_buf.size > 0 else 50.0

          min_floor = 25.0 # Piso mínimo seguro en microvoltios
          max_ceiling = 500.0 if clamp_500 else 5000.0
          if clamp_500:
            recent_max = min(recent_max, 500.0)
          target_max = min(max_ceiling, max(min_floor, recent_max * 1.20))

          now_perf = time.perf_counter()
          current_held = getattr(self, '_held_plot_ymax', min_floor)
          last_peak_time = getattr(self, '_held_plot_last_peak_time', 0.0)

          if target_max >= current_held:
            # Subida inmediata ante contracción
            current_held = target_max
            self._held_plot_last_peak_time = now_perf
          else:
            # Sostener el pico durante al menos 2.0s antes de decaer suavemente (0.992 por frame)
            if now_perf - last_peak_time > 2.0:
              current_held = max(target_max, current_held * 0.992)
          
          current_held = max(min_floor, current_held)
          self._held_plot_ymax = current_held

          # Aplicar rango al gráfico con histéresis del 6% para eliminar parpadeo y vibración de pantalla
          view_range = self.plot.getViewBox().viewRange()[1]
          delta_ratio = abs(view_range[1] - current_held) / max(1.0, current_held)
          if is_rms:
            if delta_ratio > 0.06 or view_range[0] < -2.0 or view_range[0] > 1.0:
              self.plot.setYRange(0.0, current_held, padding=0.02)
          else:
            if delta_ratio > 0.06 or abs(view_range[0] + current_held) > 1.0:
              self.plot.setYRange(-current_held, current_held, padding=0.02)
      self.check_for_trigger(processed_data, total_muestras_leidas)
      # --- NUEVO: Pasar env_buffer_datos a las mediciones para que el pico visual se evalúe sobre la envolvente RMS ---
      self.actualizar_mediciones(processed_data, self.env_buffer_datos)
      self.actualizar_espectrograma(processed_data)
      if hasattr(self, 'pulse_detector') and self.pulse_detector is not None:
        self.pulse_detector.process()
      self._actualizar_posicion_ventana_deteccion()

      # --- Actualiza el cronómetro (esto se ejecuta siempre) ---
      if self.is_recording and self.recording_start_time is not None:
        elapsed_time = time.perf_counter() - self.recording_start_time
        if getattr(self, 'is_autoforge_running', False):
          # En AutoForge, ponemos textos personalizados
          tr_str = getattr(self, 'tiempo_restante_str', '00:00')
          cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
          if not getattr(self, 'noise_calculated', False):
            self.label_rec_time.setText(f"GRABANDO RUIDO... {elapsed_time:.1f} s{cuadro_azul}")
            self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFFF00;") # Amarillo neón
          else:
            if getattr(self, 'is_autoforge_continuo', False):
              pulso_num = max(1, getattr(self, 'metro_beat_count', 1))
              palabra_cur = self.autoforge_words[(pulso_num - 1) % len(self.autoforge_words)].upper() if len(self.autoforge_words) > 0 else ""
              self.label_rec_time.setText(f"SECUENCIA: /{palabra_cur}/ (Pulso {pulso_num}) | {elapsed_time:.1f} s{cuadro_azul}")
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #00FFCC; font-size: 15px;")
              if hasattr(self, 'countdown_text'):
                self.countdown_text.hide()
            else:
              palabra = self.autoforge_words[self.autoforge_word_idx].upper() if hasattr(self, 'autoforge_words') and self.autoforge_word_idx < len(self.autoforge_words) else ""
              pulso_num = getattr(self, 'metro_beat_count', 0)
              self.label_rec_time.setText(f"GRABANDO VOCAL: /{palabra}/ (Pulso {pulso_num}) | {elapsed_time:.1f} s{cuadro_azul}")
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #00FFCC; font-size: 15px;")
              if palabra:
                vocal_colores = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
                v_col = vocal_colores.get(palabra, '#00FFCC')
                self.lbl_recording_space.setText(f"<div align='center' style='font-size: 70px; font-weight: 900; color: {v_col};'>/{palabra}/</div>")
              if hasattr(self, 'countdown_text'):
                self.countdown_text.hide()
        else:
          # Modo Manual
          if self.chk_use_metronome.isChecked():
            bpm = self.spin_bpm.value()
            if bpm <= 0: bpm = 60
            beat_interval = 60.0 / bpm
            noise_dur = 3.0 * beat_interval
            
            if not getattr(self, 'noise_calculated', False) and elapsed_time < noise_dur:
              import math
              time_remaining = noise_dur - elapsed_time
              countdown = int(math.ceil(time_remaining / beat_interval))
              if countdown > 3: countdown = 3
              if countdown < 1: countdown = 1
              
              tr_str = f"{time_remaining:.1f}s"
              cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white; font-size: 20px;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
              
              time_str = f"PREPÁRATE... {countdown}{cuadro_azul}"
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF8800; font-size: 30px;") # Naranja gigante
            else:
              signal_time = elapsed_time - noise_dur
              minutes = int(signal_time // 60)
              seconds = int(signal_time % 60)
              tenths = int((signal_time % 1) * 10)
              time_str = f"GRABANDO SEÑAL: {minutes:02d}:{seconds:02d}.{tenths}"
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF00FF;") # Magenta neón
              
              # --- NUEVO: Iniciar el conteo del metrónomo justo aquí ---
              if not getattr(self, 'counting_started', False):
                self.counting_started = True
          else:
            noise_dur = self.spin_noise_duration.value()
            if not getattr(self, 'noise_calculated', False) and elapsed_time < noise_dur:
              time_remaining = noise_dur - elapsed_time
              tr_str = f"{time_remaining:.1f}s"
              cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white; font-size: 20px;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
              
              if time_remaining <= 3.0:
                import math
                countdown = int(math.ceil(time_remaining))
                time_str = f"PREPÁRATE... {countdown}{cuadro_azul}"
                self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF8800; font-size: 30px;") # Naranja gigante
              else:
                time_str = f"GRABANDO RUIDO ({elapsed_time:.1f}s / {noise_dur:.1f}s){cuadro_azul}"
                self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFFF00;") # Amarillo neón
            else:
              signal_time = elapsed_time - noise_dur
              minutes = int(signal_time // 60)
              seconds = int(signal_time % 60)
              tenths = int((signal_time % 1) * 10)
              time_str = f"GRABANDO SEÑAL: {minutes:02d}:{seconds:02d}.{tenths}"
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF00FF;") # Magenta neón

          self.label_rec_time.setText(time_str)
          
        # --- NUEVO: Evaluador de Ruido Inter-pulso (Tester de relajación) ---
        # Para el evaluador es necesario usar el tiempo de señal (ya sin ruido base).
        if getattr(self, 'is_autoforge_running', False):
          signal_time = elapsed_time # En AutoForge, ya se reseteó el tiempo
        else:
          signal_time = elapsed_time - noise_dur # En Manual, calculamos el offset
          
        self.current_signal_time = signal_time
          
        if getattr(self, 'noise_calculated', False) and signal_time >= 0:
          # --- NUEVO: Evaluador Retrospectivo de Ruido Inter-pulso (Tester de relajación) ---
          if not hasattr(self, 'rt_peaks'):
            self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
            self.rt_last_noise_eval = [0.0] * self.NUM_CANALES

          period_s = float(getattr(self, 'current_beat_tau_s', 60.0 / max(1, self.spin_bpm.value())))
          
          for i in range(self.NUM_CANALES):
            # Obtener el tiempo del pico visual actual
            x_data, y_data = self.peak_scatters[i].getData()
            if len(x_data) > 0:
              # x_data[0] es el tiempo relativo (-PLOT_DURATION_S a 0.0)
              t_abs = elapsed_time + x_data[0]
              
              peaks = self.rt_peaks[i]
              if not peaks:
                peaks.append(t_abs)
              else:
                # Si el pico está a menos de 0.6 periodos del último, asumimos que es el mismo y afinamos su posición
                if t_abs - peaks[-1] < period_s * 0.6:
                  peaks[-1] = max(peaks[-1], t_abs)
                else:
                  # ¡Es un pulso nuevo!
                  peaks.append(t_abs)
                  if len(peaks) > 3:
                    peaks.pop(0)
            
            # Evaluación retrospectiva: necesitamos al menos 2 picos
            if len(self.rt_peaks[i]) >= 2:
              t1 = self.rt_peaks[i][-2]
              t2 = self.rt_peaks[i][-1]
              
              # Evaluar solo si t2 ya pasó lo suficiente (estamos en el valle biológico)
              # y si no hemos evaluado ya este mismo par de picos.
              if (elapsed_time - t2 > period_s * 0.3) and (t2 > self.rt_last_noise_eval[i]):
                self.rt_last_noise_eval[i] = t2
                
                midpoint_abs = (t1 + t2) / 2.0
                d = t2 - t1
                win_start_abs = midpoint_abs - d / 4.0
                win_end_abs = midpoint_abs + d / 4.0
                
                # Convertir tiempo absoluto a relativo en el buffer
                rel_start = win_start_abs - elapsed_time
                rel_end = win_end_abs - elapsed_time
                
                idx_start = np.searchsorted(self.plot_vector_tiempo, rel_start)
                idx_end = np.searchsorted(self.plot_vector_tiempo, rel_end)
                
                # Verificar que la ventana esté dentro del buffer visible actual
                if 0 <= idx_start < idx_end <= len(self.plot_vector_tiempo):
                  noise_segment = self.env_buffer_datos[i, idx_start:idx_end]
                  if len(noise_segment) > 0:
                    curr_mean = np.mean(noise_segment)
                    curr_std = np.std(noise_segment)
                    
                    init_std = getattr(self, 'initial_noise_std', [0]*self.NUM_CANALES)[i]
                    init_mean = getattr(self, 'initial_noise_mean', [0]*self.NUM_CANALES)[i]
                    
                    # Calcular el radio de deterioro
                    if init_std > 0.01:
                      ratio = curr_std / init_std
                    elif init_mean > 0.01:
                      ratio = curr_mean / init_mean
                    else:
                      ratio = 1.0
                    
                    if hasattr(self, 'noise_status_labels'):
                      bg_color = "#000000"
                      fg_color = "#00FF00" if ratio <= 1.20 else "#FF0000" 
                      text = f"Inter-pulso: x̄={curr_mean:.1f}µV (Base={init_mean:.1f}µV) | {ratio*100:.0f}%"
                      self.noise_status_labels[i].setText(text)
                      self.noise_status_labels[i].setStyleSheet(f"color: {fg_color}; background-color: {bg_color}; border: 1px solid {fg_color}; border-radius: 3px; padding: 2px; font-weight: bold; font-size: 11px;")
                      self.noise_status_labels[i].setVisible(self.chk_show_noise.isChecked())
                    
                    if hasattr(self, 'dynamic_noise_regions') and i < len(self.dynamic_noise_regions):
                      if self.chk_show_noise.isChecked():
                        brush_color = pg.mkBrush(0, 255, 128, 18) if ratio <= 1.20 else pg.mkBrush(255, 60, 60, 20)
                        is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
                        r_min = 0.0 if is_rms else -curr_mean
                        self.dynamic_noise_regions[i].setRegion([r_min, curr_mean])
                        self.dynamic_noise_regions[i].setBrush(brush_color)
                        self.dynamic_noise_regions[i].show()
                      else:
                        self.dynamic_noise_regions[i].hide()
                        self.dynamic_noise_regions[i].setRegion([0, 0])
                    
                    # SNR: Calculado sobre el último periodo (Pico máximo envolvente / Ruido Basal Inicial con envolvente)
                    samples_period = min(int(period_s * self.SAMPLE_RATE), self.PLOT_SAMPLES)
                    full_period_segment = self.env_buffer_datos[i, -samples_period:]
                    peak_val = np.max(full_period_segment)
                    
                    # Usar Ruido Basal Inicial (initial_noise_mean ya tiene la envolvente aplicada en su inicialización)
                    baseline_noise = getattr(self, 'initial_noise_mean', [1e-12]*(i+1))[i]
                    if baseline_noise <= 0: baseline_noise = 1e-12
                    curr_snr = peak_val / baseline_noise
                    
                    self.stats_time[i].append(signal_time)
                    
                    if len(self.stats_snr[i]) > 0:
                      n = len(self.stats_snr[i])
                      snr_acumulado = (self.stats_snr[i][-1] * n + curr_snr) / (n + 1)
                    else:
                      snr_acumulado = curr_snr
                      
                    self.stats_snr[i].append(snr_acumulado)
                    self.stats_noise_mean[i].append(curr_mean)
                    self.stats_noise_std[i].append(curr_std)
                    
                    if hasattr(self, 'global_snr_acumulado'):
                      c = self.global_snr_count[i]
                      self.global_snr_acumulado[i] = (self.global_snr_acumulado[i] * c + curr_snr) / (c + 1)
                      self.global_snr_count[i] += 1
      
    except queue.Empty:
      pass # Si no hay datos, no hace nada

  def actualizar_mediciones(self, chunk_data, plot_buffer_datos):
    """Calcula y actualiza los labels de Vp-p, RMS y SNR visualizando los picos."""
    if not self.is_acquiring or self.NUM_CANALES == 0:
      return

    try:
      # Cálculo instantáneo para el chunk (Vp-p y RMS del último paquete)
      max_vals = np.max(chunk_data, axis=1)
      min_vals = np.min(chunk_data, axis=1)
      vp_p = max_vals - min_vals
      
      rms = np.sqrt(np.mean(np.square(chunk_data), axis=1))

      # --- NUEVO: Dominancia Muscular (% Dom) en vivo ---
      emg_indices = [ch for ch in range(self.NUM_CANALES) if not self._is_mic_channel(ch)]
      if not emg_indices:
        emg_indices = list(range(self.NUM_CANALES))
      sum_rms_emg = float(np.sum(rms[emg_indices])) + 1e-9
      
      # --- MEJORA: Ventanas Deterministas atadas al metrónomo ---
      # Aislar exactamente el segmento de tiempo correspondiente al pulso actual
      period_s = 60.0 / max(1, self.spin_bpm.value())
      
      signal_time = getattr(self, 'current_signal_time', -1.0)
      if getattr(self, 'is_recording', False) and signal_time >= 0:
        current_cycle = int(signal_time / period_s)
        time_in_cycle = signal_time - (current_cycle * period_s)
        samples_to_search = int(time_in_cycle * self.SAMPLE_RATE)
      else:
        # Modo de pre-visualización: ventana deslizante libre de 1.5 ciclos
        samples_to_search = int((period_s * 1.5) * self.SAMPLE_RATE)
        
      samples_to_search = max(1, samples_to_search) # Prevenir 0 para evitar errores de slice
      search_start_idx = max(0, self.PLOT_SAMPLES - samples_to_search)
      
      peak_th = self.spin_peak_th.value()

      # Actualiza los labels y el marcador gráfico
      for i in range(self.NUM_CANALES):
        hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and i < len(self.active_channel_indices)) else i
        musculo = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
        if i in emg_indices:
          dom_pct = (rms[i] / sum_rms_emg) * 100.0
          dom_str = f" ({dom_pct:.1f}% Dom)"
        else:
          dom_str = ""
        texto = f"<b>ai{hw_idx} - {musculo}:</b> {vp_p[i]:.1f} µVp-p, {rms[i]:.1f} µVrms{dom_str}"
        
        # Aislar la ventana de búsqueda para este canal
        window_data = plot_buffer_datos[i, search_start_idx:]
        abs_window = np.abs(window_data)
        
        # --- NUEVO: Encontrar los puntos que superan la línea de threshold ---
        mask_superan = abs_window >= peak_th
        
        if np.any(mask_superan):
          # Obtener la amplitud máxima absoluta (pico real) en lugar del promedio
          true_peak_val = np.max(abs_window)
          
          # Encontrar el tiempo del pico máximo para ubicar el marcador visualmente
          idx_max_local = np.argmax(abs_window)
          idx_max = search_start_idx + idx_max_local
          t_max = self.plot_vector_tiempo[idx_max]
          
          # Colocar el marcador en el tiempo del pico (en la amplitud máxima)
          v_visual = true_peak_val
          
          self.peak_scatters[i].setData([t_max], [v_visual])
          if getattr(self, 'is_recording', False) and getattr(self, 'noise_calculated', False):
            if self.noise_levels[i] > 0:
              snr_inst = true_peak_val / self.noise_levels[i]
              if hasattr(self, 'global_snr_count') and self.global_snr_count[i] > 0:
                snr_mostrar = self.global_snr_acumulado[i]
                texto += f" | SNR(Acum): {snr_mostrar:.1f}"
              elif hasattr(self, 'stats_snr') and i < len(self.stats_snr):
                snr_mostrar = self.stats_snr[i][-1] if len(self.stats_snr[i]) > 0 else snr_inst
                texto += f" | SNR(Acum): {snr_mostrar:.1f}"
        else:
          self.peak_scatters[i].setData([], []) # Oculta el marcador si no supera el umbral
          if getattr(self, 'is_recording', False) and getattr(self, 'noise_calculated', False):
            if hasattr(self, 'global_snr_count') and self.global_snr_count[i] > 0:
              snr_mostrar = self.global_snr_acumulado[i]
              texto += f" | SNR(Acum): {snr_mostrar:.1f}"
            elif hasattr(self, 'stats_snr') and i < len(self.stats_snr):
              snr_mostrar = self.stats_snr[i][-1] if len(self.stats_snr[i]) > 0 else 0.0
              texto += f" | SNR(Acum): {snr_mostrar:.1f}"
            else:
              texto += f" | SNR: < Umbral"
            
        self.measure_labels[i].setText(texto)
    except Exception as e:
      print(f"Error al calcular mediciones: {e}")

  def _calcular_formantes_y_f0_audio(self, mic_seg: np.ndarray, fs: float) -> str:
    """Calcula en tiempo real F0 (pitch), F1/F2 (formantes vocálicos) y SNR acústico sobre el micrófono (Ch3)."""
    try:
      arr = np.asarray(mic_seg, dtype=np.float64)
      arr = arr - np.mean(arr)
      rms = float(np.sqrt(np.mean(arr ** 2)))
      if rms < 1e-5 or len(arr) < 128:
        return "Audio Ch3: Silencio basal"
      win = np.hanning(len(arr))
      spec = np.abs(np.fft.rfft(arr * win))
      freqs = np.fft.rfftfreq(len(arr), d=1.0 / fs)
      # Pitch F0 en banda fonatoria (85 - 350 Hz)
      m_f0 = (freqs >= 85.0) & (freqs <= 350.0)
      f0_hz = float(freqs[m_f0][np.argmax(spec[m_f0])]) if np.any(m_f0) else 0.0
      # Primer formante F1 (300 - 950 Hz)
      m_f1 = (freqs >= 300.0) & (freqs <= 950.0)
      f1_hz = float(freqs[m_f1][np.argmax(spec[m_f1])]) if np.any(m_f1) else 0.0
      # Segundo formante F2 (950 - 2800 Hz)
      m_f2 = (freqs > 950.0) & (freqs <= min(2800.0, fs * 0.48))
      f2_hz = float(freqs[m_f2][np.argmax(spec[m_f2])]) if np.any(m_f2) else 0.0
      noise_floor = float(np.percentile(spec, 25)) + 1e-9
      snr_db = 20.0 * np.log10(max(1.0, float(np.max(spec)) / noise_floor))
      return f"Audio Ch3 | F0: {f0_hz:.0f}Hz | F1: {f1_hz:.0f}Hz | F2: {f2_hz:.0f}Hz | SNR: {snr_db:.1f}dB"
    except Exception:
      return ""

  def actualizar_espectrograma(self, new_data):
    """Calcula y actualiza el gráfico del espectrograma."""
    if not self.chk_spectrogram_enable.isChecked() or not SCIPY_DISPONIBLE or self.NUM_CANALES == 0:
      return

    try:
      # Obtener los datos del canal seleccionado
      data_canal = new_data[self.spectrogram_channel_index]

      # Calcular STFT (Short-Time Fourier Transform)
      f, t, Zxx = signal.stft(data_canal, fs=self.SAMPLE_RATE, nperseg=self.CURRENT_FFT_LEN)
      
      # Tomar la magnitud y aplicar escala logarítmica para mejor visualización
      Zxx_mag = np.abs(Zxx)
      Zxx_log = np.log10(Zxx_mag + 1e-12) # Se suma un valor pequeño para evitar log(0)

      num_nuevas_columnas = Zxx_log.shape[1]
      if num_nuevas_columnas > 0:
        # Desplazar el buffer del espectrograma hacia la izquierda
        self.spectrogram_buffer = np.roll(self.spectrogram_buffer, -num_nuevas_columnas, axis=0)
        # Añadir las nuevas columnas al final
        self.spectrogram_buffer[-num_nuevas_columnas:, :] = Zxx_log.T[:num_nuevas_columnas, :]
        
        # --- CORRECCIÓN v3: Actualizar solo la imagen ---
        # La escala y el rango ya fueron configurados en setup_gui_for_channels.
        # Simplemente actualizamos los datos de la imagen.
        self.spectrogram_view.setImage(self.spectrogram_buffer.T, autoLevels=True)
    except Exception as e:
      print(f"Error al actualizar espectrograma: {e}")



  def eventFilter(self, obj, event):
    """
    Ejecuta la funcionalidad de eventFilter.

    Args:
      obj (Any): Argumento posicional obj.
      event (Any): Argumento posicional event.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if obj == getattr(self, 'plot_widget', None) and event.type() == QtCore.QEvent.Type.Resize:
      if hasattr(self, 'autoforge_overlay'):
        self.autoforge_overlay.resize(event.size())
    return super().eventFilter(obj, event)

  def detener_autoforge(self):
    """
    Detiene de forma completa, limpia e inmediata cualquier sesion de AutoForge
    (discreta o continua), cancelando timers, metronomo, beeps, overlays y estados pendientes.
    """
    self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
    self.is_autoforge_running = False
    self.is_autoforge_continuo = False
    self.is_cvm_active = False
    self.is_cvm_recording_take = False
    self.is_recording_final_noise = False
    self.is_recording = False
    self.is_finishing_sequence = False
    self.is_paused = False

    # Detener timer de fase pendiente
    if hasattr(self, '_phase_timer') and self._phase_timer.isActive():
      self._phase_timer.stop()
    self._scheduled_phase_cb = None
    self._paused_phase_remaining_ms = 0

    # Detener timer de sesion
    if hasattr(self, 'session_timer') and self.session_timer.isActive():
      self.session_timer.stop()

    # Detener metronomo y purgar cola de beeps
    self.stop_native_metronome()
    if hasattr(self, 'beep_queue') and self.beep_queue is not None:
      while not self.beep_queue.empty():
        try:
          self.beep_queue.get_nowait()
        except Exception:
          break

    # Ocultar overlays y textos temporales
    if hasattr(self, 'autoforge_overlay'):
      self.autoforge_overlay.hide()
      self.autoforge_overlay.setText("")
    if hasattr(self, 'countdown_text'):
      self.countdown_text.hide()
    if hasattr(self, 'lbl_recording_space'):
      self.lbl_recording_space.setText("")
    if hasattr(self, 'label_rec_time'):
      self.label_rec_time.setText("LISTO")
      self.label_rec_time.setStyleSheet("font-weight: bold; color: #00FFCC;")

    # Limpiar buffers de grabacion en curso
    self.current_recording = []
    if hasattr(self, 'cvm_current_take_chunks'):
      self.cvm_current_take_chunks = []
    if hasattr(self, 'final_raw_noise_accumulated'):
      self.final_raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]

    # Restaurar paneles de configuracion
    if hasattr(self, 'config_stack'):
      self.config_stack.setCurrentIndex(0)

    # Restaurar botones
    if hasattr(self, 'btn_pause'):
      self.btn_pause.setChecked(False)
      self.btn_pause.setEnabled(False)
      self.btn_pause.setText("PAUSAR")
    self.reset_autoforge_buttons()

    # Matar proceso externo de palabras si existiera
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try:
        self.word_window_process.kill()
        self.word_window_process.wait(timeout=1)
      except Exception:
        pass
      self.word_window_process = None

  def reset_autoforge_buttons(self):
    """Restaura los botones de autoforge a su estado normal - Cyberpunk."""
    self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
    self.is_autoforge_running = False
    self.is_autoforge_continuo = False
    self.is_cvm_active = False
    self.is_recording_final_noise = False
    self.btn_autoforge.setText("AUTOGRABADO (Discreto)")
    self.btn_autoforge.setStyleSheet(BTN_AUTOFORGE_DISC_CYBER)
    self.btn_autoforge.setEnabled(True)
    self.btn_autoforge_continuo.setText("SECUENCIA CONTINUA")
    self.btn_autoforge_continuo.setStyleSheet(BTN_AUTOFORGE_CONT_CYBER)
    self.btn_autoforge_continuo.setEnabled(True)
    if hasattr(self, "btn_record"):
      self.btn_record.setEnabled(bool(getattr(self, "is_acquiring", False)))

  def _imprimir_aviso_fotometria_lineal(self, asistente_confirmado: bool, modo: str):
    """Registra en consola el resultado del Asistente Lineal Simple de Fotometria."""
    if asistente_confirmado:
      print(f"[FOTOMETRIA] Asistente lineal confirmado en {modo}: montaje guardado con metadata_fotometria.json")
    else:
      print(f"[FOTOMETRIA] Asistente lineal omitido o cancelado en {modo}: la grabación continúa sin montaje fotométrico")

  def _safe_photometry_open_assistant(self, es_continua: bool = False) -> bool:
    """Ejecuta el asistente lineal de fotometría de forma aislada; si no hay cámara o se omite, desactiva hooks de cámara para esta sesión."""
    self._photometry_session_active = False
    modo_str = "secuencia continua" if es_continua else "autograbado"
    pm = getattr(self, 'photometry_mgr', None)
    if pm is None:
      return False
    res_foto = False
    try:
      res_foto = bool(pm.open_asistente_lineal(
        parent_widget=self,
        nombres_musculos=self.nombres_musculos,
        active_channel_indices=getattr(self, 'active_channel_indices', None),
        sujeto=self.autoforge_sujeto,
        prueba=self.autoforge_prueba,
      ))
    except Exception as exc:
      print(f"[FOTOMETRIA-AVISO] Error al abrir asistente lineal en {modo_str} (continuando sin cámara): {exc}")
      res_foto = False

    if res_foto:
      self._photometry_session_active = True
      try:
        pm.guardar_fotometria_inmediata(
          sujeto=self.autoforge_sujeto,
          prueba=self.autoforge_prueba,
          es_continua=bool(es_continua),
        )
      except Exception as exc:
        print(f"[FOTOMETRIA-AVISO] Persistencia inmediata falló en {modo_str}: {exc}")
    else:
      self._photometry_session_active = False
      try:
        if hasattr(pm, 'stop_worker') and callable(pm.stop_worker):
          pm.stop_worker()
      except Exception:
        pass
    self._imprimir_aviso_fotometria_lineal(asistente_confirmado=res_foto, modo=modo_str)
    return res_foto

  def _safe_photometry_start_session(self, session_name: str):
    """Inicia sesión de tomas fotométricas sólo si el montaje fue confirmado y la cámara está habilitada."""
    pm = getattr(self, 'photometry_mgr', None)
    if pm is None:
      return
    if not (getattr(self, '_photometry_session_active', False) or getattr(pm, 'is_enabled', False)):
      return
    try:
      pm.start_take_session(str(session_name))
    except Exception as exc:
      print(f"[FOTOMETRIA-AVISO] No se pudo iniciar sesión de cámara '{session_name}': {exc}")
      self._photometry_session_active = False

  def _safe_photometry_snapshot(self, fase: str, palabra: str, pulso: int):
    """Solicita un snapshot a la cámara de forma 100% segura, pasando t_daq_s y capturando cualquier excepción."""
    pm = getattr(self, 'photometry_mgr', None)
    if pm is None:
      return
    if not (getattr(self, '_photometry_session_active', False) and getattr(pm, 'is_enabled', False)):
      return
    try:
      t0 = getattr(self, 'recording_start_time', None)
      t_daq = max(0.0, time.perf_counter() - t0) if t0 is not None else 0.0
      sr = float(getattr(self, 'SAMPLE_RATE', 2000) or 2000.0)
      try:
        pm.request_snapshot(
          fase=str(fase),
          palabra=str(palabra),
          pulso=int(pulso),
          t_daq_s=float(t_daq),
          sample_rate=sr,
        )
      except TypeError:
        pm.request_snapshot(
          fase=str(fase),
          palabra=str(palabra),
          pulso=int(pulso),
          sample_rate=sr,
        )
    except Exception as exc:
      print(f"[FOTOMETRIA-AVISO] Snapshot omitido en fase '{fase}': {exc}")

  def _safe_photometry_export(self, output_dir: str):
    """Exporta metadatos fotométricos sólo si hubo sesión activa de cámara."""
    pm = getattr(self, 'photometry_mgr', None)
    if pm is None or not getattr(self, '_photometry_session_active', False):
      return None
    try:
      return pm.export_take_photometry(str(output_dir))
    except Exception as exc:
      print(f"[FOTOMETRIA-AVISO] Error al exportar fotometría en '{output_dir}': {exc}")
      return None

  def iniciar_autoforge(self):
    """
    Ejecuta la funcionalidad de iniciar_autoforge.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    try:
      import os
      if getattr(self, 'is_autoforge_running', False):
        self.detener_autoforge()
        return
        
      # Asegurar metronomo listo y visible para el autograbado
      self.stop_native_metronome()
      self.chk_use_metronome.setChecked(True)
      self.metronome_container.setVisible(True)
      self.metronome_container.show()
      self.right_side_panel.show()

      ruta_palabras = PALABRAS_TXT_PATH
      if not os.path.exists(ruta_palabras):
        with open(ruta_palabras, 'w', encoding='utf-8') as f:
          f.write("A\nE\nI\nO\nU\n")
        
      with open(ruta_palabras, 'r', encoding='utf-8') as f:
        self.autoforge_words = [line.strip() for line in f if line.strip()]
        
      if not self.autoforge_words:
        QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt está vacío.")
        return
        
      prot_cfg = self.config_mgr.get_protocol_config() if hasattr(self.config_mgr, 'get_protocol_config') else {}
      dialog = AutoForgeDialog(self)
      dialog.spin_reps.setValue(int(prot_cfg.get("repeticiones_discreta_default", 5)))
      dialog.spin_bpm.setValue(int(self.spin_bpm.value()))
      dialog.spin_subdivs.setValue(int(self.spin_metro_subdivs.value()))
      if dialog.exec() == QtWidgets.QDialog.Accepted:
        # --- NUEVO: Recargar palabras por si fueron editadas en la ventana ---
        with open(ruta_palabras, 'r', encoding='utf-8') as f:
          self.autoforge_words = [line.strip() for line in f if line.strip()]
          
        if not self.autoforge_words:
          QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt quedó vacío después de editar.")
          return

        self.autoforge_prueba = dialog.edit_prueba.text().strip()
        self.autoforge_sujeto = dialog.edit_sujeto.text().strip()
        self.autoforge_target_reps = dialog.spin_reps.value()
        self.autoforge_do_cvm = dialog.chk_calibrar_cvm.isChecked() if hasattr(dialog, 'chk_calibrar_cvm') else True
        self.cvm_completed = False
        self.final_noise_completed = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.autoforge_session_dirs = []
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
        bpm = dialog.spin_bpm.value()
        
        # Actualizar nombres de músculos elegidos
        muscles = dialog.get_muscle_names()
        if muscles:
          self.nombres_musculos = muscles
          canales_conf = self.config_mgr.get("canales") or {}
          for i, m in enumerate(muscles):
            hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and i < len(self.active_channel_indices)) else i
            if hasattr(self, 'nombres_musculos_hw') and hw_idx < len(self.nombres_musculos_hw):
              self.nombres_musculos_hw[hw_idx] = m
            key = f"Canal {hw_idx}"
            if key not in canales_conf:
              canales_conf[key] = {}
            canales_conf[key]["musculo"] = m
          self.config_mgr.set("canales", canales_conf)
          self.config_mgr.save()
          self.refrescar_interfaz_canales()

        # Actualizar el spinbox principal de BPM y subdivisiones si existen
        try:
          self.spin_bpm.setValue(bpm)
          if hasattr(self, 'spin_metro_subdivs') and hasattr(dialog, 'spin_subdivs'):
            self.spin_metro_subdivs.setValue(dialog.spin_subdivs.value())
          if hasattr(self, 'chk_metro_jitter') and hasattr(dialog, 'chk_jitter'):
            self.chk_metro_jitter.setChecked(dialog.chk_jitter.isChecked())
            self.spin_metro_jitter_pct.setValue(dialog.spin_jitter_pct.value())
            self.cmb_metro_jitter_mode.setCurrentText(dialog.cmb_jitter_mode.currentText())
          if hasattr(self, 'voice_cue_player') and self.voice_cue_player is not None:
            hablar_v = bool(getattr(dialog, 'chk_hablar_vocales', None) and dialog.chk_hablar_vocales.isChecked())
            hablar_f = bool(getattr(dialog, 'chk_hablar_fases', None) and dialog.chk_hablar_fases.isChecked())
            idx_sm = dialog.cmb_speaking_mode.currentIndex() if hasattr(dialog, 'cmb_speaking_mode') else 0
            if idx_sm == 2 or (not hablar_v and not hablar_f):
              self.voice_cue_player.speaking_mode = "mute"
              self.on_mute_speaking_toggled(True)
            elif hablar_v or idx_sm == 1:
              self.voice_cue_player.speaking_mode = "fases_y_continua"
              self.on_mute_speaking_toggled(False)
            else:
              self.voice_cue_player.speaking_mode = "solo_fases"
              self.on_mute_speaking_toggled(False)
          self.spoken_audio_log = []
          self._last_spoken_continuo_word = None
          self._pre_announced_continuo_beat = -1
          self.invalid_pulses_set = set()
          self.autoforge_use_debruijn = bool(hasattr(dialog, 'chk_debruijn') and dialog.chk_debruijn.isChecked())
          self.autoforge_full_sequence = list(self.autoforge_words)
          if hasattr(self, 'voice_cue_player'):
            self.voice_cue_player.reset_session_history()
          self._save_metronome_config() # Guardar archivo JSON con el BPM y subdivisiones
        except:
          pass
        
        self.autoforge_word_idx = 0
        self.autoforge_cvm_modo = "con_metronomo" if (hasattr(dialog, 'cmb_cvm_modo') and dialog.cmb_cvm_modo.currentIndex() == 1) else "guiado_sin_metronomo"
        self.reset_medidor_pca_vivo()

        # --- Asistente Lineal Simple de Fotometria (solo si el usuario lo activó en el diálogo) ---
        if hasattr(dialog, 'chk_abrir_fotometria') and dialog.chk_abrir_fotometria.isChecked():
          self._safe_photometry_open_assistant(es_continua=False)
        else:
          self._photometry_session_active = False

        # --- NUEVO: Mostrar Instrucciones del Experimento ---
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Instrucciones del Experimento")
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText(
            "<b>Bienvenido a la Sesión de Grabación (Discreta)</b><br><br>"
            "1. Relaja el rostro y evita moverte o tragar saliva.<br>"
            "2. Se medirán primero los niveles basales de ruido inter-pulso.<br>"
            "3. Se calibrará la Contracción Muscular Máxima (CVM) con vocales extremas (A, I, U).<br>"
            "4. A continuación, el sistema te guiará palabra por palabra.<br>"
            "5. Al finalizar se registrará nuevamente el ruido basal."
        )
        btn_continuar = msg_box.addButton("Continuar Grabación", QtWidgets.QMessageBox.AcceptRole)
        btn_entrenar = msg_box.addButton("Abrir Módulo de Entrenamiento", QtWidgets.QMessageBox.ActionRole)
        msg_box.exec()
        
        if msg_box.clickedButton() == btn_entrenar:
            import subprocess
            mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulo_de_entrenamiento.py")
            if os.path.exists(mod_path):
                subprocess.Popen([sys.executable, mod_path])
            return # Cancelar este inicio para que el usuario pueda entrenar

        if not self.is_acquiring:
          QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
          try:
            self.on_start_acq_click(start_metronome=False)
          finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not self.is_acquiring:
          return

        self.is_autoforge_running = True
        self._safe_photometry_start_session(f"{self.autoforge_prueba}_{self.autoforge_sujeto}")
        if hasattr(self, 'btn_pause'): self.btn_pause.setEnabled(True)
        self.btn_record.setEnabled(False)
        self.btn_autoforge_continuo.setEnabled(False)
        self.btn_autoforge.setText("DETENER AUTOGRABADO")
        self.btn_autoforge.setStyleSheet("background-color: #555555; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #ffffff; border-radius: 4px;")
        
        self.showMaximized() # Auto-maximizar al iniciar AutoForge
        
        self._iniciar_timer_global()
        self.estado_iniciar_palabra()
    except Exception as e:
      print(f"Error iniciando Autograbado: {e}")
      import traceback
      traceback.print_exc()
      
  def iniciar_autoforge_continuo(self):
    try:
      import os
      if getattr(self, 'is_autoforge_running', False):
        self.detener_autoforge()
        return
        
      # Asegurar metronomo listo y visible para el autograbado continuo
      self.stop_native_metronome()
      self.chk_use_metronome.setChecked(True)
      self.metronome_container.setVisible(True)
      self.metronome_container.show()
      self.right_side_panel.show()

      ruta_palabras = PALABRAS_TXT_PATH
      if not os.path.exists(ruta_palabras):
        with open(ruta_palabras, 'w', encoding='utf-8') as f:
          f.write("A\nE\nI\nO\nU\n")
        
      with open(ruta_palabras, 'r', encoding='utf-8') as f:
        self.autoforge_words = [line.strip() for line in f if line.strip()]
        
      if not self.autoforge_words:
        QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt está vacío.")
        return
        
      prot_cfg = self.config_mgr.get_protocol_config() if hasattr(self.config_mgr, 'get_protocol_config') else {}
      dialog = AutoForgeDialog(self)
      dialog.setWindowTitle("Configuración de Secuencia Continua")
      dialog.spin_reps.setValue(int(prot_cfg.get("repeticiones_continua_default", 25)))
      dialog.spin_bpm.setValue(int(self.spin_bpm.value()))
      dialog.spin_subdivs.setValue(int(self.spin_metro_subdivs.value()))
      if dialog.exec() == QtWidgets.QDialog.Accepted:
        with open(ruta_palabras, 'r', encoding='utf-8') as f:
          self.autoforge_words = [line.strip() for line in f if line.strip()]
          
        if not self.autoforge_words:
          QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt quedó vacío después de editar.")
          return

        self.autoforge_prueba = dialog.edit_prueba.text().strip()
        self.autoforge_sujeto = dialog.edit_sujeto.text().strip()
        self.autoforge_target_reps = dialog.spin_reps.value() # Total cycles
        self.autoforge_do_cvm = dialog.chk_calibrar_cvm.isChecked() if hasattr(dialog, 'chk_calibrar_cvm') else True
        self.cvm_completed = False
        self.final_noise_completed = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.autoforge_session_dirs = []
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
        bpm = dialog.spin_bpm.value()
        
        # Actualizar nombres de músculos elegidos
        muscles = dialog.get_muscle_names()
        if muscles:
          self.nombres_musculos = muscles
          canales_conf = self.config_mgr.get("canales") or {}
          for i, m in enumerate(muscles):
            hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and i < len(self.active_channel_indices)) else i
            if hasattr(self, 'nombres_musculos_hw') and hw_idx < len(self.nombres_musculos_hw):
              self.nombres_musculos_hw[hw_idx] = m
            key = f"Canal {hw_idx}"
            if key not in canales_conf:
              canales_conf[key] = {}
            canales_conf[key]["musculo"] = m
          self.config_mgr.set("canales", canales_conf)
          self.config_mgr.save()
          self.refrescar_interfaz_canales()

        try:
          self.spin_bpm.setValue(bpm)
          if hasattr(self, 'spin_metro_subdivs') and hasattr(dialog, 'spin_subdivs'):
            self.spin_metro_subdivs.setValue(dialog.spin_subdivs.value())
          if hasattr(self, 'chk_metro_jitter') and hasattr(dialog, 'chk_jitter'):
            self.chk_metro_jitter.setChecked(dialog.chk_jitter.isChecked())
            self.spin_metro_jitter_pct.setValue(dialog.spin_jitter_pct.value())
            self.cmb_metro_jitter_mode.setCurrentText(dialog.cmb_jitter_mode.currentText())
          if hasattr(self, 'voice_cue_player') and self.voice_cue_player is not None:
            hablar_v = bool(getattr(dialog, 'chk_hablar_vocales', None) and dialog.chk_hablar_vocales.isChecked())
            hablar_f = bool(getattr(dialog, 'chk_hablar_fases', None) and dialog.chk_hablar_fases.isChecked())
            idx_sm = dialog.cmb_speaking_mode.currentIndex() if hasattr(dialog, 'cmb_speaking_mode') else 0
            if idx_sm == 2 or (not hablar_v and not hablar_f):
              self.voice_cue_player.speaking_mode = "mute"
              self.on_mute_speaking_toggled(True)
            elif hablar_v or idx_sm == 1:
              self.voice_cue_player.speaking_mode = "fases_y_continua"
              self.on_mute_speaking_toggled(False)
            else:
              self.voice_cue_player.speaking_mode = "solo_fases"
              self.on_mute_speaking_toggled(False)
          self.spoken_audio_log = []
          self._last_spoken_continuo_word = None
          self._pre_announced_continuo_beat = -1
          self.invalid_pulses_set = set()
          self.autoforge_use_debruijn = bool(hasattr(dialog, 'chk_debruijn') and dialog.chk_debruijn.isChecked())
          if self.autoforge_use_debruijn:
            self.autoforge_full_sequence = generar_secuencia_balanceada_debruijn(
              self.autoforge_words, num_ciclos=self.autoforge_target_reps
            )
            print(f"[De Bruijn] Secuencia Continua balanceada generada ({len(self.autoforge_full_sequence)} pulsos): {' -> '.join(self.autoforge_full_sequence[:15])}...")
          else:
            self.autoforge_full_sequence = [
              self.autoforge_words[i % len(self.autoforge_words)]
              for i in range(len(self.autoforge_words) * self.autoforge_target_reps)
            ]
          if hasattr(self, 'voice_cue_player'):
            self.voice_cue_player.reset_session_history()
          self._save_metronome_config()
        except:
          pass
        
        self.autoforge_word_idx = 0
        self.autoforge_cvm_modo = "con_metronomo" if (hasattr(dialog, 'cmb_cvm_modo') and dialog.cmb_cvm_modo.currentIndex() == 1) else "guiado_sin_metronomo"
        self.reset_medidor_pca_vivo()

        # --- Asistente Lineal Simple de Fotometria (solo si el usuario lo activó en el diálogo) ---
        if hasattr(dialog, 'chk_abrir_fotometria') and dialog.chk_abrir_fotometria.isChecked():
          self._safe_photometry_open_assistant(es_continua=True)
        else:
          self._photometry_session_active = False

        # --- NUEVO: Mostrar Instrucciones del Experimento ---
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Instrucciones del Experimento")
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText(
            "<b>Bienvenido a la Sesión de Grabación (Secuencia Continua)</b><br><br>"
            "1. Relaja el rostro y evita moverte o tragar saliva.<br>"
            "2. Se medirán primero los niveles basales de ruido inter-pulso.<br>"
            "3. Se calibrará la Contracción Muscular Máxima (CVM) con vocales extremas (A, I, U).<br>"
            "4. A continuación, pronuncia las vocales mostradas en pantalla al ritmo del metrónomo. Mantén la fuerza constante.<br>"
            "5. Al finalizar se registrará nuevamente el ruido basal."
        )
        btn_continuar = msg_box.addButton("Continuar Grabación", QtWidgets.QMessageBox.AcceptRole)
        btn_entrenar = msg_box.addButton("Abrir Módulo de Entrenamiento", QtWidgets.QMessageBox.ActionRole)
        msg_box.exec()
        
        if msg_box.clickedButton() == btn_entrenar:
            import subprocess
            mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulo_de_entrenamiento.py")
            if os.path.exists(mod_path):
                subprocess.Popen([sys.executable, mod_path])
            return # Cancelar este inicio para que el usuario pueda entrenar

        if not self.is_acquiring:
          QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
          try:
            self.on_start_acq_click(start_metronome=False)
          finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        if not self.is_acquiring:
          return

        self.is_autoforge_running = True
        self._safe_photometry_start_session(f"{self.autoforge_prueba}_{self.autoforge_sujeto}")
        if hasattr(self, 'btn_pause'): self.btn_pause.setEnabled(True)
        self.is_autoforge_continuo = True
        self.last_continuo_beat = -1
        
        self.btn_record.setEnabled(False)
        self.btn_autoforge.setEnabled(False)
        self.btn_autoforge_continuo.setText("DETENER SEC. CONTINUA")
        self.btn_autoforge_continuo.setStyleSheet("background-color: #555555; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #ffffff; border-radius: 4px;")
        
        self.showMaximized() # Auto-maximizar al iniciar AutoForge
        
        self._iniciar_timer_global_continuo()
        self.estado_iniciar_secuencia_continua()
    except Exception as e:
      print(f"Error iniciando Autograbado Continuo: {e}")
      import traceback
      traceback.print_exc()

  def _iniciar_timer_global_continuo(self):
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    total_words = len(self.autoforge_words) * self.autoforge_target_reps
    noise_dur = self.spin_noise_duration.value()
    cvm_time = 32.5 if getattr(self, 'autoforge_do_cvm', True) else 0.0
    tiempo_por_secuencia = 3.0 + noise_dur + cvm_time + (3.0 * (60.0/bpm)) + (total_words * (60.0/bpm)) + noise_dur + 2.0
    self.tiempo_restante_global = int(tiempo_por_secuencia)
    
    self.autoforge_estado_actual_str = "Iniciando Secuencia..."
    
    if not hasattr(self, 'session_timer') or self.session_timer is None:
      self.session_timer = QtCore.QTimer(self)
      self.session_timer.timeout.connect(self._tick_session_timer)
    else:
      self.session_timer.stop()
    self.session_timer.start(1000)

  def estado_iniciar_secuencia_continua(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.is_recording = False
    self.label_rec_time.setVisible(True)
    # Ocultar hardware y trigger, mostrar vocal gigante en config_stack, dejando filtros y metrónomo visibles
    self.config_stack.setCurrentIndex(1)
    
    total_pulsos = len(self.autoforge_words) * self.autoforge_target_reps
    self.autoforge_estado_actual_str = f"Secuencia: {total_pulsos} pulsos"
    
    tr_str = getattr(self, 'tiempo_restante_str', '00:00')
    cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
    self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    
    self.autoforge_overlay.setText("<div align='center'>HAZ SILENCIO<br>PREPARANDO ENTORNO...</div>")
    self.autoforge_overlay.show()
    
    import subprocess, sys, os
    python_executable = sys.executable
    if getattr(sys, 'frozen', False):
      word_script_path = 'ventana_palabras.py'
    else:
      word_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ventana_palabras.py')
    texto_ventana = "<div align='center' style='font-size: 50px; font-weight: 900; color: #00FFCC;'>PREPARANDO ENTORNO...</div>"
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None # Mantener variable para no romper nada
    
    self._schedule_phase(3000, self.estado_grabar_ruido_continuo)

  def _procesar_ruido_inicial_50hz(self):
    self.line_noise_50hz_uV = [0.0] * self.NUM_CANALES
    for i in range(self.NUM_CANALES):
      if hasattr(self, 'raw_noise_accumulated') and i < len(self.raw_noise_accumulated) and self.raw_noise_accumulated[i]:
        raw_all = np.concatenate(self.raw_noise_accumulated[i])
        self.line_noise_50hz_uV[i] = calcular_ruido_linea_50hz(raw_all, self.SAMPLE_RATE)
        
        val_50 = self.line_noise_50hz_uV[i]
        if val_50 < 15.0:
          color_txt = "#00FFCC"
          color_bg = "#002211"
          estado_str = "ÓPTIMO"
        elif val_50 < 40.0:
          color_txt = "#FFFF00"
          color_bg = "#222200"
          estado_str = "MODERADO"
        else:
          color_txt = "#FF3300"
          color_bg = "#330000"
          estado_str = "ALTO"
          
        if hasattr(self, 'line_noise_status_labels') and i < len(self.line_noise_status_labels):
          self.line_noise_status_labels[i].setText(f"50 Hz: {val_50:.1f} µV ({estado_str})")
          self.line_noise_status_labels[i].setStyleSheet(f"color: {color_txt}; font-size: 11px; font-weight: bold;")
        if hasattr(self, 'line_noise_status_frames') and i < len(self.line_noise_status_frames):
          self.line_noise_status_frames[i].setStyleSheet(f"background-color: {color_bg}; border: 1.5px solid {color_txt}; border-radius: 3px; padding: 2px;")
    self._verificar_contacto_electrodos_post_ruido()

  def _verificar_contacto_electrodos_post_ruido(self):
    """
    Semáforo automático de calidad de contacto de electrodos tras el Ruido Basal Inicial:
    Verifica si algún canal EMG presenta ruido de 50 Hz excesivo (> 45 uV, electrodo o masa suelta)
    o desviación basal anormal (> 120 uV), registrándolo en metadata y avisando en vivo.
    """
    alertas = []
    diagnostico = {}
    for i in range(self.NUM_CANALES):
      hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and i < len(self.active_channel_indices)) else i
      musc = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
      es_mic = bool(self._is_mic_channel(i))
      v50 = float(self.line_noise_50hz_uV[i]) if (hasattr(self, 'line_noise_50hz_uV') and i < len(self.line_noise_50hz_uV)) else 0.0
      std_b = float(self.initial_noise_std[i]) if (hasattr(self, 'initial_noise_std') and i < len(self.initial_noise_std)) else 0.0
      mean_b = float(self.initial_noise_mean[i]) if (hasattr(self, 'initial_noise_mean') and i < len(self.initial_noise_mean)) else 0.0
      if not es_mic and (v50 >= 45.0 or std_b >= 120.0):
        estado = "ALERTA_ELECTRODO_O_REFERENCIA_FLOJA"
        alertas.append(f"ai{hw_idx} ({musc}): 50Hz={v50:.1f}µV, σ={std_b:.1f}µV")
      elif not es_mic and v50 >= 15.0:
        estado = "MODERADO"
      else:
        estado = "OPTIMO"
      diagnostico[f"canal_{i}"] = {
        "canal_fisico": f"ai{hw_idx}",
        "musculo": musc,
        "es_microfono": es_mic,
        "ruido_50hz_uV": round(v50, 2),
        "basal_mean_uV": round(mean_b, 2),
        "basal_std_uV": round(std_b, 2),
        "estado_contacto": estado,
      }
    self.diagnostico_contacto_electrodos = {
      "evaluado": True,
      "canales_con_alerta": alertas,
      "detalle_canales": diagnostico,
    }
    if alertas:
      msg_alerta = " | ".join(alertas)
      print(f"[SEMÁFORO ELECTRODOS - AVISO] Posible electrodo o referencia floja detectada en Ruido Basal: {msg_alerta}")
    else:
      print("[SEMÁFORO ELECTRODOS - OK] Todos los canales EMG presentan buen contacto e impedancia basal óptima.")

  def estado_iniciar_calibracion_cvm(self, target_post_cvm):
    if not getattr(self, 'is_autoforge_running', False): return
    self._post_cvm_target_method = target_post_cvm
    self.is_recording = False
    self.autoforge_estado_actual_str = "CONTRACCIÓN MUSCULAR MÁXIMA"
    modo_cvm = str(getattr(self, 'autoforge_cvm_modo', 'guiado_sin_metronomo'))

    self.cvm_step_idx = 0
    self.cvm_recording_chunks = []
    self.cvm_takes_records = []
    self.cvm_env_max_per_channel = [0.0] * self.NUM_CANALES
    self.cvm_env_p95_per_channel = [0.0] * self.NUM_CANALES
    self.cvm_env_p90_per_channel = [0.0] * self.NUM_CANALES
    self.cvm_env_rms_per_channel = [0.0] * self.NUM_CANALES
    self.cvm_energia_per_channel = [0.0] * self.NUM_CANALES
    self.cvm_auc_per_channel = [0.0] * self.NUM_CANALES

    self.is_cvm_active = True
    self.is_cvm_recording_take = False

    if modo_cvm == "guiado_sin_metronomo":
      # Modo SENIAM recomendado: Contracciones isométricas/sostenidas guiadas por temporizador (2.5s sostén + 2.5s descanso) SIN metrónomo
      self.stop_native_metronome()
      prot_cfg = self.config_mgr.get_protocol_config() if hasattr(self.config_mgr, 'get_protocol_config') else {}
      n_tomas = int(prot_cfg.get("cvm_tomas_por_vocal", 2))
      self._cvm_hold_ms = int(float(prot_cfg.get("cvm_duracion_contraccion_s", 2.5)) * 1000)
      self._cvm_rest_ms = int(float(prot_cfg.get("cvm_duracion_descanso_s", 2.5)) * 1000)
      self.cvm_sequence = [(v, t) for v in ('A', 'I', 'U') for t in range(1, n_tomas + 1)]
      self.label_rec_time.setText(f"CVM GUIADA SIN METRÓNOMO ({n_tomas} SOSTENES DE {self._cvm_hold_ms/1000:.1f}s POR VOCAL: A, I, U)...")
      self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFCC00;")
      self.autoforge_overlay.setText(
        "<div align='center'>"
        "<span style='color:#FFCC00; font-size:34px; font-weight:bold;'>CALIBRACIÓN CVM GUIADA (SIN METRÓNOMO)</span><br><br>"
        "<span style='color:#FFFFFF; font-size:22px;'>Mantén cada gesto articulatorio (A, I, U) de forma continua durante 2.5 s.</span><br>"
        "<span style='color:#00FFCC; font-size:18px;'>Sin pulsos rápidos para medir el plateau real P95 y evitar fatiga prematura.</span>"
        "</div>"
      )
      self.autoforge_overlay.show()
      self._safe_autoforge_single_shot(2000, self._cvm_guiado_preparar_toma)
    else:
      self.cvm_sequence = [
          ('A', 1), ('A', 2), ('A', 3),
          ('I', 1), ('I', 2), ('I', 3),
          ('U', 1), ('U', 2), ('U', 3)
      ]
      self.label_rec_time.setText("CONTRACCIÓN MUSCULAR MÁXIMA - 3 TOMAS POR VOCAL EXTREMA: A, I, U...")
      self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFCC00;")
      self.autoforge_overlay.setText(
          "<div align='center'>"
          "<span style='color:#FFCC00; font-size:36px; font-weight:bold;'>CONTRACCIÓN MUSCULAR MÁXIMA</span><br><br>"
          "<span style='color:#FFFFFF; font-size:24px;'>Iniciando protocolo de calibración al ritmo del metrónomo.</span><br>"
          "<span style='color:#00FFCC; font-size:20px;'>Sigue las instrucciones en pantalla : Vocales A, I, U.</span>"
          "</div>"
      )
      self.autoforge_overlay.show()
      self.start_native_metronome(count_in=4, force_start=True)

  def _cvm_guiado_preparar_toma(self):
    if not getattr(self, 'is_autoforge_running', False) or not getattr(self, 'is_cvm_active', False):
      return
    if self.cvm_step_idx >= len(self.cvm_sequence):
      self.is_cvm_active = False
      self.estado_finalizar_calibracion_cvm()
      return
    vocal, toma = self.cvm_sequence[self.cvm_step_idx]
    total_t = len(self.cvm_sequence)
    instrucciones = {
      'A': "Abre grande la boca y sostén /A/ con fuerza constante",
      'I': "Estira las comisuras (sonrisa tensa) y sostén /I/",
      'U': "Protruye y redondea los labios con fuerza en /U/"
    }
    desc = instrucciones.get(vocal, "")
    self.autoforge_overlay.setText(
      f"<div align='center'>"
      f"<span style='color:#FFCC00; font-size:28px; font-weight:bold;'>CVM GUIADA SIN METRÓNOMO — Toma {self.cvm_step_idx+1}/{total_t}</span><br><br>"
      f"<span style='color:#FFFF00; font-size:38px; font-weight:bold;'>DESCANSANDO — PREPÁRATE PARA /{vocal}/</span><br>"
      f"<span style='font-size:20px; color:#AAFFAA; font-weight:bold;'>{desc}</span>"
      f"</div>"
    )
    self.lbl_recording_space.setText(
      f"<div align='center'><span style='color:#FFFF00; font-size:36px; font-weight:bold;'>PREPÁRATE PARA SOSTENER: /{vocal}/ (Toma {self.cvm_step_idx+1}/{total_t})</span></div>"
    )
    self._publicar_estado_display_remoto(palabra=f"PREP /{vocal}/", pulso_actual=self.cvm_step_idx+1, total_pulsos=total_t, grabando=False)
    if int(toma) == 1 and hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say(f"di_{vocal.lower()}", f"Prepárate, sostén {vocal} con fuerza", fase="cvm_preparacion")
    self._safe_autoforge_single_shot(int(getattr(self, '_cvm_rest_ms', 2500)), self._cvm_guiado_iniciar_sosten)

  def _cvm_guiado_iniciar_sosten(self):
    if not getattr(self, 'is_autoforge_running', False) or not getattr(self, 'is_cvm_active', False):
      return
    vocal, toma = self.cvm_sequence[self.cvm_step_idx]
    total_t = len(self.cvm_sequence)
    hold_s = getattr(self, '_cvm_hold_ms', 2500) / 1000.0
    self.cvm_current_take_chunks = []
    self.is_cvm_recording_take = True
    self.beep_queue.put((880, 160))
    self.autoforge_overlay.setText(
      f"<div align='center'>"
      f"<span style='color:#FF3355; font-size:34px; font-weight:900;'>¡SOSTÉN LA CONTRACCIÓN AHORA! ({hold_s:.1f} s)</span><br>"
      f"<span style='color:#00FFFF; font-size:88px; font-weight:900;'>/{vocal}/</span><br>"
      f"<span style='color:#FFFFFF; font-size:18px;'>Toma {self.cvm_step_idx+1}/{total_t} — Mantén fuerza constante sin cortar</span>"
      f"</div>"
    )
    self.lbl_recording_space.setText(
      f"<div align='center'><span style='color:#FF3355; font-size:28px; font-weight:bold;'>¡SOSTÉN ({hold_s:.1f}s)!</span><br>"
      f"<span style='color:#00FFFF; font-size:76px; font-weight:900;'>/{vocal}/</span></div>"
    )
    self._publicar_estado_display_remoto(palabra=vocal, pulso_actual=self.cvm_step_idx+1, total_pulsos=total_t, flash=True, grabando=True)
    self._safe_autoforge_single_shot(int(getattr(self, '_cvm_hold_ms', 2500)), self._cvm_guiado_fin_sosten)

  def _cvm_guiado_fin_sosten(self):
    if not getattr(self, 'is_autoforge_running', False) or not getattr(self, 'is_cvm_active', False):
      return
    self.estado_cvm_procesar_toma()
    self.beep_queue.put((520, 120))
    self.cvm_step_idx += 1
    if self.cvm_step_idx >= len(self.cvm_sequence):
      self.is_cvm_active = False
      self.estado_finalizar_calibracion_cvm()
    else:
      self._cvm_guiado_preparar_toma()

  def estado_cvm_procesar_toma(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.is_cvm_recording_take = False
    
    vocal, toma = self.cvm_sequence[self.cvm_step_idx]
    if hasattr(self, 'cvm_current_take_chunks') and self.cvm_current_take_chunks:
      take_data = np.concatenate(self.cvm_current_take_chunks, axis=1)
      self.cvm_recording_chunks.append(take_data)
      
      # Convertir a uV usando factor de calibración por canal físico
      if hasattr(self, 'factores_calibracion_hw') and hasattr(self, 'active_channel_indices'):
        calib_factors = np.array([
          self.factores_calibracion_hw[self.active_channel_indices[ch]]
          if (ch < len(self.active_channel_indices) and self.active_channel_indices[ch] < len(self.factores_calibracion_hw) and self.factores_calibracion_hw[self.active_channel_indices[ch]] > 0)
          else 495.0
          for ch in range(self.NUM_CANALES)
        ]).reshape(-1, 1)
      else:
        calib_factors = 495.0
      take_uV = (take_data / calib_factors) * 1000000.0
      window_size_ms = self.spin_rms_window.value()
      window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
      if window_size < 1: window_size = 1
      
      # Calcular envolvente RMS por canal y parámetros robustos (pico, p95, rms, energía, auc)
      for ch in range(self.NUM_CANALES):
        env = calculate_rms_envelope(take_uV[ch], window_size)
        if len(env) > 0:
          peak_env = float(np.max(env))
          p95_env = float(np.percentile(env, 95))
          p90_env = float(np.percentile(env, 90))
          rms_env = float(np.sqrt(np.mean(env ** 2)))
          mean_env = float(np.mean(env))
          energia_env = float(np.sum(env ** 2) / float(self.SAMPLE_RATE))
          auc_env = float(np.sum(env) / float(self.SAMPLE_RATE))
        else:
          peak_env = p95_env = p90_env = rms_env = mean_env = energia_env = auc_env = 0.0

        if peak_env > self.cvm_env_max_per_channel[ch]: self.cvm_env_max_per_channel[ch] = peak_env
        if p95_env > self.cvm_env_p95_per_channel[ch]: self.cvm_env_p95_per_channel[ch] = p95_env
        if p90_env > self.cvm_env_p90_per_channel[ch]: self.cvm_env_p90_per_channel[ch] = p90_env
        if rms_env > self.cvm_env_rms_per_channel[ch]: self.cvm_env_rms_per_channel[ch] = rms_env
        if energia_env > self.cvm_energia_per_channel[ch]: self.cvm_energia_per_channel[ch] = energia_env
        if auc_env > self.cvm_auc_per_channel[ch]: self.cvm_auc_per_channel[ch] = auc_env

        self.cvm_takes_records.append({
          'toma_global': self.cvm_step_idx + 1,
          'vocal': vocal,
          'toma_vocal': toma,
          'canal': f"canal_{ch}",
          'peak_env_uV': peak_env,
          'p95_env_uV': p95_env,
          'p90_env_uV': p90_env,
          'rms_env_uV': rms_env,
          'mean_env_uV': mean_env,
          'energia_uV2_s': energia_env,
          'auc_uV_s': auc_env
        })

    # Nota: El metrónomo maneja la progresión (self.cvm_step_idx).

  def estado_finalizar_calibracion_cvm(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.cvm_completed = True
    
    # Excluir el canal de micrófono del cálculo del supremo global EMG
    emg_channels = [ch for ch in range(self.NUM_CANALES) if not self._is_mic_channel(ch)]
    if emg_channels:
      self.supremo_global_uV = float(max(self.cvm_env_max_per_channel[ch] for ch in emg_channels))
      self.supremo_global_p95_uV = float(max(self.cvm_env_p95_per_channel[ch] for ch in emg_channels))
      self.supremo_global_p90_uV = float(max(self.cvm_env_p90_per_channel[ch] for ch in emg_channels))
      self.supremo_global_rms_uV = float(max(self.cvm_env_rms_per_channel[ch] for ch in emg_channels))
      self.supremo_global_energia = float(max(self.cvm_energia_per_channel[ch] for ch in emg_channels))
      self.supremo_global_auc = float(max(self.cvm_auc_per_channel[ch] for ch in emg_channels))
    else:
      self.supremo_global_uV = 0.0
      self.supremo_global_p95_uV = 0.0
      self.supremo_global_p90_uV = 0.0
      self.supremo_global_rms_uV = 0.0
      self.supremo_global_energia = 0.0
      self.supremo_global_auc = 0.0
      
    c_str = ", ".join([f"C{ch}={self.cvm_env_max_per_channel[ch]:.1f}µV" for ch in range(self.NUM_CANALES)])
    print(f"[Calibración CVM (A, I, U)] Picos Envolvente: {c_str} | Canales EMG={emg_channels} | Supremo Pico={self.supremo_global_uV:.1f}µV | Supremo P95={self.supremo_global_p95_uV:.1f}µV")
    
    self.autoforge_overlay.setText(
        f"<div align='center'>"
        f"<span style='color:#FFCC00; font-size:32px; font-weight:bold;'>¡CONTRACCIÓN MUSCULAR MÁXIMA COMPLETADA!</span><br><br>"
        f"<span style='font-size:22px; color:#00FFCC;'>Supremo Global - Pico: {self.supremo_global_uV:.1f} µV</span><br>"
        f"<span style='font-size:18px; color:#AAFFAA;'>Supremo Global - P95 Robusto: {self.supremo_global_p95_uV:.1f} µV</span>"
        f"</div>"
    )
    self.autoforge_overlay.show()
    
    self._guardar_calibracion_csv()
    self._safe_autoforge_single_shot(2500, self._continuar_post_cvm)

  def _guardar_calibracion_csv(self):
    import threading
    def save_cvm_async():
      try:
        import os, csv
        from pathlib import Path
        from datetime import datetime
        fecha_str = datetime.now().strftime("%Y-%m-%d")
        prueba_actual = getattr(self, 'autoforge_prueba', 'Prueba1')
        sujeto_actual = getattr(self, 'autoforge_sujeto', 'Sujeto1')
        
        if getattr(self, 'is_autoforge_continuo', False):
          folder_name = f"SecuenciaContinua_{prueba_actual}_{sujeto_actual}"
        else:
          folder_name = f"{prueba_actual}_{sujeto_actual}"
          
        if getattr(sys, 'frozen', False):
          root_dir = os.path.dirname(os.path.abspath(sys.executable))
          if os.path.basename(root_dir) == "_internal":
            root_dir = os.path.dirname(root_dir)
        else:
          root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / folder_name
        os.makedirs(base_dir, exist_ok=True)
        
        # 1. Guardar calibracion.csv con desglose multiparamétrico de cada toma y resumen final
        calibracion_csv_path = os.path.join(base_dir, "calibracion.csv")
        with open(calibracion_csv_path, 'w', newline='', encoding='utf-8') as f:
          writer = csv.writer(f)
          writer.writerow([
            'toma_global', 'vocal', 'toma_vocal', 'canal',
            'pico_envolvente_uV', 'percentil_95_uV', 'percentil_90_uV',
            'rms_envolvente_uV', 'media_envolvente_uV', 'energia_uV2_s', 'auc_uV_s'
          ])
          for r in getattr(self, 'cvm_takes_records', []):
            writer.writerow([
              r['toma_global'], r['vocal'], r['toma_vocal'], r['canal'],
              f"{r['peak_env_uV']:.2f}", f"{r['p95_env_uV']:.2f}", f"{r['p90_env_uV']:.2f}",
              f"{r['rms_env_uV']:.2f}", f"{r['mean_env_uV']:.2f}",
              f"{r['energia_uV2_s']:.2f}", f"{r['auc_uV_s']:.2f}"
            ])
          writer.writerow([])
          writer.writerow(['RESUMEN_MAXIMOS_POR_CANAL', 'canal', 'max_pico_envolvente_uV', 'max_p95_uV', 'max_p90_uV', 'max_rms_uV', 'max_energia_uV2_s', 'max_auc_uV_s'])
          for ch in range(self.NUM_CANALES):
            pico = self.cvm_env_max_per_channel[ch] if ch < len(self.cvm_env_max_per_channel) else 0.0
            p95 = self.cvm_env_p95_per_channel[ch] if ch < len(self.cvm_env_p95_per_channel) else 0.0
            p90 = self.cvm_env_p90_per_channel[ch] if ch < len(self.cvm_env_p90_per_channel) else 0.0
            rms = self.cvm_env_rms_per_channel[ch] if ch < len(self.cvm_env_rms_per_channel) else 0.0
            ene = self.cvm_energia_per_channel[ch] if ch < len(self.cvm_energia_per_channel) else 0.0
            auc = self.cvm_auc_per_channel[ch] if ch < len(self.cvm_auc_per_channel) else 0.0
            writer.writerow(['MAXIMO_CANAL', f"canal_{ch}", f"{pico:.2f}", f"{p95:.2f}", f"{p90:.2f}", f"{rms:.2f}", f"{ene:.2f}", f"{auc:.2f}"])
          writer.writerow([])
          writer.writerow(['SUPREMO_GLOBAL_TRICANAL', 'parametro', 'valor_supremo', 'unidad'])
          writer.writerow(['SUPREMO_GLOBAL', 'pico_envolvente', f"{getattr(self, 'supremo_global_uV', 0.0):.2f}", 'uV'])
          writer.writerow(['SUPREMO_GLOBAL', 'percentil_95', f"{getattr(self, 'supremo_global_p95_uV', 0.0):.2f}", 'uV'])
          writer.writerow(['SUPREMO_GLOBAL', 'percentil_90', f"{getattr(self, 'supremo_global_p90_uV', 0.0):.2f}", 'uV'])
          writer.writerow(['SUPREMO_GLOBAL', 'rms_envolvente', f"{getattr(self, 'supremo_global_rms_uV', 0.0):.2f}", 'uV'])
          writer.writerow(['SUPREMO_GLOBAL', 'energia', f"{getattr(self, 'supremo_global_energia', 0.0):.2f}", 'uV2*s'])
          writer.writerow(['SUPREMO_GLOBAL', 'auc', f"{getattr(self, 'supremo_global_auc', 0.0):.2f}", 'uV*s'])

        print(f"[Calibración CVM] Guardado exitoso de {calibracion_csv_path}")

        # 2. Guardar datos raw en WAV y CSV crudo si existen bloques grabados
        if hasattr(self, 'cvm_recording_chunks') and self.cvm_recording_chunks:
          full_cvm = np.concatenate(self.cvm_recording_chunks, axis=1)
          guardar_grabacion_csv([full_cvm], self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "calibracion_raw")
          guardar_grabacion_wav([full_cvm], self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "calibracion")
      except Exception as e:
        print(f"[Error Guardar calibracion.csv] {e}")

    threading.Thread(target=save_cvm_async, daemon=True).start()

  def _continuar_post_cvm(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.autoforge_overlay.hide()
    target = getattr(self, '_post_cvm_target_method', None)
    self._post_cvm_target_method = None
    if callable(target):
      target()

  # ==============================================================================
  # --- RUTINA DE RUIDO FINAL (5s REPOSO AL CONCLUIR LA SESIÓN) ---
  # ==============================================================================
  def estado_grabar_ruido_final(self, target_post_noise):
    if not getattr(self, 'is_autoforge_running', False): return
    self._post_final_noise_target_method = target_post_noise
    self.is_recording = False
    self.label_rec_time.setText("GRABANDO RUIDO FINAL...")
    self.lbl_recording_space.setText("<div align='center' style='font-size: 50px; font-weight: 900; color: #FFFF00;'>GRABANDO RUIDO FINAL...</div>")
    
    self.autoforge_overlay.setText("<div align='center'>HAZ SILENCIO<br>GRABANDO RUIDO FINAL (5s)...</div>")
    self.autoforge_overlay.show()
    
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say("haz_silencio", "Haz silencio, grabando ruido final", fase="ruido_basal_final")
    self.final_raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.is_recording_final_noise = True

    # Hook fotometria: captura segura de ruido basal final
    self._safe_photometry_snapshot(fase="ruido_basal_final", palabra="ruido_final", pulso=0)
    
    noise_dur = getattr(self, 'spin_noise_duration', None)
    dur_sec = noise_dur.value() if noise_dur is not None else 5.0
    self._schedule_phase(int(dur_sec * 1000), self.estado_finalizar_ruido_final)

  def estado_finalizar_ruido_final(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.is_recording_final_noise = False
    self.final_noise_completed = True
    self.autoforge_overlay.hide()
    
    self.final_noise_mean = [0.0] * self.NUM_CANALES
    self.final_noise_std = [0.0] * self.NUM_CANALES
    self.final_noise_50hz = [0.0] * self.NUM_CANALES
    
    for i in range(self.NUM_CANALES):
      if hasattr(self, 'final_raw_noise_accumulated') and i < len(self.final_raw_noise_accumulated) and self.final_raw_noise_accumulated[i]:
        all_fn = np.concatenate(self.final_raw_noise_accumulated[i])
        self.final_noise_50hz[i] = calcular_ruido_linea_50hz(all_fn, self.SAMPLE_RATE)
        
        window_size_ms = self.spin_rms_window.value()
        window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
        if window_size < 1: window_size = 1
        fn_env = calculate_rms_envelope(all_fn, window_size)
        self.final_noise_mean[i] = float(np.mean(fn_env))
        self.final_noise_std[i] = float(np.std(fn_env))

    print(f"[Ruido Final] Evaluación completada. 50Hz: {self.final_noise_50hz}")
    
    # Actualizar metadata.json con ruido final en todas las carpetas guardadas en la sesión
    final_noise_dict = {
        "mean_uV": getattr(self, 'final_noise_mean', [0.0]*self.NUM_CANALES),
        "std_uV": getattr(self, 'final_noise_std', [0.0]*self.NUM_CANALES),
        "line_noise_50hz_uV": getattr(self, 'final_noise_50hz', [0.0]*self.NUM_CANALES)
    }
    for session_dir in getattr(self, 'autoforge_session_dirs', []):
      try:
        root_m_path = os.path.join(str(session_dir), "metadata.json")
        if os.path.exists(root_m_path):
          with open(root_m_path, 'r', encoding='utf-8') as f:
            m_data = json.load(f)
          if "ruido_basal" not in m_data:
            m_data["ruido_basal"] = {}
          m_data["ruido_basal"]["final"] = final_noise_dict
          with open(root_m_path, 'w', encoding='utf-8') as f:
            json.dump(m_data, f, indent=4)

        for ch in range(self.NUM_CANALES):
          ch_m_path = os.path.join(str(session_dir), f"canal_{ch}", "metadata.json")
          if os.path.exists(ch_m_path):
            with open(ch_m_path, 'r', encoding='utf-8') as f:
              m_ch = json.load(f)
            if "ruido_basal" not in m_ch:
              m_ch["ruido_basal"] = {}
            m_ch["ruido_basal"]["final"] = final_noise_dict
            with open(ch_m_path, 'w', encoding='utf-8') as f:
              json.dump(m_ch, f, indent=4)
      except Exception as e:
        print(f"[Ruido Final] Error actualizando metadata en {session_dir}: {e}")
    
    target = getattr(self, '_post_final_noise_target_method', None)
    if callable(target):
      target()

  def estado_grabar_ruido_continuo(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.autoforge_overlay.setText("") 
    self.autoforge_overlay.hide()
    self.label_rec_time.setVisible(True) 
    primera_v = self.autoforge_words[0].upper() if getattr(self, 'autoforge_words', None) else "A"
    tira_init = self._construir_tira_secuencia_html(getattr(self, 'autoforge_words', ["A"]), 0)
    self.lbl_recording_space.setStyleSheet(
      "background-color: #141405; color: #FFFF00; font-family: 'Courier New', monospace; "
      "border: 2px solid #FFFF00; border-radius: 6px; padding: 4px;"
    )
    self.lbl_recording_space.setText(
      f"<div align='center' style='line-height:1.15;'>"
      f"<span style='font-size:28px; font-weight:900; color:#FFFF00;'>HAZ SILENCIO &mdash; GRABANDO RUIDO BASAL</span><br>"
      f"<span style='font-size:16px; color:#00FFCC;'>Relaja el rostro. Al terminar comenzará la secuencia con /{primera_v}/:</span><br>"
      f"<span>{tira_init}</span>"
      f"</div>"
    )
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say("haz_silencio", "Haz silencio, grabando ruido basal", fase="ruido_basal_inicial")
    
    self.current_recording = []
    self._offset_muestras_ancla = 0
    
    self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.noise_levels = [0.0] * self.NUM_CANALES
    self.noise_calculated = False
    self.noise_initialized = False
    self.initial_noise_mean = [0.0] * self.NUM_CANALES
    self.initial_noise_std = [0.0] * self.NUM_CANALES
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    self.global_snr_acumulado = [0.0] * self.NUM_CANALES
    self.global_snr_count = [0] * self.NUM_CANALES
    
    import time
    self.recording_start_time = time.perf_counter() 
    self.is_recording = True

    # Hook fotometria: captura segura de ruido basal continuo
    self._safe_photometry_snapshot(fase="ruido_basal", palabra="ruido_inicial", pulso=0)
    
    # --- Resetear escala Y a rango base calibrado (500 µV) ---
    if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
      self.plot.setYRange(0.0, 500.0, padding=0)
    else:
      self.plot.setYRange(-500.0, 500.0, padding=0)
    self._held_plot_ymax = 500.0
    
    noise_dur = self.spin_noise_duration.value()
    self._schedule_phase(int(noise_dur * 1000), self.estado_mostrar_preparate_continuo)

  def estado_mostrar_preparate_continuo(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.is_recording = False
    self.autoforge_ruido_guardado = self.current_recording.copy()
    self.autoforge_estado_actual_str = "ESCUCHA EL METRÓNOMO..."
    
    tr_str = getattr(self, 'tiempo_restante_str', '00:00')
    cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
    self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF8800;")
    
    if not getattr(self, 'noise_calculated', False):
      import numpy as np
      for i in range(self.NUM_CANALES):
        if self.noise_data_accumulated[i]:
          all_noise = np.concatenate(self.noise_data_accumulated[i])
          window_size_ms = self.spin_rms_window.value()
          window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
          if window_size < 1: window_size = 1
          all_noise_env = calculate_rms_envelope(all_noise, window_size)
          
          self.noise_levels[i] = np.mean(all_noise_env)
          self.initial_noise_mean[i] = self.noise_levels[i]
          self.initial_noise_std[i] = np.std(all_noise_env)
          
          is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
          show_noise = self.chk_show_noise.isChecked()
          self.noise_lines[i].setPos(self.noise_levels[i])
          if is_rms:
            self.noise_lines_neg[i].setVisible(False)
            self.noise_regions[i].setRegion([0.0, self.noise_levels[i]])
          else:
            self.noise_lines_neg[i].setPos(-self.noise_levels[i])
            self.noise_lines_neg[i].setVisible(show_noise)
            self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          self.noise_lines[i].setVisible(show_noise)
          self.noise_regions[i].setVisible(show_noise)
          
          if hasattr(self, 'noise_status_labels'):
            self.noise_status_labels[i].setText(f"Ruido Base: x̄={self.initial_noise_mean[i]:.1f}µV, s={self.initial_noise_std[i]:.1f}µV")
            self.noise_status_labels[i].setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold; background-color: #111111; border: 1px solid #00FFFF;")
      
      if self.NUM_CANALES > 0:
        ruido_maximo_std = max(getattr(self, 'initial_noise_std', [0]))
        if ruido_maximo_std > 0:
          self.spin_peak_th.setValue(ruido_maximo_std * 5.0)
      self.noise_calculated = True
      self._procesar_ruido_inicial_50hz()
    
    # --- CVM HOOK ---
    if getattr(self, 'autoforge_do_cvm', True) and not getattr(self, 'cvm_completed', False):
      self.estado_iniciar_calibracion_cvm(target_post_cvm=self.estado_mostrar_preparate_continuo)
      return
    
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    beat_interval_s = 60.0 / bpm
    
    try:
      if hasattr(self, 'word_window_process') and self.word_window_process:
        self.word_window_process.stdin.write("PREPÁRATE\\n")
        self.word_window_process.stdin.flush()
    except: pass
    
    self.autoforge_overlay.hide()
    full_seq_init = getattr(self, 'autoforge_full_sequence', None) or getattr(self, 'autoforge_words', ["A"])
    primera_vocal = full_seq_init[0].upper() if full_seq_init else "A"
    self._last_spoken_continuo_word = primera_vocal
    self._pre_announced_continuo_beat = 1
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say(f"di_{primera_vocal.lower()}", f"Prepárate, di {primera_vocal}", fase="preparacion_secuencia_continua")
    y_center = self._get_plot_y_center()
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, y_center)
    self.countdown_text.setText("PREPÁRATE...\n3")
    self.countdown_text.show()
    self._on_count_in_finished_cb = self.estado_iniciar_grabacion_continua
    self.start_native_metronome(count_in=4, force_start=True)

  def estado_iniciar_grabacion_continua(self, anchor_perf=None):
    if not getattr(self, 'is_autoforge_running', False): return
    self.autoforge_overlay.hide()
    if hasattr(self, 'countdown_text'):
      self.countdown_text.hide()
    
    import time
    self.recording_start_time = float(anchor_perf) if anchor_perf is not None else time.perf_counter()
    self.is_recording = True
    self.is_finishing_sequence = False
    
    self.current_recording = getattr(self, 'autoforge_ruido_guardado', []).copy()
    # El .wav de la secuencia arranca con el ruido basal ya grabado
    self._offset_muestras_ancla = self._contar_muestras_recording()
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    
    self.vocal_cue_log = []
    self.sync_evaluation_log = []
    self.label_rec_time.setText("GRABANDO SECUENCIA CONTINUA...")
    # Si se invoca sin count-in (anchor_perf is None), disparar destello y snapshot inicial;
    # cuando viene del count-in de on_metro_beat, el propio on_metro_beat dispara Pulso 1 sin duplicar.
    if anchor_perf is None and len(self.autoforge_words) > 0:
        palabra_0 = self.autoforge_words[0].upper()
        palabra_1 = self.autoforge_words[1 % len(self.autoforge_words)].upper()
        total_p = len(self.autoforge_words) * self.autoforge_target_reps
        self._disparar_destello_vocal(
            palabra_actual=palabra_0,
            pulso_actual=1,
            total_pulsos=total_p,
            palabra_siguiente=palabra_1,
            ciclo_actual=1,
            total_ciclos=self.autoforge_target_reps
        )
        self._safe_photometry_snapshot(fase="onset", palabra=str(palabra_0), pulso=1)

  def estado_guardar_secuencia_continua(self):
    if not getattr(self, 'is_autoforge_running', False): return
    # Detener el metrónomo INMEDIATAMENTE al terminar el último pulso de vocal
    # para que NO siga sonando ni agregando pulsos durante los 5s de Ruido Basal Final.
    self.is_recording = False
    self.stop_native_metronome()
    if not getattr(self, 'final_noise_completed', False):
      self.estado_grabar_ruido_final(target_post_noise=self.estado_guardar_secuencia_continua)
      return
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    self.autoforge_estado_actual_str = "GUARDANDO SECUENCIA"
    self.autoforge_overlay.setText("<div align='center'>SECUENCIA COMPLETADA<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    
    local_recording = self.current_recording.copy()
    self.current_recording = []
    local_ruido = getattr(self, 'autoforge_ruido_guardado', []).copy()
    local_pulsos_metro = list(getattr(self, '_metro_pulsos', []) or [])
    local_beat_timestamps = list(getattr(self, 'metro_beat_timestamps_s', []) or [])
    local_beat_samples = list(getattr(self, 'metro_beat_sample_indices', []) or [])
    local_beat_intervals = list(getattr(self, 'metro_applied_intervals_s', []) or [])
    local_stats_time = [list(x) for x in getattr(self, 'stats_time', [])]
    local_stats_snr = [list(x) for x in getattr(self, 'stats_snr', [])]
    local_stats_noise_mean = [list(x) for x in getattr(self, 'stats_noise_mean', [])]
    local_stats_noise_std = [list(x) for x in getattr(self, 'stats_noise_std', [])]
    local_vocal_cues = list(getattr(self, 'vocal_cue_log', []) or [])
    local_sync_eval = list(getattr(self, 'sync_evaluation_log', []) or [])
    
    import threading
    def guardar_async():
      import os, json
      from pathlib import Path
      from datetime import datetime
      fecha_str = datetime.now().strftime("%Y-%m-%d")
      folder_name = f"SecuenciaContinua_{self.autoforge_prueba}_{self.autoforge_sujeto}"
      if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(root_dir) == "_internal":
          root_dir = os.path.dirname(root_dir)
      else:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / folder_name
      os.makedirs(base_dir, exist_ok=True)
      if hasattr(self, 'autoforge_session_dirs'):
        self.autoforge_session_dirs.append(str(base_dir))
      
      full_seq_saved = list(getattr(self, 'autoforge_full_sequence', None) or [])
      if not full_seq_saved:
        total_pulsos = len(self.autoforge_words) * self.autoforge_target_reps
        full_word_sequence = [self.autoforge_words[i % len(self.autoforge_words)] for i in range(total_pulsos)]
      else:
        full_word_sequence = full_seq_saved
        total_pulsos = len(full_word_sequence)
          
      now = datetime.now()
      muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
      muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}
      
      calib_cvm_dict = {
          "calibrado": getattr(self, 'cvm_completed', False),
          "metodo": "envolvente_rms_multiparametrico",
          "tomas_por_vocal": 3,
          "canales": {
              f"canal_{i}": {
                  "pico_envolvente_uV": self.cvm_env_max_per_channel[i] if hasattr(self, 'cvm_env_max_per_channel') and i < len(self.cvm_env_max_per_channel) else 0.0,
                  "percentil_95_uV": self.cvm_env_p95_per_channel[i] if hasattr(self, 'cvm_env_p95_per_channel') and i < len(self.cvm_env_p95_per_channel) else 0.0,
                  "percentil_90_uV": self.cvm_env_p90_per_channel[i] if hasattr(self, 'cvm_env_p90_per_channel') and i < len(self.cvm_env_p90_per_channel) else 0.0,
                  "rms_envolvente_uV": self.cvm_env_rms_per_channel[i] if hasattr(self, 'cvm_env_rms_per_channel') and i < len(self.cvm_env_rms_per_channel) else 0.0,
                  "energia_uV2_s": self.cvm_energia_per_channel[i] if hasattr(self, 'cvm_energia_per_channel') and i < len(self.cvm_energia_per_channel) else 0.0,
                  "auc_uV_s": self.cvm_auc_per_channel[i] if hasattr(self, 'cvm_auc_per_channel') and i < len(self.cvm_auc_per_channel) else 0.0
              } for i in range(self.NUM_CANALES)
          },
          "supremo_global_uV": getattr(self, 'supremo_global_uV', 0.0),
          "supremo_global_p95_uV": getattr(self, 'supremo_global_p95_uV', 0.0),
          "supremo_global_p90_uV": getattr(self, 'supremo_global_p90_uV', 0.0),
          "supremo_global_rms_uV": getattr(self, 'supremo_global_rms_uV', 0.0),
          "supremo_global_energia_uV2_s": getattr(self, 'supremo_global_energia', 0.0),
          "supremo_global_auc_uV_s": getattr(self, 'supremo_global_auc', 0.0)
      }

      _t0_perf = getattr(self, 'recording_start_time', None) or time.perf_counter()
      _t0_epoch = time.time() - (time.perf_counter() - _t0_perf)
      metadata = {
        "measurement_date": now.isoformat(),
        "t0_epoch": _t0_epoch,
        "t0_perf_counter": _t0_perf,
        "daq_sample_rate": self.SAMPLE_RATE,
        "metronomo_pulsos": len(local_pulsos_metro),
        "metronomo_pulsos_csv": "pulsos_metronomo.csv" if local_pulsos_metro else None,
        "beat_timestamps_s": local_beat_timestamps,
        "beat_sample_indices": local_beat_samples,
        "beat_intervals_s": local_beat_intervals,
        "jitter_config": {
            "enabled": bool(hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked()),
            "pct": float(self.spin_metro_jitter_pct.value()) if hasattr(self, 'spin_metro_jitter_pct') else 0.0,
            "mode": self.cmb_metro_jitter_mode.currentText() if hasattr(self, 'cmb_metro_jitter_mode') else "exponential_hazard"
        },
        "eventos_vocales_metronomo": local_vocal_cues,
        "sincronizacion_sujeto_vs_metronomo": local_sync_eval,
        "timestamp": int(now.timestamp()),
        "sample_rate": self.SAMPLE_RATE,
        "channels": self.CANALES_DAQ,
        "muscles": muscles_list,
        "muscles_map": muscles_map,
        "bpm": self.spin_bpm.value(),
        "noise_seconds": self.spin_noise_duration.value(),
        "pulse_count": total_pulsos,
        "is_formal": True,
        "sujeto": self.autoforge_sujeto,
        "letra": "SecuenciaContinua",
        "prueba": self.autoforge_prueba,
        "comentario": "Grabado mediante AutoForge Secuencia Continua",
        "orden_vocales_base": list(self.autoforge_words),
        "valid_words": full_word_sequence,
        "ruido_basal": {
            "inicial": {
                "mean_uV": getattr(self, 'initial_noise_mean', [0.0]*self.NUM_CANALES),
                "std_uV": getattr(self, 'initial_noise_std', [0.0]*self.NUM_CANALES),
                "line_noise_50hz_uV": getattr(self, 'line_noise_50hz_uV', [0.0]*self.NUM_CANALES)
            },
            "final": {
                "mean_uV": getattr(self, 'final_noise_mean', [0.0]*self.NUM_CANALES),
                "std_uV": getattr(self, 'final_noise_std', [0.0]*self.NUM_CANALES),
                "line_noise_50hz_uV": getattr(self, 'final_noise_50hz', [0.0]*self.NUM_CANALES)
            }
        },
        "calibracion_cvm": calib_cvm_dict
      }
      photo_meta = self._safe_photometry_export(str(base_dir))
      if photo_meta is not None:
        metadata["fotometria_espacial"] = photo_meta
      if hasattr(self, 'last_separability_metrics') and self.last_separability_metrics is not None:
        metadata["diagnostico_separabilidad_vocales"] = self.last_separability_metrics
      
      for i in range(self.NUM_CANALES):
        channel_output_dir = os.path.join(base_dir, f"canal_{i}")
        os.makedirs(channel_output_dir, exist_ok=True)
        metadata_path = os.path.join(channel_output_dir, "metadata.json")
        ch_metadata = metadata.copy()
        ch_metadata["canal"] = f"canal_{i}"
        ch_metadata["musculo"] = self.nombres_musculos[i % len(self.nombres_musculos)]
        ch_metadata["physical_channel"] = self.CANALES_DAQ[i] if i < len(self.CANALES_DAQ) else f"ai{i}"
        try:
          with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ch_metadata, f, indent=4)
        except: pass

      try:
        base_meta_path = os.path.join(base_dir, "metadata.json")
        with open(base_meta_path, 'w', encoding='utf-8') as f:
          json.dump(metadata, f, indent=4)
      except: pass

      try:
        escribir_pulsos_metronomo_csv(local_pulsos_metro, str(base_dir))
      except Exception as e:
        print(f"[Aviso] No se pudieron guardar los pulsos del metronomo: {e}")

      try:
        emg_chs = [c for c in range(self.NUM_CANALES) if not self._is_mic_channel(c)]
        gains_ch = [
          self.factores_calibracion_hw[self.active_channel_indices[c]]
          if (hasattr(self, 'active_channel_indices') and c < len(self.active_channel_indices) and self.active_channel_indices[c] < len(self.factores_calibracion_hw))
          else 495.0
          for c in range(self.NUM_CANALES)
        ]
        ds_summary = exportar_segmentos_dataset_y_fatiga(
          datos_recording=local_recording,
          sample_rate=self.SAMPLE_RATE,
          output_dir=str(base_dir),
          num_canales=self.NUM_CANALES,
          vocal_cue_log=local_vocal_cues,
          sync_evaluation_log=local_sync_eval,
          invalid_pulses_set=set(getattr(self, 'invalid_pulses_set', set()) or set()),
          factores_calibracion=gains_ch,
          emg_channel_indices=emg_chs,
          initial_noise_mean=getattr(self, 'initial_noise_mean', []),
          initial_noise_std=getattr(self, 'initial_noise_std', []),
        )
        self.last_dataset_summary = ds_summary
        metadata["dataset_y_fatiga"] = {
          k: v for k, v in ds_summary.items() if k != "segmentos"
        }
        metadata["semaforo_contacto_electrodos"] = getattr(self, 'diagnostico_contacto_electrodos', {})
        metadata["secuencia_debruijn_activa"] = bool(getattr(self, 'autoforge_use_debruijn', True))
      except Exception as e:
        print(f"[Aviso Dataset] No se pudo exportar segmentos_dataset.csv en continua: {e}")

      try: guardar_grabacion_csv(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion", canales_daq=self.CANALES_DAQ)
      except: pass
      
      try: guardar_grabacion_wav(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      try:
        generar_grafico_grabacion(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
        ruta_photo = os.path.join(str(base_dir), "photo.png")
        if os.path.exists(ruta_photo):
          self.mostrar_preview_signal.emit(ruta_photo)
      except: pass
      
      try: generar_grafico_estadisticas(local_stats_time, local_stats_snr, local_stats_noise_mean, local_stats_noise_std, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      # 5. Guardar archivo de texto con el registro del orden ejecutado y todo el protocolo
      try:
        orden_txt_path = os.path.join(str(base_dir), "registro_orden_secuencia.txt")
        with open(orden_txt_path, 'w', encoding='utf-8') as f_ord:
          f_ord.write("SECUENCIA CONTINUA - REGISTRO DE ORDEN DE VOCALES Y PROTOCOLO COMPLETO\n")
          f_ord.write(f"Fecha y Hora: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
          f_ord.write(f"Sujeto: {self.autoforge_sujeto} | Prueba: {self.autoforge_prueba}\n")
          f_ord.write(f"Orden Base de Vocales: {' -> '.join(self.autoforge_words)}\n")
          f_ord.write(f"Repeticiones (Ciclos): {self.autoforge_target_reps}\n")
          f_ord.write(f"Total de Pulsos: {total_pulsos}\n\n")
          f_ord.write("Secuencia detallada pulso a pulso:\n")
          for idx_p, w_p in enumerate(full_word_sequence, 1):
            ciclo_p = ((idx_p - 1) // len(self.autoforge_words)) + 1
            pos_p = ((idx_p - 1) % len(self.autoforge_words)) + 1
            f_ord.write(f"  Pulso {idx_p:03d} | Ciclo {ciclo_p:02d} | Pos {pos_p}: /{w_p}/\n")
          f_ord.write("\n" + self._formatear_informe_protocolo_completo_txt(metadata_dict=metadata, modo_ejecucion="Secuencia Continua"))
      except Exception: pass

      # 6. Persistir notas de experimento en metadata.json y archivo de texto
      try:
        self.persistir_notas_experimento(str(base_dir), metadata_base=metadata)
      except Exception: pass
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    self._schedule_phase(5000, self.estado_finalizar_secuencia_continua)

  def _imprimir_registro_orden(self, modo: str = "Autograbado"):
    """
    Imprime en consola y terminal el registro formal del orden en que se grabó cada vocal/palabra,
    y guarda un archivo de texto en la carpeta de la sesión de hoy.
    """
    import os
    from datetime import datetime
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    fecha_str = now.strftime("%Y-%m-%d")

    words = getattr(self, 'autoforge_words', [])
    sujeto = getattr(self, 'autoforge_sujeto', 'Sujeto')
    prueba = getattr(self, 'autoforge_prueba', 'Prueba')
    reps = getattr(self, 'autoforge_target_reps', 1)

    if not words:
      words = ["A", "E", "I", "O", "U"]

    orden_flechas = " -> ".join([f"/{w}/" for w in words])

    banner = [
      "=" * 72,
      f"REGISTRO OFICIAL DE SECUENCIA DE VOCALES ({modo.upper()})",
      "=" * 72,
      f"Fecha y Hora : {now_str}",
      f"Sujeto       : {sujeto}",
      f"Prueba       : {prueba}",
      f"Modo         : {modo}",
      f"Repeticiones : {reps}",
      "-" * 72,
      f"ORDEN DE VOCALES: {orden_flechas}",
      "-" * 72,
      "Detalle de ejecucion:"
    ]

    if modo == "Secuencia Continua":
      total_pulsos = len(words) * reps
      for i in range(total_pulsos):
        w = words[i % len(words)]
        ciclo = (i // len(words)) + 1
        pos_en_ciclo = (i % len(words)) + 1
        banner.append(f"  Pulso {i+1:03d} | Ciclo {ciclo:02d}/{reps:02d} | Pos {pos_en_ciclo}: /{w}/")
      banner.append("-" * 72)
      banner.append(f"Total de pulsos continuos: {total_pulsos}")
    else:
      for idx, w in enumerate(words, 1):
        banner.append(f"  Toma {idx:02d}/{len(words):02d} | Vocal: /{w}/ ({reps} repeticiones)")
      banner.append("-" * 72)
      banner.append(f"Total de tomas ejecutadas: {len(words)}")

    banner.append("=" * 72)
    try:
      banner.append("")
      banner.append(self._formatear_informe_protocolo_completo_txt(modo_ejecucion=modo))
    except Exception:
      pass
    texto_final = "\n".join(banner)

    # 1. Imprimir a consola estandar / terminal (visible en consola externa y en boton CONSOLA/LOGS)
    print("\n" + texto_final + "\n")

    # 2. Guardar archivo en disco
    try:
      if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(root_dir) == "_internal":
          root_dir = os.path.dirname(root_dir)
      else:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

      directorio_base = os.path.join(root_dir, "base_de_datos_electrodos", fecha_str)
      os.makedirs(directorio_base, exist_ok=True)
      modo_nom = modo.lower().replace(" ", "_")
      nombre_archivo = f"registro_orden_{modo_nom}_{prueba}_{sujeto}.txt"
      ruta_archivo = os.path.join(directorio_base, nombre_archivo)
      with open(ruta_archivo, 'w', encoding='utf-8') as f:
        f.write(texto_final + "\n")
      print(f"[Registro] Archivo de orden guardado en: {ruta_archivo}")
    except Exception as e_reg:
      print(f"[Aviso] No se pudo escribir registro de orden en disco: {e_reg}")

    # 3. Mostrar en la UI (en el espacio grande de palabras)
    if hasattr(self, 'lbl_recording_space'):
      self.lbl_recording_space.setText(
        f"<div align='center'>"
        f"<span style='font-size: 24px; color: #00FFCC; font-weight: bold;'>¡SESION COMPLETADA!</span><br>"
        f"<span style='font-size: 16px; color: #FFFF00;'>Orden registrado:</span><br>"
        f"<span style='font-size: 32px; font-weight: 900; color: #FFFFFF;'>{orden_flechas}</span>"
        f"</div>"
      )

  def estado_finalizar_secuencia_continua(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.autoforge_overlay.setText("<div align='center'>¡SECUENCIA COMPLETADA!</div>")
    QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
    
    # Imprimir y registrar orden de secuencia
    self._imprimir_registro_orden(modo="Secuencia Continua")
    
    if getattr(self, 'is_acquiring', False):
      self.on_start_acq_click() # Esto detiene el DAQ y llama a reset_autoforge_buttons()
    else:
      self.is_autoforge_running = False
      if hasattr(self, 'btn_pause'):
        self.btn_pause.setChecked(False)
        self.btn_pause.setEnabled(False)
      if hasattr(self, 'btn_pause'):
        self.btn_pause.setChecked(False)
        self.btn_pause.setEnabled(False)
      self.is_autoforge_continuo = False
      self.reset_autoforge_buttons()
      
    self.config_stack.setCurrentIndex(0) # Restaurar configuración
    if hasattr(self, 'session_timer'):
      self.session_timer.stop()
    QtCore.QTimer.singleShot(1200, lambda: self._preguntar_lanzar_pca(modo="Secuencia Continua"))

  def _iniciar_timer_global(self):
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    noise_dur = self.spin_noise_duration.value()
    tiempo_por_palabra = 3.0 + noise_dur + (3.0 * (60.0/bpm)) + (self.autoforge_target_reps * (60.0/bpm)) + 10.0
    cvm_time = 32.5 if getattr(self, 'autoforge_do_cvm', True) else 0.0
    tiempo_total = cvm_time + (len(self.autoforge_words) * tiempo_por_palabra) + noise_dur + 2.0
    self.tiempo_restante_global = int(tiempo_total)
    
    # Guardar el estado base de la UI
    self.autoforge_estado_actual_str = "Iniciando..."
    
    if not hasattr(self, 'session_timer') or self.session_timer is None:
      self.session_timer = QtCore.QTimer(self)
      self.session_timer.timeout.connect(self._tick_session_timer)
    else:
      self.session_timer.stop()
    self.session_timer.start(1000)

  def _tick_session_timer(self):
    """
    Ejecuta la funcionalidad de _tick_session_timer.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if getattr(self, 'tiempo_restante_global', 0) > 0:
      self.tiempo_restante_global -= 1
      mins = int(self.tiempo_restante_global // 60)
      secs = int(self.tiempo_restante_global % 60)
      self.tiempo_restante_str = f"{mins:02d}:{secs:02d}"
      
      # Actualizamos el label si no está grabando ruido/palabra, o si está en un descanso
      if not getattr(self, 'is_recording', False) and getattr(self, 'is_autoforge_running', False):
        if hasattr(self, 'autoforge_estado_actual_str'):
          cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {self.tiempo_restante_str}&nbsp;</span>"
          self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    else:
      self.tiempo_restante_str = "00:00"
      if hasattr(self, 'session_timer'): self.session_timer.stop()

  def calcular_tiempo_restante_autograbado(self):
    """
    Ejecuta la funcionalidad de calcular_tiempo_restante_autograbado.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    return getattr(self, 'tiempo_restante_str', "00:00")

  def estado_iniciar_palabra(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_iniciar_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if self.autoforge_word_idx >= len(self.autoforge_words):
      if not getattr(self, 'final_noise_completed', False):
        self.estado_grabar_ruido_final(target_post_noise=self.estado_iniciar_palabra)
        return

      self.autoforge_overlay.setText("<div align='center'>¡AUTOGRABADO COMPLETADO!</div>")
      self.autoforge_overlay.show()
      QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
      self.current_recording = []
      
      # Imprimir y registrar orden de secuencia
      self._imprimir_registro_orden(modo="Autograbado Discreto")
      
      if getattr(self, 'is_acquiring', False):
        self.on_start_acq_click() # Detiene el DAQ correctamente y llama a reset_autoforge_buttons()
      else:
        self.is_autoforge_running = False
        if hasattr(self, 'btn_pause'):
          self.btn_pause.setChecked(False)
          self.btn_pause.setEnabled(False)
        self.reset_autoforge_buttons()
        
      self.config_stack.setCurrentIndex(0) # Restaurar configuración
      if hasattr(self, 'session_timer'): self.session_timer.stop()
      QtCore.QTimer.singleShot(1200, lambda: self._preguntar_lanzar_pca(modo="Autograbado Discreto"))
      return
      
    palabra = self.autoforge_words[self.autoforge_word_idx]
    self.is_recording = False
    # Ocultar hardware y trigger, mostrar vocal gigante en config_stack, dejando filtros y metrónomo visibles
    self.config_stack.setCurrentIndex(1)
    
    palabra_num = self.autoforge_word_idx + 1
    total_palabras = len(self.autoforge_words)
    self.autoforge_estado_actual_str = f"Set: {palabra_num}/{total_palabras}"
    
    tr_str = getattr(self, 'tiempo_restante_str', '00:00')
    cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
    self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    
    self.autoforge_overlay.setText("<div align='center'>HAZ SILENCIO<br>PREPARANDO ENTORNO...</div>")
    self.autoforge_overlay.show()
    
    vocal_colores = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
    v_col = vocal_colores.get(palabra.upper(), '#00FFCC')
    texto_ventana = (
      f"<div align='center'>"
      f"<span style='font-size: 24px; color: #888888; font-weight: bold;'>SIGUIENTE VOCAL:</span><br>"
      f"<span style='font-size: 75px; font-weight: 900; color: {v_col};'>/{palabra.upper()}/</span>"
      f"</div>"
    )
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None
    
    # 3 Segundos de silencio ANTES de siquiera empezar a capturar el ruido
    self._schedule_phase(3000, self.estado_grabar_ruido)

  def estado_grabar_ruido(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_grabar_ruido.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.autoforge_overlay.setText("") 
    self.autoforge_overlay.hide()
    self.label_rec_time.setVisible(True) 
    palabra_next = self.autoforge_words[self.autoforge_word_idx].upper() if (getattr(self, 'autoforge_words', None) and self.autoforge_word_idx < len(self.autoforge_words)) else "A"
    self.lbl_recording_space.setStyleSheet(
      "background-color: #141405; color: #FFFF00; font-family: 'Courier New', monospace; "
      "border: 2px solid #FFFF00; border-radius: 6px; padding: 4px;"
    )
    self.lbl_recording_space.setText(
      f"<div align='center' style='line-height:1.15;'>"
      f"<span style='font-size:28px; font-weight:900; color:#FFFF00;'>HAZ SILENCIO &mdash; GRABANDO RUIDO BASAL</span><br>"
      f"<span style='font-size:18px; color:#00FFCC;'>Relaja el rostro. Siguiente vocal a pronunciar: <b>/{palabra_next}/</b></span>"
      f"</div>"
    )
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say("haz_silencio", "Haz silencio, grabando ruido basal", fase="ruido_basal_inicial")
    
    self.current_recording = []
    self._offset_muestras_ancla = 0
    
    # Reiniciar arreglos de ruido
    self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.noise_levels = [0.0] * self.NUM_CANALES
    self.noise_calculated = False
    self.noise_initialized = False
    self.initial_noise_mean = [0.0] * self.NUM_CANALES
    self.initial_noise_std = [0.0] * self.NUM_CANALES
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    
    # UI Global
    self.global_snr_acumulado = [0.0] * self.NUM_CANALES
    self.global_snr_count = [0] * self.NUM_CANALES
    
    import time
    self.recording_start_time = time.perf_counter() 
    self.is_recording = True

    # Hook fotometria: captura segura de ruido basal
    self._safe_photometry_snapshot(fase="ruido_basal", palabra="ruido_inicial", pulso=0)
    
    # --- Resetear escala Y a rango base calibrado (500 µV) ---
    if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
      self.plot.setYRange(0.0, 500.0, padding=0)
    else:
      self.plot.setYRange(-500.0, 500.0, padding=0)
    self._held_plot_ymax = 500.0
    
    # Programar la aparición de la ventana "Preparate" y el metrónomo CUANDO TERMINE EL RUIDO
    noise_dur = self.spin_noise_duration.value()
    self._schedule_phase(int(noise_dur * 1000), self.estado_mostrar_preparate)

  def estado_mostrar_preparate(self):
    if not getattr(self, 'is_autoforge_running', False): return
    # 1. El ruido base acaba de terminar. PAUSAMOS LA GRABACIÓN.
    """
    Ejecuta la funcionalidad de estado_mostrar_preparate.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.is_recording = False
    self.autoforge_ruido_guardado = self.current_recording.copy()
    self.autoforge_estado_actual_str = "ESCUCHA EL METRÓNOMO..."
    
    tr_str = getattr(self, 'tiempo_restante_str', '00:00')
    cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
    self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF8800;") # Naranja
    
    # --- NUEVO: Forzamos el cálculo matemático del ruido aquí para evitar desincronizaciones de UI ---
    if not getattr(self, 'noise_calculated', False):
      import numpy as np
      for i in range(self.NUM_CANALES):
        if self.noise_data_accumulated[i]:
          all_noise = np.concatenate(self.noise_data_accumulated[i])
          window_size_ms = self.spin_rms_window.value()
          window_size = int((window_size_ms / 1000.0) * self.SAMPLE_RATE)
          if window_size < 1: window_size = 1
          all_noise_env = calculate_rms_envelope(all_noise, window_size)
          
          self.noise_levels[i] = np.mean(all_noise_env)
          self.initial_noise_mean[i] = self.noise_levels[i]
          self.initial_noise_std[i] = np.std(all_noise_env)
          
          is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
          show_noise = self.chk_show_noise.isChecked()
          self.noise_lines[i].setPos(self.noise_levels[i])
          if is_rms:
            self.noise_lines_neg[i].setVisible(False)
            self.noise_regions[i].setRegion([0.0, self.noise_levels[i]])
          else:
            self.noise_lines_neg[i].setPos(-self.noise_levels[i])
            self.noise_lines_neg[i].setVisible(show_noise)
            self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          self.noise_lines[i].setVisible(show_noise)
          self.noise_regions[i].setVisible(show_noise)
          
          if hasattr(self, 'noise_status_labels'):
            self.noise_status_labels[i].setText(f"Ruido Base: x̄={self.initial_noise_mean[i]:.1f}µV, s={self.initial_noise_std[i]:.1f}µV")
            self.noise_status_labels[i].setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold; background-color: #111111; border: 1px solid #00FFFF;")
      
      # --- NUEVO: Auto-ajuste del umbral de SNR ---
      if self.NUM_CANALES > 0:
        ruido_maximo_std = max(getattr(self, 'initial_noise_std', [0]))
        if ruido_maximo_std > 0:
          self.spin_peak_th.setValue(ruido_maximo_std * 5.0)
          
      self.noise_calculated = True
      self._procesar_ruido_inicial_50hz()
    
    # --- CVM HOOK ---
    if getattr(self, 'autoforge_do_cvm', True) and not getattr(self, 'cvm_completed', False):
      self.estado_iniciar_calibracion_cvm(target_post_cvm=self.estado_mostrar_preparate)
      return
    
    # 2. Iniciamos el Count-In acústico (3 compases) antes de saltar a la palabra.
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    beat_interval_s = 60.0 / bpm
    
    palabra = self.autoforge_words[self.autoforge_word_idx]
    vocal_colores = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
    v_col = vocal_colores.get(palabra.upper(), '#00FFCC')
    texto_ventana = (
      f"<div align='center'>"
      f"<span style='font-size: 24px; color: #FFCC00; font-weight: bold;'>PREPÁRATE:</span><br>"
      f"<span style='font-size: 75px; font-weight: 900; color: {v_col};'>/{palabra.upper()}/</span>"
      f"</div>"
    )
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None
    
    # Lanzar el metrónomo AHORA MISMO con un Count-in de tipo carreras (3 graves, 1 ya estaba agregado)
    # (El metrónomo ya está fijo en el recuadro superior)
    
    self.autoforge_overlay.hide()
    
    # --- Sincronización exacta del texto de PREPARATE con el count-in acústico ---
    y_center = self._get_plot_y_center()
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, y_center)
    self.countdown_text.setText("PREPÁRATE...\n3")
    self.countdown_text.show()
    
    # Asegurar que el gráfico esté en modo auto-scroll limpio al comenzar preparación
    self.chk_autoscroll.setChecked(True)
    self.on_autoscroll_toggle()
    
    # Anunciar UNA SOLA VEZ la vocal durante la preparación del modo discreto (sin repetir A-A-A en cada pulso)
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say(f"di_{palabra.lower()}", f"Prepárate, di {palabra.upper()}", fase="preparacion_palabra_discreta")
    # Al sonar el pulso 1 real tras los 3 pulsos de preparación, on_metro_beat invoca estado_cambiar_ventana_palabra
    self._on_count_in_finished_cb = self.estado_cambiar_ventana_palabra
    self.start_native_metronome(count_in=4, force_start=True)

  def estado_cambiar_ventana_palabra(self, anchor_perf=None):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_cambiar_ventana_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.autoforge_overlay.hide() # Quitamos el "PREPARATE" del medio
    
    palabra = self.autoforge_words[self.autoforge_word_idx].upper()
    
    # Mostrar la vocal en grande en lbl_recording_space y en el titulo del metronomo
    vocal_colores = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
    v_col = vocal_colores.get(palabra, '#00FFCC')
    self.lbl_recording_space.setText(f"<div align='center' style='font-size: 85px; font-weight: 900; color: {v_col};'>/{palabra}/</div>")
    if hasattr(self, 'countdown_text'):
        self.countdown_text.hide()
    self.metro_lbl_title.setText(f"VOCAL /{palabra}/ (P1)")
    self.metro_lbl_title.setStyleSheet("font-size: 15px; color: #00FFCC; font-weight: bold;")
    
    # --- REACTIVAMOS LA GRABACIÓN: AQUÍ EMPIEZA LA PALABRA REAL ---
    import time
    self.recording_start_time = float(anchor_perf) if anchor_perf is not None else time.perf_counter()
    self.is_recording = True
    
    # --- NUEVO: Limpiamos los buffers de guardado y estadísticas para cada palabra ---
    self.current_recording = getattr(self, 'autoforge_ruido_guardado', []).copy()
    # El .wav de la palabra arranca con el ruido basal ya grabado
    self._offset_muestras_ancla = self._contar_muestras_recording()
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    
    # Reset de evaluador de ruido inter-pulso para la nueva vocal
    self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
    self.rt_last_noise_eval = [0.0] * self.NUM_CANALES
    self._held_plot_ymax = 20.0
    self._held_plot_last_peak_time = 0.0

    # Asegurar auto-scroll continuo y centrado
    self.chk_autoscroll.setChecked(True)
    self.on_autoscroll_toggle()
    
    self.vocal_cue_log = []
    self.sync_evaluation_log = []
    self.label_rec_time.setText(f"GRABANDO VOCAL: /{palabra}/...")
    if anchor_perf is None:
      words = getattr(self, 'autoforge_words', [palabra])
      w_idx = int(getattr(self, 'autoforge_word_idx', 0))
      p_sig = palabra if self.autoforge_target_reps > 1 else (words[w_idx + 1].upper() if (w_idx + 1) < len(words) else "FIN")
      self._disparar_destello_vocal(
          palabra_actual=palabra,
          pulso_actual=1,
          total_pulsos=self.autoforge_target_reps,
          palabra_siguiente=p_sig,
          ciclo_actual=w_idx + 1,
          total_ciclos=len(words)
      )
      self._safe_photometry_snapshot(fase="onset", palabra=str(palabra), pulso=1)
    self.word_window_process = None
    
    # El cierre de la palabra lo dispara exactamente on_metro_beat al completar self.autoforge_target_reps pulsos

  def estado_guardar_palabra(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_guardar_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.is_recording = False
    palabra = self.autoforge_words[self.autoforge_word_idx]
    if hasattr(self, 'countdown_text'): self.countdown_text.hide()
    
    self.stop_native_metronome()
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    # 5 segundos de descanso mientras se guardan los datos pesados (lag)
    self.autoforge_estado_actual_str = "DESCANSO 5s (GUARDANDO)"
    self.autoforge_overlay.setText("<div align='center'>DESCANSO 5s<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    self.lbl_recording_space.setText("<div align='center' style='color:#888888; font-size:55px; font-weight:bold;'>DESCANSO</div>")
    if hasattr(self, 'voice_cue_player'):
      self.voice_cue_player.say("descanso", "Descanso", fase="descanso_entre_tomas")
    
    local_recording = self.current_recording.copy()
    self.current_recording = []
    local_pulsos_metro = list(getattr(self, '_metro_pulsos', []) or [])
    local_beat_timestamps = list(getattr(self, 'metro_beat_timestamps_s', []) or [])
    local_beat_samples = list(getattr(self, 'metro_beat_sample_indices', []) or [])
    local_beat_intervals = list(getattr(self, 'metro_applied_intervals_s', []) or [])
    local_stats_time = [list(x) for x in getattr(self, 'stats_time', [])]
    local_stats_snr = [list(x) for x in getattr(self, 'stats_snr', [])]
    local_stats_noise_mean = [list(x) for x in getattr(self, 'stats_noise_mean', [])]
    local_stats_noise_std = [list(x) for x in getattr(self, 'stats_noise_std', [])]
    local_vocal_cues = list(getattr(self, 'vocal_cue_log', []) or [])
    local_sync_eval = list(getattr(self, 'sync_evaluation_log', []) or [])
    
    import threading
    def guardar_async():
      """
      Ejecuta la funcionalidad de guardar_async.

      Returns:
        Any: Resultado de la ejecución de la función.
      """
      import os, json, re
      from pathlib import Path
      from datetime import datetime
      fecha_str = datetime.now().strftime("%Y-%m-%d")
      
      # Calcular la prueba actual según la repetición de la palabra en la lista (ej. T1, T2, etc.)
      ocurrencia = self.autoforge_words[:self.autoforge_word_idx + 1].count(palabra)
      
      m = re.match(r"^([a-zA-Z]+)(\d+)$", self.autoforge_prueba)
      if m:
        prefix, num_str = m.groups()
        base_num = int(num_str)
        prueba_actual = f"{prefix}{base_num + ocurrencia - 1}"
      elif ocurrencia > 1:
        prueba_actual = f"{self.autoforge_prueba}_{ocurrencia}"
      else:
        prueba_actual = self.autoforge_prueba

      if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(root_dir) == "_internal":
          root_dir = os.path.dirname(root_dir)
      else:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
      # Verificar si ya existe en disco y auto-incrementar para NUNCA sobreescribir
      base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / f"{palabra}_{prueba_actual}_{self.autoforge_sujeto}"
      reintento = 0
      while base_dir.exists() and any(base_dir.iterdir()):
        reintento += 1
        if m:
          prueba_actual = f"{prefix}{base_num + ocurrencia - 1 + reintento}"
        else:
          prueba_actual = f"{self.autoforge_prueba}_{ocurrencia + reintento}"
        base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / f"{palabra}_{prueba_actual}_{self.autoforge_sujeto}"

      os.makedirs(base_dir, exist_ok=True)
      if hasattr(self, 'autoforge_session_dirs'):
        self.autoforge_session_dirs.append(str(base_dir))
      
      # 1. Preparar metadata para UI Analysis
      now = datetime.now()
      muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
      muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}
      calib_cvm_dict = {
          "calibrado": getattr(self, 'cvm_completed', False),
          "metodo": "envolvente_rms_multiparametrico",
          "tomas_por_vocal": 3,
          "canales": {
              f"canal_{i}": {
                  "pico_envolvente_uV": self.cvm_env_max_per_channel[i] if hasattr(self, 'cvm_env_max_per_channel') and i < len(self.cvm_env_max_per_channel) else 0.0,
                  "percentil_95_uV": self.cvm_env_p95_per_channel[i] if hasattr(self, 'cvm_env_p95_per_channel') and i < len(self.cvm_env_p95_per_channel) else 0.0,
                  "percentil_90_uV": self.cvm_env_p90_per_channel[i] if hasattr(self, 'cvm_env_p90_per_channel') and i < len(self.cvm_env_p90_per_channel) else 0.0,
                  "rms_envolvente_uV": self.cvm_env_rms_per_channel[i] if hasattr(self, 'cvm_env_rms_per_channel') and i < len(self.cvm_env_rms_per_channel) else 0.0,
                  "energia_uV2_s": self.cvm_energia_per_channel[i] if hasattr(self, 'cvm_energia_per_channel') and i < len(self.cvm_energia_per_channel) else 0.0,
                  "auc_uV_s": self.cvm_auc_per_channel[i] if hasattr(self, 'cvm_auc_per_channel') and i < len(self.cvm_auc_per_channel) else 0.0
              } for i in range(self.NUM_CANALES)
          },
          "supremo_global_uV": getattr(self, 'supremo_global_uV', 0.0),
          "supremo_global_p95_uV": getattr(self, 'supremo_global_p95_uV', 0.0),
          "supremo_global_p90_uV": getattr(self, 'supremo_global_p90_uV', 0.0),
          "supremo_global_rms_uV": getattr(self, 'supremo_global_rms_uV', 0.0),
          "supremo_global_energia_uV2_s": getattr(self, 'supremo_global_energia', 0.0),
          "supremo_global_auc_uV_s": getattr(self, 'supremo_global_auc', 0.0)
      }

      _t0_perf = getattr(self, 'recording_start_time', None) or time.perf_counter()
      _t0_epoch = time.time() - (time.perf_counter() - _t0_perf)
      metadata = {
        "measurement_date": now.isoformat(),
        "t0_epoch": _t0_epoch,
        "t0_perf_counter": _t0_perf,
        "daq_sample_rate": self.SAMPLE_RATE,
        "metronomo_pulsos": len(local_pulsos_metro),
        "metronomo_pulsos_csv": "pulsos_metronomo.csv" if local_pulsos_metro else None,
        "beat_timestamps_s": local_beat_timestamps,
        "beat_sample_indices": local_beat_samples,
        "beat_intervals_s": local_beat_intervals,
        "jitter_config": {
            "enabled": bool(hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked()),
            "pct": float(self.spin_metro_jitter_pct.value()) if hasattr(self, 'spin_metro_jitter_pct') else 0.0,
            "mode": self.cmb_metro_jitter_mode.currentText() if hasattr(self, 'cmb_metro_jitter_mode') else "exponential_hazard"
        },
        "eventos_vocales_metronomo": local_vocal_cues,
        "sincronizacion_sujeto_vs_metronomo": local_sync_eval,
        "timestamp": int(now.timestamp()),
        "sample_rate": self.SAMPLE_RATE,
        "channels": self.CANALES_DAQ,
        "muscles": muscles_list,
        "muscles_map": muscles_map,
        "bpm": self.spin_bpm.value(),
        "noise_seconds": self.spin_noise_duration.value(),
        "pulse_count": self.autoforge_target_reps,
        "is_formal": False,
        "sujeto": self.autoforge_sujeto,
        "letra": palabra,
        "prueba": prueba_actual,
        "comentario": "Grabado mediante AutoForge",
        "orden_sesion_vocales": list(self.autoforge_words),
        "posicion_en_secuencia": self.autoforge_word_idx + 1,
        "total_palabras_sesion": len(self.autoforge_words),
        "ruido_basal": {
            "inicial": {
                "mean_uV": getattr(self, 'initial_noise_mean', [0.0]*self.NUM_CANALES),
                "std_uV": getattr(self, 'initial_noise_std', [0.0]*self.NUM_CANALES),
                "line_noise_50hz_uV": getattr(self, 'line_noise_50hz_uV', [0.0]*self.NUM_CANALES)
            },
            "final": {
                "mean_uV": getattr(self, 'final_noise_mean', [0.0]*self.NUM_CANALES),
                "std_uV": getattr(self, 'final_noise_std', [0.0]*self.NUM_CANALES),
                "line_noise_50hz_uV": getattr(self, 'final_noise_50hz', [0.0]*self.NUM_CANALES)
            }
        },
        "calibracion_cvm": calib_cvm_dict
      }
      photo_meta = self._safe_photometry_export(str(base_dir))
      if photo_meta is not None:
        metadata["fotometria_espacial"] = photo_meta
      if hasattr(self, 'last_separability_metrics') and self.last_separability_metrics is not None:
        metadata["diagnostico_separabilidad_vocales"] = self.last_separability_metrics
      
      # 2. Crear carpetas de canales y guardar metadata.json en cada una
      for i in range(self.NUM_CANALES):
        channel_output_dir = os.path.join(base_dir, f"canal_{i}")
        os.makedirs(channel_output_dir, exist_ok=True)
        metadata_path = os.path.join(channel_output_dir, "metadata.json")
        ch_metadata = metadata.copy()
        ch_metadata["canal"] = f"canal_{i}"
        ch_metadata["musculo"] = self.nombres_musculos[i % len(self.nombres_musculos)]
        ch_metadata["physical_channel"] = self.CANALES_DAQ[i] if i < len(self.CANALES_DAQ) else f"ai{i}"
        try:
          with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(ch_metadata, f, indent=4)
        except: pass

      try:
        base_meta_path = os.path.join(base_dir, "metadata.json")
        with open(base_meta_path, 'w', encoding='utf-8') as f:
          json.dump(metadata, f, indent=4)
      except: pass

      # 3. Guardar Datos Raw CSV, Pulsos del Metrónomo, Segmentos Dataset y WAV
      try:
        escribir_pulsos_metronomo_csv(local_pulsos_metro, str(base_dir))
      except Exception as e:
        print(f"[Aviso] No se pudieron guardar los pulsos del metronomo: {e}")
      try:
        emg_chs = [c for c in range(self.NUM_CANALES) if not self._is_mic_channel(c)]
        gains_ch = [
          self.factores_calibracion_hw[self.active_channel_indices[c]]
          if (hasattr(self, 'active_channel_indices') and c < len(self.active_channel_indices) and self.active_channel_indices[c] < len(self.factores_calibracion_hw))
          else 495.0
          for c in range(self.NUM_CANALES)
        ]
        ds_summary = exportar_segmentos_dataset_y_fatiga(
          datos_recording=local_recording,
          sample_rate=self.SAMPLE_RATE,
          output_dir=str(base_dir),
          num_canales=self.NUM_CANALES,
          vocal_cue_log=local_vocal_cues,
          sync_evaluation_log=local_sync_eval,
          invalid_pulses_set=set(getattr(self, 'invalid_pulses_set', set()) or set()),
          factores_calibracion=gains_ch,
          emg_channel_indices=emg_chs,
          initial_noise_mean=getattr(self, 'initial_noise_mean', []),
          initial_noise_std=getattr(self, 'initial_noise_std', []),
        )
        self.last_dataset_summary = ds_summary
        metadata["dataset_y_fatiga"] = {
          k: v for k, v in ds_summary.items() if k != "segmentos"
        }
        metadata["semaforo_contacto_electrodos"] = getattr(self, 'diagnostico_contacto_electrodos', {})
      except Exception as e:
        print(f"[Aviso Dataset] No se pudo exportar segmentos_dataset.csv en discreto: {e}")
      try: guardar_grabacion_csv(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion", canales_daq=self.CANALES_DAQ)
      except: pass
      
      try: guardar_grabacion_wav(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      # 4. Generar Gráficos
      try:
        generar_grafico_grabacion(local_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
        ruta_photo = os.path.join(str(base_dir), "photo.png")
        if os.path.exists(ruta_photo):
          self.mostrar_preview_signal.emit(ruta_photo)
      except: pass
      try: generar_grafico_estadisticas(local_stats_time, local_stats_snr, local_stats_noise_mean, local_stats_noise_std, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      # 5. Guardar archivo con el orden completo de la sesión y todo el protocolo
      try:
        orden_txt_path = os.path.join(str(base_dir), "registro_orden_sesion.txt")
        with open(orden_txt_path, 'w', encoding='utf-8') as f_ord:
          f_ord.write("AUTOGRABADO DISCRETO - REGISTRO DE ORDEN DE VOCALES Y PROTOCOLO COMPLETO\n")
          f_ord.write(f"Fecha y Hora: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
          f_ord.write(f"Sujeto: {self.autoforge_sujeto} | Prueba: {prueba_actual}\n")
          f_ord.write(f"Vocal Actual: /{palabra}/ (Toma {self.autoforge_word_idx + 1} de {len(self.autoforge_words)})\n")
          f_ord.write(f"Orden Completo de Sesión: {' -> '.join(self.autoforge_words)}\n")
          f_ord.write(f"Repeticiones por toma: {self.autoforge_target_reps}\n\n")
          f_ord.write(self._formatear_informe_protocolo_completo_txt(metadata_dict=metadata, modo_ejecucion="Autograbado Discreto"))
      except Exception: pass

      # 6. Persistir notas de experimento en metadata.json y archivo de texto
      try:
        self.persistir_notas_experimento(str(base_dir), metadata_base=metadata)
      except Exception: pass
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    # Tras 5 segundos, pasar a verificación muscular para que el usuario pueda ver su señal fluida
    self._schedule_phase(5000, self.estado_verificacion_muscular)

  @QtCore.Slot()
  def estado_verificacion_muscular(self):
    if not getattr(self, 'is_autoforge_running', False): return
    self.autoforge_estado_actual_str = "VERIFICACIÓN MUSCULAR 5s"
    # Cerrar inmediatamente cualquier diálogo de preview abierto para no tapar el osciloscopio en vivo
    if hasattr(self, '_preview_dialog') and self._preview_dialog:
      try:
        self._preview_dialog.close()
      except Exception:
        pass
      self._preview_dialog = None
    # Ocultar el overlay oscuro para que la señal se vea limpia y sin sombra
    self.autoforge_overlay.hide()
    self.lbl_recording_space.setText(
      "<div align='center' style='color:#00FF00; font-size:32px; font-weight:bold;'>"
      "VERIFICACIÓN MUSCULAR<br>"
      "<span style='font-size:20px; color:#88FF88;'>Activa tus músculos para probar la señal</span>"
      "</div>"
    )
    
    # Ir a la siguiente palabra tras los 5 segundos de verificación (10s totales)
    self._schedule_phase(5000, self.estado_siguiente)

  @QtCore.Slot()
  def estado_siguiente(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_siguiente.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.autoforge_word_idx += 1
    self.estado_iniciar_palabra()

    pass

  def resizeEvent(self, event):
    """Maneja la redimensión de la ventana para ajustar las ventanas flotantes."""
    super().resizeEvent(event)

  def changeEvent(self, event):
    """Maneja eventos de cambio de estado (como pantalla completa)."""
    super().changeEvent(event)
    if event.type() == QtCore.QEvent.WindowStateChange:
      pass


  @QtCore.Slot(str, bool)
  def _on_log_captured(self, text: str, is_err: bool):
    if hasattr(self, 'terminal_dialog'):
      self.terminal_dialog.append_log(text, is_err)
    upper = text.upper()
    if is_err or "ERROR" in upper or "TRACEBACK" in upper:
      self._error_count = getattr(self, '_error_count', 0) + 1
      if hasattr(self, 'btn_terminal'):
        self.btn_terminal.setText(f"CONSOLA / ERRORES ({self._error_count})")
        self.btn_terminal.setStyleSheet("QPushButton { background-color: #0B0F19; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 12px; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #39FF14; color: #050508; }")

  def mostrar_consola_diagnostico(self):
    self._error_count = 0
    if hasattr(self, 'btn_terminal'):
      self.btn_terminal.setText("CONSOLA / LOGS (0)")
      self.btn_terminal.setStyleSheet("QPushButton { background-color: #0B0F19; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 12px; font-weight: bold; border-radius: 4px; } QPushButton:hover { background-color: #39FF14; color: #050508; }")
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
    if getattr(self, 'chk_zmq_bridge', None) and self.chk_zmq_bridge.isChecked():
      ep_str = self.edit_zmq_host.text().strip() if hasattr(self, 'edit_zmq_host') else "auto:5555"
      try:
        from daq_bridge_client import DAQBridgeClient, parse_host_port
        host, port = parse_host_port(ep_str, default_port=5555)
        cli = DAQBridgeClient(host=host, port=port)
        ping = cli.ping_server(timeout_ms=220)
        cli.close()
        if ping and ping.get("status") == "ok":
          dev_rem = ping.get("device", "Dev1")
          mode_rem = ping.get("terminal_config", "DIFF")
          self.lbl_daq_status.setText(f"PUENTE VM: {dev_rem} CONECTADO ({host}:{port} | {mode_rem})")
          self.lbl_daq_status.setStyleSheet(
            "background-color: #00220a; color: #00FF00; border: 1.5px solid #00FF00; padding: 4px 8px; font-weight: bold; border-radius: 3px;"
          )
        else:
          self.lbl_daq_status.setText(f"PUENTE VM ZMQ: LISTO ({host}:{port})")
          self.lbl_daq_status.setStyleSheet(
            "background-color: #241800; color: #FFAA00; border: 1.5px solid #FFAA00; padding: 4px 8px; font-weight: bold; border-radius: 3px;"
          )
      except Exception:
        self.lbl_daq_status.setText(f"PUENTE VM ZMQ: ACTIVO ({ep_str})")
        self.lbl_daq_status.setStyleSheet(
          "background-color: #241800; color: #FFAA00; border: 1.5px solid #FFAA00; padding: 4px 8px; font-weight: bold; border-radius: 3px;"
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
        status_txt = f"DAQ: {dev_target} CONECTADO [{prod_type}]"
        self.lbl_daq_status.setText(status_txt)
        self.lbl_daq_status.setStyleSheet(
          "background-color: #00220a; color: #00FF00; border: 1.5px solid #00FF00; padding: 4px 8px; font-weight: bold; border-radius: 3px;"
        )
        if getattr(self, '_last_daq_status', None) != status_txt:
          print(f"[DAQ CHECK] Dispositivo fisico detectado: {dev_target} ({prod_type})")
          self._last_daq_status = status_txt
      else:
        status_txt = f"DAQ: {dev_target} DESCONECTADO"
        self.lbl_daq_status.setText(status_txt)
        self.lbl_daq_status.setStyleSheet(
          "background-color: #2a0008; color: #FF3333; border: 1.5px solid #FF0033; padding: 4px 8px; font-weight: bold; border-radius: 3px;"
        )
        if getattr(self, '_last_daq_status', None) != status_txt:
          avail_str = ", ".join(dev_names) if dev_names else "Ninguno"
          print(f"[DAQ CHECK] {dev_target} no encontrado. Dispositivos conectados: {avail_str}")
          self._last_daq_status = status_txt
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

    # 1. Limpiar sub_plots anteriores del layout de forma segura
    for sp in getattr(self, 'sub_plots', []):
      try:
        self.plot_widget.removeItem(sp)
        sp.close()
      except Exception:
        pass
    self.sub_plots = []
    self.curvas_multitrack = []

    # 2. Modo Superpuesto: reactivar todas las curvas originales en self.plot
    if not multitrack:
      self.plot.showAxis('bottom')
      self.plot.setLabel('bottom', 'Tiempo (s)')
      self.plot.setLabel('left', 'Voltaje (µV)')
      for i, c in enumerate(self.curvas):
        if i < len(self.viz_checkboxes):
          c.setVisible(self.viz_checkboxes[i].isChecked())
        else:
          c.setVisible(True)

    # 3. Modo Pistas Separadas (Multitrack): Canal 0 en self.plot, canales 1..N en sub_plots dedicados
    else:
      self.plot.hideAxis('bottom')
      musc_0 = self.nombres_musculos[0 % len(self.nombres_musculos)] if self.nombres_musculos else "Canal 0"
      self.plot.setLabel('left', f"{musc_0} (µV)")
      if 0 < len(self.viz_checkboxes):
        self.curvas[0].setVisible(self.viz_checkboxes[0].isChecked())
      else:
        self.curvas[0].setVisible(True)

      # Ocultar canales 1..N en self.plot sin sacarlos de su ViewBox
      for i in range(1, len(self.curvas)):
        self.curvas[i].setVisible(False)

      # Crear un sub_plot nativo con su propia curva dedicada para cada canal 1..N
      for i in range(1, self.NUM_CANALES):
        sp = self.plot_widget.addPlot(row=i, col=0)
        sp.showGrid(x=True, y=True)
        sp.getAxis('left').setWidth(60)
        musc_name = self.nombres_musculos[i % len(self.nombres_musculos)] if self.nombres_musculos else f"Canal {i}"
        sp.setLabel('left', f"{musc_name} (µV)")
        sp.setXLink(self.plot)
        sp.setYLink(self.plot)
        sp.setMouseEnabled(x=False, y=True)
        sp.getViewBox().disableAutoRange(pg.ViewBox.XAxis)
        sp.getViewBox().disableAutoRange(pg.ViewBox.YAxis)
        if i < self.NUM_CANALES - 1:
          sp.hideAxis('bottom')
        else:
          sp.setLabel('bottom', 'Tiempo (s)')
        
        pen_color = self.colores_curvas[i % len(self.colores_curvas)] if hasattr(self, 'colores_curvas') else pg.intColor(i)
        sub_c = sp.plot(pen=pen_color, name=musc_name)
        if i < len(self.viz_checkboxes):
          sub_c.setVisible(self.viz_checkboxes[i].isChecked())
        self.curvas_multitrack.append(sub_c)
        self.sub_plots.append(sp)

  def closeEvent(self, event):
    if hasattr(self, '_stdout_tee'): sys.stdout = self._stdout_tee._orig
    if hasattr(self, '_stderr_tee'): sys.stderr = self._stderr_tee._orig
    # Se llama cuando el usuario cierra la ventana
    """
    Ejecuta la funcionalidad de closeEvent.

    Args:
      event (Any): Argumento posicional event.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    print("Ventana de ploteo cerrada. Deteniendo hilos...")
    
    # --- Detener limpiamente para evitar Access Violations (0xC0000005) al salir ---
    if getattr(self, 'is_autoforge_running', False):
      self.detener_autoforge()
    if getattr(self, 'is_acquiring', False):
      self.on_start_acq_click()
      
    self.stop_event.set() # Envía la señal a TODOS los hilos
    self.timer.stop() # Detiene el timer de la GUI

    # Detener limpiamente el worker de camara web / fotometria
    if getattr(self, 'photometry_mgr', None):
      try:
        self.photometry_mgr.stop_worker()
      except Exception as exc:
        print(f"[FOTOMETRIA] Aviso al detener worker: {exc}")
    
    # --- NUEVO: Guardar estado de la GUI al cerrar el programa ---
    try:
      config_data = {}
      if os.path.exists(METRONOME_CONFIG_PATH):
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          config_data = json.load(f)
      config_data['last_bpm'] = self.spin_bpm.value()
      if hasattr(self, 'spin_metro_subdivs'):
        config_data['subdivisions'] = self.spin_metro_subdivs.value()
      config_data['notch_enabled'] = self.chk_notch_enable.isChecked()
      with open(METRONOME_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    except Exception as e:
      print(f"Error guardando configuración al cerrar: {e}")
      
    event.accept() # Acepta el cierre

  def abrir_medidor_separabilidad_dialog(self):
    pass

  def abrir_test_separabilidad_vocales(self):
    pass

  def _on_pulse_detected(self, pulse_event):
    """
    Callback disparado ante la deteccion confirmada de un pulso fonatorio.
    Calcula también el desfase/sincronización (en ms) entre el pulso del metrónomo
    cuando sonó la vocal y el onset real del sujeto (con Jitter encendido o apagado).
    """
    self.latest_pulse_event = pulse_event
    method_tag = "MIC" if pulse_event.get('method') == 'MIC' else "GATE"
    sep = pulse_event.get('separability_pct', 0.0)
    props_pct = pulse_event.get('proportions_pct', [])
    p_str = " | ".join([f"Ch{i}:{props_pct[i]:.1f}%" for i in range(min(3, len(props_pct)))])
    vocal = pulse_event.get('vocal', '')
    vocal_tag = f" [/{vocal}/]" if vocal else ""
    atractor = pulse_event.get('atractor', '')
    supremo = pulse_event.get('supremo_uV', 0.0)

    sync_str = ""
    if getattr(self, 'vocal_cue_log', None):
      try:
        last_cue = self.vocal_cue_log[-1]
        fs = float(getattr(self, 'SAMPLE_RATE', 2000.0) or 2000.0)
        # Estimar instante perf del onset detectado
        t_onset_perf = float(pulse_event.get('t_detect_perf', time.perf_counter())) + float(pulse_event.get('t_onset_rel', 0.0))
        t_cue_perf = float(last_cue.get('t_perf', t_onset_perf))
        sync_latency_ms = round((t_onset_perf - t_cue_perf) * 1000.0, 1)
        pulse_event['sync_latency_ms'] = sync_latency_ms
        pulse_event['cue_pulso_idx'] = last_cue.get('pulso_vocal_idx')
        pulse_event['cue_intervalo_s'] = last_cue.get('intervalo_s')
        pulse_event['jitter_activo'] = last_cue.get('jitter_activo', False)
        if not hasattr(self, 'sync_evaluation_log') or self.sync_evaluation_log is None:
          self.sync_evaluation_log = []
        self.sync_evaluation_log.append({
          "pulse_num": pulse_event['pulse_num'],
          "cue_pulso_idx": last_cue.get('pulso_vocal_idx'),
          "palabra": last_cue.get('palabra', vocal),
          "sync_latency_ms": sync_latency_ms,
          "cue_t_wav_s": last_cue.get('t_wav_s'),
          "cue_sample_daq": last_cue.get('sample_daq'),
          "intervalo_s": last_cue.get('intervalo_s'),
          "jitter_activo": last_cue.get('jitter_activo', False),
          "jitter_mode": last_cue.get('jitter_mode', 'off'),
        })
        sync_str = f" | Sync: {sync_latency_ms:+.0f}ms"
      except Exception:
        pass

    print(f"[PULSO {method_tag} 0.5τ] #{pulse_event['pulse_num']}{vocal_tag} | {p_str} | Supremo: {supremo:.1f} µV | Sep: {sep:.1f}%{sync_str} | {atractor}")
    try:
      self._actualizar_medidor_pca_vivo(pulse_event)
      emg_raw = pulse_event.get('emg_raw_win')
      fs = float(getattr(self, 'SAMPLE_RATE', 2000.0) or 2000.0)
      if emg_raw is not None:
        self._actualizar_semaforo_fatiga_en_vivo(emg_raw, fs)
    except Exception as exc_pca:
      pass

# =============================================================================

  
  def _on_escala_cvm_toggled(self, checked):
    """Actualiza la etiqueta del eje Y entre microvoltios y % CVM."""
    if checked:
      self.plot.setLabel('left', 'Amplitud [% CVM]', color='#FFCC00')
      is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
      if is_rms:
        self.plot.setYRange(0.0, 120.0, padding=0.05)
      else:
        self.plot.setYRange(-120.0, 120.0, padding=0.05)
      print("[Escala] Visualización en % CVM activada (normalizada por contracción máxima).")
    else:
      self.plot.setLabel('left', 'Amplitud [µV]', color='#00FFFF')
      is_rms = hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked()
      if is_rms:
        self.plot.setYRange(0.0, 500.0, padding=0.05)
      else:
        self.plot.setYRange(-500.0, 500.0, padding=0.05)
      print("[Escala] Visualización en microvoltios absolutos [µV] restaurada.")

  def _actualizar_semaforo_fatiga_en_vivo(self, emg_raw_win, sample_rate):
    """
    Calcula la Frecuencia Mediana (MDF) en tiempo real con cada pulso fonatorio
    y actualiza el semáforo visual de fatiga mioeléctrica.
    """
    if not hasattr(self, 'lbl_fatiga_text') or self.lbl_fatiga_text is None:
      return
    if emg_raw_win is None or emg_raw_win.size < 32 or sample_rate <= 0:
      return

    mdfs = []
    for k in range(emg_raw_win.shape[0]):
      met = calcular_metricas_espectrales_fatiga(emg_raw_win[k], sample_rate)
      if met.get('mdf_hz', 0.0) > 10.0:
        mdfs.append(met['mdf_hz'])

    if not mdfs:
      return

    mean_mdf = float(np.mean(mdfs))
    if not hasattr(self, 'live_fatiga_mdf_history'):
      self.live_fatiga_mdf_history = []
    self.live_fatiga_mdf_history.append(mean_mdf)

    n_hist = len(self.live_fatiga_mdf_history)
    if n_hist < 4:
      self.lbl_fatiga_dot.setStyleSheet("color: #39ff14; font-size: 15px; font-weight: bold;")
      self.lbl_fatiga_text.setStyleSheet("color: #39ff14; font-size: 11px; font-weight: bold; font-family: 'Courier New', monospace;")
      self.frame_fatiga.setStyleSheet("background-color: #0b1a14; border: 1.5px solid #39ff14; border-radius: 4px;")
      self.lbl_fatiga_text.setText(f"MDF BASAL ({mean_mdf:.1f} Hz)")
      return

    n_base = min(5, n_hist)
    base_mdf = float(np.mean(self.live_fatiga_mdf_history[:n_base]))
    curr_mdf = float(np.mean(self.live_fatiga_mdf_history[-min(4, n_hist):]))
    delta_pct = ((curr_mdf - base_mdf) / (base_mdf + 1e-6)) * 100.0
    ratio_pct = max(0.0, (curr_mdf / (base_mdf + 1e-6)) * 100.0)

    if delta_pct <= -15.0:
      # Fatiga Severa (caída > 15% de la frecuencia mediana)
      self.lbl_fatiga_dot.setStyleSheet("color: #ff3355; font-size: 15px; font-weight: bold;")
      self.lbl_fatiga_text.setStyleSheet("color: #ff3355; font-size: 11px; font-weight: bold; font-family: 'Courier New', monospace;")
      self.frame_fatiga.setStyleSheet("background-color: #240a10; border: 1.5px solid #ff3355; border-radius: 4px;")
      self.lbl_fatiga_text.setText(f"FATIGA ALERTA ({delta_pct:.0f}% MDF - Sugerido descanso)")
    elif delta_pct <= -8.0:
      # Fatiga Moderada
      self.lbl_fatiga_dot.setStyleSheet("color: #ffcc00; font-size: 15px; font-weight: bold;")
      self.lbl_fatiga_text.setStyleSheet("color: #ffcc00; font-size: 11px; font-weight: bold; font-family: 'Courier New', monospace;")
      self.frame_fatiga.setStyleSheet("background-color: #241d08; border: 1.5px solid #ffcc00; border-radius: 4px;")
      self.lbl_fatiga_text.setText(f"FATIGA MODERADA ({delta_pct:.0f}% MDF)")
    else:
      # Estable
      self.lbl_fatiga_dot.setStyleSheet("color: #39ff14; font-size: 15px; font-weight: bold;")
      self.lbl_fatiga_text.setStyleSheet("color: #39ff14; font-size: 11px; font-weight: bold; font-family: 'Courier New', monospace;")
      self.frame_fatiga.setStyleSheet("background-color: #0b1a14; border: 1.5px solid #39ff14; border-radius: 4px;")
      self.lbl_fatiga_text.setText(f"FATIGA: ESTABLE ({ratio_pct:.0f}% MDF | {curr_mdf:.1f} Hz)")

  def _preguntar_lanzar_pca(self, modo="Secuencia Continua"):
    """
    Pregunta al operador si desea abrir inmediatamente el Generador PCA / UMAP
    para procesar los datos de la sesión recién guardada.
    """
    from datetime import datetime
    fecha_str = datetime.now().strftime("%Y-%m-%d")
    prueba = getattr(self, 'autoforge_prueba', 'Prueba1')
    sujeto = getattr(self, 'autoforge_sujeto', 'Sujeto1')
    if modo == "Secuencia Continua":
      med_folder = f"SecuenciaContinua_{prueba}_{sujeto}"
    else:
      med_folder = f"{prueba}_{sujeto}"
    rel_path = f"{fecha_str}/{med_folder}"

    resp = QtWidgets.QMessageBox.question(
      self,
      "Procesar PCA / UMAP",
      f"<b>¡Sesión grabada y archivada con éxito!</b><br><br>"
      f"Medición: <code>{rel_path}</code><br><br>"
      f"¿Deseas procesar y visualizar la proyección <b>PCA / UMAP</b> de esta sesión ahora?",
      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
      QtWidgets.QMessageBox.Yes
    )
    if resp == QtWidgets.QMessageBox.Yes:
      self._lanzar_generador_pca(fecha_str=fecha_str, rel_path=rel_path)

  def _lanzar_generador_pca(self, fecha_str, rel_path):
    """Lanza generador_pca_umap.py en segundo plano pre-cargando la medición."""
    import subprocess
    cand_paths = [
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "generador_pca_umap.py"),
      os.path.join(get_project_root(), "deep_learning", "pca_umap_clustering", "generador_pca_umap.py"),
      os.path.join(get_project_root(), "generador_pca_umap.py"),
    ]
    script_path = None
    for p in cand_paths:
      if os.path.isfile(p):
        script_path = p
        break

    if not script_path:
      print(f"[PCA] No se encontró generador_pca_umap.py en {cand_paths}")
      return

    try:
      cmd = [sys.executable, script_path, "--fecha", fecha_str, "--mediciones", rel_path]
      subprocess.Popen(cmd)
      print(f"[PCA] Lanzado Generador PCA/UMAP: {' '.join(cmd)}")
    except Exception as exc:
      print(f"[PCA] Error al lanzar generador PCA: {exc}")

  # ==============================================================================
  # --- MEDIDOR PCA / SEPARABILIDAD EN TIEMPO REAL Y PANTALLA EXTERNA ---
  # ==============================================================================
  def reset_medidor_pca_vivo(self):
    """Limpia los vectores acumulados del medidor PCA en tiempo real."""
    self.live_pca_features = []
    self.live_pca_labels = []
    if hasattr(self, 'pca_live_plot') and self.pca_live_plot is not None:
      self.pca_live_plot.clear()
    if hasattr(self, 'lbl_pca_live_metrics') and self.lbl_pca_live_metrics is not None:
      self.lbl_pca_live_metrics.setText("Acc: --% | Sil: -- | N=0")

  def _actualizar_medidor_pca_vivo(self, pulse_event):
    """
    Proyecta en tiempo real los pulsos detectados (emg_norm_win remuestreado a 20 pts/canal = 60D,
    idéntico a generador_pca_umap.py) sobre un espacio 2D PCA y calcula la exactitud de clustering
    no supervisado (K-Means + Algoritmo Húngaro) y el Silhouette Score en vivo.
    """
    if not hasattr(self, 'pca_live_plot') or self.pca_live_plot is None:
      return
    emg_win = pulse_event.get('emg_norm_win')
    if emg_win is None or not isinstance(emg_win, np.ndarray) or emg_win.ndim != 2 or emg_win.shape[1] < 4:
      return

    target_len = 20
    vec_parts = []
    for c_i in range(min(3, emg_win.shape[0])):
      seg = np.asarray(emg_win[c_i], dtype=np.float64)
      old_x = np.linspace(0.0, 1.0, len(seg))
      new_x = np.linspace(0.0, 1.0, target_len)
      seg_rs = np.interp(new_x, old_x, seg)
      seg_rs = np.maximum(seg_rs, 0.0)
      vec_parts.append(seg_rs)
    while len(vec_parts) < 3:
      vec_parts.append(np.zeros(target_len, dtype=np.float64))
    feat_vec = np.concatenate(vec_parts)

    vocal_lbl = str(pulse_event.get('vocal') or '').strip().upper()
    if not vocal_lbl:
      vocal_lbl = "?"

    if not hasattr(self, 'live_pca_features'):
      self.live_pca_features = []
      self.live_pca_labels = []
    self.live_pca_features.append(feat_vec)
    self.live_pca_labels.append(vocal_lbl)

    X_arr = np.asarray(self.live_pca_features, dtype=np.float64)
    Y_arr = np.asarray(self.live_pca_labels)
    n_samples = len(X_arr)

    # Proyección 2D: PCA por SVD (centrado y escalado suave) si N >= 3, o coordenadas baricéntricas si N < 3
    if n_samples >= 3:
      X_centered = X_arr - np.mean(X_arr, axis=0, keepdims=True)
      std_col = np.std(X_centered, axis=0, keepdims=True) + 1e-6
      X_scaled = X_centered / std_col
      try:
        u, s, vt = np.linalg.svd(X_scaled, full_matrices=False)
        X_2d = X_scaled @ vt[:2].T
      except Exception:
        X_2d = X_scaled[:, :2]
    else:
      # Fallback baricéntrico para los primeros 1-2 pulsos
      X_2d = np.column_stack([
        np.mean(X_arr[:, :target_len], axis=1) - np.mean(X_arr[:, target_len:2*target_len], axis=1),
        np.mean(X_arr[:, 2*target_len:3*target_len], axis=1)
      ])

    # Calcular métricas de separabilidad (K-Means + Húngaro y Silhouette) cuando hay >= 4 muestras y >= 2 vocales reales
    acc_str = "--%"
    sil_str = "--"
    valid_mask = np.array([y in ('A', 'E', 'I', 'O', 'U') for y in Y_arr], dtype=bool)
    Y_val = Y_arr[valid_mask]
    X_val = X_2d[valid_mask]
    clases_unicas = sorted(list(set(Y_val)))
    if len(clases_unicas) >= 2 and len(Y_val) >= max(4, len(clases_unicas) + 1):
      try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import confusion_matrix, silhouette_score
        from scipy.optimize import linear_sum_assignment
        n_k = len(clases_unicas)
        km = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        y_pred = km.fit_predict(X_val)
        y_true_int = np.array([clases_unicas.index(v) for v in Y_val])
        cm = confusion_matrix(y_true_int, y_pred)
        r_ind, c_ind = linear_sum_assignment(-cm)
        acc_val = (cm[r_ind, c_ind].sum() / float(len(Y_val))) * 100.0
        acc_str = f"{acc_val:.0f}%"
        if len(Y_val) > n_k:
          sil_val = float(silhouette_score(X_val, y_true_int))
          sil_str = f"{sil_val:.2f}"
      except Exception:
        pass

    sep_last = float(pulse_event.get('separability_pct', 0.0))
    self.lbl_pca_live_metrics.setText(f"Acc:{acc_str} Sil:{sil_str} Sep:{sep_last:.0f}% N={n_samples}")

    # Renderizar scatter 2D en self.pca_live_plot
    self.pca_live_plot.clear()
    colores_map = {
      'A': (255, 51, 85),
      'E': (0, 180, 216),
      'I': (57, 255, 20),
      'O': (199, 125, 255),
      'U': (255, 209, 102),
      '?': (0, 255, 204),
    }
    for v_key in sorted(set(Y_arr)):
      idx_v = np.where(Y_arr == v_key)[0]
      if len(idx_v) == 0:
        continue
      rgb = colores_map.get(v_key, (0, 255, 204))
      pts_v = X_2d[idx_v]
      scatter = pg.ScatterPlotItem(
        x=pts_v[:, 0],
        y=pts_v[:, 1],
        size=8,
        pen=pg.mkPen('#ffffff', width=0.6),
        brush=pg.mkBrush(*rgb, 210),
      )
      self.pca_live_plot.addItem(scatter)
      if len(pts_v) >= 2 and v_key in ('A', 'E', 'I', 'O', 'U'):
        cx, cy = float(np.mean(pts_v[:, 0])), float(np.mean(pts_v[:, 1]))
        centroid_item = pg.ScatterPlotItem(
          x=[cx],
          y=[cy],
          size=13,
          symbol='d',
          pen=pg.mkPen('#ffffff', width=1.4),
          brush=pg.mkBrush(*rgb, 255),
        )
        self.pca_live_plot.addItem(centroid_item)

    # Resaltar el último pulso ingresado con un halo blanco
    last_pt = X_2d[-1]
    halo = pg.ScatterPlotItem(
      x=[float(last_pt[0])],
      y=[float(last_pt[1])],
      size=15,
      symbol='o',
      pen=pg.mkPen('#ffffff', width=2.0),
      brush=pg.mkBrush(255, 255, 255, 40),
    )
    self.pca_live_plot.addItem(halo)

  def abrir_dialogo_pantalla_externa(self):
    """Abre el panel de configuración de Pantalla Externa y Modo Sujeto Ciego."""
    dlg = DialogoPantallaExterna(self)
    dlg.exec()

  def iniciar_servidor_display_remoto(self, puerto=8765, token=""):
    """Inicia el ServidorEstimulos HTTP en segundo plano para celulares/tablets por USB o Wi-Fi."""
    if ServidorEstimulos is None:
      print("[Display Remoto] Módulo display_remoto_servidor no disponible.")
      return False
    self.detener_servidor_display_remoto()
    try:
      self.remote_display_server = ServidorEstimulos(puerto=int(puerto), token=token or None)
      self.remote_display_server.start()
      ok = self.remote_display_server.esperar_listo(timeout=3.0)
      if ok:
        self.config_mgr.set("display_remoto", "habilitado", True)
        self.config_mgr.set("display_remoto", "puerto", int(puerto))
        self.config_mgr.set("display_remoto", "token", str(token or ""))
        print(f"[Display Remoto] Servidor activo en http://127.0.0.1:{self.remote_display_server.puerto}/ (Red: http://{_ip_local()}:{self.remote_display_server.puerto}/)")
        return True
    except Exception as exc:
      print(f"[Display Remoto] Error al iniciar servidor: {exc}")
    return False

  def detener_servidor_display_remoto(self):
    srv = getattr(self, 'remote_display_server', None)
    if srv is not None:
      try:
        srv.detener(timeout=1.5)
      except Exception:
        pass
      self.remote_display_server = None

  def abrir_ventana_estimulos_segundo_monitor(self):
    """Abre la ventana nativa de estímulos para un 2do monitor HDMI/VGA."""
    if getattr(self, 'external_stimulus_window', None) is None:
      self.external_stimulus_window = VentanaEstimulosExterna(self)
    self.external_stimulus_window.show()
    self.external_stimulus_window.raise_()
    self.external_stimulus_window.activateWindow()

  def _publicar_estado_display_remoto(self, palabra="", palabra_siguiente="", pulso_actual=0,
                                      total_pulsos=0, flash=False, count_in=None, grabando=None,
                                      fase="", instruccion_fase="", subtitulo_fase="",
                                      ciclo_actual=1, total_ciclos=1, pre_cambio=False,
                                      tira_secuencia=None, tira_html=""):
    '''Sincroniza el estímulo actual, las instrucciones de fase y la tira de secuencia con la Pantalla Externa (Web y 2do Monitor).'''
    bpm_val = float(self.spin_bpm.value()) if hasattr(self, 'spin_bpm') else 30.0
    sub_val = int(self.spin_metro_subdivs.value()) if hasattr(self, 'spin_metro_subdivs') else 4
    sub_act = int(getattr(self, 'metro_subdiv_index', 0)) + 1
    sesion_str = f"{getattr(self, 'autoforge_prueba', '')}_{getattr(self, 'autoforge_sujeto', '')}".strip("_") or "S1"
    is_rec = bool(getattr(self, 'is_recording', False) or getattr(self, 'is_autoforge_running', False)) if grabando is None else bool(grabando)
    jit_on = bool(hasattr(self, 'chk_metro_jitter') and self.chk_metro_jitter.isChecked())

    if tira_secuencia is None:
      words = getattr(self, 'autoforge_words', []) or []
      if getattr(self, 'is_autoforge_continuo', False):
        full_seq = getattr(self, 'autoforge_full_sequence', None) or words
      else:
        full_seq = words
      tira_secuencia = []
      if full_seq:
        n_w = len(full_seq)
        act_i = (max(0, int(pulso_actual) - 1) % n_w) if getattr(self, 'is_autoforge_continuo', False) else int(getattr(self, 'autoforge_word_idx', 0)) % n_w
        idxs = list(range(n_w)) if n_w <= 8 else [(act_i + off) % n_w for off in range(-1, 6)]
        for i_w in idxs:
          tira_secuencia.append({
            "vocal": str(full_seq[i_w]).upper(),
            "activo": (i_w == act_i),
            "siguiente": (i_w == (act_i + 1) % n_w),
          })

    srv = getattr(self, 'remote_display_server', None)
    if srv is not None and srv.activo():
      try:
        srv.set_estado(
          palabra=str(palabra or ""),
          palabra_siguiente=str(palabra_siguiente or ""),
          bpm=bpm_val,
          subdiv=sub_val,
          sub_actual=sub_act,
          sesion=sesion_str,
          grabando=is_rec,
          t0_epoch=time.time(),
          t0_perf_counter=time.perf_counter(),
          count_in=count_in,
          pulso_actual=int(pulso_actual or 0),
          total_pulsos=int(total_pulsos or 0),
          ciclo_actual=int(ciclo_actual or 1),
          total_ciclos=int(total_ciclos or 1),
          jitter_activo=jit_on,
          tau_s=float(getattr(self, 'current_beat_tau_s', 60.0 / max(1.0, bpm_val))),
          fase=str(fase or getattr(self, 'autoforge_estado_actual_str', 'EN CURSO')),
          instruccion_fase=str(instruccion_fase or ""),
          subtitulo_fase=str(subtitulo_fase or ""),
          tira_secuencia=tira_secuencia,
          pre_cambio=bool(pre_cambio),
          flash=bool(flash),
        )
      except Exception:
        pass

    ext_win = getattr(self, 'external_stimulus_window', None)
    if ext_win is not None and ext_win.isVisible():
      try:
        ext_win.actualizar_estimulo(
          palabra=palabra,
          palabra_siguiente=palabra_siguiente,
          pulso_actual=pulso_actual,
          total_pulsos=total_pulsos,
          flash=flash,
          count_in=count_in,
          sesion=sesion_str,
          bpm=bpm_val,
          instruccion_fase=instruccion_fase,
          subtitulo_fase=subtitulo_fase,
          tira_html=tira_html,
          ciclo_actual=ciclo_actual,
          total_ciclos=total_ciclos,
          sub_actual=sub_act,
          subdiv=sub_val,
          pre_cambio=pre_cambio,
        )
      except Exception:
        pass

# --- DETECTOR DE PULSOS FONATORIOS Y SEPARABILIDAD EN VENTANA TAU ---
# =============================================================================

class PhonatoryPulseDetector:
    """
    Detector continuo de eventos fonatorios en streaming bioelectrico:
    1. Evaluacion prioritaria por Gate Doble electromiografico con histeresis sobre la Norma Tricanal S_emg
       de los canales musculares. Estimacion robusta del piso basal de ruido y anclaje de la ventana
       de 0.5 tau en el onset fisiologico (cruce ascendente de u_bajo).
    2. Fallback / Alternativa acustica por Microfono (Canal 3) si Gate Doble no detecto pulso.
    3. Extraccion de ventana 0.5 tau [t_onset - 0.10*tau, t_onset + 0.40*tau] y calculo de
       proporciones relativas intercanal.
    4. Normalizacion estricta por Supremo Tricanal por Pulso Individual y computo de separabilidad.
    """
    # Parametros del Gate Doble electromiografico (EMG).
    FACTOR_ALTO = 4.0
    FACTOR_BAJO = 1.8
    MIN_PERSISTENCIA_S = 0.040
    MIN_PERSISTENCIA_BAJA_S = 0.060
    VENTANA_PRE_TAU = 0.10
    VENTANA_POST_TAU = 0.40

    def __init__(self, plotter):
        self.plotter = plotter
        self.last_pulse_t_abs = -9999.0
        self.last_pulse_wall_time = 0.0
        self.last_pulse_abs_sample = -10**12
        self.total_pulses_detected = 0

        # Estado del estimador robusto del piso de ruido basal de S_emg.
        self._emg_floor_base = None
        self._emg_floor_sigma = None
        self._emg_floor_updates = 0
        self._emg_floor_firma = None

    def _firma_calibracion_ruido(self, emg_indices):
        """
        Devuelve una firma de la calibracion de ruido vigente (media + desviacion por canal).
        Permite reiniciar el piso basal cuando el ruido se recalibra durante la adquisicion.
        """
        try:
            plotter = self.plotter
            niveles = getattr(plotter, 'noise_levels', None)
            stds = getattr(plotter, 'initial_noise_std', None)
            niveles = niveles if niveles is not None else []
            stds = stds if stds is not None else []
            firma = 0.0
            for c in emg_indices:
                if c < len(niveles):
                    firma += float(niveles[c])
                if c < len(stds):
                    firma += float(stds[c])
            return round(firma, 2)
        except Exception:
            return None

    def _estimar_piso_ruido_emg(self, s_emg, emg_indices):
        """
        Estima el piso basal y la dispersion robusta de la Norma Tricanal S_emg.

        Usa los descansos contenidos en la ventana de busqueda (incluye los primeros
        segundos de adquisicion y los intervalos inter-pulso). El piso se obtiene del
        percentil 20 de las muestras de reposo. La dispersion combina la cola inferior
        (P10-P40), que resiste rafagas de alta ocupacion, con el MAD y el rango
        intercuartil. Las muestras por encima de Q3 + 1.5*IQR se tratan como burst.
        """
        base_actual = self._emg_floor_base
        sigma_actual = self._emg_floor_sigma

        try:
            valid = np.asarray(s_emg, dtype=np.float64)
            valid = valid[np.isfinite(valid)]
        except Exception:
            return (base_actual if base_actual is not None else 0.0,
                    sigma_actual if sigma_actual is not None else 1.0)
        if valid.size < 8:
            return (base_actual if base_actual is not None else 0.0,
                    sigma_actual if sigma_actual is not None else 1.0)

        q10, q20, q25, q40, q75 = np.percentile(
            valid, [10.0, 20.0, 25.0, 40.0, 75.0]
        )
        iqr = max(0.0, float(q75 - q25))
        limite_reposo = float(q75 + 1.5 * iqr)
        reposo = valid[valid <= limite_reposo]
        if reposo.size < 8:
            reposo = valid

        base_bloque = max(0.0, float(np.percentile(reposo, 20.0)))
        mediana_reposo = float(np.median(reposo))
        mad_bloque = float(np.median(np.abs(reposo - mediana_reposo)))

        # La dispersion se estima con la cola inferior de la distribucion
        # (percentiles 10-40), que permanece en el descanso inter-pulso incluso
        # cuando las rafagas ocupan una fraccion alta de la ventana. El
        # estimador clasico por IQR se usa solo como cota superior: si hay
        # muchas muestras activas, Q3 y el IQR se contaminan y sobrestiman sigma.
        sigma_cola = (float(q40) - float(q10)) / 1.0283
        sigma_iqr = iqr / 1.349
        if sigma_cola > 0.0 and sigma_iqr > 0.0:
            sigma_bloque = min(sigma_cola, sigma_iqr)
        elif sigma_cola > 0.0:
            sigma_bloque = sigma_cola
        elif sigma_iqr > 0.0:
            sigma_bloque = sigma_iqr
        else:
            sigma_bloque = 1.4826 * mad_bloque
        sigma_bloque = max(0.5, sigma_bloque)

        # Semilla proveniente de la calibracion inicial de ruido por canal.
        sigma_semilla = 0.0
        try:
            plotter = self.plotter
            stds = getattr(plotter, 'initial_noise_std', None)
            stds = stds if stds is not None else []
            suma_var = 0.0
            for c in emg_indices:
                if c < len(stds):
                    suma_var += float(stds[c]) ** 2
            sigma_semilla = float(np.sqrt(max(0.0, suma_var)))
        except Exception:
            sigma_semilla = 0.0

        firma = self._firma_calibracion_ruido(emg_indices)
        if base_actual is None or sigma_actual is None or firma != self._emg_floor_firma:
            # Primera estimacion o recalibracion del ruido: convergencia rapida.
            # La semilla inicial se acota con el estimador robusto del bloque para
            # que una calibracion ruidosa no bloquee la deteccion durante segundos.
            sigma_sem = 0.5 * sigma_semilla
            if sigma_semilla > 0.0 and sigma_bloque > 0.0:
                sigma_sem = min(sigma_sem, 3.0 * sigma_bloque)
            self._emg_floor_base = base_bloque
            self._emg_floor_sigma = max(0.5, sigma_bloque, sigma_sem)
            self._emg_floor_updates = 1
            self._emg_floor_firma = firma
            return self._emg_floor_base, self._emg_floor_sigma

        prev_base = float(base_actual)
        prev_sigma = float(sigma_actual)
        umbral_reposo_previo = prev_base + self.FACTOR_ALTO * prev_sigma
        fraccion_reposo = float(np.mean(valid <= umbral_reposo_previo))

        # Durante el arranque se converge rapido; luego se promedia lento para
        # no arrastrar el piso hacia arriba con una rafaga de habla.
        if self._emg_floor_updates < 8:
            alpha = 0.35
        else:
            alpha = 0.06

        actualizar = (
            self._emg_floor_updates < 8
            or fraccion_reposo >= 0.20
            or base_bloque <= prev_base + 1.5 * prev_sigma
        )
        if actualizar:
            self._emg_floor_base = max(0.0, (1.0 - alpha) * prev_base + alpha * base_bloque)
            self._emg_floor_sigma = max(0.5, (1.0 - alpha) * prev_sigma + alpha * sigma_bloque)
            self._emg_floor_updates += 1

        return self._emg_floor_base, self._emg_floor_sigma

    def _calcular_umbrales_gate(self, s_emg, base_ruido, std_ruido):
        """
        Calcula u_bajo y u_alto a partir del piso basal robusto.
        u_bajo define el cruce ascendente fisiologico (onset); u_alto confirma la
        activacion y aporta la histeresis del Gate Doble.
        """
        base = max(0.0, float(base_ruido if base_ruido is not None else 0.0))
        sigma = max(0.5, float(std_ruido if std_ruido is not None else 1.0))

        u_bajo = base + self.FACTOR_BAJO * sigma
        u_alto = base + self.FACTOR_ALTO * sigma

        # Ajuste opcional a la dinamica del burst: si la activacion es clara pero
        # moderada, u_alto no debe quedar por encima de toda la rafaga.
        try:
            p99 = float(np.percentile(np.asarray(s_emg, dtype=np.float64), 99.0))
        except Exception:
            p99 = 0.0
        dinamica = max(0.0, p99 - base)
        if dinamica >= 3.0 * sigma:
            u_alto = min(u_alto, base + 0.55 * dinamica)

        u_bajo = min(u_bajo, u_alto - 0.5 * sigma)
        u_bajo = max(u_bajo, base + 0.75 * sigma)
        u_alto = max(u_alto, u_bajo + 0.5 * sigma)
        return float(u_alto), float(max(0.5, u_bajo))

    def _detectar_activaciones_emg(self, s_emg, fs, u_bajo, u_alto, min_peak_dist):
        """
        Detecta activaciones EMG con histeresis y persistencia minima.

        Una activacion es valida solo si S_emg permanece por encima de u_alto al
        menos MIN_PERSISTENCIA_S (30-50 ms), descartando espigas breves. Para cada
        pico confirmado se busca hacia atras el cruce ascendente de u_bajo, que
        define el onset fisiologico muscular.
        """
        activaciones = []
        try:
            senal = np.asarray(s_emg, dtype=np.float64)
        except Exception:
            return activaciones
        n = int(senal.size)
        if n == 0 or u_alto <= u_bajo:
            return activaciones

        min_muestras_alto = max(1, int(round(self.MIN_PERSISTENCIA_S * fs)))
        min_muestras_bajo = max(min_muestras_alto, int(round(self.MIN_PERSISTENCIA_BAJA_S * fs)))

        try:
            pks, _ = signal.find_peaks(senal, height=u_alto, distance=min_peak_dist)
        except Exception:
            return activaciones

        for pk in pks:
            pk = int(pk)
            ini_alto = pk
            while ini_alto > 0 and senal[ini_alto - 1] >= u_alto:
                ini_alto -= 1
            fin_alto = pk
            while fin_alto + 1 < n and senal[fin_alto + 1] >= u_alto:
                fin_alto += 1

            if (fin_alto - ini_alto + 1) < min_muestras_alto:
                continue

            onset = ini_alto
            while onset > 0 and senal[onset - 1] >= u_bajo:
                onset -= 1

            # Si el cruce ascendente queda fuera de la ventana de busqueda no es
            # posible anclar fisiologicamente el onset: se descarta el candidato.
            if onset <= 0:
                continue

            if (fin_alto - onset + 1) < min_muestras_bajo:
                continue

            activaciones.append((pk, int(onset), int(fin_alto)))

        return activaciones

    def process(self):
        """Evalua el buffer continuo reciente para detectar fonaciones y calcular proporciones."""
        plotter = self.plotter
        if not getattr(plotter, 'is_acquiring', False):
            return
        if plotter.plot_buffer_datos is None or plotter.NUM_CANALES < 1:
            return
        if plotter.plot_vector_tiempo is None or len(plotter.plot_vector_tiempo) < 100:
            return

        fs = float(plotter.SAMPLE_RATE)
        if fs <= 0:
            return

        bpm = float(plotter.spin_bpm.value()) if hasattr(plotter, 'spin_bpm') else 60.0
        if bpm <= 0:
            bpm = 60.0
        tau_nominal = 60.0 / bpm
        tau = float(getattr(plotter, 'current_beat_tau_s', tau_nominal) or tau_nominal)
        if hasattr(plotter, 'chk_metro_jitter') and plotter.chk_metro_jitter.isChecked():
            jit_pct = float(plotter.spin_metro_jitter_pct.value()) / 100.0
            tau_min_ref = max(0.75, tau_nominal * (1.0 - jit_pct))
        else:
            tau_min_ref = tau_nominal

        # Separar canales musculares vs canal acustico
        emg_indices = [i for i in range(plotter.NUM_CANALES) if not plotter._is_mic_channel(i)]
        mic_indices = [i for i in range(plotter.NUM_CANALES) if plotter._is_mic_channel(i)]
        if not emg_indices:
            return

        mic_idx = mic_indices[0] if mic_indices else None

        # La ventana fonatoria dura 0.5 tau y se ancla en el onset fisiologico:
        # [t_onset - 0.10*tau, t_onset + 0.40*tau]. Para que la ventana completa
        # este en el buffer pasado (t <= 0), el pico muscular debe haber alcanzado
        # al menos t <= -0.40*tau. Evaluamos los ultimos 2.5 tau de señal.
        now_wall = time.perf_counter()
        search_t_min = -min(float(plotter.PLOT_DURATION_S), 2.5 * tau)

        # Excluir el relleno previo al inicio de la adquisicion: el buffer puede
        # contener ceros o datos viejos que contaminarian el piso de ruido basal.
        try:
            acq_start = float(getattr(plotter, 'acq_start_time', 0.0) or 0.0)
        except Exception:
            acq_start = 0.0
        if acq_start > 0.0:
            elapsed_acq = max(0.0, now_wall - acq_start)
            search_t_min = max(search_t_min, -elapsed_acq + 0.02)

        search_t_max = -self.VENTANA_POST_TAU * tau
        if search_t_min >= search_t_max:
            return

        idx_start_search = np.searchsorted(plotter.plot_vector_tiempo, search_t_min)
        idx_end_search = np.searchsorted(plotter.plot_vector_tiempo, search_t_max)
        n_search = idx_end_search - idx_start_search
        min_samples = int(0.35 * tau * fs)
        if n_search < min_samples:
            return

        min_peak_dist = max(5, int(0.55 * tau_min_ref * fs))
        detected_event = None
        abs_ref = float(getattr(plotter, 'total_muestras_procesadas', 0))

        # ---------------------------------------------------------------------
        # RUTA 1: GATE DOBLE (NORMA TRICANAL S_emg) - PRIORITARIO
        # ---------------------------------------------------------------------
        emg_env = plotter.env_buffer_datos[emg_indices, idx_start_search:idx_end_search]
        noise_bases = []
        for c in emg_indices:
            lvl = plotter.noise_levels[c] if (hasattr(plotter, 'noise_levels') and c < len(plotter.noise_levels)) else 0.0
            noise_bases.append(lvl)
        noise_bases_arr = np.array(noise_bases, dtype=np.float64)[:, None]
        clean_emg_env = np.maximum(0.0, emg_env - noise_bases_arr)
        s_emg = np.sqrt(np.sum(clean_emg_env ** 2, axis=0))

        # Piso de ruido basal robusto (primeros segundos + descansos inter-pulso).
        # u_bajo marca el cruce ascendente fisiologico y u_alto confirma el burst
        # con histeresis y persistencia minima de 30-50 ms.
        base_ruido, std_ruido = self._estimar_piso_ruido_emg(s_emg, emg_indices)
        u_alto, u_bajo = self._calcular_umbrales_gate(s_emg, base_ruido, std_ruido)

        activaciones = self._detectar_activaciones_emg(
            s_emg, fs, u_bajo, u_alto, min_peak_dist
        )
        for pk_local, onset_local, _fin_local in reversed(activaciones):
            onset_global = idx_start_search + onset_local
            t_onset_rel = float(plotter.plot_vector_tiempo[onset_global])

            # La ventana 0.5 tau debe quedar completamente contenida en el pasado.
            if t_onset_rel > -self.VENTANA_POST_TAU * tau:
                continue

            onset_abs = int(round(abs_ref + t_onset_rel * fs))
            if onset_abs - self.last_pulse_abs_sample < 0.55 * tau_min_ref * fs:
                continue
            if now_wall - self.last_pulse_wall_time < 0.45 * tau_min_ref:
                continue

            pk_global = idx_start_search + pk_local
            t_pk_emg = float(plotter.plot_vector_tiempo[pk_global])
            detected_event = {
                'method': 'GATE_DOBLE',
                't_peak_rel': t_pk_emg,
                't_peak_abs': now_wall + t_pk_emg,
                't_onset_rel': t_onset_rel,
                't_anchor_rel': t_onset_rel,
                'event_abs_sample': onset_abs,
                'gate_u_alto': float(u_alto),
                'gate_u_bajo': float(u_bajo),
                'gate_base_ruido': float(base_ruido),
                'gate_std_ruido': float(std_ruido)
            }
            break

        # ---------------------------------------------------------------------
        # RUTA 2: DETECTOR ACUSTICO POR MICROFONO (FALLBACK SI GATE DOBLE NO SALTO)
        # ---------------------------------------------------------------------
        if detected_event is None and mic_idx is not None:
            raw_mic = plotter.plot_buffer_datos[mic_idx, idx_start_search:idx_end_search]
            k_len = max(3, int(0.040 * fs))
            kernel = np.ones(k_len, dtype=np.float64) / k_len
            mic_env = np.convolve(np.abs(raw_mic), kernel, mode='same')

            # Umbral de deteccion de audio
            base_mean = plotter.initial_noise_mean[mic_idx] if (hasattr(plotter, 'initial_noise_mean') and mic_idx < len(plotter.initial_noise_mean)) else 0.0
            base_std = plotter.initial_noise_std[mic_idx] if (hasattr(plotter, 'initial_noise_std') and mic_idx < len(plotter.initial_noise_std)) else 0.0
            th_mic = max(20.0, base_mean + 3.0 * max(2.0, base_std))
            if hasattr(plotter, 'spin_peak_th'):
                th_mic = max(20.0, min(th_mic, plotter.spin_peak_th.value() * 0.25))

            pks, _ = signal.find_peaks(mic_env, height=th_mic, distance=min_peak_dist)
            if len(pks) > 0:
                pk_local = pks[-1]
                pk_global = idx_start_search + pk_local
                t_pk_mic = float(plotter.plot_vector_tiempo[pk_global])
                t_pk_abs = now_wall + t_pk_mic

                # Blanking acústico: descartar picos del micrófono que coincidan con la locución de la PC ("Di A", etc.)
                vcp = getattr(plotter, 'voice_cue_player', None)
                en_blanking_voz = False
                if vcp is not None and hasattr(vcp, 'speech_blanking_windows'):
                    for w_ini, w_fin in vcp.speech_blanking_windows:
                        if w_ini <= t_pk_abs <= w_fin:
                            en_blanking_voz = True
                            break
                dt_abs = t_pk_abs - self.last_pulse_t_abs
                dt_wall = now_wall - self.last_pulse_wall_time
                if not en_blanking_voz and dt_abs >= 0.55 * tau_min_ref and dt_wall >= 0.45 * tau_min_ref:
                    # En fonacion de Cande el pico de mic coincide o retrasa levemente al EMG (~150 ms).
                    # El ancla se desplaza 0.10*tau para preservar la ventana acustica historica
                    # de 40% pre / 60% post alrededor del centro acustico.
                    emd_s = min(0.180, 0.15 * tau)
                    t_centro_mic = t_pk_mic - emd_s
                    t_anchor_rel = t_centro_mic - self.VENTANA_PRE_TAU * tau
                    detected_event = {
                        'method': 'MIC',
                        't_peak_rel': t_pk_mic,
                        't_peak_abs': t_pk_abs,
                        't_onset_rel': t_anchor_rel,
                        't_anchor_rel': t_anchor_rel,
                        'event_abs_sample': int(round(abs_ref + t_anchor_rel * fs))
                    }

        if detected_event is None:
            return

        # ---------------------------------------------------------------------
        # EXTRACCION Y CALCULO EN VENTANA 0.5 TAU
        # ---------------------------------------------------------------------
        self.last_pulse_t_abs = detected_event['t_peak_abs']
        self.last_pulse_wall_time = now_wall
        if 'event_abs_sample' in detected_event:
            self.last_pulse_abs_sample = int(detected_event['event_abs_sample'])
        self.total_pulses_detected += 1

        # Ancla absoluta del evento (onset fisiologico para GATE_DOBLE).
        # La ventana se define como [t_anchor - 0.10*tau, t_anchor + 0.40*tau].
        t_anchor = detected_event['t_anchor_rel']
        t_w_start = t_anchor - self.VENTANA_PRE_TAU * tau
        t_w_end = t_anchor + self.VENTANA_POST_TAU * tau
        t_center = t_anchor + 0.15 * tau

        w_idx_start = max(0, min(plotter.PLOT_SAMPLES - 1, np.searchsorted(plotter.plot_vector_tiempo, t_w_start)))
        w_idx_end = max(w_idx_start + 10, min(plotter.PLOT_SAMPLES, np.searchsorted(plotter.plot_vector_tiempo, t_w_end)))

        emg_raw_win = plotter.plot_buffer_datos[emg_indices, w_idx_start:w_idx_end]
        
        # 1. Normalizacion Obligatoria por Supremo Tricanal por Pulso Individual
        supremo_val = float(np.max(np.abs(emg_raw_win))) if emg_raw_win.size > 0 else 1.0
        if supremo_val < 1e-6:
            supremo_val = 1.0

        # Señal rectificada y normalizada por el supremo compartido intercanal
        emg_norm_win = np.abs(emg_raw_win) / supremo_val

        # 2. Energia RMS de cada canal normalizado en la ventana tau
        energies_norm = []
        for k in range(len(emg_indices)):
            e_k = float(np.sqrt(np.mean(emg_norm_win[k] ** 2))) if emg_norm_win.shape[1] > 0 else 0.0
            energies_norm.append(e_k)

        sum_e = sum(energies_norm) + 1e-9
        proportions = [e / sum_e for e in energies_norm] # Baricentricas [p0, p1, p2] con suma = 1.0
        proportions_pct = [p * 100.0 for p in proportions] # En porcentaje para reportes

        # Coordenadas baricentricas para el Mapa Triangular (Simplex Ternario)
        # Vertice 0 (Apertura Ch0): (0, 0)
        # Vertice 1 (Retraccion Ch1): (0.5, sqrt(3)/2)
        # Vertice 2 (Protrusion Ch2): (1, 0)
        p0 = proportions[0] if len(proportions) > 0 else 0.3333
        p1 = proportions[1] if len(proportions) > 1 else 0.3333
        p2 = proportions[2] if len(proportions) > 2 else 0.3333
        x_ternary = float(p2 + 0.5 * p1)
        y_ternary = float((np.sqrt(3.0) / 2.0) * p1)

        sorted_p = sorted(proportions_pct, reverse=True)
        if len(sorted_p) >= 2 and sorted_p[0] > 0:
            sep_pct = ((sorted_p[0] - sorted_p[1]) / sorted_p[0]) * 100.0
        else:
            sep_pct = 100.0
        sep_pct = max(0.0, min(100.0, sep_pct))

        dom_idx = int(np.argmax(proportions))
        hw_dom = plotter.active_channel_indices[emg_indices[dom_idx]] if (hasattr(plotter, 'active_channel_indices') and emg_indices[dom_idx] < len(plotter.active_channel_indices)) else dom_idx
        musc_name = plotter.nombres_musculos[emg_indices[dom_idx]] if (hasattr(plotter, 'nombres_musculos') and emg_indices[dom_idx] < len(plotter.nombres_musculos)) else f"Canal {hw_dom}"

        if dom_idx == 0:
            atractor = "/A/ - Apertura (Digastrico)"
        elif dom_idx == 1:
            atractor = "/I/-/E/ - Retraccion (Cigomatico)"
        elif dom_idx == 2:
            atractor = "/O/-/U/ - Protrusion (Orbicular)"
        else:
            atractor = f"Canal {dom_idx} - {musc_name}"

        # Obtener vocal exacta (de vocal_cue_log, de autoforge_full_sequence, del simulador o del atractor muscular)
        vocal_actual = ""
        t_onset_perf_est = now_wall + float(t_anchor)
        cues_list = getattr(plotter, 'vocal_cue_log', None) or []
        if cues_list:
            # Buscar el último pulso del metrónomo cuyo disparo haya ocurrido antes (o muy cerca) del onset
            best_cue = None
            for c_item in reversed(cues_list):
                t_c = float(c_item.get('t_perf') or 0.0)
                if t_c <= t_onset_perf_est + 0.35:
                    best_cue = c_item
                    break
            if best_cue is None:
                best_cue = cues_list[-1]
            vocal_actual = str(best_cue.get('palabra') or best_cue.get('vocal') or '').strip().upper()

        if not vocal_actual and getattr(plotter, 'is_autoforge_running', False):
            words = getattr(plotter, 'autoforge_words', []) or ["A", "E", "I", "O", "U"]
            if getattr(plotter, 'is_autoforge_continuo', False):
                full_seq = getattr(plotter, 'autoforge_full_sequence', None) or words
                b_idx = max(0, int(getattr(plotter, 'metro_beat_count', 1)) - 1)
                vocal_actual = full_seq[b_idx % len(full_seq)].upper()
            else:
                idx_w = int(getattr(plotter, 'autoforge_word_idx', 0))
                if idx_w < len(words):
                    vocal_actual = words[idx_w].upper()

        # Si estamos en Modo Prueba (ej. Cande Secuencia Continua o /a/ normal) sin AutoForge activo:
        if not vocal_actual and getattr(plotter, 'chk_modo_prueba', None) and plotter.chk_modo_prueba.isChecked():
            fuente_txt = str(plotter.cmb_fuente_prueba.currentText()).lower() if hasattr(plotter, 'cmb_fuente_prueba') else ""
            if "continua" in fuente_txt:
                vocal_actual = ["A", "E", "I", "O", "U"][(self.total_pulses_detected - 1) % 5]
            elif "/a/" in fuente_txt:
                vocal_actual = "A"

        # Si es grabación libre manual, colorear según el atractor muscular dominante (A / E / I / O / U)
        if not vocal_actual:
            if dom_idx == 0:
                vocal_actual = "A"
            elif dom_idx == 1:
                vocal_actual = "I" if p1 >= 0.50 else "E"
            elif dom_idx == 2:
                vocal_actual = "U" if p2 >= 0.50 else "O"
            else:
                vocal_actual = "A" 

        full_event = {
            'pulse_num': self.total_pulses_detected,
            'timestamp': time.time(),
            't_detect_perf': time.perf_counter(),
            'detect_sample_idx': getattr(plotter, 'total_muestras_procesadas', 0),
            'method': detected_event['method'],
            'tau': tau,
            'win_dur_s': 0.50 * tau,
            't_peak_rel': detected_event['t_peak_rel'],
            't_onset_rel': detected_event.get('t_onset_rel', t_anchor),
            't_anchor_rel': t_anchor,
            'event_abs_sample': detected_event.get(
                'event_abs_sample', int(round(abs_ref + t_anchor * fs))
            ),
            't_center': t_center,
            'gate_u_alto': detected_event.get('gate_u_alto'),
            'gate_u_bajo': detected_event.get('gate_u_bajo'),
            'gate_base_ruido': detected_event.get('gate_base_ruido'),
            'gate_std_ruido': detected_event.get('gate_std_ruido'),
            'supremo_uV': supremo_val,
            'emg_raw_win': emg_raw_win,
            'emg_norm_win': emg_norm_win,
            'energies_norm': energies_norm,
            'proportions': proportions,
            'proportions_pct': proportions_pct,
            'ternary_xy': (x_ternary, y_ternary),
            'separability_pct': sep_pct,
            'dom_idx': dom_idx,
            'dom_hw': hw_dom,
            'dom_muscle': musc_name,
            'atractor': atractor,
            'vocal': vocal_actual
        }

        if hasattr(plotter, 'detected_pulses_history'):
            plotter.detected_pulses_history.append(full_event)
            if len(plotter.detected_pulses_history) > 100:
                plotter.detected_pulses_history.pop(0)

        plotter._on_pulse_detected(full_event)

# =============================================================================
# --- DIÁLOGO DE SELECCIÓN DE MÚSCULOS AL INICIO ---
# =============================================================================
# =============================================================================
# --- DIÁLOGO DE SELECCIÓN DE MÚSCULOS ---
# =============================================================================
class MuscleSelectionDialog(QtWidgets.QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.parent_plotter = parent
    self.setWindowTitle("Configurar Músculos de la Sesión")
    self.setMinimumWidth(380)
    
    self.config_mgr = ConfigManager()
    self.canales_conf = self.config_mgr.get("canales") or {}
    
    if parent is not None and getattr(parent, 'active_channel_indices', None):
      self.active_hw_channels = list(parent.active_channel_indices)
    else:
      self.active_hw_channels = [0, 1, 2, 3]
    
    self.layout = QtWidgets.QVBoxLayout(self)
    
    self.lbl = QtWidgets.QLabel("Asigne el músculo a cada canal físico activo:")
    self.lbl.setStyleSheet("color: white; font-weight: bold; margin-bottom: 10px;")
    self.layout.addWidget(self.lbl)
    
    self.form_layout = QtWidgets.QFormLayout()
    self.combos = {}
    
    musculos_sugeridos = list(STANDARD_MUSCLES_LIST)
    
    for hw_idx in self.active_hw_channels:
      key = f"Canal {hw_idx}"
      cfg = self.config_mgr.get_channel_config(hw_idx)
      musculo_actual = cfg.get("musculo", key)
      
      cmb = QtWidgets.QComboBox()
      cmb.setEditable(True)
      for m in musculos_sugeridos:
        cmb.addItem(m)
      if musculo_actual not in musculos_sugeridos:
        cmb.insertItem(0, musculo_actual)
      cmb.setCurrentText(musculo_actual)
      
      self.form_layout.addRow(f"ai{hw_idx} - {key}:", cmb)
      self.combos[hw_idx] = cmb
        
    self.layout.addLayout(self.form_layout)
    
    self.btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
    self.btn_box.accepted.connect(self.guardar_y_cerrar)
    self.btn_box.rejected.connect(self.reject)
    self.layout.addWidget(self.btn_box)
    
  def guardar_y_cerrar(self):
    try:
      from utils.config_manager import get_unique_channel_colors
    except ImportError:
      def get_unique_channel_colors(items):
        return ["#ff754b", "#00a896", "#00cc44", "#ff0000"]

    ch_list = []
    for hw_idx in self.active_hw_channels:
      m_text = self.combos[hw_idx].currentText().strip()
      ch_list.append({
        "idx": hw_idx,
        "musculo": m_text,
        "is_mic": ("mic" in m_text.lower() or "audio" in m_text.lower() or (hw_idx == 3 and not m_text))
      })
      
    unique_cols = get_unique_channel_colors(ch_list)
    canales_conf = self.config_mgr.get("canales") or {}
    for idx, hw_idx in enumerate(self.active_hw_channels):
      key = f"Canal {hw_idx}"
      if key not in canales_conf:
        canales_conf[key] = {}
      m_text = self.combos[hw_idx].currentText().strip()
      canales_conf[key]["musculo"] = m_text
      canales_conf[key]["color_hex"] = unique_cols[idx]
      if self.parent_plotter and hasattr(self.parent_plotter, 'nombres_musculos_hw'):
        if hw_idx < len(self.parent_plotter.nombres_musculos_hw):
          self.parent_plotter.nombres_musculos_hw[hw_idx] = m_text
    
    self.config_mgr.config["canales"] = canales_conf
    self.config_mgr.save()
    if self.parent_plotter and hasattr(self.parent_plotter, 'refrescar_interfaz_canales'):
      self.parent_plotter.refrescar_interfaz_canales()
    self.accept()

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
def main():
  # Inicia la GUI
  app = QtWidgets.QApplication.instance()
  if not app:
    app = QtWidgets.QApplication(sys.argv)
  app.setStyleSheet(CYBERPUNK_QSS)

  gui = RealTimePlotter()
  if global_splash:
    global_splash.finish(gui)
  gui.show()
  
  # Inicia el bucle de la aplicación (bloqueante)
  exit_code = app.exec()
  sys.exit(exit_code)

if __name__ == '__main__':
  main()
  # hilo.join() ya no es necesario aquí, se maneja en on_start_acq_click y closeEvent
  
  print("Programa finalizado.")