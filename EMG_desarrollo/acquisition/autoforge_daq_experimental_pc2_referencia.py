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
__version__ = "5.0.0"

import numpy as np
from scipy.io.wavfile import write as write_wav
from datetime import datetime
import time
import queue
import threading
import sys
import os

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config_manager import ConfigManager
from utils.path_utils import get_project_root

METRONOME_CONFIG_PATH = os.path.join(get_project_root(), 'metronome_config.json')
PALABRAS_TXT_PATH = os.path.join(get_project_root(), 'palabras.txt')

import json
import re
import subprocess
from typing import Optional, Dict, List, Any, Union
try:
  import matplotlib
  matplotlib.use('Agg')
except Exception:
  pass

try:
  from EMG_desarrollo.acquisition.fotometria_webcam import WebcamPhotometryManager
except ImportError:
  try:
    from acquisition.fotometria_webcam import WebcamPhotometryManager
  except ImportError:
    try:
      from fotometria_webcam import WebcamPhotometryManager
    except ImportError:
      WebcamPhotometryManager = None

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

try:
  from utils.sound_utils import play_beep
except ImportError:
  def play_beep(freq=1000, duration_ms=100, async_play=True): pass

if global_progress: global_progress.setValue(30); global_splash.showMessage("Cargando PyQtGraph...", Qt.AlignBottom | Qt.AlignCenter, QColor("white")); app.processEvents()
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui 

# NUEVO: Import para SpinBox y Línea
from pyqtgraph import SpinBox, InfiniteLine


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

try:
    from numba import jit
except Exception:
    def jit(*args, **kwargs):
        def decorator(fn):
            return fn
        return decorator

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

def calcular_ruido_linea_50hz(noise_signal_uV: np.ndarray, fs: float) -> float:
    """
    Calcula el voltaje RMS en microvoltios (uV) de la componente de 50 Hz de la red eléctrica
    en un segmento de señal en reposo.
    """
    if noise_signal_uV is None or len(noise_signal_uV) < int(fs * 0.2):
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
    active_channel_indices (list, optional): Índices físicos activos (ej. [1, 3]).

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  print(f"Iniciando hilo de simulación con SR={sample_rate} Hz, Tipo={tipo_prueba}, Canales físicos={active_channel_indices}...")
  
  datos_archivo = None
  total_samples_archivo = 0
  
  if "Archivo" in tipo_prueba:
    try:
      import pandas as pd
      import os
      script_dir = os.path.dirname(os.path.abspath(__file__))
      # --- MODIFICADO: Usar señal trenzada de prueba ---
      test_dir = os.path.join(script_dir, "base_de_datos_electrodos", "2026-05-18", "_E0_1_TRENZADOMALLADOGND_Sujeto1")
      csv_path = os.path.join(test_dir, "grabacion.csv")
      
      if os.path.exists(csv_path):
        print(f"Cargando archivo de prueba {csv_path}...")
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
          print(f"Archivo de prueba cargado correctamente ({total_samples_archivo} muestras).")
        else:
          print("El CSV no tiene columnas de 'Canal'. Usando Senoidal.")
      else:
        print(f"Archivo no encontrado en {csv_path}. Asegúrate de crearlo. Usando Senoidal.")
    except Exception as e:
      print(f"Error al cargar el archivo de prueba: {e}. Usando Senoidal.")

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
    canales_daq (list, optional): Nombres de canales DAQ (ej: ['Dev1/ai1', 'Dev1/ai3']).

  Returns:
    Any: Resultado de la ejecución de la función.
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
  
  if meta_file:
    try:
      with open(meta_file, 'r', encoding='utf-8') as f:
        md = json.load(f)
        bpm = md.get("bpm")
        noise_seconds = md.get("noise_seconds", 5.0)
        muscles_map = md.get("muscles_map", {})
        muscles_list = md.get("muscles", [])
    except Exception as e_meta:
      print(f"  [Aviso] Error leyendo metadatos: {e_meta}")

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

    # Guías del metrónomo
    if bpm and noise_seconds is not None and noise_seconds > 0:
      tau = 60.0 / bpm
      win_color = '#ffffff'
      beat_color = '#ffaa00'
      first_bound = noise_seconds - tau / 2.0
      if t_min <= first_bound <= t_max:
        ax.axvline(x=first_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
      k = 0
      while True:
        t_beat = noise_seconds + k * tau
        t_bound = t_beat + tau / 2.0
        if t_min <= t_beat <= t_max:
          ax.axvline(x=t_beat, color=beat_color, ls=':', lw=0.9, alpha=0.35)
        if t_bound > t_max:
          break
        if t_bound >= t_min:
          ax.axvline(x=t_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
        k += 1

    # Detección y marcado de picos de contracción en señal cruda
    if not is_mic:
      picos_t, picos_y = [], []
      t_limite_ruido = float(noise_seconds) if (noise_seconds is not None and noise_seconds > 0) else 0.0
      if bpm and noise_seconds is not None and noise_seconds > 0:
        tau = 60.0 / bpm
        k_p = 1
        while True:
          t_beat_k = noise_seconds + k_p * tau
          t_w_start = t_beat_k - tau / 2.0
          t_w_end = t_beat_k + tau / 2.0
          if t_w_start > t_max:
            break
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
          k_p += 1
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
    tit = f"ai{hw_ch} - {musculo} (Cruda)" if not is_mic else f"ai{hw_ch} - {musculo}"
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

    daq_ch_name = canales_daq[i] if (canales_daq and i < len(canales_daq)) else f"CH{i}"
    m_ai = re.search(r"ai(\d+)", str(daq_ch_name), re.IGNORECASE)
    hw_idx = int(m_ai.group(1)) if m_ai else i

    # Subplot Izquierdo: SNR
    axs[i, 0].plot(t_plot, snr_plot, marker='o', linestyle='-', color='b', linewidth=2)
    axs[i, 0].set_title(f"ai{hw_idx} ({daq_ch_name}) - Evolución SNR Promedio")
    axs[i, 0].set_xlabel("Tiempo de Señal (s)")
    axs[i, 0].set_ylabel("SNR Promedio Acumulado")
    axs[i, 0].grid(True, alpha=0.5)

    # Subplot Derecho: Ruido
    axs[i, 1].plot(t_plot, noise_mean_plot, marker='o', linestyle='-', color='r', label='Promedio (x̄)', linewidth=2)
    axs[i, 1].plot(t_plot, noise_std_plot, marker='x', linestyle='--', color='orange', label='Desviación (σ)', linewidth=2)
    axs[i, 1].set_title(f"ai{hw_idx} ({daq_ch_name}) - Evolución Ruido Inter-pulso")
    axs[i, 1].set_xlabel("Tiempo de Señal (s)")
    axs[i, 1].set_ylabel("Amplitud (µV)")
    axs[i, 1].legend()
    axs[i, 1].grid(True, alpha=0.5)

  nombre_archivo = os.path.join(output_dir, f"{base_name}.png")
  try:
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(nombre_archivo, dpi=200)
    print(f"  Gráfico de estadísticas guardado como: {nombre_archivo}")
    return True
  except Exception as e:
    print(f"  Error al guardar gráfico de estadísticas: {e}")
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
  Permite agregar Notas/Comentarios tanto en mediciones de prueba como formales.
  """
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Guardar Medición")
    self.setMinimumWidth(420)
    self.layout = QtWidgets.QVBoxLayout(self)
    self.measurement_name = ""
    self.sujeto = ""
    self.letra = ""
    self.prueba = ""
    self.es_formal = False
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
    self.formal_group.setLayout(self.formal_layout)
    self.layout.addWidget(self.formal_group)

    # Grupo de Notas / Comentario (siempre habilitado para Prueba y Formal)
    self.notes_group = QtWidgets.QGroupBox("Notas / Comentario del Experimento")
    self.notes_layout = QtWidgets.QVBoxLayout(self.notes_group)
    self.edit_comentario = QtWidgets.QPlainTextEdit()
    self.edit_comentario.setPlaceholderText("Posición de electrodos, preparación de piel, observaciones sobre /O/ vs /U/, etc.")
    self.edit_comentario.setFixedHeight(80)
    if parent is not None and hasattr(parent, 'txt_notas_live'):
      self.edit_comentario.setPlainText(parent.txt_notas_live.toPlainText().strip())
    self.notes_layout.addWidget(self.edit_comentario)
    self.layout.addWidget(self.notes_group)

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
    self.comentario = self.edit_comentario.toPlainText().strip()
    if self.parent() is not None:
      p = self.parent()
      if hasattr(p, 'txt_notas_live') and self.comentario:
        p.txt_notas_live.setPlainText(self.comentario)
      p.notas_experimento = self.comentario

    if self.radio_formal.isChecked():
      self.es_formal = True
      self.letra = self.edit_letra.text().strip()
      self.prueba = self.edit_prueba.text().strip()
      self.sujeto = self.edit_sujeto.text().strip()
      self.measurement_name = f"{self.letra}_{self.prueba}_{self.sujeto}"
    else:
      self.es_formal = False
      self.measurement_name = f"prueba_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
      self.sujeto = "prueba"
      self.letra = ""
      self.prueba = ""
    self.accept()


class AutoForgeDialog(QtWidgets.QDialog):
  """
  Clase AutoForgeDialog.

  Representa y gestiona las operaciones relacionadas con AutoForgeDialog.
  """
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Ñandú LSD - Configuración de Autograbado")
    self.setMinimumWidth(440)
    self.setStyleSheet("""
      QDialog { background-color: #0d0d0d; color: #00ffff; font-family: 'Courier New', monospace; }
      QLabel { color: #00ffcc; font-weight: bold; }
      QLineEdit, QSpinBox, QComboBox, QPlainTextEdit { background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc; padding: 4px; border-radius: 3px; }
      QGroupBox { border: 1px solid #ff00ff; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #ff00ff; }
      QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
    """)
    
    self.layout = QtWidgets.QFormLayout(self)
    
    self.edit_prueba = QtWidgets.QLineEdit(getattr(parent, 'autoforge_prueba', "Prueba1") or "Prueba1")
    self.edit_sujeto = QtWidgets.QLineEdit(getattr(parent, 'autoforge_sujeto', "Sujeto1") or "Sujeto1")
    self.spin_reps = QtWidgets.QSpinBox()
    self.spin_reps.setRange(1, 1000)
    self.spin_reps.setValue(5)
    self.spin_bpm = QtWidgets.QSpinBox()
    self.spin_bpm.setRange(30, 300)
    self.spin_bpm.setValue(40)
    
    self.layout.addRow("Prueba:", self.edit_prueba)
    self.layout.addRow("Sujeto:", self.edit_sujeto)
    self.layout.addRow("Repeticiones:", self.spin_reps)
    self.layout.addRow("BPM Metrónomo:", self.spin_bpm)
    
    self.chk_calibrar_cvm = QtWidgets.QCheckBox("Calibrar Contracción Máxima (A, I, U)")
    self.chk_calibrar_cvm.setChecked(True)
    self.chk_calibrar_cvm.setStyleSheet("color: #00ffcc; font-weight: bold;")
    self.layout.addRow("Calibración CVM:", self.chk_calibrar_cvm)

    # --- Fotometría Webcam 3D ---
    photo_mgr = getattr(parent, 'photometry_mgr', None)
    self.chk_fotometria = QtWidgets.QCheckBox("Habilitar Fotometría Automática y Asistente Facial 3D (Webcam)")
    self.chk_fotometria.setChecked(bool(photo_mgr and getattr(photo_mgr, 'is_enabled', False)))
    self.chk_fotometria.setStyleSheet("color: #00e5ff; font-weight: bold;")
    self.layout.addRow("Fotometría 3D:", self.chk_fotometria)

    self.chk_fotometria_video = QtWidgets.QCheckBox("Grabar también clip de video (.mp4) por toma")
    self.chk_fotometria_video.setChecked(bool(photo_mgr and (getattr(photo_mgr, 'record_video_clips', False) or getattr(photo_mgr, 'record_video_clip', False))))
    self.chk_fotometria_video.setStyleSheet("color: #aaddff;")
    self.layout.addRow("Video por Toma:", self.chk_fotometria_video)
    
    self.btn_edit_words = QtWidgets.QPushButton(" Editar Palabras")
    self.btn_edit_words.setStyleSheet("background-color: #333333; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 5px; border: 2px solid #555555; border-radius: 4px;")
    self.btn_edit_words.clicked.connect(self.abrir_editor_palabras)
    self.layout.addRow("", self.btn_edit_words)

    # --- Asignación de Músculos ---
    self.group_muscles = QtWidgets.QGroupBox("Asignación de Músculos (Canales Físicos Activos)")
    self.layout_muscles = QtWidgets.QFormLayout(self.group_muscles)
    self.muscle_inputs = []
    
    active_indices = getattr(parent, 'active_channel_indices', None)
    if not active_indices:
      num_chans = getattr(parent, 'NUM_CANALES', 4) or 4
      active_indices = list(range(num_chans))
    else:
      num_chans = len(active_indices)
    parent_muscles = getattr(parent, 'nombres_musculos', [])
    parent_muscles_hw = getattr(parent, 'nombres_musculos_hw', [])
    
    musculos_sugeridos = list(STANDARD_MUSCLES_LIST)
    
    for i in range(num_chans):
      hw_idx = active_indices[i] if i < len(active_indices) else i
      if i < len(parent_muscles) and parent_muscles[i]:
        default_val = parent_muscles[i]
      elif hw_idx < len(parent_muscles_hw) and parent_muscles_hw[hw_idx]:
        default_val = parent_muscles_hw[hw_idx]
      else:
        default_val = f"Canal {hw_idx}"
      cmb = QtWidgets.QComboBox()
      cmb.setEditable(True)
      for m in musculos_sugeridos:
        cmb.addItem(m)
      if default_val not in musculos_sugeridos:
        cmb.insertItem(0, default_val)
      cmb.setCurrentText(default_val)
      
      lbl = QtWidgets.QLabel(f"ai{hw_idx} (Canal {hw_idx}):")
      self.layout_muscles.addRow(lbl, cmb)
      self.muscle_inputs.append(cmb)
      
    self.layout.addRow(self.group_muscles)

    # --- Notas del Experimento ---
    self.edit_notas = QtWidgets.QPlainTextEdit()
    self.edit_notas.setPlaceholderText("Observaciones sobre posición de electrodos, piel/gel, separabilidad /O/-/U/, ruido 50Hz...")
    self.edit_notas.setFixedHeight(68)
    if parent is not None and hasattr(parent, 'txt_notas_live'):
      self.edit_notas.setPlainText(parent.txt_notas_live.toPlainText().strip())
    self.layout.addRow("Notas Experimento:", self.edit_notas)
    
    self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    self.buttonBox.accepted.connect(self.accept)
    self.buttonBox.rejected.connect(self.reject)
    self.layout.addWidget(self.buttonBox)

  def get_muscle_names(self):
    return [inp.currentText().strip() for inp in self.muscle_inputs]

  def abrir_editor_palabras(self):
    ruta_palabras = PALABRAS_TXT_PATH
    if not os.path.exists(ruta_palabras):
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("A\nE\nI\nO\nU\n")
    
    dialog = QtWidgets.QDialog(self)
    dialog.setWindowTitle("Editor de Palabras (AutoForge)")
    dialog.resize(300, 400)
    dialog_layout = QtWidgets.QVBoxLayout(dialog)
    
    lbl = QtWidgets.QLabel("Escribe una palabra por línea:")
    dialog_layout.addWidget(lbl)
    
    text_edit = QtWidgets.QTextEdit()
    with open(ruta_palabras, 'r', encoding='utf-8') as f:
      text_edit.setPlainText(f.read())
    dialog_layout.addWidget(text_edit)
    
    btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
    btn_box.accepted.connect(dialog.accept)
    btn_box.rejected.connect(dialog.reject)
    dialog_layout.addWidget(btn_box)
    
    if dialog.exec() == QtWidgets.QDialog.Accepted:
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        texto_guardar = text_edit.toPlainText().strip() + "\n"
        f.write(texto_guardar)
      QtWidgets.QMessageBox.information(self, "Guardado", "Lista de palabras guardada con éxito.")


class TestSeparabilidadVocalesDialog(QtWidgets.QDialog):
  """
  Tester interactivo multiclase de separabilidad biomecánica/eléctrica para las 5 vocales
  (A, E, I, O, U) con múltiples repeticiones guiadas por metrónomo, proyección 2D PCA,
  matrices 5x5 (ángulo direccional, distancia de Fisher, matriz de confusión validada Leave-One-Out)
  y detector de Crosstalk por correlación cruzada a lag cero.
  """
  def __init__(self, parent=None):
    super().__init__(parent)
    self.parent_plotter = parent
    self.setWindowTitle("Test de Separabilidad Multiclase 5 Vocales (A, E, I, O, U) y Crosstalk")
    self.resize(1040, 780)
    self.setStyleSheet("""
      QDialog { background-color: #08080c; color: #e0f7fa; font-family: 'Consolas', 'Courier New', monospace; }
      QLabel { color: #00ffcc; font-size: 12px; }
      QGroupBox { border: 1.5px solid #00e5ff; border-radius: 5px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #00e5ff; }
      QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
      QPushButton { background-color: #141424; color: #00ffcc; border: 1.5px solid #00ffcc; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; }
      QPushButton:hover { background-color: #00ffcc; color: #000000; }
      QPushButton:disabled { background-color: #101015; color: #555555; border-color: #333344; }
      QTextEdit { background-color: #050508; color: #ffffff; border: 1px solid #333344; font-family: 'Consolas', monospace; font-size: 11px; }
      QCheckBox { color: #00ffcc; font-weight: bold; spacing: 5px; font-size: 12px; }
      QSpinBox, QComboBox { background-color: #101018; color: #00ffcc; border: 1px solid #00e5ff; padding: 4px; border-radius: 3px; font-weight: bold; }
    """)

    self.VOCAL_COLORS = {
      'A': '#00ffff',
      'E': '#ff00ff',
      'I': '#ffff00',
      'O': '#ff8800',
      'U': '#00ff66',
    }
    self.vocales_list = ['A', 'E', 'I', 'O', 'U']
    self.samples = []  # Lista de dicts: {'vocal': str, 'x': np.ndarray, 'raw': np.ndarray, 'rep': int}
    self.is_protocol_running = False
    self.protocol_steps = []
    self.step_idx = 0
    self.protocol_timer = QtCore.QTimer(self)
    self.protocol_timer.timeout.connect(self._avanzar_paso_protocolo)
    self.last_metrics = None
    self.last_report_text = ""

    main_layout = QtWidgets.QVBoxLayout(self)

    # Introducción
    lbl_intro = QtWidgets.QLabel(
      "<b>Evaluación Multiclase de Ortogonalidad y Crosstalk para 5 Vocales (A, E, I, O, U):</b><br>"
      "Evalúa separabilidad global y por pares (vigilando O-U, A-E, E-I, A-I) con múltiples repeticiones "
      "para diagnosticar el posicionamiento de los electrodos y evitar confusiones cruzadas."
    )
    lbl_intro.setWordWrap(True)
    lbl_intro.setStyleSheet("color: #ffffff; font-size: 12px; padding: 4px;")
    main_layout.addWidget(lbl_intro)

    # Splitter central: Izquierda (Controles + Diagnóstico + Reporte) | Derecha (Proyección 2D PCA)
    splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    left_widget = QtWidgets.QWidget()
    left_layout = QtWidgets.QVBoxLayout(left_widget)
    left_layout.setContentsMargins(0, 0, 4, 0)

    # --- Grupo Configuración ---
    grp_cfg = QtWidgets.QGroupBox("1. Vocales y Configuración de Repeticiones")
    cfg_layout = QtWidgets.QVBoxLayout(grp_cfg)

    # Checkboxes de vocales (las 5 activas por defecto)
    chk_row = QtWidgets.QHBoxLayout()
    chk_row.addWidget(QtWidgets.QLabel("Vocales activas:"))
    self.vocal_checkboxes = {}
    for v in self.vocales_list:
      cb = QtWidgets.QCheckBox(f"/{v}/")
      cb.setChecked(True)
      col = self.VOCAL_COLORS.get(v, "#00ffcc")
      cb.setStyleSheet(f"color: {col}; font-weight: bold;")
      self.vocal_checkboxes[v] = cb
      chk_row.addWidget(cb)
    chk_row.addStretch(1)
    cfg_layout.addLayout(chk_row)

    # Parámetros: repeticiones y modo
    params_row = QtWidgets.QHBoxLayout()
    params_row.addWidget(QtWidgets.QLabel("Reps / Vocal:"))
    self.spin_reps = QtWidgets.QSpinBox()
    self.spin_reps.setRange(2, 50)
    self.spin_reps.setValue(12)  # Por defecto 12 a 15 repeticiones
    self.spin_reps.setFixedWidth(65)
    params_row.addWidget(self.spin_reps)

    params_row.addSpacing(10)
    params_row.addWidget(QtWidgets.QLabel("Modo:"))
    self.cmb_mode = QtWidgets.QComboBox()
    self.cmb_mode.addItems(["Alternada (Anti-fatiga)", "Por Bloques"])
    params_row.addWidget(self.cmb_mode)
    params_row.addStretch(1)
    cfg_layout.addLayout(params_row)

    # Botones de inicio de protocolo guiado y control de datos
    btn_ctrl_row = QtWidgets.QHBoxLayout()
    self.btn_iniciar = QtWidgets.QPushButton("Iniciar Protocolo Guiado")
    self.btn_iniciar.clicked.connect(self._toggle_protocolo_guiado)
    self.btn_sumar_tanda = QtWidgets.QPushButton("+ Sumar Tanda (+5 Reps)")
    self.btn_sumar_tanda.clicked.connect(self._sumar_tanda)
    self.btn_limpiar = QtWidgets.QPushButton("Limpiar Datos")
    self.btn_limpiar.clicked.connect(self._limpiar_datos)
    btn_ctrl_row.addWidget(self.btn_iniciar)
    btn_ctrl_row.addWidget(self.btn_sumar_tanda)
    btn_ctrl_row.addWidget(self.btn_limpiar)
    cfg_layout.addLayout(btn_ctrl_row)

    # Botones de captura manual rápida individual
    man_row = QtWidgets.QHBoxLayout()
    man_row.addWidget(QtWidgets.QLabel("Captura Rápida Individual:"))
    for v in self.vocales_list:
      b = QtWidgets.QPushButton(f"+ /{v}/")
      col = self.VOCAL_COLORS.get(v, "#00ffcc")
      b.setStyleSheet(f"QPushButton {{ background-color: #111122; color: {col}; border: 1px solid {col}; padding: 3px 8px; font-weight: bold; font-size: 11px; }} QPushButton:hover {{ background-color: {col}; color: #000000; }}")
      b.clicked.connect(lambda _, voc=v: self._capturar_muestra_manual(voc))
      man_row.addWidget(b)
    man_row.addStretch(1)
    cfg_layout.addLayout(man_row)

    left_layout.addWidget(grp_cfg)

    # --- Estado y Barra de Progreso ---
    self.lbl_status = QtWidgets.QLabel("Estado: Esperando inicio de protocolo...")
    self.lbl_status.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_status.setStyleSheet("background-color: #111118; color: #ffaa00; font-size: 13px; font-weight: bold; padding: 6px; border: 1px solid #ffaa00; border-radius: 4px;")
    left_layout.addWidget(self.lbl_status)

    self.prog_bar = QtWidgets.QProgressBar()
    self.prog_bar.setRange(0, 100)
    self.prog_bar.setValue(0)
    self.prog_bar.setFixedHeight(16)
    self.prog_bar.setStyleSheet("""
      QProgressBar { border: 1px solid #333344; border-radius: 3px; text-align: center; color: white; background-color: #05050a; font-size: 10px; font-weight: bold; }
      QProgressBar::chunk { background-color: #00e5ff; }
    """)
    left_layout.addWidget(self.prog_bar)

    # --- Semáforo de Diagnóstico ---
    self.lbl_semaforo = QtWidgets.QLabel("SEMAFORO DE DIAGNOSTICO: Pendiente de datos suficientes")
    self.lbl_semaforo.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_semaforo.setWordWrap(True)
    self.lbl_semaforo.setStyleSheet("background-color: #151515; color: #aaaaaa; font-size: 14px; font-weight: bold; padding: 8px; border: 2px solid #444444; border-radius: 5px;")
    left_layout.addWidget(self.lbl_semaforo)

    # --- Reporte Textual Consolidado ---
    self.txt_resultados = QtWidgets.QTextEdit()
    self.txt_resultados.setReadOnly(True)
    left_layout.addWidget(self.txt_resultados, stretch=1)

    splitter.addWidget(left_widget)

    # --- Lado Derecho: Proyección 2D PCA de las 5 Nubes Vocálicas ---
    right_widget = QtWidgets.QWidget()
    right_layout = QtWidgets.QVBoxLayout(right_widget)
    right_layout.setContentsMargins(4, 0, 0, 0)

    grp_plot = QtWidgets.QGroupBox("2. Proyeccion 2D PCA - Separacion de Nubes Vocalicas")
    plot_box_layout = QtWidgets.QVBoxLayout(grp_plot)

    self.plot_widget = pg.PlotWidget()
    self.plot_widget.setBackground('#05050a')
    self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
    self.plot_widget.setLabel('bottom', 'Componente Principal 1 (PC1)')
    self.plot_widget.setLabel('left', 'Componente Principal 2 (PC2)')
    plot_box_layout.addWidget(self.plot_widget, stretch=1)

    self.lbl_pca_info = QtWidgets.QLabel("Varianza explicada: PC1 = --%, PC2 = --% | Total Muestras = 0")
    self.lbl_pca_info.setStyleSheet("color: #aaddff; font-size: 11px;")
    plot_box_layout.addWidget(self.lbl_pca_info)

    right_layout.addWidget(grp_plot)
    splitter.addWidget(right_widget)
    splitter.setSizes([560, 460])
    main_layout.addWidget(splitter, stretch=1)

    # --- Fila Inferior de Acciones ---
    bottom_row = QtWidgets.QHBoxLayout()
    self.btn_anexar = QtWidgets.QPushButton("Anexar Diagnostico a Notas y Metadata")
    self.btn_anexar.setEnabled(False)
    self.btn_anexar.setStyleSheet("background-color: #002b22; color: #00ffcc; border: 2px solid #00ffcc; padding: 8px 16px; font-weight: bold; font-size: 13px;")
    self.btn_anexar.clicked.connect(self._anexar_a_notas)

    btn_cerrar = QtWidgets.QPushButton("Cerrar")
    btn_cerrar.clicked.connect(self.close)

    bottom_row.addWidget(self.btn_anexar)
    bottom_row.addStretch()
    bottom_row.addWidget(btn_cerrar)
    main_layout.addLayout(bottom_row)

  def _limpiar_datos(self):
    """Reinicia todas las muestras capturadas y el visor gráfico."""
    if self.is_protocol_running:
      self._toggle_protocolo_guiado()
    self.samples.clear()
    self.plot_widget.clear()
    self.lbl_pca_info.setText("Varianza explicada: PC1 = --%, PC2 = --% | Total Muestras = 0")
    self.prog_bar.setValue(0)
    self.lbl_status.setText("Estado: Datos reiniciados. Listo para iniciar nueva toma.")
    self.lbl_status.setStyleSheet("background-color: #111118; color: #ffaa00; font-size: 13px; font-weight: bold; padding: 6px; border: 1px solid #ffaa00; border-radius: 4px;")
    self.lbl_semaforo.setText("SEMAFORO DE DIAGNOSTICO: Pendiente de datos suficientes")
    self.lbl_semaforo.setStyleSheet("background-color: #151515; color: #aaaaaa; font-size: 14px; font-weight: bold; padding: 8px; border: 2px solid #444444; border-radius: 5px;")
    self.txt_resultados.clear()
    self.btn_anexar.setEnabled(False)

  def _sumar_tanda(self):
    """Suma 5 repeticiones adicionales a las vocales activas."""
    if self.is_protocol_running:
      return
    vocales_activas = [v for v in self.vocales_list if self.vocal_checkboxes[v].isChecked()]
    if len(vocales_activas) < 2:
      QtWidgets.QMessageBox.warning(self, "Selección Insuficiente", "Seleccione al menos 2 vocales.")
      return
    self._iniciar_secuencia_protocolo(vocales_activas, n_reps=5, append_mode=True)

  def _toggle_protocolo_guiado(self):
    """Inicia o detiene el protocolo guiado por metrónomo."""
    if self.is_protocol_running:
      self.protocol_timer.stop()
      self.is_protocol_running = False
      self.btn_iniciar.setText("Iniciar Protocolo Guiado")
      self.btn_iniciar.setStyleSheet("background-color: #141424; color: #00ffcc; border: 1.5px solid #00ffcc; padding: 6px 12px; font-weight: bold;")
      self.lbl_status.setText("Estado: Protocolo detenido por el usuario.")
      self.lbl_status.setStyleSheet("background-color: #221100; color: #ffaa00; font-size: 13px; font-weight: bold; padding: 6px; border: 1px solid #ffaa00; border-radius: 4px;")
      if len(self.samples) >= 2:
        self._calcular_metricas_multiclase()
      return

    p = self.parent_plotter
    if p is None or not getattr(p, 'is_acquiring', False) or p.plot_buffer_datos is None:
      QtWidgets.QMessageBox.warning(
        self, "Adquisición Inactiva",
        "Inicie primero la adquisición ('Iniciar Adquisición') en la ventana principal para capturar sEMG."
      )
      return

    vocales_activas = [v for v in self.vocales_list if self.vocal_checkboxes[v].isChecked()]
    if len(vocales_activas) < 2:
      QtWidgets.QMessageBox.warning(
        self, "Selección Insuficiente",
        "Seleccione al menos 2 vocales (por defecto las 5 vocales A, E, I, O, U) para evaluar separabilidad."
      )
      return

    n_reps = self.spin_reps.value()
    self._iniciar_secuencia_protocolo(vocales_activas, n_reps=n_reps, append_mode=False)

  def _iniciar_secuencia_protocolo(self, vocales_activas, n_reps=12, append_mode=False):
    if not append_mode:
      self.samples.clear()

    sequence = []
    if self.cmb_mode.currentText().startswith("Alternada"):
      for r in range(1, n_reps + 1):
        for v in vocales_activas:
          sequence.append((v, r, n_reps))
    else:
      for v in vocales_activas:
        for r in range(1, n_reps + 1):
          sequence.append((v, r, n_reps))

    self.protocol_steps = []
    # Cuenta regresiva inicial de 3 segundos
    self.protocol_steps.append({'phase': 'countdown', 'count': 3, 'dur_ms': 1000})
    self.protocol_steps.append({'phase': 'countdown', 'count': 2, 'dur_ms': 1000})
    self.protocol_steps.append({'phase': 'countdown', 'count': 1, 'dur_ms': 1000})

    for v, r, tot in sequence:
      self.protocol_steps.append({'phase': 'prep', 'vocal': v, 'rep': r, 'total': tot, 'dur_ms': 1000})
      self.protocol_steps.append({'phase': 'action', 'vocal': v, 'rep': r, 'total': tot, 'dur_ms': 1500})

    self.step_idx = 0
    self.is_protocol_running = True
    self.btn_iniciar.setText("Detener Protocolo")
    self.btn_iniciar.setStyleSheet("background-color: #3b0011; color: #ff4466; border: 1.5px solid #ff4466; padding: 6px 12px; font-weight: bold;")
    self._avanzar_paso_protocolo()

  def _avanzar_paso_protocolo(self):
    if not self.is_protocol_running:
      return

    if self.step_idx >= len(self.protocol_steps):
      self.is_protocol_running = False
      self.btn_iniciar.setText("Iniciar Protocolo Guiado")
      self.btn_iniciar.setStyleSheet("background-color: #141424; color: #00ffcc; border: 1.5px solid #00ffcc; padding: 6px 12px; font-weight: bold;")
      self.prog_bar.setValue(100)
      self.lbl_status.setText("Estado: Protocolo completado. Calculando métricas multiclase...")
      self.lbl_status.setStyleSheet("background-color: #002211; color: #00ffcc; font-size: 13px; font-weight: bold; padding: 6px; border: 1px solid #00ffcc; border-radius: 4px;")
      self._calcular_metricas_multiclase()
      return

    step = self.protocol_steps[self.step_idx]
    phase = step['phase']
    dur_ms = step['dur_ms']

    p = self.parent_plotter
    action_steps_total = sum(1 for s in self.protocol_steps if s['phase'] == 'action')
    action_steps_done = sum(1 for s in self.protocol_steps[:self.step_idx] if s['phase'] == 'action')
    pct = int((action_steps_done / max(1, action_steps_total)) * 100)
    self.prog_bar.setValue(pct)

    if phase == 'countdown':
      cnt = step['count']
      self.lbl_status.setText(f"[PREPARACION] Iniciando protocolo en {cnt} segundos...")
      self.lbl_status.setStyleSheet("background-color: #1a1a2e; color: #00ffff; font-size: 14px; font-weight: bold; padding: 6px; border: 1.5px solid #00ffff; border-radius: 4px;")
      if p and hasattr(p, 'beep_queue'):
        p.beep_queue.put((800, 100))
    elif phase == 'prep':
      v = step['vocal']
      r = step['rep']
      tot = step['total']
      self.lbl_status.setText(f"[REPOSO] Prepárese para pronunciar: /{v}/  (Repetición {r}/{tot})")
      self.lbl_status.setStyleSheet("background-color: #2b2200; color: #ffdd44; font-size: 14px; font-weight: bold; padding: 6px; border: 1.5px solid #ffdd44; border-radius: 4px;")
      if p and hasattr(p, 'beep_queue'):
        p.beep_queue.put((900, 100))
    elif phase == 'action':
      v = step['vocal']
      r = step['rep']
      tot = step['total']
      col = self.VOCAL_COLORS.get(v, '#00ffcc')
      self.lbl_status.setText(f"[PRONUNCIE AHORA] /{v}/  (Repetición {r}/{tot})")
      self.lbl_status.setStyleSheet(f"background-color: #002222; color: {col}; font-size: 15px; font-weight: bold; padding: 6px; border: 2px solid {col}; border-radius: 4px;")
      if p and hasattr(p, 'beep_queue'):
        p.beep_queue.put((1400, 150))

      # Al finalizar el intervalo de acción (1.5s), capturar el segmento de la ventana
      QtCore.QTimer.singleShot(dur_ms - 50, lambda voc=v, rep_i=r: self._guardar_muestra_actual(voc, rep_i))

    self.step_idx += 1
    self.protocol_timer.singleShot(dur_ms, self._avanzar_paso_protocolo)

  def _guardar_muestra_actual(self, vocal: str, rep_idx: int):
    p = self.parent_plotter
    if p is None or p.plot_buffer_datos is None or p.env_buffer_datos is None:
      return

    emg_indices = [ch for ch in range(p.NUM_CANALES) if not p._is_mic_channel(ch)]
    if not emg_indices:
      emg_indices = list(range(p.NUM_CANALES))

    n_samples = min(int(1.5 * max(1, p.SAMPLE_RATE)), p.plot_buffer_datos.shape[1])
    raw_seg = np.nan_to_num(p.plot_buffer_datos[emg_indices, -n_samples:].copy())
    env_seg = np.nan_to_num(p.env_buffer_datos[emg_indices, -n_samples:].copy())
    feat = np.sqrt(np.mean(env_seg ** 2, axis=1))

    self.samples.append({
      'vocal': vocal,
      'x': feat,
      'raw': raw_seg,
      'rep': rep_idx
    })
    self._actualizar_proyeccion_pca()

  def _capturar_muestra_manual(self, vocal: str):
    """Captura inmediata de 1.5s de señal para una vocal específica."""
    p = self.parent_plotter
    if p is None or not getattr(p, 'is_acquiring', False) or p.plot_buffer_datos is None:
      QtWidgets.QMessageBox.warning(self, "Adquisición Inactiva", "Inicie primero la adquisición.")
      return

    emg_indices = [ch for ch in range(p.NUM_CANALES) if not p._is_mic_channel(ch)]
    if not emg_indices:
      emg_indices = list(range(p.NUM_CANALES))

    n_samples = min(int(1.5 * max(1, p.SAMPLE_RATE)), p.plot_buffer_datos.shape[1])
    raw_seg = np.nan_to_num(p.plot_buffer_datos[emg_indices, -n_samples:].copy())
    env_seg = np.nan_to_num(p.env_buffer_datos[emg_indices, -n_samples:].copy())
    feat = np.sqrt(np.mean(env_seg ** 2, axis=1))

    count_v = sum(1 for s in self.samples if s['vocal'] == vocal) + 1
    self.samples.append({
      'vocal': vocal,
      'x': feat,
      'raw': raw_seg,
      'rep': count_v
    })
    self.lbl_status.setText(f"Estado: Muestra individual /{vocal}/ capturada (Total muestras = {len(self.samples)}).")
    self.lbl_status.setStyleSheet("background-color: #002211; color: #00ffcc; font-size: 13px; font-weight: bold; padding: 6px; border: 1px solid #00ffcc; border-radius: 4px;")
    self._actualizar_proyeccion_pca()
    if len(self.samples) >= 2:
      self._calcular_metricas_multiclase()

  def _actualizar_proyeccion_pca(self):
    """Proyecta las muestras en un espacio 2D mediante PCA y las dibuja interactivamente."""
    if len(self.samples) < 2:
      return

    X = np.array([s['x'] for s in self.samples])
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean

    if X.shape[1] >= 2 and X.shape[0] >= 2:
      try:
        _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        Z = np.dot(X_c, Vt[:2].T)
        s2 = S ** 2
        tot_s2 = np.sum(s2) + 1e-12
        var_pc1 = float(s2[0] / tot_s2 * 100.0)
        var_pc2 = float(s2[1] / tot_s2 * 100.0) if len(s2) > 1 else 0.0
      except Exception:
        Z = np.zeros((len(self.samples), 2))
        Z[:, 0] = X_c[:, 0]
        var_pc1, var_pc2 = 100.0, 0.0
    else:
      Z = np.zeros((len(self.samples), 2))
      Z[:, 0] = X_c[:, 0] if X_c.shape[1] > 0 else np.zeros(len(self.samples))
      var_pc1, var_pc2 = 100.0, 0.0

    self.plot_widget.clear()
    for v in self.vocales_list:
      v_indices = [idx for idx, s in enumerate(self.samples) if s['vocal'] == v]
      if not v_indices:
        continue
      pts_x = Z[v_indices, 0]
      pts_y = Z[v_indices, 1]
      col = self.VOCAL_COLORS.get(v, '#00ffcc')

      scatter = pg.ScatterPlotItem(
        x=pts_x, y=pts_y, pen=pg.mkPen(None),
        brush=pg.mkBrush(col), size=12, name=f"/{v}/"
      )
      self.plot_widget.addItem(scatter)

      cx = float(np.mean(pts_x))
      cy = float(np.mean(pts_y))
      star = pg.ScatterPlotItem(
        x=[cx], y=[cy], symbol='star', size=20,
        pen=pg.mkPen('w', width=1.5), brush=pg.mkBrush(col)
      )
      self.plot_widget.addItem(star)

    self.lbl_pca_info.setText(
      f"Varianza explicada: PC1 = {var_pc1:.1f}%, PC2 = {var_pc2:.1f}% | Total Muestras = {len(self.samples)}"
    )

  def _calcular_metricas_multiclase(self):
    """Calcula matrices 5x5 de ángulo, Fisher, confusión validada LOO-LDA y Crosstalk."""
    vocales_activas = [v for v in self.vocales_list if any(s['vocal'] == v for s in self.samples)]
    if len(vocales_activas) < 2:
      self.lbl_status.setText("Estado: Se requieren muestras de al menos 2 vocales distintas para calcular separabilidad.")
      return

    p = self.parent_plotter
    num_ch = p.NUM_CANALES if p else 4
    emg_indices = [ch for ch in range(num_ch) if not p._is_mic_channel(ch)] if p else list(range(num_ch))
    if not emg_indices:
      emg_indices = list(range(num_ch))
    sample_feat_len = len(self.samples[0]['x']) if self.samples else len(emg_indices)
    m_len = sample_feat_len

    ch_labels = []
    for idx_pos in range(m_len):
      if idx_pos < len(emg_indices):
        ch = emg_indices[idx_pos]
        hw_idx = p.active_channel_indices[ch] if p and hasattr(p, 'active_channel_indices') and ch < len(p.active_channel_indices) else ch
        musc = p.nombres_musculos[ch] if p and hasattr(p, 'nombres_musculos') and ch < len(p.nombres_musculos) else f"Canal {hw_idx}"
        ch_labels.append(f"ai{hw_idx} ({musc})")
      else:
        ch_labels.append(f"Canal {idx_pos}")

    # 1. Agrupar muestras por vocal y calcular vectores medios y dominancia
    mu_dict = {}
    dom_dict = {}
    var_dict = {}
    unit_mu_dict = {}
    counts = {}

    for v in vocales_activas:
      v_samples = [s['x'] for s in self.samples if s['vocal'] == v]
      arr_v = np.array(v_samples)
      counts[v] = len(v_samples)
      mu_v = np.mean(arr_v, axis=0)
      var_v = np.var(arr_v, axis=0) if len(v_samples) > 1 else np.zeros(m_len)
      norm_v = float(np.linalg.norm(mu_v))
      unit_v = (mu_v / (norm_v + 1e-12))
      sum_mu = float(np.sum(mu_v)) + 1e-12
      dom_v = (mu_v / sum_mu) * 100.0

      mu_dict[v] = mu_v
      dom_dict[v] = dom_v
      var_dict[v] = var_v
      unit_mu_dict[v] = unit_v

    # 2. Matrices multiclase K x K: Ángulos y Distancia de Fisher
    K = len(vocales_activas)
    angles = np.zeros((K, K))
    fisher_mat = np.zeros((K, K))

    for i in range(K):
      for j in range(K):
        v1 = vocales_activas[i]
        v2 = vocales_activas[j]
        if i == j:
          angles[i, j] = 0.0
          fisher_mat[i, j] = 0.0
        else:
          cos_t = float(np.clip(np.dot(unit_mu_dict[v1], unit_mu_dict[v2]), -1.0, 1.0))
          angles[i, j] = float(np.degrees(np.arccos(cos_t)))
          d_mu = mu_dict[v1] - mu_dict[v2]
          denom_f = float(np.sum(var_dict[v1] + var_dict[v2])) + 1e-6
          fisher_mat[i, j] = float(np.sum(d_mu ** 2) / denom_f)

    # 3. Matriz de Confusión Leave-One-Out (LDA Regularizado)
    X_all = np.array([s['x'] for s in self.samples])
    y_all = np.array([vocales_activas.index(s['vocal']) for s in self.samples if s['vocal'] in vocales_activas])
    N_total = len(y_all)
    conf_mat = np.zeros((K, K), dtype=int)

    if N_total >= K + 2:
      for i in range(N_total):
        train_mask = np.ones(N_total, dtype=bool)
        train_mask[i] = False
        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]

        means_tr = []
        cov_sum = np.zeros((m_len, m_len))
        for c in range(K):
          c_mask = (y_tr == c)
          if np.sum(c_mask) > 0:
            mu_c = np.mean(X_tr[c_mask], axis=0)
            diff = X_tr[c_mask] - mu_c
            cov_sum += diff.T @ diff
          else:
            mu_c = np.zeros(m_len)
          means_tr.append(mu_c)

        pooled_cov = cov_sum / max(1, (N_total - 1 - K))
        reg = (np.trace(pooled_cov) * 1e-3 + 1e-6) * np.eye(m_len)
        inv_cov = np.linalg.pinv(pooled_cov + reg)

        scores = []
        for c in range(K):
          mu_c = means_tr[c]
          score = float(X_all[i] @ inv_cov @ mu_c - 0.5 * mu_c @ inv_cov @ mu_c)
          scores.append(score)
        pred_c = int(np.argmax(scores))
        conf_mat[y_all[i], pred_c] += 1
      acc_global = float(np.trace(conf_mat) / max(1, N_total) * 100.0)
    else:
      # Aproximación Nearest Centroid para pocas muestras
      for i in range(N_total):
        x_i = X_all[i]
        dists = [np.linalg.norm(x_i - mu_dict[vocales_activas[c]]) for c in range(K)]
        pred_c = int(np.argmin(dists))
        conf_mat[y_all[i], pred_c] += 1
      acc_global = float(np.trace(conf_mat) / max(1, N_total) * 100.0)

    # 4. Matriz de Correlación Cruzada a Lag Cero (Crosstalk)
    all_raw = [s['raw'] for s in self.samples if 'raw' in s and s['raw'] is not None]
    if all_raw and m_len >= 2:
      concat_raw = np.hstack(all_raw)
      raw_ch = concat_raw.shape[0]
      if raw_ch >= 2:
        full_corr = np.abs(np.nan_to_num(np.corrcoef(concat_raw)))
        corr_mat = np.ones((m_len, m_len), dtype=float)
        use_k = min(m_len, raw_ch)
        corr_mat[:use_k, :use_k] = full_corr[:use_k, :use_k]
        off_diag = [corr_mat[r, c] for r in range(m_len) for c in range(r + 1, m_len)]
        max_crosstalk = float(max(off_diag)) if off_diag else 0.0
      else:
        corr_mat = np.ones((m_len, m_len), dtype=float)
        max_crosstalk = 0.0
    else:
      corr_mat = np.ones((m_len, m_len), dtype=float)
      max_crosstalk = 0.0

    # 5. Diagnóstico Anatómico y Semáforo
    alertas = []
    if max_crosstalk >= 0.60:
      pares_xtalk = []
      for r in range(m_len):
        for c in range(r + 1, m_len):
          if corr_mat[r, c] >= 0.60:
            pares_xtalk.append(f"{ch_labels[r]} <-> {ch_labels[c]} (|r|={corr_mat[r, c]:.2f})")
      alertas.append(
        f"[CROSSTALK ALTO] |r_max| = {max_crosstalk:.2f} >= 0.60 en: {', '.join(pares_xtalk)}. "
        "Alta conduccion de volumen o puente de gel salino. Separe fisicamente los electrodos o reduzca exceso de gel."
      )

    confusiones_criticas = []
    for i in range(K):
      for j in range(i + 1, K):
        v1 = vocales_activas[i]
        v2 = vocales_activas[j]
        theta = angles[i, j]
        fisher = fisher_mat[i, j]
        conf_count = conf_mat[i, j] + conf_mat[j, i]
        tot_pair = counts[v1] + counts[v2]
        overlap_pct = (conf_count / max(1, tot_pair)) * 100.0
        if theta < 12.0 or overlap_pct >= 25.0 or fisher < 0.8:
          confusiones_criticas.append((v1, v2, theta, fisher, overlap_pct))

    back_overlap = [c for c in confusiones_criticas if c[0] in ('O', 'U') and c[1] in ('O', 'U')]
    front_overlap = [c for c in confusiones_criticas if c[0] in ('A', 'E', 'I') and c[1] in ('A', 'E', 'I')]
    other_overlap = [c for c in confusiones_criticas if c not in back_overlap and c not in front_overlap]

    if back_overlap:
      for c in back_overlap:
        alertas.append(
          f"[SOLAPAMIENTO POSTERIOR /{c[0]}/-/{c[1]}/]: theta={c[2]:.1f} deg, solapamiento={c[4]:.0f}%, Fisher J={c[3]:.2f}. "
          "Falta de discriminacion entre protrusion labial y redondeamiento. "
          "Sugerencia anatomica: reubicar electrodo hacia Orbicularis Oris / Mentoniano vs Digastrico (apertura mandibular)."
        )

    if front_overlap:
      for c in front_overlap:
        alertas.append(
          f"[SOLAPAMIENTO ANTERIOR /{c[0]}/-/{c[1]}/]: theta={c[2]:.1f} deg, solapamiento={c[4]:.0f}%, Fisher J={c[3]:.2f}. "
          "Falta de diferenciacion entre retraccion comisural y apertura mandibular. "
          "Sugerencia anatomica: revisar electrodos en Zygomaticus Major / Risorio vs Vientre Anterior del Digastrico."
        )

    if other_overlap:
      for c in other_overlap:
        alertas.append(
          f"[SOLAPAMIENTO VOCALICO /{c[0]}/-/{c[1]}/]: theta={c[2]:.1f} deg, solapamiento={c[4]:.0f}%, Fisher J={c[3]:.2f}. "
          "Verificar contacto de electrodos y pedir mayor diferenciacion articular."
        )

    # Estado del semáforo
    if max_crosstalk >= 0.60:
      estado = "CROSSTALK ALTO"
      self.lbl_semaforo.setText(f"[CROSSTALK ALTO] |r_max| = {max_crosstalk:.2f} >= 0.60 entre canales.\nRevisar posible puente de gel o conduccion de volumen.")
      self.lbl_semaforo.setStyleSheet("background-color: #3b0000; color: #ff4444; font-size: 14px; font-weight: bold; padding: 10px; border: 2px solid #ff4444; border-radius: 5px;")
    elif confusiones_criticas:
      estado = "ALERTA DE SOLAPAMIENTO VOCALICO"
      resumen_conf = "; ".join([f"/{c[0]}/-/{c[1]}/ ({c[4]:.0f}% solap)" for c in confusiones_criticas])
      self.lbl_semaforo.setText(f"[ALERTA DE SOLAPAMIENTO VOCALICO] Pares afectados: {resumen_conf}.\nExactitud LOO = {acc_global:.1f}%. Ver reporte para sugerencias anatomicas.")
      self.lbl_semaforo.setStyleSheet("background-color: #332b00; color: #ffdd00; font-size: 14px; font-weight: bold; padding: 10px; border: 2px solid #ffdd00; border-radius: 5px;")
    else:
      estado = "OPTIMO"
      self.lbl_semaforo.setText(f"[OPTIMO] Separabilidad multiclase excelente (Exactitud LOO = {acc_global:.1f}%, |r_max| = {max_crosstalk:.2f}).\nNo se observan pares colapsados.")
      self.lbl_semaforo.setStyleSheet("background-color: #002b15; color: #00ff88; font-size: 14px; font-weight: bold; padding: 10px; border: 2px solid #00ff88; border-radius: 5px;")

    # 6. Formatear reporte consolidado textual
    lines = [
      "=" * 70,
      "TEST DE SEPARABILIDAD MULTICLASE 5 VOCALES Y DETECTOR DE CROSSTALK",
      "=" * 70,
      f"Estado Diagnostico             : [{estado}]",
      f"Exactitud Global (LOO-LDA)     : {acc_global:.1f}% ({N_total} repeticiones totales)",
      f"Max. Correlacion Lag-0 |r_ij|  : {max_crosstalk:.3f} (Alerta Crosstalk >= 0.60)",
      f"Vocales Evaluadas              : {', '.join(['/' + v + '/' for v in vocales_activas])}",
      "",
      "--- ACTIVACION MEDIA RMS Y DOMINANCIA POR CANAL ---"
    ]
    for idx_pos, lbl_ch in enumerate(ch_labels):
      partes_v = []
      for v in vocales_activas:
        partes_v.append(f"/{v}/: {mu_dict[v][idx_pos]:5.1f}uV ({dom_dict[v][idx_pos]:4.1f}%Dom)")
      lines.append(f"  • {lbl_ch:<26} | " + " | ".join(partes_v))

    lines.append("")
    lines.append("--- MATRIZ DE ANGULOS DE SEPARACION DIRECCIONAL theta_v1,v2 (grados) ---")
    header_v = "Vocal".ljust(10) + "".join(f"/{v}/".rjust(10) for v in vocales_activas)
    lines.append(header_v)
    for i, v1 in enumerate(vocales_activas):
      row_str = f"/{v1}/".ljust(10) + "".join(f"{angles[i, j]:9.1f} deg" for j in range(K))
      lines.append(row_str)

    lines.append("")
    lines.append("--- MATRIZ DE DISTANCIA DISCRIMINANTE DE FISHER J_v1,v2 ---")
    lines.append(header_v)
    for i, v1 in enumerate(vocales_activas):
      row_str = f"/{v1}/".ljust(10) + "".join(f"{fisher_mat[i, j]:10.2f}" for j in range(K))
      lines.append(row_str)

    lines.append("")
    lines.append("--- MATRIZ DE CONFUSION (LEAVE-ONE-OUT LDA) ---")
    lines.append(header_v + "   | Total")
    for i, v1 in enumerate(vocales_activas):
      row_str = f"/{v1}/ (real)".ljust(10) + "".join(f"{conf_mat[i, j]:10d}" for j in range(K)) + f"   | {counts[v1]:4d}"
      lines.append(row_str)

    lines.append("")
    lines.append("--- MATRIZ DE CORRELACION CRUZADA A LAG CERO |r_ij| (Crosstalk) ---")
    header_ch = "Canal".ljust(24) + "".join(f"Ch{c}".rjust(8) for c in range(m_len))
    lines.append(header_ch)
    for r_idx in range(m_len):
      row_str = ch_labels[r_idx][:22].ljust(24) + "".join(f"{corr_mat[r_idx, c_idx]:8.2f}" for c_idx in range(m_len))
      lines.append(row_str)

    if alertas:
      lines.append("")
      lines.append("--- ALERTAS Y DIAGNOSTICO ANATOMICO ---")
      for alt in alertas:
        lines.append(f"  • {alt}")

    lines.append("=" * 70)

    self.last_report_text = "\n".join(lines)
    self.txt_resultados.setPlainText(self.last_report_text)

    # Guardar métricas estructuradas para persistencia en metadata
    min_theta = float(min([angles[i, j] for i in range(K) for j in range(i + 1, K)])) if K >= 2 else 0.0
    min_fisher = float(min([fisher_mat[i, j] for i in range(K) for j in range(i + 1, K)])) if K >= 2 else 0.0

    self.last_metrics = {
      "timestamp": datetime.now().isoformat(),
      "estado_diagnostico": estado,
      "exactitud_global_loo_pct": round(acc_global, 2),
      "total_repeticiones": N_total,
      "vocales_evaluadas": vocales_activas,
      "repeticiones_por_vocal": {v: counts[v] for v in vocales_activas},
      "canales_evaluados": ch_labels,
      "angulo_minimo_grados": round(min_theta, 2),
      "fisher_j_minimo": round(min_fisher, 3),
      "max_crosstalk_lag0": round(max_crosstalk, 3),
      "matriz_angulos_grados": [[round(float(angles[r, c]), 2) for c in range(K)] for r in range(K)],
      "matriz_fisher_j": [[round(float(fisher_mat[r, c]), 3) for c in range(K)] for r in range(K)],
      "matriz_confusion_loo": [[int(conf_mat[r, c]) for c in range(K)] for r in range(K)],
      "matriz_crosstalk_lag0": [[round(float(corr_mat[r, c]), 3) for c in range(m_len)] for r in range(m_len)],
      "activacion_rms_media_uV": {v: [round(float(x), 2) for x in mu_dict[v]] for v in vocales_activas},
      "dominancia_muscular_media_pct": {v: [round(float(x), 1) for x in dom_dict[v]] for v in vocales_activas},
      "alertas_diagnostico": alertas
    }
    self.btn_anexar.setEnabled(True)

  def _anexar_a_notas(self):
    """Anexa el diagnóstico textual y métricas a las Notas del Experimento y a metadata."""
    p = self.parent_plotter
    if p is None or not self.last_metrics:
      return
    p.last_separability_metrics = self.last_metrics
    if hasattr(p, 'txt_notas_live'):
      cur = p.txt_notas_live.toPlainText().strip()
      bloque = (
        f"[Test 5 Vocales ({datetime.now().strftime('%H:%M:%S')})]: "
        f"Estado=[{self.last_metrics['estado_diagnostico']}], "
        f"Exactitud_LOO={self.last_metrics['exactitud_global_loo_pct']:.1f}%, "
        f"theta_min={self.last_metrics['angulo_minimo_grados']:.1f} deg, "
        f"Fisher_J_min={self.last_metrics['fisher_j_minimo']:.2f}, "
        f"|r_max|={self.last_metrics['max_crosstalk_lag0']:.2f}"
      )
      nuevo_txt = f"{cur}\n{bloque}".strip() if cur else bloque
      p.txt_notas_live.setPlainText(nuevo_txt)
      if hasattr(p, 'notas_panel'):
        p.notas_panel.setVisible(True)

    if getattr(p, 'last_measurement_dir', None) and os.path.exists(p.last_measurement_dir):
      try:
        p.persistir_notas_experimento(p.last_measurement_dir)
      except Exception as exc:
        print(f"[TestVocales] Error persistiendo notas: {exc}")

    QtWidgets.QMessageBox.information(
      self, "Diagnóstico Anexado",
      "El diagnóstico del Test de Separabilidad de las 5 Vocales y Crosstalk se anexó a las Notas del Experimento y a metadata.json."
    )


# Alias de retrocompatibilidad
TestSeparabilidadOUDialog = TestSeparabilidadVocalesDialog


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
    self.BTN_START_STYLE = "background-color: #050505; color: #00FF00; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #00FF00; border-radius: 4px;"
    self.BTN_STOP_STYLE = "background-color: #050505; color: #FF0000; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #FF0000; border-radius: 4px;"
    self.BTN_REC_START_STYLE = "background-color: #050505; color: #00FFFF; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #00FFFF; border-radius: 4px;"
    self.BTN_REC_STOP_STYLE = "background-color: #FF0000; color: #000000; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #FF0000; border-radius: 4px;"

    # --- Estado interno ---
    self.is_recording = False
    self.counting_started = False # NUEVO: Bandera para controlar el inicio del conteo del metrónomo
    self.current_recording = []
    self.recording_start_time = None
    self.acquisition_thread = None
    # --- REFACTOR: Encapsular cola y evento ---
    self.data_queue = queue.Queue()
    self.stop_event = threading.Event()

    self.is_acquiring = False
    
    from utils.config_manager import ConfigManager
    self.config_mgr = ConfigManager()
    
    # --- Propiedades que se definen al iniciar adquisición ---
    self.NUM_CANALES = 0
    self.CANALES_DAQ = []
    self.SAMPLE_RATE = 2000
    self.PLOT_DURATION_S = 3.0
    self.PLOT_SAMPLES = int(self.PLOT_DURATION_S * self.SAMPLE_RATE)
    self.CHUNK_DURATION_S = 0.05
    
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

    # --- NUEVO: Gestión de canales de hardware y calibración ---
    self.nombres_musculos_hw = [None] * 16
    self.factores_calibracion_hw = [495.0] * 16
    self.active_channel_indices = []
    self._autoforge_token = 0
    self.autoforge_session_dirs = []
    self.autoforge_do_cvm = True
    self.cvm_completed = False
    self.final_noise_completed = False
    self.is_cvm_active = False
    self.is_recording_final_noise = False
    self.rt_peaks = []
    self.rt_last_noise_eval = []
    self.raw_noise_accumulated = []

    for hw in range(16):
      cfg = self.config_mgr.get_channel_config(hw)
      self.nombres_musculos_hw[hw] = cfg.get("musculo", f"Canal {hw}")
      self.factores_calibracion_hw[hw] = float(cfg.get("factor_calibracion", 495.0))

    # --- NUEVO: Estado del Filtro Notch ---
    self.notch_sos = None
    self.notch_zi = None

    self.FILTER_ORDER = 4 # Orden del filtro Butterworth

    # --- Notas del Experimento, Métricas y Fotometría Webcam ---
    self.last_measurement_dir = None
    self.last_separability_metrics = None
    self.notas_experimento = ""
    self.photometry_mgr = None
    if WebcamPhotometryManager is not None:
      try:
        self.photometry_mgr = WebcamPhotometryManager(parent=self, camera_index=0)
      except Exception as exc:
        print(f"[FotometriaWebcam] Aviso al inicializar manager: {exc}")
        self.photometry_mgr = None

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
    self.resize(init_w, init_h)
    
    # --- REFACTOR: Dividir la configuración de la UI en métodos ---
    self._setup_ui_layouts()
    self._setup_ui_config_panel()
    self._setup_ui_filter_panel()
    self._setup_ui_controls()
    self._setup_ui_plots()
    self._setup_ui_final_layout()
    self._connect_signals()
    self._load_protocol_config()
    
    # --- Configuración inicial ---
    self.on_autoscroll_toggle() 
    self.set_controls_enabled(False) # Deshabilitar controles al inicio
    
    # --- Timer para actualizar el plot ---
    self.timer = QtCore.QTimer()
    self.timer.timeout.connect(self.actualizar_plot)
    self.timer.start(30) # Actualiza la GUI cada 30 ms
    
    # --- NUEVO: Maximizar ventana por defecto ---
    self.showMaximized()

  def _load_protocol_config(self):
    """
    Ejecuta la funcionalidad de _load_protocol_config.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    bpm = 60
    pulse_count = 30
    notch_enabled = True # Por defecto activado
    if os.path.exists(METRONOME_CONFIG_PATH):
      try:
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          data = json.load(f)
          bpm = data.get('last_bpm', 60)
          pulse_count = data.get('last_beat_count', 30)
          notch_enabled = data.get('notch_enabled', True) # Recuperar el estado anterior
      except Exception:
        pass # Usar defaults si hay error
    self.spin_bpm.setValue(bpm)
    self.chk_notch_enable.setChecked(True) # Filtro Notch 50Hz forzado siempre activo por defecto

  def _setup_ui_layouts(self):
    """
    Ejecuta la funcionalidad de _setup_ui_layouts.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.main_layout = QtWidgets.QVBoxLayout()
    if getattr(self, 'is_compact_screen', False):
      self.main_layout.setContentsMargins(6, 6, 6, 6)
      self.main_layout.setSpacing(4)
    self.setLayout(self.main_layout)
    
    self.config_layout = QtWidgets.QGridLayout()
    if getattr(self, 'is_compact_screen', False):
      self.config_layout.setContentsMargins(6, 4, 6, 4)
      self.config_layout.setHorizontalSpacing(6)
      self.config_layout.setVerticalSpacing(4)
    self.config_groupbox = QtWidgets.QGroupBox("Configuración de Adquisición")
    self.config_groupbox.setLayout(self.config_layout)
    
    self.filter_layout = QtWidgets.QHBoxLayout()
    self.filter_groupbox = QtWidgets.QGroupBox("Filtro Digital (Pasa-Banda)")
    self.filter_groupbox.setLayout(self.filter_layout)

    self.button_layout = QtWidgets.QHBoxLayout()
    self._setup_notas_panel()
    self.trigger_layout = QtWidgets.QHBoxLayout()
    self.measure_layout = QtWidgets.QHBoxLayout()
    self.measure_widget = QtWidgets.QWidget()
    self.measure_widget.setLayout(self.measure_layout)

    self.spectrogram_layout = QtWidgets.QHBoxLayout()
    self.spectrogram_groupbox = QtWidgets.QGroupBox("Espectrograma")
    self.spectrogram_groupbox.setLayout(self.spectrogram_layout)

  def _setup_notas_panel(self):
    """Configura el panel desplegable de Notas del Experimento con plantillas rápidas."""
    self.notas_panel = QtWidgets.QGroupBox("Notas del Experimento")
    self.notas_panel.setStyleSheet("""
      QGroupBox {
        border: 1.5px solid #00e5ff;
        border-radius: 5px;
        margin-top: 6px;
        padding-top: 8px;
        font-weight: bold;
        color: #00e5ff;
        background-color: #070913;
      }
      QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
      }
    """)
    notas_vbox = QtWidgets.QVBoxLayout(self.notas_panel)
    notas_vbox.setContentsMargins(8, 6, 8, 8)
    notas_vbox.setSpacing(6)

    self.txt_notas_live = QtWidgets.QPlainTextEdit()
    self.txt_notas_live.setPlaceholderText(
      "Observaciones pre, durante y post-medición (posición de electrodos, preparación piel/gel, confusión vocálica, 50Hz, etc.)..."
    )
    self.txt_notas_live.setFixedHeight(75)
    self.txt_notas_live.setStyleSheet("""
      QPlainTextEdit {
        background-color: #0b0f1a;
        color: #e0f7fa;
        border: 1px solid #1e3a5f;
        border-radius: 4px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
      }
    """)
    notas_vbox.addWidget(self.txt_notas_live)

    btn_row = QtWidgets.QHBoxLayout()
    btn_tmpl_electrodos = QtWidgets.QPushButton("+ Electrodos")
    btn_tmpl_piel = QtWidgets.QPushButton("+ Piel/Gel")
    btn_tmpl_confusion = QtWidgets.QPushButton("+ Confusión Vocales")
    btn_tmpl_ruido = QtWidgets.QPushButton("+ Ruido 50Hz")
    self.btn_actualizar_notas = QtWidgets.QPushButton("Actualizar Notas en Última Medición")

    tmpl_style = """
      QPushButton {
        background-color: #101a28;
        color: #80d8ff;
        border: 1px solid #26c6da;
        border-radius: 3px;
        padding: 4px 8px;
        font-size: 11px;
        font-weight: bold;
      }
      QPushButton:hover {
        background-color: #26c6da;
        color: #000000;
      }
    """
    for b in (btn_tmpl_electrodos, btn_tmpl_piel, btn_tmpl_confusion, btn_tmpl_ruido):
      b.setStyleSheet(tmpl_style)
      btn_row.addWidget(b)

    btn_tmpl_electrodos.clicked.connect(lambda: self._insertar_plantilla_nota("[Electrodos]: Posicion = ; Adherencia = Buena; Distancia inter-electrodo = 20 mm; Reubicado canal = "))
    btn_tmpl_piel.clicked.connect(lambda: self._insertar_plantilla_nota("[Piel/Gel]: Limpieza alcohol = Si; Gel conductor = Aplicado; Impedancia contacto = Buena"))
    btn_tmpl_confusion.clicked.connect(lambda: self._insertar_plantilla_nota("[Confusion Vocales]: Pares afectados = /O/-/U/, /A/-/E/; Co-activacion = ; Separabilidad = "))
    btn_tmpl_ruido.clicked.connect(lambda: self._insertar_plantilla_nota("[Ruido 50Hz]: Nivel observado = ; Conexion tierra/DIFF = Verificado; Filtro Notch = Activo"))

    self.btn_actualizar_notas.setStyleSheet("""
      QPushButton {
        background-color: #003322;
        color: #00ffaa;
        border: 1.5px solid #00ffaa;
        border-radius: 3px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: bold;
      }
      QPushButton:hover {
        background-color: #00ffaa;
        color: #000000;
      }
    """)
    self.btn_actualizar_notas.clicked.connect(self.on_actualizar_notas_ultima_medicion)

    btn_row.addStretch(1)
    btn_row.addWidget(self.btn_actualizar_notas)
    notas_vbox.addLayout(btn_row)

    self.notas_panel.setVisible(False)

  def _setup_ui_config_panel(self):
    """Configura el panel superior de adquisición."""
    self.label_device = QtWidgets.QLabel("Dispositivo:")
    self.cmb_device = QtWidgets.QComboBox()
    self.cmb_device.addItems(["Dev1", "Dev2", "Dev3"])

    # --- NUEVO: Checkbox para Modo Prueba ---
    
    self.chk_use_mic = QtWidgets.QCheckBox("Usar Micrófono (Prueba en Casa)")
    self.chk_use_mic.setToolTip("Usa el micrófono de la PC en lugar de la DAQ.")

    self.chk_modo_prueba = QtWidgets.QCheckBox("Modo Prueba")
    self.chk_modo_prueba.setToolTip("Usa datos simulados en lugar de la tarjeta NI-DAQ.")
    
    self.cmb_fuente_prueba = QtWidgets.QComboBox()
    self.cmb_fuente_prueba.addItems(["Senoidal", "Archivo 'senal_de_prueba'"])
    self.cmb_fuente_prueba.setEnabled(False)
    self.cmb_fuente_prueba.setToolTip("Elige si simular con onda senoidal o cargar 'grabacion.csv' de la carpeta 'senal_de_prueba'")

    # --- NUEVO: Checkbox para el Metrónomo ---
    self.chk_use_metronome = QtWidgets.QCheckBox("Usar Metrónomo")
    self.chk_use_metronome.setChecked(True) # Siempre activo por defecto
    self.chk_use_metronome.setToolTip("Abre y controla el metrónomo visual durante la adquisición.")

    # --- NUEVO: Control de Frecuencia de Prueba ---
    self.label_test_freq = QtWidgets.QLabel("Frec. Prueba (Hz):")
    self.spin_test_freq = SpinBox(value=50, step=10, bounds=(1, 5000), int=True)
    self.spin_test_freq.setFixedWidth(80)

    self.label_sample_rate = QtWidgets.QLabel("Sample Rate (S/s):")
    self.cmb_sample_rate = QtWidgets.QComboBox()
    self.cmb_sample_rate.addItems(["2000", "6000", "10000", "20000", "44100", "50000"])
    self.cmb_sample_rate.setCurrentText("2000")

    self.label_plot_duration = QtWidgets.QLabel("Duración Plot (s):")
    self.spin_plot_duration = SpinBox(value=10, step=1, bounds=(1, 60), int=True)
    self.spin_plot_duration.setFixedWidth(80)

    self.label_channels = QtWidgets.QLabel("Canales:")
    self.channel_checkboxes = []
    self.channel_layout = QtWidgets.QHBoxLayout()
    
    any_checked = False
    for i in range(8): # Ofrecer 8 canales
      chk = QtWidgets.QCheckBox(f"ai{i}")
      is_active = self.config_mgr.config.get("canales", {}).get(f"Canal {i}", {}).get("activo_por_defecto", False)
      if is_active:
        chk.setChecked(True)
        any_checked = True
      chk.toggled.connect(self._on_channel_checkbox_toggled)
      self.channel_checkboxes.append(chk)
      self.channel_layout.addWidget(chk)
    
    # Si no hay ninguno activo, marcamos el 0 por defecto por seguridad
    if not any_checked and self.channel_checkboxes:
      self.channel_checkboxes[0].setChecked(True)

    self.btn_config_musculos = QtWidgets.QPushButton("Configurar Músculos")
    self.btn_config_musculos.setStyleSheet("""
      QPushButton {
        background-color: #141424;
        color: #00ffff;
        font-weight: bold;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        padding: 4px 10px;
        border: 1.5px solid #00ffff;
        border-radius: 4px;
      }
      QPushButton:hover {
        background-color: #00ffff;
        color: #000000;
      }
    """)
    self.btn_config_musculos.setToolTip("Configurar nombres de músculos asignados a los canales (detiene la adquisición si está corriendo)")
    self.btn_config_musculos.clicked.connect(self.abrir_dialogo_musculos)
    self.channel_layout.addSpacing(8)
    self.channel_layout.addWidget(self.btn_config_musculos)

    self.btn_start_acq = QtWidgets.QPushButton("Iniciar Adquisición")
    self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)

    # --- NUEVO: Controles de Protocolo ---
    self.label_bpm = QtWidgets.QLabel("BPM:")
    self.spin_bpm = SpinBox(value=60, int=True, bounds=(30, 200), step=1)
    self.spin_bpm.setFixedWidth(60) # Acortar el recuadro del BPM
    self.spin_metro_subdivs = SpinBox(value=4, int=True, bounds=(1, 8), step=1)
    self.spin_metro_subdivs.setFixedWidth(50)
    self.spin_metro_subdivs.setToolTip("Subdivisiones rítmicas del metrónomo")
    self.label_noise_duration = QtWidgets.QLabel("Noise (Inicio) (s):")
    adq_conf = self.config_mgr.get("adquisicion") or {}
    default_noise = adq_conf.get("ruido_segundos", 3.0)
    self.spin_noise_duration = SpinBox(value=default_noise, dec=True, bounds=(0.5, 20.0), step=0.5)
    self.spin_noise_duration.setFixedWidth(80)

    self.config_layout.addWidget(self.label_device, 0, 0)
    self.config_layout.addWidget(self.cmb_device, 0, 1)
    self.config_layout.addWidget(self.chk_use_mic, 0, 8)
    self.config_layout.addWidget(self.chk_modo_prueba, 0, 2)
    self.config_layout.addWidget(self.cmb_fuente_prueba, 0, 3)
    self.config_layout.addWidget(self.chk_use_metronome, 0, 4)
    self.config_layout.addWidget(self.label_test_freq, 0, 5)
    self.config_layout.addWidget(self.spin_test_freq, 0, 6)
    self.config_layout.addWidget(self.label_sample_rate, 1, 0) # Fila 1
    self.config_layout.addWidget(self.cmb_sample_rate, 1, 1) 
    self.config_layout.addWidget(self.label_plot_duration, 1, 2)
    self.config_layout.addWidget(self.spin_plot_duration, 1, 3)
    self.config_layout.addWidget(QtWidgets.QLabel("BPM / Sub:"), 1, 5)
    bpm_layout = QtWidgets.QHBoxLayout()
    bpm_layout.addWidget(self.spin_bpm)
    bpm_layout.addWidget(self.spin_metro_subdivs)
    self.config_layout.addLayout(bpm_layout, 1, 6) # BPM y Subdivs en la misma celda
    self.config_layout.addWidget(self.label_noise_duration, 2, 5)
    self.config_layout.addWidget(self.spin_noise_duration, 2, 6) # Ruido
    self.config_layout.addWidget(self.label_channels, 2, 0)
    self.config_layout.addLayout(self.channel_layout, 2, 1, 1, 4) # row, col, rowspan, colspan
    
    # --- NUEVO: Control de Modo Terminal ---
    self.label_terminal_config = QtWidgets.QLabel(" CONEXIÓN DAQ (Voltaje):")
    self.label_terminal_config.setStyleSheet("color: #FFFF00; font-weight: bold; font-size: 13px;")
    self.cmb_terminal_config = QtWidgets.QComboBox()
    self.cmb_terminal_config.setStyleSheet("background-color: #000000; color: #FFFF00; border: 2px solid #FFFF00; font-size: 14px; font-weight: bold; padding: 4px;")
    self.cmb_terminal_config.addItems(["RSE", "DIFF", "NRSE", "DEFAULT"])
    self.cmb_terminal_config.setCurrentText("RSE")
    self.cmb_terminal_config.setToolTip("Modo de conexión a tierra. Usa 'DIFF' si tienes mucho ruido de 50 Hz.")
    self.config_layout.addWidget(self.label_terminal_config, 3, 0)
    self.config_layout.addWidget(self.cmb_terminal_config, 3, 1)

    self.config_layout.addWidget(self.btn_start_acq, 0, 7, 4, 1) # Ocupa 4 filas

    self._setup_native_metronome()
    self.config_layout.addWidget(self.metronome_container, 0, 9, 4, 1)

  def _setup_ui_filter_panel(self):
    """Configura el panel de filtro."""
    self.chk_notch_enable = QtWidgets.QCheckBox("Filtro Notch 50 Hz")
    self.chk_notch_enable.setChecked(True) # Filtro Notch por predeterminado
    self.chk_notch_enable.setToolTip("Aplica un filtro para eliminar el ruido de la red eléctrica (50 Hz).")

    self.chk_rms_env = QtWidgets.QCheckBox("Envolvente RMS (Realtime)")
    self.chk_rms_env.setChecked(False)
    self.chk_rms_env.setToolTip("Calcula y grafica la envolvente RMS en tiempo real.")
    
    self.label_rms_window = QtWidgets.QLabel("Ventana (ms):")
    self.spin_rms_window = SpinBox(value=100.0, step=10.0, bounds=(10.0, 1000.0), dec=True)
    self.spin_rms_window.setFixedWidth(60)
    self.spin_rms_window.setToolTip("Tamaño de la ventana de integración de la envolvente en milisegundos.")

    self.chk_filter_enable = QtWidgets.QCheckBox("Habilitar Filtro")
    self.chk_filter_enable.setChecked(True) # Filtro Pasa-Banda habilitado por defecto
    
    self.label_low_cut = QtWidgets.QLabel("Frec. Baja (Hz):")
    self.spin_low_cut = SpinBox(value=20, step=1, bounds=(1, 20000), int=True)
    self.spin_low_cut.setFixedWidth(80)
    self.label_high_cut = QtWidgets.QLabel("Frec. Alta (Hz):")
    self.spin_high_cut = SpinBox(value=500, step=1, bounds=(2, 22000), int=True)
    self.spin_high_cut.setFixedWidth(80)
    self.label_filter_order = QtWidgets.QLabel(f"Orden: {self.FILTER_ORDER} (Butterworth)")

    self.filter_layout.addWidget(self.chk_notch_enable)
    self.filter_layout.addSpacing(20)
    self.filter_layout.addWidget(self.chk_filter_enable)
    self.filter_layout.addSpacing(20)
    
    # --- NUEVO: Toggles UI para mostrar/ocultar regiones ---
    self.chk_show_noise = QtWidgets.QCheckBox("Mostrar Ruido")
    self.chk_show_noise.setChecked(False) # Desactivado por defecto
    self.chk_show_noise.setToolTip("Muestra u oculta las regiones de ruido.")
    self.chk_show_noise.clicked.connect(self.toggle_noise_regions)
    self.filter_layout.addWidget(self.chk_show_noise)
    self.filter_layout.addSpacing(20)

    self.chk_show_peaks = QtWidgets.QCheckBox("Mostrar Picos")
    self.chk_show_peaks.setChecked(False) # Desactivado por defecto
    self.chk_show_peaks.setToolTip("Muestra u oculta los marcadores de picos (Peak-Hold).")
    self.chk_show_peaks.clicked.connect(self.toggle_peak_scatter)
    self.filter_layout.addWidget(self.chk_show_peaks)
    self.filter_layout.addSpacing(20)

    self.filter_layout.addWidget(self.chk_rms_env)
    self.filter_layout.addWidget(self.label_rms_window)
    self.filter_layout.addWidget(self.spin_rms_window)
    self.filter_layout.addSpacing(15)

    self.filter_layout.addSpacing(15)
    self.filter_layout.addWidget(self.label_low_cut)
    self.filter_layout.addWidget(self.spin_low_cut)
    self.filter_layout.addSpacing(10)
    self.filter_layout.addWidget(self.label_high_cut)
    self.filter_layout.addWidget(self.spin_high_cut)
    self.filter_layout.addSpacing(15)
    self.filter_layout.addWidget(self.label_filter_order)
    self.filter_layout.addStretch(1)

  def _setup_ui_controls(self):
    """Configura los controles de grabación, trigger y mediciones."""
    
    self.btn_autoforge = QtWidgets.QPushButton(" AUTOGRABADO")
    self.btn_autoforge.setStyleSheet("""
      QPushButton {
        background-color: #1a001a; 
        color: #ff00ff; 
        font-weight: 900; 
        font-family: 'Consolas', 'Courier New', monospace; 
        font-size: 15px; 
        padding: 8px; 
        border-radius: 4px;
        border: 2px solid #ff00ff;
        border-right: 4px solid #00ffff;
        border-bottom: 4px solid #00ffff;
      }
      QPushButton:hover {
        background-color: #ff00ff; color: #000000; border: 2px solid #00ffff;
      }
    """)
    self.btn_autoforge.clicked.connect(self.iniciar_autoforge)

    self.btn_autoforge_continuo = QtWidgets.QPushButton(" SECUENCIA CONTINUA")
    self.btn_autoforge_continuo.setStyleSheet("""
      QPushButton {
        background-color: #0d001a; 
        color: #aa00ff; 
        font-weight: 900; 
        font-family: 'Consolas', 'Courier New', monospace; 
        font-size: 15px; 
        padding: 8px; 
        border-radius: 4px;
        border: 2px solid #aa00ff;
        border-right: 4px solid #ff00ff;
        border-bottom: 4px solid #ff00ff;
      }
      QPushButton:hover {
        background-color: #aa00ff; color: #ffffff; border: 2px solid #ff00ff;
      }
    """)
    self.btn_autoforge_continuo.clicked.connect(self.iniciar_autoforge_continuo)

    self.btn_record = QtWidgets.QPushButton("Empezar a Grabar")
    self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)

    # --- Botones Superiores: Notas, Fotometría 3D y Test 5 Vocales ---
    self.btn_notas = QtWidgets.QPushButton("NOTAS")
    self.btn_notas.setStyleSheet("""
      QPushButton {
        background-color: #0b1a24;
        color: #00ffcc;
        font-weight: 900;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        padding: 7px 12px;
        border-radius: 4px;
        border: 2px solid #00ffcc;
      }
      QPushButton:hover {
        background-color: #00ffcc;
        color: #000000;
      }
    """)
    self.btn_notas.setToolTip("Desplegar panel de Notas del Experimento (pre, durante y post medición)")
    self.btn_notas.clicked.connect(self.toggle_notas_panel)

    self.btn_fotometria = QtWidgets.QPushButton("FOTOMETRIA 3D")
    self.btn_fotometria.setStyleSheet("""
      QPushButton {
        background-color: #0d1f2d;
        color: #00e5ff;
        font-weight: 900;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        padding: 7px 12px;
        border-radius: 4px;
        border: 2px solid #00e5ff;
      }
      QPushButton:hover {
        background-color: #00e5ff;
        color: #000000;
      }
    """)
    self.btn_fotometria.setToolTip("Abrir asistente de posicionamiento facial y fotometría 3D de electrodos")
    self.btn_fotometria.clicked.connect(self.abrir_asistente_fotometria)

    self.btn_test_vocales = QtWidgets.QPushButton("TEST 5 VOCALES")
    self.btn_test_vocales.setStyleSheet("""
      QPushButton {
        background-color: #240b20;
        color: #ff00aa;
        font-weight: 900;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 13px;
        padding: 7px 12px;
        border-radius: 4px;
        border: 2px solid #ff00aa;
      }
      QPushButton:hover {
        background-color: #ff00aa;
        color: #ffffff;
      }
    """)
    self.btn_test_vocales.setToolTip("Test de Separabilidad Multiclase de las 5 Vocales (A, E, I, O, U) y Crosstalk")
    self.btn_test_vocales.clicked.connect(self.abrir_test_separabilidad_5_vocales)
    
    self.label_rec_time = QtWidgets.QLabel("Grabando: --:--.-")
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #E91E63;")
    sp = self.label_rec_time.sizePolicy()
    sp.setRetainSizeWhenHidden(True)
    self.label_rec_time.setSizePolicy(sp)
    self.label_rec_time.setVisible(False)
    
    self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll (Armar Trigger)")
    self.chk_autoscroll.setChecked(True)
    
    # --- Armar layout de botones ---
    self.button_layout.addWidget(self.btn_record)
    self.button_layout.addWidget(self.btn_autoforge)
    self.button_layout.addWidget(self.btn_autoforge_continuo)
    self.button_layout.addWidget(self.btn_notas)
    self.button_layout.addWidget(self.btn_fotometria)
    self.button_layout.addWidget(self.btn_test_vocales)
    self.button_layout.addWidget(self.label_rec_time)
    
    self.button_layout.addStretch(1) # Mover todo el espacio vacío aquí
    self.button_layout.addWidget(self.chk_autoscroll)
    
    # Opciones de trigger ---
    self.chk_trigger = QtWidgets.QCheckBox("Habilitar Trigger")
    self.label_trig_chan = QtWidgets.QLabel("Canal:")
    self.cmb_trig_chan = QtWidgets.QComboBox()
    # Se poblará al iniciar la adquisición
    
    self.label_trig_level = QtWidgets.QLabel("Nivel (µV):")
    self.spin_trig_level = SpinBox(value=1000.0, step=100.0, dec=True, minStep=1.0, bounds=(-20000.0, 20000.0))
    self.spin_trig_level.setFixedWidth(80)
    
    self.label_trig_edge = QtWidgets.QLabel("Flanco:")
    self.cmb_trig_edge = QtWidgets.QComboBox()
    self.cmb_trig_edge.addItems(["Subida ", "Bajada "])
    
    # --- NUEVO: Umbral para SNR ---
    self.label_peak_th = QtWidgets.QLabel("Umbral Picos SNR (µV):")
    self.spin_peak_th = SpinBox(value=200.0, step=50.0, bounds=(0.0, 20000.0), dec=True)
    self.spin_peak_th.setFixedWidth(80)

    # --- Armar layout del Trigger ---
    self.trigger_layout.addWidget(self.chk_trigger)
    self.trigger_layout.addSpacing(15)
    self.trigger_layout.addWidget(self.label_trig_chan)
    self.trigger_layout.addWidget(self.cmb_trig_chan)
    self.trigger_layout.addSpacing(10)
    self.trigger_layout.addWidget(self.label_trig_level)
    self.trigger_layout.addWidget(self.spin_trig_level)
    self.trigger_layout.addSpacing(10)
    self.trigger_layout.addWidget(self.label_trig_edge)
    self.trigger_layout.addWidget(self.cmb_trig_edge)
    self.trigger_layout.addSpacing(30)
    self.trigger_layout.addWidget(self.label_peak_th)
    self.trigger_layout.addWidget(self.spin_peak_th)
    self.trigger_layout.addStretch(1)
    
    # --- Widgets de Mediciones ---
    self.measure_labels = [] # Lista para guardar los labels
    self.measure_layout.addStretch(1)
    
    # --- Widgets del Espectrograma ---

    self.chk_spectrogram_enable = QtWidgets.QCheckBox("Habilitar Espectrograma")
    self.label_spectrogram_chan = QtWidgets.QLabel("Canal:")
    self.cmb_spectrogram_chan = QtWidgets.QComboBox()

    self.spectrogram_layout.addWidget(self.chk_spectrogram_enable)
    self.spectrogram_layout.addSpacing(15)
    self.spectrogram_layout.addWidget(self.label_spectrogram_chan)
    self.spectrogram_layout.addWidget(self.cmb_spectrogram_chan)
    self.spectrogram_layout.addStretch(1)

  def _setup_ui_plots(self):
    """Configura los widgets de ploteo y el splitter."""
    
    # Widget contenedor para el grafico y sus controles
    self.plot_container = QtWidgets.QWidget()
    self.plot_container_layout = QtWidgets.QVBoxLayout(self.plot_container)
    self.plot_container_layout.setContentsMargins(0, 0, 0, 0)
    self.plot_container_layout.setSpacing(2)
    
    # Panel de control de visualización
    self.viz_control_panel = QtWidgets.QWidget()
    self.viz_control_panel.setStyleSheet("background-color: #050505; border-bottom: 1px solid #333;")
    self.viz_control_layout = QtWidgets.QHBoxLayout(self.viz_control_panel)
    self.viz_control_layout.setContentsMargins(5, 2, 5, 2)
    
    lbl_viz = QtWidgets.QLabel("Visualización de Canales:")
    lbl_viz.setStyleSheet("color: #00FFFF; font-weight: bold; border: none;")
    self.viz_control_layout.addWidget(lbl_viz)
    
    self.btn_hide_all_viz = QtWidgets.QPushButton("Ocultar Todos")
    self.btn_hide_all_viz.setStyleSheet("background-color: #330000; color: #ff5555; border: 1px solid #ff5555; padding: 2px 8px; font-size: 11px; font-weight: bold; border-radius: 3px;")
    self.btn_hide_all_viz.clicked.connect(self.hide_all_viz_channels)
    self.viz_control_layout.addWidget(self.btn_hide_all_viz)
    
    # Contenedor para los checkboxes dinámicos
    self.viz_checkbox_layout = QtWidgets.QHBoxLayout()
    self.viz_control_layout.addLayout(self.viz_checkbox_layout)
    self.viz_control_layout.addStretch()
    
    self.viz_checkboxes = []
    
    self.plot_container_layout.addWidget(self.viz_control_panel)

    self.plot_widget = pg.GraphicsLayoutWidget()

    # --- NUEVO: Overlay AutoForge Flotante ---
    self.autoforge_overlay = QtWidgets.QLabel(self.plot_widget)
    self.autoforge_overlay.setAlignment(QtCore.Qt.AlignCenter)
    self.autoforge_overlay.setStyleSheet("background-color: rgba(10, 5, 20, 180); color: #FF0055; font-family: 'Courier New', monospace; font-size: 75px; font-weight: 900; text-shadow: 2px 2px 5px #00FFFF; padding: 20px; border: 4px solid #00FFFF;")
    self.autoforge_overlay.hide()
    
    # Hacemos que cambie de tamaño junto con plot_widget
    self.plot_widget.installEventFilter(self)

    self.plot_container_layout.addWidget(self.plot_widget)

    self.plot_widget.setBackground('k') # Fondo Negro
    
    self.plot = self.plot_widget.addPlot(title="Canales de Adquisición")
    self.plot.setLabel('bottom', 'Tiempo (s)')
    self.plot.setLabel('left', 'Amplitud (µV)')
    # Fijar ancho del eje Y para evitar que los números grandes empujen el gráfico
    self.plot.getAxis('left').setWidth(60)

    # Widget para el espectrograma (ImageView)
    self.spectrogram_view = pg.ImageView()
    self.spectrogram_view.ui.histogram.hide()
    self.spectrogram_view.ui.roiBtn.hide()
    self.spectrogram_view.ui.menuBtn.hide()
    
    # --- NUEVO: Margen derecho para evitar cortes en pantalla completa ---
    self.plot_widget.ci.layout.setContentsMargins(10, 10, 30, 10)

    self.plot.addLegend()
    self.plot.showGrid(x=True, y=True)
    
    # --- OPTIMIZACIONES ---
    self.plot.setClipToView(True) 
    self.plot.setDownsampling(auto=True, mode='peak')
    self.plot.autoBtn.setVisible(True)

    # --- Límites de Zoom/Pan (v3.11) ---
    self.plot.getViewBox().setLimits(
      xMin=-self.spin_plot_duration.value(), # Usa el valor inicial del spinbox
      minXRange=1e-6 # 1 microsegundo de zoom máximo
    )

    # --- Línea de Trigger ---
    self.trigger_line = InfiniteLine(pos=1000.0, angle=0, movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
    self.plot.addItem(self.trigger_line)
    # --- NUEVO: Etiqueta de texto para el valor del trigger ---
    self.trigger_label = pg.TextItem(anchor=(0, 1), color=(255, 255, 0), fill=(0, 0, 0, 150))
    self.trigger_label.setZValue(100) # Asegurar que esté por encima de las curvas
    self.trigger_label.hide() # Oculto por defecto
    self.plot.addItem(self.trigger_label)
    
    # --- NUEVO: Líneas de Umbral SNR ---
    self.peak_th_line_pos = InfiniteLine(pos=200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine))
    self.peak_th_line_neg = InfiniteLine(pos=-200.0, angle=0, movable=True, pen=pg.mkPen('c', width=1.5, style=QtCore.Qt.DashLine))
    self.plot.addItem(self.peak_th_line_pos)
    self.plot.addItem(self.peak_th_line_neg)

    # --- NUEVO: Texto de Cuenta Regresiva ---
    self.countdown_text = pg.TextItem(text="", color=(0, 255, 255), anchor=(0.5, 0.5))
    font = QtGui.QFont()
    font.setPixelSize(80) # Un poco más pequeño para evitar clipping OpenGL
    font.setBold(True)
    self.countdown_text.setFont(font)
    self.countdown_text.setZValue(10000) # Traer al frente
    self.plot.addItem(self.countdown_text)
    self.countdown_text.hide()

  def _setup_ui_final_layout(self):
    
    # --- NUEVO: Divisor para los gráficos ---
    """
    Ejecuta la funcionalidad de _setup_ui_final_layout.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
    self.splitter.addWidget(self.plot_container)
    self.splitter.addWidget(self.spectrogram_view)
    plot_h = int(self.screen_h * 0.45) if hasattr(self, 'screen_h') else 350
    spec_h = int(self.screen_h * 0.15) if hasattr(self, 'screen_h') else 120
    self.splitter.setSizes([plot_h, spec_h]) # Proporcional a la pantalla

    # --- NUEVO: QStackedWidget para ocultar Configuración durante grabación ---
    self.config_stack = QtWidgets.QStackedWidget()
    self.config_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    min_config_h = 120 if getattr(self, 'is_compact_screen', False) else 150
    self.config_groupbox.setMinimumHeight(min_config_h)
    self.config_stack.setMinimumHeight(min_config_h)
    self.config_stack.addWidget(self.config_groupbox)
    
    self.empty_recording_widget = QtWidgets.QWidget()
    self.empty_recording_widget.setStyleSheet("background-color: #050505;")
    
    lbl_font_size = "30px" if getattr(self, 'is_compact_screen', False) else "42px"
    self.lbl_recording_space = QtWidgets.QLabel("VENTANAS EXTERNAS ACTIVAS", self.empty_recording_widget)
    self.lbl_recording_space.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_recording_space.setWordWrap(True)
    self.lbl_recording_space.setStyleSheet(f"background-color: #050510; color: #00FF00; font-family: 'Courier New', monospace; font-size: {lbl_font_size}; font-weight: 900; border: 2px dashed #FF0055; padding: 6px;")
    
    # En lugar de usar un layout que puede forzar anchos mínimos, usamos un layout con restricciones elásticas
    self.empty_recording_layout = QtWidgets.QHBoxLayout(self.empty_recording_widget)
    self.empty_recording_layout.setContentsMargins(6, 0, 6, 0)
    self.empty_recording_layout.addWidget(self.lbl_recording_space, stretch=4)
    self.empty_recording_layout.addWidget(self.right_side_panel, stretch=1) # Añadir fijo al recuadro de arriba
    
    self.empty_recording_widget.setLayout(self.empty_recording_layout)
    self.empty_recording_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
    
    self.config_stack.addWidget(self.empty_recording_widget)
    
    # --- NUEVO: Doble Ubicación del Metrónomo ---
    self.config_stack.currentChanged.connect(self._reubicar_metronomo_dinamico)
    # Ubicación Inicial
    self._reubicar_metronomo_dinamico(0)

    # --- Añadir layouts a la ventana ---
    self.main_layout.addWidget(self.config_stack) # Reemplaza a config_groupbox
    self.main_layout.addWidget(self.filter_groupbox)
    self.main_layout.addLayout(self.button_layout)
    self.main_layout.addWidget(self.notas_panel) # Panel desplegable de Notas del Experimento
    self.main_layout.addLayout(self.trigger_layout)
    self.main_layout.addWidget(self.measure_widget)
    self.main_layout.addWidget(self.spectrogram_groupbox) # Añadir controles del espectrograma
    self.main_layout.addWidget(self.splitter) # Añadir el divisor con los gráficos
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
    self._actualizar_canales_activos(update_gui=False)

    # Conectar señal de vista previa de gráfico
    self.mostrar_preview_signal.connect(self.mostrar_preview_plot)

  def _safe_autoforge_single_shot(self, delay_ms, callback):
    """
    Ejecuta callback tras delay_ms sólo si la sesión actual de AutoForge sigue activa
    y coincide con el token de sesión _autoforge_token. Si AutoForge fue cancelado o reiniciado,
    invalida limpiamente la ejecución del timer pendiente.
    """
    current_token = getattr(self, '_autoforge_token', 0)
    def wrapper():
      if getattr(self, '_autoforge_token', 0) == current_token and getattr(self, 'is_autoforge_running', False):
        callback()
    QtCore.QTimer.singleShot(delay_ms, wrapper)

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
    try:
      dialog = PreviewPlotDialog(ruta_img, self, duration_ms=5000)
      dialog.show()
      self._preview_dialog = dialog
    except Exception as e:
      print(f"Error al mostrar vista previa de gráfico: {e}")

  # --- NUEVO: Cambio de modo de conexión en tiempo real ---
  def on_terminal_mode_changed(self):
    """
    Ejecuta la funcionalidad de on_terminal_mode_changed.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
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

  def _on_modo_prueba_toggled(self, checked):
    """Se llama cuando el checkbox de Modo Prueba cambia."""
    self.cmb_device.setEnabled(not checked)
    self.cmb_terminal_config.setEnabled(not checked)
    # --- NUEVO: Habilitar/deshabilitar control de frecuencia ---
    self.cmb_fuente_prueba.setEnabled(checked)
    self.label_test_freq.setEnabled(checked)
    self.spin_test_freq.setEnabled(checked)
    print(f"Modo Prueba {'Activado' if checked else 'Desactivado'}.")

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
    self.measure_widget.setVisible(enabled)
    self.trigger_line.setVisible(enabled and self.chk_trigger.isChecked())
    self.spectrogram_groupbox.setVisible(enabled)
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

    # Botones superiores siempre habilitados (Notas, Fotometría 3D y Test 5 Vocales)
    if hasattr(self, 'btn_notas'):
      self.btn_notas.setEnabled(True)
    if hasattr(self, 'btn_fotometria'):
      self.btn_fotometria.setEnabled(True)
    if hasattr(self, 'btn_test_vocales'):
      self.btn_test_vocales.setEnabled(True)

  def toggle_notas_panel(self):
    """Muestra u oculta el panel de Notas del Experimento."""
    if hasattr(self, 'notas_panel'):
      visible = not self.notas_panel.isVisible()
      self.notas_panel.setVisible(visible)
      if visible and hasattr(self, 'txt_notas_live'):
        self.txt_notas_live.setFocus()

  def _insertar_plantilla_nota(self, texto: str):
    """Inserta una plantilla formateada en el editor de notas en vivo."""
    if not hasattr(self, 'txt_notas_live'):
      return
    cur = self.txt_notas_live.toPlainText().strip()
    nuevo = f"{cur}\n{texto}".strip() if cur else texto
    self.txt_notas_live.setPlainText(nuevo)
    self.notas_experimento = nuevo

  def on_actualizar_notas_ultima_medicion(self):
    """Actualiza las notas en metadata.json y notas_experimento.txt de la última medición."""
    target_dir = getattr(self, 'last_measurement_dir', None)
    if not target_dir or not os.path.exists(target_dir):
      QtWidgets.QMessageBox.warning(
        self, "Sin Medición",
        "Aún no se ha completado ni guardado ninguna medición en esta sesión."
      )
      return

    notas = self.txt_notas_live.toPlainText().strip() if hasattr(self, 'txt_notas_live') else ""
    exito = self.persistir_notas_experimento(target_dir, texto_notas=notas)
    if exito:
      QtWidgets.QMessageBox.information(
        self, "Notas Actualizadas",
        f"Las notas se han actualizado correctamente en la carpeta de la medición:\n{target_dir}"
      )
    else:
      QtWidgets.QMessageBox.critical(
        self, "Error al Actualizar",
        f"No se pudieron actualizar las notas en:\n{target_dir}"
      )

  def persistir_notas_experimento(self, target_dir: str, metadata_base: Optional[dict] = None, texto_notas: Optional[str] = None) -> bool:
    """
    Persiste las notas en:
      - metadata.json raíz y canal_*/metadata.json (claves 'notas_experimento' y 'comentario')
      - notas_experimento.txt legible con cabecera de fecha, sujeto, canales/músculos, SNR y ruido 50 Hz.
    """
    try:
      if not target_dir or not os.path.exists(target_dir):
        return False

      self.last_measurement_dir = str(target_dir)
      if texto_notas is None:
        texto_notas = self.txt_notas_live.toPlainText().strip() if hasattr(self, 'txt_notas_live') else ""
      self.notas_experimento = texto_notas

      # 1. metadata.json raíz
      root_meta_path = os.path.join(target_dir, "metadata.json")
      meta_dict = {}
      if os.path.exists(root_meta_path):
        try:
          with open(root_meta_path, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        except Exception:
          meta_dict = {}
      elif metadata_base:
        meta_dict = dict(metadata_base)

      meta_dict["notas_experimento"] = texto_notas
      meta_dict["comentario"] = texto_notas
      if getattr(self, 'last_separability_metrics', None):
        meta_dict["diagnostico_separabilidad_vocales"] = self.last_separability_metrics

      try:
        with open(root_meta_path, 'w', encoding='utf-8') as f:
          json.dump(meta_dict, f, indent=4, ensure_ascii=False)
      except Exception as exc:
        print(f"[Notas] Aviso al escribir metadata.json raíz: {exc}")

      # 2. canal_*/metadata.json
      for i in range(getattr(self, 'NUM_CANALES', 4)):
        ch_dir = os.path.join(target_dir, f"canal_{i}")
        if os.path.isdir(ch_dir):
          ch_meta_path = os.path.join(ch_dir, "metadata.json")
          ch_dict = {}
          if os.path.exists(ch_meta_path):
            try:
              with open(ch_meta_path, 'r', encoding='utf-8') as f:
                ch_dict = json.load(f)
            except Exception:
              ch_dict = {}
          else:
            ch_dict = dict(meta_dict)
            ch_dict["canal"] = f"canal_{i}"
            ch_dict["musculo"] = self.nombres_musculos[i % len(self.nombres_musculos)] if hasattr(self, 'nombres_musculos') else f"Canal {i}"

          ch_dict["notas_experimento"] = texto_notas
          ch_dict["comentario"] = texto_notas
          if getattr(self, 'last_separability_metrics', None):
            ch_dict["diagnostico_separabilidad_vocales"] = self.last_separability_metrics

          try:
            with open(ch_meta_path, 'w', encoding='utf-8') as f:
              json.dump(ch_dict, f, indent=4, ensure_ascii=False)
          except Exception as exc:
            print(f"[Notas] Aviso al escribir canal_{i}/metadata.json: {exc}")

      # 3. notas_experimento.txt legible
      txt_path = os.path.join(target_dir, "notas_experimento.txt")
      sujeto = meta_dict.get("sujeto", getattr(self, 'autoforge_sujeto', "Sujeto1"))
      prueba = meta_dict.get("prueba", getattr(self, 'autoforge_prueba', "Prueba1"))
      fecha = meta_dict.get("measurement_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

      canales_lines = []
      for i in range(getattr(self, 'NUM_CANALES', 4)):
        hw_idx = self.active_channel_indices[i] if hasattr(self, 'active_channel_indices') and i < len(self.active_channel_indices) else i
        musc = self.nombres_musculos[i % len(self.nombres_musculos)] if hasattr(self, 'nombres_musculos') else f"Canal {hw_idx}"
        canales_lines.append(f"  • Canal {i} (ai{hw_idx}): {musc}")

      noise_50hz = meta_dict.get("ruido_basal", {}).get("inicial", {}).get("line_noise_50hz_uV", getattr(self, 'line_noise_50hz_uV', []))
      ruido_mean = meta_dict.get("ruido_basal", {}).get("inicial", {}).get("mean_uV", getattr(self, 'initial_noise_mean', []))

      snr_info = []
      if hasattr(self, 'stats_snr') and self.stats_snr:
        for i, snrs in enumerate(self.stats_snr):
          val = snrs[-1] if snrs else 0.0
          snr_info.append(f"Ch{i}: {val:.1f} dB")
      snr_str = ", ".join(snr_info) if snr_info else "No calculado"

      lines_txt = [
        "=" * 70,
        "NOTAS DEL EXPERIMENTO - NANDU DAQ",
        "=" * 70,
        f"Fecha / Hora       : {fecha}",
        f"Sujeto             : {sujeto}",
        f"Prueba / Protocolo : {prueba}",
        "",
        "--- ASIGNACION DE CANALES Y MUSCULOS ---",
        *canales_lines,
        "",
        "--- CALIDAD DE SENAL Y RUIDO ---",
        f"Ruido 50 Hz (uV)   : {noise_50hz}",
        f"Ruido Basal Media  : {ruido_mean}",
        f"SNR Estimado       : {snr_str}",
        "-" * 70,
        "OBSERVACIONES Y NOTAS REGISTRADAS:",
        texto_notas if texto_notas else "(Sin observaciones adicionales registradas)",
        "=" * 70,
      ]
      with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines_txt) + "\n")

      return True
    except Exception as exc:
      print(f"[Notas] Error persistiendo notas del experimento: {exc}")
      return False

  def abrir_asistente_fotometria(self):
    """Abre el asistente interactivo de posicionamiento facial y alineación 3D de electrodos."""
    if getattr(self, 'photometry_mgr', None) is not None:
      self.photometry_mgr.open_alignment_dialog(
        parent_widget=self,
        nombres_musculos=getattr(self, 'nombres_musculos', []),
        active_channel_indices=getattr(self, 'active_channel_indices', None),
        sujeto=getattr(self, 'autoforge_sujeto', "Sujeto1"),
        prueba=getattr(self, 'autoforge_prueba', "Prueba1"),
      )
    else:
      QtWidgets.QMessageBox.warning(
        self, "Fotometria No Disponible",
        "El módulo WebcamPhotometryManager no está disponible en este sistema."
      )

  def abrir_test_separabilidad_5_vocales(self):
    """Abre el tester de separabilidad multiclase de 5 vocales y detector de Crosstalk."""
    dialog = TestSeparabilidadVocalesDialog(self)
    dialog.exec()

  def on_start_acq_click(self):
    """
    Ejecuta la funcionalidad de on_start_acq_click.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if self.is_acquiring:
      # --- DETENER ADQUISICIÓN ---
      print("Deteniendo adquisición...")
      self.stop_event.set()
      
      # --- NUEVO: Abortar AutoForge o grabación activa ---
      self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
      self.is_autoforge_running = False
      self.is_autoforge_continuo = False
      self.reset_autoforge_buttons() # Restaurar aspecto de botones
      if hasattr(self, 'autoforge_overlay'):
        self.autoforge_overlay.hide()
      if getattr(self, 'is_recording', False):
        self.on_record_click()
      
      # --- MEJORA: Detener el metrónomo solo si el proceso todavía existe ---
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

      self.active_channel_indices = []
      for ch in selected_channels:
        m = re.search(r'\d+', ch)
        self.active_channel_indices.append(int(m.group()) if m else 0)

      self.CANALES_DAQ = [f"{device}/{ch}" for ch in selected_channels]
      self.NUM_CANALES = len(self.CANALES_DAQ)
      self._actualizar_canales_activos(update_gui=False)
      
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
      while not self.data_queue.empty(): self.data_queue.get_nowait()

      # Limpiar y re-crear elementos de la GUI
      self.setup_gui_for_channels()

      # --- NUEVO: Lanzar el metrónomo si está seleccionado ---
      if self.chk_use_metronome.isChecked():
        self._save_metronome_config() # Guardar el BPM actual antes de lanzar
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
      else:
        if not NIDAQMX_DISPONIBLE:
          print("Error: nidaqmx no encontrado. No se puede correr en modo real.")
          return
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
      self.spin_plot_duration.setEnabled(False)
      self.chk_use_metronome.setEnabled(False)
      for chk in self.channel_checkboxes: chk.setEnabled(False)
      
      self.set_controls_enabled(True)
      self.on_autoscroll_toggle() # Restaura el estado del mouse según el Auto-scroll

  def _save_metronome_config(self):
    """Guarda el valor actual de BPM en el archivo de configuración del metrónomo."""
    try:
      current_bpm = self.spin_bpm.value()
      # Leemos la configuración existente para no perder el `last_beat_count`.
      config_data = {}
      if os.path.exists(METRONOME_CONFIG_PATH):
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          config_data = json.load(f) # Asegurar que se lee como UTF-8
      
      config_data['last_bpm'] = current_bpm
      with open(METRONOME_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
      print(f"Actualizado BPM en metronome_config.json a: {current_bpm}")
    except Exception as e:
      print(f"Advertencia: No se pudo guardar el BPM para el metrónomo. Error: {e}")

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
      
    self.plot.getViewBox().setLimits(xMin=-self.PLOT_DURATION_S) # Actualizar límite de zoom
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
      curva_nombre = f"ai{hw_idx} ({musculo})"
      self.curvas.append(self.plot.plot(pen=color, name=curva_nombre))
      
      # --- NUEVO: Checkbox de visualización ---
      chk_viz = QtWidgets.QCheckBox(curva_nombre)
      chk_viz.setChecked(True)
      chk_viz.setStyleSheet(f"color: {pg.mkColor(color).name()}; font-weight: bold; border: none;")
      
      def toggle_viz(state, idx=i):
        if idx < len(self.curvas):
          self.curvas[idx].setVisible(state)
          
      chk_viz.toggled.connect(toggle_viz)
      self.viz_checkbox_layout.addWidget(chk_viz)
      self.viz_checkboxes.append(chk_viz)
      
      # --- NUEVO: Línea de piso de ruido ---
      line_ruido = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=4, style=QtCore.Qt.DashLine))
      line_ruido.hide()
      self.plot.addItem(line_ruido)
      self.noise_lines.append(line_ruido)
      
      line_ruido_n = pg.InfiniteLine(angle=0, pen=pg.mkPen('r', width=4, style=QtCore.Qt.DashLine))
      line_ruido_n.hide()
      self.plot.addItem(line_ruido_n)
      self.noise_lines_neg.append(line_ruido_n)
      
      region_ruido = pg.LinearRegionItem(orientation='horizontal', brush=pg.mkBrush(255, 0, 0, 40), movable=False)
      region_ruido.lines[0].setPen(pg.mkPen(None)) # Ocultar bordes propios
      region_ruido.lines[1].setPen(pg.mkPen(None))
      region_ruido.hide()
      self.plot.addItem(region_ruido)
      self.noise_regions.append(region_ruido)
      
      # --- NUEVO: Marcador visual para el pico máximo (SNR) ---
      scatter = pg.ScatterPlotItem(size=12, pen=pg.mkPen('w', width=1.5), brush=pg.mkBrush(color))
      scatter.setVisible(self.chk_show_peaks.isChecked())
      self.plot.addItem(scatter)
      self.peak_scatters.append(scatter)
      
      # --- NUEVO: Región dinámica de ruido inter-pulso ---
      dyn_region = pg.LinearRegionItem(orientation='horizontal', brush=pg.mkBrush(0, 255, 0, 40), movable=False)
      dyn_region.lines[0].setPen(pg.mkPen(None))
      dyn_region.lines[1].setPen(pg.mkPen(None))
      dyn_region.setVisible(self.chk_show_noise.isChecked())
      self.plot.addItem(dyn_region)
      self.dynamic_noise_regions.append(dyn_region)

      # --- NUEVO: Crear un QFrame para cada medición ---
      measurement_frame = QtWidgets.QFrame()
      measurement_frame.setStyleSheet("background-color: #050505; border: 2px solid #00FFFF; border-radius: 5px;")
      measurement_frame.setMinimumWidth(100)
      
      frame_layout = QtWidgets.QVBoxLayout(measurement_frame)
      frame_layout.setContentsMargins(5, 5, 5, 5) # Padding interno
      
      label = QtWidgets.QLabel(f"<b>ai{hw_idx} ({musculo}):</b> -- µVp-p, -- µVrms (--% Dom)")
      label.setStyleSheet(f"color: {pg.mkColor(color).name()}; font-size: 14px; font-weight: bold; background-color: transparent; border: none;")
      label.setWordWrap(True)
      label.setMinimumWidth(0)
      frame_layout.addWidget(label)
      
      # --- NUEVO: Etiqueta para el tester de ruido ---
      label_ruido = QtWidgets.QLabel(f"ai{hw_idx} Ruido: Esperando...")
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
      label_50hz = QtWidgets.QLabel(f"50Hz ai{hw_idx} ({musculo}): -- µV")
      label_50hz.setStyleSheet("color: #AAAAAA; font-size: 11px; font-weight: bold;")
      label_50hz.setWordWrap(True)
      layout_50hz.addWidget(label_50hz)
      
      if hasattr(self, 'metro_noise_layout'):
          self.metro_noise_layout.addWidget(frame_50hz)
      else:
          frame_layout.addWidget(frame_50hz)
          
      self.line_noise_status_frames.append(frame_50hz)
      self.line_noise_status_labels.append(label_50hz)
      
      self.measure_layout.addWidget(measurement_frame)
      self.measure_labels.append(label) # Guardar solo el label para actualizar su texto
    self.measure_layout.addStretch(1) # Asegurar que los recuadros se alineen a la izquierda

    for i in range(self.NUM_CANALES):
      hw_idx = self.active_channel_indices[i] if (hasattr(self, 'active_channel_indices') and self.active_channel_indices and i < len(self.active_channel_indices)) else i
      musc = self.nombres_musculos[i] if (hasattr(self, 'nombres_musculos') and i < len(self.nombres_musculos)) else f"Canal {hw_idx}"
      self.cmb_trig_chan.addItem(f"ai{hw_idx} ({musc})")
      self.cmb_spectrogram_chan.addItem(f"ai{hw_idx} ({musc})")
    
    # Asegurar parámetros de sample rate y buffer válidos
    if getattr(self, 'SAMPLE_RATE', 0) <= 0:
      if hasattr(self, 'cmb_sample_rate') and self.cmb_sample_rate.currentText():
        try: self.SAMPLE_RATE = int(self.cmb_sample_rate.currentText())
        except Exception: self.SAMPLE_RATE = 2000
      else:
        self.SAMPLE_RATE = 2000

    if getattr(self, 'PLOT_DURATION_S', 0) <= 0:
      if hasattr(self, 'spin_plot_duration') and self.spin_plot_duration.value() > 0:
        self.PLOT_DURATION_S = self.spin_plot_duration.value()
      else:
        self.PLOT_DURATION_S = 3.0

    self.PLOT_SAMPLES = max(1, int(self.PLOT_DURATION_S * self.SAMPLE_RATE))

    # Inicializar buffers con el tamaño correcto
    self.plot_buffer_datos = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES))
    self.env_buffer_datos = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES))
    self.plot_vector_tiempo = np.linspace(-self.PLOT_DURATION_S, 0, self.PLOT_SAMPLES)
    self.trigger_last_values = np.zeros(self.NUM_CANALES)

    # Inicializar buffer del espectrograma
    chunk_samples_dinamico = max(2, int(self.SAMPLE_RATE * 0.05))
    self.CURRENT_FFT_LEN = min(self.SPECTROGRAM_FFT_LEN, chunk_samples_dinamico)
    if self.CURRENT_FFT_LEN < 2: self.CURRENT_FFT_LEN = 2
    if self.CURRENT_FFT_LEN % 2 != 0: self.CURRENT_FFT_LEN -= 1
    self.spectrogram_buffer = np.zeros((self.SPECTROGRAM_HISTORY_LEN, self.CURRENT_FFT_LEN // 2 + 1))
    self.on_spectrogram_enable_toggle() # Para mostrar/ocultar la vista

    # --- CORRECCIÓN v3: Configurar la transformación del espectrograma una sola vez ---
    view = self.spectrogram_view.getView()
    
    # Eje Y (Frecuencia): Va de 0 a Frecuencia de Nyquist.
    nyquist = self.SAMPLE_RATE / 2.0
    freq_scale = nyquist / (self.CURRENT_FFT_LEN / 2 + 1)

    # Eje X (Tiempo): El ancho total del historial del espectrograma en segundos.
    # Cada columna representa un segmento de tiempo (nperseg - noverlap) / fs.
    time_per_column = (self.CURRENT_FFT_LEN - (self.CURRENT_FFT_LEN // 2)) / max(1.0, float(self.SAMPLE_RATE))
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
    for reg in getattr(self, 'noise_regions', []): reg.setVisible(show)
    for reg in getattr(self, 'dynamic_noise_regions', []): reg.setVisible(show)
    for line in getattr(self, 'noise_lines', []): line.setVisible(show)
    for line in getattr(self, 'noise_lines_neg', []): line.setVisible(show)
    for lbl in getattr(self, 'noise_status_labels', []): lbl.setVisible(show)

  def toggle_peak_scatter(self):
    show = self.chk_show_peaks.isChecked()
    for scatter in getattr(self, 'peak_scatters', []): scatter.setVisible(show)

  def _reubicar_metronomo_dinamico(self, index):
    """
    Maneja la doble ubicación del metrónomo y panel derecho dependiendo del estado de la UI
    (Reposo vs Grabando).
    """
    if not hasattr(self, 'right_side_panel') or self.right_side_panel is None:
      return
    if index == 0:
      self.config_layout.addWidget(self.right_side_panel, 0, 9, 4, 1)
    else:
      self.empty_recording_layout.addWidget(self.right_side_panel, stretch=1)

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
    self.metronome_container.setMinimumWidth(144)
    metro_layout = QtWidgets.QVBoxLayout(self.metronome_container)
    
    self.metro_pulse_frame = QtWidgets.QFrame()
    self.metro_pulse_frame.setFixedHeight(54)
    self.metro_pulse_frame.setStyleSheet("background-color: #111111; border: 2px solid #00FFFF;")
    metro_layout.addWidget(self.metro_pulse_frame)

    self.metro_lbl_title = QtWidgets.QLabel("PULSO")
    self.metro_lbl_title.setAlignment(QtCore.Qt.AlignCenter)
    self.metro_lbl_title.setStyleSheet("font-size: 14px; color: #00FFFF; font-family: 'Courier New', monospace; font-weight: bold;")
    metro_layout.addWidget(self.metro_lbl_title)

    self.metro_lbl_count = QtWidgets.QLabel("0")
    self.metro_lbl_count.setAlignment(QtCore.Qt.AlignCenter)
    self.metro_lbl_count.setStyleSheet("font-size: 42px; color: #00FFFF; font-weight: bold; font-family: 'Courier New', monospace;")
    metro_layout.addWidget(self.metro_lbl_count)
    metro_layout.addStretch()
    
    self.right_side_layout.addWidget(self.metronome_container)
    
    self.noise_status_group = QtWidgets.QWidget()
    self.metro_noise_layout = QtWidgets.QVBoxLayout(self.noise_status_group)
    self.metro_noise_layout.setContentsMargins(0, 5, 0, 0)
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
    self.metro_timer.timeout.connect(self.on_metro_beat)
    
    # Hacer que el metrónomo sea visible por defecto
    self.metronome_container.show()
    self.metro_is_running = False
    self.metro_count_in_remaining = 0
    self.metro_beat_count = 0
    
    self.COLOR_IDLE = "#111111"
    self.COLOR_BEAT = "#00FFFF"
    self.COLOR_PREP = "#FF0000"

    self.metronome_container.setVisible(self.chk_use_metronome.isChecked())
    self.chk_use_metronome.toggled.connect(self.metronome_container.setVisible)
    self.right_side_panel.show()

  def start_native_metronome(self, count_in=0, force_start=False):
    if not force_start and not self.chk_use_metronome.isChecked():
        return
    self.metro_count_in_remaining = count_in
    self.metro_is_running = True
    self.metro_beat_count = 0
    self.metro_subdiv_index = 0
    self.on_metro_beat()

  def save_metronome_config(self):
    config = {
        "last_bpm": self.spin_bpm.value(),
        "last_beat_count": str(getattr(self, 'metro_beat_count', 0)),
        "subdivisions": self.spin_metro_subdivs.value()
    }
    try:
        with open(METRONOME_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)
    except: pass

  def stop_native_metronome(self):
    self.metro_is_running = False
    self.metro_timer.stop()
    self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_IDLE}; border: 2px solid #00FFFF;")
    self.metro_lbl_count.setText("0")
    self.metro_lbl_title.setText("PULSO")
    self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #00FFFF;")
    self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #00FFFF; font-weight: bold;")
    self.save_metronome_config()

  def reset_metro_color(self):
    self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_IDLE}; border: 2px solid #00FFFF;")

  def on_metro_beat(self):
    if not self.metro_is_running:
        return

    bpm = self.spin_bpm.value()
    num_subdivs = self.spin_metro_subdivs.value()
    interval_ms = int(60000 / bpm) if bpm > 0 else 1000
    sub_interval_ms = interval_ms // num_subdivs
    
    self.metro_timer.start(sub_interval_ms)

    is_main_beat = (self.metro_subdiv_index == 0)

    if is_main_beat:
        if self.metro_count_in_remaining > 1:
            self.metro_lbl_title.setText("INICIANDO")
            self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #FF0000;")
            self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #FF0000; font-weight: bold;")
            self.metro_lbl_count.setText(str(self.metro_count_in_remaining - 1))
            self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_PREP}; border: 2px solid #FF0000;")
            self.metro_count_in_remaining -= 1
            self.beep_queue.put((800, 200))
        else:
            self.metro_lbl_title.setText("PULSO")
            self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #00FFFF;")
            self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #00FFFF; font-weight: bold;")
            
            if self.metro_count_in_remaining == 1:
                self.metro_beat_count = 1
                self.metro_count_in_remaining = 0
                self.beep_queue.put((1200, 500))
            else:
                self.metro_beat_count += 1
                self.beep_queue.put((1000, 100))
            
            self.metro_lbl_count.setText(str(self.metro_beat_count))
            self.metro_pulse_frame.setStyleSheet(f"background-color: {self.COLOR_BEAT}; border: 2px solid #00FFFF;")
            
            # --- NUEVO: Integración de Calibración CVM con Metrónomo ---
            if getattr(self, 'is_cvm_active', False):
                if self.metro_beat_count > 18:
                    if self.is_cvm_recording_take:
                        self.estado_cvm_procesar_toma()
                    self.is_cvm_active = False
                    self.stop_native_metronome()
                    self.estado_finalizar_calibracion_cvm()
                else:
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
                            f"<span style='color:#FFFF00; font-size:40px; font-weight:bold;'>¡PREPÁRATE PARA /{vocal}/!</span><br>"
                            f"<span style='font-size:22px; color:#AAFFAA; font-weight:bold;'>{desc}</span>"
                            f"</div>"
                        )
                        self.lbl_recording_space.setText("<div align='center'><span style='color:#FFFF00; font-size:40px; font-weight:bold;'>PREPARANDO...</span></div>")
                        self.autoforge_overlay.show()
                        
                    else:
                        # PAR: Fase de Grabación (Toma)
                        vocal, toma = self.cvm_sequence[self.cvm_step_idx]
                        self.autoforge_overlay.setText(
                            f"<div align='center'>"
                            f"<span style='color:#FF0000; font-size:32px; font-weight:bold;'>¡CONTRAE AHORA!</span><br>"
                            f"<span style='color:#00FFFF; font-size:90px; font-weight:bold;'>/{vocal}/</span><br>"
                            f"</div>"
                        )
                        self.lbl_recording_space.setText(f"<div align='center'><span style='color:#FF0000; font-size:70px; font-weight:bold;'>¡/{vocal}/!</span></div>")
                        self.autoforge_overlay.show()
                        
                        self.cvm_current_take_chunks = []
                        self.is_cvm_recording_take = True
                        if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
                            t_daq = time.perf_counter() - getattr(self, 'recording_start_time', time.perf_counter())
                            self.photometry_mgr.request_snapshot(fase="cvm", palabra=vocal, pulso=toma, t_daq_s=t_daq, sample_rate=self.SAMPLE_RATE)

            # --- NUEVO: Intercalar Letras dinámicamente en Secuencia Continua ---
            if getattr(self, 'is_autoforge_continuo', False) and getattr(self, 'is_recording', False):
                total_pulsos = len(self.autoforge_words) * self.autoforge_target_reps
                if self.metro_beat_count > total_pulsos:
                    if not getattr(self, 'is_finishing_sequence', False):
                        self.is_finishing_sequence = True
                        self.estado_guardar_secuencia_continua()
                else:
                    idx = (self.metro_beat_count - 1) % len(self.autoforge_words)
                    palabra_actual = self.autoforge_words[idx].upper()
                    self.lbl_recording_space.setText(f"<div align='center' style='font-size: 100px; font-weight: bold;'>{palabra_actual}</div>")
                    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
                        t_daq = time.perf_counter() - getattr(self, 'recording_start_time', time.perf_counter())
                        self.photometry_mgr.request_snapshot(fase="vocal_metronomo", palabra=palabra_actual, pulso=self.metro_beat_count, t_daq_s=t_daq, sample_rate=self.SAMPLE_RATE)

            # --- NUEVO: Snapshot de fotometria para grabacion discreta o manual ---
            if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
                if not getattr(self, 'is_cvm_active', False) and not getattr(self, 'is_autoforge_continuo', False) and getattr(self, 'is_recording', False):
                    t_daq = time.perf_counter() - getattr(self, 'recording_start_time', time.perf_counter())
                    if getattr(self, 'is_autoforge_running', False) and hasattr(self, 'autoforge_words') and self.autoforge_words:
                        pal = self.autoforge_words[self.autoforge_word_idx] if self.autoforge_word_idx < len(self.autoforge_words) else "vocal"
                        self.photometry_mgr.request_snapshot(fase="vocal_metronomo", palabra=pal, pulso=self.metro_beat_count, t_daq_s=t_daq, sample_rate=self.SAMPLE_RATE)
                    else:
                        self.photometry_mgr.request_snapshot(fase="vocal_metronomo", palabra="manual", pulso=self.metro_beat_count, t_daq_s=t_daq, sample_rate=self.SAMPLE_RATE)
        
        QtCore.QTimer.singleShot(50, self.reset_metro_color)
    elif self.metro_count_in_remaining == 0:
        # Sub-pulso sónico/visual ligero
        self.metro_pulse_frame.setStyleSheet("background-color: #00AAAA; border: 2px solid #00FFFF;")
        self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #005577; font-weight: bold;")
        self.metro_lbl_count.setText(str(self.metro_subdiv_index + 1))
        self.beep_queue.put((1600, 50))
        QtCore.QTimer.singleShot(50, self.reset_metro_color)
        
    self.metro_subdiv_index = (self.metro_subdiv_index + 1) % num_subdivs

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
      Q = 2.0  # Factor de calidad
      
      # --- CORRECCIÓN DE COMPATIBILIDAD ---
      # La versión de SciPy del usuario no soporta `output='sos'`.
      # Se genera el filtro en formato (b, a) y se convierte a SOS.
      b, a = signal.iirnotch(f0, Q, fs=self.SAMPLE_RATE)
      self.notch_sos = signal.tf2sos(b, a)

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
      self.plot_widget.setBackground('k') # Color de fondo negro
    else:
      # MODO ANÁLISIS (Congelado)
      self.plot.setMouseEnabled(x=True, y=True) # Zoom X e Y
      self.plot.getViewBox().enableAutoRange(pg.ViewBox.XAxis)
      self.plot.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis) # Auto-escala todo al congelar

  def on_record_click(self):
    """
    Ejecuta la funcionalidad de on_record_click.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if not self.is_recording:
      # --- EMPEZAR A GRABAR ---
      self.is_recording = True
      self.recording_start_time = time.perf_counter()
      self.btn_record.setText("Detener Grabación")
      self.btn_record.setStyleSheet(self.BTN_REC_STOP_STYLE)
      self.label_rec_time.setVisible(True)
      self.label_rec_time.setText("GRABANDO...")
      
      # --- Resetear el zoom en Y a rango base calibrado (500 µV) ---
      if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
        self.plot.setYRange(0.0, 500.0, padding=0)
      else:
        self.plot.setYRange(-500.0, 500.0, padding=0)
      self._held_plot_ymax = 500.0
      
      self.counting_started = False # Reiniciar la bandera de conteo
      self.current_recording.clear()
      self.cmb_terminal_config.setEnabled(False) # <-- NUEVO: Bloquear cambio en plena grabación
      
      # --- NUEVO: Reiniciar variables de ruido y buffers ---
      self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.noise_levels = [0.0] * self.NUM_CANALES
      self.noise_calculated = False
      self.noise_initialized = False # <-- NUEVO: Resetear estado
      self.countdown_active = False # Permitir lanzar el countdown de nuevo
      self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
      self.rt_last_noise_eval = 0.0
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
      
      self.config_stack.setCurrentIndex(1) # Ocultar configuración
      
      # --- NUEVO: Re-lanzar el metrónomo con count-in 4 compases ---
      if self.chk_use_metronome.isChecked():
        self.stop_native_metronome()
        self.start_native_metronome(count_in=4, force_start=True)

      # --- HOOK FOTOMETRIA: Inicio de sesion y captura onset ---
      if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
        self.photometry_mgr.start_take_session("manual", t0_perf=self.recording_start_time)
        self.photometry_mgr.request_snapshot(fase="onset", palabra="manual", pulso=1, t_daq_s=0.0, sample_rate=self.SAMPLE_RATE)

    else:
      # --- DETENER GRABAR ---
      self.is_recording = False
      self.btn_record.setText("Empezar a Grabar")
      self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)
      self.label_rec_time.setVisible(False)
      self.cmb_terminal_config.setEnabled(True)
      self.stop_native_metronome()
      self.config_stack.setCurrentIndex(0)

      if self.current_recording:
        self.on_export_click()

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

    # --- NUEVO: Guardar metadata.json con la fecha, hora, timestamp, músculos, ruido y CVM ---
    metadata = {
      "measurement_date": now.isoformat(),
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

    # Exportar Fotometria 3D si esta habilitada
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      try:
        foto_resumen = self.photometry_mgr.export_take_photometry(output_dir)
        if foto_resumen:
          metadata["fotometria_espacial"] = foto_resumen
      except Exception as exc:
        print(f"--- ERROR AL EXPORTAR FOTOMETRIA 3D ---\n{exc}")

    # Guardar metadata.json en la raíz de output_dir
    try:
      root_meta_path = os.path.join(output_dir, "metadata.json")
      with open(root_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
      print(f"  Metadata raíz guardado en: {root_meta_path}")
    except Exception as e:
      print(f"--- ERROR AL GUARDAR METADATA.JSON raíz en '{output_dir}' ---\n{e}")

    # Guardar metadata en la carpeta de cada canal con su respectivo músculo
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

    # 1. Guarda el .wav
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

    # 5. Persistir Notas del Experimento en metadata y notas_experimento.txt
    self.last_measurement_dir = output_dir
    self.persistir_notas_experimento(output_dir, metadata_base=metadata, texto_notas=comentario)

    print("--- EXPORTACIÓN FINALIZADA ---")

  def actualizar_plot(self):
    """
    Ejecuta la funcionalidad de actualizar_plot.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if not self.is_acquiring:
      return # No hacer nada si la adquisición no está activa



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

      # 3. Grabar si es necesario (GUARDAR DATOS CRUDOS EN VOLTIOS)
      if self.is_recording or getattr(self, 'is_recording_final_noise', False):
        self.current_recording.append(all_new_data.copy())

      # 4. Calibrar para la visualización y análisis en tiempo real
      
      # --- NUEVO: Imprimir el voltaje RAW de la placa 1 vez por medición para depurar ---
      if not getattr(self, '_raw_print_done', False) and not self.chk_modo_prueba.isChecked():
        print(f"\n[DEBUG DAQ] Voltaje RAW puro entregado por la placa (Canal 0): {all_new_data[0, 0]:.6f} V")
        self._raw_print_done = True

      # Calibración a microvoltios por canal físico
      if hasattr(self, 'factores_calibracion_hw') and hasattr(self, 'active_channel_indices'):
        calib_factors = np.array([
          self.factores_calibracion_hw[self.active_channel_indices[i]]
          if (i < len(self.active_channel_indices) and self.active_channel_indices[i] < len(self.factores_calibracion_hw) and self.factores_calibracion_hw[self.active_channel_indices[i]] > 0)
          else 495.0
          for i in range(self.NUM_CANALES)
        ]).reshape(-1, 1)
      else:
        calib_factors = 495.0
      raw_uV_data = (all_new_data / calib_factors) * 1000000.0
      processed_data = raw_uV_data.copy()

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
              
              self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
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
                  self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
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
                  
                  self.noise_lines[i].setPos(self.noise_levels[i])
                  self.noise_lines_neg[i].setPos(-self.noise_levels[i])
                  self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
                  show_noise = self.chk_show_noise.isChecked()
                  self.noise_lines[i].setVisible(show_noise)
                  self.noise_lines_neg[i].setVisible(show_noise)
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
          
      for i in range(self.NUM_CANALES):
        try:
          if is_rms:
            # Asegurar que no hay NaNs antes de decimar
            valid_env = np.nan_to_num(self.env_buffer_datos[i])
            x_dec, y_dec = decimate_min_max(self.plot_vector_tiempo, valid_env, max_points=3000)
            self.curvas[i].setData(x_dec, y_dec)
          else:
            valid_raw = np.nan_to_num(self.plot_buffer_datos[i])
            x_dec, y_dec = decimate_min_max(self.plot_vector_tiempo, valid_raw, max_points=3000)
            self.curvas[i].setData(x_dec, y_dec)
        except Exception as e:
          print(f"[ERROR RENDER CANAL {i}] {e}")
          pass # Evita que se congele toda la GUI si hay un error matemático
      
      if self.chk_autoscroll.isChecked():
        self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)
        
        # --- Auto-Escala Calibrada (Rango 500 µV - 700 µV con Envolvente RMS) ---
        if self.NUM_CANALES > 0 and self.plot_buffer_datos.size > 0:
          if is_rms:
            current_max = float(np.max(self.env_buffer_datos))
          else:
            current_max = float(np.max(np.abs(self.plot_buffer_datos)))

          # Rango base mínimo: 500 uV. Rango máximo permitido: 700 uV.
          # Si la señal supera 500 uV se expande hasta 700 uV.
          # Si supera 700 uV, la escala NO cambia más para no confundir a la persona.
          if current_max <= 500.0:
            target_max = 500.0
          elif current_max >= 700.0:
            target_max = 700.0
          else:
            target_max = current_max * 1.15
            if target_max > 700.0:
              target_max = 700.0

          # Mantener escala estable con peak-hold suave para evitar rebotes visuales
          current_held = getattr(self, '_held_plot_ymax', 500.0)
          if target_max > current_held:
            current_held = target_max
          elif target_max < current_held:
            current_held = max(target_max, current_held * 0.995)
          
          # Clamp estricto entre 500 y 700 uV
          current_held = max(500.0, min(700.0, current_held))
          self._held_plot_ymax = current_held

          # Aplicar rango al gráfico
          view_range = self.plot.getViewBox().viewRange()[1]
          if is_rms:
            # Envolvente RMS es no-negativa: de 0 a current_held uV
            if abs(view_range[1] - current_held) > 1.0 or view_range[0] < -5.0 or view_range[0] > 0.0:
              self.plot.setYRange(0.0, current_held, padding=0)
          else:
            # Señal cruda bipolar: de -current_held a +current_held uV
            if abs(view_range[1] - current_held) > 1.0 or abs(view_range[0] + current_held) > 1.0:
              self.plot.setYRange(-current_held, current_held, padding=0)
      self.check_for_trigger(processed_data, total_muestras_leidas)
      # --- NUEVO: Pasar env_buffer_datos a las mediciones para que el pico visual se evalúe sobre la envolvente RMS ---
      self.actualizar_mediciones(processed_data, self.env_buffer_datos)
      self.actualizar_espectrograma(processed_data)

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
              bpm = self.spin_bpm.value()
              if bpm <= 0: bpm = 60
              beat_interval_s = 60.0 / bpm
              current_beat = int(elapsed_time / beat_interval_s)
              
              if current_beat != getattr(self, 'last_continuo_beat', -1):
                self.last_continuo_beat = current_beat
                total_words = len(self.autoforge_words)
                target_total = total_words * self.autoforge_target_reps
                
                if current_beat >= target_total:
                  if not getattr(self, 'is_finishing_sequence', False):
                    self.is_finishing_sequence = True
                    self.estado_guardar_secuencia_continua()
                else:
                  palabra = self.autoforge_words[current_beat % total_words]
                  try:
                    if hasattr(self, 'word_window_process') and self.word_window_process:
                      self.word_window_process.stdin.write(f"{palabra}\n")
                      self.word_window_process.stdin.flush()
                  except: pass
                  
              self.label_rec_time.setText(f"GRABANDO SECUENCIA... {elapsed_time:.1f} s{cuadro_azul}")
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #aa00ff;")
            else:
              self.label_rec_time.setText(f"GRABANDO PALABRA... {elapsed_time:.1f} s{cuadro_azul}")
              self.label_rec_time.setStyleSheet("font-weight: bold; color: #FF00FF;") # Magenta neón
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

          period_s = 60.0 / max(1, self.spin_bpm.value())
          
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
                        brush_color = pg.mkBrush(0, 255, 0, 40) if ratio <= 1.20 else pg.mkBrush(255, 0, 0, 40)
                        self.dynamic_noise_regions[i].setRegion([-curr_mean, curr_mean])
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
        texto = f"<b>ai{hw_idx} ({musculo}):</b> {vp_p[i]:.1f} µVp-p, {rms[i]:.1f} µVrms{dom_str}"
        
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

  def reset_autoforge_buttons(self):
    """Restaura los botones de autoforge a su estado normal (Cyberpunk)."""
    self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
    self.is_autoforge_running = False
    self.is_autoforge_continuo = False
    self.is_cvm_active = False
    self.is_recording_final_noise = False
    self.btn_autoforge.setText(" AUTOGRABADO")
    self.btn_autoforge.setStyleSheet("""
      QPushButton {
        background-color: #1a001a; 
        color: #ff00ff; 
        font-weight: 900; 
        font-family: 'Consolas', 'Courier New', monospace; 
        font-size: 15px; 
        padding: 8px; 
        border-radius: 4px;
        border: 2px solid #ff00ff;
        border-right: 4px solid #00ffff;
        border-bottom: 4px solid #00ffff;
      }
      QPushButton:hover {
        background-color: #ff00ff; color: #000000; border: 2px solid #00ffff;
      }
    """)
    self.btn_autoforge_continuo.setText(" SECUENCIA CONTINUA")
    self.btn_autoforge_continuo.setStyleSheet("""
      QPushButton {
        background-color: #0d001a; 
        color: #aa00ff; 
        font-weight: 900; 
        font-family: 'Consolas', 'Courier New', monospace; 
        font-size: 15px; 
        padding: 8px; 
        border-radius: 4px;
        border: 2px solid #aa00ff;
        border-right: 4px solid #ff00ff;
        border-bottom: 4px solid #ff00ff;
      }
      QPushButton:hover {
        background-color: #aa00ff; color: #ffffff; border: 2px solid #ff00ff;
      }
    """)

  def iniciar_autoforge(self):
    """
    Ejecuta la funcionalidad de iniciar_autoforge.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    try:
      if getattr(self, 'is_autoforge_running', False):
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
        self.is_autoforge_running = False
        self.is_autoforge_continuo = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.is_recording = False
        self.reset_autoforge_buttons()
        self.autoforge_overlay.hide()
        self.config_stack.setCurrentIndex(0)
        if hasattr(self, 'session_timer'): self.session_timer.stop()
        self.stop_native_metronome()
        
        # Matar ventana de palabras
        if hasattr(self, 'word_window_process') and self.word_window_process:
          try:
            self.word_window_process.kill()
            self.word_window_process.wait(timeout=1)
          except: pass
          self.word_window_process = None
        return
        
      # --- NUEVO: Apagar metrónomo general si estaba corriendo para que no choquen ---
      self.stop_native_metronome()
      self.chk_use_metronome.setChecked(False) # Reflejar en UI

      if not self.is_acquiring:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
          # Disparar automáticamente la adquisición pero sin metrónomo doble
          old_state = self.chk_use_metronome.isChecked()
          self.chk_use_metronome.setChecked(False)
          self.on_start_acq_click()
          self.chk_use_metronome.setChecked(old_state)
        finally:
          QtWidgets.QApplication.restoreOverrideCursor()
        
      ruta_palabras = PALABRAS_TXT_PATH
      if not os.path.exists(ruta_palabras):
        with open(ruta_palabras, 'w', encoding='utf-8') as f:
          f.write("A\nE\nI\nO\nU\n")
        
      with open(ruta_palabras, 'r', encoding='utf-8') as f:
        self.autoforge_words = [line.strip() for line in f if line.strip()]
        
      if not self.autoforge_words:
        QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt está vacío.")
        return
        
      dialog = AutoForgeDialog(self)
      dialog.spin_reps.setValue(5)
      dialog.spin_bpm.setValue(40)
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
        bpm = dialog.spin_bpm.value()
        
        self.autoforge_do_cvm = dialog.chk_calibrar_cvm.isChecked()
        self.cvm_completed = False
        self.final_noise_completed = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.autoforge_session_dirs = []
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1

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

        # Actualizar el spinbox principal de BPM si existe
        try:
          self.spin_bpm.setValue(bpm)
          self._save_metronome_config() # Guardar archivo JSON con el BPM
        except:
          pass

        # Sincronizar Notas del Experimento
        texto_notas = dialog.edit_notas.toPlainText().strip()
        if hasattr(self, 'txt_notas_live') and texto_notas:
          self.txt_notas_live.setPlainText(texto_notas)
        self.notas_experimento = texto_notas

        # Sincronizar Fotometría y calibrar si está habilitada sin calibración previa
        if getattr(self, 'photometry_mgr', None):
          self.photometry_mgr.is_enabled = dialog.chk_fotometria.isChecked()
          self.photometry_mgr.record_video_clips = dialog.chk_fotometria_video.isChecked()
          if self.photometry_mgr.is_enabled and not self.photometry_mgr.calibracion_inicial_data:
            self.photometry_mgr.open_alignment_dialog(
              parent_widget=self,
              nombres_musculos=self.nombres_musculos,
              active_channel_indices=self.active_channel_indices,
              sujeto=self.autoforge_sujeto,
              prueba=self.autoforge_prueba,
            )
        
        self.autoforge_word_idx = 0
        # --- NUEVO: Mostrar Instrucciones del Experimento ---
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Instrucciones del Experimento")
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText(
            "<b>Bienvenido a la Sesión de Grabación (Discreta)</b><br><br>"
            "1. Relaja el rostro y evita moverte o tragar saliva.<br>"
            "2. Se medirán primero los niveles basales de ruido inter-pulso.<br>"
            "3. Se calibrará la Contracción Muscular Máxima (CVM) con vocales extremas (A, I, U) usando el metrónomo.<br>"
            "4. A continuación, el sistema te guiará palabra por palabra.<br>"
            "5. Al finalizar se registrará nuevamente el ruido basal."
        )
        btn_continuar = msg_box.addButton("Continuar Grabación", QtWidgets.QMessageBox.AcceptRole)
        btn_entrenar = msg_box.addButton("Abrir Módulo de Entrenamiento", QtWidgets.QMessageBox.ActionRole)
        msg_box.exec()
        
        if msg_box.clickedButton() == btn_entrenar:
            import subprocess, sys, os
            mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulo_de_entrenamiento.py")
            if os.path.exists(mod_path):
                subprocess.Popen([sys.executable, mod_path])
            return # Cancelar este inicio para que el usuario pueda entrenar

        self.is_autoforge_running = True
        self.btn_autoforge.setText(" Detener Grabación")
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
      if getattr(self, 'is_autoforge_running', False):
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1
        self.is_autoforge_running = False
        self.is_autoforge_continuo = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.is_recording = False
        self.reset_autoforge_buttons()
        self.autoforge_overlay.hide()
        self.config_stack.setCurrentIndex(0)
        if hasattr(self, 'session_timer'): self.session_timer.stop()
        self.stop_native_metronome()
        
        # Matar ventana de palabras
        if hasattr(self, 'word_window_process') and self.word_window_process:
          try:
            self.word_window_process.kill()
            self.word_window_process.wait(timeout=1)
          except: pass
          self.word_window_process = None
        return
        
      self.stop_native_metronome()

      if not self.is_acquiring:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
          old_state = self.chk_use_metronome.isChecked()
          self.chk_use_metronome.setChecked(False)
          self.on_start_acq_click()
          self.chk_use_metronome.setChecked(old_state)
        finally:
          QtWidgets.QApplication.restoreOverrideCursor()
        
      ruta_palabras = PALABRAS_TXT_PATH
      if not os.path.exists(ruta_palabras):
        with open(ruta_palabras, 'w', encoding='utf-8') as f:
          f.write("A\nE\nI\nO\nU\n")
        
      with open(ruta_palabras, 'r', encoding='utf-8') as f:
        self.autoforge_words = [line.strip() for line in f if line.strip()]
        
      if not self.autoforge_words:
        QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt está vacío.")
        return
        
      dialog = AutoForgeDialog(self)
      dialog.setWindowTitle("Configuración de Secuencia Continua")
      dialog.spin_reps.setValue(25) # 25 secuencias enteras por defecto
      dialog.spin_bpm.setValue(40)
      if dialog.exec() == QtWidgets.QDialog.Accepted:
        with open(ruta_palabras, 'r', encoding='utf-8') as f:
          self.autoforge_words = [line.strip() for line in f if line.strip()]
          
        if not self.autoforge_words:
          QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt quedó vacío después de editar.")
          return

        self.autoforge_prueba = dialog.edit_prueba.text().strip()
        self.autoforge_sujeto = dialog.edit_sujeto.text().strip()
        self.autoforge_target_reps = dialog.spin_reps.value() # Total cycles
        bpm = dialog.spin_bpm.value()
        
        self.autoforge_do_cvm = dialog.chk_calibrar_cvm.isChecked()
        self.cvm_completed = False
        self.final_noise_completed = False
        self.is_cvm_active = False
        self.is_recording_final_noise = False
        self.autoforge_session_dirs = []
        self._autoforge_token = getattr(self, '_autoforge_token', 0) + 1

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
          self._save_metronome_config()
        except:
          pass

        # Sincronizar Notas del Experimento
        texto_notas = dialog.edit_notas.toPlainText().strip()
        if hasattr(self, 'txt_notas_live') and texto_notas:
          self.txt_notas_live.setPlainText(texto_notas)
        self.notas_experimento = texto_notas

        # Sincronizar Fotometría y calibrar si está habilitada sin calibración previa
        if getattr(self, 'photometry_mgr', None):
          self.photometry_mgr.is_enabled = dialog.chk_fotometria.isChecked()
          self.photometry_mgr.record_video_clips = dialog.chk_fotometria_video.isChecked()
          if self.photometry_mgr.is_enabled and not self.photometry_mgr.calibracion_inicial_data:
            self.photometry_mgr.open_alignment_dialog(
              parent_widget=self,
              nombres_musculos=self.nombres_musculos,
              active_channel_indices=self.active_channel_indices,
              sujeto=self.autoforge_sujeto,
              prueba=self.autoforge_prueba,
            )
        
        self.autoforge_word_idx = 0
        # --- NUEVO: Mostrar Instrucciones del Experimento ---
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Instrucciones del Experimento")
        msg_box.setIcon(QtWidgets.QMessageBox.Information)
        msg_box.setText(
            "<b>Bienvenido a la Sesión de Grabación (Secuencia Continua)</b><br><br>"
            "1. Relaja el rostro y evita moverte o tragar saliva.<br>"
            "2. Se medirán primero los niveles basales de ruido inter-pulso.<br>"
            "3. Se calibrará la Contracción Muscular Máxima (CVM) con vocales extremas (A, I, U) usando el metrónomo.<br>"
            "4. A continuación, pronuncia las vocales mostradas en pantalla al ritmo del metrónomo. Mantén la fuerza constante.<br>"
            "5. Al finalizar se registrará nuevamente el ruido basal."
        )
        btn_continuar = msg_box.addButton("Continuar Grabación", QtWidgets.QMessageBox.AcceptRole)
        btn_entrenar = msg_box.addButton("Abrir Módulo de Entrenamiento", QtWidgets.QMessageBox.ActionRole)
        msg_box.exec()
        
        if msg_box.clickedButton() == btn_entrenar:
            import subprocess, sys, os
            mod_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modulo_de_entrenamiento.py")
            if os.path.exists(mod_path):
                subprocess.Popen([sys.executable, mod_path])
            return # Cancelar este inicio para que el usuario pueda entrenar

        self.is_autoforge_running = True
        self.is_autoforge_continuo = True
        self.last_continuo_beat = -1
        
        self.btn_autoforge_continuo.setText(" Detener Grabación")
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
    cvm_time = (22.0 * (60.0/bpm)) if getattr(self, 'autoforge_do_cvm', True) else 0.0
    tiempo_por_secuencia = 3.0 + noise_dur + cvm_time + (3.0 * (60.0/bpm)) + (total_words * (60.0/bpm)) + noise_dur + 2.0
    self.tiempo_restante_global = int(tiempo_por_secuencia)
    
    self.autoforge_estado_actual_str = "Iniciando Secuencia..."
    
    self.session_timer = QtCore.QTimer()
    self.session_timer.timeout.connect(self._tick_session_timer)
    self.session_timer.start(1000)

  def estado_iniciar_secuencia_continua(self):
    self.is_recording = False
    self.label_rec_time.setVisible(True)
    self.config_stack.setCurrentIndex(1) # Ocultar configuración
    
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
    texto_ventana = "PREPARANDO..."
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None # Mantener variable para no romper nada
    
    self._safe_autoforge_single_shot(3000, self.estado_grabar_ruido_continuo)

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

  # ==============================================================================
  # --- RUTINA DE CALIBRACIÓN CVM (3 TOMAS /A/, /I/, /U/ MULTIPARAMÉTRICA) ---
  # ==============================================================================
  def estado_iniciar_calibracion_cvm(self, target_post_cvm):
    if not getattr(self, 'is_autoforge_running', False): return
    self._post_cvm_target_method = target_post_cvm
    self.is_recording = False
    self.autoforge_estado_actual_str = "CONTRACCIÓN MUSCULAR MÁXIMA"
    self.label_rec_time.setText("CONTRACCIÓN MUSCULAR MÁXIMA (3 TOMAS POR VOCAL EXTREMA: A, I, U)...")
    self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFCC00;")
    
    self.cvm_sequence = [
        ('A', 1), ('A', 2), ('A', 3),
        ('I', 1), ('I', 2), ('I', 3),
        ('U', 1), ('U', 2), ('U', 3)
    ]
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

    self.autoforge_overlay.setText(
        "<div align='center'>"
        "<span style='color:#FFCC00; font-size:36px; font-weight:bold;'>CONTRACCIÓN MUSCULAR MÁXIMA</span><br><br>"
        "<span style='color:#FFFFFF; font-size:24px;'>Iniciando protocolo de calibración al ritmo del metrónomo.</span><br>"
        "<span style='color:#00FFCC; font-size:20px;'>Sigue las instrucciones en pantalla (Vocales: A, I, U).</span>"
        "</div>"
    )
    self.autoforge_overlay.show()
    self.start_native_metronome(count_in=4, force_start=True)

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
        f"<span style='font-size:22px; color:#00FFCC;'>Supremo Global (Pico): {self.supremo_global_uV:.1f} µV</span><br>"
        f"<span style='font-size:18px; color:#AAFFAA;'>Supremo Global (P95 Robusto): {self.supremo_global_p95_uV:.1f} µV</span>"
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
    self.lbl_recording_space.setText("<span style='color: yellow;'>GRABANDO RUIDO FINAL...</span>")
    
    self.autoforge_overlay.setText("<div align='center'>HAZ SILENCIO<br>GRABANDO RUIDO FINAL (5s)...</div>")
    self.autoforge_overlay.show()
    
    self.final_raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    self.is_recording_final_noise = True

    # --- HOOK FOTOMETRIA: Captura no bloqueante en ruido basal final ---
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      self.photometry_mgr.request_snapshot(fase="ruido_basal_final", palabra="ruido", pulso=0, sample_rate=self.SAMPLE_RATE)
    
    noise_dur = getattr(self, 'spin_noise_duration', None)
    dur_sec = noise_dur.value() if noise_dur is not None else 5.0
    self._safe_autoforge_single_shot(int(dur_sec * 1000), self.estado_finalizar_ruido_final)

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
    self.autoforge_overlay.setText("") 
    self.autoforge_overlay.hide()
    self.label_rec_time.setVisible(True) 
    self.lbl_recording_space.setText("<span style='color: yellow;'>GRABANDO RUIDO...</span>")
    
    self.current_recording = []
    
    self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
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
    
    # --- Resetear escala Y a rango base calibrado (500 µV) ---
    if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
      self.plot.setYRange(0.0, 500.0, padding=0)
    else:
      self.plot.setYRange(-500.0, 500.0, padding=0)
    self._held_plot_ymax = 500.0

    # --- HOOK FOTOMETRIA: Instantanea de ruido basal ---
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      self.photometry_mgr.request_snapshot(fase="ruido_basal", palabra="ruido", pulso=0, sample_rate=self.SAMPLE_RATE)
    
    noise_dur = self.spin_noise_duration.value()
    self._safe_autoforge_single_shot(int(noise_dur * 1000), self.estado_mostrar_preparate_continuo)

  def estado_mostrar_preparate_continuo(self):
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
          
          self.noise_lines[i].setPos(self.noise_levels[i])
          self.noise_lines_neg[i].setPos(-self.noise_levels[i])
          self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          show_noise = self.chk_show_noise.isChecked()
          self.noise_lines[i].setVisible(show_noise)
          self.noise_lines_neg[i].setVisible(show_noise)
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
    
    self.start_native_metronome(count_in=4, force_start=True)
    
    self.autoforge_overlay.hide()
    
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
    self.countdown_text.show()
    
    self.countdown_text.setText("PREPÁRATE...\n3")
    self._safe_autoforge_single_shot(int(beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n2"))
    self._safe_autoforge_single_shot(int(2 * beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n1"))
    
    self._safe_autoforge_single_shot(int(3 * beat_interval_s * 1000), self.estado_iniciar_grabacion_continua)

  def estado_iniciar_grabacion_continua(self):
    self.autoforge_overlay.hide()
    self.countdown_text.setText("¡GO!")
    QtCore.QTimer.singleShot(1000, self.countdown_text.hide)
    
    import time
    self.recording_start_time = time.perf_counter()
    self.is_recording = True
    self.is_finishing_sequence = False
    
    self.current_recording = getattr(self, 'autoforge_ruido_guardado', []).copy()
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    
    self.label_rec_time.setText("GRABANDO SECUENCIA CONTINUA...")
    if len(self.autoforge_words) > 0:
        self.lbl_recording_space.setText(f"<div align='center' style='font-size: 100px; font-weight: bold;'>{self.autoforge_words[0].upper()}</div>")

    # --- HOOK FOTOMETRIA: Inicio de sesion continua y captura onset ---
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      self.photometry_mgr.start_take_session(f"SecuenciaContinua_{self.autoforge_prueba}_{self.autoforge_sujeto}", t0_perf=self.recording_start_time)
      self.photometry_mgr.request_snapshot(fase="onset", palabra="inicio_secuencia", pulso=1, t_daq_s=0.0, sample_rate=self.SAMPLE_RATE)

  def estado_guardar_secuencia_continua(self):
    if not getattr(self, 'final_noise_completed', False):
      self.estado_grabar_ruido_final(target_post_noise=self.estado_guardar_secuencia_continua)
      return

    self.is_recording = False
    
    self.stop_native_metronome()
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    self.autoforge_estado_actual_str = "GUARDANDO SECUENCIA"
    self.autoforge_overlay.setText("<div align='center'>SECUENCIA COMPLETADA<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    
    local_recording = self.current_recording.copy()
    local_ruido = getattr(self, 'autoforge_ruido_guardado', []).copy()
    
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
      
      total_pulsos = len(self.autoforge_words) * self.autoforge_target_reps
      
      full_word_sequence = []
      for i in range(total_pulsos):
          full_word_sequence.append(self.autoforge_words[i % len(self.autoforge_words)])
          
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

      metadata = {
        "measurement_date": now.isoformat(),
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
      
      # Exportar Fotometria 3D si esta habilitada
      if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
        try:
          foto_resumen = self.photometry_mgr.export_take_photometry(str(base_dir))
          if foto_resumen:
            metadata["fotometria_espacial"] = foto_resumen
        except Exception as exc:
          print(f"--- ERROR AL EXPORTAR FOTOMETRIA 3D ---\n{exc}")
      
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

      # Persistir Notas del Experimento (metadata.json, canal_*/metadata.json y notas_experimento.txt)
      self.last_measurement_dir = str(base_dir)
      texto_notas = getattr(self, 'notas_experimento', '')
      self.persistir_notas_experimento(str(base_dir), metadata_base=metadata, texto_notas=texto_notas)

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
      
      try: generar_grafico_estadisticas(self.stats_time, self.stats_snr, self.stats_noise_mean, self.stats_noise_std, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      self.current_recording = [] 
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    self._safe_autoforge_single_shot(5000, self.estado_finalizar_secuencia_continua)

  def estado_finalizar_secuencia_continua(self):
    self.autoforge_overlay.setText("<div align='center'>¡SECUENCIA COMPLETADA!</div>")
    QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
    
    if getattr(self, 'is_acquiring', False):
      self.on_start_acq_click() # Esto detiene el DAQ y llama a reset_autoforge_buttons()
    else:
      self.is_autoforge_running = False
      self.is_autoforge_continuo = False
      self.reset_autoforge_buttons()
      
    self.config_stack.setCurrentIndex(0) # Restaurar configuración
    if hasattr(self, 'session_timer'):
      self.session_timer.stop()

  def _iniciar_timer_global(self):
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    noise_dur = self.spin_noise_duration.value()
    tiempo_por_palabra = 3.0 + noise_dur + (3.0 * (60.0/bpm)) + (self.autoforge_target_reps * (60.0/bpm)) + 10.0
    cvm_time = (22.0 * (60.0/bpm)) if getattr(self, 'autoforge_do_cvm', True) else 0.0
    tiempo_total = cvm_time + (len(self.autoforge_words) * tiempo_por_palabra) + noise_dur + 2.0
    self.tiempo_restante_global = int(tiempo_total)
    
    # Guardar el estado base de la UI
    self.autoforge_estado_actual_str = "Iniciando..."
    
    self.session_timer = QtCore.QTimer()
    self.session_timer.timeout.connect(self._tick_session_timer)
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
      
      if getattr(self, 'is_acquiring', False):
        self.on_start_acq_click() # Detiene el DAQ correctamente y llama a reset_autoforge_buttons()
      else:
        self.is_autoforge_running = False
        self.reset_autoforge_buttons()
        
      self.config_stack.setCurrentIndex(0) # Restaurar configuración
      if hasattr(self, 'session_timer'): self.session_timer.stop()
      return
      
    palabra = self.autoforge_words[self.autoforge_word_idx]
    self.is_recording = False
    self.label_rec_time.setVisible(True) # --- NUEVO: Habilitado para no dar sensación de "congelamiento"
    self.config_stack.setCurrentIndex(1) # Ocultar configuración
    
    palabra_num = self.autoforge_word_idx + 1
    total_palabras = len(self.autoforge_words)
    self.autoforge_estado_actual_str = f"Set: {palabra_num}/{total_palabras}"
    
    tr_str = getattr(self, 'tiempo_restante_str', '00:00')
    cuadro_azul = f"&nbsp;&nbsp;<span style='background-color:#0055FF; color:white;'>&nbsp;Resta: {tr_str}&nbsp;</span>"
    self.label_rec_time.setText(f"{self.autoforge_estado_actual_str}{cuadro_azul}")
    
    self.autoforge_overlay.setText("<div align='center'>HAZ SILENCIO<br>PREPARANDO ENTORNO...</div>")
    self.autoforge_overlay.show()
    
    texto_ventana = f"SIGUIENTE:\n{palabra.upper()}"
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None
    
    # Reset buffers de picos y ruido en tiempo real
    self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
    self.rt_last_noise_eval = 0.0
    self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    
    # 3 Segundos de silencio ANTES de siquiera empezar a capturar el ruido
    self._safe_autoforge_single_shot(3000, self.estado_grabar_ruido)

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
    self.lbl_recording_space.setText("<span style='color: yellow;'>GRABANDO RUIDO...</span>")
    
    self.current_recording = []
    
    # Reiniciar arreglos de ruido
    self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
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
    
    # --- Resetear escala Y a rango base calibrado (500 µV) ---
    if hasattr(self, 'chk_rms_env') and self.chk_rms_env.isChecked():
      self.plot.setYRange(0.0, 500.0, padding=0)
    else:
      self.plot.setYRange(-500.0, 500.0, padding=0)
    self._held_plot_ymax = 500.0

    # --- HOOK FOTOMETRIA: Instantanea de ruido basal ---
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      self.photometry_mgr.request_snapshot(fase="ruido_basal", palabra="ruido", pulso=0, sample_rate=self.SAMPLE_RATE)
    
    # Programar la aparición de la ventana "Preparate" y el metrónomo CUANDO TERMINE EL RUIDO
    noise_dur = self.spin_noise_duration.value()
    self._safe_autoforge_single_shot(int(noise_dur * 1000), self.estado_mostrar_preparate)

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
          
          self.noise_lines[i].setPos(self.noise_levels[i])
          self.noise_lines_neg[i].setPos(-self.noise_levels[i])
          self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          show_noise = self.chk_show_noise.isChecked()
          self.noise_lines[i].setVisible(show_noise)
          self.noise_lines_neg[i].setVisible(show_noise)
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
    # Actualizamos la ventana flotante para que avise "PREPARATE"
    texto_ventana = f"SIGUIENTE:\n{palabra.upper()}"
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None
    
    # Lanzar el metrónomo AHORA MISMO con un Count-in de tipo carreras (3 graves, 1 ya estaba agregado)
    # (El metrónomo ya está fijo en el recuadro superior)
    
    self.start_native_metronome(count_in=4, force_start=True)
    
    self.autoforge_overlay.hide()
    
    # --- NUEVO: Sincronización del texto amarillo de PREPARATE con el count-in acústico ---
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
    self.countdown_text.show()
    
    self.countdown_text.setText("PREPÁRATE...\n3")
    self._safe_autoforge_single_shot(int(beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n2"))
    self._safe_autoforge_single_shot(int(2 * beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n1"))
    
    # A los 3 compases justos (cuando suene el pitido Agudo), pasamos a la palabra.
    self._safe_autoforge_single_shot(int(3 * beat_interval_s * 1000), self.estado_cambiar_ventana_palabra)

  def estado_cambiar_ventana_palabra(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_cambiar_ventana_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.autoforge_overlay.hide() # Quitamos el "PREPARATE" del medio
    
    # Mostrar el "GO!" en amarillo y luego esconderlo
    self.countdown_text.setText("¡GO!")
    QtCore.QTimer.singleShot(1000, self.countdown_text.hide)
    
    # --- REACTIVAMOS LA GRABACIÓN: AQUÍ EMPIEZA LA PALABRA REAL ---
    import time
    self.recording_start_time = time.perf_counter() # Reseteamos tiempo UI para empezar desde 0.0s en la palabra
    self.is_recording = True
    
    # --- NUEVO: Limpiamos los buffers de guardado y estadísticas para cada palabra ---
    self.current_recording = getattr(self, 'autoforge_ruido_guardado', []).copy()
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    self.rt_peaks = [[] for _ in range(self.NUM_CANALES)]
    self.rt_last_noise_eval = 0.0
    self.raw_noise_accumulated = [[] for _ in range(self.NUM_CANALES)]
    
    self.label_rec_time.setText("GRABANDO PALABRA...")
    
    palabra = self.autoforge_words[self.autoforge_word_idx]
    
    # Lanzamos la ventana limpia con solo la palabra para la grabación
    # El metrónomo YA ESTÁ corriendo, aquí justo emitirá el pitido Agudo (GO).
    self.lbl_recording_space.setText(palabra.upper())
    self.word_window_process = None

    # --- HOOK FOTOMETRIA: Sesion de toma discreta y captura onset ---
    if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
      self.photometry_mgr.start_take_session(f"{palabra}_{self.autoforge_sujeto}", t0_perf=self.recording_start_time)
      self.photometry_mgr.request_snapshot(fase="onset", palabra=palabra, pulso=1, t_daq_s=0.0, sample_rate=self.SAMPLE_RATE)
    
    # Iniciar la cuenta del guardado de los datos de la palabra
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    ms_totales = int((60000 / bpm) * self.autoforge_target_reps)
    self._safe_autoforge_single_shot(ms_totales, self.estado_guardar_palabra)

  def estado_guardar_palabra(self):
    if not getattr(self, 'is_autoforge_running', False): return
    """
    Ejecuta la funcionalidad de estado_guardar_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.is_recording = False
    palabra = self.autoforge_words[self.autoforge_word_idx]
    
    self.stop_native_metronome()
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    # 5 segundos de descanso mientras se guardan los datos pesados (lag)
    self.autoforge_estado_actual_str = "DESCANSO 5s (GUARDANDO)"
    self.autoforge_overlay.setText("<div align='center'>DESCANSO 5s<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    self.lbl_recording_space.setText("<div align='center' style='color:#777777; font-size:40px;'>DESCANSO</div>")
    
    local_recording = self.current_recording.copy()
    self.current_recording = []
    
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

      metadata = {
        "measurement_date": now.isoformat(),
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
      
      # Exportar Fotometria 3D si esta habilitada
      if getattr(self, 'photometry_mgr', None) is not None and getattr(self.photometry_mgr, 'is_enabled', False):
        try:
          foto_resumen = self.photometry_mgr.export_take_photometry(str(base_dir))
          if foto_resumen:
            metadata["fotometria_espacial"] = foto_resumen
        except Exception as exc:
          print(f"--- ERROR AL EXPORTAR FOTOMETRIA 3D ---\n{exc}")

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

      # Persistir Notas del Experimento (metadata.json, canal_*/metadata.json y notas_experimento.txt)
      self.last_measurement_dir = str(base_dir)
      texto_notas = getattr(self, 'notas_experimento', '')
      self.persistir_notas_experimento(str(base_dir), metadata_base=metadata, texto_notas=texto_notas)

      # 3. Guardar Datos Raw CSV y WAV (las funciones internas ya manejan la división por canales)
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
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    # Tras 5 segundos, pasar a verificación muscular para que el usuario pueda ver su señal fluida
    self._safe_autoforge_single_shot(5000, self.estado_verificacion_muscular)

  @QtCore.Slot()
  def estado_verificacion_muscular(self):
    self.autoforge_estado_actual_str = "VERIFICACIÓN MUSCULAR 5s"
    # Ocultar el overlay oscuro para que la señal se vea limpia y sin sombra
    self.autoforge_overlay.hide()
    self.lbl_recording_space.setText(
      "<div align='center' style='color:#00FF00; font-size:30px; font-weight:bold;'>"
      "VERIFICACIÓN MUSCULAR<br>"
      "<span style='font-size:18px; color:#88FF88;'>Activa tus músculos para probar la señal</span>"
      "</div>"
    )
    
    # Ir a la siguiente palabra tras los 5 segundos de verificación (10s totales)
    self._safe_autoforge_single_shot(5000, self.estado_siguiente)

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

  def closeEvent(self, event):
    # Se llama cuando el usuario cierra la ventana
    """
    Ejecuta la funcionalidad de closeEvent.

    Args:
      event (Any): Argumento posicional event.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    print("Ventana de ploteo cerrada. Deteniendo hilos...")
    
    # --- NUEVO: Detener limpiamente para evitar Access Violations (0xC0000005) al salir ---
    if getattr(self, 'is_acquiring', False):
      self.on_start_acq_click()
      
    self.stop_event.set() # Envía la señal a TODOS los hilos
    self.timer.stop() # Detiene el timer de la GUI

    if getattr(self, 'photometry_mgr', None):
      try:
        self.photometry_mgr.stop_worker()
      except Exception:
        pass
    
    # --- NUEVO: Guardar estado de la GUI al cerrar el programa ---
    try:
      config_data = {}
      if os.path.exists(METRONOME_CONFIG_PATH):
        with open(METRONOME_CONFIG_PATH, 'r', encoding='utf-8') as f:
          config_data = json.load(f)
      config_data['last_bpm'] = self.spin_bpm.value()
      config_data['notch_enabled'] = self.chk_notch_enable.isChecked()
      with open(METRONOME_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    except Exception as e:
      print(f"Error guardando configuración al cerrar: {e}")
      
    event.accept() # Acepta el cierre

# =============================================================================
# --- DIÁLOGO DE SELECCIÓN DE MÚSCULOS AL INICIO ---
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
      
      self.form_layout.addRow(f"ai{hw_idx} ({key}):", cmb)
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
  app.setStyleSheet("""
    QWidget {
      background-color: #050505;
      color: #00FF00;
      font-family: 'Courier New', Courier, monospace;
      font-size: 12px;
    }
    QGroupBox {
      border: 1px solid #FF0000;
      border-radius: 5px;
      margin-top: 1ex;
      font-weight: bold;
      color: #FF0000;
    }
    QGroupBox::title {
      subcontrol-origin: margin;
      subcontrol-position: top left;
      padding: 0 3px;
      background-color: #050505;
    }
    QLabel { color: #00FF00; font-weight: bold; }
    QCheckBox { color: #00FF00; }
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
      background-color: #111111;
      color: #00FF00;
      border: 1px solid #FF0000;
      padding: 2px;
    }
  """)

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
