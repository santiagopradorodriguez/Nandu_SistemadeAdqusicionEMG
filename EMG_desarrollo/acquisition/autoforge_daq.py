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
__version__ = "5.0.1"

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

import json
import subprocess

try:
  import winsound
except ImportError:
  winsound = None

if global_progress: global_progress.setValue(30); global_splash.showMessage("Cargando PyQtGraph...", Qt.AlignBottom | Qt.AlignCenter, QColor("white")); app.processEvents()
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui 

# NUEVO: Import para SpinBox y Línea
from pyqtgraph import SpinBox, InfiniteLine

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

def simulador_thread(chunk_samples, sample_rate, num_canales, data_queue, stop_event, test_freq=50, tipo_prueba="Senoidal"):
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

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  print(f"Iniciando hilo de simulación con SR={sample_rate} Hz, Tipo={tipo_prueba}...")
  
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
          datos_archivo = df[canales_cols].values.T
          if datos_archivo.shape[0] < num_canales:
            pad = np.zeros((num_canales - datos_archivo.shape[0], datos_archivo.shape[1]))
            datos_archivo = np.vstack((datos_archivo, pad))
          else:
            datos_archivo = datos_archivo[:num_canales, :]
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
def guardar_grabacion_csv(datos_completos, sample_rate, output_dir, num_canales, base_name="grabacion"):
  """
  Ejecuta la funcionalidad de guardar_grabacion_csv.

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
    return False

  print("Concatenando datos para CSV...")
  try:
    grabacion = np.concatenate(datos_completos, axis=1) # (3, N)
  except ValueError:
    return False
  
  # Crear vector de tiempo
  num_muestras = grabacion.shape[1]
  tiempo = np.arange(num_muestras) / float(sample_rate) # (N,)
  
  # Transponer datos de (3, N) a (N, 3)
  datos_t = grabacion.T
  
  # Apilar tiempo y datos: (N, 1) + (N, 3) -> (N, 4)
  datos_para_csv = np.hstack((tiempo.reshape(-1, 1), datos_t))
  
  # Definir nombre de archivo
  nombre_archivo_csv = os.path.join(output_dir, f"{base_name}.csv")
  
  # Guardar
  try:
    print(f"Guardando CSV en: {nombre_archivo_csv}...")
    headers = "Tiempo (s)," + ",".join([f"Canal {i}" for i in range(num_canales)])
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
# BLOQUE 3.3: GENERADOR DE GRÁFICO (Sin cambios)
# =============================================================================
def generar_grafico_grabacion(datos_completos, sample_rate, output_dir, num_canales, canales_daq, base_name="photo"):
  """
  Ejecuta la funcionalidad de generar_grafico_grabacion.

  Args:
    datos_completos (Any): Argumento posicional datos_completos.
    sample_rate (Any): Argumento posicional sample_rate.
    output_dir (Any): Argumento posicional output_dir.
    num_canales (Any): Argumento posicional num_canales.
    canales_daq (Any): Argumento posicional canales_daq.
    base_name (Any): Argumento posicional base_name.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  if not datos_completos:
    return False

  print("Generando gráfico de la grabación completa...")
  try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  except ImportError:
    print("Error: matplotlib no está instalado.")
    return False
    
  try:
    grabacion = np.concatenate(datos_completos, axis=1)
  except ValueError:
    return False
  
  # Calibrar a microvoltios para el gráfico
  grabacion = (grabacion / 495.0) * 1000000.0
  
  # Crear vector de tiempo
  num_muestras = grabacion.shape[1]
  tiempo = np.arange(num_muestras) / float(sample_rate)
  
  # Crear figura (thread-safe, sin pyplot)
  fig = Figure(figsize=(15, 3 * num_canales))
  canvas = FigureCanvas(fig)
  axs = fig.subplots(
    num_canales, 
    1, 
    sharex=True
  )
  
  # Si hay un solo canal, axs no es un array, hay que manejarlo
  if num_canales == 1:
    axs = [axs]
    
  fig.suptitle(f"Grabación Completa - {base_name}", fontsize=16)

  # Graficar cada canal
  for i in range(num_canales):
    axs[i].plot(tiempo, grabacion[i])
    axs[i].set_ylabel("Amplitud (µV)")
    axs[i].set_title(f"Canal {i} ({canales_daq[i]})")
    axs[i].grid(True)
    
  axs[-1].set_xlabel("Tiempo (s)")
  
  # Definir nombre de archivo
  nombre_archivo_grafico = os.path.join(output_dir, f"{base_name}.png")

  # Guardar
  try:
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajuste para el supertítulo
    fig.savefig(nombre_archivo_grafico, dpi=200) # dpi=200 es un buen balance
    print(f"  Gráfico guardado como: {nombre_archivo_grafico}")
    return True
  except Exception as e:
    print(f"  Error al guardar el gráfico: {e}")
    return False

# =============================================================================
# BLOQUE 3.4: GENERADOR DE GRÁFICOS ESTADÍSTICOS
# =============================================================================
def generar_grafico_estadisticas(stats_time, stats_snr, stats_noise_mean, stats_noise_std, output_dir, num_canales, canales_daq, base_name="estadisticas_evolucion"):
  # Verificar si hay datos que graficar
  """
  Ejecuta la funcionalidad de generar_grafico_estadisticas.

  Args:
    stats_time (Any): Argumento posicional stats_time.
    stats_snr (Any): Argumento posicional stats_snr.
    stats_noise_mean (Any): Argumento posicional stats_noise_mean.
    stats_noise_std (Any): Argumento posicional stats_noise_std.
    output_dir (Any): Argumento posicional output_dir.
    num_canales (Any): Argumento posicional num_canales.
    canales_daq (Any): Argumento posicional canales_daq.
    base_name (Any): Argumento posicional base_name.

  Returns:
    Any: Resultado de la ejecución de la función.
  """
  hay_datos = any(len(t) > 0 for t in stats_time)
  if not hay_datos:
    print("No hay datos estadísticos para graficar.")
    return False

  print("Generando gráfico de evolución temporal de estadísticas...")
  try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
  except ImportError:
    print("Error: matplotlib no está instalado.")
    return False
    
  fig = Figure(figsize=(15, 4 * num_canales))
  canvas = FigureCanvas(fig)
  axs = fig.subplots(num_canales, 2, squeeze=False)
  fig.suptitle(f"Evolución Temporal: SNR y Ruido Inter-pulso", fontsize=16)

  for i in range(num_canales):
    t = stats_time[i]
    if not t:
      continue
    
    # --- NUEVO: Filtrar los primeros 20 segundos para el gráfico ---
    t_arr = np.array(t)
    mask = t_arr >= 20.0
    
    # Si la grabación duró menos de 20 segundos, graficamos todo
    if not np.any(mask):
      mask = np.ones_like(t_arr, dtype=bool)
      
    t_plot = t_arr[mask]
    snr_plot = np.array(stats_snr[i])[mask]
    noise_mean_plot = np.array(stats_noise_mean[i])[mask]
    noise_std_plot = np.array(stats_noise_std[i])[mask]

    # Subplot Izquierdo: SNR
    axs[i, 0].plot(t_plot, snr_plot, marker='o', linestyle='-', color='b', linewidth=2)
    axs[i, 0].set_title(f"Canal {i} ({canales_daq[i]}) - Evolución SNR Promedio")
    axs[i, 0].set_xlabel("Tiempo de Señal (s)")
    axs[i, 0].set_ylabel("SNR Promedio Acumulado")
    axs[i, 0].grid(True, alpha=0.5)

    # Subplot Derecho: Ruido (x̄ y σ)
    axs[i, 1].plot(t_plot, noise_mean_plot, marker='o', linestyle='-', color='r', label='Promedio (x̄)', linewidth=2)
    axs[i, 1].plot(t_plot, noise_std_plot, marker='x', linestyle='--', color='orange', label='Desviación (σ)', linewidth=2)
    axs[i, 1].set_title(f"Canal {i} ({canales_daq[i]}) - Evolución Ruido Inter-pulso")
    axs[i, 1].set_xlabel("Tiempo de Señal (s)")
    axs[i, 1].set_ylabel("Amplitud (µV)")
    axs[i, 1].legend()
    axs[i, 1].grid(True, alpha=0.5)

  nombre_archivo = os.path.join(output_dir, f"{base_name}.png")
  fig.tight_layout(rect=[0, 0.03, 1, 0.96])
  fig.savefig(nombre_archivo, dpi=200)
  print(f"  Gráfico de estadísticas guardado como: {nombre_archivo}")
  return True

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
    self.setMinimumWidth(300)
    
    self.layout = QtWidgets.QFormLayout(self)
    
    self.edit_prueba = QtWidgets.QLineEdit("Prueba1")
    self.edit_sujeto = QtWidgets.QLineEdit("Sujeto1")
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
    
    self.btn_edit_words = QtWidgets.QPushButton(" Editar Palabras")
    self.btn_edit_words.setStyleSheet("background-color: #333333; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 5px; border: 2px solid #555555; border-radius: 4px;")
    self.btn_edit_words.clicked.connect(self.abrir_editor_palabras)
    self.layout.addRow("", self.btn_edit_words)
    
    self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    self.buttonBox.accepted.connect(self.accept)
    self.buttonBox.rejected.connect(self.reject)
    self.layout.addWidget(self.buttonBox)

  def abrir_editor_palabras(self):
    """
    Ejecuta la funcionalidad de abrir_editor_palabras.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_palabras = os.path.join(root_dir, "palabras.txt")
    if not os.path.exists(ruta_palabras):
      with open(ruta_palabras, 'w', encoding='utf-8') as f:
        f.write("A\nE\nI\nO\nU\n")
    
    # Crear un QDialog nativo en lugar de abrir Notepad
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
        # Asegurar que termina en nueva línea
        texto_guardar = text_edit.toPlainText().strip() + "\n"
        f.write(texto_guardar)
      QtWidgets.QMessageBox.information(self, "Guardado", "Lista de palabras guardada con éxito.")

class RealTimePlotter(QtWidgets.QWidget):
  """
  Clase RealTimePlotter.

  Representa y gestiona las operaciones relacionadas con RealTimePlotter.
  """
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
    self.metronome_process = None # Para guardar la referencia al proceso del metrónomo
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
    self.SAMPLE_RATE = 0
    self.PLOT_DURATION_S = 0
    self.PLOT_SAMPLES = 0
    self.CHUNK_DURATION_S = 0
    
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

    # --- NUEVO: Estado del Filtro Notch ---
    self.notch_sos = None
    self.notch_zi = None

    self.FILTER_ORDER = 4 # Orden del filtro Butterworth

    # --- Configurar la ventana principal ---
    self.setWindowTitle(f"Ñandú LSD - Visor y Grabador de Señales Emg v{__version__}")
    self.resize(1200, 800)
    
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
    if os.path.exists('metronome_config.json'):
      try:
        with open('metronome_config.json', 'r') as f:
          data = json.load(f)
          bpm = data.get('last_bpm', 60)
          pulse_count = data.get('last_beat_count', 30)
          notch_enabled = data.get('notch_enabled', True) # Recuperar el estado anterior
      except Exception:
        pass # Usar defaults si hay error
    self.spin_bpm.setValue(bpm)
    self.chk_notch_enable.setChecked(notch_enabled) # Aplicar a la casilla

  def _setup_ui_layouts(self):
    """
    Ejecuta la funcionalidad de _setup_ui_layouts.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.main_layout = QtWidgets.QVBoxLayout()
    self.setLayout(self.main_layout)
    
    self.config_layout = QtWidgets.QGridLayout()
    self.config_groupbox = QtWidgets.QGroupBox("Configuración de Adquisición")
    self.config_groupbox.setLayout(self.config_layout)
    
    self.filter_layout = QtWidgets.QHBoxLayout()
    self.filter_groupbox = QtWidgets.QGroupBox("Filtro Digital (Pasa-Banda)")
    self.filter_groupbox.setLayout(self.filter_layout)

    self.button_layout = QtWidgets.QHBoxLayout()
    self.trigger_layout = QtWidgets.QHBoxLayout()
    self.measure_layout = QtWidgets.QHBoxLayout()
    self.measure_widget = QtWidgets.QWidget()
    self.measure_widget.setLayout(self.measure_layout)

    self.spectrogram_layout = QtWidgets.QHBoxLayout()
    self.spectrogram_groupbox = QtWidgets.QGroupBox("Espectrograma")
    self.spectrogram_groupbox.setLayout(self.spectrogram_layout)

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
      self.channel_checkboxes.append(chk)
      self.channel_layout.addWidget(chk)
    
    # Si no hay ninguno activo, marcamos el 0 por defecto por seguridad
    if not any_checked and self.channel_checkboxes:
      self.channel_checkboxes[0].setChecked(True)

    self.btn_start_acq = QtWidgets.QPushButton("Iniciar Adquisición")
    self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)

    # --- NUEVO: Controles de Protocolo ---
    adq_conf = self.config_mgr.get("adquisicion") or {}
    
    self.label_bpm = QtWidgets.QLabel("BPM:")
    default_bpm = adq_conf.get("bpm", 60)
    self.spin_bpm = SpinBox(value=default_bpm, int=True, bounds=(30, 200), step=1)
    self.spin_bpm.setFixedWidth(80) # Acortar el recuadro del BPM
    
    self.label_noise_duration = QtWidgets.QLabel("Noise (Inicio) (s):")
    default_noise = adq_conf.get("ruido_segundos", 3.0)
    self.spin_noise_duration = SpinBox(value=default_noise, dec=True, bounds=(0.5, 20.0), step=0.5)
    self.spin_noise_duration.setFixedWidth(80)
    
    self.label_descanso = QtWidgets.QLabel("Descanso (s):")
    default_descanso = adq_conf.get("tiempo_descanso", 10.0)
    self.spin_descanso = SpinBox(value=default_descanso, dec=True, bounds=(1.0, 60.0), step=1.0)
    self.spin_descanso.setFixedWidth(80)

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
    self.config_layout.addWidget(self.label_bpm, 1, 5)
    self.config_layout.addWidget(self.spin_bpm, 1, 6) # BPM
    self.config_layout.addWidget(self.label_noise_duration, 2, 5)
    self.config_layout.addWidget(self.spin_noise_duration, 2, 6) # Ruido
    self.config_layout.addWidget(self.label_descanso, 3, 5)
    self.config_layout.addWidget(self.spin_descanso, 3, 6) # Descanso
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

  def _setup_ui_filter_panel(self):
    """Configura el panel de filtro."""
    self.chk_notch_enable = QtWidgets.QCheckBox("Filtro Notch 50 Hz")
    self.chk_notch_enable.setChecked(True) # Filtro Notch por predeterminado
    self.chk_notch_enable.setToolTip("Aplica un filtro para eliminar el ruido de la red eléctrica (50 Hz).")

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
    
    self.btn_autoforge = QtWidgets.QPushButton(" Autograbado")
    self.btn_autoforge.setStyleSheet("background-color: #ff0055; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #ff0055; border-radius: 4px;")
    self.btn_autoforge.clicked.connect(self.iniciar_autoforge)

    self.btn_autoforge_continuo = QtWidgets.QPushButton(" Secuencia Continua")
    self.btn_autoforge_continuo.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #aa00ff; border-radius: 4px;")
    self.btn_autoforge_continuo.clicked.connect(self.iniciar_autoforge_continuo)

    self.btn_record = QtWidgets.QPushButton("Empezar a Grabar")
    self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)
    
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
    self.button_layout.addWidget(self.label_rec_time)
    
    self.button_layout.addSpacing(10)
    self.button_layout.addStretch(1) # Espaciador
    self.button_layout.addWidget(self.chk_autoscroll)
    
    # --- Widgets del Trigger ---
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
    self.splitter.setSizes([600, 200]) # Tamaños iniciales

    # --- NUEVO: QStackedWidget para ocultar Configuración durante grabación ---
    self.config_stack = QtWidgets.QStackedWidget()
    self.config_stack.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
    self.config_groupbox.setMinimumHeight(180)
    self.config_stack.setMinimumHeight(180)
    self.config_stack.addWidget(self.config_groupbox)
    
    self.empty_recording_widget = QtWidgets.QWidget()
    self.empty_recording_widget.setStyleSheet("background-color: #050505;")
    
    self.lbl_recording_space = QtWidgets.QLabel("VENTANAS EXTERNAS ACTIVAS", self.empty_recording_widget)
    self.lbl_recording_space.setAlignment(QtCore.Qt.AlignCenter)
    self.lbl_recording_space.setWordWrap(True)
    self.lbl_recording_space.setStyleSheet("background-color: #050510; color: #00FF00; font-family: 'Courier New', monospace; font-size: 45px; font-weight: 900; border: 2px dashed #FF0055; padding: 10px;")
    
    # En lugar de usar un layout que puede forzar anchos mínimos, usamos un layout con restricciones duras
    empty_layout = QtWidgets.QVBoxLayout(self.empty_recording_widget)
    empty_layout.setContentsMargins(0, 0, 0, 0)
    empty_layout.addWidget(self.lbl_recording_space)
    
    self.empty_recording_widget.setLayout(empty_layout)
    
    # FUERZA BRUTA HORIZONTAL: NUNCA podrá expandir la pantalla principal
    self.empty_recording_widget.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
    
    self.config_stack.addWidget(self.empty_recording_widget)

    # --- Añadir layouts a la ventana ---
    self.main_layout.addWidget(self.config_stack) # Reemplaza a config_groupbox
    self.main_layout.addWidget(self.filter_groupbox)
    self.main_layout.addLayout(self.button_layout)
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
    self.colores_curvas = []
    self.nombres_musculos = []
    canales_conf = self.config_mgr.get("canales") or {}
    try:
      from utils.config_manager import get_muscle_color
    except ImportError:
      def get_muscle_color(name, default="#ffffff"):
        return "#ff0000" if ("mic" in str(name).lower() or "canal 3" in str(name).lower()) else default

    for i in range(16):
      key = f"Canal {i}"
      musc = canales_conf.get(key, {}).get("musculo", f"Canal {i}")
      self.nombres_musculos.append(musc)
      if i == 3 or "mic" in musc.lower():
        self.colores_curvas.append("#ff0000")
      else:
        self.colores_curvas.append(get_muscle_color(musc, canales_conf.get(key, {}).get("color_hex", "#0074D9")))

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
    self.chk_spectrogram_enable.setEnabled(enabled and SCIPY_DISPONIBLE)
    self.cmb_spectrogram_chan.setEnabled(enabled and self.chk_spectrogram_enable.isChecked())
    self.spectrogram_view.setVisible(enabled and self.chk_spectrogram_enable.isChecked())
    self.filter_groupbox.setEnabled(enabled)
    self.chk_filter_enable.setEnabled(enabled and SCIPY_DISPONIBLE)
    self.chk_notch_enable.setEnabled(enabled and SCIPY_DISPONIBLE)
    # El resto de controles del filtro dependen del checkbox
    is_filter_enabled = enabled and self.chk_filter_enable.isChecked()
    self.spin_low_cut.setEnabled(is_filter_enabled)
    self.spin_high_cut.setEnabled(is_filter_enabled)

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
      self.is_autoforge_running = False
      self.is_autoforge_continuo = False
      if hasattr(self, 'autoforge_overlay'):
        self.autoforge_overlay.hide()
      if getattr(self, 'is_recording', False):
        self.on_record_click()
      
      # --- MEJORA: Detener el metrónomo solo si el proceso todavía existe ---
      if self.metronome_process and self.metronome_process.poll() is None:
        try:
          print("Enviando comando STOP_APP al metrónomo...")
          self.metronome_process.stdin.write("STOP_APP\n")
          self.metronome_process.stdin.flush()
        except (OSError, ValueError) as e:
          # Captura errores si la tubería (pipe) ya está cerrada.
          print(f"Advertencia: No se pudo comunicar con el proceso del metrónomo (puede que ya estuviera cerrado). Error: {e}")
      
      if self.metronome_process:
        try:
          self.metronome_process.wait(timeout=2) # Esperar a que se cierre solo
        except subprocess.TimeoutExpired:
          print("Advertencia: El metrónomo no se cerró a tiempo. Forzando cierre...")
          self.metronome_process.kill()
        self.metronome_process = None

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

      # --- NUEVO: Lanzar el metrónomo si está seleccionado ---
      if self.chk_use_metronome.isChecked():
        self._save_metronome_config() # Guardar el BPM actual antes de lanzar
        try:
          python_executable = sys.executable
          if getattr(sys, 'frozen', False):
            script_path = 'acquisition/metronomo_visual.py'
          else:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
          print("Lanzando metrónomo con --autostart...")
          # --- MODIFICADO: Añadir stdin=subprocess.PIPE para poder enviarle comandos ---
          self.metronome_process = subprocess.Popen(
            [python_executable, script_path, '--autostart', '--count'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000) # Evitar consola
          )
        except Exception as e:
          print(f"Error al lanzar el metrónomo: {e}")
          self.metronome_process = None

      # Iniciar hilo
      self.stop_event.clear()

      if self.chk_use_mic.isChecked():
        chunk_samples_dinamico = int(self.SAMPLE_RATE * 0.05)
        self.acquisition_thread = threading.Thread(target=microphone_thread, args=(chunk_samples_dinamico, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event), daemon=True)
      elif self.chk_modo_prueba.isChecked():

        test_freq = self.spin_test_freq.value()
        tipo_prueba = self.cmb_fuente_prueba.currentText()
        self.acquisition_thread = threading.Thread(target=simulador_thread, args=(chunk_samples_dinamico, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event, test_freq, tipo_prueba), daemon=True)
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
      if os.path.exists('metronome_config.json'):
        with open('metronome_config.json', 'r', encoding='utf-8') as f:
          config_data = json.load(f) # Asegurar que se lee como UTF-8
      
      config_data['last_bpm'] = current_bpm
      with open('metronome_config.json', 'w', encoding='utf-8') as f:
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
    
    title_label = QtWidgets.QLabel("<b>Mediciones (chunk):</b>")
    title_label.setStyleSheet("color: white;")
    self.measure_layout.addWidget(title_label)
    for i in range(self.NUM_CANALES):
      color = self.colores_curvas[i % len(self.colores_curvas)]
      musculo = self.nombres_musculos[i % len(self.nombres_musculos)]
      self.curvas.append(self.plot.plot(pen=color, name=musculo))
      
      # --- NUEVO: Checkbox de visualización ---
      chk_viz = QtWidgets.QCheckBox(musculo)
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
      self.plot.addItem(scatter)
      self.peak_scatters.append(scatter)
      
      # --- NUEVO: Región dinámica de ruido inter-pulso ---
      dyn_region = pg.LinearRegionItem(orientation='horizontal', brush=pg.mkBrush(0, 255, 0, 40), movable=False)
      dyn_region.lines[0].setPen(pg.mkPen(None))
      dyn_region.lines[1].setPen(pg.mkPen(None))
      dyn_region.hide()
      self.plot.addItem(dyn_region)
      self.dynamic_noise_regions.append(dyn_region)

      # --- NUEVO: Crear un QFrame para cada medición ---
      measurement_frame = QtWidgets.QFrame()
      measurement_frame.setStyleSheet("background-color: #050505; border: 2px solid #00FFFF; border-radius: 5px;")
      measurement_frame.setMinimumWidth(100)
      
      frame_layout = QtWidgets.QVBoxLayout(measurement_frame)
      frame_layout.setContentsMargins(5, 5, 5, 5) # Padding interno
      
      label = QtWidgets.QLabel(f"<b>{musculo}:</b> -- µVp-p, -- µVrms")
      label.setStyleSheet(f"color: {pg.mkColor(color).name()}; font-size: 14px; font-weight: bold; background-color: transparent; border: none;")
      label.setWordWrap(True)
      label.setMinimumWidth(0)
      frame_layout.addWidget(label)
      
      # --- NUEVO: Etiqueta para el tester de ruido ---
      label_ruido = QtWidgets.QLabel("Ruido inter-pulso: Esperando grabación...")
      label_ruido.setStyleSheet("color: gray; font-size: 11px; background-color: transparent;")
      label_ruido.setWordWrap(True)
      label_ruido.setMinimumWidth(0)
      frame_layout.addWidget(label_ruido)
      self.noise_status_labels.append(label_ruido)
      
      self.measure_layout.addWidget(measurement_frame)
      self.measure_labels.append(label) # Guardar solo el label para actualizar su texto
    self.measure_layout.addStretch(1) # Asegurar que los recuadros se alineen a la izquierda
    self.cmb_trig_chan.addItems([f"{i}" for i in range(self.NUM_CANALES)])
    self.cmb_spectrogram_chan.addItems([f"{i}" for i in range(self.NUM_CANALES)])
    
    # Inicializar buffers con el tamaño correcto
    self.plot_buffer_datos = np.zeros((self.NUM_CANALES, self.PLOT_SAMPLES))
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
    nyquist = self.SAMPLE_RATE / 2.0
    freq_scale = nyquist / (self.CURRENT_FFT_LEN / 2 + 1)

    # Eje X (Tiempo): El ancho total del historial del espectrograma en segundos.
    # Cada columna representa un segmento de tiempo (nperseg - noverlap) / fs.
    time_per_column = (self.CURRENT_FFT_LEN - (self.CURRENT_FFT_LEN // 2)) / self.SAMPLE_RATE
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
      
      # --- NUEVO: Resetear el zoom en Y ---
      self.plot.setYRange(-0.01, 0.01)
      
      self.counting_started = False # Reiniciar la bandera de conteo
      self.current_recording.clear()
      self.cmb_terminal_config.setEnabled(False) # <-- NUEVO: Bloquear cambio en plena grabación
      
      # --- NUEVO: Reiniciar variables de ruido ---
      self.noise_data_accumulated = [[] for _ in range(self.NUM_CANALES)]
      self.noise_levels = [0.0] * self.NUM_CANALES
      self.noise_calculated = False
      self.noise_initialized = False # <-- NUEVO: Resetear estado
      self.countdown_active = False # Permitir lanzar el countdown de nuevo
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
      
      target_h = self.config_groupbox.height()
      self.empty_recording_widget.setFixedHeight(target_h)
      self.config_stack.setCurrentIndex(1) # Ocultar configuración
      
      # --- NUEVO: Re-lanzar el metrónomo con count-in (mismo beep que autograbado) al inicio de grabar ruido ---
      if self.chk_use_metronome.isChecked():
        if self.metronome_process:
          try:
            self.metronome_process.kill()
            self.metronome_process.wait(timeout=1)
          except:
            pass
          self.metronome_process = None
        
        try:
          python_executable = sys.executable
          script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
          bpm = self.spin_bpm.value()
          if bpm <= 0: bpm = 60
          print(f"Lanzando metrónomo para grabación con count-in (BPM={bpm}) al inicio de la fase de ruido...")
          
          # Posicionar en la esquina superior derecha, pero debajo del panel de configuración
          pos = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
          metro_x = pos.x() - 250 # 230 width + 20 margin
          metro_y = pos.y() + self.config_stack.height() + 10
          
          self.metronome_process = subprocess.Popen(
            [python_executable, script_path, '--autostart', '--count', f'--bpm={bpm}', '--count-in=4', f'--x={metro_x}', f'--y={metro_y}'],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
          )
        except Exception as e:
          print(f"Error al lanzar el metrónomo con count-in: {e}")
      
      self.btn_record.setText("Detener Grabación")
      self.btn_record.setStyleSheet(self.BTN_REC_STOP_STYLE)
      self.recording_start_time = time.perf_counter()
      self.label_rec_time.setText("Grabando: 00:00.0")
      self.label_rec_time.setVisible(True)
    else:
      # --- DETENER GRABACIÓN ---
      self.is_recording = False
      self.btn_record.setText("Empezar a Grabar")
      self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)
      self.label_rec_time.setVisible(False)
      self.config_stack.setCurrentIndex(0) # Restaurar configuración
      self.cmb_terminal_config.setEnabled(not self.chk_modo_prueba.isChecked()) # <-- NUEVO: Desbloquear
      self.recording_start_time = None
      
      # --- NUEVO: Detener la adquisición automáticamente al detener la grabación ---
      if self.is_acquiring:
        self.on_start_acq_click()
      
      # --- MEJORA: Terminar el proceso del metrónomo solo si todavía existe ---
      if self.metronome_process and self.metronome_process.poll() is None:
        try:
          print("Enviando comando STOP_APP al metrónomo al detener grabación...")
          self.metronome_process.stdin.write("STOP_APP\n")
          self.metronome_process.stdin.flush()
        except (OSError, ValueError) as e:
          print(f"Advertencia: No se pudo comunicar con el proceso del metrónomo. Error: {e}")
      
      if self.metronome_process:
        try:
          self.metronome_process.wait(timeout=2) # Esperar a que el proceso termine
        except subprocess.TimeoutExpired:
          print("Advertencia: El metrónomo no se cerró a tiempo. Forzando cierre...")
          self.metronome_process.kill()
        self.metronome_process = None

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
    
    # --- NUEVO: Evitar sobrescritura ---
    original_output_dir = output_dir
    contador = 2
    while os.path.exists(output_dir):
      output_dir = f"{original_output_dir}_v{contador}"
      contador += 1
      
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
    
    # --- NUEVO: Leer el conteo de pulsos desde el archivo de config del metrónomo ---
    pulse_count_from_metronome = None
    if os.path.exists('metronome_config.json'):
      try:
        print("Leyendo metronome_config.json para pulse_count...")
        with open('metronome_config.json', 'r') as f:
          data = json.load(f)
          pulse_count_from_metronome = data.get('last_beat_count')
          if pulse_count_from_metronome is not None:
            # --- MENSAJE DE CONFIRMACIÓN ---
            print(f" Guardados {pulse_count_from_metronome} pulsos en el metadata.")
          else:
            print("[WARN] No se encontró 'last_beat_count' en metronome_config.json.")
      except Exception as e:
        print(f"Advertencia: No se pudo leer el conteo de pulsos desde metronome_config.json. Error: {e}")

    now = datetime.now()
    muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
    muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}

    # --- NUEVO: Guardar metadata.json con fecha, hora, timestamp y músculos ---
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
      "comentario": comentario # <-- NUEVO: Añadir el comentario al metadata
    }
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

    # 1. Guarda el .wav
    try:
      guardar_grabacion_wav(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR ARCHIVOS .WAV ---\n{e}")

    # 2. Guarda el .csv
    try:
      guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR .CSV ---\n{e}")

    # 3. Genera el gráfico .png
    try:
      generar_grafico_grabacion(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES, self.CANALES_DAQ)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR .PNG ---\n{e}")
      print("  (¿Estás seguro de que 'matplotlib' está instalado? -> pip install matplotlib)")

    # 4. Genera el gráfico de estadísticas (Evolución de SNR y Ruido)
    try:
      generar_grafico_estadisticas(self.stats_time, self.stats_snr, self.stats_noise_mean, self.stats_noise_std, output_dir, self.NUM_CANALES, self.CANALES_DAQ)
    except Exception as e:
      print(f"--- ERROR FATAL AL GUARDAR GRÁFICO ESTADÍSTICO ---\n{e}")

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
      if self.is_recording:
        self.current_recording.append(all_new_data.copy())

      # 4. Calibrar para la visualización y análisis en tiempo real
      
      # --- NUEVO: Imprimir el voltaje RAW de la placa 1 vez por medición para depurar ---
      if not getattr(self, '_raw_print_done', False) and not self.chk_modo_prueba.isChecked():
        print(f"\n[DEBUG DAQ] Voltaje RAW puro entregado por la placa (Canal 0): {all_new_data[0, 0]:.6f} V")
        self._raw_print_done = True

      # Calibración a microvoltios (Ganancia = 495 con R_electrodo = 100 ohm)
      processed_data = (all_new_data / 495.0) * 1000000.0

      # --- NUEVO: Matar el pico transitorio al iniciar/cambiar el filtro ---
      if getattr(self, '_init_filter_state', False):
        initial_vals = processed_data[:, 0]
        if self.notch_zi is not None:
          for i in range(self.NUM_CANALES):
            self.notch_zi[:, :, i] = self.notch_zi[:, :, i] * initial_vals[i]
        if self.filter_zi is not None:
          for i in range(self.NUM_CANALES):
            self.filter_zi[:, :, i] = self.filter_zi[:, :, i] * initial_vals[i]
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
          for i in range(self.NUM_CANALES):
            self.noise_data_accumulated[i].append(processed_data[i, :])
            
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
                  if winsound:
                    import threading
                    threading.Thread(target=winsound.Beep, args=(800, 200), daemon=True).start()
                
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
              if winsound and getattr(self, 'countdown_active', False) and not getattr(self, 'is_autoforge_running', False) and not self.chk_use_metronome.isChecked():
                import threading
                threading.Thread(target=winsound.Beep, args=(1200, 500), daemon=True).start()
              self.countdown_text.setText("¡GO!")
              QtCore.QTimer.singleShot(1000, self.countdown_text.hide)
              self.countdown_active = False
            
              for i in range(self.NUM_CANALES):
                if self.noise_data_accumulated[i]:
                  all_noise = np.concatenate(self.noise_data_accumulated[i])
                  self.noise_levels[i] = np.mean(np.abs(all_noise))
                  self.initial_noise_mean[i] = self.noise_levels[i]
                  self.initial_noise_std[i] = np.std(all_noise)
                  
                  self.noise_lines[i].setPos(self.noise_levels[i])
                  self.noise_lines[i].show()
                  self.noise_lines_neg[i].setPos(-self.noise_levels[i])
                  self.noise_lines_neg[i].show()
                  self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
                  self.noise_regions[i].show()
                  
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
      for i in range(self.NUM_CANALES):
        self.curvas[i].setData(self.plot_vector_tiempo, self.plot_buffer_datos[i])
      
      if self.chk_autoscroll.isChecked():
        self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)
        
        # --- NUEVO: Peak-Hold Auto Scaling ---
        if self.NUM_CANALES > 0 and self.plot_buffer_datos.size > 0:
          current_max = np.max(np.abs(self.plot_buffer_datos))
          view_range = self.plot.getViewBox().viewRange()[1]
          current_y_max = max(abs(view_range[0]), abs(view_range[1]))
          
          if current_max * 1.1 > current_y_max or current_y_max < 0.0001:
            self.plot.setYRange(-current_max * 1.2, current_max * 1.2, padding=0)
          else:
            # Auto-escala dinámica con decaimiento rápido
            new_y_max = max(current_max * 1.2, current_y_max * 0.94)
            self.plot.setYRange(-new_y_max, new_y_max, padding=0)
      self.check_for_trigger(processed_data, total_muestras_leidas)
      self.actualizar_mediciones(processed_data, self.plot_buffer_datos)
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
          
        if getattr(self, 'noise_calculated', False) and signal_time >= 0:
          period_s = 60.0 / max(1, self.spin_bpm.value())
          phase = signal_time % period_s
          
          # Al cruzar la mitad del ciclo (punto de máxima relajación teórica)
          if getattr(self, 'last_phase', 0) < period_s * 0.5 <= phase:
            window_s = period_s / 4.0 # Ventana de 1/4 del pulso
            samples_window = int(window_s * self.SAMPLE_RATE)
            
            if samples_window > 0 and samples_window <= self.PLOT_SAMPLES:
              for i in range(self.NUM_CANALES):
                noise_segment = self.plot_buffer_datos[i, -samples_window:]
                curr_mean = np.mean(np.abs(noise_segment))
                curr_std = np.std(noise_segment)
                
                init_std = getattr(self, 'initial_noise_std', [0]*self.NUM_CANALES)[i]
                init_mean = getattr(self, 'initial_noise_mean', [0]*self.NUM_CANALES)[i]
                
                # Calcular el radio de deterioro (manejar división por cero en señales perfectas simuladas)
                if init_std > 0.01:
                  ratio = curr_std / init_std
                elif init_mean > 0.01:
                  ratio = curr_mean / init_mean
                else:
                  ratio = 1.0 # Señal base sin ruido
                
                if hasattr(self, 'noise_status_labels'):
                  # Neón verde si se mantiene por debajo del 120% del original, si no, Rojo Cyberpunk
                  bg_color = "#000000"
                  fg_color = "#00FF00" if ratio <= 1.20 else "#FF0000" 
                  text = f"Inter-pulso: x̄={curr_mean:.1f}µV (Base={init_mean:.1f}µV) | {ratio*100:.0f}%"
                  self.noise_status_labels[i].setText(text)
                  self.noise_status_labels[i].setStyleSheet(f"color: {fg_color}; background-color: {bg_color}; border: 1px solid {fg_color}; border-radius: 3px; padding: 2px; font-weight: bold; font-size: 11px;")
                  print(f"[DEBUG UI] Actualizada etiqueta de ruido: {text}")
                
                # --- NUEVO: Actualizar región dinámica en el gráfico ---
                if hasattr(self, 'dynamic_noise_regions'):
                  brush_color = pg.mkBrush(0, 255, 0, 40) if ratio <= 1.20 else pg.mkBrush(255, 0, 0, 40)
                  self.dynamic_noise_regions[i].setRegion([-curr_mean, curr_mean])
                  self.dynamic_noise_regions[i].setBrush(brush_color)
                  self.dynamic_noise_regions[i].show()
                
                # --- NUEVO: Guardar métricas para el gráfico final ---
                samples_period = min(int(period_s * self.SAMPLE_RATE), self.PLOT_SAMPLES)
                full_period_segment = self.plot_buffer_datos[i, -samples_period:]
                peak_val = np.max(np.abs(full_period_segment))
                baseline_noise = self.initial_noise_mean[i] if self.initial_noise_mean[i] > 0 else 1e-12
                curr_snr = peak_val / baseline_noise
                
                self.stats_time[i].append(signal_time)
                
                # --- MODIFICADO: Calcular SNR Promedio Acumulado ---
                if len(self.stats_snr[i]) > 0:
                  n = len(self.stats_snr[i])
                  snr_acumulado = (self.stats_snr[i][-1] * n + curr_snr) / (n + 1)
                else:
                  snr_acumulado = curr_snr
                  
                self.stats_snr[i].append(snr_acumulado)
                self.stats_noise_mean[i].append(curr_mean)
                self.stats_noise_std[i].append(curr_std)
                
                # --- NUEVO: Acumular para el UI global ---
                if hasattr(self, 'global_snr_acumulado'):
                  c = self.global_snr_count[i]
                  self.global_snr_acumulado[i] = (self.global_snr_acumulado[i] * c + curr_snr) / (c + 1)
                  self.global_snr_count[i] += 1
          
          self.last_phase = phase
      
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
      
      # --- MEJORA: Buscar el pico absoluto en el entorno del pulso actual ---
      # En lugar de buscar en todo el buffer visible (lo que deja el marcador "pegado" a pulsos viejos),
      # limitamos la búsqueda al tiempo equivalente a 1.5 ciclos del metrónomo.
      period_s = 60.0 / max(1, self.spin_bpm.value())
      samples_to_search = int((period_s * 1.5) * self.SAMPLE_RATE)
      search_start_idx = max(0, self.PLOT_SAMPLES - samples_to_search)
      
      peak_th = self.spin_peak_th.value()

      # Actualiza los labels y el marcador gráfico
      for i in range(self.NUM_CANALES):
        musculo = self.nombres_musculos[i % len(self.nombres_musculos)]
        texto = f"<b>{musculo}:</b> {vp_p[i]:.1f} µVp-p, {rms[i]:.1f} µVrms"
        
        # Aislar la ventana de búsqueda para este canal
        window_data = plot_buffer_datos[i, search_start_idx:]
        abs_window = np.abs(window_data)
        
        # --- NUEVO: Encontrar los puntos que superan la línea de threshold y promediarlos ---
        mask_superan = abs_window >= peak_th
        
        if np.any(mask_superan):
          # Promediar las amplitudes absolutas de todos los puntos por encima de la línea
          avg_peak_val = np.mean(abs_window[mask_superan])
          
          # Encontrar el tiempo del pico máximo para ubicar el marcador visualmente
          idx_max_local = np.argmax(abs_window)
          idx_max = search_start_idx + idx_max_local
          t_max = self.plot_vector_tiempo[idx_max]
          
          # Colocar el marcador en el tiempo del pico, ajustando el valor (Y) al promedio calculado
          signo = np.sign(plot_buffer_datos[i, idx_max])
          v_visual = avg_peak_val * (1 if signo >= 0 else -1)
          
          self.peak_scatters[i].setData([t_max], [v_visual])
          if getattr(self, 'is_recording', False) and getattr(self, 'noise_calculated', False):
            if self.noise_levels[i] > 0:
              snr_inst = avg_peak_val / self.noise_levels[i]
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

  def iniciar_autoforge(self):
    """
    Ejecuta la funcionalidad de iniciar_autoforge.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    try:
      if getattr(self, 'is_autoforge_running', False):
        self.is_autoforge_running = False
        self.btn_autoforge.setText(" Autograbado")
        self.btn_autoforge.setStyleSheet("background-color: #ff0055; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #ff0055; border-radius: 4px;")
        self.autoforge_overlay.hide()
        self.config_stack.setCurrentIndex(0)
        if hasattr(self, 'session_timer'): self.session_timer.stop()
        if hasattr(self, 'metronome_process') and self.metronome_process:
          try: self.metronome_process.kill()
          except: pass
          self.metronome_process = None
        return
        
      # --- NUEVO: Apagar metrónomo general si estaba corriendo para que no choquen ---
      if hasattr(self, 'metronome_process') and self.metronome_process:
        try:
          self.metronome_process.kill()
        except: pass
        self.metronome_process = None
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
        
      import os
      root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      ruta_palabras = os.path.join(root_dir, "palabras.txt")
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
        
        # Actualizar el spinbox principal de BPM si existe
        try:
          self.spin_bpm.setValue(bpm)
          self._save_metronome_config() # Guardar archivo JSON con el BPM
        except:
          pass
        
        self.autoforge_word_idx = 0
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
        self.is_autoforge_running = False
        self.btn_autoforge_continuo.setText(" Secuencia Continua")
        self.btn_autoforge_continuo.setStyleSheet("background-color: #aa00ff; color: white; font-weight: bold; font-family: 'Courier New'; font-size: 14px; padding: 8px; border: 2px solid #aa00ff; border-radius: 4px;")
        self.autoforge_overlay.hide()
        self.config_stack.setCurrentIndex(0)
        if hasattr(self, 'session_timer'): self.session_timer.stop()
        if hasattr(self, 'metronome_process') and self.metronome_process:
          try: self.metronome_process.kill()
          except: pass
          self.metronome_process = None
        return
        
      if hasattr(self, 'metronome_process') and self.metronome_process:
        try: self.metronome_process.kill()
        except: pass
        self.metronome_process = None
        self.chk_use_metronome.setChecked(False)

      if not self.is_acquiring:
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
          old_state = self.chk_use_metronome.isChecked()
          self.chk_use_metronome.setChecked(False)
          self.on_start_acq_click()
          self.chk_use_metronome.setChecked(old_state)
        finally:
          QtWidgets.QApplication.restoreOverrideCursor()
        
      import os
      root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      ruta_palabras = os.path.join(root_dir, "palabras.txt")
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
        
        try:
          self.spin_bpm.setValue(bpm)
          self._save_metronome_config()
        except:
          pass
        
        self.autoforge_word_idx = 0
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
    tiempo_por_secuencia = self.spin_noise_duration.value() + (3 * (60.0/bpm)) + (total_words * (60.0/bpm)) + 2.0
    self.tiempo_restante_global = int(tiempo_por_secuencia)
    
    self.autoforge_estado_actual_str = "Iniciando Secuencia..."
    
    self.session_timer = QtCore.QTimer()
    self.session_timer.timeout.connect(self._tick_session_timer)
    self.session_timer.start(1000)

  def estado_iniciar_secuencia_continua(self):
    self.is_recording = False
    self.label_rec_time.setVisible(True)
    target_h = self.config_groupbox.height()
    self.empty_recording_widget.setFixedHeight(target_h)
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
    
    QtCore.QTimer.singleShot(3000, self.estado_grabar_ruido_continuo)

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
    
    self.plot.setYRange(-0.01, 0.01)
    
    noise_dur = self.spin_noise_duration.value()
    QtCore.QTimer.singleShot(int(noise_dur * 1000), self.estado_mostrar_preparate_continuo)

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
          self.noise_levels[i] = np.mean(np.abs(all_noise))
          self.initial_noise_mean[i] = self.noise_levels[i]
          self.initial_noise_std[i] = np.std(all_noise)
          
          self.noise_lines[i].setPos(self.noise_levels[i])
          self.noise_lines[i].show()
          self.noise_lines_neg[i].setPos(-self.noise_levels[i])
          self.noise_lines_neg[i].show()
          self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          self.noise_regions[i].show()
          
          if hasattr(self, 'noise_status_labels'):
            self.noise_status_labels[i].setText(f"Ruido Base: x̄={self.initial_noise_mean[i]:.1f}µV, s={self.initial_noise_std[i]:.1f}µV")
            self.noise_status_labels[i].setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold; background-color: #111111; border: 1px solid #00FFFF;")
      
      if self.NUM_CANALES > 0:
        ruido_maximo_std = max(getattr(self, 'initial_noise_std', [0]))
        if ruido_maximo_std > 0:
          self.spin_peak_th.setValue(ruido_maximo_std * 5.0)
      self.noise_calculated = True
    
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    beat_interval_s = 60.0 / bpm
    
    try:
      if hasattr(self, 'word_window_process') and self.word_window_process:
        self.word_window_process.stdin.write("PREPÁRATE\\n")
        self.word_window_process.stdin.flush()
    except: pass
    
    import subprocess, sys, os
    python_executable = sys.executable
    if getattr(sys, 'frozen', False):
      script_path = 'metronomo_visual.py'
    else:
      script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
    # Posicionar en la esquina superior derecha, pero debajo del panel de configuración
    pos = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
    metro_x = pos.x() - 250
    metro_y = pos.y() + self.config_stack.height() + 10
    
    self.metronome_process = subprocess.Popen(
      [python_executable, script_path, '--autostart', '--count', f'--bpm={bpm}', '--count-in=4', f'--x={metro_x}', f'--y={metro_y}'],
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      text=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
    )
    
    self.autoforge_overlay.hide()
    
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
    self.countdown_text.show()
    
    self.countdown_text.setText("PREPÁRATE...\n3")
    QtCore.QTimer.singleShot(int(beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n2"))
    QtCore.QTimer.singleShot(int(2 * beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n1"))
    
    QtCore.QTimer.singleShot(int(3 * beat_interval_s * 1000), self.estado_iniciar_grabacion_continua)

  def estado_iniciar_grabacion_continua(self):
    self.autoforge_overlay.hide()
    self.countdown_text.setText("¡GO!")
    QtCore.QTimer.singleShot(1000, self.countdown_text.hide)
    
    import time
    self.recording_start_time = time.perf_counter()
    self.is_recording = True
    
    self.current_recording = getattr(self, 'autoforge_ruido_guardado', []).copy()
    self.stats_time = [[] for _ in range(self.NUM_CANALES)]
    self.stats_snr = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_mean = [[] for _ in range(self.NUM_CANALES)]
    self.stats_noise_std = [[] for _ in range(self.NUM_CANALES)]
    
    self.label_rec_time.setText("GRABANDO SECUENCIA CONTINUA...")

  def estado_guardar_secuencia_continua(self):
    """
    Guarda la sesión de grabación de la Secuencia Continua.
    A diferencia del modo palabra-por-palabra, esto guarda todo en un solo
    bloque, incluyendo un metadato 'words_sequence' que contiene la lista
    cíclica de palabras (ej. ['A', 'E', 'I', 'O', 'U']).
    Esto permite que los scripts de análisis (ej. correlaciondeseñales.py)
    lean este diccionario y etiqueten dinámicamente cada ventana recortada.
    """
    self.is_recording = False
    
    if hasattr(self, 'metronome_process') and self.metronome_process:
      try: self.metronome_process.kill()
      except: pass
      self.metronome_process = None
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    self.autoforge_estado_actual_str = "GUARDANDO SECUENCIA"
    self.autoforge_overlay.setText("<div align='center'>SECUENCIA COMPLETADA<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    
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
      
      # --- NUEVO: Evitar sobrescritura ---
      contador = 2
      while base_dir.exists():
        folder_name_v = f"{folder_name}_v{contador}"
        base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / folder_name_v
        contador += 1
        
      os.makedirs(base_dir, exist_ok=True)
      
      now = datetime.now()
      muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
      muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}
      total_pulsos = len(self.autoforge_words) * self.autoforge_target_reps
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
        "words_sequence": self.autoforge_words
      }
      
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

      try: guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      try: guardar_grabacion_wav(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      try: generar_grafico_grabacion(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      try: generar_grafico_estadisticas(self.stats_time, self.stats_snr, self.stats_noise_mean, self.stats_noise_std, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      self.current_recording = [] 
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    QtCore.QTimer.singleShot(5000, self.estado_finalizar_secuencia_continua)

  def estado_finalizar_secuencia_continua(self):
    self.autoforge_overlay.setText("<div align='center'>¡SECUENCIA COMPLETADA!</div>")
    QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
    self.is_acquiring = False
    self.is_autoforge_running = False
    self.is_autoforge_continuo = False
    self.config_stack.setCurrentIndex(0) # Restaurar configuración
    if hasattr(self, 'session_timer'):
      self.session_timer.stop()

  def _iniciar_timer_global(self):
    """
    Ejecuta la funcionalidad de _iniciar_timer_global.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    tiempo_por_palabra = 3.0 + self.spin_noise_duration.value() + (3 * (60.0/bpm)) + (self.autoforge_target_reps * (60.0/bpm)) + self.spin_descanso.value()
    self.tiempo_restante_global = int(len(self.autoforge_words) * tiempo_por_palabra)
    
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
      self.autoforge_overlay.setText("<div align='center'>¡AUTOGRABADO COMPLETADO!</div>")
      self.autoforge_overlay.show()
      QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
      self.current_recording = []
      self.is_acquiring = False
      self.is_autoforge_running = False
      self.config_stack.setCurrentIndex(0) # Restaurar configuración
      return
      
    palabra = self.autoforge_words[self.autoforge_word_idx]
    self.is_recording = False
    self.label_rec_time.setVisible(True) # --- NUEVO: Habilitado para no dar sensación de "congelamiento"
    target_h = self.config_groupbox.height()
    self.empty_recording_widget.setFixedHeight(target_h)
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
    
    # 3 Segundos de silencio ANTES de siquiera empezar a capturar el ruido
    QtCore.QTimer.singleShot(3000, self.estado_grabar_ruido)

  def estado_grabar_ruido(self):
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
    
    # --- NUEVO: Resetear escala Y para cada nueva palabra ---
    self.plot.setYRange(-0.01, 0.01)
    
    # Programar la aparición de la ventana "Preparate" y el metrónomo CUANDO TERMINE EL RUIDO
    noise_dur = self.spin_noise_duration.value()
    QtCore.QTimer.singleShot(int(noise_dur * 1000), self.estado_mostrar_preparate)

  def estado_mostrar_preparate(self):
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
          self.noise_levels[i] = np.mean(np.abs(all_noise))
          self.initial_noise_mean[i] = self.noise_levels[i]
          self.initial_noise_std[i] = np.std(all_noise)
          
          self.noise_lines[i].setPos(self.noise_levels[i])
          self.noise_lines[i].show()
          self.noise_lines_neg[i].setPos(-self.noise_levels[i])
          self.noise_lines_neg[i].show()
          self.noise_regions[i].setRegion([-self.noise_levels[i], self.noise_levels[i]])
          self.noise_regions[i].show()
          
          if hasattr(self, 'noise_status_labels'):
            self.noise_status_labels[i].setText(f"Ruido Base: x̄={self.initial_noise_mean[i]:.1f}µV, s={self.initial_noise_std[i]:.1f}µV")
            self.noise_status_labels[i].setStyleSheet("color: #00FFFF; font-size: 11px; font-weight: bold; background-color: #111111; border: 1px solid #00FFFF;")
      
      # --- NUEVO: Auto-ajuste del umbral de SNR ---
      if self.NUM_CANALES > 0:
        ruido_maximo_std = max(getattr(self, 'initial_noise_std', [0]))
        if ruido_maximo_std > 0:
          self.spin_peak_th.setValue(ruido_maximo_std * 5.0)
          
      self.noise_calculated = True
    
    # 2. Iniciamos el Count-In acústico (3 compases) antes de saltar a la palabra.
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    beat_interval_s = 60.0 / bpm
    
    palabra = self.autoforge_words[self.autoforge_word_idx]
    # Actualizamos la ventana flotante para que avise "PREPARATE"
    texto_ventana = f"SIGUIENTE:\n{palabra.upper()}"
    self.lbl_recording_space.setText(texto_ventana)
    self.word_window_process = None
    
    # Lanzar el metrónomo AHORA MISMO con un Count-in de tipo carreras (3 graves, 1 agudo)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
    
    import sys
    python_executable = sys.executable
    
    # Posicionar en la esquina superior derecha, pero debajo del panel de configuración
    pos = self.mapToGlobal(QtCore.QPoint(self.width(), 0))
    metro_x = pos.x() - 250
    metro_y = pos.y() + self.config_stack.height() + 10
    
    self.metronome_process = subprocess.Popen(
      [python_executable, script_path, '--autostart', '--count', f'--bpm={bpm}', '--count-in=4', f'--x={metro_x}', f'--y={metro_y}'],
      stdin=subprocess.PIPE,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      text=True,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
    )
    
    self.autoforge_overlay.hide()
    
    # --- NUEVO: Sincronización del texto amarillo de PREPARATE con el count-in acústico ---
    self.countdown_text.setPos(-self.PLOT_DURATION_S/2.0, 0)
    self.countdown_text.show()
    
    self.countdown_text.setText("PREPÁRATE...\n3")
    QtCore.QTimer.singleShot(int(beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n2"))
    QtCore.QTimer.singleShot(int(2 * beat_interval_s * 1000), lambda: self.countdown_text.setText("PREPÁRATE...\n1"))
    
    # A los 3 compases justos (cuando suene el pitido Agudo), pasamos a la palabra.
    QtCore.QTimer.singleShot(int(3 * beat_interval_s * 1000), self.estado_cambiar_ventana_palabra)

  def estado_cambiar_ventana_palabra(self):
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
    
    self.label_rec_time.setText("GRABANDO PALABRA...")
    
    palabra = self.autoforge_words[self.autoforge_word_idx]
    
    # Lanzamos la ventana limpia con solo la palabra para la grabación
    # El metrónomo YA ESTÁ corriendo, aquí justo emitirá el pitido Agudo (GO).
    self.lbl_recording_space.setText(palabra.upper())
    self.word_window_process = None
    
    # Iniciar la cuenta del guardado de los datos de la palabra
    bpm = self.spin_bpm.value()
    if bpm <= 0: bpm = 60
    ms_totales = int((60000 / bpm) * self.autoforge_target_reps)
    QtCore.QTimer.singleShot(ms_totales, self.estado_guardar_palabra)

  def estado_guardar_palabra(self):
    """
    Ejecuta la funcionalidad de estado_guardar_palabra.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.is_recording = False
    palabra = self.autoforge_words[self.autoforge_word_idx]
    
    if hasattr(self, 'metronome_process') and self.metronome_process:
      try: self.metronome_process.kill()
      except: pass
      self.metronome_process = None
      
    if hasattr(self, 'word_window_process') and self.word_window_process:
      try: self.word_window_process.kill()
      except: pass
      self.word_window_process = None
      
    t_descanso = self.spin_descanso.value()
    self.autoforge_estado_actual_str = f"DESCANSO {t_descanso:.1f}s (GUARDANDO)"
    self.autoforge_overlay.setText(f"<div align='center'>DESCANSO {t_descanso:.1f}s<br>GUARDANDO DATOS...</div>")
    self.autoforge_overlay.show()
    self.lbl_recording_space.setText("<div align='center' style='color:#777777; font-size:40px;'>DESCANSO</div>")
    
    import threading
    def guardar_async():
      """
      Ejecuta la funcionalidad de guardar_async.

      Returns:
        Any: Resultado de la ejecución de la función.
      """
      import os, json
      from pathlib import Path
      from datetime import datetime
      fecha_str = datetime.now().strftime("%Y-%m-%d")
      # Nombre: PALABRA_Prueba_Sujeto
      folder_name = f"{palabra}_{self.autoforge_prueba}_{self.autoforge_sujeto}"
      if getattr(sys, 'frozen', False):
        root_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(root_dir) == "_internal":
          root_dir = os.path.dirname(root_dir)
      else:
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
      base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / folder_name
      
      # --- NUEVO: Evitar sobrescritura ---
      contador = 2
      while base_dir.exists():
        folder_name_v = f"{folder_name}_v{contador}"
        base_dir = Path(root_dir) / "base_de_datos_electrodos" / fecha_str / folder_name_v
        contador += 1
        
      os.makedirs(base_dir, exist_ok=True)
      
      # 1. Preparar metadata para UI Analysis
      now = datetime.now()
      muscles_list = [self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)]
      muscles_map = {f"canal_{i}": self.nombres_musculos[i % len(self.nombres_musculos)] for i in range(self.NUM_CANALES)}
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
        "prueba": self.autoforge_prueba,
        "comentario": "Grabado mediante AutoForge"
      }
      
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

      # 3. Guardar Datos Raw CSV y WAV (las funciones internas ya manejan la división por canales)
      try: guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      try: guardar_grabacion_wav(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
      except: pass
      
      # 4. Generar Gráficos
      try: generar_grafico_grabacion(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      try: generar_grafico_estadisticas(self.stats_time, self.stats_snr, self.stats_noise_mean, self.stats_noise_std, str(base_dir), self.NUM_CANALES, self.CANALES_DAQ)
      except: pass
      
      self.current_recording = [] 
      
    threading.Thread(target=guardar_async, daemon=True).start()
    
    # Ir a la siguiente palabra tras los segundos de descanso
    QtCore.QTimer.singleShot(int(self.spin_descanso.value() * 1000), self.estado_siguiente)

  @QtCore.Slot()
  def estado_siguiente(self):
    """
    Ejecuta la funcionalidad de estado_siguiente.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.autoforge_word_idx += 1
    self.estado_iniciar_palabra()

  def _update_floating_windows_pos(self):
    pass

  def resizeEvent(self, event):
    """Maneja la redimensión de la ventana para ajustar las ventanas flotantes."""
    super().resizeEvent(event)
    QtCore.QTimer.singleShot(300, self._update_floating_windows_pos)

  def changeEvent(self, event):
    """Maneja eventos de cambio de estado (como pantalla completa)."""
    super().changeEvent(event)
    if event.type() == QtCore.QEvent.WindowStateChange:
      QtCore.QTimer.singleShot(300, self._update_floating_windows_pos)

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
    
    # --- NUEVO: Guardar estado de la GUI al cerrar el programa ---
    try:
      adq = self.config_mgr.config.get("adquisicion", {})
      adq["bpm"] = self.spin_bpm.value()
      adq["ruido_segundos"] = self.spin_noise_duration.value()
      adq["tiempo_descanso"] = self.spin_descanso.value()
      self.config_mgr.config["adquisicion"] = adq
      self.config_mgr.save()
      
      # Mantenemos esto por si alguna otra parte depende de metronome_config.json
      config_data = {}
      if os.path.exists('metronome_config.json'):
        with open('metronome_config.json', 'r', encoding='utf-8') as f:
          config_data = json.load(f)
      config_data['last_bpm'] = self.spin_bpm.value()
      config_data['notch_enabled'] = self.chk_notch_enable.isChecked()
      with open('metronome_config.json', 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=4)
    except Exception as e:
      print(f"Error guardando configuración al cerrar: {e}")
      
    event.accept() # Acepta el cierre

# =============================================================================
# --- NUEVO: DIÁLOGO DE SELECCIÓN DE MÚSCULOS AL INICIO ---
# =============================================================================
class MuscleSelectionDialog(QtWidgets.QDialog):
  def __init__(self, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Configurar Músculos de la Sesión")
    self.setMinimumWidth(350)
    
    self.config_mgr = ConfigManager()
    self.canales_conf = self.config_mgr.get("canales") or {}
    adq = self.config_mgr.get("adquisicion") or {}
    self.nidaq_chans = adq.get("nidaq_channels", ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"])
    
    self.layout = QtWidgets.QVBoxLayout(self)
    
    self.lbl = QtWidgets.QLabel("A continuación, asigne el músculo a cada canal activo:")
    self.lbl.setStyleSheet("color: white; font-weight: bold; margin-bottom: 10px;")
    self.layout.addWidget(self.lbl)
    
    self.form_layout = QtWidgets.QFormLayout()
    self.line_edits = {}
    
    for i in range(len(self.nidaq_chans)):
        key = f"Canal {i}"
        musculo_actual = self.canales_conf.get(key, {}).get("musculo", f"Canal {i}")
        le = QtWidgets.QLineEdit(musculo_actual)
        self.form_layout.addRow(f"[{self.nidaq_chans[i]}] {key}:", le)
        self.line_edits[key] = le
        
    self.layout.addLayout(self.form_layout)
    
    self.btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
    self.btn_box.accepted.connect(self.guardar_y_cerrar)
    self.btn_box.rejected.connect(self.reject)
    self.layout.addWidget(self.btn_box)
    
  def guardar_y_cerrar(self):
    try:
      from utils.config_manager import get_muscle_color
    except ImportError:
      def get_muscle_color(name, default="#00ffcc"):
        return "#ff0000" if ("mic" in str(name).lower() or "canal 3" in str(name).lower()) else default

    for key, le in self.line_edits.items():
        if key not in self.canales_conf:
            self.canales_conf[key] = {}
        m_text = le.text().strip()
        self.canales_conf[key]["musculo"] = m_text
        if key == "Canal 3" or "mic" in m_text.lower():
            self.canales_conf[key]["color_hex"] = "#ff0000"
        else:
            self.canales_conf[key]["color_hex"] = get_muscle_color(m_text, self.canales_conf[key].get("color_hex", "#00ffcc"))
    
    self.config_mgr.config["canales"] = self.canales_conf
    self.config_mgr.save()
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
  
  # --- NUEVO: Preguntar por los músculos al iniciar ---
  # Ocultamos temporalmente el splash si existe, para que no tape el messagebox
  if global_splash:
      global_splash.hide()
      
  respuesta = QtWidgets.QMessageBox.question(
      None, 
      "Ñandú LSD - Configuración", 
      "¿Desea cambiar el conjunto de músculos asignados a los canales?",
      QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
      QtWidgets.QMessageBox.No
  )
  
  if respuesta == QtWidgets.QMessageBox.Yes:
      dialog = MuscleSelectionDialog()
      dialog.exec()
      
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
