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
__version__ = "4.3"

import numpy as np
from scipy.io.wavfile import write as write_wav
from datetime import datetime
import time
import queue
import threading
import sys
import json
import subprocess # <-- NUEVO: para guardar el metadata
import os # Necesario para separar nombres de archivo

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
import matplotlib.pyplot as plt 

# Configuración de PyQtGraph para MÁXIMO rendimiento
pg.setConfigOptions(antialias=False) # Optimización

# Intenta importar nidaqmx
try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType
    from nidaqmx.stream_readers import AnalogMultiChannelReader 
    NIDAQMX_DISPONIBLE = True
except ImportError:
    NIDAQMX_DISPONIBLE = False
    print("Advertencia: La librería 'nidaqmx' no está instalada. El programa solo funcionará en MODO_PRUEBA.")

# =============================================================================
# --- CONFIGURACIÓN PRINCIPAL ---
# =============================================================================
# CANALES_DAQ = [f"{DEVICE_NAME}/ai0", f"{DEVICE_NAME}/ai1", f"{DEVICE_NAME}/ai2"]
# NUM_CANALES = len(CANALES_DAQ)

# --- ESTOS VALORES AHORA SE CONFIGURAN DESDE LA GUI ---
# SAMPLE_RATE = 6000
# PLOT_DURATION_S = 30

CHUNK_SAMPLES = 400
# CHUNK_DURATION_S y PLOT_SAMPLES se calculan dinámicamente
# =============================================================================
# BLOQUE 1: HILO DE ADQUISICIÓN (Sin cambios)
# =============================================================================
def acquisition_thread(device_channels, sample_rate, chunk_samples, num_canales, data_queue, stop_event):
    print(f"Iniciando hilo de adquisición con SR={sample_rate} Hz...")
    try:
        with nidaqmx.Task() as task:
            for canal in device_channels:
                task.ai_channels.add_ai_voltage_chan(canal)
            
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

def simulador_thread(chunk_samples, sample_rate, num_canales, data_queue, stop_event, test_freq=50):
    print(f"Iniciando hilo de simulación con SR={sample_rate} Hz y Frec. Prueba={test_freq} Hz...")
    
    # --- MEJORA: Usar el tiempo real para una simulación más fluida ---
    # En lugar de un tiempo_acumulado propenso a errores por la imprecisión de time.sleep(),
    # usamos el tiempo real del sistema para generar la señal.
    start_time = time.perf_counter()
    samples_generados = 0

    while not stop_event.is_set():
        # Calculamos el tiempo actual basado en el número de muestras que ya hemos generado.
        # Esto asegura que la fase de la onda sinusoidal sea siempre continua.
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
            print(f"   ✅ Canal {i} guardado como: {nombre_archivo_canal}")
        except Exception as e:
            print(f"   ❌ Error al guardar el .wav del canal {i}: {e}")
            exito_total = False
    return exito_total

# =============================================================================
# BLOQUE 3.2: GUARDADO DE DATOS EN .CSV (Sin cambios)
# =============================================================================
def guardar_grabacion_csv(datos_completos, sample_rate, output_dir, num_canales, base_name="grabacion"):
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
        print(f"   ✅ CSV guardado exitosamente.")
        return True
    except Exception as e:
        print(f"   ❌ Error al guardar el CSV: {e}")
        return False

# =============================================================================
# BLOQUE 3.3: GENERADOR DE GRÁFICO (Sin cambios)
# =============================================================================
def generar_grafico_grabacion(datos_completos, sample_rate, output_dir, num_canales, canales_daq, base_name="photo"):
    if not datos_completos:
        return False

    print("Generando gráfico de la grabación completa...")
    try:
        grabacion = np.concatenate(datos_completos, axis=1)
    except ValueError:
        return False
    
    # Crear vector de tiempo
    num_muestras = grabacion.shape[1]
    tiempo = np.arange(num_muestras) / float(sample_rate)
    
    # Crear figura
    fig, axs = plt.subplots(
        num_canales, 
        1, 
        figsize=(15, 3 * num_canales), # 3 pulgadas de alto por canal
        sharex=True
    )
    
    # Si hay un solo canal, axs no es un array, hay que manejarlo
    if num_canales == 1:
        axs = [axs]
        
    fig.suptitle(f"Grabación Completa - {base_name}", fontsize=16)

    # Graficar cada canal
    for i in range(num_canales):
        axs[i].plot(tiempo, grabacion[i])
        axs[i].set_ylabel(f"Amplitud (V)")
        axs[i].set_title(f"Canal {i} ({canales_daq[i]})")
        axs[i].grid(True)
        
    axs[-1].set_xlabel("Tiempo (s)")
    
    # Definir nombre de archivo
    nombre_archivo_grafico = os.path.join(output_dir, f"{base_name}.png")

    # Guardar
    try:
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Ajuste para el supertítulo
        plt.savefig(nombre_archivo_grafico, dpi=200) # dpi=200 es un buen balance
        print(f"   ✅ Gráfico guardado como: {nombre_archivo_grafico}")
        return True
    except Exception as e:
        print(f"   ❌ Error al guardar el gráfico: {e}")
        return False
    finally:
        plt.close(fig) # Liberar memoria

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
        super().__init__(parent)
        self.setWindowTitle("Guardar Medición")
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

class RealTimePlotter(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # --- Constantes de la GUI ---
        self.BTN_START_STYLE = "background-color: #007BFF; color: white; font-weight: bold; padding: 5px;"
        self.BTN_STOP_STYLE = "background-color: #DC3545; color: white; font-weight: bold; padding: 5px;"
        self.BTN_REC_START_STYLE = "background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;"
        self.BTN_REC_STOP_STYLE = "background-color: #F44336; color: white; font-weight: bold; padding: 5px;"

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

        # --- NUEVO: Buffers y configuración del espectrograma ---
        self.spectrogram_buffer = None
        self.SPECTROGRAM_HISTORY_LEN = 200 # Número de FFTs en el historial del espectrograma
        self.SPECTROGRAM_FFT_LEN = 256     # Puntos para la FFT (mejor si es potencia de 2)
        self.spectrogram_channel_index = 0
        

        # --- Estado del Trigger ---
        self.trigger_last_values = None
        self.is_trigger_line_moving = False # Para evitar bucles de señales

        # --- NUEVO: Estado del Filtro ---
        self.filter_sos = None # Coeficientes del filtro
        self.filter_zi = None  # Estado inicial del filtro para cada canal
        # --- NUEVO: Estado del Filtro Notch ---
        self.notch_sos = None
        self.notch_zi = None

        self.FILTER_ORDER = 4  # Orden del filtro Butterworth

        # --- Configurar la ventana principal ---
        self.setWindowTitle(f"Visor y Grabador de Señales Emg v{__version__}")
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

    def _load_protocol_config(self):
        bpm = 60
        pulse_count = 30
        if os.path.exists('metronome_config.json'):
            try:
                with open('metronome_config.json', 'r') as f:
                    data = json.load(f)
                    bpm = data.get('last_bpm', 60)
                    pulse_count = data.get('last_beat_count', 30)
            except Exception:
                pass # Usar defaults si hay error
        self.spin_bpm.setValue(bpm)

    def _setup_ui_layouts(self):
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
        self.chk_modo_prueba = QtWidgets.QCheckBox("Modo Prueba")
        self.chk_modo_prueba.setToolTip("Usa datos simulados en lugar de la tarjeta NI-DAQ.")

        # --- NUEVO: Checkbox para el Metrónomo ---
        self.chk_use_metronome = QtWidgets.QCheckBox("Usar Metrónomo")
        self.chk_use_metronome.setToolTip("Abre y controla el metrónomo visual durante la adquisición.")

        # --- NUEVO: Control de Frecuencia de Prueba ---
        self.label_test_freq = QtWidgets.QLabel("Frec. Prueba (Hz):")
        self.spin_test_freq = SpinBox(value=50, step=10, bounds=(1, 5000), int=True)
        self.spin_test_freq.setFixedWidth(80)

        self.label_sample_rate = QtWidgets.QLabel("Sample Rate (S/s):")
        self.cmb_sample_rate = QtWidgets.QComboBox()
        self.cmb_sample_rate.addItems(["6000", "10000", "20000", "44100", "50000"])
        self.cmb_sample_rate.setCurrentText("6000")

        self.label_plot_duration = QtWidgets.QLabel("Duración Plot (s):")
        self.spin_plot_duration = SpinBox(value=10, step=1, bounds=(1, 60), int=True)
        self.spin_plot_duration.setFixedWidth(80)

        self.label_channels = QtWidgets.QLabel("Canales:")
        self.channel_checkboxes = []
        self.channel_layout = QtWidgets.QHBoxLayout()
        for i in range(8): # Ofrecer 8 canales
            chk = QtWidgets.QCheckBox(f"ai{i}")
            self.channel_checkboxes.append(chk)
            self.channel_layout.addWidget(chk)
        # Marcar los 3 primeros por defecto
        self.channel_checkboxes[0].setChecked(True)
        self.channel_checkboxes[1].setChecked(True)
        self.channel_checkboxes[2].setChecked(True)

        self.btn_start_acq = QtWidgets.QPushButton("Iniciar Adquisición")
        self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)

        # --- NUEVO: Controles de Protocolo ---
        self.label_bpm = QtWidgets.QLabel("BPM:")
        self.spin_bpm = SpinBox(value=60, int=True, bounds=(30, 200), step=1)
        self.spin_bpm.setFixedWidth(80) # Acortar el recuadro del BPM
        self.label_noise_duration = QtWidgets.QLabel("Noise (Inicio) (s):")
        self.spin_noise_duration = SpinBox(value=2.0, dec=True, bounds=(0.5, 10.0), step=0.5)
        self.spin_noise_duration.setFixedWidth(80)

        self.config_layout.addWidget(self.label_device, 0, 0)
        self.config_layout.addWidget(self.cmb_device, 0, 1)
        self.config_layout.addWidget(self.chk_modo_prueba, 0, 2)
        self.config_layout.addWidget(self.chk_use_metronome, 0, 3)
        self.config_layout.addWidget(self.label_test_freq, 0, 4)
        self.config_layout.addWidget(self.spin_test_freq, 0, 5)
        self.config_layout.addWidget(self.label_sample_rate, 1, 0) # Fila 1
        self.config_layout.addWidget(self.cmb_sample_rate, 1, 1) 
        self.config_layout.addWidget(self.label_plot_duration, 1, 2)
        self.config_layout.addWidget(self.spin_plot_duration, 1, 3)
        self.config_layout.addWidget(self.label_bpm, 1, 5)
        self.config_layout.addWidget(self.spin_bpm, 1, 6) # BPM
        self.config_layout.addWidget(self.label_noise_duration, 2, 5)
        self.config_layout.addWidget(self.spin_noise_duration, 2, 6) # Ruido
        self.config_layout.addWidget(self.label_channels, 2, 0)
        self.config_layout.addLayout(self.channel_layout, 2, 1, 1, 4) # row, col, rowspan, colspan
        self.config_layout.addWidget(self.btn_start_acq, 0, 7, 4, 1) # Ocupa 4 filas

    def _setup_ui_filter_panel(self):
        """Configura el panel de filtro."""
        self.chk_notch_enable = QtWidgets.QCheckBox("Filtro Notch 50 Hz")
        self.chk_notch_enable.setToolTip("Aplica un filtro para eliminar el ruido de la red eléctrica (50 Hz).")

        self.chk_filter_enable = QtWidgets.QCheckBox("Habilitar Filtro")
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
        self.btn_record = QtWidgets.QPushButton("Empezar a Grabar")
        self.btn_record.setStyleSheet(self.BTN_REC_START_STYLE)
        
        self.label_rec_time = QtWidgets.QLabel("Grabando: --:--.-")
        self.label_rec_time.setStyleSheet("font-weight: bold; color: #E91E63;")
        self.label_rec_time.setVisible(False)
        
        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll (Armar Trigger)")
        self.chk_autoscroll.setChecked(True)
        
        # --- Armar layout de botones ---
        self.button_layout.addWidget(self.btn_record)
        self.button_layout.addWidget(self.label_rec_time)
        self.button_layout.addSpacing(20)
        self.button_layout.addStretch(1) # Espaciador
        self.button_layout.addWidget(self.chk_autoscroll)
        
        # --- Widgets del Trigger ---
        self.chk_trigger = QtWidgets.QCheckBox("Habilitar Trigger")
        self.label_trig_chan = QtWidgets.QLabel("Canal:")
        self.cmb_trig_chan = QtWidgets.QComboBox()
        # Se poblará al iniciar la adquisición
        
        self.label_trig_level = QtWidgets.QLabel("Nivel (V):")
        self.spin_trig_level = SpinBox(value=0.5, step=0.1, dec=True, minStep=0.01, bounds=(-10.0, 10.0))
        self.spin_trig_level.setFixedWidth(80)
        
        self.label_trig_edge = QtWidgets.QLabel("Flanco:")
        self.cmb_trig_edge = QtWidgets.QComboBox()
        self.cmb_trig_edge.addItems(["Subida ↗", "Bajada ↘"])

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
        self.trigger_layout.addStretch(1) # Espaciador
        
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
        # Widget para el ploteo de tiempo
        self.plot_widget = pg.GraphicsLayoutWidget()
        self.plot_widget.setBackground('k') # Fondo Negro
        
        self.plot = self.plot_widget.addPlot(title="Canales de Adquisición")
        self.plot.setLabel('bottom', 'Tiempo (s)')
        self.plot.setLabel('left', 'Amplitud (V)')

        # Widget para el espectrograma (ImageView)
        self.spectrogram_view = pg.ImageView()
        self.spectrogram_view.ui.histogram.hide()
        self.spectrogram_view.ui.roiBtn.hide()
        self.spectrogram_view.ui.menuBtn.hide()

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
        self.trigger_line = InfiniteLine(pos=0.5, angle=0, movable=True, pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
        self.plot.addItem(self.trigger_line)
        # --- NUEVO: Etiqueta de texto para el valor del trigger ---
        self.trigger_label = pg.TextItem(anchor=(0, 1), color=(255, 255, 0), fill=(0, 0, 0, 150))
        self.trigger_label.setZValue(100) # Asegurar que esté por encima de las curvas
        self.trigger_label.hide() # Oculto por defecto
        self.plot.addItem(self.trigger_label)

    def _setup_ui_final_layout(self):
        
        # --- NUEVO: Divisor para los gráficos ---
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.splitter.addWidget(self.plot_widget)
        self.splitter.addWidget(self.spectrogram_view)
        self.splitter.setSizes([600, 200]) # Tamaños iniciales

        # --- Añadir layouts a la ventana ---
        self.main_layout.addWidget(self.config_groupbox)
        self.main_layout.addWidget(self.filter_groupbox)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.trigger_layout)
        self.main_layout.addWidget(self.measure_widget)
        self.main_layout.addWidget(self.spectrogram_groupbox) # Añadir controles del espectrograma
        self.main_layout.addWidget(self.splitter) # Añadir el divisor con los gráficos

    def _connect_signals(self):
        
        # --- Conectar Señales (Botones) ---
        self.btn_start_acq.clicked.connect(self.on_start_acq_click)
        self.btn_record.clicked.connect(self.on_record_click)
        self.chk_autoscroll.clicked.connect(self.on_autoscroll_toggle)
        
        # --- Conectar Widgets del Trigger ---
        self.chk_trigger.clicked.connect(self.on_trigger_enable_toggle)
        self.trigger_line.sigPositionChanged.connect(self.on_trigger_line_moved)
        # --- CORREGIDO: Conectar aquí la señal para ocultar la etiqueta ---
        self.trigger_line.sigPositionChangeFinished.connect(lambda: self.trigger_label.hide())
        self.spin_trig_level.valueChanged.connect(self.on_trigger_level_changed)

        # --- NUEVO: Conectar Checkbox de Modo Prueba ---
        self.chk_modo_prueba.toggled.connect(self._on_modo_prueba_toggled)

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
        self.colores_curvas = ['y', 'c', 'm', 'g', 'r', 'w', (255, 165, 0), (128, 0, 128)] # y,c,m,g,r,w,orange,purple

    def _on_modo_prueba_toggled(self, checked):
        """Se llama cuando el checkbox de Modo Prueba cambia."""
        self.cmb_device.setEnabled(not checked)
        # --- NUEVO: Habilitar/deshabilitar control de frecuencia ---
        self.label_test_freq.setEnabled(checked)
        self.spin_test_freq.setEnabled(checked)
        print(f"Modo Prueba {'Activado' if checked else 'Desactivado'}.")

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
        if self.is_acquiring:
            # --- DETENER ADQUISICIÓN ---
            print("Deteniendo adquisición...")
            self.stop_event.set()
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
                self.metronome_process.wait(timeout=2) # Esperar a que se cierre solo
                self.metronome_process = None

            self._on_modo_prueba_toggled(self.chk_modo_prueba.isChecked()) # Restaurar estado de habilitación
            if self.acquisition_thread:
                self.acquisition_thread.join(timeout=2.0)
            self.acquisition_thread = None
            self.is_acquiring = False
            
            self.btn_start_acq.setText("Iniciar Adquisición")
            self.btn_start_acq.setStyleSheet(self.BTN_START_STYLE)
            self.cmb_device.setEnabled(True)
            self.cmb_sample_rate.setEnabled(True)
            self.chk_modo_prueba.setEnabled(True)
            self.spin_plot_duration.setEnabled(True)
            self.chk_use_metronome.setEnabled(True)
            for chk in self.channel_checkboxes: chk.setEnabled(True)
            
            self.set_controls_enabled(False)
            if self.is_recording: # Si estaba grabando, detenerla
                self.on_record_click()

        else:
            # --- INICIAR ADQUISICIÓN ---
            device = self.cmb_device.currentText()
            
            # Leer nuevos parámetros de la GUI
            self.SAMPLE_RATE = int(self.cmb_sample_rate.currentText())
            self.PLOT_DURATION_S = self.spin_plot_duration.value()
            self.PLOT_SAMPLES = int(self.PLOT_DURATION_S * self.SAMPLE_RATE)
            self.CHUNK_DURATION_S = CHUNK_SAMPLES / self.SAMPLE_RATE

            selected_channels = [chk.text() for chk in self.channel_checkboxes if chk.isChecked()]
            
            if not selected_channels:
                print("Error: Debes seleccionar al menos un canal.")
                return

            self.CANALES_DAQ = [f"{device}/{ch}" for ch in selected_channels]
            self.NUM_CANALES = len(self.CANALES_DAQ)
            print(f"Iniciando con SR={self.SAMPLE_RATE}, Plot={self.PLOT_DURATION_S}s, Canales={self.CANALES_DAQ}")

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
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
                    print("Lanzando metrónomo con --autostart...")
                    # --- MODIFICADO: Añadir stdin=subprocess.PIPE para poder enviarle comandos ---
                    self.metronome_process = subprocess.Popen(
                        [python_executable, script_path, '--autostart'],
                        stdin=subprocess.PIPE,
                        text=True # Para enviar texto en lugar de bytes
                    )
                except Exception as e:
                    print(f"Error al lanzar el metrónomo: {e}")
                    self.metronome_process = None

            # Iniciar hilo
            self.stop_event.clear()
            if self.chk_modo_prueba.isChecked():
                test_freq = self.spin_test_freq.value()
                self.acquisition_thread = threading.Thread(target=simulador_thread, args=(CHUNK_SAMPLES, self.SAMPLE_RATE, self.NUM_CANALES, self.data_queue, self.stop_event, test_freq), daemon=True)
            else:
                if not NIDAQMX_DISPONIBLE:
                    print("Error: nidaqmx no encontrado. No se puede correr en modo real.")
                    return
                self.acquisition_thread = threading.Thread(target=acquisition_thread, args=(self.CANALES_DAQ, self.SAMPLE_RATE, CHUNK_SAMPLES, self.NUM_CANALES, self.data_queue, self.stop_event), daemon=True)
            
            self.acquisition_thread.start()
            self.is_acquiring = True
            
            self.btn_start_acq.setText("Detener Adquisición")
            self.btn_start_acq.setStyleSheet(self.BTN_STOP_STYLE)
            self.cmb_device.setEnabled(False)
            self.cmb_sample_rate.setEnabled(False)
            self.chk_modo_prueba.setEnabled(False)
            self.spin_plot_duration.setEnabled(False)
            self.chk_use_metronome.setEnabled(False)
            for chk in self.channel_checkboxes: chk.setEnabled(False)
            
            self.set_controls_enabled(True)

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
        for label in self.measure_labels: label.deleteLater()
        self.curvas.clear()
        self.measure_labels.clear()
        self.cmb_spectrogram_chan.clear()
        self.cmb_trig_chan.clear()
        self.plot.clear() # Limpia todo, incluyendo leyenda
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
            self.curvas.append(self.plot.plot(pen=color, name=f'Canal {i}'))
            
            # --- NUEVO: Crear un QFrame para cada medición ---
            measurement_frame = QtWidgets.QFrame()
            measurement_frame.setStyleSheet("background-color: black; border: 1px solid gray; border-radius: 5px;")
            
            frame_layout = QtWidgets.QVBoxLayout(measurement_frame)
            frame_layout.setContentsMargins(5, 2, 5, 2) # Pequeño padding interno
            
            label = QtWidgets.QLabel(f"<b>Ch {i}:</b> -- Vp-p, -- Vrms")
            label.setStyleSheet(f"color: {pg.mkColor(color).name()};") # Color del texto igual al de la curva
            frame_layout.addWidget(label)
            
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
        self.spectrogram_buffer = np.zeros((self.SPECTROGRAM_HISTORY_LEN, self.SPECTROGRAM_FFT_LEN // 2 + 1))
        self.on_spectrogram_enable_toggle() # Para mostrar/ocultar la vista

        # --- CORRECCIÓN v3: Configurar la transformación del espectrograma una sola vez ---
        # En lugar de pasar 'scale' en cada llamada a setImage, lo que puede ser propenso a errores,
        # configuramos la transformación directamente en el ViewBox del ImageView.
        # Esto es más eficiente y robusto.
        view = self.spectrogram_view.getView()
        
        # Eje Y (Frecuencia): Va de 0 a Frecuencia de Nyquist.
        nyquist = self.SAMPLE_RATE / 2.0
        freq_scale = nyquist / (self.SPECTROGRAM_FFT_LEN / 2 + 1)

        # Eje X (Tiempo): El ancho total del historial del espectrograma en segundos.
        # Cada columna representa un segmento de tiempo (nperseg - noverlap) / fs.
        time_per_column = (self.SPECTROGRAM_FFT_LEN - (self.SPECTROGRAM_FFT_LEN // 2)) / self.SAMPLE_RATE
        time_scale = time_per_column

        view.setAspectLocked(False) # Desbloquear la relación de aspecto para escalar ejes independientemente
        view.setRange(xRange=(0, self.SPECTROGRAM_HISTORY_LEN * time_scale), yRange=(0, nyquist), padding=0)

        # Inicializar/Resetear el filtro
        self.on_filter_settings_changed()

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
            f0 = 50.0  # Frecuencia a remover
            Q = 30.0   # Factor de calidad
            
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
        self.trigger_label.setText(f"Nivel: {y_pos:.3f} V")
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
            self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)
            self.plot_widget.setBackground('k') # Color de fondo negro
        else:
            # MODO ANÁLISIS (Congelado)
            self.plot.setMouseEnabled(x=True, y=True) # Zoom X e Y
            self.plot.getViewBox().enableAutoRange(pg.ViewBox.XAxis)

    def on_record_click(self):
        if not self.is_recording:
            # --- EMPEZAR A GRABAR ---
            self.is_recording = True
            self.counting_started = False # Reiniciar la bandera de conteo
            self.current_recording.clear()
            # --- NUEVO: Re-lanzar el metrónomo si es necesario ---
            # Si se usó el metrónomo en la adquisición actual pero el proceso ya no existe
            # (porque se cerró en una grabación anterior), lo volvemos a lanzar.
            if self.chk_use_metronome.isChecked() and not self.metronome_process:
                try:
                    python_executable = sys.executable
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
                    print("Re-lanzando metrónomo para nueva grabación...")
                    self.metronome_process = subprocess.Popen(
                        [python_executable, script_path, '--autostart'],
                        stdin=subprocess.PIPE,
                        text=True
                    )
                except Exception as e:
                    print(f"Error al re-lanzar el metrónomo: {e}")
            
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
            self.recording_start_time = None
            
            # --- MEJORA: Terminar el proceso del metrónomo solo si todavía existe ---
            if self.metronome_process and self.metronome_process.poll() is None:
                try:
                    print("Enviando comando STOP_APP al metrónomo al detener grabación...")
                    self.metronome_process.stdin.write("STOP_APP\n")
                    self.metronome_process.stdin.flush()
                except (OSError, ValueError) as e:
                    print(f"Advertencia: No se pudo comunicar con el proceso del metrónomo. Error: {e}")
            
            if self.metronome_process:
                self.metronome_process.wait(timeout=2) # Esperar a que el proceso termine y guarde el archivo
                self.metronome_process = None

            if self.current_recording:
                self.on_export_click() # <-- LLAMADA AUTOMÁTICA A EXPORTAR (AHORA DESPUÉS DE CERRAR EL METRÓNOMO)

    # --- MODIFICADO v3.12: Manejo de errores de exportación individual ---
    def on_export_click(self):
        # --- MODIFICADO: Usar el nuevo diálogo personalizado ---
        dialog = SaveMeasurementDialog(self)
        result = dialog.exec_() # Esto muestra el diálogo y espera

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
        base_dir = "base_de_datos_electrodos"
        output_dir = os.path.join(base_dir, measurement_name)
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"--- ❌ ERROR FATAL AL CREAR DIRECTORIO ---\n{e}")
            return

        # --- NUEVO: Crear las carpetas de los canales desde el principio ---
        for i in range(self.NUM_CANALES):
            channel_output_dir = os.path.join(output_dir, f"canal_{i}")
            os.makedirs(channel_output_dir, exist_ok=True)
        print(f"   ✅ Creadas {self.NUM_CANALES} carpetas de canal en '{output_dir}'")

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
                        print(f"✅ Guardados {pulse_count_from_metronome} pulsos en el metadata.")
                    else:
                        print("⚠️ No se encontró 'last_beat_count' en metronome_config.json.")
            except Exception as e:
                print(f"Advertencia: No se pudo leer el conteo de pulsos desde metronome_config.json. Error: {e}")

        # --- NUEVO: Guardar metadata.json con la fecha y hora ---
        metadata = {
            "measurement_date": datetime.now().isoformat(),
            "sample_rate": self.SAMPLE_RATE,
            "channels": self.CANALES_DAQ,
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
        # --- CORRECCIÓN: Guardar metadata ÚNICAMENTE en la carpeta de cada canal ---
        for i in range(self.NUM_CANALES):
            current_dir = os.path.join(output_dir, f"canal_{i}")
            metadata_path = os.path.join(current_dir, "metadata.json")
            try:
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=4)
                print(f"   ✅ Metadata guardado en: {metadata_path}")
            except Exception as e:
                print(f"--- ❌ ERROR AL GUARDAR METADATA.JSON en '{current_dir}' ---\n{e}")

        # 1. Guarda el .wav
        try:
            guardar_grabacion_wav(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES)
        except Exception as e:
            print(f"--- ❌ ERROR FATAL AL GUARDAR ARCHIVOS .WAV ---\n{e}")

        # 2. Guarda el .csv
        try:
            guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES)
        except Exception as e:
            print(f"--- ❌ ERROR FATAL AL GUARDAR .CSV ---\n{e}")

        # 3. Genera el gráfico .png
        try:
            generar_grafico_grabacion(self.current_recording, self.SAMPLE_RATE, output_dir, self.NUM_CANALES, self.CANALES_DAQ)
        except Exception as e:
            print(f"--- ❌ ERROR FATAL AL GUARDAR .PNG ---\n{e}")
            print("   (¿Estás seguro de que 'matplotlib' está instalado? -> pip install matplotlib)")

        print("--- EXPORTACIÓN FINALIZADA ---")

    def actualizar_plot(self):
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

            # 3. Grabar si es necesario
            if self.is_recording:
                self.current_recording.append(all_new_data)

            # 4. Aplicar filtros (si están habilitados) al bloque completo
            processed_data = all_new_data

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
            self.plot_buffer_datos = np.roll(self.plot_buffer_datos, -total_muestras_leidas, axis=1)
            self.plot_buffer_datos[:, -total_muestras_leidas:] = processed_data

            # 6. Actualizar la GUI
            for i in range(self.NUM_CANALES):
                self.curvas[i].setData(self.plot_vector_tiempo, self.plot_buffer_datos[i])
            
            if self.chk_autoscroll.isChecked():
                self.plot.setXRange(-self.PLOT_DURATION_S, 0, padding=0)

            self.check_for_trigger(processed_data, total_muestras_leidas)
            self.actualizar_mediciones(processed_data)
            self.actualizar_espectrograma(processed_data)

            # --- Actualiza el cronómetro (esto se ejecuta siempre) ---
            if self.is_recording and self.recording_start_time is not None:
                elapsed_time = time.perf_counter() - self.recording_start_time
                noise_dur = self.spin_noise_duration.value()

                if elapsed_time < noise_dur:
                    time_str = f"GRABANDO RUIDO ({elapsed_time:.1f}s / {noise_dur:.1f}s)"
                    # (Opcional pero recomendado) Cambiar el color para la fase de ruido
                    self.label_rec_time.setStyleSheet("font-weight: bold; color: #FFA500;") # Naranja
                else:
                    signal_time = elapsed_time - noise_dur
                    minutes = int(signal_time // 60)
                    seconds = int(signal_time % 60)
                    tenths = int((signal_time % 1) * 10)
                    time_str = f"GRABANDO SEÑAL: {minutes:02d}:{seconds:02d}.{tenths}"
                    # Restaurar color original de grabación
                    
                    # --- NUEVO: Iniciar el conteo del metrónomo justo aquí ---
                    if not self.counting_started and self.metronome_process:
                        # --- MEJORA: Comprobar si el proceso sigue vivo antes de enviar el comando ---
                        if self.metronome_process.poll() is None:
                            try:
                                print("Enviando comando START_COUNTING al metrónomo (inicio de señal).")
                                self.metronome_process.stdin.write("START_COUNTING\n")
                                self.metronome_process.stdin.flush()
                            except (OSError, ValueError): pass # Ignorar si la pipe ya se cerró
                        self.counting_started = True

                    self.label_rec_time.setStyleSheet("font-weight: bold; color: #E91E63;")

                self.label_rec_time.setText(time_str)
            
        except queue.Empty:
            pass # Si no hay datos, no hace nada

    def actualizar_mediciones(self, chunk_data):
        """Calcula y actualiza los labels de Vp-p y RMS."""
        if not self.is_acquiring or self.NUM_CANALES == 0:
            return

        try:
            # Cálculo vectorizado (muy rápido)
            max_vals = np.max(chunk_data, axis=1)
            min_vals = np.min(chunk_data, axis=1)
            vp_p = max_vals - min_vals
            
            # RMS = sqrt( mean( x^2 ) )
            rms = np.sqrt(np.mean(np.square(chunk_data), axis=1))
            
            # Actualiza los labels
            for i in range(self.NUM_CANALES):
                self.measure_labels[i].setText(
                    f"<b>Ch {i}:</b> {vp_p[i]:.3f} Vp-p, {rms[i]:.3f} Vrms"
                )
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
            f, t, Zxx = signal.stft(data_canal, fs=self.SAMPLE_RATE, nperseg=self.SPECTROGRAM_FFT_LEN)
            
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

    def closeEvent(self, event):
        # Se llama cuando el usuario cierra la ventana
        print("Ventana de ploteo cerrada. Deteniendo hilos...")
        self.stop_event.set() # Envía la señal a TODOS los hilos
        self.timer.stop() # Detiene el timer de la GUI
        event.accept() # Acepta el cierre

# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================
if __name__ == '__main__':
    # Inicia la GUI
    app = QtWidgets.QApplication(sys.argv)
    gui = RealTimePlotter()
    gui.show()
    
    # Inicia el bucle de la aplicación (bloqueante)
    exit_code = app.exec_()
    sys.exit(exit_code)

    # --- Esto se ejecuta DESPUÉS de que se cierra la GUI ---
    # La gestión de hilos ahora se hace en los métodos de la clase GUI
    # hilo.join() ya no es necesario aquí, se maneja en on_start_acq_click y closeEvent
    
    print("Programa finalizado.")
