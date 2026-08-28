# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Visualización gráfica de señales EMG calibradas (multi-archivo).
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Visualización gráfica de señales EMG calibradas (multi-archivo).
# ==============================================================================

# -*- coding: utf-8 -*-
"""
plotter_calibrado_secuencial_final.py

Script de visualización EMG MULTI-ARCHIVO.
Guarda automáticamente en la carpeta de origen con formato: plot_calibrado_{nombre}.png
"""

import os
import pandas as pd
import numpy as np
import json
from scipy import signal

import os
import pandas as pd
import numpy as np
import json
import sys
from scipy import signal

# --- Mantenemos Matplotlib para los gráficos, forzando backend a QtAgg ---
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt

# Imports de PySide6
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QListWidget, QLabel, QLineEdit, QPushButton, 
    QMessageBox, QGroupBox, QFormLayout, QCheckBox, 
    QRadioButton, QButtonGroup, QAbstractItemView
)
from PySide6.QtCore import Qt

# --- 1. CONFIGURACIÓN GENERAL ---

if getattr(sys, 'frozen', False):
    root_dir = os.path.dirname(os.path.abspath(sys.executable))
    if os.path.basename(root_dir) == "_internal":
        root_dir = os.path.dirname(root_dir)
else:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- CORRECCIÓN: Apuntar a la base de datos en la raíz del proyecto, no en la carpeta analysis ---
BASE_DIR = os.path.join(root_dir, "base_de_datos_electrodos")

if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.config_manager import ConfigManager
config_mgr = ConfigManager()

# Parámetros Fijos
FREQ_NOTCH = 50.0            
Q_FACTOR_NOTCH = 2.0        
FREQ_PASABANDA = [20, 1000]  
ORDEN_PASABANDA = 4          
RMS_WINDOW_MS = 75           

# --- 2. CLASES DE INTERFAZ (GUI) PySide6 ---

class PlotterConfigDialog(QDialog):
    """
    Clase PlotterConfigDialog.

    Representa y gestiona las operaciones relacionadas con PlotterConfigDialog.
    """
    def __init__(self, parent=None, pre_selected_paths=None):
        """
        Ejecuta la funcionalidad de __init__.

        Args:
            parent (Any): Argumento posicional parent.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Configuración de Graficador v2.0 (PySide6)")
        self.resize(800, 500)
        
        self.ui_cfg = config_mgr.get("estetica_global") or {}
        is_dark = self.ui_cfg.get("tema_oscuro", True)
        
        if is_dark:
            self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")
            self.box_style = "QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }"
            self.list_style = "background-color: #111; color: #fff; border: 1px solid #333;"
            self.entry_style = "background-color: #111; color: #fff; border: 1px solid #444;"
        else:
            self.setStyleSheet("background-color: #f5f5f5; color: #333; font-family: 'Arial', sans-serif;")
            self.box_style = "QGroupBox { border: 1px solid #0078D7; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #0078D7; font-weight: bold; }"
            self.list_style = "background-color: #fff; color: #000; border: 1px solid #ccc;"
            self.entry_style = "background-color: #fff; color: #000; border: 1px solid #ccc;"
            
        self.resultado = None
        self.seleccionadas = []
        self.pre_selected_paths = pre_selected_paths or []
        
        main_layout = QHBoxLayout(self)
        
        # --- PANEL IZQUIERDO: SELECCIÓN ---
        left_group = QGroupBox("1. Seleccionar Mediciones")
        left_group.setStyleSheet(self.box_style)
        left_layout = QVBoxLayout(left_group)
        
        lbl_info = QLabel("(Use Ctrl o Shift para selección múltiple)")
        lbl_info.setStyleSheet("color: #888; font-size: 10px;")
        left_layout.addWidget(lbl_info)
        
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listbox.setStyleSheet(self.list_style)
        
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        
        for fecha in sorted(os.listdir(BASE_DIR)):
            fecha_path = os.path.join(BASE_DIR, fecha)
            if os.path.isdir(fecha_path):
                for medicion in sorted(os.listdir(fecha_path)):
                    medicion_path = os.path.join(fecha_path, medicion)
                    if os.path.isdir(medicion_path):
                        from PySide6.QtWidgets import QListWidgetItem
                        item = QListWidgetItem(f"{fecha}/{medicion}")
                        self.listbox.addItem(item)
                        
                        norm_pre = [os.path.normpath(p).replace('\\', '/') for p in self.pre_selected_paths] if self.pre_selected_paths else []
                        if norm_pre and os.path.normpath(f"{fecha}/{medicion}").replace('\\', '/') in norm_pre:
                            item.setSelected(True)
                            self.listbox.setCurrentItem(item)
            
        left_layout.addWidget(self.listbox)
        main_layout.addWidget(left_group, stretch=1)
        
        if self.pre_selected_paths:
            left_group.hide()
        
        # --- PANEL DERECHO: CONFIGURACIÓN ---
        right_group = QGroupBox("2. Configuración de Procesamiento")
        right_group.setStyleSheet(self.box_style)
        right_layout = QVBoxLayout(right_group)
        
        # Filtros
        filtros_group = QGroupBox("Filtros Digitales")
        filtros_layout = QVBoxLayout(filtros_group)
        self.chk_notch = QCheckBox(f"Filtro Notch ({int(FREQ_NOTCH)} Hz)")
        self.chk_notch.setChecked(True)
        self.chk_bandpass = QCheckBox(f"Filtro Pasabanda ({FREQ_PASABANDA[0]}-{FREQ_PASABANDA[1]} Hz)")
        self.chk_bandpass.setChecked(True)
        filtros_layout.addWidget(self.chk_notch)
        filtros_layout.addWidget(self.chk_bandpass)
        right_layout.addWidget(filtros_group)
        
        # Envolvente
        env_group = QGroupBox("Procesamiento / Envolvente")
        env_layout = QVBoxLayout(env_group)
        self.rb_ninguna = QRadioButton("Solo Señal Filtrada")
        self.rb_ninguna.setChecked(True)
        self.rb_hilbert = QRadioButton("Envolvente de Hilbert")
        self.rb_rms = QRadioButton(f"Envolvente RMS ({RMS_WINDOW_MS}ms)")
        
        self.env_btn_group = QButtonGroup()
        self.env_btn_group.addButton(self.rb_ninguna, id=1)
        self.env_btn_group.addButton(self.rb_hilbert, id=2)
        self.env_btn_group.addButton(self.rb_rms, id=3)
        
        env_layout.addWidget(self.rb_ninguna)
        env_layout.addWidget(self.rb_hilbert)
        env_layout.addWidget(self.rb_rms)
        right_layout.addWidget(env_group)
        
        # Tiempo
        time_group = QGroupBox("Intervalo de Tiempo (s)")
        time_layout = QHBoxLayout(time_group)
        time_layout.addWidget(QLabel("Inicio:"))
        self.entry_inicio = QLineEdit()
        self.entry_inicio.setStyleSheet(self.entry_style)
        time_layout.addWidget(self.entry_inicio)
        time_layout.addWidget(QLabel("Fin:"))
        self.entry_fin = QLineEdit()
        self.entry_fin.setStyleSheet(self.entry_style)
        time_layout.addWidget(self.entry_fin)
        right_layout.addWidget(time_group)
        
        lbl_time_hint = QLabel("Dejar en blanco para graficar todo.")
        lbl_time_hint.setStyleSheet("color: #888; font-size: 10px;")
        right_layout.addWidget(lbl_time_hint)

        # Eje Y
        y_group = QGroupBox("Límites de Amplitud (µV)")
        y_layout = QHBoxLayout(y_group)
        y_layout.addWidget(QLabel("Min:"))
        self.entry_ymin = QLineEdit()
        self.entry_ymin.setStyleSheet(self.entry_style)
        y_layout.addWidget(self.entry_ymin)
        y_layout.addWidget(QLabel("Max:"))
        self.entry_ymax = QLineEdit()
        self.entry_ymax.setStyleSheet(self.entry_style)
        y_layout.addWidget(self.entry_ymax)
        right_layout.addWidget(y_group)
        
        lbl_y_hint = QLabel("Dejar en blanco para autoescala.")
        lbl_y_hint.setStyleSheet("color: #888; font-size: 10px;")
        right_layout.addWidget(lbl_y_hint)
        
        # Opciones extra
        extra_group = QGroupBox("Visualización")
        extra_layout = QVBoxLayout(extra_group)
        self.chk_fft = QCheckBox("Añadir Espectro de Frecuencias (FFT)")
        extra_layout.addWidget(self.chk_fft)
        
        self.chk_dark_mode = QCheckBox("Tema Oscuro (Fondo Negro)")
        self.chk_dark_mode.setChecked(True)
        extra_layout.addWidget(self.chk_dark_mode)
        right_layout.addWidget(extra_group)
        
        right_layout.addStretch()
        
        self.btn_run = QPushButton("Empezar Secuencia")
        self.btn_run.setStyleSheet("QPushButton { background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px; border-radius: 3px; }")
        self.btn_run.clicked.connect(self.confirmar)
        right_layout.addWidget(self.btn_run)
        
        main_layout.addWidget(right_group, stretch=1)
        
        # Cargar config guardada
        self.cargar_config()

    def cargar_config(self):
        saved = config_mgr.get("plotter_config") or {}
        self.chk_notch.setChecked(saved.get("notch", True))
        self.chk_bandpass.setChecked(saved.get("bandpass", True))
        
        tipo_env = saved.get("tipo_env", "ninguna")
        if tipo_env == "hilbert": self.rb_hilbert.setChecked(True)
        elif tipo_env == "rms": self.rb_rms.setChecked(True)
        else: self.rb_ninguna.setChecked(True)
        
        self.chk_fft.setChecked(saved.get("graficar_fft", False))
        self.chk_dark_mode.setChecked(saved.get("tema_oscuro", True))
        
        if "start_time" in saved and saved["start_time"] is not None:
            self.entry_inicio.setText(str(saved["start_time"]))
        if "end_time" in saved and saved["end_time"] is not None:
            self.entry_fin.setText(str(saved["end_time"]))
            
        if "y_min" in saved and saved["y_min"] is not None:
            self.entry_ymin.setText(str(saved["y_min"]))
        if "y_max" in saved and saved["y_max"] is not None:
            self.entry_ymax.setText(str(saved["y_max"]))

    def confirmar(self):
        """
        Ejecuta la funcionalidad de confirmar.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        if self.pre_selected_paths:
            self.seleccionadas = [os.path.normpath(p).replace('\\', '/') for p in self.pre_selected_paths]
        else:
            selected_items = self.listbox.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "Advertencia", "Debe seleccionar al menos una medición.")
                return
                
            self.seleccionadas = [item.text() for item in selected_items]
        
        start, end = None, None
        try:
            if self.entry_inicio.text().strip(): start = float(self.entry_inicio.text())
            if self.entry_fin.text().strip(): end = float(self.entry_fin.text())
        except ValueError:
            pass

        ymin, ymax = None, None
        try:
            if self.entry_ymin.text().strip(): ymin = float(self.entry_ymin.text())
            if self.entry_ymax.text().strip(): ymax = float(self.entry_ymax.text())
        except ValueError:
            pass
            
        tipo_env = "ninguna"
        if self.rb_hilbert.isChecked(): tipo_env = "hilbert"
        elif self.rb_rms.isChecked(): tipo_env = "rms"
        
        self.resultado = {
            "notch": self.chk_notch.isChecked(),
            "bandpass": self.chk_bandpass.isChecked(),
            "tipo_env": tipo_env,
            "start_time": start,
            "end_time": end,
            "y_min": ymin,
            "y_max": ymax,
            "graficar_fft": self.chk_fft.isChecked(),
            "tema_oscuro": self.chk_dark_mode.isChecked()
        }
        
        # Guardar la config para la proxima vez
        config_mgr.set("plotter_config", "notch", self.chk_notch.isChecked())
        config_mgr.set("plotter_config", "bandpass", self.chk_bandpass.isChecked())
        config_mgr.set("plotter_config", "tipo_env", tipo_env)
        config_mgr.set("plotter_config", "start_time", start)
        config_mgr.set("plotter_config", "end_time", end)
        config_mgr.set("plotter_config", "y_min", ymin)
        config_mgr.set("plotter_config", "y_max", ymax)
        config_mgr.set("plotter_config", "graficar_fft", self.chk_fft.isChecked())
        config_mgr.set("plotter_config", "tema_oscuro", self.chk_dark_mode.isChecked())
        
        self.accept()

# --- 3. FUNCIONES DE PROCESAMIENTO ---

def calcular_rms(senal, fs, window_ms):
    """
    Ejecuta la funcionalidad de calcular_rms.

    Args:
        senal (Any): Argumento posicional senal.
        fs (Any): Argumento posicional fs.
        window_ms (Any): Argumento posicional window_ms.

    Returns:
        Any: Resultado de la ejecución de la función.
    """
    window_samples = int(fs * (window_ms / 1000.0))
    if window_samples < 1: window_samples = 1
    s = pd.Series(senal)
    rms = s.pow(2).rolling(window=window_samples, center=True).mean().apply(np.sqrt)
    return rms.fillna(0).values

def plotear_medicion_secuencial(nombre_medicion, config, limits_cache=None, mostrar_plot=True):
    """
    Procesa, GUARDA en la carpeta origen y muestra la gráfica.
    """
    if limits_cache is None:
        limits_cache = {}

    print(f"\n>>> Procesando: {nombre_medicion}...")
    
    aplicar_notch = config["notch"]
    aplicar_pasabanda = config["bandpass"]
    tipo_envolvente = config["tipo_env"]
    start_time = config["start_time"]
    end_time = config["end_time"]
    y_min = config.get("y_min")
    y_max = config.get("y_max")

    # 1. Cargar CSV
    path_medicion = os.path.join(BASE_DIR, nombre_medicion)
    archivo_csv = next((os.path.join(path_medicion, f) for f in os.listdir(path_medicion) if f.lower().endswith('.csv')), None)
    
    if not archivo_csv:
        print(f"[WARN] Saltando {nombre_medicion}: No hay CSV.")
        return

    try:
        df = pd.read_csv(archivo_csv)
    except Exception as e:
        print(f"[ERROR] Error leyendo CSV: {e}")
        return

    # 2. Filtrar Tiempo
    col_tiempo = df.columns[0]
    rango_str = "Completa"
    if start_time is not None and end_time is not None:
        if start_time < end_time:
            df = df[(df[col_tiempo] >= start_time) & (df[col_tiempo] <= end_time)]
            rango_str = f"{start_time}s - {end_time}s"

    if df.empty: return

    # Filtrar solo canales con la palabra "Canal" (ignorar Tiempos)
    cols_canales = [col for col in df.columns if "Canal" in col or "Dev" in col]
    if not cols_canales: return

    # Fs
    try: fs = 1 / (df[col_tiempo].iloc[1] - df[col_tiempo].iloc[0])
    except: fs = 2000.0

    # Metadata
    bpm, noise_seconds = None, None
    try:
        for item in os.listdir(path_medicion):
            if os.path.isdir(os.path.join(path_medicion, item)) and item.startswith("canal_"):
                meta_path = os.path.join(path_medicion, item, 'metadata.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        md = json.load(f)
                        bpm, noise_seconds = md.get('bpm'), md.get('noise_seconds')
                    break
    except: pass

    # 3. Graficar
    graficar_fft = config.get("graficar_fft", False)
    num_canales = len(cols_canales)
    ncols = 2 if graficar_fft else 1
    ancho_fig = 24 if graficar_fft else 16
    
    # Estética global / local
    is_dark = config.get("tema_oscuro", True)
    if is_dark:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    fig, axs = plt.subplots(num_canales, ncols, figsize=(ancho_fig, 5 * num_canales), squeeze=False)
    
    # Compartir ejes X por columna para mantener sincronización y ocultar labels internos
    if num_canales > 1:
        for i in range(1, num_canales):
            axs[i, 0].sharex(axs[0, 0])
            if graficar_fft:
                axs[i, 1].sharex(axs[0, 1])
        for i in range(num_canales - 1):
            plt.setp(axs[i, 0].get_xticklabels(), visible=False)
            if graficar_fft:
                plt.setp(axs[i, 1].get_xticklabels(), visible=False)
    
    canales_config = config_mgr.get("canales") or {}

    # --- NUEVO: Extraer metadatos y resolver colores únicos sin repeticiones ---
    ch_info_list = []
    for i, nombre_canal in enumerate(cols_canales):
        nom_limpio = nombre_canal.strip()
        ch_conf = canales_config.get(nom_limpio, {})
        musculo = ch_conf.get("musculo", nom_limpio)
        ganancia = ch_conf.get("factor_calibracion", 495.0)
        
        try:
            ch_idx = int(nom_limpio.split()[-1])
            meta_path = os.path.join(path_medicion, f"canal_{ch_idx}", "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f_meta:
                    md_ch = json.load(f_meta)
                    if 'musculo' in md_ch and md_ch['musculo']:
                        musculo = md_ch['musculo']
                    elif 'muscles_map' in md_ch and f"canal_{ch_idx}" in md_ch['muscles_map']:
                        musculo = md_ch['muscles_map'][f"canal_{ch_idx}"]
                    if 'resistencia_ohm' in md_ch:
                        res_ohm = float(md_ch['resistencia_ohm'])
                        ganancia = 1.0 + (49400.0 / res_ohm)
        except Exception:
            pass

        try:
            ch_idx_int = int(nom_limpio.split()[-1])
        except Exception:
            ch_idx_int = i

        is_mic = (ch_idx_int == 3 or "mic" in musculo.lower())
        ch_info_list.append({
            "idx": ch_idx_int,
            "col_name": nombre_canal,
            "musculo": musculo,
            "ganancia": ganancia,
            "color_hex": ch_conf.get("color_hex"),
            "is_mic": is_mic
        })

    try:
        from utils.config_manager import get_unique_channel_colors
        colores_canales = get_unique_channel_colors(ch_info_list)
    except Exception:
        colores_canales = ["#ffaa00", "#39ff14", "#ffff00", "#ff0000"]

    # --- PASO 1: Procesar todas las señales ---
    processed_channels = []
    for i, ch_meta in enumerate(ch_info_list):
        nombre_canal = ch_meta["col_name"]
        nom_limpio = nombre_canal.strip()
        raw = df[nombre_canal].values
        musculo = ch_meta["musculo"]
        ganancia = ch_meta["ganancia"]
        color_hex = colores_canales[i]
        is_mic = ch_meta["is_mic"]
            
        sig = (raw / ganancia) * 1e6 

        # Restar offset DC antes de filtrar (solo para modos con envolvente)
        if tipo_envolvente in ['hilbert', 'rms']:
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        dc_offset = np.median(sig[:noise_end_idx])
                        sig = sig - dc_offset

        info_filtros = []
        if aplicar_notch:
            b, a = signal.iirnotch(FREQ_NOTCH, Q_FACTOR_NOTCH, fs)
            sig = signal.filtfilt(b, a, sig)
            info_filtros.append("Notch")
        
        if aplicar_pasabanda:
            nyq = 0.5 * fs
            low, high = FREQ_PASABANDA[0]/nyq, min(FREQ_PASABANDA[1]/nyq, 0.99)
            b, a = signal.butter(ORDEN_PASABANDA, [low, high], btype='band')
            sig = signal.filtfilt(b, a, sig)
            info_filtros.append("BP")

        etiqueta_env = ""
        if tipo_envolvente == 'hilbert':
            env = np.abs(signal.hilbert(sig))
            etiqueta_env = " | Env. Hilbert"
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.median(env[:noise_end_idx])
                        env = env - noise_level
                        etiqueta_env += " (Offset restado)"
            y_plot = env
            lw = 1.2
        elif tipo_envolvente == 'rms':
            env_rms = calcular_rms(sig, fs, RMS_WINDOW_MS)
            etiqueta_env = " | Env. RMS"
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.nanmedian(env_rms[:noise_end_idx])
                        if not np.isnan(noise_level):
                            env_rms = env_rms - noise_level
                            etiqueta_env += " (Offset restado)"
            y_plot = env_rms
            lw = 1.5
        else:
            y_plot = sig
            lw = 0.8

        min_val = float(np.nanmin(y_plot)) if len(y_plot) > 0 else 0.0
        max_val = float(np.nanmax(y_plot)) if len(y_plot) > 0 else 1.0

        processed_channels.append({
            "idx": i,
            "nombre_canal": nombre_canal,
            "musculo": musculo,
            "color_hex": color_hex,
            "sig_fft": sig,
            "y_plot": y_plot,
            "lw": lw,
            "min_val": min_val,
            "max_val": max_val,
            "is_mic": is_mic,
            "info_filtros": info_filtros,
            "etiqueta_env": etiqueta_env
        })

    # --- PASO 2: Calcular escala Y compartida para los 3 primeros canales musculares ---
    muscle_channels = [ch for ch in processed_channels if not ch["is_mic"]]
    shared_muscle_ylim = None
    if muscle_channels:
        g_min = min(ch["min_val"] for ch in muscle_channels)
        g_max = max(ch["max_val"] for ch in muscle_channels)
        if g_max > g_min:
            m_margin = (g_max - g_min) * 0.08
            shared_muscle_ylim = (g_min - m_margin, g_max + m_margin)
        else:
            shared_muscle_ylim = (g_min - 5.0, g_max + 5.0)

    # --- PASO 3: Graficar cada canal en su subplot ---
    t_min = df[col_tiempo].iloc[0]
    t_max = df[col_tiempo].iloc[-1]

    for i, ch in enumerate(processed_channels):
        ax = axs[i, 0]
        y_plot = ch["y_plot"]
        color_hex = ch["color_hex"]
        musculo = ch["musculo"]
        is_mic = ch["is_mic"]
        
        ax.plot(df[col_tiempo], y_plot, color=color_hex, lw=ch["lw"])
        
        # Asignar escala Y
        if y_min is not None or y_max is not None:
            cur_y0, cur_y1 = ax.get_ylim()
            ax.set_ylim(
                y_min if y_min is not None else cur_y0,
                y_max if y_max is not None else cur_y1
            )
        elif not is_mic and shared_muscle_ylim is not None:
            # Los canales musculares comparten exactamente la misma escala para comparar amplitudes
            ax.set_ylim(shared_muscle_ylim)
        else:
            # Micrófono / canal 3 con auto-escala independiente
            min_v = ch["min_val"]
            max_v = ch["max_val"]
            if max_v > min_v:
                m_margin = (max_v - min_v) * 0.08
                ax.set_ylim(min_v - m_margin, max_v + m_margin)

        # Señalar ruido basal al comienzo
        if noise_seconds is not None and noise_seconds > 0 and noise_seconds >= t_min:
            span_color = '#00e5ff' if is_dark else '#0074D9'
            ax.axvspan(max(0.0, t_min), min(noise_seconds, t_max), color=span_color, alpha=0.12)
            ax.axvline(x=noise_seconds, color=span_color, ls='--', lw=1.5, alpha=0.75)
            
            # Etiqueta textual 'Ruido Basal'
            y_bounds = ax.get_ylim()
            y_text = y_bounds[1] - 0.08 * (y_bounds[1] - y_bounds[0])
            ax.text(noise_seconds / 2.0, y_text, "Ruido Basal", color=span_color,
                    fontsize=13, ha='center', va='top', fontweight='bold', alpha=0.9)

        # Líneas de cada ventana alrededor del metrónomo (+- tau / 2)
        if bpm and noise_seconds is not None:
            tau = 60.0 / bpm
            win_color = '#ffffff' if is_dark else '#333333'
            beat_color = '#ffaa00' if is_dark else '#d35400'
            
            # Primera frontera antes del primer pulso si está dentro del rango
            first_bound = noise_seconds - tau / 2.0
            if t_min <= first_bound <= t_max:
                ax.axvline(x=first_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
                
            k = 0
            while True:
                t_beat = noise_seconds + k * tau
                t_bound = t_beat + tau / 2.0
                
                # Línea central del metrónomo / beat
                if t_min <= t_beat <= t_max:
                    ax.axvline(x=t_beat, color=beat_color, ls=':', lw=0.9, alpha=0.35)
                    
                if t_bound > t_max:
                    break
                if t_bound >= t_min:
                    ax.axvline(x=t_bound, color=win_color, ls='--', lw=1.0, alpha=0.4)
                k += 1
                    
        # --- Detección y marcado sutil de picos + Línea de media de amplitud de picos ---
        if not is_mic:
            picos_t = []
            picos_y = []
            t_arr = df[col_tiempo].values
            
            if bpm and noise_seconds is not None:
                tau = 60.0 / bpm
                k_p = 0
                while True:
                    t_beat_k = noise_seconds + k_p * tau
                    t_w_start = t_beat_k - tau / 2.0
                    t_w_end = t_beat_k + tau / 2.0
                    
                    if t_w_start > t_max:
                        break
                    
                    # Extraer el pico máximo dentro de cada ventana periódica
                    mask_win = (t_arr >= max(t_min, t_w_start)) & (t_arr < min(t_max, t_w_end))
                    if np.any(mask_win):
                        sub_t = t_arr[mask_win]
                        sub_y = y_plot[mask_win]
                        if len(sub_y) > 0:
                            idx_max = np.argmax(sub_y)
                            p_val = sub_y[idx_max]
                            p_t = sub_t[idx_max]
                            if p_val > 0:
                                picos_t.append(p_t)
                                picos_y.append(p_val)
                    k_p += 1
            else:
                try:
                    min_dist = max(1, int(fs * 0.3))
                    h_thresh = max(0.0, float(np.mean(y_plot)))
                    p_indices, _ = signal.find_peaks(y_plot, distance=min_dist, height=h_thresh)
                    if len(p_indices) > 0:
                        picos_t = t_arr[p_indices].tolist()
                        picos_y = y_plot[p_indices].tolist()
                except Exception:
                    pass

            # Dibujar marcadores sutiles de picos y línea de media
            if len(picos_y) > 0:
                media_picos = float(np.mean(picos_y))
                
                # Puntos discretos en la cúspide de cada ciclo
                ax.scatter(picos_t, picos_y, color=color_hex, s=26, alpha=0.75, zorder=5,
                           edgecolors='white' if is_dark else '#222222', linewidths=0.6)
                
                # Línea horizontal con la media de amplitud de los picos
                ax.axhline(y=media_picos, color=color_hex, ls=':', lw=1.3, alpha=0.65, zorder=4)
                
                # Etiqueta con el valor numérico medio
                ax.text(t_max, media_picos, f"  μ_picos = {media_picos:.1f} µV",
                        color=color_hex, fontsize=13, va='center', ha='left',
                        fontweight='bold', alpha=0.9, zorder=6)

        tit = musculo
        if ch["info_filtros"]: tit += f" | {', '.join(ch['info_filtros'])}"
        tit += ch["etiqueta_env"]
        ax.set_title(tit, fontsize=25)
        ax.set_ylabel("Amplitud (µV)" if not is_mic else "Micrófono", fontsize=27)
        ax.grid(True, alpha=0.5, ls='--')
        ax.tick_params(axis='both', which='major', labelsize=20)

        # --- Espectro de frecuencias (FFT) ---
        if graficar_fft:
            ax_fft = axs[i, 1]
            sig_fft = ch["sig_fft"]
            N = len(sig_fft)
            freqs = np.fft.rfftfreq(N, d=1.0/fs)
            fft_mag = np.abs(np.fft.rfft(sig_fft))
            
            ax_fft.plot(freqs, fft_mag, color=color_hex, lw=1.5)
            ax_fft.set_title(f"Espectro - {musculo}", fontsize=25)
            ax_fft.set_ylabel("Magnitud FFT", fontsize=27)
            ax_fft.grid(True, alpha=0.5, ls='--')
            ax_fft.tick_params(axis='both', which='major', labelsize=20)
            
            limite_frecuencia = min(500, fs/2)
            ax_fft.set_xlim(0, limite_frecuencia)

    axs[-1, 0].set_xlabel("Tiempo (s)", fontsize=27)
    if graficar_fft:
        axs[-1, 1].set_xlabel("Frecuencia (Hz)", fontsize=27)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Aplicar límites guardados si existen
    if "xlim" in limits_cache:
        axs[0, 0].set_xlim(limits_cache["xlim"])
    for i in range(num_canales):
        if f"ylim_{i}" in limits_cache:
            axs[i, 0].set_ylim(limits_cache[f"ylim_{i}"])

    # --- GUARDADO AUTOMÁTICO (SOLICITUD DE USUARIO) ---
    # Formato: plot_calibrado_{nombre_medicion_limpio}.png
    # Ubicación: Dentro de la misma carpeta de la medición
    
    # Asegurar que sacamos un nombre relativo limpio incluso si pasaron una ruta absoluta
    try:
        rel_path = os.path.relpath(path_medicion, BASE_DIR)
    except ValueError:
        # Fallback si por alguna razón no está en BASE_DIR
        padre = os.path.basename(os.path.dirname(path_medicion))
        hijo = os.path.basename(path_medicion)
        rel_path = f"{padre}_{hijo}"
        
    nombre_limpio = rel_path.replace("/", "_").replace("\\", "_")
    nombre_archivo = f"plot_calibrado_{nombre_limpio}.png"
    ruta_guardado = os.path.join(path_medicion, nombre_archivo)
    
    plt.savefig(ruta_guardado, dpi=100)
    print(f"[OK] Guardado en medición: {ruta_guardado}")
    
    # NUEVO: Guardar copia en el historial de comparativas
    carpeta_comparativas = os.path.join(root_dir, "analisis_comparativos")
    if not os.path.exists(carpeta_comparativas):
        os.makedirs(carpeta_comparativas)
        
    ruta_comparativa = os.path.join(carpeta_comparativas, nombre_archivo)
    plt.savefig(ruta_comparativa, dpi=100)
    print(f"[OK] Copia guardada en historial: {ruta_comparativa}")
    
    # --- VISUALIZACIÓN BLOQUEANTE ---
    if mostrar_plot:
        # Interceptar el evento de cierre para asegurar que leemos los límites correctos antes de que Matplotlib destruya los ejes
        def on_close(event):
            try:
                limits_cache["xlim"] = axs[0, 0].get_xlim()
                for i in range(num_canales):
                    limits_cache[f"ylim_{i}"] = axs[i, 0].get_ylim()
            except:
                pass
                
        fig.canvas.mpl_connect('close_event', on_close)

        print(f"Visualizando {nombre_medicion}. Cierra la ventana del gráfico para continuar...")
        plt.show(block=True)      
        plt.close('all') 
        print(f"Pasando a la siguiente...\n")
    else:
        plt.close(fig)

def flujo_principal():
    """
    Ejecuta la funcionalidad de flujo_principal.

    Returns:
        Any: Resultado de la ejecución de la función.
    """
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    if len(sys.argv) > 1:
        pre_selected = [os.path.relpath(p, BASE_DIR).replace('\\', '/') for p in sys.argv[1:]]
        dialog = PlotterConfigDialog(pre_selected_paths=pre_selected)
    else:
        dialog = PlotterConfigDialog()
    if dialog.exec() == QDialog.Accepted:
        mediciones = dialog.seleccionadas
        config = dialog.resultado
        
        if not mediciones or not config:
            return

        total = len(mediciones)
        print(f"--- Iniciando secuencia de {total} mediciones ---")
        
        from PySide6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Procesando mediciones (Cargando CSV)...", "Cancelar", 0, total)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        app.processEvents()
        
        limits_cache = {}
        for i, nombre_medicion in enumerate(mediciones):
            if progress.wasCanceled():
                print("Secuencia cancelada por el usuario.")
                break
                
            progress.setValue(i)
            progress.setLabelText(f"Procesando {i+1} de {total}:\n{nombre_medicion}")
            app.processEvents()
            
            print(f"[{i+1}/{total}] Cargando datos...")
            plotear_medicion_secuencial(nombre_medicion, config, limits_cache, mostrar_plot=False)
            
        progress.setValue(total)

        print("--- Todas las mediciones procesadas ---")

main = flujo_principal

if __name__ == "__main__":
    main()