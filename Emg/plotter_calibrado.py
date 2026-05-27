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

# --- Mantenemos Matplotlib para los gráficos, pero sin backend forzado a TkAgg ---
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

_current_dir = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_current_dir, "base_de_datos_electrodos")

FACTORES_G = {
    'Canal 0': 495,
    'Canal 1': 495,
    'Canal 2': 495,
}

NOMBRES_CANALES_MAP = {
    'Canal 0': 'Depresor Anguli Oris',
    'Canal 1': 'Orbicularis Oris',
    'Canal 2': 'Mylohyoid',
}

# Parámetros Fijos
FREQ_NOTCH = 50.0            
Q_FACTOR_NOTCH = 30.0        
FREQ_PASABANDA = [20, 1000]  
ORDEN_PASABANDA = 4          
RMS_WINDOW_MS = 75           

# --- 2. CLASES DE INTERFAZ (GUI) PySide6 ---

class PlotterConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de Graficador v2.0 (PySide6)")
        self.resize(800, 500)
        self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")
        
        self.resultado = None
        self.seleccionadas = []
        
        main_layout = QHBoxLayout(self)
        
        # --- PANEL IZQUIERDO: SELECCIÓN ---
        left_group = QGroupBox("1. Seleccionar Mediciones")
        left_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        left_layout = QVBoxLayout(left_group)
        
        lbl_info = QLabel("(Use Ctrl o Shift para selección múltiple)")
        lbl_info.setStyleSheet("color: #888; font-size: 10px;")
        left_layout.addWidget(lbl_info)
        
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listbox.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #333;")
        
        if not os.path.exists(BASE_DIR):
            os.makedirs(BASE_DIR)
        
        for fecha in sorted(os.listdir(BASE_DIR)):
            fecha_path = os.path.join(BASE_DIR, fecha)
            if os.path.isdir(fecha_path):
                for medicion in sorted(os.listdir(fecha_path)):
                    medicion_path = os.path.join(fecha_path, medicion)
                    if os.path.isdir(medicion_path):
                        self.listbox.addItem(f"{fecha}/{medicion}")
            
        left_layout.addWidget(self.listbox)
        main_layout.addWidget(left_group, stretch=1)
        
        # --- PANEL DERECHO: CONFIGURACIÓN ---
        right_group = QGroupBox("2. Configuración de Procesamiento")
        right_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
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
        self.entry_inicio.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444;")
        time_layout.addWidget(self.entry_inicio)
        time_layout.addWidget(QLabel("Fin:"))
        self.entry_fin = QLineEdit()
        self.entry_fin.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444;")
        time_layout.addWidget(self.entry_fin)
        right_layout.addWidget(time_group)
        
        lbl_time_hint = QLabel("Dejar en blanco para graficar todo.")
        lbl_time_hint.setStyleSheet("color: #888; font-size: 10px;")
        right_layout.addWidget(lbl_time_hint)
        
        # Opciones extra
        extra_group = QGroupBox("Visualización")
        extra_layout = QVBoxLayout(extra_group)
        self.chk_fft = QCheckBox("Añadir Espectro de Frecuencias (FFT)")
        extra_layout.addWidget(self.chk_fft)
        right_layout.addWidget(extra_group)
        
        right_layout.addStretch()
        
        self.btn_run = QPushButton("Empezar Secuencia")
        self.btn_run.setStyleSheet("QPushButton { background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px; border-radius: 3px; }")
        self.btn_run.clicked.connect(self.confirmar)
        right_layout.addWidget(self.btn_run)
        
        main_layout.addWidget(right_group, stretch=1)

    def confirmar(self):
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
            
        tipo_env = "ninguna"
        if self.rb_hilbert.isChecked(): tipo_env = "hilbert"
        elif self.rb_rms.isChecked(): tipo_env = "rms"
        
        self.resultado = {
            "notch": self.chk_notch.isChecked(),
            "bandpass": self.chk_bandpass.isChecked(),
            "tipo_env": tipo_env,
            "start_time": start,
            "end_time": end,
            "graficar_fft": self.chk_fft.isChecked()
        }
        self.accept()

# --- 3. FUNCIONES DE PROCESAMIENTO ---

def calcular_rms(senal, fs, window_ms):
    window_samples = int(fs * (window_ms / 1000.0))
    if window_samples < 1: window_samples = 1
    s = pd.Series(senal)
    rms = s.pow(2).rolling(window=window_samples, center=True).mean().apply(np.sqrt)
    return rms.fillna(0).values

def plotear_medicion_secuencial(nombre_medicion, config):
    """
    Procesa, GUARDA en la carpeta origen y muestra la gráfica.
    """
    print(f"\n>>> Procesando: {nombre_medicion}...")
    
    aplicar_notch = config["notch"]
    aplicar_pasabanda = config["bandpass"]
    tipo_envolvente = config["tipo_env"]
    start_time = config["start_time"]
    end_time = config["end_time"]

    # 1. Cargar CSV
    path_medicion = os.path.join(BASE_DIR, nombre_medicion)
    archivo_csv = next((os.path.join(path_medicion, f) for f in os.listdir(path_medicion) if f.lower().endswith('.csv')), None)
    
    if not archivo_csv:
        print(f"❌ Saltando {nombre_medicion}: No hay CSV.")
        return

    try:
        df = pd.read_csv(archivo_csv)
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return

    # 2. Filtrar Tiempo
    col_tiempo = df.columns[0]
    rango_str = "Completa"
    if start_time is not None and end_time is not None:
        if start_time < end_time:
            df = df[(df[col_tiempo] >= start_time) & (df[col_tiempo] <= end_time)]
            rango_str = f"{start_time}s - {end_time}s"

    if df.empty: return

    cols_canales = [col for col in df.columns[1:] if col.strip() in FACTORES_G]
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
    
    #fig.suptitle(f"Medición: {nombre_medicion}\n({rango_str})", fontsize=18, fontweight='bold')
    colores = plt.cm.viridis(np.linspace(0, 1, num_canales))

    for i, nombre_canal in enumerate(cols_canales):
        ax = axs[i, 0]
        nom_limpio = nombre_canal.strip()
        
        raw = df[nombre_canal].values
        
        # --- CORRECCIÓN: Leer ganancia desde metadata.json si existe ---
        ganancia = FACTORES_G.get(nom_limpio, 1.0)
        try:
            ch_idx = int(nom_limpio.split()[-1])
            meta_path = os.path.join(path_medicion, f"canal_{ch_idx}", "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f_meta:
                    md_ch = json.load(f_meta)
                    if 'resistencia_ohm' in md_ch:
                        res_ohm = float(md_ch['resistencia_ohm'])
                        ganancia = 1.0 + (49400.0 / res_ohm)
        except Exception:
            pass
            
        sig = (raw / ganancia) * 1e6 

        # --- NUEVO: Restar offset DC antes de filtrar (solo para modos con envolvente) ---
        dc_offset_removido = False
        if tipo_envolvente in ['hilbert', 'rms']:
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        dc_offset = np.mean(sig[:noise_end_idx])
                        sig = sig - dc_offset
                        dc_offset_removido = True

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

            # Restar ruido si está disponible
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.mean(env[:noise_end_idx])
                        env = np.maximum(0, env - noise_level)
                        #etiqueta_env += 
            
            ax.plot(df[col_tiempo], env, color=colores[i], lw=1.2)

        elif tipo_envolvente == 'rms':
            # Se usa un colormap distinto para la señal y su envolvente RMS para mayor claridad
            colores_rms = plt.cm.tab10(np.linspace(0, 1, 10))
            color_actual = colores_rms[i % 10]

            ax.plot(df[col_tiempo], sig, color=color_actual, alpha=0.4, lw=1, label='Señal Cruda')
            env_rms = calcular_rms(sig, fs, RMS_WINDOW_MS)
            etiqueta_env = " | Env. RMS"

            # Restar ruido si está disponible
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.nanmean(env_rms[:noise_end_idx])
                        if not np.isnan(noise_level):
                            env_rms = np.maximum(0, env_rms - noise_level)
                            etiqueta_env += " (ruido restado)"

            ax.plot(df[col_tiempo], env_rms, color=color_actual, lw=1.5, label='RMS')
            ax.legend(loc='upper right', fontsize=20)
            max_rms = np.nanmax(env_rms)
            if max_rms > 0: ax.set_ylim(-5, max_rms * 2)
        else:
            ax.plot(df[col_tiempo], sig, color=colores[i], lw=0.8)
        
        if bpm and noise_seconds is not None:
            tau = 60.0/bpm
            t_max = df[col_tiempo].iloc[-1]
            k = 0
            while True:
               #la ventana esta entre tau y menos tau sobre 2
                line_t = noise_seconds + k*tau + tau/2
                if line_t > t_max: break
                if line_t >= df[col_tiempo].iloc[0]:
                    ax.axvline(x=line_t, color='black', ls='--', lw=1, alpha=0.4)
                k += 1

        tit = NOMBRES_CANALES_MAP.get(nom_limpio, nom_limpio)
        if info_filtros: tit += f" | {', '.join(info_filtros)}"
        tit += etiqueta_env
        ax.set_title(tit, fontsize=25)
        ax.set_ylabel("Amplitud (µV)", fontsize=27)
        ax.grid(True, alpha=0.5, ls='--')
        ax.tick_params(axis='both', which='major', labelsize=20)

        # --- NUEVO: Espectro de frecuencias (FFT) ---
        if graficar_fft:
            ax_fft = axs[i, 1]
            N = len(sig)
            freqs = np.fft.rfftfreq(N, d=1.0/fs)
            fft_mag = np.abs(np.fft.rfft(sig))
            
            ax_fft.plot(freqs, fft_mag, color=colores[i], lw=1.5)
            ax_fft.set_title(f"Espectro - {NOMBRES_CANALES_MAP.get(nom_limpio, nom_limpio)}", fontsize=25)
            ax_fft.set_ylabel("Magnitud FFT", fontsize=27)
            ax_fft.grid(True, alpha=0.5, ls='--')
            ax_fft.tick_params(axis='both', which='major', labelsize=20)
            
            limite_frecuencia = min(500, fs/2)
            ax_fft.set_xlim(0, limite_frecuencia)

    axs[-1, 0].set_xlabel("Tiempo (s)", fontsize=27)
    if graficar_fft:
        axs[-1, 1].set_xlabel("Frecuencia (Hz)", fontsize=27)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # --- GUARDADO AUTOMÁTICO (SOLICITUD DE USUARIO) ---
    # Formato: plot_calibrado_{nombre_medicion}.png
    # Ubicación: Dentro de la misma carpeta de la medición
    nombre_archivo = f"plot_calibrado_{nombre_medicion}.png"
    ruta_guardado = os.path.join(path_medicion, nombre_archivo)
    
    plt.savefig(ruta_guardado, dpi=100)
    print(f"✅ Guardado en: {ruta_guardado}")
    
    # --- VISUALIZACIÓN BLOQUEANTE ---
    print(f"👁️ Visualizando {nombre_medicion}. Cierra la ventana del gráfico para continuar...")
    plt.show()      
    plt.close('all') 
    print(f"⏭️ Pasando a la siguiente...\n")

def flujo_principal():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    dialog = PlotterConfigDialog()
    if dialog.exec() == QDialog.Accepted:
        mediciones = dialog.seleccionadas
        config = dialog.resultado
        
        if not mediciones or not config:
            return

        total = len(mediciones)
        print(f"--- Iniciando secuencia de {total} mediciones ---")
        
        for i, nombre_medicion in enumerate(mediciones):
            print(f"[{i+1}/{total}] Cargando datos...")
            plotear_medicion_secuencial(nombre_medicion, config)

        print("--- Todas las mediciones procesadas ---")

if __name__ == "__main__":
    flujo_principal()