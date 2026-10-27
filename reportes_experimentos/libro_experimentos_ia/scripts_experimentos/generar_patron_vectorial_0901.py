"""
Generación directa, limpia y vectorial del Patrón de Activación Muscular sEMG
Lectura directa desde base_de_datos_electrodos/2026-09-01
Normalización estricta por Supremo Tricanal por Pulso Individual
Filtros: Notch 50 Hz, Pasaaltos 20 Hz, Pasabajos 500 Hz, Envolvente RMS
Alineación centrada en el pulso fonatorio.
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paleta canónica universal
COLORES_VOCALES = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

CANALES = ['canal_0', 'canal_1', 'canal_2']
CANAL_LABELS = {
    'canal_0': 'CH0: Anterior Belly (Digástrico)',
    'canal_1': 'CH1: Orbicularis Oris',
    'canal_2': 'CH2: Modiolo / Risorio'
}

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def notch_filter(data, f0=50.0, Q=30.0, fs=2000.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def calcular_envolvente_rms(signal, window_size=100):
    squared = np.power(signal, 2)
    window = np.ones(window_size) / window_size
    rms = np.sqrt(np.convolve(squared, window, mode='same'))
    return rms

def procesar_patrones_0901(base_dir, out_png_path):
    fs = 2000.0
    window_pts = int(1.4 * fs) # 1400 ms (-600 ms a +800 ms) = 2800 puntos
    t_axis = np.linspace(-600, 800, window_pts)

    datos_vocales = {v: {c: [] for c in CANALES} for v in ['A', 'E', 'I', 'O', 'U']}

    mediciones_dir = os.path.join(base_dir, "2026-09-01")
    for toma in sorted(os.listdir(mediciones_dir)):
        vocal = toma[0].upper()
        if vocal not in datos_vocales:
            continue
        
        toma_path = os.path.join(mediciones_dir, toma)
        csv_file = os.path.join(toma_path, "grabacion.csv")
        
        # Si no hay grabacion.csv consolidado, cargar por carpetas de canales
        canales_raw = {}
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            for c in CANALES:
                if c in df.columns:
                    canales_raw[c] = df[c].values
                elif f"Canal {c[-1]}" in df.columns:
                    canales_raw[c] = df[f"Canal {c[-1]}"].values
            mic = df['canal_3'].values if 'canal_3' in df.columns else (df['Canal 3'].values if 'Canal 3' in df.columns else None)
        else:
            # Leer carpetas canal_0, canal_1, canal_2, canal_3
            for c in CANALES + ['canal_3']:
                c_dir = os.path.join(toma_path, c)
                c_csv = os.path.join(c_dir, "senial.csv") if os.path.exists(os.path.join(c_dir, "senial.csv")) else os.path.join(c_dir, "emg.csv")
                if os.path.exists(c_csv):
                    df_c = pd.read_csv(c_csv)
                    col = [col for col in df_c.columns if 'val' in col.lower() or 'emg' in col.lower() or 'raw' in col.lower()]
                    col_name = col[0] if col else df_c.columns[-1]
                    canales_raw[c] = df_c[col_name].values
            mic = canales_raw.get('canal_3', None)

        if len(canales_raw) < 3:
            continue

        # Filtrar cada canal con Notch 50 Hz y Pasabanda 20-500 Hz
        canales_filt = {}
        for c in CANALES:
            sig = canales_raw[c]
            sig_notch = notch_filter(sig, 50.0, 30.0, fs)
            sig_bp = butter_bandpass_filter(sig_notch, 20.0, 500.0, fs)
            canales_filt[c] = sig_bp

        # Normalización obligatoria por supremo tricanal
        supremo = max(np.max(np.abs(canales_filt[c])) for c in CANALES)
        if supremo <= 0:
            continue

        # Envolventes normalizadas
        envolventes = {}
        for c in CANALES:
            envolventes[c] = calcular_envolvente_rms(canales_filt[c] / supremo, window_size=int(0.09 * fs))

        # Encontrar el pico de actividad (o micrófono) para centrar en t=0
        if mic is not None:
            mic_env = calcular_envolvente_rms(np.abs(mic), window_size=int(0.05*fs))
            pico_idx = np.argmax(mic_env)
        else:
            suma_env = sum(envolventes[c] for c in CANALES)
            pico_idx = np.argmax(suma_env)

        n_pre = int(0.6 * fs)
        n_post = int(0.8 * fs)

        if pico_idx - n_pre < 0 or pico_idx + n_post > len(envolventes['canal_0']):
            continue

        for c in CANALES:
            segmento = envolventes[c][pico_idx - n_pre : pico_idx + n_post]
            datos_vocales[vocal][c].append(segmento)

    # Graficar en figura de alta resolución con estilo oscuro
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True, facecolor='#1e293b')
    plt.subplots_adjust(hspace=0.25, left=0.12, right=0.95, top=0.90, bottom=0.10)

    for i, c in enumerate(CANALES):
        ax = axes[i]
        ax.set_facecolor('#0f172a')
        ax.grid(True, linestyle='--', alpha=0.3, color='#94a3b8')

        for vocal in ['A', 'E', 'I', 'O', 'U']:
            series = datos_vocales[vocal][c]
            if not series:
                continue
            arr = np.array(series)
            media = np.mean(arr, axis=0)
            ax.plot(t_axis, media, color=COLORES_VOCALES[vocal], linewidth=2.0, label=f"/{vocal.lower()}/" if i == 0 else "")

        ax.axvline(0, color='#f8fafc', linestyle=':', alpha=0.6, linewidth=1.2)
        ax.set_ylabel("Norm. [0, 1]", color='#cbd5e1', fontsize=9)
        ax.set_title(CANAL_LABELS[c], color='#38bdf8', fontsize=10, fontweight='bold', loc='left', pad=4)
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    axes[-1].set_xlabel("Tiempo relativo al pico fonatorio [ms]", color='#cbd5e1', fontsize=10)
    axes[0].legend(loc='upper right', ncol=5, frameon=True, facecolor='#1e293b', edgecolor='#475569', labelcolor='#f8fafc', fontsize=9)

    fig.suptitle("Patrones Promedio de Activación sEMG (Filtro 20-500 Hz)", color='#f8fafc', fontsize=12, fontweight='bold', y=0.96)
    plt.savefig(out_png_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Patrones de activación generados con éxito: {out_png_path}")

if __name__ == "__main__":
    base = "EMG_desarrollo/base_de_datos_electrodos"
    out_p = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/patron_activacion_vectorial_0901.png"
    procesar_patrones_0901(base, out_p)
