import os
import sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
from scipy.signal import find_peaks, butter, filtfilt, iirnotch

import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root


def apply_dsp_pipeline(sig, sr, noise_seconds, smooth_ms=250.0):
    # 1. Filtro Pasa-Altos (20 Hz)
    nyq = 0.5 * sr
    cutoff_hp = min(20.0, nyq * 0.99)
    b, a = butter(4, cutoff_hp / nyq, btype='high', analog=False)
    sig_filt = filtfilt(b, a, sig)
    
    # 2. Filtro Notch (50 Hz)
    f0 = 50.0
    if f0 < nyq:
        b, a = iirnotch(f0, 2.0, sr)
        sig_filt = filtfilt(b, a, sig_filt)
        
    # 3. Filtro Pasa-Bajos (500 Hz)
    cutoff_lp = min(500.0, nyq * 0.99)
    b, a = butter(4, cutoff_lp / nyq, btype='low', analog=False)
    sig_filt = filtfilt(b, a, sig_filt)
    
    # 4. Envolvente RMS
    window_size = int(smooth_ms / 1000.0 * sr)
    if window_size == 0:
        window_size = 1
    env = np.sqrt(np.convolve(sig_filt**2, np.ones(window_size)/window_size, mode='same'))
        
    # 5. Calcular ruido basal inicial
    initial_noise_mean = 0.0
    noise_samples = int(noise_seconds * sr)
    if noise_samples > 0 and noise_samples < len(env):
        skip = min(int(0.1 * sr), noise_samples // 2)
        initial_noise_mean = np.mean(env[skip:noise_samples])
        
    return env, initial_noise_mean

def get_interpulse_noise(env_segment, initial_noise):
    if len(env_segment) < 3:
        return initial_noise
        
    abs_noise = np.abs(env_segment)
    q1 = np.percentile(abs_noise, 25)
    q3 = np.percentile(abs_noise, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    valid_noise = abs_noise[abs_noise <= upper_bound]
    if len(valid_noise) < 3:
        valid_noise = abs_noise
        
    curr_mean = np.mean(valid_noise)
    
    # Filtro de outliers del 500% respecto al ruido basal inicial
    if initial_noise > 0 and (curr_mean / initial_noise) > 5.0:
        return initial_noise
    return curr_mean

def main():
    if len(sys.argv) < 2:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        med_path = filedialog.askdirectory(title="Selecciona la carpeta de la medición (ej: toma_1)")
        if not med_path:
            print("No se seleccionó ninguna medición.")
            sys.exit(0)
    else:
        med_path = sys.argv[1]
        
    med_rel_path = os.path.basename(med_path)
    
    bpm_a_usar = 30
    noise_seconds_a_usar = 2.0
    
    meta_path = os.path.join(med_path, 'canal_0', 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                meta_data = json.load(f)
                bpm_a_usar = meta_data.get('bpm', bpm_a_usar)
                noise_seconds_a_usar = meta_data.get('noise_seconds', noise_seconds_a_usar)
        except:
            pass

    periodo = 60.0 / bpm_a_usar

    # Cargar canal 3 (micrófono / maestro)
    ch3_dir = os.path.join(med_path, "canal_3")
    if not os.path.exists(ch3_dir):
        print("Falta canal_3 (micrófono)")
        sys.exit(1)
    wavs3 = [f for f in os.listdir(ch3_dir) if f.endswith(".wav")]
    if not wavs3:
        print("No hay wav en canal_3")
        sys.exit(1)
        
    sig3, sr = sf.read(os.path.join(ch3_dir, wavs3[0]))
    if sig3.ndim > 1: sig3 = sig3[:, 0]
    
    # Procesar canal 3 y encontrar picos (metrónomo) con una ENVOLVENTE GRANDE (250ms)
    env3, _ = apply_dsp_pipeline(sig3, sr, noise_seconds_a_usar, smooth_ms=250.0)
    
    dist_samples = int(0.8 * periodo * sr)
    # Buscar picos ignorando la parte de ruido basal
    start_search = int(noise_seconds_a_usar * sr)
    
    # Buscar picos sólo en la porción útil
    picos, _ = find_peaks(env3[start_search:], distance=dist_samples, height=np.max(env3[start_search:])*0.2)
    picos = picos + start_search # Ajustar índices a la señal original
    
    if len(picos) == 0:
        print("No se encontraron picos en canal_3")
        sys.exit(1)
        
    # Cargar y procesar los 3 canales
    sigs = []
    initial_noises = []
    canales = ['canal_0', 'canal_1', 'canal_2']
    smooth_ms_val = 250.0  # Variable para el suavizado y la etiqueta
    
    for ch in canales:
        ch_dir = os.path.join(med_path, ch)
        if not os.path.exists(ch_dir):
            sigs.append(np.zeros_like(sig3))
            initial_noises.append(0.0)
            continue
        wavs = [f for f in os.listdir(ch_dir) if f.endswith(".wav")]
        if not wavs:
            sigs.append(np.zeros_like(sig3))
            initial_noises.append(0.0)
            continue
        sig, _ = sf.read(os.path.join(ch_dir, wavs[0]))
        if sig.ndim > 1: sig = sig[:, 0]
        
        env_ch, ini_noise = apply_dsp_pipeline(sig, sr, noise_seconds_a_usar, smooth_ms=smooth_ms_val)
        sigs.append(env_ch)
        initial_noises.append(ini_noise)
        
    # --- FIGURA 1: Vista Continua ---
    fig1 = plt.figure(figsize=(18, 6))
    
    # Nuevos colores pedidos por el usuario
    colors = ['#39ff14', '#8a2be2', '#ffd700'] # Verde fluorescente, Violeta, Amarillo
    labels = ['Miloioide (Ch0)', 'Depresor (Ch1)', 'Orbicularis (Ch2)']
    
    plt.axvspan(0, noise_seconds_a_usar, color='cyan', alpha=0.3, label='Ventana Ruido Basal')
            
    periodo_samples = int(periodo * sr)
    noise_win_samples = max(3, int(periodo_samples / 4.0))
    
    # Listas para guardar los segmentos extraídos para la Figura 2
    extracted_segments = []
    extracted_mic = [] # Para guardar los segmentos del micrófono
    
    for i, pico in enumerate(picos):
        t_pico = pico / sr
        
        if i == 0:
            midpoint = pico - (periodo_samples // 2)
        else:
            midpoint = (picos[i-1] + pico) // 2
            
        noise_start = max(0, int(midpoint - noise_win_samples // 2))
        noise_end = min(len(sigs[0]), noise_start + noise_win_samples)
        
        interpulse_noise = [0.0, 0.0, 0.0]
        if noise_end > noise_start:
            for ch in range(3):
                noise_seg = sigs[ch][noise_start:noise_end]
                interpulse_noise[ch] = get_interpulse_noise(noise_seg, initial_noises[ch])
                
        # 2. Límites temporales de la ventana de -0.5 a 0.5 del periodo (medio periodo)
        t_start = t_pico - 0.5 * periodo
        t_end = t_pico + 0.5 * periodo
        
        idx_start = int(t_start * sr)
        idx_end = int(t_end * sr)
        
        idx_start_safe = max(0, idx_start)
        idx_end_safe = min(len(sigs[0]), idx_end)
        
        if idx_end_safe <= idx_start_safe:
            continue
            
        segs = []
        for ch in range(3):
            seg_ch = sigs[ch][idx_start_safe:idx_end_safe].copy()
            seg_ch = seg_ch - interpulse_noise[ch]
            seg_ch[seg_ch < 0] = 0.0
            segs.append(seg_ch)
            
        # Extraer segmento del micrófono (env3) para plotear de fondo
        mic_seg = env3[idx_start_safe:idx_end_safe].copy()
        mic_max = np.max(mic_seg)
        if mic_max > 0:
            mic_seg = mic_seg / mic_max
        extracted_mic.append(mic_seg)
        
        max_val = max(np.max(segs[0]), np.max(segs[1]), np.max(segs[2]))
        
        t_segment = np.linspace(idx_start_safe/sr, idx_end_safe/sr, idx_end_safe - idx_start_safe)
        
        # Guardar para el gráfico concatenado
        norm_segs = []
        for ch_idx in range(3):
            seg_norm = segs[ch_idx] / max_val if max_val > 0 else segs[ch_idx]
            norm_segs.append(seg_norm)
            
            label_muscle = labels[ch_idx] if i == 0 else ""
            plt.plot(t_segment, seg_norm, color=colors[ch_idx], label=label_muscle, linewidth=1.5, alpha=0.8)
            
        extracted_segments.append(norm_segs)
        
        label_centro = 'Centro (Pico Mic)' if i == 0 else ""
        label_propuesta = 'Ventana Propuesta (0.5/0.5)' if i == 0 else ""
        
        # Como el canal 3 es rojo, cambiemos la linea del centro a blanco/gris para que destaque
        plt.axvline(t_pico, color='white', linestyle='-', linewidth=2, alpha=0.7, label=label_centro)
        plt.axvline(t_start, color='#8e44ad', linestyle='--', linewidth=2, alpha=0.8, label=label_propuesta)
        plt.axvline(t_end, color='#8e44ad', linestyle='--', linewidth=2, alpha=0.8)

    plt.xlim(0, 10.0)
    plt.title(f"Vista Completa de Ventanas | BPM: {bpm_a_usar:.0f} | Toma: {med_rel_path}", fontsize=14, pad=15)
    plt.xlabel("Tiempo Absoluto (segundos)", fontsize=11, labelpad=10)
    plt.ylabel(f"Amplitud Envolvente Filtrada {smooth_ms_val}ms (Norm. Máx por Ventana)", fontsize=11, labelpad=10)
    
    # Poner un fondo oscuro para que resalten los colores fluorescentes
    plt.style.use('dark_background')
    ax = plt.gca()
    ax.set_facecolor('#1c1c1c')
    fig1.patch.set_facecolor('#1c1c1c')
    
    plt.grid(True, linestyle=':', alpha=0.3, color='#bdc3c7')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#7f8c8d')
    ax.spines['bottom'].set_color('#7f8c8d')
    ax.tick_params(colors='#ecf0f1')
    
    legend1 = plt.legend(loc='upper right', frameon=True, facecolor='#2c3e50', edgecolor='#34495e')
    for text in legend1.get_texts(): text.set_color("white")
    
    out_path1 = os.path.join(med_path, "pulses_3musculos_filt.png")
    fig1.savefig(out_path1, dpi=200, bbox_inches='tight', facecolor=fig1.get_facecolor())
    
    # --- FIGURA 2: Vista Concatenada Simétrica ---
    fig2 = plt.figure(figsize=(18, 6))
    
    current_time = 0.0
    for idx_pulse, norm_segs in enumerate(extracted_segments):
        duration = len(norm_segs[0]) / sr
        t_segment = np.linspace(current_time, current_time + duration, len(norm_segs[0]), endpoint=False)
        
        # Trazar primero el micrófono difuminado por detrás
        mic_seg = extracted_mic[idx_pulse]
        label_mic = 'Micrófono (Ch3)' if idx_pulse == 0 else ""
        plt.plot(t_segment, mic_seg, color='red', label=label_mic, linewidth=2.5, alpha=0.15)
        
        for ch_idx in range(3):
            label_muscle = labels[ch_idx] if idx_pulse == 0 else ""
            plt.plot(t_segment, norm_segs[ch_idx], color=colors[ch_idx], label=label_muscle, linewidth=1.5, alpha=0.85)
        
        # El centro está exactamente a la mitad de la ventana
        t_centro = current_time + (duration / 2.0)
        label_centro = 'Centro Exacto' if idx_pulse == 0 else ""
        plt.axvline(t_centro, color='white', linestyle='-', linewidth=1.5, alpha=0.5, label=label_centro)
        
        # Borde final de esta ventana (que empalma con la siguiente)
        plt.axvline(current_time + duration, color='#8e44ad', linestyle='--', linewidth=2, alpha=0.6)
        
        current_time += duration
        
    plt.xlim(0, 10.0)
    
    plt.title(f"Vista Concatenada de Ventanas Propuestas (±0.5) | Toma: {med_rel_path}", fontsize=14, pad=15, color='white')
    plt.xlabel("Tiempo  (segundos)", fontsize=11, labelpad=10, color='white')
    plt.ylabel(f"Amplitud Envolvente {smooth_ms_val}ms", fontsize=11, labelpad=10, color='white')
    
    # Poner fondo oscuro también acá
    ax2 = plt.gca()
    ax2.set_facecolor('#1c1c1c')
    fig2.patch.set_facecolor('#1c1c1c')
    
    plt.grid(True, linestyle=':', alpha=0.3, color='#bdc3c7')
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#7f8c8d')
    ax2.spines['bottom'].set_color('#7f8c8d')
    ax2.tick_params(colors='#ecf0f1')
    
    legend2 = plt.legend(loc='upper right', frameon=True, facecolor='#2c3e50', edgecolor='#34495e')
    for text in legend2.get_texts(): text.set_color("white")

    out_path2 = os.path.join(med_path, "pulses_3musculos_concat.png")
    fig2.savefig(out_path2, dpi=200, bbox_inches='tight', facecolor=fig2.get_facecolor())
    
    plt.show()
    
    plt.show()

if __name__ == "__main__":
    main()
