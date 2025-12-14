#%%
import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hilbert, butter, filtfilt, iirnotch, correlate, correlation_lags
from scipy import interpolate
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
import subprocess

# --- Imports para GUI ---
import tkinter as tk
from tkinter import filedialog, messagebox

# --- Versión del script de análisis ---
__version__ = "7.1 (Con Espectrograma del Promedio)"

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

# --- Función para la barra de progreso en consola ---
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

# ---------------------- Utilities -----------------------------------------
def rms(x):
    return np.sqrt(np.mean(x**2)) if len(x) > 0 else 0.0

def _resample_to(x, L):
    if len(x) == L:
        return x
    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, L)
    f = interpolate.interp1d(old, x, kind='linear', fill_value="extrapolate")
    return f(new)

# ---------------------- ALINEACIÓN ESTRATEGIA "LÍDER" -------------------
def _alinear_por_lider_calculo(segmentos_rs, samplerate):
    """
    MASTER: Calcula la alineación basándose en la SILUETA (Envolvente Suavizada).
    """
    if len(segmentos_rs) < 2:
        return segmentos_rs, np.zeros(len(segmentos_rs))

    # Pre-procesamiento suavizado para encontrar la forma general
    win_len = int(0.1 * samplerate) 
    if win_len < 1: win_len = 5
    window = np.ones(win_len) / win_len

    seg_smooth_list = []
    for s in segmentos_rs:
        s_rect = np.abs(s)
        s_smooth = np.convolve(s_rect, window, mode='same')
        seg_smooth_list.append(s_smooth)
    
    seg_smooth_arr = np.array(seg_smooth_list)

    # Encontrar al Líder
    energies = [np.sum(s**2) for s in seg_smooth_arr]
    best_idx = np.argmax(energies)
    ref_signal_smooth = seg_smooth_arr[best_idx]
    
    aligned_segments = []
    shifts_calculated = []
    
    for i, seg in enumerate(segmentos_rs):
        corr = correlate(seg_smooth_arr[i], ref_signal_smooth, mode='same')
        lags = correlation_lags(len(seg), len(ref_signal_smooth), mode='same')
        lag = lags[np.argmax(corr)]
        
        shift_val = -int(lag)
        shifts_calculated.append(shift_val)
        
        aligned_seg = np.roll(seg, shift_val)
        aligned_segments.append(aligned_seg)
        
    return np.array(aligned_segments), shifts_calculated

def _aplicar_alineacion_forzada(segmentos_rs, shifts):
    """
    SLAVE: Aplica shifts pre-calculados.
    """
    if len(segmentos_rs) != len(shifts):
        print(f"Advertencia CRÍTICA: Ventanas ({len(segmentos_rs)}) != Shifts ({len(shifts)}).")
        min_len = min(len(segmentos_rs), len(shifts))
        segmentos_rs = segmentos_rs[:min_len]
        shifts = shifts[:min_len]

    aligned_segments = []
    for i, seg in enumerate(segmentos_rs):
        shift = int(shifts[i])
        aligned_seg = np.roll(seg, shift)
        aligned_segments.append(aligned_seg)
    
    return np.array(aligned_segments)

# ---------------------- I/O & envelope ------------------------------------
def _read_wav_mono(filepath):
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal[:, 0]
    return np.asarray(signal, dtype=float), sr

def _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate):
    if apply_envelope:
        try:
            env_full = np.abs(hilbert(signal_abs))
        except Exception:
            env_full = signal_abs.copy()
    else:
        env_full = signal_abs.copy()

    if smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            window = np.ones(win_len, dtype=float) / float(win_len)
            env_full = np.convolve(env_full, window, mode='same')
    return env_full

# ---------------------- Estimación de Ruido -------------------
def _estimate_noise_window(signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral):
    start_sample_noise = int(round(noise_seconds * samplerate))
    if start_sample_noise <= 0:
        start_sample_noise = 0
    if start_sample_noise >= len(signal_recortada):
        start_sample_noise = min(len(signal_recortada)-1, int(round(0.01 * len(signal_recortada))))

    if start_sample_noise > 0:
        noise_segment = signal_recortada[:start_sample_noise]
        env_noise = np.abs(hilbert(np.abs(noise_segment))) if len(noise_segment) > 0 else np.array([])
        if smooth_ms is not None and smooth_ms > 0 and len(env_noise) > 1:
            win_len_n = int(max(1, round(smooth_ms * samplerate / 1000.0)))
            if win_len_n > 1:
                window_n = np.ones(win_len_n, dtype=float) / float(win_len_n)
                env_noise = np.convolve(env_noise, window_n, mode='same')

        if len(env_noise) >= 5:
            mad = np.median(np.abs(env_noise - np.median(env_noise)))
            sigma_est = mad * 1.4826
        else:
            sigma_est = np.std(env_noise) if len(env_noise) > 0 else 0.0

        umbral = np.mean(env_noise) if len(env_noise) > 0 else 0.0
        noise_rms_from_noise_window = rms(env_noise) if len(env_noise) > 0 else 0.0

        print(f"[Ruido Inicial] {noise_seconds}s, Umbral={umbral:.5e}")
        return start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window
    else:
        print(f"[Ruido] No se definió ventana de ruido.")
        return start_sample_noise, np.array([]), None, None, None

# ---------------------- CORTE: BEAT EN EL CENTRO ---------------------
def _cortar_centrado_en_beat(env_recortada,
                             start_sample_noise, 
                             muestras_pulso,     # Periodo completo
                             n_pulsos_manual=None,
                             excluded_windows=None,
                             forced_indices=None):
    if len(env_recortada) == 0: return [], [], []

    if n_pulsos_manual is None or n_pulsos_manual <= 0:
        print(f"--- ERROR: Se requiere conteo de pulsos. ---")
        return [], [], []

    centros_metronomo = []
    segmentos = []
    valid_indices = [] 
    
    excluded_set = set(excluded_windows) if excluded_windows else set()
    iterable_indices = forced_indices if forced_indices is not None else range(int(n_pulsos_manual))

    half_period = int(muestras_pulso / 2)

    for i in iterable_indices:
        if forced_indices is None:
            window_number = i + 1
            if window_number in excluded_set:
                print(f"    -> Omitiendo ventana #{window_number} (excluida).")
                continue

        beat_loc = start_sample_noise + i * muestras_pulso
        seg_start = beat_loc - half_period
        seg_end = beat_loc + half_period

        if seg_start < 0: 
            if seg_end < 0: continue
            seg_start = 0 
        
        if seg_end > len(env_recortada):
            if forced_indices is not None:
                pad_width = seg_end - len(env_recortada)
                if pad_width > 0:
                    temp_seg = env_recortada[seg_start:]
                    segmento = np.pad(temp_seg, (0, pad_width), 'constant')
                else:
                    segmento = env_recortada[seg_start:seg_end].copy()
            else:
                seg_end = len(env_recortada)
                segmento = env_recortada[seg_start:seg_end].copy()
                if segmento.size < half_period: continue
        else:
            segmento = env_recortada[seg_start:seg_end].copy()

        if forced_indices is None:
            if segmento.size < half_period: 
                continue

        centros_metronomo.append(int(beat_loc))
        segmentos.append(segmento)
        valid_indices.append(i)

    return centros_metronomo, segmentos, valid_indices

# ---------------------- Resample & pulse statistics -----------------------
def _resample_segments(segmentos, resample_len):
    lengths = [len(s) for s in segmentos]
    target_len = resample_len if resample_len is not None else int(np.median(lengths))
    segmentos_rs = np.vstack([_resample_to(s, target_len) for s in segmentos])
    return segmentos_rs, target_len

def _compute_pulse_stats(segmentos_rs):
    segmentos_norm = segmentos_rs.copy()
    pulso_promedio = np.mean(segmentos_norm, axis=0)
    pulso_sigma = np.std(segmentos_norm, axis=0, ddof=1)
    Np = segmentos_norm.shape[0]
    pulso_err = pulso_sigma / np.sqrt(Np)
    return segmentos_norm, pulso_promedio, pulso_sigma, pulso_err, Np

# ---------------------- Fallback umbral -----------------------------------
def _fallback_umbral(segmentos_norm, pulso_promedio, factor_umbral):
    residuos_baseline = (segmentos_norm - pulso_promedio).ravel()
    if residuos_baseline.size > 0:
        mad = np.median(np.abs(residuos_baseline - np.median(residuos_baseline)))
        sigma_est = mad * 1.4826
    else:
        sigma_est = 0.0
    umbral = float(max(0.0, factor_umbral * sigma_est))
    return sigma_est, umbral

# ---------------------- Plot pulso promedio --------------------------
def _plot_pulse_full(
    t_pulso, segmentos_norm, pulso_promedio, pulso_err, color_prom,
    snr_manual, snr_uncertainty, umbral, mostrar_umbral, filename, out_prom,
    plot_mode='mean', individual_alpha=0.25, mostrar_individuales=True, show_plot=False
):
    print_progress_bar(0, 1, prefix='Graficando Avg:', suffix='...', length=20)
    plt.figure(figsize=(12, 8))

    if mostrar_individuales and (segmentos_norm is not None) and len(segmentos_norm) > 0:
        for p in segmentos_norm:
            plt.plot(t_pulso, p, color='gray', alpha=individual_alpha, linewidth=1)

    plt.fill_between(t_pulso, pulso_promedio - pulso_err, pulso_promedio + pulso_err,
                     color=color_prom if not isinstance(color_prom, str) else None,
                     alpha=0.25, label="Error (1σ/√N)")

    plt.plot(t_pulso, pulso_promedio, color=color_prom, linewidth=2,
             label=rf"Promedio (SNR={snr_manual:.2f})")

    if mostrar_umbral and (umbral is not None):
        plt.axhline(umbral, color="green", linestyle="--", alpha=0.9, label=f"Umbral Ruido ({umbral:.2f})")
        plt.fill_between(t_pulso, -umbral, umbral, color="red", alpha=0.06)

    plt.title(f"PULSO PROMEDIO (Forma Muscular) - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    max_y_val = np.max(pulso_promedio) if len(pulso_promedio) > 0 else 0.7
    plt.ylim(0, max_y_val * 1.6)
    plt.grid(True, alpha=0.5)
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.8, label="Centro Ajustado")
    plt.legend(loc='upper right')
    plt.savefig(out_prom, dpi=300, bbox_inches='tight')
    
    print_progress_bar(1, 1, prefix='Graficando Avg:', suffix='OK', length=20)
    if show_plot:
        plt.show()
    else:
        plt.close(plt.gcf())

# ---------------------- Plot ESPECTROGRAMA (NUEVO) -----------------------
def _plot_espectrograma(pulso_promedio, samplerate, filename, out_spec):
    """
    Genera el espectrograma del pulso promedio procesado.
    """
    print_progress_bar(0, 1, prefix='Graficando Spec:', suffix='...', length=20)
    plt.figure(figsize=(10, 6))
    
    # NFFT dinámico según la longitud de la señal
    nfft = 256
    if len(pulso_promedio) < 256:
        nfft = len(pulso_promedio)
    
    # Espectrograma de matplotlib
    Pxx, freqs, bins, im = plt.specgram(pulso_promedio, NFFT=nfft, Fs=samplerate, noverlap=int(nfft/2), cmap='inferno')
    
    plt.title(f"Espectrograma - {filename}")
    plt.ylabel("Frecuencia [Hz]")
    plt.ylim(20,500)
    plt.xlabel("Tiempo [s]")
    cbar = plt.colorbar(im)
    cbar.set_label("Intensidad (dB)")
    
    # Ajustar ejes de tiempo para que coincidan con la duración (centrado en 0)
    duration = len(pulso_promedio) / samplerate
    half_dur = duration / 2
    # plt.specgram no deja configurar extent fácilmente para centrar en 0, 
    # pero el eje X mostrará de 0 a T. Lo dejamos así por simplicidad.
    
    plt.savefig(out_spec, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Graficando Spec:', suffix='OK', length=20)
    plt.close(plt.gcf())

# ---------------------- Plot recortes (GRID EN BORDES) --------------------------
def _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds,
                   start_sample_noise, samplerate, centros_metronomo, periodo, muestras_pulso, out_rec, filename, 
                   excluded_windows=None, show_plot=False, signal_original_unfiltered=None):
    
    plt.figure(figsize=(12, 4))
    if signal_original_unfiltered is not None:
        plt.plot(t_recortada, signal_original_unfiltered, color="red", linewidth=1.0, alpha=0.4, label="Original")
    plt.plot(t_recortada, signal_recortada, color="black", linewidth=1.2, alpha=0.7, label="Procesada")
    
    noise_t0 = t_recortada[0]
    noise_t1 = noise_t0 + noise_seconds
    plt.axvspan(noise_t0, noise_t1, color='violet', alpha=0.75, label=f"Ruido Inicial ({noise_seconds}s)")

    plt.plot(t_recortada, env_recortada, color="Blue", linewidth=1.5, linestyle='-', alpha=0.9, label="Envolvente")

    offset_start = t_recortada[0] + float(start_sample_noise)/samplerate
    duracion_analizable_grafico = len(env_recortada) - start_sample_noise
    n_pulsos = math.ceil(duracion_analizable_grafico / muestras_pulso)
    
    half_period_sec = periodo / 2.0
    
    for i in range(n_pulsos + 2):
        beat_t = offset_start + i * periodo
        cut_line_t = beat_t - half_period_sec
        if cut_line_t >= t_recortada[0] and cut_line_t <= t_recortada[-1]:
            plt.axvline(x=cut_line_t, color="Black", linestyle="--", alpha=0.6)

    excluded_set_plot = set(excluded_windows) if excluded_windows else set()

    for i in range(n_pulsos):
        beat_t = offset_start + i * periodo
        start_t = beat_t - half_period_sec
        end_t = beat_t + half_period_sec
        
        window_number = i + 1
        color = "red" if window_number in excluded_set_plot else "orange"
        alpha = 0.2 if window_number in excluded_set_plot else 0.05
        
        if end_t > t_recortada[0]:
             plt.axvspan(start_t, end_t, color=color, alpha=alpha)

    if len(centros_metronomo) > 0:
        t_centers = [t_recortada[idx] for idx in centros_metronomo if idx < len(t_recortada)]
        v_env = [env_recortada[idx] for idx in centros_metronomo if idx < len(env_recortada)]
        plt.scatter(t_centers, v_env, color='green', s=50, zorder=5, label='Beat (Centro)')
    
    plt.title(f"Señal y Ventanas (Beat al Centro) - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    max_y_val = np.max(env_recortada) if len(env_recortada) > 0 else 1.3
    plt.ylim(0, max_y_val * 1.5)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    
    print_progress_bar(0, 1, prefix='Graficando recortes:', suffix='...', length=20)
    plt.savefig(out_rec, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Graficando recortes:', suffix='OK', length=20)

    if show_plot:
        plt.show()
    else:
        plt.close(plt.gcf())

# ---------------------- NUEVA FUNCIÓN: Overlay de Músculos ---------------------
def _plot_muscle_overlay(measure_name, channels_dict, out_dir):
    all_files = set()
    for c_data in channels_dict.values():
        all_files.update(c_data.keys())
        
    for fname in all_files:
        plt.figure(figsize=(10, 6))
        
        colors = {'canal_0': 'blue', 'canal_1': 'orange', 'canal_2': 'green'}
        labels = {'canal_0': 'Músculo 1', 'canal_1': 'Músculo 2', 'canal_2': 'Músculo 3'}
        
        found_any = False
        sorted_chans = sorted(channels_dict.keys())
        
        for ch in sorted_chans:
            if fname in channels_dict[ch]:
                data = channels_dict[ch][fname]
                t = data['pulse_time']
                y = data['mean_pulse']
                
                lbl = labels.get(ch, ch)
                col = colors.get(ch, None)
                
                plt.plot(t, y, label=lbl, color=col, linewidth=2, alpha=0.8)
                found_any = True
                
        if found_any:
            plt.title(f"Patrón Muscular Centrado - {measure_name} - {fname}")
            plt.xlabel("Tiempo [s]")
            plt.ylabel("Amplitud [V]")
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5, label="Pico Músculo 1")
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.5)
            
            name_clean = os.path.splitext(fname)[0]
            path = os.path.join(out_dir, f"patron_muscular_{name_clean}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Generado gráfico patrón: {path}")
            plt.close()

# ---------------------- Export results ---------------------
def export_results_for_file(out_dir, filename, resultados_entry):
    os.makedirs(out_dir, exist_ok=True)
    export = {}
    keys = ['mean_pulse', 'pulse_time', 'amp_uncertainty',
            'umbral', 'segmentos_rs', 'shifts', 'valid_indices']
    for k in keys:
        export[k] = resultados_entry.get(k, None)
    export['file'] = filename
    
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump(export, fh, indent=2, default=lambda x: float(np.nan) if (isinstance(x, np.ndarray)) else x)
    
    full_results_path = os.path.join(out_dir, 'analisis_results.json')
    try:
        with open(full_results_path, 'w') as f:
            json.dump(resultados_entry, f, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
    except Exception as e:
        print(f"Error guardando arrays: {e}")

# ---------------------- Comparative plotting --------------------
def _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados, nombre_salida,
                       show_overlay=True, show_amplitude=True):
    
    import matplotlib.cm as cm
    n_files = len(promedios_globales)
    if n_files == 0: return

    all_lengths = [len(p) for p in promedios_globales]
    if len(set(all_lengths)) > 1:
        target_len = int(np.median(all_lengths))
        promedios_globales = [_resample_to(np.array(p), target_len) for p in promedios_globales]

    plot_colors = cm.viridis(np.linspace(0, 1, n_files))
    pulse_matrix = np.vstack(promedios_globales)
    t_plot = tiempos_globales[0] if tiempos_globales else np.linspace(0, 1, pulse_matrix.shape[1])

    if show_overlay:
        fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
        for i, pulso in enumerate(pulse_matrix):
            ax_ov.plot(t_plot, pulso, label=str(i+1), linewidth=2, alpha=0.9, color=plot_colors[i])
        ax_ov.set_title('Overlay: FORMAS MUSCULARES (Líder)')
        ax_ov.set_xlabel('Tiempo [s]')
        ax_ov.set_ylabel('Amplitud [V]')
        ax_ov.grid(True, alpha=0.4)
        ax_ov.legend(title='Archivo #', fontsize=8, loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(nombre_salida)[0]}_overlay_lider.png", dpi=300)
        plt.close(fig_ov)

    if show_amplitude:
        max_amplitudes = np.array([np.nanmax(p) for p in promedios_globales])
        sort_indices = np.argsort(max_amplitudes)[::-1]
        sorted_amplitudes = max_amplitudes[sort_indices]
        original_indices = [np.where(np.array(nombres_globales) == nombres_globales[i])[0][0] for i in sort_indices]
        
        all_amp_uncs = []
        for name in nombres_globales:
            r = resultados.get(name, {})
            all_amp_uncs.append(r.get('amp_uncertainty', 0.0))
        sorted_amp_uncs = np.array(all_amp_uncs)[sort_indices]

        fig_amp, ax_amp = plt.subplots(figsize=(max(8, 0.6 * n_files), 6))
        x = np.arange(n_files)
        bars = ax_amp.bar(x, sorted_amplitudes, yerr=sorted_amp_uncs, capsize=5, alpha=0.85, color=plot_colors[sort_indices])
        
        ax_amp.set_xticks(x)
        ax_amp.set_xticklabels([str(i+1) for i in original_indices])
        ax_amp.set_ylabel('Amplitud [V]')
        ax_amp.set_title('Amplitud Máxima (Ordenada)')
        
        for i, bar in enumerate(bars):
             height = bar.get_height()
             ax_amp.text(bar.get_x() + bar.get_width()/2.0, height, f"{height:.2f}", ha='center', va='bottom', fontsize=9)
             
        plt.tight_layout()
        plt.savefig(f"{os.path.splitext(nombre_salida)[0]}_amplitud_lider.png", dpi=300)
        plt.close(fig_amp)

# ---------------------- Main function ----------------
def procesar_wavs_promedio(
    carpeta,
    bpm=50,
    colorgrafico="blue",
    tiempoinicial=0,
    tiempofinal=25,
    nombre_salida="resultado_promedio.png",
    mostrar_individuales=True,
    mostrar_recortes=True,
    mostrar_espectrograma=True, # <-- AHORA SÍ SE USA
    frecuenciamaxima=1000,
    frecuenciaminima=0,
    colores_aleatorios=False,
    seed=None,
    espectrograma_db=False,
    peak_distance_sec=0.4,
    pre_window_sec=None,
    post_window_sec=None,
    resample_len=None,
    n_pulsos_manual=None,
    apply_envelope=True,
    smooth_ms=5,
    excluded_windows=None,
    peak_search_threshold=0.25, 
    # RESTAURADO: ruido
    noise_seconds=2,
    factor_umbral=6,
    mostrar_umbral=True,
    mostrar_tabla=True,
    plot_mode='mean',
    individual_alpha=0.25,
    lowpass_cutoff_hz=None,
    highpass_cutoff_hz=None,
    output_root="",
    display_name_for_plot="",
    show_interactive_plot=False,
    show_average_plot=False,
    apply_notch_filter=False,
    # --- ARGUMENTOS NUEVOS PARA ALINEACIÓN FORZADA ---
    dict_shifts_externos=None, # Diccionario { 'archivo.wav': [shift1, shift2...] }
    indices_validos_externos=None # Lista [0, 1, 3...] con los índices que el Master aceptó
):
    rng = np.random.RandomState(seed)
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(".wav")]
    if not archivos:
        print("No se encontraron archivos WAV en la carpeta.")
        return {}

    periodo = 60.0 / bpm
    print(f"Período estimado del pulso: {periodo:.3f} s")

    resultados = {}
    promedios_globales = []
    tiempos_globales = []
    nombres_globales = []
    plot_title_name = display_name_for_plot

    for filename in archivos:
        filepath = os.path.join(carpeta, filename)
        
        # Calibración
        calibration_factor = 1.0
        try:
            parent_dir = os.path.dirname(carpeta)
            csv_files = [f for f in os.listdir(parent_dir) if f.lower().endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(parent_dir, csv_files[0])
                df_csv = pd.read_csv(csv_path)
                channel_idx = int(os.path.basename(carpeta).split('_')[-1])
                channel_col_name = f"Canal {channel_idx}"
                if channel_col_name in df_csv.columns:
                    calibration_factor = np.max(np.abs(df_csv[channel_col_name].values))
                    print(f"[Calibración] Factor: {calibration_factor:.4f}")
        except Exception:
            calibration_factor = 1.0

        signal_normalized, samplerate = _read_wav_mono(filepath)
        signal = signal_normalized * calibration_factor 
        
        # Offset
        ns_samples = int(noise_seconds * samplerate)
        if ns_samples > 0 and ns_samples < len(signal):
            dc_offset = np.mean(signal[:ns_samples])
            signal = signal - dc_offset
            print(f"[Offset] Restado nivel DC base: {dc_offset:.5f} V")

        # Pasa-Altos
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0:
            try:
                b, a = butter(4, highpass_cutoff_hz / (0.5*samplerate), btype='high')
                signal = filtfilt(b, a, signal)
                print(f"[Filtro] Pasa-Altos aplicado a {highpass_cutoff_hz} Hz")
            except Exception as e: print(f"Error filtro HP: {e}")

        # Filtro Notch
        if apply_notch_filter:
            try:
                b, a = iirnotch(50.0, 30.0, samplerate)
                signal = filtfilt(b, a, signal)
            except Exception: pass

        signal_unfiltered = signal.copy()

        # Filtro Low-pass
        if lowpass_cutoff_hz is not None and lowpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                b, a = butter(4, lowpass_cutoff_hz / (0.5*samplerate), btype='low')
                signal = filtfilt(b, a, signal)
            except Exception: pass
        
        final_plot_title = plot_title_name or filename
        signal_abs = np.abs(signal)
        env_full = _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate)

        t = np.linspace(0, len(signal)/samplerate, len(signal), endpoint=False)
        duracion_total = len(signal)/samplerate

        mask = (t >= tiempoinicial) & (t <= duracion_total)
        signal_recortada = signal[mask]
        t_recortada = t[mask]
        env_recortada = env_full[mask]

        if len(signal_recortada) == 0: continue

        muestras_pulso = int(round(periodo * samplerate))

        start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window = _estimate_noise_window(
            signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral
        )
        if start_sample_noise <= 0: start_sample_noise = 0
        
        # --- Obtener indices válidos para el corte (Slave usa los del Master) ---
        idx_forzados = None
        if indices_validos_externos is not None and filename in indices_validos_externos:
            idx_forzados = indices_validos_externos[filename]

        # Extracción CENTRADA EN BEAT (La ventana es de -T/2 a +T/2)
        centros_metronomo, segmentos, valid_indices_local = _cortar_centrado_en_beat(
            np.abs(env_recortada), start_sample_noise, muestras_pulso, 
            n_pulsos_manual=n_pulsos_manual, excluded_windows=excluded_windows,
            forced_indices=idx_forzados
        )

        if len(segmentos) == 0:
            print(f"{filename}: no se extrajeron segmentos. Omitido.")
            continue

        segmentos_rs, target_len = _resample_segments(segmentos, resample_len)
        
        # --- LÓGICA MASTER / SLAVE (SHIFTS) ---
        shifts_to_save = []
        
        # Caso SLAVE: Tenemos shifts externos
        if dict_shifts_externos is not None and filename in dict_shifts_externos:
            print(f"[Alineación] MODO SLAVE: Aplicando alineación forzada...")
            forced_shifts = dict_shifts_externos[filename]
            segmentos_rs = _aplicar_alineacion_forzada(segmentos_rs, forced_shifts)
            shifts_to_save = forced_shifts 
            
        # Caso MASTER: Calculamos la mejor alineación con SUAVIZADO
        else:
            if len(segmentos_rs) > 1:
                print(f"[Alineación] MODO MASTER: Calculando mejor alineación por SILUETA...")
                segmentos_rs, shifts_calculated = _alinear_por_lider_calculo(segmentos_rs, samplerate)
                shifts_to_save = shifts_calculated
            else:
                shifts_to_save = [0] * len(segmentos_rs)

        segmentos_norm, pulso_promedio, pulso_sigma, pulso_err, Np = _compute_pulse_stats(segmentos_rs)

        if (sigma_est is None) or (umbral is None):
            sigma_est, umbral = _fallback_umbral(segmentos_norm, pulso_promedio, factor_umbral)

        max_amp = np.max(pulso_promedio)
        snr_manual = max_amp / umbral if (umbral is not None and umbral > 0) else np.inf

        half_T_sec = periodo / 2.0
        t_pulso = np.linspace(-half_T_sec, half_T_sec, target_len, endpoint=False)

        color_prom = tuple(rng.rand(3).tolist()) if colores_aleatorios else colorgrafico
        idx_peak = int(np.argmax(pulso_promedio))
        amp_uncertainty = pulso_err[idx_peak] if idx_peak < len(pulso_err) else 0.0
        snr_uncertainty = amp_uncertainty / umbral if (umbral is not None and umbral > 0) else np.nan
        
        out_dir = output_root
        out_prom = os.path.join(out_dir, "avg_lider.png")
        out_spec = os.path.join(out_dir, "spec_lider.png")
        out_rec = os.path.join(out_dir, "pulses_centrados.png")

        _plot_pulse_full(
            t_pulso, segmentos_norm, pulso_promedio, pulso_err, color_prom,
            filename=final_plot_title, out_prom=out_prom,
            plot_mode=plot_mode, individual_alpha=individual_alpha,
            mostrar_individuales=mostrar_individuales, show_plot=show_average_plot,
            snr_manual=snr_manual, snr_uncertainty=snr_uncertainty, 
            umbral=umbral, mostrar_umbral=mostrar_umbral
        )

        if mostrar_espectrograma:
            _plot_espectrograma(pulso_promedio, samplerate, final_plot_title, out_spec)

        if mostrar_recortes:
            _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds,
                           start_sample_noise, samplerate, centros_metronomo, periodo, muestras_pulso, out_rec, final_plot_title, 
                           excluded_windows=excluded_windows, show_plot=show_interactive_plot,
                           signal_original_unfiltered=signal_unfiltered[mask])

        promedios_globales.append(pulso_promedio)
        tiempos_globales.append(t_pulso)
        nombres_globales.append(filename)

        resultados[filename] = {
            'mean_pulse': pulso_promedio,
            'pulse_time': t_pulso,
            'amp_uncertainty': amp_uncertainty,
            'segmentos_rs': segmentos_rs,
            'periodo': periodo,
            'umbral': umbral,
            'shifts': shifts_to_save,
            'valid_indices': valid_indices_local # IMPORTANTE: Pasar indices al GUI
        }

        export_results_for_file(out_dir, filename, resultados[filename])

    if mostrar_tabla and promedios_globales:
        _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados, nombre_salida)

    return resultados

# ---------------------- GUI Classes ----------------------
class ProcessingOptionsDialog(tk.Toplevel):
    def __init__(self, root):
        self.root = root
        super().__init__(root)
        self.title("Opciones de Procesamiento")
        self.geometry("450x500")
        self.transient(root)
        self.grab_set()

        self.mediciones_a_procesar = []
        self.canales_seleccionados = {}

        main_frame = tk.Frame(self, padx=15, pady=15)
        main_frame.pack(fill="both", expand=True)

        channels_frame = tk.LabelFrame(main_frame, text="2. Seleccionar Canales", padx=10, pady=10)
        channels_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.channel_list_frame = tk.Frame(channels_frame)
        self.channel_list_frame.pack(fill="both", expand=True)

        opts_frame = tk.LabelFrame(main_frame, text="Opciones", padx=10, pady=5)
        opts_frame.pack(fill="x", pady=(0, 15))
        
        self.var_mostrar_recortes = tk.BooleanVar(value=True)
        self.var_mostrar_espectrograma = tk.BooleanVar(value=False) # Nueva variable
        self.var_excluded_windows = tk.StringVar(value="")
        self.var_lowpass_cutoff = tk.StringVar(value="1000")
        self.var_highpass_cutoff = tk.StringVar(value="20")
        self.var_notch_filter = tk.BooleanVar(value=True)
        self.var_interactive_analysis = tk.BooleanVar(value=True)

        tk.Checkbutton(opts_frame, text="Generar gráfico recortes", variable=self.var_mostrar_recortes).pack(anchor="w")
        tk.Checkbutton(opts_frame, text="Generar espectrograma", variable=self.var_mostrar_espectrograma).pack(anchor="w") # Nueva casilla
        tk.Checkbutton(opts_frame, text="Filtro Notch 50Hz", variable=self.var_notch_filter).pack(anchor="w")
        tk.Checkbutton(opts_frame, text="Modo Interactivo (Curación)", variable=self.var_interactive_analysis).pack(anchor="w")

        exclude_frame = tk.Frame(opts_frame)
        exclude_frame.pack(fill='x', pady=5)
        tk.Label(exclude_frame, text="Excluir ventanas:").pack(side="left")
        tk.Entry(exclude_frame, textvariable=self.var_excluded_windows).pack(side="left", fill="x", expand=True)

        filt_frame = tk.Frame(opts_frame)
        filt_frame.pack(fill='x', pady=5)
        tk.Label(filt_frame, text="Pasa-Altos (Hz):").pack(side="left")
        tk.Entry(filt_frame, textvariable=self.var_highpass_cutoff, width=5).pack(side="left", padx=(0, 10))
        tk.Label(filt_frame, text="Pasa-Bajos (Hz):").pack(side="left")
        tk.Entry(filt_frame, textvariable=self.var_lowpass_cutoff, width=5).pack(side="left")

        tk.Button(main_frame, text="PROCESAR", command=self.procesar, bg="#007BFF", fg="white").pack(fill="x", pady=10)

    def populate_channels(self, base_dir, mediciones):
        self.mediciones_a_procesar = mediciones
        self.BASE_DIR = base_dir
        for nombre in self.mediciones_a_procesar:
            path = os.path.join(self.BASE_DIR, nombre)
            try:
                canales = sorted([x for x in os.listdir(path) if x.startswith("canal_")])
                if channels_frame := tk.LabelFrame(self.channel_list_frame, text=nombre):
                    channels_frame.pack(fill="x")
                    for c in canales:
                        var = tk.BooleanVar(value=True)
                        self.canales_seleccionados[os.path.join(nombre, c)] = var
                        tk.Checkbutton(channels_frame, text=c, variable=var).pack(anchor="w")
            except: pass

    def procesar(self):
        canales = [k for k,v in self.canales_seleccionados.items() if v.get()]
        if not canales: return
        
        try:
            lp_freq = float(self.var_lowpass_cutoff.get())
            hp_freq = float(self.var_highpass_cutoff.get())
            excl_list = [int(x) for x in self.var_excluded_windows.get().split(',') if x.strip()]
        except: return

        apply_notch = self.var_notch_filter.get()
        interactive = self.var_interactive_analysis.get()

        self.destroy()
        self.root.destroy()

        # Agrupar por medición para manejar Master-Slave
        chans_by_meas = {}
        for rel_path in canales:
            med_name, chan_name = os.path.split(rel_path)
            if med_name not in chans_by_meas: chans_by_meas[med_name] = []
            chans_by_meas[med_name].append(chan_name)

        results_global = {}

        for med_name, chan_list in chans_by_meas.items():
            chan_list.sort() 
            master_shifts = None 
            master_valid_indices = None # NUEVO: indices que el master aceptó
            
            if med_name not in results_global: results_global[med_name] = {}

            for i, chan_name in enumerate(chan_list):
                full_path = os.path.join(self.BASE_DIR, med_name, chan_name)
                print(f"\n--- Procesando: {med_name}/{chan_name} ---")
                
                bpm, pulsos, noise_sec = 50, None, 2.0
                excl_meta = []
                meta_path = os.path.join(full_path, 'metadata.json')
                try:
                    with open(meta_path) as f:
                        d = json.load(f)
                        bpm = d.get('bpm', 50)
                        pulsos = d.get('pulse_count', None)
                        noise_sec = d.get('noise_seconds', 2.0)
                        excl_meta = d.get('excluded_windows', [])
                except: pass

                final_excl = sorted(list(set(excl_list + excl_meta)))

                # Modo Interactivo (solo Master)
                if interactive and i == 0:
                    procesar_wavs_promedio(
                        full_path, output_root=full_path, nombre_salida="dummy.png",
                        bpm=bpm, n_pulsos_manual=pulsos, noise_seconds=noise_sec, excluded_windows=[],
                        show_interactive_plot=True, apply_notch_filter=apply_notch, 
                        lowpass_cutoff_hz=lp_freq, highpass_cutoff_hz=hp_freq
                    )
                    user_in = input(f"Excluir ventanas para {chan_name} (actual {final_excl}): ")
                    if user_in.strip():
                        try:
                            new_excl = [int(x) for x in user_in.split(',')]
                            final_excl = sorted(list(set(new_excl)))
                            try:
                                with open(meta_path) as f: md = json.load(f)
                                md['excluded_windows'] = final_excl
                                with open(meta_path, 'w') as f: json.dump(md, f, indent=4)
                            except: pass
                        except: pass
                
                # Input para Slave
                shifts_input = master_shifts if i > 0 else None
                valid_indices_input = master_valid_indices if i > 0 else None
                
                # Análisis Final
                res_dict = procesar_wavs_promedio(
                    full_path, output_root=full_path, nombre_salida="analisis_final.png",
                    bpm=bpm, n_pulsos_manual=pulsos, noise_seconds=noise_sec, excluded_windows=final_excl,
                    show_interactive_plot=False, apply_notch_filter=apply_notch, 
                    lowpass_cutoff_hz=lp_freq, highpass_cutoff_hz=hp_freq,
                    mostrar_recortes=self.var_mostrar_recortes.get(),
                    mostrar_espectrograma=self.var_mostrar_espectrograma.get(), # Pasa el estado del checkbox
                    dict_shifts_externos=shifts_input,
                    indices_validos_externos=valid_indices_input # Pasamos indices al Slave
                )
                
                # Guardar info del Master
                if i == 0:
                    master_shifts = {}
                    master_valid_indices = {}
                    for fname, data in res_dict.items():
                        if 'shifts' in data: master_shifts[fname] = data['shifts']
                        if 'valid_indices' in data: master_valid_indices[fname] = data['valid_indices']
                    print(f"-> Canal Maestro ({chan_name}) definió {len(master_shifts)} archivos para sincronizar.")

                results_global[med_name][chan_name] = res_dict

                if interactive and i==0:
                    try: subprocess.run(["start", os.path.join(full_path, "pulses_centrados.png")], shell=True)
                    except: pass

        # --- APLICAR RECENTRADO POST-PROCESO (NUEVO EN V7.0) ---
        print("\n--- Aplicando Centrado Global basado en Músculo 1 ---")
        for med_name, chans_data in results_global.items():
            if 'canal_0' not in chans_data: continue # Necesitamos el Master
            
            # Iterar sobre cada archivo wav común
            archivos_wav = list(chans_data['canal_0'].keys())
            
            for fname in archivos_wav:
                # 1. Obtener datos del master
                master_pulse = chans_data['canal_0'][fname]['mean_pulse']
                center_idx = len(master_pulse) // 2
                peak_idx = np.argmax(master_pulse)
                
                # 2. Calcular shift necesario
                shift_centering = center_idx - peak_idx
                
                # 3. Aplicar a TODOS los canales de esta medición
                for ch in chans_data:
                    if fname in chans_data[ch]:
                        y_old = chans_data[ch][fname]['mean_pulse']
                        # Shift circular (np.roll) es seguro aquí porque los bordes son ruido bajo
                        y_new = np.roll(y_old, shift_centering)
                        chans_data[ch][fname]['mean_pulse'] = y_new

        print("\n--- Generando Patrones Musculares ---")
        for med_name, chans_data in results_global.items():
            meas_dir = os.path.join(self.BASE_DIR, med_name)
            _plot_muscle_overlay(med_name, chans_data, meas_dir)

        print("\n=== PROCESAMIENTO FINALIZADO ===")

class ComparativeOptionsDialog(tk.Toplevel):
    def __init__(self, root):
        self.root = root
        super().__init__(root)
        self.title("Comparación")
        self.geometry("400x300")
        
        self.mediciones = []
        self.BASE_DIR = ""
        
        self.var_canal = tk.StringVar()
        tk.Label(self, text="Canal común:").pack(pady=5)
        self.menu = tk.OptionMenu(self, self.var_canal, "")
        self.menu.pack()
        
        self.var_ov = tk.BooleanVar(value=True)
        self.var_amp = tk.BooleanVar(value=True)
        tk.Checkbutton(self, text="Overlay", variable=self.var_ov).pack()
        tk.Checkbutton(self, text="Amplitud Max", variable=self.var_amp).pack()
        
        tk.Button(self, text="LANZAR", command=self.lanzar, bg="green", fg="white").pack(pady=20)

    def populate_common_channels(self, base_dir, mediciones):
        self.mediciones = mediciones
        self.BASE_DIR = base_dir
        common = set()
        for m in mediciones:
            p = os.path.join(base_dir, m)
            try:
                ch = set(x for x in os.listdir(p) if x.startswith("canal_"))
                if not common: common = ch
                else: common &= ch
            except: pass
        
        m = self.menu['menu']
        m.delete(0, 'end')
        for c in sorted(list(common)):
            m.add_command(label=c, command=lambda v=c: self.var_canal.set(v))
        if common: self.var_canal.set(list(common)[0])

    def lanzar(self):
        c = self.var_canal.get()
        if not c: return
        self.destroy()
        self.root.destroy()
        
        glob_res = {}
        for m in self.mediciones:
            path = os.path.join(self.BASE_DIR, m, c, 'analisis_results.json')
            try:
                with open(path) as f:
                    d = json.load(f)
                    glob_res[f"{m}-{c}"] = d
            except: pass
            
        proms = [v['mean_pulse'] for v in glob_res.values()]
        times = [v['pulse_time'] for v in glob_res.values()]
        names = list(glob_res.keys())
        
        os.makedirs("comparativos", exist_ok=True)
        out = os.path.join("comparativos", f"comp_{datetime.now().strftime('%H%M%S')}.png")
        
        _comparative_plots(proms, times, names, glob_res, out, 
                           show_overlay=self.var_ov.get(), show_amplitude=self.var_amp.get())

class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Lanzador v{__version__}")
        self.root.geometry("500x400")
        
        s_dir = os.path.dirname(os.path.abspath(__file__))
        r = s_dir
        while os.path.basename(r) != 'Emg' and os.path.dirname(r) != r:
            r = os.path.dirname(r)
        self.BASE_DIR = os.path.join(r, "base_de_datos_electrodos")

        self.lst = tk.Listbox(root, selectmode=tk.EXTENDED)
        self.lst.pack(fill="both", expand=True)
        
        btn_fr = tk.Frame(root)
        btn_fr.pack(fill="x")
        tk.Button(btn_fr, text="Procesar", command=self.open_proc).pack(side="left", expand=True, fill="x")
        tk.Button(btn_fr, text="Comparar", command=self.open_comp).pack(side="left", expand=True, fill="x")
        
        self.load()

    def load(self):
        try:
            for x in sorted(os.listdir(self.BASE_DIR)):
                if os.path.isdir(os.path.join(self.BASE_DIR, x)):
                    self.lst.insert(tk.END, x)
        except: pass

    def open_proc(self):
        sel = [self.lst.get(i) for i in self.lst.curselection()]
        if sel:
            d = ProcessingOptionsDialog(self.root)
            d.populate_channels(self.BASE_DIR, sel)

    def open_comp(self):
        sel = [self.lst.get(i) for i in self.lst.curselection()]
        if len(sel) > 1:
            d = ComparativeOptionsDialog(self.root)
            d.populate_common_channels(self.BASE_DIR, sel)

if __name__ == "__main__":
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()