# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Procesamiento, filtrado y análisis de señales por track integrado.
# Versión de script: Trevisan 1.0
# ==============================================================================

import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, iirnotch
from scipy import interpolate
import csv
import pandas as pd
import math
import re
import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
from collections import defaultdict, Counter
from itertools import product, combinations

import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root


__version__ = "Trevisan-1.0"

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

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

# ---------------------- I/O & envelope ------------------------------------
def _read_wav_mono(filepath):
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal[:, 0]
    return np.asarray(signal, dtype=float), sr

def _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_env="media_movil", extreme_smooth=False):
    if tipo_env == "rms" and smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            sig_sq = signal_abs ** 2
            window = np.ones(win_len, dtype=float) / float(win_len)
            rms = np.sqrt(np.convolve(sig_sq, window, mode='same'))
            return rms
        else:
            return signal_abs.copy()

    if apply_envelope:
        try:
            from scipy.fft import next_fast_len
            N = len(signal_abs)
            fast_len = next_fast_len(N)
            env_full = np.abs(hilbert(signal_abs, N=fast_len)[:N])
        except Exception as e:
            print(f"Error en hilbert: {e}")
            env_full = signal_abs.copy()
    else:
        env_full = signal_abs.copy()

    if tipo_env == "media_movil" and smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            window = np.ones(win_len, dtype=float) / float(win_len)
            env_full = np.convolve(env_full, window, mode='same')
            
    if extreme_smooth:
        from scipy.signal import butter, filtfilt
        nyq = 0.5 * samplerate
        b_lp, a_lp = butter(2, 5.0 / nyq, btype='low', analog=False)
        env_full = filtfilt(b_lp, a_lp, env_full)
        env_full[env_full < 0] = 0
            
    return env_full

# ---------------------- Noise estimation (initial window) -------------------
def _estimate_noise_window(signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral, tipo_env="media_movil", verbose=True):
    start_sample_noise = int(round(noise_seconds * samplerate))
    if start_sample_noise <= 0:
        start_sample_noise = 0
    if start_sample_noise >= len(signal_recortada):
        start_sample_noise = min(len(signal_recortada)-1, int(round(0.01 * len(signal_recortada))))

    if start_sample_noise > 0:
        skip_samples = int(round(1.0 * samplerate))
        if start_sample_noise <= skip_samples + int(0.1 * samplerate):
            skip_samples = min(int(round(0.1 * samplerate)), start_sample_noise // 2)
            
        noise_segment = signal_recortada[skip_samples:start_sample_noise]
        if len(noise_segment) > 0:
            env_noise = _compute_env_full(np.abs(noise_segment), True, smooth_ms, samplerate, tipo_env)
        else:
            env_noise = np.array([])
        
        if len(env_noise) >= 5:
            mad = np.median(np.abs(env_noise - np.median(env_noise)))
            sigma_est = mad * 1.4826
        else:
            sigma_est = np.std(env_noise) if len(env_noise) > 0 else 0.0

        umbral = np.mean(env_noise) if len(env_noise) > 0 else 0.0
        noise_rms_from_noise_window = rms(env_noise) if len(env_noise) > 0 else 0.0
        if verbose:
            if noise_rms_from_noise_window > 0:
                print(f"[Umbral por ventana inicial] noise_seconds={noise_seconds}s, umbral (promedio)={umbral:.5e}, noise_rms_window={noise_rms_from_noise_window:.5e}")
            else:
                print(f"[Umbral] no se proporcionó ventana de ruido valida (noise_seconds={noise_seconds}).")
        return start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window
    else:
        if verbose: print(f"[Umbral] no se proporcionó ventana de ruido valida (noise_seconds={noise_seconds}).")
        return start_sample_noise, np.array([]), None, None, None

# ---------------------- Detect maxima per cut & extract ---------------------
def _detect_maxima_and_extract(env_recortada,
                               start_sample_noise,
                               muestras_pulso,
                               pre_samples,
                               post_samples,
                               peak_search_threshold,
                               n_pulsos_manual=None,
                               min_peak_distance_factor=0.5,
                               excluded_windows=None,
                               verbose=True):
    if len(env_recortada) == 0:
        return [], []

    if n_pulsos_manual is not None and n_pulsos_manual > 0:
        n_pulsos = int(n_pulsos_manual)
        if verbose: print(f"[Análisis] Usando conteo de pulsos obligatorio del metrónomo: {n_pulsos}")
    else:
        if verbose: print("--- ERROR: No se encontró un 'pulse_count' válido en metadata.json. Omitiendo archivo. ---")
        return [], []

    maxima_per_cut = []
    segmentos = []
    min_dist_samples = max(1, int(round(min_peak_distance_factor * float(muestras_pulso))))
    excluded_set = set(excluded_windows) if excluded_windows else set()

    for i in range(n_pulsos):
        cut_start = start_sample_noise + i * muestras_pulso
        cut_end = cut_start + muestras_pulso
        if cut_end > len(env_recortada):
            cut_end = len(env_recortada)
        
        window_number = i + 1
        if window_number in excluded_set:
            if verbose: print(f"    -> Omitiendo ventana #{window_number} (excluida por el usuario).")
            continue

        local_segment = env_recortada[cut_start:cut_end]
        if local_segment.size == 0:
            continue

        rel_max = int(np.argmax(local_segment))
        max_sample = cut_start + rel_max
        max_value = env_recortada[max_sample]

        if max_value < peak_search_threshold:
            continue

        seg_start = max_sample - pre_samples
        seg_end = max_sample + post_samples
        if seg_start < 0:
            continue
        if seg_end > len(env_recortada):
            seg_end = len(env_recortada)

        if len(maxima_per_cut) > 0:
            prev_idx = maxima_per_cut[-1]
            prev_val = env_recortada[prev_idx]
            if abs(max_sample - prev_idx) < min_dist_samples:
                if max_value > prev_val:
                    maxima_per_cut[-1] = int(max_sample)
                    segmentos[-1] = env_recortada[seg_start:seg_end].copy()
                continue

        segmento = env_recortada[seg_start:seg_end].copy()
        maxima_per_cut.append(int(max_sample))
        segmentos.append(segmento)

    return maxima_per_cut, segmentos

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

# ---------------------- Plot recortes (modificado para señal corregida) -----
def _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds,
                   start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso, out_rec, filename, 
                   excluded_windows=None, show_plot=False, signal_original_unfiltered=None, mostrar_senal_cruda=True,
                   show_interpulse_noise=True, is_corrected=False):
    
    plt.figure(figsize=(12, 4))
    is_dark = plt.rcParams.get('axes.facecolor', '') == 'black'
    color_env = "#08F7FE" if is_dark else "Blue"
    color_line = "white" if is_dark else "Black"

    if is_corrected:
        max_v = np.max(env_recortada) if len(env_recortada) > 0 else 1.0
        env_plot = env_recortada / max_v if max_v > 0 else env_recortada
        ylabel_str = "Amplitud Normalizada"
    else:
        env_plot = env_recortada * 1e6
        ylabel_str = "Amplitud (µV)"

    if not is_corrected:
        noise_t0 = t_recortada[0]
        noise_t1 = noise_t0 + noise_seconds
        plt.axvspan(noise_t0, noise_t1, color='violet', alpha=0.4, label=f"Ventana ruido ({noise_seconds}s)")

    plt.plot(t_recortada, env_plot, color=color_env, linewidth=2.0, linestyle='-', alpha=1.0, label="Envolvente")

    offset_start = t_recortada[0] + float(start_sample_noise)/samplerate
    duracion_analizable_grafico = len(env_recortada) - start_sample_noise
    n_pulsos = math.ceil(duracion_analizable_grafico / muestras_pulso)
    for i in range(n_pulsos+1):
        xline = offset_start + i*muestras_pulso/samplerate
        plt.axvline(x=xline, color=color_line, linestyle="--", alpha=0.6)

    excluded_set_plot = set(excluded_windows) if excluded_windows else set()
    spans = []
    noise_win_samples = max(3, int(round((periodo / 4.0) * samplerate)))

    for i in range(n_pulsos):
        start_t = offset_start + i*muestras_pulso/samplerate
        end_t = start_t + periodo
        window_number = i + 1
        color = "red" if window_number in excluded_set_plot else "orange"
        alpha = 0.3 if window_number in excluded_set_plot else 0.06
        span = plt.axvspan(start_t, end_t, color=color, alpha=alpha)
        spans.append((window_number, start_t, end_t, span))

        if show_interpulse_noise:
            if len(maxima_per_cut) > 0:
                if i == 0:
                    midpoint = maxima_per_cut[0] - (muestras_pulso // 2)
                elif i < len(maxima_per_cut):
                    midpoint = (maxima_per_cut[i-1] + maxima_per_cut[i]) // 2
                else:
                    midpoint = maxima_per_cut[-1] + (muestras_pulso // 2) * (i - len(maxima_per_cut) + 1)
            else:
                cut_start = start_sample_noise + i * muestras_pulso
                midpoint = cut_start + (muestras_pulso // 2)
                
            noise_start = max(0, int(midpoint - noise_win_samples // 2))
            noise_end = min(len(env_recortada)-1, noise_start + noise_win_samples)

            if noise_start < len(t_recortada) and noise_end < len(t_recortada):
                n_start_t = t_recortada[noise_start]
                n_end_t = t_recortada[noise_end]
                plt.axvspan(n_start_t, n_end_t, color='cyan', alpha=0.4 if is_dark else 0.3, label='Ruido inter-pulso' if i==0 else "")

    if len(maxima_per_cut) > 0:
        t_maxima = [t_recortada[idx] for idx in maxima_per_cut]
        v_max_env = [env_plot[idx] for idx in maxima_per_cut]
        plt.scatter(t_maxima, v_max_env, color='red', s=50, zorder=5, label='Máximos (envolvente)')
    
    fig = plt.gcf()
    ax = plt.gca()
    if show_plot:
        plt.title(f"Señal corregida interactiva (Click para excluir) - {filename}")
    elif is_corrected:
        plt.title(f"Señal corregida normalizada - {filename}")
    else:
        plt.title(f"Señal original - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel(ylabel_str)
    max_y_val = np.max(env_plot) if len(env_plot) > 0 else 1.3
    plt.ylim(0, max_y_val * 1.3)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    
    print_progress_bar(0, 1, prefix='Guardando gráfico...', suffix='Completado', length=40)
    plt.savefig(out_rec, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Guardando gráfico...', suffix='Completado', length=40)

    if show_plot:
        print("\nMostrando gráfico... Haz click en las ventanas para excluirlas/incluirlas. Cierra la ventana al terminar.")
        def onclick(event):
            if event.inaxes != ax: return
            x = event.xdata
            for window_number, start_t, end_t, span in spans:
                if start_t <= x <= end_t:
                    if window_number in excluded_set_plot:
                        excluded_set_plot.remove(window_number)
                        span.set_color("orange")
                        span.set_alpha(0.06)
                        print(f"[DEBUG] Ventana {window_number} incluida")
                    else:
                        excluded_set_plot.add(window_number)
                        span.set_color("red")
                        span.set_alpha(0.3)
                        print(f"[DEBUG] Ventana {window_number} excluida")
                    fig.canvas.draw_idle()
                    break
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)
        print(f"[DEBUG] Exclusiones finales para {filename}: {sorted(list(excluded_set_plot))}")
    plt.close(fig)
    return sorted(list(excluded_set_plot))

# ---------------------- Export results ---------------------
def export_results_for_file(out_dir, filename, resultados_entry):
    os.makedirs(out_dir, exist_ok=True)
    export = {}
    keys = ['mean_pulse', 'pulse_time', 'snr_manual', 'amp_uncertainty',
            'noise_rms_from_noise_window', 'umbral', 'segmentos_rs', 'env_recortada',
            't_recortada', 'env_corregida_concat', 't_concat',
            'picos_ventana', 'activation_threshold', 'activation_flags', 'excluded_windows',
            'noise_levels']
    for k in keys:
        export[k] = resultados_entry.get(k, None)
    export['file'] = filename
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump(export, fh, indent=2, default=lambda x: 'Array omitido' if isinstance(x, np.ndarray) else x)
    
    full_results_path = os.path.join(out_dir, 'analisis_results.json')
    try:
        with open(full_results_path, 'w') as f:
            json.dump(resultados_entry, f, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
    except Exception as e:
        print(f"Error guardando arrays para {filename}: {e}")

# ---------------------- Main function (nuevo flujo) ------------------------
def procesar_wavs_promedio(
    carpeta,
    bpm=50,
    colorgrafico="blue",
    tiempoinicial=0,
    tiempofinal=25,
    nombre_salida="resultado_promedio.png",
    mostrar_individuales=True,
    mostrar_recortes=True,
    mostrar_espectrograma=True,
    frecuenciamaxima=1000,
    frecuenciaminima=0,
    colores_aleatorios=False,
    seed=None,
    espectrograma_db=False,
    calcular_umbral=True,
    metodo_umbral='outside_windows',
    factor_umbral=6,
    mostrar_umbral=True,
    usar_picos=True,
    peak_prominence=None,
    peak_height=None,
    peak_distance_sec=0.4,
    pre_window_sec=None,
    post_window_sec=None,
    pre_pct=0.4,
    post_pct=0.6,
    normalize_by='rms',
    resample_len=None,
    one_max_per_cut=True,
    n_pulsos_manual=None,
    fixed_umbral_abs=0.5,
    apply_envelope=True,
    smooth_ms=250,
    tipo_envolvente="rms",
    noise_seconds=2,
    excluded_windows=None,
    peak_search_threshold=0.25,
    plot_mode='mean',
    individual_alpha=0.25,
    lowpass_cutoff_hz=500.0,
    highpass_cutoff_hz=20.0,
    output_root="/home/santiago/Documentos/codigos/Labo 6",
    display_name_for_plot="",
    show_interactive_plot=False,
    show_average_plot=False,
    apply_notch_filter=True,
    notch_q_factor=30.0,
    mostrar_senal_cruda=True,
    is_final_curation_pass=False,
    activation_percentile=90,
    verbose=True,
    extreme_smooth=False
):
    rng = np.random.RandomState(seed)
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(".wav")]
    if not archivos:
        print("No se encontraron archivos WAV en la carpeta.")
        return {}

    periodo = 60.0 / bpm
    if verbose: print(f"Período estimado del pulso: {periodo:.3f} s")

    resultados = {}
    plot_title_name = display_name_for_plot

    for filename in archivos:
        filepath = os.path.join(carpeta, filename)
        
        calibration_factor = 1.0
        try:
            parent_dir = os.path.dirname(carpeta)
            csv_files = [f for f in os.listdir(parent_dir) if f.lower().endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No se encontró archivo CSV en la carpeta padre.")
            csv_path = os.path.join(parent_dir, csv_files[0])
            df_csv = pd.read_csv(csv_path)
            channel_num_str = os.path.basename(carpeta).split('_')[-1]
            channel_idx = int(channel_num_str)
            channel_col_name = f"Canal {channel_idx}"
            if channel_col_name not in df_csv.columns:
                raise ValueError(f"La columna '{channel_col_name}' no se encontró en '{csv_path}'.")
            calibration_factor = np.max(np.abs(df_csv[channel_col_name].values))
        except Exception as e:
            calibration_factor = 1.0

        signal_normalized, samplerate = _read_wav_mono(filepath)
        signal_v = signal_normalized * calibration_factor
        
        resistencia_ohm = 100.0
        meta_path = os.path.join(carpeta, 'metadata.json')
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    if 'resistencia_ohm' in meta_data:
                        resistencia_ohm = meta_data['resistencia_ohm']
        except Exception:
            pass
            
        r_fija = 49400.0
        ganancia = 1.0 + (r_fija / resistencia_ohm)
        signal = (signal_v / ganancia) * 1e6

        if apply_notch_filter:
            try:
                b, a = iirnotch(50.0, notch_q_factor, samplerate)
                signal = filtfilt(b, a, signal)
            except Exception: pass

        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                cutoff_hp = highpass_cutoff_hz
                if cutoff_hp >= nyquist: cutoff_hp = nyquist * 0.99
                b, a = butter(4, cutoff_hp / nyquist, btype='high', analog=False)
                signal = filtfilt(b, a, signal)
            except Exception: pass

        signal_unfiltered = signal.copy()

        if lowpass_cutoff_hz is not None and lowpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                cutoff_usar = lowpass_cutoff_hz
                if cutoff_usar >= nyquist: cutoff_usar = nyquist * 0.99
                b, a = butter(4, cutoff_usar / nyquist, btype='low', analog=False)
                signal = filtfilt(b, a, signal)
            except Exception: pass
        
        duracion_total_signal = len(signal) / samplerate
        final_plot_title = plot_title_name or filename
        signal_abs = np.abs(signal)

        env_full = _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_envolvente, extreme_smooth)

        t = np.linspace(0, len(signal)/samplerate, len(signal), endpoint=False)
        mask = (t >= tiempoinicial) & (t <= duracion_total_signal)
        signal_recortada = signal[mask]
        t_recortada = t[mask]
        env_recortada = env_full[mask]

        if len(signal_recortada) == 0:
            continue

        if pre_window_sec is None:
            pre_w = pre_pct * periodo
        else:
            pre_w = pre_window_sec
        if post_window_sec is None:
            post_w = post_pct * periodo
        else:
            post_w = post_window_sec
        pre_samples = int(round(pre_w * samplerate))
        post_samples = int(round(post_w * samplerate))

        muestras_pulso = int(round(periodo * samplerate))
        if muestras_pulso <= 0:
            continue

        start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window = _estimate_noise_window(
            signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral, tipo_envolvente, verbose=verbose)
        if start_sample_noise <= 0:
            start_sample_noise = 0

        env_for_cuts = env_recortada[start_sample_noise:]
        if len(env_for_cuts) == 0:
            continue

        n_pulsos_total = int(n_pulsos_manual) if (n_pulsos_manual is not None and n_pulsos_manual > 0) else (len(env_for_cuts) // muestras_pulso)
        if n_pulsos_total == 0:
            continue

        initial_excluded = excluded_windows if excluded_windows else []
        base_threshold = umbral if umbral is not None and umbral > 0 else peak_search_threshold
        first_pass_peaks = []
        for i in range(n_pulsos_total):
            c_start = start_sample_noise + i * muestras_pulso
            c_end = min(c_start + muestras_pulso, len(env_recortada))
            if c_start < len(env_recortada):
                seg = env_recortada[c_start:c_end]
                if seg.size > 0:
                    m_val = np.max(seg)
                    if m_val >= base_threshold:
                        first_pass_peaks.append(m_val)
                        
        search_threshold_dinamico = np.mean(first_pass_peaks) * 0.5 if first_pass_peaks else base_threshold
            
        maxima_per_cut, _ = _detect_maxima_and_extract(
            np.abs(env_recortada), start_sample_noise, muestras_pulso, pre_samples, post_samples,
            search_threshold_dinamico, n_pulsos_manual=n_pulsos_manual, excluded_windows=initial_excluded, verbose=verbose)

        # --- Calcular nivel de ruido interpulso para TODAS las ventanas (1..n_pulsos_total) ---
        noise_levels = []
        noise_win_samples = max(3, int(round((periodo / 4.0) * samplerate)))
        for i in range(n_pulsos_total):
            cut_start = start_sample_noise + i * muestras_pulso
            if len(maxima_per_cut) > 0:
                pico_en_ventana = None
                for idx in maxima_per_cut:
                    if cut_start <= idx < cut_start + muestras_pulso:
                        pico_en_ventana = idx
                        break
                if pico_en_ventana is not None:
                    pos = maxima_per_cut.index(pico_en_ventana)
                    if pos == 0:
                        midpoint = pico_en_ventana - (muestras_pulso // 2)
                    else:
                        midpoint = (maxima_per_cut[pos-1] + pico_en_ventana) // 2
                else:
                    midpoint = cut_start + (muestras_pulso // 2)
            else:
                midpoint = cut_start + (muestras_pulso // 2)

            n_start = max(0, int(midpoint - noise_win_samples // 2))
            n_end = min(len(env_recortada)-1, n_start + noise_win_samples)
            if n_start < n_end:
                noise_segment = env_recortada[n_start:n_end]
                noise_level = np.mean(noise_segment) if len(noise_segment) > 0 else 0.0
            else:
                noise_level = 0.0
                
            # --- NUEVO: Condición de seguridad para ruido excesivo (pico camuflado) ---
            if noise_level > (umbral * 10.0):
                if verbose: print(f"[Aviso] Ruido interpulso muy alto en pulso {i+1} ({noise_level:.3f}). Usando ruido base ({umbral:.3f}).")
                noise_level = umbral

            noise_levels.append(noise_level)

        # --- Crear señal corregida completa (restar ruido a cada ventana) ---
        env_corregida_full = env_recortada.copy()
        for i in range(n_pulsos_total):
            cut_start = start_sample_noise + i * muestras_pulso
            cut_end = min(cut_start + muestras_pulso, len(env_recortada))
            env_corregida_full[cut_start:cut_end] -= noise_levels[i]
        env_corregida_full = np.maximum(env_corregida_full, 0)

        # --- Gráfico original (con ruido interpulso) no interactivo ---
        out_original = os.path.join(output_root, "pulses_original.png")
        if mostrar_recortes:
            _plot_recortes(
                t_recortada, signal_recortada, env_recortada, noise_seconds,
                start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso,
                out_original, final_plot_title, excluded_windows=initial_excluded,
                show_plot=False, show_interpulse_noise=True, is_corrected=False
            )

        # --- Gráfico Corregido Siempre (guardarlo en disco) ---
        out_corrected = os.path.join(output_root, "pulses_corrected.png")
        if mostrar_recortes and not show_interactive_plot:
            _plot_recortes(
                t_recortada, signal_recortada, env_corregida_full, noise_seconds,
                start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso,
                out_corrected, final_plot_title, excluded_windows=initial_excluded,
                show_plot=False, show_interpulse_noise=False, is_corrected=True
            )

        # --- Determinar exclusiones finales (interactivo o no) ---
        if show_interactive_plot:
            final_excluded = _plot_recortes(
                t_recortada, signal_recortada, env_corregida_full, noise_seconds,
                start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso,
                out_corrected, final_plot_title, excluded_windows=initial_excluded,
                show_plot=True, show_interpulse_noise=False, is_corrected=True
            )
        else:
            final_excluded = initial_excluded

        # --- Construir señal concatenada solo con ventanas buenas corregidas ---
        good_corrected = []
        for i in range(n_pulsos_total):
            if (i+1) in final_excluded:
                continue
            cut_start = start_sample_noise + i * muestras_pulso
            cut_end = min(cut_start + muestras_pulso, len(env_recortada))
            window_corrected = env_corregida_full[cut_start:cut_end]
            if window_corrected.size == 0:
                continue
            good_corrected.append(window_corrected)

        if good_corrected:
            env_corregida_concat = np.concatenate(good_corrected)
            boundaries = [0]
            cum = 0
            for w in good_corrected:
                cum += len(w)
                boundaries.append(cum)
            window_boundaries = boundaries[:-1]
            picos_ventana = [np.max(w) for w in good_corrected]
        else:
            env_corregida_concat = np.array([])
            window_boundaries = []
            picos_ventana = []

        # --- Umbral de activación (percentil de picos de ventanas buenas) ---
        if picos_ventana:
            activation_threshold = np.percentile(picos_ventana, activation_percentile)
            if verbose: print(f"[Umbral activación] Calculado con P{int(activation_percentile)} de {len(picos_ventana)} ventanas buenas: {activation_threshold:.4f}")
        else:
            activation_threshold = 0.0
        activation_flags = [1 if p > activation_threshold else 0 for p in picos_ventana]

        t_concat = np.arange(len(env_corregida_concat)) / samplerate if len(env_corregida_concat) > 0 else np.array([])

        max_amp = np.max(env_corregida_concat) if len(env_corregida_concat) > 0 else 0
        snr_manual = max_amp / umbral if (umbral is not None and umbral > 0) else np.inf

        if verbose: print(f"\nRESUMEN {filename}: Ventanas={n_pulsos_total}, buenas={len(good_corrected)}, SNR={snr_manual:.2f}, Umbral P{int(activation_percentile)}={activation_threshold:.4f}")

        resultados[filename] = {
            'maxima_per_cut': maxima_per_cut,
            'env_recortada': env_recortada,
            'env_corregida_concat': env_corregida_concat,
            't_recortada': t_recortada,
            't_concat': t_concat,
            'start_sample_noise': start_sample_noise,
            'samplerate': samplerate,
            'muestras_pulso': muestras_pulso,
            'periodo': periodo,
            'noise_seconds': noise_seconds,
            'excluded_windows': final_excluded,
            'picos_ventana': picos_ventana,
            'activation_threshold': activation_threshold,
            'activation_flags': activation_flags,
            'window_boundaries': window_boundaries,
            'noise_levels': noise_levels
        }

        export_results_for_file(output_root, filename, resultados[filename])
        plt.close('all')

    return resultados

# ================== INTERFAZ GRÁFICA ==================
class ProcessingOptionsDialog(tk.Toplevel):
    def __init__(self, root, out_dir):
        self.root = root
        self.OUT_DIR = out_dir
        super().__init__(root)
        self.title("Ñandú LSD - Opciones de Procesamiento Trevisan")
        self.geometry("600x750")
        self.transient(root)
        self.grab_set()

        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.green_neon = "#00FF00"
        self.fg_text = "#C5C6C7"
        
        self.configure(bg=self.bg_dark)
        self.mediciones_a_procesar = []
        self.canales_seleccionados = {}

        main_frame = tk.Frame(self, padx=15, pady=15, bg=self.bg_dark)
        main_frame.pack(fill="both", expand=True)

        channels_frame = tk.LabelFrame(main_frame, text="1. Seleccionar Canales a Procesar", padx=10, pady=10, bg=self.bg_panel, fg=self.cyan_neon)
        channels_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.channel_list_frame = tk.Frame(channels_frame, bg=self.bg_panel)
        self.channel_list_frame.pack(fill="both", expand=True)

        # --- Opciones de umbral ---
        threshold_frame = tk.LabelFrame(main_frame, text="2. Método de umbral", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon)
        threshold_frame.pack(fill="x", pady=(0, 15))

        self.var_threshold_mode = tk.StringVar(value="percentil")
        tk.Radiobutton(threshold_frame, text="Percentil", variable=self.var_threshold_mode, value="percentil",
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark,
                       command=self.toggle_sweep).pack(anchor="w")
        tk.Radiobutton(threshold_frame, text="Umbral fijo (0-1)", variable=self.var_threshold_mode, value="fijo",
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark,
                       command=self.toggle_sweep).pack(anchor="w")
        tk.Radiobutton(threshold_frame, text="Barrido por canal", variable=self.var_threshold_mode, value="barrido_canal",
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark,
                       command=self.toggle_sweep).pack(anchor="w")

        self.perc_frame = tk.Frame(threshold_frame, bg=self.bg_panel)
        self.perc_frame.pack(fill='x', pady=(5,0))
        tk.Label(self.perc_frame, text="Percentil (%):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_percentile = tk.StringVar(value="90")
        self.perc_entry = tk.Entry(self.perc_frame, textvariable=self.var_percentile, width=6, bg=self.bg_dark, fg=self.cyan_neon)
        self.perc_entry.pack(side="left", padx=(5,0))

        self.fixed_frame = tk.Frame(threshold_frame, bg=self.bg_panel)
        self.fixed_frame.pack(fill='x', pady=(5,0))
        tk.Label(self.fixed_frame, text="Umbral fijo:", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_fixed_threshold = tk.StringVar(value="0.50")
        self.fixed_entry = tk.Entry(self.fixed_frame, textvariable=self.var_fixed_threshold, width=6, bg=self.bg_dark, fg=self.cyan_neon)
        self.fixed_entry.pack(side="left", padx=(5,0))

        # Barrido por canal
        self.sweep_canal_frame = tk.Frame(threshold_frame, bg=self.bg_panel)
        self.sweep_canal_frame.pack(fill='x', pady=5)
        tk.Label(self.sweep_canal_frame, text="Rango común (inicio, fin, paso):", bg=self.bg_panel, fg=self.fg_text).pack(anchor="w")
        rango_frame = tk.Frame(self.sweep_canal_frame, bg=self.bg_panel)
        rango_frame.pack(fill='x')
        self.var_sweep_start = tk.StringVar(value="0.0")
        self.start_entry = tk.Entry(rango_frame, textvariable=self.var_sweep_start, width=5, bg=self.bg_dark, fg=self.cyan_neon)
        self.start_entry.pack(side="left", padx=2)
        tk.Label(rango_frame, text="Fin:", bg=self.bg_panel, fg=self.fg_text).pack(side="left", padx=(10,0))
        self.var_sweep_stop = tk.StringVar(value="1.0")
        self.stop_entry = tk.Entry(rango_frame, textvariable=self.var_sweep_stop, width=5, bg=self.bg_dark, fg=self.cyan_neon)
        self.stop_entry.pack(side="left", padx=2)
        tk.Label(rango_frame, text="Paso:", bg=self.bg_panel, fg=self.fg_text).pack(side="left", padx=(10,0))
        self.var_sweep_step = tk.StringVar(value="0.1")
        self.step_entry = tk.Entry(rango_frame, textvariable=self.var_sweep_step, width=5, bg=self.bg_dark, fg=self.cyan_neon)
        self.step_entry.pack(side="left", padx=2)

        # Opciones avanzadas de barrido
        sweep_adv_frame = tk.Frame(self.sweep_canal_frame, bg=self.bg_panel)
        sweep_adv_frame.pack(fill='x', pady=(5,0))
        tk.Label(sweep_adv_frame, text="Pureza mínima objetivo (0-1):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_min_purity = tk.StringVar(value="0.70")
        tk.Entry(sweep_adv_frame, textvariable=self.var_min_purity, width=5, bg=self.bg_dark, fg=self.cyan_neon).pack(side="left", padx=2)
        
        tk.Label(sweep_adv_frame, text="Forzar modas (0=Auto):", bg=self.bg_panel, fg=self.fg_text).pack(side="left", padx=(10,0))
        self.var_target_modas = tk.StringVar(value="0")
        tk.Entry(sweep_adv_frame, textvariable=self.var_target_modas, width=4, bg=self.bg_dark, fg=self.cyan_neon).pack(side="left", padx=2)

        # Filtro SNR
        snr_frame = tk.Frame(threshold_frame, bg=self.bg_panel)
        snr_frame.pack(fill='x', pady=(5,0))
        tk.Label(snr_frame, text="Umbral SNR mínimo (0=sin filtro):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_snr_threshold = tk.StringVar(value="0.0")
        tk.Entry(snr_frame, textvariable=self.var_snr_threshold, width=6, bg=self.bg_dark, fg=self.cyan_neon).pack(side="left", padx=(5,0))

        # --- Opciones de Envolvente ---
        env_frame = tk.LabelFrame(main_frame, text="2.5. Configuración de Envolvente", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon)
        env_frame.pack(fill="x", pady=(0, 15))

        tk.Label(env_frame, text="Tipo:", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_tipo_env = tk.StringVar(value="rms")
        ttk.Combobox(env_frame, textvariable=self.var_tipo_env, values=["rms", "media_movil"], state="readonly", width=12).pack(side="left", padx=(5, 15))

        tk.Label(env_frame, text="Suavizado (ms):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_smooth_ms = tk.IntVar(value=250)
        ttk.Spinbox(env_frame, from_=10, to=1000, increment=10, textvariable=self.var_smooth_ms, width=6).pack(side="left", padx=(5, 0))

        # --- Opciones de curación ---
        curation_frame = tk.LabelFrame(main_frame, text="3. Curación", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon)
        curation_frame.pack(fill="x", pady=(0, 15))

        self.var_mostrar_recortes = tk.BooleanVar(value=True)
        tk.Checkbutton(curation_frame, text="Graficar recortes (original y corregido)", variable=self.var_mostrar_recortes,
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(anchor="w")

        self.var_aplicar_mediana = tk.BooleanVar(value=True)
        tk.Checkbutton(curation_frame, text="1. Suavizar Picos (Sliding Window / Mediana)", variable=self.var_aplicar_mediana,
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(anchor="w")
        self.var_aplicar_detrending = tk.BooleanVar(value=True)
        tk.Checkbutton(curation_frame, text="2. Corregir Fatiga (Detrending Lineal)", variable=self.var_aplicar_detrending,
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(anchor="w")

        exclude_frame = tk.Frame(curation_frame, bg=self.bg_panel)
        exclude_frame.pack(fill='x', pady=(5,0))
        self.var_excluir_primera = tk.BooleanVar(value=True)
        tk.Checkbutton(exclude_frame, text="Excluir 1ra Ventana", variable=self.var_excluir_primera,
                       bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(side="left", padx=(0,5))
        tk.Label(exclude_frame, text="Otras (ej: 2,24):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        self.var_excluded_windows = tk.StringVar(value="")
        tk.Entry(exclude_frame, textvariable=self.var_excluded_windows, bg=self.bg_dark, fg=self.cyan_neon).pack(side="left", fill="x", expand=True, padx=(5,0))

        btn_frame = tk.Frame(main_frame, bg=self.bg_dark)
        btn_frame.pack(fill="x", pady=(10, 0))
        
        btn_procesar = tk.Button(btn_frame, text="Procesar y Curar", command=lambda: self.procesar(interactivo=True),
                                 bg="#111111", fg=self.green_neon)
        btn_procesar.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 5))
        
        btn_rapido = tk.Button(btn_frame, text="Reprocesar Rápido", command=lambda: self.procesar(interactivo=False),
                               bg="#111111", fg=self.cyan_neon)
        btn_rapido.pack(side="right", fill="x", expand=True, ipady=5, padx=(5, 0))

        self.toggle_sweep()

    def toggle_sweep(self):
        mode = self.var_threshold_mode.get()
        if mode == "percentil":
            self.perc_entry.config(state='normal')
            self.fixed_entry.config(state='disabled')
            self.start_entry.config(state='disabled')
            self.stop_entry.config(state='disabled')
            self.step_entry.config(state='disabled')
        elif mode == "fijo":
            self.perc_entry.config(state='disabled')
            self.fixed_entry.config(state='normal')
            self.start_entry.config(state='disabled')
            self.stop_entry.config(state='disabled')
            self.step_entry.config(state='disabled')
        elif mode == "barrido_canal":
            self.perc_entry.config(state='disabled')
            self.fixed_entry.config(state='disabled')
            self.start_entry.config(state='normal')
            self.stop_entry.config(state='normal')
            self.step_entry.config(state='normal')

    def populate_channels(self, base_dir, mediciones):
        self.mediciones_a_procesar = mediciones
        self.BASE_DIR = base_dir
        canales_unicos = set()
        for nombre_medicion in self.mediciones_a_procesar:
            path_medicion = os.path.join(self.BASE_DIR, nombre_medicion)
            try:
                canales = [item for item in os.listdir(path_medicion) 
                           if os.path.isdir(os.path.join(path_medicion, item)) 
                           and item.startswith("canal_") 
                           and item != "canal_3"]
                canales_unicos.update(canales)
            except Exception: pass

        canales_ordenados = sorted(list(canales_unicos), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        if canales_ordenados:
            for canal in canales_ordenados:
                var = tk.BooleanVar(value=True)
                self.canales_seleccionados[canal] = var
                tk.Checkbutton(self.channel_list_frame, text=canal, variable=var, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(anchor="w")

    def procesar(self, interactivo=True):
        canales_globales = [canal for canal, var in self.canales_seleccionados.items() if var.get()]
        meas_to_channels = defaultdict(list)
        for nombre_medicion in self.mediciones_a_procesar:
            for canal in canales_globales:
                rel_path = os.path.join(nombre_medicion, canal)
                abs_path = os.path.join(self.BASE_DIR, rel_path)
                if os.path.exists(abs_path):
                    meas_to_channels[nombre_medicion].append(rel_path)
        
        try:
            base_excluded = [int(x.strip()) for x in self.var_excluded_windows.get().split(',') if x.strip()]
            if self.var_excluir_primera.get() and 1 not in base_excluded:
                base_excluded.append(1)
        except ValueError:
            base_excluded = [1]

        threshold_mode = self.var_threshold_mode.get()
        sweep_canal = (threshold_mode == "barrido_canal")
        percentil = None
        fixed_threshold = None
        if not sweep_canal:
            if threshold_mode == "percentil":
                try:
                    percentil = float(self.var_percentile.get().strip())
                    if not (0 < percentil <= 100):
                        percentil = 90.0
                except ValueError:
                    percentil = 90.0
            else:
                try:
                    fixed_threshold = float(self.var_fixed_threshold.get().strip())
                    if not (0.0 <= fixed_threshold <= 1.0):
                        fixed_threshold = 0.5
                except ValueError:
                    fixed_threshold = 0.5
        else:
            try:
                sweep_start = float(self.var_sweep_start.get().strip())
                sweep_stop = float(self.var_sweep_stop.get().strip())
                sweep_step = float(self.var_sweep_step.get().strip())
                thresholds_vals = np.arange(sweep_start, sweep_stop + sweep_step/2, sweep_step)
                thresholds_vals = thresholds_vals[(thresholds_vals >= 0) & (thresholds_vals <= 1)]
                if len(thresholds_vals) == 0:
                    raise ValueError
            except:
                thresholds_vals = np.arange(0.0, 1.01, 0.1)

        try:
            snr_threshold = float(self.var_snr_threshold.get().strip())
        except ValueError:
            snr_threshold = 0.0

        self.destroy()
        self.root.withdraw()

        exclusion_por_medicion = {}
        for med_name, ch_list in meas_to_channels.items():
            if interactivo:
                meta_path = os.path.join(self.BASE_DIR, med_name, "canal_0", "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                        final_excl = meta_data.get("excluded_windows", [])
                else:
                    final_excl = base_excluded
                exclusion_por_medicion[med_name] = final_excl
            else:
                meta_path = os.path.join(self.BASE_DIR, med_name, "canal_0", "metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                        final_excl = meta_data.get("excluded_windows", [])
                else:
                    final_excl = base_excluded
                exclusion_por_medicion[med_name] = final_excl

        # Suavizado y tipo desde UI
        try:
            smooth_ms = self.var_smooth_ms.get()
        except:
            smooth_ms = 250
        tipo_env = self.var_tipo_env.get()
        aplicar_mediana = self.var_aplicar_mediana.get()
        aplicar_detrending = self.var_aplicar_detrending.get()

        def procesar_mediciones(smooth_ms, tipo_env, exclusion_por_medicion, snr_threshold, aplicar_mediana, aplicar_detrending):
            datos_para_plot_combinado = {}
            for med_name, ch_list in meas_to_channels.items():
                final_excl = exclusion_por_medicion[med_name]
                for ch_rel in ch_list:
                    if os.path.basename(ch_rel) == "canal_3":
                        continue
                    carpeta = os.path.join(self.BASE_DIR, ch_rel)
                    bpm_u, noise_u, pulsos_u = 50, 2.0, None
                    meta_path = os.path.join(carpeta, 'metadata.json')
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            bpm_u = meta.get('bpm', bpm_u)
                            noise_u = meta.get('noise_seconds', noise_u)
                            pulsos_u = meta.get('pulse_count', pulsos_u)
                    except: pass
                    res_final = procesar_wavs_promedio(
                        carpeta=carpeta, output_root=carpeta,
                        bpm=bpm_u, mostrar_recortes=False,
                        display_name_for_plot=f"{med_name} ({os.path.basename(ch_rel)})",
                        noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
                        excluded_windows=final_excl,
                        show_interactive_plot=False,
                        smooth_ms=smooth_ms,
                        tipo_envolvente=tipo_env,
                        activation_percentile=90  # no se usa realmente
                    )
                    if med_name not in datos_para_plot_combinado:
                        datos_para_plot_combinado[med_name] = {}
                    if res_final:
                        fname = list(res_final.keys())[0]
                        datos_para_plot_combinado[med_name][os.path.basename(ch_rel)] = res_final[fname]

            sweep_meas_data = []
            for med_name, canales_data in datos_para_plot_combinado.items():
                if not canales_data or len(canales_data) < 2:
                    continue
                folder_name = med_name.split('\\')[-1] if '\\' in med_name else med_name.split('/')[-1]
                vocal = folder_name.split('_')[0].upper()
                sorted_chs = sorted(canales_data.keys())
                excl_set = set(exclusion_por_medicion[med_name])

                picos_list, noise_lists, envs_concat, boundaries_list = [], [], [], []
                muestras_pulso = None
                samplerate = None
                for ch in sorted_chs:
                    data = canales_data[ch]
                    full_noise = np.array(data.get('noise_levels', []))
                    n_total = len(full_noise)
                    good_indices = [i for i in range(n_total) if (i+1) not in excl_set]
                    noise_good = full_noise[good_indices].tolist()
                    picos_list.append(data.get('picos_ventana', []))
                    noise_lists.append(noise_good)
                    envs_concat.append(data.get('env_corregida_concat', np.array([])))
                    boundaries_list.append(data.get('window_boundaries', []))
                    if muestras_pulso is None:
                        muestras_pulso = data.get('muestras_pulso', 2400)
                    if samplerate is None:
                        samplerate = data.get('samplerate', 2000)

                n_w = len(picos_list[0])
                if any(len(p) != n_w for p in picos_list) or any(len(n) != n_w for n in noise_lists):
                    continue

                snr_per_ch = np.zeros((3, n_w))
                for c in range(3):
                    for j in range(n_w):
                        p = picos_list[c][j]
                        n_val = noise_lists[c][j]
                        snr_per_ch[c, j] = p / n_val if n_val > 0 else 0.0
                snr_mask = np.all(snr_per_ch >= snr_threshold, axis=0)

                if not np.all(snr_mask):
                    picos_list_filt = [np.array(p)[snr_mask].tolist() for p in picos_list]
                    noise_lists_filt = [np.array(n)[snr_mask].tolist() for n in noise_lists]
                    new_envs, new_boundaries = [], []
                    for idx_ch in range(len(sorted_chs)):
                        env_orig = envs_concat[idx_ch]
                        boundaries_orig = boundaries_list[idx_ch]
                        new_env_parts, new_bounds, cum = [], [0], 0
                        for j, start_idx in enumerate(boundaries_orig):
                            end_idx = start_idx + muestras_pulso
                            if end_idx > len(env_orig): end_idx = len(env_orig)
                            if snr_mask[j]:
                                segment = env_orig[start_idx:end_idx]
                                new_env_parts.append(segment)
                                cum += len(segment)
                                new_bounds.append(cum)
                        new_envs.append(np.concatenate(new_env_parts) if new_env_parts else np.array([]))
                        new_boundaries.append(new_bounds[:-1])
                    for idx_ch, ch in enumerate(sorted_chs):
                        canales_data[ch]['env_corregida_concat'] = new_envs[idx_ch]
                        canales_data[ch]['window_boundaries'] = new_boundaries[idx_ch]
                        canales_data[ch]['picos_ventana'] = picos_list_filt[idx_ch]
                        canales_data[ch]['noise_levels'] = noise_lists_filt[idx_ch]
                        canales_data[ch]['t_concat'] = np.arange(len(new_envs[idx_ch])) / samplerate
                    n_w = len(picos_list_filt[0])
                    picos_list = picos_list_filt
                    noise_lists = noise_lists_filt

                if n_w == 0:
                    continue

                picos_matrix = np.column_stack(picos_list)
                
                # Paso 1: Sliding Window (Mediana Móvil)
                if aplicar_mediana:
                    import pandas as pd
                    df_picos = pd.DataFrame(picos_matrix)
                    picos_procesados = df_picos.rolling(window=15, center=True, min_periods=1).median().values
                else:
                    picos_procesados = picos_matrix.copy()

                # Paso 2: Detrending Lineal
                if aplicar_detrending:
                    picos_detrended = picos_procesados.copy()
                    x_idx = np.arange(len(picos_detrended))
                    for c_idx in range(picos_detrended.shape[1]):
                        y_vals = picos_detrended[:, c_idx]
                        if len(y_vals) > 1:
                            slope, intercept = np.polyfit(x_idx, y_vals, 1)
                            trend = slope * x_idx + intercept
                            picos_detrended[:, c_idx] = np.maximum(y_vals - trend + np.mean(y_vals), 0.0)
                    picos_procesados = picos_detrended

                # Paso 3: Normalización cruzada (Relativa entre canales)
                max_pico_ventana = np.max(picos_procesados, axis=1) + 1e-9
                picos_norm = picos_procesados / max_pico_ventana[:, np.newaxis]

                if sweep_canal or (percentil is not None) or (fixed_threshold is not None):
                    sweep_meas_data.append({'vocal': vocal, 'picos_norm': picos_norm.copy()})

                for idx_ch, ch in enumerate(sorted_chs):
                    canales_data[ch]['picos_ventana_norm'] = picos_norm[:, idx_ch]
                    env_orig = canales_data[ch]['env_corregida_concat']
                    boundaries = canales_data[ch]['window_boundaries']
                    raw_peaks_ch = canales_data[ch]['picos_ventana']
                    if len(env_orig) == 0:
                        continue
                    env_norm = env_orig.copy()
                    for i, start_idx in enumerate(boundaries):
                        end_idx = start_idx + muestras_pulso
                        if end_idx > len(env_norm): end_idx = len(env_norm)
                        # Escalar la señal para que el dibujo coincida EXACTAMENTE con el pico suavizado
                        factor = picos_norm[i, idx_ch] / (raw_peaks_ch[i] + 1e-9)
                        env_norm[start_idx:end_idx] = env_orig[start_idx:end_idx] * factor
                    canales_data[ch]['env_corregida_concat_norm'] = env_norm

            return sweep_meas_data, datos_para_plot_combinado

        sweep_meas_data, datos_para_plot_combinado = procesar_mediciones(smooth_ms, tipo_env, exclusion_por_medicion, snr_threshold, aplicar_mediana, aplicar_detrending)

        # Generar boxplot de separabilidad estadística de los picos
        if sweep_meas_data:
            try:
                fig_sep, axs_sep = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
                channels = ['Canal 0', 'Canal 1', 'Canal 2']
                vocales_sep = sorted(list(set(m['vocal'] for m in sweep_meas_data)))
                for idx_ch in range(3):
                    ax = axs_sep[idx_ch]
                    data_por_vocal = []
                    for v in vocales_sep:
                        vocal_peaks = np.concatenate([m['picos_norm'][:, idx_ch] for m in sweep_meas_data if m['vocal'] == v])
                        data_por_vocal.append(vocal_peaks)
                    
                    bp = ax.boxplot(data_por_vocal, labels=vocales_sep, patch_artist=True)
                    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'][:len(vocales_sep)]):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.6)
                        
                    ax.set_title(channels[idx_ch])
                    ax.set_xlabel('Vocal')
                    ax.set_ylabel('Amplitud de Pico Normalizada')
                    ax.grid(True, alpha=0.3)
                fig_sep.suptitle('Separabilidad Estadística de Picos por Canal y Vocal', fontsize=14)
                plt.tight_layout()
                boxplot_sep_path = os.path.join(self.OUT_DIR, "separabilidad_picos_boxplot.png")
                plt.savefig(boxplot_sep_path, dpi=300, bbox_inches='tight')
                plt.close(fig_sep)
                print(f"\nBoxplot de separabilidad estadística guardado en {boxplot_sep_path}")
            except Exception as e:
                print(f"Error generando boxplot de separabilidad: {e}")

        if sweep_canal and sweep_meas_data:
            print("\n===== Barrido de umbrales por canal =====")
            try:
                min_avg_purity = float(self.var_min_purity.get())
                if min_avg_purity > 1.0:
                    min_avg_purity /= 100.0
            except ValueError:
                min_avg_purity = 0.7
            try:
                target_min_modas = int(self.var_target_modas.get())
            except ValueError:
                target_min_modas = 0

            best_score = -1
            best_sum_purity = -1.0
            best_thresholds = None
            best_modas = None
            best_candidates = []
            best_thresholds_list = []
            all_best_triplets = []  # (th0, th1, th2, per_vowel_purity)
            # Para el fallback por pureza mínima
            best_for_unique = {}  # unique -> (thresholds, sum_purity, avg_purity, modas, per_vowel_purity)
            candidates_by_unique = defaultdict(list)  # unique -> list of (th0,th1,th2, avg_purity, per_vowel_purity)
            total_comb = len(thresholds_vals)**3
            comb_index = 0
            for th0 in thresholds_vals:
                for th1 in thresholds_vals:
                    for th2 in thresholds_vals:
                        comb_index += 1
                        print_progress_bar(comb_index, total_comb, prefix=f"  ({th0:.2f},{th1:.2f},{th2:.2f})", length=50)
                        global_pats = defaultdict(list)
                        for med in sweep_meas_data:
                            vocal = med['vocal']
                            pn = med['picos_norm']
                            flags0 = (pn[:,0] > th0).astype(int)
                            flags1 = (pn[:,1] > th1).astype(int)
                            flags2 = (pn[:,2] > th2).astype(int)
                            for j in range(pn.shape[0]):
                                global_pats[vocal].append((flags0[j], flags1[j], flags2[j]))
                        modas = {}
                        per_vowel_purity = {}
                        for v, pats in global_pats.items():
                            cnt = Counter(pats)
                            mode_pat, mode_cnt = cnt.most_common(1)[0]
                            modas[v] = mode_pat
                            per_vowel_purity[v] = mode_cnt / len(pats)
                        unique = len(set(modas.values()))
                        sum_purity = sum(per_vowel_purity.values())
                        num_vocals = len(per_vowel_purity)
                        avg_purity = sum_purity / num_vocals if num_vocals > 0 else 0.0

                        # Guardar en candidates_by_unique
                        candidates_by_unique[unique].append((th0, th1, th2, avg_purity, per_vowel_purity.copy()))

                        # Actualizar best_for_unique
                        if unique not in best_for_unique or sum_purity > best_for_unique[unique][1]:
                            best_for_unique[unique] = ((th0, th1, th2), sum_purity, avg_purity, modas, per_vowel_purity.copy())

                        # Lógica original para best_score (sin usar para la selección final)
                        if unique > best_score:
                            best_score = unique
                            best_sum_purity = sum_purity
                            best_thresholds = (th0, th1, th2)
                            best_modas = modas
                            best_candidates = [(best_thresholds, best_score, avg_purity, per_vowel_purity)]
                            best_thresholds_list = [(th0, th1, th2)]
                            all_best_triplets = [(th0, th1, th2, per_vowel_purity.copy())]
                        elif unique == best_score:
                            best_thresholds_list.append((th0, th1, th2))
                            all_best_triplets.append((th0, th1, th2, per_vowel_purity.copy()))
                            if sum_purity > best_sum_purity:
                                best_sum_purity = sum_purity
                                best_thresholds = (th0, th1, th2)
                                best_modas = modas
                            if not any(abs(c[2] - avg_purity) < 1e-9 for c in best_candidates):
                                best_candidates.append(((th0, th1, th2), unique, avg_purity, per_vowel_purity))

            # Seleccionar la mejor combinación que cumpla los requisitos
            selected_unique = None
            for unique in sorted(best_for_unique.keys(), reverse=True):
                _, _, avg_pur, _, _ = best_for_unique[unique]
                if avg_pur >= min_avg_purity and (target_min_modas == 0 or unique >= target_min_modas):
                    selected_unique = unique
                    break
            
            # Si ninguna cumple ambas y exigimos modas, priorizamos las modas a costa de la pureza
            if selected_unique is None and target_min_modas > 0:
                for unique in sorted(best_for_unique.keys(), reverse=True):
                    if unique >= target_min_modas:
                        selected_unique = unique
                        break

            if selected_unique is None:
                selected_unique = max(best_for_unique.keys())
                print(f"\nNo se alcanzó pureza={min_avg_purity} ni modas={target_min_modas}. Usando la mejor disponible ({selected_unique}).")
            else:
                print(f"\nSe seleccionó configuración con {selected_unique} modas únicas.")
            best_thresholds, _, best_avg_purity, best_modas, best_purities = best_for_unique[selected_unique]

            print(f"\nCombinaciones evaluadas: {total_comb}")
            print(f"Configuración elegida: umbrales = {best_thresholds} -> {selected_unique} modas únicas, pureza promedio = {best_avg_purity:.3f}")
            for v, moda in best_modas.items():
                print(f"  {v}: {moda}")

            # Obtener tripletes para el selected_unique
            selected_triplets = candidates_by_unique[selected_unique]
            # Filtrar por pureza mínima (opcional)
            feasible_selected = [t for t in selected_triplets if t[3] >= min_avg_purity]

            # ---- Boxplot de umbrales ----
            if feasible_selected:
                th0_vals = [t[0] for t in feasible_selected]
                th1_vals = [t[1] for t in feasible_selected]
                th2_vals = [t[2] for t in feasible_selected]
                fig, ax = plt.subplots(figsize=(8,6))
                bp = ax.boxplot([th0_vals, th1_vals, th2_vals], tick_labels=['Canal 0', 'Canal 1', 'Canal 2'], patch_artist=True)
                for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightcoral']):
                    patch.set_facecolor(color)
                ax.scatter([1,2,3], [best_thresholds[0], best_thresholds[1], best_thresholds[2]],
                           color='red', marker='D', s=80, label='Mejor combinación')
                if th0_vals:
                    ax.hlines(y=[min(th0_vals), max(th0_vals)], xmin=0.8, xmax=1.2, colors='gray', linestyles='dashed')
                    ax.hlines(y=[min(th1_vals), max(th1_vals)], xmin=1.8, xmax=2.2, colors='gray', linestyles='dashed')
                    ax.hlines(y=[min(th2_vals), max(th2_vals)], xmin=2.8, xmax=3.2, colors='gray', linestyles='dashed')
                ax.set_title(f'Umbrales que producen {selected_unique} modas únicas (candidatos)')
                ax.set_ylabel('Valor del umbral')
                ax.legend()
                plt.tight_layout()
                boxplot_path = os.path.join(self.OUT_DIR, "umbrales_boxplot.png")
                plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"\nBoxplot de umbrales guardado en {boxplot_path}")
            else:
                print("\nNo se generará boxplot de umbrales (no hay candidatos suficientes).")

            # ---- Gráfico de purezas con error ----
            vowels = sorted(best_purities.keys())
            purity_min = {v: 1.0 for v in vowels}
            purity_max = {v: 0.0 for v in vowels}
            for (_, _, _, _, purities) in feasible_selected:
                for v, p in purities.items():
                    if p < purity_min[v]: purity_min[v] = p
                    if p > purity_max[v]: purity_max[v] = p
            mean_purities = [best_purities[v] for v in vowels]
            
            fig2, ax2 = plt.subplots(figsize=(10,6))
            x = np.arange(len(vowels))
            
            if feasible_selected:
                yerr_low = [best_purities[v] - purity_min[v] for v in vowels]
                yerr_high = [purity_max[v] - best_purities[v] for v in vowels]
                bars = ax2.bar(x, mean_purities, yerr=[yerr_low, yerr_high], capsize=8,
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            else:
                bars = ax2.bar(x, mean_purities, capsize=8,
                               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                

            ax2.set_xticks(x)
            ax2.set_xticklabels(vowels)
            ax2.set_ylabel('Pureza (probabilidad de la moda)')
            ax2.set_title(f'Pureza por vocal con mejor combinación ({best_thresholds})\nBarras de error: rango entre combinaciones óptimas ({selected_unique} modas)')
            ax2.set_ylim(0, 1.05)
            for i, bar in enumerate(bars):
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                         f'{mean_purities[i]:.2f}', ha='center', va='bottom')
            plt.tight_layout()
            purity_path = os.path.join(self.OUT_DIR, "porcentajes_vocales.png")
            plt.savefig(purity_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Gráfico de purezas guardado en {purity_path}")

            # ---- Sensibilidad ----
            print("\n===== Análisis de sensibilidad =====")
            sens_range = thresholds_vals
            fig_sens, axs_sens = plt.subplots(1, 3, figsize=(18, 6))
            channels = ['Canal 0', 'Canal 1', 'Canal 2']
            opt_thresholds = best_thresholds
            for idx_ch in range(3):
                ax = axs_sens[idx_ch]
                th_vals = []
                purity_curves = defaultdict(list)
                for th_val in sens_range:
                    th_trio = list(opt_thresholds)
                    th_trio[idx_ch] = th_val
                    th0, th1, th2 = th_trio
                    global_pats = defaultdict(list)
                    for med in sweep_meas_data:
                        vocal = med['vocal']
                        pn = med['picos_norm']
                        flags0 = (pn[:,0] > th0).astype(int)
                        flags1 = (pn[:,1] > th1).astype(int)
                        flags2 = (pn[:,2] > th2).astype(int)
                        for j in range(pn.shape[0]):
                            global_pats[vocal].append((flags0[j], flags1[j], flags2[j]))
                    modas = {}
                    per_vowel_purity = {}
                    for v, pats in global_pats.items():
                        cnt = Counter(pats)
                        mode_pat, mode_cnt = cnt.most_common(1)[0]
                        modas[v] = mode_pat
                        per_vowel_purity[v] = mode_cnt / len(pats)
                    for v in sorted(per_vowel_purity.keys()):
                        purity_curves[v].append(per_vowel_purity[v])
                    th_vals.append(th_val)
                for v in sorted(purity_curves.keys()):
                    ax.plot(th_vals, purity_curves[v], label=v, linewidth=2)
                ax.axvline(opt_thresholds[idx_ch], color='red', linestyle='--', label=f'Óptimo ({opt_thresholds[idx_ch]:.3f})')
                ax.set_title(f'{channels[idx_ch]}')
                ax.set_xlabel('Umbral')
                ax.set_ylabel('Pureza')
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)
            fig_sens.suptitle('Sensibilidad de la pureza por vocal al variar cada umbral\n(los otros dos umbrales se mantienen en su valor óptimo)', fontsize=14)
            plt.tight_layout(rect=[0,0.03,1,0.95])
            sens_path = os.path.join(self.OUT_DIR, "sensibilidad_umbrales.png")
            plt.savefig(sens_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Gráfico de sensibilidad guardado en {sens_path}")

            # ---- Rango de umbrales con pureza mínima ----
            min_purity = 0.5
            feasible = []
            for (th0, th1, th2, _, purities) in feasible_selected:
                if all(p >= min_purity for p in purities.values()):
                    feasible.append((th0, th1, th2))
            if feasible:
                th0_feas = [t[0] for t in feasible]
                th1_feas = [t[1] for t in feasible]
                th2_feas = [t[2] for t in feasible]
                print(f"\nRangos de umbrales que mantienen pureza ≥ {min_purity}:")
                print(f"  Canal 0: [{min(th0_feas):.3f}, {max(th0_feas):.3f}]")
                print(f"  Canal 1: [{min(th1_feas):.3f}, {max(th1_feas):.3f}]")
                print(f"  Canal 2: [{min(th2_feas):.3f}, {max(th2_feas):.3f}]")
                print(f"  Total combinaciones en esta región: {len(feasible)}")
            else:
                print(f"\nNo se encontraron combinaciones con pureza ≥ {min_purity} para todas las vocales.")

            th0_opt, th1_opt, th2_opt = best_thresholds
        else:
            if fixed_threshold is not None:
                th = fixed_threshold
                print(f"\nUmbral fijo: {th:.4f}")
            else:
                all_norm_peaks = np.concatenate([m['picos_norm'].flatten() for m in sweep_meas_data]) if sweep_meas_data else []
                th = np.percentile(all_norm_peaks, percentil) if len(all_norm_peaks) > 0 else 0.5
                print(f"Umbral P{int(percentil)} = {th:.4f}")
            th0_opt = th1_opt = th2_opt = th

        # Generar gráficos y resumen final
        threshold_norm = (th0_opt, th1_opt, th2_opt) if sweep_canal else th0_opt
        global_patterns = defaultdict(list)

        for med_name, canales_data in datos_para_plot_combinado.items():
            if not canales_data or len(canales_data) < 2:
                continue
            folder_name = med_name.split('\\')[-1] if '\\' in med_name else med_name.split('/')[-1]
            vocal = folder_name.split('_')[0].upper()
            sorted_chs = sorted(canales_data.keys())
            n_w = len(canales_data[sorted_chs[0]].get('picos_ventana', []))
            if n_w == 0:
                continue

            flags_matrix = []
            for idx_ch, ch in enumerate(sorted_chs):
                pn = canales_data[ch].get('picos_ventana_norm')
                if pn is None:
                    picos_orig = canales_data[ch]['picos_ventana']
                    max_peak = np.max(np.column_stack([canales_data[c]['picos_ventana'] for c in sorted_chs]), axis=1)
                    pn = np.array(picos_orig) / (max_peak + 1e-9)
                ch_id = int(ch.split('_')[-1]) if '_' in ch else idx_ch
                th_ch = [th0_opt, th1_opt, th2_opt][ch_id] if sweep_canal else th0_opt
                flags = [1 if p > th_ch else 0 for p in pn]
                canales_data[ch]['activation_flags'] = flags
                canales_data[ch]['activation_threshold'] = th_ch
                flags_matrix.append(flags)

            for j in range(n_w):
                pattern = tuple(flags_matrix[i][j] for i in range(len(sorted_chs)))
                global_patterns[vocal].append(pattern)

            local_cnt = Counter()
            for j in range(n_w):
                local_cnt[tuple(flags_matrix[i][j] for i in range(len(sorted_chs)))] += 1
            mode_pat, mode_cnt = local_cnt.most_common(1)[0]
            mode_prob = mode_cnt / n_w
            mode_str = "(" + ",".join(str(b) for b in mode_pat) + ")"
            print(f"\n{med_name} (vocal {vocal}): Moda={mode_str} ({mode_cnt}, p={mode_prob:.2f})")

            n_canales = len(sorted_chs)
            fig, axs = plt.subplots(n_canales, 1, figsize=(12, 4*n_canales), sharex=True)
            if n_canales == 1:
                axs = [axs]
            is_dark = plt.rcParams.get('axes.facecolor', '') == 'black'
            colors_env = ["#08F7FE", "#FE53BB", "#F5D300"] if is_dark else ["Blue", "Red", "Green"]
            for idx_ax, ch in enumerate(sorted_chs):
                data = canales_data[ch]
                t_concat = data.get('t_concat', np.array([]))
                env_norm_dyn = data.get('env_corregida_concat_norm', None)
                if env_norm_dyn is None or len(t_concat) == 0:
                    continue
                boundaries = data['window_boundaries']
                flags = data['activation_flags']
                muestras_pulso = data['muestras_pulso']
                ch_id = int(ch.split('_')[-1]) if '_' in ch else idx_ax
                th_ch = [th0_opt, th1_opt, th2_opt][ch_id] if sweep_canal else th0_opt
                ax = axs[idx_ax]
                ax.plot(t_concat, env_norm_dyn, color=colors_env[idx_ax%len(colors_env)], linewidth=2, label=ch)
                if th_ch > 0:
                    ax.fill_between(t_concat, th_ch, env_norm_dyn, where=env_norm_dyn>=th_ch,
                                    color='lime', alpha=0.3, interpolate=True)
                    ax.axhline(th_ch, color='magenta', linestyle='--', linewidth=2,
                               label=f'Umbral={th_ch:.4f}')
                for b in boundaries:
                    if 0 < b < len(t_concat):
                        ax.axvline(x=t_concat[b], color='gray', linestyle=':', alpha=0.6)
                for i, st in enumerate(boundaries):
                    end = st + muestras_pulso
                    if end > len(env_norm_dyn): end = len(env_norm_dyn)
                    if st >= end: continue
                    tc = t_concat[st + (end-st)//2]
                    label = str(flags[i]) if i < len(flags) else "?"
                    ymax = np.max(env_norm_dyn[st:end])
                    ypos = max(ymax, th_ch) * 1.05 + 0.02*(ax.get_ylim()[1]-ax.get_ylim()[0])
                    ax.text(tc, ypos, label, ha='center', va='bottom', fontsize=10, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7))
                ax.set_title(f"Canal: {ch}")
                ax.set_ylabel("Amplitud norm.")
                ax.set_ylim(0, 1.05)
                ax.grid(True, alpha=0.5)
                ax.legend(loc='upper right', fontsize=8)
            axs[-1].set_xlabel("Tiempo [s] concatenado")
            plt.suptitle(f"Señal Corregida Dinámica - {med_name} (vocal {vocal})\n"
                         f"Umbrales: {threshold_norm} | Moda: {mode_str} (p={mode_prob:.2f})",
                         fontsize=14)
            plt.tight_layout(rect=[0,0.03,1,0.93])
            out_med = os.path.join(self.OUT_DIR, med_name)
            os.makedirs(out_med, exist_ok=True)
            out = os.path.join(out_med, "senal_corregida_combinada.png")
            plt.savefig(out, dpi=300, bbox_inches='tight')
            plt.close()

        if global_patterns:
            print("\n" + "="*60)
            print("RESUMEN GLOBAL POR VOCAL")
            print("="*60)
            summary_lines = []
            for v in sorted(global_patterns.keys()):
                pats = global_patterns[v]
                total = len(pats)
                cnt = Counter(pats)
                moda, moda_cnt = cnt.most_common(1)[0]
                prob = moda_cnt / total
                pat_str = "(" + ",".join(str(b) for b in moda) + ")"
                line = f"Vocal {v}: total={total}, moda={pat_str} (p={prob:.2f})"
                print(line)
                summary_lines.append(line)
                print("  Distribución completa:")
                for p, c in cnt.most_common():
                    ps = "(" + ",".join(str(b) for b in p) + ")"
                    print(f"    {ps}: {c} (p={c/total:.2f})")
                summary_lines.append("  " + "─"*30)
            messagebox.showinfo("Resumen global por vocal", "\n".join(summary_lines))

            # === Gráfico 3D del Espacio Discreto ===
            try:
                fig3d = plt.figure(figsize=(10, 8))
                ax3d = fig3d.add_subplot(111, projection='3d')
                
                r = [0, 1]
                for s, e in combinations(np.array(list(product(r, r, r))), 2):
                    if np.sum(np.abs(s-e)) == r[1]-r[0]:
                        ax3d.plot3D(*zip(s, e), color="gray", alpha=0.3, linestyle="--")
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
                for i, v in enumerate(sorted(global_patterns.keys())):
                    pats = global_patterns[v]
                    cnt = Counter(pats)
                    moda, _ = cnt.most_common(1)[0]
                    
                    jitter_x = np.random.uniform(-0.05, 0.05)
                    jitter_y = np.random.uniform(-0.05, 0.05)
                    jitter_z = np.random.uniform(-0.05, 0.05)
                    
                    ax3d.scatter(moda[0] + jitter_x, moda[1] + jitter_y, moda[2] + jitter_z, 
                                 color=colors[i % len(colors)], s=300, label=f"Vocal {v}", depthshade=True, alpha=0.8)
                    ax3d.text(moda[0] + jitter_x + 0.05, moda[1] + jitter_y + 0.05, moda[2] + jitter_z + 0.05, 
                              v, fontsize=12, fontweight='bold', color='black')

                ax3d.set_xticks([0, 1])
                ax3d.set_yticks([0, 1])
                ax3d.set_zticks([0, 1])
                ax3d.set_xlabel('Canal 0')
                ax3d.set_ylabel('Canal 1')
                ax3d.set_zlabel('Canal 2')
                ax3d.set_title('Espacio Motor Discreto (Modas por Vocal)')
                ax3d.legend()
                
                cube_path = os.path.join(self.OUT_DIR, "espacio_motor_3d.png")
                plt.savefig(cube_path, dpi=300, bbox_inches='tight')
                plt.close(fig3d)
                print(f"Gráfico 3D del espacio motor guardado en {cube_path}")
            except Exception as e:
                print(f"Error generando gráfico 3D: {e}")

        self.root.destroy()
        import sys
        sys.exit(0)

class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Análisis Trevisan v{__version__}")
        self.root.geometry("500x400")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "base_de_datos_electrodos")
        self.OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "resultados", "resultados_binarizacion")
        os.makedirs(self.OUT_DIR, exist_ok=True)
        main_frame = tk.Frame(root, padx=15, pady=15, bg="#0B0C10")
        main_frame.pack(fill="both", expand=True)
        meas_frame = tk.LabelFrame(main_frame, text="1. Seleccionar Mediciones", padx=10, pady=10, bg="#1F2833", fg="#66FCF1")
        meas_frame.pack(fill="both", expand=True, pady=(0,15))
        self.listbox_mediciones = tk.Listbox(meas_frame, selectmode=tk.EXTENDED, bg="#0B0C10", fg="#66FCF1")
        self.listbox_mediciones.pack(side="left", fill="both", expand=True)
        sb = tk.Scrollbar(meas_frame, orient="vertical", command=self.listbox_mediciones.yview)
        sb.pack(side="right", fill="y")
        self.listbox_mediciones.config(yscrollcommand=sb.set)
        self.listbox_mediciones.bind("<<ListboxSelect>>", self.on_selection_change)
        act_frame = tk.Frame(main_frame, pady=10, bg="#0B0C10")
        act_frame.pack(fill="x", side="bottom")
        self.btn_procesar = tk.Button(act_frame, text="PROCESAR DATOS TREVISAN...", command=self.open_processing_dialog,
                                      state="disabled", bg="#111111", fg="#00FF00")
        self.btn_procesar.pack(fill="x", ipady=5, pady=(0,10))
        self.cargar_mediciones()

    def cargar_mediciones(self):
        self.listbox_mediciones.delete(0, tk.END)
        if os.path.isdir(self.BASE_DIR):
            date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
            for date_folder in sorted(os.listdir(self.BASE_DIR), reverse=True):
                date_path = os.path.join(self.BASE_DIR, date_folder)
                if os.path.isdir(date_path) and date_pattern.match(date_folder):
                    for med_folder in sorted(os.listdir(date_path)):
                        med_path = os.path.join(date_path, med_folder)
                        if os.path.isdir(med_path):
                            if any(f.startswith("canal_") for f in os.listdir(med_path) if os.path.isdir(os.path.join(med_path, f))):
                                self.listbox_mediciones.insert(tk.END, os.path.join(date_folder, med_folder))

    def on_selection_change(self, event=None):
        self.btn_procesar.config(state="normal" if len(self.listbox_mediciones.curselection()) > 0 else "disabled")

    def open_processing_dialog(self):
        mediciones = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        dialog = ProcessingOptionsDialog(self.root, self.OUT_DIR)
        dialog.populate_channels(self.BASE_DIR, mediciones)

def main():
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()