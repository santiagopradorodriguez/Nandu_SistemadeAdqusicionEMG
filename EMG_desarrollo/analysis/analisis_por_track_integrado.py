# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Procesamiento, filtrado y análisis de señales por track integrado.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Procesamiento, filtrado y análisis de señales por track integrado.
# ==============================================================================

#%%
#%%
import os
import json
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hilbert, butter, filtfilt, iirnotch
from scipy import interpolate
import csv
import pandas as pd
import math
import matplotlib.pyplot as plt
import re
from datetime import datetime
import argparse
import subprocess

# --- NUEVO: Imports para el diálogo de selección de carpeta ---
# Esta es la última versión funcional conocida.
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# --- Versión del script de análisis ---
__version__ = "5.0.1"

plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 15,
})

# --- NUEVO: Función para la barra de progreso en consola ---
def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█'):
    """
    Dibuja una barra de progreso en la consola.
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print() # Nueva línea al completar

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


def _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_env="media_movil"):
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
            from scipy.signal import hilbert
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
            
    return env_full


# ---------------------- Noise estimation (initial window) -------------------
def _estimate_noise_window(signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral, tipo_env="media_movil"):
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

        # --- MODIFICACIÓN: El umbral ahora es el promedio de la ventana de ruido ---
        umbral = np.mean(env_noise) if len(env_noise) > 0 else 0.0
        noise_rms_from_noise_window = rms(env_noise) if len(env_noise) > 0 else 0.0

        print(f"[Umbral por ventana inicial] noise_seconds={noise_seconds}s, umbral (promedio)={umbral:.5e}, noise_rms_window={noise_rms_from_noise_window:.5e}")
        # Se mantiene el cálculo de sigma_est por si se usa en otro lado (ej. incertidumbre)
        return start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window
    else:
        print(f"[Umbral] no se proporcionó ventana de ruido valida (noise_seconds={noise_seconds}).")
        return start_sample_noise, np.array([]), None, None, None


# ---------------------- Detect maxima per cut & extract ---------------------
# ---------------------- Detect maxima per cut & extract (modificada) ---------------------
def _detect_maxima_and_extract(env_recortada,
                               start_sample_noise,
                               muestras_pulso,
                               pre_samples,
                               post_samples,
                               peak_search_threshold,
                               n_pulsos_manual=None,
                               min_peak_distance_factor=0.5,
                               excluded_windows=None):
    """
    Detecta un máximo por corte periódico (como antes) pero evita aceptar
    dos máximos consecutivos separados por menos de
    min_peak_distance_factor * muestras_pulso muestras.
    - Si dos máximos quedan demasiado cerca, se conserva el de mayor amplitud
      (puede reemplazar al anterior aceptado).
    Args:
      env_recortada: envolvente de la señal recortada (array).
      start_sample_noise: índice absoluto desde donde empiezan los cortes.
      muestras_pulso: tamaño (en muestras) de cada corte/periodo.
      pre_samples, post_samples: tamaño de ventana alrededor del máximo.
      peak_search_threshold: umbral mínimo para aceptar un máximo.
      min_peak_distance_factor: fracción del periodo mínima permitida entre máximos (0.5 = medio periodo).
      excluded_windows: lista de enteros con los números de las ventanas a excluir (contando desde 1).
    Returns:
      maxima_per_cut: lista de índices absolutos de máximos aceptados (en env_recortada).
      segmentos: lista de arrays (segmentos) centrados en cada máximo aceptado.
    """
    if len(env_recortada) == 0:
        return [], []

    # --- MODIFICACIÓN: El conteo de pulsos del metrónomo ahora es obligatorio ---
    if n_pulsos_manual is not None and n_pulsos_manual > 0:
        n_pulsos = int(n_pulsos_manual)
        print(f"[Análisis] Usando conteo de pulsos obligatorio del metrónomo: {n_pulsos}")
    else:
        print(f"--- ERROR: No se encontró un 'pulse_count' válido en metadata.json. El análisis requiere el conteo de pulsos del metrónomo. Omitiendo archivo. ---")
        return [], []

    maxima_per_cut = []
    segmentos = []

    # distancia minima en muestras entre picos aceptados
    min_dist_samples = max(1, int(round(min_peak_distance_factor * float(muestras_pulso))))

    # --- NUEVO: Convertir a set para búsqueda rápida ---
    excluded_set = set()
    if excluded_windows:
        excluded_set = set(excluded_windows)

    # guardamos (idx, value) del último máximo aceptado para comparar
    for i in range(n_pulsos):
        cut_start = start_sample_noise + i * muestras_pulso
        cut_end = cut_start + muestras_pulso
        # --- CORRECCIÓN: Asegurar que el último pulso se analice hasta el final de la señal ---
        if cut_end > len(env_recortada):
            cut_end = len(env_recortada) # Ajustar el final del corte al final de la señal
        
        # --- NUEVO: Omitir la ventana si está en la lista de exclusión ---
        window_number = i + 1
        if window_number in excluded_set:
            # --- NUEVO: Printear la ventana que se está omitiendo ---
            print(f"    -> Omitiendo ventana #{window_number} (excluida por el usuario).")
            continue

        local_segment = env_recortada[cut_start:cut_end]
        
        # --- SOLUCIÓN: Comprobar si el segmento está vacío antes de procesar ---
        # Esto evita el error si la grabación es más corta que el número de pulsos esperados.
        if local_segment.size == 0:
            continue # Salta al siguiente pulso si no hay datos en este segmento

        # obtener índice relativo del máximo en el corte
        rel_max = int(np.argmax(local_segment))
        max_sample = cut_start + rel_max
        max_value = env_recortada[max_sample]

        # umbral mínimo para considerar máximo
        if max_value < peak_search_threshold:
            continue

        # chequeo de encaje del segmento alrededor del máximo
        seg_start = max_sample - pre_samples
        seg_end = max_sample + post_samples
        if seg_start < 0 or seg_end > len(env_recortada):
            # segmento no cabe completamente -> omitir
            # (esto evita indices fuera de rango)
            continue

        # si el máximo está demasiado cerca del último aceptado, decidir:
        if len(maxima_per_cut) > 0:
            prev_idx = maxima_per_cut[-1]
            prev_val = env_recortada[prev_idx]
            if abs(max_sample - prev_idx) < min_dist_samples:
                # conservar el pico de mayor valor: si el actual es mayor, reemplazamos
                if max_value > prev_val:
                    # reemplazar el segmento y el índice previo
                    maxima_per_cut[-1] = int(max_sample)
                    segmentos[-1] = env_recortada[seg_start:seg_end].copy()
                # si el actual es menor o igual, lo descartamos
                continue

        # si pasa todas las comprobaciones, agregar
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


# ---------------------- Plot pulso promedio (restaurado completo) ----------
def _plot_pulse_full(
    t_pulso,
    segmentos_norm,
    pulso_promedio,
    pulso_err,
    color_prom,
    snr_manual,
    snr_uncertainty,
    noise_signal_from_fixed,
    umbral,
    calcular_umbral,
    mostrar_umbral,
    factor_umbral,
    filename,
    out_prom,
    # NUEVOS/PARAMS EXISTENTES:
    plot_mode='mean',        # 'mean' (original), 'median' (robusto), 'mean_filtered'
    individual_alpha=0.25,   # transparencia para trazas individuales (0 = invisibles)
    filter_pct_low=20,       # si plot_mode=='mean_filtered' descarta el X% más bajo RMS
    mostrar_individuales=True, # controla si se dibujan las trazas individuales
    show_plot=False,         # <-- NUEVO: para mostrar el gráfico interactivamente
    noise_drift_pct=np.nan,  # <-- NUEVO: para mostrar en el gráfico
    snr_drop_pct=np.nan,     # <-- NUEVO: para mostrar en el gráfico
    pulso_sigma=None,        # <-- NUEVO: para mostrar la desviación estándar
    noise_sigma=None         # <-- NUEVO: para mostrar el error en el umbral
):
    """
    Plot del pulso completo manteniendo el estilo original,
    pero respetando mostrar_individuales.
    """
    # decidir pulso a mostrar (no alteramos pulso_promedio original, sólo la visual)
    pulso_display = pulso_promedio
    if plot_mode == 'median':
        try:
            pulso_display = np.median(segmentos_norm, axis=0)
        except Exception:
            pulso_display = pulso_promedio
    elif plot_mode == 'mean_filtered':
        try:
            rms_per_segment = np.array([np.sqrt(np.mean(s**2)) for s in segmentos_norm])
            if len(rms_per_segment) == 0:
                # Si no hay segmentos, no se puede filtrar, usar el promedio normal
                pulso_display = pulso_promedio
                raise ValueError("No hay segmentos para filtrar")
            pct = np.percentile(rms_per_segment, filter_pct_low)
            mask_good = rms_per_segment > pct
            if np.sum(mask_good) >= 1:
                pulso_display = np.mean(segmentos_norm[mask_good], axis=0)
            else:
                pulso_display = pulso_promedio
        except Exception:
            pulso_display = pulso_promedio

    # --- NUEVO: Barra de progreso para el gráfico del pulso promedio ---
    print_progress_bar(0, 1, prefix='Cargando gráfico de pulso promedio (avg.png):', suffix='Completado', length=40)
    
    plt.figure(figsize=(12, 8))

    # --- trazas individuales (solo si mostrar_individuales=True) ---
    if mostrar_individuales and (segmentos_norm is not None) and len(segmentos_norm) > 0:
        for p in segmentos_norm:
            plt.plot(t_pulso, p, color='gray', alpha=individual_alpha, linewidth=1)

    # banda de desviación estándar
    if pulso_sigma is not None:
        plt.fill_between(t_pulso,
                         pulso_promedio - pulso_sigma,
                         pulso_promedio + pulso_sigma,
                         color='orange',
                         alpha=0.15, label="Desviación estándar (±1σ)")

    # banda de error (usa pulso_promedio y pulso_err para dejar idéntico al original)
    plt.fill_between(t_pulso,
                     pulso_promedio - pulso_err,
                     pulso_promedio + pulso_err,
                     color=color_prom if not isinstance(color_prom, str) else None,
                     alpha=0.35, label="Error del promedio (1σ/√N)")

    # curva principal
    plt.plot(t_pulso, pulso_display, color=color_prom, linewidth=2,
             label=rf"Promedio (SNR_amplitud={snr_manual:.2f}$\pm${snr_uncertainty:.2f})")

    # ruido derivado del promedio (línea discontinua roja)
    if noise_signal_from_fixed is not None:
        plt.plot(t_pulso, noise_signal_from_fixed, linestyle='--', linewidth=2, color='red', alpha=0.9,
                 label=f"Ruido")

    # umbral y sombreado
    if calcular_umbral and mostrar_umbral and (umbral is not None):
        plt.axhline(umbral, color="green", linestyle="--", alpha=0.9, label=f"Umbral ({umbral:.2f})")
        if noise_sigma is not None and not np.isnan(noise_sigma):
            plt.fill_between(t_pulso, umbral - noise_sigma, umbral + noise_sigma, color="green", alpha=0.2, label=f"Error Umbral (±1σ)")
        plt.fill_between(t_pulso, -umbral, umbral, color="red", alpha=0.06)
        porc_ruido_samples = float(100.0 * np.mean(np.abs(pulso_promedio) < umbral)) if umbral > 0 else 0.0
        plt.annotate(f"% muestras |x|<umbral: {porc_ruido_samples:.1f}%%", xy=(0.98, 0.95),
                     xycoords='axes fraction', ha='right', va='top', fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    # --- NUEVO: Mostrar métricas de fatiga en el gráfico ---
    if not np.isnan(noise_drift_pct) and not np.isnan(snr_drop_pct):
        text_fatiga = f"Deriva Ruido: {noise_drift_pct:+.1f}%\nCaída SNR: {snr_drop_pct:+.1f}%"
        plt.annotate(text_fatiga, xy=(0.02, 0.95), xycoords='axes fraction', ha='left', va='top', fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.title(f"Pulso promedio - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [µV]")
    # --- MODIFICACIÓN: Ajustar ylim al 90% por encima del máximo del pulso promedio ---
    max_y_val = np.max(pulso_promedio) if len(pulso_promedio) > 0 else 0.7
    plt.ylim(0, max_y_val * 1.9)
    plt.grid(True, alpha=0.5)
    # En lugar de plt.legend() o plt.legend(loc='best')
    plt.legend(loc='best') # O 'upper left', 'lower right', etc.
    # Guardar y actualizar la barra de progreso
    plt.savefig(out_prom, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Cargando gráfico de pulso promedio (avg.png):', suffix='Completado', length=40)
    if show_plot:
        plt.show()
    plt.close(plt.gcf()) # Cierra la figura para liberar memoria y evitar que se muestre


# ---------------------- Plot espectrograma (idéntico) ----------------------
def _plot_espectro_and_spectrogram(pulso_promedio, target_len, pre_w, post_w,
                                   espectrograma_db, frecuenciamaxima, frecuenciaminima, out_spec, filename):
    try:
        duration = (pre_w + post_w)
        if duration <= 0:
            fs_seg = 1.0
        else:
            fs_seg = float(target_len) / float(duration)

        nperseg = min(256, target_len)
        if nperseg < 4:
            nperseg = max(4, target_len)
        noverlap = int(nperseg * 0.875)
        f_s, t_s, Sxx = spectrogram(pulso_promedio, fs=fs_seg, nperseg=nperseg, noverlap=noverlap)
        if espectrograma_db:
            Sdisp = 10.0 * np.log10(Sxx + 1e-20)
        else:
            Sdisp = Sxx

        freqs = np.fft.rfftfreq(len(pulso_promedio), d=duration/float(len(pulso_promedio)))
        spec = np.abs(np.fft.rfft(pulso_promedio))
        spec_db = 20.0 * np.log10(spec / (np.max(spec) + 1e-20) + 1e-20)

        fmax_plot = min(frecuenciamaxima, fs_seg/2.0)
        fmin_plot = max(frecuenciaminima, 0.0)

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        im = axs[0].pcolormesh(t_s - pre_w, f_s, Sdisp, shading='gouraud')
        axs[0].set_ylabel('Frecuencia [Hz]')
        axs[0].set_title(f"Espectrograma del pulso promedio - {filename}")
        axs[0].set_ylim(fmin_plot, fmax_plot)
        axs[0].set_xlim(-pre_w, post_w)
        axs[0].grid(True, alpha=0.3)
        fig.colorbar(im, ax=axs[0], label='dB' if espectrograma_db else 'Power')

        mask_freq = (freqs >= fmin_plot) & (freqs <= fmax_plot)
        axs[1].plot(freqs[mask_freq], np.abs(spec_db[mask_freq]))
        axs[1].set_xlabel('Frecuencia [Hz]')
        axs[1].set_ylabel('Amplitud [dB rel.]')
        axs[1].set_title('Espectro de frecuencias del pulso promedio')
        axs[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_spec, dpi=300, bbox_inches='tight')
        plt.close(plt.gcf()) # Cierra la figura para liberar memoria y evitar que se muestre
    except Exception as e:
        print(f"No se pudo generar espectrograma/espectro para {filename}: {e}")


# ---------------------- Plot recortes (idéntico) --------------------------
def _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds,
                   start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso, out_rec, filename, 
                   excluded_windows=None, show_plot=False, signal_original_unfiltered=None, mostrar_senal_cruda=True):
    
    plt.figure(figsize=(12, 4))
    # --- NUEVO: Graficar la señal original sin filtrar para comparación ---
    if signal_original_unfiltered is not None and mostrar_senal_cruda:
        plt.plot(t_recortada, np.abs(signal_original_unfiltered), color="red", linewidth=1.0, alpha=0.4, label="Módulo original (sin filtrar)")

    # sombrear ventana inicial de ruido en violeta
    noise_t0 = t_recortada[0]
    noise_t1 = noise_t0 + noise_seconds
    plt.axvspan(noise_t0, noise_t1, color='violet', alpha=0.75, label=f"Ventana ruido ({noise_seconds}s)")

    # envolvente superpuesta
    plt.plot(t_recortada, env_recortada, color="Blue", linewidth=1.5, linestyle='-', alpha=0.9, label="Envolvente (global)")

    # líneas verticales de corte (cada periodo)
    offset_start = t_recortada[0] + float(start_sample_noise)/samplerate
    # --- CORRECCIÓN: El cálculo de n_pulsos para el gráfico debe ser idéntico al del análisis ---
    # Usamos math.ceil para asegurar que se dibuje la última ventana, incluso si es parcial.
    duracion_analizable_grafico = len(env_recortada) - start_sample_noise
    n_pulsos = math.ceil(duracion_analizable_grafico / muestras_pulso)
    for i in range(n_pulsos+1):
        xline = offset_start + i*muestras_pulso/samplerate
        plt.axvline(x=xline, color="Black", linestyle="--", alpha=0.6)

    # --- NUEVO: Preparar set de ventanas excluidas para el ploteo ---
    excluded_set_plot = set()
    if excluded_windows:
        excluded_set_plot = set(excluded_windows)
    spans = []

    for i in range(n_pulsos):
        start_t = offset_start + i*muestras_pulso/samplerate
        end_t = start_t + periodo
        window_number = i + 1
        color = "red" if window_number in excluded_set_plot else "orange"
        alpha = 0.3 if window_number in excluded_set_plot else 0.06
        span = plt.axvspan(start_t, end_t, color=color, alpha=alpha)
        spans.append((window_number, start_t, end_t, span))

    if len(maxima_per_cut) > 0:
        t_maxima = [t_recortada[idx] for idx in maxima_per_cut]
        v_max_env = [env_recortada[idx] for idx in maxima_per_cut]
        plt.scatter(t_maxima, v_max_env, color='red', s=50, zorder=5, label='Máximos (envolvente)')
    
    fig = plt.gcf()
    ax = plt.gca()
    if show_plot:
        plt.title(f"Señal original y cortes (Click para excluir/incluir) - {filename}")
    else:
        plt.title(f"Señal original y cortes periódicos - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [µV]")
    # --- MODIFICACIÓN: Ajustar ylim al 90% por encima del máximo de la envolvente ---
    max_y_val = np.max(env_recortada) if len(env_recortada) > 0 else 1.3
    plt.ylim(0, max_y_val * 1.9)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best', fontsize=8)
    
    # --- MODIFICACIÓN: Añadir mensaje de carga antes de guardar ---
    print_progress_bar(0, 1, prefix='Cargando gráfico de recortes (pulses.png):', suffix='Guardando...', length=40)
    plt.savefig(out_rec, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Cargando gráfico de recortes (pulses.png):', suffix='Completado', length=40)

    if show_plot:
        print("\nMostrando gráfico... Haz click en las ventanas para excluirlas/incluirlas interactivamente. Cierra la ventana al terminar.")
        
        def onclick(event):
            if event.inaxes != ax: return
            x = event.xdata
            for window_number, start_t, end_t, span in spans:
                if start_t <= x <= end_t:
                    if window_number in excluded_set_plot:
                        excluded_set_plot.remove(window_number)
                        span.set_color("orange")
                        span.set_alpha(0.06)
                    else:
                        excluded_set_plot.add(window_number)
                        span.set_color("red")
                        span.set_alpha(0.3)
                    fig.canvas.draw_idle()
                    break
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True) # <-- Fuerza a pausar el script hasta que cierres el gráfico
    plt.close(fig) # Cierra la figura para liberar memoria y evitar que se muestre

    return sorted(list(excluded_set_plot))


# ---------------------- Plot evolucion temporal --------------------------
def _plot_evolucion_temporal(stats_time, stats_snr, stats_snr_err, stats_snr_inst, stats_snr_inst_err, stats_noise_mean, stats_noise_err, stats_cv_snr, out_path, filename, t_start, t_end):
    if not stats_time:
        return
    
    t_arr = np.array(stats_time)
    
    # Filtrar por el rango de tiempo seleccionado
    mask = (t_arr >= t_start) & (t_arr <= t_end)
    if not np.any(mask):
        mask = np.ones_like(t_arr, dtype=bool)
        
    t_plot = t_arr[mask]
    snr_plot = np.array(stats_snr)[mask]
    snr_err_plot = np.array(stats_snr_err)[mask]
    snr_inst_plot = np.array(stats_snr_inst)[mask]
    snr_inst_err_plot = np.array(stats_snr_inst_err)[mask]
    noise_mean_plot = np.array(stats_noise_mean)[mask]
    noise_err_plot = np.array(stats_noise_err)[mask]
    cv_snr_plot = np.array(stats_cv_snr)[mask]

    print_progress_bar(0, 1, prefix='Cargando gráfico de evolución (evolucion.png):', suffix='Guardando...', length=40)
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    fig.suptitle(f"Evolución Temporal: SNR, Ruido Inter-pulso y CV - {filename}", fontsize=16)

    # Subplot Izquierdo (Arriba): SNR
    ax0.errorbar(t_plot, snr_plot, yerr=snr_err_plot, marker='o', linestyle='-', color='b', ecolor='lightblue', capsize=3, linewidth=2, label="SNR Acumulado")
    ax0.errorbar(t_plot, snr_inst_plot, yerr=snr_inst_err_plot, marker='x', linestyle='--', color='orange', ecolor='bisque', capsize=3, linewidth=2, label="SNR Interpulso Acumulado")
    ax0.set_title(f"Evolución SNR Acumulado y SNR Interpulso Acumulado")
    ax0.set_xlabel("t(s)")
    ax0.set_ylabel("SNR")
    ax0.legend(loc='best', fontsize=10)
    ax0.grid(True, alpha=0.5)

    # Subplot Derecho (Arriba): Ruido (x̄ y error)
    ax1.plot(t_plot, noise_mean_plot, marker='o', linestyle='-', color='r', label='Ruido entre Pulsos (x̄)', linewidth=2)
    ax1.fill_between(t_plot, noise_mean_plot - noise_err_plot, noise_mean_plot + noise_err_plot, color='r', alpha=0.2, label='± Error (σ/√N)')
    ax1.axhline(100.0, color='green', linestyle='--', alpha=0.7, label='Ruido Basal (100%)')
    
    if len(noise_mean_plot) > 0:
        noise_mean_valid = noise_mean_plot[~np.isnan(noise_mean_plot)]
        if len(noise_mean_valid) > 0:
            overall_mean = np.mean(noise_mean_valid)
            overall_std = np.std(noise_mean_valid)
            ax1.axhline(overall_mean, color='purple', linestyle=':', linewidth=2, label=f'Promedio Ruido Entre Pulsos  ({overall_mean:.1f}%)')
            ax1.fill_between(t_plot, overall_mean - overall_std, overall_mean + overall_std, color='purple', alpha=0.1, label=f'± 1σ Total ({overall_std:.1f}%)')

    ax1.set_title(f"Evolución Ruido Inter-pulso Normalizado")
    ax1.set_xlabel("t(s)")
    ax1.set_ylabel("Nivel de Ruido (%)")
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.5)

    # Subplot Abajo: CV SNR
    ax2.plot(t_plot, cv_snr_plot, marker='o', linestyle='-', color='purple', linewidth=2, label=r"$CV_{SNR} = \frac{\sigma(SNR_{inter})}{\mu(SNR_{inter})}$")
    ax2.set_title(f"Coeficiente de Variación del SNR Interpulso Acumulado")
    ax2.set_xlabel("t(s)")
    ax2.set_ylabel("CV SNR")
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print_progress_bar(1, 1, prefix='Cargando gráfico de evolución (evolucion.png):', suffix='Completado', length=40)


# ---------------------- Export results (nueva función) ---------------------
def export_results_for_file(out_dir, filename, resultados_entry):
    """
    Crea carpeta out_dir (si no existe) y guarda:
      - results.json con campos clave
      - mean+std arrays en pulse_mean_std.npz
      - (los PNG ya deben estar guardados en out_dir por las funciones de plotting)
    """
    os.makedirs(out_dir, exist_ok=True)
    # Guardar JSON con valores principales
    export = {}
    keys = ['mean_pulse', 'pulse_time', 'snr_mean', 'snr_per_pulse', 'snr_manual', 'amp_uncertainty',
            'snr_uncertainty', 'noise_sigma', 'noise_rms',
            'noise_rms_from_noise_window', 'umbral', 'segmentos_rs', 'snr_per_pulse']
    for k in keys:
        export[k] = resultados_entry.get(k, None)
    export['file'] = filename
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w') as fh:
        json.dump(export, fh, indent=2, default=lambda x: float(np.nan) if (isinstance(x, np.ndarray)) else x)
    
    # --- NUEVO: Guardar todos los resultados en un único archivo JSON ---
    # Esto simplifica la carga posterior para comparaciones.
    full_results_path = os.path.join(out_dir, 'analisis_results.json')
    try:
        with open(full_results_path, 'w') as f:
            # Usamos un default para convertir arrays de numpy a listas para que sea serializable
            json.dump(resultados_entry, f, indent=4, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else o)
    except Exception as e:
        print(f"Error guardando arrays para {filename}: {e}")
    print(f"Exportados resultados en {out_dir}")


#SACAR COLORES DEL NOMBRE


def detect_brand(name):
    """
    Detecta si el nombre (filename o label) comienza con '3M' o 'MT' (cualquier caso)
    Devuelve '3M', 'MT' o None.
    Acepta nombres con prefijos tipo '3m_', 'MT-', ' 3m.', etc.
    """
    if name is None:
        return None
    # extraer base sin extension y basename por si pasás rutas
    base = os.path.splitext(os.path.basename(str(name)))[0]
    # bajar a minúsculas y quitar prefijos no alfanuméricos al inicio
    base_clean = re.sub(r'^[^a-z0-9]+', '', base.lower())
    if base_clean.startswith('3m'):
        return '3M'
    if base_clean.startswith('mt'):
        return 'MT'
    return None

# ---------------------- Comparative plotting (modificada) --------------------
# ---------------------- Comparative plotting (modificada, con errores de amplitud y SNR energy robusto) ---------------------
def _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados, nombre_salida,
                       show_overlay=True,
                       show_snr=True,
                       show_amplitude=True,
                       show_table=True,
                       show_snr_time=True,
                       show_amp_time=True
                       ):
    """
    Comparative plots and table (autocontenida).
    - Añadido: barras de error en el plot de amplitud usando 'amp_uncertainty' de resultados.
    - Recalculado SNR energy usando ruido entre pulsos promediado para todas las ventanas:
        snr_energy_per_pulse = (pulse_rms**2) / (noise_rms_per_pulse**2)
        snr_energy = mean(snr_energy_per_pulse)
        snr_energy_unc = SE(snr_energy_per_pulse)
      Para esto se reconstruyen pulse_rms a partir de 'segmentos_rs' (si está disponible),
      y noise_rms_per_pulse se infiere como pulse_rms / snr_per_pulse cuando es posible.
    """
    import os
    import re
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.patches as mpatches

    def _local_rms(arr):
        return np.sqrt(np.mean(np.asarray(arr, dtype=float)**2)) if len(arr) > 0 else 0.0

    def _detect_brand_local(name):
        if name is None:
            return None
        base = os.path.splitext(os.path.basename(str(name)))[0]
        base_clean = re.sub(r'^[^a-z0-9]+', '', base.lower())
        if base_clean.startswith('3m'):
            return '3M'
        if base_clean.startswith('mt'):
            return 'MT'
        return None

    n_files = len(promedios_globales)
    if n_files == 0:
        print("No hay pulsos para comparar.")
        return
        
    # --- NUEVO: Ordenar cronológicamente por defecto ---
    try:
        from datetime import datetime
        sorted_indices = []
        for i, name in enumerate(nombres_globales):
            r = resultados.get(name, {})
            m_date = r.get('measurement_date', '')
            dt = datetime.max # Pone al final los que no tienen fecha
            if m_date:
                try:
                    dt = datetime.fromisoformat(m_date)
                except Exception:
                    pass
            sorted_indices.append((dt, i))
        sorted_indices.sort(key=lambda x: x[0])
        sorted_idx = [idx for _, idx in sorted_indices]
        
        promedios_globales = [promedios_globales[i] for i in sorted_idx]
        tiempos_globales = [tiempos_globales[i] for i in sorted_idx]
        nombres_globales = [nombres_globales[i] for i in sorted_idx]
    except Exception as e:
        print(f"Advertencia al ordenar cronológicamente: {e}")

    # --- CORRECCIÓN: Remuestrear todos los pulsos a una longitud común (la mediana) ---
    # Esto evita el error de np.vstack si los pulsos tienen longitudes diferentes,
    # lo que puede ocurrir si se comparan análisis con diferentes BPM.
    try:
        all_lengths = [len(p) for p in promedios_globales]
        if len(set(all_lengths)) > 1: # Solo remuestrear si hay longitudes diferentes
            target_len = int(np.median(all_lengths))
            print(f"Remuestreando pulsos a una longitud común de {target_len} muestras.")
            promedios_resampled = [_resample_to(np.array(p), target_len) for p in promedios_globales]
            promedios_globales = promedios_resampled # Usar los pulsos reescalados
    except Exception as e:
        print(f"Advertencia: No se pudieron remuestrear los pulsos a una longitud común. Error: {e}")

    # --- MODIFICACIÓN: Usar un colormap para asignar un color único a cada medición ---
    plot_colors = cm.viridis(np.linspace(0, 1, n_files))

    # --- Preparar matriz base para superposición (sin normalizar) ---
    pulse_matrix = np.vstack(promedios_globales)
    if isinstance(tiempos_globales, (list, tuple)) and len(tiempos_globales) > 0:
        t_plot = tiempos_globales[0]
    else:
        t_plot = np.linspace(0, 1, pulse_matrix.shape[1])

    # --- NUEVO: Barra de progreso para los gráficos comparativos ---
    num_plots = sum([show_overlay, show_snr, show_amplitude, show_table, show_snr_time, show_amp_time])
    plot_counter = 0
    print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)

    # --- FIGURA: overlay ---
    if show_overlay:
        print("Cargando... Generando gráfico de overlay de pulsos.")
        fig_ov, ax_ov = plt.subplots(figsize=(12, 5))
        for i, pulso in enumerate(pulse_matrix): # Usar la matriz sin normalizar
            ax_ov.plot(t_plot, pulso, label=str(i + 1), linewidth=2, alpha=0.9, color=plot_colors[i])
        ax_ov.set_title('Overlay de pulsos promedio')
        ax_ov.set_xlabel('Tiempo [s]')
        ax_ov.set_ylabel('Amplitud [µV]')
        ax_ov.grid(True, alpha=0.4)
        ax_ov.legend(title='Archivo #', fontsize=8, loc='upper right')
        plt.tight_layout()
        out_overlay = f"{os.path.splitext(nombre_salida)[0]}_overlay.png"
        plt.savefig(out_overlay, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_ov)
        plot_counter += 1
        print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
        print(f"Overlay guardado en: {out_overlay}")

    # ---------------- prepare SNR values and uncertainties ------------------
    rows = []
    snr_manual_vals = []
    snr_manual_uncs = []
    short_names = []
    bpms = []

    # --- NUEVO: Imprimir los BPMs de cada medición para verificación ---
    print("\n--- Resumen de Parámetros de Análisis ---")
    for name in nombres_globales:
        r = resultados.get(name, {})
        snr_manual = r.get('snr_manual', np.nan)
        snr_manual_unc = r.get('snr_uncertainty', np.nan)
        noise_drift = r.get('noise_drift_pct', np.nan)
        snr_drop = r.get('snr_drop_pct', np.nan)

        measurement_date = r.get('measurement_date', '')
        comentario = r.get('comentario', '')
        hora_str = ""
        dt_obj = None
        if measurement_date:
            try:
                dt_obj = datetime.fromisoformat(measurement_date)
                hora_str = dt_obj.strftime("%H:%M:%S")
            except Exception:
                hora_str = str(measurement_date)
                
        # Extraer amplitud máxima para este archivo
        max_amp = np.nanmax(promedios_globales[len(rows)]) if len(rows) < len(promedios_globales) else np.nan
        aunc = r.get('amp_uncertainty', 0.0)
        try:
            amp_unc = float(aunc) if aunc is not None else 0.0
        except Exception:
            amp_unc = 0.0

        # Extraer BPM desde el periodo guardado
        periodo = r.get('periodo')
        bpm_calculado = (60.0 / periodo) if (periodo and periodo > 0) else np.nan
        bpms.append(bpm_calculado)
        bpm_str = f"{bpm_calculado:.1f}" if not np.isnan(bpm_calculado) else "N/A"
        print(f"  - Archivo {len(short_names) + 1} ({os.path.splitext(name)[0]}): BPM = {bpm_str}")

        snr_manual_vals.append(np.nan if snr_manual is None else float(snr_manual))
        snr_manual_uncs.append(np.nan if snr_manual_unc is None else float(snr_manual_unc))
        short_names.append(os.path.splitext(name)[0])

        rows.append({
            'filename': name,
            'snr_manual': snr_manual if snr_manual is not None else np.nan,
            'snr_manual_unc': snr_manual_unc if snr_manual_unc is not None else np.nan,
            'hora': hora_str,
            'comentario': comentario,
            'noise_drift': noise_drift,
            'snr_drop': snr_drop,
            'dt_obj': dt_obj,
            'max_amp': max_amp,
            'amp_unc': amp_unc
        })

    snr_manual_arr = np.array(snr_manual_vals, dtype=float)
    snr_manual_unc_arr = np.array(snr_manual_uncs, dtype=float)

    x = np.arange(n_files)

    # mapping items
    mapping_items = [f"{i+1}-{short_names[i]}" for i in range(n_files)]
    max_line_len = 120
    lines = []
    cur = ""
    for item in mapping_items:
        if len(cur) + len(item) + 3 <= max_line_len:
            cur = (cur + "   " + item).strip()
        else:
            lines.append(cur)
            cur = item
    if cur:
        lines.append(cur)

    # Guardar CSV mapping
    mapping_rows = []
    for i, name in enumerate(nombres_globales):
        mapping_rows.append({'index': i + 1, 'short_name': short_names[i]})
    mapping_csv = f"{os.path.splitext(nombre_salida)[0]}_mapping_index_shortname.csv"
    with open(mapping_csv, 'w', newline='', encoding='utf-8') as mf:
        fieldnames = ['Número', 'Nombre']
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        for mr in mapping_rows:
            writer.writerow({'Número': mr['index'], 'Nombre': mr['short_name']})
    print(f"CSV mapping guardado en: {mapping_csv}")

    # Generar imagen PNG con la tabla simplificada (Número, Nombre)
    try:
        table_data_map = [[str(mr['index']), mr['short_name']] for mr in mapping_rows]
        col_labels_map = ['Número', 'Nombre']
        nrows_map = len(table_data_map)
    
        fig_map, ax_map = plt.subplots(figsize=(6, max(1.5, 0.25 * nrows_map)))
        ax_map.axis('off')
        table_map = ax_map.table(
            cellText=table_data_map,
            colLabels=col_labels_map,
            cellLoc='left',
            loc='center',
            colWidths=[0.15, 0.85]
        )
        table_map.auto_set_font_size(False)
        table_map.set_fontsize(9)
        table_map.scale(1, 1.1)
    
        out_map_png = f"{os.path.splitext(nombre_salida)[0]}_mapping_index_shortname.png"
        plt.savefig(out_map_png, dpi=300, bbox_inches='tight')
        plt.close(fig_map)
        print(f"Imagen mapping guardada en: {out_map_png}")
    except Exception as e:
        print(f"No se pudo generar la imagen de mapping: {e}")
    
# Generar tabla en formato LaTeX (Número, Nombre)
    try:
        latex_table = r"\begin{table}[H]" + "\n"
        latex_table += r"\centering" + "\n"
        latex_table += r"\caption{Asignación de número a cada configuración}" + "\n"
        latex_table += r"\label{tabla:mapping_index_shortname}" + "\n"
        latex_table += r"\begin{tabular}{|c|l|}" + "\n"
        latex_table += r"\hline" + "\n"
        latex_table += r"\textbf{Número} & \textbf{Nombre} \\ \hline" + "\n"
        for mr in mapping_rows:
            latex_table += f"{mr['index']} & {mr['short_name']} \\\\ \\hline\n"
        latex_table += r"\end{tabular}" + "\n"
        latex_table += r"\end{table}" + "\n"
    
        # Guardar en archivo .tex
        out_map_tex = f"{os.path.splitext(nombre_salida)[0]}_mapping_index_shortname.tex"
        with open(out_map_tex, "w", encoding="utf-8") as tf:
            tf.write(latex_table)
    
        print(f"Tabla LaTeX guardada en: {out_map_tex}")
    except Exception as e:
        print(f"No se pudo generar la tabla LaTeX: {e}")
        

    # -------------- Bar plot per-file: SNR manual --------------
    if show_snr:
        print("Cargando... Generando gráfico de SNR.")
        fig_snrs, ax_snrs = plt.subplots(figsize=(max(8, 0.6 * n_files), 6))
        width = 0.6

        # --- MODIFICACIÓN: Ordenar el gráfico por SNR de amplitud descendente ---
        # Reemplazar NaNs con un valor muy bajo para que no afecten el ordenamiento
        snr_for_sorting = np.nan_to_num(snr_manual_arr, nan=-np.inf)
        sort_indices_snr = np.argsort(snr_for_sorting)[::-1]

        # Reordenar todos los arrays de datos del SNR
        sorted_snr_manual = snr_manual_arr[sort_indices_snr]
        sorted_snr_manual_unc = snr_manual_unc_arr[sort_indices_snr]
        sorted_plot_colors_snr = plot_colors[sort_indices_snr]
        
        # Mantener los números originales para las etiquetas del eje X
        original_indices_snr = [np.where(np.array(nombres_globales) == nombres_globales[i])[0][0] for i in sort_indices_snr]

        for i in range(n_files):
            y_manual = sorted_snr_manual[i] if not np.isnan(sorted_snr_manual[i]) else 0.0

            yerr_manual = sorted_snr_manual_unc[i] if (not np.isnan(sorted_snr_manual_unc[i])) else None

            ax_snrs.bar(x[i], y_manual, width,
                        yerr=(yerr_manual if yerr_manual is not None else None),
                        capsize=5, alpha=0.9, color=sorted_plot_colors_snr[i]) 

        ax_snrs.set_xticks(x)
        ax_snrs.set_xticklabels([str(idx + 1) for idx in original_indices_snr], rotation=0, fontsize=10)
        ax_snrs.set_ylabel('SNR')
        
        # --- MODIFICACIÓN: Ajustar el límite Y al máximo SNR + 10% de margen ---
        # Se consideran los valores de SNR y sus incertidumbres para el cálculo del máximo.
        max_snr_manual = np.nanmax(snr_manual_arr + np.nan_to_num(snr_manual_unc_arr)) if len(snr_manual_arr) > 0 else 0
        ax_snrs.set_ylim(0, max_snr_manual * 1.1 if max_snr_manual > 0 else 10)

        ax_snrs.set_title('SNR: Amplitud (ordenado por SNR Amplitud)')
        ax_snrs.grid(True, axis='y', alpha=0.5)

        plt.tight_layout(rect=[0, 0.06 - 0.04*len(lines), 1, 1])
        out_snrs_grouped = f"{os.path.splitext(nombre_salida)[0]}_snr_grouped.png"
        plt.savefig(out_snrs_grouped, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_snrs)
        plot_counter += 1
        print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
        print(f"Gráfico SNR agrupado guardado en: {out_snrs_grouped}")

    # -------------- Bar plot of max amplitude of mean pulse (colored by brand) --------------
    if show_amplitude:
        print("Cargando... Generando gráfico de amplitud máxima.")
        # --- MODIFICACIÓN: Ordenar el gráfico de amplitud de forma descendente ---
        max_amplitudes = np.array([np.nanmax(p) for p in promedios_globales])
        
        # Crear un índice de ordenamiento
        sort_indices = np.argsort(max_amplitudes)[::-1]

        # Reordenar todos los datos según el índice de amplitud
        sorted_amplitudes = max_amplitudes[sort_indices]
        sorted_names = [nombres_globales[i] for i in sort_indices]
        sorted_short_names = [short_names[i] for i in sort_indices]
        sorted_plot_colors = plot_colors[sort_indices]
        # El número original (antes de ordenar) se mantiene para consistencia con otras tablas
        original_indices = [np.where(np.array(nombres_globales) == name)[0][0] for name in sorted_names]

        # recoger incertidumbres de amplitud (si existen) para errorbars
        all_amp_uncs = []
        for name in nombres_globales:
            r = resultados.get(name, {})
            aunc = r.get('amp_uncertainty', None)
            if aunc is None:
                all_amp_uncs.append(0.0)
            else:
                try:
                    all_amp_uncs.append(float(aunc))
                except Exception:
                    all_amp_uncs.append(0.0)
        all_amp_uncs = np.array(all_amp_uncs, dtype=float)
        sorted_amp_uncs = all_amp_uncs[sort_indices]

        # --- NUEVO: Imprimir los valores de amplitud máxima en la consola ---
        print("\n--- Amplitud Máxima de Pulso Promedio (ordenado de mayor a menor) ---")
        for i in range(n_files):
            nombre = sorted_short_names[i]
            amplitud = sorted_amplitudes[i]
            incertidumbre = sorted_amp_uncs[i]
            print(f"  - Archivo #{original_indices[i] + 1} ({nombre}): {amplitud:.4f} ± {incertidumbre:.4f} µV")
        print("------------------------------------------------------------------\n")

        fig_amp, ax_amp = plt.subplots(figsize=(max(8, 0.6 * n_files), 6))
        # Usar los datos ordenados para graficar
        bars_amp = ax_amp.bar(x, sorted_amplitudes, yerr=sorted_amp_uncs, capsize=5, alpha=0.85, color=sorted_plot_colors)

        ax_amp.set_xticks(x)
        ax_amp.set_xticklabels([str(i + 1) for i in original_indices], rotation=0, fontsize=10)
        ax_amp.set_ylabel('Amplitud [µV]')
        ax_amp.set_title('Amplitud máxima del pulso promedio (ordenado)')
        ax_amp.grid(True, axis='y', alpha=0.3)

        # mostrar valores arriba de cada barra (con incertidumbre si existe)
        for i, bar in enumerate(bars_amp):
            height = bar.get_height()
            unc = sorted_amp_uncs[i]
            if not np.isnan(height):
                if unc and unc > 0:
                    label = f"{height:.2f} ± {unc:.2f}"
                else:
                    label = f"{height:.2f}"
                ax_amp.text(bar.get_x() + bar.get_width() / 2.0, height, label, ha='center', va='bottom', fontsize=9)

        # --- MODIFICACIÓN: La leyenda ahora es la de los números de archivo ---
        # La leyenda de colores ya no es necesaria porque cada barra tiene un color único
        # y se corresponde con el número en el eje X.
        if n_files > 0:
            pass # La leyenda ya no es necesaria aquí.

        plt.tight_layout()
        out_amp_bar = f"{os.path.splitext(nombre_salida)[0]}_amplitud_max_bar.png"
        plt.savefig(out_amp_bar, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig_amp)
        plot_counter += 1
        print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
        print(f"Amplitud máxima guardada en: {out_amp_bar}")

    # -------------------- CSV + PNG table with "value ± uncertainty" in same cell --------------------
    if show_table:
        print("Cargando... Generando tablas de resultados (CSV y PNG).")
        rows_sorted = sorted(rows, key=lambda r: (-(r['snr_manual']) if (r['snr_manual'] is not None and not np.isnan(r['snr_manual'])) else float('inf')))

        csv_path = f"{os.path.splitext(nombre_salida)[0]}_snr_table.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['filename', 'Hora', 'Comentario', 'SNR_manual ± unc', 'Deriva Ruido (%)', 'Caida SNR (%)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows_sorted:
                if r['snr_manual'] is None or np.isnan(r['snr_manual']):
                    manual_str = ""
                else:
                    if r['snr_manual_unc'] is None or np.isnan(r['snr_manual_unc']):
                        manual_str = f"{r['snr_manual']:.6g}"
                    else:
                        manual_str = f"{r['snr_manual']:.6g} ± {r['snr_manual_unc']:.6g}"
                writer.writerow({
                    'filename': r['filename'],
                    'Hora': r['hora'],
                    'Comentario': r['comentario'],
                    'SNR_manual ± unc': manual_str,
                    'Deriva Ruido (%)': f"{r['noise_drift']:.1f}%" if not np.isnan(r['noise_drift']) else "",
                    'Caida SNR (%)': f"{r['snr_drop']:.1f}%" if not np.isnan(r['snr_drop']) else ""
                })
        print(f"Tabla CSV guardada en: {csv_path}")

        # Tabla PNG
        try:
            table_data = []
            for r in rows_sorted:
                fname_noext = os.path.splitext(r['filename'])[0] if isinstance(r['filename'], str) else r['filename']
                if r['snr_manual'] is None or np.isnan(r['snr_manual']):
                    manual_cell = ""
                else:
                    if r['snr_manual_unc'] is None or np.isnan(r['snr_manual_unc']):
                        manual_cell = f"{r['snr_manual']:.3f}"
                    else:
                        manual_cell = f"{r['snr_manual']:.3f} ± {r['snr_manual_unc']:.3f}"
                        
                drift_cell = f"{r['noise_drift']:.1f}%" if not np.isnan(r['noise_drift']) else ""
                drop_cell = f"{r['snr_drop']:.1f}%" if not np.isnan(r['snr_drop']) else ""
                
                comentario_corto = r['comentario'][:25] + ('...' if len(r['comentario']) > 25 else '')
                table_data.append([fname_noext, r['hora'], comentario_corto, manual_cell, drift_cell, drop_cell])

            col_labels = ['Filename', 'Hora', 'Comentario', 'SNR ± unc', 'Deriva Ruido', 'Fatiga(SNR)']
            nrows = len(table_data)
            fig_tab, ax_tab = plt.subplots(figsize=(16, max(2, 0.35 * nrows)))
            ax_tab.axis('off')
            table = ax_tab.table(cellText=table_data, colLabels=col_labels, cellLoc='left', loc='center',
                                 colWidths=[0.25, 0.08, 0.27, 0.15, 0.12, 0.13])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.2)
            out_table_png = f"{os.path.splitext(nombre_salida)[0]}_snr_table.png"
            plt.title("Ñandú LSD - SNR ")
            plt.savefig(out_table_png, dpi=300, bbox_inches='tight')
            plot_counter += 1
            print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
            plt.close(fig_tab)
            print(f"Imagen de tabla guardada en: {out_table_png}")
        except Exception as e:
            print(f"No se pudo generar la imagen de la tabla: {e}")

        # console summary
        print("\nResumen (ordenado por SNR manual):")
        for r in rows_sorted:
            man = ("" if (r['snr_manual'] is None or np.isnan(r['snr_manual'])) else f"{r['snr_manual']:.3f}")
            man_unc = ("" if (r['snr_manual_unc'] is None or np.isnan(r['snr_manual_unc'])) else f"{r['snr_manual_unc']:.3f}")
            combined_man = f"{man} ± {man_unc}" if man_unc != "" else man
            
            drift_str = f"{r['noise_drift']:+.1f}%" if not np.isnan(r['noise_drift']) else "N/A"
            drop_str = f"{r['snr_drop']:+.1f}%" if not np.isnan(r['snr_drop']) else "N/A"
            print(f"{r['filename']}: SNR={combined_man} | Deriva Ruido={drift_str} | Caída SNR={drop_str}")

    # -------------- SNR vs Time Plot --------------
    if show_snr_time:
        print("Cargando... Generando gráfico de SNR vs Tiempo.")
        valid_rows = [(i, r) for i, r in enumerate(rows) if r['dt_obj'] is not None and not np.isnan(r['snr_manual'])]
        if len(valid_rows) > 1:
            times_dt = [r['dt_obj'] for i, r in valid_rows]
            t0 = min(times_dt)
            rel_times = [(t - t0).total_seconds() / 60.0 for t in times_dt]
            snrs = [r['snr_manual'] for i, r in valid_rows]
            snrs_unc = [r['snr_manual_unc'] for i, r in valid_rows]
            
            fig_time, ax_time = plt.subplots(figsize=(10, 5))
            yerrs = [u if not np.isnan(u) else 0.0 for u in snrs_unc]
            
            ax_time.errorbar(rel_times, snrs, yerr=yerrs, fmt='-o', capsize=5, markersize=8, color='darkorange', ecolor='red', linewidth=2)
            
            ax_time.set_xlabel('Tiempo desde inicio (minutos)')
            ax_time.set_ylabel('SNR Amplitud')
            ax_time.set_title('Evolución del SNR a lo largo de la sesión')
            ax_time.grid(True, alpha=0.5)
            
            for index_in_valid, (original_i, r) in enumerate(valid_rows):
                ax_time.annotate(str(original_i + 1), (rel_times[index_in_valid], snrs[index_in_valid]), 
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
                
            plt.tight_layout()
            out_snr_time = f"{os.path.splitext(nombre_salida)[0]}_snr_vs_tiempo.png"
            plt.savefig(out_snr_time, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_time)
            
        plot_counter += 1
        print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
        if len(valid_rows) > 1: print(f"Gráfico SNR vs Tiempo guardado en: {out_snr_time}")
        else: print("No hay suficientes datos con hora válida para graficar SNR vs Tiempo.")

    # -------------- Amplitude vs Time Plot --------------
    if show_amp_time:
        print("Cargando... Generando gráfico de Amplitud vs Tiempo.")
        valid_rows = [(i, r) for i, r in enumerate(rows) if r['dt_obj'] is not None and not np.isnan(r['max_amp'])]
        if len(valid_rows) > 1:
            times_dt = [r['dt_obj'] for i, r in valid_rows]
            t0 = min(times_dt)
            rel_times = [(t - t0).total_seconds() / 60.0 for t in times_dt]
            amps = [r['max_amp'] for i, r in valid_rows]
            amps_unc = [r['amp_unc'] for i, r in valid_rows]
            
            fig_amp_time, ax_amp_time = plt.subplots(figsize=(10, 5))
            yerrs_amp = [u if not np.isnan(u) else 0.0 for u in amps_unc]
            
            ax_amp_time.errorbar(rel_times, amps, yerr=yerrs_amp, fmt='-o', capsize=5, markersize=8, color='purple', ecolor='black', linewidth=2)
            
            ax_amp_time.set_xlabel('Tiempo desde inicio (minutos)')
            ax_amp_time.set_ylabel('Amplitud máxima [µV]')
            ax_amp_time.set_title('Evolución de la Amplitud a lo largo de la sesión')
            ax_amp_time.grid(True, alpha=0.5)
            
            for index_in_valid, (original_i, r) in enumerate(valid_rows):
                ax_amp_time.annotate(str(original_i + 1), (rel_times[index_in_valid], amps[index_in_valid]), 
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
                
            plt.tight_layout()
            out_amp_time = f"{os.path.splitext(nombre_salida)[0]}_amplitud_vs_tiempo.png"
            plt.savefig(out_amp_time, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close(fig_amp_time)
            
        plot_counter += 1
        print_progress_bar(plot_counter, num_plots, prefix='Generando Gráficos Comparativos:', suffix='Completado', length=50)
        if len(valid_rows) > 1: print(f"Gráfico Amplitud vs Tiempo guardado en: {out_amp_time}")
        else: print("No hay suficientes datos con hora válida para graficar Amplitud vs Tiempo.")

# ---------------------- Main function (misma firma y lógica) ----------------
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
    mostrar_tabla=True,
    usar_picos=True,
    peak_prominence=None,
    peak_height=None,
    peak_distance_sec=0.4,
    pre_window_sec=None,
    post_window_sec=None,
    normalize_by='rms',
    resample_len=None,
    one_max_per_cut=True,
    n_pulsos_manual=None,
    # ADICIONES
    fixed_umbral_abs=0.5,    # umbral fijo ABSOLUTO para comparar con el pulso promedio
    apply_envelope=True,     # calcula envolvente sobre la señal completa antes de recortar
    smooth_ms=50,             # suavizado por media móvil de la envolvente en ms (0 = sin suavizado)
    tipo_envolvente="media_movil",
    # NUEVAS OPCIONES (ruido inicial)
    noise_seconds=2,         # primeros segundos (relativos al inicio de la señal recortada) a usar como ruido
    excluded_windows=None,   # Lista de ventanas a excluir
    peak_search_threshold=0.25,  # umbral mínimo en la envolvente para aceptar un máximo en la búsqueda por cortes
    # NUEVOS ARGUMENTOS PARA PLOTTING (por defecto como tú lo tenías)
    plot_mode='mean',         # 'mean'|'median'|'mean_filtered' (por defecto 'mean' = comport. original)
    individual_alpha=0.25,    # opacidad por defecto igual a la que tenías
    lowpass_cutoff_hz=500.0,  # <-- NUEVO: Frecuencia de corte para filtro pasa-bajos
    highpass_cutoff_hz=20.0,  # <-- NUEVO: Frecuencia de corte para filtro pasa-altos
    output_root="/home/santiago/Documentos/codigos/Labo 6",          # si se provee, todas las carpetas de resultados se crean dentro de esta raíz
    display_name_for_plot="", # <-- Argumento que faltaba
    show_interactive_plot=False, # <-- para mostrar el gráfico de recortes
    show_average_plot=False,     # <-- NUEVO: para mostrar el gráfico de pulso promedio
    apply_notch_filter=False,    # <-- NUEVO: para controlar el filtro notch
    mostrar_evolucion=False,
    evol_t_start=25.0,
    evol_t_end=100.0,
    mostrar_senal_cruda=True
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
    
    # Usar el display_name si se proporciona, si no, usar el nombre del archivo
    plot_title_name = display_name_for_plot

    for filename in archivos:
        filepath = os.path.join(carpeta, filename)
        
        # --- NUEVO: Calibración de la señal usando el archivo CSV ---
        calibration_factor = 1.0
        try:
            # 1. Encontrar el archivo CSV en la carpeta padre
            parent_dir = os.path.dirname(carpeta)
            csv_files = [f for f in os.listdir(parent_dir) if f.lower().endswith('.csv')]
            
            if not csv_files:
                raise FileNotFoundError("No se encontró archivo CSV en la carpeta padre.")
            
            csv_path = os.path.join(parent_dir, csv_files[0])
            
            # 2. Leer el CSV y encontrar el factor de calibración
            df_csv = pd.read_csv(csv_path)
            
            # Extraer el número de canal de la carpeta (ej: 'canal_0' -> 0)
            channel_num_str = os.path.basename(carpeta).split('_')[-1]
            channel_idx = int(channel_num_str)
            channel_col_name = f"Canal {channel_idx}"
            
            if channel_col_name not in df_csv.columns:
                raise ValueError(f"La columna '{channel_col_name}' no se encontró en '{csv_path}'.")
            
            # El factor de calibración es el máximo absoluto del voltaje original
            calibration_factor = np.max(np.abs(df_csv[channel_col_name].values))
            print(f"[Calibración] Usando CSV '{csv_files[0]}'. Factor para {channel_col_name}: {calibration_factor:.4f} V")

        except Exception as e:
            print(f"ADVERTENCIA: No se pudo calibrar la señal. Se usará amplitud normalizada. Error: {e}")
            calibration_factor = 1.0

        signal_normalized, samplerate = _read_wav_mono(filepath)
        signal_v = signal_normalized * calibration_factor # Aplicar calibración para volver a Voltios
        
        # --- NUEVO: Calibración a microvoltios usando metadata ---
        resistencia_ohm = 100.0 # Por defecto
        measurement_date = ""
        comentario = ""
        meta_path = os.path.join(carpeta, 'metadata.json')
        try:
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta_data = json.load(f)
                    if 'resistencia_ohm' in meta_data:
                        resistencia_ohm = meta_data['resistencia_ohm']
                    measurement_date = meta_data.get('measurement_date', '')
                    comentario = meta_data.get('comentario', '')
        except Exception as e:
            pass
            
        r_fija = 49400.0
        ganancia = 1.0 + (r_fija / resistencia_ohm)
        signal = (signal_v / ganancia) * 1e6
        print(f"[Calibración a uV] Resistencia: {resistencia_ohm} Ohms | Ganancia: {ganancia:.2f} | Señal convertida a microvoltios (uV).")

        # --- NUEVO: Filtro Pasa-Altos ---
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                cutoff_hp = highpass_cutoff_hz
                if cutoff_hp >= nyquist:
                    cutoff_hp = nyquist * 0.99
                    print(f"ADVERTENCIA: Frecuencia pasa-altos excede Nyquist. Ajustando a {cutoff_hp:.2f} Hz.")
                
                b, a = butter(4, cutoff_hp / nyquist, btype='high', analog=False)
                signal = filtfilt(b, a, signal)
                print(f"[Filtro] Aplicado filtro pasa-altos a {cutoff_hp:.2f} Hz.")
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo aplicar el filtro pasa-altos. Error: {e}")
        
        # --- MODIFICADO: Filtro Notch para 50 Hz (condicional) ---
        if apply_notch_filter:
            try:
                f0 = 50.0  # Frecuencia a remover
                Q = 2.0    # Factor de calidad (Quality factor)
                b, a = iirnotch(f0, Q, samplerate)
                signal = filtfilt(b, a, signal) # Aplicar filtro sin desfase
                print(f"[Filtro] Aplicado filtro notch a {f0} Hz.")
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo aplicar el filtro notch. Error: {e}")
        else:
            print("[Filtro] Filtro notch de 50 Hz desactivado por el usuario.")

        # --- NUEVO: Guardar una copia de la señal original antes de filtrar ---
        signal_unfiltered = signal.copy()

        # --- NUEVO: Filtro Low-pass opcional ---
        if lowpass_cutoff_hz is not None and lowpass_cutoff_hz > 0:
            try:
                # Diseñar el filtro Butterworth de 4to orden
                nyquist = 0.5 * samplerate
                cutoff_usar = lowpass_cutoff_hz
                
                if cutoff_usar >= nyquist:
                    cutoff_usar = nyquist * 0.99
                    print(f"ADVERTENCIA: Frecuencia de corte pasa-bajos ({lowpass_cutoff_hz} Hz) excede o iguala Nyquist ({nyquist} Hz). Ajustando a {cutoff_usar:.2f} Hz para aplicar el filtro solicitado.")
                
                normal_cutoff = cutoff_usar / nyquist
                b, a = butter(4, normal_cutoff, btype='low', analog=False)
                signal = filtfilt(b, a, signal) # Aplicar filtro sin desfase
                print(f"[Filtro] Aplicado filtro pasa-bajos a {cutoff_usar:.2f} Hz.")
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo aplicar el filtro pasa-bajos. Error: {e}")
        
        # --- CORRECCIÓN: Usar la duración total de la señal para el recorte ---
        duracion_total_signal = len(signal) / samplerate

        # Si no se pasó un nombre para el título, usar el del archivo
        final_plot_title = plot_title_name or filename

        # tomar módulo para señales bipolares (si corresponde)
        signal_abs = np.abs(signal)

        # ---------- Envolvente calculada sobre la señal completa (antes de recortar) ----------
        env_full = _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_envolvente)

        t = np.linspace(0, len(signal)/samplerate, len(signal), endpoint=False)

        # recortar la señal ORIGINAL (para graficar) y la envolvente ya calculada (para deteccion)
        mask = (t >= tiempoinicial) & (t <= duracion_total_signal)
        signal_recortada = signal[mask]
        t_recortada = t[mask]
        env_recortada = env_full[mask]

        if len(signal_recortada) == 0:
            print(f"{filename}: no hay muestras en [{tiempoinicial},{tiempofinal}] s. Omitido.")
            continue

        # decidir pre/post window
        if pre_window_sec is None:
            pre_w = 0.4 * periodo
        else:
            pre_w = pre_window_sec
        if post_window_sec is None:
            post_w = 0.6 * periodo
        else:
            post_w = post_window_sec
        pre_samples = int(round(pre_w * samplerate))
        post_samples = int(round(post_w * samplerate))

        # calculo del numero de cortes periodicos en la ventana recortada (usando env_recortada)
        muestras_pulso = int(round(periodo * samplerate))
        if muestras_pulso <= 0:
            print("Periodo demasiado corto o samplerate demasiado bajo.")
            continue

        # --- NUEVO: estimar ruido a partir de los primeros `noise_seconds` de la señal recortada ---
        start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window = _estimate_noise_window(signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral, tipo_envolvente)
        if start_sample_noise <= 0:
            start_sample_noise = 0

        # ahora construiremos las ventanas de corte empezando DESPUES de la ventana de ruido
        env_for_cuts = env_recortada[start_sample_noise:]
        if len(env_for_cuts) == 0:
            print(f"{filename}: no queda señal despues de la ventana de ruido para buscar pulsos.")
            continue

        n_pulsos = len(env_for_cuts) // muestras_pulso
        if n_pulsos == 0:
            print(f"{filename}: señal demasiado corta para un pulso completo (periodo en muestras={muestras_pulso}) despues de la ventana de ruido.")
            continue

        # --- CORRECCIÓN: Usar un umbral de búsqueda de picos variable ---
        # En lugar de usar el 'peak_search_threshold' fijo, usamos el 'umbral'
        # --- CORRECCIÓN: Usar un umbral de búsqueda de picos variable ---
        base_threshold = umbral if umbral is not None and umbral > 0 else peak_search_threshold
        
        # Primera pasada: promediar picos que superan el umbral base
        first_pass_peaks = []
        for i in range(n_pulsos):
            c_start = start_sample_noise + i * muestras_pulso
            c_end = min(c_start + muestras_pulso, len(env_recortada))
            if c_start < len(env_recortada):
                seg = env_recortada[c_start:c_end]
                if seg.size > 0:
                    m_val = np.max(seg)
                    if m_val >= base_threshold:
                        first_pass_peaks.append(m_val)
                        
        if len(first_pass_peaks) > 0:
            search_threshold_dinamico = np.mean(first_pass_peaks) * 0.5
        else:
            search_threshold_dinamico = base_threshold
            
        print(f"[Análisis] Usando umbral de búsqueda de picos dinámico: {search_threshold_dinamico:.4f}")

        # listas
        maxima_per_cut, segmentos = _detect_maxima_and_extract(np.abs(env_recortada), start_sample_noise, muestras_pulso, pre_samples, post_samples, search_threshold_dinamico, n_pulsos_manual=n_pulsos_manual, excluded_windows=excluded_windows)

        if len(segmentos) == 0:
            print(f"{filename}: no se extrajeron segmentos centrados en máximos por corte (umbral dinámico={search_threshold_dinamico:.4f}). Omitido.")
            continue

        # remuestrear segmentos a la misma longitud si hace falta
        segmentos_rs, target_len = _resample_segments(segmentos, resample_len)

        # calcular estadísticos del pulso
        segmentos_norm, pulso_promedio, pulso_sigma, pulso_err, Np = _compute_pulse_stats(segmentos_rs)

        # ---------- Calculo del umbral ya realizado arriba (por ventana de ruido) ----------
        if (sigma_est is None) or (umbral is None):
            sigma_est_fb, umbral_fb = _fallback_umbral(segmentos_norm, pulso_promedio, factor_umbral)
            sigma_est = sigma_est_fb
            umbral = umbral_fb
            print(f"[Umbral fallback] {filename}: sigma_est={sigma_est:.5e}, umbral={umbral:.5e}")

        # -----------------------------
        # Estimación de noise_rms: preferimos el valor desde la ventana de ruido si existe
        noise_rms = None
        if noise_rms_from_noise_window is not None and noise_rms_from_noise_window > 0:
            noise_rms = noise_rms_from_noise_window
            print(f"[noise est.] {filename}: usando ventana inicial de ruido: noise_rms={noise_rms:.5e}")
        else:
            L = len(t_recortada)
            mask_outside = np.ones(L, dtype=bool)
            for max_idx in maxima_per_cut:
                start = max(0, int(max_idx) - pre_samples)
                end = min(L, int(max_idx) + post_samples)
                mask_outside[start:end] = False
            idx_out = np.where(mask_outside)[0]
            if len(idx_out) >= max(10, int(0.01 * L)):
                residuos_out = signal_recortada[idx_out].ravel()
                noise_rms = rms(np.abs(residuos_out))
                print(f"[noise est.] {filename}: outside_windows fallback: n_pts={residuos_out.size}, noise_rms={noise_rms:.5e}")
            else:
                noise_rms = float(sigma_est) if sigma_est is not None and sigma_est > 0 else 1e-12
                print(f"[noise est.] {filename}: fallback a sigma_est: noise_rms={noise_rms:.5e}")

        # ---------- Calculo SNR ----------
        # ---------- Calculo SNR y Ruido Interpulso Normalizado ----------
        # Calcular ruido base (ahora usando la envolvente)
        if start_sample_noise > 0:
            skip_samples = int(round(1.0 * samplerate))
            if start_sample_noise <= skip_samples + int(0.1 * samplerate):
                skip_samples = min(int(round(0.1 * samplerate)), start_sample_noise // 2)
                
            initial_noise_segment = env_recortada[skip_samples:start_sample_noise]
            initial_noise_mean = np.mean(np.abs(initial_noise_segment))
            initial_noise_std = np.std(initial_noise_segment)
        else:
            initial_noise_mean = 1e-12
            initial_noise_std = 1e-12
            
        if initial_noise_mean <= 0: initial_noise_mean = 1e-12
        if initial_noise_std <= 0: initial_noise_std = 1e-12

        # inicializar vectores
        snr_per_pulse = np.zeros(len(segmentos_rs))
        noise_rms_per_pulse = np.zeros(len(segmentos_rs))

        # Para la gráfica de evolución:
        stats_time = []
        stats_noise_mean = []
        stats_noise_err = []
        stats_snr = []
        stats_snr_err = []
        stats_snr_inst = []
        stats_snr_inst_err = []
        stats_cv_snr = []
        
        stats_peak_vals = []
        stats_raw_noise_vals = []

        # longitud (en muestras) de la ventana local de ruido (1/4 del periodo, +/- tau/8)
        noise_win_samples = max(3, int(round((periodo / 4.0) * samplerate)))
        
        valid_idx = 0
        n_pulsos_total = int(n_pulsos_manual) if n_pulsos_manual is not None and n_pulsos_manual > 0 else n_pulsos
        excluded_set_local = set(excluded_windows) if excluded_windows else set()

        for i in range(n_pulsos_total):
            window_number = i + 1
            if window_number in excluded_set_local:
                # Ventana excluida: omitimos insertar datos y matplotlib unirá los puntos válidos con una línea
                continue
                
            if valid_idx >= len(segmentos_rs):
                continue
                
            max_idx = maxima_per_cut[valid_idx]
            
            seg_start = int(max_idx) - pre_samples
            
            # --- MODIFICACIÓN: Buscar el valle real (punto medio entre timestamps del metrónomo) ---
            cut_start = start_sample_noise + i * muestras_pulso
            midpoint = cut_start + (muestras_pulso // 2)
                
            noise_start = max(0, int(midpoint - noise_win_samples // 2))
            noise_end = min(len(env_recortada), noise_start + noise_win_samples)
            
            t_pulso_abs = t_recortada[int(max_idx)]
            stats_time.append(t_pulso_abs)
            
            if noise_end - noise_start >= 3:
                noise_segment_sig = env_recortada[noise_start:noise_end]
                
                # --- NUEVO: Remover outliers (picos no detectados) del cálculo de ruido inter-pulso ---
                abs_noise = np.abs(noise_segment_sig)
                q1 = np.percentile(abs_noise, 25)
                q3 = np.percentile(abs_noise, 75)
                iqr = q3 - q1
                upper_bound = q3 + 1.5 * iqr
                
                valid_noise = abs_noise[abs_noise <= upper_bound]
                if len(valid_noise) < 3:
                    valid_noise = abs_noise # Fallback

                curr_mean_temp = np.mean(valid_noise)
                
                # --- NUEVO: Filtro de outliers del 300% ---
                if initial_noise_mean > 0 and (curr_mean_temp / initial_noise_mean) > 3.0:
                    curr_mean = np.nan
                    curr_std = np.nan
                    curr_err = np.nan
                    noise_rms_per_pulse[valid_idx] = np.nan
                else:
                    curr_mean = curr_mean_temp
                    curr_std = np.std(valid_noise, ddof=1) if len(valid_noise) > 1 else 0.0
                    curr_err = curr_std / np.sqrt(len(valid_noise)) if len(valid_noise) > 0 else 0.0
                    noise_rms_per_pulse[valid_idx] = rms(valid_noise)
                
                # Normalizado (porcentaje respecto al inicial)
                stats_noise_mean.append((curr_mean / initial_noise_mean) * 100.0 if not np.isnan(curr_mean) else np.nan)
                stats_noise_err.append((curr_err / initial_noise_mean) * 100.0 if not np.isnan(curr_err) else np.nan)
                
            else:
                curr_mean = initial_noise_mean if initial_noise_mean > 0 else 1e-12
                curr_err = initial_noise_std / np.sqrt(3) if initial_noise_std > 0 else 0.0
                stats_noise_mean.append(100.0)
                stats_noise_err.append(0.0)
                noise_rms_per_pulse[valid_idx] = noise_rms if noise_rms is not None else 1e-12
                
            # SNR como en unificador: pico del pulso / ruido base
            # Calculamos el pico máximo sobre la envolvente
            seg_end = int(max_idx) + post_samples
            env_segment = env_recortada[max(0, seg_start):min(len(env_recortada), seg_end)]
            peak_val = np.max(np.abs(env_segment)) if len(env_segment) > 0 else 0.0
            
            curr_snr = peak_val / initial_noise_mean
            snr_per_pulse[valid_idx] = curr_snr
            
            # --- NUEVO: SNR Interpulso Acumulado ---
            if curr_mean > 0 and not np.isnan(curr_mean):
                stats_peak_vals.append(peak_val)
                stats_raw_noise_vals.append(curr_mean)
                
                avg_peak_accum = np.mean(stats_peak_vals)
                avg_noise_accum = np.mean(stats_raw_noise_vals)
                
                curr_snr_inter_acum = avg_peak_accum / avg_noise_accum
                
                if len(stats_peak_vals) > 1:
                    err_peak = np.std(stats_peak_vals, ddof=1) / np.sqrt(len(stats_peak_vals))
                    err_noise = np.std(stats_raw_noise_vals, ddof=1) / np.sqrt(len(stats_raw_noise_vals))
                else:
                    err_peak = 0.0
                    err_noise = 0.0
                
                if avg_peak_accum > 0 and avg_noise_accum > 0:
                    curr_snr_inter_acum_err = curr_snr_inter_acum * np.sqrt((err_peak / avg_peak_accum)**2 + (err_noise / avg_noise_accum)**2)
                else:
                    curr_snr_inter_acum_err = 0.0
            else:
                curr_snr_inter_acum = np.nan
                curr_snr_inter_acum_err = 0.0
                
            stats_snr_inst.append(curr_snr_inter_acum)
            stats_snr_inst_err.append(curr_snr_inter_acum_err)

            # --- NUEVO: CV SNR ---
            current_snrs_inst = [val for val in stats_snr_inst[:valid_idx+1] if not np.isnan(val)]
            if len(current_snrs_inst) > 1:
                mu_snr_inst = np.mean(current_snrs_inst)
                std_snr_inst = np.std(current_snrs_inst, ddof=1)
                cv_snr = std_snr_inst / mu_snr_inst if mu_snr_inst > 0 else 0.0
            else:
                cv_snr = 0.0
            stats_cv_snr.append(cv_snr)

            # Calcular SNR acumulado y su error estándar
            current_snrs = snr_per_pulse[:valid_idx+1]
            snr_acumulado = np.mean(current_snrs)
            snr_err = np.std(current_snrs, ddof=1) / np.sqrt(len(current_snrs)) if len(current_snrs) > 1 else 0.0

            stats_snr.append(snr_acumulado)
            stats_snr_err.append(snr_err)
            
            valid_idx += 1
                
        snr_mean = np.mean(snr_per_pulse)
        snr_db = np.nan

        # --- NUEVO: Calcular Deriva de Ruido y Caída de SNR (Evaluación de fatiga y cable) ---
        valid_snr = snr_per_pulse[:valid_idx]
        valid_noise = noise_rms_per_pulse[:valid_idx]
        
        # Filtrar nans
        valid_noise_clean = valid_noise[~np.isnan(valid_noise)]
        
        if len(valid_noise_clean) >= 4:
            q = max(1, len(valid_noise_clean) // 4) # Tomar el 25% inicial y final
            noise_start = np.mean(valid_noise_clean[:q])
            noise_end = np.mean(valid_noise_clean[-q:])
            noise_drift_pct = ((noise_end / noise_start) - 1.0) * 100.0 if noise_start > 0 else 0.0

            snr_start = np.mean(valid_snr[:q])
            snr_end = np.mean(valid_snr[-q:])
            snr_drop_pct = ((snr_start - snr_end) / snr_start) * 100.0 if snr_start > 0 else 0.0
        else:
            noise_drift_pct = np.nan
            snr_drop_pct = np.nan
            
# -------------------------------------------------------------------------------

# -------------------------------------------------------------------------------

        # eje de tiempo del pulso promedio
        t_pulso = np.linspace(-pre_w, post_w, target_len, endpoint=False)

        # ---------- Umbral fijo absoluto solicitado ----------
        umbral_fijo_abs = float(fixed_umbral_abs)
        noise_mask_fixed = np.abs(pulso_promedio) < umbral
        noise_signal_from_fixed = np.where(noise_mask_fixed, pulso_promedio, np.nan)
        if np.any(noise_mask_fixed):
            noise_rms_from_fixed = rms(pulso_promedio[noise_mask_fixed])
        else:
            noise_rms_from_fixed = 0.0
        pulso_promedio = np.mean(segmentos_norm, axis=0)
        pulso_std = np.std(segmentos_norm, axis=0)

        # color
        color_prom = tuple(rng.rand(3).tolist()) if colores_aleatorios else colorgrafico
        # renombro el snr (sustituido por la definicion pedida: amplitud maxima sobre umbral)
        max_amp = np.max(pulso_promedio)
        snr_manual = max_amp / umbral if (umbral is not None and umbral > 0) else np.inf

        # incertidumbres pedidas: incert. de amplitud = error del promedio en el índice del maximo
        idx_peak = int(np.argmax(pulso_promedio))
        amp_uncertainty = pulso_err[idx_peak] if idx_peak < len(pulso_err) else np.nan
        noise_sigma = sigma_est if sigma_est is not None else np.nan
        snr_uncertainty = amp_uncertainty / umbral if (umbral is not None and umbral > 0) else np.nan

        print(f"{filename}: max_amp={max_amp:.5e}, umbral={umbral:.5e}, snr_manual={snr_manual:.3f}, amp_uncert={amp_uncertainty:.3e}, noise_sigma={noise_sigma:.3e}")

        # ---------------------- Prepare output folder per file ----------------
        # Usar el output_root directamente como directorio de salida para los archivos de este canal.
        out_dir = output_root

        # nombres de archivo simplificados para que tu otro programa los lea fácilmente
        out_prom = os.path.join(out_dir, "avg.png")       # pulso promedio
        out_spec = os.path.join(out_dir, "spec.png")      # espectro/espectrograma
        out_rec = os.path.join(out_dir, "pulses.png")       # recortes / señal original
        out_evol = os.path.join(out_dir, "evolucion.png")   # evolucion temporal

        # --- GRAFICO: pulsos individuales y promedio (restaurado completo) ---
        _plot_pulse_full(
            t_pulso=t_pulso,
            segmentos_norm=segmentos_norm,
            pulso_promedio=pulso_promedio,
            pulso_err=pulso_err, # <- Se envía el error estándar del promedio
            color_prom=color_prom,
            snr_manual=snr_manual,
            snr_uncertainty=snr_uncertainty,
            noise_signal_from_fixed=noise_signal_from_fixed,
            umbral=umbral,
            calcular_umbral=calcular_umbral,
            mostrar_umbral=mostrar_umbral,
            factor_umbral=factor_umbral,
            filename=final_plot_title,
            out_prom=out_prom,
            plot_mode=plot_mode,
            individual_alpha=individual_alpha,
            mostrar_individuales=mostrar_individuales,
            show_plot=show_average_plot, # <-- Pasar el nuevo parámetro
            noise_drift_pct=noise_drift_pct,
            snr_drop_pct=snr_drop_pct,
            pulso_sigma=pulso_sigma,
            noise_sigma=noise_sigma
        )

        # --- BLOQUE: Espectrograma y espectro de frecuencias del pulso promedio ---
        if mostrar_espectrograma:
            _plot_espectro_and_spectrogram(pulso_promedio, target_len, pre_w, post_w,
                                           espectrograma_db, frecuenciamaxima, frecuenciaminima, out_spec, final_plot_title)

        # --- GRAFICO: señal original recortada con cortes periódicos y puntos de máximo (deteccion en envolvente) ---
        interactive_excluded = excluded_windows
        if mostrar_recortes:
            interactive_excluded = _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds, start_sample_noise, samplerate, maxima_per_cut, periodo, muestras_pulso, out_rec, final_plot_title, excluded_windows=excluded_windows,
                           show_plot=show_interactive_plot,
                           signal_original_unfiltered=signal_unfiltered[mask], # <-- Pasar la señal original recortada
                           mostrar_senal_cruda=mostrar_senal_cruda
                           )
                           
        # --- GRAFICO: Evolución Temporal ---
        if mostrar_evolucion:
            _plot_evolucion_temporal(
                stats_time, stats_snr, stats_snr_err, stats_snr_inst, stats_snr_inst_err, stats_noise_mean, stats_noise_err, stats_cv_snr, 
                out_evol, final_plot_title, evol_t_start, evol_t_end
            )

        # --- NUEVO: Mensaje de resumen mejorado ---
        print(f"\nCargando, por favor espere... RESUMEN para {filename}:\nVentanas totales={n_pulsos}, Ventanas promediadas={len(segmentos_rs)}\nSNR_manual={snr_manual:.2f}, Deriva Ruido={noise_drift_pct:+.1f}%, Caída SNR={snr_drop_pct:+.1f}%")

        promedios_globales.append(np.mean(segmentos_norm, axis=0))
        tiempos_globales.append(t_pulso)
        nombres_globales.append(filename)

        resultados[filename] = {
            'maxima_per_cut': maxima_per_cut,
            'segmentos_rs': segmentos_rs,
            'segmentos_norm': segmentos_norm,
            'mean_pulse': np.mean(segmentos_norm, axis=0),
            'std_pulse': np.std(segmentos_norm, axis=0),
            'snr_per_pulse': snr_per_pulse,
            'snr_mean': snr_mean,
            'snr_db': snr_db,
            'snr_manual': snr_manual,
            'umbral': umbral,
            'noise_rms': noise_rms,
            'fixed_umbral_abs': umbral_fijo_abs,
            'noise_rms_from_fixed': noise_rms_from_fixed,
            'noise_signal_from_fixed': noise_signal_from_fixed,
            'noise_seconds_used': noise_seconds,
            'noise_rms_from_noise_window': noise_rms_from_noise_window,
            'noise_sigma': noise_sigma,
            'amp_uncertainty': amp_uncertainty,
            'snr_uncertainty': snr_uncertainty,
            'noise_drift_pct': noise_drift_pct,
            'snr_drop_pct': snr_drop_pct,
            'out_dir': out_dir,
            'out_prom': out_prom,
            'out_spec': out_spec,
            'out_rec': out_rec,
            'out_evol': out_evol,
            'pulse_time': t_pulso,
            # --- NUEVO: Añadir datos para poder regenerar el gráfico de recortes ---
            't_recortada': t_recortada,
            'signal_recortada': signal_recortada,
            'env_recortada': env_recortada,
            'samplerate': samplerate,
            'periodo': periodo,
            'muestras_pulso': muestras_pulso,
            'display_name_for_plot': final_plot_title,
            'excluded_windows': excluded_windows,
            'interactive_excluded_windows': interactive_excluded,
            'start_sample_noise': start_sample_noise,
            'measurement_date': measurement_date,
            'comentario': comentario,
            'stats_time': stats_time,
            'stats_snr': stats_snr,
            'stats_snr_err': stats_snr_err,
            'stats_snr_inst': stats_snr_inst,
            'stats_snr_inst_err': stats_snr_inst_err,
            'stats_noise_mean': stats_noise_mean,
            'stats_noise_err': stats_noise_err,
            'stats_cv_snr': stats_cv_snr
        }

        # ---------------------- Export results into per-file folder -----------------
        export_results_for_file(out_dir, filename, resultados[filename])
        
        plt.close('all') # --- NUEVO: Forzar limpieza de memoria de Matplotlib ---

    # tabla comparativa y histogramas (modificada)
    if mostrar_tabla and promedios_globales:
        _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados, nombre_salida)

    return resultados

class ProcessingOptionsDialog(tk.Toplevel):
    """Diálogo para seleccionar canales y opciones de procesamiento individual."""
    def __init__(self, root):
        self.root = root
        super().__init__(root)
        self.title("Ñandú LSD - Opciones de Procesamiento")
        self.geometry("450x400")
        self.transient(root)
        self.grab_set()

        # --- Estilo Cyberpunk ---
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.cyan_muted = "#45A29E"
        self.green_neon = "#00FF00"
        self.fg_text = "#C5C6C7"
        
        self.configure(bg=self.bg_dark)

        self.mediciones_a_procesar = []
        self.canales_seleccionados = {} # { 'canal': var_bool }

        main_frame = tk.Frame(self, padx=15, pady=15, bg=self.bg_dark)
        main_frame.pack(fill="both", expand=True)

        # --- Sección de Selección de Canales ---
        channels_frame = tk.LabelFrame(main_frame, text="2. Seleccionar Canales a Procesar (Global)", padx=10, pady=10, bg=self.bg_panel, fg=self.cyan_neon, font=("Consolas", 11, "bold"), bd=2, relief="ridge")
        channels_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.channel_list_frame = tk.Frame(channels_frame, bg=self.bg_panel)
        self.channel_list_frame.pack(fill="both", expand=True)

        # --- Opciones de Análisis Individual ---
        individual_plots_frame = tk.LabelFrame(main_frame, text="Opciones de Análisis Individual", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon, font=("Consolas", 11, "bold"), bd=2, relief="ridge")
        individual_plots_frame.pack(fill="x", pady=(0, 15))
        
        self.var_mostrar_recortes = tk.BooleanVar(value=True)
        self.var_mostrar_senal_cruda = tk.BooleanVar(value=True)
        self.var_mostrar_espectrograma = tk.BooleanVar(value=False)
        self.var_excluded_windows = tk.StringVar(value="")
        # --- NUEVO: Opción para filtro pasa-bajos ---
        self.var_lowpass_cutoff = tk.StringVar(value="500") # Valor por defecto para el filtro pasa-bajos
        # --- NUEVO: Opción para filtro pasa-altos ---
        self.var_highpass_cutoff = tk.StringVar(value="20") # Valor por defecto para pasa-altos
        # --- NUEVO: Opción para filtro notch ---
        self.var_notch_filter = tk.BooleanVar(value=False) # Por defecto activado
        # --- NUEVO: Parámetro de suavizado de envolvente ---
        self.var_smooth_ms = tk.StringVar(value="50")
        self.var_tipo_env = tk.StringVar(value="media_movil") # Puede ser "media_movil" o "rms"
        # --- NUEVO: Opción Evolución Temporal ---
        self.var_mostrar_evolucion = tk.BooleanVar(value=True)
        self.var_evol_t_start = tk.StringVar(value="10")
        self.var_evol_t_end = tk.StringVar(value="1000")


        tk.Checkbutton(individual_plots_frame, text="Graficar recortes (pulses.png)", variable=self.var_mostrar_recortes, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(individual_plots_frame, text="Mostrar señal cruda en recortes", variable=self.var_mostrar_senal_cruda, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(individual_plots_frame, text="Generar espectrograma (spec.png)", variable=self.var_mostrar_espectrograma, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")

        evol_frame = tk.Frame(individual_plots_frame, bg=self.bg_panel)
        evol_frame.pack(fill='x', pady=(5,0))
        tk.Checkbutton(evol_frame, text="Gráfico Evolución Temporal SNR y Ruido", variable=self.var_mostrar_evolucion, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark).pack(side="left")
        
        exclude_frame = tk.Frame(individual_plots_frame, bg=self.bg_panel)
        exclude_frame.pack(fill='x', pady=(5,0))
        tk.Label(exclude_frame, text="Excluir ventanas (ej: 1,24):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        tk.Entry(exclude_frame, textvariable=self.var_excluded_windows, bg=self.bg_dark, fg=self.cyan_neon, insertbackground=self.cyan_neon).pack(side="left", fill="x", expand=True, padx=(5,0))

        # --- NUEVO: Campo para suavizado de envolvente ---
        smooth_frame = tk.Frame(individual_plots_frame, bg=self.bg_panel)
        smooth_frame.pack(fill='x', pady=(5,0))
        tk.Label(smooth_frame, text="Tipo Envolvente:", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        env_options = ["media_movil", "rms", "ninguna"]
        tk.OptionMenu(smooth_frame, self.var_tipo_env, *env_options).pack(side="left", padx=(5,15))
        
        tk.Label(smooth_frame, text="Suavizado (ms):", bg=self.bg_panel, fg=self.fg_text).pack(side="left")
        tk.Entry(smooth_frame, textvariable=self.var_smooth_ms, width=8, bg=self.bg_dark, fg=self.cyan_neon, insertbackground=self.cyan_neon).pack(side="left", padx=(5,0))

        # --- NUEVO: Filtros ---
        filter_frame = tk.Frame(individual_plots_frame, bg=self.bg_panel)
        filter_frame.pack(fill='x', pady=(5,0))
        
        tk.Checkbutton(filter_frame, text="Aplicar filtro Notch (50Hz)", variable=self.var_notch_filter, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))
        
        tk.Label(filter_frame, text="Filtro Pasa-bajos (Hz):", bg=self.bg_panel, fg=self.fg_text).grid(row=1, column=0, sticky="w")
        tk.Entry(filter_frame, textvariable=self.var_lowpass_cutoff, width=10, bg=self.bg_dark, fg=self.cyan_neon, insertbackground=self.cyan_neon).grid(row=1, column=1, sticky="w")
        
        tk.Label(filter_frame, text="Filtro Pasa-altos (Hz):", bg=self.bg_panel, fg=self.fg_text).grid(row=2, column=0, sticky="w")
        tk.Entry(filter_frame, textvariable=self.var_highpass_cutoff, width=10, bg=self.bg_dark, fg=self.cyan_neon, insertbackground=self.cyan_neon).grid(row=2, column=1, sticky="w")

        # --- Botones de Procesar ---
        btn_frame = tk.Frame(main_frame, bg=self.bg_dark)
        btn_frame.pack(fill="x", pady=(10, 0))
        
        btn_procesar = tk.Button(btn_frame, text="Procesar y Curar", command=lambda: self.procesar(interactivo=True), bg="#111111", fg=self.green_neon, activebackground=self.green_neon, activeforeground="#111111", font=("Consolas", 11, "bold"), bd=2, relief="solid", cursor="hand2")
        btn_procesar.pack(side="left", fill="x", expand=True, ipady=5, padx=(0, 5))
        
        btn_rapido = tk.Button(btn_frame, text="Reprocesar Rápido", command=lambda: self.procesar(interactivo=False), bg="#111111", fg=self.cyan_neon, activebackground=self.cyan_neon, activeforeground="#111111", font=("Consolas", 11, "bold"), bd=2, relief="solid", cursor="hand2")
        btn_rapido.pack(side="right", fill="x", expand=True, ipady=5, padx=(5, 0))

    def populate_channels(self, base_dir, mediciones):
        self.mediciones_a_procesar = mediciones
        self.BASE_DIR = base_dir
        
        canales_unicos = set()
        for nombre_medicion in self.mediciones_a_procesar:
            path_medicion = os.path.join(self.BASE_DIR, nombre_medicion)
            try:
                canales = [item for item in os.listdir(path_medicion) if os.path.isdir(os.path.join(path_medicion, item)) and item.startswith("canal_")]
                canales_unicos.update(canales)
            except Exception as e:
                print(f"Error al leer canales de {nombre_medicion}: {e}")

        canales_ordenados = sorted(list(canales_unicos), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

        if canales_ordenados:
            for canal in canales_ordenados:
                var = tk.BooleanVar(value=True)
                self.canales_seleccionados[canal] = var
                tk.Checkbutton(self.channel_list_frame, text=canal, variable=var, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        else:
            tk.Label(self.channel_list_frame, text="No se encontraron canales en las mediciones seleccionadas.", fg="red", bg=self.bg_panel).pack(anchor="w")

    def procesar(self, interactivo=True):
        canales_globales = [canal for canal, var in self.canales_seleccionados.items() if var.get()]
        if not canales_globales:
            tk.messagebox.showerror("Error", "No se ha seleccionado ningún canal para procesar.", parent=self)
            return

        canales_a_procesar = []
        for nombre_medicion in self.mediciones_a_procesar:
            for canal in canales_globales:
                canal_path_rel = os.path.join(nombre_medicion, canal)
                canal_path_abs = os.path.join(self.BASE_DIR, canal_path_rel)
                if os.path.exists(canal_path_abs) and os.path.isdir(canal_path_abs):
                    canales_a_procesar.append(canal_path_rel)
                    
        if not canales_a_procesar:
            tk.messagebox.showerror("Error", "Los canales seleccionados no existen en las mediciones elegidas.", parent=self)
            return

        try:
            excluded_windows_list = [int(x.strip()) for x in self.var_excluded_windows.get().split(',') if x.strip()]
        except ValueError:
            tk.messagebox.showerror("Error de Formato", "El formato de las ventanas a excluir es incorrecto.", parent=self)
            return

        # --- NUEVO: Validar la entrada de los filtros ---
        try:
            lowpass_freq = float(self.var_lowpass_cutoff.get())
            highpass_freq = float(self.var_highpass_cutoff.get())
            evol_t_start = float(self.var_evol_t_start.get())
            evol_t_end = float(self.var_evol_t_end.get())
            smooth_val = float(self.var_smooth_ms.get())
        except ValueError:
            tk.messagebox.showerror("Error de Formato", "Las frecuencias de los filtros y tiempos deben ser números.", parent=self)
            return
            
        # --- NUEVO: Obtener el estado del checkbox del filtro notch ---
        apply_notch = self.var_notch_filter.get()

        if not tk.messagebox.askyesno("Confirmar", f"Se procesarán {len(canales_a_procesar)} canales. Esto puede tardar. ¿Continuar?", parent=self):
            return

        # --- NUEVO: Cerrar la ventana de opciones al iniciar el procesamiento ---
        self.destroy()

        # --- NUEVO: Ocultar la ventana principal en lugar de destruirla para que funcionen los pop-ups ---
        self.root.withdraw()

        total_canales = len(canales_a_procesar)
        print_progress_bar(0, total_canales, prefix='Procesando Canales:', suffix='Completado', length=50)

        for i, canal_path_rel in enumerate(canales_a_procesar):
            nombre_medicion, item = os.path.split(canal_path_rel)
            carpeta_a_analizar = os.path.join(self.BASE_DIR, canal_path_rel)
            is_interactive = interactivo
            print(f"\n--- Procesando: {canal_path_rel} ---")
            
            # --- LÓGICA MEJORADA: Cargar metadata, incluyendo ventanas excluidas ---
            bpm_a_usar, noise_seconds_a_usar, pulsos_a_usar = 50, 2.0, None
            excluded_from_meta = []
            # --- NUEVO: Bandera para controlar si se hace la curación interactiva ---
            perform_curation = True
            final_excluded_windows = []

            meta_data = {}
            meta_path = os.path.join(carpeta_a_analizar, 'metadata.json')
            try:
                with open(meta_path, 'r') as f:
                    meta_data = json.load(f)
                    bpm_a_usar = meta_data.get('bpm', bpm_a_usar)
                    noise_seconds_a_usar = meta_data.get('noise_seconds', noise_seconds_a_usar)
                    pulsos_a_usar = meta_data.get('pulse_count', pulsos_a_usar)
                    excluded_from_meta = meta_data.get('excluded_windows', [])
                    print(f"     Cargado desde metadata: BPM={bpm_a_usar}, Ruido={noise_seconds_a_usar}s, Pulsos={pulsos_a_usar}, Excluidas={excluded_from_meta}")
            except Exception as e:
                print(f"     Advertencia: No se pudo leer metadata.json. Usando defaults. Error: {e}")

            # Combinar exclusiones de la GUI y del metadata para la lista inicial
            initial_excluded_windows = sorted(list(set(excluded_windows_list + excluded_from_meta)))

            if is_interactive:
                # --- NUEVO: Preguntar si se quiere curar si ya hay ventanas excluidas ---
                if initial_excluded_windows:
                    perform_curation = tk.messagebox.askyesno("Curación Opcional",
                        f"Canal: {nombre_medicion} - {item}\nYa existen ventanas excluidas: {initial_excluded_windows}.\n\n"
                        "¿Desea realizar una nueva curación para modificar esta lista?")

            if is_interactive and perform_curation:
                print("\n[Paso 1 de 2] Realizando análisis inicial para visualización...")
                # Primer paso: siempre genera el gráfico de recortes, mostrando las ventanas ya excluidas.
                # Esto permite ver todas las ventanas para decidir cuáles quitar.
                res_inicial = procesar_wavs_promedio(
                    carpeta=carpeta_a_analizar, output_root=carpeta_a_analizar, nombre_salida="analisis_inicial.png",
                    bpm=bpm_a_usar, mostrar_individuales=False, mostrar_recortes=True, mostrar_espectrograma=False,
                    mostrar_tabla=False, display_name_for_plot=f"{nombre_medicion} ({item})",
                    noise_seconds=noise_seconds_a_usar, n_pulsos_manual=pulsos_a_usar, excluded_windows=initial_excluded_windows,
                    show_interactive_plot=True, # <-- Mostrar el gráfico
                    apply_notch_filter=apply_notch, # <-- Pasar estado del filtro
                    lowpass_cutoff_hz=lowpass_freq,
                    highpass_cutoff_hz=highpass_freq,
                    smooth_ms=smooth_val, # <-- NUEVO
                    tipo_envolvente=self.var_tipo_env.get(),
                    mostrar_evolucion=False,
                    evol_t_start=evol_t_start,
                    evol_t_end=evol_t_end,
                    mostrar_senal_cruda=self.var_mostrar_senal_cruda.get()
                )
                
                interactive_excl = initial_excluded_windows
                if res_inicial:
                    for fname, res in res_inicial.items():
                        if 'interactive_excluded_windows' in res:
                            interactive_excl = res['interactive_excluded_windows']
                            break

                print("\n[Paso 2 de 2] Curación de datos (opcional).")
                print("Se ha generado el gráfico 'pulses.png' en la carpeta del canal.")
                
                # --- NUEVO: Mostrar y actualizar root para evitar que el diálogo falle en canal 0 ---
                self.root.deiconify()
                self.root.update()
                
                # --- NUEVO: Pop-up interactivo en lugar de consola ---
                user_input = simpledialog.askstring(
                    "Curación de datos",
                    f"Medición: {nombre_medicion} ({item})\n\n"
                    "Se han registrado tus selecciones interactivas.\n"
                    "Puedes editar la LISTA COMPLETA de ventanas a excluir, separadas por comas:\n"
                    f"(Exclusiones actuales: {interactive_excl})",
                    initialvalue=str(interactive_excl).strip("[]"),
                    parent=self.root
                )
                
                self.root.withdraw() # Volver a ocultar la ventana
                
                additional_exclusions = []
                if user_input is not None and user_input.strip():
                    try:
                        additional_exclusions = [int(x.strip()) for x in user_input.split(',') if x.strip()]
                        # La lista final para guardar es la unión de las que ya había y las nuevas
                        windows_to_save = sorted(list(set(additional_exclusions))) # La nueva entrada reemplaza a la anterior
                        print(f"Se re-analizará excluyendo las ventanas: {windows_to_save}")
                        final_excluded_windows = windows_to_save # Actualizar la lista final

                    except ValueError:
                        print("Entrada inválida. Se usarán las exclusiones interactivas.")
                        final_excluded_windows = interactive_excl
                elif user_input is not None and not user_input.strip():
                    # Si el usuario borra todo y da OK
                    print("Se limpió la lista de exclusiones.")
                    final_excluded_windows = []
                else:
                    print("Operación cancelada. Se mantendrán las exclusiones interactivas.")
                    final_excluded_windows = interactive_excl
                    
                # --- NUEVO: Guardar las ventanas excluidas en metadata.json de forma segura ---
                try:
                    md_temp = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            md_temp = json.load(f)
                    md_temp['excluded_windows'] = final_excluded_windows
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(md_temp, f, indent=4)
                    print(f"Lista de exclusión ({final_excluded_windows}) guardada en '{meta_path}'.")
                except Exception as e:
                    print(f"Advertencia: No se pudo actualizar metadata.json: {e}")
                    
                print("\nRealizando análisis final con las ventanas seleccionadas...")
            else: # Modo no interactivo o el usuario eligió no curar
                final_excluded_windows = initial_excluded_windows
                print(f"Aplicando exclusión de ventanas pre-configurada: {final_excluded_windows}")

            # Análisis final (o único si no es interactivo)
            procesar_wavs_promedio(
                    carpeta=carpeta_a_analizar,
                    output_root=carpeta_a_analizar,
                    nombre_salida="analisis_final.png",
                    bpm=bpm_a_usar,
                    mostrar_individuales=False,
                    mostrar_recortes=self.var_mostrar_recortes.get(),
                    mostrar_espectrograma=self.var_mostrar_espectrograma.get(),
                    mostrar_tabla=False,
                    display_name_for_plot=f"{nombre_medicion} ({item})",
                    noise_seconds=noise_seconds_a_usar,
                    lowpass_cutoff_hz=lowpass_freq, # <-- Pasar la frecuencia del filtro
                    highpass_cutoff_hz=highpass_freq, # <-- Pasar la frecuencia del filtro pasa-altos
                    n_pulsos_manual=pulsos_a_usar,
                    show_average_plot=is_interactive, # <-- Mostrar el gráfico promedio si es interactivo
                    show_interactive_plot=False, # <-- El análisis final no necesita ser mostrado
                    excluded_windows=final_excluded_windows, # Usar la lista final de exclusión
                    apply_notch_filter=apply_notch, # <-- Pasar estado del filtro
                    smooth_ms=smooth_val, # <-- NUEVO
                    tipo_envolvente=self.var_tipo_env.get(),
                    mostrar_evolucion=self.var_mostrar_evolucion.get(),
                    evol_t_start=evol_t_start,
                    evol_t_end=evol_t_end,
                    mostrar_senal_cruda=self.var_mostrar_senal_cruda.get()
                )

            print_progress_bar(i + 1, total_canales, prefix='Procesando Canales:', suffix='Completado', length=50)
        
        # --- NUEVO: Mostrar el gráfico de recortes final si fue interactivo ---
        if is_interactive:
            print("\n--- ¡Procesamiento de mediciones individuales completado! ---")
            print("\nMostrando gráfico de recortes final con las ventanas excluidas marcadas en rojo.")
            print("Cierra la ventana del gráfico para continuar...")
            # --- CORRECCIÓN: En lugar de regenerar el gráfico, abrimos el que ya se guardó ---
            try:
                # La ruta al gráfico de recortes se guardó durante el análisis final
                final_pulses_png_path = os.path.join(carpeta_a_analizar, "pulses.png")
                if os.path.exists(final_pulses_png_path):
                    subprocess.run(["start", final_pulses_png_path], shell=True, check=True)
            except Exception as e:
                print(f"No se pudo abrir el gráfico de recortes final '{final_pulses_png_path}'. Error: {e}")
            print("Ahora puedes volver a abrir el script para lanzar un análisis comparativo.")
        else:
            print("\n--- ¡Procesamiento de mediciones individuales completado! ---")
            print("Ahora puedes volver a abrir el script para lanzar un análisis comparativo.")
        
        # --- NUEVO: Cerrar completamente la aplicación al finalizar ---
        self.root.destroy()


class ComparativeOptionsDialog(tk.Toplevel):
    """Diálogo para configurar y lanzar el análisis comparativo."""
    def __init__(self, root):
        self.root = root
        super().__init__(root)
        self.title("Ñandú LSD - Opciones de Comparación")
        self.geometry("450x520")
        self.transient(root)
        self.grab_set()

        # --- Estilo Cyberpunk ---
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.cyan_muted = "#45A29E"
        self.magenta_neon = "#FF00FF"
        self.fg_text = "#C5C6C7"
        
        self.configure(bg=self.bg_dark)

        self.mediciones_a_comparar = []
        self.BASE_DIR = ""

        main_frame = tk.Frame(self, padx=15, pady=15, bg=self.bg_dark)
        main_frame.pack(fill="both", expand=True)

        # --- Nombre del Set de Análisis ---
        name_frame = tk.Frame(main_frame, bg=self.bg_panel, bd=2, relief="ridge")
        name_frame.pack(fill="x", pady=(0, 10))
        tk.Label(name_frame, text="Nombre del Set de Análisis (opcional):", bg=self.bg_panel, fg=self.fg_text).pack(side="left", padx=5, pady=5)
        self.var_nombre_analisis = tk.StringVar()
        tk.Entry(name_frame, textvariable=self.var_nombre_analisis, bg=self.bg_dark, fg=self.cyan_neon, insertbackground=self.cyan_neon).pack(side="left", fill="x", expand=True, padx=(5, 5))

        # --- Sección de Selección de Canal ---
        self.channel_frame = tk.LabelFrame(main_frame, text="2. Comparar datos del Canal:", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon, font=("Consolas", 11, "bold"), bd=2, relief="ridge")
        self.channel_frame.pack(fill="x", pady=(0, 15))
        
        self.var_canal_a_usar = tk.StringVar()
        self.channel_menu = tk.OptionMenu(self.channel_frame, self.var_canal_a_usar, "")
        self.channel_menu.config(bg=self.bg_dark, fg=self.cyan_neon, activebackground=self.cyan_muted, activeforeground=self.bg_dark)
        self.channel_menu.pack(fill="x")
        self.channel_menu.config(state="disabled")

        # --- Opciones de Gráficos Comparativos ---
        comparative_plots_frame = tk.LabelFrame(main_frame, text="Opciones de Gráficos Comparativos", padx=10, pady=5, bg=self.bg_panel, fg=self.cyan_neon, font=("Consolas", 11, "bold"), bd=2, relief="ridge")
        comparative_plots_frame.pack(fill="x", expand=True, pady=(0, 15))
        
        self.var_show_overlay = tk.BooleanVar(value=True)
        self.var_show_snr = tk.BooleanVar(value=True)
        self.var_show_amplitude = tk.BooleanVar(value=True)
        self.var_show_table = tk.BooleanVar(value=True)
        self.var_show_snr_time = tk.BooleanVar(value=True)
        self.var_show_amp_time = tk.BooleanVar(value=True)

        tk.Checkbutton(comparative_plots_frame, text="Generar Overlay de Pulsos", variable=self.var_show_overlay, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(comparative_plots_frame, text="Generar Gráfico SNR (Amplitud)", variable=self.var_show_snr, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(comparative_plots_frame, text="Generar Gráfico Amplitud Máxima", variable=self.var_show_amplitude, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(comparative_plots_frame, text="Generar Gráfico SNR vs Tiempo", variable=self.var_show_snr_time, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(comparative_plots_frame, text="Generar Gráfico Amplitud vs Tiempo", variable=self.var_show_amp_time, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")
        tk.Checkbutton(comparative_plots_frame, text="Generar Tabla de Resultados (CSV y PNG)", variable=self.var_show_table, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark, activebackground=self.bg_panel, activeforeground=self.cyan_neon).pack(anchor="w")

        # --- Botón de Lanzar ---
        btn_lanzar = tk.Button(main_frame, text="LANZAR ANÁLISIS COMPARATIVO", command=self.lanzar, bg="#111111", fg=self.magenta_neon, activebackground=self.magenta_neon, activeforeground="#111111", font=("Consolas", 11, "bold"), bd=2, relief="solid", cursor="hand2")
        btn_lanzar.pack(fill="x", ipady=5, pady=(10, 0))

    def populate_common_channels(self, base_dir, mediciones):
        self.mediciones_a_comparar = mediciones
        self.BASE_DIR = base_dir
        
        canales_comunes = None
        for nombre_medicion in self.mediciones_a_comparar:
            path_medicion = os.path.join(self.BASE_DIR, nombre_medicion)
            canales_actuales = set()
            try:
                for item in os.listdir(path_medicion):
                    channel_dir = os.path.join(path_medicion, item)
                    # Un canal es común si tiene el archivo de resultados
                    if os.path.isdir(channel_dir) and item.startswith("canal_") and os.path.exists(os.path.join(channel_dir, 'analisis_results.json')):
                        canales_actuales.add(item)
            except Exception as e:
                print(f"Advertencia al leer {nombre_medicion}: {e}")
                continue
            
            if canales_comunes is None:
                canales_comunes = canales_actuales
            else:
                canales_comunes.intersection_update(canales_actuales)

        menu = self.channel_menu["menu"]
        menu.delete(0, "end")
        
        if canales_comunes:
            sorted_canales = sorted(list(canales_comunes), key=lambda x: int(x.split('_')[-1]))
            for canal in sorted_canales:
                menu.add_command(label=canal, command=lambda value=canal: self.var_canal_a_usar.set(value))
            self.var_canal_a_usar.set(sorted_canales[0])
            self.channel_menu.config(state="normal")
        else:
            self.var_canal_a_usar.set("")
            self.channel_menu.config(state="disabled")
            tk.messagebox.showwarning("Advertencia", "No se encontraron canales comunes con datos ya procesados entre las mediciones seleccionadas.", parent=self)

    def lanzar(self):
        canal_a_usar = self.var_canal_a_usar.get()
        if not self.mediciones_a_comparar or not canal_a_usar:
            tk.messagebox.showerror("Error", "Debes seleccionar al menos dos mediciones y un canal común.", parent=self)
            return

        nombre_custom = self.var_nombre_analisis.get().strip()
        nombre_custom = re.sub(r'[\\/*?:"<>|]', "", nombre_custom) # Quitar caracteres inválidos para carpetas

        self.destroy()
        self.root.destroy()

        resultados_globales = {}
        total_mediciones = len(self.mediciones_a_comparar)
        print_progress_bar(0, total_mediciones, prefix='Cargando Resultados:', suffix='Completado', length=50)

        for i, nombre_medicion in enumerate(self.mediciones_a_comparar):
            clave_resultado = f"{nombre_medicion}-{canal_a_usar}"
            carpeta_a_cargar = os.path.join(self.BASE_DIR, nombre_medicion, canal_a_usar)
            results_path = os.path.join(carpeta_a_cargar, 'analisis_results.json')
            
            print(f"\n--- Cargando resultados para: {clave_resultado} ---")
            try:
                with open(results_path, 'r', encoding='utf-8') as f:
                    resultados_cargados = json.load(f)
                
                # Intentar leer metadatos para hora y comentario si no vinieron en results
                meta_path = os.path.join(carpeta_a_cargar, 'metadata.json')
                try:
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f_meta:
                            meta_data = json.load(f_meta)
                            if 'measurement_date' not in resultados_cargados:
                                resultados_cargados['measurement_date'] = meta_data.get('measurement_date', '')
                            if 'comentario' not in resultados_cargados:
                                resultados_cargados['comentario'] = meta_data.get('comentario', '')
                except Exception as e_meta:
                    pass

                resultados_cargados['file'] = clave_resultado
                resultados_globales[clave_resultado] = resultados_cargados
                print(f"Resultados de '{clave_resultado}' cargados exitosamente.")
            except Exception as e:
                print(f"ERROR: No se pudo cargar el archivo '{results_path}'.")
                print(f"Asegúrate de haber procesado esta medición primero.")
                print(f"Error detallado: {e}")
            
            print_progress_bar(i + 1, total_mediciones, prefix='Cargando Resultados:', suffix='Completado', length=50)

        if len(resultados_globales) > 1:
            print("\n--- Generando Gráficos Comparativos ---")
            promedios_globales = [res['mean_pulse'] for res in resultados_globales.values() if 'mean_pulse' in res]
            tiempos_globales = [res['pulse_time'] for res in resultados_globales.values() if 'pulse_time' in res]
            nombres_globales = [res['file'] for res in resultados_globales.values() if 'file' in res]
            
            today_str = datetime.now().strftime("%Y-%m-%d")
            timestamp = datetime.now().strftime("%H%M%S")
            
            if not nombre_custom:
                nombre_carpeta = f"comparacion_{timestamp}"
            else:
                nombre_carpeta = nombre_custom
                
            output_comp_dir = os.path.join("analisis_comparativos", today_str, nombre_carpeta)
            os.makedirs(output_comp_dir, exist_ok=True)
            
            nombre_salida_comp = os.path.join(output_comp_dir, "comparativa.png")
            
            _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados_globales, nombre_salida_comp,
                               show_overlay=self.var_show_overlay.get(),
                               show_snr=self.var_show_snr.get(),
                               show_amplitude=self.var_show_amplitude.get(),
                               show_table=self.var_show_table.get(),
                               show_snr_time=self.var_show_snr_time.get(),
                               show_amp_time=self.var_show_amp_time.get())
        else:
            print("\nNo se generaron gráficos comparativos. Se necesitan al menos dos mediciones con resultados válidos.")

class AnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Lanzador de Análisis v{__version__}")
        self.root.geometry("500x400")

        # --- Estilo Cyberpunk ---
        bg_dark = "#0B0C10"
        bg_panel = "#1F2833"
        cyan_neon = "#66FCF1"
        cyan_muted = "#45A29E"
        magenta_neon = "#FF00FF"
        green_neon = "#00FF41"
        fg_text = "#C5C6C7"
        
        self.root.configure(bg=bg_dark)

        # --- CORRECCIÓN: Usar una ruta absoluta para encontrar la carpeta de datos ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        emg_root_dir = script_dir
        while os.path.basename(emg_root_dir) != 'Emg' and emg_root_dir != os.path.dirname(emg_root_dir): emg_root_dir = os.path.dirname(emg_root_dir)
        self.BASE_DIR = os.path.join(emg_root_dir, "base_de_datos_electrodos")

        main_frame = tk.Frame(root, padx=15, pady=15, bg=bg_dark)
        main_frame.pack(fill="both", expand=True)

        # --- Sección de Selección de Mediciones ---
        measurements_frame = tk.LabelFrame(main_frame, text="1. Seleccionar Mediciones", padx=10, pady=10, bg=bg_panel, fg=cyan_neon, font=("Consolas", 12, "bold"), bd=2, relief="ridge")
        measurements_frame.pack(fill="both", expand=True, pady=(0, 15))

        self.listbox_mediciones = tk.Listbox(measurements_frame, selectmode=tk.EXTENDED, exportselection=False, bg=bg_dark, fg=cyan_neon, selectbackground=cyan_muted, selectforeground=bg_dark, font=("Consolas", 10), bd=1, highlightthickness=1, highlightcolor=cyan_neon)
        self.listbox_mediciones.pack(side="left", fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(measurements_frame, orient="vertical", command=self.listbox_mediciones.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox_mediciones.config(yscrollcommand=scrollbar.set)
        self.listbox_mediciones.bind("<<ListboxSelect>>", self.on_selection_change)

        # --- Botones de Acción ---
        action_frame = tk.Frame(main_frame, pady=10, bg=bg_dark)
        action_frame.pack(fill="x", side="bottom")

        self.btn_procesar = tk.Button(action_frame, text="PROCESAR DATOS INDIVIDUALES...", command=self.open_processing_dialog, state="disabled", bg="#111111", fg=green_neon, activebackground=green_neon, activeforeground="#111111", font=("Consolas", 11, "bold"), bd=2, relief="solid", cursor="hand2")
        self.btn_procesar.pack(fill="x", ipady=5, pady=(0, 10))

        self.btn_comparar = tk.Button(action_frame, text="LANZAR ANÁLISIS COMPARATIVO...", command=self.open_comparative_dialog, state="disabled", bg="#111111", fg=magenta_neon, activebackground=magenta_neon, activeforeground="#111111", font=("Consolas", 11, "bold"), bd=2, relief="solid", cursor="hand2")
        self.btn_comparar.pack(fill="x", ipady=5)

        self.cargar_mediciones()

    def cargar_mediciones(self):
        self.listbox_mediciones.delete(0, tk.END)
        try:
            if os.path.isdir(self.BASE_DIR):
                date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
                # Primero listamos carpetas de fecha
                for date_folder in sorted(os.listdir(self.BASE_DIR), reverse=True):
                    date_path = os.path.join(self.BASE_DIR, date_folder)
                    if os.path.isdir(date_path) and date_pattern.match(date_folder):
                        # Luego iteramos sobre mediciones en cada fecha
                        for med_folder in sorted(os.listdir(date_path)):
                            med_path = os.path.join(date_path, med_folder)
                            if os.path.isdir(med_path):
                                # Verificamos si contiene canales
                                has_channels = any(f.startswith("canal_") for f in os.listdir(med_path) if os.path.isdir(os.path.join(med_path, f)))
                                if has_channels:
                                    rel_path = os.path.join(date_folder, med_folder)
                                    self.listbox_mediciones.insert(tk.END, rel_path)
        except Exception as e:
            tk.messagebox.showerror("Error", f"No se pudo leer el directorio base:\n{e}")

    def on_selection_change(self, event=None):
        """Habilita los botones según la cantidad de mediciones seleccionadas."""
        selection_count = len(self.listbox_mediciones.curselection())
        
        if selection_count > 0:
            self.btn_procesar.config(state="normal")
        else:
            self.btn_procesar.config(state="disabled")

        if selection_count > 1:
            self.btn_comparar.config(state="normal")
        else:
            self.btn_comparar.config(state="disabled")

    def open_processing_dialog(self):
        mediciones = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        dialog = ProcessingOptionsDialog(self.root)
        dialog.populate_channels(self.BASE_DIR, mediciones)

    def open_comparative_dialog(self):
        mediciones = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        dialog = ComparativeOptionsDialog(self.root)
        dialog.populate_common_channels(self.BASE_DIR, mediciones)

if __name__ == "__main__":
    print(f"--- Script de Análisis de Pistas v{__version__} ---")
    root = tk.Tk()
    app = AnalysisGUI(root)
    root.mainloop()

# %%
