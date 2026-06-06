# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Cálculo y visualización de correlación cruzada entre diferentes señales.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Cálculo y visualización de correlación cruzada entre diferentes señales.
# ==============================================================================

#%%
import os
import json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, hilbert, butter, filtfilt, iirnotch, correlate, correlation_lags
from scipy import interpolate
import csv
import pandas as pd
import math
import re
from datetime import datetime
import argparse
import subprocess

# --- Imports para GUI ---
import sys

# --- Import para ConfigManager ---
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
try:
    from utils.config_manager import ConfigManager
    config_mgr = ConfigManager()
except Exception:
    config_mgr = None

from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox,
    QMessageBox, QFileDialog, QGroupBox, QSpinBox, QDoubleSpinBox, QWidget,
    QTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor

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

# ---------------------- Entry point ----------------------
def select_directories():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    master_dir = QFileDialog.getExistingDirectory(None, "SELECCIONA CARPETA DEL LÍDER (Ej: Canal 0)")
    if not master_dir:
        print("Operación cancelada (Falta Líder).")
        return None, None
        
    reply = QMessageBox.question(None, "Modo Multicanal", "¿Deseas seleccionar canales Slave para aplicarles la misma alineación?", QMessageBox.Yes | QMessageBox.No)
    slave_dirs = []
    
    if reply == QMessageBox.Yes:
        while True:
            slave_dir = QFileDialog.getExistingDirectory(None, "SELECCIONA CARPETA SLAVE (Ej: Canal 1 o 2). CANCELAR para terminar.")
            if not slave_dir:
                break
            slave_dirs.append(slave_dir)
            
    return master_dir, slave_dirs

class StdoutRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout

    def write(self, string):
        self.original_stdout.write(string)
        self.text_widget.moveCursor(QTextCursor.End)
        self.text_widget.insertPlainText(string)
        self.text_widget.moveCursor(QTextCursor.End)
        QApplication.processEvents()

    def flush(self):
        self.original_stdout.flush()

class ConsoleWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Procesando Datos...")
        self.resize(700, 400)
        
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("background-color: #000; color: #00ff00; border: 1px solid #333;")
        
        font = QFont("Consolas", 10)
        self.text_edit.setFont(font)
        
        layout.addWidget(self.text_edit)
        
        # Redirigir stdout
        self.redirector = StdoutRedirector(self.text_edit)
        sys.stdout = self.redirector

    def closeEvent(self, event):
        sys.stdout = self.redirector.original_stdout
        super().closeEvent(event)

def main(mediciones_dirs=None, medicion_dir=None, master_dir=None, slave_dirs=None):
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    if medicion_dir and not mediciones_dirs:
        mediciones_dirs = [medicion_dir]
        
    if not mediciones_dirs and not master_dir:
        return
        
    dialog = ProcessingOptionsDialog(medicion_dir=mediciones_dirs[0] if mediciones_dirs else None, master_dir=master_dir, slave_dirs=slave_dirs)
    if dialog.exec() == QDialog.Accepted:
        opts = dialog.result
        if not opts: return
        
        master_name = opts.pop("selected_master_name", None)
        slaves_names = opts.pop("selected_slaves_names", [])
        
        is_dark = opts.pop("tema_oscuro", True)
        if is_dark:
            plt.style.use('dark_background')
        else:
            plt.style.use('default')
            
        normalizar = opts.pop("normalizar_overlay", False)
    else:
        return

    console_win = None
    if not opts.get("show_interactive_plot", False):
        console_win = ConsoleWindow()
        console_win.show()
        QApplication.processEvents()

    for m_dir in (mediciones_dirs or [None]):
        if m_dir:
            current_master_dir = os.path.join(m_dir, master_name) if master_name else master_dir
            current_slave_dirs = [os.path.join(m_dir, s) for s in slaves_names] if slaves_names else slave_dirs
            print(f"\n======================================")
            print(f" PROCESANDO MEDICIÓN: {os.path.basename(m_dir)}")
            print(f"======================================")
        else:
            current_master_dir = master_dir
            current_slave_dirs = slave_dirs

        # --- 1. PROCESAR MASTER ---
        print(f"\n--- PROCESANDO LÍDER: {os.path.basename(current_master_dir)} ---")
        
        if not os.path.exists(current_master_dir):
            print(f"El directorio Líder no existe: {current_master_dir}. Saltando medición.")
            continue
            
        resultados_master = procesar_wavs_promedio(
            carpeta=current_master_dir,
            output_root=current_master_dir,
            **opts
        )
        
        if not resultados_master:
            print("El procesamiento Líder falló o no devolvió resultados.")
            continue

        master_shifts = {}
        master_valid_indices = {}
        for filename, data in resultados_master.items():
            master_shifts[filename] = data['shifts']
            master_valid_indices[filename] = data.get('valid_indices')

        # --- 2. PROCESAR SLAVES ---
        resultados_canales = {os.path.basename(current_master_dir): resultados_master}
        
        for s_dir in current_slave_dirs:
            if not os.path.exists(s_dir):
                print(f"\n--- ADVERTENCIA: Directorio SLAVE no encontrado: {s_dir}. Omitiendo... ---")
                continue
                
            ch_name = os.path.basename(s_dir)
            print(f"\n--- PROCESANDO SLAVE: {ch_name} ---")
            
            res_slave = procesar_wavs_promedio(
                carpeta=s_dir,
                output_root=s_dir,
                dict_shifts_externos=master_shifts,
                indices_validos_externos=master_valid_indices,
                **opts
            )
            resultados_canales[ch_name] = res_slave
            
        # --- 3. OVERLAY FINAL ---
        if current_slave_dirs:
            print("\nGenerando Overlay de músculos sincronizados...")
            parent_meas_dir = os.path.dirname(current_master_dir)
            meas_name = os.path.basename(parent_meas_dir)
            master_basename = os.path.basename(current_master_dir)
            _plot_muscle_overlay(meas_name, resultados_canales, parent_meas_dir, master_basename, normalize_all=normalizar)

    if console_win:
        print("\n\n✅ --- PROCESAMIENTO FINALIZADO --- ✅")
        print("Puede cerrar esta ventana para volver al menú principal.")
        console_win.exec()

if __name__ == "__main__":
    main()

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

def _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_env="media_movil"):
    if tipo_env == "rms" and smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            sig_sq = signal_abs ** 2
            window = np.ones(win_len, dtype=float) / float(win_len)
            return np.sqrt(np.convolve(sig_sq, window, mode='same'))
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

    if smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            window = np.ones(win_len, dtype=float) / float(win_len)
            env_full = np.convolve(env_full, window, mode='same')
    return env_full

# ---------------------- Estimación de Ruido -------------------
def _estimate_noise_window(signal_recortada, samplerate, noise_seconds, smooth_ms, factor_umbral, tipo_env="media_movil"):
    start_sample_noise = int(round(noise_seconds * samplerate))
    if start_sample_noise <= 0:
        start_sample_noise = 0
    if start_sample_noise >= len(signal_recortada):
        start_sample_noise = min(len(signal_recortada)-1, int(round(0.01 * len(signal_recortada))))

    if start_sample_noise > 0:
        # IGUAL A analisis_por_track: saltar 1 segundo inicial para evitar artefactos de inicio
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
        print(f"[Ruido Inicial] {noise_seconds}s, Umbral={umbral:.5e}")
        return start_sample_noise, env_noise, sigma_est, umbral, noise_rms_from_noise_window
    else:
        print(f"[Ruido] No se definió ventana de ruido.")
        return start_sample_noise, np.array([]), None, None, None


# ---------------------- DETECCION DE PICOS CENTRADA EN BEAT ---------------------
def _cortar_centrado_en_beat(env_recortada,
                             start_sample_noise,
                             muestras_pulso,
                             pre_samples,
                             post_samples,
                             peak_search_threshold,
                             n_pulsos_manual=None,
                             excluded_windows=None,
                             forced_maxima=None,
                             min_peak_distance_factor=0.5):
    """
    Detecta el pico máximo en cada ventana periódica del metrónomo.

    Geometría (igual a _detect_maxima_and_extract de analisis_por_track_integrado):
      - Ventana i: [start_sample_noise + i*T,  start_sample_noise + (i+1)*T]
      - El pulso del metrónomo suena en la mitad de esa ventana
      - Pico = argmax(envolvente) dentro de la ventana completa
      - Si pico < peak_search_threshold → ventana descartada (ruido)
      - Segmento extraído: [peak - pre_samples, peak + post_samples]
    """
    if len(env_recortada) == 0: return [], [], []
    if n_pulsos_manual is None or n_pulsos_manual <= 0:
        print("--- ERROR: Se requiere conteo de pulsos. ---")
        return [], [], []

    n_pulsos = int(n_pulsos_manual)

    maxima_detectados = [None] * n_pulsos
    segmentos_full = [None] * n_pulsos

    excluded_set = set(excluded_windows) if excluded_windows else set()
    
    min_dist_samples = max(1, int(round(min_peak_distance_factor * float(muestras_pulso))))
    last_valid_i = -1

    for i in range(n_pulsos):
        window_number = i + 1
        if window_number in excluded_set:
            print(f"    -> Omitiendo ventana #{window_number} (excluida).")
            continue

        if forced_maxima is None:
            # ── MASTER: Búsqueda de picos igual a analisis_por_track_integrado ──
            cut_start = start_sample_noise + i * muestras_pulso
            cut_end   = cut_start + muestras_pulso
            if cut_end > len(env_recortada):
                cut_end = len(env_recortada)
            if cut_start >= len(env_recortada):
                break

            local_seg = env_recortada[cut_start:cut_end]
            if local_seg.size == 0:
                continue

            rel_max    = int(np.argmax(local_seg))
            max_sample = cut_start + rel_max
            max_value  = env_recortada[max_sample]

            if max_value < peak_search_threshold:
                print(f"    -> Ventana #{window_number}: pico {max_value:.4e} < umbral {peak_search_threshold:.4e}. Omitida.")
                continue

            seg_start = max_sample - pre_samples
            seg_end   = max_sample + post_samples
            if seg_start < 0 or seg_end > len(env_recortada):
                continue
                
            # Distancia mínima con el pico anterior
            if last_valid_i != -1:
                prev_idx = maxima_detectados[last_valid_i]
                prev_val = env_recortada[prev_idx]
                if abs(max_sample - prev_idx) < min_dist_samples:
                    # Si están muy cerca, conservar el mayor
                    if max_value > prev_val:
                        maxima_detectados[last_valid_i] = None
                        segmentos_full[last_valid_i] = None
                    else:
                        continue # descartar actual

            maxima_detectados[i] = int(max_sample)
            segmentos_full[i] = env_recortada[seg_start:seg_end].copy()
            last_valid_i = i

        else:
            # ── SLAVE: hereda los picos exactos del Master ──
            if i >= len(forced_maxima):
                break
            max_sample = forced_maxima[i]
            if max_sample is None:
                continue

            seg_start = max_sample - pre_samples
            seg_end   = max_sample + post_samples
            if seg_start < 0 or seg_end > len(env_recortada):
                continue

            maxima_detectados[i] = int(max_sample)
            segmentos_full[i] = env_recortada[seg_start:seg_end].copy()

    # Extraer solo las válidas para el return
    centros_metronomo = []
    segmentos = []
    for i in range(n_pulsos):
        if maxima_detectados[i] is not None:
            centros_metronomo.append(maxima_detectados[i])
            segmentos.append(segmentos_full[i])

    return centros_metronomo, segmentos, maxima_detectados

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
    
    noise_t0 = t_recortada[0]
    noise_t1 = noise_t0 + noise_seconds
    plt.axvspan(noise_t0, noise_t1, color='violet', alpha=0.75, label=f"Ruido Inicial ({noise_seconds}s)")

    plt.plot(t_recortada, env_recortada, color="Blue", linewidth=1.5, linestyle='-', alpha=0.9, label="Envolvente")

    offset_start = t_recortada[0] + float(start_sample_noise) / samplerate
    duracion_analizable_grafico = len(env_recortada) - start_sample_noise
    n_pulsos = math.ceil(duracion_analizable_grafico / muestras_pulso)

    excluded_set_plot = set(excluded_windows) if excluded_windows else set()
    spans = []

    for i in range(n_pulsos):
        # Ventana EMPIEZA en offset_start + i*T, dura un periodo completo
        # (igual que la extracción: cut_start = start_sample_noise + i*muestras_pulso)
        win_start_t = offset_start + i * periodo
        win_end_t   = win_start_t + periodo
        beat_t      = win_start_t + periodo / 2.0  # beat en el centro visual

        window_number = i + 1
        color = "red" if window_number in excluded_set_plot else "orange"
        alpha = 0.2 if window_number in excluded_set_plot else 0.05

        if win_end_t > t_recortada[0] and win_start_t < t_recortada[-1]:
            # Línea divisora entre ventanas
            plt.axvline(x=win_start_t, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
            # Sombreado de la ventana
            span = plt.axvspan(win_start_t, win_end_t, color=color, alpha=alpha)
            spans.append((window_number, win_start_t, win_end_t, span))
            # Marca del beat (centro de la ventana)
            plt.axvline(x=beat_t, color="purple", linestyle=":", alpha=0.4, linewidth=0.8)

    if len(centros_metronomo) > 0:
        t_centers = [t_recortada[idx] for idx in centros_metronomo if idx < len(t_recortada)]
        v_env     = [env_recortada[idx] for idx in centros_metronomo if idx < len(env_recortada)]
        plt.scatter(t_centers, v_env, color='green', s=60, zorder=5, label='Pico detectado')

    
    fig = plt.gcf()
    ax = plt.gca()
    if show_plot:
        plt.title(f"Señal y Ventanas (Click para excluir/incluir) - {filename}")
    else:
        plt.title(f"Señal y Ventanas (Beat al Centro) - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [V]")
    max_y_val = np.max(env_recortada) if len(env_recortada) > 0 else 1.3
    plt.ylim(-0.05 * max_y_val, max_y_val * 1.5)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='upper right')
    
    print_progress_bar(0, 1, prefix='Graficando recortes:', suffix='...', length=20)
    plt.savefig(out_rec, dpi=300, bbox_inches='tight')
    print_progress_bar(1, 1, prefix='Graficando recortes:', suffix='OK', length=20)

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
                        span.set_alpha(0.05)
                    else:
                        excluded_set_plot.add(window_number)
                        span.set_color("red")
                        span.set_alpha(0.2)
                    fig.canvas.draw_idle()
                    break
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=False)
        # Fuerza a pausar el script hasta que cierres el gráfico (compatible con PySide6)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
    
    plt.close(fig)
        
    return sorted(list(excluded_set_plot))

# ---------------------- NUEVA FUNCIÓN: Overlay de Músculos ---------------------
def _plot_muscle_overlay(measure_name, channels_dict, out_dir, master_name=None, normalize_all=False):
    all_files = set()
    for c_data in channels_dict.values():
        all_files.update(c_data.keys())
        
    canales_config = config_mgr.get("canales") if config_mgr else {}
    fallback_colors = {'canal_0': 'blue', 'canal_1': 'orange', 'canal_2': 'green', 'canal_3': 'red'}

    for fname in all_files:
        plt.figure(figsize=(10, 6))
        
        found_any = False
        sorted_chans = sorted(channels_dict.keys())
        
        # 1. Encontrar el desplazamiento de tiempo (time_shift) del Master
        time_shift = 0.0
        if master_name and master_name in channels_dict and fname in channels_dict[master_name]:
            master_t = channels_dict[master_name][fname]['pulse_time']
            master_y = channels_dict[master_name][fname]['mean_pulse']
            
            # Ignorar el 10% de los bordes para evitar detectar artefactos de filtrado como el pico
            margin = int(0.1 * len(master_y))
            if margin > 0 and len(master_y) > 2 * margin:
                center_y = master_y[margin:-margin]
                peak_idx = int(np.argmax(center_y)) + margin
            else:
                peak_idx = int(np.argmax(master_y))
                
            time_shift = master_t[peak_idx]
        
        # --- Calcular la amplitud máxima de los slaves y del master para normalizar ---
        max_slave_amp = 0.0
        master_max_amp = 1.0
        if master_name and master_name in channels_dict and fname in channels_dict[master_name]:
            master_y_raw = np.array(channels_dict[master_name][fname]['mean_pulse'])
            master_max_amp = np.max(master_y_raw) - np.min(master_y_raw)
            
        for ch in sorted_chans:
            if ch != master_name and fname in channels_dict[ch]:
                y_slave = np.array(channels_dict[ch][fname]['mean_pulse'])
                m_amp = np.max(y_slave) - np.min(y_slave)
                if m_amp > max_slave_amp:
                    max_slave_amp = m_amp
                    
        scale_factor = 1.0
        if max_slave_amp > 0 and master_max_amp > 0:
            scale_factor = max_slave_amp / master_max_amp

        # 2. Graficar todos los canales con el tiempo corregido y offset restado
        max_y_overlay = 0
        for ch in sorted_chans:
            if fname in channels_dict[ch]:
                data = channels_dict[ch][fname]
                t = np.array(data['pulse_time']) - time_shift
                y = np.array(data['mean_pulse'])
                err = np.array(data.get('pulse_err', np.zeros_like(y)))
                
                # Offset: para asegurar que inicie desde el 0 en el eje Y
                # aplicando el equivalente a un pasa altos ideal sobre la línea base
                y_min = np.min(y)
                y = y - y_min
                
                if normalize_all:
                    y_max = np.max(y)
                    if y_max > 0:
                        y = y / y_max
                        err = err / y_max
                else:
                    # Normalizar la señal líder a la escala de los esclavos (solo si no se normaliza todo al 100%)
                    if ch == master_name and scale_factor != 1.0:
                        y = y * scale_factor
                        err = err * scale_factor
                    
                if np.max(y + err) > max_y_overlay:
                    max_y_overlay = np.max(y + err)
                
                ch_idx_str = ch.replace('canal_', '')
                conf_key = f"Canal {ch_idx_str}"
                ch_conf = canales_config.get(conf_key, {})
                
                lbl = ch_conf.get("musculo", f"Canal {ch_idx_str}")
                if ch == master_name:
                    lbl += " (Master Normalizado)"
                col = ch_conf.get("color_hex", fallback_colors.get(ch, 'gray'))
                
                if ch == 'canal_3':
                    lbl = ch_conf.get("musculo", "Micrófono")
                    col = 'red'
                
                plt.plot(t, y, label=lbl, color=col, linewidth=2, alpha=0.8)
                plt.fill_between(t, y - err, y + err, color=col, alpha=0.2)
                found_any = True
                
        if found_any:
            if normalize_all:
                plt.title(f"Patrón Muscular Sincronizado (Normalizado) - {measure_name} - {fname}")
                plt.ylabel("Amplitud Normalizada [0-1]")
            else:
                plt.title(f"Patrón Muscular Sincronizado - {measure_name} - {fname}")
                plt.ylabel("Amplitud [µV]")
            
            plt.xlabel("Tiempo respecto al pico de la señal de micrófono [s]")
            
            # Dibujar línea exactamente en 0
            plt.axvline(x=0, color='gray', linestyle='--', alpha=0.8, label="Pico señal de micrófono")
            
            plt.legend(loc='upper right', fontsize=8)
            plt.grid(True, alpha=0.5)
            plt.ylim(bottom=0, top=max_y_overlay * 1.2 if max_y_overlay > 0 else 1.0)
            
            # Forzar simetría en el eje X (Tiempo) respecto al cero
            try:
                current_xlims = plt.gca().get_xlim()
                max_x = max(abs(current_xlims[0]), abs(current_xlims[1]))
                plt.xlim(-max_x, max_x)
            except:
                pass
            
            name_clean = os.path.splitext(fname)[0]
            path = os.path.join(out_dir, f"patron_muscular_{name_clean}.png")
            plt.savefig(path, dpi=300, bbox_inches='tight')
            print(f"Generado gráfico patrón: {path}")
            plt.close()

# ---------------------- Export results ---------------------
def export_results_for_file(out_dir, filename, resultados_entry):
    os.makedirs(out_dir, exist_ok=True)
    export = {}
    keys = ['mean_pulse', 'pulse_time', 'pulse_err', 'amp_uncertainty',
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
                       show_overlay=True, show_amplitude=True, normalize_overlay=False):
    
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
        if normalize_overlay:
            ax_ov.set_ylabel('Normalizado')
        else:
            ax_ov.set_ylabel('Unidades Arbitrarias')
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
    normalize_overlay=False,
    # --- ARGUMENTOS NUEVOS PARA ALINEACIÓN FORZADA ---
    dict_shifts_externos=None, # Diccionario { 'archivo.wav': [shift1, shift2...] }
    indices_validos_externos=None # Pasó a ser forced_maxima
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
        ganancia = 495.0
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
                    
            import json
            meta_path = os.path.join(carpeta, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f_meta:
                    md_ch = json.load(f_meta)
                    if 'resistencia_ohm' in md_ch:
                        res_ohm = float(md_ch['resistencia_ohm'])
                        ganancia = 1.0 + (49400.0 / res_ohm)
        except Exception:
            pass

        signal_normalized, samplerate = _read_wav_mono(filepath)
        raw_signal = signal_normalized * calibration_factor 
        signal = (raw_signal / ganancia) * 1e6
        print(f"[Calibración] Factor: {calibration_factor:.4f}, Ganancia: {ganancia:.1f} -> Convertido a µV")
        
        # Offset
        ns_samples = int(noise_seconds * samplerate)
        if ns_samples > 0 and ns_samples < len(signal):
            dc_offset = np.mean(signal[:ns_samples])
            signal = signal - dc_offset
            print(f"[Offset] Restado nivel DC base: {dc_offset:.5f} V")

        # Pasa-Altos
        if highpass_cutoff_hz is not None and highpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                cutoff_hp_usar = highpass_cutoff_hz
                
                if cutoff_hp_usar >= nyquist:
                    cutoff_hp_usar = nyquist * 0.99
                    print(f"ADVERTENCIA: Frecuencia pasa-altos ({highpass_cutoff_hz} Hz) excede Nyquist ({nyquist} Hz). Ajustando a {cutoff_hp_usar:.2f} Hz para aplicar el filtro.")
                
                b, a = butter(4, cutoff_hp_usar / nyquist, btype='high')
                signal = filtfilt(b, a, signal)
                print(f"[Filtro] Pasa-Altos aplicado a {cutoff_hp_usar:.2f} Hz")
            except Exception as e: print(f"Error filtro HP: {e}")

        # Filtro Notch
        if apply_notch_filter:
            try:
                b, a = iirnotch(50.0, 30.0, fs=samplerate)
                signal = filtfilt(b, a, signal)
            except Exception: pass

        signal_unfiltered = signal.copy()

        # Filtro Low-pass
        if lowpass_cutoff_hz is not None and lowpass_cutoff_hz > 0:
            try:
                nyquist = 0.5 * samplerate
                cutoff_usar = lowpass_cutoff_hz
                
                if cutoff_usar >= nyquist:
                    cutoff_usar = nyquist * 0.99
                    print(f"ADVERTENCIA: Frecuencia pasa-bajos ({lowpass_cutoff_hz} Hz) excede Nyquist ({nyquist} Hz). Ajustando a {cutoff_usar:.2f} Hz para aplicar el filtro.")
                
                b, a = butter(4, cutoff_usar / nyquist, btype='low')
                signal = filtfilt(b, a, signal)
                print(f"[Filtro] Aplicado filtro pasa-bajos a {cutoff_usar:.2f} Hz.")
            except Exception as e:
                print(f"Error filtro LP: {e}")
        
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
        
        # --- Obtener maximos forzados (Slave usa los del Master) ---
        maximos_forzados = None
        if indices_validos_externos is not None and filename in indices_validos_externos:
            maximos_forzados = indices_validos_externos[filename]

        interactive_excluded = list(excluded_windows) if excluded_windows else []
        
        # IGUAL a analisis_por_track_integrado: ventana ASIMETRICA 40% pre / 60% post
        pre_samples  = int(round(0.4 * periodo * samplerate))
        post_samples = int(round(0.6 * periodo * samplerate))
        
        # --- LOOP INTERACTIVO ---
        while True:
            # Umbral de pico: la MEDIA del ruido inicial (igual a referencia)
            peak_threshold = float(umbral) if (umbral is not None and umbral > 0) else 0.0
            print(f"[Análisis] Usando umbral de búsqueda de picos dinámico: {peak_threshold:.4f}")
            
            centros_metronomo, segmentos, valid_indices_local = _cortar_centrado_en_beat(
                np.abs(env_recortada), start_sample_noise, muestras_pulso,
                pre_samples, post_samples,
                peak_search_threshold=peak_threshold,
                n_pulsos_manual=n_pulsos_manual, excluded_windows=interactive_excluded,
                forced_maxima=maximos_forzados
            )

            if len(segmentos) == 0:
                print(f"{filename}: no se extrajeron segmentos. Omitido.")
                break

            segmentos_rs, target_len = _resample_segments(segmentos, resample_len)
            
            # --- LÓGICA MASTER / SLAVE (SHIFTS) ---
            shifts_to_save = []
            
            # Caso SLAVE: Tenemos shifts externos
            if dict_shifts_externos is not None and filename in dict_shifts_externos:
                if not show_interactive_plot: print(f"[Alineación] MODO SLAVE: Aplicando alineación forzada...")
                forced_shifts = dict_shifts_externos[filename]
                segmentos_rs = _aplicar_alineacion_forzada(segmentos_rs, forced_shifts)
                shifts_to_save = forced_shifts 
                
            # Caso MASTER: Calculamos la mejor alineación con SUAVIZADO
            else:
                if len(segmentos_rs) > 1:
                    if not show_interactive_plot: print(f"[Alineación] MODO MASTER: Calculando mejor alineación por SILUETA...")
                    segmentos_rs, shifts_calculated = _alinear_por_lider_calculo(segmentos_rs, samplerate)
                    shifts_to_save = shifts_calculated
                else:
                    shifts_to_save = [0] * len(segmentos_rs)

            segmentos_norm, pulso_promedio, pulso_sigma, pulso_err, Np = _compute_pulse_stats(segmentos_rs)

            if (sigma_est is None) or (umbral is None):
                sigma_est, umbral = _fallback_umbral(segmentos_norm, pulso_promedio, factor_umbral)

            max_amp = np.max(pulso_promedio)
            snr_manual = max_amp / umbral if (umbral is not None and umbral > 0) else np.inf

            pre_w_sec  = 0.4 * periodo  # igual que analisis_por_track
            post_w_sec = 0.6 * periodo
            t_pulso = np.linspace(-pre_w_sec, post_w_sec, target_len, endpoint=False)

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

            new_excluded = interactive_excluded
            is_master = (indices_validos_externos is None)
            if mostrar_recortes and is_master:
                new_excluded = _plot_recortes(t_recortada, signal_recortada, env_recortada, noise_seconds,
                               start_sample_noise, samplerate, centros_metronomo, periodo, muestras_pulso, out_rec, final_plot_title, 
                               excluded_windows=interactive_excluded, show_plot=show_interactive_plot,
                               signal_original_unfiltered=signal_unfiltered[mask])

                if show_interactive_plot and set(new_excluded) != set(interactive_excluded):
                    interactive_excluded = list(new_excluded)
                    print("Recalculando con nuevas exclusiones...")
                    continue

            promedios_globales.append(pulso_promedio)
            tiempos_globales.append(t_pulso)
            nombres_globales.append(filename)

            resultados[filename] = {
                'mean_pulse': pulso_promedio,
                'pulse_time': t_pulso,
                'pulse_err': pulso_err,
                'amp_uncertainty': amp_uncertainty,
                'segmentos_rs': segmentos_rs,
                'periodo': periodo,
                'umbral': umbral,
                'shifts': shifts_to_save,
                'valid_indices': valid_indices_local, # IMPORTANTE: Pasar indices al GUI
                'interactive_excluded_windows': interactive_excluded
            }

            export_results_for_file(out_dir, filename, resultados[filename])
            break # Sale del loop interactivo si no hubo cambios

        plt.close('all') # --- Limpieza forzada ---

    if mostrar_tabla and promedios_globales:
        _comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados, nombre_salida, normalize_overlay=normalize_overlay)
    
    return resultados

# ---------------------- GUI Classes (PySide6) ----------------------
class ProcessingOptionsDialog(QDialog):
    def __init__(self, medicion_dir=None, master_dir=None, slave_dirs=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Configuración de Análisis")
        self.resize(600, 700)
        self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")
        
        self.result = None
        self.medicion_dir = medicion_dir
        self.slave_checkboxes = []
        
        main_layout = QVBoxLayout(self)
        
        # Selección de Canales
        if self.medicion_dir:
            chan_group = QGroupBox("Selección de Canales")
            chan_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; }")
            chan_layout = QVBoxLayout(chan_group)
            
            self.subdirs = [os.path.join(medicion_dir, d) for d in os.listdir(medicion_dir) if os.path.isdir(os.path.join(medicion_dir, d)) and d.startswith('canal_')]
            self.subdirs.sort()
            nombres_canales = [os.path.basename(d) for d in self.subdirs]
            
            master_layout = QHBoxLayout()
            master_layout.addWidget(QLabel("Músculo Líder (Master):"))
            self.cmb_master = QComboBox()
            self.cmb_master.addItems(nombres_canales)
            
            # Buscar "canal_0" y ponerlo como predeterminado
            idx_ch0 = self.cmb_master.findText("canal_0")
            if idx_ch0 >= 0:
                self.cmb_master.setCurrentIndex(idx_ch0)
                
            self.cmb_master.setStyleSheet("background-color: #111; color: #fff;")
            master_layout.addWidget(self.cmb_master)
            chan_layout.addLayout(master_layout)
            
            chan_layout.addWidget(QLabel("Músculos Esclavos (Slaves):"))
            for ch in nombres_canales:
                chk = QCheckBox(ch)
                chk.setChecked(True)
                self.slave_checkboxes.append((ch, chk))
                chan_layout.addWidget(chk)
                
            main_layout.addWidget(chan_group)
            
        # Parámetros...
        form_group = QGroupBox("Parámetros")
        form_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        
        self.form_layout = QFormLayout(form_group)
        self.entries = {}
        
        self._add_entry("Frec. Mínima (Hz):", "0", "frecuenciaminima")
        self._add_entry("Frec. Máxima (Hz):", "1000", "frecuenciamaxima")
        self._add_entry("Longitud Resample (pts):", "", "resample_len", hint="Opcional. Ej: 1000")
        
        # Agregamos la entrada para excluir ventanas (excluir la primera por defecto)
        self._add_entry("Ventanas a Excluir:", "1", "excluded_windows", hint="Ej: 1, 3, 5")

        self.chk_apply_envelope = QCheckBox("Aplicar Envolvente (Hilbert)")
        self.chk_apply_envelope.setChecked(True)
        self.form_layout.addRow(self.chk_apply_envelope)
        
        self._add_entry("Suavizado Env (ms):", "250", "smooth_ms", hint="0 para desactivar")
        
        self.chk_apply_notch = QCheckBox("Aplicar Filtro Notch (50Hz)")
        self.chk_apply_notch.setChecked(True)
        self.form_layout.addRow(self.chk_apply_notch)
        
        self.chk_dark_mode = QCheckBox("Gráficos en Tema Oscuro (Fondo Negro)")
        self.chk_dark_mode.setChecked(True)
        self.form_layout.addRow(self.chk_dark_mode)
        
        self._add_entry("Filtro Low-pass (Hz):", "500", "lowpass_cutoff_hz", hint="Vacío para no usar")
        self._add_entry("Filtro High-pass (Hz):", "20", "highpass_cutoff_hz", hint="Vacío para no usar")

        # Visualizaciones
        self.chk_indiv = QCheckBox("Mostrar Recortes Individuales en Promedio")
        self.chk_indiv.setChecked(True)
        self.form_layout.addRow(self.chk_indiv)

        self.chk_recortes = QCheckBox("Generar PNG de Recortes (Interactivo)")
        self.chk_recortes.setChecked(True)
        self.form_layout.addRow(self.chk_recortes)
        
        self.chk_spec = QCheckBox("Generar Espectrograma del Promedio")
        self.chk_spec.setChecked(True)
        self.form_layout.addRow(self.chk_spec)

        self.chk_table = QCheckBox("Generar Comparativa Final (Overlay/Amplitud)")
        self.chk_table.setChecked(True)
        self.form_layout.addRow(self.chk_table)
        
        self.chk_normalize_overlay = QCheckBox("Normalizar Patrón (Visualización de Desfases)")
        self.chk_normalize_overlay.setChecked(False)
        self.form_layout.addRow(self.chk_normalize_overlay)
        
        self.chk_rand_color = QCheckBox("Usar Colores Aleatorios")
        self.chk_rand_color.setChecked(False)
        self.form_layout.addRow(self.chk_rand_color)
        
        self._add_entry("Color Fijo (si no aleatorio):", "blue", "color_fijo")

        main_layout.addWidget(form_group)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.btn_run_interactive = QPushButton("Procesar Interactivo")
        self.btn_run_interactive.setStyleSheet("QPushButton { background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px 20px; border-radius: 3px; }")
        self.btn_run_interactive.clicked.connect(lambda: self.on_ok(interactivo=True))
        btn_layout.addWidget(self.btn_run_interactive)
        
        self.btn_run_fast = QPushButton("Procesar Rápido")
        self.btn_run_fast.setStyleSheet("QPushButton { background-color: #ff00ff; color: #fff; font-weight: bold; padding: 10px 20px; border-radius: 3px; }")
        self.btn_run_fast.clicked.connect(lambda: self.on_ok(interactivo=False))
        btn_layout.addWidget(self.btn_run_fast)
        
        main_layout.addLayout(btn_layout)

    def _add_entry(self, label_text, default_val, key, hint=None):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        entry = QLineEdit(default_val)
        entry.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444;")
        layout.addWidget(entry)
        
        if hint:
            lbl_hint = QLabel(hint)
            lbl_hint.setStyleSheet("color: #888; font-size: 10px;")
            layout.addWidget(lbl_hint)
            
        self.form_layout.addRow(label_text, widget)
        self.entries[key] = entry

    def on_ok(self, interactivo=True):
        try:
            bpm_val = 50.0
            n_pulsos_val = 10
            noise_val = 2.0
            
            if hasattr(self, 'medicion_dir') and self.medicion_dir:
                import json
                meta_path = os.path.join(self.medicion_dir, "metadata.json")
                if not os.path.exists(meta_path):
                    meta_path = os.path.join(self.medicion_dir, "canal_0", "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r') as f:
                            meta = json.load(f)
                            # CORRECTO: usar 'noise_seconds' igual que analisis_por_track_integrado
                            if 'bpm' in meta: bpm_val = float(meta['bpm'])
                            if 'pulse_count' in meta: n_pulsos_val = int(meta['pulse_count'])
                            if 'noise_seconds' in meta: noise_val = float(meta['noise_seconds'])
                            elif 'ruido_inicial_segundos' in meta: noise_val = float(meta['ruido_inicial_segundos'])
                            print(f"[Auto-Config] BPM={bpm_val}, Pulsos={n_pulsos_val}, Ruido={noise_val}s")
                    except Exception as e:
                        print(f"No se pudo leer metadata para Auto-Config: {e}")

            self.result = {
                "bpm": bpm_val,
                "n_pulsos_manual": n_pulsos_val,
                "noise_seconds": noise_val,
                "tiempoinicial": 0.0,
                "tiempofinal": 99999.0,
                "factor_umbral": 6.0,
                
                "frecuenciaminima": float(self.entries["frecuenciaminima"].text()),
                "frecuenciamaxima": float(self.entries["frecuenciamaxima"].text()),
                
                "resample_len": int(self.entries["resample_len"].text()) if self.entries["resample_len"].text() else None,
                "smooth_ms": float(self.entries["smooth_ms"].text()),
                
                "excluded_windows": [int(x.strip()) for x in self.entries["excluded_windows"].text().split(',') if x.strip().isdigit()] if self.entries["excluded_windows"].text() else [],
                
                "lowpass_cutoff_hz": float(self.entries["lowpass_cutoff_hz"].text()) if self.entries["lowpass_cutoff_hz"].text() else None,
                "highpass_cutoff_hz": float(self.entries["highpass_cutoff_hz"].text()) if self.entries["highpass_cutoff_hz"].text() else None,

                "apply_envelope": self.chk_apply_envelope.isChecked(),
                "apply_notch_filter": self.chk_apply_notch.isChecked(),
                
                "mostrar_individuales": self.chk_indiv.isChecked(),
                "mostrar_recortes": self.chk_recortes.isChecked(),
                "show_interactive_plot": interactivo and self.chk_recortes.isChecked(),
                "mostrar_espectrograma": self.chk_spec.isChecked(),
                "mostrar_tabla": self.chk_table.isChecked(),
                "normalizar_overlay": self.chk_normalize_overlay.isChecked(),
                "colores_aleatorios": self.chk_rand_color.isChecked(),
                "colorgrafico": self.entries["color_fijo"].text(),
                "tema_oscuro": self.chk_dark_mode.isChecked(),
            }
            
            if hasattr(self, 'medicion_dir') and self.medicion_dir:
                master_name = self.cmb_master.currentText()
                self.result["selected_master_name"] = master_name
                slaves = []
                for name, chk in self.slave_checkboxes:
                    if chk.isChecked() and name != master_name:
                        slaves.append(name)
                self.result["selected_slaves_names"] = slaves
                
            self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Error de Validación", f"Por favor verifica los valores numéricos.\n{e}")

class ComparativeOptionsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Comparación")
        self.resize(400, 300)
        
        self.mediciones = []
        self.BASE_DIR = ""
        
        layout = QVBoxLayout(self)
        
        self.combo_canal = QComboBox()
        layout.addWidget(QLabel("Canal común:"))
        layout.addWidget(self.combo_canal)
        
        self.chk_ov = QCheckBox("Overlay")
        self.chk_ov.setChecked(True)
        layout.addWidget(self.chk_ov)
        
        self.chk_amp = QCheckBox("Amplitud Max")
        self.chk_amp.setChecked(True)
        layout.addWidget(self.chk_amp)
        
        btn = QPushButton("LANZAR")
        btn.clicked.connect(self.lanzar)
        layout.addWidget(btn)

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
                           show_overlay=self.chk_ov.isChecked(), show_amplitude=self.chk_amp.isChecked())

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