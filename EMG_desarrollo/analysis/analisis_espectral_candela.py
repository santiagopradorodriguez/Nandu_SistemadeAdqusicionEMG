# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Análisis espectral, espectrogramas RGB, FFT y PSD en habla submáximal.
#              - Sin filtro pasa-banda (preservación de todo el rango espectral).
#              - Cancelador adaptativo NLMS para 50 Hz y armónicos.
#              - Colores oficiales: Ch0 Rojo (Digástrico), Ch1 Verde (Zigo), Ch2 Amarillo (Orbicular).
# ==============================================================================

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
import soundfile as sf
import shutil
from scipy.signal import find_peaks, spectrogram, welch, butter, filtfilt
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import ScalarFormatter

# Configuración de rutas internas del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
emg_desarrollo_dir = os.path.abspath(os.path.join(script_dir, ".."))
if emg_desarrollo_dir not in sys.path:
    sys.path.append(emg_desarrollo_dir)
if script_dir not in sys.path:
    sys.path.append(script_dir)

try:
    from deep_learning.binarizacion.analisis_trevisan import aplicar_filtro_adaptativo_nlms
except ImportError:
    try:
        from binarizacion.analisis_trevisan import aplicar_filtro_adaptativo_nlms
    except ImportError:
        try:
            from numba import njit
            @njit(fastmath=True)
            def _nlms_loop_local(signal, X_ref, mu, eps):
                N, n_weights = X_ref.shape
                w = np.zeros(n_weights)
                e = np.zeros(N)
                for n in range(N):
                    x_n = X_ref[n]
                    y_n = np.dot(w, x_n)
                    err = signal[n] - y_n
                    e[n] = err
                    norm_x = np.dot(x_n, x_n) + eps
                    w += (mu / norm_x) * err * x_n
                return e
        except Exception:
            def _nlms_loop_local(signal, X_ref, mu, eps):
                N, n_weights = X_ref.shape
                w = np.zeros(n_weights)
                e = np.zeros(N)
                for n in range(N):
                    x_n = X_ref[n, :]
                    y_n = np.dot(w, x_n)
                    err = signal[n] - y_n
                    e[n] = err
                    norm_x = np.dot(x_n, x_n) + eps
                    w += (mu / norm_x) * err * x_n
                return e

        def aplicar_filtro_adaptativo_nlms(signal, samplerate, f0=50.0, armonicos=(50, 100, 150, 200, 250, 300, 350, 400), mu=0.02, eps=1e-10):
            sig = np.asarray(signal, dtype=np.float64)
            N = len(sig)
            if N < 10:
                return sig
            t = np.arange(N) / float(samplerate)
            valid_harmonics = [fh for fh in armonicos if fh < (samplerate * 0.5)]
            if not valid_harmonics:
                return sig
            n_weights = 2 * len(valid_harmonics)
            X_ref = np.zeros((N, n_weights), dtype=np.float64)
            for k, fh in enumerate(valid_harmonics):
                omega = 2.0 * np.pi * fh * t
                X_ref[:, 2 * k] = np.sin(omega)
                X_ref[:, 2 * k + 1] = np.cos(omega)
            return _nlms_loop_local(sig, X_ref, float(mu), float(eps))

# Paleta cromática oficial solicitada por el usuario
COLORES_VOCALES = {
    'A': '#d62728',
    'E': '#1f77b4',
    'I': '#2ca02c',
    'O': '#9467bd',
    'U': '#ffb700'
}

# Código cromático estricto: Ch0 Rojo, Ch1 Verde, Ch2 Amarillo
COLORES_CANALES = {
    0: '#d62728',  # Rojo: Anterior Belly (Digástrico / Apertura)
    1: '#2ca02c',  # Verde: Zygomaticus Major (Zigo / Sonrisa)
    2: '#ffb700'   # Amarillo: Orbicularis Oris (Orbicular / Labial)
}

NOMBRES_CANALES = {
    0: 'Ch0: Digástrico (Anterior Belly)',
    1: 'Ch1: Zigo (Zygomaticus Major)',
    2: 'Ch2: Orbicularis (Orbicularis Oris)'
}


def fijar_semilla(seed=42):
    """Garantiza reproducibilidad estricta universal en NumPy, random y algoritmos."""
    random.seed(seed)
    np.random.seed(seed)


def leer_wav_mono(filepath):
    """Lee archivo de audio WAV en mono con precisión de punto flotante de 64 bits."""
    signal, sr = sf.read(filepath)
    if signal.ndim > 1:
        signal = signal[:, 0]
    return np.asarray(signal, dtype=np.float64), sr


def get_interpulse_noise(segment, initial_noise):
    """Calcula el nivel de ruido dinámico interpulso depurado de espigas mediante IQR.

    Maneja valores NaN/inf de forma robusta para no romper el análisis si alguna ventana
    presenta artefactos aislados o segmentos parcialmente corruptos.
    """
    if segment is None:
        return float(np.nan_to_num(initial_noise, nan=0.0, posinf=0.0, neginf=0.0))

    arr = np.asarray(segment, dtype=np.float64)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float(np.nan_to_num(initial_noise, nan=0.0, posinf=0.0, neginf=0.0))

    if arr.size < 10:
        base_noise = float(np.nan_to_num(initial_noise, nan=0.0, posinf=0.0, neginf=0.0))
        return float(np.mean(arr)) if np.isfinite(np.mean(arr)) else base_noise

    abs_n = np.abs(arr)
    q1 = np.percentile(abs_n, 25)
    q3 = np.percentile(abs_n, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid = abs_n[abs_n <= upper_bound]
    if valid.size < 3:
        valid = abs_n

    curr_mean = float(np.mean(valid))
    base_noise = float(np.nan_to_num(initial_noise, nan=0.0, posinf=0.0, neginf=0.0))
    if base_noise > 0 and np.isfinite(curr_mean) and (curr_mean / base_noise) > 5.0:
        return base_noise
    return curr_mean if np.isfinite(curr_mean) else base_noise


def acondicionar_senal_canal(signal_raw, fs=2000.0, noise_samples=6000):
    """
    Acondicionamiento oficial de generador_pca_umap y analisis_trevisan:
    1. Remoción de offset de continua medido en el reposo basal.
    2. Cancelador adaptativo NLMS para fundamental de 50 Hz y armónicos (50 a 400 Hz).
    3. Filtro pasa-altos Butterworth de 4to orden a 20 Hz (corte fisiológico sin distorsión de fase con filtfilt).
    """
    # 1. Remoción de continua constante
    base_dc = np.mean(signal_raw[:noise_samples]) if len(signal_raw) > noise_samples else np.mean(signal_raw)
    sig_dc = signal_raw - base_dc

    # 2. Cancelador adaptativo NLMS para fundamental de 50 Hz y armónicos de red
    armonicos_linea = (50, 100, 150, 200, 250, 300, 350, 400)
    sig_nlms = aplicar_filtro_adaptativo_nlms(
        sig_dc, samplerate=fs, f0=50.0, armonicos=armonicos_linea, mu=0.02
    )

    # 3. Filtro pasa-altos Butterworth a 20 Hz (cadena de acondicionamiento Trevisan / generador_pca_umap)
    nyquist = 0.5 * fs
    cutoff_hp = 20.0
    b_hp, a_hp = butter(4, cutoff_hp / nyquist, btype='high', analog=False)
    sig_clean = filtfilt(b_hp, a_hp, sig_nlms)

    return sig_clean


def calcular_envolvente_rms(sig, fs=2000, smooth_ms=100):
    """Calcula la envolvente suave RMS con ventana Hanning centrada para detección de pulsos."""
    win_len = max(1, int(round((smooth_ms * fs) / 1000.0)))
    if win_len > 1:
        w = np.hanning(win_len)
        w = w / np.sum(w)
        return np.sqrt(np.maximum(np.convolve(sig ** 2, w, mode='same'), 0.0))
    return np.abs(sig)


def calcular_espectrograma(signal_pulse, fs=2000, nperseg=256, noverlap=230, f_max=600.0):
    """
    Calcula el espectrograma STFT de un pulso individual en el rango 0 a f_max Hz.
    Retorna frecuencias f, tiempos t, y matriz de potencia Sxx en unidades físicas.
    """
    f, t, Sxx = spectrogram(
        signal_pulse,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    mask_f = (f >= 0.0) & (f <= f_max)
    return f[mask_f], t, Sxx[mask_f, :]


def construir_espectrograma_coloreado(S0, S1, S2, gamma=0.5):
    """
    Construye el espectrograma compuesto donde cada músculo adopta su color canónico:
    - Canal 0 (Digástrico): Rojo [1.0, 0.0, 0.0]
    - Canal 1 (Zigo): Verde [0.0, 0.90, 0.0]
    - Canal 2 (Orbicularis): Amarillo dorado cálido [1.0, 0.78, 0.0]
    """
    mat_tricanal = np.stack([S0, S1, S2], axis=-1)  # (F, T, 3)
    supremo = np.max(mat_tricanal) + 1e-12

    # Normalización por el Supremo y compresión gamma para balance visual
    mat_norm = np.power(np.clip(mat_tricanal / supremo, 0.0, 1.0), gamma)

    # Síntesis cromática aditiva:
    # R_total = Ch0 + Ch2 (Amarillo aporta rojo completo)
    # G_total = 0.90*Ch1 + 0.78*Ch2 (Amarillo aporta verde controlado para tono dorado cálido sin tinte lima)
    # B_total = 0.0 (canal azul a cero para no contaminar)
    r = np.clip(mat_norm[:, :, 0] * 1.0 + mat_norm[:, :, 2] * 1.0, 0.0, 1.0)
    g = np.clip(mat_norm[:, :, 1] * 0.90 + mat_norm[:, :, 2] * 0.78, 0.0, 1.0)
    b = np.zeros_like(r)

    rgb = np.stack([r, g, b], axis=-1)
    return rgb


def construir_espectrograma_rgb_ortogonal(S0, S1, S2, gamma=0.5):
    """Construye versión RGB ortogonal clásica (R=Ch0 Digástrico, G=Ch1 Zigo, B=Ch2 Orbicularis)."""
    mat_tricanal = np.stack([S0, S1, S2], axis=-1)
    supremo = np.max(mat_tricanal) + 1e-12
    return np.power(np.clip(mat_tricanal / supremo, 0.0, 1.0), gamma)


from scipy.signal import detrend as signal_detrend


def calcular_fft_amplitud(signal_pulse, fs=2000, f_max=600.0, detrend_dc=True):
    """
    Calcula el espectro de amplitud mono-lateral |X(f)| (en uV) hasta f_max.
    Aplica remoción de continua y deriva lineal local (detrend) en la ventana
    para evitar que el offset electroquímico estático de electrodo en 0 Hz aplaste
    la escala vertical de la actividad mioeléctrica biológica (preservando íntegras
    todas las frecuencias dinámicas de 0.5 a 600 Hz).
    """
    N = len(signal_pulse)
    sig_proc = signal_detrend(signal_pulse, type='linear') if detrend_dc else signal_pulse
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    X_fft = np.fft.rfft(sig_proc)
    
    # Amplitud física unilateral: (2/N)*|X(f)| para f > 0, y (1/N)*|X(0)| para f = 0
    amplitud = (2.0 / N) * np.abs(X_fft)
    amplitud[0] = (1.0 / N) * np.abs(X_fft[0])
    
    mask_f = (freqs >= 0.0) & (freqs <= f_max)
    return freqs[mask_f], amplitud[mask_f]


def calcular_psd_welch(signal_pulse, fs=2000, nperseg=512, noverlap=256, f_max=600.0):
    """
    Calcula la Densidad Espectral de Potencia (PSD) mediante el método de Welch.
    Retorna frecuencias, PSD (uV^2/Hz), Frecuencia Mediana (MDF) y Frecuencia Media (MNF).
    """
    f, Pxx = welch(
        signal_pulse,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )
    mask_f = (f >= 0.0) & (f <= f_max)
    f_band = f[mask_f]
    Pxx_band = Pxx[mask_f]

    potencia_total = np.sum(Pxx_band) + 1e-12
    mnf = float(np.sum(f_band * Pxx_band) / potencia_total)

    cumsum_p = np.cumsum(Pxx_band)
    half_p = 0.5 * cumsum_p[-1]
    idx_mdf = np.where(cumsum_p >= half_p)[0]
    mdf = float(f_band[idx_mdf[0]]) if len(idx_mdf) > 0 else mnf

    return f_band, Pxx_band, mdf, mnf


def ejecutar_analisis_completo(session_paths=None, salida_base_dir=None, logger=print):
    """Ejecuta el flujo completo de análisis espectral sobre las 20 tomas de Candela del 2026-09-16."""
    fijar_semilla(seed=42)
    tiempo_inicio = time.time()

    if session_paths is None:
        base_sesiones_dir = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos", "2026-09-16")
        session_paths = sorted([
            os.path.join(base_sesiones_dir, d) for d in os.listdir(base_sesiones_dir)
            if os.path.isdir(os.path.join(base_sesiones_dir, d)) and any(d.startswith(f"{v}_") for v in ['A', 'E', 'I', 'O', 'U'])
        ])
    if salida_base_dir is None:
        salida_base_dir = os.path.join(emg_desarrollo_dir, "resultados", "analisis_espectral_candela_2026-09-16")
    if os.path.exists(salida_base_dir):
        logger(f"Limpiando directorio previo para eliminar figuras y espectrogramas viejos: {salida_base_dir}")
        shutil.rmtree(salida_base_dir)

    subdirs = {
        'resumen': os.path.join(salida_base_dir, "resumen_comparativo_global"),
        'separados': os.path.join(salida_base_dir, "espectros_separados"),
        'rgb': os.path.join(salida_base_dir, "espectros_rgb"),
        'fft': os.path.join(salida_base_dir, "espectros_frecuencia_fft"),
        'psd': os.path.join(salida_base_dir, "espectros_potencia_psd"),
        'correlacion': os.path.join(salida_base_dir, "correlacion_emg_audio")
    }
    for d in subdirs.values():
        os.makedirs(d, exist_ok=True)
    for v in ['A', 'E', 'I', 'O', 'U']:
        os.makedirs(os.path.join(subdirs['separados'], f"vocal_{v}"), exist_ok=True)
        os.makedirs(os.path.join(subdirs['rgb'], f"vocal_{v}"), exist_ok=True)
        os.makedirs(os.path.join(subdirs['fft'], f"vocal_{v}"), exist_ok=True)
        os.makedirs(os.path.join(subdirs['psd'], f"vocal_{v}"), exist_ok=True)
        os.makedirs(os.path.join(subdirs['correlacion'], f"vocal_{v}"), exist_ok=True)

    logger("=" * 80)
    logger("ANALISIS ESPECTRAL MULTIMODAL CANDELA (2026-09-16)")
    logger("Acondicionamiento: Filtro Adaptativo NLMS (50 a 400 Hz) + Pasa-Altos 20 Hz (Trevisan)")
    logger("Visualización: Eje X Lineal (20-600 Hz) | Amplitud FFT fijada en [0.0, 4.0] µV")
    logger("Canales: Ch0 Rojo (Digástrico), Ch1 Verde (Zigo), Ch2 Amarillo (Orbicular)")
    logger("=" * 80)

    todas_tomas = sorted([
        d for d in os.listdir(base_sesiones_dir)
        if os.path.isdir(os.path.join(base_sesiones_dir, d)) and any(d.startswith(f"{v}_") for v in ['A', 'E', 'I', 'O', 'U'])
    ])
    total_tomas = len(todas_tomas)
    print(f"Detectadas {total_tomas} tomas en {base_sesiones_dir}")

    todos_los_pulsos = []

    for idx_toma, nombre_toma in enumerate(todas_tomas):
        toma_path = os.path.join(base_sesiones_dir, nombre_toma)
        vocal = nombre_toma.split('_')[0].upper()

        pct = ((idx_toma + 1) / total_tomas) * 100.0
        logger(f"[Carga] Toma {idx_toma + 1}/{total_tomas} ({pct:.1f}%) - {nombre_toma}")

        meta_file = os.path.join(toma_path, "canal_0", "metadata.json")
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        bpm = float(meta.get('bpm', 30))
        noise_seconds = float(meta.get('noise_seconds', 3.0))
        fs = int(meta.get('sample_rate', 2000))
        resistencia_ohm = float(meta.get('resistencia_ohm', 100.0))
        ganancia = 1.0 + (49400.0 / resistencia_ohm)

        csv_file = os.path.join(toma_path, "grabacion.csv")
        calib_factors = [1.0, 1.0, 1.0, 1.0]
        if os.path.exists(csv_file):
            try:
                df_csv = pd.read_csv(csv_file)
                for ch_i in range(4):
                    col_name = f"Canal {ch_i}"
                    if col_name in df_csv.columns:
                        calib_factors[ch_i] = float(np.max(np.abs(df_csv[col_name].values)))
            except Exception:
                pass

        excl_file = os.path.join(toma_path, "excluded_windows.json")
        excluded_windows = []
        if os.path.exists(excl_file):
            try:
                with open(excl_file, 'r', encoding='utf-8') as f:
                    excluded_windows = json.load(f).get("excluded_windows", [])
            except Exception:
                pass

        senales_clean = []
        envolventes = []
        inits_noise = []
        noise_samples_init = max(10, int(noise_seconds * fs))

        for ch_idx in range(3):
            wav_ch = os.path.join(toma_path, f"canal_{ch_idx}", "grabacion.wav")
            s_raw, _ = leer_wav_mono(wav_ch)
            s_uv = (s_raw * calib_factors[ch_idx] / ganancia) * 1e6

            # Acondicionamiento SIN pasa-banda (solo cancelador adaptativo NLMS)
            s_cond = acondicionar_senal_canal(s_uv, fs=fs, noise_samples=noise_samples_init)
            env_ch = calcular_envolvente_rms(s_cond, fs=fs, smooth_ms=100)
            init_n = np.median(env_ch[:noise_samples_init])

            senales_clean.append(s_cond)
            envolventes.append(env_ch)
            inits_noise.append(init_n)

        wav_mic = os.path.join(toma_path, "canal_3", "grabacion.wav")
        mic_sig, _ = leer_wav_mono(wav_mic)
        mic_win = max(5, int(0.05 * fs))
        mic_env = np.convolve(np.abs(mic_sig), np.ones(mic_win) / mic_win, mode='same')
        deriv_mic = np.gradient(mic_env)
        win_d = max(1, int(fs * 0.05))
        sig_ref_align = np.convolve(deriv_mic, np.ones(win_d) / win_d, mode='same')

        muestras_pulso = int(round((60.0 / bpm) * fs))
        start_sample = int(noise_seconds * fs)
        n_pulsos = int(meta.get('pulse_count', 12))

        pre_samples = int(round(muestras_pulso * 0.40))
        post_samples = int(round(muestras_pulso * 0.60))

        noise_win_samples = max(10, int(muestras_pulso / 4.0))

        for w_idx in range(n_pulsos):
            if (w_idx + 1) in excluded_windows or w_idx in excluded_windows:
                continue

            cut_start = start_sample + w_idx * muestras_pulso
            cut_end = min(len(sig_ref_align), cut_start + muestras_pulso)
            if cut_end - cut_start < muestras_pulso // 2:
                continue

            local_slot = sig_ref_align[cut_start:cut_end]
            if len(local_slot) == 0:
                continue

            rel_max = int(np.argmax(local_slot))
            p_idx = cut_start + rel_max

            p_start = p_idx - pre_samples
            p_end = p_idx + post_samples
            if p_start < 0 or p_end > len(senales_clean[0]):
                continue

            ruidos_c = []
            for ch_i in range(3):
                env_ch = envolventes[ch_i]
                n_start_pre = max(0, int(p_idx - 0.5 * muestras_pulso - noise_win_samples))
                n_end_pre = min(len(env_ch), n_start_pre + noise_win_samples)
                r_pre = get_interpulse_noise(env_ch[n_start_pre:n_end_pre], inits_noise[ch_i])

                n_start_post = min(len(env_ch), int(p_idx + 0.5 * muestras_pulso))
                n_end_post = min(len(env_ch), n_start_post + noise_win_samples)
                r_post = get_interpulse_noise(env_ch[n_start_post:n_end_post], inits_noise[ch_i])
                ruidos_c.append((r_pre + r_post) / 2.0)

            segs_lineales = []
            segs_envolvente = []
            for ch_i in range(3):
                seg_lin = senales_clean[ch_i][p_start:p_end]
                seg_env = envolventes[ch_i][p_start:p_end]

                win_edge = max(3, int(0.12 * len(seg_env)))
                base_pre = max(ruidos_c[ch_i], float(np.median(seg_env[:win_edge])))
                base_post = max(ruidos_c[ch_i], float(np.median(seg_env[-win_edge:])))
                t_ramp = np.linspace(0.0, 1.0, len(seg_env))
                ramp = base_pre + t_ramp * (base_post - base_pre)
                env_depurada = np.maximum(0.0, seg_env - ramp)

                segs_lineales.append(seg_lin)
                segs_envolvente.append(env_depurada)
                
            seg_audio = mic_sig[p_start:p_end]

            todos_los_pulsos.append({
                'toma': nombre_toma,
                'win_idx': w_idx + 1,
                'vocal': vocal,
                'segs_lineales': np.array(segs_lineales),      # (3, 4000) en uV
                'segs_envolvente': np.array(segs_envolvente),  # (3, 4000)
                'seg_audio': seg_audio,
                'fs': fs,
                'ruidos': ruidos_c
            })

    total_contracciones = len(todos_los_pulsos)
    print(f"\nTotal de contracciones segmentadas: {total_contracciones}")

    print("\nAplicando Isolation Forest (10% contaminación, random_state=42) por clase de vocal...")
    vocales_lista = [p['vocal'] for p in todos_los_pulsos]
    mascara_inliers = np.ones(total_contracciones, dtype=bool)
    lista_outliers = []

    for v in ['A', 'E', 'I', 'O', 'U']:
        indices_v = [i for i, voc in enumerate(vocales_lista) if voc == v]
        if len(indices_v) < 6:
            continue

        features_v = []
        for idx_p in indices_v:
            env_3ch = todos_los_pulsos[idx_p]['segs_envolvente']
            resamp_3ch = []
            for ch_i in range(3):
                x_old = np.linspace(0, 1, len(env_3ch[ch_i]))
                x_new = np.linspace(0, 1, 50)
                resamp_3ch.append(np.interp(x_new, x_old, env_3ch[ch_i]))
            features_v.append(np.concatenate(resamp_3ch))

        features_v = np.array(features_v)
        iso = IsolationForest(contamination=0.10, random_state=42)
        preds = iso.fit_predict(features_v)

        for local_i, pred in enumerate(preds):
            global_i = indices_v[local_i]
            p_info = todos_los_pulsos[global_i]
            if pred == -1:
                mascara_inliers[global_i] = False
                lista_outliers.append({
                    'toma': p_info['toma'],
                    'ventana': p_info['win_idx'],
                    'vocal': p_info['vocal'],
                    'motivo': 'Outlier morfologico en envolvente multicanal (Isolation Forest)'
                })

    n_outliers = len(lista_outliers)
    n_validos = np.sum(mascara_inliers)
    logger(f"Outliers detectados y excluidos: {n_outliers} ({n_outliers / total_contracciones * 100:.1f}%)")
    logger(f"Pulsos válidos consolidados para análisis espectral: {n_validos}")

    with open(os.path.join(salida_base_dir, "lista_outliers.json"), 'w', encoding='utf-8') as f:
        json.dump(lista_outliers, f, indent=4, ensure_ascii=False)

    with open(os.path.join(salida_base_dir, "informe_outliers.txt"), 'w', encoding='utf-8') as f:
        f.write("INFORME DE OUTLIERS DETECTADOS (ISOLATION FOREST - 10% CONTAMINACION)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Fecha de analisis: 2026-09-16\n")
        f.write(f"Total contracciones iniciales: {total_contracciones}\n")
        f.write(f"Total outliers excluidos: {n_outliers}\n")
        f.write(f"Total contracciones validas analizadas: {n_validos}\n\n")
        f.write("DETALLE DE CONTRACCIONES EXCLUIDAS:\n")
        f.write("-" * 70 + "\n")
        for o in lista_outliers:
            f.write(f"- Vocal /{o['vocal']}/ | Toma: {o['toma']} | Pulso: {o['ventana']:02d} | Motivo: {o['motivo']}\n")

    # Filtrado por Isolation Forest confirmado por el usuario (N=10 válido cuando hay 2 outliers)
    pulsos_validos = [p for i, p in enumerate(todos_los_pulsos) if mascara_inliers[i]]
    logger(f"Pulsos válidos consolidados para análisis espectral: {len(pulsos_validos)} (excluidos {n_outliers} outliers)")

    logger("\nCalculando matrices espectrales (20-600 Hz: STFT, Compuesto, FFT y PSD)...")

    datos_por_vocal = {v: [] for v in ['A', 'E', 'I', 'O', 'U']}
    f_stft, t_stft = None, None
    f_fft, f_psd = None, None

    for idx_p, p in enumerate(pulsos_validos):
        v = p['vocal']
        fs = p['fs']
        segs = p['segs_lineales']

        stft_res = []
        for ch_i in range(3):
            f_stft, t_stft, Sxx = calcular_espectrograma(segs[ch_i], fs=fs, nperseg=256, noverlap=230, f_max=600.0)
            stft_res.append(Sxx)
        stft_res = np.array(stft_res)
        
        # Audio Spectrogram (up to Nyquist since fs=2000, max=1000)
        f_audio, t_audio, Sxx_audio = calcular_espectrograma(p['seg_audio'], fs=fs, nperseg=128, noverlap=100, f_max=1000.0)

        # Espectrograma coloreado según la paleta solicitada (Rojo, Verde, Amarillo)
        color_img = construir_espectrograma_coloreado(stft_res[0], stft_res[1], stft_res[2])
        # Versión RGB ortogonal estándar
        rgb_img = construir_espectrograma_rgb_ortogonal(stft_res[0], stft_res[1], stft_res[2])

        fft_res = []
        for ch_i in range(3):
            f_fft, mag_fft = calcular_fft_amplitud(segs[ch_i], fs=fs, f_max=600.0)
            fft_res.append(mag_fft)
        fft_res = np.array(fft_res)

        psd_res = []
        mdf_list = []
        mnf_list = []
        for ch_i in range(3):
            f_psd, Pxx, mdf, mnf = calcular_psd_welch(segs[ch_i], fs=fs, nperseg=512, noverlap=256, f_max=600.0)
            psd_res.append(Pxx)
            mdf_list.append(mdf)
            mnf_list.append(mnf)
        psd_res = np.array(psd_res)

        datos_por_vocal[v].append({
            'info': p,
            'stft': stft_res,
            'color_img': color_img,
            'rgb_img': rgb_img,
            'fft': fft_res,
            'psd': psd_res,
            'mdf': mdf_list,
            'mnf': mnf_list,
            'audio_stft': Sxx_audio
        })

    logger("Guardando datos espectrales consolidados en formato NPZ...")
    npz_data = {
        'vocales': np.array([p['info']['vocal'] for v_list in datos_por_vocal.values() for p in v_list]),
        'tomas': np.array([p['info']['toma'] for v_list in datos_por_vocal.values() for p in v_list]),
        'ventanas': np.array([p['info']['win_idx'] for v_list in datos_por_vocal.values() for p in v_list]),
        'f_stft': f_stft,
        't_stft': t_stft,
        'f_fft': f_fft,
        'f_psd': f_psd
    }
    np.savez_compressed(os.path.join(salida_base_dir, "datos_espectrales_consolidados.npz"), **npz_data)

    logger("\nGenerando figuras de evaluación visual (DPI 300)...")

    # Parámetros unificados de visualización en banda fisiológica EMG (20 a 600 Hz) en escala lineal
    F_MIN_LOG = 20.0
    F_MAX_LOG = 600.0
    TICKS_LOG = [20, 100, 200, 300, 400, 500, 600]

    extent_stft = [t_stft[0], t_stft[-1], f_stft[0], f_stft[-1]]
    mask_f_log = (f_fft >= F_MIN_LOG) & (f_fft <= F_MAX_LOG)
    mask_psd_log = (f_psd >= F_MIN_LOG) & (f_psd <= F_MAX_LOG)

    # -------------------------------------------------------------------------
    # Determinación de Escalas Verticales Unificadas Globales
    # -------------------------------------------------------------------------
    all_ffts = np.array([it['fft'] for v_list in datos_por_vocal.values() for it in v_list])  # (N_total, 3, F_fft)
    all_psds = np.array([it['psd'] for v_list in datos_por_vocal.values() for it in v_list])  # (N_total, 3, F_psd)

    # 1. Escala unificada para FFT fijada estrictamente en 0.0 a 4.0 µV según solicitud directa
    y_lim_fft_unificado = 6.0
    y_lim_fft_por_vocal = {v: 6.0 for v in datos_por_vocal.keys()}
    logger(f"  [Escala FFT Unificada Global] Rango fijado estrictamente a: 0.0 - {y_lim_fft_unificado:.1f} µV")

    # 2. Escala unificada para PSD Welch (dB/Hz)
    all_psd_db = 10.0 * np.log10(np.maximum(all_psds, 1e-6))
    p_min_db = float(np.floor(np.percentile(all_psd_db, 0.5) / 5.0) * 5.0)
    p_max_db = float(np.ceil(np.percentile(all_psd_db, 99.8) / 5.0) * 5.0)
    y_lim_psd_unificado = (min(p_min_db, -25.0), max(p_max_db, 10.0))
    logger(f"  [Escala PSD Unificada] Rango fijado a: {y_lim_psd_unificado[0]:.1f} a {y_lim_psd_unificado[1]:.1f} dB/Hz")

    # 5.1 Espectrogramas Separados
    logger("  -> Generando espectrogramas separados por músculo...")
    for v, lista_v in datos_por_vocal.items():
        if not lista_v:
            continue
        v_dir = os.path.join(subdirs['separados'], f"vocal_{v}")

        stft_medio = np.mean([item['stft'] for item in lista_v], axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig.suptitle(f"Espectrogramas Promedio - Vocal /{v}/ (N={len(lista_v)} contracciones)", fontsize=14, fontweight='bold')

        supremo_v = np.max(stft_medio) + 1e-12
        for ch_i in range(3):
            ax = axes[ch_i]
            stft_db = 10.0 * np.log10(np.maximum(stft_medio[ch_i] / supremo_v, 1e-4))
            im = ax.imshow(stft_db, origin='lower', aspect='auto', extent=extent_stft, cmap='magma', vmin=-35, vmax=0)
            ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=11, fontweight='bold', color=COLORES_CANALES[ch_i])
            ax.set_ylabel("Frecuencia [Hz]")
            
            ax.set_ylim(F_MIN_LOG, F_MAX_LOG)
            fig.colorbar(im, ax=ax, label="Potencia [dB]", pad=0.01)

        axes[-1].set_xlabel("Tiempo relativo [s]")
        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"espectrogramas_promedio_vocal_{v}.png"), dpi=300)
        plt.close(fig)

        # Guardar todos los pulsos individuales para espectrogramas separados (STFT por músculo)
        for item in lista_v:
            p_inf = item['info']
            stft_p = item['stft']
            supremo_p = np.max(stft_p) + 1e-12
            fig_sep_p, axes_sep_p = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
            fig_sep_p.suptitle(f"STFT Separado - {p_inf['toma']} Pulso #{p_inf['win_idx']} (Vocal /{v}/)", fontsize=11, fontweight='bold')
            for ch_i in range(3):
                ax_s = axes_sep_p[ch_i]
                stft_db_p = 10.0 * np.log10(np.maximum(stft_p[ch_i] / supremo_p, 1e-4))
                im_s = ax_s.imshow(stft_db_p, origin='lower', aspect='auto', extent=extent_stft, cmap='magma', vmin=-35, vmax=0)
                ax_s.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=9, fontweight='bold', color=COLORES_CANALES[ch_i])
                ax_s.set_ylabel("Frecuencia [Hz]", fontsize=8)
                ax_s.set_ylim(F_MIN_LOG, F_MAX_LOG)
            axes_sep_p[-1].set_xlabel("Tiempo relativo [s]", fontsize=8)
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"stft_separado_{p_inf['toma']}_pulso_{p_inf['win_idx']:02d}.png"), dpi=150)
            plt.close(fig_sep_p)

    # 5.2 Espectrogramas Compuestos Coloreados
    logger("  -> Generando espectrogramas compuestos con código cromático (Rojo/Verde/Amarillo)...")
    for v, lista_v in datos_por_vocal.items():
        if not lista_v:
            continue
        v_dir = os.path.join(subdirs['rgb'], f"vocal_{v}")

        stft_medio = np.mean([item['stft'] for item in lista_v], axis=0)
        comp_medio = construir_espectrograma_coloreado(stft_medio[0], stft_medio[1], stft_medio[2])

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(comp_medio, origin='lower', aspect='auto', extent=extent_stft)
        ax.set_title(f"Espectrograma Compuesto Promedio - Vocal /{v}/\nRojo: Ch0 Digástrico | Verde: Ch1 Zigo | Amarillo: Ch2 Orbicularis", fontsize=12, fontweight='bold')
        ax.set_xlabel("Tiempo relativo [s]")
        ax.set_ylabel("Frecuencia [Hz]")
        ax.set_ylim(F_MIN_LOG, F_MAX_LOG)

        patches = [
            mpatches.Patch(color='#d62728', label='Ch0: Digástrico (Rojo)'),
            mpatches.Patch(color='#2ca02c', label='Ch1: Zigo (Verde)'),
            mpatches.Patch(color='#ffb700', label='Ch2: Orbicularis (Amarillo)'),
            mpatches.Patch(color='#ff7f0e', label='Co-activación Ch0 + Ch2 (Naranja)'),
            mpatches.Patch(color='#a3e635', label='Co-activación Ch1 + Ch2 (Lima/Amarillo)')
        ]
        ax.legend(handles=patches, loc='upper right', facecolor='#111111', edgecolor='white', labelcolor='white', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"espectrograma_compuesto_promedio_vocal_{v}.png"), dpi=300)
        plt.close(fig)

        # Guardar todos los pulsos individuales para espectrogramas compuestos RGB
        for item in lista_v:
            p_inf = item['info']
            fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
            ax_ind.imshow(item['color_img'], origin='lower', aspect='auto', extent=extent_stft)
            ax_ind.set_title(f"Espectrograma Compuesto - {p_inf['toma']} Pulso #{p_inf['win_idx']} (Vocal /{v}/)\nRojo: Ch0 | Verde: Ch1 | Amarillo: Ch2", fontsize=10, fontweight='bold')
            ax_ind.set_xlabel("Tiempo [s]")
            ax_ind.set_ylabel("Frecuencia [Hz]")
            ax_ind.set_ylim(F_MIN_LOG, F_MAX_LOG)
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"espectrograma_rgb_{p_inf['toma']}_pulso_{p_inf['win_idx']:02d}.png"), dpi=150)
            plt.close(fig_ind)

    # 5.3 FFT (por toma individual, comparativa inter-series y promedio general)
    logger("  -> Generando espectros de frecuencias FFT (por toma y promedios)...")
    colores_series = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for v, lista_v in datos_por_vocal.items():
        if not lista_v:
            continue
        v_dir = os.path.join(subdirs['fft'], f"vocal_{v}")

        # Agrupar contracciones por toma
        tomas_vocal = {}
        for item in lista_v:
            t_name = item['info']['toma']
            if t_name not in tomas_vocal:
                tomas_vocal[t_name] = []
            tomas_vocal[t_name].append(item)

        # A) FFT para cada toma individual
        series_medias_fft = {}
        for t_name, items_toma in sorted(tomas_vocal.items()):
            ffts_toma = np.array([it['fft'] for it in items_toma])  # (P, 3, F)
            fft_toma_media = np.mean(ffts_toma, axis=0)  # (3, F)
            series_medias_fft[t_name] = fft_toma_media

            fig_t, axes_t = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
            fig_t.suptitle(f"Espectro FFT (|X(f)|) - {t_name} (N={len(items_toma)} pulsos válidos)", fontsize=13, fontweight='bold')

            for ch_i in range(3):
                ax = axes_t[ch_i]
                for p_idx in range(len(items_toma)):
                    ax.plot(f_fft[mask_f_log], ffts_toma[p_idx, ch_i][mask_f_log], color=COLORES_CANALES[ch_i], alpha=0.30, lw=0.9)
                ax.plot(f_fft[mask_f_log], fft_toma_media[ch_i][mask_f_log], color=COLORES_CANALES[ch_i], lw=2.2, label=f"Media {NOMBRES_CANALES[ch_i]}")
                ax.set_ylabel("Amplitud [µV]")
                ax.set_xscale('linear')
                ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
                ax.set_ylim(0.0, 6.0)
                ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                ax.grid(True, which='major', linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize=9)

            axes_t[-1].xaxis.set_major_formatter(ScalarFormatter())
            axes_t[-1].set_xticks(TICKS_LOG)
            axes_t[-1].set_xlabel("Frecuencia [Hz]")
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"fft_{t_name}.png"), dpi=200)
            plt.close(fig_t)

        # B) Comparativa inter-series de la vocal
        fig_comp_fft, axes_comp_fft = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig_comp_fft.suptitle(f"Comparativa Inter-Series FFT - Vocal /{v}/", fontsize=13, fontweight='bold')
        for ch_i in range(3):
            ax = axes_comp_fft[ch_i]
            for s_idx, (t_name, media_s) in enumerate(sorted(series_medias_fft.items())):
                c_s = colores_series[s_idx % len(colores_series)]
                ax.plot(f_fft[mask_f_log], media_s[ch_i][mask_f_log], label=t_name, color=c_s, lw=1.8)
            ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=10, fontweight='bold', color=COLORES_CANALES[ch_i])
            ax.set_ylabel("Amplitud [µV]")
            ax.set_xscale('linear')
            ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
            ax.set_ylim(0.0, 6.0)
            ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        axes_comp_fft[-1].xaxis.set_major_formatter(ScalarFormatter())
        axes_comp_fft[-1].set_xticks(TICKS_LOG)
        axes_comp_fft[-1].set_xlabel("Frecuencia [Hz]")
        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"comparativa_series_fft_vocal_{v}.png"), dpi=200)
        plt.close(fig_comp_fft)

        # C) Promedio consolidado de la vocal
        ffts_todos = np.array([item['fft'] for item in lista_v])
        fft_media = np.mean(ffts_todos, axis=0)
        fft_std = np.std(ffts_todos, axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig.suptitle(f"Espectro de Frecuencia FFT (|X(f)|) - Vocal /{v}/ (N={len(lista_v)} pulsos válidos)", fontsize=14, fontweight='bold')

        for ch_i in range(3):
            ax = axes[ch_i]
            for p_idx in range(len(lista_v)):
                ax.plot(f_fft[mask_f_log], ffts_todos[p_idx, ch_i][mask_f_log], color=COLORES_CANALES[ch_i], alpha=0.18, lw=0.7)
            ax.plot(f_fft[mask_f_log], fft_media[ch_i][mask_f_log], color=COLORES_CANALES[ch_i], lw=2.2, label=f"Media {NOMBRES_CANALES[ch_i]}")
            ax.fill_between(f_fft[mask_f_log], np.maximum(0, fft_media[ch_i][mask_f_log] - fft_std[ch_i][mask_f_log]), fft_media[ch_i][mask_f_log] + fft_std[ch_i][mask_f_log], color=COLORES_CANALES[ch_i], alpha=0.25, label="±1 Desv. Est.")
            ax.set_ylabel("Amplitud [µV]")
            ax.set_xscale('linear')
            ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
            ax.set_ylim(0.0, 6.0)
            ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=9)

        axes[-1].xaxis.set_major_formatter(ScalarFormatter())
        axes[-1].set_xticks(TICKS_LOG)
        axes[-1].set_xlabel("Frecuencia [Hz]")
        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"espectro_fft_promedio_vocal_{v}.png"), dpi=300)
        plt.close(fig)

        # D) Todos los pulsos individuales para FFT (escala lineal de 20 a 600 Hz, Y en [0, 4] µV)
        for item in lista_v:
            p_inf = item['info']
            fft_p = item['fft']
            fig_p, axes_p = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
            fig_p.suptitle(f"Espectro FFT - {p_inf['toma']} Pulso #{p_inf['win_idx']} (Vocal /{v}/)", fontsize=11, fontweight='bold')
            for ch_i in range(3):
                ax = axes_p[ch_i]
                ax.plot(f_fft[mask_f_log], fft_p[ch_i][mask_f_log], color=COLORES_CANALES[ch_i], lw=1.6)
                ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=9, fontweight='bold', color=COLORES_CANALES[ch_i])
                ax.set_ylabel("Amplitud [µV]", fontsize=8)
                ax.set_xscale('linear')
                ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
                ax.set_ylim(0.0, 6.0)
                ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                ax.grid(True, which='major', linestyle='--', alpha=0.5)
            axes_p[-1].xaxis.set_major_formatter(ScalarFormatter())
            axes_p[-1].set_xticks(TICKS_LOG)
            axes_p[-1].set_xlabel("Frecuencia [Hz]")
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"fft_{p_inf['toma']}_pulso_{p_inf['win_idx']:02d}.png"), dpi=150)
            plt.close(fig_p)

    # 5.4 PSD (por toma individual, comparativa inter-series y promedio general)
    logger("  -> Generando análisis de potencia PSD (Welch, por toma y promedios)...")
    for v, lista_v in datos_por_vocal.items():
        if not lista_v:
            continue
        v_dir = os.path.join(subdirs['psd'], f"vocal_{v}")

        # Agrupar contracciones por toma
        tomas_vocal = {}
        for item in lista_v:
            t_name = item['info']['toma']
            if t_name not in tomas_vocal:
                tomas_vocal[t_name] = []
            tomas_vocal[t_name].append(item)

        # A) PSD para cada toma individual
        series_medias_psd = {}
        for t_name, items_toma in sorted(tomas_vocal.items()):
            psds_toma = np.array([it['psd'] for it in items_toma])  # (P, 3, F)
            psd_toma_media = np.mean(psds_toma, axis=0)  # (3, F)
            series_medias_psd[t_name] = psd_toma_media
            mdfs_toma = np.mean([it['mdf'] for it in items_toma], axis=0)
            mnfs_toma = np.mean([it['mnf'] for it in items_toma], axis=0)

            fig_t, axes_t = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
            fig_t.suptitle(f"Densidad Espectral de Potencia (PSD Welch) - {t_name} (N={len(items_toma)} pulsos válidos)", fontsize=13, fontweight='bold')

            for ch_i in range(3):
                ax = axes_t[ch_i]
                for p_idx in range(len(items_toma)):
                    p_db = 10.0 * np.log10(np.maximum(psds_toma[p_idx, ch_i][mask_psd_log], 1e-6))
                    ax.plot(f_psd[mask_psd_log], p_db, color=COLORES_CANALES[ch_i], alpha=0.25, lw=0.8)
                psd_db = 10.0 * np.log10(np.maximum(psd_toma_media[ch_i][mask_psd_log], 1e-6))
                ax.plot(f_psd[mask_psd_log], psd_db, color=COLORES_CANALES[ch_i], lw=2.2, label=f"{NOMBRES_CANALES[ch_i]}")
                ax.axvline(mdfs_toma[ch_i], color='black', linestyle='--', lw=1.2, label=f"MDF: {mdfs_toma[ch_i]:.1f} Hz")
                ax.axvline(mnfs_toma[ch_i], color='blue', linestyle=':', lw=1.2, label=f"MNF: {mnfs_toma[ch_i]:.1f} Hz")
                ax.set_ylabel("PSD [dB/Hz]")
                ax.set_xscale('linear')
                ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
                ax.set_ylim(y_lim_psd_unificado)
                ax.grid(True, which='major', linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize=9)

            axes_t[-1].xaxis.set_major_formatter(ScalarFormatter())
            axes_t[-1].set_xticks(TICKS_LOG)
            axes_t[-1].set_xlabel("Frecuencia [Hz]")
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"psd_{t_name}.png"), dpi=200)
            plt.close(fig_t)

        # B) Comparativa inter-series de la vocal (PSD)
        fig_comp_psd, axes_comp_psd = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig_comp_psd.suptitle(f"Comparativa Inter-Series PSD Welch - Vocal /{v}/", fontsize=13, fontweight='bold')
        for ch_i in range(3):
            ax = axes_comp_psd[ch_i]
            for s_idx, (t_name, media_s) in enumerate(sorted(series_medias_psd.items())):
                c_s = colores_series[s_idx % len(colores_series)]
                p_db = 10.0 * np.log10(np.maximum(media_s[ch_i][mask_psd_log], 1e-6))
                ax.plot(f_psd[mask_psd_log], p_db, label=t_name, color=c_s, lw=1.8)
            ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=10, fontweight='bold', color=COLORES_CANALES[ch_i])
            ax.set_ylabel("PSD [dB/Hz]")
            ax.set_xscale('linear')
            ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
            ax.set_ylim(y_lim_psd_unificado)
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=8, ncol=2)
        axes_comp_psd[-1].xaxis.set_major_formatter(ScalarFormatter())
        axes_comp_psd[-1].set_xticks(TICKS_LOG)
        axes_comp_psd[-1].set_xlabel("Frecuencia [Hz]")
        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"comparativa_series_psd_vocal_{v}.png"), dpi=200)
        plt.close(fig_comp_psd)

        # C) Promedio consolidado de la vocal (PSD)
        psds_todos = np.array([item['psd'] for item in lista_v])
        psd_media = np.mean(psds_todos, axis=0)

        mdfs_medios = np.mean([item['mdf'] for item in lista_v], axis=0)
        mnfs_medios = np.mean([item['mnf'] for item in lista_v], axis=0)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, sharey=True)
        fig.suptitle(f"Densidad Espectral de Potencia (PSD Welch) - Vocal /{v}/ (N={len(lista_v)} pulsos válidos)", fontsize=14, fontweight='bold')
       
        for ch_i in range(3):
            ax = axes[ch_i]
            for p_idx in range(len(lista_v)):
                p_db = 10.0 * np.log10(np.maximum(psds_todos[p_idx, ch_i][mask_psd_log], 1e-6))
                ax.plot(f_psd[mask_psd_log], p_db, color=COLORES_CANALES[ch_i], alpha=0.15, lw=0.7)
            psd_db = 10.0 * np.log10(np.maximum(psd_media[ch_i][mask_psd_log], 1e-6))
            ax.plot(f_psd[mask_psd_log], psd_db, color=COLORES_CANALES[ch_i], lw=2.2, label=f"{NOMBRES_CANALES[ch_i]}")
            ax.axvline(mdfs_medios[ch_i], color='black', linestyle='--', lw=1.2, label=f"MDF: {mdfs_medios[ch_i]:.1f} Hz")
            ax.axvline(mnfs_medios[ch_i], color='blue', linestyle=':', lw=1.2, label=f"MNF: {mnfs_medios[ch_i]:.1f} Hz")
            ax.set_ylabel("PSD [dB/Hz]")
            ax.set_xscale('linear')
            ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
            ax.set_ylim(y_lim_psd_unificado)
            ax.grid(True, which='major', linestyle='--', alpha=0.5)
            ax.legend(loc='upper right', fontsize=9)

        axes[-1].xaxis.set_major_formatter(ScalarFormatter())
        axes[-1].set_xticks(TICKS_LOG)
        axes[-1].set_xlabel("Frecuencia [Hz]")
        plt.tight_layout()
        plt.savefig(os.path.join(v_dir, f"psd_potencia_promedio_vocal_{v}.png"), dpi=300)
        plt.close(fig)

        # D) Todos los pulsos individuales para PSD (escala lineal de 20 a 600 Hz)
        for item in lista_v:
            p_inf = item['info']
            psd_p = item['psd']
            mdf_p = item['mdf']
            mnf_p = item['mnf']
            fig_p, axes_p = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
            fig_p.suptitle(f"PSD Welch - {p_inf['toma']} Pulso #{p_inf['win_idx']} (Vocal /{v}/)", fontsize=11, fontweight='bold')
            for ch_i in range(3):
                ax = axes_p[ch_i]
                p_db = 10.0 * np.log10(np.maximum(psd_p[ch_i][mask_psd_log], 1e-6))
                ax.plot(f_psd[mask_psd_log], p_db, color=COLORES_CANALES[ch_i], lw=1.6)
                ax.axvline(mdf_p[ch_i], color='black', linestyle='--', lw=1.1, label=f"MDF: {mdf_p[ch_i]:.1f} Hz")
                ax.axvline(mnf_p[ch_i], color='blue', linestyle=':', lw=1.1, label=f"MNF: {mnf_p[ch_i]:.1f} Hz")
                ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=9, fontweight='bold', color=COLORES_CANALES[ch_i])
                ax.set_ylabel("PSD [dB/Hz]", fontsize=8)
                ax.set_xscale('linear')
                ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
                ax.set_ylim(y_lim_psd_unificado)
                ax.grid(True, which='major', linestyle='--', alpha=0.5)
                ax.legend(loc='upper right', fontsize=8)
            axes_p[-1].xaxis.set_major_formatter(ScalarFormatter())
            axes_p[-1].set_xticks(TICKS_LOG)
            axes_p[-1].set_xlabel("Frecuencia [Hz]")
            plt.tight_layout()
            plt.savefig(os.path.join(v_dir, f"psd_{p_inf['toma']}_pulso_{p_inf['win_idx']:02d}.png"), dpi=150)
            plt.close(fig_p)


    # 5.5 Comparativas Globales
    logger("  -> Generando paneles comparativos globales de las 5 vocales...")
    vocales_orden = ['A', 'E', 'I', 'O', 'U']

    # A) Espectrogramas
    fig_glob_stft, axes_stft = plt.subplots(3, 5, figsize=(18, 9), sharex=True, sharey=True)
    fig_glob_stft.suptitle("Comparativa Espectral Global (STFT 20-600 Hz) - Candela (2026-09-16)", fontsize=15, fontweight='bold')

    for col_i, v in enumerate(vocales_orden):
        lista_v = datos_por_vocal[v]
        if not lista_v:
            continue
        stft_medio = np.mean([item['stft'] for item in lista_v], axis=0)
        supremo_v = np.max(stft_medio) + 1e-12

        for ch_i in range(3):
            ax = axes_stft[ch_i, col_i]
            stft_db = 10.0 * np.log10(np.maximum(stft_medio[ch_i] / supremo_v, 1e-4))
            ax.imshow(stft_db, origin='lower', aspect='auto', extent=extent_stft, cmap='magma', vmin=-35, vmax=0)
            ax.set_ylim(F_MIN_LOG, F_MAX_LOG)
            if ch_i == 0:
                ax.set_title(f"Vocal /{v}/ (N={len(lista_v)})", fontsize=12, fontweight='bold', color=COLORES_VOCALES[v])
            if col_i == 0:
                ax.set_ylabel(f"{NOMBRES_CANALES[ch_i]}\nFrecuencia [Hz]", fontsize=10, fontweight='bold', color=COLORES_CANALES[ch_i])
            if ch_i == 2:
                ax.set_xlabel("Tiempo [s]", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(subdirs['resumen'], "comparativa_espectrogramas_5vocales.png"), dpi=300)
    plt.close(fig_glob_stft)

    # B) Compuesto Rojo/Verde/Amarillo
    fig_glob_rgb, axes_rgb = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    fig_glob_rgb.suptitle("Espectrogramas Compuestos por Vocal (20-600 Hz) - Candela (2026-09-16)\nRojo: Ch0 Digástrico | Verde: Ch1 Zigo | Amarillo: Ch2 Orbicularis", fontsize=14, fontweight='bold')

    for col_i, v in enumerate(vocales_orden):
        lista_v = datos_por_vocal[v]
        ax = axes_rgb[col_i]
        if not lista_v:
            continue
        stft_medio = np.mean([item['stft'] for item in lista_v], axis=0)
        comp_medio = construir_espectrograma_coloreado(stft_medio[0], stft_medio[1], stft_medio[2])

        ax.imshow(comp_medio, origin='lower', aspect='auto', extent=extent_stft)
        ax.set_title(f"Vocal /{v}/ (N={len(lista_v)})", fontsize=13, fontweight='bold', color=COLORES_VOCALES[v])
        ax.set_xlabel("Tiempo relativo [s]", fontsize=10)
        if col_i == 0:
            ax.set_ylabel("Frecuencia [Hz]", fontsize=11, fontweight='bold')
        ax.set_ylim(F_MIN_LOG, F_MAX_LOG)

    plt.tight_layout()
    plt.savefig(os.path.join(subdirs['resumen'], "comparativa_rgb_5vocales.png"), dpi=300)
    plt.close(fig_glob_rgb)

    # B.2) Correlación EMG-Audio (Promedios Globales por Vocal)
    fig_glob_corr, axes_corr = plt.subplots(2, 5, figsize=(20, 8), sharex=True)
    fig_glob_corr.suptitle("Correlación EMG vs Audio (Promedio por Vocal)\nFila Superior: EMG RGB (Rojo=Dig, Verde=Zig, Amar=Orb) | Fila Inferior: Audio Micrófono", fontsize=14, fontweight='bold')
    
    for col_i, v in enumerate(vocales_orden):
        lista_v = datos_por_vocal[v]
        ax_rgb = axes_corr[0, col_i]
        ax_aud = axes_corr[1, col_i]
        
        if not lista_v:
            continue
            
        stft_medio = np.mean([item['stft'] for item in lista_v], axis=0)
        comp_medio = construir_espectrograma_coloreado(stft_medio[0], stft_medio[1], stft_medio[2])
        
        avg_audio = np.mean([item['audio_stft'] for item in lista_v], axis=0)
        fs = lista_v[0]['info']['fs']
        f_audio, t_audio, _ = calcular_espectrograma(lista_v[0]['info']['seg_audio'], fs=fs, nperseg=128, noverlap=100, f_max=1000.0)
        
        ax_rgb.imshow(comp_medio, origin='lower', aspect='auto', extent=extent_stft)
        ax_rgb.set_title(f"EMG Vocal /{v}/ (N={len(lista_v)})", fontsize=12, fontweight='bold', color=COLORES_VOCALES[v])
        ax_rgb.set_ylim(F_MIN_LOG, F_MAX_LOG)
        if col_i == 0:
            ax_rgb.set_ylabel("Frec. EMG [Hz]", fontsize=11, fontweight='bold')
            
        im = ax_aud.imshow(10.0 * np.log10(np.maximum(avg_audio, 1e-12)), origin='lower', aspect='auto', extent=[t_audio[0], t_audio[-1], f_audio[0], f_audio[-1]], cmap='magma')
        ax_aud.set_xlabel("Tiempo relativo [s]", fontsize=10)
        if col_i == 0:
            ax_aud.set_ylabel("Frec. Audio [Hz]", fontsize=11, fontweight='bold')
            
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs['resumen'], "comparativa_correlacion_audio_5vocales.png"), dpi=300)
    plt.close(fig_glob_corr)


    # C) FFT
    fig_glob_fft, axes_fft = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig_glob_fft.suptitle("Comparativa Espectral FFT por Canal Muscular (20-600 Hz)", fontsize=14, fontweight='bold')

    for ch_i in range(3):
        ax = axes_fft[ch_i]
        for v in vocales_orden:
            lista_v = datos_por_vocal[v]
            if not lista_v:
                continue
            ffts_v = np.mean([item['fft'][ch_i] for item in lista_v], axis=0)
            ax.plot(f_fft[mask_f_log], ffts_v[mask_f_log], label=f"Vocal /{v}/", color=COLORES_VOCALES[v], lw=1.6)

        ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=11, fontweight='bold', color=COLORES_CANALES[ch_i])
        ax.set_ylabel("Amplitud [µV]")
        ax.set_xscale('linear')
        ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
        ax.set_ylim(0.0, 6.0)
        ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', ncol=5, fontsize=9)

    axes_fft[-1].xaxis.set_major_formatter(ScalarFormatter())
    axes_fft[-1].set_xticks(TICKS_LOG)
    axes_fft[-1].set_xlabel("Frecuencia [Hz]", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs['resumen'], "comparativa_fft_5vocales.png"), dpi=300)
    plt.close(fig_glob_fft)

    # D) PSD
    fig_glob_psd, axes_psd = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig_glob_psd.suptitle("Comparativa de Densidad Espectral de Potencia (PSD Welch 20-600 Hz)", fontsize=14, fontweight='bold')

    for ch_i in range(3):
        ax = axes_psd[ch_i]
        for v in vocales_orden:
            lista_v = datos_por_vocal[v]
            if not lista_v:
                continue
            psd_v = np.mean([item['psd'][ch_i] for item in lista_v], axis=0)
            psd_db = 10.0 * np.log10(np.maximum(psd_v[mask_psd_log], 1e-6))
            ax.plot(f_psd[mask_psd_log], psd_db, label=f"Vocal /{v}/", color=COLORES_VOCALES[v], lw=1.6)

        ax.set_title(f"{NOMBRES_CANALES[ch_i]}", fontsize=11, fontweight='bold', color=COLORES_CANALES[ch_i])
        ax.set_ylabel("PSD [dB/Hz]")
        ax.set_xscale('linear')
        ax.set_xlim(F_MIN_LOG, F_MAX_LOG)
        ax.set_ylim(y_lim_psd_unificado)
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.legend(loc='upper right', ncol=5, fontsize=9)

    axes_psd[-1].xaxis.set_major_formatter(ScalarFormatter())
    axes_psd[-1].set_xticks(TICKS_LOG)
    axes_psd[-1].set_xlabel("Frecuencia [Hz]", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(subdirs['resumen'], "comparativa_psd_5vocales.png"), dpi=300)
    plt.close(fig_glob_psd)

    duracion_total = time.time() - tiempo_inicio
    logger("\n" + "=" * 80)
    logger(f"PROCESAMIENTO ESPECTRAL CULMINADO CON EXITO EN {duracion_total:.2f} s")
    logger(f"Resultados guardados en: {salida_base_dir}")
    logger("=" * 80)


if __name__ == '__main__':
    ejecutar_analisis_completo()
