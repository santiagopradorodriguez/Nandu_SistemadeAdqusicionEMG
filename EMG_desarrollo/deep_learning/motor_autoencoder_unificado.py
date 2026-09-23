# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor Unificado de Autoencoders No Supervisados (Cero Etiquetas)
#              - Auditoría estricta de metadatos (músculos por canal, BPM, fs).
#              - Extracción DSP con ruido dinámico interpulso (IQR) y Supremo Tricanal.
#              - Soporte trilateral de modalidades:
#                1. Envolvente 1D (RMS / Filtrada a 200 Hz).
#                2. Señal Cruda 1D (2000 Hz, GAP+GMP).
#                3. Espectrogramas 2D / 3D (STFT Calibrada dB con clausura morfológica).
#              - Modelado 100% No Supervisado (Zero-Labels) y evaluación GMM canónica.
# ==============================================================================

import os
import sys
import json
import re
import time
import random
from datetime import datetime
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt, find_peaks, spectrogram, iirnotch
from scipy.ndimage import grey_closing
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

try:
    from deep_learning.soft_dtw import SoftDTW, SoftDTWDivergence, HibridaLoss
except ImportError:
    try:
        from .soft_dtw import SoftDTW, SoftDTWDivergence, HibridaLoss
    except ImportError:
        try:
            import soft_dtw
            SoftDTW = soft_dtw.SoftDTW
            SoftDTWDivergence = soft_dtw.SoftDTWDivergence
            HibridaLoss = soft_dtw.HibridaLoss
        except Exception:
            SoftDTW, SoftDTWDivergence, HibridaLoss = None, None, None

# Rutas estándar del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
emg_desarrollo_dir = os.path.abspath(os.path.join(script_dir, ".."))
base_datos_dir = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos")
analysis_dir = os.path.join(emg_desarrollo_dir, "analysis")
resultados_dir = os.path.join(emg_desarrollo_dir, "resultados", "resultados_autoencoder")
cache_dir = os.path.join(resultados_dir, "cache_datasets")
modelos_dir = os.path.join(resultados_dir, "modelos_entrenados")
figuras_dir = os.path.join(resultados_dir, "figuras_evaluacion")
legacy_cache_dir = os.path.join(emg_desarrollo_dir, "cache_datos")

os.makedirs(resultados_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(modelos_dir, exist_ok=True)
os.makedirs(figuras_dir, exist_ok=True)
os.makedirs(legacy_cache_dir, exist_ok=True)

if analysis_dir not in sys.path:
    sys.path.append(analysis_dir)

try:
    from filtro_adaptativo import cancelar_ruido_linea_adaptativo
except ImportError:
    cancelar_ruido_linea_adaptativo = None

COLORES_VOCALES = {
    'A': '#d62728',
    'E': '#1f77b4',
    'I': '#2ca02c',
    'O': '#9467bd',
    'U': '#d4ac0d'
}

def fijar_semilla(seed=42):
    """Garantiza reproducibilidad estricta universal en PyTorch, NumPy y Python random (Directiva /learn)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def abrir_imagen_en_visor(ruta_imagen):
    """
    Abre la imagen procesada en el visor gráfico predeterminado del sistema operativo.
    En Linux invoca xdg-open en un subproceso desacoplado para visualizar inmediatamente
    el informe sin bloquear la ejecución.
    """
    if not ruta_imagen or not os.path.exists(ruta_imagen):
        return False
    try:
        import subprocess
        if sys.platform.startswith('linux'):
            subprocess.Popen(['xdg-open', ruta_imagen], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', ruta_imagen], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == 'win32':
            os.startfile(ruta_imagen)
        return True
    except Exception as e:
        print(f"Aviso: no se pudo abrir el visor gráfico del sistema: {e}")
        return False

# ==============================================================================
# 1. AUDITORÍA OBLIGATORIA DE METADATOS INTER-DÍA
# ==============================================================================
try:
    from utils.metadata_auditor import auditar_metadatos_sesiones, leer_metadata_toma
except ImportError:
    try:
        from ..utils.metadata_auditor import auditar_metadatos_sesiones, leer_metadata_toma
    except ImportError:
        def auditar_metadatos_sesiones(rutas_tomas):
            return {'compatible': True, 'advertencias': [], 'info_sesiones': {}, 'musculos_resumen': {}, 'bpm_comun': 40, 'fs_comun': 2000, 'fechas_detectadas': []}

# ==============================================================================
# 2. FUNCIONES DSP Y ESTIMACIÓN DE RUIDO DINÁMICO INTERPULSO
# ==============================================================================

def leer_wav_mono(filepath):
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype == np.int16:
        signal = data.astype(np.float64) / 32768.0
    elif data.dtype == np.int32:
        signal = data.astype(np.float64) / 2147483648.0
    else:
        signal = data.astype(np.float64)
    return signal, sr

def acondicionar_senal_cruda(raw_signal, fs=2000.0, hp_cutoff=20.0, lp_cutoff=450.0, tipo_filtro_linea="adaptativo", notch_q=2.0):
    tipo_filtro = str(tipo_filtro_linea or "adaptativo").lower().strip()
    nyq = 0.5 * fs
    
    if tipo_filtro.startswith("adapt"):
        if cancelar_ruido_linea_adaptativo is not None:
            armonicos_linea = (50, 100, 150, 200, 250, 300, 350, 400)
            sig_linea, _, _ = cancelar_ruido_linea_adaptativo(
                raw_signal, fs=fs, f0=50.0, mu=0.01, normalizado=True, armonicos=armonicos_linea
            )
        else:
            sig_linea = raw_signal
    elif tipo_filtro.startswith("notch"):
        # Filtro Notch IIR en cascada para 50 Hz y armónicos
        sig_linea = raw_signal.copy()
        q_val = max(0.5, float(notch_q))
        for f_notch in [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]:
            if f_notch < nyq - 5.0:
                b_n, a_n = iirnotch(f_notch, q_val, fs)
                sig_linea = filtfilt(b_n, a_n, sig_linea)
    else:
        sig_linea = raw_signal

    hp_c = max(1.0, min(float(hp_cutoff), nyq - 10.0))
    lp_c = max(hp_c + 5.0, min(float(lp_cutoff), nyq - 5.0))
    b_hp, a_hp = butter(4, hp_c / nyq, btype='high')
    sig_hp = filtfilt(b_hp, a_hp, sig_linea)
    b_lp, a_lp = butter(4, lp_c / nyq, btype='low')
    sig_bp = filtfilt(b_lp, a_lp, sig_hp)
    return sig_bp

def get_interpulse_noise(segment, initial_noise):
    if len(segment) < 10:
        return initial_noise
    abs_n = np.abs(segment)
    q1 = np.percentile(abs_n, 25)
    q3 = np.percentile(abs_n, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid = abs_n[abs_n <= upper_bound]
    if len(valid) < 3:
        valid = abs_n
    curr_mean = float(np.mean(valid))
    if initial_noise > 0 and (curr_mean / initial_noise) > 5.0:
        return initial_noise
    return curr_mean

def calcular_envolvente(s_raw, s_bp, fs=2000, tipo_envolvente="rms", smooth_ms=100):
    """
    Calcula la envolvente de la señal bioeléctrica según la técnica seleccionada:
    - 'rms': Valor cuadrático medio con ventana móvil centrada.
    - 'media_movil': Rectificación completa y filtro de media móvil.
    - 'tkeo': Operador de Energía Teager-Kaiser con suavizado.
    - 'hilbert': Magnitud analítica de Hilbert con filtro suavizador.
    """
    tipo = str(tipo_envolvente).lower().strip()
    win_len = max(1, int(round((smooth_ms * fs) / 1000.0)))

    if tipo == "tkeo":
        x = np.asarray(s_bp, dtype=np.float64)
        psi = np.zeros_like(x)
        if len(x) >= 3:
            psi[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
            psi[0] = psi[1]
            psi[-1] = psi[-2]
        psi = np.maximum(psi, 0.0)
        if win_len > 1:
            w = np.hanning(win_len)
            w = w / np.sum(w)
            return np.sqrt(np.maximum(np.convolve(psi, w, mode='same'), 0.0)).astype(np.float32)
        return np.sqrt(psi).astype(np.float32)

    elif tipo == "media_movil":
        s_abs = np.abs(s_bp)
        if win_len > 1:
            w = np.hanning(win_len)
            w = w / np.sum(w)
            return np.convolve(s_abs, w, mode='same').astype(np.float32)
        return s_abs.astype(np.float32)

    elif tipo == "hilbert":
        from scipy.signal import hilbert
        env_h = np.abs(hilbert(s_bp))
        if win_len > 1:
            w = np.hanning(win_len)
            w = w / np.sum(w)
            return np.convolve(env_h, w, mode='same').astype(np.float32)
        return env_h.astype(np.float32)

    else: # Por defecto "rms"
        sig_sq = s_bp ** 2
        if win_len > 1:
            w = np.hanning(win_len)
            w = w / np.sum(w)
            return np.sqrt(np.maximum(np.convolve(sig_sq, w, mode='same'), 0.0)).astype(np.float32)
        return np.abs(s_bp).astype(np.float32)

def procesar_stft_calibrada(pulso_nch, fs=2000, nperseg=128, noverlap=120, umbral_db=-22.0, target_size=(32, 64)):
    n_canales = pulso_nch.shape[0]
    canales_raw = []
    for c in range(n_canales):
        f, t, Sxx = spectrogram(
            pulso_nch[c], fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, scaling='density'
        )
        mask_f = (f >= 20.0) & (f <= 450.0)
        canales_raw.append(Sxx[mask_f, :])

    mat_power = np.stack(canales_raw, axis=0)
    supremo_pulso = np.max(mat_power) + 1e-12
    mat_db = 10.0 * np.log10(np.maximum(mat_power / supremo_pulso, 1e-6))

    canales_cerrados = []
    for c in range(n_canales):
        ch_c = grey_closing(mat_db[c], size=(3, 3))
        ch_norm = np.clip((ch_c - umbral_db) / (0.0 - umbral_db), 0.0, 1.0)
        canales_cerrados.append(ch_norm)

    mat_depurada = np.stack(canales_cerrados, axis=0)
    t_in = torch.tensor(mat_depurada, dtype=torch.float32).unsqueeze(0)
    t_interp = F.interpolate(t_in, size=target_size, mode='bilinear', align_corners=False).squeeze(0).numpy()
    return t_interp

# ==============================================================================
# 3. EXTRACTOR UNIFICADO DE DATASETS MULTI-MODALES
# ==============================================================================

def extraer_dataset_unificado(
    rutas_tomas,
    aplicar_correccion_intersesion=True,
    usar_calibracion_p95=True,
    modo_alineacion="Pico Volumen Micrófono",
    carpeta_salida=None,
    tipo_envolvente="rms",
    smooth_ms=100,
    alpha_ruido=1.0,
    target_len=100,
    outlier_contamination=0.10,
    w_canales=(1.0, 1.0, 1.0),
    callback_log=None,
    canales_features=("canal_0", "canal_1", "canal_2"),
    **kwargs
):
    aplicar_correccion_intersesion = kwargs.get('aplicar_correccion_intersesion', aplicar_correccion_intersesion)
    canales_features = list(kwargs.get('canales_features', canales_features))
    if not canales_features or len(canales_features) < 2:
        canales_features = ["canal_0", "canal_1", "canal_2"]

    indices_canales = [int(c.split('_')[1]) for c in canales_features]
    n_canales = len(canales_features)
    fijar_semilla(seed=42)
    def _log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg, flush=True)

    _log("Iniciando extracción unificada de señales sEMG (Estándar Trevisan / PCA-UMAP)...")
    _log(f"Canales seleccionados ({n_canales}): {', '.join(canales_features)}")
    _log(f"Modo de alineación seleccionado: '{modo_alineacion}'")
    t0 = time.time()

    mediciones_datos = []
    musculos_canales_global = [f"Canal {idx}" for idx in indices_canales]
    fs_global = 2000
    bpm_global = 40
    total_tomas = len(rutas_tomas)

    for i, toma_dir in enumerate(rutas_tomas):
        toma_name = os.path.basename(toma_dir)
        vocal = toma_name.split('_')[0].upper()
        if vocal not in ['A', 'E', 'I', 'O', 'U']:
            continue

        _log(f"[Carga] Toma {i+1}/{total_tomas} ({((i+1)/total_tomas)*100:.1f}%) - {toma_name}")

        # 1. Lectura obligatoria de metadatos por toma
        meta_path = os.path.join(toma_dir, "canal_0", "metadata.json")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(toma_dir, "metadata.json")
            if not os.path.exists(meta_path):
                _log(f"  [Aviso] Ignorando {toma_name}: no contiene metadata.json")
                continue

        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        bpm = float(meta.get('bpm', 40))
        noise_seconds = float(meta.get('noise_seconds', 3.0))
        pulsos_u = meta.get('pulse_count', None)
        fs = int(meta.get('sample_rate', 2000))
        fs_global = fs
        bpm_global = bpm

        # Identificación de fecha y sesión para calibración intersesión y trazabilidad
        m_date_raw = str(meta.get('measurement_date') or meta.get('date') or '').strip()
        if m_date_raw:
            fecha_toma = m_date_raw.split('T')[0][:10]
        else:
            fecha_toma = "Fecha_Desconocida"
            partes_r = toma_dir.replace('\\', '/').split('/')
            for p in partes_r:
                if len(p) == 10 and p[4] == '-' and p[7] == '-':
                    fecha_toma = p
                    break

        parent_folder = os.path.basename(os.path.dirname(toma_dir))
        session_key = parent_folder

        # Mapeo anatómico de canales
        m_map = meta.get('muscles_map', {})
        if not m_map:
            m_list = meta.get('muscles', [])
            if len(m_list) >= 3:
                m_map = {f"canal_{j}": m_list[j] for j in range(len(m_list))}
            else:
                m_map = {"canal_0": "Canal 0", "canal_1": "Canal 1", "canal_2": "Canal 2"}
        musculos_canales_global = [m_map.get(ch, f"Ch{idx}") for idx, ch in zip(indices_canales, canales_features)]

        # 2. Lectura de excluded_windows.json si existe
        excluded_windows = []
        exclude_path = os.path.join(toma_dir, 'excluded_windows.json')
        if not os.path.exists(exclude_path):
            exclude_path = os.path.join(toma_dir, 'canal_0', 'excluded_windows.json')
        if os.path.exists(exclude_path):
            try:
                with open(exclude_path, 'r', encoding='utf-8') as f_excl:
                    data_excl = json.load(f_excl)
                    excluded_windows = data_excl.get("excluded_windows", [])
            except Exception:
                pass

        # 3. Carga de canales sEMG seleccionados y canal 3 (micrófono)
        canales_ok = True
        sigs_raw = []
        sigs_bp = []
        sigs_env = []
        inits_noise = []

        noise_samples_init = max(10, int(noise_seconds * fs))
        hp_cutoff = float(kwargs.get('highpass_cutoff_hz', 20.0))
        lp_cutoff = float(kwargs.get('lowpass_cutoff_hz', 450.0))
        tipo_filtro_linea = kwargs.get('tipo_filtro_linea', 'adaptativo')
        notch_q = float(kwargs.get('notch_q', 2.0))

        for ch in canales_features:
            wav_path = os.path.join(toma_dir, ch, "grabacion.wav")
            if not os.path.exists(wav_path):
                canales_ok = False
                break
            s_raw, _ = leer_wav_mono(wav_path)
            s_bp = acondicionar_senal_cruda(
                s_raw, fs=fs, hp_cutoff=hp_cutoff, lp_cutoff=lp_cutoff,
                tipo_filtro_linea=tipo_filtro_linea, notch_q=notch_q
            )
            s_env = calcular_envolvente(s_raw, s_bp, fs=fs, tipo_envolvente=tipo_envolvente, smooth_ms=smooth_ms)
            init_n = np.median(s_env[:noise_samples_init]) if len(s_env) > noise_samples_init else np.median(s_env)

            sigs_raw.append(s_raw)
            sigs_bp.append(s_bp)
            sigs_env.append(s_env)
            inits_noise.append(init_n)

        if not canales_ok:
            continue

        mic_wav = os.path.join(toma_dir, "canal_3", "grabacion.wav")
        mic_sig = None
        if os.path.exists(mic_wav):
            mic_sig, _ = leer_wav_mono(mic_wav)

        # 4. Segmentación periódica por ranura de metrónomo (Estándar Oficial Trevisan)
        muestras_pulso = int(round((60.0 / bpm) * fs))
        start_sample_noise = int(noise_seconds * fs)
        n_pulsos_total = int(pulsos_u) if (pulsos_u is not None and pulsos_u > 0) else max(1, (len(sigs_env[0]) - start_sample_noise) // muestras_pulso)

        pre_pct = 0.40
        post_pct = 0.60
        pre_samples = int(round(muestras_pulso * pre_pct))
        post_samples = int(round(muestras_pulso * post_pct))
        noise_win_samples = max(10, int(muestras_pulso / 4.0))

        # Señal de referencia según modo_alineacion
        if modo_alineacion.startswith("Pico Derivada Micrófono") and mic_sig is not None:
            win_mic = max(5, int(0.05 * fs))
            mic_env = np.convolve(np.abs(mic_sig), np.ones(win_mic)/win_mic, mode='same')
            deriv_mic = np.gradient(mic_env)
            win_d = max(1, int(fs * 0.05))
            sig_ref_align = np.convolve(deriv_mic, np.ones(win_d)/win_d, mode='same')
        elif modo_alineacion.startswith("Pico Volumen Micrófono") and mic_sig is not None:
            win_mic = max(5, int(0.05 * fs))
            sig_ref_align = np.convolve(np.abs(mic_sig), np.ones(win_mic)/win_mic, mode='same')
        elif modo_alineacion == "Pico Canal 0" and "canal_0" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_0")]
        elif modo_alineacion == "Pico Canal 1" and "canal_1" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_1")]
        elif modo_alineacion == "Pico Canal 2" and "canal_2" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_2")]
        else: # Pico Envolvente Muscular (Supremo)
            sig_ref_align = sigs_env[0]
            for c in range(1, n_canales):
                sig_ref_align = np.maximum(sig_ref_align, sigs_env[c])

        # Iterar exactamente por cada ranura de pulso (inmune a desfasaje y colapso en t=0)
        for win_idx in range(n_pulsos_total):
            if (win_idx + 1) in excluded_windows or win_idx in excluded_windows:
                continue

            cut_start = start_sample_noise + win_idx * muestras_pulso
            cut_end = min(len(sig_ref_align), cut_start + muestras_pulso)
            if cut_end - cut_start < muestras_pulso // 2:
                continue

            local_slot = sig_ref_align[cut_start:cut_end]
            if len(local_slot) == 0:
                continue

            # Pico fisiológico dentro de la ranura de metrónomo
            rel_max = int(np.argmax(local_slot))
            p_idx = cut_start + rel_max

            pulse_start = p_idx - pre_samples
            pulse_end = p_idx + post_samples
            if pulse_start < 0 or pulse_end > len(sigs_bp[0]):
                continue

            # Estimación dinámica de ruido interpulso
            ruidos_c = []
            for c_idx in range(n_canales):
                env_ch = sigs_env[c_idx]
                init_n = inits_noise[c_idx]
                n_start_pre = max(0, int(p_idx - 0.5 * muestras_pulso - noise_win_samples))
                n_end_pre = min(len(env_ch), n_start_pre + noise_win_samples)
                r_pre = get_interpulse_noise(env_ch[n_start_pre:n_end_pre], init_n)

                n_start_post = min(len(env_ch), int(p_idx + 0.5 * muestras_pulso))
                n_end_post = min(len(env_ch), n_start_post + noise_win_samples)
                r_post = get_interpulse_noise(env_ch[n_start_post:n_end_post], init_n)
                ruidos_c.append((r_pre + r_post) / 2.0)

            rect_segs = []
            env_segs = []
            es_valido = True
            w_c = []
            for ch_idx in indices_canales:
                if w_canales is not None and ch_idx < len(w_canales):
                    w_c.append(float(w_canales[ch_idx]))
                else:
                    w_c.append(1.0)
            w_c = np.array(w_c, dtype=np.float32)

            for c_idx in range(n_canales):
                raw_seg = sigs_bp[c_idx][pulse_start:pulse_end]
                env_seg = sigs_env[c_idx][pulse_start:pulse_end]
                if len(raw_seg) < 50:
                    es_valido = False
                    break
                r_c = ruidos_c[c_idx] * float(alpha_ruido)
                peso_ch = float(w_c[c_idx])

                # Medir piso basal local en ambos extremos y sustraer rampa lineal para garantizar que inicie y termine estrictamente en 0.0
                win_edge = max(3, int(0.12 * len(env_seg)))
                base_env_pre = max(float(r_c), float(np.median(env_seg[:win_edge])))
                base_env_post = max(float(r_c), float(np.median(env_seg[-win_edge:])))
                t_ramp = np.linspace(0.0, 1.0, len(env_seg))
                ramp_env = base_env_pre + t_ramp * (base_env_post - base_env_pre)
                env_clean = np.maximum(0.0, env_seg - ramp_env) * peso_ch

                base_raw_pre = max(float(r_c), float(np.median(np.abs(raw_seg)[:win_edge])))
                base_raw_post = max(float(r_c), float(np.median(np.abs(raw_seg)[-win_edge:])))
                ramp_raw = base_raw_pre + t_ramp * (base_raw_post - base_raw_pre)
                rect_clean = np.maximum(0.0, np.abs(raw_seg) - ramp_raw) * peso_ch

                rect_segs.append(rect_clean)
                env_segs.append(env_clean)

            if not es_valido or len(rect_segs) != n_canales:
                continue

            # Interpolar a longitud fija y normalizar por Supremo del pulso individual
            target_cruda_len = 1000
            target_env_len = int(target_len) if target_len and target_len > 0 else 100

            x_orig_w = np.linspace(0.0, 1.0, len(env_segs[0]))
            x_tgt_e = np.linspace(0.0, 1.0, target_env_len)
            x_tgt_c = np.linspace(0.0, 1.0, target_cruda_len)

            env_i = np.array([np.interp(x_tgt_e, x_orig_w, env_segs[c]) for c in range(n_canales)], dtype=np.float32)
            raw_i = np.array([np.interp(x_tgt_c, x_orig_w, rect_segs[c]) for c in range(n_canales)], dtype=np.float32)

            if float(np.max(env_i)) <= 1e-6:
                continue

            mediciones_datos.append({
                'win_idx': win_idx,
                'env_i': env_i,
                'raw_i': raw_i,
                'vocal': vocal,
                'toma': f"{toma_name}_W{win_idx}",
                'fecha': fecha_toma,
                'session_key': session_key
            })

    N_total = len(mediciones_datos)
    _log(f"Extracción de ventanas completada. Total de contracciones: {N_total}")
    if N_total == 0:
        raise RuntimeError("No se pudieron extraer contracciones válidas de las tomas seleccionadas.")

    # ------------------------------------------------------------------
    # 5. CALIBRACIÓN INTERSESIÓN POR LOTE P95 (ESTÁNDAR PCA-UMAP)
    # ------------------------------------------------------------------
    session_factors = {}
    if aplicar_correccion_intersesion:
        _log("Calculando calibración intersesión por lote P95 (Estándar PCA/UMAP)...")
        sessions_dict = {}
        for w in mediciones_datos:
            s_key = (w['fecha'], w['session_key'])
            sessions_dict.setdefault(s_key, []).append(w)

        for s_key, wins_sesion in sessions_dict.items():
            s_fecha, s_tag = s_key
            V = []
            for c_idx in range(n_canales):
                picos_ch = [float(np.max(w['env_i'][c_idx])) for w in wins_sesion]
                p95 = float(np.percentile(picos_ch, 95)) if picos_ch else 1.0
                V.append(p95)
            V = np.array(V, dtype=np.float32)
            V_ref = float(np.max(V))
            if V_ref > 1e-9:
                alpha = V / V_ref
                # alpha_piso = 0.20 garantiza acotar la amplificación a un máximo de 5.0x
                C = 1.0 / np.maximum(alpha, 0.20)
            else:
                C = np.ones(n_canales, dtype=np.float32)
            session_factors[s_key] = C
            ch_str = ", ".join([f"Ch{indices_canales[c]}: C={C[c]:.2f} (P95={V[c]:.4f})" for c in range(n_canales)])
            _log(f"  [Intersesión] Sesión '{s_tag}' ({s_fecha}) -> {ch_str}")
    else:
        c_ones_str = ", ".join(["1.0" for _ in range(n_canales)])
        _log(f"[Intersesión] Calibración intersesión desactivada (factores C = [{c_ones_str}]).")

    # ------------------------------------------------------------------
    # 6. ESCALADO Y NORMALIZACIÓN POR SUPREMO POR PULSO INDIVIDUAL
    # ------------------------------------------------------------------
    for w in mediciones_datos:
        s_key = (w['fecha'], w['session_key'])
        C = session_factors.get(s_key, np.ones(n_canales, dtype=np.float32)) if aplicar_correccion_intersesion else np.ones(n_canales, dtype=np.float32)
        C_col = C.reshape(n_canales, 1)

        env_scaled = w['env_i'] * C_col
        raw_scaled = w['raw_i'] * C_col

        supremo_pulso = max(float(np.max(env_scaled)), 1e-9)
        if supremo_pulso <= 1e-6:
            w['env_norm'] = np.zeros_like(env_scaled)
            w['raw_norm'] = np.zeros_like(raw_scaled)
        else:
            w['env_norm'] = env_scaled / supremo_pulso
            raw_norm = raw_scaled / supremo_pulso
            max_raw = float(np.max(raw_norm))
            if max_raw > 1.0:
                raw_norm = raw_norm / max_raw
            w['raw_norm'] = raw_norm

    # 7. Re-escalado Fisiológico por Promedio de Pulsos hacia 1.0 (Directiva del Usuario)
    # Todos los pulsos ya están normalizados por su supremo individual.
    if usar_calibracion_p95:
        _log("Calculando reescalado fisiológico sobre los pulsos normalizados...")
        promedios = {}
        for v in ['A', 'E', 'I', 'O', 'U']:
            wins_v = [w['env_norm'] for w in mediciones_datos if w['vocal'] == v]
            if wins_v:
                promedios[v] = np.mean(wins_v, axis=0)

        k_factors = np.ones(n_canales, dtype=np.float32)
        for i_pos, ch_idx in enumerate(indices_canales):
            if ch_idx == 0:
                # Canal 0: Milohioideo / Digástrico (diana: /a/)
                p_a = float(np.max(promedios['A'][i_pos])) if 'A' in promedios else 1.0
                k_factors[i_pos] = 1.0 / max(p_a, 1e-6)
            elif ch_idx == 1:
                # Canal 1: Depresor / Modiolo / Cigomático (diana: /i/ o /e/)
                p_i = float(np.max(promedios['I'][i_pos])) if 'I' in promedios else (
                    float(np.max(promedios['E'][i_pos])) if 'E' in promedios else 1.0
                )
                k_factors[i_pos] = 1.0 / max(p_i, 1e-6)
            elif ch_idx == 2:
                # Canal 2: Orbicular (diana: /u/ o /o/)
                p_u = float(np.max(promedios['U'][i_pos])) if 'U' in promedios else (
                    float(np.max(promedios['O'][i_pos])) if 'O' in promedios else 1.0
                )
                k_factors[i_pos] = 1.0 / max(p_u, 1e-6)

        # Salvaguardas de dominancia motora si los pares están presentes
        if 0 in indices_canales and 1 in indices_canales:
            i0 = indices_canales.index(0)
            i1 = indices_canales.index(1)
            if 'A' in promedios:
                p_1_a = float(np.max(promedios['A'][i1]))
                if (p_1_a * k_factors[i1]) >= 0.95:
                    k_factors[i1] = min(k_factors[i1], 0.85 / max(p_1_a, 1e-6))
            if 'I' in promedios:
                p_0_i = float(np.max(promedios['I'][i0]))
                if (p_0_i * k_factors[i0]) >= 0.95:
                    k_factors[i0] = min(k_factors[i0], 0.85 / max(p_0_i, 1e-6))

        if 0 in indices_canales and 2 in indices_canales:
            i0 = indices_canales.index(0)
            i2 = indices_canales.index(2)
            if 'U' in promedios:
                p_0_u = float(np.max(promedios['U'][i0]))
                if (p_0_u * k_factors[i0]) >= 0.95:
                    k_factors[i0] = min(k_factors[i0], 0.85 / max(p_0_u, 1e-6))

        if 1 in indices_canales and 2 in indices_canales:
            i1 = indices_canales.index(1)
            i2 = indices_canales.index(2)
            if 'I' in promedios:
                p_2_i = float(np.max(promedios['I'][i2]))
                if (p_2_i * k_factors[i2]) >= 0.95:
                    k_factors[i2] = min(k_factors[i2], 0.85 / max(p_2_i, 1e-6))

        k_vec = k_factors.reshape(n_canales, 1)
        k_str = ", ".join([f"Ch{indices_canales[c]}={k_factors[c]:.4f}" for c in range(n_canales)])
        _log(f"  [Reescalado Fisiológico Directo por Promedios] Factores: {k_str}")

        # Aplicar reescalado lineal directo sobre todos los pulsos normalizados
        for w in mediciones_datos:
            w['env_norm'] = np.maximum(0.0, w['env_norm'] * k_vec)
            w['raw_norm'] = np.maximum(0.0, w['raw_norm'] * k_vec)

    # 7. Consolidación de matrices para dataset NPZ
    _log("Consolidando matrices finales de dataset...")
    X_env_list = []
    X_cruda_list = []
    X_spec_list = []
    Y_list = []
    Tomas_list = []
    Fechas_list = []

    for w in mediciones_datos:
        spec_interp = procesar_stft_calibrada(w['raw_norm'], fs=fs_global, target_size=(32, 64))
        X_cruda_list.append(w['raw_norm'])
        X_env_list.append(w['env_norm'])
        X_spec_list.append(spec_interp)
        Y_list.append(w['vocal'])
        Tomas_list.append(w['toma'])
        Fechas_list.append(w['fecha'])

    X_cruda_np = np.array(X_cruda_list, dtype=np.float32)
    X_env_np = np.array(X_env_list, dtype=np.float32)
    X_spec_np = np.array(X_spec_list, dtype=np.float32)
    Y_np = np.array(Y_list)
    Tomas_np = np.array(Tomas_list)
    Fechas_np = np.array(Fechas_list)

    # 8. Purga única con Isolation Forest por clase de vocal
    c_pct = int(round(outlier_contamination * 100))
    _log(f"Aplicando purga no supervisada de anomalías con Isolation Forest ({c_pct}%) por clase de vocal...")
    mask_clean = np.ones(len(Y_np), dtype=bool)
    for v in ['A', 'E', 'I', 'O', 'U']:
        idx_v = np.where(Y_np == v)[0]
        if len(idx_v) > 6:
            feat_v = X_env_np[idx_v].reshape(len(idx_v), -1)
            iso = IsolationForest(contamination=float(outlier_contamination), random_state=42)
            preds_v = iso.fit_predict(feat_v)
            mask_clean[idx_v[preds_v == -1]] = False

    X_cruda_clean = X_cruda_np[mask_clean]
    X_env_clean = X_env_np[mask_clean]
    X_spec_clean = X_spec_np[mask_clean]
    Y_clean = Y_np[mask_clean]
    Tomas_clean = Tomas_np[mask_clean]
    Fechas_clean = Fechas_np[mask_clean]

    # 9. Guardado en jerarquía oficial de resultados con metadatos completos
    dir_salida_npz = carpeta_salida if carpeta_salida is not None else cache_dir
    os.makedirs(dir_salida_npz, exist_ok=True)
    archivo_npz = os.path.join(dir_salida_npz, "dataset_autoencoder_unificado.npz")
    np.savez_compressed(
        archivo_npz,
        X_cruda=X_cruda_clean,
        X_env=X_env_clean,
        X_spec=X_spec_clean,
        Y=Y_clean,
        Tomas=Tomas_clean,
        Fechas=Fechas_clean,
        Musculos_Canales=np.array(musculos_canales_global),
        bpm=bpm_global,
        fs=fs_global,
        tipo_envolvente=tipo_envolvente,
        smooth_ms=smooth_ms,
        canales_features=np.array(canales_features)
    )
    try:
        cache_npz = os.path.join(cache_dir, "dataset_autoencoder_unificado.npz")
        np.savez_compressed(cache_npz, X_cruda=X_cruda_clean, X_env=X_env_clean, X_spec=X_spec_clean, Y=Y_clean, Tomas=Tomas_clean, Fechas=Fechas_clean, Musculos_Canales=np.array(musculos_canales_global), bpm=bpm_global, fs=fs_global, tipo_envolvente=tipo_envolvente, smooth_ms=smooth_ms, canales_features=np.array(canales_features))
        legacy_npz = os.path.join(legacy_cache_dir, "dataset_autoencoder_unificado.npz")
        np.savez_compressed(
            legacy_npz,
            X_cruda=X_cruda_clean,
            X_env=X_env_clean,
            X_spec=X_spec_clean,
            Y=Y_clean,
            Tomas=Tomas_clean,
            Fechas=Fechas_clean,
            Musculos_Canales=np.array(musculos_canales_global),
            bpm=bpm_global,
            fs=fs_global,
            tipo_envolvente=tipo_envolvente,
            smooth_ms=smooth_ms,
            canales_features=np.array(canales_features)
        )
    except Exception:
        pass

    _log(f"Dataset consolidado guardado exitosamente en: {archivo_npz}")
    _log(f"Contracciones limpias preservadas: {len(Y_clean)}/{len(Y_np)} (Tiempo: {time.time()-t0:.1f} s)")

    return archivo_npz, len(Y_clean)

# ==============================================================================
# 4. ARQUITECTURAS DE AUTOENCODERS NO SUPERVISADOS (CERO ETIQUETAS)
# ==============================================================================

class AutoencoderEnvolvente1D(nn.Module):
    """
    Autoencoder Convolucional 1D para envolventes continuas con Invarianza Temporal:
    Utiliza convoluciones 1D para extraer dinámicas mioeléctricas locales, seguido de
    pooling dual (GAP + GMP) para capturar la integral de activación y el pico
    de contracción de forma completamente independiente de la longitud temporal de la ventana.
    """
    def __init__(self, in_channels=3, latent_dim=2, target_len=100):
        super(AutoencoderEnvolvente1D, self).__init__()
        self.in_channels = in_channels
        self.target_len = target_len
        self.latent_dim = latent_dim
        act = lambda: nn.LeakyReLU(0.1)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3), act(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), act(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2), act()
        )
        with torch.no_grad():
            num_features = self.conv(torch.zeros(1, in_channels, 100)).shape[1]

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc_enc = nn.Sequential(
            nn.Linear(num_features * 2, 32),
            act(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            act(),
            nn.Linear(32, in_channels * target_len)
        )

    def encode(self, x):
        h = self.conv(x)
        avg_f = self.gap(h).squeeze(-1)
        max_f = self.gmp(h).squeeze(-1)
        v = torch.cat([avg_f, max_f], dim=1)
        z = self.fc_enc(v)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decoder(z).view(-1, self.in_channels, self.target_len)
        if x.shape[-1] != self.target_len:
            x_rec = torch.nn.functional.interpolate(x_rec, size=x.shape[-1], mode='linear', align_corners=False)
        return x_rec, z

class AutoencoderConvGAPGMP(nn.Module):
    """
    Autoencoder Convolucional 1D con Invarianza de Fase Estocástica e Invarianza Temporal (Línea Base Histórica 54.3%):
    Utiliza núcleos amplios (k=31 ~ 15.5 ms a 2000 Hz) con activación ReLU para
    efectuar rectificación de onda completa/media onda aprendida, seguido de pooling
    dual (GAP + GMP) para capturar la energía integrada y el pico de contracción de forma
    independiente de la duración y desfasaje temporal.
    """
    def __init__(self, in_channels=3, latent_dim=2, num_filters=32, target_len=1000):
        super(AutoencoderConvGAPGMP, self).__init__()
        self.target_len = target_len
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=31, padding=15),
            nn.ReLU(),
            nn.Conv1d(num_filters, 16, kernel_size=15, padding=7),
            nn.ReLU()
        )
        with torch.no_grad():
            num_features = self.conv(torch.zeros(1, in_channels, 1000)).shape[1]

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc_enc = nn.Sequential(
            nn.Linear(num_features * 2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_channels * target_len)
        )

    def encode(self, x):
        h = self.conv(x)
        avg_f = self.gap(h).squeeze(-1)
        max_f = self.gmp(h).squeeze(-1)
        v = torch.cat([avg_f, max_f], dim=1)
        z = self.fc_enc(v)
        return z

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decoder(z).view(-1, self.in_channels, self.target_len)
        if x.shape[-1] != self.target_len:
            x_rec = torch.nn.functional.interpolate(x_rec, size=x.shape[-1], mode='linear', align_corners=False)
        return x_rec, z

AutoencoderCruda1D = AutoencoderConvGAPGMP

class AutoencoderEspectrograma2D(nn.Module):
    """Autoencoder 2D para espectrogramas calibrados en dB (3x32x64)"""
    def __init__(self, in_channels=3, latent_dim=2):
        super(AutoencoderEspectrograma2D, self).__init__()
        act = lambda: nn.LeakyReLU(0.1)

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), act(),
            nn.MaxPool2d(2, 2), # 16x32
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.MaxPool2d(2, 2), # 8x16
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.MaxPool2d(2, 2)  # 4x8
        )
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 64), act(),
            nn.Linear(64, latent_dim)
        )

        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64), act(),
            nn.Linear(64, 64 * 4 * 8), act()
        )
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), act())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), act())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), act())
        self.out_conv = nn.Conv2d(16, in_channels, 3, padding=1)

    def forward(self, x):
        h = self.enc(x)
        z = self.fc_enc(h)
        h_dec = self.fc_dec(z).view(z.shape[0], 64, 4, 8)
        d = self.dec3(self.up3(h_dec))
        d = self.dec2(self.up2(d))
        d = self.dec1(self.up1(d))
        rec = self.out_conv(d)
        return rec, z

class OrthogonalAutoencoder2D(nn.Module):
    """
    Autoencoder Ortogonal 2D para descubrimiento no supervisado de variedades latentes sEMG.
    Arquitectura totalmente conexa simétrica sin sesgo (bias=False) con regularización ortogonal:
    D -> hidden_dim (32) -> 16 -> latent_dim (2) -> 16 -> hidden_dim (32) -> D
    Admite entrada aplanada (N, D) o tridimensional (N, n_canales, target_len).
    """
    def __init__(self, input_dim=60, hidden_dim=32, latent_dim=2):
        super(OrthogonalAutoencoder2D, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 16, bias=False)
        self.fc3 = nn.Linear(16, latent_dim, bias=False)
        self.act = nn.Tanh()
        self.dfc1 = nn.Linear(latent_dim, 16, bias=False)
        self.dfc2 = nn.Linear(16, hidden_dim, bias=False)
        self.dfc3 = nn.Linear(hidden_dim, input_dim, bias=False)

    def encode(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        z = self.fc3(h2)
        return z

    def decode(self, z):
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        recon = self.dfc3(dh2)
        return recon

    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.contiguous().view(x.shape[0], -1)
        else:
            x_flat = x
        z = self.encode(x_flat)
        recon_flat = self.decode(z)
        if len(orig_shape) == 3:
            recon = recon_flat.view(orig_shape)
        else:
            recon = recon_flat
        return recon, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.fc3, self.dfc1, self.dfc2, self.dfc3]:
            W = layer.weight
            if W.shape[0] < W.shape[1]:
                gram = torch.mm(W, W.t())
                I = torch.eye(W.shape[0], device=W.device)
            else:
                gram = torch.mm(W.t(), W)
                I = torch.eye(W.shape[1], device=W.device)
            loss += torch.norm(gram - I, p='fro')**2
        return loss

def extraer_sesion_agnostica(toma_str):
    """Extrae el identificador de sesión (ej. 'T1', 'T2', 'S1', 'PRUEBA1') de una cadena de toma."""
    s = str(toma_str)
    m = re.search(r'(Prueba\d+|Sesion\d+|Session\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    parts = s.split('_')
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win'):
            continue
        if any(char.isdigit() for char in p_clean) and len(p_clean) <= 10:
            return p_clean.upper()
    return 'S1'

def acondicionar_reposo_impedancia(X_array, sesiones, n_canales=3, n_pts_reposo=10):
    """
    Acondicionamiento por reposo basal pre-contracción y rango dinámico P95 por sesión y canal.
    Normaliza cada canal muscular para que el silencio basal sea 0.0 y el pico de activación sea ~1.0.
    """
    orig_shape = X_array.shape
    if X_array.ndim == 2:
        N, D = X_array.shape
        n_pts = D // n_canales
        X_reshaped = X_array.reshape(N, n_canales, n_pts).copy()
    else:
        N, n_canales, n_pts = X_array.shape
        X_reshaped = X_array.copy()

    X_filt = np.zeros_like(X_reshaped)
    if n_pts >= 12:
        try:
            b, a = butter(N=3, Wn=0.3, btype='low')
            for i in range(N):
                for c in range(n_canales):
                    X_filt[i, c, :] = filtfilt(b, a, X_reshaped[i, c, :])
        except Exception:
            X_filt = X_reshaped.copy()
    else:
        X_filt = X_reshaped.copy()

    unique_ses = np.unique(sesiones)
    X_norm = np.zeros_like(X_filt)
    pts_base = max(1, min(n_pts_reposo, n_pts // 4))

    for s in unique_ses:
        mask = (sesiones == s)
        for c in range(n_canales):
            base_mean = np.mean(X_filt[mask, c, :pts_base])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    if len(orig_shape) == 2:
        return X_norm.reshape(N, -1)
    return X_norm

def extraer_4_vertices(z_ses):
    """Extrae 4 vértices extremos no supervisados de la variedad latente para alineación topológica."""
    gmm_b = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=5)
    labels = gmm_b.fit_predict(z_ses)
    idx_up = 0 if np.mean(z_ses[labels == 0, 1]) > np.mean(z_ses[labels == 1, 1]) else 1
    z_up = z_ses[labels == idx_up]
    z_low = z_ses[labels == (1 - idx_up)]
    vert_U = z_up[np.argmin(z_up[:, 0])]
    vert_O = z_up[np.argmax(z_up[:, 0])]
    vert_I = z_low[np.argmin(z_low[:, 0])]
    vert_A = z_low[np.argmax(z_low[:, 0])]
    return np.array([vert_A, vert_I, vert_O, vert_U])

def alinear_topologia_sesiones_so2(Z, sesiones, ref_session='T2'):
    """
    Alinea rígidamente en SO(2) los planos latentes inter-sesión mediante el algoritmo de Kabsch.
    Garantiza det(R) = +1 evitando reflexiones espurias.
    """
    unique_ses = np.unique(sesiones)
    if len(unique_ses) <= 1:
        return Z.copy()
    if ref_session not in unique_ses:
        ref_session = unique_ses[0]

    mask_ref = (sesiones == ref_session)
    L_ref = extraer_4_vertices(Z[mask_ref])
    mu_ref = np.mean(L_ref, axis=0)
    L_ref_c = L_ref - mu_ref

    Z_aligned = np.zeros_like(Z)
    for ses in unique_ses:
        mask_s = (sesiones == ses)
        z_s = Z[mask_s]
        L_s = extraer_4_vertices(z_s)
        mu_s = np.mean(L_s, axis=0)
        L_s_c = L_s - mu_s
        M = np.dot(L_s_c.T, L_ref_c)
        U, S, Vt = np.linalg.svd(M)
        R = np.dot(U, Vt)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = np.dot(U, Vt)
        Z_aligned[mask_s] = np.dot(z_s - mu_s, R) + mu_ref
    return Z_aligned


def compilar_modelo_desde_codigo(codigo_str, modalidad="envolvente", latent_dim=2, target_len=None, in_channels=3):
    """
    Compila dinámicamente una arquitectura PyTorch desde una cadena de código,
    localizando la clase que hereda de nn.Module e instanciándola con los parámetros adecuados.
    """
    if not codigo_str or not codigo_str.strip():
        raise ValueError("El código de la arquitectura no puede estar vacío.")

    import random
    from torch.utils.data import DataLoader, TensorDataset

    espacio_local = {
        'torch': torch,
        'nn': nn,
        'F': torch.nn.functional,
        'optim': optim,
        'np': np,
        'random': random,
        'DataLoader': DataLoader,
        'TensorDataset': TensorDataset,
        'SEED': 42,
        'LATENT_DIM': latent_dim,
        'LAMBDA_RELATIVE': 1.0,
        'LAMBDA_DERIV': 1.0,
        'LOSS_EPS': 1e-8,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    try:
        exec(codigo_str, espacio_local)
    except Exception as e:
        raise SyntaxError(f"Error de sintaxis o ejecución al compilar el código PyTorch: {e}")

    # Búsqueda de clases nn.Module definidas en el código
    candidatos = []
    for k, v in espacio_local.items():
        if isinstance(v, type) and issubclass(v, nn.Module) and v is not nn.Module:
            candidatos.append((k, v))

    if not candidatos:
        raise ValueError("No se encontró ninguna clase que herede de nn.Module en el código proporcionado.")

    # Priorizar la clase denominada 'AutoencoderPersonalizado' si existe
    cls_modelo = None
    for nom, cls_c in candidatos:
        if nom == 'AutoencoderPersonalizado':
            cls_modelo = cls_c
            break
    if cls_modelo is None:
        cls_modelo = candidatos[-1][1]

    # Determinación de parámetros de inicialización según la firma de __init__
    import inspect
    sig = inspect.signature(cls_modelo.__init__)
    kwargs_init = {}
    if 'in_channels' in sig.parameters:
        kwargs_init['in_channels'] = in_channels
    if 'latent_dim' in sig.parameters:
        kwargs_init['latent_dim'] = latent_dim
    if 'target_len' in sig.parameters:
        p_target = sig.parameters['target_len']
        if p_target.default is inspect.Parameter.empty:
            if target_len is not None:
                kwargs_init['target_len'] = target_len
        # Si el usuario definió un valor por defecto en su código (ej. target_len=1000 o target_len=100),
        # se respeta rígidamente su valor y no se altera.

    try:
        modelo = cls_modelo(**kwargs_init)
    except TypeError:
        # Fallback intentando instanciar sin parámetros o con parámetros posicionales
        try:
            modelo = cls_modelo(in_channels=in_channels, latent_dim=latent_dim)
        except Exception:
            modelo = cls_modelo()

    # Detección de función de pérdida personalizada definida en el código del usuario
    for fn_name in ('reconstruction_loss', 'custom_loss', 'loss_fn', 'criterio_loss'):
        if fn_name in espacio_local and callable(espacio_local[fn_name]):
            modelo.custom_loss_fn = espacio_local[fn_name]
            break

    return modelo

def verificar_arquitectura_codigo(codigo_str, modalidad="envolvente", latent_dim=2, target_len=None, in_channels=3):
    """
    Verifica si una cadena de código PyTorch compila y puede ejecutar un paso hacia adelante
    con un tensor sintético acorde a la modalidad seleccionada.
    Retorna (valido: bool, mensaje: str).
    """
    try:
        t_len = target_len if target_len is not None else (1000 if modalidad == "cruda" else 100)
        modelo = compilar_modelo_desde_codigo(codigo_str, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_channels)
        modelo.eval()

        if hasattr(modelo, 'target_len') and modelo.target_len is not None and isinstance(modelo.target_len, int) and modelo.target_len > 0:
            t_len = modelo.target_len

        if modalidad.startswith("espectrograma"):
            dummy_x = torch.zeros(2, in_channels, 32, 64, dtype=torch.float32)
        else:
            dummy_x = torch.zeros(2, in_channels, t_len, dtype=torch.float32)

        with torch.no_grad():
            res = modelo(dummy_x)

        if not isinstance(res, (tuple, list)) or len(res) != 2:
            return False, "La red debe retornar una tupla (x_rec, z) en forward(x)."

        rec, z = res
        if not isinstance(z, torch.Tensor) or not isinstance(rec, torch.Tensor):
            return False, "Los elementos retornados (x_rec, z) deben ser tensores de PyTorch."

        if z.shape != (2, latent_dim):
            return False, f"Dimensión latente incorrecta: se esperaba (2, {latent_dim}) y se obtuvo {tuple(z.shape)}."

        if rec.shape != dummy_x.shape:
            return False, f"Dimensión de reconstrucción incorrecta: se esperaba {tuple(dummy_x.shape)} y se obtuvo {tuple(rec.shape)}."

        n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        return True, f"Arquitectura válida ({modelo.__class__.__name__} | {n_params:,} parámetros entrenables). Dimensiones verificadas."
    except Exception as e:
        msg = str(e)
        if "mat1 and mat2 shapes cannot be multiplied" in msg:
            import re
            m = re.search(r'\((\d+)x(\d+) and (\d+)x(\d+)\)', msg)
            if m:
                b, in_feat, exp_feat, out_feat = m.groups()
                return False, (
                    f"Desajuste de dimensiones lineales: mat1 ({b}x{in_feat}) y mat2 ({exp_feat}x{out_feat}). "
                    f"La etapa convolucional produce {in_feat} características para una longitud temporal de entrada de {t_len} muestras, "
                    f"pero la capa lineal espera {exp_feat}. "
                    f"La arquitectura ConvAE requiere que 'Puntos Envolvente' en la Sección 4 esté configurado exactamente en 100 muestras "
                    f"(o ajustar nn.Linear({in_feat}, ...) en la arquitectura)."
                )
        return False, f"Error al validar arquitectura: {e}"

def crear_modelo_autoencoder(modalidad="envolvente", latent_dim=2, codigo_custom=None, target_len=None, in_channels=3):
    """Instancia la arquitectura de autoencoder correspondiente según la modalidad fisiológica, dimensión o código personalizado."""
    if codigo_custom and codigo_custom.strip():
        return compilar_modelo_desde_codigo(codigo_custom, modalidad=modalidad, latent_dim=latent_dim, target_len=target_len, in_channels=in_channels)

    if modalidad == "envolvente":
        t_len = target_len if target_len is not None else 100
        return AutoencoderEnvolvente1D(in_channels=in_channels, latent_dim=latent_dim, target_len=t_len)
    elif modalidad == "cruda":
        t_len = target_len if target_len is not None else 1000
        return AutoencoderCruda1D(in_channels=in_channels, latent_dim=latent_dim, target_len=t_len)
    elif modalidad.startswith("espectrograma"):
        return AutoencoderEspectrograma2D(in_channels=in_channels, latent_dim=latent_dim)
    else:
        raise ValueError(f"Modalidad desconocida: {modalidad}")

AutoencoderGenerico = crear_modelo_autoencoder

# ==============================================================================
# 5. ENTRENAMIENTO NO SUPERVISADO (ZERO-LABELS)
# ==============================================================================

def entrenar_autoencoder(archivo_npz=None, modalidad="envolvente", latent_dim=2, epochs=150, batch_size=32, lr=0.002, carpeta_salida=None, callback_log=None, npz_path=None, usar_custom_arch=False, codigo_custom_arch=None, tipo_perdida="mse", gamma_sdtw=1.0, alpha_hibrida=1.0, lambda_orto=0.0, tipo_arquitectura="ortogonal", lambda_w=0.30, lambda_z=0.45, usar_impedancia_reposo=True, usar_alineacion_so2=True, ref_session='T2'):
    fijar_semilla(seed=42)
    archivo_npz = archivo_npz or npz_path
    if not archivo_npz:
        raise ValueError("Debe especificarse el archivo NPZ del dataset.")
    def _log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg, flush=True)

    if str(archivo_npz).lower().endswith('.csv'):
        df_csv = pd.read_csv(archivo_npz)
        y = df_csv['Vocal'].values if 'Vocal' in df_csv.columns else np.array(['A'] * len(df_csv))
        tomas = df_csv['Toma'].values if 'Toma' in df_csv.columns else np.array([f"T1_p{i}" for i in range(len(df_csv))])
        cols_feat = [c for c in df_csv.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
        X_raw = df_csv[cols_feat].values
        n_ch = 3
        n_pts = X_raw.shape[1] // n_ch
        X_env_csv = X_raw.reshape(len(df_csv), n_ch, n_pts)
        datos = {
            'X_env': X_env_csv,
            'X_cruda': X_env_csv,
            'X_spec': X_env_csv,
            'Y': y,
            'Tomas': tomas,
            'Fechas': np.array(['2026-07-10'] * len(df_csv)),
            'Musculos_Canales': np.array(["Canal 0", "Canal 1", "Canal 2"])
        }
    else:
        datos = np.load(archivo_npz)
    tipo_arq = str(tipo_arquitectura).lower().strip()
    es_ortogonal = (tipo_arq in ("ortogonal", "orthogonal", "record", "record_91", "optimo"))

    if modalidad == "envolvente":
        X = datos['X_env']
        in_ch = X.shape[1] if X.ndim > 2 else 3
        t_len = X.shape[-1] if X.ndim > 2 else (X.shape[1] // in_ch)
        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  [Arquitectura Personalizada] Compilando modelo desde editor de código...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_ch)
            _log(f"  [Arquitectura Personalizada] Modelo instanciado: {modelo.__class__.__name__}")
        elif es_ortogonal:
            if usar_impedancia_reposo:
                tomas = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
                sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas])
                _log(f"  [Reposo Basal e Impedancia] Acondicionando {len(X)} ventanas para {len(np.unique(sesiones))} sesiones...")
                X = acondicionar_reposo_impedancia(X, sesiones, n_canales=in_ch, n_pts_reposo=10)
            input_dim_total = in_ch * t_len
            modelo = OrthogonalAutoencoder2D(input_dim=input_dim_total, hidden_dim=32, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado: {input_dim_total} -> 32 -> 16 -> {latent_dim} (Tanh, bias=False)")
            if batch_size == 32 or batch_size < len(X):
                batch_size = len(X)
                _log(f"  [Régimen Ortogonal] Batch size ajustado a Full-Batch ({batch_size} muestras) para estabilidad de Cov(Z).")
        else:
            modelo = AutoencoderEnvolvente1D(in_channels=in_ch, latent_dim=latent_dim, target_len=t_len)
    elif modalidad == "cruda":
        X = datos['X_cruda']
        in_ch = X.shape[1] if X.ndim > 2 else 3
        t_len = X.shape[-1] if X.ndim > 2 else 1000
        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  [Arquitectura Personalizada] Compilando modelo desde editor de código...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_ch)
            _log(f"  [Arquitectura Personalizada] Modelo instanciado: {modelo.__class__.__name__}")
        elif es_ortogonal:
            if usar_impedancia_reposo:
                tomas = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
                sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas])
                _log(f"  [Reposo Basal e Impedancia] Acondicionando {len(X)} ventanas para {len(np.unique(sesiones))} sesiones...")
                X = acondicionar_reposo_impedancia(X, sesiones, n_canales=in_ch, n_pts_reposo=10)
            input_dim_total = in_ch * t_len
            modelo = OrthogonalAutoencoder2D(input_dim=input_dim_total, hidden_dim=32, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado: {input_dim_total} -> 32 -> 16 -> {latent_dim} (Tanh, bias=False)")
            if batch_size == 32 or batch_size < len(X):
                batch_size = len(X)
                _log(f"  [Régimen Ortogonal] Batch size ajustado a Full-Batch ({batch_size} muestras).")
        else:
            modelo = AutoencoderCruda1D(in_channels=in_ch, latent_dim=latent_dim, target_len=t_len)
        if tipo_perdida == "mse" and not es_ortogonal:
            if batch_size == 32:
                batch_size = len(X)
                _log(f"  [Régimen Cruda MSE] Batch size ajustado a Full-Batch ({batch_size} muestras) según línea base.")
            if lr == 0.002:
                lr = 0.008
                _log(f"  [Régimen Cruda MSE] Learning rate ajustado a {lr} según línea base.")
            if epochs == 150:
                epochs = 250
                _log(f"  [Régimen Cruda MSE] Épocas ajustadas a {epochs} según línea base.")
    elif modalidad.startswith("espectrograma"):
        X = datos['X_spec']
        in_ch = X.shape[1]
        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  [Arquitectura Personalizada] Compilando modelo desde editor de código...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, in_channels=in_ch)
            _log(f"  [Arquitectura Personalizada] Modelo instanciado: {modelo.__class__.__name__}")
        else:
            modelo = AutoencoderEspectrograma2D(in_channels=in_ch, latent_dim=latent_dim)
    else:
        raise ValueError(f"Modalidad desconocida: {modalidad}")

    tensor_x = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=True)

    optimizador = optim.Adam(modelo.parameters(), lr=lr, weight_decay=1e-5)

    # Configuración de función de pérdida (MSE, Soft-DTW, Divergencia Soft-DTW, Híbrida o Personalizada)
    tipo_perdida = (tipo_perdida or "mse").lower().strip()
    if hasattr(modelo, 'custom_loss_fn') and modelo.custom_loss_fn is not None:
        criterio = modelo.custom_loss_fn
        fn_name = getattr(modelo.custom_loss_fn, '__name__', 'custom_loss')
        nombre_loss = f"Personalizada del Editor ({fn_name})"
        _log(f"  [Pérdida del Editor] Utilizando directamente la función '{fn_name}' definida en el código.")
    elif modalidad.startswith("espectrograma") and tipo_perdida != "mse":
        _log("  [Aviso] La modalidad espectrograma es 2D; se utiliza MSE como función de pérdida estándar.")
        criterio = nn.MSELoss()
        nombre_loss = "MSE"
    elif tipo_perdida in ("soft_dtw_divergence", "divergencia_soft_dtw", "divergencia"):
        if SoftDTWDivergence is not None:
            criterio = SoftDTWDivergence(gamma=gamma_sdtw)
            nombre_loss = f"Soft-DTW Divergencia (gamma={gamma_sdtw})"
        else:
            _log("  [Aviso] Módulo SoftDTW no disponible; utilizando MSE.")
            criterio = nn.MSELoss()
            nombre_loss = "MSE"
    elif tipo_perdida in ("hibrida", "hybrid", "mse_sdtw"):
        if HibridaLoss is not None:
            criterio = HibridaLoss(gamma=gamma_sdtw, alpha=alpha_hibrida, normalize_sdtw=True)
            nombre_loss = f"Híbrida MSE+sDTW (gamma={gamma_sdtw}, alpha={alpha_hibrida})"
        else:
            _log("  [Aviso] Módulo SoftDTW no disponible; utilizando MSE.")
            criterio = nn.MSELoss()
            nombre_loss = "MSE"
    elif tipo_perdida in ("soft_dtw", "sdtw"):
        if SoftDTW is not None:
            criterio = SoftDTW(gamma=gamma_sdtw, normalize=False)
            nombre_loss = f"Soft-DTW (gamma={gamma_sdtw})"
        else:
            _log("  [Aviso] Módulo SoftDTW no disponible; utilizando MSE.")
            criterio = nn.MSELoss()
            nombre_loss = "MSE"
    elif tipo_perdida in ("multiobjetivo", "convae", "relativa_derivada"):
        def criterio_multiobjetivo(xhat, x, lambda_rel=1.0, lambda_deriv=1.0, eps=1e-8):
            mse_abs = torch.mean((xhat - x) ** 2)
            signal_energy = torch.mean(x ** 2, dim=2).clamp_min(eps)
            error_energy = torch.mean((xhat - x) ** 2, dim=2)
            mse_relative = torch.mean(error_energy / signal_energy)
            dx = x[:, :, 1:] - x[:, :, :-1]
            dxhat = xhat[:, :, 1:] - xhat[:, :, :-1]
            deriv_energy = torch.mean(dx ** 2, dim=2).clamp_min(eps)
            deriv_error = torch.mean((dxhat - dx) ** 2, dim=2)
            mse_deriv = torch.mean(deriv_error / deriv_energy)
            return mse_abs + lambda_rel * mse_relative + lambda_deriv * mse_deriv
        criterio = criterio_multiobjetivo
        nombre_loss = "Multiobjetivo (Absoluta + Relativa + Derivada)"
    else:
        criterio = nn.MSELoss()
        nombre_loss = "MSE"

    orto_info = []
    if lambda_w > 0 and hasattr(modelo, 'weight_orthogonality_loss'):
        orto_info.append(f"Orto-W(\u03bb={lambda_w})")
    if lambda_z > 0:
        orto_info.append(f"Orto-Z(\u03bb={lambda_z})")
    elif lambda_orto > 0:
        orto_info.append(f"Orto-Lat(\u03bb={lambda_orto})")
    orto_txt = f" + {' + '.join(orto_info)}" if orto_info else ""

    _log(f"Iniciando entrenamiento ({modalidad.upper()}, Latent: {latent_dim}D, Pérdida: {nombre_loss}{orto_txt}, Épocas: {epochs})...")
    t0 = time.time()

    for ep in range(epochs):
        t_ep_start = time.time()
        modelo.train()
        loss_ep = 0.0
        for (batch,) in loader:
            optimizador.zero_grad()
            rec, z = modelo(batch)
            loss_rec = criterio(rec, batch)
            
            # 1. Regularización de ortogonalidad de pesos (W W^T - I)
            if hasattr(modelo, 'weight_orthogonality_loss') and lambda_w > 0:
                loss_w = modelo.weight_orthogonality_loss()
            else:
                loss_w = torch.tensor(0.0, device=batch.device)

            # 2. Regularización de decorrelación y esfericidad latente Cov(Z) - I
            if lambda_z > 0 and z.shape[0] > 1:
                z_centered = z - torch.mean(z, dim=0, keepdim=True)
                cov_z = torch.matmul(z_centered.t(), z_centered) / max(1, (z.shape[0] - 1))
                eye_lat = torch.eye(z.shape[1], device=z.device)
                loss_z = torch.norm(cov_z - eye_lat, p='fro') ** 2
            elif lambda_orto > 0 and z.shape[0] > 1:
                z_centered = z - torch.mean(z, dim=0, keepdim=True)
                cov_z = torch.matmul(z_centered.t(), z_centered) / (z.shape[0] - 1)
                traza = torch.trace(cov_z) + 1e-8
                cov_norm = cov_z / traza
                eye = torch.eye(cov_norm.shape[0], device=cov_norm.device)
                loss_z = torch.sum((cov_norm * (1.0 - eye)) ** 2) * float(lambda_orto)
            else:
                loss_z = torch.tensor(0.0, device=batch.device)

            loss_total = loss_rec + float(lambda_w) * loss_w + float(lambda_z) * loss_z
            loss_total.backward()
            optimizador.step()
            loss_ep += loss_total.item() * batch.size(0)

        dt_ep = time.time() - t_ep_start
        # Monitoreo obligatorio de progreso: época 1, cada 10 épocas y final con porcentaje y ETA
        if ep == 0 or (ep + 1) % 10 == 0 or ep == epochs - 1:
            pct = ((ep + 1) / epochs) * 100.0
            eta_s = (epochs - (ep + 1)) * dt_ep
            _log(f"  Época [{ep+1:3d}/{epochs}] ({pct:5.1f}%) - Pérdida: {loss_ep/len(tensor_x):.5f} | ETA: {eta_s:.1f}s")

    _log(f"Entrenamiento completado en {time.time()-t0:.1f} s.")
    
    # Guardado del modelo en carpeta de corrida y en modelos_dir
    dir_salida_mod = carpeta_salida if carpeta_salida is not None else modelos_dir
    os.makedirs(dir_salida_mod, exist_ok=True)
    if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
        arch_py_path = os.path.join(dir_salida_mod, "arquitectura_autoencoder.py")
        try:
            with open(arch_py_path, 'w', encoding='utf-8') as f_arch:
                f_arch.write(codigo_custom_arch)
            _log(f"  Código de arquitectura personalizada guardado en: {arch_py_path}")
        except Exception:
            pass

    modelo_pth = os.path.join(dir_salida_mod, f"autoencoder_{modalidad}_{latent_dim}d.pth")
    torch.save(modelo.state_dict(), modelo_pth)
    try:
        torch.save(modelo.state_dict(), os.path.join(modelos_dir, f"autoencoder_{modalidad}_{latent_dim}d.pth"))
        torch.save(modelo.state_dict(), os.path.join(resultados_dir, f"autoencoder_{modalidad}_{latent_dim}d.pth"))
        torch.save(modelo.state_dict(), os.path.join(resultados_dir, f"autoencoder_unificado_{modalidad}_{latent_dim}d.pth"))
    except Exception:
        pass
    info_config = {
        'tipo_arquitectura': tipo_arquitectura,
        'modalidad': modalidad,
        'latent_dim': latent_dim,
        'epochs': epochs,
        'lr': lr,
        'batch_size': batch_size,
        'lambda_w': float(lambda_w),
        'lambda_z': float(lambda_z),
        'usar_impedancia_reposo': bool(usar_impedancia_reposo),
        'usar_alineacion_so2': bool(usar_alineacion_so2),
        'ref_session': str(ref_session)
    }
    try:
        cfg_path = os.path.join(dir_salida_mod, "config_autoencoder.json")
        with open(cfg_path, 'w', encoding='utf-8') as f_cfg:
            json.dump(info_config, f_cfg, indent=4)
    except Exception:
        pass

    return modelo, modelo_pth

# ==============================================================================
# 6. EVALUACIÓN Y PLOTEO LATENTE (ANCLAJE CANÓNICO + GMM)
# ==============================================================================

def alinear_canonicamente(z, y_labels):
    z_centrado = z - np.mean(z, axis=0)
    idx_a = np.where(y_labels == 'A')[0]
    if len(idx_a) == 0:
        return z_centrado
    c_a = np.mean(z_centrado[idx_a], axis=0)
    theta_a = np.arctan2(c_a[1], c_a[0])
    alpha = (np.pi / 2.0) - theta_a
    R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    z_rot = z_centrado.copy()
    z_rot[:, :2] = z_centrado[:, :2] @ R.T

    idx_sonrisa = np.where(np.isin(y_labels, ['I', 'E']))[0]
    if len(idx_sonrisa) > 0 and np.mean(z_rot[idx_sonrisa, 0]) < 0:
        z_rot[:, 0] = -z_rot[:, 0]
    return z_rot

def evaluar_espacio_latente(archivo_npz=None, modelo=None, modalidad="envolvente", latent_dim=2, carpeta_salida=None, callback_log=None, npz_path=None, usar_custom_arch=False, codigo_custom_arch=None, mostrar_grafico=True, algoritmo_clustering="gmm", usar_alineacion_so2=None, ref_session=None):
    fijar_semilla(seed=42)
    archivo_npz = archivo_npz or npz_path
    if not archivo_npz:
        raise ValueError("Debe especificarse el archivo NPZ del dataset.")
    def _log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg, flush=True)

    if str(archivo_npz).lower().endswith('.csv'):
        df_csv = pd.read_csv(archivo_npz)
        y = df_csv['Vocal'].values if 'Vocal' in df_csv.columns else np.array(['A'] * len(df_csv))
        tomas = df_csv['Toma'].values if 'Toma' in df_csv.columns else np.array([f"T1_p{i}" for i in range(len(df_csv))])
        cols_feat = [c for c in df_csv.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
        X_raw = df_csv[cols_feat].values
        n_ch = 3
        n_pts = X_raw.shape[1] // n_ch
        X_env_csv = X_raw.reshape(len(df_csv), n_ch, n_pts)
        datos = {
            'X_env': X_env_csv,
            'X_cruda': X_env_csv,
            'X_spec': X_env_csv,
            'Y': y,
            'Tomas': tomas,
            'Fechas': np.array(['2026-07-10'] * len(df_csv)),
            'Musculos_Canales': np.array(["Canal 0", "Canal 1", "Canal 2"])
        }
    else:
        datos = np.load(archivo_npz)
    if modalidad == "envolvente":
        X = datos['X_env']
    elif modalidad == "cruda":
        X = datos['X_cruda']
    else:
        X = datos['X_spec']
    Y_labels = datos['Y']

    dir_salida_mod = carpeta_salida if carpeta_salida is not None else modelos_dir
    cfg_auto = {}
    cfg_path = os.path.join(dir_salida_mod, "config_autoencoder.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r', encoding='utf-8') as f_cfg:
                cfg_auto = json.load(f_cfg)
        except Exception:
            pass

    tipo_arq = cfg_auto.get('tipo_arquitectura', 'ortogonal').lower().strip()
    es_orto = (tipo_arq in ("ortogonal", "orthogonal", "record", "record_91", "optimo"))
    imp_reposo = cfg_auto.get('usar_impedancia_reposo', True)
    if usar_alineacion_so2 is None:
        usar_alineacion_so2 = cfg_auto.get('usar_alineacion_so2', True)
    if ref_session is None:
        ref_session = cfg_auto.get('ref_session', 'T2')

    in_ch = X.shape[1] if X.ndim > 2 else 3
    t_len = X.shape[-1] if X.ndim > 2 else (X.shape[1] // in_ch)

    if imp_reposo and es_orto:
        tomas_raw = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
        sesiones_raw = np.array([extraer_sesion_agnostica(t) for t in tomas_raw])
        _log(f"  [Acondicionamiento Reposo/Impedancia] Acondicionando {len(X)} ventanas para evaluación...")
        X = acondicionar_reposo_impedancia(X, sesiones_raw, n_canales=in_ch, n_pts_reposo=10)

    if modelo is None:
        arch_guardada = os.path.join(dir_salida_mod, "arquitectura_autoencoder.py")
        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  Cargando arquitectura personalizada desde parámetros...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_ch)
        elif es_orto:
            modelo = OrthogonalAutoencoder2D(input_dim=in_ch * t_len, hidden_dim=32, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado para inferencia ({in_ch * t_len} -> 32 -> 16 -> {latent_dim})")
        elif os.path.exists(arch_guardada):
            try:
                with open(arch_guardada, 'r', encoding='utf-8') as f_a:
                    cod_recup = f_a.read()
                modelo = compilar_modelo_desde_codigo(cod_recup, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len)
                _log(f"  Arquitectura personalizada recuperada desde: {arch_guardada}")
            except Exception:
                modelo = AutoencoderGenerico(modalidad=modalidad, latent_dim=latent_dim, target_len=t_len)
        else:
            modelo = AutoencoderGenerico(modalidad=modalidad, latent_dim=latent_dim, target_len=t_len)

        pth_candidatos = [
            os.path.join(dir_salida_mod, f"autoencoder_{modalidad}_{latent_dim}d.pth"),
            os.path.join(modelos_dir, f"autoencoder_{modalidad}_{latent_dim}d.pth"),
            os.path.join(resultados_dir, f"autoencoder_{modalidad}_{latent_dim}d.pth")
        ]
        cargado = False
        for pth in pth_candidatos:
            if os.path.exists(pth):
                modelo.load_state_dict(torch.load(pth, map_location='cpu'))
                _log(f"Modelo cargado exitosamente desde: {pth}")
                cargado = True
                break
        if not cargado:
            raise FileNotFoundError(f"No se encontró un modelo entrenado para la modalidad '{modalidad}' ({latent_dim}D).")

    modelo.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(X, dtype=torch.float32)
        rec_tensor, z_tensor = modelo(tensor_x)
        z = z_tensor.numpy()
        x_rec = rec_tensor.numpy()

    tomas = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
    sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas])

    # Guardar proyecciones latentes crudas
    try:
        df_crudo = pd.DataFrame({'Vocal': Y_labels, 'Toma': tomas, 'Sesion': sesiones, 'Z1': z[:, 0], 'Z2': z[:, 1]})
        df_crudo.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d_crudo.csv"), index=False)
    except Exception:
        pass

    # 1. Alineación Topológica Determinística SO(2) o Canónica
    if usar_alineacion_so2 and latent_dim == 2 and len(np.unique(sesiones)) > 1:
        _log(f"  [Alineación Topológica SO(2)] Alineando sesiones respecto a referencia '{ref_session}'...")
        z_align = alinear_topologia_sesiones_so2(z, sesiones, ref_session=ref_session)
        try:
            df_alin = pd.DataFrame({'Vocal': Y_labels, 'Toma': tomas, 'Sesion': sesiones, 'Z1': z_align[:, 0], 'Z2': z_align[:, 1]})
            df_alin.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d_alineado.csv"), index=False)
            df_alin.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass
    else:
        z_align = alinear_canonicamente(z, Y_labels)
        try:
            df_crudo.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass

    # 2. Purga Robusta de Outliers en Espacio Latente (Regla 9 de Memoria)
    # Descarta puntos atípicos aislados para que no distorsionen la escala ni las elipses GMM
    z_center = np.median(z_align, axis=0)
    dist_lat = np.linalg.norm(z_align - z_center, axis=1)
    q1_d, q3_d = np.percentile(dist_lat, 25), np.percentile(dist_lat, 75)
    iqr_d = q3_d - q1_d
    umbral_outlier = q3_d + 2.5 * iqr_d if iqr_d > 1e-9 else 1e9
    mask_inliers = dist_lat <= umbral_outlier
    n_outliers = int(np.sum(~mask_inliers))

    if n_outliers > 0:
        _log(f"  [Purga Latente] {n_outliers} puntos atípicos aislados purgados (distancia IQR > {umbral_outlier:.2f})")

    z_eval = z_align[mask_inliers]
    Y_eval = Y_labels[mask_inliers]
    X_eval = X[mask_inliers]
    x_rec_eval = x_rec[mask_inliers]

    Fechas = datos['Fechas'] if 'Fechas' in datos else np.array(["Fecha_Desconocida"] * len(Y_labels))
    Fechas_eval = Fechas[mask_inliers] if len(Fechas) == len(mask_inliers) else np.array(["Fecha_Desconocida"] * len(z_eval))
    fechas_unicas = sorted(list(set(str(f) for f in Fechas_eval if str(f))))

    Musculos_Canales = list(datos['Musculos_Canales']) if 'Musculos_Canales' in datos else ["Canal 0", "Canal 1", "Canal 2"]
    while len(Musculos_Canales) < 3:
        Musculos_Canales.append(f"Canal {len(Musculos_Canales)}")

    MARKERS = ['o', 's', '^', 'D', 'v', 'p', '*']
    marker_por_fecha = {f: MARKERS[i % len(MARKERS)] for i, f in enumerate(fechas_unicas)}

    # Diagnóstico de Clustering no supervisado con Asignación Húngara (GMM o K-Means)
    map_v = {'A': 0, 'E': 1, 'I': 2, 'O': 3, 'U': 4}
    y_num = np.array([map_v[v] for v in Y_eval])

    algoritmo_nombre = "K-Means" if str(algoritmo_clustering).lower().strip() in ("kmeans", "k-means") else "GMM"
    if algoritmo_nombre == "K-Means":
        modelo_clustering = KMeans(n_clusters=5, random_state=42, n_init=10)
    else:
        modelo_clustering = GaussianMixture(n_components=5, covariance_type='full', random_state=42, max_iter=200)

    clusters = modelo_clustering.fit_predict(z_eval)

    conf = np.zeros((5, 5), dtype=np.int64)
    for t_l, c_l in zip(y_num, clusters):
        conf[t_l, c_l] += 1

    filas, cols = linear_sum_assignment(-conf)
    mapeo = {c: f for f, c in zip(filas, cols)}
    preds = np.array([mapeo[c] for c in clusters])

    acc_voc = {}
    for voc, idx_c in map_v.items():
        mask_c = (y_num == idx_c)
        tot = np.sum(mask_c)
        aciertos = np.sum(preds[mask_c] == idx_c) if tot > 0 else 0
        acc_voc[voc] = float(aciertos / tot * 100.0) if tot > 0 else 0.0

    cluster_acc = float(np.sum(preds == y_num) / len(y_num) * 100.0)
    sil = float(silhouette_score(z_eval, Y_eval))
    db = float(davies_bouldin_score(z_eval, Y_eval))

    _log("=" * 60)
    _log(f"RESULTADOS DIAGNOSTICOS {algoritmo_nombre.upper()} ({modalidad.upper()}, {latent_dim}D)")
    _log("=" * 60)
    _log(f"  Exactitud {algoritmo_nombre} Global: {cluster_acc:.2f}%")
    _log(f"  Indice de Silueta:        {sil:+.3f}")
    _log(f"  Indice Davies-Bouldin:    {db:.3f}")
    for v in ['A', 'E', 'I', 'O', 'U']:
        _log(f"    Vocal /{v.lower()}/: {acc_voc[v]:5.1f}%")
    _log("=" * 60)

    # Renderizado Gráfico en carpeta de procesamiento dedicada
    dir_salida = carpeta_salida if carpeta_salida is not None else figuras_dir
    os.makedirs(dir_salida, exist_ok=True)
    fig_path = os.path.join(dir_salida, f"informe_autoencoder_{modalidad}_{latent_dim}d.png")
    
    if latent_dim == 2:
        fig = plt.figure(figsize=(16, 9), dpi=180)
        gs = fig.add_gridspec(2, 5, height_ratios=[1.3, 0.8])

        ax_lat = fig.add_subplot(gs[0, :3])

        # Límites ajustados dinámicamente al rango real de los datos (sin padding excesivo fijo)
        x_span = max(float(z_eval[:, 0].max() - z_eval[:, 0].min()), 0.05)
        y_span = max(float(z_eval[:, 1].max() - z_eval[:, 1].min()), 0.05)
        pad_x = 0.15 * x_span
        pad_y = 0.12 * y_span

        x_min, x_max = float(z_eval[:, 0].min()) - pad_x, float(z_eval[:, 0].max()) + pad_x
        y_min, y_max = float(z_eval[:, 1].min()) - pad_y, float(z_eval[:, 1].max()) + pad_y

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
        grid = np.c_[xx.ravel(), yy.ravel()]

        z_grid = modelo_clustering.predict(grid)
        z_grid_mapped = np.vectorize(mapeo.get)(z_grid).reshape(xx.shape)

        palette_list = [COLORES_VOCALES['A'], COLORES_VOCALES['E'], COLORES_VOCALES['I'], COLORES_VOCALES['O'], COLORES_VOCALES['U']]
        cmap_bg = mcolors.ListedColormap(palette_list)

        ax_lat.pcolormesh(xx, yy, z_grid_mapped, cmap=cmap_bg, alpha=0.18, shading='auto', zorder=0)
        ax_lat.contour(xx, yy, z_grid_mapped, levels=[0.5, 1.5, 2.5, 3.5], colors='k', linewidths=0.6, alpha=0.55, zorder=1)

        for v, col in COLORES_VOCALES.items():
            mask_v = (Y_eval == v)
            if np.sum(mask_v) > 0:
                for f_idx, f in enumerate(fechas_unicas):
                    mask_vf = mask_v & (Fechas_eval == f)
                    if np.sum(mask_vf) > 0:
                        m_shape = marker_por_fecha.get(f, 'o')
                        lbl = f"/{v.lower()}/ ({acc_voc[v]:.1f}%)" if f_idx == 0 else None
                        ax_lat.scatter(
                            z_eval[mask_vf, 0], z_eval[mask_vf, 1],
                            c=col, marker=m_shape, label=lbl,
                            alpha=0.88, s=48, edgecolors='white', linewidth=0.6, zorder=4
                        )
                pts = z_eval[mask_v]
                if len(pts) > 4:
                    cov = np.cov(pts.T)
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    w, h = 2 * 1.5 * np.sqrt(np.maximum(vals, 1e-6))
                    ell = Ellipse(xy=np.mean(pts, axis=0), width=w, height=h, angle=angle,
                                  edgecolor=col, facecolor='none', lw=1.8, linestyle='--', zorder=5)
                    ax_lat.add_patch(ell)

        ax_lat.axhline(0, color='gray', linestyle=':', alpha=0.5, zorder=2)
        ax_lat.axvline(0, color='gray', linestyle=':', alpha=0.5, zorder=2)
        ax_lat.set_xlim(x_min, x_max)
        ax_lat.set_ylim(y_min, y_max)
        ax_lat.set_xlabel("Eje Latente Z1", fontsize=11, fontweight='bold')
        ax_lat.set_ylabel("Eje Latente Z2", fontsize=11, fontweight='bold')
        ax_lat.set_title(f"Espacio Latente 2D Canónico ({algoritmo_nombre})\nExactitud: {cluster_acc:.1f}% | Silueta: {sil:+.3f}", fontsize=12, fontweight='bold')
        ax_lat.legend(loc='upper right', fontsize=9)
        ax_lat.grid(True, alpha=0.3)

        if len(fechas_unicas) > 1:
            handles_fechas = [
                plt.Line2D([0], [0], marker=marker_por_fecha[f], color='w', markerfacecolor='gray', markersize=8, label=f)
                for f in fechas_unicas
            ]
            leg_fechas = ax_lat.legend(handles=handles_fechas, loc='lower left', fontsize=8, title="Fechas Registradas", framealpha=0.8)
            ax_lat.add_artist(leg_fechas)

        ax_bar = fig.add_subplot(gs[0, 3:])
        vocales = ['A', 'E', 'I', 'O', 'U']
        accs = [acc_voc[v] for v in vocales]
        bar_cols = [COLORES_VOCALES[v] for v in vocales]
        bars = ax_bar.bar(vocales, accs, color=bar_cols, alpha=0.85, edgecolor='black')
        ax_bar.axhline(cluster_acc, color='blue', linestyle='-.', alpha=0.7, label=f"Media: {cluster_acc:.1f}%")
        ax_bar.set_ylim(0, 105)
        ax_bar.set_ylabel("Exactitud Porcentual", fontsize=10, fontweight='bold')
        ax_bar.set_title(f"Balance Multiclase ({algoritmo_nombre})", fontsize=12, fontweight='bold')
        for b, a in zip(bars, accs):
            ax_bar.text(b.get_x() + b.get_width()/2.0, a + 2.0, f"{a:.1f}%", ha='center', fontsize=10, fontweight='bold')
        ax_bar.legend(loc='upper right')
        ax_bar.grid(True, alpha=0.3)

        # Fila de Reconstrucciones de Prueba con nombres de músculos
        cols_ch = ['#d62728', '#2ca02c', '#ff7f0e', '#1f77b4', '#9467bd']
        n_canales_eval = X_eval.shape[1]

        # Cálculo dinámico de escala vertical para no comprimir señales de baja amplitud promedio (ej. señal cruda)
        max_promedio_global = 0.0
        if modalidad != "espectrograma":
            for v in vocales:
                idx_v = np.where(Y_eval == v)[0]
                if len(idx_v) > 0:
                    mean_in_v = np.mean(X_eval[idx_v], axis=0)
                    mean_rec_v = np.mean(x_rec_eval[idx_v], axis=0)
                    max_promedio_global = max(max_promedio_global, float(np.max(mean_in_v)), float(np.max(mean_rec_v)))

        if modalidad == "cruda" or (max_promedio_global > 0 and max_promedio_global < 0.65):
            y_max_plot = max(0.05, float(max_promedio_global * 1.15))
            y_min_plot = -0.05 * y_max_plot
        else:
            y_max_plot = 1.05
            y_min_plot = -0.05

        for i, v in enumerate(vocales):
            ax_v = fig.add_subplot(gs[1, i])
            idx_v = np.where(Y_eval == v)[0]
            if len(idx_v) > 0:
                if modalidad == "espectrograma":
                    img_in = np.clip(np.mean(X_eval[idx_v], axis=0).transpose(1, 2, 0), 0, 1)
                    img_rec = np.clip(np.mean(x_rec_eval[idx_v], axis=0).transpose(1, 2, 0), 0, 1)
                    if img_in.shape[2] == 2:
                        img_in = np.dstack([img_in, np.zeros_like(img_in[:, :, :1])])
                        img_rec = np.dstack([img_rec, np.zeros_like(img_rec[:, :, :1])])
                    sep = np.ones((32, 2, 3), dtype=np.float32)
                    comp = np.concatenate([img_in, sep, img_rec], axis=1)
                    ax_v.imshow(comp, origin='lower', aspect='auto')
                    ax_v.set_xticks([16, 50])
                    ax_v.set_xticklabels(['Entrada', 'Reconstrucción'], fontsize=7)
                    ax_v.set_yticks([])
                else:
                    mean_in = np.mean(X_eval[idx_v], axis=0)
                    mean_rec = np.clip(np.mean(x_rec_eval[idx_v], axis=0), 0.0, 1.05)
                    t_axis = np.linspace(0, 1, mean_in.shape[1])
                    for c in range(n_canales_eval):
                        m_name = Musculos_Canales[c] if c < len(Musculos_Canales) else f"Ch{c}"
                        lbl_in = f"Ch{c}: {m_name} (Entr.)" if i == 0 else None
                        lbl_rec = f"Ch{c}: {m_name} (Rec.)" if i == 0 else None
                        ax_v.plot(t_axis, mean_in[c], color=cols_ch[c % len(cols_ch)], alpha=0.50, linestyle=':', label=lbl_in)
                        ax_v.plot(t_axis, mean_rec[c], color=cols_ch[c % len(cols_ch)], linewidth=1.8, label=lbl_rec)
                    ax_v.set_ylim(y_min_plot, y_max_plot)
                    ax_v.grid(True, alpha=0.2)
                    if i == 0:
                        ax_v.legend(loc='upper right', fontsize=6, framealpha=0.75)
                ax_v.set_title(f"Promedio /{v.lower()}/ (N={len(idx_v)})", fontsize=10, fontweight='bold')
                
        fechas_txt = ", ".join(fechas_unicas) if fechas_unicas else "N/A"
        musc_txt = " | ".join([f"Ch{c}: {Musculos_Canales[c]}" for c in range(len(Musculos_Canales))])
        plt.suptitle(
            f"Evaluación No Supervisada del Autoencoder ({modalidad.upper()}) | Fechas: {fechas_txt}\n"
            f"Músculos: {musc_txt}",
            fontsize=13, fontweight='bold', y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.94])

    else: # 3D
        fig = plt.figure(figsize=(16, 8), dpi=180)
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        for v, col in COLORES_VOCALES.items():
            mask_v = (Y_eval == v)
            if np.sum(mask_v) > 0:
                for f_idx, f in enumerate(fechas_unicas):
                    mask_vf = mask_v & (Fechas_eval == f)
                    if np.sum(mask_vf) > 0:
                        m_shape = marker_por_fecha.get(f, 'o')
                        lbl = f"/{v.lower()}/ ({acc_voc[v]:.1f}%)" if f_idx == 0 else None
                        ax_3d.scatter(
                            z_eval[mask_vf, 0], z_eval[mask_vf, 1], z_eval[mask_vf, 2],
                            c=col, marker=m_shape, label=lbl,
                            alpha=0.85, s=44, edgecolors='white', linewidth=0.5
                        )
        ax_3d.set_xlabel("Z1", fontsize=10, fontweight='bold')
        ax_3d.set_ylabel("Z2", fontsize=10, fontweight='bold')
        ax_3d.set_zlabel("Z3", fontsize=10, fontweight='bold')
        ax_3d.set_title(f"Espacio Latente 3D ({algoritmo_nombre})\nExactitud: {cluster_acc:.1f}% | Silueta: {sil:+.3f}", fontsize=12, fontweight='bold')
        ax_3d.legend(loc='upper right', fontsize=9)
        if len(fechas_unicas) > 1:
            handles_fechas = [
                plt.Line2D([0], [0], marker=marker_por_fecha[f], color='w', markerfacecolor='gray', markersize=8, label=f)
                for f in fechas_unicas
            ]
            leg_fechas = ax_3d.legend(handles=handles_fechas, loc='lower left', fontsize=8, title="Fechas Registradas", framealpha=0.8)
            ax_3d.add_artist(leg_fechas)
        ax_3d.view_init(elev=25, azim=45)

        ax_bar = fig.add_subplot(1, 2, 2)
        vocales = ['A', 'E', 'I', 'O', 'U']
        accs = [acc_voc[v] for v in vocales]
        bar_cols = [COLORES_VOCALES[v] for v in vocales]
        bars = ax_bar.bar(vocales, accs, color=bar_cols, alpha=0.85, edgecolor='black')
        ax_bar.axhline(cluster_acc, color='blue', linestyle='-.', alpha=0.7, label=f"Media: {cluster_acc:.1f}%")
        ax_bar.set_ylim(0, 105)
        ax_bar.set_ylabel("Exactitud Porcentual", fontsize=11, fontweight='bold')
        ax_bar.set_title(f"Balance Multiclase en 3D ({algoritmo_nombre})", fontsize=13, fontweight='bold')
        for b, a in zip(bars, accs):
            ax_bar.text(b.get_x() + b.get_width()/2.0, a + 2.0, f"{a:.1f}%", ha='center', fontsize=11, fontweight='bold')
        ax_bar.legend(loc='upper right')
        ax_bar.grid(True, alpha=0.3)

        fechas_txt = ", ".join(fechas_unicas) if fechas_unicas else "N/A"
        musc_txt = " | ".join([f"Ch{c}: {Musculos_Canales[c]}" for c in range(len(Musculos_Canales))])
        plt.suptitle(
            f"Evaluación No Supervisada en Tres Dimensiones ({modalidad.upper()}) | Fechas: {fechas_txt}\n"
            f"Músculos: {musc_txt}",
            fontsize=13, fontweight='bold', y=0.98
        )
        plt.tight_layout()

    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    try:
        alt_fig = os.path.join(resultados_dir, f"informe_autoencoder_{modalidad}_{latent_dim}d.png")
        plt.savefig(alt_fig, dpi=180, bbox_inches='tight')
    except Exception:
        pass
    plt.close()
    _log(f"Informe visual renderizado y guardado en: {fig_path}")

    if mostrar_grafico:
        _log(f"Mostrando informe gráfico en el visor del sistema...")
        abrir_imagen_en_visor(fig_path)

    metricas_dict = {
        'cluster_acc': cluster_acc,
        'gmm_acc': cluster_acc,
        'algoritmo_clustering': algoritmo_nombre,
        'acc_voc': acc_voc,
        'silhouette': sil,
        'davies_bouldin': db,
        'fig_path': fig_path,
        'modalidad': modalidad,
        'latent_dim': latent_dim,
        'fechas': [str(f) for f in fechas_unicas],
        'musculos': [str(m) for m in Musculos_Canales]
    }
    json_path = os.path.join(dir_salida, "metricas.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f_m:
            json.dump(metricas_dict, f_m, indent=2)
        _log(f"Métricas y diagnósticos exportados en: {json_path}")
    except Exception:
        pass

    return {
        'cluster_acc': cluster_acc,
        'gmm_acc': cluster_acc,
        'algoritmo_clustering': algoritmo_nombre,
        'acc_voc': acc_voc,
        'silhouette': sil,
        'davies_bouldin': db,
        'fig_path': fig_path,
        'dir_salida': dir_salida
    }
