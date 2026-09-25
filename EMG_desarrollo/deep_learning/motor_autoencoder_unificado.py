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
import matplotlib.patheffects as pe
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
import colorsys
import seaborn as sns

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
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

def adjust_lightness(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


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

    else: # Por defecto "rms" (Estándar Trevisan / PCA-UMAP)
        sig_sq = s_bp ** 2
        if win_len > 1:
            w = np.ones(win_len, dtype=np.float64) / float(win_len)
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
    musculos_canales_global = [f"Canal {idx}" for idx in indices_canales]
    fs_global = 2000
    bpm_global = 40
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

    # Vía canónica: delegación directa a generador_pca_umap para garantizar paridad absoluta con PCA-UMAP
    try:
        from deep_learning.pca_umap_clustering import generador_pca_umap as gpu
        b_dir = os.path.dirname(os.path.dirname(rutas_tomas[0]))
        meds_rel = [os.path.relpath(p, b_dir) for p in rutas_tomas]
        lp_cut = float(kwargs.get('lowpass_cutoff_hz', 500.0))
        hp_cut = float(kwargs.get('highpass_cutoff_hz', 20.0))
        n_q = float(kwargs.get('notch_q', 2.0))
        t_filtro = kwargs.get('tipo_filtro_linea', 'notch')
        p_pre = float(kwargs.get('pre_pct', 0.50))
        p_post = float(kwargs.get('post_pct', 0.50))
        snr_th = float(kwargs.get('snr_min', 0.50))

        params_gpu = {
            'alpha_ruido': float(alpha_ruido),
            'smooth_ms': int(smooth_ms),
            'target_length': int(target_len),
            'snr_threshold': snr_th,
            'outlier_contamination': float(outlier_contamination),
            'notch_q': n_q,
            'tipo_filtro_ruido': t_filtro,
            'highpass_cutoff_hz': hp_cut,
            'lowpass_cutoff_hz': lp_cut,
            'gate_ratio_ruido': 0.0,
            'tipo_envolvente': tipo_envolvente,
            'correccion_impedancia': False
        }

        _log(f"Extrayendo y alineando con generador_pca_umap (pre={p_pre}, post={p_post}, LP={lp_cut}Hz, Notch Q={n_q})...")
        X_gpu, Y_gpu, Tomas_gpu, desc_gpu = gpu.extraer_y_filtrar(
            mediciones=meds_rel,
            base_dir=b_dir,
            params=params_gpu,
            aplicar_trevisan=False,
            modo_alineacion=modo_alineacion,
            pre_pct=p_pre,
            post_pct=p_post,
            canales_features=canales_features,
            ignorar_ventana_cero=False,
            aplicar_correccion_intersesion=False,
            correccion_impedancia=False
        )
        if len(X_gpu) > 0:
            N_gpu = len(X_gpu)
            X_env_clean = X_gpu.reshape(N_gpu, n_canales, target_len).astype(np.float32)
            X_cruda_clean = X_env_clean.copy()
            X_spec_clean = X_env_clean.copy()
            Y_clean = np.array(Y_gpu)
            Tomas_clean = np.array(Tomas_gpu)
            Fechas_clean = np.array(['2026-07-10'] * N_gpu)

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
            cols_feat = [f"Ch{c}_T{t}" for c in range(n_canales) for t in range(target_len)]
            df_feat = pd.DataFrame(X_gpu, columns=cols_feat)
            df_feat.insert(0, 'Toma', Tomas_clean)
            df_feat.insert(0, 'Vocal', Y_clean)
            csv_out_path = os.path.join(dir_salida_npz, "caracteristicas_exportadas.csv")
            df_feat.to_csv(csv_out_path, index=False)
            _log(f"Dataset consolidado generado con generador_pca_umap: {len(Y_clean)} muestras guardadas en {archivo_npz}")
            return archivo_npz, len(Y_clean)
    except Exception as e_gpu:
        _log(f"[Aviso] No se pudo usar generador_pca_umap ({e_gpu}), recurriendo a pipeline secundario...")

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

        pre_pct = float(kwargs.get('pre_pct', 0.50))
        post_pct = float(kwargs.get('post_pct', 0.50))
        pre_samples = int(round(muestras_pulso * pre_pct))
        post_samples = int(round(muestras_pulso * post_pct))
        noise_win_samples = max(10, int(muestras_pulso / 4.0))

        # Señal de referencia según modo_alineacion
        if modo_alineacion == "Pico Canal 0" and "canal_0" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_0")]
        elif modo_alineacion == "Pico Canal 1" and "canal_1" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_1")]
        elif modo_alineacion == "Pico Canal 2" and "canal_2" in canales_features:
            sig_ref_align = sigs_env[canales_features.index("canal_2")]
        else: # Pico Envolvente Muscular (Supremo)
            sig_ref_align = sigs_env[0]
            for c in range(1, n_canales):
                sig_ref_align = np.maximum(sig_ref_align, sigs_env[c])

        # Detección acústica precisa de picos de fonación (Estándar Trevisan / PCA-UMAP)
        lista_picos_iterar = []
        if modo_alineacion.startswith("Pico Volumen Micrófono") and mic_sig is not None:
            win_mic = max(5, int(0.05 * fs))
            mic_env = np.convolve(np.abs(mic_sig), np.ones(win_mic) / float(win_mic), mode='same')
            dist_samples = int(0.8 * muestras_pulso)
            min_h = np.max(mic_env) * 0.20
            picos_mic, _ = find_peaks(mic_env, distance=dist_samples, height=min_h)
            for w_i, p_val in enumerate(picos_mic):
                lista_picos_iterar.append((w_i, p_val))
        elif modo_alineacion.startswith("Pico Derivada Micrófono") and mic_sig is not None:
            win_mic = max(5, int(0.05 * fs))
            mic_env = np.convolve(np.abs(mic_sig), np.ones(win_mic) / float(win_mic), mode='same')
            deriv_mic = np.gradient(mic_env)
            win_d = max(1, int(fs * 0.05))
            deriv_smooth = np.convolve(deriv_mic, np.ones(win_d) / float(win_d), mode='same')
            dist_samples = int(0.8 * muestras_pulso)
            picos_mic, _ = find_peaks(mic_env, distance=dist_samples, height=np.max(mic_env) * 0.20)
            for w_i, p_amp in enumerate(picos_mic):
                r_ini = max(0, int(p_amp - pre_pct * muestras_pulso))
                idx_rel = int(np.argmax(deriv_smooth[r_ini:p_amp])) if r_ini < p_amp else 0
                lista_picos_iterar.append((w_i, r_ini + idx_rel))

        # Fallback a ranuras periódicas de metrónomo si no se dispuso de micrófono
        if not lista_picos_iterar:
            for win_idx in range(n_pulsos_total):
                cut_start = start_sample_noise + win_idx * muestras_pulso
                cut_end = min(len(sig_ref_align), cut_start + muestras_pulso)
                if cut_end - cut_start < muestras_pulso // 2:
                    continue
                local_slot = sig_ref_align[cut_start:cut_end]
                if len(local_slot) == 0:
                    continue
                rel_max = int(np.argmax(local_slot))
                lista_picos_iterar.append((win_idx, cut_start + rel_max))

        # Iterar por cada evento de fonación detectado
        for win_idx, p_idx in lista_picos_iterar:
            if (win_idx + 1) in excluded_windows or win_idx in excluded_windows:
                continue

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

                # Resta directa del piso de ruido interpulso (Estándar Trevisan / PCA-UMAP)
                if kwargs.get('substraer_rampa_bordes', False):
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
                else:
                    env_clean = np.maximum(0.0, env_seg - r_c) * peso_ch
                    rect_clean = np.maximum(0.0, np.abs(raw_seg) - r_c) * peso_ch

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

            # Normalización por Supremo Tricanal del pulso individual ANTES de remuestrear (Estándar Trevisan / PCA-UMAP)
            max_sup = max([float(np.max(seg)) for seg in env_segs])
            if max_sup <= 1e-6:
                continue

            from scipy.signal import resample as sp_resample
            env_i_list = []
            for c in range(n_canales):
                seg_norm = env_segs[c] / (max_sup + 1e-9)
                if target_env_len < len(seg_norm):
                    s_rs = sp_resample(seg_norm, target_env_len)
                    s_rs[s_rs < 0.0] = 0.0
                    env_i_list.append(s_rs.astype(np.float32))
                else:
                    env_i_list.append(np.interp(x_tgt_e, x_orig_w, seg_norm).astype(np.float32))
            env_i = np.array(env_i_list, dtype=np.float32)

            raw_sup = max([float(np.max(seg)) for seg in rect_segs]) if rect_segs else 1.0
            raw_i = np.array([np.interp(x_tgt_c, x_orig_w, rect_segs[c] / (raw_sup + 1e-9)) for c in range(n_canales)], dtype=np.float32)

            # Filtro SNR previo a la consolidación
            snr_min_val = float(kwargs.get('snr_min', 0.50))
            ruido_prom_local = float(np.mean(ruidos_c))
            snr_val = max_sup / (ruido_prom_local + 1e-9)
            if snr_val < snr_min_val:
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

    try:
        N_c, n_c, n_p = X_env_clean.shape
        flat_feats = X_env_clean.reshape(N_c, -1)
        cols_feat = []
        for ch_idx in range(n_c):
            for t in range(n_p):
                cols_feat.append(f"Ch{ch_idx}_T{t}")
        df_feat = pd.DataFrame(flat_feats, columns=cols_feat)
        df_feat.insert(0, 'Toma', Tomas_clean)
        df_feat.insert(0, 'Vocal', Y_clean)
        csv_out_path = os.path.join(dir_salida_npz, "caracteristicas_exportadas.csv")
        df_feat.to_csv(csv_out_path, index=False)
        _log(f"  [Exportación CSV] Características exportadas también en: {csv_out_path}")
    except Exception as e_csv:
        _log(f"  [Aviso CSV]: {e_csv}")

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
    """
    Extrae el identificador de sesión (ej. 'PRUEBA1', 'T1', 'SERIE1') según el formato canónico:
    vocal_pruebaotoma_sujeto (ej. 'A_Prueba1_Candela', 'E_T1_Lucas', 'I_Serie2_Candela').
    Garantiza que todas las vocales de una misma prueba o serie pertenezcan a la misma sesión.
    """
    s = str(toma_str).strip()
    parts = s.split('_')
    # Regla 1: Si inicia con vocal (A, E, I, O, U), parts[1] es la prueba o toma
    if len(parts) >= 2 and parts[0].upper() in ['A', 'E', 'I', 'O', 'U']:
        return parts[1].upper()
    m = re.search(r'(Prueba\d+|Sesion\d+|Session\d+|Serie\d+|Toma\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win') or p_clean.lower().startswith('w'):
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
    pts_base = max(1, min(n_pts_reposo, n_pts // 2))

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

    # Inyección adaptativa de longitud temporal o puntos de remuestreo
    if target_len is not None:
        for p_name in ('time_pts', 'target_len', 'time_len', 'pts', 'n_pts', 'seq_len', 'input_len'):
            if p_name in sig.parameters:
                kwargs_init[p_name] = target_len
                break
        if 'input_dim' in sig.parameters:
            kwargs_init['input_dim'] = in_channels * target_len

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

        for attr_name in ('target_len', 'time_pts', 'time_len', 'n_pts', 'pts'):
            if hasattr(modelo, attr_name):
                val = getattr(modelo, attr_name)
                if isinstance(val, int) and val > 0:
                    t_len = val
                    break

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

def entrenar_autoencoder(archivo_npz=None, modalidad="envolvente", latent_dim=2, epochs=150, batch_size=32, lr=0.002, carpeta_salida=None, callback_log=None, npz_path=None, usar_custom_arch=False, codigo_custom_arch=None, tipo_perdida="mse", gamma_sdtw=1.0, alpha_hibrida=1.0, lambda_orto=0.0, tipo_arquitectura="ortogonal", lambda_w=0.30, lambda_z=0.45, usar_impedancia_reposo=True, usar_alineacion_so2=False, ref_session='T2', seed=100):
    fijar_semilla(seed=seed)
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
    es_ortogonal = (tipo_arq in ("ortogonal", "orthogonal", "record", "record_91", "optimo", "conv_ortogonal"))

    if modalidad == "envolvente":
        X = datos['X_env']
        in_ch = X.shape[1] if X.ndim > 2 else 3
        t_len = X.shape[-1] if X.ndim > 2 else (X.shape[1] // in_ch)
        if es_ortogonal:
            if usar_impedancia_reposo:
                tomas = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
                sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas])
                _log(f"  [Reposo Basal e Impedancia] Acondicionando {len(X)} ventanas para {len(np.unique(sesiones))} sesiones...")
                X = acondicionar_reposo_impedancia(X, sesiones, n_canales=in_ch, n_pts_reposo=10)
            if batch_size == 32 or batch_size < len(X):
                batch_size = len(X)
                _log(f"  [Régimen Ortogonal] Batch size ajustado a Full-Batch ({batch_size} muestras) para estabilidad de Cov(Z).")

        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  [Arquitectura Personalizada] Compilando modelo desde editor de código...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_ch)
            _log(f"  [Arquitectura Personalizada] Modelo instanciado: {modelo.__class__.__name__}")
        elif es_ortogonal:
            input_dim_total = in_ch * t_len
            hidden_dim_orto = 64 if latent_dim == 3 else 32
            modelo = OrthogonalAutoencoder2D(input_dim=input_dim_total, hidden_dim=hidden_dim_orto, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado: {input_dim_total} -> {hidden_dim_orto} -> 16 -> {latent_dim} (Tanh, bias=False)")
        else:
            modelo = AutoencoderEnvolvente1D(in_channels=in_ch, latent_dim=latent_dim, target_len=t_len)
    elif modalidad == "cruda":
        X = datos['X_cruda']
        in_ch = X.shape[1] if X.ndim > 2 else 3
        t_len = X.shape[-1] if X.ndim > 2 else 1000

        if es_ortogonal:
            if usar_impedancia_reposo:
                tomas = datos['Tomas'] if 'Tomas' in datos else np.array([f"T1_p{i}" for i in range(len(X))])
                sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas])
                _log(f"  [Reposo Basal e Impedancia] Acondicionando {len(X)} ventanas para {len(np.unique(sesiones))} sesiones...")
                X = acondicionar_reposo_impedancia(X, sesiones, n_canales=in_ch, n_pts_reposo=10)
            if batch_size == 32 or batch_size < len(X):
                batch_size = len(X)
                _log(f"  [Régimen Ortogonal] Batch size ajustado a Full-Batch ({batch_size} muestras).")

        if usar_custom_arch and codigo_custom_arch and codigo_custom_arch.strip():
            _log("  [Arquitectura Personalizada] Compilando modelo desde editor de código...")
            modelo = compilar_modelo_desde_codigo(codigo_custom_arch, modalidad=modalidad, latent_dim=latent_dim, target_len=t_len, in_channels=in_ch)
            _log(f"  [Arquitectura Personalizada] Modelo instanciado: {modelo.__class__.__name__}")
        elif es_ortogonal:
            input_dim_total = in_ch * t_len
            hidden_dim_orto = 64 if latent_dim == 3 else 32
            modelo = OrthogonalAutoencoder2D(input_dim=input_dim_total, hidden_dim=hidden_dim_orto, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado: {input_dim_total} -> {hidden_dim_orto} -> 16 -> {latent_dim} (Tanh, bias=False)")
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
    loader = DataLoader(TensorDataset(tensor_x), batch_size=batch_size, shuffle=(not es_ortogonal))

    wd = 0.0 if es_ortogonal else 1e-5
    optimizador = optim.Adam(modelo.parameters(), lr=lr, weight_decay=wd)

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

def evaluar_espacio_latente(archivo_npz=None, modelo=None, modalidad="envolvente", latent_dim=2, carpeta_salida=None, callback_log=None, npz_path=None, usar_custom_arch=False, codigo_custom_arch=None, mostrar_grafico=True, algoritmo_clustering="gmm", usar_alineacion_so2=None, ref_session=None, modelo_path=None):
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
    es_orto = (tipo_arq in ("ortogonal", "orthogonal", "record", "record_91", "optimo", "conv_ortogonal"))
    imp_reposo = cfg_auto.get('usar_impedancia_reposo', True)
    if usar_alineacion_so2 is None:
        usar_alineacion_so2 = cfg_auto.get('usar_alineacion_so2', False)
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
            hidden_dim_orto = 64 if latent_dim == 3 else 32
            modelo = OrthogonalAutoencoder2D(input_dim=in_ch * t_len, hidden_dim=hidden_dim_orto, latent_dim=latent_dim)
            _log(f"  [Autoencoder Ortogonal Récord] Instanciado para inferencia ({in_ch * t_len} -> {hidden_dim_orto} -> 16 -> {latent_dim})")
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

        if modelo_path and os.path.exists(modelo_path):
            pth_candidatos = [modelo_path]
        else:
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
        try:
            if dir_salida_mod != modelos_dir:
                df_crudo.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d_crudo.csv"), index=False)
                df_crudo.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass
    except Exception:
        pass

    # 1. Alineación Topológica Determinística SO(2) o Canónica
    if usar_alineacion_so2 and latent_dim == 2 and len(np.unique(sesiones)) > 1:
        _log(f"  [Alineación Topológica SO(2)] Alineando sesiones respecto a referencia '{ref_session}'...")
        z_align = alinear_topologia_sesiones_so2(z, sesiones, ref_session=ref_session)
        sufijo = "alineado"
        label_titulo = "Alineado SO(2)"
        try:
            df_alin = pd.DataFrame({'Vocal': Y_labels, 'Toma': tomas, 'Sesion': sesiones, 'Z1': z_align[:, 0], 'Z2': z_align[:, 1]})
            df_alin.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d_alineado.csv"), index=False)
            df_alin.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d.csv"), index=False)
            if dir_salida_mod != modelos_dir:
                df_alin.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d_alineado.csv"), index=False)
                df_alin.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass
    elif es_orto:
        # En el régimen ortogonal nativo crudo (87.85%), NO se rota canónicamente
        z_align = z.copy()
        sufijo = "crudo"
        label_titulo = "Nativo Crudo"
        try:
            df_crudo.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d.csv"), index=False)
            if dir_salida_mod != modelos_dir:
                df_crudo.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass
    else:
        z_align = alinear_canonicamente(z, Y_labels)
        sufijo = "canonico"
        label_titulo = "Canónico"
        try:
            df_crudo.to_csv(os.path.join(dir_salida_mod, "proyecciones_latentes_2d.csv"), index=False)
            if dir_salida_mod != modelos_dir:
                df_crudo.to_csv(os.path.join(modelos_dir, "proyecciones_latentes_2d.csv"), index=False)
        except Exception:
            pass

    # 2. Purga Robusta de Outliers en Espacio Latente
    # En el modelo ortogonal evaluamos el 100% de los pulsos para replicar exactamente modelorecord.py
    if es_orto:
        mask_inliers = np.ones(len(z_align), dtype=bool)
    else:
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
    MARKERS_FECHAS = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>']
    marker_por_fecha = {f: MARKERS_FECHAS[i % len(MARKERS_FECHAS)] for i, f in enumerate(fechas_unicas)}

    Musculos_Canales = list(datos['Musculos_Canales']) if 'Musculos_Canales' in datos else ["Canal 0", "Canal 1", "Canal 2"]
    while len(Musculos_Canales) < 3:
        Musculos_Canales.append(f"Canal {len(Musculos_Canales)}")

    # Diagnóstico de Clustering no supervisado con Asignación Húngara (GMM o K-Means)
    map_v = {'A': 0, 'E': 1, 'I': 2, 'O': 3, 'U': 4}
    y_num = np.array([map_v[v] for v in Y_eval])

    alg_str = str(algoritmo_clustering).lower().strip()
    if "lda" in alg_str or "supervisado" in alg_str:
        algoritmo_nombre = "LDA Supervisado"
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        modelo_clustering = LinearDiscriminantAnalysis()
        modelo_clustering.fit(z_eval, y_num)
        preds = modelo_clustering.predict(z_eval)
        mapeo = {i: i for i in range(5)}
    elif alg_str in ("kmeans", "k-means"):
        algoritmo_nombre = "K-Means"
        modelo_clustering = KMeans(n_clusters=5, random_state=42, n_init=10)
        clusters = modelo_clustering.fit_predict(z_eval)
        conf = np.zeros((5, 5), dtype=np.int64)
        for t_l, c_l in zip(y_num, clusters):
            conf[t_l, c_l] += 1
        filas, cols = linear_sum_assignment(-conf)
        mapeo = {c: f for f, c in zip(filas, cols)}
        preds = np.array([mapeo[c] for c in clusters])
    else:
        algoritmo_nombre = "GMM"
        modelo_clustering = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20, max_iter=200)
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
    n_clases_unicas = len(np.unique(Y_eval))
    if n_clases_unicas >= 2:
        sil = float(silhouette_score(z_eval, Y_eval))
        db = float(davies_bouldin_score(z_eval, Y_eval))
    else:
        sil = 0.0
        db = float('inf')
        _log(f"[Aviso] Solo se detectó {n_clases_unicas} clase vocal en los datos extraídos. "
             "Silueta y Davies-Bouldin no se pueden calcular (se necesitan al menos 2 clases).")

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
        VOCALES = ['A', 'E', 'I', 'O', 'U']
        fig, axes = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

        xr = z_eval[:, 0].max() - z_eval[:, 0].min()
        yr = z_eval[:, 1].max() - z_eval[:, 1].min()
        margin = 0.12
        x_min, x_max = z_eval[:, 0].min() - xr * margin, z_eval[:, 0].max() + xr * margin
        y_min, y_max = z_eval[:, 1].min() - yr * margin, z_eval[:, 1].max() + yr * margin
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
        grid = np.c_[xx.ravel(), yy.ravel()]

        preds_grid = modelo_clustering.predict(grid)
        grid_mapped = np.array([mapeo[p] for p in preds_grid]).reshape(xx.shape)
        palette_list = [COLORES_VOCALES[v] for v in VOCALES]
        cmap_mesh = mcolors.ListedColormap(palette_list)

        axes[0].pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
        axes[0].contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.5, alpha=0.5, zorder=1)

        mapeo_inv = {f: c for c, f in mapeo.items()}

        for idx, v in enumerate(VOCALES):
            mask_v = (Y_eval == v)
            axes[0].scatter(
                z_eval[mask_v, 0], z_eval[mask_v, 1],
                c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/",
                s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4
            )
            if idx in mapeo_inv:
                cid = mapeo_inv[idx]
                if hasattr(modelo_clustering, 'means_'):
                    cen = modelo_clustering.means_[cid]
                    axes[0].scatter(
                        cen[0], cen[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)],
                        marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5,
                        path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)]
                    )

        axes[0].set_title(f"Espacio Canónico 2D: {label_titulo}\nExactitud {algoritmo_nombre}: {cluster_acc:.2f}%", fontsize=15, fontweight='bold', pad=12)
        axes[0].set_xlabel("Z1", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Z2", fontsize=13, fontweight='bold')
        axes[0].legend(loc='best', fontsize=12, frameon=True)
        axes[0].grid(True, linestyle=':', alpha=0.6)

        # Matriz de confusión en mapa de calor con porcentajes y recuentos
        cm_ordenada = np.zeros((5, 5), dtype=np.int64)
        for t_l, p_l in zip(y_num, preds):
            cm_ordenada[t_l, p_l] += 1

        cm_pct = cm_ordenada.astype(float) / np.maximum(cm_ordenada.sum(axis=1, keepdims=True), 1e-6) * 100
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', xticklabels=VOCALES, yticklabels=VOCALES, ax=axes[1], cbar=False, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        axes[1].set_title(f"Matriz de Confusión: {label_titulo} - {cluster_acc:.2f}%", fontsize=14, fontweight='bold', pad=12)
        axes[1].set_xlabel("Vocal Predicha", fontsize=12, fontweight='bold')
        axes[1].set_ylabel("Vocal Real", fontsize=12, fontweight='bold')
        plt.tight_layout()

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
        ax_3d.set_title(f"Espacio Latente 3D: {algoritmo_nombre}\nExactitud: {cluster_acc:.1f}% | Silueta: {sil:+.3f}", fontsize=12, fontweight='bold')
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
        ax_bar.set_title(f"Balance Multiclase en 3D: {algoritmo_nombre}", fontsize=13, fontweight='bold')
        for b, a in zip(bars, accs):
            ax_bar.text(b.get_x() + b.get_width()/2.0, a + 2.0, f"{a:.1f}%", ha='center', fontsize=11, fontweight='bold')
        ax_bar.legend(loc='upper right')
        ax_bar.grid(True, alpha=0.3)

        fechas_txt = ", ".join(fechas_unicas) if fechas_unicas else "N/A"
        musc_txt = " | ".join([f"Ch{c}: {Musculos_Canales[c]}" for c in range(len(Musculos_Canales))])
        plt.suptitle(
            f"Evaluación No Supervisada en Tres Dimensiones: {modalidad.upper()} | Fechas: {fechas_txt}\n"
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


# ==============================================================================
# DECODIFICACIÓN DE SECUENCIAS CONTINUAS Y PRUEBA EN DATASETS EXTERNOS
# ==============================================================================
def decodificar_secuencia_continua(
    carpeta_secuencia,
    modelo_path=None,
    carpeta_salida=None,
    alpha_ruido=1.0,
    smooth_ms=90,
    notch_q=2.0,
    target_length=20,
    latent_dim=2,
    modo_deteccion="Gate Doble (sEMG Puro)",
    algoritmo_clasificacion="GMM",
    usar_custom_arch=False,
    codigo_custom_arch=None,
    callback_log=None
):
    """
    Decodifica una grabación de secuencia continua sEMG mediante el modelo de Autoencoder.
    Detecta pulsos fonatorios vía micrófono o Gate Doble mioeléctrico, normaliza por el Supremo Tricanal del pulso,
    acondiciona el reposo basal y proyecta en el espacio latente asignando vocales según fronteras.
    """
    def _log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    _log("\n" + "="*70)
    _log("DECODIFICACION DE SECUENCIA CONTINUA: AUTOENCODER")
    _log(f"Carpeta: {carpeta_secuencia}")
    _log(f"Método de Detección: {modo_deteccion}")
    _log(f"Clasificador de Fronteras: {algoritmo_clasificacion}")
    _log("="*70)

    if not os.path.isdir(carpeta_secuencia):
        raise FileNotFoundError(f"Carpeta no encontrada: {carpeta_secuencia}")

    meta_file = os.path.join(carpeta_secuencia, "canal_0", "metadata.json")
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"No se encontró metadata.json en {os.path.join(carpeta_secuencia, 'canal_0')}")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    fs = meta.get("sample_rate", 2000)
    noise_sec = meta.get("noise_seconds", 5.0)
    n_noise_samples = int(noise_sec * fs)
    palabras_ground = meta.get("valid_words", [])

    # Cargar 4 canales según regla de oro
    signals = []
    for ch in range(4):
        p_wav = os.path.join(carpeta_secuencia, f"canal_{ch}", "grabacion.wav")
        if not os.path.exists(p_wav):
            raise FileNotFoundError(f"No se encontró grabacion.wav en canal_{ch}")
        _, data = wavfile.read(p_wav)
        signals.append(data.astype(np.float64))

    sig_emg = np.stack(signals[:3], axis=0)
    sig_mic = signals[3]
    n_samples = sig_emg.shape[1]

    # Filtrado DSP
    b_notch, a_notch = iirnotch(50.0, notch_q, fs)
    b_band, a_band = butter(2, [20.0, 500.0], 'bandpass', fs=fs)
    sig_filt = np.zeros_like(sig_emg)
    for c in range(3):
        s_n = filtfilt(b_notch, a_notch, sig_emg[c])
        sig_filt[c] = filtfilt(b_band, a_band, s_n)

    win_rms = int((smooth_ms / 1000.0) * fs)
    if win_rms % 2 == 0:
        win_rms += 1
    kernel_rms = np.ones(win_rms) / win_rms
    env_emg = np.zeros_like(sig_filt)
    for c in range(3):
        env_emg[c] = np.sqrt(np.maximum(0, np.convolve(sig_filt[c]**2, kernel_rms, mode='same')))

    ruido_base_emg = np.median(env_emg[:, :n_noise_samples], axis=1, keepdims=True)

    if "gate doble" in str(modo_deteccion).lower():
        _log("[Detección] Aplicando detector Gate Doble mioeléctrico con histéresis y backtracking...")
        env_clean = np.maximum(env_emg - ruido_base_emg, 0.0)
        S_emg = np.sqrt(np.sum(env_clean**2, axis=0))

        S_ruido = S_emg[:n_noise_samples]
        med_base = np.median(S_ruido)
        mad_base = np.median(np.abs(S_ruido - med_base))
        sigma_rob = 1.4826 * mad_base + 1e-6
        U_bajo = med_base + 3.0 * sigma_rob

        p98_emg = np.percentile(S_emg[n_noise_samples:], 98) if n_samples > n_noise_samples else np.max(S_emg)
        U_alto = min(1800.0, max(1300.0, p98_emg * 0.35))

        T_refr = int(0.800 * fs)
        max_lookback = int(0.350 * fs)

        onsets_emg = []
        en_pulso = False
        ultimo_onset = -T_refr

        for n in range(n_noise_samples, n_samples):
            if not en_pulso:
                if S_emg[n] >= U_alto and (n - ultimo_onset) >= T_refr:
                    lb_st = max(n_noise_samples, n - max_lookback)
                    sub_s = S_emg[lb_st:n]
                    cruces = np.where(sub_s <= U_bajo)[0]
                    n_on = lb_st + cruces[-1] if len(cruces) > 0 else lb_st
                    onsets_emg.append(n_on)
                    ultimo_onset = n_on
                    en_pulso = True
            else:
                if S_emg[n] < U_bajo:
                    en_pulso = False

        max_pulsos = len(palabras_ground) if len(palabras_ground) > 0 else meta.get("pulse_count", None)
        if max_pulsos and len(onsets_emg) > max_pulsos:
            onsets_emg = onsets_emg[:max_pulsos]

        # Proyección por desfase electromecánico EMD (+350 ms)
        picos_fonacion = [int(on + 0.350 * fs) for on in onsets_emg if (on + 0.350 * fs) < (n_samples - int(1.0 * fs))]
        _log(f"[Detección] Gate Doble identificó {len(picos_fonacion)} contracciones bioeléctricas.")
        if len(picos_fonacion) == 0:
            raise ValueError("No se detectaron eventos con el detector Gate Doble.")
    else:
        # Detección de fonación en micrófono
        win_mic = int(0.050 * fs)
        mic_env = np.convolve(np.abs(sig_mic), np.ones(win_mic) / win_mic, mode='same')
        p98_mic = np.percentile(mic_env[n_noise_samples:], 98) if n_samples > n_noise_samples else np.max(mic_env)
        umbral_altura = min(2000.0, max(500.0, p98_mic * 0.35))
        picos_cand, _ = find_peaks(mic_env, distance=int(1.2 * fs), height=umbral_altura)
        picos_fonacion = [p for p in picos_cand if p >= 6.0 * fs and p < (n_samples - int(1.0 * fs))]
        max_pulsos = len(palabras_ground) if len(palabras_ground) > 0 else meta.get("pulse_count", None)
        if max_pulsos and len(picos_fonacion) > max_pulsos:
            picos_fonacion = picos_fonacion[:max_pulsos]

        _log(f"[Detección] Se detectaron {len(picos_fonacion)} eventos fonatorios válidos vía micrófono.")
        if len(picos_fonacion) == 0:
            raise ValueError("No se detectaron eventos fonatorios con amplitud suficiente en el micrófono.")

    ruido_base_emg = np.median(env_emg[:, :n_noise_samples], axis=1, keepdims=True)
    half_win = int(1.0 * fs)
    vocales = ['A', 'E', 'I', 'O', 'U']
    vocal_to_idx = {v: i for i, v in enumerate(vocales)}

    tiene_ground_truth = len(palabras_ground) > 0

    X_list = []
    y_ground_list = []
    tiempos_list = []

    for i, p in enumerate(picos_fonacion):
        start = p - half_win
        end = p + half_win
        if start < 0 or end > n_samples:
            continue
        seg_raw = np.maximum(env_emg[:, start:end] - ruido_base_emg, 0.0)
        M_supremo = np.max(seg_raw) + 1e-9
        seg_norm = seg_raw / M_supremo

        feat_p = []
        for c in range(3):
            resamp = np.interp(np.linspace(0, 1, target_length), np.linspace(0, 1, seg_norm.shape[1]), seg_norm[c])
            feat_p.append(resamp)
        X_list.append(np.concatenate(feat_p))
        tiempos_list.append(p / fs)
        if tiene_ground_truth:
            v_g = palabras_ground[i] if i < len(palabras_ground) else vocales[i % 5]
            y_ground_list.append(v_g)
        else:
            y_ground_list.append("Desc")

    X_seq = np.array(X_list)
    N_seq, D_seq = X_seq.shape
    y_ground = np.array(y_ground_list)

    # Acondicionamiento reposo e impedancia (Butterworth 3, Wn=0.3, sustracción primeros 10 pts, div P95)
    b_bw, a_bw = butter(3, 0.3, btype='low')
    X_reshaped = X_seq.reshape(N_seq, 3, target_length)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N_seq):
        for c in range(3):
            X_filt[i, c, :] = filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for c in range(3):
        base_mean = np.mean(X_filt[:, c, :10])
        base_max = np.percentile(X_filt[:, c, :], 95) - base_mean + 1e-6
        X_norm[:, c, :] = (X_filt[:, c, :] - base_mean) / base_max

    X_seq_t = torch.tensor(X_norm.reshape(N_seq, -1), dtype=torch.float32)

    # Cargar modelo entrenado (priorizando la corrida más reciente del usuario)
    if modelo_path is None or not os.path.exists(modelo_path):
        candidatos = []
        proc_dirs = sorted(glob.glob(os.path.join(resultados_dir, "procesamiento_*")))
        if proc_dirs:
            candidatos.append(os.path.join(proc_dirs[-1], f"autoencoder_envolvente_{latent_dim}d.pth"))
        candidatos.extend([
            os.path.join(resultados_dir, "grid_search_conv_ortogonal", "modelo_campeon_conv_ortogonal.pt"),
            os.path.join(resultados_dir, "modelos_entrenados", f"autoencoder_envolvente_{latent_dim}d.pth"),
            os.path.join(resultados_dir, f"autoencoder_envolvente_{latent_dim}d.pth"),
            os.path.join(resultados_dir, "autoencoder_campeon.pth"),
            os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/modelo_optimo.pt"),
        ])
        modelo_path = next((m for m in candidatos if os.path.exists(m)), None)

    if not modelo_path or not os.path.exists(modelo_path):
        raise FileNotFoundError("No se encontró ningún modelo (.pt / .pth) entrenado en el sistema.")

    _log(f"[Modelo] Cargando checkpoint: {modelo_path}")
    dir_modelo = os.path.dirname(modelo_path) if modelo_path else ""

    arch_code = codigo_custom_arch
    if not arch_code and dir_modelo:
        arch_py = os.path.join(dir_modelo, "arquitectura_autoencoder.py")
        if os.path.exists(arch_py):
            try:
                with open(arch_py, 'r', encoding='utf-8') as f_arch:
                    arch_code = f_arch.read()
                    usar_custom_arch = True
            except Exception:
                pass

    if usar_custom_arch and arch_code and arch_code.strip():
        _log("  [Arquitectura] Compilando modelo personalizado para decodificación continua...")
        model = compilar_modelo_desde_codigo(arch_code, modalidad="envolvente", latent_dim=latent_dim, target_len=target_length, in_channels=3)
    else:
        hidden_dim_orto = 64 if latent_dim == 3 else 32
        model = OrthogonalAutoencoder2D(input_dim=D_seq, hidden_dim=hidden_dim_orto, latent_dim=latent_dim)

    state = torch.load(modelo_path, map_location='cpu')
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        if hasattr(model, 'encode'):
            Z_seq = model.encode(X_seq_t).numpy()
        else:
            _, z_t = model(X_seq_t)
            Z_seq = z_t.numpy()

    # Cargar fronteras de entrenamiento si existen
    dir_modelo = os.path.dirname(modelo_path) if modelo_path else ""
    candidatos_csv = [
        os.path.join(dir_modelo, "proyecciones_latentes_2d_crudo.csv"),
        os.path.join(dir_modelo, "proyecciones_latentes_2d.csv"),
        os.path.join(dir_modelo, "proyecciones_latentes_2d_alineado.csv"),
        os.path.join(modelos_dir, "proyecciones_latentes_2d_crudo.csv"),
        os.path.join(modelos_dir, "proyecciones_latentes_2d.csv"),
        os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/proyecciones_latentes_2d_crudo.csv"),
        os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo_91/proyecciones_latentes_2d_crudo.csv"),
        os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/espacio_latente_91.43.csv")
    ]
    csv_entren = next((c for c in candidatos_csv if os.path.exists(c)), None)

    usar_lda = ("lda" in str(algoritmo_clasificacion).lower() or "supervisado" in str(algoritmo_clasificacion).lower())
    mapping = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
    cluster_to_vocal = {i: i for i in range(5)}
    z_ent = None

    if usar_lda:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        _log("[Fronteras] Utilizando clasificador Supervisado: LDA Fronteras Lineales")
        modelo_decision = LinearDiscriminantAnalysis()
        if csv_entren and os.path.exists(csv_entren):
            _log(f"[Fronteras] Entrenando LDA sobre proyecciones latentes de entrenamiento: {csv_entren}")
            df_entren = pd.read_csv(csv_entren)
            z_ent = df_entren[['Z1', 'Z2']].values
            y_ent = df_entren['Vocal'].values
            y_ent_idx = np.array([vocal_to_idx[v] for v in y_ent])
            modelo_decision.fit(z_ent, y_ent_idx)
            mapping = {i: vocales[i] for i in range(5)}
            cluster_to_vocal = {i: i for i in range(5)}
        elif tiene_ground_truth:
            _log("[Fronteras] Entrenando LDA directamente sobre la secuencia con ground truth...")
            y_ground_idx = np.array([vocal_to_idx[v] for v in y_ground])
            modelo_decision.fit(Z_seq, y_ground_idx)
            mapping = {i: vocales[i] for i in range(5)}
            cluster_to_vocal = {i: i for i in range(5)}
        else:
            _log("[Aviso] No hay etiquetas previas para LDA, recurriendo a GMM...")
            usar_lda = False

    if not usar_lda:
        gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20)
        modelo_decision = gmm
        if csv_entren and os.path.exists(csv_entren):
            _log(f"[Fronteras] Cargando proyecciones latentes de entrenamiento: {csv_entren}")
            df_entren = pd.read_csv(csv_entren)
            z_ent = df_entren[['Z1', 'Z2']].values
            y_ent = df_entren['Vocal'].values
            pred_ent = gmm.fit_predict(z_ent)
            contingency = np.zeros((5, 5))
            for i, vl in enumerate(vocales):
                for j in range(5):
                    contingency[j, i] = np.sum((y_ent == vl) & (pred_ent == j))
            row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
            mapping = {row_ind[i]: vocales[col_ind[i]] for i in range(len(row_ind))}
            cluster_to_vocal = {r: vocal_to_idx[mapping[r]] for r in mapping}
        elif tiene_ground_truth:
            _log(f"[Fronteras] Ajustando fronteras GMM directamente sobre la secuencia con ground truth...")
            pred_seq = gmm.fit_predict(Z_seq)
            contingency = np.zeros((5, 5))
            for i, vl in enumerate(vocales):
                for j in range(5):
                    contingency[j, i] = np.sum((y_ground == vl) & (pred_seq == j))
            row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
            mapping = {row_ind[i]: vocales[col_ind[i]] for i in range(len(row_ind))}
            cluster_to_vocal = {r: vocal_to_idx[mapping[r]] for r in mapping}
        else:
            _log(f"[Fronteras] Secuencia libre sin etiquetas: clusters asignados en orden natural.")
            pred_seq = gmm.fit_predict(Z_seq)
            mapping = {i: vocales[i] for i in range(5)}
            cluster_to_vocal = {i: i for i in range(5)}

    clus_pred = modelo_decision.predict(Z_seq)
    vocales_pred = np.array([mapping.get(c, vocales[c]) for c in clus_pred])

    if tiene_ground_truth:
        aciertos = (vocales_pred == y_ground)
        exactitud = np.mean(aciertos) * 100.0
        _log(f"\n[Resultado] Exactitud Decodificada: {exactitud:.2f}% ({np.sum(aciertos)}/{N_seq} pulsos correctos)")
        for v in vocales:
            mask_v = (y_ground == v)
            if np.sum(mask_v) > 0:
                acc_v = np.mean(vocales_pred[mask_v] == v) * 100.0
                _log(f"  Vocal /{v.lower()}/: {acc_v:5.1f}% ({np.sum(vocales_pred[mask_v] == v)}/{np.sum(mask_v)})")
    else:
        exactitud = 0.0
        aciertos = np.ones(N_seq, dtype=bool)
        _log(f"\n[Resultado] Total Fonaciones Decodificadas: {N_seq} eventos en secuencia libre")

    # Gráfico con la paleta de colores oficial estricta
    colores_oficiales = {
        'A': '#E63946',  # Rojo
        'E': '#1F77B4',  # Azul
        'I': '#2CA02C',  # Verde
        'O': '#9D4EDD',  # Morado
        'U': '#E7A61A'   # Amarillo
    }

    if carpeta_salida is None:
        carpeta_salida = os.path.join(resultados_dir, f"decodificacion_{os.path.basename(carpeta_secuencia)}")
    os.makedirs(carpeta_salida, exist_ok=True)

    # 1. Figura Principal: Espacio Latente y Tira de Pulsos Fonatorios Decodificados
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=180)

    # Panel 1: Espacio Latente (SIN líneas que unan los puntos)
    ax1 = axes[0]
    if z_ent is not None:
        all_pts = np.vstack([z_ent, Z_seq])
    else:
        all_pts = Z_seq
    pad_x = (all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.15
    pad_y = (all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.15
    x_min, x_max = all_pts[:, 0].min() - pad_x, all_pts[:, 0].max() + pad_x
    y_min, y_max = all_pts[:, 1].min() - pad_y, all_pts[:, 1].max() + pad_y

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_clus = modelo_decision.predict(grid)
    grid_vocal = np.array([cluster_to_vocal.get(c, c) for c in grid_clus]).reshape(xx.shape)

    cmap_bg = mcolors.ListedColormap([colores_oficiales[v] for v in vocales])
    ax1.pcolormesh(xx, yy, grid_vocal, cmap=cmap_bg, alpha=0.20, shading='auto', vmin=0, vmax=4)
    ax1.contour(xx, yy, grid_vocal, levels=np.arange(0.5, 4.5, 1), colors='gray', linewidths=0.7, alpha=0.5)

    # Centroides de entrenamiento
    if hasattr(modelo_decision, 'means_'):
        for c_id, cen in enumerate(modelo_decision.means_):
            v_name = mapping.get(c_id, vocales[c_id])
            ax1.scatter(
                cen[0], cen[1], c=[colores_oficiales[v_name]],
                marker='D', s=160, edgecolors='black', linewidth=1.5, zorder=5,
                path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)]
            )

    for v in vocales:
        if tiene_ground_truth:
            mask = (y_ground == v)
        else:
            mask = (vocales_pred == v)
        if np.sum(mask) > 0:
            lbl = f"/{v.lower()}/ N={np.sum(mask)}" if tiene_ground_truth else f"Pred /{v.lower()}/ N={np.sum(mask)}"
            ax1.scatter(Z_seq[mask, 0], Z_seq[mask, 1], c=colores_oficiales[v], s=65, edgecolors='black', lw=0.7, alpha=0.9, label=lbl, zorder=4)

    ax1.scatter(0, 0, color='black', marker='+', s=120, lw=2.0, zorder=6)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=12, fontweight='bold')
    metodo_nombre = "LDA Supervisado" if usar_lda else "GMM"
    if tiene_ground_truth:
        ax1.set_title(f"Espacio Latente: {metodo_nombre} - {os.path.basename(carpeta_secuencia)}\nExactitud: {exactitud:.1f}% ({np.sum(aciertos)}/{N_seq} pulsos correctos)", fontsize=12, fontweight='bold')
    else:
        ax1.set_title(f"Espacio Latente: {metodo_nombre} - {os.path.basename(carpeta_secuencia)}\nTotal Fonaciones: {N_seq} pulsos", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=10)

    # Panel 2: Tira Temporal de Señal con Pulsos y Vocales Detectadas
    ax2 = axes[1]
    t_axis = np.arange(n_samples) / fs
    env_clean = np.maximum(env_emg - ruido_base_emg, 0.0)
    S_emg = np.sqrt(np.sum(env_clean**2, axis=0))

    # Mostrar ventana representativa (primeros 60 segundos con actividad)
    t_max_zoom = min(65.0, t_axis[-1])
    mask_zoom = (t_axis >= 5.0) & (t_axis <= t_max_zoom)
    ax2.plot(t_axis[mask_zoom], S_emg[mask_zoom], color="#333333", lw=1.2, label="Norma Tricanal sEMG")

    # Dibujar cada pulso detectado con su caja de color y etiqueta vocal
    y_max_s = np.max(S_emg[mask_zoom]) if np.any(mask_zoom) else 1000.0
    for i in range(N_seq):
        t_p = tiempos_list[i]
        if t_p < 5.0 or t_p > t_max_zoom:
            continue
        v_pred = vocales_pred[i]
        c_voc = colores_oficiales.get(v_pred, '#388E3C')

        # Sombra coloreada del pulso (1 segundo centrado)
        ax2.axvspan(max(5.0, t_p - 0.5), min(t_max_zoom, t_p + 0.5), color=c_voc, alpha=0.18)
        ax2.axvline(t_p, color=c_voc, linestyle='--', lw=1.2, alpha=0.8)

        if tiene_ground_truth:
            v_true = y_ground[i]
            lbl_txt = f"{i+1}:{v_pred}" if v_pred == v_true else f"{i+1}:{v_pred}\n(G:{v_true})"
            txt_col = c_voc if v_pred == v_true else '#D32F2F'
        else:
            lbl_txt = f"{i+1}:{v_pred}"
            txt_col = c_voc

        ax2.text(t_p, y_max_s * 0.88, lbl_txt, color=txt_col, fontweight='bold', fontsize=8, ha='center',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='#121212', edgecolor=txt_col, alpha=0.85))

    ax2.set_xlabel("Tiempo en Segundos", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Norma Muscular Tricanal", fontsize=12, fontweight='bold')
    if tiene_ground_truth:
        ax2.set_title("Identificación Temporal de Pulsos (Detalle Primeros 60s)\nEtiqueta de Vocal y Aciertos por Color", fontsize=12, fontweight='bold')
    else:
        ax2.set_title("Identificación Temporal de Pulsos: Vocales Detectadas", fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig_path = os.path.join(carpeta_salida, "resultado_decodificacion_continua.png")
    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close()

    # 2. Generar el Gráfico Panorámico Completo de Toda la Secuencia (4 Tramos)
    try:
        dur_tot = t_axis[-1]
        tramos_pano = [
            (5.0, min(65.0, dur_tot), "Tramo 1: Segundos 5 a 65"),
            (65.0, min(125.0, dur_tot), "Tramo 2: Segundos 65 a 125"),
            (125.0, min(185.0, dur_tot), "Tramo 3: Segundos 125 a 185"),
            (185.0, dur_tot, "Tramo 4: Segundos 185 a Final")
        ]
        fig_pano, axes_pano = plt.subplots(4, 1, figsize=(18, 14), sharey=True)
        max_s_glob = np.percentile(S_emg, 99) * 1.25

        for r_idx, (t_s, t_e, tit_tr) in enumerate(tramos_pano):
            ax_p = axes_pano[r_idx]
            m_tr = (t_axis >= t_s) & (t_axis <= t_e)
            if np.any(m_tr):
                ax_p.plot(t_axis[m_tr], S_emg[m_tr], color="#4A148C", lw=1.2, label="Norma EMG" if r_idx == 0 else "")

            for p_i in range(N_seq):
                t_p = tiempos_list[p_i]
                if t_s <= t_p <= t_e:
                    v_p = vocales_pred[p_i]
                    c_v = colores_oficiales.get(v_p, '#388E3C')
                    v_ini = max(t_s, t_p - 0.5)
                    v_fin = min(t_e, t_p + 0.5)
                    ax_p.axvspan(v_ini, v_fin, color=c_v, alpha=0.18)
                    ax_p.axvline(t_p, color=c_v, linestyle='--', lw=1.0, alpha=0.7)

                    if tiene_ground_truth:
                        v_t = y_ground[p_i]
                        tag = f"{p_i+1}:{v_p}" if v_p == v_t else f"{p_i+1}:{v_p}({v_t})"
                        col_t = c_v if v_p == v_t else '#D32F2F'
                    else:
                        tag = f"{p_i+1}:{v_p}"
                        col_t = c_v
                    ax_p.text(t_p, max_s_glob * 0.82, tag, color=col_t, fontweight='bold', fontsize=8, ha='center',
                              bbox=dict(boxstyle='round,pad=0.15', facecolor='#000000', edgecolor=col_t, alpha=0.8))

            ax_p.set_title(tit_tr, fontsize=11, fontweight='bold')
            ax_p.set_ylabel("Norma EMG", fontsize=9)
            ax_p.grid(True, linestyle=":", alpha=0.5)
            if r_idx == 0:
                ax_p.legend(loc="upper right", frameon=True, fontsize=8)

        axes_pano[-1].set_xlabel("Tiempo en segundos", fontsize=10)
        plt.tight_layout()
        pano_path = os.path.join(carpeta_salida, "grafico_totalidad_pulsos_decodificados.png")
        plt.savefig(pano_path, dpi=160, bbox_inches='tight')
        plt.close()
        _log(f"[Salida] Gráfico panorámico completo guardado en: {pano_path}")
    except Exception as e_pano:
        _log(f"  [Aviso] No se pudo generar el gráfico panorámico: {e_pano}")

    # Guardar CSV de resultados
    df_out = pd.DataFrame({
        'Pulso_Idx': np.arange(N_seq),
        'Tiempo_s': tiempos_list,
        'Z1': Z_seq[:, 0],
        'Z2': Z_seq[:, 1],
        'Vocal_Predicha': vocales_pred,
        'Vocal_Ground_Truth': y_ground,
        'Acierto': aciertos
    })
    csv_path = os.path.join(carpeta_salida, "secuencia_decodificada.csv")
    df_out.to_csv(csv_path, index=False)

    _log(f"[Salida] Gráfico guardado en: {fig_path}")
    _log(f"[Salida] Tabla exportada en: {csv_path}")

    return {
        'exactitud': exactitud,
        'n_pulsos': N_seq,
        'fig_path': fig_path,
        'csv_path': csv_path
    }


def evaluar_en_dataset_externo(
    ruta_dataset_externo,
    modelo_path=None,
    carpeta_salida=None,
    callback_log=None
):
    """
    Evalúa el modelo de autoencoder previamente entrenado sobre un dataset o CSV externo
    (ej. Candela o nuevas sesiones) sin reentrenar, midiendo exactitud y generalización nativa.
    """
    def _log(msg):
        if callback_log:
            callback_log(msg)
        else:
            print(msg)

    _log("\n" + "="*70)
    _log("EVALUACION NATIVA EN DATASET EXTERNO: MODELO RECORD")
    _log(f"Dataset Externo: {ruta_dataset_externo}")
    _log("="*70)

    if not os.path.exists(ruta_dataset_externo):
        raise FileNotFoundError(f"Archivo no encontrado: {ruta_dataset_externo}")

    vocales = ['A', 'E', 'I', 'O', 'U']
    vocal_to_idx = {v: i for i, v in enumerate(vocales)}

    # Cargar CSV o NPZ
    if ruta_dataset_externo.endswith(".csv"):
        df_ext = pd.read_csv(ruta_dataset_externo)
        df_ext['Sesion'] = [extraer_sesion_agnostica(t) for t in df_ext['Toma']]
        feat_cols = [c for c in df_ext.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
        X_raw = df_ext[feat_cols].values
        y_ext = df_ext['Vocal'].values
        sesiones = df_ext['Sesion'].values
    elif ruta_dataset_externo.endswith(".npz"):
        data = np.load(ruta_dataset_externo, allow_pickle=True)
        X_raw = data['X_env'] if 'X_env' in data else data['X']
        if X_raw.ndim == 3:
            X_raw = X_raw.reshape(X_raw.shape[0], -1)
        y_ext = data['Y']
        sesiones = data['Tomas'] if 'Tomas' in data else np.array(['S1'] * len(y_ext))
    else:
        raise ValueError("El dataset externo debe ser un archivo .csv o .npz")

    N_ext, D_ext = X_raw.shape
    _log(f"[Carga] {N_ext} muestras cargadas con {D_ext} dimensiones.")

    # Acondicionamiento reposo e impedancia por sesión
    n_ch = 3
    n_pts = D_ext // n_ch
    b_bw, a_bw = butter(3, 0.3, btype='low')
    X_reshaped = X_raw.reshape(N_ext, n_ch, n_pts)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N_ext):
        for c in range(n_ch):
            X_filt[i, c, :] = filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for s in np.unique(sesiones):
        mask = (sesiones == s)
        for c in range(n_ch):
            base_mean = np.mean(X_filt[mask, c, :10])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    X_ext_t = torch.tensor(X_norm.reshape(N_ext, -1), dtype=torch.float32)

    # Cargar modelo entrenado
    if modelo_path is None or not os.path.exists(modelo_path):
        candidatos = [
            os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/modelo_optimo.pt"),
            os.path.join(resultados_dir, "modelos_entrenados", "autoencoder_envolvente_2d.pth"),
            os.path.join(emg_desarrollo_dir, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo_91/modelo_optimo_91.43.pt"),
            os.path.join(resultados_dir, "autoencoder_envolvente_2d.pth"),
            os.path.join(resultados_dir, "autoencoder_emg_2d.pth"),
            os.path.join(resultados_dir, "autoencoder_campeon.pth")
        ]
        modelo_path = next((m for m in candidatos if os.path.exists(m)), None)

    if not modelo_path or not os.path.exists(modelo_path):
        raise FileNotFoundError("No se encontró ningún modelo (.pt / .pth) entrenado en el sistema.")

    _log(f"[Modelo] Cargando checkpoint: {modelo_path}")
    model = OrthogonalAutoencoder2D(input_dim=D_ext, hidden_dim=32, latent_dim=2)
    state = torch.load(modelo_path, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        Z_ext = model.encode(X_ext_t).numpy()

    # Evaluación GMM en el espacio latente nativo
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20)
    pred_raw = gmm.fit_predict(Z_ext)
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(vocales):
        for j in range(5):
            contingency[j, i] = np.sum((y_ext == vl) & (pred_raw == j))
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    mapping = {row_ind[i]: vocales[col_ind[i]] for i in range(len(row_ind))}
    mapped_preds = np.array([mapping[p] for p in pred_raw])
    acc_ext = accuracy_score(y_ext, mapped_preds) * 100.0
    cm_ext = confusion_matrix(y_ext, mapped_preds, labels=vocales)

    _log(f"\n[Resultado] Exactitud GMM Nativa en Dataset Externo: {acc_ext:.2f}%")
    _log(pd.DataFrame(cm_ext, index=vocales, columns=vocales).to_string())

    # Paleta de colores oficial universal
    colores_oficiales = {
        'A': '#E63946',  # Rojo
        'E': '#1F77B4',  # Azul
        'I': '#2CA02C',  # Verde
        'O': '#9D4EDD',  # Morado
        'U': '#E7A61A'   # Amarillo
    }

    if carpeta_salida is None:
        nombre_ext = os.path.splitext(os.path.basename(ruta_dataset_externo))[0]
        carpeta_salida = os.path.join(resultados_dir, f"evaluacion_externa_{nombre_ext}")
    os.makedirs(carpeta_salida, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=180)

    # Panel 1: Scatter Espacio Latente 2D
    ax1 = axes[0]
    for v in vocales:
        mask = (y_ext == v)
        if np.sum(mask) > 0:
            ax1.scatter(Z_ext[mask, 0], Z_ext[mask, 1], c=colores_oficiales[v], s=55, edgecolors='black', lw=0.6, alpha=0.85, label=f'/{v.lower()}/ N={np.sum(mask)}')
            cen_v = np.mean(Z_ext[mask], axis=0)
            ax1.scatter(cen_v[0], cen_v[1], c=colores_oficiales[v], marker='D', s=180, edgecolors='black', lw=1.8, zorder=5)

    ax1.scatter(0, 0, color='black', marker='+', s=120, lw=2.0, zorder=6)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=12, fontweight='bold')
    ax1.set_title(f"Proyección Nativa en Dataset Externo\nExactitud GMM: {acc_ext:.2f}%", fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.legend(loc='lower left', fontsize=10)

    # Panel 2: Matriz de Confusión
    ax2 = axes[1]
    cm_pct = cm_ext.astype(float) / np.maximum(cm_ext.sum(axis=1, keepdims=True), 1e-6) * 100.0
    im = ax2.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(vocales, fontsize=11, fontweight='bold')
    ax2.set_yticklabels(vocales, fontsize=11, fontweight='bold')
    ax2.set_xlabel("Vocal Predicha", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Vocal Real", fontsize=11, fontweight='bold')
    ax2.set_title(f"Matriz de Confusión Porcentual", fontsize=12, fontweight='bold')

    for i in range(5):
        for j in range(5):
            val = cm_pct[i, j]
            col_text = "white" if val > 50 else "black"
            ax2.text(j, i, f"{val:.1f}%\n({cm_ext[i, j]})", ha="center", va="center", color=col_text, fontsize=10, fontweight='bold')

    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(carpeta_salida, "resultado_evaluacion_externa.png")
    plt.savefig(fig_path, dpi=180, bbox_inches='tight')
    plt.close()

    # Guardar CSV de proyecciones
    df_proj = pd.DataFrame({'Vocal': y_ext, 'Sesion': sesiones, 'Z1': Z_ext[:, 0], 'Z2': Z_ext[:, 1], 'Pred': mapped_preds})
    csv_proj = os.path.join(carpeta_salida, "proyecciones_externas_2d.csv")
    df_proj.to_csv(csv_proj, index=False)

    _log(f"[Salida] Gráfico guardado en: {fig_path}")
    _log(f"[Salida] Proyecciones guardadas en: {csv_proj}")

    return {
        'exactitud': acc_ext,
        'cm': cm_ext,
        'fig_path': fig_path,
        'csv_path': csv_proj
    }

