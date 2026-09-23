# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generador oficial de figuras multimodales de alta resolución (4 paneles)
#              para publicaciones científicas y reportes de laboratorio:
#              - Panel 0: Espectrograma de audio STFT (Pre-énfasis 0.97, Greys, 0-2500 Hz, 45 dB)
#              - Panel 1: Micrófono rectificado y envolvente acústica rápida normalizada a 1.0
#              - Panel 2: Envolventes EMG acondicionadas (NLMS, 75 ms, Supremo Tricanal del pulso)
#              - Panel 3: Espectrograma EMG RGB con Pre-énfasis 0.95 (20-600 Hz)
#              - Alineación causal exacta con t=0 en el inicio acústico y línea vertical punteada
# ==============================================================================

import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl_nandu')
import json
import argparse
import numpy as np
import scipy.io.wavfile as wav
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Configuración de rutas del proyecto
analysis_dir = os.path.dirname(os.path.abspath(__file__))
emg_desarrollo_dir = os.path.abspath(os.path.join(analysis_dir, ".."))
if emg_desarrollo_dir not in sys.path:
    sys.path.append(emg_desarrollo_dir)
if analysis_dir not in sys.path:
    sys.path.append(analysis_dir)

from analysis.analisis_espectral_candela import (
    acondicionar_senal_canal,
    calcular_espectrograma,
    construir_espectrograma_coloreado
)

def acondicionar_emg_canal(senal, fs):
    """Acondiciona la señal EMG mediante NLMS adaptativo y calcula envolvente suave (75 ms)."""
    s_filt = acondicionar_senal_canal(senal, fs=fs, noise_samples=int(0.2 * fs))
    win_len = max(5, int(0.075 * fs))
    env = np.convolve(np.abs(s_filt), np.ones(win_len) / win_len, mode='same')
    return s_filt, env

def generar_figura_paper_multimodal(toma_path, out_file=None, pulso_idx=1, logger=print):
    """
    Genera la figura multimodal de 4 paneles para una toma individual de registro.
    
    Parámetros:
        toma_path: Ruta absoluta o relativa al directorio de la medición.
        out_file: Ruta de guardado opcional. Si es None, guarda 'plot_paper_combined.png' en toma_path.
        pulso_idx: Índice del pulso a graficar (por defecto 1, segundo pulso).
        logger: Función para emitir mensajes de log.
        
    Retorna:
        str: Ruta absoluta al archivo PNG generado.
    """
    toma_path = os.path.abspath(toma_path)
    if not os.path.isdir(toma_path):
        raise ValueError(f"Directorio de medición no válido: {toma_path}")
        
    # 1. Leer metadatos de la toma
    meta_path = os.path.join(toma_path, "canal_0", "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(toma_path, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No se encontró metadata.json en {toma_path}")
        
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    fs = int(meta.get('sample_rate', 6000))
    bpm = float(meta.get('bpm', 30.0))
    noise_sec = float(meta.get('noise_seconds', 5.0))
    pulse_count = int(meta.get('pulse_count', 5))
    vocal = meta.get('letra', os.path.basename(toma_path)[0]).upper()
    prueba_nom = meta.get('prueba', os.path.basename(toma_path))
    sujeto = meta.get('sujeto', 'Sujeto')
    
    # Identificar nombres de músculos por canal
    muscles_map = meta.get('muscles_map', {})
    m0_name = muscles_map.get('canal_0', meta.get('musculo', 'Canal 0')).strip()
    m1_name = muscles_map.get('canal_1', '-').strip()
    m2_name = muscles_map.get('canal_2', 'Canal 2').strip()
    
    # 2. Leer audios de los canales
    def cargar_audio(ch_name):
        p = os.path.join(toma_path, ch_name, "grabacion.wav")
        if os.path.exists(p):
            _, d = wav.read(p)
            if d.ndim > 1:
                d = d[:, 0]
            return d.astype(float)
        return None
        
    s0_raw = cargar_audio("canal_0")
    s1_raw = cargar_audio("canal_1")
    s2_raw = cargar_audio("canal_2")
    s3_raw = cargar_audio("canal_3") # Micrófono
    
    if s3_raw is None or s0_raw is None:
        raise ValueError(f"Faltan canales esenciales (canal_0 o canal_3) en {toma_path}")
        
    # Remoción de offset de continua del micrófono
    s3 = s3_raw - np.mean(s3_raw)
    
    # Conversión física sEMG
    s0_uv = (s0_raw / 32768.0) * 10.0 * 1000.0
    raw0, env0 = acondicionar_emg_canal(s0_uv, fs)
    
    raw1, env1 = None, None
    ch1_activo = (s1_raw is not None) and (m1_name not in ['-', '', 'sin medicion', 'none'])
    if ch1_activo:
        s1_uv = (s1_raw / 32768.0) * 10.0 * 1000.0
        raw1, env1 = acondicionar_emg_canal(s1_uv, fs)
    else:
        raw1 = np.zeros_like(raw0)
        env1 = np.zeros_like(env0)
        
    raw2, env2 = None, None
    ch2_activo = (s2_raw is not None) and (m2_name not in ['-', '', 'sin medicion', 'none'])
    if ch2_activo:
        s2_uv = (s2_raw / 32768.0) * 10.0 * 1000.0
        raw2, env2 = acondicionar_emg_canal(s2_uv, fs)
    else:
        raw2 = np.zeros_like(raw0)
        env2 = np.zeros_like(env0)
        
    # 3. Segmentación y centrado del evento fonatorio
    muestras_pulso = int(round((60.0 / bpm) * fs))
    if pulso_idx >= pulse_count:
        pulso_idx = max(0, pulse_count - 1)
        
    start_idx = int(noise_sec * fs) + pulso_idx * muestras_pulso
    end_idx = min(len(s3), start_idx + muestras_pulso)
    
    # Envolvente gruesa del micrófono para localizar el pico acustico
    win_coarse = max(5, int(0.05 * fs))
    mic_env_temp = np.convolve(np.abs(s3[start_idx:end_idx]), np.ones(win_coarse) / win_coarse, mode='same')
    rel_max = np.argmax(mic_env_temp)
    p_center = start_idx + rel_max
    
    pre = int(0.45 * muestras_pulso)
    post = int(0.55 * muestras_pulso)
    p_start = max(0, p_center - pre)
    p_end = min(len(s3), p_center + post)
    
    seg_mic = s3[p_start:p_end]
    seg_mic = seg_mic - np.mean(seg_mic)
    
    # Detección robusta de inicio acústico (búsqueda hacia atrás desde el pico)
    win_local = max(5, int(0.02 * fs))
    mic_env_local = np.convolve(np.abs(seg_mic), np.ones(win_local) / win_local, mode='same')
    center_idx = np.argmax(mic_env_local)
    max_local_mic = mic_env_local[center_idx]
    thresh_onset = 0.10 * max_local_mic
    
    below_thresh = np.where(mic_env_local[:center_idx] < thresh_onset)[0]
    if len(below_thresh) > 0:
        onset_idx = below_thresh[-1] + 1
    else:
        onset_idx = max(0, center_idx - int(0.15 * fs))
        
    # Eje temporal con t=0 en el inicio acústico
    t_axis = (np.arange(len(seg_mic)) - onset_idx) / fs
    
    seg_env0 = env0[p_start:p_end]
    seg_env1 = env1[p_start:p_end] if ch1_activo else np.zeros_like(seg_env0)
    seg_env2 = env2[p_start:p_end] if ch2_activo else np.zeros_like(seg_env0)
    
    # Sustracción robusta de ruido basal interpulso (primeros 100 ms)
    n_base = max(10, int(0.10 * fs))
    ruido_0 = np.median(seg_env0[:n_base]) if len(seg_env0) > n_base else np.min(seg_env0)
    seg_env0 = np.maximum(0.0, seg_env0 - ruido_0)
    
    if ch1_activo:
        ruido_1 = np.median(seg_env1[:n_base]) if len(seg_env1) > n_base else np.min(seg_env1)
        seg_env1 = np.maximum(0.0, seg_env1 - ruido_1)
        
    if ch2_activo:
        ruido_2 = np.median(seg_env2[:n_base]) if len(seg_env2) > n_base else np.min(seg_env2)
        seg_env2 = np.maximum(0.0, seg_env2 - ruido_2)
        
    # 4. Creación del Panel Gráfico de 4 Subplots
    fig = plt.figure(figsize=(10, 11))
    gs = fig.add_gridspec(4, 2, width_ratios=[0.97, 0.03], height_ratios=[1.5, 1, 1, 1.5], wspace=0.02, hspace=0.28)
    
    ax0 = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)
    axes = [ax0, ax1, ax2, ax3]
    
    # Ocultar etiquetas numéricas intermedias para evitar superposiciones tipográficas
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.tick_params(labelbottom=False)
    
    # --- PANEL 0: Espectrograma de Audio STFT con Pre-énfasis ---
    seg_mic_pre = np.append(seg_mic[0], seg_mic[1:] - 0.97 * seg_mic[:-1])
    nperseg_audio = 128 if fs >= 4000 else 64
    noverlap_audio = int(0.96 * nperseg_audio)
    f, t_spec, Sxx = spectrogram(seg_mic_pre, fs, nperseg=nperseg_audio, noverlap=noverlap_audio, mode='magnitude')
    
    f_mask = f <= 2500
    f = f[f_mask]
    Sxx = Sxx[f_mask, :]
    
    Sxx_db = 20 * np.log10(np.maximum(Sxx, 1e-10))
    vmax_db = np.max(Sxx_db)
    vmin_db = vmax_db - 45.0
    
    dt_step_audio = t_spec[1] - t_spec[0]
    t_spec_aligned = t_spec - (onset_idx / fs)
    extent_audio = [t_spec_aligned[0] - dt_step_audio / 2, t_spec_aligned[-1] + dt_step_audio / 2, f[0], f[-1]]
    
    im = ax0.imshow(Sxx_db, aspect='auto', origin='lower',
                    extent=extent_audio, cmap='Greys',
                    vmin=vmin_db, vmax=vmax_db, interpolation='bilinear')
    ax0.set_title(f"Análisis Multimodal Fonético: Vocal /{vocal}/ - {prueba_nom} ({sujeto})", fontsize=13, fontweight='bold')
    ax0.set_ylabel("Frecuencia (Hz)\n[Audio STFT]", fontweight='bold')
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Magnitud (dB)", rotation=270, labelpad=18, fontweight='bold')
    
    # --- PANEL 1: Micrófono (Oscilograma rectificado + Envolvente normalizada) ---
    max_mic = np.max(np.abs(seg_mic)) if np.max(np.abs(seg_mic)) > 0 else 1.0
    seg_mic_norm = seg_mic / max_mic
    
    win_fast = max(5, int(0.015 * fs))
    mic_fast_env = np.convolve(np.abs(seg_mic_norm), np.ones(win_fast) / win_fast, mode='same')
    max_mic_env = np.max(mic_fast_env) if np.max(mic_fast_env) > 0 else 1.0
    mic_fast_env_norm = mic_fast_env / max_mic_env
    
    ax1.plot(t_axis, np.abs(seg_mic_norm), color='gray', linewidth=0.6, alpha=0.55, label='Audio rectificado')
    ax1.plot(t_axis, mic_fast_env_norm, color='black', linewidth=1.5, label='Envolvente acústica')
    ax1.set_ylabel("Amplitud Norm.\n[Micrófono]", fontweight='bold')
    ax1.set_ylim([-0.05, 1.1])
    ax1.legend(loc='upper right', framealpha=0.85)
    ax1.grid(True, alpha=0.3)
    
    # --- PANEL 2: Activación Muscular Normalizada (Supremo Tricanal) ---
    activas_env = [seg_env0]
    if ch1_activo:
        activas_env.append(seg_env1)
    if ch2_activo:
        activas_env.append(seg_env2)
    max_supremo = max([np.max(e) for e in activas_env])
    if max_supremo <= 0:
        max_supremo = 1.0
        
    ax2.plot(t_axis, seg_env0 / max_supremo, color='#E63946', linewidth=2.5, label=f'{m0_name} (Ch0)')
    if ch1_activo:
        ax2.plot(t_axis, seg_env1 / max_supremo, color='#2A9D8F', linewidth=2.5, label=f'{m1_name} (Ch1)')
    if ch2_activo:
        ax2.plot(t_axis, seg_env2 / max_supremo, color='#F77F00', linewidth=2.5, label=f'{m2_name} (Ch2)')
        
    ax2.set_ylabel("Activación Norm.\n[Envolvente EMG]", fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.85)
    ax2.grid(True, alpha=0.3)
    
    # --- PANEL 3: Espectrograma RGB Muscular con Pre-énfasis ---
    seg_raw0 = raw0[p_start:p_end]
    seg_raw1 = raw1[p_start:p_end] if ch1_activo else np.zeros_like(seg_raw0)
    seg_raw2 = raw2[p_start:p_end] if ch2_activo else np.zeros_like(seg_raw0)
    
    seg_raw0_pre = np.append(seg_raw0[0], seg_raw0[1:] - 0.95 * seg_raw0[:-1])
    seg_raw1_pre = np.append(seg_raw1[0], seg_raw1[1:] - 0.95 * seg_raw1[:-1]) if ch1_activo else seg_raw1
    seg_raw2_pre = np.append(seg_raw2[0], seg_raw2[1:] - 0.95 * seg_raw2[:-1]) if ch2_activo else seg_raw2
    
    nperseg_emg = 256 if fs >= 4000 else 128
    noverlap_emg = int(0.90 * nperseg_emg)
    
    f_stft, t_stft, Sxx0 = calcular_espectrograma(seg_raw0_pre, fs, nperseg=nperseg_emg, noverlap=noverlap_emg, f_max=600.0)
    _, _, Sxx1 = calcular_espectrograma(seg_raw1_pre, fs, nperseg=nperseg_emg, noverlap=noverlap_emg, f_max=600.0)
    _, _, Sxx2 = calcular_espectrograma(seg_raw2_pre, fs, nperseg=nperseg_emg, noverlap=noverlap_emg, f_max=600.0)
    
    color_img = construir_espectrograma_coloreado(Sxx0, Sxx1, Sxx2)
    
    dt_step_emg = t_stft[1] - t_stft[0]
    t_stft_aligned = t_stft - (onset_idx / fs)
    extent_emg = [t_stft_aligned[0] - dt_step_emg / 2, t_stft_aligned[-1] + dt_step_emg / 2, 20, 600]
    
    ax3.imshow(color_img, origin='lower', aspect='auto', extent=extent_emg, interpolation='bicubic')
    ax3.set_ylabel("Frecuencia EMG (Hz)", fontweight='bold')
    ax3.set_xlabel("Tiempo relativo a la fonación (s) [t = 0: inicio acústico]", fontweight='bold', fontsize=11)
    
    label_rgb = f"Rojo: {m0_name}"
    if ch1_activo:
        label_rgb += f" | Verde: {m1_name}"
    if ch2_activo:
        label_rgb += f" | Amarillo: {m2_name}"
    ax3.set_title(f"Espectrograma EMG RGB con Pre-énfasis ({label_rgb})", fontsize=11, fontweight='bold')
    
    # Línea vertical punteada de referencia t=0 sincronizada
    for ax in axes:
        ax.axvline(0.0, color='black', linestyle='--', linewidth=1.3, alpha=0.85)
        
    ax0.set_xlim([-0.45, 0.85])
    
    if out_file is None:
        out_file = os.path.join(toma_path, "plot_paper_combined.png")
        
    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    plt.savefig(out_file, dpi=300)
    plt.close()
    
    logger(f"[Figura Multimodal] Generada exitosamente: {out_file}")
    return out_file

def procesar_sesion_completa(session_dir, out_central_dir=None, logger=print):
    """
    Procesa todas las mediciones dentro de una carpeta de sesión y genera su figura multimodal.
    """
    session_dir = os.path.abspath(session_dir)
    if not os.path.isdir(session_dir):
        raise ValueError(f"Directorio de sesión no existe: {session_dir}")
        
    tomas = sorted([
        d for d in os.listdir(session_dir) 
        if os.path.isdir(os.path.join(session_dir, d)) and not d.startswith('.')
    ])
    
    logger(f"Iniciando procesamiento de sesión: {session_dir} ({len(tomas)} mediciones)")
    generadas = []
    
    for i, t in enumerate(tomas):
        t_path = os.path.join(session_dir, t)
        meta_p = os.path.join(t_path, "canal_0", "metadata.json")
        if not os.path.exists(meta_p):
            meta_p = os.path.join(t_path, "metadata.json")
        if not os.path.exists(meta_p):
            continue
            
        logger(f"[Progreso {i+1}/{len(tomas)} - {((i+1)/len(tomas))*100:.1f}%] Procesando {t}...")
        try:
            # 1. Guardar en la carpeta individual de la toma como plot_paper_combined.png
            img_path = generar_figura_paper_multimodal(t_path, pulso_idx=1, logger=logger)
            generadas.append(img_path)
            
            # 2. Si se solicita directorio centralizado, guardar copia con nombre identificatorio
            if out_central_dir:
                os.makedirs(out_central_dir, exist_ok=True)
                dest = os.path.join(out_central_dir, f"figura_paper_{t}.png")
                import shutil
                shutil.copyfile(img_path, dest)
        except Exception as e:
            logger(f"Error procesando {t}: {e}")
            
    logger(f"Procesamiento culminado: {len(generadas)} figuras generadas.")
    return generadas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de figuras multimodales paper (4 paneles) para Ñandú EMG")
    parser.add_argument("--toma", help="Ruta a una toma individual")
    parser.add_argument("--sesion", help="Ruta o fecha de la sesión (ej. 2026-09-23)")
    parser.add_argument("--out", help="Ruta de salida específica")
    args = parser.parse_args()
    
    if args.toma:
        generar_figura_paper_multimodal(args.toma, out_file=args.out)
    elif args.sesion:
        target = args.sesion
        if not os.path.exists(target):
            repo_base = os.path.abspath(os.path.join(analysis_dir, "..", "base_de_datos_electrodos"))
            target = os.path.join(repo_base, args.sesion)
        procesar_sesion_completa(target, out_central_dir=args.out)
    else:
        print("Uso: python generador_figura_multimodal.py --sesion 2026-09-23")
