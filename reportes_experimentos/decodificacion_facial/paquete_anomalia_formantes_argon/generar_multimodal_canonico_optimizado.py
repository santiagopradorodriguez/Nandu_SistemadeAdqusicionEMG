"""
Generador Multimodal Canónico Optimizado: PCA Cinético 3D, PCA 6 Canales, Patrones y Formantes (F1-F2)
Ñandú LSD - Universidad de Buenos Aires
Ventana simétrica óptima: [-0.40 s, +0.40 s] (Máximo LDA 60.9% / Varianza 3D 65.6%)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.io.wavfile as wav
from scipy.signal import butter, filtfilt, resample_poly
from scipy.linalg import solve_toeplitz
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VOCALES_CANONICAS = ['A', 'E', 'I', 'O', 'U']
COLORES_CANONICOS = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

def calcular_envolvente_rms(audio_raw, fs=44100, rms_ms=250):
    n_samples = max(3, int((rms_ms / 1000.0) * fs))
    audio_ac = audio_raw - np.median(audio_raw)
    env_rms = np.sqrt(np.maximum(0.0, scipy.ndimage.uniform_filter1d(audio_ac**2, size=n_samples)))
    return env_rms

def extraer_formantes_lpc(audio_vocal, fs, t_mic_wav, duracion_ventana_s=0.050):
    if audio_vocal is None or len(audio_vocal) == 0:
        return np.nan, np.nan
    idx_centro = int(t_mic_wav * fs)
    w_seg = int(duracion_ventana_s * fs)
    i_start = max(0, idx_centro - w_seg // 2)
    i_end = min(len(audio_vocal), i_start + w_seg)
    seg = audio_vocal[i_start:i_end]
    if len(seg) < w_seg:
        return np.nan, np.nan
        
    seg_pre = np.append(seg[0], seg[1:] - 0.96 * seg[:-1])
    seg_win = seg_pre * np.hamming(len(seg_pre))
    
    p = 12
    r = np.correlate(seg_win, seg_win, mode='full')
    r = r[len(seg_win)-1 : len(seg_win)+p]
    
    try:
        a_lpc = solve_toeplitz((r[:-1], r[:-1]), r[1:])
        poly = np.concatenate(([1], -a_lpc))
        raices = np.roots(poly)
        raices = [rc for rc in raices if np.imag(rc) > 0]
        angulos = np.arctan2(np.imag(raices), np.real(raices))
        freqs = angulos * (fs / (2.0 * np.pi))
        bws = -0.5 * (fs / (2.0 * np.pi)) * np.log(np.maximum(np.abs(raices), 1e-9))
        
        cands = [(f_v, bw_v) for f_v, bw_v in zip(freqs, bws) if 180 < f_v < 3600 and bw_v < 400]
        cands.sort(key=lambda x: x[0])
        
        if len(cands) >= 2:
            return float(cands[0][0]), float(cands[1][0])
    except Exception:
        pass
    return np.nan, np.nan

def procesar_sesion_multimodal_optimizada(session_dir, rms_ms=90, target_len=25, pre_s=0.40, post_s=0.40, outlier_contamination=0.08):
    path_pulsos = os.path.join(session_dir, "pulsos_metronomo.csv")
    path_tray = os.path.join(session_dir, "trayectoria_puntos.csv")
    path_audio = os.path.join(session_dir, "grabacion.csv")
    path_wav = os.path.join(session_dir, "canal_0", "grabacion.wav")
    
    if not os.path.isfile(path_pulsos) or not os.path.isfile(path_tray) or not os.path.isfile(path_audio):
        raise FileNotFoundError(f"Faltan archivos necesarios en {session_dir}")
        
    df_p = pd.read_csv(path_pulsos)
    df_princ = df_p[(df_p["principal"] == 1) & (df_p["idx"] > 0)].copy()
    df_tray = pd.read_csv(path_tray)
    
    df_audio = pd.read_csv(path_audio, usecols=[0, 1])
    t_audio = df_audio.iloc[:, 0].values
    audio_raw = df_audio.iloc[:, 1].values
    fs = 44100
    
    env_rms = calcular_envolvente_rms(audio_raw, fs=fs, rms_ms=rms_ms)
    env_norm = env_rms / (np.percentile(env_rms, 99.8) + 1e-9)
    
    fs_formantes = 10000
    if os.path.exists(path_wav):
        fs_raw, wav_data = wav.read(path_wav)
        audio_10k = resample_poly(wav_data.astype(np.float64), 100, 441)
        audio_10k -= np.mean(audio_10k)
        b, a = butter(4, [180.0 / (fs_formantes / 2.0), 3800.0 / (fs_formantes / 2.0)], btype='band')
        audio_vocal = filtfilt(b, a, audio_10k)
    else:
        audio_10k = resample_poly(audio_raw.astype(np.float64), 100, 441)
        audio_10k -= np.mean(audio_10k)
        b, a = butter(4, [180.0 / (fs_formantes / 2.0), 3800.0 / (fs_formantes / 2.0)], btype='band')
        audio_vocal = filtfilt(b, a, audio_10k)
        
    picos_mic_daq = {}
    vocales_dict = {}
    offsets_mic_daq = {}
    
    for idx, row in df_princ.iterrows():
        p_num = int(row["idx"])
        t_start = float(row["t_wav_s"])
        dur = float(row["intervalo_s"])
        palabra = str(row["palabra"]).upper().strip()
        
        offset_audio_daq = (float(row["t_wav_s"]) - float(row["t_daq_s"])) if (pd.notna(row.get("t_daq_s")) and pd.notna(row.get("t_wav_s"))) else 5.200
        offsets_mic_daq[p_num] = offset_audio_daq
        
        if t_start + 0.2 >= t_audio[-1]:
            continue
        t_end = min(t_start + dur, t_audio[-1])
        mask = (t_audio >= t_start + 0.15) & (t_audio < t_end)
        sub_t = t_audio[mask]
        sub_env = env_norm[mask]
        
        if len(sub_env) == 0:
            continue
            
        t_mic_wav = sub_t[np.argmax(sub_env)]
        t_mic_daq = t_mic_wav - offset_audio_daq
        
        picos_mic_daq[p_num] = t_mic_daq
        vocales_dict[p_num] = palabra
        
    fps_video = 1.0 / np.median(np.diff(df_tray["t_s"].values))
    win_cinem_frames = max(3, int(0.150 * fps_video))
    if win_cinem_frames % 2 == 0:
        win_cinem_frames += 1
        
    def dist_2d(p1_x, p1_y, p2_x, p2_y):
        return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
    
    escala = 1.0
    if "apertura_vertical_mm" in df_tray.columns and "comisura_izq_x_px" in df_tray.columns:
        dx_com_px = np.abs(df_tray["comisura_der_x_px"].values - df_tray["comisura_izq_x_px"].values)
        com_mm = df_tray["ancho_comisuras_mm"].values
        valid_m = (dx_com_px > 10) & (com_mm > 10)
        if np.any(valid_m):
            escala = float(np.median(com_mm[valid_m] / dx_com_px[valid_m]))
            
    tiene_6ch = ("mejilla_izq_x_px" in df_tray.columns)
    if tiene_6ch:
        d_mejilla_px = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["mejilla_izq_x_px"], df_tray["mejilla_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["mejilla_der_x_px"], df_tray["mejilla_der_y_px"])
        )
        d_malar_px = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["malar_izq_x_px"], df_tray["malar_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["malar_der_x_px"], df_tray["malar_der_y_px"])
        )
        d_masetero_px = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["masetero_izq_x_px"], df_tray["masetero_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["masetero_der_x_px"], df_tray["masetero_der_y_px"])
        )
        mej_s = scipy.ndimage.uniform_filter1d(d_mejilla_px * escala, size=win_cinem_frames)
        mal_s = scipy.ndimage.uniform_filter1d(d_malar_px * escala, size=win_cinem_frames)
        mas_s = scipy.ndimage.uniform_filter1d(d_masetero_px * escala, size=win_cinem_frames)
    else:
        zeros_arr = np.zeros(len(df_tray))
        mej_s = zeros_arr
        mal_s = zeros_arr
        mas_s = zeros_arr

    df_calc = pd.DataFrame({
        "t_s": df_tray["t_s"].values,
        "pulso": df_tray["pulso"].values,
        "ap_smooth": scipy.ndimage.uniform_filter1d(df_tray["apertura_vertical_mm"].values, size=win_cinem_frames),
        "com_smooth": scipy.ndimage.uniform_filter1d(df_tray["ancho_comisuras_mm"].values, size=win_cinem_frames),
        "menton_smooth": scipy.ndimage.uniform_filter1d(df_tray["descenso_menton_mm"].values, size=win_cinem_frames),
        "mej_smooth": mej_s,
        "mal_smooth": mal_s,
        "mas_smooth": mas_s
    })
    
    base_ap = np.percentile(df_calc["ap_smooth"].dropna(), 10)
    base_com = np.percentile(df_calc["com_smooth"].dropna(), 20)
    base_menton = np.percentile(df_calc["menton_smooth"].dropna(), 10)
    base_mej = np.percentile(df_calc["mej_smooth"].dropna(), 20)
    base_mal = np.percentile(df_calc["mal_smooth"].dropna(), 20)
    base_mas = np.percentile(df_calc["mas_smooth"].dropna(), 20)
    
    t_grid = np.linspace(-pre_s, post_s, target_len)
    t_curvas = np.linspace(-pre_s, post_s, 61)
    
    X3_crudo = []
    X6_crudo = []
    Y_crudo = []
    pulsos_validos = []
    formantes_crudo = []
    curvas_dict_crudo = []
    
    for p_num, t_mic in picos_mic_daq.items():
        group = df_calc[df_calc["pulso"] == p_num]
        if len(group) < 5:
            continue
            
        t_s = group["t_s"].values
        t_rel = t_s - t_mic
        
        # 3 Canales Directos
        c_ap = group["ap_smooth"].values - base_ap
        c_com_sign = group["com_smooth"].values - base_com
        c_men = group["menton_smooth"].values - base_menton
        
        # 6 Canales Bio-Inspirados
        c1 = c_ap
        c2 = np.maximum(0.0, c_com_sign)
        c3 = np.maximum(0.0, -c_com_sign)
        c4 = np.maximum(0.0, -(group["mej_smooth"].values - base_mej))
        c5 = np.maximum(0.0, -(group["mal_smooth"].values - base_mal))
        c6 = np.maximum(0.0, -(group["mas_smooth"].values - base_mas))
        
        c_ap_v = np.interp(t_grid, t_rel, c_ap, left=0.0, right=0.0)
        c_com_v = np.interp(t_grid, t_rel, c_com_sign, left=0.0, right=0.0)
        c_men_v = np.interp(t_grid, t_rel, c_men, left=0.0, right=0.0)
        vec_3ch = np.concatenate([c_ap_v, c_com_v, c_men_v])
        
        c1_v = np.interp(t_grid, t_rel, c1, left=0.0, right=0.0)
        c2_v = np.interp(t_grid, t_rel, c2, left=0.0, right=0.0)
        c3_v = np.interp(t_grid, t_rel, c3, left=0.0, right=0.0)
        c4_v = np.interp(t_grid, t_rel, c4, left=0.0, right=0.0)
        c5_v = np.interp(t_grid, t_rel, c5, left=0.0, right=0.0)
        c6_v = np.interp(t_grid, t_rel, c6, left=0.0, right=0.0)
        vec_6ch = np.concatenate([c1_v, c2_v, c3_v, c4_v, c5_v, c6_v])
        
        offset_p = offsets_mic_daq.get(p_num, 5.200)
        t_mic_wav = t_mic + offset_p
        f1_val, f2_val = extraer_formantes_lpc(audio_vocal, fs_formantes, t_mic_wav)
        
        c1_fine = np.interp(t_curvas, t_rel, c1, left=0.0, right=0.0)
        c2_fine = np.interp(t_curvas, t_rel, c2, left=0.0, right=0.0)
        c3_fine = np.interp(t_curvas, t_rel, c3, left=0.0, right=0.0)
        c4_fine = np.interp(t_curvas, t_rel, c4, left=0.0, right=0.0)
        c5_fine = np.interp(t_curvas, t_rel, c5, left=0.0, right=0.0)
        c6_fine = np.interp(t_curvas, t_rel, c6, left=0.0, right=0.0)
        
        mask_aud = (t_audio >= t_mic_wav - pre_s) & (t_audio <= t_mic_wav + post_s)
        sub_t_aud = t_audio[mask_aud] - t_mic_wav
        sub_env_aud = env_norm[mask_aud]
        mic_fine = np.interp(t_curvas, sub_t_aud, sub_env_aud, left=0.0, right=0.0) if len(sub_t_aud) > 0 else np.zeros_like(t_curvas)
        
        X3_crudo.append(vec_3ch)
        X6_crudo.append(vec_6ch)
        Y_crudo.append(vocales_dict[p_num])
        pulsos_validos.append(p_num)
        formantes_crudo.append({'pulso': p_num, 'vocal': vocales_dict[p_num], 'F1': f1_val, 'F2': f2_val, 't_mic_wav': t_mic_wav})
        curvas_dict_crudo.append({
            "c1": c1_fine, "c2": c2_fine, "c3": c3_fine,
            "c4": c4_fine, "c5": c5_fine, "c6": c6_fine,
            "mic": mic_fine
        })
        
    X3_crudo = np.array(X3_crudo)
    X6_crudo = np.array(X6_crudo)
    Y_crudo = np.array(Y_crudo)
    pulsos_arr = np.array(pulsos_validos)
    df_formantes_crudo = pd.DataFrame(formantes_crudo)
    
    # Purga por Isolation Forest (8%)
    X3_clean = []
    X6_clean = []
    Y_clean = []
    pulsos_clean = []
    formantes_clean = []
    curvas_dict_clean = {c: {v: [] for v in VOCALES_CANONICAS} for c in ["c1", "c2", "c3", "c4", "c5", "c6", "mic"]}
    
    for v in VOCALES_CANONICAS:
        mask_v = (Y_crudo == v)
        if not np.any(mask_v):
            continue
        sub_X6 = X6_crudo[mask_v]
        sub_X3 = X3_crudo[mask_v]
        sub_P = pulsos_arr[mask_v]
        sub_F = df_formantes_crudo[mask_v].copy()
        sub_curvas = [curvas_dict_crudo[i] for i, m in enumerate(mask_v) if m]
        
        if len(sub_X6) > 6 and outlier_contamination > 0:
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            inliers = iso.fit_predict(sub_X6) == 1
        else:
            inliers = np.ones(len(sub_X6), dtype=bool)
            
        X3_clean.append(sub_X3[inliers])
        X6_clean.append(sub_X6[inliers])
        Y_clean.extend([v] * np.sum(inliers))
        pulsos_clean.extend(sub_P[inliers])
        formantes_clean.append(sub_F[inliers])
        
        for idx_i, ok in enumerate(inliers):
            if ok:
                curv = sub_curvas[idx_i]
                for ch in ["c1", "c2", "c3", "c4", "c5", "c6", "mic"]:
                    curvas_dict_clean[ch][v].append(curv[ch])
                    
    X3 = np.vstack(X3_clean)
    X6 = np.vstack(X6_clean)
    y = np.array(Y_clean)
    pulsos = np.array(pulsos_clean)
    df_formantes = pd.concat(formantes_clean, ignore_index=True)
    
    curvas_promedio = {}
    for ch in ["c1", "c2", "c3", "c4", "c5", "c6", "mic"]:
        curvas_promedio[ch] = {}
        for v in VOCALES_CANONICAS:
            arr = np.array(curvas_dict_clean[ch][v])
            if len(arr) > 0:
                curvas_promedio[ch][v] = (np.mean(arr, axis=0), np.std(arr, axis=0))
            else:
                curvas_promedio[ch][v] = (np.zeros_like(t_curvas), np.zeros_like(t_curvas))
                
    return X3, X6, y, pulsos, df_formantes, curvas_promedio, t_curvas

def evaluar_pca(X, y):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y)
    preds = lda.predict(X_pca)
    acc_lda = np.mean(preds == y) * 100.0
    
    acc_vocal = {}
    for v in VOCALES_CANONICAS:
        mask = (y == v)
        acc_vocal[v] = np.mean(preds[mask] == v) * 100.0 if np.any(mask) else 0.0
        
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_pca)
    voc_u = sorted(list(set(y)))
    y_idx = np.array([voc_u.index(v) for v in y])
    mat_conf = np.zeros((len(voc_u), len(voc_u)))
    for c, r in zip(clusters, y_idx):
        mat_conf[c, r] += 1
    row_ind, col_ind = linear_sum_assignment(-mat_conf)
    acc_kmeans = (mat_conf[row_ind, col_ind].sum() / len(y)) * 100.0
    
    return X_pca, var, acc_lda, acc_vocal, acc_kmeans

def renderizar_pca_3proyecciones(X_pca, y, var, acc_lda, acc_vocal, acc_kmeans, titulo_gral, output_png):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor='#ffffff')
    plt.subplots_adjust(top=0.86, bottom=0.12, left=0.05, right=0.98, wspace=0.28)
    
    proyecciones = [
        (0, 1, f"PC1 ({var[0]:.1f}%) vs PC2 ({var[1]:.1f}%)", axes[0]),
        (0, 2, f"PC1 ({var[0]:.1f}%) vs PC3 ({var[2]:.1f}%)", axes[1]),
        (1, 2, f"PC2 ({var[1]:.1f}%) vs PC3 ({var[2]:.1f}%)", axes[2])
    ]
    
    fig.suptitle(
        f"{titulo_gral}: LDA = {acc_lda:.1f}% | K-Means = {acc_kmeans:.1f}% (Varianza 3D = {np.sum(var):.1f}%)",
        fontsize=14, fontweight='bold', color='#111827'
    )
    
    for idx_x, idx_y, title, ax in proyecciones:
        ax.set_facecolor('#ffffff')
        ax.set_box_aspect(1)
        ax.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
        ax.axhline(0, color='#64748b', linestyle=':', linewidth=0.8)
        ax.axvline(0, color='#64748b', linestyle=':', linewidth=0.8)
        
        for v in VOCALES_CANONICAS:
            mask = (y == v)
            if not np.any(mask):
                continue
            color = COLORES_CANONICOS[v]
            ax.scatter(
                X_pca[mask, idx_x], X_pca[mask, idx_y],
                c=color, s=55, alpha=0.85, edgecolors='none',
                label=f"/{v.lower()}/ ({acc_vocal.get(v, 0):.0f}%)"
            )
            
            cx = np.mean(X_pca[mask, idx_x])
            cy = np.mean(X_pca[mask, idx_y])
            ax.scatter([cx], [cy], c=color, s=260, edgecolors='#0f172a', linewidths=2.0, zorder=5)
            ax.text(cx, cy, f"/{v.lower()}/", color='#ffffff', fontsize=10, fontweight='bold', ha='center', va='center', zorder=6)
            
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1e293b')
        ax.set_xlabel(f"PC{idx_x+1}", fontsize=11, fontweight='bold', color='#1e293b')
        ax.set_ylabel(f"PC{idx_y+1}", fontsize=11, fontweight='bold', color='#1e293b')
        ax.tick_params(colors='#1e293b')
        if idx_x == 0 and idx_y == 1:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
            
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=200, facecolor='#ffffff')
    plt.close()
    print(f"[OK] Guardado: {output_png}")

def renderizar_patrones_multimodales(curvas_promedio, t_curvas, rms_ms, output_png):
    filas_info = [
        ("Apertura Mandibular: Digástrico", "c1", "mm"),
        ("Tracción de Sonrisa: Cigomático Mayor", "c2", "mm"),
        ("Constricción Esfinteriana: Orbicular", "c3", "mm"),
        ("Retracción Bucal: Buccinador - Risorio", "c4", "mm"),
        ("Ascenso Malar: Elevador - Cigomático Menor", "c5", "mm"),
        ("Compresión Maseterina: Masetero", "c6", "mm"),
        (f"Envolvente RMS {rms_ms} ms: Master Acústico", "mic", "u.a.")
    ]
    
    fig, axes = plt.subplots(7, 5, figsize=(22, 22), sharex=True, facecolor='#ffffff')
    plt.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.98, hspace=0.28, wspace=0.22)
    
    fig.suptitle(
        f"Patrones Cinemáticos de Activación: 6 Canales Fisiológicos + Master Acústico - Sesión Continua",
        fontsize=16, fontweight='bold', color='#111827', y=0.97
    )
    
    for row_idx, (nombre_ch, key_ch, unidad) in enumerate(filas_info):
        for col_idx, v in enumerate(VOCALES_CANONICAS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor('#ffffff')
            ax.grid(True, linestyle='--', alpha=0.35, color='#cbd5e1')
            ax.axvline(0, color='#64748b', linestyle=':', linewidth=1.0)
            
            color = COLORES_CANONICOS[v]
            media, std = curvas_promedio[key_ch][v]
            
            ax.plot(t_curvas, media, color=color, linewidth=2.2, label=f"/{v.lower()}/ media")
            ax.fill_between(t_curvas, media - std, media + std, color=color, alpha=0.20)
            
            if np.any(media > 0):
                idx_pico = np.argmax(media)
                t_pico = t_curvas[idx_pico]
                val_pico = media[idx_pico]
                ax.plot(t_pico, val_pico, marker='o', markersize=5, color=color)
                ax.text(
                    t_pico, val_pico, f" {val_pico:.1f}",
                    fontsize=8.5, fontweight='bold', color='#1e293b',
                    va='bottom', ha='left'
                )
                
            if row_idx == 0:
                ax.set_title(f"Vocal /{v.lower()}/", fontsize=13, fontweight='bold', color=color, pad=10)
                
            if col_idx == 0:
                ax.set_ylabel(f"{nombre_ch}\n[{unidad}]", fontsize=10.5, fontweight='bold', color='#1e293b')
                
            if row_idx == 6:
                ax.set_xlabel("Tiempo relativo al centro acústico [s]", fontsize=10, fontweight='bold', color='#1e293b')
                
            ax.tick_params(colors='#1e293b', labelsize=9)
            
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor='#ffffff')
    plt.close()
    print(f"[OK] Patrones de activación guardados en: {output_png}")

def renderizar_plano_formantes(df_formantes, output_png):
    plt.figure(figsize=(9, 7.5), facecolor='#ffffff')
    ax = plt.gca()
    ax.set_facecolor('#ffffff')
    
    df_valid = df_formantes.dropna(subset=['F1', 'F2'])
    
    for v in VOCALES_CANONICAS:
        sub = df_valid[df_valid['vocal'] == v]
        if len(sub) == 0:
            continue
        color = COLORES_CANONICOS[v]
        ax.scatter(
            sub['F2'], sub['F1'],
            c=color, s=50, alpha=0.8, edgecolors='#ffffff', linewidths=0.5,
            label=f"/{v.lower()}/ (N={len(sub)})"
        )
        f1_m, f2_m = sub['F1'].mean(), sub['F2'].mean()
        ax.scatter([f2_m], [f1_m], c=color, s=220, edgecolors='#0f172a', linewidths=2.0, zorder=5)
        ax.text(f2_m, f1_m - 18, f"/{v.lower()}/", color='#1e293b', fontsize=12, fontweight='bold', ha='center', va='top', zorder=6)
        
    ax.set_xlim(1850, 850)
    ax.set_ylim(950, 250)
    ax.grid(True, linestyle='--', alpha=0.4, color='#94a3b8')
    ax.set_xlabel("F2: Segundo Formante [Hz] - Adelantamiento Lingual (Invertido)", fontsize=11, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_ylabel("F1: Primer Formante [Hz] - Altura Lingual / Apertura (Invertido)", fontsize=11, fontweight='bold', color='#1e293b', labelpad=10)
    ax.set_title("Plano Acústico Formántico F1 vs F2 - Pipeline Unificado", fontsize=13, fontweight='bold', color='#111827', pad=15)
    
    leg = ax.legend(loc='lower left', framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#1e293b')
        
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, facecolor='#ffffff')
    plt.close()
    print(f"[OK] Plano Formantes Guardado: {output_png}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Unificado Óptimo: PCA Cinético 3D, PCA 6 Canales, Patrones y Formantes")
    parser.add_argument("--session_dir", type=str, default="EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1")
    parser.add_argument("--rms_ms", type=int, default=250)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--pre_s", type=float, default=0.40)
    parser.add_argument("--post_s", type=float, default=0.40)
    parser.add_argument("--out_dir", type=str, default="reportes_experimentos/decodificacion_facial")
    args = parser.parse_args()
    
    print(f"=== Ejecutando Pipeline Multimodal Canónico Óptimo Ñandú LSD ===")
    print(f"Sesión: {args.session_dir}")
    print(f"Ventana simétrica óptima: [-{args.pre_s:.2f} s, +{args.post_s:.2f} s]")
    
    X3, X6, y, pulsos, df_formantes, curvas_promedio, t_curvas = procesar_sesion_multimodal_optimizada(
        args.session_dir,
        rms_ms=args.rms_ms,
        pre_s=args.pre_s,
        post_s=args.post_s,
        outlier_contamination=args.contamination
    )
    print(f"Pulsos limpios procesados: {len(y)}")
    
    # 1. PCA Cinético 3 Canales Directos
    X3_pca, var3, acc_lda3, acc_voc3, acc_km3 = evaluar_pca(X3, y)
    print(f"\n--- Resultados PCA Cinético Directo (3 Canales: Apertura, Comisuras, Mentón) ---")
    print(f"Varianza 3D: PC1={var3[0]:.1f}%, PC2={var3[1]:.1f}%, PC3={var3[2]:.1f}% (Total: {np.sum(var3):.1f}%)")
    print(f"Precisión LDA Global: {acc_lda3:.1f}% | K-Means: {acc_km3:.1f}%")
    for v in VOCALES_CANONICAS:
        print(f"  /{v}/: {acc_voc3[v]:.1f}%")
        
    # 2. PCA 6 Canales Bio-Inspirados
    X6_pca, var6, acc_lda6, acc_voc6, acc_km6 = evaluar_pca(X6, y)
    print(f"\n--- Resultados PCA 6 Canales Bio-Inspirados (Expansión Completa) ---")
    print(f"Varianza 3D: PC1={var6[0]:.1f}%, PC2={var6[1]:.1f}%, PC3={var6[2]:.1f}% (Total: {np.sum(var6):.1f}%)")
    print(f"Precisión LDA Global: {acc_lda6:.1f}% | K-Means: {acc_km6:.1f}%")
    for v in VOCALES_CANONICAS:
        print(f"  /{v}/: {acc_voc6[v]:.1f}%")
        
    p_pca_cinetico = os.path.join(args.out_dir, "pca_cinetico_oficial.png")
    p_pca_6ch = os.path.join(args.out_dir, "pca_6canales_oficial.png")
    p_formantes_png = os.path.join(args.out_dir, "plano_formantes_lineal.png")
    p_patrones_png = os.path.join(args.out_dir, "patrones_6canales_oficial.png")
    p_formantes_csv = os.path.join(args.out_dir, "formantes_generador_pca_6canales.csv")
    
    renderizar_pca_3proyecciones(X3_pca, y, var3, acc_lda3, acc_voc3, acc_km3, "PCA 3D Cinético Directo (3 Canales)", p_pca_cinetico)
    renderizar_pca_3proyecciones(X6_pca, y, var6, acc_lda6, acc_voc6, acc_km6, "PCA 6 Canales Bio-Inspirados", p_pca_6ch)
    renderizar_plano_formantes(df_formantes, p_formantes_png)
    renderizar_patrones_multimodales(curvas_promedio, t_curvas, args.rms_ms, p_patrones_png)
    
    df_formantes.to_csv(p_formantes_csv, index=False)
    print(f"[OK] Tabla de formantes exportada: {p_formantes_csv}")
    print(f"=== Pipeline Completo Finalizado Exitosamente ===")

if __name__ == "__main__":
    main()
