#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Cinemática Facial
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generador de Patrones de Activación Cinemáticos y PCA para 6 Canales
#              con pipeline idéntico a generador_pca_cinetico.py:
#              - Master Clock Acústico dinámico (t_wav_s - t_daq_s)
#              - Suavizado temporal continuo (~150 ms)
#              - Reposo global por percentiles (sin clamping local a cero)
#              - Purga por Isolation Forest (8%)
# ==============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.ndimage
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

COLORES_VOCALES = {
    "a": "#E63946",
    "e": "#1F77B4",
    "i": "#2CA02C",
    "o": "#9D4EDD",
    "u": "#E7A61A"
}
VOCALES_CANONICAS = ["a", "e", "i", "o", "u"]

def get_interpulse_noise(segmento, initial_noise=0.0):
    if len(segmento) < 5:
        return float(initial_noise)
    abs_vals = np.abs(segmento)
    q1 = np.percentile(abs_vals, 25)
    q3 = np.percentile(abs_vals, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid = abs_vals[abs_vals <= upper_bound]
    if len(valid) < 3:
        valid = abs_vals
    return float(np.mean(valid))

def calcular_envolvente_rms(audio_raw, fs=44100, rms_ms=90):
    n_samples = max(3, int((rms_ms / 1000.0) * fs))
    audio_ac = audio_raw - np.median(audio_raw)
    env_rms = np.sqrt(np.maximum(0.0, scipy.ndimage.uniform_filter1d(audio_ac**2, size=n_samples)))
    return env_rms

def procesar_sesion_6canales(session_dir, rms_ms=90, target_len=25, win_cinem_frames=3,
                             pre_s=0.5, post_s=0.5, outlier_contamination=0.08, min_snr=1.5):
    path_pulsos = os.path.join(session_dir, "pulsos_metronomo.csv")
    path_tray = os.path.join(session_dir, "trayectoria_puntos.csv")
    path_audio = os.path.join(session_dir, "grabacion.csv")
    
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
    
    # Offset dinámico de audio respecto al DAQ
    picos_mic_daq = {}
    vocales_dict = {}
    offsets_mic_daq = {}
    
    for idx, row in df_princ.iterrows():
        p_num = int(row["idx"])
        t_start = float(row["t_wav_s"])
        dur = float(row["intervalo_s"])
        palabra = str(row["palabra"]).lower().strip()
        
        offset_audio_daq = (float(row["t_wav_s"]) - float(row["t_daq_s"])) if (pd.notna(row.get("t_daq_s")) and pd.notna(row.get("t_wav_s"))) else 5.200
        offsets_mic_daq[p_num] = offset_audio_daq
        
        if t_start + 0.2 >= t_audio[-1]:
            break
            
        t_end = min(t_start + dur, t_audio[-1])
        mask = (t_audio >= t_start + 0.15) & (t_audio < t_end)
        if not np.any(mask):
            continue
            
        sub_env = env_rms[mask]
        sub_t = t_audio[mask]
        i_max = np.argmax(sub_env)
        t_mic_wav = sub_t[i_max]
        t_mic_daq = t_mic_wav - offset_audio_daq
        
        picos_mic_daq[p_num] = t_mic_daq
        vocales_dict[p_num] = palabra

    # Calcular distancias en píxeles si faltan o directamente de landmarks
    # Escala métrica desde DIP (distancia interpupilar = 63 mm canónica)
    # Si ya vienen en mm o px en trayectoria_puntos:
    # 1. Apertura Mandibular (labio superior a labio inferior)
    # 2. Ancho Comisuras (comisura izq a comisura der)
    # 3. Retracción Mejilla (promedio comisura-mejilla izq/der)
    # 4. Ascenso Malar (promedio comisura-malar izq/der)
    # 5. Compresión Maseterina (promedio comisura-masetero izq/der)
    
    # Calcular distancias euclídeas crudas si existen las columnas de puntos
    def dist_2d(p1_x, p1_y, p2_x, p2_y):
        return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
    
    # Si tenemos apertura_vertical_mm y ancho_comisuras_mm, usamos su escala
    escala = 1.0
    if "apertura_vertical_mm" in df_tray.columns and "comisura_izq_x_px" in df_tray.columns:
        dx_com_px = np.abs(df_tray["comisura_der_x_px"].values - df_tray["comisura_izq_x_px"].values)
        com_mm = df_tray["ancho_comisuras_mm"].values
        valid_m = (dx_com_px > 10) & (com_mm > 10)
        if np.any(valid_m):
            escala = float(np.median(com_mm[valid_m] / dx_com_px[valid_m]))
            
    # Calcular las 3 series de mejillas/mejillares en mm
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
    
    df_tray["d_mejilla_mm"] = d_mejilla_px * escala
    df_tray["d_malar_mm"] = d_malar_px * escala
    df_tray["d_masetero_mm"] = d_masetero_px * escala
    
    # Suavizado cinemático continuo (~150 ms) idéntico al generador
    df_tray["ap_smooth"] = scipy.ndimage.uniform_filter1d(df_tray["apertura_vertical_mm"].values, size=win_cinem_frames)
    df_tray["com_smooth"] = scipy.ndimage.uniform_filter1d(df_tray["ancho_comisuras_mm"].values, size=win_cinem_frames)
    df_tray["mej_smooth"] = scipy.ndimage.uniform_filter1d(df_tray["d_mejilla_mm"].values, size=win_cinem_frames)
    df_tray["mal_smooth"] = scipy.ndimage.uniform_filter1d(df_tray["d_malar_mm"].values, size=win_cinem_frames)
    df_tray["mas_smooth"] = scipy.ndimage.uniform_filter1d(df_tray["d_masetero_mm"].values, size=win_cinem_frames)
    
    # Baselines globales de reposo por percentil
    base_ap = np.percentile(df_tray["ap_smooth"].dropna(), 10)
    base_com = np.percentile(df_tray["com_smooth"].dropna(), 20)
    base_mej = np.percentile(df_tray["mej_smooth"].dropna(), 20)
    base_mal = np.percentile(df_tray["mal_smooth"].dropna(), 20)
    base_mas = np.percentile(df_tray["mas_smooth"].dropna(), 20)
    
    t_grid = np.linspace(-pre_s, post_s, target_len)
    t_curvas = np.linspace(-0.6, 0.8, 71)
    
    X_crudo = []
    Y_crudo = []
    pulsos_validos = []
    snr_list = []
    curvas_dict_crudo = []
    
    for p_num, t_mic in picos_mic_daq.items():
        group = df_tray[df_tray["pulso"] == p_num]
        if len(group) < 5:
            continue
            
        t_s = group["t_s"].values
        t_rel = t_s - t_mic
        
        # 6 Canales Cinemáticos Bio-Inspirados
        # 1. Apertura Mandibular (Digástrico)
        c1 = group["ap_smooth"].values - base_ap
        
        # 2. Tracción de Sonrisa (Cigomático)
        com_delta = group["com_smooth"].values - base_com
        c2 = np.maximum(0.0, com_delta)
        
        # 3. Constricción Esfinteriana (Orbicular)
        c3 = np.maximum(0.0, -com_delta)
        
        # 4. Retracción de Mejilla (Buccinador / Risorio): reducción de distancia comisura-mejilla
        c4 = np.maximum(0.0, -(group["mej_smooth"].values - base_mej))
        
        # 5. Ascenso Malar (Elevador del labio / Cigomático menor): acortamiento comisura-malar
        c5 = np.maximum(0.0, -(group["mal_smooth"].values - base_mal))
        
        # 6. Compresión Maseterina (Masetero): acortamiento comisura-masetero
        c6 = np.maximum(0.0, -(group["mas_smooth"].values - base_mas))
        
        # Interpolación en la grilla para features de clasificación
        c1_vec = np.interp(t_grid, t_rel, c1, left=0.0, right=0.0)
        c2_vec = np.interp(t_grid, t_rel, c2, left=0.0, right=0.0)
        c3_vec = np.interp(t_grid, t_rel, c3, left=0.0, right=0.0)
        c4_vec = np.interp(t_grid, t_rel, c4, left=0.0, right=0.0)
        c5_vec = np.interp(t_grid, t_rel, c5, left=0.0, right=0.0)
        c6_vec = np.interp(t_grid, t_rel, c6, left=0.0, right=0.0)
        
        # Interpolación para visualización de patrones
        c1_fine = np.interp(t_curvas, t_rel, c1, left=0.0, right=0.0)
        c2_fine = np.interp(t_curvas, t_rel, c2, left=0.0, right=0.0)
        c3_fine = np.interp(t_curvas, t_rel, c3, left=0.0, right=0.0)
        c4_fine = np.interp(t_curvas, t_rel, c4, left=0.0, right=0.0)
        c5_fine = np.interp(t_curvas, t_rel, c5, left=0.0, right=0.0)
        c6_fine = np.interp(t_curvas, t_rel, c6, left=0.0, right=0.0)
        
        # Audio de referencia
        offset_p = offsets_mic_daq.get(p_num, 5.200)
        t_mic_wav = t_mic + offset_p
        mask_aud = (t_audio >= t_mic_wav - 0.6) & (t_audio <= t_mic_wav + 0.8)
        sub_t_aud = t_audio[mask_aud] - t_mic_wav
        sub_env_aud = env_norm[mask_aud]
        mic_fine = np.interp(t_curvas, sub_t_aud, sub_env_aud, left=0.0, right=0.0) if len(sub_t_aud) > 0 else np.zeros_like(t_curvas)
        
        # Ruido y SNR
        ruido_ap = get_interpulse_noise(c1_vec[:3], 0.1)
        pico_ap = np.max(c1_vec)
        snr_pulso = pico_ap / (ruido_ap + 1e-4)
        
        vec_concat = np.concatenate([c1_vec, c2_vec, c3_vec, c4_vec, c5_vec, c6_vec])
        
        X_crudo.append(vec_concat)
        Y_crudo.append(vocales_dict[p_num])
        pulsos_validos.append(p_num)
        snr_list.append(snr_pulso)
        curvas_dict_crudo.append({
            "c1": c1_fine, "c2": c2_fine, "c3": c3_fine,
            "c4": c4_fine, "c5": c5_fine, "c6": c6_fine,
            "mic": mic_fine
        })
        
    X_crudo = np.array(X_crudo)
    Y_crudo = np.array(Y_crudo)
    snr_arr = np.array(snr_list)
    pulsos_arr = np.array(pulsos_validos)
    
    # Purga por Isolation Forest (8%)
    X_clean = []
    Y_clean = []
    pulsos_clean = []
    curvas_clean = {v: {"c1": [], "c2": [], "c3": [], "c4": [], "c5": [], "c6": [], "mic": []} for v in VOCALES_CANONICAS}
    
    for v in np.unique(Y_crudo):
        mask_v = (Y_crudo == v)
        indices_v = np.where(mask_v)[0]
        X_v = X_crudo[mask_v]
        p_v = pulsos_arr[mask_v]
        snr_v = snr_arr[mask_v]
        
        mask_snr = snr_v >= min_snr
        X_v_snr = X_v[mask_snr]
        p_v_snr = p_v[mask_snr]
        indices_v_snr = indices_v[mask_snr]
        
        if len(X_v_snr) > 5 and outlier_contamination > 0:
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            inliers = iso.fit_predict(X_v_snr)
            for i, ok in enumerate(inliers):
                if ok == 1:
                    orig_idx = indices_v_snr[i]
                    X_clean.append(X_v_snr[i])
                    Y_clean.append(v)
                    pulsos_clean.append(p_v_snr[i])
                    if v in curvas_clean:
                        for k in ["c1", "c2", "c3", "c4", "c5", "c6", "mic"]:
                            curvas_clean[v][k].append(curvas_dict_crudo[orig_idx][k])
        else:
            for i in range(len(X_v_snr)):
                orig_idx = indices_v_snr[i]
                X_clean.append(X_v_snr[i])
                Y_clean.append(v)
                pulsos_clean.append(p_v_snr[i])
                if v in curvas_clean:
                    for k in ["c1", "c2", "c3", "c4", "c5", "c6", "mic"]:
                        curvas_clean[v][k].append(curvas_dict_crudo[orig_idx][k])
                        
    return np.array(X_clean), np.array(Y_clean), np.array(pulsos_clean), curvas_clean, t_curvas

def renderizar_patrones_6canales(curvas_dict, t_comun, rms_ms, output_png):
    nombres_canales = [
        ("Apertura Mandibular: Digástrico", "c1", "mm"),
        ("Tracción de Sonrisa: Cigomático Mayor", "c2", "mm"),
        ("Constricción Esfinteriana: Orbicular", "c3", "mm"),
        ("Retracción Bucal: Buccinador - Risorio", "c4", "mm"),
        ("Ascenso Malar: Cigomático Menor - Elevador", "c5", "mm"),
        ("Compresión Maseterina: Masetero", "c6", "mm"),
        (f"Envolvente RMS {rms_ms} ms: Master Acústico", "mic", "u.a.")
    ]
    
    fig, axes = plt.subplots(7, 5, figsize=(22, 22), sharex=True)
    plt.subplots_adjust(top=0.94, bottom=0.04, left=0.07, right=0.98, hspace=0.28, wspace=0.22)
    
    fig.suptitle(f"Patrones Cinemáticos de Activación: 6 Canales Fisiológicos Sincronizados con Micrófono RMS {rms_ms} ms (t=0 ms)", 
                 fontsize=16, fontweight="bold", color="#111827")
                 
    for row_idx, (nombre, clave, unidad) in enumerate(nombres_canales):
        todos = []
        for v in VOCALES_CANONICAS:
            arr = curvas_dict[v][clave]
            if len(arr) > 0:
                todos.extend(np.array(arr).flatten())
        y_max = np.percentile(todos, 99.5) + (1.2 if unidad == "mm" else 0.1) if len(todos) > 0 else 5.0
        
        for col_idx, v in enumerate(VOCALES_CANONICAS):
            ax = axes[row_idx, col_idx]
            curvas = np.array(curvas_dict[v][clave])
            color = COLORES_VOCALES[v]
            
            for c in curvas:
                ax.plot(t_comun * 1000, c, color=color, alpha=0.15, linewidth=0.8)
                
            media = np.mean(curvas, axis=0) if len(curvas) > 0 else np.zeros_like(t_comun)
            std = np.std(curvas, axis=0) if len(curvas) > 0 else np.zeros_like(t_comun)
            
            ax.plot(t_comun * 1000, media, color=color, linewidth=2.5)
            ax.fill_between(t_comun * 1000, media - std, media + std, color=color, alpha=0.22)
            
            ax.axvline(0, color="#E63946" if row_idx == 6 else "#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.axhline(0, color="#94a3b8", linestyle=":", linewidth=0.8)
            
            if row_idx < 6:
                pico_val = np.max(media)
                if pico_val > 0.8:
                    t_pico = t_comun[np.argmax(media)] * 1000
                    ax.text(0.95, 0.88, f"Pico: {t_pico:+.0f} ms\n({pico_val:.1f} mm)", 
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=9, fontweight="bold", color=color,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffffff", edgecolor=color, alpha=0.8))
                            
            ax.set_ylim(-0.5 if unidad == "mm" else 0.0, y_max)
            ax.grid(True, linestyle="--", alpha=0.3)
            
            if row_idx == 0:
                ax.set_title(f"Vocal /{v.upper()}/ (N={len(curvas)})", fontsize=14, fontweight="bold", color=color)
            if col_idx == 0:
                ax.set_ylabel(f"{nombre}\n({unidad})", fontsize=9.5, fontweight="bold")
            if row_idx == 6:
                ax.set_xlabel("Tiempo relativo al Micrófono (ms)", fontsize=10)
                
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()
    print(f"[OK] Patrones de activación de 6 canales guardados en: {output_png}")

def ejecutar_pca_6ch(X, y):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    var = pca.explained_variance_ratio_ * 100
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y)
    preds_lda = lda.predict(X_pca)
    acc_lda = np.mean(preds_lda == y) * 100.0
    
    acc_por_vocal = {}
    for v in VOCALES_CANONICAS:
        mask_v = (y == v)
        acc_por_vocal[v] = np.mean(preds_lda[mask_v] == v) * 100.0 if np.any(mask_v) else 0.0
        
    kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_pca)
    vocales_u = sorted(list(set(y)))
    y_idx = np.array([vocales_u.index(v) for v in y])
    mat_conf = np.zeros((len(vocales_u), len(vocales_u)))
    for c, r in zip(clusters, y_idx):
        mat_conf[c, r] += 1
    row_ind, col_ind = linear_sum_assignment(-mat_conf)
    acc_kmeans = (mat_conf[row_ind, col_ind].sum() / len(y)) * 100.0
    
    return X_pca, var, acc_lda, acc_por_vocal, acc_kmeans

def main():
    parser = argparse.ArgumentParser(description="Patrones y PCA de 6 Canales Cinemáticos - Ñandú LSD")
    parser.add_argument("--session_dir", type=str, default="EMG_desarrollo/base_de_datos_electrodos/2026-11-08/SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02")
    parser.add_argument("--rms_ms", type=int, default=90)
    parser.add_argument("--contamination", type=float, default=0.08)
    parser.add_argument("--output_patrones_png", type=str, default="/home/santiago/.gemini/antigravity/brain/d39cbfe4-07c6-46ea-aca5-430bff873705/plot_patrones_6canales_oficial.png")
    parser.add_argument("--output_pca_png", type=str, default="/home/santiago/.gemini/antigravity/brain/d39cbfe4-07c6-46ea-aca5-430bff873705/plot_pca_6canales_oficial.png")
    args = parser.parse_args()
    
    print(f"=== Procesando 6 Canales Cinemáticos (Pipeline Idéntico a generador_pca_cinetico) ===")
    X, y, pulsos, curvas_dict, t_curvas = procesar_sesion_6canales(
        args.session_dir,
        rms_ms=args.rms_ms,
        outlier_contamination=args.contamination
    )
    print(f"Matriz procesada: {X.shape[0]} pulsos limpios x {X.shape[1]} features (6 canales x 25 pts)")
    
    renderizar_patrones_6canales(curvas_dict, t_curvas, args.rms_ms, args.output_patrones_png)
    
    X_pca, var, acc_lda, acc_por_vocal, acc_kmeans = ejecutar_pca_6ch(X, y)
    print(f"\n--- Resultados Oficiales 6 Canales ---")
    print(f"Varianza 3D: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}% (Total: {np.sum(var):.1f}%)")
    print(f"Precisión LDA Global en PCA 3D: {acc_lda:.1f}%")
    print(f"Clustering No Supervisado K-Means: {acc_kmeans:.1f}%")
    for v in VOCALES_CANONICAS:
        print(f"  /{v}/: {acc_por_vocal[v]:.1f}%")

if __name__ == "__main__":
    main()
