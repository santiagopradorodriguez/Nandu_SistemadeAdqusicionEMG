#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Diagnostico forense de anomalias y separabilidad 2026-10-23 vs 2026-10-20.
# ==============================================================================

import os
import sys
import json
import glob
import random
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt, iirnotch, resample, find_peaks
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

# Semilla universal obligatoria
random.seed(42)
np.random.seed(42)

REPO_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
DB_DIR = os.path.join(REPO_ROOT, "EMG_desarrollo", "base_de_datos_electrodos")
SESSION_DIR = os.path.join(DB_DIR, "2026-10-23")
OUT_DIR = os.path.join(REPO_ROOT, "reportes_experimentos", "diagnostico_forense_2026_10_23")
os.makedirs(OUT_DIR, exist_ok=True)


def eval_hungarian_acc(X_proj, Y, method="gmm"):
    vocales = sorted(list(set(Y)))
    n_clases = len(vocales)
    if n_clases < 2 or len(Y) < n_clases * 2:
        return 0.0, {}, 0.0
    y_true = np.array([vocales.index(v) for v in Y])
    if method == "gmm":
        model = GaussianMixture(n_components=n_clases, covariance_type="full", random_state=42, n_init=30, max_iter=300)
        y_pred = model.fit_predict(X_proj)
    elif method == "kmeans":
        model = KMeans(n_clusters=n_clases, random_state=42, n_init=30)
        y_pred = model.fit_predict(X_proj)
    elif method == "lda":
        model = LinearDiscriminantAnalysis()
        model.fit(X_proj, y_true)
        y_pred = model.predict(X_proj)
        per_v = {}
        for idx_v, v in enumerate(vocales):
            mask = (y_true == idx_v)
            per_v[v] = float(np.mean(y_pred[mask] == idx_v) * 100.0) if np.sum(mask) > 0 else 0.0
        sil = float(silhouette_score(X_proj, Y)) if len(set(Y)) > 1 else 0.0
        return float(np.mean(list(per_v.values()))), per_v, sil

    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    y_mapped = np.array([mapping.get(c, -1) for c in y_pred])
    per_v = {}
    for idx_v, v in enumerate(vocales):
        mask = (y_true == idx_v)
        per_v[v] = float(np.mean(y_mapped[mask] == idx_v) * 100.0) if np.sum(mask) > 0 else 0.0
    macro_acc = float(np.mean(list(per_v.values())))
    sil = float(silhouette_score(X_proj, Y)) if len(set(Y)) > 1 else 0.0
    return macro_acc, per_v, sil


def filter_emg(raw, sr, hp=20.0, lp=500.0, notch_q=2.0):
    sig = raw.astype(float) - np.median(raw)
    # Notch harmonics 50, 100, 150, 200, 250, 300
    for f0 in range(50, int(min(sr // 2 - 10, 450)), 50):
        b_n, a_n = iirnotch(f0, notch_q, sr)
        sig = filtfilt(b_n, a_n, sig)
    nyq = 0.5 * sr
    low = max(0.001, hp / nyq)
    high = min(0.99, lp / nyq)
    b_bp, a_bp = butter(4, [low, high], btype="band")
    sig = filtfilt(b_bp, a_bp, sig)
    return sig


def get_rms_env(sig, sr, smooth_ms=150):
    win = max(3, int(sr * (smooth_ms / 1000.0)))
    sq = sig ** 2
    kernel = np.ones(win) / float(win)
    return np.sqrt(np.maximum(np.convolve(sq, kernel, mode="same"), 0.0))


def get_iqr_noise(seg):
    if len(seg) < 5:
        return 0.0
    q25, q75 = np.percentile(seg, [25, 75])
    iqr = q75 - q25
    valid = seg[seg <= q75 + 1.5 * iqr]
    return float(np.mean(valid)) if len(valid) > 0 else float(np.mean(seg))


def audit_all_tomas():
    rows = []
    for entry in sorted(os.listdir(SESSION_DIR)):
        full_p = os.path.join(SESSION_DIR, entry)
        if not os.path.isdir(full_p) or "calibracion" in entry.lower() or "secuencia" in entry.lower():
            continue
        meta_p = os.path.join(full_p, "canal_0", "metadata.json")
        if not os.path.exists(meta_p):
            continue
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        vocal = entry.split("_")[0].upper()
        if vocal not in ["A", "E", "I", "O", "U"]:
            continue
        sr = int(meta.get("sample_rate", 2000))
        noise_sec = float(meta.get("noise_seconds", 3.0))
        bpm = float(meta.get("bpm", 50.0))
        muscles = meta.get("muscles", ["ch0", "ch1", "ch2", "mic"])
        if "pruebabelly" in entry.lower():
            exp = "Exp2_Belly"
        elif "pruebatricanal" in entry.lower():
            exp = "Exp3_Tricanal"
        elif "silabas" in entry.lower():
            exp = "Exp4_Silabas"
        else:
            exp = "Exp1_Masetero"

        ch_stats = {}
        for ch_i in range(4):
            wav_p = os.path.join(full_p, f"canal_{ch_i}", "grabacion.wav")
            ch_meta_p = os.path.join(full_p, f"canal_{ch_i}", "metadata.json")
            if not os.path.exists(wav_p):
                continue
            import soundfile as sf
            data_w, sr_w = sf.read(wav_p)
            if data_w.ndim > 1:
                data_w = data_w[:, 0]
            scale_f = 1.0
            if os.path.exists(ch_meta_p):
                try:
                    with open(ch_meta_p, "r", encoding="utf-8") as fm:
                        cm = json.load(fm)
                        scale_f = float(cm.get("scale_factor", 1.0))
                except Exception:
                    pass
            data_w = data_w.astype(float) * scale_f
            n_basal = int(noise_sec * sr_w)
            if ch_i < 3:
                filt = filter_emg(data_w, sr_w, hp=20.0, lp=500.0, notch_q=2.0)
                env = get_rms_env(filt, sr_w, smooth_ms=120)
                basal_rms = float(np.sqrt(np.mean(filt[:n_basal] ** 2)))
                active_p95 = float(np.percentile(np.abs(filt[n_basal:]), 95))
                raw_basal_rms = float(np.sqrt(np.mean((data_w[:n_basal] - np.median(data_w[:n_basal])) ** 2)))
                raw_active_rms = float(np.sqrt(np.mean((data_w[n_basal:] - np.median(data_w[n_basal:])) ** 2)))
                # Detectar explosion de basal al inicio (primeros 3s mayores que el resto)
                basal_explosion_ratio = basal_rms / (float(np.percentile(env[n_basal:], 25)) + 1e-6)
                ch_stats[f"ch{ch_i}_basal_uV"] = round(basal_rms, 2)
                ch_stats[f"ch{ch_i}_active_p95_uV"] = round(active_p95, 2)
                ch_stats[f"ch{ch_i}_snr"] = round(active_p95 / (basal_rms + 1e-6), 2)
                ch_stats[f"ch{ch_i}_raw_50hz_uV"] = round(raw_active_rms, 2)
                ch_stats[f"ch{ch_i}_basal_explosion"] = round(basal_explosion_ratio, 2)
            else:
                mic_basal = float(np.std(data_w[:n_basal]))
                mic_p98 = float(np.percentile(np.abs(data_w[n_basal:] - np.median(data_w)), 98))
                ch_stats["mic_basal"] = round(mic_basal, 2)
                ch_stats["mic_p98"] = round(mic_p98, 2)

        row = {
            "exp": exp,
            "toma": entry,
            "vocal": vocal,
            "sr": sr,
            "noise_sec": noise_sec,
            "bpm": bpm,
            "m0": muscles[0] if len(muscles) > 0 else "",
            "m1": muscles[1] if len(muscles) > 1 else "",
            "m2": muscles[2] if len(muscles) > 2 else "",
        }
        row.update(ch_stats)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "auditoria_completa_tomas_2026_10_23.csv"), index=False)
    return df


def extract_pulses_custom(exp_filter, ch_indices, smooth_ms=150, lp=500.0, pre_pct=0.4, post_pct=0.6,
                          weights=None, exclude_anomalous=False, add_temporal_stats=False,
                          add_spatial_diff=False, align_mode="mic", alpha_ruido=0.7, target_len=20):
    if weights is None:
        weights = [1.0] * len(ch_indices)
    X_list, Y_list, T_list, peak_latencies = [], [], [], []

    for entry in sorted(os.listdir(SESSION_DIR)):
        full_p = os.path.join(SESSION_DIR, entry)
        if not os.path.isdir(full_p) or "calibracion" in entry.lower() or "secuencia" in entry.lower():
            continue
        if not any(f.lower() in entry.lower() for f in exp_filter):
            continue
        vocal = entry.split("_")[0].upper()
        if vocal not in ["A", "E", "I", "O", "U"]:
            continue
        meta_p = os.path.join(full_p, "canal_0", "metadata.json")
        if not os.path.exists(meta_p):
            continue
        with open(meta_p, "r", encoding="utf-8") as f:
            meta = json.load(f)
        sr = int(meta.get("sample_rate", 2000))
        noise_sec = float(meta.get("noise_seconds", 3.0))
        bpm = float(meta.get("bpm", 50.0))
        n_basal = int(noise_sec * sr)
        muestras_pulso = int((60.0 / bpm) * sr)

        # Load mic
        mic_p = os.path.join(full_p, "canal_3", "grabacion.wav")
        _, mic_w = wavfile.read(mic_p)
        if mic_w.ndim > 1:
            mic_w = mic_w[:, 0]
        mic_w = np.abs(mic_w.astype(float) - np.median(mic_w))
        mic_rec = mic_w[n_basal:]
        mic_env = np.convolve(mic_rec, np.ones(max(5, int(0.08 * sr))) / max(5, int(0.08 * sr)), mode="same")

        # Load channels
        import soundfile as sf
        envs = {}
        for c_i in ch_indices:
            wp = os.path.join(full_p, f"canal_{c_i}", "grabacion.wav")
            cmp = os.path.join(full_p, f"canal_{c_i}", "metadata.json")
            cw, _ = sf.read(wp)
            if cw.ndim > 1:
                cw = cw[:, 0]
            sf_ch = 1.0
            if os.path.exists(cmp):
                try:
                    with open(cmp, "r", encoding="utf-8") as fm:
                        sf_ch = float(json.load(fm).get("scale_factor", 1.0))
                except Exception:
                    pass
            cw = cw.astype(float) * sf_ch
            filt = filter_emg(cw, sr, hp=20.0, lp=lp, notch_q=2.0)
            env_full = get_rms_env(filt, sr, smooth_ms=smooth_ms)
            envs[c_i] = env_full[n_basal:]

        # Check anomaly in C1 (modiolo detachment explosion in specific tomas)
        if exclude_anomalous and entry in [
            "A_Pruebatricanal_2_Sujeto1",
            "E_Pruebatricanal_Sujeto1",
            "E_Pruebatricanal_2_Sujeto1",
            "I_Pruebatricanal_4_Sujeto1"
        ]:
            continue

        if align_mode == "mic":
            peaks, _ = find_peaks(mic_env, distance=int(0.8 * muestras_pulso), height=np.max(mic_env) * 0.20)
        else:
            ref_env = envs[ch_indices[0]]
            peaks, _ = find_peaks(ref_env, distance=int(0.8 * muestras_pulso), height=np.max(ref_env) * 0.20)

        pre_s = int(pre_pct * muestras_pulso)
        post_s = int(post_pct * muestras_pulso)
        noise_win = max(5, int(muestras_pulso * 0.25))

        for w_idx, pk in enumerate(peaks):
            s_i = pk - pre_s
            s_f = pk + post_s
            if s_i < 0 or s_f > len(mic_env):
                continue
            segs = []
            max_sup = 1e-9
            lat_ch0 = 0.0
            for idx_Pos, c_i in enumerate(ch_indices):
                env_c = envs[c_i]
                seg = env_c[s_i:s_f].copy()
                # Interpulse noise pre & post
                n_pre_i = max(0, int(pk - 0.5 * muestras_pulso - noise_win))
                n_pre_f = min(len(env_c), n_pre_i + noise_win)
                n_post_i = min(len(env_c), int(pk + 0.5 * muestras_pulso))
                n_post_f = min(len(env_c), n_post_i + noise_win)
                r_pre = get_iqr_noise(env_c[n_pre_i:n_pre_f]) if n_pre_f > n_pre_i else np.percentile(seg, 10)
                r_post = get_iqr_noise(env_c[n_post_i:n_post_f]) if n_post_f > n_post_i else r_pre
                r_mean = 0.5 * (r_pre + r_post)
                seg = np.maximum(seg - alpha_ruido * r_mean, 0.0)
                if c_i == 0:
                    lat_ch0 = ((np.argmax(seg) - pre_s) / float(sr)) * 1000.0
                segs.append(seg)
                if np.max(seg) > max_sup:
                    max_sup = np.max(seg)

            if max_sup <= 1e-6:
                continue

            feat_parts = []
            norm_resamp = []
            for idx_pos, seg in enumerate(segs):
                seg_n = (seg / max_sup) * weights[idx_pos]
                seg_r = np.maximum(resample(seg_n, target_len), 0.0)
                norm_resamp.append(seg_r)
                feat_parts.append(seg_r)
                if add_temporal_stats:
                    # Centro de masa temporal, asimetria pre/post pico acustico y derivada
                    t_grid = np.linspace(-1.0, 1.0, target_len)
                    com = np.sum(t_grid * seg_r) / (np.sum(seg_r) + 1e-9)
                    mid = int(target_len * (pre_pct / (pre_pct + post_pct)))
                    e_pre = np.mean(seg_r[:mid])
                    e_post = np.mean(seg_r[mid:])
                    deriv = np.gradient(seg_r)
                    feat_parts.append(np.array([com * 2.0, (e_post - e_pre) * 2.0]))
                    feat_parts.append(deriv)

            if add_spatial_diff and len(norm_resamp) >= 2:
                # Diferencial espacial entre canales (ej. Orbi - Modiolo, Masetero - Modiolo)
                for i in range(len(norm_resamp)):
                    for j in range(i + 1, len(norm_resamp)):
                        feat_parts.append(norm_resamp[i] - norm_resamp[j])

            X_list.append(np.concatenate(feat_parts))
            Y_list.append(vocal)
            T_list.append(f"{entry}_W{w_idx}")
            peak_latencies.append({"exp": exp_filter[0], "vocal": vocal, "toma": entry, "lat_ch0_ms": lat_ch0})

    return np.array(X_list), np.array(Y_list), np.array(T_list), pd.DataFrame(peak_latencies)


def run_ablation_suite():
    results = []
    lat_dfs = []

    configs = [
        # Experimento 1: Masetero Monocanal
        ("Exp1_Masetero", ["Prueba1", "Prueba2", "Prueba3", "Prueba4", "Prueba5"], [0], [1.0], 0.4, 0.6, False, False, False, "1. Base Monocanal C0 (Pre=0.4, Post=0.6)"),
        ("Exp1_Masetero", ["Prueba1", "Prueba2", "Prueba3", "Prueba4", "Prueba5"], [0], [1.0], 0.5, 0.7, False, True, False, "2. Monocanal C0 + Ventana Extendida + Fase/Derivada"),
        # Experimento 2: Belly + Orbi
        ("Exp2_Belly_Orbi", ["pruebabelly"], [0, 2], [1.0, 1.0], 0.4, 0.6, False, False, False, "1. Base Bicanal C0-C2 (Pesos 1:1)"),
        ("Exp2_Belly_Orbi", ["pruebabelly"], [0, 2], [1.0, 2.5], 0.4, 0.6, False, False, False, "2. Bicanal C0-C2 con Belly x2.5"),
        ("Exp2_Belly_Orbi", ["pruebabelly"], [0, 2], [1.0, 3.0], 0.4, 0.6, False, False, False, "3. Bicanal C0-C2 con Belly x3.0"),
        ("Exp2_Belly_Orbi", ["pruebabelly"], [0, 2], [1.0, 2.5], 0.5, 0.7, False, True, True, "4. Bicanal C0-C2 Belly x2.5 + Diferencial + Fase"),
        # Experimento 3: Triada Masetero + Modiolo + Orbi
        ("Exp3_Triada", ["Pruebatricanal"], [0, 1, 2], [1.0, 1.0, 1.0], 0.4, 0.6, False, False, False, "1. Base Triada C0-C1-C2 (Pesos 1:1:1)"),
        ("Exp3_Triada", ["Pruebatricanal"], [0, 1, 2], [2.5, 1.0, 1.0], 0.4, 0.6, False, False, False, "2. Triada con Masetero x2.5"),
        ("Exp3_Triada", ["Pruebatricanal"], [0, 1, 2], [3.0, 1.0, 1.0], 0.4, 0.6, False, False, False, "3. Triada con Masetero x3.0"),
        ("Exp3_Triada", ["Pruebatricanal"], [0, 1, 2], [2.5, 1.0, 1.0], 0.4, 0.6, True, False, False, "4. Triada Masetero x2.5 Purgando Tomas con Modiolo Despegado"),
        ("Exp3_Triada", ["Pruebatricanal"], [0, 1, 2], [2.5, 1.0, 1.5], 0.5, 0.7, True, True, True, "5. Triada Masetero x2.5 + Purga + Diferencial Orbi-Mod + Fase"),
        ("Exp3_Triada", ["Pruebatricanal"], [0, 2], [2.5, 1.0], 0.4, 0.6, False, False, False, "6. Par C0-C2 (Masetero x2.5 + Orbi, sin Modiolo ruidoso)"),
    ]

    for exp_name, filt, chs, wts, pre_p, post_p, excl, t_stat, s_diff, desc in configs:
        best_2d = 0.0
        best_3d = 0.0
        best_lda = 0.0
        best_v2d = {}
        best_v3d = {}
        for sm in [80, 125, 180, 220]:
            for tl in [10, 20, 40]:
                for al in [0.3, 0.7, 1.0]:
                    X, Y, T, df_lat = extract_pulses_custom(
                        filt, chs, smooth_ms=sm, lp=500.0, pre_pct=pre_p, post_pct=post_p,
                        weights=wts, exclude_anomalous=excl, add_temporal_stats=t_stat,
                        add_spatial_diff=s_diff, alpha_ruido=al, target_len=tl
                    )
                    if sm == 125 and tl == 20 and al == 0.7 and not df_lat.empty:
                        lat_dfs.append(df_lat)
                    if len(X) < 15:
                        continue
                    pca2 = PCA(n_components=2, random_state=42).fit_transform(X)
                    acc2, pv2, _ = eval_hungarian_acc(pca2, Y, "gmm")
                    if acc2 > best_2d and min(pv2.values()) > 0:
                        best_2d = acc2
                        best_v2d = pv2

                    pca3 = PCA(n_components=min(3, X.shape[1]), random_state=42).fit_transform(X)
                    acc3, pv3, _ = eval_hungarian_acc(pca3, Y, "gmm")
                    if acc3 > best_3d and min(pv3.values()) > 0:
                        best_3d = acc3
                        best_v3d = pv3

                    pca5 = PCA(n_components=min(6, X.shape[1]), random_state=42).fit_transform(X)
                    accl, _, _ = eval_hungarian_acc(pca5, Y, "lda")
                    if accl > best_lda:
                        best_lda = accl

        results.append({
            "Experimento": exp_name,
            "Configuracion": desc,
            "PCA_2D_GMM_%": round(best_2d, 2),
            "PCA_3D_GMM_%": round(best_3d, 2),
            "PCA_LDA_%": round(best_lda, 2),
            "Desglose_2D": json.dumps(best_v2d),
            "Desglose_3D": json.dumps(best_v3d)
        })
        print(f"[{exp_name}] {desc} -> 2D: {best_2d:.2f}% | 3D: {best_3d:.2f}% | LDA: {best_lda:.2f}%")

    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUT_DIR, "ablacion_separabilidad_2026_10_23.csv"), index=False)
    if lat_dfs:
        df_all_lat = pd.concat(lat_dfs, ignore_index=True)
        df_all_lat.to_csv(os.path.join(OUT_DIR, "latencias_masetero_por_vocal.csv"), index=False)
    return df_res


if __name__ == "__main__":
    print("1. Auditando todas las tomas del 2026-10-23...")
    df_audit = audit_all_tomas()
    print(f"Tomas auditadas: {len(df_audit)}")
    print("\nResumen por Experimento y Vocal (Mediana de P95 activo en uV):")
    print(df_audit.groupby(["exp", "vocal"])[["ch0_active_p95_uV", "ch1_active_p95_uV", "ch2_active_p95_uV", "ch0_basal_explosion"]].median())
    print("\n2. Ejecutando bateria de ablacion y separabilidad...")
    df_ablation = run_ablation_suite()
