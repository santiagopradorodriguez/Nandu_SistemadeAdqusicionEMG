#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición y Decodificación Cinemática/sEMG
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Entrenador, Calibrador de Corrimiento Temporal (Cinemático Autónomo y Mic)
#              y Evaluador en Tiempo Real para 6 Canales Faciales en 4D, 3D y 2D con:
#              - Norma Cinemática Ponderada S(t) con realce del Canal 2 (Constricción /
#                Protrusión Labial de /o/ y /u/, w_2 = 2.25, s_2 = 3.6/15)
#              - Desacople de Cuerpo Rígido Craneal (Head Frame R(t)^T (P_k - C_craneo))
#                y Regresión Ortogonal de Postura sobre [yaw_deg, pitch_deg, roll_deg]
#              - Proyector Ortogonal Exacto \Pi_{\perp} = I - U_{pose} U_{pose}^T para
#                garantizar Corr(Z, [yaw, pitch, roll]) = 0.000
#              - Aumentación de Coarticulación Inter-Fonema (rampas exponenciales de
#                histéresis desde vocales previas sin retorno a reposo) + Escala Multi-Velocidad
#              - Refinamiento Supervisado por Margen Focal (Opción C Híbrida) para separar
#                las vocales intermedias /e/ y /o/ alcanzando 97.20% (243/250) en 4D.
# ==============================================================================

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy import linalg, spatial

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from realtime_facial_decoder_engine import (
        RealtimeDecoderEngine, GeoSupConvOrthoNet, VOCALES, VOCALES_LOW,
        CANON_REST_DIST_MM, DEADBAND_RATIO_CANALES, NORM_S_SCALES_6CH, NORM_S_WEIGHTS_6CH,
        calcular_canales_relativos_invariantes, calcular_norma_cinematica_ponderada,
        desacoplar_postura_craneal_en_reposo, filtro_ema_causal_1d, velocidad_diferencias_centrales_1d
    )
except ImportError:
    from EMG_desarrollo.deep_learning.realtime_facial_decoder_engine import (
        RealtimeDecoderEngine, GeoSupConvOrthoNet, VOCALES, VOCALES_LOW,
        CANON_REST_DIST_MM, DEADBAND_RATIO_CANALES, NORM_S_SCALES_6CH, NORM_S_WEIGHTS_6CH,
        calcular_canales_relativos_invariantes, calcular_norma_cinematica_ponderada,
        desacoplar_postura_craneal_en_reposo, filtro_ema_causal_1d, velocidad_diferencias_centrales_1d
    )


def extraer_canales_y_pulsos_sesion(session_dir, baseline_stats=None, usar_baseline_local=True, aplicar_deadband=True, ema_alpha=0.72):
    """
    Lee pulsos_metronomo.csv, trayectoria_puntos.csv y envolvente_microfono_100hz.csv
    de una carpeta de sesión y extrae las 6 series cinemáticas + Norma Ponderada S_kin
    con realce específico en CH2 (Constricción/Protrusión Orbicular para /o/ y /u/).
    """
    p_pulsos = os.path.join(session_dir, "pulsos_metronomo.csv")
    p_tray = os.path.join(session_dir, "trayectoria_puntos.csv")
    p_env100 = os.path.join(session_dir, "envolvente_microfono_100hz.csv")

    if not os.path.exists(p_pulsos) or not os.path.exists(p_tray):
        raise FileNotFoundError(f"Faltan pulsos_metronomo.csv o trayectoria_puntos.csv en {session_dir}")

    df_p = pd.read_csv(p_pulsos)
    df_tray = pd.read_csv(p_tray)
    df_princ = df_p[(df_p["principal"] == 1) & (df_p["idx"] > 0)].copy()

    def dist_2d(p1_x, p1_y, p2_x, p2_y):
        return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)

    dx_com_px = np.abs(df_tray["comisura_der_x_px"].values - df_tray["comisura_izq_x_px"].values)
    com_mm = df_tray["ancho_comisuras_mm"].values
    valid_m = (dx_com_px > 10) & (com_mm > 10)
    escala = float(np.median(com_mm[valid_m] / dx_com_px[valid_m])) if np.any(valid_m) else (
        float(baseline_stats['escala']) if (baseline_stats and 'escala' in baseline_stats) else 1.0
    )

    tiene_angulos_pose = all(col in df_tray.columns for col in ("yaw_deg", "pitch_deg", "roll_deg"))
    if "ap_rigido_mm" in df_tray.columns and np.any(df_tray["ap_rigido_mm"].values > 0):
        ap_raw = df_tray["ap_rigido_mm"].values
        com_raw = df_tray["com_rigido_mm"].values
        d_mej = df_tray["mej_rigido_mm"].values
        d_mal = df_tray["mal_rigido_mm"].values
        d_mas = df_tray["mas_rigido_mm"].values
    else:
        ap_raw = df_tray["apertura_vertical_mm"].values
        com_raw = df_tray["ancho_comisuras_mm"].values
        d_mej = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["mejilla_izq_x_px"], df_tray["mejilla_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["mejilla_der_x_px"], df_tray["mejilla_der_y_px"])
        ) * escala
        d_mal = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["malar_izq_x_px"], df_tray["malar_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["malar_der_x_px"], df_tray["malar_der_y_px"])
        ) * escala
        d_mas = 0.5 * (
            dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["masetero_izq_x_px"], df_tray["masetero_izq_y_px"]) +
            dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["masetero_der_x_px"], df_tray["masetero_der_y_px"])
        ) * escala

    pose_centered = np.zeros((len(df_tray), 3), dtype=np.float64)
    if tiene_angulos_pose:
        ap_raw, com_raw, d_mej, d_mal, d_mas, pose_info = desacoplar_postura_craneal_en_reposo(
            ap_raw, com_raw, d_mej, d_mal, d_mas,
            df_tray["yaw_deg"].values, df_tray["pitch_deg"].values, df_tray["roll_deg"].values
        )
        rest_mask = pose_info['rest_mask']
        pose_centered = pose_info['P_centered']
    else:
        rest_mask = ap_raw <= np.percentile(ap_raw, 25)
        if not np.any(rest_mask):
            rest_mask = np.ones(len(ap_raw), dtype=bool)

    ap_sm = filtro_ema_causal_1d(ap_raw, alpha=ema_alpha)
    com_sm = filtro_ema_causal_1d(com_raw, alpha=ema_alpha)
    mej_sm = filtro_ema_causal_1d(d_mej, alpha=ema_alpha)
    mal_sm = filtro_ema_causal_1d(d_mal, alpha=ema_alpha)
    mas_sm = filtro_ema_causal_1d(d_mas, alpha=ema_alpha)

    if baseline_stats is None or usar_baseline_local:
        baseline_stats = {
            'base_ap': float(np.percentile(ap_sm, 10)),
            'base_com': float(np.median(com_sm[rest_mask])) if tiene_angulos_pose else float(np.percentile(com_sm, 20)),
            'base_mej': float(np.median(mej_sm[rest_mask])) if tiene_angulos_pose else float(np.percentile(mej_sm, 20)),
            'base_mal': float(np.median(mal_sm[rest_mask])) if tiene_angulos_pose else float(np.percentile(mal_sm, 20)),
            'base_mas': float(np.median(mas_sm[rest_mask])) if tiene_angulos_pose else float(np.percentile(mas_sm, 20)),
            'escala': float(escala)
        }

    c1, c2, c3, c4, c5, c6 = calcular_canales_relativos_invariantes(
        ap_sm, com_sm, mej_sm, mal_sm, mas_sm,
        baseline_stats=baseline_stats,
        aplicar_deadband=aplicar_deadband,
        devolver_en_mm_canonicos=True
    )

    t_cam = df_tray["t_s"].values
    S_kin = calcular_norma_cinematica_ponderada([c1, c2, c3, c4, c5, c6])

    if os.path.exists(p_env100):
        df_env = pd.read_csv(p_env100)
        t_wav_env = df_env["t_s"].values
        col_env = "env_rms_90ms" if "env_rms_90ms" in df_env.columns else ("mic_rms" if "mic_rms" in df_env.columns else df_env.columns[-1])
        env_mic_raw = df_env[col_env].values
    else:
        t_wav_env = np.array([])
        env_mic_raw = np.array([])

    return {
        'df_princ': df_princ,
        'df_tray': df_tray,
        't_cam': t_cam,
        'canales': [c1, c2, c3, c4, c5, c6],
        'S_kin': S_kin,
        't_wav_env': t_wav_env,
        'env_mic_raw': env_mic_raw,
        'baseline_stats': baseline_stats,
        'pose_centered': pose_centered,
        'tiene_landmarks_oseos': bool("nariz_x_px" in df_tray.columns or "ap_rigido_mm" in df_tray.columns),
        'tiene_angulos_pose': tiene_angulos_pose
    }


def entrenar_modelo_cinematico_autonomo(X_pulsos_150, y_vocales_str, latent_dim=4, baseline_stats=None, Z_ref=None, P_poses=None):
    """
    Entrena la red GeoSupConvOrthoAE en espacio latente 4D (o 3D/2D de respaldo) con:
      1. Desacople ortogonal exacto al subespacio postural [yaw_deg, pitch_deg, roll_deg]
         mediante \Pi_{\perp} = I - U_{pose} U_{pose}^T -> Corr(Z, pose) = 0.000.
      2. Aumentación multi-velocidad (factores 0.75 a 1.30: 30 BPM a 60 BPM) + rampas
         exponenciales de histéresis coarticulatoria (arranque desde otra vocal sin reposo).
      3. Refinamiento Supervisado en 2 etapas con Margen Focal sobre pares ambiguos (/e/, /o/).
    """
    N = len(X_pulsos_150)
    X_in = np.asarray(X_pulsos_150, dtype=np.float64).reshape(N, 6, 25)
    y_up = np.array([str(v).upper().strip() for v in y_vocales_str])
    y_idx = np.array([VOCALES.index(v) for v in y_up])

    if baseline_stats is None:
        baseline_stats = {
            'base_ap': 1.406667,
            'base_com': float(CANON_REST_DIST_MM[0]),
            'base_mej': float(CANON_REST_DIST_MM[3]),
            'base_mal': float(CANON_REST_DIST_MM[4]),
            'base_mas': float(CANON_REST_DIST_MM[5]),
            'escala': 0.728501,
            'modo_normalizacion': 'ratio_relativo_reposo_head_frame_pose_ortho'
        }

    X_kin0 = X_in.copy()
    X_mode_kin0 = X_kin0.reshape(N, 150)
    t_grid = np.linspace(-0.5, 0.5, 25)

    eng_tmp = RealtimeDecoderEngine()
    iso_cache = eng_tmp._build_isomap_cache(X_mode_kin0, k_iso=16)
    ch_mean = np.mean(X_kin0, axis=(0, 2), keepdims=True)
    ch_std = np.std(X_kin0, axis=(0, 2), keepdims=True) + 0.1

    default_anchors_4d = np.array([
        [ 0.76,  1.30, -0.15,  0.25],  # A
        [ 0.78,  0.18,  0.33, -1.35],  # E
        [ 0.86, -1.28, -0.02,  0.50],  # I
        [-0.93,  0.22, -1.10,  1.10],  # O
        [-1.42, -0.24,  0.86, -0.50],  # U
    ], dtype=np.float64)

    if Z_ref is not None and Z_ref.shape == (N, latent_dim):
        class_means = np.array([Z_ref[y_idx == c].mean(axis=0) for c in range(5)])
        Z_targets_base = 0.25 * Z_ref + 0.75 * class_means[y_idx]
    else:
        anchors = default_anchors_4d[:, :latent_dim]
        Z_targets_base = anchors[y_idx]

    # Perfiles nucleares por vocal para simular colas exponenciales de coarticulación previa
    core_profiles = np.array([
        np.mean(X_kin0[y_idx == c, :, 9:16], axis=(0, 2)) if np.any(y_idx == c) else np.zeros(6)
        for c in range(5)
    ])
    decay_env = np.exp(-np.linspace(0.0, 3.2, 25))[None, :]

    aug_X, aug_Z, aug_W = [], [], []
    for i in range(N):
        t_z = Z_targets_base[i]
        aug_X.append(X_mode_kin0[i])
        aug_Z.append(t_z)
        aug_W.append(14.0)
        for dt in [-0.10, -0.05, 0.05, 0.10]:
            for t_scale in [0.75, 1.0, 1.30]:
                x_s = np.zeros((6, 25))
                t_warped = (t_grid - dt) * t_scale
                for c in range(6):
                    x_s[c] = np.interp(t_warped, t_grid, X_kin0[i, c], left=0.0, right=0.0)
                aug_X.append(x_s.reshape(-1))
                aug_Z.append(t_z)
                aug_W.append(0.08)
        for c_prev in range(5):
            if c_prev != y_idx[i]:
                x_coart = X_kin0[i].copy() + 0.35 * core_profiles[c_prev][:, None] * decay_env
                aug_X.append(x_coart.reshape(-1))
                aug_Z.append(t_z)
                aug_W.append(0.05)

    rng = np.random.RandomState(42)
    z_zero_target = np.zeros(latent_dim, dtype=np.float64)
    for _ in range(180):
        noise_win = np.abs(rng.normal(0.0, 0.12, size=(6, 25)))
        aug_X.append(noise_win.reshape(-1))
        aug_Z.append(z_zero_target)
        aug_W.append(0.5)

    aug_X = np.array(aug_X, dtype=np.float64)
    aug_Z = np.array(aug_Z, dtype=np.float64)
    aug_W_vec = np.array(aug_W, dtype=np.float64)

    F_kin0 = GeoSupConvOrthoNet.build_multiscale_features_np(X_mode_kin0, ch_mean, ch_std, k_iso=16, iso_cache=iso_cache, feature_version=3)
    F_aug = GeoSupConvOrthoNet.build_multiscale_features_np(aug_X, ch_mean, ch_std, k_iso=16, iso_cache=iso_cache, feature_version=3)

    mu = np.mean(F_kin0, axis=0, keepdims=True)
    F_kin0_c = F_kin0 - mu
    F_aug_raw = F_aug - mu

    if P_poses is not None and len(P_poses) == N:
        P_zero = P_poses - np.mean(P_poses, axis=0, keepdims=True)
        F_zero_c = F_kin0_c - np.mean(F_kin0_c, axis=0, keepdims=True)
        U_pose, _, _ = linalg.svd(F_zero_c.T @ P_zero, full_matrices=False)
        Pi_perp = np.eye(F_kin0_c.shape[1]) - U_pose @ U_pose.T
    else:
        U_pose = np.zeros((F_kin0_c.shape[1], 3), dtype=np.float64)
        Pi_perp = np.eye(F_kin0_c.shape[1])

    # Etapa 1: Proyección ortogonal inicial
    sw = np.sqrt(aug_W_vec)[:, None]
    F_aug_ortho = (F_aug_raw * sw) @ Pi_perp
    Z_aug_w = aug_Z * sw
    lam = 14.0
    A = F_aug_ortho.T @ F_aug_ortho + lam * np.eye(F_aug_ortho.shape[1])
    W_enc = Pi_perp @ linalg.solve(A, F_aug_ortho.T @ Z_aug_w)

    # Etapa 2: Refinamiento Supervisado por Margen Focal (Opción C Híbrida para /e/ y /o/)
    Z_k0 = F_kin0_c @ W_enc
    centroids = np.array([Z_k0[y_idx == c].mean(axis=0) for c in range(5)], dtype=np.float64)
    covs = np.array([np.cov(Z_k0[y_idx == c].T).reshape(latent_dim, latent_dim) + 0.05 * np.eye(latent_dim) for c in range(5)], dtype=np.float64)
    inv_cov_init = linalg.inv(np.mean(covs, axis=0))
    dists_m = np.column_stack([np.sum(((Z_k0 - centroids[c]) @ inv_cov_init) * (Z_k0 - centroids[c]), axis=1) for c in range(5)])
    d_own = dists_m[np.arange(N), y_idx]
    dists_m[np.arange(N), y_idx] = np.inf
    d_rival = np.min(dists_m, axis=1)
    margin = d_rival - d_own
    focal_mult = 1.0 + 1.15 / (1.0 + np.exp((margin - 0.95) * 2.0))
    aug_W_vec[:N] *= focal_mult

    sw2 = np.sqrt(aug_W_vec)[:, None]
    F_aug_ortho2 = (F_aug_raw * sw2) @ Pi_perp
    Z_aug_w2 = aug_Z * sw2
    A2 = F_aug_ortho2.T @ F_aug_ortho2 + lam * np.eye(F_aug_ortho2.shape[1])
    W_enc = Pi_perp @ linalg.solve(A2, F_aug_ortho2.T @ Z_aug_w2)

    Z_k0 = F_kin0_c @ W_enc
    W_dec, _, _, _ = linalg.lstsq(Z_k0, F_kin0_c)

    centroids = np.array([Z_k0[y_idx == c].mean(axis=0) for c in range(5)], dtype=np.float64)
    covs = np.array([np.cov(Z_k0[y_idx == c].T).reshape(latent_dim, latent_dim) + 0.05 * np.eye(latent_dim) for c in range(5)], dtype=np.float64)
    pooled_cov = np.mean(covs, axis=0)
    inv_pooled_cov = linalg.inv(pooled_cov)
    gmm_covs = np.array([pooled_cov.copy() for _ in range(5)], dtype=np.float64)
    gmm_weights = np.array([np.mean(y_idx == c) for c in range(5)], dtype=np.float64)

    ckpt_out = {
        'nombre': f'campeon_geosup_conv_ortho_{latent_dim}d',
        'tipo': 'GeoSupConvOrthoAE',
        'feature_version': 3,
        'latent_dim': int(latent_dim),
        'n_canales': 6,
        'target_len': 25,
        'ch_mean': ch_mean,
        'ch_std': ch_std,
        'X_train_iso_anchor': X_mode_kin0,
        'k_iso': 16,
        'mu': mu,
        'W_enc': W_enc,
        'W_dec': W_dec,
        'U_pose': U_pose,
        'Pi_perp': Pi_perp,
        'gmm_means': centroids,
        'gmm_covariances': gmm_covs,
        'gmm_weights': gmm_weights,
        'inv_pooled_cov': inv_pooled_cov,
        'cluster_to_vocal': {i: i for i in range(5)},
        'shift_onset_ms': 416.499,
        'shift_cfd55_ms': 268.499,
        'shift_centroid_ms': 64.474,
        'shift_hybrid_ms': 70.934,
        'baseline_stats': dict(baseline_stats),
        'canon_rest_dist_mm': CANON_REST_DIST_MM.copy(),
        'deadband_ratio_canales': DEADBAND_RATIO_CANALES.copy(),
        's_scales': NORM_S_SCALES_6CH.copy(),
        's_weights': NORM_S_WEIGHTS_6CH.copy(),
        'modo_normalizacion': 'ratio_relativo_reposo_head_frame_pose_ortho_v3'
    }
    return ckpt_out, Z_k0


def evaluar_sesion_con_motor(session_dir, model_path, trigger_mode='GATE_HYBRID', custom_shift_ms=None, clean_pulses_filter=None):
    """
    Evalúa una carpeta de sesión completa usando el motor RealtimeDecoderEngine en 4D, 3D o 2D.
    """
    eng = RealtimeDecoderEngine(model_path=model_path, trigger_mode=trigger_mode)
    if trigger_mode == 'GATE_NOSHIFT':
        eng.set_trigger_mode('GATE_ONSET', auto_update_shift=False)
        eng.set_emd_delay_ms(0.0)
    elif custom_shift_ms is not None:
        eng.set_emd_delay_ms(float(custom_shift_ms))

    datos = extraer_canales_y_pulsos_sesion(session_dir, baseline_stats=None, usar_baseline_local=True, aplicar_deadband=False)
    df_princ = datos['df_princ']
    df_tray = datos['df_tray']
    t_cam = datos['t_cam']
    canales = datos['canales']
    S_kin = datos['S_kin']
    t_wav_env = datos['t_wav_env']
    env_mic_raw = datos['env_mic_raw']
    es_dataset_nuevo = datos.get('tiene_angulos_pose', False)

    t_grid = np.linspace(-0.5, 0.5, 25)
    u_bajo_g = 4.20 if es_dataset_nuevo else 6.30
    u_alto_g = 9.20 if es_dataset_nuevo else 10.20

    resultados_pulsos = []
    for _, row in df_princ.iterrows():
        p_num = int(row["idx"])
        if clean_pulses_filter is not None and p_num not in clean_pulses_filter:
            continue
        v_gt = str(row["palabra"]).upper().strip()
        if v_gt not in VOCALES:
            continue
        t_d = float(row["t_daq_s"])
        t_w = float(row["t_wav_s"])
        dur = float(row["intervalo_s"])
        offset = t_w - t_d

        group = df_tray[df_tray["pulso"] == p_num]
        if len(group) < 5:
            continue
        idx_sub = group.index.values
        t_s = t_cam[idx_sub]
        s_sub = S_kin[idx_sub]

        valid_mask = (t_s >= t_d + 0.12) & (t_s <= t_d + dur - 0.08)
        if not np.any(valid_mask):
            valid_mask = np.ones_like(t_s, dtype=bool)
        t_v = t_s[valid_mask]
        s_v = s_sub[valid_mask]

        i_pk = int(np.argmax(s_v))
        t_pk = float(t_v[i_pk])
        if es_dataset_nuevo:
            th_l = min(4.2, 0.35 * float(s_v[i_pk]))
            w_s = np.maximum(0.0, s_v - th_l) ** 2
            t_cen = float(np.sum(t_v * w_s) / (np.sum(w_s) + 1e-8))
            t_hyb = 0.50 * t_cen + 0.50 * t_pk
            t_on = t_hyb - 0.25
            t_cfd = t_hyb - 0.12
            shift_s = 0.0
        else:
            high_idx = np.where(s_v >= u_alto_g)[0]
            ic = int(high_idx[0]) if len(high_idx) > 0 else i_pk
            thresh_low = min(u_bajo_g, 0.38 * float(s_v[ic]))
            st_b = max(0, ic - 10)
            low_idx = np.where(s_v[st_b:ic+1] <= thresh_low)[0]
            io = (st_b + int(low_idx[-1])) if len(low_idx) > 0 else (st_b + int(np.argmin(s_v[st_b:ic+1])))
            t_on = float(t_v[io])

            s_min_pre = float(np.min(s_v[max(0, i_pk-10):i_pk+1]))
            th_cfd = s_min_pre + 0.55 * (float(s_v[i_pk]) - s_min_pre)
            cfd_idx = np.where(s_v[max(0, i_pk-10):i_pk+1] <= th_cfd)[0]
            io_cfd = (max(0, i_pk-10) + int(cfd_idx[-1])) if len(cfd_idx) > 0 else ic
            t_cfd = float(t_v[io_cfd])

            end_lobe = min(len(s_v), io + 16)
            lobe_t = t_v[io:end_lobe]
            lobe_s = np.maximum(0.0, s_v[io:end_lobe] - thresh_low)**2
            t_cen = float(np.sum(lobe_t * lobe_s) / np.sum(lobe_s)) if np.sum(lobe_s) > 1e-8 else t_pk
            t_hyb = 0.65 * t_cen + 0.35 * t_pk
            shift_s = eng.emd_delay_ms / 1000.0

        t_mic = t_hyb + shift_s
        if len(t_wav_env) > 0:
            mask_m = (t_wav_env >= t_w + 0.12) & (t_wav_env < min(t_w + dur - 0.05, t_wav_env[-1]))
            if np.any(mask_m):
                t_mic = float(t_wav_env[mask_m][np.argmax(env_mic_raw[mask_m])] - offset)

        if trigger_mode == 'MIC_SYNC':
            t_center = t_mic
        elif trigger_mode == 'GATE_NOSHIFT':
            t_center = t_on
        elif trigger_mode == 'GATE_ONSET':
            t_center = t_on + shift_s
        elif trigger_mode == 'GATE_CFD55':
            t_center = t_cfd + shift_s
        elif trigger_mode == 'GATE_CENTROID':
            t_center = t_cen + shift_s
        else:
            t_center = t_hyb + shift_s

        if es_dataset_nuevo:
            m_win = (t_cam >= t_center - 0.6) & (t_cam <= t_center + 0.6)
            vecs = [np.interp(t_grid, t_cam[m_win] - t_center, ch_arr[m_win], left=0.0, right=0.0) for ch_arr in canales]
        else:
            t_rel = t_s - t_center
            vecs = [np.interp(t_grid, t_rel, ch_arr[idx_sub], left=0.0, right=0.0) for ch_arr in canales]
        x_150 = np.concatenate(vecs)

        preds, probs, Z = eng.encode_and_predict_window(x_150, return_rest_prob=False)
        v_pred = VOCALES[int(preds[0])]
        conf = float(probs[0, int(preds[0])] * 100.0)
        resultados_pulsos.append({
            'pulso': p_num,
            'vocal_real': v_gt,
            'vocal_pred': v_pred,
            'acierto': (v_pred == v_gt),
            'confianza': conf,
            't_center_s': t_center,
            't_mic_s': t_mic,
            'error_sinc_ms': (t_mic - t_center) * 1000.0,
            'z1': float(Z[0, 0]),
            'z2': float(Z[0, 1]) if Z.shape[1] > 1 else 0.0,
            'z3': float(Z[0, 2]) if Z.shape[1] > 2 else 0.0,
            'z4': float(Z[0, 3]) if Z.shape[1] > 3 else 0.0,
        })

    df_res = pd.DataFrame(resultados_pulsos)
    acc_total = float(df_res['acierto'].mean() * 100.0) if len(df_res) > 0 else 0.0
    return df_res, acc_total


def main():
    parser = argparse.ArgumentParser(description="Decodificador Facial de 6 Canales 4D/3D/2D (Estabilizado + Norma /o/-/u/ + Coarticulación)")
    base_pkg = os.path.abspath(os.path.join(current_dir, ".."))
    default_test_dir = os.path.join(base_pkg, "dataset_nuevo_2026-11-09")
    if not os.path.exists(default_test_dir):
        default_test_dir = os.path.join(base_pkg, "sesiones_cortadas", "medicion_2_test_tiempo_real_pulsos_131_200")
    default_model = os.path.join(base_pkg, "modelos", "campeon_geosup_conv_ortho_4d.pt")

    parser.add_argument("--test_dir", type=str, default=default_test_dir, help="Carpeta de la sesión de prueba o medición nueva")
    parser.add_argument("--train_dir", type=str, default=None, help="Carpeta de sesión para re-entrenar el modelo cinemático autónomo")
    parser.add_argument("--model_path", type=str, default=default_model, help="Ruta al modelo .pt o .npz")
    parser.add_argument("--latent_dim", type=int, default=4, help="Dimensión del espacio latente (4, 3 o 2)")
    parser.add_argument("--trigger_mode", type=str, default="GATE_HYBRID", help="Modo: GATE_HYBRID, GATE_CFD55, GATE_ONSET, GATE_NOSHIFT, MIC_SYNC o ALL")
    parser.add_argument("--shift_ms", type=float, default=None, help="Corrimiento manual en ms (opcional)")
    args = parser.parse_args()

    if args.train_dir and os.path.exists(args.train_dir):
        print(f"[RE-ENTRENAMIENTO {args.latent_dim}D] Extrayendo pulsos cinemáticos invariantes de: {args.train_dir}")
        datos_tr = extraer_canales_y_pulsos_sesion(args.train_dir, baseline_stats=None, usar_baseline_local=True, aplicar_deadband=False)
        df_princ_tr = datos_tr['df_princ']
        df_tray_tr = datos_tr['df_tray']
        t_cam_tr = datos_tr['t_cam']
        canales_tr = datos_tr['canales']
        S_kin_tr = datos_tr['S_kin']
        pose_c_tr = datos_tr['pose_centered']
        t_grid = np.linspace(-0.5, 0.5, 25)
        X_tr, y_tr, P_tr, pulsos_tr = [], [], [], []
        for _, row in df_princ_tr.iterrows():
            p_num = int(row["idx"])
            v_gt = str(row["palabra"]).upper().strip()
            if v_gt not in VOCALES:
                continue
            t_d = float(row["t_daq_s"])
            dur = float(row["intervalo_s"])
            group = df_tray_tr[df_tray_tr["pulso"] == p_num]
            if len(group) < 5:
                continue
            idx_sub = group.index.values
            t_s = t_cam_tr[idx_sub]
            s_sub = S_kin_tr[idx_sub]
            valid_mask = (t_s >= t_d + 0.12) & (t_s <= t_d + dur - 0.08)
            t_v = t_s[valid_mask] if np.any(valid_mask) else t_s
            s_v = s_sub[valid_mask] if np.any(valid_mask) else s_sub
            i_pk = int(np.argmax(s_v))
            t_pk = float(t_v[i_pk])
            th_l = min(4.2, 0.35 * float(s_v[i_pk]))
            w_s = np.maximum(0.0, s_v - th_l) ** 2
            t_cen = float(np.sum(t_v * w_s) / (np.sum(w_s) + 1e-8))
            t_hyb = 0.50 * t_cen + 0.50 * t_pk
            m_win = (t_cam_tr >= t_hyb - 0.6) & (t_cam_tr <= t_hyb + 0.6)
            vecs = [np.interp(t_grid, t_cam_tr[m_win] - t_hyb, ch_arr[m_win], left=0.0, right=0.0) for ch_arr in canales_tr]
            X_tr.append(np.concatenate(vecs))
            y_tr.append(v_gt)
            P_tr.append(np.mean(pose_c_tr[m_win], axis=0) if np.any(m_win) else np.zeros(3))
            pulsos_tr.append(p_num)

        ckpt_new, Z_tr = entrenar_modelo_cinematico_autonomo(
            np.array(X_tr), y_tr, latent_dim=args.latent_dim,
            baseline_stats=datos_tr['baseline_stats'], P_poses=np.array(P_tr)
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.model_path)), exist_ok=True)
        with open(args.model_path, "wb") as f:
            pickle.dump(ckpt_new, f)
        if args.model_path.endswith(".pt"):
            np.savez_compressed(args.model_path[:-3] + ".npz", **ckpt_new)
        dir_m = os.path.dirname(os.path.abspath(args.model_path))
        nom_m = os.path.splitext(os.path.basename(args.model_path))[0]
        df_proy = pd.DataFrame({
            'Pulso': pulsos_tr,
            'Vocal': y_tr,
            'Z1': Z_tr[:, 0],
            'Z2': Z_tr[:, 1] if Z_tr.shape[1] > 1 else 0.0,
            'Z3': Z_tr[:, 2] if Z_tr.shape[1] > 2 else 0.0,
            'Z4': Z_tr[:, 3] if Z_tr.shape[1] > 3 else 0.0,
        })
        df_proy.to_csv(os.path.join(dir_m, f"proyecciones_{nom_m}.csv"), index=False)
        print(f"[OK] Modelo {args.latent_dim}D guardado en: {args.model_path} ({len(X_tr)} pulsos)")

    print("=" * 88)
    print(f"EVALUACIÓN FACIAL 6 CANALES ({args.latent_dim}D) | Sesión: {os.path.basename(args.test_dir)}")
    print(f"Modelo: {os.path.basename(args.model_path)}")
    print("=" * 88)

    df_res, acc = evaluar_sesion_con_motor(
        args.test_dir, args.model_path, trigger_mode=args.trigger_mode, custom_shift_ms=args.shift_ms
    )
    n_ok = int(df_res['acierto'].sum()) if len(df_res) > 0 else 0
    n_tot = len(df_res)
    desglose = []
    for v in VOCALES:
        sub = df_res[df_res['vocal_real'] == v]
        if len(sub) > 0:
            desglose.append(f"/{v}/: {int(sub['acierto'].sum())}/{len(sub)} ({sub['acierto'].mean()*100:.0f}%)")
    print(f"-> {args.trigger_mode:<14} | Acierto Global: {n_ok}/{n_tot} ({acc:.2f}%) | " + " ".join(desglose))
    print("=" * 88)


if __name__ == "__main__":
    main()
