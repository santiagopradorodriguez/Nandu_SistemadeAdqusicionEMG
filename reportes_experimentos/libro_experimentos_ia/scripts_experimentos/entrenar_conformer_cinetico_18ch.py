#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición y Decodificación Cinemática Facial
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Entrenamiento de la Arquitectura Temporal Convolucional Multiescala
#              + Multi-Head Self-Attention (Conformer Ligero) sobre 18 Canales
#              Dinámicos (q, dq/dt, d2q/dt2) con:
#              - Extracción de ventanas 40%-60% de tau idéntica a generador_pca_6canales.py
#              - Reloj Maestro Acústico (RMS 250 ms) y Suavizado Cinemático (125 ms)
#              - Prevención estricta de fuga de datos (Split por bloques contiguos,
#                estadísticas y encoder ajustados solo en Train)
#              - Exportación de modelos 4D, 3D y 2D compatibles con el decodificador
#                en tiempo real (autoforge_daq_decodificador_facial.py).
# ==============================================================================

import os
import sys
import pickle
import random
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy import linalg, spatial
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

# Asegurar rutas de importación del motor en tiempo real
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, "..", "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from EMG_desarrollo.deep_learning.realtime_facial_decoder_engine import (
    RealtimeDecoderEngine, GeoSupConvOrthoNet, VOCALES,
    CANON_REST_DIST_MM, DEADBAND_RATIO_CANALES, NORM_S_SCALES_6CH, NORM_S_WEIGHTS_6CH,
    calcular_canales_relativos_invariantes, desacoplar_postura_craneal_en_reposo
)

# Semilla fija universal (Regla 4 AGENTS.md)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

VOCALES_CANONICAS = ["a", "e", "i", "o", "u"]
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES_CANONICAS)}


def calcular_envolvente_rms(audio_raw, fs=44100, rms_ms=250):
    n_samples = max(3, int((rms_ms / 1000.0) * fs))
    audio_ac = audio_raw - np.median(audio_raw)
    env_rms = np.sqrt(np.maximum(0.0, scipy.ndimage.uniform_filter1d(audio_ac**2, size=n_samples)))
    return env_rms


def dist_2d(p1_x, p1_y, p2_x, p2_y):
    return np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)


def extraer_pulsos_18ch_canonico(
    session_dir, rms_ms=250, target_len=35, pre_pct=0.40, post_pct=0.60,
    outlier_contamination=0.08
):
    """
    Extrae exactamente las ventanas con el mismo corte de generador_pca_6canales.py:
      - Reloj Maestro Acústico: RMS 250 ms, offset dinámico t_wav_s - t_daq_s
      - Suavizado cinemático: 125 ms (0.125 * fps_video)
      - Ventana dinámica: [-0.40 * tau, +0.60 * tau] vinculada al BPM
      - 6 canales cinemáticos (mm) + primera derivada suave dq/dt + excursión diferencial
        pre-onset (d2q/delta_onset) conformando el tensor X de (N, 18, T=35), y también X_25
        de (N, 6, 25) compatible con RealtimeDecoderEngine.
    """
    path_pulsos = os.path.join(session_dir, "pulsos_metronomo.csv")
    path_tray = os.path.join(session_dir, "trayectoria_puntos.csv")
    path_audio = os.path.join(session_dir, "grabacion.csv")

    df_p = pd.read_csv(path_pulsos)
    df_princ = df_p[(df_p["principal"] == 1) & (df_p["idx"] > 0)].copy()
    df_tray = pd.read_csv(path_tray)

    df_audio = pd.read_csv(path_audio, usecols=[0, 1])
    t_audio = df_audio.iloc[:, 0].values
    audio_raw = df_audio.iloc[:, 1].values
    fs = 44100

    env_rms = calcular_envolvente_rms(audio_raw, fs=fs, rms_ms=rms_ms)

    picos_mic_daq = {}
    vocales_dict = {}
    for _, row in df_princ.iterrows():
        p_num = int(row["idx"])
        t_start = float(row["t_wav_s"])
        dur = float(row["intervalo_s"])
        palabra = str(row["palabra"]).lower().strip()
        if palabra not in VOCAL_TO_IDX:
            continue

        offset_audio_daq = (
            float(row["t_wav_s"]) - float(row["t_daq_s"])
            if (pd.notna(row.get("t_daq_s")) and pd.notna(row.get("t_wav_s")))
            else 5.200
        )
        if t_start + 0.2 >= t_audio[-1]:
            break

        t_end = min(t_start + dur, t_audio[-1])
        mask = (t_audio >= t_start + 0.15) & (t_audio < t_end)
        if not np.any(mask):
            continue

        sub_env = env_rms[mask]
        sub_t = t_audio[mask]
        t_mic_wav = sub_t[np.argmax(sub_env)]
        picos_mic_daq[p_num] = t_mic_wav - offset_audio_daq
        vocales_dict[p_num] = palabra

    # Escala interocular y canales como en generador_pca_6canales.py
    escala = 1.0
    if "apertura_vertical_mm" in df_tray.columns and "comisura_izq_x_px" in df_tray.columns:
        dx_com_px = np.abs(df_tray["comisura_der_x_px"].values - df_tray["comisura_izq_x_px"].values)
        com_mm = df_tray["ancho_comisuras_mm"].values
        valid_m = (dx_com_px > 10) & (com_mm > 10)
        if np.any(valid_m):
            escala = float(np.median(com_mm[valid_m] / dx_com_px[valid_m]))

    ap_raw = df_tray["apertura_vertical_mm"].values.copy()
    com_raw = df_tray["ancho_comisuras_mm"].values.copy()
    mej_raw = 0.5 * (
        dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["mejilla_izq_x_px"], df_tray["mejilla_izq_y_px"]) +
        dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["mejilla_der_x_px"], df_tray["mejilla_der_y_px"])
    ) * escala
    mal_raw = 0.5 * (
        dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["malar_izq_x_px"], df_tray["malar_izq_y_px"]) +
        dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["malar_der_x_px"], df_tray["malar_der_y_px"])
    ) * escala
    mas_raw = 0.5 * (
        dist_2d(df_tray["comisura_izq_x_px"], df_tray["comisura_izq_y_px"], df_tray["masetero_izq_x_px"], df_tray["masetero_izq_y_px"]) +
        dist_2d(df_tray["comisura_der_x_px"], df_tray["comisura_der_y_px"], df_tray["masetero_der_x_px"], df_tray["masetero_der_y_px"])
    ) * escala

    tiene_pose = all(col in df_tray.columns for col in ("yaw_deg", "pitch_deg", "roll_deg"))
    if tiene_pose:
        ap_raw, com_raw, mej_raw, mal_raw, mas_raw, pose_info = desacoplar_postura_craneal_en_reposo(
            ap_raw, com_raw, mej_raw, mal_raw, mas_raw,
            df_tray["yaw_deg"].values, df_tray["pitch_deg"].values, df_tray["roll_deg"].values
        )
        pose_centered = pose_info["P_centered"]
    else:
        pose_centered = np.zeros((len(df_tray), 3), dtype=np.float64)

    # Suavizado cinemático continuo de 125 ms
    fps_video = 1.0 / np.median(np.diff(df_tray["t_s"].values))
    win_cinem_frames = max(3, int(0.125 * fps_video))
    if win_cinem_frames % 2 == 0:
        win_cinem_frames += 1

    df_tray["ap_smooth"] = scipy.ndimage.uniform_filter1d(ap_raw, size=win_cinem_frames)
    df_tray["com_smooth"] = scipy.ndimage.uniform_filter1d(com_raw, size=win_cinem_frames)
    df_tray["mej_smooth"] = scipy.ndimage.uniform_filter1d(mej_raw, size=win_cinem_frames)
    df_tray["mal_smooth"] = scipy.ndimage.uniform_filter1d(mal_raw, size=win_cinem_frames)
    df_tray["mas_smooth"] = scipy.ndimage.uniform_filter1d(mas_raw, size=win_cinem_frames)

    # Percentiles globales de reposo como en generador_pca_6canales.py
    base_ap = float(np.percentile(df_tray["ap_smooth"].dropna(), 10))
    base_com = float(np.percentile(df_tray["com_smooth"].dropna(), 20))
    base_mej = float(np.percentile(df_tray["mej_smooth"].dropna(), 20))
    base_mal = float(np.percentile(df_tray["mal_smooth"].dropna(), 20))
    base_mas = float(np.percentile(df_tray["mas_smooth"].dropna(), 20))

    baseline_stats = {
        "base_ap": base_ap,
        "base_com": base_com,
        "base_mej": base_mej,
        "base_mal": base_mal,
        "base_mas": base_mas,
        "escala": float(escala)
    }

    # Ventana canónica 40-60 vinculada a tau de BPM (idéntica a generador_pca_6canales.py) para el Conformer 18ch (T=35)
    # y ventana fija de [-0.50 s, +0.50 s] (1000 ms) para X_6ch_25 compatible 1:1 con RealtimeDecoderEngine (pre_win_ms=500, post_win_ms=500)
    row_ref = df_princ.iloc[0] if len(df_princ) > 0 else df_p.iloc[0]
    bpm_val = float(row_ref.get("bpm", 24) or 24)
    tau_s = float(row_ref.get("intervalo_s", 60.0 / max(1.0, bpm_val)) or (60.0 / max(1.0, bpm_val)))
    pre_s = round(tau_s * pre_pct, 4)
    post_s = round(tau_s * post_pct, 4)
    t_grid_35 = np.linspace(-pre_s, post_s, target_len)
    t_grid_25 = np.linspace(-0.50, 0.50, 25)

    # Kernels derivativos suaves (evitan amplificar ruido de cuantización frame a frame)
    k_vel = np.array([-0.25, -0.50, 0.0, 0.50, 0.25], dtype=np.float32)
    k_acc = np.array([0.20, -0.10, -0.20, -0.10, 0.20], dtype=np.float32)

    X_6ch_25 = []
    X_18ch_35 = []
    P_poses = []
    Y_raw = []
    pulsos_raw = []

    # 6 canales invariantes escalados por distancia canónica de reposo (idéntico al hilo en vivo WebcamAudioLiveThread)
    c1_s, c2_s, c3_s, c4_s, c5_s, c6_s = calcular_canales_relativos_invariantes(
        df_tray["ap_smooth"].values,
        df_tray["com_smooth"].values,
        df_tray["mej_smooth"].values,
        df_tray["mal_smooth"].values,
        df_tray["mas_smooth"].values,
        baseline_stats=baseline_stats,
        aplicar_deadband=True,
        devolver_en_mm_canonicos=True
    )

    for p_num, t_mic in picos_mic_daq.items():
        group = df_tray[df_tray["pulso"] == p_num]
        if len(group) < 5:
            continue

        idx_g = group.index.values
        t_s = group["t_s"].values
        t_rel = t_s - t_mic

        c1 = c1_s[idx_g]
        c2 = c2_s[idx_g]
        c3 = c3_s[idx_g]
        c4 = c4_s[idx_g]
        c5 = c5_s[idx_g]
        c6 = c6_s[idx_g]

        q_35 = np.vstack([
            np.interp(t_grid_35, t_rel, c_arr, left=0.0, right=0.0)
            for c_arr in (c1, c2, c3, c4, c5, c6)
        ]).astype(np.float32)  # (6, 35)

        q_25 = np.vstack([
            np.interp(t_grid_25, t_rel, c_arr, left=0.0, right=0.0)
            for c_arr in (c1, c2, c3, c4, c5, c6)
        ]).astype(np.float32)  # (6, 25)

        # Excursión diferencial respecto a bordes locales (elimina postura tónica congelada a 40 BPM)
        edge_35 = 0.5 * (np.percentile(q_35[:, :6], 25, axis=1, keepdims=True) + np.percentile(q_35[:, -6:], 25, axis=1, keepdims=True))
        q_exc_35 = q_35 - edge_35

        # Primera derivada suave dq/dt y segunda derivada / curvatura en el núcleo
        dq_35 = scipy.ndimage.convolve1d(q_exc_35, k_vel, axis=1, mode="nearest")
        d2q_35 = scipy.ndimage.convolve1d(q_exc_35, k_acc, axis=1, mode="nearest") + 0.5 * q_exc_35

        tensor_18 = np.vstack([q_35, dq_35, d2q_35])  # (18, 35)
        X_6ch_25.append(q_25.reshape(-1))
        X_18ch_35.append(tensor_18)
        P_poses.append(np.mean(pose_centered[group.index.values], axis=0))
        Y_raw.append(vocales_dict[p_num])
        pulsos_raw.append(p_num)

    X_6ch_25 = np.array(X_6ch_25, dtype=np.float32)
    X_18ch_35 = np.array(X_18ch_35, dtype=np.float32)
    P_poses = np.array(P_poses, dtype=np.float64)
    Y_raw = np.array(Y_raw)
    pulsos_raw = np.array(pulsos_raw)

    # Purga por Isolation Forest (8%, idéntico a generador_pca_6canales.py)
    keep_indices = []
    for v in VOCALES_CANONICAS:
        idx_v = np.where(Y_raw == v)[0]
        if len(idx_v) == 0:
            continue
        sub_x = X_6ch_25[idx_v]
        if len(sub_x) >= 6 and outlier_contamination > 0:
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            preds = iso.fit_predict(sub_x)
            inliers = (preds == 1)
        else:
            inliers = np.ones(len(sub_x), dtype=bool)
        keep_indices.extend(idx_v[inliers])

    keep_indices = sorted(keep_indices)
    return (
        X_18ch_35[keep_indices],
        X_6ch_25[keep_indices],
        P_poses[keep_indices],
        np.array([VOCAL_TO_IDX[v] for v in Y_raw[keep_indices]], dtype=np.int64),
        Y_raw[keep_indices],
        pulsos_raw[keep_indices],
        baseline_stats
    )


class TemporalConvConformerNet(nn.Module):
    """
    Arquitectura de la Sección 4:
      - Entrada: Tensor X in R^{18 x 35} (6 trayectorias [x(t)] + [dx/dt] + [d2x/dt2])
      - 2 bloques Conv1D multi-escala (kernels de 3 y 7 muestras, 60 ms y 140 ms)
      - 1 bloque Multi-Head Self-Attention (4 cabezas, d_model=32) con máscara de atención
        concentrada en el núcleo del gesto
      - Cuello de botella ortogonal z in R^d (d=4 a 6).
    """
    def __init__(self, in_channels=18, seq_len=35, d_model=32, n_heads=4, latent_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Bloque 1 Multi-Escala (Kernels 3 y 7)
        self.b1_k3 = nn.Conv1d(in_channels, d_model // 2, kernel_size=3, padding=1)
        self.b1_k7 = nn.Conv1d(in_channels, d_model // 2, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(d_model)

        # Bloque 2 Multi-Escala (Kernels 3 y 7)
        self.b2_k3 = nn.Conv1d(d_model, d_model // 2, kernel_size=3, padding=1)
        self.b2_k7 = nn.Conv1d(d_model, d_model // 2, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(d_model)

        # Bloque Transformer / Multi-Head Self-Attention (4 cabezas, d_model=32)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)

        # Máscara de atención temporal alpha(t)
        self.temporal_gate = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

        # Cuello de botella ortogonal
        self.fc_enc = nn.Linear(d_model * 2, latent_dim, bias=False)
        self.fc_dec = nn.Linear(latent_dim, in_channels * seq_len)

        # Ventana gaussiana prior centrada en el núcleo articulatorio (evita colas coarticuladas)
        t_pos = torch.linspace(-1.0, 1.0, seq_len)
        core_prior = torch.exp(-2.5 * (t_pos ** 2)).view(1, seq_len, 1)
        self.register_buffer("core_prior", core_prior)

    def encode(self, x):
        h1 = torch.tanh(self.bn1(torch.cat([self.b1_k3(x), self.b1_k7(x)], dim=1)))
        h2 = torch.tanh(self.bn2(torch.cat([self.b2_k3(h1), self.b2_k7(h1)], dim=1)) + h1)

        h_seq = h2.transpose(1, 2)  # (B, 35, 32)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)
        h_seq = self.ln_attn(h_seq + attn_out)

        logits_a = self.temporal_gate(h_seq) + torch.log(self.core_prior + 1e-4)
        alpha = F.softmax(logits_a, dim=1)
        ctx_attn = torch.sum(alpha * h_seq, dim=1)
        ctx_core = torch.mean(h_seq[:, 9:26, :], dim=1)
        h_pool = torch.cat([ctx_attn, ctx_core], dim=-1)

        z = self.fc_enc(h_pool)
        return z, alpha.squeeze(-1)

    def forward(self, x):
        z, alpha = self.encode(x)
        x_hat = self.fc_dec(z).view(-1, self.in_channels, self.seq_len)
        return x_hat, z, alpha


def entrenar_conformer_18ch_fold(X_tr, X_te, latent_dim=4, epochs=120, lr=0.008):
    """
    Entrena el Autoencoder Temporal Conv + MHSA de forma 100% NO SUPERVISADA (Zero-Labels)
    exclusivamente en X_tr (sin fuga de datos hacia X_te), ponderando la reconstrucción
    en el núcleo del gesto y proyectando en conjunto con las características multiescala
    diferenciales.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    m_tr = np.mean(X_tr, axis=(0, 2), keepdims=True)
    s_tr = np.std(X_tr, axis=(0, 2), keepdims=True) + 0.20
    X_tr_n = (X_tr - m_tr) / s_tr
    X_te_n = (X_te - m_tr) / s_tr

    in_ch, seq_len = X_tr.shape[1], X_tr.shape[2]
    model = TemporalConvConformerNet(in_channels=in_ch, seq_len=seq_len, d_model=32, n_heads=4, latent_dim=latent_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    x_t = torch.tensor(X_tr_n, dtype=torch.float32)
    # Máscara temporal que prioriza reconstruir el núcleo del gesto (muestras 8 a 27) y las posiciones/velocidades
    w_time = torch.ones((1, in_ch, seq_len), dtype=torch.float32)
    w_time[:, :, :7] = 0.25
    w_time[:, :, 28:] = 0.25
    w_time[:, 12:, :] *= 0.50  # menor peso reconstructivo a d2q para no aprender ruido de cámara

    # Ancla espectral PCA sin etiquetas sobre el núcleo para estabilizar el subespacio inicial
    X_core_flat = (X_tr_n * w_time.numpy()).reshape(len(X_tr), -1)
    pca_init = PCA(n_components=latent_dim, random_state=42)
    z_pca_target = torch.tensor(pca_init.fit_transform(X_core_flat), dtype=torch.float32)
    z_pca_target = z_pca_target / (torch.std(z_pca_target, dim=0, keepdim=True) + 1e-5)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        x_hat, z, _ = model(x_t)
        loss_mse = torch.mean(w_time * ((x_hat - x_t) ** 2))
        loss_align = F.mse_loss(z, z_pca_target)
        W = model.fc_enc.weight
        eye_d = torch.eye(latent_dim)
        loss_ortho = torch.norm(torch.matmul(W, W.t()) - eye_d, p="fro") ** 2
        loss = loss_mse + 0.45 * loss_align + 0.15 * loss_ortho
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        z_tr, _ = model.encode(x_t)
        z_te, _ = model.encode(torch.tensor(X_te_n, dtype=torch.float32))
    return z_tr.numpy(), z_te.numpy()


def evaluar_bloque_completo(X_18, X_6_25, y_all, latent_dim=4):
    """
    Compara en igualdad de condiciones (sin fuga de datos en 5-Fold y en ajuste in-block):
      1) PCA 6ch (solo posición q)
      2) PCA 18ch Conformer-Features (q + dq/dt + d2q/dt2 ponderado en núcleo)
      3) Temporal Conv (k=3,7) + Multi-Head Self-Attention (4H, d_model=32)
    """
    N = len(X_18)
    X_6_flat = X_6_25.reshape(N, -1)

    # 1. PCA 6ch estándar
    Z6_all = PCA(n_components=latent_dim, random_state=42).fit_transform(X_6_flat)
    acc_in_6ch = LinearDiscriminantAnalysis().fit(Z6_all, y_all).score(Z6_all, y_all) * 100.0

    # 2. auto-encoder Conv+MHSA in-block
    Z_ae_all, _ = entrenar_conformer_18ch_fold(X_18, X_18, latent_dim=latent_dim, epochs=120, lr=0.008)
    acc_in_ae = LinearDiscriminantAnalysis().fit(Z_ae_all, y_all).score(Z_ae_all, y_all) * 100.0

    # 3. Validación Cruzada 5-Fold por bloques contiguos (Cero Fuga de Datos)
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    preds_6ch = np.zeros(N, dtype=int)
    preds_ae = np.zeros(N, dtype=int)

    for tr_idx, te_idx in skf.split(X_6_flat, y_all):
        # 6ch
        pca6 = PCA(n_components=latent_dim, random_state=42)
        z6_tr = pca6.fit_transform(X_6_flat[tr_idx])
        z6_te = pca6.transform(X_6_flat[te_idx])
        preds_6ch[te_idx] = LinearDiscriminantAnalysis().fit(z6_tr, y_all[tr_idx]).predict(z6_te)

        # Conv1D + MHSA (18ch)
        z_tr, z_te = entrenar_conformer_18ch_fold(X_18[tr_idx], X_18[te_idx], latent_dim=latent_dim, epochs=100, lr=0.008)
        preds_ae[te_idx] = LinearDiscriminantAnalysis().fit(z_tr, y_all[tr_idx]).predict(z_te)

    return {
        "acc_in_6ch": acc_in_6ch,
        "acc_cv_6ch": np.mean(preds_6ch == y_all) * 100.0,
        "acc_in_ae": acc_in_ae,
        "acc_cv_ae": np.mean(preds_ae == y_all) * 100.0,
    }


def actualizar_modelos_decodificador_tiempo_real(sessions):
    """
    Entrena y actualiza los modelos oficiales 4D, 3D y 2D del decodificador facial en tiempo real
    (modelos/campeon_geosup_conv_ortho_{4d,3d,2d}.pt) usando el motor de entrenamiento de
    decodificador_continuo_facial_6canales.py sobre los 4 bloques multicadencia (24, 30, 40 BPM + Invarianza),
    para que autoforge_daq_decodificador_facial.py funcione en vivo sobre el usuario actual.
    """
    from EMG_desarrollo.deep_learning.decodificador_continuo_facial_6canales import (
        entrenar_modelo_cinematico_autonomo, extraer_canales_y_pulsos_sesion
    )

    X_all_25, Y_all_str, P_all, pulsos_all = [], [], [], []
    base_stats_ref = None
    t_grid_25 = np.linspace(-0.5, 0.5, 25)

    for idx_b, (nombre, s_dir) in enumerate(sessions):
        datos_tr = extraer_canales_y_pulsos_sesion(s_dir, baseline_stats=None, usar_baseline_local=True, aplicar_deadband=False)
        if base_stats_ref is None:
            base_stats_ref = datos_tr['baseline_stats']
        df_princ_tr = datos_tr['df_princ']
        df_tray_tr = datos_tr['df_tray']
        t_cam_tr = datos_tr['t_cam']
        canales_tr = datos_tr['canales']
        S_kin_tr = datos_tr['S_kin']
        pose_c_tr = datos_tr['pose_centered']
        t_wav_env = datos_tr['t_wav_env']
        env_mic_raw = datos_tr['env_mic_raw']

        for _, row in df_princ_tr.iterrows():
            p_num = int(row["idx"])
            v_gt = str(row["palabra"]).upper().strip()
            if v_gt not in VOCALES:
                continue
            t_d = float(row["t_daq_s"])
            t_w = float(row["t_wav_s"])
            dur = float(row["intervalo_s"])
            offset = t_w - t_d
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

            # Agregar ventana centrada en GATE_HYBRID (cinemática pura, realineada internamente al pico cinemático)
            m_win = (t_cam_tr >= t_hyb - 0.6) & (t_cam_tr <= t_hyb + 0.6)
            vecs = [np.interp(t_grid_25, t_cam_tr[m_win] - t_hyb, ch_arr[m_win], left=0.0, right=0.0) for ch_arr in canales_tr]
            X_all_25.append(np.concatenate(vecs))
            Y_all_str.append(v_gt)
            P_all.append(np.mean(pose_c_tr[m_win], axis=0) if np.any(m_win) else np.zeros(3))
            pulsos_all.append(int(p_num) + idx_b * 1000)

    X_concat = np.vstack(X_all_25)
    P_concat = np.vstack(P_all)

    modelos_dirs = [
        os.path.join(_project_root, "modelos"),
        os.path.join(_project_root, "EMG_desarrollo", "modelos")
    ]
    for d in modelos_dirs:
        os.makedirs(d, exist_ok=True)

    print("\n" + "=" * 96)
    print(f"ENTRENANDO Y DESPLEGANDO MODELOS EN DECODIFICADOR FACIAL EN TIEMPO REAL (N={len(Y_all_str)} pulsos)")
    print("=" * 96)

    for dim in (4, 3, 2):
        ckpt, Z_tr = entrenar_modelo_cinematico_autonomo(
            X_concat, Y_all_str, latent_dim=dim, baseline_stats=base_stats_ref, P_poses=P_concat
        )
        # Evaluar precisión con el propio motor RealtimeDecoderEngine
        for m_dir in modelos_dirs:
            pt_path = os.path.join(m_dir, f"campeon_geosup_conv_ortho_{dim}d.pt")
            with open(pt_path, "wb") as f:
                pickle.dump(ckpt, f)
            np.savez_compressed(pt_path[:-3] + ".npz", **ckpt)
            df_proy = pd.DataFrame({
                "Pulso": pulsos_all,
                "Vocal": Y_all_str,
                "Z1": Z_tr[:, 0],
                "Z2": Z_tr[:, 1] if Z_tr.shape[1] > 1 else 0.0,
                "Z3": Z_tr[:, 2] if Z_tr.shape[1] > 2 else 0.0,
                "Z4": Z_tr[:, 3] if Z_tr.shape[1] > 3 else 0.0,
            })
            df_proy.to_csv(os.path.join(m_dir, f"proyecciones_campeon_geosup_conv_ortho_{dim}d.csv"), index=False)

        eng_test = RealtimeDecoderEngine(model_path=os.path.join(modelos_dirs[0], f"campeon_geosup_conv_ortho_{dim}d.pt"))
        preds, _, _ = eng_test.encode_and_predict_window(X_concat, return_rest_prob=False)
        y_gt_idx = np.array([VOCALES.index(v) for v in Y_all_str])
        acc_eng = float(np.mean(preds == y_gt_idx) * 100.0)
        print(f"  [OK] Modelo {dim}D desplegado en decodificador facial -> Exactitud Multicadencia: {acc_eng:.2f}% ({int(np.sum(preds == y_gt_idx))}/{len(y_gt_idx)})")


def main():
    sessions = [
        ("Landmarks (Nov 08)", os.path.join(_project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-11-08/SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02")),
        ("Bloque 1 (24 BPM)", os.path.join(_project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-11-10/SecuenciaContinua_FacialHiperarticulado24BPMDeBruijn_Sujeto1")),
        ("Bloque 2 (30 BPM)", os.path.join(_project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-11-10/SecuenciaContinua_FacialNatural30BPMJitter15DeBruijn_Sujeto1")),
        ("Bloque 3 (40 BPM)", os.path.join(_project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-11-10/SecuenciaContinua_FacialRapido40BPMJitter20DeBruijn_Sujeto1")),
        ("Bloque 4 (Invar.)", os.path.join(_project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-11-10/SecuenciaContinua_FacialInvarianzaDistancia24BPMDeBruijn_Sujeto1")),
    ]

    print("=" * 96)
    print("COMPARATIVA SIN FUGA DE DATOS: 6CH ESTÁTICO vs CONV1D MULTIESCALA (K=3,7) + MHSA (18CH DINÁMICOS)")
    print("=" * 96)

    for nombre, s_dir in sessions:
        X_18, X_6_25, _, y_idx, _, _, _ = extraer_pulsos_18ch_canonico(s_dir, rms_ms=250, target_len=35)
        r4 = evaluar_bloque_completo(X_18, X_6_25, y_idx, latent_dim=4)
        r6 = evaluar_bloque_completo(X_18, X_6_25, y_idx, latent_dim=6)
        print(
            f"{nombre:<20} (N={len(y_idx):3d}) | "
            f"PCA 6ch 4D: {r4['acc_in_6ch']:5.1f}% (CV:{r4['acc_cv_6ch']:5.1f}%) | "
            f"Conv+MHSA 18ch 4D: {r4['acc_in_ae']:5.1f}% (CV:{r4['acc_cv_ae']:5.1f}%) | "
            f"Conv+MHSA 18ch 6D: {r6['acc_in_ae']:5.1f}% (CV:{r6['acc_cv_ae']:5.1f}%)"
        )

    # Desplegar modelo entrenado sobre las 4 sesiones directamente en el decodificador facial
    actualizar_modelos_decodificador_tiempo_real(sessions)
    print("=" * 96)


if __name__ == "__main__":
    main()
