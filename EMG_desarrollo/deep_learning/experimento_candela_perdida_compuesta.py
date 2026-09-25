#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación de Pérdida Compuesta Dual-Head (RMS + beta*TKEO) en Candela 01/09 (Risorio).
Entrada: Envolvente RMS limpia (3 canales x 20 puntos).
Salida latente: Coordenadas Z en 2D.
Reconstrucción dual: Cabeza 1 -> RMS | Cabeza 2 -> TKEO.
Barrido de beta: [0.00, 0.02, 0.05, 0.10, 0.20].
Garantía: Agrupamiento canónico por sesión completa (PRUEBA1, PRUEBA2, PRUEBA3, PRUEBA4).
"""

import os
import sys
import glob
import time
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
emg_desarrollo = os.path.join(project_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

from deep_learning.pca_umap_clustering import generador_pca_umap as gpu
from deep_learning.motor_autoencoder_unificado import acondicionar_reposo_impedancia, extraer_sesion_agnostica

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}
COLORES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

# ==============================================================================
# 1. ARQUITECTURA DUAL-HEAD CONV ORTOGONAL
# ==============================================================================
class DualHeadConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2

        # Codificador compartido (recibe solo RMS limpia)
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh()

        self.fc1 = nn.Linear(c2 * time_pts, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, latent_dim, bias=False)

        # Cabeza Decodificadora 1: Reconstrucción de RMS
        self.dfc1_rms = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2_rms = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1_rms = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2_rms = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)

        # Cabeza Decodificadora 2: Reconstrucción de TKEO
        self.dfc1_tkeo = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2_tkeo = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1_tkeo = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2_tkeo = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)

        self.c2 = c2

        # Buffers de identidad para regularización ortogonal
        dims_necesarias = set()
        for layer in [self.fc1, self.fc2, self.dfc1_rms, self.dfc2_rms, self.dfc1_tkeo, self.dfc2_tkeo]:
            W = layer.weight
            dims_necesarias.add(min(W.shape[0], W.shape[1]))
        for conv in [self.conv1, self.conv2, self.deconv1_rms, self.deconv2_rms, self.deconv1_tkeo, self.deconv2_tkeo]:
            W_flat = conv.weight.view(conv.weight.shape[0], -1)
            dims_necesarias.add(min(W_flat.shape[0], W_flat.shape[1]))

        for d in dims_necesarias:
            self.register_buffer(f'_eye_{d}', torch.eye(d))

    def encode(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        return self.fc2(h3)

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        z = self.fc2(h3)

        # Rama RMS
        dh1_r = self.act(self.dfc1_rms(z))
        dh2_r = self.act(self.dfc2_rms(dh1_r)).view(dh1_r.shape[0], self.c2, self.time_pts)
        dh3_r = self.act(self.deconv1_rms(dh2_r))
        rec_rms = self.deconv2_rms(dh3_r).view(dh3_r.shape[0], self.in_channels, self.time_pts)

        # Rama TKEO
        dh1_t = self.act(self.dfc1_tkeo(z))
        dh2_t = self.act(self.dfc2_tkeo(dh1_t)).view(dh1_t.shape[0], self.c2, self.time_pts)
        dh3_t = self.act(self.deconv1_tkeo(dh2_t))
        rec_tkeo = self.deconv2_tkeo(dh3_t).view(dh3_t.shape[0], self.in_channels, self.time_pts)

        return rec_rms, rec_tkeo, z

    def loss_ortogonal(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1_rms, self.dfc2_rms, self.dfc1_tkeo, self.dfc2_tkeo]:
            W = layer.weight
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)

        for conv in [self.conv1, self.conv2, self.deconv1_rms, self.deconv2_rms, self.deconv1_tkeo, self.deconv2_tkeo]:
            W = conv.weight.view(conv.weight.shape[0], -1)
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)
        return loss

# ==============================================================================
# 2. CARGA DE DATOS SINCRONIZADOS (RMS Y TKEO)
# ==============================================================================
def cargar_datos_candela():
    base_dir = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos")
    tomas_0901 = sorted([
        os.path.relpath(p, base_dir) 
        for p in glob.glob(os.path.join(base_dir, "2026-09-01", "*_Candela")) 
        if os.path.isdir(p) and not os.path.basename(p).startswith("SecuenciaContinua") and "Prueba5" not in os.path.basename(p)
    ])

    print(f"[Carga] Extrayendo RMS para {len(tomas_0901)} tomas...")
    X_rms_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_dir, mediciones=tomas_0901,
        alpha_ruido=1.0, gate_ratio_ruido=0.0, smooth_ms=90, notch_q=2.0,
        target_len=20, modo_alineacion="Pico Volumen Micrófono", pre_pct=0.4, post_pct=0.6,
        canales_features=["canal_0", "canal_1", "canal_2"], aplicar_correccion_intersesion=True,
        tipo_envolvente="rms", lowpass_cutoff_hz=500.0, tipo_filtro_ruido="notch", highpass_cutoff_hz=20.0
    )

    print(f"[Carga] Extrayendo TKEO para {len(tomas_0901)} tomas...")
    X_tkeo_raw, _, _, _ = gpu.extraer_features_concatenadas(
        base_dir=base_dir, mediciones=tomas_0901,
        alpha_ruido=1.0, gate_ratio_ruido=0.0, smooth_ms=90, notch_q=2.0,
        target_len=20, modo_alineacion="Pico Volumen Micrófono", pre_pct=0.4, post_pct=0.6,
        canales_features=["canal_0", "canal_1", "canal_2"], aplicar_correccion_intersesion=True,
        tipo_envolvente="tkeo", lowpass_cutoff_hz=500.0, tipo_filtro_ruido="notch", highpass_cutoff_hz=20.0
    )

    X_rms = np.array(X_rms_raw, dtype=np.float32).reshape(-1, 3, 20)
    X_tkeo = np.array(X_tkeo_raw, dtype=np.float32).reshape(-1, 3, 20)
    Y = np.array(Y_raw)
    sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas_wins])

    # Acondicionamiento de reposo e impedancia (4 sesiones canónicas completas)
    print(f"[Acondicionamiento] Aplicando reposo basal e impedancia sobre {len(np.unique(sesiones))} sesiones canónicas...")
    X_rms_norm = acondicionar_reposo_impedancia(X_rms, sesiones, n_canales=3, n_pts_reposo=6)
    X_tkeo_norm = acondicionar_reposo_impedancia(X_tkeo, sesiones, n_canales=3, n_pts_reposo=6)

    # Purga única de outliers con Isolation Forest sobre RMS
    iso = IsolationForest(contamination=0.10, random_state=42)
    mask = (iso.fit_predict(X_rms_norm.reshape(len(X_rms_norm), -1)) == 1)

    X_rms_clean = X_rms_norm[mask]
    X_tkeo_clean = X_tkeo_norm[mask]
    Y_clean = Y[mask]

    print(f"[OK] Ventanas válidas post-purga: {len(Y_clean)} / {len(Y)}")
    return X_rms_clean, X_tkeo_clean, Y_clean

# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACIÓN POR VALOR DE BETA
# ==============================================================================
def evaluar_beta(X_rms, X_tkeo, Y, beta, epochs=350, lr=0.003, lambda_w=2.0, lambda_z=0.5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DualHeadConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, hidden_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tx_rms = torch.tensor(X_rms, dtype=torch.float32, device=device)
    tx_tkeo = torch.tensor(X_tkeo, dtype=torch.float32, device=device)
    N = len(tx_rms)
    eye_z = torch.eye(2, device=device)

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        rec_rms, rec_tkeo, z = model(tx_rms)

        mse_r = nn.functional.mse_loss(rec_rms, tx_rms)
        mse_t = nn.functional.mse_loss(rec_tkeo, tx_tkeo)

        z_cent = z - z.mean(dim=0, keepdim=True)
        cov_z = (z_cent.T @ z_cent) / max(1, N - 1)
        loss_z = torch.norm(cov_z - eye_z, p='fro') ** 2

        loss_w = model.loss_ortogonal()

        total_loss = mse_r + beta * mse_t + lambda_w * loss_w + lambda_z * loss_z
        total_loss.backward()
        optimizer.step()

    dur = time.time() - t0

    # Evaluación GMM 100% no supervisada
    model.eval()
    with torch.no_grad():
        _, _, z_final = model(tx_rms)
        Z = z_final.cpu().numpy()

    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    pred_clusters = gmm.fit_predict(Z)

    contingency = np.zeros((5, 5))
    for i_v, v in enumerate(VOCALES):
        for j_c in range(5):
            contingency[j_c, i_v] = np.sum((Y == v) & (pred_clusters == j_c))

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    c2v = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}
    pred_vocal_idx = np.array([c2v[c] for c in pred_clusters])
    y_true_idx = np.array([VOCAL_TO_IDX[v] for v in Y])

    acc_global = accuracy_score(y_true_idx, pred_vocal_idx) * 100.0
    cm = confusion_matrix(y_true_idx, pred_vocal_idx, labels=range(5))

    acc_vocales = {}
    for i_v, v in enumerate(VOCALES):
        tot = np.sum(y_true_idx == i_v)
        acc_vocales[v] = (cm[i_v, i_v] / tot * 100.0) if tot > 0 else 0.0

    sil = silhouette_score(Z, pred_vocal_idx)
    db = davies_bouldin_score(Z, pred_vocal_idx)

    # Separación /o/ vs /u/
    mask_ou = np.isin(y_true_idx, [VOCAL_TO_IDX['O'], VOCAL_TO_IDX['U']])
    cm_ou = confusion_matrix(y_true_idx[mask_ou], pred_vocal_idx[mask_ou], labels=[VOCAL_TO_IDX['O'], VOCAL_TO_IDX['U']])
    acc_ou = np.trace(cm_ou) / max(1, np.sum(cm_ou)) * 100.0

    # Separación /e/ vs /i/
    mask_ei = np.isin(y_true_idx, [VOCAL_TO_IDX['E'], VOCAL_TO_IDX['I']])
    cm_ei = confusion_matrix(y_true_idx[mask_ei], pred_vocal_idx[mask_ei], labels=[VOCAL_TO_IDX['E'], VOCAL_TO_IDX['I']])
    acc_ei = np.trace(cm_ei) / max(1, np.sum(cm_ei)) * 100.0

    return {
        'beta': beta,
        'acc_global': acc_global,
        'acc_vocales': acc_vocales,
        'acc_ou': acc_ou,
        'acc_ei': acc_ei,
        'sil': sil,
        'db': db,
        'Z': Z,
        'Y': Y,
        'cm': cm,
        'dur': dur
    }

# ==============================================================================
# 4. EJECUCIÓN PRINCIPAL
# ==============================================================================
def main():
    out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_candela_perdida_compuesta")
    os.makedirs(out_dir, exist_ok=True)

    X_rms, X_tkeo, Y = cargar_datos_candela()

    betas = [0.00, 0.02, 0.05, 0.10, 0.20]
    resultados = []

    print("\n" + "="*80)
    print(f"BARRIDO DE BETA EN PÉRDIDA COMPUESTA DUAL (RMS + beta*TKEO) - CANDELA 01/09")
    print("="*80)

    for b in betas:
        res = evaluar_beta(X_rms, X_tkeo, Y, beta=b, epochs=350, lr=0.003)
        resultados.append(res)
        desglose_str = " | ".join([f"/{v.lower()}/: {res['acc_vocales'][v]:.1f}%" for v in VOCALES])
        print(f"Beta = {b:0.2f} -> Global: {res['acc_global']:.2f}% | Sil: {res['sil']:+.3f} | {desglose_str} | /o/-/u/: {res['acc_ou']:.1f}% | /e/-/i/: {res['acc_ei']:.1f}% ({res['dur']:.1f}s)")

    # Gráfico comparativo de barrido
    fig, axes = plt.subplots(1, len(betas), figsize=(4 * len(betas), 4), constrained_layout=True)
    for idx, (b, res) in enumerate(zip(betas, resultados)):
        ax = axes[idx]
        Z = res['Z']
        Y_lbl = res['Y']
        for v in VOCALES:
            m = (Y_lbl == v)
            ax.scatter(Z[m, 0], Z[m, 1], c=COLORES[v], label=f"/{v.lower()}/", s=25, alpha=0.75)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
        ax.set_title(f"Beta = {b:0.2f}: Exactitud {res['acc_global']:.1f}%\n/a/: {res['acc_vocales']['A']:.1f}% | /o/-/u/: {res['acc_ou']:.1f}%", fontsize=9, fontweight='bold')
        ax.set_xlabel("Z1", fontsize=8)
        if idx == 0:
            ax.set_ylabel("Z2", fontsize=8)
            ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, linestyle=':', alpha=0.5)

    fig_path = os.path.join(out_dir, "comparativa_candela_betas_tkeo.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"\n[OK] Gráfico comparativo guardado en: {fig_path}")

if __name__ == '__main__':
    main()
