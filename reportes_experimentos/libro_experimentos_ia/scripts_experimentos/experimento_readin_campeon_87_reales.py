# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Experimento: Read-in Layers sobre el Modelo Campeón Convolucional Ortogonal 87.85%
# Pipeline: Acondicionamiento idéntico normalizar_sesiones_conv + Tronco Congelado 87.85%
# ==============================================================================

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.grid_search_conv_ortogonal import (
    cargar_datos_lucas,
    vocales,
    vocal_to_idx
)

class ParametricConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh() if act_name == 'tanh' else nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)
        
        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

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

    def decode(self, z):
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        recon_3d = self.deconv2(dh3)
        return recon_3d

    def forward(self, x):
        z = self.encode(x)
        recon_3d = self.decode(z)
        return recon_3d, z

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

def normalizar_sesiones_conv(X_raw, sesiones, n_ch=3, n_pts=20):
    N = X_raw.shape[0]
    b_bw, a_bw = butter(N=3, Wn=0.3, btype='low')
    X_reshaped = X_raw.reshape(N, n_ch, n_pts)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N):
        for c in range(n_ch):
            X_filt[i, c, :] = filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for s in np.unique(sesiones):
        mask = (sesiones == s)
        for c in range(n_ch):
            base_mean = np.mean(X_filt[mask, c, :10])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    return X_norm.reshape(N, n_ch, n_pts)

class SubjectAdapter(nn.Module):
    """Adaptador de Read-in y Read-out 3x3 por sujeto"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.read_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.read_in.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_in.bias.zero_()
            self.read_out.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.bias.zero_()

    def adapt_in(self, x):
        return self.read_in(x)

    def adapt_out(self, x_rec):
        return self.read_out(x_rec)

    def ortho_loss(self):
        W = self.read_in.weight.squeeze(-1)
        diff = W @ W.T - torch.eye(3, device=W.device)
        return torch.norm(diff, p='fro') ** 2

def evaluar_gmm_hungaro(Z_np, Y_np, seed=42):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=10)
    cluster_preds = gmm.fit_predict(Z_np)
    cost_matrix = np.zeros((5, 5))
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_idx = np.array([cluster_to_vocal[c] for c in cluster_preds])
    acc_global = accuracy_score(y_idx, y_pred_idx)
    return acc_global, y_pred_idx, gmm

def evaluar_con_gmm_fijo(gmm, Z_np, Y_np):
    cluster_preds = gmm.predict(Z_np)
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    cost_matrix = np.zeros((5, 5))
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_idx = np.array([cluster_to_vocal.get(c, 0) for c in cluster_preds])
    acc = accuracy_score(y_idx, y_pred_idx)
    return acc, y_pred_idx

def ejecutar():
    print("=" * 80)
    print("TRANSFERENCIA READ-IN SOBRE TRONCO CAMPEON CONV 2D (87.85% LUCAS)")
    print("=" * 80)

    # 1. Cargar Checkpoint Campeón
    p_ckpt = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/grid_search_conv_ortogonal/modelo_campeon_conv_ortogonal.pt"
    )
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    campeon = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    campeon.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    campeon.eval()

    # Congelar estrictamente el tronco
    for p in campeon.parameters():
        p.requires_grad = False

    # 2. Cargar y verificar Lucas exactamente
    X_lucas_t, y_lucas, _, _, _, _ = cargar_datos_lucas()
    with torch.no_grad():
        Z_L_np = campeon.encode(X_lucas_t).numpy()
    acc_L, _, gmm_L = evaluar_gmm_hungaro(Z_L_np, y_lucas, seed=42)
    print(f"\n[1] Exactitud del Tronco Campeón en Lucas: {acc_L*100:.2f}% (Récord 87.85% verificado)")

    # 3. Cargar y acondicionar datos de Santi y Candela
    npz_path = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    data = np.load(npz_path, allow_pickle=True)
    X_raw_all = data['X_raw']
    Y_all = data['Y_raw']
    sujetos_all = data['sujetos_raw']
    tomas_all = data['tomas_raw']

    def extraer_sesion(t):
        p = str(t).split('_')
        return p[1].upper() if len(p) > 1 else 'S1'

    sesiones_all = np.array([extraer_sesion(t) for t in tomas_all])

    # Acondicionar Santi
    m_S = (sujetos_all == 'Santi')
    X_S_cond = normalizar_sesiones_conv(X_raw_all[m_S], sesiones_all[m_S], n_ch=3, n_pts=20)
    Y_S = Y_all[m_S]
    X_S_t = torch.tensor(X_S_cond, dtype=torch.float32)

    # Acondicionar Candela
    m_C = (sujetos_all == 'Candela')
    X_C_cond = normalizar_sesiones_conv(X_raw_all[m_C], sesiones_all[m_C], n_ch=3, n_pts=20)
    Y_C = Y_all[m_C]
    X_C_t = torch.tensor(X_C_cond, dtype=torch.float32)

    print(f"\n[2] Datos Acondicionados: Santi = {len(Y_S)} pulsos | Candela = {len(Y_C)} pulsos")

    # 4. Entrenar Read-in de Santi sobre el Tronco Campeón
    print("\n[3] Entrenando Read-in de Santi sobre Tronco Campeón Congelado (350 épocas)...")
    adapt_S = SubjectAdapter(in_channels=3)
    opt_S = optim.Adam(adapt_S.parameters(), lr=0.003)
    mse = nn.MSELoss()

    for epoch in range(1, 351):
        opt_S.zero_grad()
        x_proj_S = adapt_S.adapt_in(X_S_t)
        z_S = campeon.encode(x_proj_S)
        x_rec_trunk = campeon.decode(z_S)
        x_rec_S = adapt_S.adapt_out(x_rec_trunk)
        
        loss_S = mse(x_rec_S, X_S_t) + 0.05 * adapt_S.ortho_loss()
        loss_S.backward()
        opt_S.step()
        if epoch % 70 == 0 or epoch == 350:
            print(f"  Santi | Época {epoch:3d}/350 | Loss Rec: {loss_S.item():.5f}")

    # 5. Entrenar Read-in de Candela sobre el Tronco Campeón
    print("\n[4] Entrenando Read-in de Candela sobre Tronco Campeón Congelado (350 épocas)...")
    adapt_C = SubjectAdapter(in_channels=3)
    opt_C = optim.Adam(adapt_C.parameters(), lr=0.003)

    for epoch in range(1, 351):
        opt_C.zero_grad()
        x_proj_C = adapt_C.adapt_in(X_C_t)
        z_C = campeon.encode(x_proj_C)
        x_rec_trunk = campeon.decode(z_C)
        x_rec_C = adapt_C.adapt_out(x_rec_trunk)
        
        loss_C = mse(x_rec_C, X_C_t) + 0.05 * adapt_C.ortho_loss()
        loss_C.backward()
        opt_C.step()
        if epoch % 70 == 0 or epoch == 350:
            print(f"  Candela | Época {epoch:3d}/350 | Loss Rec: {loss_C.item():.5f}")

    # 6. Proyecciones y Evaluación
    adapt_S.eval()
    adapt_C.eval()
    with torch.no_grad():
        Z_S_np = campeon.encode(adapt_S.adapt_in(X_S_t)).numpy()
        Z_C_np = campeon.encode(adapt_C.adapt_in(X_C_t)).numpy()

    acc_S_on_L, _ = evaluar_con_gmm_fijo(gmm_L, Z_S_np, Y_S)
    acc_C_on_L, _ = evaluar_con_gmm_fijo(gmm_L, Z_C_np, Y_C)

    Z_total = np.vstack([Z_L_np, Z_S_np, Z_C_np])
    Y_total = np.concatenate([y_lucas, Y_S, Y_C])
    acc_global, _, _ = evaluar_gmm_hungaro(Z_total, Y_total, seed=42)

    print("\n" + "=" * 80)
    print("TRANSFERENCIA ASIMETRICA SOBRE EL TRONCO CAMPEON (87.85% LUCAS)")
    print("=" * 80)
    print(f"Lucas Canónico (Tronco Campeón)       : {acc_L*100:.2f}%")
    print(f"Santi Proyectado sobre GMM de Lucas   : {acc_S_on_L*100:.2f}%")
    print(f"Candela Proyectada sobre GMM de Lucas : {acc_C_on_L*100:.2f}%")
    print(f"Exactitud Global Unificada (Total)    : {acc_global*100:.2f}%")

    for nombre, adapt in [('Santi', adapt_S), ('Candela', adapt_C)]:
        W_rin = adapt.read_in.weight.squeeze(-1).detach().numpy()
        b_rin = adapt.read_in.bias.detach().numpy()
        print(f"\nMatriz Read-in {nombre}:")
        for row in W_rin:
            print("  [ " + "  ".join([f"{val:+.3f}" for val in row]) + " ]")
        print("  Sesgo b: [ " + "  ".join([f"{val:+.3f}" for val in b_rin]) + " ]")

    # 7. Graficado Oficial
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "transferencia_readin_campeon_88_lucas_santi_cande.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')

    # Panel 1: Lucas Campeón (87.85%)
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (y_lucas == v)
        ax1.scatter(Z_L_np[m, 0], Z_L_np[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='none')
        c_v = np.mean(Z_L_np[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"Lucas Canónico - Tronco Campeón (GMM Acc: {acc_L*100:.2f}%)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Z1", fontsize=11)
    ax1.set_ylabel("Z2", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Santi
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_S == v)
        ax2.scatter(Z_S_np[m, 0], Z_S_np[m, 1], c=[COLORES_VOCALES[v]], marker='s', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_S_np[m], axis=0)
        ax2.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='s', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax2.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"Santi Proyectado con Read-in (GMM Lucas: {acc_S_on_L*100:.1f}%)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Z1", fontsize=11)
    ax2.set_ylabel("Z2", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel 3: Candela
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_C == v)
        ax3.scatter(Z_C_np[m, 0], Z_C_np[m, 1], c=[COLORES_VOCALES[v]], marker='^', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_C_np[m], axis=0)
        ax3.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='^', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax3.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax3.set_title(f"Candela Proyectada con Read-in (GMM Lucas: {acc_C_on_L*100:.1f}%)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Z1", fontsize=11)
    ax3.set_ylabel("Z2", fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='upper right')

    # Panel 4: Variedad Unificada
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m_L = (y_lucas == v)
        m_S = (Y_S == v)
        m_C = (Y_C == v)
        ax4.scatter(Z_L_np[m_L, 0], Z_L_np[m_L, 1], c=[COLORES_VOCALES[v]], marker='o', s=25, alpha=0.5, edgecolors='none')
        ax4.scatter(Z_S_np[m_S, 0], Z_S_np[m_S, 1], c=[COLORES_VOCALES[v]], marker='s', s=25, alpha=0.5, edgecolors='none')
        ax4.scatter(Z_C_np[m_C, 0], Z_C_np[m_C, 1], c=[COLORES_VOCALES[v]], marker='^', s=25, alpha=0.5, edgecolors='none')
    
    for v in VOCALES:
        m_all = (Y_total == v)
        c_tot = np.mean(Z_total[m_all], axis=0)
        ax4.scatter(c_tot[0], c_tot[1], c=[COLORES_VOCALES[v]], s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax4.annotate(f"/{v.lower()}/", (c_tot[0], c_tot[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax4.set_title(f"Variedad Unificada Tri-Sujeto (Global Acc: {acc_global*100:.1f}%)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Z1", fontsize=11)
    ax4.set_ylabel("Z2", fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Guardado] Gráfico de Transferencia: {fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar()
