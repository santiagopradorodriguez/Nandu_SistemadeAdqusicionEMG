#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Barrido de Parametros en Lucas con Ventana Fisiologica 40/60
# Modos: Conv Ortogonal (2D y 3D) y MLP Ortogonal sin correccion SO(2) (2D y 3D)
# Metrica: Exactitud GMM No Supervisada, Separacion de Pares y Balance Multiclase
# ==============================================================================

import os
import sys
import json
import time
import re
import argparse
import itertools
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
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
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.pca_umap_clustering import generador_pca_umap as gpu

vocales = ['A', 'E', 'I', 'O', 'U']
vocal_to_idx = {v: i for i, v in enumerate(vocales)}
colores_oficiales = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

out_dir_base = os.path.join(
    project_root,
    "EMG_desarrollo/resultados/grid_search_lucas_ventana4060"
)
os.makedirs(out_dir_base, exist_ok=True)


# ==============================================================================
# 1. ARQUITECTURAS PARAMETRIZABLES (CONV 1D Y MLP ORTOGONAL)
# ==============================================================================

class ParametricConvOrthogonalAE(nn.Module):
    """
    Autoencoder Convolucional 1D con regularizacion ortogonal en capas convolucionales
    y lineales, parametrizable para 2D o 3D.
    """
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        
        if act_name == 'tanh':
            self.act = nn.Tanh()
        elif act_name == 'gelu':
            self.act = nn.GELU()
        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)
        
        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

        # Registro de buffers para identidades precalculadas
        dims_necesarias = set()
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            dims_necesarias.add(min(W.shape[0], W.shape[1]))
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
            W_flat = conv.weight.view(conv.weight.shape[0], -1)
            dims_necesarias.add(min(W_flat.shape[0], W_flat.shape[1]))
            
        for d in dims_necesarias:
            self.register_buffer(f'_eye_{d}', torch.eye(d))
        self.register_buffer('_eye_latent', torch.eye(latent_dim))

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
        
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        recon_3d = self.deconv2(dh3)
        recon_flat = recon_3d.view(recon_3d.shape[0], -1)
        return recon_flat, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)
            
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
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


class ParametricMLPOrthogonalAE(nn.Module):
    """
    Autoencoder Totalmente Conexo (MLP) Ortogonal Simetrico sin sesgo:
    D -> h1 -> h2 -> latent_dim (2 o 3) -> h2 -> h1 -> D
    Apto para evaluacion en espacio latente nativo sin correccion SO(2).
    """
    def __init__(self, input_dim=60, hidden_dims=(32, 16), latent_dim=2, act_name='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        h1, h2 = hidden_dims
        
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, latent_dim, bias=False)
        
        if act_name == 'tanh':
            self.act = nn.Tanh()
        elif act_name == 'gelu':
            self.act = nn.GELU()
        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()
            
        self.dfc1 = nn.Linear(latent_dim, h2, bias=False)
        self.dfc2 = nn.Linear(h2, h1, bias=False)
        self.dfc3 = nn.Linear(h1, input_dim, bias=False)

        dims_necesarias = set()
        for layer in [self.fc1, self.fc2, self.fc3, self.dfc1, self.dfc2, self.dfc3]:
            W = layer.weight
            dims_necesarias.add(min(W.shape[0], W.shape[1]))
            
        for d in dims_necesarias:
            self.register_buffer(f'_eye_{d}', torch.eye(d))
        self.register_buffer('_eye_latent', torch.eye(latent_dim))

    def encode(self, x):
        if x.dim() == 3:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        h1 = self.act(self.fc1(x_flat))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x):
        if x.dim() == 3:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        h1 = self.act(self.fc1(x_flat))
        h2 = self.act(self.fc2(h1))
        z = self.fc3(h2)
        
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        recon_flat = self.dfc3(dh2)
        return recon_flat, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.fc3, self.dfc1, self.dfc2, self.dfc3]:
            W = layer.weight
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
# 2. EXTRACCION Y CARGA DE DATOS DE LUCAS CON VENTANA 40/60
# ==============================================================================

def extraer_sesion_agnostica(toma_str):
    s = str(toma_str).strip()
    parts = s.split('_')
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

def acondicionar_reposo_impedancia(X_array, sesiones, n_canales=3, n_pts_reposo=6):
    N, n_canales, n_pts = X_array.shape
    b, a = butter(N=3, Wn=0.3, btype='low')
    X_filt = np.zeros_like(X_array)
    for i in range(N):
        for c in range(n_canales):
            X_filt[i, c, :] = filtfilt(b, a, X_array[i, c, :])

    unique_ses = np.unique(sesiones)
    X_norm = np.zeros_like(X_filt)
    pts_base = max(1, min(n_pts_reposo, n_pts // 2))

    for s in unique_ses:
        mask = (sesiones == s)
        for c in range(n_canales):
            base_mean = np.mean(X_filt[mask, c, :pts_base])
            base_p95 = np.percentile(X_filt[mask, c, :], 95)
            scale = max(base_p95 - base_mean, 1e-4)
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / scale

    return X_norm

def cargar_o_extraer_lucas_4060(cache_dir=out_dir_base):
    npz_cache = os.path.join(cache_dir, "dataset_lucas_ventana4060.npz")
    if os.path.exists(npz_cache):
        print(f"[Caché] Cargando dataset de Lucas previamente extraído con ventana 40/60:")
        print(f"        {npz_cache}")
        data = np.load(npz_cache, allow_pickle=True)
        X_clean = data['X_clean']
        Y_clean = data['Y_clean']
        print(f"        Muestras disponibles: {len(Y_clean)}")
        return torch.tensor(X_clean.reshape(len(X_clean), -1), dtype=torch.float32), Y_clean

    base_lucas = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-07-10")
    tomas_todas = sorted([d for d in os.listdir(base_lucas) if os.path.isdir(os.path.join(base_lucas, d)) and d != "UMBRALES"])
    tomas_lucas = [t for t in tomas_todas if t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']]
    
    print(f"\n[Extracción Lucas] Extrayendo {len(tomas_lucas)} tomas con ventana asimétrica 40% pre / 60% post...")
    X_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_lucas,
        mediciones=tomas_lucas,
        alpha_ruido=1.0,
        gate_ratio_ruido=0.0,
        smooth_ms=90,
        notch_q=2.0,
        target_len=20,
        modo_alineacion="Pico Volumen Micrófono",
        pre_pct=0.40,
        post_pct=0.60,
        canales_features=["canal_0", "canal_1", "canal_2"],
        aplicar_correccion_intersesion=True,
        tipo_envolvente="rms",
        lowpass_cutoff_hz=500.0,
        tipo_filtro_ruido="notch",
        highpass_cutoff_hz=20.0
    )

    X_arr = np.array(X_raw, dtype=np.float32).reshape(-1, 3, 20)
    Y_arr = np.array(Y_raw)
    sesiones_arr = np.array([extraer_sesion_agnostica(t) for t in tomas_wins])

    print(f"  [Acondicionamiento] Aplicando corrección de reposo basal e impedancia inter-sesión ({len(np.unique(sesiones_arr))} sesiones)...")
    X_arr = acondicionar_reposo_impedancia(X_arr, sesiones_arr, n_canales=3, n_pts_reposo=6)

    print(f"  [Purga] Aplicando Isolation Forest (10%) sobre {len(X_arr)} ventanas...")
    X_flat = X_arr.reshape(len(X_arr), -1)
    iso = IsolationForest(contamination=0.10, random_state=42)
    mask_inliers = (iso.fit_predict(X_flat) == 1)
    X_clean = X_arr[mask_inliers]
    Y_clean = Y_arr[mask_inliers]
    print(f"  [OK] Ventanas válidas post-purga: {len(X_clean)} / {len(X_arr)}")

    counts = {v: int(np.sum(Y_clean == v)) for v in vocales}
    print(f"  [Desglose] " + " | ".join([f"/{v.lower()}/: {counts[v]}" for v in vocales]))

    np.savez_compressed(
        npz_cache,
        X_clean=X_clean,
        Y_clean=Y_clean,
        sesiones=sesiones_arr[mask_inliers]
    )
    print(f"  [Caché Guardada] {npz_cache}")
    return torch.tensor(X_clean.reshape(len(X_clean), -1), dtype=torch.float32), Y_clean


# ==============================================================================
# 3. EVALUACION DE UNA CONFIGURACION (GMM CON ASIGNACION HUNGARA)
# ==============================================================================

def evaluar_configuracion(tipo_modelo, cfg, X_t, y_true, device, latent_dim=2, seed=100, epochs=350):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_ch, n_pts = 3, 20
    input_dim = n_ch * n_pts
    N = X_t.shape[0]
    X_dev = X_t.to(device)

    if tipo_modelo == 'conv':
        model = ParametricConvOrthogonalAE(
            in_channels=n_ch,
            time_pts=n_pts,
            conv_channels=cfg['channels'],
            kernel_size=cfg['kernel_size'],
            latent_dim=latent_dim,
            act_name=cfg['act']
        ).to(device)
    else: # 'mlp'
        model = ParametricMLPOrthogonalAE(
            input_dim=input_dim,
            hidden_dims=cfg['hidden_dims'],
            latent_dim=latent_dim,
            act_name=cfg['act']
        ).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    I_lat = torch.eye(latent_dim, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        recon, z = model(X_dev)
        loss_recon = nn.functional.mse_loss(recon, X_dev)
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / max(1, N - 1)
        loss_z = torch.sum((cov_z - I_lat) ** 2)
        loss = loss_recon + cfg['lambda_w'] * loss_w + cfg['lambda_z'] * loss_z
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        recon_final, z_final = model(X_dev)
        mse_final = nn.functional.mse_loss(recon_final, X_dev).item()
        Z = z_final.cpu().numpy()

    # Ajuste GMM No Supervisado
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
    pred_raw = gmm.fit_predict(Z)
    
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(vocales):
        for j in range(5):
            contingency[j, i] = np.sum((y_true == vl) & (pred_raw == j))
            
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    
    pred_idx = np.array([cluster_to_vocal[c] for c in pred_raw])
    y_idx = np.array([vocal_to_idx[v] for v in y_true])
    acc_global = accuracy_score(y_idx, pred_idx) * 100.0

    # Desglose por vocal
    acc_por_vocal = {}
    for idx_v, vl in enumerate(vocales):
        mask_v = (y_idx == idx_v)
        if np.sum(mask_v) > 0:
            acc_por_vocal[vl] = float(np.mean(pred_idx[mask_v] == idx_v) * 100.0)
        else:
            acc_por_vocal[vl] = 0.0

    # Separacion de pares criticos
    mask_ou = np.isin(y_true, ['O', 'U'])
    acc_ou = float(accuracy_score(y_idx[mask_ou], pred_idx[mask_ou]) * 100.0) if np.sum(mask_ou) > 0 else 0.0

    mask_ei = np.isin(y_true, ['E', 'I'])
    acc_ei = float(accuracy_score(y_idx[mask_ei], pred_idx[mask_ei]) * 100.0) if np.sum(mask_ei) > 0 else 0.0

    # Metricas de agrupamiento
    try:
        sil = float(silhouette_score(Z, y_true))
    except Exception:
        sil = 0.0
    try:
        db = float(davies_bouldin_score(Z, y_true))
    except Exception:
        db = 99.0

    res = {
        'acc_global': acc_global,
        'acc_A': acc_por_vocal['A'],
        'acc_E': acc_por_vocal['E'],
        'acc_I': acc_por_vocal['I'],
        'acc_O': acc_por_vocal['O'],
        'acc_U': acc_por_vocal['U'],
        'acc_ou': acc_ou,
        'acc_ei': acc_ei,
        'min_vocal': min(acc_por_vocal.values()),
        'silhouette': sil,
        'davies_bouldin': db,
        'mse': mse_final
    }
    return res, model, Z


# ==============================================================================
# 4. GRAFICADO DEL MODELO CAMPEON POR MODO
# ==============================================================================

def graficar_campeon(nombre_modo, cfg, res, Z, y_true, p_png, latent_dim=2):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 6), dpi=150)
    
    if latent_dim == 3:
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.set_facecolor('#0B0C10')
        for vl in vocales:
            mask = (y_true == vl)
            ax.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                       c=colores_oficiales[vl], label=f"/{vl.lower()}/ ({res[f'acc_{vl}']:.1f}%)",
                       alpha=0.8, edgecolors='none', s=40)
        ax.set_title(f"{nombre_modo}: Espacio Latente 3D", color='#66FCF1', fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Z1", color='#C5C6C7')
        ax.set_ylabel("Z2", color='#C5C6C7')
        ax.set_zlabel("Z3", color='#C5C6C7')
    else:
        ax = fig.add_subplot(1, 2, 1)
        ax.set_facecolor('#0B0C10')
        for vl in vocales:
            mask = (y_true == vl)
            ax.scatter(Z[mask, 0], Z[mask, 1],
                       c=colores_oficiales[vl], label=f"/{vl.lower()}/ ({res[f'acc_{vl}']:.1f}%)",
                       alpha=0.8, edgecolors='none', s=45)
        ax.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
        ax.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
        ax.set_title(f"{nombre_modo}: Espacio Latente 2D", color='#66FCF1', fontsize=12, fontweight='bold')
        ax.set_xlabel("Coordenada Z1", color='#C5C6C7')
        ax.set_ylabel("Coordenada Z2", color='#C5C6C7')
        ax.legend(frameon=True, facecolor='#1F2833', edgecolor='none')
        ax.grid(True, color='#1F2833', linestyle=':', alpha=0.6)

    # Panel de metricas y desglose
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#0B0C10')
    ax2.axis('off')

    info_text = (
        f"CONFIGURACION GANADORA: {nombre_modo.upper()}\n"
        f"--------------------------------------------------\n"
        f"Exactitud Global GMM: {res['acc_global']:.2f}%\n"
        f"Separacion /o/ vs /u/: {res['acc_ou']:.2f}%\n"
        f"Separacion /e/ vs /i/: {res['acc_ei']:.2f}%\n"
        f"Piso Minimo por Vocal: {res['min_vocal']:.2f}%\n"
        f"Coeficiente de Silueta: {res['silhouette']:+.3f}\n"
        f"Indice Davies-Bouldin: {res['davies_bouldin']:.2f}\n"
        f"Error Reconstruccion MSE: {res['mse']:.5f}\n\n"
        f"DESGLOSE POR VOCAL:\n"
        f"  /a/: {res['acc_A']:.1f}%\n"
        f"  /e/: {res['acc_E']:.1f}%\n"
        f"  /i/: {res['acc_I']:.1f}%\n"
        f"  /o/: {res['acc_O']:.1f}%\n"
        f"  /u/: {res['acc_U']:.1f}\n\n"
        f"HIPERPARAMETROS:\n"
    )
    for k, v in cfg.items():
        info_text += f"  {k}: {v}\n"

    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes,
             color='#E0E0E0', fontsize=10, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round,pad=1', facecolor='#1F2833', edgecolor='#66FCF1', alpha=0.8))

    plt.tight_layout()
    fig.savefig(p_png, dpi=150)
    plt.close(fig)


# ==============================================================================
# 5. MOTOR PRINCIPAL DE BARRIDO DE HIPERPARAMETROS
# ==============================================================================

def ejecutar_grid_search_modo(nombre_modo, tipo_modelo, latent_dim, tier, X_t, y_true, device, epochs=350, seed=100, resume=True):
    modo_dir = os.path.join(out_dir_base, nombre_modo)
    os.makedirs(modo_dir, exist_ok=True)
    tier_tag = "5760" if tier in ['5760', 'full'] else tier
    csv_results = os.path.join(modo_dir, f"resultados_{nombre_modo}_{tier_tag}.csv")

    if tipo_modelo == 'conv':
        if tier == 'fast':
            grid_params = {
                'channels': [(4, 8), (6, 12), (8, 16)],
                'kernel_size': [3, 5],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003],
                'lambda_w': [0.8, 1.5],
                'lambda_z': [0.30, 0.50]
            }
        elif tier == 'standard':
            grid_params = {
                'channels': [(4, 8), (4, 12), (6, 12), (8, 16)],
                'kernel_size': [3, 5, 7],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003, 0.004],
                'lambda_w': [0.6, 1.0, 1.5, 2.0],
                'lambda_z': [0.25, 0.45, 0.70]
            }
        elif tier == '3600':
            grid_params = {
                'channels': [(4, 8), (4, 12), (6, 12), (8, 16)],
                'kernel_size': [3, 5, 7, 9],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003, 0.004],
                'lambda_w': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
                'lambda_z': [0.15, 0.25, 0.35, 0.50, 0.70]
            }
        else: # '5760' o 'full' (5760 combinaciones exactas)
            grid_params = {
                'channels': [(3, 6), (4, 8), (4, 12), (6, 12), (8, 16), (8, 24)],
                'kernel_size': [3, 5, 7, 9],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003, 0.004, 0.006],
                'lambda_w': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
                'lambda_z': [0.15, 0.25, 0.35, 0.50, 0.70]
            }
    else: # 'mlp'
        if tier == 'fast':
            grid_params = {
                'hidden_dims': [(32, 16), (64, 16), (64, 32)],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003],
                'lambda_w': [0.6, 1.2],
                'lambda_z': [0.25, 0.45]
            }
        elif tier == 'standard':
            grid_params = {
                'hidden_dims': [(32, 16), (64, 16), (64, 32), (128, 32)],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003, 0.004],
                'lambda_w': [0.3, 0.6, 0.9, 1.2, 1.5],
                'lambda_z': [0.15, 0.30, 0.45, 0.60]
            }
        elif tier == '3600':
            grid_params = {
                'hidden_dims': [(32, 16), (48, 24), (64, 16), (64, 32), (96, 32), (128, 32), (128, 64)],
                'act': ['tanh', 'gelu'],
                'lr': [0.002, 0.003, 0.004],
                'lambda_w': [0.3, 0.6, 0.9, 1.2, 1.5, 2.0],
                'lambda_z': [0.15, 0.25, 0.35, 0.45, 0.60]
            }
        else: # '5760' o 'full' (5040 combinaciones exactas)
            grid_params = {
                'hidden_dims': [
                    (24, 12), (32, 16), (32, 24),
                    (48, 16), (48, 24), (64, 16),
                    (64, 32), (64, 48), (96, 32),
                    (96, 48), (128, 32), (128, 64)
                ],
                'act': ['tanh', 'gelu'],
                'lr': [0.0015, 0.002, 0.003, 0.004, 0.006],
                'lambda_w': [0.3, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
                'lambda_z': [0.15, 0.25, 0.35, 0.45, 0.60, 0.75]
            }

    keys = list(grid_params.keys())
    combinations = [dict(zip(keys, prod)) for prod in itertools.product(*grid_params.values())]
    total_comb = len(combinations)

    print(f"\n==============================================================================")
    print(f"MODO: {nombre_modo.upper()} | Tipo: {tipo_modelo.upper()} | Latente: {latent_dim}D | Nivel: {tier.upper()}")
    print(f"Total de combinaciones a evaluar: {total_comb} | Epocas: {epochs}")
    print(f"Archivo de resultados: {csv_results}")
    print(f"==============================================================================\n")

    records = []
    completed_indices = set()
    best_acc = 0.0
    best_cfg = None
    best_res = None
    best_model = None
    best_Z = None

    if resume and os.path.exists(csv_results):
        try:
            df_prev = pd.read_csv(csv_results)
            completed_indices = set(df_prev['idx'].values)
            records = df_prev.to_dict('records')
            if len(df_prev) > 0 and 'acc_global' in df_prev.columns:
                best_acc = float(df_prev['acc_global'].max())
            print(f"[Reanudación] Se detectaron {len(completed_indices)} configuraciones previas.")
            print(f"[Reanudación] Récord actual del modo: {best_acc:.2f}%\n")
        except Exception as e:
            print(f"[Aviso] No se pudo leer progreso previo ({e}). Reiniciando.")

    t_start = time.time()

    for idx, cfg in enumerate(combinations):
        current_idx = idx + 1
        if current_idx in completed_indices:
            continue

        t0_iter = time.time()
        res, model, Z = evaluar_configuracion(
            tipo_modelo=tipo_modelo,
            cfg=cfg,
            X_t=X_t,
            y_true=y_true,
            device=device,
            latent_dim=latent_dim,
            seed=seed,
            epochs=epochs
        )
        iter_time = time.time() - t0_iter
        elapsed = time.time() - t_start
        done = len(records) + 1
        eta = (elapsed / done) * (total_comb - done)

        # Construir registro para CSV
        row = {'idx': current_idx}
        for k, v in cfg.items():
            row[k] = str(v)
        row.update(res)
        row['iter_time_s'] = round(iter_time, 2)
        records.append(row)

        # Guardado incremental en CSV para evitar perdida de progreso
        pd.DataFrame(records).to_csv(csv_results, index=False)

        # Imprimir progreso cada corrida
        print(f"[{current_idx}/{total_comb}] ({(current_idx/total_comb)*100:.1f}%) | "
              f"GMM: {res['acc_global']:.2f}% (A:{res['acc_A']:.0f} E:{res['acc_E']:.0f} I:{res['acc_I']:.0f} O:{res['acc_O']:.0f} U:{res['acc_U']:.0f}) | "
              f"O-U: {res['acc_ou']:.1f}% | Sil: {res['silhouette']:+.2f} | ETA: {eta:.0f}s")

        # Deteccion de nuevo campeon
        if res['acc_global'] > best_acc:
            best_acc = res['acc_global']
            best_cfg = cfg
            best_res = res
            best_model = model
            best_Z = Z
            print(f"  >>> NUEVO RECORD [{nombre_modo}]: {best_acc:.2f}% | Min Vocal: {res['min_vocal']:.1f}% | O-U: {res['acc_ou']:.1f}% <<<")

            # Guardar artefactos del campeon (principal y por tier)
            p_png = os.path.join(modo_dir, f"campeon_{nombre_modo}.png")
            p_pt = os.path.join(modo_dir, f"campeon_{nombre_modo}.pt")
            p_json = os.path.join(modo_dir, f"config_campeon_{nombre_modo}.json")
            torch.save(best_model.state_dict(), p_pt)
            with open(p_json, 'w') as f:
                json.dump({**{k: str(v) for k, v in best_cfg.items()}, **best_res}, f, indent=2)
            graficar_campeon(nombre_modo, best_cfg, best_res, best_Z, y_true, p_png, latent_dim=latent_dim)

            if tier_tag != "fast":
                p_png_tier = os.path.join(modo_dir, f"campeon_{nombre_modo}_{tier_tag}.png")
                p_pt_tier = os.path.join(modo_dir, f"campeon_{nombre_modo}_{tier_tag}.pt")
                p_json_tier = os.path.join(modo_dir, f"config_campeon_{nombre_modo}_{tier_tag}.json")
                torch.save(best_model.state_dict(), p_pt_tier)
                with open(p_json_tier, 'w') as f:
                    json.dump({**{k: str(v) for k, v in best_cfg.items()}, **best_res}, f, indent=2)
                graficar_campeon(nombre_modo, best_cfg, best_res, best_Z, y_true, p_png_tier, latent_dim=latent_dim)

    print(f"\n[Finalizado] Modo {nombre_modo} completado en {time.time()-t_start:.1f}s.")
    print(f"             Mejor Exactitud Global: {best_acc:.2f}%\n")
    return best_acc, best_res


# ==============================================================================
# 6. ENTRADA PRINCIPAL (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Grid Search en Lucas con Ventana 40/60")
    parser.add_argument('--modo', type=str, default='todos',
                        choices=['todos', 'conv_2d', 'mlp_2d', 'conv_3d', 'mlp_3d'],
                        help="Modo arquitectural a evaluar")
    parser.add_argument('--tier', type=str, default='5760',
                        choices=['fast', 'standard', '3600', '5760', 'full'],
                        help="Nivel de exhaustividad de la grilla (fast: 48, standard: 200, 3600: 2880, 5760/full: 5760)")
    parser.add_argument('--epochs', type=int, default=350, help="Epocas de entrenamiento por combinacion")
    parser.add_argument('--seed', type=int, default=100, help="Semilla aleatoria")
    parser.add_argument('--no-resume', action='store_true', help="Reiniciar desde cero ignorando CSV previo")
    args = parser.parse_args()

    if hasattr(os, 'cpu_count') and os.cpu_count():
        torch.set_num_threads(min(8, os.cpu_count()))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo de calculo: {device} | Hilos CPU: {torch.get_num_threads()}")

    # Cargar o extraer dataset una sola vez
    X_t, y_true = cargar_o_extraer_lucas_4060()

    modos_a_ejecutar = []
    if args.modo == 'todos':
        modos_a_ejecutar = [
            ('conv_2d', 'conv', 2),
            ('mlp_2d_sin_so2', 'mlp', 2),
            ('conv_3d', 'conv', 3),
            ('mlp_3d_sin_so2', 'mlp', 3)
        ]
    elif args.modo == 'conv_2d':
        modos_a_ejecutar = [('conv_2d', 'conv', 2)]
    elif args.modo == 'mlp_2d':
        modos_a_ejecutar = [('mlp_2d_sin_so2', 'mlp', 2)]
    elif args.modo == 'conv_3d':
        modos_a_ejecutar = [('conv_3d', 'conv', 3)]
    elif args.modo == 'mlp_3d':
        modos_a_ejecutar = [('mlp_3d_sin_so2', 'mlp', 3)]

    resumen_global = {}
    for nombre_modo, tipo_modelo, latent_dim in modos_a_ejecutar:
        acc, res = ejecutar_grid_search_modo(
            nombre_modo=nombre_modo,
            tipo_modelo=tipo_modelo,
            latent_dim=latent_dim,
            tier=args.tier,
            X_t=X_t,
            y_true=y_true,
            device=device,
            epochs=args.epochs,
            seed=args.seed,
            resume=not args.no_resume
        )
        resumen_global[nombre_modo] = acc

    print("\n==============================================================================")
    print("RESUMEN CONSOLIDADO DE BARRIDO DE HIPERPARAMETROS (LUCAS VENTANA 40/60)")
    print("==============================================================================")
    for m, acc in resumen_global.items():
        print(f"  Modo {m.ljust(18)}: {acc:.2f}% Exactitud GMM")
    print("==============================================================================\n")

if __name__ == '__main__':
    main()
