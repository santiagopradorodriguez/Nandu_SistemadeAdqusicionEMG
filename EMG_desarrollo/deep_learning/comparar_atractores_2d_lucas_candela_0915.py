#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa de Espacio Latente 2D y Atractores:
Lucas vs Candela (Sesiones 2026-09-15/16 con Tríada Canónica: Belly, Zygomaticus, Orbicularis).
Modelo: Campeón Convolucional Ortogonal 2D - Récord 89.04% Lucas (Configuración 3179).
Hiperparámetros: Canales=(6, 12), Kernel=5, Tanh, lr=0.003, lw=2.0, lz=0.5, 350 épocas.
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.signal import butter, filtfilt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.cross_decomposition import CCA
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

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}
COLORES = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# ------------------------------------------------------------------------------
# 1. Arquitectura Convolucional Ortogonal 2D
# ------------------------------------------------------------------------------
class ParametricConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
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
        z = self.encode(x_3d)
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        dh2_3d = dh2.view(dh2.shape[0], 12, self.time_pts)
        dh3 = self.act(self.deconv1(dh2_3d))
        recon_3d = self.deconv2(dh3)
        recon_flat = recon_3d.view(recon_3d.shape[0], -1)
        return recon_flat, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            d0, d1 = W.shape
            gram = torch.mm(W, W.t()) if d0 < d1 else torch.mm(W.t(), W)
            I = torch.eye(min(d0, d1), device=W.device)
            loss = loss + torch.sum((gram - I) ** 2)
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
            W = conv.weight.view(conv.weight.shape[0], -1)
            d0, d1 = W.shape
            gram = torch.mm(W, W.t()) if d0 < d1 else torch.mm(W.t(), W)
            I = torch.eye(min(d0, d1), device=W.device)
            loss = loss + torch.sum((gram - I) ** 2)
        return loss

# ------------------------------------------------------------------------------
# 2. Carga y Preparación de Datos de Lucas (50/50)
# ------------------------------------------------------------------------------
def extraer_sesion_agnostica(toma_str):
    s = str(toma_str)
    m = re.search(r'(Serie\d+|Prueba\d+|Sesion\d+|Session\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    parts = s.split('_')
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win') or p_clean.lower().startswith('w'):
            continue
        if any(char.isdigit() for char in p_clean) and len(p_clean) <= 10:
            return p_clean.upper()
    return 'S1'

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

    return X_norm.reshape(N, -1)

def cargar_datos_lucas():
    csv_lucas = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv"
    )
    df = pd.read_csv(csv_lucas)
    df['Sesion'] = [extraer_sesion_agnostica(t) for t in df['Toma']]
    feat_cols = [c for c in df.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_raw = df[feat_cols].values
    y = df['Vocal'].values
    sesiones = df['Sesion'].values
    X_flat = normalizar_sesiones_conv(X_raw, sesiones, n_ch=3, n_pts=20)
    return torch.tensor(X_flat, dtype=torch.float32), y

# ------------------------------------------------------------------------------
# 3. Obtención del Modelo 89.04% (Cargar o Entrenar Configuración 3179)
# ------------------------------------------------------------------------------
def obtener_modelo_89(dir_conv, X_lucas_t, y_lucas):
    p_pt_89 = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal_89.pt")
    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')

    if os.path.exists(p_pt_89):
        print(f"[Modelo 89%] Cargando checkpoint existente: {p_pt_89}")
        ckpt = torch.load(p_pt_89, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.eval()
        with torch.no_grad():
            Z_l = model.encode(X_lucas_t).numpy()
        return model, Z_l

    print("[Modelo 89%] Entrenando Configuración 3179 (K=5, Ch=(6, 12), Tanh, lr=0.003, lw=2.0, lz=0.5, 350 épocas)...")
    torch.manual_seed(100)
    np.random.seed(100)

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    I_2d = torch.eye(2)
    N_lucas = X_lucas_t.shape[0]

    for epoch in range(350):
        optimizer.zero_grad()
        recon, z = model(X_lucas_t)
        loss_recon = nn.functional.mse_loss(recon, X_lucas_t)
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N_lucas - 1)
        loss_z = torch.sum((cov_z - I_2d) ** 2)
        loss = loss_recon + 2.0 * loss_w + 0.5 * loss_z
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z_l = model.encode(X_lucas_t).numpy()

    # Evaluar GMM para verificar el 89%
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
    pred_raw = gmm.fit_predict(Z_l)
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(VOCALES):
        for j in range(5):
            contingency[j, i] = np.sum((y_lucas == vl) & (pred_raw == j))
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    pred_idx = np.array([cluster_to_vocal[c] for c in pred_raw])
    y_idx = np.array([VOCAL_TO_IDX[v] for v in y_lucas])
    acc = accuracy_score(y_idx, pred_idx) * 100.0

    print(f"  [OK] Exactitud GMM verificada en Lucas: {acc:.2f}% (Récord 89.04%)")

    torch.save({
        'model_state_dict': model.state_dict(),
        'acc_lucas': acc,
        'cfg': {'conv_channels': (6, 12), 'kernel_size': 5, 'act': 'tanh', 'lr': 0.003, 'lambda_w': 2.0, 'lambda_z': 0.5},
        'gmm_weights': gmm.weights_,
        'gmm_means': gmm.means_,
        'gmm_covariances': gmm.covariances_,
        'cluster_to_vocal': cluster_to_vocal
    }, p_pt_89)
    print(f"  [Guardado] Checkpoint exportado en: {p_pt_89}")

    return model, Z_l

# ------------------------------------------------------------------------------
# 4. Carga y Extracción de Candela (2026-09-15/16)
# ------------------------------------------------------------------------------
def cargar_candela_zigomatico_0915(dir_conv):
    cache_path = os.path.join(dir_conv, "dataset_candela_0915_zigomatico_conv2d.npz")
    if os.path.exists(cache_path):
        print(f"[Caché] Cargando dataset Candela Cigomático (2026-09-15/16):")
        print(f"        {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X_clean'], d['Y_clean']

    base_cande = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-09-16")
    tomas_todas = sorted([t for t in os.listdir(base_cande) if os.path.isdir(os.path.join(base_cande, t))])
    tomas_cande = [t for t in tomas_todas if t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']]

    print(f"[Extracción Candela 09-15] Extrayendo {len(tomas_cande)} tomas con Belly, Zygomaticus y Orbicularis...")
    X_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_cande,
        mediciones=tomas_cande,
        alpha_ruido=1.0,
        gate_ratio_ruido=0.0,
        smooth_ms=90,
        notch_q=2.0,
        target_len=20,
        modo_alineacion="Pico Volumen Micrófono",
        pre_pct=0.50,
        post_pct=0.50,
        canales_features=["canal_0", "canal_1", "canal_2"],
        aplicar_correccion_intersesion=True,
        tipo_envolvente="rms",
        lowpass_cutoff_hz=500.0,
        tipo_filtro_ruido="notch",
        highpass_cutoff_hz=20.0
    )

    X_arr = np.array(X_raw, dtype=np.float32)
    Y_arr = np.array(Y_raw)
    sesiones_arr = np.array([extraer_sesion_agnostica(t) for t in tomas_wins])

    X_norm = normalizar_sesiones_conv(X_arr, sesiones_arr, n_ch=3, n_pts=20)

    iso = IsolationForest(contamination=0.10, random_state=42)
    mask_inliers = (iso.fit_predict(X_norm) == 1)
    X_clean = X_norm[mask_inliers]
    Y_clean = Y_arr[mask_inliers]

    np.savez_compressed(cache_path, X_clean=X_clean, Y_clean=Y_clean)
    print(f"[Caché] Guardado dataset de Candela: {len(Y_clean)} ventanas válidas post-purga.")
    return X_clean, Y_clean

# ------------------------------------------------------------------------------
# 5. Flujo Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("COMPARATIVA 2D: LUCAS RECORD 89% VS CANDELA CIGOMATICO MAYOR (2026-09-15)")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    os.makedirs(dir_conv, exist_ok=True)

    # 1. Cargar Lucas y Modelo 89.04%
    print("[Lucas] Cargando datos base de Lucas...")
    X_L_t, Y_L = cargar_datos_lucas()
    model, Z_L = obtener_modelo_89(dir_conv, X_L_t, Y_L)

    # 2. Cargar Candela 09-15 (Tríada Canónica)
    X_C, Y_C = cargar_candela_zigomatico_0915(dir_conv)

    # 3. Proyectar Candela en el Modelo 89%
    with torch.no_grad():
        Z_C = model.encode(torch.tensor(X_C, dtype=torch.float32)).numpy()

    print(f"\n[Proyecciones 2D Listas]")
    print(f"  Lucas:   {Z_L.shape[0]} muestras en R^2")
    print(f"  Candela: {Z_C.shape[0]} muestras en R^2 (con Cigomático Mayor)")

    # 4. Atractores Centroides en 2D
    atractores_L = {}
    atractores_C = {}
    matriz_L = []
    matriz_C = []

    print("\n" + "-" * 80)
    print("ATRACTORES VOCALICOS 2D (MODELO 89%) - COORDENADAS, NORMAS Y ANGULOS")
    print("-" * 80)
    print(f"{'Vocal':<6} | {'Lucas Atractor (Z1, Z2)':<26} | {'||z_L||':<7} | {'Cande Atractor (Z1, Z2)':<26} | {'||z_C||':<7} | {'Angulo Dif'}")
    print("-" * 80)

    angulos_dif = {}
    for v in VOCALES:
        m_l = (Y_L == v)
        m_c = (Y_C == v)
        c_l = np.mean(Z_L[m_l], axis=0)
        c_c = np.mean(Z_C[m_c], axis=0)

        atractores_L[v] = c_l
        atractores_C[v] = c_c
        matriz_L.append(c_l)
        matriz_C.append(c_c)

        norm_l = np.linalg.norm(c_l)
        norm_c = np.linalg.norm(c_c)

        cos_sim = np.dot(c_l, c_c) / max(1e-7, (norm_l * norm_c))
        ang_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        angulos_dif[v] = ang_deg

        str_l = f"({c_l[0]:+.2f}, {c_l[1]:+.2f})"
        str_c = f"({c_c[0]:+.2f}, {c_c[1]:+.2f})"
        print(f" /{v.lower()}/   | {str_l:<26} | {norm_l:<7.2f} | {str_c:<26} | {norm_c:<7.2f} | {ang_deg:>6.1f}°")

    matriz_L = np.array(matriz_L)
    matriz_C = np.array(matriz_C)

    # 5. Analisis de Jordan y CCA en 2D
    print("\n" + "-" * 80)
    print("ANALISIS CANONICO DE JORDAN Y CORRELACIONES CANONICAS (CCA 2D)")
    print("-" * 80)

    subspace_rad = la.subspace_angles(matriz_L.T, matriz_C.T)
    subspace_deg = np.degrees(subspace_rad)
    for idx_ang, d in enumerate(subspace_deg, 1):
        print(f"  Theta_{idx_ang} (Jordan): {d:.2f}°  -->  cos(Theta_{idx_ang}) = {np.cos(np.radians(d)):.4f}")

    cca = CCA(n_components=2)
    cca.fit(matriz_C, matriz_L)
    Z_C_cca, Z_L_cca = cca.transform(matriz_C, matriz_L)
    r1 = np.corrcoef(Z_C_cca[:, 0], Z_L_cca[:, 0])[0, 1]
    r2 = np.corrcoef(Z_C_cca[:, 1], Z_L_cca[:, 1])[0, 1]
    print(f"\nCorrelaciones Canonicas (CCA 2D):")
    print(f"  Dimension Canonica 1: rho_1 = {r1:.4f}  (Angulo efectivo: {np.degrees(np.arccos(np.clip(r1, -1, 1))):.2f}°)")
    print(f"  Dimension Canonica 2: rho_2 = {r2:.4f}  (Angulo efectivo: {np.degrees(np.arccos(np.clip(r2, -1, 1))):.2f}°)")

    # --------------------------------------------------------------------------
    # 6. Graficos Comparativos en 2D (3 Paneles)
    # --------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.patch.set_facecolor('#0B0C10')

    # Panel 1: Lucas 2D
    ax1 = axes[0]
    ax1.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_L == v)
        ax1.scatter(Z_L[m, 0], Z_L[m, 1], c=COLORES[v], alpha=0.35, s=25, edgecolors='none')
        c = atractores_L[v]
        ax1.scatter(c[0], c[1], c=COLORES[v], s=180, edgecolors='#FFFFFF', linewidths=1.5)
        ax1.plot([0, c[0]], [0, c[1]], color=COLORES[v], linewidth=2.5, label=f"/{v.lower()}/ (||z||={np.linalg.norm(c):.2f})")
    ax1.scatter(0, 0, c='#FFFFFF', s=60, marker='x', label='Reposo (0,0)')
    ax1.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax1.set_title("Lucas: Modelo Record 89.04% (Conv Ortogonal)", color='#66FCF1', fontsize=11, fontweight='bold')
    ax1.set_xlabel("Coordenada Z1", color='#C5C6C7')
    ax1.set_ylabel("Coordenada Z2", color='#C5C6C7')
    ax1.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Panel 2: Candela 2D con Cigomatico Mayor
    ax2 = axes[1]
    ax2.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_C == v)
        ax2.scatter(Z_C[m, 0], Z_C[m, 1], c=COLORES[v], alpha=0.35, s=25, edgecolors='none')
        c = atractores_C[v]
        ax2.scatter(c[0], c[1], c=COLORES[v], s=180, edgecolors='#FFFFFF', linewidths=1.5)
        ax2.plot([0, c[0]], [0, c[1]], color=COLORES[v], linewidth=2.5, label=f"/{v.lower()}/ (dif={angulos_dif[v]:.0f}°)")
    ax2.scatter(0, 0, c='#FFFFFF', s=60, marker='x', label='Reposo (0,0)')
    ax2.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax2.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax2.set_title("Candela: Triada Zygomaticus (Sin Alinear)", color='#FF6B6B', fontsize=11, fontweight='bold')
    ax2.set_xlabel("Coordenada Z1", color='#C5C6C7')
    ax2.set_ylabel("Coordenada Z2", color='#C5C6C7')
    ax2.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Panel 3: Superposicion de Atractores
    ax3 = axes[2]
    ax3.set_facecolor('#0B0C10')
    for v in VOCALES:
        cL = atractores_L[v]
        cC = atractores_C[v]
        ax3.plot([0, cL[0]], [0, cL[1]], color=COLORES[v], linewidth=3.0, alpha=0.9, label=f"Lucas /{v.lower()}/")
        ax3.scatter(cL[0], cL[1], c=COLORES[v], s=160, edgecolors='#FFFFFF', linewidths=1.5)
        ax3.plot([0, cC[0]], [0, cC[1]], color=COLORES[v], linewidth=2.5, linestyle='--', alpha=0.8)
        ax3.scatter(cC[0], cC[1], c=COLORES[v], s=160, marker='D', edgecolors='#FF6B6B', linewidths=1.5)

    ax3.scatter(0, 0, c='#FFFFFF', s=60, marker='x')
    ax3.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax3.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax3.set_title("Superposicion: Lucas (-) vs Candela (--)", color='#F7B731', fontsize=11, fontweight='bold')
    ax3.set_xlabel("Coordenada Z1", color='#C5C6C7')
    ax3.set_ylabel("Coordenada Z2", color='#C5C6C7')
    ax3.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    fig.tight_layout()
    p_fig = os.path.join(dir_conv, "comparativa_atractores_2d_lucas_vs_cande_89.png")
    fig.savefig(p_fig, dpi=150)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")
    print("[OK] Analisis 2D con Modelo 89% completado con exito.")

if __name__ == '__main__':
    main()
