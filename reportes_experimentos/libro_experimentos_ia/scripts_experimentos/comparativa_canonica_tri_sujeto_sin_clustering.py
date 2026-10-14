#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa Canónica Tri-Sujeto Sin Clustering: Lucas, Candela y Petra
Proyecta y alinea afínmente los datos electromiográficos puros de los tres sujetos
sobre el espacio latente canónico de Lucas (Opción B: vértice en el eje Y y /a/ a 90°).
Sin algoritmos de clustering (sin GMM ni K-Means), únicamente datos empíricos,
atractores centroides y rayos directores desde el vértice de reposo.
Límites xlim e ylim estrictos sin espacios vacíos.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import torch
import torch.nn as nn
import shutil

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLORES_VOCALES = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

def adjust_lightness(color, amount=0.5):
    import colorsys
    try:
        c = mcolors.cnames[color]
    except Exception:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

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

def main():
    print("=" * 80)
    print("COMPARATIVA CANONICA TRI-SUJETO: LUCAS, CANDELA Y PETRA (SIN CLUSTERING)")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    
    # --------------------------------------------------------------------------
    # 1. Lucas: Espacio Canónico (Opción B)
    # --------------------------------------------------------------------------
    df_l = pd.read_csv(os.path.join(dir_conv, "proyecciones_campeon_lucas.csv"))
    Z_L_raw = df_l[['Z1', 'Z2']].values
    Y_L = df_l['Vocal'].values
    cents_L_raw = {v: np.mean(Z_L_raw[Y_L == v], axis=0) for v in VOCALES}

    # Traslación del vértice /I/ al origen y rotación de /A/ a 90° (+Y)
    v0 = cents_L_raw['I'].copy()
    v_dir = cents_L_raw['A'] - v0
    ang_A = np.arctan2(v_dir[1], v_dir[0])
    theta_rot = np.pi/2 - ang_A
    R = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot),  np.cos(theta_rot)]
    ])

    Z_L = (Z_L_raw - v0) @ R.T
    cents_L = {v: np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES}
    mat_L = np.array([cents_L[v] for v in VOCALES])

    # --------------------------------------------------------------------------
    # 2. Cargar Modelo Campeón Convolucional Ortogonal 2D
    # --------------------------------------------------------------------------
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # --------------------------------------------------------------------------
    # 3. Candela (2026-09-15 con Cigomático Mayor)
    # --------------------------------------------------------------------------
    d_c = np.load(os.path.join(dir_conv, "dataset_candela_0915_zigomatico_conv2d.npz"), allow_pickle=True)
    with torch.no_grad():
        Z_C_raw = model.encode(torch.tensor(d_c['X_clean'], dtype=torch.float32)).numpy()
    Y_C = d_c['Y_clean']
    cents_C_raw = {v: np.mean(Z_C_raw[Y_C == v], axis=0) for v in VOCALES}
    mat_C_raw = np.array([cents_C_raw[v] for v in VOCALES])

    # Alineación Afín / CCA de Candela hacia Lucas Canónico
    W_C, _, _, _ = np.linalg.lstsq(mat_C_raw - np.mean(mat_C_raw, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_C_aff = (Z_C_raw - np.mean(mat_C_raw, axis=0)) @ W_C + np.mean(mat_L, axis=0)
    cents_C_aff = {v: np.mean(Z_C_aff[Y_C == v], axis=0) for v in VOCALES}

    # --------------------------------------------------------------------------
    # 4. Petra (2026-08-21 y 2026-08-28 con Cigomático y Levator)
    # --------------------------------------------------------------------------
    d_p1 = np.load(os.path.join(dir_conv, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d_p2 = np.load(os.path.join(dir_conv, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_P = np.vstack([d_p1['X_clean'], d_p2['X_clean']])
    Y_P = np.concatenate([d_p1['Y_clean'], d_p2['Y_clean']])
    with torch.no_grad():
        Z_P_raw = model.encode(torch.tensor(X_P, dtype=torch.float32)).numpy()
    cents_P_raw = {v: np.mean(Z_P_raw[Y_P == v], axis=0) for v in VOCALES}
    mat_P_raw = np.array([cents_P_raw[v] for v in VOCALES])

    # Alineación Afín / CCA de Petra hacia Lucas Canónico
    W_P, _, _, _ = np.linalg.lstsq(mat_P_raw - np.mean(mat_P_raw, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_P_aff = (Z_P_raw - np.mean(mat_P_raw, axis=0)) @ W_P + np.mean(mat_L, axis=0)
    cents_P_aff = {v: np.mean(Z_P_aff[Y_P == v], axis=0) for v in VOCALES}

    # --------------------------------------------------------------------------
    # 5. Métricas Comparativas de Atractores
    # --------------------------------------------------------------------------
    print("\n" + "-" * 95)
    print("ATRACTORES CANONICOS COMPARATIVOS (LUCAS vs CANDELA ALINEADA vs PETRA ALINEADA)")
    print("-" * 95)
    print(f"{'Vocal':<6} | {'Lucas Canónico':<22} | {'Candela Alineada':<22} | {'Petra Alineada':<22}")
    print("-" * 95)
    for v in VOCALES:
        cL = cents_L[v]
        cC = cents_C_aff[v]
        cP = cents_P_aff[v]
        sL = f"({cL[0]:+.2f}, {cL[1]:+.2f})"
        sC = f"({cC[0]:+.2f}, {cC[1]:+.2f})"
        sP = f"({cP[0]:+.2f}, {cP[1]:+.2f})"
        print(f" /{v.lower()}/   | {sL:<22} | {sC:<22} | {sP:<22}")

    # --------------------------------------------------------------------------
    # 6. Graficado Oficial de 4 Paneles Sin Clustering
    # --------------------------------------------------------------------------
    xlim = (-3.2, 1.8)
    ylim = (-0.8, 3.5)

    plt.style.use('default')
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), facecolor='white')

    # Panel 1: Lucas Canónico
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_L == v)
        ax1.scatter(Z_L[m, 0], Z_L[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=45, edgecolors='black', linewidth=0.4, alpha=0.7, zorder=3)
        cL = cents_L[v]
        ax1.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax1.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=170, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax1.scatter(0, 0, c='black', s=80, marker='x', zorder=6, label='Vértice Canónico (0,0)')
    ax1.set_title("Lucas Canónico: Vértice en Y y /a/ a 90°", fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=9, frameon=True)

    # Panel 2: Candela Alineada
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_C == v)
        ax2.scatter(Z_C_aff[m, 0], Z_C_aff[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=45, edgecolors='black', linewidth=0.4, alpha=0.7, zorder=3)
        cC = cents_C_aff[v]
        ax2.plot([0, cC[0]], [0, cC[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax2.scatter(cC[0], cC[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='s', s=160, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax2.scatter(0, 0, c='black', s=80, marker='x', zorder=6, label='Vértice Canónico (0,0)')
    ax2.set_title("Candela: Alineada sobre Lucas (Afín/CCA)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=9, frameon=True)

    # Panel 3: Petra Alineada
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_P == v)
        ax3.scatter(Z_P_aff[m, 0], Z_P_aff[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=45, edgecolors='black', linewidth=0.4, alpha=0.7, zorder=3)
        cP = cents_P_aff[v]
        ax3.plot([0, cP[0]], [0, cP[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax3.scatter(cP[0], cP[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='o', s=160, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax3.scatter(0, 0, c='black', s=80, marker='x', zorder=6, label='Vértice Canónico (0,0)')
    ax3.set_title("Petra: Alineada sobre Lucas (Afín/CCA)", fontsize=12, fontweight='bold', pad=12)
    ax3.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(xlim); ax3.set_ylim(ylim); ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=9, frameon=True)

    # Panel 4: Superposición Tri-Sujeto (Rayos y Atractores)
    ax4 = axes[3]
    ax4.set_facecolor('white')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        cL = cents_L[v]
        cC = cents_C_aff[v]
        cP = cents_P_aff[v]

        # Lucas: Sólido con Diamante
        ax4.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=3.2, linestyle='-', alpha=0.9, label=f"Lucas /{v.lower()}/")
        ax4.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=170, edgecolors='black', linewidth=1.5, zorder=6, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        # Candela: Discontinua con Cuadrado
        ax4.plot([0, cC[0]], [0, cC[1]], color=COLORES_VOCALES[v], linewidth=2.0, linestyle='--', alpha=0.85, zorder=5)
        ax4.scatter(cC[0], cC[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.55)], marker='s', s=140, edgecolors='#1B4F72', linewidth=1.3, zorder=7)

        # Petra: Punteada con Estrella
        ax4.plot([0, cP[0]], [0, cP[1]], color=COLORES_VOCALES[v], linewidth=1.8, linestyle=':', alpha=0.85, zorder=4)
        ax4.scatter(cP[0], cP[1], c=[COLORES_VOCALES[v]], marker='*', s=180, edgecolors='black', linewidth=1.1, zorder=8)

    ax4.scatter(0, 0, c='black', s=80, marker='x', zorder=9)
    ax4.set_title("Superposición Tri-Sujeto: Lucas (-) Cande (--) Petra (*)", fontsize=12, fontweight='bold', pad=12)
    ax4.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax4.set_xlim(xlim); ax4.set_ylim(ylim); ax4.grid(True, linestyle=':', alpha=0.6)
    ax4.legend(loc='lower left', fontsize=8.5, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "comparativa_canonica_tri_sujeto_lucas_candela_petra.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")

    art_dir = "/home/santiago/.gemini/antigravity/brain/1e3163af-3535-4d14-ad6f-17b50aea53eb"
    shutil.copy(p_fig, os.path.join(art_dir, "comparativa_canonica_tri_sujeto_lucas_candela_petra.png"))
    print(f"[Copia Artifacts] Imagen copiada a {art_dir}")
    print("[OK] Comparativa tri-sujeto sin clustering completada con éxito.")

if __name__ == '__main__':
    main()
