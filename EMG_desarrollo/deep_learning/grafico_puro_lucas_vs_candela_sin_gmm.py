#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espacio Latente Puro y Atractores (Sin GMM ni Fronteras Artificiales):
Lucas vs Candela (Sesiones 2026-09-15/16 con Tríada Canónica: Belly, Zygomaticus, Orbicularis).
Modelo: Campeón Convolucional Ortogonal 2D.
Estética: Fondo blanco limpio, nubes de puntos oficiales, vectores radiales y centroides.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import torch
import torch.nn as nn

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.pca_umap_clustering import generador_pca_umap as gpu

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}
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

# ------------------------------------------------------------------------------
# 2. Carga y Normalización
# ------------------------------------------------------------------------------
def cargar_lucas():
    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    df = pd.read_csv(os.path.join(dir_conv, "proyecciones_campeon_lucas.csv"))
    return df[['Z1', 'Z2']].values, df['Vocal'].values

def cargar_candela():
    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    d = np.load(os.path.join(dir_conv, "dataset_candela_0915_zigomatico_conv2d.npz"), allow_pickle=True)
    return d['X_clean'], d['Y_clean']

# ------------------------------------------------------------------------------
# 3. Flujo Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("GRAFICANDO ESPACIO LATENTE PURO Y ATRACTORES (SIN GMM)")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # 1. Lucas nativo
    Z_L, Y_L = cargar_lucas()

    # 2. Candela RAW (sin alinear)
    X_C, Y_C = cargar_candela()
    with torch.no_grad():
        Z_C_raw = model.encode(torch.tensor(X_C, dtype=torch.float32)).numpy()

    # 3. Atractores centroides
    cents_L = {v: np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES}
    cents_C = {v: np.mean(Z_C_raw[Y_C == v], axis=0) for v in VOCALES}

    # Escala compartida
    all_pts = np.vstack([Z_L, Z_C_raw])
    max_val = max(np.abs(all_pts).max() * 1.15, 3.5)
    lim = (-max_val, max_val)

    # --------------------------------------------------------------------------
    # Graficado de 3 Paneles: Lucas Puro | Candela Raw Puro | Superposición
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')

    # Panel 1: Lucas Puro
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_L == v)
        # Nube de puntos
        ax1.scatter(Z_L[m, 0], Z_L[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=55, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        # Rayo desde origen a atractor
        cL = cents_L[v]
        ax1.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        # Atractor diamante
        ax1.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=200, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax1.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax1.set_title("Lucas: Espacio Latente 2D y Atractores Nativos", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 2: Candela RAW Puro (Sin Alinear)
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_C == v)
        # Nube de puntos
        ax2.scatter(Z_C_raw[m, 0], Z_C_raw[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=55, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        # Rayo desde origen a atractor
        cC = cents_C[v]
        ax2.plot([0, cC[0]], [0, cC[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        # Atractor diamante
        ax2.scatter(cC[0], cC[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=200, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax2.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax2.set_title("Candela (2026-09-15): Proyección Cruda Sin Alinear", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 3: Superposición Limpia de Atractores
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        cL = cents_L[v]
        cC = cents_C[v]

        # Lucas: Línea sólida + Círculo con resplandor
        ax3.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=3.2, linestyle='-', alpha=0.9, zorder=3, label=f"Lucas /{v.lower()}/")
        ax3.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], s=180, marker='o', edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        # Candela Raw: Línea discontinua + Diamante
        ax3.plot([0, cC[0]], [0, cC[1]], color=COLORES_VOCALES[v], linewidth=2.8, linestyle='--', alpha=0.85, zorder=4)
        ax3.scatter(cC[0], cC[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], s=180, marker='D', edgecolors='#C0392B', linewidth=1.5, zorder=6, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

    ax3.scatter(0, 0, c='black', s=70, marker='x', zorder=7)
    ax3.set_title("Superposición de Atractores: Lucas (Sólido) vs Candela Raw (Discontinuo)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(lim)
    ax3.set_ylim(lim)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=10, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "espacio_latente_puro_lucas_vs_candela_raw_sin_gmm.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")
    print("[OK] Gráficos puros sin GMM generados con éxito.")

if __name__ == '__main__':
    main()
