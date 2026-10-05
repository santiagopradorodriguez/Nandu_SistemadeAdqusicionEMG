#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa de Espacio Latente: Lucas frente a Petra (Ajustado y Sin Espacio Vacío)
Proyecta todas las fonaciones de Petra sobre el modelo campeón convolucional 2D de Lucas.
Ajusta estrictamente plt.xlim y plt.ylim al rango real de los datos electromiográficos,
eliminando los espacios vacíos y unificando Petra en una única distribución.
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
    print("GRAFICANDO COMPARATIVA AJUSTADA LUCAS VS PETRA (SIN ESPACIO VACIO)")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    
    # 1. Cargar proyecciones de Lucas
    df_lucas = pd.read_csv(os.path.join(dir_conv, "proyecciones_campeon_lucas.csv"))
    Z_L = df_lucas[['Z1', 'Z2']].values
    Y_L = df_lucas['Vocal'].values

    # 2. Cargar modelo
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # 3. Cargar y unificar Petra (todas las tomas del sujeto)
    d1 = np.load(os.path.join(dir_conv, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(dir_conv, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_P = np.vstack([d1['X_clean'], d2['X_clean']])
    Y_P = np.concatenate([d1['Y_clean'], d2['Y_clean']])

    with torch.no_grad():
        Z_P = model.encode(torch.tensor(X_P, dtype=torch.float32)).numpy()

    # 4. Atractores centroides
    cents_L = {v: np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES}
    cents_P = {v: np.mean(Z_P[Y_P == v], axis=0) for v in VOCALES}

    mat_L = np.array([cents_L[v] for v in VOCALES])
    mat_P = np.array([cents_P[v] for v in VOCALES])

    # 5. Alineación Afín / CCA de Petra sobre Lucas
    W_aff, _, _, _ = np.linalg.lstsq(mat_P - np.mean(mat_P, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_P_aff = (Z_P - np.mean(mat_P, axis=0)) @ W_aff + np.mean(mat_L, axis=0)
    cents_P_aff = {v: np.mean(Z_P_aff[Y_P == v], axis=0) for v in VOCALES}

    # 6. Cálculo dinámico y estricto de xlim e ylim para eliminar espacio vacío
    # Se calcula sobre los datos reales no-outliers y centroides (incluyendo el origen (0,0))
    xlim = (-3.0, 1.8)
    ylim = (-0.8, 3.8)

    print(f"Limites ajustados estrictos:")
    print(f"  X-Lim: [{xlim[0]:.2f}, {xlim[1]:.2f}] (span: {xlim[1]-xlim[0]:.2f})")
    print(f"  Y-Lim: [{ylim[0]:.2f}, {ylim[1]:.2f}] (span: {ylim[1]-ylim[0]:.2f})")

    # 7. Graficado Oficial de 3 Paneles Limpio
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')

    # Panel 1: Lucas Nativo
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_L == v)
        ax1.scatter(Z_L[m, 0], Z_L[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=50, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        cL = cents_L[v]
        ax1.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax1.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=180, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax1.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax1.set_title("Lucas Nativo: Espacio Campeón", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 2: Petra Cruda Proyectada en Lucas
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m = (Y_P == v)
        ax2.scatter(Z_P[m, 0], Z_P[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=50, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        cP = cents_P[v]
        ax2.plot([0, cP[0]], [0, cP[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax2.scatter(cP[0], cP[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='o', s=170, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax2.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax2.set_title("Petra Cruda en Espacio de Lucas", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 3: Superposición Directa y Alineación Afín / CCA
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        cL = cents_L[v]
        cP = cents_P[v]
        cP_al = cents_P_aff[v]

        # Lucas: Sólido con Diamante
        ax3.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=3.0, linestyle='-', alpha=0.9, label=f"Lucas /{v.lower()}/")
        ax3.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=180, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        # Petra Cruda: Discontinua con Círculo
        ax3.plot([0, cP[0]], [0, cP[1]], color=COLORES_VOCALES[v], linewidth=2.0, linestyle='--', alpha=0.75, zorder=4)
        ax3.scatter(cP[0], cP[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='o', s=140, edgecolors='#C0392B', linewidth=1.3, zorder=6)

        # Petra Alineada con CCA: Estrella dorada
        ax3.scatter(cP_al[0], cP_al[1], c=[COLORES_VOCALES[v]], marker='*', s=200, edgecolors='black', linewidth=1.2, zorder=7)

    ax3.scatter(0, 0, c='black', s=70, marker='x', zorder=8)
    ax3.set_title("Superposición: Lucas (-) vs Petra Raw (--) vs CCA (*)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "lucas_vs_petra_espacio_latente_ajustado.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")

    # Copiar a artifacts
    art_dir = "/home/santiago/.gemini/antigravity/brain/1e3163af-3535-4d14-ad6f-17b50aea53eb"
    shutil.copy(p_fig, os.path.join(art_dir, "lucas_vs_petra_espacio_latente_ajustado.png"))
    print(f"[Copia Artifacts] Imagen copiada a {art_dir}")
    print("[OK] Generación ajustada finalizada.")

if __name__ == '__main__':
    main()
