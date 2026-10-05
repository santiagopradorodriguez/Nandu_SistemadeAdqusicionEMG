#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa de Espacio Latente: Lucas (Alineación Canónica B) frente a Petra
Aplica traslación para ubicar el vértice de Lucas en el eje Y y rota para anclar /a/ en el semieje vertical positivo (+Y, 90°).
Proyecta Petra y alinea mediante CCA/Afín sobre la variedad canónica de Lucas.
Límites xlim e ylim estrictos sin espacio vacío.
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
    print("ALINEACION CANONICA B: VERTICE EN EJE Y Y /A/ EN +Y")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    
    # 1. Cargar proyecciones de Lucas
    df_lucas = pd.read_csv(os.path.join(dir_conv, "proyecciones_campeon_lucas.csv"))
    Z_L_raw = df_lucas[['Z1', 'Z2']].values
    Y_L = df_lucas['Vocal'].values

    # Centroides crudos de Lucas
    cents_L_raw = {v: np.mean(Z_L_raw[Y_L == v], axis=0) for v in VOCALES}

    # 2. Transformación Canónica B de Lucas:
    # Vértice de origen v0 = base de la bifurcación (centroide de /i/ en reposo mandibular)
    v0 = cents_L_raw['I'].copy()
    v_dir_A = cents_L_raw['A'] - v0
    ang_A = np.arctan2(v_dir_A[1], v_dir_A[0])
    theta_rot = np.pi/2 - ang_A
    R = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot),  np.cos(theta_rot)]
    ])

    # Aplicar traslación + rotación a Lucas
    Z_L = (Z_L_raw - v0) @ R.T
    cents_L = {v: np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES}

    print("Lucas Canónico B:")
    for v in VOCALES:
        c = cents_L[v]
        ang = np.degrees(np.arctan2(c[1], c[0])) if np.linalg.norm(c) > 1e-4 else 0.0
        print(f"  /{v}/: ({c[0]:+.2f}, {c[1]:+.2f}), norma={np.linalg.norm(c):.2f}, angulo={ang:.1f}°")

    # 3. Cargar y unificar Petra
    d1 = np.load(os.path.join(dir_conv, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(dir_conv, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_P = np.vstack([d1['X_clean'], d2['X_clean']])
    Y_P = np.concatenate([d1['Y_clean'], d2['Y_clean']])

    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    with torch.no_grad():
        Z_P_raw = model.encode(torch.tensor(X_P, dtype=torch.float32)).numpy()

    # Petra proyectada con la misma base canónica de Lucas
    Z_P = (Z_P_raw - v0) @ R.T
    cents_P = {v: np.mean(Z_P[Y_P == v], axis=0) for v in VOCALES}

    # 4. Alineación Afín / CCA de Petra sobre el espacio canónico de Lucas
    mat_L = np.array([cents_L[v] for v in VOCALES])
    mat_P = np.array([cents_P[v] for v in VOCALES])

    W_aff, _, _, _ = np.linalg.lstsq(mat_P - np.mean(mat_P, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_P_aff = (Z_P - np.mean(mat_P, axis=0)) @ W_aff + np.mean(mat_L, axis=0)
    cents_P_aff = {v: np.mean(Z_P_aff[Y_P == v], axis=0) for v in VOCALES}

    # 5. Cálculo estricto de xlim e ylim
    all_pts = np.vstack([Z_L, Z_P, Z_P_aff, np.array([[0.0, 0.0]])])
    # Excluir outliers para los límites
    p_x_min, p_x_max = np.percentile(all_pts[:, 0], 1), np.percentile(all_pts[:, 0], 99)
    p_y_min, p_y_max = np.percentile(all_pts[:, 1], 1), np.percentile(all_pts[:, 1], 99)
    
    x_min = min(p_x_min, cents_L['U'][0] - 0.3, -0.3)
    x_max = max(p_x_max, cents_P['A'][0] + 0.3, 0.3)
    y_min = min(p_y_min, -0.4)
    y_max = max(p_y_max, cents_L['A'][1] + 0.4)

    xlim = (x_min, x_max)
    ylim = (y_min, y_max)
    print(f"\nLimites ajustados: X={xlim}, Y={ylim}")

    # 6. Graficado Oficial de 3 Paneles
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')

    # Panel 1: Lucas Canónico (Vértice en Y)
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

    ax1.scatter(0, 0, c='black', s=80, marker='x', zorder=6, label='Vértice Canónico (0,0)')
    ax1.set_title("Lucas Canónico: Vértice en Eje Y y /a/ a 90°", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(xlim); ax1.set_ylim(ylim); ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 2: Petra en Espacio Canónico de Lucas
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

    ax2.scatter(0, 0, c='black', s=80, marker='x', zorder=6, label='Vértice Canónico (0,0)')
    ax2.set_title("Petra Cruda en Base Canónica de Lucas", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 3: Superposición y Alineación Afín / CCA
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

        # Petra Alineada CCA: Estrella
        ax3.scatter(cP_al[0], cP_al[1], c=[COLORES_VOCALES[v]], marker='*', s=200, edgecolors='black', linewidth=1.2, zorder=7)

    ax3.scatter(0, 0, c='black', s=80, marker='x', zorder=8)
    ax3.set_title("Superposición: Lucas (-) vs Petra Raw (--) vs CCA (*)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(xlim); ax3.set_ylim(ylim); ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "lucas_vs_petra_canonico_opcion_b.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")

    art_dir = "/home/santiago/.gemini/antigravity/brain/1e3163af-3535-4d14-ad6f-17b50aea53eb"
    shutil.copy(p_fig, os.path.join(art_dir, "lucas_vs_petra_canonico_opcion_b.png"))
    print(f"[Copia Artifacts] Imagen copiada a {art_dir}")
    print("[OK] Generación de Opción B finalizada con éxito.")

if __name__ == '__main__':
    main()
