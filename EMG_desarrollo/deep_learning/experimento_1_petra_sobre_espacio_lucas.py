#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1 (Ampliación): Proyección de Petra sobre el Espacio Latente de Lucas
Evalúa las tomas de Petra (Día 1: 2026-08-21 y Día 2: 2026-08-28) proyectadas en el espacio
latente del modelo campeón convolucional ortogonal 2D de Lucas.
Compara los atractores de Lucas frente a los de Petra, calcula ángulos de Jordan y CCA,
y genera visualizaciones limpias sin GMM con fondo blanco.
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
    print("PROYECCION DE PETRA SOBRE EL ESPACIO LATENTE DE LUCAS (CAMPEON CONV 2D)")
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

    # 3. Cargar datos de Petra
    d1 = np.load(os.path.join(dir_conv, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    X_P1, Y_P1 = d1['X_clean'], d1['Y_clean']

    d2 = np.load(os.path.join(dir_conv, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_P2, Y_P2 = d2['X_clean'], d2['Y_clean']

    with torch.no_grad():
        Z_P1 = model.encode(torch.tensor(X_P1, dtype=torch.float32)).numpy()
        Z_P2 = model.encode(torch.tensor(X_P2, dtype=torch.float32)).numpy()

    # 4. Calcular atractores centroides
    cents_L = {v: np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES}
    cents_P1 = {v: np.mean(Z_P1[Y_P1 == v], axis=0) for v in VOCALES}
    cents_P2 = {v: np.mean(Z_P2[Y_P2 == v], axis=0) for v in VOCALES}

    mat_L = np.array([cents_L[v] for v in VOCALES])
    mat_P1 = np.array([cents_P1[v] for v in VOCALES])
    mat_P2 = np.array([cents_P2[v] for v in VOCALES])

    # 5. Ángulos de Jordan y CCA con Lucas
    print("\n[Analisis de Jordan: Lucas vs Petra Dia 1 (2026-08-21)]")
    ang_jordan_P1 = np.degrees(la.subspace_angles(mat_L.T, mat_P1.T))
    for i, a in enumerate(ang_jordan_P1, 1):
        print(f"  Theta_{i} (Jordan): {a:.2f}°  (cos = {np.cos(np.radians(a)):.4f})")

    cca_P1 = CCA(n_components=2)
    cca_P1.fit(mat_P1, mat_L)
    mat_P1_c, mat_L_c1 = cca_P1.transform(mat_P1, mat_L)
    rho1_P1 = np.corrcoef(mat_P1_c[:, 0], mat_L_c1[:, 0])[0, 1]
    rho2_P1 = np.corrcoef(mat_P1_c[:, 1], mat_L_c1[:, 1])[0, 1]
    print(f"  CCA: rho_1 = {rho1_P1:.4f} ({np.degrees(np.arccos(np.clip(rho1_P1, -1, 1))):.2f}°), rho_2 = {rho2_P1:.4f} ({np.degrees(np.arccos(np.clip(rho2_P1, -1, 1))):.2f}°)")

    print("\n[Analisis de Jordan: Lucas vs Petra Dia 2 (2026-08-28)]")
    ang_jordan_P2 = np.degrees(la.subspace_angles(mat_L.T, mat_P2.T))
    for i, a in enumerate(ang_jordan_P2, 1):
        print(f"  Theta_{i} (Jordan): {a:.2f}°  (cos = {np.cos(np.radians(a)):.4f})")

    cca_P2 = CCA(n_components=2)
    cca_P2.fit(mat_P2, mat_L)
    mat_P2_c, mat_L_c2 = cca_P2.transform(mat_P2, mat_L)
    rho1_P2 = np.corrcoef(mat_P2_c[:, 0], mat_L_c2[:, 0])[0, 1]
    rho2_P2 = np.corrcoef(mat_P2_c[:, 1], mat_L_c2[:, 1])[0, 1]
    print(f"  CCA: rho_1 = {rho1_P2:.4f} ({np.degrees(np.arccos(np.clip(rho1_P2, -1, 1))):.2f}°), rho_2 = {rho2_P2:.4f} ({np.degrees(np.arccos(np.clip(rho2_P2, -1, 1))):.2f}°)")

    # 6. Tabla comparativa de atractores
    print("\n" + "-" * 95)
    print("COMPARATIVA DE ATRACTORES: LUCAS vs PETRA DIA 1 vs PETRA DIA 2")
    print("-" * 95)
    print(f"{'Vocal':<6} | {'Lucas Nativo':<20} | {'Petra D1 Raw':<20} | {'Ang Dif D1':<10} | {'Petra D2 Raw':<20} | {'Ang Dif D2'}")
    print("-" * 95)
    for v in VOCALES:
        cL = cents_L[v]
        cP1 = cents_P1[v]
        cP2 = cents_P2[v]
        ang1 = np.degrees(np.arccos(np.clip(np.dot(cL, cP1) / (np.linalg.norm(cL) * np.linalg.norm(cP1)), -1, 1)))
        ang2 = np.degrees(np.arccos(np.clip(np.dot(cL, cP2) / (np.linalg.norm(cL) * np.linalg.norm(cP2)), -1, 1)))
        strL = f"({cL[0]:+.2f}, {cL[1]:+.2f})"
        strP1 = f"({cP1[0]:+.2f}, {cP1[1]:+.2f})"
        strP2 = f"({cP2[0]:+.2f}, {cP2[1]:+.2f})"
        print(f" /{v.lower()}/   | {strL:<20} | {strP1:<20} | {ang1:>7.1f}°  | {strP2:<20} | {ang2:>7.1f}°")

    # 7. Alineaciones Afines sobre Lucas
    W_aff1, _, _, _ = np.linalg.lstsq(mat_P1 - np.mean(mat_P1, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_P1_aff = (Z_P1 - np.mean(mat_P1, axis=0)) @ W_aff1 + np.mean(mat_L, axis=0)
    cents_P1_aff = {v: np.mean(Z_P1_aff[Y_P1 == v], axis=0) for v in VOCALES}

    W_aff2, _, _, _ = np.linalg.lstsq(mat_P2 - np.mean(mat_P2, axis=0), mat_L - np.mean(mat_L, axis=0), rcond=None)
    Z_P2_aff = (Z_P2 - np.mean(mat_P2, axis=0)) @ W_aff2 + np.mean(mat_L, axis=0)
    cents_P2_aff = {v: np.mean(Z_P2_aff[Y_P2 == v], axis=0) for v in VOCALES}

    # 8. Graficado Oficial de 4 Paneles
    plt.style.use('default')
    fig, axes = plt.subplots(1, 4, figsize=(28, 7), facecolor='white')
    all_pts = np.vstack([Z_L, Z_P1, Z_P2, Z_P1_aff, Z_P2_aff])
    max_val = max(np.abs(all_pts).max() * 1.1, 3.8)
    lim = (-max_val, max_val)

    # Panel 1: Lucas Nativo
    ax = axes[0]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        m = (Y_L == v)
        ax.scatter(Z_L[m, 0], Z_L[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=45, edgecolors='black', linewidth=0.4, alpha=0.7)
        c = cents_L[v]
        ax.plot([0, c[0]], [0, c[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9)
        ax.scatter(c[0], c[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=180, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])
    ax.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax.set_title("Lucas Nativo: Espacio Campeón", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9, frameon=True)

    # Panel 2: Petra Proyectada Cruda (Día 1 y Día 2)
    ax = axes[1]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        # Día 1: círculos
        m1 = (Y_P1 == v)
        ax.scatter(Z_P1[m1, 0], Z_P1[m1, 1], c=[COLORES_VOCALES[v]], s=40, edgecolors='black', linewidth=0.4, alpha=0.6, marker='o')
        # Día 2: cuadrados
        m2 = (Y_P2 == v)
        ax.scatter(Z_P2[m2, 0], Z_P2[m2, 1], c=[COLORES_VOCALES[v]], s=40, edgecolors='black', linewidth=0.4, alpha=0.6, marker='s')
        
        cP1 = cents_P1[v]
        cP2 = cents_P2[v]
        ax.plot([0, cP1[0]], [0, cP1[1]], color=COLORES_VOCALES[v], linewidth=2.0, linestyle='-', alpha=0.8)
        ax.plot([0, cP2[0]], [0, cP2[1]], color=COLORES_VOCALES[v], linewidth=2.0, linestyle='--', alpha=0.8)
        ax.scatter(cP1[0], cP1[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='o', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax.scatter(cP2[0], cP2[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='s', s=150, edgecolors='#C0392B', linewidth=1.5, zorder=5)

    ax.scatter(0, 0, c='black', s=70, marker='x', zorder=6)
    ax.set_title("Petra Cruda: Día 1 (o) y Día 2 (s)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(True, linestyle=':', alpha=0.6)

    # Panel 3: Rayos Directores Comparativos (Lucas vs Petra)
    ax = axes[2]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        cL = cents_L[v]
        cP1 = cents_P1[v]
        cP2 = cents_P2[v]
        # Lucas: Sólido grueso con diamante
        ax.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=3.2, linestyle='-', alpha=0.9, label=f"Lucas /{v.lower()}/")
        ax.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=160, edgecolors='black', linewidth=1.5, zorder=6)
        # Petra D1: Discontinua fina con círculo
        ax.plot([0, cP1[0]], [0, cP1[1]], color=COLORES_VOCALES[v], linewidth=1.8, linestyle=':', alpha=0.8)
        ax.scatter(cP1[0], cP1[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='o', s=110, edgecolors='black', linewidth=1.2, zorder=5)
        # Petra D2: Discontinua con cuadrado
        ax.plot([0, cP2[0]], [0, cP2[1]], color=COLORES_VOCALES[v], linewidth=1.8, linestyle='--', alpha=0.8)
        ax.scatter(cP2[0], cP2[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='s', s=110, edgecolors='#C0392B', linewidth=1.2, zorder=5)

    ax.scatter(0, 0, c='black', s=70, marker='x', zorder=7)
    ax.set_title("Rayos: Lucas (-) vs Petra D1 (:) vs Petra D2 (--)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9, frameon=True)

    # Panel 4: Petra Alineada con CCA/Afín sobre Lucas
    ax = axes[3]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        cL = cents_L[v]
        cP1_al = cents_P1_aff[v]
        cP2_al = cents_P2_aff[v]
        
        # Lucas referencia
        ax.scatter(cL[0], cL[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=160, edgecolors='black', linewidth=1.5, zorder=4)
        ax.plot([0, cL[0]], [0, cL[1]], color=COLORES_VOCALES[v], linewidth=2.8, linestyle='-', alpha=0.4)

        # Petra D1 alineada: estrella
        ax.scatter(cP1_al[0], cP1_al[1], c=[COLORES_VOCALES[v]], marker='*', s=180, edgecolors='black', linewidth=1.0, zorder=6, label=f"Petra D1 CCA /{v.lower()}/")
        # Petra D2 alineada: cruz +
        ax.scatter(cP2_al[0], cP2_al[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.4)], marker='P', s=140, edgecolors='black', linewidth=1.0, zorder=7)

    ax.scatter(0, 0, c='black', s=70, marker='x', zorder=8)
    ax.set_title("Petra Alineada sobre Lucas (Afín/CCA)", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax.set_xlim(lim); ax.set_ylim(lim); ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower left', fontsize=9, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "experimento_1_petra_sobre_espacio_lucas.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")

    # Copiar a artifacts
    art_dir = "/home/santiago/.gemini/antigravity/brain/1e3163af-3535-4d14-ad6f-17b50aea53eb"
    shutil.copy(p_fig, os.path.join(art_dir, "experimento_1_petra_sobre_espacio_lucas.png"))
    shutil.copy(os.path.join(dir_conv, "experimento_1_petra_interdia_atractores_cca.png"),
                os.path.join(art_dir, "experimento_1_petra_interdia_atractores_cca.png"))
    print(f"[Copia Artifacts] Imágenes copiadas exitosamente a {art_dir}")
    print("[OK] Procesamiento y proyeccion completados con éxito.")

if __name__ == '__main__':
    main()
