#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis Comparativo de PCA Lineal y Alineación de Sara Solla (CCA) Tri-Sujeto:
Lucas (2026-07-10), Candela (2026-09-15) y Petra (2026-08-28 Día 2).
Una única sesión por sujeto, sin algoritmos de clustering (sin GMM ni K-Means).
Alineación de variedades latentes motoras por Análisis de Correlación Canónica (CCA)
y ángulos de Jordan según la metodología de Gallego, Perich, Miller & Solla.
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.linalg as la
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.grid_search_conv_ortogonal import cargar_datos_lucas

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

def main():
    print("=" * 80)
    print("ANALISIS COMPARATIVO PCA Y ALINEACION SARA SOLLA TRI-SUJETO")
    print("=" * 80)

    dir_out = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    os.makedirs(dir_out, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1. Cargar Datos: Una Única Sesión por Sujeto
    # --------------------------------------------------------------------------
    print("[1/5] Cargando conjuntos de datos (sesion unica por sujeto)...")
    
    # Lucas: Sesión única benchmark 2026-07-10 (502 muestras)
    X_L_t, y_l, _, _, _, _ = cargar_datos_lucas()
    X_L = X_L_t.numpy()
    Y_L = np.array(y_l)
    print(f"  Lucas (2026-07-10): {X_L.shape[0]} muestras, {X_L.shape[1]} features.")

    # Candela: Sesión única 2026-09-15 con Cigomatico Mayor (202 muestras)
    d_c = np.load(os.path.join(dir_out, "dataset_candela_0915_zigomatico_conv2d.npz"), allow_pickle=True)
    X_C = d_c['X_clean']
    Y_C = d_c['Y_clean']
    print(f"  Candela (2026-09-15): {X_C.shape[0]} muestras, {X_C.shape[1]} features.")

    # Petra: Sesión única Dia 2 2026-08-28 con Cigomatico y Levator (168 muestras)
    d_p2 = np.load(os.path.join(dir_out, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_P = d_p2['X_clean']
    Y_P = d_p2['Y_clean']
    print(f"  Petra (2026-08-28 Dia 2): {X_P.shape[0]} muestras, {X_P.shape[1]} features.")

    # --------------------------------------------------------------------------
    # 2. PCA Lineal Independiente 2D por Sujeto
    # --------------------------------------------------------------------------
    print("\n[2/5] Computando PCA lineal independiente 2D...")
    pca_L = PCA(n_components=2, random_state=42).fit(X_L)
    Z_L_nat = pca_L.transform(X_L)
    evr_L = pca_L.explained_variance_ratio_

    pca_C = PCA(n_components=2, random_state=42).fit(X_C)
    Z_C_nat = pca_C.transform(X_C)
    evr_C = pca_C.explained_variance_ratio_

    pca_P = PCA(n_components=2, random_state=42).fit(X_P)
    Z_P_nat = pca_P.transform(X_P)
    evr_P = pca_P.explained_variance_ratio_

    print(f"  Lucas EVR: PC1={evr_L[0]*100:.1f}%, PC2={evr_L[1]*100:.1f}%, Total={sum(evr_L)*100:.1f}%")
    print(f"  Candela EVR: PC1={evr_C[0]*100:.1f}%, PC2={evr_C[1]*100:.1f}%, Total={sum(evr_C)*100:.1f}%")
    print(f"  Petra EVR: PC1={evr_P[0]*100:.1f}%, PC2={evr_P[1]*100:.1f}%, Total={sum(evr_P)*100:.1f}%")

    # Centroides Nativos por Vocal
    cents_L_nat = {v: np.mean(Z_L_nat[Y_L == v], axis=0) for v in VOCALES}
    cents_C_nat = {v: np.mean(Z_C_nat[Y_C == v], axis=0) for v in VOCALES}
    cents_P_nat = {v: np.mean(Z_P_nat[Y_P == v], axis=0) for v in VOCALES}

    mat_L_nat = np.array([cents_L_nat[v] for v in VOCALES])
    mat_C_nat = np.array([cents_C_nat[v] for v in VOCALES])
    mat_P_nat = np.array([cents_P_nat[v] for v in VOCALES])

    # --------------------------------------------------------------------------
    # 3. Alineación Canónica y Método de Sara Solla (CCA de Variedades Motoras)
    # --------------------------------------------------------------------------
    print("\n[3/5] Aplicando Metodo de Sara Solla (CCA y Angulos de Jordan)...")
    
    # 3a. Anclaje Canónico de la Base de Lucas (Opción B: Vértice /I/ al origen, /A/ a 90°)
    v0_L = cents_L_nat['I'].copy()
    v_dir_L = cents_L_nat['A'] - v0_L
    ang_A_L = np.arctan2(v_dir_L[1], v_dir_L[0])
    theta_rot = np.pi/2 - ang_A_L
    R = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot),  np.cos(theta_rot)]
    ])

    Z_L_can = (Z_L_nat - v0_L) @ R.T
    cents_L_can = {v: np.mean(Z_L_can[Y_L == v], axis=0) for v in VOCALES}
    mat_L_can = np.array([cents_L_can[v] for v in VOCALES])

    # 3b. CCA Candela vs Lucas
    cca_C = CCA(n_components=2)
    cca_C.fit(mat_C_nat, mat_L_can)
    mat_C_cca, mat_L_cca_c = cca_C.transform(mat_C_nat, mat_L_can)
    rho1_C = np.corrcoef(mat_C_cca[:, 0], mat_L_cca_c[:, 0])[0, 1]
    rho2_C = np.corrcoef(mat_C_cca[:, 1], mat_L_cca_c[:, 1])[0, 1]
    ang1_C = np.degrees(np.arccos(np.clip(rho1_C, -1.0, 1.0)))
    ang2_C = np.degrees(np.arccos(np.clip(rho2_C, -1.0, 1.0)))

    # Mapeo afín de Sara Solla: Candela -> Lucas Canónico
    mu_C = np.mean(mat_C_nat, axis=0)
    mu_L = np.mean(mat_L_can, axis=0)
    W_C, _, _, _ = np.linalg.lstsq(mat_C_nat - mu_C, mat_L_can - mu_L, rcond=None)
    Z_C_solla = (Z_C_nat - mu_C) @ W_C + mu_L
    cents_C_solla = {v: np.mean(Z_C_solla[Y_C == v], axis=0) for v in VOCALES}

    # 3c. CCA Petra vs Lucas
    cca_P = CCA(n_components=2)
    cca_P.fit(mat_P_nat, mat_L_can)
    mat_P_cca, mat_L_cca_p = cca_P.transform(mat_P_nat, mat_L_can)
    rho1_P = np.corrcoef(mat_P_cca[:, 0], mat_L_cca_p[:, 0])[0, 1]
    rho2_P = np.corrcoef(mat_P_cca[:, 1], mat_L_cca_p[:, 1])[0, 1]
    ang1_P = np.degrees(np.arccos(np.clip(rho1_P, -1.0, 1.0)))
    ang2_P = np.degrees(np.arccos(np.clip(rho2_P, -1.0, 1.0)))

    # Mapeo afín de Sara Solla: Petra -> Lucas Canónico
    mu_P = np.mean(mat_P_nat, axis=0)
    W_P, _, _, _ = np.linalg.lstsq(mat_P_nat - mu_P, mat_L_can - mu_L, rcond=None)
    Z_P_solla = (Z_P_nat - mu_P) @ W_P + mu_L
    cents_P_solla = {v: np.mean(Z_P_solla[Y_P == v], axis=0) for v in VOCALES}

    print("\n--- METRICAS DE ALINEACION DE SARA SOLLA (CCA Y ANGULOS DE JORDAN) ---")
    print(f"  Candela vs Lucas: rho_1 = {rho1_C:.4f} (theta_1 = {ang1_C:.2f}°), rho_2 = {rho2_C:.4f} (theta_2 = {ang2_C:.2f}°)")
    print(f"  Petra vs Lucas:   rho_1 = {rho1_P:.4f} (theta_1 = {ang1_P:.2f}°), rho_2 = {rho2_P:.4f} (theta_2 = {ang2_P:.2f}°)")

    print("\n--- ATRACTORES CANONICOS PCA ALINEADOS (Z1, Z2) ---")
    for v in VOCALES:
        cL = cents_L_can[v]
        cC = cents_C_solla[v]
        cP = cents_P_solla[v]
        print(f"  /{v.lower()}/: Lucas=({cL[0]:+5.2f}, {cL[1]:+5.2f}) | Candela=({cC[0]:+5.2f}, {cC[1]:+5.2f}) | Petra=({cP[0]:+5.2f}, {cP[1]:+5.2f})")

    # --------------------------------------------------------------------------
    # 4. FIGURA 1: Los 3 PCA Nativos (Sin Alinear, 3 Paneles)
    # --------------------------------------------------------------------------
    print("\n[4/5] Generando Figura 1: Los 3 PCA Nativos...")
    fig_nat, axes_nat = plt.subplots(1, 3, figsize=(18, 6.2), facecolor='white')

    subjs_nat = [
        ("Lucas: PCA 2D Nativo - Sesion 2026-07-10", Z_L_nat, Y_L, cents_L_nat, evr_L, axes_nat[0], "Lucas"),
        ("Candela: PCA 2D Nativo - Sesion 2026-09-15", Z_C_nat, Y_C, cents_C_nat, evr_C, axes_nat[1], "Candela"),
        ("Petra: PCA 2D Nativo - Sesion 2026-08-28 Dia 2", Z_P_nat, Y_P, cents_P_nat, evr_P, axes_nat[2], "Petra")
    ]

    for title, Z_sub, Y_sub, cents_sub, evr_sub, ax, sname in subjs_nat:
        ax.set_facecolor('white')
        ax.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')

        # Nube de puntos por vocal
        for v in VOCALES:
            mask = (Y_sub == v)
            ax.scatter(
                Z_sub[mask, 0], Z_sub[mask, 1],
                c=COLORES_VOCALES[v], alpha=0.32, s=34, edgecolors='none'
            )

        # Rayos y centroides desde /i/
        c_i = cents_sub['I']
        for v in VOCALES:
            c_v = cents_sub[v]
            ax.plot([c_i[0], c_v[0]], [c_i[1], c_v[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.0, alpha=0.85)
            ax.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], s=140, marker='D', edgecolors='#263238', linewidths=1.5, zorder=5)
            
            dx = 0.18 if c_v[0] >= c_i[0] else -0.32
            dy = 0.16 if c_v[1] >= c_i[1] else -0.25
            txt = ax.text(c_v[0] + dx, c_v[1] + dy, f"/{v.lower()}/",
                          fontsize=12, fontweight='bold', color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
            txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

        # Ajuste de límites estricto por sujeto
        pad_x = 0.08 * (Z_sub[:, 0].max() - Z_sub[:, 0].min())
        pad_y = 0.08 * (Z_sub[:, 1].max() - Z_sub[:, 1].min())
        ax.set_xlim(Z_sub[:, 0].min() - pad_x, Z_sub[:, 0].max() + pad_x)
        ax.set_ylim(Z_sub[:, 1].min() - pad_y, Z_sub[:, 1].max() + pad_y)

        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel(f"PC1: {evr_sub[0]*100:.1f}% Varianza", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"PC2: {evr_sub[1]*100:.1f}% Varianza", fontsize=11, fontweight='bold')

    plt.tight_layout()
    p_fig_nat = os.path.join(dir_out, "comparativa_pca_nativos_tri_sujeto.png")
    plt.savefig(p_fig_nat, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Figura 1 guardada en: {p_fig_nat}")

    # --------------------------------------------------------------------------
    # 5. FIGURA 2: Comparativa Canónica Tri-Sujeto Alineada con Sara Solla (4 Paneles)
    # --------------------------------------------------------------------------
    print("\n[5/5] Generando Figura 2: Comparativa Sara Solla (4 Paneles)...")
    fig, axes = plt.subplots(1, 4, figsize=(24, 6.0), facecolor='white')

    # Bounding box común para comparar en idéntica escala
    all_X = np.concatenate([Z_L_can[:, 0], Z_C_solla[:, 0], Z_P_solla[:, 0]])
    all_Y = np.concatenate([Z_L_can[:, 1], Z_C_solla[:, 1], Z_P_solla[:, 1]])
    
    # Rango estricto sin espacios vacíos
    x_min = np.percentile(all_X, 0.5) - 0.4
    x_max = np.percentile(all_X, 99.5) + 0.5
    y_min = np.percentile(all_Y, 0.5) - 0.4
    y_max = np.percentile(all_Y, 99.5) + 0.5

    # Panel 1: Lucas Canónico Referencia
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        mask = (Y_L == v)
        ax1.scatter(Z_L_can[mask, 0], Z_L_can[mask, 1], c=COLORES_VOCALES[v], alpha=0.32, s=30, edgecolors='none')
        c = cents_L_can[v]
        ax1.plot([cents_L_can['I'][0], c[0]], [cents_L_can['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.2, alpha=0.85)
        ax1.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=140, marker='D', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax1.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax1.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax1.set_title("Lucas: PCA Canonico de Referencia", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)

    # Panel 2: Candela Alineada CCA Sara Solla
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        mask = (Y_C == v)
        ax2.scatter(Z_C_solla[mask, 0], Z_C_solla[mask, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_C_solla[v]
        ax2.plot([cents_C_solla['I'][0], c[0]], [cents_C_solla['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=2.2, alpha=0.85)
        ax2.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=130, marker='s', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax2.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax2.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax2.set_title(f"Candela: Alineada Sara Solla CCA - rho={rho1_C:.3f}", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)

    # Panel 3: Petra Alineada CCA Sara Solla
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        mask = (Y_P == v)
        ax3.scatter(Z_P_solla[mask, 0], Z_P_solla[mask, 1], c=COLORES_VOCALES[v], alpha=0.38, s=34, edgecolors='none')
        c = cents_P_solla[v]
        ax3.plot([cents_P_solla['I'][0], c[0]], [cents_P_solla['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle=':', linewidth=2.4, alpha=0.85)
        ax3.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=170, marker='*', edgecolors='#263238', linewidths=1.3, zorder=5)
        txt = ax3.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax3.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax3.set_title(f"Petra: Alineada Sara Solla CCA - rho={rho1_P:.3f}", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)

    # Panel 4: Superposición Tri-Sujeto
    ax4 = axes[3]
    ax4.set_facecolor('white')
    ax4.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')

    for v in VOCALES:
        cL = cents_L_can[v]
        cC = cents_C_solla[v]
        cP = cents_P_solla[v]

        ax4.plot([cents_L_can['I'][0], cL[0]], [cents_L_can['I'][1], cL[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.0, alpha=0.85)
        ax4.plot([cents_C_solla['I'][0], cC[0]], [cents_C_solla['I'][1], cC[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=1.8, alpha=0.75)
        ax4.plot([cents_P_solla['I'][0], cP[0]], [cents_P_solla['I'][1], cP[1]], color=COLORES_VOCALES[v], linestyle=':', linewidth=2.0, alpha=0.75)

        ax4.scatter(cL[0], cL[1], c=COLORES_VOCALES[v], s=130, marker='D', edgecolors='#263238', linewidths=1.2, zorder=6)
        ax4.scatter(cC[0], cC[1], c=COLORES_VOCALES[v], s=110, marker='s', edgecolors='#263238', linewidths=1.2, zorder=6)
        ax4.scatter(cP[0], cP[1], c=COLORES_VOCALES[v], s=160, marker='*', edgecolors='#263238', linewidths=1.2, zorder=6)

        txt = ax4.text(cL[0] + 0.15, cL[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    h_l = ax4.scatter([], [], marker='D', color='#455A64', s=70, label='Lucas: Sesion Benchmark')
    h_c = ax4.scatter([], [], marker='s', color='#455A64', s=60, label='Candela: Cigomatico Mayor')
    h_p = ax4.scatter([], [], marker='*', color='#455A64', s=90, label='Petra: Cigomatico y Levator')
    ax4.legend(handles=[h_l, h_c, h_p], loc='upper left', framealpha=0.9, fontsize=9.5)

    ax4.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax4.set_title("Superposicion Tri-Sujeto: Variedades PCA Alineadas", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)

    plt.tight_layout()
    p_fig_solla = os.path.join(dir_out, "comparativa_pca_sara_solla_tri_sujeto.png")
    plt.savefig(p_fig_solla, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Figura 2 guardada en: {p_fig_solla}")
    print("=" * 80)

if __name__ == "__main__":
    main()
