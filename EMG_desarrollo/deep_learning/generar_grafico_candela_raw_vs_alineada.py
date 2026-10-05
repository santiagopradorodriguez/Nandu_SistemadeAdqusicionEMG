#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa Oficial: Candela RAW (Sin Alinear) vs Candela Alineada (CCA)
Evaluadas sobre las Fronteras de Decisión GMM de Lucas.
Estética oficial idéntica a motor_autoencoder_unificado.py.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import seaborn as sns
import torch
import torch.nn as nn

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.generar_reporte_alineacion_candela_lucas import (
    ParametricConvOrthogonalAE,
    cargar_lucas_campeon,
    cargar_candela_zigomatico,
    adjust_lightness,
    VOCALES,
    VOCAL_TO_IDX,
    COLORES_VOCALES
)

def main():
    print("=" * 80)
    print("GENERANDO REPORTE DE CANDELA SIN ALINEAR VS CANDELA ALINEADA")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # 1. Lucas y su GMM
    Z_L, Y_L, gmm, cluster_to_vocal, acc_lucas = cargar_lucas_campeon()
    cents_L = np.array([np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES])

    # 2. Candela RAW
    X_C, Y_C = cargar_candela_zigomatico()
    with torch.no_grad():
        Z_C_raw = model.encode(torch.tensor(X_C, dtype=torch.float32)).numpy()
    cents_C_raw = np.array([np.mean(Z_C_raw[Y_C == v], axis=0) for v in VOCALES])

    y_idx_cande = np.array([VOCAL_TO_IDX[v] for v in Y_C])
    preds_raw = np.array([cluster_to_vocal[c] for c in gmm.predict(Z_C_raw)])
    acc_raw = accuracy_score(y_idx_cande, preds_raw) * 100.0
    print(f"[Candela RAW] Exactitud sin alinear: {acc_raw:.2f}%")

    # 3. Candela Alineada (CCA / Afín)
    W_aff, _, _, _ = np.linalg.lstsq(cents_C_raw - np.mean(cents_C_raw, axis=0), cents_L - np.mean(cents_L, axis=0), rcond=None)
    Z_C_aligned = (Z_C_raw - np.mean(cents_C_raw, axis=0)) @ W_aff + np.mean(cents_L, axis=0)
    cents_C_aligned = np.array([np.mean(Z_C_aligned[Y_C == v], axis=0) for v in VOCALES])

    preds_aligned = np.array([cluster_to_vocal[c] for c in gmm.predict(Z_C_aligned)])
    acc_aligned = accuracy_score(y_idx_cande, preds_aligned) * 100.0
    print(f"[Candela Alineada] Exactitud tras CCA: {acc_aligned:.2f}%")

    # 4. Malla de Fronteras de Decisión GMM de Lucas
    all_points = np.vstack([Z_L, Z_C_raw, Z_C_aligned])
    xr = all_points[:, 0].max() - all_points[:, 0].min()
    yr = all_points[:, 1].max() - all_points[:, 1].min()
    margin = 0.12
    x_min, x_max = all_points[:, 0].min() - xr * margin, all_points[:, 0].max() + xr * margin
    y_min, y_max = all_points[:, 1].min() - yr * margin, all_points[:, 1].max() + yr * margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 450), np.linspace(y_min, y_max, 450))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds_grid = gmm.predict(grid)
    grid_mapped = np.array([cluster_to_vocal[p] for p in preds_grid]).reshape(xx.shape)

    palette_list = [COLORES_VOCALES[v] for v in VOCALES]
    cmap_mesh = mcolors.ListedColormap(palette_list)

    # --------------------------------------------------------------------------
    # FIGURA 1: CANDELA RAW (SIN ALINEAR) SOBRE FRONTERAS DE LUCAS
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
    ax_raw = axes1[0]
    ax_raw.set_facecolor('white')

    ax_raw.pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
    ax_raw.contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.6, alpha=0.5, zorder=1)

    for idx, v in enumerate(VOCALES):
        mask_v = (Y_C == v)
        # Puntos crudos
        ax_raw.scatter(
            Z_C_raw[mask_v, 0], Z_C_raw[mask_v, 1],
            c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/",
            s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4
        )
        # Diamante: Atractor crudo de Candela
        cen_c = cents_C_raw[idx]
        ax_raw.scatter(
            cen_c[0], cen_c[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)],
            marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)]
        )
        # Cruz negra: Referencia de Lucas
        cen_l = cents_L[idx]
        ax_raw.scatter(cen_l[0], cen_l[1], marker='+', s=120, c='black', linewidths=2.0, zorder=6)

    ax_raw.set_title(f"Candela (Sin Alinear): Proyección Cruda en Modelo de Lucas\nExactitud en Fronteras GMM de Lucas: {acc_raw:.2f}%", fontsize=15, fontweight='bold', pad=12)
    ax_raw.set_xlabel("Coordenada Z1", fontsize=13, fontweight='bold')
    ax_raw.set_ylabel("Coordenada Z2", fontsize=13, fontweight='bold')
    ax_raw.legend(loc='upper right', fontsize=11, frameon=True)
    ax_raw.grid(True, linestyle=':', alpha=0.6)
    ax_raw.set_xlim(x_min, x_max)
    ax_raw.set_ylim(y_min, y_max)

    # Matriz de confusión Candela RAW
    cm_raw = confusion_matrix(y_idx_cande, preds_raw, labels=range(5))
    cm_pct_raw = cm_raw.astype(float) / np.maximum(cm_raw.sum(axis=1, keepdims=True), 1e-6) * 100

    sns.heatmap(cm_pct_raw, annot=True, fmt='.1f', cmap='Blues', xticklabels=VOCALES, yticklabels=VOCALES, ax=axes1[1], cbar=False, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes1[1].set_title(f"Matriz de Confusión: Candela Sin Alinear - {acc_raw:.2f}%", fontsize=14, fontweight='bold', pad=12)
    axes1[1].set_xlabel("Vocal Predicha (Fronteras Lucas)", fontsize=12, fontweight='bold')
    axes1[1].set_ylabel("Vocal Real Ground Truth", fontsize=12, fontweight='bold')

    plt.tight_layout()
    p_fig1 = os.path.join(dir_conv, "reporte_oficial_candela_raw_sin_alinear.png")
    fig1.savefig(p_fig1, dpi=160)
    plt.close(fig1)
    print(f"[Guardado] {p_fig1}")

    # --------------------------------------------------------------------------
    # FIGURA 2: COMPARATIVA DIRECTA RAW VS ALINEADA (LADO A LADO)
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

    for ax, data_z, cents_data, ttl, acc_val in [
        (axes2[0], Z_C_raw, cents_C_raw, f"Candela Sin Alinear: {acc_raw:.2f}% (Fronteras Lucas)", acc_raw),
        (axes2[1], Z_C_aligned, cents_C_aligned, f"Candela Alineada con CCA: {acc_aligned:.2f}% (Fronteras Lucas)", acc_aligned)
    ]:
        ax.set_facecolor('white')
        ax.pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
        ax.contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.6, alpha=0.5, zorder=1)

        for idx, v in enumerate(VOCALES):
            m_v = (Y_C == v)
            ax.scatter(data_z[m_v, 0], data_z[m_v, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4)
            cen = cents_data[idx]
            ax.scatter(cen[0], cen[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])
            # Centroide de Lucas de referencia
            cen_l = cents_L[idx]
            ax.scatter(cen_l[0], cen_l[1], marker='+', s=120, c='black', linewidths=2.0, zorder=6)

        ax.set_title(ttl, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Coordenada Z1", fontsize=13, fontweight='bold')
        ax.set_ylabel("Coordenada Z2", fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    p_fig2 = os.path.join(dir_conv, "comparativa_candela_raw_vs_alineada_fronteras_lucas.png")
    fig2.savefig(p_fig2, dpi=160)
    plt.close(fig2)
    print(f"[Guardado] {p_fig2}")

    print("\n[OK] Reportes de Candela sin alinear generados con éxito.")

if __name__ == '__main__':
    main()
