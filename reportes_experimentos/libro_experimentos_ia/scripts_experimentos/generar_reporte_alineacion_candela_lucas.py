#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alineación CCA y Fronteras de Decisión GMM: Lucas vs Candela (2026-09-15)
Estética oficial idéntica a motor_autoencoder_unificado.py:
- Fondo blanco limpio (facecolor='white').
- Malla de regiones de decisión con pcolormesh y mapa de colores oficial.
- Contornos de frontera en negro fino.
- Puntos de datos con borde negro.
- Centroides / Atractores en diamantes grandes con resplandor blanco (path_effects).
- Matriz de confusión en mapa de calor con porcentajes y recuentos.
"""

import os
import sys
import re
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest
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

# ------------------------------------------------------------------------------
# 2. Carga y Normalización de Datos
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

def cargar_lucas_campeon():
    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_csv_lucas = os.path.join(dir_conv, "proyecciones_campeon_lucas.csv")
    df = pd.read_csv(p_csv_lucas)
    Z_L = df[['Z1', 'Z2']].values
    Y_L = df['Vocal'].values

    # Ajustar GMM de Lucas
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    pred_raw = gmm.fit_predict(Z_L)
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(VOCALES):
        for j in range(5):
            contingency[j, i] = np.sum((Y_L == vl) & (pred_raw == j))
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    
    pred_idx = np.array([cluster_to_vocal[c] for c in pred_raw])
    y_idx = np.array([VOCAL_TO_IDX[v] for v in Y_L])
    acc_lucas = accuracy_score(y_idx, pred_idx) * 100.0

    return Z_L, Y_L, gmm, cluster_to_vocal, acc_lucas

def cargar_candela_zigomatico():
    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    cache_path = os.path.join(dir_conv, "dataset_candela_0915_zigomatico_conv2d.npz")
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        return d['X_clean'], d['Y_clean']

    base_cande = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-09-16")
    tomas_todas = sorted([t for t in os.listdir(base_cande) if os.path.isdir(os.path.join(base_cande, t))])
    tomas_cande = [t for t in tomas_todas if t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']]

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
    return X_clean, Y_clean

# ------------------------------------------------------------------------------
# 3. Flujo Principal y Graficación Oficial
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("GENERANDO REPORTES CON ESTETICA OFICIAL DE AUTOENCODERS")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    # 1. Cargar Lucas y su GMM
    Z_L, Y_L, gmm, cluster_to_vocal, acc_lucas = cargar_lucas_campeon()
    print(f"[Lucas] Exactitud GMM: {acc_lucas:.2f}% | {Z_L.shape[0]} muestras")

    # 2. Cargar y Proyectar Candela
    X_C, Y_C = cargar_candela_zigomatico()
    with torch.no_grad():
        Z_C = model.encode(torch.tensor(X_C, dtype=torch.float32)).numpy()
    print(f"[Candela] Proyectada en modelo de Lucas: {Z_C.shape[0]} muestras")

    # 3. Atractores Centroides por Vocal
    cents_L = np.array([np.mean(Z_L[Y_L == v], axis=0) for v in VOCALES])  # (5, 2)
    cents_C = np.array([np.mean(Z_C[Y_C == v], axis=0) for v in VOCALES])  # (5, 2)

    # 4. Alineación Afín / CCA de Candela hacia Lucas
    # Transformación afín: Z_C_aligned = (Z_C - mean_C) @ W + mean_L
    # Calculamos W óptima por mínimos cuadrados sobre los atractores:
    cents_C_cent = cents_C - np.mean(cents_C, axis=0)
    cents_L_cent = cents_L - np.mean(cents_L, axis=0)
    W_affine, _, _, _ = np.linalg.lstsq(cents_C_cent, cents_L_cent, rcond=None)

    Z_C_aligned = (Z_C - np.mean(cents_C, axis=0)) @ W_affine + np.mean(cents_L, axis=0)
    cents_C_aligned = np.array([np.mean(Z_C_aligned[Y_C == v], axis=0) for v in VOCALES])

    # 5. Evaluación de Candela Alineada en el GMM de Lucas
    pred_raw_cande = gmm.predict(Z_C_aligned)
    pred_idx_cande = np.array([cluster_to_vocal[c] for c in pred_raw_cande])
    y_idx_cande = np.array([VOCAL_TO_IDX[v] for v in Y_C])
    acc_cande = accuracy_score(y_idx_cande, pred_idx_cande) * 100.0
    print(f"[Candela Alineada] Exactitud en fronteras de Lucas: {acc_cande:.2f}%")

    # --------------------------------------------------------------------------
    # 6. Preparación de Malla de Fronteras de Decisión
    # --------------------------------------------------------------------------
    all_z = np.vstack([Z_L, Z_C_aligned])
    xr = all_z[:, 0].max() - all_z[:, 0].min()
    yr = all_z[:, 1].max() - all_z[:, 1].min()
    margin = 0.12
    x_min, x_max = all_z[:, 0].min() - xr * margin, all_z[:, 0].max() + xr * margin
    y_min, y_max = all_z[:, 1].min() - yr * margin, all_z[:, 1].max() + yr * margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 450), np.linspace(y_min, y_max, 450))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds_grid = gmm.predict(grid)
    grid_mapped = np.array([cluster_to_vocal[p] for p in preds_grid]).reshape(xx.shape)

    palette_list = [COLORES_VOCALES[v] for v in VOCALES]
    cmap_mesh = mcolors.ListedColormap(palette_list)

    # --------------------------------------------------------------------------
    # FIGURA 1: LUCAS (CANÓNICO CON FRONTERAS DE DECISIÓN Y MATRIZ)
    # --------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
    ax_l = axes1[0]
    ax_l.set_facecolor('white')

    ax_l.pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
    ax_l.contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.6, alpha=0.5, zorder=1)

    for idx, v in enumerate(VOCALES):
        mask_v = (Y_L == v)
        ax_l.scatter(
            Z_L[mask_v, 0], Z_L[mask_v, 1],
            c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/",
            s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4
        )
        cen = cents_L[idx]
        ax_l.scatter(
            cen[0], cen[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)],
            marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)]
        )

    ax_l.set_title(f"Espacio Canónico 2D: Lucas (Entrenamiento)\nExactitud GMM No Supervisada: {acc_lucas:.2f}%", fontsize=15, fontweight='bold', pad=12)
    ax_l.set_xlabel("Coordenada Z1", fontsize=13, fontweight='bold')
    ax_l.set_ylabel("Coordenada Z2", fontsize=13, fontweight='bold')
    ax_l.legend(loc='upper right', fontsize=11, frameon=True)
    ax_l.grid(True, linestyle=':', alpha=0.6)
    ax_l.set_xlim(x_min, x_max)
    ax_l.set_ylim(y_min, y_max)

    # Matriz de confusión Lucas
    y_idx_lucas = np.array([VOCAL_TO_IDX[v] for v in Y_L])
    pred_raw_lucas = gmm.predict(Z_L)
    pred_idx_lucas = np.array([cluster_to_vocal[c] for c in pred_raw_lucas])
    cm_l = confusion_matrix(y_idx_lucas, pred_idx_lucas, labels=range(5))
    cm_pct_l = cm_l.astype(float) / np.maximum(cm_l.sum(axis=1, keepdims=True), 1e-6) * 100

    sns.heatmap(cm_pct_l, annot=True, fmt='.1f', cmap='Blues', xticklabels=VOCALES, yticklabels=VOCALES, ax=axes1[1], cbar=False, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes1[1].set_title(f"Matriz de Confusión: Lucas - {acc_lucas:.2f}%", fontsize=14, fontweight='bold', pad=12)
    axes1[1].set_xlabel("Vocal Predicha", fontsize=12, fontweight='bold')
    axes1[1].set_ylabel("Vocal Real Ground Truth", fontsize=12, fontweight='bold')

    plt.tight_layout()
    p_fig1 = os.path.join(dir_conv, "reporte_oficial_lucas_2d_fronteras.png")
    fig1.savefig(p_fig1, dpi=160)
    plt.close(fig1)
    print(f"[Guardado] {p_fig1}")

    # --------------------------------------------------------------------------
    # FIGURA 2: CANDELA ALINEADA EN FRONTERAS DE LUCAS
    # --------------------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
    ax_c = axes2[0]
    ax_c.set_facecolor('white')

    # Fronteras de Lucas de fondo
    ax_c.pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
    ax_c.contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.6, alpha=0.5, zorder=1)

    for idx, v in enumerate(VOCALES):
        mask_v = (Y_C == v)
        # Puntos de Candela alineados
        ax_c.scatter(
            Z_C_aligned[mask_v, 0], Z_C_aligned[mask_v, 1],
            c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/",
            s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4
        )
        # Diamante: Atractor de Candela alineado
        cen_c = cents_C_aligned[idx]
        ax_c.scatter(
            cen_c[0], cen_c[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)],
            marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5,
            path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)]
        )
        # Cruz fina: Centroide de referencia de Lucas
        cen_l = cents_L[idx]
        ax_c.scatter(cen_l[0], cen_l[1], marker='+', s=120, c='black', linewidths=2.0, zorder=6)

    ax_c.set_title(f"Candela (2026-09-15 Tríada Zygomaticus): Alineación CCA\nEvaluada en Fronteras GMM de Lucas: {acc_cande:.2f}%", fontsize=15, fontweight='bold', pad=12)
    ax_c.set_xlabel("Coordenada Z1", fontsize=13, fontweight='bold')
    ax_c.set_ylabel("Coordenada Z2", fontsize=13, fontweight='bold')
    ax_c.legend(loc='upper right', fontsize=11, frameon=True)
    ax_c.grid(True, linestyle=':', alpha=0.6)
    ax_c.set_xlim(x_min, x_max)
    ax_c.set_ylim(y_min, y_max)

    # Matriz de confusión Candela
    cm_c = confusion_matrix(y_idx_cande, pred_idx_cande, labels=range(5))
    cm_pct_c = cm_c.astype(float) / np.maximum(cm_c.sum(axis=1, keepdims=True), 1e-6) * 100

    sns.heatmap(cm_pct_c, annot=True, fmt='.1f', cmap='Blues', xticklabels=VOCALES, yticklabels=VOCALES, ax=axes2[1], cbar=False, annot_kws={'fontsize': 14, 'fontweight': 'bold'})
    axes2[1].set_title(f"Matriz de Confusión: Candela en Modelo Lucas - {acc_cande:.2f}%", fontsize=14, fontweight='bold', pad=12)
    axes2[1].set_xlabel("Vocal Predicha (Fronteras Lucas)", fontsize=12, fontweight='bold')
    axes2[1].set_ylabel("Vocal Real Ground Truth", fontsize=12, fontweight='bold')

    plt.tight_layout()
    p_fig2 = os.path.join(dir_conv, "reporte_oficial_candela_alineada_fronteras_lucas.png")
    fig2.savefig(p_fig2, dpi=160)
    plt.close(fig2)
    print(f"[Guardado] {p_fig2}")

    # --------------------------------------------------------------------------
    # FIGURA 3: COMPARATIVA DIRECTA LADO A LADO
    # --------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

    for ax, data_z, data_y, cents_data, ttl, acc_val in [
        (axes3[0], Z_L, Y_L, cents_L, f"Lucas (Canónico): {acc_lucas:.2f}%", acc_lucas),
        (axes3[1], Z_C_aligned, Y_C, cents_C_aligned, f"Candela Alineada (CCA): {acc_cande:.2f}%", acc_cande)
    ]:
        ax.set_facecolor('white')
        ax.pcolormesh(xx, yy, grid_mapped, cmap=cmap_mesh, alpha=0.25, zorder=0, shading='auto')
        ax.contour(xx, yy, grid_mapped, levels=np.arange(0.5, len(VOCALES) - 0.5, 1), colors='k', linewidths=0.6, alpha=0.5, zorder=1)

        for idx, v in enumerate(VOCALES):
            m_v = (data_y == v)
            ax.scatter(data_z[m_v, 0], data_z[m_v, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=70, edgecolors='black', linewidth=0.5, alpha=0.85, zorder=4)
            cen = cents_data[idx]
            ax.scatter(cen[0], cen[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=220, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

        ax.set_title(ttl, fontsize=15, fontweight='bold', pad=12)
        ax.set_xlabel("Coordenada Z1", fontsize=13, fontweight='bold')
        ax.set_ylabel("Coordenada Z2", fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    p_fig3 = os.path.join(dir_conv, "comparativa_directa_lucas_vs_candela_alineada.png")
    fig3.savefig(p_fig3, dpi=160)
    plt.close(fig3)
    print(f"[Guardado] {p_fig3}")

    print("\n[OK] Generación de reportes con estética oficial completada.")

if __name__ == '__main__':
    main()
