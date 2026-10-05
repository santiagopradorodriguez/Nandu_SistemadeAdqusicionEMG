#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento 1: Estabilidad Inter-Día y Alineación CCA en Sujeto Petra
Compara dos jornadas independientes separadas por una semana:
- Día 1: 2026-08-21 (Prueba1silicona y silicona_aeiou - 10 tomas)
- Día 2: 2026-08-28 (med1_clase24 y med2_clase4 - 10 tomas)
Tríada Muscular común: Anterior Belly (CH0), Zygomaticus Major (CH1), Levator Anguli Oris (CH2).
Modelo: Campeón Convolucional Ortogonal 2D.
Estética: Fondo blanco limpio, nubes por vocal, vectores radiales y atractores sin GMM.
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.signal import butter, filtfilt
from sklearn.ensemble import IsolationForest
from sklearn.cross_decomposition import CCA
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
# 2. Extracción y Normalización de Petra
# ------------------------------------------------------------------------------
def extraer_sesion_agnostica(toma_str):
    s = str(toma_str)
    m = re.search(r'(med\d+|Prueba\d+|silicona_aeiou|Serie\d+|Sesion\d+|Session\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    parts = s.split('_')
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win') or p_clean.lower().startswith('w'):
            continue
        if any(char.isdigit() for char in p_clean) and len(p_clean) <= 12:
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

def cargar_o_extraer_dia(fecha, nombre_tag, out_dir):
    cache_path = os.path.join(out_dir, f"dataset_petra_{nombre_tag}_conv2d.npz")
    if os.path.exists(cache_path):
        print(f"[Caché] Cargando dataset Petra {nombre_tag} ({fecha}):")
        print(f"        {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X_clean'], d['Y_clean']

    base_fecha = os.path.join(project_root, f"EMG_desarrollo/base_de_datos_electrodos/{fecha}")
    tomas_todas = sorted([t for t in os.listdir(base_fecha) if os.path.isdir(os.path.join(base_fecha, t))])
    tomas_petra = [t for t in tomas_todas if t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']]

    print(f"[Extracción {nombre_tag}] Procesando {len(tomas_petra)} tomas de Petra en {fecha}...")
    X_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_fecha,
        mediciones=tomas_petra,
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
    print(f"[Caché] Guardado dataset Petra {nombre_tag}: {len(Y_clean)} ventanas válidas post-purga.")
    return X_clean, Y_clean

# ------------------------------------------------------------------------------
# 3. Flujo Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 80)
    print("EXPERIMENTO 1: ESTABILIDAD INTER-DIA EN PETRA (2026-08-21 VS 2026-08-28)")
    print("=" * 80)

    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_ckpt = os.path.join(dir_conv, "modelo_campeon_conv_ortogonal.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
    model.eval()

    # 1. Cargar Día 1 (2026-08-21) y Día 2 (2026-08-28)
    X_D1, Y_D1 = cargar_o_extraer_dia("2026-08-21", "dia1_20260821", dir_conv)
    X_D2, Y_D2 = cargar_o_extraer_dia("2026-08-28", "dia2_20260828", dir_conv)

    # 2. Proyecciones Latentes en R^2
    with torch.no_grad():
        Z_D1 = model.encode(torch.tensor(X_D1, dtype=torch.float32)).numpy()
        Z_D2 = model.encode(torch.tensor(X_D2, dtype=torch.float32)).numpy()

    print(f"\n[Proyecciones 2D Petra]")
    print(f"  Día 1 (2026-08-21): {Z_D1.shape[0]} muestras")
    print(f"  Día 2 (2026-08-28): {Z_D2.shape[0]} muestras")

    # 3. Atractores Centroides por Vocal
    cents_D1 = {v: np.mean(Z_D1[Y_D1 == v], axis=0) for v in VOCALES}
    cents_D2 = {v: np.mean(Z_D2[Y_D2 == v], axis=0) for v in VOCALES}
    matriz_D1 = np.array([cents_D1[v] for v in VOCALES])  # (5, 2)
    matriz_D2 = np.array([cents_D2[v] for v in VOCALES])  # (5, 2)

    print("\n" + "-" * 80)
    print("ATRACTORES PETRA INTER-DIA - COORDENADAS MEDIAS, NORMAS Y DESVIO ANGULAR")
    print("-" * 80)
    print(f"{'Vocal':<6} | {'Día 1: 21-08 (Z1, Z2)':<25} | {'||z1||':<7} | {'Día 2: 28-08 (Z1, Z2)':<25} | {'||z2||':<7} | {'Angulo Dif'}")
    print("-" * 80)

    angulos_interdia = {}
    for v in VOCALES:
        c1 = cents_D1[v]
        c2 = cents_D2[v]
        n1 = np.linalg.norm(c1)
        n2 = np.linalg.norm(c2)

        cos_sim = np.dot(c1, c2) / max(1e-7, (n1 * n2))
        ang_deg = np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0)))
        angulos_interdia[v] = ang_deg

        str1 = f"({c1[0]:+.2f}, {c1[1]:+.2f})"
        str2 = f"({c2[0]:+.2f}, {c2[1]:+.2f})"
        print(f" /{v.lower()}/   | {str1:<25} | {n1:<7.2f} | {str2:<25} | {n2:<7.2f} | {ang_deg:>6.1f}°")

    # 4. Análisis Canónico de Jordan y CCA
    print("\n" + "-" * 80)
    print("ANALISIS CANONICO DE JORDAN Y CORRELACIONES CANONICAS (PETRA INTER-DIA)")
    print("-" * 80)

    subspace_rad = la.subspace_angles(matriz_D1.T, matriz_D2.T)
    subspace_deg = np.degrees(subspace_rad)
    for idx_ang, d in enumerate(subspace_deg, 1):
        print(f"  Theta_{idx_ang} (Jordan): {d:.2f}°  -->  cos(Theta_{idx_ang}) = {np.cos(np.radians(d)):.4f}")

    cca = CCA(n_components=2)
    cca.fit(matriz_D2, matriz_D1)
    Z_D2_cca, Z_D1_cca = cca.transform(matriz_D2, matriz_D1)
    r1 = np.corrcoef(Z_D2_cca[:, 0], Z_D1_cca[:, 0])[0, 1]
    r2 = np.corrcoef(Z_D2_cca[:, 1], Z_D1_cca[:, 1])[0, 1]
    print(f"\nCorrelaciones Canónicas (CCA Inter-Día):")
    print(f"  Dimensión Canónica 1: rho_1 = {r1:.4f}  (Ángulo efectivo: {np.degrees(np.arccos(np.clip(r1, -1, 1))):.2f}°)")
    print(f"  Dimensión Canónica 2: rho_2 = {r2:.4f}  (Ángulo efectivo: {np.degrees(np.arccos(np.clip(r2, -1, 1))):.2f}°)")

    # 5. Alineación Afín / CCA de Día 2 hacia Día 1
    W_aff, _, _, _ = np.linalg.lstsq(matriz_D2 - np.mean(matriz_D2, axis=0), matriz_D1 - np.mean(matriz_D1, axis=0), rcond=None)
    Z_D2_aligned = (Z_D2 - np.mean(matriz_D2, axis=0)) @ W_aff + np.mean(matriz_D1, axis=0)
    cents_D2_aligned = {v: np.mean(Z_D2_aligned[Y_D2 == v], axis=0) for v in VOCALES}

    # --------------------------------------------------------------------------
    # 6. Graficado Limpio en 3 Paneles (Estética Oficial, Sin GMM)
    # --------------------------------------------------------------------------
    all_pts = np.vstack([Z_D1, Z_D2, Z_D2_aligned])
    max_val = max(np.abs(all_pts).max() * 1.15, 3.5)
    lim = (-max_val, max_val)

    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), facecolor='white')

    # Panel 1: Petra Día 1 (2026-08-21)
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m1 = (Y_D1 == v)
        ax1.scatter(Z_D1[m1, 0], Z_D1[m1, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=55, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        c1 = cents_D1[v]
        ax1.plot([0, c1[0]], [0, c1[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax1.scatter(c1[0], c1[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=200, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax1.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax1.set_title("Petra Día 1 (2026-08-21): Atractores Nativos", fontsize=13, fontweight='bold', pad=12)
    ax1.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(lim)
    ax1.set_ylim(lim)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 2: Petra Día 2 (2026-08-28)
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        m2 = (Y_D2 == v)
        ax2.scatter(Z_D2[m2, 0], Z_D2[m2, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=55, edgecolors='black', linewidth=0.4, alpha=0.75, zorder=3)
        c2 = cents_D2[v]
        ax2.plot([0, c2[0]], [0, c2[1]], color=COLORES_VOCALES[v], linewidth=2.8, alpha=0.9, zorder=4)
        ax2.scatter(c2[0], c2[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], marker='D', s=200, edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])

    ax2.scatter(0, 0, c='black', s=70, marker='x', zorder=6, label='Reposo (0,0)')
    ax2.set_title("Petra Día 2 (2026-08-28): Atractores Nativos", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(lim)
    ax2.set_ylim(lim)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='lower left', fontsize=10, frameon=True)

    # Panel 3: Superposición Inter-Día
    ax3 = axes[2]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)

    for v in VOCALES:
        c1 = cents_D1[v]
        c2 = cents_D2[v]
        c2_al = cents_D2_aligned[v]

        # Día 1: Línea sólida + Círculo
        ax3.plot([0, c1[0]], [0, c1[1]], color=COLORES_VOCALES[v], linewidth=3.2, linestyle='-', alpha=0.9, zorder=3, label=f"Día 1 /{v.lower()}/")
        ax3.scatter(c1[0], c1[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], s=180, marker='o', edgecolors='black', linewidth=1.5, zorder=5, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        # Día 2 Raw: Línea discontinua + Diamante rojo
        ax3.plot([0, c2[0]], [0, c2[1]], color=COLORES_VOCALES[v], linewidth=2.5, linestyle='--', alpha=0.85, zorder=4)
        ax3.scatter(c2[0], c2[1], c=[adjust_lightness(COLORES_VOCALES[v], 0.65)], s=180, marker='D', edgecolors='#C0392B', linewidth=1.5, zorder=6, path_effects=[pe.withStroke(linewidth=3, foreground="white", alpha=0.8)])

        # Día 2 Alineado con CCA: Estrella dorada pequeña
        ax3.scatter(c2_al[0], c2_al[1], c=[COLORES_VOCALES[v]], s=140, marker='*', edgecolors='black', linewidth=1.0, zorder=7)

    ax3.scatter(0, 0, c='black', s=70, marker='x', zorder=8)
    ax3.set_title("Superposición Inter-Día: Día 1 (-) vs Día 2 Raw (--) vs CCA (*)", fontsize=13, fontweight='bold', pad=12)
    ax3.set_xlabel("Coordenada Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(lim)
    ax3.set_ylim(lim)
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='lower left', fontsize=10, frameon=True)

    plt.tight_layout()
    p_fig = os.path.join(dir_conv, "experimento_1_petra_interdia_atractores_cca.png")
    fig.savefig(p_fig, dpi=160)
    plt.close(fig)
    print(f"\n[Figura Guardada] {p_fig}")
    print("[OK] Experimento 1 completado con éxito.")

if __name__ == '__main__':
    main()
