#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativa de Espacios Latentes y Atractores: Lucas vs Candela (2026-09-01)
Modelo Base: Campeón MLP 3D sin SO2 (Ventana 40/60, Hitos 130+)
Calcula:
- Proyecciones latentes Z_L y Z_C (sin ninguna rotacion previa).
- Atractores centroides por vocal (vectores radiales desde el origen).
- Angulos relativos entre atractores individuales.
- Angulos Canonicos de Jordan y Correlaciones Canonicas (CCA).
- Graficos comparativos en 3D y proyecciones 2D (Z1-Z2 y Z1-Z3).
"""

import os
import sys
import re
import json
import numpy as np
import scipy.linalg as la
from scipy.signal import butter, filtfilt
from sklearn.ensemble import IsolationForest
from sklearn.cross_decomposition import CCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn

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
# 1. Definicion de Arquitectura MLP 3D
# ------------------------------------------------------------------------------
class ParametricMLPOrthogonalAE(nn.Module):
    def __init__(self, input_dim=60, hidden_dims=(48, 24), latent_dim=3, act_name='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        h1, h2 = hidden_dims
        
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, latent_dim, bias=False)
        self.act = nn.Tanh() if act_name == 'tanh' else nn.ReLU()
            
        self.dfc1 = nn.Linear(latent_dim, h2, bias=False)
        self.dfc2 = nn.Linear(h2, h1, bias=False)
        self.dfc3 = nn.Linear(h1, input_dim, bias=False)

    def encode(self, x):
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        z = self.fc3(h2)
        return z

    def forward(self, x):
        if x.dim() == 3:
            x = x.view(x.shape[0], -1)
        z = self.encode(x)
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        recon = self.dfc3(dh2)
        return recon, z

# ------------------------------------------------------------------------------
# 2. Funciones de Acondicionamiento
# ------------------------------------------------------------------------------
def extraer_sesion_agnostica(toma_str):
    s = str(toma_str).strip()
    parts = s.split('_')
    if len(parts) >= 2 and parts[0].upper() in ['A', 'E', 'I', 'O', 'U']:
        return parts[1].upper()
    m = re.search(r'(Prueba\d+|Sesion\d+|Session\d+|Serie\d+|Toma\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win') or p_clean.lower().startswith('w'):
            continue
        if any(char.isdigit() for char in p_clean) and len(p_clean) <= 10:
            return p_clean.upper()
    return 'S1'

def acondicionar_reposo_impedancia(X_array, sesiones, n_canales=3, n_pts_reposo=6):
    N, n_canales, n_pts = X_array.shape
    b, a = butter(N=3, Wn=0.3, btype='low')
    X_filt = np.zeros_like(X_array)
    for i in range(N):
        for c in range(n_canales):
            X_filt[i, c, :] = filtfilt(b, a, X_array[i, c, :])

    unique_ses = np.unique(sesiones)
    X_norm = np.zeros_like(X_filt)
    pts_base = max(1, min(n_pts_reposo, n_pts // 2))

    for s in unique_ses:
        mask = (sesiones == s)
        for c in range(n_canales):
            base_mean = np.mean(X_filt[mask, c, :pts_base])
            base_p95 = np.percentile(X_filt[mask, c, :], 95)
            scale = max(base_p95 - base_mean, 1e-4)
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / scale

    return X_norm

def cargar_o_extraer_candela_4060(out_dir):
    cache_path = os.path.join(out_dir, "dataset_candela_20260901_ventana4060.npz")
    if os.path.exists(cache_path):
        print(f"[Caché] Cargando dataset de Candela (2026-09-01) previamente extraído:")
        print(f"        {cache_path}")
        d = np.load(cache_path, allow_pickle=True)
        return d['X_clean'], d['Y_clean']

    base_cande = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-09-01")
    tomas_todas = sorted([t for t in os.listdir(base_cande) if os.path.isdir(os.path.join(base_cande, t))])
    tomas_cande = [t for t in tomas_todas if t.split('_')[0].upper() in ['A', 'E', 'I', 'O', 'U']]

    print(f"[Extracción Candela] Procesando {len(tomas_cande)} tomas con ventana 40/60...")
    X_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_cande,
        mediciones=tomas_cande,
        alpha_ruido=1.0,
        gate_ratio_ruido=0.0,
        smooth_ms=90,
        notch_q=2.0,
        target_len=20,
        modo_alineacion="Pico Volumen Micrófono",
        pre_pct=0.40,
        post_pct=0.60,
        canales_features=["canal_0", "canal_1", "canal_2"],
        aplicar_correccion_intersesion=True,
        tipo_envolvente="rms",
        lowpass_cutoff_hz=500.0,
        tipo_filtro_ruido="notch",
        highpass_cutoff_hz=20.0
    )

    X_arr = np.array(X_raw, dtype=np.float32).reshape(-1, 3, 20)
    Y_arr = np.array(Y_raw)
    sesiones_arr = np.array([extraer_sesion_agnostica(t) for t in tomas_wins])

    X_arr = acondicionar_reposo_impedancia(X_arr, sesiones_arr, n_canales=3, n_pts_reposo=6)
    X_flat = X_arr.reshape(len(X_arr), -1)

    iso = IsolationForest(contamination=0.10, random_state=42)
    mask_inliers = (iso.fit_predict(X_flat) == 1)
    X_clean = X_arr[mask_inliers]
    Y_clean = Y_arr[mask_inliers]

    np.savez_compressed(cache_path, X_clean=X_clean, Y_clean=Y_clean)
    print(f"[Caché] Guardado dataset de Candela: {len(Y_clean)} ventanas válidas post-purga.")
    return X_clean, Y_clean

# ------------------------------------------------------------------------------
# 3. Flujo Principal
# ------------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("ANALISIS COMPARATIVO DE ATRACTORES: LUCAS VS CANDELA (MODELO SIN SO2)")
    print("=" * 75)

    base_out = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana4060")
    os.makedirs(base_out, exist_ok=True)

    # 1. Cargar modelo campeon de Lucas
    p_ckpt = os.path.join(base_out, "mlp_3d_sin_so2/campeon_mlp_3d_sin_so2_zona_dulce_completo.pt")
    if not os.path.exists(p_ckpt):
        p_ckpt = os.path.join(base_out, "mlp_3d_sin_so2/campeon_mlp_3d_sin_so2.pt")
    print(f"[Modelo] Cargando checkpoint: {p_ckpt}")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    model = ParametricMLPOrthogonalAE(input_dim=60, hidden_dims=(48, 24), latent_dim=3, act_name='tanh')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # 2. Cargar datos de Lucas
    p_lucas_npz = os.path.join(base_out, "dataset_lucas_ventana4060.npz")
    print(f"[Lucas] Cargando dataset: {p_lucas_npz}")
    d_lucas = np.load(p_lucas_npz, allow_pickle=True)
    X_L = d_lucas['X_clean'].reshape(len(d_lucas['X_clean']), -1)
    Y_L = d_lucas['Y_clean']

    # 3. Cargar / Extraer datos de Candela
    X_C_3d, Y_C = cargar_o_extraer_candela_4060(base_out)
    X_C = X_C_3d.reshape(len(X_C_3d), -1)

    # 4. Proyeccion en espacio latente
    with torch.no_grad():
        Z_L = model.encode(torch.tensor(X_L, dtype=torch.float32)).numpy()
        Z_C = model.encode(torch.tensor(X_C, dtype=torch.float32)).numpy()

    print(f"\n[Proyeccion Latente OK]")
    print(f"  Lucas:   {Z_L.shape[0]} muestras en R^3")
    print(f"  Candela: {Z_C.shape[0]} muestras en R^3 (proyectadas por el modelo de Lucas)")

    # 5. Calculo de Atractores (Centroides por Vocal)
    atractores_L = {}
    atractores_C = {}
    matriz_atractores_L = []
    matriz_atractores_C = []

    print("\n" + "-" * 75)
    print("ATRACTORES VOCALICOS - COORDENADAS MEDIAS Y NORMAS RADIALES")
    print("-" * 75)
    print(f"{'Vocal':<6} | {'Lucas Atractor (Z1, Z2, Z3)':<30} | {'||z_L||':<7} | {'Cande Atractor (Z1, Z2, Z3)':<30} | {'||z_C||':<7} | {'Angulo (deg)'}")
    print("-" * 75)

    angulos_por_vocal = {}
    for v in VOCALES:
        mask_l = (Y_L == v)
        mask_c = (Y_C == v)
        c_l = np.mean(Z_L[mask_l], axis=0)
        c_c = np.mean(Z_C[mask_c], axis=0)

        atractores_L[v] = c_l
        atractores_C[v] = c_c
        matriz_atractores_L.append(c_l)
        matriz_atractores_C.append(c_c)

        norm_l = np.linalg.norm(c_l)
        norm_c = np.linalg.norm(c_c)

        cos_sim = np.dot(c_l, c_c) / max(1e-7, (norm_l * norm_c))
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        ang_deg = np.degrees(np.arccos(cos_sim))
        angulos_por_vocal[v] = ang_deg

        str_l = f"({c_l[0]:+.2f}, {c_l[1]:+.2f}, {c_l[2]:+.2f})"
        str_c = f"({c_c[0]:+.2f}, {c_c[1]:+.2f}, {c_c[2]:+.2f})"
        print(f" /{v.lower()}/   | {str_l:<30} | {norm_l:<7.2f} | {str_c:<30} | {norm_c:<7.2f} | {ang_deg:>6.1f}°")

    matriz_atractores_L = np.array(matriz_atractores_L)  # (5, 3)
    matriz_atractores_C = np.array(matriz_atractores_C)  # (5, 3)

    # 6. Analisis de Jordan y CCA
    print("\n" + "-" * 75)
    print("ANALISIS CANONICO DE JORDAN Y CORRELACIONES CANONICAS (CCA)")
    print("-" * 75)

    # Angulos principales entre los subespacios de atractores (Jordan)
    subspace_angles_rad = la.subspace_angles(matriz_atractores_L.T, matriz_atractores_C.T)
    subspace_angles_deg = np.degrees(subspace_angles_rad)
    cos_jordan = np.cos(subspace_angles_rad)

    print("Angulos Principales de Jordan entre subespacios de atractores:")
    for idx_ang, (deg, c_j) in enumerate(zip(subspace_angles_deg, cos_jordan), 1):
        print(f"  Theta_{idx_ang} (Jordan): {deg:.2f}°  -->  cos(Theta_{idx_ang}) = {c_j:.4f}")

    # Ajuste de CCA entre los 5 pares de atractores
    cca = CCA(n_components=3)
    cca.fit(matriz_atractores_C, matriz_atractores_L)
    Z_C_cca_cents, Z_L_cca_cents = cca.transform(matriz_atractores_C, matriz_atractores_L)

    corrs_cca = [np.corrcoef(Z_C_cca_cents[:, i], Z_L_cca_cents[:, i])[0, 1] for i in range(3)]
    print("\nCorrelaciones Canonicas (CCA) sobre los atractores:")
    for i, r in enumerate(corrs_cca, 1):
        ang_cca = np.degrees(np.arccos(np.clip(r, -1.0, 1.0)))
        print(f"  Dimension Canonica {i}: rho_{i} = {r:.4f}  (Angulo efectivo: {ang_cca:.2f}°)")

    # --------------------------------------------------------------------------
    # 7. Graficacion 1: Espacios Latentes 3D con Atractores
    # --------------------------------------------------------------------------
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 7), dpi=150)

    # Panel Lucas 3D
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_facecolor('#0B0C10')
    for v in VOCALES:
        mask = (Y_L == v)
        ax1.scatter(Z_L[mask, 0], Z_L[mask, 1], Z_L[mask, 2],
                    c=COLORES[v], alpha=0.35, s=25, edgecolors='none')
        # Atractor
        c = atractores_L[v]
        ax1.scatter([c[0]], [c[1]], [c[2]], c=COLORES[v], s=180, edgecolors='#FFFFFF', linewidths=1.5, marker='o')
        # Vector desde el origen
        ax1.plot([0, c[0]], [0, c[1]], [0, c[2]], color=COLORES[v], linewidth=2.5, alpha=0.9,
                 label=f"/{v.lower()}/ (||z||={np.linalg.norm(c):.2f})")

    ax1.scatter([0], [0], [0], c='#FFFFFF', s=60, marker='x', label='Reposo (0,0,0)')
    ax1.set_title("Lucas: Espacio Latente 3D y Atractores Nativos", color='#66FCF1', fontsize=12, fontweight='bold', pad=12)
    ax1.set_xlabel("Z1", color='#C5C6C7')
    ax1.set_ylabel("Z2", color='#C5C6C7')
    ax1.set_zlabel("Z3", color='#C5C6C7')
    ax1.legend(loc='upper right', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Panel Candela 3D
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_facecolor('#0B0C10')
    for v in VOCALES:
        mask = (Y_C == v)
        ax2.scatter(Z_C[mask, 0], Z_C[mask, 1], Z_C[mask, 2],
                    c=COLORES[v], alpha=0.35, s=25, edgecolors='none')
        # Atractor
        c = atractores_C[v]
        ax2.scatter([c[0]], [c[1]], [c[2]], c=COLORES[v], s=180, edgecolors='#FFFFFF', linewidths=1.5, marker='o')
        # Vector desde el origen
        ax2.plot([0, c[0]], [0, c[1]], [0, c[2]], color=COLORES[v], linewidth=2.5, alpha=0.9,
                 label=f"/{v.lower()}/ (||z||={np.linalg.norm(c):.2f}, desv={angulos_por_vocal[v]:.0f}°)")

    ax2.scatter([0], [0], [0], c='#FFFFFF', s=60, marker='x', label='Reposo (0,0,0)')
    ax2.set_title("Candela: Proyeccion en Modelo de Lucas (Sin Alinear)", color='#FF6B6B', fontsize=12, fontweight='bold', pad=12)
    ax2.set_xlabel("Z1", color='#C5C6C7')
    ax2.set_ylabel("Z2", color='#C5C6C7')
    ax2.set_zlabel("Z3", color='#C5C6C7')
    ax2.legend(loc='upper right', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    p_fig_3d = os.path.join(base_out, "comparativa_atractores_lucas_vs_candela_3d.png")
    fig.tight_layout()
    fig.savefig(p_fig_3d, dpi=150)
    plt.close(fig)
    print(f"\n[Figura 1 Guardada] {p_fig_3d}")

    # --------------------------------------------------------------------------
    # 8. Graficacion 2: Proyecciones Planas 2D (Z1-Z2 y Z1-Z3)
    # --------------------------------------------------------------------------
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=150)
    fig2.patch.set_facecolor('#0B0C10')

    # Fila 1: Lucas
    # Subplot (0, 0): Lucas Z1 vs Z2
    ax_l_12 = axes[0, 0]
    ax_l_12.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_L == v)
        ax_l_12.scatter(Z_L[m, 0], Z_L[m, 1], c=COLORES[v], alpha=0.3, s=20, edgecolors='none')
        c = atractores_L[v]
        ax_l_12.scatter(c[0], c[1], c=COLORES[v], s=140, edgecolors='#FFFFFF', linewidths=1.2)
        ax_l_12.plot([0, c[0]], [0, c[1]], color=COLORES[v], linewidth=2.0, label=f"/{v.lower()}/")
    ax_l_12.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_l_12.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_l_12.set_title("Lucas: Plano Z1 - Z2", color='#66FCF1', fontsize=11, fontweight='bold')
    ax_l_12.set_xlabel("Z1", color='#C5C6C7')
    ax_l_12.set_ylabel("Z2", color='#C5C6C7')
    ax_l_12.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax_l_12.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Subplot (0, 1): Lucas Z1 vs Z3
    ax_l_13 = axes[0, 1]
    ax_l_13.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_L == v)
        ax_l_13.scatter(Z_L[m, 0], Z_L[m, 2], c=COLORES[v], alpha=0.3, s=20, edgecolors='none')
        c = atractores_L[v]
        ax_l_13.scatter(c[0], c[2], c=COLORES[v], s=140, edgecolors='#FFFFFF', linewidths=1.2)
        ax_l_13.plot([0, c[0]], [0, c[2]], color=COLORES[v], linewidth=2.0, label=f"/{v.lower()}/")
    ax_l_13.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_l_13.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_l_13.set_title("Lucas: Plano Z1 - Z3", color='#66FCF1', fontsize=11, fontweight='bold')
    ax_l_13.set_xlabel("Z1", color='#C5C6C7')
    ax_l_13.set_ylabel("Z3", color='#C5C6C7')
    ax_l_13.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax_l_13.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Fila 2: Candela
    # Subplot (1, 0): Candela Z1 vs Z2
    ax_c_12 = axes[1, 0]
    ax_c_12.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_C == v)
        ax_c_12.scatter(Z_C[m, 0], Z_C[m, 1], c=COLORES[v], alpha=0.3, s=20, edgecolors='none')
        c = atractores_C[v]
        ax_c_12.scatter(c[0], c[1], c=COLORES[v], s=140, edgecolors='#FFFFFF', linewidths=1.2)
        ax_c_12.plot([0, c[0]], [0, c[1]], color=COLORES[v], linewidth=2.0, label=f"/{v.lower()}/")
    ax_c_12.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_c_12.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_c_12.set_title("Candela: Plano Z1 - Z2 (Sin Alinear)", color='#FF6B6B', fontsize=11, fontweight='bold')
    ax_c_12.set_xlabel("Z1", color='#C5C6C7')
    ax_c_12.set_ylabel("Z2", color='#C5C6C7')
    ax_c_12.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax_c_12.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    # Subplot (1, 1): Candela Z1 vs Z3
    ax_c_13 = axes[1, 1]
    ax_c_13.set_facecolor('#0B0C10')
    for v in VOCALES:
        m = (Y_C == v)
        ax_c_13.scatter(Z_C[m, 0], Z_C[m, 2], c=COLORES[v], alpha=0.3, s=20, edgecolors='none')
        c = atractores_C[v]
        ax_c_13.scatter(c[0], c[2], c=COLORES[v], s=140, edgecolors='#FFFFFF', linewidths=1.2)
        ax_c_13.plot([0, c[0]], [0, c[2]], color=COLORES[v], linewidth=2.0, label=f"/{v.lower()}/")
    ax_c_13.axhline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_c_13.axvline(0, color='#45A29E', linestyle='--', alpha=0.3)
    ax_c_13.set_title("Candela: Plano Z1 - Z3 (Sin Alinear)", color='#FF6B6B', fontsize=11, fontweight='bold')
    ax_c_13.set_xlabel("Z1", color='#C5C6C7')
    ax_c_13.set_ylabel("Z3", color='#C5C6C7')
    ax_c_13.grid(True, color='#1F2833', linestyle=':', alpha=0.6)
    ax_c_13.legend(loc='lower left', frameon=True, facecolor='#1F2833', edgecolor='none', fontsize=8)

    p_fig_2d = os.path.join(base_out, "comparativa_atractores_lucas_vs_candela_proyecciones2d.png")
    fig2.tight_layout()
    fig2.savefig(p_fig_2d, dpi=150)
    plt.close(fig2)
    print(f"[Figura 2 Guardada] {p_fig_2d}")

    print("\n[OK] Analisis completado con exito.")

if __name__ == '__main__':
    main()
