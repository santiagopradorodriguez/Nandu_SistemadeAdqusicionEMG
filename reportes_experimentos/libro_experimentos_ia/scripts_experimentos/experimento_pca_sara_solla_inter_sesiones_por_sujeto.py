#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alineación Inter-Sesión con PCA Lineal y Método de Sara Solla (CCA y Jordan) por Sujeto:
- Lucas: Sesiones T1, T2, T3, T4, T5, T6, T7 (2026-07-10)
- Petra: Sesiones Med 1 vs Med 2 (2026-08-28)
- Candela: Sesiones Prueba 1, Prueba 2, Prueba 3, Prueba 4 (2026-09-01)
Sin algoritmos de clustering (sin GMM ni K-Means).
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from scipy import signal
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

def filtrar_y_normalizar(X_raw, sesiones, n_ch=3):
    N, D = X_raw.shape
    n_pts = D // n_ch
    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')
    X_reshaped = X_raw.reshape(N, n_ch, n_pts)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N):
        for c in range(n_ch):
            X_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for s in np.unique(sesiones):
        mask = (sesiones == s)
        for c in range(n_ch):
            base_mean = np.mean(X_filt[mask, c, :10])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    return X_norm.reshape(N, -1)

def main():
    print("=" * 80)
    print("ALINEACION INTER-SESION DE VARIEDADES MOTORAS PCA (METODO DE SARA SOLLA)")
    print("=" * 80)

    dir_out = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    os.makedirs(dir_out, exist_ok=True)

    # ==========================================================================
    # 1. LUCAS: INTER-SESION T1, T2, T3, T4, T5, T6, T7
    # ==========================================================================
    print("\n[1/3] Procesando Lucas: Inter-Sesion T1 a T7...")
    X_l_t, y_l, _, _, _, _ = cargar_datos_lucas()
    X_L = X_l_t.numpy()
    Y_L = np.array(y_l)

    csv_lucas = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv"
    )
    df_l = pd.read_csv(csv_lucas)
    tomas_l = df_l['Toma'].apply(lambda x: re.search(r'T\d+', str(x)).group(0) if re.search(r'T\d+', str(x)) else 'Unknown').values

    # Sesión de referencia: T1
    mask_T1 = (tomas_l == 'T1')
    pca_T1 = PCA(n_components=2, random_state=42).fit(X_L[mask_T1])
    Z_T1_nat = pca_T1.transform(X_L[mask_T1])
    cents_T1_nat = {v: np.mean(Z_T1_nat[Y_L[mask_T1] == v], axis=0) for v in VOCALES}

    # Anclaje canónico de T1 (Vértice /I/ al origen, /A/ a 90° en +Y)
    v0_T1 = cents_T1_nat['I'].copy()
    v_dir_T1 = cents_T1_nat['A'] - v0_T1
    ang_A_T1 = np.arctan2(v_dir_T1[1], v_dir_T1[0])
    theta_rot = np.pi/2 - ang_A_T1
    R_T1 = np.array([
        [np.cos(theta_rot), -np.sin(theta_rot)],
        [np.sin(theta_rot),  np.cos(theta_rot)]
    ])

    Z_T1_can = (Z_T1_nat - v0_T1) @ R_T1.T
    cents_T1_can = {v: np.mean(Z_T1_can[Y_L[mask_T1] == v], axis=0) for v in VOCALES}
    mat_T1_can = np.array([cents_T1_can[v] for v in VOCALES])

    # Alinear cada sesión de Lucas hacia T1 Canónico con Sara Solla (CCA)
    Z_lucas_aligned = {'T1': Z_T1_can}
    cents_lucas_aligned = {'T1': cents_T1_can}
    metricas_lucas = {}

    for t in ['T2', 'T3', 'T4', 'T5', 'T6', 'T7']:
        mask_t = (tomas_l == t)
        pca_t = PCA(n_components=2, random_state=42).fit(X_L[mask_t])
        Z_t_nat = pca_t.transform(X_L[mask_t])
        cents_t_nat = {v: np.mean(Z_t_nat[Y_L[mask_t] == v], axis=0) for v in VOCALES}
        mat_t_nat = np.array([cents_t_nat[v] for v in VOCALES])

        # CCA Sara Solla
        cca = CCA(n_components=2)
        cca.fit(mat_t_nat, mat_T1_can)
        c_t_cca, c_ref_cca = cca.transform(mat_t_nat, mat_T1_can)
        r1 = np.corrcoef(c_t_cca[:, 0], c_ref_cca[:, 0])[0, 1]
        r2 = np.corrcoef(c_t_cca[:, 1], c_ref_cca[:, 1])[0, 1]
        ang1 = np.degrees(np.arccos(np.clip(r1, -1.0, 1.0)))
        ang2 = np.degrees(np.arccos(np.clip(r2, -1.0, 1.0)))
        metricas_lucas[t] = (r1, ang1, r2, ang2)

        # Mapeo afín de Sara Solla
        mu_t = np.mean(mat_t_nat, axis=0)
        mu_ref = np.mean(mat_T1_can, axis=0)
        W, _, _, _ = np.linalg.lstsq(mat_t_nat - mu_t, mat_T1_can - mu_ref, rcond=None)
        Z_t_aligned = (Z_t_nat - mu_t) @ W + mu_ref
        Z_lucas_aligned[t] = Z_t_aligned
        cents_lucas_aligned[t] = {v: np.mean(Z_t_aligned[Y_L[mask_t] == v], axis=0) for v in VOCALES}

        print(f"  Lucas {t} vs T1: rho_1 = {r1:.4f} ({ang1:.2f}°), rho_2 = {r2:.4f} ({ang2:.2f}°)")

    # Graficar Lucas: 4 paneles (T1 Referencia, T2 Alineada, T3 Alineada, Superposición T1-T7)
    fig_l, axes_l = plt.subplots(1, 4, figsize=(24, 6.0), facecolor='white')
    all_X_l = np.concatenate([Z_lucas_aligned[t][:, 0] for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']])
    all_Y_l = np.concatenate([Z_lucas_aligned[t][:, 1] for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']])
    x_min_l = np.percentile(all_X_l, 0.5) - 0.4
    x_max_l = np.percentile(all_X_l, 99.5) + 0.5
    y_min_l = np.percentile(all_Y_l, 0.5) - 0.4
    y_max_l = np.percentile(all_Y_l, 99.5) + 0.5

    # Panel 1: T1
    ax1 = axes_l[0]
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        m = (Y_L[mask_T1] == v)
        ax1.scatter(Z_T1_can[m, 0], Z_T1_can[m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_T1_can[v]
        ax1.plot([cents_T1_can['I'][0], c[0]], [cents_T1_can['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.2, alpha=0.85)
        ax1.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=140, marker='D', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax1.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax1.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax1.set_title("Lucas: Sesion T1 Canonica de Referencia - N=75", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(x_min_l, x_max_l)
    ax1.set_ylim(y_min_l, y_max_l)

    # Panel 2: T2
    ax2 = axes_l[1]
    ax2.set_facecolor('white')
    ax2.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    m_T2 = (tomas_l == 'T2')
    r1, a1, _, _ = metricas_lucas['T2']
    for v in VOCALES:
        m = (Y_L[m_T2] == v)
        ax2.scatter(Z_lucas_aligned['T2'][m, 0], Z_lucas_aligned['T2'][m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_lucas_aligned['T2'][v]
        ax2.plot([cents_lucas_aligned['T2']['I'][0], c[0]], [cents_lucas_aligned['T2']['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=2.2, alpha=0.85)
        ax2.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=130, marker='s', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax2.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax2.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax2.set_title(f"Lucas: Sesion T2 Alineada Sara Solla - rho={r1:.3f} - N=112", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(x_min_l, x_max_l)
    ax2.set_ylim(y_min_l, y_max_l)

    # Panel 3: T3
    ax3 = axes_l[2]
    ax3.set_facecolor('white')
    ax3.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    m_T3 = (tomas_l == 'T3')
    r1, a1, _, _ = metricas_lucas['T3']
    for v in VOCALES:
        m = (Y_L[m_T3] == v)
        ax3.scatter(Z_lucas_aligned['T3'][m, 0], Z_lucas_aligned['T3'][m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_lucas_aligned['T3'][v]
        ax3.plot([cents_lucas_aligned['T3']['I'][0], c[0]], [cents_lucas_aligned['T3']['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle=':', linewidth=2.2, alpha=0.85)
        ax3.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=160, marker='^', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax3.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax3.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax3.set_title(f"Lucas: Sesion T3 Alineada Sara Solla - rho={r1:.3f} - N=105", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(x_min_l, x_max_l)
    ax3.set_ylim(y_min_l, y_max_l)

    # Panel 4: Superposición Inter-Sesión (T1 a T7)
    ax4 = axes_l[3]
    ax4.set_facecolor('white')
    ax4.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    markers_ses = {'T1': 'D', 'T2': 's', 'T3': '^', 'T4': 'v', 'T5': 'o', 'T6': '<', 'T7': '>'}
    
    for t in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']:
        for v in VOCALES:
            c = cents_lucas_aligned[t][v]
            ax4.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=90 if t in ['T1','T2','T3'] else 60,
                        marker=markers_ses[t], edgecolors='#263238', linewidths=1.0, zorder=6,
                        alpha=0.9 if t in ['T1','T2','T3'] else 0.7)
            # Rayo para T1, T2, T3
            if t in ['T1', 'T2', 'T3']:
                ls = '-' if t == 'T1' else ('--' if t == 'T2' else ':')
                ax4.plot([cents_lucas_aligned[t]['I'][0], c[0]], [cents_lucas_aligned[t]['I'][1], c[1]],
                         color=COLORES_VOCALES[v], linestyle=ls, linewidth=1.5, alpha=0.6)

    for v in VOCALES:
        cL = cents_T1_can[v]
        txt = ax4.text(cL[0] + 0.15, cL[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    # Leyenda
    handles_leg = [
        ax4.scatter([], [], marker='D', color='#455A64', s=60, label='T1 (N=75) Ref'),
        ax4.scatter([], [], marker='s', color='#455A64', s=60, label='T2 (N=112) rho=0.986'),
        ax4.scatter([], [], marker='^', color='#455A64', s=60, label='T3 (N=105) rho=0.996'),
        ax4.scatter([], [], marker='v', color='#455A64', s=45, label='T4 (N=53) rho=1.000'),
        ax4.scatter([], [], marker='o', color='#455A64', s=45, label='T5 (N=55) rho=0.993'),
        ax4.scatter([], [], marker='<', color='#455A64', s=45, label='T6 (N=49) rho=1.000'),
        ax4.scatter([], [], marker='>', color='#455A64', s=45, label='T7 (N=53) rho=1.000')
    ]
    ax4.legend(handles=handles_leg, loc='upper left', framealpha=0.9, fontsize=8.5)
    ax4.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax4.set_title("Superposicion Inter-Sesion Lucas: 7 Sesiones Alineadas", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax4.set_xlim(x_min_l, x_max_l)
    ax4.set_ylim(y_min_l, y_max_l)

    plt.tight_layout()
    p_fig_lucas = os.path.join(dir_out, "lucas_pca_inter_sesion_sara_solla.png")
    plt.savefig(p_fig_lucas, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Figura Lucas guardada en: {p_fig_lucas}")

    # ==========================================================================
    # 2. PETRA: INTER-SESION MED 1 VS MED 2 (2026-08-28)
    # ==========================================================================
    print("\n[2/3] Procesando Petra: Inter-Sesion Med 1 vs Med 2...")
    csv_petra = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/petra/petra_max_config_ultimas_dos_medicines_2d/caracteristicas_exportadas.csv"
    )
    df_p = pd.read_csv(csv_petra)
    ses_p = df_p['Toma'].apply(lambda x: 'med1' if 'med1' in str(x) else ('med2' if 'med2' in str(x) else 'other')).values
    feat_cols_p = [c for c in df_p.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_raw_p = df_p[feat_cols_p].values
    Y_p = df_p['Vocal'].values
    X_clean_p = filtrar_y_normalizar(X_raw_p, ses_p, n_ch=3)

    # Med 1 (Referencia)
    mask_m1 = (ses_p == 'med1')
    pca_m1 = PCA(n_components=2, random_state=42).fit(X_clean_p[mask_m1])
    Z_m1_nat = pca_m1.transform(X_clean_p[mask_m1])
    cents_m1_nat = {v: np.mean(Z_m1_nat[Y_p[mask_m1] == v], axis=0) for v in VOCALES}

    # Anclaje canónico de Med 1 (Vértice /I/ al origen, /A/ a 90° en +Y)
    v0_m1 = cents_m1_nat['I'].copy()
    v_dir_m1 = cents_m1_nat['A'] - v0_m1
    ang_A_m1 = np.arctan2(v_dir_m1[1], v_dir_m1[0])
    theta_rot_p = np.pi/2 - ang_A_m1
    R_m1 = np.array([
        [np.cos(theta_rot_p), -np.sin(theta_rot_p)],
        [np.sin(theta_rot_p),  np.cos(theta_rot_p)]
    ])

    Z_m1_can = (Z_m1_nat - v0_m1) @ R_m1.T
    cents_m1_can = {v: np.mean(Z_m1_can[Y_p[mask_m1] == v], axis=0) for v in VOCALES}
    mat_m1_can = np.array([cents_m1_can[v] for v in VOCALES])

    # Med 2 Alineada con Sara Solla (CCA)
    mask_m2 = (ses_p == 'med2')
    pca_m2 = PCA(n_components=2, random_state=42).fit(X_clean_p[mask_m2])
    Z_m2_nat = pca_m2.transform(X_clean_p[mask_m2])
    cents_m2_nat = {v: np.mean(Z_m2_nat[Y_p[mask_m2] == v], axis=0) for v in VOCALES}
    mat_m2_nat = np.array([cents_m2_nat[v] for v in VOCALES])

    cca_p = CCA(n_components=2)
    cca_p.fit(mat_m2_nat, mat_m1_can)
    c_m2_cca, c_m1_cca = cca_p.transform(mat_m2_nat, mat_m1_can)
    r1_p = np.corrcoef(c_m2_cca[:, 0], c_m1_cca[:, 0])[0, 1]
    r2_p = np.corrcoef(c_m2_cca[:, 1], c_m1_cca[:, 1])[0, 1]
    ang1_p = np.degrees(np.arccos(np.clip(r1_p, -1.0, 1.0)))
    ang2_p = np.degrees(np.arccos(np.clip(r2_p, -1.0, 1.0)))

    mu_m2 = np.mean(mat_m2_nat, axis=0)
    mu_m1 = np.mean(mat_m1_can, axis=0)
    W_p, _, _, _ = np.linalg.lstsq(mat_m2_nat - mu_m2, mat_m1_can - mu_m1, rcond=None)
    Z_m2_aligned = (Z_m2_nat - mu_m2) @ W_p + mu_m1
    cents_m2_aligned = {v: np.mean(Z_m2_aligned[Y_p[mask_m2] == v], axis=0) for v in VOCALES}

    print(f"  Petra Med 2 vs Med 1: rho_1 = {r1_p:.4f} ({ang1_p:.2f}°), rho_2 = {r2_p:.4f} ({ang2_p:.2f}°)")

    # Graficar Petra: 3 paneles
    fig_p, axes_p = plt.subplots(1, 3, figsize=(18, 6.0), facecolor='white')
    all_X_p = np.concatenate([Z_m1_can[:, 0], Z_m2_aligned[:, 0]])
    all_Y_p = np.concatenate([Z_m1_can[:, 1], Z_m2_aligned[:, 1]])
    x_min_p = np.percentile(all_X_p, 0.5) - 0.4
    x_max_p = np.percentile(all_X_p, 99.5) + 0.5
    y_min_p = np.percentile(all_Y_p, 0.5) - 0.4
    y_max_p = np.percentile(all_Y_p, 99.5) + 0.5

    # Panel 1: Med 1
    ax1 = axes_p[0]
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        m = (Y_p[mask_m1] == v)
        ax1.scatter(Z_m1_can[m, 0], Z_m1_can[m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_m1_can[v]
        ax1.plot([cents_m1_can['I'][0], c[0]], [cents_m1_can['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.2, alpha=0.85)
        ax1.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=140, marker='D', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax1.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax1.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax1.set_title("Petra: Med 1 Canonica de Referencia - N=86", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(x_min_p, x_max_p)
    ax1.set_ylim(y_min_p, y_max_p)

    # Panel 2: Med 2
    ax2 = axes_p[1]
    ax2.set_facecolor('white')
    ax2.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        m = (Y_p[mask_m2] == v)
        ax2.scatter(Z_m2_aligned[m, 0], Z_m2_aligned[m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_m2_aligned[v]
        ax2.plot([cents_m2_aligned['I'][0], c[0]], [cents_m2_aligned['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=2.2, alpha=0.85)
        ax2.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=130, marker='s', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax2.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax2.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax2.set_title(f"Petra: Med 2 Alineada Sara Solla - rho={r1_p:.3f} - N=83", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(x_min_p, x_max_p)
    ax2.set_ylim(y_min_p, y_max_p)

    # Panel 3: Superposición
    ax3 = axes_p[2]
    ax3.set_facecolor('white')
    ax3.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        c1 = cents_m1_can[v]
        c2 = cents_m2_aligned[v]
        ax3.plot([cents_m1_can['I'][0], c1[0]], [cents_m1_can['I'][1], c1[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.0, alpha=0.85)
        ax3.plot([cents_m2_aligned['I'][0], c2[0]], [cents_m2_aligned['I'][1], c2[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=2.0, alpha=0.85)
        ax3.scatter(c1[0], c1[1], c=COLORES_VOCALES[v], s=130, marker='D', edgecolors='#263238', linewidths=1.2, zorder=6)
        ax3.scatter(c2[0], c2[1], c=COLORES_VOCALES[v], s=120, marker='s', edgecolors='#263238', linewidths=1.2, zorder=6)
        txt = ax3.text(c1[0] + 0.15, c1[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    h1 = ax3.scatter([], [], marker='D', color='#455A64', s=70, label='Med 1 (N=86) Ref')
    h2 = ax3.scatter([], [], marker='s', color='#455A64', s=65, label='Med 2 (N=83) rho=1.000')
    ax3.legend(handles=[h1, h2], loc='upper left', framealpha=0.9, fontsize=9.5)
    ax3.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax3.set_title("Superposicion Inter-Sesion Petra: Med 1 vs Med 2", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(x_min_p, x_max_p)
    ax3.set_ylim(y_min_p, y_max_p)

    plt.tight_layout()
    p_fig_petra = os.path.join(dir_out, "petra_pca_inter_sesion_sara_solla.png")
    plt.savefig(p_fig_petra, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Figura Petra guardada en: {p_fig_petra}")

    # ==========================================================================
    # 3. CANDELA: INTER-SESION PRUEBA 1 A PRUEBA 4 (2026-09-01)
    # ==========================================================================
    print("\n[3/3] Procesando Candela: Inter-Sesion Prueba 1 a 4...")
    csv_candela = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/candela/cande_09/caracteristicas_exportadas.csv"
    )
    df_c = pd.read_csv(csv_candela)
    ses_c = df_c['Toma'].apply(lambda x: str(x).split('_')[1] if len(str(x).split('_'))>1 else str(x)).values
    feat_cols_c = [c for c in df_c.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_raw_c = df_c[feat_cols_c].values
    Y_c = df_c['Vocal'].values
    X_clean_c = filtrar_y_normalizar(X_raw_c, ses_c, n_ch=3)

    # Prueba 1 (Referencia)
    mask_p1 = (ses_c == 'Prueba1')
    pca_p1 = PCA(n_components=2, random_state=42).fit(X_clean_c[mask_p1])
    Z_p1_nat = pca_p1.transform(X_clean_c[mask_p1])
    cents_p1_nat = {v: np.mean(Z_p1_nat[Y_c[mask_p1] == v], axis=0) for v in VOCALES}

    # Anclaje canónico de Prueba 1
    v0_p1 = cents_p1_nat['I'].copy()
    v_dir_p1 = cents_p1_nat['A'] - v0_p1
    ang_A_p1 = np.arctan2(v_dir_p1[1], v_dir_p1[0])
    theta_rot_c = np.pi/2 - ang_A_p1
    R_p1 = np.array([
        [np.cos(theta_rot_c), -np.sin(theta_rot_c)],
        [np.sin(theta_rot_c),  np.cos(theta_rot_c)]
    ])

    Z_p1_can = (Z_p1_nat - v0_p1) @ R_p1.T
    cents_p1_can = {v: np.mean(Z_p1_can[Y_c[mask_p1] == v], axis=0) for v in VOCALES}
    mat_p1_can = np.array([cents_p1_can[v] for v in VOCALES])

    # Alinear Prueba 2, 3, 4 con Sara Solla
    Z_candela_aligned = {'Prueba1': Z_p1_can}
    cents_candela_aligned = {'Prueba1': cents_p1_can}
    metricas_candela = {}

    for s in ['Prueba2', 'Prueba3', 'Prueba4']:
        mask_s = (ses_c == s)
        pca_s = PCA(n_components=2, random_state=42).fit(X_clean_c[mask_s])
        Z_s_nat = pca_s.transform(X_clean_c[mask_s])
        cents_s_nat = {v: np.mean(Z_s_nat[Y_c[mask_s] == v], axis=0) for v in VOCALES}
        mat_s_nat = np.array([cents_s_nat[v] for v in VOCALES])

        cca = CCA(n_components=2)
        cca.fit(mat_s_nat, mat_p1_can)
        c_s_cca, c_ref_cca = cca.transform(mat_s_nat, mat_p1_can)
        r1 = np.corrcoef(c_s_cca[:, 0], c_ref_cca[:, 0])[0, 1]
        r2 = np.corrcoef(c_s_cca[:, 1], c_ref_cca[:, 1])[0, 1]
        ang1 = np.degrees(np.arccos(np.clip(r1, -1.0, 1.0)))
        ang2 = np.degrees(np.arccos(np.clip(r2, -1.0, 1.0)))
        metricas_candela[s] = (r1, ang1, r2, ang2)

        mu_s = np.mean(mat_s_nat, axis=0)
        mu_ref = np.mean(mat_p1_can, axis=0)
        W, _, _, _ = np.linalg.lstsq(mat_s_nat - mu_s, mat_p1_can - mu_ref, rcond=None)
        Z_s_aligned = (Z_s_nat - mu_s) @ W + mu_ref
        Z_candela_aligned[s] = Z_s_aligned
        cents_candela_aligned[s] = {v: np.mean(Z_s_aligned[Y_c[mask_s] == v], axis=0) for v in VOCALES}
        print(f"  Candela {s} vs Prueba1: rho_1 = {r1:.4f} ({ang1:.2f}°), rho_2 = {r2:.4f} ({ang2:.2f}°)")

    # Graficar Candela: 4 paneles
    fig_c, axes_c = plt.subplots(1, 4, figsize=(24, 6.0), facecolor='white')
    all_X_c = np.concatenate([Z_candela_aligned[s][:, 0] for s in ['Prueba1', 'Prueba2', 'Prueba3', 'Prueba4']])
    all_Y_c = np.concatenate([Z_candela_aligned[s][:, 1] for s in ['Prueba1', 'Prueba2', 'Prueba3', 'Prueba4']])
    x_min_c = np.percentile(all_X_c, 0.5) - 0.4
    x_max_c = np.percentile(all_X_c, 99.5) + 0.5
    y_min_c = np.percentile(all_Y_c, 0.5) - 0.4
    y_max_c = np.percentile(all_Y_c, 99.5) + 0.5

    # Panel 1: Prueba 1
    ax1 = axes_c[0]
    ax1.set_facecolor('white')
    ax1.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    for v in VOCALES:
        m = (Y_c[mask_p1] == v)
        ax1.scatter(Z_p1_can[m, 0], Z_p1_can[m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_p1_can[v]
        ax1.plot([cents_p1_can['I'][0], c[0]], [cents_p1_can['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='-', linewidth=2.2, alpha=0.85)
        ax1.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=140, marker='D', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax1.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax1.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax1.set_title("Candela: Prueba 1 Canonica de Referencia - N=49", fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax1.set_xlim(x_min_c, x_max_c)
    ax1.set_ylim(y_min_c, y_max_c)

    # Panel 2: Prueba 2
    ax2 = axes_c[1]
    ax2.set_facecolor('white')
    ax2.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    m_p2 = (ses_c == 'Prueba2')
    r1, _, _, _ = metricas_candela['Prueba2']
    for v in VOCALES:
        m = (Y_c[m_p2] == v)
        ax2.scatter(Z_candela_aligned['Prueba2'][m, 0], Z_candela_aligned['Prueba2'][m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_candela_aligned['Prueba2'][v]
        ax2.plot([cents_candela_aligned['Prueba2']['I'][0], c[0]], [cents_candela_aligned['Prueba2']['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle='--', linewidth=2.2, alpha=0.85)
        ax2.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=130, marker='s', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax2.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax2.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax2.set_title(f"Candela: Prueba 2 Alineada Sara Solla - rho={r1:.3f} - N=49", fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax2.set_xlim(x_min_c, x_max_c)
    ax2.set_ylim(y_min_c, y_max_c)

    # Panel 3: Prueba 3
    ax3 = axes_c[2]
    ax3.set_facecolor('white')
    ax3.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    m_p3 = (ses_c == 'Prueba3')
    r1, _, _, _ = metricas_candela['Prueba3']
    for v in VOCALES:
        m = (Y_c[m_p3] == v)
        ax3.scatter(Z_candela_aligned['Prueba3'][m, 0], Z_candela_aligned['Prueba3'][m, 1], c=COLORES_VOCALES[v], alpha=0.35, s=34, edgecolors='none')
        c = cents_candela_aligned['Prueba3'][v]
        ax3.plot([cents_candela_aligned['Prueba3']['I'][0], c[0]], [cents_candela_aligned['Prueba3']['I'][1], c[1]], color=COLORES_VOCALES[v], linestyle=':', linewidth=2.2, alpha=0.85)
        ax3.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=160, marker='^', edgecolors='#263238', linewidths=1.5, zorder=5)
        txt = ax3.text(c[0] + 0.15, c[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=6)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])
    ax3.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax3.set_title(f"Candela: Prueba 3 Alineada Sara Solla - rho={r1:.3f} - N=47", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax3.set_xlim(x_min_c, x_max_c)
    ax3.set_ylim(y_min_c, y_max_c)

    # Panel 4: Superposición Prueba 1 a 4
    ax4 = axes_c[3]
    ax4.set_facecolor('white')
    ax4.grid(True, linestyle=':', alpha=0.55, color='#B0BEC5')
    markers_c = {'Prueba1': 'D', 'Prueba2': 's', 'Prueba3': '^', 'Prueba4': 'v'}
    
    for s in ['Prueba1', 'Prueba2', 'Prueba3', 'Prueba4']:
        for v in VOCALES:
            c = cents_candela_aligned[s][v]
            ax4.scatter(c[0], c[1], c=COLORES_VOCALES[v], s=90, marker=markers_c[s],
                        edgecolors='#263238', linewidths=1.2, zorder=6)
            ls = '-' if s == 'Prueba1' else ('--' if s == 'Prueba2' else ':')
            ax4.plot([cents_candela_aligned[s]['I'][0], c[0]], [cents_candela_aligned[s]['I'][1], c[1]],
                     color=COLORES_VOCALES[v], linestyle=ls, linewidth=1.5, alpha=0.6)

    for v in VOCALES:
        cC = cents_p1_can[v]
        txt = ax4.text(cC[0] + 0.15, cC[1] + 0.15, f"/{v.lower()}/", fontsize=12, fontweight='bold',
                       color=adjust_lightness(COLORES_VOCALES[v], 0.75), zorder=7)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white')])

    handles_leg_c = [
        ax4.scatter([], [], marker='D', color='#455A64', s=60, label='Prueba 1 (N=49) Ref'),
        ax4.scatter([], [], marker='s', color='#455A64', s=60, label='Prueba 2 (N=49) rho=0.999'),
        ax4.scatter([], [], marker='^', color='#455A64', s=60, label='Prueba 3 (N=47) rho=1.000'),
        ax4.scatter([], [], marker='v', color='#455A64', s=60, label='Prueba 4 (N=46) rho=1.000')
    ]
    ax4.legend(handles=handles_leg_c, loc='upper left', framealpha=0.9, fontsize=8.5)
    ax4.scatter(0, 0, marker='+', s=120, color='black', linewidths=2.0, zorder=7)
    ax4.set_title("Superposicion Inter-Sesion Candela: 4 Pruebas Alineadas", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Coordenada Latente Z1", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Coordenada Latente Z2", fontsize=11, fontweight='bold')
    ax4.set_xlim(x_min_c, x_max_c)
    ax4.set_ylim(y_min_c, y_max_c)

    plt.tight_layout()
    p_fig_candela = os.path.join(dir_out, "candela_pca_inter_sesion_sara_solla.png")
    plt.savefig(p_fig_candela, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"  Figura Candela guardada en: {p_fig_candela}")
    print("=" * 80)

if __name__ == "__main__":
    main()
