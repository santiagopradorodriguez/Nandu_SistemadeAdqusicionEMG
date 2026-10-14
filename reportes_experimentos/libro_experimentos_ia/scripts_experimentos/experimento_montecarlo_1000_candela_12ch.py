# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Experimento: Ensamble Monte Carlo 1000 Muestras (200 por Vocal) - 12 Canales
# Resolucion: 20 Puntos Temporales (240D) - Acondicionamiento Reposo/P95
# ==============================================================================

import os
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experimento_pseudo_hd_candela_multicanal_20pts import (
    procesar_toma_tricanale_adaptativa,
    acondicionar_reposo_impedancia,
    VOCALES,
    COLOR_MAP_HEX,
    TARGET_LEN,
    CANALES_CONFIG,
    base_dir
)

# Fijar semillas
np.random.seed(42)
random.seed(42)

project_root = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG'
out_dir = os.path.join(project_root, 'EMG_desarrollo/resultados/grid_search_conv_ortogonal')
os.makedirs(out_dir, exist_ok=True)

def calcular_elipse_2d(pts, n_std=1.5):
    cov = np.cov(pts[:, 0], pts[:, 1])
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(1e-6, vals[idx])
    vecs = vecs[:, idx]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2.0 * n_std * np.sqrt(vals[0])
    h = 2.0 * n_std * np.sqrt(vals[1])
    return w, h, theta

def extraer_pool_canales(target_len=20):
    datos_musculos = {cfg['musculo']: {v: [] for v in VOCALES} for cfg in CANALES_CONFIG}
    for cfg in CANALES_CONFIG:
        m_nom = cfg['musculo']
        f_dir = os.path.join(base_dir, cfg['fecha'])
        if not os.path.exists(f_dir):
            continue
        tomas = sorted([t for t in os.listdir(f_dir) if os.path.isdir(os.path.join(f_dir, t))])
        for toma in tomas:
            if cfg['filtro'].lower() not in toma.lower():
                continue
            vocal = toma.split('_')[0].upper()
            if vocal not in VOCALES:
                continue
            toma_path = os.path.join(f_dir, toma)
            res_toma = procesar_toma_tricanale_adaptativa(toma_path)
            if res_toma is not None and len(res_toma[cfg['canal_idx']]) > 0:
                datos_musculos[m_nom][vocal].extend(res_toma[cfg['canal_idx']])
    return datos_musculos

def sintetizar_montecarlo(datos_musculos, n_trials_por_vocal=200, target_len=20):
    rng = np.random.RandomState(42)
    X_syn = []
    y_syn = []
    for vi, v in enumerate(VOCALES):
        for _ in range(n_trials_por_vocal):
            trial = []
            for cfg in CANALES_CONFIG:
                m = cfg['musculo']
                pool = datos_musculos[m][v]
                if len(pool) > 0:
                    trial.append(pool[rng.choice(len(pool))])
                else:
                    trial.append(np.zeros(target_len))
            X_syn.append(trial)
            y_syn.append(vi)
    return np.array(X_syn, dtype=np.float32), np.array(y_syn, dtype=np.int64)

def main():
    print("=" * 80)
    print("MONTE CARLO 1000 MUESTRAS (200 POR VOCAL) - CANDELA 12 CANALES (240D)")
    print("=" * 80)

    # 1. Extracción de pools y síntesis Monte Carlo N=1000
    datos_musculos = extraer_pool_canales(target_len=TARGET_LEN)
    N_TRIALS = 200 # 200 * 5 = 1000 muestras
    X_raw, y = sintetizar_montecarlo(datos_musculos, n_trials_por_vocal=N_TRIALS, target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw.shape
    D = n_ch * n_pts
    print(f"Dataset sintetizado: {N} muestras ({N // 5} por vocal) | Tensor: {X_raw.shape} | Espacio: {D}D")

    # 2. Acondicionamiento de reposo e impedancia P95
    sesiones_dummy = np.zeros(N, dtype=int)
    X_cond_3d = acondicionar_reposo_impedancia(X_raw, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 3. Clasificación en espacio ambiente 240D
    acc_amb = cross_val_score(LinearDiscriminantAnalysis(), X_cond, y, cv=cv).mean() * 100.0
    print(f"\n[Espacio Ambiente 240D] Exactitud LDA (5-Fold CV): {acc_amb:.2f}%")

    # 4. PCA 2D y 3D
    pca_full = PCA(n_components=3, random_state=42)
    Z_pca_3d = pca_full.fit_transform(X_cond)
    Z_pca_2d = Z_pca_3d[:, :2]
    evr_pca = pca_full.explained_variance_ratio_ * 100.0
    acc_pca_2d = cross_val_score(LinearDiscriminantAnalysis(), Z_pca_2d, y, cv=cv).mean() * 100.0
    acc_pca_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_pca_3d, y, cv=cv).mean() * 100.0
    print(f"[PCA] 2D LDA: {acc_pca_2d:.2f}% (EVR: {evr_pca[:2].sum():.1f}%) | 3D LDA: {acc_pca_3d:.2f}% (EVR: {evr_pca.sum():.1f}%)")

    # 5. Isomap 2D y 3D (k=35)
    K_NEIGHBORS = 35
    iso_2d = Isomap(n_neighbors=K_NEIGHBORS, n_components=2)
    Z_iso_2d_raw = iso_2d.fit_transform(X_cond)
    
    # Orientación canónica: /a/ hacia +Y, /i/ hacia +X
    c_a_raw = np.mean(Z_iso_2d_raw[y == 0], axis=0)
    c_i_raw = np.mean(Z_iso_2d_raw[y == 2], axis=0)
    flip_x = 1.0 if c_i_raw[0] >= 0 else -1.0
    flip_y = 1.0 if c_a_raw[1] >= 0 else -1.0
    Z_iso_2d = np.column_stack([flip_x * Z_iso_2d_raw[:, 0], flip_y * Z_iso_2d_raw[:, 1]])

    acc_iso_2d = cross_val_score(LinearDiscriminantAnalysis(), Z_iso_2d, y, cv=cv).mean() * 100.0

    # GMM en 2D
    gmm_2d = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(Z_iso_2d)
    preds_gmm_2d = gmm_2d.predict(Z_iso_2d)
    cost = -np.array([[np.sum((preds_gmm_2d == c) & (y == tr)) for tr in range(5)] for c in range(5)])
    ri, ci = linear_sum_assignment(cost)
    acc_gmm_2d = -cost[ri, ci].sum() / N * 100.0

    iso_3d = Isomap(n_neighbors=K_NEIGHBORS, n_components=3)
    Z_iso_3d = iso_3d.fit_transform(X_cond)
    acc_iso_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_iso_3d, y, cv=cv).mean() * 100.0
    print(f"[Isomap k={K_NEIGHBORS}] 2D LDA: {acc_iso_2d:.2f}% (GMM: {acc_gmm_2d:.2f}%) | 3D LDA: {acc_iso_3d:.2f}%")

    # 6. Guardar Gráfico 2D
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    # Panel A: PCA 2D
    ax1 = axes[0]
    for vi, v in enumerate(VOCALES):
        pts = Z_pca_2d[y == vi]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)
        ax1.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=20, alpha=0.55, edgecolors='none')
        ax1.scatter(mean_pt[0], mean_pt[1], c=color, s=150, marker='D', edgecolors='black', linewidth=1.6, zorder=6)
        w, h, ang = calcular_elipse_2d(pts, n_std=1.5)
        ell = Ellipse(xy=mean_pt, width=w, height=h, angle=ang, facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.5)
        ax1.add_patch(ell)
    ax1.set_title(f"A: PCA 2D - N=1000 Muestras Monte Carlo: LDA {acc_pca_2d:.1f}% - Varianza {evr_pca[:2].sum():.1f}%", fontsize=11.5, fontweight='bold', pad=12)
    ax1.set_xlabel(f"PC1: {evr_pca[0]:.1f}% Varianza", fontsize=10, fontweight='semibold')
    ax1.set_ylabel(f"PC2: {evr_pca[1]:.1f}% Varianza", fontsize=10, fontweight='semibold')
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Isomap 2D
    ax2 = axes[1]
    for vi, v in enumerate(VOCALES):
        pts = Z_iso_2d[y == vi]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)
        ax2.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=20, alpha=0.55, edgecolors='none')
        ax2.scatter(mean_pt[0], mean_pt[1], c=color, s=150, marker='D', edgecolors='black', linewidth=1.6, zorder=6)
        w, h, ang = calcular_elipse_2d(pts, n_std=1.5)
        ell = Ellipse(xy=mean_pt, width=w, height=h, angle=ang, facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.5)
        ax2.add_patch(ell)
    ax2.set_title(f"B: Isomap 2D - N=1000 Muestras Monte Carlo: k={K_NEIGHBORS} - LDA {acc_iso_2d:.1f}%", fontsize=11.5, fontweight='bold', pad=12)
    ax2.set_xlabel("Dimension Geodesica 1: Retraccion vs Protrusion", fontsize=10, fontweight='semibold')
    ax2.set_ylabel("Dimension Geodesica 2: Apertura Mandibular Canonica", fontsize=10, fontweight='semibold')
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle("Variedad Articulatoria con N=1000 Muestras Monte Carlo (12 Canales, 240D)", fontsize=13.5, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.96, wspace=0.16)
    png_2d = os.path.join(out_dir, "isomap_2d_montecarlo_1000.png")
    plt.savefig(png_2d, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Guardado] Imagen 2D Monte Carlo 1000 guardada en: {png_2d}")

    # 7. Guardar Gráfico 3D
    fig3d = plt.figure(figsize=(20, 9.5), dpi=300)
    elev_pca, azim_pca = 24, -45
    elev_iso, azim_iso = 22, 50

    # Subplot 1: PCA 3D
    ax_3d_1 = fig3d.add_subplot(1, 2, 1, projection='3d')
    z_floor_pca = np.min(Z_pca_3d[:, 2]) - 0.20 * np.ptp(Z_pca_3d[:, 2])
    ax_3d_1.xaxis.pane.fill = False; ax_3d_1.yaxis.pane.fill = False; ax_3d_1.zaxis.pane.fill = False
    ax_3d_1.grid(True, linestyle=':', color='#BDC3C7', alpha=0.5)
    for vi, v in enumerate(VOCALES):
        pts = Z_pca_3d[y == vi]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)
        ax_3d_1.scatter(pts[:, 0], pts[:, 1], zs=z_floor_pca, zdir='z', c=color, alpha=0.12, s=12, edgecolors='none')
        ax_3d_1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, label=f'/{v.lower()}/', s=24, alpha=0.65, edgecolors='white', linewidth=0.3)
        ax_3d_1.scatter(mean_pt[0], mean_pt[1], mean_pt[2], c=color, s=160, marker='D', edgecolors='black', linewidth=1.8, zorder=10)
        ax_3d_1.plot([mean_pt[0], mean_pt[0]], [mean_pt[1], mean_pt[1]], [z_floor_pca, mean_pt[2]], color=color, linestyle='--', linewidth=1.4)
    ax_3d_1.view_init(elev=elev_pca, azim=azim_pca)
    ax_3d_1.set_title(f"A: PCA 3D - N=1000 Monte Carlo: {evr_pca.sum():.1f}% Varianza - LDA {acc_pca_3d:.1f}%", fontsize=12, fontweight='bold', pad=14)
    ax_3d_1.set_xlabel(f"PC1: {evr_pca[0]:.1f}%", fontsize=9.5); ax_3d_1.set_ylabel(f"PC2: {evr_pca[1]:.1f}%", fontsize=9.5); ax_3d_1.set_zlabel(f"PC3: {evr_pca[2]:.1f}%", fontsize=9.5)
    ax_3d_1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), fontsize=9.5)

    # Subplot 2: Isomap 3D
    ax_3d_2 = fig3d.add_subplot(1, 2, 2, projection='3d')
    z_floor_iso = np.min(Z_iso_3d[:, 2]) - 0.20 * np.ptp(Z_iso_3d[:, 2])
    ax_3d_2.xaxis.pane.fill = False; ax_3d_2.yaxis.pane.fill = False; ax_3d_2.zaxis.pane.fill = False
    ax_3d_2.grid(True, linestyle=':', color='#BDC3C7', alpha=0.5)
    for vi, v in enumerate(VOCALES):
        pts = Z_iso_3d[y == vi]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)
        ax_3d_2.scatter(pts[:, 0], pts[:, 1], zs=z_floor_iso, zdir='z', c=color, alpha=0.12, s=12, edgecolors='none')
        ax_3d_2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=color, label=f'/{v.lower()}/', s=24, alpha=0.65, edgecolors='white', linewidth=0.3)
        ax_3d_2.scatter(mean_pt[0], mean_pt[1], mean_pt[2], c=color, s=160, marker='D', edgecolors='black', linewidth=1.8, zorder=10)
        ax_3d_2.plot([mean_pt[0], mean_pt[0]], [mean_pt[1], mean_pt[1]], [z_floor_iso, mean_pt[2]], color=color, linestyle='--', linewidth=1.4)
    ax_3d_2.view_init(elev=elev_iso, azim=azim_iso)
    ax_3d_2.set_title(f"B: Isomap 3D - N=1000 Monte Carlo: k={K_NEIGHBORS} - LDA {acc_iso_3d:.1f}%", fontsize=12, fontweight='bold', pad=14)
    ax_3d_2.set_xlabel("Dim Geodesica 1", fontsize=9.5); ax_3d_2.set_ylabel("Dim Geodesica 2", fontsize=9.5); ax_3d_2.set_zlabel("Dim Geodesica 3", fontsize=9.5)
    ax_3d_2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), fontsize=9.5)

    fig3d.suptitle("Variedad Articulatoria 3D con N=1000 Muestras Monte Carlo (12 Canales, 240D)", fontsize=14, fontweight='bold', y=0.97)
    fig3d.subplots_adjust(top=0.88, bottom=0.06, left=0.03, right=0.97, wspace=0.08)
    png_3d = os.path.join(out_dir, "pca_isomap_3d_montecarlo_1000.png")
    plt.savefig(png_3d, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Guardado] Imagen 3D Monte Carlo 1000 guardada en: {png_3d}")

    # 8. Exportar HTML interactivo Plotly
    fig_html = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[
            f"PCA 3D - N=1000: LDA {acc_pca_3d:.1f}%",
            f"Isomap 3D - N=1000: k={K_NEIGHBORS} - LDA {acc_iso_3d:.1f}%"
        ]
    )
    for vi, v in enumerate(VOCALES):
        pts_pca = Z_pca_3d[y == vi]
        pts_iso = Z_iso_3d[y == vi]
        color = COLOR_MAP_HEX[v]
        fig_html.add_trace(
            go.Scatter3d(
                x=pts_pca[:, 0], y=pts_pca[:, 1], z=pts_pca[:, 2],
                mode='markers', marker=dict(size=3.5, color=color, opacity=0.7),
                name=f'/{v.lower()}/', legendgroup=f'v_{v}'
            ), row=1, col=1
        )
        fig_html.add_trace(
            go.Scatter3d(
                x=pts_iso[:, 0], y=pts_iso[:, 1], z=pts_iso[:, 2],
                mode='markers', marker=dict(size=3.5, color=color, opacity=0.7),
                name=f'/{v.lower()}/', legendgroup=f'v_{v}', showlegend=False
            ), row=1, col=2
        )
    fig_html.update_layout(
        title=dict(text="Variedad 3D Interactivo: N=1000 Muestras Monte Carlo (12 Canales, 240D)", x=0.5, font=dict(size=18, weight="bold")),
        margin=dict(l=20, r=20, t=80, b=40)
    )
    html_3d = os.path.join(out_dir, "variedad_3d_interactiva_montecarlo_1000.html")
    fig_html.write_html(html_3d, include_plotlyjs='cdn')
    print(f"[Guardado] HTML interactivo 3D guardado en: {html_3d}")

    # Guardar métricas JSON
    metricas = {
        "n_muestras_total": N,
        "n_muestras_por_vocal": N_TRIALS,
        "n_canales": n_ch,
        "puntos_por_canal": TARGET_LEN,
        "dimension_ambiente": D,
        "acc_ambiente_lda": float(acc_amb),
        "pca_2d_lda": float(acc_pca_2d),
        "pca_3d_lda": float(acc_pca_3d),
        "isomap_2d_lda": float(acc_iso_2d),
        "isomap_2d_gmm": float(acc_gmm_2d),
        "isomap_3d_lda": float(acc_iso_3d),
        "evr_pca_3d": [float(v) for v in evr_pca]
    }
    json_path = os.path.join(out_dir, "metricas_montecarlo_1000.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=4)
    print(f"[Guardado] Metricas JSON guardadas en: {json_path}")

if __name__ == '__main__':
    main()
