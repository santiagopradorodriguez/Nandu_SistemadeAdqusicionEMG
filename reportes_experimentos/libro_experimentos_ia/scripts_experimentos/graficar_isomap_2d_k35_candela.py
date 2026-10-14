# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Script: Visualizacion 2D de la Variedad Motora: Isomap 2D (k=35) vs PCA 2D
# Sujeto: Candela (13 Sub-Canales Independientes, 20 Puntos Temporales = 260D)
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
    extraer_dataset_multicanal,
    acondicionar_reposo_impedancia,
    VOCALES,
    COLOR_MAP_HEX,
    TARGET_LEN,
    CANALES_CONFIG
)

np.random.seed(42)
random.seed(42)

project_root = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG'
out_dir = os.path.join(project_root, 'EMG_desarrollo/resultados/grid_search_conv_ortogonal')
os.makedirs(out_dir, exist_ok=True)

def calcular_elipse_2d(pts, n_std=1.5):
    """Calcula ancho, alto y ángulo de la elipse de covarianza 2D a n_std sigmas."""
    cov = np.cov(pts[:, 0], pts[:, 1])
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(1e-6, vals[idx])
    vecs = vecs[:, idx]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w = 2.0 * n_std * np.sqrt(vals[0])
    h = 2.0 * n_std * np.sqrt(vals[1])
    return w, h, theta

def main():
    print("=" * 80)
    print("GENERACION GRAFICA 2D: ISOMAP 2D (k=35) vs PCA 2D - CANDELA 13 CH x 20 PTS")
    print("=" * 80)

    # 1. Extracción y acondicionamiento
    X_raw_syn, y_syn, _ = extraer_dataset_multicanal(target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw_syn.shape
    D = n_ch * n_pts

    sesiones_dummy = np.zeros(N, dtype=int)
    X_cond_3d = acondicionar_reposo_impedancia(X_raw_syn, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond_flat = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. PCA 2D
    pca_2d = PCA(n_components=2, random_state=42)
    Z_pca = pca_2d.fit_transform(X_cond_flat)
    evr_pca = pca_2d.explained_variance_ratio_ * 100.0
    acc_pca_2d = cross_val_score(LinearDiscriminantAnalysis(), Z_pca, y_syn, cv=cv).mean() * 100.0

    # 3. Isomap 2D (k=35)
    K_NEIGHBORS = 35
    iso_2d = Isomap(n_neighbors=K_NEIGHBORS, n_components=2)
    Z_iso_raw = iso_2d.fit_transform(X_cond_flat)
    
    # Orientación canónica:
    # /a/ hacia +Y (semieje vertical positivo, 12 en punto)
    # /i/ hacia +X (semieje horizontal derecho, 3 en punto)
    c_a_raw = np.mean(Z_iso_raw[y_syn == 0], axis=0)
    c_i_raw = np.mean(Z_iso_raw[y_syn == 2], axis=0)

    flip_x = 1.0 if c_i_raw[0] >= 0 else -1.0
    flip_y = 1.0 if c_a_raw[1] >= 0 else -1.0
    Z_iso = np.column_stack([flip_x * Z_iso_raw[:, 0], flip_y * Z_iso_raw[:, 1]])

    acc_iso_2d = cross_val_score(LinearDiscriminantAnalysis(), Z_iso, y_syn, cv=cv).mean() * 100.0

    # GMM con asignación húngara
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(Z_iso)
    preds_gmm = gmm.predict(Z_iso)
    cost = -np.array([[np.sum((preds_gmm == c) & (y_syn == tr)) for tr in range(5)] for c in range(5)])
    ri, ci = linear_sum_assignment(cost)
    acc_gmm_iso = -cost[ri, ci].sum() / len(y_syn) * 100.0

    print(f"PCA 2D: EVR = [{evr_pca[0]:.1f}%, {evr_pca[1]:.1f}%] - Total = {evr_pca.sum():.1f}% | LDA = {acc_pca_2d:.2f}%")
    print(f"Isomap 2D (k={K_NEIGHBORS}): LDA = {acc_iso_2d:.2f}% | GMM = {acc_gmm_iso:.2f}% | Error Geodesico = {iso_2d.reconstruction_error():.4f}")

    # 4. Renderizado en PNG de Alta Resolución
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

    # --- Subplot 1: PCA 2D (Lineal) ---
    ax_pca = axes[0]
    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_pca[idx]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)

        # Nube
        ax_pca.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=45, alpha=0.75, edgecolors='white', linewidth=0.5)
        # Centroide
        ax_pca.scatter(mean_pt[0], mean_pt[1], c=color, s=150, marker='D', edgecolors='black', linewidth=1.6, zorder=6)
        # Elipse de covarianza
        w, h, ang = calcular_elipse_2d(pts, n_std=1.5)
        ell = Ellipse(xy=mean_pt, width=w, height=h, angle=ang, facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.5)
        ax_pca.add_patch(ell)

    ax_pca.set_title(f"A: PCA 2D - Proyeccion Lineal: LDA {acc_pca_2d:.1f}% - Varianza {evr_pca.sum():.1f}%", fontsize=11.5, fontweight='bold', pad=12)
    ax_pca.set_xlabel(f"PC1: {evr_pca[0]:.1f}% Varianza", fontsize=10, fontweight='semibold')
    ax_pca.set_ylabel(f"PC2: {evr_pca[1]:.1f}% Varianza", fontsize=10, fontweight='semibold')
    ax_pca.legend(loc='upper right', framealpha=0.92, fontsize=10)
    ax_pca.grid(True, linestyle=':', alpha=0.5)

    # --- Subplot 2: Isomap 2D (No Lineal, k=35) ---
    ax_iso = axes[1]
    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_iso[idx]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)

        # Nube
        ax_iso.scatter(pts[:, 0], pts[:, 1], c=color, label=f'/{v.lower()}/', s=45, alpha=0.75, edgecolors='white', linewidth=0.5)
        # Centroide
        ax_iso.scatter(mean_pt[0], mean_pt[1], c=color, s=150, marker='D', edgecolors='black', linewidth=1.6, zorder=6)
        # Elipse de covarianza
        w, h, ang = calcular_elipse_2d(pts, n_std=1.5)
        ell = Ellipse(xy=mean_pt, width=w, height=h, angle=ang, facecolor=color, alpha=0.16, edgecolor=color, linewidth=1.5)
        ax_iso.add_patch(ell)

    ax_iso.set_title(f"B: Isomap 2D - Variedad No Lineal Geodesica: k={K_NEIGHBORS} - LDA {acc_iso_2d:.1f}%", fontsize=11.5, fontweight='bold', pad=12)
    ax_iso.set_xlabel("Dimension Geodesica 1: Retraccion vs Protrusion", fontsize=10, fontweight='semibold')
    ax_iso.set_ylabel("Dimension Geodesica 2: Apertura Mandibular Canónica", fontsize=10, fontweight='semibold')
    ax_iso.legend(loc='lower right', framealpha=0.92, fontsize=10)
    ax_iso.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle(
        "Variedad Articulatoria de Candela en 2D: PCA Lineal vs Isomap Geodesico - 13 Sub-Canales Independientes",
        fontsize=13.5, fontweight='bold', y=0.98
    )

    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.06, right=0.96, wspace=0.16)
    png_path = os.path.join(out_dir, "isomap_2d_k35_candela.png")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Guardado] Imagen 2D guardada en: {png_path}")

    # 5. Generar versión interactiva HTML con Plotly
    fig_html = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"A: PCA 2D - Proyeccion Lineal: LDA {acc_pca_2d:.1f}%",
            f"B: Isomap 2D - Variedad Geodesica: k={K_NEIGHBORS} - LDA {acc_iso_2d:.1f}%"
        ],
        horizontal_spacing=0.08
    )

    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        color = COLOR_MAP_HEX[v]

        # PCA 2D
        pts_pca = Z_pca[idx]
        mean_pca = np.mean(pts_pca, axis=0)
        fig_html.add_trace(
            go.Scatter(
                x=pts_pca[:, 0], y=pts_pca[:, 1],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.75, line=dict(color='white', width=0.8)),
                name=f'/{v.lower()}/',
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Vocal: /{v.lower()}/<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )
        fig_html.add_trace(
            go.Scatter(
                x=[mean_pca[0]], y=[mean_pca[1]],
                mode='markers',
                marker=dict(size=14, symbol='diamond', color=color, line=dict(color='black', width=2)),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Centroide /{v.lower()}/<extra></extra>"
            ),
            row=1, col=1
        )

        # Isomap 2D
        pts_iso = Z_iso[idx]
        mean_iso = np.mean(pts_iso, axis=0)
        fig_html.add_trace(
            go.Scatter(
                x=pts_iso[:, 0], y=pts_iso[:, 1],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.75, line=dict(color='white', width=0.8)),
                name=f'/{v.lower()}/',
                legendgroup=f'vocal_{v}',
                showlegend=False,
                hovertemplate=f"Vocal: /{v.lower()}/<br>Dim 1: %{{x:.2f}}<br>Dim 2: %{{y:.2f}}<extra></extra>"
            ),
            row=1, col=2
        )
        fig_html.add_trace(
            go.Scatter(
                x=[mean_iso[0]], y=[mean_iso[1]],
                mode='markers',
                marker=dict(size=14, symbol='diamond', color=color, line=dict(color='black', width=2)),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Centroide /{v.lower()}/<extra></extra>"
            ),
            row=1, col=2
        )

    fig_html.update_layout(
        title=dict(
            text="Variedad Motora Articulatoria de Candela en 2D: PCA vs Isomap k=35 (13 Sub-Canales Independientes)",
            x=0.5, font=dict(size=16, weight="bold")
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#F8F9FA",
        legend=dict(
            orientation='h', x=0.5, y=-0.12, xanchor='center',
            bgcolor='rgba(255,255,255,0.9)', bordercolor='#D0D7DE', borderwidth=1
        )
    )
    fig_html.update_xaxes(showgrid=True, gridcolor="#E1E4E8", zeroline=True, zerolinecolor="#95A5A6")
    fig_html.update_yaxes(showgrid=True, gridcolor="#E1E4E8", zeroline=True, zerolinecolor="#95A5A6")

    html_path = os.path.join(out_dir, "isomap_2d_interactivo_candela.html")
    fig_html.write_html(html_path, include_plotlyjs='cdn')
    print(f"[Guardado] Visualizador interactivo 2D guardado en: {html_path}")

    # Métricas JSON
    metricas_2d = {
        "n_muestras": N,
        "n_subcanales": n_ch,
        "puntos_temporales": n_pts,
        "k_vecinos": K_NEIGHBORS,
        "pca_2d": {
            "evr": [float(v) for v in evr_pca],
            "evr_total": float(evr_pca.sum()),
            "acc_lda": float(acc_pca_2d)
        },
        "isomap_2d": {
            "error_reconstruccion": float(iso_2d.reconstruction_error()),
            "acc_lda": float(acc_iso_2d),
            "acc_gmm": float(acc_gmm_iso)
        }
    }
    json_path = os.path.join(out_dir, "metricas_isomap_2d_k35_candela.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_2d, f, indent=4)
    print(f"[Guardado] Metricas 2D guardadas en: {json_path}")

if __name__ == '__main__':
    main()
