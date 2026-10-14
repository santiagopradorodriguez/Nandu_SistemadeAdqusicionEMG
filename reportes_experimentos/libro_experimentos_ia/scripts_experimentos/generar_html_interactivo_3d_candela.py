# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Script: Generacion de Visualizador 3D Interactivo HTML (PCA 3D vs Isomap 3D)
# Sujeto: Candela (13 Sub-Canales Independientes, 20 Puntos Temporales = 260D)
# ==============================================================================

import os
import sys
import json
import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

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

def main():
    print("=" * 80)
    print("GENERADOR 3D INTERACTIVO HTML: CANDELA (13 CH x 20 PTS)")
    print("PCA 3D vs Isomap 3D (k=35)")
    print("=" * 80)

    # 1. Extracción y acondicionamiento
    X_raw_syn, y_syn, _ = extraer_dataset_multicanal(target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw_syn.shape
    D = n_ch * n_pts

    sesiones_dummy = np.zeros(N, dtype=int)
    X_cond_3d = acondicionar_reposo_impedancia(X_raw_syn, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond_flat = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. PCA 3D
    pca_3d = PCA(n_components=3, random_state=42)
    Z_pca = pca_3d.fit_transform(X_cond_flat)
    evr_pca = pca_3d.explained_variance_ratio_ * 100.0
    acc_pca_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_pca, y_syn, cv=cv).mean() * 100.0

    # 3. Isomap 3D (k=35)
    K_NEIGHBORS = 35
    iso_3d = Isomap(n_neighbors=K_NEIGHBORS, n_components=3)
    Z_iso = iso_3d.fit_transform(X_cond_flat)
    acc_iso_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_iso, y_syn, cv=cv).mean() * 100.0

    print(f"PCA 3D: EVR Total = {evr_pca.sum():.1f}% | LDA = {acc_pca_3d:.2f}%")
    print(f"Isomap 3D (k={K_NEIGHBORS}): Error = {iso_3d.reconstruction_error():.4f} | LDA = {acc_iso_3d:.2f}%")

    # 4. Crear figura de Plotly con dos escenas 3D
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[
            f"A: PCA 3D - Proyeccion Lineal: {evr_pca.sum():.1f}% Varianza - LDA {acc_pca_3d:.1f}%",
            f"B: Isomap 3D - Variedad No Lineal Geodesica: k={K_NEIGHBORS} - LDA {acc_iso_3d:.1f}%"
        ],
        horizontal_spacing=0.04
    )

    z_floor_pca = np.min(Z_pca[:, 2]) - 0.20 * np.ptp(Z_pca[:, 2])
    z_floor_iso = np.min(Z_iso[:, 2]) - 0.20 * np.ptp(Z_iso[:, 2])

    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        color = COLOR_MAP_HEX[v]

        # ----------------- PANEL 1: PCA 3D -----------------
        pts_pca = Z_pca[idx]
        mean_pca = np.mean(pts_pca, axis=0)

        # Nube de puntos
        fig.add_trace(
            go.Scatter3d(
                x=pts_pca[:, 0], y=pts_pca[:, 1], z=pts_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5.5,
                    color=color,
                    opacity=0.85,
                    line=dict(color='white', width=0.8)
                ),
                name=f'/{v.lower()}/',
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Vocal: /{v.lower()}/<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )

        # Sombra en piso
        fig.add_trace(
            go.Scatter3d(
                x=pts_pca[:, 0], y=pts_pca[:, 1], z=np.full(len(pts_pca), z_floor_pca),
                mode='markers',
                marker=dict(size=3.5, color=color, opacity=0.22),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Tallo vertical del centroide
        fig.add_trace(
            go.Scatter3d(
                x=[mean_pca[0], mean_pca[0]],
                y=[mean_pca[1], mean_pca[1]],
                z=[z_floor_pca, mean_pca[2]],
                mode='lines',
                line=dict(color=color, width=4, dash='dash'),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        # Centroide
        fig.add_trace(
            go.Scatter3d(
                x=[mean_pca[0]], y=[mean_pca[1]], z=[mean_pca[2]],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='diamond',
                    color=color,
                    line=dict(color='black', width=2)
                ),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Centroide /{v.lower()}/<br>PC1: %{{x:.2f}}<br>PC2: %{{y:.2f}}<br>PC3: %{{z:.2f}}<extra></extra>"
            ),
            row=1, col=1
        )

        # ----------------- PANEL 2: ISOMAP 3D -----------------
        pts_iso = Z_iso[idx]
        mean_iso = np.mean(pts_iso, axis=0)

        # Nube de puntos
        fig.add_trace(
            go.Scatter3d(
                x=pts_iso[:, 0], y=pts_iso[:, 1], z=pts_iso[:, 2],
                mode='markers',
                marker=dict(
                    size=5.5,
                    color=color,
                    opacity=0.85,
                    line=dict(color='white', width=0.8)
                ),
                name=f'/{v.lower()}/',
                legendgroup=f'vocal_{v}',
                showlegend=False,
                hovertemplate=f"Vocal: /{v.lower()}/<br>Dim 1: %{{x:.2f}}<br>Dim 2: %{{y:.2f}}<br>Dim 3: %{{z:.2f}}<extra></extra>"
            ),
            row=1, col=2
        )

        # Sombra en piso
        fig.add_trace(
            go.Scatter3d(
                x=pts_iso[:, 0], y=pts_iso[:, 1], z=np.full(len(pts_iso), z_floor_iso),
                mode='markers',
                marker=dict(size=3.5, color=color, opacity=0.22),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hoverinfo='skip'
            ),
            row=1, col=2
        )

        # Tallo vertical del centroide
        fig.add_trace(
            go.Scatter3d(
                x=[mean_iso[0], mean_iso[0]],
                y=[mean_iso[1], mean_iso[1]],
                z=[z_floor_iso, mean_iso[2]],
                mode='lines',
                line=dict(color=color, width=4, dash='dash'),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hoverinfo='skip'
            ),
            row=1, col=2
        )

        # Centroide
        fig.add_trace(
            go.Scatter3d(
                x=[mean_iso[0]], y=[mean_iso[1]], z=[mean_iso[2]],
                mode='markers',
                marker=dict(
                    size=12,
                    symbol='diamond',
                    color=color,
                    line=dict(color='black', width=2)
                ),
                showlegend=False,
                legendgroup=f'vocal_{v}',
                hovertemplate=f"Centroide /{v.lower()}/<br>Dim 1: %{{x:.2f}}<br>Dim 2: %{{y:.2f}}<br>Dim 3: %{{z:.2f}}<extra></extra>"
            ),
            row=1, col=2
        )

    # Configuración de escenas 3D (cámaras, ejes y estilo)
    fig.update_layout(
        title=dict(
            text="Variedad Motora Articulatoria de Candela en 3D: 13 Sub-Canales Independientes a 20 Puntos - 260D",
            x=0.5, y=0.96, font=dict(size=18, family="sans-serif", weight="bold")
        ),
        paper_bgcolor="#F8F9FA",
        scene=dict(
            xaxis=dict(title=f"PC1: {evr_pca[0]:.1f}% Varianza", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            yaxis=dict(title=f"PC2: {evr_pca[1]:.1f}% Varianza", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            zaxis=dict(title=f"PC3: {evr_pca[2]:.1f}% Varianza", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.9))
        ),
        scene2=dict(
            xaxis=dict(title="Dimension Geodesica 1", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            yaxis=dict(title="Dimension Geodesica 2", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            zaxis=dict(title="Dimension Geodesica 3", backgroundcolor="#FFFFFF", gridcolor="#E1E4E8"),
            camera=dict(eye=dict(x=1.3, y=-1.6, z=1.1))
        ),
        legend=dict(
            x=0.5, y=0.02, xanchor='center', yanchor='bottom',
            orientation='h', bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#D0D7DE', borderwidth=1, font=dict(size=13)
        ),
        margin=dict(l=20, r=20, t=80, b=40)
    )

    html_path = os.path.join(out_dir, "variedad_3d_interactiva_candela.html")
    fig.write_html(html_path, include_plotlyjs='cdn')
    print(f"\n[Guardado] Visualizador interactivo 3D guardado en: {html_path}")

if __name__ == '__main__':
    main()
