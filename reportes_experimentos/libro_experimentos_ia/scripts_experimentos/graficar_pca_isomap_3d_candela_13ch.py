# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Script: Visualizacion 3D de la Variedad Motora: PCA 3D vs Isomap 3D (k=35)
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
from mpl_toolkits.mplot3d import Axes3D
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

def configurar_ejes_3d_limpios(ax, x_label, y_label, z_label, z_min, z_max):
    """Estiliza los paneles 3D para un aspecto limpio, sobrio y profesional."""
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('#D0D7DE')
    ax.yaxis.pane.set_edgecolor('#D0D7DE')
    ax.zaxis.pane.set_edgecolor('#D0D7DE')
    
    ax.grid(True, linestyle=':', color='#BDC3C7', alpha=0.6)
    
    ax.set_xlabel(x_label, fontsize=10, labelpad=9, fontweight='semibold')
    ax.set_ylabel(y_label, fontsize=10, labelpad=9, fontweight='semibold')
    ax.set_zlabel(z_label, fontsize=10, labelpad=9, fontweight='semibold')
    
    ax.set_zlim(z_min, z_max)

def main():
    print("=" * 80)
    print("GENERACION GRAFICA 3D: VARIEDAD ARTICULATORIA EN CANDELA (13 CH x 20 PTS)")
    print("PCA 3D vs Isomap 3D (k=35) - Diseno Profesional")
    print("=" * 80)

    # 1. Cargar y acondicionar datos
    X_raw_syn, y_syn, _ = extraer_dataset_multicanal(target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw_syn.shape
    D = n_ch * n_pts

    sesiones_dummy = np.zeros(N, dtype=int)
    X_cond_3d = acondicionar_reposo_impedancia(X_raw_syn, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond_flat = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 2. Ajuste PCA 3D
    pca_3d = PCA(n_components=3, random_state=42)
    Z_pca = pca_3d.fit_transform(X_cond_flat)
    evr_pca = pca_3d.explained_variance_ratio_ * 100.0
    acc_pca_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_pca, y_syn, cv=cv).mean() * 100.0

    # 3. Ajuste Isomap 3D (k=35)
    K_NEIGHBORS = 35
    iso_3d = Isomap(n_neighbors=K_NEIGHBORS, n_components=3)
    Z_iso = iso_3d.fit_transform(X_cond_flat)
    acc_iso_3d = cross_val_score(LinearDiscriminantAnalysis(), Z_iso, y_syn, cv=cv).mean() * 100.0

    print(f"\nResultados de Reduccion 3D:")
    print(f"PCA 3D: EVR = [{evr_pca[0]:.1f}%, {evr_pca[1]:.1f}%, {evr_pca[2]:.1f}%] - Total = {evr_pca.sum():.1f}% | LDA = {acc_pca_3d:.2f}%")
    print(f"Isomap 3D (k={K_NEIGHBORS}): Error Geodesico = {iso_3d.reconstruction_error():.4f} | LDA = {acc_iso_3d:.2f}%")

    # 4. Configurar Figura
    fig = plt.figure(figsize=(20, 9.5), dpi=300)

    # Ángulos de visión elegidos para máxima claridad
    elev_pca, azim_pca = 24, -45
    elev_iso, azim_iso = 22, 50

    # --------------------------------------------------------------------------
    # Panel 1: PCA 3D
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    z_floor_pca = np.min(Z_pca[:, 2]) - 0.20 * np.ptp(Z_pca[:, 2])
    z_ceil_pca = np.max(Z_pca[:, 2]) + 0.15 * np.ptp(Z_pca[:, 2])

    configurar_ejes_3d_limpios(
        ax1,
        x_label=f"PC1: {evr_pca[0]:.1f}% Varianza",
        y_label=f"PC2: {evr_pca[1]:.1f}% Varianza",
        z_label=f"PC3: {evr_pca[2]:.1f}% Varianza",
        z_min=z_floor_pca,
        z_max=z_ceil_pca
    )

    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_pca[idx]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)

        # Proyección basal (sombra)
        ax1.scatter(
            pts[:, 0], pts[:, 1], zs=z_floor_pca, zdir='z',
            c=color, alpha=0.18, s=24, edgecolors='none', depthshade=False
        )
        # Sombra de centroide en piso
        ax1.scatter(
            mean_pt[0], mean_pt[1], zs=z_floor_pca, zdir='z',
            c=color, alpha=0.55, s=90, marker='o', edgecolors='black', linewidth=1.0, depthshade=False
        )
        # Línea de proyección vertical (guía de altura)
        ax1.plot(
            [mean_pt[0], mean_pt[0]],
            [mean_pt[1], mean_pt[1]],
            [z_floor_pca, mean_pt[2]],
            color=color, linestyle='--', linewidth=1.4, alpha=0.75
        )

        # Nube 3D principal
        ax1.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, label=f'/{v.lower()}/', s=55, alpha=0.85,
            edgecolors='white', linewidth=0.6, depthshade=True
        )

        # Centroide principal
        ax1.scatter(
            mean_pt[0], mean_pt[1], mean_pt[2],
            c=color, s=180, marker='D', edgecolors='black', linewidth=1.8, zorder=10
        )

    ax1.view_init(elev=elev_pca, azim=azim_pca)
    ax1.set_title(
        f"A: PCA 3D - Proyeccion Lineal: {evr_pca.sum():.1f}% Varianza Explicada - LDA {acc_pca_3d:.1f}%",
        fontsize=12, fontweight='bold', pad=14
    )
    ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), framealpha=0.92, fontsize=10)

    # --------------------------------------------------------------------------
    # Panel 2: Isomap 3D (k=35)
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    z_floor_iso = np.min(Z_iso[:, 2]) - 0.20 * np.ptp(Z_iso[:, 2])
    z_ceil_iso = np.max(Z_iso[:, 2]) + 0.15 * np.ptp(Z_iso[:, 2])

    configurar_ejes_3d_limpios(
        ax2,
        x_label="Dimension Geodesica 1",
        y_label="Dimension Geodesica 2",
        z_label="Dimension Geodesica 3",
        z_min=z_floor_iso,
        z_max=z_ceil_iso
    )

    for vi, v in enumerate(VOCALES):
        idx = np.where(y_syn == vi)[0]
        pts = Z_iso[idx]
        color = COLOR_MAP_HEX[v]
        mean_pt = np.mean(pts, axis=0)

        # Proyección basal (sombra)
        ax2.scatter(
            pts[:, 0], pts[:, 1], zs=z_floor_iso, zdir='z',
            c=color, alpha=0.18, s=24, edgecolors='none', depthshade=False
        )
        # Sombra de centroide en piso
        ax2.scatter(
            mean_pt[0], mean_pt[1], zs=z_floor_iso, zdir='z',
            c=color, alpha=0.55, s=90, marker='o', edgecolors='black', linewidth=1.0, depthshade=False
        )
        # Línea de proyección vertical (guía de altura)
        ax2.plot(
            [mean_pt[0], mean_pt[0]],
            [mean_pt[1], mean_pt[1]],
            [z_floor_iso, mean_pt[2]],
            color=color, linestyle='--', linewidth=1.4, alpha=0.75
        )

        # Nube 3D principal
        ax2.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            c=color, label=f'/{v.lower()}/', s=55, alpha=0.85,
            edgecolors='white', linewidth=0.6, depthshade=True
        )

        # Centroide principal
        ax2.scatter(
            mean_pt[0], mean_pt[1], mean_pt[2],
            c=color, s=180, marker='D', edgecolors='black', linewidth=1.8, zorder=10
        )

    ax2.view_init(elev=elev_iso, azim=azim_iso)
    ax2.set_title(
        f"B: Isomap 3D - Variedad No Lineal Geodesica: k={K_NEIGHBORS} - LDA {acc_iso_3d:.1f}%",
        fontsize=12, fontweight='bold', pad=14
    )
    ax2.legend(loc='upper right', bbox_to_anchor=(0.98, 0.96), framealpha=0.92, fontsize=10)

    # Título General y Espaciado Limpio
    fig.suptitle(
        f"Variedad Motora Articulatoria de Candela en 3D: 13 Sub-Canales Independientes a 20 Puntos - 260D",
        fontsize=14, fontweight='bold', y=0.97
    )

    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.03, right=0.97, wspace=0.08)
    save_path = os.path.join(out_dir, "pca_isomap_3d_candela_13ch.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Guardado] Grafico 3D guardado exitosamente en: {save_path}")

    # Guardar métricas 3D en JSON
    metricas_3d = {
        "n_muestras": N,
        "n_subcanales": n_ch,
        "puntos_temporales": n_pts,
        "dimension_ambiente": D,
        "pca_3d": {
            "evr_componentes": [float(v) for v in evr_pca],
            "evr_total": float(evr_pca.sum()),
            "acc_lda_cv5": float(acc_pca_3d)
        },
        "isomap_3d": {
            "k_vecinos": K_NEIGHBORS,
            "error_reconstruccion": float(iso_3d.reconstruction_error()),
            "acc_lda_cv5": float(acc_iso_3d)
        }
    }
    json_path = os.path.join(out_dir, "metricas_pca_isomap_3d_candela.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_3d, f, indent=4)
    print(f"[Guardado] Metricas 3D guardadas en: {json_path}")

if __name__ == '__main__':
    main()
