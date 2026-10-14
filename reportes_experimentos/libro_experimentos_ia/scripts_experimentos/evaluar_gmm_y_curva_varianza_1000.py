# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Script: Evaluacion GMM 3D y Curva de Varianza/Dimensionalidad (PCA vs Isomap)
# Sujeto: Candela (12 Canales, 20 Puntos = 240D, N=1000 Monte Carlo)
# ==============================================================================

import os
import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from experimento_montecarlo_1000_candela_12ch import (
    extraer_pool_canales,
    sintetizar_montecarlo,
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

def calcular_exactitud_gmm(Z, y, n_clases=5):
    """Ajusta un GMM de 5 componentes y evalúa con asignación húngara óptima."""
    gmm = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42).fit(Z)
    preds = gmm.predict(Z)
    cost = -np.array([[np.sum((preds == c) & (y == tr)) for tr in range(n_clases)] for c in range(n_clases)])
    ri, ci = linear_sum_assignment(cost)
    acc = -cost[ri, ci].sum() / len(y) * 100.0
    return acc

def calcular_error_residual_tenenbaum(dist_geodesicas_condensadas, Z_embedded):
    """
    Calcula el error residual de Tenenbaum (1 - R^2) entre distancias geodésicas y euclídeas.
    Fórmula canónica: 1 - r^2(D_G, D_Y)
    """
    d_euclid = pdist(Z_embedded)
    r = np.corrcoef(dist_geodesicas_condensadas, d_euclid)[0, 1]
    return 1.0 - (r ** 2)

def main():
    print("=" * 80)
    print("CALCULO GMM 3D Y CURVAS DE VARIANZA / DIMENSIONALIDAD (PCA vs ISOMAP)")
    print("=" * 80)

    # 1. Cargar datos N=1000
    datos_musculos = extraer_pool_canales(target_len=TARGET_LEN)
    X_raw, y = sintetizar_montecarlo(datos_musculos, n_trials_por_vocal=200, target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw.shape
    D = n_ch * n_pts

    X_cond_3d = acondicionar_reposo_impedancia(X_raw, np.zeros(N, int), n_ch, 5)
    X_cond = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------------------------
    # 2. Evaluación GMM en 2D y 3D
    # --------------------------------------------------------------------------
    # PCA
    pca_3d = PCA(n_components=3, random_state=42).fit_transform(X_cond)
    pca_2d = pca_3d[:, :2]
    gmm_pca_2d = calcular_exactitud_gmm(pca_2d, y)
    gmm_pca_3d = calcular_exactitud_gmm(pca_3d, y)

    # Isomap (k=35)
    K_NEIGHBORS = 35
    iso_fit = Isomap(n_neighbors=K_NEIGHBORS, n_components=3)
    iso_3d = iso_fit.fit_transform(X_cond)
    iso_2d = Isomap(n_neighbors=K_NEIGHBORS, n_components=2).fit_transform(X_cond)
    gmm_iso_2d = calcular_exactitud_gmm(iso_2d, y)
    gmm_iso_3d = calcular_exactitud_gmm(iso_3d, y)

    print("\n--- EXACTITUD GMM NO SUPERVISADO (ASIGNACION HUNGARA) ---")
    print(f"PCA 2D:    {gmm_pca_2d:.2f}%")
    print(f"PCA 3D:    {gmm_pca_3d:.2f}%")
    print(f"Isomap 2D: {gmm_iso_2d:.2f}%")
    print(f"Isomap 3D: {gmm_iso_3d:.2f}%")

    # --------------------------------------------------------------------------
    # 3. Barrido de Dimensionalidad (d = 1 a 10)
    # --------------------------------------------------------------------------
    max_d = 10
    dims = np.arange(1, max_d + 1)

    # PCA Varianza y LDA/GMM
    pca_max = PCA(n_components=max_d, random_state=42)
    Z_pca_all = pca_max.fit_transform(X_cond)
    evr_ind = pca_max.explained_variance_ratio_ * 100.0
    evr_cum = np.cumsum(evr_ind)

    lda_pca_dims = []
    gmm_pca_dims = []
    for d in dims:
        sub_z = Z_pca_all[:, :d]
        lda_acc = cross_val_score(LinearDiscriminantAnalysis(), sub_z, y, cv=cv).mean() * 100.0
        gmm_acc = calcular_exactitud_gmm(sub_z, y)
        lda_pca_dims.append(lda_acc)
        gmm_pca_dims.append(gmm_acc)

    # Isomap Error Residual de Tenenbaum y LDA/GMM
    # Isomap y PCA Error Residual de Tenenbaum y LDA/GMM
    # Matriz geodésica del grafo k-NN y matriz euclídea ambiental
    idx_sub = np.random.RandomState(42).choice(N, size=300, replace=False)
    X_sub = X_cond[idx_sub]
    y_sub = y[idx_sub]

    # Distancias Euclídeas ambientales en X_sub (para residual PCA de Tenenbaum)
    d_ambient_condensed = pdist(X_sub)
    pca_sub_all = PCA(n_components=max_d, random_state=42).fit_transform(X_sub)

    iso_graph = Isomap(n_neighbors=K_NEIGHBORS, n_components=max_d)
    Z_iso_all = iso_graph.fit_transform(X_sub)
    d_geo_matrix = iso_graph.dist_matrix_
    i_upper = np.triu_indices(len(X_sub), k=1)
    d_geo_condensed = d_geo_matrix[i_upper]

    tenenbaum_errors_isomap = []
    tenenbaum_errors_pca = []
    pca_residual_unexplained = [100.0 - float(v) for v in evr_cum]
    lda_iso_dims = []
    gmm_iso_dims = []

    for d in dims:
        sub_iso = Z_iso_all[:, :d]
        sub_pca_sub = pca_sub_all[:, :d]

        # Error Tenenbaum: Isomap (geodésico) vs PCA (euclídeo ambiental)
        err_res_iso = calcular_error_residual_tenenbaum(d_geo_condensed, sub_iso) * 100.0
        err_res_pca = calcular_error_residual_tenenbaum(d_ambient_condensed, sub_pca_sub) * 100.0
        tenenbaum_errors_isomap.append(err_res_iso)
        tenenbaum_errors_pca.append(err_res_pca)
        
        # LDA y GMM sobre submuestra
        cv_sub = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        lda_acc = cross_val_score(LinearDiscriminantAnalysis(), sub_iso, y_sub, cv=cv_sub).mean() * 100.0
        gmm_acc = calcular_exactitud_gmm(sub_iso, y_sub)
        lda_iso_dims.append(lda_acc)
        gmm_iso_dims.append(gmm_acc)

    print("\n--- CURVA DE VARIANZA / RESIDUAL POR DIMENSION ---")
    for d in dims:
        print(f"d={d:2d} | PCA EVR Cum: {evr_cum[d-1]:5.1f}% | Residual PCA (1-R²): {tenenbaum_errors_pca[d-1]:5.2f}% | Residual Isomap: {tenenbaum_errors_isomap[d-1]:5.2f}% | LDA PCA: {lda_pca_dims[d-1]:5.1f}% | LDA Isomap: {lda_iso_dims[d-1]:5.1f}%")

    # --------------------------------------------------------------------------
    # 4. Renderizado de Gráficos de Varianza y Dimensionalidad (3 Paneles)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), dpi=300)

    # Panel A: Varianza Explicada PCA (Scree Plot)
    ax_a = axes[0]
    ax_a.bar(dims, evr_ind, color='#2B5B84', alpha=0.8, width=0.55, label='Varianza Individual')
    ax_a.plot(dims, evr_cum, color='#E63946', marker='o', linewidth=2.2, label='Varianza Acumulada')
    for i, (vi, vc) in enumerate(zip(evr_ind, evr_cum)):
        ax_a.text(i+1, vi + 1.0, f"{vi:.1f}%", ha='center', fontsize=8, fontweight='bold')
    ax_a.set_title("A: Espectro de Varianza Explicada PCA: Scree Plot", fontsize=11.5, fontweight='bold', pad=12)
    ax_a.set_xlabel("Dimension Latente: Componentes Principales", fontsize=10, fontweight='semibold')
    ax_a.set_ylabel("Varianza Explicada (%)", fontsize=10, fontweight='semibold')
    ax_a.set_xticks(dims)
    ax_a.set_ylim(0, 105)
    ax_a.legend(loc='center right', framealpha=0.92, fontsize=9.5)
    ax_a.grid(True, linestyle=':', alpha=0.5)

    # Panel B: Error Residual Tenenbaum (Isomap vs PCA)
    ax_b = axes[1]
    ax_b.plot(dims, tenenbaum_errors_pca, color='#E76F51', marker='o', linewidth=2.2, markersize=6, label='PCA: Tenenbaum 1 - R²(D_X, D_Y)')
    ax_b.plot(dims, pca_residual_unexplained, color='#2B5B84', linestyle='--', linewidth=1.8, label='PCA: Varianza No Explicada 100 - EVR')
    ax_b.plot(dims, tenenbaum_errors_isomap, color='#9D4EDD', marker='s', linewidth=2.4, markersize=7, label=f'Isomap: Tenenbaum 1 - R²(D_G, D_Y): k={K_NEIGHBORS}')
    # Resaltar codo en d=3 de Isomap
    ax_b.scatter(3, tenenbaum_errors_isomap[2], color='#E63946', s=160, zorder=6, edgecolors='black', linewidth=1.8)
    ax_b.annotate(
        f"Codo Isomap d=3: {tenenbaum_errors_isomap[2]:.1f}%",
        xy=(3, tenenbaum_errors_isomap[2]), xytext=(4.2, tenenbaum_errors_isomap[2] + 4.0),
        arrowprops=dict(facecolor='#E63946', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold', fontsize=9.5, color='#E63946'
    )
    ax_b.set_title("B: Curva de Error Residual: PCA vs Isomap", fontsize=11.5, fontweight='bold', pad=12)
    ax_b.set_xlabel("Dimension Latente Embebida: d = 1 a 10", fontsize=10, fontweight='semibold')
    ax_b.set_ylabel("Error Residual 1 - R² (%)", fontsize=10, fontweight='semibold')
    ax_b.set_xticks(dims)
    ax_b.set_ylim(0, 100)
    ax_b.legend(loc='upper right', framealpha=0.92, fontsize=8.8)
    ax_b.grid(True, linestyle=':', alpha=0.5)

    # Panel C: Separabilidad Vocálica (LDA y GMM) vs Dimensión
    ax_c = axes[2]
    ax_c.plot(dims, lda_iso_dims, color='#2A9D8F', marker='^', linewidth=2.2, label=f'Isomap LDA: k={K_NEIGHBORS}')
    ax_c.plot(dims, lda_pca_dims, color='#E76F51', marker='o', linewidth=2.2, label='PCA LDA: Lineal')
    ax_c.plot(dims, gmm_iso_dims, color='#264653', marker='v', linewidth=1.8, linestyle='--', label='Isomap GMM: Ciego')
    ax_c.plot(dims, gmm_pca_dims, color='#A0AAB2', marker='d', linewidth=1.8, linestyle=':', label='PCA GMM: Ciego')
    ax_c.axhline(20.0, color='#6C757D', linestyle='-.', alpha=0.7, label='Azar Teorico: 20.0%')
    ax_c.set_title("C: Exactitud de Decodificacion Vocalica vs Dimension Latente", fontsize=11.5, fontweight='bold', pad=12)
    ax_c.set_xlabel("Dimension Latente: d = 1 a 10", fontsize=10, fontweight='semibold')
    ax_c.set_ylabel("Exactitud Multiclase (%)", fontsize=10, fontweight='semibold')
    ax_c.set_xticks(dims)
    ax_c.set_ylim(15, 100)
    ax_c.legend(loc='lower right', framealpha=0.92, fontsize=8.5)
    ax_c.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle("Espectro de Dimensionalidad de la Variedad Motora: PCA Lineal vs Isomap Geodesico - 12 Canales, 240D", fontsize=13.5, fontweight='bold', y=0.98)
    fig.subplots_adjust(top=0.88, bottom=0.10, left=0.05, right=0.96, wspace=0.18)

    save_path = os.path.join(out_dir, "curva_varianza_dimensionalidad_pca_isomap_1000m.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"\n[Guardado] Curva de varianza guardada en: {save_path}")

    # --------------------------------------------------------------------------
    # 5. Gráfico Dedicado: Curva de Error Residual PCA vs Isomap (Tenenbaum Canónico)
    # --------------------------------------------------------------------------
    fig_res, ax_res = plt.subplots(figsize=(9, 6), dpi=300)
    ax_res.plot(dims, tenenbaum_errors_pca, color='#E76F51', marker='o', linewidth=2.5, markersize=8, label='PCA: Tenenbaum 1 - R²(D_X, D_Y)')
    ax_res.plot(dims, pca_residual_unexplained, color='#2B5B84', linestyle='--', linewidth=2.0, marker='.', markersize=6, label='PCA: Varianza No Explicada (100 - EVR_cum)')
    ax_res.plot(dims, tenenbaum_errors_isomap, color='#9D4EDD', marker='s', linewidth=2.6, markersize=8, label=f'Isomap: Tenenbaum 1 - R²(D_G, D_Y) - k={K_NEIGHBORS}')

    # Resaltar codo
    ax_res.scatter(3, tenenbaum_errors_isomap[2], color='#E63946', s=180, zorder=6, edgecolors='black', linewidth=2.0)
    ax_res.annotate(
        f"Codo Geodesico d=3\nError Residual: {tenenbaum_errors_isomap[2]:.1f}%\n(Meseta de estabilidad)",
        xy=(3, tenenbaum_errors_isomap[2]), xytext=(4.3, tenenbaum_errors_isomap[2] + 7.0),
        arrowprops=dict(facecolor='#E63946', shrink=0.08, width=1.8, headwidth=7),
        fontweight='bold', fontsize=10, color='#E63946'
    )
    ax_res.set_title("Varianza Residual vs Dimension Latente: PCA vs Isomap - 12 Canales, 240D", fontsize=12, fontweight='bold', pad=14)
    ax_res.set_xlabel("Dimension Latente: d = 1 a 10", fontsize=11, fontweight='semibold')
    ax_res.set_ylabel("Varianza / Error Residual (%)", fontsize=11, fontweight='semibold')
    ax_res.set_xticks(dims)
    ax_res.set_ylim(0, 100)
    ax_res.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax_res.grid(True, linestyle=':', alpha=0.6)

    dedicated_save_path = os.path.join(out_dir, "curva_varianza_residual_pca_vs_isomap.png")
    plt.savefig(dedicated_save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Guardado] Curva dedicada de varianza residual guardada en: {dedicated_save_path}")

    # Exportar métricas en JSON
    metricas = {
        "n_muestras": N,
        "gmm_pca_2d": float(gmm_pca_2d),
        "gmm_pca_3d": float(gmm_pca_3d),
        "gmm_isomap_2d": float(gmm_iso_2d),
        "gmm_isomap_3d": float(gmm_iso_3d),
        "scree_pca_evr_cum": [float(v) for v in evr_cum],
        "pca_residual_unexplained": [float(v) for v in pca_residual_unexplained],
        "tenenbaum_residual_error_pca": [float(v) for v in tenenbaum_errors_pca],
        "tenenbaum_residual_error_isomap": [float(v) for v in tenenbaum_errors_isomap],
        "lda_pca_por_dimension": [float(v) for v in lda_pca_dims],
        "lda_isomap_por_dimension": [float(v) for v in lda_iso_dims]
    }
    json_path = os.path.join(out_dir, "metricas_curva_varianza_1000m.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas, f, indent=4)
    print(f"[Guardado] Metricas JSON guardadas en: {json_path}")

if __name__ == '__main__':
    main()
