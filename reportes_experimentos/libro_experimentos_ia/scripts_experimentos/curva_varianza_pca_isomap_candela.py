# ==============================================================================
# PROYECTO NANDU LSD - DECODIFICACION BIOELECTRICA ORAL
# Curva de Varianza Explicada y Dimensionalidad Intrínseca (Scree Plot & Tenenbaum)
# Comparativa PCA vs Isomap hasta 10 Dimensiones sobre Pseudo HD-sEMG de Candela
# ==============================================================================

import os
import sys
import json
import random
import numpy as np
import scipy.spatial.distance as dist
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ------------------------------------------------------------------------------
# 1. Semillas y Configuración
# ------------------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

project_root = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG'
base_dir = os.path.join(project_root, 'EMG_desarrollo/base_de_datos_electrodos')
out_dir = os.path.join(project_root, 'EMG_desarrollo/resultados/grid_search_conv_ortogonal')
os.makedirs(out_dir, exist_ok=True)

VOCALES = ['A', 'E', 'I', 'O', 'U']
TARGET_LEN = 20
MAX_DIMS = 10

sys.path.insert(0, project_root)

# Importamos la configuración y extracción del script de 13 sub-canales
from EMG_desarrollo.deep_learning.experimento_pseudo_hd_candela_multicanal_20pts import (
    extraer_dataset_multicanal,
    acondicionar_reposo_impedancia,
    CANALES_CONFIG
)

def calcular_varianza_residual_isomap(D_G, Y_embedded):
    """
    Calcula el error residual geodésico canónico de Tenenbaum (2000):
    E_R = 1 - R^2(D_G, D_Y)
    donde R es el coeficiente de correlación de Pearson entre las distancias
    geodésicas D_G y las distancias euclidianas del embedding D_Y.
    """
    N = len(Y_embedded)
    D_Y = dist.squareform(dist.pdist(Y_embedded, metric='euclidean'))
    
    # Extraer triángulo superior sin diagonal
    idx_upper = np.triu_indices(N, k=1)
    d_g_flat = D_G[idx_upper]
    d_y_flat = D_Y[idx_upper]
    
    # Manejo de varianza cero
    std_g = np.std(d_g_flat)
    std_y = np.std(d_y_flat)
    if std_g < 1e-12 or std_y < 1e-12:
        return 1.0, 0.0
        
    r = np.corrcoef(d_g_flat, d_y_flat)[0, 1]
    r2 = r**2 if not np.isnan(r) else 0.0
    residual_variance = max(0.0, 1.0 - r2)
    explained_variance = r2 * 100.0  # En porcentaje
    return residual_variance, explained_variance

def main():
    print("="*95)
    print("ANALISIS DE DIMENSIONALIDAD INTRINSECA Y CURVA DE VARIANZA EXPLICADA")
    print(f"Comparativa PCA (Scree Plot) vs Isomap (Tenenbaum Residual) en d = 1..{MAX_DIMS}")
    print("="*95)

    X_raw_syn, y_syn, _ = extraer_dataset_multicanal(target_len=TARGET_LEN)
    N, n_ch, n_pts = X_raw_syn.shape
    D = n_ch * n_pts

    # Acondicionamiento oficial de reposo basal e impedancia P95
    sesiones_dummy = np.zeros(N, dtype=int)
    X_cond_3d = acondicionar_reposo_impedancia(X_raw_syn, sesiones_dummy, n_canales=n_ch, n_pts_reposo=5)
    X_cond_flat = X_cond_3d.reshape(N, -1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --------------------------------------------------------------------------
    # 1. PCA: Varianza Explicada Acumulada y LDA vs Dimensión
    # --------------------------------------------------------------------------
    pca_full = PCA(n_components=MAX_DIMS)
    Z_pca_all = pca_full.fit_transform(X_cond_flat)
    pca_evr_cum = np.cumsum(pca_full.explained_variance_ratio_) * 100.0
    pca_evr_ind = pca_full.explained_variance_ratio_ * 100.0
    pca_residual = 100.0 - pca_evr_cum

    pca_lda_accs = []
    for d in range(1, MAX_DIMS + 1):
        acc = cross_val_score(LinearDiscriminantAnalysis(), Z_pca_all[:, :d], y_syn, cv=cv).mean() * 100.0
        pca_lda_accs.append(acc)

    # --------------------------------------------------------------------------
    # 2. Isomap: Ajuste Geodésico R^2, Error Residual de Tenenbaum y LDA vs Dimensión
    # --------------------------------------------------------------------------
    dims = np.arange(1, MAX_DIMS + 1)
    iso_explained_var = []
    iso_residual_var = []
    iso_lda_accs = []

    # Ajustamos Isomap con n_neighbors=15 a la máxima dimensión para obtener la matriz geodésica D_G
    iso_base = Isomap(n_neighbors=15, n_components=MAX_DIMS)
    Z_iso_all = iso_base.fit_transform(X_cond_flat)
    D_G = iso_base.dist_matrix_  # Matriz de distancias geodésicas en el grafo

    for d in range(1, MAX_DIMS + 1):
        if d == MAX_DIMS:
            Z_d = Z_iso_all
        else:
            iso_d = Isomap(n_neighbors=15, n_components=d)
            Z_d = iso_d.fit_transform(X_cond_flat)
            
        res_var, exp_var = calcular_varianza_residual_isomap(D_G, Z_d)
        iso_residual_var.append(res_var * 100.0)
        iso_explained_var.append(exp_var)
        
        acc_iso = cross_val_score(LinearDiscriminantAnalysis(), Z_d, y_syn, cv=cv).mean() * 100.0
        iso_lda_accs.append(acc_iso)

    # --------------------------------------------------------------------------
    # Tabla de Resultados en Terminal
    # --------------------------------------------------------------------------
    print("\n" + "="*95)
    print("DIMENSION | PCA Var Acum (%) | PCA Residual (%) | PCA LDA (%) | Isomap R^2 (%) | Isomap Res (%) | Iso LDA (%)")
    print("="*95)
    for i, d in enumerate(dims):
        print(f"  d = {d:2d}  |      {pca_evr_cum[i]:5.2f}%     |      {pca_residual[i]:5.2f}%     |   {pca_lda_accs[i]:5.2f}%   |     {iso_explained_var[i]:5.2f}%    |     {iso_residual_var[i]:5.2f}%    |   {iso_lda_accs[i]:5.2f}%")
    print("="*95)

    # --------------------------------------------------------------------------
    # 3. Gráfico Oficial de Curvas de Varianza (Scree Plot y Tenenbaum)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)

    # Gráfico 1: Varianza Explicada Acumulada (Scree Plot / R^2)
    ax1 = axes[0]
    ax1.plot(dims, pca_evr_cum, color='#1F77B4', marker='o', linewidth=2.2, label='PCA (Lineal - Var Acumulada)')
    ax1.plot(dims, iso_explained_var, color='#2A9D8F', marker='s', linewidth=2.2, label='Isomap (Geodésico - R^2 Tenenbaum)')
    ax1.set_title("A: Curva de Varianza Explicada Acumulada vs Dimension", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Dimension del Espacio Latente (d)", fontsize=10)
    ax1.set_ylabel("Varianza Explicada (%)", fontsize=10)
    ax1.set_xticks(dims)
    ax1.set_ylim(0, 105)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=9)

    # Gráfico 2: Varianza Residual (Curva de Codo de Tenenbaum)
    ax2 = axes[1]
    ax2.plot(dims, pca_residual, color='#E63946', marker='o', linewidth=2.2, label='PCA Varianza Residual (100 - Var%)')
    ax2.plot(dims, iso_residual_var, color='#9D4EDD', marker='s', linewidth=2.2, label='Isomap Error Residual 1 - R^2 (Tenenbaum)')
    ax2.axvline(3, color='gray', linestyle='--', linewidth=1.2, label='Codo Fisiológico (d = 3)')
    ax2.set_title("B: Curva de Error Residual vs Dimension (Codo de Cattell)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Dimension del Espacio Latente (d)", fontsize=10)
    ax2.set_ylabel("Error / Varianza Residual (%)", fontsize=10)
    ax2.set_xticks(dims)
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Gráfico 3: Exactitud de Decodificación Vocalica (LDA) vs Dimensión
    ax3 = axes[2]
    ax3.plot(dims, pca_lda_accs, color='#1F77B4', marker='o', linewidth=2.2, label='PCA + LDA 5-fold')
    ax3.plot(dims, iso_lda_accs, color='#2A9D8F', marker='s', linewidth=2.2, label='Isomap + LDA 5-fold')
    ax3.axhline(96.67, color='#2B5B84', linestyle=':', linewidth=1.4, label='Techo 260D Directo (96.7%)')
    ax3.axhline(20.0, color='#6C757D', linestyle='--', linewidth=1.2, label='Azar Teórico (20.0%)')
    ax3.set_title("C: Desempeño LDA de Decodificacion Vocalica vs Dimension", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Dimension del Espacio Latente (d)", fontsize=10)
    ax3.set_ylabel("Exactitud Multiclase (%)", fontsize=10)
    ax3.set_xticks(dims)
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend(loc='lower right', framealpha=0.9, fontsize=9)

    fig.suptitle("Analisis de Dimensionalidad Intrinseca del Pseudo HD-sEMG de Candela (13 Canales, 20 Pts = 260D)", fontsize=13, fontweight='bold', y=0.98)

    save_path = os.path.join(out_dir, "curva_varianza_dimensionalidad_pca_isomap.png")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"\n[Guardado] Curva de dimensionalidad guardada en: {save_path}")

    # Guardar JSON con las métricas completas
    res_dict = {
        "dimensiones": dims.tolist(),
        "pca_var_acumulada": pca_evr_cum.tolist(),
        "pca_residual": pca_residual.tolist(),
        "pca_lda_accuracy": pca_lda_accs,
        "isomap_r2_explained": iso_explained_var,
        "isomap_residual_variance": iso_residual_var,
        "isomap_lda_accuracy": iso_lda_accs
    }
    json_path = os.path.join(out_dir, "metricas_curva_varianza_pca_isomap.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(res_dict, f, indent=4)
    print(f"[Guardado] Metricas JSON guardadas en: {json_path}")

if __name__ == '__main__':
    main()
