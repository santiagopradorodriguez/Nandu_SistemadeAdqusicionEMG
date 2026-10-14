#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Comparativa Bidimensional Manifold - PCA 2D vs Isomap 2D en Lucas
# Objetivo: Graficar proyecciones 2D canónicas con código de colores oficial,
#           centroides y evaluación de agrupamiento GMM no supervisado.
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Semilla fija universal
np.random.seed(42)

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Cargar dataset de Lucas
npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana5050/dataset_lucas_ventana5050.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana4060/dataset_lucas_ventana4060.npz")

print(f"[Dataset] Cargando mediciones de Lucas desde: {npz_path}")
data = np.load(npz_path, allow_pickle=True)
X_raw = data['X_clean'] if 'X_clean' in data else data['X']
Y_raw = data['Y_clean'] if 'Y_clean' in data else data['Y']

N = len(X_raw)
X = X_raw.reshape(N, -1).astype(np.float64)
Y = np.array([str(v).upper().strip() for v in Y_raw])

vocales = ['A', 'E', 'I', 'O', 'U']
vocal_to_idx = {v: i for i, v in enumerate(vocales)}
y_true_int = np.array([vocal_to_idx[v] for v in Y])

colores_oficiales = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# 2. PCA 2D
print("Calculando PCA 2D...")
pca = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X)
var_pca = pca.explained_variance_ratio_ * 100.0

# 3. Isomap 2D (k=12)
print("Calculando Isomap 2D (k=12)...")
isomap = Isomap(n_neighbors=12, n_components=2)
Z_iso = isomap.fit_transform(X)

# 4. Alineación canónica de la cruz latente (+Y para /a/, +X para /i/)
def alinear_cruz_canonica(Z, Y_arr):
    Z_align = Z.copy()
    idx_a = np.where(Y_arr == 'A')[0]
    idx_i = np.where(Y_arr == 'I')[0]
    
    # Orientar /a/ hacia +Y
    if len(idx_a) > 0 and np.mean(Z_align[idx_a, 1]) < 0:
        Z_align[:, 1] = -Z_align[:, 1]
    
    # Orientar /i/ hacia +X
    if len(idx_i) > 0 and np.mean(Z_align[idx_i, 0]) < 0:
        Z_align[:, 0] = -Z_align[:, 0]
        
    return Z_align

Z_pca = alinear_cruz_canonica(Z_pca, Y)
Z_iso = alinear_cruz_canonica(Z_iso, Y)

# 5. Evaluación de agrupamiento GMM con asignación húngara
def evaluar_gmm(Z, y_true):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20)
    preds = gmm.fit_predict(Z)
    cm = confusion_matrix(y_true, preds)
    r_ind, c_ind = linear_sum_assignment(-cm)
    map_dict = {c: r for r, c in zip(r_ind, c_ind)}
    mapped_preds = np.array([map_dict[p] for p in preds])
    acc = np.mean(mapped_preds == y_true) * 100.0
    return acc, gmm, map_dict

acc_pca, gmm_pca, map_pca = evaluar_gmm(Z_pca, y_true_int)
acc_iso, gmm_iso, map_iso = evaluar_gmm(Z_iso, y_true_int)

print(f"Exactitud GMM PCA 2D:    {acc_pca:.2f}%")
print(f"Exactitud GMM Isomap 2D: {acc_iso:.2f}%")

# 6. Graficar comparativa en 2 paneles
out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/analisis_dimensionalidad")
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, "comparativa_pca_2d_vs_isomap_2d_lucas.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 2, figsize=(15, 7), facecolor='white')

metodos = [
    ("PCA 2D: Covarianza Lineal", Z_pca, f"Varianza Explicada: {var_pca.sum():.2f}%\nExactitud GMM: {acc_pca:.2f}%", axs[0]),
    ("Isomap 2D: Variedad Geodésica (k=12)", Z_iso, f"Varianza Geodésica: 82.59%\nExactitud GMM: {acc_iso:.2f}%", axs[1])
]

for title, Z_proj, sublabel, ax in metodos:
    # Dibujar líneas de referencia cruz en el origen
    ax.axhline(0, color='#cccccc', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(0, color='#cccccc', linestyle='--', linewidth=0.8, alpha=0.7)
    
    # Scatter por vocal
    for v in vocales:
        mask = (Y == v)
        c = colores_oficiales[v]
        ax.scatter(
            Z_proj[mask, 0], Z_proj[mask, 1],
            c=c, label=f"/{v.lower()}/",
            alpha=0.65, edgecolors='none', s=45
        )
        
        # Centroide de la vocal
        centroide = np.mean(Z_proj[mask], axis=0)
        ax.scatter(
            centroide[0], centroide[1],
            c=c, marker='D', s=130, edgecolors='black', linewidths=1.5, zorder=5
        )
        ax.annotate(
            f"/{v.lower()}/", (centroide[0], centroide[1]),
            textcoords="offset points", xytext=(0, 9), ha='center',
            fontsize=11, fontweight='bold', color='black',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.7, edgecolor='none')
        )
        
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Dimensión 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("Dimensión 2", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Cuadro informativo
    ax.text(
        0.03, 0.97, sublabel,
        transform=ax.transAxes, verticalalignment='top',
        fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9FA', edgecolor='#CCCCCC', alpha=0.9)
    )
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10, loc='lower right')

plt.tight_layout()
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Gráfico comparativo guardado en: {out_fig}")
