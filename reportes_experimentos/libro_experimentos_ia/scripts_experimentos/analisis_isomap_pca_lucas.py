#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Analisis de Dimensionalidad Manifold - PCA vs Isomap en Lucas
# Objetivo: Evaluar y graficar Varianza Explicada y Residual vs Dimensiones
#           (Criterio de Tenenbaum et al. Science 2000 y Espectro de Autovalores)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

# Fijar semilla universal
np.random.seed(42)

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Cargar dataset de Lucas
rutas_candidatas = [
    os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana5050/dataset_lucas_ventana5050.npz"),
    os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana4060/dataset_lucas_ventana4060.npz"),
    os.path.join(project_root, "EMG_desarrollo/deep_learning/codigos_de_prueba/cache_datos/dataset_lucas_20260710_503pulsos.npz"),
]

dataset_path = None
for r in rutas_candidatas:
    if os.path.exists(r):
        dataset_path = r
        break

if dataset_path is None:
    raise FileNotFoundError("No se encontro un dataset en cache de Lucas.")

print(f"[Dataset] Cargando mediciones de Lucas desde:\n          {dataset_path}")
data = np.load(dataset_path, allow_pickle=True)

if 'X_clean' in data:
    X_raw = data['X_clean']
    Y_raw = data['Y_clean']
elif 'X' in data:
    X_raw = data['X']
    Y_raw = data['Y']
else:
    raise KeyError(f"Claves disponibles en npz: {list(data.keys())}")

# Vectorizar a matriz 2D (N, D)
if X_raw.ndim > 2:
    N = len(X_raw)
    X = X_raw.reshape(N, -1).astype(np.float64)
else:
    N = len(X_raw)
    X = X_raw.astype(np.float64)

D_in = X.shape[1]
print(f"[Dataset] Muestras: N = {N}, Dimension original: D = {D_in}")

# Centrar datos
X_cent = X - np.mean(X, axis=0)

# Rango de dimensiones a explorar
d_max = min(15, D_in)
dims = np.arange(1, d_max + 1)

# ==============================================================================
# 2. CALCULO DE PCA (Lineal)
# ==============================================================================
print("\n--- Ejecutando PCA ---")
pca_full = PCA(n_components=d_max, random_state=42)
pca_full.fit(X_cent)

# Varianza por autovalores de covarianza
pca_var_ind = pca_full.explained_variance_ratio_ * 100.0
pca_var_cum = np.cumsum(pca_full.explained_variance_ratio_) * 100.0
pca_res_var_eigen = 1.0 - np.cumsum(pca_full.explained_variance_ratio_)

# Varianza residual por distancias euclidianas (Tenenbaum)
D_euclid = squareform(pdist(X_cent, metric='euclidean'))
triu_idx = np.triu_indices(N, k=1)
d_euclid_flat = D_euclid[triu_idx]

pca_res_var_dist = []
for d in dims:
    Y_pca = pca_full.transform(X_cent)[:, :d]
    D_Y = squareform(pdist(Y_pca, metric='euclidean'))
    d_y_flat = D_Y[triu_idx]
    r, _ = pearsonr(d_euclid_flat, d_y_flat)
    pca_res_var_dist.append(1.0 - (r ** 2))

pca_res_var_dist = np.array(pca_res_var_dist)

# ==============================================================================
# 3. CALCULO DE ISOMAP (No lineal / Geodesico)
# ==============================================================================
print("\n--- Ejecutando Isomap ---")
# Probar con k=12 (vecindad local robusta)
k_neighbors = 12
print(f"[Isomap] Parametro de vecindad: k = {k_neighbors}")

# Construir grafo de k-vecinos y calcular matriz geodésica exacta
adj_graph = kneighbors_graph(X_cent, n_neighbors=k_neighbors, mode='distance', include_self=False)
D_geodesic = shortest_path(adj_graph, directed=False)

# Verificar conexidad
if np.isinf(D_geodesic).any():
    print("  [Aviso] Grafo con componentes disconexas. Ajustando k a 15 para conexidad completa...")
    k_neighbors = 15
    adj_graph = kneighbors_graph(X_cent, n_neighbors=k_neighbors, mode='distance', include_self=False)
    D_geodesic = shortest_path(adj_graph, directed=False)
    if np.isinf(D_geodesic).any():
        max_finite = np.max(D_geodesic[np.isfinite(D_geodesic)])
        D_geodesic[np.isinf(D_geodesic)] = max_finite * 2.0

d_geo_flat = D_geodesic[triu_idx]

# 3A. Varianza explicada por autovalores del Kernel MDS Centrado de Isomap
H = np.eye(N) - (1.0 / N) * np.ones((N, N))
B_kernel = -0.5 * (H @ (D_geodesic ** 2) @ H)
evals_kernel = np.linalg.eigvalsh(B_kernel)
evals_pos = evals_kernel[evals_kernel > 0][::-1]
sum_evals = np.sum(evals_pos)

isomap_var_ind = (evals_pos[:d_max] / sum_evals) * 100.0
isomap_var_cum = np.cumsum(evals_pos[:d_max] / sum_evals) * 100.0

# 3B. Varianza residual de distancias geodesicas (Criterio Tenenbaum 2000: 1 - r^2)
isomap_res_var_dist = []
for d in dims:
    iso = Isomap(n_neighbors=k_neighbors, n_components=d)
    Y_iso = iso.fit_transform(X_cent)
    D_Y_iso = squareform(pdist(Y_iso, metric='euclidean'))
    d_y_iso_flat = D_Y_iso[triu_idx]
    r_iso, _ = pearsonr(d_geo_flat, d_y_iso_flat)
    isomap_res_var_dist.append(1.0 - (r_iso ** 2))

isomap_res_var_dist = np.array(isomap_res_var_dist)

# Isomap con k=20 alternativo para confirmar invariancia de dimensionalidad
iso_res_k20 = []
adj_k20 = kneighbors_graph(X_cent, n_neighbors=20, mode='distance', include_self=False)
D_geo_k20 = shortest_path(adj_k20, directed=False)
d_geo_k20_flat = D_geo_k20[triu_idx]
for d in dims:
    iso = Isomap(n_neighbors=20, n_components=d)
    Y_iso20 = iso.fit_transform(X_cent)
    D_Y_iso20 = squareform(pdist(Y_iso20, metric='euclidean'))
    r_k20, _ = pearsonr(d_geo_k20_flat, D_Y_iso20[triu_idx])
    iso_res_k20.append(1.0 - (r_k20 ** 2))
iso_res_k20 = np.array(iso_res_k20)

# ==============================================================================
# 4. TABLA COMPARATIVA CUANTITATIVA
# ==============================================================================
print("\n" + "=" * 80)
print(f"{'Dim':^5} | {'PCA Var Ind (%)':^15} | {'PCA Cum (%)':^13} | {'Isomap Cum (%)':^15} | {'PCA Res Var':^12} | {'Isomap Res (k=' + str(k_neighbors) + ')':^15}")
print("=" * 80)
for i, d in enumerate(dims):
    print(f"{d:^5} | {pca_var_ind[i]:^15.2f} | {pca_var_cum[i]:^13.2f} | {isomap_var_cum[i]:^15.2f} | {pca_res_var_dist[i]:^12.4f} | {isomap_res_var_dist[i]:^15.4f}")
print("=" * 80)

# ==============================================================================
# 5. GENERACION DEL GRAFICO (3 PANELES)
# ==============================================================================
out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/analisis_dimensionalidad")
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, "varianza_vs_dimensiones_lucas_pca_isomap.png")

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')

# Panel A: Varianza Explicada Acumulada
ax = axs[0]
ax.plot(dims, pca_var_cum, marker='o', color='#1F77B4', linewidth=2.2, label='PCA - Covarianza Lineal')
ax.plot(dims, isomap_var_cum, marker='s', color='#2CA02C', linewidth=2.2, label=f'Isomap - Kernel Geodésico (k={k_neighbors})')
ax.axhline(80, color='#888888', linestyle='--', linewidth=1.0, alpha=0.7, label='Umbral 80% Varianza')
ax.axhline(90, color='#E63946', linestyle=':', linewidth=1.2, alpha=0.8, label='Umbral 90% Varianza')
ax.axvline(2, color='#E7A61A', linestyle='--', linewidth=1.5, alpha=0.8, label='Dimensión 2D')
ax.axvline(3, color='#9D4EDD', linestyle='--', linewidth=1.5, alpha=0.8, label='Dimensión 3D')

ax.set_title("Varianza Explicada Acumulada vs Dimensiones: Lucas", fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel("Número de Dimensiones", fontsize=11, fontweight='bold')
ax.set_ylabel("Varianza Explicada Acumulada (%)", fontsize=11, fontweight='bold')
ax.set_xticks(dims)
ax.set_ylim(0, 105)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)

# Panel B: Varianza Residual (Criterio de Tenenbaum et al. Science 2000)
ax = axs[1]
ax.plot(dims, pca_res_var_dist, marker='o', color='#1F77B4', linewidth=2.2, label='PCA - Distancias Euclidianas')
ax.plot(dims, isomap_res_var_dist, marker='s', color='#2CA02C', linewidth=2.2, label=f'Isomap - Distancias Geodésicas (k={k_neighbors})')
ax.plot(dims, iso_res_k20, marker='^', color='#9D4EDD', linewidth=1.8, linestyle='-.', label='Isomap - Distancias Geodésicas (k=20)')
ax.axvline(2, color='#E7A61A', linestyle='--', linewidth=1.5, alpha=0.8, label='Codo Variedad 2D')

ax.set_title("Varianza Residual vs Dimensiones: Criterio Tenenbaum", fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel("Número de Dimensiones", fontsize=11, fontweight='bold')
ax.set_ylabel("Varianza Residual (1 - r^2)", fontsize=11, fontweight='bold')
ax.set_xticks(dims)
ax.set_ylim(-0.02, max(pca_res_var_dist.max(), isomap_res_var_dist.max()) * 1.08)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

# Panel C: Scree Plot - Varianza Individual por Componente
ax = axs[2]
ax.bar(dims - 0.18, pca_var_ind, width=0.36, color='#1F77B4', alpha=0.85, label='PCA (Autovalores Lineales)')
ax.bar(dims + 0.18, isomap_var_ind, width=0.36, color='#2CA02C', alpha=0.85, label='Isomap (Autovalores Kernel)')

ax.set_title("Espectro de Autovalores: Scree Plot Individual", fontsize=12, fontweight='bold', pad=10)
ax.set_xlabel("Componente / Dimensión", fontsize=11, fontweight='bold')
ax.set_ylabel("Varianza Individual (%)", fontsize=11, fontweight='bold')
ax.set_xticks(dims)
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Grafico guardado exitosamente en:\n     {out_fig}")
