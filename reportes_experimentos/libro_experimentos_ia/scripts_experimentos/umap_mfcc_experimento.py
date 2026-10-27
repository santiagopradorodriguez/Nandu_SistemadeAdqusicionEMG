import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import extraer_mfcc_estricto_oficial as ext_mfcc

X_mfcc = ext_mfcc.X_mfcc_arr
Y = ext_mfcc.Y_arr
vocales = ['A', 'E', 'I', 'O', 'U']
mask_ou = np.isin(Y, ['O', 'U'])

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y) * 100

configs = [
    {"n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean"},
    {"n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"},
    {"n_neighbors": 30, "min_dist": 0.2, "metric": "euclidean"},
    {"n_neighbors": 30, "min_dist": 0.2, "metric": "cosine"},
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

print("Resultados UMAP 2D sobre los 48 MFCCs sEMG:")
print("-" * 75)
print(f"{'Configuración':<38} | {'K-Means (5v)':<14} | {'Sep. O vs U':<12}")
print("-" * 75)

for idx, cfg in enumerate(configs):
    reducer = umap.UMAP(
        n_neighbors=cfg['n_neighbors'],
        min_dist=cfg['min_dist'],
        metric=cfg['metric'],
        random_state=42
    )
    X_umap = reducer.fit_transform(X_mfcc)
    
    acc_km = eval_km(X_umap)
    lda_ou = LinearDiscriminantAnalysis().fit(X_umap[mask_ou], Y[mask_ou])
    acc_ou = lda_ou.score(X_umap[mask_ou], Y[mask_ou]) * 100
    
    desc = f"k={cfg['n_neighbors']}, min_d={cfg['min_dist']}, {cfg['metric']}"
    print(f"{desc:<38} | {acc_km:6.1f}%       | {acc_ou:6.1f}%")
    
    ax = axes[idx]
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_umap[m, 0], X_umap[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
        
    ax.set_title(f"UMAP 2D ({desc})\nK-Means (5v): {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontweight='bold', fontsize=11)
    ax.set_xlabel("UMAP 1", fontweight='bold')
    ax.set_ylabel("UMAP 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/umap_2d_mfcc_comparativa.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("-" * 75)
print("Panel comparativo UMAP 2D guardado en:", out_f)
