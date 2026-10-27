import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, n_components=3, metric='cosine', random_state=42)
X_umap_3d = reducer.fit_transform(X_mfcc)

# LDA y K-Means en 3D
km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_umap_3d)
cost = np.zeros((5, 5))
for i, v in enumerate(vocales):
    for j in range(5):
        cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
r, c = linear_sum_assignment(cost)
map_c = {c[i]: vocales[r[i]] for i in range(5)}
preds = np.array([map_c[cl] for cl in km.labels_])
acc_km = np.mean(preds == Y) * 100

lda_ou = LinearDiscriminantAnalysis().fit(X_umap_3d[mask_ou], Y[mask_ou])
acc_ou = lda_ou.score(X_umap_3d[mask_ou], Y[mask_ou]) * 100
lda_5v = LinearDiscriminantAnalysis().fit(X_umap_3d, Y).score(X_umap_3d, Y) * 100

print(f"UMAP 3D (cosine, k=20) -> K-Means (5v): {acc_km:.1f}% | LDA (5v): {lda_5v:.1f}% | O vs U (LDA): {acc_ou:.1f}%")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for v in vocales:
    m = (Y == v)
    ax.scatter(X_umap_3d[m, 0], X_umap_3d[m, 1], X_umap_3d[m, 2], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

ax.set_title(f"UMAP 3D sobre MFCCs sEMG (Métrica Coseno, k=20)\nK-Means (5v): {acc_km:.1f}% | LDA Global: {lda_5v:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontweight='bold')
ax.set_xlabel("UMAP 1", fontweight='bold')
ax.set_ylabel("UMAP 2", fontweight='bold')
ax.set_zlabel("UMAP 3", fontweight='bold')
ax.view_init(elev=25, azim=130)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/umap_3d_mfcc_cosine.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
