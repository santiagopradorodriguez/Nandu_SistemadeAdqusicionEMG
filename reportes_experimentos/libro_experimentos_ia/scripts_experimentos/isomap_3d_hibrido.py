import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import evaluar_features_clasicas_emg as base

Y = base.Y
mask_ou = base.mask_ou
vocales = ['A', 'E', 'I', 'O', 'U']

V_env = base.V_env
V_mel = base.V_mel * base.gamma_mel
X_hib = np.hstack([V_env, V_mel])

# Isomap 3D sobre Hibrido (Tiempo + Mel EMG)
iso3d = Isomap(n_neighbors=12, n_components=3)
X_iso3d = iso3d.fit_transform(X_hib)

# K-Means 5v
km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_iso3d)
cost = np.zeros((5, 5))
for i, v in enumerate(vocales):
    for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
r, c = linear_sum_assignment(cost)
map_c = {c[i]: vocales[r[i]] for i in range(5)}
preds = np.array([map_c[cl] for cl in km.labels_])
acc_km = np.mean(preds == Y) * 100

lda_5v = LinearDiscriminantAnalysis().fit(X_iso3d, Y).score(X_iso3d, Y) * 100
lda_ou = LinearDiscriminantAnalysis().fit(X_iso3d[mask_ou], Y[mask_ou]).score(X_iso3d[mask_ou], Y[mask_ou]) * 100

print(f"Isomap 3D Híbrido -> K-Means (5v): {acc_km:.1f}% | LDA (5v): {lda_5v:.1f}% | O vs U (LDA): {lda_ou:.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for v in vocales:
    m = (Y == v)
    ax.scatter(X_iso3d[m, 0], X_iso3d[m, 1], X_iso3d[m, 2], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

ax.set_title(f"Isomap 3D - Vector Híbrido (Tiempo + Mel Muscular, k=12)\nK-Means (5v): {acc_km:.1f}% | LDA Global: {lda_5v:.1f}% | Sep. Lineal /o/ vs /u/: {lda_ou:.1f}%", fontweight='bold')
ax.set_xlabel("Iso 1", fontweight='bold')
ax.set_ylabel("Iso 2", fontweight='bold')
ax.set_zlabel("Iso 3", fontweight='bold')
ax.view_init(elev=25, azim=125)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/isomap_3d_hibrido.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
