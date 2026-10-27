import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import experimento_mfcc_pca_2026_10_23 as exp

COLOR_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

muestras = exp.extraer_datos_triada()
Y = np.array([m['vocal'] for m in muestras])
X_env = np.array([m['feat_env'] for m in muestras])
X_mfcc = np.array([m['feat_mfcc_emg'] for m in muestras])
vocales = ['A', 'E', 'I', 'O', 'U']

mask_ou = np.isin(Y, ['O', 'U'])
Y_ou = Y[mask_ou]

def get_kmeans_acc(X_proj, Y_true):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_proj)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y_true == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_true) * 100

# 1. PCA 2D Envolvente (sin scaler)
pca_env = PCA(n_components=2, random_state=42)
X_pca_env = pca_env.fit_transform(X_env)
acc_km_pca_env = get_kmeans_acc(X_pca_env, Y)
lda_pca_env_ou = LinearDiscriminantAnalysis().fit(X_pca_env[mask_ou], Y_ou)
acc_lda_pca_env_ou = lda_pca_env_ou.score(X_pca_env[mask_ou], Y_ou) * 100

# 2. Isomap 2D Envolvente (k=12, sin scaler)
iso_env = Isomap(n_neighbors=12, n_components=2)
X_iso_env = iso_env.fit_transform(X_env)
acc_km_iso_env = get_kmeans_acc(X_iso_env, Y)
lda_iso_env_ou = LinearDiscriminantAnalysis().fit(X_iso_env[mask_ou], Y_ou)
acc_lda_iso_env_ou = lda_iso_env_ou.score(X_iso_env[mask_ou], Y_ou) * 100

# 3. PCA 2D MFCC sEMG (sin scaler)
pca_mfcc = PCA(n_components=2, random_state=42)
X_pca_mfcc = pca_mfcc.fit_transform(X_mfcc)
acc_km_pca_mfcc = get_kmeans_acc(X_pca_mfcc, Y)
lda_pca_mfcc_ou = LinearDiscriminantAnalysis().fit(X_pca_mfcc[mask_ou], Y_ou)
acc_lda_pca_mfcc_ou = lda_pca_mfcc_ou.score(X_pca_mfcc[mask_ou], Y_ou) * 100

# 4. Isomap 2D MFCC sEMG (k=12, sin scaler)
iso_mfcc = Isomap(n_neighbors=12, n_components=2)
X_iso_mfcc = iso_mfcc.fit_transform(X_mfcc)
acc_km_iso_mfcc = get_kmeans_acc(X_iso_mfcc, Y)
lda_iso_mfcc_ou = LinearDiscriminantAnalysis().fit(X_iso_mfcc[mask_ou], Y_ou)
acc_lda_iso_mfcc_ou = lda_iso_mfcc_ou.score(X_iso_mfcc[mask_ou], Y_ou) * 100

fig, axes = plt.subplots(2, 2, figsize=(15, 13))

paneles = [
    (axes[0, 0], X_pca_env, "1. PCA 2D - Envolvente sEMG", acc_km_pca_env, acc_lda_pca_env_ou, "PC 1", "PC 2"),
    (axes[0, 1], X_iso_env, "2. Isomap 2D - Envolvente sEMG (k=12)", acc_km_iso_env, acc_lda_iso_env_ou, "Iso 1", "Iso 2"),
    (axes[1, 0], X_pca_mfcc, "3. PCA 2D - MFCC sEMG puro", acc_km_pca_mfcc, acc_lda_pca_mfcc_ou, "PC 1", "PC 2"),
    (axes[1, 1], X_iso_mfcc, "4. Isomap 2D - MFCC sEMG puro (k=12)", acc_km_iso_mfcc, acc_lda_iso_mfcc_ou, "Iso 1", "Iso 2"),
]

for ax, X_p, titulo, acc_km, acc_ou, xl, yl in paneles:
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{titulo}\nK-Means (5v): {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontsize=11, fontweight='bold')
    ax.set_xlabel(xl, fontweight='bold')
    ax.set_ylabel(yl, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True, edgecolor='black', facecolor='white')

plt.tight_layout()
out_fig = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_2d_sin_scaler.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print("Guardado en:", out_fig)
