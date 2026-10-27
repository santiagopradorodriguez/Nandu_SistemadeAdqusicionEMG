import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import extraer_mfcc_estricto_oficial as ext_mfcc

# Cargar MFCCs sin pesos (16 features por canal: 8 medias + 8 desvíos)
X_mfcc = ext_mfcc.X_mfcc_arr.copy()
Y_mfcc = ext_mfcc.Y_arr
vocales = ['A', 'E', 'I', 'O', 'U']

# Caso A: MFCC sin pesos (pesos = [1.0, 1.0, 1.0])
pca_unweighted = PCA(n_components=2).fit_transform(X_mfcc)

# Caso B: MFCC CON PESOS [2.5, 1.0, 1.0] (2.5x al Masetero Ch0)
# Cada canal tiene 16 features (Ch0: 0..16, Ch1: 16..32, Ch2: 32..48)
X_mfcc_weighted = X_mfcc.copy()
X_mfcc_weighted[:, 0:16] *= 2.5 # Masetero x2.5

pca_weighted = PCA(n_components=2).fit_transform(X_mfcc_weighted)

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y_mfcc == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_mfcc) * 100

mask_ou = np.isin(Y_mfcc, ['O', 'U'])
lda_unw_ou = LinearDiscriminantAnalysis().fit(pca_unweighted[mask_ou], Y_mfcc[mask_ou])
lda_w_ou = LinearDiscriminantAnalysis().fit(pca_weighted[mask_ou], Y_mfcc[mask_ou])

acc_km_unw = eval_km(pca_unweighted)
acc_km_w = eval_km(pca_weighted)
acc_ou_unw = lda_unw_ou.score(pca_unweighted[mask_ou], Y_mfcc[mask_ou]) * 100
acc_ou_w = lda_w_ou.score(pca_weighted[mask_ou], Y_mfcc[mask_ou]) * 100

print(f"MFCC SIN pesos [1, 1, 1]   -> K-Means: {acc_km_unw:.1f}% | Sep. O vs U: {acc_ou_unw:.1f}%")
print(f"MFCC CON pesos [2.5, 1, 1] -> K-Means: {acc_km_w:.1f}% | Sep. O vs U: {acc_ou_w:.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, X_p, tit, acc_km, acc_ou in [
    (axes[0], pca_unweighted, "MFCC sEMG - Pesos [1.0, 1.0, 1.0]", acc_km_unw, acc_ou_unw),
    (axes[1], pca_weighted, "MFCC sEMG - Pesos [2.5, 1.0, 1.0] (Masetero x2.5)", acc_km_w, acc_ou_w)
]:
    for v in vocales:
        m = (Y_mfcc == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means: {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontweight='bold')
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_mfcc_con_vs_sin_pesos.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
