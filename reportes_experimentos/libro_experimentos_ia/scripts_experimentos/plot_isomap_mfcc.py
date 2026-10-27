import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

mask_ou = np.isin(Y, ['O', 'U'])
Y_ou = Y[mask_ou]

# 1. Isomap 3D (k=12)
iso = Isomap(n_neighbors=12, n_components=3)
X_iso = iso.fit_transform(X_env)
lda_iso = LinearDiscriminantAnalysis().fit(X_iso[mask_ou], Y_ou)
acc_iso_ou = lda_iso.score(X_iso[mask_ou], Y_ou) * 100

# 2. PCA 3D MFCC (sin scaler)
pca_mfcc = PCA(n_components=3, random_state=42)
X_p_mfcc = pca_mfcc.fit_transform(X_mfcc)
lda_mfcc = LinearDiscriminantAnalysis().fit(X_p_mfcc[mask_ou], Y_ou)
acc_mfcc_ou = lda_mfcc.score(X_p_mfcc[mask_ou], Y_ou) * 100

fig = plt.figure(figsize=(16, 7))

# Subplot 1: Isomap 3D
ax1 = fig.add_subplot(121, projection='3d')
for v in ['A', 'E', 'I', 'O', 'U']:
    m = (Y == v)
    ax1.scatter(X_iso[m, 0], X_iso[m, 1], X_iso[m, 2], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=45, alpha=0.8, edgecolors='black', linewidths=0.5)
ax1.set_title(f"Isomap 3D - Envolventes sEMG (k=12)\nSeparabilidad lineal /o/ vs /u/: {acc_iso_ou:.1f}%", fontweight='bold')
ax1.set_xlabel("Iso 1")
ax1.set_ylabel("Iso 2")
ax1.set_zlabel("Iso 3")
ax1.view_init(elev=20, azim=120)
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: PCA 3D MFCC
ax2 = fig.add_subplot(122, projection='3d')
for v in ['A', 'E', 'I', 'O', 'U']:
    m = (Y == v)
    ax2.scatter(X_p_mfcc[m, 0], X_p_mfcc[m, 1], X_p_mfcc[m, 2], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=45, alpha=0.8, edgecolors='black', linewidths=0.5)
ax2.set_title(f"PCA 3D - MFCC sEMG puro (sin scaler)\nSeparabilidad lineal /o/ vs /u/: {acc_mfcc_ou:.1f}%", fontweight='bold')
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.set_zlabel("PC 3")
ax2.view_init(elev=20, azim=45)
ax2.legend(loc='upper right')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_fig = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_isomap3d_vs_pca3d_mfcc.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print("Guardado en:", out_fig)
