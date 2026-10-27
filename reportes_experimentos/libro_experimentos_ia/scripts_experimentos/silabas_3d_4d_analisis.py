import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import analisis_silabas_2026_10_23 as base

X = base.X     # (57, 180) - envolventes tricanal
Y = base.Y     # 9 silabas

# Evaluacion de dimension latente sobre las 9 clases
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("Exactitud LDA (9 Sílabas) según la Dimensión Latente del PCA:")
print("-" * 75)
print(f"{'Espacio Latente':<25} | {'Var Explicada':<15} | {'LDA (Ajuste)':<14} | {'LDA (CV-3)':<12}")
print("-" * 75)

for d in [2, 3, 4, 5, 8, 12]:
    pca = PCA(n_components=d, random_state=42)
    X_p = pca.fit_transform(X)
    var = np.sum(pca.explained_variance_ratio_) * 100
    
    lda_fit = LinearDiscriminantAnalysis().fit(X_p, Y).score(X_p, Y) * 100
    sc = []
    for tr, ts in cv.split(X_p, Y):
        clf = LinearDiscriminantAnalysis().fit(X_p[tr], Y[tr])
        sc.append(clf.score(X_p[ts], Y[ts]))
    print(f"PCA {d:2d}D                   | {var:5.1f}%          | {lda_fit:5.1f}%         | {np.mean(sc)*100:5.1f}% (+/-{np.std(sc)*100:4.1f}%)")

# Espacio original 180D
sc_orig = []
for tr, ts in cv.split(X, Y):
    clf = LinearDiscriminantAnalysis().fit(X[tr], Y[tr])
    sc_orig.append(clf.score(X[ts], Y[ts]))
print(f"Espacio Completo 180D     | 100.0%          | 100.0%         | {np.mean(sc_orig)*100:5.1f}% (+/-{np.std(sc_orig)*100:4.1f}%)")

# Ahora con Isomap
print("\nExactitud Isomap (k=8) según dimensión latente:")
for d in [2, 3, 4]:
    iso = Isomap(n_neighbors=8, n_components=d)
    X_iso = iso.fit_transform(X)
    lda_fit = LinearDiscriminantAnalysis().fit(X_iso, Y).score(X_iso, Y) * 100
    sc = []
    for tr, ts in cv.split(X_iso, Y):
        clf = LinearDiscriminantAnalysis().fit(X_iso[tr], Y[tr])
        sc.append(clf.score(X_iso[ts], Y[ts]))
    print(f"Isomap {d}D               | -               | {lda_fit:5.1f}%         | {np.mean(sc)*100:5.1f}% (+/-{np.std(sc)*100:4.1f}%)")

# Graficar PCA 3D y las proyecciones cruzadas 4D
pca4 = PCA(n_components=4, random_state=42)
X_4d = pca4.fit_transform(X)
var4 = pca4.explained_variance_ratio_ * 100

silabas_unicas = np.unique(Y)
cmap = plt.get_cmap('tab10')

fig = plt.figure(figsize=(18, 6))

# Panel 1: PCA 3D
ax1 = fig.add_subplot(131, projection='3d')
for idx, s in enumerate(silabas_unicas):
    m = (Y == s)
    ax1.scatter(X_4d[m, 0], X_4d[m, 1], X_4d[m, 2], label=f"/{s.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5, color=cmap(idx % 10))
ax1.set_title(f"PCA 3D Sílabas (PC1-PC2-PC3)\nVar: {var4[0]+var4[1]+var4[2]:.1f}%", fontweight='bold')
ax1.set_xlabel(f"PC 1 ({var4[0]:.1f}%)")
ax1.set_ylabel(f"PC 2 ({var4[1]:.1f}%)")
ax1.set_zlabel(f"PC 3 ({var4[2]:.1f}%)")
ax1.view_init(elev=20, azim=130)
ax1.legend(loc='upper right', fontsize=8)

# Panel 2: PCA 4D plano PC1 vs PC4
ax2 = fig.add_subplot(132)
for idx, s in enumerate(silabas_unicas):
    m = (Y == s)
    ax2.scatter(X_4d[m, 0], X_4d[m, 3], label=f"/{s.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5, color=cmap(idx % 10))
ax2.set_title(f"PCA 4D: Proyección PC 1 vs PC 4\nPC4 ({var4[3]:.1f}%) desenreda consonantes", fontweight='bold')
ax2.set_xlabel(f"PC 1 ({var4[0]:.1f}%)")
ax2.set_ylabel(f"PC 4 ({var4[3]:.1f}%)")
ax2.grid(True, linestyle='--', alpha=0.5)

# Panel 3: PCA 4D plano PC2 vs PC3
ax3 = fig.add_subplot(133)
for idx, s in enumerate(silabas_unicas):
    m = (Y == s)
    ax3.scatter(X_4d[m, 1], X_4d[m, 2], label=f"/{s.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5, color=cmap(idx % 10))
ax3.set_title(f"PCA 4D: Proyección PC 2 vs PC 3\nModiolo vs Orbicular dinámico", fontweight='bold')
ax3.set_xlabel(f"PC 2 ({var4[1]:.1f}%)")
ax3.set_ylabel(f"PC 3 ({var4[2]:.1f}%)")
ax3.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_3d_4d_silabas.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
