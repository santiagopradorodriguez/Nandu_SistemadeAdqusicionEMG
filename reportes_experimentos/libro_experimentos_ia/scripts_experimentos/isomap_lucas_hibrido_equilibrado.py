import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# Importamos los datos limpios y corregidos del script anterior
import comparar_triangulo_vs_mfcc_corregido as base

X_env = base.X_env_clean        # Envolvente oficial triangulo (492, 300)
X_mfcc = base.X_mfcc_crudo      # Mel Muscular desde senal cruda AC (492, 48)
X_hib = base.X_hib              # Hibrido Equilibrado (492, 348)
Y = base.Y_clean
vocales = ['A', 'E', 'I', 'O', 'U']
mask_ou = np.isin(Y, ['O', 'U'])

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y) * 100

# 1. ISOMAP 2D
iso2d_env = Isomap(n_neighbors=15, n_components=2).fit_transform(X_env)
iso2d_mfcc = Isomap(n_neighbors=15, n_components=2).fit_transform(X_mfcc)
iso2d_hib = Isomap(n_neighbors=15, n_components=2).fit_transform(X_hib)

# 2. ISOMAP 3D
iso3d_hib = Isomap(n_neighbors=15, n_components=3).fit_transform(X_hib)

# Metricas 2D
km_iso_env = eval_km(iso2d_env)
km_iso_mfcc = eval_km(iso2d_mfcc)
km_iso_hib = eval_km(iso2d_hib)

lda_ou_iso_env = LinearDiscriminantAnalysis().fit(iso2d_env[mask_ou], Y[mask_ou]).score(iso2d_env[mask_ou], Y[mask_ou]) * 100
lda_ou_iso_mfcc = LinearDiscriminantAnalysis().fit(iso2d_mfcc[mask_ou], Y[mask_ou]).score(iso2d_mfcc[mask_ou], Y[mask_ou]) * 100
lda_ou_iso_hib = LinearDiscriminantAnalysis().fit(iso2d_hib[mask_ou], Y[mask_ou]).score(iso2d_hib[mask_ou], Y[mask_ou]) * 100

# Metricas 3D Hibrido
km_iso3d_hib = eval_km(iso3d_hib)
lda_ou_iso3d_hib = LinearDiscriminantAnalysis().fit(iso3d_hib[mask_ou], Y[mask_ou]).score(iso3d_hib[mask_ou], Y[mask_ou]) * 100
lda_5v_iso3d_hib = LinearDiscriminantAnalysis().fit(iso3d_hib, Y).score(iso3d_hib, Y) * 100

print("Resultados ISOMAP sobre Lucas 2026-07-10 (Señal AC Corregida):")
print("-" * 80)
print(f"Isomap 2D - Envolvente Pura (180D)   | K-Means: {km_iso_env:5.1f}% | O vs U: {lda_ou_iso_env:5.1f}%")
print(f"Isomap 2D - Mel Muscular AC (48D)    | K-Means: {km_iso_mfcc:5.1f}% | O vs U: {lda_ou_iso_mfcc:5.1f}%")
print(f"Isomap 2D - Híbrido Equilibrado      | K-Means: {km_iso_hib:5.1f}% | O vs U: {lda_ou_iso_hib:5.1f}%")
print(f"Isomap 3D - Híbrido Equilibrado      | K-Means: {km_iso3d_hib:5.1f}% | O vs U: {lda_ou_iso3d_hib:5.1f}% | LDA 5v: {lda_5v_iso3d_hib:5.1f}%")
print("-" * 80)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig = plt.figure(figsize=(19, 6))

# Panel 1: Isomap 2D Envolvente
ax1 = fig.add_subplot(131)
for v in vocales:
    m = (Y == v)
    ax1.scatter(iso2d_env[m, 0], iso2d_env[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
ax1.set_title(f"1. Isomap 2D - Envolvente Oficial\nK-Means: {km_iso_env:.1f}% | Sep. O vs U: {lda_ou_iso_env:.1f}%", fontweight='bold', fontsize=11)
ax1.set_xlabel("Iso 1", fontweight='bold')
ax1.set_ylabel("Iso 2", fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='best', frameon=True)

# Panel 2: Isomap 2D Hibrido Equilibrado
ax2 = fig.add_subplot(132)
for v in vocales:
    m = (Y == v)
    ax2.scatter(iso2d_hib[m, 0], iso2d_hib[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
ax2.set_title(f"2. Isomap 2D - Híbrido Equilibrado (Tiempo + Mel AC)\nK-Means: {km_iso_hib:.1f}% | Sep. O vs U: {lda_ou_iso_hib:.1f}%", fontweight='bold', fontsize=11)
ax2.set_xlabel("Iso 1", fontweight='bold')
ax2.set_ylabel("Iso 2", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='best', frameon=True)

# Panel 3: Isomap 3D Hibrido Equilibrado
ax3 = fig.add_subplot(133, projection='3d')
for v in vocales:
    m = (Y == v)
    ax3.scatter(iso3d_hib[m, 0], iso3d_hib[m, 1], iso3d_hib[m, 2], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
ax3.set_title(f"3. Isomap 3D - Híbrido Equilibrado\nK-Means: {km_iso3d_hib:.1f}% | Sep. O vs U: {lda_ou_iso3d_hib:.1f}%", fontweight='bold', fontsize=11)
ax3.set_xlabel("Iso 1", fontweight='bold')
ax3.set_ylabel("Iso 2", fontweight='bold')
ax3.set_zlabel("Iso 3", fontweight='bold')
ax3.view_init(elev=20, azim=130)
ax3.grid(True, linestyle='--', alpha=0.5)
ax3.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lucas_isomap_hibrido_equilibrado.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
