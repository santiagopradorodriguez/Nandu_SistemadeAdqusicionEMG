import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import evaluar_features_clasicas_emg as base

Y = base.Y
mask_ou = base.mask_ou
vocales = ['A', 'E', 'I', 'O', 'U']

# Tomamos las 4 matrices principales
V_env = base.V_env                   # Envolvente pura (180D)
V_mel = base.V_mel * base.gamma_mel  # Mel muscular equilibrada (48D)
X_hib = np.hstack([V_env, V_mel])    # Hibrido Tiempo + Mel (228D)
X_combo = np.hstack([V_env, V_mel, base.V_ar * base.gamma_ar, base.V_hudgins * base.gamma_hud]) # Combo (255D)

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y) * 100

variantes = {
    "1. Isomap 2D - Envolvente (180D)": V_env,
    "2. Isomap 2D - Mel Muscular (48D)": V_mel,
    "3. Isomap 2D - Híbrido Tiempo + Mel (228D)": X_hib,
    "4. Isomap 2D - Combo Completo (255D)": X_combo
}

print("Resultados de ISOMAP 2D (k=12, Geodésico, Sin Scaler):")
print("-" * 80)
print(f"{'Variante':<45} | {'K-Means':<10} | {'LDA (5v)':<10} | {'O vs U':<8}")
print("-" * 80)

proyecciones = {}
for nombre, X_mat in variantes.items():
    iso = Isomap(n_neighbors=12, n_components=2)
    X_iso = iso.fit_transform(X_mat)
    proyecciones[nombre] = X_iso
    
    acc_km = eval_km(X_iso)
    lda_5v = LinearDiscriminantAnalysis().fit(X_iso, Y).score(X_iso, Y) * 100
    lda_ou = LinearDiscriminantAnalysis().fit(X_iso[mask_ou], Y[mask_ou]).score(X_iso[mask_ou], Y[mask_ou]) * 100
    print(f"{nombre:<45} | {acc_km:6.1f}%    | {lda_5v:6.1f}%    | {lda_ou:6.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (nombre, X_iso) in enumerate(proyecciones.items()):
    ax = axes[idx]
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_iso[m, 0], X_iso[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
    
    acc_km = eval_km(X_iso)
    lda_ou = LinearDiscriminantAnalysis().fit(X_iso[mask_ou], Y[mask_ou]).score(X_iso[mask_ou], Y[mask_ou]) * 100
    ax.set_title(f"{nombre}\nK-Means (5v): {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {lda_ou:.1f}%", fontweight='bold', fontsize=11)
    ax.set_xlabel("Iso 1", fontweight='bold')
    ax.set_ylabel("Iso 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/isomap_2d_hibrido_comparativa.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("-" * 80)
print("Figura guardada en:", out_f)
