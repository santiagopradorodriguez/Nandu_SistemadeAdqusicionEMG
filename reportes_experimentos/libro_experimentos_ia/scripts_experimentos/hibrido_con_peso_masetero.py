import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

import hibrido_envolvente_mel_equilibrado as base

V_T = base.V_T.copy() # (174, 180) -> 60 pts por canal
V_F = base.V_F.copy() # (174, 48)  -> 16 features por canal (8 medias + 8 desvíos)
Y = base.Y
vocales_list = ['A', 'E', 'I', 'O', 'U']
mask_ou = np.isin(Y, ['O', 'U'])

# Aplicar el peso canónico oficial de la sesión: 2.5x al Masetero (Ch0)
# En tiempo:
V_T_w = V_T.copy()
V_T_w[:, 0:60] *= 2.5

# En frecuencia:
V_F_w = V_F.copy()
V_F_w[:, 0:16] *= 2.5

# Equilibrado de bloque
s2_T = np.mean(np.var(V_T_w, axis=0))
s2_F = np.mean(np.var(V_F_w, axis=0))
gamma_w = np.sqrt(s2_T / (s2_F + 1e-9))

V_F_eq_w = V_F_w * gamma_w

variantes_w = {
    "1. Envolvente (Masetero 2.5x)": V_T_w,
    "2. Mel Muscular (Masetero 2.5x)": V_F_eq_w,
    "3. Híbrido Equilibrado (Tiempo + Mel 2.5x)": np.hstack([V_T_w, V_F_eq_w]),
    "4. Híbrido Ponderado (Tiempo + 0.6*Mel 2.5x)": np.hstack([V_T_w, 0.6 * V_F_eq_w])
}

print("Resultados Híbrido con Boost Masetero 2.5x:")
print("-" * 80)
print(f"{'Variante':<45} | {'K-Means':<10} | {'LDA (5v)':<10} | {'O vs U':<8}")
print("-" * 80)

for nombre, X_mat in variantes_w.items():
    pca = PCA(n_components=2, random_state=42)
    X_p = pca.fit_transform(X_mat)
    
    # K-Means
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales_list):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales_list[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    acc_km = np.mean(preds == Y) * 100
    
    lda_5v = LinearDiscriminantAnalysis().fit(X_p, Y).score(X_p, Y) * 100
    lda_ou = LinearDiscriminantAnalysis().fit(X_p[mask_ou], Y[mask_ou]).score(X_p[mask_ou], Y[mask_ou]) * 100
    print(f"{nombre:<45} | {acc_km:6.1f}%    | {lda_5v:6.1f}%    | {lda_ou:6.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (nombre, X_mat) in enumerate(variantes_w.items()):
    pca = PCA(n_components=2, random_state=42)
    X_p = pca.fit_transform(X_mat)
    var = np.sum(pca.explained_variance_ratio_) * 100
    
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales_list):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales_list[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    acc_km = np.mean(preds == Y) * 100
    lda_ou = LinearDiscriminantAnalysis().fit(X_p[mask_ou], Y[mask_ou]).score(X_p[mask_ou], Y[mask_ou]) * 100
    
    ax = axes[idx]
    for v in vocales_list:
        m = (Y == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{nombre}\nK-Means: {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {lda_ou:.1f}% (Var: {var:.1f}%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/hibrido_con_boost_masetero.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
