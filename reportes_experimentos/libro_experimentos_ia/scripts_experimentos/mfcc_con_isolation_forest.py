import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# Importamos la extracción idéntica del script anterior (174 pulsos crudos)
sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import comparativa_cruda_total_sin_filtros as base

muestras = base.muestras
vocales = ['A', 'E', 'I', 'O', 'U']
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

# Aplicar Isolation Forest por vocal (contamination=0.10, idéntico a generador_pca_umap)
X_clean_env = []
Y_clean_env = []

X_clean_mfcc = []
Y_clean_mfcc = []

descartes_mfcc = 0
descartes_env = 0

for v in vocales:
    m_v = [m for m in muestras if m['vocal'] == v]
    X_v_env = np.array([m['feat_env'] for m in m_v])
    X_v_mfcc = np.array([m['feat_mfcc'] for m in m_v])
    
    # Isolation Forest sobre Envolvente
    iso_env = IsolationForest(contamination=0.10, random_state=42)
    preds_env = iso_env.fit_predict(X_v_env)
    for i, p in enumerate(preds_env):
        if p == 1:
            X_clean_env.append(X_v_env[i])
            Y_clean_env.append(v)
        else:
            descartes_env += 1
            
    # Isolation Forest sobre MFCC
    iso_mfcc = IsolationForest(contamination=0.10, random_state=42)
    preds_mfcc = iso_mfcc.fit_predict(X_v_mfcc)
    for i, p in enumerate(preds_mfcc):
        if p == 1:
            X_clean_mfcc.append(X_v_mfcc[i])
            Y_clean_mfcc.append(v)
        else:
            descartes_mfcc += 1

X_clean_env = np.array(X_clean_env)
Y_clean_env = np.array(Y_clean_env)

X_clean_mfcc = np.array(X_clean_mfcc)
Y_clean_mfcc = np.array(Y_clean_mfcc)

print(f"Envolvente: {len(X_clean_env)} inliers (descartados {descartes_env})")
print(f"MFCC:       {len(X_clean_mfcc)} inliers (descartados {descartes_mfcc})")

pca_env = PCA(n_components=2, random_state=42).fit_transform(X_clean_env)
pca_mfcc = PCA(n_components=2, random_state=42).fit_transform(X_clean_mfcc)

def eval_km(X_p, Y_true):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y_true == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_true) * 100

mask_ou_env = np.isin(Y_clean_env, ['O', 'U'])
mask_ou_mfcc = np.isin(Y_clean_mfcc, ['O', 'U'])

lda_env_ou = LinearDiscriminantAnalysis().fit(pca_env[mask_ou_env], Y_clean_env[mask_ou_env])
lda_mfcc_ou = LinearDiscriminantAnalysis().fit(pca_mfcc[mask_ou_mfcc], Y_clean_mfcc[mask_ou_mfcc])

acc_km_env = eval_km(pca_env, Y_clean_env)
acc_km_mfcc = eval_km(pca_mfcc, Y_clean_mfcc)
acc_ou_env = lda_env_ou.score(pca_env[mask_ou_env], Y_clean_env[mask_ou_env]) * 100
acc_ou_mfcc = lda_mfcc_ou.score(pca_mfcc[mask_ou_mfcc], Y_clean_mfcc[mask_ou_mfcc]) * 100

print(f"Envolvente (con Isolation Forest 10%) -> K-Means: {acc_km_env:.1f}% | Sep. O vs U: {acc_ou_env:.1f}%")
print(f"MFCC sEMG  (con Isolation Forest 10%) -> K-Means: {acc_km_mfcc:.1f}% | Sep. O vs U: {acc_ou_mfcc:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, X_p, Y_p, tit, acc_km, acc_ou in [
    (axes[0], pca_env, Y_clean_env, "1. Envolvente sEMG (Isolation Forest 10%)", acc_km_env, acc_ou_env),
    (axes[1], pca_mfcc, Y_clean_mfcc, "2. MFCC sEMG puro (Isolation Forest 10%)", acc_km_mfcc, acc_ou_mfcc)
]:
    for v in vocales:
        m = (Y_p == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={np.sum(m)})", s=45, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means (5v): {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontweight='bold')
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_con_isolation_forest.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
