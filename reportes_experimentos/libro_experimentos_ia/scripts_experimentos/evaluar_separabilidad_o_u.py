#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Evaluacion Cuantitativa de Separabilidad del Par /o/ - /u/
# Objetivo: Medir distancia de centroides, d-prime (Fisher), silueta binaria
#           y exactitud del par /o/ vs /u/ en PCA e Isomap (2D y 3D) para distintos k.
# ==============================================================================

import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import silhouette_score, accuracy_score
from scipy.spatial.distance import cdist

np.random.seed(42)

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Cargar dataset
npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana5050/dataset_lucas_ventana5050.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana4060/dataset_lucas_ventana4060.npz")

data = np.load(npz_path, allow_pickle=True)
X_raw = data['X_clean'] if 'X_clean' in data else data['X']
Y_raw = data['Y_clean'] if 'Y_clean' in data else data['Y']

N = len(X_raw)
X = X_raw.reshape(N, -1).astype(np.float64)
Y = np.array([str(v).upper().strip() for v in Y_raw])

mask_ou = np.isin(Y, ['O', 'U'])
X_ou = X[mask_ou]
Y_ou = Y[mask_ou]
y_bin = (Y_ou == 'U').astype(int)

def metricas_par_ou(Z_total, Y_total):
    mask = np.isin(Y_total, ['O', 'U'])
    Z_sub = Z_total[mask]
    y_sub = Y_total[mask]
    
    Z_o = Z_sub[y_sub == 'O']
    Z_u = Z_sub[y_sub == 'U']
    
    c_o = np.mean(Z_o, axis=0)
    c_u = np.mean(Z_u, axis=0)
    
    # 1. Distancia euclidiana entre centroides
    dist_cent = np.linalg.norm(c_o - c_u)
    
    # 2. Dispersion interna (varianza media)
    var_o = np.mean(np.sum((Z_o - c_o)**2, axis=1))
    var_u = np.mean(np.sum((Z_u - c_u)**2, axis=1))
    sigma_pooled = np.sqrt(0.5 * (var_o + var_u))
    
    # 3. Ratio d-prime de Fisher
    d_prime = dist_cent / (sigma_pooled + 1e-9)
    
    # 4. Silueta binaria del par
    labels_bin = (y_sub == 'U').astype(int)
    sil = silhouette_score(Z_sub, labels_bin)
    
    # 5. Exactitud LDA lineal directa entre /o/ y /u/
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_sub, labels_bin)
    acc_lda = accuracy_score(labels_bin, lda.predict(Z_sub)) * 100.0
    
    return dist_cent, d_prime, sil, acc_lda

# Espacio original (D = 60)
d_c_orig, dp_orig, sil_orig, acc_orig = metricas_par_ou(X, Y)

# PCA
pca_2d = PCA(n_components=2, random_state=42).fit_transform(X)
d_c_pca2, dp_pca2, sil_pca2, acc_pca2 = metricas_par_ou(pca_2d, Y)

pca_3d = PCA(n_components=3, random_state=42).fit_transform(X)
d_c_pca3, dp_pca3, sil_pca3, acc_pca3 = metricas_par_ou(pca_3d, Y)

# Isomap en 2D y 3D
k_vals = [6, 12, 20, 35, 60]

print("=" * 95)
print(f"{'Método':^16} | {'Dim':^5} | {'k':^5} | {'Dist Centroides':^16} | {'d-prime (Fisher)':^16} | {'Silueta (/o/, /u/)':^18} | {'Exactitud LDA':^13}")
print("=" * 95)

print(f"{'Espacio Crudo':^16} | {'60D':^5} | {'N/A':^5} | {d_c_orig:^16.2f} | {dp_orig:^16.2f} | {sil_orig:^18.3f} | {acc_orig:^12.1f}%")
print(f"{'PCA':^16} | {'2D':^5} | {'N/A':^5} | {d_c_pca2:^16.2f} | {dp_pca2:^16.2f} | {sil_pca2:^18.3f} | {acc_pca2:^12.1f}%")
print(f"{'PCA':^16} | {'3D':^5} | {'N/A':^5} | {d_c_pca3:^16.2f} | {dp_pca3:^16.2f} | {sil_pca3:^18.3f} | {acc_pca3:^12.1f}%")
print("-" * 95)

for dim in [2, 3]:
    for k in k_vals:
        iso = Isomap(n_neighbors=k, n_components=dim)
        Z_iso = iso.fit_transform(X)
        dc, dp, sil, acc = metricas_par_ou(Z_iso, Y)
        print(f"{'Isomap':^16} | {str(dim) + 'D':^5} | {k:^5} | {dc:^16.2f} | {dp:^16.2f} | {sil:^18.3f} | {acc:^12.1f}%")
    if dim == 2:
        print("-" * 95)
print("=" * 95)
