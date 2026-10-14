#!/usr/bin/env python3
# ==============================================================================
# Detalle exacto de clasificacion /o/ frente a /u/
# 1) Evaluacion global de 5 clases (Confusion especifica O -> U y U -> O)
# 2) Clasificacion binaria directa del par O vs U (LDA y GMM K=2)
# ==============================================================================

import os
import sys
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

np.random.seed(42)

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana5050/dataset_lucas_ventana5050.npz")
data = np.load(npz_path, allow_pickle=True)
X_raw = data['X_clean'] if 'X_clean' in data else data['X']
Y_raw = data['Y_clean'] if 'Y_clean' in data else data['Y']

N = len(X_raw)
X = X_raw.reshape(N, -1).astype(np.float64)
Y = np.array([str(v).upper().strip() for v in Y_raw])

vocales = ['A', 'E', 'I', 'O', 'U']
vocal_to_idx = {v: i for i, v in enumerate(vocales)}
y_true_5 = np.array([vocal_to_idx[v] for v in Y])

idx_o = vocal_to_idx['O']
idx_u = vocal_to_idx['U']

n_o_total = np.sum(Y == 'O')
n_u_total = np.sum(Y == 'U')

print(f"Total muestras: /o/ = {n_o_total}, /u/ = {n_u_total} (Total par: {n_o_total + n_u_total})\n")

configs = [
    ("PCA 2D", PCA(n_components=2, random_state=42).fit_transform(X)),
    ("PCA 3D", PCA(n_components=3, random_state=42).fit_transform(X)),
    ("Isomap 2D (k=12)", Isomap(n_neighbors=12, n_components=2).fit_transform(X)),
    ("Isomap 2D (k=35)", Isomap(n_neighbors=35, n_components=2).fit_transform(X)),
    ("Isomap 3D (k=12)", Isomap(n_neighbors=12, n_components=3).fit_transform(X)),
    ("Isomap 3D (k=20)", Isomap(n_neighbors=20, n_components=3).fit_transform(X)),
]

print("=" * 105)
print(f"{'Método':^18} | {'Exact. /o/ (5-clases)':^22} | {'Exact. /u/ (5-clases)':^22} | {'Confusión O<->U':^16} | {'LDA Binario O-U':^15}")
print("=" * 105)

for nombre, Z in configs:
    # 1. GMM 5 clases global
    gmm5 = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20)
    preds5 = gmm5.fit_predict(Z)
    cm5 = confusion_matrix(y_true_5, preds5)
    r_ind, c_ind = linear_sum_assignment(-cm5)
    map_dict = {c: r for r, c in zip(r_ind, c_ind)}
    mapped5 = np.array([map_dict[p] for p in preds5])
    
    # Metricas en O y U
    acc_o_5 = np.mean(mapped5[Y == 'O'] == idx_o) * 100.0
    acc_u_5 = np.mean(mapped5[Y == 'U'] == idx_u) * 100.0
    
    # Confusión cruzada entre O y U
    o_como_u = np.sum(mapped5[Y == 'O'] == idx_u)
    u_como_o = np.sum(mapped5[Y == 'U'] == idx_o)
    
    # 2. LDA Binario directo O vs U
    mask_ou = np.isin(Y, ['O', 'U'])
    Z_ou = Z[mask_ou]
    y_ou = (Y[mask_ou] == 'U').astype(int)
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_ou, y_ou)
    acc_lda = np.mean(lda.predict(Z_ou) == y_ou) * 100.0
    
    print(f"{nombre:^18} | {acc_o_5:^6.1f}% ({np.sum(mapped5[Y == 'O'] == idx_o)}/{n_o_total})   | {acc_u_5:^6.1f}% ({np.sum(mapped5[Y == 'U'] == idx_u)}/{n_u_total})   | O->U: {o_como_u:2d} | U->O: {u_como_o:2d} | {acc_lda:^13.1f}%")

print("=" * 105)
