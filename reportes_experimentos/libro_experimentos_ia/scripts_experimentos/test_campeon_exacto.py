import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.grid_search_conv_ortogonal import (
    ParametricConvOrthogonalAE,
    cargar_datos_lucas,
    vocales,
    vocal_to_idx
)

p_ckpt = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal/modelo_campeon_conv_ortogonal.pt")
ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

model = ParametricConvOrthogonalAE(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh')
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
model.eval()

X_lucas_t, y_lucas, n_ch, n_pts, b_bw, a_bw = cargar_datos_lucas()

with torch.no_grad():
    Z_lucas = model.encode(X_lucas_t).numpy()

gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
pred_raw = gmm.fit_predict(Z_lucas)

contingency = np.zeros((5, 5))
for i, vl in enumerate(vocales):
    for j in range(5):
        contingency[j, i] = np.sum((y_lucas == vl) & (pred_raw == j))
        
row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
pred_lucas_idx = np.array([cluster_to_vocal[c] for c in pred_raw])
y_lucas_idx = np.array([vocal_to_idx[v] for v in y_lucas])
acc = accuracy_score(y_lucas_idx, pred_lucas_idx) * 100.0

print(f"EXACTITUD REPRODUCIDA DE LUCAS: {acc:.2f}%")
