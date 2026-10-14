# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Comparativa de Adaptador Afín GL(3) vs SO(3) en Petra.
# ==============================================================================

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experimento_grados_de_libertad_petra import ParametricConvOrthogonalAE3D, normalizar_sesiones_conv, CENTROIDES_CANONICOS_3D, VOCALES

def test_afin():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Cargar modelo
    p_ckpt = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/modelo_trisujeto_3d_gauge.pt")
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    tronco = ParametricConvOrthogonalAE3D(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=3, act_name='tanh')
    tronco.load_state_dict(ckpt['tronco'])
    tronco.eval()

    # 2. Cargar datos tri-sujeto y ajustar LDA
    npz_tri = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz")
    data_tri = np.load(npz_tri, allow_pickle=True)
    X_raw_all = data_tri['X_raw']
    Y_all = data_tri['Y_raw']
    sujetos_all = data_tri['sujetos_raw']
    tomas_all = data_tri['tomas_raw']

    is_test_mask = np.zeros(len(Y_all), dtype=bool)
    for i in range(len(Y_all)):
        suj = sujetos_all[i]
        toma = str(tomas_all[i])
        if suj == 'Lucas' and any(x in toma.upper() for x in ['T4', 'T5', 'T6', 'T7']):
            is_test_mask[i] = True
        elif suj == 'Santi' and any(x in toma.upper() for x in ['PRUEBA3', 'PRUEBA4']):
            is_test_mask[i] = True
        elif suj == 'Candela' and '2026-09-01' in str(data_tri['fechas_raw'][i]):
            is_test_mask[i] = True

    is_train_mask = ~is_test_mask
    z_tr_list = []
    y_tr_list = []
    with torch.no_grad():
        for s_nom in ['Lucas', 'Santi', 'Candela']:
            m_s = (sujetos_all == s_nom)
            X_s_cond = normalizar_sesiones_conv(X_raw_all[m_s], np.array(['S1']*np.sum(m_s)), n_ch=3, n_pts=20)
            idx_rel_tr = np.where(is_train_mask[m_s])[0]
            X_tensor = torch.tensor(X_s_cond[idx_rel_tr], dtype=torch.float32)
            from experimento_grados_de_libertad_petra import SubjectAdapter
            adapt = SubjectAdapter(3)
            adapt.load_state_dict(ckpt['adaptadores'][s_nom])
            adapt.eval()
            z_s = tronco.encode(adapt.adapt_in(X_tensor)).numpy()
            z_tr_list.append(z_s)
            y_tr_list.append(Y_all[m_s][idx_rel_tr])

    Z_tr = np.vstack(z_tr_list)
    y_tr_idx = np.array([VOCALES.index(v) for v in np.concatenate(y_tr_list)])
    lda_tri = LinearDiscriminantAnalysis()
    lda_tri.fit(Z_tr, y_tr_idx)

    # 3. Cargar Petra
    dir_grid = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    d1 = np.load(os.path.join(dir_grid, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(dir_grid, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)
    X_p1 = d1['X_clean'].reshape(-1, 3, 20)
    Y_p1 = d1['Y_clean']
    X_p2 = d2['X_clean'].reshape(-1, 3, 20)
    Y_p2 = d2['Y_clean']

    y_p2_idx = np.array([VOCALES.index(v) for v in Y_p2])
    X_p1_t = torch.tensor(X_p1, dtype=torch.float32)
    X_p2_t = torch.tensor(X_p2, dtype=torch.float32)

    # Adaptador Afín GL(3): Conv1d(3, 3, 1) sin forzar isometría estricta
    adapt_afin = nn.Conv1d(3, 3, kernel_size=1, bias=True)
    with torch.no_grad():
        adapt_afin.weight.copy_(torch.eye(3).unsqueeze(-1))
        adapt_afin.bias.zero_()

    opt = optim.Adam(adapt_afin.parameters(), lr=0.015)
    for ep in range(150):
        opt.zero_grad()
        x_proj = adapt_afin(X_p1_t)
        z = tronco.encode(x_proj)
        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_p1):
            mask_v = (Y_p1 == v)
            if mask_v.sum() > 0:
                z_v = z[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
        # Suave regularización de Frobenius para evitar divergencia
        W = adapt_afin.weight.squeeze(-1)
        loss_reg = 0.05 * torch.norm(W @ W.T - torch.eye(3), p='fro') ** 2
        loss = loss_pot + loss_reg
        loss.backward()
        opt.step()

    adapt_afin.eval()
    with torch.no_grad():
        Z_p2_afin = tronco.encode(adapt_afin(X_p2_t)).numpy()

    acc_afin = lda_tri.score(Z_p2_afin, y_p2_idx)
    y_p2_pred = lda_tri.predict(Z_p2_afin)
    cm_afin = confusion_matrix(y_p2_idx, y_p2_pred, labels=range(5))

    print(f"Exactitud Adaptador Afín GL(3) en Petra Día 2 (Test Ciego): {acc_afin*100:.2f}%")
    print("Desglose por vocal:")
    for i, v in enumerate(VOCALES):
        n_v = np.sum(y_p2_idx == i)
        acc_v = cm_afin[i, i] / n_v * 100.0 if n_v > 0 else 0.0
        print(f"  /{v.lower()}/: {acc_v:5.1f}% (N={n_v})")

if __name__ == '__main__':
    test_afin()
