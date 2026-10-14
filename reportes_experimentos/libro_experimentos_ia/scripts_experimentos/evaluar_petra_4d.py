# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación en Sujeto Inédito Petra del Autoencoder 4D con Gauge SO(3)
# ==============================================================================

import os
import sys
import random
import numpy as np
from scipy.signal import butter, filtfilt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ParametricConvOrthogonalAE4D(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=4, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        c1, c2 = conv_channels
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh() if act_name == 'tanh' else nn.ReLU()

        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)

        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

    def encode(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        return self.fc2(h3)

    def decode(self, z):
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        recon_3d = self.deconv2(dh3)
        return recon_3d

class SubjectAdapter(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.read_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)

    def adapt_in(self, x):
        return self.read_in(x)

VOCALES = ['A', 'E', 'I', 'O', 'U']
CENTROIDES_CANONICOS_4D = {
    'A': torch.tensor([0.0, 2.0, 0.0, 0.0], dtype=torch.float32),
    'E': torch.tensor([1.4, 1.0, 0.0, -1.0], dtype=torch.float32),
    'I': torch.tensor([2.0, 0.0, 0.0, 1.0], dtype=torch.float32),
    'O': torch.tensor([0.0, 1.0, 1.4, -1.0], dtype=torch.float32),
    'U': torch.tensor([0.0, 0.0, 2.0, 1.0], dtype=torch.float32),
}

def normalizar_sesiones_conv(X_raw, sesiones, n_ch=3, n_pts=20):
    N = X_raw.shape[0]
    b_bw, a_bw = butter(N=3, Wn=0.3, btype='low')
    X_reshaped = X_raw.reshape(N, n_ch, n_pts)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N):
        for c in range(n_ch):
            X_filt[i, c, :] = filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for s in np.unique(sesiones):
        mask = (sesiones == s)
        for c in range(n_ch):
            base_mean = np.mean(X_filt[mask, c, :10])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    return X_norm.reshape(N, n_ch, n_pts)

def main():
    fijar_semillas(42)
    p_ckpt = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/modelo_trisujeto_4d_gauge.pt"
    )
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)

    tronco = ParametricConvOrthogonalAE4D(latent_dim=4)
    tronco.load_state_dict(ckpt['tronco'])
    tronco.eval()

    adaptadores = {}
    for s, st in ckpt['adaptadores'].items():
        ad = SubjectAdapter(in_channels=3)
        ad.load_state_dict(st)
        ad.eval()
        adaptadores[s] = ad

    # Cargar datos de entrenamiento
    npz_path = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    data = np.load(npz_path, allow_pickle=True)
    X_raw_all = data['X_raw']
    Y_all = data['Y_raw']
    sujetos_all = data['sujetos_raw']
    fechas_all = data['fechas_raw']
    tomas_all = data['tomas_raw']

    def extraer_sesion(t):
        p = str(t).split('_')
        return p[1].upper() if len(p) > 1 else 'S1'

    sesiones_all = np.array([extraer_sesion(t) for t in tomas_all])

    is_test_mask = np.zeros(len(Y_all), dtype=bool)
    for i in range(len(Y_all)):
        suj = sujetos_all[i]
        toma = str(tomas_all[i])
        fecha = str(fechas_all[i])
        if suj == 'Lucas' and any(x in toma.upper() for x in ['T4', 'T5', 'T6', 'T7']):
            is_test_mask[i] = True
        elif suj == 'Santi' and any(x in toma.upper() for x in ['PRUEBA3', 'PRUEBA4']):
            is_test_mask[i] = True
        elif suj == 'Candela' and '2026-09-01' in fecha:
            is_test_mask[i] = True

    is_train_mask = ~is_test_mask
    sujetos_nombres = ['Lucas', 'Santi', 'Candela']

    z_tr_list, y_tr_list = [], []
    with torch.no_grad():
        for s_nom in sujetos_nombres:
            m_s = (sujetos_all == s_nom)
            X_s_cond = normalizar_sesiones_conv(X_raw_all[m_s], sesiones_all[m_s], n_ch=3, n_pts=20)
            idx_rel_tr = np.where(is_train_mask[m_s])[0]
            X_tensor = torch.tensor(X_s_cond[idx_rel_tr], dtype=torch.float32)
            adapt = adaptadores[s_nom]
            z_s = tronco.encode(adapt.adapt_in(X_tensor)).numpy()
            z_tr_list.append(z_s)
            y_tr_list.append(Y_all[m_s][idx_rel_tr])

    Z_tr_global = np.vstack(z_tr_list)
    Y_tr_global = np.concatenate(y_tr_list)
    y_tr_idx = np.array([VOCALES.index(v) for v in Y_tr_global])

    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_tr_global, y_tr_idx)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Z_tr_global, y_tr_idx)

    # Cargar datos de Petra
    dir_grid = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_d1 = os.path.join(dir_grid, "dataset_petra_dia1_20260821_conv2d.npz")
    p_d2 = os.path.join(dir_grid, "dataset_petra_dia2_20260828_conv2d.npz")

    d1 = np.load(p_d1, allow_pickle=True)
    d2 = np.load(p_d2, allow_pickle=True)
    X1_raw, Y1_raw = d1['X_clean'], d1['Y_clean']
    X2_raw, Y2_raw = d2['X_clean'], d2['Y_clean']

    if X1_raw.ndim == 2:
        X1_raw = X1_raw.reshape(-1, 3, 20)
    if X2_raw.ndim == 2:
        X2_raw = X2_raw.reshape(-1, 3, 20)

    X1_t = torch.tensor(X1_raw, dtype=torch.float32)
    X2_t = torch.tensor(X2_raw, dtype=torch.float32)
    y1_idx = np.array([VOCALES.index(v) for v in Y1_raw])
    y2_idx = np.array([VOCALES.index(v) for v in Y2_raw])

    # 1. Zero-shot puro en Petra (W = I)
    with torch.no_grad():
        Z1_zs = tronco.encode(X1_t).numpy()
        Z2_zs = tronco.encode(X2_t).numpy()
    acc_zs_d1 = lda.score(Z1_zs, y1_idx)
    acc_zs_d2 = lda.score(Z2_zs, y2_idx)
    print("=" * 80)
    print(f"PETRA ZERO-SHOT PURO (W = I en 4D):")
    print(f"  Día 1 (86 pulsos):  LDA = {acc_zs_d1*100:5.2f}%")
    print(f"  Día 2 (168 pulsos): LDA = {acc_zs_d2*100:5.2f}%")

    # 2. Calibración rápida en Día 1 (Gauge SO(3) 3x3)
    W_petra = nn.Parameter(torch.eye(3))
    opt_p = optim.Adam([W_petra], lr=0.03)
    for ep_p in range(150):
        opt_p.zero_grad()
        X1_rot = torch.matmul(W_petra, X1_t)
        z1_p = tronco.encode(X1_rot)
        loss_pot_p = torch.tensor(0.0)
        for v in VOCALES:
            m_v = (Y1_raw == v)
            if m_v.sum() > 0:
                z_v = z1_p[m_v]
                target_v = CENTROIDES_CANONICOS_4D[v]
                loss_pot_p = loss_pot_p + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
        loss_ortho_p = torch.norm(W_petra @ W_petra.T - torch.eye(3), p='fro') ** 2
        loss_p = loss_pot_p + 1.0 * loss_ortho_p
        loss_p.backward()
        opt_p.step()

    with torch.no_grad():
        U, _, Vt = torch.linalg.svd(W_petra)
        W_petra.copy_(U @ Vt)
        Z1_cal = tronco.encode(torch.matmul(W_petra, X1_t)).numpy()
        Z2_cal = tronco.encode(torch.matmul(W_petra, X2_t)).numpy()

    acc_cal_d1_lda = lda.score(Z1_cal, y1_idx)
    acc_cal_d2_lda = lda.score(Z2_cal, y2_idx)
    acc_cal_d1_knn = knn.score(Z1_cal, y1_idx)
    acc_cal_d2_knn = knn.score(Z2_cal, y2_idx)

    print("\n" + "=" * 80)
    print("PETRA CON CALIBRACION RAPIDA DE GAUGE (0.73s en Día 1 -> test en Día 2):")
    print("=" * 80)
    print(f"Día 1 (86 pulsos)  | LDA 4D: {acc_cal_d1_lda*100:5.2f}% | k-NN (k=7) 4D: {acc_cal_d1_knn*100:5.2f}%")
    print(f"Día 2 (168 pulsos) | LDA 4D: {acc_cal_d2_lda*100:5.2f}% | k-NN (k=7) 4D: {acc_cal_d2_knn*100:5.2f}%")

    print("\nDesglose Petra Día 2 por vocal con LDA 4D:")
    y2_pred_p = lda.predict(Z2_cal)
    for i, v in enumerate(VOCALES):
        m_v = (y2_idx == i)
        if m_v.sum() > 0:
            acc_v = np.mean(y2_pred_p[m_v] == i) * 100.0
            print(f"  /{v.lower()}/: {acc_v:5.1f}% ({np.sum(y2_pred_p[m_v] == i)}/{m_v.sum()})")

    # Confusión en Día 2
    cm_d2 = confusion_matrix(y2_idx, y2_pred_p)
    print("\nMatriz de Confusión Petra Día 2 (Filas: Real, Columnas: Prediccion LDA 4D):")
    print(f"{'':>6} " + " ".join([f"{v:>6}" for v in VOCALES]))
    for i, v in enumerate(VOCALES):
        fila_str = " ".join([f"{cm_d2[i, j]:6d}" for j in range(len(VOCALES))])
        print(f"{v:>6} {fila_str}")

if __name__ == '__main__':
    main()
