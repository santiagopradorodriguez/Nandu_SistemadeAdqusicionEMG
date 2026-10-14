# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generación y graficación de Matrices de Confusión de Petra
#              (Día 1 vs Día 2 vs Total) con Autoencoder 3D.
# ==============================================================================

import os
import sys
import time
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
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

class ParametricConvOrthogonalAE3D(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=3, act_name='tanh'):
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

    def forward(self, x):
        z = self.encode(x)
        recon_3d = self.decode(z)
        return recon_3d, z

VOCALES = ['A', 'E', 'I', 'O', 'U']

CENTROIDES_CANONICOS_3D = {
    'A': torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
    'E': torch.tensor([1.4, 1.0, 0.0], dtype=torch.float32),
    'I': torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32),
    'O': torch.tensor([0.0, 1.0, 1.4], dtype=torch.float32),
    'U': torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
}

class SubjectAdapter(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.read_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.read_in.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_in.bias.zero_()
            self.read_out.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.bias.zero_()

    def adapt_in(self, x):
        return self.read_in(x)

    def adapt_out(self, x_rec):
        return self.read_out(x_rec)

    def ortho_loss(self):
        W = self.read_in.weight.squeeze(-1)
        diff = W @ W.T - torch.eye(3, device=W.device)
        return torch.norm(diff, p='fro') ** 2

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

    # 1. Cargar Modelo 3D
    p_ckpt = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/modelo_trisujeto_3d_gauge.pt"
    )
    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    tronco = ParametricConvOrthogonalAE3D(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=3, act_name='tanh'
    )
    tronco.load_state_dict(ckpt['tronco'])
    tronco.eval()

    # 2. Cargar Train Tri-Sujeto y Ajustar LDA
    npz_tri = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    data_tri = np.load(npz_tri, allow_pickle=True)
    X_raw_all = data_tri['X_raw']
    Y_all = data_tri['Y_raw']
    sujetos_all = data_tri['sujetos_raw']
    fechas_all = data_tri['fechas_raw']
    tomas_all = data_tri['tomas_raw']

    def extraer_sesion(t):
        p = str(t).split('_')
        return p[1].upper() if len(p) > 1 else 'S1'

    sesiones_all = np.array([extraer_sesion(t) for t in tomas_all])

    sujetos_nombres = ['Lucas', 'Santi', 'Candela']
    adaptadores = {s: SubjectAdapter(in_channels=3) for s in sujetos_nombres}
    for s in sujetos_nombres:
        adaptadores[s].load_state_dict(ckpt['adaptadores'][s])
        adaptadores[s].eval()

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

    z_tr_list = []
    y_tr_list = []
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

    lda_tri = LinearDiscriminantAnalysis()
    lda_tri.fit(Z_tr_global, y_tr_idx)

    # 3. Cargar Petra
    dir_grid = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    d1 = np.load(os.path.join(dir_grid, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(dir_grid, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)

    X_p1, Y_p1 = d1['X_clean'], d1['Y_clean']
    X_p2, Y_p2 = d2['X_clean'], d2['Y_clean']
    if X_p1.ndim == 2:
        X_p1 = X_p1.reshape(-1, 3, 20)
    if X_p2.ndim == 2:
        X_p2 = X_p2.reshape(-1, 3, 20)

    # 4. Calibrar Adaptador de Petra con Día 1 (86 pulsos)
    adapt_petra = SubjectAdapter(in_channels=3)
    opt_petra = optim.Adam(adapt_petra.parameters(), lr=0.01)
    mse_fn = nn.MSELoss()
    X_p1_t = torch.tensor(X_p1, dtype=torch.float32)

    for ep in range(120):
        opt_petra.zero_grad()
        x_proj = adapt_petra.adapt_in(X_p1_t)
        z_p = tronco.encode(x_proj)
        x_rec = adapt_petra.adapt_out(tronco.decode(z_p))
        loss_rec = mse_fn(x_rec, X_p1_t)
        loss_ortho = adapt_petra.ortho_loss()
        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_p1):
            mask_v = (Y_p1 == v)
            if mask_v.sum() > 0:
                z_v = z_p[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
        loss = loss_rec + 0.05 * loss_ortho + 0.20 * loss_pot
        loss.backward()
        opt_petra.step()

    adapt_petra.eval()

    # 5. Extraer predicciones
    with torch.no_grad():
        Z_p1 = tronco.encode(adapt_petra.adapt_in(X_p1_t)).numpy()
        Z_p2 = tronco.encode(adapt_petra.adapt_in(torch.tensor(X_p2, dtype=torch.float32))).numpy()

    y_p1_true = np.array([VOCALES.index(v) for v in Y_p1])
    y_p2_true = np.array([VOCALES.index(v) for v in Y_p2])

    y_p1_pred = lda_tri.predict(Z_p1)
    y_p2_pred = lda_tri.predict(Z_p2)

    acc_d1 = accuracy_score(y_p1_true, y_p1_pred)
    acc_d2 = accuracy_score(y_p2_true, y_p2_pred)

    Z_pall = np.vstack([Z_p1, Z_p2])
    y_pall_true = np.concatenate([y_p1_true, y_p2_true])
    y_pall_pred = np.concatenate([y_p1_pred, y_p2_pred])
    acc_all = accuracy_score(y_pall_true, y_pall_pred)

    cm_d1 = confusion_matrix(y_p1_true, y_p1_pred, labels=range(5))
    cm_d2 = confusion_matrix(y_p2_true, y_p2_pred, labels=range(5))
    cm_all = confusion_matrix(y_pall_true, y_pall_pred, labels=range(5))

    print(f"Exactitud Día 1 (Calibración N={len(Y_p1)}): {acc_d1*100:.2f}%")
    print(f"Exactitud Día 2 (Test Ciego N={len(Y_p2)}):   {acc_d2*100:.2f}%")
    print(f"Exactitud Total Petra (N={len(y_pall_true)}):      {acc_all*100:.2f}%")

    print("\n--- MATRIZ DE CONFUSION PETRA DIA 1 (2026-08-21, N=86) ---")
    print(f"{'Vocal':<6} | {'/a/':<8} | {'/e/':<8} | {'/i/':<8} | {'/o/':<8} | {'/u/':<8} | Exactitud")
    print("-" * 65)
    for i, v in enumerate(VOCALES):
        n_tot = np.sum(cm_d1[i])
        acc_v = (cm_d1[i, i] / n_tot * 100.0) if n_tot > 0 else 0.0
        print(f" /{v.lower()}/  | {cm_d1[i,0]:^8d} | {cm_d1[i,1]:^8d} | {cm_d1[i,2]:^8d} | {cm_d1[i,3]:^8d} | {cm_d1[i,4]:^8d} | {acc_v:6.1f}% (N={n_tot})")

    print("\n--- MATRIZ DE CONFUSION PETRA DIA 2 (2026-08-28, N=168) ---")
    print(f"{'Vocal':<6} | {'/a/':<8} | {'/e/':<8} | {'/i/':<8} | {'/o/':<8} | {'/u/':<8} | Exactitud")
    print("-" * 65)
    for i, v in enumerate(VOCALES):
        n_tot = np.sum(cm_d2[i])
        acc_v = (cm_d2[i, i] / n_tot * 100.0) if n_tot > 0 else 0.0
        print(f" /{v.lower()}/  | {cm_d2[i,0]:^8d} | {cm_d2[i,1]:^8d} | {cm_d2[i,2]:^8d} | {cm_d2[i,3]:^8d} | {cm_d2[i,4]:^8d} | {acc_v:6.1f}% (N={n_tot})")

    print("\n--- MATRIZ DE CONFUSION TOTAL PETRA (AMBOS DIAS, N=254) ---")
    print(f"{'Vocal':<6} | {'/a/':<8} | {'/e/':<8} | {'/i/':<8} | {'/o/':<8} | {'/u/':<8} | Exactitud")
    print("-" * 65)
    for i, v in enumerate(VOCALES):
        n_tot = np.sum(cm_all[i])
        acc_v = (cm_all[i, i] / n_tot * 100.0) if n_tot > 0 else 0.0
        print(f" /{v.lower()}/  | {cm_all[i,0]:^8d} | {cm_all[i,1]:^8d} | {cm_all[i,2]:^8d} | {cm_all[i,3]:^8d} | {cm_all[i,4]:^8d} | {acc_v:6.1f}% (N={n_tot})")

    # 6. Graficar Figura Comparativa 1x3
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "matrices_confusion_petra_dos_dias.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor='white')

    matrices = [
        (cm_d1, f"Petra Día 1 (2026-08-21)\nCalibración (N=86) - Exactitud: {acc_d1*100:.1f}%", 'Blues'),
        (cm_d2, f"Petra Día 2 (2026-08-28)\nTest Ciego (N=168) - Exactitud: {acc_d2*100:.1f}%", 'Greens'),
        (cm_all, f"Petra Total (Ambos Días)\nGlobal (N=254) - Exactitud: {acc_all*100:.1f}%", 'Purples')
    ]

    for ax, (mat, titulo, cmap_name) in zip(axes, matrices):
        mat_norm = mat.astype('float') / np.maximum(mat.sum(axis=1)[:, np.newaxis], 1)
        im = ax.imshow(mat_norm, cmap=cmap_name, vmin=0, vmax=1)
        for r in range(5):
            for c in range(5):
                n_c = mat[r, c]
                pct = mat_norm[r, c] * 100.0
                col_txt = 'white' if mat_norm[r, c] > 0.5 else 'black'
                ax.text(c, r, f"{pct:.1f}%\n({n_c})", ha='center', va='center', color=col_txt, fontsize=9, fontweight='bold')

        ax.set_xticks(range(5))
        ax.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontsize=10, fontweight='bold')
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"/{v.lower()}/" for v in VOCALES], fontsize=10, fontweight='bold')
        ax.set_xlabel("Predicción LDA 3D", fontsize=10, fontweight='bold')
        ax.set_ylabel("Etiqueta Real", fontsize=10, fontweight='bold')
        ax.set_title(titulo, fontsize=11, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nFigura guardada en: {fig_path}")

if __name__ == '__main__':
    main()
