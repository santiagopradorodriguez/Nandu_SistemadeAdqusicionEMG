# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación de Ganancia Relativa Inter-Canal sobre el Test Ciego
#              Tri-Sujeto (718 pulsos: Lucas, Santi, Candela).
# ==============================================================================

import os
import sys
import time
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

class SubjectAdapterWithGain(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.log_gain = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.rot_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bias_in = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)

        with torch.no_grad():
            self.rot_in.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.bias.zero_()

    def adapt_in(self, x):
        g = torch.exp(self.log_gain)
        x_scaled = x * g
        return self.rot_in(x_scaled) + self.bias_in

    def adapt_out(self, x_rec):
        return self.read_out(x_rec)

    def ortho_loss(self):
        W = self.rot_in.weight.squeeze(-1)
        diff = W @ W.T - torch.eye(3, device=W.device)
        return torch.norm(diff, p='fro') ** 2

    def gain_regularization(self):
        return torch.mean(self.log_gain ** 2)

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
    print("=" * 80)
    print("ENSAYO: GANANCIA RELATIVA INTER-CANAL SOBRE TEST HOLDOUT TRI-SUJETO (718 PULSOS)")
    print("=" * 80)

    # Cargar Modelo 3D
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

    # Cargar Datos
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

    # 1. Baseline actual (sin calibrar ganancia diag, solo SO(3)):
    acc_te_base = ckpt['acc_te_lda']
    acc_suj_base = ckpt['acc_lda_sujeto']
    print(f"Baseline SO(3) 3D Actual: Test Global LDA = {acc_te_base*100:.2f}%")
    print(f"  Lucas = {acc_suj_base['Lucas']*100:.2f}% | Santi = {acc_suj_base['Santi']*100:.2f}% | Candela = {acc_suj_base['Candela']*100:.2f}%")

    # 2. Entrenar adaptadores con Ganancia Relativa diag(G)*SO(3) para cada sujeto
    CENTROIDES_3D = {
        'A': torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
        'E': torch.tensor([1.4, 1.0, 0.0], dtype=torch.float32),
        'I': torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32),
        'O': torch.tensor([0.0, 1.0, 1.4], dtype=torch.float32),
        'U': torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
    }

    adapt_gains = {s: SubjectAdapterWithGain(in_channels=3) for s in sujetos_nombres}
    mse_fn = nn.MSELoss()

    datos_train = {}
    datos_test = {}
    for s_nom in sujetos_nombres:
        m_s = (sujetos_all == s_nom)
        X_s_cond = normalizar_sesiones_conv(X_raw_all[m_s], sesiones_all[m_s], n_ch=3, n_pts=20)
        idx_tr = np.where(is_train_mask[m_s])[0]
        idx_te = np.where(is_test_mask[m_s])[0]
        datos_train[s_nom] = {
            'X': torch.tensor(X_s_cond[idx_tr], dtype=torch.float32),
            'Y': Y_all[m_s][idx_tr]
        }
        datos_test[s_nom] = {
            'X': torch.tensor(X_s_cond[idx_te], dtype=torch.float32),
            'Y': Y_all[m_s][idx_te]
        }

    # Calibrar adaptadores por sujeto en sus datos de entrenamiento
    for s_nom in sujetos_nombres:
        adapt = adapt_gains[s_nom]
        d_tr = datos_train[s_nom]
        X_tr = d_tr['X']
        Y_tr = d_tr['Y']
        opt = optim.Adam(adapt.parameters(), lr=0.01)

        for ep in range(150):
            opt.zero_grad()
            x_proj = adapt.adapt_in(X_tr)
            z = tronco.encode(x_proj)
            x_rec = adapt.adapt_out(tronco.decode(z))
            loss_rec = mse_fn(x_rec, X_tr)
            loss_ortho = adapt.ortho_loss()
            loss_g_reg = adapt.gain_regularization()

            loss_pot = torch.tensor(0.0)
            for v in np.unique(Y_tr):
                mask_v = (Y_tr == v)
                if mask_v.sum() > 0:
                    z_v = z[mask_v]
                    target_v = CENTROIDES_3D[v]
                    loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))

            loss = loss_rec + 0.05 * loss_ortho + 0.05 * loss_g_reg + 0.25 * loss_pot
            loss.backward()
            opt.step()
        adapt.eval()

    # Evaluar LDA con Ganancias
    z_tr_g_list = []
    y_tr_g_list = []
    z_te_g_dict = {}
    y_te_g_dict = {}

    with torch.no_grad():
        for s_nom in sujetos_nombres:
            adapt = adapt_gains[s_nom]
            z_tr = tronco.encode(adapt.adapt_in(datos_train[s_nom]['X'])).numpy()
            z_te = tronco.encode(adapt.adapt_in(datos_test[s_nom]['X'])).numpy()
            z_tr_g_list.append(z_tr)
            y_tr_g_list.append(datos_train[s_nom]['Y'])
            z_te_g_dict[s_nom] = z_te
            y_te_g_dict[s_nom] = datos_test[s_nom]['Y']

    Z_tr_g_all = np.vstack(z_tr_g_list)
    Y_tr_g_all = np.concatenate(y_tr_g_list)
    y_tr_g_idx = np.array([VOCALES.index(v) for v in Y_tr_g_all])

    lda_gain = LinearDiscriminantAnalysis()
    lda_gain.fit(Z_tr_g_all, y_tr_g_idx)

    acc_te_gain_suj = {}
    z_te_all = []
    y_te_all = []
    for s_nom in sujetos_nombres:
        y_te_idx = np.array([VOCALES.index(v) for v in y_te_g_dict[s_nom]])
        acc_s = lda_gain.score(z_te_g_dict[s_nom], y_te_idx)
        acc_te_gain_suj[s_nom] = acc_s
        z_te_all.append(z_te_g_dict[s_nom])
        y_te_all.append(y_te_idx)

    Z_te_g_all = np.vstack(z_te_all)
    Y_te_g_all = np.concatenate(y_te_all)
    acc_te_gain_global = lda_gain.score(Z_te_g_all, Y_te_g_all)

    print("\n" + "=" * 80)
    print("COMPARATIVA DE GANANCIA RELATIVA SOBRE TEST CIEGO (718 PULSOS):")
    print("=" * 80)
    print(f"{'Sujeto / Métrica':<25} | {'Base SO(3) (%)':<16} | {'Con Ganancia diag(G)*SO(3) (%)':<32} | {'Diferencia':<12}")
    print("-" * 90)
    print(f"{'Lucas Test (N=234)':<25} | {acc_suj_base['Lucas']*100:^16.2f} | {acc_te_gain_suj['Lucas']*100:^32.2f} | {'+' if acc_te_gain_suj['Lucas']>=acc_suj_base['Lucas'] else ''}{(acc_te_gain_suj['Lucas']-acc_suj_base['Lucas'])*100:5.2f}%")
    print(f"{'Santi Test (N=240)':<25} | {acc_suj_base['Santi']*100:^16.2f} | {acc_te_gain_suj['Santi']*100:^32.2f} | {'+' if acc_te_gain_suj['Santi']>=acc_suj_base['Santi'] else ''}{(acc_te_gain_suj['Santi']-acc_suj_base['Santi'])*100:5.2f}%")
    print(f"{'Candela Test (N=244)':<25} | {acc_suj_base['Candela']*100:^16.2f} | {acc_te_gain_suj['Candela']*100:^32.2f} | {'+' if acc_te_gain_suj['Candela']>=acc_suj_base['Candela'] else ''}{(acc_te_gain_suj['Candela']-acc_suj_base['Candela'])*100:5.2f}%")
    print("-" * 90)
    print(f"{'TEST GLOBAL (N=718)':<25} | {acc_te_base*100:^16.2f} | {acc_te_gain_global*100:^32.2f} | {'+' if acc_te_gain_global>=acc_te_base else ''}{(acc_te_gain_global-acc_te_base)*100:5.2f}%")

    # Imprimir ganancias aprendidas por sujeto
    print("\nGANANCIAS RELATIVAS APRENDIDAS POR CANAL:")
    for s_nom in sujetos_nombres:
        g = torch.exp(adapt_gains[s_nom].log_gain).squeeze().detach().numpy()
        print(f"  {s_nom:<8}: Ch0 (Digástrico) = {g[0]:.2f} | Ch1 (Cigomático) = {g[1]:.2f} | Ch2 (Orbicular) = {g[2]:.2f} | Ratio Ch0/Ch1 = {g[0]/g[1]:.2f} | Ratio Ch0/Ch2 = {g[0]/g[2]:.2f}")

    # Desglose por Vocal Global
    y_pred_all = lda_gain.predict(Z_te_g_all)
    cm_gain_all = confusion_matrix(Y_te_g_all, y_pred_all, labels=range(5))
    print("\nDESGLOSE POR VOCAL EN TEST GLOBAL (CON GANANCIA):")
    for i, v in enumerate(VOCALES):
        n_v = np.sum(Y_te_g_all == i)
        acc_v = cm_gain_all[i, i] / n_v * 100.0 if n_v > 0 else 0.0
        print(f"  /{v.lower()}/: {acc_v:5.1f}% (N={n_v})")

if __name__ == '__main__':
    main()
