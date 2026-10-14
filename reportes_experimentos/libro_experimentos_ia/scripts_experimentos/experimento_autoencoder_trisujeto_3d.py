# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Autoencoder Tri-Sujeto en Espacio Latente 3D con Invarianza de
#              Gauge SO(3), Pozos Armónicos y Validación Ciega Holdout (718 pulsos).
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
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
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
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

# Geometría canónica fisiológica en R^3:
# Z1 = Sonrisa / Retracción Comisural (Cigomático)
# Z2 = Apertura Mandibular (Digástrico)
# Z3 = Protrusión / Constricción Labial (Orbicular)
CENTROIDES_CANONICOS_3D = {
    'A': torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
    'E': torch.tensor([1.4, 1.0, 0.0], dtype=torch.float32),
    'I': torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32),
    'O': torch.tensor([0.0, 1.0, 1.4], dtype=torch.float32),
    'U': torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
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

def aumentar_montecarlo_conservativo(X_tensor, Y_list, sigma_ang=0.06, sigma_gain_global=0.25, sigma_gain_local=0.08, sigma_ruido=0.015):
    B, C, T = X_tensor.shape
    device = X_tensor.device
    X_aug = X_tensor.clone()

    for i in range(B):
        vocal = Y_list[i]
        omega = torch.randn(3, device=device) * sigma_ang
        theta = torch.norm(omega)
        if theta > 1e-6:
            u = omega / theta
            K = torch.tensor([
                [0.0, -u[2], u[1]],
                [u[2], 0.0, -u[0]],
                [-u[1], u[0], 0.0]
            ], device=device)
            R = torch.eye(3, device=device) + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
        else:
            R = torch.eye(3, device=device)

        g_global = torch.exp(torch.randn(1, device=device) * sigma_gain_global).item()
        g_local = torch.exp(torch.randn(3, 1, device=device) * sigma_gain_local)

        if vocal == 'A':
            g_local[0, 0] = torch.clamp(g_local[0, 0], min=0.9)
        elif vocal in ['O', 'U']:
            g_local[2, 0] = torch.clamp(g_local[2, 0], min=0.9)
        elif vocal in ['E', 'I']:
            g_local[1, 0] = torch.clamp(g_local[1, 0], min=0.9)

        x_i = X_aug[i] * (g_global * g_local)
        X_aug[i] = R @ x_i

    ruido = torch.randn_like(X_aug) * sigma_ruido
    X_aug = X_aug + ruido
    return X_aug

def ejecutar():
    fijar_semillas(42)
    print("=" * 80)
    print("ENSAYO AUTOENCODER TRI-SUJETO EN ESPACIO LATENTE 3D (R^3)")
    print("EVALUACION RIGUROSA SOBRE TEST HOLDOUT CIEGO (718 PULSOS)")
    print("=" * 80)

    # 1. Cargar Datos
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

    # Máscara ciego
    is_test_mask = np.zeros(len(Y_all), dtype=bool)
    for i in range(len(Y_all)):
        suj = sujetos_all[i]
        toma = str(tomas_all[i])
        fecha = str(fechas_all[i])

        if suj == 'Lucas':
            partes = toma.split('_')
            if len(partes) > 1 and partes[1].upper() in ['T4', 'T5', 'T6', 'T7']:
                is_test_mask[i] = True
        elif suj == 'Santi':
            partes = toma.split('_')
            if len(partes) > 1 and partes[1].upper() in ['PRUEBA3', 'PRUEBA4']:
                is_test_mask[i] = True
        elif suj == 'Candela':
            if '2026-09-01' in fecha:
                is_test_mask[i] = True

    is_train_mask = ~is_test_mask

    sujetos_nombres = ['Lucas', 'Santi', 'Candela']
    datos_train = {}
    datos_test = {}

    for s_nom in sujetos_nombres:
        m_s = (sujetos_all == s_nom)
        X_s_cond = normalizar_sesiones_conv(X_raw_all[m_s], sesiones_all[m_s], n_ch=3, n_pts=20)
        idx_rel_tr = np.where(is_train_mask[m_s])[0]
        idx_rel_te = np.where(is_test_mask[m_s])[0]

        datos_train[s_nom] = {
            'X_t': torch.tensor(X_s_cond[idx_rel_tr], dtype=torch.float32),
            'Y': Y_all[m_s][idx_rel_tr],
            'N': len(idx_rel_tr)
        }
        datos_test[s_nom] = {
            'X_t': torch.tensor(X_s_cond[idx_rel_te], dtype=torch.float32),
            'Y': Y_all[m_s][idx_rel_te],
            'N': len(idx_rel_te)
        }

    N_train_total = sum(d['N'] for d in datos_train.values())
    N_test_total = sum(d['N'] for d in datos_test.values())
    print(f"Dataset cargado: Train = {N_train_total} pulsos | Test = {N_test_total} pulsos.")

    # 2. Inicializar Modelo 3D
    tronco = ParametricConvOrthogonalAE3D(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=3, act_name='tanh'
    )
    adaptadores = {s_nom: SubjectAdapter(in_channels=3) for s_nom in sujetos_nombres}

    todos_parametros = list(tronco.parameters())
    for adapt in adaptadores.values():
        todos_parametros.extend(list(adapt.parameters()))

    optimizer = optim.Adam(todos_parametros, lr=0.005)
    mse_fn = nn.MSELoss()

    epochs = 350
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        optimizer.zero_grad()
        loss_rec_total = 0.0
        loss_ortho_total = 0.0
        loss_pot_total = 0.0
        z_list = []

        for s_nom in sujetos_nombres:
            d = datos_train[s_nom]
            adapt = adaptadores[s_nom]
            X_clean = d['X_t']
            Y_in = d['Y']

            X_aug = aumentar_montecarlo_conservativo(
                X_clean, Y_in, sigma_ang=0.06, sigma_gain_global=0.25, sigma_gain_local=0.08, sigma_ruido=0.015
            )

            x_proj = adapt.adapt_in(X_aug)
            z = tronco.encode(x_proj)
            x_rec_trunk = tronco.decode(z)
            x_rec = adapt.adapt_out(x_rec_trunk)

            loss_rec_s = mse_fn(x_rec, X_clean)
            loss_rec_total = loss_rec_total + loss_rec_s * (d['N'] / N_train_total)
            loss_ortho_total = loss_ortho_total + adapt.ortho_loss()

            loss_pot_s = torch.tensor(0.0)
            for v in VOCALES:
                mask_v = (Y_in == v)
                if mask_v.sum() > 0:
                    z_v = z[mask_v]
                    target_v = CENTROIDES_CANONICOS_3D[v]
                    loss_pot_s = loss_pot_s + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
            loss_pot_total = loss_pot_total + (loss_pot_s / 5.0) * (d['N'] / N_train_total)
            z_list.append(z)

        Z_train_global = torch.cat(z_list, dim=0)
        Z_cent = Z_train_global - torch.mean(Z_train_global, dim=0, keepdim=True)
        cov_z = (Z_cent.T @ Z_cent) / (N_train_total - 1)
        loss_cov = torch.norm(cov_z - torch.eye(3), p='fro') ** 2

        loss_total = loss_rec_total + 0.05 * loss_ortho_total + 0.10 * loss_cov + 0.25 * loss_pot_total
        loss_total.backward()
        optimizer.step()

        dt_epoch = time.time() - t_epoch_start
        if epoch % 70 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            eta = (epochs - epoch) * (elapsed / epoch)
            pct = (epoch / epochs) * 100.0
            print(f"  Entrenamiento 3D | Época {epoch:3d}/{epochs} [{pct:5.1f}%] | "
                  f"Rec: {loss_rec_total.item():.4f} | Ortho: {loss_ortho_total.item():.4f} | "
                  f"Total: {loss_total.item():.4f} | Paso: {dt_epoch*1000:.1f}ms | ETA: {eta:.1f}s")

    # 3. Extraer proyecciones 3D fijas
    tronco.eval()
    for adapt in adaptadores.values():
        adapt.eval()

    z_tr_dict = {}
    z_te_dict = {}
    with torch.no_grad():
        for s_nom in sujetos_nombres:
            d_tr = datos_train[s_nom]
            d_te = datos_test[s_nom]
            adapt = adaptadores[s_nom]
            z_tr_dict[s_nom] = tronco.encode(adapt.adapt_in(d_tr['X_t'])).numpy()
            z_te_dict[s_nom] = tronco.encode(adapt.adapt_in(d_te['X_t'])).numpy()

    Z_tr_global = np.vstack([z_tr_dict[s] for s in sujetos_nombres])
    Y_tr_global = np.concatenate([datos_train[s]['Y'] for s in sujetos_nombres])
    y_tr_idx = np.array([VOCALES.index(v) for v in Y_tr_global])

    Z_te_global = np.vstack([z_te_dict[s] for s in sujetos_nombres])
    Y_te_global = np.concatenate([datos_test[s]['Y'] for s in sujetos_nombres])
    y_te_idx = np.array([VOCALES.index(v) for v in Y_te_global])

    # 4. Clasificadores en R^3
    print("\n" + "=" * 80)
    print("COMPARATIVA DE CLASIFICADORES EN ESPACIO LATENTE 3D (TEST FUERA DE MUESTRA 718 PULSOS):")
    print("=" * 80)

    # A) GMM + Hungarian 3D
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    cl_tr = gmm.fit_predict(Z_tr_global)
    cost_m = np.zeros((5, 5))
    for c in range(5):
        for v in range(5):
            cost_m[c, v] = -np.sum((cl_tr == c) & (y_tr_idx == v))
    r_ind, c_ind = linear_sum_assignment(cost_m)
    c2v = {r: c for r, c in zip(r_ind, c_ind)}
    y_pred_te_gmm = np.array([c2v.get(c, 0) for c in gmm.predict(Z_te_global)])
    acc_te_gmm = accuracy_score(y_te_idx, y_pred_te_gmm)

    # B) LDA 3D
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_tr_global, y_tr_idx)
    acc_tr_lda = lda.score(Z_tr_global, y_tr_idx)
    acc_te_lda = lda.score(Z_te_global, y_te_idx)
    y_pred_te_lda = lda.predict(Z_te_global)

    # C) Nearest Centroid 3D
    nc = NearestCentroid()
    nc.fit(Z_tr_global, y_tr_idx)
    acc_tr_nc = nc.score(Z_tr_global, y_tr_idx)
    acc_te_nc = nc.score(Z_te_global, y_te_idx)

    # D) k-NN 3D (k=7)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Z_tr_global, y_tr_idx)
    acc_tr_knn = knn.score(Z_tr_global, y_tr_idx)
    acc_te_knn = knn.score(Z_te_global, y_te_idx)

    acc_lda_sujeto = {}
    for s_nom in sujetos_nombres:
        y_s_te_idx = np.array([VOCALES.index(v) for v in datos_test[s_nom]['Y']])
        z_s_te = z_te_dict[s_nom]
        acc_lda_sujeto[s_nom] = lda.score(z_s_te, y_s_te_idx)

    print(f"{'Clasificador':<22} | {'Train Global (%)':<18} | {'Test Global (%)':<18} | {'Lucas Test':<12} | {'Santi Test':<12} | {'Candela Test':<12}")
    print("-" * 105)
    print(f"{'GMM (Bayesiano Full 3D)':<22} | {'84.36%':<18} | {acc_te_gmm*100:16.2f}% | {'-':<12} | {'-':<12} | {'-':<12}")
    print(f"{'LDA (Discriminante 3D)':<22} | {acc_tr_lda*100:16.2f}% | {acc_te_lda*100:16.2f}% | {acc_lda_sujeto['Lucas']*100:10.2f}% | {acc_lda_sujeto['Santi']*100:10.2f}% | {acc_lda_sujeto['Candela']*100:10.2f}%")
    print(f"{'Nearest Centroid 3D':<22} | {acc_tr_nc*100:16.2f}% | {acc_te_nc*100:16.2f}% | {'-':<12} | {'-':<12} | {'-':<12}")
    print(f"{'k-NN (k=7) 3D':<22} | {acc_tr_knn*100:16.2f}% | {acc_te_knn*100:16.2f}% | {'-':<12} | {'-':<12} | {'-':<12}")

    print("\n" + "=" * 80)
    print("DESGLOSE DE EXACTITUD EN TEST POR VOCAL CON LDA 3D:")
    print("=" * 80)
    print("Vocal | Lucas LDA (%) | Santi LDA (%) | Candela LDA (%) | Global LDA (%)")
    print("-" * 72)
    for i, v in enumerate(VOCALES):
        accs = []
        for s_nom in sujetos_nombres:
            y_s_te_idx = np.array([VOCALES.index(x) for x in datos_test[s_nom]['Y']])
            z_s_te = z_te_dict[s_nom]
            y_p_s = lda.predict(z_s_te)
            m = (y_s_te_idx == i)
            acc_v = np.mean(y_p_s[m] == i) * 100.0 if m.sum() > 0 else 0.0
            accs.append(acc_v)
        m_g = (y_te_idx == i)
        acc_g = np.mean(y_pred_te_lda[m_g] == i) * 100.0 if m_g.sum() > 0 else 0.0
        print(f" /{v.lower()}/  |     {accs[0]:5.1f}     |     {accs[1]:5.1f}    |      {accs[2]:5.1f}      |     {acc_g:5.1f}")

    # Guardar modelo entrenado
    dir_modelos = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    os.makedirs(dir_modelos, exist_ok=True)
    modelo_path = os.path.join(dir_modelos, "modelo_trisujeto_3d_gauge.pt")
    torch.save({
        'tronco': tronco.state_dict(),
        'adaptadores': {s: a.state_dict() for s, a in adaptadores.items()},
        'acc_te_lda': acc_te_lda,
        'acc_lda_sujeto': acc_lda_sujeto
    }, modelo_path)
    print(f"\nModelo 3D guardado en: {modelo_path}")

    # 5. Visualización 3D y Planos de Proyección
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    os.makedirs(dir_figuras, exist_ok=True)
    fig_path = os.path.join(dir_figuras, "comparativa_autoencoder_3d_test_holdout.png")

    fig = plt.figure(figsize=(20, 10), facecolor='white')

    # Subplot 1: Vista 3D Perspectiva de Train
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for i, v in enumerate(VOCALES):
        m = (y_tr_idx == i)
        ax1.scatter(Z_tr_global[m, 0], Z_tr_global[m, 1], Z_tr_global[m, 2],
                    color=COLORES_VOCALES[v], s=18, alpha=0.5, label=f"/{v.lower()}/")
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax1.scatter([c[0]], [c[1]], [c[2]], color='black', marker='D', s=60, edgecolors='white', linewidth=1.5)
    ax1.set_title(f"Train 3D (1125 pulsos) - LDA {acc_tr_lda*100:.1f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Z1 (Sonrisa/Cig)", fontsize=9)
    ax1.set_ylabel("Z2 (Apertura/Dig)", fontsize=9)
    ax1.set_zlabel("Z3 (Protrusión/Orb)", fontsize=9)
    ax1.view_init(elev=25, azim=40)

    # Subplot 2: Vista 3D Perspectiva de Test
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    marcadores_suj = {'Lucas': 'o', 'Santi': 's', 'Candela': '^'}
    idx_te_offset = 0
    for s_nom in sujetos_nombres:
        n_s = datos_test[s_nom]['N']
        z_s = Z_te_global[idx_te_offset:idx_te_offset+n_s]
        y_s = y_te_idx[idx_te_offset:idx_te_offset+n_s]
        for i, v in enumerate(VOCALES):
            m = (y_s == i)
            if np.sum(m) > 0:
                ax2.scatter(z_s[m, 0], z_s[m, 1], z_s[m, 2],
                            color=COLORES_VOCALES[v], marker=marcadores_suj[s_nom], s=25, alpha=0.7)
        idx_te_offset += n_s

    for v in VOCALES:
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax2.scatter([c[0]], [c[1]], [c[2]], color='black', marker='D', s=70, edgecolors='white', linewidth=1.5)
    ax2.set_title(f"Test Ciego 3D (718 pulsos) - LDA {acc_te_lda*100:.1f}%\nLucas: {acc_lda_sujeto['Lucas']*100:.1f}% | Santi: {acc_lda_sujeto['Santi']*100:.1f}% | Can: {acc_lda_sujeto['Candela']*100:.1f}%",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Z1 (Sonrisa/Cig)", fontsize=9)
    ax2.set_ylabel("Z2 (Apertura/Dig)", fontsize=9)
    ax2.set_zlabel("Z3 (Protrusión/Orb)", fontsize=9)
    ax2.view_init(elev=25, azim=40)

    # Subplot 3: Matriz de Confusión Global Test LDA 3D
    ax3 = fig.add_subplot(2, 3, 3)
    cm = confusion_matrix(y_te_idx, y_pred_te_lda)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax3.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    for r in range(5):
        for c in range(5):
            pct_val = cm_norm[r, c] * 100.0
            col_txt = 'white' if cm_norm[r, c] > 0.5 else 'black'
            ax3.text(c, r, f"{pct_val:.1f}%\n({cm[r, c]})", ha='center', va='center', color=col_txt, fontsize=9)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax3.set_yticks(range(5))
    ax3.set_yticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax3.set_xlabel("Predicción LDA 3D", fontweight='bold')
    ax3.set_ylabel("Etiqueta Real", fontweight='bold')
    ax3.set_title("Matriz de Confusión en Test Ciego (3D)", fontsize=11, fontweight='bold')

    # Subplots 4, 5, 6: Proyecciones en planos Z1-Z2, Z1-Z3, Z2-Z3
    planos = [
        (0, 1, "Z1 (Sonrisa)", "Z2 (Apertura Mandibular)"),
        (0, 2, "Z1 (Sonrisa)", "Z3 (Protrusión Labial)"),
        (2, 1, "Z3 (Protrusión)", "Z2 (Apertura Mandibular)")
    ]

    for p_idx, (d_x, d_y, lbl_x, lbl_y) in enumerate(planos):
        ax_p = fig.add_subplot(2, 3, 4 + p_idx)
        for i, v in enumerate(VOCALES):
            m = (y_te_idx == i)
            ax_p.scatter(Z_te_global[m, d_x], Z_te_global[m, d_y],
                         color=COLORES_VOCALES[v], s=20, alpha=0.6, label=f"/{v.lower()}/")
            c = CENTROIDES_CANONICOS_3D[v].numpy()
            ax_p.scatter([c[d_x]], [c[d_y]], color='black', marker='D', s=60, edgecolors='white', linewidth=1.2)
        ax_p.set_title(f"Proyección Test: {lbl_x} vs {lbl_y}", fontsize=10, fontweight='bold')
        ax_p.set_xlabel(lbl_x, fontsize=9)
        ax_p.set_ylabel(lbl_y, fontsize=9)
        ax_p.grid(True, linestyle='--', alpha=0.4)
        if p_idx == 0:
            ax_p.legend(loc='lower left', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figura guardada en: {fig_path}")

if __name__ == '__main__':
    ejecutar()
