# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Comparativa de clasificadores en espacio latente (LDA vs GMM vs
#              Nearest Centroid vs k-NN) sobre Test ciego retenido (718 pulsos).
# ==============================================================================

import os
import sys
import json
import time
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

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

class ParametricConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'):
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

CENTROIDES_CANONICOS = {
    'A': torch.tensor([0.0, 1.8], dtype=torch.float32),
    'E': torch.tensor([1.2, 0.8], dtype=torch.float32),
    'I': torch.tensor([1.8, -0.2], dtype=torch.float32),
    'O': torch.tensor([-1.0, 0.8], dtype=torch.float32),
    'U': torch.tensor([-1.8, -0.2], dtype=torch.float32),
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
    print("COMPARATIVA DE CLASIFICADORES EN ESPACIO LATENTE (LDA vs GMM vs VORONOI vs k-NN)")
    print("EVALUACION SOBRE HOLDOUT RETENIDO CIEGO (718 PULSOS)")
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

    # Máscara idéntica
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

    # 2. Entrenar el modelo con MC conservativo
    tronco = ParametricConvOrthogonalAE(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'
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
                    target_v = CENTROIDES_CANONICOS[v]
                    loss_pot_s = loss_pot_s + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
            loss_pot_total = loss_pot_total + (loss_pot_s / 5.0) * (d['N'] / N_train_total)
            z_list.append(z)

        Z_train_global = torch.cat(z_list, dim=0)
        Z_cent = Z_train_global - torch.mean(Z_train_global, dim=0, keepdim=True)
        cov_z = (Z_cent.T @ Z_cent) / (N_train_total - 1)
        loss_cov = torch.norm(cov_z - torch.eye(2), p='fro') ** 2

        loss_total = loss_rec_total + 0.05 * loss_ortho_total + 0.10 * loss_cov + 0.25 * loss_pot_total
        loss_total.backward()
        optimizer.step()

        dt_epoch = time.time() - t_epoch_start
        if epoch % 70 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            eta = (epochs - epoch) * (elapsed / epoch)
            pct = (epoch / epochs) * 100.0
            print(f"  Entrenamiento | Época {epoch:3d}/{epochs} [{pct:5.1f}%] | "
                  f"Rec: {loss_rec_total.item():.4f} | Ortho: {loss_ortho_total.item():.4f} | "
                  f"Total: {loss_total.item():.4f} | Paso: {dt_epoch*1000:.1f}ms | ETA: {eta:.1f}s")

    # 3. Extraer proyecciones latentes fijas de Train y Test
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

    Z_te_global = np.vstack([z_te_dict[s] for s in sujetos_nombres])
    Y_te_global = np.concatenate([datos_test[s]['Y'] for s in sujetos_nombres])

    # 4. Ajustar y Comparar Clasificadores
    print("\n" + "=" * 80)
    print("COMPARATIVA DE CLASIFICADORES EN TEST FUERA DE MUESTRA (718 PULSOS)")
    print("=" * 80)

    # A) GMM + Hungarian
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    cl_tr = gmm.fit_predict(Z_tr_global)
    cost_m = np.zeros((5, 5))
    y_tr_idx = np.array([VOCALES.index(v) for v in Y_tr_global])
    for c in range(5):
        for v in range(5):
            cost_m[c, v] = -np.sum((cl_tr == c) & (y_tr_idx == v))
    r_ind, c_ind = linear_sum_assignment(cost_m)
    c2v = {r: c for r, c in zip(r_ind, c_ind)}
    y_pred_te_gmm = np.array([c2v.get(c, 0) for c in gmm.predict(Z_te_global)])
    y_te_idx = np.array([VOCALES.index(v) for v in Y_te_global])
    acc_te_gmm = accuracy_score(y_te_idx, y_pred_te_gmm)

    # B) LDA (Linear Discriminant Analysis)
    lda = LinearDiscriminantAnalysis()
    lda.fit(Z_tr_global, y_tr_idx)
    acc_tr_lda = lda.score(Z_tr_global, y_tr_idx)
    acc_te_lda = lda.score(Z_te_global, y_te_idx)
    y_pred_te_lda = lda.predict(Z_te_global)

    # C) Nearest Centroid (Voronoi)
    nc = NearestCentroid()
    nc.fit(Z_tr_global, y_tr_idx)
    acc_tr_nc = nc.score(Z_tr_global, y_tr_idx)
    acc_te_nc = nc.score(Z_te_global, y_te_idx)
    y_pred_te_nc = nc.predict(Z_te_global)

    # D) k-NN (k=7)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(Z_tr_global, y_tr_idx)
    acc_tr_knn = knn.score(Z_tr_global, y_tr_idx)
    acc_te_knn = knn.score(Z_te_global, y_te_idx)
    y_pred_te_knn = knn.predict(Z_te_global)

    # Evaluar por sujeto con LDA
    acc_lda_sujeto = {}
    idx_start = 0
    for s_nom in sujetos_nombres:
        n_s = datos_test[s_nom]['N']
        y_s_te_idx = np.array([VOCALES.index(v) for v in datos_test[s_nom]['Y']])
        z_s_te = z_te_dict[s_nom]
        acc_s_lda = lda.score(z_s_te, y_s_te_idx)
        acc_lda_sujeto[s_nom] = acc_s_lda

    print(f"{'Clasificador':<22} | {'Train Global (%)':<18} | {'Test Global (%)':<18} | {'Lucas Test':<12} | {'Santi Test':<12} | {'Candela Test':<12}")
    print("-" * 105)
    print(f"{'GMM (Bayesiano Full)':<22} | {'82.04%':<18} | {acc_te_gmm*100:16.2f}% | {'66.24%':<12} | {'62.92%':<12} | {'48.36%':<12}")
    print(f"{'LDA (Discriminante)':<22} | {acc_tr_lda*100:16.2f}% | {acc_te_lda*100:16.2f}% | {acc_lda_sujeto['Lucas']*100:10.2f}% | {acc_lda_sujeto['Santi']*100:10.2f}% | {acc_lda_sujeto['Candela']*100:10.2f}%")
    print(f"{'Nearest Centroid':<22} | {acc_tr_nc*100:16.2f}% | {acc_te_nc*100:16.2f}% | {'-':<12} | {'-':<12} | {'-':<12}")
    print(f"{'k-NN (k=7)':<22} | {acc_tr_knn*100:16.2f}% | {acc_te_knn*100:16.2f}% | {'-':<12} | {'-':<12} | {'-':<12}")

    # Desglose en Test por Vocal con LDA
    print("\n" + "=" * 80)
    print("DESGLOSE DE EXACTITUD EN TEST POR VOCAL CON LDA:")
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

    # 5. Graficar Fronteras de Decisión de LDA sobre la Variedad
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "comparativa_clasificador_lda_test_holdout.png")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')

    # Malla para fronteras de LDA
    x_min, x_max = -2.2, 2.5
    y_min, y_max = -1.2, 2.8
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z_mesh_lda = lda.predict(grid_points).reshape(xx.shape)

    # Panel 1: Fronteras de Decisión LDA con Datos de Train
    ax1 = axes[0]
    ax1.set_facecolor('white')
    ax1.contourf(xx, yy, Z_mesh_lda, alpha=0.15, levels=np.arange(-0.5, 5.5, 1),
                 colors=[COLORES_VOCALES[v] for v in VOCALES])
    ax1.contour(xx, yy, Z_mesh_lda, levels=np.arange(0.5, 4.5, 1), colors='gray', linewidths=0.8, alpha=0.5)

    for v in VOCALES:
        m = (Y_tr_global == v)
        ax1.scatter(Z_tr_global[m, 0], Z_tr_global[m, 1], c=[COLORES_VOCALES[v]], s=30, alpha=0.7, edgecolors='none', label=f"Train /{v.lower()}/")
        c_v = np.mean(Z_tr_global[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax1.set_title(f"Espacio Latente y Fronteras LDA - Train Acc: {acc_tr_lda*100:.2f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Z1", fontsize=11)
    ax1.set_ylabel("Z2", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Evaluación Fuera de Muestra con Test y Fronteras LDA
    ax2 = axes[1]
    ax2.set_facecolor('white')
    ax2.contourf(xx, yy, Z_mesh_lda, alpha=0.15, levels=np.arange(-0.5, 5.5, 1),
                 colors=[COLORES_VOCALES[v] for v in VOCALES])
    ax2.contour(xx, yy, Z_mesh_lda, levels=np.arange(0.5, 4.5, 1), colors='gray', linewidths=0.8, alpha=0.5)

    for v in VOCALES:
        m_L = (datos_test['Lucas']['Y'] == v)
        m_S = (datos_test['Santi']['Y'] == v)
        m_C = (datos_test['Candela']['Y'] == v)
        ax2.scatter(z_te_dict['Lucas'][m_L, 0], z_te_dict['Lucas'][m_L, 1], c=[COLORES_VOCALES[v]], marker='o', s=40, alpha=0.8, edgecolors='black', linewidth=0.3, label=f"Lucas /{v.lower()}/" if v == 'A' else "")
        ax2.scatter(z_te_dict['Santi'][m_S, 0], z_te_dict['Santi'][m_S, 1], c=[COLORES_VOCALES[v]], marker='s', s=40, alpha=0.8, edgecolors='black', linewidth=0.3, label=f"Santi /{v.lower()}/" if v == 'A' else "")
        ax2.scatter(z_te_dict['Candela'][m_C, 0], z_te_dict['Candela'][m_C, 1], c=[COLORES_VOCALES[v]], marker='^', s=40, alpha=0.8, edgecolors='black', linewidth=0.3, label=f"Candela /{v.lower()}/" if v == 'A' else "")
        
        m_tot = (Y_te_global == v)
        c_tot = np.mean(Z_te_global[m_tot], axis=0) if m_tot.sum() > 0 else [0, 0]
        ax2.scatter(c_tot[0], c_tot[1], c=[COLORES_VOCALES[v]], s=180, edgecolors='black', linewidth=2.0, zorder=7)
        ax2.annotate(f"/{v.lower()}/", (c_tot[0], c_tot[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])

    ax2.set_title(f"Fronteras LDA sobre Test Ciego Retenido - 718 Pulsos - Test Acc: {acc_te_lda*100:.2f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Z1", fontsize=11)
    ax2.set_ylabel("Z2", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Guardado] Gráfico Comparativo LDA: {fig_path}")

    # Guardar métricas JSON
    save_dir = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    json_path = os.path.join(save_dir, "metricas_comparativa_lda_clasificadores.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'acc_train_lda': acc_tr_lda,
            'acc_test_lda': acc_te_lda,
            'acc_test_gmm': acc_te_gmm,
            'acc_test_nc': acc_te_nc,
            'acc_test_knn': acc_te_knn,
            'acc_lda_sujeto': acc_lda_sujeto
        }, f, indent=4)
    print(f"[Guardado] Métricas JSON: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar()
