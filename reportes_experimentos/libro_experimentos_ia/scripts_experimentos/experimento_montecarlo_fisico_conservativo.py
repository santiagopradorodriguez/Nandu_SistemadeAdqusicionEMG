# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Data Augmentation Monte Carlo con sentido biofísico conservativo:
#              invarianza de sinergia muscular, perturbaciones SO(3) pequeñas (<5°),
#              preservación de envolvente y geometría de vocales intermedias (/e/ y /o/).
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

# Geometría Fisiológica Real de Vocales:
# /a/: Apertura máxima (0.0, 1.8)
# /e/: Estado intermedio sonrisa + apertura moderada (1.2, 0.8)
# /i/: Sonrisa pura / retracción comisural sin apertura (1.8, -0.2)
# /o/: Estado intermedio protrusión + apertura moderada (-1.0, 0.8)
# /u/: Protrusión máxima sin apertura (-1.8, -0.2)
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
    """
    Data Augmentation Monte Carlo Físicamente Coherente:
    1. Invarianza a la energía global (fatiga): escala los 3 canales uniformemente (preserva sinergia).
    2. Jitter local suave por canal (impedancia de contacto realista, sin despegues destructivos).
    3. Rotación angular pequeña en SO(3) (desviación típica < 4°).
    4. Preservación estricta de la firma de apertura mandibular en /a/ y de protrusión en /u/.
    """
    B, C, T = X_tensor.shape
    device = X_tensor.device
    X_aug = X_tensor.clone()

    for i in range(B):
        vocal = Y_list[i]
        
        # 1. Rotación SO(3) pequeña (realista de colocación cutánea: +-3.5 grados)
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

        # 2. Factor de escala global (simula fatiga o volumen fonatorio sin alterar ratios)
        g_global = torch.exp(torch.randn(1, device=device) * sigma_gain_global).item()
        
        # 3. Factores locales suaves (pequeñas asimetrías de impedancia)
        g_local = torch.exp(torch.randn(3, 1, device=device) * sigma_gain_local)
        
        # Si la vocal es /a/, garantizar que Ch0 no sufra atenuación artificial
        if vocal == 'A':
            g_local[0, 0] = torch.clamp(g_local[0, 0], min=0.9)
        elif vocal in ['O', 'U']:
            g_local[2, 0] = torch.clamp(g_local[2, 0], min=0.9)
        elif vocal in ['E', 'I']:
            g_local[1, 0] = torch.clamp(g_local[1, 0], min=0.9)

        # Aplicar transformación
        x_i = X_aug[i] * (g_global * g_local)
        X_aug[i] = R @ x_i

    # Ruido aditivo sutil
    ruido = torch.randn_like(X_aug) * sigma_ruido
    X_aug = X_aug + ruido

    return X_aug

def ajustar_clasificador_train(Z_train, Y_train, seed=42):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=10)
    cluster_train = gmm.fit_predict(Z_train)
    cost_matrix = np.zeros((5, 5))
    y_idx = np.array([VOCALES.index(v) for v in Y_train])
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_train == c) & (y_idx == v))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_train_idx = np.array([cluster_to_vocal[c] for c in cluster_train])
    acc_train = accuracy_score(y_idx, y_pred_train_idx)
    return gmm, cluster_to_vocal, acc_train

def evaluar_clasificador_test(gmm, cluster_to_vocal, Z_test, Y_test):
    cluster_test = gmm.predict(Z_test)
    y_pred_test_idx = np.array([cluster_to_vocal.get(c, 0) for c in cluster_test])
    y_test_idx = np.array([VOCALES.index(v) for v in Y_test])
    acc_test = accuracy_score(y_test_idx, y_pred_test_idx)
    return acc_test, y_pred_test_idx

def ejecutar():
    fijar_semillas(42)
    print("=" * 80)
    print("DATA AUGMENTATION MONTE CARLO FISICO Y CONSERVACION DE SINERGIAS")
    print("GEOMETRIA CANONICA CON VOCALES INTERMEDIAS (/e/ Y /o/ SEPARADAS)")
    print("=" * 80)

    # 1. Cargar Dataset Completo
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

    # 2. Máscara Idéntica de Holdout Estricto
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

    print(f"\nPartición Total:")
    print(f"  Pulsos de Train (Monte Carlo Físico)         : {np.sum(is_train_mask)} pulsos")
    print(f"  Pulsos de Test (evaluación ciega sin tocar) : {np.sum(is_test_mask)} pulsos")

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
        print(f"  {s_nom:7s}: Train = {len(idx_rel_tr):3d} | Test Retenido = {len(idx_rel_te):3d}")

    N_train_total = sum(d['N'] for d in datos_train.values())
    N_test_total = sum(d['N'] for d in datos_test.values())

    # 3. Modelos
    tronco = ParametricConvOrthogonalAE(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'
    )
    adaptadores = {
        s_nom: SubjectAdapter(in_channels=3) for s_nom in sujetos_nombres
    }

    todos_parametros = list(tronco.parameters())
    for adapt in adaptadores.values():
        todos_parametros.extend(list(adapt.parameters()))

    optimizer = optim.Adam(todos_parametros, lr=0.005)
    mse_fn = nn.MSELoss()

    lambda_ortho = 0.05
    lambda_cov = 0.10
    lambda_pot = 0.25  # Firmeza en los atractores de vocales intermedias
    epochs = 350

    print("\n" + "=" * 80)
    print(f"ENTRENAMIENTO FISICO CONSERVATIVO ({epochs} EPOCAS)")
    print("=" * 80)

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

            # Aumentación física que conserva sinergias
            X_aug = aumentar_montecarlo_conservativo(
                X_clean, Y_in, sigma_ang=0.06, sigma_gain_global=0.25, sigma_gain_local=0.08, sigma_ruido=0.015
            )

            x_proj = adapt.adapt_in(X_aug)
            z = tronco.encode(x_proj)
            x_rec_trunk = tronco.decode(z)
            x_rec = adapt.adapt_out(x_rec_trunk)

            # Reconstruir la señal limpia canónica
            loss_rec_s = mse_fn(x_rec, X_clean)
            loss_rec_total = loss_rec_total + loss_rec_s * (d['N'] / N_train_total)
            loss_ortho_total = loss_ortho_total + adapt.ortho_loss()

            # Potencial en espacio latente
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

        loss_total = (
            loss_rec_total +
            lambda_ortho * loss_ortho_total +
            lambda_cov * loss_cov +
            lambda_pot * loss_pot_total
        )

        loss_total.backward()
        optimizer.step()

        dt_epoch = time.time() - t_epoch_start
        if epoch % 50 == 0 or epoch == epochs:
            elapsed = time.time() - t_start
            eta = (epochs - epoch) * (elapsed / epoch)
            pct = (epoch / epochs) * 100.0
            print(f"  Físico Train | Época {epoch:3d}/{epochs} [{pct:5.1f}%] | "
                  f"Rec: {loss_rec_total.item():.4f} | Ortho: {loss_ortho_total.item():.4f} | "
                  f"Cov: {loss_cov.item():.4f} | Pot: {loss_pot_total.item():.4f} | "
                  f"Total: {loss_total.item():.4f} | Paso: {dt_epoch*1000:.1f}ms | ETA: {eta:.1f}s")

    # 4. Evaluación Ciega sobre Señales Reales de Test
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
    gmm_global, mapping_global, acc_tr_global = ajustar_clasificador_train(Z_tr_global, Y_tr_global, seed=42)

    Z_te_global = np.vstack([z_te_dict[s] for s in sujetos_nombres])
    Y_te_global = np.concatenate([datos_test[s]['Y'] for s in sujetos_nombres])
    acc_te_global, y_pred_te_global = evaluar_clasificador_test(gmm_global, mapping_global, Z_te_global, Y_te_global)

    acc_tr_sujeto = {}
    acc_te_sujeto = {}
    y_pred_te_sujeto = {}

    for s_nom in sujetos_nombres:
        acc_tr_s, _ = evaluar_clasificador_test(gmm_global, mapping_global, z_tr_dict[s_nom], datos_train[s_nom]['Y'])
        acc_te_s, y_pred_s = evaluar_clasificador_test(gmm_global, mapping_global, z_te_dict[s_nom], datos_test[s_nom]['Y'])
        acc_tr_sujeto[s_nom] = acc_tr_s
        acc_te_sujeto[s_nom] = acc_te_s
        y_pred_te_sujeto[s_nom] = y_pred_s

    print("\n" + "=" * 80)
    print("RESULTADOS CON MONTE CARLO FISICO CONSERVATIVO: TRAIN VS TEST")
    print("=" * 80)
    print(f"{'Sujeto':<12} | {'Train Acc (%)':<15} | {'Test Acc Fuera de Muestra (%)':<30} | {'Delta':<10}")
    print("-" * 75)
    for s_nom in sujetos_nombres:
        a_tr = acc_tr_sujeto[s_nom] * 100.0
        a_te = acc_te_sujeto[s_nom] * 100.0
        delta = a_te - a_tr
        print(f"{s_nom:<12} | {a_tr:13.2f}% | {a_te:28.2f}% | {delta:+6.2f}%")
    print("-" * 75)
    delta_global = (acc_te_global - acc_tr_global) * 100.0
    print(f"{'GLOBAL':<12} | {acc_tr_global*100.0:13.2f}% | {acc_te_global*100.0:28.2f}% | {delta_global:+6.2f}%")

    print("\nDesglose de Exactitud en TEST por Vocal:")
    print("Vocal | Lucas Test (%) | Santi Test (%) | Candela Test (%) | Global Test (%)")
    print("-" * 75)
    y_te_glob_idx = np.array([VOCALES.index(v) for v in Y_te_global])
    for i, v in enumerate(VOCALES):
        accs = []
        for s_nom in sujetos_nombres:
            y_s_te = datos_test[s_nom]['Y']
            y_s_idx = np.array([VOCALES.index(x) for x in y_s_te])
            y_p_s = y_pred_te_sujeto[s_nom]
            m = (y_s_idx == i)
            acc_v = np.mean(y_p_s[m] == i) * 100.0 if m.sum() > 0 else 0.0
            accs.append(acc_v)
        m_g = (y_te_glob_idx == i)
        acc_g = np.mean(y_pred_te_global[m_g] == i) * 100.0 if m_g.sum() > 0 else 0.0
        print(f" /{v.lower()}/  |     {accs[0]:5.1f}      |     {accs[1]:5.1f}     |      {accs[2]:5.1f}       |     {acc_g:5.1f}")

    # 5. Graficado Oficial
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "validacion_holdout_montecarlo_conservativo.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')

    # Panel 1: Lucas
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    z_tr = z_tr_dict['Lucas']
    y_tr = datos_train['Lucas']['Y']
    z_te = z_te_dict['Lucas']
    y_te = datos_test['Lucas']['Y']
    for v in VOCALES:
        m_tr = (y_tr == v)
        m_te = (y_te == v)
        ax1.scatter(z_tr[m_tr, 0], z_tr[m_tr, 1], c=[COLORES_VOCALES[v]], alpha=0.25, s=25, edgecolors='none')
        ax1.scatter(z_te[m_te, 0], z_te[m_te, 1], c=[COLORES_VOCALES[v]], marker='o', alpha=0.85, s=60, edgecolors='black', linewidth=0.5, label=f"Test /{v.lower()}/")
        c_te = np.mean(z_te[m_te], axis=0) if m_te.sum() > 0 else [0, 0]
        ax1.scatter(c_te[0], c_te[1], c=[COLORES_VOCALES[v]], s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax1.annotate(f"/{v.lower()}/", (c_te[0], c_te[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"Lucas - Holdout Tomas T4-T7 - Test Acc: {acc_te_sujeto['Lucas']*100:.2f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Z1", fontsize=11)
    ax1.set_ylabel("Z2", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Santi
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    z_tr = z_tr_dict['Santi']
    y_tr = datos_train['Santi']['Y']
    z_te = z_te_dict['Santi']
    y_te = datos_test['Santi']['Y']
    for v in VOCALES:
        m_tr = (y_tr == v)
        m_te = (y_te == v)
        ax2.scatter(z_tr[m_tr, 0], z_tr[m_tr, 1], c=[COLORES_VOCALES[v]], alpha=0.25, s=25, edgecolors='none')
        ax2.scatter(z_te[m_te, 0], z_te[m_te, 1], c=[COLORES_VOCALES[v]], marker='s', alpha=0.85, s=60, edgecolors='black', linewidth=0.5, label=f"Test /{v.lower()}/")
        c_te = np.mean(z_te[m_te], axis=0) if m_te.sum() > 0 else [0, 0]
        ax2.scatter(c_te[0], c_te[1], c=[COLORES_VOCALES[v]], marker='s', s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax2.annotate(f"/{v.lower()}/", (c_te[0], c_te[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"Santi - Holdout Tomas P3-P4 - Test Acc: {acc_te_sujeto['Santi']*100:.2f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Z1", fontsize=11)
    ax2.set_ylabel("Z2", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel 3: Candela
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    z_tr = z_tr_dict['Candela']
    y_tr = datos_train['Candela']['Y']
    z_te = z_te_dict['Candela']
    y_te = datos_test['Candela']['Y']
    for v in VOCALES:
        m_tr = (y_tr == v)
        m_te = (y_te == v)
        ax3.scatter(z_tr[m_tr, 0], z_tr[m_tr, 1], c=[COLORES_VOCALES[v]], alpha=0.25, s=25, edgecolors='none')
        ax3.scatter(z_te[m_te, 0], z_te[m_te, 1], c=[COLORES_VOCALES[v]], marker='^', alpha=0.85, s=60, edgecolors='black', linewidth=0.5, label=f"Test /{v.lower()}/")
        c_te = np.mean(z_te[m_te], axis=0) if m_te.sum() > 0 else [0, 0]
        ax3.scatter(c_te[0], c_te[1], c=[COLORES_VOCALES[v]], marker='^', s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax3.annotate(f"/{v.lower()}/", (c_te[0], c_te[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax3.set_title(f"Candela - Holdout Dia 2026-09-01 - Test Acc: {acc_te_sujeto['Candela']*100:.2f}%", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Z1", fontsize=11)
    ax3.set_ylabel("Z2", fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='upper right')

    # Panel 4: Test Global Unificado
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    
    for v in VOCALES:
        c_t = CENTROIDES_CANONICOS[v].numpy()
        circle = plt.Circle((c_t[0], c_t[1]), 0.45, color=COLORES_VOCALES[v], fill=False, linestyle='--', linewidth=1.5, alpha=0.6)
        ax4.add_patch(circle)

    for v in VOCALES:
        m_L_te = (datos_test['Lucas']['Y'] == v)
        m_S_te = (datos_test['Santi']['Y'] == v)
        m_C_te = (datos_test['Candela']['Y'] == v)
        ax4.scatter(z_te_dict['Lucas'][m_L_te, 0], z_te_dict['Lucas'][m_L_te, 1], c=[COLORES_VOCALES[v]], marker='o', s=35, alpha=0.7, edgecolors='none')
        ax4.scatter(z_te_dict['Santi'][m_S_te, 0], z_te_dict['Santi'][m_S_te, 1], c=[COLORES_VOCALES[v]], marker='s', s=35, alpha=0.7, edgecolors='none')
        ax4.scatter(z_te_dict['Candela'][m_C_te, 0], z_te_dict['Candela'][m_C_te, 1], c=[COLORES_VOCALES[v]], marker='^', s=35, alpha=0.7, edgecolors='none')
    
    for v in VOCALES:
        m_tot_te = (Y_te_global == v)
        c_tot_te = np.mean(Z_te_global[m_tot_te], axis=0) if m_tot_te.sum() > 0 else [0, 0]
        ax4.scatter(c_tot_te[0], c_tot_te[1], c=[COLORES_VOCALES[v]], s=180, edgecolors='black', linewidth=2.0, zorder=7)
        ax4.annotate(f"/{v.lower()}/", (c_tot_te[0], c_tot_te[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax4.set_title(f"Variedad Test con MC Conservativo - {N_test_total} Pulsos - Test Acc: {acc_te_global*100:.2f}%", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Z1", fontsize=11)
    ax4.set_ylabel("Z2", fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Guardado] Gráfico Monte Carlo Conservativo: {fig_path}")

    save_dir = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    json_path = os.path.join(save_dir, "metricas_montecarlo_conservativo.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'acc_train_global': acc_tr_global,
            'acc_test_global': acc_te_global,
            'acc_train_por_sujeto': acc_tr_sujeto,
            'acc_test_por_sujeto': acc_te_sujeto,
            'delta_global': delta_global
        }, f, indent=4)
    print(f"[Guardado] Métricas JSON: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar()
