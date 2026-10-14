# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Modelo unificado multi-sujeto basado en invariancia de gauge y
#              potenciales armónicos atractores canónicos en el espacio latente.
#              Entrenamiento conjunto de tronco convolucional y Read-in layers.
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

from EMG_desarrollo.deep_learning.grid_search_conv_ortogonal import (
    cargar_datos_lucas,
    vocales,
    vocal_to_idx
)

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

# Anclaje Canónico de Físico Teórico: Geometría de Cruz Latente y Pentágono Fonético
# /a/: +Y (12 en punto, apertura mandibular máxima)
# /e/, /i/: +X > 0 (sonrisas)
# /o/, /u/: -X < 0 (protrusiones)
CENTROIDES_CANONICOS = {
    'A': torch.tensor([0.0, 1.8], dtype=torch.float32),
    'E': torch.tensor([1.5, 0.6], dtype=torch.float32),
    'I': torch.tensor([1.8, -0.6], dtype=torch.float32),
    'O': torch.tensor([-1.2, 0.5], dtype=torch.float32),
    'U': torch.tensor([-1.8, -0.5], dtype=torch.float32),
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
    """Adaptador de Read-in y Read-out 3x3 por sujeto con inicialización identidad"""
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

def evaluar_gmm_hungaro(Z_np, Y_np, seed=42):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=10)
    cluster_preds = gmm.fit_predict(Z_np)
    cost_matrix = np.zeros((5, 5))
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_idx = np.array([cluster_to_vocal[c] for c in cluster_preds])
    acc_global = accuracy_score(y_idx, y_pred_idx)
    return acc_global, y_pred_idx, gmm

def evaluar_con_gmm_fijo(gmm, Z_np, Y_np):
    cluster_preds = gmm.predict(Z_np)
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    cost_matrix = np.zeros((5, 5))
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    y_pred_idx = np.array([cluster_to_vocal.get(c, 0) for c in cluster_preds])
    acc = accuracy_score(y_idx, y_pred_idx)
    return acc, y_pred_idx

def ejecutar():
    fijar_semillas(42)
    print("=" * 80)
    print("MODELO MULTI-SUJETO UNIFICADO - FISICA TEORICA DEL MACHINE LEARNING")
    print("INVARIANZA DE GAUGE + POZOS DE POTENCIAL ARMONICOS + TRONCO COMPARTIDO")
    print("=" * 80)

    # 1. Cargar y acondicionar los tres sujetos
    npz_path = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    data = np.load(npz_path, allow_pickle=True)
    X_raw_all = data['X_raw']
    Y_all = data['Y_raw']
    sujetos_all = data['sujetos_raw']
    tomas_all = data['tomas_raw']

    def extraer_sesion(t):
        p = str(t).split('_')
        return p[1].upper() if len(p) > 1 else 'S1'

    sesiones_all = np.array([extraer_sesion(t) for t in tomas_all])

    # Filtrar por sujeto y normalizar con el pipeline estándar
    sujetos_nombres = ['Lucas', 'Santi', 'Candela']
    datos_sujetos = {}

    for s_nom in sujetos_nombres:
        mask = (sujetos_all == s_nom)
        X_s_cond = normalizar_sesiones_conv(X_raw_all[mask], sesiones_all[mask], n_ch=3, n_pts=20)
        Y_s = Y_all[mask]
        datos_sujetos[s_nom] = {
            'X_t': torch.tensor(X_s_cond, dtype=torch.float32),
            'Y': Y_s,
            'N': len(Y_s)
        }
        print(f"  Sujeto {s_nom:7s}: {len(Y_s)} pulsos condicionados")

    N_total = sum(d['N'] for d in datos_sujetos.values())
    print(f"Total del corpus integrado: {N_total} pulsos")

    # 2. Inicializar el Tronco Convolucional Compartido y Adaptadores de Sujeto
    tronco = ParametricConvOrthogonalAE(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'
    )
    
    adaptadores = {
        s_nom: SubjectAdapter(in_channels=3) for s_nom in sujetos_nombres
    }

    # Colección de parámetros para optimización conjunta
    todos_parametros = list(tronco.parameters())
    for adapt in adaptadores.values():
        todos_parametros.extend(list(adapt.parameters()))

    optimizer = optim.Adam(todos_parametros, lr=0.005)
    mse_fn = nn.MSELoss()

    # Hiperparámetros del Hamiltoniano
    lambda_ortho = 0.05
    lambda_cov = 0.10
    lambda_pot = 0.20
    epochs = 300

    print("\n" + "=" * 80)
    print(f"ENTRENAMIENTO CONJUNTO (JOINT TRAINING) - {epochs} EPOCAS")
    print(f"Ponderaciones: Rec=1.0 | Ortho={lambda_ortho} | Cov={lambda_cov} | Potencial={lambda_pot}")
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
            d = datos_sujetos[s_nom]
            adapt = adaptadores[s_nom]
            X_in = d['X_t']
            Y_in = d['Y']

            # Pase hacia adelante por adaptador y tronco
            x_proj = adapt.adapt_in(X_in)
            z = tronco.encode(x_proj)
            x_rec_trunk = tronco.decode(z)
            x_rec = adapt.adapt_out(x_rec_trunk)

            # Reconstrucción
            loss_rec_s = mse_fn(x_rec, X_in)
            loss_rec_total = loss_rec_total + loss_rec_s * (d['N'] / N_total)

            # Ortogonalidad de Read-in
            loss_ortho_total = loss_ortho_total + adapt.ortho_loss()

            # Potencial Atractor Canónico por Vocal
            loss_pot_s = torch.tensor(0.0)
            for v in VOCALES:
                mask_v = (Y_in == v)
                if mask_v.sum() > 0:
                    z_v = z[mask_v]
                    target_v = CENTROIDES_CANONICOS[v]
                    loss_pot_s = loss_pot_s + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
            loss_pot_total = loss_pot_total + (loss_pot_s / 5.0) * (d['N'] / N_total)

            z_list.append(z)

        # Decorrelación de Covarianza en el espacio latente global unificado
        Z_global = torch.cat(z_list, dim=0)
        Z_cent = Z_global - torch.mean(Z_global, dim=0, keepdim=True)
        cov_z = (Z_cent.T @ Z_cent) / (N_total - 1)
        loss_cov = torch.norm(cov_z - torch.eye(2), p='fro') ** 2

        # Hamiltoniano total
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
            print(f"  Época {epoch:3d}/{epochs} [{pct:5.1f}%] | "
                  f"Rec: {loss_rec_total.item():.4f} | Ortho: {loss_ortho_total.item():.4f} | "
                  f"Cov: {loss_cov.item():.4f} | Pot: {loss_pot_total.item():.4f} | "
                  f"Total: {loss_total.item():.4f} | Paso: {dt_epoch*1000:.1f}ms | ETA: {eta:.1f}s")

    # 3. Evaluación de Variedades y Exactitud
    tronco.eval()
    for adapt in adaptadores.values():
        adapt.eval()

    proyecciones_z = {}
    with torch.no_grad():
        for s_nom in sujetos_nombres:
            d = datos_sujetos[s_nom]
            adapt = adaptadores[s_nom]
            x_proj = adapt.adapt_in(d['X_t'])
            z_s = tronco.encode(x_proj).numpy()
            proyecciones_z[s_nom] = z_s

    # Evaluación GMM Húngaro en espacio latente unificado
    Z_unif = np.vstack([proyecciones_z[s] for s in sujetos_nombres])
    Y_unif = np.concatenate([datos_sujetos[s]['Y'] for s in sujetos_nombres])
    acc_global, y_pred_unif, gmm_unificado = evaluar_gmm_hungaro(Z_unif, Y_unif, seed=42)

    # Exactitud por sujeto usando el GMM unificado
    acc_por_sujeto = {}
    y_pred_por_sujeto = {}
    idx_inicio = 0
    for s_nom in sujetos_nombres:
        n_s = datos_sujetos[s_nom]['N']
        z_s = proyecciones_z[s_nom]
        y_s = datos_sujetos[s_nom]['Y']
        acc_s, y_pred_s = evaluar_con_gmm_fijo(gmm_unificado, z_s, y_s)
        acc_por_sujeto[s_nom] = acc_s
        y_pred_por_sujeto[s_nom] = y_pred_s

    print("\n" + "=" * 80)
    print("RESULTADOS FINALES DEL MODELO UNIFICADO MULTI-SUJETO")
    print("=" * 80)
    for s_nom in sujetos_nombres:
        print(f"Exactitud {s_nom:7s} sobre Variedad Común : {acc_por_sujeto[s_nom]*100:.2f}%")
    print(f"Exactitud Global Unificada (3 Sujetos, 1658 pulsos): {acc_global*100:.2f}%")

    # Matrices Read-in aprendidas
    print("\nMatrices Read-in (Operadores de Gauge Local en SO(3)):")
    for s_nom in sujetos_nombres:
        adapt = adaptadores[s_nom]
        W = adapt.read_in.weight.squeeze(-1).detach().numpy()
        b = adapt.read_in.bias.detach().numpy()
        det_W = np.linalg.det(W)
        print(f"\nRead-in {s_nom} (det = {det_W:+.4f}):")
        for row in W:
            print("  [ " + "  ".join([f"{val:+.3f}" for val in row]) + " ]")
        print("  b: [ " + "  ".join([f"{val:+.3f}" for val in b]) + " ]")

    # Desglose de Exactitud por Vocal
    print("\nDesglose de Exactitud por Vocal:")
    print("Vocal | Lucas (%) | Santi (%) | Candela (%) | Global Unificado (%)")
    print("-" * 68)
    for i, v in enumerate(VOCALES):
        accs = []
        for s_nom in sujetos_nombres:
            y_s = datos_sujetos[s_nom]['Y']
            y_s_idx = np.array([VOCALES.index(x) for x in y_s])
            y_p_s = y_pred_por_sujeto[s_nom]
            m = (y_s_idx == i)
            acc_v = np.mean(y_p_s[m] == i) * 100.0 if m.sum() > 0 else 0.0
            accs.append(acc_v)
        
        y_u_idx = np.array([VOCALES.index(x) for x in Y_unif])
        m_u = (y_u_idx == i)
        acc_u = np.mean(y_pred_unif[m_u] == i) * 100.0 if m_u.sum() > 0 else 0.0
        print(f" /{v.lower()}/  |   {accs[0]:5.1f}   |   {accs[1]:5.1f}   |    {accs[2]:5.1f}    |         {acc_u:5.1f}")

    # 4. Graficado Oficial Cuatro Paneles (Sin Paréntesis en Títulos)
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "variedad_unificada_fisica_teorica_multisujeto.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')

    # Panel 1: Lucas
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    Z_L = proyecciones_z['Lucas']
    Y_L = datos_sujetos['Lucas']['Y']
    for v in VOCALES:
        m = (Y_L == v)
        ax1.scatter(Z_L[m, 0], Z_L[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='none')
        c_v = np.mean(Z_L[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"Lucas - Tronco Común y Read-in Específica - GMM Acc: {acc_por_sujeto['Lucas']*100:.2f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Z1", fontsize=11)
    ax1.set_ylabel("Z2", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Santi
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    Z_S = proyecciones_z['Santi']
    Y_S = datos_sujetos['Santi']['Y']
    for v in VOCALES:
        m = (Y_S == v)
        ax2.scatter(Z_S[m, 0], Z_S[m, 1], c=[COLORES_VOCALES[v]], marker='s', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_S[m], axis=0)
        ax2.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='s', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax2.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"Santi - Tronco Común y Read-in Específica - GMM Acc: {acc_por_sujeto['Santi']*100:.2f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Z1", fontsize=11)
    ax2.set_ylabel("Z2", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel 3: Candela
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    Z_C = proyecciones_z['Candela']
    Y_C = datos_sujetos['Candela']['Y']
    for v in VOCALES:
        m = (Y_C == v)
        ax3.scatter(Z_C[m, 0], Z_C[m, 1], c=[COLORES_VOCALES[v]], marker='^', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_C[m], axis=0)
        ax3.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='^', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax3.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax3.set_title(f"Candela - Tronco Común y Read-in Específica - GMM Acc: {acc_por_sujeto['Candela']*100:.2f}%", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Z1", fontsize=11)
    ax3.set_ylabel("Z2", fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='upper right')

    # Panel 4: Variedad Unificada Total
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    
    # Dibujar pozos canónicos
    for v in VOCALES:
        c_t = CENTROIDES_CANONICOS[v].numpy()
        circle = plt.Circle((c_t[0], c_t[1]), 0.45, color=COLORES_VOCALES[v], fill=False, linestyle='--', linewidth=1.5, alpha=0.6)
        ax4.add_patch(circle)

    for v in VOCALES:
        m_L = (Y_L == v)
        m_S = (Y_S == v)
        m_C = (Y_C == v)
        ax4.scatter(Z_L[m_L, 0], Z_L[m_L, 1], c=[COLORES_VOCALES[v]], marker='o', s=25, alpha=0.4, edgecolors='none')
        ax4.scatter(Z_S[m_S, 0], Z_S[m_S, 1], c=[COLORES_VOCALES[v]], marker='s', s=25, alpha=0.4, edgecolors='none')
        ax4.scatter(Z_C[m_C, 0], Z_C[m_C, 1], c=[COLORES_VOCALES[v]], marker='^', s=25, alpha=0.4, edgecolors='none')
    
    for v in VOCALES:
        m_tot = (Y_unif == v)
        c_tot = np.mean(Z_unif[m_tot], axis=0)
        ax4.scatter(c_tot[0], c_tot[1], c=[COLORES_VOCALES[v]], s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax4.annotate(f"/{v.lower()}/", (c_tot[0], c_tot[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax4.set_title(f"Variedad Unificada Tri-Sujeto - 1658 Pulsos - Global Acc: {acc_global*100:.2f}%", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Z1", fontsize=11)
    ax4.set_ylabel("Z2", fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Guardado] Gráfico de Variedad Unificada: {fig_path}")

    # 5. Guardar Checkpoint y Métricas
    save_dir = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "modelo_unificado_fisica_teorica_multisujeto.pt")
    torch.save({
        'tronco_state_dict': tronco.state_dict(),
        'adaptadores_state_dict': {s: adaptadores[s].state_dict() for s in sujetos_nombres},
        'acc_global': acc_global,
        'acc_por_sujeto': acc_por_sujeto,
        'centroides_canonicos': {v: CENTROIDES_CANONICOS[v].tolist() for v in VOCALES}
    }, ckpt_path)
    print(f"[Guardado] Checkpoint Completo: {ckpt_path}")

    json_path = os.path.join(save_dir, "metricas_fisica_teorica_multisujeto.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'acc_global': acc_global,
            'acc_por_sujeto': acc_por_sujeto,
            'epochs': epochs,
            'lambda_ortho': lambda_ortho,
            'lambda_cov': lambda_cov,
            'lambda_pot': lambda_pot
        }, f, indent=4)
    print(f"[Guardado] Métricas JSON: {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar()
