# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Calibración de Ganancia Relativa Inter-Canal (G = diag(g0, g1, g2))
#              acoplada con rotación SO(3) para desacoplar /e/-/i/ y /o/-/u/.
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
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

CENTROIDES_CANONICOS_3D = {
    'A': torch.tensor([0.0, 2.0, 0.0], dtype=torch.float32),
    'E': torch.tensor([1.4, 1.0, 0.0], dtype=torch.float32),
    'I': torch.tensor([2.0, 0.0, 0.0], dtype=torch.float32),
    'O': torch.tensor([0.0, 1.0, 1.4], dtype=torch.float32),
    'U': torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32),
}

class SubjectAdapterStandard(nn.Module):
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

# Adaptador Desacoplado: Ganancia Relativa Diagonal G + Rotación Ortogonal R
class SubjectAdapterWithGain(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # Parámetro log_gain para g_c = exp(log_gain_c) > 0
        self.log_gain = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.rot_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False)
        self.bias_in = nn.Parameter(torch.zeros(1, in_channels, 1))
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)

        with torch.no_grad():
            self.rot_in.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.bias.zero_()

    def adapt_in(self, x):
        # 1. Ecualización de Ganancia por Canal: x_gain = x * exp(log_gain)
        g = torch.exp(self.log_gain)
        x_scaled = x * g
        # 2. Rotación Ortogonal SO(3)
        return self.rot_in(x_scaled) + self.bias_in

    def adapt_out(self, x_rec):
        return self.read_out(x_rec)

    def ortho_loss(self):
        W = self.rot_in.weight.squeeze(-1)
        diff = W @ W.T - torch.eye(3, device=W.device)
        return torch.norm(diff, p='fro') ** 2

    def gain_regularization(self):
        # Mantiene la ganancia relativa promedio cercana a 1 (log_gain medio ~ 0)
        # y penaliza variaciones extremas
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
    print("ENSAYO: CALIBRACION DE GANANCIA RELATIVA INTER-CANAL (DIAG(G) + SO(3))")
    print("OBJETIVO: DESACOPLAR /e/-/i/ Y /o/-/u/ EN SUJETO INEDITO PETRA")
    print("=" * 80)

    # 1. Cargar Modelo 3D Congelado
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

    # 2. Cargar Train Tri-Sujeto y Ajustar LDA Referencia
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
    adaptadores = {s: SubjectAdapterStandard(in_channels=3) for s in sujetos_nombres}
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

    y_p1_true = np.array([VOCALES.index(v) for v in Y_p1])
    y_p2_true = np.array([VOCALES.index(v) for v in Y_p2])

    X_p1_t = torch.tensor(X_p1, dtype=torch.float32)
    X_p2_t = torch.tensor(X_p2, dtype=torch.float32)

    # --------------------------------------------------------------------------
    # MODELO A: Baseline Ortogonal Puro SO(3)
    # --------------------------------------------------------------------------
    adapt_base = SubjectAdapterStandard(in_channels=3)
    opt_base = optim.Adam(adapt_base.parameters(), lr=0.01)
    mse_fn = nn.MSELoss()

    for ep in range(120):
        opt_base.zero_grad()
        x_proj = adapt_base.adapt_in(X_p1_t)
        z_p = tronco.encode(x_proj)
        x_rec = adapt_base.adapt_out(tronco.decode(z_p))
        loss_rec = mse_fn(x_rec, X_p1_t)
        loss_ortho = adapt_base.ortho_loss()
        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_p1):
            mask_v = (Y_p1 == v)
            if mask_v.sum() > 0:
                z_v = z_p[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))
        loss = loss_rec + 0.05 * loss_ortho + 0.20 * loss_pot
        loss.backward()
        opt_base.step()

    adapt_base.eval()
    with torch.no_grad():
        Z_p2_base = tronco.encode(adapt_base.adapt_in(X_p2_t)).numpy()
    y_p2_pred_base = lda_tri.predict(Z_p2_base)
    acc_base = accuracy_score(y_p2_true, y_p2_pred_base)

    # --------------------------------------------------------------------------
    # MODELO B: Adaptador con Calibración de Ganancia Relativa diag(g0, g1, g2)
    # --------------------------------------------------------------------------
    adapt_gain = SubjectAdapterWithGain(in_channels=3)
    opt_gain = optim.Adam(adapt_gain.parameters(), lr=0.01)

    for ep in range(150):
        opt_gain.zero_grad()
        x_proj = adapt_gain.adapt_in(X_p1_t)
        z_p = tronco.encode(x_proj)
        x_rec = adapt_gain.adapt_out(tronco.decode(z_p))
        loss_rec = mse_fn(x_rec, X_p1_t)
        loss_ortho = adapt_gain.ortho_loss()
        loss_g_reg = adapt_gain.gain_regularization()

        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_p1):
            mask_v = (Y_p1 == v)
            if mask_v.sum() > 0:
                z_v = z_p[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))

        loss = loss_rec + 0.05 * loss_ortho + 0.05 * loss_g_reg + 0.25 * loss_pot
        loss.backward()
        opt_gain.step()

    adapt_gain.eval()
    with torch.no_grad():
        g_final = torch.exp(adapt_gain.log_gain).squeeze().numpy()
        Z_p2_gain = tronco.encode(adapt_gain.adapt_in(X_p2_t)).numpy()

    y_p2_pred_gain = lda_tri.predict(Z_p2_gain)
    acc_gain = accuracy_score(y_p2_true, y_p2_pred_gain)

    knn_gain = KNeighborsClassifier(n_neighbors=7)
    knn_gain.fit(Z_tr_global, y_tr_idx)
    acc_knn_gain = knn_gain.score(Z_p2_gain, y_p2_true)

    cm_base = confusion_matrix(y_p2_true, y_p2_pred_base, labels=range(5))
    cm_gain = confusion_matrix(y_p2_true, y_p2_pred_gain, labels=range(5))

    print("\n" + "=" * 80)
    print("RESULTADOS COMPARATIVOS EN TEST CIEGO PETRA DIA 2 (N=168):")
    print("=" * 80)
    print(f"Ganancias relativas aprendidas en Petra D1: g = [Ch0={g_final[0]:.2f}, Ch1={g_final[1]:.2f}, Ch2={g_final[2]:.2f}]")
    print(f"Ratio de Apertura Ch0 / Ch1 (Digástrico / Cigomático): {g_final[0]/g_final[1]:.2f}")
    print(f"Ratio de Apertura Ch0 / Ch2 (Digástrico / Orbicular):  {g_final[0]/g_final[2]:.2f}")
    print("-" * 80)
    print(f"  A) Baseline SO(3) Puro:               LDA = {acc_base*100:6.2f}%")
    print(f"  B) Con Ganancia Relativa diag(G)*SO(3): LDA = {acc_gain*100:6.2f}% | k-NN = {acc_knn_gain*100:6.2f}%")
    print(f"  Variación Global:                     {'+' if acc_gain >= acc_base else ''}{(acc_gain - acc_base)*100:.2f} puntos porcentuales")

    print("\nDESGLOSE DE EXACTITUD POR VOCAL EN PETRA DIA 2:")
    print(f"{'Vocal':<6} | {'N':<5} | {'Base SO(3) (%)':<16} | {'Con Ganancia diag(G) (%)':<24} | {'Diferencia':<12}")
    print("-" * 75)
    for i, v in enumerate(VOCALES):
        n_v = np.sum(y_p2_true == i)
        if n_v > 0:
            acc_b = cm_base[i, i] / n_v * 100.0
            acc_g = cm_gain[i, i] / n_v * 100.0
            diff = acc_g - acc_b
            print(f" /{v.lower()}/  | {n_v:^5d} | {acc_b:^16.1f} | {acc_g:^24.1f} | {'+' if diff>=0 else ''}{diff:5.1f}%")

    # 4. Graficar Comparativa de Matrices y Espacio Latente
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "comparativa_calibracion_ganancia_relativa_petra.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')

    # Panel (0, 0): Matriz Confusión Baseline SO(3)
    ax0 = axes[0, 0]
    cm_b_norm = cm_base.astype('float') / np.maximum(cm_base.sum(axis=1)[:, np.newaxis], 1)
    ax0.imshow(cm_b_norm, cmap='Blues', vmin=0, vmax=1)
    for r in range(5):
        for c in range(5):
            pct = cm_b_norm[r, c] * 100.0
            col = 'white' if cm_b_norm[r, c] > 0.5 else 'black'
            ax0.text(c, r, f"{pct:.1f}%\n({cm_base[r, c]})", ha='center', va='center', color=col, fontsize=9, fontweight='bold')
    ax0.set_xticks(range(5))
    ax0.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax0.set_yticks(range(5))
    ax0.set_yticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax0.set_title(f"A) Baseline SO(3) Puro (Sin Ganancia)\nTest Ciego Día 2: {acc_base*100:.1f}%", fontweight='bold', fontsize=11)
    ax0.set_ylabel("Etiqueta Real", fontweight='bold')

    # Panel (0, 1): Matriz Confusión Con Ganancia diag(G)
    ax1 = axes[0, 1]
    cm_g_norm = cm_gain.astype('float') / np.maximum(cm_gain.sum(axis=1)[:, np.newaxis], 1)
    ax1.imshow(cm_g_norm, cmap='Greens', vmin=0, vmax=1)
    for r in range(5):
        for c in range(5):
            pct = cm_g_norm[r, c] * 100.0
            col = 'white' if cm_g_norm[r, c] > 0.5 else 'black'
            ax1.text(c, r, f"{pct:.1f}%\n({cm_gain[r, c]})", ha='center', va='center', color=col, fontsize=9, fontweight='bold')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
    ax1.set_title(f"B) Con Ganancia Relativa diag(G)*SO(3)\nTest Ciego Día 2: {acc_gain*100:.1f}%", fontweight='bold', fontsize=11)

    # Panel (1, 0): Dispersión Z1-Z2 Baseline
    ax2 = axes[1, 0]
    for v in np.unique(Y_p2):
        m = (Y_p2 == v)
        ax2.scatter(Z_p2_base[m, 0], Z_p2_base[m, 1], color=COLORES_VOCALES[v], s=25, alpha=0.6, label=f"/{v.lower()}/")
    for v in VOCALES:
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax2.scatter([c[0]], [c[1]], color='black', marker='D', s=50, edgecolors='white', linewidth=1.2)
    ax2.set_title("Z1 (Sonrisa) vs Z2 (Apertura) - Baseline", fontweight='bold', fontsize=10)
    ax2.set_xlabel("Z1", fontsize=9)
    ax2.set_ylabel("Z2", fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='lower left', fontsize=8)

    # Panel (1, 1): Dispersión Z1-Z2 Con Ganancia
    ax3 = axes[1, 1]
    for v in np.unique(Y_p2):
        m = (Y_p2 == v)
        ax3.scatter(Z_p2_gain[m, 0], Z_p2_gain[m, 1], color=COLORES_VOCALES[v], s=25, alpha=0.6, label=f"/{v.lower()}/")
    for v in VOCALES:
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax3.scatter([c[0]], [c[1]], color='black', marker='D', s=50, edgecolors='white', linewidth=1.2)
    ax3.set_title(f"Z1 vs Z2 - Con Ganancia Relativa (g0={g_final[0]:.2f}, g1={g_final[1]:.2f})", fontweight='bold', fontsize=10)
    ax3.set_xlabel("Z1", fontsize=9)
    ax3.set_ylabel("Z2", fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nFigura comparativa guardada en: {fig_path}")

if __name__ == '__main__':
    main()
