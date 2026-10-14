# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Prueba Extrema de Transferencia hacia Sujeto Inédito (Petra)
#              sobre el Autoencoder 3D Tri-Sujeto congelado.
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

def ejecutar():
    fijar_semillas(42)
    print("=" * 80)
    print("PRUEBA EXTREMA: EVALUACION EN SUJETO INEDITO PETRA (ZERO-SHOT Y TRANSFERENCIA 3D)")
    print("=" * 80)

    # 1. Cargar Modelo 3D Entrenado
    p_ckpt = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/modelo_trisujeto_3d_gauge.pt"
    )
    if not os.path.exists(p_ckpt):
        print(f"Error: No se encontro el checkpoint en {p_ckpt}")
        return

    ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
    tronco = ParametricConvOrthogonalAE3D(
        in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=3, act_name='tanh'
    )
    tronco.load_state_dict(ckpt['tronco'])
    tronco.eval()
    print("Tronco Convolucional 3D cargado con exito.")

    # 2. Cargar Datos Tri-Sujeto (para ajustar los clasificadores LDA y k-NN)
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

    # Reconstruir adaptadores de los sujetos de entrenamiento
    sujetos_nombres = ['Lucas', 'Santi', 'Candela']
    adaptadores = {s: SubjectAdapter(in_channels=3) for s in sujetos_nombres}
    for s in sujetos_nombres:
        adaptadores[s].load_state_dict(ckpt['adaptadores'][s])
        adaptadores[s].eval()

    # Extraer Z de entrenamiento
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

    nc_tri = NearestCentroid()
    nc_tri.fit(Z_tr_global, y_tr_idx)

    knn_tri = KNeighborsClassifier(n_neighbors=7)
    knn_tri.fit(Z_tr_global, y_tr_idx)
    print(f"Clasificadores entrenados sobre {len(Z_tr_global)} pulsos del modelo 3D (Lucas, Santi, Candela).")

    # 3. Cargar Datos de Petra (Día 1 y Día 2)
    dir_grid = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_d1 = os.path.join(dir_grid, "dataset_petra_dia1_20260821_conv2d.npz")
    p_d2 = os.path.join(dir_grid, "dataset_petra_dia2_20260828_conv2d.npz")

    d1 = np.load(p_d1, allow_pickle=True)
    d2 = np.load(p_d2, allow_pickle=True)

    X_p1, Y_p1 = d1['X_clean'], d1['Y_clean']
    X_p2, Y_p2 = d2['X_clean'], d2['Y_clean']

    # Asegurar formato (N, 3, 20)
    if X_p1.ndim == 2:
        X_p1 = X_p1.reshape(-1, 3, 20)
    if X_p2.ndim == 2:
        X_p2 = X_p2.reshape(-1, 3, 20)

    X_petra_all = np.vstack([X_p1, X_p2])
    Y_petra_all = np.concatenate([Y_p1, Y_p2])
    dias_petra = np.array(['D1'] * len(Y_p1) + ['D2'] * len(Y_p2))

    print(f"\nDatos de Petra cargados:")
    print(f"  Día 1 (2026-08-21): {len(Y_p1)} pulsos | Vocales: {np.unique(Y_p1)}")
    print(f"  Día 2 (2026-08-28): {len(Y_p2)} pulsos | Vocales: {np.unique(Y_p2)}")
    print(f"  Total Petra: {len(Y_petra_all)} pulsos.")

    # 4. Modo A: Zero-Shot Puro (Sin Adaptador, W=I)
    X_petra_tensor = torch.tensor(X_petra_all, dtype=torch.float32)
    with torch.no_grad():
        Z_petra_zero = tronco.encode(X_petra_tensor).numpy()

    y_petra_idx = np.array([VOCALES.index(v) for v in Y_petra_all])

    acc_zero_lda = lda_tri.score(Z_petra_zero, y_petra_idx)
    acc_zero_nc = nc_tri.score(Z_petra_zero, y_petra_idx)
    acc_zero_knn = knn_tri.score(Z_petra_zero, y_petra_idx)

    print("\n" + "=" * 80)
    print("A) RESULTADOS ZERO-SHOT PURO EN PETRA (SIN ADAPTADOR, W = Identidad):")
    print("=" * 80)
    print(f"  LDA Tri-Sujeto:           {acc_zero_lda*100:6.2f}% (Azar: 20.0%)")
    print(f"  Nearest Centroid:         {acc_zero_nc*100:6.2f}%")
    print(f"  k-NN (k=7):               {acc_zero_knn*100:6.2f}%")

    print("\n  Desglose Zero-Shot por Día y Vocal:")
    for dia in ['D1', 'D2']:
        m_dia = (dias_petra == dia)
        acc_d = lda_tri.score(Z_petra_zero[m_dia], y_petra_idx[m_dia])
        print(f"    Petra {dia} Global LDA: {acc_d*100:5.1f}% (N={np.sum(m_dia)})")

    # 5. Modo B: Calibración Rápida de Adaptador SO(3) de Petra
    # Usamos Día 1 (86 pulsos) como calibración ultrarrápida (solo 9 pesos W_in y 9 W_out)
    # y testeamos ciego en Día 2 (168 pulsos, 1 semana después).
    print("\n" + "=" * 80)
    print("B) TRANSFERENCIA INTER-DIA CALIBRADA: CALIBRACION EN D1 -> TEST CIEGO EN D2")
    print("=" * 80)

    adapt_petra = SubjectAdapter(in_channels=3)
    opt_petra = optim.Adam(adapt_petra.parameters(), lr=0.01)
    mse_fn = nn.MSELoss()

    X_p1_t = torch.tensor(X_p1, dtype=torch.float32)
    t0_cal = time.time()
    epochs_cal = 120
    for ep in range(1, epochs_cal + 1):
        opt_petra.zero_grad()
        x_proj = adapt_petra.adapt_in(X_p1_t)
        z_p = tronco.encode(x_proj)
        x_rec_t = tronco.decode(z_p)
        x_rec = adapt_petra.adapt_out(x_rec_t)

        loss_rec = mse_fn(x_rec, X_p1_t)
        loss_ortho = adapt_petra.ortho_loss()

        # Guía armónica suave en los centroides de Petra D1
        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_p1):
            mask_v = (Y_p1 == v)
            if mask_v.sum() > 0:
                z_v = z_p[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))

        loss_cal = loss_rec + 0.05 * loss_ortho + 0.20 * loss_pot
        loss_cal.backward()
        opt_petra.step()

    dt_cal = time.time() - t0_cal
    print(f"Calibración de adaptador de Petra convergida en {dt_cal:.2f}s ({epochs_cal} épocas).")

    adapt_petra.eval()
    with torch.no_grad():
        X_p2_t = torch.tensor(X_p2, dtype=torch.float32)
        Z_p1_adapt = tronco.encode(adapt_petra.adapt_in(X_p1_t)).numpy()
        Z_p2_adapt = tronco.encode(adapt_petra.adapt_in(X_p2_t)).numpy()

    y_p1_idx = np.array([VOCALES.index(v) for v in Y_p1])
    y_p2_idx = np.array([VOCALES.index(v) for v in Y_p2])

    acc_p1_adapt_lda = lda_tri.score(Z_p1_adapt, y_p1_idx)
    acc_p2_adapt_lda = lda_tri.score(Z_p2_adapt, y_p2_idx)
    acc_p2_adapt_nc = nc_tri.score(Z_p2_adapt, y_p2_idx)
    acc_p2_adapt_knn = knn_tri.score(Z_p2_adapt, y_p2_idx)

    y_p2_pred_lda = lda_tri.predict(Z_p2_adapt)

    print(f"  Petra Día 1 (Calibración N={len(Y_p1)}): LDA = {acc_p1_adapt_lda*100:6.2f}%")
    print(f"  Petra Día 2 (Test Ciego N={len(Y_p2)}):    LDA = {acc_p2_adapt_lda*100:6.2f}%")
    print(f"                                             Nearest Centroid = {acc_p2_adapt_nc*100:6.2f}%")
    print(f"                                             k-NN (k=7)       = {acc_p2_adapt_knn*100:6.2f}%")

    print("\n  Desglose en Petra Día 2 (Test Ciego) por Vocal con LDA:")
    for i, v in enumerate(VOCALES):
        m = (y_p2_idx == i)
        if m.sum() > 0:
            acc_v = np.mean(y_p2_pred_lda[m] == i) * 100.0
            print(f"    /{v.lower()}/: {acc_v:5.1f}% (N={m.sum()})")

    # 6. Generar Figura Oficial
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    os.makedirs(dir_figuras, exist_ok=True)
    fig_path = os.path.join(dir_figuras, "evaluacion_extrema_petra_3d.png")

    fig = plt.figure(figsize=(18, 9), facecolor='white')

    # Panel 1: Vista 3D Zero-Shot Puro de Petra
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    for v in np.unique(Y_petra_all):
        m = (Y_petra_all == v)
        ax1.scatter(Z_petra_zero[m, 0], Z_petra_zero[m, 1], Z_petra_zero[m, 2],
                    color=COLORES_VOCALES[v], s=25, alpha=0.6, label=f"/{v.lower()}/")
    for v in VOCALES:
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax1.scatter([c[0]], [c[1]], [c[2]], color='black', marker='D', s=60, edgecolors='white', linewidth=1.2)
    ax1.set_title(f"Petra Zero-Shot Puro (N={len(Y_petra_all)})\nLDA: {acc_zero_lda*100:.1f}%", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Z1 (Sonrisa)", fontsize=8)
    ax1.set_ylabel("Z2 (Apertura)", fontsize=8)
    ax1.set_zlabel("Z3 (Protrusión)", fontsize=8)
    ax1.view_init(elev=25, azim=40)

    # Panel 2: Vista 3D Petra Día 2 Test Ciego Calibrado
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    for v in np.unique(Y_p2):
        m = (Y_p2 == v)
        ax2.scatter(Z_p2_adapt[m, 0], Z_p2_adapt[m, 1], Z_p2_adapt[m, 2],
                    color=COLORES_VOCALES[v], s=28, alpha=0.7, label=f"/{v.lower()}/")
    for v in VOCALES:
        c = CENTROIDES_CANONICOS_3D[v].numpy()
        ax2.scatter([c[0]], [c[1]], [c[2]], color='black', marker='D', s=70, edgecolors='white', linewidth=1.5)
    ax2.set_title(f"Petra Día 2 Test Ciego (N={len(Y_p2)})\nLDA Calibrado: {acc_p2_adapt_lda*100:.1f}% | k-NN: {acc_p2_adapt_knn*100:.1f}%",
                  fontsize=11, fontweight='bold')
    ax2.set_xlabel("Z1 (Sonrisa)", fontsize=8)
    ax2.set_ylabel("Z2 (Apertura)", fontsize=8)
    ax2.set_zlabel("Z3 (Protrusión)", fontsize=8)
    ax2.view_init(elev=25, azim=40)

    # Panel 3: Matriz de Confusión Petra Día 2
    ax3 = fig.add_subplot(2, 3, 3)
    cm_p = confusion_matrix(y_p2_idx, y_p2_pred_lda, labels=range(5))
    vocales_presentes = [i for i in range(5) if np.sum(y_p2_idx == i) > 0]
    cm_sub = cm_p[vocales_presentes, :][:, vocales_presentes]
    cm_norm = cm_sub.astype('float') / cm_sub.sum(axis=1)[:, np.newaxis]
    ax3.imshow(cm_norm, cmap='Greens', vmin=0, vmax=1)
    for r_idx, r in enumerate(vocales_presentes):
        for c_idx, c in enumerate(vocales_presentes):
            pct_val = cm_norm[r_idx, c_idx] * 100.0
            col_txt = 'white' if cm_norm[r_idx, c_idx] > 0.5 else 'black'
            ax3.text(c_idx, r_idx, f"{pct_val:.1f}%\n({cm_sub[r_idx, c_idx]})", ha='center', va='center', color=col_txt, fontsize=9)
    ax3.set_xticks(range(len(vocales_presentes)))
    ax3.set_xticklabels([f"/{VOCALES[i].lower()}/" for i in vocales_presentes], fontweight='bold')
    ax3.set_yticks(range(len(vocales_presentes)))
    ax3.set_yticklabels([f"/{VOCALES[i].lower()}/" for i in vocales_presentes], fontweight='bold')
    ax3.set_xlabel("Predicción LDA", fontweight='bold')
    ax3.set_ylabel("Etiqueta Real", fontweight='bold')
    ax3.set_title("Matriz de Confusión Petra Día 2 (Ciego)", fontsize=11, fontweight='bold')

    # Paneles 4, 5, 6: Proyecciones 2D Petra Día 2
    planos = [
        (0, 1, "Z1 (Sonrisa)", "Z2 (Apertura Mandibular)"),
        (0, 2, "Z1 (Sonrisa)", "Z3 (Protrusión Labial)"),
        (2, 1, "Z3 (Protrusión)", "Z2 (Apertura Mandibular)")
    ]

    for p_idx, (d_x, d_y, lbl_x, lbl_y) in enumerate(planos):
        ax_p = fig.add_subplot(2, 3, 4 + p_idx)
        for v in np.unique(Y_p2):
            m = (Y_p2 == v)
            ax_p.scatter(Z_p2_adapt[m, d_x], Z_p2_adapt[m, d_y],
                         color=COLORES_VOCALES[v], s=24, alpha=0.6, label=f"/{v.lower()}/")
        for v in VOCALES:
            c = CENTROIDES_CANONICOS_3D[v].numpy()
            ax_p.scatter([c[d_x]], [c[d_y]], color='black', marker='D', s=60, edgecolors='white', linewidth=1.2)
        ax_p.set_title(f"Petra Día 2: {lbl_x} vs {lbl_y}", fontsize=10, fontweight='bold')
        ax_p.set_xlabel(lbl_x, fontsize=8)
        ax_p.set_ylabel(lbl_y, fontsize=8)
        ax_p.grid(True, linestyle='--', alpha=0.4)
        if p_idx == 0:
            ax_p.legend(loc='lower left', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nFigura oficial guardada en: {fig_path}")

if __name__ == '__main__':
    ejecutar()
