# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Rescate de Grados de Libertad Ocultos en Petra (T=20 puntos):
#              1. Desmezcla Tisular de Modos Común/Diferencial (Cigomático vs Levador).
#              2. Dinámica Temporal de Fases / Takens (Derivada y Desfase Delta t).
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

def transformar_modo_diferencial_petra(X_petra):
    # En Petra:
    # Canal 0: Anterior Belly (Apertura)
    # Canal 1: Zygomaticus Major (Sonrisa)
    # Canal 2: Levator Anguli Oris (Elevación)
    # Creamos:
    # Ch0_new = Ch0 (Apertura)
    # Ch1_new = (Ch1 + Ch2) / sqrt(2) (Modo común comisural)
    # Ch2_new = (Ch1 - Ch2) / sqrt(2) * factor_amplificacion (Modo diferencial específico)
    X_mod = X_petra.copy()
    ch0 = X_petra[:, 0, :]
    ch1 = X_petra[:, 1, :]
    ch2 = X_petra[:, 2, :]

    modo_comun = (ch1 + ch2) / 1.4142
    modo_dif = (ch1 - ch2) / 1.4142

    # Normalizar la escala del modo diferencial respecto al común
    scale = (np.std(modo_comun) + 1e-6) / (np.std(modo_dif) + 1e-6)
    modo_dif_amplificado = modo_dif * min(scale * 0.7, 2.5)

    X_mod[:, 0, :] = ch0
    X_mod[:, 1, :] = modo_comun
    X_mod[:, 2, :] = modo_dif_amplificado
    return X_mod

def transformar_dinamica_takens(X_petra):
    # En T = 20 puntos, calculamos la derivada temporal discreta Delta x / Delta t
    # de los canales 0 y 1 para inyectar información de aceleración de apertura
    # y desfase en el Canal 2:
    X_mod = X_petra.copy()
    N, C, T = X_petra.shape
    v_ch0 = np.gradient(X_petra[:, 0, :], axis=1) # Velocidad de apertura mandibular
    v_ch1 = np.gradient(X_petra[:, 1, :], axis=1) # Velocidad de retracción de comisura

    # En /i/, v_ch1 sube bruscamente mientras v_ch0 es nula.
    # En /e/, v_ch0 tiene un pulso de velocidad negativo/positivo simultáneo.
    # Modulamos el Canal 2 con el desfase y producto cruzado de velocidades:
    fase_dinamica = v_ch0 * v_ch1
    std_f = np.std(fase_dinamica) + 1e-6
    fase_norm = fase_dinamica / std_f * 0.5

    # Reemplazamos Canal 2 por la combinación del canal original + derivada de fase
    X_mod[:, 2, :] = X_petra[:, 2, :] + fase_norm
    return X_mod

def evaluar_transferencia_petra(X_tr, Y_tr, X_te, Y_te, tronco, lda_tri, nombre_variante):
    fijar_semillas(42)
    adapt = SubjectAdapter(in_channels=3)
    opt = optim.Adam(adapt.parameters(), lr=0.01)
    mse_fn = nn.MSELoss()

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)

    for ep in range(120):
        opt.zero_grad()
        x_proj = adapt.adapt_in(X_tr_t)
        z = tronco.encode(x_proj)
        x_rec = adapt.adapt_out(tronco.decode(z))
        loss_rec = mse_fn(x_rec, X_tr_t)
        loss_ortho = adapt.ortho_loss()

        loss_pot = torch.tensor(0.0)
        for v in np.unique(Y_tr):
            mask_v = (Y_tr == v)
            if mask_v.sum() > 0:
                z_v = z[mask_v]
                target_v = CENTROIDES_CANONICOS_3D[v]
                loss_pot = loss_pot + torch.mean(torch.sum((z_v - target_v) ** 2, dim=-1))

        loss = loss_rec + 0.05 * loss_ortho + 0.20 * loss_pot
        loss.backward()
        opt.step()

    adapt.eval()
    with torch.no_grad():
        Z_tr = tronco.encode(adapt.adapt_in(X_tr_t)).numpy()
        Z_te = tronco.encode(adapt.adapt_in(X_te_t)).numpy()

    y_tr_idx = np.array([VOCALES.index(v) for v in Y_tr])
    y_te_idx = np.array([VOCALES.index(v) for v in Y_te])

    acc_tr = lda_tri.score(Z_tr, y_tr_idx)
    acc_te = lda_tri.score(Z_te, y_te_idx)
    y_pred = lda_tri.predict(Z_te)
    cm = confusion_matrix(y_te_idx, y_pred, labels=range(5))

    acc_vocales = {}
    for i, v in enumerate(VOCALES):
        n_v = np.sum(y_te_idx == i)
        acc_v = (cm[i, i] / n_v * 100.0) if n_v > 0 else 0.0
        acc_vocales[v] = acc_v

    return {
        'nombre': nombre_variante,
        'acc_tr': acc_tr,
        'acc_te': acc_te,
        'cm': cm,
        'acc_vocales': acc_vocales,
        'Z_te': Z_te,
        'y_te_idx': y_te_idx
    }

def main():
    fijar_semillas(42)
    print("=" * 80)
    print("ENSAYO: RESCATE DE GRADOS DE LIBERTAD EN PETRA (T = 20 PUNTOS)")
    print("COMPARATIVA: BASELINE vs DESMEZCLA TISULAR vs DINAMICA TAKENS vs HIBRIDO")
    print("=" * 80)

    # 1. Cargar Modelo 3D y Ajustar LDA Tri-Sujeto
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

    # 2. Cargar Petra
    dir_grid = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    d1 = np.load(os.path.join(dir_grid, "dataset_petra_dia1_20260821_conv2d.npz"), allow_pickle=True)
    d2 = np.load(os.path.join(dir_grid, "dataset_petra_dia2_20260828_conv2d.npz"), allow_pickle=True)

    X_p1, Y_p1 = d1['X_clean'], d1['Y_clean']
    X_p2, Y_p2 = d2['X_clean'], d2['Y_clean']
    if X_p1.ndim == 2:
        X_p1 = X_p1.reshape(-1, 3, 20)
    if X_p2.ndim == 2:
        X_p2 = X_p2.reshape(-1, 3, 20)

    # 3. Evaluar las variantes
    # Variante 0: Baseline
    res_base = evaluar_transferencia_petra(X_p1, Y_p1, X_p2, Y_p2, tronco, lda_tri, "Baseline SO(3)")

    # Variante 1: Desmezcla Tisular Común/Diferencial (Cigomático vs Levador)
    X_p1_dif = transformar_modo_diferencial_petra(X_p1)
    X_p2_dif = transformar_modo_diferencial_petra(X_p2)
    res_dif = evaluar_transferencia_petra(X_p1_dif, Y_p1, X_p2_dif, Y_p2, tronco, lda_tri, "Desmezcla Tisular (Común/Dif)")

    # Variante 2: Dinámica Takens (Velocidades y Desfase Delta t en T=20)
    X_p1_tak = transformar_dinamica_takens(X_p1)
    X_p2_tak = transformar_dinamica_takens(X_p2)
    res_tak = evaluar_transferencia_petra(X_p1_tak, Y_p1, X_p2_tak, Y_p2, tronco, lda_tri, "Dinámica Takens (Delta t)")

    # Variante 3: Híbrido (Desmezcla + Dinámica)
    X_p1_hib = transformar_dinamica_takens(X_p1_dif)
    X_p2_hib = transformar_dinamica_takens(X_p2_dif)
    res_hib = evaluar_transferencia_petra(X_p1_hib, Y_p1, X_p2_hib, Y_p2, tronco, lda_tri, "Híbrido (Tisular + Takens)")

    resultados = [res_base, res_dif, res_tak, res_hib]

    print("\n" + "=" * 80)
    print("RESULTADOS EN PETRA DIA 2 (TEST CIEGO N=168, T=20 PUNTOS):")
    print("=" * 80)
    print(f"{'Método':<28} | {'Día 1 Calib':<12} | {'Día 2 Ciego':<12} | {'/a/ (%)':<8} | {'/e/ (%)':<8} | {'/i/ (%)':<8} | {'/o/ (%)':<8} | {'/u/ (%)':<8}")
    print("-" * 105)
    for r in resultados:
        voc = r['acc_vocales']
        print(f"{r['nombre']:<28} | {r['acc_tr']*100:^12.1f} | {r['acc_te']*100:^12.1f} | {voc['A']:^8.1f} | {voc['E']:^8.1f} | {voc['I']:^8.1f} | {voc['O']:^8.1f} | {voc['U']:^8.1f}")

    # 4. Graficar Comparativa 2x2 de Matrices de Confusión
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "comparativa_grados_libertad_petra_takens_ica.png")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11), facecolor='white')
    axes = axes.flatten()

    for idx, r in enumerate(resultados):
        ax = axes[idx]
        cm = r['cm']
        cm_norm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)
        ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        for row in range(5):
            for col in range(5):
                pct = cm_norm[row, col] * 100.0
                col_txt = 'white' if cm_norm[row, col] > 0.5 else 'black'
                ax.text(col, row, f"{pct:.1f}%\n({cm[row, col]})", ha='center', va='center', color=col_txt, fontsize=8, fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"/{v.lower()}/" for v in VOCALES], fontweight='bold')
        ax.set_title(f"{r['nombre']}\nDía 2 Ciego: {r['acc_te']*100:.1f}% (/i/: {r['acc_vocales']['I']:.1f}%)", fontweight='bold', fontsize=10)
        ax.set_ylabel("Etiqueta Real", fontweight='bold', fontsize=9)
        ax.set_xlabel("Predicción LDA", fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nFigura comparativa guardada en: {fig_path}")

if __name__ == '__main__':
    main()
