#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Autoencoder con Decodificador Dual y Perdida Compuesta (RMS + TKEO)
# Concepto: Entrada limpia RMS (3ch x 20pts) -> Latente Z -> Reconstruye RMS + TKEO
# Evaluacion: Exactitud GMM Lucas + Transferencia Directa P5 en 2D
# ==============================================================================

import os
import sys
import json
import time
import re
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
import scipy.signal as signal
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
emg_desarrollo = os.path.join(project_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

from deep_learning.pca_umap_clustering import generador_pca_umap as gpu
import deep_learning.grid_search_conv_ortogonal as gsc

out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_perdida_compuesta")
os.makedirs(out_dir, exist_ok=True)

vocales = ['A', 'E', 'I', 'O', 'U']
vocal_to_idx = {v: i for i, v in enumerate(vocales)}
colores_oficiales = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

# ==============================================================================
# 1. ARQUITECTURA CON DECODIFICADOR DUAL (DOBLE CABEZA)
# ==============================================================================
class DualHeadConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, hidden_dim=32, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        # Codificador compartido (Recibe solo RMS limpia)
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh() if act_name == 'tanh' else nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        
        # Cabeza Decodificadora 1: Reconstruccion de Envolvente RMS
        self.dfc1_rms = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2_rms = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1_rms = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2_rms = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)

        # Cabeza Decodificadora 2: Reconstruccion de Perfil de Energia Teager-Kaiser (TKEO)
        self.dfc1_tkeo = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2_tkeo = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1_tkeo = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2_tkeo = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)

        self.c2 = c2

        # Precalcular matrices identidad
        dims_necesarias = set()
        for layer in [self.fc1, self.fc2, self.dfc1_rms, self.dfc2_rms, self.dfc1_tkeo, self.dfc2_tkeo]:
            W = layer.weight
            dims_necesarias.add(min(W.shape[0], W.shape[1]))
        for conv in [self.conv1, self.conv2, self.deconv1_rms, self.deconv2_rms, self.deconv1_tkeo, self.deconv2_tkeo]:
            W_flat = conv.weight.view(conv.weight.shape[0], -1)
            dims_necesarias.add(min(W_flat.shape[0], W_flat.shape[1]))
            
        for d in dims_necesarias:
            self.register_buffer(f'_eye_{d}', torch.eye(d))

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

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        z = self.fc2(h3)
        
        # Rama RMS
        dh1_r = self.act(self.dfc1_rms(z))
        dh2_r = self.act(self.dfc2_rms(dh1_r)).view(dh1_r.shape[0], self.c2, self.time_pts)
        dh3_r = self.act(self.deconv1_rms(dh2_r))
        recon_rms = self.deconv2_rms(dh3_r).view(dh3_r.shape[0], -1)

        # Rama TKEO
        dh1_t = self.act(self.dfc1_tkeo(z))
        dh2_t = self.act(self.dfc2_tkeo(dh1_t)).view(dh1_t.shape[0], self.c2, self.time_pts)
        dh3_t = self.act(self.deconv1_tkeo(dh2_t))
        recon_tkeo = self.deconv2_tkeo(dh3_t).view(dh3_t.shape[0], -1)

        return recon_rms, recon_tkeo, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1_rms, self.dfc2_rms, self.dfc1_tkeo, self.dfc2_tkeo]:
            W = layer.weight
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)
            
        for conv in [self.conv1, self.conv2, self.deconv1_rms, self.deconv2_rms, self.deconv1_tkeo, self.deconv2_tkeo]:
            W = conv.weight.view(conv.weight.shape[0], -1)
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)
        return loss

# ==============================================================================
# 2. CARGA DE DATOS ALINEADOS
# ==============================================================================
def cargar_datos():
    # 1. Lucas RMS oficial
    csv_ref = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv")
    df_ref = pd.read_csv(csv_ref)
    tomas_ref = df_ref['Toma'].tolist()
    y_lucas = df_ref['Vocal'].values
    sesiones_l = np.array([gsc.extraer_sesion_agnostica(t) for t in tomas_ref])
    feat_cols = [c for c in df_ref.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_l_rms_raw = df_ref[feat_cols].values
    N_l = len(X_l_rms_raw)
    n_ch, n_pts = 3, 20

    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')

    # 2. Lucas TKEO
    base_dir = os.path.join(project_root, 'EMG_desarrollo/base_de_datos_electrodos')
    meds = [os.path.join('2026-07-10', d) for d in sorted(os.listdir(os.path.join(base_dir, '2026-07-10'))) if d.startswith(('A_', 'E_', 'I_', 'O_', 'U_'))]
    params_tkeo = {
        'alpha_ruido': 0.5, 'smooth_ms': 90, 'target_length': 20,
        'snr_threshold': 0.0, 'outlier_contamination': 0.0, 'notch_q': 2.0,
        'highpass_cutoff_hz': 20.0, 'lowpass_cutoff_hz': 500.0,
        'tipo_envolvente': 'tkeo', 'gate_ratio_ruido': 0.0,
        'tipo_filtro_ruido': 'notch', 'correccion_impedancia': False
    }
    X_tkeo_all, _, Tomas_tkeo_all, _ = gpu.extraer_y_filtrar(
        mediciones=meds, base_dir=base_dir, params=params_tkeo,
        aplicar_trevisan=False, modo_alineacion='Pico Volumen Micrófono',
        pre_pct=0.5, post_pct=0.5, canales_features=['canal_0', 'canal_1', 'canal_2'],
        ignorar_ventana_cero=False, aplicar_correccion_intersesion=False
    )
    tkeo_map = {t: X_tkeo_all[i] for i, t in enumerate(Tomas_tkeo_all)}
    X_l_tkeo_raw = np.zeros_like(X_l_rms_raw)
    for i, t in enumerate(tomas_ref):
        if t in tkeo_map:
            X_l_tkeo_raw[i] = tkeo_map[t]
        else:
            prefix = "_".join(t.split("_")[:3])
            cands = [k for k in tkeo_map if k.startswith(prefix)]
            X_l_tkeo_raw[i] = tkeo_map[cands[0]] if cands else X_l_rms_raw[i]

    def norm_lucas(mat):
        X_res = mat.reshape(N_l, n_ch, n_pts)
        X_f = np.zeros_like(X_res)
        for i in range(N_l):
            for c in range(n_ch):
                X_f[i, c, :] = signal.filtfilt(b_bw, a_bw, X_res[i, c, :])
        X_n = np.zeros_like(X_f)
        for s in np.unique(sesiones_l):
            mask = (sesiones_l == s)
            for c in range(n_ch):
                base_mean = np.mean(X_f[mask, c, :10])
                base_max = np.percentile(X_f[mask, c, :], 95) - base_mean + 1e-6
                X_n[mask, c, :] = (X_f[mask, c, :] - base_mean) / base_max
        return X_n

    X_l_rms = norm_lucas(X_l_rms_raw)
    X_l_tkeo = norm_lucas(X_l_tkeo_raw)

    # 3. P5 Oficial
    X_p5_rms_flat, y_p5 = gsc.cargar_datos_p5(b_bw, a_bw)
    N_p5 = len(X_p5_rms_flat)
    X_p5_rms = X_p5_rms_flat.view(N_p5, 3, 20).numpy()

    return {
        'X_l_rms': torch.tensor(X_l_rms, dtype=torch.float32),
        'X_l_tkeo': torch.tensor(X_l_tkeo, dtype=torch.float32),
        'y_l': y_lucas,
        'X_p5_rms': torch.tensor(X_p5_rms, dtype=torch.float32),
        'y_p5': y_p5
    }

# ==============================================================================
# 3. ENTRENAMIENTO CON PERDIDA COMPUESTA
# ==============================================================================
def entrenar_modelo_compuesto(beta_tkeo, datos, device, epochs=350, seed=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_l_rms = datos['X_l_rms'].to(device)
    X_l_tkeo = datos['X_l_tkeo'].to(device)
    y_l = datos['y_l']
    X_p5_rms = datos['X_p5_rms'].to(device)
    y_p5 = datos['y_p5']
    
    N_lucas = X_l_rms.shape[0]
    X_l_rms_flat = X_l_rms.view(N_lucas, -1)
    X_l_tkeo_flat = X_l_tkeo.view(N_lucas, -1)

    model = DualHeadConvOrthogonalAE(
        in_channels=3,
        time_pts=20,
        conv_channels=(6, 12),
        kernel_size=5,
        latent_dim=2,
        hidden_dim=32,
        act_name='tanh'
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    lambda_w = 2.0
    lambda_z = 0.5
    I_2d = torch.eye(2, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon_rms, recon_tkeo, z = model(X_l_rms)
        loss_rms = nn.functional.mse_loss(recon_rms, X_l_rms_flat)
        loss_tkeo = nn.functional.mse_loss(recon_tkeo, X_l_tkeo_flat) if beta_tkeo > 0 else torch.tensor(0.0, device=device)
        
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N_lucas - 1)
        loss_z = torch.sum((cov_z - I_2d) ** 2)
        
        loss = loss_rms + beta_tkeo * loss_tkeo + lambda_w * loss_w + lambda_z * loss_z
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z_lucas = model.encode(X_l_rms).cpu().numpy()
        Z_p5 = model.encode(X_p5_rms).cpu().numpy()

    # GMM Lucas
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
    pred_raw_lucas = gmm.fit_predict(Z_lucas)
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(vocales):
        for j in range(5):
            contingency[j, i] = np.sum((y_l == vl) & (pred_raw_lucas == j))
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    pred_lucas_idx = np.array([cluster_to_vocal[c] for c in pred_raw_lucas])
    y_lucas_idx = np.array([vocal_to_idx[v] for v in y_l])
    acc_lucas = accuracy_score(y_lucas_idx, pred_lucas_idx) * 100.0

    # Transferencia P5
    pred_raw_p5 = gmm.predict(Z_p5)
    pred_p5_idx = np.array([cluster_to_vocal[c] for c in pred_raw_p5])
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p5])
    acc_p5 = accuracy_score(y_p5_idx, pred_p5_idx) * 100.0

    harmonic_acc = 2 * (acc_lucas * acc_p5) / (acc_lucas + acc_p5 + 1e-9)

    l_accs = {v: (np.mean(pred_lucas_idx[y_lucas_idx == idx] == idx) * 100.0 if np.sum(y_lucas_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}
    p_accs = {v: (np.mean(pred_p5_idx[y_p5_idx == idx] == idx) * 100.0 if np.sum(y_p5_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}

    return {
        'beta': beta_tkeo,
        'acc_lucas': acc_lucas,
        'acc_p5': acc_p5,
        'harmonic_acc': harmonic_acc,
        'l_accs': l_accs,
        'p_accs': p_accs,
        'min_lucas': min(l_accs.values()),
        'min_p5': min(p_accs.values()),
        'Z_lucas': Z_lucas,
        'Z_p5': Z_p5
    }

# ==============================================================================
# 4. EJECUCION EXPERIMENTAL
# ==============================================================================
def main():
    print("=" * 85)
    print("EXPERIMENTO: PERDIDA COMPUESTA (RMS + AUXILIAR TEAGER-KAISER)")
    print("=" * 85)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de cálculo: {device}")

    print("\n[Carga] Cargando datos con paridad exacta...")
    datos = cargar_datos()
    print(f"  Lucas: {len(datos['y_l'])} ventanas | P5: {len(datos['y_p5'])} pulsos.")

    # Barrido del factor beta de perdida auxiliar TKEO
    betas = [0.0, 0.05, 0.10, 0.20, 0.35, 0.50]
    resultados = []

    print("\n[Entrenamiento] Evaluando valores de factor auxiliar beta_TKEO...")
    for b in betas:
        t0 = time.time()
        res = entrenar_modelo_compuesto(b, datos, device=device, epochs=350, seed=100)
        resultados.append(res)
        num_ok_p5 = int(round(res['acc_p5'] * 123 / 100.0))
        print(f"  beta={b:4.2f} | Lucas: {res['acc_lucas']:5.2f}% | P5: {res['acc_p5']:5.2f}% ({num_ok_p5}/123) | Armonica: {res['harmonic_acc']:5.2f}% | Piso Lucas: {res['min_lucas']:4.1f}% | Piso P5: {res['min_p5']:4.1f}% ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 95)
    print(f"{'Beta TKEO':<12} | {'Lucas Acc':<12} | {'P5 Acc':<14} | {'Armonica':<12} | {'Piso Lucas':<12} | {'Piso P5':<12}")
    print("-" * 95)
    for r in resultados:
        num_p5 = int(round(r['acc_p5'] * 123 / 100.0))
        p5_str = f"{r['acc_p5']:.2f}% ({num_p5})"
        print(f"{r['beta']:<12.2f} | {r['acc_lucas']:>10.2f}% | {p5_str:>14} | {r['harmonic_acc']:>10.2f}% | {r['min_lucas']:>10.1f}% | {r['min_p5']:>10.1f}%")
    print("=" * 95)

    print("\nDESGLOSE POR VOCAL EN LUCAS (ENTRENAMIENTO):")
    print(f"{'Beta TKEO':<12} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 65)
    for r in resultados:
        la = r['l_accs']
        print(f"{r['beta']:<12.2f} | {la['A']:>5.1f}% | {la['E']:>5.1f}% | {la['I']:>5.1f}% | {la['O']:>5.1f}% | {la['U']:>5.1f}%")
    print("-" * 65)

    print("\nDESGLOSE POR VOCAL EN CANDELA P5 (SECUENCIA CONTINUA):")
    print(f"{'Beta TKEO':<12} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 65)
    for r in resultados:
        pa = r['p_accs']
        print(f"{r['beta']:<12.2f} | {pa['A']:>5.1f}% | {pa['E']:>5.1f}% | {pa['I']:>5.1f}% | {pa['O']:>5.1f}% | {pa['U']:>5.1f}%")
    print("-" * 65)

    # Grafico comparativo de las 6 betas
    fig, axes = plt.subplots(len(betas), 2, figsize=(14, 4 * len(betas)))
    y_l = datos['y_l']
    y_p = datos['y_p5']
    y_p_idx = np.array([vocal_to_idx[v] for v in y_p])

    for row_idx, r in enumerate(resultados):
        b = r['beta']
        # Lucas
        ax_l = axes[row_idx, 0]
        Z_l = r['Z_lucas']
        for v in vocales:
            mask = (y_l == v)
            ax_l.scatter(Z_l[mask, 0], Z_l[mask, 1], c=colores_oficiales[v], label=f"/{v.lower()}/", alpha=0.7, s=25)
        ax_l.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.set_title(f"Beta TKEO = {b:.2f} - Lucas: {r['acc_lucas']:.1f}% - Piso: {r['min_lucas']:.1f}%", fontsize=10, fontweight='bold')
        ax_l.grid(True, linestyle=':', alpha=0.5)
        if row_idx == 0: ax_l.legend(loc='upper right', fontsize=8)

        # P5
        ax_p = axes[row_idx, 1]
        Z_p = r['Z_p5']
        
        gmm_p = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
        pred_raw_l = gmm_p.fit_predict(Z_l)
        contingency = np.zeros((5, 5))
        for i_v, vl in enumerate(vocales):
            for j_v in range(5):
                contingency[j_v, i_v] = np.sum((y_l == vl) & (pred_raw_l == j_v))
        row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
        c2v = {row_ind[i_c]: col_ind[i_c] for i_c in range(len(row_ind))}
        pred_p_idx = np.array([c2v[c] for c in gmm_p.predict(Z_p)])
        aciertos = (y_p_idx == pred_p_idx)

        for v in vocales:
            mask_ok = (y_p == v) & aciertos
            ax_p.scatter(Z_p[mask_ok, 0], Z_p[mask_ok, 1], c=colores_oficiales[v], marker='o', alpha=0.85, s=35)
            mask_err = (y_p == v) & (~aciertos)
            if np.any(mask_err):
                ax_p.scatter(Z_p[mask_err, 0], Z_p[mask_err, 1], c=colores_oficiales[v], marker='x', alpha=0.9, s=45, linewidths=1.5)
        ax_p.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_p.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        num_ok = int(np.sum(aciertos))
        ax_p.set_title(f"Beta TKEO = {b:.2f} - P5: {r['acc_p5']:.1f}% ({num_ok}/123) - Piso: {r['min_p5']:.1f}%", fontsize=10, fontweight='bold')
        ax_p.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    p_png = os.path.join(out_dir, "comparativa_perdida_compuesta_betas.png")
    plt.savefig(p_png, dpi=150)
    plt.close()
    print(f"\n[Visualización] Gráfico guardado en: {p_png}")
    print("=" * 85)

if __name__ == '__main__':
    main()
