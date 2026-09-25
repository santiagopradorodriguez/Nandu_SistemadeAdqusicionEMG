#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Experimento de Resolucion Temporal: 20 vs 25 vs 30 vs 35 Puntos
# Objetivo: Evaluar si mayor resolucion temporal despega /i/ y /e/ manteniendo balance
# Arquitectura: Conv Orthogonal (K=5, Ch=(6, 12), lr=0.003, lw=2.0, lz=0.5, 350 epocas)
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

out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_resolucion_temporal")
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
# 1. ARQUITECTURA PARAMETRIZADA POR PUNTOS TEMPORALES
# ==============================================================================
class DynamicTimeConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, hidden_dim=32, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh() if act_name == 'tanh' else nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        
        self.dfc1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2 = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

        dims_necesarias = set()
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            dims_necesarias.add(min(W.shape[0], W.shape[1]))
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
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
        
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        recon_3d = self.deconv2(dh3)
        recon_flat = recon_3d.view(recon_3d.shape[0], -1)
        return recon_flat, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            d0, d1 = W.shape
            if d0 < d1:
                gram = torch.mm(W, W.t())
                I = getattr(self, f'_eye_{d0}')
            else:
                gram = torch.mm(W.t(), W)
                I = getattr(self, f'_eye_{d1}')
            loss = loss + torch.sum((gram - I) ** 2)
            
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
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
# 2. EXTRACCION EFICIENTE POR PUNTOS TEMPORALES CON CACHE
# ==============================================================================
def extraer_datos_puntos(pts_target, cache_canales, base_dir, meds):
    params = {
        'alpha_ruido': 0.5,
        'smooth_ms': 90,
        'target_length': pts_target,
        'snr_threshold': 0.5,
        'outlier_contamination': 0.1,
        'notch_q': 2.0,
        'highpass_cutoff_hz': 20.0,
        'lowpass_cutoff_hz': 500.0,
        'tipo_envolvente': 'rms',
        'gate_ratio_ruido': 0.0,
        'tipo_filtro_ruido': 'notch',
        'correccion_impedancia': False
    }
    
    X_l_raw, Y_l, Tomas_l, _ = gpu.extraer_y_filtrar(
        mediciones=meds,
        base_dir=base_dir,
        params=params,
        aplicar_trevisan=False,
        modo_alineacion='Pico Volumen Micrófono',
        pre_pct=0.5,
        post_pct=0.5,
        canales_features=['canal_0', 'canal_1', 'canal_2'],
        ignorar_ventana_cero=False,
        cache_canales_data=cache_canales,
        aplicar_correccion_intersesion=False
    )
    
    N_l = len(X_l_raw)
    sesiones_l = np.array([gsc.extraer_sesion_agnostica(t) for t in Tomas_l])
    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')
    
    # Normalizacion P95 estándar
    X_l_res = np.array(X_l_raw).reshape(N_l, 3, pts_target)
    X_l_filt = np.zeros_like(X_l_res)
    for i in range(N_l):
        for c in range(3):
            X_l_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_l_res[i, c, :])
    X_l_norm = np.zeros_like(X_l_filt)
    for s in np.unique(sesiones_l):
        mask = (sesiones_l == s)
        for c in range(3):
            base_mean = np.mean(X_l_filt[mask, c, :max(5, pts_target // 4)])
            base_max = np.percentile(X_l_filt[mask, c, :], 95) - base_mean + 1e-6
            X_l_norm[mask, c, :] = (X_l_filt[mask, c, :] - base_mean) / base_max

    # Extraccion de P5 continua con el mismo numero de puntos
    toma_p5 = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-06-10/SecuenciaContinua_Prueba5_Sujeto1")
    with open(os.path.join(toma_p5, "canal_0", "metadata.json"), "r") as f:
        meta_p5 = json.load(f)
    fs = meta_p5.get("sample_rate", 2000)
    noise_sec = meta_p5.get("noise_seconds", 5.0)
    n_noise_samples = int(noise_sec * fs)
    palabras_ground = meta_p5.get("valid_words", [])

    signals = []
    for ch in range(4):
        p_wav = os.path.join(toma_p5, f"canal_{ch}", "grabacion.wav")
        _, data = wavfile.read(p_wav)
        signals.append(data.astype(np.float64))
    sig_emg = np.stack(signals[:3], axis=0)
    sig_mic = signals[3]
    n_samples = sig_emg.shape[1]

    b_notch, a_notch = signal.iirnotch(50.0, 2.0, fs)
    b_band, a_band = signal.butter(2, [20.0, 500.0], 'bandpass', fs=fs)
    sig_filt = np.zeros_like(sig_emg)
    for c in range(3):
        s_n = signal.filtfilt(b_notch, a_notch, sig_emg[c])
        sig_filt[c] = signal.filtfilt(b_band, a_band, s_n)

    win_rms = int(0.090 * fs)
    if win_rms % 2 == 0: win_rms += 1
    kernel_rms = np.ones(win_rms) / win_rms
    env_emg = np.zeros_like(sig_filt)
    for c in range(3):
        env_emg[c] = np.sqrt(np.maximum(0, np.convolve(sig_filt[c]**2, kernel_rms, mode='same')))

    win_mic = int(0.050 * fs)
    mic_env = np.convolve(np.abs(sig_mic), np.ones(win_mic) / win_mic, mode='same')
    picos_candidatos, _ = signal.find_peaks(mic_env, distance=int(1.2 * fs), height=2000)
    picos_fonacion = [p for p in picos_candidatos if p >= 6.0 * fs and p < (n_samples - int(1.0 * fs))]
    if len(picos_fonacion) > 125: picos_fonacion = picos_fonacion[:125]

    ruido_base_emg = np.median(env_emg[:, :n_noise_samples], axis=1, keepdims=True)
    half_win = int(1.0 * fs)
    X_p5_list, y_p5_list = [], []

    for i, p in enumerate(picos_fonacion):
        start = p - half_win
        end = p + half_win
        if start < 0 or end > n_samples: continue
        seg_raw = np.maximum(env_emg[:, start:end] - ruido_base_emg, 0.0)
        M_sup = np.max(seg_raw) + 1e-9
        seg_norm = seg_raw / M_sup
        feat_p = []
        for c in range(3):
            feat_p.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_norm.shape[1]), seg_norm[c]))
        X_p5_list.append(np.concatenate(feat_p))
        v_ground = palabras_ground[i] if i < len(palabras_ground) else vocales[i % 5]
        y_p5_list.append(v_ground)

    N_p5 = len(X_p5_list)
    X_p5_arr = np.array(X_p5_list).reshape(N_p5, 3, pts_target)
    X_p5_filt = np.zeros_like(X_p5_arr)
    for i in range(N_p5):
        for c in range(3):
            X_p5_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_p5_arr[i, c, :])
    X_p5_norm = np.zeros_like(X_p5_filt)
    for c in range(3):
        base_mean = np.mean(X_p5_filt[:, c, :max(5, pts_target // 4)])
        base_max = np.percentile(X_p5_filt[:, c, :], 95) - base_mean + 1e-6
        X_p5_norm[:, c, :] = (X_p5_filt[:, c, :] - base_mean) / base_max

    return {
        'X_l': torch.tensor(X_l_norm, dtype=torch.float32),
        'y_l': np.array(Y_l),
        'X_p': torch.tensor(X_p5_norm, dtype=torch.float32),
        'y_p': np.array(y_p5_list)
    }

# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACION CONTROLADA
# ==============================================================================
def evaluar_resolucion(pts_target, datos, device, seed=100, epochs=350):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    X_l = datos['X_l'].to(device)
    y_l = datos['y_l']
    X_p = datos['X_p'].to(device)
    y_p = datos['y_p']
    N_lucas = X_l.shape[0]

    model = DynamicTimeConvOrthogonalAE(
        in_channels=3,
        time_pts=pts_target,
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
    X_l_flat = X_l.view(N_lucas, -1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, z = model(X_l)
        loss_recon = nn.functional.mse_loss(recon, X_l_flat)
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N_lucas - 1)
        loss_z = torch.sum((cov_z - I_2d) ** 2)
        loss = loss_recon + lambda_w * loss_w + lambda_z * loss_z
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z_lucas = model.encode(X_l).cpu().numpy()
        Z_p5 = model.encode(X_p).cpu().numpy()

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
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p])
    acc_p5 = accuracy_score(y_p5_idx, pred_p5_idx) * 100.0

    harmonic_acc = 2 * (acc_lucas * acc_p5) / (acc_lucas + acc_p5 + 1e-9)

    l_accs = {v: (np.mean(pred_lucas_idx[y_lucas_idx == idx] == idx) * 100.0 if np.sum(y_lucas_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}
    p_accs = {v: (np.mean(pred_p5_idx[y_p5_idx == idx] == idx) * 100.0 if np.sum(y_p5_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}

    return {
        'pts': pts_target,
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
# 4. MAIN DEL BARRIDO TEMPORAL
# ==============================================================================
def main():
    print("=" * 80)
    print("BARRIDO DE RESOLUCION TEMPORAL: 20 vs 25 vs 30 vs 35 PUNTOS POR CANAL")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo de cálculo: {device}")

    base_dir = os.path.join(project_root, 'EMG_desarrollo/base_de_datos_electrodos')
    meds = [os.path.join('2026-07-10', d) for d in sorted(os.listdir(os.path.join(base_dir, '2026-07-10'))) if d.startswith(('A_', 'E_', 'I_', 'O_', 'U_'))]
    cache_canales = {}

    puntos_a_probar = [20, 25, 30, 35]
    resultados = []

    for pts in puntos_a_probar:
        t0 = time.time()
        print(f"\n[Procesando] Resolución de {pts} puntos por canal (vector de {3*pts} dimensiones)...")
        datos = extraer_datos_puntos(pts, cache_canales, base_dir, meds)
        print(f"  Datos listos: Lucas = {len(datos['y_l'])} | P5 = {len(datos['y_p'])} pulsos ({time.time()-t0:.1f}s)")
        
        res = evaluar_resolucion(pts, datos, device=device, seed=100, epochs=350)
        resultados.append(res)
        print(f"  -> Resultado {pts} pts: Lucas = {res['acc_lucas']:.2f}% | P5 = {res['acc_p5']:.2f}% | Armónica = {res['harmonic_acc']:.2f}% | Piso Lucas = {res['min_lucas']:.1f}% | Piso P5 = {res['min_p5']:.1f}%")

    print("\n" + "=" * 90)
    print(f"{'Puntos':<10} | {'Lucas Acc':<12} | {'P5 Acc':<12} | {'Armónica':<12} | {'Piso Lucas':<12} | {'Piso P5':<12}")
    print("-" * 90)
    for r in resultados:
        print(f"{r['pts']:<10} | {r['acc_lucas']:>10.2f}% | {r['acc_p5']:>10.2f}% | {r['harmonic_acc']:>10.2f}% | {r['min_lucas']:>10.1f}% | {r['min_p5']:>10.1f}%")
    print("=" * 90)

    print("\nDESGLOSE EN ENTRENAMIENTO (LUCAS):")
    print(f"{'Puntos':<10} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 60)
    for r in resultados:
        la = r['l_accs']
        print(f"{r['pts']:<10} | {la['A']:>5.1f}% | {la['E']:>5.1f}% | {la['I']:>5.1f}% | {la['O']:>5.1f}% | {la['U']:>5.1f}%")
    print("-" * 60)

    print("\nDESGLOSE EN SECUENCIA CONTINUA (P5):")
    print(f"{'Puntos':<10} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 60)
    for r in resultados:
        pa = r['p_accs']
        print(f"{r['pts']:<10} | {pa['A']:>5.1f}% | {pa['E']:>5.1f}% | {pa['I']:>5.1f}% | {pa['O']:>5.1f}% | {pa['U']:>5.1f}%")
    print("-" * 60)

    # Grafico comparativo de las 4 resoluciones
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    for row_idx, r in enumerate(resultados):
        pts = r['pts']
        # Lucas
        ax_l = axes[row_idx, 0]
        Z_l = r['Z_lucas']
        y_l = datos['y_l']
        for v in vocales:
            mask = (y_l == v)
            ax_l.scatter(Z_l[mask, 0], Z_l[mask, 1], c=colores_oficiales[v], label=f"/{v.lower()}/", alpha=0.7, s=25)
        ax_l.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.set_title(f"Resolución {pts} Puntos: Lucas {r['acc_lucas']:.1f}% - Piso {r['min_lucas']:.1f}%", fontsize=10, fontweight='bold')
        ax_l.grid(True, linestyle=':', alpha=0.5)
        if row_idx == 0: ax_l.legend(loc='upper right', fontsize=8)

        # P5
        ax_p = axes[row_idx, 1]
        Z_p = r['Z_p5']
        y_p = datos['y_p']
        y_p_idx = np.array([vocal_to_idx[v] for v in y_p])
        # Reconstruir aciertos
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
        ax_p.set_title(f"Resolución {pts} Puntos: P5 {r['acc_p5']:.1f}% ({num_ok}/123) - Piso {r['min_p5']:.1f}%", fontsize=10, fontweight='bold')
        ax_p.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    p_png = os.path.join(out_dir, "comparativa_resoluciones_temporales.png")
    plt.savefig(p_png, dpi=150)
    plt.close()
    print(f"\n[Visualización] Gráfico guardado en: {p_png}")
    print("=" * 80)

if __name__ == '__main__':
    main()
