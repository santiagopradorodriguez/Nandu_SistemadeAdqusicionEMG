#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Grid Search de Autoencoders Convolucionales 1D Ortogonales en 3D
# Evaluacion: Exactitud GMM Lucas 3D + Transferencia Directa a Secuencia Continua P5
# Deteccion y anuncio de record en tiempo real con guardado automatico
# ==============================================================================

import os
import sys
import json
import time
import re
import argparse
import itertools
import subprocess
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
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim

# Configuracion de directorios
project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
out_dir_default = os.path.join(
    project_root,
    "EMG_desarrollo/resultados/grid_search_conv_ortogonal_3d"
)

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
# 1. ARQUITECTURA PARAMETRIZABLE EN 3D CON BUFFER DE IDENTIDADES PRECALCULADO
# ==============================================================================
class ParametricConvOrthogonalAE3D(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(8, 16), kernel_size=3, latent_dim=3, hidden_dim=64, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        
        if act_name == 'tanh':
            self.act = nn.Tanh()
        elif act_name == 'gelu':
            self.act = nn.GELU()
        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        
        self.dfc1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2 = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

        # Precalcular matrices identidad en buffers para evitar alocaciones en cada epoca
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
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ==============================================================================
def extraer_sesion_agnostica(toma_str):
    s = str(toma_str)
    m = re.search(r'(Prueba\d+|Sesion\d+|Session\d+|T\d+|S\d+)', s, re.IGNORECASE)
    if m:
        return m.group(0).upper()
    parts = s.split('_')
    for p in parts:
        p_clean = p.strip()
        if p_clean.lower().startswith('win'):
            continue
        if any(char.isdigit() for char in p_clean) and len(p_clean) <= 10:
            return p_clean.upper()
    return 'S1'

def cargar_datos_lucas():
    csv_lucas = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv"
    )
    df = pd.read_csv(csv_lucas)
    df['Sesion'] = [extraer_sesion_agnostica(t) for t in df['Toma']]
    feat_cols = [c for c in df.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_raw = df[feat_cols].values
    N, D = X_raw.shape
    n_ch, n_pts = 3, D // 3
    y = df['Vocal'].values
    sesiones = df['Sesion'].values

    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')
    X_reshaped = X_raw.reshape(N, n_ch, n_pts)
    X_filt = np.zeros_like(X_reshaped)
    for i in range(N):
        for c in range(n_ch):
            X_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_reshaped[i, c, :])

    X_norm = np.zeros_like(X_filt)
    for s in np.unique(sesiones):
        mask = (sesiones == s)
        for c in range(n_ch):
            base_mean = np.mean(X_filt[mask, c, :10])
            base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
            X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max

    X_flat = X_norm.reshape(N, -1)
    return torch.tensor(X_flat, dtype=torch.float32), y, n_ch, n_pts, b_bw, a_bw

def cargar_datos_p5(b_bw, a_bw):
    toma_p5 = os.path.join(
        project_root,
        "EMG_desarrollo/base_de_datos_electrodos/2026-06-10/SecuenciaContinua_Prueba5_Sujeto1"
    )
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
    if win_rms % 2 == 0:
        win_rms += 1
    kernel_rms = np.ones(win_rms) / win_rms
    env_emg = np.zeros_like(sig_filt)
    for c in range(3):
        env_emg[c] = np.sqrt(np.maximum(0, np.convolve(sig_filt[c]**2, kernel_rms, mode='same')))

    win_mic = int(0.050 * fs)
    mic_env = np.convolve(np.abs(sig_mic), np.ones(win_mic) / win_mic, mode='same')
    picos_candidatos, _ = signal.find_peaks(mic_env, distance=int(1.2 * fs), height=2000)
    picos_fonacion = [p for p in picos_candidatos if p >= 6.0 * fs and p < (n_samples - int(1.0 * fs))]
    if len(picos_fonacion) > 125:
        picos_fonacion = picos_fonacion[:125]

    ruido_base_emg = np.median(env_emg[:, :n_noise_samples], axis=1, keepdims=True)
    half_win = int(1.0 * fs)
    pts_target = 20
    X_p5_list, y_p5_list = [], []

    for i, p in enumerate(picos_fonacion):
        start = p - half_win
        end = p + half_win
        if start < 0 or end > n_samples:
            continue
        seg_raw = np.maximum(env_emg[:, start:end] - ruido_base_emg, 0.0)
        M_supremo = np.max(seg_raw) + 1e-9
        seg_norm = seg_raw / M_supremo
        feat_p = []
        for c in range(3):
            resamp = np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_norm.shape[1]), seg_norm[c])
            feat_p.append(resamp)
        X_p5_list.append(np.concatenate(feat_p))
        v_ground = palabras_ground[i] if i < len(palabras_ground) else vocales[i % 5]
        y_p5_list.append(v_ground)

    X_p5 = np.array(X_p5_list)
    y_p5 = np.array(y_p5_list)
    N_p5 = len(X_p5)

    X_p5_reshaped = X_p5.reshape(N_p5, 3, pts_target)
    X_p5_filt = np.zeros_like(X_p5_reshaped)
    for i in range(N_p5):
        for c in range(3):
            X_p5_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_p5_reshaped[i, c, :])

    X_p5_norm = np.zeros_like(X_p5_filt)
    for c in range(3):
        base_mean = np.mean(X_p5_filt[:, c, :10])
        base_max = np.percentile(X_p5_filt[:, c, :], 95) - base_mean + 1e-6
        X_p5_norm[:, c, :] = (X_p5_filt[:, c, :] - base_mean) / base_max

    return torch.tensor(X_p5_norm.reshape(N_p5, -1), dtype=torch.float32), y_p5

# ==============================================================================
# 3. EVALUACION DE UNA CONFIGURACION EN 3D
# ==============================================================================
def evaluar_configuracion_3d(cfg, X_lucas_t, y_lucas, X_p5_t, y_p5, device, seed=100, epochs=350):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    n_ch, n_pts = 3, 20
    model = ParametricConvOrthogonalAE3D(
        in_channels=n_ch,
        time_pts=n_pts,
        conv_channels=cfg['channels'],
        kernel_size=cfg['kernel_size'],
        latent_dim=3,
        hidden_dim=64,
        act_name=cfg['act']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    I_3d = torch.eye(3, device=device)
    N_lucas = X_lucas_t.shape[0]

    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, z = model(X_lucas_t)
        loss_recon = nn.functional.mse_loss(recon, X_lucas_t)
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N_lucas - 1)
        loss_z = torch.sum((cov_z - I_3d) ** 2)
        loss = loss_recon + cfg['lambda_w'] * loss_w + cfg['lambda_z'] * loss_z
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        Z_lucas = model.encode(X_lucas_t).cpu().numpy()
        Z_p5 = model.encode(X_p5_t).cpu().numpy()

    # GMM en Lucas 3D
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
    pred_raw_lucas = gmm.fit_predict(Z_lucas)
    
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(vocales):
        for j in range(5):
            contingency[j, i] = np.sum((y_lucas == vl) & (pred_raw_lucas == j))
            
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    
    pred_lucas_idx = np.array([cluster_to_vocal[c] for c in pred_raw_lucas])
    y_lucas_idx = np.array([vocal_to_idx[v] for v in y_lucas])
    acc_lucas = accuracy_score(y_lucas_idx, pred_lucas_idx) * 100.0

    # Transferencia directa a P5 en 3D
    pred_raw_p5 = gmm.predict(Z_p5)
    pred_p5_idx = np.array([cluster_to_vocal[c] for c in pred_raw_p5])
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p5])
    acc_p5 = accuracy_score(y_p5_idx, pred_p5_idx) * 100.0

    # Desglose por vocal en P5
    vocal_accs = {}
    for idx, v in enumerate(vocales):
        mask = (y_p5_idx == idx)
        if np.sum(mask) > 0:
            vocal_accs[v] = np.mean(pred_p5_idx[mask] == idx) * 100.0
        else:
            vocal_accs[v] = 0.0

    # Media armonica entre Lucas y P5
    harmonic_acc = 2 * (acc_lucas * acc_p5) / (acc_lucas + acc_p5 + 1e-9)

    return {
        'acc_lucas': acc_lucas,
        'acc_p5': acc_p5,
        'harmonic_acc': harmonic_acc,
        'acc_A': vocal_accs['A'],
        'acc_E': vocal_accs['E'],
        'acc_I': vocal_accs['I'],
        'acc_O': vocal_accs['O'],
        'acc_U': vocal_accs['U'],
        'min_vocal_p5': min(vocal_accs.values()),
        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
        'Z_lucas': Z_lucas,
        'Z_p5': Z_p5,
        'pred_lucas_idx': pred_lucas_idx,
        'pred_p5_idx': pred_p5_idx,
        'gmm': gmm,
        'cluster_to_vocal': cluster_to_vocal
    }

# ==============================================================================
# 4. GRAFICO Y GUARDADO DE MODELO CAMPEON 3D
# ==============================================================================
def graficar_campeon_record_3d(cfg, res, y_lucas, y_p5, p_png):
    fig = plt.figure(figsize=(19, 5.8))
    
    # Subplot 1: Lucas en 3D
    ax0 = fig.add_subplot(1, 3, 1, projection='3d')
    Z_l = res['Z_lucas']
    for v in vocales:
        mask = (y_lucas == v)
        ax0.scatter(Z_l[mask, 0], Z_l[mask, 1], Z_l[mask, 2],
                    c=colores_oficiales[v], label=f"Vocal /{v.lower()}/",
                    alpha=0.7, edgecolors='none', s=25)
    ax0.set_title(f"Espacio Latente Lucas 3D: Exactitud {res['acc_lucas']:.1f}%", fontsize=11, fontweight='bold')
    ax0.set_xlabel("Z1")
    ax0.set_ylabel("Z2")
    ax0.set_zlabel("Z3")
    ax0.legend(frameon=True, fontsize=8, loc='upper right')
    ax0.grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 2: P5 en 3D
    ax1 = fig.add_subplot(1, 3, 2, projection='3d')
    Z_p = res['Z_p5']
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p5])
    pred_p5 = res['pred_p5_idx']
    aciertos = (y_p5_idx == pred_p5)
    
    for v in vocales:
        mask = (y_p5 == v) & aciertos
        ax1.scatter(Z_p[mask, 0], Z_p[mask, 1], Z_p[mask, 2],
                    c=colores_oficiales[v], marker='o', label=f"/{v.lower()}/ Correcto",
                    alpha=0.85, s=35)
        mask_err = (y_p5 == v) & (~aciertos)
        if np.any(mask_err):
            ax1.scatter(Z_p[mask_err, 0], Z_p[mask_err, 1], Z_p[mask_err, 2],
                        c=colores_oficiales[v], marker='x', label=f"/{v.lower()}/ Error",
                        alpha=0.9, s=45, linewidths=1.5)
            
    num_correct = int(np.sum(aciertos))
    ax1.set_title(f"Secuencia Continua P5 3D: {res['acc_p5']:.1f}% - {num_correct}/123 Pulsos", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Z1")
    ax1.set_ylabel("Z2")
    ax1.set_zlabel("Z3")
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # Subplot 3: Matriz de Confusion en P5
    ax2 = fig.add_subplot(1, 3, 3)
    cm = confusion_matrix(y_p5_idx, pred_p5, labels=range(5))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0
    im = ax2.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set(xticks=np.arange(5), yticks=np.arange(5),
            xticklabels=[f"/{v.lower()}/" for v in vocales],
            yticklabels=[f"/{v.lower()}/" for v in vocales],
            xlabel="Prediccion del Modelo",
            ylabel="Vocal Real Ground Truth",
            title=f"Matriz de Confusion P5: Armonica {res['harmonic_acc']:.1f}%")
    for r in range(5):
        for c in range(5):
            val = cm[r, c]
            pct = cm_norm[r, c]
            color = "white" if pct > 50 else "black"
            ax2.text(c, r, f"{val}\n{pct:.0f}%", ha="center", va="center", color=color, fontsize=9)
            
    plt.tight_layout()
    plt.savefig(p_png, dpi=150)
    plt.close()

def guardar_campeon_3d(cfg, res, y_lucas, y_p5, out_dir, records_count, prev_record):
    os.makedirs(out_dir, exist_ok=True)
    
    p_pt = os.path.join(out_dir, "modelo_campeon_conv_ortogonal_3d.pt")
    p_json = os.path.join(out_dir, "config_campeon_3d.json")
    p_p5_csv = os.path.join(out_dir, "proyecciones_campeon_p5_3d.csv")
    p_lucas_csv = os.path.join(out_dir, "proyecciones_campeon_lucas_3d.csv")
    p_png = os.path.join(out_dir, "grafico_campeon_record_3d.png")
    
    # Checkpoint completo
    checkpoint = {
        'model_state_dict': res['model_state_dict'],
        'cfg': cfg,
        'harmonic_acc': res['harmonic_acc'],
        'acc_lucas': res['acc_lucas'],
        'acc_p5': res['acc_p5'],
        'vocal_accs': {v: res[f'acc_{v}'] for v in vocales},
        'min_vocal_p5': res['min_vocal_p5'],
        'gmm_weights': res['gmm'].weights_,
        'gmm_means': res['gmm'].means_,
        'gmm_covariances': res['gmm'].covariances_,
        'cluster_to_vocal': res['cluster_to_vocal']
    }
    torch.save(checkpoint, p_pt)
    
    # JSON de configuracion y metricas
    cfg_export = {
        'channels': str(cfg['channels']),
        'kernel_size': int(cfg['kernel_size']),
        'act': str(cfg['act']),
        'lr': float(cfg['lr']),
        'lambda_w': float(cfg['lambda_w']),
        'lambda_z': float(cfg['lambda_z']),
        'harmonic_acc': float(res['harmonic_acc']),
        'acc_lucas': float(res['acc_lucas']),
        'acc_p5': float(res['acc_p5']),
        'min_vocal_p5': float(res['min_vocal_p5']),
        'acc_A': float(res['acc_A']),
        'acc_E': float(res['acc_E']),
        'acc_I': float(res['acc_I']),
        'acc_O': float(res['acc_O']),
        'acc_U': float(res['acc_U'])
    }
    with open(p_json, 'w') as f:
        json.dump(cfg_export, f, indent=2)
        
    # Proyecciones P5 3D
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p5])
    pred_p5 = res['pred_p5_idx']
    df_p5 = pd.DataFrame({
        'Pulso': np.arange(len(y_p5)) + 1,
        'Z1': res['Z_p5'][:, 0],
        'Z2': res['Z_p5'][:, 1],
        'Z3': res['Z_p5'][:, 2],
        'GroundTruth': y_p5,
        'Prediccion': [vocales[p] for p in pred_p5],
        'Acierto': (y_p5_idx == pred_p5).astype(int)
    })
    df_p5.to_csv(p_p5_csv, index=False)
    
    # Proyecciones Lucas 3D
    df_lucas = pd.DataFrame({
        'Muestra': np.arange(len(y_lucas)) + 1,
        'Z1': res['Z_lucas'][:, 0],
        'Z2': res['Z_lucas'][:, 1],
        'Z3': res['Z_lucas'][:, 2],
        'Vocal': y_lucas,
        'Prediccion': [vocales[p] for p in res['pred_lucas_idx']],
        'Acierto': (np.array([vocal_to_idx[v] for v in y_lucas]) == res['pred_lucas_idx']).astype(int)
    })
    df_lucas.to_csv(p_lucas_csv, index=False)
    
    # Grafico comparativo
    graficar_campeon_record_3d(cfg, res, y_lucas, y_p5, p_png)

def anunciar_record_3d(cfg, res, total_comb, current_idx, records_count, prev_record, out_dir):
    num_correct_p5 = int(round(res['acc_p5'] * 123 / 100.0))
    sep = "=" * 80
    print("\n" + sep)
    print(f"RECORD HISTORICO 3D SUPERADO: {res['harmonic_acc']:.2f}% | Record previo: {prev_record:.2f}%")
    print(f"Combinacion [{current_idx}/{total_comb}] | Hito #{records_count}")
    print(f"Arquitectura 3D: Kernel={cfg['kernel_size']} | Canales={cfg['channels']} | Activacion={cfg['act']}")
    print(f"Optimizacion: Tasa de Aprendizaje={cfg['lr']} | Ortogonalidad W={cfg['lambda_w']} | Decorrelacion Z={cfg['lambda_z']}")
    print("Rendimiento:")
    print(f"  - Media Armonica: {res['harmonic_acc']:.2f}%")
    print(f"  - Lucas (Entrenamiento 3D): {res['acc_lucas']:.2f}%")
    print(f"  - Secuencia Continua P5 (Transferencia 3D): {res['acc_p5']:.2f}% ({num_correct_p5}/123 pulsos)")
    print(f"  - Desglose P5: /a/={res['acc_A']:.1f}% | /e/={res['acc_E']:.1f}% | /i/={res['acc_I']:.1f}% | /o/={res['acc_O']:.1f}% | /u/={res['acc_U']:.1f}% | Piso: {res['min_vocal_p5']:.1f}%")
    print(f"Artefactos del campeon 3D exportados en:")
    print(f"  -> {os.path.join(out_dir, 'modelo_campeon_conv_ortogonal_3d.pt')}")
    print(f"  -> {os.path.join(out_dir, 'config_campeon_3d.json')}")
    print(f"  -> {os.path.join(out_dir, 'proyecciones_campeon_p5_3d.csv')}")
    print(f"  -> {os.path.join(out_dir, 'grafico_campeon_record_3d.png')}")
    print(sep + "\n")
    sys.stdout.flush()
    
    # Triple campanilla en terminal
    sys.stdout.write('\a\a\a')
    sys.stdout.flush()
    
    # Locucion no bloqueante si esta disponible
    try:
        msg_voz = f"Nuevo record 3D detectado. Media armonica {res['harmonic_acc']:.1f} por ciento."
        subprocess.Popen(['spd-say', '-t', 'female2', msg_voz], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

# ==============================================================================
# 5. BARRIDO COMPLETO 3D
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Grid Search de Autoencoders Convolucionales Ortogonales en 3D")
    parser.add_argument('--mode', type=str, default='3600', choices=['3600', '5760', 'quick', 'test'],
                        help="Modo del barrido: 3600 (default), 5760 (max epic), quick (96) o test (4)")
    parser.add_argument('--epochs', type=int, default=350, help="Numero de epocas por configuracion (default: 350)")
    parser.add_argument('--seed', type=int, default=100, help="Semilla aleatoria (default: 100)")
    parser.add_argument('--baseline-record', type=float, default=80.00, help="Record previo de media armonica a batir (default: 80.00)")
    parser.add_argument('--resume', action='store_true', help="Reanudar barrido si ya existe el CSV de resultados")
    args = parser.parse_args()

    os.makedirs(out_dir_default, exist_ok=True)
    csv_results = os.path.join(out_dir_default, f"resultados_grid_search_3d_{args.mode}.csv")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print("=" * 80)
    print("INICIANDO GRID SEARCH DE AUTOENCODERS CONVOLUCIONALES ORTOGONALES EN 3D")
    print(f"Dispositivo de computo detectado: {device}")
    print("=" * 80)

    print("[Carga] Cargando y preprocesando datos de Lucas...")
    X_lucas_t, y_lucas, n_ch, n_pts, b_bw, a_bw = cargar_datos_lucas()
    X_lucas_t = X_lucas_t.to(device)
    print(f"  Lucas: {len(X_lucas_t)} muestras cargadas en {device}.")

    print("[Carga] Extrayendo ventanas continuas de SecuenciaContinua_Prueba5_Sujeto1...")
    X_p5_t, y_p5 = cargar_datos_p5(b_bw, a_bw)
    X_p5_t = X_p5_t.to(device)
    print(f"  Secuencia Continua P5: {len(X_p5_t)} pulsos cargados en {device}.")

    # Definir espacio de busqueda segun modo
    if args.mode == 'test':
        grid_params = {
            'channels': [(4, 8)],
            'kernel_size': [7],
            'act': ['tanh'],
            'lr': [0.003],
            'lambda_w': [1.2, 2.0],
            'lambda_z': [0.25, 0.50]
        }
    elif args.mode == 'quick':
        grid_params = {
            'channels': [(4, 8), (6, 12)],
            'kernel_size': [5, 7],
            'act': ['tanh', 'gelu'],
            'lr': [0.002, 0.003],
            'lambda_w': [0.8, 1.2, 2.0],
            'lambda_z': [0.25, 0.50]
        }
    elif args.mode == '3600':
        grid_params = {
            'channels': [(3, 6), (4, 8), (4, 12), (6, 12), (8, 16)],
            'kernel_size': [3, 5, 7, 9],
            'act': ['tanh', 'gelu'],
            'lr': [0.002, 0.003, 0.004],
            'lambda_w': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
            'lambda_z': [0.15, 0.25, 0.35, 0.45, 0.60]
        }
    elif args.mode == '5760':
        grid_params = {
            'channels': [(3, 6), (4, 8), (4, 12), (6, 12), (8, 16), (8, 24)],
            'kernel_size': [3, 5, 7, 9],
            'act': ['tanh', 'gelu'],
            'lr': [0.002, 0.003, 0.004, 0.006],
            'lambda_w': [0.6, 0.8, 1.0, 1.2, 1.5, 2.0],
            'lambda_z': [0.15, 0.25, 0.35, 0.50, 0.70]
        }

    keys = list(grid_params.keys())
    combinations = [dict(zip(keys, prod)) for prod in itertools.product(*grid_params.values())]
    total_comb = len(combinations)
    
    print(f"\n[Configuracion] Modo seleccionado: {args.mode}")
    print(f"[Configuracion] Total de combinaciones a explorar: {total_comb}")
    print(f"[Configuracion] Epocas por ejecucion: {args.epochs} | Semilla: {args.seed}")
    print(f"[Configuracion] Linea base a batir: {args.baseline_record:.2f}% (Media Armonica)")
    print(f"[Configuracion] Archivo de resultados: {csv_results}\n")

    records = []
    completed_indices = set()
    best_harmonic = float(args.baseline_record)
    best_cfg = None
    records_count = 0

    if args.resume and os.path.exists(csv_results):
        try:
            df_prev = pd.read_csv(csv_results)
            completed_indices = set(df_prev['idx'].values)
            records = df_prev.to_dict('records')
            if len(df_prev) > 0 and 'harmonic_acc' in df_prev.columns:
                max_prev = df_prev['harmonic_acc'].max()
                if max_prev > best_harmonic:
                    best_harmonic = float(max_prev)
            print(f"[Reanudacion] Se detectaron {len(completed_indices)} configuraciones previas en el archivo.")
            print(f"[Reanudacion] Record maximo detectado hasta ahora: {best_harmonic:.2f}%\n")
        except Exception as e:
            print(f"[Reanudacion] No se pudo leer el archivo existente ({e}). Iniciando desde el principio.")

    t_start = time.time()

    for idx, cfg in enumerate(combinations):
        current_idx = idx + 1
        if current_idx in completed_indices:
            continue

        t_iter_start = time.time()
        res = evaluar_configuracion_3d(cfg, X_lucas_t, y_lucas, X_p5_t, y_p5, device=device, seed=args.seed, epochs=args.epochs)
        t_iter = time.time() - t_iter_start

        row = {
            'idx': current_idx,
            'channels': str(cfg['channels']),
            'kernel_size': cfg['kernel_size'],
            'act': cfg['act'],
            'lr': cfg['lr'],
            'lambda_w': cfg['lambda_w'],
            'lambda_z': cfg['lambda_z'],
            'acc_lucas': round(res['acc_lucas'], 2),
            'acc_p5': round(res['acc_p5'], 2),
            'harmonic_acc': round(res['harmonic_acc'], 2),
            'acc_A': round(res['acc_A'], 1),
            'acc_E': round(res['acc_E'], 1),
            'acc_I': round(res['acc_I'], 1),
            'acc_O': round(res['acc_O'], 1),
            'acc_U': round(res['acc_U'], 1),
            'min_vocal_p5': round(res['min_vocal_p5'], 1),
            'duration_s': round(t_iter, 2)
        }
        records.append(row)

        # Guardado incremental inmediato
        pd.DataFrame(records).to_csv(csv_results, index=False)

        # Deteccion de nuevo record historico
        # Criterio: superar la media armonica y mantener un piso minimo equilibrado en P5 (>= 50%)
        if res['harmonic_acc'] > best_harmonic and res['min_vocal_p5'] >= 50.0:
            prev_record = best_harmonic
            best_harmonic = res['harmonic_acc']
            records_count += 1
            best_cfg = row
            guardar_campeon_3d(cfg, res, y_lucas, y_p5, out_dir_default, records_count, prev_record)
            anunciar_record_3d(cfg, res, total_comb, current_idx, records_count, prev_record, out_dir_default)

        # Monitoreo de progreso en tiempo real
        elapsed = time.time() - t_start
        processed = len(records)
        avg_time = elapsed / max(processed, 1)
        remaining = total_comb - current_idx
        eta_s = avg_time * remaining

        print(f"[{current_idx:4d}/{total_comb:4d}] ({((current_idx)/total_comb)*100:5.1f}%) | "
              f"K={cfg['kernel_size']} Ch={cfg['channels']} Act={cfg['act']} lr={cfg['lr']} "
              f"lw={cfg['lambda_w']} lz={cfg['lambda_z']} | "
              f"Lucas 3D: {res['acc_lucas']:5.1f}% | P5 3D: {res['acc_p5']:5.1f}% | Armonica: {res['harmonic_acc']:5.1f}% | "
              f"ETA: {eta_s/60:4.1f}m")

    print("\n" + "=" * 80)
    print("GRID SEARCH EN 3D FINALIZADO")
    print(f"Resultados consolidados exportados a: {csv_results}")
    print(f"Mejor combinacion historica hallada: {best_harmonic:.2f}% (Media Armonica)")
    print(f"Configuracion campeona: {best_cfg}")
    print("=" * 80)

if __name__ == '__main__':
    main()
