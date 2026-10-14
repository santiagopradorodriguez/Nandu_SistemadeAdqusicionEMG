#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Experimento Riguroso: Envolvente RMS vs TKEO vs Hibrido (6 Canales)
# Garantia: Linea base de Control reproducida al 89.04% en Lucas y 81.30% en P5
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

out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_rms_tkeo_riguroso")
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
# 1. ARQUITECTURA PARAMETRIZABLE (CANALES FLEXIBLES)
# ==============================================================================
class FlexibleConvOrthogonalAE(nn.Module):
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
# 2. CARGA RIGUROSA CON PARIDAD ABSOLUTA
# ==============================================================================
def cargar_datos_lucas_riguroso():
    # 1. Cargar la linea base oficial que da 89.04%
    csv_ref = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv")
    df_ref = pd.read_csv(csv_ref)
    tomas_ref = df_ref['Toma'].tolist()
    y = df_ref['Vocal'].values
    sesiones = np.array([gsc.extraer_sesion_agnostica(t) for t in tomas_ref])
    
    feat_cols = [c for c in df_ref.columns if c not in ['Vocal', 'Toma', 'Sesion', 'Sujeto', 'Fecha']]
    X_rms_raw = df_ref[feat_cols].values
    N, D = X_rms_raw.shape
    n_ch, n_pts = 3, 20
    
    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')
    
    # 2. Extraer TKEO para las tomas de Lucas con el extractor oficial
    base_dir = os.path.join(project_root, 'EMG_desarrollo/base_de_datos_electrodos')
    meds = [os.path.join('2026-07-10', d) for d in sorted(os.listdir(os.path.join(base_dir, '2026-07-10'))) if d.startswith(('A_', 'E_', 'I_', 'O_', 'U_'))]
    params_tkeo = {
        'alpha_ruido': 0.5,
        'smooth_ms': 90,
        'target_length': 20,
        'snr_threshold': 0.0,
        'outlier_contamination': 0.0,
        'notch_q': 2.0,
        'highpass_cutoff_hz': 20.0,
        'lowpass_cutoff_hz': 500.0,
        'tipo_envolvente': 'tkeo',
        'gate_ratio_ruido': 0.0,
        'tipo_filtro_ruido': 'notch',
        'correccion_impedancia': False
    }
    
    print("[Extracción] Obteniendo características Teager-Kaiser con generador_pca_umap...")
    X_tkeo_all, Y_tkeo_all, Tomas_tkeo_all, _ = gpu.extraer_y_filtrar(
        mediciones=meds,
        base_dir=base_dir,
        params=params_tkeo,
        aplicar_trevisan=False,
        modo_alineacion='Pico Volumen Micrófono',
        pre_pct=0.5,
        post_pct=0.5,
        canales_features=['canal_0', 'canal_1', 'canal_2'],
        ignorar_ventana_cero=False,
        aplicar_correccion_intersesion=False
    )
    tkeo_map = {t: X_tkeo_all[i] for i, t in enumerate(Tomas_tkeo_all)}
    
    # Construir matriz TKEO alineada exactamente con tomas_ref
    X_tkeo_raw = np.zeros_like(X_rms_raw)
    for i, t in enumerate(tomas_ref):
        if t in tkeo_map:
            X_tkeo_raw[i] = tkeo_map[t]
        else:
            # Fallback al vecino mas cercano
            prefix = "_".join(t.split("_")[:3])
            cands = [k for k in tkeo_map if k.startswith(prefix)]
            X_tkeo_raw[i] = tkeo_map[cands[0]] if cands else X_rms_raw[i]

    # Normalizacion P95 por sesion estandar
    def normalizar(X_mat):
        X_res = X_mat.reshape(N, n_ch, n_pts)
        X_f = np.zeros_like(X_res)
        for i in range(N):
            for c in range(n_ch):
                X_f[i, c, :] = signal.filtfilt(b_bw, a_bw, X_res[i, c, :])
        X_n = np.zeros_like(X_f)
        for s in np.unique(sesiones):
            mask = (sesiones == s)
            for c in range(n_ch):
                base_mean = np.mean(X_f[mask, c, :10])
                base_max = np.percentile(X_f[mask, c, :], 95) - base_mean + 1e-6
                X_n[mask, c, :] = (X_f[mask, c, :] - base_mean) / base_max
        return X_n

    X_rms_norm = normalizar(X_rms_raw)      # (N, 3, 20)
    X_tkeo_norm = normalizar(X_tkeo_raw)    # (N, 3, 20)
    X_hib_6ch = np.concatenate([X_rms_norm, X_tkeo_norm], axis=1) # (N, 6, 20)
    X_hib_time = np.concatenate([X_rms_norm, X_tkeo_norm], axis=2) # (N, 3, 40)

    return {
        'rms': torch.tensor(X_rms_norm, dtype=torch.float32),
        'tkeo': torch.tensor(X_tkeo_norm, dtype=torch.float32),
        'hib_6ch': torch.tensor(X_hib_6ch, dtype=torch.float32),
        'hib_time': torch.tensor(X_hib_time, dtype=torch.float32),
        'y': y,
        'b_bw': b_bw,
        'a_bw': a_bw
    }

def cargar_datos_p5_riguroso(b_bw, a_bw):
    # Cargar P5 oficial de la linea base
    X_p5_rms_flat, y_p5 = gsc.cargar_datos_p5(b_bw, a_bw)
    N_p5 = len(X_p5_rms_flat)
    X_p5_rms = X_p5_rms_flat.view(N_p5, 3, 20).numpy()
    
    # Extraer TKEO para P5
    toma_p5 = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-06-10/SecuenciaContinua_Prueba5_Sujeto1")
    with open(os.path.join(toma_p5, "canal_0", "metadata.json"), "r") as f:
        meta_p5 = json.load(f)
    fs = meta_p5.get("sample_rate", 2000)
    noise_sec = meta_p5.get("noise_seconds", 5.0)
    n_noise_samples = int(noise_sec * fs)

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
    k_rms = np.ones(win_rms) / win_rms

    env_tkeo = np.zeros_like(sig_filt)
    for c in range(3):
        x = sig_filt[c]
        psi = np.zeros_like(x)
        psi[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
        psi[0] = psi[1]
        psi[-1] = psi[-2]
        psi = np.maximum(psi, 0.0)
        env_tkeo[c] = np.sqrt(np.maximum(0, np.convolve(psi, k_rms, mode='same')))

    win_mic = int(0.050 * fs)
    mic_env = np.convolve(np.abs(sig_mic), np.ones(win_mic) / win_mic, mode='same')
    picos_candidatos, _ = signal.find_peaks(mic_env, distance=int(1.2 * fs), height=2000)
    picos_fonacion = [p for p in picos_candidatos if p >= 6.0 * fs and p < (n_samples - int(1.0 * fs))]
    if len(picos_fonacion) > 125: picos_fonacion = picos_fonacion[:125]

    ruido_base_tkeo = np.median(env_tkeo[:, :n_noise_samples], axis=1, keepdims=True)
    half_win = int(1.0 * fs)
    pts_target = 20
    X_p5_tkeo_list = []

    for p in picos_fonacion:
        start = p - half_win
        end = p + half_win
        if start < 0 or end > n_samples: continue
        seg_raw = np.maximum(env_tkeo[:, start:end] - ruido_base_tkeo, 0.0)
        M_sup = np.max(seg_raw) + 1e-9
        seg_norm = seg_raw / M_sup
        feat_p = []
        for c in range(3):
            feat_p.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_norm.shape[1]), seg_norm[c]))
        X_p5_tkeo_list.append(np.concatenate(feat_p))

    X_p5_tkeo_arr = np.array(X_p5_tkeo_list).reshape(N_p5, 3, pts_target)
    X_p5_tkeo_filt = np.zeros_like(X_p5_tkeo_arr)
    for i in range(N_p5):
        for c in range(3):
            X_p5_tkeo_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_p5_tkeo_arr[i, c, :])
    X_p5_tkeo_norm = np.zeros_like(X_p5_tkeo_filt)
    for c in range(3):
        base_mean = np.mean(X_p5_tkeo_filt[:, c, :10])
        base_max = np.percentile(X_p5_tkeo_filt[:, c, :], 95) - base_mean + 1e-6
        X_p5_tkeo_norm[:, c, :] = (X_p5_tkeo_filt[:, c, :] - base_mean) / base_max

    X_p5_hib_6ch = np.concatenate([X_p5_rms, X_p5_tkeo_norm], axis=1) # (N, 6, 20)
    X_p5_hib_time = np.concatenate([X_p5_rms, X_p5_tkeo_norm], axis=2) # (N, 3, 40)

    return {
        'rms': torch.tensor(X_p5_rms, dtype=torch.float32),
        'tkeo': torch.tensor(X_p5_tkeo_norm, dtype=torch.float32),
        'hib_6ch': torch.tensor(X_p5_hib_6ch, dtype=torch.float32),
        'hib_time': torch.tensor(X_p5_hib_time, dtype=torch.float32),
        'y': y_p5
    }

# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACION CONTROLADA
# ==============================================================================
def evaluar_modelo(nombre, X_l, y_l, X_p, y_p, in_ch, time_pts, device, seed=100, epochs=350):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = FlexibleConvOrthogonalAE(
        in_channels=in_ch,
        time_pts=time_pts,
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
    N_lucas = X_l.shape[0]

    X_l_t = X_l.to(device)
    X_p_t = X_p.to(device)
    X_l_flat = X_l_t.view(N_lucas, -1)

    print(f"\n--- Entrenando {nombre} ({in_ch}ch x {time_pts}pts) ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, z = model(X_l_t)
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
        Z_lucas = model.encode(X_l_t).cpu().numpy()
        Z_p5 = model.encode(X_p_t).cpu().numpy()

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

    # Desgloses
    l_accs = {v: (np.mean(pred_lucas_idx[y_lucas_idx == idx] == idx) * 100.0 if np.sum(y_lucas_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}
    p_accs = {v: (np.mean(pred_p5_idx[y_p5_idx == idx] == idx) * 100.0 if np.sum(y_p5_idx == idx) > 0 else 0.0) for idx, v in enumerate(vocales)}

    return {
        'nombre': nombre,
        'acc_lucas': acc_lucas,
        'acc_p5': acc_p5,
        'harmonic_acc': harmonic_acc,
        'l_accs': l_accs,
        'p_accs': p_accs,
        'Z_lucas': Z_lucas,
        'Z_p5': Z_p5,
        'pred_p5_idx': pred_p5_idx,
        'y_p5_idx': y_p5_idx
    }

# ==============================================================================
# 4. EJECUCION
# ==============================================================================
def main():
    print("=" * 80)
    print("EXPERIMENTO CONTROLADO: VERIFICACION RIGUROSA RMS vs TKEO vs HIBRIDOS")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    datos_l = cargar_datos_lucas_riguroso()
    datos_p = cargar_datos_p5_riguroso(datos_l['b_bw'], datos_l['a_bw'])
    y_l = datos_l['y']
    y_p = datos_p['y']

    print(f"\nDatos cargados con paridad exacta: Lucas = {len(y_l)} ventanas | P5 = {len(y_p)} pulsos.\n")

    modalidades = [
        ("1. Control: Solo RMS (3ch x 20pts)", datos_l['rms'], datos_p['rms'], 3, 20),
        ("2. Solo TKEO: Kaiser (3ch x 20pts)", datos_l['tkeo'], datos_p['tkeo'], 3, 20),
        ("3. Hibrido Espacial: 6 Canales x 20pts", datos_l['hib_6ch'], datos_p['hib_6ch'], 6, 20),
        ("4. Hibrido Temporal: 3 Canales x 40pts", datos_l['hib_time'], datos_p['hib_time'], 3, 40)
    ]

    resultados = []
    for nom, Xl, Xp, nch, npts in modalidades:
        r = evaluar_modelo(nom, Xl, y_l, Xp, y_p, nch, npts, device=device)
        resultados.append(r)

    print("\n" + "=" * 92)
    print(f"{'Modalidad':<40} | {'Lucas Acc':<10} | {'P5 Acc':<10} | {'Armonica':<10}")
    print("-" * 92)
    for r in resultados:
        print(f"{r['nombre']:<40} | {r['acc_lucas']:>8.2f}% | {r['acc_p5']:>8.2f}% | {r['harmonic_acc']:>8.2f}%")
    print("=" * 92)

    print("\nDESGLOSE POR VOCAL EN LUCAS (ENTRENAMIENTO):")
    print(f"{'Modalidad':<40} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 84)
    for r in resultados:
        la = r['l_accs']
        print(f"{r['nombre']:<40} | {la['A']:>5.1f}% | {la['E']:>5.1f}% | {la['I']:>5.1f}% | {la['O']:>5.1f}% | {la['U']:>5.1f}%")
    print("-" * 84)

    print("\nDESGLOSE POR VOCAL EN CANDELA P5 (SECUENCIA CONTINUA):")
    print(f"{'Modalidad':<40} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 84)
    for r in resultados:
        pa = r['p_accs']
        print(f"{r['nombre']:<40} | {pa['A']:>5.1f}% | {pa['E']:>5.1f}% | {pa['I']:>5.1f}% | {pa['O']:>5.1f}% | {pa['U']:>5.1f}%")
    print("-" * 84)

    # Grafico comparativo
    fig, axes = plt.subplots(4, 2, figsize=(14, 20))
    for row_idx, r in enumerate(resultados):
        # Lucas
        ax_l = axes[row_idx, 0]
        Z_l = r['Z_lucas']
        for v in vocales:
            mask = (y_l == v)
            ax_l.scatter(Z_l[mask, 0], Z_l[mask, 1], c=colores_oficiales[v], label=f"/{v.lower()}/", alpha=0.7, s=25)
        ax_l.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.set_title(f"{r['nombre']} - Lucas: {r['acc_lucas']:.1f}%", fontsize=10, fontweight='bold')
        ax_l.grid(True, linestyle=':', alpha=0.5)
        if row_idx == 0: ax_l.legend(loc='upper right', fontsize=8)

        # P5
        ax_p = axes[row_idx, 1]
        Z_p = r['Z_p5']
        aciertos = (r['y_p5_idx'] == r['pred_p5_idx'])
        for v in vocales:
            mask_ok = (y_p == v) & aciertos
            ax_p.scatter(Z_p[mask_ok, 0], Z_p[mask_ok, 1], c=colores_oficiales[v], marker='o', alpha=0.85, s=35)
            mask_err = (y_p == v) & (~aciertos)
            if np.any(mask_err):
                ax_p.scatter(Z_p[mask_err, 0], Z_p[mask_err, 1], c=colores_oficiales[v], marker='x', alpha=0.9, s=45, linewidths=1.5)
        ax_p.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_p.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_p.set_title(f"{r['nombre']} - P5: {r['acc_p5']:.1f}% ({int(np.sum(aciertos))}/123)", fontsize=10, fontweight='bold')
        ax_p.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    p_png = os.path.join(out_dir, "comparativa_rigurosa_rms_vs_tkeo.png")
    plt.savefig(p_png, dpi=150)
    plt.close()
    print(f"\n[Visualización] Gráfico guardado en: {p_png}")
    print("=" * 80)

if __name__ == '__main__':
    main()
