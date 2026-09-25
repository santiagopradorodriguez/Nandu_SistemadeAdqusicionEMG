#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Experimento Comparativo: RMS vs TKEO vs Hibrido (RMS + TKEO)
# Evaluacion: Exactitud GMM Lucas + Transferencia Directa P5
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
out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_rms_tkeo")
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
# 1. ARQUITECTURA PARAMETRIZABLE (CANALES DE ENTRADA CONFIGURABLES)
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
        
        if act_name == 'tanh':
            self.act = nn.Tanh()
        else:
            self.act = nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * time_pts, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, latent_dim, bias=False)
        
        self.dfc1 = nn.Linear(latent_dim, hidden_dim, bias=False)
        self.dfc2 = nn.Linear(hidden_dim, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

        # Buffers de identidad
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
# 2. EXTRACCION DUAL DIRECTA: RMS Y TEAGER-KAISER EN LUCAS Y P5
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

def extraer_lucas_dual():
    """
    Extrae las 35 tomas de Lucas calculando simultaneamente la envolvente RMS
    y la envolvente Teager-Kaiser (TKEO) para cada pulso exacto.
    """
    base_lucas = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos/2026-07-10")
    tomas = sorted([d for d in os.listdir(base_lucas) if os.path.isdir(os.path.join(base_lucas, d)) and not d.startswith('.') and d != 'UMBRALES'])
    
    # Cargamos la lista de tomas validas del CSV de referencia para mantener paridad
    csv_ref = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv")
    df_ref = pd.read_csv(csv_ref)
    valid_tomas_set = set(df_ref['Toma'].values)
    
    pts_target = 20
    b_bw, a_bw = signal.butter(N=3, Wn=0.3, btype='low')
    
    X_rms_list, X_tkeo_list, y_list, tomas_list = [], [], [], []
    
    print(f"[Extracción Lucas] Procesando {len(tomas)} tomas...")
    for idx_toma, t_name in enumerate(tomas):
        t_path = os.path.join(base_lucas, t_name)
        p_meta = os.path.join(t_path, "canal_0", "metadata.json")
        if not os.path.exists(p_meta):
            continue
        with open(p_meta, "r") as f:
            meta = json.load(f)
        fs = meta.get("sample_rate", 2000)
        vocal = meta.get("letra", t_name.split('_')[0])
        noise_sec = meta.get("noise_seconds", 5.0)
        n_noise_samples = int(noise_sec * fs)
        bpm = meta.get("bpm", 40)
        muestras_pulso = int(round((60.0 / bpm) * fs))
        
        # Cargar los 4 canales
        sigs = []
        for ch in range(4):
            p_wav = os.path.join(t_path, f"canal_{ch}", "grabacion.wav")
            _, d_wav = wavfile.read(p_wav)
            sigs.append(d_wav.astype(np.float64))
        sig_emg = np.stack(sigs[:3], axis=0)
        sig_mic = sigs[3]
        n_samples = sig_emg.shape[1]
        
        # Filtros bioelectricos estándar: Notch 50 Hz + Bandpass 20-500 Hz
        b_notch, a_notch = signal.iirnotch(50.0, 2.0, fs)
        b_band, a_band = signal.butter(2, [20.0, 500.0], 'bandpass', fs=fs)
        sig_filt = np.zeros_like(sig_emg)
        for c in range(3):
            s_n = signal.filtfilt(b_notch, a_notch, sig_emg[c])
            sig_filt[c] = signal.filtfilt(b_band, a_band, s_n)
            
        # 1. Envolvente RMS (90 ms)
        win_rms = int(0.090 * fs)
        if win_rms % 2 == 0:
            win_rms += 1
        k_rms = np.ones(win_rms) / win_rms
        env_rms = np.zeros_like(sig_filt)
        for c in range(3):
            env_rms[c] = np.sqrt(np.maximum(0, np.convolve(sig_filt[c]**2, k_rms, mode='same')))
            
        # 2. Envolvente Teager-Kaiser (TKEO) con suavizado de 90 ms
        env_tkeo = np.zeros_like(sig_filt)
        for c in range(3):
            x = sig_filt[c]
            psi = np.zeros_like(x)
            psi[1:-1] = x[1:-1]**2 - x[:-2] * x[2:]
            psi[0] = psi[1]
            psi[-1] = psi[-2]
            psi = np.maximum(psi, 0.0)
            env_tkeo[c] = np.sqrt(np.maximum(0, np.convolve(psi, k_rms, mode='same')))
            
        # Deteccion de picos en micrófono
        win_mic = int(0.050 * fs)
        mic_env = np.convolve(np.abs(sig_mic), np.ones(win_mic) / win_mic, mode='same')
        dist_picos = int(0.8 * muestras_pulso)
        min_h = np.max(mic_env) * 0.20
        picos_mic, _ = signal.find_peaks(mic_env, distance=dist_picos, height=min_h)
        
        # Segmentacion simétrica pre/post 50%
        half_win = int(0.50 * muestras_pulso)
        ruido_rms = np.median(env_rms[:, :n_noise_samples], axis=1, keepdims=True)
        ruido_tkeo = np.median(env_tkeo[:, :n_noise_samples], axis=1, keepdims=True)
        
        for w_idx, p in enumerate(picos_mic):
            toma_win_id = f"{t_name}_Win{w_idx}"
            if toma_win_id not in valid_tomas_set:
                continue
            start = p - half_win
            end = p + half_win
            if start < 0 or end > n_samples:
                continue
                
            # Ventana RMS
            seg_rms = np.maximum(env_rms[:, start:end] - ruido_rms, 0.0)
            M_rms = np.max(seg_rms) + 1e-9
            seg_rms_norm = seg_rms / M_rms
            feat_rms = []
            for c in range(3):
                feat_rms.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_rms_norm.shape[1]), seg_rms_norm[c]))
                
            # Ventana TKEO
            seg_tkeo = np.maximum(env_tkeo[:, start:end] - ruido_tkeo, 0.0)
            M_tkeo = np.max(seg_tkeo) + 1e-9
            seg_tkeo_norm = seg_tkeo / M_tkeo
            feat_tkeo = []
            for c in range(3):
                feat_tkeo.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_tkeo_norm.shape[1]), seg_tkeo_norm[c]))
                
            X_rms_list.append(np.concatenate(feat_rms))
            X_tkeo_list.append(np.concatenate(feat_tkeo))
            y_list.append(vocal)
            tomas_list.append(toma_win_id)
            
        print(f"  [{idx_toma+1}/{len(tomas)}] {t_name}: {len(X_rms_list)} pulsos acumulados.")

    X_rms = np.array(X_rms_list)
    X_tkeo = np.array(X_tkeo_list)
    y = np.array(y_list)
    sesiones = np.array([extraer_sesion_agnostica(t) for t in tomas_list])
    N = len(X_rms)
    
    # Filtrado Butterworth suave y normalizacion P95 por canal (estándar de la línea base)
    def normalizar_tensor(X_arr):
        X_reshaped = X_arr.reshape(N, 3, pts_target)
        X_filt = np.zeros_like(X_reshaped)
        for i in range(N):
            for c in range(3):
                X_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_reshaped[i, c, :])
        X_norm = np.zeros_like(X_filt)
        for s in np.unique(sesiones):
            mask = (sesiones == s)
            for c in range(3):
                base_mean = np.mean(X_filt[mask, c, :10])
                base_max = np.percentile(X_filt[mask, c, :], 95) - base_mean + 1e-6
                X_norm[mask, c, :] = (X_filt[mask, c, :] - base_mean) / base_max
        return X_norm.reshape(N, 3, pts_target)

    X_rms_norm = normalizar_tensor(X_rms)
    X_tkeo_norm = normalizar_tensor(X_tkeo)
    
    # Hibrido: 6 canales x 20 puntos
    X_hibrido = np.concatenate([X_rms_norm, X_tkeo_norm], axis=1) # (N, 6, 20)
    
    return {
        'rms': torch.tensor(X_rms_norm, dtype=torch.float32),
        'tkeo': torch.tensor(X_tkeo_norm, dtype=torch.float32),
        'hibrido': torch.tensor(X_hibrido, dtype=torch.float32),
        'y': y,
        'b_bw': b_bw,
        'a_bw': a_bw
    }

def extraer_p5_dual(b_bw, a_bw):
    """
    Extrae la secuencia continua P5 calculando RMS y TKEO en paralelo para cada pulso.
    """
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
    if win_rms % 2 == 0:
        win_rms += 1
    k_rms = np.ones(win_rms) / win_rms
    
    # RMS
    env_rms = np.zeros_like(sig_filt)
    for c in range(3):
        env_rms[c] = np.sqrt(np.maximum(0, np.convolve(sig_filt[c]**2, k_rms, mode='same')))
        
    # TKEO
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
    if len(picos_fonacion) > 125:
        picos_fonacion = picos_fonacion[:125]

    ruido_rms = np.median(env_rms[:, :n_noise_samples], axis=1, keepdims=True)
    ruido_tkeo = np.median(env_tkeo[:, :n_noise_samples], axis=1, keepdims=True)
    half_win = int(1.0 * fs)
    pts_target = 20

    X_p5_rms_list, X_p5_tkeo_list, y_p5_list = [], [], []

    for i, p in enumerate(picos_fonacion):
        start = p - half_win
        end = p + half_win
        if start < 0 or end > n_samples:
            continue
            
        # RMS P5
        seg_rms = np.maximum(env_rms[:, start:end] - ruido_rms, 0.0)
        M_rms = np.max(seg_rms) + 1e-9
        seg_rms_norm = seg_rms / M_rms
        feat_rms = []
        for c in range(3):
            feat_rms.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_rms_norm.shape[1]), seg_rms_norm[c]))
            
        # TKEO P5
        seg_tkeo = np.maximum(env_tkeo[:, start:end] - ruido_tkeo, 0.0)
        M_tkeo = np.max(seg_tkeo) + 1e-9
        seg_tkeo_norm = seg_tkeo / M_tkeo
        feat_tkeo = []
        for c in range(3):
            feat_tkeo.append(np.interp(np.linspace(0, 1, pts_target), np.linspace(0, 1, seg_tkeo_norm.shape[1]), seg_tkeo_norm[c]))
            
        X_p5_rms_list.append(np.concatenate(feat_rms))
        X_p5_tkeo_list.append(np.concatenate(feat_tkeo))
        v_ground = palabras_ground[i] if i < len(palabras_ground) else vocales[i % 5]
        y_p5_list.append(v_ground)

    N_p5 = len(X_p5_rms_list)
    def normalizar_p5(X_arr, n_c=3):
        X_reshaped = np.array(X_arr).reshape(N_p5, n_c, pts_target)
        X_filt = np.zeros_like(X_reshaped)
        for i in range(N_p5):
            for c in range(n_c):
                X_filt[i, c, :] = signal.filtfilt(b_bw, a_bw, X_reshaped[i, c, :])
        X_norm = np.zeros_like(X_filt)
        for c in range(n_c):
            base_mean = np.mean(X_filt[:, c, :10])
            base_max = np.percentile(X_filt[:, c, :], 95) - base_mean + 1e-6
            X_norm[:, c, :] = (X_filt[:, c, :] - base_mean) / base_max
        return X_norm

    X_p5_rms_norm = normalizar_p5(X_p5_rms_list, 3)
    X_p5_tkeo_norm = normalizar_p5(X_p5_tkeo_list, 3)
    X_p5_hibrido = np.concatenate([X_p5_rms_norm, X_p5_tkeo_norm], axis=1) # (N, 6, 20)

    return {
        'rms': torch.tensor(X_p5_rms_norm, dtype=torch.float32),
        'tkeo': torch.tensor(X_p5_tkeo_norm, dtype=torch.float32),
        'hibrido': torch.tensor(X_p5_hibrido, dtype=torch.float32),
        'y': np.array(y_p5_list)
    }

# ==============================================================================
# 3. ENTRENAMIENTO Y EVALUACION CONTROLADA
# ==============================================================================
def entrenar_y_evaluar(nombre_modalidad, X_lucas_t, y_lucas, X_p5_t, y_p5, in_channels, device, epochs=350, seed=100):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    time_pts = 20
    model = FlexibleConvOrthogonalAE(
        in_channels=in_channels,
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
    N_lucas = X_lucas_t.shape[0]

    X_lucas_flat = X_lucas_t.view(N_lucas, -1).to(device)
    X_p5_flat = X_p5_t.view(X_p5_t.shape[0], -1).to(device)

    print(f"\n--- Entrenando {nombre_modalidad} ({in_channels} canales de entrada) ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        recon, z = model(X_lucas_t.to(device))
        loss_recon = nn.functional.mse_loss(recon, X_lucas_flat)
        loss_w = model.weight_orthogonality_loss()
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N_lucas - 1)
        loss_z = torch.sum((cov_z - I_2d) ** 2)
        loss = loss_recon + lambda_w * loss_w + lambda_z * loss_z
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0 or (epoch + 1) == epochs:
            print(f"  Época [{epoch+1:3d}/{epochs}] | MSE={loss_recon.item():.4f} | LossW={loss_w.item():.4f} | LossZ={loss_z.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z_lucas = model.encode(X_lucas_t.to(device)).cpu().numpy()
        Z_p5 = model.encode(X_p5_t.to(device)).cpu().numpy()

    # GMM Lucas
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

    # Transferencia P5
    pred_raw_p5 = gmm.predict(Z_p5)
    pred_p5_idx = np.array([cluster_to_vocal[c] for c in pred_raw_p5])
    y_p5_idx = np.array([vocal_to_idx[v] for v in y_p5])
    acc_p5 = accuracy_score(y_p5_idx, pred_p5_idx) * 100.0

    # Desglose por vocal en Lucas
    lucas_vocal_accs = {}
    for idx, v in enumerate(vocales):
        m = (y_lucas_idx == idx)
        lucas_vocal_accs[v] = np.mean(pred_lucas_idx[m] == idx) * 100.0 if np.sum(m) > 0 else 0.0

    # Desglose por vocal en P5
    p5_vocal_accs = {}
    for idx, v in enumerate(vocales):
        m = (y_p5_idx == idx)
        p5_vocal_accs[v] = np.mean(pred_p5_idx[m] == idx) * 100.0 if np.sum(m) > 0 else 0.0

    harmonic_acc = 2 * (acc_lucas * acc_p5) / (acc_lucas + acc_p5 + 1e-9)

    return {
        'nombre': nombre_modalidad,
        'acc_lucas': acc_lucas,
        'acc_p5': acc_p5,
        'harmonic_acc': harmonic_acc,
        'lucas_vocal_accs': lucas_vocal_accs,
        'p5_vocal_accs': p5_vocal_accs,
        'Z_lucas': Z_lucas,
        'Z_p5': Z_p5,
        'pred_p5_idx': pred_p5_idx,
        'y_p5_idx': y_p5_idx
    }

# ==============================================================================
# 4. MAIN DEL EXPERIMENTO
# ==============================================================================
def main():
    print("=" * 80)
    print("EXPERIMENTO CONTROLADO: ENVOLVENTE RMS vs TEAGER-KAISER vs HIBRIDO")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo detectado: {device}")
    
    # 1. Cargar datos
    datos_lucas = extraer_lucas_dual()
    datos_p5 = extraer_p5_dual(datos_lucas['b_bw'], datos_lucas['a_bw'])
    
    y_lucas = datos_lucas['y']
    y_p5 = datos_p5['y']
    print(f"\nDatos listos: Lucas = {len(y_lucas)} muestras | P5 = {len(y_p5)} pulsos.\n")

    # 2. Entrenar y evaluar los 3 modelos
    modelos = [
        ("1. Solo RMS (Control 3 Canales)", datos_lucas['rms'], datos_p5['rms'], 3),
        ("2. Solo TKEO (Kaiser 3 Canales)", datos_lucas['tkeo'], datos_p5['tkeo'], 3),
        ("3. Hibrido (RMS + TKEO: 6 Canales)", datos_lucas['hibrido'], datos_p5['hibrido'], 6)
    ]
    
    resultados = []
    for nombre, X_l, X_p, in_ch in modelos:
        res = entrenar_y_evaluar(nombre, X_l, y_lucas, X_p, y_p5, in_channels=in_ch, device=device)
        resultados.append(res)

    # 3. Mostrar tabla comparativa consolidada
    print("\n" + "=" * 90)
    print(f"{'Modalidad':<36} | {'Lucas Acc':<10} | {'P5 Acc':<10} | {'Armonica':<10}")
    print("-" * 90)
    for r in resultados:
        print(f"{r['nombre']:<36} | {r['acc_lucas']:>8.2f}% | {r['acc_p5']:>8.2f}% | {r['harmonic_acc']:>8.2f}%")
    print("=" * 90)

    print("\nDESGLOSE POR VOCAL EN LUCAS (ENTRENAMIENTO):")
    print(f"{'Modalidad':<36} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 80)
    for r in resultados:
        la = r['lucas_vocal_accs']
        print(f"{r['nombre']:<36} | {la['A']:>5.1f}% | {la['E']:>5.1f}% | {la['I']:>5.1f}% | {la['O']:>5.1f}% | {la['U']:>5.1f}%")
    print("-" * 80)

    print("\nDESGLOSE POR VOCAL EN CANDELA P5 (SECUENCIA CONTINUA):")
    print(f"{'Modalidad':<36} | {'/a/':<7} | {'/e/':<7} | {'/i/':<7} | {'/o/':<7} | {'/u/':<7}")
    print("-" * 80)
    for r in resultados:
        pa = r['p5_vocal_accs']
        print(f"{r['nombre']:<36} | {pa['A']:>5.1f}% | {pa['E']:>5.1f}% | {pa['I']:>5.1f}% | {pa['O']:>5.1f}% | {pa['U']:>5.1f}%")
    print("-" * 80)

    # 4. Generar grafico comparativo de los tres espacios latentes
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for row_idx, r in enumerate(resultados):
        # Lucas
        ax_l = axes[row_idx, 0]
        Z_l = r['Z_lucas']
        for v in vocales:
            mask = (y_lucas == v)
            ax_l.scatter(Z_l[mask, 0], Z_l[mask, 1], c=colores_oficiales[v], label=f"/{v.lower()}/", alpha=0.7, s=25)
        ax_l.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_l.set_title(f"{r['nombre']} - Lucas: {r['acc_lucas']:.1f}%", fontsize=10, fontweight='bold')
        ax_l.grid(True, linestyle=':', alpha=0.5)
        if row_idx == 0:
            ax_l.legend(loc='upper right', fontsize=8)
            
        # P5
        ax_p = axes[row_idx, 1]
        Z_p = r['Z_p5']
        aciertos = (r['y_p5_idx'] == r['pred_p5_idx'])
        for v in vocales:
            mask_ok = (y_p5 == v) & aciertos
            ax_p.scatter(Z_p[mask_ok, 0], Z_p[mask_ok, 1], c=colores_oficiales[v], marker='o', alpha=0.85, s=35)
            mask_err = (y_p5 == v) & (~aciertos)
            if np.any(mask_err):
                ax_p.scatter(Z_p[mask_err, 0], Z_p[mask_err, 1], c=colores_oficiales[v], marker='x', alpha=0.9, s=45, linewidths=1.5)
        ax_p.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_p.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax_p.set_title(f"{r['nombre']} - P5: {r['acc_p5']:.1f}% ({int(np.sum(aciertos))}/123)", fontsize=10, fontweight='bold')
        ax_p.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    p_png = os.path.join(out_dir, "comparativa_rms_vs_tkeo_vs_hibrido.png")
    plt.savefig(p_png, dpi=150)
    plt.close()
    print(f"\n[Visualización] Gráfico comparativo exportado en: {p_png}")
    print("=" * 80)

if __name__ == '__main__':
    main()
