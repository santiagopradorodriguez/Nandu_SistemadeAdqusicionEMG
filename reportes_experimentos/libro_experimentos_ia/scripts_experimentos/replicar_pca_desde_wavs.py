import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
mediciones = [
    f"{v}_Pruebatricanal_{suf}"
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}"))
]

with open("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/parametros_optimos_pca_2d.json") as f:
    params = json.load(f)

pesos = params.get('pesos_canales', [2.5, 1.0, 1.0])
target_len = params.get('target_length', 60)
alpha_ruido = params.get('alpha_ruido', 0.1)

def procesar_canal_emg(raw, sr, smooth_ms=50, lowpass=500.0, highpass=20.0, notch_q=2.0):
    # 1. Filtro notch 50Hz
    b_notch, a_notch = iirnotch(50.0, notch_q, sr)
    sig = filtfilt(b_notch, a_notch, raw)
    # 2. Bandpass 20 - 500 Hz
    b_bp, a_bp = butter(4, [highpass / (sr/2), min(lowpass / (sr/2), 0.99)], btype='bandpass')
    sig = filtfilt(b_bp, a_bp, sig)
    # 3. Rectificacion
    sig_rect = np.abs(sig)
    # 4. Envolvente suave (smooth_ms = 50ms)
    win_len = max(3, int(sr * (smooth_ms / 1000.0)))
    kernel = np.ones(win_len) / win_len
    env = np.convolve(sig_rect, kernel, mode='same')
    return env

X_list = []
Y_list = []
Tomas_list = []

for med in mediciones:
    vocal = med.split('_')[0].upper()
    med_p = os.path.join(base_dir, med)
    
    meta_p = os.path.join(med_p, "canal_0", "metadata.json")
    with open(meta_p) as f: meta = json.load(f)
    sr = meta.get('sample_rate', 6000)
    bpm = meta.get('bpm', 30)
    noise_sec = meta.get('noise_seconds', 3.0)
    offset_noise = int(noise_sec * sr)
    muestras_pulso = int(sr * (60.0 / bpm))
    
    # Cargar los 4 canales
    wavs = {}
    for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
        wp = os.path.join(med_p, ch, "grabacion.wav")
        _, d = wavfile.read(wp)
        wavs[ch] = d.astype(np.float32)
        
    # Microfono: cortar ruido inicial y calcular envolvente de audio
    audio = wavs["canal_3"][offset_noise:]
    win_mic = int(sr * 0.05)
    env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
    
    # Picos en microfono
    dist_samples = int(0.75 * muestras_pulso)
    min_height = np.max(env_mic) * 0.20
    picos_mic, _ = find_peaks(env_mic, distance=dist_samples, height=min_height)
    
    # Envolventes de los 3 canales sEMG recortadas tras el ruido inicial
    envs_emg = {}
    noises_emg = {}
    for ch in ["canal_0", "canal_1", "canal_2"]:
        env_full = procesar_canal_emg(wavs[ch], sr, smooth_ms=params['smooth_ms'], lowpass=params['lowpass_cutoff_hz'], highpass=params['highpass_cutoff_hz'])
        # nivel de ruido inicial
        noises_emg[ch] = np.mean(env_full[:offset_noise])
        envs_emg[ch] = env_full[offset_noise:]
        
    pre_samp = int(muestras_pulso * 0.4)
    post_samp = int(muestras_pulso * 0.6)
    
    for win_idx, pico in enumerate(picos_mic):
        start = pico - pre_samp
        end = pico + post_samp
        
        if start < 0 or end > len(env_mic): continue
        
        segs = []
        max_supremo = 1e-9
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s = envs_emg[ch][start:end].copy()
            s = np.maximum(s - noises_emg[ch] * alpha_ruido, 0.0)
            m = np.max(s)
            if m > max_supremo: max_supremo = m
            segs.append(s)
            
        if max_supremo <= 1e-6: continue
        
        vec = []
        for c_idx, s in enumerate(segs):
            s_norm = resample(s / max_supremo, target_len)
            s_norm[s_norm < 0] = 0.0
            s_norm *= pesos[c_idx] # peso oficial [2.5, 1.0, 1.0]
            vec.append(s_norm)
            
        X_list.append(np.concatenate(vec))
        Y_list.append(vocal)
        Tomas_list.append(f"{med}_Win{win_idx}")

# Isolation Forest por vocal (contamination=0.10)
X_clean = []
Y_clean = []
Tomas_clean = []

vocales_unicas = ['A', 'E', 'I', 'O', 'U']
for v in vocales_unicas:
    idxs = [i for i, y in enumerate(Y_list) if y == v]
    X_v = np.array([X_list[i] for i in idxs])
    iso = IsolationForest(contamination=0.10, random_state=42)
    preds = iso.fit_predict(X_v)
    for k, is_in in enumerate(preds):
        if is_in == 1:
            X_clean.append(X_v[k])
            Y_clean.append(v)
            Tomas_clean.append(Tomas_list[idxs[k]])

X_clean = np.array(X_clean)
Y_clean = np.array(Y_clean)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clean)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, ax = plt.subplots(figsize=(9, 8))
for v in vocales_unicas:
    m = (Y_clean == v)
    ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={np.sum(m)})", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

var = pca.explained_variance_ratio_ * 100.0
ax.set_title(f"PCA 2D Oficial Replicado desde WAVs Crudos (N={len(X_pca)})\nVar: PC1={var[0]:.1f}%, PC2={var[1]:.1f}% (Total: {np.sum(var):.1f}%)", fontweight='bold')
ax.set_xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold')
ax.set_ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_replicado_desde_wavs.png"
plt.savefig(out_f, dpi=300)
plt.close()

print(f"Pulsos replicados desde WAVs: {len(X_pca)}")
print(f"Grafico guardado en: {out_f}")
