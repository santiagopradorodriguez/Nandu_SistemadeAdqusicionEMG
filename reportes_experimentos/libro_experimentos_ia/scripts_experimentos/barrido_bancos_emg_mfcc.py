import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Cargar las tomas y picos de sincronizacion exactos de la sesion
base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
mediciones = [
    f"{v}_Pruebatricanal_{suf}"
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}"))
]

# Cachear grabaciones crudas por pulso
pulsos_crudos = []
for med in mediciones:
    vocal = med.split('_')[0].upper()
    med_p = os.path.join(base_dir, med)
    meta_p = os.path.join(med_p, "canal_0", "metadata.json")
    with open(meta_p) as f: meta = json.load(f)
    sr = meta.get('sample_rate', 6000)
    bpm = meta.get('bpm', 30)
    noise_sec = meta.get('noise_seconds', 3.0)
    muestras_pulso = int(sr * (60.0 / bpm))
    offset_n = int(noise_sec * sr)
    
    wavs = {}
    for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
        wp = os.path.join(med_p, ch, "grabacion.wav")
        _, d = wavfile.read(wp)
        wavs[ch] = d.astype(np.float32)
        
    audio = wavs["canal_3"][offset_n:]
    win_mic = int(sr * 0.05)
    env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
    picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
    
    pre_s = int(0.4 * muestras_pulso)
    post_s = int(0.6 * muestras_pulso)
    
    for idx_p, p in enumerate(picos):
        st = offset_n + p - pre_s
        en = offset_n + p + post_s
        if st < 0 or en > len(wavs["canal_0"]): continue
        pulsos_crudos.append({
            'vocal': vocal,
            'ch0': wavs["canal_0"][st:en],
            'ch1': wavs["canal_1"][st:en],
            'ch2': wavs["canal_2"][st:en],
            'sr': sr
        })

print(f"Total pulsos crudos listos para barrido espectral: {len(pulsos_crudos)}")

def compute_cepstrum(signal, sr, escala='linear', n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
    num_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal[:num_frames * hop_length + n_fft - hop_length], n_fft)[::hop_length]
    frames = frames * np.hamming(n_fft)
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    
    if escala == 'linear':
        hz_points = np.linspace(fmin, fmax, n_filtros + 2)
    elif escala == 'log':
        hz_points = np.geomspace(fmin, fmax, n_filtros + 2)
    elif escala == 'mel_emg': # Escala Mel adaptada a musculo con pivote en 150 Hz
        def hz_to_mel_emg(hz): return 1000.0 * np.log10(1.0 + hz / 150.0)
        def mel_emg_to_hz(mel): return 150.0 * (10.0**(mel / 1000.0) - 1.0)
        m_pts = np.linspace(hz_to_mel_emg(fmin), hz_to_mel_emg(fmax), n_filtros + 2)
        hz_points = mel_emg_to_hz(m_pts)
    elif escala == 'mel_standard':
        def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
        def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        m_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_filtros + 2)
        hz_points = mel_to_hz(m_pts)

    bin_points = np.clip(np.floor((n_fft + 1) * hz_points / sr).astype(int), 0, int(n_fft // 2))
    
    fbank = np.zeros((n_filtros, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filtros + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m): fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus): fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (f_m_plus - f_m))
        
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20.0 * np.log10(filter_banks)
    return dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_ceps]

vocales = ['A', 'E', 'I', 'O', 'U']
Y = np.array([p['vocal'] for p in pulsos_crudos])
mask_ou = np.isin(Y, ['O', 'U'])

def eval_config(escala, fmin, fmax, n_filtros=16, n_ceps=8):
    X = []
    for p in pulsos_crudos:
        sr = p['sr']
        feats = []
        for ch_key in ['ch0', 'ch1', 'ch2']:
            s = p[ch_key]
            s_c = s - np.mean(s)
            s_n = s_c / (np.std(s_c) + 1e-9)
            ceps = compute_cepstrum(s_n, sr, escala=escala, n_filtros=n_filtros, n_ceps=n_ceps, fmin=fmin, fmax=fmax)
            feats.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
        X.append(np.concatenate(feats))
    X = np.array(X)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    var = np.sum(pca.explained_variance_ratio_) * 100
    
    # K-Means 5 vocales
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_pca)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    acc_km = np.mean(preds == Y) * 100
    
    # LDA binario O vs U
    lda_ou = LinearDiscriminantAnalysis().fit(X_pca[mask_ou], Y[mask_ou])
    acc_ou = lda_ou.score(X_pca[mask_ou], Y[mask_ou]) * 100
    
    # LDA multiclase 5v
    lda_5v = LinearDiscriminantAnalysis().fit(X_pca, Y).score(X_pca, Y) * 100
    
    return acc_km, lda_5v, acc_ou, var, X_pca

bancos = [
    ("1. Mel Estándar (20 - 500 Hz)", 'mel_standard', 20, 500),
    ("2. Mel Muscular Adaptada (f_pivote=150Hz, 20 - 450 Hz)", 'mel_emg', 20, 450),
    ("3. Banco Lineal Uniforme (20 - 400 Hz)", 'linear', 20, 400),
    ("4. Banco Logarítmico Puro (20 - 450 Hz)", 'log', 20, 450),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

print("Comparativa de Escalas y Bancos Frecuenciales para sEMG en PCA 2D:")
print("-" * 80)
print(f"{'Escala Frecuencial':<45} | {'K-Means':<10} | {'LDA (5v)':<10} | {'O vs U':<8}")
print("-" * 80)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

for idx, (nombre, escala, fmin, fmax) in enumerate(bancos):
    acc_km, acc_5v, acc_ou, var, X_pca = eval_config(escala, fmin, fmax)
    print(f"{nombre:<45} | {acc_km:6.1f}%    | {acc_5v:6.1f}%    | {acc_ou:6.1f}%")
    
    ax = axes[idx]
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{nombre}\nK-Means: {acc_km:.1f}% | LDA 5v: {acc_5v:.1f}% | O vs U: {acc_ou:.1f}% (Var: {var:.1f}%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_bancos_emg_mfcc.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("-" * 80)
print("Figura comparativa guardada en:", out_f)
