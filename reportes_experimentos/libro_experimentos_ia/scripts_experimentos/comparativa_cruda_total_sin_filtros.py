import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, resample
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
vocales = ['A', 'E', 'I', 'O', 'U']

def compute_mfcc(signal, sr, n_mfcc=8, n_mels=16, n_fft=512, hop_length=256, fmin=20, fmax=500):
    if len(signal) < n_fft:
        pad_width = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')
    num_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal[:num_frames * hop_length + n_fft - hop_length], n_fft)[::hop_length]
    frames = frames * np.hamming(n_fft)
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0**(mel / 2595.0) - 1.0)
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    bin_points = np.clip(np.floor((n_fft + 1) * mel_to_hz(mel_points) / sr).astype(int), 0, int(n_fft // 2))
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m): fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
        for k in range(f_m, f_m_plus): fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20.0 * np.log10(filter_banks)
    return dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

muestras = []
# Procesamiento 100% puro sin filtro de outliers ni SNR ni gate
for v in vocales:
    for suf in ["Sujeto1", "2_Sujeto1", "3_Sujeto1", "4_Sujeto1"]:
        carp = os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}")
        if not os.path.isdir(carp): continue
        meta_p = os.path.join(carp, "canal_0", "metadata.json")
        with open(meta_p) as f: meta = json.load(f)
        sr = meta.get('sample_rate', 6000)
        bpm = meta.get('bpm', 30)
        noise_sec = meta.get('noise_seconds', 3.0)
        muestras_pulso = int(sr * (60.0 / bpm))
        offset_n = int(noise_sec * sr)
        
        wavs = {}
        for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
            _, d = wavfile.read(os.path.join(carp, ch, "grabacion.wav"))
            wavs[ch] = d.astype(np.float32)
            
        # Picos de microfono exclusivamente para sincronizar
        audio_rec = wavs["canal_3"][offset_n:]
        win_size = int(sr * 0.05)
        env_mic = np.convolve(np.abs(audio_rec), np.ones(win_size)/win_size, mode='same')
        picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
        
        pre_s = int(0.4 * muestras_pulso)
        post_s = int(0.6 * muestras_pulso)
        
        for idx_p, p in enumerate(picos):
            st = offset_n + p - pre_s
            en = offset_n + p + post_s
            if st < 0 or en > len(wavs["canal_0"]): continue
            
            # 1. Envolventes sin tocar
            win_emg = int(sr * 0.05) # 50ms como la corrida original
            e0 = np.convolve(np.abs(wavs["canal_0"][st:en]), np.ones(win_emg)/win_emg, mode='same')
            e1 = np.convolve(np.abs(wavs["canal_1"][st:en]), np.ones(win_emg)/win_emg, mode='same')
            e2 = np.convolve(np.abs(wavs["canal_2"][st:en]), np.ones(win_emg)/win_emg, mode='same')
            
            # Normalizacion por supremo tricanal
            supremo = max(np.max(e0), np.max(e1), np.max(e2), 1e-9)
            e0_n = resample(e0 / supremo, 60)
            e1_n = resample(e1 / supremo, 60)
            e2_n = resample(e2 / supremo, 60)
            feat_env = np.concatenate([e0_n * 2.5, e1_n, e2_n]) # con el peso 2.5 oficial
            
            # 2. MFCC sEMG
            mfccs_ch = []
            for ch in ["canal_0", "canal_1", "canal_2"]:
                s = wavs[ch][st:en]
                s_c = s - np.mean(s)
                s_n = s_c / (np.std(s_c) + 1e-9)
                m = compute_mfcc(s_n, sr)
                mfccs_ch.extend([np.mean(m, axis=0), np.std(m, axis=0)])
            feat_mfcc = np.concatenate(mfccs_ch)
            
            muestras.append({
                'vocal': v,
                'toma': f"{v}_{suf}_Win{idx_p}",
                'feat_env': feat_env,
                'feat_mfcc': feat_mfcc
            })

print(f"Total absoluto de pulsos (CERO FILTROS, CERO DESCARTES): {len(muestras)}")

Y = np.array([m['vocal'] for m in muestras])
X_env = np.array([m['feat_env'] for m in muestras])
X_mfcc = np.array([m['feat_mfcc'] for m in muestras])

pca_env = PCA(n_components=2, random_state=42).fit_transform(X_env)
pca_mfcc = PCA(n_components=2, random_state=42).fit_transform(X_mfcc)

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y) * 100

mask_ou = np.isin(Y, ['O', 'U'])
lda_env_ou = LinearDiscriminantAnalysis().fit(pca_env[mask_ou], Y[mask_ou])
lda_mfcc_ou = LinearDiscriminantAnalysis().fit(pca_mfcc[mask_ou], Y[mask_ou])

acc_km_env = eval_km(pca_env)
acc_km_mfcc = eval_km(pca_mfcc)
acc_ou_env = lda_env_ou.score(pca_env[mask_ou], Y[mask_ou]) * 100
acc_ou_mfcc = lda_mfcc_ou.score(pca_mfcc[mask_ou], Y[mask_ou]) * 100

print(f"Envolvente (180 pulsos, sin descartar nada) -> K-Means: {acc_km_env:.1f}% | Sep. O vs U: {acc_ou_env:.1f}%")
print(f"MFCC sEMG  (180 pulsos, sin descartar nada) -> K-Means: {acc_km_mfcc:.1f}% | Sep. O vs U: {acc_ou_mfcc:.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, X_p, tit, acc_km, acc_ou in [(axes[0], pca_env, "1. Envolvente sEMG (Sin filtro de outliers ni descartes)", acc_km_env, acc_ou_env),
                                     (axes[1], pca_mfcc, "2. MFCC sEMG puro (Sin filtro de outliers ni descartes)", acc_km_mfcc, acc_ou_mfcc)]:
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={np.sum(m)})", s=45, alpha=0.8, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means (5v): {acc_km:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou:.1f}%", fontweight='bold')
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_100pct_cruda_sin_outliers.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
