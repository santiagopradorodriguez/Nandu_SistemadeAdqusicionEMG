import os
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
silabas_carpetas = [d for d in sorted(os.listdir(base_dir)) if "SILABAS" in d and os.path.isdir(os.path.join(base_dir, d))]

def compute_cepstrum_emg_cruda(signal, sr=6000, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
    if len(signal) < n_fft:
        signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
    num_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal[:num_frames * hop_length + n_fft - hop_length], n_fft)[::hop_length]
    frames = frames * np.hamming(n_fft)
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    def hz_to_mel_emg(hz): return 1000.0 * np.log10(1.0 + hz / 150.0)
    def mel_emg_to_hz(mel): return 150.0 * (10.0**(mel / 1000.0) - 1.0)
    m_pts = np.linspace(hz_to_mel_emg(fmin), hz_to_mel_emg(fmax), n_filtros + 2)
    bin_points = np.clip(np.floor((n_fft + 1) * mel_emg_to_hz(m_pts) / sr).astype(int), 0, int(n_fft // 2))
    fbank = np.zeros((n_filtros, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_filtros + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m): fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (f_m - f_m_minus))
        for k in range(f_m, f_m_plus): fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (f_m_plus - f_m))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return dct(20.0 * np.log10(filter_banks), type=2, axis=1, norm='ortho')[:, :n_ceps]

pulsos_mfcc = []
pulsos_env = []
labels = []

b_n, a_n = iirnotch(50.0, 2.0, 6000)
b_b, a_b = butter(4, [20.0/3000, 450.0/3000], btype='bandpass')

for carp in silabas_carpetas:
    silaba = carp.split('_')[0].upper()
    carp_path = os.path.join(base_dir, carp)
    meta_p = os.path.join(carp_path, "canal_0", "metadata.json")
    if not os.path.exists(meta_p):
        continue
    with open(meta_p) as f:
        meta = json.load(f)
    sr = meta.get('sample_rate', 6000)
    
    sigs = []
    for c in range(3):
        w_path = os.path.join(carp_path, f"canal_{c}", "grabacion.wav")
        _, sig = wavfile.read(w_path)
        sig = sig.astype(np.float64)
        sig_f = filtfilt(b_n, a_n, sig)
        sig_f = filtfilt(b_b, a_b, sig_f)
        sigs.append(sig_f)
    
    w_mic = os.path.join(carp_path, "canal_3", "grabacion.wav")
    _, sig_mic = wavfile.read(w_mic)
    sig_mic = sig_mic.astype(np.float64)
    env_mic = np.abs(sig_mic)
    win_env = int(sr * 0.05)
    env_mic = np.convolve(env_mic, np.ones(win_env)/win_env, mode='same')
    
    peaks, _ = find_peaks(env_mic, distance=int(sr * 0.8), prominence=np.percentile(env_mic, 85))
    
    b_env, a_env = butter(2, 6.0 / (sr / 2.0), btype='low')
    
    pre = int(sr * 0.3)
    post = int(sr * 0.4)
    
    for p in peaks:
        ini = p - pre
        fin = p + post
        if ini < 0 or fin > len(sigs[0]):
            continue
        
        supremo = max(np.max(np.abs(sigs[c][ini:fin])) for c in range(3))
        if supremo < 1e-6:
            continue
        
        mfccs_c = []
        envs_c = []
        for c in range(3):
            seg_ac = sigs[c][ini:fin] / supremo
            mfcc_feat = compute_cepstrum_emg_cruda(seg_ac, sr=sr, n_filtros=16, n_ceps=8)
            mfcc_vec = np.hstack([np.mean(mfcc_feat, axis=0), np.std(mfcc_feat, axis=0)])
            mfccs_c.append(mfcc_vec)
            
            rect = np.abs(seg_ac)
            env = filtfilt(b_env, a_env, rect)
            idx_rs = np.linspace(0, len(env)-1, 60).astype(int)
            envs_c.append(env[idx_rs])
            
        pulsos_mfcc.append(np.concatenate(mfccs_c))
        pulsos_env.append(np.concatenate(envs_c))
        labels.append(silaba)

X_env = np.array(pulsos_env)
X_mfcc = np.array(pulsos_mfcc)
y = np.array(labels)
clases = sorted(list(set(y)))

var_env = np.var(X_env)
var_mfcc = np.var(X_mfcc)
gamma = np.sqrt(var_env / var_mfcc)
X_hib = np.hstack([X_env, gamma * X_mfcc])

# PCA 5D
pca_env = PCA(n_components=5, random_state=42).fit_transform(X_env)
pca_mfcc = PCA(n_components=5, random_state=42).fit_transform(X_mfcc)
pca_hib = PCA(n_components=5, random_state=42).fit_transform(X_hib)

lda_env = LinearDiscriminantAnalysis().fit(pca_env, y)
lda_mfcc = LinearDiscriminantAnalysis().fit(pca_mfcc, y)
lda_hib = LinearDiscriminantAnalysis().fit(pca_hib, y)

pred_env = lda_env.predict(pca_env)
pred_mfcc = lda_mfcc.predict(pca_mfcc)
pred_hib = lda_hib.predict(pca_hib)

cm_env = confusion_matrix(y, pred_env, labels=clases, normalize='true') * 100
cm_mfcc = confusion_matrix(y, pred_mfcc, labels=clases, normalize='true') * 100
cm_hib = confusion_matrix(y, pred_hib, labels=clases, normalize='true') * 100

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

sns.heatmap(cm_env, annot=True, fmt='.0f', cmap='Blues', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[0])
axes[0].set_title(f'PCA 5D: Solo Envolvente\nExactitud: {lda_env.score(pca_env, y)*100:.1f}%', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Prediccion')
axes[0].set_ylabel('Clase Real')

sns.heatmap(cm_mfcc, annot=True, fmt='.0f', cmap='Oranges', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[1])
axes[1].set_title(f'PCA 5D: Solo MFCC Mel AC\nExactitud: {lda_mfcc.score(pca_mfcc, y)*100:.1f}%', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Prediccion')
axes[1].set_ylabel('Clase Real')

sns.heatmap(cm_hib, annot=True, fmt='.0f', cmap='Greens', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[2])
axes[2].set_title(f'PCA 5D: Hibrido Envolvente + MFCC\nExactitud: {lda_hib.score(pca_hib, y)*100:.1f}%', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Prediccion')
axes[2].set_ylabel('Clase Real')

plt.suptitle('Comparativa en 5D: Envolvente vs MFCC vs Hibrido (Porcentajes)', fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = 'reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_5d_mfcc_vs_envolvente.png'
plt.savefig(out_path, dpi=200)

print(f"PCA 5D Envolvente: {lda_env.score(pca_env, y)*100:.1f}%")
print(f"PCA 5D MFCC: {lda_mfcc.score(pca_mfcc, y)*100:.1f}%")
print(f"PCA 5D Hibrido: {lda_hib.score(pca_hib, y)*100:.1f}%")
