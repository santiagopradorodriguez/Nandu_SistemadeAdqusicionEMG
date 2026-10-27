import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
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

b_n, a_n = iirnotch(50.0, 2.0, 6000)
b_b, a_b = butter(4, [20.0/3000, 450.0/3000], btype='bandpass')

pulsos_env = []
pulsos_mfcc = []
pulsos_labels = []

for carp in silabas_carpetas:
    silaba = carp.split('_')[0].upper()
    carp_path = os.path.join(base_dir, carp)
    meta_p = os.path.join(carp_path, "canal_0", "metadata.json")
    if not os.path.exists(meta_p): continue
    with open(meta_p) as f: meta = json.load(f)
    sr = meta.get('sample_rate', 6000)
    bpm = meta.get('bpm', 30)
    noise_sec = meta.get('noise_seconds', 3.0)
    muestras_pulso = int(sr * (60.0 / bpm))
    offset_n = int(noise_sec * sr)
    
    wavs = {}
    for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
        wp = os.path.join(carp_path, ch, "grabacion.wav")
        if not os.path.exists(wp): break
        _, d = wavfile.read(wp)
        wavs[ch] = d.astype(np.float32)
    if len(wavs) < 4: continue
    
    audio = wavs["canal_3"][offset_n:]
    win_mic = int(sr * 0.05)
    env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
    picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
    pre_s, post_s = int(0.4 * muestras_pulso), int(0.6 * muestras_pulso)
    
    for p in picos:
        st, en = offset_n + p - pre_s, offset_n + p + post_s
        if st < 0 or en > len(wavs["canal_0"]): continue
        
        # Filtrado AC de los 3 canales
        s_filt = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s_f = filtfilt(b_n, a_n, wavs[ch][st:en])
            s_f = filtfilt(b_b, a_b, s_f)
            s_filt.append(s_f)
            
        # Supremo tricanal
        supremo = max(np.max(np.abs(s)) for s in s_filt)
        if supremo < 1e-9: continue
        
        # 1. Envolventes temporales
        win_e = int(sr * 0.05)
        envs = [np.convolve(np.abs(s), np.ones(win_e)/win_e, mode='same') for s in s_filt]
        v_env = np.concatenate([resample(e / supremo, 60) for e in envs])
        
        # 2. MFCCs Mel AC
        mfccs = []
        for s in s_filt:
            s_norm = s / supremo
            mf = compute_cepstrum_emg_cruda(s_norm, sr=sr, n_filtros=16, n_ceps=8)
            mfccs.append(np.hstack([np.mean(mf, axis=0), np.std(mf, axis=0)]))
        v_mfcc = np.concatenate(mfccs)
        
        pulsos_env.append(v_env)
        pulsos_mfcc.append(v_mfcc)
        pulsos_labels.append(silaba)

X_env = np.array(pulsos_env)
X_mfcc = np.array(pulsos_mfcc)
y = np.array(pulsos_labels)
clases = sorted(list(np.unique(y)))

gamma = np.sqrt(np.var(X_env) / np.var(X_mfcc))
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

acc_env = np.mean(pred_env == y) * 100
acc_mfcc = np.mean(pred_mfcc == y) * 100
acc_hib = np.mean(pred_hib == y) * 100

print(f"Total pulsos: {len(y)}")
print(f"Exactitud PCA 5D (Solo Envolvente):    {acc_env:.1f}%")
print(f"Exactitud PCA 5D (Solo MFCC Mel AC):   {acc_mfcc:.1f}%")
print(f"Exactitud PCA 5D (Hibrido Equilibrado): {acc_hib:.1f}%")

# Validacion cruzada CV-3
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
def calc_cv(X_rep):
    sc = []
    for tr, ts in cv.split(X_rep, y):
        sc.append(LinearDiscriminantAnalysis().fit(X_rep[tr], y[tr]).score(X_rep[ts], y[ts]))
    return np.mean(sc) * 100, np.std(sc) * 100

cv_env_mean, cv_env_std = calc_cv(pca_env)
cv_mfcc_mean, cv_mfcc_std = calc_cv(pca_mfcc)
cv_hib_mean, cv_hib_std = calc_cv(pca_hib)

print(f"CV-3 PCA 5D Envolvente: {cv_env_mean:.1f}% (+/- {cv_env_std:.1f}%)")
print(f"CV-3 PCA 5D MFCC:       {cv_mfcc_mean:.1f}% (+/- {cv_mfcc_std:.1f}%)")
print(f"CV-3 PCA 5D Hibrido:    {cv_hib_mean:.1f}% (+/- {cv_hib_std:.1f}%)")

# Matrices de confusion en porcentajes
cm_env = confusion_matrix(y, pred_env, labels=clases, normalize='true') * 100
cm_mfcc = confusion_matrix(y, pred_mfcc, labels=clases, normalize='true') * 100
cm_hib = confusion_matrix(y, pred_hib, labels=clases, normalize='true') * 100

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

sns.heatmap(cm_env, annot=True, fmt='.0f', cmap='Blues', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[0])
axes[0].set_title(f'PCA 5D: Solo Envolvente\nAjuste: {acc_env:.1f}% | CV: {cv_env_mean:.1f}%', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Prediccion')
axes[0].set_ylabel('Clase Real')

sns.heatmap(cm_mfcc, annot=True, fmt='.0f', cmap='Oranges', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[1])
axes[1].set_title(f'PCA 5D: Solo MFCC Mel AC\nAjuste: {acc_mfcc:.1f}% | CV: {cv_mfcc_mean:.1f}%', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Prediccion')
axes[1].set_ylabel('Clase Real')

sns.heatmap(cm_hib, annot=True, fmt='.0f', cmap='Greens', xticklabels=clases, yticklabels=clases, cbar=False, ax=axes[2])
axes[2].set_title(f'PCA 5D: Hibrido Equilibrado\nAjuste: {acc_hib:.1f}% | CV: {cv_hib_mean:.1f}%', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Prediccion')
axes[2].set_ylabel('Clase Real')

plt.suptitle('Comparativa de Discriminacion en PCA 5D: 9 Silabas (Porcentajes)', fontsize=13, fontweight='bold')
plt.tight_layout()
out_fig = 'reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_5d_mfcc_vs_envolvente.png'
plt.savefig(out_fig, dpi=200)
print('Figura guardada en:', out_fig)
