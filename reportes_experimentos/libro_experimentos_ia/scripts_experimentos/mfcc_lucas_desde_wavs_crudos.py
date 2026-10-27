import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-07-10"
vocales = ['A', 'E', 'I', 'O', 'U']

def compute_cepstrum_emg_cruda(signal, sr=2000, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=256, hop_length=128):
    if len(signal) < n_fft: signal = np.pad(signal, (0, n_fft - len(signal)), mode='constant')
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

# Cargamos los datos de Lucas desde los WAVs crudos oscilatorios
b_n, a_n = iirnotch(50.0, 2.0, 2000)
b_b, a_b = butter(4, [20.0/1000, 450.0/1000], btype='bandpass')

pulsos = []
for v in vocales:
    for t_idx in range(1, 8):
        carp = os.path.join(base_dir, f"{v}_T{t_idx}_Lucas")
        if not os.path.isdir(carp): continue
        meta_p = os.path.join(carp, "canal_0", "metadata.json")
        with open(meta_p) as f: meta = json.load(f)
        sr = meta.get('sample_rate', 2000)
        bpm = meta.get('bpm', 50)
        noise_sec = meta.get('noise_seconds', 2.0)
        muestras_pulso = int(sr * (60.0 / bpm))
        offset_n = int(noise_sec * sr)
        
        wavs = {}
        for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
            _, d = wavfile.read(os.path.join(carp, ch, "grabacion.wav"))
            wavs[ch] = d.astype(np.float32)
            
        audio = wavs["canal_3"][offset_n:]
        win_mic = int(sr * 0.05)
        env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
        picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
        pre_s, post_s = int(0.4 * muestras_pulso), int(0.6 * muestras_pulso)
        
        for p in picos:
            st, en = offset_n + p - pre_s, offset_n + p + post_s
            if st < 0 or en > len(wavs["canal_0"]): continue
            
            # Senal cruda filtrada (AC oscilatoria)
            cruda_ch = {}
            for ch in ["canal_0", "canal_1", "canal_2"]:
                s = filtfilt(b_n, a_n, wavs[ch][st:en])
                s = filtfilt(b_b, a_b, s)
                cruda_ch[ch] = s
                
            # MFCC sobre la señal cruda AC
            m_feats = []
            for ch in ["canal_0", "canal_1", "canal_2"]:
                s = cruda_ch[ch]
                s_n = (s - np.mean(s)) / (np.std(s) + 1e-9)
                ceps = compute_cepstrum_emg_cruda(s_n, sr=2000)
                m_feats.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
                
            pulsos.append({
                'vocal': v,
                'feat_mfcc_cruda': np.concatenate(m_feats)
            })

print(f"Pulsos crudos extraidos: {len(pulsos)}")
X_mfcc = np.array([p['feat_mfcc_cruda'] for p in pulsos])
Y = np.array([p['vocal'] for p in pulsos])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_mfcc)
var = pca.explained_variance_ratio_ * 100

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y) * 100

mask_ou = np.isin(Y, ['O', 'U'])
lda_ou = LinearDiscriminantAnalysis().fit(X_pca[mask_ou], Y[mask_ou]).score(X_pca[mask_ou], Y[mask_ou]) * 100
acc_km = eval_km(X_pca)

print(f"PCA 2D MFCC CRUDA OSCILATORIA -> K-Means: {acc_km:.1f}% | O vs U: {lda_ou:.1f}% | Var: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
plt.figure(figsize=(9, 8))
for v in vocales:
    m = (Y == v)
    plt.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=40, alpha=0.85, edgecolors='black', linewidths=0.5)

plt.title(f"PCA 2D MFCC sobre Señal Cruda Oscilatoria AC (Lucas 2026-07-10)\nK-Means: {acc_km:.1f}% | Sep. O vs U: {lda_ou:.1f}% (Var: {np.sum(var):.1f}%)", fontweight='bold')
plt.xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold')
plt.ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lucas_mfcc_crudo_corregido.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
