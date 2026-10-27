import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
silabas_carpetas = [d for d in sorted(os.listdir(base_dir)) if "SILABAS" in d and os.path.isdir(os.path.join(base_dir, d))]

def compute_cepstrum_emg_cruda(signal, sr=6000, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
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

pulsos_mfcc = []
pulsos_env = []
labels = []

b_n, a_n = iirnotch(50.0, 2.0, 6000)
b_b, a_b = butter(4, [20.0/3000, 450.0/3000], btype='bandpass')

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
        
        # 1. Señal cruda oscilatoria filtrada (AC)
        cruda = {}
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s_f = filtfilt(b_n, a_n, wavs[ch][st:en])
            s_f = filtfilt(b_b, a_b, s_f)
            cruda[ch] = s_f
            
        # MFCCs sobre la señal cruda AC (16 por canal = 48)
        m_feats = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s = cruda[ch]
            s_n = (s - np.mean(s)) / (np.std(s) + 1e-9)
            ceps = compute_cepstrum_emg_cruda(s_n, sr=6000)
            m_feats.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
        v_mfcc = np.concatenate(m_feats)
        
        # 2. Envolvente
        from scipy.signal import resample
        win_e = int(sr * 0.05)
        envs = [np.convolve(np.abs(cruda[ch]), np.ones(win_e)/win_e, mode='same') for ch in ["canal_0", "canal_1", "canal_2"]]
        max_sup = max(np.max(envs[0]), np.max(envs[1]), np.max(envs[2]), 1e-9)
        v_env = np.concatenate([resample(e / max_sup, 60) for e in envs])
        
        pulsos_mfcc.append(v_mfcc)
        pulsos_env.append(v_env)
        labels.append(silaba)

X_mfcc = np.array(pulsos_mfcc)
X_env = np.array(pulsos_env)
Y = np.array(labels)
silabas = sorted(list(np.unique(Y)))

print(f"Total pulsos de sílabas analizados: {len(Y)} (9 clases)")

# Factor equilibrado
gamma = np.sqrt(np.mean(np.var(X_env, axis=0)) / (np.mean(np.var(X_mfcc, axis=0)) + 1e-9))
X_hib = np.hstack([X_env, X_mfcc * gamma])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def evaluar_espacio(X_mat, nombre):
    lda = LinearDiscriminantAnalysis().fit(X_mat, Y)
    acc_fit = lda.score(X_mat, Y) * 100
    sc = []
    for tr, ts in cv.split(X_mat, Y):
        clf = LinearDiscriminantAnalysis().fit(X_mat[tr], Y[tr])
        sc.append(clf.score(X_mat[ts], Y[ts]))
    return acc_fit, np.mean(sc)*100, np.std(sc)*100

print("\nComparativa de Exactitud Multiclase (9 Sílabas):")
print("-" * 85)
print(f"{'Representación':<40} | {'Dim':<5} | {'LDA (Ajuste)':<14} | {'LDA (CV-3)':<12}")
print("-" * 85)

for nombre, mat in [
    ("1. Solo Envolvente Temporal (Espacio 180D)", X_env),
    ("2. Solo MFCC Mel Muscular AC (Espacio 48D)", X_mfcc),
    ("3. PCA 5D - Solo Envolvente", PCA(n_components=5, random_state=42).fit_transform(X_env)),
    ("4. PCA 5D - Solo MFCC Mel AC", PCA(n_components=5, random_state=42).fit_transform(X_mfcc)),
    ("5. PCA 5D - Híbrido Equilibrado", PCA(n_components=5, random_state=42).fit_transform(X_hib)),
    ("6. PCA 8D - Híbrido Equilibrado", PCA(n_components=8, random_state=42).fit_transform(X_hib)),
    ("7. Espacio Directo Híbrido (228D)", X_hib)
]:
    f, c, s = evaluar_espacio(mat, nombre)
    print(f"{nombre:<40} | {mat.shape[1]:<5} | {f:5.1f}%         | {c:5.1f}% (+/-{s:4.1f}%)")

# Matriz de confusion del PCA 5D Hibrido y PCA 5D MFCC
pca5_mfcc = PCA(n_components=5, random_state=42).fit_transform(X_mfcc)
lda5_mfcc = LinearDiscriminantAnalysis().fit(pca5_mfcc, Y)
cm_mfcc = confusion_matrix(Y, lda5_mfcc.predict(pca5_mfcc), labels=silabas)
cm_mfcc_pct = cm_mfcc.astype('float') / cm_mfcc.sum(axis=1)[:, np.newaxis] * 100.0

pca5_hib = PCA(n_components=5, random_state=42).fit_transform(X_hib)
lda5_hib = LinearDiscriminantAnalysis().fit(pca5_hib, Y)
cm_hib = confusion_matrix(Y, lda5_hib.predict(pca5_hib), labels=silabas)
cm_hib_pct = cm_hib.astype('float') / cm_hib.sum(axis=1)[:, np.newaxis] * 100.0

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
paneles = [
    (axes[0], cm_mfcc_pct, "PCA 5D - Solo MFCC Mel AC (58.9%)"),
    (axes[1], cm_hib_pct, f"PCA 5D - Híbrido Tiempo+Mel ({lda5_hib.score(pca5_hib, Y)*100:.1f}%)")
]

for ax, cm_pct, tit in paneles:
    im = ax.imshow(cm_pct, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
    ax.set_title(tit, fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(silabas)))
    ax.set_yticks(range(len(silabas)))
    ax.set_xticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_yticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_xlabel("Predicción", fontweight='bold')
    ax.set_ylabel("Real", fontweight='bold')
    for i in range(len(silabas)):
        for j in range(len(silabas)):
            val = cm_pct[i, j]
            col = "white" if val > 50 else "black"
            txt = f"{val:.0f}%" if val > 0 else "-"
            ax.text(j, i, txt, ha="center", va="center", color=col, fontweight='bold', fontsize=10)

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
cbar.set_label('Porcentaje de Clasificación (%)', fontweight='bold')
plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_silabas_mfcc_hibrido.png"
plt.savefig(out_f, dpi=300)
plt.close()
print(f"Heatmap comparativo guardado en: {out_f}")
