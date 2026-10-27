import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
# Carpetas de silabas grabadas en esa sesion
silabas_carpetas = [d for d in sorted(os.listdir(base_dir)) if "SILABAS" in d and os.path.isdir(os.path.join(base_dir, d))]

print("Carpetas de sílabas encontradas:", len(silabas_carpetas))
for c in silabas_carpetas: print("  -", c)

pulsos_silabas = []
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
        
        # Envolvente sEMG (50 ms)
        win_e = int(sr * 0.05)
        envs = []
        max_sup = 1e-9
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s_f = filtfilt(b_n, a_n, wavs[ch][st:en])
            s_f = filtfilt(b_b, a_b, s_f)
            e = np.convolve(np.abs(s_f), np.ones(win_e)/win_e, mode='same')
            m = np.max(e)
            if m > max_sup: max_sup = m
            envs.append(e)
            
        # Normalizacion por supremo tricanal
        v_env = np.concatenate([resample(e / max_sup, 60) for e in envs])
        pulsos_silabas.append({
            'silaba': silaba,
            'feat_env': v_env
        })

print(f"\nTotal pulsos de sílabas extraídos: {len(pulsos_silabas)}")
Y = np.array([p['silaba'] for p in pulsos_silabas])
X = np.array([p['feat_env'] for p in pulsos_silabas])

silabas_unicas, counts = np.unique(Y, return_counts=True)
for s, c in zip(silabas_unicas, counts):
    print(f"  /{s}/: {c} muestras")

# 1. Evaluacion LDA en espacio completo (180D)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores_cv = []
for tr, ts in cv.split(X, Y):
    lda = LinearDiscriminantAnalysis().fit(X[tr], Y[tr])
    scores_cv.append(lda.score(X[ts], Y[ts]))

acc_cv = np.mean(scores_cv) * 100
print(f"\nExactitud Clasificación Sílabas (LDA CV-3, {len(silabas_unicas)} clases): {acc_cv:.1f}% (+/- {np.std(scores_cv)*100:.1f}%)")

# 2. PCA 2D
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
var = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(10, 8))
cmap = plt.get_cmap('tab10')
for idx, s in enumerate(silabas_unicas):
    m = (Y == s)
    plt.scatter(X_pca[m, 0], X_pca[m, 1], label=f"/{s.lower()}/", s=55, alpha=0.85, edgecolors='black', linewidths=0.5, color=cmap(idx % 10))

plt.title(f"PCA 2D Sílabas CV (2026-10-23) - {len(silabas_unicas)} Clases\nVar Explicada: PC1={var[0]:.1f}%, PC2={var[1]:.1f}% (Total: {np.sum(var):.1f}%)", fontweight='bold')
plt.xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold')
plt.ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_silabas_2026_10_23.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
