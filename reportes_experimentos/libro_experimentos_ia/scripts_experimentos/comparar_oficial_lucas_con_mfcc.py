import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# Cargar el X_clean que acaba de extraer generador_pca_umap (492 muestras de Lucas)
import os, sys
sys.path.append('EMG_desarrollo/deep_learning/pca_umap_clustering')
import generador_pca_umap as gpu

base_dir = 'EMG_desarrollo/base_de_datos_electrodos'
mediciones = [
    os.path.join('2026-07-10', f'{v}_T{t}_Lucas')
    for v in ['A', 'E', 'I', 'O', 'U']
    for t in range(1, 8)
    if os.path.isdir(os.path.join(base_dir, '2026-07-10', f'{v}_T{t}_Lucas'))
]

params = {
    'smooth_ms': 250,
    'target_length': 100,
    'alpha_ruido': 1.0,
    'notch_q': 2.0,
    'tipo_filtro_ruido': 'notch',
    'highpass_cutoff_hz': 20.0,
    'lowpass_cutoff_hz': 300.0,
    'tipo_envolvente': 'rms',
    'snr_threshold': 0.0,
    'gate_ratio_ruido': 0.0,
    'outlier_contamination': 0.05
}

# Obtener raw cache de las mismas ventanas exactas
X_raw_list, Y_list, Tomas_list, SNRs = gpu.extraer_features_concatenadas(
    base_dir=base_dir, mediciones=mediciones,
    canales_features=['canal_0', 'canal_1', 'canal_2'],
    modo_alineacion='Pico Volumen Micrófono',
    return_raw_cache=True, smooth_ms=250, notch_q=2.0
)

# Funcion cepstrum Mel muscular
from scipy.fftpack import dct
def compute_cepstrum_emg(signal, sr=2000, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
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

# Extraer MFCC sobre esos mismos segmentos
from scipy.signal import resample
X_env_all = []
X_mfcc_all = []
for segs in X_raw_list:
    # segs es lista de 3 arrays (ch0, ch1, ch2)
    m_sup = max(np.max(segs[0]), np.max(segs[1]), np.max(segs[2]), 1e-9)
    # env
    v_e = np.concatenate([resample(s / m_sup, 100) for s in segs])
    X_env_all.append(v_e)
    # mfcc
    m_feats = []
    for s in segs:
        s_n = (s - np.mean(s)) / (np.std(s) + 1e-9)
        c = compute_cepstrum_emg(s_n, sr=2000)
        m_feats.extend([np.mean(c, axis=0), np.std(c, axis=0)])
    X_mfcc_all.append(np.concatenate(m_feats))

X_env_all = np.array(X_env_all)
X_mfcc_all = np.array(X_mfcc_all)
Y_all = np.array(Y_list)

# Isolation Forest por vocal (contamination=0.05)
from sklearn.ensemble import IsolationForest
valid_idx = []
vocales = ['A', 'E', 'I', 'O', 'U']
for v in vocales:
    idxs = np.where(Y_all == v)[0]
    iso = IsolationForest(contamination=0.05, random_state=42)
    p = iso.fit_predict(X_env_all[idxs])
    valid_idx.extend(idxs[p == 1])

valid_idx = sorted(valid_idx)
X_env_clean = X_env_all[valid_idx]
X_mfcc_clean = X_mfcc_all[valid_idx]
Y_clean = Y_all[valid_idx]

# PCA 2D
pca_env = PCA(n_components=2).fit_transform(X_env_clean)
pca_mfcc = PCA(n_components=2).fit_transform(X_mfcc_clean)

# Equilibrado Hibrido
gamma = np.sqrt(np.mean(np.var(X_env_clean, axis=0)) / (np.mean(np.var(X_mfcc_clean, axis=0)) + 1e-9))
X_hib_clean = np.hstack([X_env_clean, X_mfcc_clean * gamma])
pca_hib = PCA(n_components=2).fit_transform(X_hib_clean)

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y_clean == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_clean) * 100

mask_ou = np.isin(Y_clean, ['O', 'U'])
acc_km_env = eval_km(pca_env)
acc_km_mfcc = eval_km(pca_mfcc)
acc_km_hib = eval_km(pca_hib)

lda_ou_env = LinearDiscriminantAnalysis().fit(pca_env[mask_ou], Y_clean[mask_ou]).score(pca_env[mask_ou], Y_clean[mask_ou]) * 100
lda_ou_mfcc = LinearDiscriminantAnalysis().fit(pca_mfcc[mask_ou], Y_clean[mask_ou]).score(pca_mfcc[mask_ou], Y_clean[mask_ou]) * 100
lda_ou_hib = LinearDiscriminantAnalysis().fit(pca_hib[mask_ou], Y_clean[mask_ou]).score(pca_hib[mask_ou], Y_clean[mask_ou]) * 100

print(f"Envolvente Oficial Triángulo -> K-Means: {acc_km_env:.1f}% | Sep. O vs U: {lda_ou_env:.1f}%")
print(f"MFCC Mel Muscular           -> K-Means: {acc_km_mfcc:.1f}% | Sep. O vs U: {lda_ou_mfcc:.1f}%")
print(f"Híbrido (Tiempo + Mel)       -> K-Means: {acc_km_hib:.1f}% | Sep. O vs U: {lda_ou_hib:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

for ax, X_p, tit, km_v, ou_v in [
    (axes[0], pca_env, "1. Envolvente Oficial (Triángulo Canónico)", acc_km_env, lda_ou_env),
    (axes[1], pca_mfcc, "2. Mel Muscular sEMG", acc_km_mfcc, lda_ou_mfcc),
    (axes[2], pca_hib, "3. Híbrido Tiempo + Mel", acc_km_hib, lda_ou_hib)
]:
    for v in vocales:
        m = (Y_clean == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means: {km_v:.1f}% | O vs U: {ou_v:.1f}%", fontweight='bold', fontsize=11)
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lucas_triangulo_vs_mfcc.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
