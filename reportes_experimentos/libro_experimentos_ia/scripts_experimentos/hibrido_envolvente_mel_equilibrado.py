import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import IsolationForest
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
mediciones = [
    f"{v}_Pruebatricanal_{suf}"
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}"))
]

# Funcion cepstral con escala Mel Muscular (f_pivote = 150 Hz, 20-450 Hz)
def compute_cepstrum_emg(signal, sr, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
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
    hz_points = mel_emg_to_hz(m_pts)
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

# Cargar y extraer por pulso respetando la norma tricanal
datos_pulsos = []
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
        
    # Sincronizacion por microfono (Canal 3)
    audio = wavs["canal_3"][offset_n:]
    win_mic = int(sr * 0.05)
    env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
    picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
    
    pre_s = int(0.4 * muestras_pulso)
    post_s = int(0.6 * muestras_pulso)
    
    for p in picos:
        st = offset_n + p - pre_s
        en = offset_n + p + post_s
        if st < 0 or en > len(wavs["canal_0"]): continue
        
        # 1. Envolventes sEMG (50 ms, filtradas)
        win_emg = int(sr * 0.05)
        e0 = np.convolve(np.abs(wavs["canal_0"][st:en]), np.ones(win_emg)/win_emg, mode='same')
        e1 = np.convolve(np.abs(wavs["canal_1"][st:en]), np.ones(win_emg)/win_emg, mode='same')
        e2 = np.convolve(np.abs(wavs["canal_2"][st:en]), np.ones(win_emg)/win_emg, mode='same')
        
        # NORMA TRICANAL SUPREMO
        M_supremo = max(np.max(e0), np.max(e1), np.max(e2), 1e-9)
        
        # Envolventes normalizadas por supremo (60 pts c/u)
        target_len = 60
        e0_n = resample(e0 / M_supremo, target_len)
        e1_n = resample(e1 / M_supremo, target_len)
        e2_n = resample(e2 / M_supremo, target_len)
        # Vector temporal tricanal (180 puntos)
        v_tiempo = np.concatenate([e0_n, e1_n, e2_n])
        
        # 2. MFCCs por canal (señal normalizada por su desvio interno para forma espectral pura)
        mfcc_feats = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s = wavs[ch][st:en]
            s_c = s - np.mean(s)
            s_norm = s_c / (np.std(s_c) + 1e-9)
            ceps = compute_cepstrum_emg(s_norm, sr, n_filtros=16, n_ceps=8, fmin=20, fmax=450)
            # Medias y desvios de los 8 coeficientes cepstrales = 16 features
            mfcc_feats.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
            
        v_frecuencia = np.concatenate(mfcc_feats) # 48 features
        
        datos_pulsos.append({
            'vocal': vocal,
            'v_tiempo': v_tiempo,
            'v_frecuencia': v_frecuencia
        })

print(f"Total pulsos procesados: {len(datos_pulsos)}")

# Convertir a matrices
V_T = np.array([p['v_tiempo'] for p in datos_pulsos])      # (N, 180) - acotado en [0, 1]
V_F = np.array([p['v_frecuencia'] for p in datos_pulsos])  # (N, 48)  - escala dB/cepstral
Y = np.array([p['vocal'] for p in datos_pulsos])

# EQUILIBRADO DE ESCALA POR INERCIA DE BLOQUE (Sin StandardScaler variable a variable)
# Calculamos la varianza media por dimension de cada bloque
sigma2_T = np.mean(np.var(V_T, axis=0)) # Varianza media por punto de la envolvente
sigma2_F = np.mean(np.var(V_F, axis=0)) # Varianza media por coeficiente MFCC

gamma = np.sqrt(sigma2_T / (sigma2_F + 1e-9))
print(f"Varianza media Tiempo: {sigma2_T:.4f} | Frecuencia: {sigma2_F:.4f}")
print(f"Factor de equilibrado de bloque gamma: {gamma:.4f}")

# V_F escalado uniformemente: preserva 100% la estructura interna y la norma tricanal de V_T
V_F_equilibrado = V_F * gamma

# Generamos variantes para comparar
vocales_list = ['A', 'E', 'I', 'O', 'U']
mask_ou = np.isin(Y, ['O', 'U'])

variantes = {
    "1. Solo Envolvente Tricanal (180D)": V_T,
    "2. Solo MFCC Mel Muscular (48D)": V_F_equilibrado,
    "3. Híbrido Equilibrado (Tiempo + Mel EMG)": np.hstack([V_T, V_F_equilibrado]),
    "4. Híbrido Ponderado (Tiempo + 0.5 * Mel EMG)": np.hstack([V_T, 0.5 * V_F_equilibrado])
}

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

print("\nResultados Comparativos en PCA 2D (Equilibrado de Bloque, Sin StandardScaler):")
print("-" * 85)
print(f"{'Variante':<45} | {'K-Means':<10} | {'LDA (5v)':<10} | {'O vs U':<8}")
print("-" * 85)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

for idx, (nombre, X_mat) in enumerate(variantes.items()):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_mat)
    var = np.sum(pca.explained_variance_ratio_) * 100
    
    # K-Means
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_pca)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales_list):
        for j in range(5): cost[i, j] = -np.sum((Y == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales_list[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    acc_km = np.mean(preds == Y) * 100
    
    # LDA
    lda_5v = LinearDiscriminantAnalysis().fit(X_pca, Y).score(X_pca, Y) * 100
    lda_ou = LinearDiscriminantAnalysis().fit(X_pca[mask_ou], Y[mask_ou]).score(X_pca[mask_ou], Y[mask_ou]) * 100
    
    print(f"{nombre:<45} | {acc_km:6.1f}%    | {lda_5v:6.1f}%    | {lda_ou:6.1f}%")
    
    ax = axes[idx]
    for v in vocales_list:
        m = (Y == v)
        ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{nombre}\nK-Means: {acc_km:.1f}% | LDA 5v: {lda_5v:.1f}% | O vs U: {lda_ou:.1f}% (Var: {var:.1f}%)", fontweight='bold', fontsize=11)
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/hibrido_tiempo_mel_emg.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("-" * 85)
print("Figura comparativa guardada en:", out_f)
