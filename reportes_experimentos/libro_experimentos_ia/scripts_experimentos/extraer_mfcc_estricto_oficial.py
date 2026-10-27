import os
import sys
import json
import pandas as pd
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# 1. Cargar las tomas y ventanas exactas oficiales de generador_pca_umap
csv_oficial = "reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/caracteristicas_exportadas.csv"
df_oficial = pd.read_csv(csv_oficial)
print(f"Pulsos oficiales cargados: {len(df_oficial)}")

vocales_oficial = df_oficial['Vocal'].values
tomas_oficial = df_oficial['Toma'].values
X_env_oficial = df_oficial.iloc[:, 2:].values # 180 features (60 por canal)

# Cargar pesos [2.5, 1.0, 1.0] aplicados en la oficial
pesos = [2.5, 1.0, 1.0]
pts_per_ch = 60
X_env_weighted = X_env_oficial.copy()
for c_idx, w in enumerate(pesos):
    X_env_weighted[:, c_idx*pts_per_ch:(c_idx+1)*pts_per_ch] *= w

# 2. Función MFCC idéntica
def compute_mfcc(signal, sr, n_mfcc=13, n_mels=26, n_fft=512, hop_length=256, fmin=20, fmax=500):
    if len(signal) < n_fft:
        pad_width = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')
    num_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal[:num_frames * hop_length + n_fft - hop_length], n_fft)[::hop_length]
    window = np.hamming(n_fft)
    frames = frames * window
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
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20.0 * np.log10(filter_banks)
    return dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]

# 3. Extraer exactamente sobre las tomas de la lista oficial usando la ventana alineada por micrófono
base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"

# Cacheamos grabaciones
cache_wavs = {}
X_mfcc_list = []
indices_validos = []

for idx, (vocal, toma_str) in enumerate(zip(vocales_oficial, tomas_oficial)):
    # toma_str ej: A_Pruebatricanal_2_Sujeto1_Win3
    partes = toma_str.split('_Win')
    carpeta_nom = partes[0]
    win_idx = int(partes[1])
    
    med_path = os.path.join(base_dir, carpeta_nom)
    if carpeta_nom not in cache_wavs:
        # Cargar metadata y wavs
        meta_p = os.path.join(med_path, "canal_0", "metadata.json")
        with open(meta_p) as f: meta = json.load(f)
        sr = meta.get('sample_rate', 6000)
        bpm = meta.get('bpm', 30)
        noise_sec = meta.get('noise_seconds', 3.0)
        
        wavs = {}
        for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
            _, d = wavfile.read(os.path.join(med_path, ch, "grabacion.wav"))
            wavs[ch] = d.astype(np.float32)
            
        # Calcular los picos de micrófono idéntico a analisis_trevisan
        audio = wavs["canal_3"]
        offset_n = int(noise_sec * sr)
        audio_rec = audio[offset_n:]
        
        # Envolvente suave mic
        win_size = int(sr * 0.05)
        env_mic = np.convolve(np.abs(audio_rec), np.ones(win_size)/win_size, mode='same')
        muestras_pulso = int(sr * (60.0 / bpm))
        from scipy.signal import find_peaks
        picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
        cache_wavs[carpeta_nom] = {
            'wavs': wavs,
            'sr': sr,
            'muestras_pulso': muestras_pulso,
            'offset_n': offset_n,
            'picos': picos
        }
        
    c_info = cache_wavs[carpeta_nom]
    sr = c_info['sr']
    picos = c_info['picos']
    muestras_pulso = c_info['muestras_pulso']
    offset_n = c_info['offset_n']
    
    if win_idx < len(picos):
        p_mic = picos[win_idx]
        pre_s = int(0.4 * muestras_pulso)
        post_s = int(0.6 * muestras_pulso)
        st = offset_n + p_mic - pre_s
        en = offset_n + p_mic + post_s
        
        mfccs_ch = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s_raw = c_info['wavs'][ch][st:en]
            # Normalizado por desviación estándar independiente por canal (sin tocar amplitudes relativas espurias)
            s_c = s_raw - np.mean(s_raw)
            s_norm = s_c / (np.std(s_c) + 1e-9)
            m = compute_mfcc(s_norm, sr, n_mfcc=8, n_mels=16, n_fft=512, hop_length=256, fmin=20, fmax=500)
            mfccs_ch.extend([np.mean(m, axis=0), np.std(m, axis=0)])
        
        X_mfcc_list.append(np.concatenate(mfccs_ch))
        indices_validos.append(idx)

print(f"Pulsos perfectamente alineados y extraídos con MFCC: {len(X_mfcc_list)}")

X_mfcc_arr = np.array(X_mfcc_list)
Y_arr = vocales_oficial[indices_validos]
X_env_arr = X_env_weighted[indices_validos]
vocales = ['A', 'E', 'I', 'O', 'U']

# Comparativa 1 a 1 exacta sobre el mismo dataset exacto
pca_env = PCA(n_components=2, random_state=42)
X_pca_env = pca_env.fit_transform(X_env_arr)

pca_mfcc = PCA(n_components=2, random_state=42)
X_pca_mfcc = pca_mfcc.fit_transform(X_mfcc_arr)

def eval_km(X_p, Y_t):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5):
            cost[i, j] = -np.sum((Y_t == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_t) * 100

mask_ou = np.isin(Y_arr, ['O', 'U'])
lda_env_ou = LinearDiscriminantAnalysis().fit(X_pca_env[mask_ou], Y_arr[mask_ou])
lda_mfcc_ou = LinearDiscriminantAnalysis().fit(X_pca_mfcc[mask_ou], Y_arr[mask_ou])

acc_km_env = eval_km(X_pca_env, Y_arr)
acc_km_mfcc = eval_km(X_pca_mfcc, Y_arr)
acc_ou_env = lda_env_ou.score(X_pca_env[mask_ou], Y_arr[mask_ou]) * 100
acc_ou_mfcc = lda_mfcc_ou.score(X_pca_mfcc[mask_ou], Y_arr[mask_ou]) * 100

print(f"Envolvente Oficial (sin scaler) -> K-Means: {acc_km_env:.1f}% | Sep. O vs U: {acc_ou_env:.1f}%")
print(f"MFCC sEMG Oficial (sin scaler)  -> K-Means: {acc_km_mfcc:.1f}% | Sep. O vs U: {acc_ou_mfcc:.1f}%")

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Oficial Envolvente
for v in vocales:
    m = (Y_arr == v)
    axes[0].scatter(X_pca_env[m, 0], X_pca_env[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
axes[0].set_title(f"1. Envolvente sEMG (Alineación Micrófono Oficial)\nK-Means: {acc_km_env:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou_env:.1f}%", fontweight='bold')
axes[0].set_xlabel("PC 1", fontweight='bold')
axes[0].set_ylabel("PC 2", fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='best', frameon=True)

# Panel 2: MFCC sobre mismas ventanas
for v in vocales:
    m = (Y_arr == v)
    axes[1].scatter(X_pca_mfcc[m, 0], X_pca_mfcc[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
axes[1].set_title(f"2. MFCC sEMG puro (Mismas Ventanas, Sin Scaler)\nK-Means: {acc_km_mfcc:.1f}% | Sep. Lineal /o/ vs /u/: {acc_ou_mfcc:.1f}%", fontweight='bold')
axes[1].set_xlabel("PC 1", fontweight='bold')
axes[1].set_ylabel("PC 2", fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='best', frameon=True)

plt.tight_layout()
out_fig = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_oficial_vs_mfcc_estricta.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print("Grafico guardado en:", out_fig)
