import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import find_peaks, resample, butter, filtfilt, iirnotch
from scipy.fftpack import dct
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-07-10"
vocales = ['A', 'E', 'I', 'O', 'U']

# Escala Mel adaptada a musculo (f_pivote = 150 Hz)
def compute_cepstrum_emg(signal, sr, n_filtros=16, n_ceps=8, fmin=20, fmax=450, n_fft=512, hop_length=256):
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

def extraer_features_hudgins(sig, sr, threshold=0.01):
    mav = np.mean(np.abs(sig))
    wl = np.sum(np.abs(np.diff(sig)))
    rms = np.sqrt(np.mean(sig**2))
    zc = np.sum((sig[:-1] * sig[1:] < 0) & (np.abs(sig[:-1] - sig[1:]) > threshold))
    diff1 = np.diff(sig)
    ssc = np.sum((diff1[:-1] * diff1[1:] < 0) & (np.abs(diff1[:-1] - diff1[1:]) > threshold))
    return np.array([mav, wl, rms, zc, ssc])

b_n, a_n = iirnotch(50.0, 2.0, 2000) # En julio sr = 2000 Hz
b_b, a_b = butter(4, [20.0/1000, 450.0/1000], btype='bandpass')

pulsos = []
for v in vocales:
    for t_idx in range(1, 8):
        med_folder = f"{v}_T{t_idx}_Lucas"
        carp = os.path.join(base_dir, med_folder)
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
            wp = os.path.join(carp, ch, "grabacion.wav")
            _, d = wavfile.read(wp)
            wavs[ch] = d.astype(np.float32)
            
        audio = wavs["canal_3"][offset_n:]
        win_mic = int(sr * 0.05)
        env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
        picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
        pre_s, post_s = int(0.4 * muestras_pulso), int(0.6 * muestras_pulso)
        
        for p in picos:
            st, en = offset_n + p - pre_s, offset_n + p + post_s
            if st < 0 or en > len(wavs["canal_0"]): continue
            
            segs = {}
            max_supremo = 1e-9
            for ch in ["canal_0", "canal_1", "canal_2"]:
                s = filtfilt(b_n, a_n, wavs[ch][st:en])
                s = filtfilt(b_b, a_b, s)
                segs[ch] = s
                m = np.max(np.abs(s))
                if m > max_supremo: max_supremo = m
                
            # 1. Envolvente
            win_e = int(sr * 0.05)
            e_list = []
            for ch in ["canal_0", "canal_1", "canal_2"]:
                env = np.convolve(np.abs(segs[ch]), np.ones(win_e)/win_e, mode='same')
                e_norm = resample(env / max_supremo, 60)
                e_list.append(e_norm)
            v_env = np.concatenate(e_list)
            
            # 2. Mel Muscular
            mel_list = []
            for ch in ["canal_0", "canal_1", "canal_2"]:
                s = segs[ch]
                s_n = (s - np.mean(s)) / (np.std(s) + 1e-9)
                ceps = compute_cepstrum_emg(s_n, sr, n_filtros=16, n_ceps=8, fmin=20, fmax=450)
                mel_list.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
            v_mel = np.concatenate(mel_list)
            
            # 3. Hudgins
            h_list = [extraer_features_hudgins(segs[ch] / max_supremo, sr) for ch in ["canal_0", "canal_1", "canal_2"]]
            v_hud = np.concatenate(h_list)
            
            pulsos.append({
                'vocal': v,
                'env': v_env,
                'mel': v_mel,
                'hud': v_hud
            })

print(f"Total pulsos extraidos de Lucas (2026-07-10): {len(pulsos)}")

Y = np.array([p['vocal'] for p in pulsos])
V_env = np.array([p['env'] for p in pulsos])
V_mel = np.array([p['mel'] for p in pulsos])
V_hud = np.array([p['hud'] for p in pulsos])

# Equilibrado de bloque
gamma_mel = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_mel, axis=0)) + 1e-9))
gamma_hud = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_hud, axis=0)) + 1e-9))

X_hib = np.hstack([V_env, V_mel * gamma_mel])

# Evaluacion con Stratified 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mask_ou = np.isin(Y, ['O', 'U'])
Y_ou = Y[mask_ou]

familias_lucas = {
    "1. Envolvente Temporal (180D)": V_env,
    "2. Mel Muscular sEMG (48D)": V_mel,
    "3. Hudgins Clásico (15D)": V_hud,
    "4. Híbrido Equilibrado (Tiempo + Mel)": X_hib
}

print("\nResultados Validación Cruzada (CV-5) - Sesión Lucas 2026-07-10:")
print("-" * 80)
print(f"{'Familia de Features':<42} | {'Exactitud 5v (CV-5)':<20} | {'O vs U (CV-5)':<12}")
print("-" * 80)

for nombre, X_mat in familias_lucas.items():
    sc_5v = []
    for tr, ts in cv.split(X_mat, Y):
        lda = LinearDiscriminantAnalysis().fit(X_mat[tr], Y[tr])
        sc_5v.append(lda.score(X_mat[ts], Y[ts]))
        
    X_ou = X_mat[mask_ou]
    sc_ou = []
    for tr, ts in cv.split(X_ou, Y_ou):
        lda = LinearDiscriminantAnalysis().fit(X_ou[tr], Y_ou[tr])
        sc_ou.append(lda.score(X_ou[ts], Y_ou[ts]))
        
    print(f"{nombre:<42} | {np.mean(sc_5v)*100:5.1f}% (+/- {np.std(sc_5v)*100:4.1f}%)    | {np.mean(sc_ou)*100:5.1f}%")

# Gráfico PCA 2D lado a lado en Lucas
pca_env = PCA(n_components=2, random_state=42).fit_transform(V_env)
pca_mel = PCA(n_components=2, random_state=42).fit_transform(V_mel)
pca_hib = PCA(n_components=2, random_state=42).fit_transform(X_hib)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

for ax, X_p, tit in [
    (axes[0], pca_env, "1. Envolvente Temporal"),
    (axes[1], pca_mel, "2. Mel Muscular sEMG"),
    (axes[2], pca_hib, "3. Híbrido Tiempo + Mel")
]:
    for v in vocales:
        m = (Y == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=40, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(tit, fontweight='bold')
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/comparativa_lucas_07_10_mfcc.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("-" * 80)
print("Figura guardada en:", out_f)
