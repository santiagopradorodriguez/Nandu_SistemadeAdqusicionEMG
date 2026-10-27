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

base_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-10-23"
mediciones = [
    f"{v}_Pruebatricanal_{suf}"
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, f"{v}_Pruebatricanal_{suf}"))
]

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

def extraer_features_espectrales(sig, sr):
    fft_vals = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0/sr)
    pot = fft_vals**2
    sum_pot = np.sum(pot) + 1e-9
    mnf = np.sum(freqs * pot) / sum_pot
    cumsum = np.cumsum(pot)
    mdf = freqs[np.where(cumsum >= sum_pot / 2.0)[0][0]] if len(np.where(cumsum >= sum_pot / 2.0)[0]) > 0 else 0.0
    pfr = freqs[np.argmax(pot)]
    return np.array([mnf, mdf, pfr])

def extraer_features_ar(sig, order=4):
    from numpy.linalg import lstsq
    if len(sig) <= order: return np.zeros(order)
    y = sig[order:]
    X = np.column_stack([sig[order-i-1:-i-1] for i in range(order)])
    ar_coeffs, _, _, _ = lstsq(X, y, rcond=None)
    return ar_coeffs

muestras = []
b_n, a_n = iirnotch(50.0, 2.0, 6000)
b_b, a_b = butter(4, [20.0/3000, 450.0/3000], btype='bandpass')

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
        
    audio = wavs["canal_3"][offset_n:]
    win_mic = int(sr * 0.05)
    env_mic = np.convolve(np.abs(audio), np.ones(win_mic)/win_mic, mode='same')
    picos, _ = find_peaks(env_mic, distance=int(0.75 * muestras_pulso), height=np.max(env_mic)*0.20)
    pre_s, post_s = int(0.4 * muestras_pulso), int(0.6 * muestras_pulso)
    
    for p in picos:
        st, en = offset_n + p - pre_s, offset_n + p + post_s
        if st < 0 or en > len(wavs["canal_0"]): continue
        
        # Segmentos filtrados
        segs = {}
        max_supremo = 1e-9
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s_raw = wavs[ch][st:en]
            s_f = filtfilt(b_n, a_n, s_raw)
            s_f = filtfilt(b_b, a_b, s_f)
            segs[ch] = s_f
            m = np.max(np.abs(s_f))
            if m > max_supremo: max_supremo = m
            
        # 1. Envolvente
        from scipy.signal import resample
        win_e = int(sr * 0.05)
        e_list = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            env = np.convolve(np.abs(segs[ch]), np.ones(win_e)/win_e, mode='same')
            e_norm = resample(env / max_supremo, 60)
            e_list.append(e_norm)
        v_env = np.concatenate(e_list)
        
        # 2. Hudgins
        h_list = [extraer_features_hudgins(segs[ch] / max_supremo, sr) for ch in ["canal_0", "canal_1", "canal_2"]]
        v_hudgins = np.concatenate(h_list)
        
        # 3. Espectrales (MNF, MDF, Peak)
        spec_list = [extraer_features_espectrales(segs[ch], sr) for ch in ["canal_0", "canal_1", "canal_2"]]
        v_spec = np.concatenate(spec_list)
        
        # 4. AR orden 4
        ar_list = [extraer_features_ar(segs[ch] / (np.std(segs[ch]) + 1e-9), order=4) for ch in ["canal_0", "canal_1", "canal_2"]]
        v_ar = np.concatenate(ar_list)
        
        # 5. Mel EMG
        mel_list = []
        for ch in ["canal_0", "canal_1", "canal_2"]:
            s = segs[ch]
            s_n = (s - np.mean(s)) / (np.std(s) + 1e-9)
            ceps = compute_cepstrum_emg(s_n, sr, n_filtros=16, n_ceps=8, fmin=20, fmax=450)
            mel_list.extend([np.mean(ceps, axis=0), np.std(ceps, axis=0)])
        v_mel = np.concatenate(mel_list)
        
        muestras.append({
            'vocal': vocal,
            'env': v_env,
            'hudgins': v_hudgins,
            'spec': v_spec,
            'ar': v_ar,
            'mel': v_mel
        })

print(f"Total pulsos extraidos con todas las familias: {len(muestras)}")

Y = np.array([m['vocal'] for m in muestras])
mask_ou = np.isin(Y, ['O', 'U'])

V_env = np.array([m['env'] for m in muestras])
V_hudgins = np.array([m['hudgins'] for m in muestras])
V_spec = np.array([m['spec'] for m in muestras])
V_ar = np.array([m['ar'] for m in muestras])
V_mel = np.array([m['mel'] for m in muestras])

# Equilibrado de bloques frente a envolvente
gamma_mel = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_mel, axis=0)) + 1e-9))
gamma_ar = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_ar, axis=0)) + 1e-9))
gamma_spec = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_spec, axis=0)) + 1e-9))
gamma_hud = np.sqrt(np.mean(np.var(V_env, axis=0)) / (np.mean(np.var(V_hudgins, axis=0)) + 1e-9))

familias = {
    "1. Envolvente Temporal (180D)": V_env,
    "2. Hudgins Clásico (MAV, WL, ZC, SSC, RMS - 15D)": V_hudgins,
    "3. Frecuencias sEMG (MNF, MDF, Peak Freq - 9D)": V_spec,
    "4. Coeficientes AR Orden 4 (LPC sEMG - 12D)": V_ar,
    "5. Cepstrum Mel Muscular (20-450 Hz - 48D)": V_mel,
    "6. Híbrido Tiempo + Mel Muscular (228D)": np.hstack([V_env, V_mel * gamma_mel]),
    "7. Combo Completo (Tiempo + Mel + AR + Hudgins)": np.hstack([V_env, V_mel * gamma_mel, V_ar * gamma_ar, V_hudgins * gamma_hud])
}

print("\nComparativa Oficial de Familias de Features EMG (Exactitud Lineal y Separabilidad O vs U):")
print("-" * 90)
print(f"{'Familia de Features':<48} | {'Dim':<5} | {'LDA (5v)':<10} | {'O vs U':<8}")
print("-" * 90)

for nombre, X_mat in familias.items():
    lda_5v = LinearDiscriminantAnalysis().fit(X_mat, Y).score(X_mat, Y) * 100
    lda_ou = LinearDiscriminantAnalysis().fit(X_mat[mask_ou], Y[mask_ou]).score(X_mat[mask_ou], Y[mask_ou]) * 100
    print(f"{nombre:<48} | {X_mat.shape[1]:<5} | {lda_5v:6.1f}%    | {lda_ou:6.1f}%")

