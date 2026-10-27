# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Experimento comparativo de PCA con features temporales (envolvente)
#              vs features frecuenciales (MFCC audio Ch3 y MFCC sEMG Ch0-Ch2)
#              en la sesion Triada Masetero-Modiolo-Orbicularis (2026-10-23).
# ==============================================================================

import os
import sys
import json
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import resample, find_peaks
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

np.random.seed(42)

# Colores oficiales del proyecto
COLOR_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

def compute_mfcc(signal, sr, n_mfcc=13, n_mels=26, n_fft=512, hop_length=256, fmin=50, fmax=None):
    if fmax is None:
        fmax = sr / 2.0
    if len(signal) < n_fft:
        pad_width = n_fft - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')
    
    num_frames = max(1, 1 + (len(signal) - n_fft) // hop_length)
    frames = np.lib.stride_tricks.sliding_window_view(signal[:num_frames * hop_length + n_fft - hop_length], n_fft)[::hop_length]
    window = np.hamming(n_fft)
    frames = frames * window
    
    mag_frames = np.abs(np.fft.rfft(frames, n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)
    
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)
        
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bin_points = np.clip(bin_points, 0, int(n_fft // 2))
    
    fbank = np.zeros((n_mels, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, n_mels + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))
            
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20.0 * np.log10(filter_banks)
    
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    return mfcc

def extraer_datos_triada(base_dir="EMG_desarrollo/base_de_datos_electrodos/2026-10-23"):
    vocales = ['A', 'E', 'I', 'O', 'U']
    muestras = []
    
    for v in vocales:
        for sufijo in ["Sujeto1", "2_Sujeto1", "3_Sujeto1", "4_Sujeto1"]:
            carpeta_toma = os.path.join(base_dir, f"{v}_Pruebatricanal_{sufijo}")
            if not os.path.isdir(carpeta_toma):
                continue
                
            # Cargar metadata general
            meta_path = os.path.join(carpeta_toma, "canal_0", "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            sr = meta.get('sample_rate', 6000)
            bpm = meta.get('bpm', 30)
            noise_sec = meta.get('noise_seconds', 3.0)
            muestras_pulso = int(sr * (60.0 / bpm))
            offset_noise = int(noise_sec * sr)
            
            # Cargar audios crudos de los 4 canales
            raw_channels = {}
            for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
                wav_p = os.path.join(carpeta_toma, ch, "grabacion.wav")
                if not os.path.exists(wav_p):
                    break
                _, data = wavfile.read(wav_p)
                raw_channels[ch] = data.astype(np.float32)
                
            if len(raw_channels) < 4:
                continue
                
            # Canal de audio para encontrar picos
            audio_raw = raw_channels["canal_3"][offset_noise:]
            # Envolvente suave de audio para alinear
            win_env = int(sr * 0.05)
            env_audio = np.convolve(np.abs(audio_raw), np.ones(win_env)/win_env, mode='same')
            
            dist_samples = int(0.7 * muestras_pulso)
            min_height = np.max(env_audio) * 0.20
            picos, _ = find_peaks(env_audio, distance=dist_samples, height=min_height)
            
            pre_samp = int(0.4 * muestras_pulso)
            post_samp = int(0.6 * muestras_pulso)
            
            for idx_p, p in enumerate(picos):
                # Omitir primer pulso transitorio si corresponde
                if idx_p == 0:
                    continue
                start_idx = offset_noise + p - pre_samp
                end_idx = offset_noise + p + post_samp
                
                if start_idx < 0 or end_idx > len(raw_channels["canal_0"]):
                    continue
                    
                # Extraer ventanas crudas
                seg_ch0 = raw_channels["canal_0"][start_idx:end_idx]
                seg_ch1 = raw_channels["canal_1"][start_idx:end_idx]
                seg_ch2 = raw_channels["canal_2"][start_idx:end_idx]
                seg_ch3 = raw_channels["canal_3"][start_idx:end_idx]
                
                # Envolvente sEMG (rectificada + filtrada/suavizada)
                win_emg = int(sr * 0.10) # 100 ms
                e0 = np.convolve(np.abs(seg_ch0), np.ones(win_emg)/win_emg, mode='same')
                e1 = np.convolve(np.abs(seg_ch1), np.ones(win_emg)/win_emg, mode='same')
                e2 = np.convolve(np.abs(seg_ch2), np.ones(win_emg)/win_emg, mode='same')
                
                # Normalización Tricanal Clásica por Supremo
                supremo = max(np.max(e0), np.max(e1), np.max(e2), 1e-9)
                e0_norm = resample(e0 / supremo, 100)
                e1_norm = resample(e1 / supremo, 100)
                e2_norm = resample(e2 / supremo, 100)
                feat_env = np.concatenate([e0_norm, e1_norm, e2_norm])
                
                # MFCC sobre Audio (Canal 3)
                mfcc_audio = compute_mfcc(seg_ch3, sr, n_mfcc=13, n_mels=26, n_fft=512, hop_length=256, fmin=80, fmax=sr/2.0)
                # Resumen temporal: media y desvio estandar de cada coeficiente
                feat_mfcc_audio = np.concatenate([np.mean(mfcc_audio, axis=0), np.std(mfcc_audio, axis=0)])
                
                # MFCC sobre sEMG (Canal 0, 1, 2) adaptado a 20-500 Hz
                mfccs_emg = []
                for s_emg in [seg_ch0, seg_ch1, seg_ch2]:
                    # Normalizar por desviación de canal (sin supremo intercanal)
                    s_clean = s_emg - np.mean(s_emg)
                    s_norm = s_clean / (np.std(s_clean) + 1e-9)
                    m_emg = compute_mfcc(s_norm, sr, n_mfcc=8, n_mels=16, n_fft=512, hop_length=256, fmin=20, fmax=500)
                    mfccs_emg.extend([np.mean(m_emg, axis=0), np.std(m_emg, axis=0)])
                feat_mfcc_emg = np.concatenate(mfccs_emg)
                
                muestras.append({
                    'vocal': v,
                    'toma': f"{v}_Pruebatricanal_{sufijo}_p{idx_p}",
                    'feat_env': feat_env,
                    'feat_mfcc_audio': feat_mfcc_audio,
                    'feat_mfcc_emg': feat_mfcc_emg,
                    'feat_hibrido': np.concatenate([feat_env, feat_mfcc_audio])
                })
                
    return muestras

def evaluar_clustering(X, Y):
    vocales_unicas = sorted(list(set(Y)))
    k = len(vocales_unicas)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20).fit(X)
    labels_km = kmeans.labels_
    
    # Matriz de asignacion hungara
    cost = np.zeros((k, k))
    for i, true_l in enumerate(vocales_unicas):
        for j in range(k):
            cost[i, j] = -np.sum((np.array(Y) == true_l) & (labels_km == j))
            
    row_ind, col_ind = linear_sum_assignment(cost)
    map_cluster = {col_ind[i]: vocales_unicas[row_ind[i]] for i in range(k)}
    preds = [map_cluster[c] for c in labels_km]
    acc = np.mean(np.array(preds) == np.array(Y))
    sil = silhouette_score(X, Y) if len(set(Y)) > 1 else 0.0
    return acc, sil

def main():
    print("Cargando y procesando tomas de la Triada (2026-10-23)...")
    muestras = extraer_datos_triada()
    print(f"Total de pulsos extraidos: {len(muestras)}")
    if len(muestras) == 0:
        print("Error: No se pudieron extraer pulsos.")
        return
        
    Y = [m['vocal'] for m in muestras]
    
    variantes = {
        '1. Envolvente sEMG (Tricanal Supremo)': np.array([m['feat_env'] for m in muestras]),
        '2. MFCC sEMG (Ch0, Ch1, Ch2 - 20-500 Hz)': np.array([m['feat_mfcc_emg'] for m in muestras]),
        '3. MFCC Audio (Canal 3 - Microfono)': np.array([m['feat_mfcc_audio'] for m in muestras]),
        '4. Hibrido (Envolvente sEMG + MFCC Audio)': np.array([m['feat_hibrido'] for m in muestras])
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    print("\nResultados Comparativos PCA (2D):")
    print("-" * 70)
    print(f"{'Variante':<45} | {'Acc (K-Means)':<12} | {'Silueta':<8}")
    print("-" * 70)
    
    for idx, (nombre, X_raw) in enumerate(variantes.items()):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        acc, sil = evaluar_clustering(X_pca, Y)
        var_exp = np.sum(pca.explained_variance_ratio_) * 100.0
        
        print(f"{nombre:<45} | {acc*100:6.1f}%      | {sil:6.3f}")
        
        ax = axes[idx]
        for v in ['A', 'E', 'I', 'O', 'U']:
            mask = np.array(Y) == v
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f"/{v.lower()}/",
                       color=COLOR_VOCALES[v], alpha=0.75, edgecolors='black', linewidths=0.5, s=45)
                       
        ax.set_title(f"{nombre}\nAcc K-Means: {acc*100:.1f}% | Silueta: {sil:.2f} | Var: {var_exp:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', frameon=True)
        
    plt.tight_layout()
    output_dir = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos"
    os.makedirs(output_dir, exist_ok=True)
    out_img = os.path.join(output_dir, "comparativa_pca_mfcc_vs_envolvente_2026_10_23.png")
    plt.savefig(out_img, dpi=300)
    plt.close()
    print("-" * 70)
    print(f"Grafico comparativo guardado exitosamente en:\n{out_img}")

if __name__ == "__main__":
    main()
