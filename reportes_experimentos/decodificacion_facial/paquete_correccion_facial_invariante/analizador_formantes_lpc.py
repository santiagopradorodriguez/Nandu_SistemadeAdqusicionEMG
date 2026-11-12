import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, resample_poly
from scipy.linalg import solve_toeplitz
from sklearn.ensemble import IsolationForest

s_dir = 'EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1'
p_wav = f'{s_dir}/canal_0/grabacion.wav'
fs_raw, audio_raw = wav.read(p_wav)

# Decimar a 10 kHz
fs = 10000
audio_10k = resample_poly(audio_raw.astype(np.float64), 100, 441)
audio_10k -= np.mean(audio_10k)

# Filtro pasabanda vocal (180 a 3800 Hz)
b, a = butter(4, [180.0 / (fs / 2.0), 3800.0 / (fs / 2.0)], btype='band')
audio_vocal = filtfilt(b, a, audio_10k)

df_pulsos = pd.read_csv(f'{s_dir}/pulsos_metronomo.csv')
df_princ = df_pulsos[(df_pulsos['principal'] == 1) & (df_pulsos['t_wav_s'].notna())].copy()

records = []
colores_canonicos = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

for idx, row in df_princ.iterrows():
    p_num = int(row['idx'])
    palabra = str(row['palabra']).upper().strip()
    tw = float(row['t_wav_s'])
    dur = float(row['intervalo_s'])
    
    # 1. Ventana acotada estrictamente al intervalo de este pulso [tw + 0.10, tw + dur]
    s0 = int((tw + 0.10) * fs)
    s1 = int((tw + dur) * fs)
    if s1 > len(audio_vocal): break
    sub = audio_vocal[s0:s1]
    
    # Envolvente RMS local (40 ms)
    w_env = int(0.040 * fs)
    if len(sub) < w_env * 2: continue
    env = np.convolve(sub**2, np.ones(w_env)/w_env, mode='same')
    
    # Buscar pico de mayor energía fonatoria
    i_max = np.argmax(env)
    pico_rms = np.sqrt(env[i_max])
    
    # Extraer 50 ms centrados en el pico fonatorio
    w_seg = int(0.050 * fs)
    i_start = max(0, i_max - w_seg // 2)
    i_end = min(len(sub), i_start + w_seg)
    seg = sub[i_start:i_end]
    if len(seg) < w_seg: continue
    
    # Preénfasis
    seg_pre = np.append(seg[0], seg[1:] - 0.96 * seg[:-1])
    seg_win = seg_pre * np.hamming(len(seg_pre))
    
    # LPC orden 12
    p = 12
    r = np.correlate(seg_win, seg_win, mode='full')
    r = r[len(seg_win)-1 : len(seg_win)+p]
    
    try:
        a_lpc = solve_toeplitz((r[:-1], r[:-1]), r[1:])
        poly = np.concatenate(([1], -a_lpc))
        raices = np.roots(poly)
        raices = [rc for rc in raices if np.imag(rc) > 0]
        angulos = np.arctan2(np.imag(raices), np.real(raices))
        freqs = angulos * (fs / (2.0 * np.pi))
        bws = -0.5 * (fs / (2.0 * np.pi)) * np.log(np.maximum(np.abs(raices), 1e-9))
        
        # Filtrar candidatos a formantes fisiológicos (ancho de banda < 400 Hz)
        cands = [(f_v, bw_v) for f_v, bw_v in zip(freqs, bws) if 180 < f_v < 3600 and bw_v < 400]
        cands.sort(key=lambda x: x[0])
        
        if len(cands) >= 2:
            records.append({
                'pulso': p_num,
                'vocal': palabra,
                'F1': cands[0][0],
                'F2': cands[1][0],
                'pico_rms': pico_rms
            })
    except Exception:
        pass

df_all = pd.DataFrame(records)
print(f"Total pulsos analizados: {len(df_all)}")

# 2. Purgar outliers mediante Isolation Forest por clase (10% contaminación canónica del proyecto)
clean_records = []
for v in ['A', 'E', 'I', 'O', 'U']:
    sub_v = df_all[df_all['vocal'] == v].copy()
    if len(sub_v) > 6:
        feats = sub_v[['F1', 'F2']].values
        iso = IsolationForest(contamination=0.10, random_state=42)
        inliers = iso.fit_predict(feats)
        sub_v = sub_v[inliers == 1]
    clean_records.append(sub_v)

df_clean = pd.concat(clean_records, ignore_index=True)
print(f"Total pulsos limpios (post Isolation Forest 10%): {len(df_clean)}")

print("\n" + "="*70)
print(f"{'Vocal':<8} | {'N':<4} | {'F1 (Media ± Std)':<22} | {'F2 (Media ± Std)':<22}")
print("="*70)
for v in ['A', 'E', 'I', 'O', 'U']:
    s = df_clean[df_clean['vocal'] == v]
    print(f"/{v}/      | {len(s):<4} | {s['F1'].mean():.1f} ± {s['F1'].std():.1f} Hz       | {s['F2'].mean():.1f} ± {s['F2'].std():.1f} Hz")
print("="*70)

# Gráfico Formántico F1 vs F2
plt.figure(figsize=(9, 7), facecolor='#0f172a')
ax = plt.gca()
ax.set_facecolor('#1e293b')

for v in ['A', 'E', 'I', 'O', 'U']:
    s = df_clean[df_clean['vocal'] == v]
    ax.scatter(s['F2'], s['F1'], color=colores_canonicos[v],
               label=f"/{v}/ (N={len(s)})", s=45, alpha=0.8, edgecolors='white', linewidths=0.5)
    c_f2 = s['F2'].median()
    c_f1 = s['F1'].median()
    ax.scatter(c_f2, c_f1, color=colores_canonicos[v], s=180, edgecolors='#ffffff', linewidths=2.0)
    ax.text(c_f2 + 30, c_f1 + 15, f"/{v}/", color='white', fontsize=14, fontweight='bold')

ax.invert_xaxis()
ax.invert_yaxis()

ax.set_title("Plano Acústico Formántico F1 vs F2 - Purga Isolation Forest (40 BPM)", color='white', fontsize=13, fontweight='bold', pad=15)
ax.set_xlabel("F2: Segundo Formante (Hz) - Adelantamiento Lingual (Invertido)", color='#cbd5e1', fontsize=11, labelpad=10)
ax.set_ylabel("F1: Primer Formante (Hz) - Altura Lingual / Cierre Mandibular (Invertido)", color='#cbd5e1', fontsize=11, labelpad=10)
ax.tick_params(colors='#94a3b8', labelsize=10)
ax.grid(True, linestyle='--', alpha=0.3, color='#475569')

leg = ax.legend(loc='lower left', facecolor='#0f172a', edgecolor='#334155', fontsize=10)
for text in leg.get_texts(): text.set_color('white')

plt.tight_layout()
png_out = 'reportes_experimentos/decodificacion_facial/paquete_correccion_facial_invariante/plano_formantes_f1_vs_f2.png'
plt.savefig(png_out, dpi=200)
plt.close()

csv_out = 'reportes_experimentos/decodificacion_facial/paquete_correccion_facial_invariante/formantes_acusticos_lpc_250pulsos.csv'
df_clean.to_csv(csv_out, index=False)
print(f"[OK] Archivos actualizados exitosamente.")
