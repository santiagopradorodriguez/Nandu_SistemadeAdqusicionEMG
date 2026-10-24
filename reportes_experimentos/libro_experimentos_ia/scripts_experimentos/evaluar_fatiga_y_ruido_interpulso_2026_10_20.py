# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Análisis de estabilidad temporal, piso de ruido inter-pulso SIN NOTCH
#              y fatiga muscular a lo largo de las sesiones del 2026-10-20.
# ==============================================================================

import os
import sys
import json
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Semilla fija universal
np.random.seed(42)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
base_dir = os.path.join(emg_desarrollo, "base_de_datos_electrodos", "2026-10-20")
out_fig_dir = os.path.join(emg_desarrollo, "resultados", "resultados_autoencoder", "figuras_evaluacion")
os.makedirs(out_fig_dir, exist_ok=True)

COLORES_CANALES = {
    0: '#E63946', # Ch0: Orbicular (Rojo)
    1: '#1F77B4', # Ch1: Modíolo (Azul)
    2: '#2CA02C'  # Ch2: Masetero (Verde)
}
NOMBRES_CANALES = {
    0: 'Ch0: Orbicular',
    1: 'Ch1: Modíolo',
    2: 'Ch2: Masetero'
}

sesiones = [
    {
        "id": "Candela_Exp1",
        "nombre": "Candela - Exp 1: Masetero 2.4 mm",
        "patron": "_Exp1_"
    },
    {
        "id": "Candela_Exp2",
        "nombre": "Candela - Exp 2: Masetero 23 mm",
        "patron": "_Exp2_"
    },
    {
        "id": "Santi_Exploracion",
        "nombre": "Santiago - Exploración: Auto-montaje",
        "patron": "_Santi"
    }
]

# Filtro Pasabanda 20 - 500 Hz SIN FILTRO NOTCH (para auditar ruido de línea de 50 Hz y despegue)
fs = 2000
nyq = 0.5 * fs
b_band, a_band = butter(4, [20.0 / nyq, 500.0 / nyq], btype='bandpass')
ganancia = 495.0 # AD620 RG = 100 Ohm
calib_factor = (1.0 / ganancia) * 1e6 # a microvoltios

metricas_salida = {}

fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=False)
plt.subplots_adjust(hspace=0.38, wspace=0.18)

for row_idx, ses in enumerate(sesiones):
    s_id = ses["id"]
    s_nom = ses["nombre"]
    pat = ses["patron"]
    
    tomas = [d for d in os.listdir(base_dir) if pat in d]
    tomas_meta = []
    for t in tomas:
        # Excluir tomas fallidas de Santiago
        if s_id == "Santi_Exploracion" and any(k in t.lower() for k in ["a_prueba1_santi", "a_prueba2_santi"]):
            continue
        m_p = os.path.join(base_dir, t, "canal_0", "metadata.json")
        ts = 0
        mdate = ""
        if os.path.exists(m_p):
            with open(m_p) as f:
                meta = json.load(f)
                ts = meta.get("timestamp", 0)
                mdate = meta.get("measurement_date", "")
        tomas_meta.append((t, ts, mdate))
    tomas_meta.sort(key=lambda x: (x[1], x[2], x[0]))
    
    picos_dict = {0: [], 1: [], 2: []}
    ruido_no_notch_dict = {0: [], 1: [], 2: []}
    
    # Mapeo canónico a carpetas de canales:
    # Para Candela: 0->canal_0 (orbi), 1->canal_1 (modiolo), 2->canal_2 (masetero)
    # Para Santi (hardware [mast, orbi, nudo]): permutación [1, 2, 0] -> 0->canal_1 (orbi), 1->canal_2 (nudo), 2->canal_0 (mast)
    ch_map = {0: 1, 1: 2, 2: 0} if s_id == "Santi_Exploracion" else {0: 0, 1: 1, 2: 2}
    
    for t, _, _ in tomas_meta:
        for c_idx in [0, 1, 2]:
            ch_folder = f"canal_{ch_map[c_idx]}"
            ch_p = os.path.join(base_dir, t, ch_folder, "analisis_results.json")
            wav_p = os.path.join(base_dir, t, ch_folder, "grabacion.wav")
            if not os.path.exists(ch_p) or not os.path.exists(wav_p):
                continue
            with open(ch_p) as f:
                d = json.load(f)
            v_idx = [x for x in d.get("valid_indices", []) if x is not None]
            if len(v_idx) < 2:
                continue
                
            sig, sr = sf.read(wav_p)
            # Filtrado 20-500 Hz SIN NOTCH
            sig_no_notch = filtfilt(b_band, a_band, sig)
            
            # Envolvente RMS (ventana 75 ms) SIN NOTCH
            w_len = int(0.075 * sr)
            w = np.ones(w_len) / w_len
            env_no_notch = np.sqrt(np.convolve(sig_no_notch**2, w, mode='same')) * calib_factor
            
            for i in range(len(v_idx)):
                # Amplitud de pico del pulso fonatorio
                p_c = v_idx[i]
                p_start = max(0, p_c - int(0.3 * sr))
                p_end = min(len(env_no_notch), p_c + int(0.3 * sr))
                peak_val = np.max(env_no_notch[p_start:p_end])
                picos_dict[c_idx].append(peak_val)
                
                # Ruido interpulso en el valle entre picos SIN NOTCH
                if i < len(v_idx) - 1:
                    mid = (v_idx[i] + v_idx[i+1]) // 2
                else:
                    mid = min(len(env_no_notch) - int(0.25*sr), v_idx[i] + int(0.8*sr))
                w_start = max(0, mid - int(0.25 * sr))
                w_end = min(len(env_no_notch), mid + int(0.25 * sr))
                valle = env_no_notch[w_start:w_end]
                
                # Depuración IQR local de espigas espurias
                q1, q3 = np.percentile(valle, [25, 75])
                iqr = q3 - q1
                valid_v = valle[valle <= q3 + 1.5 * iqr]
                ruido_val = np.mean(valid_v) if len(valid_v) > 0 else np.mean(valle)
                ruido_no_notch_dict[c_idx].append(ruido_val)
                
    n_pulsos = min(len(picos_dict[0]), len(picos_dict[1]), len(picos_dict[2]))
    x_axis = np.arange(1, n_pulsos + 1)
    
    ax_amp = axes[row_idx, 0]
    ax_noise = axes[row_idx, 1]
    
    metricas_salida[s_id] = {
        "nombre": s_nom,
        "n_tomas": len(tomas_meta),
        "n_pulsos_total": n_pulsos,
        "canales": {}
    }
    
    for c_idx in [0, 1, 2]:
        p = np.array(picos_dict[c_idx][:n_pulsos])
        r = np.array(ruido_no_notch_dict[c_idx][:n_pulsos])
        col = COLORES_CANALES[c_idx]
        ch_label = NOMBRES_CANALES[c_idx]
        
        # Ajustes lineales
        m_p, b_p = np.polyfit(x_axis, p, 1)
        m_r, b_r = np.polyfit(x_axis, r, 1)
        
        drift_p_pct = (m_p * n_pulsos) / b_p * 100.0 if b_p != 0 else 0.0
        drift_r_pct = (m_r * n_pulsos) / b_r * 100.0 if b_r != 0 else 0.0
        
        metricas_salida[s_id]["canales"][f"canal_{c_idx}"] = {
            "nombre": ch_label,
            "amp_media_uv": float(np.mean(p)),
            "amp_std_uv": float(np.std(p)),
            "amp_inicio_uv": float(b_p),
            "amp_fin_uv": float(b_p + m_p * n_pulsos),
            "amp_drift_pct": float(drift_p_pct),
            "ruido_no_notch_media_uv": float(np.mean(r)),
            "ruido_no_notch_std_uv": float(np.std(r)),
            "ruido_no_notch_inicio_uv": float(b_r),
            "ruido_no_notch_fin_uv": float(b_r + m_r * n_pulsos),
            "ruido_no_notch_drift_pct": float(drift_r_pct),
            "snr_medio": float(np.mean(p) / (np.mean(r) + 1e-6))
        }
        
        # Curva suave
        w_size = 9
        p_smooth = np.convolve(p, np.ones(w_size)/w_size, mode='same')
        r_smooth = np.convolve(r, np.ones(w_size)/w_size, mode='same')
        
        # Plot Amplitud (Fatiga)
        ax_amp.plot(x_axis, p, color=col, alpha=0.20, linewidth=0.8)
        ax_amp.plot(x_axis, p_smooth, color=col, linewidth=2.0, label=f"{ch_label} ({drift_p_pct:+.1f}%)")
        ax_amp.plot(x_axis, m_p * x_axis + b_p, color=col, linestyle="--", linewidth=1.2, alpha=0.8)
        
        # Plot Ruido Interpulso SIN NOTCH (Estabilidad y Despegue / 50 Hz)
        ax_noise.plot(x_axis, r, color=col, alpha=0.20, linewidth=0.8)
        ax_noise.plot(x_axis, r_smooth, color=col, linewidth=2.0, label=f"{ch_label} ({drift_r_pct:+.1f}%)")
        ax_noise.plot(x_axis, m_r * x_axis + b_r, color=col, linestyle="--", linewidth=1.2, alpha=0.8)
        
    # Estética ax_amp
    ax_amp.set_title(f"{s_nom} - Amplitud de Picos Fonatorios: Diagnóstico de Fatiga", fontsize=11, fontweight="bold")
    ax_amp.set_ylabel("Amplitud [µV]", fontsize=10)
    ax_amp.set_xlabel("Índice de Pulso Consecutivo en Sesión", fontsize=10)
    ax_amp.grid(True, linestyle=":", alpha=0.6)
    ax_amp.set_xlim(1, n_pulsos)
    ax_amp.legend(loc="upper right", fontsize=8, framealpha=0.9)
    
    # Estética ax_noise
    ax_noise.set_title(f"{s_nom} - Piso de Ruido Inter-pulso SIN NOTCH: Despegue y 50 Hz", fontsize=11, fontweight="bold")
    ax_noise.set_ylabel("Ruido Inter-pulso [µV]", fontsize=10)
    ax_noise.set_xlabel("Índice de Pulso Consecutivo en Sesión", fontsize=10)
    ax_noise.grid(True, linestyle=":", alpha=0.6)
    ax_noise.set_xlim(1, n_pulsos)
    ax_noise.legend(loc="upper right", fontsize=8, framealpha=0.9)
    
fig_out = os.path.join(out_fig_dir, "analisis_fatiga_ruido_interpulso_2026_10_20.png")
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close(fig)

json_out = os.path.join(out_fig_dir, "metricas_fatiga_ruido_interpulso_2026_10_20.json")
with open(json_out, "w") as f:
    json.dump(metricas_salida, f, indent=4)

print(f"Figura SIN NOTCH guardada en: {fig_out}")
print(f"Métricas SIN NOTCH guardadas en: {json_out}")
