# ==============================================================================
# Script de Comparativa: Meditrace vs Electrodos Nuevos (Más Caros)
# Calcula estadísticas agregadas sobre la totalidad de mediciones de /o/ y /u/
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

REPO_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
BASE_DIR = os.path.join(REPO_ROOT, "EMG_desarrollo", "base_de_datos_electrodos", "2026-09-29")
OUT_IMG = os.path.join(REPO_ROOT, "reportes_experimentos", "presentacion", "imagenes", "comparativa_electrodos_meditrace_vs_nuevos.png")

GAIN = 495.0  # Ganancia AD620 fija
FS = 2000.0   # 2000 Hz

def aplicar_filtros(sig, fs=2000.0):
    b_high, a_high = butter(2, 10.0 / (fs / 2.0), btype='highpass')
    sig_f = filtfilt(b_high, a_high, sig)
    b_low, a_low = butter(2, 450.0 / (fs / 2.0), btype='lowpass')
    sig_f = filtfilt(b_low, a_low, sig_f)
    b_notch, a_notch = butter(2, [48.0 / (fs / 2.0), 52.0 / (fs / 2.0)], btype='bandstop')
    sig_f = filtfilt(b_notch, a_notch, sig_f)
    return sig_f

def procesar_toma(carpeta):
    p = os.path.join(BASE_DIR, carpeta, "grabacion.csv")
    df = pd.read_csv(p)
    t = df["Tiempo (s)"].values
    raw_v = df["Canal 0"].values
    uV = (raw_v / GAIN) * 1e6
    uV_filt = aplicar_filtros(uV, FS)
    
    # Envolvente RMS (50 ms)
    win = int(0.05 * FS)
    env = np.sqrt(np.convolve(uV_filt**2, np.ones(win)/win, mode='same'))
    
    # Ruido en reposo inicial [0.5, 5.0]s
    mask_ruido = (t >= 0.5) & (t <= 5.0)
    ruido_rms = np.std(uV_filt[mask_ruido])
    
    # Encontrar los 5 pulsos en [6.0, 15.0]s
    mask_pulsos = (t >= 6.0) & (t <= 15.0)
    t_p = t[mask_pulsos]
    uV_p = uV_filt[mask_pulsos]
    env_p = env[mask_pulsos]
    
    pico_max = np.max(np.abs(uV_p))
    pulsos_rms = np.sqrt(np.mean(uV_p**2))
    
    # SNR global de la sesión
    snr_db = 20 * np.log10(pulsos_rms / (ruido_rms + 1e-9))
    snr_lineal = pico_max / (ruido_rms + 1e-9)
    
    return {
        "carpeta": carpeta,
        "t": t,
        "uV": uV_filt,
        "env": env,
        "ruido_rms": ruido_rms,
        "pico_max": pico_max,
        "pulsos_rms": pulsos_rms,
        "snr_db": snr_db,
        "snr_lineal": snr_lineal
    }

# 1. Procesar todas las tomas de Meditrace (viejos)
tomas_meditrace = [
    "o_Pruebaelectrodosviejos_Sujeto1",
    "o_Pruebaelectrodosviejos_2_Sujeto1",
    "u_Pruebaelectrodosviejos_Sujeto1",
    "u_Pruebaelectrodosviejos_2_Sujeto1"
]
res_meditrace = [procesar_toma(c) for c in tomas_meditrace]

# 2. Procesar todas las tomas de Electrodos Nuevos (más caros)
tomas_nuevos = [
    "o_Pruebaelectrodosnuevos_Sujeto1",
    "o_Pruebaelectrodosnuevos_2_Sujeto1",
    "u_Pruebaelectrodosnuevos_Sujeto1",
    "u_Pruebaelectrodosnuevos_2_Sujeto1"
]
res_nuevos = [procesar_toma(c) for c in tomas_nuevos]

# 3. Estadísticas globales agregadas
def resumen_estadistico(res_lista, nombre):
    ruidos = [r["ruido_rms"] for r in res_lista]
    picos  = [r["pico_max"] for r in res_lista]
    snrs_db = [r["snr_db"] for r in res_lista]
    snrs_lin = [r["snr_lineal"] for r in res_lista]
    
    print(f"\n=== DETALLE DE TOMAS: {nombre} ===")
    for r in res_lista:
        print(f"  {r['carpeta'][:35]:35s} -> Ruido: {r['ruido_rms']:.2f} uV | Pico: {r['pico_max']:.1f} uV | SNR: {r['snr_db']:.2f} dB (x{r['snr_lineal']:.1f})")
        
    print(f"\n--- RESUMEN GLOBAL PROMEDIO ({nombre}) ---")
    print(f"  Ruido Base RMS: {np.mean(ruidos):.2f} +/- {np.std(ruidos):.2f} uV (min: {np.min(ruidos):.2f}, max: {np.max(ruidos):.2f})")
    print(f"  Pico Fonación:  {np.mean(picos):.1f} +/- {np.std(picos):.1f} uV (min: {np.min(picos):.1f}, max: {np.max(picos):.1f})")
    print(f"  SNR Promedio:   {np.mean(snrs_db):.2f} +/- {np.std(snrs_db):.2f} dB (min: {np.min(snrs_db):.2f}, max: {np.max(snrs_db):.2f})")
    print(f"  SNR Lineal:     x{np.mean(snrs_lin):.1f} +/- {np.std(snrs_lin):.1f}")
    
    return {
        "ruido_mean": np.mean(ruidos), "ruido_std": np.std(ruidos),
        "pico_mean": np.mean(picos), "pico_std": np.std(picos),
        "snr_mean": np.mean(snrs_db), "snr_std": np.std(snrs_db),
        "snr_lin_mean": np.mean(snrs_lin), "snr_lin_std": np.std(snrs_lin),
    }

stats_meditrace = resumen_estadistico(res_meditrace, "MEDITRACE (ESTÁNDAR)")
stats_nuevos    = resumen_estadistico(res_nuevos, "ELECTRODOS NUEVOS (PREMIUM)")

# 4. Generar la figura comparativa para la presentación
plt.style.use('dark_background')
fig, axs = plt.subplots(2, 1, figsize=(11.5, 5.8), sharex=True, sharey=True, dpi=300)
fig.patch.set_facecolor('#0B0F19')

c_meditrace_sig = '#38BDF8'
c_meditrace_env = '#F59E0B'
c_nuevos_sig    = '#10B981'
c_nuevos_env    = '#EC4899'

# Tomas representativas para la visualización (toma 1 de /o/)
rep_meditrace = res_meditrace[0]
rep_nuevos    = res_nuevos[0]

# Panel 1: Meditrace
ax1 = axs[0]
ax1.set_facecolor('#111827')
ax1.plot(rep_meditrace['t'], rep_meditrace['uV'], color=c_meditrace_sig, lw=0.6, alpha=0.85, label='EMG Crudo Filtrado')
ax1.plot(rep_meditrace['t'], rep_meditrace['env'], color=c_meditrace_env, lw=1.6, label='Envolvente RMS (50 ms)')
ax1.plot(rep_meditrace['t'], -rep_meditrace['env'], color=c_meditrace_env, lw=1.6)
ax1.axvspan(0.5, 5.0, color='#64748B', alpha=0.18, label='Reposo Base')

title_med = (f"Electrodos Meditrace - Promedio 4 Sesiones /o/ y /u/: "
             f"Ruido {stats_meditrace['ruido_mean']:.1f}±{stats_meditrace['ruido_std']:.1f} μV | "
             f"Pico {stats_meditrace['pico_mean']:.0f}±{stats_meditrace['pico_std']:.0f} μV | "
             f"SNR {stats_meditrace['snr_mean']:.1f}±{stats_meditrace['snr_std']:.1f} dB")
ax1.set_title(title_med, fontsize=10.0, fontweight='bold', color='#E2E8F0', pad=8)
ax1.set_ylabel("Amplitud [μV]", fontsize=9, color='#CBD5E1')
ax1.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', fontsize=8, facecolor='#1E293B', edgecolor='#475569')

# Panel 2: Electrodos Nuevos
ax2 = axs[1]
ax2.set_facecolor('#111827')
ax2.plot(rep_nuevos['t'], rep_nuevos['uV'], color=c_nuevos_sig, lw=0.6, alpha=0.85, label='EMG Crudo Filtrado')
ax2.plot(rep_nuevos['t'], rep_nuevos['env'], color=c_nuevos_env, lw=1.6, label='Envolvente RMS [50 ms]')
ax2.plot(rep_nuevos['t'], -rep_nuevos['env'], color=c_nuevos_env, lw=1.6)
ax2.axvspan(0.5, 5.0, color='#64748B', alpha=0.18, label='Reposo Base')

title_nuev = (f"Electrodos Premium Nuevos - Promedio 4 Sesiones /o/ y /u/: "
              f"Ruido {stats_nuevos['ruido_mean']:.1f}±{stats_nuevos['ruido_std']:.1f} μV | "
              f"Pico {stats_nuevos['pico_mean']:.0f}±{stats_nuevos['pico_std']:.0f} μV | "
              f"SNR {stats_nuevos['snr_mean']:.1f}±{stats_nuevos['snr_std']:.1f} dB")
ax2.set_title(title_nuev, fontsize=10.0, fontweight='bold', color='#E2E8F0', pad=8)
ax2.set_xlabel("Tiempo [s]", fontsize=9.5, color='#CBD5E1')
ax2.set_ylabel("Amplitud [μV]", fontsize=9, color='#CBD5E1')
ax2.grid(True, color='#334155', linestyle='--', alpha=0.5)
ax2.legend(loc='upper right', fontsize=8, facecolor='#1E293B', edgecolor='#475569')

ax1.set_ylim(-700, 700)
ax2.set_xlim(0, 15.1)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_IMG), exist_ok=True)
plt.savefig(OUT_IMG, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()
print(f"\nFigura guardada exitosamente en: {OUT_IMG}")
