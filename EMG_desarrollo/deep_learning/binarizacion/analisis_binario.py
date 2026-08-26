# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Pipeline de normalizacion y binarizacion de ventanas EMG
#              multicanal para alimentar una red neuronal.
#
# Flujo:
#   1. Cargar WAVs multicanal de una medicion (SecuenciaContinua o individual)
#   2. Filtrar, calcular envolvente y segmentar en ventanas (metronomicas)
#   3. Baseline Tracking: restar ruido interpulso local a cada ventana
#   4. Calcular energia MAV por canal por ventana
#   5. Normalizar por Esfuerzo Relativo Espacial (M1%, M2%, M3%)
#   6. Binarizar con umbral porcentual comun
#   7. GUI Interactiva para ajustar el threshold visualmente y exportar
# ==============================================================================

import os
import sys
import json
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt, iirnotch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

script_dir_abs = os.path.dirname(os.path.abspath(__file__))
emg_root = os.path.dirname(os.path.dirname(script_dir_abs))
if emg_root not in sys.path:
    sys.path.insert(0, emg_root)
analysis_dir = os.path.join(emg_root, "analysis")
if analysis_dir not in sys.path:
    sys.path.insert(0, analysis_dir)

# --- Reutilizar funciones del motor de analisis principal ---
try:
    from analysis.analisis_por_track_integrado import (
        _read_wav_mono,
        _compute_env_full,
        _estimate_noise_window,
        _detect_maxima_and_extract,
    )
except ImportError:
    from analisis_por_track_integrado import (
        _read_wav_mono,
        _compute_env_full,
        _estimate_noise_window,
        _detect_maxima_and_extract,
    )

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
})

# =====================================================================
# 1. Correccion de Linea Base (Baseline Tracking Dinamico)
# =====================================================================
def corregir_baseline_por_ventana(env_recortada, maxima_per_cut, muestras_pulso, samplerate):
    ventanas_corregidas = []
    baselines = []
    noise_segments_info = []

    noise_win_samples = max(3, int(round((muestras_pulso / 4.0))))

    for idx in range(len(maxima_per_cut)):
        max_idx = maxima_per_cut[idx]

        pre_samples = muestras_pulso // 2
        post_samples = muestras_pulso - pre_samples
        win_start = max(0, max_idx - pre_samples)
        win_end = min(len(env_recortada), max_idx + post_samples)

        ventana = env_recortada[win_start:win_end].copy()

        if idx == 0:
            midpoint = max_idx - (muestras_pulso // 2)
        else:
            prev_max = maxima_per_cut[idx - 1]
            midpoint = (prev_max + max_idx) // 2

        noise_start = max(0, int(midpoint - noise_win_samples // 2))
        noise_end = min(len(env_recortada), noise_start + noise_win_samples)

        if noise_end - noise_start >= 3:
            noise_seg = env_recortada[noise_start:noise_end]

            q1 = np.percentile(noise_seg, 25)
            q3 = np.percentile(noise_seg, 75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            valid_noise = noise_seg[noise_seg <= upper_bound]
            if len(valid_noise) < 3:
                valid_noise = noise_seg

            baseline = np.mean(valid_noise)
        else:
            baseline = 0.0

        ventana_corregida = ventana - baseline
        ventana_corregida = np.clip(ventana_corregida, 0, None) 

        ventanas_corregidas.append(ventana_corregida)
        baselines.append(baseline)
        noise_segments_info.append({
            'idx_pulso': idx,
            'noise_start': noise_start,
            'noise_end': noise_end,
            'baseline': baseline,
        })

    return ventanas_corregidas, baselines, noise_segments_info

def calcular_mav(ventana):
    return np.mean(np.abs(ventana))

def normalizar_esfuerzo_relativo(mav_canales, umbral_silencio=1e-6):
    mav_arr = np.array(mav_canales, dtype=float)
    e_total = np.sum(mav_arr)

    if e_total < umbral_silencio:
        return None, e_total, True

    porcentajes = mav_arr / e_total
    return porcentajes, e_total, False

def binarizar(porcentajes, umbral_pct=0.15):
    if porcentajes is None:
        return None
    return (porcentajes > umbral_pct).astype(int)

# =====================================================================
# Interfaz Grafica Interactiva (Matplotlib)
# =====================================================================
def _print_vectores(df, umbral_pct):
    """
    Imprime el vector binario [ch0, ch1, ch2] por ventana y el promedio.
    """
    canales = sorted(df['canal'].unique())
    ventanas = sorted(df['ventana'].unique())
    
    print(f"\n{'='*60}")
    print(f"  VECTORES BINARIOS (Threshold: {umbral_pct*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  {'Ventana':<10} {'Label':<8} Vector [{', '.join(canales)}]")
    print(f"  {'-'*50}")
    
    all_vectors = []
    
    for v in ventanas:
        df_v = df[df['ventana'] == v].sort_values('canal')
        label = df_v['label'].iloc[0]
        binarios = df_v['binario'].values.astype(int)
        vec_str = str(list(binarios))
        all_vectors.append(binarios)
        print(f"  {v:<10} {label:<8} {vec_str}")
    
    # Promedio
    if all_vectors:
        mat = np.array(all_vectors, dtype=float)
        promedio = np.mean(mat, axis=0)
        promedio_bin = (promedio > 0.5).astype(int)
        
        print(f"  {'-'*50}")
        print(f"  {'PROMEDIO':<10} {'---':<8} {list(np.round(promedio, 3))}")
        print(f"  {'BINARIZADO':<10} {'---':<8} {list(promedio_bin)}")
        print(f"{'='*60}")


def graficar_resultados_interactivo(df, umbral_pct_inicial, carpeta_salida):
    """
    Grafica el esfuerzo relativo (MAV%) y permite ajustar el umbral en vivo.
    """
    canales = sorted(df['canal'].unique())
    n_canales = len(canales)
    
    if n_canales == 0:
        return
    
    # Imprimir vectores iniciales
    _print_vectores(df, umbral_pct_inicial)
        
    fig, axes = plt.subplots(n_canales, 1, figsize=(14, 3 * n_canales + 1.5), sharex=True)
    plt.subplots_adjust(bottom=0.25)
    
    if n_canales == 1:
        axes = [axes]
        
    umbral_val = umbral_pct_inicial * 100.0
    hline_list = []
    
    def draw_fills(ax, x, y, umbral):
        for c in list(ax.collections):
            c.remove()
        ax.fill_between(x, umbral, y, where=(y > umbral), interpolate=True, color='lime', alpha=0.4, zorder=1)
        ax.fill_between(x, 0, y, where=(y <= umbral), interpolate=True, color='red', alpha=0.2, zorder=1)

    for i, ch in enumerate(canales):
        ax = axes[i]
        df_ch = df[df['canal'] == ch]
        
        x = df_ch['ventana']
        y = df_ch['mav_pct'] * 100.0
        
        ax.plot(x, y, marker='o', linestyle='-', color='white', linewidth=2, zorder=3)
        ax.plot(x, y, marker='', linestyle='-', color='black', linewidth=4, zorder=2)
        
        hline = ax.axhline(umbral_val, color='cyan', linestyle='--', linewidth=2)
        hline_list.append(hline)
        
        draw_fills(ax, x, y, umbral_val)
        
        ax.set_title(f'Musculo: {ch}', fontweight='bold')
        ax.set_ylabel('Esfuerzo Relativo (%)')
        ax.set_ylim(0, max(100, y.max() + 10 if not y.empty else 100))
        
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#121212')
        ax.tick_params(colors='white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.grid(True, linestyle=':', alpha=0.3, color='white')
        
        if i == n_canales - 1:
            ax.set_xticks(x)
            labels = df_ch['label'].tolist()
            ax.set_xticklabels(labels, rotation=45, ha='right', color='white')
            ax.set_xlabel('Ventana (Palabra / Etiqueta)', color='white', fontweight='bold')

    # --- Slider ---
    ax_umbral = plt.axes([0.15, 0.1, 0.60, 0.03], facecolor='#333333')
    s_umbral = Slider(ax_umbral, 'Threshold %', 0.1, 100.0, valinit=umbral_val, color='cyan')
    s_umbral.label.set_color('white')
    s_umbral.valtext.set_color('white')

    def update(val):
        umbral = s_umbral.val
        for i, ch in enumerate(canales):
            ax = axes[i]
            df_ch = df[df['canal'] == ch]
            x = df_ch['ventana']
            y = df_ch['mav_pct'] * 100.0
            
            hline_list[i].set_ydata([umbral, umbral])
            draw_fills(ax, x, y, umbral)
            
        fig.canvas.draw_idle()
        
    s_umbral.on_changed(update)

    # --- Boton Guardar ---
    ax_save = plt.axes([0.80, 0.08, 0.15, 0.05])
    btn_save = Button(ax_save, 'Guardar Datos', color='#0055FF', hovercolor='#0088FF')
    btn_save.label.set_color('white')
    btn_save.label.set_fontweight('bold')

    def save_csv(event):
        nuevo_umbral_pct = s_umbral.val / 100.0
        
        df['binario'] = (df['mav_pct'] > nuevo_umbral_pct).astype(int)
        
        out_path = os.path.join(carpeta_salida, "binarizacion.csv")
        df.to_csv(out_path, index=False)
        
        # Re-imprimir vectores con el nuevo threshold
        _print_vectores(df, nuevo_umbral_pct)
        print(f"  Ruta CSV: {out_path}")
        
    btn_save.on_clicked(save_csv)

    plt.show()

# =====================================================================
# Selector Interactivo de Sesiones
# =====================================================================
def seleccionar_sesion_interactiva():
    """
    Escanea base_de_datos_electrodos, muestra las fechas y mediciones
    y deja al usuario elegir por numero en la consola.
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "base_de_datos_electrodos")
    base_dir = os.path.normpath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"ERROR: No se encontro la carpeta base: {base_dir}")
        return None
    
    # --- Paso 1: Listar fechas ---
    fechas = sorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ], reverse=True)
    
    if not fechas:
        print("ERROR: No hay carpetas de fecha en base_de_datos_electrodos")
        return None
    
    print(f"\n{'='*60}")
    print("  SELECTOR DE SESION - Pipeline de Binarizacion")
    print(f"{'='*60}")
    print(f"  Directorio base: {base_dir}")
    print(f"\n  Fechas disponibles:")
    print(f"  {'-'*40}")
    for i, f in enumerate(fechas):
        # Contar mediciones dentro
        mediciones = [
            d for d in os.listdir(os.path.join(base_dir, f))
            if os.path.isdir(os.path.join(base_dir, f, d))
        ]
        print(f"  [{i+1:>2}] {f}  ({len(mediciones)} mediciones)")
    
    print(f"  {'-'*40}")
    try:
        sel_fecha = input("  >> Selecciona una fecha (numero): ").strip()
        idx_fecha = int(sel_fecha) - 1
        if idx_fecha < 0 or idx_fecha >= len(fechas):
            print("Seleccion invalida.")
            return None
    except (ValueError, EOFError):
        print("Cancelado.")
        return None
    
    fecha_elegida = fechas[idx_fecha]
    fecha_dir = os.path.join(base_dir, fecha_elegida)
    
    # --- Paso 2: Listar mediciones dentro de la fecha ---
    mediciones = sorted([
        d for d in os.listdir(fecha_dir)
        if os.path.isdir(os.path.join(fecha_dir, d))
    ])
    
    if not mediciones:
        print(f"ERROR: No hay mediciones en {fecha_dir}")
        return None
    
    print(f"\n  Mediciones del {fecha_elegida}:")
    print(f"  {'-'*50}")
    for i, m in enumerate(mediciones):
        # Intentar leer metadata para dar info util
        meta_path = os.path.join(fecha_dir, m, "canal_0", "metadata.json")
        info_extra = ""
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                sujeto = meta.get('sujeto', '?')
                bpm = meta.get('bpm', '?')
                pulsos = meta.get('pulse_count', '?')
                info_extra = f" | Suj: {sujeto} | BPM: {bpm} | Pulsos: {pulsos}"
            except:
                pass
        print(f"  [{i+1:>2}] {m}{info_extra}")
    
    print(f"  {'-'*50}")
    try:
        sel_med = input("  >> Selecciona una medicion (numero): ").strip()
        idx_med = int(sel_med) - 1
        if idx_med < 0 or idx_med >= len(mediciones):
            print("Seleccion invalida.")
            return None
    except (ValueError, EOFError):
        print("Cancelado.")
        return None
    
    carpeta = os.path.join(fecha_dir, mediciones[idx_med])
    print(f"\n  Seleccionado: {carpeta}")
    return carpeta


# =====================================================================
# 5. Pipeline Completo: Procesar una medicion multicanal
# =====================================================================
def procesar_medicion_binario(
    carpeta_medicion,
    bpm=40,
    noise_seconds=5.0,
    smooth_ms=50,
    tipo_envolvente="media_movil",
    highpass_cutoff_hz=20.0,
    lowpass_cutoff_hz=500.0,
    apply_notch_filter=True,
    umbral_binarizacion_pct=0.15,
    umbral_silencio_abs=1e-6,
    n_pulsos_manual=None,
    excluded_windows=None,
):
    canal_dirs = sorted([
        d for d in os.listdir(carpeta_medicion)
        if os.path.isdir(os.path.join(carpeta_medicion, d)) and d.startswith("canal_")
    ])

    # Ignorar canal_3 (microfono)
    canal_dirs = [d for d in canal_dirs if "canal_3" not in d]

    if not canal_dirs:
        print(f"ERROR: No se encontraron canales validos en {carpeta_medicion}")
        return None, None

    meta_path = os.path.join(carpeta_medicion, canal_dirs[0], "metadata.json")
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

    bpm = metadata.get('bpm', bpm)
    noise_seconds = metadata.get('noise_seconds', noise_seconds)
    n_pulsos_manual = metadata.get('pulse_count', n_pulsos_manual)
    samplerate_meta = metadata.get('sample_rate', None)
    valid_words = metadata.get('valid_words', metadata.get('words_sequence', None))

    periodo = 60.0 / bpm
    print(f"\n{'='*60}")
    print(f"PIPELINE DE BINARIZACION - {os.path.basename(carpeta_medicion)}")
    print(f"Canales detectados: {len(canal_dirs)} (ignorando canal_3)")
    print(f"BPM={bpm} | Periodo={periodo:.3f}s | Ruido={noise_seconds}s | Pulsos={n_pulsos_manual}")
    print(f"{'='*60}")

    all_envolventes = []
    all_maxima = []
    sr = None

    for canal_dir in canal_dirs:
        wav_path = os.path.join(carpeta_medicion, canal_dir, "grabacion.wav")
        if not os.path.exists(wav_path):
            print(f"  WARN: No se encontro {wav_path}, saltando.")
            all_envolventes.append(None)
            all_maxima.append(None)
            continue

        signal, samplerate = _read_wav_mono(wav_path)
        sr = samplerate
        print(f"  [{canal_dir}] {len(signal)} muestras @ {samplerate} Hz ({len(signal)/samplerate:.2f}s)")

        nyquist = 0.5 * samplerate

        if apply_notch_filter:
            b, a = iirnotch(50.0, 2.0, samplerate)
            signal = filtfilt(b, a, signal)

        if highpass_cutoff_hz > 0 and highpass_cutoff_hz < nyquist:
            b, a = butter(4, highpass_cutoff_hz / nyquist, btype='high')
            signal = filtfilt(b, a, signal)

        if lowpass_cutoff_hz > 0 and lowpass_cutoff_hz < nyquist:
            cutoff = min(lowpass_cutoff_hz, nyquist * 0.99)
            b, a = butter(4, cutoff / nyquist, btype='low')
            signal = filtfilt(b, a, signal)

        signal_abs = np.abs(signal)
        env_full = _compute_env_full(signal_abs, True, smooth_ms, samplerate, tipo_envolvente)

        t = np.linspace(0, len(signal) / samplerate, len(signal), endpoint=False)
        duracion_total = len(signal) / samplerate

        mask = (t >= 0) & (t <= duracion_total)
        env_recortada = env_full[mask]

        muestras_pulso = int(round(periodo * samplerate))

        start_sample_noise, env_noise, sigma_est, umbral, noise_rms = _estimate_noise_window(
            signal[mask], samplerate, noise_seconds, smooth_ms, 6.0, tipo_envolvente
        )
        if start_sample_noise <= 0:
            start_sample_noise = 0

        env_for_cuts = env_recortada[start_sample_noise:]
        n_pulsos_auto = len(env_for_cuts) // muestras_pulso if muestras_pulso > 0 else 0

        search_threshold = umbral if umbral is not None and umbral > 0 else 0.25

        pre_w = 0.4 * periodo
        post_w = 0.6 * periodo
        pre_samples = int(round(pre_w * samplerate))
        post_samples = int(round(post_w * samplerate))

        maxima_per_cut, segmentos = _detect_maxima_and_extract(
            np.abs(env_recortada), start_sample_noise, muestras_pulso,
            pre_samples, post_samples, search_threshold,
            n_pulsos_manual=n_pulsos_manual, excluded_windows=excluded_windows
        )

        all_envolventes.append(env_recortada)
        all_maxima.append(maxima_per_cut)

    if sr is None:
        print("ERROR: No se pudo leer ningun archivo WAV.")
        return None, None

    n_canales = len(canal_dirs)
    n_ventanas_por_canal = [len(m) if m is not None else 0 for m in all_maxima]
    n_ventanas = min(n_ventanas_por_canal) if n_ventanas_por_canal else 0

    if n_ventanas == 0:
        print("ERROR: No se detectaron ventanas en ningun canal.")
        return None, None

    print(f"\n  Ventanas a procesar: {n_ventanas} (minimo entre canales)")

    muestras_pulso = int(round(periodo * sr))
    filas = []

    for v in range(n_ventanas):
        mav_por_canal = []
        baselines_por_canal = []

        for ch in range(n_canales):
            if all_envolventes[ch] is None or all_maxima[ch] is None:
                mav_por_canal.append(0.0)
                baselines_por_canal.append(0.0)
                continue

            env = all_envolventes[ch]
            maxima = all_maxima[ch]

            ventanas_corr, baselines, _ = corregir_baseline_por_ventana(
                env, maxima, muestras_pulso, sr
            )

            if v < len(ventanas_corr):
                mav = calcular_mav(ventanas_corr[v])
                bl = baselines[v]
            else:
                mav = 0.0
                bl = 0.0

            mav_por_canal.append(mav)
            baselines_por_canal.append(bl)

        porcentajes, e_total, es_silencio = normalizar_esfuerzo_relativo(
            mav_por_canal, umbral_silencio=umbral_silencio_abs
        )

        binario = binarizar(porcentajes, umbral_pct=umbral_binarizacion_pct)

        if valid_words and v < len(valid_words):
            label = valid_words[v]
        else:
            label = f"V{v+1}"

        for ch in range(n_canales):
            filas.append({
                'ventana': v + 1,
                'canal': canal_dirs[ch],
                'mav': mav_por_canal[ch],
                'mav_pct': porcentajes[ch] if porcentajes is not None else np.nan,
                'binario': binario[ch] if binario is not None else np.nan,
                'baseline': baselines_por_canal[ch],
                'e_total': e_total,
                'es_silencio': es_silencio,
                'label': label,
            })

    df = pd.DataFrame(filas)

    # Iniciar GUI Interactiva
    print("\n  >>> Abriendo ventana de graficos interactivos...")
    graficar_resultados_interactivo(df, umbral_binarizacion_pct, carpeta_medicion)

    return df, metadata

# =====================================================================
# CLI
# =====================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de Binarizacion EMG - Proyecto Nandu LSD")
    parser.add_argument("carpeta", nargs='?', default=None, help="Ruta a la carpeta de medicion (opcional, si no se pasa se abre selector)")
    parser.add_argument("--bpm", type=int, default=40)
    parser.add_argument("--noise-seconds", type=float, default=5.0)
    parser.add_argument("--smooth-ms", type=float, default=50.0)
    parser.add_argument("--tipo-env", type=str, default="media_movil", choices=["media_movil", "rms"])
    parser.add_argument("--umbral-pct", type=float, default=0.15, help="Umbral de binarizacion (0-1)")
    parser.add_argument("--umbral-silencio", type=float, default=1e-6, help="Energia minima para no descartar ventana")
    parser.add_argument("--hp", type=float, default=20.0, help="Filtro pasa-altos Hz")
    parser.add_argument("--lp", type=float, default=500.0, help="Filtro pasa-bajos Hz")
    parser.add_argument("--no-notch", action="store_true", help="Desactivar filtro notch 50Hz")

    args = parser.parse_args()

    carpeta_medicion = args.carpeta
    
    # Si no se provee carpeta, abrir el selector interactivo por consola
    if carpeta_medicion is None:
        carpeta_medicion = seleccionar_sesion_interactiva()
        if carpeta_medicion is None:
            print("Operacion cancelada.")
            sys.exit(0)

    procesar_medicion_binario(
        carpeta_medicion=carpeta_medicion,
        bpm=args.bpm,
        noise_seconds=args.noise_seconds,
        smooth_ms=args.smooth_ms,
        tipo_envolvente=args.tipo_env,
        highpass_cutoff_hz=args.hp,
        lowpass_cutoff_hz=args.lp,
        apply_notch_filter=not args.no_notch,
        umbral_binarizacion_pct=args.umbral_pct,
        umbral_silencio_abs=args.umbral_silencio,
    )

if __name__ == "__main__":
    main()
