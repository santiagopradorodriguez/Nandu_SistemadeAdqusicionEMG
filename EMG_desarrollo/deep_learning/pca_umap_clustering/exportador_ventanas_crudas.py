# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG
# Autores: Lucas Braunstein y Santiago Prado (adaptación: agente)
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Exporta las VENTANAS CRUDAS de todas las vocales de una fecha,
#              usando la MISMA cadena del generador PCA/UMAP:
#                metadata del metrónomo (bpm / noise_seconds / pulse_count)
#                -> analisis_trevisan (notch + highpass + lowpass, envolvente RMS,
#                   recorte por pulsos, ruido interpulso)
#                -> alineación por "Pico Volumen Micrófono" (canal_3)
#                -> ventanas [pre 40% / post 60% del período]
#              PERO exporta la señal CRUDA (µV) de cada ventana, SIN ENVELOPTE.
#              Se generan dos resoluciones (150 y 600 puntos/canal) + README.
# Uso: venv/bin/python exportador_ventanas_crudas.py [fecha, default 2026-10-27]
# ==============================================================================
import os
import sys
import json
import io
import shutil
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, iirnotch, butter, filtfilt, sosfiltfilt

script_dir_abs = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir_abs, "..", "binarizacion"))
import analisis_trevisan as at

# ============================ CONFIGURACIÓN ============================
BASE_DIR = os.path.abspath(os.path.join(script_dir_abs, "..", "..", "base_de_datos_electrodos"))
FECHA = sys.argv[1] if len(sys.argv) > 1 else "2026-10-27"
OUT_DIR = os.path.join(script_dir_abs, "ventanas_crudas", FECHA)

CANALES_FEATURES = ["canal_0", "canal_1", "canal_2"]
CANAL_ALINEACION = "canal_3"
PRE_PCT, POST_PCT = 0.4, 0.6
ALPHA_RUIDO = 1.0
SMOOTH_MS = 125
NOTCH_Q = 2.0
TIPO_FILTRO_RUIDO = "notch"
HIGHPASS_HZ = 20.0
LOWPASS_HZ = 300.0
RESOLUCIONES = [150, 600]   # puntos por canal para el CSV

# ============================ HELPERS (idem generador) ============================

def get_interpulse_noise(processed_segment, initial_noise):
    """Copia exacta de generador_pca_umap.py"""
    if len(processed_segment) < 10:
        return initial_noise
    abs_noise = np.abs(processed_segment)
    q1 = np.percentile(abs_noise, 25)
    q3 = np.percentile(abs_noise, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    valid_noise = abs_noise[abs_noise <= upper_bound]
    if len(valid_noise) < 3:
        valid_noise = abs_noise
    curr_mean = np.mean(valid_noise)
    if initial_noise > 0 and (curr_mean / initial_noise) > 5.0:
        return initial_noise
    return curr_mean


def cargar_senal_cruda(carpeta_canal):
    """Réplica de la cadena de carga/filtrado de analisis_trevisan.procesar_wavs_promedio
    (calibración por CSV padre -> µV -> notch 50Hz -> highpass -> lowpass).
    Devuelve (señal_cruda_µV, samplerate)."""
    archivos = [f for f in os.listdir(carpeta_canal) if f.lower().endswith(".wav")]
    if not archivos:
        return None, None
    filepath = os.path.join(carpeta_canal, archivos[0])

    # --- Calibración con el CSV del padre (idem analisis_trevisan) ---
    calibration_factor = 1.0
    try:
        parent_dir = os.path.dirname(carpeta_canal)
        csv_files = [f for f in os.listdir(parent_dir) if f.lower().endswith(".csv")]
        if "grabacion.csv" in csv_files:
            csv_path = os.path.join(parent_dir, "grabacion.csv")
        else:
            senales = [f for f in csv_files if not f.lower().startswith(("calibracion", "estadisticas", "features", "resumen"))]
            csv_path = os.path.join(parent_dir, senales[0]) if senales else os.path.join(parent_dir, csv_files[0])
        df_csv = pd.read_csv(csv_path)
        ch_idx = int(os.path.basename(carpeta_canal).split("_")[-1])
        col = f"Canal {ch_idx}"
        calibration_factor = np.max(np.abs(df_csv[col].values))
    except Exception:
        calibration_factor = 1.0

    signal_normalized, samplerate = at._read_wav_mono(filepath)
    signal_v = signal_normalized * calibration_factor

    # --- Conversión a µV con ganancia del circuito (idem analisis_trevisan) ---
    resistencia_ohm = 100.0
    meta_path = os.path.join(carpeta_canal, "metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
            resistencia_ohm = meta.get("resistencia_ohm", 100.0)
    except Exception:
        pass
    ganancia = 1.0 + (49400.0 / resistencia_ohm)
    signal = (signal_v / ganancia) * 1e6

    # --- Notch 50 Hz (idem cadena) ---
    if "desact" not in TIPO_FILTRO_RUIDO.lower() and TIPO_FILTRO_RUIDO.lower() != "none":
        try:
            b, a = iirnotch(50.0, NOTCH_Q, samplerate)
            signal = filtfilt(b, a, signal)
        except Exception:
            pass
    # --- Highpass 20 Hz orden 4 (idem cadena) ---
    nyquist = 0.5 * samplerate
    try:
        b, a = butter(4, min(HIGHPASS_HZ, nyquist * 0.99) / nyquist, btype="high")
        signal = filtfilt(b, a, signal)
    except Exception:
        pass
    # --- Lowpass 300 Hz orden 6 (idem cadena: orden 6 si ~300 Hz) ---
    try:
        orden_lp = 6 if abs(LOWPASS_HZ - 300.0) < 20.0 else 4
        sos_lp = butter(orden_lp, min(LOWPASS_HZ, nyquist * 0.99) / nyquist, btype="low", output="sos")
        signal = sosfiltfilt(sos_lp, signal)
    except Exception:
        pass

    return np.asarray(signal, dtype=np.float32), samplerate

# ============================ EXPORTADOR ============================

def descubrir_mediciones():
    date_path = os.path.join(BASE_DIR, FECHA)
    if not os.path.isdir(date_path):
        raise FileNotFoundError(f"No existe la carpeta de fecha: {date_path}")
    meds = []
    for med_folder in sorted(os.listdir(date_path)):
        med_path = os.path.join(date_path, med_folder)
        if os.path.isdir(med_path):
            if any(f.startswith("canal_") for f in os.listdir(med_path) if os.path.isdir(os.path.join(med_path, f))):
                meds.append((med_folder, med_path))
    return meds


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meds = descubrir_mediciones()
    print(f"Fecha {FECHA}: {len(meds)} mediciones con canal_ detectadas")

    filas = {res: [] for res in RESOLUCIONES}
    cache_proc = {}
    contador_sesion = {}

    for med_name, med_path in meds:
        partes = med_name.split("_")
        vocal = partes[0].upper()
        if vocal not in ["A", "E", "I", "O", "U"]:
            continue
        sesion_tag = "_".join(partes[1:])

        # --- Exclusiones manuales si existen (idem generador) ---
        excluded_windows = []
        exclude_path = os.path.join(med_path, "excluded_windows.json")
        if os.path.exists(exclude_path):
            try:
                with open(exclude_path, "r") as fh:
                    excluded_windows = json.load(fh).get("excluded_windows", [])
            except Exception:
                pass

        # --- Procesar cada canal con analisis_trevisan (env para alinear/SNR) ---
        canales_data = {}
        for ch in CANALES_FEATURES + [CANAL_ALINEACION]:
            carpeta = os.path.join(med_path, ch)
            if not os.path.isdir(carpeta):
                continue
            key = os.path.join(FECHA, med_name, ch)
            if key not in cache_proc:
                bpm_u, noise_u, pulsos_u = 50, 2.0, None
                try:
                    with open(os.path.join(carpeta, "metadata.json"), encoding="utf-8") as fh:
                        meta = json.load(fh)
                        bpm_u = meta.get("bpm", bpm_u)
                        noise_u = meta.get("noise_seconds", noise_u)
                        pulsos_u = meta.get("pulse_count", pulsos_u)
                except Exception:
                    pass
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    res = at.procesar_wavs_promedio(
                        carpeta=carpeta, output_root=carpeta, bpm=bpm_u,
                        mostrar_recortes=False, noise_seconds=noise_u,
                        n_pulsos_manual=pulsos_u, excluded_windows=excluded_windows,
                        show_interactive_plot=False, notch_q_factor=NOTCH_Q,
                        tipo_filtro_ruido=TIPO_FILTRO_RUIDO,
                        highpass_cutoff_hz=HIGHPASS_HZ, lowpass_cutoff_hz=LOWPASS_HZ,
                        tipo_envolvente="rms", smooth_ms=SMOOTH_MS,
                        pre_pct=PRE_PCT, post_pct=POST_PCT
                    )
                finally:
                    sys.stdout = old_stdout
                if res:
                    cache_proc[key] = res[list(res.keys())[0]]
            if key in cache_proc:
                canales_data[ch] = cache_proc[key]

        feats_ch = [c for c in CANALES_FEATURES if c in canales_data]
        if not feats_ch:
            print(f"[SKIP] {med_name}: sin canales de features")
            continue
        align_ch = CANAL_ALINEACION if CANAL_ALINEACION in canales_data else next(
            (c for c in ["canal_2", "canal_0", "canal_1"] if c in canales_data), None)
        if align_ch is None:
            print(f"[SKIP] {med_name}: sin canal de alineación")
            continue

        env_ref = canales_data[align_ch]["env_recortada"]
        muestras_pulso = canales_data[align_ch]["muestras_pulso"]
        sr = canales_data[align_ch].get("samplerate", 6000)

        # --- Señal cruda por canal de features (se carga y descarta por medición) ---
        raws = {}
        for ch in feats_ch:
            sig, sr_raw = cargar_senal_cruda(os.path.join(med_path, ch))
            raws[ch] = sig
        if any(r is None for r in raws.values()):
            print(f"[SKIP] {med_name}: no se pudo cargar la señal cruda")
            continue

        # --- Alineación idem generador: Pico Volumen Micrófono ---
        if align_ch == CANAL_ALINEACION:
            dist_samples = int(0.8 * muestras_pulso)
            min_height = np.max(env_ref) * 0.2
            picos, _ = find_peaks(env_ref, distance=dist_samples, height=min_height)
        else:
            # Formato viejo sin micrófono: alineación por pico del canal muscular
            # (idem rama "Pico Canal" del generador)
            win_alto = max(5, int(sr * 0.25))
            if win_alto % 2 == 0:
                win_alto += 1
            env_smooth = np.convolve(env_ref, np.ones(win_alto) / win_alto, mode="same")
            nl_a = canales_data[align_ch].get("noise_levels", [0])
            ch_noise = nl_a[0] if len(nl_a) > 0 else 0.0
            min_height = max(np.max(env_smooth) * 0.15, ch_noise * 1.2)
            candidatos, _ = find_peaks(env_smooth, distance=int(0.8 * muestras_pulso), height=min_height)
            picos = np.array([p for p in candidatos
                              if env_smooth[p] >= 0.10 * np.max(env_smooth)
                              and env_smooth[p] > ch_noise * 1.1])

        n_vent_ok = 0
        for win_idx, pico in enumerate(picos):
            pre_samples = int(muestras_pulso * PRE_PCT)
            post_samples = int(muestras_pulso * POST_PCT)
            s0, s1 = pico - pre_samples, pico + post_samples
            if s0 < 0 or s1 > len(env_ref):
                continue

            segs = []
            max_supremo = 1e-9
            ruido_acum = 0.0
            valido = True
            for ch in CANALES_FEATURES:
                env_ch = canales_data[ch]["env_recortada"]
                if s1 > len(env_ch):
                    valido = False
                    break
                seg = env_ch[s0:s1].copy()
                nl = canales_data[ch].get("noise_levels", [0])
                init_noise = nl[0] if len(nl) > 0 else 0.0
                noise_win = max(3, int(muestras_pulso / 4.0))
                n0 = max(0, int(pico - 0.5 * muestras_pulso - noise_win))
                n1 = min(len(env_ch), n0 + noise_win)
                ruido_pre = get_interpulse_noise(env_ch[n0:n1], init_noise) if n1 > n0 else init_noise
                n0p = min(len(env_ch), int(pico + 0.5 * muestras_pulso))
                n1p = min(len(env_ch), n0p + noise_win)
                ruido_post = get_interpulse_noise(env_ch[n0p:n1p], ruido_pre) if n1p > n0p else ruido_pre
                ruido_prom = (ruido_pre + ruido_post) / 2.0
                ruido_acum += ruido_prom
                seg = np.maximum(seg - ruido_prom * ALPHA_RUIDO, 0.0)
                max_supremo = max(max_supremo, np.max(seg))
                segs.append((ch, seg))
            if not valido or max_supremo <= 1e-6:
                continue

            snr = max_supremo / (ruido_acum / len(CANALES_FEATURES) + 1e-9)

            for res in RESOLUCIONES:
                fila = {
                    "fecha": FECHA, "sesion": sesion_tag, "toma": med_name,
                    "vocal": vocal, "ventana": win_idx,
                    "snr_envolvente": round(float(snr), 3),
                }
                for ch, seg in segs:
                    seg32 = np.asarray(seg, dtype=np.float32)
                    vent = np.interp(
                        np.linspace(0, len(seg32) - 1, res),
                        np.arange(len(seg32)), seg32
                    )
                    for i, v in enumerate(vent):
                        fila[f"{ch}_{i}"] = float(v)
                filas[res].append(fila)
            n_vent_ok += 1

        contador_sesion.setdefault(sesion_tag, 0)
        contador_sesion[sesion_tag] += n_vent_ok
        print(f"[OK] {med_name} ({sesion_tag}): {n_vent_ok} ventanas crudas")

    if not filas[RESOLUCIONES[0]]:
        print("No se exportó ninguna ventana.")
        return

    resumen = {
        "fecha": FECHA,
        "que_es": ("Ventanas CRUDAS de EMG alineadas con la cadena estándar del generador PCA/UMAP "
                   "(metrónomo via metadata.json, alineación por Pico Volumen Micrófono en canal_3, "
                   "ventana 40% pre / 60% post del período del pulso, resta de ruido interpulso). "
                   "La señal exportada es la CRUDA filtrada por la cadena estándar "
                   "(notch 50 Hz Q=2, highpass 20 Hz, lowpass 300 Hz, orden 6), en microvoltios (µV). "
                   "SIN ENVELOPTE TEMPORAL: se exporta la oscilación completa."),
        "resolucion": ("Cada fila es una ventana completa. Columnas canal_X_i = muestra i del canal X "
                       "(re-muestreo UNIFORME por interpolación lineal, sin alterar la amplitud µV). "
                       "Se incluyen dos resoluciones por archivo."),
        "canales_musculos": {"canal_0": "belly", "canal_1": "orbicularis mezcla",
                              "canal_2": "depresor mezcla", "canal_3": "Micrófono (solo alineación, NO exportado)"},
        "samplerate_original_hz": sr,
        "snr_envolvente": ("SNR de la ventana en el dominio de envolvente (parte de la cadena estándar); "
                            "útil para filtrar ventanas de mala calidad (umbral típico 0.5)."),
        "ventanas_por_sesion": contador_sesion,
        "archivos": {f"ventanas_crudas_{FECHA}_{r}pts.csv": f"{r} puntos por canal" for r in RESOLUCIONES},
    }

    salida = []
    for res in RESOLUCIONES:
        df = pd.DataFrame(filas[res])
        cols_meta = ["fecha", "sesion", "toma", "vocal", "ventana", "snr_envolvente"]
        cols_datos = [c for c in df.columns if c not in cols_meta]
        cols_datos = sorted(cols_datos, key=lambda c: (int(c.split("_")[1]), int(c.split("_")[2])))
        df = df[cols_meta + cols_datos]
        out_csv = os.path.join(OUT_DIR, f"ventanas_crudas_{FECHA}_{res}pts.csv")
        df.to_csv(out_csv, index=False, float_format="%.4f")
        peso_mb = os.path.getsize(out_csv) / 1e6
        salida.append((out_csv, len(df), peso_mb))
        print(f"[CSV] {out_csv} -> {len(df)} ventanas, {peso_mb:.1f} MB")

    out_readme = os.path.join(OUT_DIR, f"README_ventanas_crudas_{FECHA}.json")
    with open(out_readme, "w", encoding="utf-8") as fh:
        json.dump(resumen, fh, indent=4, ensure_ascii=False)

    # Copia lista para compartir (Gemini) a Descargas
    dest = os.path.expanduser(f"~/Descargas/ventanas_crudas_{FECHA}")
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(OUT_DIR, dest)
    print(f"[COPIA] {dest}")
    print("LISTO")


if __name__ == "__main__":
    main()
