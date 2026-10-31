#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: analysis/extractor_pulsos_vocales.py
# Descripción: Extrae pulsos individuales de una Secuencia Continua (con soporte
#              exacto para las columnas reales de segmentos_dataset.csv,
#              eventos_vocales_metronomo y pulsos_metronomo.csv con Jitter y
#              De Bruijn) y genera 5 carpetas agrupadas por vocal (A_..., E_...,
#              I_..., O_..., U_...) 100% compatibles con CsvViewerWidget y
#              generador_pca_umap.py.
# ==============================================================================

import os
import sys
import csv
import json
import glob
import shutil
import argparse
from datetime import datetime
import numpy as np
from scipy.io import wavfile


VOCALES_ESTANDAR = ("A", "E", "I", "O", "U")


def _safe_float(val, default=None):
    if val is None or val == "" or val == "-":
        return default
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val, default=None):
    if val is None or val == "" or val == "-":
        return default
    try:
        return int(float(val))
    except Exception:
        return default


def _leer_eventos_vocales(secuencia_dir: str, fs_hint: float = 2000.0):
    """
    Reconstruye la lista exacta de pulsos vocales grabados en una Secuencia Continua
    respetando la estructura nativa de autoforge_daq_experimental.py:
      1. metadata.json / autoforge_metadata.json -> 'eventos_vocales_metronomo'
         (usa 'pulso_vocal_idx', 'palabra', 't_wav_s', 't_daq_s', 'sample_daq', 'intervalo_s')
      2. pulsos_metronomo.csv -> filtrando estrictamente principal == 1, count_in == 0
         y usando 'idx', 'palabra', 't_wav_s', 'sample_daq', 'intervalo_s'
      3. segmentos_dataset.csv -> enriqueciendo con 'sample_beat', 'sample_onset_real',
         'sample_ventana_ini', 'sample_ventana_fin', 't_beat_s', 't_onset_s', 'invalido_manual'
    """
    csv_pulsos = os.path.join(secuencia_dir, "pulsos_metronomo.csv")
    meta_af = os.path.join(secuencia_dir, "autoforge_metadata.json")
    meta_root = os.path.join(secuencia_dir, "metadata.json")
    meta_ch0 = os.path.join(secuencia_dir, "canal_0", "metadata.json")
    csv_seg = os.path.join(secuencia_dir, "segmentos_dataset.csv")

    meta = {}
    for mp in (meta_ch0, meta_root, meta_af):
        if os.path.exists(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        meta.update(loaded)
            except Exception:
                pass

    fs = float(
        meta.get("sample_rate")
        or meta.get("daq_sample_rate")
        or (meta.get("hardware", {}) or {}).get("sample_rate_hz", fs_hint)
    )
    bpm_default = float(meta.get("bpm") or (meta.get("metronomo", {}) or {}).get("bpm", 30.0))
    tau_default = 60.0 / max(1.0, bpm_default)
    noise_seconds = float(meta.get("noise_seconds") or (meta.get("metronomo", {}) or {}).get("noise_seconds", 3.0))

    eventos_by_idx = {}

    # 1. Leer desde metadata -> eventos_vocales_metronomo (fuente canónica de los pulsos vocales reales)
    evs_meta = (
        meta.get("eventos_vocales_metronomo")
        or meta.get("vocal_cue_log")
        or (meta.get("locuciones", {}) or {}).get("eventos_vocales_metronomo", [])
    )
    if isinstance(evs_meta, list) and evs_meta:
        for pos, ev in enumerate(evs_meta, start=1):
            p_idx = _safe_int(ev.get("pulso_vocal_idx") or ev.get("pulso_idx") or ev.get("idx"), pos)
            vocal = str(ev.get("palabra") or ev.get("vocal") or "").strip().upper().replace("/", "")
            sample_daq = _safe_int(ev.get("sample_daq"))
            t_wav = _safe_float(ev.get("t_wav_s"))
            t_daq = _safe_float(ev.get("t_daq_s") or ev.get("t_s") or ev.get("timestamp_relativo_s"))
            if t_wav is None:
                if sample_daq is not None and fs > 0:
                    t_wav = sample_daq / fs
                elif t_daq is not None:
                    t_wav = t_daq + noise_seconds
                else:
                    continue
            tau_s = _safe_float(ev.get("intervalo_s") or ev.get("tau_s"), tau_default)
            eventos_by_idx[p_idx] = {
                "pulso_idx": p_idx,
                "vocal": vocal,
                "t_beat_s": float(t_wav),
                "sample_beat": sample_daq if sample_daq is not None else int(round(float(t_wav) * fs)),
                "tau_s": float(tau_s),
                "win_ini_s": max(0.0, float(t_wav) - 0.35 * float(tau_s)),
                "win_fin_s": float(t_wav) + 0.65 * float(tau_s),
                "invalido_manual": False,
            }

    # 2. Leer/completar desde pulsos_metronomo.csv (filtrando sub-pulsos y count-in)
    if os.path.exists(csv_pulsos):
        try:
            with open(csv_pulsos, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                # Si tiene columnas 'principal' y 'count_in' (formato nativo autoforge_daq_experimental.py)
                if "principal" in r and _safe_int(r.get("principal"), 1) != 1:
                    continue
                if "count_in" in r and _safe_int(r.get("count_in"), 0) > 0:
                    continue
                p_idx = _safe_int(r.get("idx") or r.get("pulso_idx"), 0)
                if p_idx is None or p_idx <= 0:
                    continue
                vocal = str(r.get("palabra") or r.get("vocal") or "").strip().upper().replace("/", "")
                if not vocal:
                    continue
                t_wav = _safe_float(r.get("t_wav_s") or r.get("timestamp_relativo_s") or r.get("t_s"))
                sample_daq = _safe_int(r.get("sample_daq"))
                t_daq = _safe_float(r.get("t_daq_s"))
                if t_wav is None:
                    if sample_daq is not None and fs > 0:
                        t_wav = sample_daq / fs
                    elif t_daq is not None:
                        t_wav = t_daq + noise_seconds
                    else:
                        continue
                tau_s = _safe_float(r.get("intervalo_s") or r.get("tau_s"), tau_default)
                if p_idx not in eventos_by_idx:
                    eventos_by_idx[p_idx] = {
                        "pulso_idx": p_idx,
                        "vocal": vocal,
                        "t_beat_s": float(t_wav),
                        "sample_beat": sample_daq if sample_daq is not None else int(round(float(t_wav) * fs)),
                        "tau_s": float(tau_s),
                        "win_ini_s": max(0.0, float(t_wav) - 0.35 * float(tau_s)),
                        "win_fin_s": float(t_wav) + 0.65 * float(tau_s),
                        "invalido_manual": False,
                    }
        except Exception as e:
            print(f"[Aviso] Error leyendo pulsos_metronomo.csv: {e}")

    # 3. Enriquecer o reconstruir desde segmentos_dataset.csv (formato exacto de autoforge_daq_experimental.py)
    if os.path.exists(csv_seg):
        try:
            with open(csv_seg, "r", encoding="utf-8") as f:
                seg_rows = list(csv.DictReader(f))
            for pos, sr in enumerate(seg_rows, start=1):
                p_idx = _safe_int(sr.get("pulso_id") or sr.get("pulso_idx"), pos)
                vocal = str(sr.get("vocal") or "").strip().upper().replace("/", "")
                inv_man = bool(_safe_int(sr.get("invalido_manual"), 0) == 1)
                s_beat = _safe_int(sr.get("sample_beat"))
                s_onset = _safe_int(sr.get("sample_onset_real"))
                s_ini = _safe_int(sr.get("sample_ventana_ini"))
                s_fin = _safe_int(sr.get("sample_ventana_fin"))
                t_beat = _safe_float(sr.get("t_beat_s") or sr.get("beat_time_s"))
                t_onset = _safe_float(sr.get("t_onset_s"))

                if t_beat is None and s_beat is not None and fs > 0:
                    t_beat = s_beat / fs
                if t_beat is None:
                    continue

                w_ini = (s_ini / fs) if (s_ini is not None and fs > 0) else _safe_float(sr.get("ventana_inicio_s"), max(0.0, t_beat - 0.35 * tau_default))
                w_fin = (s_fin / fs) if (s_fin is not None and fs > 0) else _safe_float(sr.get("ventana_fin_s"), t_beat + 0.65 * tau_default)

                if p_idx in eventos_by_idx:
                    ev = eventos_by_idx[p_idx]
                    if vocal:
                        ev["vocal"] = vocal
                    if w_fin > w_ini:
                        ev["win_ini_s"] = float(w_ini)
                        ev["win_fin_s"] = float(w_fin)
                        ev["tau_s"] = max(0.25, float(w_fin - w_ini))
                    if t_onset is not None and t_onset > 0:
                        ev["t_onset_s"] = float(t_onset)
                    if s_onset is not None and s_onset > 0:
                        ev["sample_onset_real"] = int(s_onset)
                    ev["invalido_manual"] = inv_man
                else:
                    eventos_by_idx[p_idx] = {
                        "pulso_idx": p_idx,
                        "vocal": vocal,
                        "t_beat_s": float(t_beat),
                        "t_onset_s": float(t_onset) if t_onset is not None else float(t_beat),
                        "sample_beat": s_beat if s_beat is not None else int(round(float(t_beat) * fs)),
                        "tau_s": max(0.25, float(w_fin - w_ini)),
                        "win_ini_s": float(w_ini),
                        "win_fin_s": float(w_fin),
                        "invalido_manual": inv_man,
                    }
        except Exception as e:
            print(f"[Aviso] Error leyendo segmentos_dataset.csv: {e}")

    eventos = [eventos_by_idx[k] for k in sorted(eventos_by_idx.keys()) if not eventos_by_idx[k].get("invalido_manual", False)]

    # Ajustar tau_s e intervalos de ventana usando la distancia real entre beats consecutivos (soporte Jitter)
    for i, ev in enumerate(eventos):
        if i + 1 < len(eventos):
            dt_next = eventos[i + 1]["t_beat_s"] - ev["t_beat_s"]
            if 0.35 <= dt_next <= 8.0:
                ev["tau_s"] = float(dt_next)
        # Asegurar que la ventana no se solape con el pulso siguiente
        tau_eff = float(ev["tau_s"])
        if "sample_onset_real" not in ev:
            ev["win_ini_s"] = max(0.0, ev["t_beat_s"] - 0.30 * tau_eff)
            ev["win_fin_s"] = ev["t_beat_s"] + 0.65 * tau_eff

    return eventos, meta


def _cargar_matriz_cruda(secuencia_dir: str, meta: dict):
    """
    Carga la matriz multicanal cruda (n_canales, n_muestras), los nombres de columna,
    el vector de tiempo real y el sample_rate desde grabacion.csv (separando siempre
    la primera columna 'Tiempo (s)') o desde canal_X/ Crudo.wav.
    """
    fs = float(
        meta.get("sample_rate")
        or meta.get("daq_sample_rate")
        or (meta.get("hardware", {}) or {}).get("sample_rate_hz", 2000.0)
    )
    csv_grab = os.path.join(secuencia_dir, "grabacion.csv")

    if os.path.exists(csv_grab):
        try:
            with open(csv_grab, "r", encoding="utf-8", errors="ignore") as fh:
                first_line = fh.readline().strip()
            if first_line.startswith("#"):
                first_line = first_line.lstrip("#").strip()
            cols_raw = [c.strip() for c in first_line.split(",")]
            cols_lower = [c.lower() for c in cols_raw]
            has_header = any(any(ch.isalpha() for ch in col) for col in cols_lower)

            raw_mat = np.loadtxt(csv_grab, delimiter=",", skiprows=1 if has_header else 0)
            if raw_mat.ndim == 1:
                raw_mat = raw_mat[:, np.newaxis]

            # Detectar si la columna 0 es el vector de Tiempo ('Tiempo (s)', 'Time', etc.)
            tiene_col_tiempo = False
            if has_header and cols_lower and any(t_k in cols_lower[0] for t_k in ("tiempo", "time")):
                tiene_col_tiempo = True
            elif raw_mat.shape[1] >= 2:
                col0 = raw_mat[:, 0]
                diffs = np.diff(col0[:min(500, len(col0))])
                if col0[0] >= 0.0 and len(diffs) > 0 and np.all(diffs > 0):
                    tiene_col_tiempo = True

            if tiene_col_tiempo and raw_mat.shape[1] >= 2:
                t_vec = raw_mat[:, 0]
                if len(t_vec) > 10:
                    dt_med = float(np.median(np.diff(t_vec[:min(2000, len(t_vec))])))
                    if dt_med > 1e-6:
                        fs = round(1.0 / dt_med)
                sig_mat = raw_mat[:, 1:].T
                ch_headers = cols_raw[1:] if (has_header and len(cols_raw) == raw_mat.shape[1]) else [f"Canal {i}" for i in range(sig_mat.shape[0])]
                return sig_mat, t_vec, int(round(fs)), ch_headers
            else:
                sig_mat = raw_mat.T
                t_vec = np.arange(sig_mat.shape[1], dtype=np.float64) / float(fs)
                ch_headers = [f"Canal {i}" for i in range(sig_mat.shape[0])]
                return sig_mat, t_vec, int(round(fs)), ch_headers
        except Exception as e:
            print(f"[Aviso] Falló lectura de grabacion.csv ({e}), leyendo desde Crudo.wav...")

    # Fallback: leer canal_0/ Crudo.wav, canal_1/ Crudo.wav, etc.
    ch_dirs = sorted(
        [d for d in glob.glob(os.path.join(secuencia_dir, "canal_*")) if os.path.isdir(d)],
        key=lambda x: int(os.path.basename(x).split("_")[-1]) if os.path.basename(x).split("_")[-1].isdigit() else 0
    )
    canales_arrays = []
    for cd in ch_dirs:
        wav_cands = [os.path.join(cd, " Crudo.wav"), os.path.join(cd, "Crudo.wav")]
        wav_path = next((w for w in wav_cands if os.path.exists(w)), None)
        if wav_path:
            sr_w, data_w = wavfile.read(wav_path)
            fs = float(sr_w)
            if data_w.dtype == np.int16:
                arr = (data_w.astype(np.float64) / 32767.0) * 5.0
            else:
                arr = data_w.astype(np.float64)
            canales_arrays.append(arr)

    if not canales_arrays:
        raise FileNotFoundError(f"No se encontró grabacion.csv ni archivos Crudo.wav en {secuencia_dir}")

    min_len = min(len(a) for a in canales_arrays)
    mat = np.vstack([a[:min_len] for a in canales_arrays])
    t_vec = np.arange(min_len, dtype=np.float64) / float(fs)
    ch_headers = [f"Canal {i}" for i in range(mat.shape[0])]
    return mat, t_vec, int(round(fs)), ch_headers


def _refinar_centro_pulso_con_envolvente(
    raw_matrix: np.ndarray,
    fs: int,
    t_beat: float,
    tau_s: float,
    t_onset_meta: float = None,
    centrar_en_pico: bool = True,
) -> float:
    """
    Alinea la ventana de extracción con el centro real de activación fonatoria/muscular:
    - Si segmentos_dataset.csv ya calculó t_onset_real, lo toma como referencia inicial.
    - Evalúa la envolvente del micrófono (Canal 3) y de los canales EMG (Canales 0..2)
      en [t_beat - 0.15*tau, t_beat + 0.55*tau] para no cortar pulsos con latencia de Jitter.
    """
    if not centrar_en_pico:
        return t_beat
    if t_onset_meta is not None and abs(t_onset_meta - t_beat) <= 0.45 * tau_s:
        return float(t_onset_meta)

    n_canales, n_samples = raw_matrix.shape
    s_ini = max(0, int(round((t_beat - 0.15 * tau_s) * fs)))
    s_fin = min(n_samples, int(round((t_beat + 0.55 * tau_s) * fs)))
    if s_fin - s_ini < int(0.12 * fs):
        return t_beat

    win_smooth = max(5, int(round(0.050 * fs)))
    kernel = np.ones(win_smooth, dtype=np.float64) / float(win_smooth)

    if n_canales >= 4:
        mic_seg = raw_matrix[3, s_ini:s_fin]
        mic_centered = np.abs(mic_seg - np.median(mic_seg))
        if np.max(mic_centered) > 3.5 * (np.median(mic_centered) + 1e-6):
            env = np.convolve(mic_centered, kernel, mode="same")
            idx_peak = int(np.argmax(env))
            t_peak = (s_ini + idx_peak) / float(fs)
            return float(np.clip(t_peak, t_beat - 0.10 * tau_s, t_beat + 0.45 * tau_s))

    emg_chs = min(3, n_canales)
    env_sum = np.zeros(s_fin - s_ini, dtype=np.float64)
    for c in range(emg_chs):
        seg = raw_matrix[c, s_ini:s_fin]
        rect = np.abs(seg - np.median(seg))
        env_sum += np.convolve(rect, kernel, mode="same")
    if np.max(env_sum) > 2.5 * (np.median(env_sum) + 1e-9):
        idx_peak = int(np.argmax(env_sum))
        t_peak = (s_ini + idx_peak) / float(fs)
        return float(np.clip(t_peak, t_beat - 0.10 * tau_s, t_beat + 0.45 * tau_s))

    return t_beat


def extraer_pulsos_por_vocal(
    secuencia_dir: str,
    output_parent_dir: str = None,
    prefijo_prueba: str = None,
    sujeto_override: str = None,
    incluir_ruido_basal: bool = True,
    duracion_ventana_fija_s: float = None,
    centrar_en_pico_energia: bool = True,
    callback_log=print,
) -> dict:
    """
    Extrae todos los pulsos de una carpeta SecuenciaContinua_* (con o sin Jitter / De Bruijn)
    y construye una carpeta de medición por cada vocal (A_<Prueba>_<Sujeto>, E_..., I_..., O_..., U_...)
    con grabacion.csv ("Tiempo (s),Canal 0,Canal 1,...") 100% compatible con CsvViewerWidget
    y generador_pca_umap.py.
    """
    secuencia_dir = os.path.abspath(secuencia_dir)
    if not os.path.isdir(secuencia_dir):
        raise ValueError(f"No existe el directorio: {secuencia_dir}")

    if output_parent_dir is None:
        output_parent_dir = os.path.dirname(secuencia_dir)
    os.makedirs(output_parent_dir, exist_ok=True)

    callback_log(f"==> Analizando secuencia continua: {os.path.basename(secuencia_dir)}")
    raw_matrix, t_vec_orig, fs, ch_headers = _cargar_matriz_cruda(secuencia_dir, {})
    eventos, meta = _leer_eventos_vocales(secuencia_dir, fs_hint=float(fs))
    if not eventos:
        raise RuntimeError(
            "No se encontraron eventos de pulsos en metadata.json, pulsos_metronomo.csv ni segmentos_dataset.csv."
        )

    n_canales, n_samples = raw_matrix.shape
    duracion_total_s = float(t_vec_orig[-1]) if len(t_vec_orig) > 0 else (n_samples / float(fs))

    sujeto = sujeto_override or meta.get("sujeto") or (meta.get("identificacion", {}) or {}).get("sujeto")
    prueba = prefijo_prueba or meta.get("prueba") or (meta.get("identificacion", {}) or {}).get("prueba")
    base_folder = os.path.basename(secuencia_dir)
    parts = base_folder.split("_")
    if not prueba:
        prueba = parts[1] if len(parts) >= 3 else "SecuenciaExtraida"
    if not sujeto:
        sujeto = parts[2] if len(parts) >= 3 else "Sujeto"

    noise_seconds = float(meta.get("noise_seconds") or (meta.get("metronomo", {}) or {}).get("noise_seconds", 3.0))
    noise_samples = min(n_samples, max(int(0.5 * fs), int(round(noise_seconds * fs))))
    segmento_ruido = raw_matrix[:, :noise_samples] if incluir_ruido_basal else np.zeros((n_canales, int(1.0 * fs)))
    noise_dur_efectiva = segmento_ruido.shape[1] / float(fs)
    offset_ruido_samples = int(segmento_ruido.shape[1])

    eventos_por_vocal = {}
    for ev in eventos:
        v = ev["vocal"]
        if not v:
            continue
        eventos_por_vocal.setdefault(v, []).append(ev)

    callback_log(
        f"==> Canales: {n_canales} ({', '.join(ch_headers)}) | Fs: {fs} Hz | Duración: {duracion_total_s:.1f}s | "
        f"Vocales detectadas: {', '.join(f'{k} ({len(v)} pulsos)' for k, v in sorted(eventos_por_vocal.items()))}"
    )

    carpetas_creadas = {}
    for vocal, lista_evs in sorted(eventos_por_vocal.items()):
        folder_name = f"{vocal}_{prueba}_{sujeto}"
        vocal_dir = os.path.join(output_parent_dir, folder_name)
        os.makedirs(vocal_dir, exist_ok=True)

        segmentos_vocal = [segmento_ruido]
        nuevos_timestamps_wav_s = []
        nuevos_sample_indices = []
        nuevos_intervalos_s = []
        registros_pulsos_nativos = []
        cursor_muestras = offset_ruido_samples

        # Pausa limpia (150 ms) entre recortes concatenados con nivel basal mediano de cada canal
        pad_samples = int(round(0.15 * fs))
        baseline_por_canal = np.median(segmento_ruido, axis=1, keepdims=True)
        bloque_pad = np.repeat(baseline_por_canal, pad_samples, axis=1)

        for idx_local, ev in enumerate(lista_evs, start=1):
            t_beat_nominal = float(ev["t_beat_s"])
            tau_ev = float(ev.get("tau_s", 2.0))
            t_anchor = _refinar_centro_pulso_con_envolvente(
                raw_matrix,
                fs,
                t_beat_nominal,
                tau_ev,
                t_onset_meta=ev.get("t_onset_s"),
                centrar_en_pico=centrar_en_pico_energia,
            )

            if duracion_ventana_fija_s is not None and duracion_ventana_fija_s > 0:
                w_ini = max(0.0, t_anchor - 0.38 * duracion_ventana_fija_s)
                w_fin = min(duracion_total_s, t_anchor + 0.62 * duracion_ventana_fija_s)
            else:
                ancho_orig = max(0.8, min(4.5, tau_ev))
                w_ini = max(0.0, t_anchor - 0.35 * ancho_orig)
                w_fin = min(duracion_total_s, t_anchor + 0.65 * ancho_orig)

            idx_ini = int(np.searchsorted(t_vec_orig, w_ini, side="left"))
            idx_fin = int(np.searchsorted(t_vec_orig, w_fin, side="right"))
            idx_ini = max(0, min(n_samples - 1, idx_ini))
            idx_fin = max(idx_ini + int(0.25 * fs), min(n_samples, idx_fin))
            recorte = raw_matrix[:, idx_ini:idx_fin]

            offset_anchor_samples = max(0, int(round(t_anchor * fs)) - idx_ini)
            sample_daq_nuevo = int(cursor_muestras + offset_anchor_samples)
            t_wav_nuevo = round(sample_daq_nuevo / float(fs), 6)
            t_daq_nuevo = round(max(0.0, t_wav_nuevo - noise_dur_efectiva), 6)
            tau_clip = round((recorte.shape[1] + pad_samples) / float(fs), 6)

            nuevos_timestamps_wav_s.append(t_wav_nuevo)
            nuevos_sample_indices.append(sample_daq_nuevo)
            nuevos_intervalos_s.append(tau_clip)

            registros_pulsos_nativos.append({
                "seq": idx_local,
                "idx": idx_local,
                "sub": 0,
                "principal": 1,
                "count_in": 0,
                "t_epoch": round(time.time() if "time" in globals() else 0.0, 6),
                "t_perf": t_daq_nuevo,
                "t_daq_s": t_daq_nuevo,
                "t_wav_s": t_wav_nuevo,
                "sample_daq": sample_daq_nuevo,
                "offset_muestras": offset_ruido_samples,
                "bpm": int(round(60.0 / max(0.2, tau_clip))),
                "subdiv": 1,
                "intervalo_s": tau_clip,
                "jitter_activo": 1,
                "jitter_mode": "extracted_from_continua",
                "palabra": vocal,
            })

            segmentos_vocal.append(recorte)
            cursor_muestras += recorte.shape[1]
            segmentos_vocal.append(bloque_pad)
            cursor_muestras += bloque_pad.shape[1]

        matriz_vocal = np.concatenate(segmentos_vocal, axis=1)
        total_muestras_vocal = matriz_vocal.shape[1]
        tiempo_vocal = np.arange(total_muestras_vocal, dtype=np.float64) / float(fs)

        # 1. Guardar grabacion.csv con el formato nativo exacto de CsvViewerWidget:
        #    "Tiempo (s),Canal 0,Canal 1,Canal 2,Canal 3"
        csv_out_path = os.path.join(vocal_dir, "grabacion.csv")
        datos_para_csv = np.hstack((tiempo_vocal.reshape(-1, 1), matriz_vocal.T))
        headers_norm = [f"Canal {i}" for i in range(n_canales)]
        headers_str = "Tiempo (s)," + ",".join(headers_norm)
        np.savetxt(
            csv_out_path,
            datos_para_csv,
            delimiter=",",
            header=headers_str,
            comments="",
            fmt="%.5f",
        )

        # 2. Guardar pulsos_metronomo.csv con las 17 columnas nativas de escribir_pulsos_metronomo_csv
        pulsos_csv_path = os.path.join(vocal_dir, "pulsos_metronomo.csv")
        campos_nativos = [
            "seq", "idx", "sub", "principal", "count_in", "t_epoch", "t_perf",
            "t_daq_s", "t_wav_s", "sample_daq", "offset_muestras", "bpm", "subdiv",
            "intervalo_s", "jitter_activo", "jitter_mode", "palabra"
        ]
        with open(pulsos_csv_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=campos_nativos)
            writer.writeheader()
            writer.writerows(registros_pulsos_nativos)

        # 3. Guardar cada canal en canal_X/ Crudo.wav + metadata.json compatible con generador_pca_umap.py y CsvViewerWidget
        muscles_map = meta.get("muscles_map") or {}
        muscles_list = meta.get("muscles") or []
        tau_medio = float(np.mean(nuevos_intervalos_s)) if nuevos_intervalos_s else 2.0
        bpm_medio = round(60.0 / max(0.2, tau_medio), 2)

        eventos_vocales_meta = [
            {
                "pulso_vocal_idx": r["idx"],
                "palabra": vocal,
                "t_daq_s": r["t_daq_s"],
                "t_wav_s": r["t_wav_s"],
                "sample_daq": r["sample_daq"],
                "intervalo_s": r["intervalo_s"],
                "jitter_activo": True,
                "jitter_mode": "extracted_from_continua",
            }
            for r in registros_pulsos_nativos
        ]

        for ch in range(n_canales):
            ch_dir = os.path.join(vocal_dir, f"canal_{ch}")
            os.makedirs(ch_dir, exist_ok=True)
            wav_data = np.int16(np.clip(matriz_vocal[ch, :] / 5.0, -1.0, 1.0) * 32767)
            wavfile.write(os.path.join(ch_dir, " Crudo.wav"), fs, wav_data)

            meta_ch_orig = os.path.join(secuencia_dir, f"canal_{ch}", "metadata.json")
            ch_meta = dict(meta)
            if os.path.exists(meta_ch_orig):
                try:
                    with open(meta_ch_orig, "r", encoding="utf-8") as fm:
                        ch_meta.update(json.load(fm))
                except Exception:
                    pass

            musc_name = (
                ch_meta.get("musculo")
                or muscles_map.get(f"canal_{ch}")
                or muscles_map.get(str(ch))
                or (muscles_list[ch] if ch < len(muscles_list) else f"Canal {ch}")
            )
            ch_meta.update({
                "canal": f"canal_{ch}",
                "palabra": vocal,
                "vocal": vocal,
                "letra": vocal,
                "sujeto": sujeto,
                "prueba": prueba,
                "musculo": musc_name,
                "sample_rate": fs,
                "daq_sample_rate": fs,
                "bpm": bpm_medio,
                "pulse_count": len(nuevos_timestamps_wav_s),
                "noise_seconds": round(noise_dur_efectiva, 4),
                "pulse_timestamps": nuevos_timestamps_wav_s,
                "beat_timestamps_s": nuevos_timestamps_wav_s,
                "beat_sample_indices": nuevos_sample_indices,
                "beat_intervals_s": nuevos_intervalos_s,
                "origen_extraccion": os.path.basename(secuencia_dir),
                "fecha_extraccion": datetime.now().isoformat(),
                "eventos_vocales_metronomo": eventos_vocales_meta,
            })
            with open(os.path.join(ch_dir, "metadata.json"), "w", encoding="utf-8") as fm_out:
                json.dump(ch_meta, fm_out, indent=4, ensure_ascii=False)

        with open(os.path.join(vocal_dir, "metadata.json"), "w", encoding="utf-8") as f_root_meta:
            json.dump(ch_meta, f_root_meta, indent=4, ensure_ascii=False)

        for cal_file in ("calibracion_espacial_electrodos.json", "calibracion_cvm.csv"):
            src_cal = os.path.join(secuencia_dir, cal_file)
            if os.path.exists(src_cal):
                shutil.copy2(src_cal, os.path.join(vocal_dir, cal_file))

        carpetas_creadas[vocal] = {
            "carpeta": vocal_dir,
            "nombre": folder_name,
            "n_pulsos": len(nuevos_timestamps_wav_s),
            "duracion_s": round(total_muestras_vocal / float(fs), 2),
        }
        callback_log(
            f"  [OK] {folder_name}: {len(nuevos_timestamps_wav_s)} pulsos extraídos "
            f"({total_muestras_vocal / float(fs):.1f}s) -> {vocal_dir}"
        )

    return carpetas_creadas


def lanzar_gui_extractor(initial_dir: str = None):
    """Interfaz gráfica PySide6 para extraer pulsos por vocal y hacerlos compatibles con CsvViewer y PCA/UMAP."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    own_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        own_app = True

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Ñandú LSD — Extractor de Pulsos por Vocal (Secuencia Continua / Jitter -> CSV & PCA/UMAP)")
    dlg.resize(780, 540)
    dlg.setStyleSheet("""
        QDialog, QWidget { background-color: #0b0e14; color: #e6edf3; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
        QLineEdit, QDoubleSpinBox { background-color: #161b22; color: #00ffcc; border: 1px solid #30363d; padding: 5px; border-radius: 4px; }
        QPushButton { background-color: #14261f; color: #00ffcc; border: 1px solid #00ffcc; padding: 7px 14px; font-weight: bold; border-radius: 4px; }
        QPushButton:hover { background-color: #00ffcc; color: #000000; }
        QTextEdit { background-color: #050508; color: #39ff14; border: 1px solid #222; font-family: 'Consolas', monospace; font-size: 12px; }
    """)

    layout = QtWidgets.QVBoxLayout(dlg)
    lbl_header = QtWidgets.QLabel(
        "<b style='color:#00ffcc; font-size:16px;'>Extractor de Pulsos por Vocal (Compatible con Visor CSV y PCA / UMAP)</b><br>"
        "<span style='color:#9ca3af;'>Lee las marcas exactas (<code>t_wav_s</code>, <code>sample_daq</code>, <code>segmentos_dataset.csv</code>) "
        "de una <b>Secuencia Continua</b> (con o sin Jitter / De Bruijn) y genera las 5 carpetas "
        "(<code>A_...</code>, <code>E_...</code>, <code>I_...</code>, <code>O_...</code>, <code>U_...</code>) "
        "con <code>grabacion.csv</code> nativo (<code>Tiempo (s),Canal 0,Canal 1,...</code>).</span>"
    )
    lbl_header.setWordWrap(True)
    layout.addWidget(lbl_header)

    form = QtWidgets.QFormLayout()
    row_dir = QtWidgets.QHBoxLayout()
    inp_secuencia = QtWidgets.QLineEdit(initial_dir or "")
    btn_browse = QtWidgets.QPushButton("Seleccionar Carpeta SecuenciaContinua...")
    row_dir.addWidget(inp_secuencia, 1)
    row_dir.addWidget(btn_browse)
    form.addRow("Carpeta Origen:", row_dir)

    inp_prueba = QtWidgets.QLineEdit("")
    inp_prueba.setPlaceholderText("Auto-detectado (ej. Prueba1)")
    form.addRow("Identificador de Prueba:", inp_prueba)

    inp_sujeto = QtWidgets.QLineEdit("")
    inp_sujeto.setPlaceholderText("Auto-detectado (ej. Candela)")
    form.addRow("Nombre del Sujeto:", inp_sujeto)

    chk_refinar_pico = QtWidgets.QCheckBox("Refinar centro del pulso con onset real / pico de energía Micrófono & sEMG")
    chk_refinar_pico.setChecked(True)
    form.addRow("Alineación Fina:", chk_refinar_pico)

    chk_ventana_fija = QtWidgets.QCheckBox("Forzar ancho de ventana fijo por pulso (opcional)")
    spin_ventana = QtWidgets.QDoubleSpinBox()
    spin_ventana.setRange(0.5, 6.0)
    spin_ventana.setValue(2.0)
    spin_ventana.setSuffix(" s")
    spin_ventana.setEnabled(False)
    chk_ventana_fija.toggled.connect(spin_ventana.setEnabled)
    row_win = QtWidgets.QHBoxLayout()
    row_win.addWidget(chk_ventana_fija)
    row_win.addWidget(spin_ventana)
    form.addRow("Recorte Temporal:", row_win)

    layout.addLayout(form)

    btn_run = QtWidgets.QPushButton("EXTRAER PULSOS Y CREAR CARPETAS POR VOCAL (A, E, I, O, U)")
    btn_run.setStyleSheet("""
        QPushButton { background-color: #003b2a; color: #00ffcc; border: 2px solid #00ffcc; padding: 12px; font-size: 14px; font-weight: 900; }
        QPushButton:hover { background-color: #00ffcc; color: #000; }
    """)
    layout.addWidget(btn_run)

    txt_log = QtWidgets.QTextEdit()
    txt_log.setReadOnly(True)
    layout.addWidget(txt_log, 1)

    def _log(msg):
        txt_log.append(str(msg))
        QtWidgets.QApplication.processEvents()

    def _elegir_carpeta():
        base_start = inp_secuencia.text().strip() or os.getcwd()
        d = QtWidgets.QFileDialog.getExistingDirectory(dlg, "Seleccionar carpeta SecuenciaContinua_*", base_start)
        if d:
            inp_secuencia.setText(d)

    def _ejecutar():
        target = inp_secuencia.text().strip()
        if not target or not os.path.isdir(target):
            QtWidgets.QMessageBox.warning(dlg, "Carpeta inválida", "Selecciona una carpeta de Secuencia Continua válida.")
            return
        try:
            win_fija = spin_ventana.value() if chk_ventana_fija.isChecked() else None
            res = extraer_pulsos_por_vocal(
                secuencia_dir=target,
                prefijo_prueba=inp_prueba.text().strip() or None,
                sujeto_override=inp_sujeto.text().strip() or None,
                duracion_ventana_fija_s=win_fija,
                centrar_en_pico_energia=chk_refinar_pico.isChecked(),
                callback_log=_log,
            )
            resumen = "\n".join([f"• {v}: {info['nombre']} ({info['n_pulsos']} pulsos)" for v, info in res.items()])
            QtWidgets.QMessageBox.information(
                dlg,
                "Extracción Completada",
                f"Se crearon {len(res)} carpetas compatibles con el Visor CSV y PCA/UMAP:\n\n{resumen}\n\n"
                "Haz clic en 'Refrescar Directorio' en el Gestor de Sesiones para visualizarlas."
            )
        except Exception as e:
            import traceback
            _log(f"[ERROR] {e}\n{traceback.format_exc()}")
            QtWidgets.QMessageBox.critical(dlg, "Error en Extracción", str(e))

    btn_browse.clicked.connect(_elegir_carpeta)
    btn_run.clicked.connect(_ejecutar)

    dlg.exec()
    if own_app:
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Extractor de pulsos por vocal desde Secuencia Continua (con Jitter / De Bruijn).")
    parser.add_argument("carpeta", nargs="?", default=None, help="Ruta a la carpeta SecuenciaContinua_*")
    parser.add_argument("--out", default=None, help="Directorio padre de salida (por defecto misma carpeta de fecha)")
    parser.add_argument("--prueba", default=None, help="Nombre de la prueba (ej. Prueba1)")
    parser.add_argument("--sujeto", default=None, help="Nombre del sujeto (ej. Candela)")
    parser.add_argument("--ventana-fija", type=float, default=None, help="Ancho fijo de ventana en segundos (opcional)")
    parser.add_argument("--no-refinar-pico", action="store_true", help="Desactivar refinamiento por pico de energía")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en modo consola sin abrir GUI")
    args = parser.parse_args()

    if args.cli and args.carpeta:
        extraer_pulsos_por_vocal(
            secuencia_dir=args.carpeta,
            output_parent_dir=args.out,
            prefijo_prueba=args.prueba,
            sujeto_override=args.sujeto,
            duracion_ventana_fija_s=args.ventana_fija,
            centrar_en_pico_energia=not args.no_refinar_pico,
        )
    else:
        lanzar_gui_extractor(initial_dir=args.carpeta)


if __name__ == "__main__":
    main()
