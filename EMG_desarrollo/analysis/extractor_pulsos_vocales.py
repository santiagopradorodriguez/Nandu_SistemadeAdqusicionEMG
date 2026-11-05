#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: analysis/extractor_pulsos_vocales.py
# Descripción: Extrae pulsos individuales de una Secuencia Continua anclando
#              OBLIGATORIAMENTE cada ventana a la referencia acústica real del
#              Micrófono (canal_3/grabacion.wav o Crudo.wav) mediante envolvente
#              de Hilbert / RMS en banda vocal (80-900 Hz), descartando con
#              rigurosidad absoluta los pulsos de preparación (count_in > 0) y
#              exportando carpetas A_..., E_..., I_..., O_..., U_... con
#              grabacion.csv ("Tiempo (s),Canal 0,Canal 1,Canal 2,Canal 3") y
#              subcarpetas canal_0..3 compatibles con CsvViewerWidget y PCA/UMAP.
# ==============================================================================

import os
import sys
import csv
import json
import glob
import shutil
import argparse
import time
from datetime import datetime
import numpy as np
from scipy.io import wavfile
from scipy import signal


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


def _cargar_audio_referencia_canal3(secuencia_dir: str, raw_matrix: np.ndarray, fs_fallback: int):
    """
    Carga obligatoriamente la pista acústica del micrófono desde canal_3/grabacion.wav
    (o canal_3/ Crudo.wav) como verdad de campo (ground truth) de fonación.
    Si el WAV tiene distinta tasa de muestreo que la matriz sEMG, lo remuestrea o
    convierte los índices temporales en segundos.
    """
    ch3_dir = os.path.join(secuencia_dir, "canal_3")
    candidates = [
        os.path.join(ch3_dir, "grabacion.wav"),
        os.path.join(ch3_dir, " Crudo.wav"),
        os.path.join(ch3_dir, "Crudo.wav"),
    ]
    for wav_p in candidates:
        if os.path.isfile(wav_p):
            try:
                sr_mic, data_mic = wavfile.read(wav_p)
                if data_mic.ndim > 1:
                    data_mic = data_mic[:, 0]
                if data_mic.dtype == np.int16:
                    arr_mic = data_mic.astype(np.float64) / 32767.0
                else:
                    arr_mic = data_mic.astype(np.float64)
                arr_mic = arr_mic - np.median(arr_mic)
                return arr_mic, int(sr_mic), wav_p
            except Exception as e:
                print(f"[Aviso] No se pudo leer {wav_p}: {e}")

    # Fallback a la fila 3 de raw_matrix si existe Canal 3 en grabacion.csv
    if raw_matrix.shape[0] >= 4:
        arr_mic = raw_matrix[3, :].astype(np.float64).copy()
        arr_mic = arr_mic - np.median(arr_mic)
        return arr_mic, int(fs_fallback), "grabacion.csv[Canal 3]"

    return None, int(fs_fallback), None


def _calcular_envolvente_acustica_vocal(mic_sig: np.ndarray, fs_mic: int) -> np.ndarray:
    """
    Filtra la señal del micrófono en la banda vocal fundamental + primer formante
    (80 Hz - 900 Hz) para eliminar golpes de aire graves y clicks agudos del parlante,
    y calcula la envolvente suave combinada (Hilbert + RMS de 40 ms).
    """
    if mic_sig is None or len(mic_sig) < 64:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(mic_sig, dtype=np.float64)
    arr = arr - np.median(arr)
    nyq = 0.5 * float(fs_mic)
    low = max(60.0, min(85.0, nyq * 0.2))
    high = min(900.0, nyq * 0.92)
    if low < high:
        try:
            b, a = signal.butter(4, [low / nyq, high / nyq], btype="band")
            arr = signal.filtfilt(b, a, arr)
        except Exception:
            pass

    # Envolvente de energía RMS con ventana de 45 ms
    win_len = max(5, int(round(0.045 * fs_mic)))
    kernel = np.ones(win_len, dtype=np.float64) / float(win_len)
    env_rms = np.sqrt(np.convolve(arr ** 2, kernel, mode="same"))
    return env_rms


def _leer_eventos_vocales_estricto(secuencia_dir: str, fs: float, duracion_total_s: float):
    """
    Lee los eventos de pulsos de fonación descartando CON RIGUROSIDAD ABSOLUTA:
      - Cualquier fila con count_in > 0 (pulsos de cuenta regresiva "Prepárate 3, 2, 1")
      - Cualquier fila con principal != 1 (sub-pulsos rítmicos 2, 3, 4)
      - Cualquier fila sin vocal asignada (palabra vacía) o anterior al fin del ruido basal
      - Cualquier pulso marcado como invalido_manual == 1 en segmentos_dataset.csv
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

    bpm_default = float(meta.get("bpm") or (meta.get("metronomo", {}) or {}).get("bpm", 30.0))
    tau_default = 60.0 / max(1.0, bpm_default)
    noise_seconds = float(meta.get("noise_seconds") or (meta.get("metronomo", {}) or {}).get("noise_seconds", 3.0))

    invalid_ids = set()
    seg_by_idx = {}
    if os.path.exists(csv_seg):
        try:
            with open(csv_seg, "r", encoding="utf-8") as f:
                for sr in csv.DictReader(f):
                    pid = _safe_int(sr.get("pulso_id") or sr.get("pulso_idx"))
                    if pid is not None and pid > 0:
                        seg_by_idx[pid] = sr
                        if _safe_int(sr.get("invalido_manual"), 0) == 1:
                            invalid_ids.add(pid)
        except Exception:
            pass

    eventos_by_idx = {}

    # 1. Leer pulsos_metronomo.csv filtrando estrictamente principal == 1 y count_in == 0
    if os.path.exists(csv_pulsos):
        try:
            with open(csv_pulsos, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                principal_val = _safe_int(r.get("principal"), 1)
                count_in_val = _safe_int(r.get("count_in"), 0)
                sub_val = _safe_int(r.get("sub"), 0)
                if principal_val != 1 or count_in_val > 0 or sub_val != 0:
                    continue

                p_idx = _safe_int(r.get("idx") or r.get("pulso_idx"), 0)
                if p_idx is None or p_idx <= 0 or p_idx in invalid_ids:
                    continue

                vocal = str(r.get("palabra") or r.get("vocal") or "").strip().upper().replace("/", "")
                if vocal not in VOCALES_ESTANDAR:
                    continue

                sample_daq = _safe_int(r.get("sample_daq"))
                t_wav = _safe_float(r.get("t_wav_s"))
                t_daq = _safe_float(r.get("t_daq_s") or r.get("timestamp_relativo_s") or r.get("t_s"))
                offset_m = _safe_int(r.get("offset_muestras"), int(round(noise_seconds * fs)))

                if t_wav is None:
                    if sample_daq is not None and fs > 0:
                        t_wav = sample_daq / fs
                    elif t_daq is not None:
                        # Si t_daq_s empieza en 0 tras el ruido basal, sumarle el offset de ruido
                        t_wav = t_daq + (offset_m / fs if fs > 0 else noise_seconds)
                    else:
                        continue

                # Descartar cualquier pulso espurio que caiga dentro del ruido basal inicial
                if t_wav < max(0.2, noise_seconds - 0.25) or t_wav >= duracion_total_s - 0.15:
                    continue

                tau_s = _safe_float(r.get("intervalo_s") or r.get("tau_s"), tau_default)
                eventos_by_idx[p_idx] = {
                    "pulso_idx": p_idx,
                    "vocal": vocal,
                    "t_beat_s": float(t_wav),
                    "tau_s": float(tau_s),
                }
        except Exception as e:
            print(f"[Aviso] Error leyendo pulsos_metronomo.csv: {e}")

    # 2. Fallback / complemento desde metadata -> eventos_vocales_metronomo
    if not eventos_by_idx and meta:
        evs_meta = (
            meta.get("eventos_vocales_metronomo")
            or meta.get("vocal_cue_log")
            or (meta.get("locuciones", {}) or {}).get("eventos_vocales_metronomo", [])
        )
        if isinstance(evs_meta, list):
            for pos, ev in enumerate(evs_meta, start=1):
                p_idx = _safe_int(ev.get("pulso_vocal_idx") or ev.get("pulso_idx") or ev.get("idx"), pos)
                if p_idx in invalid_ids:
                    continue
                vocal = str(ev.get("palabra") or ev.get("vocal") or "").strip().upper().replace("/", "")
                if vocal not in VOCALES_ESTANDAR:
                    continue
                sample_daq = _safe_int(ev.get("sample_daq"))
                t_wav = _safe_float(ev.get("t_wav_s"))
                t_daq = _safe_float(ev.get("t_daq_s") or ev.get("t_s"))
                if t_wav is None:
                    if sample_daq is not None and fs > 0:
                        t_wav = sample_daq / fs
                    elif t_daq is not None:
                        t_wav = t_daq + noise_seconds
                    else:
                        continue
                if t_wav < max(0.2, noise_seconds - 0.25) or t_wav >= duracion_total_s - 0.15:
                    continue
                tau_s = _safe_float(ev.get("intervalo_s") or ev.get("tau_s"), tau_default)
                eventos_by_idx[p_idx] = {
                    "pulso_idx": p_idx,
                    "vocal": vocal,
                    "t_beat_s": float(t_wav),
                    "tau_s": float(tau_s),
                }

    # 3. Enriquecer con segmentos_dataset.csv si existe
    for p_idx, sr in seg_by_idx.items():
        if p_idx in invalid_ids:
            continue
        vocal = str(sr.get("vocal") or "").strip().upper().replace("/", "")
        t_beat = _safe_float(sr.get("t_beat_s"))
        s_beat = _safe_int(sr.get("sample_beat"))
        t_onset = _safe_float(sr.get("t_onset_s"))
        s_onset = _safe_int(sr.get("sample_onset_real"))
        if t_beat is None and s_beat is not None and fs > 0:
            t_beat = s_beat / fs
        if t_onset is None and s_onset is not None and fs > 0:
            t_onset = s_onset / fs
        if p_idx in eventos_by_idx:
            if vocal in VOCALES_ESTANDAR:
                eventos_by_idx[p_idx]["vocal"] = vocal
            if t_onset is not None and t_onset > noise_seconds:
                eventos_by_idx[p_idx]["t_onset_dataset_s"] = float(t_onset)
        elif t_beat is not None and vocal in VOCALES_ESTANDAR and t_beat >= max(0.2, noise_seconds - 0.25):
            eventos_by_idx[p_idx] = {
                "pulso_idx": p_idx,
                "vocal": vocal,
                "t_beat_s": float(t_beat),
                "t_onset_dataset_s": float(t_onset) if t_onset is not None else None,
                "tau_s": tau_default,
            }

    eventos = [eventos_by_idx[k] for k in sorted(eventos_by_idx.keys())]
    for i, ev in enumerate(eventos):
        if i + 1 < len(eventos):
            dt_next = eventos[i + 1]["t_beat_s"] - ev["t_beat_s"]
            if 0.40 <= dt_next <= 8.0:
                ev["tau_s"] = float(dt_next)
            ev["next_beat_s"] = float(eventos[i + 1]["t_beat_s"])
        else:
            ev["next_beat_s"] = float(duracion_total_s)

    return eventos, meta


def _cargar_matriz_cruda(secuencia_dir: str, meta: dict):
    """
    Carga la matriz multicanal cruda (n_canales, n_muestras), el vector de tiempo real
    y el sample_rate desde grabacion.csv (separando la columna 'Tiempo (s)') o desde
    canal_X/grabacion.wav o Crudo.wav.
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
                first_line = fh.readline().strip().lstrip("#").strip()
            cols_raw = [c.strip() for c in first_line.split(",")]
            cols_lower = [c.lower() for c in cols_raw]
            has_header = any(any(ch.isalpha() for ch in col) for col in cols_lower)

            raw_mat = np.loadtxt(csv_grab, delimiter=",", skiprows=1 if has_header else 0)
            if raw_mat.ndim == 1:
                raw_mat = raw_mat[:, np.newaxis]

            tiene_col_tiempo = False
            if has_header and cols_lower and any(tk in cols_lower[0] for tk in ("tiempo", "time")):
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
                return sig_mat, t_vec, int(round(fs))
            else:
                sig_mat = raw_mat.T
                t_vec = np.arange(sig_mat.shape[1], dtype=np.float64) / float(fs)
                return sig_mat, t_vec, int(round(fs))
        except Exception as e:
            print(f"[Aviso] Falló lectura de grabacion.csv ({e}), leyendo desde WAVs...")

    ch_dirs = sorted(
        [d for d in glob.glob(os.path.join(secuencia_dir, "canal_*")) if os.path.isdir(d)],
        key=lambda x: int(os.path.basename(x).split("_")[-1]) if os.path.basename(x).split("_")[-1].isdigit() else 0
    )
    canales_arrays = []
    for cd in ch_dirs:
        wav_cands = [os.path.join(cd, "grabacion.wav"), os.path.join(cd, " Crudo.wav"), os.path.join(cd, "Crudo.wav")]
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
        raise FileNotFoundError(f"No se encontró grabacion.csv ni archivos WAV en {secuencia_dir}")

    min_len = min(len(a) for a in canales_arrays)
    mat = np.vstack([a[:min_len] for a in canales_arrays])
    t_vec = np.arange(min_len, dtype=np.float64) / float(fs)
    return mat, t_vec, int(round(fs))


def _localizar_fonacion_acustica_canal3(
    env_mic: np.ndarray,
    fs_mic: int,
    raw_matrix: np.ndarray,
    fs_emg: int,
    t_beat_s: float,
    tau_s: float,
    next_beat_s: float,
    umbral_basal_mic: float,
) -> tuple:
    """
    Busca en la envolvente acústica de Canal 3 (micrófono) la emisión real de voz
    correspondiente al pulso t_beat_s.
    - Ventana de búsqueda: [t_beat_s - 0.20s, min(t_beat_s + 0.80*tau_s, next_beat_s - 0.15s)]
    - Retorna (t_acustico_centro_s, t_onset_acustico_s, detectado_por_mic)
    """
    t_search_ini = max(0.0, t_beat_s - 0.20)
    t_search_fin = min(next_beat_s - 0.12, t_beat_s + max(0.65, 0.82 * tau_s))
    if t_search_fin <= t_search_ini + 0.15:
        t_search_fin = t_search_ini + 0.50

    if env_mic is not None and len(env_mic) > 0 and fs_mic > 0:
        idx_m0 = max(0, min(len(env_mic) - 1, int(round(t_search_ini * fs_mic))))
        idx_m1 = max(idx_m0 + 1, min(len(env_mic), int(round(t_search_fin * fs_mic))))
        seg_m = env_mic[idx_m0:idx_m1]
        if len(seg_m) > 8:
            local_max = float(np.max(seg_m))
            local_floor = float(np.percentile(seg_m, 20))
            th_voz = max(umbral_basal_mic * 2.2, local_floor + 0.30 * (local_max - local_floor))
            if local_max > umbral_basal_mic * 1.8:
                idx_peak_rel = int(np.argmax(seg_m))
                # Primer cruce sobre el 35% de la amplitud neta del pico (onset de fonación)
                above = np.where(seg_m >= th_voz)[0]
                idx_onset_rel = int(above[0]) if len(above) > 0 else max(0, idx_peak_rel - int(0.08 * fs_mic))
                # Centro de masa de la energía acústica sobre el umbral (centroide de la vocal emitida)
                mask_voicing = seg_m >= th_voz
                if np.any(mask_voicing):
                    weights = seg_m * mask_voicing
                    idx_centroid_rel = float(np.sum(np.arange(len(seg_m)) * weights) / (np.sum(weights) + 1e-12))
                else:
                    idx_centroid_rel = float(idx_peak_rel)

                t_onset_acustico = (idx_m0 + idx_onset_rel) / float(fs_mic)
                t_centro_acustico = (idx_m0 + idx_centroid_rel) / float(fs_mic)
                return float(t_centro_acustico), float(t_onset_acustico), True

    # Fallback sobre la envolvente sEMG (Canales 0..2) si el micrófono estuvo desconectado
    n_emg = min(3, raw_matrix.shape[0])
    idx_e0 = max(0, min(raw_matrix.shape[1] - 1, int(round(t_search_ini * fs_emg))))
    idx_e1 = max(idx_e0 + 1, min(raw_matrix.shape[1], int(round(t_search_fin * fs_emg))))
    win_sm = max(5, int(round(0.050 * fs_emg)))
    kern = np.ones(win_sm, dtype=np.float64) / float(win_sm)
    env_emg = np.zeros(idx_e1 - idx_e0, dtype=np.float64)
    for c in range(n_emg):
        s_c = raw_matrix[c, idx_e0:idx_e1]
        env_emg += np.convolve(np.abs(s_c - np.median(s_c)), kern, mode="same")
    if len(env_emg) > 4:
        idx_p = int(np.argmax(env_emg))
        t_emg_peak = (idx_e0 + idx_p) / float(fs_emg)
        return float(t_emg_peak), float(max(t_search_ini, t_emg_peak - 0.10)), False

    return float(t_beat_s), float(t_beat_s), False


def extraer_pulsos_por_vocal(
    secuencia_dir: str,
    output_parent_dir: str = None,
    prefijo_prueba: str = None,
    sujeto_override: str = None,
    incluir_ruido_basal: bool = True,
    duracion_ventana_fija_s: float = None,
    centrar_en_pico_energia: bool = True,
    pre_activacion_s: float = 0.14,
    callback_log=print,
) -> dict:
    """
    Extrae todos los pulsos de una carpeta SecuenciaContinua_* anclando cada recorte
    a la emisión de voz real en Canal 3 (canal_3/grabacion.wav) y compensando la
    pre-activación mioeléctrica (~140 ms antes del onset acústico).
    """
    secuencia_dir = os.path.abspath(secuencia_dir)
    if not os.path.isdir(secuencia_dir):
        raise ValueError(f"No existe el directorio: {secuencia_dir}")

    if output_parent_dir is None:
        output_parent_dir = os.path.dirname(secuencia_dir)
    os.makedirs(output_parent_dir, exist_ok=True)

    callback_log(f"==> Analizando secuencia continua: {os.path.basename(secuencia_dir)}")
    raw_matrix, t_vec_orig, fs = _cargar_matriz_cruda(secuencia_dir, {})
    n_canales, n_samples = raw_matrix.shape
    duracion_total_s = float(t_vec_orig[-1]) if len(t_vec_orig) > 0 else (n_samples / float(fs))

    # 1. Cargar obligatoriamente el audio de referencia de Canal 3 (Micrófono)
    mic_sig, fs_mic, mic_source_name = _cargar_audio_referencia_canal3(secuencia_dir, raw_matrix, fs)
    env_mic = _calcular_envolvente_acustica_vocal(mic_sig, fs_mic) if mic_sig is not None else None

    # 2. Leer eventos filtrando estrictamente count_in == 0 y principal == 1
    eventos, meta = _leer_eventos_vocales_estricto(secuencia_dir, fs=float(fs), duracion_total_s=duracion_total_s)
    if not eventos:
        raise RuntimeError(
            "No se encontraron pulsos vocales válidos (principal==1, count_in==0) en la secuencia."
        )

    noise_seconds = float(meta.get("noise_seconds") or (meta.get("metronomo", {}) or {}).get("noise_seconds", 3.0))
    noise_samples = min(n_samples, max(int(0.5 * fs), int(round(noise_seconds * fs))))
    segmento_ruido = raw_matrix[:, :noise_samples] if incluir_ruido_basal else np.zeros((n_canales, int(1.0 * fs)))
    noise_dur_efectiva = segmento_ruido.shape[1] / float(fs)
    offset_ruido_samples = int(segmento_ruido.shape[1])

    # Umbral acústico basal del micrófono durante los segundos de silencio inicial
    umbral_basal_mic = 1e-4
    if env_mic is not None and len(env_mic) > int(0.5 * fs_mic):
        n_mic_noise = min(len(env_mic), int(round(noise_seconds * fs_mic)))
        umbral_basal_mic = float(np.percentile(env_mic[:n_mic_noise], 85)) + 1e-5

    sujeto = sujeto_override or meta.get("sujeto") or (meta.get("identificacion", {}) or {}).get("sujeto")
    prueba = prefijo_prueba or meta.get("prueba") or (meta.get("identificacion", {}) or {}).get("prueba")
    base_folder = os.path.basename(secuencia_dir)
    parts = base_folder.split("_")
    if not prueba:
        prueba = parts[1] if len(parts) >= 3 else "SecuenciaExtraida"
    if not sujeto:
        sujeto = parts[2] if len(parts) >= 3 else "Sujeto"

    eventos_por_vocal = {}
    for ev in eventos:
        v = ev["vocal"]
        if v in VOCALES_ESTANDAR:
            eventos_por_vocal.setdefault(v, []).append(ev)

    callback_log(
        f"==> Audio Ref: {os.path.basename(mic_source_name) if mic_source_name else 'sEMG'} ({fs_mic} Hz) | "
        f"Canales: {n_canales} @ {fs} Hz | Pulsos válidos (sin count-in): {len(eventos)} -> "
        + ", ".join(f"{k}:{len(v)}" for k, v in sorted(eventos_por_vocal.items()))
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

        pad_samples = int(round(0.12 * fs))
        baseline_por_canal = np.median(segmento_ruido, axis=1, keepdims=True)
        bloque_pad = np.repeat(baseline_por_canal, pad_samples, axis=1)

        n_anclados_mic = 0
        for idx_local, ev in enumerate(lista_evs, start=1):
            t_beat_nominal = float(ev["t_beat_s"])
            tau_ev = float(ev.get("tau_s", 2.0))
            next_beat_s = float(ev.get("next_beat_s", duracion_total_s))

            if centrar_en_pico_energia:
                t_acustico, t_onset_ac, ok_mic = _localizar_fonacion_acustica_canal3(
                    env_mic=env_mic,
                    fs_mic=fs_mic,
                    raw_matrix=raw_matrix,
                    fs_emg=fs,
                    t_beat_s=t_beat_nominal,
                    tau_s=tau_ev,
                    next_beat_s=next_beat_s,
                    umbral_basal_mic=umbral_basal_mic,
                )
                if ok_mic:
                    n_anclados_mic += 1
                # Anclar ventana incluyendo la pre-activación muscular (~140 ms antes del onset acústico)
                t_centro_emg = max(noise_seconds, t_acustico - 0.05)
                t_ini_emg = min(t_onset_ac - pre_activacion_s, t_centro_emg - 0.35 * min(2.2, tau_ev))
            else:
                t_acustico = t_beat_nominal
                t_onset_ac = t_beat_nominal
                t_centro_emg = t_beat_nominal
                t_ini_emg = t_beat_nominal - 0.35 * min(2.2, tau_ev)

            ancho_ventana = duracion_ventana_fija_s if (duracion_ventana_fija_s and duracion_ventana_fija_s > 0) else max(0.85, min(2.4, 0.90 * tau_ev))
            w_ini = max(noise_seconds * 0.8, min(t_ini_emg, t_centro_emg - 0.42 * ancho_ventana))
            w_fin = min(duracion_total_s, w_ini + ancho_ventana)

            idx_ini = int(np.searchsorted(t_vec_orig, w_ini, side="left"))
            idx_fin = int(np.searchsorted(t_vec_orig, w_fin, side="right"))
            idx_ini = max(0, min(n_samples - 1, idx_ini))
            idx_fin = max(idx_ini + int(0.30 * fs), min(n_samples, idx_fin))
            recorte = raw_matrix[:, idx_ini:idx_fin]

            # Posición exacta del pico acústico/fonatorio dentro del nuevo archivo concatenado
            offset_acustico_samples = int(np.clip(int(round(t_acustico * fs)) - idx_ini, int(0.10 * recorte.shape[1]), max(1, recorte.shape[1] - 1)))
            sample_daq_nuevo = int(cursor_muestras + offset_acustico_samples)
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
                "t_epoch": round(time.time(), 6),
                "t_perf": t_daq_nuevo,
                "t_daq_s": t_daq_nuevo,
                "t_wav_s": t_wav_nuevo,
                "sample_daq": sample_daq_nuevo,
                "offset_muestras": offset_ruido_samples,
                "bpm": int(round(60.0 / max(0.2, tau_clip))),
                "subdiv": 1,
                "intervalo_s": tau_clip,
                "jitter_activo": 1,
                "jitter_mode": "acoustic_aligned_ch3",
                "palabra": vocal,
            })

            segmentos_vocal.append(recorte)
            cursor_muestras += recorte.shape[1]
            segmentos_vocal.append(bloque_pad)
            cursor_muestras += bloque_pad.shape[1]

        matriz_vocal = np.concatenate(segmentos_vocal, axis=1)
        total_muestras_vocal = matriz_vocal.shape[1]
        tiempo_vocal = np.arange(total_muestras_vocal, dtype=np.float64) / float(fs)

        # 1. Guardar grabacion.csv con formato nativo exacto: "Tiempo (s),Canal 0,Canal 1,Canal 2,Canal 3"
        csv_out_path = os.path.join(vocal_dir, "grabacion.csv")
        datos_para_csv = np.hstack((tiempo_vocal.reshape(-1, 1), matriz_vocal.T))
        headers_str = "Tiempo (s)," + ",".join([f"Canal {i}" for i in range(n_canales)])
        np.savetxt(
            csv_out_path,
            datos_para_csv,
            delimiter=",",
            header=headers_str,
            comments="",
            fmt="%.5f",
        )

        # 2. Guardar pulsos_metronomo.csv con las 17 columnas estándar
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

        # 3. Guardar cada canal en canal_X/grabacion.wav + canal_X/ Crudo.wav + metadata.json
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
                "jitter_mode": "acoustic_aligned_ch3",
            }
            for r in registros_pulsos_nativos
        ]

        for ch in range(n_canales):
            ch_dir = os.path.join(vocal_dir, f"canal_{ch}")
            os.makedirs(ch_dir, exist_ok=True)
            wav_data = np.int16(np.clip(matriz_vocal[ch, :] / 5.0, -1.0, 1.0) * 32767)
            # Guardar ambos nombres (grabacion.wav y " Crudo.wav") para compatibilidad universal
            wavfile.write(os.path.join(ch_dir, "grabacion.wav"), fs, wav_data)
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
                "alineacion_acustica_ch3": True,
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
            "n_anclados_mic": n_anclados_mic,
            "duracion_s": round(total_muestras_vocal / float(fs), 2),
        }
        callback_log(
            f"  [OK] {folder_name}: {len(nuevos_timestamps_wav_s)} pulsos extraídos "
            f"({n_anclados_mic} anclados a voz en Canal 3, {total_muestras_vocal / float(fs):.1f}s) -> {vocal_dir}"
        )

    return carpetas_creadas


def lanzar_gui_extractor(initial_dir: str = None):
    """Interfaz gráfica PySide6 para extraer pulsos por vocal anclados a Canal 3."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    own_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        own_app = True

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Ñandú LSD — Extractor Acústico de Pulsos por Vocal (Canal 3 -> PCA & UMAP)")
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
        "<b style='color:#00ffcc; font-size:16px;'>Extractor Acústico de Pulsos por Vocal (Anclado a Canal 3 — Micrófono)</b><br>"
        "<span style='color:#9ca3af;'>Localiza la emisión fonatoria real en <code>canal_3/grabacion.wav</code>, descarta rigurosamente "
        "los pulsos de preparación (<code>count_in &gt; 0</code>) y genera las 5 carpetas "
        "(<code>A_...</code>, <code>E_...</code>, <code>I_...</code>, <code>O_...</code>, <code>U_...</code>) compatibles con PCA/UMAP.</span>"
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
    inp_sujeto.setPlaceholderText("Auto-detectado (ej. Sujeto1)")
    form.addRow("Nombre del Sujeto:", inp_sujeto)

    chk_refinar_pico = QtWidgets.QCheckBox("Anclar ventana al sonido real en Canal 3 (compensando 140 ms de pre-activación sEMG)")
    chk_refinar_pico.setChecked(True)
    form.addRow("Alineación Acústica:", chk_refinar_pico)

    chk_ventana_fija = QtWidgets.QCheckBox("Forzar ancho de ventana fijo por pulso (opcional)")
    spin_ventana = QtWidgets.QDoubleSpinBox()
    spin_ventana.setRange(0.5, 6.0)
    spin_ventana.setValue(1.8)
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
            resumen = "\n".join([
                f"• {v}: {info['nombre']} ({info['n_pulsos']} pulsos, {info['n_anclados_mic']} anclados a Ch3)"
                for v, info in res.items()
            ])
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
    parser = argparse.ArgumentParser(description="Extractor acústico de pulsos por vocal desde Secuencia Continua (anclado a Canal 3).")
    parser.add_argument("carpeta", nargs="?", default=None, help="Ruta a la carpeta SecuenciaContinua_*")
    parser.add_argument("--out", default=None, help="Directorio padre de salida (por defecto misma carpeta de fecha)")
    parser.add_argument("--prueba", default=None, help="Nombre de la prueba (ej. Prueba1)")
    parser.add_argument("--sujeto", default=None, help="Nombre del sujeto (ej. Sujeto1)")
    parser.add_argument("--ventana-fija", type=float, default=None, help="Ancho fijo de ventana en segundos (opcional)")
    parser.add_argument("--no-refinar-pico", action="store_true", help="Desactivar alineación acústica con Canal 3")
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
