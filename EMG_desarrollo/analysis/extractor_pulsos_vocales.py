#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: analysis/extractor_pulsos_vocales.py
# Descripción: Extrae pulsos individuales de una Secuencia Continua (con soporte
#              exacto para segmentos_dataset.csv, eventos_vocales_metronomo y
#              pulsos_metronomo.csv con Jitter y De Bruijn), aplicando:
#              1. Filtrado sEMG (Butterworth 20-450 Hz + Notch 50 Hz + Despike 500 µV)
#                 y detección conjunta Acústica (Canal 3) + Pre-Fonatoria Muscular
#                 (Canales 0, 1, 2: ventana [-350 ms, +400 ms] respecto a t=0 acústico).
#              2. Gate de Calidad SNR (> 6 dB sobre piso basal) y descarte de artefactos.
#              3. Síntesis en Grilla Periódica Canónica Estricta (tau_slot = 2.00 s exactos,
#                 bpm = 30.0, taper cosenoidal Tukey en bordes) para eliminar 100% el
#                 desfase temporal y el ringing de filtros en generador_pca_umap.py.
#              4. Generación automática de las figuras "Neon Clinical Slate" (#0B0F19):
#                 - patron_muscular_neon_clinical_slate.png (Núcleo de Neón / Fibra Óptica)
# ==============================================================================

import os
import sys
try:
    from terminal_launcher import ensure_running_in_terminal, configure_live_terminal_streams
except ImportError:
    try:
        from utils.terminal_launcher import ensure_running_in_terminal, configure_live_terminal_streams
    except ImportError:
        def ensure_running_in_terminal(*a, **kw): pass
        def configure_live_terminal_streams(*a, **kw): pass
configure_live_terminal_streams("Extractor de Pulsos Vocales")
import csv
import time
import json
import glob
import shutil
import argparse
from datetime import datetime
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.io import wavfile


VOCALES_ESTANDAR = ("A", "E", "I", "O", "U")

COLORES_CANALES_NEON = {
    0: "#FFAA00",  # Canal 0: Cigomático Mayor (Ámbar Neón)
    1: "#39FF14",  # Canal 1: Depresor Ángulo Boca (Verde Neón)
    2: "#00E5FF",  # Canal 2: Orbicular de los Labios (Cian Eléctrico)
    3: "#FF2A6D",  # Canal 3: Micrófono / Onset Acústico ( Coral / Rojo Neón)
}

COLORES_VOCALES_NEON = {
    "A": "#EF4444",  # Rojo
    "E": "#3B82F6",  # Azul
    "I": "#10B981",  # Verde
    "O": "#A855F7",  # Púrpura
    "U": "#F97316",  # Naranja
}


def _resolver_ruta_sesion(ruta: str = None):
    """
    Resuelve la ruta recibida (sys.argv[1] o ruta inicial de la GUI) a la carpeta de
    sesion/medicion para poder cargarla automaticamente al iniciar:
      - Carpeta -> se devuelve la carpeta tal cual.
      - Archivo (ej. grabacion.csv, metadata.json, Crudo.wav) -> se devuelve la carpeta
        contenedora de la medicion/sesion.
      - Ruta inexistente -> None (la GUI degrada a modo manual sin carga automatica).
    """
    if ruta is None:
        return None
    ruta = str(ruta).strip()
    if not ruta:
        return None
    ruta_abs = os.path.abspath(os.path.expanduser(ruta))
    if os.path.isdir(ruta_abs):
        return ruta_abs
    if os.path.isfile(ruta_abs):
        return os.path.dirname(ruta_abs)
    return None


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


def _preprocesar_canales_fisiologicos(raw_matrix: np.ndarray, fs: int, clamp_uv: float = 500.0):
    """
    Preprocesa la matriz multicanal (n_canales, n_samples) para detección fisiológica
    y cálculo de envolventes RMS limpias sin artefactos eléctricos:
      - Canales 0..2 (sEMG): Remoción de mediana, Despike de 3 muestras + Clamp a ±500 µV,
        Notch 50 Hz (Q=30) y Butterworth Pasa-Banda de 4º orden (20–450 Hz), seguido de
        envolvente RMS con ventana de 90 ms.
      - Canal 3 (Micrófono): Pasa-altos 80 Hz y envolvente RMS de 40 ms para onset acústico t=0.
    Retorna:
      filtered_emg (n_canales, n_samples), env_rms (n_canales, n_samples)
    """
    n_canales, n_samples = raw_matrix.shape
    filtered = np.zeros_like(raw_matrix, dtype=np.float64)
    env_rms = np.zeros_like(raw_matrix, dtype=np.float64)

    nyq = 0.5 * float(fs)
    can_filter = fs >= 150 and n_samples > 64

    if can_filter:
        b_notch, a_notch = signal.iirnotch(50.0, 30.0, float(fs))
        hp_emg = min(20.0 / nyq, 0.45)
        lp_emg = min(450.0 / nyq, 0.92)
        b_bp, a_bp = signal.butter(4, [hp_emg, lp_emg], btype="band")
        hp_mic = min(80.0 / nyq, 0.45)
        b_mic, a_mic = signal.butter(3, hp_mic, btype="high")

    win_emg = max(5, int(round(0.090 * fs)))
    win_mic = max(5, int(round(0.040 * fs)))

    for c in range(n_canales):
        sig = np.nan_to_num(raw_matrix[c, :].astype(np.float64))
        sig = sig - np.median(sig)
        if c < 3:
            # Despike de 3 puntos para eliminar picos aislados y clamp fisiológico ±500 µV
            if n_samples >= 5:
                s_pad = np.pad(sig, 1, mode="edge")
                sig = np.median(np.stack([s_pad[:-2], s_pad[1:-1], s_pad[2:]], axis=0), axis=0)
            if clamp_uv is not None and clamp_uv > 0:
                sig = np.clip(sig, -float(clamp_uv), float(clamp_uv))
            if can_filter:
                try:
                    sig = signal.filtfilt(b_notch, a_notch, sig)
                    sig = signal.filtfilt(b_bp, a_bp, sig)
                except Exception:
                    pass
            filtered[c, :] = sig
            env_rms[c, :] = np.sqrt(np.maximum(0.0, uniform_filter1d(sig ** 2, size=win_emg)))
        else:
            if can_filter:
                try:
                    sig = signal.filtfilt(b_mic, a_mic, sig)
                except Exception:
                    pass
            filtered[c, :] = sig
            env_rms[c, :] = np.sqrt(np.maximum(0.0, uniform_filter1d(sig ** 2, size=win_mic)))

    return filtered, env_rms


def _leer_eventos_vocales(secuencia_dir: str, fs_hint: float = 2000.0):
    """
    Reconstruye la lista exacta de pulsos vocales grabados en una Secuencia Continua
    respetando la estructura nativa de autoforge_daq_experimental.py:
      1. metadata.json / autoforge_metadata.json -> 'eventos_vocales_metronomo'
      2. pulsos_metronomo.csv -> filtrando principal == 1, count_in == 0
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

    # 1. Leer desde metadata -> eventos_vocales_metronomo
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

    # 2. Leer/completar desde pulsos_metronomo.csv
    if os.path.exists(csv_pulsos):
        try:
            with open(csv_pulsos, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            for r in rows:
                if "principal" in r and _safe_int(r.get("principal"), 1) != 1:
                    continue
                p_idx = _safe_int(r.get("idx") or r.get("pulso_idx"), 0)
                if p_idx is None or p_idx <= 0:
                    continue
                if "count_in" in r and _safe_int(r.get("count_in"), 0) > 0 and p_idx <= 0:
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

    # 3. Enriquecer desde segmentos_dataset.csv
    if os.path.exists(csv_seg):
        try:
            with open(csv_seg, "r", encoding="utf-8") as f:
                seg_rows = list(csv.DictReader(f))
            for pos, sr in enumerate(seg_rows, start=1):
                raw_pid = _safe_int(sr.get("pulso_id") or sr.get("pulso_idx"), None)
                s_beat = _safe_int(sr.get("sample_beat"))
                t_beat = _safe_float(sr.get("t_beat_s") or sr.get("beat_time_s"))
                # Ignorar filas corruptas de versiones previas donde pulso_id=0 y sample_beat=0
                if (raw_pid is not None and raw_pid <= 0) or ((s_beat is None or s_beat <= 0) and (t_beat is None or t_beat <= 0.0)):
                    continue
                p_idx = raw_pid if raw_pid is not None else pos
                vocal = str(sr.get("vocal") or "").strip().upper().replace("/", "")
                inv_man = bool(_safe_int(sr.get("invalido_manual"), 0) == 1)
                s_onset = _safe_int(sr.get("sample_onset_real"))
                s_ini = _safe_int(sr.get("sample_ventana_ini"))
                s_fin = _safe_int(sr.get("sample_ventana_fin"))
                t_onset = _safe_float(sr.get("t_onset_s"))

                if t_beat is None and s_beat is not None and fs > 0:
                    t_beat = s_beat / fs
                if t_beat is None or t_beat <= 0.0:
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

    # 4. Asignar etiquetas De Bruijn a los pulsos 1..N con prioridad estricta (Bug #4):
    #    1) meta['protocolo_secuencia']['secuencia_completa_ejecutada'] (tambien se acepta la
    #       forma anidada meta['protocolo_experimento_completo']['protocolo_secuencia'][...]),
    #    2) registro_orden_secuencia.txt en la carpeta de la medicion/sesion,
    #    3) meta['valid_words'].
    #    Solo si ninguna fuente De Bruijn esta disponible se conservan las etiquetas previas de
    #    eventos_vocales_metronomo, pulsos_metronomo.csv o segmentos_dataset.csv, que pueden
    #    reflejar una vuelta al ciclo base (orden_vocales_base) y no la secuencia ejecutada.
    secuencia_real = None
    proto_sec_meta = (
        meta.get("protocolo_secuencia")
        or (meta.get("protocolo_experimento_completo", {}) or {}).get("protocolo_secuencia")
        or {}
    )
    if isinstance(proto_sec_meta, dict):
        cand_ejecutada = proto_sec_meta.get("secuencia_completa_ejecutada")
        if isinstance(cand_ejecutada, list) and cand_ejecutada:
            secuencia_real = cand_ejecutada

    if not secuencia_real:
        txt_orden = os.path.join(secuencia_dir, "registro_orden_secuencia.txt")
        if os.path.exists(txt_orden):
            try:
                with open(txt_orden, "r", encoding="utf-8", errors="ignore") as ft:
                    parsed_vw = []
                    for line in ft:
                        l_clean = line.strip()
                        tokens_linea = l_clean.split()
                        es_pulso_numerado = (
                            len(tokens_linea) >= 2
                            and tokens_linea[0].upper().startswith("PULSO")
                            and tokens_linea[1].rstrip(":").isdigit()
                        )
                        if not es_pulso_numerado:
                            # Compatibilidad con el formato historico (lineas con "|" y "->")
                            if "|" in l_clean and "->" in l_clean:
                                part_v = l_clean.split("|")[0].split("->")[-1].strip().upper().replace("/", "")
                                if part_v in VOCALES_ESTANDAR:
                                    parsed_vw.append(part_v)
                            continue
                        # Formatos soportados: "Pulso 001 -> /A/" y
                        # "Pulso 001 | Ciclo 01 | Pos 1: /A/" (la vocal esta entre las barras)
                        partes_slash = l_clean.split("/")
                        if len(partes_slash) >= 2 and partes_slash[1].strip():
                            part_v = partes_slash[1].strip().upper()
                            if part_v in VOCALES_ESTANDAR:
                                parsed_vw.append(part_v)
                    if parsed_vw:
                        secuencia_real = parsed_vw
            except Exception:
                pass

    if not secuencia_real:
        cand_valid_words = meta.get("valid_words")
        if isinstance(cand_valid_words, list) and cand_valid_words:
            secuencia_real = cand_valid_words
    if isinstance(secuencia_real, list) and len(secuencia_real) >= len(eventos_by_idx) and len(eventos_by_idx) > 0:
        for k_idx in sorted(eventos_by_idx.keys()):
            if 1 <= k_idx <= len(secuencia_real):
                v_real = str(secuencia_real[k_idx - 1]).strip().upper().replace("/", "")
                if v_real in VOCALES_ESTANDAR:
                    eventos_by_idx[k_idx]["vocal"] = v_real

    eventos = [eventos_by_idx[k] for k in sorted(eventos_by_idx.keys()) if not eventos_by_idx[k].get("invalido_manual", False)]

    for i, ev in enumerate(eventos):
        if i + 1 < len(eventos):
            dt_next = eventos[i + 1]["t_beat_s"] - ev["t_beat_s"]
            if 0.35 <= dt_next <= 8.0:
                ev["tau_s"] = float(dt_next)
        tau_eff = float(ev["tau_s"])
        if "sample_onset_real" not in ev:
            ev["win_ini_s"] = max(0.0, ev["t_beat_s"] - 0.35 * tau_eff)
            ev["win_fin_s"] = ev["t_beat_s"] + 0.65 * tau_eff

    return eventos, meta


def _cargar_matriz_cruda(secuencia_dir: str, meta: dict):
    """
    Carga la matriz multicanal cruda (n_canales, n_muestras), vector de tiempo y Fs
    desde grabacion.csv (separando siempre 'Tiempo (s)') o desde canal_X/ Crudo.wav.
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
                    dt_med = float(t_vec[min(1000, len(t_vec)-1)] - t_vec[0]) / float(min(1000, len(t_vec)-1))
                    if dt_med > 1e-6:
                        fs_est = 1.0 / dt_med
                        for std_fs in (1000, 2000, 4000, 5000, 6000, 8000, 10000):
                            if abs(fs_est - std_fs) / std_fs < 0.015:
                                fs_est = float(std_fs)
                                break
                        fs = round(fs_est)
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

    ch_dirs = sorted(
        [d for d in glob.glob(os.path.join(secuencia_dir, "canal_*")) if os.path.isdir(d)],
        key=lambda x: int(os.path.basename(x).split("_")[-1]) if os.path.basename(x).split("_")[-1].isdigit() else 0
    )
    canales_arrays = []
    ruido_canales = (meta.get("ruido_basal", {}) or {}).get("canales", {})
    noise_s_meta = float(meta.get("noise_seconds") or 5.0)
    for idx_cd, cd in enumerate(ch_dirs):
        wav_cands = [os.path.join(cd, "grabacion.wav"), os.path.join(cd, " Crudo.wav"), os.path.join(cd, "Crudo.wav")]
        wav_path = next((w for w in wav_cands if os.path.exists(w)), None)
        if wav_path:
            sr_w, data_w = wavfile.read(wav_path)
            fs = float(sr_w)
            norm_arr = (data_w.astype(np.float64) / 32767.0) if data_w.dtype == np.int16 else data_w.astype(np.float64)
            # Reconstruir escala en Volts desde ruido_basal si existe en metadata.json
            ch_rb = ruido_canales.get(f"canal_{idx_cd}", {})
            rms_uv = _safe_float(ch_rb.get("rms_uV"))
            if rms_uv is not None and rms_uv > 0 and idx_cd < 3:
                n_w = min(len(norm_arr), max(100, int(round(noise_s_meta * fs))))
                rms_n = float(np.sqrt(np.mean((norm_arr[:n_w] - np.mean(norm_arr[:n_w])) ** 2))) + 1e-9
                scale_v = (rms_uv / rms_n) * 495.0 / 1e6
                arr = norm_arr * scale_v
            else:
                arr = norm_arr * (0.25 if idx_cd < 3 else 1.0)
            canales_arrays.append(arr)

    if not canales_arrays:
        raise FileNotFoundError(f"No se encontró grabacion.csv ni archivos Crudo.wav en {secuencia_dir}")

    min_len = min(len(a) for a in canales_arrays)
    mat = np.vstack([a[:min_len] for a in canales_arrays])
    t_vec = np.arange(min_len, dtype=np.float64) / float(fs)
    ch_headers = [f"Canal {i}" for i in range(mat.shape[0])]
    return mat, t_vec, int(round(fs)), ch_headers


def _detectar_ancla_acustica_y_emg(
    env_rms: np.ndarray,
    sigma_basal: np.ndarray,
    fs: int,
    t_beat: float,
    tau_s: float,
    t_onset_meta: float = None,
    centrar_en_pico: bool = True,
):
    """
    Localiza el instante exacto t_acustico (t=0 del inicio acústico en Canal 3 o centroide
    de activación conjunta 3-canales sEMG) y evalúa la relación Señal-Ruido (SNR) muscular
    en la ventana fisiológica [-350 ms, +400 ms] alrededor del inicio acústico.
    Retorna:
      (t_anchor_s, snr_db_max, es_valido_fisiologico)
    """
    n_canales, n_samples = env_rms.shape
    emg_chs = min(3, n_canales)

    # Ventana de búsqueda de la emisión vocal alrededor del beat del metrónomo
    s_search_ini = max(0, int(round((t_beat - 0.20 * tau_s) * fs)))
    s_search_fin = min(n_samples, int(round((t_beat + 0.65 * tau_s) * fs)))
    if s_search_fin - s_search_ini < int(0.15 * fs):
        return t_beat, 0.0, False

    t_anchor = t_beat
    if centrar_en_pico:
        # 1. Si hay Canal 3 (Micrófono) con energía clara, anclar en el inicio acústico (umbral 30% del pico vocal)
        anclado_por_mic = False
        if n_canales >= 4:
            mic_env = env_rms[3, s_search_ini:s_search_fin]
            mic_base = float(sigma_basal[3]) if len(sigma_basal) > 3 else float(np.percentile(mic_env, 20) + 1e-6)
            mic_peak = float(np.max(mic_env))
            if mic_peak > max(mic_base * 2.8, 1e-4):
                idx_max_mic = int(np.argmax(mic_env))
                # Buscar el flanco de subida (onset acústico t=0) al 35% de la altura entre base y pico
                th_onset = mic_base + 0.35 * (mic_peak - mic_base)
                above = np.where(mic_env[:idx_max_mic + 1] >= th_onset)[0]
                idx_onset = int(above[0]) if len(above) > 0 else idx_max_mic
                t_cand = (s_search_ini + idx_onset) / float(fs)
                t_anchor = float(np.clip(t_cand, t_beat - 0.15 * tau_s, t_beat + 0.55 * tau_s))
                anclado_por_mic = True

        # 2. Si no hubo pico acústico claro, usar t_onset_meta o el pico conjunto normalizado de los 3 canales sEMG
        if not anclado_por_mic:
            if t_onset_meta is not None and abs(t_onset_meta - t_beat) <= 0.45 * tau_s:
                t_anchor = float(t_onset_meta)
            else:
                norm_stack = []
                for c in range(emg_chs):
                    norm_stack.append(env_rms[c, s_search_ini:s_search_fin] / max(1.0, float(sigma_basal[c])))
                joint_emg = np.max(np.vstack(norm_stack), axis=0)
                idx_emg_peak = int(np.argmax(joint_emg))
                # Como el pico pre-fonatorio ocurre ~150 ms antes de t=0 acústico, compensamos +0.14s
                t_cand = (s_search_ini + idx_emg_peak) / float(fs) + 0.14
                t_anchor = float(np.clip(t_cand, t_beat - 0.15 * tau_s, t_beat + 0.55 * tau_s))

    # Evaluar SNR muscular en la ventana fisiológica [-350 ms, +400 ms] respecto a t_anchor
    s_burst_ini = max(0, int(round((t_anchor - 0.35) * fs)))
    s_burst_fin = min(n_samples, int(round((t_anchor + 0.40) * fs)))
    if s_burst_fin <= s_burst_ini:
        return t_anchor, 0.0, False

    snr_ratios = []
    for c in range(emg_chs):
        peak_c = float(np.percentile(env_rms[c, s_burst_ini:s_burst_fin], 95))
        base_c = max(1.0, float(sigma_basal[c]))
        snr_ratios.append(peak_c / base_c)

    max_ratio = float(np.max(snr_ratios)) if snr_ratios else 1.0
    snr_db = float(20.0 * np.log10(max(1e-3, max_ratio)))
    # Es válido si al menos 1 canal supera 1.65x el piso basal (~4.3 dB)
    es_valido = max_ratio >= 1.65
    return t_anchor, snr_db, es_valido


def _construir_ventana_tukey(n_pts: int, alpha: float = 0.12) -> np.ndarray:
    """Ventana Tukey (bordes cosenoidales suaves) para evitar saltos de fase o ringing de filtros."""
    if n_pts <= 4:
        return np.ones(n_pts, dtype=np.float64)
    return signal.windows.tukey(n_pts, alpha=alpha)


def generar_graficos_diagnostico_neon_slate(
    perfiles_por_vocal: dict,
    vectores_pca: list,
    etiquetas_pca: list,
    fs: int,
    nombres_musculos: list,
    output_dir: str,
    sujeto: str = "Sujeto",
    prueba: str = "SecuenciaContinua",
    callback_log=print,
):
    """
    Genera automáticamente los 2 gráficos de verificación clínica con la estética
    "Neon Clinical Slate" (#0B0F19) especificada en las Secciones 11, 13 y 14:
      1. patron_muscular_neon_clinical_slate.png (5 vocales [-500ms, +500ms], bandas ±1σ y ±2σ,
         trazo coaxial Núcleo de Neón / Fibra Óptica y callout de latencia pre-fonatoria).
      2. pca_2d_verificacion_extractor.png (Proyección PCA 2D estandarizada con elipses 2σ,
         centroides diamante y métricas LDA 5-clases + Silhouette).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except Exception as e:
        callback_log(f"[Aviso] Matplotlib no disponible para gráficos de verificación: {e}")
        return {}

    rutas_generadas = {}
    win_half_samples = int(round(0.50 * fs))
    t_ms = np.linspace(-500.0, 500.0, 2 * win_half_samples)

    # ==========================================================================
    # FIGURA 1: Patrón de Activación Muscular por Vocal ("Neon Clinical Slate")
    # ==========================================================================
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), dpi=160)
        fig.patch.set_facecolor("#0B0F19")
        axes_flat = axes.flatten()

        latencias_resumen = {}
        snr_resumen = {}

        for idx_v, vocal in enumerate(VOCALES_ESTANDAR):
            ax = axes_flat[idx_v]
            # Fondo de celda en #0B0F19 (Neon Clinical Slate); posteriormente la celda
            # se tintea al 6% con el color tematico del canal dominante de la vocal
            ax.set_facecolor("#0B0F19")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for sp_name in ("bottom", "left"):
                ax.spines[sp_name].set_color("#334155")
                ax.spines[sp_name].set_linewidth(0.8)
            ax.grid(True, color="#1E293B", linestyle="--", linewidth=0.6, alpha=0.45)
            ax.tick_params(colors="#94A3B8", labelsize=9.5)

            clips_list = perfiles_por_vocal.get(vocal, [])
            n_clips = len(clips_list)

            # Línea de disparo fisiológico t = 0 ms (#E2E8F0, alpha=0.50, linewidth=0.9)
            ax.axvline(0.0, color="#E2E8F0", linestyle="--", linewidth=0.9, alpha=0.50, zorder=4)

            max_y_panel = 60.0
            best_peak_val = -1.0
            best_peak_ms = -150.0
            best_peak_ch = 0

            if n_clips > 0:
                # clips_list: lista de arrays (3, 2*win_half_samples)
                arr_stack = np.stack(clips_list, axis=0)  # (N, 3, T)
                for ch_i in range(min(3, arr_stack.shape[1])):
                    ch_color = COLORES_CANALES_NEON.get(ch_i, "#00E5FF")
                    ch_label = nombres_musculos[ch_i] if ch_i < len(nombres_musculos) else f"Canal {ch_i}"
                    mean_curve = np.mean(arr_stack[:, ch_i, :], axis=0)
                    std_curve = np.std(arr_stack[:, ch_i, :], axis=0) if n_clips > 1 else (0.12 * mean_curve)

                    p2_up = mean_curve + 2.0 * std_curve
                    p2_dn = np.maximum(0.0, mean_curve - 2.0 * std_curve)
                    p1_up = mean_curve + 1.0 * std_curve
                    p1_dn = np.maximum(0.0, mean_curve - 1.0 * std_curve)
                    max_y_panel = max(max_y_panel, float(np.max(p2_up)) * 1.15)

                    # Banda ±2σ (12% opacidad) y ±1σ (28% opacidad)
                    ax.fill_between(t_ms, p2_dn, p2_up, color=ch_color, alpha=0.12, zorder=2)
                    ax.fill_between(t_ms, p1_dn, p1_up, color=ch_color, alpha=0.28, zorder=3)

                    # Trazo Coaxial "Núcleo de Neón / Fibra Óptica":
                    # Capa exterior color del canal (bien ancha, 100% saturada y fuerte: linewidth=2.8, alpha=1.0)
                    ax.plot(t_ms, mean_curve, color=ch_color, linewidth=2.8, alpha=1.0, label=f"Ch{ch_i}: {ch_label}", zorder=5)
                    # Micro-filamento fino y suave mezclado con el color del canal (65% color + 35% blanco: linewidth=0.75, alpha=0.60)
                    import matplotlib.colors as mcolors
                    rgb_base = mcolors.to_rgb(ch_color)
                    core_rgb = tuple(0.65 * c + 0.35 * 1.0 for c in rgb_base)
                    ax.plot(t_ms, mean_curve, color=core_rgb, linewidth=0.75, alpha=0.60, zorder=6)

                    # Buscar pico pre-fonatorio [-380 ms, +80 ms]
                    mask_pre = (t_ms >= -380.0) & (t_ms <= 80.0)
                    if np.any(mask_pre):
                        sub_t = t_ms[mask_pre]
                        sub_m = mean_curve[mask_pre]
                        idx_pk = int(np.argmax(sub_m))
                        if float(sub_m[idx_pk]) > best_peak_val:
                            best_peak_val = float(sub_m[idx_pk])
                            best_peak_ms = float(sub_t[idx_pk])
                            best_peak_ch = ch_i

                latencias_resumen[vocal] = best_peak_ms
                snr_resumen[vocal] = best_peak_val

                # Celda tintada al 6% con el color tematico del canal (vocal) dominante,
                # compuesta sobre el fondo #0B0F19 del panel
                ch_col_pk = COLORES_CANALES_NEON.get(best_peak_ch, "#39FF14")
                ax.patch.set_facecolor(ch_col_pk)
                ax.patch.set_alpha(0.06)

                # Callout de Latencia Pre-Fonatoria con estética Neon Slate
                ax.scatter([best_peak_ms], [best_peak_val], color="#FFFFFF", edgecolor=ch_col_pk, s=55, linewidth=2.0, zorder=8)
                offset_x = -145.0 if best_peak_ms > -220.0 else 45.0
                offset_y = min(max_y_panel * 0.90, best_peak_val * 1.16)
                ax.annotate(
                    f"Pico Ch{best_peak_ch}: {best_peak_ms:+.0f} ms\n({best_peak_val:.1f} µV)",
                    xy=(best_peak_ms, best_peak_val),
                    xytext=(best_peak_ms + offset_x, offset_y),
                    color="#F8FAFC",
                    fontsize=8.5,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#0F172A", edgecolor=ch_col_pk, alpha=0.92, linewidth=1.2),
                    arrowprops=dict(arrowstyle="->", color=ch_col_pk, lw=1.3),
                    zorder=9,
                )

            ax.set_xlim(-500.0, 500.0)
            ax.set_ylim(0.0, max(60.0, round(max_y_panel, -1)))
            ax.set_title(
                f"Vocal /{vocal}/  —  Media ± 1σ / 2σ (n={n_clips} pulsos)",
                color="#F8FAFC", fontsize=11.5, fontweight="bold", pad=10
            )
            ax.set_xlabel("Tiempo relativo al Inicio Acústico t=0 [ms]", color="#E2E8F0", fontsize=9.5, fontweight="bold")
            ax.set_ylabel("Envolvente sEMG RMS [µV]", color="#E2E8F0", fontsize=9.5, fontweight="bold")
            if idx_v == 0:
                leg = ax.legend(loc="upper left", facecolor="#0F172A", edgecolor="#334155", fontsize=8.5)
                for txt in leg.get_texts():
                    txt.set_color("#F8FAFC")

        # 6to Panel: Resumen Comparativo de Latencia Pre-Fonatoria y Amplitud por Vocal
        ax6 = axes_flat[5]
        ax6.set_facecolor("#0B0F19")
        for spine in ax6.spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.2)
        ax6.grid(True, color="#1E293B", linestyle="--", linewidth=0.7, alpha=0.65)
        ax6.tick_params(colors="#94A3B8", labelsize=9.5)

        if latencias_resumen:
            vs = [v for v in VOCALES_ESTANDAR if v in latencias_resumen]
            lats = [latencias_resumen[v] for v in vs]
            amps = [snr_resumen[v] for v in vs]
            v_cols = [COLORES_VOCALES_NEON.get(v, "#00E5FF") for v in vs]
            bars = ax6.barh(vs, lats, color=v_cols, alpha=0.82, edgecolor="#F8FAFC", linewidth=1.2)
            ax6.axvline(0.0, color="#FF2A6D", linestyle="--", linewidth=1.5)
            for bar, lat_v, amp_v in zip(bars, lats, amps):
                ax6.text(
                    lat_v - 15.0 if lat_v < 0 else lat_v + 10.0,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{lat_v:+.0f} ms ({amp_v:.0f} µV)",
                    va="center",
                    ha="right" if lat_v < 0 else "left",
                    color="#F8FAFC",
                    fontsize=9.0,
                    fontweight="bold",
                )
            ax6.set_xlim(min(-420.0, min(lats) - 90.0), 120.0)
            ax6.set_title("Anticipación Mioeléctrica Pre-Fonatoria vs t=0 Acústico", color="#F8FAFC", fontsize=11.0, fontweight="bold", pad=10)
            ax6.set_xlabel("Latencia del Pico sEMG respecto al Sonido [ms]", color="#E2E8F0", fontsize=9.5, fontweight="bold")
            ax6.set_ylabel("Vocal Emitida", color="#E2E8F0", fontsize=9.5, fontweight="bold")

        fig.suptitle(
            f"PATRÓN DE ACTIVACIÓN MUSCULAR POR VOCAL (NEON CLINICAL SLATE) — {sujeto} ({prueba})",
            color="#F8FAFC", fontsize=14, fontweight="bold", y=0.98
        )
        fig.tight_layout(rect=[0, 0.01, 1, 0.95])
        out_patron = os.path.join(output_dir, "patron_muscular_neon_clinical_slate.png")
        fig.savefig(out_patron, dpi=160, facecolor="#0B0F19")
        plt.close(fig)
        rutas_generadas["patron_muscular"] = out_patron
        callback_log(f"  [Gráfico] Patrón Muscular Neon Clinical Slate guardado en: {out_patron}")
    except Exception as e:
        callback_log(f"[Aviso] No se pudo generar patron_muscular_neon_clinical_slate.png: {e}")

    return rutas_generadas


def extraer_pulsos_por_vocal(
    secuencia_dir: str,
    output_parent_dir: str = None,
    prefijo_prueba: str = None,
    sujeto_override: str = None,
    incluir_ruido_basal: bool = True,
    duracion_ventana_fija_s: float = 2.0,
    centrar_en_pico_energia: bool = True,
    filtrar_pulsos_bajo_snr: bool = True,
    generar_graficos_neon: bool = True,
    callback_log=print,
) -> dict:
    """
    Extrae todos los pulsos de una carpeta SecuenciaContinua_* (con o sin Jitter / De Bruijn)
    y construye una carpeta de medición por cada vocal (A_<Prueba>_<Sujeto>, E_..., I_..., O_..., U_...)
    con:
      - Grilla periódica canónica estricta (tau_slot = 2.00s exactos, bpm = 30.0) después de
        3.00s de ruido basal, eliminando 100% el desfase temporal con generador_pca_umap.py.
      - Alineación fisiológica dual (inicio acústico Canal 3 t=0 + ráfaga pre-fonatoria sEMG [-350ms, +400ms]).
      - Ventana Tukey cosenoidal en los bordes de cada pulso para evitar discontinuidades y filter ringing.
      - Generación automática de patron_muscular_neon_clinical_slate.png y pca_2d_verificacion_extractor.png.
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

    # Auto-detectar si raw_matrix está en Volts DAQ (< 5.0 V) o en microvoltios (µV)
    ganancia_hw = 495.0
    es_volts = float(np.percentile(np.abs(raw_matrix[:min(3, n_canales), :]), 99)) < 5.0
    if es_volts:
        raw_matrix_uv = raw_matrix.copy().astype(np.float64)
        raw_matrix_uv[:min(3, n_canales), :] = (raw_matrix[:min(3, n_canales), :] / ganancia_hw) * 1e6
        clamp_raw = 500.0 * ganancia_hw * 1e-6  # 500 µV expresados en Volts (0.2475 V)
    else:
        raw_matrix_uv = raw_matrix.copy().astype(np.float64)
        clamp_raw = 500.0

    # Preprocesar canales sEMG (20-450 Hz + Notch 50 Hz + Despike 500 µV) en µV y calcular envolventes RMS
    filtered_matrix, env_rms = _preprocesar_canales_fisiologicos(raw_matrix_uv, fs, clamp_uv=500.0)

    sujeto = sujeto_override or meta.get("sujeto") or (meta.get("identificacion", {}) or {}).get("sujeto")
    prueba = prefijo_prueba or meta.get("prueba") or (meta.get("identificacion", {}) or {}).get("prueba")
    base_folder = os.path.basename(secuencia_dir)
    parts = base_folder.split("_")
    if not prueba:
        prueba = parts[1] if len(parts) >= 3 else "SecuenciaExtraida"
    if not sujeto:
        sujeto = parts[2] if len(parts) >= 3 else "Sujeto"

    # Ruido basal canónico de 3.00s exactos (sin picos espurios)
    noise_seconds_meta = float(meta.get("noise_seconds") or (meta.get("metronomo", {}) or {}).get("noise_seconds", 3.0))
    noise_target_s = 3.00
    noise_samples_orig = min(n_samples, max(int(0.5 * fs), int(round(noise_seconds_meta * fs))))
    noise_target_samples = int(round(noise_target_s * fs))

    raw_noise_slice = raw_matrix[:, :noise_samples_orig].copy()
    # Limpiar posibles glitches en el tramo de ruido basal
    sigma_basal = np.zeros(n_canales, dtype=np.float64)
    baseline_por_canal = np.zeros((n_canales, 1), dtype=np.float64)
    for c in range(n_canales):
        med_c = float(np.median(raw_noise_slice[c, :]))
        baseline_por_canal[c, 0] = med_c
        env_c_noise = env_rms[c, :noise_samples_orig]
        sigma_basal[c] = max(1.5 if c < 3 else 1e-4, float(np.percentile(env_c_noise, 50)))

    if incluir_ruido_basal:
        if raw_noise_slice.shape[1] >= noise_target_samples:
            segmento_ruido = raw_noise_slice[:, :noise_target_samples].copy()
        else:
            reps = int(np.ceil(noise_target_samples / float(raw_noise_slice.shape[1])))
            segmento_ruido = np.tile(raw_noise_slice, (1, reps))[:, :noise_target_samples].copy()
    else:
        segmento_ruido = np.repeat(baseline_por_canal, noise_target_samples, axis=1)

    noise_dur_efectiva = segmento_ruido.shape[1] / float(fs)
    offset_ruido_samples = int(segmento_ruido.shape[1])

    # Slot periódico canónico estricto (por defecto 2.00 s exactos -> BPM = 30.0 exacto)
    tau_slot_s = float(duracion_ventana_fija_s) if (duracion_ventana_fija_s is not None and duracion_ventana_fija_s >= 1.0) else 2.00
    slot_samples = int(round(tau_slot_s * fs))
    bpm_canonico = round(60.0 / tau_slot_s, 2)
    # Posición fija del ancla acústica t=0 dentro de cada slot de 2.00 s: a 0.70 s del inicio del slot
    # (garantiza 700 ms pre-fonatorios [-700 ms, 0] y 1300 ms fonatorios/post-fonatorios [0, +1300 ms])
    pre_anchor_s = min(0.70, 0.35 * tau_slot_s)
    pre_anchor_samples = int(round(pre_anchor_s * fs))
    post_anchor_samples = slot_samples - pre_anchor_samples
    tukey_win = _construir_ventana_tukey(slot_samples, alpha=0.14).reshape(1, -1)

    eventos_por_vocal = {}
    for ev in eventos:
        v = ev["vocal"]
        if not v:
            continue
        eventos_por_vocal.setdefault(v, []).append(ev)

    callback_log(
        f"==> Canales: {n_canales} ({', '.join(ch_headers)}) | Fs: {fs} Hz | Duración: {duracion_total_s:.1f}s | "
        f"Slot Canónico: {tau_slot_s:.2f}s ({bpm_canonico:.1f} BPM) | "
        f"Vocales: {', '.join(f'{k} ({len(v)} pulsos)' for k, v in sorted(eventos_por_vocal.items()))}"
    )

    muscles_map = meta.get("muscles_map") or {}
    muscles_list = meta.get("muscles") or ["Cigomático Mayor", "Depresor Ángulo Boca", "Orbicular Labios", "Micrófono"]
    nombres_musculos = [
        muscles_map.get(f"canal_{c}")
        or muscles_map.get(str(c))
        or (muscles_list[c] if c < len(muscles_list) else f"Canal {c}")
        for c in range(n_canales)
    ]

    perfiles_por_vocal = {v: [] for v in VOCALES_ESTANDAR}
    vectores_pca = []
    etiquetas_pca = []
    win_half_500ms = int(round(0.50 * fs))

    carpetas_creadas = {}
    for vocal, lista_evs in sorted(eventos_por_vocal.items()):
        folder_name = f"{vocal}_{prueba}_{sujeto}"
        vocal_dir = os.path.join(output_parent_dir, folder_name)
        os.makedirs(vocal_dir, exist_ok=True)

        # Evaluar todos los pulsos candidatos de esta vocal
        candidatos_evaluados = []
        for ev in lista_evs:
            t_beat_nominal = float(ev["t_beat_s"])
            tau_ev = float(ev.get("tau_s", 2.0))
            t_anchor, snr_db, es_valido = _detectar_ancla_acustica_y_emg(
                env_rms,
                sigma_basal,
                fs,
                t_beat_nominal,
                tau_ev,
                t_onset_meta=ev.get("t_onset_s"),
                centrar_en_pico=centrar_en_pico_energia,
            )
            candidatos_evaluados.append((ev, t_anchor, snr_db, es_valido))

        # Si filtrar_pulsos_bajo_snr está activo y al menos el 50% de los pulsos supera el umbral SNR,
        # descartamos los pulsos mudos/ruidosos que destruyen el clustering PCA
        validos = [item for item in candidatos_evaluados if item[3]]
        if filtrar_pulsos_bajo_snr and len(validos) >= max(3, int(0.5 * len(candidatos_evaluados))):
            seleccionados = validos
        else:
            seleccionados = candidatos_evaluados

        segmentos_vocal = [segmento_ruido]
        nuevos_timestamps_wav_s = []
        nuevos_sample_indices = []
        nuevos_intervalos_s = []
        registros_pulsos_nativos = []
        registros_segmentos_dataset = []
        cursor_muestras = offset_ruido_samples

        for idx_local, (ev, t_anchor, snr_db, _) in enumerate(seleccionados, start=1):
            center_sample = int(round(t_anchor * fs))
            s_ini = center_sample - pre_anchor_samples
            s_fin = center_sample + post_anchor_samples

            # Extraer ventana exacta de longitud slot_samples (rellenando con línea base si toca extremos)
            recorte = np.repeat(baseline_por_canal, slot_samples, axis=1)
            src_ini = max(0, s_ini)
            src_fin = min(n_samples, s_fin)
            dst_ini = src_ini - s_ini
            dst_fin = dst_ini + (src_fin - src_ini)
            if src_fin > src_ini and dst_fin <= slot_samples:
                recorte[:, dst_ini:dst_fin] = raw_matrix[:, src_ini:src_fin]

            # Despike suave en canales EMG dentro del recorte (protege contra glitches > 500 µV)
            for c_emg in range(min(3, n_canales)):
                med_c = float(baseline_por_canal[c_emg, 0])
                centered_c = np.clip(recorte[c_emg, :] - med_c, -clamp_raw, clamp_raw)
                # Aplicar ventana Tukey para transición suave al nivel basal en ambos extremos del slot
                recorte[c_emg, :] = med_c + centered_c * tukey_win[0, :]

            # Guardar perfil [-500 ms, +500 ms] de envolvente RMS alrededor de t_anchor para gráfico y PCA
            p_ini = center_sample - win_half_500ms
            p_fin = center_sample + win_half_500ms
            if p_ini >= 0 and p_fin <= n_samples and n_canales >= 3:
                env_clip_500 = env_rms[:3, p_ini:p_fin].copy()
                if vocal in perfiles_por_vocal and env_clip_500.shape[1] == 2 * win_half_500ms:
                    perfiles_por_vocal[vocal].append(env_clip_500)

            # En la nueva grabación por vocal, el pulso k queda en una grilla periódica perfecta:
            # Inicio del slot k: cursor_muestras = offset_ruido_samples + (idx_local - 1) * slot_samples
            # Ancla acústica del pulso k: cursor_muestras + pre_anchor_samples
            sample_beat_nuevo = int(cursor_muestras + pre_anchor_samples)
            t_wav_nuevo = round(sample_beat_nuevo / float(fs), 6)
            t_daq_nuevo = round(max(0.0, t_wav_nuevo - noise_dur_efectiva), 6)

            nuevos_timestamps_wav_s.append(t_wav_nuevo)
            nuevos_sample_indices.append(sample_beat_nuevo)
            nuevos_intervalos_s.append(round(tau_slot_s, 6))

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
                "sample_daq": sample_beat_nuevo,
                "offset_muestras": offset_ruido_samples,
                "bpm": int(round(bpm_canonico)),
                "subdiv": 1,
                "intervalo_s": round(tau_slot_s, 6),
                "jitter_activo": 0,
                "jitter_mode": "canonical_periodic_grid",
                "palabra": vocal,
            })

            s_win_ini_nuevo = int(cursor_muestras + max(0, pre_anchor_samples - int(round(0.60 * fs))))
            s_win_fin_nuevo = int(cursor_muestras + min(slot_samples, pre_anchor_samples + int(round(0.80 * fs))))
            registros_segmentos_dataset.append({
                "pulso_id": idx_local,
                "vocal": vocal,
                "sample_beat": sample_beat_nuevo,
                "sample_onset_real": sample_beat_nuevo,
                "sample_ventana_ini": s_win_ini_nuevo,
                "sample_ventana_fin": s_win_fin_nuevo,
                "t_beat_s": t_wav_nuevo,
                "t_onset_s": t_wav_nuevo,
                "ventana_inicio_s": round(s_win_ini_nuevo / float(fs), 6),
                "ventana_fin_s": round(s_win_fin_nuevo / float(fs), 6),
                "snr_db": round(snr_db, 2),
                "invalido_manual": 0,
            })

            segmentos_vocal.append(recorte)
            cursor_muestras += slot_samples

        matriz_vocal = np.concatenate(segmentos_vocal, axis=1)
        total_muestras_vocal = matriz_vocal.shape[1]
        tiempo_vocal = np.arange(total_muestras_vocal, dtype=np.float64) / float(fs)

        # 1. Guardar grabacion.csv con el formato nativo exacto de CsvViewerWidget
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

        # 2. Guardar pulsos_metronomo.csv
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

        # 3. Guardar segmentos_dataset.csv dentro de la carpeta extraída para alineación directa en PCA/UMAP
        seg_csv_path = os.path.join(vocal_dir, "segmentos_dataset.csv")
        campos_seg = [
            "pulso_id", "vocal", "sample_beat", "sample_onset_real",
            "sample_ventana_ini", "sample_ventana_fin", "t_beat_s", "t_onset_s",
            "ventana_inicio_s", "ventana_fin_s", "snr_db", "invalido_manual"
        ]
        with open(seg_csv_path, "w", newline="", encoding="utf-8") as fs_out:
            writer_s = csv.DictWriter(fs_out, fieldnames=campos_seg)
            writer_s.writeheader()
            writer_s.writerows(registros_segmentos_dataset)

        # 4. Guardar cada canal en canal_X/ Crudo.wav + metadata.json compatible con generador_pca_umap.py
        eventos_vocales_meta = [
            {
                "pulso_vocal_idx": r["idx"],
                "palabra": vocal,
                "t_daq_s": r["t_daq_s"],
                "t_wav_s": r["t_wav_s"],
                "sample_daq": r["sample_daq"],
                "intervalo_s": r["intervalo_s"],
                "jitter_activo": False,
                "jitter_mode": "canonical_periodic_grid",
            }
            for r in registros_pulsos_nativos
        ]

        # Importante: noise_seconds en metadata apunta a (noise_dur_efectiva + pre_anchor_s - tau_slot_s)
        # o exacto al primer beat para que cualquier lector periódico (noise_seconds + k * 60/bpm)
        # caiga exactamente sobre sample_beat_nuevo de cada pulso k
        noise_seconds_grid_aligned = round(noise_dur_efectiva + pre_anchor_s, 4)

        for ch in range(n_canales):
            ch_dir = os.path.join(vocal_dir, f"canal_{ch}")
            os.makedirs(ch_dir, exist_ok=True)
            max_val_ch = float(np.max(np.abs(matriz_vocal[ch, :])))
            if max_val_ch <= 1e-12:
                max_val_ch = 1.0
            wav_data = np.int16(np.clip(matriz_vocal[ch, :] / max_val_ch, -1.0, 1.0) * 32767)
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

            musc_name = nombres_musculos[ch] if ch < len(nombres_musculos) else f"Canal {ch}"
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
                "bpm": bpm_canonico,
                "pulse_count": len(nuevos_timestamps_wav_s),
                "noise_seconds": round(noise_dur_efectiva, 4),
                "first_beat_s": noise_seconds_grid_aligned,
                "slot_duration_s": round(tau_slot_s, 4),
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
            f"  [OK] {folder_name}: {len(nuevos_timestamps_wav_s)} pulsos sincronizados en grilla de {tau_slot_s:.2f}s "
            f"({total_muestras_vocal / float(fs):.1f}s) -> {vocal_dir}"
        )

    if generar_graficos_neon:
        graficos = generar_graficos_diagnostico_neon_slate(
            perfiles_por_vocal=perfiles_por_vocal,
            vectores_pca=vectores_pca,
            etiquetas_pca=etiquetas_pca,
            fs=fs,
            nombres_musculos=nombres_musculos,
            output_dir=output_parent_dir,
            sujeto=sujeto,
            prueba=prueba,
            callback_log=callback_log,
        )
        # Copiar también dentro de secuencia_dir para acceso inmediato
        for k_g, g_path in graficos.items():
            if os.path.isfile(g_path):
                try:
                    shutil.copy2(g_path, os.path.join(secuencia_dir, os.path.basename(g_path)))
                except Exception:
                    pass

    return carpetas_creadas



def _inferir_prueba_y_sujeto(secuencia_dir: str) -> tuple:
    """Extrae automáticamente (prueba, sujeto) desde metadata.json o el nombre de la carpeta."""
    if not secuencia_dir:
        return "", ""
    secuencia_dir = os.path.abspath(secuencia_dir)
    if os.path.isfile(secuencia_dir):
        secuencia_dir = os.path.dirname(secuencia_dir)
    if os.path.basename(secuencia_dir).startswith("canal_"):
        secuencia_dir = os.path.dirname(secuencia_dir)

    meta = {}
    for rel in ("metadata.json", "autoforge_metadata.json", os.path.join("canal_0", "metadata.json")):
        mp = os.path.join(secuencia_dir, rel)
        if os.path.isfile(mp):
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    d = json.load(f)
                    if isinstance(d, dict):
                        meta.update(d)
            except Exception:
                pass

    sujeto = str(meta.get("sujeto") or (meta.get("identificacion", {}) or {}).get("sujeto") or "").strip()
    prueba = str(meta.get("prueba") or (meta.get("identificacion", {}) or {}).get("prueba") or "").strip()

    base = os.path.basename(secuencia_dir)
    parts = [p for p in base.split("_") if p]
    if not prueba:
        if len(parts) >= 3 and parts[0].lower().startswith("secuenciacontinua"):
            prueba = parts[1]
        elif len(parts) >= 3 and parts[0].upper() in VOCALES_ESTANDAR:
            prueba = parts[1]
        elif len(parts) >= 2:
            prueba = parts[1]
    if not sujeto:
        if len(parts) >= 3 and parts[0].lower().startswith("secuenciacontinua"):
            # Si termina en hora HHMMSS, tomar parts[2]
            sujeto = parts[2]
        elif len(parts) >= 3:
            sujeto = parts[-1] if not parts[-1].isdigit() else parts[2]
    return prueba, sujeto


def _resolver_directorio_inicial_gestor(initial_dir: str = None) -> str:
    """Resuelve la carpeta seleccionada actualmente en el Gestor de Sesiones si initial_dir es vacío."""
    if initial_dir and os.path.exists(initial_dir):
        p = os.path.abspath(initial_dir)
        if os.path.isfile(p):
            p = os.path.dirname(p)
        if os.path.basename(p).startswith("canal_"):
            p = os.path.dirname(p)
        return p

    env_dir = os.environ.get("NANDU_SELECTED_MEDICION_DIR", "").strip()
    if env_dir and os.path.exists(env_dir):
        return os.path.abspath(env_dir)

    # Buscar archivo de estado compartido escrito por SessionExplorer / CsvViewerWidget
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emg_dir = os.path.dirname(script_dir)
    cands_state = [
        os.path.join(emg_dir, "cache_datos", "_last_selected_session.json"),
        os.path.join(script_dir, "_last_selected_session.json"),
        os.path.expanduser("~/.nandu_last_selected_session.json"),
    ]
    for st_file in cands_state:
        if os.path.isfile(st_file):
            try:
                with open(st_file, "r", encoding="utf-8") as f:
                    st = json.load(f)
                sel_p = st.get("selected_path") or st.get("medicion_dir") or ""
                if sel_p and os.path.exists(sel_p):
                    return os.path.abspath(sel_p)
            except Exception:
                pass
    return initial_dir or ""


def lanzar_gui_extractor(initial_dir: str = None):
    """Interfaz gráfica PySide6 con estética Neon Clinical Slate para segmentar pulsos por vocal."""
    from PySide6 import QtWidgets

    app = QtWidgets.QApplication.instance()
    own_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        own_app = True

    resolved_dir = _resolver_directorio_inicial_gestor(initial_dir)
    auto_prueba, auto_sujeto = _inferir_prueba_y_sujeto(resolved_dir)

    dlg = QtWidgets.QDialog()
    dlg.setWindowTitle("Ñandú LSD — Extractor Fisiológico de Pulsos por Vocal (Neon Clinical Slate)")
    dlg.resize(820, 560)
    dlg.setStyleSheet("""
        QDialog, QWidget { background-color: #07090E; color: #E6EAF2; font-family: 'Inter', 'Segoe UI', sans-serif; font-size: 13px; }
        QLineEdit, QDoubleSpinBox { background-color: #141A25; color: #22D3EE; border: 1px solid #1F2937; padding: 5px; border-radius: 4px; font-family: 'JetBrains Mono', 'Consolas', monospace; }
        QLineEdit:focus, QDoubleSpinBox:focus { border: 1px solid #22D3EE; }
        QPushButton { background-color: rgba(52, 211, 153, 0.14); color: #34D399; border: 1.5px solid #34D399; padding: 7px 14px; font-weight: bold; border-radius: 6px; }
        QPushButton:hover { background-color: #34D399; color: #07090E; }
        QTextEdit { background-color: #0D1119; color: #34D399; border: 1px solid #1F2937; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 12px; }
    """)

    layout = QtWidgets.QVBoxLayout(dlg)
    lbl_header = QtWidgets.QLabel(
        "<b style='color:#22D3EE; font-size:16px;'>Extractor Fisiológico de Pulsos por Vocal (Grilla Canónica 2.0s)</b><br>"
        "<span style='color:#8B95A7;'>Alinea cada pulso en el inicio acústico real (<code>t=0</code> Canal 3) + ráfaga pre-fonatoria "
        "muscular (<code>[-600 ms, +800 ms]</code>), filtra pulsos mudos por SNR y exporta las 5 carpetas "
        "(<code>A_...</code>, <code>E_...</code>, <code>I_...</code>, <code>O_...</code>, <code>U_...</code>) en una grilla periódica exacta "
        "junto con el gráfico de <b>Patrón Muscular Neon Clinical Slate</b>.</span>"
    )
    lbl_header.setWordWrap(True)
    layout.addWidget(lbl_header)

    form = QtWidgets.QFormLayout()
    row_dir = QtWidgets.QHBoxLayout()
    inp_secuencia = QtWidgets.QLineEdit(resolved_dir or "")
    btn_browse = QtWidgets.QPushButton("Seleccionar Carpeta SecuenciaContinua...")
    row_dir.addWidget(inp_secuencia, 1)
    row_dir.addWidget(btn_browse)
    form.addRow("Carpeta Origen:", row_dir)

    inp_prueba = QtWidgets.QLineEdit(auto_prueba or "")
    inp_prueba.setPlaceholderText("Auto-detectado (ej. Pruebainicial)")
    form.addRow("Identificador de Prueba:", inp_prueba)

    inp_sujeto = QtWidgets.QLineEdit(auto_sujeto or "")
    inp_sujeto.setPlaceholderText("Auto-detectado (ej. Candela)")
    form.addRow("Nombre del Sujeto:", inp_sujeto)

    def _autocompletar_campos_desde_ruta(ruta_str: str):
        p_clean = (ruta_str or "").strip()
        if p_clean and os.path.exists(p_clean):
            pr_det, su_det = _inferir_prueba_y_sujeto(p_clean)
            if pr_det:
                inp_prueba.setText(pr_det)
            if su_det:
                inp_sujeto.setText(su_det)

    inp_secuencia.textChanged.connect(_autocompletar_campos_desde_ruta)

    chk_refinar_pico = QtWidgets.QCheckBox("Alinear en onset acústico (Ch3 t=0) + ráfaga pre-fonatoria sEMG [-600 ms, +800 ms]")
    chk_refinar_pico.setChecked(True)
    form.addRow("Alineación Fisiológica:", chk_refinar_pico)

    chk_snr_gate = QtWidgets.QCheckBox("Filtrar pulsos mudos / artefactos eléctricos mediante Gate SNR (> 4.3 dB)")
    chk_snr_gate.setChecked(True)
    form.addRow("Control de Calidad SNR:", chk_snr_gate)

    chk_ventana_fija = QtWidgets.QCheckBox("Grilla periódica canónica estricta por pulso (tau = 2.00s, 30.0 BPM)")
    chk_ventana_fija.setChecked(True)
    spin_ventana = QtWidgets.QDoubleSpinBox()
    spin_ventana.setRange(1.0, 6.0)
    spin_ventana.setValue(2.0)
    spin_ventana.setSuffix(" s")
    spin_ventana.setEnabled(True)
    chk_ventana_fija.toggled.connect(spin_ventana.setEnabled)
    row_win = QtWidgets.QHBoxLayout()
    row_win.addWidget(chk_ventana_fija)
    row_win.addWidget(spin_ventana)
    form.addRow("Slot Canónico:", row_win)

    layout.addLayout(form)

    btn_run = QtWidgets.QPushButton("EXTRAER PULSOS, ALINEAR GRILLA Y GENERAR PATRÓN MUSCULAR NEON SLATE")
    btn_run.setStyleSheet("""
        QPushButton { background-color: rgba(52, 211, 153, 0.16); color: #34D399; border: 2px solid #34D399; padding: 12px; font-size: 14px; font-weight: 800; border-radius: 6px; }
        QPushButton:hover { background-color: #34D399; color: #07090E; }
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
            _autocompletar_campos_desde_ruta(d)

    def _ejecutar():
        target = inp_secuencia.text().strip()
        if not target or not os.path.isdir(target):
            QtWidgets.QMessageBox.warning(dlg, "Carpeta inválida", "Selecciona una carpeta de Secuencia Continua válida.")
            return
        try:
            win_fija = spin_ventana.value() if chk_ventana_fija.isChecked() else 2.0
            res = extraer_pulsos_por_vocal(
                secuencia_dir=target,
                prefijo_prueba=inp_prueba.text().strip() or None,
                sujeto_override=inp_sujeto.text().strip() or None,
                duracion_ventana_fija_s=win_fija,
                centrar_en_pico_energia=chk_refinar_pico.isChecked(),
                filtrar_pulsos_bajo_snr=chk_snr_gate.isChecked(),
                generar_graficos_neon=True,
                callback_log=_log,
            )
            resumen = "\n".join([f"• {v}: {info['nombre']} ({info['n_pulsos']} pulsos sincronizados)" for v, info in res.items()])
            QtWidgets.QMessageBox.information(
                dlg,
                "Extracción Completada",
                f"Se crearon {len(res)} carpetas sincronizadas por vocal:\n\n{resumen}\n\n"
                "También se generó 'patron_muscular_neon_clinical_slate.png'."
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
    parser = argparse.ArgumentParser(description="Extractor fisiológico de pulsos por vocal desde Secuencia Continua.")
    parser.add_argument("carpeta", nargs="?", default=None, help="Ruta a la carpeta SecuenciaContinua_*")
    parser.add_argument("--out", default=None, help="Directorio padre de salida (por defecto misma carpeta de fecha)")
    parser.add_argument("--prueba", default=None, help="Nombre de la prueba (ej. Prueba1)")
    parser.add_argument("--sujeto", default=None, help="Nombre del sujeto (ej. Candela)")
    parser.add_argument("--ventana-fija", type=float, default=2.0, help="Duración fija del slot canónico en segundos (default: 2.0s)")
    parser.add_argument("--no-refinar-pico", action="store_true", help="Desactivar alineación fisiológica por inicio acústico/sEMG")
    parser.add_argument("--no-snr-gate", action="store_true", help="No descartar pulsos mudos bajo umbral SNR")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en modo consola sin abrir GUI")
    args = parser.parse_args()

    if args.cli and args.carpeta:
        # sys.argv[1] puede ser carpeta de sesion o un archivo de la medicion:
        # se resuelve a la carpeta contenedora antes de ejecutar la extraccion.
        carpeta_resuelta = _resolver_ruta_sesion(args.carpeta) or args.carpeta
        extraer_pulsos_por_vocal(
            secuencia_dir=carpeta_resuelta,
            output_parent_dir=args.out,
            prefijo_prueba=args.prueba,
            sujeto_override=args.sujeto,
            duracion_ventana_fija_s=args.ventana_fija,
            centrar_en_pico_energia=not args.no_refinar_pico,
            filtrar_pulsos_bajo_snr=not args.no_snr_gate,
            generar_graficos_neon=True,
        )
    else:
        lanzar_gui_extractor(initial_dir=args.carpeta)


if __name__ == "__main__":
    ensure_running_in_terminal("Extractor de Pulsos Vocales")
    main()
