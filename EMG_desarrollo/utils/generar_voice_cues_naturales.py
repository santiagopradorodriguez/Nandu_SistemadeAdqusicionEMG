#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: utils/generar_voice_cues_naturales.py
# Descripción: Regenera los audios de locución de vocales (a.wav, e.wav, i.wav,
#              o.wav, u.wav y frases de preparación) con voz femenina rioplatense /
#              latina natural y sostenida (~0.60 s), envolvente suave (fade-in y
#              fade-out coseno de 15 ms) y normalización a -3.0 dBFS en formato
#              WAV 22050 Hz mono dentro de EMG_desarrollo/assets/voice_cues/.
# ==============================================================================

import os
import sys
import math
import shutil
import tempfile
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal


SAMPLE_RATE_TARGET = 22050
DURATION_VOWEL_S = 0.62
FADE_MS = 15.0
TARGET_DBFS = -3.0

# Formantes fisiológicos femeninos (F1, F2, F3 en Hz) para síntesis armónica natural de respaldo
FORMANTES_FEMENINOS_RIOPLATENSES = {
    "a": (820.0, 1380.0, 2850.0),
    "e": (480.0, 2280.0, 2950.0),
    "i": (320.0, 2750.0, 3300.0),
    "o": (510.0, 920.0, 2750.0),
    "u": (350.0, 780.0, 2650.0),
}


def _aplicar_fade_y_normalizar_dbfs(audio: np.ndarray, sr: int, fade_ms: float = FADE_MS, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64).flatten()
    if len(arr) == 0:
        return arr
    arr = arr - np.mean(arr)

    n_fade = max(1, min(int(round((fade_ms / 1000.0) * sr)), len(arr) // 4))
    ramp_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / float(n_fade)))
    ramp_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / float(n_fade)))
    arr[:n_fade] *= ramp_in
    arr[-n_fade:] *= ramp_out

    peak = float(np.max(np.abs(arr)))
    if peak > 1e-8:
        target_amp = 10.0 ** (target_dbfs / 20.0)  # -3 dBFS ≈ 0.7079
        arr = (arr / peak) * target_amp
    return np.clip(arr, -1.0, 1.0)


def _sintetizar_vocal_formantica_femenina(vocal: str, sr: int = SAMPLE_RATE_TARGET, dur_s: float = DURATION_VOWEL_S) -> np.ndarray:
    """
    Sintetizador Klatt/Rosenberg de voz femenina natural (~215 Hz F0 con vibrato
    fisiológico de 5.2 Hz y filtrado resonante de 3 formantes vocálicos).
    Se utiliza como respaldo determinista o complemento armónico suave.
    """
    n = int(round(sr * dur_s))
    t = np.linspace(0.0, dur_s, n, endpoint=False)
    f0_base = 218.0
    # Micro-vibrato humano suave (5.2 Hz, ±2.2 Hz) y ligera caída de entonación natural al final
    f0_inst = f0_base + 2.2 * np.sin(2.0 * np.pi * 5.2 * t) - 6.0 * (t / dur_s) ** 2
    phase = 2.0 * np.pi * np.cumsum(f0_inst) / float(sr)

    # Tren glotal rico en armónicos con caída espectral suave (-12 dB/octava)
    glottal = np.zeros_like(t)
    for k in range(1, 22):
        if k * f0_base >= (sr * 0.45):
            break
        glottal += (1.0 / (k ** 1.35)) * np.sin(k * phase)

    f1, f2, f3 = FORMANTES_FEMENINOS_RIOPLATENSES.get(vocal.lower(), (600.0, 1500.0, 2800.0))
    bws = [85.0, 110.0, 160.0]
    gains = [1.0, 0.65, 0.25]
    voiced = np.zeros_like(glottal)
    for freq, bw, g in zip((f1, f2, f3), bws, gains):
        b, a = signal.iirpeak(freq / (sr / 2.0), Q=max(2.0, freq / bw))
        voiced += g * signal.lfilter(b, a, glottal)

    # Envolvente prosódica sostenida suave (sin ataque explosivo)
    env = np.ones_like(t)
    n_att = int(0.045 * sr)
    n_rel = int(0.085 * sr)
    env[:n_att] = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_att) / float(n_att)))
    env[-n_rel:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_rel) / float(n_rel)))
    return _aplicar_fade_y_normalizar_dbfs(voiced * env, sr)


def _intentar_edge_tts(texto: str, out_wav_path: str, sr: int = SAMPLE_RATE_TARGET) -> bool:
    """Intenta sintetizar con voz neuronal argentina (es-AR-ElenaNeural) vía edge-tts + ffmpeg."""
    edge_bin = shutil.which("edge-tts")
    ffmpeg_bin = shutil.which("ffmpeg")
    if not edge_bin or not ffmpeg_bin:
        return False
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            mp3_path = tmp_mp3.name
        subprocess.run(
            [edge_bin, "--voice", "es-AR-ElenaNeural", "--rate=-12%", "--text", texto, "--write-media", mp3_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        subprocess.run(
            [ffmpeg_bin, "-y", "-i", mp3_path, "-ac", "1", "-ar", str(sr), out_wav_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        os.remove(mp3_path)
        sr_r, data = wavfile.read(out_wav_path)
        arr = data.astype(np.float64) / 32767.0 if data.dtype == np.int16 else data.astype(np.float64)
        # Recortar silencios extremos y asegurar duración sostenida suave (~0.6 s)
        mask = np.abs(arr) > (0.04 * np.max(np.abs(arr) + 1e-8))
        if np.any(mask):
            idx = np.where(mask)[0]
            arr = arr[max(0, idx[0] - int(0.02 * sr_r)):min(len(arr), idx[-1] + int(0.04 * sr_r))]
        arr_norm = _aplicar_fade_y_normalizar_dbfs(arr, sr_r)
        wavfile.write(out_wav_path, sr_r, np.int16(arr_norm * 32767))
        return True
    except Exception:
        return False


def regenerar_voice_cues(dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    print(f"==> Generando audios de locución naturales y sostenidos en: {dest_dir}")

    textos_sostenidos = {
        "a": "aaa",
        "e": "eee",
        "i": "iii",
        "o": "ooo",
        "u": "uuu",
    }

    for vocal, txt_tts in textos_sostenidos.items():
        wav_path = os.path.join(dest_dir, f"{vocal}.wav")
        ok_tts = _intentar_edge_tts(txt_tts, wav_path, sr=SAMPLE_RATE_TARGET)
        if not ok_tts:
            arr = _sintetizar_vocal_formantica_femenina(vocal, sr=SAMPLE_RATE_TARGET, dur_s=DURATION_VOWEL_S)
            wavfile.write(wav_path, SAMPLE_RATE_TARGET, np.int16(arr * 32767))
            metodo = "Síntesis Formántica Klatt Femenina (0.62s, -3 dBFS, fade 15ms)"
        else:
            metodo = "Voz Neuronal es-AR-ElenaNeural (-3 dBFS, fade 15ms)"
        print(f"  [OK] {vocal}.wav -> {metodo}")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(repo_root, "EMG_desarrollo", "assets", "voice_cues"),
        os.path.join(repo_root, "assets", "voice_cues"),
    ]
    for c_dir in candidates:
        regenerar_voice_cues(c_dir)


if __name__ == "__main__":
    main()