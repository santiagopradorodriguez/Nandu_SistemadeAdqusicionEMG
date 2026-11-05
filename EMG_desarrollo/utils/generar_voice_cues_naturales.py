#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: utils/generar_voice_cues_naturales.py
# Descripción: Generador híbrido de locuciones vocales ("A", "E", "I", "O", "U")
#              en español rioplatense con 3 motores en cascada:
#              1) Neural TTS (edge-tts: es-AR-ElenaNeural / es-AR-TomasNeural)
#              2) Síntesis Klatt Glotal Mejorada (LF glottal source + vibrato
#                 fisiológico + aspiración suave + 4 formantes vocálicos)
#              Salva en assets/voice_cues/{A,E,I,O,U}.wav a 44100 Hz PCM 16-bit.
# ==============================================================================

import os
import sys
import wave
import shutil
import asyncio
import tempfile
import subprocess
import numpy as np
from scipy import signal


FORMANTES_VOCALES_ES = {
    # (F1, F2, F3, F4) en Hz y anchos de banda (BW1, BW2, BW3, BW4)
    "A": ([750.0, 1280.0, 2550.0, 3600.0], [80.0, 95.0, 130.0, 180.0]),
    "E": ([480.0, 1980.0, 2600.0, 3650.0], [65.0, 95.0, 130.0, 180.0]),
    "I": ([290.0, 2320.0, 3050.0, 3750.0], [55.0, 100.0, 140.0, 190.0]),
    "O": ([500.0, 880.0, 2500.0, 3500.0], [70.0, 85.0, 130.0, 180.0]),
    "U": ([320.0, 720.0, 2420.0, 3450.0], [60.0, 80.0, 130.0, 180.0]),
}

TEXTO_LOCUCION_NEURAL = {
    "A": "a",
    "E": "e",
    "I": "i",
    "O": "o",
    "U": "u",
}


def _guardar_wav_mono_44k(filepath: str, y: np.ndarray, sr: int = 44100):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    y = np.nan_to_num(y.astype(np.float64))
    peak = float(np.max(np.abs(y)))
    if peak > 1e-8:
        y = (y / peak) * 0.88
    pcm16 = np.clip(y * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(pcm16.tobytes())


def sintetizar_vocal_klatt_natural(vocal: str, duracion_s: float = 0.28, sr: int = 44100, f0_base: float = 195.0) -> np.ndarray:
    """
    Síntesis Klatt de fuente glotal Liljencrants-Fant (LF) con micro-vibrato natural (5.2 Hz),
    jitter fisiológico suave (0.35%), ruido de aspiración glotal acoplado y cascada de 4 resonadores
    de formantes + radiación labial. Elimina el timbre robótico de tren de impulsos rígido.
    """
    vocal = vocal.strip().upper()
    freqs, bws = FORMANTES_VOCALES_ES.get(vocal, FORMANTES_VOCALES_ES["A"])
    n_samples = int(round(duracion_s * sr))
    t = np.arange(n_samples, dtype=np.float64) / float(sr)

    # Contorno entonacional cálido y natural (ligero ascenso inicial + caída suave final)
    intonation = 1.0 + 0.025 * np.sin(np.pi * t / duracion_s) - 0.04 * (t / duracion_s) ** 2
    vibrato = 0.012 * np.sin(2.0 * np.pi * 5.2 * t)
    rng = np.random.default_rng(abs(hash(vocal)) % 10000)
    micro_flutter = signal.lfilter([1.0], [1.0, -0.97], rng.normal(0.0, 0.0006, n_samples))
    f0_t = f0_base * (intonation + vibrato + micro_flutter)

    phase = np.cumsum(f0_t / float(sr)) % 1.0
    # Pulso glotal asimétrico suave (apertura gradual 0..0.48, retorno rápido 0.48..0.68, fase cerrada 0.68..1.0)
    glottal = np.zeros_like(phase)
    m_open = phase < 0.48
    m_close = (phase >= 0.48) & (phase < 0.68)
    glottal[m_open] = 0.5 * (1.0 - np.cos(np.pi * phase[m_open] / 0.48))
    glottal[m_close] = np.cos(0.5 * np.pi * (phase[m_close] - 0.48) / 0.20)
    # Derivada glotal suavizada + aspiración suave en fase abierta
    glottal_deriv = np.gradient(glottal)
    aspiration = rng.normal(0.0, 0.035, n_samples) * np.maximum(0.0, glottal)
    source = glottal_deriv + aspiration

    # Cascada de 4 resonadores de segundo orden (Klatt cascade) con ligero desplazamiento coarticulatorio
    y = source.copy()
    for f_hz, bw_hz in zip(freqs, bws):
        r = np.exp(-np.pi * bw_hz / float(sr))
        theta = 2.0 * np.pi * f_hz / float(sr)
        b_res = [1.0 - 2.0 * r * np.cos(theta) + r * r]
        a_res = [1.0, -2.0 * r * np.cos(theta), r * r]
        y = signal.lfilter(b_res, a_res, y)

    # Filtro de calidez espectral (atenúa brillo metálico por encima de 4.2 kHz)
    b_lp, a_lp = signal.butter(2, 4200.0 / (0.5 * sr), btype="low")
    y = signal.filtfilt(b_lp, a_lp, y)

    # Envolvente ADSR fonatoria natural (ataque suave de 28 ms, caída suave de 55 ms)
    env = np.ones(n_samples, dtype=np.float64)
    n_att = max(1, int(0.028 * sr))
    n_rel = max(1, int(0.055 * sr))
    env[:n_att] = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_att) / float(n_att)))
    if n_rel < n_samples:
        env[-n_rel:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_rel) / float(n_rel)))
    return y * env


async def _intentar_edge_tts(vocal: str, out_wav: str, voice: str = "es-AR-ElenaNeural") -> bool:
    try:
        import edge_tts
    except ImportError:
        return False

    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False

    tmp_mp3 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_mp3 = tf.name
        communicate = edge_tts.Communicate(TEXTO_LOCUCION_NEURAL.get(vocal, vocal.lower()), voice=voice, rate="+10%")
        await communicate.save(tmp_mp3)
        cmd = [
            ffmpeg_bin, "-y", "-i", tmp_mp3,
            "-af", "silenceremove=start_periods=1:start_threshold=-42dB:stop_periods=1:stop_threshold=-42dB,atrim=0:0.30,afade=t=out:st=0.22:d=0.07",
            "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", out_wav
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return os.path.exists(out_wav) and os.path.getsize(out_wav) > 1024
    except Exception:
        return False
    finally:
        if tmp_mp3 and os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except Exception:
                pass


def generar_todos_los_voice_cues(output_dir: str = None, prefer_neural: bool = True, voice: str = "es-AR-ElenaNeural") -> dict:
    if output_dir is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, "assets", "voice_cues")
    os.makedirs(output_dir, exist_ok=True)

    resultados = {}
    for v in ("A", "E", "I", "O", "U"):
        out_wav = os.path.join(output_dir, f"{v}.wav")
        ok_neural = False
        if prefer_neural:
            try:
                ok_neural = asyncio.run(_intentar_edge_tts(v, out_wav, voice=voice))
            except Exception:
                ok_neural = False
        if ok_neural:
            resultados[v] = {"path": out_wav, "engine": f"edge-tts ({voice})"}
        else:
            y_klatt = sintetizar_vocal_klatt_natural(v, duracion_s=0.28, sr=44100, f0_base=195.0)
            _guardar_wav_mono_44k(out_wav, y_klatt, sr=44100)
            resultados[v] = {"path": out_wav, "engine": "Klatt-LF Natural Warm"}

        # Generar alias de nombres para búsqueda directa insensible a mayúsculas
        for alias_name in (f"{v.lower()}.wav", f"vocal_{v.lower()}.wav", f"vocal_{v}.wav", f"di_{v.lower()}.wav", f"di_{v}.wav"):
            alias_path = os.path.join(output_dir, alias_name)
            if not os.path.exists(alias_path) or os.path.getsize(alias_path) != os.path.getsize(out_wav):
                try:
                    shutil.copy2(out_wav, alias_path)
                except Exception:
                    pass

        # Sincronizar hacia todas las carpetas candidatas del proyecto
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        carpetas_destino = [
            os.path.join(repo_root, "assets", "voice_cues"),
            os.path.join(repo_root, "EMG_desarrollo", "acquisition", "assets", "voice_cues"),
            os.path.join(repo_root, "acquisition", "assets", "voice_cues"),
        ]
        for c_dir in carpetas_destino:
            try:
                os.makedirs(c_dir, exist_ok=True)
                for f_name in (f"{v}.wav", f"{v.lower()}.wav", f"vocal_{v.lower()}.wav", f"vocal_{v}.wav", f"di_{v.lower()}.wav", f"di_{v}.wav"):
                    f_src = os.path.join(output_dir, f_name)
                    f_dst = os.path.join(c_dir, f_name)
                    if os.path.isfile(f_src):
                        shutil.copy2(f_src, f_dst)
            except Exception:
                pass

        print(f"[VoiceCue] /{v}/ -> {out_wav} ({resultados[v]['engine']})")
    return resultados


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.getcwd(), "assets", "voice_cues")
    generar_todos_los_voice_cues(output_dir=out)
