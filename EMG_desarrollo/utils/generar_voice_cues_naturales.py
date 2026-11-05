#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Módulo: utils/generar_voice_cues_naturales.py
# Descripción: Regenera los audios de locución de vocales (a.wav, e.wav, i.wav,
#              o.wav, u.wav, di_a.wav..di_u.wav, preparate.wav, haz_silencio.wav,
#              descanso.wav) usando TTS neuronal de alta definición en español
#              argentino/latino (edge-tts 'es-AR-ElenaNeural' o gTTS 'com.ar'),
#              con envolvente suave (fade-in/out de 15 ms) y normalización a
#              -3.0 dBFS en formato WAV 22050 Hz mono.
#              PROHIBIDO el sintetizador formántico robótico Klatt.
# ==============================================================================

import os
import sys
import shutil
import asyncio
import tempfile
import subprocess
import numpy as np
from scipy.io import wavfile
from scipy import signal


SAMPLE_RATE_TARGET = 22050
FADE_MS = 15.0
TARGET_DBFS = -3.0

LOCUCIONES_OFICIALES = {
    "a": "Aaa",
    "e": "Eee",
    "i": "Iii",
    "o": "Ooo",
    "u": "Uuu",
    "di_a": "Di A",
    "di_e": "Di E",
    "di_i": "Di I",
    "di_o": "Di O",
    "di_u": "Di U",
    "preparate": "Prepárate",
    "haz_silencio": "Haz silencio",
    "descanso": "Descansa",
}


def _aplicar_fade_y_normalizar_dbfs(audio: np.ndarray, sr: int, fade_ms: float = FADE_MS, target_dbfs: float = TARGET_DBFS) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float64).flatten()
    if len(arr) == 0:
        return arr
    arr = arr - np.mean(arr)

    # Recortar silencios digitales en los extremos antes de aplicar el fade suave
    peak_raw = float(np.max(np.abs(arr)) + 1e-9)
    mask = np.abs(arr) > (0.025 * peak_raw)
    if np.any(mask):
        idx = np.where(mask)[0]
        pad_pre = int(0.015 * sr)
        pad_post = int(0.035 * sr)
        arr = arr[max(0, idx[0] - pad_pre):min(len(arr), idx[-1] + pad_post)]

    n_fade = max(1, min(int(round((fade_ms / 1000.0) * sr)), len(arr) // 4))
    ramp_in = 0.5 * (1.0 - np.cos(np.pi * np.arange(n_fade) / float(n_fade)))
    ramp_out = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_fade) / float(n_fade)))
    arr[:n_fade] *= ramp_in
    arr[-n_fade:] *= ramp_out

    peak = float(np.max(np.abs(arr)))
    if peak > 1e-8:
        target_amp = 10.0 ** (target_dbfs / 20.0)
        arr = (arr / peak) * target_amp
    return np.clip(arr, -1.0, 1.0)


def _decodificar_mp3_a_wav_mono(mp3_path: str, out_wav_path: str, sr: int = SAMPLE_RATE_TARGET) -> bool:
    """Convierte un MP3 generado por edge-tts o gTTS a WAV mono 22050 Hz normalizado a -3 dBFS."""
    for bin_cand in ("ffmpeg", "avconv"):
        exe = shutil.which(bin_cand)
        if exe:
            try:
                subprocess.run(
                    [exe, "-y", "-i", mp3_path, "-ac", "1", "-ar", str(sr), out_wav_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                )
                sr_r, data = wavfile.read(out_wav_path)
                arr = data.astype(np.float64) / 32767.0 if data.dtype == np.int16 else data.astype(np.float64)
                arr_norm = _aplicar_fade_y_normalizar_dbfs(arr, sr_r)
                wavfile.write(out_wav_path, sr_r, np.int16(arr_norm * 32767))
                return True
            except Exception:
                pass
    return False


def sintetizar_con_edge_tts_python(texto: str, out_wav_path: str, voice: str = "es-AR-ElenaNeural", sr: int = SAMPLE_RATE_TARGET) -> bool:
    """
    Usa la librería Python edge_tts (o el ejecutable edge-tts) con la voz neuronal
    femenina argentina es-AR-ElenaNeural.
    """
    tmp_mp3 = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_mp3 = tf.name

        # 1. Intentar con el módulo Python edge_tts directamente
        try:
            import edge_tts
            async def _gen():
                comm = edge_tts.Communicate(texto, voice=voice, rate="-10%")
                await comm.save(tmp_mp3)
            asyncio.run(_gen())
            if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 200:
                if _decodificar_mp3_a_wav_mono(tmp_mp3, out_wav_path, sr=sr):
                    return True
        except Exception:
            pass

        # 2. Intentar vía CLI `python3 -m edge_tts`
        subprocess.run(
            [sys.executable, "-m", "edge_tts", "--voice", voice, "--rate=-10%", "--text", texto, "--write-media", tmp_mp3],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 200:
            if _decodificar_mp3_a_wav_mono(tmp_mp3, out_wav_path, sr=sr):
                return True
    except Exception:
        pass
    finally:
        if tmp_mp3 and os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except Exception:
                pass
    return False


def sintetizar_con_gtts_python(texto: str, out_wav_path: str, sr: int = SAMPLE_RATE_TARGET) -> bool:
    """Fallback secundario usando Google TTS en español rioplatense (tld='com.ar')."""
    tmp_mp3 = None
    try:
        from gtts import gTTS
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
            tmp_mp3 = tf.name
        tts = gTTS(text=texto, lang="es", tld="com.ar", slow=False)
        tts.save(tmp_mp3)
        if os.path.exists(tmp_mp3) and os.path.getsize(tmp_mp3) > 200:
            return _decodificar_mp3_a_wav_mono(tmp_mp3, out_wav_path, sr=sr)
    except Exception:
        pass
    finally:
        if tmp_mp3 and os.path.exists(tmp_mp3):
            try:
                os.remove(tmp_mp3)
            except Exception:
                pass
    return False


def regenerar_voice_cues(dest_dir: str) -> int:
    os.makedirs(dest_dir, exist_ok=True)
    print(f"==> Generando locuciones neuronales humanas (es-AR-ElenaNeural) en: {dest_dir}")
    generados = 0
    for key, texto in LOCUCIONES_OFICIALES.items():
        wav_path = os.path.join(dest_dir, f"{key}.wav")
        if sintetizar_con_edge_tts_python(texto, wav_path, voice="es-AR-ElenaNeural", sr=SAMPLE_RATE_TARGET):
            print(f"  [OK] {key}.wav -> Voz Neuronal es-AR-ElenaNeural ('{texto}')")
            generados += 1
        elif sintetizar_con_gtts_python(texto, wav_path, sr=SAMPLE_RATE_TARGET):
            print(f"  [OK] {key}.wav -> Voz gTTS Rioplatense ('{texto}')")
            generados += 1
        else:
            print(
                f"  [AVISO] {key}.wav: Sin conexión a edge-tts/gTTS en este entorno. "
                "Ejecuta `pip install edge-tts && python utils/generar_voice_cues_naturales.py` en tu PC con internet."
            )
    return generados


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