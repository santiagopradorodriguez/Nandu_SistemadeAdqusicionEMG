# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo multiplataforma para emisión de beeps y sonidos de metrónomo.
# ==============================================================================

import threading
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

def _play_sounddevice(freq, duration_ms, volume=0.3):
    fs = 44100
    n_samples = int(fs * (duration_ms / 1000.0))
    if n_samples <= 0:
        return
    t = np.linspace(0, duration_ms / 1000.0, n_samples, False)
    wave = np.sin(2 * np.pi * freq * t) * volume
    fade_len = min(int(fs * 0.005), len(wave) // 2)
    if fade_len > 0:
        fade = np.linspace(0, 1, fade_len)
        wave[:fade_len] *= fade
        wave[-fade_len:] *= fade[::-1]
    sd.play(wave.astype(np.float32), samplerate=fs)

def play_beep(freq=1000, duration_ms=100, async_play=True):
    """
    Emite un tono/beep de audio multiplataforma.
    En Windows usa winsound.Beep nativo (clásico, nítido y sin latencia).
    En Linux/macOS usa sounddevice con onda senoidal sintética.
    """
    def _execute():
        if winsound is not None:
            try:
                winsound.Beep(int(freq), int(duration_ms))
                return
            except Exception:
                pass
        if sd is not None:
            try:
                _play_sounddevice(freq, duration_ms)
                return
            except Exception:
                pass

    if async_play:
        threading.Thread(target=_execute, daemon=True).start()
    else:
        _execute()
