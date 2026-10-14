# ==============================================================================
# Archivo: EMG_desarrollo/core/dsp_engine.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Descripción: Motor DSP compilado con Numba y filtros SOS causales sin transitorios.
# ==============================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from scipy import signal

try:
    from numba import jit
except ImportError:
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


@jit(nopython=True, nogil=True, cache=True)
def calculate_rms_envelope(buffer: np.ndarray, window_size: int) -> np.ndarray:
    """
    Calcula la envolvente RMS sobre un buffer 1D usando una ventana deslizante O(N).
    Recalcula la suma cuadrática cada 16384 muestras para evitar deriva de punto flotante.
    """
    n = len(buffer)
    out = np.zeros(n, dtype=np.float64)
    if window_size <= 0 or n == 0:
        return out

    sum_sq = 0.0
    limit = min(window_size, n)
    for i in range(limit):
        val = 0.0 if np.isnan(buffer[i]) else buffer[i]
        sum_sq += val * val
        out[i] = np.sqrt(max(0.0, sum_sq) / (i + 1))

    for i in range(window_size, n):
        if i % 16384 == 0:
            sum_sq = 0.0
            for k in range(i - window_size + 1, i + 1):
                vk = 0.0 if np.isnan(buffer[k]) else buffer[k]
                sum_sq += vk * vk
        else:
            val_add = 0.0 if np.isnan(buffer[i]) else buffer[i]
            val_sub = 0.0 if np.isnan(buffer[i - window_size]) else buffer[i - window_size]
            sum_sq += (val_add * val_add) - (val_sub * val_sub)

        if sum_sq < 0.0 or np.isnan(sum_sq):
            sum_sq = 0.0
        out[i] = np.sqrt(sum_sq / window_size)

    return out


@jit(nopython=True, nogil=True, cache=True)
def calculate_rms_envelope_2d(matrix: np.ndarray, window_size: int) -> np.ndarray:
    """Calcula la envolvente RMS para todos los canales (num_channels, num_samples)."""
    num_channels, num_samples = matrix.shape
    out = np.zeros((num_channels, num_samples), dtype=np.float64)
    for ch in range(num_channels):
        out[ch, :] = calculate_rms_envelope(matrix[ch, :], window_size)
    return out


# Pre-compilar Numba al importar para eliminar lag en tiempo real
_dummy_1d = calculate_rms_envelope(np.zeros(16, dtype=np.float64), 4)
_dummy_2d = calculate_rms_envelope_2d(np.zeros((2, 16), dtype=np.float64), 4)


def decimate_min_max(data_x: np.ndarray, data_y: np.ndarray, max_points: int = 3000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decimación dinámica usando Min/Max Peaking sin copias innecesarias.
    Preserva la envolvente visual de alta frecuencia del sEMG evitando aliasing.
    """
    n = len(data_y)
    if n <= max_points or max_points < 4:
        return data_x, data_y

    num_chunks = max_points // 2
    chunk_size = n // num_chunks
    trunc_length = chunk_size * num_chunks

    y_trunc = data_y[:trunc_length].reshape((num_chunks, chunk_size))
    x_trunc = data_x[:trunc_length].reshape((num_chunks, chunk_size))

    min_indices = np.argmin(y_trunc, axis=1)
    max_indices = np.argmax(y_trunc, axis=1)
    idx_range = np.arange(num_chunks)

    take_min_first = min_indices <= max_indices
    first_indices = np.where(take_min_first, min_indices, max_indices)
    second_indices = np.where(take_min_first, max_indices, min_indices)

    out_x = np.empty(num_chunks * 2, dtype=data_x.dtype)
    out_y = np.empty(num_chunks * 2, dtype=data_y.dtype)

    out_x[0::2] = x_trunc[idx_range, first_indices]
    out_y[0::2] = y_trunc[idx_range, first_indices]
    out_x[1::2] = x_trunc[idx_range, second_indices]
    out_y[1::2] = y_trunc[idx_range, second_indices]

    return out_x, out_y


def calcular_ruido_linea_50hz(noise_signal_uV: np.ndarray, fs: float) -> float:
    """
    Calcula el voltaje RMS en microvoltios (µV) de la componente de 50 Hz (48-52 Hz)
    en un segmento de señal en reposo.
    """
    if noise_signal_uV is None or len(noise_signal_uV) < int(fs * 0.2):
        return 0.0
    try:
        n = len(noise_signal_uV)
        fft_vals = np.fft.rfft(noise_signal_uV - np.mean(noise_signal_uV))
        freqs = np.fft.rfftfreq(n, 1.0 / fs)
        idx_50 = np.where((freqs >= 48.0) & (freqs <= 52.0))[0]
        if len(idx_50) == 0:
            return 0.0
        power_50 = np.sum(np.abs(fft_vals[idx_50]) ** 2) / (n ** 2)
        return float(np.sqrt(2.0 * power_50))
    except Exception:
        return 0.0
