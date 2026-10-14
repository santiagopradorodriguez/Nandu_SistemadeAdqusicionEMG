# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Archivo: EMG_desarrollo/core/preroll_and_filters.py
# Descripción: Buffer circular de pre-grabación (PreRoll) y cadena modular de filtros DSP.
# ==============================================================================

from __future__ import annotations
import collections
import threading
from typing import List, Optional, Tuple
import numpy as np

try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class PreRollRecordingBuffer:
    """
    Buffer FIFO de pre-grabación para rescatar automáticamente los N segundos
    previos al inicio de una toma manual o activación por trigger.
    Mantiene la coherencia de canales y evita consumo excesivo de memoria.
    """

    def __init__(self, sample_rate: int = 2000, preroll_seconds: float = 1.0):
        self.sample_rate = int(sample_rate)
        self.preroll_seconds = float(preroll_seconds)
        self._chunks: collections.deque[np.ndarray] = collections.deque()
        self._total_samples: int = 0
        self._lock = threading.Lock()

    def set_config(self, sample_rate: int, preroll_seconds: float) -> None:
        """Actualiza la tasa de muestreo y duración configurada del buffer."""
        with self._lock:
            self.sample_rate = max(1, int(sample_rate))
            self.preroll_seconds = max(0.0, float(preroll_seconds))
            self._prune()

    def clear(self) -> None:
        """Vacía el contenido acumulado en el buffer de pre-roll."""
        with self._lock:
            self._chunks.clear()
            self._total_samples = 0

    def push_raw_chunk(self, chunk: np.ndarray) -> None:
        """Agrega un bloque crudo entrante al buffer circular de pre-roll."""
        if chunk is None or chunk.size == 0:
            return
        with self._lock:
            if self.preroll_seconds <= 0.0:
                self._chunks.clear()
                self._total_samples = 0
                return

            self._chunks.append(chunk.copy())
            self._total_samples += chunk.shape[1]
            self._prune()

    def _prune(self) -> None:
        """Recorta bloques excedentes que sobrepasen el horizonte temporal."""
        max_samples = int(self.sample_rate * self.preroll_seconds)
        if max_samples <= 0:
            self._chunks.clear()
            self._total_samples = 0
            return

        while self._chunks and (self._total_samples - self._chunks[0].shape[1] >= max_samples):
            removed = self._chunks.popleft()
            self._total_samples -= removed.shape[1]

    def dump_preroll_for_recording(self) -> List[np.ndarray]:
        """
        Extrae y devuelve el bloque continuo exacto de pre-roll en una lista
        compatible con la estructura de chunks de self.current_recording.
        """
        with self._lock:
            if not self._chunks or self.preroll_seconds <= 0.0:
                return []

            max_samples = int(self.sample_rate * self.preroll_seconds)
            concatenated = np.concatenate(list(self._chunks), axis=1)

            if concatenated.shape[1] > max_samples:
                concatenated = concatenated[:, -max_samples:]

            return [concatenated.copy()]


class ConfigurableFilterChain:
    """
    Cadena modular de filtros digitales IIR (Notch, Butterworth pasa-banda y Peine armónico).
    Calcula coeficientes SOS y mantiene el estado inicial zi por canal.
    """

    def __init__(self, sample_rate: int = 2000, num_channels: int = 3):
        self.sample_rate = int(sample_rate)
        self.num_channels = int(num_channels)
        self.notch_q = 2.0
        self.enable_comb = False
        self.notch_sos: Optional[np.ndarray] = None
        self.notch_zi: Optional[np.ndarray] = None
        self.bandpass_sos: Optional[np.ndarray] = None
        self.bandpass_zi: Optional[np.ndarray] = None

    def configure_notch(self, f0: float = 50.0, q: float = 2.0, enable_comb: bool = False) -> None:
        """Configura el filtro Notch para la red eléctrica y armónicos impares opcionales."""
        if not SCIPY_AVAILABLE:
            return

        self.notch_q = float(q)
        self.enable_comb = bool(enable_comb)
        nyquist = 0.5 * self.sample_rate

        frequencies = [f0]
        if self.enable_comb:
            frequencies.extend([150.0, 250.0, 350.0])

        sos_list = []
        for freq in frequencies:
            if freq < nyquist:
                b, a = signal.iirnotch(freq, self.notch_q, fs=self.sample_rate)
                sos_list.append(signal.tf2sos(b, a))

        if sos_list:
            self.notch_sos = np.vstack(sos_list)
            zi_single = signal.sosfilt_zi(self.notch_sos)
            self.notch_zi = np.stack([zi_single] * self.num_channels, axis=-1)
        else:
            self.notch_sos = None
            self.notch_zi = None

    def configure_bandpass(self, low_cut: float = 20.0, high_cut: float = 500.0, order: int = 4) -> None:
        """Configura el filtro pasa-banda Butterworth."""
        if not SCIPY_AVAILABLE:
            return

        nyquist = 0.5 * self.sample_rate
        low = max(0.1, low_cut) / nyquist
        high = min(nyquist * 0.99, high_cut) / nyquist

        if low >= high:
            self.bandpass_sos = None
            self.bandpass_zi = None
            return

        self.bandpass_sos = signal.butter(order, [low, high], btype='band', output='sos')
        zi_single = signal.sosfilt_zi(self.bandpass_sos)
        self.bandpass_zi = np.stack([zi_single] * self.num_channels, axis=-1)
