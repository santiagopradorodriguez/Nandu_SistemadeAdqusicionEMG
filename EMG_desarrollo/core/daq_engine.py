# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Archivo: EMG_desarrollo/core/daq_engine.py
# Descripción: Motor DAQ unificado (NI-DAQmx, Mic, Simulador, WAV) con RingBuffer
#              y soporte nativo de Pausa sin desbordamiento de buffer de hardware.
# ==============================================================================

from __future__ import annotations
import os
import time
import queue
import threading
from enum import Enum
from typing import List, Optional
import numpy as np
from scipy.io import wavfile
from PySide6.QtCore import QThread, Signal

try:
    import nidaqmx
    from nidaqmx.constants import AcquisitionType, TerminalConfiguration
    from nidaqmx.stream_readers import AnalogMultiChannelReader
    NIDAQMX_DISPONIBLE = True
except Exception:
    NIDAQMX_DISPONIBLE = False

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False


class AcquisitionSource(Enum):
    NIDAQ = "nidaq"
    MICROPHONE = "microphone"
    SIMULATOR = "simulator"
    WAV_PLAYBACK = "wav_playback"


class RingBuffer:
    """Buffer circular basado en NumPy estático para evitar recolección de basura."""
    def __init__(self, capacity: int, num_channels: int):
        self.capacity = int(capacity)
        self.num_channels = int(num_channels)
        self.buffer = np.zeros((self.num_channels, self.capacity), dtype=np.float64)
        self.head = 0
        self.is_full = False
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self.buffer.fill(0.0)
            self.head = 0
            self.is_full = False

    def push(self, data: np.ndarray) -> None:
        samples = data.shape[1]
        if samples == 0:
            return
        with self._lock:
            if samples >= self.capacity:
                self.buffer[:, :] = data[:, -self.capacity:]
                self.head = 0
                self.is_full = True
                return

            if self.head + samples <= self.capacity:
                self.buffer[:, self.head:self.head + samples] = data
                self.head += samples
                if self.head == self.capacity:
                    self.head = 0
                    self.is_full = True
            else:
                first_part = self.capacity - self.head
                self.buffer[:, self.head:self.capacity] = data[:, :first_part]
                second_part = samples - first_part
                self.buffer[:, 0:second_part] = data[:, first_part:]
                self.head = second_part
                self.is_full = True

    def get_latest(self, num_samples: int) -> np.ndarray:
        with self._lock:
            n = min(int(num_samples), self.capacity)
            if self.head >= n:
                return self.buffer[:, self.head - n:self.head].copy()
            if not self.is_full:
                return self.buffer[:, :self.head].copy()
            first_part = n - self.head
            res = np.empty((self.num_channels, n), dtype=np.float64)
            res[:, :first_part] = self.buffer[:, self.capacity - first_part:self.capacity]
            res[:, first_part:] = self.buffer[:, :self.head]
            return res


class UnifiedAcquisitionWorker(QThread):
    """
    Hilo dedicado de adquisición y reproducción con soporte de Pausa en tiempo real.
    - En modo WAV_PLAYBACK y SIMULATOR: cuando está en pausa, congela el avance temporal.
    - En modo NIDAQ y MICROPHONE: sigue drenando el hardware para evitar Buffer Overflow,
      pero si `freeze_stream_on_pause=True` no encola paquetes en data_queue.
    """
    error_occurred = Signal(str)
    playback_finished = Signal()
    pause_state_changed = Signal(bool)

    def __init__(
        self,
        source: AcquisitionSource,
        sample_rate: int,
        num_channels: int,
        data_queue: queue.Queue,
        chunk_samples: Optional[int] = None,
        daq_channels: Optional[List[str]] = None,
        terminal_mode: str = "RSE",
        sim_freq_hz: float = 50.0,
        wav_paths: Optional[List[str]] = None,
        loop_wav: bool = False,
        freeze_stream_on_pause: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.sample_rate = int(sample_rate)
        self.num_channels = int(num_channels)
        self.data_queue = data_queue
        self.chunk_samples = chunk_samples or max(1, int(self.sample_rate * 0.05))
        self.daq_channels = daq_channels or [f"Dev1/ai{i}" for i in range(self.num_channels)]
        self.terminal_mode = terminal_mode
        self.sim_freq_hz = float(sim_freq_hz)
        self.wav_paths = wav_paths or []
        self.loop_wav = bool(loop_wav)
        self.freeze_stream_on_pause = bool(freeze_stream_on_pause)

        self.stop_event = threading.Event()
        self._is_paused = False
        self._pause_lock = threading.Lock()

    @property
    def is_paused(self) -> bool:
        with self._pause_lock:
            return self._is_paused

    def set_paused(self, paused: bool) -> None:
        with self._pause_lock:
            self._is_paused = bool(paused)
        self.pause_state_changed.emit(self._is_paused)

    def toggle_pause(self) -> bool:
        new_state = not self.is_paused
        self.set_paused(new_state)
        return new_state

    def stop(self) -> None:
        self.stop_event.set()
        self.wait(2000)

    def run(self) -> None:
        self.stop_event.clear()
        try:
            if self.source == AcquisitionSource.NIDAQ:
                self._run_nidaq()
            elif self.source == AcquisitionSource.MICROPHONE:
                self._run_microphone()
            elif self.source == AcquisitionSource.WAV_PLAYBACK:
                self._run_wav_playback()
            else:
                self._run_simulator()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def _run_nidaq(self) -> None:
        if not NIDAQMX_DISPONIBLE:
            raise RuntimeError("nidaqmx no está disponible en este entorno.")

        modes = {
            "RSE": TerminalConfiguration.RSE,
            "DIFF": TerminalConfiguration.DIFF,
            "NRSE": TerminalConfiguration.NRSE,
            "DEFAULT": TerminalConfiguration.DEFAULT,
        }
        t_cfg = modes.get(self.terminal_mode.upper(), TerminalConfiguration.RSE)
        buffer_daq = np.zeros((self.num_channels, self.chunk_samples), dtype=np.float64)

        with nidaqmx.Task() as task:
            for ch in self.daq_channels:
                task.ai_channels.add_ai_voltage_chan(ch, terminal_config=t_cfg, min_val=-10.0, max_val=10.0)

            task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=max(self.sample_rate * 2, self.chunk_samples * 10),
            )
            reader = AnalogMultiChannelReader(task.in_stream)
            task.start()

            while not self.stop_event.is_set():
                try:
                    reader.read_many_sample(
                        buffer_daq,
                        number_of_samples_per_channel=self.chunk_samples,
                        timeout=max(0.2, (self.chunk_samples / self.sample_rate) * 5.0),
                    )
                    # Siempre leemos de la placa para evitar overflow (-200279)
                    if not (self.is_paused and self.freeze_stream_on_pause):
                        self.data_queue.put(buffer_daq.copy())
                except nidaqmx.errors.DaqError as e:
                    if e.error_code == -200284:
                        continue
                    raise

    def _run_microphone(self) -> None:
        if not SD_AVAILABLE:
            raise RuntimeError("La librería 'sounddevice' no está disponible.")

        def audio_callback(indata, frames, time_info, status):
            if not self.stop_event.is_set():
                if self.is_paused and self.freeze_stream_on_pause:
                    return
                data = indata[:, 0]
                buf = np.zeros((self.num_channels, len(data)), dtype=np.float64)
                for i in range(self.num_channels):
                    buf[i, :] = data
                self.data_queue.put(buf)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, callback=audio_callback, blocksize=self.chunk_samples):
            while not self.stop_event.is_set():
                time.sleep(0.05)

    def _run_simulator(self) -> None:
        samples_generados = 0
        dt = 1.0 / self.sample_rate
        t_chunk = self.chunk_samples * dt

        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            if self.is_paused and self.freeze_stream_on_pause:
                time.sleep(0.05)
                continue

            t_actual = samples_generados * dt
            t_vec = np.linspace(t_actual, t_actual + t_chunk, self.chunk_samples, endpoint=False)

            senales = [1.0 * np.sin(2.0 * np.pi * self.sim_freq_hz * t_vec)]
            if self.num_channels > 1:
                senales.append(0.8 * np.sin(2.0 * np.pi * 120.0 * t_vec))
            if self.num_channels > 2:
                senales.append(0.6 * np.sin(2.0 * np.pi * 50.0 * t_vec) + 0.4 * np.sin(2.0 * np.pi * 250.0 * t_vec))
            for i in range(3, self.num_channels):
                senales.append(0.7 * np.sin(2.0 * np.pi * (100 + i * 110) * t_vec))

            self.data_queue.put(np.array(senales, dtype=np.float64))
            samples_generados += self.chunk_samples

            elapsed = time.perf_counter() - t0
            if t_chunk - elapsed > 0:
                time.sleep(t_chunk - elapsed)

    def _run_wav_playback(self) -> None:
        signals = []
        min_len = float("inf")
        for p in self.wav_paths:
            if os.path.exists(p):
                _, d = wavfile.read(p)
                signals.append(d.astype(np.float32))
                if len(d) < min_len:
                    min_len = len(d)

        if not signals or min_len == float("inf"):
            raise RuntimeError("No se pudieron cargar las pistas WAV.")

        sig_all = np.stack([s[:int(min_len)] for s in signals], axis=0)
        total_muestras = sig_all.shape[1]
        idx = 0
        t_chunk_s = self.chunk_samples / self.sample_rate

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()
            if idx + self.chunk_samples <= total_muestras:
                chunk = sig_all[:, idx:idx + self.chunk_samples]
                idx += self.chunk_samples
                self.data_queue.put(chunk)
            elif self.loop_wav:
                idx = 0
                continue
            else:
                rem = total_muestras - idx
                if rem > 0:
                    chunk = np.zeros((sig_all.shape[0], self.chunk_samples), dtype=np.float32)
                    chunk[:, :rem] = sig_all[:, idx:]
                    self.data_queue.put(chunk)
                self.playback_finished.emit()
                break

            elapsed = time.perf_counter() - t_start
            if t_chunk_s - elapsed > 0:
                time.sleep(t_chunk_s - elapsed)
