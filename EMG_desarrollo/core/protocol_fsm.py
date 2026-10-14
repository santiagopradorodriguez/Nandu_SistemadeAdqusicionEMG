# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Archivo: EMG_desarrollo/core/protocol_fsm.py
# Descripción: Máquina de estados finitos (FSM) con soporte de Pausa/Reanudar.
# ==============================================================================

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional
from PySide6.QtCore import QObject, QTimer, Signal


class ProtocolState(Enum):
    IDLE = auto()
    SILENCE_PREP = auto()
    BASAL_NOISE_INITIAL = auto()
    CVM_CALIBRATION_PREP = auto()
    CVM_CALIBRATION_TAKE = auto()
    METRONOME_COUNT_IN = auto()
    RECORDING_ACTIVE = auto()
    REST_AND_SAVE = auto()
    MUSCLE_VERIFICATION = auto()
    BASAL_NOISE_FINAL = auto()
    COMPLETED = auto()


class ProtocolMode(Enum):
    MANUAL = "manual"
    DISCRETE_WORDS = "discrete"
    CONTINUOUS_SEQUENCE = "continuous"


@dataclass
class ProtocolConfig:
    mode: ProtocolMode = ProtocolMode.DISCRETE_WORDS
    subject_id: str = "Sujeto1"
    trial_id: str = "Prueba1"
    words: List[str] = field(default_factory=lambda: ["A", "E", "I", "O", "U"])
    repetitions: int = 5
    bpm: int = 40
    noise_duration_s: float = 5.0
    enable_cvm: bool = True
    rest_duration_s: float = 5.0
    verification_duration_s: float = 5.0


class AcquisitionProtocolFSM(QObject):
    """
    Controlador determinista del protocolo con Pausa/Reanudación en cualquier fase.
    """
    state_changed = Signal(ProtocolState, str)
    pause_toggled = Signal(bool)                  # True = En Pausa, False = Reanudado
    overlay_updated = Signal(str, bool)           # (HTML, Visible)
    prompt_word_changed = Signal(str, str, str)   # (Anterior, Actual, Siguiente)
    save_word_requested = Signal(str, int)
    save_sequence_requested = Signal()
    protocol_finished = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.state = ProtocolState.IDLE
        self.cfg = ProtocolConfig()
        self.word_idx = 0
        self.cvm_sequence = [(v, t) for v in ("A", "I", "O") for t in (1, 2, 3)]
        self.cvm_idx = 0
        self._cvm_done = False

        self.is_paused = False
        self._paused_remaining_ms: int = 0
        self._last_overlay_html: str = ""
        self._last_overlay_visible: bool = False

        self._phase_timer = QTimer(self)
        self._phase_timer.setSingleShot(True)
        self._phase_timer.timeout.connect(self._on_phase_timeout)

    def start_protocol(self, config: ProtocolConfig) -> None:
        self.abort()
        self.cfg = config
        self.word_idx = 0
        self.cvm_idx = 0
        self.is_paused = False
        self._paused_remaining_ms = 0
        self._cvm_done = not config.enable_cvm
        self._transition_to(ProtocolState.SILENCE_PREP)

    def pause(self) -> None:
        """Pausa inmediatamente el temporizador de la fase actual sin perder el progreso."""
        if self.state == ProtocolState.IDLE or self.is_paused:
            return
        self.is_paused = True
        if self._phase_timer.isActive():
            self._paused_remaining_ms = max(50, self._phase_timer.remainingTime())
            self._phase_timer.stop()
        else:
            self._paused_remaining_ms = 0

        self.overlay_updated.emit(
            "<div align='center' style='color:#FFCC00;'>⏸ SESIÓN EN PAUSA<br>"
            "<span style='font-size:24px; color:#FFFFFF;'>Presiona REANUDAR para continuar</span></div>",
            True,
        )
        self.pause_toggled.emit(True)

    def resume(self) -> None:
        """Reanuda la fase exactamente con los milisegundos que faltaban antes de pausar."""
        if self.state == ProtocolState.IDLE or not self.is_paused:
            return
        self.is_paused = False
        self.overlay_updated.emit(self._last_overlay_html, self._last_overlay_visible)
        if self._paused_remaining_ms > 0:
            self._phase_timer.start(self._paused_remaining_ms)
            self._paused_remaining_ms = 0
        self.pause_toggled.emit(False)

    def toggle_pause(self) -> bool:
        if self.is_paused:
            self.resume()
        else:
            self.pause()
        return self.is_paused

    def abort(self) -> None:
        self._phase_timer.stop()
        self.is_paused = False
        self._paused_remaining_ms = 0
        if self.state != ProtocolState.IDLE:
            self.state = ProtocolState.IDLE
            self._set_overlay("", False)
            self.state_changed.emit(ProtocolState.IDLE, "Detenido")

    def _set_overlay(self, html: str, visible: bool) -> None:
        self._last_overlay_html = html
        self._last_overlay_visible = visible
        if not self.is_paused:
            self.overlay_updated.emit(html, visible)

    def _get_word_triplet(self, idx: int) -> tuple[str, str, str]:
        words = self.cfg.words
        if not words:
            return ("--", "--", "--")
        curr_w = words[idx % len(words)]
        prev_w = words[(idx - 1) % len(words)] if idx > 0 else "--"
        next_w = words[(idx + 1) % len(words)] if (idx + 1) < len(words) else "FIN"
        return (prev_w, curr_w, next_w)

    def notify_metronome_beat(self, beat_number: int) -> None:
        if self.is_paused:
            return
        if self.state in (ProtocolState.CVM_CALIBRATION_PREP, ProtocolState.CVM_CALIBRATION_TAKE):
            self._step_cvm_on_beat(beat_number)
        elif self.state == ProtocolState.RECORDING_ACTIVE and self.cfg.mode == ProtocolMode.CONTINUOUS_SEQUENCE:
            total_beats = len(self.cfg.words) * self.cfg.repetitions
            if beat_number > total_beats:
                self._transition_to(ProtocolState.BASAL_NOISE_FINAL)
            else:
                idx = beat_number - 1
                prev_w = self.cfg.words[(idx - 1) % len(self.cfg.words)] if idx > 0 else "--"
                curr_w = self.cfg.words[idx % len(self.cfg.words)]
                next_w = self.cfg.words[(idx + 1) % len(self.cfg.words)] if (idx + 1) < total_beats else "FIN"
                self.prompt_word_changed.emit(prev_w, curr_w, next_w)

    def _transition_to(self, new_state: ProtocolState) -> None:
        self._phase_timer.stop()
        self.state = new_state

        if new_state == ProtocolState.SILENCE_PREP:
            prev_w, curr_w, next_w = self._get_word_triplet(self.word_idx)
            self.prompt_word_changed.emit(prev_w, f"SIGUIENTE: {curr_w}", next_w)
            self._set_overlay("<div align='center'>HAZ SILENCIO<br>PREPARANDO ENTORNO...</div>", True)
            self.state_changed.emit(new_state, "Preparando entorno (3s)...")
            self._phase_timer.start(3000)

        elif new_state == ProtocolState.BASAL_NOISE_INITIAL:
            self._set_overlay("", False)
            self.prompt_word_changed.emit("--", "RUIDO BASAL", self.cfg.words[0] if self.cfg.words else "--")
            self.state_changed.emit(new_state, f"Grabando ruido basal ({self.cfg.noise_duration_s:.1f}s)...")
            self._phase_timer.start(int(self.cfg.noise_duration_s * 1000))

        elif new_state == ProtocolState.CVM_CALIBRATION_PREP:
            self.state_changed.emit(new_state, "Calibración de Contracción Muscular Máxima (CVM)")

        elif new_state == ProtocolState.METRONOME_COUNT_IN:
            beat_ms = int((60.0 / max(1, self.cfg.bpm)) * 1000)
            self._set_overlay("", False)
            self.state_changed.emit(new_state, "Escucha el metrónomo (Count-in)...")
            self._phase_timer.start(beat_ms * 3)

        elif new_state == ProtocolState.RECORDING_ACTIVE:
            self._set_overlay("", False)
            prev_w, curr_w, next_w = self._get_word_triplet(self.word_idx)
            self.prompt_word_changed.emit(prev_w, curr_w, next_w)
            if self.cfg.mode == ProtocolMode.DISCRETE_WORDS:
                self.state_changed.emit(new_state, f"Grabando palabra '{curr_w}' ({self.word_idx + 1}/{len(self.cfg.words)})")
                duration_ms = int((60.0 / max(1, self.cfg.bpm)) * self.cfg.repetitions * 1000)
                self._phase_timer.start(duration_ms)
            else:
                self.state_changed.emit(new_state, "Grabando secuencia continua...")

        elif new_state == ProtocolState.REST_AND_SAVE:
            word = self.cfg.words[self.word_idx]
            self.save_word_requested.emit(word, self.word_idx)
            self._set_overlay("<div align='center'>DESCANSO 5s<br>GUARDANDO DATOS...</div>", True)
            self.state_changed.emit(new_state, "Descanso y guardado...")
            self._phase_timer.start(int(self.cfg.rest_duration_s * 1000))

        elif new_state == ProtocolState.MUSCLE_VERIFICATION:
            self._set_overlay("", False)
            self.prompt_word_changed.emit("--", "VERIFICACIÓN MUSCULAR", "--")
            self.state_changed.emit(new_state, "Verificación muscular libre (5s)...")
            self._phase_timer.start(int(self.cfg.verification_duration_s * 1000))

        elif new_state == ProtocolState.BASAL_NOISE_FINAL:
            self._set_overlay("<div align='center'>HAZ SILENCIO<br>GRABANDO RUIDO FINAL...</div>", True)
            self.state_changed.emit(new_state, f"Grabando ruido final ({self.cfg.noise_duration_s:.1f}s)...")
            self._phase_timer.start(int(self.cfg.noise_duration_s * 1000))

        elif new_state == ProtocolState.COMPLETED:
            if self.cfg.mode == ProtocolMode.CONTINUOUS_SEQUENCE:
                self.save_sequence_requested.emit()
            self._set_overlay("<div align='center'>¡SESIÓN COMPLETADA!</div>", True)
            self.state_changed.emit(new_state, "Completado")
            self.protocol_finished.emit()

    def _on_phase_timeout(self) -> None:
        if self.is_paused:
            return
        if self.state == ProtocolState.SILENCE_PREP:
            self._transition_to(ProtocolState.BASAL_NOISE_INITIAL)
        elif self.state == ProtocolState.BASAL_NOISE_INITIAL:
            if not self._cvm_done:
                self._transition_to(ProtocolState.CVM_CALIBRATION_PREP)
            else:
                self._transition_to(ProtocolState.METRONOME_COUNT_IN)
        elif self.state == ProtocolState.METRONOME_COUNT_IN:
            self._transition_to(ProtocolState.RECORDING_ACTIVE)
        elif self.state == ProtocolState.RECORDING_ACTIVE:
            if self.cfg.mode == ProtocolMode.DISCRETE_WORDS:
                self._transition_to(ProtocolState.REST_AND_SAVE)
        elif self.state == ProtocolState.REST_AND_SAVE:
            self._transition_to(ProtocolState.MUSCLE_VERIFICATION)
        elif self.state == ProtocolState.MUSCLE_VERIFICATION:
            self.word_idx += 1
            if self.word_idx >= len(self.cfg.words):
                self._transition_to(ProtocolState.BASAL_NOISE_FINAL)
            else:
                self._transition_to(ProtocolState.SILENCE_PREP)
        elif self.state == ProtocolState.BASAL_NOISE_FINAL:
            self._transition_to(ProtocolState.COMPLETED)

    def _step_cvm_on_beat(self, beat_number: int) -> None:
        total_cvm_beats = len(self.cvm_sequence) * 2
        if beat_number > total_cvm_beats:
            self._cvm_done = True
            self._transition_to(ProtocolState.METRONOME_COUNT_IN)
            return

        self.cvm_idx = (beat_number - 1) // 2
        vocal, _ = self.cvm_sequence[self.cvm_idx]
        if beat_number % 2 == 1:
            self.state = ProtocolState.CVM_CALIBRATION_PREP
            self._set_overlay(
                f"<div align='center'>CVM Toma {self.cvm_idx + 1}/9<br>¡PREPÁRATE PARA /{vocal}/!</div>", True
            )
        else:
            self.state = ProtocolState.CVM_CALIBRATION_TAKE
            self._set_overlay(
                f"<div align='center' style='color:#00FFFF;'>¡CONTRAE AL MÁXIMO!<br>/{vocal}/</div>", True
            )
