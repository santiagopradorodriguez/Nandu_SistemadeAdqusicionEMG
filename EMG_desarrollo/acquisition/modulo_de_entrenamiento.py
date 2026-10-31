# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning (v6.0.0)
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de Entrenamiento Interactivo v6.0.0 para practicar tanto
#              Autograbado Discreto como Secuencia Continua (con Secuencias
#              Balanceadas De Bruijn, Modo Jitter Anti-Sincronización, 4 sub-pulsos
#              a 30 BPM por defecto, Destello Visual en la vocal, Pre-aviso de
#              cambio en el último sub-pulso y Guía de Voz con SPEAKING: ON/MUTE).
# ==============================================================================

import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

__version__ = "6.0.0"

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    SD_AVAILABLE = False


def get_project_root() -> str:
    if getattr(sys, "frozen", False):
        base = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(base) == "_internal":
            base = os.path.dirname(base)
        return base
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def play_beep_training(freq: int = 1000, duration_ms: int = 100) -> None:
    """Reproduce un beep corto de baja latencia usando utils.sound_utils, sounddevice o winsound."""
    try:
        from utils.sound_utils import play_beep
        play_beep(freq, duration_ms, async_play=False)
        return
    except Exception:
        pass
    try:
        if SD_AVAILABLE:
            sr = 22050
            t = np.linspace(0.0, duration_ms / 1000.0, int(sr * (duration_ms / 1000.0)), endpoint=False)
            wave = 0.25 * np.sin(2.0 * np.pi * float(freq) * t).astype(np.float32)
            sd.play(wave, sr, blocking=True)
            return
        if sys.platform == "win32":
            import winsound
            winsound.Beep(int(freq), int(duration_ms))
    except Exception:
        pass


def generar_secuencia_balanceada_debruijn(
    vocales_base: List[str], num_ciclos: int = 5, seed: Optional[int] = None
) -> List[str]:
    """
    Construye una secuencia balanceada de longitud len(vocales_base) * num_ciclos
    donde:
      1. Cada vocal aparece exactamente `num_ciclos` veces.
      2. Todas las transiciones dirigidas (v_i -> v_j, con v_i != v_j) ocurren de forma
         equitativa (bloques balanceados de Williams / grafo euleriano de De Bruijn)
         para eliminar el sesgo de coarticulación en el entrenamiento de Deep Learning.
      3. Nunca se repite la misma vocal en la frontera entre dos bloques consecutivos.
    """
    items = [str(v).strip().upper() for v in vocales_base if str(v).strip()]
    if not items:
        items = ["A", "E", "I", "O", "U"]
    n = len(items)
    if n == 1 or num_ciclos <= 0:
        return items * max(1, num_ciclos)

    rng = random.Random(seed)
    # Matriz de conteo de transiciones dirigidas usadas (i -> j)
    trans_counts = [[0] * n for _ in range(n)]
    secuencia_indices: List[int] = []

    for c in range(num_ciclos):
        if c == 0:
            perm = list(range(n))
            rng.shuffle(perm)
        else:
            prev_last = secuencia_indices[-1]
            # Construir permutación del ciclo buscando minimizar las transiciones ya usadas
            disponibles = list(range(n))
            candidatas_inicio = [idx for idx in disponibles if idx != prev_last]
            candidatas_inicio.sort(key=lambda idx: (trans_counts[prev_last][idx], rng.random()))
            primero = candidatas_inicio[0]
            perm = [primero]
            disponibles.remove(primero)
            while disponibles:
                actual = perm[-1]
                disponibles.sort(key=lambda idx: (trans_counts[actual][idx], rng.random()))
                sig = disponibles.pop(0)
                perm.append(sig)

        for idx in perm:
            if secuencia_indices:
                ant = secuencia_indices[-1]
                trans_counts[ant][idx] += 1
            secuencia_indices.append(idx)

    return [items[i] for i in secuencia_indices]


def generate_jittered_intervals(
    base_bpm: float,
    num_beats: int,
    jitter_pct: float = 25.0,
    mode: str = "exponential_hazard",
    seed: Optional[int] = None,
) -> List[float]:
    """Genera intervalos tau_k con tasa de riesgo no anticipable (exponential_hazard, uniform, gaussian)."""
    tau_0 = 60.0 / max(1.0, float(base_bpm))
    delta = max(0.0, min(0.50, float(jitter_pct) / 100.0))
    if delta <= 0.0 or num_beats <= 0:
        return [round(tau_0, 4) for _ in range(max(1, num_beats))]

    rng = np.random.default_rng(seed)
    tau_min = tau_0 * (1.0 - delta)
    tau_max = tau_0 * (1.0 + delta)
    intervals: List[float] = []
    for _ in range(num_beats):
        if mode == "uniform":
            val = float(rng.uniform(tau_min, tau_max))
        elif mode == "gaussian":
            sigma = (tau_0 * delta) / 2.0
            val = float(np.clip(rng.normal(tau_0, sigma), tau_min, tau_max))
        else:
            lam = (tau_0 - tau_min) * 0.65
            val = float(np.clip(tau_min + rng.exponential(max(0.01, lam)), tau_min, tau_max))
        intervals.append(round(val, 4))
    return intervals


class TrainingVoiceCuePlayer:
    """Reproductor asíncrono de voz guía para el módulo de entrenamiento con soporte SPEAKING: ON/MUTE."""

    def __init__(self) -> None:
        self.enabled: bool = True
        self.muted: bool = (os.environ.get("NANDU_MUTE_SPEAKING") == "1")
        self.speaking_mode: str = "fases_y_continua"
        self.last_speech_end_perf: float = 0.0
        self._cache = {}
        self._queue: queue.Queue = queue.Queue()
        root = get_project_root()
        self._search_dirs = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "voice_cues"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "voz_asistente"),
            os.path.join(root, "assets", "voice_cues"),
            os.path.join(root, "voice_cues"),
            os.path.join(root, "assets", "sounds"),
            os.path.join(root, "sounds"),
        ]
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        os.environ["NANDU_MUTE_SPEAKING"] = "1" if self.muted else "0"
        if self.muted:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            try:
                if SD_AVAILABLE:
                    sd.stop()
            except Exception:
                pass

    def say(self, cue_key: str, fallback_text: str = "", fase: str = "fase") -> None:
        if not self.enabled or self.muted or self.speaking_mode == "mute":
            return
        if fase == "pulso_continua" and self.speaking_mode == "solo_fases":
            return
        try:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self.last_speech_end_perf = time.perf_counter() + 0.45
            self._queue.put_nowait((str(cue_key).strip().lower(), str(fallback_text or cue_key)))
        except Exception:
            pass

    def _find_wav(self, key: str) -> Optional[str]:
        if key in self._cache:
            return self._cache[key]
        slug = re.sub(r"[^a-z0-9_]", "_", key.lower())
        candidates = [f"{slug}.wav", f"vocal_{slug}.wav", f"cue_{slug}.wav"]
        if slug.startswith("di_") and len(slug) > 3:
            base_v = slug[3:]
            candidates.extend([f"di_{base_v}.wav", f"{base_v}.wav"])
        for d in self._search_dirs:
            if os.path.isdir(d):
                for cname in candidates:
                    cand = os.path.join(d, cname)
                    if os.path.isfile(cand):
                        self._cache[key] = cand
                        return cand
        self._cache[key] = None
        return None

    def _run_worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            key, text = item
            if not self.enabled or self.muted:
                continue
            wav_path = self._find_wav(key)
            if wav_path and os.path.isfile(wav_path):
                try:
                    if SD_AVAILABLE:
                        from scipy.io import wavfile
                        sr_w, data_w = wavfile.read(wav_path)
                        if not self.muted:
                            sd.play(data_w, sr_w, blocking=False)
                        continue
                    elif sys.platform == "win32" and not self.muted:
                        import winsound
                        winsound.PlaySound(wav_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
                        continue
                except Exception:
                    pass
            try:
                import utils.sound_utils as _su
                for fn_name in ("play_voice_cue", "speak_async", "say_text", "speak_text"):
                    fn = getattr(_su, fn_name, None)
                    if callable(fn) and not self.muted:
                        fn(text)
                        break
            except Exception:
                pass


class ModuloEntrenamiento(QWidget):
    """
    Entrenador Interactivo AutoForge v6.0.0:
    Permite al sujeto practicar tanto Secuencia Continua (con balanceo De Bruijn)
    como Autograbado Discreto y Calibración CVM exactamente con el mismo visor de
    destello visual, tira de secuencia, metrónomo con subdivisiones, Jitter Anti-Sync
    y locución anticipada en el último sub-pulso previo.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"Ñandú LSD - Entrenador Interactivo de Protocolo sEMG v{__version__}")
        self.resize(980, 680)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setStyleSheet(
            """
            QWidget { background-color: #06080e; color: #00FFCC; font-family: 'Courier New', monospace; font-weight: bold; }
            QGroupBox { border: 1.5px solid #00FFFF; border-radius: 6px; margin-top: 8px; padding-top: 10px; color: #00FFFF; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QSpinBox, QComboBox { background-color: #111622; color: #00FFCC; border: 1px solid #00FFFF; padding: 4px; border-radius: 3px; }
            QPushButton { background-color: #111827; color: #00FFCC; border: 1.5px solid #00FFCC; padding: 8px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #00FFCC; color: #050508; }
            QPushButton:disabled { border-color: #444444; color: #555555; }
            """
        )

        self.voice_player = TrainingVoiceCuePlayer()
        self.beep_queue: queue.Queue = queue.Queue()
        self._beep_worker = threading.Thread(target=self._run_beep_worker, daemon=True)
        self._beep_worker.start()

        # Estado del metrónomo de alta precisión (monótono)
        self.metro_timer = QTimer(self)
        self.metro_timer.setTimerType(Qt.PreciseTimer)
        self.metro_timer.setSingleShot(True)
        self.metro_timer.timeout.connect(self.on_metro_beat)

        self.phase_timer = QTimer(self)
        self.phase_timer.setSingleShot(True)

        self.is_running: bool = False
        self.current_phase: str = "IDLE"  # "RUIDO", "COUNT_IN", "FONACION", "DESCANSO"
        self.base_words: List[str] = self.get_words()
        self.practice_sequence: List[str] = []
        self.current_word_idx: int = 0
        self.metro_beat_count: int = 0
        self.metro_subdiv_index: int = 0
        self.metro_count_in_remaining: int = 0
        self._metro_next_perf: Optional[float] = None
        self.jitter_schedule: List[float] = []
        self.current_beat_tau_s: float = 2.0
        self._ultimo_estado_visor: Optional[Tuple] = None
        self._last_spoken_word: Optional[str] = None
        self._pre_announced_beat: int = -1
        self.invalid_pulses_marked: List[int] = []

        self._build_ui()

    def _run_beep_worker(self) -> None:
        while True:
            item = self.beep_queue.get()
            if item is None:
                break
            freq, dur = item
            play_beep_training(freq, dur)

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Barra superior de transporte
        top_bar = QFrame()
        top_bar.setStyleSheet("QFrame { background-color: #0b0f19; border: 1px solid #00FFFF; border-radius: 5px; }")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(8, 6, 8, 6)

        self.btn_start = QPushButton("INICIAR PRÁCTICA")
        self.btn_start.setStyleSheet(
            "QPushButton { background-color: #003d29; color: #39FF14; border: 2px solid #39FF14; font-size: 13px; }"
            "QPushButton:hover { background-color: #39FF14; color: #050508; }"
        )
        self.btn_start.clicked.connect(self.start_training)

        self.btn_stop = QPushButton("DETENER")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #3a0d16; color: #FF4455; border: 2px solid #FF4455; font-size: 13px; }"
            "QPushButton:hover { background-color: #FF4455; color: #050508; }"
        )
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)

        self.btn_mark_invalid = QPushButton("MARCAR PULSO INVÁLIDO [Backspace]")
        self.btn_mark_invalid.setToolTip("Marca el pulso actual como inválido (tos/error) sin detener la práctica")
        self.btn_mark_invalid.setStyleSheet(
            "QPushButton { background-color: #2b1d00; color: #FFCC00; border: 1.5px solid #FFCC00; }"
            "QPushButton:hover { background-color: #FFCC00; color: #050508; }"
        )
        self.btn_mark_invalid.clicked.connect(self.mark_current_pulse_invalid)
        self.btn_mark_invalid.setEnabled(False)

        self.btn_mute_speaking = QPushButton("SPEAKING: MUTE" if self.voice_player.muted else "SPEAKING: ON")
        self.btn_mute_speaking.setCheckable(True)
        self.btn_mute_speaking.setChecked(self.voice_player.muted)
        self.btn_mute_speaking.setStyleSheet(
            "QPushButton { background-color: #0f2419; color: #39FF14; border: 1.5px solid #39FF14; padding: 7px 10px; } "
            "QPushButton:checked { background-color: #3a0d16; color: #FF4455; border: 1.5px solid #FF4455; }"
        )
        self.btn_mute_speaking.toggled.connect(self.on_mute_speaking_toggled)

        self.lbl_status = QLabel("LISTO PARA ENTRENAR")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("color: #00FFFF; font-size: 13px; background-color: #050508; padding: 4px 10px; border: 1px solid #222;")

        top_layout.addWidget(self.btn_start)
        top_layout.addWidget(self.btn_stop)
        top_layout.addWidget(self.btn_mark_invalid)
        top_layout.addWidget(self.btn_mute_speaking)
        top_layout.addWidget(self.lbl_status, stretch=1)
        main_layout.addWidget(top_bar)

        # Panel central: Configuración a la izquierda + Metrónomo a la derecha
        mid_layout = QHBoxLayout()
        mid_layout.setSpacing(10)

        cfg_group = QGroupBox("Configuración del Protocolo de Práctica (v6.0.0)")
        form = QFormLayout(cfg_group)
        form.setSpacing(6)

        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems([
            "Secuencia Continua (De Bruijn / Sin Pausas)",
            "Autograbado Discreto (Palabra por Palabra + Descanso)",
        ])
        form.addRow("Modo de Protocolo:", self.cmb_mode)

        bpm_sub_row = QHBoxLayout()
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(20, 200)
        self.spin_bpm.setValue(30)  # Por defecto 30 BPM
        self.spin_subdivs = QSpinBox()
        self.spin_subdivs.setRange(1, 8)
        self.spin_subdivs.setValue(4)  # Por defecto 4 subdivisiones
        bpm_sub_row.addWidget(QLabel("BPM:"))
        bpm_sub_row.addWidget(self.spin_bpm)
        bpm_sub_row.addWidget(QLabel("Subdivisiones:"))
        bpm_sub_row.addWidget(self.spin_subdivs)
        form.addRow("Ritmo del Metrónomo:", bpm_sub_row)

        reps_noise_row = QHBoxLayout()
        self.spin_reps = QSpinBox()
        self.spin_reps.setRange(1, 100)
        self.spin_reps.setValue(5)
        self.spin_noise_s = QSpinBox()
        self.spin_noise_s.setRange(1, 20)
        self.spin_noise_s.setValue(3)
        self.spin_rest_s = QSpinBox()
        self.spin_rest_s.setRange(1, 30)
        self.spin_rest_s.setValue(5)
        reps_noise_row.addWidget(QLabel("Ciclos/Reps:"))
        reps_noise_row.addWidget(self.spin_reps)
        reps_noise_row.addWidget(QLabel("Ruido (s):"))
        reps_noise_row.addWidget(self.spin_noise_s)
        reps_noise_row.addWidget(QLabel("Descanso Disc (s):"))
        reps_noise_row.addWidget(self.spin_rest_s)
        form.addRow("Duraciones:", reps_noise_row)

        self.chk_debruijn = QCheckBox("Balancear Transiciones (Secuencia De Bruijn Anti-Coarticulación)")
        self.chk_debruijn.setChecked(True)
        self.chk_debruijn.setStyleSheet("color: #00FFCC;")
        form.addRow("Orden de Vocales:", self.chk_debruijn)

        jit_row = QHBoxLayout()
        self.chk_jitter = QCheckBox("Activar Jitter Anti-Sync")
        self.chk_jitter.setChecked(False)
        self.chk_jitter.setStyleSheet("color: #FF55FF;")
        self.spin_jitter_pct = QSpinBox()
        self.spin_jitter_pct.setRange(5, 50)
        self.spin_jitter_pct.setSingleStep(5)
        self.spin_jitter_pct.setValue(25)
        self.cmb_jitter_mode = QComboBox()
        self.cmb_jitter_mode.addItems(["exponential_hazard", "uniform", "gaussian"])
        jit_row.addWidget(self.chk_jitter)
        jit_row.addWidget(QLabel("+/- %:"))
        jit_row.addWidget(self.spin_jitter_pct)
        jit_row.addWidget(self.cmb_jitter_mode)
        form.addRow("Modo Jitter:", jit_row)

        self.cmb_speaking_mode = QComboBox()
        self.cmb_speaking_mode.addItems([
            "Fases + 'Di Vocal' Anticipado en Sub-pulso Previo",
            "Solo en Ruido, Preparación y Descanso (Sin voz en pulsos)",
            "Silenciado (Mute Speaking)",
        ])
        self.cmb_speaking_mode.currentIndexChanged.connect(self._on_speaking_mode_changed)
        form.addRow("Locución (Speaking):", self.cmb_speaking_mode)

        mid_layout.addWidget(cfg_group, stretch=3)

        # Panel derecho: Metrónomo visual
        metro_group = QGroupBox("Metrónomo Visual")
        metro_group.setMinimumWidth(200)
        metro_v = QVBoxLayout(metro_group)
        self.metro_pulse_frame = QFrame()
        self.metro_pulse_frame.setFixedHeight(64)
        self.metro_pulse_frame.setStyleSheet("background-color: #111111; border: 2px solid #00FFFF; border-radius: 6px;")
        self.metro_lbl_title = QLabel("EN ESPERA")
        self.metro_lbl_title.setAlignment(Qt.AlignCenter)
        self.metro_lbl_title.setStyleSheet("font-size: 15px; color: #00FFFF;")
        self.metro_lbl_count = QLabel("0")
        self.metro_lbl_count.setAlignment(Qt.AlignCenter)
        self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #00FFFF; font-weight: 900;")
        metro_v.addWidget(self.metro_pulse_frame)
        metro_v.addWidget(self.metro_lbl_title)
        metro_v.addWidget(self.metro_lbl_count)
        mid_layout.addWidget(metro_group, stretch=1)

        main_layout.addLayout(mid_layout)

        # Visor Gigante de Vocales y Destello Visual (idéntico al de autoforge_daq_experimental.py v6.0)
        self.lbl_visor_vocal = QLabel(
            "<div align='center' style='line-height:1.2;'>"
            "<span style='font-size:24px; color:#00FFFF;'>ENTRENADOR INTERACTIVO ÑANDÚ LSD v6.0</span><br>"
            "<span style='font-size:15px; color:#AAAAAA;'>Presiona <b>INICIAR PRÁCTICA</b> para entrenar Ruido Basal, Count-In, "
            "Destello Visual de Vocales y Transiciones De Bruijn</span>"
            "</div>"
        )
        self.lbl_visor_vocal.setAlignment(Qt.AlignCenter)
        self.lbl_visor_vocal.setMinimumHeight(260)
        self.lbl_visor_vocal.setStyleSheet(
            "background-color: #050510; color: #00FFCC; border: 2px solid #00FFFF; border-radius: 8px; padding: 10px;"
        )
        main_layout.addWidget(self.lbl_visor_vocal, stretch=1)

    def get_words(self) -> List[str]:
        root = get_project_root()
        candidates = [
            os.path.join(root, "palabras.txt"),
            os.path.join(root, "acquisition", "palabras.txt"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "palabras.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        words = [l.strip().upper() for l in f if l.strip()]
                    if words:
                        return words
                except Exception:
                    pass
        return ["A", "E", "I", "O", "U"]

    def on_mute_speaking_toggled(self, muted: bool) -> None:
        self.voice_player.set_muted(muted)
        self.btn_mute_speaking.setText("SPEAKING: MUTE" if muted else "SPEAKING: ON")

    def _on_speaking_mode_changed(self, idx: int) -> None:
        if idx == 1:
            self.voice_player.speaking_mode = "solo_fases"
            self.btn_mute_speaking.setChecked(False)
        elif idx == 2:
            self.voice_player.speaking_mode = "mute"
            self.btn_mute_speaking.setChecked(True)
        else:
            self.voice_player.speaking_mode = "fases_y_continua"
            self.btn_mute_speaking.setChecked(False)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (Qt.Key_Backspace, Qt.Key_Delete, Qt.Key_Space):
            if self.is_running and self.current_phase == "FONACION":
                self.mark_current_pulse_invalid()
                return
        super().keyPressEvent(event)

    def mark_current_pulse_invalid(self) -> None:
        if not self.is_running or self.metro_beat_count <= 0:
            return
        if self.metro_beat_count not in self.invalid_pulses_marked:
            self.invalid_pulses_marked.append(self.metro_beat_count)
        self.lbl_status.setText(f"[MARCADO INVÁLIDO] Pulso #{self.metro_beat_count} registrado como error/tos")
        self.lbl_status.setStyleSheet("color: #FFCC00; font-size: 13px; background-color: #2b1d00; padding: 4px 10px; border: 1px solid #FFCC00;")

    def _construir_tira_secuencia_html(self, words: List[str], active_idx: int) -> str:
        if not words:
            return ""
        vocal_colores = {"A": "#FF3355", "E": "#00B4D8", "I": "#39FF14", "O": "#C77DFF", "U": "#FFD166"}
        chips = []
        # Mostrar ventana deslizante de hasta 8 ítems para no saturar si es una secuencia larga De Bruijn
        n_w = len(words)
        start_i = max(0, min(active_idx - 2, max(0, n_w - 8)))
        end_i = min(n_w, start_i + 8)
        for idx in range(start_i, end_i):
            w_up = str(words[idx]).upper()
            c = vocal_colores.get(w_up, "#00FFCC")
            if idx == active_idx:
                chips.append(
                    f"<span style='background-color:{c}; color:#050508; font-weight:900; font-size:18px; "
                    f"padding:2px 10px; border:2px solid #FFFFFF; border-radius:4px;'>&nbsp;/{w_up}/&nbsp;</span>"
                )
            elif idx < active_idx:
                chips.append(f"<span style='color:#555555; font-size:15px;'>/{w_up}/</span>")
            else:
                chips.append(f"<span style='color:{c}; font-size:16px; font-weight:bold;'>/{w_up}/</span>")
        return " &rarr; ".join(chips)

    def _renderizar_visor_vocal(
        self,
        palabra_actual: str,
        pulso_actual: int,
        total_pulsos: int,
        palabra_siguiente: str = "",
        ciclo_actual: int = 1,
        total_ciclos: int = 1,
        flash: bool = False,
        pre_cambio: bool = False,
    ) -> None:
        p_act = str(palabra_actual or "").upper()
        p_sig = str(palabra_siguiente or "").upper()
        vocal_colores = {"A": "#FF3355", "E": "#00B4D8", "I": "#39FF14", "O": "#C77DFF", "U": "#FFD166"}
        vocal_rgb = {"A": "255, 51, 85", "E": "0, 180, 216", "I": "57, 255, 20", "O": "199, 125, 255", "U": "255, 209, 102"}
        v_col = vocal_colores.get(p_act, "#00FFCC")
        v_rgb = vocal_rgb.get(p_act, "0, 255, 204")
        sig_col = vocal_colores.get(p_sig, "#FFFF00")

        es_continua = (self.cmb_mode.currentIndex() == 0)
        seq_display = self.practice_sequence if es_continua else self.base_words
        active_seq_idx = max(0, pulso_actual - 1) if es_continua else self.current_word_idx
        tira_html = self._construir_tira_secuencia_html(seq_display, active_seq_idx)

        if flash:
            self.lbl_visor_vocal.setStyleSheet(
                f"background-color: rgba({v_rgb}, 0.38); color: #FFFFFF; border: 4px solid #FFFFFF; border-radius: 8px; padding: 8px;"
            )
            badge_top = (
                f"<span style='background-color:#FFFFFF; color:#050508; font-size:16px; font-weight:900; "
                f"padding:2px 12px; border-radius:3px;'>&#9733; ¡PRONUNCIA AHORA! &mdash; PULSO {pulso_actual}/{total_pulsos} &#9733;</span>"
            )
            letra_style = "font-size: 88px; font-weight: 900; color: #FFFFFF;"
        else:
            b_col = "#FFFF00" if pre_cambio else v_col
            self.lbl_visor_vocal.setStyleSheet(
                f"background-color: #050510; color: {v_col}; border: 2px solid {b_col}; border-radius: 8px; padding: 8px;"
            )
            if pre_cambio and p_sig and p_sig != "FIN":
                badge_top = (
                    f"<span style='color:#FFFF00; font-size:15px; font-weight:bold;'>"
                    f"ATENCIÓN: PRÓXIMO PULSO CAMBIA A /{p_sig}/ &mdash; PULSO {pulso_actual}/{total_pulsos}</span>"
                )
            else:
                badge_top = (
                    f"<span style='color:#AAAAAA; font-size:15px; font-weight:bold;'>"
                    f"MANTÉN LA VOCAL &mdash; PULSO {pulso_actual}/{total_pulsos} (CICLO {ciclo_actual}/{total_ciclos})</span>"
                )
            letra_style = f"font-size: 78px; font-weight: 900; color: {v_col};"

        sig_html = (
            f"&nbsp;&nbsp;|&nbsp;&nbsp;<span style='color:#CCCCCC; font-size:15px;'>SIGUIENTE: "
            f"<b style='color:{sig_col}; font-size:20px;'>/{p_sig}/</b></span>"
            if p_sig
            else ""
        )
        self.lbl_visor_vocal.setText(
            f"<div align='center' style='line-height:1.1;'>"
            f"{badge_top}<br><span style='{letra_style}'>/{p_act}/</span><br>"
            f"<span>{tira_html}{sig_html}</span></div>"
        )

    def _disparar_destello_vocal(
        self,
        palabra_actual: str,
        pulso_actual: int,
        total_pulsos: int,
        palabra_siguiente: str = "",
        ciclo_actual: int = 1,
        total_ciclos: int = 1,
    ) -> None:
        self._ultimo_estado_visor = (
            palabra_actual,
            pulso_actual,
            total_pulsos,
            palabra_siguiente,
            ciclo_actual,
            total_ciclos,
        )
        self._renderizar_visor_vocal(
            palabra_actual,
            pulso_actual,
            total_pulsos,
            palabra_siguiente=palabra_siguiente,
            ciclo_actual=ciclo_actual,
            total_ciclos=total_ciclos,
            flash=True,
            pre_cambio=False,
        )

        def _apagar():
            if not self.is_running or self.current_phase != "FONACION":
                return
            st = self._ultimo_estado_visor
            if st and st[1] == pulso_actual:
                self._renderizar_visor_vocal(
                    st[0], st[1], st[2],
                    palabra_siguiente=st[3], ciclo_actual=st[4], total_ciclos=st[5],
                    flash=False, pre_cambio=False,
                )

        QTimer.singleShot(200, _apagar)

    def start_training(self) -> None:
        self.base_words = self.get_words()
        reps = self.spin_reps.value()
        es_continua = (self.cmb_mode.currentIndex() == 0)

        if self.chk_debruijn.isChecked():
            if es_continua:
                self.practice_sequence = generar_secuencia_balanceada_debruijn(self.base_words, num_ciclos=reps)
            else:
                self.practice_sequence = generar_secuencia_balanceada_debruijn(self.base_words, num_ciclos=1)
        else:
            if es_continua:
                self.practice_sequence = self.base_words * reps
            else:
                self.practice_sequence = list(self.base_words)

        self.is_running = True
        self.current_word_idx = 0
        self.invalid_pulses_marked = []
        self._last_spoken_word = None
        self._pre_announced_beat = -1
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_mark_invalid.setEnabled(True)

        self._iniciar_fase_ruido()

    def _iniciar_fase_ruido(self) -> None:
        if not self.is_running:
            return
        self.current_phase = "RUIDO"
        self.metro_timer.stop()
        noise_s = self.spin_noise_s.value()
        primera_v = self.practice_sequence[self.current_word_idx] if self.practice_sequence else "A"
        self.lbl_status.setText(f"GRABANDO RUIDO BASAL ({noise_s}s) — HAZ SILENCIO")
        self.lbl_visor_vocal.setStyleSheet(
            "background-color: #141405; color: #FFFF00; border: 2px solid #FFFF00; border-radius: 8px; padding: 10px;"
        )
        self.lbl_visor_vocal.setText(
            f"<div align='center' style='line-height:1.2;'>"
            f"<span style='font-size:32px; font-weight:900; color:#FFFF00;'>HAZ SILENCIO &mdash; RUIDO BASAL ({noise_s}s)</span><br>"
            f"<span style='font-size:18px; color:#00FFCC;'>Relaja el rostro. Al terminar comenzará con <b>/{primera_v}/</b></span>"
            f"</div>"
        )
        self.voice_player.say("haz_silencio", "Haz silencio", fase="ruido_basal")
        try:
            self.phase_timer.timeout.disconnect()
        except Exception:
            pass
        self.phase_timer.timeout.connect(self._iniciar_count_in)
        self.phase_timer.start(int(noise_s * 1000))

    def _iniciar_count_in(self) -> None:
        if not self.is_running:
            return
        self.current_phase = "COUNT_IN"
        es_continua = (self.cmb_mode.currentIndex() == 0)
        palabra_init = (
            self.practice_sequence[0]
            if es_continua
            else self.practice_sequence[self.current_word_idx]
        )
        self.lbl_status.setText(f"CUENTA REGRESIVA DEL METRÓNOMO — PREPÁRATE PARA /{palabra_init}/")
        # En modo Discreto o al inicio de Continua, anunciar una sola vez en preparación
        self.voice_player.say(f"di_{palabra_init.lower()}", f"Prepárate, di {palabra_init}", fase="preparacion")
        self._last_spoken_word = palabra_init if not es_continua else None
        self._pre_announced_beat = 1 if es_continua else -1

        bpm = max(1, self.spin_bpm.value())
        total_beats = len(self.practice_sequence) if es_continua else self.spin_reps.value()
        if self.chk_jitter.isChecked():
            self.jitter_schedule = generate_jittered_intervals(
                base_bpm=bpm,
                num_beats=total_beats + 4,
                jitter_pct=self.spin_jitter_pct.value(),
                mode=self.cmb_jitter_mode.currentText(),
            )
        else:
            self.jitter_schedule = []

        self.metro_count_in_remaining = 4
        self.metro_beat_count = 0
        self.metro_subdiv_index = 0
        self.current_beat_tau_s = 60.0 / float(bpm)
        self._metro_next_perf = time.perf_counter()
        self.on_metro_beat()

    def _programar_siguiente_pulso(self, intervalo_s: float) -> None:
        ahora = time.perf_counter()
        intervalo = max(0.005, float(intervalo_s))
        siguiente = self._metro_next_perf if self._metro_next_perf is not None else ahora
        siguiente += intervalo
        restante = siguiente - ahora
        if restante < 0:
            siguiente = ahora
            restante = 0.0
        self._metro_next_perf = siguiente
        self.metro_timer.start(max(0, int(round(restante * 1000.0))))

    def on_metro_beat(self) -> None:
        if not self.is_running:
            return
        bpm = max(1, self.spin_bpm.value())
        es_continua = (self.cmb_mode.currentIndex() == 0)
        jitter_active = self.chk_jitter.isChecked() and self.metro_count_in_remaining <= 1
        num_subdivs = 1 if jitter_active else max(1, self.spin_subdivs.value())
        is_main_beat = (self.metro_subdiv_index == 0)

        if jitter_active and self.jitter_schedule and is_main_beat:
            idx_j = max(0, self.metro_beat_count) % len(self.jitter_schedule)
            self.current_beat_tau_s = float(self.jitter_schedule[idx_j])
        elif not jitter_active:
            self.current_beat_tau_s = 60.0 / float(bpm)

        sub_interval_s = self.current_beat_tau_s / float(num_subdivs)
        self._programar_siguiente_pulso(sub_interval_s)

        if is_main_beat:
            if self.metro_count_in_remaining > 1:
                cnt_val = self.metro_count_in_remaining - 1
                primera_v = (
                    self.practice_sequence[0]
                    if es_continua
                    else self.practice_sequence[self.current_word_idx]
                )
                self.metro_lbl_title.setText("PREPÁRATE")
                self.metro_lbl_title.setStyleSheet("font-size: 16px; color: #FF4444;")
                self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #FF4444;")
                self.metro_lbl_count.setText(str(cnt_val))
                self.metro_pulse_frame.setStyleSheet("background-color: #FF0000; border: 2px solid #FFFFFF;")
                self.lbl_visor_vocal.setStyleSheet(
                    "background-color: #180808; color: #FFCC00; border: 2px solid #FF4444; border-radius: 8px; padding: 10px;"
                )
                self.lbl_visor_vocal.setText(
                    f"<div align='center' style='line-height:1.15;'>"
                    f"<span style='color:#FF8800; font-size:20px;'>CUENTA REGRESIVA DEL METRÓNOMO: {cnt_val}</span><br>"
                    f"<span style='color:#FFFFFF; font-size:56px; font-weight:900;'>PREPÁRATE PARA: /{primera_v}/</span>"
                    f"</div>"
                )
                self.metro_count_in_remaining -= 1
                self.beep_queue.put((800, 180))
            else:
                if self.metro_count_in_remaining == 1:
                    self.metro_beat_count = 1
                    self.metro_count_in_remaining = 0
                    self.current_phase = "FONACION"
                    self.beep_queue.put((1200, 350))
                else:
                    self.metro_beat_count += 1
                    total_target = len(self.practice_sequence) if es_continua else self.spin_reps.value()
                    if self.metro_beat_count > total_target:
                        self.metro_timer.stop()
                        if es_continua:
                            self._finalizar_entrenamiento()
                        else:
                            self._iniciar_descanso_discreto()
                        return
                    self.beep_queue.put((1000, 100))

                if es_continua:
                    total_pulsos = len(self.practice_sequence)
                    idx_p = (self.metro_beat_count - 1) % total_pulsos
                    palabra_actual = self.practice_sequence[idx_p]
                    palabra_sig = (
                        self.practice_sequence[idx_p + 1]
                        if (idx_p + 1) < total_pulsos
                        else "FIN"
                    )
                    n_base = max(1, len(self.base_words))
                    ciclo_act = ((self.metro_beat_count - 1) // n_base) + 1
                    total_ciclos = self.spin_reps.value()
                else:
                    total_pulsos = self.spin_reps.value()
                    palabra_actual = self.practice_sequence[self.current_word_idx]
                    palabra_sig = (
                        palabra_actual
                        if self.metro_beat_count < total_pulsos
                        else (
                            self.practice_sequence[self.current_word_idx + 1]
                            if (self.current_word_idx + 1) < len(self.practice_sequence)
                            else "FIN"
                        )
                    )
                    ciclo_act = self.current_word_idx + 1
                    total_ciclos = len(self.practice_sequence)

                self.metro_lbl_title.setText(f"VOCAL /{palabra_actual}/ (P{self.metro_beat_count})")
                self.metro_lbl_title.setStyleSheet("font-size: 15px; color: #00FFCC;")
                self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #00FFFF;")
                self.metro_lbl_count.setText("1" if num_subdivs > 1 else str(self.metro_beat_count))
                self.metro_pulse_frame.setStyleSheet("background-color: #00FFFF; border: 2px solid #FFFFFF;")
                self.lbl_status.setText(
                    f"FONACIÓN EN VIVO: /{palabra_actual}/ (Pulso {self.metro_beat_count}/{total_pulsos})"
                )

                self._disparar_destello_vocal(
                    palabra_actual=palabra_actual,
                    pulso_actual=self.metro_beat_count,
                    total_pulsos=total_pulsos,
                    palabra_siguiente=palabra_sig,
                    ciclo_actual=ciclo_act,
                    total_ciclos=total_ciclos,
                )

                # Si estamos en Secuencia Continua con 1 sola subdivisión (p. ej. Jitter activo)
                # y el pulso no fue pre-anunciado en un sub-pulso previo, decir "Di <vocal>" sólo si cambió la vocal
                if (
                    es_continua
                    and self._pre_announced_beat != self.metro_beat_count
                    and palabra_actual != self._last_spoken_word
                ):
                    self._last_spoken_word = palabra_actual
                    self.voice_player.say(f"di_{palabra_actual.lower()}", f"Di {palabra_actual}", fase="pulso_continua")

            QTimer.singleShot(70, self._reset_metro_frame)
        else:
            sub_num = self.metro_subdiv_index + 1
            self.beep_queue.put((1600, 40))
            if self.metro_count_in_remaining > 0:
                self.metro_pulse_frame.setStyleSheet("background-color: #660000; border: 2px solid #FF0000;")
            else:
                self.metro_pulse_frame.setStyleSheet("background-color: #00AAAA; border: 2px solid #00FFFF;")
                self.metro_lbl_count.setStyleSheet("font-size: 48px; color: #00AAAA;")
                self.metro_lbl_count.setText(str(sub_num))
                # En el ÚLTIMO sub-pulso de Secuencia Continua, pre-avisar visualmente Y adelantar el "Di <Vocal>"
                # para no pisar la voz del sujeto cuando caiga el pulso 1
                if sub_num == num_subdivs and es_continua and self.current_phase == "FONACION":
                    st = self._ultimo_estado_visor
                    if st:
                        self._renderizar_visor_vocal(
                            st[0], st[1], st[2],
                            palabra_siguiente=st[3], ciclo_actual=st[4], total_ciclos=st[5],
                            flash=False, pre_cambio=True,
                        )
                        next_word = st[3]
                        next_beat = self.metro_beat_count + 1
                        if next_word and next_word != "FIN" and next_word != self._last_spoken_word:
                            self._last_spoken_word = next_word
                            self._pre_announced_beat = next_beat
                            self.voice_player.say(
                                f"di_{next_word.lower()}", f"Di {next_word}", fase="pulso_continua"
                            )
            QTimer.singleShot(70, self._reset_metro_frame)

        self.metro_subdiv_index = (self.metro_subdiv_index + 1) % num_subdivs

    def _reset_metro_frame(self) -> None:
        self.metro_pulse_frame.setStyleSheet("background-color: #111111; border: 2px solid #00FFFF; border-radius: 6px;")

    def _iniciar_descanso_discreto(self) -> None:
        if not self.is_running:
            return
        self.current_word_idx += 1
        if self.current_word_idx >= len(self.practice_sequence):
            self._finalizar_entrenamiento()
            return
        self.current_phase = "DESCANSO"
        rest_s = self.spin_rest_s.value()
        sig_w = self.practice_sequence[self.current_word_idx]
        self.lbl_status.setText(f"DESCANSO ({rest_s}s) — Siguiente vocal: /{sig_w}/")
        self.lbl_visor_vocal.setStyleSheet(
            "background-color: #0b1320; color: #00FFFF; border: 2px solid #00FFFF; border-radius: 8px; padding: 10px;"
        )
        self.lbl_visor_vocal.setText(
            f"<div align='center' style='line-height:1.2;'>"
            f"<span style='font-size:42px; color:#888888; font-weight:900;'>DESCANSO ({rest_s}s)</span><br>"
            f"<span style='font-size:22px; color:#00FFCC;'>Siguiente toma ({self.current_word_idx + 1}/{len(self.practice_sequence)}): "
            f"<b style='color:#FFFF00;'>/{sig_w}/</b></span>"
            f"</div>"
        )
        self.voice_player.say("descanso", "Descanso", fase="descanso")
        try:
            self.phase_timer.timeout.disconnect()
        except Exception:
            pass
        self.phase_timer.timeout.connect(self._iniciar_count_in)
        self.phase_timer.start(int(rest_s * 1000))

    def _finalizar_entrenamiento(self) -> None:
        self.is_running = False
        self.current_phase = "IDLE"
        self.metro_timer.stop()
        self.phase_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_mark_invalid.setEnabled(False)
        inv_txt = (
            f"Pulsos marcados como inválidos: {self.invalid_pulses_marked}"
            if self.invalid_pulses_marked
            else "Todos los pulsos fueron válidos (0 errores marcados)."
        )
        self.lbl_status.setText("¡PRÁCTICA COMPLETADA CON ÉXITO!")
        self.lbl_visor_vocal.setStyleSheet(
            "background-color: #062216; color: #39FF14; border: 2px solid #39FF14; border-radius: 8px; padding: 10px;"
        )
        self.lbl_visor_vocal.setText(
            f"<div align='center' style='line-height:1.25;'>"
            f"<span style='font-size:36px; font-weight:900; color:#39FF14;'>¡ENTRENAMIENTO COMPLETADO!</span><br>"
            f"<span style='font-size:16px; color:#FFFFFF;'>Secuencia practicada ({len(self.practice_sequence)} ítems): "
            f"{' &rarr; '.join(self.practice_sequence[:15])}{'...' if len(self.practice_sequence) > 15 else ''}</span><br>"
            f"<span style='font-size:14px; color:#FFCC00;'>{inv_txt}</span>"
            f"</div>"
        )

    def stop_training(self) -> None:
        self.is_running = False
        self.current_phase = "IDLE"
        self.metro_timer.stop()
        self.phase_timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_mark_invalid.setEnabled(False)
        self.lbl_status.setText("Entrenamiento detenido por el usuario.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_training()
        event.accept()


def main():
    app = QApplication.instance()
    is_standalone = False
    if not app:
        app = QApplication(sys.argv)
        is_standalone = True
    window = ModuloEntrenamiento()
    window.show()
    if is_standalone:
        sys.exit(app.exec())
    return window


if __name__ == "__main__":
    main()