# ==============================================================================
# Archivo: EMG_desarrollo/acquisition/modulo_de_entrenamiento.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de entrenamiento para practicar la cadencia de autograbado
#              con soporte de Pausa y Reanudación exacta.
# ==============================================================================

import sys
import os
import random
import subprocess
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QFormLayout,
    QSpinBox, QPushButton, QLabel, QHBoxLayout
)
from PySide6.QtCore import QTimer, Qt


class ModuloEntrenamiento(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ñandú LSD - Entrenamiento AutoForge")
        self.resize(480, 380)
        self.setStyleSheet("""
            QWidget { background-color: #0B0C10; color: #00FFCC; font-family: 'Courier New', monospace; font-weight: bold; }
            QSpinBox { background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; padding: 4px; border-radius: 3px; }
            QPushButton { background-color: #1F2833; border: 2px solid #00FFCC; color: #00FFCC; padding: 10px; border-radius: 4px; }
            QPushButton:hover { background-color: #00FFCC; color: #0B0C10; }
            QPushButton:disabled { border-color: #444444; color: #555555; background-color: #111111; }
        """)

        layout = QVBoxLayout(self)

        lbl_title = QLabel("ENTRENAMIENTO AUTOGRABADO (SIMULADOR)")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16px; color: #66FCF1; margin-bottom: 10px;")
        layout.addWidget(lbl_title)

        form = QFormLayout()

        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(30, 200)
        self.spin_bpm.setValue(60)

        self.spin_beats_word = QSpinBox()
        self.spin_beats_word.setRange(1, 100)
        self.spin_beats_word.setValue(5)

        self.spin_rest_s = QSpinBox()
        self.spin_rest_s.setRange(1, 60)
        self.spin_rest_s.setValue(5)

        self.spin_num_words = QSpinBox()
        self.spin_num_words.setRange(1, 500)
        self.spin_num_words.setValue(10)

        form.addRow("BPM (Metrónomo):", self.spin_bpm)
        form.addRow("Duración de palabra (Beats):", self.spin_beats_word)
        form.addRow("Descanso entre palabras (s):", self.spin_rest_s)
        form.addRow("Total de palabras a practicar:", self.spin_num_words)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("INICIAR")
        self.btn_start.clicked.connect(self.start_training)

        self.btn_pause = QPushButton("PAUSAR")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setStyleSheet("""
            QPushButton { border-color: #FFCC00; color: #FFCC00; }
            QPushButton:hover { background-color: #332800; color: #FFCC00; }
            QPushButton:checked { background-color: #FFCC00; color: #000000; border-color: #FFFFFF; }
            QPushButton:disabled { border-color: #444444; color: #555555; }
        """)
        self.btn_pause.toggled.connect(self.toggle_pause_training)

        self.btn_stop = QPushButton("DETENER")
        self.btn_stop.setStyleSheet("""
            QPushButton { border-color: #FF0055; color: #FF0055; }
            QPushButton:hover { background-color: #FF0055; color: #FFFFFF; }
            QPushButton:disabled { border-color: #444444; color: #555555; }
        """)
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.lbl_status = QLabel("Esperando para iniciar...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px;")
        layout.addWidget(self.lbl_status)

        self.word_process = None
        self.metronome_process = None
        self.practice_list = []
        self.current_idx = 0

        self.is_paused = False
        self._paused_timer = None
        self._paused_remaining_ms = 0

        self.timer_prepare = QTimer(self)
        self.timer_prepare.setSingleShot(True)
        self.timer_prepare.timeout.connect(self.state_palabra)

        self.timer_word = QTimer(self)
        self.timer_word.setSingleShot(True)
        self.timer_word.timeout.connect(self.state_descanso)

        self.timer_rest = QTimer(self)
        self.timer_rest.setSingleShot(True)
        self.timer_rest.timeout.connect(self.next_word)

    def get_words(self):
        candidates = []
        if getattr(sys, "frozen", False):
            base_path = os.path.dirname(os.path.abspath(sys.executable))
            if os.path.basename(base_path) == "_internal":
                base_path = os.path.dirname(base_path)
            candidates.append(os.path.join(base_path, "palabras.txt"))
            candidates.append(os.path.join(base_path, "acquisition", "palabras.txt"))
        else:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates.append(os.path.join(root_dir, "palabras.txt"))
            candidates.append(os.path.join(root_dir, "acquisition", "palabras.txt"))

        for path in candidates:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        words = [l.strip() for l in f if l.strip()]
                    if words:
                        return words
                except Exception:
                    pass
        return ["A", "E", "I", "O", "U"]

    def start_training(self):
        words = self.get_words()
        num = self.spin_num_words.value()
        self.practice_list = []
        while len(self.practice_list) < num:
            shuffled = words.copy()
            random.shuffle(shuffled)
            self.practice_list.extend(shuffled)
        self.practice_list = self.practice_list[:num]

        self.current_idx = 0
        self.is_paused = False
        self.btn_pause.blockSignals(True)
        self.btn_pause.setChecked(False)
        self.btn_pause.setText("PAUSAR")
        self.btn_pause.blockSignals(False)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.next_word()

    def toggle_pause_training(self, paused: bool):
        self.is_paused = paused
        if paused:
            self.btn_pause.setText("REANUDAR")
            for t in (self.timer_prepare, self.timer_word, self.timer_rest):
                if t.isActive():
                    self._paused_timer = t
                    self._paused_remaining_ms = max(50, t.remainingTime())
                    t.stop()
                    break
            if self.metronome_process and self.metronome_process.poll() is None:
                try:
                    self.metronome_process.stdin.write("MUTE\n")
                    self.metronome_process.stdin.flush()
                except Exception:
                    pass
            self.launch_word_window("PAUSA")
            self.lbl_status.setText("Entrenamiento en PAUSA (Presiona REANUDAR)")
            self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #FFCC00;")
        else:
            self.btn_pause.setText("PAUSAR")
            if self.metronome_process and self.metronome_process.poll() is None:
                try:
                    self.metronome_process.stdin.write("UNMUTE\n")
                    self.metronome_process.stdin.flush()
                except Exception:
                    pass
            palabra = self.practice_list[self.current_idx] if self.current_idx < len(self.practice_list) else ""
            if self._paused_timer == self.timer_prepare:
                self.launch_word_window(f"PREPÁRATE\n{palabra}")
                self.lbl_status.setText(f"Preparando: {palabra} ({self.current_idx+1}/{len(self.practice_list)})")
            elif self._paused_timer == self.timer_word:
                self.launch_word_window(palabra)
                self.lbl_status.setText(f"Hablando: {palabra} ({self.current_idx+1}/{len(self.practice_list)})")
            elif self._paused_timer == self.timer_rest:
                self.launch_word_window("DESCANSO")
                self.lbl_status.setText("Descanso...")

            if self._paused_timer is not None and self._paused_remaining_ms > 0:
                self._paused_timer.start(self._paused_remaining_ms)
                self._paused_timer = None
                self._paused_remaining_ms = 0

    def stop_training(self):
        self.timer_prepare.stop()
        self.timer_word.stop()
        self.timer_rest.stop()
        self.is_paused = False
        self._paused_timer = None
        self._paused_remaining_ms = 0
        self.kill_processes()
        self.lbl_status.setText("Entrenamiento detenido.")
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #00FFCC;")
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.blockSignals(True)
        self.btn_pause.setChecked(False)
        self.btn_pause.setText("PAUSAR")
        self.btn_pause.blockSignals(False)
        self.btn_stop.setEnabled(False)

    def kill_processes(self):
        for proc_attr in ("word_process", "metronome_process"):
            proc = getattr(self, proc_attr, None)
            if proc:
                try:
                    proc.kill()
                except Exception:
                    pass
                setattr(self, proc_attr, None)

    def launch_word_window(self, text):
        if self.word_process:
            try:
                self.word_process.kill()
            except Exception:
                pass
        if getattr(sys, "frozen", False):
            cmd = [sys.executable, "acquisition/ventana_palabras.py", f"--word={text}"]
        else:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ventana_palabras.py")
            cmd = [sys.executable, script_path, f"--word={text}"]
        self.word_process = subprocess.Popen(cmd)

    def launch_metronome(self, count_in=True):
        if self.metronome_process:
            try:
                self.metronome_process.kill()
            except Exception:
                pass
        bpm = self.spin_bpm.value()
        if getattr(sys, "frozen", False):
            args = [sys.executable, "acquisition/metronomo_visual.py", "--autostart", "--count", f"--bpm={bpm}"]
        else:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metronomo_visual.py")
            args = [sys.executable, script_path, "--autostart", "--count", f"--bpm={bpm}"]
        if count_in:
            args.append("--count-in=4")
        self.metronome_process = subprocess.Popen(args, stdin=subprocess.PIPE, text=True)

    def next_word(self):
        if self.current_idx >= len(self.practice_list):
            self.lbl_status.setText("¡Práctica completada con éxito!")
            self.launch_word_window("¡COMPLETADO!")
            self.stop_training()
            QTimer.singleShot(3000, self.kill_processes)
            return

        palabra = self.practice_list[self.current_idx]
        self.lbl_status.setText(f"Preparando: {palabra} ({self.current_idx+1}/{len(self.practice_list)})")
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #00FFCC;")
        self.launch_word_window(f"PREPÁRATE\n{palabra}")
        self.launch_metronome(count_in=True)

        ms_espera = int(3 * 60000 / self.spin_bpm.value())
        self.timer_prepare.start(ms_espera)

    def state_palabra(self):
        palabra = self.practice_list[self.current_idx]
        self.lbl_status.setText(f"Hablando: {palabra} ({self.current_idx+1}/{len(self.practice_list)})")
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #FF00FF;")
        self.launch_word_window(palabra)

        ms_duracion = int(self.spin_beats_word.value() * 60000 / self.spin_bpm.value())
        self.timer_word.start(ms_duracion)

    def state_descanso(self):
        self.lbl_status.setText(f"Descanso de {self.spin_rest_s.value()} segundos...")
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #66FCF1;")
        self.launch_word_window("DESCANSO")

        if self.metronome_process:
            try:
                self.metronome_process.kill()
            except Exception:
                pass
            self.metronome_process = None

        self.current_idx += 1
        ms_descanso = int(self.spin_rest_s.value() * 1000)
        self.timer_rest.start(ms_descanso)

    def closeEvent(self, event):
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