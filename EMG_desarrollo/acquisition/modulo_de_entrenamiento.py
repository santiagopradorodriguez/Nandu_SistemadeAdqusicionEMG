# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo modulo_de_entrenamiento.py del sistema NANDU LSD.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de entrenamiento para practicar la cadencia de autograbado.
# ==============================================================================

import sys
import os
import random
import subprocess
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFormLayout, 
                               QSpinBox, QPushButton, QLabel, QHBoxLayout)
from PySide6.QtCore import QTimer, Qt

class ModuloEntrenamiento(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ñandú LSD - Entrenamiento AutoForge")
        self.resize(450, 350)
        self.setStyleSheet("""
            QWidget { background-color: #050505; color: #00FF00; font-family: 'Courier New', monospace; font-weight: bold; }
            QSpinBox { background-color: #111111; color: #00FFFF; border: 1px solid #00FF00; padding: 4px; }
            QPushButton { background-color: #111111; border: 2px solid #00FF00; padding: 10px; }
            QPushButton:hover { background-color: #00FF00; color: #000000; }
            QPushButton:disabled { border-color: #555555; color: #555555; }
        """)

        layout = QVBoxLayout(self)
        
        lbl_title = QLabel("ENTRENAMIENTO AUTOGRABADO (SIMULADOR)")
        lbl_title.setAlignment(Qt.AlignCenter)
        lbl_title.setStyleSheet("font-size: 16px; color: #00FFFF; margin-bottom: 10px;")
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
        self.btn_start = QPushButton("INICIAR PRÁCTICA")
        self.btn_start.clicked.connect(self.start_training)
        
        self.btn_stop = QPushButton("DETENER")
        self.btn_stop.setStyleSheet("QPushButton { border-color: #FF0000; color: #FF0000; } QPushButton:hover { background-color: #FF0000; color: #000000; }")
        self.btn_stop.clicked.connect(self.stop_training)
        self.btn_stop.setEnabled(False)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)
        
        self.lbl_status = QLabel("Esperando para iniciar...")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px;")
        layout.addWidget(self.lbl_status)
        
        # Control de procesos hijos
        self.word_process = None
        self.metronome_process = None
        
        # Variables de estado
        self.practice_list = []
        self.current_idx = 0

        # Temporizadores para las fases
        self.timer_prepare = QTimer()
        self.timer_prepare.setSingleShot(True)
        self.timer_prepare.timeout.connect(self.state_palabra)

        self.timer_word = QTimer()
        self.timer_word.setSingleShot(True)
        self.timer_word.timeout.connect(self.state_descanso)

        self.timer_rest = QTimer()
        self.timer_rest.setSingleShot(True)
        self.timer_rest.timeout.connect(self.next_word)

    def get_words(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root_dir, "palabras.txt")
        if not os.path.exists(path):
            return ["A", "E", "I", "O", "U"]
        with open(path, 'r', encoding='utf-8') as f:
            words = [l.strip() for l in f if l.strip()]
        return words if words else ["A", "E", "I", "O", "U"]

    def start_training(self):
        words = self.get_words()
        num = self.spin_num_words.value()
        self.practice_list = []
        
        # Llenar la lista randomizando, ciclando si es necesario
        while len(self.practice_list) < num:
            random.shuffle(words)
            self.practice_list.extend(words)
        self.practice_list = self.practice_list[:num]
        
        self.current_idx = 0
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.next_word()

    def stop_training(self):
        self.timer_prepare.stop()
        self.timer_word.stop()
        self.timer_rest.stop()
        self.kill_processes()
        self.lbl_status.setText("Entrenamiento detenido.")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def kill_processes(self):
        if self.word_process:
            try: self.word_process.kill()
            except: pass
            self.word_process = None
        if self.metronome_process:
            try: self.metronome_process.kill()
            except: pass
            self.metronome_process = None

    def launch_word_window(self, text):
        if self.word_process:
            try: self.word_process.kill()
            except: pass
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ventana_palabras.py')
        self.word_process = subprocess.Popen([sys.executable, script_path, f'--word={text}'])
        
    def launch_metronome(self, count_in=True):
        if self.metronome_process:
            try: self.metronome_process.kill()
            except: pass
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
        bpm = self.spin_bpm.value()
        args = [sys.executable, script_path, '--autostart', '--count', f'--bpm={bpm}']
        if count_in:
            args.append('--count-in=4')
        self.metronome_process = subprocess.Popen(args, stdin=subprocess.PIPE, text=True)

    def next_word(self):
        if self.current_idx >= len(self.practice_list):
            self.lbl_status.setText("¡Práctica completada con éxito!")
            self.launch_word_window("¡COMPLETADO!")
            self.stop_training()
            # Limpiar mensaje después de 3 seg
            QTimer.singleShot(3000, lambda: self.kill_processes())
            return
            
        palabra = self.practice_list[self.current_idx]
        self.lbl_status.setText(f"Preparando: {palabra} ({self.current_idx+1}/{len(self.practice_list)})")
        
        self.launch_word_window(f"PREPÁRATE\n{palabra}")
        self.launch_metronome(count_in=True)
        
        # El count-in consta de 4 compases, pero la 4ta pulsación es el "¡GO!" simultáneo con la lectura
        # Esperamos exactamente 3 compases antes de saltar a la palabra
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
        self.lbl_status.setStyleSheet("margin-top: 15px; font-size: 14px; color: #00FFFF;")
        self.launch_word_window("DESCANSO")
        
        # Detenemos el metrónomo durante el descanso para relajar
        if self.metronome_process:
            try: self.metronome_process.kill()
            except: pass
            self.metronome_process = None
            
        self.current_idx += 1
        ms_descanso = int(self.spin_rest_s.value() * 1000)
        self.timer_rest.start(ms_descanso)

    def closeEvent(self, event):
        self.stop_training()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModuloEntrenamiento()
    window.show()
    sys.exit(app.exec())