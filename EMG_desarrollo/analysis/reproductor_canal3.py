# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo reproductor_canal3.py del sistema NANDU LSD.
# ==============================================================================

import sys
import os
import glob
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QSlider, QLabel, QListWidget, 
                               QFileDialog)
from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

class AudioPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mini-DAW: Reproductor Audios (Canal 3 / Micrófono)")
        self.setGeometry(200, 200, 700, 500)
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        self.init_ui()
        
        self.player.positionChanged.connect(self.update_slider)
        self.player.durationChanged.connect(self.update_duration)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Estilos generales para un look "mini-DAW"
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0d0d1a;
            }
            QWidget {
                color: #00ffff;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QPushButton {
                background-color: #1a001a;
                color: #ff00ff;
                border: 2px solid #ff00ff;
                border-right: 4px solid #00ffff;
                border-bottom: 4px solid #00ffff;
                padding: 8px;
                border-radius: 2px;
                font-weight: 900;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ff00ff;
                color: #1a001a;
                border: 2px solid #00ffff;
                border-right: 4px solid #00ffff;
                border-bottom: 4px solid #00ffff;
            }
            QPushButton:pressed {
                background-color: #00ffff;
                color: #000000;
                border: 2px solid #ffffff;
            }
            QListWidget {
                background-color: #05050d;
                border: 2px solid #00ffff;
                color: #00ffcc;
                padding: 5px;
                font-size: 13px;
                selection-background-color: #ff0055;
                selection-color: #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ffff;
                height: 8px;
                background: #0d0d1a;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #ff00ff;
                border: 2px solid #ffffff;
                width: 14px;
                margin: -4px 0;
            }
        """)
        
        # --- Controles Superiores: Selección de Carpeta ---
        top_layout = QHBoxLayout()
        self.btn_open = QPushButton("📁 Abrir Base de Datos")
        self.btn_open.clicked.connect(self.open_folder)
        top_layout.addWidget(self.btn_open)
        
        self.lbl_folder = QLabel("Ninguna carpeta seleccionada")
        self.lbl_folder.setStyleSheet("color: #ff0055; font-weight: bold;")
        top_layout.addWidget(self.lbl_folder, stretch=1)
        layout.addLayout(top_layout)
        
        # --- Lista de Archivos de Audio ---
        self.file_list = QListWidget()
        self.file_list.itemDoubleClicked.connect(self.play_selected)
        layout.addWidget(QLabel("🎙️ Archivos de audio disponibles:"))
        layout.addWidget(self.file_list, stretch=1)
        
        # --- Línea de Tiempo (Timeline) ---
        time_layout = QHBoxLayout()
        self.lbl_current_time = QLabel("00:00")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self.set_position)
        self.lbl_total_time = QLabel("00:00")
        
        time_layout.addWidget(self.lbl_current_time)
        time_layout.addWidget(self.slider, stretch=1)
        time_layout.addWidget(self.lbl_total_time)
        layout.addLayout(time_layout)
        
        # --- Controles de Reproducción ---
        ctrl_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setMinimumWidth(100)
        self.btn_play.clicked.connect(self.play_audio)
        
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setMinimumWidth(100)
        self.btn_pause.clicked.connect(self.pause_audio)
        
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.btn_play)
        ctrl_layout.addWidget(self.btn_pause)
        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)
        
    def open_folder(self):
        root_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_dir = os.path.join(root_project_dir, "base_de_datos_electrodos")
        if not os.path.exists(default_dir):
            default_dir = root_project_dir
            
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar Carpeta", default_dir)
        if folder:
            self.lbl_folder.setText(folder)
            self.load_files(folder)
            
    def load_files(self, folder):
        self.file_list.clear()
        search_pattern = os.path.join(folder, "**", "*.wav")
        wav_files = glob.glob(search_pattern, recursive=True)
        
        # Priorizar audios del canal 3 o micrófono
        canal3_files = [f for f in wav_files if "canal_3" in f.lower() or "mic" in f.lower()]
        files_to_list = canal3_files if canal3_files else wav_files
        
        if not files_to_list:
            self.file_list.addItem("No se encontraron archivos .wav en esta carpeta.")
            return

        for f in sorted(files_to_list):
            self.file_list.addItem(f)
            
    def play_selected(self, item):
        file_path = item.text()
        if not os.path.exists(file_path):
            return
            
        import tempfile
        import scipy.io.wavfile as wav
        import scipy.signal as signal
        import numpy as np

        try:
            # Leer el archivo WAV original
            sr, data = wav.read(file_path)
            
            # Windows/PySide6 suele fallar con tasas de muestreo bajas (ej. 2000 Hz o 1250 Hz)
            # Solución: Remuestrear a 44100 Hz estándar temporalmente.
            target_sr = 44100
            if sr != target_sr and sr > 0:
                num_samples = int(len(data) * target_sr / sr)
                resampled_data = signal.resample(data, num_samples, axis=0)
                
                # Normalizar para que se escuche bien (micrófonos pueden ser bajos)
                max_val = np.max(np.abs(resampled_data))
                if max_val > 0:
                    resampled_data = np.int16(resampled_data / max_val * 32767)
                else:
                    resampled_data = np.int16(resampled_data)
                    
                # Guardar en archivo temporal
                temp_dir = tempfile.gettempdir()
                temp_wav = os.path.join(temp_dir, "temp_nandu_playback.wav")
                wav.write(temp_wav, target_sr, resampled_data)
                
                play_path = temp_wav
            else:
                play_path = file_path
                
            self.player.setSource(QUrl.fromLocalFile(play_path))
            self.player.play()
        except Exception as e:
            print(f"Error procesando audio para reproducción: {e}")
        
    def play_audio(self):
        if self.player.source().isEmpty() and self.file_list.currentItem():
            self.play_selected(self.file_list.currentItem())
        else:
            self.player.play()
            
    def pause_audio(self):
        self.player.pause()
        
    def set_position(self, position):
        self.player.setPosition(position)
        
    def update_slider(self, position):
        self.slider.setValue(position)
        self.lbl_current_time.setText(self.format_time(position))
        
    def update_duration(self, duration):
        self.slider.setRange(0, duration)
        self.lbl_total_time.setText(self.format_time(duration))
        
    def format_time(self, ms):
        s = ms // 1000
        m = s // 60
        s = s % 60
        return f"{m:02d}:{s:02d}"

def main():
    app = QApplication(sys.argv)
    window = AudioPlayer()
    
    # --- Ruteo Automático ---
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
        if os.path.basename(base_dir) == "_internal":
            base_dir = os.path.dirname(base_dir)
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    db_path = os.path.join(base_dir, "base_de_datos_electrodos")

    # Si viene con argumentos desde el Gestor de Sesiones de main_app.py
    if len(sys.argv) > 1:
        ruta_inicial = sys.argv[1]
        if os.path.exists(ruta_inicial):
            window.lbl_folder.setText(ruta_inicial)
            window.load_files(ruta_inicial)
    # Si no hay argumentos, cargar la base de datos entera por defecto
    elif os.path.exists(db_path):
        window.lbl_folder.setText(db_path)
        window.load_files(db_path)
            
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
