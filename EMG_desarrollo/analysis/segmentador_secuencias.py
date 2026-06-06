import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import math

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import tempfile
import scipy.io.wavfile as wav
import scipy.signal as signal

# =============================================================================
# Funciones DSP copiadas de analisis_por_track_integrado.py
# =============================================================================
def _compute_env_full(signal_abs, apply_envelope, smooth_ms, samplerate, tipo_env="media_movil"):
    if tipo_env == "rms" and smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            sig_sq = signal_abs ** 2
            window = np.ones(win_len, dtype=float) / float(win_len)
            rms_val = np.sqrt(np.convolve(sig_sq, window, mode='same'))
            return rms_val
        else:
            return signal_abs.copy()

    if apply_envelope:
        try:
            from scipy.fft import next_fast_len
            from scipy.signal import hilbert
            N = len(signal_abs)
            fast_len = next_fast_len(N)
            env_full = np.abs(hilbert(signal_abs, N=fast_len)[:N])
        except Exception as e:
            env_full = signal_abs.copy()
    else:
        env_full = signal_abs.copy()

    if tipo_env == "media_movil" and smooth_ms is not None and smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * samplerate / 1000.0)))
        if win_len > 1:
            window = np.ones(win_len, dtype=float) / float(win_len)
            env_full = np.convolve(env_full, window, mode='same')
            
    return env_full

# =============================================================================
# Interfaz Gráfica
# =============================================================================
class SegmentadorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentador Interactivo de Secuencia Continua")
        self.resize(1200, 800)
        
        # Tema Oscuro (Cyberpunk)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0b0f19; color: #08F7FE; font-family: 'Consolas', monospace; }
            QListWidget { background-color: #12182b; border: 1px solid #08F7FE; outline: none; }
            QListWidget::item { padding: 5px; }
            QListWidget::item:selected { background-color: #FE53BB; color: white; }
            QPushButton { background-color: #08F7FE; color: #0b0f19; font-weight: bold; padding: 10px; border-radius: 5px; }
            QPushButton:hover { background-color: #FE53BB; color: white; }
            QLabel { font-size: 14px; }
            QComboBox { background-color: #12182b; border: 1px solid #08F7FE; color: white; padding: 5px; }
        """)
        
        self.base_dir = Path(__file__).resolve().parent.parent / "base_de_datos_electrodos"
        
        # Data
        self.current_folder = None
        self.df = None
        self.metadata = None
        self.samplerate = 2000.0
        self.bpm = 60.0
        self.noise_seconds = 5.0
        self.window_spans = [] # Para interactividad
        self.selected_start_window = None # Indice de la ventana inicial seleccionada
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        self.assigned_labels = {} # index -> letter
        self.text_items = {} # index -> pg.TextItem
        
        self.init_ui()
        self.load_folders()
        
    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        
        # Panel Izquierdo (Lista)
        left_panel = QtWidgets.QVBoxLayout()
        
        left_panel.addWidget(QtWidgets.QLabel("Carpetas de Secuencia Continua:"))
        self.list_folders = QtWidgets.QListWidget()
        self.list_folders.itemSelectionChanged.connect(self.on_folder_selected)
        left_panel.addWidget(self.list_folders)
        
        main_layout.addLayout(left_panel, 1)
        
        # Panel Derecho (Gráfico + Controles)
        right_panel = QtWidgets.QVBoxLayout()
        
        # Plot
        self.plot_widget = pg.PlotWidget(title="Envolvente de la Señal (Haz click para seleccionar inicio)")
        self.plot_widget.setBackground('#0b0f19')
        self.plot_widget.setLabel('bottom', 'Tiempo', units='s')
        self.plot_widget.setLabel('left', 'Amplitud', units='µV')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked)
        right_panel.addWidget(self.plot_widget, 3)
        
        # Controles
        controls_layout = QtWidgets.QGridLayout()
        
        self.lbl_info = QtWidgets.QLabel("Seleccione una carpeta para comenzar.")
        controls_layout.addWidget(self.lbl_info, 0, 0, 1, 2)
        
        self.lbl_selected_win = QtWidgets.QLabel("Selecciona una ventana en el gráfico para escuchar y etiquetar")
        self.lbl_selected_win.setStyleSheet("color: #FE53BB; font-weight: bold; font-size: 16px;")
        controls_layout.addWidget(self.lbl_selected_win, 1, 0, 1, 2)
        
        lbl_inst = QtWidgets.QLabel("Teclas: [A, E, I, O, U] asignan vocal. [Suprimir] descarta ventana.")
        lbl_inst.setStyleSheet("color: #08F7FE; font-size: 14px; font-weight: bold;")
        controls_layout.addWidget(lbl_inst, 2, 0, 1, 2)
        
        self.btn_segment = QtWidgets.QPushButton("EXPORTAR VENTANAS ETIQUETADAS")
        self.btn_segment.clicked.connect(self.segment_data)
        self.btn_segment.setEnabled(False)
        controls_layout.addWidget(self.btn_segment, 3, 0, 1, 2)
        
        right_panel.addLayout(controls_layout, 1)
        
        main_layout.addLayout(right_panel, 3)
        
    def load_folders(self):
        if not self.base_dir.exists():
            self.lbl_info.setText("Error: base_de_datos_electrodos no existe.")
            return
            
        folders = []
        folders.extend(self.base_dir.rglob("SECUENCIA_PRUEBA_*"))
        folders.extend(self.base_dir.rglob("SecuenciaContinua_*"))
        folders.extend(self.base_dir.rglob("SC_*"))
        
        folders = list(set([f for f in folders if f.is_dir()]))
        folders.sort(key=lambda x: x.name)
        
        for f in folders:
            item = QtWidgets.QListWidgetItem(f.name)
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(f))
            self.list_folders.addItem(item)
            
    def on_folder_selected(self):
        selected = self.list_folders.selectedItems()
        if not selected: return
        
        folder_path = Path(selected[0].data(QtCore.Qt.ItemDataRole.UserRole))
        self.current_folder = folder_path
        
        self.lbl_info.setText(f"Cargando {folder_path.name}...")
        QtWidgets.QApplication.processEvents()
        
        csv_path = folder_path / "grabacion.csv"
        meta_path = folder_path / "metadata.json"
        if not meta_path.exists():
            meta_path = folder_path / "canal_0" / "metadata.json"
            
        if not csv_path.exists():
            self.lbl_info.setText("Error: No se encontró grabacion.csv en la carpeta.")
            return
            
        self.metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"Error leyendo metadata: {e}")
                
        self.bpm = self.metadata.get("bpm", 60.0)
        self.noise_seconds = self.metadata.get("noise_seconds", 5.0)
        self.samplerate = self.metadata.get("sample_rate", 2000.0)
        
        try:
            self.df = pd.read_csv(csv_path)
            if "Tiempo (s)" in self.df.columns and len(self.df) > 1:
                dt = self.df["Tiempo (s)"].iloc[1] - self.df["Tiempo (s)"].iloc[0]
                self.samplerate = 1.0 / dt
                t_arr = self.df["Tiempo (s)"].values
            else:
                t_arr = np.arange(len(self.df)) / self.samplerate
                
            canal_col = "Canal 0" if "Canal 0" in self.df.columns else self.df.columns[1]
            signal_raw = self.df[canal_col].values
            
            # Calcular envolvente igual que analisis_por_track_integrado (RMS 50ms)
            env = _compute_env_full(np.abs(signal_raw), apply_envelope=True, smooth_ms=50, samplerate=self.samplerate, tipo_env="rms")
            
            self.assigned_labels.clear()
            self.text_items.clear()
            self.plot_signal(t_arr, env)
            self.lbl_info.setText(f"Cargado. Haz clic en las ventanas y usa el teclado para etiquetar.")
            self.selected_start_window = None
            self.update_selection_label()
            self.btn_segment.setEnabled(False)
            
        except Exception as e:
            self.lbl_info.setText(f"Error al leer CSV: {e}")
            
    def plot_signal(self, t_arr, env):
        self.plot_widget.clear()
        self.window_spans = []
        
        # Plot envelope
        self.plot_widget.plot(t_arr, env, pen=pg.mkPen('#08F7FE', width=2))
        
        # Llenar zona de ruido
        ruido_region = pg.LinearRegionItem(values=[0, self.noise_seconds], brush=QtGui.QColor(255, 0, 255, 30), movable=False)
        self.plot_widget.addItem(ruido_region)
        
        start_t = self.noise_seconds
        beat_interval = 60.0 / self.bpm
        
        duracion_analizable = t_arr[-1] - start_t
        if duracion_analizable <= 0: return
        
        n_pulsos = math.ceil(duracion_analizable / beat_interval)
        
        # Dibujar rectángulos seleccionables
        for i in range(n_pulsos):
            t0 = start_t + i * beat_interval
            t1 = t0 + beat_interval
            
            region = pg.LinearRegionItem(values=[t0, t1], brush=QtGui.QColor(255, 255, 255, 15), movable=False)
            region.setZValue(-10)
            
            for line in region.lines:
                line.setPen(pg.mkPen(color='#FFFFFF', style=QtCore.Qt.PenStyle.DashLine, width=1))
                line.setHoverPen(pg.mkPen(color='#FE53BB', width=2))
                
            self.plot_widget.addItem(region)
            
            text_item = pg.TextItem(text="?", color='#888888', anchor=(0.5, 0))
            text_item.setPos((t0 + t1) / 2.0, max(env) if len(env) > 0 else 1.0)
            self.plot_widget.addItem(text_item)
            self.text_items[i] = text_item
            
            self.window_spans.append({
                "index": i,
                "t0": t0,
                "t1": t1,
                "region": region
            })
            
    def play_window_audio(self, idx):
        if self.df is None or "Canal 3" not in self.df.columns:
            return
            
        muestras_pulso = int(round(self.samplerate * 60.0 / self.bpm))
        start_sample = int(round(self.noise_seconds * self.samplerate))
        c_start = start_sample + idx * muestras_pulso
        c_end = c_start + muestras_pulso
        
        if c_end > len(self.df):
            c_end = len(self.df)
            
        audio_data = self.df["Canal 3"].iloc[c_start:c_end].values
        
        target_sr = 44100
        if self.samplerate != target_sr and self.samplerate > 0:
            num_samples = int(len(audio_data) * target_sr / self.samplerate)
            resampled_data = signal.resample(audio_data, num_samples, axis=0)
        else:
            resampled_data = audio_data
            
        max_val = np.max(np.abs(resampled_data))
        if max_val > 0:
            resampled_data = np.int16(resampled_data / max_val * 32767)
        else:
            resampled_data = np.int16(resampled_data)
            
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, "temp_segment_playback.wav")
        wav.write(temp_wav, target_sr, resampled_data)
        
        self.player.setSource(QtCore.QUrl.fromLocalFile(temp_wav))
        self.player.play()
        
    def keyPressEvent(self, event):
        if self.selected_start_window is not None:
            key = event.key()
            letra = None
            if key == QtCore.Qt.Key_A: letra = "A"
            elif key == QtCore.Qt.Key_E: letra = "E"
            elif key == QtCore.Qt.Key_I: letra = "I"
            elif key == QtCore.Qt.Key_O: letra = "O"
            elif key == QtCore.Qt.Key_U: letra = "U"
            elif key == QtCore.Qt.Key_Backspace or key == QtCore.Qt.Key_Delete:
                letra = "DESCARTADO"
                
            if letra:
                idx = self.selected_start_window
                if letra == "DESCARTADO":
                    if idx in self.assigned_labels:
                        del self.assigned_labels[idx]
                else:
                    self.assigned_labels[idx] = letra
                    
                self.update_visual_labels()
                self.update_selection_label()
                self.btn_segment.setEnabled(len(self.assigned_labels) > 0)
                
    def on_plot_clicked(self, event):
        if not self.window_spans: return
        
        pos = event.scenePos()
        view_coords = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        x_click = view_coords.x()
        
        clicked_idx = None
        for span in self.window_spans:
            if span["t0"] <= x_click <= span["t1"]:
                clicked_idx = span["index"]
                break
                
        if clicked_idx is not None:
            self.selected_start_window = clicked_idx
            self.play_window_audio(clicked_idx)
            self.update_selection_label()
            self.update_visual_labels()
            
    def update_visual_labels(self):
        for span in self.window_spans:
            idx = span["index"]
            region = span["region"]
            
            if idx == self.selected_start_window:
                region.setBrush(QtGui.QColor(254, 83, 187, 80)) # Seleccionada (Rosa)
            else:
                if idx in self.assigned_labels:
                    region.setBrush(QtGui.QColor(0, 255, 0, 30)) # Verde (Asignada)
                else:
                    region.setBrush(QtGui.QColor(255, 255, 255, 15)) # Gris (No asignada)
                    
            if idx in self.text_items:
                text_item = self.text_items[idx]
                if idx in self.assigned_labels:
                    text_item.setText(f"{self.assigned_labels[idx]}", color='#00FF00')
                else:
                    text_item.setText("?", color='#888888')
            
    def update_selection_label(self):
        if self.selected_start_window is None:
            self.lbl_selected_win.setText("Haz clic en una ventana del gráfico")
            self.lbl_selected_win.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")
        else:
            letra_actual = self.assigned_labels.get(self.selected_start_window, "Ninguna")
            self.lbl_selected_win.setText(f"Ventana seleccionada: {self.selected_start_window} | Etiqueta actual: {letra_actual}")
            self.lbl_selected_win.setStyleSheet("color: #08F7FE; font-weight: bold; font-size: 16px;")
            
    def segment_data(self):
        if not self.assigned_labels or self.df is None: return
        
        self.btn_segment.setEnabled(False)
        self.lbl_info.setText("Concatenando y exportando secuencias...")
        QtWidgets.QApplication.processEvents()
        
        muestras_pulso = int(round(self.samplerate * 60.0 / self.bpm))
        start_sample = int(round(self.noise_seconds * self.samplerate))
        
        ruido_df = self.df.iloc[:start_sample].copy()
        canales_req = [c for c in ["Canal 0", "Canal 1", "Canal 2", "Canal 3"] if c in self.df.columns]
        
        pulsos_por_letra = {}
        for idx, letter in self.assigned_labels.items():
            if letter not in pulsos_por_letra:
                pulsos_por_letra[letter] = []
                
            c_start = start_sample + idx * muestras_pulso
            c_end = c_start + muestras_pulso
            
            if c_end <= len(self.df):
                pulso_df = self.df.iloc[c_start:c_end].copy()
                pulsos_por_letra[letter].append(pulso_df)
            
        extracciones = 0
        parent_date_dir = self.current_folder.parent 
        
        for letter, lista_pulsos in pulsos_por_letra.items():
            if not lista_pulsos:
                continue
                
            # Concatenar: Ruido Basal + (Todos los pulsos de esta letra)
            recorte_df = pd.concat([ruido_df] + lista_pulsos, ignore_index=True)
            if "Tiempo (s)" in recorte_df.columns:
                recorte_df["Tiempo (s)"] = np.arange(len(recorte_df)) / self.samplerate
                
            pulse_folder_name = f"{letter}_{self.current_folder.name}"
            pulse_dir = parent_date_dir / pulse_folder_name
            pulse_dir.mkdir(parents=True, exist_ok=True)
            
            sub_metadata = {
                "measurement_date": self.metadata.get("measurement_date", ""),
                "sample_rate": self.samplerate,
                "channels": self.metadata.get("channels", []),
                "bpm": self.bpm,
                "noise_seconds": self.noise_seconds,
                "pulse_count": len(lista_pulsos),
                "is_formal": False,
                "sujeto": self.metadata.get("sujeto", "Desconocido"),
                "letra": letter,
                "prueba": f"{self.metadata.get('prueba', 'Prueba')}_concat_{letter}",
                "comentario": "Secuencia concatenada interactiva (GUI)"
            }
            
            csv_out = pulse_dir / "grabacion.csv"
            recorte_df.to_csv(csv_out, index=False)
            
            for ch_idx in range(len(canales_req)):
                ch_col = canales_req[ch_idx]
                ch_dir = pulse_dir / f"canal_{ch_idx}"
                ch_dir.mkdir(parents=True, exist_ok=True)
                with open(ch_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(sub_metadata, f, indent=4)
                    
                # Exportar datos de este canal como WAV
                try:
                    import soundfile as sf
                    audio_data = recorte_df[ch_col].values
                    sf.write(str(ch_dir / "grabacion.wav"), audio_data, int(self.samplerate))
                except Exception as e:
                    print(f"Error guardando WAV para canal {ch_idx}: {e}")
            
            extracciones += 1
            
        QtWidgets.QMessageBox.information(self, "Éxito", f"Se exportaron {extracciones} secuencias concatenadas a {self.current_folder.parent}")
        self.lbl_info.setText(f"Exportación de {extracciones} secuencias completada.")
        self.btn_segment.setEnabled(True)

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        
    window = SegmentadorWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
