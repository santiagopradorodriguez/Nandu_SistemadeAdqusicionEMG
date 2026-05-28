import re
import sys

with open("Nandu_AutoForge_DAQ.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Inject AutoForgeDialog before RealTimePlotter
dialog_code = """
class AutoForgeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de AutoForge")
        self.setMinimumWidth(300)
        
        self.layout = QtWidgets.QFormLayout(self)
        
        self.edit_prueba = QtWidgets.QLineEdit("Prueba1")
        self.edit_sujeto = QtWidgets.QLineEdit("Sujeto1")
        self.spin_reps = QtWidgets.QSpinBox()
        self.spin_reps.setRange(1, 1000)
        self.spin_reps.setValue(25)
        self.spin_bpm = QtWidgets.QSpinBox()
        self.spin_bpm.setRange(30, 300)
        self.spin_bpm.setValue(60)
        
        self.layout.addRow("Prueba:", self.edit_prueba)
        self.layout.addRow("Sujeto:", self.edit_sujeto)
        self.layout.addRow("Repeticiones:", self.spin_reps)
        self.layout.addRow("BPM Metrónomo:", self.spin_bpm)
        
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

"""

if "class AutoForgeDialog" not in content:
    content = content.replace("class RealTimePlotter", dialog_code + "class RealTimePlotter")

# 2. Fix the overlay parent and visibility in _setup_ui_plots
overlay_code = """
        # --- NUEVO: Overlay AutoForge Flotante ---
        self.autoforge_overlay = QtWidgets.QLabel(self.plot_widget)
        self.autoforge_overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.autoforge_overlay.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: #00FFFF; font-size: 150px; font-weight: bold;")
        self.autoforge_overlay.hide()
        
        # Hacemos que cambie de tamaño junto con plot_widget
        self.plot_widget.installEventFilter(self)
"""
if "self.autoforge_overlay = QtWidgets.QLabel(\"\")" in content:
    content = content.replace(
        "        self.autoforge_overlay = QtWidgets.QLabel(\"\")\n" +
        "        self.autoforge_overlay.setAlignment(QtCore.Qt.AlignCenter)\n" +
        "        self.autoforge_overlay.setStyleSheet(\"background-color: rgba(0, 0, 0, 200); color: #00FFFF; font-size: 150px; font-weight: bold;\")\n" +
        "        self.autoforge_overlay.hide()\n",
        ""
    )
    content = content.replace("self.plot_widget = pg.GraphicsLayoutWidget()", "self.plot_widget = pg.GraphicsLayoutWidget()\n" + overlay_code)

# We need an eventFilter to resize the overlay
event_filter_code = """
    def eventFilter(self, obj, event):
        if obj == self.plot_widget and event.type() == QtCore.QEvent.Resize:
            if hasattr(self, 'autoforge_overlay'):
                self.autoforge_overlay.resize(self.plot_widget.size())
        return super().eventFilter(obj, event)

    def iniciar_autoforge(self):
"""
if "def eventFilter" not in content:
    content = content.replace("    def iniciar_autoforge(self):", event_filter_code)

# 3. Revert graficos_container logic
if "self.graficos_container = QtWidgets.QStackedWidget()" in content:
    old_layout = """        # Contenedor para intercambiar gráficos y overlay
        self.graficos_container = QtWidgets.QStackedWidget()
        self.graficos_container.addWidget(self.splitter)
        self.graficos_container.addWidget(self.autoforge_overlay)
        
        # --- Añadir layouts a la ventana ---
        self.main_layout.addWidget(self.config_groupbox)
        self.main_layout.addWidget(self.filter_groupbox)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.trigger_layout)
        self.main_layout.addWidget(self.measure_widget)
        self.main_layout.addWidget(self.spectrogram_groupbox) # Añadir controles del espectrograma
        self.main_layout.addWidget(self.graficos_container) # Añadir el divisor con los gráficos"""
    
    new_layout = """        # --- Añadir layouts a la ventana ---
        self.main_layout.addWidget(self.config_groupbox)
        self.main_layout.addWidget(self.filter_groupbox)
        self.main_layout.addLayout(self.button_layout)
        self.main_layout.addLayout(self.trigger_layout)
        self.main_layout.addWidget(self.measure_widget)
        self.main_layout.addWidget(self.spectrogram_groupbox) # Añadir controles del espectrograma
        self.main_layout.addWidget(self.splitter) # Añadir el divisor con los gráficos"""
    content = content.replace(old_layout, new_layout)

# 4. Replace AutoForge methods completely
start_idx = content.find("    def iniciar_autoforge(self):")
end_idx = content.find("    def closeEvent(self, event):")

new_autoforge_methods = """    def iniciar_autoforge(self):
        try:
            if not self.is_acquiring:
                # Disparar automáticamente la adquisición
                self.on_start_acq_click()
                
            import os
            ruta_palabras = os.path.join(os.path.dirname(os.path.abspath(__file__)), "palabras.txt")
            if not os.path.exists(ruta_palabras):
                QtWidgets.QMessageBox.warning(self, "Error", "Falta palabras.txt en la carpeta del script.")
                return
                
            with open(ruta_palabras, 'r', encoding='utf-8') as f:
                self.autoforge_words = [line.strip() for line in f if line.strip()]
                
            if not self.autoforge_words:
                QtWidgets.QMessageBox.warning(self, "Error", "palabras.txt está vacío.")
                return
                
            dialog = AutoForgeDialog(self)
            if dialog.exec() == QtWidgets.QDialog.Accepted:
                self.autoforge_prueba = dialog.edit_prueba.text().strip()
                self.autoforge_sujeto = dialog.edit_sujeto.text().strip()
                self.autoforge_target_reps = dialog.spin_reps.value()
                bpm = dialog.spin_bpm.value()
                
                # Actualizar el spinbox principal de BPM si existe
                try:
                    self.spin_bpm.setValue(bpm)
                except:
                    pass
                
                self.autoforge_word_idx = 0
                
                # Lanzar el metrónomo silenciado si no estaba
                if not self.metronome_process or self.metronome_process.poll() is not None:
                    python_executable = sys.executable
                    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metronomo_visual.py')
                    self.metronome_process = subprocess.Popen(
                        [python_executable, script_path, '--autostart', '--mute'],
                        stdin=subprocess.PIPE, text=True
                    )
                else:
                    self._send_metronome_cmd("MUTE")
                    
                self.estado_1_baseline()
        except Exception as e:
            import traceback
            with open("error_autoforge.txt", "w") as f:
                f.write(traceback.format_exc())

    def _send_metronome_cmd(self, cmd):
        if self.metronome_process and self.metronome_process.poll() is None:
            try:
                # CRÍTICO: Windows Popen usa \\r\\n para el enter a veces, y requiere flush
                self.metronome_process.stdin.write(cmd + "\\r\\n")
                self.metronome_process.stdin.flush()
            except Exception as e:
                print(f"Error metronomo cmd: {e}")

    def estado_1_baseline(self):
        self.autoforge_overlay.setText("SILENCIO ABSOLUTO\\n(10s)")
        self.autoforge_overlay.show()
        self.autoforge_overlay.raise_()
        
        self.current_recording = []
        self.is_recording = True
        
        QtCore.QTimer.singleShot(10000, self.terminar_baseline)

    def terminar_baseline(self):
        self.is_recording = False
        import threading
        def guardar_async():
            import os
            from pathlib import Path
            from datetime import datetime
            fecha_str = datetime.now().strftime("%Y-%m-%d")
            # Nombre de baseline: BASELINE_Prueba_Sujeto
            folder_name = f"BASELINE_{self.autoforge_prueba}_{self.autoforge_sujeto}"
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "base_de_datos_electrodos" / fecha_str / folder_name
            os.makedirs(base_dir, exist_ok=True)
            guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
        threading.Thread(target=guardar_async, daemon=True).start()
        
        self.estado_2_preparacion()

    def estado_2_preparacion(self):
        if self.autoforge_word_idx >= len(self.autoforge_words):
            self.autoforge_overlay.setText("¡COMPLETADO!")
            QtCore.QTimer.singleShot(2000, self.autoforge_overlay.hide)
            self._send_metronome_cmd("STOP_APP")
            return
            
        palabra = self.autoforge_words[self.autoforge_word_idx]
        self._send_metronome_cmd("UNMUTE")
        
        self.autoforge_countdown = 3
        self.autoforge_overlay.setText(f"{palabra}\\n{self.autoforge_countdown}")
        QtCore.QTimer.singleShot(1000, self._estado_2_tick)

    def _estado_2_tick(self):
        self.autoforge_countdown -= 1
        palabra = self.autoforge_words[self.autoforge_word_idx]
        if self.autoforge_countdown > 0:
            self.autoforge_overlay.setText(f"{palabra}\\n{self.autoforge_countdown}")
            QtCore.QTimer.singleShot(1000, self._estado_2_tick)
        else:
            self.estado_3_accion()

    def estado_3_accion(self):
        palabra = self.autoforge_words[self.autoforge_word_idx]
        self.autoforge_overlay.setText(f"{palabra}\\nGRABANDO...")
        
        self.current_recording = []
        self.is_recording = True
        self._send_metronome_cmd("START_COUNTING")
        
        # Calcular tiempo: reps * (60/BPM)
        bpm = self.spin_bpm.value()
        if bpm <= 0: bpm = 60
        ms_totales = int((60000 / bpm) * self.autoforge_target_reps)
        QtCore.QTimer.singleShot(ms_totales, self.estado_4_guardado)

    def estado_4_guardado(self):
        self.is_recording = False
        palabra = self.autoforge_words[self.autoforge_word_idx]
        self.autoforge_overlay.setText("GUARDANDO...")
        
        import threading
        def guardar_async():
            import os
            from pathlib import Path
            from datetime import datetime
            fecha_str = datetime.now().strftime("%Y-%m-%d")
            # Nombre: PALABRA_Prueba_Sujeto
            folder_name = f"{palabra}_{self.autoforge_prueba}_{self.autoforge_sujeto}"
            base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "base_de_datos_electrodos" / fecha_str / folder_name
            os.makedirs(base_dir, exist_ok=True)
            guardar_grabacion_csv(self.current_recording, self.SAMPLE_RATE, str(base_dir), self.NUM_CANALES, "grabacion")
            
            QtCore.QMetaObject.invokeMethod(self, "estado_5_reset", QtCore.Qt.QueuedConnection)
        threading.Thread(target=guardar_async, daemon=True).start()

    @QtCore.Slot()
    def estado_5_reset(self):
        self.autoforge_word_idx += 1
        self.estado_2_preparacion()

"""

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + new_autoforge_methods + content[end_idx:]

with open("Nandu_AutoForge_DAQ.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Patcher script finished")
