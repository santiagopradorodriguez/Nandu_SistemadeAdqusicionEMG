# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo de configuración de hardware y parámetros del sistema.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo de configuración de hardware y parámetros del sistema.
# ==============================================================================

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QPushButton, QGroupBox, QFormLayout, QColorDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget, QWidget
)
from PySide6.QtCore import Qt
import os
import sys

# Añadir ruta base
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.config_manager import ConfigManager

class ConfiguracionDialog(QDialog):
    """
    Clase ConfiguracionDialog.

    Representa y gestiona las operaciones relacionadas con ConfiguracionDialog.
    """
    def __init__(self, parent=None):
        """
        Ejecuta la funcionalidad de __init__.

        Args:
            parent (Any): Argumento posicional parent.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Configuración General")
        self.resize(600, 600)
        self.setStyleSheet("background-color: #0f0f0f; color: #00ffcc; font-family: 'Consolas', monospace;")
        
        self.config_manager = ConfigManager()
        self.config_data = self.config_manager.config.copy()
        
        main_layout = QVBoxLayout(self)
        
        # Tabs para separar categorías
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { background: #222; color: #fff; padding: 10px; margin: 2px; }
            QTabBar::tab:selected { background: #00ffcc; color: #000; font-weight: bold; }
            QTabWidget::pane { border: 1px solid #444; }
        """)
        
        self.tab_adq = QWidget()
        self.tab_canales = QWidget()
        self.tab_ui = QWidget()
        
        self.tabs.addTab(self.tab_adq, "Adquisición")
        self.tabs.addTab(self.tab_canales, "Canales y Colores")
        self.tabs.addTab(self.tab_ui, "Interfaz Gráfica")
        
        main_layout.addWidget(self.tabs)
        
        # --- TAB: ADQUISICION ---
        adq_layout = QFormLayout(self.tab_adq)
        adq_conf = self.config_data.get("adquisicion", {})
        
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(100, 10000)
        self.spin_fs.setValue(adq_conf.get("frecuencia_muestreo", 2000.0))
        self.spin_fs.setStyleSheet("background: #222; color: #fff;")
        adq_layout.addRow("Frecuencia Muestreo (Hz):", self.spin_fs)
        
        self.spin_ruido = QDoubleSpinBox()
        self.spin_ruido.setRange(0.0, 10.0)
        self.spin_ruido.setSingleStep(0.5)
        self.spin_ruido.setValue(adq_conf.get("ruido_segundos", 3.0))
        self.spin_ruido.setStyleSheet("background: #222; color: #fff;")
        adq_layout.addRow("Ventana de Ruido (s):", self.spin_ruido)
        
        self.line_nidaq = QLineEdit()
        self.line_nidaq.setText(", ".join(adq_conf.get("nidaq_channels", [])))
        self.line_nidaq.setStyleSheet("background: #222; color: #fff;")
        adq_layout.addRow("Canales NIDAQ (separados por coma):", self.line_nidaq)

        # --- TAB: INTERFAZ ---
        ui_layout = QFormLayout(self.tab_ui)
        ui_conf = self.config_data.get("estetica_global", {})
        
        self.chk_dark_mode = QCheckBox("Tema Oscuro (Cyberpunk)")
        self.chk_dark_mode.setChecked(ui_conf.get("tema_oscuro", True))
        ui_layout.addRow(self.chk_dark_mode)
        
        # --- TAB: CANALES ---
        can_layout = QVBoxLayout(self.tab_canales)
        self.canal_widgets = {}
        
        canales_conf = self.config_data.get("canales", {})
        for i in range(4):
            key = f"Canal {i}"
            ch_data = canales_conf.get(key, {"musculo": key, "color_hex": "#ffffff", "factor_calibracion": 495.0})
            
            gb = QGroupBox(f"Configuración {key}")
            gb.setStyleSheet("QGroupBox { border: 1px solid #444; border-radius: 5px; margin-top: 10px; }")
            fl = QFormLayout(gb)
            
            le_musculo = QLineEdit(ch_data.get("musculo", ""))
            le_musculo.setStyleSheet("background: #222; color: #fff;")
            
            btn_color = QPushButton("Seleccionar Color")
            color_actual = ch_data.get("color_hex", "#ffffff")
            btn_color.setStyleSheet(f"background-color: {color_actual}; color: #000; font-weight: bold;")
            
            # Helper logic to pick color
            def make_color_picker(btn, current_hex):
                """
                Ejecuta la funcionalidad de make_color_picker.

                Args:
                    btn (Any): Argumento posicional btn.
                    current_hex (Any): Argumento posicional current_hex.

                Returns:
                    Any: Resultado de la ejecución de la función.
                """
                def pick_color():
                    """
                    Ejecuta la funcionalidad de pick_color.

                    Returns:
                        Any: Resultado de la ejecución de la función.
                    """
                    from PySide6.QtGui import QColor
                    color = QColorDialog.getColor(QColor(current_hex), self, "Elegir Color")
                    if color.isValid():
                        btn.setStyleSheet(f"background-color: {color.name()}; color: #000; font-weight: bold;")
                        btn.setProperty("hex_value", color.name())
                return pick_color
                
            btn_color.setProperty("hex_value", color_actual)
            btn_color.clicked.connect(make_color_picker(btn_color, color_actual))
            
            spin_cal = QDoubleSpinBox()
            spin_cal.setRange(1.0, 10000.0)
            spin_cal.setValue(ch_data.get("factor_calibracion", 495.0))
            spin_cal.setStyleSheet("background: #222; color: #fff;")
            
            fl.addRow("Músculo:", le_musculo)
            fl.addRow("Color Gráfico:", btn_color)
            fl.addRow("Factor Calibración:", spin_cal)
            
            can_layout.addWidget(gb)
            
            self.canal_widgets[key] = {
                "musculo_widget": le_musculo,
                "color_widget": btn_color,
                "calibracion_widget": spin_cal
            }
            
        # Botones Guardar/Cancelar
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Guardar Configuración")
        btn_save.setStyleSheet("background: #00ffcc; color: #000; font-weight: bold; padding: 10px;")
        btn_save.clicked.connect(self.save_and_close)
        
        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setStyleSheet("background: #555; color: #fff; padding: 10px;")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        main_layout.addLayout(btn_layout)

    def save_and_close(self):
        # Actualizar Dict
        """
        Ejecuta la funcionalidad de save_and_close.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        self.config_data["adquisicion"]["frecuencia_muestreo"] = self.spin_fs.value()
        self.config_data["adquisicion"]["ruido_segundos"] = self.spin_ruido.value()
        self.config_data["adquisicion"]["nidaq_channels"] = [x.strip() for x in self.line_nidaq.text().split(",") if x.strip()]
        
        self.config_data["estetica_global"]["tema_oscuro"] = self.chk_dark_mode.isChecked()
        
        if "canales" not in self.config_data:
            self.config_data["canales"] = {}
            
        for key, w in self.canal_widgets.items():
            self.config_data["canales"][key] = {
                "musculo": w["musculo_widget"].text().strip(),
                "color_hex": w["color_widget"].property("hex_value"),
                "factor_calibracion": w["calibracion_widget"].value()
            }
            
        self.config_manager.config = self.config_data
        self.config_manager.save_config()
        self.accept()
