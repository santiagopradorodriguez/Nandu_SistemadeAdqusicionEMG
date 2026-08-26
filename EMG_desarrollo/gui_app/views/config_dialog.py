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

import os
import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QFormLayout, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QListWidget, QInputDialog, QMessageBox,
    QColorDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt

import sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.config_manager import ConfigManager
config_mgr = ConfigManager()

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
        self.setWindowTitle("Ñandú LSD - Configuración General de Ñandú LSD")
        self.resize(700, 500)
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.layout.addWidget(self.tabs)
        
        # --- TAB: Adquisición ---
        self.tab_adq = QWidget()
        self.form_adq = QFormLayout(self.tab_adq)
        
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(100, 100000)
        self.spin_fs.setDecimals(0)
        
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0.5, 30.0)
        self.spin_noise.setSingleStep(0.5)
        
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(30, 250)
        
        self.spin_descanso = QDoubleSpinBox()
        self.spin_descanso.setRange(1.0, 60.0)
        self.spin_descanso.setSingleStep(1.0)
        
        self.list_channels = QListWidget()
        self.btn_add_chan = QPushButton("+ Agregar")
        self.btn_remove_chan = QPushButton("- Quitar")
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_add_chan)
        btn_layout.addWidget(self.btn_remove_chan)
        
        self.form_adq.addRow("Frecuencia Muestreo por defecto (Hz):", self.spin_fs)
        self.form_adq.addRow("Segundos de Ruido por defecto:", self.spin_noise)
        self.form_adq.addRow("BPM (Metrónomo) por defecto:", self.spin_bpm)
        self.form_adq.addRow("Tiempo de Descanso (s):", self.spin_descanso)
        self.form_adq.addRow("Canales Físicos NIDAQ:", self.list_channels)
        self.form_adq.addRow("", btn_layout)
        
        self.btn_add_chan.clicked.connect(self._add_channel)
        self.btn_remove_chan.clicked.connect(self._remove_channel)
        
        self.tabs.addTab(self.tab_adq, "Adquisición (DAQ)")
        
        # --- TAB: Mapeo de Canales y Músculos ---
        self.tab_map = QWidget()
        self.layout_map = QVBoxLayout(self.tab_map)
        
        self.table_map = QTableWidget(16, 5) # Soporte para 16 canales lógicos
        self.table_map.setHorizontalHeaderLabels(["ID Canal", "Nombre Músculo", "Factor Calibración", "Color HEX", "Activo Default"])
        self.table_map.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_map.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.layout_map.addWidget(self.table_map)
        
        self.tabs.addTab(self.tab_map, "Músculos y Colores")
        
        # --- TAB: Estética ---
        self.tab_estetica = QWidget()
        self.form_estetica = QFormLayout(self.tab_estetica)
        
        self.chk_dark = QCheckBox("Modo Cyberpunk/Oscuro Activo")
        self.form_estetica.addRow("Tema Global:", self.chk_dark)
        
        self.tabs.addTab(self.tab_estetica, "Estética Global")
        
        # --- BOTONES GUARDAR/CANCELAR ---
        self.btn_save = QPushButton("Guardar Configuración")
        self.btn_save.setStyleSheet("background-color: #00aa00; font-weight: bold;")
        self.btn_save.clicked.connect(self._save_config)
        
        self.layout.addWidget(self.btn_save)
        
        # --- Cargar Datos ---
        self._load_config()
        
    def _load_config(self):
        # Cargar DAQ
        """
        Ejecuta la funcionalidad de _load_config.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        adq = config_mgr.get("adquisicion") or {}
        self.spin_fs.setValue(adq.get("frecuencia_muestreo", 2000.0))
        self.spin_noise.setValue(adq.get("ruido_segundos", 3.0))
        self.spin_bpm.setValue(adq.get("bpm", 60))
        self.spin_descanso.setValue(adq.get("tiempo_descanso", 10.0))
        
        nidaq_chans = adq.get("nidaq_channels", ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"])
        for ch in nidaq_chans:
            self.list_channels.addItem(ch)
            
        # Cargar Tabla de Músculos
        canales = config_mgr.get("canales") or {}
        for i in range(16):
            key = f"Canal {i}"
            data = canales.get(key, {})
            
            it_id = QTableWidgetItem(key)
            it_id.setFlags(Qt.ItemIsEnabled) # Solo lectura
            self.table_map.setItem(i, 0, it_id)
            
            it_musc = QTableWidgetItem(data.get("musculo", f"Musculo {i}"))
            self.table_map.setItem(i, 1, it_musc)
            
            it_cal = QTableWidgetItem(str(data.get("factor_calibracion", 495.0)))
            self.table_map.setItem(i, 2, it_cal)
            
            it_color = QTableWidgetItem(data.get("color_hex", "#00ffcc"))
            it_color.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            from PySide6.QtGui import QColor, QBrush
            it_color.setBackground(QBrush(QColor(data.get("color_hex", "#00ffcc"))))
            # Set text color to contrast background
            luma = QColor(data.get("color_hex", "#00ffcc")).lightness()
            it_color.setForeground(QBrush(QColor("black") if luma > 128 else QColor("white")))
            self.table_map.setItem(i, 3, it_color)
            
            it_activo = QTableWidgetItem("")
            it_activo.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it_activo.setCheckState(Qt.Checked if data.get("activo_por_defecto", False) else Qt.Unchecked)
            self.table_map.setItem(i, 4, it_activo)
            
        # Cargar Estética
        estetica = config_mgr.get("estetica_global") or {}
        self.chk_dark.setChecked(estetica.get("tema_oscuro", True))
        
    def _on_cell_double_clicked(self, row, column):
        """
        Ejecuta la funcionalidad de _on_cell_double_clicked.

        Args:
            row (Any): Argumento posicional row.
            column (Any): Argumento posicional column.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        if column == 3: # Color column
            item = self.table_map.item(row, column)
            current_color = item.text()
            from PySide6.QtGui import QColor, QBrush
            color = QColorDialog.getColor(QColor(current_color), self, f"Color para Canal {row}")
            if color.isValid():
                item.setText(color.name())
                item.setBackground(QBrush(color))
                luma = color.lightness()
                item.setForeground(QBrush(QColor("black") if luma > 128 else QColor("white")))

    def _add_channel(self):
        """
        Ejecuta la funcionalidad de _add_channel.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        text, ok = QInputDialog.getText(self, "Añadir Canal NIDAQ", "Ingresa el nombre físico (ej. Dev1/ai4):")
        if ok and text:
            self.list_channels.addItem(text)
            
    def _remove_channel(self):
        """
        Ejecuta la funcionalidad de _remove_channel.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        for item in self.list_channels.selectedItems():
            self.list_channels.takeItem(self.list_channels.row(item))
            
    def _save_config(self):
        # 1. Guardar DAQ
        """
        Ejecuta la funcionalidad de _save_config.

        Returns:
            Any: Resultado de la ejecución de la función.
        """
        adq = config_mgr.get("adquisicion") or {}
        adq["frecuencia_muestreo"] = self.spin_fs.value()
        adq["ruido_segundos"] = self.spin_noise.value()
        adq["bpm"] = self.spin_bpm.value()
        adq["tiempo_descanso"] = self.spin_descanso.value()
        
        nidaq_chans = []
        for i in range(self.list_channels.count()):
            nidaq_chans.append(self.list_channels.item(i).text())
        adq["nidaq_channels"] = nidaq_chans
        
        config_mgr.config["adquisicion"] = adq
        
        # 2. Guardar Canales
        canales = config_mgr.config.get("canales", {})
        for i in range(16):
            key = f"Canal {i}"
            try:
                musc = self.table_map.item(i, 1).text()
                cal = float(self.table_map.item(i, 2).text())
                color = self.table_map.item(i, 3).text()
                activo = (self.table_map.item(i, 4).checkState() == Qt.Checked)
                canales[key] = {
                    "musculo": musc,
                    "factor_calibracion": cal,
                    "color_hex": color,
                    "activo_por_defecto": activo
                }
            except Exception as e:
                print(f"Error guardando canal {i}: {e}")
        config_mgr.config["canales"] = canales
        
        # 3. Guardar Estética
        est = config_mgr.config.get("estetica_global", {})
        est["tema_oscuro"] = self.chk_dark.isChecked()
        config_mgr.config["estetica_global"] = est
        
        config_mgr.save_config()
        QMessageBox.information(self, "Guardado", "Configuración global guardada correctamente. Reinicia los módulos que tengas abiertos para que surtan efecto los cambios.")
        self.accept()
