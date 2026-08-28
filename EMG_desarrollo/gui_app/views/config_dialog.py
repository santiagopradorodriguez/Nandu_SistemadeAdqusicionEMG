# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo de configuración de hardware y parámetros del sistema.
# ==============================================================================

import os
import json
import sys
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QWidget,
    QLabel, QLineEdit, QPushButton, QFormLayout, QSpinBox, 
    QDoubleSpinBox, QCheckBox, QListWidget, QInputDialog, QMessageBox,
    QColorDialog, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QBrush

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.config_manager import ConfigManager, get_muscle_color, get_unique_channel_colors
config_mgr = ConfigManager()

class ConfiguracionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Configuración General")
        self.resize(750, 560)
        self.setStyleSheet("""
            QDialog { background-color: #0f0f0f; color: #fff; }
            QLabel { color: #ccc; }
            QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #00ffcc; font-weight: bold; }
        """)
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { background: #1a1a1a; color: #aaa; padding: 8px 16px; margin: 2px; font-weight: bold; border-radius: 4px; }
            QTabBar::tab:selected { background: #00ffcc; color: #000; }
        """)
        self.layout.addWidget(self.tabs)
        
        # --- TAB 1: Adquisición ---
        self.tab_adq = QWidget()
        self.form_adq = QFormLayout(self.tab_adq)
        self.form_adq.setContentsMargins(15, 15, 15, 15)
        self.form_adq.setSpacing(10)
        
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(100, 100000)
        self.spin_fs.setDecimals(0)
        self.spin_fs.setStyleSheet("background-color: #222; color: #fff; padding: 4px;")
        
        self.spin_noise = QDoubleSpinBox()
        self.spin_noise.setRange(0.5, 30.0)
        self.spin_noise.setSingleStep(0.5)
        self.spin_noise.setStyleSheet("background-color: #222; color: #fff; padding: 4px;")
        
        self.spin_bpm = QSpinBox()
        self.spin_bpm.setRange(30, 250)
        self.spin_bpm.setStyleSheet("background-color: #222; color: #fff; padding: 4px;")
        
        self.spin_descanso = QDoubleSpinBox()
        self.spin_descanso.setRange(1.0, 60.0)
        self.spin_descanso.setSingleStep(1.0)
        self.spin_descanso.setStyleSheet("background-color: #222; color: #fff; padding: 4px;")
        
        self.list_channels = QListWidget()
        self.list_channels.setStyleSheet("background-color: #151515; color: #00ffcc; border: 1px solid #333;")
        self.btn_add_chan = QPushButton("+ Agregar")
        self.btn_remove_chan = QPushButton("- Quitar")
        btn_adq_style = """
            QPushButton { background-color: #222; color: #00ffcc; border: 1px solid #00ffcc; padding: 4px 10px; border-radius: 3px; font-weight: bold; }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
        """
        self.btn_add_chan.setStyleSheet(btn_adq_style)
        self.btn_remove_chan.setStyleSheet(btn_adq_style)
        
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.btn_add_chan)
        btn_layout.addWidget(self.btn_remove_chan)
        
        self.form_adq.addRow("Frecuencia de Muestreo (Hz):", self.spin_fs)
        self.form_adq.addRow("Segundos de Ruido Base:", self.spin_noise)
        self.form_adq.addRow("BPM (Metrónomo):", self.spin_bpm)
        self.form_adq.addRow("Tiempo de Descanso (s):", self.spin_descanso)
        self.form_adq.addRow("Canales Físicos NIDAQ:", self.list_channels)
        self.form_adq.addRow("", btn_layout)
        
        self.btn_add_chan.clicked.connect(self._add_channel)
        self.btn_remove_chan.clicked.connect(self._remove_channel)
        
        self.tabs.addTab(self.tab_adq, "Adquisición (DAQ)")
        
        # --- TAB 2: Mapeo de Canales, Músculos y Colores ---
        self.tab_map = QWidget()
        self.layout_map = QVBoxLayout(self.tab_map)
        self.layout_map.setContentsMargins(10, 10, 10, 10)
        self.layout_map.setSpacing(8)
        
        lbl_info_map = QLabel("Personaliza los nombres de músculos y colores por canal. Haz doble clic en 'Color HEX' o usa los botones inferiores para elegir un color.")
        lbl_info_map.setStyleSheet("color: #888; font-size: 11px;")
        self.layout_map.addWidget(lbl_info_map)
        
        self.table_map = QTableWidget(16, 5)
        self.table_map.setHorizontalHeaderLabels(["ID Canal", "Nombre Músculo", "Factor Calibración", "Color HEX", "Activo Default"])
        self.table_map.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_map.setStyleSheet("""
            QTableWidget { background-color: #121212; color: #fff; border: 1px solid #333; gridline-color: #222; }
            QHeaderView::section { background-color: #1a1a1a; color: #00ffcc; font-weight: bold; border: 1px solid #333; padding: 4px; }
        """)
        self.table_map.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.layout_map.addWidget(self.table_map)
        
        # Botones de gestión de colores
        h_color_btns = QHBoxLayout()
        self.btn_pick_color = QPushButton("Cambiar Color del Canal Seleccionado")
        self.btn_pick_color.setStyleSheet("""
            QPushButton { background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc; padding: 6px 14px; font-weight: bold; border-radius: 4px; font-size: 11px; }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
        """)
        self.btn_pick_color.clicked.connect(self._pick_color_for_selected)
        
        self.btn_auto_colors = QPushButton("Auto-Asignar Colores Únicos (Sin Repeticiones)")
        self.btn_auto_colors.setStyleSheet("""
            QPushButton { background-color: #1a1a1a; color: #ffaa00; border: 1px solid #ffaa00; padding: 6px 14px; font-weight: bold; border-radius: 4px; font-size: 11px; }
            QPushButton:hover { background-color: #ffaa00; color: #000; }
        """)
        self.btn_auto_colors.clicked.connect(self._auto_assign_unique_colors)
        
        h_color_btns.addWidget(self.btn_pick_color)
        h_color_btns.addWidget(self.btn_auto_colors)
        self.layout_map.addLayout(h_color_btns)
        
        self.tabs.addTab(self.tab_map, "Músculos y Colores")
        
        # --- TAB 3: Estética Global ---
        self.tab_estetica = QWidget()
        self.form_estetica = QFormLayout(self.tab_estetica)
        self.form_estetica.setContentsMargins(15, 15, 15, 15)
        self.form_estetica.setSpacing(10)
        
        self.chk_dark = QCheckBox("Modo Cyberpunk/Oscuro Activo")
        self.form_estetica.addRow("Tema Global:", self.chk_dark)
        
        self.tabs.addTab(self.tab_estetica, "Estética Global")
        
        # --- BOTÓN GUARDAR ---
        self.btn_save = QPushButton("Guardar Configuración")
        self.btn_save.setStyleSheet("""
            QPushButton { background-color: #00cc88; color: #000; font-weight: bold; font-size: 13px; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #00ffaa; }
        """)
        self.btn_save.clicked.connect(self._save_config)
        self.layout.addWidget(self.btn_save)
        
        # Cargar Datos
        self._load_config()
        
    def _load_config(self):
        adq = config_mgr.get("adquisicion") or {}
        self.spin_fs.setValue(adq.get("frecuencia_muestreo", 2000.0))
        self.spin_noise.setValue(adq.get("ruido_segundos", 3.0))
        self.spin_bpm.setValue(adq.get("bpm", 60))
        self.spin_descanso.setValue(adq.get("tiempo_descanso", 10.0))
        
        nidaq_chans = adq.get("nidaq_channels", ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3"])
        self.list_channels.clear()
        for ch in nidaq_chans:
            self.list_channels.addItem(ch)
            
        # Cargar Tabla de Músculos
        canales = config_mgr.get("canales") or {}
        for i in range(16):
            key = f"Canal {i}"
            data = canales.get(key, {})
            
            it_id = QTableWidgetItem(key)
            it_id.setFlags(Qt.ItemIsEnabled)
            self.table_map.setItem(i, 0, it_id)
            
            it_musc = QTableWidgetItem(data.get("musculo", f"Canal {i}"))
            self.table_map.setItem(i, 1, it_musc)
            
            it_cal = QTableWidgetItem(str(data.get("factor_calibracion", 495.0)))
            self.table_map.setItem(i, 2, it_cal)
            
            # Obtener color con resolución única
            col_hex = data.get("color_hex")
            if not col_hex:
                col_hex = get_muscle_color(it_musc.text(), default="#00ffcc")
                
            it_color = QTableWidgetItem(col_hex)
            it_color.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it_color.setBackground(QBrush(QColor(col_hex)))
            luma = QColor(col_hex).lightness()
            it_color.setForeground(QBrush(QColor("black") if luma > 128 else QColor("white")))
            self.table_map.setItem(i, 3, it_color)
            
            it_activo = QTableWidgetItem("")
            it_activo.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it_activo.setCheckState(Qt.Checked if data.get("activo_por_defecto", i < 4) else Qt.Unchecked)
            self.table_map.setItem(i, 4, it_activo)
            
        # Cargar Estética
        estetica = config_mgr.get("estetica_global") or {}
        self.chk_dark.setChecked(estetica.get("tema_oscuro", True))
        
    def _pick_color_for_selected(self):
        row = self.table_map.currentRow()
        if row >= 0:
            self._on_cell_double_clicked(row, 3)
        else:
            QMessageBox.information(self, "Selección", "Por favor selecciona una fila de la tabla para cambiar su color.")

    def _auto_assign_unique_colors(self):
        ch_list = []
        for i in range(16):
            musc = self.table_map.item(i, 1).text().strip()
            ch_list.append({"idx": i, "musculo": musc, "is_mic": (i == 3 or "mic" in musc.lower())})
        
        unique_colors = get_unique_channel_colors(ch_list)
        for i in range(16):
            it_color = self.table_map.item(i, 3)
            new_c = unique_colors[i]
            it_color.setText(new_c)
            it_color.setBackground(QBrush(QColor(new_c)))
            luma = QColor(new_c).lightness()
            it_color.setForeground(QBrush(QColor("black") if luma > 128 else QColor("white")))

    def _on_cell_double_clicked(self, row, column):
        if column == 3:
            item = self.table_map.item(row, column)
            current_color = item.text()
            color = QColorDialog.getColor(QColor(current_color), self, f"Seleccionar Color para Canal {row}")
            if color.isValid():
                item.setText(color.name())
                item.setBackground(QBrush(color))
                luma = color.lightness()
                item.setForeground(QBrush(QColor("black") if luma > 128 else QColor("white")))

    def _add_channel(self):
        text, ok = QInputDialog.getText(self, "Añadir Canal NIDAQ", "Ingresa el nombre físico (ej. Dev1/ai4):")
        if ok and text:
            self.list_channels.addItem(text)
            
    def _remove_channel(self):
        for item in self.list_channels.selectedItems():
            self.list_channels.takeItem(self.list_channels.row(item))
            
    def _save_config(self):
        # 1. Guardar DAQ
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
        
        # 2. Guardar Canales y Colores Personalizados por Músculo
        canales = config_mgr.config.get("canales", {})
        colores_musculos = config_mgr.config.get("colores_musculos", {})
        
        for i in range(16):
            key = f"Canal {i}"
            try:
                musc = self.table_map.item(i, 1).text().strip()
                cal = float(self.table_map.item(i, 2).text().strip())
                color = self.table_map.item(i, 3).text().strip()
                activo = (self.table_map.item(i, 4).checkState() == Qt.Checked)
                canales[key] = {
                    "musculo": musc,
                    "factor_calibracion": cal,
                    "color_hex": color,
                    "activo_por_defecto": activo
                }
                # Guardar asociación personalizada músculo -> color
                if musc and not ("mic" in musc.lower() or i == 3):
                    colores_musculos[musc.lower()] = color
            except Exception as e:
                print(f"Error guardando canal {i}: {e}")
                
        config_mgr.config["canales"] = canales
        config_mgr.config["colores_musculos"] = colores_musculos
        
        # 3. Guardar Estética
        est = config_mgr.config.get("estetica_global", {})
        est["tema_oscuro"] = self.chk_dark.isChecked()
        config_mgr.config["estetica_global"] = est
        
        config_mgr.save_config()
        QMessageBox.information(self, "Guardado", "Configuración guardada con éxito. Los colores únicos asignados se aplicarán a todos los gráficos y módulos del sistema.")
        self.accept()
