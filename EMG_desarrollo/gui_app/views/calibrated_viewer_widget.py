# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo calibrated_viewer_widget.py del sistema NANDU LSD.
# ==============================================================================

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap

class CalibratedViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # Barra superior
        top_bar = QHBoxLayout()
        self.lbl_status = QLabel("Listo. Seleccione una medición en el Gestor de Sesiones.")
        self.lbl_status.setStyleSheet("color: #888; font-family: monospace;")
        top_bar.addWidget(self.lbl_status)
        
        self.btn_zoom_in = QPushButton("🔍 +")
        self.btn_zoom_out = QPushButton("🔍 -")
        self.btn_zoom_reset = QPushButton("🔍 1:1")
        
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_reset.clicked.connect(self.zoom_reset)
        
        top_bar.addStretch()
        top_bar.addWidget(self.btn_zoom_out)
        top_bar.addWidget(self.btn_zoom_reset)
        top_bar.addWidget(self.btn_zoom_in)
        self.layout.addLayout(top_bar)

        # Área de Scroll para la imagen
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        
        self.layout.addWidget(self.scroll_area)
        
        self.current_pixmap = None
        self.zoom_factor = 1.0

    def load_calibrated_plot(self, measurement_path):
        """Busca y carga el gráfico calibrado asociado a esta medición"""
        self.lbl_status.setText(f"Buscando gráfico para: {os.path.basename(measurement_path)} ...")
        
        # 1. Buscar en la propia carpeta de la medición
        # Usar la misma lógica de nombre que en plotter_calibrado (Fecha_Medicion)
        padre = os.path.basename(os.path.dirname(measurement_path))
        hijo = os.path.basename(measurement_path)
        rel_path = f"{padre}_{hijo}"
        nombre_limpio = rel_path.replace("/", "_").replace("\\", "_")
        nombre_archivo = f"plot_calibrado_{nombre_limpio}.png"
        
        ruta_img = os.path.join(measurement_path, nombre_archivo)
        
        # 2. Si no está ahí, buscar en analisis_comparativos (como fallback por retrocompatibilidad)
        if not os.path.exists(ruta_img):
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ruta_img = os.path.join(root_dir, "analisis_comparativos", nombre_archivo)
            
        if os.path.exists(ruta_img):
            self.current_pixmap = QPixmap(ruta_img)
            self.zoom_reset()
            self.lbl_status.setText(f"Mostrando: {nombre_archivo}")
        else:
            self.current_pixmap = None
            self.image_label.clear()
            self.image_label.setText("No se encontró ningún gráfico procesado para esta medición.\nEjecuta el Plotter Calibrado primero.")
            self.lbl_status.setText("Gráfico no encontrado.")

    def zoom_in(self):
        self._apply_zoom(1.2)

    def zoom_out(self):
        self._apply_zoom(0.8)

    def zoom_reset(self):
        if self.current_pixmap:
            # Empezamos con un zoom menor por defecto (50%) para que entre mejor en pantalla
            self.zoom_factor = 0.5
            new_size = self.current_pixmap.size() * self.zoom_factor
            scaled_pixmap = self.current_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(new_size)

    def _apply_zoom(self, factor):
        if not self.current_pixmap:
            return
        self.zoom_factor *= factor
        new_size = self.current_pixmap.size() * self.zoom_factor
        scaled_pixmap = self.current_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.resize(new_size)
