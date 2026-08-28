# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo calibrated_viewer_widget.py del sistema NANDU LSD.
# ==============================================================================

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton,
    QGroupBox, QCheckBox, QRadioButton, QLineEdit, QFormLayout, QComboBox,
    QDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QSize, QEvent
from PySide6.QtGui import QPixmap, QCursor

class CalibratedViewerWidget(QWidget):
    request_generate_plots = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(4)

        self.current_pixmap = None
        self.current_measurement_path = None
        self.zoom_factor = 1.0
        self.fit_to_window = True

        # Barra superior
        top_bar = QHBoxLayout()
        top_bar.setSpacing(5)
        
        self.lbl_status = QLabel("Listo. Seleccione una medición en el Gestor de Sesiones.")
        self.lbl_status.setStyleSheet("color: #888; font-family: monospace; font-size: 11px;")
        top_bar.addWidget(self.lbl_status)
        
        top_bar.addStretch()
        
        btn_style = """
            QPushButton {
                background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc;
                padding: 4px 12px; font-weight: bold; border-radius: 4px; margin: 1px; font-size: 11px;
            }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
        """
        self.btn_plot_3m = QPushButton("Plot para Paper (3 Músculos)")
        self.btn_plot_3m.setStyleSheet(btn_style)
        top_bar.addWidget(self.btn_plot_3m)
        
        # Selector de tipo de plot
        self.cmb_tipo_plot = QComboBox()
        self.cmb_tipo_plot.addItems([
            "Vista: Plot Calibrado (Auto)", 
            "Vista: Plot Paper (Completo)"
        ])
        self.cmb_tipo_plot.setStyleSheet("background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc; padding: 3px; font-weight: bold; border-radius: 4px; font-size: 11px;")
        self.cmb_tipo_plot.currentIndexChanged.connect(self._reload_current_plot)
        top_bar.addWidget(self.cmb_tipo_plot)
        
        # Selector de suavizado
        lbl_smooth = QLabel("Suavizado (ms):")
        lbl_smooth.setStyleSheet("font-size: 11px; color: #ccc;")
        top_bar.addWidget(lbl_smooth)
        self.inp_smooth = QLineEdit("250")
        self.inp_smooth.setMaximumWidth(45)
        self.inp_smooth.setStyleSheet("background-color: #222; color: white; border: 1px solid gray; padding: 2px; font-size: 11px;")
        top_bar.addWidget(self.inp_smooth)
        
        # Controles de Zoom y Ajuste
        self.btn_zoom_fit = QPushButton(" Ajustar")
        self.btn_zoom_fit.setToolTip("Ajustar imagen a la ventana (Auto-Fit)")
        self.btn_zoom_reset = QPushButton(" 1:1")
        self.btn_zoom_reset.setToolTip("Tamaño real 100%")
        self.btn_zoom_out = QPushButton(" -")
        self.btn_zoom_out.setToolTip("Alejar zoom")
        self.btn_zoom_in = QPushButton(" +")
        self.btn_zoom_in.setToolTip("Acercar zoom")
        self.btn_fullscreen = QPushButton(" Pantalla Completa")
        self.btn_fullscreen.setToolTip("Ver en alta resolución a pantalla completa")
        
        for btn in [self.btn_zoom_fit, self.btn_zoom_reset, self.btn_zoom_out, self.btn_zoom_in, self.btn_fullscreen]:
            btn.setStyleSheet(btn_style)
            top_bar.addWidget(btn)

        self.btn_zoom_fit.clicked.connect(self.fit_to_view)
        self.btn_zoom_reset.clicked.connect(self.zoom_reset)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_fullscreen.clicked.connect(self.show_fullscreen)
        
        self.lbl_zoom_indicator = QLabel("Auto-Ajustado")
        self.lbl_zoom_indicator.setStyleSheet("color: #888; font-size: 11px; padding-left: 2px;")
        top_bar.addWidget(self.lbl_zoom_indicator)
        
        self.layout.addLayout(top_bar)

        # Layout horizontal para panel izquierdo y área de imagen
        h_layout = QHBoxLayout()
        h_layout.setSpacing(6)
        
        # --- PANEL IZQUIERDO: Configuración del Graficador ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 5, 0)
        left_layout.setSpacing(6)
        
        # Filtros
        filtros_group = QGroupBox("Filtros Digitales")
        filtros_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 8px; font-size: 11px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #00ffcc; font-weight: bold; }")
        filtros_layout = QVBoxLayout(filtros_group)
        filtros_layout.setContentsMargins(6, 8, 6, 6)
        filtros_layout.setSpacing(4)
        self.chk_notch = QCheckBox("Filtro Notch (50 Hz)")
        self.chk_notch.setChecked(True)
        self.chk_bandpass = QCheckBox("Filtro Pasabanda (20-1000 Hz)")
        self.chk_bandpass.setChecked(True)
        filtros_layout.addWidget(self.chk_notch)
        filtros_layout.addWidget(self.chk_bandpass)
        left_layout.addWidget(filtros_group)
        
        # Envolvente
        env_group = QGroupBox("Procesamiento / Envolvente")
        env_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 8px; font-size: 11px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #00ffcc; font-weight: bold; }")
        env_layout = QVBoxLayout(env_group)
        env_layout.setContentsMargins(6, 8, 6, 6)
        env_layout.setSpacing(4)
        self.rb_ninguna = QRadioButton("Solo Señal Filtrada")
        self.rb_ninguna.setChecked(True)
        self.rb_hilbert = QRadioButton("Envolvente de Hilbert")
        self.rb_rms = QRadioButton("Envolvente RMS (75ms)")
        env_layout.addWidget(self.rb_ninguna)
        env_layout.addWidget(self.rb_hilbert)
        env_layout.addWidget(self.rb_rms)
        left_layout.addWidget(env_group)
        
        # Intervalo de Tiempo
        time_group = QGroupBox("Intervalo de Tiempo (s)")
        time_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 8px; font-size: 11px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #00ffcc; font-weight: bold; }")
        time_layout = QFormLayout(time_group)
        time_layout.setContentsMargins(6, 8, 6, 6)
        time_layout.setSpacing(4)
        self.entry_inicio = QLineEdit()
        self.entry_fin = QLineEdit()
        time_layout.addRow("Inicio:", self.entry_inicio)
        time_layout.addRow("Fin:", self.entry_fin)
        lbl_hint = QLabel("Dejar en blanco para graficar todo.")
        lbl_hint.setStyleSheet("color: gray; font-size: 10px;")
        time_layout.addRow(lbl_hint)
        left_layout.addWidget(time_group)
        
        # Eje Y
        y_group = QGroupBox("Límites de Amplitud (µV)")
        y_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 8px; font-size: 11px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #00ffcc; font-weight: bold; }")
        y_layout = QFormLayout(y_group)
        y_layout.setContentsMargins(6, 8, 6, 6)
        y_layout.setSpacing(4)
        self.entry_ymin = QLineEdit()
        self.entry_ymax = QLineEdit()
        y_layout.addRow("Min:", self.entry_ymin)
        y_layout.addRow("Max:", self.entry_ymax)
        lbl_y_hint = QLabel("Dejar en blanco para autoescala.")
        lbl_y_hint.setStyleSheet("color: gray; font-size: 10px;")
        y_layout.addRow(lbl_y_hint)
        left_layout.addWidget(y_group)
        
        # Visualización
        viz_group = QGroupBox("Visualización")
        viz_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 8px; font-size: 11px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; color: #00ffcc; font-weight: bold; }")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setContentsMargins(6, 8, 6, 6)
        viz_layout.setSpacing(4)
        self.chk_fft = QCheckBox("Añadir Espectro de Frecuencias (FFT)")
        self.chk_oscuro = QCheckBox("Tema Oscuro (Fondo Negro)")
        self.chk_oscuro.setChecked(True)
        viz_layout.addWidget(self.chk_fft)
        viz_layout.addWidget(self.chk_oscuro)
        left_layout.addWidget(viz_group)
        
        # Botón para generar
        self.btn_generar_graficos = QPushButton("Generar Gráficos")
        self.btn_generar_graficos.setStyleSheet("""
            QPushButton {
                background-color: #00ffcc; color: black; font-weight: bold;
                font-size: 13px; padding: 8px; border-radius: 5px; margin-top: 4px;
            }
            QPushButton:hover { background-color: #00ccaa; }
        """)
        self.btn_generar_graficos.clicked.connect(self._emit_generate_request)
        left_layout.addWidget(self.btn_generar_graficos)
        
        left_layout.addStretch()
        
        # Scroll area para el panel izquierdo para soportar pantallas pequeñas
        self.left_scroll = QScrollArea()
        self.left_scroll.setWidgetResizable(True)
        self.left_scroll.setFixedWidth(280)
        self.left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.left_scroll.setFrameShape(QScrollArea.NoFrame)
        self.left_scroll.setStyleSheet("QScrollArea { background-color: transparent; border: none; }")
        self.left_scroll.setWidget(left_panel)
        h_layout.addWidget(self.left_scroll)

        # --- ÁREA DERECHA: Scroll para la imagen ---
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: #0c0c0c; border: 1px solid #222;")
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #0c0c0c; color: #666; font-size: 13px;")
        self.image_label.setCursor(Qt.PointingHandCursor)
        self.image_label.mouseDoubleClickEvent = lambda e: self.show_fullscreen()
        self.scroll_area.setWidget(self.image_label)
        
        h_layout.addWidget(self.scroll_area, stretch=1)
        self.layout.addLayout(h_layout)

    def _emit_generate_request(self):
        config = {
            "notch": self.chk_notch.isChecked(),
            "bandpass": self.chk_bandpass.isChecked(),
            "tipo_env": "ninguna",
            "start_time": None,
            "end_time": None,
            "graficar_fft": self.chk_fft.isChecked(),
            "tema_oscuro": self.chk_oscuro.isChecked()
        }
        if self.rb_hilbert.isChecked(): config["tipo_env"] = "hilbert"
        elif self.rb_rms.isChecked(): config["tipo_env"] = "rms"
        
        try:
            if self.entry_inicio.text().strip(): config["start_time"] = float(self.entry_inicio.text())
            if self.entry_fin.text().strip(): config["end_time"] = float(self.entry_fin.text())
        except ValueError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Los campos de tiempo deben ser numéricos.")
            return

        config["y_min"] = None
        config["y_max"] = None
        try:
            if self.entry_ymin.text().strip(): config["y_min"] = float(self.entry_ymin.text())
            if self.entry_ymax.text().strip(): config["y_max"] = float(self.entry_ymax.text())
        except ValueError:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Los campos del eje Y deben ser numéricos.")
            return
            
        self.request_generate_plots.emit(config)
        self.current_pixmap = None
        self.zoom_factor = 1.0
        self.fit_to_window = True

    def load_calibrated_plot(self, measurement_path):
        """Busca y carga el gráfico calibrado asociado a esta medición"""
        self.current_measurement_path = measurement_path
        self._reload_current_plot()
        
    def _reload_current_plot(self):
        if getattr(self, 'current_measurement_path', None) is None:
            return
            
        measurement_path = self.current_measurement_path
        tipo_plot = self.cmb_tipo_plot.currentIndex()
        
        self.lbl_status.setText(f"Buscando gráfico para: {os.path.basename(measurement_path)} ...")
        
        padre = os.path.basename(os.path.dirname(measurement_path))
        hijo = os.path.basename(measurement_path)
        rel_path = f"{padre}_{hijo}"
        nombre_limpio = rel_path.replace("/", "_").replace("\\", "_")
        
        if tipo_plot == 0:
            nombre_archivo = f"plot_calibrado_{nombre_limpio}.png"
            ruta_img = os.path.join(measurement_path, nombre_archivo)
            
            # Fallback a analisis_comparativos
            if not os.path.exists(ruta_img):
                root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                ruta_img = os.path.join(root_dir, "analisis_comparativos", nombre_archivo)
        else:
            nombre_archivo = "plot_paper_combined.png"
            ruta_img = os.path.join(measurement_path, nombre_archivo)
            
        if os.path.exists(ruta_img):
            pixmap = QPixmap(ruta_img)
            if not pixmap.isNull():
                self.current_pixmap = pixmap
                self.fit_to_view()
                
                # Leer información de músculos desde metadata.json
                meta_info = ""
                meta_p = os.path.join(measurement_path, "canal_0", "metadata.json")
                if not os.path.exists(meta_p):
                    meta_p = os.path.join(measurement_path, "metadata.json")
                if os.path.exists(meta_p):
                    try:
                        import json
                        with open(meta_p, 'r', encoding='utf-8') as fm:
                            m_data = json.load(fm)
                            musc_list = m_data.get("muscles", [])
                            if musc_list:
                                meta_info = f" | Músculos: {', '.join(musc_list)}"
                    except Exception:
                        pass
                        
                self.lbl_status.setText(f"Mostrando: {nombre_archivo}{meta_info}")
            else:
                self.current_pixmap = None
                self.image_label.clear()
                self.image_label.setText(f"Error: La imagen existe pero no se pudo cargar.\n({nombre_archivo})")
                self.lbl_status.setText("Error al cargar la imagen.")
                self.lbl_zoom_indicator.setText("-")
        else:
            self.current_pixmap = None
            self.image_label.clear()
            msg = "No se encontró ningún gráfico procesado.\nEjecuta 'Generar Gráficos' primero." if tipo_plot == 0 else "No se encontró el plot de paper.\nEjecuta 'Plot para Paper' primero."
            self.image_label.setText(msg)
            self.lbl_status.setText("Gráfico no encontrado.")
            self.lbl_zoom_indicator.setText("-")

    def fit_to_view(self):
        """Ajusta la imagen exactamente al tamaño disponible en el visor manteniendo la relación de aspecto."""
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        self.fit_to_window = True
        self.update_image_display()

    def zoom_fit(self):
        self.fit_to_view()

    def zoom_reset(self):
        """Restaura la escala al 100% (1:1 tamaño real de píxeles)."""
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        self.fit_to_window = False
        self.zoom_factor = 1.0
        self.update_image_display()

    def zoom_in(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        if self.fit_to_window:
            cur_w = self.image_label.pixmap().width() if self.image_label.pixmap() else self.current_pixmap.width()
            self.zoom_factor = cur_w / max(1, self.current_pixmap.width())
            self.fit_to_window = False
        self.zoom_factor = min(self.zoom_factor * 1.25, 8.0)
        self.update_image_display()

    def zoom_out(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        if self.fit_to_window:
            cur_w = self.image_label.pixmap().width() if self.image_label.pixmap() else self.current_pixmap.width()
            self.zoom_factor = cur_w / max(1, self.current_pixmap.width())
            self.fit_to_window = False
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.1)
        self.update_image_display()

    def update_image_display(self):
        """Renderiza la imagen en el QLabel según el modo de ajuste o el factor de zoom actual."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return
            
        if self.fit_to_window:
            vp_size = self.scroll_area.viewport().size()
            vp_w = max(50, vp_size.width() - 8)
            vp_h = max(50, vp_size.height() - 8)
            
            scaled_pixmap = self.current_pixmap.scaled(
                QSize(vp_w, vp_h), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())
            self.lbl_zoom_indicator.setText("Auto-Ajustado")
        else:
            target_w = max(1, int(self.current_pixmap.width() * self.zoom_factor))
            target_h = max(1, int(self.current_pixmap.height() * self.zoom_factor))
            target_size = QSize(target_w, target_h)
            scaled_pixmap = self.current_pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(target_size)
            self.lbl_zoom_indicator.setText(f"{int(self.zoom_factor * 100)}%")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if getattr(self, 'fit_to_window', True) and getattr(self, 'current_pixmap', None) and not self.current_pixmap.isNull():
            self.update_image_display()

    def show_fullscreen(self):
        """Abre un diálogo modal para visualizar la imagen a resolución completa y en pantalla completa."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Ñandú LSD - Visor de Gráficos de Alta Resolución")
        dialog.setStyleSheet("background-color: #050505; color: #fff;")
        dlg_lyt = QVBoxLayout(dialog)
        dlg_lyt.setContentsMargins(5, 5, 5, 5)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none; background: #000;")
        
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setPixmap(self.current_pixmap)
        scroll.setWidget(lbl)
        dlg_lyt.addWidget(scroll)
        
        screen = self.screen().availableGeometry()
        max_w = int(screen.width() * 0.95)
        max_h = int(screen.height() * 0.95)
        dialog.resize(min(self.current_pixmap.width() + 40, max_w), min(self.current_pixmap.height() + 40, max_h))
        dialog.exec()
