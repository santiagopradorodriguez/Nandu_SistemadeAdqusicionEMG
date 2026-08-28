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
from PySide6.QtCore import Qt, Signal, QSize, QEvent, QTimer
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
        self.fit_mode = "width"  # Modos: 'width' (ajuste panorámico), 'window' (ajuste completo), 'manual'

        # Debounce timer para evitar bucles de redimensionamiento
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._on_debounced_resize)

        # Barra superior Fila 1: Información de medición y selector de plot
        top_bar_1 = QHBoxLayout()
        top_bar_1.setSpacing(6)
        
        self.lbl_status = QLabel("Listo. Seleccione una medición en el Gestor de Sesiones.")
        self.lbl_status.setStyleSheet("color: #00ffcc; font-family: monospace; font-size: 11px; font-weight: bold;")
        self.lbl_status.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        top_bar_1.addWidget(self.lbl_status, stretch=1)
        
        btn_style = """
            QPushButton {
                background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc;
                padding: 4px 10px; font-weight: bold; border-radius: 4px; margin: 1px; font-size: 11px;
            }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
        """
        self.btn_plot_3m = QPushButton("Plot para Paper (3 Músculos)")
        self.btn_plot_3m.setStyleSheet(btn_style)
        top_bar_1.addWidget(self.btn_plot_3m)
        
        # Selector de tipo de plot
        self.cmb_tipo_plot = QComboBox()
        self.cmb_tipo_plot.addItems([
            "Vista: Plot Calibrado (Auto)", 
            "Vista: Plot Paper (Completo)"
        ])
        self.cmb_tipo_plot.setStyleSheet("background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc; padding: 3px 6px; font-weight: bold; border-radius: 4px; font-size: 11px;")
        self.cmb_tipo_plot.currentIndexChanged.connect(self._reload_current_plot)
        top_bar_1.addWidget(self.cmb_tipo_plot)
        self.layout.addLayout(top_bar_1)

        # Barra superior Fila 2: Controles de visualización y zoom
        top_bar_2 = QHBoxLayout()
        top_bar_2.setSpacing(4)

        # Selector de suavizado
        lbl_smooth = QLabel("Suavizado (ms):")
        lbl_smooth.setStyleSheet("font-size: 11px; color: #aaa;")
        top_bar_2.addWidget(lbl_smooth)
        self.inp_smooth = QLineEdit("250")
        self.inp_smooth.setMaximumWidth(42)
        self.inp_smooth.setStyleSheet("background-color: #222; color: white; border: 1px solid gray; padding: 2px; font-size: 11px;")
        top_bar_2.addWidget(self.inp_smooth)
        
        top_bar_2.addSpacing(10)

        # Controles de Zoom y Ajuste
        self.btn_zoom_fit_w = QPushButton(" Ajustar Ancho")
        self.btn_zoom_fit_w.setToolTip("Ajustar al ancho disponible (ideal para ver señales panorámicas)")
        self.btn_zoom_fit_h = QPushButton(" Ajustar Alto")
        self.btn_zoom_fit_h.setToolTip("Ajustar toda la imagen a la ventana sin scroll")
        self.btn_zoom_reset = QPushButton(" 1:1")
        self.btn_zoom_reset.setToolTip("Tamaño real 100%")
        self.btn_zoom_out = QPushButton(" -")
        self.btn_zoom_out.setToolTip("Alejar zoom")
        self.btn_zoom_in = QPushButton(" +")
        self.btn_zoom_in.setToolTip("Acercar zoom")
        self.btn_fullscreen = QPushButton(" Pantalla Completa")
        self.btn_fullscreen.setToolTip("Ver en alta resolución a pantalla completa")
        
        for btn in [self.btn_zoom_fit_w, self.btn_zoom_fit_h, self.btn_zoom_reset, self.btn_zoom_out, self.btn_zoom_in, self.btn_fullscreen]:
            btn.setStyleSheet(btn_style)
            top_bar_2.addWidget(btn)

        self.btn_zoom_fit_w.clicked.connect(self.fit_to_width)
        self.btn_zoom_fit_h.clicked.connect(self.fit_to_view)
        self.btn_zoom_reset.clicked.connect(self.zoom_reset)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        self.btn_fullscreen.clicked.connect(self.show_fullscreen)
        
        self.lbl_zoom_indicator = QLabel("Ajustado al Ancho")
        self.lbl_zoom_indicator.setStyleSheet("color: #888; font-size: 11px; padding-left: 6px;")
        top_bar_2.addWidget(self.lbl_zoom_indicator)
        top_bar_2.addStretch()
        
        self.layout.addLayout(top_bar_2)

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
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.scroll_area.setStyleSheet("background-color: #0c0c0c; border: 1px solid #222;")
        self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll_area.setMinimumSize(150, 150)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #0c0c0c; color: #666; font-size: 13px;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.image_label.setScaledContents(False)
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
                self.update_image_display()
                
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

    def fit_to_width(self):
        """Ajusta la imagen al ancho completo disponible con scroll vertical libre."""
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        self.fit_mode = "width"
        self.update_image_display()

    def fit_to_view(self):
        """Ajusta la imagen exactamente al tamaño disponible en el visor manteniendo la relación de aspecto."""
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        self.fit_mode = "window"
        self.update_image_display()

    def zoom_fit(self):
        self.fit_to_width()

    def zoom_reset(self):
        """Restaura la escala al 100% (1:1 tamaño real de píxeles)."""
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        self.fit_mode = "manual"
        self.zoom_factor = 1.0
        self.update_image_display()

    def zoom_in(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        if self.fit_mode != "manual":
            cur_w = self.image_label.pixmap().width() if self.image_label.pixmap() else self.current_pixmap.width()
            self.zoom_factor = cur_w / max(1, self.current_pixmap.width())
            self.fit_mode = "manual"
        self.zoom_factor = min(self.zoom_factor * 1.25, 8.0)
        self.update_image_display()

    def zoom_out(self):
        if self.current_pixmap is None or self.current_pixmap.isNull():
            return
        if self.fit_mode != "manual":
            cur_w = self.image_label.pixmap().width() if self.image_label.pixmap() else self.current_pixmap.width()
            self.zoom_factor = cur_w / max(1, self.current_pixmap.width())
            self.fit_mode = "manual"
        self.zoom_factor = max(self.zoom_factor / 1.25, 0.05)
        self.update_image_display()

    def update_image_display(self):
        """Renderiza la imagen en el QLabel según el modo de ajuste (ancho, ventana completa o zoom manual)."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return
            
        if self.fit_mode == "width":
            self.scroll_area.setWidgetResizable(False)
            self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
            vp_w = max(50, self.scroll_area.viewport().width() - 16)
            ratio = vp_w / max(1, self.current_pixmap.width())
            target_h = max(50, int(self.current_pixmap.height() * ratio))
            target_size = QSize(vp_w, target_h)
            
            scaled_pixmap = self.current_pixmap.scaled(
                target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(target_size)
            self.lbl_zoom_indicator.setText("Ajustado al Ancho")
            
        elif self.fit_mode == "window":
            self.scroll_area.setWidgetResizable(True)
            self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
            
            vp_size = self.scroll_area.viewport().size()
            vp_w = max(50, vp_size.width())
            vp_h = max(50, vp_size.height())
            
            scaled_pixmap = self.current_pixmap.scaled(
                QSize(vp_w, vp_h), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.lbl_zoom_indicator.setText("Ajustado a Ventana")
            
        else: # modo manual
            self.scroll_area.setWidgetResizable(False)
            self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            
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
        if getattr(self, 'fit_mode', 'width') in ('width', 'window') and getattr(self, 'current_pixmap', None) and not self.current_pixmap.isNull():
            self._resize_timer.start(50)

    def _on_debounced_resize(self):
        if getattr(self, 'fit_mode', 'width') in ('width', 'window') and self.current_pixmap and not self.current_pixmap.isNull():
            self.update_image_display()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def show_fullscreen(self):
        """Abre un diálogo modal para visualizar la imagen a resolución completa y en pantalla completa."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("Ñandú LSD - Visor de Gráficos de Alta Resolución")
        dialog.setStyleSheet("background-color: #050505; color: #fff;")
        dlg_lyt = QVBoxLayout(dialog)
        dlg_lyt.setContentsMargins(8, 8, 8, 8)
        
        # Toolbar en la ventana modal
        tb = QHBoxLayout()
        btn_in = QPushButton("+ Zoom")
        btn_out = QPushButton("- Zoom")
        btn_fit = QPushButton("Ajustar")
        btn_100 = QPushButton("100% (1:1)")
        btn_close = QPushButton("Cerrar")
        
        btn_style = """
            QPushButton {
                background-color: #151515; color: #00ffcc; border: 1px solid #333;
                padding: 4px 12px; font-size: 12px; font-weight: bold; border-radius: 3px;
            }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
        """
        for b in [btn_in, btn_out, btn_fit, btn_100]:
            b.setStyleSheet(btn_style)
            tb.addWidget(b)
            
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #331111; color: #ff5555; border: 1px solid #ff3333;
                padding: 4px 12px; font-size: 12px; font-weight: bold; border-radius: 3px;
            }
            QPushButton:hover { background-color: #ff3333; color: #fff; }
        """)
        btn_close.clicked.connect(dialog.accept)
        tb.addStretch()
        dlg_lyt.addLayout(tb)
        
        # Scroll area con ImageLabel responsivo
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: 1px solid #222; background: #000;")
        scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lbl.setStyleSheet("background-color: #000;")
        scroll.setWidget(lbl)
        dlg_lyt.addWidget(scroll, stretch=1)
        
        # Estado interno de zoom del modal
        modal_state = {"fit": True, "scale": 1.0}
        
        def update_modal_display():
            if not self.current_pixmap: return
            if modal_state["fit"]:
                scroll.setWidgetResizable(True)
                lbl.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                vp_s = scroll.viewport().size()
                sc = self.current_pixmap.scaled(vp_s, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(sc)
            else:
                scroll.setWidgetResizable(False)
                lbl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                tw = max(10, int(self.current_pixmap.width() * modal_state["scale"]))
                th = max(10, int(self.current_pixmap.height() * modal_state["scale"]))
                sc = self.current_pixmap.scaled(QSize(tw, th), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl.setPixmap(sc)
                lbl.resize(QSize(tw, th))
                
        def modal_zoom_in():
            modal_state["fit"] = False
            modal_state["scale"] = min(modal_state["scale"] * 1.25, 8.0)
            update_modal_display()
            
        def modal_zoom_out():
            modal_state["fit"] = False
            modal_state["scale"] = max(modal_state["scale"] / 1.25, 0.1)
            update_modal_display()
            
        def modal_zoom_fit():
            modal_state["fit"] = True
            update_modal_display()
            
        def modal_zoom_100():
            modal_state["fit"] = False
            modal_state["scale"] = 1.0
            update_modal_display()
            
        btn_in.clicked.connect(modal_zoom_in)
        btn_out.clicked.connect(modal_zoom_out)
        btn_fit.clicked.connect(modal_zoom_fit)
        btn_100.clicked.connect(modal_zoom_100)
        
        orig_resize = dialog.resizeEvent
        def modal_resize(e):
            orig_resize(e)
            if modal_state["fit"]:
                QTimer.singleShot(10, update_modal_display)
        dialog.resizeEvent = modal_resize
        
        dialog.showMaximized()
        QTimer.singleShot(60, update_modal_display)
        dialog.exec()
