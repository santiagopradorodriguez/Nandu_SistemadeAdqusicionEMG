"""
Proyecto NANDU LSD - Sistema de Adquisición sEMG
Herramienta de Mapeo Espacial 3D y Triangulación Métrica de Electrodos sobre Fotos de Candela
Reportes y Cuaderno de Experimentos IA - Subetapa de Mapeo Biomecánico
"""

import os
import sys
import json
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

from PySide6 import QtCore, QtGui, QtWidgets

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from EMG_desarrollo.acquisition.calibracion_espacial_electrodos import (
    AntigravityAgent,
    INDICE_NARIZ_PUNTA,
    INDICE_MENTON_GNATHION,
    INDICE_COMISURA_IZQ,
    INDICE_COMISURA_DER,
    INDICE_TRAGO_OREJA_IZQ,
    INDICE_TRAGO_OREJA_DER,
)


class ClickablePhotoLabel(QtWidgets.QWidget):
    """
    Visor desacoplado de imagenes de alta resolucion.
    Renderiza la foto escalada y centrada sin alterar la geometria del layout (evita bucles de resize).
    Emite coordenadas (x, y) en el espacio original de pixeles de la foto.
    """

    pointClicked = QtCore.Signal(float, float)
    rightClicked = QtCore.Signal(float, float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._img_w: int = 1
        self._img_h: int = 1
        self._target_rect: QtCore.QRect = QtCore.QRect()
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(480, 360)

    def set_image_dimensions(self, w: int, h: int) -> None:
        self._img_w = max(w, 1)
        self._img_h = max(h, 1)

    def set_display_pixmap(self, pix: QtGui.QPixmap, orig_w: int, orig_h: int) -> None:
        self._pixmap = pix
        self._img_w = max(orig_w, 1)
        self._img_h = max(orig_h, 1)
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor("#0d0d0d"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QtGui.QColor("#555555"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Seleccione una foto de montaje")
            painter.end()
            return

        w_disp = self.width()
        h_disp = self.height()
        aspect_img = self._img_w / float(self._img_h)
        aspect_widget = w_disp / float(h_disp) if h_disp > 0 else 1.0

        if aspect_widget > aspect_img:
            render_h = h_disp
            render_w = int(round(render_h * aspect_img))
            render_x = (w_disp - render_w) // 2
            render_y = 0
        else:
            render_w = w_disp
            render_h = int(round(render_w / aspect_img))
            render_x = 0
            render_y = (h_disp - render_h) // 2

        self._target_rect = QtCore.QRect(render_x, render_y, render_w, render_h)
        painter.drawPixmap(self._target_rect, self._pixmap)
        painter.end()

    def _map_widget_to_image(self, pos: QtCore.QPoint) -> Tuple[float, float]:
        if self._target_rect.isEmpty() or self._target_rect.width() <= 0 or self._target_rect.height() <= 0:
            return float(pos.x()), float(pos.y())

        rx = (pos.x() - self._target_rect.x()) / float(self._target_rect.width())
        ry = (pos.y() - self._target_rect.y()) / float(self._target_rect.height())

        clamped_x = float(np.clip(rx, 0.0, 1.0) * self._img_w)
        clamped_y = float(np.clip(ry, 0.0, 1.0) * self._img_h)
        return clamped_x, clamped_y

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            ix, iy = self._map_widget_to_image(event.position().toPoint())
            self.pointClicked.emit(ix, iy)
        elif event.button() == QtCore.Qt.RightButton:
            ix, iy = self._map_widget_to_image(event.position().toPoint())
            self.rightClicked.emit(ix, iy)
        super().mousePressEvent(event)


class CandelaSpatialMapperDialog(QtWidgets.QDialog):
    """
    Dialogo interactivo de mapeo espacial y triangulacion sobre fotos de Candela.
    Permite navegar imagenes con tira de miniaturas visuales, seleccionar canales libremente,
    y medir distancias anatomicas e IED en milimetros sin requerir electrodo de tierra.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.agent = AntigravityAgent(refine_landmarks=True)
        self.base_drive = os.path.join(REPO_ROOT, "Candela_Organizado_Drive")

        self.current_img: Optional[np.ndarray] = None
        self.current_fids: Optional[Dict[str, Any]] = None
        self.current_path: str = ""
        self.current_photo_name: str = ""

        # Mapeo de pares por canal: "Ch0", "Ch1", "Ch2"
        # Cada entrada contiene las coordenadas, la foto de origen y las metricas
        self._channels_data: Dict[str, Dict[str, Any]] = {}
        self._pending_pos_pole: Optional[Tuple[float, float]] = None

        # Cache de fiduciarias faciales por foto para rapidez
        self._fids_cache: Dict[str, Optional[Dict[str, Any]]] = {}

        self.setWindowTitle("Mapeo Espacial 3D y Triangulacion de Electrodos - Candela")
        self.resize(1340, 860)
        self._apply_dark_theme()
        self._build_ui()
        self._populate_sessions()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #16181a;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }
            QGroupBox {
                background-color: #202428;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00ffcc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #e6edf3;
            }
            QPushButton {
                background-color: #282f37;
                color: #ffffff;
                border: 1px solid #00ffcc;
                border-radius: 5px;
                padding: 6px 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004d40;
                border-color: #33ffdd;
            }
            QComboBox {
                background-color: #21262d;
                color: #00ffcc;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QListWidget {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                border-radius: 4px;
                font-size: 11px;
            }
            QTableWidget {
                background-color: #0d1117;
                color: #e6edf3;
                border: 1px solid #30363d;
                gridline-color: #21262d;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #21262d;
                color: #00ffcc;
                padding: 4px;
                border: 1px solid #30363d;
                font-weight: bold;
            }
            """
        )

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # ================= COLUMNA IZQUIERDA: FOTO Y GALERIA =================
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(6)

        # Barra de navegacion superior
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Sesion:"))
        self.combo_sessions = QtWidgets.QComboBox()
        self.combo_sessions.currentIndexChanged.connect(self._on_session_changed)
        top_bar.addWidget(self.combo_sessions, stretch=3)

        self.btn_prev_photo = QtWidgets.QPushButton("< Ant")
        self.btn_prev_photo.setFixedWidth(50)
        self.btn_prev_photo.clicked.connect(self._prev_photo)
        top_bar.addWidget(self.btn_prev_photo)

        self.combo_photos = QtWidgets.QComboBox()
        self.combo_photos.currentIndexChanged.connect(self._on_photo_changed)
        top_bar.addWidget(self.combo_photos, stretch=3)

        self.btn_next_photo = QtWidgets.QPushButton("Sig >")
        self.btn_next_photo.setFixedWidth(50)
        self.btn_next_photo.clicked.connect(self._next_photo)
        top_bar.addWidget(self.btn_next_photo)

        left_col.addLayout(top_bar)

        # Visor interactivo principal
        self.photo_label = ClickablePhotoLabel(self)
        self.photo_label.setStyleSheet("background-color: #0d0d0d; border: 2px solid #00ffcc; border-radius: 6px;")
        self.photo_label.pointClicked.connect(self._on_photo_left_clicked)
        self.photo_label.rightClicked.connect(self._on_photo_right_clicked)
        left_col.addWidget(self.photo_label, stretch=4)

        # Barra de estado de la imagen
        self.lbl_photo_status = QtWidgets.QLabel("Cargando imagen...")
        self.lbl_photo_status.setStyleSheet("color: #b0bec5; font-size: 11px;")
        left_col.addWidget(self.lbl_photo_status)

        # Galeria visual de miniaturas
        thumb_label = QtWidgets.QLabel("Galeria Visual de Fotos de Montaje - Clic en cualquier miniatura para ver:")
        thumb_label.setStyleSheet("color: #8b949e; font-size: 10px; font-weight: bold;")
        left_col.addWidget(thumb_label)

        self.thumb_list = QtWidgets.QListWidget()
        self.thumb_list.setViewMode(QtWidgets.QListView.IconMode)
        self.thumb_list.setIconSize(QtCore.QSize(100, 75))
        self.thumb_list.setGridSize(QtCore.QSize(120, 105))
        self.thumb_list.setMovement(QtWidgets.QListView.Static)
        self.thumb_list.setResizeMode(QtWidgets.QListView.Adjust)
        self.thumb_list.setFixedHeight(115)
        self.thumb_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.thumb_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.thumb_list.setStyleSheet(
            """
            QListWidget {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                color: #8b949e;
                border: 1px solid #21262d;
                border-radius: 4px;
                padding: 2px;
                font-size: 10px;
            }
            QListWidget::item:selected {
                background-color: #00332c;
                border: 2px solid #00ffcc;
                color: #ffffff;
                font-weight: bold;
            }
            """
        )
        self.thumb_list.currentRowChanged.connect(self._on_thumbnail_selected)
        left_col.addWidget(self.thumb_list)

        main_layout.addLayout(left_col, stretch=3)

        # ================= COLUMNA DERECHA: PANELES Y MEDICIONES =================
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(10)

        # 1. Dimensiones de Electrodos y Referencia Metrica
        param_group = QtWidgets.QGroupBox("1: Dimensiones de Electrodos y Referencia Metrica")
        param_layout = QtWidgets.QGridLayout(param_group)

        param_layout.addWidget(QtWidgets.QLabel("Diametro Gel - Parche:"), 0, 0)
        self.spin_gel_diam = QtWidgets.QDoubleSpinBox()
        self.spin_gel_diam.setRange(2.0, 40.0)
        self.spin_gel_diam.setValue(10.0)
        self.spin_gel_diam.setSuffix(" mm")
        self.spin_gel_diam.valueChanged.connect(self._update_display)
        param_layout.addWidget(self.spin_gel_diam, 0, 1)

        param_layout.addWidget(QtWidgets.QLabel("IED Nominal Esperada:"), 1, 0)
        self.spin_ied_nom = QtWidgets.QDoubleSpinBox()
        self.spin_ied_nom.setRange(4.0, 50.0)
        self.spin_ied_nom.setValue(20.0)
        self.spin_ied_nom.setSuffix(" mm")
        self.spin_ied_nom.valueChanged.connect(self._update_display)
        param_layout.addWidget(self.spin_ied_nom, 1, 1)

        param_layout.addWidget(QtWidgets.QLabel("D_IP Sujeto - Candela:"), 2, 0)
        self.spin_dip = QtWidgets.QDoubleSpinBox()
        self.spin_dip.setRange(45.0, 75.0)
        self.spin_dip.setValue(61.0)
        self.spin_dip.setSuffix(" mm")
        self.spin_dip.valueChanged.connect(self._update_display)
        param_layout.addWidget(self.spin_dip, 2, 1)

        self.chk_show_mesh = QtWidgets.QCheckBox("Mostrar Malla 3D MediaPipe en Piel")
        self.chk_show_mesh.setChecked(True)
        self.chk_show_mesh.toggled.connect(self._update_display)
        param_layout.addWidget(self.chk_show_mesh, 3, 0, 1, 2)

        right_col.addWidget(param_group)

        # 2. Marcado de Canales Bipolares
        mark_group = QtWidgets.QGroupBox("2: Marcado de Canales Bipolares - Polo Positivo y Negativo")
        mark_layout = QtWidgets.QVBoxLayout(mark_group)

        self.lbl_instruction = QtWidgets.QLabel("Seleccione el canal y haga 2 clics en la foto: Polo + y Polo -")
        self.lbl_instruction.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 11px;")
        mark_layout.addWidget(self.lbl_instruction)

        chan_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_ch0 = QtWidgets.QPushButton("Ch0: Digastrico")
        self.btn_ch0.setCheckable(True)
        self.btn_ch0.setChecked(True)
        self.btn_ch0.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; font-weight: bold; }")

        self.btn_ch1 = QtWidgets.QPushButton("Ch1: Cigomatico")
        self.btn_ch1.setCheckable(True)
        self.btn_ch1.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; font-weight: bold; }")

        self.btn_ch2 = QtWidgets.QPushButton("Ch2: Orbicular")
        self.btn_ch2.setCheckable(True)
        self.btn_ch2.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; font-weight: bold; }")

        self.chan_btn_group = QtWidgets.QButtonGroup(self)
        self.chan_btn_group.addButton(self.btn_ch0, 0)
        self.chan_btn_group.addButton(self.btn_ch1, 1)
        self.chan_btn_group.addButton(self.btn_ch2, 2)
        self.chan_btn_group.buttonClicked.connect(self._on_channel_selected)

        chan_btn_layout.addWidget(self.btn_ch0)
        chan_btn_layout.addWidget(self.btn_ch1)
        chan_btn_layout.addWidget(self.btn_ch2)
        mark_layout.addLayout(chan_btn_layout)

        # Selector de musculo especifico para Ch1
        m_row = QtWidgets.QHBoxLayout()
        m_row.addWidget(QtWidgets.QLabel("Musculo Asignado a Ch1:"))
        self.combo_ch1_target = QtWidgets.QComboBox()
        self.combo_ch1_target.addItems([
            "Zygomaticus Major - Cigomatico",
            "Modiolus - Modiolo",
            "Masseter - Masetero",
            "Risorius - Risorio",
            "Mentalis - Mentoniano",
        ])
        self.combo_ch1_target.currentIndexChanged.connect(self._on_ch1_muscle_changed)
        m_row.addWidget(self.combo_ch1_target, stretch=1)
        mark_layout.addLayout(m_row)

        action_row = QtWidgets.QHBoxLayout()
        self.btn_clear_curr = QtWidgets.QPushButton("Borrar Canal Seleccionado")
        self.btn_clear_curr.clicked.connect(self._clear_selected_channel)
        action_row.addWidget(self.btn_clear_curr)

        self.btn_clear_all = QtWidgets.QPushButton("Limpiar Todos")
        self.btn_clear_all.clicked.connect(self._clear_all_points)
        action_row.addWidget(self.btn_clear_all)
        mark_layout.addLayout(action_row)

        self.list_pairs = QtWidgets.QListWidget()
        self.list_pairs.setMaximumHeight(80)
        mark_layout.addWidget(self.list_pairs)

        right_col.addWidget(mark_group)

        # 3. Distancias Anatomicas Calculadas
        dist_group = QtWidgets.QGroupBox("3: Distancias Anatomicas Calculadas - en mm")
        dist_layout = QtWidgets.QVBoxLayout(dist_group)

        self.table_distances = QtWidgets.QTableWidget(0, 8)
        self.table_distances.setHorizontalHeaderLabels([
            "Canal", "Foto", "Polo", "Nariz - mm", "Menton - mm", "Comisura - mm", "IED - mm", "Angulo - deg"
        ])
        self.table_distances.horizontalHeader().setStretchLastSection(True)
        dist_layout.addWidget(self.table_distances)

        right_col.addWidget(dist_group)

        # 4. Boton Guardar y Exportar
        self.btn_export = QtWidgets.QPushButton("Guardar Calibracion y Exportar Imagen 3D")
        self.btn_export.setStyleSheet(
            "QPushButton { background-color: #00c853; color: #0a0a0a; border: 2px solid #00ffcc; "
            "padding: 10px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00e676; }"
        )
        self.btn_export.clicked.connect(self._export_calibration)
        right_col.addWidget(self.btn_export)

        main_layout.addLayout(right_col, stretch=2)

    def _prev_photo(self) -> None:
        idx = self.combo_photos.currentIndex()
        if idx > 0:
            self.combo_photos.setCurrentIndex(idx - 1)

    def _next_photo(self) -> None:
        idx = self.combo_photos.currentIndex()
        if idx < self.combo_photos.count() - 1:
            self.combo_photos.setCurrentIndex(idx + 1)

    def _populate_sessions(self) -> None:
        if not os.path.exists(self.base_drive):
            self.lbl_photo_status.setText(f"Carpeta no encontrada: {self.base_drive}")
            return

        sessions = sorted([
            d for d in os.listdir(self.base_drive)
            if os.path.isdir(os.path.join(self.base_drive, d)) and "Candela" in d
        ])

        self.combo_sessions.clear()
        for s in sessions:
            self.combo_sessions.addItem(s)

        # Seleccionar 2026-09-15 por defecto si existe
        idx_15 = self.combo_sessions.findText("2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela", QtCore.Qt.MatchContains)
        if idx_15 >= 0:
            self.combo_sessions.setCurrentIndex(idx_15)

    def _on_session_changed(self) -> None:
        sess_name = self.combo_sessions.currentText()
        if not sess_name:
            return

        montaje_dir = os.path.join(self.base_drive, sess_name, "fotos_montaje")
        photos: List[str] = []
        if os.path.exists(montaje_dir):
            photos = sorted([
                f for f in os.listdir(montaje_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        # Reiniciar marcas de sesion y cache
        self._channels_data.clear()
        self._pending_pos_pole = None
        self._fids_cache.clear()

        # Ajustar dimensiones de gel segun sesion (recortados vs estandar)
        if "25" in sess_name or "29" in sess_name or "23" in sess_name:
            self.spin_gel_diam.setValue(6.0)
            self.spin_ied_nom.setValue(11.0)
        else:
            self.spin_gel_diam.setValue(10.0)
            self.spin_ied_nom.setValue(20.0)

        # Cargar calibracion previa si ya existe
        prev_json = os.path.join(self.base_drive, sess_name, "calibracion_espacial_reconstruida", "calibracion_espacial.json")
        if os.path.exists(prev_json):
            try:
                with open(prev_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    meta = data.get("metadatos_calibracion_espacial", {})
                    # Si habia pares marcados previamente, recuperarlos
                    pares = meta.get("pares_diferenciales", [])
                    for p in pares:
                        c_tag = f"Ch{p.get('canal_index', 0)}"
                        self._channels_data[c_tag] = {
                            "canal": c_tag,
                            "musculo": p.get("musculo", c_tag),
                            "foto": meta.get("foto_origen", ""),
                            "polo_positivo": tuple(p.get("polo_positivo_2d_px", [0, 0])),
                            "polo_negativo": tuple(p.get("polo_negativo_2d_px", [0, 0])),
                            "ied_px": float(p.get("ied_px", 0.0)),
                            "ied_mm": float(p.get("ied_mm", 0.0)),
                            "angulo_deg": float(p.get("angulo_vector_grados", 0.0)),
                        }
            except Exception:
                pass

        # Poblar combo y miniaturas
        self.combo_photos.blockSignals(True)
        self.combo_photos.clear()
        for p in photos:
            self.combo_photos.addItem(p)
        self.combo_photos.blockSignals(False)

        self._populate_thumbnail_strip(photos, montaje_dir)

        # Seleccionar foto frontal preferentemente
        idx_front = -1
        for i, p in enumerate(photos):
            txt = p.lower()
            if "frontal" in txt or "vista_general" in txt or "camara" in txt:
                idx_front = i
                break
        if idx_front < 0 and photos:
            idx_front = 0

        if idx_front >= 0 and idx_front < self.combo_photos.count():
            self.combo_photos.setCurrentIndex(idx_front)
            self.thumb_list.setCurrentRow(idx_front)

        self._refresh_list_and_table()

    def _populate_thumbnail_strip(self, photos: List[str], montaje_dir: str) -> None:
        self.thumb_list.blockSignals(True)
        self.thumb_list.clear()

        for f in photos:
            img_p = os.path.join(montaje_dir, f)
            img = cv2.imread(img_p)
            if img is not None:
                # Extraer o verificar presencia de rostro
                fids = self.agent.extraer_malla_facial_3d(img)
                self._fids_cache[f] = fids
                has_face = fids is not None

                thumb_bgr = cv2.resize(img, (100, 75))
                thumb_rgb = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = thumb_rgb.shape
                qimg = QtGui.QImage(thumb_rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
                pix = QtGui.QPixmap.fromImage(qimg)

                # Etiqueta corta de miniatura
                tag = "[Cara 3D]" if has_face else "[Detalle]"
                short_name = f if len(f) <= 15 else f[:12] + "..."
                label_txt = f"{short_name}\n{tag}"

                item = QtWidgets.QListWidgetItem(QtGui.QIcon(pix), label_txt)
                item.setSizeHint(QtCore.QSize(115, 100))
                item.setToolTip(f"{f} - {tag}")
                self.thumb_list.addItem(item)
            else:
                item = QtWidgets.QListWidgetItem(f[:12])
                self.thumb_list.addItem(item)

        self.thumb_list.blockSignals(False)

    def _on_thumbnail_selected(self, row: int) -> None:
        if 0 <= row < self.combo_photos.count():
            if self.combo_photos.currentIndex() != row:
                self.combo_photos.setCurrentIndex(row)

    def _on_photo_changed(self) -> None:
        sess_name = self.combo_sessions.currentText()
        photo_name = self.combo_photos.currentText()
        if not sess_name or not photo_name:
            return

        # Sincronizar tira de miniaturas
        curr_row = self.combo_photos.currentIndex()
        if self.thumb_list.currentRow() != curr_row and curr_row >= 0:
            self.thumb_list.blockSignals(True)
            self.thumb_list.setCurrentRow(curr_row)
            self.thumb_list.blockSignals(False)

        img_path = os.path.join(self.base_drive, sess_name, "fotos_montaje", photo_name)
        if not os.path.exists(img_path):
            return

        self.current_path = img_path
        self.current_photo_name = photo_name
        self.current_img = cv2.imread(img_path)
        if self.current_img is None:
            self.lbl_photo_status.setText("Error al leer imagen.")
            return

        # Obtener fiduciarias desde cache o calcular
        if photo_name in self._fids_cache:
            self.current_fids = self._fids_cache[photo_name]
        else:
            self.current_fids = self.agent.extraer_malla_facial_3d(self.current_img)
            self._fids_cache[photo_name] = self.current_fids

        has_face = self.current_fids is not None

        if has_face and self.current_fids is not None:
            dip_px = float(self.current_fids.get("distancia_interpupilar_px", 1.0))
            escala = float(self.spin_dip.value() / dip_px) if dip_px > 0 else 0.25
            self.lbl_photo_status.setText(
                f"[Malla Facial 3D Detectada] Res: {self.current_img.shape[1]}x{self.current_img.shape[0]} | "
                f"D_IP: {dip_px:.1f} px | Escala: {escala:.3f} mm/px"
            )
            self.lbl_photo_status.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_photo_status.setText(
                f"[Aviso] No se detecto rostro frontal en esta toma (Res: {self.current_img.shape[1]}x{self.current_img.shape[0]}). Puede marcar manualmente sobre el musculo visible."
            )
            self.lbl_photo_status.setStyleSheet("color: #ffb300; font-size: 11px;")

        self._pending_pos_pole = None
        self._update_display()

    def _get_current_channel_info(self) -> Tuple[str, str]:
        if self.btn_ch1.isChecked():
            raw_m = self.combo_ch1_target.currentText()
            m_clean = raw_m.split(" - ")[0]
            return "Ch1", m_clean
        elif self.btn_ch2.isChecked():
            return "Ch2", "Orbicularis Oris"
        return "Ch0", "Anterior Belly Digastric"

    def _on_ch1_muscle_changed(self) -> None:
        if "Ch1" in self._channels_data:
            _, m_clean = self._get_current_channel_info()
            self._channels_data["Ch1"]["musculo"] = m_clean
            self._refresh_list_and_table()
            self._update_display()

    def _on_channel_selected(self, _button: QtWidgets.QAbstractButton) -> None:
        self._pending_pos_pole = None
        c_tag, m_name = self._get_current_channel_info()
        self.lbl_instruction.setText(f"Marcando {c_tag} - {m_name}: Clic 1 = Polo +, Clic 2 = Polo -")
        self.lbl_instruction.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 11px;")
        self._update_display()

    def _on_photo_left_clicked(self, ix: float, iy: float) -> None:
        c_tag, m_name = self._get_current_channel_info()

        if self._pending_pos_pole is None:
            self._pending_pos_pole = (ix, iy)
            self.lbl_instruction.setText(f"{c_tag}: Polo + marcado en ({ix:.0f}, {iy:.0f}). Clic en Polo -.")
            self.lbl_instruction.setStyleSheet("color: #33ccff; font-weight: bold;")
        else:
            pos_pt = self._pending_pos_pole
            neg_pt = (ix, iy)
            self._pending_pos_pole = None

            ied_px = float(np.hypot(neg_pt[0] - pos_pt[0], neg_pt[1] - pos_pt[1]))
            escala = self._get_scale_mm_per_px()
            ied_mm = ied_px * escala
            ang_deg = float(np.degrees(np.arctan2(neg_pt[1] - pos_pt[1], neg_pt[0] - pos_pt[0])))

            pair = {
                "canal": c_tag,
                "musculo": m_name,
                "foto": self.current_photo_name,
                "polo_positivo": pos_pt,
                "polo_negativo": neg_pt,
                "ied_px": ied_px,
                "ied_mm": ied_mm,
                "angulo_deg": ang_deg,
            }
            self._channels_data[c_tag] = pair
            self.lbl_instruction.setText(f"[OK] {c_tag} marcado: IED = {ied_mm:.1f} mm. Puede marcar otro canal o Guardar.")
            self.lbl_instruction.setStyleSheet("color: #00ffcc; font-weight: bold;")

            # Auto-avanzar al siguiente canal si esta vacio
            if c_tag == "Ch0" and "Ch1" not in self._channels_data:
                self.btn_ch1.setChecked(True)
            elif c_tag == "Ch1" and "Ch2" not in self._channels_data:
                self.btn_ch2.setChecked(True)

        self._refresh_list_and_table()
        self._update_display()

    def _on_photo_right_clicked(self, _ix: float, _iy: float) -> None:
        if self._pending_pos_pole is not None:
            self._pending_pos_pole = None
        else:
            c_tag, _ = self._get_current_channel_info()
            if c_tag in self._channels_data:
                del self._channels_data[c_tag]
        self._refresh_list_and_table()
        self._update_display()

    def _clear_selected_channel(self) -> None:
        c_tag, _ = self._get_current_channel_info()
        self._pending_pos_pole = None
        if c_tag in self._channels_data:
            del self._channels_data[c_tag]
        self._refresh_list_and_table()
        self._update_display()

    def _clear_all_points(self) -> None:
        self._channels_data.clear()
        self._pending_pos_pole = None
        c_tag, m_name = self._get_current_channel_info()
        self.lbl_instruction.setText(f"Marcando {c_tag} - {m_name}: Clic 1 = Polo +, Clic 2 = Polo -")
        self.lbl_instruction.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 11px;")
        self._refresh_list_and_table()
        self._update_display()

    def _get_scale_mm_per_px(self) -> float:
        dip_mm = float(self.spin_dip.value())
        if self.current_fids and "distancia_interpupilar_px" in self.current_fids:
            dip_px = float(self.current_fids["distancia_interpupilar_px"])
            if dip_px > 1.0:
                return float(dip_mm / dip_px)
        return 0.25

    def _refresh_list_and_table(self) -> None:
        self.list_pairs.clear()
        for c_tag in ["Ch0", "Ch1", "Ch2"]:
            if c_tag in self._channels_data:
                p = self._channels_data[c_tag]
                short_foto = p.get("foto", "")
                if len(short_foto) > 20:
                    short_foto = short_foto[:17] + "..."
                txt = f"[OK] {p['canal']}: {p['musculo']} [{short_foto}] -> IED: {p['ied_mm']:.1f} mm ({p['ied_px']:.0f} px) [{p['angulo_deg']:.1f} deg]"
                self.list_pairs.addItem(txt)

        self.table_distances.setRowCount(0)
        escala = self._get_scale_mm_per_px()
        fids = self.current_fids

        for c_tag in ["Ch0", "Ch1", "Ch2"]:
            if c_tag not in self._channels_data:
                continue
            p = self._channels_data[c_tag]
            for polo_nombre, pt in [("Polo +", p["polo_positivo"]), ("Polo -", p["polo_negativo"])]:
                row_idx = self.table_distances.rowCount()
                self.table_distances.insertRow(row_idx)

                d_nariz_mm = d_menton_mm = d_comisura_mm = "--"
                # Solo calcular distancias anatomicas si esta foto tiene malla facial
                if p.get("foto") == self.current_photo_name and fids and "puntos_2d_px" in fids:
                    pts_2d = fids["puntos_2d_px"]
                    pt_nariz = (pts_2d[INDICE_NARIZ_PUNTA, 0], pts_2d[INDICE_NARIZ_PUNTA, 1])
                    pt_menton = (pts_2d[INDICE_MENTON_GNATHION, 0], pts_2d[INDICE_MENTON_GNATHION, 1])
                    es_izq = (pt[0] <= pt_nariz[0])
                    idx_c = INDICE_COMISURA_IZQ if es_izq else INDICE_COMISURA_DER

                    d_nariz_mm = f"{float(np.hypot(pt[0] - pt_nariz[0], pt[1] - pt_nariz[1]) * escala):.1f}"
                    d_menton_mm = f"{float(np.hypot(pt[0] - pt_menton[0], pt[1] - pt_menton[1]) * escala):.1f}"
                    d_comisura_mm = f"{float(np.hypot(pt[0] - pts_2d[idx_c, 0], pt[1] - pts_2d[idx_c, 1]) * escala):.1f}"

                self.table_distances.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(p["canal"]))
                self.table_distances.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(p.get("foto", "")[:15]))
                self.table_distances.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(polo_nombre))
                self.table_distances.setItem(row_idx, 3, QtWidgets.QTableWidgetItem(str(d_nariz_mm)))
                self.table_distances.setItem(row_idx, 4, QtWidgets.QTableWidgetItem(str(d_menton_mm)))
                self.table_distances.setItem(row_idx, 5, QtWidgets.QTableWidgetItem(str(d_comisura_mm)))
                self.table_distances.setItem(row_idx, 6, QtWidgets.QTableWidgetItem(f"{p['ied_mm']:.1f}"))
                self.table_distances.setItem(row_idx, 7, QtWidgets.QTableWidgetItem(f"{p['angulo_deg']:.1f} deg"))

    def _update_display(self) -> None:
        if self.current_img is None:
            return

        canvas = self.current_img.copy()
        alto, ancho = canvas.shape[:2]
        self.photo_label.set_image_dimensions(ancho, alto)

        escala = self._get_scale_mm_per_px()
        diam_gel = float(self.spin_gel_diam.value())
        radio_px = max(int(round((diam_gel / 2.0) / escala)), 3)

        # 1. Malla 3D facial fiduciaria
        if self.chk_show_mesh.isChecked() and self.current_fids and "puntos_2d_px" in self.current_fids:
            pts = self.current_fids["puntos_2d_px"]
            for pt in pts:
                cv2.circle(canvas, (int(pt[0]), int(pt[1])), 1, (40, 180, 160), -1)

            nariz = (int(pts[INDICE_NARIZ_PUNTA, 0]), int(pts[INDICE_NARIZ_PUNTA, 1]))
            menton = (int(pts[INDICE_MENTON_GNATHION, 0]), int(pts[INDICE_MENTON_GNATHION, 1]))
            cv2.circle(canvas, nariz, 4, (0, 0, 255), -1)
            cv2.circle(canvas, menton, 4, (0, 255, 0), -1)

        # 2. Dibujar canales que corresponden a esta foto
        colores = {
            "Ch0": (0, 255, 128),
            "Ch1": (0, 190, 255),
            "Ch2": (255, 100, 255),
        }
        for c_tag, p in self._channels_data.items():
            # Solo dibujar sobre la foto donde se marco el canal
            if p.get("foto") and p.get("foto") != self.current_photo_name:
                continue

            color = colores.get(c_tag, (255, 255, 0))
            pos_x, pos_y = int(round(p["polo_positivo"][0])), int(round(p["polo_positivo"][1]))
            neg_x, neg_y = int(round(p["polo_negativo"][0])), int(round(p["polo_negativo"][1]))

            # Flecha de dipolo
            cv2.arrowedLine(canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25)

            # Circulo de gel a escala
            cv2.circle(canvas, (pos_x, pos_y), radio_px, (0, 255, 204), 1, cv2.LINE_AA)
            cv2.circle(canvas, (neg_x, neg_y), radio_px, (0, 255, 204), 1, cv2.LINE_AA)

            # Polo (+)
            cv2.circle(canvas, (pos_x, pos_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (pos_x, pos_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "+", (pos_x - 4, pos_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            # Polo (-)
            cv2.circle(canvas, (neg_x, neg_y), 7, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (neg_x, neg_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "-", (neg_x - 3, neg_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

            mid_x, mid_y = int(round(0.5 * (pos_x + neg_x))), int(round(0.5 * (pos_y + neg_y)))
            cv2.putText(
                canvas,
                f"{p['canal']}: {p['musculo']} [{p['ied_mm']:.1f}mm | {p['angulo_deg']:.0f}deg]",
                (mid_x + 10, mid_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                2,
                cv2.LINE_AA,
            )

        # Polo (+) pendiente en esta foto
        if self._pending_pos_pole is not None:
            c_tag, _ = self._get_current_channel_info()
            px, py = int(round(self._pending_pos_pole[0])), int(round(self._pending_pos_pole[1]))
            cv2.circle(canvas, (px, py), radio_px, (0, 255, 204), 1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7, (0, 69, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{c_tag} + [Marque Polo -]", (px + 10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        self.photo_label.set_display_pixmap(pix, ancho, alto)

    def _export_calibration(self) -> None:
        if not self._channels_data:
            QtWidgets.QMessageBox.warning(self, "Sin electrodos", "Marque al menos un canal bipolar antes de exportar.")
            return

        sess_name = self.combo_sessions.currentText()
        photo_name = self.combo_photos.currentText()
        out_dir = os.path.join(self.base_drive, sess_name, "calibracion_espacial_reconstruida")
        os.makedirs(out_dir, exist_ok=True)

        json_path = os.path.join(out_dir, "calibracion_espacial.json")
        img_out_path = os.path.join(out_dir, f"reconstruccion_3d_{os.path.splitext(photo_name)[0]}.png")

        pairs_list = list(self._channels_data.values())
        escala = self._get_scale_mm_per_px()
        dip_cm = float(self.spin_dip.value() / 10.0)

        resultado = self.agent.triangular_electrodos(
            self.current_img,
            puntos_centroides_manuales=pairs_list,
            asignar_musculos_automatico=False,
            d_ip_real_cm=dip_cm,
            escala_mm_por_px=escala,
            diametro_gel_mm=float(self.spin_gel_diam.value()),
            ied_nominal_mm=float(self.spin_ied_nom.value()),
        )

        vis = self.agent.renderizar_visualizacion(self.current_img, resultado)
        cv2.imwrite(img_out_path, vis)

        meta = resultado.get("metadatos_calibracion_espacial", {})
        meta["sujeto"] = "Candela"
        meta["sesion"] = sess_name
        meta["foto_origen"] = photo_name
        meta["diametro_gel_mm"] = float(self.spin_gel_diam.value())
        meta["ied_nominal_mm"] = float(self.spin_ied_nom.value())

        # Guardar resumen de canales con su foto de origen exacta
        meta["resumen_canales"] = {
            c_tag: {
                "musculo": p["musculo"],
                "foto_origen": p.get("foto", photo_name),
                "ied_mm": p["ied_mm"],
                "ied_px": p["ied_px"],
                "angulo_deg": p["angulo_deg"],
                "polo_positivo_2d_px": list(p["polo_positivo"]),
                "polo_negativo_2d_px": list(p["polo_negativo"]),
            }
            for c_tag, p in self._channels_data.items()
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"exito": True, "metadatos_calibracion_espacial": meta}, f, indent=2, ensure_ascii=False)

        QtWidgets.QMessageBox.information(
            self,
            "Calibracion Exitosa",
            f"Se exporto correctamente la calibracion espacial 3D de Candela:\n\n"
            f"- JSON: {json_path}\n"
            f"- Imagen: {img_out_path}\n\n"
            f"Electrodos triangulados: {len(meta.get('electrodos', []))} polos ({len(meta.get('pares_diferenciales', []))} pares bipolares).",
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = CandelaSpatialMapperDialog()
    dialog.show()
    sys.exit(app.exec())
