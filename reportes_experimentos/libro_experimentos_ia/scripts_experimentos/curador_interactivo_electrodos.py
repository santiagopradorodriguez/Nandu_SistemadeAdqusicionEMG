"""
Proyecto NANDU LSD - Sistema de Adquisición sEMG
Curador Interactivo de Detecciones de Electrodos sobre Fotos de Montaje
Permite arrastrar interactivamente con el mouse los círculos a su posición exacta,
reasignar canales (ej. de CH1 a CH2 para unificar pares bipolares), editar músculos y guardar.
"""

import os
import sys
import csv
import glob
import json
import math
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2
from PySide6 import QtCore, QtGui, QtWidgets

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CSV_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/electrodos_anotados_agente.csv")
AUDIT_MD_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/registro_auditoria_detecciones.md")
BASE_DRIVE = os.path.join(REPO_ROOT, "Sujeto_Organizado_Drive")
OUT_DIR = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29'

sys.path.append(os.path.join(REPO_ROOT, "EMG_desarrollo", "acquisition"))
from calibracion_espacial_electrodos import AntigravityAgent

from compilador_maestro_atlas import compilar_diapositiva_maestra
from ejecutar_lote_atlas_restante import CONFIGS

COLORES = {
    "CH0": (0, 255, 0),      # Verde
    "CH1": (255, 191, 0),    # Cyan/Celeste (BGR)
    "CH2": (214, 0, 255),    # Magenta/Rosa (BGR)
    "REF": (0, 255, 255),    # Amarillo (BGR)
}


def resolver_ruta_foto(campania: str, archivo_foto: str) -> str:
    if os.path.isabs(archivo_foto) and os.path.exists(archivo_foto):
        return archivo_foto
    p1 = os.path.join(BASE_DRIVE, campania, "fotos_montaje", archivo_foto)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DRIVE, campania, archivo_foto)
    if os.path.exists(p2):
        return p2

    montaje_dir = os.path.join(BASE_DRIVE, campania, "fotos_montaje")
    if os.path.exists(montaje_dir):
        archivos = sorted(os.listdir(montaje_dir))
        prefijo = archivo_foto[:3]
        # Primero coincidencia exacta o prefijo numérico "01_", "02_", "03_"
        for f in archivos:
            if f.lower() == archivo_foto.lower():
                return os.path.join(montaje_dir, f)
        for f in archivos:
            if f.startswith(prefijo) and f.lower().endswith((".png", ".jpeg", ".jpg")):
                return os.path.join(montaje_dir, f)
        # Coincidencia por palabra clave
        for f in archivos:
            if "frontal" in archivo_foto.lower() and "frontal" in f.lower():
                return os.path.join(montaje_dir, f)
            if "cuello" in archivo_foto.lower() and ("cuello" in f.lower() or "cigomatico" in f.lower()):
                return os.path.join(montaje_dir, f)
            if "modiolo" in archivo_foto.lower() and "modiolo" in f.lower():
                return os.path.join(montaje_dir, f)
            if "lateral" in archivo_foto.lower() and "lateral" in f.lower():
                return os.path.join(montaje_dir, f)
    return ""


class InteractiveCuratorCanvas(QtWidgets.QWidget):
    """
    Visor interactivo con capacidad de arrastrar puntos (drag and drop)
    y menu contextual de clic derecho para reasignar canal o eliminar/agregar.
    """

    pointMoved = QtCore.Signal()
    structureChanged = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._img_bgr: Optional[np.ndarray] = None
        self._pixmap: Optional[QtGui.QPixmap] = None
        self._img_w: int = 1
        self._img_h: int = 1
        self._target_rect: QtCore.QRect = QtCore.QRect()

        self.points: List[Dict[str, Any]] = []
        self._dragged_idx: Optional[int] = None
        self._hover_idx: Optional[int] = None
        self.crop_rect_img: Optional[QtCore.QRectF] = None
        self.fiduciarios_px: Dict[str, Tuple[float, float]] = {}
        self.canal_musculo_map: Dict[str, str] = {}

        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(600, 450)

    def load_image_and_points(self, img_path: str, points: List[Dict[str, Any]]) -> bool:
        if not os.path.exists(img_path):
            return False
        self._img_bgr = cv2.imread(img_path)
        if self._img_bgr is None:
            return False

        self._img_h, self._img_w = self._img_bgr.shape[:2]
        self.points = points
        self._dragged_idx = None
        self._hover_idx = None

        rgb = cv2.cvtColor(self._img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888).copy()
        self._pixmap = QtGui.QPixmap.fromImage(qimg)
        self.update()
        return True

    def _map_widget_to_image(self, pos: QtCore.QPoint) -> Tuple[float, float]:
        if self._target_rect.isEmpty() or self._target_rect.width() <= 0 or self._target_rect.height() <= 0:
            return float(pos.x()), float(pos.y())
        rx = (pos.x() - self._target_rect.x()) / float(self._target_rect.width())
        ry = (pos.y() - self._target_rect.y()) / float(self._target_rect.height())
        clamped_x = float(np.clip(rx, 0.0, 1.0) * self._img_w)
        clamped_y = float(np.clip(ry, 0.0, 1.0) * self._img_h)
        return clamped_x, clamped_y

    def _map_image_to_widget(self, ix: float, iy: float) -> QtCore.QPointF:
        if self._img_w <= 0 or self._img_h <= 0 or self._target_rect.isEmpty():
            return QtCore.QPointF(ix, iy)
        wx = self._target_rect.x() + (ix / float(self._img_w)) * self._target_rect.width()
        wy = self._target_rect.y() + (iy / float(self._img_h)) * self._target_rect.height()
        return QtCore.QPointF(wx, wy)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor("#0d0d0d"))

        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QtGui.QColor("#888888"))
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "Sin imagen cargada")
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

        # 1. Dibujar líneas de dipolos (pares bipolares del mismo canal)
        canales: Dict[str, List[int]] = {}
        for idx, pt in enumerate(self.points):
            canales.setdefault(pt["Canal"], []).append(idx)

        for ch, idxs in canales.items():
            if len(idxs) >= 2:
                bgr = COLORES.get(ch, (255, 255, 255))
                qcol = QtGui.QColor(bgr[2], bgr[1], bgr[0], 220)
                pen = QtGui.QPen(qcol, 2.5, QtCore.Qt.DashLine)
                painter.setPen(pen)

                p1_w = self._map_image_to_widget(float(self.points[idxs[0]]["X_px"]), float(self.points[idxs[0]]["Y_px"]))
                p2_w = self._map_image_to_widget(float(self.points[idxs[1]]["X_px"]), float(self.points[idxs[1]]["Y_px"]))
                painter.drawLine(p1_w, p2_w)

        # 2. Dibujar marcadores interactivos
        for idx, pt in enumerate(self.points):
            ch = pt["Canal"]
            bgr = COLORES.get(ch, (255, 255, 255))
            qcol = QtGui.QColor(bgr[2], bgr[1], bgr[0])

            ix = float(pt["X_px"])
            iy = float(pt["Y_px"])
            wpt = self._map_image_to_widget(ix, iy)

            is_drag = (idx == self._dragged_idx)
            is_hover = (idx == self._hover_idx)
            radius = 12 if (is_drag or is_hover) else 9

            # Halo de resaltado al arrastrar o pasar el mouse
            if is_drag or is_hover:
                painter.setPen(QtCore.Qt.NoPen)
                painter.setBrush(QtGui.QColor(qcol.red(), qcol.green(), qcol.blue(), 90))
                painter.drawEllipse(wpt, radius + 8, radius + 8)

            # Aro exterior de canal
            painter.setPen(QtGui.QPen(qcol, 3))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawEllipse(wpt, radius, radius)

            # Aro blanco de contraste
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.2))
            painter.drawEllipse(wpt, radius - 2, radius - 2)

            # Centro solido
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor("#ff3333") if "red" in pt.get("Tipo_Deteccion", "").lower() else QtGui.QColor("#111111"))
            painter.drawEllipse(wpt, radius - 4, radius - 4)

            # Etiqueta con Canal y Musculo exacto de la sesion sin parentesis
            musc_slide = self.canal_musculo_map.get(ch, "")
            musculo_tag = pt.get("Musculo_Polo", "")
            if not musculo_tag or "manual" in musculo_tag.lower() or musculo_tag == pt.get("ID_Electrodo", ""):
                musculo_tag = musc_slide if musc_slide else pt.get("ID_Electrodo", ch)
            elif musc_slide and musc_slide.lower() not in musculo_tag.lower():
                musculo_tag = f"{musc_slide} - {musculo_tag}"

            # Regla estricta: cero parentesis en etiquetas
            musculo_tag = musculo_tag.replace("(", "- ").replace(")", "").strip()
            id_txt = f"{pt['Canal']}: {musculo_tag} - {int(round(ix))}, {int(round(iy))}"
            painter.setFont(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Bold))
            painter.setPen(QtGui.QColor("#000000"))
            painter.drawText(int(wpt.x()) + radius + 7, int(wpt.y()) + 5, id_txt)
            painter.setPen(qcol)
            painter.drawText(int(wpt.x()) + radius + 6, int(wpt.y()) + 4, id_txt)

        painter.end()

    def _find_point_near(self, pos: QtCore.QPoint, threshold_px: float = 22.0) -> Optional[int]:
        for idx, pt in enumerate(self.points):
            ix = float(pt["X_px"])
            iy = float(pt["Y_px"])
            wpt = self._map_image_to_widget(ix, iy)
            dist = np.hypot(pos.x() - wpt.x(), pos.y() - wpt.y())
            if dist <= threshold_px:
                return idx
        return None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            idx = self._find_point_near(event.position().toPoint())
            if idx is not None:
                self._dragged_idx = idx
                self.setCursor(QtCore.Qt.ClosedHandCursor)
                self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragged_idx is not None:
            ix, iy = self._map_widget_to_image(event.position().toPoint())
            self.points[self._dragged_idx]["X_px"] = f"{ix:.1f}"
            self.points[self._dragged_idx]["Y_px"] = f"{iy:.1f}"
            self.pointMoved.emit()
            self.update()
        else:
            hover = self._find_point_near(event.position().toPoint())
            if hover != self._hover_idx:
                self._hover_idx = hover
                self.setCursor(QtCore.Qt.OpenHandCursor if hover is not None else QtCore.Qt.ArrowCursor)
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton and self._dragged_idx is not None:
            self._dragged_idx = None
            self.setCursor(QtCore.Qt.ArrowCursor)
            self.pointMoved.emit()
            self.update()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:
        idx = self._find_point_near(event.pos(), threshold_px=22.0)
        menu = QtWidgets.QMenu(self)

        if idx is not None:
            pt = self.points[idx]
            ch_curr = pt.get("Canal", "")
            menu.addSection(f"{ch_curr} - {pt.get('Musculo_Polo', '')}")

            ch_menu = menu.addMenu("Reasignar a otro Canal...")
            for c in ["CH0", "CH1", "CH2", "REF"]:
                musc_nom = self.canal_musculo_map.get(c, "")
                lbl = f"Canal {c}: {musc_nom}" if musc_nom else f"Canal {c}"
                act = ch_menu.addAction(lbl)
                if c == ch_curr:
                    act.setEnabled(False)
                act.triggered.connect(lambda _, c=c, i=idx: self._set_point_channel(i, c))

            del_act = menu.addAction("Eliminar este electrodo")
            del_act.triggered.connect(lambda _, i=idx: self._remove_point(i))
        else:
            ix, iy = self._map_widget_to_image(event.pos())
            add_menu = menu.addMenu(f"Agregar Electrodo en {int(ix)}, {int(iy)} px...")
            for c in ["CH0", "CH1", "CH2", "REF"]:
                musc_nom = self.canal_musculo_map.get(c, "").replace("(", "- ").replace(")", "")
                lbl = f"Agregar como {c} - {musc_nom}" if musc_nom else f"Agregar como {c}"
                act = add_menu.addAction(lbl)
                act.triggered.connect(lambda _, c=c, x=ix, y=iy: self._add_point(c, x, y))

        menu.exec(event.globalPos())

    def _set_point_channel(self, idx: int, new_ch: str) -> None:
        if 0 <= idx < len(self.points):
            self.points[idx]["Canal"] = new_ch
            musc_nom = self.canal_musculo_map.get(new_ch, "")
            if musc_nom:
                self.points[idx]["Musculo_Polo"] = musc_nom
            # Si el ID empezaba con CH1 y pasa a CH2, adaptar ID
            id_old = self.points[idx].get("ID_Electrodo", "")
            if id_old.startswith("CH"):
                self.points[idx]["ID_Electrodo"] = f"{new_ch}_{id_old.split('_', 1)[-1]}"
            self.structureChanged.emit()
            self.update()

    def _remove_point(self, idx: int) -> None:
        if 0 <= idx < len(self.points):
            del self.points[idx]
            self.structureChanged.emit()
            self.update()

    def _add_point(self, ch: str, ix: float, iy: float) -> None:
        musc_nom = self.canal_musculo_map.get(ch, f"{ch} Manual")
        nuevo = {
            "Canal": ch,
            "ID_Electrodo": f"{ch}_E{len(self.points) + 1}",
            "Musculo_Polo": musc_nom,
            "Tipo_Deteccion": "manual_curated",
            "X_px": f"{ix:.1f}",
            "Y_px": f"{iy:.1f}",
            "X_norm": f"{round(ix / float(self._img_w), 4)}",
            "Y_norm": f"{round(iy / float(self._img_h), 4)}",
            "IED_Bipolar_px": "",
            "Angulo_Bipolar_deg": "",
        }
        self.points.append(nuevo)
        self.structureChanged.emit()
        self.update()


class CuradorInteractivoElectrodosDialog(QtWidgets.QDialog):
    """
    Ventana principal de curacion de electrodos con cambio flexible de canales,
    recalculo automatico de IED y persistencia limpia en CSV.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Curador Interactivo de Posiciones de Electrodos - Sujeto")
        self.resize(1340, 870)

        self._csv_rows: List[Dict[str, str]] = []
        self._unique_photos: List[Tuple[str, str]] = []
        self._current_photo_idx: int = 0
        self._current_img_path: str = ""

        # Integración con el Atlas Oficial sEMG
        self.agente_vision = AntigravityAgent()
        self._fiduciarios_cache: Dict[str, Any] = {}
        self._scale_px_to_mm: float = 1.0
        # Medidas biométricas reales de Sujeto (Patrón de Calibración Oficial):
        # - Distancia inter-ocular / interpupilar frontal: 7.0 cm = 70.0 mm
        # - Distancia labio (Cheilion) a trago/oreja (Tragus) lateral: 8.0 cm = 80.0 mm
        self._ref_interocular_mm: float = 70.0
        self._ref_labio_trago_mm: float = 80.0
        self._modo_escala_biometrica: str = "Auto (Frontal 70mm / Lateral 80mm)" 
        self._atlas_configs: List[Dict[str, Any]] = []
        self._current_slide_idx: int = 0
        self._init_atlas_configs()

        self._load_csv()
        self._build_ui()
        self._apply_dark_theme()
        self._setup_atlas_ui()
        self._load_current_photo()

    def _init_atlas_configs(self) -> None:
        """Inicializa las diapositivas del Atlas ordenadas: 2026-09-25 primero, dejando agosto al final."""
        diapos_sin_agosto = [cfg for cfg in CONFIGS if not ('08-28' in cfg['id'] or '08-29' in cfg['id'] or '08-30' in cfg['id'])]
        diapos_agosto = [cfg for cfg in CONFIGS if ('08-28' in cfg['id'] or '08-29' in cfg['id'] or '08-30' in cfg['id'])]

        idx_0925 = [i for i, cfg in enumerate(diapos_sin_agosto) if '09-25' in cfg['id']]
        if idx_0925:
            start_i = idx_0925[0]
            self._atlas_configs = diapos_sin_agosto[start_i:] + diapos_sin_agosto[:start_i] + diapos_agosto
        else:
            self._atlas_configs = diapos_sin_agosto + diapos_agosto

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #14171a;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }
            QGroupBox {
                background-color: #1e2227;
                border: 1px solid #30363d;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
                color: #00ffcc;
            }
            QPushButton {
                background-color: #262c34;
                color: #ffffff;
                border: 1px solid #00ffcc;
                border-radius: 5px;
                padding: 7px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #004d40;
                border-color: #33ffdd;
            }
            QComboBox {
                background-color: #1f242c;
                color: #00ffcc;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
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

    def _load_csv(self) -> None:
        if not os.path.exists(CSV_PATH):
            return
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self._csv_rows = list(reader)

        vistos = set()
        self._unique_photos = []
        for r in self._csv_rows:
            key = (r["Campaña"], r["Archivo_Foto"])
            if key not in vistos:
                vistos.add(key)
                self._unique_photos.append(key)
        self._all_csv_photos = list(self._unique_photos)

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(12)

        # Columna Izquierda: Visor Interactivo
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(8)

        # Barra superior de navegacion
        top_bar = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("< Foto Anterior")
        self.btn_prev.clicked.connect(self._prev_photo)
        top_bar.addWidget(self.btn_prev)

        self.combo_photos = QtWidgets.QComboBox()
        for idx, (camp, foto) in enumerate(self._unique_photos):
            camp_corta = camp.split("_")[0]
            self.combo_photos.addItem(f"[{idx + 1}/{len(self._unique_photos)}] {camp_corta} - {foto}")
        self.combo_photos.currentIndexChanged.connect(self._on_combo_changed)
        top_bar.addWidget(self.combo_photos, stretch=1)

        self.btn_next = QtWidgets.QPushButton("Foto Siguiente >")
        self.btn_next.clicked.connect(self._next_photo)
        top_bar.addWidget(self.btn_next)

        left_col.addLayout(top_bar)

        # Canvas con arrastre de puntos y menu contextual
        self.canvas = InteractiveCuratorCanvas(self)
        self.canvas.pointMoved.connect(self._on_point_moved)
        self.canvas.structureChanged.connect(self._on_structure_changed)
        left_col.addWidget(self.canvas, stretch=1)

        self.lbl_help = QtWidgets.QLabel(
            "Uso: Arrastre los circulos con clic izquierdo. Clic derecho sobre un punto para reasignar su Canal o eliminarlo. Clic derecho en espacio vacio para agregar uno nuevo."
        )
        self.lbl_help.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 10px;")
        left_col.addWidget(self.lbl_help)

        main_layout.addLayout(left_col, stretch=3)

        # Columna Derecha: Tabla y Acciones
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(10)

        # Info de Foto
        info_group = QtWidgets.QGroupBox("Informacion de Toma")
        info_layout = QtWidgets.QVBoxLayout(info_group)
        self.lbl_camp_info = QtWidgets.QLabel("Campaña: --")
        self.lbl_camp_info.setWordWrap(True)
        self.lbl_camp_info.setStyleSheet("color: #b0bec5; font-size: 11px;")
        info_layout.addWidget(self.lbl_camp_info)

        self.lbl_res_info = QtWidgets.QLabel("Resolucion: --")
        self.lbl_res_info.setStyleSheet("color: #b0bec5; font-size: 11px;")
        info_layout.addWidget(self.lbl_res_info)

        # Panel de Calibración Biométrica Real (Sujeto: 70.0 mm Inter-ocular / 80.0 mm Labio-Trago)
        calib_layout = QtWidgets.QGridLayout()
        calib_layout.setSpacing(4)
        lbl_modo_esc = QtWidgets.QLabel("Modo Escala:")
        lbl_modo_esc.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold;")
        self.cmb_modo_escala = QtWidgets.QComboBox()
        self.cmb_modo_escala.addItems([
            "Auto (Frontal 70mm / Lateral 80mm)",
            "Frontal: Inter-ocular (mm)",
            "Lateral: Labio-Trago (mm)"
        ])
        self.cmb_modo_escala.currentIndexChanged.connect(self._recalcular_escala_biometrica)

        lbl_io = QtWidgets.QLabel("Inter-ocular (mm):")
        self.spin_interocular_mm = QtWidgets.QDoubleSpinBox()
        self.spin_interocular_mm.setRange(40.0, 110.0)
        self.spin_interocular_mm.setDecimals(1)
        self.spin_interocular_mm.setSingleStep(0.5)
        self.spin_interocular_mm.setValue(70.0)
        self.spin_interocular_mm.setToolTip("Distancia real entre iris (Sujeto = 70.0 mm / 7.0 cm)")
        self.spin_interocular_mm.valueChanged.connect(self._recalcular_escala_biometrica)

        lbl_lt = QtWidgets.QLabel("Labio-Trago (mm):")
        self.spin_labio_trago_mm = QtWidgets.QDoubleSpinBox()
        self.spin_labio_trago_mm.setRange(45.0, 140.0)
        self.spin_labio_trago_mm.setDecimals(1)
        self.spin_labio_trago_mm.setSingleStep(0.5)
        self.spin_labio_trago_mm.setValue(80.0)
        self.spin_labio_trago_mm.setToolTip("Distancia real Comisura Labial (Cheilion) a Trago de la oreja (Sujeto = 80.0 mm / 8.0 cm)")
        self.spin_labio_trago_mm.valueChanged.connect(self._recalcular_escala_biometrica)

        calib_layout.addWidget(lbl_modo_esc, 0, 0)
        calib_layout.addWidget(self.cmb_modo_escala, 0, 1, 1, 3)
        calib_layout.addWidget(lbl_io, 1, 0)
        calib_layout.addWidget(self.spin_interocular_mm, 1, 1)
        calib_layout.addWidget(lbl_lt, 1, 2)
        calib_layout.addWidget(self.spin_labio_trago_mm, 1, 3)
        info_layout.addLayout(calib_layout)

        right_col.addWidget(info_group)

        # Tabla de Puntos
        table_group = QtWidgets.QGroupBox("Electrodos Detectados en esta Foto")
        table_layout = QtWidgets.QVBoxLayout(table_group)

        self.lbl_legend = QtWidgets.QLabel("Leyenda: CH0 (Verde) | CH1 (Cyan) | CH2 (Magenta) | REF (Amarillo)")
        self.lbl_legend.setStyleSheet("color: #b0bec5; font-size: 10px; font-weight: bold;")
        table_layout.addWidget(self.lbl_legend)

        self.table_points = QtWidgets.QTableWidget(0, 6)
        self.table_points.setHorizontalHeaderLabels(["Canal (Combo)", "Musculo / Polo", "ID", "X (px)", "Y (px)", "IED (px)"])
        self.table_points.horizontalHeader().setStretchLastSection(True)
        self.table_points.itemChanged.connect(self._on_table_item_changed)
        table_layout.addWidget(self.table_points)

        # Botones rapidos para agregar o borrar puntos
        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_pt = QtWidgets.QPushButton("+ Agregar Punto")
        self.btn_add_pt.clicked.connect(self._add_point_btn)
        btn_row.addWidget(self.btn_add_pt)

        self.btn_del_pt = QtWidgets.QPushButton("- Eliminar Punto")
        self.btn_del_pt.clicked.connect(self._del_point_btn)
        btn_row.addWidget(self.btn_del_pt)
        table_layout.addLayout(btn_row)

        right_col.addWidget(table_group)

        # Botones de Accion
        self.btn_guardar = QtWidgets.QPushButton("Guardar Posiciones Corregidas")
        self.btn_guardar.setStyleSheet(
            "QPushButton { background-color: #00c853; color: #0a0a0a; border: 2px solid #00ffcc; "
            "padding: 10px; font-size: 13px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00e676; }"
        )
        self.btn_guardar.clicked.connect(self._save_changes)
        right_col.addWidget(self.btn_guardar)

        self.btn_aprobar = QtWidgets.QPushButton("Aprobar Toma (Marcar como Buena)")
        self.btn_aprobar.setStyleSheet(
            "QPushButton { background-color: #0288d1; color: #ffffff; border: 1px solid #4fc3f7; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #039be5; }"
        )
        self.btn_aprobar.clicked.connect(self._marcar_aprobada)
        right_col.addWidget(self.btn_aprobar)

        self.btn_marcar_mala = QtWidgets.QPushButton("Marcar como Mala / A Revisar")
        self.btn_marcar_mala.setStyleSheet(
            "QPushButton { background-color: #d32f2f; color: #ffffff; border: 1px solid #ef5350; "
            "padding: 8px; font-weight: bold; }"
            "QPushButton:hover { background-color: #e53935; }"
        )
        self.btn_marcar_mala.clicked.connect(self._marcar_mala)
        right_col.addWidget(self.btn_marcar_mala)

        # ==============================================================================
        # SECCIÓN ATLAS sEMG: Navegación por Diapositivas, Biometría y Recompilación
        # ==============================================================================
        atlas_group = QtWidgets.QGroupBox("Atlas Oficial sEMG - Biometría y PDF")
        atlas_layout = QtWidgets.QVBoxLayout(atlas_group)
        atlas_layout.setSpacing(6)

        atlas_layout.addWidget(QtWidgets.QLabel("Diapositiva en Edición:"))
        self.combo_slides = QtWidgets.QComboBox()
        for idx_s, cfg in enumerate(self._atlas_configs):
            self.combo_slides.addItem(f"[{idx_s + 1:02d}/{len(self._atlas_configs):02d}] {cfg['id']}")
        self.combo_slides.currentIndexChanged.connect(self._on_atlas_slide_changed)
        atlas_layout.addWidget(self.combo_slides)

        atlas_layout.addWidget(QtWidgets.QLabel("Canal Muscular a Curar:"))
        self.combo_channels = QtWidgets.QComboBox()
        self.combo_channels.currentIndexChanged.connect(self._on_atlas_channel_changed)
        atlas_layout.addWidget(self.combo_channels)

        # Ajuste de encuadre y zoom del marco 3:4
        zoom_row = QtWidgets.QHBoxLayout()
        zoom_row.addWidget(QtWidgets.QLabel("Alto Marco 3:4 (px):"))
        self.spin_crop_height = QtWidgets.QSpinBox()
        self.spin_crop_height.setRange(150, 1500)
        self.spin_crop_height.setValue(380)
        self.spin_crop_height.setSingleStep(25)
        self.spin_crop_height.valueChanged.connect(self._recalc_atlas_biometrics)
        zoom_row.addWidget(self.spin_crop_height)
        atlas_layout.addLayout(zoom_row)

        self.lbl_atlas_metrics = QtWidgets.QLabel("Métricas: Pendiente de cálculo...")
        self.lbl_atlas_metrics.setStyleSheet(
            "color: #fbbf24; font-size: 11px; font-weight: bold; background-color: #0d1117; "
            "padding: 6px; border: 1px solid #30363d; border-radius: 4px;"
        )
        self.lbl_atlas_metrics.setWordWrap(True)
        atlas_layout.addWidget(self.lbl_atlas_metrics)

        self.btn_save_atlas = QtWidgets.QPushButton("Guardar Recorte 3:4 y Recompilar PDF")
        self.btn_save_atlas.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: #ffffff; border: 2px solid #38bdf8; "
            "padding: 9px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        self.btn_save_atlas.clicked.connect(self._save_and_apply_atlas)
        atlas_layout.addWidget(self.btn_save_atlas)

        self.btn_foto_perdida = QtWidgets.QPushButton("Marcar Masetero como Imagen Perdida")
        self.btn_foto_perdida.setStyleSheet(
            "QPushButton { background-color: #b45309; color: #ffffff; border: 1px solid #f59e0b; "
            "padding: 6px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #92400e; }"
        )
        self.btn_foto_perdida.clicked.connect(self._marcar_foto_perdida_atlas)
        atlas_layout.addWidget(self.btn_foto_perdida)

        self.btn_next_slide = QtWidgets.QPushButton("Avanzar a Siguiente Diapositiva Atlas >")
        self.btn_next_slide.setStyleSheet(
            "QPushButton { background-color: #374151; color: #ffffff; border: 1px solid #6b7280; "
            "padding: 6px; font-size: 11px; font-weight: bold; }"
            "QPushButton:hover { background-color: #4b5563; }"
        )
        self.btn_next_slide.clicked.connect(self._next_atlas_slide)
        atlas_layout.addWidget(self.btn_next_slide)

        right_col.addWidget(atlas_group)

        main_layout.addLayout(right_col, stretch=1)

    def _prev_photo(self) -> None:
        idx = self.combo_photos.currentIndex()
        if idx > 0:
            self.combo_photos.setCurrentIndex(idx - 1)

    def _next_photo(self) -> None:
        idx = self.combo_photos.currentIndex()
        if idx < self.combo_photos.count() - 1:
            self.combo_photos.setCurrentIndex(idx + 1)

    def _on_combo_changed(self, idx: int) -> None:
        if 0 <= idx < len(self._unique_photos):
            self._current_photo_idx = idx
            self._load_current_photo()

    def _load_current_photo(self) -> None:
        if not self._unique_photos:
            return

        camp, foto = self._unique_photos[self._current_photo_idx]
        self.lbl_camp_info.setText(f"Campaña: {camp}\nFoto: {foto}")

        ruta_img = resolver_ruta_foto(camp, foto)
        self._current_img_path = ruta_img

        if not ruta_img or not os.path.exists(ruta_img):
            self.lbl_res_info.setText(f"[Aviso] Archivo no encontrado en disco: {foto}")
            return

        # Puntos de esta foto en el CSV
        puntos_foto = [
            r for r in self._csv_rows
            if r["Campaña"] == camp and r["Archivo_Foto"] == foto
        ]

        ok = self.canvas.load_image_and_points(ruta_img, puntos_foto)
        if ok:
            # Extraer o reutilizar fiduciarios MediaPipe
            if ruta_img in self._fiduciarios_cache:
                fid = self._fiduciarios_cache[ruta_img]
            else:
                fid = self.agente_vision.extraer_malla_facial_3d(self.canvas._img_bgr)
                self._fiduciarios_cache[ruta_img] = fid

            if fid is not None and "puntos_2d_px" in fid:
                p2d = fid["puntos_2d_px"]
                # Elegir el par Tragus/Cheilion del perfil más visible en la imagen
                trago_izq = tuple(p2d[234].astype(float))
                trago_der = tuple(p2d[454].astype(float)) if len(p2d) > 454 else trago_izq
                com_izq = tuple(p2d[61].astype(float))
                com_der = tuple(p2d[291].astype(float)) if len(p2d) > 291 else com_izq
                d_izq = math.hypot(com_izq[0] - trago_izq[0], com_izq[1] - trago_izq[1])
                d_der = math.hypot(com_der[0] - trago_der[0], com_der[1] - trago_der[1])
                if d_der > d_izq * 1.15:
                    trago_sel, com_sel, d_lt_px = trago_der, com_der, d_der
                else:
                    trago_sel, com_sel, d_lt_px = trago_izq, com_izq, d_izq

                self.canvas.fiduciarios_px = {
                    "Pronasale (Nariz)": tuple(p2d[1].astype(float)),
                    "Gnathion (Mentón)": tuple(p2d[152].astype(float)),
                    "Tragus (Oreja)": trago_sel,
                    "Cheilion (Comisura)": com_sel,
                    "_d_ip_px": float(fid.get("distancia_interpupilar_px", 100.0)),
                    "_d_lt_px": float(d_lt_px),
                }
                self._recalcular_escala_biometrica(update_points=False)
            else:
                self._scale_px_to_mm = 100.0 / float(max(self.canvas._img_h, 1))
                self.canvas.fiduciarios_px = {}

            # Si la foto no tenía puntos registrados, colocar únicamente 2 electrodos para el canal activo
            if not self.canvas.points:
                cw = float(self.canvas._img_w)
                ch = float(self.canvas._img_h)
                ch_cur = "CH0"
                if hasattr(self, "combo_channels") and self.combo_channels.count() > 0:
                    c_txt = self.combo_channels.currentText()
                    ch_cur = "CH0" if "0" in c_txt else ("CH1" if "1" in c_txt else "CH2")
                self.canvas._add_point(ch_cur, cw * 0.42, ch * 0.52)
                self.canvas._add_point(ch_cur, cw * 0.48, ch * 0.58)

            self.lbl_res_info.setText(f"Resolucion: {self.canvas._img_w}x{self.canvas._img_h} px | Electrodos: {len(self.canvas.points)}")
            self._on_point_moved()

    def _recalcular_escala_biometrica(self, *args, update_points: bool = True) -> None:
        ref_io = float(self.spin_interocular_mm.value()) if hasattr(self, "spin_interocular_mm") else 70.0
        ref_lt = float(self.spin_labio_trago_mm.value()) if hasattr(self, "spin_labio_trago_mm") else 80.0
        self._ref_interocular_mm = ref_io
        self._ref_labio_trago_mm = ref_lt
        modo_idx = self.cmb_modo_escala.currentIndex() if hasattr(self, "cmb_modo_escala") else 0

        fid = getattr(self.canvas, "fiduciarios_px", {}) or {}
        d_ip_px = float(fid.get("_d_ip_px", 0.0))
        d_lt_px = float(fid.get("_d_lt_px", 0.0))
        foto_lower = os.path.basename(self._current_img_path or "").lower()
        es_vista_lateral = any(k in foto_lower for k in ("lateral", "perfil", "modiolo", "cuello", "cigomatico")) or (d_lt_px > d_ip_px * 1.35)

        if modo_idx == 1 and d_ip_px > 10.0:
            self._scale_px_to_mm = ref_io / d_ip_px
            ref_tag = f"Frontal ({ref_io:.1f} mm / {d_ip_px:.1f} px)"
        elif modo_idx == 2 and d_lt_px > 10.0:
            self._scale_px_to_mm = ref_lt / d_lt_px
            ref_tag = f"Lateral Labio-Trago ({ref_lt:.1f} mm / {d_lt_px:.1f} px)"
        else:
            if es_vista_lateral and d_lt_px > 10.0:
                self._scale_px_to_mm = ref_lt / d_lt_px
                ref_tag = f"Auto-Lateral Labio-Trago ({ref_lt:.1f} mm)"
            elif d_ip_px > 10.0:
                self._scale_px_to_mm = ref_io / d_ip_px
                ref_tag = f"Auto-Frontal Inter-ocular ({ref_io:.1f} mm)"
            else:
                self._scale_px_to_mm = 100.0 / float(max(self.canvas._img_h, 1))
                ref_tag = "Fallback altura"

        if hasattr(self, "lbl_res_info") and self.canvas._img_w > 0:
            self.lbl_res_info.setText(
                f"Res: {self.canvas._img_w}x{self.canvas._img_h} px | Escala: {self._scale_px_to_mm:.4f} mm/px [{ref_tag}]"
            )
        if update_points:
            self._on_point_moved()

    def _on_structure_changed(self) -> None:
        self._on_point_moved()

    def _on_point_moved(self) -> None:
        # Recalcular IED entre pares del mismo canal
        canales: Dict[str, List[Dict[str, Any]]] = {}
        for p in self.canvas.points:
            canales.setdefault(p["Canal"], []).append(p)

        for ch, pts in canales.items():
            if len(pts) == 2:
                x1, y1 = float(pts[0]["X_px"]), float(pts[0]["Y_px"])
                x2, y2 = float(pts[1]["X_px"]), float(pts[1]["Y_px"])
                ied = float(np.hypot(x2 - x1, y2 - y1))
                ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                pts[0]["IED_Bipolar_px"] = f"{ied:.1f}"
                pts[1]["IED_Bipolar_px"] = f"{ied:.1f}"
                pts[0]["Angulo_Bipolar_deg"] = f"{ang:.1f}"
                pts[1]["Angulo_Bipolar_deg"] = f"{ang:.1f}"
            else:
                for pt in pts:
                    pt["IED_Bipolar_px"] = ""
                    pt["Angulo_Bipolar_deg"] = ""

        self._refresh_table()
        self._recalc_atlas_biometrics()

    def _refresh_table(self) -> None:
        self.table_points.blockSignals(True)
        self.table_points.setRowCount(0)

        for idx, p in enumerate(self.canvas.points):
            r_idx = self.table_points.rowCount()
            self.table_points.insertRow(r_idx)

            # Col 0: Combo de seleccion de canal (CH0, CH1, CH2, REF)
            cb = QtWidgets.QComboBox()
            cb.addItems(["CH0", "CH1", "CH2", "REF"])
            cb.setCurrentText(p["Canal"])
            cb.currentIndexChanged.connect(lambda _, cb=cb, i=idx: self._on_table_combo_changed(i, cb.currentText()))
            self.table_points.setCellWidget(r_idx, 0, cb)

            # Col 1: Musculo / Polo (editable)
            val_musc = p.get("Musculo_Polo", "")
            musc_slide = getattr(self, "canal_musculo_map", {}).get(p["Canal"], "")
            if not val_musc or "manual" in val_musc.lower():
                val_musc = musc_slide if musc_slide else val_musc
            item_musc = QtWidgets.QTableWidgetItem(val_musc)
            self.table_points.setItem(r_idx, 1, item_musc)

            # Col 2: ID (editable)
            item_id = QtWidgets.QTableWidgetItem(p.get("ID_Electrodo", ""))
            self.table_points.setItem(r_idx, 2, item_id)

            # Col 3: X_px
            self.table_points.setItem(r_idx, 3, QtWidgets.QTableWidgetItem(str(p["X_px"])))
            # Col 4: Y_px
            self.table_points.setItem(r_idx, 4, QtWidgets.QTableWidgetItem(str(p["Y_px"])))
            # Col 5: IED
            self.table_points.setItem(r_idx, 5, QtWidgets.QTableWidgetItem(str(p.get("IED_Bipolar_px", "--"))))

        self.table_points.blockSignals(False)

    def _on_table_combo_changed(self, idx: int, new_ch: str) -> None:
        if 0 <= idx < len(self.canvas.points):
            self.canvas.points[idx]["Canal"] = new_ch
            musc_nom = getattr(self, "canal_musculo_map", {}).get(new_ch, "")
            if musc_nom:
                self.canvas.points[idx]["Musculo_Polo"] = musc_nom
            id_old = self.canvas.points[idx].get("ID_Electrodo", "")
            if id_old.startswith("CH"):
                self.canvas.points[idx]["ID_Electrodo"] = f"{new_ch}_{id_old.split('_', 1)[-1]}"
            self._on_point_moved()
            self.canvas.update()

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        r = item.row()
        c = item.column()
        if 0 <= r < len(self.canvas.points):
            val = item.text().strip()
            if c == 1:
                self.canvas.points[r]["Musculo_Polo"] = val
                self.canvas.update()
            elif c == 2:
                self.canvas.points[r]["ID_Electrodo"] = val
                self.canvas.update()

    def _add_point_btn(self) -> None:
        # Agregar punto al centro de la imagen
        cx = float(self.canvas._img_w // 2)
        cy = float(self.canvas._img_h // 2)
        self.canvas._add_point("CH1", cx, cy)

    def _del_point_btn(self) -> None:
        row = self.table_points.currentRow()
        if 0 <= row < len(self.canvas.points):
            self.canvas._remove_point(row)

    def _save_changes(self) -> None:
        camp, foto = self._unique_photos[self._current_photo_idx]
        w_img = float(self.canvas._img_w)
        h_img = float(self.canvas._img_h)

        # Actualizar datos normalizados en los puntos de la toma actual
        for p in self.canvas.points:
            ix = float(p["X_px"])
            iy = float(p["Y_px"])
            p["Campaña"] = camp
            p["Archivo_Foto"] = foto
            p["Resolucion_WxH"] = f"{int(w_img)}x{int(h_img)}"
            p["X_norm"] = f"{round(ix / w_img, 4)}"
            p["Y_norm"] = f"{round(iy / h_img, 4)}"
            if "Tipo_Deteccion" not in p:
                p["Tipo_Deteccion"] = "curated"
            if "Vista" not in p:
                p["Vista"] = "Lateral / Frontal"

        # Reemplazar completamente las filas de esta foto en el master CSV
        nuevas_filas = [
            r for r in self._csv_rows
            if not (r["Campaña"] == camp and r["Archivo_Foto"] == foto)
        ]
        nuevas_filas.extend(self.canvas.points)
        self._csv_rows = nuevas_filas

        # Escribir CSV actualizado
        fieldnames = [
            "Campaña", "Archivo_Foto", "Resolucion_WxH", "Vista", "Canal",
            "ID_Electrodo", "Musculo_Polo", "Tipo_Deteccion", "X_px", "Y_px",
            "X_norm", "Y_norm", "IED_Bipolar_px", "Angulo_Bipolar_deg"
        ]
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._csv_rows)

        QtWidgets.QMessageBox.information(
            self,
            "Guardado Correcto",
            f"Se guardaron las coordenadas y canales de {foto} en el CSV maestro.",
        )

    def _marcar_aprobada(self) -> None:
        self._save_changes()
        camp, foto = self._unique_photos[self._current_photo_idx]
        with open(AUDIT_MD_PATH, "a", encoding="utf-8") as f:
            f.write(f"| {camp} | {foto} | Aprobada Curada | Corregida y aprobada interactivamente |\n")
        QtWidgets.QMessageBox.information(self, "Aprobada", f"La toma {foto} se marco como Aprobada Curada.")
        self._next_photo()

    def _marcar_mala(self) -> None:
        camp, foto = self._unique_photos[self._current_photo_idx]
        with open(AUDIT_MD_PATH, "a", encoding="utf-8") as f:
            f.write(f"| {camp} | {foto} | Mala | Desplazada o descartada |\n")
        QtWidgets.QMessageBox.information(self, "Registrada", f"La toma {foto} se registro en la lista de Malas.")
        self._next_photo()

    def _setup_atlas_ui(self) -> None:
        """Configura los combos iniciales de Diapositiva y Canal del Atlas."""
        if hasattr(self, "combo_slides") and self.combo_slides.count() > 0:
            self._on_atlas_slide_changed(0)

    def _on_atlas_slide_changed(self, idx: int) -> None:
        """Carga los canales y fotografías de la campaña de la diapositiva seleccionada."""
        if idx < 0 or idx >= len(self._atlas_configs):
            return
        self._current_slide_idx = idx
        cfg = self._atlas_configs[idx]
        camp_dir = cfg["camp"]
        camp_nombre = os.path.basename(camp_dir)

        # Actualizar mapa de canales y musculos de esta diapositiva sin parentesis
        self.canal_musculo_map = {}
        for ch in cfg["canales"]:
            c_csv = ch["col_csv"]
            ch_k = "CH0" if "0" in c_csv else ("CH1" if "1" in c_csv else ("CH2" if "2" in c_csv else "REF"))
            m1 = ch.get("musculo_linea1", "").replace("(", "- ").replace(")", "")
            m2 = ch.get("musculo_linea2", "").replace("(", "- ").replace(")", "")
            self.canal_musculo_map[ch_k] = f"{m1} - {m2}" if m2 else m1
        self.canal_musculo_map["REF"] = "Referencia Mastoides"
        self.canvas.canal_musculo_map = self.canal_musculo_map

        # Actualizar leyenda visual de colores y musculos
        if hasattr(self, "lbl_legend"):
            col_html = {"CH0": "#22c55e", "CH1": "#38bdf8", "CH2": "#e879f9", "REF": "#facc15"}
            parts = []
            for ch_k in ["CH0", "CH1", "CH2", "REF"]:
                if ch_k in self.canal_musculo_map:
                    nom = self.canal_musculo_map[ch_k]
                    c_h = col_html.get(ch_k, "#ffffff")
                    parts.append(f"<span style='color:{c_h};'><b>{ch_k}: {nom}</b></span>")
            self.lbl_legend.setText(" | ".join(parts))

        # Actualizar canales en el combo
        self.combo_channels.blockSignals(True)
        self.combo_channels.clear()
        for c_idx, ch in enumerate(cfg["canales"]):
            if ch.get("activo", True):
                self.combo_channels.addItem(f"{ch['col_csv']}: {ch['musculo_linea1']}")
        self.combo_channels.blockSignals(False)

        # Filtrar fotos de la campaña priorizando las que ya están en el CSV curado
        fotos_camp = [
            (c, f) for (c, f) in getattr(self, "_all_csv_photos", self._unique_photos)
            if camp_nombre.lower() in c.lower() or c.lower() in camp_nombre.lower()
        ]

        # Agregar tomas adicionales de fotos_montaje sin duplicar prefijos ni nombres
        montaje_dir = os.path.join(camp_dir, "fotos_montaje")
        vistos_prefijos = set(f.split("_")[0] for (_, f) in fotos_camp if "_" in f)
        vistos_nombres = set(f for (_, f) in fotos_camp)

        if os.path.exists(montaje_dir):
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                for p in sorted(glob.glob(os.path.join(montaje_dir, ext))):
                    fname = os.path.basename(p)
                    pref = fname.split("_")[0] if "_" in fname else fname
                    if "curada" not in fname and "render" not in fname:
                        if pref not in vistos_prefijos and fname not in vistos_nombres:
                            vistos_prefijos.add(pref)
                            vistos_nombres.add(fname)
                            fotos_camp.append((camp_nombre, fname))

        # Si hay fotos de la campaña, actualizar el visor para mostrar solo las fotos de esta campaña
        if fotos_camp:
            self._unique_photos = fotos_camp
            self.combo_photos.blockSignals(True)
            self.combo_photos.clear()
            for f_i, (c_n, f_n) in enumerate(fotos_camp):
                self.combo_photos.addItem(f"[{f_i + 1}/{len(fotos_camp)}] {f_n}")
            self.combo_photos.blockSignals(False)
            self._current_photo_idx = 0
            self.combo_photos.setCurrentIndex(0)
            self._load_current_photo()

        self._on_atlas_channel_changed(0)

    def _on_atlas_channel_changed(self, idx: int) -> None:
        """Actualiza el canal activo y recalcula las métricas biométricas."""
        self._recalc_atlas_biometrics()

    def _recalc_atlas_biometrics(self) -> None:
        """Calcula el IED real en mm y distancias a landmarks anatómicos, actualizando el marco 3:4."""
        if not hasattr(self, "combo_channels") or self.combo_channels.count() == 0:
            return
        if not hasattr(self, "combo_slides") or self._current_slide_idx >= len(self._atlas_configs):
            return

        cfg = self._atlas_configs[self._current_slide_idx]
        ch_idx = self.combo_channels.currentIndex()
        if ch_idx < 0 or ch_idx >= len(cfg["canales"]):
            return

        ch_cfg = cfg["canales"][ch_idx]
        col_csv = ch_cfg["col_csv"]
        ch_key = "CH0" if "0" in col_csv else ("CH1" if "1" in col_csv else "CH2")

        pts = [p for p in self.canvas.points if p["Canal"] == ch_key]
        if len(pts) >= 2:
            x1, y1 = float(pts[0]["X_px"]), float(pts[0]["Y_px"])
            x2, y2 = float(pts[1]["X_px"]), float(pts[1]["Y_px"])
            ied_px = math.hypot(x2 - x1, y2 - y1)
            ied_mm = ied_px * self._scale_px_to_mm
            c_dip = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            fid = self.canvas.fiduciarios_px
            p_menton = fid.get("Gnathion (Mentón)", c_dip)
            p_nariz = fid.get("Pronasale (Nariz)", c_dip)
            p_trago = fid.get("Tragus (Oreja)", c_dip)
            p_comisura = fid.get("Cheilion (Comisura)", c_dip)

            d_menton_mm = math.hypot(c_dip[0] - p_menton[0], c_dip[1] - p_menton[1]) * self._scale_px_to_mm
            d_nariz_mm = math.hypot(c_dip[0] - p_nariz[0], c_dip[1] - p_nariz[1]) * self._scale_px_to_mm
            d_trago_mm = math.hypot(c_dip[0] - p_trago[0], c_dip[1] - p_trago[1]) * self._scale_px_to_mm
            d_com_mm = math.hypot(c_dip[0] - p_comisura[0], c_dip[1] - p_comisura[1]) * self._scale_px_to_mm

            self.lbl_atlas_metrics.setText(
                f"Canal: {col_csv} - {ch_cfg['musculo_linea1']}\n"
                f"IED Bipolar: {ied_mm:.1f} mm ± 3.5 mm\n"
                f"Dist. a Mentón: {d_menton_mm:.1f} mm | Nariz: {d_nariz_mm:.1f} mm\n"
                f"Dist. a Trago: {d_trago_mm:.1f} mm | Comisura: {d_com_mm:.1f} mm"
            )

            # Actualizar marco de recorte 3:4 portrait centrado en el dipolo
            h_orig, w_orig = self.canvas._img_h, self.canvas._img_w
            target_h = float(self.spin_crop_height.value()) if hasattr(self, "spin_crop_height") else 380.0
            crop_h = min(float(h_orig), target_h)
            crop_w = crop_h * 0.75
            if crop_w > w_orig:
                crop_w = float(w_orig)
                crop_h = crop_w / 0.75

            rx1 = max(0.0, c_dip[0] - crop_w / 2.0)
            ry1 = max(0.0, c_dip[1] - crop_h / 2.0)
            if rx1 + crop_w > w_orig:
                rx1 = w_orig - crop_w
            if ry1 + crop_h > h_orig:
                ry1 = h_orig - crop_h
            rx1, ry1 = max(0.0, rx1), max(0.0, ry1)
            self.canvas.crop_rect_img = QtCore.QRectF(rx1, ry1, crop_w, crop_h)
        else:
            self.lbl_atlas_metrics.setText(
                f"Canal: {col_csv} ({ch_cfg['musculo_linea1']})\n"
                f"[Aviso] Se requieren al menos 2 electrodos marcados como {ch_key}."
            )
            self.canvas.crop_rect_img = None

        self.canvas.update()

    def _save_and_apply_atlas(self) -> None:
        """Genera el recorte nativo 3:4 portrait, guarda archivos de posición y recompila la diapositiva."""
        if not self._atlas_configs or self._current_slide_idx >= len(self._atlas_configs):
            QtWidgets.QMessageBox.warning(self, "Atlas", "No hay diapositiva seleccionada.")
            return

        cfg = self._atlas_configs[self._current_slide_idx]
        slide_id = cfg["id"]
        camp_dir = cfg["camp"]
        canales = cfg["canales"]

        ch_idx = self.combo_channels.currentIndex()
        if ch_idx < 0 or ch_idx >= len(canales):
            QtWidgets.QMessageBox.warning(self, "Atlas", "Seleccione un canal válido.")
            return

        ch_cfg = canales[ch_idx]
        col_csv = ch_cfg["col_csv"]
        m_nom = ch_cfg["musculo_linea1"]
        ch_key = "CH0" if "0" in col_csv else ("CH1" if "1" in col_csv else "CH2")

        pts = [p for p in self.canvas.points if p["Canal"] == ch_key]
        if len(pts) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "Atlas",
                f"El canal {col_csv} ({m_nom}) necesita al menos 2 electrodos marcados como {ch_key}."
            )
            return

        x1, y1 = float(pts[0]["X_px"]), float(pts[0]["Y_px"])
        x2, y2 = float(pts[1]["X_px"]), float(pts[1]["Y_px"])
        c_dip = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        ied_px = math.hypot(x2 - x1, y2 - y1)
        ied_mm = ied_px * self._scale_px_to_mm

        # Distancias a puntos anatómicos
        fid = self.canvas.fiduciarios_px
        p_menton = fid.get("Gnathion (Mentón)", c_dip)
        p_nariz = fid.get("Pronasale (Nariz)", c_dip)
        p_trago = fid.get("Tragus (Oreja)", c_dip)
        p_comisura = fid.get("Cheilion (Comisura)", c_dip)

        d_menton_mm = math.hypot(c_dip[0] - p_menton[0], c_dip[1] - p_menton[1]) * self._scale_px_to_mm
        d_nariz_mm = math.hypot(c_dip[0] - p_nariz[0], c_dip[1] - p_nariz[1]) * self._scale_px_to_mm
        d_trago_mm = math.hypot(c_dip[0] - p_trago[0], c_dip[1] - p_trago[1]) * self._scale_px_to_mm
        d_com_mm = math.hypot(c_dip[0] - p_comisura[0], c_dip[1] - p_comisura[1]) * self._scale_px_to_mm

        # Generar recorte 3:4 portrait nativo sin barras negras
        h_orig, w_orig = self.canvas._img_h, self.canvas._img_w
        target_h = float(self.spin_crop_height.value()) if hasattr(self, "spin_crop_height") else 380.0
        crop_h = int(round(min(float(h_orig), target_h)))
        crop_w = int(round(crop_h * 0.75))
        if crop_w > w_orig:
            crop_w = int(w_orig)
            crop_h = int(round(crop_w / 0.75))

        cx, cy = int(round(c_dip[0])), int(round(c_dip[1]))
        rx1 = max(0, cx - crop_w // 2)
        ry1 = max(0, cy - crop_h // 2)
        if rx1 + crop_w > w_orig:
            rx1 = w_orig - crop_w
        if ry1 + crop_h > h_orig:
            ry1 = h_orig - crop_h
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(w_orig, rx1 + crop_w), min(h_orig, ry1 + crop_h)

        crop_clean = self.canvas._img_bgr[ry1:ry2, rx1:rx2].copy()

        # Dibujar dipolo sobre el recorte
        p1_c = (int(round(x1 - rx1)), int(round(y1 - ry1)))
        p2_c = (int(round(x2 - rx1)), int(round(y2 - ry1)))
        col_ch = (68, 68, 239)

        cv2.line(crop_clean, p1_c, p2_c, col_ch, 2, cv2.LINE_AA)
        cv2.circle(crop_clean, p1_c, 14, col_ch, 2, cv2.LINE_AA)
        cv2.circle(crop_clean, p1_c, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(crop_clean, p1_c, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(crop_clean, "E1 (+)", (p1_c[0] - 60, p1_c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_ch, 1, cv2.LINE_AA)

        cv2.circle(crop_clean, p2_c, 14, col_ch, 2, cv2.LINE_AA)
        cv2.circle(crop_clean, p2_c, 10, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(crop_clean, p2_c, 4, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.putText(crop_clean, "E2 (-)", (p2_c[0] + 12, p2_c[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col_ch, 1, cv2.LINE_AA)

        # Tarjeta superior elegante
        ov = crop_clean.copy()
        cv2.rectangle(ov, (8, 8, 260, 52), (15, 23, 42), -1)
        cv2.addWeighted(ov, 0.85, crop_clean, 0.15, 0, crop_clean)
        cv2.rectangle(crop_clean, (8, 8, 260, 52), (51, 65, 85), 1)
        cv2.putText(crop_clean, f"{col_csv} {m_nom}", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (56, 189, 248), 1, cv2.LINE_AA)
        cv2.putText(crop_clean, f"Dipolo: {ied_mm:.1f} +- 3.5 mm", (14, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (251, 191, 36), 1, cv2.LINE_AA)

        # Guardar archivos
        nombre_base = f"curada_{col_csv.replace(' ', '_').lower()}_{m_nom.lower().replace(' ', '_')}"
        nombre_archivo = f"{nombre_base}_3x4.png"
        nombre_json = f"{nombre_base}_posiciones.json"

        out_foto_camp = os.path.join(camp_dir, "fotos_montaje", nombre_archivo)
        out_foto_art = os.path.join(OUT_DIR, f"{slide_id}_{nombre_archivo}")
        out_json_camp = os.path.join(camp_dir, "fotos_montaje", nombre_json)

        os.makedirs(os.path.dirname(out_foto_camp), exist_ok=True)
        cv2.imwrite(out_foto_camp, crop_clean)
        cv2.imwrite(out_foto_art, crop_clean)

        tam_elect = "5 x 5 mm (±0.02 mm)" if ("recortado" in m_nom.lower() or "5x5" in m_nom.lower()) else "10 mm (disco ag/agcl)"
        datos_guardar = {
            "diapositiva_id": slide_id,
            "canal": col_csv,
            "musculo": m_nom,
            "foto_origen": self._current_img_path,
            "dimensiones_origen": [w_orig, h_orig],
            "posiciones_px": {
                "E1": [x1, y1],
                "E2": [x2, y2],
                "centro_dipolo": list(c_dip),
                "crop_box": [int(rx1), int(ry1), int(rx2), int(ry2)]
            },
            "escala_px_to_mm": float(self._scale_px_to_mm),
            "ied_mm": round(ied_mm, 1),
            "distancias_anatomicas_mm": {
                "gnathion_menton": round(d_menton_mm, 1),
                "pronasale_nariz": round(d_nariz_mm, 1),
                "tragus_oreja": round(d_trago_mm, 1),
                "cheilion_comisura": round(d_com_mm, 1)
            },
            "tam_electrodo": tam_elect
        }
        with open(out_json_camp, "w", encoding="utf-8") as fj:
            json.dump(datos_guardar, fj, indent=4, ensure_ascii=False)

        # Actualizar diccionario del canal
        ch_cfg["foto"] = out_foto_art
        ch_cfg["tam_electrodo"] = tam_elect
        ch_cfg["ied"] = f"{ied_mm:.1f} ± 3.5 mm"
        ch_cfg["dist_menton"] = f"{d_menton_mm:.1f} ± 3.5 mm"
        ch_cfg["dist_nariz"] = f"{d_nariz_mm:.1f} ± 3.5 mm"
        ch_cfg["dist_trago"] = f"{d_trago_mm:.1f} ± 3.5 mm"
        ch_cfg["dist_comisura"] = f"{d_com_mm:.1f} ± 3.5 mm"

        # Recompilar Diapositiva Maestra en PDF y PNG
        png_out = os.path.join(OUT_DIR, f"diapositiva_{slide_id}.png")
        pdf_out = os.path.join(OUT_DIR, f"diapositiva_{slide_id}.pdf")
        try:
            compilar_diapositiva_maestra(
                titulo_experimento=cfg["titulo"],
                base_dir_camp=camp_dir,
                canales_cfg=canales,
                ses_list_curvas=cfg.get("ses_curvas", []),
                filtro_mediciones_pca=cfg.get("filtro_pca"),
                info_cuaderno=cfg.get("cuaderno", {}),
                png_out=png_out,
                pdf_out=pdf_out,
                subfolder_mediciones=cfg.get("subfolder", "mediciones")
            )
            QtWidgets.QMessageBox.information(
                self,
                "Atlas Actualizado",
                f"Diapositiva {slide_id} ({col_csv} {m_nom}) actualizada y compilada exitosamente en PDF y PNG.\n\nRecorte guardado en:\n{out_foto_camp}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al Compilar", f"Fallo al compilar diapositiva: {e}")

        # Avanzar al siguiente canal de la diapositiva si queda alguno
        if ch_idx + 1 < len(canales):
            self.combo_channels.setCurrentIndex(ch_idx + 1)
        else:
            self._next_atlas_slide()

    def _marcar_foto_perdida_atlas(self) -> None:
        """Marca el canal actual como FOTO_PERDIDA y recompila la diapositiva."""
        if not self._atlas_configs or self._current_slide_idx >= len(self._atlas_configs):
            return
        cfg = self._atlas_configs[self._current_slide_idx]
        ch_idx = self.combo_channels.currentIndex()
        if ch_idx < 0 or ch_idx >= len(cfg["canales"]):
            return
        ch_cfg = cfg["canales"][ch_idx]
        ch_cfg["foto"] = "FOTO_PERDIDA"
        ch_cfg["ied"] = "No disp. (Foto perdida)"
        ch_cfg["dist_menton"] = "No disp. (Foto perdida)"
        ch_cfg["dist_nariz"] = "No disp. (Foto perdida)"
        ch_cfg["dist_trago"] = "No disp. (Foto perdida)"
        ch_cfg["dist_comisura"] = "No disp. (Foto perdida)"

        slide_id = cfg["id"]
        png_out = os.path.join(OUT_DIR, f"diapositiva_{slide_id}.png")
        pdf_out = os.path.join(OUT_DIR, f"diapositiva_{slide_id}.pdf")
        try:
            compilar_diapositiva_maestra(
                titulo_experimento=cfg["titulo"],
                base_dir_camp=cfg["camp"],
                canales_cfg=cfg["canales"],
                ses_list_curvas=cfg.get("ses_curvas", []),
                filtro_mediciones_pca=cfg.get("filtro_pca"),
                info_cuaderno=cfg.get("cuaderno", {}),
                png_out=png_out,
                pdf_out=pdf_out,
                subfolder_mediciones=cfg.get("subfolder", "mediciones")
            )
            QtWidgets.QMessageBox.information(
                self,
                "Canal Actualizado",
                f"Canal {ch_cfg['col_csv']} ({ch_cfg['musculo_linea1']}) marcado como FOTO_PERDIDA y diapositiva recompilada."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

        if ch_idx + 1 < len(cfg["canales"]):
            self.combo_channels.setCurrentIndex(ch_idx + 1)
        else:
            self._next_atlas_slide()

    def _next_atlas_slide(self) -> None:
        """Avanza a la siguiente diapositiva del Atlas."""
        c_idx = self.combo_slides.currentIndex()
        if c_idx + 1 < len(self._atlas_configs):
            self.combo_slides.setCurrentIndex(c_idx + 1)
        else:
            QtWidgets.QMessageBox.information(self, "Atlas Completado", "Se han recorrido todas las diapositivas del Atlas.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dialog = CuradorInteractivoElectrodosDialog()
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()

    def _posicionar_campania_y_foto(camp_sub: str, foto_sub: str):
        """Posiciona la diapositiva del Atlas en la campania pedida (re-filtra el visor de fotos)
        y devuelve el indice de la foto solicitada dentro de esa campania, o None."""
        # 1. Buscar la diapositiva del Atlas de esa campania
        slide_idx = None
        for s_i, cfg in enumerate(dialog._atlas_configs):
            camp_b = os.path.basename(cfg["camp"]).lower()
            titulo_l = str(cfg["titulo"]).lower()
            if (camp_sub and camp_sub in camp_b) or (camp_sub and camp_sub in titulo_l):
                slide_idx = s_i
                break
        if slide_idx is None:
            print(f"[Aviso] No hay diapositiva del Atlas para '{camp_sub}'. Campanias disponibles:")
            for s_i, cfg in enumerate(dialog._atlas_configs):
                print(f"  [{s_i}] {os.path.basename(cfg['camp'])}")
            return None
        # 2. Cambiar la diapositiva -> re-filtra _unique_photos a esa campania
        dialog.combo_slides.setCurrentIndex(slide_idx)
        # 3. Buscar la foto pedida dentro de la campania
        if not foto_sub:
            return 0
        for f_i, (c, f) in enumerate(dialog._unique_photos):
            if foto_sub in f.lower():
                return f_i
        print(f"[Aviso] La campania no tiene foto con '{foto_sub}'. Fotos disponibles:")
        for f_i, (c, f) in enumerate(dialog._unique_photos):
            print(f"  [{f_i}] {f}")
        return None

    target_idx = None
    if len(sys.argv) > 1:
        camp_sub = sys.argv[1].lower()
        foto_sub = sys.argv[2].lower() if len(sys.argv) > 2 else ""
        target_idx = _posicionar_campania_y_foto(camp_sub, foto_sub)
    else:
        # Sin argumentos: elegir de TODO el CSV de fotos curadas y posicionar su campania
        opciones = [f"{camp}  -  {foto}" for camp, foto in dialog._all_csv_photos]
        sel, ok = QtWidgets.QInputDialog.getItem(
            dialog, "Curador de Electrodos",
            "Elegi la foto que queres curar:", opciones, 0, False)
        if ok and sel:
            camp_p, foto_p = dialog._all_csv_photos[opciones.index(sel)]
            target_idx = _posicionar_campania_y_foto(camp_p.lower(), foto_p.lower())

    if target_idx is not None and 0 <= target_idx < len(dialog._unique_photos):
        dialog.combo_photos.setCurrentIndex(target_idx)  # dispara _on_combo_changed -> carga la foto
        dialog._load_current_photo()

    sys.exit(app.exec())
