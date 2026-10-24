# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para visualizar datos crudos desde archivos CSV.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para visualizar datos crudos desde archivos CSV.
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QFileDialog, QScrollArea, QSplitter, 
    QSpinBox, QDoubleSpinBox, QSlider, QComboBox, QPlainTextEdit,
    QDialog, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
import pyqtgraph as pg
import pyqtgraph.exporters
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

emg_base_dir = os.path.dirname(root_dir)
if emg_base_dir not in sys.path:
    sys.path.append(emg_base_dir)

from utils.config_manager import ConfigManager
config_mgr = ConfigManager()

try:
    from analysis.filtro_adaptativo import cancelar_ruido_linea_adaptativo
except Exception:
    try:
        from filtro_adaptativo import cancelar_ruido_linea_adaptativo
    except Exception:
        cancelar_ruido_linea_adaptativo = None

# --- OPTIMIZACIÓN EXTREMA DE FLUIDEZ ---
# Antialiasing se ve "suave" pero devora la CPU al hacer zoom con cientos de miles de puntos.
pg.setConfigOptions(antialias=False) 

is_dark = config_mgr.get("estetica_global", "tema_oscuro")
if is_dark:
    pg.setConfigOption('background', '#050505')
    pg.setConfigOption('foreground', '#d3d3d3')
else:
    pg.setConfigOption('background', '#ffffff')
    pg.setConfigOption('foreground', '#000000')

MAX_POINTS_TO_PLOT = 25_000 # Límite optimizado para LTTB (muy fluido y preserva picos)

def downsample_lttb_fast(x, y, threshold):
    if len(x) <= threshold: return x, y
    n_blocks = threshold // 2
    block_size = len(x) // n_blocks
    trunc_len = n_blocks * block_size
    x_trunc = x[:trunc_len]
    y_trunc = y[:trunc_len]
    x_blocks = x_trunc.reshape(n_blocks, block_size)
    y_blocks = y_trunc.reshape(n_blocks, block_size)
    min_idxs = np.argmin(y_blocks, axis=1)
    max_idxs = np.argmax(y_blocks, axis=1)
    offsets = np.arange(n_blocks) * block_size
    abs_min_idxs = min_idxs + offsets
    abs_max_idxs = max_idxs + offsets
    all_idxs = np.concatenate((abs_min_idxs, abs_max_idxs))
    all_idxs.sort()
    if all_idxs[0] != 0: all_idxs = np.insert(all_idxs, 0, 0)
    if all_idxs[-1] != len(x) - 1: all_idxs = np.append(all_idxs, len(x) - 1)
    return x[all_idxs], y[all_idxs]

class DataLoaderThread(QThread):
    finished_loading = Signal(object, list, str, float) # df, cols_canales, time_col, global_max_y
    error_loading = Signal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            target_path = self.filepath
            with open(target_path, 'r', encoding='utf-8', errors='ignore') as f_peek:
                first_line = f_peek.readline()
            if first_line.lstrip().startswith('#'):
                df = pd.read_csv(target_path, comment='#')
            else:
                df = pd.read_csv(target_path)
            df.columns = df.columns.str.strip()
            time_col = None
            for c in ['Time', 'Tiempo', 'Time(s)', 'Tiempo (s)', 'Time (s)']:
                if c in df.columns:
                    time_col = c
                    break
            
            # Si se seleccionó calibracion.csv y no tiene columna de tiempo pero existe grabacion.csv o calibracion_raw.csv:
            if not time_col and os.path.basename(target_path).lower().startswith("calibracion"):
                med_dir = os.path.dirname(target_path)
                for alt_name in ("grabacion.csv", "calibracion_raw.csv"):
                    alt_path = os.path.join(med_dir, alt_name)
                    if os.path.isfile(alt_path):
                        try:
                            df_alt = pd.read_csv(alt_path, comment='#')
                            df_alt.columns = df_alt.columns.str.strip()
                            for c in ['Time', 'Tiempo', 'Time(s)', 'Tiempo (s)', 'Time (s)']:
                                if c in df_alt.columns:
                                    df = df_alt
                                    time_col = c
                                    target_path = alt_path
                                    break
                            if time_col:
                                break
                        except Exception:
                            pass

            if not time_col and len(df.columns) > 0:
                time_col = df.columns[0]
            
            canales = [c for c in df.columns if c != time_col]
            
            # Filtrar y asegurar que los canales sean numéricos (tolerante a filas de texto/resumen en calibracion.csv)
            canales_validos = []
            for c in canales:
                try:
                    s_num = pd.to_numeric(df[c], errors='coerce')
                    if s_num.notna().sum() > 0:
                        df[c] = s_num.fillna(0.0)
                        canales_validos.append(c)
                except Exception:
                    pass
            canales = canales_validos

            # Normalizar vector de tiempo si contiene texto o NaNs
            if time_col:
                s_t = pd.to_numeric(df[time_col], errors='coerce')
                if s_t.notna().sum() > 0:
                    df[time_col] = s_t.ffill().fillna(0.0)
                else:
                    df[time_col] = np.arange(len(df), dtype=float)

            if not canales:
                raise ValueError("El archivo seleccionado no contiene señales numéricas válidas para el visor.")

            # Calibración a microvoltios
            measurement_dir = os.path.dirname(target_path)
            if measurement_dir and os.path.basename(measurement_dir).startswith("canal_"):
                parent_dir = os.path.dirname(measurement_dir)
                if os.path.exists(os.path.join(parent_dir, "metadata.json")) or os.path.exists(os.path.join(parent_dir, "fotometria")):
                    measurement_dir = parent_dir

            meta_path = os.path.join(measurement_dir, 'metadata.json')
            if not os.path.exists(meta_path):
                # Fallback para sesiones de AutoForge guardadas por canal
                meta_canal0 = os.path.join(measurement_dir, 'canal_0', 'metadata.json')
                if os.path.exists(meta_canal0):
                    meta_path = meta_canal0

            resistencia_ohm = 100.0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        if 'resistencia_ohm' in meta:
                            resistencia_ohm = float(meta['resistencia_ohm'])
                except Exception:
                    pass
            
            r_fija = 49400.0
            ganancia = 1.0 + (r_fija / resistencia_ohm)
            
            # Convertir canales a microvoltios solo si no están ya expresados en microvoltios
            ya_en_uV = any('_uv' in c.lower() or 'microvolt' in c.lower() for c in canales)
            if not ya_en_uV:
                for c in canales:
                    df[c] = (df[c] / ganancia) * 1e6

            global_max_y = 1.0
            if len(canales) > 0:
                global_max_y = float(np.max(np.abs(df[canales].values)))
                if np.isnan(global_max_y) or global_max_y == 0:
                    global_max_y = 1.0

            self.finished_loading.emit(df, canales, time_col, global_max_y)
        except Exception as e:
            self.error_loading.emit(str(e))

class PhotometryGalleryDialog(QDialog):
    """
    Diálogo para explorar las capturas de fotometría 3D (ref_frontal_triangulada.png,
    ref_frontal_cruda.jpg, ref_perfil_45.jpg, capturas_toma/*.jpg) y visualizar la
    tabla de telemetría facial (ángulos yaw, pitch, roll, apertura labial y coordenadas
    3D de electrodos).
    """
    def __init__(self, medicion_dir: str, parent=None):
        super().__init__(parent)
        self.medicion_dir = medicion_dir
        self.fotometria_dir = os.path.join(medicion_dir, "fotometria") if medicion_dir else ""
        self._current_pixmap = None

        med_name = os.path.basename(medicion_dir) if medicion_dir else "Medición"
        self.setWindowTitle(f"📷 Fotometría 3D y Telemetría Facial — {med_name}")
        self.resize(1180, 740)
        self.setStyleSheet(
            "QDialog { background-color: #0d1117; color: #e6edf3; }"
            "QGroupBox { font-weight: bold; color: #00ffcc; border: 1px solid #263545; border-radius: 6px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }"
            "QListWidget { background-color: #141b22; color: #e6edf3; border: 1px solid #263545; border-radius: 4px; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #1c2631; }"
            "QListWidget::item:selected { background-color: #14362d; color: #00ffcc; font-weight: bold; }"
            "QTableWidget { background-color: #141b22; color: #e6edf3; gridline-color: #263545; border: 1px solid #263545; font-size: 12px; }"
            "QHeaderView::section { background-color: #1b2430; color: #00ffcc; font-weight: bold; padding: 5px; border: 1px solid #263545; }"
        )

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Resumen superior de telemetría y calibración
        self.lbl_summary = QLabel("Cargando resumen de fotometría y calibración espacial 3D...")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet(
            "background-color: #14261f; border-left: 4px solid #00ffcc; color: #e0f7f4; "
            "padding: 8px 12px; border-radius: 4px; font-size: 13px;"
        )
        main_layout.addWidget(self.lbl_summary)

        splitter_main = QSplitter(Qt.Horizontal)

        # --- PANEL IZQUIERDO: Galería de Imágenes ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        grp_gallery = QGroupBox("🖼️ Capturas de Referencia 3D y Tomas de Sesión")
        lyt_gallery = QHBoxLayout(grp_gallery)

        self.list_images = QListWidget()
        self.list_images.setMinimumWidth(230)
        self.list_images.setMaximumWidth(310)
        self.list_images.currentItemChanged.connect(self._on_image_selected)
        lyt_gallery.addWidget(self.list_images)

        preview_container = QWidget()
        lyt_preview = QVBoxLayout(preview_container)
        lyt_preview.setContentsMargins(4, 0, 0, 0)

        self.lbl_image_preview = QLabel("Seleccione una imagen de la lista")
        self.lbl_image_preview.setAlignment(Qt.AlignCenter)
        self.lbl_image_preview.setMinimumSize(420, 360)
        self.lbl_image_preview.setStyleSheet(
            "background-color: #080b0f; border: 1px solid #263545; border-radius: 6px; color: #888;"
        )
        lyt_preview.addWidget(self.lbl_image_preview, stretch=1)

        self.lbl_image_info = QLabel("")
        self.lbl_image_info.setAlignment(Qt.AlignCenter)
        self.lbl_image_info.setStyleSheet("color: #9ab; font-size: 12px; padding: 3px;")
        lyt_preview.addWidget(self.lbl_image_info)

        lyt_gallery.addWidget(preview_container, stretch=1)
        left_layout.addWidget(grp_gallery)
        splitter_main.addWidget(left_widget)

        # --- PANEL DERECHO: Tablas de Coordenadas 3D y Telemetría ---
        right_splitter = QSplitter(Qt.Vertical)

        # Tabla 1: Coordenadas 3D de Electrodos (calibracion_espacial.json)
        grp_elec = QGroupBox("📍 Coordenadas 3D de Electrodos (calibracion_espacial.json)")
        lyt_elec = QVBoxLayout(grp_elec)
        self.tbl_electrodos = QTableWidget()
        self.tbl_electrodos.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_electrodos.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_electrodos.setAlternatingRowColors(True)
        lyt_elec.addWidget(self.tbl_electrodos)
        right_splitter.addWidget(grp_elec)

        # Tabla 2: Telemetría Facial (Yaw, Pitch, Roll, Apertura Labial)
        grp_telem = QGroupBox("📊 Telemetría Facial: Pose 3D (Yaw, Pitch, Roll) y Apertura Labial")
        lyt_telem = QVBoxLayout(grp_telem)
        self.tbl_telemetria = QTableWidget()
        self.tbl_telemetria.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_telemetria.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_telemetria.setAlternatingRowColors(True)
        self.tbl_telemetria.itemSelectionChanged.connect(self._on_telemetry_row_selected)
        lyt_telem.addWidget(self.tbl_telemetria)
        right_splitter.addWidget(grp_telem)

        right_splitter.setSizes([240, 380])
        splitter_main.addWidget(right_splitter)
        splitter_main.setSizes([640, 520])
        main_layout.addWidget(splitter_main, stretch=1)

        # Botón cerrar
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_close = QPushButton("Cerrar Visor")
        btn_close.setStyleSheet(
            "QPushButton { background-color: #1f2937; color: #00ffcc; border: 1px solid #00ffcc; "
            "border-radius: 4px; padding: 6px 18px; font-weight: bold; }"
            "QPushButton:hover { background-color: #374151; }"
        )
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

        self._load_images()
        self._load_calibration_and_telemetry()

    def _load_images(self):
        self.list_images.clear()
        if not self.fotometria_dir or not os.path.isdir(self.fotometria_dir):
            return

        added_paths = set()
        priority_files = [
            ("ref_frontal_triangulada.png", "🎯 Ref. Frontal Triangulada"),
            ("ref_frontal_triangulada.jpg", "🎯 Ref. Frontal Triangulada"),
            ("ref_frontal_cruda.jpg", "📷 Ref. Frontal Cruda"),
            ("ref_frontal_cruda.png", "📷 Ref. Frontal Cruda"),
            ("ref_perfil_45.jpg", "📐 Ref. Perfil 45°"),
            ("ref_perfil_45.png", "📐 Ref. Perfil 45°"),
        ]

        for fname, label_prefix in priority_files:
            fpath = os.path.join(self.fotometria_dir, fname)
            if os.path.isfile(fpath) and fpath not in added_paths:
                item = QListWidgetItem(f"{label_prefix} ({fname})")
                item.setData(Qt.UserRole, fpath)
                self.list_images.addItem(item)
                added_paths.add(fpath)

        # Otros archivos de imagen en la raíz de fotometria/
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        try:
            for fname in sorted(os.listdir(self.fotometria_dir)):
                fpath = os.path.join(self.fotometria_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(valid_exts) and fpath not in added_paths:
                    item = QListWidgetItem(f"🖼️ {fname}")
                    item.setData(Qt.UserRole, fpath)
                    self.list_images.addItem(item)
                    added_paths.add(fpath)
        except Exception:
            pass

        # Capturas dentro de capturas_toma/
        # Capturas dentro de capturas_toma/ o capturas/
        for sub_name in ("capturas_toma", "capturas"):
            capturas_dir = os.path.join(self.fotometria_dir, sub_name)
            if os.path.isdir(capturas_dir):
                try:
                    for fname in sorted(os.listdir(capturas_dir)):
                        fpath = os.path.join(capturas_dir, fname)
                        if os.path.isfile(fpath) and fname.lower().endswith(valid_exts) and fpath not in added_paths:
                            item = QListWidgetItem(f"📸 Toma: {fname}")
                            item.setData(Qt.UserRole, fpath)
                            self.list_images.addItem(item)
                            added_paths.add(fpath)
                except Exception:
                    pass

        # Si aún no hay fotos, buscar en fotos/ de la medición
        if self.list_images.count() == 0 and self.medicion_dir:
            fotos_dir = os.path.join(self.medicion_dir, "fotos")
            if os.path.isdir(fotos_dir):
                try:
                    for fname in sorted(os.listdir(fotos_dir)):
                        fpath = os.path.join(fotos_dir, fname)
                        if os.path.isfile(fpath) and fname.lower().endswith(valid_exts) and fpath not in added_paths:
                            item = QListWidgetItem(f"📸 {fname}")
                            item.setData(Qt.UserRole, fpath)
                            self.list_images.addItem(item)
                            added_paths.add(fpath)
                except Exception:
                    pass

        if self.list_images.count() > 0:
            self.list_images.setCurrentRow(0)
        else:
            self.lbl_image_preview.setText("No se encontraron imágenes en la carpeta fotometria/")

    def _on_image_selected(self, current: QListWidgetItem, _previous: QListWidgetItem):
        if not current:
            return
        img_path = current.data(Qt.UserRole)
        self._show_image_from_path(img_path)
        if img_path:
            self._sync_telemetry_table_with_image(os.path.basename(img_path))

    def _show_image_from_path(self, img_path: str):
        if not img_path or not os.path.isfile(img_path):
            self.lbl_image_preview.setText("Archivo de imagen no encontrado")
            self._current_pixmap = None
            self.lbl_image_info.setText("")
            return

        pix = QPixmap(img_path)
        if pix.isNull():
            self.lbl_image_preview.setText("No se pudo decodificar la imagen")
            self._current_pixmap = None
            self.lbl_image_info.setText("")
            return

        self._current_pixmap = pix
        self._update_preview_pixmap()
        rel_name = os.path.relpath(img_path, self.fotometria_dir) if self.fotometria_dir else os.path.basename(img_path)
        self.lbl_image_info.setText(f"{rel_name} — {pix.width()}×{pix.height()} px")

    def _sync_telemetry_table_with_image(self, base_name: str):
        if not base_name or self.tbl_telemetria.rowCount() == 0:
            return
        self.tbl_telemetria.blockSignals(True)
        try:
            for r in range(self.tbl_telemetria.rowCount()):
                for c in range(self.tbl_telemetria.columnCount()):
                    cell = self.tbl_telemetria.item(r, c)
                    if cell and os.path.basename(cell.text().strip()) == base_name:
                        self.tbl_telemetria.selectRow(r)
                        self.tbl_telemetria.scrollToItem(cell)
                        return
        finally:
            self.tbl_telemetria.blockSignals(False)

    def _update_preview_pixmap(self):
        if self._current_pixmap and not self._current_pixmap.isNull():
            scaled = self._current_pixmap.scaled(
                self.lbl_image_preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.lbl_image_preview.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _on_telemetry_row_selected(self):
        selected = self.tbl_telemetria.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        for col in range(self.tbl_telemetria.columnCount()):
            cell = self.tbl_telemetria.item(row, col)
            if not cell:
                continue
            val = cell.text().strip()
            if val.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                base_target = os.path.basename(val)
                self.list_images.blockSignals(True)
                try:
                    for idx in range(self.list_images.count()):
                        it = self.list_images.item(idx)
                        path_it = it.data(Qt.UserRole) or ""
                        if os.path.basename(path_it) == base_target:
                            self.list_images.setCurrentRow(idx)
                            self._show_image_from_path(path_it)
                            return
                finally:
                    self.list_images.blockSignals(False)

    def _load_calibration_and_telemetry(self):
        summary_parts = []

        # 1. Cargar calibracion_espacial.json
        cal_data = {}
        cal_candidates = [
            os.path.join(self.fotometria_dir, "calibracion_espacial.json"),
            os.path.join(self.medicion_dir, "calibracion_espacial.json"),
        ]
        for cpath in cal_candidates:
            if cpath and os.path.isfile(cpath):
                try:
                    with open(cpath, "r", encoding="utf-8") as f:
                        cal_data = json.load(f)
                    break
                except Exception:
                    pass

        meta_esp = cal_data.get("metadatos_calibracion_espacial", cal_data) if isinstance(cal_data, dict) else {}
        if not isinstance(meta_esp, dict):
            meta_esp = {}

        electrodos = meta_esp.get("electrodos", [])
        if not isinstance(electrodos, list):
            electrodos = []

        dist_ip_px = meta_esp.get("distancia_interpupilar_px")
        dist_ip_3d = meta_esp.get("distancia_interpupilar_3d")
        if dist_ip_px is not None:
            try:
                summary_parts.append(f"<b>D. Interpupilar:</b> {float(dist_ip_px):.1f} px")
            except Exception:
                summary_parts.append(f"<b>D. Interpupilar:</b> {dist_ip_px}")
        if dist_ip_3d is not None:
            try:
                summary_parts.append(f"<b>Escala 3D Base:</b> {float(dist_ip_3d):.4f}")
            except Exception:
                pass
        summary_parts.append(f"<b>Electrodos 3D:</b> {len(electrodos)}")

        # Poblar tabla de electrodos 3D
        headers_el = [
            "Canal", "Músculo Asignado/Estimado", "Centroide 2D (px)",
            "Vértice 3D", "Vector 3D Rel. Norm. (X, Y, Z)", "Dist. 3D Norm."
        ]
        self.tbl_electrodos.setColumnCount(len(headers_el))
        self.tbl_electrodos.setHorizontalHeaderLabels(headers_el)
        self.tbl_electrodos.setRowCount(len(electrodos))

        for r_idx, el in enumerate(electrodos):
            if not isinstance(el, dict):
                continue
            canal_str = str(el.get("canal", f"CH_{el.get('id', r_idx)}"))
            phys = el.get("canal_fisico_ai")
            if phys and str(phys).lower() != str(canal_str).lower():
                canal_str = f"{canal_str} ({phys})"

            musc_str = str(el.get("musculo_asignado") or el.get("musculo_estimado") or el.get("musculo", "-"))

            c2d = el.get("centroide_2d_px", el.get("coordenadas_2d_px", []))
            if isinstance(c2d, (list, tuple)) and len(c2d) >= 2:
                try:
                    c2d_str = f"({float(c2d[0]):.1f}, {float(c2d[1]):.1f})"
                except Exception:
                    c2d_str = f"({c2d[0]}, {c2d[1]})"
            else:
                c2d_str = str(c2d)

            vert_str = str(el.get("vertice_malla_3d", el.get("landmark_id", "-")))

            v3d = el.get(
                "vector_3d_relativo_normalizado",
                el.get("coordenadas_3d_absolutas", el.get("coordenadas_3d", []))
            )
            if isinstance(v3d, (list, tuple)) and len(v3d) >= 3:
                try:
                    v3d_str = f"({float(v3d[0]):.4f}, {float(v3d[1]):.4f}, {float(v3d[2]):.4f})"
                except Exception:
                    v3d_str = str(v3d)
            else:
                v3d_str = str(v3d)

            d_norm = el.get("distancia_3d_normalizada", el.get("distancia_3d_al_origen", None))
            if d_norm is not None:
                try:
                    d_norm_str = f"{float(d_norm):.4f}"
                except Exception:
                    d_norm_str = str(d_norm)
            else:
                d_norm_str = "-"

            row_vals = [canal_str, musc_str, c2d_str, vert_str, v3d_str, d_norm_str]
            for c_idx, val in enumerate(row_vals):
                it = QTableWidgetItem(val)
                it.setTextAlignment(Qt.AlignCenter)
                self.tbl_electrodos.setItem(r_idx, c_idx, it)

        self.tbl_electrodos.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # 2. Cargar fotometria_telemetria.csv (o telemetría embebida en JSON)
        telem_csv_candidates = [
            os.path.join(self.fotometria_dir, "fotometria_telemetria.csv"),
            os.path.join(self.medicion_dir, "fotometria_telemetria.csv"),
        ]
        df_telem = None
        for tpath in telem_csv_candidates:
            if tpath and os.path.isfile(tpath):
                try:
                    df_telem = pd.read_csv(tpath, comment='#')
                    df_telem.columns = df_telem.columns.str.strip()
                    break
                except Exception:
                    pass

        if df_telem is not None and not df_telem.empty:
            cols = list(df_telem.columns)
            self.tbl_telemetria.setColumnCount(len(cols))
            self.tbl_telemetria.setHorizontalHeaderLabels(cols)
            self.tbl_telemetria.setRowCount(len(df_telem))

            for r_idx in range(len(df_telem)):
                for c_idx, col_name in enumerate(cols):
                    val = df_telem.iloc[r_idx, c_idx]
                    if isinstance(val, (float, np.floating)):
                        text_val = "-" if np.isnan(val) else f"{float(val):.4f}"
                    else:
                        text_val = str(val)
                    it = QTableWidgetItem(text_val)
                    it.setTextAlignment(Qt.AlignCenter)
                    self.tbl_telemetria.setItem(r_idx, c_idx, it)

            self.tbl_telemetria.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            self.tbl_telemetria.horizontalHeader().setStretchLastSection(True)

            # Extraer promedios de yaw, pitch, roll y apertura labial para el resumen
            col_map = {c.lower(): c for c in cols}
            pose_strs = []
            for angle_key, label_ang in [("yaw", "Yaw"), ("pitch", "Pitch"), ("roll", "Roll")]:
                matched_col = next((col_map[k] for k in col_map if angle_key in k), None)
                if matched_col:
                    series_num = pd.to_numeric(df_telem[matched_col], errors='coerce').dropna()
                    if not series_num.empty:
                        pose_strs.append(f"{label_ang}: {series_num.mean():.1f}° (±{series_num.std():.1f}°)")
            if pose_strs:
                summary_parts.append("<b>Pose Cabeza:</b> " + " | ".join(pose_strs))

            lip_col = next(
                (col_map[k] for k in col_map if "apertura" in k or "labial" in k or "lip" in k),
                None
            )
            if lip_col:
                lip_series = pd.to_numeric(df_telem[lip_col], errors='coerce').dropna()
                if not lip_series.empty:
                    summary_parts.append(
                        f"<b>Apertura Labial ({lip_col}):</b> media={lip_series.mean():.4f} (máx={lip_series.max():.4f})"
                    )
        else:
            # Fallback: mostrar métricas de pose / apertura desde calibracion_espacial.json si existen
            fallback_rows = []
            pose_dict = meta_esp.get("pose_cabeza", meta_esp.get("angulos_cabeza", {}))
            if isinstance(pose_dict, dict):
                for k, v in pose_dict.items():
                    fallback_rows.append((f"Pose: {k}", str(v)))
            for key_direct in ("yaw", "pitch", "roll", "apertura_labial", "apertura_labial_norm", "separabilidad_o_u"):
                if key_direct in meta_esp:
                    fallback_rows.append((key_direct, str(meta_esp[key_direct])))
                elif isinstance(cal_data, dict) and key_direct in cal_data:
                    fallback_rows.append((key_direct, str(cal_data[key_direct])))

            self.tbl_telemetria.setColumnCount(2)
            self.tbl_telemetria.setHorizontalHeaderLabels(["Parámetro Fotométrico", "Valor"])
            self.tbl_telemetria.setRowCount(len(fallback_rows))
            for r_idx, (k_str, v_str) in enumerate(fallback_rows):
                it_k = QTableWidgetItem(str(k_str))
                it_v = QTableWidgetItem(str(v_str))
                it_k.setTextAlignment(Qt.AlignCenter)
                it_v.setTextAlignment(Qt.AlignCenter)
                self.tbl_telemetria.setItem(r_idx, 0, it_k)
                self.tbl_telemetria.setItem(r_idx, 1, it_v)
            self.tbl_telemetria.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.lbl_summary.setText(" &nbsp;•&nbsp; ".join(summary_parts) if summary_parts else "Sin metadatos de telemetría adicionales.")


class CsvViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.canales_originales = {}
        self.time_data = None
        self.global_max_y = 100.0
        self._updating_sliders = False
        self.current_filepath = None
        self.current_medicion_dir = None
        
        canales_conf = config_mgr.get("canales")
        self.channel_colors = []
        self.canal_nombres = []
        for i in range(16):
            c_key = f"Canal {i}"
            if c_key in canales_conf:
                self.channel_colors.append(canales_conf[c_key].get("color_hex", "#ffffff"))
                self.canal_nombres.append(canales_conf[c_key].get("musculo", c_key))
            else:
                self.channel_colors.append('#0074D9')
                self.canal_nombres.append(c_key)

        self._init_ui()

    def _init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_status = QLabel("Listo. Seleccione una medición en el Gestor de Sesiones (Izquierda).")
        self.lbl_status.setStyleSheet("color: #888; font-family: monospace; padding: 5px;")
        self.layout.addWidget(self.lbl_status)
        
        self.lbl_filename = QLabel("Ningún archivo cargado")
        self.lbl_filename.setAlignment(Qt.AlignCenter)
        self.lbl_filename.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffcc; padding: 10px;")
        self.layout.addWidget(self.lbl_filename)

        self.lbl_notas_banner = QLabel("")
        self.lbl_notas_banner.setWordWrap(True)
        self.lbl_notas_banner.setStyleSheet(
            "QLabel {"
            " background-color: #14261f;"
            " border-left: 4px solid #00ffcc;"
            " color: #e0f7f4;"
            " padding: 8px 12px;"
            " border-radius: 4px;"
            " font-size: 13px;"
            " margin: 2px 6px 6px 6px;"
            "}"
        )
        self.lbl_notas_banner.setVisible(False)
        self.layout.addWidget(self.lbl_notas_banner)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setClipToView(True) # OPTIMIZACIÓN: Solo dibuja los puntos en pantalla
        self.plot_widget.setLabel('bottom', 'Tiempo', units='s')
        self.plot_widget.setLabel('left', 'Amplitud', units='µV')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.sigXRangeChanged.connect(self._on_plot_xrange_changed)
        self.plot_widget.sigYRangeChanged.connect(self._on_plot_yrange_changed)
        splitter.addWidget(self.plot_widget)
        
        # Panel Derecho
        self.ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(self.ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        scroll_content = QWidget()
        self.slayout = QVBoxLayout(scroll_content)
        self.slayout.setSpacing(15)
        
        # Canales
        self.grp_channels = QGroupBox("Canales")
        self.grp_channels.setStyleSheet("font-weight: bold; color: #aaa;")
        self.lyt_channels = QVBoxLayout(self.grp_channels)
        self.slayout.addWidget(self.grp_channels)

        # Notas del Experimento y Fotometría 3D
        self.grp_notas = QGroupBox("📝 Notas del Experimento")
        self.grp_notas.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_notas = QVBoxLayout(self.grp_notas)

        self.txt_notas_sesion = QPlainTextEdit()
        self.txt_notas_sesion.setFixedHeight(90)
        self.txt_notas_sesion.setPlaceholderText("Agregue o edite notas de esta medición...")
        self.txt_notas_sesion.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #11181c;"
            " color: #e0e0e0;"
            " border: 1px solid #2a3b47;"
            " border-radius: 4px;"
            " padding: 6px;"
            " font-weight: normal;"
            " font-size: 12px;"
            "}"
        )
        lyt_notas.addWidget(self.txt_notas_sesion)

        self.btn_guardar_notas = QPushButton("💾 Guardar Notas en Medición")
        self.btn_guardar_notas.setStyleSheet(
            "QPushButton {"
            " background-color: #1b4332;"
            " color: #00ffcc;"
            " border: 1px solid #00ffcc;"
            " border-radius: 4px;"
            " padding: 6px;"
            " font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #2d6a4f; }"
        )
        self.btn_guardar_notas.clicked.connect(self._save_session_notes)
        lyt_notas.addWidget(self.btn_guardar_notas)

        self.btn_ver_fotometria = QPushButton("📷 Ver Fotometría / Capturas 3D")
        self.btn_ver_fotometria.setStyleSheet(
            "QPushButton {"
            " background-color: #1a2a3a;"
            " color: #4da8da;"
            " border: 1px solid #4da8da;"
            " border-radius: 4px;"
            " padding: 6px;"
            " font-weight: bold;"
            "}"
            "QPushButton:hover { background-color: #243b53; }"
            "QPushButton:disabled { background-color: #1a1a1a; color: #555; border: 1px solid #333; }"
        )
        self.btn_ver_fotometria.setEnabled(False)
        self.btn_ver_fotometria.setVisible(False)
        self.btn_ver_fotometria.clicked.connect(self._open_photometry_viewer)
        lyt_notas.addWidget(self.btn_ver_fotometria)

        self.slayout.addWidget(self.grp_notas)
        
        # Navegación (Sliders como el original)
        grp_nav = QGroupBox("Navegación")
        grp_nav.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_nav = QVBoxLayout(grp_nav)
        
        lyt_nav.addWidget(QLabel("Posición (s):"))
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 100)
        self.slider_time.valueChanged.connect(self._on_time_slider_changed)
        lyt_nav.addWidget(self.slider_time)
        
        lyt_nav.addWidget(QLabel("Posición Y:"))
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 100)
        self.slider_y.setValue(50)
        self.slider_y.valueChanged.connect(self._on_y_slider_changed)
        lyt_nav.addWidget(self.slider_y)
        self.slayout.addWidget(grp_nav)
        
        # Zoom
        grp_zoom = QGroupBox("Zoom")
        grp_zoom.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_zoom = QVBoxLayout(grp_zoom)
        
        lyt_zoom.addWidget(QLabel("Duración en Pantalla (s):"))
        self.spin_zoom_x = QDoubleSpinBox()
        self.spin_zoom_x.setRange(0.1, 1000)
        self.spin_zoom_x.setValue(5.0)
        self.spin_zoom_x.valueChanged.connect(self._apply_view_ranges)
        lyt_zoom.addWidget(self.spin_zoom_x)
        
        lyt_zoom.addWidget(QLabel("Amplitud (%):"))
        self.slider_zoom_y = QSlider(Qt.Horizontal)
        self.slider_zoom_y.setRange(1, 100)
        self.slider_zoom_y.setValue(100)
        self.slider_zoom_y.valueChanged.connect(self._apply_view_ranges)
        lyt_zoom.addWidget(self.slider_zoom_y)
        self.btn_autoscale = QPushButton("Ajuste Automático")
        self.btn_autoscale.clicked.connect(self._autoscale)
        lyt_zoom.addWidget(self.btn_autoscale)
        self.slayout.addWidget(grp_zoom)
        
        # Filtros
        self.grp_filters = QGroupBox("Filtros")
        self.grp_filters.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_filt = QVBoxLayout(self.grp_filters)
        
        self.chk_adaptive = QCheckBox("Filtro Adaptativo")
        self.chk_adaptive.stateChanged.connect(self.update_plot)
        lyt_filt.addWidget(self.chk_adaptive)
        
        row_notch = QHBoxLayout()
        self.chk_notch = QCheckBox("Notch 50Hz")
        self.chk_notch.stateChanged.connect(self.update_plot)
        row_notch.addWidget(self.chk_notch)
        
        row_notch.addWidget(QLabel("Q:"))
        self.spin_notch_q = QDoubleSpinBox()
        self.spin_notch_q.setRange(0.1, 100.0)
        self.spin_notch_q.setValue(2.0)
        self.spin_notch_q.setSingleStep(1.0)
        self.spin_notch_q.editingFinished.connect(self.update_plot)
        row_notch.addWidget(self.spin_notch_q)
        
        lyt_filt.addLayout(row_notch)
        
        self.chk_bandpass = QCheckBox("Pasa-Banda")
        self.chk_bandpass.setChecked(True)
        self.chk_bandpass.stateChanged.connect(self.update_plot)
        lyt_filt.addWidget(self.chk_bandpass)
        
        row_hp = QHBoxLayout()
        row_hp.addWidget(QLabel("Highpass (Hz):"))
        self.spin_hp = QDoubleSpinBox()
        self.spin_hp.setRange(0, 1000)
        self.spin_hp.setValue(20) # Por defecto 20Hz
        row_hp.addWidget(self.spin_hp)
        lyt_filt.addLayout(row_hp)
        
        row_lp = QHBoxLayout()
        row_lp.addWidget(QLabel("Lowpass (Hz):"))
        self.spin_lp = QDoubleSpinBox()
        self.spin_lp.setRange(0, 10000)
        self.spin_lp.setValue(500) # Por defecto 500Hz
        row_lp.addWidget(self.spin_lp)
        lyt_filt.addLayout(row_lp)
        
        self.spin_lp.editingFinished.connect(self.update_plot)
        self.spin_hp.editingFinished.connect(self.update_plot)
        self.slayout.addWidget(self.grp_filters)
        
        # Envolvente
        grp_env = QGroupBox("Envolvente")
        grp_env.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_env = QVBoxLayout(grp_env)
        self.cmb_env = QComboBox()
        self.cmb_env.addItems(["ninguna", "media_movil", "rms"])
        self.cmb_env.currentTextChanged.connect(self._on_config_changed)
        lyt_env.addWidget(self.cmb_env)
        row_env = QHBoxLayout()
        row_env.addWidget(QLabel("Ventana (ms):"))
        self.spin_env = QSpinBox()
        self.spin_env.setRange(1, 5000)
        self.spin_env.setValue(50)
        self.spin_env.editingFinished.connect(self._on_config_changed)
        row_env.addWidget(self.spin_env)
        lyt_env.addLayout(row_env)
        
        self.chk_env_offset = QCheckBox("Eliminar Offset Base")
        self.chk_env_offset.stateChanged.connect(self._on_config_changed)
        lyt_env.addWidget(self.chk_env_offset)
        
        self.slayout.addWidget(grp_env)
        
        self.chk_adaptive.stateChanged.connect(self._on_config_changed)
        self.chk_notch.stateChanged.connect(self._on_config_changed)
        self.chk_bandpass.stateChanged.connect(self._on_config_changed)
        self.spin_lp.editingFinished.connect(self._on_config_changed)
        self.spin_hp.editingFinished.connect(self._on_config_changed)

        self._load_config_state()
        
        # Extras
        grp_opts = QGroupBox("Extras")
        grp_opts.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_opts = QVBoxLayout(grp_opts)
        self.btn_export = QPushButton(" Exportar PNG")
        self.btn_export.clicked.connect(self.export_png)
        lyt_opts.addWidget(self.btn_export)
        self.slayout.addWidget(grp_opts)
        
        self.slayout.addStretch()
        scroll.setWidget(scroll_content)
        ctrl_layout.addWidget(scroll)
        
        splitter.addWidget(self.ctrl_panel)
        splitter.setSizes([800, 260])
        self.layout.addWidget(splitter, stretch=1)
        self.channel_checkboxes = {}

    def load_csv(self, filepath):
        if not os.path.exists(filepath):
            self.lbl_status.setText(f"Error: No se encuentra el archivo {filepath}")
            return
        
        if not hasattr(self, 'offsets_por_archivo'):
            self.offsets_por_archivo = {}
        
        self.current_filepath = filepath
        if filepath not in self.offsets_por_archivo:
            self.offsets_por_archivo[filepath] = {}
            
        self.channel_offsets = self.offsets_por_archivo[filepath]

        medicion_dir = os.path.dirname(filepath)
        if medicion_dir and os.path.basename(medicion_dir).startswith("canal_"):
            parent_dir = os.path.dirname(medicion_dir)
            if os.path.exists(os.path.join(parent_dir, "metadata.json")) or os.path.exists(os.path.join(parent_dir, "fotometria")):
                medicion_dir = parent_dir
            elif os.path.dirname(parent_dir):
                medicion_dir = parent_dir
        self.current_medicion_dir = medicion_dir

        # Extraer notas del experimento y estado de fotometría inmediatamente
        notas_cargadas, _ = self._extract_session_notes_and_summary(medicion_dir)
        self.txt_notas_sesion.setPlainText(notas_cargadas)
        self._update_notas_banner(notas_cargadas, medicion_dir)

        has_fotometria = bool(medicion_dir and os.path.isdir(os.path.join(medicion_dir, "fotometria")))
        self.btn_ver_fotometria.setVisible(has_fotometria)
        self.btn_ver_fotometria.setEnabled(has_fotometria)
            
        medicion_name = os.path.basename(medicion_dir) if medicion_dir else os.path.basename(os.path.dirname(filepath))
        self.lbl_filename.setText(f"Medición: {medicion_name}")
        self.lbl_status.setText(f"Cargando medición: {medicion_name} ...")
        self.loader = DataLoaderThread(filepath)
        self.loader.finished_loading.connect(self._on_csv_loaded)
        self.loader.error_loading.connect(lambda err: self.lbl_status.setText(f"⚠️ {err}"))
        self.loader.start()

    def _extract_session_notes_and_summary(self, medicion_dir):
        """
        Extrae las notas del experimento desde metadata.json (o canal_*/metadata.json
        o notas_experimento.txt) y construye un resumen de separabilidad /O/ vs /U/
        y fotometría 3D si está disponible.
        """
        if not medicion_dir or not os.path.isdir(medicion_dir):
            return "", ""

        notas_str = ""
        meta_dicts = []

        root_meta_path = os.path.join(medicion_dir, "metadata.json")
        ch0_meta_path = os.path.join(medicion_dir, "canal_0", "metadata.json")
        for mpath in (root_meta_path, ch0_meta_path):
            if os.path.isfile(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as f:
                        md = json.load(f)
                    if isinstance(md, dict):
                        meta_dicts.append(md)
                        if not notas_str:
                            val_n = md.get("notas_experimento") or md.get("comentario") or ""
                            if isinstance(val_n, str) and val_n.strip():
                                notas_str = val_n.strip()
                except Exception:
                    pass

        # Buscar en subcarpetas canal_* si aún no hay notas
        if not notas_str:
            try:
                for item in sorted(os.listdir(medicion_dir)):
                    if item.startswith("canal_"):
                        sub_meta = os.path.join(medicion_dir, item, "metadata.json")
                        if os.path.isfile(sub_meta):
                            try:
                                with open(sub_meta, "r", encoding="utf-8") as f_sub:
                                    s_md = json.load(f_sub)
                                    if isinstance(s_md, dict):
                                        meta_dicts.append(s_md)
                                        val_s = s_md.get("notas_experimento") or s_md.get("comentario") or ""
                                        if isinstance(val_s, str) and val_s.strip():
                                            notas_str = val_s.strip()
                                            break
                            except Exception:
                                pass
            except Exception:
                pass

        if not notas_str:
            txt_path = os.path.join(medicion_dir, "notas_experimento.txt")
            if os.path.isfile(txt_path):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        notas_str = f.read().strip()
                except Exception:
                    pass

        extra_parts = []
        # 1. Separabilidad /O/ vs /U/ en metadata.json
        for md in meta_dicts:
            sep_info = (
                md.get("separabilidad_o_u")
                or md.get("resumen_separabilidad")
                or md.get("metricas_separabilidad")
            )
            if sep_info:
                if isinstance(sep_info, dict):
                    sep_items = [f"{k}: {v}" for k, v in sep_info.items()]
                    extra_parts.append("📊 Separabilidad /O/ vs /U/: " + ", ".join(sep_items))
                else:
                    extra_parts.append(f"📊 Separabilidad /O/ vs /U/: {sep_info}")
                break

        # 2. Fotometría 3D: calibración espacial y telemetría
        fotometria_dir = os.path.join(medicion_dir, "fotometria")
        cal_path = os.path.join(fotometria_dir, "calibracion_espacial.json")
        if not os.path.isfile(cal_path):
            cal_path = os.path.join(medicion_dir, "calibracion_espacial.json")

        cal_info_str = None
        if os.path.isfile(cal_path):
            try:
                with open(cal_path, "r", encoding="utf-8") as f:
                    cal = json.load(f)
                m_cal = cal.get("metadatos_calibracion_espacial", cal) if isinstance(cal, dict) else {}
                if isinstance(m_cal, dict):
                    n_el = len(m_cal.get("electrodos", []))
                    dip = m_cal.get("distancia_interpupilar_px")
                    if dip is not None:
                        cal_info_str = f"📐 Calibración 3D: {n_el} electrodos (D.IP={float(dip):.1f}px)"
                    elif n_el > 0:
                        cal_info_str = f"📐 Calibración 3D: {n_el} electrodos"
            except Exception:
                pass

        if cal_info_str:
            extra_parts.append(cal_info_str)

        # Telemetría / separabilidad fotométrica /O/ vs /U/
        telem_path = os.path.join(fotometria_dir, "fotometria_telemetria.csv")
        if not os.path.isfile(telem_path):
            telem_path = os.path.join(medicion_dir, "fotometria_telemetria.csv")

        if os.path.isfile(telem_path):
            try:
                df_t = pd.read_csv(telem_path, comment='#')
                df_t.columns = df_t.columns.str.strip()
                col_map = {c.lower(): c for c in df_t.columns}
                lip_col = next((col_map[k] for k in col_map if "apertura" in k or "labial" in k or "lip" in k), None)
                lbl_col = next((col_map[k] for k in col_map if k in ("estimulo", "palabra", "vocal", "etiqueta", "clase", "evento")), None)
                if lip_col and lbl_col:
                    s_lbl = df_t[lbl_col].astype(str).str.upper().str.strip()
                    s_lip = pd.to_numeric(df_t[lip_col], errors='coerce')
                    mask_o = s_lbl.isin(["O", "/O/", "VOCAL O"]) | s_lbl.str.contains(r"\bO\b|/O/")
                    mask_u = s_lbl.isin(["U", "/U/", "VOCAL U"]) | s_lbl.str.contains(r"\bU\b|/U/")
                    vals_o = s_lip[mask_o].dropna()
                    vals_u = s_lip[mask_u].dropna()
                    if len(vals_o) > 0 and len(vals_u) > 0:
                        mean_o, mean_u = vals_o.mean(), vals_u.mean()
                        std_pool = np.sqrt(0.5 * (vals_o.var(ddof=0) + vals_u.var(ddof=0))) + 1e-9
                        d_cohen = abs(mean_o - mean_u) / std_pool
                        extra_parts.append(
                            f"👄 Fotometría /O/ vs /U/: /O/={mean_o:.3f}, /U/={mean_u:.3f} (Δ={abs(mean_o-mean_u):.3f}, d={d_cohen:.2f})"
                        )
                # Pose promedio si no había calibración espacial
                if not cal_info_str:
                    pose_strs = []
                    for a_key, a_label in [("yaw", "Yaw"), ("pitch", "Pitch"), ("roll", "Roll")]:
                        m_c = next((col_map[k] for k in col_map if a_key in k), None)
                        if m_c:
                            s_ang = pd.to_numeric(df_t[m_c], errors='coerce').dropna()
                            if not s_ang.empty:
                                pose_strs.append(f"{a_label}: {s_ang.mean():.1f}°")
                    if pose_strs:
                        extra_parts.append("📷 Pose Cabeza: " + ", ".join(pose_strs))
            except Exception:
                pass
        elif os.path.isdir(fotometria_dir) and not cal_info_str:
            valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
            n_fotos = 0
            try:
                for root_c, _, fnames in os.walk(fotometria_dir):
                    n_fotos += sum(1 for f in fnames if f.lower().endswith(valid_exts))
            except Exception:
                pass
            if n_fotos > 0:
                extra_parts.append(f"📷 Fotometría 3D: {n_fotos} capturas disponibles")
            else:
                extra_parts.append("📷 Fotometría 3D: Activa")

        return notas_str, " | ".join(extra_parts)

    def _update_notas_banner(self, notas_str: str, medicion_dir: str):
        _, summary_str = self._extract_session_notes_and_summary(medicion_dir)
        lines = []
        if notas_str and notas_str.strip():
            lines.append(f"📝 <b>Notas del Experimento:</b> {notas_str.strip()}")
        if summary_str:
            lines.append(summary_str)

        if lines:
            self.lbl_notas_banner.setText("<br>".join(lines))
            self.lbl_notas_banner.setVisible(True)
        else:
            self.lbl_notas_banner.setText("")
            self.lbl_notas_banner.setVisible(False)

    def _save_session_notes(self):
        if not self.current_medicion_dir or not os.path.isdir(self.current_medicion_dir):
            self.lbl_status.setText("⚠️ No hay una medición cargada para guardar notas.")
            return

        notas_text = self.txt_notas_sesion.toPlainText().strip()
        updated_files = 0

        # 1. Actualizar metadata.json en raíz de la medición
        root_meta_path = os.path.join(self.current_medicion_dir, "metadata.json")
        root_meta = {}
        if os.path.isfile(root_meta_path):
            try:
                with open(root_meta_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        root_meta = loaded
            except Exception:
                root_meta = {}
        root_meta["notas_experimento"] = notas_text
        root_meta["comentario"] = notas_text
        try:
            with open(root_meta_path, "w", encoding="utf-8") as f:
                json.dump(root_meta, f, indent=4, ensure_ascii=False)
            updated_files += 1
        except Exception as e:
            self.lbl_status.setText(f"Error al guardar metadata.json: {e}")
            return

        # 2. Actualizar cualquier canal_*/metadata.json dentro de la medición
        try:
            for item in os.listdir(self.current_medicion_dir):
                sub_dir = os.path.join(self.current_medicion_dir, item)
                if os.path.isdir(sub_dir) and item.startswith("canal_"):
                    ch_meta_path = os.path.join(sub_dir, "metadata.json")
                    ch_meta = {}
                    if os.path.isfile(ch_meta_path):
                        try:
                            with open(ch_meta_path, "r", encoding="utf-8") as f:
                                loaded_ch = json.load(f)
                                if isinstance(loaded_ch, dict):
                                    ch_meta = loaded_ch
                        except Exception:
                            ch_meta = {}
                        ch_meta["notas_experimento"] = notas_text
                        ch_meta["comentario"] = notas_text
                        try:
                            with open(ch_meta_path, "w", encoding="utf-8") as f:
                                json.dump(ch_meta, f, indent=4, ensure_ascii=False)
                            updated_files += 1
                        except Exception:
                            pass
        except Exception:
            pass

        # 3. Escribir/actualizar notas_experimento.txt
        txt_path = os.path.join(self.current_medicion_dir, "notas_experimento.txt")
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(notas_text + "\n")
        except Exception:
            pass

        # 4. Actualizar banner y barra de estado
        self._update_notas_banner(notas_text, self.current_medicion_dir)
        med_name = os.path.basename(self.current_medicion_dir)
        self.lbl_status.setText(
            f"✅ Notas guardadas en '{med_name}' ({updated_files} archivo(s) metadata.json y notas_experimento.txt)."
        )

    def _open_photometry_viewer(self):
        if not self.current_medicion_dir or not os.path.isdir(self.current_medicion_dir):
            self.lbl_status.setText("⚠️ Seleccione primero una medición con datos de fotometría.")
            return
        fotometria_dir = os.path.join(self.current_medicion_dir, "fotometria")
        if not os.path.isdir(fotometria_dir):
            self.lbl_status.setText("⚠️ Esta medición no contiene carpeta 'fotometria/'.")
            return
        dlg = PhotometryGalleryDialog(self.current_medicion_dir, self)
        dlg.exec()

    def _on_csv_loaded(self, df, canales=None, time_col=None, max_y=1.0):
        # Soporte dual: señal (df, canales, time_col, max_y) o llamada directa (df, filepath)
        if isinstance(canales, str) and time_col is None:
            self.current_filepath = canales
            for c in ['Time', 'Tiempo', 'Time(s)', 'Tiempo (s)', 'Time (s)']:
                if c in df.columns:
                    time_col = c
                    break
            if not time_col and len(df.columns) > 0:
                time_col = df.columns[0]
            canales = [c for c in df.columns if c != time_col]
            if len(canales) > 0:
                try:
                    max_y = float(np.max(np.abs(df[canales].values)))
                    if np.isnan(max_y) or max_y == 0:
                        max_y = 1.0
                except Exception:
                    max_y = 1.0

        self.df = df
        self.time_data = self.df[time_col].values
        self.current_file_max_y = max_y
        
        if not hasattr(self, '_sliders_initialized'):
            self.global_max_y = max_y
        
        fs = 1.0
        if len(self.time_data) > 1:
            fs = 1.0 / (self.time_data[1] - self.time_data[0])
            
        self.lbl_status.setText(f"Archivo cargado. {len(self.df)} ptos. Fs: {fs:.1f}Hz. Tiempo: {self.time_data[-1]:.2f}s")
        
        # Configurar Sliders de Navegación
        t_max = self.time_data[-1]
        self._updating_sliders = True
        self.slider_time.setRange(0, int(t_max * 100))
        
        if not hasattr(self, '_sliders_initialized'):
            self.slider_time.setValue(0)
            self.spin_zoom_x.setValue(min(5.0, t_max))
            self.slider_y.setValue(50)
            self.slider_zoom_y.setValue(100)
            self._sliders_initialized = True
            
        self._updating_sliders = False
        
        # Guardar el estado de los canales deseleccionados antes de borrarlos
        canales_desactivados = set()
        for c, chk in self.channel_checkboxes.items():
            if not chk.isChecked():
                canales_desactivados.add(c)

        # Limpiar canales previos
        if not hasattr(self, 'channel_rows'):
            self.channel_rows = []
        for row_widget in self.channel_rows:
            self.lyt_channels.removeWidget(row_widget)
            row_widget.deleteLater()
        self.channel_rows.clear()
        self.channel_checkboxes.clear()
        
        # Guardar data original y crear checkboxes
        self.canales_originales.clear()
        medicion_dir = os.path.dirname(self.current_filepath) if hasattr(self, 'current_filepath') and self.current_filepath else None
        if medicion_dir and os.path.basename(medicion_dir).startswith("canal_"):
            parent_dir = os.path.dirname(medicion_dir)
            if os.path.exists(os.path.join(parent_dir, "metadata.json")) or os.path.exists(os.path.join(parent_dir, "fotometria")):
                medicion_dir = parent_dir
        self.current_medicion_dir = medicion_dir

        # Cargar Notas del Experimento y estado de Fotometría 3D
        notas_cargadas, _ = self._extract_session_notes_and_summary(medicion_dir)
        self.txt_notas_sesion.setPlainText(notas_cargadas)
        self._update_notas_banner(notas_cargadas, medicion_dir)

        has_fotometria = bool(medicion_dir and os.path.isdir(os.path.join(medicion_dir, "fotometria")))
        self.btn_ver_fotometria.setVisible(has_fotometria)
        self.btn_ver_fotometria.setEnabled(has_fotometria)
        
        # Cargar todos los metadatos disponibles de la medición para mapear nombres de músculos
        meta_dict_combined = {}
        ch_folders_meta = {}
        
        if medicion_dir and os.path.isdir(medicion_dir):
            root_m = os.path.join(medicion_dir, "metadata.json")
            if os.path.isfile(root_m):
                try:
                    with open(root_m, "r", encoding="utf-8") as fm:
                        meta_dict_combined = json.load(fm)
                except Exception:
                    pass
            ch0_m = os.path.join(medicion_dir, "canal_0", "metadata.json")
            if os.path.isfile(ch0_m):
                try:
                    with open(ch0_m, "r", encoding="utf-8") as fm0:
                        ch0_dict = json.load(fm0)
                        if not meta_dict_combined:
                            meta_dict_combined = ch0_dict
                        else:
                            for k, v in ch0_dict.items():
                                if k not in meta_dict_combined:
                                    meta_dict_combined[k] = v
                except Exception:
                    pass

            try:
                for item in sorted(os.listdir(medicion_dir)):
                    if item.startswith("canal_"):
                        sub_meta = os.path.join(medicion_dir, item, "metadata.json")
                        if os.path.isfile(sub_meta):
                            try:
                                with open(sub_meta, "r", encoding="utf-8") as f_sub:
                                    sub_d = json.load(f_sub)
                                    ch_folders_meta[item] = sub_d
                                    phys = sub_d.get("physical_channel")
                                    if phys:
                                        ch_folders_meta[str(phys).lower().strip()] = sub_d
                            except Exception:
                                pass
            except Exception:
                pass

        # También revisar calibracion_espacial.json para nombres de músculos de electrodos
        electrodos_espaciales = {}
        cal_path = os.path.join(medicion_dir, "fotometria", "calibracion_espacial.json")
        if not os.path.isfile(cal_path):
            cal_path = os.path.join(medicion_dir, "calibracion_espacial.json")
        if os.path.isfile(cal_path):
            try:
                with open(cal_path, "r", encoding="utf-8") as f_cal:
                    cal_d = json.load(f_cal)
                m_cal = cal_d.get("metadatos_calibracion_espacial", cal_d) if isinstance(cal_d, dict) else {}
                for el in m_cal.get("electrodos", []):
                    m_name = el.get("musculo_asignado") or el.get("musculo_estimado") or el.get("musculo")
                    if m_name:
                        c_k = el.get("canal")
                        p_k = el.get("canal_fisico_ai")
                        if c_k: electrodos_espaciales[str(c_k).lower().strip()] = m_name
                        if p_k: electrodos_espaciales[str(p_k).lower().strip()] = m_name
            except Exception:
                pass

        meta_channels = meta_dict_combined.get("channels", [])
        meta_muscles = meta_dict_combined.get("muscles", [])
        meta_muscles_map = meta_dict_combined.get("muscles_map", {})

        for idx, canal in enumerate(canales):
            self.canales_originales[canal] = self.df[canal].values
            
            canal_clean = str(canal).strip()
            canal_lower = canal_clean.lower()
            ch_digits = ''.join(filter(str.isdigit, canal_clean))
            ch_num = int(ch_digits) if ch_digits else idx
            ai_str = f"ai{ch_num}"

            nombre_musc = None
            physical_ch = None

            # 1. Resolver canal físico si se grabó con canales no contiguos (ej: ai0 y ai2)
            if idx < len(meta_channels):
                physical_ch = str(meta_channels[idx])
            elif canal_lower in ch_folders_meta:
                physical_ch = ch_folders_meta[canal_lower].get("physical_channel")
            elif f"canal_{ch_num}" in ch_folders_meta:
                physical_ch = ch_folders_meta[f"canal_{ch_num}"].get("physical_channel")

            # 2. Claves candidatas de búsqueda en muscles_map
            candidates_keys = [
                canal_clean, canal_lower, f"canal_{idx}", f"canal_{ch_num}",
                ai_str, f"Canal {ch_num}", f"Canal {idx}", str(idx), str(ch_num)
            ]
            if physical_ch:
                candidates_keys.extend([physical_ch, physical_ch.lower(), f"ai{ch_num}"])

            for k in candidates_keys:
                if k in meta_muscles_map and meta_muscles_map[k]:
                    nombre_musc = meta_muscles_map[k]
                    break

            # 3. Mapeo por canales/músculos de DAQ (canales no contiguos en meta_channels)
            if not nombre_musc:
                for c_pos, ch_cand in enumerate(meta_channels):
                    cand_clean = str(ch_cand).lower().strip()
                    if cand_clean in (canal_lower, ai_str) or cand_clean.endswith(f"/{canal_lower}"):
                        if c_pos < len(meta_muscles):
                            nombre_musc = meta_muscles[c_pos]
                            if not physical_ch:
                                physical_ch = str(ch_cand)
                            break

            # 4. Posición ordinal si coincide número de músculos
            if not nombre_musc and idx < len(meta_muscles):
                nombre_musc = meta_muscles[idx]

            # 5. Buscar en carpetas canal_* (canal_X/metadata.json)
            if not nombre_musc:
                for sub_key in (f"canal_{ch_num}", f"canal_{idx}", canal_lower, ai_str):
                    sub_meta = ch_folders_meta.get(sub_key)
                    if sub_meta and 'musculo' in sub_meta and sub_meta['musculo']:
                        nombre_musc = sub_meta['musculo']
                        if not physical_ch:
                            physical_ch = sub_meta.get("physical_channel")
                        break

            # 6. Buscar en calibracion_espacial.json
            if not nombre_musc:
                for k in candidates_keys:
                    if k.lower() in electrodos_espaciales:
                        nombre_musc = electrodos_espaciales[k.lower()]
                        break

            # 7. Fallback a configuración global
            if not nombre_musc:
                if ch_num < len(self.canal_nombres):
                    nombre_musc = self.canal_nombres[ch_num]
                elif idx < len(self.canal_nombres):
                    nombre_musc = self.canal_nombres[idx]
                else:
                    nombre_musc = canal_clean

            try:
                from utils.config_manager import get_muscle_color
                if ch_num == 3 or "mic" in nombre_musc.lower():
                    color_canal = get_muscle_color("micrófono", default="#ff0000")
                else:
                    default_c = self.channel_colors[ch_num] if ch_num < len(self.channel_colors) else '#ffffff'
                    color_canal = get_muscle_color(nombre_musc, default_c)
            except Exception:
                color_canal = "#ff0000" if ch_num == 3 else (self.channel_colors[ch_num] if ch_num < len(self.channel_colors) else '#ffffff')
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            
            # Etiqueta con canal físico si difiere (ej. 'Canal 1 (ai2) - Depresor')
            chk_text_parts = [canal_clean]
            if physical_ch and str(physical_ch).lower() != canal_lower:
                chk_text_parts.append(f"({physical_ch})")
            chk_text_parts.append(f"- {nombre_musc}")
            chk_label = " ".join(chk_text_parts)

            chk = QCheckBox(chk_label)
            if canal in canales_desactivados:
                chk.setChecked(False)
            else:
                chk.setChecked(True)
                
            chk.setStyleSheet(f"color: {color_canal}; font-weight: bold;")
            chk.stateChanged.connect(self.update_plot)
            
            chk.setProperty("ch_num", ch_num)
            chk.setProperty("nombre_plot", nombre_musc)
            chk.setProperty("color_plot", color_canal)
            self.channel_checkboxes[canal] = chk
            
            row_layout.addWidget(chk)
            
            lbl_offset = QLabel("Offset(µV):")
            lbl_offset.setStyleSheet("color: #888; font-size: 10px;")
            row_layout.addWidget(lbl_offset)
            
            spin_offset = QSpinBox()
            spin_offset.setRange(-20000, 20000)
            spin_offset.setSingleStep(100)
            spin_offset.setValue(int(self.channel_offsets.get(canal, 0)))
            
            # Lambda helper para capturar 'canal' correctamente
            def make_offset_handler(c_name):
                return lambda val: self._on_offset_changed(c_name, val)
            spin_offset.valueChanged.connect(make_offset_handler(canal))
            
            row_layout.addWidget(spin_offset)
            
            self.lyt_channels.addWidget(row_widget)
            self.channel_rows.append(row_widget)
            
        self.update_plot()
        self._apply_view_ranges()

    def update_plot(self):
        if self.df is None: return
        self.plot_widget.clear()
        
        # Limpiar la leyenda para evitar que se acumulen nombres viejos (canales "fantasma")
        if getattr(self.plot_widget.plotItem, 'legend', None) is not None:
            self.plot_widget.plotItem.legend.clear()
        
        fs = 1.0
        if len(self.time_data) > 1:
            fs = 1.0 / (self.time_data[1] - self.time_data[0])
            
        notch = self.chk_notch.isChecked()
        lp = self.spin_lp.value()
        hp = self.spin_hp.value()
        tipo_env = self.cmb_env.currentText()
        env_window = int((self.spin_env.value() / 1000.0) * fs)
        if env_window < 1: env_window = 1
        
        for idx_canal, (canal, chk) in enumerate(self.channel_checkboxes.items()):
            if chk.isChecked():
                y_data = self.canales_originales[canal].copy()
                
                ch_num = chk.property("ch_num")
                nombre_plot = chk.property("nombre_plot")
                color = chk.property("color_plot")
                if not color:
                    try:
                        from utils.config_manager import get_muscle_color
                        if ch_num == 3 or "mic" in str(nombre_plot).lower():
                            color = get_muscle_color("micrófono", default="#ff0000")
                        else:
                            default_c = self.channel_colors[ch_num] if ch_num < len(self.channel_colors) else '#ffffff'
                            color = get_muscle_color(nombre_plot, default_c)
                    except Exception:
                        color = "#ff0000" if ch_num == 3 else (self.channel_colors[ch_num] if ch_num < len(self.channel_colors) else '#ffffff')

                is_emg = (ch_num != 3 and "mic" not in str(nombre_plot).lower())
                
                # Filtro Adaptativo (NLMS de 50 Hz con referencia sintética en cuadratura)
                if self.chk_adaptive.isChecked() and is_emg and fs > 110 and cancelar_ruido_linea_adaptativo is not None:
                    e_clean, _, _ = cancelar_ruido_linea_adaptativo(
                        y_data, fs, f0=50.0, mu=0.01, normalizado=True
                    )
                    y_data = e_clean

                # 3. Filtros convencionales
                if notch and fs > 110:
                    notch_q = self.spin_notch_q.value()
                    b, a = signal.iirnotch(50.0, notch_q, fs)
                    y_data = signal.filtfilt(b, a, y_data)
                
                if self.chk_bandpass.isChecked():
                    if hp > 0 and hp < fs/2:
                        b, a = signal.butter(4, hp / (0.5 * fs), btype='high')
                        y_data = signal.filtfilt(b, a, y_data)
                    if lp > 0 and lp < fs/2:
                        b, a = signal.butter(4, lp / (0.5 * fs), btype='low')
                        y_data = signal.filtfilt(b, a, y_data)
                
                # 4. Envolvente
                if tipo_env != "ninguna":
                    from scipy.ndimage import uniform_filter1d
                    if tipo_env == "rms":
                        y_data_sq = y_data**2
                        y_data = np.sqrt(uniform_filter1d(y_data_sq, size=env_window))
                    else: # media_movil
                        y_data = np.abs(y_data)
                        y_data = uniform_filter1d(y_data, size=env_window)
                        
                    if self.chk_env_offset.isChecked():
                        y_data = y_data - np.min(y_data)
                    
                # 5. Offset Manual
                offset_val = self.channel_offsets.get(canal, 0)
                y_data = y_data + offset_val
                    
                # 6. Downsampling y Graficación
                x_plot, y_plot = downsample_lttb_fast(self.time_data, y_data, MAX_POINTS_TO_PLOT)
                self.plot_widget.plot(x_plot, y_plot, name=nombre_plot, pen=pg.mkPen(color, width=1.5))

    def _on_offset_changed(self, canal, offset_val):
        self.channel_offsets[canal] = offset_val
        self.update_plot()

    def _autoscale(self):
        if self.df is None: return
        # Al apretar Ajuste Automático, redefinimos el máximo global al archivo actual
        if hasattr(self, 'current_file_max_y'):
            self.global_max_y = self.current_file_max_y
            
        self._updating_sliders = True
        self.slider_zoom_y.setValue(100)
        self.slider_y.setValue(50)
        self._updating_sliders = False
        
        # En vez de autoRange() que rompe la sincronización, aplicamos manualmente
        self._apply_view_ranges()

    def _apply_view_ranges(self):
        if self.df is None or self._updating_sliders: return
        
        t_start = self.slider_time.value() / 100.0
        duracion = self.spin_zoom_x.value()
        self.plot_widget.setXRange(t_start, t_start + duracion, padding=0)
        
        amp_pct = self.slider_zoom_y.value() / 100.0
        y_span = (self.global_max_y * 2.5) * amp_pct
        
        y_center_pct = self.slider_y.value() / 100.0 # 0 a 1
        y_min_limit = -self.global_max_y * 1.5
        y_max_limit = self.global_max_y * 1.5
        y_center = y_min_limit + (y_max_limit - y_min_limit) * y_center_pct
        
        self.plot_widget.setYRange(y_center - y_span/2, y_center + y_span/2, padding=0)

    def _on_time_slider_changed(self):
        self._apply_view_ranges()

    def _on_y_slider_changed(self):
        self._apply_view_ranges()

    def _on_plot_xrange_changed(self, _, range_tuple):
        """Sincronizar slider cuando el usuario arrastra el gráfico con el mouse en el eje X"""
        if self._updating_sliders or self.df is None: return
        self._updating_sliders = True
        self.slider_time.setValue(int(range_tuple[0] * 100))
        duracion = range_tuple[1] - range_tuple[0]
        self.spin_zoom_x.setValue(max(0.1, duracion))
        self._updating_sliders = False

    def _on_plot_yrange_changed(self, _, range_tuple):
        """Sincronizar slider cuando el usuario arrastra el gráfico con el mouse en el eje Y"""
        if self._updating_sliders or self.df is None: return
        self._updating_sliders = True
        
        y_min_limit = -self.global_max_y * 1.5
        y_max_limit = self.global_max_y * 1.5
        
        y_center = (range_tuple[1] + range_tuple[0]) / 2.0
        y_span = range_tuple[1] - range_tuple[0]
        
        y_center_pct = (y_center - y_min_limit) / (y_max_limit - y_min_limit)
        amp_pct = y_span / (self.global_max_y * 2.5)
        
        self.slider_y.setValue(int(np.clip(y_center_pct * 100, 0, 100)))
        self.slider_zoom_y.setValue(int(np.clip(amp_pct * 100, 1, 100)))
        
        self._updating_sliders = False

    def export_png(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Exportar Gráfico", "export.png", "PNG (*.png)")
        if filepath:
            exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
            exporter.export(filepath)
            self.lbl_status.setText(f"Exportado a: {filepath}")

    def _on_config_changed(self):
        config_mgr.set("csv_viewer", "notch", self.chk_notch.isChecked())
        config_mgr.set("csv_viewer", "notch_q", self.spin_notch_q.value())
        config_mgr.set("csv_viewer", "bandpass", self.chk_bandpass.isChecked())
        config_mgr.set("csv_viewer", "lp", self.spin_lp.value())
        config_mgr.set("csv_viewer", "hp", self.spin_hp.value())
        config_mgr.set("csv_viewer", "env", self.cmb_env.currentText())
        config_mgr.set("csv_viewer", "env_ms", self.spin_env.value())
        config_mgr.set("csv_viewer", "env_offset", self.chk_env_offset.isChecked())
        # Persistencia de filtro adaptativo
        config_mgr.set("csv_viewer", "adaptive_filter", self.chk_adaptive.isChecked())
        self.update_plot()

    def _load_config_state(self):
        saved = config_mgr.get("csv_viewer") or {}
        self.chk_notch.setChecked(saved.get("notch", False))
        self.spin_notch_q.setValue(saved.get("notch_q", 2.0))
        self.chk_bandpass.setChecked(saved.get("bandpass", True))
        self.spin_lp.setValue(saved.get("lp", 500.0))
        self.spin_hp.setValue(saved.get("hp", 20.0))
        env_text = saved.get("env", "ninguna")
        idx = self.cmb_env.findText(env_text)
        if idx >= 0: self.cmb_env.setCurrentIndex(idx)
        self.spin_env.setValue(saved.get("env_ms", 50))
        self.chk_env_offset.setChecked(saved.get("env_offset", False))
        
        # Carga de estado de filtro adaptativo
        self.chk_adaptive.setChecked(saved.get("adaptive_filter", False))
