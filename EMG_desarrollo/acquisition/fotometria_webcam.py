# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de fotometría automatizada por webcam, biometría facial 3D
#              (pose craneal y articulación labial/mandibular) y asistente
#              interactivo de alineación y mapeo de electrodos sEMG pre-sesión.
# ==============================================================================

import csv
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets
        if not hasattr(QtCore, "Signal") and hasattr(QtCore, "pyqtSignal"):
            QtCore.Signal = QtCore.pyqtSignal
        if not hasattr(QtCore, "Slot") and hasattr(QtCore, "pyqtSlot"):
            QtCore.Slot = QtCore.pyqtSlot
    except ImportError:
        QtCore = QtGui = QtWidgets = None

if QtCore is not None and hasattr(QtCore, "QCoreApplication"):
    QtCore.QCoreApplication.setLibraryPaths([p for p in QtCore.QCoreApplication.libraryPaths() if "cv2" not in p])

import cv2
import numpy as np

try:
    from .calibracion_espacial_electrodos import (
        AntigravityAgent,
        INDICE_COMISURA_DER,
        INDICE_COMISURA_IZQ,
        INDICE_GLABELA,
        INDICE_LABIO_INF_INTERNO,
        INDICE_LABIO_SUP_INTERNO,
        INDICE_MENTON_GNATHION,
        INDICE_NARIZ_PUNTA,
        INDICE_POMULO_DER,
        INDICE_POMULO_IZQ,
        INDICE_TRAGO_OREJA_IZQ,
        INDICE_TRAGO_OREJA_DER,
        TrianguladorElectrodosFaciales,
    )
except ImportError:
    from calibracion_espacial_electrodos import (
        AntigravityAgent,
        INDICE_COMISURA_DER,
        INDICE_COMISURA_IZQ,
        INDICE_GLABELA,
        INDICE_LABIO_INF_INTERNO,
        INDICE_LABIO_SUP_INTERNO,
        INDICE_MENTON_GNATHION,
        INDICE_NARIZ_PUNTA,
        INDICE_POMULO_DER,
        INDICE_POMULO_IZQ,
        INDICE_TRAGO_OREJA_IZQ,
        INDICE_TRAGO_OREJA_DER,
        TrianguladorElectrodosFaciales,
    )


class WebcamCaptureThread(QtCore.QThread):
    """
    Hilo demonio de captura continua para cámara web con buffer mínimo (CAP_PROP_BUFFERSIZE = 1).

    Drena continuamente el buffer USB del sistema operativo a 640x480 para garantizar
    latencia despreciable (< 0.1 ms en lectura desde memoria) cuando el hilo DAQ solicita
    instantáneas sincronizadas con pulsos de metrónomo.
    """

    cameraStatusChanged = QtCore.Signal(bool, str)

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.camera_index: int = int(camera_index)
        self.frame_width: int = int(width)
        self.frame_height: int = int(height)

        self._lock = threading.Lock()
        self._running: bool = False
        self._is_open: bool = False
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0

        # Estado de grabación opcional de clips de video
        self._video_lock = threading.Lock()
        self._recording_video: bool = False
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_filepath: Optional[str] = None
        self._video_fps: float = 15.0
        self._last_video_write_t: float = 0.0
        self._video_frame_timestamps: List[Tuple[int, float, str]] = []
        self._video_frame_counter: int = 0

    @property
    def is_camera_open(self) -> bool:
        with self._lock:
            return self._is_open

    def get_latest_frame(self, copy: bool = True) -> Tuple[Optional[np.ndarray], float]:
        """
        Retorna de forma thread-safe una copia (o referencia directa si copy=False) del último
        cuadro BGR capturado y su timestamp `time.perf_counter()`. Operación en memoria (< 0.1 ms).
        """
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return (self._latest_frame.copy() if copy else self._latest_frame), self._latest_timestamp

    def start_video_recording(self, filepath: str, fps: float = 15.0) -> bool:
        """Inicia la grabación de un clip de video en segundo plano junto con el registro de timestamps."""
        with self._video_lock:
            self._close_video_writer_unlocked()
            try:
                os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
                ext = os.path.splitext(filepath)[1].lower()
                fourcc = (
                    cv2.VideoWriter_fourcc(*"mp4v")
                    if ext == ".mp4"
                    else cv2.VideoWriter_fourcc(*"XVID")
                )
                writer = cv2.VideoWriter(
                    filepath,
                    fourcc,
                    float(fps),
                    (self.frame_width, self.frame_height),
                )
                if not writer.isOpened():
                    return False

                self._video_writer = writer
                self._video_filepath = filepath
                self._video_fps = max(float(fps), 1.0)
                self._last_video_write_t = 0.0
                self._video_frame_timestamps = []
                self._video_frame_counter = 0
                self._recording_video = True
                return True
            except Exception as exc:
                sys.stderr.write(f"[FotometriaWebcam] Error al iniciar grabacion de video: {exc}\n")
                self._close_video_writer_unlocked()
                return False

    def stop_video_recording(self) -> Optional[str]:
        """Detiene la grabación de video activa y exporta el archivo CSV de timestamps por cuadro."""
        with self._video_lock:
            saved_path = self._video_filepath
            timestamps = list(self._video_frame_timestamps)
            self._close_video_writer_unlocked()

        if saved_path and timestamps:
            try:
                base_no_ext = os.path.splitext(saved_path)[0]
                csv_ts_path = f"{base_no_ext}_timestamps.csv"
                with open(csv_ts_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["frame_idx", "t_perf_s", "timestamp_iso"])
                    writer.writerows(timestamps)
            except Exception as exc:
                sys.stderr.write(f"[FotometriaWebcam] Aviso al guardar timestamps de video: {exc}\n")

        return saved_path

    def _close_video_writer_unlocked(self) -> None:
        self._recording_video = False
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

    def stop(self) -> None:
        """Detiene el bucle de captura y libera el dispositivo de video."""
        self._running = False
        self.stop_video_recording()
        if self.isRunning():
            self.wait(1500)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def _open_capture_device(self) -> Optional[cv2.VideoCapture]:
        cap = None
        try:
            if sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self.camera_index)
            elif sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self.camera_index)
            else:
                cap = cv2.VideoCapture(self.camera_index)

            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                return cap
        except Exception as exc:
            sys.stderr.write(f"[FotometriaWebcam] Excepcion abriendo camara {self.camera_index}: {exc}\n")

        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        return None

    def run(self) -> None:
        self._running = True
        cap: Optional[cv2.VideoCapture] = None
        consecutive_failures = 0
        last_open_attempt_t = 0.0

        try:
            while self._running:
                # Si el dispositivo no está abierto o se perdió, reintentar abrir periódicamente cada 600 ms
                if cap is None or not cap.isOpened():
                    t_now_wall = time.time()
                    if (t_now_wall - last_open_attempt_t) >= 0.6:
                        last_open_attempt_t = t_now_wall
                        cap = self._open_capture_device()
                        if cap is not None and cap.isOpened():
                            with self._lock:
                                self._is_open = True
                            self.cameraStatusChanged.emit(True, f"Cámara {self.camera_index} conectada.")
                        else:
                            with self._lock:
                                self._is_open = False
                    self.msleep(60)
                    continue

                ret, frame = cap.read()
                t_now = time.perf_counter()
                if not ret or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        # Si falló muchas veces seguidas, liberar cap y volver a modo búsqueda en caliente
                        try:
                            cap.release()
                        except Exception:
                            pass
                        cap = None
                        with self._lock:
                            self._is_open = False
                        self.cameraStatusChanged.emit(False, f"Buscando cámara {self.camera_index}...")
                        consecutive_failures = 0
                    else:
                        self.msleep(15)
                    continue

                consecutive_failures = 0
                if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))

                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = t_now

                if self._recording_video:
                    with self._video_lock:
                        if self._recording_video and self._video_writer is not None:
                            min_dt = 1.0 / max(self._video_fps, 1.0)
                            if (t_now - self._last_video_write_t) >= (min_dt * 0.85):
                                try:
                                    self._video_writer.write(frame)
                                    self._last_video_write_t = t_now
                                    self._video_frame_timestamps.append(
                                        (self._video_frame_counter, round(t_now, 6), datetime.now().isoformat())
                                    )
                                    self._video_frame_counter += 1
                                except Exception:
                                    pass
        finally:
            self.stop_video_recording()
            try:
                cap.release()
            except Exception:
                pass
            with self._lock:
                self._is_open = False


class _ClickableVideoLabel(QtWidgets.QLabel):
    """
    QLabel interactivo que mantiene relación de aspecto y traduce clics del mouse
    a coordenadas exactas de píxel sobre la imagen original (640x480).
    """

    pointClicked = QtCore.Signal(float, float)
    rightClicked = QtCore.Signal(float, float)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setCursor(QtCore.Qt.CrossCursor)
        self._image_size: Tuple[int, int] = (640, 480)

    def set_image_dimensions(self, width: int, height: int) -> None:
        if width > 0 and height > 0:
            self._image_size = (width, height)

    def _map_widget_to_image(self, pos_x: float, pos_y: float) -> Optional[Tuple[float, float]]:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return None

        pw = float(pixmap.width())
        ph = float(pixmap.height())
        if pw <= 0 or ph <= 0:
            return None

        lbl_w = float(self.width())
        lbl_h = float(self.height())
        offset_x = (lbl_w - pw) / 2.0
        offset_y = (lbl_h - ph) / 2.0

        rel_x = pos_x - offset_x
        rel_y = pos_y - offset_y
        if rel_x < 0 or rel_y < 0 or rel_x > pw or rel_y > ph:
            return None

        img_w, img_h = self._image_size
        img_x = (rel_x / pw) * float(img_w)
        img_y = (rel_y / ph) * float(img_h)
        return (
            float(np.clip(img_x, 0.0, img_w - 1.0)),
            float(np.clip(img_y, 0.0, img_h - 1.0)),
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position() if hasattr(event, "position") else event.pos()
        mapped = self._map_widget_to_image(pos.x(), pos.y())
        if mapped is not None:
            if event.button() == QtCore.Qt.LeftButton:
                self.pointClicked.emit(mapped[0], mapped[1])
            elif event.button() == QtCore.Qt.RightButton:
                self.rightClicked.emit(mapped[0], mapped[1])
        super().mousePressEvent(event)


class InitialAlignmentDialog(QtWidgets.QDialog):
    """
    Asistente interactivo pre-sesión de alineación craneal 3D, superposición fantasma
    ('Ghost Overlay') y mapeo/triangulación de electrodos sEMG.
    """

    def __init__(
        self,
        capture_thread: Optional[WebcamCaptureThread],
        agent: AntigravityAgent,
        parent: Optional[QtWidgets.QWidget] = None,
        nombres_musculos: Optional[List[str]] = None,
        active_channel_indices: Optional[List[int]] = None,
        sujeto: str = "",
        prueba: str = "",
        initial_calibration: Optional[Dict[str, Any]] = None,
        initial_references: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__(parent)
        self.capture_thread = capture_thread
        self.agent = agent
        self.sujeto = sujeto or "Sujeto"
        self.prueba = prueba or "Experimento"

        # Configurar canales activos y nombres de músculos
        if active_channel_indices and len(active_channel_indices) > 0:
            self.active_channel_indices: List[int] = [int(c) for c in active_channel_indices]
        else:
            n_ch = len(nombres_musculos) if nombres_musculos else 4
            self.active_channel_indices = list(range(max(n_ch, 1)))

        self.nombres_musculos: List[str] = []
        for idx, ch_num in enumerate(self.active_channel_indices):
            if nombres_musculos and idx < len(nombres_musculos) and str(nombres_musculos[idx]).strip():
                nombre_m = str(nombres_musculos[idx]).strip()
            else:
                nombre_m = f"Músculo {ch_num}"
            self.nombres_musculos.append(nombre_m)

        self.etiquetas_canales: List[str] = [
            f"ai{ch}: {musc}" for ch, musc in zip(self.active_channel_indices, self.nombres_musculos)
        ]

        self.calibracion_inicial_data: Optional[Dict[str, Any]] = (
            dict(initial_calibration) if initial_calibration else None
        )
        self.reference_images: Dict[str, np.ndarray] = (
            {k: v.copy() for k, v in initial_references.items()} if initial_references else {}
        )

        # Estado interactivo del diálogo
        self._frozen_frame: Optional[np.ndarray] = None
        self._last_live_frame: Optional[np.ndarray] = None
        self._last_fiduciarios: Optional[Dict[str, Any]] = None
        self._last_pose_metrics: Dict[str, Any] = {}
        self._manual_points_px: List[Tuple[float, float]] = []
        self._differential_pairs: List[Dict[str, Any]] = []
        self._pending_pos_pole: Optional[Tuple[float, float]] = None
        self._pending_pos_anchor: Optional[Dict[str, Any]] = None
        self._gnd_electrode_px: Optional[Tuple[float, float]] = None
        self._gnd_electrode_anchor: Optional[Dict[str, Any]] = None
        self._manual_points_anchors: List[Optional[Dict[str, Any]]] = []
        self._marking_gnd_active: bool = False
        self._ghost_reference_data: Optional[Dict[str, Any]] = None
        self._ghost_reference_path: str = ""
        self._profile_45_mode: bool = False
        self._alignment_pose_target: str = "frontal"

        self.setWindowTitle(
            f"Asistente de Alineacion Facial 3D y Mapeo de Electrodos sEMG - {self.sujeto} : {self.prueba}"
        )
        self.resize(1180, 760)
        self._apply_dark_theme()
        self._build_ui()

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_preview)
        self._timer.start()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }
            QGroupBox {
                background-color: #222222;
                border: 1px solid #333333;
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
                background-color: #2a2a2a;
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
            QPushButton:pressed {
                background-color: #00ffcc;
                color: #111111;
            }
            QComboBox {
                background-color: #262626;
                color: #00ffcc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QListWidget {
                background-color: #141414;
                color: #e6edf3;
                border: 1px solid #333333;
                border-radius: 4px;
                font-size: 12px;
            }
            QCheckBox {
                color: #e6edf3;
                font-weight: bold;
            }
            """
        )

    def _build_ui(self) -> None:
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(14)

        # --- Columna Izquierda: Monitor en Vivo + Semáforos de Pose ---
        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(8)

        header_lbl = QtWidgets.QLabel(
            "<b>Vista en Vivo y Guía de Posicionamiento Craneal 3D</b> "
            "<span style='color:#00ffcc;'>(Clic izquierdo sobre el rostro para marcar electrodo | Clic derecho: deshacer)</span>"
        )
        header_lbl.setWordWrap(True)
        left_col.addWidget(header_lbl)

        self.video_label = _ClickableVideoLabel(self)
        self.video_label.setStyleSheet(
            "background-color: #0d0d0d; border: 2px solid #00ffcc; border-radius: 6px;"
        )
        self.video_label.pointClicked.connect(self._on_video_left_clicked)
        self.video_label.rightClicked.connect(self._on_video_right_clicked)
        left_col.addWidget(self.video_label, stretch=1)

        # Barra de semáforos en tiempo real (Yaw, Pitch, Roll, Distancia)
        hud_group = QtWidgets.QGroupBox("Semaforo de Alineacion 3D y Biometria Articular en Vivo")
        hud_layout = QtWidgets.QGridLayout(hud_group)
        hud_layout.setSpacing(8)

        self.badge_yaw = self._create_metric_badge("Yaw (Giro): --°")
        self.badge_pitch = self._create_metric_badge("Pitch (Inclinación): --°")
        self.badge_roll = self._create_metric_badge("Roll (Ladeo): --°")
        self.badge_dist = self._create_metric_badge("Distancia: -- cm")

        hud_layout.addWidget(self.badge_yaw, 0, 0)
        hud_layout.addWidget(self.badge_pitch, 0, 1)
        hud_layout.addWidget(self.badge_roll, 0, 2)
        hud_layout.addWidget(self.badge_dist, 0, 3)

        self.lbl_biometria_live = QtWidgets.QLabel(
            "Labios (apertura): -- | Comisuras: -- | Mandíbula: -- | Protrusión Z: --"
        )
        self.lbl_biometria_live.setStyleSheet(
            "color: #b0bec5; font-family: monospace; font-size: 12px; padding: 3px;"
        )
        hud_layout.addWidget(self.lbl_biometria_live, 1, 0, 1, 4)
        left_col.addWidget(hud_group)

        main_layout.addLayout(left_col, stretch=3)

        # --- Columna Derecha: Controles de Electrodos, Ghost Overlay y Capturas ---
        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(10)

        # 1. Grupo de Configuración y Marcado de Electrodos
        elec_group = QtWidgets.QGroupBox("1: Configuracion y Marcado de Electrodos")
        elec_layout = QtWidgets.QVBoxLayout(elec_group)

        type_row = QtWidgets.QHBoxLayout()
        type_lbl = QtWidgets.QLabel("Tipo:")
        type_lbl.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.combo_electrode_type = QtWidgets.QComboBox()
        self.combo_electrode_type.addItems([
            "[Bipolar] Par Diferencial (2 polos: + y -)",
            "[Monopolar] Electrodo Único (1 polo)",
        ])
        self.combo_electrode_type.currentIndexChanged.connect(self._on_electrode_type_changed)
        type_row.addWidget(type_lbl)
        type_row.addWidget(self.combo_electrode_type, stretch=1)
        elec_layout.addLayout(type_row)

        self.combo_detection_mode = QtWidgets.QComboBox()
        self.combo_detection_mode.addItems([
            "Marcado Manual por Clic en Imagen",
            "Automático: Marcadores HSV (Verde Flúor)",
            "Automático: Broches Metálicos (Hough)",
        ])
        elec_layout.addWidget(self.combo_detection_mode)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_freeze = QtWidgets.QPushButton("Congelar")
        self.btn_freeze.setCheckable(True)
        self.btn_freeze.toggled.connect(self._on_toggle_freeze)
        btn_row.addWidget(self.btn_freeze)

        self.btn_mark_gnd = QtWidgets.QPushButton("Tierra GND")
        self.btn_mark_gnd.setCheckable(True)
        self.btn_mark_gnd.setStyleSheet("QPushButton:checked { background-color: #00e676; color: #000; font-weight: bold; }")
        self.btn_mark_gnd.toggled.connect(self._on_toggle_mark_gnd)
        btn_row.addWidget(self.btn_mark_gnd)

        self.btn_clear_pts = QtWidgets.QPushButton("Limpiar")
        self.btn_clear_pts.clicked.connect(self._clear_manual_points)
        btn_row.addWidget(self.btn_clear_pts)
        elec_layout.addLayout(btn_row)

        self.lbl_next_channel = QtWidgets.QLabel("")
        self.lbl_next_channel.setWordWrap(True)
        self.lbl_next_channel.setStyleSheet("color: #00ffcc; font-weight: bold;")
        elec_layout.addWidget(self.lbl_next_channel)

        self.list_channels = QtWidgets.QListWidget()
        self.list_channels.setMaximumHeight(115)
        elec_layout.addWidget(self.list_channels)
        self._refresh_channel_list_ui()

        right_col.addWidget(elec_group)

        # 2. Grupo de Referencia Métrica y Calibración
        calib_group = QtWidgets.QGroupBox("2: Referencia Metrica y Dimensiones de Electrodos")
        calib_layout = QtWidgets.QVBoxLayout(calib_group)

        self.combo_calibration_mode = QtWidgets.QComboBox()
        self.combo_calibration_mode.addItems([
            "D_IP Real Medida del Sujeto (mm)",
            "IED Fija de Parches Bipolares (mm)",
            "Distancia Sujeto-Cámara Fija (cm)",
            "Automática Antropométrica (D_IP = 63 mm)",
        ])
        self.combo_calibration_mode.currentIndexChanged.connect(self._on_calibration_mode_changed)
        calib_layout.addWidget(self.combo_calibration_mode)

        calib_param_row = QtWidgets.QHBoxLayout()
        self.lbl_calib_param = QtWidgets.QLabel("D_IP Sujeto:")
        calib_param_row.addWidget(self.lbl_calib_param)

        self.spin_dip_mm = QtWidgets.QDoubleSpinBox()
        self.spin_dip_mm.setRange(40.0, 90.0)
        self.spin_dip_mm.setValue(63.0)
        self.spin_dip_mm.setSingleStep(0.5)
        self.spin_dip_mm.setSuffix(" mm")
        self.spin_dip_mm.valueChanged.connect(self._update_preview)
        calib_param_row.addWidget(self.spin_dip_mm)

        self.spin_ied_mm = QtWidgets.QDoubleSpinBox()
        self.spin_ied_mm.setRange(4.0, 60.0)
        self.spin_ied_mm.setValue(20.0)
        self.spin_ied_mm.setSingleStep(0.5)
        self.spin_ied_mm.setSuffix(" mm")
        self.spin_ied_mm.setVisible(False)
        self.spin_ied_mm.valueChanged.connect(self._update_preview)
        calib_param_row.addWidget(self.spin_ied_mm)

        self.spin_cam_cm = QtWidgets.QDoubleSpinBox()
        self.spin_cam_cm.setRange(15.0, 250.0)
        self.spin_cam_cm.setValue(60.0)
        self.spin_cam_cm.setSingleStep(1.0)
        self.spin_cam_cm.setSuffix(" cm")
        self.spin_cam_cm.setVisible(False)
        self.spin_cam_cm.valueChanged.connect(self._update_preview)
        calib_param_row.addWidget(self.spin_cam_cm)

        calib_layout.addLayout(calib_param_row)

        # Dimensiones de electrodos para protocolo experimental (siempre visibles)
        dims_grid = QtWidgets.QGridLayout()
        lbl_gel = QtWidgets.QLabel("Diametro Gel / Parche:")
        lbl_gel.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_gel_diam_mm = QtWidgets.QDoubleSpinBox()
        self.spin_gel_diam_mm.setRange(1.0, 50.0)
        self.spin_gel_diam_mm.setValue(10.0)
        self.spin_gel_diam_mm.setSingleStep(0.5)
        self.spin_gel_diam_mm.setSuffix(" mm")
        self.spin_gel_diam_mm.valueChanged.connect(self._update_preview)

        lbl_ied_nom = QtWidgets.QLabel("Distancia Inter-Electrodo Centro a Centro:")
        lbl_ied_nom.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_ied_nominal_mm = QtWidgets.QDoubleSpinBox()
        self.spin_ied_nominal_mm.setRange(2.0, 80.0)
        self.spin_ied_nominal_mm.setValue(20.0)
        self.spin_ied_nominal_mm.setSingleStep(0.5)
        self.spin_ied_nominal_mm.setSuffix(" mm")
        self.spin_ied_nominal_mm.valueChanged.connect(self._update_preview)

        dims_grid.addWidget(lbl_gel, 0, 0)
        dims_grid.addWidget(self.spin_gel_diam_mm, 0, 1)
        dims_grid.addWidget(lbl_ied_nom, 1, 0)
        dims_grid.addWidget(self.spin_ied_nominal_mm, 1, 1)
        calib_layout.addLayout(dims_grid)

        self.chk_show_ears = QtWidgets.QCheckBox("Mostrar Detección de Orejas (Trago y Lóbulo)")
        self.chk_show_ears.setChecked(True)
        self.chk_show_ears.toggled.connect(self._update_preview)
        calib_layout.addWidget(self.chk_show_ears)

        right_col.addWidget(calib_group)

        # 3. Grupo Ghost Overlay (Referencia Anterior)
        ghost_group = QtWidgets.QGroupBox("3: Repetibilidad entre Sesiones - Ghost Overlay")
        ghost_layout = QtWidgets.QVBoxLayout(ghost_group)

        self.btn_load_ghost = QtWidgets.QPushButton("Cargar Referencia Anterior (Ghost Overlay)")
        self.btn_load_ghost.clicked.connect(self._on_load_ghost_reference)
        ghost_layout.addWidget(self.btn_load_ghost)

        self.chk_show_ghost = QtWidgets.QCheckBox("Mostrar círculos objetivo proyectados en malla 3D")
        self.chk_show_ghost.setChecked(True)
        self.chk_show_ghost.setEnabled(False)
        ghost_layout.addWidget(self.chk_show_ghost)

        self.lbl_ghost_status = QtWidgets.QLabel("Sin referencia previa cargada.")
        self.lbl_ghost_status.setWordWrap(True)
        self.lbl_ghost_status.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        ghost_layout.addWidget(self.lbl_ghost_status)

        right_col.addWidget(ghost_group)

        # 4. Grupo de Capturas de Referencia (3 fotos guiadas + close-up)
        captures_group = QtWidgets.QGroupBox("4: Capturas de Referencia de la Sesion")
        captures_layout = QtWidgets.QVBoxLayout(captures_group)
        captures_layout.setSpacing(6)

        # Selector de pose guia interactiva
        guide_row = QtWidgets.QHBoxLayout()
        self.btn_guide_frontal = QtWidgets.QPushButton("Guia: Frente")
        self.btn_guide_frontal.setCheckable(True)
        self.btn_guide_frontal.setChecked(True)
        self.btn_guide_frontal.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; }")

        self.btn_guide_perfil_izq = QtWidgets.QPushButton("Guia: Perfil Izq +45°")
        self.btn_guide_perfil_izq.setCheckable(True)
        self.btn_guide_perfil_izq.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; }")

        self.btn_guide_perfil_der = QtWidgets.QPushButton("Guia: Perfil Der -45°")
        self.btn_guide_perfil_der.setCheckable(True)
        self.btn_guide_perfil_der.setStyleSheet("QPushButton:checked { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; }")

        self.guide_group = QtWidgets.QButtonGroup(self)
        self.guide_group.addButton(self.btn_guide_frontal, 0)
        self.guide_group.addButton(self.btn_guide_perfil_izq, 1)
        self.guide_group.addButton(self.btn_guide_perfil_der, 2)
        self.guide_group.buttonClicked.connect(self._on_guide_button_clicked)

        guide_row.addWidget(self.btn_guide_frontal)
        guide_row.addWidget(self.btn_guide_perfil_izq)
        guide_row.addWidget(self.btn_guide_perfil_der)
        captures_layout.addLayout(guide_row)

        self.btn_capture_frontal = QtWidgets.QPushButton(
            "1: Capturar Foto Frontal + Triangular Electrodos"
        )
        self.btn_capture_frontal.setStyleSheet(
            "QPushButton { background-color: #004d40; border: 2px solid #00ffcc; padding: 8px; font-size: 12px; }"
            "QPushButton:hover { background-color: #00695c; }"
        )
        self.btn_capture_frontal.clicked.connect(self._on_capture_frontal)
        captures_layout.addWidget(self.btn_capture_frontal)

        perfiles_btn_row = QtWidgets.QHBoxLayout()
        self.btn_capture_perfil_izq = QtWidgets.QPushButton("2: Foto Perfil Izq")
        self.btn_capture_perfil_izq.setStyleSheet(
            "QPushButton { background-color: #1a3a4b; border: 1px solid #00b0ff; padding: 7px; font-size: 11px; }"
            "QPushButton:hover { background-color: #245269; }"
        )
        self.btn_capture_perfil_izq.clicked.connect(self._on_capture_perfil_izquierdo)
        perfiles_btn_row.addWidget(self.btn_capture_perfil_izq)

        self.btn_capture_perfil_der = QtWidgets.QPushButton("3: Foto Perfil Der")
        self.btn_capture_perfil_der.setStyleSheet(
            "QPushButton { background-color: #1a3a4b; border: 1px solid #00b0ff; padding: 7px; font-size: 11px; }"
            "QPushButton:hover { background-color: #245269; }"
        )
        self.btn_capture_perfil_der.clicked.connect(self._on_capture_perfil_derecho)
        perfiles_btn_row.addWidget(self.btn_capture_perfil_der)
        captures_layout.addLayout(perfiles_btn_row)

        # Referencia retrocompatible
        self.btn_capture_profile = self.btn_capture_perfil_izq
        self.chk_profile_guide = self.btn_guide_perfil_izq

        # Miniaturas de verificación 2x2
        thumbs_grid = QtWidgets.QGridLayout()
        thumbs_grid.setSpacing(6)
        self.lbl_thumb_frontal = self._create_thumbnail_label("1: Frontal 3D\nPendiente")
        self.lbl_thumb_perfil_izq = self._create_thumbnail_label("2: Perfil Izq\nPendiente")
        self.lbl_thumb_perfil_der = self._create_thumbnail_label("3: Perfil Der\nPendiente")
        self.lbl_thumb_lips = self._create_thumbnail_label("Close-up Labios\nPendiente")
        self.lbl_thumb_profile = self.lbl_thumb_perfil_izq

        thumbs_grid.addWidget(self.lbl_thumb_frontal, 0, 0)
        thumbs_grid.addWidget(self.lbl_thumb_perfil_izq, 0, 1)
        thumbs_grid.addWidget(self.lbl_thumb_perfil_der, 1, 0)
        thumbs_grid.addWidget(self.lbl_thumb_lips, 1, 1)
        captures_layout.addLayout(thumbs_grid)

        self.lbl_capture_summary = QtWidgets.QLabel(
            "Alinee el rostro dentro de la elipse guia y capture las 3 fotos de referencia."
        )
        self.lbl_capture_summary.setWordWrap(True)
        self.lbl_capture_summary.setStyleSheet("color: #00ffcc; font-size: 11px;")
        captures_layout.addWidget(self.lbl_capture_summary)

        right_col.addWidget(captures_group)

        right_col.addStretch(1)

        # 4. Botones de Confirmación / Cancelación
        action_row = QtWidgets.QHBoxLayout()
        self.btn_cancel = QtWidgets.QPushButton("Cancelar")
        self.btn_cancel.setStyleSheet(
            "QPushButton { border-color: #757575; color: #bdbdbd; }"
        )
        self.btn_cancel.clicked.connect(self.reject)
        action_row.addWidget(self.btn_cancel)

        self.btn_confirm = QtWidgets.QPushButton("Confirmar y Usar en Sesión")
        self.btn_confirm.setStyleSheet(
            "QPushButton { background-color: #00c853; color: #0a0a0a; border: 2px solid #00ffcc; "
            "padding: 10px 16px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00e676; }"
        )
        self.btn_confirm.clicked.connect(self._on_confirm_session)
        action_row.addWidget(self.btn_confirm, stretch=1)

        right_col.addLayout(action_row)
        main_layout.addLayout(right_col, stretch=2)

        # Si había calibración o imágenes previas, actualizar miniaturas
        self._refresh_thumbnails_from_state()

    @staticmethod
    def _create_metric_badge(text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setStyleSheet(
            "background-color: #2b2b2b; color: #ffffff; border: 1px solid #555555; "
            "border-radius: 4px; padding: 6px; font-weight: bold; font-size: 12px;"
        )
        return lbl

    @staticmethod
    def _create_thumbnail_label(placeholder: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(placeholder)
        lbl.setAlignment(QtCore.Qt.AlignCenter)
        lbl.setFixedSize(110, 70)
        lbl.setStyleSheet(
            "background-color: #121212; color: #757575; border: 1px dashed #444444; "
            "border-radius: 4px; font-size: 10px;"
        )
        return lbl

    def _set_badge_status(self, badge: QtWidgets.QLabel, text: str, status: str) -> None:
        if status == "ok":
            bg, fg, border = "#003d1f", "#00ff88", "#00e676"
        elif status == "warn":
            bg, fg, border = "#4d3800", "#ffd54f", "#ffb300"
        else:
            bg, fg, border = "#4a1212", "#ff8a80", "#ff5252"
        badge.setText(text)
        badge.setStyleSheet(
            f"background-color: {bg}; color: {fg}; border: 1px solid {border}; "
            "border-radius: 4px; padding: 6px; font-weight: bold; font-size: 12px;"
        )

    def _on_electrode_type_changed(self, _index: int) -> None:
        self._clear_manual_points()

    def _on_calibration_mode_changed(self, index: int) -> None:
        # 0: D_IP mm, 1: IED mm, 2: Dist Cam cm, 3: Auto
        if index == 0:
            self.lbl_calib_param.setText("D_IP Sujeto:")
            self.lbl_calib_param.setVisible(True)
            self.spin_dip_mm.setVisible(True)
            self.spin_ied_mm.setVisible(False)
            self.spin_cam_cm.setVisible(False)
        elif index == 1:
            self.lbl_calib_param.setText("IED Parches:")
            self.lbl_calib_param.setVisible(True)
            self.spin_dip_mm.setVisible(False)
            self.spin_ied_mm.setVisible(True)
            self.spin_cam_cm.setVisible(False)
        elif index == 2:
            self.lbl_calib_param.setText("Dist. Cámara:")
            self.lbl_calib_param.setVisible(True)
            self.spin_dip_mm.setVisible(False)
            self.spin_ied_mm.setVisible(False)
            self.spin_cam_cm.setVisible(True)
        else:
            self.lbl_calib_param.setVisible(False)
            self.spin_dip_mm.setVisible(False)
            self.spin_ied_mm.setVisible(False)
            self.spin_cam_cm.setVisible(False)
        self._update_preview()

    def _on_toggle_mark_gnd(self, checked: bool) -> None:
        self._marking_gnd_active = bool(checked)
        self._refresh_channel_list_ui()

    def _get_current_scale_mm_per_px(self) -> Optional[float]:
        mode = self.combo_calibration_mode.currentIndex()
        if mode == 1:  # IED nominal conocida
            ied_nominal = self.spin_ied_mm.value()
            if self._differential_pairs:
                ied_px = self._differential_pairs[0].get("ied_px", 0.0)
                if ied_px > 1.0:
                    return float(ied_nominal / ied_px)
        elif mode == 0:  # D_IP nominal conocida
            dip_mm = self.spin_dip_mm.value()
            if self._last_fiduciarios:
                dip_px = float(self._last_fiduciarios.get("distancia_interpupilar_px", 0.0))
                if dip_px > 1.0:
                    return float(dip_mm / dip_px)
        return None

    def _get_active_calibration_params(self) -> Tuple[float, Optional[float], Optional[float]]:
        mode = self.combo_calibration_mode.currentIndex()
        d_ip_real_cm = 6.3
        dist_cam_fija_cm = None
        escala_mm_px = None

        if mode == 0:
            d_ip_real_cm = float(self.spin_dip_mm.value() / 10.0)
        elif mode == 1:
            escala_mm_px = self._get_current_scale_mm_per_px()
        elif mode == 2:
            dist_cam_fija_cm = float(self.spin_cam_cm.value())

        return d_ip_real_cm, dist_cam_fija_cm, escala_mm_px

    def _refresh_channel_list_ui(self) -> None:
        self.list_channels.clear()
        total_ch = len(self.etiquetas_canales)
        es_bipolar = (self.combo_electrode_type.currentIndex() == 0)

        if self._marking_gnd_active:
            self.lbl_next_channel.setText("HAGA CLIC EN LA OREJA PARA UBICAR LA TIERRA (GND)")
            self.lbl_next_channel.setStyleSheet("color: #ffff00; font-weight: bold;")
        elif es_bipolar:
            num_pares = len(self._differential_pairs)
            if self._pending_pos_pole is not None:
                ch_idx = num_pares
                lbl = self.etiquetas_canales[ch_idx] if ch_idx < total_ch else f"Canal {ch_idx}"
                self.lbl_next_channel.setText(f"Siguiente clic: Polo (-) de {lbl}")
                self.lbl_next_channel.setStyleSheet("color: #33ccff; font-weight: bold;")
            elif num_pares < total_ch:
                lbl = self.etiquetas_canales[num_pares]
                self.lbl_next_channel.setText(f"Siguiente clic: Polo (+) de {lbl} ({num_pares}/{total_ch} canales)")
                self.lbl_next_channel.setStyleSheet("color: #ff5252; font-weight: bold;")
            else:
                self.lbl_next_channel.setText(f"[OK] Todos los pares diferenciales ({total_ch}/{total_ch}) fueron marcados.")
                self.lbl_next_channel.setStyleSheet("color: #00ffcc; font-weight: bold;")
        else:
            num_marked = len(self._manual_points_px)
            if num_marked < total_ch:
                self.lbl_next_channel.setText(
                    f"Siguiente clic asigna: {self.etiquetas_canales[num_marked]} ({num_marked}/{total_ch})"
                )
                self.lbl_next_channel.setStyleSheet("color: #00ffcc; font-weight: bold;")
            else:
                self.lbl_next_channel.setText(
                    f"[OK] Todos los canales activos ({total_ch}/{total_ch}) fueron marcados."
                )
                self.lbl_next_channel.setStyleSheet("color: #00ffcc; font-weight: bold;")

        if es_bipolar:
            for idx, label in enumerate(self.etiquetas_canales):
                if idx < len(self._differential_pairs):
                    pair = self._differential_pairs[idx]
                    ied_txt = f"{pair.get('ied_mm', 0):.1f} mm" if pair.get('ied_mm') else f"{pair.get('ied_px', 0):.0f} px"
                    item_txt = f"[OK] [{idx + 1}] {label} -> Par (+/-) [IED: {ied_txt}]"
                elif idx == len(self._differential_pairs):
                    if self._pending_pos_pole is not None:
                        item_txt = f"[>] [{idx + 1}] {label} (Polo + marcado, marque Polo -)"
                    else:
                        item_txt = f"[>] [{idx + 1}] {label} (Marque Polo +)"
                else:
                    item_txt = f"[ ] [{idx + 1}] {label} (Pendiente)"
                self.list_channels.addItem(item_txt)
        else:
            for idx, label in enumerate(self.etiquetas_canales):
                if idx < len(self._manual_points_px):
                    px, py = self._manual_points_px[idx]
                    item_txt = f"[OK] [{idx + 1}] {label} -> ({px:.0f}, {py:.0f}) px"
                elif idx == len(self._manual_points_px):
                    item_txt = f"[>] [{idx + 1}] {label} (Haga clic en la imagen)"
                else:
                    item_txt = f"[ ] [{idx + 1}] {label} (Pendiente)"
                self.list_channels.addItem(item_txt)

        if self._gnd_electrode_px is not None:
            gx, gy = self._gnd_electrode_px
            self.list_channels.addItem(f"[GND] Tierra Oreja -> ({gx:.0f}, {gy:.0f}) px")

    def _on_video_left_clicked(self, img_x: float, img_y: float) -> None:
        if self.combo_detection_mode.currentIndex() != 0:
            self.combo_detection_mode.setCurrentIndex(0)

        # Anclaje dinamico a la malla facial 3D (Skin-Tracking)
        anchor_clic = None
        if self._last_fiduciarios and "puntos_2d_px" in self._last_fiduciarios:
            pts_2d = self._last_fiduciarios["puntos_2d_px"]
            dists = np.hypot(pts_2d[:, 0] - img_x, pts_2d[:, 1] - img_y)
            v_idx = int(np.argmin(dists))
            anchor_clic = {
                "v_idx": v_idx,
                "dx": float(img_x - pts_2d[v_idx, 0]),
                "dy": float(img_y - pts_2d[v_idx, 1]),
            }

        # Si el usuario activo marcar Tierra GND
        if self._marking_gnd_active:
            self._gnd_electrode_px = (img_x, img_y)
            self._gnd_electrode_anchor = anchor_clic
            self._marking_gnd_active = False
            self.btn_mark_gnd.setChecked(False)
            self._refresh_channel_list_ui()
            self._update_preview()
            return

        es_bipolar = (self.combo_electrode_type.currentIndex() == 0)
        total_ch = len(self.etiquetas_canales)

        if es_bipolar:
            if self._pending_pos_pole is None:
                # Primer clic: Polo (+)
                if len(self._differential_pairs) < total_ch:
                    self._pending_pos_pole = (img_x, img_y)
                    self._pending_pos_anchor = anchor_clic
            else:
                # Segundo clic: Polo (-)
                pos_pt = self._pending_pos_pole
                pos_anc = self._pending_pos_anchor
                neg_pt = (img_x, img_y)
                neg_anc = anchor_clic
                self._pending_pos_pole = None
                self._pending_pos_anchor = None

                ch_idx = len(self._differential_pairs)
                ch_num = self.active_channel_indices[ch_idx] if ch_idx < len(self.active_channel_indices) else ch_idx
                musc = self.nombres_musculos[ch_idx] if ch_idx < len(self.nombres_musculos) else f"Músculo {ch_num}"

                cx = 0.5 * (pos_pt[0] + neg_pt[0])
                cy = 0.5 * (pos_pt[1] + neg_pt[1])
                ied_px = float(np.hypot(neg_pt[0] - pos_pt[0], neg_pt[1] - pos_pt[1]))
                ang_deg = float(np.degrees(np.arctan2(neg_pt[1] - pos_pt[1], neg_pt[0] - pos_pt[0])))

                escala = self._get_current_scale_mm_per_px()
                ied_mm = (ied_px * escala) if escala is not None else None

                pair_info = {
                    "es_bipolar": True,
                    "polo_positivo": pos_pt,
                    "polo_negativo": neg_pt,
                    "anchor_positivo": pos_anc,
                    "anchor_negativo": neg_anc,
                    "ied_px": ied_px,
                    "ied_mm": ied_mm,
                    "angulo_deg": ang_deg,
                    "canal_fisico": ch_num,
                    "nombre": musc,
                    "diametro_gel_mm": float(self.spin_gel_diam_mm.value()),
                    "distancia_interelectrodo_nominal_mm": float(self.spin_ied_nominal_mm.value()),
                }
                self._differential_pairs.append(pair_info)
        else:
            max_channels = max(len(self.etiquetas_canales), 1)
            if len(self._manual_points_px) < max_channels:
                self._manual_points_px.append((img_x, img_y))
                self._manual_points_anchors.append(anchor_clic)
            else:
                self._manual_points_px[-1] = (img_x, img_y)
                if self._manual_points_anchors:
                    self._manual_points_anchors[-1] = anchor_clic

        self._refresh_channel_list_ui()
        self._update_preview()

    def _on_video_right_clicked(self, _img_x: float, _img_y: float) -> None:
        if self._marking_gnd_active:
            self._marking_gnd_active = False
            self.btn_mark_gnd.setChecked(False)
        elif self._pending_pos_pole is not None:
            self._pending_pos_pole = None
            self._pending_pos_anchor = None
        elif self.combo_electrode_type.currentIndex() == 0:
            if self._differential_pairs:
                self._differential_pairs.pop()
            if self._manual_points_px:
                self._manual_points_px.pop()
            if self._manual_points_anchors:
                self._manual_points_anchors.pop()
        else:
            if self._manual_points_px:
                self._manual_points_px.pop()
            if self._manual_points_anchors:
                self._manual_points_anchors.pop()
        self._refresh_channel_list_ui()
        self._update_preview()

    def _clear_manual_points(self) -> None:
        self._manual_points_px.clear()
        self._manual_points_anchors.clear()
        self._differential_pairs.clear()
        self._pending_pos_pole = None
        self._pending_pos_anchor = None
        self._gnd_electrode_px = None
        self._gnd_electrode_anchor = None
        self._marking_gnd_active = False
        self.btn_mark_gnd.setChecked(False)
        self._refresh_channel_list_ui()
        self._update_preview()

    def _on_toggle_freeze(self, checked: bool) -> None:
        if checked and self._last_live_frame is not None:
            self._frozen_frame = self._last_live_frame.copy()
            self.btn_freeze.setText("Reanudar Video en Vivo")
        else:
            self._frozen_frame = None
            self.btn_freeze.setText("Congelar Cuadro para Marcar")
        self._update_preview()

    def _on_toggle_profile_mode(self, checked: bool) -> None:
        self._profile_45_mode = bool(checked)
        self._update_preview()

    def _on_load_ghost_reference(self) -> None:
        start_dir = os.getcwd()
        filepath, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Seleccionar calibracion_espacial.json de sesión anterior",
            start_dir,
            "Archivos JSON (*.json);;Todos los archivos (*)",
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadatos_calibracion_espacial", data)
            electrodos = meta.get("electrodos", [])
            if not electrodos:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Referencia sin electrodos",
                    "El archivo JSON seleccionado no contiene vectores de electrodos triangulados.",
                )
                return

            self._ghost_reference_data = data
            self._ghost_reference_path = filepath
            self.chk_show_ghost.setEnabled(True)
            self.chk_show_ghost.setChecked(True)
            self.lbl_ghost_status.setText(
                f"[Ghost activo] {os.path.basename(filepath)} ({len(electrodos)} electrodos)"
            )
            self.lbl_ghost_status.setStyleSheet("color: #00ffcc; font-size: 11px; font-weight: bold;")
            self._update_preview()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Error al cargar referencia",
                f"No se pudo leer el archivo JSON:\n{exc}",
            )

    def _get_current_source_frame(self) -> Optional[np.ndarray]:
        if self._frozen_frame is not None:
            return self._frozen_frame.copy()
        if self.capture_thread is not None:
            frame, _ = self.capture_thread.get_latest_frame()
            if frame is not None:
                self._last_live_frame = frame
                return frame
        if self._last_live_frame is not None:
            return self._last_live_frame.copy()
        return None

    def _update_preview(self) -> None:
        frame = self._get_current_source_frame()
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                placeholder,
                "Esperando senal de camara web (640x480)...",
                (75, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (204, 255, 0),
                2,
                cv2.LINE_AA,
            )
            self._display_bgr_on_label(placeholder, self.video_label)
            return

        alto, ancho = frame.shape[:2]
        self.video_label.set_image_dimensions(ancho, alto)

        fiduciarios = self.agent.extraer_malla_facial_3d(frame)
        self._last_fiduciarios = fiduciarios

        canvas = frame.copy()
        self._draw_alignment_guides_and_hud(canvas, fiduciarios)

        # Dibujar Ghost Overlay si está habilitado
        if (
            fiduciarios is not None
            and self._ghost_reference_data is not None
            and self.chk_show_ghost.isChecked()
        ):
            self._draw_ghost_overlay(canvas, fiduciarios)

        # Dibujar electrodos manuales o detectados automáticamente en vivo
        self._draw_live_electrodes_preview(canvas, frame, fiduciarios)

        self._display_bgr_on_label(canvas, self.video_label)

    def _draw_alignment_guides_and_hud(
        self,
        canvas: np.ndarray,
        fiduciarios: Optional[Dict[str, Any]],
    ) -> None:
        alto, ancho = canvas.shape[:2]
        cx, cy = ancho // 2, alto // 2
        rx, ry = int(ancho * 0.21), int(alto * 0.34)
        eye_line_y = int(cy - ry * 0.22)

        if fiduciarios is None:
            # Rostro no detectado
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, (0, 80, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (cx - rx, eye_line_y), (cx + rx, eye_line_y), (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                "UBIQUE EL ROSTRO DENTRO DE LA ELIPSE GUIA",
                (max(20, cx - 210), 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 100, 255),
                2,
                cv2.LINE_AA,
            )
            self._set_badge_status(self.badge_yaw, "Yaw: Sin rostro", "bad")
            self._set_badge_status(self.badge_pitch, "Pitch: Sin rostro", "bad")
            self._set_badge_status(self.badge_roll, "Roll: Sin rostro", "bad")
            self._set_badge_status(self.badge_dist, "Distancia: --", "bad")
            return

        d_ip_cm, dist_fija_cm, escala_mm_px = self._get_active_calibration_params()
        pose = self.agent.calcular_pose_y_biometria_facial(
            fiduciarios,
            (alto, ancho),
            d_ip_real_cm=d_ip_cm,
            distancia_fija_camara_cm=dist_fija_cm,
            escala_mm_por_px=escala_mm_px,
        )
        self._last_pose_metrics = pose

        yaw = pose["yaw_deg"]
        pitch = pose["pitch_deg"]
        roll = pose["roll_deg"]
        dist_cm = pose["distancia_camara_cm"]
        metodo_d = pose.get("metodo_distancia", "antropometria")

        # Evaluar tolerancias según pose activa (Frontal, Perfil Izq, Perfil Der)
        target_mode = getattr(self, "_alignment_pose_target", "frontal")
        if target_mode == "perfil_izquierdo":
            yaw_ok = 30.0 <= yaw <= 55.0
            yaw_warn = 22.0 <= yaw <= 62.0
            modo_str = "+45°"
            target_str = "Perfil Izquierdo: Gire cabeza a la derecha - Yaw +45 deg"
        elif target_mode == "perfil_derecho":
            yaw_ok = -55.0 <= yaw <= -30.0
            yaw_warn = -62.0 <= yaw <= -22.0
            modo_str = "-45°"
            target_str = "Perfil Derecho: Gire cabeza a la izquierda - Yaw -45 deg"
        elif self._profile_45_mode:
            yaw_ok = 30.0 <= abs(yaw) <= 55.0
            yaw_warn = 22.0 <= abs(yaw) <= 62.0
            modo_str = "45°"
            target_str = "Modo Perfil: Gire rostro hasta 30 a 55 grados"
        else:
            yaw_ok = abs(yaw) <= 6.0
            yaw_warn = abs(yaw) <= 10.0
            modo_str = "0°"
            target_str = "Ajuste giro e inclinacion hasta poner semaforos en verde"

        pitch_ok = abs(pitch) <= 8.0
        pitch_warn = abs(pitch) <= 13.0
        roll_ok = abs(roll) <= 4.0
        roll_warn = abs(roll) <= 7.5
        dist_ok = 30.0 <= dist_cm <= 70.0
        dist_warn = 24.0 <= dist_cm <= 82.0

        all_aligned = yaw_ok and pitch_ok and roll_ok and dist_ok
        guide_color = (118, 230, 0) if all_aligned else (0, 215, 255)

        # Elipse guía y retículo nivel ocular
        cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, guide_color, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - rx - 15, eye_line_y), (cx + rx + 15, eye_line_y), guide_color, 1, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy - ry - 10), (cx, cy + ry + 10), (90, 90, 90), 1, cv2.LINE_AA)

        # Dibujar línea interpupilar y origen nasal
        pt_ojo_izq = (int(fiduciarios["ojo_izq_px"][0]), int(fiduciarios["ojo_izq_px"][1]))
        pt_ojo_der = (int(fiduciarios["ojo_der_px"][0]), int(fiduciarios["ojo_der_px"][1]))
        pt_nariz = (int(fiduciarios["punta_nariz_px"][0]), int(fiduciarios["punta_nariz_px"][1]))

        cv2.line(canvas, pt_ojo_izq, pt_ojo_der, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_ojo_izq, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_ojo_der, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_nariz, 5, (0, 0, 255), -1, cv2.LINE_AA)

        # Visualización de Orejas (tragos y lóbulos)
        if self.chk_show_ears.isChecked() and "oreja_trago_izq_px" in fiduciarios:
            pt_t_izq = (int(fiduciarios["oreja_trago_izq_px"][0]), int(fiduciarios["oreja_trago_izq_px"][1]))
            pt_t_der = (int(fiduciarios["oreja_trago_der_px"][0]), int(fiduciarios["oreja_trago_der_px"][1]))
            pt_l_izq = (int(fiduciarios["oreja_lobulo_izq_px"][0]), int(fiduciarios["oreja_lobulo_izq_px"][1]))
            pt_l_der = (int(fiduciarios["oreja_lobulo_der_px"][0]), int(fiduciarios["oreja_lobulo_der_px"][1]))

            cv2.circle(canvas, pt_t_izq, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_t_izq, pt_l_izq, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_l_izq, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja L", (pt_t_izq[0] - 50, pt_t_izq[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

            cv2.circle(canvas, pt_t_der, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_t_der, pt_l_der, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_l_der, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja R", (pt_t_der[0] + 6, pt_t_der[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

        # Contorno de labios y mentón para referencia biométrica
        pts_2d = fiduciarios["puntos_2d_px"]
        for lm_idx in (
            INDICE_LABIO_SUP_INTERNO,
            INDICE_LABIO_INF_INTERNO,
            INDICE_COMISURA_IZQ,
            INDICE_COMISURA_DER,
            INDICE_MENTON_GNATHION,
        ):
            if lm_idx < len(pts_2d):
                lx, ly = int(pts_2d[lm_idx, 0]), int(pts_2d[lm_idx, 1])
                cv2.circle(canvas, (lx, ly), 3, (204, 255, 0), -1, cv2.LINE_AA)

        # Actualizar badges en el panel Qt
        self._set_badge_status(
            self.badge_yaw,
            f"Yaw - {modo_str}: {yaw:+.1f}°",
            "ok" if yaw_ok else ("warn" if yaw_warn else "bad"),
        )
        self._set_badge_status(
            self.badge_pitch,
            f"Pitch: {pitch:+.1f}°",
            "ok" if pitch_ok else ("warn" if pitch_warn else "bad"),
        )
        self._set_badge_status(
            self.badge_roll,
            f"Roll: {roll:+.1f}°",
            "ok" if roll_ok else ("warn" if roll_warn else "bad"),
        )

        dist_tag = "Auto" if "antropometria" in metodo_d else ("D_IP" if "d_ip" in metodo_d else ("IED" if "ied" in metodo_d else "Fija"))
        self._set_badge_status(
            self.badge_dist,
            f"Dist: {dist_cm:.1f}cm ({dist_tag})",
            "ok" if dist_ok else ("warn" if dist_warn else "bad"),
        )

        ancho_orejas_txt = f"{pose.get('ancho_biauricular_cm', 0.0):.1f} cm" if pose.get('ancho_biauricular_cm') else "--"
        self.lbl_biometria_live.setText(
            f"Labios: {pose['apertura_labial_norm']:.3f} | "
            f"Comisuras: {pose['ancho_comisuras_norm']:.3f} | "
            f"Mandíbula: {pose['apertura_mandibular_norm']:.3f} | "
            f"Orejas: {ancho_orejas_txt}"
        )

        # Banner superior sobre el video
        banner_bg = (20, 75, 20) if all_aligned else (20, 45, 70)
        cv2.rectangle(canvas, (10, 8), (ancho - 10, 40), banner_bg, -1)
        cv2.rectangle(canvas, (10, 8), (ancho - 10, 40), guide_color, 1)
        estado_txt = (
            "ALINEACION OPTIMA (VERDE) - LISTO PARA CAPTURAR"
            if all_aligned
            else target_str.upper()
        )
        cv2.putText(
            canvas,
            estado_txt,
            (20, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_ghost_overlay(self, canvas: np.ndarray, fiduciarios: Dict[str, Any]) -> None:
        if self._ghost_reference_data is None:
            return
        proyecciones = self.agent.proyectar_referencia_fantasma(fiduciarios, self._ghost_reference_data)
        for idx, item in enumerate(proyecciones):
            tx, ty = int(item["target_2d_px"][0]), int(item["target_2d_px"][1])
            cv2.circle(canvas, (tx, ty), 13, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, (tx, ty), 6, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.line(canvas, (tx - 16, ty), (tx + 16, ty), (255, 255, 0), 1, cv2.LINE_AA)
            cv2.line(canvas, (tx, ty - 16), (tx, ty + 16), (255, 255, 0), 1, cv2.LINE_AA)

            etiqueta = f"Ghost {item.get('canal', idx)}"
            if idx < len(self._manual_points_px):
                mx, my = self._manual_points_px[idx]
                err_px = float(np.hypot(mx - tx, my - ty))
                etiqueta += f" ({err_px:.1f}px)"
            cv2.putText(
                canvas,
                etiqueta,
                (tx + 14, ty - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_live_electrodes_preview(
        self,
        canvas: np.ndarray,
        raw_frame: np.ndarray,
        fiduciarios: Optional[Dict[str, Any]],
    ) -> None:
        mode_idx = self.combo_detection_mode.currentIndex()
        es_bipolar = (self.combo_electrode_type.currentIndex() == 0)

        nariz_pt = (
            (int(fiduciarios["punta_nariz_px"][0]), int(fiduciarios["punta_nariz_px"][1]))
            if fiduciarios is not None
            else None
        )

        colores = [
            (0, 255, 128),
            (0, 190, 255),
            (255, 100, 255),
            (255, 255, 0),
            (100, 255, 255),
            (255, 150, 100),
        ]

        # Calculo de factor de escala para radio fisico en pixeles
        escala_preview = self._get_current_scale_mm_per_px()
        if escala_preview is None or escala_preview <= 0:
            if fiduciarios is not None and "distancia_interpupilar_px" in fiduciarios:
                d_ip_px = float(fiduciarios["distancia_interpupilar_px"])
                if d_ip_px > 1e-4:
                    d_ip_mm = float(self.spin_dip_mm.value())
                    escala_preview = d_ip_mm / d_ip_px

        # Actualizacion dinamica de coordenadas ancladas a la piel facial (Skin-Tracking)
        if fiduciarios is not None and "puntos_2d_px" in fiduciarios:
            pts_2d = fiduciarios["puntos_2d_px"]
            for pair in self._differential_pairs:
                anc_pos = pair.get("anchor_positivo")
                if anc_pos and anc_pos.get("v_idx") is not None and anc_pos["v_idx"] < len(pts_2d):
                    v_p = anc_pos["v_idx"]
                    pair["polo_positivo"] = (
                        float(pts_2d[v_p, 0] + anc_pos.get("dx", 0.0)),
                        float(pts_2d[v_p, 1] + anc_pos.get("dy", 0.0)),
                    )
                anc_neg = pair.get("anchor_negativo")
                if anc_neg and anc_neg.get("v_idx") is not None and anc_neg["v_idx"] < len(pts_2d):
                    v_n = anc_neg["v_idx"]
                    pair["polo_negativo"] = (
                        float(pts_2d[v_n, 0] + anc_neg.get("dx", 0.0)),
                        float(pts_2d[v_n, 1] + anc_neg.get("dy", 0.0)),
                    )

            if self._pending_pos_pole is not None and getattr(self, "_pending_pos_anchor", None):
                anc_pend = self._pending_pos_anchor
                if anc_pend.get("v_idx") is not None and anc_pend["v_idx"] < len(pts_2d):
                    v_pen = anc_pend["v_idx"]
                    self._pending_pos_pole = (
                        float(pts_2d[v_pen, 0] + anc_pend.get("dx", 0.0)),
                        float(pts_2d[v_pen, 1] + anc_pend.get("dy", 0.0)),
                    )

            if self._gnd_electrode_px is not None and getattr(self, "_gnd_electrode_anchor", None):
                anc_gnd = self._gnd_electrode_anchor
                if anc_gnd.get("v_idx") is not None and anc_gnd["v_idx"] < len(pts_2d):
                    v_gnd = anc_gnd["v_idx"]
                    self._gnd_electrode_px = (
                        float(pts_2d[v_gnd, 0] + anc_gnd.get("dx", 0.0)),
                        float(pts_2d[v_gnd, 1] + anc_gnd.get("dy", 0.0)),
                    )

            if hasattr(self, "_manual_points_anchors"):
                for m_i, anc_m in enumerate(self._manual_points_anchors):
                    if anc_m and m_i < len(self._manual_points_px) and anc_m.get("v_idx") is not None and anc_m["v_idx"] < len(pts_2d):
                        v_m = anc_m["v_idx"]
                        self._manual_points_px[m_i] = (
                            float(pts_2d[v_m, 0] + anc_m.get("dx", 0.0)),
                            float(pts_2d[v_m, 1] + anc_m.get("dy", 0.0)),
                        )

        if mode_idx == 0 and es_bipolar:
            # Dibujar pares diferenciales bipolares (dos polos fisicos reales: + y -)
            for idx, pair in enumerate(self._differential_pairs):
                color = colores[idx % len(colores)]
                pos_x, pos_y = int(round(pair["polo_positivo"][0])), int(round(pair["polo_positivo"][1]))
                neg_x, neg_y = int(round(pair["polo_negativo"][0])), int(round(pair["polo_negativo"][1]))

                # Flecha que representa el vector dipolo diferencial D = P_+ - P_- (de negativo a positivo)
                cv2.arrowedLine(canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25)

                # Radio fisico a escala del parche/gel alrededor de cada polo
                diam_gel = float(pair.get("diametro_gel_mm", self.spin_gel_diam_mm.value()))
                radio_px = (
                    max(int(round((diam_gel / 2.0) / escala_preview)), 4)
                    if (escala_preview is not None and escala_preview > 0)
                    else 12
                )
                cv2.circle(canvas, (pos_x, pos_y), radio_px, (0, 255, 204), 1, cv2.LINE_AA)
                cv2.circle(canvas, (neg_x, neg_y), radio_px, (0, 255, 204), 1, cv2.LINE_AA)

                # Polo (+) en rojo/naranja
                cv2.circle(canvas, (pos_x, pos_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (pos_x, pos_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "+", (pos_x - 4, pos_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Polo (-) en azul eléctrico
                cv2.circle(canvas, (neg_x, neg_y), 7, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, (neg_x, neg_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "-", (neg_x - 3, neg_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Líneas fiduciarias tenues desde la nariz a cada electrodo individual
                if nariz_pt is not None:
                    cv2.line(canvas, nariz_pt, (pos_x, pos_y), color, 1, cv2.LINE_AA)
                    cv2.line(canvas, nariz_pt, (neg_x, neg_y), color, 1, cv2.LINE_AA)

                ied_txt = f"{pair['ied_mm']:.1f}mm" if pair.get("ied_mm") else f"{pair['ied_px']:.0f}px"
                ch_label = self.etiquetas_canales[idx] if idx < len(self.etiquetas_canales) else f"Canal {idx}"
                mid_x = int(round(0.5 * (pos_x + neg_x)))
                mid_y = int(round(0.5 * (pos_y + neg_y)))
                cv2.putText(
                    canvas,
                    f"{ch_label} [{ied_txt}]",
                    (mid_x + 10, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            # Si hay un polo (+) pendiente de completar con su polo (-)
            if self._pending_pos_pole is not None:
                px, py = int(round(self._pending_pos_pole[0])), int(round(self._pending_pos_pole[1]))
                diam_gel_p = float(self.spin_gel_diam_mm.value())
                radio_px_p = (
                    max(int(round((diam_gel_p / 2.0) / escala_preview)), 4)
                    if (escala_preview is not None and escala_preview > 0)
                    else 12
                )
                cv2.circle(canvas, (px, py), radio_px_p, (0, 255, 204), 1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 7, (0, 69, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "+ [Marque Polo -]", (px + 10, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        else:
            # Monopolar o Automático
            pts_to_draw = []
            if mode_idx == 0:
                pts_to_draw = list(self._manual_points_px)
            elif mode_idx == 1:
                _, dets = self.agent.segmentar_marcadores_hsv(raw_frame)
                pts_to_draw = [d["centroide"] for d in dets]
            else:
                _, dets = self.agent.detectar_broches_metalicos(raw_frame)
                pts_to_draw = [d["centroide"] for d in dets]

            for idx, (px, py) in enumerate(pts_to_draw):
                color = colores[idx % len(colores)]
                ix, iy = int(round(px)), int(round(py))
                diam_gel_m = float(self.spin_gel_diam_mm.value())
                radio_px_m = (
                    max(int(round((diam_gel_m / 2.0) / escala_preview)), 4)
                    if (escala_preview is not None and escala_preview > 0)
                    else 12
                )
                cv2.circle(canvas, (ix, iy), radio_px_m, color, 1, cv2.LINE_AA)
                if nariz_pt is not None:
                    cv2.line(canvas, nariz_pt, (ix, iy), color, 1, cv2.LINE_AA)
                cv2.circle(canvas, (ix, iy), 7, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (ix, iy), 9, (255, 255, 255), 1, cv2.LINE_AA)

                ch_label = (
                    self.etiquetas_canales[idx]
                    if idx < len(self.etiquetas_canales)
                    else f"Electrodo {idx}"
                )
                cv2.putText(
                    canvas,
                    ch_label,
                    (ix + 11, iy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Detección de orejas en vivo si está habilitado en la UI
        if self.chk_show_ears.isChecked() and fiduciarios is not None and "oreja_trago_izq_px" in fiduciarios:
            pt_t_izq = (int(fiduciarios["oreja_trago_izq_px"][0]), int(fiduciarios["oreja_trago_izq_px"][1]))
            pt_t_der = (int(fiduciarios["oreja_trago_der_px"][0]), int(fiduciarios["oreja_trago_der_px"][1]))
            pt_l_izq = (int(fiduciarios["oreja_lobulo_izq_px"][0]), int(fiduciarios["oreja_lobulo_izq_px"][1]))
            pt_l_der = (int(fiduciarios["oreja_lobulo_der_px"][0]), int(fiduciarios["oreja_lobulo_der_px"][1]))

            cv2.circle(canvas, pt_t_izq, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_t_izq, pt_l_izq, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_l_izq, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja L", (pt_t_izq[0] - 50, pt_t_izq[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

            cv2.circle(canvas, pt_t_der, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_t_der, pt_l_der, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_l_der, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja R", (pt_t_der[0] + 8, pt_t_der[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

        # Dibujar electrodo de Tierra (GND) si fue marcado en la oreja
        if self._gnd_electrode_px is not None:
            gx, gy = int(round(self._gnd_electrode_px[0])), int(round(self._gnd_electrode_px[1]))
            cv2.circle(canvas, (gx, gy), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 11, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "GND (Tierra Oreja)", (gx + 12, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

    @staticmethod
    def _crop_closeup_labios(
        bgr_image: np.ndarray,
        fiduciarios: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        """Recorta automáticamente la región perioral/submentoniana (boca y mentón)."""
        alto, ancho = bgr_image.shape[:2]
        if fiduciarios is not None and "puntos_2d_px" in fiduciarios:
            pts_2d = fiduciarios["puntos_2d_px"]
            indices_boca = [
                INDICE_NARIZ_PUNTA,
                INDICE_COMISURA_IZQ,
                INDICE_COMISURA_DER,
                INDICE_LABIO_SUP_INTERNO,
                INDICE_LABIO_INF_INTERNO,
                INDICE_MENTON_GNATHION,
            ]
            coords = np.array([pts_2d[i] for i in indices_boca if i < len(pts_2d)], dtype=np.float64)
            if len(coords) > 0:
                min_x, min_y = np.min(coords, axis=0)
                max_x, max_y = np.max(coords, axis=0)
                pad_x = max(35.0, (max_x - min_x) * 0.48)
                pad_y_top = max(20.0, (max_y - min_y) * 0.25)
                pad_y_bot = max(35.0, (max_y - min_y) * 0.45)

                x1 = int(np.clip(min_x - pad_x, 0, ancho - 1))
                x2 = int(np.clip(max_x + pad_x, x1 + 20, ancho))
                y1 = int(np.clip(min_y - pad_y_top, 0, alto - 1))
                y2 = int(np.clip(max_y + pad_y_bot, y1 + 20, alto))
                crop = bgr_image[y1:y2, x1:x2].copy()
                if crop.size > 0:
                    return crop

        # Fallback centrado en tercio inferior del rostro
        x1, x2 = int(ancho * 0.28), int(ancho * 0.72)
        y1, y2 = int(alto * 0.45), int(alto * 0.88)
        return bgr_image[y1:y2, x1:x2].copy()

    def _on_capture_frontal(self) -> bool:
        frame = self._get_current_source_frame()
        if frame is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Cámara no disponible",
                "No se recibió ningún cuadro desde la cámara web.",
            )
            return False

        mode_idx = self.combo_detection_mode.currentIndex()
        es_bipolar = (self.combo_electrode_type.currentIndex() == 0)
        usar_manual = mode_idx == 0 or len(self._manual_points_px) > 0 or len(self._differential_pairs) > 0
        usar_metal = mode_idx == 2

        d_ip_cm, dist_fija_cm, escala_mm_px = self._get_active_calibration_params()

        puntos_manuales = None
        if usar_manual:
            if es_bipolar and len(self._differential_pairs) > 0:
                puntos_manuales = list(self._differential_pairs)
            elif len(self._manual_points_px) > 0:
                puntos_manuales = list(self._manual_points_px)

        resultado = self.agent.triangular_electrodos(
            frame,
            puntos_centroides_manuales=puntos_manuales,
            usar_broches_metalicos=usar_metal,
            asignar_musculos_automatico=True,
            nombres_canales=self.nombres_musculos,
            indices_canales_fisicos=self.active_channel_indices,
            d_ip_real_cm=d_ip_cm,
            distancia_fija_camara_cm=dist_fija_cm,
            escala_mm_por_px=escala_mm_px,
            punto_referencia_gnd=self._gnd_electrode_px,
            diametro_gel_mm=float(self.spin_gel_diam_mm.value()),
            ied_nominal_mm=float(self.spin_ied_nominal_mm.value()),
        )

        fiduciarios = resultado.get("_fiduciarios_cache") or self.agent.extraer_malla_facial_3d(frame)
        vis_frontal = self.agent.renderizar_visualizacion(frame, resultado)
        closeup_labios = self._crop_closeup_labios(frame, fiduciarios)

        self.reference_images["frontal_cruda"] = frame.copy()
        self.reference_images["frontal_triangulada"] = vis_frontal.copy()
        self.reference_images["closeup_labios"] = closeup_labios.copy()

        meta = resultado.get("metadatos_calibracion_espacial", {})
        meta["sujeto"] = self.sujeto
        meta["prueba"] = self.prueba
        meta["modo_deteccion"] = (
            "bipolar_manual" if (usar_manual and es_bipolar) else ("monopolar_manual" if usar_manual else ("broches_metalicos" if usar_metal else "hsv_verde_fluor"))
        )
        meta["canales_activos_fisicos"] = list(self.active_channel_indices)
        meta["nombres_musculos_canales"] = list(self.nombres_musculos)
        if self._ghost_reference_path:
            meta["referencia_fantasma_origen"] = self._ghost_reference_path

        self.calibracion_inicial_data = meta
        self._refresh_thumbnails_from_state()

        n_elec = len(meta.get("electrodos", []))
        n_pares = len(meta.get("pares_diferenciales", []))
        exito_rostro = resultado.get("exito", False)
        if exito_rostro:
            ch_info = f"{n_pares} pares diferenciales ({n_elec} electrodos)" if n_pares > 0 else f"{n_elec} canales"
            self.lbl_capture_summary.setText(
                f"[OK] Frontal capturada: {ch_info} triangulados | "
                f"Dist: {meta.get('pose_y_biometria_facial', {}).get('distancia_camara_cm', 0.0):.1f} cm."
            )
        else:
            self.lbl_capture_summary.setText(
                "[Aviso] Captura frontal guardada (sin malla 3D detectada en este cuadro)."
            )
        return True

    def _on_guide_button_clicked(self, button: QtWidgets.QAbstractButton) -> None:
        if button == self.btn_guide_perfil_izq:
            self._alignment_pose_target = "perfil_izquierdo"
            self._profile_45_mode = True
        elif button == self.btn_guide_perfil_der:
            self._alignment_pose_target = "perfil_derecho"
            self._profile_45_mode = True
        else:
            self._alignment_pose_target = "frontal"
            self._profile_45_mode = False
        self._update_preview()

    def _on_capture_perfil_izquierdo(self) -> None:
        if self._alignment_pose_target != "perfil_izquierdo":
            self.btn_guide_perfil_izq.setChecked(True)
            self._alignment_pose_target = "perfil_izquierdo"
            self._profile_45_mode = True

        frame = self._get_current_source_frame()
        if frame is None:
            return

        fiduciarios = self.agent.extraer_malla_facial_3d(frame)
        pose = (
            self.agent.calcular_pose_y_biometria_facial(fiduciarios, frame.shape[:2])
            if fiduciarios is not None
            else {}
        )
        vis_perfil = frame.copy()
        yaw_val = pose.get("yaw_deg", 0.0)
        dist_val = pose.get("distancia_camara_cm", 0.0)
        cv2.rectangle(vis_perfil, (10, 10), (450, 62), (0, 0, 0), -1)
        cv2.rectangle(vis_perfil, (10, 10), (450, 62), (0, 255, 204), 1)
        cv2.putText(
            vis_perfil,
            f"Referencia Perfil Izquierdo - {self.sujeto}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_perfil,
            f"Yaw: {yaw_val:+.1f} deg | Dist: {dist_val:.1f} cm",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 204),
            1,
            cv2.LINE_AA,
        )

        self.reference_images["perfil_izquierdo"] = vis_perfil
        self.reference_images["perfil_45"] = vis_perfil
        if self.calibracion_inicial_data is not None:
            self.calibracion_inicial_data["pose_perfil_izquierdo"] = pose
            self.calibracion_inicial_data["pose_perfil_45"] = pose

        self._refresh_thumbnails_from_state()
        self.lbl_capture_summary.setText(
            f"[OK] Perfil Izquierdo capturado - Yaw: {yaw_val:+.1f} deg. Proceda a Perfil Derecho o Confirmar."
        )

    def _on_capture_perfil_derecho(self) -> None:
        if self._alignment_pose_target != "perfil_derecho":
            self.btn_guide_perfil_der.setChecked(True)
            self._alignment_pose_target = "perfil_derecho"
            self._profile_45_mode = True

        frame = self._get_current_source_frame()
        if frame is None:
            return

        fiduciarios = self.agent.extraer_malla_facial_3d(frame)
        pose = (
            self.agent.calcular_pose_y_biometria_facial(fiduciarios, frame.shape[:2])
            if fiduciarios is not None
            else {}
        )
        vis_perfil = frame.copy()
        yaw_val = pose.get("yaw_deg", 0.0)
        dist_val = pose.get("distancia_camara_cm", 0.0)
        cv2.rectangle(vis_perfil, (10, 10), (450, 62), (0, 0, 0), -1)
        cv2.rectangle(vis_perfil, (10, 10), (450, 62), (0, 255, 204), 1)
        cv2.putText(
            vis_perfil,
            f"Referencia Perfil Derecho - {self.sujeto}",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            vis_perfil,
            f"Yaw: {yaw_val:+.1f} deg | Dist: {dist_val:.1f} cm",
            (20, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 204),
            1,
            cv2.LINE_AA,
        )

        self.reference_images["perfil_derecho"] = vis_perfil
        if self.calibracion_inicial_data is not None:
            self.calibracion_inicial_data["pose_perfil_derecho"] = pose

        self._refresh_thumbnails_from_state()
        self.lbl_capture_summary.setText(
            f"[OK] Perfil Derecho capturado - Yaw: {yaw_val:+.1f} deg. Calibracion de 3 fotos completa."
        )

    def _on_capture_profile_45(self) -> None:
        self._on_capture_perfil_izquierdo()

    def _on_confirm_session(self) -> None:
        tiene_puntos_marcados = bool(self._differential_pairs or self._manual_points_px)
        electrodos_guardados = (self.calibracion_inicial_data or {}).get("electrodos", [])
        pares_guardados = (self.calibracion_inicial_data or {}).get("pares_diferenciales", [])
        if (
            "frontal_cruda" not in self.reference_images
            or self.calibracion_inicial_data is None
            or (tiene_puntos_marcados and not electrodos_guardados and not pares_guardados)
            or (len(self._differential_pairs) > len(pares_guardados))
            or (len(self._manual_points_px) > len(electrodos_guardados))
        ):
            self._on_capture_frontal()
        self.accept()

    def _refresh_thumbnails_from_state(self) -> None:
        if "frontal_triangulada" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["frontal_triangulada"], self.lbl_thumb_frontal
            )
        if "closeup_labios" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["closeup_labios"], self.lbl_thumb_lips
            )
        if "perfil_izquierdo" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["perfil_izquierdo"], self.lbl_thumb_perfil_izq
            )
        elif "perfil_45" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["perfil_45"], self.lbl_thumb_perfil_izq
            )
        if "perfil_derecho" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["perfil_derecho"], self.lbl_thumb_perfil_der
            )

    @staticmethod
    def _display_bgr_on_label(bgr_img: np.ndarray, label: QtWidgets.QLabel) -> None:
        if bgr_img is None or bgr_img.size == 0:
            return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        label.setPixmap(scaled)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        self._timer.stop()
        super().done(result)


# ==============================================================================
# Asistente Lineal Simple de Fotometria y Guia AR en Vivo - Montaje de Electrodos
# ==============================================================================

NOMBRE_FOTO_FRONTAL = "01_frontal.png"
NOMBRE_FOTO_PERFIL_IZQUIERDO = "02_perfil_izquierdo.png"
NOMBRE_FOTO_PERFIL_DERECHO = "03_perfil_derecho.png"
NOMBRE_METADATA_FOTOMETRIA = "metadata_fotometria.json"
D_IP_NOMINAL_DEFECTO_MM = 63.0
TICKS_VERDE_CAPTURA = 16

PALETA_CANALES_MONTAJE = [
    (0, 255, 128),
    (0, 190, 255),
    (255, 100, 255),
    (255, 255, 0),
    (100, 255, 255),
    (255, 150, 100),
]
COLOR_FANTASMA = (255, 255, 0)
COLOR_FANTASMA_LINEA = (170, 200, 90)


def resolver_directorio_proyecto() -> str:
    """Resuelve la raiz del proyecto en ejecucion dev y en distribucion empaquetada."""
    if getattr(sys, "frozen", False):
        raiz = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.basename(raiz) == "_internal":
            raiz = os.path.dirname(raiz)
    else:
        raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return raiz


def listar_sesiones_fotometria_previas() -> List[Dict[str, Any]]:
    """
    Escanea base_de_datos_electrodos y lista las mediciones previas guardadas
    que incluyen metadata_fotometria.json, de la mas reciente a la mas antigua.

    Retorna elementos con: etiqueta, ruta_meta, carpeta_sesion, fecha y nombre.
    """
    sesiones: List[Dict[str, Any]] = []
    raiz_base = os.path.join(resolver_directorio_proyecto(), "base_de_datos_electrodos")
    if not os.path.isdir(raiz_base):
        return sesiones
    fechas_ordenadas = sorted(
        [d for d in os.listdir(raiz_base) if os.path.isdir(os.path.join(raiz_base, d))],
        reverse=True,
    )
    for fecha in fechas_ordenadas:
        dir_fecha = os.path.join(raiz_base, fecha)
        carpetas_ordenadas = sorted(
            [c for c in os.listdir(dir_fecha) if os.path.isdir(os.path.join(dir_fecha, c))],
            reverse=True,
        )
        for carpeta in carpetas_ordenadas:
            dir_sesion = os.path.join(dir_fecha, carpeta)
            ruta_meta = os.path.join(dir_sesion, NOMBRE_METADATA_FOTOMETRIA)
            if os.path.isfile(ruta_meta):
                sesiones.append({
                    "etiqueta": f"{fecha} - {carpeta}",
                    "ruta_meta": ruta_meta,
                    "carpeta_sesion": dir_sesion,
                    "fecha": fecha,
                    "nombre": carpeta,
                })
    return sesiones


def limpiar_para_json(valor: Any) -> Any:
    """Convierte tipos numpy y estructuras compuestas a tipos nativos serializables."""
    if isinstance(valor, dict):
        return {str(clave): limpiar_para_json(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [limpiar_para_json(item) for item in valor]
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    return valor


def dibujar_circulo_punteado(
    imagen: np.ndarray,
    centro_px: Optional[Tuple[float, float]],
    radio_px: float,
    color: Tuple[int, int, int],
    grosor: int = 2,
) -> None:
    """Dibuja un circulo de trazo punteado: holograma objetivo de la guia AR."""
    if centro_px is None or radio_px <= 1:
        return
    cx = int(round(centro_px[0]))
    cy = int(round(centro_px[1]))
    radio = max(int(round(radio_px)), 2)
    angulo = 0.0
    while angulo < 360.0:
        fin = min(angulo + 16.0, 360.0)
        cv2.ellipse(imagen, (cx, cy), (radio, radio), 0.0, angulo, fin, color, grosor, cv2.LINE_AA)
        angulo += 26.0


def posicion_polo_desde_anclaje(
    ancla: Optional[Dict[str, Any]],
    punto_fallback: Optional[Tuple[float, float]],
    fiduciarios: Optional[Dict[str, Any]],
) -> Optional[Tuple[float, float]]:
    """
    Re-ubica un polo anclado a la piel con v_idx y offsets dx, dy sobre la
    malla facial en vivo. Si no hay malla o ancla, usa la posicion cruda.
    """
    if fiduciarios is not None and ancla:
        v_idx = ancla.get("v_idx")
        if v_idx is not None:
            puntos_2d = fiduciarios.get("puntos_2d_px")
            if puntos_2d is not None and 0 <= int(v_idx) < len(puntos_2d):
                return (
                    float(puntos_2d[int(v_idx), 0] + float(ancla.get("dx", 0.0))),
                    float(puntos_2d[int(v_idx), 1] + float(ancla.get("dy", 0.0))),
                )
    if punto_fallback is not None:
        return (float(punto_fallback[0]), float(punto_fallback[1]))
    return None


# =====================================================================
# Guia por voz WAV-first del asistente lineal
# Frases canonica compartida por la pantalla y el audio; los WAV se
# generan una sola vez con la misma voz Piper del asistente TARS.
# =====================================================================
DIRECTORIO_VOZ_ASISTENTE = pathlib.Path(__file__).resolve().parent / "voz_asistente"
DIRECTORIO_CACHE_VOZ = pathlib.Path("/tmp/voz_asistente_cache")
MODELO_PIPER_RUTA = pathlib.Path(
    "~/.local/share/piper-voices/es_ES-sharvard-medium.onnx"
).expanduser()
PIPER_BINARIO_RUTA = pathlib.Path("~/.local/bin/piper/piper").expanduser()

FRASES_VOZ_ASISTENTE: Dict[str, str] = {
    "paso1": "Paso 1 de 4. Mire de frente a la camara hasta que el semaforo quede verde. "
    "La foto se toma automaticamente",
    "paso2": "Paso 2 de 4. Gire la cabeza 45 grados a la derecha hasta que el semaforo "
    "quede verde. La foto se toma automaticamente",
    "paso3": "Paso 3 de 4. Vuelva al centro y gire la cabeza 45 grados a la izquierda "
    "para la foto de control. La foto se toma automaticamente",
    "paso4": "Paso 4 de 4. Se abre la ventana del comentario de la sesion. "
    "Escribalo si lo desea y confirme para grabar la sesion",
    "activada": "Guia por voz activada",
    "prueba": "Guia por voz activada: si escucha esta frase, el motor funciona correctamente",
    "comentario": "Desea guardar un comentario sobre la sesion? Escribalo y presione "
    "Guardar, o continue sin comentario",
    "canal0": "Canal 0: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "canal1": "Canal 1: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "canal2": "Canal 2: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "canal3": "Canal 3: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "canal4": "Canal 4: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "canal5": "Canal 5: haga clic en el polo positivo y despues en el polo negativo "
    "sobre la foto congelada",
    "sin_rostro": "Ningun rostro detectado: ubiquese frente a la camara",
    "alejese": "Alejese un poco de la camara",
    "acerquese": "Acerquese un poco a la camara",
    "centro": "Gire la cabeza hacia el centro: mire de frente a la camara",
    "casi_frontal": "Casi: alinee la mirada con el centro de la camara",
    "siga_derecha": "Siga girando la cabeza hasta los 45 grados a la derecha",
    "siga_izquierda": "Siga girando la cabeza hasta los 45 grados a la izquierda",
    "regrese": "Regrese un poco el giro: se paso de los 45 grados",
    "casi_perfil": "Casi: sostenga el giro de 45 grados",
    "no_incline": "No incline la cabeza adelante ni atras",
    "enderece": "Enderece la cabeza: esta inclinada hacia uno de los hombros",
    "ajuste_dist": "Ajuste la distancia: quedo muy cerca o muy lejos de la camara",
    "casi_barbilla": "Casi: nivele la barbilla arriba y abajo",
    "casi_enderece": "Casi: enderece la cabeza ligeramente",
    "mantenga": "Mantenga la pose: capturando la foto automaticamente",
}

# Calma de la guia hablada: el consejo solo se dice cuando se sostiene un rato
# en pantalla y con una pausa grande entre comentarios, para no agobiar.
TICKS_VOZ_ESTABLE = 40
PAUSA_SEGUNDOS_VOZ = 8.0


def _rango_absoluto(rango: Tuple[float, float]) -> Tuple[float, float]:
    """Version absoluta de un rango; las bandas simetricas parten de cero."""
    inferior, superior = float(rango[0]), float(rango[1])
    if inferior <= 0.0 <= superior:
        return (0.0, superior)
    if inferior >= 0.0:
        return (inferior, superior)
    return (abs(inferior), abs(superior))


def _texto_sin_acentos(texto: str) -> str:
    """Version ASCII sin tildes de un texto para dibujarlo con cv2.putText."""
    mapa = str.maketrans({
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ü": "U", "Ñ": "N",
    })
    return str(texto or "").translate(mapa)


def resolver_wav_asistente(clave: str) -> Optional[pathlib.Path]:
    """Ruta del WAV pre-generado de una clave de la guia por voz."""
    if not clave:
        return None
    candidata = DIRECTORIO_VOZ_ASISTENTE / f"{clave}.wav"
    if candidata.is_file():
        return candidata
    candidata_cache = DIRECTORIO_CACHE_VOZ / f"clave_{clave}.wav"
    if candidata_cache.is_file():
        return candidata_cache
    return None


class _VozGuia(QtCore.QObject):
    """Guia por voz del asistente lineal basada en archivos WAV.

    Los audios principales se pre-generan una vez en acquisition/voz_asistente
    con la voz Piper de TARS; el asistente solo los reproduce con pw-play o
    paplay. Para textos dinamicos, como los nombres de musculos, sintetiza
    a demanda en el directorio temporal. Si no hay WAV ni Piper, intenta los
    sintetizadores por consola y por ultimo la voz nativa de Qt.
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.reproductor: str = ""
        self.motor: Any = None
        self.metodo: str = "ninguno"
        self.disponible: bool = False
        self._proceso_anterior: Any = None
        self._pyttsx3_listo = threading.Event()
        self._inicializar()
        print(
            f"[FotometriaWebcam] Guia por voz del asistente lineal: "
            f"metodo={self.metodo} disponible={self.disponible}"
        )

    def _inicializar(self) -> None:
        # Reproduce el patron del asistente TARS: WAVs generados con Piper y
        # reproducidos con pw-play o paplay.
        for comando in ("pw-play", "paplay", "aplay"):
            ruta = shutil.which(comando)
            if ruta:
                self.reproductor = ruta
                break
        if self.reproductor and DIRECTORIO_VOZ_ASISTENTE.is_dir() and any(
            DIRECTORIO_VOZ_ASISTENTE.glob("*.wav")
        ):
            self.metodo = "wav"
            self.disponible = True
            return
        if self.reproductor and PIPER_BINARIO_RUTA.is_file() and MODELO_PIPER_RUTA.is_file():
            self.metodo = "piper"
            self.disponible = True
            return
        # Prioridad: sintetizadores por linea de comandos, verificados en este
        # equipo; el motor nativo de Qt queda como ultimo recurso.
        for comando, metodo in (("spd-say", "spd"), ("espeak-ng", "espeak"), ("espeak", "espeak")):
            ruta = shutil.which(comando)
            if ruta:
                self.motor = ruta
                self.metodo = metodo
                self.disponible = True
                return
        try:
            from PySide6.QtTextToSpeech import QTextToSpeech  # type: ignore
            motor = QTextToSpeech(self)
            nombre_engines = [str(e) for e in motor.availableEngines()]
            eng_actual = ""
            try:
                eng_acceso = getattr(motor, "engine", None)
                eng_actual = str(eng_acceso()) if callable(eng_acceso) else str(eng_acceso)
            except Exception:
                eng_actual = ""
            if eng_actual == "mock" and "speechd" in nombre_engines:
                motor.setEngine("speechd")
                eng_actual = "speechd"
            voces = motor.availableVoices()
            if not voces:
                raise RuntimeError("sin voces")
            for voz in voces:
                try:
                    if str(voz.locale().name()).lower().startswith("es"):
                        motor.setVoice(voz)
                        break
                except Exception:
                    continue
            motor.setVolume(1.0)
            motor.setRate(-0.1)
            self.motor = motor
            self.metodo = "qt"
            self.disponible = True
            return
        except Exception:
            self.motor = None
        threading.Thread(target=self._preparar_pyttsx3, daemon=True).start()

    def _preparar_pyttsx3(self) -> None:
        """Prepara pyttsx3 en un hilo aparte para no congelar la interfaz."""
        motor = None
        try:
            import pyttsx3  # type: ignore
            motor = pyttsx3.init()
        except Exception:
            motor = None
        finally:
            self._pyttsx3_listo.set()
        if motor is not None and self.metodo == "ninguno":
            self.motor = motor
            self.metodo = "pyttsx3"
            self.disponible = True

    def decir(self, texto: str, clave: Optional[str] = None) -> None:
        if not texto or os.environ.get("NANDU_MUTE_SPEAKING") == "1":
            return
        try:
            if self.reproductor and clave:
                ruta = resolver_wav_asistente(clave)
                if ruta is not None:
                    self._reproducir(ruta)
                    return
            if self.metodo in ("wav", "piper"):
                ruta = self._generar_wav_piper(texto, clave)
                if ruta is not None:
                    self._reproducir(ruta)
                    return
                # Piper no disponible para este texto: sintetizadores por consola
                for comando, extras in (
                    ("spd-say", ("-l", "es")),
                    ("espeak-ng", ("-v", "es")),
                    ("espeak", ("-v", "es")),
                ):
                    binario = shutil.which(comando)
                    if binario:
                        subprocess.Popen(
                            [binario, *extras, texto],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        return
                return
            if self.metodo == "spd" and self.motor:
                subprocess.Popen(
                    [str(self.motor), "-l", "es", texto],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif self.metodo == "qt" and self.motor is not None:
                self.motor.say(texto)
            elif self.metodo == "pyttsx3" and self.motor is not None and self._pyttsx3_listo.is_set():
                threading.Thread(
                    target=self._hablar_pyttsx3, args=(texto,), daemon=True
                ).start()
            elif self.metodo == "espeak" and self.motor:
                subprocess.Popen(
                    [str(self.motor), "-v", "es", texto],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def _generar_wav_piper(self, texto: str, clave: Optional[str]) -> Optional[pathlib.Path]:
        """Sintetiza un WAV a demanda con la voz Piper de TARS y lo cachea."""
        try:
            if not (PIPER_BINARIO_RUTA.is_file() and MODELO_PIPER_RUTA.is_file()):
                return None
            DIRECTORIO_CACHE_VOZ.mkdir(parents=True, exist_ok=True)
            if clave:
                destino = DIRECTORIO_CACHE_VOZ / f"clave_{clave}.wav"
            else:
                huella = hashlib.sha1(texto.encode("utf-8")).hexdigest()[:10]
                destino = DIRECTORIO_CACHE_VOZ / f"texto_{huella}.wav"
            if destino.is_file() and destino.stat().st_size > 0:
                return destino
            entorno = dict(os.environ)
            entorno.setdefault("LC_ALL", "C.utf8")
            entorno.setdefault("LANG", "C.utf8")
            resultado = subprocess.run(
                [str(PIPER_BINARIO_RUTA), "--model", str(MODELO_PIPER_RUTA),
                 "--output_file", str(destino)],
                input=texto.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
                env=entorno,
            )
            if resultado.returncode == 0 and destino.is_file() and destino.stat().st_size > 0:
                return destino
            if destino.is_file():
                destino.unlink()
        except Exception:
            pass
        return None

    def _reproducir(self, ruta: Optional[pathlib.Path]) -> None:
        try:
            if not self.reproductor or ruta is None or not pathlib.Path(ruta).is_file():
                return
            anterior = self._proceso_anterior
            if anterior is not None and anterior.poll() is None:
                anterior.terminate()
            self._proceso_anterior = subprocess.Popen(
                [self.reproductor, str(ruta)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    def _hablar_pyttsx3(self, texto: str) -> None:
        try:
            self.motor.say(texto)
            self.motor.runAndWait()
        except Exception:
            pass


class _DialogoComentarioSesion(QtWidgets.QDialog):
    """Ventana del Paso 4 que pregunta por el comentario de la sesion y
    permite ajustar el diametro de gel y el IED nominal antes de confirmar."""

    def __init__(self, dialogo_padre: QtWidgets.QWidget) -> None:
        super().__init__(dialogo_padre)
        self._padre = dialogo_padre
        self.setWindowTitle("Paso 4 : Comentario de la Sesion y Parametros")
        self.setModal(True)
        self.setMinimumWidth(500)
        self.setStyleSheet(
            """
            QDialog { background-color: #1a1a1a; color: #e6edf3; }
            QLabel { color: #e6edf3; }
            QLineEdit {
                background-color: #141414; color: #e6edf3;
                border: 1px solid #444444; border-radius: 4px; padding: 6px;
            }
            QDoubleSpinBox {
                background-color: #141414; color: #00ffcc;
                border: 1px solid #444444; border-radius: 4px; padding: 4px; font-weight: bold;
            }
            QPushButton {
                background-color: #2a2a2a; color: #ffffff;
                border: 1px solid #00ffcc; border-radius: 5px;
                padding: 8px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #004d40; }
            """
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        lbl_pregunta = QtWidgets.QLabel("¿Desea guardar un comentario sobre la sesión?")
        lbl_pregunta.setStyleSheet("color: #00ffcc; font-weight: bold; font-size: 15px;")
        lbl_pregunta.setWordWrap(True)
        layout.addWidget(lbl_pregunta)

        self.edit_comentario = QtWidgets.QLineEdit()
        self.edit_comentario.setText(str(self._padre.edit_comentario.text()))
        self.edit_comentario.setPlaceholderText("Escriba aqui el comentario de la sesion")
        layout.addWidget(self.edit_comentario)

        cuadricula = QtWidgets.QGridLayout()
        lbl_gel = QtWidgets.QLabel("Diametro de Gel / Parche:")
        lbl_gel.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_gel = QtWidgets.QDoubleSpinBox()
        self.spin_gel.setRange(1.0, 50.0)
        self.spin_gel.setSingleStep(0.5)
        self.spin_gel.setSuffix(" mm")
        self.spin_gel.setValue(float(self._padre.spin_gel_diam_mm.value()))
        lbl_ied = QtWidgets.QLabel("Distancia Inter-Electrodo Nominal:")
        lbl_ied.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_ied = QtWidgets.QDoubleSpinBox()
        self.spin_ied.setRange(2.0, 80.0)
        self.spin_ied.setSingleStep(0.5)
        self.spin_ied.setSuffix(" mm")
        self.spin_ied.setValue(float(self._padre.spin_ied_nominal_mm.value()))
        cuadricula.addWidget(lbl_gel, 0, 0)
        cuadricula.addWidget(self.spin_gel, 0, 1)
        cuadricula.addWidget(lbl_ied, 1, 0)
        cuadricula.addWidget(self.spin_ied, 1, 1)
        layout.addLayout(cuadricula)

        layout.addStretch(1)

        self.btn_guardar = QtWidgets.QPushButton("Guardar Comentario y Confirmar Sesion")
        self.btn_guardar.setStyleSheet(
            "QPushButton { background-color: #00c853; color: #0a0a0a; border: 2px solid #00ffcc; }"
        )
        self.btn_guardar.clicked.connect(self._al_guardar)
        layout.addWidget(self.btn_guardar)

        self.btn_sin_comentario = QtWidgets.QPushButton("Continuar Sin Comentario")
        self.btn_sin_comentario.clicked.connect(self._al_continuar_sin_comentario)
        layout.addWidget(self.btn_sin_comentario)

        self.btn_volver = QtWidgets.QPushButton("Volver al Asistente")
        self.btn_volver.setStyleSheet("QPushButton { border-color: #757575; color: #bdbdbd; }")
        self.btn_volver.clicked.connect(self.reject)
        layout.addWidget(self.btn_volver)

    def _copiar_valores(self, comentario: str) -> None:
        self._padre.edit_comentario.setText(comentario)
        self._padre.spin_gel_diam_mm.setValue(float(self.spin_gel.value()))
        self._padre.spin_ied_nominal_mm.setValue(float(self.spin_ied.value()))

    def _al_guardar(self) -> None:
        self._copiar_valores(self.edit_comentario.text().strip())
        self.accept()

    def _al_continuar_sin_comentario(self) -> None:
        self._copiar_valores("")
        self.accept()


class AsistenteLinealFotometriaDialog(QtWidgets.QDialog):
    """
    Asistente Lineal Simple de Fotometria: guia foto a foto para registrar la
    colocacion de electrodos sEMG de la sesion y proyectar en vivo, como hologramas
    punteados sobre la piel, la colocacion de una medicion anterior.

    Secuencia lineal:
      - Paso 1 de 4: Foto Frontal + marcado de Canal 0 y Canal 2
      - Paso 2 de 4: Foto Perfil Izquierdo + marcado de Canal 1
      - Paso 3 de 4: Foto de Control Perfil Derecho
      - Paso 4 de 4: Parametros de montaje y comentario de sesion

    Al confirmar entrega las fotos del montaje y metadata_fotometria con
    coordenadas 2D, vertices 3D anclados por v_idx y offsets dx/dy, IED en mm,
    escala mm/px, diametro de gel y comentario.
    """

    PASO_FRONTAL = 0
    PASO_MARCA_FRONTAL = 1
    PASO_PERFIL_IZQ = 2
    PASO_MARCA_PERFIL = 3
    PASO_PERFIL_DER = 4
    PASO_PARAMETROS = 5

    def __init__(
        self,
        capture_thread: Optional[WebcamCaptureThread],
        agent: AntigravityAgent,
        parent: Optional[QtWidgets.QWidget] = None,
        nombres_musculos: Optional[List[str]] = None,
        active_channel_indices: Optional[List[int]] = None,
        sujeto: str = "",
        prueba: str = "",
        sesiones_previas: Optional[List[Dict[str, Any]]] = None,
        d_ip_nominal_mm: float = D_IP_NOMINAL_DEFECTO_MM,
    ) -> None:
        super().__init__(parent)
        self.capture_thread = capture_thread
        self.agent = agent
        self.sujeto = sujeto or "Sujeto"
        self.prueba = prueba or "Experimento"
        self.d_ip_nominal_mm = float(d_ip_nominal_mm)

        if active_channel_indices and len(active_channel_indices) > 0:
            self.active_channel_indices: List[int] = [int(c) for c in active_channel_indices]
        else:
            total_inicial = len(nombres_musculos) if nombres_musculos else 3
            self.active_channel_indices = list(range(max(total_inicial, 1)))

        self.nombres_musculos: List[str] = []
        for pos in range(len(self.active_channel_indices)):
            if nombres_musculos and pos < len(nombres_musculos) and str(nombres_musculos[pos]).strip():
                self.nombres_musculos.append(str(nombres_musculos[pos]).strip())
            else:
                self.nombres_musculos.append(f"Canal {self.active_channel_indices[pos]}")

        # Filtrar canales que sean Micrófono para que solo se marquen electrodos EMG reales
        emg_positions: List[int] = []
        for pos_idx, musc_name in enumerate(self.nombres_musculos):
            m_low = str(musc_name or "").strip().lower()
            hw_c = self.active_channel_indices[pos_idx] if pos_idx < len(self.active_channel_indices) else pos_idx
            if "mic" in m_low or "audio" in m_low or (hw_c == 3 and m_low in ("", "canal 3", "canal_3", "ch3", "ai3")):
                continue
            emg_positions.append(pos_idx)
        if not emg_positions:
            emg_positions = list(range(len(self.active_channel_indices)))
        self._emg_positions: List[int] = emg_positions

        # Por defecto: marcar TODOS los canales EMG en orden correlativo (0, 1, 2...) en la foto frontal
        # para que jamás se salte ni se pierda el canal intermedio (Canal 1). Si el usuario prefiere
        # marcar el Canal 1 en la foto de Perfil Izquierdo, también puede hacerlo en el Paso 2.
        self._canales_frontal: List[int] = list(emg_positions)
        self._canales_perfil: List[int] = []

        # Estado del asistente
        self._paso: int = self.PASO_FRONTAL
        self._pares_por_canal: Dict[int, Dict[str, Any]] = {}
        self._cola_marcado: List[int] = []
        self._canales_marcado_paso: List[int] = []
        self._pend_pos_pole: Optional[Tuple[float, float]] = None
        self._pend_pos_anchor: Optional[Dict[str, Any]] = None
        self._fotos: Dict[str, Optional[np.ndarray]] = {
            "frontal": None,
            "perfil_izquierdo": None,
            "perfil_derecho": None,
        }
        self._pose_fotos: Dict[str, Any] = {}
        self._dip_px_fotos: Dict[str, Optional[float]] = {"frontal": None}
        self._escala_mm_px_frontal: Optional[float] = None
        self._meta_calibracion_frontal: Optional[Dict[str, Any]] = None
        self._frozen_frame: Optional[np.ndarray] = None
        self._frozen_fiduciarios: Optional[Dict[str, Any]] = None
        self._last_live_frame: Optional[np.ndarray] = None
        self._fiduciarios_marco: Optional[Dict[str, Any]] = None
        self._ultimo_pose: Dict[str, Any] = {}
        # Signo del yaw observado al capturar el perfil izquierdo; el perfil
        # derecho exige el signo opuesto para no auto-capturar la pose anterior.
        self._yaw_signo_perfil: Optional[int] = None
        self._frames_verde: int = 0
        self._destello_hasta: float = 0.0
        self._datos_sesion: Optional[Dict[str, Any]] = None

        # Guia AR de repeticion
        self._fantasma_activado: bool = False
        self._montaje_referencia: Optional[Dict[str, Dict[str, Any]]] = None
        self._fantasma_dist_ref_cm: Optional[float] = None
        self._lista_sesiones_previas: List[Dict[str, Any]] = list(sesiones_previas or [])

        # Guia hablada y consejos dinamicos de colocacion
        self._voz = _VozGuia(self)
        self._ultimo_consejo: str = ""
        self._ultima_vez_voz: float = 0.0
        self._canal_anunciado: Optional[int] = None
        self._clave_pendiente: str = ""
        self._frames_clave: int = 0
        self._dialogo_comentario_activo: bool = False

        self.setWindowTitle(f"Asistente Lineal Simple de Fotometria - {self.sujeto} : {self.prueba}")
        self.resize(1240, 780)
        self._aplicar_tema_oscuro()
        self._construir_ui()
        self._ir_a_paso(self.PASO_FRONTAL)

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._update_preview)
        self._timer.start()

    def _aplicar_tema_oscuro(self) -> None:
        self.setStyleSheet(
            """
            QDialog {
                background-color: #1a1a1a;
                color: #e6edf3;
                font-family: 'Segoe UI', 'Ubuntu', sans-serif;
            }
            QGroupBox {
                background-color: #222222;
                border: 1px solid #333333;
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
                background-color: #2a2a2a;
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
            QPushButton:pressed {
                background-color: #00ffcc;
                color: #111111;
            }
            QPushButton:disabled {
                border-color: #444444;
                color: #777777;
            }
            QComboBox {
                background-color: #262626;
                color: #00ffcc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
                font-weight: bold;
            }
            QListWidget {
                background-color: #141414;
                color: #e6edf3;
                border: 1px solid #333333;
                border-radius: 4px;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #141414;
                color: #e6edf3;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 6px;
            }
            QDoubleSpinBox {
                background-color: #141414;
                color: #00ffcc;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
            }
            QScrollArea {
                border: none;
                background-color: #1a1a1a;
            }
            QCheckBox {
                color: #e6edf3;
                font-weight: bold;
                padding: 4px;
            }
            """
        )

    def _construir_ui(self) -> None:
        layout_principal = QtWidgets.QHBoxLayout(self)
        layout_principal.setContentsMargins(12, 12, 12, 12)
        layout_principal.setSpacing(14)

        # --- Columna izquierda: video en vivo + semaforo de alineacion ---
        col_izq = QtWidgets.QVBoxLayout()
        col_izq.setSpacing(8)

        lbl_cabecera = QtWidgets.QLabel(
            "<b>Vista en Vivo con Guia AR</b> - Clic izquierdo marca Polos | Clic derecho deshace"
        )
        lbl_cabecera.setWordWrap(True)
        col_izq.addWidget(lbl_cabecera)

        self.video_label = _ClickableVideoLabel(self)
        self.video_label.setStyleSheet(
            "background-color: #0d0d0d; border: 2px solid #00ffcc; border-radius: 6px;"
        )
        self.video_label.pointClicked.connect(self._al_clic_izquierdo)
        self.video_label.rightClicked.connect(self._al_clic_derecho)
        col_izq.addWidget(self.video_label, stretch=1)

        grupo_semaforo = QtWidgets.QGroupBox("Semáforo de Alineación en Vivo")
        cuadricula = QtWidgets.QGridLayout(grupo_semaforo)
        cuadricula.setSpacing(8)
        self.badge_yaw = self._crear_badge("Yaw: --")
        self.badge_pitch = self._crear_badge("Pitch: --")
        self.badge_roll = self._crear_badge("Roll: --")
        self.badge_dist = self._crear_badge("Dist: -- cm")
        cuadricula.addWidget(self.badge_yaw, 0, 0)
        cuadricula.addWidget(self.badge_pitch, 0, 1)
        cuadricula.addWidget(self.badge_roll, 0, 2)
        cuadricula.addWidget(self.badge_dist, 0, 3)
        col_izq.addWidget(grupo_semaforo)

        self.lbl_consejo = QtWidgets.QLabel("")
        self.lbl_consejo.setWordWrap(True)
        self.lbl_consejo.setStyleSheet(
            "color: #ffd54f; font-weight: bold; font-size: 14px; padding: 2px;"
        )
        col_izq.addWidget(self.lbl_consejo)

        self.lbl_estado_paso = QtWidgets.QLabel("")
        self.lbl_estado_paso.setWordWrap(True)
        self.lbl_estado_paso.setStyleSheet("color: #00ffcc; font-weight: bold; padding: 2px;")
        col_izq.addWidget(self.lbl_estado_paso)

        layout_principal.addLayout(col_izq, stretch=3)

        # --- Columna derecha: pasos, progreso, guia AR, parametros y acciones ---
        panel = QtWidgets.QWidget()
        layout_panel = QtWidgets.QVBoxLayout(panel)
        layout_panel.setContentsMargins(0, 0, 0, 0)
        layout_panel.setSpacing(10)

        grupo_pasos = QtWidgets.QGroupBox("Guia Paso a Paso")
        layout_pasos = QtWidgets.QVBoxLayout(grupo_pasos)
        layout_pasos.setSpacing(6)

        self.lbl_titulo_paso = QtWidgets.QLabel("")
        self.lbl_titulo_paso.setStyleSheet("color: #00ffcc; font-size: 15px; font-weight: bold;")
        self.lbl_titulo_paso.setWordWrap(True)
        layout_pasos.addWidget(self.lbl_titulo_paso)

        self.lbl_instruccion_paso = QtWidgets.QLabel("")
        self.lbl_instruccion_paso.setStyleSheet("color: #e6edf3; font-size: 12px;")
        self.lbl_instruccion_paso.setWordWrap(True)
        layout_pasos.addWidget(self.lbl_instruccion_paso)

        self.lbl_sub_paso = QtWidgets.QLabel("")
        self.lbl_sub_paso.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        self.lbl_sub_paso.setWordWrap(True)
        layout_pasos.addWidget(self.lbl_sub_paso)

        fila_foto = QtWidgets.QHBoxLayout()
        self.btn_tomar_ahora = QtWidgets.QPushButton("Tomar de Todas Formas")
        self.btn_tomar_ahora.clicked.connect(self._al_tomar_de_todas_formas)
        fila_foto.addWidget(self.btn_tomar_ahora)
        self.btn_repetir_foto = QtWidgets.QPushButton("Repetir Foto de este Paso")
        self.btn_repetir_foto.clicked.connect(self._repetir_foto)
        fila_foto.addWidget(self.btn_repetir_foto)
        layout_pasos.addLayout(fila_foto)

        self.chk_marcar_c1_en_perfil = QtWidgets.QCheckBox("Marcar Canal intermedio (Canal 1) en foto de Perfil Izquierdo")
        self.chk_marcar_c1_en_perfil.setChecked(False)
        self.chk_marcar_c1_en_perfil.setToolTip(
            "Desmarcado (por defecto): marca TODOS los canales EMG (0, 1, 2) seguidos en la foto Frontal.\n"
            "Marcado: marca Canal 0 y 2 en Frontal y el Canal 1 en la foto de Perfil Izquierdo."
        )
        self.chk_marcar_c1_en_perfil.toggled.connect(self._al_cambiar_modo_distribucion_canales)
        layout_pasos.addWidget(self.chk_marcar_c1_en_perfil)

        self.btn_continuar_paso = QtWidgets.QPushButton("Continuar al Siguiente Paso")
        self.btn_continuar_paso.setStyleSheet(
            "QPushButton { background-color: #004d40; border-color: #00ffcc; color: #00ffcc; padding: 8px; }"
            "QPushButton:hover { background-color: #00695c; }"
        )
        self.btn_continuar_paso.clicked.connect(self._continuar_paso)
        layout_pasos.addWidget(self.btn_continuar_paso)

        fila_voz = QtWidgets.QHBoxLayout()
        self.chk_voz = QtWidgets.QCheckBox("Guia por voz durante el asistente")
        self.chk_voz.setChecked(self._voz.disponible)
        self.chk_voz.setEnabled(self._voz.disponible)
        if not self._voz.disponible:
            self.chk_voz.setToolTip(
                "Sintesis de voz no disponible en este equipo; la guia seguira solo en pantalla"
            )
        self.chk_voz.toggled.connect(self._al_alternar_voz)
        fila_voz.addWidget(self.chk_voz, stretch=1)
        self.btn_probar_voz = QtWidgets.QPushButton("Probar Voz")
        self.btn_probar_voz.setToolTip("Escucha una frase de prueba del motor de voz")
        self.btn_probar_voz.clicked.connect(self._probar_voz)
        fila_voz.addWidget(self.btn_probar_voz)
        layout_pasos.addLayout(fila_voz)

        layout_panel.addWidget(grupo_pasos)

        grupo_progreso = QtWidgets.QGroupBox("Montaje Registrado")
        layout_progreso = QtWidgets.QVBoxLayout(grupo_progreso)
        self.list_progreso = QtWidgets.QListWidget()
        self.list_progreso.setMaximumHeight(190)
        layout_progreso.addWidget(self.list_progreso)
        layout_panel.addWidget(grupo_progreso)

        # Guia AR de repeticion: superponer la colocacion de una medicion previa
        grupo_fantasma = QtWidgets.QGroupBox("Guia AR de Repeticion - Sesion Anterior")
        layout_fantasma = QtWidgets.QVBoxLayout(grupo_fantasma)
        layout_fantasma.setSpacing(6)

        layout_selector = QtWidgets.QHBoxLayout()
        lbl_selector = QtWidgets.QLabel("Medición de referencia:")
        lbl_selector.setStyleSheet("color: #b0bec5; font-weight: bold;")
        layout_selector.addWidget(lbl_selector)
        self.combo_mediciones_previas = QtWidgets.QComboBox()
        for sesion_previa in self._lista_sesiones_previas:
            self.combo_mediciones_previas.addItem(sesion_previa["etiqueta"])
        layout_selector.addWidget(self.combo_mediciones_previas, stretch=1)
        layout_fantasma.addLayout(layout_selector)

        self.btn_fantasma = QtWidgets.QPushButton(
            "Ver cómo estaban ubicados los electrodos en la sesión anterior"
        )
        self.btn_fantasma.setCheckable(True)
        self.btn_fantasma.setDisabled(len(self._lista_sesiones_previas) == 0)
        if len(self._lista_sesiones_previas) == 0:
            self.btn_fantasma.setToolTip(
                "No hay mediciones previas con metadata_fotometria.json en base_de_datos_electrodos"
            )
        self.btn_fantasma.toggled.connect(self._al_alternar_fantasma)
        layout_fantasma.addWidget(self.btn_fantasma)

        self.lbl_fantasma_estado = QtWidgets.QLabel("Sin referencia previa cargada.")
        self.lbl_fantasma_estado.setWordWrap(True)
        self.lbl_fantasma_estado.setStyleSheet("color: #9e9e9e; font-size: 11px;")
        layout_fantasma.addWidget(self.lbl_fantasma_estado)
        layout_panel.addWidget(grupo_fantasma)

        # Parametros del montaje: paso 4
        self.grupo_parametros = QtWidgets.QGroupBox("Paso 4 : Parametros de Montaje y Comentario")
        layout_parametros = QtWidgets.QVBoxLayout(self.grupo_parametros)
        layout_parametros.setSpacing(6)

        cuadricula_param = QtWidgets.QGridLayout()
        lbl_gel = QtWidgets.QLabel("Diametro de Gel / Parche:")
        lbl_gel.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_gel_diam_mm = QtWidgets.QDoubleSpinBox()
        self.spin_gel_diam_mm.setRange(1.0, 50.0)
        self.spin_gel_diam_mm.setValue(10.0)
        self.spin_gel_diam_mm.setSingleStep(0.5)
        self.spin_gel_diam_mm.setSuffix(" mm")
        lbl_ied = QtWidgets.QLabel("Distancia Inter-Electrodo Nominal Esperada:")
        lbl_ied.setStyleSheet("color: #b0bec5; font-weight: bold;")
        self.spin_ied_nominal_mm = QtWidgets.QDoubleSpinBox()
        self.spin_ied_nominal_mm.setRange(2.0, 80.0)
        self.spin_ied_nominal_mm.setValue(20.0)
        self.spin_ied_nominal_mm.setSingleStep(0.5)
        self.spin_ied_nominal_mm.setSuffix(" mm")
        cuadricula_param.addWidget(lbl_gel, 0, 0)
        cuadricula_param.addWidget(self.spin_gel_diam_mm, 0, 1)
        cuadricula_param.addWidget(lbl_ied, 1, 0)
        cuadricula_param.addWidget(self.spin_ied_nominal_mm, 1, 1)
        layout_parametros.addLayout(cuadricula_param)

        lbl_comentario = QtWidgets.QLabel("¿Desea guardar un comentario sobre la sesión?")
        lbl_comentario.setStyleSheet("color: #ffd54f; font-weight: bold;")
        layout_parametros.addWidget(lbl_comentario)
        self.edit_comentario = QtWidgets.QLineEdit()
        self.edit_comentario.setPlaceholderText("Escriba aqui el comentario de la sesion")
        layout_parametros.addWidget(self.edit_comentario)

        self.btn_abrir_comentario = QtWidgets.QPushButton("Ventana de Comentario y Confirmacion")
        self.btn_abrir_comentario.clicked.connect(self._abrir_dialogo_comentario)
        layout_parametros.addWidget(self.btn_abrir_comentario)

        layout_panel.addWidget(self.grupo_parametros)

        layout_panel.addStretch(1)

        fila_acciones = QtWidgets.QHBoxLayout()
        self.btn_omitir = QtWidgets.QPushButton("Omitir Asistente")
        self.btn_omitir.setStyleSheet("QPushButton { border-color: #757575; color: #bdbdbd; }")
        self.btn_omitir.clicked.connect(self.reject)
        fila_acciones.addWidget(self.btn_omitir)
        self.btn_confirmar = QtWidgets.QPushButton("Confirmar y Grabar Sesión")
        self.btn_confirmar.setStyleSheet(
            "QPushButton { background-color: #00c853; color: #0a0a0a; border: 2px solid #00ffcc; "
            "padding: 10px 16px; font-size: 14px; font-weight: bold; }"
            "QPushButton:hover { background-color: #00e676; }"
        )
        self.btn_confirmar.clicked.connect(self._al_confirmar)
        fila_acciones.addWidget(self.btn_confirmar, stretch=1)
        layout_panel.addLayout(fila_acciones)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        col_der = QtWidgets.QVBoxLayout()
        col_der.addWidget(scroll)
        layout_principal.addLayout(col_der, stretch=2)

    @staticmethod
    def _crear_badge(texto: str) -> QtWidgets.QLabel:
        badge = QtWidgets.QLabel(texto)
        badge.setAlignment(QtCore.Qt.AlignCenter)
        badge.setStyleSheet(
            "background-color: #2b2b2b; color: #ffffff; border: 1px solid #555555; "
            "border-radius: 4px; padding: 6px; font-weight: bold; font-size: 12px;"
        )
        return badge

    def _set_badge_status(self, badge: QtWidgets.QLabel, texto: str, estado: str) -> None:
        if estado == "ok":
            fondo, texto_color, borde = "#003d1f", "#00ff88", "#00e676"
        elif estado == "warn":
            fondo, texto_color, borde = "#4d3800", "#ffd54f", "#ffb300"
        else:
            fondo, texto_color, borde = "#4a1212", "#ff8a80", "#ff5252"
        badge.setText(texto)
        badge.setStyleSheet(
            f"background-color: {fondo}; color: {texto_color}; border: 1px solid {borde}; "
            "border-radius: 4px; padding: 6px; font-weight: bold; font-size: 12px;"
        )

    @staticmethod
    def _mostrar_bgr(bgr_img: np.ndarray, etiqueta: QtWidgets.QLabel) -> None:
        if bgr_img is None or bgr_img.size == 0:
            return
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        alto, ancho, canales = rgb.shape
        qimg = QtGui.QImage(rgb.data, ancho, alto, canales * ancho, QtGui.QImage.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        escalado = pix.scaled(
            etiqueta.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        etiqueta.setPixmap(escalado)

    def _nombre_canal(self, pos_canal: int) -> Tuple[int, str]:
        """Retorna el indice fisico y el nombre del musculo de un canal activo."""
        fisico = (
            self.active_channel_indices[pos_canal]
            if pos_canal < len(self.active_channel_indices)
            else pos_canal
        )
        nombre = (
            self.nombres_musculos[pos_canal]
            if pos_canal < len(self.nombres_musculos)
            else f"Canal {fisico}"
        )
        return fisico, nombre

    # =====================================================================
    # Maquina de pasos lineal
    # =====================================================================

    def _textos_paso_actual(self) -> Dict[str, str]:
        """Textos de guia del paso actual para la interfaz y el marco del video."""
        paso = self._paso
        if paso == self.PASO_FRONTAL:
            canales_txt = ", ".join(
                f"Canal {self._nombre_canal(p)[0]} - {self._nombre_canal(p)[1]}"
                for p in self._canales_frontal
            ) or "sin canales activos"
            return {
                "titulo": "Paso 1 de 4 : Foto Frontal",
                "instruccion": "Mire de frente hasta que el semaforo quede en verde. "
                               "La foto frontal se toma automaticamente.",
                "sub": f"Luego se marcaran los canales: {canales_txt}",
            }
        if paso == self.PASO_MARCA_FRONTAL:
            return {
                "titulo": "Paso 1 de 4 : Foto Frontal - Marcado de Electrodos",
                "instruccion": "",
                "sub": "Clic izquierdo marca los Polos sobre la foto congelada | Clic derecho deshace | Continuar avanza al terminar",
            }
        if paso == self.PASO_PERFIL_IZQ:
            sub_txt = (
                ", ".join(f"Canal {self._nombre_canal(p)[0]} ({self._nombre_canal(p)[1]})" for p in self._canales_perfil)
                if len(self._canales_perfil) > 0
                else "Foto de referencia lateral (+45° hacia la derecha)"
            )
            return {
                "titulo": "Paso 2 de 4 : Foto Perfil Izquierdo (Gire a la DERECHA +45°)",
                "instruccion": "Gire la cabeza 45 grados hacia la DERECHA (Yaw positivo +25° a +65°) hasta poner el semaforo en verde.",
                "sub": sub_txt,
            }
        if paso == self.PASO_MARCA_PERFIL:
            return {
                "titulo": "Paso 2 de 4 : Foto Perfil Izquierdo - Marcado de Electrodos",
                "instruccion": "",
                "sub": "Clic izquierdo marca los Polos sobre la foto congelada | Clic derecho deshace | Continuar avanza al terminar",
            }
        if paso == self.PASO_PERFIL_DER:
            return {
                "titulo": "Paso 3 de 4 : Foto Perfil Derecho (Gire a la IZQUIERDA -45°)",
                "instruccion": "Ahora gire la cabeza 45 grados hacia la IZQUIERDA (Yaw negativo -25° a -65°) hasta poner el semaforo en verde.",
                "sub": "Exige orientacion opuesta al paso anterior (solo se activa al girar a la izquierda)",
            }
        return {
            "titulo": "Paso 4 de 4 : Parametros de Montaje",
            "instruccion": "Revise diametro de gel y distancia inter-electrodo nominal; "
                           "agregue un comentario si lo desea. Confirme para grabar la sesion.",
            "sub": "Al confirmar, el montaje se guarda en la carpeta activa de la medicion",
        }

    def _ir_a_paso(self, nuevo_paso: int) -> None:
        self._paso = nuevo_paso
        self._frames_verde = 0
        # Recordar el lado solo dentro de una secuencia de escaneo: al arrancar
        # (o repetir) el perfil izquierdo se olvida la referencia anterior.
        if nuevo_paso in (self.PASO_FRONTAL, self.PASO_PERFIL_IZQ):
            self._yaw_signo_perfil = None
        es_paso_congelado = nuevo_paso in (self.PASO_MARCA_FRONTAL, self.PASO_MARCA_PERFIL)
        if not es_paso_congelado:
            self._frozen_frame = None
            self._frozen_fiduciarios = None
        if nuevo_paso == self.PASO_MARCA_FRONTAL:
            self._iniciar_marcado(self._canales_frontal)
        elif nuevo_paso == self.PASO_MARCA_PERFIL:
            self._iniciar_marcado(self._canales_perfil)
        self._canal_anunciado = None
        self._clave_pendiente = ""
        self._frames_clave = 0
        self._anunciar_paso(nuevo_paso)
        self._refrescar_estado_ui()
        if nuevo_paso == self.PASO_PARAMETROS:
            QtCore.QTimer.singleShot(400, self._abrir_dialogo_comentario)

    def _iniciar_marcado(self, canales: List[int]) -> None:
        self._canales_marcado_paso = list(canales)
        # Solo encola canales que aun no tengan su par registrado.
        self._cola_marcado = [p for p in canales if p not in self._pares_por_canal]
        self._pend_pos_pole = None
        self._pend_pos_anchor = None

    def _canal_actual_marca(self) -> Optional[int]:
        return self._cola_marcado[0] if self._cola_marcado else None

    def _refrescar_estado_ui(self) -> None:
        textos = self._textos_paso_actual()
        paso_marca = self._paso in (self.PASO_MARCA_FRONTAL, self.PASO_MARCA_PERFIL)
        if paso_marca:
            canal = self._canal_actual_marca()
            if canal is not None:
                fisico, nombre = self._nombre_canal(canal)
                self.lbl_titulo_paso.setText(
                    f"Ubique el Canal {fisico}: Clic en Polo + y Clic en Polo -"
                )
                self.lbl_instruccion_paso.setText(
                    f"Musculo: {nombre} - "
                    + ("Marque ahora el Polo -" if self._pend_pos_pole is not None else
                       "Marque primero el Polo + y luego el Polo -")
                )
                self._mostrar_consejo(
                    f"Marque el Canal {fisico} - {nombre}: "
                    "clic en Polo + y luego en Polo - sobre la foto congelada",
                    "info",
                )
                if canal != self._canal_anunciado:
                    self._canal_anunciado = canal
                    texto_canal = (
                        f"Marque el Canal {fisico}: {nombre}. Haga clic en el Polo + "
                        "y despues en el Polo - sobre la foto congelada"
                    )
                    self._decir(texto_canal, clave=f"canal{fisico}")
            else:
                self.lbl_titulo_paso.setText("Todos los canales de esta foto estan marcados")
                self.lbl_instruccion_paso.setText(
                    "Clic derecho para deshacer un par; Continuar para avanzar al siguiente paso"
                )
                self._mostrar_consejo(
                    "Foto completa: revise los pares y pulse Continuar al Siguiente Paso", "ok"
                )
        else:
            self.lbl_titulo_paso.setText(textos["titulo"])
            self.lbl_instruccion_paso.setText(textos["instruccion"])
        self.lbl_sub_paso.setText(textos["sub"])

        es_paso_espera = self._paso in (self.PASO_FRONTAL, self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER)
        self.btn_tomar_ahora.setVisible(es_paso_espera)
        self.btn_tomar_ahora.setEnabled(es_paso_espera)
        self.btn_repetir_foto.setVisible(paso_marca)
        self.btn_repetir_foto.setEnabled(paso_marca)
        continuar_visible = paso_marca and not self._cola_marcado
        self.btn_continuar_paso.setVisible(continuar_visible)
        self.btn_continuar_paso.setEnabled(continuar_visible)

        # El grupo queda siempre activo para cargar el comentario antes del Paso 4
        self.grupo_parametros.setEnabled(True)
        self.btn_confirmar.setEnabled(self._paso == self.PASO_PARAMETROS)

        self._actualizar_lista_progreso()
        self._actualizar_estado_inferior()

    def _actualizar_estado_inferior(self) -> None:
        """Texto de estado bajo el semaforo segun el paso activo."""
        total_pares = len(self._pares_por_canal)
        total_canales = len(self.active_channel_indices)
        textos = self._textos_paso_actual()
        self.lbl_estado_paso.setText(
            f"{textos['titulo']} | Pares marcados: {total_pares} de {total_canales}"
        )

    def _actualizar_lista_progreso(self) -> None:
        self.list_progreso.clear()
        for clave, nombre_foto in (
            ("frontal", "01_frontal.png"),
            ("perfil_izquierdo", "02_perfil_izquierdo.png"),
            ("perfil_derecho", "03_perfil_derecho.png"),
        ):
            toma = self._fotos.get(clave)
            estado_foto = "capturada" if toma is not None else "pendiente"
            self.list_progreso.addItem(
                f"[{'OK' if toma is not None else '  '}] {nombre_foto} - {estado_foto}"
            )
        for pos in range(len(self.active_channel_indices)):
            fisico, nombre = self._nombre_canal(pos)
            par = self._pares_por_canal.get(pos)
            en_cola = pos in self._cola_marcado and self._canal_actual_marca() == pos
            if par is not None:
                ied_txt = (
                    f"{par['ied_mm']:.1f} mm" if par.get("ied_mm") is not None
                    else f"{par['ied_px']:.0f} px"
                )
                self.list_progreso.addItem(
                    f"[OK] Canal {fisico} - {nombre} - Polo + y Polo - marcados | IED: {ied_txt}"
                )
            elif en_cola:
                if self._pend_pos_pole is not None:
                    self.list_progreso.addItem(
                        f"[ >] Canal {fisico} - {nombre} - Polo + marcado, marque Polo -"
                    )
                else:
                    self.list_progreso.addItem(
                        f"[ >] Canal {fisico} - {nombre} - Marque Polo +"
                    )
            else:
                self.list_progreso.addItem(
                    f"[  ] Canal {fisico} - {nombre} - Pendiente"
                )

    # =====================================================================
    # Marcado de polos sobre la foto congelada
    # =====================================================================

    def _registrar_anclaje(
        self, fiduciarios: Optional[Dict[str, Any]], img_x: float, img_y: float
    ) -> Optional[Dict[str, Any]]:
        """Ancla un clic al vertice de malla mas cercano: v_idx, offsets dx dy y datos 3D."""
        if fiduciarios is None or "puntos_2d_px" not in fiduciarios:
            return None
        puntos_2d = fiduciarios["puntos_2d_px"]
        distancias = np.hypot(puntos_2d[:, 0] - img_x, puntos_2d[:, 1] - img_y)
        v_idx = int(np.argmin(distancias))
        ancla = {
            "v_idx": v_idx,
            "dx": float(img_x - puntos_2d[v_idx, 0]),
            "dy": float(img_y - puntos_2d[v_idx, 1]),
            "dip_px": float(fiduciarios.get("distancia_interpupilar_px", 0.0)),
        }
        puntos_3d = fiduciarios.get("puntos_3d")
        if puntos_3d is not None and v_idx < len(puntos_3d):
            ancla["vertice_3d"] = [float(v) for v in puntos_3d[v_idx]]
        return ancla

    def _al_clic_izquierdo(self, img_x: float, img_y: float) -> None:
        if self._paso not in (self.PASO_MARCA_FRONTAL, self.PASO_MARCA_PERFIL):
            return
        canal_pos = self._canal_actual_marca()
        if canal_pos is None:
            return
        ancla = self._registrar_anclaje(self._frozen_fiduciarios, img_x, img_y)
        if self._pend_pos_pole is None:
            self._pend_pos_pole = (float(img_x), float(img_y))
            self._pend_pos_anchor = ancla
            self._refrescar_estado_ui()
        else:
            pos_pt = self._pend_pos_pole
            pos_anc = self._pend_pos_anchor or {}
            self._completar_par(canal_pos, pos_pt, pos_anc, (float(img_x), float(img_y)), ancla)
            self._cola_marcado.pop(0)
            self._pend_pos_pole = None
            self._pend_pos_anchor = None
            # Sin salto automatico: si quedan canales se pide el siguiente;
            # si esta foto ya esta completa, el usuario revisa y pulsa Continuar.
            self._refrescar_estado_ui()
        self._actualizar_lista_progreso()

    def _al_clic_derecho(self, _img_x: float, _img_y: float) -> None:
        if self._paso not in (self.PASO_MARCA_FRONTAL, self.PASO_MARCA_PERFIL):
            return
        if self._pend_pos_pole is not None:
            self._pend_pos_pole = None
            self._pend_pos_anchor = None
        else:
            # Deshacer: retomar el ultimo par ya completado entre los canales
            # que corresponden a la foto del paso actual.
            completados_paso = [
                pos for pos in self._canales_marcado_paso if pos in self._pares_por_canal
            ]
            if completados_paso:
                objetivo = max(completados_paso)
                self._pares_por_canal.pop(objetivo)
                self._pend_pos_pole = None
                self._pend_pos_anchor = None
                if objetivo not in self._cola_marcado:
                    self._cola_marcado.insert(0, objetivo)
        self._refrescar_estado_ui()
        self._actualizar_lista_progreso()

    def _completar_par(
        self,
        canal_pos: int,
        pos_pt: Tuple[float, float],
        pos_anc: Dict[str, Any],
        neg_pt: Tuple[float, float],
        neg_anc: Optional[Dict[str, Any]],
    ) -> None:
        """Registra un par bipolar completo con anclajes, IED y angulo."""
        fisico, nombre = self._nombre_canal(canal_pos)
        ied_px = float(np.hypot(neg_pt[0] - pos_pt[0], neg_pt[1] - pos_pt[1]))
        dip_px = self._dip_px_del_paso()
        escala = (self.d_ip_nominal_mm / dip_px) if dip_px and dip_px > 1.0 else None
        ied_mm = ied_px * escala if escala is not None else None
        angulo = float(np.degrees(np.arctan2(neg_pt[1] - pos_pt[1], neg_pt[0] - pos_pt[0])))
        par = {
            "es_bipolar": True,
            "canal_pos": canal_pos,
            "canal_fisico": fisico,
            "nombre": nombre,
            "polo_positivo": (float(pos_pt[0]), float(pos_pt[1])),
            "polo_negativo": (float(neg_pt[0]), float(neg_pt[1])),
            "anchor_positivo": pos_anc or {},
            "anchor_negativo": neg_anc or {},
            "ied_px": ied_px,
            "ied_mm": ied_mm,
            "angulo_deg": angulo,
            "dip_px_foto": float(dip_px) if dip_px else 0.0,
            "escala_mm_px": escala,
            "diametro_gel_mm": float(self.spin_gel_diam_mm.value()),
            "distancia_interelectrodo_nominal_mm": float(self.spin_ied_nominal_mm.value()),
        }
        self._pares_por_canal[canal_pos] = par

    def _dip_px_del_paso(self) -> Optional[float]:
        fiduciarios = self._frozen_fiduciarios if self._frozen_frame is not None else self._fiduciarios_marco
        if fiduciarios is not None:
            dip = float(fiduciarios.get("distancia_interpupilar_px", 0.0))
            if dip > 1.0:
                return dip
        return None

    def _continuar_paso(self) -> None:
        """Avanza al paso despues de revisar la foto marcada de este paso."""
        if self._cola_marcado:
            return
        if self._paso == self.PASO_MARCA_FRONTAL:
            self._ir_a_paso(self.PASO_PERFIL_IZQ)
        elif self._paso == self.PASO_MARCA_PERFIL:
            self._ir_a_paso(self.PASO_PERFIL_DER)

    # =====================================================================
    # Guia hablada y consejos dinamicos de colocacion
    # =====================================================================

    def _voz_habilitada(self) -> bool:
        """Indica si la sintesis de voz esta activada en este momento."""
        voz = getattr(self, "_voz", None)
        casillero = getattr(self, "chk_voz", None)
        return bool(
            voz is not None
            and voz.disponible
            and casillero is not None
            and casillero.isChecked()
        )

    def _decir(self, texto: str, clave: Optional[str] = None) -> None:
        """Habla una instruccion: prefiere el WAV de la clave pre-generado."""
        if texto and self._voz_habilitada():
            self._voz.decir(texto, clave)

    def _al_alternar_voz(self, activado: bool) -> None:
        if activado:
            self._decir(FRASES_VOZ_ASISTENTE["activada"], clave="activada")

    def _probar_voz(self) -> None:
        """Prueba audible del motor de voz elegido; reporta en la consola."""
        if not self._voz.disponible:
            self._mostrar_consejo(
                "Sintesis de voz no disponible en este equipo: instale espeak-ng o speech-dispatcher",
                "rojo",
            )
            QtWidgets.QMessageBox.information(
                self,
                "Guia por Voz No Disponible",
                "No hay motor de voz en este equipo.\n"
                "Instale el paquete espeak-ng o el servicio speech-dispatcher "
                "y vuelva a abrir el asistente.",
            )
            return
        print(
            f"[FotometriaWebcam] Prueba de voz del asistente lineal: metodo={self._voz.metodo}"
        )
        self._decir(FRASES_VOZ_ASISTENTE["prueba"], clave="prueba")

    def _abrir_dialogo_comentario(self) -> None:
        """Abre la ventana del Paso 4 que pregunta el comentario de la sesion."""
        if not self.isVisible() or self._dialogo_comentario_activo:
            return
        self._dialogo_comentario_activo = True
        self._decir(FRASES_VOZ_ASISTENTE["comentario"], clave="comentario")
        self._ultima_vez_voz = time.perf_counter()
        dialogo = _DialogoComentarioSesion(self)
        resultado = dialogo.exec()
        self._dialogo_comentario_activo = False
        if resultado == QtWidgets.QDialog.Accepted:
            self._al_confirmar()

    def _mostrar_consejo(self, texto: str, estado: str = "info") -> None:
        """Panel de consejo grande debajo del semaforo."""
        color = {"ok": "#00ff88", "info": "#ffd54f", "rojo": "#ff8a80"}.get(estado, "#ffd54f")
        self.lbl_consejo.setText(texto)
        self.lbl_consejo.setStyleSheet(
            f"color: {color}; font-weight: bold; font-size: 14px; padding: 2px;"
        )

    def _anunciar_paso(self, paso: int) -> None:
        """Anuncia por voz la instruccion principal de un paso de captura.
        Los pasos de marcado los anuncia el cambio de canal."""
        claves = {
            self.PASO_FRONTAL: "paso1",
            self.PASO_PERFIL_IZQ: "paso2",
            self.PASO_PERFIL_DER: "paso3",
            self.PASO_PARAMETROS: "paso4",
        }
        clave = claves.get(paso)
        if clave:
            self._decir(FRASES_VOZ_ASISTENTE[clave], clave=clave)
            self._ultima_vez_voz = time.perf_counter()

    def _al_cambiar_modo_distribucion_canales(self, en_perfil: bool) -> None:
        """Permite elegir si el Canal 1 se marca en Frontal junto a 0 y 2, o en Perfil Izquierdo."""
        emg_pos = list(getattr(self, "_emg_positions", range(len(self.active_channel_indices))))
        if en_perfil and len(emg_pos) >= 2:
            c_mid = emg_pos[1]
            self._canales_frontal = [p for p in emg_pos if p != c_mid]
            self._canales_perfil = [c_mid]
        else:
            self._canales_frontal = list(emg_pos)
            self._canales_perfil = []
        if self._paso == self.PASO_MARCA_FRONTAL:
            self._iniciar_marcado(self._canales_frontal)
        elif self._paso == self.PASO_MARCA_PERFIL:
            self._iniciar_marcado(self._canales_perfil)
        self._refrescar_estado_ui()

    def _sugerencia_pose(self, pose: Dict[str, Any]) -> str:
        """Clave del consejo comparando la pose medida con la objetivo del paso (con signo direccional estricto)."""
        paso = self._paso
        if paso == self.PASO_PARAMETROS:
            return "paso4"
        bandas = self._bandas_objetivo()
        yaw = float(pose.get("yaw_deg", 0.0))
        pitch = float(pose.get("pitch_deg", 0.0))
        roll = float(pose.get("roll_deg", 0.0))
        dist = float(pose.get("distancia_camara_cm", 0.0))
        dist_medible = pose.get("metodo_distancia", "estimada") not in (None, "no_disponible")

        if dist_medible and dist < bandas["dist_warn"][0]:
            return "alejese"
        if dist_medible and dist > bandas["dist_warn"][1]:
            return "acerquese"
        if paso == self.PASO_FRONTAL:
            if abs(yaw) > bandas["yaw_warn"][1]:
                return "centro"
            if abs(yaw) > bandas["yaw"][1]:
                return "casi_frontal"
        elif paso == self.PASO_PERFIL_IZQ:
            # Perfil Izquierdo exige girar cabeza hacia la derecha (Yaw positivo +25° a +65°)
            if yaw < 22.0:
                return "siga_derecha"
            if yaw > 70.0:
                return "regrese"
            if not (25.0 <= yaw <= 65.0):
                return "casi_perfil"
        elif paso == self.PASO_PERFIL_DER:
            # Perfil Derecho exige girar cabeza hacia la izquierda (Yaw negativo -65° a -25°)
            if yaw > -22.0:
                return "siga_izquierda"
            if yaw < -70.0:
                return "regrese"
            if not (-65.0 <= yaw <= -25.0):
                return "casi_perfil"
        if not self._en_rango(pitch, bandas["pitch_warn"]):
            return "no_incline"
        if not self._en_rango(roll, bandas["roll_warn"]):
            return "enderece"
        if dist_medible and not self._en_rango(dist, bandas["dist"]):
            return "ajuste_dist"
        if not self._en_rango(pitch, bandas["pitch"]):
            return "casi_barbilla"
        if not self._en_rango(roll, bandas["roll"]):
            return "casi_enderece"
        return "mantenga" 

    def _guiar_en_vivo(self, pose: Dict[str, Any], verde: bool, fiduciarios: Any) -> None:
        """Consejo en pantalla en vivo; la voz solo cuando el mismo consejo se
        sostiene un rato y con una pausa larga entre comentarios."""
        if fiduciarios is None or not pose:
            clave = "sin_rostro"
            estado = "rojo"
        elif verde:
            # Sincronizado con el semaforo: nunca pide sostener pose si no captura
            clave = "mantenga"
            estado = "ok"
        else:
            clave = self._sugerencia_pose(pose)
            estado = "info"
        texto = FRASES_VOZ_ASISTENTE.get(clave, clave)
        self._mostrar_consejo(texto, estado)
        # El consejo debe mantenerse estable un rato antes de llegar a la voz
        if clave != self._clave_pendiente:
            self._clave_pendiente = clave
            self._frames_clave = 0
        else:
            self._frames_clave += 1
        ahora = time.perf_counter()
        if (
            clave != "paso4"
            and texto != self._ultimo_consejo
            and self._frames_clave >= TICKS_VOZ_ESTABLE
            and (ahora - self._ultima_vez_voz) >= PAUSA_SEGUNDOS_VOZ
        ):
            self._decir(texto, clave=clave)
            self._ultima_vez_voz = ahora
            self._ultimo_consejo = texto

    def _repetir_foto(self) -> None:
        """Vuelve al paso de captura actual para repetir la foto."""
        if self._paso == self.PASO_MARCA_FRONTAL:
            for pos_canal in self._canales_frontal:
                self._pares_por_canal.pop(pos_canal, None)
            self._ir_a_paso(self.PASO_FRONTAL)
        elif self._paso == self.PASO_MARCA_PERFIL:
            for pos_canal in self._canales_perfil:
                self._pares_por_canal.pop(pos_canal, None)
            self._ir_a_paso(self.PASO_PERFIL_IZQ)

    def _al_tomar_de_todas_formas(self) -> None:
        if self._paso in (self.PASO_FRONTAL, self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER):
            self._capturar_paso()

    # =====================================================================
    # Alineacion, semaforo y captura de fotos guiadas
    # =====================================================================

    def _bandas_objetivo(self) -> Dict[str, Any]:
        """Rangos direccionales con signo del semaforo segun la pose objetivo del paso actual:
        - PASO_FRONTAL    : [-8°, +8°]
        - PASO_PERFIL_IZQ : [+25°, +65°] (giro hacia la derecha)
        - PASO_PERFIL_DER : [-65°, -25°] (giro hacia la izquierda)
        """
        paso = self._paso
        if paso == self.PASO_PERFIL_IZQ:
            yaw_rango, yaw_rango_warn = (25.0, 65.0), (18.0, 72.0)
        elif paso == self.PASO_PERFIL_DER:
            yaw_rango, yaw_rango_warn = (-65.0, -25.0), (-72.0, -18.0)
        else:
            yaw_rango, yaw_rango_warn = (-8.0, 8.0), (-14.0, 14.0)
        return {
            "yaw": yaw_rango,
            "yaw_warn": yaw_rango_warn,
            "yaw_absoluto": (25.0, 65.0) if paso in (self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER) else (0.0, 8.0),
            "yaw_absoluto_warn": (18.0, 72.0) if paso in (self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER) else (0.0, 14.0),
            "pitch": (-10.0, 10.0),
            "pitch_warn": (-16.0, 16.0),
            "roll": (-6.0, 6.0),
            "roll_warn": (-11.0, 11.0),
            "dist": (30.0, 70.0),
            "dist_warn": (24.0, 82.0),
        }

    def _en_rango(self, valor: float, rango: Tuple[float, float]) -> bool:
        return rango[0] <= valor <= rango[1]

    def _yaw_ok(self, yaw: float, estricto: bool) -> bool:
        """Compara el yaw CON SIGNO con la banda direccional del paso actual (no admite girar al lado contrario)."""
        bandas = self._bandas_objetivo()
        clave = "yaw" if estricto else "yaw_warn"
        return self._en_rango(float(yaw), bandas[clave])

    def _alineacion_verde(self, pose: Dict[str, Any]) -> bool:
        bandas = self._bandas_objetivo()
        if not (
            self._yaw_ok(pose.get("yaw_deg", 0.0), True)
            and self._en_rango(pose.get("pitch_deg", 0.0), bandas["pitch"])
            and self._en_rango(pose.get("roll_deg", 0.0), bandas["roll"])
            and self._en_rango(pose.get("distancia_camara_cm", 0.0), bandas["dist"])
        ):
            return False
        # --- FIX auto-captura de perfil derecho ---
        # Para los perfiles el yaw se compara en valor absoluto (el signo depende
        # de la convencion de la malla, no de la anatomia), pero eso hacia que al
        # pasar del perfil izquierdo al derecho la pose anterior siguiera "verde"
        # y la foto se tomara antes de que la persona girara. Solucion: registrar
        # el signo del yaw del perfil izquierdo ya capturado y exigir el signo
        # OPUESTO en el perfil derecho. Es agnostico a la convencion de la malla.
        if self._paso == self.PASO_PERFIL_DER and self._yaw_signo_perfil is not None:
            yaw = float(pose.get("yaw_deg", 0.0) or 0.0)
            if yaw == 0.0 or (1 if yaw > 0.0 else -1) == self._yaw_signo_perfil:
                return False
        return True

    def _actualizar_badges(self, pose: Dict[str, Any]) -> None:
        bandas = self._bandas_objetivo()
        yaw = pose.get("yaw_deg", 0.0)
        pitch = pose.get("pitch_deg", 0.0)
        roll = pose.get("roll_deg", 0.0)
        dist = pose.get("distancia_camara_cm", 0.0)
        if self._yaw_ok(yaw, True):
            estado_yaw = "ok"
        elif self._yaw_ok(yaw, False):
            estado_yaw = "warn"
        else:
            estado_yaw = "bad"
        estado_pitch = "ok" if self._en_rango(pitch, bandas["pitch"]) else (
            "warn" if self._en_rango(pitch, bandas["pitch_warn"]) else "bad"
        )
        estado_roll = "ok" if self._en_rango(roll, bandas["roll"]) else (
            "warn" if self._en_rango(roll, bandas["roll_warn"]) else "bad"
        )
        estado_dist = "ok" if self._en_rango(dist, bandas["dist"]) else (
            "warn" if self._en_rango(dist, bandas["dist_warn"]) else "bad"
        )
        self._set_badge_status(self.badge_yaw, f"Yaw: {yaw:+.1f}°", estado_yaw)
        self._set_badge_status(self.badge_pitch, f"Pitch: {pitch:+.1f}°", estado_pitch)
        self._set_badge_status(self.badge_roll, f"Roll: {roll:+.1f}°", estado_roll)
        self._set_badge_status(self.badge_dist, f"Dist: {dist:.1f} cm", estado_dist)

    def _get_frame_origen(self) -> Optional[np.ndarray]:
        if self._frozen_frame is not None:
            return self._frozen_frame
        if self.capture_thread is not None:
            frame, _ = self.capture_thread.get_latest_frame()
            if frame is not None:
                self._last_live_frame = frame
                return frame
        return self._last_live_frame

    def _update_preview(self) -> None:
        congelado = self._frozen_frame is not None
        frame = self._get_frame_origen()
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            hilo = self.capture_thread
            if hilo is None:
                texto_espera = "Cierre y abra el asistente: sin hilo de captura"
            elif not hilo.is_camera_open:
                texto_espera = f"Camara {hilo.camera_index} no responde"
            else:
                texto_espera = f"Camara {hilo.camera_index}: esperando fotograma"
            cv2.putText(
                placeholder, texto_espera, (40, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (204, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                placeholder, "Revise la tecla de privacidad de la camara", (40, 255),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (204, 255, 0), 2, cv2.LINE_AA,
            )
            cv2.putText(
                placeholder, "y cierre otros programas que la usen", (40, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (204, 255, 0), 2, cv2.LINE_AA,
            )
            self._mostrar_bgr(placeholder, self.video_label)
            return

        alto, ancho = frame.shape[:2]
        self.video_label.set_image_dimensions(ancho, alto)

        if congelado:
            fiduciarios = self._frozen_fiduciarios
        elif self.agent is not None:
            fiduciarios = self.agent.extraer_malla_facial_3d(frame)
        else:
            fiduciarios = None
        self._fiduciarios_marco = fiduciarios

        canvas = frame.copy()
        pose = None
        verde = False
        if fiduciarios is not None and self.agent is not None:
            pose = self.agent.calcular_pose_y_biometria_facial(
                fiduciarios, (alto, ancho), d_ip_real_cm=self.d_ip_nominal_mm / 10.0
            )
            self._ultimo_pose = pose
            verde = self._alineacion_verde(pose)
            self._actualizar_badges(pose)
        else:
            self._set_badge_status(self.badge_yaw, "Yaw: Sin rostro", "bad")
            self._set_badge_status(self.badge_pitch, "Pitch: Sin rostro", "bad")
            self._set_badge_status(self.badge_roll, "Roll: Sin rostro", "bad")
            self._set_badge_status(self.badge_dist, "Dist: --", "bad")

        self._dibujar_marco(canvas, verde, congelado)
        self._dibujar_marcas(canvas, fiduciarios, congelado)

        if not congelado:
            es_paso_espera = self._paso in (
                self.PASO_FRONTAL, self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER
            )
            if verde and es_paso_espera and fiduciarios is not None:
                self._frames_verde += 1
                if self._frames_verde >= TICKS_VERDE_CAPTURA:
                    self._capturar_paso()
                    return
            else:
                self._frames_verde = 0
            self._guiar_en_vivo(pose, verde and es_paso_espera, fiduciarios)
            if self._fantasma_activado:
                self._dibujar_fantasma(canvas, fiduciarios, pose or {})

        self._aplicar_destello(canvas)
        self._mostrar_bgr(canvas, self.video_label)

    def _dibujar_marco(self, canvas: np.ndarray, verde: bool, congelado: bool) -> None:
        """Elipse guia, semaforo circular y banner con el paso actual."""
        alto, ancho = canvas.shape[:2]
        cx, cy = ancho // 2, alto // 2
        rx, ry = int(ancho * 0.21), int(alto * 0.34)
        color_guia = (118, 230, 0) if verde else (0, 215, 255)
        if not congelado:
            cv2.ellipse(canvas, (cx, cy), (rx, ry), 0, 0, 360, color_guia, 2, cv2.LINE_AA)

        es_paso_espera = self._paso in (self.PASO_FRONTAL, self.PASO_PERFIL_IZQ, self.PASO_PERFIL_DER)
        if es_paso_espera and not congelado:
            centro_semaforo = (ancho - 46, 26)
            color_semaforo = (90, 225, 90) if verde else (60, 60, 230)
            cv2.circle(canvas, centro_semaforo, 13, color_semaforo, -1, cv2.LINE_AA)
            cv2.circle(canvas, centro_semaforo, 13, (255, 255, 255), 1, cv2.LINE_AA)

        textos = self._textos_paso_actual()
        canal = self._canal_actual_marca()
        if canal is not None and congelado:
            fisico, _nombre = self._nombre_canal(canal)
            estado_banner = f"Ubique el Canal {fisico}: Clic en Polo + y Clic en Polo -"
        elif congelado:
            estado_banner = "Marcado de electrodos sobre la foto congelada"
        elif verde and es_paso_espera:
            restante_s = max(TICKS_VERDE_CAPTURA - self._frames_verde, 0) * 0.05
            estado_banner = f"Semaforo verde: mantenga la pose, auto captura en {restante_s:0.1f} s"
        elif es_paso_espera:
            estado_banner = "Alinee el rostro hasta que el semaforo quede en verde"
        else:
            estado_banner = textos["titulo"]
        fondo = (20, 75, 20) if verde else (20, 45, 70)
        if congelado:
            fondo = (70, 55, 20)
        cv2.rectangle(canvas, (10, 8), (ancho - 10, 42), fondo, -1)
        cv2.rectangle(canvas, (10, 8), (ancho - 10, 42), color_guia, 1)
        cv2.putText(
            canvas, estado_banner, (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        fila_pie = alto - 12
        if self._frozen_frame is not None:
            cv2.putText(
                canvas, "Foto congelada para marcado - Clic izquierdo: Polo | Clic derecho: deshacer",
                (12, fila_pie), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA,
            )

    def _dibujar_marcas(
        self,
        canvas: np.ndarray,
        fiduciarios: Optional[Dict[str, Any]],
        congelado: bool,
    ) -> None:
        """Dibuja los pares ya marcados y el polo pendiente de completar."""
        dip_px = self._dip_px_del_paso()
        escala = self.d_ip_nominal_mm / dip_px if dip_px and dip_px > 1.0 else None
        canales_visibles = (
            set(self._canales_marcado_paso)
            if (congelado and self._paso in (self.PASO_MARCA_FRONTAL, self.PASO_MARCA_PERFIL) and self._canales_marcado_paso)
            else set(self._pares_por_canal.keys())
        )
        for pos_canal, par in sorted(self._pares_por_canal.items()):
            if pos_canal not in canales_visibles:
                continue
            color = PALETA_CANALES_MONTAJE[pos_canal % len(PALETA_CANALES_MONTAJE)]
            p_pos = posicion_polo_desde_anclaje(
                par.get("anchor_positivo"), par.get("polo_positivo"), fiduciarios
            )
            p_neg = posicion_polo_desde_anclaje(
                par.get("anchor_negativo"), par.get("polo_negativo"), fiduciarios
            )
            if p_pos is None or p_neg is None:
                continue
            pos_x, pos_y = int(round(p_pos[0])), int(round(p_pos[1]))
            neg_x, neg_y = int(round(p_neg[0])), int(round(p_neg[1]))
            cv2.arrowedLine(canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25)
            radio = (
                max(int(round((par.get("diametro_gel_mm", 10.0) / 2.0) / escala)), 4)
                if escala is not None else 12
            )
            cv2.circle(canvas, (pos_x, pos_y), radio, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (neg_x, neg_y), radio, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, (pos_x, pos_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (pos_x, pos_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "+", (pos_x - 4, pos_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(canvas, (neg_x, neg_y), 7, (255, 120, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (neg_x, neg_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "-", (neg_x - 3, neg_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 255, 255), 1, cv2.LINE_AA)
            etiqueta = f"Canal {par['canal_fisico']} - {par['nombre']}"
            mid_x, mid_y = (pos_x + neg_x) // 2, (pos_y + neg_y) // 2
            cv2.putText(canvas, etiqueta, (mid_x + 10, mid_y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, color, 2, cv2.LINE_AA)

        if congelado and self._pend_pos_pole is not None:
            pend_x, pend_y = int(round(self._pend_pos_pole[0])), int(round(self._pend_pos_pole[1]))
            radio = (
                max(int(round((self.spin_gel_diam_mm.value() / 2.0) / escala)), 4)
                if escala is not None else 12
            )
            cv2.circle(canvas, (pend_x, pend_y), radio, COLOR_FANTASMA_LINEA, 1, cv2.LINE_AA)
            cv2.circle(canvas, (pend_x, pend_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (pend_x, pend_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "+ [Marque Polo -]", (pend_x + 10, pend_y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    def _aplicar_destello(self, canvas: np.ndarray) -> None:
        if time.perf_counter() < self._destello_hasta:
            blanco = np.full_like(canvas, 255)
            cv2.addWeighted(blanco, 0.35, canvas, 0.65, 0.0, dst=canvas)

    def _capturar_paso(self) -> None:
        """Toma la foto del paso actual, congela el cuadro y avanza la secuencia."""
        frame = self._last_live_frame
        if frame is None:
            return
        fiduciarios = None
        if self.agent is not None:
            fiduciarios = self.agent.extraer_malla_facial_3d(frame)
        pose = {"rostro_detectado": False}
        if self.agent is not None and fiduciarios is not None:
            pose = self.agent.calcular_pose_y_biometria_facial(
                fiduciarios, frame.shape[:2], d_ip_real_cm=self.d_ip_nominal_mm / 10.0
            )
        pose["rostro_detectado"] = bool(fiduciarios)

        if self._paso == self.PASO_FRONTAL:
            self._fotos["frontal"] = frame.copy()
            self._pose_fotos["frontal"] = limpiar_para_json(pose)
            dip = float(fiduciarios["distancia_interpupilar_px"]) if fiduciarios else None
            self._dip_px_fotos["frontal"] = dip
            self._escala_mm_px_frontal = (
                self.d_ip_nominal_mm / dip if dip and dip > 1.0 else None
            )
            destino = (
                self.PASO_MARCA_FRONTAL
                if len(self._canales_frontal) > 0
                else self.PASO_PERFIL_IZQ
            )
        elif self._paso == self.PASO_PERFIL_IZQ:
            self._fotos["perfil_izquierdo"] = frame.copy()
            self._pose_fotos["perfil_izquierdo"] = limpiar_para_json(pose)
            # Memorizar el lado realmente usado para exigir el opuesto en el
            # perfil derecho (ver _alineacion_verde).
            yaw_izq = float(pose.get("yaw_deg", 0.0) or 0.0)
            if abs(yaw_izq) > 1.0:
                self._yaw_signo_perfil = 1 if yaw_izq > 0.0 else -1
            destino = (
                self.PASO_MARCA_PERFIL
                if len(self._canales_perfil) > 0
                else self.PASO_PERFIL_DER
            )
        else:
            self._fotos["perfil_derecho"] = frame.copy()
            self._pose_fotos["perfil_derecho"] = limpiar_para_json(pose)
            destino = self.PASO_PARAMETROS

        self._frozen_frame = frame.copy()
        self._frozen_fiduciarios = fiduciarios
        self._destello_hasta = time.perf_counter() + 0.30
        self._ir_a_paso(destino)

    # =====================================================================
    # Guia AR de Repeticion: copiar la colocacion de la sesion anterior
    # =====================================================================

    @staticmethod
    def _normalizar_polo_referente(polo: Any) -> Optional[Dict[str, Any]]:
        """Valida y normaliza un polo de referencia con v_idx y offsets dx, dy."""
        if not isinstance(polo, dict):
            return None
        v_idx = polo.get("v_idx")
        if v_idx is None:
            return None
        try:
            return {
                "v_idx": int(v_idx),
                "dx": float(polo.get("dx") or 0.0),
                "dy": float(polo.get("dy") or 0.0),
                "dip_px": float(polo.get("dip_px") or 0.0),
            }
        except (TypeError, ValueError):
            return None

    def _al_cambiar_sesion_previa(self) -> None:
        indice = self.combo_mediciones_previas.currentIndex()
        if not 0 <= indice < len(self._lista_sesiones_previas):
            self.lbl_fantasma_estado.setText("No hay mediciones previas disponibles.")
            return
        sesion = self._lista_sesiones_previas[indice]
        if self._cargar_montaje_referencia(sesion["ruta_meta"]):
            self.lbl_fantasma_estado.setText(
                f"[Guia AR activa] {sesion['etiqueta']} - hologramas punteados sobre la piel en verde"
            )
            self.lbl_fantasma_estado.setStyleSheet(
                "color: #00ff88; font-size: 11px; font-weight: bold;"
            )
        else:
            self._fantasma_activado = False
            self.btn_fantasma.blockSignals(True)
            self.btn_fantasma.setChecked(False)
            self.btn_fantasma.blockSignals(False)
            self.lbl_fantasma_estado.setText("No se pudo leer esa medicion previa.")
            self.lbl_fantasma_estado.setStyleSheet("color: #ff8a80; font-size: 11px;")

    def _al_alternar_fantasma(self, activado: bool) -> None:
        self._fantasma_activado = bool(activado)
        if not activado:
            self.lbl_fantasma_estado.setText("Guia AR desactivada.")
            self.lbl_fantasma_estado.setStyleSheet("color: #9e9e9e; font-size: 11px;")
            return
        if not getattr(self, "_fantasma_selector_conectado", False):
            self.combo_mediciones_previas.currentIndexChanged.connect(self._al_cambiar_sesion_previa)
            self._fantasma_selector_conectado = True
        self._al_cambiar_sesion_previa()

    def _cargar_montaje_referencia(self, ruta_archivo: str) -> bool:
        """
        Lee el metadata_fotometria.json de una medicion previa y extrae, por cada
        canal, los indices de vertices MediaPipe y offsets locales de cada polo.
        """
        try:
            with open(ruta_archivo, "r", encoding="utf-8") as handle:
                datos = json.load(handle)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error al Cargar Referencia", f"No se pudo leer el archivo:\n{exc}"
            )
            return False

        montaje = datos.get("montaje", {}) if isinstance(datos, dict) else {}
        if not montaje:
            QtWidgets.QMessageBox.warning(
                self,
                "Referencia Invalida",
                "El archivo no contiene el montaje de canales del asistente lineal.",
            )
            return False

        referencia: Dict[str, Dict[str, Any]] = {}
        for clave, info in montaje.items():
            if not isinstance(info, dict):
                continue
            foto_origen = str(info.get("foto", ""))
            pose_ref = "perfil_izq" if foto_origen.startswith("02") else "frontal"
            referencia[str(clave)] = {
                "musculo": str(info.get("musculo", "")),
                "pose": pose_ref,
                "gel": float(info.get("diametro_gel_mm", 10.0) or 10.0),
                "polo_positivo": self._normalizar_polo_referente(info.get("polo_positivo")),
                "polo_negativo": self._normalizar_polo_referente(info.get("polo_negativo")),
            }

        pose_fotos = datos.get("pose_fotos", {}) or {}
        frontal = pose_fotos.get("frontal", {}) or {}
        dist_ref = frontal.get("distancia_camara_cm")
        self._fantasma_dist_ref_cm = (
            float(dist_ref)
            if isinstance(dist_ref, (int, float)) and float(dist_ref) > 0
            else None
        )
        self._montaje_referencia = referencia
        return len(referencia) > 0

    def _dibujar_fantasma(
        self,
        canvas: np.ndarray,
        fiduciarios: Optional[Dict[str, Any]],
        pose: Dict[str, Any],
    ) -> None:
        """Proyecta hologramas punteados con la colocacion de la medicion previa."""
        if not self._fantasma_activado or self._montaje_referencia is None:
            return
        alto, ancho = canvas.shape[:2]
        if fiduciarios is None or not pose:
            self._franja_aviso_fantasma(canvas, ancho,
                "Guia AR activa: ubique el rostro frente a la camara")
            return

        dist_cm = float(pose.get("distancia_camara_cm", 0.0))
        pitch = float(pose.get("pitch_deg", 0.0))
        roll = float(pose.get("roll_deg", 0.0))
        if self._fantasma_dist_ref_cm is not None:
            banda_min = max(20.0, self._fantasma_dist_ref_cm * 0.75)
            banda_max = min(150.0, self._fantasma_dist_ref_cm * 1.30)
        else:
            banda_min, banda_max = 30.0, 70.0
        dist_ok = banda_min <= dist_cm <= banda_max
        pose_ok = abs(pitch) <= 16.0 and abs(roll) <= 11.0
        if not (dist_ok and pose_ok):
            if not dist_ok:
                aviso = (
                    "Guia AR: muestrese mas cerca de la camara"
                    if dist_cm > banda_max
                    else "Guia AR: alejese de la camara"
                )
                aviso += f" - Referencia: {banda_min:.0f} a {banda_max:.0f} cm"
            else:
                aviso = "Guia AR: nivelar pitch y roll para superponer los objetivos"
            self._franja_aviso_fantasma(canvas, ancho, aviso)
            return

        dip_live = float(fiduciarios.get("distancia_interpupilar_px", 0.0))
        if dip_live <= 1.0:
            return
        yaw = float(pose.get("yaw_deg", 0.0))
        puntos_2d = fiduciarios["puntos_2d_px"]

        for clave, info in self._montaje_referencia.items():
            pose_ref = info.get("pose", "frontal")
            if pose_ref in ("perfil_izq", "perfil_der"):
                compatible = 20.0 <= abs(yaw) <= 65.0
            else:
                compatible = abs(yaw) <= 15.0
            if not compatible:
                continue
            gel_mm = float(info.get("gel", 10.0))
            escala_live = self.d_ip_nominal_mm / dip_live
            radio = max(int(round((gel_mm / 2.0) / escala_live)), 6)
            posiciones = {}
            for clave_polo, simbolo in (("polo_positivo", "+"), ("polo_negativo", "-")):
                polo_ref = info.get(clave_polo)
                if polo_ref is None:
                    continue
                v_idx = int(polo_ref["v_idx"])
                if not 0 <= v_idx < len(puntos_2d):
                    continue
                dip_ref = float(polo_ref.get("dip_px") or 0.0)
                factor = (dip_live / dip_ref) if dip_ref > 1.0 else 1.0
                tx = float(puntos_2d[v_idx, 0] + polo_ref["dx"] * factor)
                ty = float(puntos_2d[v_idx, 1] + polo_ref["dy"] * factor)
                posiciones[simbolo] = (tx, ty)
                dibujar_circulo_punteado(canvas, (tx, ty), radio, COLOR_FANTASMA, 2)
                texto = f"{clave} - Polo {simbolo}"
                cv2.putText(canvas, texto, (int(tx) + 12, int(ty) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLOR_FANTASMA, 1, cv2.LINE_AA)
            if "+" in posiciones and "-" in posiciones:
                self._dibujar_linea_punteada(
                    canvas, posiciones["-"], posiciones["+"], COLOR_FANTASMA_LINEA
                )

    def _franja_aviso_fantasma(self, canvas: np.ndarray, ancho: int, texto: str) -> None:
        cv2.rectangle(canvas, (10, 48), (ancho - 10, 76), (90, 80, 20), -1)
        cv2.rectangle(canvas, (10, 48), (ancho - 10, 76), COLOR_FANTASMA, 1)
        cv2.putText(canvas, texto, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _dibujar_linea_punteada(
        canvas: np.ndarray,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        color: Tuple[int, int, int],
        partes: int = 10,
    ) -> None:
        dif_x = (float(p2[0]) - float(p1[0])) / float(partes)
        dif_y = (float(p2[1]) - float(p1[1])) / float(partes)
        for par_de_segmento in range(partes):
            if par_de_segmento % 2 != 0:
                continue
            x1 = int(round(p1[0] + dif_x * par_de_segmento))
            y1 = int(round(p1[1] + dif_y * par_de_segmento))
            x2 = int(round(p1[0] + dif_x * (par_de_segmento + 1))) - 1
            y2 = int(round(p1[1] + dif_y * (par_de_segmento + 1))) - 1
            cv2.line(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # =====================================================================
    # Confirmacion y metadata del montaje
    # =====================================================================

    def _calibrar_montaje_frontal(self) -> None:
        """Genera el payload clasico de triangulacion para compatibilidad retroactiva."""
        foto_frontal = self._fotos.get("frontal")
        if foto_frontal is None:
            return
        posiciones_marcadas = [
            pos for pos in sorted(self._pares_por_canal.keys())
            if pos in self._canales_frontal or self._fotos.get("perfil_izquierdo") is None
        ]
        pares_frontales = [self._pares_por_canal[pos] for pos in posiciones_marcadas]
        if not pares_frontales:
            return
        nombres_marcados = [self._nombre_canal(pos)[1] for pos in posiciones_marcadas]
        indices_fisicos_marcados = [self._nombre_canal(pos)[0] for pos in posiciones_marcadas]
        try:
            resultado = self.agent.triangular_electrodos(
                foto_frontal,
                puntos_centroides_manuales=pares_frontales,
                usar_broches_metalicos=False,
                asignar_musculos_automatico=True,
                nombres_canales=nombres_marcados,
                indices_canales_fisicos=indices_fisicos_marcados,
                d_ip_real_cm=self.d_ip_nominal_mm / 10.0,
                escala_mm_por_px=self._escala_mm_px_frontal,
                punto_referencia_gnd=None,
                diametro_gel_mm=float(self.spin_gel_diam_mm.value()),
                ied_nominal_mm=float(self.spin_ied_nominal_mm.value()),
            )
            meta = resultado.get("metadatos_calibracion_espacial", {})
            if meta:
                self._meta_calibracion_frontal = limpiar_para_json(meta)
        except Exception as exc:
            sys.stderr.write(
                f"[FotometriaWebcam] Aviso de triangulacion frontal del asistente: {exc}\n"
            )

    def _polo_del_par(self, par: Dict[str, Any], clave_ancla: str, clave_polo: str) -> Dict[str, Any]:
        ancla = par.get(clave_ancla) or {}
        punto = par.get(clave_polo)
        dip = par.get("dip_px_foto") or 0.0
        vertice_3d = ancla.get("vertice_3d") or []
        return {
            "px_2d": [round(float(punto[0]), 2), round(float(punto[1]), 2)] if punto else None,
            "v_idx": int(ancla["v_idx"]) if ancla.get("v_idx") is not None else None,
            "dx": round(float(ancla.get("dx", 0.0)), 3),
            "dy": round(float(ancla.get("dy", 0.0)), 3),
            "dip_px": round(float(dip), 2) if dip > 1.0 else None,
            "vertice_3d": [round(float(v), 5) for v in vertice_3d] if vertice_3d else None,
        }

    def _construir_metadata_fotometria(self) -> Dict[str, Any]:
        """Construye el documento metadata_fotometria.json de la sesion."""
        gel = float(self.spin_gel_diam_mm.value())
        ied_nominal = float(self.spin_ied_nominal_mm.value())
        comentario = str(self.edit_comentario.text().strip())

        montaje: Dict[str, Any] = {}
        for pos_canal, par in self._pares_por_canal.items():
            fisico, nombre = self._nombre_canal(pos_canal)
            foto_canal = (
                NOMBRE_FOTO_FRONTAL
                if pos_canal in self._canales_frontal
                else NOMBRE_FOTO_PERFIL_IZQUIERDO
            )
            montaje[f"Ch{pos_canal}"] = {
                "canal_pos": pos_canal,
                "canal_fisico": fisico,
                "musculo": nombre,
                "foto": foto_canal,
                "diametro_gel_mm": round(gel, 2),
                "distancia_interelectrodo_nominal_mm": round(ied_nominal, 2),
                "ied_px": round(float(par["ied_px"]), 2),
                "ied_mm": round(float(par["ied_mm"]), 2) if par.get("ied_mm") is not None else None,
                "angulo_deg": round(float(par["angulo_deg"]), 2),
                "dip_px_foto": round(float(par.get("dip_px_foto") or 0.0), 2),
                "polo_positivo": self._polo_del_par(par, "anchor_positivo", "polo_positivo"),
                "polo_negativo": self._polo_del_par(par, "anchor_negativo", "polo_negativo"),
            }

        # Construir listas explícitas de electrodos y pares_diferenciales con TODOS los canales marcados
        # (tanto frontales como de perfil izquierdo) para que nunca queden vacíos en calibracion_espacial.json
        electrodos_lista: List[Dict[str, Any]] = []
        pares_lista: List[Dict[str, Any]] = []
        for pos_canal, par in sorted(self._pares_por_canal.items()):
            fisico, nombre = self._nombre_canal(pos_canal)
            foto_canal = (
                NOMBRE_FOTO_FRONTAL
                if pos_canal in self._canales_frontal
                else NOMBRE_FOTO_PERFIL_IZQUIERDO
            )
            polo_pos_info = self._polo_del_par(par, "anchor_positivo", "polo_positivo")
            polo_neg_info = self._polo_del_par(par, "anchor_negativo", "polo_negativo")
            pos_pt = par.get("polo_positivo") or (0.0, 0.0)
            neg_pt = par.get("polo_negativo") or (0.0, 0.0)
            cx = round(0.5 * (float(pos_pt[0]) + float(neg_pt[0])), 2)
            cy = round(0.5 * (float(pos_pt[1]) + float(neg_pt[1])), 2)
            par_entry = {
                "canal_pos": pos_canal,
                "canal_fisico": fisico,
                "canal": f"ai{fisico}",
                "musculo": nombre,
                "foto_origen": foto_canal,
                "centroide_px": [cx, cy],
                "polo_positivo": polo_pos_info,
                "polo_negativo": polo_neg_info,
                "ied_px": round(float(par["ied_px"]), 2),
                "ied_mm": round(float(par["ied_mm"]), 2) if par.get("ied_mm") is not None else None,
                "angulo_deg": round(float(par["angulo_deg"]), 2),
                "diametro_gel_mm": round(gel, 2),
            }
            pares_lista.append(par_entry)
            electrodos_lista.append({
                "id_electrodo": f"Ch{pos_canal}_pos",
                "canal_pos": pos_canal,
                "canal_fisico": fisico,
                "canal": f"ai{fisico}",
                "polo": "+",
                "musculo": nombre,
                "foto_origen": foto_canal,
                **polo_pos_info,
            })
            electrodos_lista.append({
                "id_electrodo": f"Ch{pos_canal}_neg",
                "canal_pos": pos_canal,
                "canal_fisico": fisico,
                "canal": f"ai{fisico}",
                "polo": "-",
                "musculo": nombre,
                "foto_origen": foto_canal,
                **polo_neg_info,
            })

        pose_frontal = self._pose_fotos.get("frontal") or {}
        meta_calib = dict(self._meta_calibracion_frontal) if self._meta_calibracion_frontal else {}
        meta_calib["sujeto"] = self.sujeto
        meta_calib["prueba"] = self.prueba
        meta_calib["canales_activos_fisicos"] = list(self.active_channel_indices)
        meta_calib["nombres_musculos_canales"] = list(self.nombres_musculos)
        meta_calib["total_electrodos_detectados"] = len(electrodos_lista)
        meta_calib["total_pares_diferenciales"] = len(pares_lista)
        meta_calib["electrodos"] = electrodos_lista
        meta_calib["pares_diferenciales"] = pares_lista
        meta_calib["montaje"] = montaje
        self._meta_calibracion_frontal = meta_calib

        metadata = {
            "tipo": "asistente_lineal_fotometria",
            "version": 2,
            "fecha_iso": datetime.now().isoformat(),
            "sujeto": self.sujeto,
            "prueba": self.prueba,
            "canales_activos_fisicos": list(self.active_channel_indices),
            "nombres_musculos_canales": list(self.nombres_musculos),
            "diametro_gel_mm": round(gel, 2),
            "distancia_interelectrodo_nominal_mm": round(ied_nominal, 2),
            "distancia_interpupilar_nominal_mm": self.d_ip_nominal_mm,
            "escala_mm_px": (
                round(float(self._escala_mm_px_frontal), 5)
                if self._escala_mm_px_frontal is not None
                else None
            ),
            "comentario_sesion": comentario,
            "fotos_montaje": [
                NOMBRE_FOTO_FRONTAL,
                NOMBRE_FOTO_PERFIL_IZQUIERDO,
                NOMBRE_FOTO_PERFIL_DERECHO,
            ],
            "fotos_montaje_dir": "fotos_montaje",
            "total_electrodos_detectados": len(electrodos_lista),
            "total_pares_diferenciales": len(pares_lista),
            "electrodos": electrodos_lista,
            "pares_diferenciales": pares_lista,
            "montaje": montaje,
            "pose_fotos": self._pose_fotos,
            "pose_resumen_frontal": {
                "distancia_camara_cm": pose_frontal.get("distancia_camara_cm"),
                "yaw_deg": pose_frontal.get("yaw_deg"),
                "rostro_detectado": pose_frontal.get("rostro_detectado", False),
            },
            "metadatos_calibracion_espacial": meta_calib,
        }
        return limpiar_para_json(metadata)

    def _construir_fotos_con_marcas(self) -> Dict[str, np.ndarray]:
        """Copia de cada foto con los electrodos marcados dibujados encima.

        Usa el mismo estilo del marcado en pantalla: anillo del color del canal
        con el diametro de gel escogido, punto rojo en el Polo +, naranja en el
        Polo - y la etiqueta del canal. Guardar la foto cruda sin anotar
        resultaba en puntos perdidos para la revision posterior.
        """
        resultado: Dict[str, np.ndarray] = {}
        for clave, canales_foto in (
            ("frontal", self._canales_frontal),
            ("perfil_izquierdo", self._canales_perfil),
        ):
            base = self._fotos.get(clave)
            if base is None or base.size == 0:
                continue
            pares = [
                (pos, self._pares_por_canal[pos])
                for pos in canales_foto
                if pos in self._pares_por_canal
            ]
            if not pares:
                continue
            canvas = base.copy()
            for pos_canal, par in pares:
                color = PALETA_CANALES_MONTAJE[pos_canal % len(PALETA_CANALES_MONTAJE)]
                p_pos = par.get("polo_positivo")
                p_neg = par.get("polo_negativo")
                if not p_pos or not p_neg:
                    continue
                pos_x, pos_y = int(round(float(p_pos[0]))), int(round(float(p_pos[1])))
                neg_x, neg_y = int(round(float(p_neg[0]))), int(round(float(p_neg[1])))
                escala = par.get("escala_mm_px")
                radio = (
                    max(int(round((float(par.get("diametro_gel_mm", 10.0)) / 2.0) / escala)), 4)
                    if escala
                    else 12
                )
                cv2.arrowedLine(
                    canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25
                )
                cv2.circle(canvas, (pos_x, pos_y), radio, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, (neg_x, neg_y), radio, color, 1, cv2.LINE_AA)
                cv2.circle(canvas, (pos_x, pos_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (pos_x, pos_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas, "+", (pos_x - 4, pos_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.circle(canvas, (neg_x, neg_y), 7, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, (neg_x, neg_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(
                    canvas, "-", (neg_x - 3, neg_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                )
                etiqueta = (
                    f"Canal {par.get('canal_fisico', pos_canal)} "
                    f"- {_texto_sin_acentos(par.get('nombre', ''))}"
                )
                mid_x, mid_y = (pos_x + neg_x) // 2, (pos_y + neg_y) // 2
                cv2.putText(
                    canvas, etiqueta, (mid_x + 10, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 2, cv2.LINE_AA,
                )
            resultado[clave] = canvas
        return resultado

    def _al_confirmar(self) -> None:
        if self._fotos.get("frontal") is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Faltan Capturas",
                "Complete la captura frontal antes de confirmar la sesion.",
            )
            return
        self._calibrar_montaje_frontal()
        self._datos_sesion = {
            "fotos": dict(self._fotos),
            "fotos_con_marcas": self._construir_fotos_con_marcas(),
            "metadata": self._construir_metadata_fotometria(),
            "comentario": str(self.edit_comentario.text().strip()),
        }
        self.accept()

    def obtener_datos_sesion(self) -> Dict[str, Any]:
        """Retorna fotos, metadata_fotometria y el comentario de la sesion."""
        return dict(self._datos_sesion) if self._datos_sesion else {}

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._timer.stop()
        super().closeEvent(event)

    def done(self, result: int) -> None:
        self._timer.stop()
        super().done(result)


class WebcamPhotometryManager(QtCore.QObject):
    """
    Gestor central de fotometría sincrónica por webcam para AutoForge DAQ.

    Coordina el hilo de captura de baja latencia (`WebcamCaptureThread`), el asistente
    interactivo de alineación pre-sesión (`InitialAlignmentDialog`), los disparos no
    bloqueantes (`request_snapshot`) durante las fases de metrónomo y la exportación
    asíncrona de telemetría 3D y biometría labial/mandibular (`export_take_photometry`).
    """

    def __init__(self, parent: Optional[QtCore.QObject] = None, camera_index: int = 0) -> None:
        if isinstance(parent, (int, np.integer)):
            camera_index = int(parent)
            parent = None
        super().__init__(parent)
        self.is_enabled: bool = False
        self.mute_speaking: bool = False
        self.record_video_clips: bool = False
        self.camera_index: int = int(camera_index)
        self.calibracion_inicial_data: Optional[Dict[str, Any]] = None
        self.reference_images: Dict[str, np.ndarray] = {}
        # Montaje del Asistente Lineal Simple de Fotometria de la sesion activa
        self.fotometria_sesion: Optional[Dict[str, Any]] = None

        self.agent = AntigravityAgent(refine_landmarks=True)
        self.capture_thread: Optional[WebcamCaptureThread] = None
        self._retired_threads: List[QtCore.QThread] = []

        self._snapshots_lock = threading.Lock()
        self._agent_lock = threading.Lock()
        self._take_snapshots: List[Tuple[np.ndarray, str, float, int, str, str, int]] = []
        self._take_t0_perf: float = time.perf_counter()
        self._session_label: str = ""
        self._temp_video_path: Optional[str] = None

    @property
    def record_video_clip(self) -> bool:
        """Alias para compatibilidad con llamadas que usan el nombre en singular."""
        return self.record_video_clips

    @record_video_clip.setter
    def record_video_clip(self, value: bool) -> None:
        self.record_video_clips = bool(value)

    def start_worker(self, camera_index: Optional[int] = None) -> bool:
        """Inicia o reutiliza el hilo de captura de cámara web en segundo plano."""
        if camera_index is not None:
            self.camera_index = int(camera_index)

        if (
            self.capture_thread is not None
            and self.capture_thread.isRunning()
            and self.capture_thread.camera_index == self.camera_index
            and self.capture_thread.is_camera_open
        ):
            return True

        self.stop_worker()

        self.capture_thread = WebcamCaptureThread(camera_index=self.camera_index)
        self.capture_thread.start()

        # Esperar brevemente hasta que el dispositivo entregue el primer cuadro o reporte fallo
        t_deadline = time.perf_counter() + 3.5
        while time.perf_counter() < t_deadline:
            if self.capture_thread.is_camera_open:
                frame, _ = self.capture_thread.get_latest_frame()
                if frame is not None:
                    return True
            if not self.capture_thread.isRunning():
                break
            time.sleep(0.03)

        hilo = self.capture_thread
        if hilo is None or not hilo.is_camera_open:
            sys.stderr.write(
                f"[FotometriaWebcam] Camara indice {self.camera_index} no se pudo abrir: "
                "el dispositivo no existe o esta ocupado por otro programa\n"
            )
            return False
        if not self._esperar_senal_camara(0.2):
            sys.stderr.write(
                f"[FotometriaWebcam] Camara indice {self.camera_index} abierta pero sin "
                "fotogramas: revise privacidad o driver V4L2\n"
            )
        return True

    def stop_worker(self) -> None:
        """Detiene de forma segura el hilo de captura evitando QThread::~QThread() abort si el driver tarda en liberar."""
        if hasattr(self, "_retired_threads") and self._retired_threads:
            self._retired_threads = [t for t in self._retired_threads if t is not None and t.isRunning()]
        if self.capture_thread is not None:
            th = self.capture_thread
            self.capture_thread = None
            try:
                th.stop()
            except Exception as exc:
                sys.stderr.write(f"[FotometriaWebcam] Aviso al detener worker: {exc}\n")
            try:
                if th.isRunning():
                    if not hasattr(self, "_retired_threads"):
                        self._retired_threads = []
                    self._retired_threads.append(th)
            except Exception:
                pass

    def _esperar_senal_camara(self, segundos: float) -> bool:
        """Espera hasta que el hilo de captura entregue un cuadro real."""
        hilo = self.capture_thread
        if hilo is None:
            return False
        limite = time.perf_counter() + max(float(segundos), 0.0)
        while time.perf_counter() < limite:
            if hilo.is_camera_open:
                frame, _ = hilo.get_latest_frame()
                if frame is not None:
                    return True
            if not hilo.isRunning():
                return False
            time.sleep(0.05)
        return False

    def _asegurar_camara_disponible(self, parent_widget: Optional[QtWidgets.QWidget]) -> bool:
        """Verifica senal de camara, prueba indices alternativos y avisua si no hay."""
        if self._esperar_senal_camara(3.5):
            return True
        for indice in range(0, 4):
            if indice == self.camera_index:
                continue
            self.start_worker(indice)
            if self._esperar_senal_camara(2.0):
                return True
        self.stop_worker()
        QtWidgets.QMessageBox.warning(
            parent_widget,
            "Camara No Disponible",
            "No hay senal de ninguna camara web.\n\n"
            "Revise:\n"
            " - La tecla de privacidad o la tapa fisica de la camara\n"
            " - Que otro programa no la este usando\n"
            " - Reconectar la camara externa o reiniciar el equipo",
        )
        return False

    def open_alignment_dialog(
        self,
        parent_widget: Optional[QtWidgets.QWidget] = None,
        nombres_musculos: Optional[List[str]] = None,
        active_channel_indices: Optional[List[int]] = None,
        sujeto: str = "",
        prueba: str = "",
    ) -> bool:
        """
        Abre el asistente interactivo de alineación facial y mapeo 3D de electrodos.
        Retorna True si el usuario confirmó la calibración para la sesión.
        """
        self.start_worker(self.camera_index)
        if not self._asegurar_camara_disponible(parent_widget):
            return False

        dialog = InitialAlignmentDialog(
            capture_thread=self.capture_thread,
            agent=self.agent,
            parent=parent_widget,
            nombres_musculos=nombres_musculos,
            active_channel_indices=active_channel_indices,
            sujeto=sujeto,
            prueba=prueba,
            initial_calibration=self.calibracion_inicial_data,
            initial_references=self.reference_images,
        )
        result = dialog.exec()
        if result == QtWidgets.QDialog.Accepted:
            self.calibracion_inicial_data = dialog.calibracion_inicial_data
            self.reference_images = {k: v.copy() for k, v in dialog.reference_images.items()}
            # Sincronizar también con fotometria_sesion para que guardar_fotometria_en_carpeta
            # guarde siempre metadata_fotometria.json y las fotos de referencia con electrodos
            fotos_dict = {
                "frontal": self.reference_images.get("frontal_cruda") if "frontal_cruda" in self.reference_images else self.reference_images.get("frontal_triangulada"),
                "perfil_izquierdo": self.reference_images.get("perfil_izquierdo") if "perfil_izquierdo" in self.reference_images else self.reference_images.get("perfil_45"),
                "perfil_derecho": self.reference_images.get("perfil_derecho"),
            }
            fotos_marcas_dict = {
                "frontal": self.reference_images.get("frontal_triangulada"),
                "perfil_izquierdo": self.reference_images.get("perfil_izquierdo"),
            }
            meta_compat = dict(self.calibracion_inicial_data or {})
            meta_compat.setdefault("tipo", "alignment_dialog_3d")
            meta_compat.setdefault("fecha_iso", datetime.now().isoformat())
            meta_compat["metadatos_calibracion_espacial"] = dict(self.calibracion_inicial_data or {})
            self.fotometria_sesion = {
                "fotos": {k: v for k, v in fotos_dict.items() if v is not None},
                "fotos_con_marcas": {k: v for k, v in fotos_marcas_dict.items() if v is not None},
                "metadata": limpiar_para_json(meta_compat),
                "comentario": "",
            }
            self.is_enabled = True
            return True
        return False

    def set_mute_speaking(self, muted: bool) -> None:
        """Activa o desactiva el silencio de voz (Mute Speaking) en el módulo de fotometría."""
        self.mute_speaking = bool(muted)
        os.environ["NANDU_MUTE_SPEAKING"] = "1" if self.mute_speaking else "0"

    def listar_sesiones_fotometria_previas(self) -> List[Dict[str, Any]]:
        """
        Escanea base_de_datos_electrodos y retorna las mediciones previas guardadas
        con metadata_fotometria.json, de la mas reciente a la mas antigua.
        """
        return listar_sesiones_fotometria_previas()

    def open_asistente_lineal(
        self,
        parent_widget: Optional[QtWidgets.QWidget] = None,
        nombres_musculos: Optional[List[str]] = None,
        active_channel_indices: Optional[List[int]] = None,
        sujeto: str = "",
        prueba: str = "",
    ) -> bool:
        """
        Abre el Asistente Lineal Simple de Fotometria: guia foto a foto de
        colocacion de electrodos con guia AR de repeticion en vivo.

        Retorna True si el usuario confirmo el montaje (o si ya habia confirmado
        previamente el montaje en el boton FOTOMETRIA 3D y decidio usar ese).
        """
        self.start_worker(self.camera_index)
        if not self._asegurar_camara_disponible(parent_widget):
            return bool(self.fotometria_sesion is not None and self.is_enabled)

        dialog = AsistenteLinealFotometriaDialog(
            capture_thread=self.capture_thread,
            agent=self.agent,
            parent=parent_widget,
            nombres_musculos=nombres_musculos,
            active_channel_indices=active_channel_indices,
            sujeto=sujeto,
            prueba=prueba,
            sesiones_previas=self.listar_sesiones_fotometria_previas(),
        )
        resultado = dialog.exec()
        if resultado == QtWidgets.QDialog.Accepted:
            self.fotometria_sesion = dialog.obtener_datos_sesion()
            # Sincronizar inmediatamente calibracion_inicial_data y reference_images
            # para que export_take_photometry NUNCA pise los electrodos marcados con una detección vacía!
            meta_ses = (self.fotometria_sesion or {}).get("metadata", {}) or {}
            self.calibracion_inicial_data = (
                meta_ses.get("metadatos_calibracion_espacial") or meta_ses
            )
            fotos_raw = (self.fotometria_sesion or {}).get("fotos", {}) or {}
            fotos_mk = (self.fotometria_sesion or {}).get("fotos_con_marcas", {}) or {}
            if fotos_raw.get("frontal") is not None:
                self.reference_images["frontal_cruda"] = fotos_raw["frontal"].copy()
            if fotos_mk.get("frontal") is not None:
                self.reference_images["frontal_triangulada"] = fotos_mk["frontal"].copy()
            elif fotos_raw.get("frontal") is not None:
                self.reference_images["frontal_triangulada"] = fotos_raw["frontal"].copy()
            if fotos_mk.get("perfil_izquierdo") is not None:
                self.reference_images["perfil_izquierdo"] = fotos_mk["perfil_izquierdo"].copy()
                self.reference_images["perfil_45"] = fotos_mk["perfil_izquierdo"].copy()
            elif fotos_raw.get("perfil_izquierdo") is not None:
                self.reference_images["perfil_izquierdo"] = fotos_raw["perfil_izquierdo"].copy()
                self.reference_images["perfil_45"] = fotos_raw["perfil_izquierdo"].copy()
            if fotos_raw.get("perfil_derecho") is not None:
                self.reference_images["perfil_derecho"] = fotos_raw["perfil_derecho"].copy()
            self.is_enabled = True
            return True
        # Si el usuario ya había calibrado con FOTOMETRIA 3D antes de iniciar, conservar esos electrodos
        if self.fotometria_sesion is not None and self.calibracion_inicial_data is not None:
            self.is_enabled = True
            return True
        return False

    def guardar_fotometria_inmediata(
        self, sujeto: str, prueba: str, es_continua: bool = False
    ) -> Optional[str]:
        """
        Persiste de inmediato el montaje confirmado del asistente lineal en la
        carpeta activa de la sesion, replicando el esquema de nombres del DAQ:
          - Autograbado: base_de_datos_electrodos/fecha/prueba_sujeto
          - Secuencia Continua: base_de_datos_electrodos/fecha/SecuenciaContinua_prueba_sujeto

        Retorna la carpeta de sesion o None si no hay montaje pendiente.
        """
        if not self.fotometria_sesion:
            return None
        fecha = datetime.now().strftime("%Y-%m-%d")
        nombre_carpeta = (
            f"SecuenciaContinua_{prueba}_{sujeto}" if es_continua else f"{prueba}_{sujeto}"
        )
        carpeta_sesion = os.path.join(
            resolver_directorio_proyecto(),
            "base_de_datos_electrodos",
            fecha,
            nombre_carpeta,
        )
        if self.guardar_fotometria_en_carpeta(carpeta_sesion):
            return carpeta_sesion
        return None

    def guardar_fotometria_en_carpeta(self, output_dir: str) -> bool:
        """
        Escribe el montaje del asistente lineal dentro de una carpeta de medicion:
          - fotos_montaje/01_frontal.png
          - fotos_montaje/02_perfil_izquierdo.png
          - fotos_montaje/03_perfil_derecho.png
          - metadata_fotometria.json

        Retorna True si existia montaje pendiente y se persistio sin errores.
        """
        datos = self.fotometria_sesion
        if not datos:
            return False
        fotos = datos.get("fotos", {}) or {}
        fotos_con_marcas = datos.get("fotos_con_marcas", {}) or {}
        metadata = datos.get("metadata", {}) or {}
        mapeo_fotos = {
            "frontal": NOMBRE_FOTO_FRONTAL,
            "perfil_izquierdo": NOMBRE_FOTO_PERFIL_IZQUIERDO,
            "perfil_derecho": NOMBRE_FOTO_PERFIL_DERECHO,
        }
        try:
            dir_montaje = os.path.join(str(output_dir), "fotos_montaje")
            os.makedirs(dir_montaje, exist_ok=True)
            escritas: List[str] = []
            marcas_escritas: List[str] = []
            for clave, nombre_archivo in mapeo_fotos.items():
                imagen = fotos.get(clave)
                if imagen is None or imagen.size == 0:
                    continue
                destino = os.path.join(dir_montaje, nombre_archivo)
                cv2.imwrite(destino, imagen)
                escritas.append(nombre_archivo)
                imagen_marcas = fotos_con_marcas.get(clave)
                if imagen_marcas is None or imagen_marcas.size == 0:
                    continue
                base, ext = os.path.splitext(nombre_archivo)
                nombre_marcas = f"{base}_marcas{ext}"
                destino_marcas = os.path.join(dir_montaje, nombre_marcas)
                cv2.imwrite(destino_marcas, imagen_marcas)
                marcas_escritas.append(nombre_marcas)
            metadata["fotos_montaje"] = escritas
            metadata["fotos_montaje_marcas"] = marcas_escritas
            metadata.setdefault(
                "comentario_sesion", str(datos.get("comentario", "") or "")
            )
            ruta_json = os.path.join(str(output_dir), NOMBRE_METADATA_FOTOMETRIA)
            with open(ruta_json, "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, ensure_ascii=False)

            # Garantizar también fotometria/calibracion_espacial.json con la ubicación exacta de los electrodos
            dir_fotometria = os.path.join(str(output_dir), "fotometria")
            os.makedirs(dir_fotometria, exist_ok=True)
            calibracion_payload = metadata.get("metadatos_calibracion_espacial") or self.calibracion_inicial_data or metadata
            ruta_calib_json = os.path.join(dir_fotometria, "calibracion_espacial.json")
            with open(ruta_calib_json, "w", encoding="utf-8") as handle_c:
                json.dump(
                    {
                        "exito": True,
                        "metadatos_calibracion_espacial": limpiar_para_json(calibracion_payload),
                    },
                    handle_c,
                    indent=2,
                    ensure_ascii=False,
                )
            return True
        except Exception as exc:
            sys.stderr.write(
                f"[FotometriaWebcam] Error al escribir metadata_fotometria: {exc}\n"
            )
            return False

    def start_take_session(
        self,
        session_label: str = "",
        t0_perf: Optional[float] = None,
    ) -> None:
        """
        Reinicia la cola de instantáneas para una nueva toma experimental y, si está
        activo `record_video_clips`, comienza la grabación temporal del video de la toma.
        """
        with self._snapshots_lock:
            self._take_snapshots.clear()
            self._session_label = str(session_label)
            self._take_t0_perf = float(t0_perf) if t0_perf is not None else time.perf_counter()

        if not self.is_enabled:
            return

        if self.capture_thread is None or not self.capture_thread.isRunning():
            self.start_worker(self.camera_index)

        if self.record_video_clips and self.capture_thread is not None and self.capture_thread.is_camera_open:
            try:
                tmp_dir = tempfile.gettempdir()
                safe_lbl = re.sub(r"[^a-zA-Z0-9_-]", "_", session_label or "toma")
                self._temp_video_path = os.path.join(
                    tmp_dir, f"nandu_clip_{safe_lbl}_{int(time.time() * 1000)}.mp4"
                )
                self.capture_thread.start_video_recording(self._temp_video_path, fps=15.0)
            except Exception as exc:
                sys.stderr.write(f"[FotometriaWebcam] No se pudo iniciar clip temporal: {exc}\n")
                self._temp_video_path = None

    def _capture_snapshot_now(
        self,
        fase: str,
        palabra: str,
        pulso: int,
        t_daq_s: Optional[float],
        sample_rate: float,
    ) -> None:
        if not self.is_enabled or self.capture_thread is None:
            return
        frame, frame_t_perf = self.capture_thread.get_latest_frame(copy=False)
        if frame is None:
            return

        t_wall = time.time()
        if t_daq_s is not None:
            t_rel = float(t_daq_s)
        elif frame_t_perf > 0.0:
            t_rel = max(0.0, float(frame_t_perf - self._take_t0_perf))
        else:
            t_rel = max(0.0, float(time.perf_counter() - self._take_t0_perf))

        sample_idx = int(round(t_rel * float(sample_rate)))
        with self._snapshots_lock:
            self._take_snapshots.append((
                frame,
                t_wall,
                round(t_rel, 6),
                sample_idx,
                str(fase),
                str(palabra),
                int(pulso),
            ))

    def request_snapshot(
        self,
        fase: str,
        palabra: str = "",
        pulso: int = 0,
        t_daq_s: Optional[float] = None,
        sample_rate: float = 1000.0,
        delay_ms: int = 0,
    ) -> None:
        """
        Solicita una instantánea fotométrica de forma no bloqueante (< 0.1 ms).
        Si `delay_ms > 0`, programa la captura mediante `QtCore.QTimer.singleShot`.
        """
        if not self.is_enabled or self.capture_thread is None:
            return

        if delay_ms <= 0:
            self._capture_snapshot_now(fase, palabra, pulso, t_daq_s, sample_rate)
            return

        delayed_t_daq = (
            float(t_daq_s) + (float(delay_ms) / 1000.0) if t_daq_s is not None else None
        )

        app_inst = QtWidgets.QApplication.instance()
        if app_inst is not None and QtCore.QThread.currentThread() == app_inst.thread():
            QtCore.QTimer.singleShot(
                int(delay_ms),
                lambda: self._capture_snapshot_now(
                    fase, palabra, pulso, delayed_t_daq, sample_rate
                ),
            )
        else:
            timer = threading.Timer(
                float(delay_ms) / 1000.0,
                self._capture_snapshot_now,
                args=(fase, palabra, pulso, delayed_t_daq, sample_rate),
            )
            timer.daemon = True
            timer.start()

    def export_take_photometry(self, output_dir: str) -> Dict[str, Any]:
        """
        Procesa en segundo plano (thread-safe para `guardar_async`) todas las instantáneas
        de la toma, extrae la malla 3D y biometría labial/mandibular, y escribe:
          - `fotometria/capturas_toma/*.jpg`
          - `fotometria/calibracion_espacial.json`
          - `fotometria/ref_*.jpg`
          - `fotometria/fotometria_telemetria.csv`
          - `fotometria/fotometria_landmarks.json`
        Retorna el diccionario resumen para `metadata["fotometria_espacial"]`.
        """
        with self._snapshots_lock:
            snapshots = list(self._take_snapshots)
            self._take_snapshots.clear()

        recorded_clip_tmp = None
        if self.capture_thread is not None and self.record_video_clips:
            try:
                recorded_clip_tmp = self.capture_thread.stop_video_recording()
            except Exception:
                recorded_clip_tmp = None

        if not self.is_enabled and not snapshots and not self.reference_images and not self.calibracion_inicial_data:
            return {
                "habilitado": False,
                "total_capturas_toma": 0,
            }

        try:
            fotometria_dir = os.path.join(output_dir, "fotometria")
            capturas_dir = os.path.join(fotometria_dir, "capturas_toma")
            os.makedirs(capturas_dir, exist_ok=True)

            # 0. Persistir el montaje del Asistente Lineal Simple de Fotometria
            #    en la carpeta activa de la medicion: fotos_montaje/ + metadata_fotometria.json
            archivo_metadata_fotometria: Optional[str] = None
            fotos_montaje_escritas: List[str] = []
            if self.fotometria_sesion is not None:
                if self.guardar_fotometria_en_carpeta(output_dir):
                    archivo_metadata_fotometria = NOMBRE_METADATA_FOTOMETRIA
                    dir_montaje_sesion = os.path.join(str(output_dir), "fotos_montaje")
                    for nombre_foto in (
                        NOMBRE_FOTO_FRONTAL,
                        NOMBRE_FOTO_PERFIL_IZQUIERDO,
                        NOMBRE_FOTO_PERFIL_DERECHO,
                    ):
                        if os.path.isfile(os.path.join(dir_montaje_sesion, nombre_foto)):
                            fotos_montaje_escritas.append(nombre_foto)
                else:
                    sys.stderr.write(
                        "[FotometriaWebcam] Aviso: no se pudo persistir el montaje "
                        "del asistente lineal en la medicion\n"
                    )

            # 1. Asegurar generación y guardado de imágenes de referencia pre-sesión:
            #    ref_frontal_cruda.jpg, ref_frontal_triangulada.png, ref_perfil_45.jpg, ref_closeup_labios.jpg
            calibracion_data = (
                dict(self.calibracion_inicial_data)
                if self.calibracion_inicial_data is not None
                else None
            )

            with self._agent_lock:
                # Si faltan referencias pre-sesión pero hay capturas en la toma, auto-generarlas
                if len(snapshots) > 0:
                    frame_base = snapshots[0][0]
                    for snap in snapshots:
                        fids_check = self.agent.extraer_malla_facial_3d(snap[0])
                        if fids_check is not None:
                            frame_base = snap[0]
                            break

                    if "frontal_cruda" not in self.reference_images:
                        self.reference_images["frontal_cruda"] = frame_base.copy()

                    if "frontal_triangulada" not in self.reference_images:
                        res_auto = self.agent.triangular_electrodos(frame_base)
                        vis_auto = self.agent.renderizar_visualizacion(frame_base, res_auto)
                        self.reference_images["frontal_triangulada"] = vis_auto
                        if calibracion_data is None:
                            calibracion_data = res_auto.get("metadatos_calibracion_espacial", {})

                    if "closeup_labios" not in self.reference_images:
                        fids_base = self.agent.extraer_malla_facial_3d(frame_base)
                        crop_lips = InitialAlignmentDialog._crop_closeup_labios(frame_base, fids_base)
                        self.reference_images["closeup_labios"] = crop_lips

                    if "perfil_izquierdo" not in self.reference_images and "perfil_45" not in self.reference_images:
                        for snap in snapshots:
                            fids_snap = self.agent.extraer_malla_facial_3d(snap[0])
                            if fids_snap is not None:
                                pose_snap = self.agent.calcular_pose_y_biometria_facial(fids_snap, snap[0].shape[:2])
                                if 25.0 <= pose_snap.get("yaw_deg", 0.0) <= 65.0:
                                    vis_p = snap[0].copy()
                                    cv2.rectangle(vis_p, (10, 10), (450, 62), (0, 0, 0), -1)
                                    cv2.rectangle(vis_p, (10, 10), (450, 62), (0, 255, 204), 1)
                                    cv2.putText(
                                        vis_p,
                                        f"Perfil Izq Extraido de Toma - Yaw: {pose_snap['yaw_deg']:+.1f} deg",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.48,
                                        (0, 255, 204),
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    self.reference_images["perfil_izquierdo"] = vis_p
                                    self.reference_images["perfil_45"] = vis_p
                                    break

                    if "perfil_derecho" not in self.reference_images:
                        for snap in snapshots:
                            fids_snap = self.agent.extraer_malla_facial_3d(snap[0])
                            if fids_snap is not None:
                                pose_snap = self.agent.calcular_pose_y_biometria_facial(fids_snap, snap[0].shape[:2])
                                if -65.0 <= pose_snap.get("yaw_deg", 0.0) <= -25.0:
                                    vis_p = snap[0].copy()
                                    cv2.rectangle(vis_p, (10, 10), (450, 62), (0, 0, 0), -1)
                                    cv2.rectangle(vis_p, (10, 10), (450, 62), (0, 255, 204), 1)
                                    cv2.putText(
                                        vis_p,
                                        f"Perfil Der Extraido de Toma - Yaw: {pose_snap['yaw_deg']:+.1f} deg",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.48,
                                        (0, 255, 204),
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    self.reference_images["perfil_derecho"] = vis_p
                                    break

            archivos_referencia: Dict[str, str] = {}
            mapping_nombres_exactos = {
                "frontal_cruda": "ref_frontal_cruda.jpg",
                "frontal_triangulada": "ref_frontal_triangulada.png",
                "perfil_izquierdo": "ref_perfil_izquierdo.jpg",
                "perfil_derecho": "ref_perfil_derecho.jpg",
                "perfil_45": "ref_perfil_izquierdo.jpg",
                "closeup_labios": "ref_closeup_labios.jpg",
            }

            for key, img in list(self.reference_images.items()):
                if img is not None and img.size > 0:
                    nombre_archivo = mapping_nombres_exactos.get(key)
                    if nombre_archivo is None:
                        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", str(key))
                        ext = ".png" if "triangulada" in safe_key.lower() else ".jpg"
                        nombre_archivo = f"ref_{safe_key}{ext}"
                    rel_path = os.path.join("fotometria", nombre_archivo)
                    abs_path = os.path.join(output_dir, rel_path)
                    if nombre_archivo.endswith(".png"):
                        cv2.imwrite(abs_path, img)
                    else:
                        cv2.imwrite(abs_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    archivos_referencia[key] = rel_path

            # 2. Si se grabó clip de video temporal, moverlo a fotometria/
            archivo_video_rel: Optional[str] = None
            if recorded_clip_tmp and os.path.exists(recorded_clip_tmp):
                try:
                    dest_video = os.path.join(fotometria_dir, "clip_toma.mp4")
                    shutil.move(recorded_clip_tmp, dest_video)
                    archivo_video_rel = os.path.join("fotometria", "clip_toma.mp4")
                    ts_src = f"{os.path.splitext(recorded_clip_tmp)[0]}_timestamps.csv"
                    if os.path.exists(ts_src):
                        shutil.move(ts_src, os.path.join(fotometria_dir, "clip_toma_timestamps.csv"))
                except Exception as exc:
                    sys.stderr.write(f"[FotometriaWebcam] Aviso al mover clip de video: {exc}\n")

            # 3. Procesar todas las instantáneas de la toma con MediaPipe Face Mesh
            telemetria_rows: List[Dict[str, Any]] = []
            landmarks_records: List[Dict[str, Any]] = []

            with self._agent_lock:
                if calibracion_data is None and len(snapshots) > 0:
                    res_auto = self.agent.triangular_electrodos(snapshots[0][0])
                    calibracion_data = res_auto.get("metadatos_calibracion_espacial", {})

                for idx, (frame, ts_raw, t_daq_s, sample_idx, fase, palabra, pulso) in enumerate(snapshots):
                    ts_iso = (
                        datetime.fromtimestamp(ts_raw).isoformat()
                        if isinstance(ts_raw, (int, float))
                        else str(ts_raw)
                    )
                    safe_fase = re.sub(r"[^a-zA-Z0-9_-]", "_", fase or "evento")
                    safe_palabra = re.sub(r"[^a-zA-Z0-9_-]", "_", palabra or "na")
                    img_filename = f"{idx:03d}_{safe_fase}_p{int(pulso):02d}_{safe_palabra}.jpg"
                    img_abs_path = os.path.join(capturas_dir, img_filename)
                    img_rel_path = os.path.join("fotometria", "capturas_toma", img_filename)

                    try:
                        cv2.imwrite(img_abs_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    except Exception:
                        pass

                    fiduciarios = self.agent.extraer_malla_facial_3d(frame)
                    rostro_ok = fiduciarios is not None
                    if rostro_ok and fiduciarios is not None:
                        pose_bio = self.agent.calcular_pose_y_biometria_facial(
                            fiduciarios, frame.shape[:2]
                        )
                        d_ip_px = round(float(fiduciarios.get("distancia_interpupilar_px", 0.0)), 3)
                        d_ip_3d = round(float(fiduciarios.get("distancia_interpupilar_3d", 0.0)), 6)
                        pts_3d = fiduciarios["puntos_3d"]
                        pts_2d = fiduciarios["puntos_2d_px"]
                        t_mat = fiduciarios.get("transformation_matrix")

                        indices_clave = {
                            "nariz_punta": INDICE_NARIZ_PUNTA,
                            "glabela": INDICE_GLABELA,
                            "menton_gnathion": INDICE_MENTON_GNATHION,
                            "labio_sup_interno": INDICE_LABIO_SUP_INTERNO,
                            "labio_inf_interno": INDICE_LABIO_INF_INTERNO,
                            "comisura_izq": INDICE_COMISURA_IZQ,
                            "comisura_der": INDICE_COMISURA_DER,
                            "pomulo_izq": INDICE_POMULO_IZQ,
                            "pomulo_der": INDICE_POMULO_DER,
                        }
                        fiduciarios_clave = {}
                        for nombre_k, lm_i in indices_clave.items():
                            if lm_i < len(pts_3d):
                                fiduciarios_clave[nombre_k] = {
                                    "indice": lm_i,
                                    "xyz_norm": [round(float(v), 5) for v in pts_3d[lm_i]],
                                    "uv_px": [round(float(v), 2) for v in pts_2d[lm_i]],
                                }

                        landmarks_records.append({
                            "idx": idx,
                            "timestamp_iso": ts_iso,
                            "t_daq_s": t_daq_s,
                            "sample_idx": sample_idx,
                            "fase": fase,
                            "palabra": palabra,
                            "pulso": pulso,
                            "archivo_imagen": img_rel_path,
                            "rostro_detectado": True,
                            "distancia_interpupilar_px": d_ip_px,
                            "distancia_interpupilar_3d": d_ip_3d,
                            "pose_y_biometria": pose_bio,
                            "transformation_matrix": (
                                np.round(t_mat, 6).tolist() if t_mat is not None else None
                            ),
                            "fiduciarios_anatomicos": fiduciarios_clave,
                            "malla_3d_compacta": np.round(pts_3d, 5).tolist(),
                        })
                    else:
                        pose_bio = {
                            "yaw_deg": 0.0,
                            "pitch_deg": 0.0,
                            "roll_deg": 0.0,
                            "distancia_camara_cm": 0.0,
                            "apertura_labial_norm": 0.0,
                            "ancho_comisuras_norm": 0.0,
                            "apertura_mandibular_norm": 0.0,
                            "protrusion_labial_rel": 0.0,
                        }
                        d_ip_px = 0.0
                        d_ip_3d = 0.0
                        landmarks_records.append({
                            "idx": idx,
                            "timestamp_iso": ts_iso,
                            "t_daq_s": t_daq_s,
                            "sample_idx": sample_idx,
                            "fase": fase,
                            "palabra": palabra,
                            "pulso": pulso,
                            "archivo_imagen": img_rel_path,
                            "rostro_detectado": False,
                            "pose_y_biometria": pose_bio,
                        })

                    t_seg = round(float(t_daq_s), 6) if t_daq_s is not None else 0.0
                    telemetria_rows.append({
                        "Tiempo (s)": t_seg,
                        "idx": idx,
                        "timestamp_iso": ts_iso,
                        "t_daq_s": t_daq_s,
                        "sample_idx": sample_idx,
                        "fase": fase,
                        "palabra": palabra,
                        "pulso": pulso,
                        "archivo_imagen": img_rel_path,
                        "rostro_detectado": int(rostro_ok),
                        "yaw_deg": pose_bio["yaw_deg"],
                        "pitch_deg": pose_bio["pitch_deg"],
                        "roll_deg": pose_bio["roll_deg"],
                        "distancia_camara_cm": pose_bio["distancia_camara_cm"],
                        "distancia_interpupilar_px": d_ip_px,
                        "distancia_interpupilar_3d": d_ip_3d,
                        "apertura_labial_norm": pose_bio["apertura_labial_norm"],
                        "ancho_comisuras_norm": pose_bio["ancho_comisuras_norm"],
                        "apertura_mandibular_norm": pose_bio["apertura_mandibular_norm"],
                        "protrusion_labial_rel": pose_bio["protrusion_labial_rel"],
                    })

            # 4. Escribir calibracion_espacial.json
            calibracion_json_path = os.path.join(fotometria_dir, "calibracion_espacial.json")
            payload_calibracion = {
                "exito": bool(calibracion_data),
                "metadatos_calibracion_espacial": calibracion_data or {},
            }
            with open(calibracion_json_path, "w", encoding="utf-8") as f:
                json.dump(payload_calibracion, f, indent=2, ensure_ascii=False)

            # 5. Escribir fotometria_telemetria.csv
            csv_path = os.path.join(fotometria_dir, "fotometria_telemetria.csv")
            csv_fields = [
                "Tiempo (s)",
                "idx",
                "timestamp_iso",
                "t_daq_s",
                "sample_idx",
                "fase",
                "palabra",
                "pulso",
                "archivo_imagen",
                "rostro_detectado",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "distancia_camara_cm",
                "distancia_interpupilar_px",
                "distancia_interpupilar_3d",
                "apertura_labial_norm",
                "ancho_comisuras_norm",
                "apertura_mandibular_norm",
                "protrusion_labial_rel",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writeheader()
                for row in telemetria_rows:
                    writer.writerow(row)

            # 6. Escribir fotometria_landmarks.json
            landmarks_json_path = os.path.join(fotometria_dir, "fotometria_landmarks.json")
            with open(landmarks_json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "session_label": self._session_label,
                        "timestamp_exportacion": datetime.now().isoformat(),
                        "total_capturas": len(landmarks_records),
                        "capturas": landmarks_records,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            # 7. Calcular resumen estadístico para metadata["fotometria_espacial"]
            valid_rows = [r for r in telemetria_rows if r["rostro_detectado"] == 1]
            resumen_pose: Dict[str, Any] = {}
            if valid_rows:
                for key in (
                    "yaw_deg",
                    "pitch_deg",
                    "roll_deg",
                    "distancia_camara_cm",
                    "apertura_labial_norm",
                    "ancho_comisuras_norm",
                    "apertura_mandibular_norm",
                    "protrusion_labial_rel",
                ):
                    arr = np.array([float(r[key]) for r in valid_rows], dtype=np.float64)
                    resumen_pose[key] = {
                        "media": round(float(np.mean(arr)), 4),
                        "std": round(float(np.std(arr)), 4),
                        "min": round(float(np.min(arr)), 4),
                        "max": round(float(np.max(arr)), 4),
                    }

            return {
                "habilitado": True,
                "camera_index": self.camera_index,
                "total_capturas_toma": len(telemetria_rows),
                "capturas_con_rostro_detectado": len(valid_rows),
                "archivos_referencia": archivos_referencia,
                "archivo_calibracion_espacial": os.path.join("fotometria", "calibracion_espacial.json"),
                "archivo_telemetria_csv": os.path.join("fotometria", "fotometria_telemetria.csv"),
                "archivo_landmarks_json": os.path.join("fotometria", "fotometria_landmarks.json"),
                "archivo_video_toma": archivo_video_rel,
                "archivo_metadata_fotometria": archivo_metadata_fotometria,
                "fotos_montaje": fotos_montaje_escritas,
                "resumen_estadistico_toma": resumen_pose,
                "calibracion_inicial_resumen": {
                    "distancia_interpupilar_3d": (calibracion_data or {}).get("distancia_interpupilar_3d"),
                    "distancia_interpupilar_px": (calibracion_data or {}).get("distancia_interpupilar_px"),
                    "total_electrodos_detectados": (calibracion_data or {}).get(
                        "total_electrodos_detectados", 0
                    ),
                    "electrodos": (calibracion_data or {}).get("electrodos", []),
                },
            }
        except Exception as exc:
            sys.stderr.write(f"[FotometriaWebcam] Error en export_take_photometry: {exc}\n")
            return {
                "habilitado": bool(self.is_enabled),
                "error_exportacion": str(exc),
                "total_capturas_toma": len(snapshots),
            }
