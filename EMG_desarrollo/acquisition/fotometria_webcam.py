# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de fotometría automatizada por webcam, biometría facial 3D
#              (pose craneal y articulación labial/mandibular) y asistente
#              interactivo de alineación y mapeo de electrodos sEMG pre-sesión.
# ==============================================================================

import csv
import json
import os
import re
import shutil
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
        self._gnd_electrode_px: Optional[Tuple[float, float]] = None
        self._marking_gnd_active: bool = False
        self._ghost_reference_data: Optional[Dict[str, Any]] = None
        self._ghost_reference_path: str = ""
        self._profile_45_mode: bool = False

        self.setWindowTitle(
            f"Asistente de Alineacion Facial 3D y Mapeo de Electrodos sEMG — {self.sujeto} ({self.prueba})"
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
        hud_group = QtWidgets.QGroupBox("Semáforo de Alineación 3D y Biometría Articular en Vivo")
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
        elec_group = QtWidgets.QGroupBox("1. Configuración y Marcado de Electrodos")
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
        calib_group = QtWidgets.QGroupBox("2. Referencia Métrica y Calibración de Escala")
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

        self.chk_show_ears = QtWidgets.QCheckBox("Mostrar Detección de Orejas (Trago y Lóbulo)")
        self.chk_show_ears.setChecked(True)
        self.chk_show_ears.toggled.connect(self._update_preview)
        calib_layout.addWidget(self.chk_show_ears)

        right_col.addWidget(calib_group)

        # 3. Grupo Ghost Overlay (Referencia Anterior)
        ghost_group = QtWidgets.QGroupBox("3. Repetibilidad entre Sesiones (Ghost Overlay)")
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

        # 4. Grupo de Capturas de Referencia
        captures_group = QtWidgets.QGroupBox("4. Capturas de Referencia de la Sesión")
        captures_layout = QtWidgets.QVBoxLayout(captures_group)

        self.btn_capture_frontal = QtWidgets.QPushButton(
            "1. Capturar Frontal + Triangular Electrodos"
        )
        self.btn_capture_frontal.setStyleSheet(
            "QPushButton { background-color: #004d40; border: 2px solid #00ffcc; padding: 9px; font-size: 13px; }"
            "QPushButton:hover { background-color: #00695c; }"
        )
        self.btn_capture_frontal.clicked.connect(self._on_capture_frontal)
        captures_layout.addWidget(self.btn_capture_frontal)

        profile_row = QtWidgets.QHBoxLayout()
        self.chk_profile_guide = QtWidgets.QCheckBox("Guía Yaw 45°")
        self.chk_profile_guide.toggled.connect(self._on_toggle_profile_mode)
        profile_row.addWidget(self.chk_profile_guide)

        self.btn_capture_profile = QtWidgets.QPushButton("2. Capturar Perfil 45° (Opcional)")
        self.btn_capture_profile.clicked.connect(self._on_capture_profile_45)
        profile_row.addWidget(self.btn_capture_profile, stretch=1)
        captures_layout.addLayout(profile_row)

        # Miniaturas de verificación
        thumbs_row = QtWidgets.QHBoxLayout()
        self.lbl_thumb_frontal = self._create_thumbnail_label("Frontal 3D\n(Pendiente)")
        self.lbl_thumb_lips = self._create_thumbnail_label("Close-up Labios\n(Pendiente)")
        self.lbl_thumb_profile = self._create_thumbnail_label("Perfil 45°\n(Opcional)")
        thumbs_row.addWidget(self.lbl_thumb_frontal)
        thumbs_row.addWidget(self.lbl_thumb_lips)
        thumbs_row.addWidget(self.lbl_thumb_profile)
        captures_layout.addLayout(thumbs_row)

        self.lbl_capture_summary = QtWidgets.QLabel(
            "Alinee el rostro dentro de la elipse guía y capture la referencia frontal."
        )
        self.lbl_capture_summary.setWordWrap(True)
        self.lbl_capture_summary.setStyleSheet("color: #00ffcc; font-size: 12px;")
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
        lbl.setFixedSize(115, 86)
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

        # Si el usuario activó marcar Tierra GND
        if self._marking_gnd_active:
            self._gnd_electrode_px = (img_x, img_y)
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
            else:
                # Segundo clic: Polo (-)
                pos_pt = self._pending_pos_pole
                neg_pt = (img_x, img_y)
                self._pending_pos_pole = None

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
                    "centroide": (cx, cy),
                    "ied_px": ied_px,
                    "ied_mm": ied_mm,
                    "angulo_deg": ang_deg,
                    "canal_fisico": ch_num,
                    "nombre": musc,
                }
                self._differential_pairs.append(pair_info)
                self._manual_points_px.append((cx, cy))
        else:
            max_channels = max(len(self.etiquetas_canales), 1)
            if len(self._manual_points_px) < max_channels:
                self._manual_points_px.append((img_x, img_y))
            else:
                self._manual_points_px[-1] = (img_x, img_y)

        self._refresh_channel_list_ui()
        self._update_preview()

    def _on_video_right_clicked(self, _img_x: float, _img_y: float) -> None:
        if self._marking_gnd_active:
            self._marking_gnd_active = False
            self.btn_mark_gnd.setChecked(False)
        elif self._pending_pos_pole is not None:
            self._pending_pos_pole = None
        elif self.combo_electrode_type.currentIndex() == 0:
            if self._differential_pairs:
                self._differential_pairs.pop()
            if self._manual_points_px:
                self._manual_points_px.pop()
        else:
            if self._manual_points_px:
                self._manual_points_px.pop()
        self._refresh_channel_list_ui()
        self._update_preview()

    def _clear_manual_points(self) -> None:
        self._manual_points_px.clear()
        self._differential_pairs.clear()
        self._pending_pos_pole = None
        self._gnd_electrode_px = None
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

        # Evaluar tolerancias según modo frontal o perfil 45°
        if self._profile_45_mode:
            yaw_ok = 30.0 <= abs(yaw) <= 55.0
            yaw_warn = 22.0 <= abs(yaw) <= 62.0
        else:
            yaw_ok = abs(yaw) <= 6.0
            yaw_warn = abs(yaw) <= 10.0

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
        modo_str = "45°" if self._profile_45_mode else "0°"
        self._set_badge_status(
            self.badge_yaw,
            f"Yaw ({modo_str}): {yaw:+.1f}°",
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
            else (
                "MODO PERFIL 45: GIRE EL ROSTRO HASTA 30-55 GRADOS"
                if self._profile_45_mode
                else "AJUSTE GIRO / INCLINACION HASTA PONER SEMAFOROS EN VERDE"
            )
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

        if mode_idx == 0 and es_bipolar:
            # Dibujar pares diferenciales bipolares (dos polos fisicos reales: + y -)
            for idx, pair in enumerate(self._differential_pairs):
                color = colores[idx % len(colores)]
                pos_x, pos_y = int(round(pair["polo_positivo"][0])), int(round(pair["polo_positivo"][1]))
                neg_x, neg_y = int(round(pair["polo_negativo"][0])), int(round(pair["polo_negativo"][1]))

                # Flecha que representa el vector dipolo diferencial D = P_+ - P_- (de negativo a positivo)
                cv2.arrowedLine(canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25)

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

    def _on_capture_profile_45(self) -> None:
        # Si el usuario aún no activó la guía 45°, activarla y capturar el cuadro actual
        if not self.chk_profile_guide.isChecked():
            self.chk_profile_guide.setChecked(True)

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
        cv2.rectangle(vis_perfil, (10, 10), (430, 62), (0, 0, 0), -1)
        cv2.rectangle(vis_perfil, (10, 10), (430, 62), (0, 255, 204), 1)
        cv2.putText(
            vis_perfil,
            f"Referencia Perfil 45 Grados | {self.sujeto}",
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

        self.reference_images["perfil_45"] = vis_perfil
        if self.calibracion_inicial_data is not None:
            self.calibracion_inicial_data["pose_perfil_45"] = pose

        self._refresh_thumbnails_from_state()
        self.lbl_capture_summary.setText(
            f"[OK] Perfil 45 capturado (Yaw = {yaw_val:+.1f} deg). Puede volver a Frontal o Confirmar."
        )

    def _on_confirm_session(self) -> None:
        if "frontal_cruda" not in self.reference_images or self.calibracion_inicial_data is None:
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
        if "perfil_45" in self.reference_images:
            self._display_bgr_on_label(
                self.reference_images["perfil_45"], self.lbl_thumb_profile
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
        self.record_video_clips: bool = False
        self.camera_index: int = int(camera_index)
        self.calibracion_inicial_data: Optional[Dict[str, Any]] = None
        self.reference_images: Dict[str, np.ndarray] = {}

        self.agent = AntigravityAgent(refine_landmarks=True)
        self.capture_thread: Optional[WebcamCaptureThread] = None

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
        t_deadline = time.perf_counter() + 1.8
        while time.perf_counter() < t_deadline:
            if self.capture_thread.is_camera_open:
                frame, _ = self.capture_thread.get_latest_frame()
                if frame is not None:
                    return True
            if not self.capture_thread.isRunning():
                break
            time.sleep(0.03)

        return bool(self.capture_thread and self.capture_thread.is_camera_open)

    def stop_worker(self) -> None:
        """Detiene de forma segura el hilo de captura de cámara web."""
        if self.capture_thread is not None:
            try:
                self.capture_thread.stop()
            except Exception as exc:
                sys.stderr.write(f"[FotometriaWebcam] Aviso al detener worker: {exc}\n")
            self.capture_thread = None

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
            self.is_enabled = True
            return True
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

                    if "perfil_45" not in self.reference_images:
                        for snap in snapshots:
                            fids_snap = self.agent.extraer_malla_facial_3d(snap[0])
                            if fids_snap is not None:
                                pose_snap = self.agent.calcular_pose_y_biometria_facial(fids_snap, snap[0].shape[:2])
                                if 25.0 <= abs(pose_snap.get("yaw_deg", 0.0)) <= 65.0:
                                    vis_p = snap[0].copy()
                                    cv2.rectangle(vis_p, (10, 10), (430, 62), (0, 0, 0), -1)
                                    cv2.rectangle(vis_p, (10, 10), (430, 62), (0, 255, 204), 1)
                                    cv2.putText(
                                        vis_p,
                                        f"Perfil Extraido de Toma | Yaw: {pose_snap['yaw_deg']:+.1f} deg",
                                        (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.50,
                                        (0, 255, 204),
                                        1,
                                        cv2.LINE_AA,
                                    )
                                    self.reference_images["perfil_45"] = vis_p
                                    break

            archivos_referencia: Dict[str, str] = {}
            mapping_nombres_exactos = {
                "frontal_cruda": "ref_frontal_cruda.jpg",
                "frontal_triangulada": "ref_frontal_triangulada.png",
                "perfil_45": "ref_perfil_45.jpg",
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

                    telemetria_rows.append({
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
