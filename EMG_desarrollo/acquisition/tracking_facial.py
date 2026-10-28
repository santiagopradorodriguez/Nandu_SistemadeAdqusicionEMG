# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Hilo de inferencia facial en tiempo real (FaceTrackingThread) que
#              consume frames de cámara/archivo/video compartido, corre MediaPipe
#              FaceLandmarker en modo VIDEO y escribe un CSV por frame (pose +
#              biometría) alineable con la señal EMG por ancla temporal.
# ==============================================================================

"""
Módulo de trackeo facial en tiempo real para NANDU LSD.

Provee la clase :class:`FaceTrackingThread`, un hilo (QThread con fallback a
threading.Thread) que:

* Consume frames de una fuente configurable (cámara compartida, cámara propia o
  archivo de video) con back-pressure (descarta frames viejos, nunca encola).
* Corre ``mediapipe.tasks.vision.FaceLandmarker`` en modo ``VIDEO`` (seguimiento
  temporal entre cuadros) con ``num_faces=1``.
* Reusa la lógica de extracción de malla y de cálculo de pose/biometría de
  ``calibracion_espacial_electrodos.AntigravityAgent`` (no se duplica la
  matemática), y el criterio de búsqueda de ruta del modelo
  ``face_landmarker.task`` de ese mismo módulo.
* Escribe ``tracking_facial.csv`` con el schema congelado:

  ``t_perf,t_epoch,frame_idx,fps_inst,pitch,yaw,roll,apertura_boca,ancho_boca,ok_deteccion``

  y, opcionalmente con ``--landmarks``, ``tracking_landmarks.csv`` con los
  puntos 3D normalizados (478 x 3 por cuadro).

Uso standalone::

    venv/bin/python EMG_desarrollo/acquisition/tracking_facial.py \\
        --fuente EMG_desarrollo/base_de_datos_electrodos/2026-10-23/WIN_*.mp4 \\
        --duracion 15 --salida /tmp/opencode
"""

import argparse
import csv
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Import robusto de Qt (el módulo debe ser importable aunque no haya PySide6)
# ---------------------------------------------------------------------------
try:
    from PySide6 import QtCore
except ImportError:
    try:
        from PyQt5 import QtCore  # type: ignore
        if not hasattr(QtCore, "Signal") and hasattr(QtCore, "pyqtSignal"):
            QtCore.Signal = QtCore.pyqtSignal
        if not hasattr(QtCore, "Slot") and hasattr(QtCore, "pyqtSlot"):
            QtCore.Slot = QtCore.pyqtSlot
    except ImportError:
        QtCore = None  # type: ignore

if QtCore is not None and hasattr(QtCore, "QCoreApplication"):
    try:
        QtCore.QCoreApplication.setLibraryPaths(
            [p for p in QtCore.QCoreApplication.libraryPaths() if "cv2" not in p]
        )
    except Exception:
        pass

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

# Reuso del agente existente (sin efectos secundarios al importar el módulo)
try:
    from .calibracion_espacial_electrodos import AntigravityAgent
except ImportError:
    try:
        from calibracion_espacial_electrodos import AntigravityAgent
    except ImportError:
        AntigravityAgent = None  # type: ignore

try:
    from .fotometria_webcam import WebcamCaptureThread
except ImportError:
    try:
        from fotometria_webcam import WebcamCaptureThread
    except ImportError:
        WebcamCaptureThread = None  # type: ignore


# ---------------------------------------------------------------------------
# Clase base del hilo: QThread si está disponible, si no threading.Thread
# ---------------------------------------------------------------------------
if QtCore is not None and hasattr(QtCore, "QThread"):
    _BaseThread = QtCore.QThread
else:
    _BaseThread = threading.Thread


class _VideoLandmarkerWrapper:
    """Adapta un FaceLandmarker en modo VIDEO a la interfaz `.detect(mp_image)`.

    ``AntigravityAgent.extraer_malla_facial_3d`` llama a ``self.detector.detect(...)``
    (API de modo IMAGE). Como el landmarker de modo VIDEO sólo acepta
    ``detect_for_video``, este wrapper incrementa un contador de timestamps
    monotónico por llamada y delega en ``detect_for_video``. Así se reusa
    exactamente la lógica de extracción de malla del agente existente.
    """

    def __init__(self, landmarker: Any, step_ms: float) -> None:
        self._landmarker = landmarker
        self._step_ms = float(step_ms) if step_ms and step_ms > 0 else 33.333
        self._counter = 0
        self._lock = threading.Lock()

    def detect(self, mp_image: Any) -> Any:
        with self._lock:
            ts_ms = int(self._counter * self._step_ms)
            self._counter += 1
            return self._landmarker.detect_for_video(mp_image, ts_ms)


class FaceTrackingThread(_BaseThread):
    """Hilo de inferencia facial en tiempo real.

    Parámetros del constructor (todos opcionales, se infieren si no se pasan):

    * ``fuente`` / ``modo`` / ``source``: ``"compartido"``, ``"propio"``,
      ``"archivo"`` o una especificación estilo CLI (``"cam:0"``,
      ``"/ruta/video.mp4"``).
    * ``capture_thread`` / ``webcam_thread`` / ``hilo_captura`` /
      ``compartido``: referencia a un ``WebcamCaptureThread`` vivo (modo
      compartido).
    * ``camera_index`` / ``cam_index``: índice de cámara (modo propio).
    * ``video_path`` / ``ruta_video`` / ``archivo``: ruta de video (modo
      archivo, sólo tests/offline).
    * ``salida`` / ``output_dir`` / ``directorio_salida``: directorio de salida.
    * ``landmarks`` / ``escribir_landmarks`` / ``guardar_landmarks``: activa
      ``tracking_landmarks.csv``.
    * ``duracion`` / ``duration``: duración en segundos (``None`` = hasta stop()).
    * ``parent``: parent Qt (si aplica).
    """

    def __init__(
        self,
        fuente: Any = None,
        *,
        modo: Optional[str] = None,
        source: Any = None,
        camera_index: int = 0,
        cam_index: Optional[int] = None,
        video_path: Optional[str] = None,
        ruta_video: Optional[str] = None,
        archivo: Optional[str] = None,
        capture_thread: Any = None,
        webcam_thread: Any = None,
        hilo_captura: Any = None,
        compartido: Any = None,
        salida: Optional[str] = None,
        output_dir: Optional[str] = None,
        directorio_salida: Optional[str] = None,
        landmarks: bool = False,
        escribir_landmarks: Optional[bool] = None,
        guardar_landmarks: Optional[bool] = None,
        duracion: Optional[float] = None,
        duration: Optional[float] = None,
        parent: Any = None,
        **kwargs: Any,
    ) -> None:
        # Compatibilidad de firma con QThread/threading.Thread
        if QtCore is not None and _BaseThread is getattr(QtCore, "QThread", None):
            try:
                super().__init__(parent)
            except TypeError:
                super().__init__()
        else:
            try:
                super().__init__()
            except Exception:
                pass

        # Resolver alias
        if cam_index is not None:
            camera_index = int(cam_index)
        if escribir_landmarks is not None:
            landmarks = bool(escribir_landmarks)
        if guardar_landmarks is not None:
            landmarks = bool(guardar_landmarks)
        if duration is not None:
            duracion = duration

        ct = capture_thread or webcam_thread or hilo_captura or compartido
        vp = video_path or ruta_video or archivo

        # Resolver modo a partir de `fuente`/`modo`/`source` si no vino explícito
        modo_resuelto = modo
        fuente_str: Optional[str] = None
        if isinstance(fuente, str):
            fuente_str = fuente.strip()
        elif fuente is not None and not isinstance(fuente, str):
            # Podría ser un WebcamCaptureThread pasado posicionalmente
            if WebcamCaptureThread is not None and isinstance(fuente, WebcamCaptureThread):
                ct = fuente
                modo_resuelto = "compartido"
            elif os.path.exists(str(fuente)):
                vp = str(fuente)
                modo_resuelto = "archivo"
            else:
                fuente_str = str(fuente).strip()

        if modo_resuelto is None:
            # inferir
            if ct is not None:
                modo_resuelto = "compartido"
            elif vp is not None:
                modo_resuelto = "archivo"
            elif fuente_str is not None:
                if fuente_str.lower().startswith("cam:"):
                    modo_resuelto = "propio"
                    try:
                        camera_index = int(fuente_str.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
                elif fuente_str.lower() == "compartido":
                    modo_resuelto = "compartido"
                elif os.path.exists(fuente_str):
                    modo_resuelto = "archivo"
                    vp = fuente_str
                else:
                    # Por defecto asumimos ruta de archivo (tests offline)
                    modo_resuelto = "archivo"
                    vp = fuente_str
            else:
                modo_resuelto = "propio"

        if source is not None:
            if isinstance(source, str):
                if source.lower().startswith("cam:"):
                    modo_resuelto = "propio"
                    try:
                        camera_index = int(source.split(":", 1)[1])
                    except (ValueError, IndexError):
                        pass
                elif os.path.exists(source):
                    modo_resuelto = "archivo"
                    vp = source
                elif source.lower() == "compartido":
                    modo_resuelto = "compartido"
            elif WebcamCaptureThread is not None and isinstance(source, WebcamCaptureThread):
                ct = source
                modo_resuelto = "compartido"

        # kwargs sueltos (por compatibilidad)
        for k, v in (kwargs or {}).items():
            kl = k.lower()
            if kl in ("camera_index", "cam_index", "indice_camara"):
                camera_index = int(v)
            elif kl in ("video_path", "ruta_video", "archivo", "path"):
                vp = str(v)
                modo_resuelto = "archivo"
            elif kl in ("capture_thread", "webcam_thread", "hilo_captura", "compartido"):
                ct = v
                modo_resuelto = "compartido"
            elif kl in ("salida", "output_dir", "directorio_salida", "outdir"):
                salida = str(v)
            elif kl in ("landmarks", "escribir_landmarks", "guardar_landmarks"):
                landmarks = bool(v)
            elif kl in ("duracion", "duration"):
                duracion = v

        self.modo: str = str(modo_resuelto or "propio").lower()
        self.camera_index: int = int(camera_index)
        self.video_path: Optional[str] = vp
        self.capture_thread: Any = ct
        self.output_dir: str = str(
            salida or output_dir or directorio_salida or "/tmp/opencode"
        )
        self.escribir_landmarks: bool = bool(landmarks)
        self.duracion: Optional[float] = (
            float(duracion) if duracion is not None else None
        )

        # Estado interno
        self._running: bool = False
        self._t_inicio: float = 0.0
        self._csv_file: Any = None
        self._csv_writer: Any = None
        self._lm_file: Any = None
        self._lm_writer: Any = None
        self._landmarker: Any = None
        self._agente: Any = None
        self._cap: Any = None
        self._lock: threading.Lock = threading.Lock()
        self._frame_idx: int = 0
        self._t_perf_prev: Optional[float] = None
        self._stats: Dict[str, Any] = {
            "frames": 0,
            "fps_medio": 0.0,
            "deteccion": 0,
            "deteccion_pct": 0.0,
            "csv_path": "",
            "landmarks_path": "",
        }
        self._fps_acum: List[float] = []

    # ------------------------------------------------------------------ #
    # API de control
    # ------------------------------------------------------------------ #
    @property
    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def isRunning(self) -> bool:  # noqa: N802 (compat Qt)
        if hasattr(super(), "isRunning"):
            try:
                return bool(super().isRunning())
            except Exception:
                pass
        if hasattr(self, "is_alive"):
            try:
                return bool(self.is_alive())
            except Exception:
                pass
        return bool(self._running)

    def wait(self, timeout: Optional[int] = None) -> bool:  # noqa: N802
        if hasattr(super(), "wait"):
            try:
                if timeout is None:
                    return bool(super().wait())
                return bool(super().wait(int(timeout)))
            except Exception:
                pass
        if hasattr(self, "join"):
            try:
                secs = None if timeout is None else float(timeout) / 1000.0
                self.join(secs)
                return True
            except Exception:
                return False
        return True

    def stop(self) -> None:
        """Detiene el hilo y espera su finalización (cierre limpio)."""
        self._running = False
        try:
            self.wait(8000)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Internos
    # ------------------------------------------------------------------ #
    def _localizar_modelo(self) -> Optional[str]:
        """Mismo criterio de búsqueda de ruta que AntigravityAgent (calibracion)."""
        rutas_candidatas = [
            None,
            os.path.join(
                getattr(sys, "_MEIPASS", ""), "DataConfig", "modelos_vision", "face_landmarker.task"
            )
            if hasattr(sys, "_MEIPASS")
            else None,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "DataConfig", "modelos_vision", "face_landmarker.task"
                )
            ),
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "EMG_desarrollo",
                    "DataConfig",
                    "modelos_vision",
                    "face_landmarker.task",
                )
            ),
            "/home/santiago/Documentos/codigos/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task",
            "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task",
        ]
        for r in rutas_candidatas:
            if r and os.path.exists(r):
                return r
        return None

    def _init_landmarker(self) -> bool:
        if mp is None:
            sys.stderr.write(
                "[FaceTrackingThread] mediapipe no disponible: no se puede inicializar FaceLandmarker.\n"
            )
            return False
        ruta_modelo = self._localizar_modelo()
        if not ruta_modelo:
            sys.stderr.write(
                "[FaceTrackingThread] ADVERTENCIA: no se encontró face_landmarker.task. "
                "Trackeo facial deshabilitado (fallo limpio, sin crash del proceso).\n"
            )
            return False

        # step_ms para timestamps monotónicos del modo VIDEO
        if self.modo == "archivo" and self.video_path:
            try:
                _cap_tmp = cv2.VideoCapture(self.video_path)
                fps_v = float(_cap_tmp.get(cv2.CAP_PROP_FPS) or 0.0)
                _cap_tmp.release()
                step_ms = (1000.0 / fps_v) if fps_v > 0 else 33.333
            except Exception:
                step_ms = 33.333
        else:
            step_ms = 33.333

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            base_options = mp_python.BaseOptions(model_asset_path=ruta_modelo)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.VIDEO,
                num_faces=1,
                output_facial_transformation_matrixes=True,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        except Exception as exc:
            sys.stderr.write(
                f"[FaceTrackingThread] Error inicializando FaceLandmarker: {exc}\n"
            )
            self._landmarker = None
            return False

        # Agente de cálculo (reuso de calcular_pose_y_biometria_facial)
        if AntigravityAgent is not None:
            try:
                self._agente = AntigravityAgent(refine_landmarks=True)
                wrapper = _VideoLandmarkerWrapper(self._landmarker, step_ms)
                self._agente.modo_moderno = True
                self._agente._modo_api = "tasks"
                self._agente.detector = wrapper
            except Exception as exc:
                sys.stderr.write(
                    f"[FaceTrackingThread] Aviso: no se pudo preparar el agente de pose: {exc}\n"
                )
                self._agente = None
        else:
            sys.stderr.write(
                "[FaceTrackingThread] AntigravityAgent no disponible: pose/biometría en 0.\n"
            )
            self._agente = None

        return True

    def _open_capture_propio(self) -> bool:
        """Abre VideoCapture propio con fallback de APIs (igual que fotometría)."""
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
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self._cap = cap
                return True
        except Exception as exc:
            sys.stderr.write(
                f"[FaceTrackingThread] Excepción abriendo cámara {self.camera_index}: {exc}\n"
            )
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        self._cap = None
        return False

    def _preparar_salida(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "tracking_facial.csv")
        self._csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            [
                "t_perf",
                "t_epoch",
                "frame_idx",
                "fps_inst",
                "pitch",
                "yaw",
                "roll",
                "apertura_boca",
                "ancho_boca",
                "ok_deteccion",
            ]
        )
        self._csv_file.flush()
        self._stats["csv_path"] = csv_path

        if self.escribir_landmarks:
            lm_path = os.path.join(self.output_dir, "tracking_landmarks.csv")
            self._lm_file = open(lm_path, "w", newline="", encoding="utf-8")
            self._lm_writer = csv.writer(self._lm_file)
            header = ["t_perf", "frame_idx"]
            # 478 puntos x 3 coords (lo que entrega el modelo MediaPipe)
            n_lm = 478
            for i in range(n_lm):
                header += [f"lm_{i}_x", f"lm_{i}_y", f"lm_{i}_z"]
            self._lm_writer.writerow(header)
            self._lm_file.flush()
            self._stats["landmarks_path"] = lm_path

    def _cerrar_salida(self) -> None:
        try:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
        except Exception:
            pass
        self._csv_file = None
        self._csv_writer = None
        try:
            if self._lm_file is not None:
                self._lm_file.flush()
                self._lm_file.close()
        except Exception:
            pass
        self._lm_file = None
        self._lm_writer = None

    def _extraer_fiduciarios(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        if self._landmarker is None:
            return None
        try:
            # El agente tiene inyectado el landmarker en modo VIDEO; se reusa su
            # extracción de malla 3D exacta (sin duplicar la lógica).
            if self._agente is None:
                return None
            fiduciarios = self._agente.extraer_malla_facial_3d(frame)
            return fiduciarios
        except Exception as exc:
            sys.stderr.write(f"[FaceTrackingThread] Aviso extracción malla: {exc}\n")
            return None

    def _procesar_frame(self, frame: np.ndarray, t_perf: float) -> None:
        fiduciarios = self._extraer_fiduciarios(frame)
        ok_det = 1 if fiduciarios is not None else 0

        pitch = yaw = roll = 0.0
        apertura = ancho = 0.0
        puntos_3d = None
        if ok_det and self._agente is not None:
            try:
                pose = self._agente.calcular_pose_y_biometria_facial(
                    fiduciarios, frame.shape[:2]
                )
                pitch = float(pose.get("pitch_deg", 0.0))
                yaw = float(pose.get("yaw_deg", 0.0))
                roll = float(pose.get("roll_deg", 0.0))
                apertura = float(pose.get("apertura_labial_norm", 0.0))
                ancho = float(pose.get("ancho_comisuras_norm", 0.0))
                puntos_3d = fiduciarios.get("puntos_3d")
            except Exception as exc:
                sys.stderr.write(f"[FaceTrackingThread] Aviso pose: {exc}\n")
                ok_det = 0

        # fps instantáneo
        if self._t_perf_prev is not None and (t_perf - self._t_perf_prev) > 0:
            fps_inst = 1.0 / (t_perf - self._t_perf_prev)
        else:
            fps_inst = 0.0
        self._t_perf_prev = t_perf

        t_epoch = time.time()
        self._csv_writer.writerow(
            [
                repr(float(t_perf)),
                repr(float(t_epoch)),
                int(self._frame_idx),
                repr(float(fps_inst)),
                repr(float(pitch)),
                repr(float(yaw)),
                repr(float(roll)),
                repr(float(apertura)),
                repr(float(ancho)),
                int(ok_det),
            ]
        )
        self._csv_file.flush()

        if self.escribir_landmarks and self._lm_writer is not None:
            row = [repr(float(t_perf)), int(self._frame_idx)]
            n_lm = 478
            if puntos_3d is not None:
                arr = np.asarray(puntos_3d, dtype=np.float64)
                # rellenar/truncar a n_lm
                for i in range(n_lm):
                    if i < arr.shape[0]:
                        row += [repr(float(arr[i, 0])), repr(float(arr[i, 1])), repr(float(arr[i, 2]))]
                    else:
                        row += ["0.0", "0.0", "0.0"]
            else:
                row += ["0.0", "0.0", "0.0"] * n_lm
            self._lm_writer.writerow(row)
            self._lm_file.flush()

        self._frame_idx += 1
        self._stats["frames"] = self._frame_idx
        self._stats["deteccion"] += int(ok_det)
        if self._frame_idx > 0:
            self._stats["deteccion_pct"] = 100.0 * self._stats["deteccion"] / self._frame_idx
        if fps_inst > 0:
            self._fps_acum.append(fps_inst)
            if self._fps_acum:
                self._stats["fps_medio"] = float(
                    sum(self._fps_acum) / len(self._fps_acum)
                )

    def run(self) -> None:
        self._running = True
        self._t_inicio = time.perf_counter()
        self._frame_idx = 0
        self._t_perf_prev = None
        self._fps_acum = []

        # Preparar salida
        try:
            self._preparar_salida()
        except Exception as exc:
            sys.stderr.write(f"[FaceTrackingThread] Error preparando salida: {exc}\n")
            self._running = False
            return

        # Inicializar landmarker
        if not self._init_landmarker():
            # Falo limpio: cerrar archivos y salir
            self._cerrar_salida()
            self._running = False
            return

        # Abrir fuente
        if self.modo == "compartido":
            if self.capture_thread is None:
                sys.stderr.write(
                    "[FaceTrackingThread] Modo compartido sin WebcamCaptureThread: abortando.\n"
                )
                self._cerrar_salida()
                self._running = False
                return
            # No abrimos captura propia
        elif self.modo == "propio":
            if not self._open_capture_propio():
                sys.stderr.write(
                    f"[FaceTrackingThread] No se pudo abrir cámara {self.camera_index}.\n"
                )
                self._cerrar_salida()
                self._running = False
                return
        elif self.modo == "archivo":
            if not self.video_path or not os.path.exists(self.video_path):
                sys.stderr.write(
                    f"[FaceTrackingThread] Archivo de video no encontrado: {self.video_path}\n"
                )
                self._cerrar_salida()
                self._running = False
                return
            try:
                self._cap = cv2.VideoCapture(self.video_path)
                if not self._cap.isOpened():
                    raise RuntimeError("VideoCapture no se pudo abrir")
            except Exception as exc:
                sys.stderr.write(f"[FaceTrackingThread] Error abriendo video: {exc}\n")
                self._cerrar_salida()
                self._running = False
                return
        else:
            sys.stderr.write(f"[FaceTrackingThread] Modo desconocido: {self.modo}\n")
            self._cerrar_salida()
            self._running = False
            return

        # Bucle principal
        try:
            if self.modo == "compartido":
                ct = self.capture_thread
                last_ts = 0.0
                while self._running:
                    if self.duracion is not None and (time.perf_counter() - self._t_inicio) >= self.duracion:
                        break
                    try:
                        frame, ts = ct.get_latest_frame(copy=True)
                    except Exception:
                        frame, ts = None, 0.0
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    if ts == last_ts or ts == 0.0:
                        # Sin frame nuevo: ceder CPU
                        time.sleep(0.005)
                        continue
                    last_ts = ts
                    t_perf = float(ts) if ts and ts > 0 else time.perf_counter()
                    self._procesar_frame(frame, t_perf)

            elif self.modo == "propio":
                cap = self._cap
                consecutive_failures = 0
                while self._running:
                    if self.duracion is not None and (time.perf_counter() - self._t_inicio) >= self.duracion:
                        break
                    ret, frame = cap.read()
                    t_perf = time.perf_counter()
                    if not ret or frame is None:
                        consecutive_failures += 1
                        if consecutive_failures > 30:
                            break
                        time.sleep(0.015)
                        continue
                    consecutive_failures = 0
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))
                    self._procesar_frame(frame, t_perf)

            elif self.modo == "archivo":
                cap = self._cap
                fps_v = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if fps_v <= 0:
                    fps_v = 30.0
                max_frames = (
                    int(self.duracion * fps_v) if self.duracion is not None else None
                )
                while self._running:
                    if max_frames is not None and self._frame_idx >= max_frames:
                        break
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    t_perf = time.perf_counter()
                    if frame.shape[1] != 640 or frame.shape[0] != 480:
                        frame = cv2.resize(frame, (640, 480))
                    self._procesar_frame(frame, t_perf)
        except Exception as exc:
            sys.stderr.write(f"[FaceTrackingThread] Error en bucle principal: {exc}\n")
        finally:
            # Cierre limpio
            self._running = False
            try:
                if self._cap is not None:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                except Exception:
                    pass
                self._landmarker = None
            self._cerrar_salida()

    # ------------------------------------------------------------------ #
    # Alias de compatibilidad
    # ------------------------------------------------------------------ #
    def get_stats(self) -> Dict[str, Any]:
        return self.stats


def _main() -> int:
    parser = argparse.ArgumentParser(description="FaceTrackingThread standalone")
    parser.add_argument(
        "--fuente",
        required=True,
        help="cam:0 | /ruta/video.mp4 | compartido",
    )
    parser.add_argument("--duracion", type=float, default=10.0)
    parser.add_argument("--salida", default="/tmp/opencode")
    parser.add_argument("--landmarks", action="store_true")
    args = parser.parse_args()

    thread = FaceTrackingThread(
        fuente=args.fuente,
        duracion=args.duracion,
        salida=args.salida,
        landmarks=args.landmarks,
    )
    thread.start()
    thread.wait()

    st = thread.stats
    print("[tracking_facial] CSV:", st.get("csv_path"))
    if st.get("landmarks_path"):
        print("[tracking_facial] Landmarks:", st.get("landmarks_path"))
    print(
        "[tracking_facial] frames={frames} fps_medio={fps:.2f} deteccion={det:.1f}%".format(
            frames=st.get("frames", 0),
            fps=st.get("fps_medio", 0.0),
            det=st.get("deteccion_pct", 0.0),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except KeyboardInterrupt:
        raise SystemExit(0)