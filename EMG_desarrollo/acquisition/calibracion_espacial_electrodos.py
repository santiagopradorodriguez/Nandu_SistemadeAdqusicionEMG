# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo de calibración geométrica y triangulación espacial 3D de electrodos sEMG faciales mediante MediaPipe Face Mesh y OpenCV.
# ==============================================================================

"""
Módulo de Calibración Espacial 3D y Triangulación Geométrica de Electrodos sEMG Faciales.

Este módulo implementa el paso de calibración geométrica previo a la adquisición de
señales electromiográficas. Utiliza visión por computadora para detectar puntos fiduciarios
faciales tridimensionales (MediaPipe Face Landmarker / Face Mesh) y segmentar marcadores
de electrodos (OpenCV en espacio HSV o detección por correlación de broches metálicos).
Proyecta los centroides 2D sobre la variedad facial 3D, calcula las distancias euclidianas
hacia la punta de la nariz (origen) y normaliza las métricas dividiendo por la distancia
interpupilar para lograr invarianza frente a la escala y distancia de la cámara.

Desglose de Variables y Formulación Matemática:
-----------------------------------------------
1. Distancia Interpupilar Tridimensional (Base de Normalización):
   Sean $\\mathbf{P}_{\\text{ojo, izq}} = (X_{\\text{izq}}, Y_{\\text{izq}}, Z_{\\text{izq}})$ y
   $\\mathbf{P}_{\\text{ojo, der}} = (X_{\\text{der}}, Y_{\\text{der}}, Z_{\\text{der}})$ las
   coordenadas 3D de los centros oculares (o cantos de los ojos) extraídos por la malla facial.
   La distancia interpupilar $D_{\\text{interpupilar}}$ se define como:
   $$D_{\\text{interpupilar}} = \\|\\mathbf{P}_{\\text{ojo, der}} - \\mathbf{P}_{\\text{ojo, izq}}\\|_2 = \\sqrt{(X_{\\text{der}} - X_{\\text{izq}})^2 + (Y_{\\text{der}} - Y_{\\text{izq}})^2 + (Z_{\\text{der}} - Z_{\\text{izq}})^2}$$
   Unidades: Coordenadas adimensionales normalizadas de la malla MediaPipe.

2. Centroide 2D de Marcadores en Espacio de Imagen:
   A partir de la máscara binaria en el espacio de color HSV, para cada contorno de electrodo
   se computan los momentos espaciales de orden cero ($M_{00}$) y primer orden ($M_{10}, M_{01}$):
   $$c_x = \\frac{M_{10}}{M_{00}}, \\quad c_y = \\frac{M_{01}}{M_{00}}$$
   Unidades: Píxeles en el sistema coordenado de la imagen ($c_x \\in [0, W]$, $c_y \\in [0, H]$).

3. Asociación con el Vértice 3D Más Cercano:
   Cada vértice $j$ de la malla facial posee coordenadas proyectadas $(u_j, v_j) = (x_j \\cdot W, y_j \\cdot H)$.
   El vértice fiduciario asociado al electrodo $v^*$ es aquel que minimiza la distancia euclidiana en el plano de proyección:
   $$v^* = \\arg\\min_{j \\in \\{0, \\dots, N-1\\}} \\sqrt{(u_j - c_x)^2 + (v_j - c_y)^2}$$

4. Distancia Euclidiana Tridimensional al Origen Nasal:
   Fijando la punta de la nariz (Landmark 1, pronasale) como origen $\\mathbf{P}_{\\text{nariz}} = (X_{\\text{nariz}}, Y_{\\text{nariz}}, Z_{\\text{nariz}})$,
   el vector relativo de posición tridimensional es:
   $$\\Delta \\mathbf{P} = \\mathbf{P}_{v^*} - \\mathbf{P}_{\\text{nariz}} = \\begin{bmatrix} X_{v^*} - X_{\\text{nariz}} \\\\ Y_{v^*} - Y_{\\text{nariz}} \\\\ Z_{v^*} - Z_{\\text{nariz}} \\end{bmatrix}$$
   La distancia euclidiana escalar $d_{3\\text{D}}$ al origen es:
   $$d_{3\\text{D}} = \\|\\Delta \\mathbf{P}\\|_2 = \\sqrt{(X_{v^*} - X_{\\text{nariz}})^2 + (Y_{v^*} - Y_{\\text{nariz}})^2 + (Z_{v^*} - Z_{\\text{nariz}})^2}$$

5. Métrica Normalizada e Invariante:
   Dividiendo por la distancia interpupilar se elimina la dependencia del factor de aumento de la lente o zoom:
   $$\\tilde{d} = \\frac{d_{3\\text{D}}}{D_{\\text{interpupilar}}}$$
   $$\\tilde{\\mathbf{P}} = \\frac{\\Delta \\mathbf{P}}{D_{\\text{interpupilar}}}$$
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None


# ==============================================================================
# CONSTANTES FIDUCIARIAS DE MEDIAPIPE FACE MESH
# ==============================================================================
# Índice de la punta de la nariz (pronasale): origen del sistema de referencia facial
INDICE_NARIZ_PUNTA: int = 1

# Índices para cantos de los ojos (fijación de distancia interocular/interpupilar)
INDICE_CANTO_EXTERNO_OJO_IZQ: int = 33
INDICE_CANTO_INTERNO_OJO_IZQ: int = 133
INDICE_CANTO_INTERNO_OJO_DER: int = 362
INDICE_CANTO_EXTERNO_OJO_DER: int = 263

# Índices refinados de iris/pupila (MediaPipe refine_landmarks=True)
INDICE_IRIS_CENTRO_IZQ: int = 468
INDICE_IRIS_CENTRO_DER: int = 473

# Índices anatómicos adicionales para pose craneal 3D y biometría labial/mandibular
INDICE_GLABELA: int = 10
INDICE_MENTON_GNATHION: int = 152
INDICE_LABIO_SUP_INTERNO: int = 13
INDICE_LABIO_INF_INTERNO: int = 14
INDICE_COMISURA_IZQ: int = 61
INDICE_COMISURA_DER: int = 291
INDICE_POMULO_IZQ: int = 234
INDICE_POMULO_DER: int = 454

# Índices anatómicos de orejas (región preauricular, trago, raíz y base de lóbulo)
INDICE_TRAGO_OREJA_IZQ: int = 234  # Punto anterior al trago de la oreja izquierda
INDICE_TRAGO_OREJA_DER: int = 454  # Punto anterior al trago de la oreja derecha
INDICE_RAIZ_OREJA_IZQ: int = 127   # Inserción superior oreja izquierda
INDICE_RAIZ_OREJA_DER: int = 356   # Inserción superior oreja derecha
INDICE_LOBULO_OREJA_IZQ_BASE: int = 137 # Base mandibular anterior al lóbulo izquierdo
INDICE_LOBULO_OREJA_DER_BASE: int = 366 # Base mandibular anterior al lóbulo derecho


@dataclass
class ElectrodoEspacial:
    """Representación estructurada de un electrodo sEMG físico individual en la malla facial 3D."""

    id: int
    canal: str
    polo: str = "monopolar"  # "positivo", "negativo", "monopolar", "tierra_gnd"
    musculo_estimado: str = "No especificado"
    coordenadas_2d_px: List[float] = field(default_factory=list)
    vertice_malla_3d: int = 0
    error_reproyeccion_2d_px: float = 0.0
    coordenadas_3d_absolutas: List[float] = field(default_factory=list)
    vector_3d_relativo_normalizado: List[float] = field(default_factory=list)
    distancia_3d_normalizada: float = 0.0
    distancia_3d_absoluta: float = 0.0
    canal_fisico_ai: Optional[int] = None
    musculo_asignado: Optional[str] = None
    vector_norm_dip: Optional[List[float]] = None
    id_par: Optional[int] = None
    centroide_2d_px: Optional[List[float]] = None

    def __post_init__(self) -> None:
        if not self.coordenadas_2d_px and self.centroide_2d_px:
            self.coordenadas_2d_px = list(self.centroide_2d_px)
        elif self.coordenadas_2d_px and not self.centroide_2d_px:
            self.centroide_2d_px = list(self.coordenadas_2d_px)

    def to_dict(self) -> Dict[str, Any]:
        datos = asdict(self)
        if datos.get("vector_norm_dip") is None:
            datos["vector_norm_dip"] = list(self.vector_3d_relativo_normalizado)
        datos["centroide_2d_px"] = list(self.coordenadas_2d_px)
        return datos


@dataclass
class ParDiferencialEspacial:
    """Representación de un canal diferencial sEMG bipolar compuesto por dos electrodos físicos (+ y -)."""

    id: int
    canal_fisico_ai: int
    etiqueta_canal: str
    musculo: str
    electrodo_positivo: ElectrodoEspacial
    electrodo_negativo: ElectrodoEspacial
    vector_dipolo_3d: List[float]
    vector_dipolo_normalizado: List[float]
    distancia_interelectrodo_3d: float
    distancia_interelectrodo_px: float
    distancia_interelectrodo_mm: Optional[float]
    angulo_orientacion_deg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "canal_fisico_ai": self.canal_fisico_ai,
            "etiqueta_canal": self.etiqueta_canal,
            "musculo": self.musculo,
            "polo_positivo": self.electrodo_positivo.to_dict(),
            "polo_negativo": self.electrodo_negativo.to_dict(),
            "vector_dipolo_3d": [round(float(v), 6) for v in self.vector_dipolo_3d],
            "vector_dipolo_normalizado": [round(float(v), 6) for v in self.vector_dipolo_normalizado],
            "distancia_interelectrodo_3d": round(float(self.distancia_interelectrodo_3d), 6),
            "distancia_interelectrodo_px": round(float(self.distancia_interelectrodo_px), 2),
            "distancia_interelectrodo_mm": round(float(self.distancia_interelectrodo_mm), 2) if self.distancia_interelectrodo_mm is not None else None,
            "angulo_orientacion_deg": round(float(self.angulo_orientacion_deg), 2),
        }


class AntigravityAgent:
    """
    Agente de triangulación espacial 3D y calibración geométrica de electrodos sEMG.

    Automatiza la extracción de puntos fiduciarios anatómicos mediante MediaPipe,
    la segmentación cromática de marcadores en espacio HSV y la normalización métrica
    invariante a la distancia focal y posición de la cámara.
    """

    def __init__(
        self,
        hsv_lower: Optional[np.ndarray] = None,
        hsv_upper: Optional[np.ndarray] = None,
        min_contour_area: float = 30.0,
        max_contour_area: float = 5000.0,
        refine_landmarks: bool = True,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_asset_path: Optional[str] = None,
    ) -> None:
        """
        Inicializa el agente de triangulación espacial.

        Args:
            hsv_lower: Límite inferior del espacio HSV para marcadores de electrodos [H, S, V].
                       Por defecto se calibra para verde flúor: [35, 70, 70].
            hsv_upper: Límite superior del espacio HSV para marcadores de electrodos [H, S, V].
                       Por defecto se calibra para verde flúor: [88, 255, 255].
            min_contour_area: Área mínima en píxeles para aceptar un contorno de marcador.
            max_contour_area: Área máxima en píxeles para descartar agrupamientos masivos.
            refine_landmarks: Habilita el modelo refinado con 478 puntos (incluye iris).
            max_num_faces: Cantidad máxima de rostros a rastrear en la toma (típicamente 1).
            min_detection_confidence: Umbral de confianza del detector facial.
            min_tracking_confidence: Umbral de confianza de seguimiento entre cuadros.
            model_asset_path: Ruta al archivo face_landmarker.task para mediapipe moderno.
        """
        if hsv_lower is None:
            self.hsv_lower: np.ndarray = np.array([35, 70, 70], dtype=np.uint8)
        else:
            self.hsv_lower = np.array(hsv_lower, dtype=np.uint8)

        if hsv_upper is None:
            self.hsv_upper: np.ndarray = np.array([88, 255, 255], dtype=np.uint8)
        else:
            self.hsv_upper = np.array(hsv_upper, dtype=np.uint8)

        self.min_contour_area: float = float(min_contour_area)
        self.max_contour_area: float = float(max_contour_area)
        self.refine_landmarks: bool = bool(refine_landmarks)

        # Rutas candidatas para el modelo face_landmarker.task
        rutas_candidatas = [
            model_asset_path,
            os.path.join(getattr(sys, '_MEIPASS', ''), "DataConfig", "modelos_vision", "face_landmarker.task") if hasattr(sys, '_MEIPASS') else None,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "DataConfig", "modelos_vision", "face_landmarker.task"
                )
            ),
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), "..", "..", "EMG_desarrollo", "DataConfig", "modelos_vision", "face_landmarker.task"
                )
            ),
            "/home/santiago/Documentos/codigos/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task",
            "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task",
        ]

        ruta_modelo_valida = None
        for r in rutas_candidatas:
            if r and os.path.exists(r):
                ruta_modelo_valida = r
                break

        self.modo_moderno = False
        self._modo_api: Optional[str] = None
        self.detector = None

        if ruta_modelo_valida:
            try:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                base_options = python.BaseOptions(model_asset_path=ruta_modelo_valida)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=True,
                    num_faces=max_num_faces,
                    min_face_detection_confidence=min_detection_confidence,
                    min_face_presence_confidence=min_tracking_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                )
                self.detector = vision.FaceLandmarker.create_from_options(options)
                self.modo_moderno = True
                self._modo_api = "tasks"
            except Exception as e:
                sys.stderr.write(f"Aviso: No se pudo inicializar MediaPipe Tasks: {e}. Intentando solutions...\n")

        if not self.modo_moderno and hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=self.refine_landmarks,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._modo_api = "solutions"
        elif not self.modo_moderno:
            sys.stderr.write("Aviso: MediaPipe Face Mesh no se pudo cargar completamente.\n")

    def segmentar_marcadores_hsv(
        self,
        bgr_image: np.ndarray,
        lower_override: Optional[np.ndarray] = None,
        upper_override: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Segmenta marcadores físicos de electrodos en el espacio de color HSV.

        Aplica filtrado cromático, operaciones morfológicas de apertura y clausura,
        y calcula los centroides 2D a través de los momentos espaciales de los contornos.

        Args:
            bgr_image: Imagen de entrada en formato BGR (OpenCV estándar).
            lower_override: Rango inferior HSV alternativo opcional.
            upper_override: Rango superior HSV alternativo opcional.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]:
                - Máscara binaria resultante de la segmentación.
                - Lista de diccionarios con información geométrica de cada marcador.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        lower = self.hsv_lower if lower_override is None else lower_override
        upper = self.hsv_upper if upper_override is None else upper_override

        # Máscara cromática primaria
        mask = cv2.inRange(hsv, lower, upper)

        # Filtrado morfológico para eliminar ruido y compactar marcadores
        kernel_apertura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_clausura = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_apertura, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_clausura, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        marcadores_detectados: List[Dict[str, Any]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_contour_area <= area <= self.max_contour_area:
                moments = cv2.moments(cnt)
                if moments["m00"] > 0:
                    cx = float(moments["m10"] / moments["m00"])
                    cy = float(moments["m01"] / moments["m00"])
                    marcadores_detectados.append({
                        "centroide": (cx, cy),
                        "area": area,
                        "contorno": cnt,
                    })

        marcadores_detectados.sort(key=lambda item: item["centroide"][1])
        return mask, marcadores_detectados

    def detectar_broches_metalicos(
        self,
        bgr_image: np.ndarray,
        min_radius: int = 5,
        max_radius: int = 40,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Detecta broches de presión (snaps metálicos) mediante contraste de gradiente circular.

        Args:
            bgr_image: Imagen de entrada BGR.
            min_radius: Radio mínimo en píxeles.
            max_radius: Radio máximo en píxeles.

        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Máscara y lista de broches detectados.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 1.5)

        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=15,
            param1=50,
            param2=28,
            minRadius=min_radius,
            maxRadius=max_radius,
        )

        marcadores: List[Dict[str, Any]] = []
        mask_vis = np.zeros_like(gray)

        if circles is not None:
            circles_rounded = np.uint16(np.around(circles))
            for pt in circles_rounded[0, :]:
                cx, cy, r = float(pt[0]), float(pt[1]), float(pt[2])
                cv2.circle(mask_vis, (int(cx), int(cy)), int(r), 255, -1)
                marcadores.append({
                    "centroide": (cx, cy),
                    "area": float(np.pi * (r ** 2)),
                    "radio": r,
                })

        marcadores.sort(key=lambda item: item["centroide"][1])
        return mask_vis, marcadores

    def extraer_malla_facial_3d(
        self,
        bgr_image: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        Extrae los puntos fiduciarios 3D de la malla facial mediante MediaPipe.

        Retorna un diccionario con puntos 3D normalizados, coordenadas en píxeles,
        la punta nasal (origen), la distancia interpupilar base de escala y, si está
        disponible en MediaPipe Tasks, la matriz de transformación 4x4.
        """
        if bgr_image is None or bgr_image.size == 0:
            return None

        alto, ancho, _ = bgr_image.shape
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        landmarks_extraidos = None
        transformation_matrix: Optional[np.ndarray] = None

        if (self.modo_moderno or self._modo_api == "tasks") and self.detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            resultado_det = self.detector.detect(mp_image)
            if resultado_det.face_landmarks and len(resultado_det.face_landmarks) > 0:
                landmarks_extraidos = resultado_det.face_landmarks[0]
            if (
                self._modo_api == "tasks"
                and getattr(resultado_det, "facial_transformation_matrixes", None)
                and len(resultado_det.facial_transformation_matrixes) > 0
            ):
                transformation_matrix = np.array(
                    resultado_det.facial_transformation_matrixes[0], dtype=np.float64
                )
        elif hasattr(self, "face_mesh"):
            resultados_legacy = self.face_mesh.process(rgb_image)
            if resultados_legacy.multi_face_landmarks:
                landmarks_extraidos = resultados_legacy.multi_face_landmarks[0].landmark

        if landmarks_extraidos is None:
            return None

        num_puntos = len(landmarks_extraidos)
        puntos_3d = np.zeros((num_puntos, 3), dtype=np.float64)
        puntos_2d_px = np.zeros((num_puntos, 2), dtype=np.float64)

        for i, lm in enumerate(landmarks_extraidos):
            puntos_3d[i, 0] = lm.x
            puntos_3d[i, 1] = lm.y
            puntos_3d[i, 2] = lm.z
            puntos_2d_px[i, 0] = lm.x * ancho
            puntos_2d_px[i, 1] = lm.y * alto

        # Origen de coordenadas: punta de la nariz (pronasale)
        punta_nariz_3d = puntos_3d[INDICE_NARIZ_PUNTA].copy()
        punta_nariz_px = puntos_2d_px[INDICE_NARIZ_PUNTA].copy()

        # Distancia interpupilar / interocular
        if num_puntos >= 478 and self.refine_landmarks:
            ojo_izq_3d = puntos_3d[INDICE_IRIS_CENTRO_IZQ]
            ojo_der_3d = puntos_3d[INDICE_IRIS_CENTRO_DER]
            ojo_izq_px = puntos_2d_px[INDICE_IRIS_CENTRO_IZQ]
            ojo_der_px = puntos_2d_px[INDICE_IRIS_CENTRO_DER]
            metodo_ojos = "centro_iris_refinado"
        else:
            ojo_izq_3d = (puntos_3d[INDICE_CANTO_EXTERNO_OJO_IZQ] + puntos_3d[INDICE_CANTO_INTERNO_OJO_IZQ]) / 2.0
            ojo_der_3d = (puntos_3d[INDICE_CANTO_EXTERNO_OJO_DER] + puntos_3d[INDICE_CANTO_INTERNO_OJO_DER]) / 2.0
            ojo_izq_px = (puntos_2d_px[INDICE_CANTO_EXTERNO_OJO_IZQ] + puntos_2d_px[INDICE_CANTO_INTERNO_OJO_IZQ]) / 2.0
            ojo_der_px = (puntos_2d_px[INDICE_CANTO_EXTERNO_OJO_DER] + puntos_2d_px[INDICE_CANTO_INTERNO_OJO_DER]) / 2.0
            metodo_ojos = "cantos_oculares_punto_medio"

        dist_ip_3d = float(np.linalg.norm(ojo_der_3d - ojo_izq_3d))
        dist_ip_px = float(np.linalg.norm(ojo_der_px - ojo_izq_px))

        if dist_ip_3d < 1e-6:
            dist_ip_3d = 1e-6

        # Puntos fiduciarios de orejas (trago, inserción superior y estimación de lóbulo)
        trago_izq_3d = puntos_3d[INDICE_TRAGO_OREJA_IZQ]
        trago_der_3d = puntos_3d[INDICE_TRAGO_OREJA_DER]
        trago_izq_px = puntos_2d_px[INDICE_TRAGO_OREJA_IZQ]
        trago_der_px = puntos_2d_px[INDICE_TRAGO_OREJA_DER]
        raiz_izq_px = puntos_2d_px[INDICE_RAIZ_OREJA_IZQ]
        raiz_der_px = puntos_2d_px[INDICE_RAIZ_OREJA_DER]
        lobulo_base_izq_px = puntos_2d_px[INDICE_LOBULO_OREJA_IZQ_BASE]
        lobulo_base_der_px = puntos_2d_px[INDICE_LOBULO_OREJA_DER_BASE]

        # Desplazamiento anatómico hacia el lóbulo auricular (inferior-lateral desde la base mandibular)
        vec_nariz_izq = trago_izq_px - punta_nariz_px
        vec_nariz_der = trago_der_px - punta_nariz_px
        dir_izq = vec_nariz_izq / max(float(np.linalg.norm(vec_nariz_izq)), 1e-4)
        dir_der = vec_nariz_der / max(float(np.linalg.norm(vec_nariz_der)), 1e-4)

        lobulo_izq_est_px = lobulo_base_izq_px + dir_izq * (0.22 * dist_ip_px)
        lobulo_der_est_px = lobulo_base_der_px + dir_der * (0.22 * dist_ip_px)

        ancho_biauricular_px = float(np.linalg.norm(trago_der_px - trago_izq_px))
        ancho_biauricular_3d = float(np.linalg.norm(trago_der_3d - trago_izq_3d))

        return {
            "puntos_3d": puntos_3d,
            "puntos_2d_px": puntos_2d_px,
            "punta_nariz_3d": punta_nariz_3d,
            "punta_nariz_px": punta_nariz_px,
            "distancia_interpupilar_3d": dist_ip_3d,
            "distancia_interpupilar_px": dist_ip_px,
            "ojo_izq_3d": ojo_izq_3d,
            "ojo_der_3d": ojo_der_3d,
            "ojo_izq_px": ojo_izq_px,
            "ojo_der_px": ojo_der_px,
            "metodo_ojos": metodo_ojos,
            "oreja_trago_izq_3d": trago_izq_3d,
            "oreja_trago_der_3d": trago_der_3d,
            "oreja_trago_izq_px": trago_izq_px,
            "oreja_trago_der_px": trago_der_px,
            "oreja_raiz_izq_px": raiz_izq_px,
            "oreja_raiz_der_px": raiz_der_px,
            "oreja_lobulo_izq_px": lobulo_izq_est_px,
            "oreja_lobulo_der_px": lobulo_der_est_px,
            "ancho_biauricular_px": ancho_biauricular_px,
            "ancho_biauricular_3d": ancho_biauricular_3d,
            "dimensiones_imagen": (ancho, alto),
            "transformation_matrix": transformation_matrix,
        }

    def calcular_pose_y_biometria_facial(
        self,
        fiduciarios: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
        d_ip_real_cm: float = 6.3,
        distancia_fija_camara_cm: Optional[float] = None,
        escala_mm_por_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calcula los ángulos de Euler de pose craneal 3D (yaw, pitch, roll), la distancia
        estimada o calibrada a la cámara en centímetros y las métricas biométricas normalizadas
        de articulación labial, mandibular y auricular (invariantes a la escala mediante D_IP).

        Args:
            fiduciarios: Diccionario retornado por `extraer_malla_facial_3d`.
            image_shape: Tupla opcional con dimensiones de imagen (alto, ancho) o (ancho, alto).
            d_ip_real_cm: Distancia interpupilar real del sujeto en cm (default 6.3 cm estándar).
            distancia_fija_camara_cm: Distancia física medida rostro-cámara (e.g. en trípode fijo).
            escala_mm_por_px: Escala calibrada en mm/px (e.g. obtenida de distancia inter-electrodo conocida).

        Returns:
            Dict[str, Any]: Diccionario con ángulos en grados, distancia en cm y biometría normalizada.
        """
        default_metrics: Dict[str, Any] = {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 0.0,
            "distancia_camara_cm": 0.0,
            "metodo_distancia": "no_disponible",
            "apertura_labial_norm": 0.0,
            "ancho_comisuras_norm": 0.0,
            "apertura_mandibular_norm": 0.0,
            "protrusion_labial_rel": 0.0,
            "ancho_biauricular_norm": 0.0,
            "ancho_biauricular_cm": 0.0,
        }
        if not fiduciarios or "puntos_3d" not in fiduciarios or "puntos_2d_px" not in fiduciarios:
            return default_metrics

        puntos_3d: np.ndarray = fiduciarios["puntos_3d"]
        puntos_2d_px: np.ndarray = fiduciarios["puntos_2d_px"]
        num_pts = puntos_3d.shape[0]
        if num_pts <= max(INDICE_POMULO_DER, INDICE_CANTO_EXTERNO_OJO_DER):
            return default_metrics

        # Determinar el ancho W de la imagen en píxeles
        ancho_w = 640.0
        if "dimensiones_imagen" in fiduciarios and fiduciarios["dimensiones_imagen"]:
            ancho_w = float(fiduciarios["dimensiones_imagen"][0])
        elif image_shape is not None and len(image_shape) >= 2:
            # En OpenCV image.shape[:2] es (H, W); tomamos el mayor o el segundo elemento si es estándar
            ancho_w = float(image_shape[1] if image_shape[1] >= image_shape[0] else image_shape[0])

        d_ip_3d = max(float(fiduciarios.get("distancia_interpupilar_3d", 1e-6)), 1e-6)
        d_ip_px = max(float(fiduciarios.get("distancia_interpupilar_px", 1.0)), 1.0)

        # 1. Respaldo geométrico sobre los 478 puntos de la malla facial
        # Roll geométrico (ladeo ocular en el plano de proyección 2D)
        ojo_izq_px = fiduciarios.get("ojo_izq_px", puntos_2d_px[INDICE_CANTO_EXTERNO_OJO_IZQ])
        ojo_der_px = fiduciarios.get("ojo_der_px", puntos_2d_px[INDICE_CANTO_EXTERNO_OJO_DER])
        vec_ojos_px = np.asarray(ojo_der_px, dtype=np.float64) - np.asarray(ojo_izq_px, dtype=np.float64)
        roll_geom = float(np.degrees(np.arctan2(vec_ojos_px[1], max(abs(vec_ojos_px[0]), 1e-6))))

        # Yaw geométrico (giro horizontal respecto a cantos oculares 33/263 y pómulos 234/454)
        nariz_3d = puntos_3d[INDICE_NARIZ_PUNTA]
        canto_izq_3d = puntos_3d[INDICE_CANTO_EXTERNO_OJO_IZQ]
        canto_der_3d = puntos_3d[INDICE_CANTO_EXTERNO_OJO_DER]
        pomulo_izq_3d = puntos_3d[INDICE_POMULO_IZQ]
        pomulo_der_3d = puntos_3d[INDICE_POMULO_DER]

        centro_cantos = 0.5 * (canto_izq_3d + canto_der_3d)
        centro_pomulos = 0.5 * (pomulo_izq_3d + pomulo_der_3d)
        span_cantos = max(float(np.linalg.norm(canto_der_3d - canto_izq_3d)), 1e-6)
        span_pomulos = max(float(np.linalg.norm(pomulo_der_3d - pomulo_izq_3d)), 1e-6)

        dx_cantos = float(nariz_3d[0] - centro_cantos[0])
        dx_pomulos = float(nariz_3d[0] - centro_pomulos[0])
        profundidad_ref = max(
            abs(float(nariz_3d[2] - centro_pomulos[2])),
            0.45 * span_pomulos,
            0.65 * span_cantos,
            1e-6,
        )
        asimetria_horiz = 0.65 * dx_pomulos + 0.35 * dx_cantos
        yaw_geom = float(np.degrees(np.arctan2(asimetria_horiz, 0.5 * profundidad_ref)))

        # Pitch geométrico (inclinación vertical de punta nasal respecto a línea interocular, glabela 10 y mentón 152)
        glabela_3d = puntos_3d[INDICE_GLABELA]
        menton_3d = puntos_3d[INDICE_MENTON_GNATHION]
        ojo_medio_3d = 0.5 * (
            np.asarray(fiduciarios.get("ojo_izq_3d", canto_izq_3d), dtype=np.float64)
            + np.asarray(fiduciarios.get("ojo_der_3d", canto_der_3d), dtype=np.float64)
        )
        alto_rostro = max(float(menton_3d[1] - glabela_3d[1]), 1e-6)
        span_ojo_menton = max(float(menton_3d[1] - ojo_medio_3d[1]), 1e-6)
        # En postura neutral, la punta de la nariz se ubica a ~0.365 del segmento ojo_medio -> mentón
        ratio_nariz_vert = (float(nariz_3d[1] - ojo_medio_3d[1]) / span_ojo_menton) - 0.365
        inclinacion_z = float(menton_3d[2] - glabela_3d[2]) / alto_rostro
        pitch_geom = float(np.degrees(np.arctan2(0.7 * ratio_nariz_vert + 0.3 * inclinacion_z, 0.45)))

        # 2. Pose craneal usando facial_transformation_matrixes de MediaPipe (con respaldo geométrico)
        metodo_pose = "respaldo_geometrico_478pts"
        yaw_deg = yaw_geom
        pitch_deg = pitch_geom
        roll_deg = roll_geom

        t_mat = fiduciarios.get("transformation_matrix")
        if t_mat is not None:
            try:
                mat_arr = np.asarray(t_mat, dtype=np.float64)
                if mat_arr.size == 16:
                    mat_arr = mat_arr.reshape((4, 4))
                if mat_arr.shape in ((4, 4), (3, 3)):
                    R = mat_arr[:3, :3]
                    det_r = float(np.linalg.det(R))
                    if abs(det_r) > 0.05 and not np.isnan(R).any():
                        if det_r < 0:
                            R = R.copy()
                            R[:, 0] *= -1.0
                        angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
                        p_mat = float(angles[0])
                        y_mat = float(angles[1])
                        r_mat = float(angles[2])
                        if not (np.isnan(p_mat) or np.isnan(y_mat) or np.isnan(r_mat)):
                            pitch_deg = p_mat
                            yaw_deg = y_mat
                            roll_deg = r_mat
                            metodo_pose = "facial_transformation_matrix"
            except Exception:
                yaw_deg = yaw_geom
                pitch_deg = pitch_geom
                roll_deg = roll_geom
                metodo_pose = "respaldo_geometrico_478pts"

        # 3. Distancia estimada o calibrada a la cámara en cm
        d_ip_efectiva_cm = float(d_ip_real_cm)
        if escala_mm_por_px is not None and escala_mm_por_px > 0:
            d_ip_efectiva_cm = float((d_ip_px * escala_mm_por_px) / 10.0)

        f_px = 0.85 * ancho_w
        if distancia_fija_camara_cm is not None and distancia_fija_camara_cm > 0:
            distancia_camara_cm = float(distancia_fija_camara_cm)
            metodo_distancia = "distancia_fija_referencia"
        elif escala_mm_por_px is not None and escala_mm_por_px > 0:
            distancia_camara_cm = float(np.clip((d_ip_efectiva_cm * f_px) / d_ip_px, 5.0, 300.0))
            metodo_distancia = "escala_ied_calibrada"
        elif abs(d_ip_real_cm - 6.3) > 1e-4:
            distancia_camara_cm = float(np.clip((d_ip_efectiva_cm * f_px) / d_ip_px, 5.0, 300.0))
            metodo_distancia = "d_ip_usuario_personalizada"
        else:
            distancia_camara_cm = float(np.clip((6.3 * f_px) / d_ip_px, 5.0, 300.0))
            metodo_distancia = "antropometria_estandar_6.3cm"

        # 4. Biometría labial, mandibular y biauricular normalizada por D_IP 3D
        labio_sup_3d = puntos_3d[INDICE_LABIO_SUP_INTERNO]
        labio_inf_3d = puntos_3d[INDICE_LABIO_INF_INTERNO]
        comisura_izq_3d = puntos_3d[INDICE_COMISURA_IZQ]
        comisura_der_3d = puntos_3d[INDICE_COMISURA_DER]

        apertura_labial_norm = float(np.linalg.norm(labio_sup_3d - labio_inf_3d) / d_ip_3d)
        ancho_comisuras_norm = float(np.linalg.norm(comisura_der_3d - comisura_izq_3d) / d_ip_3d)
        apertura_mandibular_norm = float(np.linalg.norm(menton_3d - nariz_3d) / d_ip_3d)

        centro_labios_3d = 0.5 * (labio_sup_3d + labio_inf_3d)
        centro_comisuras_3d = 0.5 * (comisura_izq_3d + comisura_der_3d)
        # En MediaPipe Z negativo apunta hacia la cámara; comisuras_z - labios_z > 0 indica protrusión frontal
        protrusion_labial_rel = float((centro_comisuras_3d[2] - centro_labios_3d[2]) / d_ip_3d)
        ratio_aspecto_labial = float(apertura_labial_norm / max(ancho_comisuras_norm, 1e-4))

        # 5. Métricas de orejas (ancho biauricular)
        ancho_biauricular_px = float(fiduciarios.get("ancho_biauricular_px", 0.0))
        ancho_biauricular_3d = float(fiduciarios.get("ancho_biauricular_3d", 0.0))
        ancho_biauricular_norm = float(ancho_biauricular_3d / d_ip_3d) if d_ip_3d > 1e-6 else 0.0
        ancho_biauricular_cm = float((ancho_biauricular_px / d_ip_px) * d_ip_efectiva_cm) if d_ip_px > 1e-6 else 0.0

        return {
            "yaw_deg": round(yaw_deg, 4),
            "pitch_deg": round(pitch_deg, 4),
            "roll_deg": round(roll_deg, 4),
            "yaw_geom_deg": round(yaw_geom, 4),
            "pitch_geom_deg": round(pitch_geom, 4),
            "roll_geom_deg": round(roll_geom, 4),
            "metodo_pose": metodo_pose,
            "distancia_camara_cm": round(distancia_camara_cm, 4),
            "metodo_distancia": metodo_distancia,
            "d_ip_referencia_cm": round(d_ip_efectiva_cm, 4),
            "apertura_labial_norm": round(apertura_labial_norm, 4),
            "ancho_comisuras_norm": round(ancho_comisuras_norm, 4),
            "apertura_mandibular_norm": round(apertura_mandibular_norm, 4),
            "protrusion_labial_rel": round(protrusion_labial_rel, 4),
            "ratio_aspecto_labial": round(ratio_aspecto_labial, 4),
            "ancho_biauricular_norm": round(ancho_biauricular_norm, 4),
            "ancho_biauricular_cm": round(ancho_biauricular_cm, 2),
        }

    def proyectar_referencia_fantasma(
        self,
        fiduciarios: Dict[str, Any],
        datos_referencia: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Proyecta los vectores 3D normalizados de una sesión previa ('Ghost Overlay')
        sobre la malla facial actual utilizando:
            P_target_3d = punta_nariz_actual + vector_norm_dip * D_IP_actual
        """
        if not fiduciarios or not datos_referencia:
            return []

        meta = datos_referencia.get("metadatos_calibracion_espacial", datos_referencia)
        electrodos_ref = meta.get("electrodos", [])
        if not electrodos_ref:
            return []

        nariz_3d = np.asarray(fiduciarios["punta_nariz_3d"], dtype=np.float64)
        d_ip_3d = float(fiduciarios["distancia_interpupilar_3d"])
        puntos_3d = fiduciarios["puntos_3d"]
        puntos_2d_px = fiduciarios["puntos_2d_px"]
        ancho, alto = fiduciarios.get("dimensiones_imagen", (640, 480))

        proyecciones: List[Dict[str, Any]] = []
        for idx, el in enumerate(electrodos_ref):
            vec_norm = (
                el.get("vector_norm_dip")
                or el.get("vector_3d_relativo_normalizado")
                or el.get("vector_3d_normalizado")
            )
            if vec_norm is None or len(vec_norm) < 3:
                continue

            vec_arr = np.asarray(vec_norm[:3], dtype=np.float64)
            target_3d = nariz_3d + vec_arr * d_ip_3d

            # Coordenadas proyectadas directas y vértice actual más cercano en 3D
            dist_malla = np.linalg.norm(puntos_3d - target_3d, axis=1)
            v_idx = int(np.argmin(dist_malla))
            px_directo = (float(target_3d[0] * ancho), float(target_3d[1] * alto))
            px_vertice = (float(puntos_2d_px[v_idx, 0]), float(puntos_2d_px[v_idx, 1]))
            px_final = (
                0.5 * px_directo[0] + 0.5 * px_vertice[0],
                0.5 * px_directo[1] + 0.5 * px_vertice[1],
            )

            proyecciones.append({
                "id": el.get("id", idx),
                "canal": el.get("canal", f"Canal {idx}"),
                "musculo": el.get("musculo_asignado") or el.get("musculo_estimado") or el.get("musculo", ""),
                "target_2d_px": [round(px_final[0], 2), round(px_final[1], 2)],
                "vertice_malla_3d": v_idx,
                "vector_norm_dip": [round(float(v), 6) for v in vec_arr],
            })

        return proyecciones

    def triangular_electrodos(
        self,
        bgr_image: np.ndarray,
        puntos_centroides_manuales: Optional[List[Any]] = None,
        usar_broches_metalicos: bool = False,
        asignar_musculos_automatico: bool = True,
        nombres_canales: Optional[List[str]] = None,
        indices_canales_fisicos: Optional[List[int]] = None,
        d_ip_real_cm: float = 6.3,
        distancia_fija_camara_cm: Optional[float] = None,
        escala_mm_por_px: Optional[float] = None,
        punto_referencia_gnd: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecuta la triangulación geométrica completa de los electrodos sEMG.

        Asocia cada centroide 2D (monopolar o par diferencial bipolar) al vértice 3D
        de la malla facial más cercano, calcula el vector 3D relativo respecto a la
        punta nasal y normaliza la distancia euclidiana dividiendo por la distancia interpupilar.
        """
        alto, ancho, _ = bgr_image.shape
        fiduciarios = self.extraer_malla_facial_3d(bgr_image)

        if fiduciarios is None:
            return {
                "exito": False,
                "mensaje": "No se detecto ningun rostro en la imagen.",
                "electrodos": [],
                "metadatos_calibracion_espacial": {
                    "timestamp": datetime.now().isoformat(),
                    "dimensiones_imagen_px": [ancho, alto],
                    "total_electrodos_detectados": 0,
                    "electrodos": [],
                },
            }

        centroides_2d: List[Dict[str, Any]] = []
        if puntos_centroides_manuales is not None and len(puntos_centroides_manuales) > 0:
            for pt in puntos_centroides_manuales:
                if isinstance(pt, dict) and ("polo_positivo" in pt or "centroide" in pt):
                    c_xy = pt.get("centroide") or pt.get("polo_positivo")
                    centroides_2d.append({
                        "centroide": (float(c_xy[0]), float(c_xy[1])),
                        "area": float(pt.get("area", 0.0)),
                        "fuente": "manual",
                        "raw_info": pt,
                    })
                elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    centroides_2d.append({
                        "centroide": (float(pt[0]), float(pt[1])),
                        "area": 0.0,
                        "fuente": "manual",
                        "raw_info": {},
                    })
            mascara = np.zeros((alto, ancho), dtype=np.uint8)
        elif usar_broches_metalicos:
            mascara, centroides_2d = self.detectar_broches_metalicos(bgr_image)
        else:
            mascara, centroides_2d = self.segmentar_marcadores_hsv(bgr_image)

        puntos_3d = fiduciarios["puntos_3d"]
        puntos_2d_px = fiduciarios["puntos_2d_px"]
        nariz_3d = fiduciarios["punta_nariz_3d"]
        dist_ip_3d = fiduciarios["distancia_interpupilar_3d"]
        dist_ip_px = fiduciarios["distancia_interpupilar_px"]

        def _asociar_punto_a_malla(
            px: float,
            py: float,
            polo: str,
            canal: str,
            musculo: str,
            ch_num: Optional[int],
            par_id: Optional[int],
            elec_id: int,
        ) -> ElectrodoEspacial:
            distancias_2d = np.hypot(puntos_2d_px[:, 0] - px, puntos_2d_px[:, 1] - py)
            vertice_cercano = int(np.argmin(distancias_2d))
            error_reproyeccion_px = float(distancias_2d[vertice_cercano])

            vertice_3d = puntos_3d[vertice_cercano]
            delta_3d = vertice_3d - nariz_3d
            distancia_3d_absoluta = float(np.linalg.norm(delta_3d))
            distancia_3d_normalizada = float(distancia_3d_absoluta / dist_ip_3d) if dist_ip_3d > 0 else 0.0
            delta_3d_normalizado = (delta_3d / dist_ip_3d).tolist() if dist_ip_3d > 0 else [0.0, 0.0, 0.0]

            return ElectrodoEspacial(
                id=elec_id,
                canal=canal,
                polo=polo,
                musculo_estimado=musculo,
                coordenadas_2d_px=[round(float(px), 2), round(float(py), 2)],
                vertice_malla_3d=vertice_cercano,
                error_reproyeccion_2d_px=round(error_reproyeccion_px, 2),
                coordenadas_3d_absolutas=[round(float(v), 6) for v in vertice_3d],
                vector_3d_relativo_normalizado=[round(float(v), 6) for v in delta_3d_normalizado],
                distancia_3d_normalizada=round(distancia_3d_normalizada, 6),
                distancia_3d_absoluta=round(distancia_3d_absoluta, 6),
                canal_fisico_ai=ch_num,
                musculo_asignado=musculo,
                vector_norm_dip=[round(float(v), 6) for v in delta_3d_normalizado],
                id_par=par_id,
            )

        electrodos_objetos: List[ElectrodoEspacial] = []
        pares_diferenciales_objetos: List[ParDiferencialEspacial] = []
        canales_nombres_defecto = ["Canal 0", "Canal 1", "Canal 2", "Canal 3"]

        # Determinar si los datos de entrada provienen de pares diferenciales bipolares
        es_conjunto_bipolar = False
        if puntos_centroides_manuales and len(puntos_centroides_manuales) > 0:
            p0 = puntos_centroides_manuales[0]
            if isinstance(p0, dict) and (p0.get("es_bipolar") or ("polo_positivo" in p0 and "polo_negativo" in p0)):
                es_conjunto_bipolar = True

        if es_conjunto_bipolar:
            for k, pair_dict in enumerate(puntos_centroides_manuales):
                if not isinstance(pair_dict, dict):
                    continue
                pos_pt = pair_dict.get("polo_positivo")
                neg_pt = pair_dict.get("polo_negativo")
                if not pos_pt or not neg_pt:
                    continue

                ch_num: Optional[int] = None
                if indices_canales_fisicos is not None and k < len(indices_canales_fisicos):
                    ch_num = int(indices_canales_fisicos[k])
                elif "canal_fisico" in pair_dict:
                    ch_num = int(pair_dict["canal_fisico"])
                else:
                    ch_num = k

                musculo_tag = "No especificado"
                if nombres_canales is not None and k < len(nombres_canales) and nombres_canales[k]:
                    musculo_tag = str(nombres_canales[k]).strip()
                elif "nombre" in pair_dict and pair_dict["nombre"]:
                    musculo_tag = str(pair_dict["nombre"]).strip()
                else:
                    musculo_tag = f"Canal {ch_num}"

                id_pos = len(electrodos_objetos)
                elec_pos = _asociar_punto_a_malla(
                    px=pos_pt[0],
                    py=pos_pt[1],
                    polo="positivo",
                    canal=f"ai{ch_num} (+)",
                    musculo=musculo_tag,
                    ch_num=ch_num,
                    par_id=k,
                    elec_id=id_pos,
                )
                electrodos_objetos.append(elec_pos)

                id_neg = len(electrodos_objetos)
                elec_neg = _asociar_punto_a_malla(
                    px=neg_pt[0],
                    py=neg_pt[1],
                    polo="negativo",
                    canal=f"ai{ch_num} (-)",
                    musculo=musculo_tag,
                    ch_num=ch_num,
                    par_id=k,
                    elec_id=id_neg,
                )
                electrodos_objetos.append(elec_neg)

                # Calcular vector dipolo 3D: de Polo (-) hacia Polo (+)
                v_pos = np.asarray(elec_pos.coordenadas_3d_absolutas, dtype=np.float64)
                v_neg = np.asarray(elec_neg.coordenadas_3d_absolutas, dtype=np.float64)
                v_dipolo_3d = v_pos - v_neg
                norm_dipolo_3d = float(np.linalg.norm(v_dipolo_3d))
                vec_dipolo_norm = (v_dipolo_3d / dist_ip_3d).tolist() if dist_ip_3d > 0 else [0.0, 0.0, 0.0]

                ied_px = float(np.hypot(neg_pt[0] - pos_pt[0], neg_pt[1] - pos_pt[1]))
                ied_mm = pair_dict.get("ied_mm")
                if ied_mm is None and escala_mm_por_px is not None and escala_mm_por_px > 0:
                    ied_mm = float(ied_px * escala_mm_por_px)

                # Orientación polar en grados: vector que apunta de (-) hacia (+)
                ang_deg = float(np.degrees(np.arctan2(pos_pt[1] - neg_pt[1], pos_pt[0] - neg_pt[0])))

                par_obj = ParDiferencialEspacial(
                    id=k,
                    canal_fisico_ai=ch_num,
                    etiqueta_canal=f"ai{ch_num}",
                    musculo=musculo_tag,
                    electrodo_positivo=elec_pos,
                    electrodo_negativo=elec_neg,
                    vector_dipolo_3d=[round(float(v), 6) for v in v_dipolo_3d.tolist()],
                    vector_dipolo_normalizado=[round(float(v), 6) for v in vec_dipolo_norm],
                    distancia_interelectrodo_3d=round(norm_dipolo_3d, 6),
                    distancia_interelectrodo_px=round(ied_px, 2),
                    distancia_interelectrodo_mm=round(ied_mm, 2) if ied_mm is not None else None,
                    angulo_orientacion_deg=round(ang_deg, 2),
                )
                pares_diferenciales_objetos.append(par_obj)

        else:
            for idx, item in enumerate(centroides_2d):
                cx, cy = item["centroide"]
                ch_fisico = None
                if indices_canales_fisicos is not None and idx < len(indices_canales_fisicos):
                    ch_fisico = int(indices_canales_fisicos[idx])
                    canal_nombre = f"Canal {ch_fisico} (ai{ch_fisico})"
                else:
                    ch_fisico = idx
                    canal_nombre = canales_nombres_defecto[idx] if idx < len(canales_nombres_defecto) else f"Canal {idx}"

                musculo_tentativo = "No especificado"
                if nombres_canales is not None and idx < len(nombres_canales) and nombres_canales[idx]:
                    musculo_tentativo = str(nombres_canales[idx]).strip()
                elif asignar_musculos_automatico:
                    distancias_2d = np.hypot(puntos_2d_px[:, 0] - cx, puntos_2d_px[:, 1] - cy)
                    v_cercano = int(np.argmin(distancias_2d))
                    v_3d = puntos_3d[v_cercano]
                    d_rel = v_3d - nariz_3d
                    dy_rel, dx_rel = d_rel[1], d_rel[0]
                    if dy_rel > 0.35:
                        musculo_tentativo = "Mylohyoid / Anterior Belly (Submentoniano)"
                    elif 0.15 < dy_rel <= 0.35 and abs(dx_rel) > 0.10:
                        musculo_tentativo = "Depresor Anguli Oris / Risorio"
                    elif 0.05 < dy_rel <= 0.25 and abs(dx_rel) <= 0.10:
                        musculo_tentativo = "Orbicularis Oris (Perioral)"

                elec_obj = _asociar_punto_a_malla(
                    px=cx,
                    py=cy,
                    polo="monopolar",
                    canal=canal_nombre,
                    musculo=musculo_tentativo,
                    ch_num=ch_fisico,
                    par_id=None,
                    elec_id=len(electrodos_objetos),
                )
                electrodos_objetos.append(elec_obj)

        # Si se proporcionó punto de referencia o Tierra GND
        if punto_referencia_gnd is not None:
            gx, gy = float(punto_referencia_gnd[0]), float(punto_referencia_gnd[1])
            elec_gnd = _asociar_punto_a_malla(
                px=gx,
                py=gy,
                polo="tierra_gnd",
                canal="GND (Tierra Oreja)",
                musculo="Referencia / Tierra GND",
                ch_num=None,
                par_id=None,
                elec_id=len(electrodos_objetos),
            )
            electrodos_objetos.append(elec_gnd)

        pose_biometria = self.calcular_pose_y_biometria_facial(
            fiduciarios,
            (alto, ancho),
            d_ip_real_cm=d_ip_real_cm,
            distancia_fija_camara_cm=distancia_fija_camara_cm,
            escala_mm_por_px=escala_mm_por_px,
        )

        electrodos_procesados = [e.to_dict() for e in electrodos_objetos]
        pares_procesados = [p.to_dict() for p in pares_diferenciales_objetos]

        meta_calib: Dict[str, Any] = {
            "version_modulo": "1.3.0",
            "timestamp": datetime.now().isoformat(),
            "dimensiones_imagen_px": [ancho, alto],
            "distancia_interpupilar_3d": round(dist_ip_3d, 6),
            "distancia_interpupilar_px": round(dist_ip_px, 2),
            "metodo_fiduciario_ojos": fiduciarios["metodo_ojos"],
            "pose_y_biometria_facial": pose_biometria,
            "calibracion_metrica_referencia": {
                "d_ip_real_cm": float(d_ip_real_cm),
                "distancia_fija_camara_cm": float(distancia_fija_camara_cm) if distancia_fija_camara_cm else None,
                "escala_mm_por_px": float(escala_mm_por_px) if escala_mm_por_px else None,
                "metodo": pose_biometria.get("metodo_distancia", "antropometria"),
            },
            "origen_coordenadas": {
                "descripcion": "Punta de la nariz (pronasale)",
                "indice_mediapipe": INDICE_NARIZ_PUNTA,
                "coordenadas_3d_absolutas": [round(float(v), 6) for v in nariz_3d],
                "coordenadas_2d_px": [round(float(v), 2) for v in fiduciarios["punta_nariz_px"]],
            },
            "referencias_orejas": {
                "trago_izq_px": [round(float(v), 2) for v in fiduciarios.get("oreja_trago_izq_px", [0, 0])],
                "trago_der_px": [round(float(v), 2) for v in fiduciarios.get("oreja_trago_der_px", [0, 0])],
                "lobulo_izq_estimado_px": [round(float(v), 2) for v in fiduciarios.get("oreja_lobulo_izq_px", [0, 0])],
                "lobulo_der_estimado_px": [round(float(v), 2) for v in fiduciarios.get("oreja_lobulo_der_px", [0, 0])],
                "ancho_biauricular_px": round(float(fiduciarios.get("ancho_biauricular_px", 0.0)), 2),
                "ancho_biauricular_cm": pose_biometria.get("ancho_biauricular_cm", 0.0),
            },
            "total_electrodos_fisicos": len(electrodos_procesados),
            "total_pares_diferenciales": len(pares_procesados),
            "total_electrodos_detectados": len(electrodos_procesados),
            "electrodos": electrodos_procesados,
            "pares_diferenciales": pares_procesados,
        }

        if punto_referencia_gnd is not None:
            meta_calib["electrodo_referencia_gnd"] = {
                "coordenadas_2d_px": [round(float(punto_referencia_gnd[0]), 2), round(float(punto_referencia_gnd[1]), 2)],
                "descripcion": "Electrodo de Tierra / Referencia sEMG (Oreja / Mastoides)",
            }

        resultado: Dict[str, Any] = {
            "exito": True,
            "electrodos": electrodos_objetos,
            "pares_diferenciales": pares_diferenciales_objetos,
            "metadatos_calibracion_espacial": meta_calib,
            "_fiduciarios_cache": fiduciarios,
            "_mascara_cache": mascara,
        }

        return resultado

    def renderizar_visualizacion(
        self,
        bgr_image: np.ndarray,
        resultado_triangulacion: Dict[str, Any],
        dibujar_malla_completa: bool = False,
    ) -> np.ndarray:
        """
        Dibuja los puntos fiduciarios, orejas, centroides de electrodos (monopolares o pares diferenciales),
        y vectores métricos sobre la imagen.
        """
        canvas = bgr_image.copy()

        if not resultado_triangulacion.get("exito", False):
            cv2.putText(
                canvas,
                "ROSTRO NO DETECTADO",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return canvas

        fiduciarios = resultado_triangulacion["_fiduciarios_cache"]
        meta = resultado_triangulacion["metadatos_calibracion_espacial"]
        nariz_px = meta["origen_coordenadas"]["coordenadas_2d_px"]
        pt_nariz = (int(nariz_px[0]), int(nariz_px[1]))

        # Puntos de la distancia interpupilar
        pt_ojo_izq = (int(fiduciarios["ojo_izq_px"][0]), int(fiduciarios["ojo_izq_px"][1]))
        pt_ojo_der = (int(fiduciarios["ojo_der_px"][0]), int(fiduciarios["ojo_der_px"][1]))
        cv2.line(canvas, pt_ojo_izq, pt_ojo_der, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, pt_ojo_izq, 4, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, pt_ojo_der, 4, (0, 255, 255), -1, cv2.LINE_AA)

        # Orejas (tragos y lóbulos estimados)
        if "oreja_trago_izq_px" in fiduciarios:
            pt_trago_izq = (int(fiduciarios["oreja_trago_izq_px"][0]), int(fiduciarios["oreja_trago_izq_px"][1]))
            pt_trago_der = (int(fiduciarios["oreja_trago_der_px"][0]), int(fiduciarios["oreja_trago_der_px"][1]))
            pt_lob_izq = (int(fiduciarios["oreja_lobulo_izq_px"][0]), int(fiduciarios["oreja_lobulo_izq_px"][1]))
            pt_lob_der = (int(fiduciarios["oreja_lobulo_der_px"][0]), int(fiduciarios["oreja_lobulo_der_px"][1]))

            # Dibujar oreja izquierda
            cv2.circle(canvas, pt_trago_izq, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_trago_izq, pt_lob_izq, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_lob_izq, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja L", (pt_trago_izq[0] - 50, pt_trago_izq[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

            # Dibujar oreja derecha
            cv2.circle(canvas, pt_trago_der, 4, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.line(canvas, pt_trago_der, pt_lob_der, (255, 204, 0), 1, cv2.LINE_AA)
            cv2.circle(canvas, pt_lob_der, 3, (255, 204, 0), -1, cv2.LINE_AA)
            cv2.putText(canvas, "Oreja R", (pt_trago_der[0] + 8, pt_trago_der[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 204, 0), 1, cv2.LINE_AA)

        # Origen nasal
        cv2.circle(canvas, pt_nariz, 6, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "Origen (Nariz)",
            (pt_nariz[0] + 10, pt_nariz[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        colores_canales = [
            (0, 255, 0),
            (0, 165, 255),
            (255, 0, 255),
            (255, 255, 0),
            (100, 255, 255),
            (255, 128, 128),
        ]

        puntos_2d_px = fiduciarios["puntos_2d_px"]

        pares_dif = meta.get("pares_diferenciales", [])
        electrodos_lista = meta.get("electrodos", [])

        if pares_dif:
            # Renderizado riguroso de pares diferenciales bipolares
            for p_idx, pair in enumerate(pares_dif):
                color = colores_canales[p_idx % len(colores_canales)]
                pos_xy = pair["polo_positivo"]["coordenadas_2d_px"]
                neg_xy = pair["polo_negativo"]["coordenadas_2d_px"]
                pos_x, pos_y = int(round(pos_xy[0])), int(round(pos_xy[1]))
                neg_x, neg_y = int(round(neg_xy[0])), int(round(neg_xy[1]))

                # Flecha que representa el vector dipolo diferencial D = P_+ - P_- (de negativo a positivo)
                cv2.arrowedLine(canvas, (neg_x, neg_y), (pos_x, pos_y), (220, 220, 220), 2, tipLength=0.25)

                # Polo (+) rojo/naranja con signo '+'
                cv2.circle(canvas, (pos_x, pos_y), 7, (0, 69, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (pos_x, pos_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "+", (pos_x - 4, pos_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Polo (-) azul eléctrico con signo '-'
                cv2.circle(canvas, (neg_x, neg_y), 7, (255, 120, 0), -1, cv2.LINE_AA)
                cv2.circle(canvas, (neg_x, neg_y), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(canvas, "-", (neg_x - 3, neg_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

                # Líneas fiduciarias tenues desde la nariz a cada electrodo físico individual
                cv2.line(canvas, pt_nariz, (pos_x, pos_y), color, 1, cv2.LINE_AA)
                cv2.line(canvas, pt_nariz, (neg_x, neg_y), color, 1, cv2.LINE_AA)

                # Etiqueta del canal e IED
                ied_info = ""
                if pair.get("distancia_interelectrodo_mm"):
                    ied_info = f" [IED:{pair['distancia_interelectrodo_mm']:.1f}mm]"
                elif pair.get("distancia_interelectrodo_px"):
                    ied_info = f" [IED:{pair['distancia_interelectrodo_px']:.0f}px]"

                texto = f"{pair['etiqueta_canal']}: {pair['musculo']}{ied_info}"
                mid_x = int(round(0.5 * (pos_x + neg_x)))
                mid_y = int(round(0.5 * (pos_y + neg_y)))
                cv2.putText(
                    canvas,
                    texto,
                    (mid_x + 12, mid_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        else:
            # Renderizado de electrodos monopolares o detecciones continuas
            for el in electrodos_lista:
                if el.get("polo") == "tierra_gnd":
                    continue
                cid = el["id"]
                color = colores_canales[cid % len(colores_canales)]
                cx = int(round(el["coordenadas_2d_px"][0] if "coordenadas_2d_px" in el else el["centroide_2d_px"][0]))
                cy = int(round(el["coordenadas_2d_px"][1] if "coordenadas_2d_px" in el else el["centroide_2d_px"][1]))
                v_idx = el["vertice_malla_3d"]
                vx, vy = int(puntos_2d_px[v_idx, 0]), int(puntos_2d_px[v_idx, 1])

                cv2.circle(canvas, (cx, cy), 7, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(canvas, (vx, vy), 4, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.line(canvas, (cx, cy), (vx, vy), (200, 200, 200), 1, cv2.LINE_AA)
                cv2.line(canvas, pt_nariz, (vx, vy), color, 2, cv2.LINE_AA)

                dist_norm = el["distancia_3d_normalizada"]
                texto = f"{el['canal']} (d={dist_norm:.3f})"
                cv2.putText(
                    canvas,
                    texto,
                    (cx + 12, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        # Electrodo de referencia (Tierra / GND) si está marcado
        if "electrodo_referencia_gnd" in meta:
            gnd_coords = meta["electrodo_referencia_gnd"]["coordenadas_2d_px"]
            gx, gy = int(round(gnd_coords[0])), int(round(gnd_coords[1]))
            cv2.circle(canvas, (gx, gy), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, (gx, gy), 11, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, "GND (Tierra Oreja)", (gx + 12, gy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2, cv2.LINE_AA)

        dist_ip_px = meta["distancia_interpupilar_px"]
        dist_cm = meta["pose_y_biometria_facial"].get("distancia_camara_cm", 0.0)
        metodo_d = meta.get("calibracion_metrica_referencia", {}).get("metodo", "antropometria")

        cv2.rectangle(canvas, (10, 10), (460, 85), (0, 0, 0), -1)
        cv2.rectangle(canvas, (10, 10), (460, 85), (255, 255, 255), 1)
        cv2.putText(
            canvas,
            "Calibracion Geometrica 3D sEMG | NANDU LSD",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"D. Interpupilar: {dist_ip_px:.1f} px | Dist: {dist_cm:.1f} cm ({metodo_d})",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        n_pares = len(pares_dif)
        n_elec_tot = len(electrodos_lista)
        info_ch = f"Pares bipolares: {n_pares} ({n_elec_tot} electrodos)" if n_pares > 0 else f"Electrodos: {n_elec_tot}"
        cv2.putText(
            canvas,
            f"{info_ch} | Orejas detectadas (Trago L/R)",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return canvas

    @staticmethod
    def exportar_a_json(
        resultado_triangulacion: Dict[str, Any],
        ruta_archivo: str,
    ) -> bool:
        """Serializa y almacena los metadatos en formato JSON."""
        try:
            datos_para_guardar = {
                "exito": resultado_triangulacion.get("exito", False),
                "metadatos_calibracion_espacial": resultado_triangulacion.get("metadatos_calibracion_espacial", {}),
            }
            os.makedirs(os.path.dirname(os.path.abspath(ruta_archivo)), exist_ok=True)
            with open(ruta_archivo, "w", encoding="utf-8") as f:
                json.dump(datos_para_guardar, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            sys.stderr.write(f"Error al exportar metadatos a JSON: {e}\n")
            return False

    @staticmethod
    def inyectar_metadatos_en_csv(
        ruta_csv: str,
        resultado_triangulacion: Dict[str, Any],
        ruta_salida: Optional[str] = None,
    ) -> bool:
        """Inyecta los metadatos de calibración en la cabecera de un CSV sEMG."""
        try:
            if not os.path.exists(ruta_csv):
                sys.stderr.write(f"Archivo CSV no encontrado: {ruta_csv}\n")
                return False

            with open(ruta_csv, "r", encoding="utf-8") as f:
                lineas_originales = f.readlines()

            lineas_datos = [l for l in lineas_originales if not l.startswith("# METADATOS_GEOMETRIA_EMG:")]
            meta = resultado_triangulacion.get("metadatos_calibracion_espacial", {})
            json_linea = json.dumps(meta, ensure_ascii=False)

            cabecera_inyectada = [
                "# ==============================================================================\n",
                "# METADATOS_GEOMETRIA_EMG: " + json_linea + "\n",
                "# ==============================================================================\n",
            ]

            lineas_finales = cabecera_inyectada + lineas_datos
            destino = ruta_csv if ruta_salida is None else ruta_salida

            with open(destino, "w", encoding="utf-8") as f:
                f.writelines(lineas_finales)

            return True
        except Exception as e:
            sys.stderr.write(f"Error al inyectar metadatos en CSV: {e}\n")
            return False


# Alias semántico del agente
TrianguladorElectrodosFaciales = AntigravityAgent


# ==============================================================================
# EJECUCIÓN EN TIEMPO REAL / INTERFAZ DE PRUEBA
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibracion espacial 3D de electrodos sEMG - Agente Antigravity"
    )
    parser.add_argument(
        "--imagen",
        type=str,
        default=None,
        help="Ruta a fotografia fija para calibrar. Si se omite, abre la camara web.",
    )
    parser.add_argument(
        "--modo-metal",
        action="store_true",
        help="Habilita deteccion de broches metalicos en lugar de marcadores de color.",
    )
    parser.add_argument(
        "--guardar-json",
        type=str,
        default=None,
        help="Ruta para exportar metadatos en formato JSON.",
    )
    args = parser.parse_args()

    agente = AntigravityAgent()

    if args.imagen is not None:
        if not os.path.exists(args.imagen):
            print(f"Error: La imagen especificada no existe: {args.imagen}")
            sys.exit(1)

        print(f"Cargando imagen: {args.imagen}...")
        frame = cv2.imread(args.imagen)
        if frame is None:
            print("Error: No se pudo decodificar la imagen.")
            sys.exit(1)

        resultado = agente.triangular_electrodos(
            frame,
            usar_broches_metalicos=args.modo_metal,
        )

        if not resultado["exito"]:
            print(f"Resultado: {resultado['mensaje']}")
            sys.exit(1)

        meta = resultado["metadatos_calibracion_espacial"]
        print("\n--- METADATOS DE CALIBRACION ESPACIAL ---")
        print(f"Distancia Interpupilar 3D: {meta['distancia_interpupilar_3d']:.6f}")
        print(f"Distancia Interpupilar en Imagen: {meta['distancia_interpupilar_px']:.2f} px")
        print(f"Origen Nasal: {meta['origen_coordenadas']['coordenadas_3d_absolutas']}")
        print(f"Electrodos Detectados: {len(meta['electrodos'])}")
        for el in meta["electrodos"]:
            print(
                f"  [{el['canal']}] {el['musculo_estimado']} | "
                f"Vertice: {el['vertice_malla_3d']} | "
                f"Dist. 3D Normalizada: {el['distancia_3d_normalizada']:.4f} | "
                f"Vector 3D: {el['vector_3d_relativo_normalizado']}"
            )

        if args.guardar_json:
            agente.exportar_a_json(resultado, args.guardar_json)
            print(f"Metadatos guardados en: {args.guardar_json}")

        vis = agente.renderizar_visualizacion(frame, resultado)
        mask = resultado["_mascara_cache"]
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        h, w, _ = vis.shape
        if mask_bgr.shape != vis.shape:
            mask_bgr = cv2.resize(mask_bgr, (w, h))

        vista_dual = np.hstack((vis, mask_bgr))
        if vista_dual.shape[1] > 1600:
            escala = 1600.0 / vista_dual.shape[1]
            vista_dual = cv2.resize(vista_dual, (0, 0), fx=escala, fy=escala)

        cv2.imshow("Calibracion Geometrica sEMG - Agente Antigravity", vista_dual)
        print("\nPresione cualquier tecla en la ventana de OpenCV para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    print("Iniciando captura en tiempo real desde la camara web...")
    print("Controles de teclado:")
    print("  'q' - Salir")
    print("  's' - Guardar instantanea de calibracion (snapshot.json y snapshot.png)")
    print("  'm' - Conmutar entre marcadores HSV y broches metalicos")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo acceder a la camara web (indice 0).")
        sys.exit(1)

    usar_metal = args.modo_metal

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Aviso: Cuadro de video no disponible.")
            break

        resultado = agente.triangular_electrodos(
            frame,
            usar_broches_metalicos=usar_metal,
        )

        vis = agente.renderizar_visualizacion(frame, resultado)

        if "_mascara_cache" in resultado:
            mask = resultado["_mascara_cache"]
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if mask_bgr.shape != vis.shape:
                mask_bgr = cv2.resize(mask_bgr, (vis.shape[1], vis.shape[0]))
            vista_dual = np.hstack((vis, mask_bgr))
        else:
            vista_dual = vis

        cv2.imshow("Calibracion Geometrica sEMG - Agente Antigravity", vista_dual)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("m"):
            usar_metal = not usar_metal
            print(f"Modo de deteccion cambiado a: {'Broches metalicos' if usar_metal else 'Marcadores HSV'}")
        elif key == ord("s"):
            agente.exportar_a_json(resultado, "snapshot_calibracion.json")
            cv2.imwrite("snapshot_calibracion.png", vis)
            print("Instantanea guardada exitosamente (snapshot_calibracion.json y .png).")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
