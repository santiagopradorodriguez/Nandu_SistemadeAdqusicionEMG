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
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import numpy as np
import mediapipe as mp


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
            "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task",
        ]

        ruta_modelo_valida = None
        for r in rutas_candidatas:
            if r and os.path.exists(r):
                ruta_modelo_valida = r
                break

        self.modo_moderno = False
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
        la punta nasal (origen) y la distancia interpupilar base de escala.
        """
        alto, ancho, _ = bgr_image.shape
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        landmarks_extraidos = None

        if self.modo_moderno and self.detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            resultado_det = self.detector.detect(mp_image)
            if resultado_det.face_landmarks and len(resultado_det.face_landmarks) > 0:
                landmarks_extraidos = resultado_det.face_landmarks[0]
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
            "dimensiones_imagen": (ancho, alto),
        }

    def triangular_electrodos(
        self,
        bgr_image: np.ndarray,
        puntos_centroides_manuales: Optional[List[Tuple[float, float]]] = None,
        usar_broches_metalicos: bool = False,
        asignar_musculos_automatico: bool = True,
    ) -> Dict[str, Any]:
        """
        Ejecuta la triangulación geométrica completa de los electrodos sEMG.

        Asocia cada centroide 2D al vértice 3D de la malla facial más cercano, calcula
        el vector 3D relativo respecto a la punta nasal y normaliza la distancia euclidiana
        dividiendo por la distancia interpupilar.
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

        if puntos_centroides_manuales is not None and len(puntos_centroides_manuales) > 0:
            centroides_2d = [
                {"centroide": (float(pt[0]), float(pt[1])), "area": 0.0, "fuente": "manual"}
                for pt in puntos_centroides_manuales
            ]
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

        electrodos_procesados: List[Dict[str, Any]] = []
        canales_nombres = ["Canal 0", "Canal 1", "Canal 2", "Canal 3"]

        for idx, item in enumerate(centroides_2d):
            cx, cy = item["centroide"]

            distancias_2d = np.hypot(puntos_2d_px[:, 0] - cx, puntos_2d_px[:, 1] - cy)
            vertice_cercano = int(np.argmin(distancias_2d))
            error_reproyeccion_px = float(distancias_2d[vertice_cercano])

            vertice_3d = puntos_3d[vertice_cercano]
            delta_3d = vertice_3d - nariz_3d
            distancia_3d_absoluta = float(np.linalg.norm(delta_3d))
            distancia_3d_normalizada = float(distancia_3d_absoluta / dist_ip_3d)
            delta_3d_normalizado = (delta_3d / dist_ip_3d).tolist()

            canal_nombre = canales_nombres[idx] if idx < len(canales_nombres) else f"Canal {idx}"
            musculo_tentativo = "No especificado"

            if asignar_musculos_automatico:
                dy_rel = delta_3d[1]
                dx_rel = delta_3d[0]
                if dy_rel > 0.35:
                    musculo_tentativo = "Mylohyoid / Anterior Belly (Submentoniano)"
                elif 0.15 < dy_rel <= 0.35 and abs(dx_rel) > 0.10:
                    musculo_tentativo = "Depresor Anguli Oris / Risorio"
                elif 0.05 < dy_rel <= 0.25 and abs(dx_rel) <= 0.10:
                    musculo_tentativo = "Orbicularis Oris (Perioral)"

            electrodos_procesados.append({
                "id": idx,
                "canal": canal_nombre,
                "musculo_estimado": musculo_tentativo,
                "centroide_2d_px": [round(cx, 2), round(cy, 2)],
                "vertice_malla_3d": vertice_cercano,
                "error_reproyeccion_2d_px": round(error_reproyeccion_px, 2),
                "coordenadas_3d_absolutas": [round(float(v), 6) for v in vertice_3d],
                "vector_3d_relativo_normalizado": [round(float(v), 6) for v in delta_3d_normalizado],
                "distancia_3d_normalizada": round(distancia_3d_normalizada, 6),
                "distancia_3d_absoluta": round(distancia_3d_absoluta, 6),
            })

        resultado: Dict[str, Any] = {
            "exito": True,
            "metadatos_calibracion_espacial": {
                "version_modulo": "1.0.0",
                "timestamp": datetime.now().isoformat(),
                "dimensiones_imagen_px": [ancho, alto],
                "distancia_interpupilar_3d": round(dist_ip_3d, 6),
                "distancia_interpupilar_px": round(dist_ip_px, 2),
                "metodo_fiduciario_ojos": fiduciarios["metodo_ojos"],
                "origen_coordenadas": {
                    "descripcion": "Punta de la nariz (pronasale)",
                    "indice_mediapipe": INDICE_NARIZ_PUNTA,
                    "coordenadas_3d_absolutas": [round(float(v), 6) for v in nariz_3d],
                    "coordenadas_2d_px": [round(float(v), 2) for v in fiduciarios["punta_nariz_px"]],
                },
                "total_electrodos_detectados": len(electrodos_procesados),
                "electrodos": electrodos_procesados,
            },
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
        Dibuja los puntos fiduciarios, centroides de electrodos y vectores métricos sobre la imagen.
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
        ]

        puntos_2d_px = fiduciarios["puntos_2d_px"]

        for el in meta["electrodos"]:
            cid = el["id"]
            color = colores_canales[cid % len(colores_canales)]
            cx, cy = int(el["centroide_2d_px"][0]), int(el["centroide_2d_px"][1])
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
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

        dist_ip_px = meta["distancia_interpupilar_px"]
        cv2.rectangle(canvas, (10, 10), (450, 75), (0, 0, 0), -1)
        cv2.rectangle(canvas, (10, 10), (450, 75), (255, 255, 255), 1)
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
            f"D. Interpupilar: {dist_ip_px:.1f} px | Electrodos: {len(meta['electrodos'])}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Escala: Invariante 3D dividida por D. Interpupilar",
            (20, 68),
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
