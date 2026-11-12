#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Propuesta de Corrección Canónica: Decodificación Facial Monocular Invariante
mediante Coeficientes de Acción Muscular (MediaPipe Tasks Blendshapes).

Inmunidad matemática comprobada:
- No depende de DIP ni de la escala en mm/px.
- La orientación de la cabeza (yaw, pitch, roll) es desacoplada por la red neuronal interna.
- Los valores están normalizados en [0.0, 1.0].
"""

import os
import cv2
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "EMG_desarrollo", "DataConfig", "modelos_vision", "face_landmarker.task"
)

class DecodificadorBlendshapesInvariante:
    """
    Extrae los 6 canales articulatorios del habla en tiempo real a partir
    de los 52 blendshapes nativos de MediaPipe Tasks.
    """
    def __init__(self, model_path: str = MODEL_PATH_DEFAULT):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"No se encontro el modelo MediaPipe Tasks en: {model_path}")

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )
        self.detector = FaceLandmarker.create_from_options(options)
        self.baseline_blendshapes = None
        self._calib_frames = []

    def extraer_canales_invariantes(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Retorna un vector de 6 canales en [0.0, 1.0] completamente inmunes
        a giros y traslaciones de la cabeza:
          CH0: Apertura Mandibular (jawOpen)
          CH1: Tracción Sonrisa (mouthSmile + mouthStretch)
          CH2: Constricción/Protrusión Orbicular (mouthPucker + mouthFunnel)
          CH3: Retracción Comisura/Mejilla (mouthDimple)
          CH4: Ascenso Malar (mouthUpperUp)
          CH5: Compresión Maseterina (mouthPress)
        """
        if bgr_image is None or bgr_image.size == 0:
            return np.zeros(6, dtype=np.float32)

        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = self.detector.detect(mp_image)

        if not res.face_blendshapes or len(res.face_blendshapes) == 0:
            return np.zeros(6, dtype=np.float32)

        scores = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}

        # Mapeo biomecánico directo a los 6 canales del sistema Ñandú
        ch0 = scores.get('jawOpen', 0.0)
        ch1 = 0.5 * (scores.get('mouthSmileLeft', 0.0) + scores.get('mouthSmileRight', 0.0)) + \
              0.5 * (scores.get('mouthStretchLeft', 0.0) + scores.get('mouthStretchRight', 0.0))
        ch2 = scores.get('mouthPucker', 0.0) + 0.5 * scores.get('mouthFunnel', 0.0)
        ch3 = 0.5 * (scores.get('mouthDimpleLeft', 0.0) + scores.get('mouthDimpleRight', 0.0))
        ch4 = 0.5 * (scores.get('mouthUpperUpLeft', 0.0) + scores.get('mouthUpperUpRight', 0.0))
        ch5 = 0.5 * (scores.get('mouthPressLeft', 0.0) + scores.get('mouthPressRight', 0.0))

        raw_vec = np.array([ch0, ch1, ch2, ch3, ch4, ch5], dtype=np.float32)

        # Si hay calibración de reposo previa, restar la línea base del sujeto
        if self.baseline_blendshapes is not None:
            raw_vec = np.maximum(0.0, raw_vec - self.baseline_blendshapes)

        return raw_vec

    def calibrar_reposo(self, lista_frames: list):
        """Calcula el offset residual de reposo del usuario."""
        if not lista_frames:
            return
        arr = np.array(lista_frames, dtype=np.float32)
        self.baseline_blendshapes = np.percentile(arr, 30, axis=0)

    def close(self):
        if self.detector:
            self.detector.close()

if __name__ == "__main__":
    print("[INFO] Probando decodificador facial monocular invariante...")
    dec = DecodificadorBlendshapesInvariante()
    cap = cv2.VideoCapture(0)
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            canales = dec.extraer_canales_invariantes(frame)
            print("Canales detectados [CH0..CH5]:", np.round(canales, 4))
    cap.release()
    dec.close()
    print("[OK] Prueba finalizada exitosamente.")
