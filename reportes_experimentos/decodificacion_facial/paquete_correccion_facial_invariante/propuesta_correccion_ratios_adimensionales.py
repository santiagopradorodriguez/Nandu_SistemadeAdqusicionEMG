#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Propuesta Alternativa: Decodificación Facial Monocular por Ratios Geométricos Adimensionales.

Inmunidad matemática comprobada:
- Divide distancias articulatorias móviles por la distancia fija entre cantos oculares o cresta nasal.
- El factor de escala métrica (px/mm) se cancela algebraicamente: (px * s) / (px * s) = 1.
- Inmune al alejamiento/acercamiento de la cabeza a la webcam.
"""

import cv2
import numpy as np

def calcular_ratios_adimensionales(puntos_2d_px: np.ndarray) -> np.ndarray:
    """
    Calcula 6 canales normalizados adimensionales:
      CH0: Apertura labial relativa a la altura del puente nasal (13-14 / 1-10)
      CH1: Ancho de comisuras relativo al ancho bi-ocular (61-291 / 33-263)
      CH2: Constricción orbicular (protrusión relativa)
      CH3: Distancia comisura-mejilla normalizada
      CH4: Ascenso malar normalizado
      CH5: Compresión maseterina normalizada
    """
    pts = np.asarray(puntos_2d_px, dtype=np.float64)
    if len(pts) < 468:
        return np.zeros(6, dtype=np.float32)

    def _d2(i1, i2):
        return float(np.linalg.norm(pts[i1] - pts[i2]))

    # Segmentos óseos rígidos de referencia local (invariantes a articulación)
    ref_altura_nasal = max(_d2(1, 10), 10.0)      # Pronasale (1) a Glabella (10)
    ref_ancho_ojos = max(_d2(33, 263), 15.0)      # Canto externo izq (33) a der (263)

    # 1. Apertura labial (13 a 14) normalizada por altura nasal
    r0 = _d2(13, 14) / ref_altura_nasal

    # 2. Ancho comisural (61 a 291) normalizado por ancho entre ojos
    r1 = _d2(61, 291) / ref_ancho_ojos

    # 3. Constricción / Protrusión (relación comisuras / apertura)
    r2 = ref_ancho_ojos / max(_d2(61, 291), 10.0)

    # 4. Mejilla izquierda y derecha respecto a comisuras
    r3 = 0.5 * (_d2(61, 207) + _d2(291, 427)) / ref_ancho_ojos

    # 5. Malar
    r4 = 0.5 * (_d2(61, 123) + _d2(291, 352)) / ref_altura_nasal

    # 6. Masetero
    r5 = 0.5 * (_d2(61, 58) + _d2(291, 288)) / ref_ancho_ojos

    return np.array([r0, r1, r2, r3, r4, r5], dtype=np.float32)
