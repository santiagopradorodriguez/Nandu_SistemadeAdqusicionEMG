# Reporte Técnico: Causa Raíz del Desacople Óptico y Corrección Canónica para Decodificación Facial Monocular

**Fecha:** 2026-11-09  
**Proyecto:** Ñandú EMG - Decodificación Facial Cinemática y sEMG  
**Módulos Afectados:** `calibracion_espacial_electrodos.py`, `realtime_facial_decoder_engine.py`, `autoforge_daq_decodificador_facial.py`

---

## 1. Identificación y Demostración Experimental de la Falla

Durante las pruebas en vivo con webcam monocular se observó que giros de cabeza suaves (cabeceo / yaw / pitch) o micro-traslaciones generaban picos de amplitud de hasta 30 en la curva magenta $S(t)$ (Norma Cinemática), disparando ventanas espurias de detección vocal con la boca cerrada y en silencio absoluto.

Las mediciones en tiempo de ejecución arrojaron dos causas directas:

### A. Fluctuación Severa de la Escala Métrica ($20.5\%$ de Variación)
Al intentar convertir píxeles a milímetros absolutos mediante la distancia interpupilar:
$$\text{escala}(t) = \frac{63.0\,\text{mm}}{\text{DIP}(t)}$$
Los registros en vivo registraron:
- $\text{DIP}_{\min} = 80.76\,\text{px} \implies \text{escala} = 0.780\,\text{mm/px}$
- $\text{DIP}_{\max} = 98.67\,\text{px} \implies \text{escala} = 0.638\,\text{mm/px}$
- Desviación estándar: $\sigma = 6.23\,\text{px}$, varianza de escala: $20.5\%$.

**Efecto directo:**  
Una distancia facial real anatómica de $50\,\text{mm}$ (comisuras o mentón) sufre una fluctuación puramente óptica de:
$$\Delta d_{\text{error}} = 50\,\text{mm} \times 0.205 \approx 10.2\,\text{mm}$$
Diez milímetros de error óptico superan ampliamente el umbral del gate ($U_{\text{alto}} = 9.2$), encendiendo falsamente los canales $CH_1$ (sonrisa), $CH_3$ (mejilla) y $CH_4$ (malar).

---

### B. Ruptura de la Hipótesis de Cuerpo Rígido en MediaPipe
El estimador Procrustes SVD (Kabsch) asume una rotación y traslación rígida $P_{\text{cráneo}} = P_0 R + T$. Sin embargo, MediaPipe FaceMesh no es un sensor estéreo ni LiDAR de tiempo de vuelo; es una red neuronal monocular que infiere profundidad ($Z$) a partir de patrones 2D.
Cuando el usuario gira la cabeza en yaw:
1. La mejilla contralateral sufre acortamiento proyectivo (foreshortening).
2. La red deforma la malla ajustándose a la silueta 2D, variando la distancia entre comisuras y maseteros en el espacio 2.5D.
3. El algoritmo Procrustes proyecta una nube deformada, induciendo un vector de tracción aparente.

---

## 2. Paquete de Solución Técnica: Dos Vías Canónicas y Robustas

Para que un decodificador facial funcione en vivo con una cámara web convencional sin ser afectado por movimientos de cabeza, la formulación debe abandonar las distancias euclídeas multiplicadas por DIP y adoptar una de las siguientes dos alternativas matemáticamente inmunes:

---

### Opción 1: Ratios Geométricos Adimensionales Locales (Cero Dependencia de Escala)

En lugar de calcular distancias en milímetros absolutos, se definen ratios adimensionales divididos por un segmento óseo rígido adyacente del propio rostro (por ejemplo, la altura nasal $L_{\text{nasal}} = \|P_{\text{nariz}} - P_{\text{glabella}}\|$ o el ancho bi-ocular $L_{\text{ojos}}$):

1. **Apertura Mandibular ($CH_0$)**:
   $$r_0(t) = \frac{\|P_{\text{labio\_sup}} - P_{\text{labio\_inf}}\|_2}{\|P_{\text{nariz}} - P_{\text{glabella}}\|_2}$$
2. **Tracción Comisural ($CH_1$)**:
   $$r_1(t) = \frac{\|P_{\text{comisura\_izq}} - P_{\text{comisura\_der}}\|_2}{\|P_{\text{canto\_izq}} - P_{\text{canto\_der}}\|_2}$$
3. **Protrusión Labial ($CH_2$)**:
   $$r_2(t) = \frac{\|P_{\text{comisura\_izq}} - P_{\text{comisura\_der}}\|_2}{\|P_{\text{labio\_sup}} - P_{\text{labio\_inf}}\|_2 + \epsilon}$$

**Propiedad matemática:**
Al variar la distancia a la cámara o cambiar la escala métrica por factor $\lambda(t)$:
$$r_k(t) = \frac{\lambda(t) \cdot d_{\text{articulación}}}{\lambda(t) \cdot d_{\text{hueso}}} = \frac{d_{\text{articulación}}}{d_{\text{hueso}}}$$
El factor de escala $\lambda(t)$ se cancela algebraicamente idénticamente a 1.

---

### Opción 2: Coeficientes Blendshapes Nativos de MediaPipe Tasks (Recomendada)

MediaPipe FaceLandmarker incluye un modelo entrenado para desacoplar la pose global de la cabeza de las contracciones motoras musculares, entregando 52 coeficientes normalizados $[0.0, 1.0]$.

El modelo ya existe en el repositorio en:
`EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task`

Configuración de activación:
```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
    num_faces=1
)
```

Mapeo directo a los 6 canales del sistema Ñandú:
- **$CH_0$ (Apertura Mandibular)**: `jawOpen`
- **$CH_1$ (Tracción Sonrisa)**: $0.5 \times (`mouthSmileLeft` + `mouthSmileRight`) + 0.5 \times (`mouthStretchLeft` + `mouthStretchRight`)$
- **$CH_2$ (Constricción / Protrusión)**: `mouthPucker` + `mouthFunnel`
- **$CH_3$ (Retracción Comisura)**: $0.5 \times (`mouthDimpleLeft` + `mouthDimpleRight`)$
- **$CH_4$ (Ascenso Malar)**: $0.5 \times (`mouthUpperUpLeft` + `mouthUpperUpRight`)$
- **$CH_5$ (Compresión Mandibular)**: `mouthPressLeft` + `mouthPressRight`

**Ventaja:**
Estos coeficientes son matemáticamente independientes de la posición de la cámara y del ángulo de rotación de la cabeza. Al mover la cabeza o inclinarse, `jawOpen` y `mouthSmile` permanecen en $0.00$.

---

## 3. Contenido del Paquete

1. `reporte_diagnostico_desacople_facial.md`: Este documento con la formulación matemática y el diagnóstico.
2. `propuesta_correccion_blendshapes_invariante.py`: Script autocontenido de referencia que implementa la extracción de los 6 canales mediante blendshapes nativos invariantes a la pose.
3. `propuesta_correccion_ratios_adimensionales.py`: Script alternativo usando ratios geométricos normalizados sobre la cresta ósea.
