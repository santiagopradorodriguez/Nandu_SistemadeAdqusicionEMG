# Tareas Pendientes para Argon (Backlog Técnico)

Fecha de creación: 2026-11-04

---

## 1. Fotometría y Marcación Fiduciaria de Electrodos
- **Pregunta previa sobre canal submentoniano:**
  - Al iniciar el asistente de fotometría, consultar explícitamente si alguno de los canales sEMG está montado en la región submentoniana (vientre anterior del digástrico / milohioideo).
- **Asignación contextual de vistas según anatomía:**
  - Si un canal se designa como submentoniano, excluirlo de la marcación en la fotografía frontal (evitando proyecciones forzadas o distorsiones de perspectiva). Debe solicitarse su marcación únicamente en la fotografía submentoniana (vista inferior mandibular).
  - La foto frontal queda reservada exclusivamente para músculos faciales directos (cigomático, orbicular, depresor del ángulo de la boca).
- **Exclusión obligatoria de canal acústico (Ch3):**
  - Ch3 corresponde al micrófono acústico (no porta electrodo físico adherido a la piel).
  - Omitir Ch3 de la lista de marcadores fiduciarios y de las fotos de montaje anatómico.

---

## 2. Metrónomo en AutoForge DAQ (Audio Glitch / Clic de Fin de Pulso)
- **Diagnóstico:**
  - El sonido del metrónomo se reproduce bien durante el cuerpo del beep, pero en la cola final se percibe un corte abrupto o chasquido (similar a buffer underrun o salto a cero con baja latencia).
- **Causas identificadas:**
  - Truncamiento abrupto del buffer de audio sin rampa de desvanecimiento (*fade-out* suave de 5 a 10 ms con ventana Hann o coseno), lo que produce una discontinuidad de alta frecuencia audible (click).
  - Liberación o cierre prematuro del dispositivo/stream de audio antes del vaciado completo del buffer DMA de reproducción.
- **Acción requerida:**
  - Aplicar ventana con rampa de decaimiento suave en el sintetizador/buffer del metrónomo.
  - Asegurar tiempo de guarda (drain/flush) para que el final del audio sea limpio y natural.

---

## 3. Identidad Visual y Nombre Oficial en la Ventana Principal (Main Hub)
- **Nombre Oficial del Software:**
  - El título de la aplicación en la ventana principal, barra de título y encabezado superior debe decir: **Ñandú LSD** (reemplazando "Plataforma de Investigación EMG v6.2").
- **Icono Oficial del Ñandú:**
  - En la barra de título, en el icono de la ventana del sistema operativo y en el avatar del encabezado superior (arriba a la izquierda, al lado del título), debe utilizarse el logo oficial del ñandú (`logo_nandu_lsd.png`, el mismo que aparece en la pantalla de carga / splash screen), con resolución nítida y renderizado circular limpio.

---

## 4. Asistente de Fotometría y Pose de Cabeza
- **Traducción de Badges / Semáforos a Español:**
  - Traducir las etiquetas anglosajonas de la barra superior del asistente a español legible:
    - `Yaw` -> `Giro` (o `Giro (Yaw)`)
    - `Pitch` -> `Inclinación` (o `Inclinación (Pitch)`)
    - `Roll` -> `Ladeo` (o `Ladeo (Roll)`)
    - `Dist` -> `Distancia`
- **Falla de Disparo al Girar a la Izquierda o Derecha (Perfiles):**
  - Diagnóstico: Al girar la cabeza a $\pm 45^\circ$, el asistente no logra poner el semáforo en verde ni capturar automáticamente la foto.
  - Causas a revisar y corregir:
    - Convención de signos de `yaw` entre cámara con espejo (`cv2.flip(frame, 1)`) vs coordenadas crudas de MediaPipe FaceMesh.
    - Rangos angulares y tolerancias excesivamente estrictas en `roll` ($\pm 6^\circ$, muy fácil de violar al rotar) y `distancia`.
    - Bloqueo por signo previo en `_yaw_signo_perfil`.
    - Incorporación de botón manual de respaldo ("Capturar Foto Ahora") por si el usuario desea forzar la toma sin quedar trabado en el semáforo.

---

## 5. Decodificador Facial en Tiempo Real (Fallas Críticas y Falsos Positivos)
- **Problema Reportado por el Usuario:**
  - El decodificador cinemático facial en vivo funciona muy mal: detecta palabras/fonemas continuamente de forma errática sin que el usuario mueva la cara ("modo sin sonido").
- **Causas Raíz Detectadas:**
  1. **Falta de Invarianza Intersujeto en el Reposo Facial (Baseline Fijo):**
     - El pipeline y el checkpoint `campeon_geosup_conv_ortho_3d.pt` asumían distancias biométricas de reposo fijas (heredadas de un solo sujeto, e.g. comisuras 52.07 mm). Si otro usuario tiene comisuras en reposo de 45.5 mm, la constricción labial ($c_3$) se dispara a más de 6.5 mm en reposo absoluto, superando permanentemente el umbral $U_{\text{alto}}$ del Gate Doble.
     - Se requiere que el modelo o el extractor normalice cada sujeto respecto a su propio reposo en línea (calibración adaptativa robusta de los percentiles de reposo para $c_1 \dots c_6$).
  2. **Dependencia y Falla en Modo Autónomo ("Sin Sonido"):**
     - En la validación offline con micrófono (`MIC_SYNC`) el pico acústico actúa como ancla temporal forzada. Pero en tiempo real puramente cinemático (sin audio / silencioso), el detector de flancos del Gate Doble se satura o desincroniza la ventana de 150 puntos respecto a los fonemas reales.
  3. **Dispersión y Acoplamiento de Canales:**
     - El mapeo cinemático actual 6D genera vectores que caen fuera de las variedades motoras del entrenamiento cuando no hay sincronismo exacto.
- **Acción Requerida para Argon:**
  - Re-entrenar o re-formular el decodificador facial cinemático para que sea robusto a variaciones anatómicas intersujeto reales sin requerir micrófono.
  - Rediseñar el mecanismo de disparo para que no dependa de baselines rígidos ni se auto-dispare con fluctuaciones menores de MediaPipe.
