# Tareas Pendientes para Argon (2026-11-08)

Fecha de registro: 2026-11-08

---

## 1. Asistente de Fotometría y Captura de Fotos

- **Tiempo de acomodación en la primera foto (Paso Frontal):**
  - **Problema:** Al iniciar el asistente, toma la primera foto casi de inmediato y no da tiempo suficiente para ubicar y acomodar la cabeza dentro del círculo guía.
  - **Acción:**
    - Agregar una cuenta regresiva previa o requerir una mayor cantidad de frames estables continuos en verde (ej. 2 a 3 segundos de estabilidad comprobada) antes de disparar el primer paso.
    - Opcionalmente, agregar un botón explícito de "Listo / Comenzar" para que el usuario inicie la detección recién cuando ya esté centrado y preparado.

- **Detección y disparo en giro de perfil (Izquierda y Derecha):**
  - **Problema:** Al girar la cabeza a la izquierda o a la derecha, no logra disparar la foto ni poner el semáforo en verde.
  - **Acción:**
    - Auditar la convención de signos del ángulo `yaw` frente al volteo horizontal (`cv2.flip(frame, 1)`) de la vista previa de la webcam.
    - Flexibilizar las tolerancias de `roll` (actualmente en $\pm 6^\circ$, muy difícil de mantener al rotar el cuello) y de distancia para los perfiles.
    - Incorporar un botón manual de respaldo ("Capturar Foto Ahora") para no trabar el flujo de adquisición si el semáforo automático oscila.

- **Traducción de Badges / Semáforos a Español:**
  - **Problema:** Los indicadores de pose en la barra superior están en inglés.
  - **Acción:**
    - `Yaw` -> `Giro` (o `Giro (Yaw)`)
    - `Pitch` -> `Inclinación` (o `Inclinación (Pitch)`)
    - `Roll` -> `Ladeo` (o `Ladeo (Roll)`)
    - `Dist` -> `Distancia`

---

## 2. Visualización y Grilla en Gráficos en Tiempo Real

- **Visibilidad de la grilla:**
  - **Problema:** La grilla de fondo en los gráficos de visualización en tiempo real es poco visible.
  - **Acción:**
    - Aumentar el contraste, opacidad o grosor de las líneas de la grilla en PyQtGraph (`showGrid(x=True, y=True, alpha=...)`).

- **Submuestreo excesivo en Visor CSV (`csv_viewer_widget.py`):**
  - **Problema:** En el visor de grabaciones CSV, al hacer zoom sobre una ventana corta (ej. 0.63 segundos de audio/EMG), la señal se ve marcadamente pixelada o poligonal (submuestreada en exceso), perdiendo la morfología continua de la onda y de los picos de acción muscular.
  - **Causa identificada:** En la línea 1738 se aplica `downsample_lttb_fast(self.time_data, y_data, MAX_POINTS_TO_PLOT)` sobre **todo el array completo de la grabación** antes de recortar la vista, en lugar de aplicarlo dinámicamente sobre la ventana temporal visible en pantalla (`[x_min, x_max]`).
  - **Acción:**
    - Recortar primero los datos al rango visible actual (`x_min` a `x_max`) y recién después aplicar el algoritmo LTTB / MinMax sobre la ventana visible, o bien aumentar dinámicamente el presupuesto de puntos cuando la ventana en pantalla es inferior a 2-3 segundos.
    - Asegurar que a resoluciones de zoom fino la señal se grafique con todas sus muestras reales sin pérdida de detalle.

---

## 3. Audio del Metrónomo (AutoForge DAQ)

- **Corte abrupto en la cola del sonido:**
  - **Problema:** Con auriculares de buena calidad se percibe un leve chasquido o corte abrupto al final del sonido del metrónomo.
  - **Acción:**
    - Aplicar una rampa de desvanecimiento suave (*fade-out* de 5 a 10 ms con ventana Hann o coseno) en la cola final del buffer de audio para evitar la discontinuidad a cero.
    - Asegurar que el buffer DMA del reproductor no se interrumpa anticipadamente antes de completar el decaimiento sonoro.

---

## 4. Persistencia, Carpetas y Timestamp de Sesiones (Evitar Sobreescritura)

- **Colisión de sesiones con el mismo nombre de prueba:**
  - **Problema:** Al repetir una toma con el mismo sujeto y prueba (ej. `SecuenciaContinua_Prueba1_Sujeto1`), el sistema no genera una carpeta nueva ni añade timestamp, sino que reutiliza la misma carpeta física existente. Esto provoca mezcla de archivos de corridas distintas (ej. `grabacion.csv` actualizado a las 14:20 pero `registro_orden_secuencia.txt` remanente de las 12:56 con orden de vocales desincronizado e invertido).
  - **Acción:**
    - Incorporar obligatoriamente timestamp o sufijo incremental por toma (`YYYYMMDD_HHMMSS` o `_01`, `_02`) en el nombre de la carpeta de salida si ya existe una toma previa.
    - Eliminar los bloques `except: pass` silenciosos en el pipeline de guardado de metadatos e informes de protocolo para que cualquier fallo de escritura se visibilice y se capture inmediatamente.
    - Garantizar que la fuente única de la verdad para el orden de vocales ejecutado sea `pulsos_metronomo.csv` y `metadata.json`.
