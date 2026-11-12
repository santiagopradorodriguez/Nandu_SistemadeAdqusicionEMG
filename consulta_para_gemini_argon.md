# Consulta Técnica para Gemini Argon: Coarticulación Facial, Discrepancia de Features e Invarianza Interpulso

## 1. Contexto del Problema y Síntoma en Vivo
Se cuenta con un decodificador facial cinemático tricanal/6-canales que en pruebas offline y bloques estacionarios alcanza entre el 80% y 95% de precisión en la decodificación de las 5 vocales del castellano (/a/, /e/, /i/, /o/, /u/). Sin embargo, al pasar a la inferencia en tiempo real (`autoforge_daq_decodificador_facial.py` y `realtime_facial_decoder_engine.py`), el desempeño cae y se observan dificultades marcadas (particularmente en vocales intermedias como /e/ y en transiciones continuas sin retorno a reposo neutro).

Hemos detectado dos frentes críticos que requieren análisis y formulación matemática:

---

## 2. Frente A: Discrepancia en la Extracción de Features y Sincronización Temporal
En los generadores de referencia (`generador_pca_6canales.py`, `generador_pca_cinetico.py`):
1. **Reloj Maestro Acústico:** Se utiliza el micrófono con una envolvente suavizada mediante una ventana móvil de **250 ms** (`win_size = sr * 0.25`), lo cual elimina la estructura armónica fina y produce una campana unimodal cuyo máximo define el centro acústico exacto $t=0$ de la fonación.
2. **Modo Gate Doble (Mioeléctrico / Cinemático):** En ausencia de audio, el trigger por norma cinemática $S(t) = \sqrt{\sum_c (x_c(t) - x_{\text{rest},c})^2}$ detecta el onset por superación de umbral, pero el pico cinemático tiene un desfasaje fisiológico natural respecto a la apertura máxima o al pico de energía.
3. **Discrepancia en Tiempo Real:** En el decodificador en vivo, el hilo de audio (`audio_callback`) no mantenía un buffer circular de 250 ms ni la ganancia calibrada de los scripts offline, introduciendo jitter temporal en el corte de las ventanas ($N=25$ muestras a 100 Hz, $[-600\text{ ms}, +800\text{ ms}]$). Al recortar la ventana con desalineación de apenas 50–100 ms, las features proyectadas en el espacio latente caen fuera de las elipses de confianza de los clasificadores.

---

## 3. Frente B: Discusión Teórica de la Coarticulación y No Retorno al Reposo
1. **El Problema del Reposo Dinámico:** El pipeline tradicional asume un piso de reposo facial neutro calculado como percentil bajo global de la sesión (p. ej. percentil 10 o 20) o un reposo interpulso entre vocales aisladas.
2. **En habla continua o cadencias altas (30–40 BPM y superiores):** El aparato articulatorio no retorna a la postura neutra de reposo entre fonemas sucesivos. Existe una rampa preparatoria pre-fonatoria y un arrastre inercial post-fonatorio (coarticulación hacia adelante y hacia atrás).
3. **Colapso del Filtro de Piso de Ruido:** Si se sustrae un reposo estimado localmente cuando el sujeto aún mantiene la boca semiabierta o comisuras retraídas, se distorsiona la escala física de la deformación $\Delta x_c(t) = x_c(t) - x_{\text{rest},c}$, comprimiendo o deformando artificialmente la trayectoria en el espacio latente.
4. **Impacto en PCA y Conformer:**
   - En PCA lineal, las vocales intermedias (/e/, /o/) quedan solapadas o tironeadas hacia el centro si el origen de coordenadas está desplazado por el residuo coarticulatorio.
   - En el Conformer (`entrenar_conformer_cinetico_18ch.py`), la atención multi-cabezal y las convoluciones temporales tienen el potencial de modelar dependencias contextuales y velocidades/aceleraciones, pero requieren que la parametrización de la ventana no asuma reposo cero estático en los bordes.

---

## 4. Preguntas Concretas para Gemini Argon
1. **Modelado de Coarticulación:** ¿Cómo formular matemáticamente una normalización o representación de estado cinemático que desacople la vocal objetivo del estado articulatorio de partida (offset de coarticulación) sin depender de un retorno estricto a reposo neutro?
2. **Alineación Híbrida Robusta (Gate vs Mic):** ¿Cuál es la función óptima de snapping/desfasaje para fusionar el onset del Gate cinemático con el reloj acústico del micrófono con ventana de 250 ms, asegurando que el centro $t=0$ sea invariante entre entrenamiento y tiempo real?
3. **Adaptación en Espacio PCA y Conformer:** ¿Qué transformaciones de características (derivadas temporales $\Delta$, aceleraciones $\Delta^2$, normalización por supremo de pulso o proyecciones diferenciales) maximizan la separabilidad de /e/ frente a /o/ e /i/ bajo regímenes coarticulatorios?
