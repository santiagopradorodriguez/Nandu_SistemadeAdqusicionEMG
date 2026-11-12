# Especificación Técnica para Argon: Desacople de Cuerpo Rígido Craneal, Tracking Cinemático Rápido y Re-entrenamiento del Decodificador

Fecha: 2026-11-09
Autor: LSD - Laboratorio de Sistemas Dinámicos (FCEyN - UBA)

---

## 1. Motivación y Diagnóstico Físico

En las pruebas en vivo del decodificador facial y en los análisis cinemáticos de secuencias continuas, se detectó una fuente crítica de ruido y error sistemático:
- **Sensibilidad al movimiento de la cabeza:** Cualquier giro cervical (yaw), cabeceo (pitch) o inclinación lateral (roll), así como acercamientos o acomodamientos del sujeto, modifican las coordenadas de los labios y mandíbula en el plano 2D de la cámara.
- **Falsos positivos fonatorios:** El algoritmo interpreta este movimiento rígido global como una contracción articulatoria (ej. apertura mandibular o retracción de comisuras), disparando eventos espurios sin que el sujeto esté fonando.
- **Falta de resolución en transitorios rápidos:** Al decir vocales a ritmo natural o con jitter, las transiciones articulatorias rápidas (60–100 ms) se empastan si se aplica un suavizado excesivo o si no se captura la velocidad instantánea.
- **Limitación de los datos previos:** Al auditar `trayectoria_puntos.csv` del 2026-11-08, se constató que se guardaron hasta 18 landmarks periorales/musculares, pero **no se registraron las coordenadas de puntos óseos de anclaje (nariz, ojos, orejas)**, impidiendo calcular la matriz de rotación rígida retrospectivamente.

---

## 2. Plan de Acción y Tareas para Argon

### Tarea A: Desacople de Cuerpo Rígido en `calibracion_espacial_electrodos.py` y `fotometria_webcam.py`
1. **Adición obligatoria de Landmarks Óseos Invariantes a la Fonación:**
   Incorporar en el trazador de puntos (`PUNTOS_TRAZADOS`) y en la exportación continua los puntos fiduciarios que no se deforman con el habla:
   - Pronasale / Puente nasal: índice 1 (y opcionalmente 4).
   - Cantos oculares / Centros de ojos: cantos externos 33 (izq) y 263 (der) o centros de iris 468, 473.
   - Trago / Inserción auricular: trago izq 234, trago der 454 (para anclaje biauricular profundo).
   - Frente / Glabela: índice 10.
2. **Definición del Sistema de Coordenadas Local del Cráneo (Head Frame):**
   - Centro de masa óseo: $C_{\text{craneo}}(t) = \frac{1}{3} (P_{\text{nariz}} + P_{\text{ojo\_izq}} + P_{\text{ojo\_der}})$.
   - Vector horizontal ocular (roll): $u_x = \frac{P_{\text{ojo\_der}} - P_{\text{ojo\_izq}}}{\|P_{\text{ojo\_der}} - P_{\text{ojo\_izq}}\|}$.
   - Vector vertical (pitch): $u_y$ ortogonal a $u_x$ proyectado hacia la glabela.
   - Vector normal al rostro (yaw/profundidad): $u_z = u_x \times u_y$.
3. **Proyección Local Libre de Movimiento Craneal:**
   Cada punto articulatorio móvil $k$ (labio sup/inf 13/14, comisuras 61/291, mentón 152, mejillas 207/427, maseteros 58/288) debe referenciarse al marco craneal:
   $$r_{k,\text{local}}(t) = R(t)^T \big(P_k(t) - C_{\text{craneo}}(t)\big)$$
   Donde $R(t) = [u_x, u_y, u_z]$. De esta forma, si el sujeto cabecea o se mueve, $r_{k,\text{local}}(t)$ permanece invariante en cero.

---

### Tarea B: Detección y Corte de Ventanas en Movimientos Rápidos
1. **Reducción de latencia por filtrado:**
   - Reemplazar los filtros uniformes pesados (`win_cinem_frames = 3..5` que retrasan ~100-150 ms la señal a 30 FPS) por diferencias centrales de orden 2 para velocidad instantánea $\frac{d}{dt}r(t)$ o filtros causales ligeros (EMA con $\alpha \ge 0.65$).
2. **Segmentación y Detección Robusta de Ventanas (Histeresis y Energía Articulatoria):**
   - La ventana de extracción debe anclarse al pico de velocidad articulatoria y permitir ventanas asimétricas o adaptativas (ej. $-250\,\text{ms}$ a $+350\,\text{ms}$) si el fonema es corto, en lugar de forzar siempre $\pm 500\,\text{ms}$ que captura silencios adyacentes.
   - Implementar un mecanismo de doble umbral con histéresis (Trigger High para iniciar detección de movimiento y Trigger Low para cierre de ventana), evitando múltiples ventanas solapadas o disparos espurios por jitter de microexpresiones.
3. **Visualización Fidedigna de la Ventana en la Interfaz Gráfica:**
   - En el panel de visualización en tiempo real (gráficos temporales de la GUI), resaltar con claridad milimétrica la **región exacta de la ventana detectada** (ej. mediante un sombreado vertical translúcido `LinearRegionItem` de PyQtGraph o recuadro bien delimitado).
   - Mostrar con precisión los límites $[t_{\text{inicio}}, t_{\text{fin}}]$ y la duración exacta en ms de la ventana que ingresó al decodificador, para que el usuario pueda auditar visualmente al instante si la ventana abarcó la articulación completa o si cortó antes/después.

---

### Tarea C: Re-entrenamiento del Decodificador Autónomo e Invarianza Intersujeto
Argon debe definir y ejecutar el protocolo de entrenamiento:
1. **Con qué datos entrenar (Consulta Previa Obligatoria a Santiago):**
   - **Sesión de Referencia Canónica:** La sesión real balanceada de 200 pulsos con 18 landmarks faciales es `SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02` (contiene 40 pulsos reales por vocal /a/, /e/, /i/, /o/, /u/ descontando el count-in).
   - **IMPORTANTE:** Como las grabaciones previas no guardaron las coordenadas (x, y) de los landmarks óseos de referencia (ojos, nariz, orejas), **Argon debe consultar a Santiago antes de asumir el entrenamiento** si requiere que se genere y grabe un **nuevo dataset con la adquisición extendida de landmarks** (o si prefiere generar primero un pipeline de prueba con datos sintéticos/aumentados de validación).
   - Entrenar con las matrices cinemáticas transformadas al **Head Frame local** y convertidas a **ratios relativos de deformación** ($\Delta d / d_{\text{reposo}}$) para que sea 100% independiente del tamaño facial del sujeto.
2. **Velocidades y Ritmos (Jitters):**
   - Generar aumentación de datos (data augmentation) con variaciones de escala temporal ($\pm 25\%$) y corrimientos de jitter temporal ($\pm 50\,\text{ms}$ a $\pm 120\,\text{ms}$) para que el modelo sea robusto ante ritmos de metrónomo variables (40 a 90 BPM) y ante la falta del pico acústico de micrófono.
3. **Condición de Reposo Estricto:**
   - Incluir una clase o estado explícito de "Reposo / Silencio" para que ante movimientos menores o ruido residual de MediaPipe la probabilidad latente colapse a reposo en lugar de proyectar a una vocal aleatoria.

---

### Tarea E: Corrección Crítica en Adquisición y Modo Gate (Desconexión de Micrófono y Sesgo a /I/)
En las pruebas en vivo se detectaron dos fallas operativas graves que deben corregirse:
1. **Desconexión Total del Micrófono en Modo Gate Cinemático:**
   - Si el sistema está operando en modo Gate Doble (`GATE_HYBRID`, `GATE_CENTROID`, etc.), **el micrófono no debe intervenir bajo ninguna circunstancia**:
     - No debe abrirse el stream de audio innecesariamente o debe forzarse a cero (`chunk[6, :] = 0.0`), evitando que el ruido ambiental o acústico abra falsas ventanas o interfiera con el clasificador.
     - Agregar un control visible en la GUI (switch/checkbox) para "Habilitar / Deshabilitar Micrófono" de forma explícita.
2. **Colapso Sistemático a /I/ (100%) por Umbral Rígido del Gate:**
   - Se observó que el sistema predice repetidamente /I/ al 100% de confianza. Esto ocurre porque la norma de energía cinemática facial $S_{\text{emg}}(t)$ (curva magenta) queda fluctuando cerca del piso sin superar holgadamente la línea roja discontinua ($U_{\text{alto}} \approx 0.75$), provocando que el decodificador evalúe ventanas espurias o parciales donde el cluster más cercano resulta ser /I/.
   - **Solución requerida a Argon:**
     - Hacer que $U_{\text{alto}}$ y $U_{\text{bajo}}$ sean dinámicos y autocalibrables según el rango de movimiento real del usuario en la sesión (o reducir $U_{\text{alto}}$ a ~0.50–0.55).
3. **Corte Espurio en Silencio: La Norma Cinemática No Está Funcionando (Falta de Contraste y Dinámica):**
   - En la prueba visual se constató la falla de raíz: **la curva magenta de la Norma ($S_{\text{emg}}$) está prácticamente aplastada en el piso**, fluctuando en valores bajos (alrededor de 1–2) sin responder con dinamismo a los movimientos reales de la cara.
   - **Comportamiento físico esperado vs lo que ocurre hoy:**
     - Cuando el sujeto gesticula y abre la boca (como en $-7.8\,\text{s}$ y $-3.4\,\text{s}$ donde CH0 sube a 17 mm), **la norma combinada debería dispararse en un pico gigante** (con una relación señal-a-ruido limpia respecto al reposo) para que la detección del pulso sea trivial e inequívoca mediante el cruce de umbral.
     - En cambio, hoy la norma está atenuada por escalas y pesos incorrectos (`self.s_scales` descalibradas), de modo que los picos reales apenas sobresalen de las fluctuaciones de reposo, y cualquier variación pequeña en silencio cruza el umbral artificial y abre ventanas en el vacío.
   - **Solución requerida a Argon:**
     - Reformular y calibrar rigurosamente la norma cinemática para que actúe como un detector de energía limpio:
       $$S(t) = \sqrt{\sum_{c=0}^{5} w_c \cdot \left(\frac{x_c(t)}{s_c}\right)^2}$$
       con escalas $s_c$ normalizadas al rango fisiológico real del sujeto (usando la desviación de reposo o percentiles de movimiento) y pesos $w_c$ focalizados en los efectores primarios (CH0, CH1, CH2).
     - **Contraste obligatorio:** En reposo la norma debe mantenerse clavada cerca de 0, y ante cualquier emisión fonatoria debe producir un **pico gigante y nítido** que garantice una detección del pulso 100% certera sin falsos disparos interpulso.
