# Informe Detallado: Procesamiento Matemático de la Señal EMG

Este documento detalla el paso a paso del procesamiento digital de señales (DSP) aplicado a los registros de electromiografía (EMG) mediante el script `analisis_por_track_integrado.py`. Se incluye la fundamentación matemática de cada etapa, desde la lectura de los datos crudos hasta el cálculo de métricas como el SNR.

---

## Fase 1: Lectura y Calibración Física (De Digital a Microvoltios)

El objetivo de esta fase es convertir los valores digitales normalizados (del archivo `.wav`) de nuevo a sus unidades físicas reales ($\mu V$), compensando la ganancia del hardware.

### 1.1. Restauración de la Amplitud en Voltios
El archivo `.wav` almacena la señal normalizada en el rango $[-1.0, 1.0]$. Para recuperar el voltaje original, el script busca el archivo `.csv` original y extrae el **factor de calibración** (el valor máximo absoluto registrado).

$$ V_{out}(t) = \text{Señal}_{norm}(t) \times \max(|V_{csv}|) $$

### 1.2. Compensación de Ganancia Analógica y Conversión a $\mu V$
El sistema de hardware utiliza un amplificador cuya ganancia depende de una resistencia fija ($R_{fija} = 49400 \ \Omega$) y la resistencia específica del electrodo ($R_{electrodo}$), la cual se lee del archivo `metadata.json`.

1. **Cálculo de la Ganancia del hardware:**
   $$ \text{Ganancia} (G) = 1 + \frac{R_{fija}}{R_{electrodo}} $$

2. **Cálculo del voltaje real del músculo (Entrada) en microvoltios:**
   $$ \text{Señal}_{\mu V}(t) = \left( \frac{V_{out}(t)}{G} \right) \times 10^6 $$

---

## Fase 2: Filtrado Digital (Acondicionamiento)

La señal en microvoltios pasa por una cascada de filtros IIR (Infinite Impulse Response) para eliminar artefactos. El script utiliza la función `filtfilt` de SciPy, que filtra la señal hacia adelante y hacia atrás. Esto es crucial porque **anula el desfase temporal (zero-phase distortion)**, manteniendo los picos exactamente en su instante fisiológico.

La ecuación de diferencias general de estos filtros es:
$$ y[n] = \sum_{i=0}^{N} b_i x[n-i] - \sum_{j=1}^{M} a_j y[n-j] $$

1. **Filtro Pasa-Altos (High-pass):** Típicamente a 20 Hz (Butterworth de 4to orden). Elimina el *baseline wander* (fluctuaciones lentas por movimiento de cables o respiración), centrando la señal perfectamente en cero.
2. **Filtro Notch:** A 50 Hz con factor de calidad $Q = 2.0$. Elimina la interferencia de la red eléctrica.
3. **Filtro Pasa-Bajos (Low-pass):** Típicamente a 500 Hz (Butterworth de 4to orden). Elimina ruido estático de alta frecuencia y previene *aliasing*.

---

## Fase 3: Extracción de la Envolvente (Dinámica Muscular)

Para analizar la "fuerza" o volumen de reclutamiento muscular a lo largo del tiempo, se extrae la envolvente de la señal filtrada. Esto se hace en tres pasos matemáticos:

### 3.1. Rectificación Total (Módulo)
Se toma el valor absoluto de todas las muestras, volviendo positivos los hemiciclos negativos.
$$ S_{abs}(t) = |S_{filtrada}(t)| $$

### 3.2. Transformada Analítica de Hilbert
Se calcula la envolvente analítica aplicando la Transformada de Hilbert ($\mathcal{H}$). Esto extrae la amplitud instantánea de la señal, delineando los contornos superiores de las ráfagas motoras de alta frecuencia.
$$ \text{Env}_{hilbert}(t) = \sqrt{S_{abs}^2(t) + \mathcal{H}\{S_{abs}(t)\}^2} $$

### 3.3. Suavizado por Media Móvil (SMA)
Para obtener una curva suave que represente la tensión general (y no picos nerviosos individuales), se aplica una convolución con una ventana rectangular ($w$) de longitud $N$ (definida por el parámetro `smooth_ms`, por defecto 50 ms).
$$ \text{Env}_{full}[n] = \frac{1}{N} \sum_{k=0}^{N-1} \text{Env}_{hilbert}[n-k] $$

---

## Fase 4: Estimación del Ruido Basal (El "Umbral")

Antes de que comiencen las contracciones voluntarias, se graba una ventana de silencio (típicamente los primeros 2 segundos, definido por `noise_seconds`).

Se toma este segmento inicial de la envolvente ($Env_{ruido}$) y se calcula su media matemática. Este valor se convierte en el **Umbral Basal** o Nivel de Ruido Promedio de referencia.
$$ \text{Umbral} (\mu_{ruido\_inicial}) = \frac{1}{M} \sum_{i=1}^{M} \text{Env}_{ruido}[i] $$

---

## Fase 5: Segmentación y Detección de Pulsos

El script utiliza la frecuencia del metrónomo (BPM) guardada en los metadatos para conocer la periodicidad teórica de las contracciones.

1. **Cálculo del Período:** $T = \frac{60}{\text{BPM}}$ (en segundos).
2. **Ventaneo:** La señal se divide en bloques consecutivos de duración $T$.
3. **Búsqueda del Pico Máximo:** Dentro de cada bloque $i$, se busca el índice $t_{max}$ donde la envolvente sea máxima:
   $$ t_{max, i} = \arg\max_{t \in [i \cdot T, \ (i+1) \cdot T]} (\text{Env}_{full}(t)) $$
4. **Validación:** El valor del pico $\text{Env}_{full}(t_{max, i})$ debe ser mayor que el Umbral dinámico calculado en la Fase 4.
5. **Recorte:** Se extrae un segmento de la señal original alrededor de $t_{max, i}$ (definido por `pre_samples` y `post_samples`).

---

## Fase 6: Cálculo del Pulso Promedio y Estadística

Todos los segmentos válidos extraídos se apilan y alinean utilizando su pico máximo como centro. Sea $P$ la matriz donde cada fila $k$ es el segmento extraído de un pulso, y $N_p$ el número total de pulsos válidos.

1. **Pulso Promedio ($\bar{p}$):**
   $$ \bar{p}[n] = \frac{1}{N_p} \sum_{k=1}^{N_p} P_{k,n} $$

2. **Desviación Estándar ($\sigma$):**
   $$ \sigma[n] = \sqrt{ \frac{1}{N_p - 1} \sum_{k=1}^{N_p} (P_{k,n} - \bar{p}[n])^2 } $$

3. **Error Estándar de la Media (SEM):** Define la incertidumbre o banda de error del promedio.
   $$ \text{SEM}[n] = \frac{\sigma[n]}{\sqrt{N_p}} $$

---

## Fase 7: SNR, Fatiga y Evolución Temporal (`evolucion.png`)

En esta fase se calcula cómo se comportan la señal y el ruido dinámicamente a lo largo de los minutos que dura la medición.

### 7.1. SNR Global (Por Amplitud del Promedio)
Se evalúa la magnitud del reclutamiento muscular máximo respecto al ruido basal. Se busca el valor más alto del Pulso Promedio general y se divide por la media de la ventana de silencio.
$$ \text{SNR}_{global} = \frac{\max(\bar{p})}{\mu_{ruido\_inicial}} $$

### 7.2. Ruido Inter-pulso Normalizado (Relajación y Calidad)
Para saber si el músculo logra relajarse entre contracciones o si el electrodo pierde contacto y captura interferencia, se analiza el valle de silencio **entre** cada pulso.

Para cada pulso válido $k$, se toma una ventana estrecha de tiempo en el punto medio exacto de relajación (el valle entre el pulso actual y el anterior). Se calcula la media absoluta de esa ventana $\mu_{valle, k}$ y se normaliza dividiéndola por el ruido basal inicial de los primeros 2 segundos ($\mu_{ruido\_inicial}$).

$$ \text{Ruido Inter-pulso}_\% [k] = \left( \frac{\mu_{valle, k}}{\mu_{ruido\_inicial}} \right) \times 100 $$

* **Si se mantiene $\approx 100\%$**: La señal es estable y el músculo descansa bien.
* **Si crece progresivamente**: Indica que el electrodo pudo haberse despegado, ganando estática (Deriva de Ruido), o el paciente no logra relajar el músculo (tensión residual por fatiga).

### 7.3. SNR Promedio Acumulado (Curva de Evolución)
En lugar de mostrar una nube de puntos caótica con el SNR individual de cada contracción (que puede variar por esfuerzo voluntario), se grafica la tendencia acumulativa para identificar agotamiento muscular progresivo.

1. Se calcula el **SNR individual** para cada pulso $k$:
   $$ \text{SNR}_{k} = \frac{\max(\text{Env}_{segmento, k})}{\mu_{ruido\_inicial}} $$

2. Se calcula el **Promedio Acumulado** en el instante de tiempo del pulso $N$:
   $$ \text{SNR}_{acumulado}[N] = \frac{1}{N} \sum_{k=1}^{N} \text{SNR}_{k} $$
   *(Acompañado de su error estándar en las barras de error de la gráfica $\pm \frac{\sigma}{\sqrt{N}}$)*.

Si la curva de `SNR Acumulado` tiende a la baja de forma consistente (Caída de SNR), es un marcador cuantitativo directo de que el músculo está perdiendo capacidad de reclutamiento (fatiga) a medida que avanza la serie.
