# Know-How: Generación de Vectores para PCA y Visualización
*(Documento de transferencia desde la rama `master`)*

Este documento detalla cómo el script `generador_pca_tensorial.py` construye el DataFrame (los vectores de características) y cómo es posible reconstruir la visualización de las mediciones de manera precisa.

## 1. Alineación Estricta por Ventana Física
A diferencia de usar `numpy.roll` o correlación cruzada sobre señales ya cortadas, la estrategia actual funciona **cortando la señal desde cero en base al micrófono**:
1. Busca el pico principal del audio en el `canal_3` usando `scipy.signal.find_peaks`.
2. Define un tamaño de ventana físico: **40% del tiempo antes del pico y 60% después del pico**.
3. Se recorta ese rango exacto de índices `[inicio:fin]` en el `canal_0`, `canal_1` y `canal_2`. Esto garantiza que los 3 músculos estén perfectamente alineados en el tiempo respecto al sonido.

## 2. Reducción de Ruido Dinámica
En vez de restar un ruido estático, el script mira un "pedacito" de señal *justo antes* de que empiece la ventana y otro *justo después*. Promedia esa basura (ruido inter-pulso) y se la resta a la ventana de los músculos para que la línea base quede planchada en cero (`np.maximum(señal - ruido, 0)`).

## 3. Normalización Global y Remuestreo (FFT)
1. **Normalización:** Se busca el "Pico Supremo" (el valor más alto entre los 3 canales de esa ventana). Los 3 canales se dividen por ese valor. Así se conserva la proporción de qué músculo hizo más fuerza.
2. **Remuestreo (La Magia):** Como a veces la persona habla más lento o más rápido, la ventana en bruto puede tener 1500 muestras o 2000 muestras. Para que la IA (PCA/UMAP) pueda comparar peras con peras, se usa `scipy.signal.resample(canal, 100)` para estirar o comprimir la curva de cada canal a **exactamente 100 muestras** (`TARGET_LEN = 100`).

## 4. Armado del Vector (DataFrame)
Al tener 3 canales de exactamente 100 muestras cada uno, se genera una matriz de `3 x 100`. Para guardar esto en un `.csv` (DataFrame) y que sea digerible por algoritmos clásicos, la matriz se "aplana" en un vector de **300 columnas**:
*   `Columna 0 a 99`: Datos del Canal 0.
*   `Columna 100 a 199`: Datos del Canal 1.
*   `Columna 200 a 299`: Datos del Canal 2.

## 5. ¿Cómo se reconstruye para visualizar?
Como la estructura matemática es fija y conocida (100 muestras por canal), si querés visualizar una medición desde el CSV, el proceso es inverso y trivial:
1. Agarrás una fila del DataFrame (las 300 columnas).
2. Cortás la fila en pedazos de a 100: el primer pedazo es el músculo 1, el segundo el músculo 2, etc.
3. Ploteás los 3 pedazos superpuestos en un gráfico cuyo eje X va de 0 a 100.
4. **¿Dónde está el pico acústico?** Como al principio se definió que la ventana tenía un 40% de muestras "pre-pico", sabés con certeza matemática que el sonido original ocurrió exactamente en el **índice X = 40** del gráfico. Los picos musculares se verán alineados de forma natural alrededor de ese punto.

---
*Nota: Este documento fue generado para portar el conocimiento a otras ramas (branches) y estandarizar la generación de vectores.*
