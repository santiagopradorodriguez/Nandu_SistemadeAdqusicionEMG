# Justificación e Implementación del Filtro Notch

Este documento detalla la implementación del filtro digital Notch utilizado en el sistema de adquisición y análisis de señales EMG para eliminar la interferencia de la red eléctrica (50 Hz). En todos los códigos del proyecto se utiliza la librería **SciPy** (específicamente el módulo `scipy.signal`).

## 1. Implementación según el entorno de ejecución

Todos los scripts utilizan `scipy.signal.iirnotch` para crear la "forma" del filtro de 50 Hz, pero la aplicación práctica varía si es en tiempo real o en post-procesamiento.

### Tiempo Real (`CodigoUnificador_integrado.py`)
* **Diseño:** Se usa `signal.iirnotch(f0, Q, fs)` que genera los coeficientes `b, a`. Luego, por estabilidad matemática en tiempo real, se convierten a secciones de segundo orden (SOS) usando `signal.tf2sos(b, a)`.
* **Aplicación:** Se usa `signal.sosfilt()`.
* **Justificación:** Se usa `sosfilt` en lugar de `filtfilt` porque la adquisición se hace "por pedazos" (chunks) en tiempo real y esta función permite guardar el "estado" (`zi`) del filtro entre un bloque de datos y el siguiente, garantizando que la señal sea continua sin cortes ni saltos.

### Post-procesamiento
Utilizado en: `analisis_por_track_integrado.py`, `plotter_calibrado.py` y `visor_csv_interactivo.py`.
* **Diseño:** Se usa `signal.iirnotch(f0, Q, fs)` para generar los coeficientes `b, a`.
* **Aplicación:** Se usa `signal.filtfilt(b, a, signal)`.
* **Justificación:** `filtfilt` es ideal para post-procesamiento porque aplica el filtro hacia adelante y hacia atrás sobre la señal completa ya grabada. Esto tiene la gran ventaja de que **no introduce ningún desfase (phase shift)** en la señal EMG, manteniendo los picos exactamente en el mismo instante de tiempo fisiológico original.

---

## 2. Parámetros del Filtro (`b` y `a`)

La función principal de diseño es `b, a = signal.iirnotch(f0, Q, fs)`. En el procesamiento de señales digitales, `b` y `a` son los coeficientes del filtro digital IIR (Infinite Impulse Response):

* **`b` (Coeficientes del numerador):** Multiplican a las muestras de la señal de entrada (los valores crudos de EMG recién leídos).
* **`a` (Coeficientes del denominador):** Multiplican a las muestras de la señal de salida calculadas previamente. Esto le otorga la característica "IIR", dotando al filtro de "memoria" o retroalimentación.

La ecuación de diferencias que calcula cada nuevo punto de la señal filtrada sigue esta forma base:
> `y[n] = (b[0]*x[n] + b[1]*x[n-1] + ...) - (a[1]*y[n-1] + a[2]*y[n-2] + ...)`

Estos coeficientes son los que posteriormente se pasan a `filtfilt()` o se transforman a formato SOS.

---

## 3. Factor de Calidad (`Q`) y Ancho de Banda

En todos los scripts de este proyecto, el factor de calidad **`Q` está fijado temporalmente en 2.0**.

### ¿Qué significa Q = 2.0 en la práctica?
El factor de calidad `Q` define qué tan "agudo" o "estrecho" es el corte del filtro. La fórmula que relaciona `Q`, la frecuencia central (`f0` = 50 Hz) y el ancho de banda (`BW` = el rango de frecuencias atenuadas) es:

$$ BW = \frac{f0}{Q} $$

Aplicando los valores del proyecto:

$$ BW = \frac{50.0 \text{ Hz}}{2.0} = 25.0 \text{ Hz} $$

### Justificación fisiológica
Esto significa que el filtro Notch elimina un bloque de frecuencias de **25.0 Hz de ancho** centrado en 50 Hz. Es decir, atenúa fuertemente todo el espectro que recae entre **37.5 Hz y 62.5 Hz**.

* **¿Por qué este valor?** Un valor de Q tan bajo permite absorber fluctuaciones severas en la red eléctrica, aunque puede atenuar algunas frecuencias útiles musculares cercanas a los 50Hz.