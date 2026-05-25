# Explicación del Cálculo de SNR (Relación Señal-Ruido)

Este documento detalla matemáticamente cómo el script `analisis_por_track_integrado.py` calcula el **SNR de Amplitud** para las mediciones de electromiografía (EMG), y qué tipo específico de envolvente se está utilizando.

## 1. La Fórmula Principal

El cálculo final se rige por la siguiente fórmula base:

**`SNR = Amplitud Máxima del Pulso Promedio / Nivel de Ruido Promedio`**

* *(En el código: `snr_manual = max_amp / umbral`)*

Para obtener el numerador y el denominador, la señal cruda atraviesa las siguientes etapas de procesamiento digital.

---

## 2. Tipo de Envolvente Utilizada

Para extraer la "silueta" o forma de la activación muscular sin perder la dinámica real de la fuerza, se emplea una envolvente basada en la **Transformada de Hilbert y Suavizado por Media Móvil (SMA)**.

El proceso de la envolvente se hace en tres pasos (ver función `_compute_env_full` en el código):

1. **Rectificación Total:** Se toma el valor absoluto de la señal previamente filtrada (donde ya pasaron el filtro Notch de 50 Hz y Pasa-Bandas). 
   * `signal_abs = np.abs(signal)`
2. **Transformada de Hilbert:** Se aplica la magnitud de la transformada analítica de Hilbert. Matemáticamente, esto delinea los "picos" rápidos de las ráfagas motoras de la EMG.
   * `env_full = np.abs(hilbert(signal_abs))`
3. **Suavizado (Moving Average):** Finalmente, para obtener una silueta "muscular" limpia que represente la tensión general, se le aplica una Media Móvil mediante una convolución (`np.convolve`). El ancho de este filtro está definido por el parámetro `smooth_ms` (por defecto de 50 milisegundos).
   * *Resultado final:* Una curva suave, estrictamente positiva, que sigue fielmente la amplitud global de reclutamiento muscular del paciente.

---

## 3. Obtención del Numerador: La Amplitud Máxima (`max_amp`)

1. Utilizando los tiempos dados por el metrónomo, el script recorta la **Envolvente Suavizada** en múltiples "ventanas" (pulsos).
2. Alinea todos los pulsos temporalmente y los promedia para crear el **Pulso Promedio** (la forma característica de esa contracción en particular).
3. Se busca simplemente el pico más alto de este pulso promedio y se denomina `max_amp`.

---

## 4. Obtención del Denominador: El Nivel de Ruido (`umbral`)

El nivel de ruido basal del canal se evalúa al comienzo de la grabación (la "fase de ruido" que se graba en los primeros segundos de la medición).

1. El código toma los primeros segundos de la señal (ej. 2 o 5 segundos iniciales configurados en la interfaz) donde se asume que el paciente está totalmente relajado.
2. Le aplica **exactamente la misma envolvente matemática** descrita arriba (Hilbert + Media Móvil).
3. El Nivel de Ruido Base (`umbral`) se establece calculando el promedio matemático (`np.mean()`) de todos los puntos de esa envolvente de reposo.

## Resumen de la Lógica
El SNR que reporta el software no compara "voltajes crudos", compara la **Envolvente de Hilbert Suavizada del máximo esfuerzo muscular**, dividida contra la **Envolvente de Hilbert Suavizada del ruido electromagnético en reposo**, dándonos el factor exacto de "cuántas veces por encima de la línea base estática logró contraerse el músculo".