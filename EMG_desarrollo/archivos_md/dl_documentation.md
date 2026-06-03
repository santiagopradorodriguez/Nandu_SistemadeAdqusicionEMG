# Documentación del Pipeline de Deep Learning

Este documento explica el flujo de procesamiento de datos y la generación de tensores de PyTorch para el modelo de Deep Learning implementados en `dl_data_pipeline.py`.

## 1. Ingestión y Limpieza Acústica (Preprocesamiento DSP)
El proceso comienza analizando grabaciones en formato CSV empleando una arquitectura de procesamiento por lotes (Batch Processing). Para cada archivo (que representa una repetición o sílaba particular):
1. **Filtro Pasa-Banda**: Se aplica un filtro Butterworth de fase cero (`filtfilt`) entre 20 Hz y 500 Hz para aislar la banda útil del EMG, eliminando frecuencias bajas generadas por movimientos y altas frecuencias asociadas a ruidos.
2. **Filtro Notch**: Se utiliza un filtro Notch IIR (`iirnotch`) centrado en 50 Hz (con un factor de calidad $Q=30$) para remover la interferencia proveniente de la línea eléctrica.
3. **Envolvente RMS**: Se convoluciona el cuadrado de la señal con una ventana móvil de 50 ms y se extrae la raíz cuadrada para obtener la amplitud de la activación muscular suavizada.

## 2. Segmentación y Alineación (Master-Slave)
Dado que los músculos fonatorios operan de manera conjunta y sincrónica, el pipeline unifica el contexto temporal utilizando el canal 1 como *Master*:
- Se encuentra el índice de activación máxima en el canal Master (`np.argmax`).
- Se recorta una ventana constante de exactamente 1 segundo de duración (10000 muestras a $f_s = 10000$ Hz) centrada de forma simétrica alrededor del pico hallado.
- Esta misma ventana temporal se usa de forma idéntica para recortar todos los demás canales (los *Slaves*), garantizando así una perfecta alineación de fase entre los diferentes canales que formarán el tensor.

## 3. Estandarización Tensorial y Normalización Min-Max
Las redes neuronales modernas (como Autoencoders, CNNs y arquitecturas recurrentes) entrenadas sobre señales sEMG requieren datos temporalmente estacionarios y con un formato estructurado estricto.

1. **Remuestreo (Resampling)**: La ventana de 10000 muestras original se submuestrea a un tamaño definido `TARGET_SAMPLES = 500` utilizando el método de Transformada de Fourier implementado en `scipy.signal.resample`. Esto comprime la información temporal conservando adecuadamente las dinámicas de baja frecuencia características de la envolvente.
2. **Normalización Min-Max**: Cada canal de la señal se escala independientemente al rango continuo de $[0, 1]$ a partir de sus mínimos y máximos locales:
   $$
   x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}
   $$
   Esta normalización asegura que los gradientes de la retropropagación en PyTorch mantengan estabilidad, previniendo su explosión o desvanecimiento y manteniendo las activaciones en un dominio comparable inter-canal.

## 4. Estructura del Tensor y Persistencia
- Las características preprocesadas y resampleadas de cada canal son apiladas para formar una matriz bidimensional de Numpy con la forma estructurada de `(3, 500)` (asumiendo 3 canales registrados).
- Estas matrices multidimensionales se persisten localmente en disco en formato binario estructurado `.npy` (dentro del directorio `datasets_ml/`). Simultáneamente, se mantiene un índice maestro en formato JSON (`dataset_index.json`) que mapea cada uno de estos tensores a un `label_id` numérico determinístico basado en la categoría de la etiqueta de la clase (la sílaba pronunciada).

## 5. Dataloader de PyTorch
La clase de conjunto de datos `EMGDataset` extiende de la interfaz de conjunto de datos fundamental de PyTorch (`torch.utils.data.Dataset`), estableciendo una eficiente tubería de datos para la ingesta en la GPU durante las iteraciones de entrenamiento:
- **`__getitem__`**: En tiempo de ejecución y bajo demanda temporal, recupera desde el disco físico el archivo binario correspondiente `.npy`. Este comportamiento garantiza un bajo consumo de memoria RAM, siendo altamente escalable para bases de datos extensas.
- **Transformación de tipos a tensores de GPU**: El array base de Numpy se convierte explícitamente a un tensor de punto flotante de 32 bits (`torch.float32`), en tanto la etiqueta categórica correspondiente es instanciada como un entero largo de tipo tensor (`torch.long`). Esto resulta indispensable como requisito técnico de funciones objetivo como `CrossEntropyLoss`.
- **Inferencia de Lotes (Batching)**: La función de retorno estandarizada produce finalmente la tupla elemental `(tensor, label)`, perfectamente dimensionada para que el objeto general `DataLoader` la pueda compactar con agilidad en sus *minibatches* de entrenamiento.
