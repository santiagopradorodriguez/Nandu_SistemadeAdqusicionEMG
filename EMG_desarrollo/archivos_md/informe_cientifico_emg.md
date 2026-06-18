# Informe Científico: Adquisición y Extracción de Features EMG para Interfaz de Habla Silenciosa (SSI)

Este documento detalla los fundamentos metodológicos y biomecánicos detrás del procesamiento de señales de Electromiografía (EMG) de superficie en el contexto del proyecto Ñandú LSD. Las decisiones arquitectónicas sobre la extracción de características (features) y el entrenamiento de modelos de Machine Learning se basan estrictamente en la necesidad de preservar el paradigma de una verdadera Interfaz de Habla Silenciosa.

## 1. Biomecánica de la Sinergia Vocal: Importancia del Desfase (Lag) y Correlación Cruzada

El proceso de fonación y articulación del habla humana no es el resultado de la contracción aislada de un solo músculo, sino de una **sinergia biomecánica compleja** que involucra la activación coordinada y secuencial de múltiples grupos musculares faciales y del tracto vocal (e.g., orbicular de los labios, masetero, cigomático mayor, mentoniano). 

Para que un modelo de Deep Learning o Machine Learning pueda distinguir eficazmente entre distintos fonemas (por ejemplo, diferenciar la vocal "A", que requiere una apertura mandibular significativa, de la vocal "U", que requiere protrusión labial), no basta con medir la amplitud de contracción aislada en cada canal EMG. Es imperativo capturar la **dinámica espaciotemporal** de la articulación.

### El Rol del Desfase (Lag) y la Correlación Máxima

1. **Firma Temporal Única:** Cada fonema posee una "firma" o patrón de activación donde ciertos músculos preceden a otros. Al calcular la correlación cruzada entre pares de canales EMG (músculos), extraemos dos métricas vitales:
   - **Correlación Máxima:** Indica el grado de similitud morfológica y reclutamiento compartido entre dos músculos durante la articulación de un fonema.
   - **Desfase (Lag):** Cuantifica el retraso temporal exacto entre la activación del músculo A y el músculo B. 

2. **Separabilidad en el Espacio PCA:** Al proyectar las características extraídas mediante Análisis de Componentes Principales (PCA), las métricas puramente basadas en amplitud (RMS, MAV) tienden a generar solapamientos (clusters confusos) entre fonemas que requieren un esfuerzo muscular general similar. Sin embargo, al introducir el **Lag** y la **Correlación Máxima inter-muscular**, añadimos dimensiones ortogonales que representan la *secuencia biomecánica*. El modelo deja de preguntar "¿cuánta fuerza se hizo?" para preguntar "¿en qué orden exacto se movieron la mandíbula y los labios?". Esto permite que los fonemas se separen en clusters densos y bien definidos en el espacio de características (PCA), reduciendo drásticamente la tasa de error de clasificación.

---

## 2. Prevención de Fuga de Datos (Data Leakage): El Paradigma de Interfaz de Habla Silenciosa

Una Interfaz de Habla Silenciosa (Silent Speech Interface - SSI) tiene como objetivo fundamental decodificar el lenguaje directamente desde los bioseñales articulatorios (EMG), permitiendo la comunicación **incluso cuando no hay emisión acústica** (habla silenciosa o pacientes con disfunciones laríngeas).

En nuestro setup experimental, se utiliza un micrófono direccional junto con los electrodos EMG para grabar el audio real producido por el sujeto durante la adquisición de datos vocalizados. Sin embargo, el tratamiento de esta señal acústica debe ser metodológicamente inmaculado para evitar invalidar todo el proyecto.

### ¿Por qué es metodológicamente inaceptable incluir la correlación del micrófono en las features?

Incluir en la matriz de features métricas que dependan del micrófono (como la correlación cruzada entre un canal EMG y el canal de audio acústico) constituye un caso de manual de **Data Leakage (Fuga de Datos)**.

1. **Dependencia Artificial en Entrenamiento:** Si el modelo de Machine Learning recibe como feature la correlación entre el músculo y el audio, descubrirá rápidamente que esta variable tiene un altísimo poder predictivo (dado que el sonido es el resultado directo del movimiento). El modelo le asignará un peso desproporcionado, ignorando las sutilezas de las correlaciones inter-musculares.

2. **Falla Catastrófica en Inferencia (Silent Speech):** El objetivo final del sistema es operar en silencio total. Durante la etapa de inferencia (uso real), el usuario no emitirá sonido. Por lo tanto, el micrófono registrará silencio puro o ruido ambiente. Si el modelo fue entrenado asumiendo que dispondría de una correlación válida EMG-Micrófono, al recibir silencio en inferencia, sus predicciones colapsarán por completo. 

3. **Invalidación del Paradigma SSI:** Al depender de variables acústicas, el sistema deja de ser una "Interfaz de Habla Silenciosa" y se convierte en un simple clasificador de audio glorificado y redundante.

### El Rol Correcto del Micrófono: 'Ground Truth' y Alineación

Para preservar la validez del modelo, la señal del micrófono tiene una y solo una función temporal en el pipeline de procesamiento: **Actuar como "Master" para la alineación temporal.**

*   Se utiliza el pico de amplitud o la envolvente del micrófono (Ground Truth) para determinar con precisión de milisegundos cuándo ocurrió el evento fonético real.
*   Los canales EMG actúan como "Slaves" (Esclavos) que se alinean y recortan en función del timestamp provisto por el micrófono.
*   Una vez que los datos EMG están correctamente segmentados y alineados en torno al inicio de la fonación, **la señal del micrófono debe ser completamente descartada**.

Las matrices de features (`X_train`, `X_test`) que se alimentan al algoritmo de clasificación (SVM, Random Forest, Redes Neuronales) deben estar compuestas **únicamente** por descriptores intrínsecos del EMG (MAV, Zero Crossing, Correlaciones inter-EMG, Lags inter-EMG). De esta manera, aseguramos un entrenamiento metodológicamente robusto y preparamos al modelo para desempeñarse con éxito en el mundo real del habla silenciosa.
