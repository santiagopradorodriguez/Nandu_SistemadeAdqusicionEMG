# Plan de Trabajo: Entrenamiento de Umbrales Discretos (Assaneo et al. 2013)

Este plan detalla los pasos para implementar un algoritmo de búsqueda de umbrales óptimos que asigne códigos discretos únicos a cada vocal, separando datos de entrenamiento y prueba.

## Paso 1: Preservación de los Parámetros de Envolvente
**Objetivo:** Evitar que el reprocesamiento con distintos tamaños de ventana sobreescriba los datos útiles para el entrenamiento.
* **Acciones:**
  * Modificar el módulo de *Procesamiento Individual*.
  * Guardar el tamaño de la ventana de suavizado/envolvente explícitamente en los metadatos exportados (ej. en el archivo `analisis_results.json`).
  * Cambiar el nombrado de las carpetas o archivos de salida (ej. `analisis_results_env50ms.json`) o generar un subdirectorio, de forma que convivan distintos procesamientos.
  * Actualizar los títulos de los gráficos para que muestren la ventana usada.

## Paso 2: Diseño de la Nueva Interfaz Gráfica
**Objetivo:** Crear el espacio interactivo para configurar el "Training".
* **Acciones:**
  * Añadir una nueva subpestaña (ej: `Entrenamiento Motor`) dentro de `2. ANÁLISIS Y EXTRACCIÓN`.
  * **Asignación de Vocales:** Crear un cuadro dinámico o tabla que permita listar las mediciones seleccionadas y asignarle a cada una qué vocal representa (A, E, I, O, U).
  * **Filtro SNR:** Agregar un Checkbox "Descartar medición por bajo SNR" y un selector numérico (default = 4).
  * Botón de ejecución: `ENTRENAR UMBRALES (TRAIN)`.

## Paso 3: Validación, Carga de Datos y Filtrado por Segmento
**Objetivo:** Asegurar que los datos ingresados al algoritmo sean comparables entre sí y de alta fidelidad, descartando pulsos anómalos.
* **Acciones:**
  * Leer el archivo de resultados principal de cada medición seleccionada.
  * Extraer el tamaño de la ventana de la envolvente (`smooth_ms`).
  * Lanzar un error fatal si se detectan mediciones con tamaños de ventana dispares (ej. mezclar 50ms con 100ms arruinaría la normalización fisiológica).
  * Evaluar el SNR (Signal-to-Noise Ratio) basal. Se relaja el filtro de SNR promedio global de la medición y en su lugar se aplica un riguroso **Filtro SNR por Segmento**:
    * Para cada recorte de la señal (cada pulso), se calcula la Desviación Estándar (RMS) de las muestras que componen el 15% de sus orillas (el ruido basal).
    * Se obtiene la amplitud máxima de ese pulso.
    * Si la relación (Amplitud Max / Ruido RMS) del músculo de mayor contracción en ese pulso **no** supera el límite establecido por el usuario (ej. 4.0), **el pulso entero se descarta**.
  * Cargar y alinear temporalmente los recortes de los canales musculares válidos (ignorando micrófonos u otros sensores auxiliares).

## Paso 4: Extracción y Normalización Biométrica
**Objetivo:** Acondicionar la señal pulso a pulso usando matemática relativa.
* **Acciones:**
  * Para cada recorte de las mediciones válidas, identificar el *ruido inter-pulso posterior* asociado a ese recorte.
  * Restar dicho ruido (offset) a cada uno de los 3 canales de ese recorte, obteniendo un "0 real".
  * Identificar el pico máximo *global* de ese recorte entre los 3 canales.
  * Normalizar los canales válidos de ese recorte dividiéndolos por ese máximo global.
  * Agrupar estos recortes limpios en bolsas o matrices clasificadas por vocal (según la asignación del Paso 2).
  * **(Debug Visual):** Exportar un gráfico `training_debug_norm_{vocal}.png` en la carpeta de la medición que muestre los recortes concatenados en **tiempo real (s)**.
    * Cada gráfico indicará el Nombre del Músculo configurado y el canal.
    * La señal se trazará en negro, pero **el 15% de los bordes de cada pulso se resaltará en Rojo vibrante** para permitirle al investigador verificar visualmente qué datos exactos usó el algoritmo para estimar el ruido promedio de ese segmento, garantizando que no se esté comiendo parte del pico fisiológico.

## Paso 5: Barrido de Umbrales (Threshold Sweep) y Frecuencias Relativas
**Objetivo:** Encontrar qué nivel de corte genera el mapa espacial de bits más limpio y visualizar su comportamiento estadístico.
* **Acciones:**
  * Crear un bucle (loop) que barra los valores de umbral $T$ desde `0.01` hasta `0.99`.
  * Para cada $T$, aplicar el umbral a los recortes y registrar la tupla binaria (Coordenada Discreta) resultante para cada vocal.
  * Seleccionar el umbral que produzca la menor cantidad de colisiones entre vocales.
  * **(Métricas y Reporte):** Para el umbral óptimo encontrado, contabilizar la aparición de TODOS los vectores booleanos en cada vocal.
  * Exportar un gráfico final tipo **Tabla** (`training_results_table.png`) con 5 columnas:
    1. **Vocal**: La letra asignada.
    2. **Total (N pulsos)**: Cantidad total de recortes analizados para esa vocal.
    3. **Moda Global**: El vector booleano más repetido.
    4. **Vector**: Lista de todos los vectores que aparecieron (incluso anomalías).
    5. **Frecuencia Relativa**: Porcentaje de aparición de cada vector listado.
