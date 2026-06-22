# Plan de Clustering con UMAP/SUMAP para Señales EMG

## Motivación
El objetivo de implementar UMAP (Uniform Manifold Approximation and Projection) es evaluar visual y matemáticamente qué tan separables son las vocales basándonos en la información recolectada de las mediciones de EMG. A diferencia de PCA (que asume relaciones lineales), UMAP es excelente para encontrar relaciones no lineales y proyectar la estructura local de los datos de alta dimensionalidad en un espacio 2D comprensible.

## Construcción del Vector de Características (Feature Vector)
Para que el algoritmo pueda agrupar los latidos (pulsos), debemos representar cada pulso como un vector $X_i$. Se ofrecen dos enfoques configurables en la UI:

1. **Baja Dimensionalidad (Picos Máximos):**
   - **Formato:** $X_i = [Max(Canal_1), Max(Canal_2), Max(Canal_3)]$
   - **Ventaja:** Rápido, fácil de interpretar computacionalmente, se alinea directo con la lógica binaria del umbral actual.
   - **Desventaja:** Pierde toda la información temporal y morfológica de la onda.

2. **Alta Dimensionalidad (Onda Completa / Flatten):**
   - **Formato:** Se concatena la onda normalizada completa de cada canal. Si un pulso tiene 150 muestras, el vector será de $150 \times 3 = 450$ dimensiones.
   - **Ventaja:** UMAP analiza la morfología y el tiempo del pulso, detectando asimetrías o formas sutiles que diferencian las vocales y que los picos ignoran.
   - **Desventaja:** Más sensible al ruido ambiental si el filtrado SNR previo no fue estricto.

## Hiperparámetros Clave
- `n_neighbors`: Controla el balance entre la topología local y global. Valores bajos (ej. 5-15) enfatizan micro-grupos, valores altos (ej. 30-50) muestran la macro-estructura general.
- `min_dist`: Qué tan juntos permite UMAP agrupar físicamente los puntos en el gráfico 2D. Afecta la "densidad" visual del clúster.
- `metric`: La forma de medir distancias en alta dimensión (Usaremos distancia Euclidiana por defecto para EMG).

## Variantes
- **UMAP No Supervisado:** El algoritmo agrupa sin saber de qué vocal proviene cada pulso. Sirve para descubrir si las señales musculares de distintas vocales se agrupan de forma "natural" por pura física.
- **SUMAP (Supervisado):** Se le inyectan las etiquetas (A, E, I, O, U) a la matemática interna. Esto rompe el espacio obligando a separar lo más posible a las distintas vocales, sirviendo como una antesala directa para demostrar que los datos son viables para entrenar un clasificador estadístico predictivo (Machine Learning).

## Evaluación y Reporte de Resultados
Para dar solidez a los experimentos, la herramienta genera un set de auditoría automático:

1. **Métrica (Silhouette Score):** En lugar de confiar sólo en la vista, se calcula el `silhouette_score` (de -1 a 1) sobre la proyección 2D final para cuantificar objetivamente qué tan perfecta es la separación.
2. **Exportación Cruda (CSV):** Justo antes de ejecutar el algoritmo, el script exporta la matriz completa de características $X$ y $Y$ a un archivo `.csv` para permitir revisiones manuales de los números crudos.
3. **Perspectivas Visuales:** Se renderizan ambas proyecciones: un plano 2D clásico y un gráfico 3D (`Axes3D`) para observar la profundidad del manifold. Todo usando la paleta base de Matplotlib (`tab10`) adaptada a `[tab:red, tab:green, tab:blue, tab:purple, tab:orange]` para alto contraste.
4. **Almacenamiento por Sesión:** Todos los outputs (PDF, PNGs, CSV) no se tiran a una carpeta general, sino que se ubican en una subcarpeta `UMAP/` generada adentro de la fecha de la sesión que se esté analizando, garantizando la trazabilidad.
