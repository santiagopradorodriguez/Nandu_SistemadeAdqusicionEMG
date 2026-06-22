# Plan de Acción: Entrenamiento de Umbrales por Canal

## Objetivo
Mejorar sustancialmente la frecuencia relativa (predictibilidad) de cada vocal al pasar de un enfoque de **Umbral Común** (un solo valor para todos los músculos) a un **Umbral por Canal** (un valor óptimo independiente para cada músculo).

## Paso 1: Actualización de la Interfaz de Usuario (UI)
* En la pestaña de `Entrenamiento de Umbrales`, crear un nuevo grupo lógico llamado `2. Metodología de Discretización`.
* Añadir un selector (RadioButtons o ComboBox) con las opciones:
  * **Umbral Común:** El método actual (un único valor entre 0.01 y 0.99 para todos los canales).
  * **Umbral por Canal (Maximización):** Habilita la búsqueda en grilla (Grid Search) multidimensional.
* Añadir un selector para la "Resolución" o paso de búsqueda del algoritmo, para evitar que la interfaz se congele si hay demasiados canales. Por defecto, pasos de `0.05` (evalúa ~8,000 combinaciones para 3 canales) o `0.02` (~125,000 combinaciones).

## Paso 2: Motor de Barrido Multidimensional (`training_motor.py`)
* Si el usuario selecciona "Umbral por Canal", el algoritmo abandonará el barrido unidimensional y construirá un espacio de búsqueda usando el producto cartesiano de los umbrales para cada canal (`itertools.product`).
* **Vectorización (Optimización de Velocidad):** 
  Dado que 3 canales con pasos de 0.01 equivalen a 1,000,000 de combinaciones de umbrales, el algoritmo se escribirá utilizando matrices de `NumPy`. Esto permitirá calcular las discretizaciones de todos los pulsos y evaluar las colisiones en menos de un segundo.
* **Sistema de Puntuación (Scoring):**
  Para cada combinación de umbrales (Ej: Masetero=0.60, Cigomático=0.45, Digástrico=0.80):
  1. Se discretizan los picos.
  2. Se extrae la "Moda" (vector más repetido) de cada vocal.
  3. **Regla de Oro:** Si dos vocales comparten la misma Moda (colisión), la combinación de umbrales recibe una puntuación fatal (`-1`) y se descarta.
  4. **Maximización:** Si no hay colisiones, se calcula el promedio de las frecuencias relativas de las modas. La combinación ganadora será aquella que eleve lo más cerca del 100% el promedio global de las vocales.

## Paso 3: Exportación y Reportes
* Modificar la función que genera la tabla (`plot_results_table`) y el archivo LaTeX.
* El título y el nombre de los archivos reflejarán la modalidad elegida (ej. `training_results_table_FiltroAmbos_UmbralesCanal.png`).
* En lugar de mostrar `Umbral seleccionado: 0.55`, la tabla detallará el umbral específico asignado a cada músculo (ej. `Masetero: 0.45 | Digástrico: 0.70`).

## Paso 4: Pruebas y Ajustes
* Ejecutar un análisis usando el modo "Umbral por Canal" sobre las mediciones actuales para verificar que la frecuencia relativa suba considerablemente respecto al <50% obtenido con el umbral común.

## Fundamentos Matemáticos y Computacionales
La velocidad y eficiencia para calcular los intervalos de umbrales óptimos radica en las siguientes técnicas matemáticas y computacionales empleadas en el código:

1. **Producto Cartesiano (Combinatoria Multidimensional):**
   Al independizar los canales, el algoritmo transforma una búsqueda lineal (1D) en una búsqueda de volumen $N$-dimensional (donde $N$ es el número de canales). Matemáticamente se calcula el producto cartesiano de los vectores de posibles umbrales. Ej: $U_{masetero} \times U_{digastrico} \times U_{cigomatico}$. Esto permite mapear el espacio completo de permutaciones usando la librería iteradora `itertools.product`.

2. **Broadcasting y Vectorización Numérica:**
   Para no recorrer pulso por pulso con bucles lentos de Python (lo cual congelaría la interfaz al tener cientos de miles de combinaciones), las comparaciones lógicas (`amplitud >= umbral`) se evalúan a nivel bajo de C mediante `NumPy`. El algoritmo evalúa matrices enteras de picos simultáneamente.

3. **Mapeo de Intervalos Óptimos (Proyección de Hipervolúmenes):**
   En lugar de encontrar un único punto escalar, el código encuentra un clúster de puntos en el hiperespacio que maximizan una función de costo $J(\vec{u})$ (donde $J$ premia frecuencias relativas altas y penaliza colisiones con $-\infty$). Luego, se hace una proyección ortogonal de ese "volumen óptimo" sobre cada eje de los músculos. Esto es lo que arroja los valores mínimos ($u_{min}$) y máximos ($u_{max}$) en los que la clasificación se mantiene estable (la meseta de resolución del ~90% de la que habla el paper).
