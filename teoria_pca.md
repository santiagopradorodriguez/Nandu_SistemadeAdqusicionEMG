# Teoría: Análisis de Componentes Principales (PCA) en Electromiografía

## 1. ¿Qué es PCA?
El **Análisis de Componentes Principales (PCA)** es un algoritmo fundamental en el aprendizaje automático y la estadística. A diferencia de UMAP (que es un algoritmo no lineal enfocado en preservar la topología de la variedad local), **PCA es una transformación estrictamente lineal**.

Su objetivo principal es tomar un conjunto de datos altamente dimensional (por ejemplo, nuestros vectores musculares de 300 muestras) y encontrar un nuevo sistema de coordenadas (ejes) que explique la **máxima varianza posible** en los datos. Estos nuevos ejes se denominan "Componentes Principales".

### ¿Por qué las distancias en PCA son 100% interpretables?
Como PCA realiza rotaciones y proyecciones ortogonales en el espacio euclidiano (es decir, una multiplicación de matrices simple), el espacio resultante mantiene proporcionalidad geométrica estricta. Si dos puntos están separados por una distancia euclidiana de $X$ en el gráfico PCA de alta varianza, esa distancia refleja una diferencia real y mensurable de fuerza muscular. En UMAP, las distancias visuales entre clústeres a veces están distorsionadas para favorecer la agrupación.

## 2. Varianza Explicada
La magia matemática detrás de PCA radica en los *autovalores* (eigenvalues) de la matriz de covarianza de los datos. Cada Componente Principal tiene asociado un porcentaje que indica cuánta información total del conjunto original de datos (su varianza) es capaz de resumir. 
Por ejemplo, si la "Componente 1" retiene el 70% de la varianza y la "Componente 2" el 20%, un gráfico 2D usando solo estas dos componentes nos está mostrando el **90% de toda la información electromiográfica** existente en los 300 puntos de tiempo originales, perdiendo solamente un 10% de detalle (que suele ser ruido).

## 3. El Pipeline Perfecto: PCA $\rightarrow$ UMAP
Un enfoque estándar y altamente recomendado (conocido como *Dimensionality Reduction Pipeline*) es combinar lo mejor de ambos algoritmos:

*   **Problema de UMAP puro:** Si a UMAP se le entrega el vector crudo de 300 dimensiones, la redundancia temporal y el ruido eléctrico residual pueden afectar su capacidad de construir grafos de vecindad eficientes ("Maldición de la Dimensionalidad").
*   **La Solución PCA:** Primero le pedimos a PCA que reduzca las 300 dimensiones a un número intermedio (por ejemplo, **15 componentes principales**). Como es lineal, PCA va a concentrar la señal verdadera (varianza alta) en estas 15 columnas, y va a descartar automáticamente el ruido blanco estocástico en las 285 columnas descartadas.
*   **Sinergia:** Luego, le pasamos estas 15 componentes súper-densas y libres de ruido a **UMAP**. UMAP ahora puede concentrarse exclusivamente en desenredar las relaciones no-lineales complejas entre las diferentes vocales.

## 4. PCA "Supervisado"
Por naturaleza matemática, PCA es **No Supervisado**. No sabe qué es una "A" ni qué es una "U". Simplemente busca varianza geométrica.
Sin embargo, en este orquestador incluimos una opción "Supervisada". Esto **no altera** cómo se calculan las componentes principales, sino que nos permite:
1. Colorear los puntos proyectados según su vocal original para visualizar si el movimiento muscular genera varianza por sí mismo.
2. Calcular los **Centroides (Promedios)** de cada vocal en el espacio PCA.
3. Extraer métricas (Silhouette Score y distancias entre clústeres) para evaluar cuantitativamente qué vocales se separan más en el mapa de activación.
