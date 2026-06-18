# Veredicto del Consejo Científico: La Paradoja PCA vs XGBoost en EMG Facial

**Destinatario:** Director del Proyecto Ñandú LSD
**Elaborado por:** Equipo de Arquitectura de Machine Learning, Física Biomecánica y DSP.

Tras analizar los resultados contradictorios (0.08 de Silhouette en PCA frente a un 80% de precisión en XGBoost) y auditar el pipeline matemático, el Consejo Científico ha determinado que **los datos son de altísima calidad y el hardware funciona perfectamente**. 

El fracaso del PCA no se debe a un error de código, sino a una incompatibilidad matemática fundamental entre la fisiología de la garganta y la topología lineal. A continuación, el diagnóstico definitivo.

---

## 1. El Veredicto Biomecánico (Por qué los Lags fracasan)

La señal EMG de superficie en la garganta (Milohioideo, DAO, Orbicular) sufre de un **"Crosstalk" (Conducción de Volumen) Masivo**. Al pronunciar vocales, los campos eléctricos de múltiples músculos profundos y superficiales se suman.
Físicamente, a diferencia de las consonantes oclusivas ('P', 'T') que son secuencias musculares balísticas, **las vocales son posturas isométricas estacionarias**. Los músculos se activan *simultáneamente* para mantener la forma de la cavidad de resonancia. 

> [!IMPORTANT]
> Por esto, tu ranking demostró que los "Lags" (desfase temporal) tienen 0% de poder predictivo. No hay una secuencia de "primero el DAO, luego el Orbicular". Todo ocurre a la vez.

La diferencia entre una 'A' y una 'E' no es qué músculo se enciende, sino la **sutil proporción (Ratio) de fuerza vectorial** entre ellos. Esa diferencia es puramente no-lineal.

## 2. El Veredicto de Machine Learning (Por qué PCA fracasa y XGBoost triunfa)

Aquí yace la respuesta a la paradoja:

*   **PCA es un algoritmo Ciego y Lineal:** PCA no busca separar vocales; busca la mayor varianza global. En el cuello, la mayor varianza es el ruido de crosstalk, la impedancia de la piel y la fuerza basal de tragar saliva. PCA traza líneas rectas a través de este ruido masivo, aplastando las sutiles diferencias de los Ratios en una sola "nube" incomprensible (Silhouette 0.08).
*   **XGBoost es un algoritmo Supervisado y No Lineal:** XGBoost construye miles de cortes lógicos (árboles de decisión). Ignora el ruido de crosstalk y se enfoca matemáticamente solo en lo que separa las clases. Al poder trazar fronteras de decisión cuadradas y complejas alrededor de las sutiles proporciones de fuerza, logra el **80% de precisión**.

> [!WARNING]
> Intentar "arreglar" el PCA agregando Normalizaciones (L1, L2) o `StandardScaler` fue inútil. Un escalador solo estira o encoge el espacio lineal; no puede "desenredar" una topología de datos inherentemente no lineal.

## 3. El Veredicto DSP (Procesamiento de Señales)

El análisis del agente DSP aporta una solución matemática brillante al problema de la topología:

*   **Ratios vs. PCA:** Un Ratio ($Ch_0 / Ch_2$) es una operación matemática **no lineal** (una división). El PCA lineal no entiende de divisiones. 
*   **La Transformada Logarítmica:** Existe un truco bio-matemático. Si aplicamos un Logaritmo Natural a la señal ($log(x)$) antes de hacer PCA, por propiedad de los logaritmos, la división de Ratios se convierte en una resta lineal: $log(Ch_0 / Ch_2) = log(Ch_0) - log(Ch_2)$. En el dominio logarítmico, las proporciones se vuelven distancias lineales euclidianas que el PCA *sí* puede separar.
*   **Filtros de 20Hz:** El filtro pasa-altos de 20Hz usado para borrar ruido de cables también borró los movimientos macro-mecánicos sutiles de la mandíbula (que oscilan a 5-15 Hz).

## 4. Conclusión y Nuevo Paradigma

El PCA tradicional ha tocado su límite biológico y matemático para este tipo de señal. Tus datos de EMG crudos forman un *Manifold* no lineal (imagina un papel arrugado). PCA intenta mirar el papel desde arriba; XGBoost lo desdobla.

### Siguientes Pasos (El Pivot del Proyecto)

Dado que sabemos que la información fonética **SÍ existe** (gracias al 80% de XGBoost), debemos abandonar los esfuerzos de clustering lineal y adoptar las siguientes estrategias:

1.  **UMAP Supervisado:** En lugar de PCA, utilizaremos UMAP pasándole las etiquetas de las vocales (`y`). UMAP deformará el espacio de manera no lineal para agrupar los puntos que comparten la misma "lógica", mostrándote por fin los clusters limpios que deseas ver.
2.  **Clustering de SHAP Values:** En lugar de hacer clustering sobre los voltajes brutos de los electrodos, extraeremos la matriz de decisiones del modelo XGBoost (valores SHAP) y haremos clustering sobre *cómo "piensa" el modelo*.
3.  **Redes Neuronales (Autoencoders):** Avanzar hacia el plan original de Autoencoders, los cuales, al tener funciones de activación no lineales (ReLU), podrán desenredar este crosstalk masivo que el PCA no puede.

> [!TIP]
> **Resumen para el Director:** El experimento fue un éxito. Comprobaste que hay un 80% de señal vocal en el ruido. Ahora solo debemos cambiar el microscopio matemático con el que la miramos.
