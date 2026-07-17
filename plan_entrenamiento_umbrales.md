# Plan de Acción: Unificación y Ordenamiento del Entrenamiento de Umbrales

## 1. Reemplazo por el Extractor Unificado (El "Motor de PCA")
*   **Diagnóstico:** Actualmente, `training_motor.py` hace el mismo trabajo redundante (y propenso a errores de filtrado) que hacía el viejo UMAP: lee los recortes, calcula el ruido en las orillas y rechaza pulsos de más.
*   **Acción:** Eliminaré todo ese bloque de código y lo reemplazaré por una llamada a `build_pca_features`. Esto garantiza que los umbrales se entrenen sobre **exactamente las mismas características, pulsos y niveles de SNR** que usamos en PCA y UMAP. Todo el software estará 100% calibrado al unísono.

## 2. Ordenamiento de Resultados y Carpetas (No Sobreescritura)
*   **Diagnóstico:** Los resultados del entrenamiento y las matrices de barrido se están tirando "sueltos" en la carpeta principal de la fecha o sobreescribiendo corridas previas.
*   **Acción:** Crearé un sistema de carpetas dinámico idéntico al de UMAP y PCA. Todo se guardará en `base_de_datos/Fecha/UMBRALES/UMBRALES_SNR10-0_1`, `_2`, etc. Ningún experimento pisará al anterior. Los gráficos del barrido y los JSON resultantes vivirán prolijamente ahí adentro.

## 3. Logs y Métricas en el Orquestador (Consola Multihilo)
*   **Diagnóstico:** El orquestador te avisa que empezó y que terminó, pero en el medio estás "a ciegas".
*   **Acción:** 
    1. Se imprimirá en pantalla la cantidad de mediciones importadas.
    2. Usando la información que nos retorna el motor unificado, imprimiré el conteo de **Pulsos Totales Brutos**, **Pulsos Filtrados por SNR** y **Pulsos Resultantes Aprobados**.
    3. Se detallará en consola la configuración del barrido seleccionado.

## 4. Reporte Final y Trazabilidad
*   **Acción:** Se generará un pequeño resumen/archivo (similar a los resúmenes en LaTeX que hicimos, o directamente un log en el `.txt` de la corrida) que documente qué carpetas entraron al entrenamiento, cuáles fueron rechazadas y con qué umbral matemático terminaron ganando, todo guardado en la carpeta autogenerada.

De esta manera cerramos el círculo: Extracción $\rightarrow$ Entrenamiento $\rightarrow$ Reducción (PCA) $\rightarrow$ Proyección (UMAP) compartirán exactamente la misma filosofía, matemáticas de ruido, y estética de ordenamiento de archivos.
