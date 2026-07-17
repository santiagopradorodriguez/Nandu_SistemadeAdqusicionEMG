# Plan de Acción: Unificación y Ordenamiento del Entrenamiento de Umbrales (V3 - Modificado)

## 1. Reemplazo por el Extractor Unificado (El "Motor de PCA")
*   **Aclaración sobre el Ruido Basal:** Exacto, la observación que viste en el auditor es porque probablemente cargaste un CSV viejo. En nuestro **nuevo** motor unificado (`build_pca_features`), el algoritmo busca el silencio *fuera* del pulso, promedia el ruido, y **se lo resta** a la señal (`segmento - ruido`). 
*   **Acción:** Reemplazaré la extracción manual de Entrenamiento por `build_pca_features`. Así nos aseguramos de que el Entrenamiento trabaje con las señales perfectamente purgadas del DC Offset y calibradas al milímetro con PCA y UMAP.

## 2. Preservación y Adaptación de los Gráficos Visuales
*   **Gráfico de Validación de Umbral (El que corta la señal):** ¡Se queda! Voy a desarmar la matriz resultante de PCA y la voy a volver a agrupar por medición. De esta manera, al final del barrido, se generará el gráfico por cada medición donde se dibuja la línea del umbral ganador y se sombrea qué parte de la señal quedó arriba y cuál abajo.
*   **Gráfico de Debug Inicial (El del ruido rojo):** Como ahora el ruido se evalúa de manera inteligente por *fuera* de la ventana del pulso, la señal que procesamos ya es 100% actividad muscular con línea base clavada en cero. Por lo tanto, adaptaré este gráfico para que simplemente te muestre la señal limpia y normalizada de la medición, lista para el barrido. Así preservamos toda la trazabilidad visual que tenías antes.

## 3. Ordenamiento de Resultados y Carpetas (Sintaxis Segura)
*   Todo se guardará en una carpeta maestra `UMBRALES` (ej. `UMBRALES_SNR10-0_1`). 
*   **Cambio Estricto:** Reemplazaré todos los puntos decimales por guiones medios (`-`) en los nombres de archivos y carpetas para evitar conflictos con falsas extensiones. Todas las imágenes que pediste preservar se guardarán ordenadamente allí dentro en lugar de "sueltas" en la carpeta general.

## 4. Logs y Métricas Trazables
*   **Consola:** El orquestador te mostrará el triple conteo idéntico al de PCA: **Pulsos Totales Brutos, Filtrados por SNR y Resultantes Aprobados**.
*   **Reporte Final:** En la carpeta autogenerada se creará un archivo de texto o JSON limpio enumerando exactamente qué mediciones se aceptaron, cuáles se filtraron y el resultado de la matemática.
