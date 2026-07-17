# Plan de Acción: Unificación y Ordenamiento del Entrenamiento de Umbrales (V2)

## 1. Reemplazo por el Extractor Unificado (El "Motor de PCA")
*   **Diagnóstico:** Actualmente, `training_motor.py` extrae los recortes de manera rudimentaria.
*   **Acción:** Eliminaré todo ese bloque de código manual y lo reemplazaré por una llamada a `build_pca_features`. Esto garantiza que los umbrales se entrenen sobre **exactamente las mismas características, pulsos y niveles de SNR** que usamos en PCA y UMAP. Todo el software estará 100% calibrado al unísono.
*   **Aclaración sobre el Ruido Basal:** La observación que viste en el auditor es clave. Seguramente abriste un CSV de una corrida vieja de PCA. En el motor **nuevo** unificado (`build_pca_features` que ya reescribimos), el algoritmo va a la señal cruda, calcula el ruido interpulso real antes y después de cada pico, y **se lo resta agresivamente** a la ventana (`segmento - ruido_promedio`). Al usar este motor para todo, nos aseguramos de que el ruido basal esté perfectamente restado en el Entrenamiento, en PCA y en UMAP.

## 2. Ordenamiento de Resultados y Carpetas (Sintaxis Estricta)
*   **Diagnóstico:** Los resultados del entrenamiento y las matrices de barrido se están tirando "sueltos" y sobreescribiendo corridas previas.
*   **Acción:** Crearé un sistema de carpetas dinámico idéntico al de UMAP y PCA. Todo se guardará en `base_de_datos/Fecha/UMBRALES/UMBRALES_SNR10-0_1`, `_2`, etc. **Ningún nombre de carpeta o archivo tendrá puntos (`.`) en su sintaxis numérica**; todo decimal será reemplazado por un guion medio (`-`) para evitar problemas de rutas o extensiones fantasmas. Ningún experimento pisará al anterior.

## 3. Logs y Métricas en el Orquestador (Consola Multihilo)
*   **Diagnóstico:** El orquestador actual te deja a ciegas durante el proceso.
*   **Acción:** 
    1. Se imprimirá en pantalla la cantidad de mediciones importadas y los canales elegidos.
    2. Se imprimirá el conteo triple: **Pulsos Totales Brutos**, **Pulsos Filtrados por SNR** y **Pulsos Resultantes Aprobados** (mismos números que escupe PCA).
    3. Se detallará en consola la configuración del barrido seleccionado.

## 4. Reporte Final y Trazabilidad
*   **Acción:** Dentro de la carpeta `UMBRALES_...`, se generará un archivo de reporte que enumere claramente las mediciones aceptadas (y cuántos pulsos aportaron) y las mediciones rechazadas (y por qué fallaron el SNR). Así tendrás un registro perfecto de la materia prima que usaste para el entrenamiento.
