# Plan de Acción: Unificación de Extracción y Filtrado (PCA / UMAP)

## Diagnóstico de los problemas
1. **Error de shape inhomogéneo (`setting an array element with a sequence`)**: 
   En `umap_motor.py`, si una medición carecía de un canal específico, el vector resultante de esa medición era más corto que los demás (por ejemplo, 200 características en lugar de 300). Al intentar convertir la lista de vectores a una matriz de Numpy (`np.array(X_data)`), fallaba porque no era una matriz perfecta.
2. **Filtrado agresivo dispar en UMAP**:
   Actualmente, el UMAP crudo no lee la señal pura, sino los recortes ya generados (`segmentos_rs`). Al intentar calcular el ruido para el filtro SNR, UMAP tomaba las "orillas" (el 15% inicial y final) de ese recorte. Sin embargo, en esos recortes ya suele haber actividad muscular, por lo que el "ruido basal" calculado era irrealmente alto, lo que desplomaba el cálculo del SNR y provocaba el rechazo masivo de pulsos.
   Por el contrario, el motor de PCA (que habíamos reescrito) viaja a la señal cruda (`raw.json`), extrae el ruido real de los silencios interpulsos, alinea todo usando los picos del micrófono y extrae la matriz de características de forma robusta.

## Solución Propuesta (Unificación de Pipeline)
Dado que queremos que tanto PCA como UMAP evalúen exactamente el mismo subconjunto de datos de la misma manera:

1. **Reutilizar el motor de características**: 
   Modificar `umap_motor.py` para que, cuando necesite extraer datos crudos, invoque directamente a la función `build_pca_features` (que reside en `pca_motor.py`).
2. **Homogeneidad de Filtrado**:
   Al usar el mismo extractor, UMAP usará la señal cruda, la alineación por micrófono y el cálculo de ruido interpulso, garantizando que filtre y apruebe **exactamente los mismos pulsos** que PCA bajo el mismo límite de SNR.
3. **Gestión de Vectores (Completo vs Picos)**:
   La función nos devolverá siempre el vector completo (ej. 300 variables si hay 3 canales x 100 muestras). 
   - Si elegiste **"Completa"** en UMAP, usamos la matriz tal cual.
   - Si elegiste **"Picos"**, tomamos esa matriz robusta, la dividimos matemáticamente en bloques de 100 muestras por canal y extraemos el valor máximo absoluto. Así obtenemos el vector de "Picos" (3 variables) pero habiendo gozado del filtrado avanzado de ruido.
4. **Manejo del Error de Shape**:
   `build_pca_features` ya tiene validaciones estrictas que ignoran mediciones si les falta algún canal requerido, lo que elimina de raíz el error de dimensiones inhomogéneas.

Esta refactorización no solo arregla ambos problemas, sino que asegura que la metodología científica de comparación entre algoritmos sea 100% fidedigna (comparan peras con peras).
