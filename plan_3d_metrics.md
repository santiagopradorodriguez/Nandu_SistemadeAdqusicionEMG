# Plan de Acción: Métricas y Clustering en 3D (PCA y UMAP)

## 1. Modificaciones en PCA (`pca_motor.py`)
*   **Cálculo de Silhouette 3D:** Luego de generar `embedding_3d`, se añadirá el cálculo de la métrica Silhouette usando esos 3 componentes espaciales.
*   **Log en el Orquestador:** Se registrará en la consola (`logger`) el valor obtenido de "Silhouette Score (3D)" justo debajo del actual 2D.
*   **K-Means y Matriz de Confusión 3D:** Dentro del bloque de evaluación K-Means, se instanciará un modelo independiente que hará `fit` sobre el Train en 3D y `predict` sobre el Test en 3D. Se calculará la matriz de confusión (con su porcentaje y absoluto respectivo) y se exportará como `matriz_confusion_pca_3d.png`.

## 2. Modificaciones en UMAP (`umap_motor.py`)
*   **Cálculo de Silhouette 3D:** Similar a PCA, tras el transform de `reducer_3d`, se correrá la métrica de Silhouette en 3D.
*   **Log en el Orquestador:** Se registrará en la consola el nuevo Silhouette Score 3D.
*   **K-Means y Matriz de Confusión 3D:** Se entrenará un KMeans específico en `embedding_3d` y se exportará su propia matriz de confusión como `matriz_confusion_umap_3d.png`, incluyendo la doble nomenclatura de porcentajes y valores absolutos.

## 3. Actualización de Reportes LaTeX
*   Se modificarán las funciones internas de ploteo (`plot_pca_results` y `plot_umap_results`) para que los archivos `.tex` y `.pdf` finales incluyan la viñeta con el resultado de **Silhouette Score (3D)** en sus resúmenes estadísticos.
