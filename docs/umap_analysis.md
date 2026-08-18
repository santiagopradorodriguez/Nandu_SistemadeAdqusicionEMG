# Documentación: umap_analysis.py

## Descripción General
El script `umap_analysis.py` procesa y proyecta las señales preprocesadas (envolventes) hacia un sub-espacio topológico 2D/3D utilizando UMAP (Uniform Manifold Approximation and Projection). Es la contraparte independiente del algoritmo PCA, diseñada tras el rediseño y separación estructural del archivo monolítico original de Linux (`generador_pca_umap.py`).

## Funciones Principales
- **Construcción del Embedding No Supervisado:** Extrae automáticamente las distancias no-lineales entre las formas de onda de cada gesto/vocal para proyectarlas de forma no supervisada.
- **Configuración Topológica (Hiperparámetros):** Proporciona al usuario controles detallados sobre la métrica de distancia, tamaño de la vecindad (`n_neighbors`), distancia mínima de clusters (`min_dist`) e inicialización de UMAP (aleatoria o espectral).
- **Consolidación Estética:** El entorno de visualización fue ajustado para coincidir visualmente con el aspecto Dark Theme del módulo de análisis original, asegurando una experiencia de usuario estandarizada. 

## Uso
Para ejecutar UMAP:
1. Inicie `umap_analysis.py`.
2. Configure el subset de mediciones a evaluar.
3. Determine los canales EMG activos.
4. Ajuste la tasa de contaminación por *IsolationForest* y la sensibilidad del SNR (para rechazar eventos biológicos corruptos o ruidosos antes de enviarlos al manifold).
5. Exporte el informe. Se generará un PDF automático en la carpeta origen.
