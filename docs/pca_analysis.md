# Documentación: pca_analysis.py

## Descripción General
El script `pca_analysis.py` forma parte del módulo de Deep Learning de NANDU EMG. Reemplaza la funcionalidad combinada que antes residía en `generador_pca_umap.py` (Linux), extrayendo específicamente la lógica de Análisis de Componentes Principales (PCA) estricta, y acoplándola a la estética cyberpunk y la interfaz gráfica del branch web-viewer.

## Funciones Principales
- **Extracción de Características (Feature Extraction):** Construye la matriz de datos iterando sobre las ventanas válidas seleccionadas. Usa la misma lógica robusta (exclusión de outliers, umbrales de SNR, promediación de envolventes de 100 muestras por canal).
- **Reducción de Dimensionalidad PCA:** Realiza un mapeo de las dimensiones del espacio original (generalmente `100 x N_canales`) a componentes principales de varianza máxima (2D o 3D).
- **Estética de Interfaz:** La UI fue modernizada, integrando los colores corporativos cyberpunk (`bg="#0B0C10"`, `fg="#66FCF1"`).

## Uso
El módulo se ejecuta de manera independiente o desde `main_app.py`. Al iniciarse, carga automáticamente el listado de archivos disponibles en el directorio `base_de_datos_electrodos` y permite al usuario configurar:
1. Umbral de SNR.
2. Nivel de Agresividad para eliminación de ruido de fondo.
3. Cantidad de componentes (2D vs 3D).

Los resultados y gráficos generados se guardan directamente en el directorio de la sesión analizada, permitiendo un flujo de trabajo trazable y consistente.
