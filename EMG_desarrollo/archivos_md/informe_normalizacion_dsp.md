# Informe de Actualización DSP: Normalización, Filtrado y Umbrales

Este documento resume los avances implementados en el procesamiento clásico de señales (pipeline de Trevisan) y define los próximos pasos estratégicos para la integración con los modelos de Machine Learning (PCA, UMAP y Autoencoders).

## 1. Métodos de Normalización y Filtrado Agregados Hoy

Hemos llevado el procesamiento determinista de señales a su límite físico para mitigar la variabilidad biológica, el ruido de alta frecuencia y los artefactos por fatiga muscular o despegue de electrodos.

### 1.1. Filtro de Mediana por Ventana Deslizante (Time Slide Windows)
- **Problema**: El suavizado RMS clásico dejaba picos erráticos ("serrucho") que provocaban que una misma contracción continua tuviera caídas de amplitud falsas.
- **Solución**: Se implementó un filtro estadístico de mediana móvil (`rolling(window=15).median()`) sobre la matriz de picos.
- **Resultado**: Las curvas de activación de las vocales ahora son "chatas" y estables a lo largo del tiempo, respetando la tendencia biológica pero eliminando los valores atípicos (outliers).

### 1.2. Detrending Lineal (Corrección de Deriva)
- **Problema**: Durante grabaciones largas (30 segundos), los músculos se fatigan de forma asimétrica o los electrodos ceden microscópicamente. Esto hacía que, por ejemplo, el Canal 2 arrancara en `0.80` y terminara en `0.20`, destruyendo cualquier intento de binarización con umbrales fijos.
- **Solución**: Se aplicó una técnica de *Detrending Lineal*. El algoritmo calcula la pendiente de caída (o subida) a lo largo de los 30 segundos, resta esa pendiente para enderezar la curva, y vuelve a anclar la señal a su media original (garantizando límites `[0, 1]`).
- **Resultado**: Curvas perfectamente horizontales. La vocal "A" mantiene su firma espacial constante desde el segundo 1 hasta el segundo 30.

---

## 2. Trabajo en Progreso: Umbrales Adaptativos

- **Problema**: A pesar del detrending, existe una **variabilidad inter-medición**. El mismo sujeto diciendo la misma vocal en dos tomas distintas produce amplitudes relativas diferentes (ej: Canal 0 en `0.80` en una toma, y en `0.50` en la siguiente). La herramienta de *Grid Search* (Barrido por canal) demostró que **no existe una tripleta de umbrales estáticos universales** que pueda separar perfectamente todas las vocales en todas las tomas.
- **Estado Actual**: Se introdujo el método de binarización adaptativa **Otsu 1D** en la interfaz gráfica de `analisis_trevisan.py`.
- **Próximos Pasos**: 
  - [ ] Testear y validar exhaustivamente el método Otsu.
  - [ ] Verificar si la línea de corte dinámica logra aislar los picos altos de los bajos de manera robusta sin requerir intervención manual.

---

## 3. Integración Pendiente con Machine Learning

Durante el análisis del código, descubrimos que los generadores de datasets tensoriales para Machine Learning (`generador_pca_umap.py`, `generador_umap_supervisado.py`, etc.) **bypassean** la lógica de picos de Trevisan. Importan `analisis_trevisan.py` únicamente para extraer la onda cruda (`env_recortada`) y armar sus propios tensores de 300 puntos (100 por canal).

> [!WARNING]
> Actualmente, el PCA, UMAP y el Autoencoder **no se están beneficiando** del Filtro de Mediana ni del Detrending Lineal. Están entrenando con los datos crudos, ruidosos y con fatiga.

### Próximos Pasos Estratégicos (To-Do List)
1. **Refactorizar Generadores ML**: Modificar los scripts de extracción para que, en lugar de tomar la onda cruda, apliquen las matemáticas de corrección de *Detrending* a los tensores de 100 puntos.
2. **Entrenamiento Limpio**: Volver a generar los tensores PCA/UMAP con la señal detrended y observar la mejora en la separabilidad de los clusters.
3. **Autoencoder**: Entrenar nuevamente el Autoencoder con este dataset purificado. Si el Autoencoder ya lograba buenos resultados con datos sucios, la inyección de estas normalizaciones debería disparar la precisión (Accuracy) a nuevos máximos históricos.
