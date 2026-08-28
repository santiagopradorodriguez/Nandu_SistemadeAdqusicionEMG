# CHANGELOG

## [v6.1.0] - 2026-08-26
### Agregado (Added)
- **Sistema de Colores y Asignación Anatómica por Músculo**: Convención estandarizada en UI y exportaciones: Depresor Anguli Oris (violeta), Mylohyoid (verde), Orbicularis Oris (amarillo) y Micrófono/Canal 3 (rojo permanente). Diálogo interactivo inicial de confirmación de músculos (`MuscleSelectionDialog`).
- **Metadatos Enriquecidos**: Registro de `muscles`, `muscles_map` y `timestamp` en `metadata.json`, con propagación automática en recortes y segmentaciones (`segmentador_secuencias.py`).
- **Módulo Central de Rutas (`path_utils.py`)**: Resolución robusta de rutas que garantiza que `base_de_datos_electrodos`, `analisis_comparativos` y datos de usuario residan junto al ejecutable tanto en desarrollo como en binarios congelados (PyInstaller), desacoplándolos de `_internal/`.
- **Nuevos Motores Analíticos**: Incorporación de `discrete_motor`, `pca_motor`, `training_motor`, `umap_motor`, `generar_graficos_y_ranking` y `plot_metricas_tesis` para procesamiento por lotes, rankings de experimentos y análisis estadístico.
- **Launcher CLI y Multiplexor Universal**: Soporte para ejecución transparente de scripts temporales desde disco vía `runpy.run_path()` en `main_app.py`, y reenvío de argumentos de consola en el Launcher C# de Windows (`launcher.cs`).

### Eliminado (Removed)
- **Depuración de Dependencias**: Eliminación completa de `xgboost` y del módulo `analisis_xgboost.py` en favor del pipeline centrado en PCA, UMAP, Autoencoders Convolucionales 1D en PyTorch y Binarización por Método Trevisan.

### Modificado (Changed)
- **Herramientas de Build**: Optimización de `crear_entorno_ejecutable.py` con filtros de exclusión de datos temporales, y sincronización de `crear_spec_ejecutable.py` y `aplicar_parches_ejecutable.py`.

## [v6.0.0] - 2026-08-20
### Agregado (Added)
- **Deep Learning Release**: Pipeline maestro de Autoencoders Convolucionales 1D (`pipeline_autoencoder_gui.py`) para compresión de señales sEMG y proyección a espacios latentes.
- **Análisis Topológico**: Módulos de reducción dimensional con PCA 2D/3D y UMAP supervisado y no supervisado (`generador_pca_umap.py`, `generador_umap_supervisado.py`).
- **Binarización Trevisan**: Módulos de análisis temporal por bandas y binarización de potenciales (`analisis_trevisan.py`, `analisis_trevisan_bandas.py`, `analisis_binario.py`).
- **Visor de Features**: Herramienta interactiva para inspección de características mioeléctricas extraídas.

## [v5.1.0] - 2026-06-05
### Agregado (Added)
- **Desarrollo IA**: Integración profunda del flujo de trabajo con Agentes Autónomos (Ecosistema Antigravity).
- **DSP en Vivo**: Nuevo Evaluador Retrospectivo Determinista de Ruido Inter-pulso (atado al metrónomo) en el DAQ en tiempo real.
- **UI Cyberpunk**: Integración de Metrónomo Nativo en PySide6 y filtro estético Cyberpunk para el análisis de señales.
- **Métricas Robustas**: El SNR acumulado ahora se calcula usando la envolvente del ruido basal inicial, con exactitud biológica.

## [v5.0.1] - 2026-06-05
### Solucionado (Fixed)
- Corregida la creación del directorio `base_de_datos_electrodos` que fallaba al usar el Launcher C#. Se reemplazó `sys.executable` por `os.getcwd()` en `user_data_path`.
- Solucionado el error `ModuleNotFoundError` al lanzar scripts externos (`metronomo_visual.py` y `ventana_palabras.py`) a través del ejecutable PyInstaller al ajustar las rutas relativas.
## [v5.0.0] - 2026-05-29
### Agregado (Added)
- Editor de palabras AutoForge para automatizar las capturas.
- Centralización del gestor de configuración para todo el ecosistema.
- Inyección masiva de docstrings en el código para mejorar la mantenibilidad.

### Modificado (Changed)
- Refactorización completa de la arquitectura de la interfaz gráfica, migrando totalmente de Tkinter a PySide6 (salto MAYOR).
- Habilitación de colores persistentes para las curvas visualizadas en los gráficos.

### Solucionado (Fixed)
- Corrección de bugs relacionados con el alcance (scoping) de variables en Python.
