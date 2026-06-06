# CHANGELOG

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
