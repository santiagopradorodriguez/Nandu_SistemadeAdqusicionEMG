# CHANGELOG

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
