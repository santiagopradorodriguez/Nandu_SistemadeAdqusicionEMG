# Plan de Acción: Selector de Temas para Exportación PNG (Visor CSV)

## 1. Modificación de la Interfaz (UI)
*   **Diagnóstico:** Actualmente, el botón `btn_export` exporta literalmente lo que ves en pantalla (WYSIWYG), que por la estética general del programa siempre tiene fondo oscuro y líneas neón.
*   **Acción:** Justo debajo del botón "📸 Exportar PNG" (en el apartado de "Extras" de `csv_viewer_widget.py`), añadiré un nuevo `QCheckBox` etiquetado como **"Usar Tema Cyberpunk"**. 
*   **Comportamiento por defecto:** Estará **tildado** para mantener el comportamiento clásico, dándote la opción manual de destildarlo cuando necesites el gráfico para un informe formal o impresión (Tema Claro).

## 2. Inyección de Temática Clara (Light Theme)
*   **Acción en la función de exportación:** Modificaré la función `export_png`. Cuando toques exportar y el checkbox esté *destildado*:
    1. **Fondo:** El fondo de la escena (`plot_widget`) mutará temporalmente a blanco (`#FFFFFF`).
    2. **Ejes y Texto:** Todos los bordes, números, etiquetas de ejes y textos de la leyenda pasarán de gris claro a negro intenso (`#000000`).
    3. **Líneas:** Si bien mantendremos los colores originales para que identifiques el músculo, los oscureceremos levemente (o simplemente los plotearemos sobre blanco) para que el contraste en un PDF o papel sea óptimo.
    
## 3. Reversión Instantánea
*   **Acción final:** Como no queremos romper la interfaz visual de la app, toda la mutación estética descrita arriba se hará "entre bambalinas" en fracciones de milisegundo. Inmediatamente después de generar y guardar el archivo `.png`, el código restaurará el fondo (`#0B0C10`), los textos (`#C5C6C7`) y el grosor neón original, dejando tu interfaz intocable.
