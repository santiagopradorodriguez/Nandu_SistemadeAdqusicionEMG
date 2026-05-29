# Plan de Reestructuración de Base de Datos - Sesión de Hoy

Este documento detalla las tareas a realizar para migrar la arquitectura de la base de datos de mediciones de un formato plano a un formato jerárquico basado en fechas (`Fecha -> Medición`).

## Fase 1: Script de Migración Histórica (`migrar_mediciones_por_fecha.py`)
**Objetivo:** Crear un script independiente que reorganice las mediciones actuales sin perder datos.
*   **Escaneo:** Leer el directorio raíz `base_de_datos_electrodos`.
*   **Extracción de Metadatos:** Para cada medición existente, entrar al directorio de un canal (ej. `canal_0`) y leer el archivo `metadata.json`.
*   **Detección de Fecha:** Extraer el campo `measurement_date` (ej. aislar la parte `YYYY-MM-DD`).
*   **Creación y Movimiento:** Crear una nueva carpeta con la fecha correspondiente en la raíz de la base de datos y mover la carpeta de la medición completa hacia este nuevo subdirectorio.

## Fase 2: Adaptación del Software de Análisis y Visualización
**Objetivo:** Refactorizar los scripts existentes para que reconozcan la nueva estructura de dos niveles (`base_de_datos_electrodos/[FECHA]/[MEDICION]`).
*   **Visor CSV Interactivo (`visor_csv_interactivo.py`):** 
    *   Modificar la interfaz (UI) para que el menú de selección se vuelva de dos pasos: primero un desplegable o lista para seleccionar la **Fecha (Carpeta)**, y luego cargar el menú desplegable secundario con las **Mediciones** correspondientes a ese día.
*   **Análisis por Track (`analisis_por_track_integrado.py`):** 
    *   Modificar el diálogo de selección de carpetas para que agrupe las mediciones por fecha o reconozca correctamente las subcarpetas al escanear canales.
*   **Electrode Viewer (`electrode_viewer_4.py`):**
    *   Ajustar el escáner de la grilla (grid) para que busque mediciones en las subcarpetas de fecha, o añadir un árbol de navegación.
*   **Análisis por Correlación y Otros (`extractor_de_datos_procesados.py`, etc.):** 
    *   Añadir iteración sobre los directorios de fechas antes de iterar sobre las mediciones.

## Fase 3: Actualización del Sistema de Adquisición (`CodigoUnificador_integrado.py`)
**Objetivo:** Automatizar la nueva jerarquía de carpetas para todas las mediciones futuras.
*   **Modificación de Rutas:** Interceptar la función `on_export_click` (o donde se maneje el guardado).
*   **Creación de Carpeta Diaria:** Generar la fecha del día actual usando `datetime.now().strftime('%Y-%m-%d')`.
*   **Guardado Estructurado:** Asegurarse de que la ruta final (`output_dir`) sea `base_de_datos_electrodos/[FECHA_ACTUAL]/[NOMBRE_MEDICION]`.
# Plan de Reestructuración de Base de Datos - Sesión de Hoy

Este documento detalla las tareas a realizar para migrar la arquitectura de la base de datos de mediciones de un formato plano a un formato jerárquico basado en fechas (`Fecha -> Medición`).

## Fase 1: Script de Migración Histórica (`migrar_mediciones_por_fecha.py`)
**Objetivo:** Crear un script independiente que reorganice las mediciones actuales sin perder datos.
*   **Escaneo:** Leer el directorio raíz `base_de_datos_electrodos`.
*   **Extracción de Metadatos:** Para cada medición existente, entrar al directorio de un canal (ej. `canal_0`) y leer el archivo `metadata.json`.
*   **Detección de Fecha:** Extraer el campo `measurement_date` (ej. aislar la parte `YYYY-MM-DD`).
*   **Creación y Movimiento:** Crear una nueva carpeta con la fecha correspondiente en la raíz de la base de datos y mover la carpeta de la medición completa hacia este nuevo subdirectorio.

## Fase 2: Adaptación del Software de Análisis y Visualización
**Objetivo:** Refactorizar los scripts existentes para que reconozcan la nueva estructura de dos niveles (`base_de_datos_electrodos/[FECHA]/[MEDICION]`).
*   **Visor CSV Interactivo (`visor_csv_interactivo.py`):** 
    *   Modificar la interfaz (UI) para que el menú de selección se vuelva de dos pasos: primero un desplegable o lista para seleccionar la **Fecha (Carpeta)**, y luego cargar el menú desplegable secundario con las **Mediciones** correspondientes a ese día.
*   **Análisis por Track (`analisis_por_track_integrado.py`):** 
    *   Modificar el diálogo de selección de carpetas para que agrupe las mediciones por fecha o reconozca correctamente las subcarpetas al escanear canales.
*   **Electrode Viewer (`electrode_viewer_4.py`):**
    *   Ajustar el escáner de la grilla (grid) para que busque mediciones en las subcarpetas de fecha, o añadir un árbol de navegación.
*   **Análisis por Correlación y Otros (`extractor_de_datos_procesados.py`, etc.):** 
    *   Añadir iteración sobre los directorios de fechas antes de iterar sobre las mediciones.

## Fase 3: Actualización del Sistema de Adquisición (`CodigoUnificador_integrado.py`)
**Objetivo:** Automatizar la nueva jerarquía de carpetas para todas las mediciones futuras.
*   **Modificación de Rutas:** Interceptar la función `on_export_click` (o donde se maneje el guardado).
*   **Creación de Carpeta Diaria:** Generar la fecha del día actual usando `datetime.now().strftime('%Y-%m-%d')`.
*   **Guardado Estructurado:** Asegurarse de que la ruta final (`output_dir`) sea `base_de_datos_electrodos/[FECHA_ACTUAL]/[NOMBRE_MEDICION]`.
# Plan de Reestructuración de Base de Datos - Sesión de Hoy

Este documento detalla las tareas a realizar para migrar la arquitectura de la base de datos de mediciones de un formato plano a un formato jerárquico basado en fechas (`Fecha -> Medición`).

## Fase 1: Script de Migración Histórica (`migrar_mediciones_por_fecha.py`)
**Objetivo:** Crear un script independiente que reorganice las mediciones actuales sin perder datos.
*   **Escaneo:** Leer el directorio raíz `base_de_datos_electrodos`.
*   **Extracción de Metadatos:** Para cada medición existente, entrar al directorio de un canal (ej. `canal_0`) y leer el archivo `metadata.json`.
*   **Detección de Fecha:** Extraer el campo `measurement_date` (ej. aislar la parte `YYYY-MM-DD`).
*   **Creación y Movimiento:** Crear una nueva carpeta con la fecha correspondiente en la raíz de la base de datos y mover la carpeta de la medición completa hacia este nuevo subdirectorio.

## Fase 2: Adaptación del Software de Análisis y Visualización
**Objetivo:** Refactorizar los scripts existentes para que reconozcan la nueva estructura de dos niveles (`base_de_datos_electrodos/[FECHA]/[MEDICION]`).
*   **Visor CSV Interactivo (`visor_csv_interactivo.py`):** 
    *   Modificar la interfaz (UI) para que el menú de selección se vuelva de dos pasos: primero un desplegable o lista para seleccionar la **Fecha (Carpeta)**, y luego cargar el menú desplegable secundario con las **Mediciones** correspondientes a ese día.
*   **Análisis por Track (`analisis_por_track_integrado.py`):** 
    *   Modificar el diálogo de selección de carpetas para que agrupe las mediciones por fecha o reconozca correctamente las subcarpetas al escanear canales.
*   **Electrode Viewer (`electrode_viewer_4.py`):**
    *   Ajustar el escáner de la grilla (grid) para que busque mediciones en las subcarpetas de fecha, o añadir un árbol de navegación.
*   **Análisis por Correlación y Otros (`extractor_de_datos_procesados.py`, etc.):** 
    *   Añadir iteración sobre los directorios de fechas antes de iterar sobre las mediciones.

## Fase 3: Actualización del Sistema de Adquisición (`CodigoUnificador_integrado.py`)
**Objetivo:** Automatizar la nueva jerarquía de carpetas para todas las mediciones futuras.
*   **Modificación de Rutas:** Interceptar la función `on_export_click` (o donde se maneje el guardado).
*   **Creación de Carpeta Diaria:** Generar la fecha del día actual usando `datetime.now().strftime('%Y-%m-%d')`.
*   **Guardado Estructurado:** Asegurarse de que la ruta final (`output_dir`) sea `base_de_datos_electrodos/[FECHA_ACTUAL]/[NOMBRE_MEDICION]`.
# Plan de Reestructuración de Base de Datos - Sesión de Hoy

Este documento detalla las tareas a realizar para migrar la arquitectura de la base de datos de mediciones de un formato plano a un formato jerárquico basado en fechas (`Fecha -> Medición`).

## Fase 1: Script de Migración Histórica (`migrar_mediciones_por_fecha.py`)
**Objetivo:** Crear un script independiente que reorganice las mediciones actuales sin perder datos.
*   **Escaneo:** Leer el directorio raíz `base_de_datos_electrodos`.
*   **Extracción de Metadatos:** Para cada medición existente, entrar al directorio de un canal (ej. `canal_0`) y leer el archivo `metadata.json`.
*   **Detección de Fecha:** Extraer el campo `measurement_date` (ej. aislar la parte `YYYY-MM-DD`).
*   **Creación y Movimiento:** Crear una nueva carpeta con la fecha correspondiente en la raíz de la base de datos y mover la carpeta de la medición completa hacia este nuevo subdirectorio.

## Fase 2: Adaptación del Software de Análisis y Visualización
**Objetivo:** Refactorizar los scripts existentes para que reconozcan la nueva estructura de dos niveles (`base_de_datos_electrodos/[FECHA]/[MEDICION]`).
*   **Visor CSV Interactivo (`visor_csv_interactivo.py`):** 
    *   Modificar la interfaz (UI) para que el menú de selección se vuelva de dos pasos: primero un desplegable o lista para seleccionar la **Fecha (Carpeta)**, y luego cargar el menú desplegable secundario con las **Mediciones** correspondientes a ese día.
*   **Análisis por Track (`analisis_por_track_integrado.py`):** 
    *   Modificar el diálogo de selección de carpetas para que agrupe las mediciones por fecha o reconozca correctamente las subcarpetas al escanear canales.
*   **Electrode Viewer (`electrode_viewer_4.py`):**
    *   Ajustar el escáner de la grilla (grid) para que busque mediciones en las subcarpetas de fecha, o añadir un árbol de navegación.
*   **Análisis por Correlación y Otros (`extractor_de_datos_procesados.py`, etc.):** 
    *   Añadir iteración sobre los directorios de fechas antes de iterar sobre las mediciones.

## Fase 3: Actualización del Sistema de Adquisición (`CodigoUnificador_integrado.py`)
**Objetivo:** Automatizar la nueva jerarquía de carpetas para todas las mediciones futuras.
*   **Modificación de Rutas:** Interceptar la función `on_export_click` (o donde se maneje el guardado).
*   **Creación de Carpeta Diaria:** Generar la fecha del día actual usando `datetime.now().strftime('%Y-%m-%d')`.
*   **Guardado Estructurado:** Asegurarse de que la ruta final (`output_dir`) sea `base_de_datos_electrodos/[FECHA_ACTUAL]/[NOMBRE_MEDICION]`.
