# Plan de Implementación: Análisis Comparativo de Ruidos (Exp 01234)

## Descripción del Objetivo
Crear el script `Emg/comparacion_ruidos_exp01234.py` para comparar el ruido (inicial e inter-pulso) de diferentes mediciones con y sin un filtro Notch de 50 Hz. 
El programa permitirá al usuario seleccionar interactivamente qué carpetas analizar (por ejemplo '2026-05-22', '2026-05-20', '2026-05-18') a través de una interfaz gráfica (GUI) similar a la usada en otros códigos del proyecto.

El resultado final debe ser una lista/gráfico donde las mediciones se ordenen **de menor a mayor ruido**, mostrando siempre el **nombre completo del archivo .csv** utilizado (el cual incluye la información del tipo de cable como 'TRENZADOMALLADOGND' u otros).

Se respetarán estrictamente las siguientes reglas:
1. Reutilización de métodos matemáticos y filtros ya definidos (`scipy.signal.iirnotch`, envolvente con Transformada de Hilbert + Media Móvil, etc.).
2. Selección manual de archivos mediante interfaz.
3. El programa solo expone los resultados ordenados, sin tomar decisiones ni conclusiones automáticas.

## Proposed Changes

### Script Principal

#### [NEW] [Emg/comparacion_ruidos_exp01234.py](file:///home/lbraun/Repos/Nandu_SistemadeAdqusicionEMG/Emg/comparacion_ruidos_exp01234.py)
Creación del script en Python.
**Lógica interna:**
1. **Selección Manual por GUI:** Implementar una ventana interactiva (usando PySide6 o Tkinter, basándonos en la estructura de `plotter_calibrado.py`) donde el usuario pueda elegir qué subcarpetas de `base_de_datos_electrodos` desea comparar.
2. **Reutilización de Funciones:** 
   - Diseño del filtro pasabanda (20-500Hz) y Notch (50Hz) utilizando `scipy.signal.iirnotch` y `scipy.signal.butter`.
   - Implementación de la **envolvente de Hilbert + Media Móvil (SMA)**.
   - Utilización de `rms()` y extracción de la ventana de ruido inicial basándonos en `noise_seconds` del `metadata.json`.
3. **Pipeline Dual para cada medición seleccionada:**
   - Cargar el `grabacion.csv` asociado y obtener el nombre completo del archivo.
   - **Rama A (Sin Notch):** Se aplica Pasa-altos + Pasa-bajos. Se calcula Ruido Inicial y Ruido Inter-pulso a la envolvente resultante.
   - **Rama B (Con Notch):** Se aplica Pasa-altos + Notch + Pasa-bajos. Se calcula Ruido Inicial y Ruido Inter-pulso a la envolvente resultante.
4. **Resultados y Ordenamiento:** 
   - Se guardarán los valores de ruido calculados junto al nombre completo del `.csv`.
   - Los resultados se presentarán ordenados rigurosamente de **menor a mayor ruido**.
5. **Generación de Visualizaciones:**
   - Se generará un gráfico de barras comparativo, ordenado, mostrando el nombre de los archivos en el eje y el nivel de ruido, con el estilo oscuro y moderno característico del proyecto.

## Verification Plan

### Manual Verification
- La verificación de que el script funciona, agrupa correctamente los archivos, despliega la GUI y los ordena adecuadamente correrá por cuenta del usuario directamente. Yo solo proveeré el código listo para ejecutarse.
