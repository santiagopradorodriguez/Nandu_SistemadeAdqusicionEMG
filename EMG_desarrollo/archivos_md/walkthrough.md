# Resumen de Implementación: Análisis Comparativo de Ruidos

He creado el script `Emg/comparacion_ruidos_exp01234.py` siguiendo las especificaciones del plan de implementación aprobado. A continuación, un resumen de las características principales implementadas.

## Cambios Realizados

- **Creación del Script:** Se creó el archivo en [comparacion_ruidos_exp01234.py](file:///home/lbraun/Repos/Nandu_SistemadeAdqusicionEMG/Emg/comparacion_ruidos_exp01234.py).
- **Interfaz Gráfica de Selección:** Implementada usando PySide6. La GUI permite:
  - Navegar por todos los directorios usando la ruta base absoluta `/home/lbraun/Repos/Nandu_SistemadeAdqusicionEMG/Emg/base_de_datos_electrodos/`.
  - Hacer selecciones múltiples usando `Ctrl` o `Shift`.
  - **Seleccionar el modo de visualización**: "Ambos", "Solo Con Notch" o "Solo Sin Notch" mediante un menú desplegable.
- **Análisis Matemático e Incertezas:**
  - El script extrae la frecuencia de muestreo de la primera columna de tiempo.
  - Implementa idénticamente la **envolvente de Hilbert con Suavizado de Media Móvil (ventana de 50ms)** tal como está explícito en `analisis_por_track_integrado.py`.
  - Ejecuta un pipeline dual de procesamiento ("Sin Notch" vs "Con Notch").
  - El **Ruido Inicial** se calcula promediando la envolvente en la ventana de tiempo en reposo. Para calcular su incerteza (Error Estándar de la Media - SEM), la ventana se subdivide en bloques de 250ms.
  - El **Ruido Inter-pulso** estima el piso de fluctuación usando la Desviación Estándar. Para la incerteza, la señal global se divide en bloques de 1 segundo.
- **Formateo y Orden de Resultados:**
  - Los resultados se procesan y se **ordenan cronológicamente** (extrayendo lógicamente los números del nombre de la medición original).
  - Los resultados numéricos se muestran por consola, indicando el **nombre y ruta absoluta del archivo CSV**.
- **Visualización y Gráficos:**
  - Se implementaron dos gráficos de barras comparativos con el estilo oscuro predefinido.
  - El script **ajusta las barras automáticamente** según tu selección (dibujando solo una barra ancha si eliges ver un único filtro, o las dos si eliges comparar).
  - El **título del gráfico y el nombre del archivo guardado** indican ahora explícitamente el filtro visualizado (por ejemplo, añadiendo `_con_notch.png` o `_ambos.png`).
  - Se añadieron las **barras de incerteza (error bars con corchetes negros)** a cada barra del gráfico.

> [!TIP]
> Puedes ejecutar el script desde la consola usando el comando `python Emg/comparacion_ruidos_exp01234.py`. Recuerda que como incluye una interfaz gráfica, el programa abrirá automáticamente la ventana de selección antes de realizar los cálculos matemáticos y graficar.
