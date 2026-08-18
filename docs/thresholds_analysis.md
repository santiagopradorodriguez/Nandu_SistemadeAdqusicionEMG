# Documentación: training_motor.py y discrete_motor.py (Análisis de Umbrales)

## Descripción General
Esta sección abarca los módulos de estimación probabilística y matemática estática `training_motor.py` y su subsecuente clasificador `discrete_motor.py`. Estos módulos fueron consolidados de la versión branch web-viewer hacia el entorno principal de Linux.

## Funciones Principales
1. **Detrending Trevisan (Corrección de Drift de Base):** 
   Se incorporó de forma nativa el algoritmo de detrending para corregir el 'drift' en la amplitud de los picos. Dado que los electrodos secos o de gel pierden conductancia o el músculo entra en fatiga mieloeléctrica (cambio en pH), las amplitudes absolutas varían en el tiempo. El motor toma el vector de picos y sustrae la tendencia lineal de las envolventes, asegurando métricas consistentes a nivel biológico a lo largo de los minutos que dura la sesión de adquisición.
2. **Entrenamiento de Umbrales (`training_motor.py`):**
   Calcula estadísticamente, a través de percentiles IQR, un límite óptimo discriminante capaz de discernir ruido de activación muscular para la binarización de la señal en tiempo real.
3. **Validación Discreta (`discrete_motor.py`):**
   Una vez calculado el umbral en una porción de entrenamiento (Train set), se aplica este factor escalar crudo en una matriz "Test" separada para verificar que las activaciones cruzan exitosamente las líneas paramétricas configuradas, midiendo la Tasa de Falsos Positivos.

## Modo de Ejecución
Los módulos se pueden disparar directamente desde la ventana `ui_analysis.py`, dentro de la sección "Entrenamiento". Es necesario contar con un dataset que tenga los JSON actualizados y limpios (utilizando el filtro Notch_Q correcto configurado en el DSP previo de la ventana).
