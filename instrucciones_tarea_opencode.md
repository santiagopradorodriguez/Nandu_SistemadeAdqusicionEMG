# Tarea para OpenCode: Sistema de Testeo y Evaluación Interactiva en Decodificador Facial

## Contexto y Archivo Objetivo
El usuario solicitó implementar un sistema interactivo de testeo en vivo dentro del decodificador facial:
- Archivo principal: `EMG_desarrollo/acquisition/autoforge_daq_decodificador_facial.py`
- Motor de decodificación: `EMG_desarrollo/deep_learning/realtime_facial_decoder_engine.py`

## Requerimientos Específicos Solicitados por el Usuario
1. **Botón de Testeo en la Interfaz:**
   - Agregar un botón prominente "Iniciar Test de Vocales" (o "Testear Secuencia").
2. **Lanzar una Lista de Vocales (Secuencia Guiada):**
   - El sistema debe presentar una secuencia o lista guiada de vocales (por ejemplo una secuencia aleatoria balanceada o de Bruijn de /a/, /e/, /i/, /o/, /u/, o una lista configurable).
   - Indicar visualmente al usuario con claridad cuál es la vocal objetivo que debe pronunciar en cada pulso (ej. cartel grande "Pronuncie: /A/", con cuenta regresiva o sincronización de pulso).
3. **Contrastar y Comparar con lo Reconocido en Tiempo Real:**
   - Cuando el decodificador capture la ventana y emita el evento decodificado (vocal reconocida):
     - Contrastar inmediatamente: `Vocal Esperada: X vs Reconocida: Y`.
     - Mostrar explícitamente en la UI: por ejemplo "Esperada: /A/ -> Reconoció: /A/ (ACIERTO)" o "Esperada: /A/ -> Reconoció: /E/ (CONFUSIÓN)".
4. **Estadísticas Acumulativas y Matriz de Aciertos:**
   - Llevar el conteo exacto de aciertos y errores:
     - Total de aciertos vs total de intentos: `Aciertos: N_ok / N_total (Acc %)`.
     - Desglose por cada una de las 5 vocales: `/a/: ok/tot`, `/e/: ok/tot`, etc.
     - Registrar matriz de confusión en memoria para saber qué vocal se confunde con cuál.
5. **Persistencia de Resultados:**
   - Guardar el registro de la sesión de testeo en un archivo CSV o JSON (por ejemplo en `reportes_experimentos/decodificacion_facial/tests_en_vivo/` o `registro_test_vivo_YYYYMMDD_HHMMSS.csv`) con cada pulso, vocal esperada, vocal predicha, probabilidades y si fue acierto o error.

## Reglas Obligatorias del Proyecto
- Cero emojis en código, logs e interfaz.
- Edición estrictamente no destructiva: preservar todo el funcionamiento existente de cámara, audio RMS 250 ms, osciloscopio, modelos cargados y layouts.
- Mantener la estética visual oscura cyberpunk (#0B0C10, #1F2833, #66FCF1, #45A29E).
