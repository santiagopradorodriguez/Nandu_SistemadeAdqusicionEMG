# Paquete de Investigación: Anomalía Acústica en Espacio de Formantes F1-F2

## 1. Contexto del Problema
Se analizaron tres sesiones de habla continua del mismo sujeto bajo metrónomo y secuencias balanceadas:
1. `2026-11-08: Landmarks Varios 02` (25 BPM) -> Triángulo vocálico estándar y limpio.
2. `2026-11-08: Prueba Final` (25 BPM) -> **ANOMALÍA**: clusters colapsados hacia el centro (F2 ~1200 Hz), /a/ comprimida (F1 ~600 Hz en vez de 760 Hz) y /e/ desfasada hacia /u/-/o/.
3. `2026-11-09: Nuevos Landmarks` (40 BPM) -> Triángulo vocálico reproducible con el de la sesión 1, mostrando el undershoot por velocidad.

## 2. Hipótesis a Auditar
- Posible desfasaje o error de sincronización entre el reloj acústico y el disparo del pulso en `Prueba Final`.
- Saturación, ganancia o diferencias acústicas en el canal de micrófono de `Prueba Final`.
- Fonación gutural/reducida real del hablante en esa toma vs. artefacto del algoritmo LPC / selección de ventana.

## 3. Rutas de las Sesiones
- Sesión 1: `EMG_desarrollo/base_de_datos_electrodos/2026-11-08/SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02`
- Sesión 2 (Anomalía): `EMG_desarrollo/base_de_datos_electrodos/2026-11-08/SecuenciaContinua_Pruebafinal_Sujeto1`
- Sesión 3: `EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1`

## 4. Pipeline Canónico
- Script de extracción: `generar_multimodal_canonico_optimizado.py` (RMS 250 ms reloj maestro).
