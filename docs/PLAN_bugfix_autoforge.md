# Plan de bugfix: AutoForge DAQ

Fecha de apertura: 2026-10-29
Origen: sesión con Santiago — bugs vistos en el labo (Windows 10 + DAQ real) y en desarrollo.

---

## F1 — Fotometría: auto-captura del perfil derecho ✅ COMPLETADO (2026-10-29)

**Síntoma**: a veces toma la última foto (perfil derecho) sin que la persona se haya girado
al lado correcto.

**Causa raíz**: `AsistenteLinealFotometriaDialog._bandas_objetivo()` valida los perfiles con
`|yaw|` en 25–65°, sin signo. Al pasar de `PASO_PERFIL_IZQ` a `PASO_PERFIL_DER`, la pose
anterior (todavía girada al lado del perfil izquierdo) seguía dando semáforo verde; tras
`TICKS_VERDE_CAPTURA=16` ticks (0,8 s) disparaba la foto sin giro.

**Fix**: al capturar el perfil izquierdo se memoriza el signo del yaw
(`_yaw_signo_perfil`); el perfil derecho exige el **signo opuesto** (agnóstico a la convención
de la malla). Se resetea al entrar a frontal/izquierdo. Test de regresión: 9 casos, todos OK.
Commit `4a68eae`.

---

## F2 — Medidor de separabilidad vocálica (Hito 23) ⏳ PENDIENTE

**Estado**: el diálogo `TestSeparabilidadVocalesDialog` + botón `btn_test_vocales` **no
existen** en el árbol actual, pero **sí en la historia de git**: fue eliminado en el commit
`2c84eab` (27/10, refactor grande de build/libro). La persistencia sigue viva: 7 sitios que
escriben `diagnostico_separabilidad_vocales` desde `last_separability_metrics` en
metadata.json (sesión y canales).

**Tarea**: recuperar la clase completa desde `2c84eab^` (o `82a4b1e`) e integrarla:
- Extraer `TestSeparabilidadVocalesDialog` y todo su wiring (imports, botón en la barra de
  transporte, conexiones, persistencia a `last_separability_metrics`).
- Verificar que compile con el estado actual del archivo (posibles dependencias renombradas).
- Probar el **modo simulación sintética** (3 escenarios: Óptimo, Solapamiento O-U/A-E,
  Crosstalk alto) sin hardware.
- Confirmar que al grabar una sesión el diagnóstico cae en `metadata.json`.

**Aceptación**: diálogo abre desde el botón TEST VOCALES, simulación corre los 3 escenarios,
`last_separability_metrics` se persiste en metadata al grabar.

**Estimación**: 0,5–1 día.

---

## F3 — DAQ: pérdida de muestras / robustez de adquisición ⏳ PENDIENTE

Hallazgos de la auditoría:
1. `DAQThread` (línea 616) es **código muerto** (nunca se instancia) y tiene un bug: timeout
   fijo de 0,1 s mientras el chunk a 6 kHz dura 0,167 s → timeouts permanentes si se usara.
2. El camino vivo (`acquisition_thread`, línea 942) usa timeout correcto
   (`chunk/sr*5`) pero el búfer de la DAQ es de **1 s** (`samps_per_chan=sample_rate`):
   cualquier stall de Python > 1 s produce overflow (error -200279) que hoy es **fatal** y
   detiene la adquisición.
3. No hay contador/reporte de overflow ni verificación de continuidad de muestras.

**Tarea**:
- Eliminar `DAQThread` (o arreglar su timeout si se quiere conservar; preferible eliminarlo).
- Subir el búfer a 3–5 s y manejar -200279 como **recuperable**: log + contador de muestras
  perdidas (estimar por diferencia de reloj), continuar.
- Agregar al metadata de sesión `daq_overflows`, `daq_samples_lost_est` (0 en sesión ideal).
- Test: no se puede con hardware desde acá → validar por revisión + prueba de humo en el labo.

**Estimación**: 0,5 día + validación en labo.

---

## F4 — Guardado de datos: auditoría ⏳ PENDIENTE

**Tarea** (revisión dirigida):
- `guardar_grabacion_wav` / `guardar_grabacion_csv`: escalado int16, columnas, flush, cierre,
  nombres y colisiones.
- Savers async (`guardar_async` en 3 flujos): excepciones silenciadas, escritura parcial,
  orden metadata ↔ wav/csv, carpetas de sesión.
- Verificar que metadata siempre tenga las claves nuevas (`t0_epoch`, `t0_perf_counter`,
  `daq_sample_rate`, `diagnostico_separabilidad_vocales` si aplica).

**Aceptación**: informe de hallazgos + fixes de los bugs concretos encontrados con test de
escritura/lectura.

**Estimación**: 0,5 día.

---

## F5 — Metrónomo: tiempos exactos + modo jitter ⏳ PARCIAL (2026-10-30)

### ✅ Hecho: registro de tiempos exactos de cada pulso (commits `3666e2b`, `8cf7426`)

- `on_metro_beat` captura `time.time()` y `time.perf_counter()` **al entrar** al beat
  (antes de UI/beep) y llama a `_registrar_pulso_metronomo`.
- Una línea por pulso en consola:
  `[METRONOMO] #seq principal=1 idx=3 sub=0 t_epoch=... t_perf=... t_daq_s=... sample_daq=... offset=0 bpm=60 subdiv=4 palabra=A`
- `pulsos_metronomo.csv` en la carpeta de la sesión (los 3 flujos de guardado) con
  columnas: `seq, idx, sub, principal, count_in, t_epoch, t_perf, t_daq_s, t_wav_s,
  sample_daq, offset_muestras, bpm, subdiv, palabra`.
- `t_daq_s` = relativo al ancla de grabación (`recording_start_time`); `t_wav_s` y
  `sample_daq` = índice **dentro del .wav final**, incluyendo el offset de muestras que
  el archivo ya trae al fijar el ancla (`offset_muestras`: pre-roll manual ≈1 s, o el
  ruido basal en AutoForge). Antes se asumía erróneamente que el ancla era la muestra 0.
- Metadata de sesión: `metronomo_pulsos` (cantidad) y `metronomo_pulsos_csv`.
- `t0_epoch`/`t0_perf_counter` de metadata ahora se derivan de `recording_start_time`
  (no del instante de guardado), así el CSV de pulsos y la metadata usan el mismo reloj.
- Copia congelada de los pulsos en los guardados asíncronos (el metrónomo puede
  reiniciarse mientras el hilo escribe).

### ✅ Hecho: temporizador del metrónomo sin deriva

- `Qt.CoarseTimer` (±5 %) reemplazado por `Qt.PreciseTimer`.
- Programación por **reloj absoluto monotónico** (`_programar_siguiente_pulso`):
  el objetivo es `anterior + intervalo`, no "intervalo desde que terminó el callback".
  Medido a 120 BPM: antes 0,481→0,516 s con deriva acumulada (~14 ms/pulso ≈ 1,7 s/min);
  ahora 0,493–0,505 s sin acumulación.
- Intervalo calculado en segundos flotantes (sin `int()` en ms, que sesgaba 60/90 BPM).

### ⏳ Pendiente: modo jitter

**Objetivo**: que el pulso no caiga siempre exacto en el período, para evitar anticipación
del sujeto (el estímulo deja de ser predecible).

**Diseño propuesto**:
- Amplitud configurable en la UI (p. ej. ±% del período), distribución uniforme por pulso.
- Se apoya en `_programar_siguiente_pulso` (ya usa reloj absoluto): el objetivo pasa a ser
  `anterior + base*(1 + U(-j, +j))` en lugar de `anterior + base`.
- Registrar el jitter aplicado en `pulsos_metronomo.csv` (columna `jitter_ms`) y el
  parámetro en `metadata.json`.
- Mantener el reloj absoluto re-anclando el objetivo si el atraso supera el intervalo.

**Decisión pendiente del usuario**: amplitud (¿±ms o ±%?), distribución (uniforme vs normal)
y si aplica a todos los pulsos o solo a algunos.

**Estimación**: 0,5 día.

---

## F5b — Fotometría: doble snapshot en CVM 🐞 DETECTADO (2026-10-30)

En `on_metro_beat`, rama CVM "PAR" (contracción), hay **dos** llamadas consecutivas a
`photometry_mgr.request_snapshot(fase="cvm", ...)` en el mismo beat (~líneas 4279 y 4286):
la primera sin `t_daq_s`, la segunda con `t_daq_s`. Probable artefacto de merge → cada
toma CVM pide dos snapshots. Revisar y dejar una sola (la que pasa `t_daq_s`).

---


## F6 — Compatibilidad con "el main" ⏳ PENDIENTE de reproducción

**Síntoma reportado**: "se traba con ciertas cosas".

**Pistas**: `gui_app/main_app.py` lanza scripts externos por dos vías:
- `_launch_external` → subproceso (correcto);
- multiplexor `--run-script` (build congelado) → **`runpy.run_path` en el MISMO proceso**,
  compartiendo `QApplication` con la ventana principal: candidato directo a congelamientos.

**Tarea**: reproducir el caso exacto (¿módulo?, ¿.exe congelado o dev?, ¿al abrir/cerrar?),
y si es el multiplexor, lanzar siempre subproceso. Necesita que Santiago precise el caso.

**Estimación**: 0,5 día tras reproducción.
