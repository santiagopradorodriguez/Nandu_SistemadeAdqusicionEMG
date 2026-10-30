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

## F5 — Metrónomo: modo jitter ⏳ PENDIENTE (necesita definición)

**Objetivo**: que el pulso no caiga siempre exacto en el período, para evitar anticipación
del sujeto (el estímulo deja de ser predecible).

**Diseño propuesto**:
- `metronomo_visual.py`: flag `--jitter-ms=X` (o `--jitter-pct=Y`), distribución uniforme
  por pulso; el próximo intervalo = `base ± U(0, X)`.
- Programación por reloj absoluto (monotónico) para evitar drift acumulado del `after()`.
- Registro de los tiempos reales de cada pulso (`pulsos_reales.csv` o en consola) para
  validación.
- Persistir `jitter_ms`/`jitter_pct` en `metadata.json` de la sesión.
- Nota de compatibilidad: la cadena de análisis alinea por picos del micrófono (canal_3), así
  que el jitter es compatible; limitar a ±10–15 % para no clipar las ventanas 40/60 %.

**Decisión pendiente del usuario**: amplitud (¿±ms o ±%?), distribución (uniforme vs normal)
y si aplica a todos los pulsos o solo a algunos.

**Estimación**: 0,5 día.

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
