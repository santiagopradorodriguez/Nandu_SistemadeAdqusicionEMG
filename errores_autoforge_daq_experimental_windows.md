# Registro de Errores: autoforge_daq_experimental (Windows)

Fecha: 2026-11-17  
Entorno: Windows (`Nandulsd_Core.exe` / `Nandulsd.exe --run-script acquisition/autoforge_daq_experimental`)  
Ruta de script reportada en traceback: `C:\Users\LSD\Downloads\ultimo nandu\Nandulsd\_internal\acquisition\autoforge_daq_experimental.py`

---

## 1. Error Crítico en Adquisición / GUI: IndexError en `actualizar_plot`

### Traceback registrado en la consola de diagnóstico:
```text
Traceback (most recent call last):
  File "C:\Users\LSD\Downloads\ultimo nandu\Nandulsd\_internal\acquisition\autoforge_daq_experimental.py", line 8577, in actualizar_plot
    if len(self.rt_peaks[i]) >= 2:
           ~~~~~~~~~~~~~~^^^
IndexError: list index out of range
```

### Síntoma:
- El traceback se repite de forma ininterrumpida en cada refresco de la GUI (`actualizar_plot`).
- En la gráfica inferior se observa que las curvas se están actualizando o congelando parcialmente debido a la excepción en el callback de dibujo.

### Causa:
- En la línea 8577 de `autoforge_daq_experimental.py`, se accede a `self.rt_peaks[i]` dentro de un bucle sobre canales o trazas, pero el índice `i` excede la longitud de `self.rt_peaks` (por ejemplo, `self.rt_peaks` inicializado con menos elementos que la cantidad de canales configurados o reseteado en vacío `[]`).

### Solución recomendada en el código:
1. **Inicialización segura de `self.rt_peaks`:**
   Asegurar que `self.rt_peaks` se dimensione con sublistas para cada canal en el `__init__` y en cada reinicio de adquisición:
   ```python
   self.rt_peaks = [[] for _ in range(self.num_canales)]
   ```
2. **Guarda defensiva en `actualizar_plot`:**
   Proteger el chequeo antes de indexar:
   ```python
   if i < len(self.rt_peaks) and len(self.rt_peaks[i]) >= 2:
       # lógica de cálculo o dibujo de picos
   ```

---

## 2. Advertencia Secundaria Detectada en la Captura (Antigravity)

### Mensaje:
```text
A customization failed to load
Skill release-manager/SKILL.md
invalid frontmatter format
```

### Observación:
- El archivo de configuración de la skill `release-manager/SKILL.md` tiene un error de sintaxis en el encabezado YAML frontmatter.


---

## 3. Diagnóstico Completo y Reparaciones Implementadas - 2026-11-18

A continuación se documentan los errores detectados en Windows 10/11 y Linux junto con las correcciones aplicadas en el repositorio.

### 3.1 Dimensionamiento Dinámico de Canales y Picos en Tiempo Real - `autoforge_daq_experimental.py`
- **Causa raíz:** `self.rt_peaks`, `self.rt_last_noise_eval` y buffers por canal solo se creaban si faltaba `rt_peaks`, provocando `IndexError: list index out of range` en `actualizar_plot` al cambiar la cantidad de canales o iniciar sesiones consecutivas.
- **Solución:** Se incorporó `_asegurar_dimension_arreglos_canales()` llamado en `setup_gui_for_channels`, inicios de grabación discreta/continua/manual, ruido final y antes del bucle inter-pulso de `actualizar_plot`, con guardas de índice en `self.peak_scatters`.

### 3.2 Consola de Diagnóstico Seleccionable y Persistencia Automática de Logs - `autoforge_daq_experimental.py`
- **Causa raíz:** `DiagnosticTerminalDialog.append_log` invocaba `setTextCursor()` y `ensureCursorVisible()` en cada escritura, borrando la selección de texto del usuario e impidiendo copiar errores repetidos; además no existía respaldo persistente en `.txt` en disco.
- **Solución:**
  - `_TeeLogStream` ahora graba en tiempo real en `logs/autoforge_daq_YYYYMMDD_HHMMSS.log` con `flush()` inmediato e instala `sys.excepthook` y `threading.excepthook`.
  - `DiagnosticTerminalDialog.append_log` inserta mediante un `QTextCursor` sobre `document()` sin alterar la selección activa, agrega casilla `Pausar Auto-Scroll`, botón `Guardar Log (.txt)`, deduplicación/throttling de tracebacks repetidos y confirmación en `Copiar Todo`.
  - `_guardar_log_consola_en_medicion()` guarda `consola_daq_log.txt` dentro de cada carpeta de medición exportada.

### 3.3 Secuencia Continua, Destello No Epiléptico y Comandos de Fase en Pantalla Remota - `autoforge_daq_experimental.py` y `cliente_estimulos.html`
- **Causa raíz:** En Secuencia Continua la vocal desaparecía a los `200 ms` por apagar `p_act` al terminar `flash=True`, el destello cambiaba a fondo blanco/cian brillante de pantalla completa, y varias fases del protocolo no enviaban su instrucción a la pantalla remota.
- **Solución:**
  - En `_disparar_destello_vocal` y `_renderizar_visor_vocal`, la vocal permanece visible durante el 65% del período `tau` y desaparece en el 35% restante usando el estado `_visor_mostrar_vocal`.
  - Destello suave no epiléptico manteniendo fondo oscuro `#050510` con borde y resplandor en el color canónico de la vocal.
  - Espejado completo hacia `_publicar_estado_display_remoto` y `VentanaEstimulosExterna` para todas las fases: `HAZ SILENCIO`, `GRABANDO RUIDO`, `PREPÁRATE`, `CALIBRACIÓN CVM`, `DESCANSO 5s`, `VERIFICACIÓN MUSCULAR - PROBAR MÚSCULO` y `SESIÓN COMPLETADA`, con tamaño de fuente adaptativo en web y segundo monitor.

### 3.4 Lag de Audio en Windows 10/11 y Latencia de Red Tailscale - `autoforge_daq_experimental.py`, `display_remoto_servidor.py` y `cliente_estimulos.html`
- **Causa raíz:** Abrir y cerrar un `sounddevice.OutputStream` por cada beep agregaba 150-350 ms en Windows; `winsound.PlaySound` bloqueaba el hilo de locución; `_ip_tailscale()` dependía de `ip` de Linux; y la comunicación HTTP usaba polling sin `TCP_NODELAY`.
- **Solución:** Stream `OutputStream` persistente en `_drain_and_play_beep_sync`, uso de `winsound.SND_ASYNC` en Windows, detección de IP Tailscale en rango CGNAT `100.64.0.0/10` compatible con Windows, servidor `_FastThreadingHTTPServer` con `TCP_NODELAY` y push instantáneo Server-Sent Events `GET /stream` consumido por `EventSource` con respaldo de `/estado`.

### 3.5 Guardado Completo de CSV, Detector de Pulsos y Metadata sin Vacíos - `autoforge_daq_experimental.py`
- **Causa raíz:** Discrepancia de claves entre `_registrar_pulso_metronomo` y `exportar_segmentos_dataset_y_fatiga` / `_formatear_informe_protocolo_completo_txt` dejaba `pulso_id=0`, `sample_beat=0` y variación MDF en `0.0%`; `pulse_detector.process()` no corría si los checkboxes visuales 2D estaban desmarcados; e iniciar Autograbado sobrescribía `_photometry_session_active = False` aunque ya se hubiera calibrado por el botón superior.
- **Solución:** Emisión y lectura compatible con ambas convenciones de claves en `vocal_cue_log` y `sync_evaluation_log`, ejecución continua de `pulse_detector.process()` durante cualquier grabación activa, propagación de `last_saved_metadata` en `_imprimir_registro_orden`, y preservación de `_photometry_session_active` si la sesión fotométrica ya estaba calibrada.

### 3.6 Asistente de Fotometría 3D en Dos Etapas y Cero Campos `None` - `fotometria_webcam.py`
- **Causa raíz:** El asistente intercalaba capturas y marcado impidiendo elegir otra vista tras tomar las fotos, auto-capturaba la vista submental antes de recentrar desde el perfil izquierdo, y generaba valores `None` y claves incompatibles con `PhotometryGalleryDialog` cuando MediaPipe no detectaba malla 3D en vistas inclinadas.
- **Solución:**
  - Captura secuencial de las 4 fotos primero (`01_frontal`, `02_perfil_izquierdo`, `03_submental` con `yaw` acotado a `+-12` grados y espera mínima de `1.2 s`, `04_perfil_derecho`), seguida de `PASO_MARCADO_ELECTRODOS` unificado con selector libre de las 4 fotos y cola de canales EMG que preselecciona `03_submental` para canales submentonianos/digástricos.
  - Fallbacks geométricos garantizados en `_registrar_anclaje`, `_completar_par`, `_polo_del_par` y `_construir_metadata_fotometria` con todas las claves consumidas por `PhotometryGalleryDialog`, y uso de `tempfile.gettempdir()` + `winsound` en `_VozGuia`.

### 3.7 Estabilidad del Visor CSV y Lanzador PyInstaller en Windows - `csv_viewer_widget.py`, `main_app.py` y `autoforge_daq_experimental.py`
- **Causa raíz:** Reemplazo o cierre de `DataLoaderThread` activo destruía el `QThread` en ejecución; rangos `NaN` o `global_max_y == 0` lanzaban excepciones en sliders; `signal.filtfilt` fallaba con señales cortas; `numba` sin fallback impedía abrir en Windows si fallaba JIT; y el multiplexor `--run-script` en `main_app.py` fallaba silenciosamente con `sys.stdout`/`sys.stdin` en `None`.
- **Solución:** Lista de vida segura `_active_loader_threads` y `closeEvent` en `CsvViewerWidget`, guardas `np.isfinite` / divisor > 0 y `len(y_data) > 27` para `filtfilt`, fallback NumPy vectorizado para `calculate_rms_envelope` si falla Numba, y manejo de `sys.stdout`/`sys.stderr` nulos + `multiplexer_error.log` + `QMessageBox.critical` en `main_app.py`.

---

## 4. Correcciones de Estado en Secuencia Continua, Pausar, Detener y Grabación Manual - 2026-11-18

### 4.1 Crash en `_publicar_estado_display_remoto` por `tira_secuencia = None` y división por cero
- **Causa raíz:** Al invocar `_publicar_estado_display_remoto` con `tira_secuencia=None` fuera de `is_autoforge_continuo and is_recording` (por ejemplo al iniciar Secuencia Continua, al detener AutoForge o en grabación manual), la variable `tira_secuencia` permanecía en `None` y fallaba en `tira_secuencia.append(...)` con `AttributeError`. Adicionalmente, si `autoforge_words` estaba vacío, el cálculo `% n_w` lanzaba `ZeroDivisionError`.
- **Solución:** Inicialización incondicional `tira_secuencia = []` antes de poblar la tira y guarda `if n_w > 0:`, limitando el armado de tira al modo Autograbado Discreto en ejecución.

### 4.2 Fallo al Detener Secuencia Continua o Autograbado - `detener_autoforge`
- **Causa raíz:** Un fallo intermedio en `detener_autoforge` o al iniciar el protocolo dejaba el botón con el texto `DETENER SEC. CONTINUA` mientras `is_autoforge_running` estaba en `False`, provocando que el botón abriera nuevamente el diálogo de inicio en lugar de detener. Además, `btn_pause.setChecked(False)` disparaba la señal `toggled(False)` y reactivaba timers recién cancelados.
- **Solución:**
  - `iniciar_autoforge` e `iniciar_autoforge_continuo` ahora verifican también `is_autoforge_continuo` y `"DETENER"` en el texto del botón, e invocan `detener_autoforge()` ante cualquier excepción de arranque.
  - `detener_autoforge` bloquea señales de `btn_pause` antes de desmarcarlo, incrementa `_flash_beat_token`, purga `beep_queue` y `voice_cue_player._queue`, protege `_publicar_estado_display_remoto` y garantiza `reset_autoforge_buttons()`.

### 4.3 Botón `PAUSAR` / `REANUDAR` Deshabilitado y Timers Activos en Pausa
- **Causa raíz:** `set_controls_enabled` nunca habilitaba `btn_pause` al iniciar adquisición; `on_metro_beat`, `_programar_siguiente_pulso` y los `QTimer.singleShot` de destello/ocultamiento de vocales seguían ejecutándose o sobrescribiendo la UI durante la pausa; y al detener manualmente se deshabilitaba `btn_pause` aunque la adquisición continuara.
- **Solución:** Habilitación de `btn_pause` vinculada al estado de adquisición en `set_controls_enabled`, bloqueo por `self.is_paused` en metrónomo y callbacks de destello, drenado de colas de audio al pausar, estado visible `EN PAUSA` en visor local y display remoto, ajuste de `recording_start_time` y `acq_start_time` al reanudar, y limpieza de `_paused_phase_remaining_ms` / `_paused_metro_remaining_ms`.

### 4.4 Grabación Manual sin Adquisición Previa y Consistencia De Bruijn / Ruido Basal
- **Causa raíz:** `on_record_click` llamaba al método inexistente `self.start_acquisition()` cuando `is_acquiring` era `False`; la cuenta regresiva en Secuencia Continua anunciaba la primera vocal de `autoforge_words` en lugar de `autoforge_full_sequence[0]`; `_registrar_pulso_metronomo` indexaba `(0 - 1)` en `full_seq` durante el count-in; `actualizar_plot` sobrescribía `lbl_recording_space` en modo discreto y mostraba `np.float64(...)` en estadísticas de ruido.
- **Solución:** Reemplazo por `self.on_start_acq_click(start_metronome=False)`, lectura coherente desde `autoforge_full_sequence` en count-in y fin de secuencia continua, eliminación de la sobrescritura de `lbl_recording_space` en `actualizar_plot` y conversión explícita a `float(...)` en `noise_levels`, `initial_noise_mean` e `initial_noise_std`.
