# Plan de desarrollo por agentes: Trackeo facial en tiempo real + Display remoto de estímulos

Proyecto: NANDU LSD — Sistema de Adquisición EMG
Fecha: 2026-10-27
Estado: **PAUSADO el 2026-10-29** (por decisión del usuario: priorizar bugs de AutoForge).
Fase 1 completada; Fases 2 y 3 pendientes — retomar por el orden del §4.

---

## 0. Estado del plan (actualizado 2026-10-29)

| Fase | Estado | Detalle |
|---|---|---|
| 1 (B1, B2, A1, A2) | ✅ COMPLETA | Commits `a5b2885` (plan), `3d98a49` (B1 servidor), `09bb8f7` (B2 cliente HTML), `2e3b0d6` (A1 ancla metadata), `efc86c3` (A2 FaceTrackingThread). Tests de aceptación de las 4 fichas pasados; demo servidor+cliente verificada en navegador. |
| 2 (B3, A3, A4) | ⏸ PAUSADA | Nada integrado aún en `autoforge_daq_experimental.py` (0 referencias a `display_remoto`/`tracking_facial`). |
| 3 (A5, B4) | ⏸ PAUSADA | A5 puede empezar ya (A1+A2 están listos); B4 después de B3. |
| 4 (A6) / 5 (C1) | ⏸ Pendientes | C1 requiere la PC del labo (Windows 10 + DAQ). |

**Punto exacto de retomada**: Fase 2 en orden estricto B3 → A3 → A4 (serial, mismo archivo).
Sugerencia heredada de B2 para B3: publicar `count_in` en el pre-roll y `t0_epoch` = arranque del beat
(después del count-in), para fase perfecta con el audio del metrónomo en canal_3.

---

## 1. Contexto de máquinas y redes (IMPORTANTE — leído con el usuario)

| Máquina | Dónde | Rol | Red |
|---|---|---|---|
| `pop-os` (esta) | **Casa** | Desarrollo, repo, venv, tests | WiFi personal + **Tailscale activo** (100.118.12.94) |
| PC del labo | **Labo** | Corre AutoForge DAQ + nidaqmx + servidor de estímulos | **Windows 10, SIN Tailscale**. Red del Dpto. Física: estado desconocido |
| Celu (`a03-de-santiago`) | Con la persona | **Solo muestra la vocal/sílaba** (navegador) | Android, **ya está en la tailnet** (100.102.47.75, hoy offline) |
| Lenovo Yoga | Opcional | Segunda pantalla / compu de respaldo | Linux, ya en la tailnet (hoy offline) |

**Conclusión de red**: el desarrollo y los tests se hacen en casa (Linux); el despliegue es en
Windows 10 del labo. La red del departamento NO debe ser una dependencia: hay 4 caminos de
transporte y los dos primeros no la necesitan.

### Escalera de transporte para el display remoto (en el labo)

1. **Hotspot del celu** (recomendado): el celu crea un WiFi, la PC del labo se une. Red privada
   de 2 equipos, cero dependencia del labo. Contra: la PC queda sin internet durante la sesión.
2. **USB tethering**: celu conectado por cable a la PC de Windows. Sin WiFi, estable, y el celu
   se carga durante la sesión. Windows 10 monta la interfaz RNDIS solo.
3. **Red del labo**: PC por cable + celu por WiFi del depto. Puede funcionar o puede haber
   aislamiento de clientes → **probar un día con ping**, no asumir.
4. **Tailscale en la PC del labo** (opcional, bonus): si se puede instalar un .exe en la PC
   (admín del labo), el celu ya está en la tailnet y además Santiago podría entrar a la PC del
   labo desde casa. Único camino que pide instalar algo.

### Notas de sincronía para el estímulo

- El navegador del celu **renderiza la secuencia localmente** con su reloj sincronizado al
  servidor (handshake NTP-lite). El jitter de red queda absorvido: la animación no depende de
  la latencia de cada frame.
- Tolerancia requerida: ±100 ms es inocuo — el ancla real del análisis es el **audio del
  metrónomo en canal_3** y los picos del micrófono.
- El display envía **solo JSON/texto** (KB por minuto, sin video) → funciona hasta en el peor
  enlace.

---

## 2. Dos proyectos, un objetivo combinado

- **Proyecto A — Trackeo facial en tiempo real durante la DAQ**: hilo de inferencia MediaPipe
  que graba un CSV por frame (pose + biometría) alineable con la señal EMG por ancla temporal.
- **Proyecto B — Display remoto de estímulos**: servidor HTTP stdlib en la PC de adquisición +
  página fullscreen en el celu/otra compu que muestra la palabra actual (palabras.txt: vocales
  A/E/I/O/U) y el metrónomo visual, réplica de `ventana_palabras.py`.
- **Sinergia**: con ambos, la persona ve la consigna en el celu mientras la cámara trackea su
  cara y la DAQ graba — el experimento completo sin que nadie vea la pantalla del operador.

---

## 3. Reglas para TODOS los agentes

- Python de pruebas: `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/venv/bin/python`
  (tiene soundfile, mediapipe, umap; NO tiene flask/websockets/aiohttp → red con stdlib only).
- Prohibido: sudo, `git push`, modificar `config_general.json` (modificación local del usuario),
  tocar archivos fuera de la ficha asignada.
- Código **multiplataforma** (debe correr en Windows 10 del labo): rutas con `os.path`, UTF-8
  explícito al abrir archivos y en `<meta charset>` del HTML, sin comandos específicos de Linux.
- Estilo del repo: encabezado de proyecto en cada archivo nuevo, comentarios en español.
- Cada agente entrega: resumen del diff + comando exacto para probar.
- Commits: uno por tarea, mensaje según convención del repo (`fix(...)`, `feat(...)`, `docs(...)`).

---

## 4. Roadmap y dependencias

```
Fase 1 (hoy, en casa — 4 tareas en PARALELO, archivos disjuntos):
  B1 servidor estímulos      ┐
  B2 cliente HTML celu        ├── sin colisiones
  A1 ancla temporal metadata  │
  A2 FaceTrackingThread      ┘

Fase 2 (SERIAL, todos tocan autoforge_daq_experimental.py — orden estricto):
  B3 gancho de publicación de palabra/estado  →  A3 integración tracking  →  A4 destello sync

Fase 3 (paralelo, archivos disjuntos):
  A5 alineación offline   |   B4 UX de conexión (URL/QR)

Fase 4 (opcional): A6 HUD en vivo

Fase 5 (en el labo, con la DAQ): C1 dry-run de validación integral
```

---

## 5. Fichas de tareas (una por agente)

### B1 — Servidor de estímulos (stdlib, sin dependencias)
- **Archivo nuevo**: `EMG_desarrollo/acquisition/display_remoto_servidor.py`
- **Objetivo**: servidor HTTP en hilo daemon que expone el estado del estímulo y una página.
- **Contrato**:
  - Clase `ServidorEstimulos(threading.Thread)` — `ThreadingHTTPServer` bind `0.0.0.0`,
    puerto default `8765` (configurable), cierre limpio (`shutdown()`).
  - Método thread-safe `set_estado(**kw)` con claves: `palabra, bpm, subdiv, sesion,
    grabando, t0_epoch, t0_perf_counter` + contador interno `secuencia` (int que incrementa
    en cada cambio — el cliente lo usa para detectar cambios al hacer polling).
  - Endpoints:
    - `GET /` → sirve `cliente_estimulos.html` (leerlo del directorio del script).
    - `GET /estado?t=<token>` → JSON: estado + `ts_epoch` (ms, `time.time()*1000`).
    - `GET /sync?c=<ms_cliente>` → `{"c": c, "s": <epoch_ms>}` — eco para estimar offset de
      reloj (el cliente hace el handshake).
  - Token: si se configura (random al iniciar), endpoints sin token → 403.
  - CLI demo para probar sin autoforge: `--demo` cicla las palabras de `palabras.txt` cada 2 s;
    flags `--puerto`, `--token`.
- **Criterios de aceptación**:
  1. `curl localhost:8765/estado` devuelve JSON válido con `secuencia` creciente en demo.
  2. `/` sirve el HTML y se ve en el navegador (probar con browser in-app).
  3. `/sync` responde < 5 ms local.
  4. 5 min de demo sin errores en stderr, cierre limpio con Ctrl+C.
- **Estimación**: 2–3 h.

### B2 — Cliente HTML fullscreen para el celu
- **Archivo nuevo**: `EMG_desarrollo/acquisition/display_remoto/cliente_estimulos.html`
- **Objetivo**: página única, sin CDN ni dependencias externas (debe funcionar sin internet).
- **Funcional**:
  - Palabra gigante centrada (font-size en `vh`), fondo `#050505`, texto cian `#00FFFF` —
    réplica de `ventana_palabras.py`. En count-in: "INICIANDO N" en rojo.
  - Contador de pulso grande + flash de pulso renderizado **localmente**: el cliente calcula
    el offset de reloj con 5 peticiones `/sync` (offset = mediana de `s + rtt/2 − c`) y usa
    `requestAnimationFrame` sobre el reloj corregido para el flash y el contador de pulsos.
  - Polling de `/estado` cada 1 s **solo** para cambios de control (comparar `secuencia`).
  - Botón para `requestFullscreen()`, **Wake Lock API** (`navigator.wakeLock.request('screen')`
    con re-adquisición al volver a `visibilitychange`) para que el celu no se apague en sesión.
  - Reconexión automática con reintento cada 2 s y banner "SIN CONEXIÓN".
- **Criterios de aceptación**:
  1. Abre en Chrome/Android (validar en desktop primero con browser in-app).
  2. Cambio de palabra del servidor visible en < 1.5 s.
  3. Flash de pulso estable (sin tirones) con red simulada lenta (throttling del devtools).
- **Estimación**: 3 h.

### B3 — Gancho en AutoForge: publicar palabra/estado al servidor
- **Archivos**: patch mínimo en `acquisition/autoforge_daq_experimental.py` + nuevo
  `acquisition/display_remoto_estado.py` (singleton thread-safe con el estado y el server).
- **Objetivo**: que cada palabra que la app manda a `ventana_palabras.py` también publique en
  el display remoto.
- **Puntos de integración**:
  - Spawn de `ventana_palabras.py` en línea ~5966 — localizar dónde se escribe la palabra en
    el stdin del subprocess (`stdin.write`) y llamar junto a cada escritura a
    `notificar_display_remoto(palabra, ...)`.
  - Al iniciar grabación: publicar `sesion` + `grabando=True` + `t0_epoch/t0_perf_counter`
    (mismos valores que A1 escribe en metadata); al terminar: `grabando=False`.
  - Al arrancar la app: iniciar el servidor según `config_general.json`
    (`"display_remoto": {"activo": true, "puerto": 8765, "token": ""}`) — leerlo con
    fallback seguro si la clave no existe; NO editar config_general.json (solo leer).
- **Criterios de aceptación**: con la app en modo demo (sin DAQ), cambiar palabra en la GUI
  actualiza el navegador del celu en < 1.5 s.
- **Estimación**: 2 h. **SERIAL** (archivo compartido con A3/A4).

### B4 — UX de conexión (URL + QR)
- **Archivos**: `acquisition/display_remoto_estado.py` (extender) + patch pequeño en autoforge
  para el botón/diálogo.
- **Objetivo**: diálogo "DISPLAY REMOTO" que lista las IPs locales (socket trick contra
  8.8.8.8 / iterate interfaces con fallback), armando `http://<ip>:<puerto>/?t=<token>` por
  cada una, con botón copiar y QR si hay librería (`qrcode`/`segno` — NO están en el venv:
  si no, texto grande; consultar antes de agregar dependencia).
- **Criterios**: en el labo, celu escanea/escribe la URL y conecta. En casa: local + tailscale.
- **Estimación**: 1–2 h.

### A1 — Ancla temporal en metadata.json
- **Archivos**: patch en `acquisition/autoforge_daq_experimental.py` (4 sitios de escritura de
  metadata: ~4703, ~6535, ~7191, ~8085).
- **Objetivo**: dejar el puente entre relojes en cada sesión.
- **Contrato**: junto al `now` existente, agregar al dict de metadata:
  `"t0_epoch": time.time()`, `"t0_perf_counter": time.perf_counter()`,
  `"daq_sample_rate": <sr>`. Capturar los tres en el mismo instante. Claves viejas intactas
  (compatibilidad hacia atrás).
- **Aceptación**: diff revisado; metadata de prueba con las 3 claves; ningún write afectado.
- **Estimación**: 30–45 min. **SERIAL** (autoforge).

### A2 — FaceTrackingThread (módulo nuevo, aislado)
- **Archivo nuevo**: `acquisition/tracking_facial.py`
- **Objetivo**: QThread que consume frames y produce CSV de tracking por frame.
- **Contrato**:
  - Fuente de frames: si `WebcamCaptureThread` está vivo, leer su frame con lock
    (`_latest_frame`/`_latest_timestamp` — no abrir un segundo `VideoCapture`); si no,
    capture propio (`cv2.VideoCapture`, con fallback de APIs igual que fotometría_webcam).
  - MediaPipe `FaceLandmarker` modo **VIDEO** (tracking temporal entre frames), reusando la
    lógica y ruta del modelo de `calibracion_espacial_electrodos.py` (localizar
    `face_landmarker.task` con el mismo criterio; error claro si falta).
  - Cálculos: reusar/adaptar `calcular_pose_y_biometria_facial` (pitch/yaw/roll, apertura de
    boca, ancho) — no duplicar si se puede importar el agente existente.
  - **Schema de salida CONGELADO** (`tracking_facial.csv`, append con flush cada frame):
    `t_perf,t_epoch,frame_idx,fps_inst,pitch,yaw,roll,apertura_boca,ancho_boca,ok_deteccion`
    + opcional `tracking_landmarks.csv` (468×3 por frame, flag `--landmarks`).
  - Back-pressure: si la inferencia va más lenta que la cámara, descartar frames (nunca
    encolar). Nunca bloquear el hilo de GUI ni el de DAQ.
  - Test standalone: `--fuente cam:0` o `--fuente <video.mp4>` (usar un `WIN_*.mp4` del
    dataset para validar en casa sin cámara) + `--duracion 10` → imprime fps medio y escribe
    CSV de prueba.
- **Criterios**: 60 s continuos; ≥ 20 fps a 640×480 en CPU; CSV válido parseable con pandas;
  cierre limpio.
- **Estimación**: 1 día.

### A3 — Integración del tracking en AutoForge
- **Archivos**: patch en `autoforge_daq_experimental.py` (SERIAL, después de B3).
- **Objetivo**: botón "TRACKING FACIAL" junto a "FOTOMETRIA 3D"; arranca/para con la grabación
  de sesión; salida en la carpeta de sesión junto a `canal_N/`.
- **Regla de cámara única**: fotometría abierta → compartir frames del capture existente;
  cerrada → capture propio del thread.
- **Aceptación**: sesión de prueba produce `tracking_facial.csv` en la carpeta de la sesión y
  el botón queda deshabilitado si no hay cámara.
- **Estimación**: 0.5–1 día.

### A4 — Destello de sincronización (opcional, recomendado)
- **Archivos**: `acquisition/sincronia_destello.py` (nuevo) + patch mínimo en autoforge.
- **Objetivo**: flash de pantalla con timestamp doble (perf_counter + sample DAQ conocido)
  cada N s o al inicio de cada pulso — reancla y permite medir la deriva real.
- **Aceptación**: en el video/trackeo, el destello aparece en el frame esperado ±2 frames.
- **Estimación**: 0.5 día. **SERIAL** (autoforge).

### A5 — Alineación offline tracking ↔ DAQ
- **Archivo nuevo**: `acquisition/alinear_tracking_daq.py`
- **Objetivo**: lee `tracking_facial.csv` + `metadata.json` (ancla A1) + `grabacion.csv`
  (columna `Tiempo (s)`) → produce `tracking_alineado.csv` con `t_daq` por frame; detecta
  picos en canal_3 → `pulsos.csv` (n pulso, t_daq, t_video_del_frame más cercano); si hay
  destellos de A4, reporta el residual (ms) por destello.
- **Criterios**: en el dataset de prueba, residual medio < 20 ms; script CLI reutilizable.
- **Estimación**: 0.5 día. Depende de A1 + A2.

### A6 — HUD en vivo (opcional)
- Overlay en el preview existente: pose + apertura de boca en tiempo real. Est 0.5 día.
- Postergable sin impacto.

### C1 — Dry run integral (en el labo, con la DAQ real)
- Checklist: metadata con las 3 claves de ancla · `tracking_facial.csv` completo ·
  `tracking_alineado.csv` con residual < 20 ms · destellos detectados · palabra visible en el
  celu vía hotspot con pantalla despierta · ningún impacto en fps de DAQ (verificar drop de
  muestras en el ring buffer).
- **Estimación**: 0.5 día.

---

## 6. Riesgos conocidos

| Riesgo | Mitigación |
|---|---|
| Firewall de Windows bloquea el puerto del server (1ª vez) | Diálogo "Permitir acceso"; si la red queda como "Pública" (hotspot), agregar regla con admin: `netsh advfirewall firewall add rule name="NANDU display" dir=in action=allow protocol=TCP localport=8765` |
| Celu se apaga a mitad de sesión | Wake Lock en el cliente + USB tethering (carga mientras muestra) |
| Cámara ocupada por fotometría | Regla de cámara única de A3 (compartir frames) |
| MediaPipe en el build de Windows | Ya resuelto por el spec/build existente (EMG_Ejecutable_Build) |
| Colisiones de agentes en `autoforge_daq_experimental.py` (8236 líneas) | Fase 2 estrictamente serial (B3 → A3 → A4) |
| Red del labo con aislamiento de clientes | No dependemos de ella (escalera de transporte §1) |

## 7. Decisiones abiertas

1. ¿Tailscale en la PC del labo? (requiere permiso de instalación; habilitaría acceso remoto
   desde casa — no es necesario para el objetivo).
2. ¿QR en B4 con librería nueva (`segno`, ligera y pip-installable) o solo URL en texto grande?
3. ¿Landmarks crudos (468×3) grabados siempre o solo con flag? (peso ~1 MB/min — trivial,
   pero es hipótesis de análisis; por defecto OFF, activable).
