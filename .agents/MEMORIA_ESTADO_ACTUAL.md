# Memoria Activa del Proyecto (Ñandú EMG)

**Fecha de consolidación:** 2026-09-30  
**Historial completo:** `.agents/historial/bitacora_completa_hitos_001_al_138.md`

---

## 1. Hardware y Canales sEMG
- **Placa y Ganancia:** AD620 con ganancia fija por hardware $R_G = 100\,\Omega \implies G \approx 495\,\text{V/V}$.
- **Músculos oficiales por canal:**
  - **Canal 0:** Vientre anterior del digástrico (`Anterior Belly`)
  - **Canal 1:** Cigomático mayor / Depresor (`Zygomaticus Major` / `Depresor Anguli Oris`)
  - **Canal 2:** Orbicular de los labios (`Orbicularis Oris`)
  - **Canal 3:** Micrófono / Señal acústica de sincronismo
- **Frecuencia de muestreo:** $f_s = 2000\,\text{Hz}$. Metrónomo: 30 BPM.

---

## 2. Acondicionamiento y Normalización Obligatoria
- **Normalización por Supremo Tricanal por Pulso Individual:**
  $$M_{\text{supremo, pulso}} = \max_{c \in \{0, 1, 2\}} \left( \max_{t \in \text{ventana}} |x_c(t)| \right), \quad \tilde{x}_c(t) = \frac{|x_c(t)|}{M_{\text{supremo, pulso}}}$$
- **Purga de artefactos:** Isolation Forest con contaminación del 10% sobre la envolvente macroscópica.
- **Piso de ruido:** Estimación dinámica interpulso con depuración IQR local pre y post pulso. Prohibido descarte ciego por umbrales fijos de SNR.
- **Reproducibilidad:** Semilla universal fija en 42 (`torch`, `numpy`, `random`).

---

## 3. Modelos Campeones Vigentes
- **Autoencoder Convolucional 1D (Zero-Labels estricto, 100% no supervisado):**
  - **Récord Absoluto Lucas GMM (Ventana 50/50, Conv 2D):** **89.04% exactitud GMM**, Silueta +0.425, DB 0.795 (`idx 3179`, Canales `(6, 12)`, Kernel 5, Tanh, $\text{lr}=0.003$, $\lambda_W=2.0$, $\lambda_Z=0.5$).
  - **Récord Lucas MLP (Ventana 50/50):** **91.43%** con SO(2) y **87.85%** nativo (`(32, 16)`, Tanh). En 3D: **87.4%** (Ventana 40/60, `(48, 24)`).
  - **Secuencia Continua P5:** Corresponde a habla continua del propio Lucas (`SecuenciaContinua_Prueba5_Sujeto1`), no de Candela.
  - **Resultados en Ventana 60/60 (Compensación de P5):** Media armónica 86.58% en 3D y 85.55% en 2D impulsada por P5 (87-90%), pero en Lucas GMM la ventana 60/60 rinde menos (83.4% en 2D, 85.4% en 3D) por solapamiento del metrónomo de 30 BPM.

---

## 4. Geometría Latente y Colores de Vocales
- **Cruz Latente:**
  - Semieje $+Y$ ($\theta = 90^\circ$): Apertura mandibular exclusiva de **/a/**.
  - Semiplano derecho $+X > 0$: Sonrisa y retracción comisural para **/i/** y **/e/**.
  - Semiplano inferior $-Y < 0$: Constricción labial para **/o/** y **/u/**.
- **Colores Oficiales Universales:**
  - **/a/**: Rojo (`#E63946`)
  - **/e/**: Azul (`#1F77B4`)
  - **/i/**: Verde (`#2CA02C`)
  - **/o/**: Morado (`#9D4EDD`)
  - **/u/**: Amarillo (`#E7A61A`)

---

## 5. Próximo Paso Inmediato
- Extender la representación latente no supervisada conectando el encoder a una cabeza clasificadora supervisada ligera (Linear Probe / MLP) para evaluar la separación fina de /e/ vs /i/ y /o/ vs /u/.

---

## 6. Infraestructura de Memoria y Tokens
- **Memoria de Trabajo:** `.agents/MEMORIA_ACTIVA.md` (~50 líneas, ~600 tokens al arrancar).
- **Historial Completo:** `.agents/historial/bitacora_completa_hitos_001_al_138.md` indexado por `.agents/INDICE_HITOS.md`.
- **Reglas del Proyecto:** `.agents/AGENTS.md` compactado a 5.3 KB (ahorro del 81.4% de tokens base por turno).
- **Auditoría Local:** Script ejecutable `.agents/auditar_tokens_sesion.py` para medir consumo de tokens en conversaciones.


- **Hito 108:** Adaptabilidad de red dinámica y soporte multicasa/remoto para la WebApp de TARS:
  - **Detección automática de IP local (`get_local_ip`):** Se eliminó la IP estática hardcodeada (`192.168.1.59`) en `tars_web_server.py`. Ahora detecta dinámicamente la IP asignada por cualquier router o red Wi-Fi al iniciar.
  - **Auto-generación de certificados SSL con SAN multihost (`ensure_ssl_certificates`):** Si cambia de red Wi-Fi o casa, el servidor audita y regenera en 0.1s el certificado SSL incluyendo `DNS:pop-os.local`, `DNS:localhost`, `IP:127.0.0.1` y la nueva IP local (`IP Address`), habilitando el acceso por `https://pop-os.local:8443` o por la IP nueva sin bloqueos de navegador móvil.
  - **Soporte preparado para Tailscale:** Detección de IP remota de Tailscale (`tailscale ip -4`) para acceso a distancia fuera del hogar.
  - **Notificación visual de escritorio:** Al iniciar el servicio, Pop!_OS emite una notificación con la URL exacta para conectarse desde el celular.

- **Hito 109:** Mejoras en la experiencia de usuario (UX) móvil y resumen conversacional de audio:
  - **Renderizado de Markdown y código nativo en el celular (`app.js` y `style.css`):** Se implementó un conversor ligero de Markdown en la WebApp móvil (`renderMarkdown`). Ahora los mensajes del asistente muestran negritas reales, encabezados jerárquicos, listas ordenadas/desordenadas y bloques de código oscuros con botón táctil para "Copiar" al portapapeles.
  - **Feedback háptico (Vibración de walkie-talkie):** Integración de `navigator.vibrate` en el celular: vibra al iniciar la grabación (35ms), vibra al enviar el audio ([20, 30, 20]ms), vibra suavemente al recibir la respuesta ([40, 60, 40]ms) y al copiar código.
  - **Resumen conversacional hablado (`extract_conversational_speech`):** En `tars_web_server.py`, las respuestas largas ya no se cortan abruptamente por conteo de caracteres. Elena sintetiza las primeras 1-2 oraciones clave del resultado y concluye con una frase natural de cierre ("El detalle completo y código lo tenés en pantalla"), permitiendo leer el reporte técnico en el celular sin saturar el audio.

- **Hito 110:** Verificación y activación del enlace remoto de Tailscale en TARS:
  - Se confirmó la conexión de Tailscale en la notebook con IP fija `100.118.12.94`.
  - El servidor `tars_web_server.py` regeneró automáticamente los certificados SSL multihost incorporando tanto la IP local (`192.168.1.59`), los nombres mDNS (`pop-os.local`, `pop-os`) y la IP de la malla privada de Tailscale (`100.118.12.94`).
  - Ambos endpoints (`https://100.118.12.94:8443` y `https://192.168.1.59:8443`) probados y respondiendo con éxito.

- **Hito 111:** Corrección de sintaxis en `app.js` y restauración del selector de chats:
  - Se corrigió un error de sintaxis en la expresión regular de bloques de código en `app.js` que impedía la ejecución del archivo JavaScript en navegadores móviles.
  - Se validó la sintaxis completa con `node --check` con 0 errores.
  - Se añadieron etiquetas de versionado contra caché (`?v=2.1`) en `index.html` para `app.js` y `style.css`, forzando a los navegadores de Android/iOS a cargar el script corregido.
  - Restaurada la funcionalidad completa del botón `#chat-selector-btn` y la apertura del modal `#chats-modal`.

- **Hito 112:** Depuración y reinicio en caliente del proceso de red TARS:
  - Se confirmó la apertura del firewall en Pop!_OS (`8443/tcp` y `tailscale0` registrados en `user.rules`).
  - Se detectó que el proceso anterior (PID 4013) venía ejecutándose desde el arranque previo al enlace de Tailscale sin actualizar sockets.
  - Se finalizó dicho proceso y systemd inició un nuevo worker activo (PID 27208) con enlace verificado en `0.0.0.0:8443`.
  - Se verificó respuesta `HTTP/1.1 200 OK` en las tres vías: `https://100.118.12.94:8443`, `https://pop-os.tail0ec03b.ts.net:8443` y `https://192.168.1.59:8443`.

- **Hito 113:** Transmisión en vivo de pensamientos y comandos (Thinking/Tool Stream) a la pantalla del celular:
  - **Extracción de estado interno en backend:** `get_latest_transcript_message` en `tars_web_server.py` ahora analiza los bloques `PLANNER_RESPONSE` activos sin contenido final para extraer el pensamiento (`thinking`), el nombre de la herramienta (`tool_name`), la acción ejecutada (`tool_action`) y los detalles del comando o archivo (`tool_detail`).
  - **Tarjeta visual interactiva en el chat (`app.js` y `style.css`):** Cuando Antigravity procesa, se renderiza una tarjeta flotante azul con animación de pulso (`#processing-bubble`) dentro del contenedor de mensajes (`#chat-container`). Muestra en tiempo real lo que el modelo va pensando (`💭`) y las herramientas que ejecuta (`⚡ run_command`, `view_file`, etc.). Al llegar la respuesta final, la tarjeta se reemplaza limpiamente por el mensaje formateado.
  - Se actualizó la versión de caché a `?v=2.2`.
