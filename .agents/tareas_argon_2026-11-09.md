# Tareas para Argon - 2026-11-09

Fecha: 2026-11-09

---

## 1. Decodificador Facial Cinemático en Tiempo Real (Fallas Críticas y Falsos Positivos)

### Diagnóstico del Problema
El decodificador facial cinemático de 6 canales en vivo funciona muy mal al operar en modo sin sonido (modo autónomo sin micrófono):
- Detecta fonemas y palabras continuamente de forma errática sin que el usuario mueva la cara.
- La precisión y estabilidad del clasificador colapsa cuando no interviene el canal de micrófono.

### Causas Raíz Técnicas Identificadas
1. **Falta de Invarianza Intersujeto en el Reposo Facial (Baseline Rígido):**
   - El pipeline actual (`realtime_facial_decoder_engine.py` / `decodificador_continuo_facial_6canales.py`) y el checkpoint (`campeon_geosup_conv_ortho_3d.pt`) fueron ajustados con baselines de distancias fijos heredados de un solo sujeto (Candela: comisuras en reposo 52.07 mm, mejillas 24.02 mm, etc.).
   - Al probar con otro usuario cuya anatomía en reposo tiene comisuras a 45.5 mm, la constricción labial ($c_3 = \max(0, 52.07 - 45.56)$) se clava en más de 6.5 mm en reposo absoluto. Esto genera una norma $S_{\text{cam}} \ge 1.08$, superando de forma ininterrumpida el umbral alto del Gate Doble ($U_{\text{alto}} = 0.68$) y disparando inferencias espurias al vacío.
2. **Dependencia Crítica del Ancla Acústica (Modo sin Sonido Inválido):**
   - El resultado del 85% de acierto reportado offline dependía estrictamente del pico del micrófono (`MIC_SYNC`) como ancla temporal.
   - En modo autónomo puramente cinemático (silencioso), el detector de flancos del Gate Doble sobre la norma cinemática no encuentra centros de ventana consistentes, desalineando los 150 puntos de entrada frente a los centroides del espacio latente.
3. **Acoplamiento y Ruido de Micro-movimientos:**
   - Variaciones menores de pose o micro-oscilaciones de MediaPipe mueven las coordenadas proyectadas hacia regiones de fonemas aleatorios sin contracción real.

### Tareas Requeridas
1. **Normalización Intersujeto en Línea (Zero-Shot / Calibración Rápida):**
   - Re-entrenar o re-formular el extractor de características para que sea independiente de la fisionomía absoluta de la cara (por ejemplo usando ratios dinámicos $\Delta d / d_{\text{reposo}}$ o normalización por percentiles locales del usuario).
2. **Re-entrenamiento del Decodificador Autónomo:**
   - Asegurar que el modelo funcione con alta precisión en modo 100% cinemático (sin depender del micrófono para la segmentación temporal).
3. **Depuración del Gate Doble Cinemático:**
   - Ajustar los umbrales de activación para que solo disparen ante movimientos articulatorios voluntarios reales y nunca en reposo facial.

---

## 2. AutoForge DAQ: Audio Glitch en la Cola del Metrónomo

- **Problema:** Con auriculares de buena calidad se percibe un leve chasquido, corte abrupto o clic de alta frecuencia audible al finalizar cada beep del metrónomo.
- **Causa:** Truncamiento abrupto del buffer de audio sin rampa de desvanecimiento suave (*fade-out*), generando una discontinuidad brusca hacia cero, o bien liberación anticipada del stream DMA antes de completar la reproducción del buffer.
- **Acción Requerida:**
  - Aplicar una rampa de decaimiento suave (*fade-out* de 5 a 10 ms con ventana Hann o coseno) en la cola final del sintetizador/buffer de audio del metrónomo.
  - Asegurar el vaciado (drain/flush) completo del buffer de reproducción antes de cerrar el evento sonoro.

---

## 3. AutoForge DAQ: Persistencia y Nomenclatura de Sesiones (Evitar Sobreescritura)

- **Problema:** Al repetir una toma con el mismo sujeto y prueba (ej. `SecuenciaContinua_Prueba1_Sujeto1`), el sistema no genera una carpeta nueva ni añade timestamp, sino que reutiliza la misma carpeta física existente. Esto provoca mezcla y corrupción de archivos de corridas distintas (ej. `grabacion.csv` actualizado en una corrida posterior pero `registro_orden_secuencia.txt` remanente de una corrida previa con orden de vocales desincronizado e invertido).
- **Acción Requerida:**
  - Incorporar obligatoriamente timestamp o sufijo incremental por toma (`YYYYMMDD_HHMMSS` o `_01`, `_02`) en el nombre de la carpeta de salida si ya existe una toma previa con ese identificador.
  - Eliminar los bloques `except: pass` silenciosos en el pipeline de guardado de metadatos e informes de protocolo para que cualquier fallo de escritura se visibilice y se capture inmediatamente.
  - Garantizar que la fuente única de la verdad para el orden de vocales ejecutado sea `pulsos_metronomo.csv` y `metadata.json`.
