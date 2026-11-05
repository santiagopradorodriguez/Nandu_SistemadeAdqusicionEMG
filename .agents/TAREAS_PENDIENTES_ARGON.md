# Tareas Pendientes para Argon (Backlog Técnico)

Fecha de creación: 2026-11-04

---

## 1. Fotometría y Marcación Fiduciaria de Electrodos
- **Pregunta previa sobre canal submentoniano:**
  - Al iniciar el asistente de fotometría, consultar explícitamente si alguno de los canales sEMG está montado en la región submentoniana (vientre anterior del digástrico / milohioideo).
- **Asignación contextual de vistas según anatomía:**
  - Si un canal se designa como submentoniano, excluirlo de la marcación en la fotografía frontal (evitando proyecciones forzadas o distorsiones de perspectiva). Debe solicitarse su marcación únicamente en la fotografía submentoniana (vista inferior mandibular).
  - La foto frontal queda reservada exclusivamente para músculos faciales directos (cigomático, orbicular, depresor del ángulo de la boca).
- **Exclusión obligatoria de canal acústico (Ch3):**
  - Ch3 corresponde al micrófono acústico (no porta electrodo físico adherido a la piel).
  - Omitir Ch3 de la lista de marcadores fiduciarios y de las fotos de montaje anatómico.

---

## 2. Metrónomo en AutoForge DAQ (Audio Glitch / Clic de Fin de Pulso)
- **Diagnóstico:**
  - El sonido del metrónomo se reproduce bien durante el cuerpo del beep, pero en la cola final se percibe un corte abrupto o chasquido (similar a buffer underrun o salto a cero con baja latencia).
- **Causas identificadas:**
  - Truncamiento abrupto del buffer de audio sin rampa de desvanecimiento (*fade-out* suave de 5 a 10 ms con ventana Hann o coseno), lo que produce una discontinuidad de alta frecuencia audible (click).
  - Liberación o cierre prematuro del dispositivo/stream de audio antes del vaciado completo del buffer DMA de reproducción.
- **Acción requerida:**
  - Aplicar ventana con rampa de decaimiento suave en el sintetizador/buffer del metrónomo.
  - Asegurar tiempo de guarda (drain/flush) para que el final del audio sea limpio y natural.

---

## 3. Identidad Visual y Nombre Oficial en la Ventana Principal (Main Hub)
- **Nombre Oficial del Software:**
  - El título de la aplicación en la ventana principal, barra de título y encabezado superior debe decir: **Ñandú LSD** (reemplazando "Plataforma de Investigación EMG v6.2").
- **Icono Oficial del Ñandú:**
  - En la barra de título, en el icono de la ventana del sistema operativo y en el avatar del encabezado superior (arriba a la izquierda, al lado del título), debe utilizarse el logo oficial del ñandú (`logo_nandu_lsd.png`, el mismo que aparece en la pantalla de carga / splash screen), con resolución nítida y renderizado circular limpio.
