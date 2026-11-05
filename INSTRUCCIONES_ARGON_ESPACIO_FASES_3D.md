# Instrucciones para Agente Argon — Módulo Espacio de Fases 3D (Atractor Muscular por Pulso)

## Regla Cardinal y Estricta
**SOLO MODIFICAR EL PANEL EXPERIMENTAL DE PCA EN `autoforge_daq_experimental.py`**.
Está terminantemente prohibido alterar el DSP, la adquisición por hardware/micrófono, el motor de audio, los timers de refresco o cualquier otra sección del script.

---

## Objetivo
Reemplazar el panel `pca_live_groupbox` (PCA 2D estático) por un **Cubo de Proyección 3D Isométrica** que renderice la **Trayectoria del Atractor Muscular en el Espacio de Fases** $\vec{r}(t) = \big( \text{Ch}_0(t),\, \text{Ch}_1(t),\, \text{Ch}_2(t) \big)$ correspondiente a los 3 canales sEMG durante la ventana de detección de cada pulso fonatorio.

---

## Detalles Técnicos de Implementación

### 1. Dependencias y Render
- **Sin dependencias externas pesadas ni OpenGL nativo** (el entorno no tiene `PyOpenGL` instalado).
- Utilizar `pg.PlotWidget` estándar con **proyección isométrica/oblicua 3D sobre 2D** vectorial o un canvas `pg.PlotCurveItem` y `pg.ScatterPlotItem`, trazando los 12 bordes del cubo de referencia $[0, 1]^3$ y los 3 ejes coordenados etiquetados:
  - Eje X: $\text{Ch}_0$ (Vientre Anterior / Digástrico)
  - Eje Y: $\text{Ch}_1$ (Cigomático Mayor / Risorio)
  - Eje Z: $\text{Ch}_2$ (Orbicular de los Labios)

### 2. Proyección Isométrica Canónica
Para mapear $(x, y, z) \in [0, 1]^3$ a $(u, v)$ en 2D:
```python
# Ángulo isométrico clásico (θ = 30° / 0.5236 rad)
cos30 = 0.866025
sin30 = 0.5
u = (y - x) * cos30
v = z - (x + y) * sin30
```
- Dibujar los 12 bordes del cubo con líneas sutiles (`pen=pg.mkPen('#263244', width=1, style=QtCore.Qt.DashLine)`).
- Dibujar los 3 ejes canónicos desde el origen $(0,0,0)$ con los colores oficiales de los canales:
  - Eje $\text{Ch}_0$: Naranja (`#FF8A4C`)
  - Eje $\text{Ch}_1$: Turquesa (`#2DD4BF`)
  - Eje $\text{Ch}_2$: Verde (`#4ADE80`)

### 3. Trayectoria del Pulso en el Espacio de Fases
Al ingresar cada `pulse_event`:
- Extraer `emg_norm_win` (los 3 canales normalizados por el supremo del pulso individual, valores en $[0, 1]$).
- Si viene crudo o en microvoltios, normalizar por $\max_{c \in \{0,1,2\}} (\max_t |x_c(t)|)$ tal como exige el protocolo.
- Generar los puntos proyectados $(u_t, v_t)$ para $t = 0 \dots N-1$.
- Trazar la **curva orbital cerrada** del pulso con el color de la vocal detectada (/a/: `#E63946`, /e/: `#1F77B4`, /i/: `#2CA02C`, /o/: `#9D4EDD`, /u/: `#E7A61A`) con `width=2.0`.
- Dibujar un punto brillante en el vértice de máxima contracción (pico de energía) y un marcador sutil en el origen $(0,0,0)$ (reposo basal).
- Mantener en memoria tenue (alpha bajo) las últimas 3-5 trayectorias previas como estela histórica, resaltando la actual con brillo completo.

### 4. Nombres de Variables e Interfaz
- Título del contenedor: `Espacio de Fases 3D (Atractor Muscular)`
- Checkbox de activación: Mantener `self.chk_pca_experimental` o renombrar su texto visual a `Espacio de Fases 3D (Exp.)`.
- Badge inferior: Indicar el músculo dominante y la energía relativa:
  `Ch0: XX% | Ch1: YY% | Ch2: ZZ% [Vocal /X/]`
- Botón `R`: Limpiar las órbitas históricas acumuladas en el cubo.
