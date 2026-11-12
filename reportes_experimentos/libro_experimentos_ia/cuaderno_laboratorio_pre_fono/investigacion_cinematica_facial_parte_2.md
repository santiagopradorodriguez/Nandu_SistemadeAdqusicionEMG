# Investigación Cinemática Facial: Parte 2 - Topología de 6 Canales y Variedad Geodésica Isomap 3D

## Resumen Ejecutivo

Este informe consolida los avances experimentales de la segunda fase de cinemática facial y decodificación articulatoria en secuencias continuas de fonación sobre la sesión `SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02` (200 pulsos, 8 minutos).

A partir de la auditoría rigurosa de sincronización temporal y el barrido sistemático de parámetros, se determinó que la cinemática de las 5 vocales del español (/a/, /e/, /i/, /o/, /u/) responde a una **dimensión intrínseca 3D embebida en una variedad fuertemente no lineal**. Mediante reducción no lineal geodésica (**Isomap 3D, $k=16$**), el sistema alcanza un **$78.3\% \pm 2.7\%$ de precisión en 5-Fold Cross-Validation** (con un pico de $84.4\%$ en 6D), superando en más de 14 puntos porcentuales al modelo lineal rígido (PCA 3D).

---

## 1. Parámetros DSP Fijos Consolidados

Toda la serie experimental quedó normalizada bajo las siguientes reglas cardinales de procesamiento digital de señales:

1. **Master Clock Acústico Dinámico:**
   Alineación estricta al centroide de energía acústica del micrófono mediante ventana RMS de $90\,\text{ms}$, con desfasaje dinámico calculado pulso a pulso desde `pulsos_metronomo.csv`:
   $$\Delta t_{\text{offset}} = t_{\text{wav\_s}} - t_{\text{daq\_s}} = 4.950\,\text{s}$$
   Eliminando cualquier valor estático o cableado manual en código.
2. **Ventana de Suavizado Cinemático Óptima ($125\,\text{ms}$):**
   A la tasa de adquisición de la cámara ($\sim 21.3\,\text{Hz}$), la ventana media móvil de **$w = 3$ frames ($141\,\text{ms}$)** demostró ser el punto dulce global: filtra el jitter frame a frame sin planchar las rampas preparatorias motoras (onset motor de $100$-$200\,\text{ms}$).
3. **Línea de Base Fisiológica por Percentil Global:**
   Cálculo del reposo basal mediante percentiles de la sesión completa ($10\%$ apertura, $20\%$ comisuras y mejillas) sin sustracciones destructivas locales en pre-onset.
4. **Purga de Outliers:**
   *Isolation Forest* con contaminación fijada al $8\%$, reteniendo $N = 180$ a $184$ pulsos de alta consistencia biomecánica.

---

## 2. Los 6 Canales Fisiológicos Canónicos

| Canal | Músculo / Estructura | Puntos Fiduciarios MediaPipe | Operación Matemática | Rol Articulatorio |
| :--- | :--- | :--- | :--- | :--- |
| **Ch1** | Digástrico / Depresores | `labio_sup` (13) a `labio_inf` (14) | Distancia euclídea vertical ($Y$) en mm | Descenso mandibular masivo en **/a/** ($16.8\,\text{mm}$) |
| **Ch2** | Cigomático Mayor | `comisura_izq` (61) a `comisura_der` (291) | $\max(0,\, \Delta \text{Ancho})$ en mm | Tracción lateral pura en **/i/** ($7.2\,\text{mm}$) y **/e/** ($4.8\,\text{mm}$) |
| **Ch3** | Orbicular de los Labios | `comisura_izq` (61) a `comisura_der` (291) | $\max(0,\, -\Delta \text{Ancho})$ en mm | Constricción esfinteriana en **/u/** ($8.4\,\text{mm}$) y **/o/** ($7.1\,\text{mm}$) |
| **Ch4** | Buccinador / Risorio | Comisuras a Mejillas (207, 427) | Distancia euclídea media en mm | Retracción bucal tensa en **/i/** (baja dispersión $0.15\,\text{mm}$) |
| **Ch5** | Malar / Elevador | Comisuras a Malar (123, 352) | Distancia euclídea media en mm | Ascenso oblicuo perilabial hacia el pómulo |
| **Ch6** | Masetero | Comisuras a Masetero (58, 288) | Distancia euclídea media en mm | Tensión profunda del ángulo mandibular en **/a/** y **/u/** |

---

## 3. Comparativa Cuantitativa: PCA Lineal vs Isomap Geodésico

Sobre la matriz completa de $180$ pulsos $\times$ $150$ features temporales ($6$ canales $\times$ $25$ puntos en $[-500\,\text{ms}, +500\,\text{ms}]$):

| Modelo | Dimensión | LDA Train | 5-Fold CV | /a/ | /e/ | /i/ | /o/ | /u/ | Diagnóstico Geométrico |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PCA (Lineal)** | **3D** | $64.4\%$ | $60.6\% \pm 6.8\%$ | $50.0\%$ | $47.2\%$ | $61.1\%$ | $77.8\%$ | $86.1\%$ | Hiperplano rígido euclídeo. Solapa /e/ con /a/. |
| **Isomap ($k=16$)** | **3D** | **$79.4\%$** | **$78.3\% \pm 2.7\%$** | **$86.1\%$** | **$61.1\%$** | **$88.9\%$** | **$80.6\%$** | **$80.6\%$** | **ÓPTIMO 3D**: Desenrolla la curvatura labio-facial. |
| **Isomap ($k=16$)** | **5D** | **$82.8\%$** | **$81.7\% \pm 3.8\%$** | $92.0\%$ | $58.0\%$ | $89.0\%$ | $92.0\%$ | $83.0\%$ | Despliega grados secundarios de libertad. |
| **Isomap ($k=16$)** | **6D** | **$84.4\%$** | **$84.4\% \pm 6.2\%$** | $83.0\%$ | **$67.0\%$** | $89.0\%$ | **$94.0\%$** | $89.0\%$ | **MÁXIMO GLOBAL**: Cada canal despliega su eje propio. |

---

## 4. Hallazgos de Ablación y Redundancia

1. **Tríada Canónica Invariable:**
   La remoción de Ch1 (Apertura) o Ch2 (Sonrisa) desploma la CV en $-19.4\,\text{pp}$. La remoción de Ch3 (Esfínter) la hace caer en $-13.3\,\text{pp}$.
2. **Candidatos Descartados:**
   - **Deformación Submentoniana:** Ruido puro por oclusión geométrica de la cámara frontal ($35\%$ CV aislada).
   - **Protrusión Sagital $Z$:** Estimación monocular sintética que no añade información ortogonal ($0.0\,\text{pp}$ de ganancia).
   - **Espesor de Bermellón:** Excelente para separar /e/ de /i/ en aislamiento, pero al sumarse como 7mo canal genera atajos colineales que bajan la CV a $64.2\%$.

---

## 5. Figuras y Láminas Generadas

- `plot_patrones_6canales_oficial.png`: Lámina completa de 6 filas cinemáticas + micrófono RMS 90 ms.
- `plot_isomap_6canales_ganador.png`: 3 proyecciones ortogonales de la variedad Isomap 3D ($k=16$).
- `plot_comparativa_7canales_pca_vs_isomap.png`: Evaluación de PCA vs Isomap sobre deformación sagital.
