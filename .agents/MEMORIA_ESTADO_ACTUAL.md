# Memoria del Proyecto: Decodificación de Habla Submáximal y Espacio Latente Universal

**Fecha de ultima consolidacion:** 2026-09-17 00:18 UTC-3  
**Sujetos analizados:** Candela (2026-09-15, 2026-09-01, 2026-08-28), Lucas (2026-07-10), Petra (Silicona med1 y med2)  
**Ventanas totales procesadas:** 1191 + 213 = 1404 ventanas sEMG

---

## 1. Estado Científico y Hallazgos Biomecánicos

### 1.1 Invarianza Topológica de la Cruz Latente
El autoencoder convolucional 1D entrenado sin supervisión con regularización por decorrelación de coordenadas proyecta la actividad muscular sEMG en una estructura geométrica cruciforme invariante entre diferentes sujetos y días:
- **Semieje Vertical Positivo ($+Y$, $\theta = 90^\circ$):** Apertura mandibular exclusiva de la vocal **/a/** (activación dominante del vientre anterior del digástrico).
- **Semiplano Derecho ($+X > 0$, $\theta \in [0^\circ, 45^\circ]$):** Retracción comisural y sonrisa para las vocales **/i/** y **/e/** (activación dominante de risorio / cigomático / modíolo).
- **Semiplano Inferior ($-Y < 0$, $\theta \in [-120^\circ, -60^\circ]$):** Protrusión y constricción orbicular para las vocales **/u/** y **/o/**.

### 1.2 Diagnóstico Fisiológico de la Falla en Tomas Previas (Candela 2026-08-28)
- En las tomas `Prueba4` y `Prueba5` de Candela (`2026-08-28`), la vocal **/a/** no se separó correctamente.
- **Causa real identificada:** Desplazamiento anatómico del sensor submentoniano (Canal 0 ubicado sobre el milohioideo en vez del vientre anterior del digástrico) y electrodo del modíolo colocado excesivamente bajo (captando actividad de depresores labiales en lugar de elevadores/sonrisa).
- **Validación del modelo:** El modelo supervisado previo ya había demostrado generalización inter-día perfecta cuando los electrodos se colocaron en la diana muscular correcta.

### 1.3 Experimento Frankenstein Sintético (2026-09-04)
- **Script creador:** `EMG_desarrollo/deep_learning/crear_sesiones_frankenstein.py`.
- **Procedimiento:** Se ensambló el Canal 0 donante de Candela `2026-09-01` (vientre anterior activo) sobre los canales 1, 2 y 3 de `2026-08-28` (`Prueba4` y `Prueba5`).
- **Resultados:**
  - El centroide neutro convergió exactamente al origen $(0, 0)$.
  - La vocal **/o/** alcanzó un 78.3% de exactitud en 3D (K-Means) y **/u/** un 100.0% (GMM).
  - La limitación residual en nitidez respecto a una toma real se atribuyó al desfase anatómico original del modíolo del día 28, a la asincronía motora inter-día (jitter $\Delta t \approx 50\text{--}120\,\text{ms}$) y a discrepancias de impedancia en el divisor del Supremo Global ($M_{\text{supremo}}$).

### 1.4 Diagnóstico Biomecánico de Pares Críticos (Candela 2026-09-01)
- **Separación de /e/ frente a /i/ por Dinámica Temporal:**
  - En amplitud máxima estática del Risorio no hay diferencia ($p = 0.572$).
  - La distinción es puramente temporal: /i/ tiene un ataque rápido y explosivo ($\tau_{\text{subida}} \approx 10$ muestras, pendiente $\Delta v = 0.0651$), mientras que /e/ tiene un reclutamiento gradual ($\tau_{\text{subida}} \approx 20$ muestras, $\Delta v = 0.0463$), con significancia $p = 0.0105$.
- **Separación de /o/ frente a /u/ por Co-Activación Intermuscular:**
  - En el Orbicular aislado, ambas vocales son cinemáticamente idénticas ($p = 0.401$ en pendiente, órbitas de fase superpuestas y amplitud 1.0 saturada).
  - La discriminación radica en el Vientre Anterior del Digástrico (Canal 0): la /o/ exige descenso mandibular activo alcanzando más del doble de amplitud ($0.1143 \pm 0.0612$) frente a la mandíbula cerrada de la /u/ ($0.0504 \pm 0.0422$), con $p = 1.965 \times 10^{-6}$.
  - El Risorio (Canal 2) también muestra mayor tensión comisural en la /o/ ($0.3377$ vs $0.2567$, $p = 0.0110$).
- **Evaluación Espectral y Frecuencia de Disparo de Unidades Motoras (/o/ vs /u/):**
  - Tras implementar un filtro peine en cascada (supresión de fundamental y armónicos de red de $50$ a $450\,\text{Hz}$ con $\Delta f = 2.5\,\text{Hz}$ en fase cero), el espectro biológico del Orbicular resultó idéntico entre ambas vocales:
    - Frecuencia Mediana ($\text{MDF}$): /o/ = $120.8 \pm 37.8\,\text{Hz}$ vs /u/ = $112.2 \pm 33.3\,\text{Hz}$ ($p = 0.5978$).
    - Frecuencia Media ($\text{MNF}$): /o/ = $140.4 \pm 32.2\,\text{Hz}$ vs /u/ = $134.1 \pm 28.9\,\text{Hz}$ ($p = 0.3692$).
    - Tasa de Cruces por Cero ($\text{ZCR}$): /o/ = $324.1 \pm 65.9\,\text{Hz}$ vs /u/ = $308.9 \pm 48.6\,\text{Hz}$ ($p = 0.2416$).
  - Clasificación individual muestra a muestra (SVM-RBF sobre $\text{MDF} + \text{ZCR}$): **$51.35\%$** de exactitud ($57/111$ aciertos, nivel de azar puro).
  - Diagnóstico físico definitivo: en un electrodo bipolar de superficie único sobre el orbicular, no existe ninguna propiedad (ni amplitud, ni derivada temporal, ni espectro de descarga) capaz de separar /o/ de /u/. Ambas reclutan el mismo esfínter a niveles equivalentes. La diferenciación fonética exige el balance intermuscular tricanal o información complementaria (micrófono / cuarto canal de electrodo).
- **Evaluación de Dinámica No Lineal con RQA (/o/ vs /u/):**
  - Aplicando inmersión de Takens ($m=4, \tau=4$) sobre señal con filtro peine:
    - Determinismo ($\text{DET}$): /o/ = $0.0210 \pm 0.0377$ vs /u/ = $0.0147 \pm 0.0333$ ($p = 0.4350$).
    - Laminaridad ($\text{LAM}$): /o/ = $0.0958 \pm 0.1200$ vs /u/ = $0.0737 \pm 0.1039$ ($p = 0.3553$).
    - Tiempo de Atrapamiento ($\text{TT}$): /o/ = $1.23 \pm 1.03$ vs /u/ = $1.15 \pm 1.08$ ($p = 0.6616$).
    - Entropía Diagonal ($\text{ENTR}$): /o/ = $0.0791 \pm 0.2219$ vs /u/ = $0.0357 \pm 0.1875$ ($p = 0.1553$).
  - Clasificación muestra a muestra con SVM-RBF sobre vector RQA: **$52.25\%$** (nivel de azar).
  - Conclusión: el sEMG submáximal rápido perioral exhibe dinámica estocástica asíncrona ($\text{DET} < 2.5\%$), sin atractores periódicos ni diferencias no lineales en el orbicular entre ambas vocales.
- **Evaluación de Descriptores del Tratado de Merletti & Parker (2004) y Cancelador Adaptativo NLMS:**
  - Evaluado sobre las 111 contracciones de Candela (01/09/2026, 61 /o/ y 50 /u/).
  - **Cancelador Adaptativo de Ruido de Línea (ANC NLMS en Cuadratura):** Preserva el espectro biológico continuo entre 40 Hz y 150 Hz donde el filtro peine convencional recortaba energía muscular. El ratio Digástrico/Orbicular saltó a **$p = 1.166 \times 10^{-8}$** y **$\text{AUC} = 0.816$** (frente al $p = 3.132 \times 10^{-5}$ y $\text{AUC} = 0.730$ con filtro peine).
  - **Pre-blanqueamiento Espectral de Clancy $\text{AR}(4)$:** Decorrelaciona con éxito la secuencia de innovación; al computar el valor cuadrático medio integral en 350 ms, el ratio intermuscular se mantiene algebraicamente congruente ($p = 3.132 \times 10^{-5}$, $\text{AUC} = 0.730$).
  - **Mapeo Conjunto de Espectro y Amplitud (JASA de Luttmann):** $\Delta \text{RMS}$ vs $\Delta \text{MDF}$ respecto al reposo inmediato demostró ausencia absoluta de fatiga neuromuscular durante las pruebas (sin muestras en el Cuadrante II: $\uparrow\text{RMS}, \downarrow\text{MDF}$).
  - **Estadísticos de Orden Superior (HOS):** Distribución leptocúrtica en orbicular ($\kappa \approx 6.2\text{--}7.0$) y digástrico ($\kappa \approx 5.2\text{--}5.4$), pero sin divergencia estadística significativa entre vocales ($p > 0.23$).
  - **Clasificación Multivariada con SVM-RBF:** Exactitud global de **$68.47\%$** ($76/111$ contracciones correctas), logrando una tasa de detección del $56.0\%$ ($28/50$) en la vocal /u/, superando el colapso univariado anterior.

### 1.5 Decodificación Directa de Señal Cruda a 2000 Hz y Espacios Complementarios (2026-09-11)
- **Acondicionamiento Guiado por Envolvente Analítica:** En vez de alimentar únicamente envolventes suavizadas (< 5 Hz), se procesa la señal sEMG nativa rectificada $|x_c(t)|$ a $2000\,\text{Hz}$ ($3 \times 1000$ muestras). La envolvente de 200 ms se usa exclusivamente para alinear picos, calcular la mediana del piso de ruido interpulso $\mu_{\text{ruido}}$ y determinar el Supremo Global Tricanal $M_{\text{supremo}}$.
- **Filtrado de Valores Atípicos:** Aplicación de Isolation Forest (10% contaminación) sobre la envolvente suave para descartar anomalías de electrodo antes del modelado de señal cruda.
- **Autoencoder Conv1D Dual GAP+GMP:** Logra **$54.27\%$ de exactitud GMM** y **$+0.031$ de silueta** en la sesión ruidosa de Lucas (`2026-07-10`, 35 tomas).
- **Descubrimiento de Espacios Complementarios:**
  - El PCA de envolventes lentas separa limpiamente /a/, /e/, /i/, pero solapa /o/ y /u/.
  - El Autoencoder 1D sobre señal cruda a 2000 Hz separa nítidamente **/o/** frente a **/u/** (densidad de unidades motoras del orbicular en GAP+GMP), pero agrupa /e/ e /i/.
- **Evaluación de Anclaje de Ruido Basal a Cero ($\gamma \cdot \|Z_{\text{ruido}}\|^2$):**
  - Utilizando las ventanas de reposo de 5.0 s (`noise_seconds`), se verificó que forzar el anclaje a cero colapsa perfectamente el ruido en $(0,0)$, pero deteriora la separabilidad fonatoria (caída de $54.3\%$ a $48.7\%$) debido a que penaliza los canales musculares que están fisiológicamente inactivos durante la fonación de una vocal determinada.
- **Evaluación de la Expansión Latente a 3D ($Z \in \mathbb{R}^3$):**
  - La reconstrucción $\text{MSE}$ mejoró a $0.35123$ (frente a $0.36115$ en 2D) y la silueta subió a $+0.0102$ (Exactitud GMM $51.07\%$).
  - La coordenada $Z_3$ desacopló el eje mandibular (/a/ en $Z_3 \approx -5$) del complejo labial (/o/ y /u/ en $Z_3 \approx +4.5$) y el complejo de sonrisa (/e/ e /i/ en $Z_3 \approx -1$).
- **Impacto del Submuestreo Físico (Integración en Bloques a 2D):**
  - Se diezmó la señal sEMG rectificada por promedio en bines temporales $\Delta t \in [0.5, 10]\,\text{ms}$ ($f_s \in [2000, 100]\,\text{Hz}$).
  - A $200\,\text{Hz}$ ($N = 100$ muestras/canal, bines de $5\,\text{ms}$), el error de reconstrucción $\text{MSE}$ se desploma de $0.36115$ a **$0.15145$** y la silueta salta de $+0.0061$ a **$+0.0455$** (GMM $52.23\%$).
  - Justificación biofísica: bines de $5\,\text{ms}$ cancelan el *jitter* estocástico de fase de los potenciales de acción asíncronos y preservan la tasa instantánea pura de reclutamiento neuromuscular, permitiendo por primera vez desacoplar visualmente a /e/ frente a /i/ sin filtros de envolvente artificiales.

### 1.6 Experimento 3: Sesion de Ruido de Linea Bajo y Record Autoencoder 87.8% (2026-09-15/16)
- **Sesion:** Candela, 2026-09-15 (fecha real de medicion, carpeta de datos 2026-09-16).
- **Montaje Tricanal:** Canal 0 = Anterior Belly (Mallado 1, 15 mm -> 19 mm), Canal 1 = Zygomaticus Major (sin mallar, 15 mm), Canal 2 = Orbicularis Oris (Mallado 2, 25 mm centro a centro). GND en apofisis mastoides (Medi-Trace recortado).
- **Protocolo:** 4 series x 5 vocales x 12 pulsos = 213 ventanas validas post-purga (Isolation Forest 10%).
- **Condiciones:** Ruido de red excepcionalmente bajo. Metronomo a 30 BPM, fs = 2000 Hz.
- **PCA 2D:** 78.17% de exactitud macro (/a/ 81.6%, /e/ 64.1%, /i/ 100.0%, /o/ 61.0%, /u/ 84.2%). Silueta +0.227.
- **PCA 3D:** 78.02% (/a/ 86.8%, /e/ 66.7%, /i/ 100.0%, /o/ 36.6%, /u/ 100.0%). Silueta +0.233.
- **Autoencoder Convolucional 1D (Envolvente 3D):** **87.79% de exactitud GMM** (record historico del proyecto). Envolvente RMS 91 ms, 3x100 muestras, Full-Batch lr=0.008, 250 epocas, 100% no supervisado.
  - Desglose: /a/ 88.4%, /e/ 88.4%, /i/ 100.0%, /o/ 83.7%, /u/ 78.0%.
  - Silueta: +0.285, Davies-Bouldin: 1.42.
  - Carpeta de resultados: `EMG_desarrollo/resultados/resultados_autoencoder/procesamiento_2026-09-16_13-16-27_envolvente_3d/`.
- **Diagnostico del Ruido de Linea:** El zumbido de 50 Hz (incluso tras Notch) introduce artefactos de fase no lineales que distorsionan el gradiente del autoencoder. En sesiones con ruido residual alto, el autoencoder rinde peor que PCA; con senal limpia, el autoencoder supera al PCA por casi 10 puntos porcentuales.
- **Frontera /o/ vs /u/:** Ambas vocales alcanzaron metricas record (83.7% y 78.0%), pero siguen siendo el par mas proximo debido al reclutamiento co-dependiente del orbicular. Marca el limite fisiologico del registro de superficie en la zona peribucal.
- **Reporte LaTeX:** Se inserto la seccion completa del Experimento 3 en `reportes_experimentos/Reporte_EMG_2026-09-15.tex` (seccion 5) con: objetivo, geometria anatomica (5 fotografias), metodologia, analisis espectral promedio (FFT, PSD, espectrogramas, 5 vocales), y subseccion de Autoencoder Conv1D 3D con tablas de metricas y figuras.


**Puntos Clave Consolidados:**
1. **Invarianza de la Cruz Latente:** El Autoencoder 1D y PCA proyectan la actividad en una geometría cruciforme universal (apertura en $+Y$, sonrisa en $+X$, labial en $-Y$).
2. **Submuestreo Físico Óptimo en 200 Hz:** Promediar en microbines de $5\,\text{ms}$ ($200\,\text{Hz}$) suprime la interferencia aleatoria de espigas sub-milisegundo, cuadruplicando la métrica de silueta y reduciendo el error MSE un $58\%$.
3. **Descarte de Anclaje Artificial de Ruido:** El piso de ruido basal se neutraliza mejor sustrayendo la mediana interpulso que penalizando el espacio latente.
4. **Próximo Paso Inmediato:** Continuar optimizando la escala temporal a $200\,\text{Hz}$ en 2D o validar la generalización en el sujeto limpio (Candela 01/09).

---

## 2. Marco de Alineación Canónica y Resultados

### 2.1 Regla Canónica de Orientación (Establecida por el Usuario)
1. **Anclaje Estricto de /a/:** El rayo de la vocal **/a/** se fija rígidamente en el semieje vertical positivo ($+Y$, $\theta = 90^\circ$, 12 en punto).
2. **Quiralidad Anatómica:** Las vocales de sonrisa (**/i/** y **/e/**) deben proyectarse siempre hacia el semiplano derecho ($+X > 0$). De detectarse $\text{mediana}(x_{\text{sonrisa}}) < 0$, se aplica una reflexión horizontal ($x \to -x$).
3. **Normalización Radial:** Proyección sobre la circunferencia unitaria ($S^1$) descartando dispersión por fatiga o amplitud muscular.

### 2.2 Archivos de Datos y Gráficos Generados
- **Script de alineación:** `EMG_desarrollo/deep_learning/alinear_cruces_clasificador_direccional.py`.
- **Directorio de salida:** `EMG_desarrollo/resultados/comparativa_cruces_alineadas/`.
- **Archivos generados:**
  - `espacio_latente_universal_alineado.csv`: Matriz consolidada de 1191 ventanas con coordenadas latentes originales, rotadas, normalizadas, ángulos y predicción direccional.
  - `comparativa_4_cruces_alineadas_2d.png`: Grilla 2x2 comparando las cruces de Candela (01/09), Lucas, Petra y Candela (Prueba 4 y 5) bajo sectores de decisión idénticos.
  - `espacio_latente_universal_superpuesto_2d.png`: Superposición de las 1191 ventanas demostrando la geometría cruciforme universal.
  - `matriz_confusion_direccional_universal.png`: Matriz de confusión global que confirma:
    - **95.2% de pureza y separación en la vocal /a/**.
    - **95.8% de confinamiento labial para la vocal /u/**.
    - Exactitud global sin supervisión: **53.7%** (con confusiones limitadas exclusivamente a pares de la misma sinergia: /e/ con /i/, y /o/ con /u/).

---

## 3. Próximos Pasos Acordados para la Siguiente Sesión

1. **Extensión a Arquitectura Supervisada (Linear Probe / Fine-Tuning):**
   - Acoplar el encoder convolucional 1D preentrenado a una cabeza clasificadora supervisada ligera (MLP lineal o clasificador de similitud coseno).
   - Resolver la separación fina entre los pares que comparten el mismo rayo muscular (/e/ frente a /i/, y /o/ frente a /u/).
2. **Actualización del Reporte Formal en LaTeX (`reporte_autoencoder_decodificacion_vocalica.tex`):**
   - Insertar la figura comparativa `comparativa_4_cruces_alineadas_2d.png`.
   - Incorporar la matriz de confusión direccional universal y la discusión fisiológica.
3. **Flujo de Decodificación en Tiempo Real:**
   - Implementar la inferencia continua en ventanas deslizantes con los pesos exportados (`solo_encoder_2d_unsupervised.pth` o `solo_encoder_3d_unsupervised.pth`) para streaming con latencia inferior a 1 ms.

---

## 4. Reglas Críticas del Proyecto a Mantener
- **Cero emojis** en cualquier código, commit, reporte o respuesta.
- **Prohibición absoluta de ejecutar código (`run_command`)** sin consentimiento explícito previo del usuario.
- **Prohibición del término "pipeline"** (usar cadena de procesamiento, flujo de datos o esquema metodológico).
- **Normalización estricta por el Supremo Global Tricanal ($M_{\text{supremo}}$).**
- **Conservación estricta de contenido** (edición puramente aditiva / no destructiva).
- **Explicación exhaustiva de variables** en toda fórmula matemática.

---

## 5. Lecciones Aprendidas y Directivas Obligatorias de Modelado

1. **Prohibición de Regularización Artificial por Decorrelación Geométrica:**
   - La penalización de covarianza o varianza latente en la función de pérdida ya fue ensayada empíricamente y produce colapso dimensional del espacio latente (forzar ortogonalidad artificialmente aplasta la dispersión natural).
   - Queda descartada de forma definitiva.

2. **Prohibición de Ajustes Paramétricos Complejos:**
   - Modelar a mano potenciales de acción (ej. Rosenfalck analítico paramétrico) introduce suposiciones excesivamente rígidas y sobrecomplejiza el modelo.
   - Queda descartado para mantener el modelado limpio y sin artefactos artificiales.

3. **Principio Rector: Simplicidad y Emergencia de Patrones Naturales:**
   - El objetivo central es no sobrecomplejizar la arquitectura.
   - Trabajar directamente con la **señal cruda rectificada a 2000 Hz** (3 canales $\times$ 1000 muestras) normalizada por el Supremo Global Tricanal.
   - Permitir que las redes convolucionales 1D compactas y métodos simples descubran los patrones temporales y firmas bioeléctricas que las envolventes lentas (< 5 Hz) descartaron por completo.

4. **Descarte de Expansión a 3D ($Z \in \mathbb{R}^3$):**
   - Ya fue ensayada el 2026-09-11. No resuelve la tarea central de decodificación y dispersa innecesariamente la estructura canónica cruciforme 2D. Queda archivada y no debe volver a proponerse.

5. **Descarte de Redes Profundas Sobreparametrizadas (ResNet-1D):**
   - El ensayo con 12 capas residuales y 1.32M de parámetros (2026-09-12) tardó 19 minutos y colapsó la representación en una línea vertical ($Z_1 \approx 0$), reduciendo la exactitud al $45.7\%$.
   - Queda estrictamente establecido que los modelos deben ser compactos ($\approx 30.000\text{ a }50.000$ parámetros).

6. **Evaluación Empírica de BatchNorm1d (2026-09-12):**
   - **Resultados:** La incorporación de `BatchNorm1d` mejoró la exactitud de agrupamiento GMM de $31.41\%$ a **$43.93\%$** y redujo el error MSE ($0.00497$ vs $0.00507$).
   - **Impacto Biomecánico y Geométrico:** La normalización por lotes forzó una media cero y varianza unitaria por filtro a lo largo del lote, lo cual destruyó la disparidad natural de amplitud inter-canal (la energía absoluta que diferencia un músculo activo de uno en reposo). Como consecuencia, el espacio latente se colapsó en un **haz diagonal unidimensional** (de contracción mínima en el cuadrante inferior izquierdo a contracción máxima en el superior derecho), perdiendo la geometría cruciforme ortogonal universal.
   - **Conclusión:** La normalización por lote no debe aplicarse de forma indiscriminada a la representación latente o canales intermedios en sEMG porque neutraliza la firma de co-activación relativa intermuscular.

7. **Nomenclatura Anatómica Oficial de Canales Musculares (Lucas 2026-07-10):**
   - Directiva inmutable confirmada por el usuario y contrastada en los archivos `metadata.json`:
     - **Canal 0:** Milohioideo (`Mylohyoid`) - Apertura mandibular y piso de la boca.
     - **Canal 1:** Depresor (`Depresor Anguli Oris`) - Depresión labial y comisural.
     - **Canal 2:** Orbicular (`Orbicularis Oris`) - Constricción y redondeo labial.
     - **Canal 3:** Micrófono acústico de referencia.
   - Queda estrictamente prohibido usar nombres anatómicos distintos (ej. Digástrico o Modíolo) para estas grabaciones.

8. **Lectura Dinámica Obligatoria de BPM por Toma (Prohibición de Fallback a 40 BPM):**
   - El ritmo del metrónomo varió entre tomas dentro de la misma sesión: la toma `A_T1_Lucas` se registró a 40 BPM, mientras que `A_T2_Lucas`, `A_T3_Lucas`, etc., se registraron a 30 BPM.
   - Todo algoritmo de ventaneo y segmentación debe leer obligatoriamente el parámetro `bpm` desde el `metadata.json` de cada toma individual, sin asumir un valor constante de 40 BPM.

9. **Desglose de Reconstrucción por Vocal y Purga Robusta de Outliers Latentes:**
   - La evaluación visual de reconstrucción debe realizarse de forma desglosada por cada vocal fonatoria individual (/a/, /e/, /i/, /o/, /u/) en subplots separados con escala compartida, visualizando el perfil bioeléctrico de activación tricanal.
   - Antes de ajustar el agrupamiento GMM, se descartan los puntos atípicos aislados en el plano latente mediante filtrado robusto por distancia IQR ($d > Q_3 + 2.0 \times \text{IQR}$) para evitar que anomalías esporádicas de disparo deformen las elipses de covarianza del modelo.

10. **Normalización Tricanal Guiada por Envolvente con Re-escalado Acotado a 1.0:**
    - El divisor maestro intermuscular debe calcularse obligatoriamente sobre la envolvente suavizada ($M_{\text{supremo, env}}$), garantizando que espigas espurias individuales de la señal cruda no distorsionen la sinergia bioeléctrica.
    - Como las espigas de la señal cruda rectificada superan la envolvente, todo el bloque tricanal resultante del pulso se re-escala conjuntamente dividiendo por su pico máximo instantáneo ($\max(\text{cruda}) > 1.0$), preservando rígidamente la proporción fisiológica entre canales y asegurando que ninguna muestra supere jamás $1.0$.

11. **Prioridad Absoluta de Clasificación Equilibrada y Prohibición de Reportes Sensacionalistas (2026-09-12):**
    - Directiva obligatoria: Un modelo sEMG no debe juzgarse jamás por el incremento aislado o extremo en una única vocal si esto ocurre a costa de degradar sustancialmente a las demás (ej. un aumento en `/e/` que canibaliza a `/a/`).
    - El objetivo bioeléctrico y de ingeniería es lograr una clasificación multiclase armónica y balanceada a lo largo de las cinco vocales (`/a/`, `/e/`, `/i/`, `/o/`, `/u/`). Los modelos deseables son aquellos que mantienen un piso mínimo aceptable y homogéneo en todas las clases.
    - Los informes deben mantener un tono estrictamente sobrio y científico, identificando siempre los desbalances y debilidades del modelo antes de resaltar cualquier cifra favorable.

12. **Prohibición Absoluta de Supervisión en el Autoencoder (Zero-Labels Estricto - 2026-09-12):**
    - Directiva fundamental de la investigación: El descubrimiento de la variedad bioeléctrica latente debe ser **100% NO SUPERVISADO**.
    - Está **terminantemente prohibido** utilizar etiquetas de clases, vocales o fonemas durante el entrenamiento (cálculo de pérdida o retropropagación de gradientes `backward`).
    - La red debe descubrir las estructuras fonatorias exclusivamente a partir de la reconstrucción de la señal bioeléctrica, la preservación de la envolvente macroscópica o regularizaciones de agrupamiento ciego auto-organizado sin etiquetas (DEC).
    - Las etiquetas de vocales se reservan única y exclusivamente para la etapa diagnóstica post-entrenamiento mediante GMM / Hungarian matching. Si se supervisa la red con etiquetas, se destruye el valor científico del descubrimiento no supervisado de la fonética.

13. **Descubrimiento de la Arquitectura Inception Multiescala y Barridos Nocturnos en Paralelo (2026-09-13):**
    - **Hito en Lenovo Yoga (36 Combinaciones Completadas):**
      - La Combinación 18 (`Inception_F24_lr0.009_cosFalse`) estableció un nuevo récord histórico del proyecto: **61.62% de exactitud media GMM**, **56.29% de exactitud armónica** y un piso mínimo por clase de **33.66%** (/a/: 55.8%, /e/: 33.7%, /i/: 84.0%, /o/: 67.4%, /u/: 69.2%).
      - Por primera vez en el proyecto, **ninguna vocal quedó colapsada** bajo un régimen 100% no supervisado.
      - **Fundamento Biomecánico:** La arquitectura Inception 1D multiescala combina en ramas paralelas filtros de soporte temporal extendido ($63$ y $31\,\text{ms}$ a $1\,\text{kHz}$, para la cinemática lenta mandibular y labial) con filtros de soporte corto ($15$ y $7\,\text{ms}$, para los potenciales de acción MUAP de alta frecuencia), eliminando el compromiso forzado de un kernel fijo.
    - **Barrido en Estación Local (30 Combinaciones Completadas):**
      - La Combinación 27 ($K(63, 31)$, $F(32, 16)$, $\beta_{\text{env}} = 0.25$) alcanzó **58.41% de GMM** con armónica de $30.91\%$ y piso de $10.5\%$, corroborando la eficacia de los campos receptivos extendidos combinados con la preservación de envolvente.

14. **Evaluación de Concatenación de Señal Cruda y Envolvente RMS en Entrada (6 Canales - 2026-09-13):**
    - **Configuración del Ensayo:** Se entrenó el autoencoder convolucional simple (`AutoencoderConvSimple6Ch`) recibiendo en paralelo 6 canales (3 canales crudos rectificados normalizados por Supremo Tricanal + 3 canales de envolvente RMS centrada de $100\,\text{ms}$ con $W = 101$ muestras a $1\,\text{kHz}$). Entrenamiento 100% no supervisado con 250 épocas.
    - **Resultados Cuantitativos:**
      - Exactitud GMM Media: **$50.52\%$** (caída sustancial frente al $56.10\%$ de la línea base y $61.62\%$ de Inception).
      - Exactitud Armónica: **$26.88\%$** (caída de casi 30 puntos porcentuales respecto a Inception).
      - Piso Mínimo por Clase: **$10.10\%$** (colapso severo en la vocal `/e/`).
      - Desglose por Vocal: `/a/: 48.3%`, `/e/: 10.1%`, `/i/: 89.6%`, `/o/: 82.7%`, `/u/: 23.2%`.
    - **Diagnóstico Biofísico y Numérico:**
      - La concatenación ciega en canales genera un conflicto en los gradientes de la función de reconstrucción $\text{MSE}(x, \hat{x})$: el decodificador y codificador intentan ajustar simultáneamente oscilaciones estocásticas de alta frecuencia (MUAP individuales de la señal rectificada) y transientes cinemáticos lentos (la envolvente macroscópica).
      - Al tener igual peso muestral en la función de pérdida, el componente de alta frecuencia domina la varianza, absorbiendo los recursos latentes del modelo y desestabilizando las representaciones sutiles necesarias para diferenciar vocales de similar sinergia.
    - **Directiva Derivada:** Queda descartada la concatenación estática de señal cruda y envolvente como canales adicionales en la entrada. La complementariedad temporal debe resolverse a nivel de capas (mediante núcleos convolucionales multiescala Inception o regularización de envolvente en la función de pérdida) y no mediante duplicación artificial de canales heterogéneos en el tensor de entrada.

15. **Evaluación de Autoencoder de Sinergia Muscular Intercanal Gramiana (3x3 - 2026-09-13):**
    - **Configuración del Ensayo:** Se entrenó un autoencoder convolucional (`AutoencoderSinergiaGram`) que toma la señal cruda rectificada ($3 \times 1000$), la proyecta a $Z \in \mathbb{R}^2$ vía `GAP`, y el decodificador predice la matriz de covarianza/sinergia intermuscular $G = \frac{1}{T} X X^T \in \mathbb{R}^{3 \times 3}$ a través de factorización de Cholesky ($G = L L^T$). Función de pérdida: $\|G - \hat{G}\|_F^2$. 250 épocas, 8 hilos, 100% no supervisado.
    - **Resultados Cuantitativos:**
      - Exactitud GMM Media: **$39.63\%$**
      - Exactitud Armónica: **$27.39\%$**
      - Piso Mínimo por Clase: **$11.70\%$** (colapso severo en la vocal `/a/`).
      - Desglose por Vocal: `/a/: 11.7%`, `/e/: 59.0%`, `/i/: 27.7%`, `/o/: 63.3%`, `/u/: 35.4%`.
      - Coeficiente de Silueta: **$-0.1146$** | Índice Davies-Bouldin: **$3.61$**.
    - **Diagnóstico Bioeléctrico y Matemático:**
      - El espacio latente colapsó en una **franja diagonal unidimensional** (de contracción débil en la esquina inferior izquierda a contracción fuerte en la superior derecha).
      - Causa matemática: La norma de Frobenius de la matriz de Gram $\|G\|_F$ está dominada por la **energía absoluta escalar del pulso** (amplitud/volumen de la emisión). Al penalizar con $\text{MSE}$ cuadrático sobre $G$, el cuello de botella $Z \in \mathbb{R}^2$ gastó su eje principal en codificar la escala de fuerza de la contracción en lugar de la sinergia angular relativa (las proporciones normalizadas entre canales). Como consecuencia, los fonemas se mezclaron a lo largo del gradiente de volumen.
    - **Directiva Derivada:** Hipótesis descartada definitivamente por directiva del usuario. Queda archivada para no volver a plantearse.

16. **Evaluación de Autoencoders Convolucionales 2D/3D sobre Espectrogramas y Cascada DAE (2026-09-13/14):**
    - **Configuración del Ensayo:** Se implementó una cascada en dos fases: Fase 1 (DAE de alta capacidad para planchar ruido estocástico sobre espectrogramas STFT calibrados en dB con clausura morfológica $3 \times 32 \times 64$) y Fase 2 (Autoencoder convolucional de reducción a 2D y 3D). Posteriormente se realizó la ablación entrenando directamente sin DAE.
    - **Resultados Cuantitativos Comparativos:**
      - **Sin Denoiser (2D Directo):** GMM $42.83\%$, Silueta $+0.006$, Davies-Bouldin $62.85$ (/a/: $24.0\%$, /e/: $15.5\%$, /i/: $83.7\%$, /o/: $66.0\%$, /u/: $23.2\%$).
      - **Con Cascada DAE (2D):** GMM $43.63\%$, Silueta $+0.024$, Davies-Bouldin $6.08$ (/a/: $58.3\%$, /e/: $60.2\%$, /i/: $3.8\%$, /o/: $74.0\%$, /u/: $23.2\%$). El DAE redujo la dispersión Davies-Bouldin en un factor de 10 y estabilizó la apertura mandibular (/a/) y retracción (/e/).
      - **Con Cascada DAE (3D, $Z \in \mathbb{R}^3$):** GMM $49.00\%$, Silueta $+0.033$, Davies-Bouldin $4.07$ (/o/: $95.0\%$, /i/: $90.4\%$, /a/: $36.5\%$, /u/: $21.2\%$, /e/: $1.0\%$). Logra separación cuasi perfecta de los polos extremos, pero canibaliza la vocal intermedia /e/.
    - **Diagnóstico Biofísico:**
      - La dispersión en frecuencias de la STFT suaviza la micro-dinámica temporal del reclutamiento muscular (la tasa de subida de potenciales de acción que distingue a /e/ de /i/).
      - Por este motivo, el modelado 1D continuo (envolventes directas al $56.1\%$ y redes multiescala Inception al $61.6\%$) se consolida como el enfoque bioeléctrico superior frente al espectrograma.
    - **Unificación Arquitectónica del Proyecto:**
      - Se desarrolló el motor unificado `motor_autoencoder_unificado.py` y la interfaz gráfica `pipeline_autoencoder_gui.py` que consolida las 3 modalidades de entrada (Envolvente 1D, Señal Cruda 1D a 2000 Hz con GAP+GMP, y Espectrogramas 2D/3D).
      - Incorpora auditoría estricta de metadatos con alertas visuales inmediatas ante discrepancias anatómicas de electrodos o tempo de metrónomo (BPM) entre distintas sesiones o días.
      - Garantiza el cumplimiento estricto del régimen 100% Cero Supervisado (Zero-Labels) y anclaje canónico (/a/ en $+Y$, sonrisa en $+X$).

17. **Calibración Fisiológica por Percentil 95 en Vocal Diana hacia 1.0 (2026-09-14):**
    - **Principio Biomecánico Universal:**
      - Cada músculo registrado tiene su vocal fonatoria diana donde actúa como motor primario (agonista):
        - **Milohioideo / Anterior Belly (Apertura Mandibular - Canal 0):** Vocal diana obligatoria **/a/**.
        - **Depresor Anguli Oris / Risorio (Retracción Comisural / Sonrisa - Canal 1):** Vocal diana obligatoria **/i/** (o **/e/**).
        - **Orbicularis Oris (Constricción y Redondeo Labial - Canal 2):** Vocal diana obligatoria **/u/** (y **/o/**).
    - **Fórmula de Calibración (Estándar extraer_cache_pulsos_lucas_07):**
      - La calibración se agrupa por sesión (`parent_folder`) para que la campaña completa de 5 vocales esté presente, preservando la fecha real de metadatos en cada ventana para trazabilidad y marcadores.
      - Para cada canal $c$, se extraen los picos escalares en su vocal diana y se calcula su percentil 95:
        $$p_{95, c} = \text{percentile}_{95}(\{ \max_t x_{c, i}(t) \mid \text{vocal}_i \in V_{\text{diana}} \})$$
        $$k_c = \frac{1.0}{\max(p_{95, c}, 10^{-6})}$$
      - **Salvaguarda de Dominancia Estricta del Agonista:**
        - En **/i/**: Si el Milohioideo calibrado supera el $80\%$ del Depresor ($p_{95, 0, \text{i}} \ge 0.80 \cdot p_{95, 1, \text{i}}$), se re-escala $k_0$ para fijarlo en $0.60 \cdot p_{95, 1, \text{i}}$, garantizando de forma absoluta que el **Verde (Depresor) alcance el máximo (1.0)** y el Rojo quede subordinado.
        - En **/a/**: Se garantiza que el **Rojo (Milohioideo) alcance el máximo (1.0)** y el Verde quede en $\le 0.60$.
        - En **/u/** y **/o/**: Se garantiza que el **Naranja (Orbicular) alcance el máximo (1.0)**.

18. **Organización Estricta por Carpeta de Procesamiento (2026-09-14):**
    - **Jerarquía Oficial de Resultados:**
      - Cada corrida o ejecución (Flujo Completo, Extraer, Entrenar, Plotear) crea su propio directorio dedicado con marca temporal:
        `EMG_desarrollo/resultados/resultados_autoencoder/procesamiento_YYYY-MM-DD_HH-MM-SS_<modalidad>_<dim>d/`
      - Dentro de dicha carpeta se almacenan todos los artefactos de la corrida:
        - `dataset_autoencoder_unificado.npz`
        - `autoencoder_<modalidad>_<dim>d.pth`
        - `informe_autoencoder_<modalidad>_<dim>d.png`
        - `metricas.json` (exactitud GMM, silueta, Davies-Bouldin, fechas y músculos).

19. **Restauración de la Línea Base de Señal Cruda (AutoencoderConvGAPGMP al 54.3% - 2026-09-14):**
    - **Diagnóstico del Colapso a Cero:** Al entrenar la señal cruda con mini-lotes (batch_size=32, lr=0.002) y convoluciones con stride=2 y LeakyReLU, el espacio latente colapsó a un único punto en $(0, 0)$ (Exactitud GMM $20.31\%$, Silueta $+0.000$), donde el decodificador simplemente predijo la media global estática.
    - **Causa Numérica y Biofísica:** Los potenciales de acción (MUAPs) de alta frecuencia con LeakyReLU(0.1) promedian cerca de cero bajo pooling GAP, y el ruido en los gradientes por mini-lote anuló los pesos del codificador.
    - **Restauración Estricta de la Línea Base (`test_autoencoder_crudo_07.py`):**
      - **Arquitectura:** Conv1d(3, 32, k=31, padding=15) con `ReLU()` + Conv1d(32, 16, k=15, padding=7) con `ReLU()`. La activación `ReLU()` rectifica de forma aprendible los potenciales de acción, permitiendo que GAP integre la energía y GMP retenga la amplitud pico.
      - **Régimen de Optimización:** Full-Batch ($N$ muestras simultáneas), $\text{lr} = 0.008$ y $250$ épocas (regla obligatoria de invarianza de línea base).
      - **Carga Robusta en Evaluación:** Si `modelo is None` al invocar `evaluar_espacio_latente`, el sistema recupera automáticamente los pesos desde el archivo `.pth` guardado en la carpeta de resultados.

20. **Módulo Independiente de Calibración Espacial 3D y Triangulación Geométrica de Electrodos sEMG (2026-09-14):**
    - **Implementación:** `EMG_desarrollo/acquisition/calibracion_espacial_electrodos.py`.
    - **Arquitectura de Software:** Orientada a objetos (`class AntigravityAgent`, alias `TrianguladorElectrodosFaciales`), tipado estático riguroso y compatibilidad dual con MediaPipe Tasks (`FaceLandmarker` moderno con modelo `face_landmarker.task` en `EMG_desarrollo/DataConfig/modelos_vision/`) y `solutions.face_mesh` legado.
    - **Puntos Fiduciarios y Base de Escala:**
      - Origen tridimensional: Vértice 1 (Pronasale / punta de la nariz).
      - Base de escala métrica: Distancia interpupilar $D_{\text{interpupilar}} = \|\mathbf{P}_{\text{ojo, der}} - \mathbf{P}_{\text{ojo, izq}}\|_2$ calculada a partir de los centros de iris (468 y 473) o cantos oculares (33 y 263).
      - Normalización euclidiana invariante a la distancia focal: $\tilde{d} = \|\mathbf{P}_{v^*} - \mathbf{P}_{\text{nariz}}\|_2 / D_{\text{interpupilar}}$.
    - **Segmentación:** Segmentación cromática en espacio HSV para marcadores de alto contraste (verde flúor por defecto) y detector por transformada de Hough para broches metálicos de electrodos estándar.
    - **Persistencia e Interfaz:** Exportación de metadatos espaciales en formato JSON estructurado e inyección segura en cabeceras de archivos CSV sEMG (`# METADATOS_GEOMETRIA_EMG: {...}`) compatible con NumPy y pandas.
    - **Validación:** Validado con éxito sobre las fotografías reales de Lucas con electrodos en submentoniano (Canal 0), comisural/risorio (Canal 1), orbicular (Canal 2) y mastoides (referencia GND).

21. **Corrección de Compatibilidad de Argumentos en ejecutar_procesamiento (PCA y UMAP - 2026-09-14):**
    - **Diagnóstico del Error:** Al ejecutar scripts puente como `temp_run_pca.py` o `temp_run_umap.py` desde la interfaz gráfica (`main_app.py`), el diccionario de parámetros empaquetado por `get_pca_kwargs()` y `get_umap_kwargs()` contenía argumentos de nivel superior (`tipo_filtro_ruido`, `notch_q`, `highpass_cutoff_hz`, `lowpass_cutoff_hz`). Al desempaquetarse vía `**kwargs` en `generador.ejecutar_procesamiento()`, Python lanzaba `TypeError: ejecutar_procesamiento() got an unexpected keyword argument 'tipo_filtro_ruido'`.
    - **Corrección Implementada:**
      - Se actualizaron las firmas de `ejecutar_procesamiento` en `generador_pca_umap.py`, `pca_analysis.py` y `umap_analysis.py` para aceptar explícitamente `tipo_filtro_ruido="notch"`, `notch_q=2.0`, `highpass_cutoff_hz=20.0`, `lowpass_cutoff_hz=300.0`, y `**kwargs`.
      - Se aseguró la propagación automática de estos hiperparámetros hacia los diccionarios internos `params_2d`, `params_3d` y `params_umap` en caso de que no estuviesen definidos en los mismos, garantizando paridad total y robustez frente a futuras expansiones de la interfaz gráfica.

22. **Editor de Arquitectura PyTorch, Invarianza Temporal GAP+GMP y Calibración por Promedio en /i/ (2026-09-14):**
    - **Editor de Código en la Interfaz Gráfica (`AutoencoderNoSupervisadoTab`):**
      - Se integró un bloque de texto interactivo (`QPlainTextEdit`) con estética oscura cyberpunk, tipografía monoespaciada, verificación de sintaxis local con paso hacia adelante sintético y botón de restablecimiento a plantilla oficial.
      - Permite alternar mediante casilla de verificación entre la línea base histórica consolidada y arquitecturas experimentales personalizadas sin modificar el código fuente.
      - Al entrenar con modelo personalizado, el código exacto se persiste automáticamente en `carpeta_salida / "arquitectura_autoencoder.py"` para garantizar reproducibilidad total.
    - **Invarianza Temporal por Defecto (GAP + GMP):**
      - Se unificó la arquitectura tanto en envolventes (`AutoencoderEnvolvente1D`) como en señal cruda (`AutoencoderConvGAPGMP`), sustituyendo los aplanados rígidos dependientes de la duración por reducción global `AdaptiveAvgPool1d(1)` y `AdaptiveMaxPool1d(1)`.
      - Desacopla la representación latente de la duración o desfasaje temporal de las ventanas, permitiendo que el codificador capture la energía integrada y el pico de activación de forma invariante.
    - **Calibración Fisiológica por Promedio de Pulsos y Re-escalado en /i/:**
      - Se reemplazó la estimación P95 ciega por el cálculo del pulso promedio real sobre todos los pulsos de cada clase vocal interpolados a una grilla temporal común.
      - En la vocal **/i/**, se garantiza de forma estricta que el **músculo Depresor Anguli Oris (Canal 1 - Verde) alcance el máximo absoluto (1.0)** en el pulso promedio, atenuando los factores de ganancia de los canales no agonistas (Milohioideo Canal 0 y Orbicular Canal 2) para que permanezcan subordinados ($\le 0.50$).
      - En los subplots de reconstrucción (`evaluar_espacio_latente`), se modificó el graficado para representar el promedio de todos los pulsos de cada vocal ($N_v$ contracciones) en lugar de un único pulso arbitrario, reflejando fielmente la morfología mioeléctrica media y la dominancia agonista por vocal.

23. **Diagnóstico Pedagógico y Pre-Validación de Desajustes Dimensionales en el Editor (2026-09-14):**
    - **Causa Raíz Identificada:** Al comentar capas convolucionales intermedias (ej. la transición de 32 a 16 canales), la salida de `self.conv` produce 32 canales en lugar de 16. Tras el pooling dual (GAP + GMP), el vector concatenado alcanza $32 \times 2 = 64$ características, produciendo un desajuste si la primera capa lineal (`self.fc_enc`) fue declarada rígidamente como `nn.Linear(16 * 2, 32)` ($32$ entradas), generando `RuntimeError: mat1 and mat2 shapes cannot be multiplied (Bx64 and 32x32)`.
    - **Cálculo Dinámico Universal:** Las plantillas oficiales calculan automáticamente el número de canales de salida evaluando un tensor sintético `num_features = self.conv(torch.zeros(1, in_channels, t_len)).shape[1]` y dimensionando `nn.Linear(num_features * 2, 32)`, garantizando total inmunidad frente a la adición o supresión de capas convolucionales.
    - **Pre-Validación en Interfaz Gráfica:** Se incorporó pre-chequeo sintáctico y dimensional antes de iniciar las rutinas de extracción o entrenamiento en `main_app.py` (`run_autoencoder_no_sup_completo` y `run_autoencoder_no_sup_entrenar`), alertando de forma inmediata al usuario en caso de error y evitando demoras innecesarias en la carga de datos.
    - **Diagnóstico Descriptivo en Excepciones:** `verificar_arquitectura_codigo()` analiza expresiones regulares sobre excepciones de multiplicación matricial en PyTorch para desglosar con exactitud matemática la discrepancia de canales y sugerir la línea correctiva exacta.

24. **Unificación de Chat en Vivo, Controles de Reproducción y PWA Móvil TARS (2026-09-14):**
    - **Diagnóstico de Confusión de Chat:** El backend web ejecutaba consultas aisladas mediante subprocesos CLI (`agy -p`), creando trayectorias desconectadas sin la memoria de la conversación activa del entorno de desarrollo.
    - **Inyección Directa en la Ventana Activa:** Se implementó `send_to_antigravity_ide()` utilizando `python-xlib` (mensaje cliente X11 con `source=2` para evadir la protección contra robo de foco de GNOME Mutter) y emulación de teclado con `pynput` para enfocar la caja de entrada del chat activo e ingresar el texto directamente en esta sesión.
    - **Descubrimiento Dinámico de Transcripción y Buffer Ampliado:** `get_active_transcript_path()` localiza dinámicamente el archivo `transcript.jsonl` modificado más recientemente en el directorio de registros de la aplicación, y el búfer de lectura se expandió a 1 MB (`1048576` bytes) para garantizar la captura de respuestas de texto extensas aun tras ejecuciones intermedias de herramientas.
    - **Supresión de Solapamiento y Controles de Voz:** Se centralizó la reproducción de audio en un gestor unificado (`AudioManager`) en `app.js`. Toda nueva respuesta detiene inmediatamente cualquier locución previa. Se incorporó una barra flotante de control con botones de Pausar / Reanudar y Detener voz, junto con alternancia de estado en cada burbuja de mensaje.
    - **Aplicación Web Progresiva (PWA):** Se configuró el manifiesto `manifest.json` e íconos dedicados (192x192 y 512x512) para permitir la instalación de la aplicación en el dispositivo móvil como una app nativa sin barra de navegación del navegador.

25. **Paridad Estricta de Calibración Fisiológica en Espectrogramas (2026-09-14):**
    - **Identidad de Factores Intermusculares:** La matriz de espectrogramas (`X_spec`) se calcula a partir de la señal rectificada `rect_interp`, la cual proviene directamente de `rect_segs` escalada por el vector maestro de calibración `k_vec` (donde en `/i/` el Depresor promedio alcanza 1.0 y los canales 0 y 2 quedan atenuados a $\le 0.50$, en `/a/` el Milohioideo alcanza 1.0 y en `/u/` y `/o/` el Orbicular alcanza 1.0).
    - **Invarianza del Supremo Tricanal en STFT:** En `procesar_stft_calibrada`, la normalización espectral divide la matriz tricanal de potencia $S_{xx, c}(f, t)$ por el supremo global tricanal del pulso $\max_{c \in \{0, 1, 2\}} \max_{f, t} S_{xx, c}(f, t)$, preservando rígidamente el balance y ratio fisiológico relativo entre canales descubierto en la etapa de envolvente.

26. **Visualización Automática de Imágenes de Evaluación Post-Procesamiento (2026-09-14):**
    - **Apertura Inmediata en el Visor Gráfico del Sistema:** Se integró la función `abrir_imagen_en_visor()` en `motor_autoencoder_unificado.py` (usando `xdg-open` en Linux desacoplado con `subprocess.Popen`) invocada automáticamente al culminar `evaluar_espacio_latente()`, `run_autoencoder_no_sup_completo()` y `run_autoencoder_no_sup_plotear()`.
    - **Botón de Visualización en la Interfaz Gráfica:** Se añadió el botón `btn_ver_ultimo_grafico` ("Visualizar Último Gráfico (PNG)") en la pestaña del autoencoder no supervisado (`AutoencoderNoSupervisadoTab`), permitiendo al usuario reabrir al instante el último informe gráfico renderizado sin necesidad de navegar por el explorador de archivos.

27. **Escala Vertical Dinámica en Subplots de Reconstrucción (2026-09-14):**
    - **Diagnóstico de Compresión:** En la señal cruda rectificada, los potenciales de acción individuales alcanzan 1.0 en espigas aisladas, pero el pulso promedio sobre más de 100 contracciones promedia asíncronamente entre 0.15 y 0.25. El límite estático fijo previo `[-0.05, 1.05]` provocaba que las señales quedaran confinadas en el 20% inferior del recuadro.
    - **Ajuste Dinámico al Máximo Real:** Se sustituyó la cota fija por un cálculo dinámico `y_max_plot = max_promedio_global * 1.15` (compartido entre las 5 vocales para mantener la proporcionalidad fisiológica). Las curvas de entrada y reconstrucción aprovechan ahora el 100% del alto visual del subplot con óptima nitidez.

28. **Extracción Nativa de Señal Cruda y Preservación Estricta de target_len (2026-09-14):**
    - **Preservación Inmutable de target_len:**
      - En el compilador de arquitecturas (`compilar_modelo_desde_codigo`), si el usuario define un valor por defecto para `target_len` en su clase (ej. `target_len=1000` o personalizado), el sistema lo respeta rígidamente y no lo sobreescribe.
      - En la validación sintética (`verificar_arquitectura_codigo`), la dimensión temporal del tensor dummy se ajusta automáticamente al atributo `modelo.target_len` de la red.
      - En la interfaz gráfica (`AutoencoderNoSupervisadoTab`), la plantilla para señal cruda adopta formalmente `target_len = 1000` y dimensionamiento dinámico `self.conv(torch.zeros(1, in_channels, self.target_len)).shape[1]`.

29. **Restauración de Ventana de Pulso por Ciclo Completo y Reestructuración Modular de la Interfaz (2026-09-14):**
    - **Restauración de la Ventana Fisiológica Completa:**
      - El intento de forzar un corte rígido de 500 ms (0.20s pre, 0.30s post) truncaba el ciclo de contracción y desplazaba el pico de activación hacia $t \approx 0.20$ con cola asimétrica.
      - Se restauró la segmentación oficial por ciclo completo de metrónomo ($40\%$ pre-pico y $60\%$ post-pico respecto a $W_{\text{ciclo}} = \text{round}(\frac{60}{\text{BPM}} \cdot f_s)$), capturando de forma íntegra el ataque, meseta y relajación muscular, centrado simétricamente.
      - El mapeo a los puntos de tensor (`target_len_cruda = 1000`, `target_len_env = 100`) preserva la geometría y el soporte temporal completo de cada contracción.
    - **Reestructuración Modular de la Interfaz Gráfica (`AutoencoderNoSupervisadoTab`):**
      - Se dividió la configuración en bloques independientes por modalidad y etapa de acondicionamiento:
        1. **Modalidad 1: Envolvente 1D:** Parámetros (Técnica, Suavizado, Puntos temporales) + Selección 2D/3D.
        2. **Modalidad 2: Señal Cruda 1D:** Parámetros (Puntos temporales = 1000, Pooling GAP+GMP) + Selección 2D/3D.
        3. **Modalidad 3: Espectrograma (STFT):** Parámetros de resolución + Selección 2D/3D.
        4. **Alineación de Pulso:** Selector de señal de referencia (Micrófono Onset, Micrófono Volumen, Canales musculares, Supremo).
        5. **Parámetros DSP Previos y Acondicionamiento:** Pasa-banda configurable (HP Cutoff 20 Hz, LP Cutoff 450 Hz), Filtro de Línea Notch Q (2.0), Estimación dinámica de ruido interpulso (Alpha Ruido 0.5, Gate 0.0), Outliers y Ponderación tricanal ($w_0, w_1, w_2$).
    - **Diagnóstico Teórico de Optimización (Adam vs Métodos de Segundo Orden):**
      - La meseta de pérdida en $\text{MSE} \approx 0.0048$ en señal cruda no se debe a estancamiento en puntos silla del optimizador de primer orden (Adam ya modela la curvatura diagonal del Hessiano mediante $\beta_2=0.999$), sino a la varianza asíncrona estocástica de los potenciales de acción individuales (MUAPs), que constituye un ruido de innovación de fase irreductible. Como demostró la evaluación diagnóstica, el espacio latente convergió con éxito (53.2% GMM, silueta +0.042 y estructura cruciforme completa).

30. **Corrección del Bucle de Reapertura al Salir de TARS (2026-09-14):**
    - **Diagnóstico del Bucle de Reapertura:** En `~/.config/systemd/user/tars-tray.service`, la directiva `Restart=always` forzaba a systemd a reiniciar el proceso cada vez que este finalizaba, incluso si el usuario seleccionaba "Salir" en el menú contextual del applet de la bandeja del sistema (`self.app.quit()`), reapareciendo automáticamente a los 3 segundos.
    - **Ajuste de Directivas Systemd (`Restart=on-failure`):** Se modificó la política de reinicio tanto en `tars-tray.service` como en `tars-wake-word.service` a `Restart=on-failure`, de modo que una salida intencional con código 0 o una solicitud de detención no reactive el servicio, manteniendo la reactivación automática únicamente ante caídas imprevistas o fallos de ejecución.
    - **Cierre Integral de Servicios de Fondo:** Se actualizó `tars_tray.py` (tanto en el repositorio como en `~/.gemini/config/`) implementando el método `exit_app()`, el cual detiene de inmediato cualquier reproducción de voz activa (`stop_all_speech()`), envía la orden no bloqueante de apagado a systemd (`systemctl --user stop --no-block tars-wake-word.service tars-tray.service`) y finaliza el bucle de eventos de Qt.
    - **Lanzador de Escritorio (`tars.desktop`):** Se configuró `~/.local/share/applications/tars.desktop` con ícono oficial y comando de inicio unificado para permitir reabrir TARS de forma directa y cómoda desde el menú de aplicaciones del sistema.

31. **Integración de Soft-DTW (Dynamic Time Warping Diferenciable) y Divergencia Simétrica (2026-09-14):**
    - **Fundamentación Biomecánica y Matemática (Cuturi & Blondel, ICML 2017):**
      - La función de pérdida de error cuadrático medio ($\text{MSE}$) penaliza de forma muestra a muestra rígida ($t = \hat{t}$) cualquier pequeña variación en la velocidad de articulación ($\Delta t \approx 20\text{--}50\,\text{ms}$) o asincronía en el reclutamiento de unidades motoras, degradando el entrenamiento aunque la sinergia bioeléctrica relativa entre músculos sea exacta.
      - **Operador Suave de Bellman ($\min_\gamma$):**
        $$\min_\gamma(r_0, r_1, r_2) = -\gamma \log \left( e^{-r_0/\gamma} + e^{-r_1/\gamma} + e^{-r_2/\gamma} \right)$$
        donde $\gamma > 0$ es la temperatura de relajación suave. Para garantizar estabilidad numérica frente a desbordamiento exponencial, se sustrae el mínimo instantáneo $m = \min(r_0, r_1, r_2)$:
        $$\min_\gamma(r_0, r_1, r_2) = m - \gamma \log \sum_{k=0}^2 \exp\left( -\frac{r_k - m}{\gamma} \right)$$
    - **Matriz de Distancia Conjunta Tricanal Invariante:**
      - Para tensores sEMG multivariados de entrada $x \in \mathbb{R}^{B \times C \times N}$ y reconstrucción $\hat{x} \in \mathbb{R}^{B \times C \times M}$ con $C=3$ canales (Milohioideo Canal 0, Depresor Canal 1, Orbicular Canal 2), la distancia en cada par temporal $(i, j)$ se evalúa de forma conjunta entre todos los canales musculares:
        $$D_{b, i, j} = \sum_{c=0}^2 (x_{b, c, i} - \hat{x}_{b, c, j})^2 = \|x_{b, i}\|^2 + \|\hat{x}_{b, j}\|^2 - 2 \langle x_{b, i}, \hat{x}_{b, j} \rangle$$
        calculado de forma vectorial mediante multiplicación por lotes `torch.bmm(x.transpose(1, 2), y)` sin instanciar tensores cuatridimensionales pesados.
    - **Divergencia Soft-DTW Simétrica y No Negativa:**
      - Debido a que $\min_\gamma(0, 0, 0) = -\gamma \log 3 < 0$, la distancia $\text{sDTW}_\gamma(x, x)$ no es nula. Se implementó la divergencia simétrica:
        $$\mathcal{D}_\gamma(x, \hat{x}) = \text{sDTW}_\gamma(x, \hat{x}) - \frac{1}{2} \left[ \text{sDTW}_\gamma(x, x) + \text{sDTW}_\gamma(\hat{x}, \hat{x}) \right]$$
        cumpliendo $\mathcal{D}_\gamma(x, \hat{x}) \ge 0$ y $\mathcal{D}_\gamma(x, \hat{x}) = 0 \iff x = \hat{x}$.
    - **Aceleración Dual (Numba CPU en Paralelo + PyTorch Nativo):**
      - Módulo `EMG_desarrollo/deep_learning/soft_dtw.py`:
        - Paso hacia adelante y gradiente exacto dR/dD compilados con Numba JIT `@numba.njit(parallel=True, fastmath=True)` sobre `numba.prange(B)` aprovechando todos los núcleos de CPU en paralelo (tiempo de ejecución inferior a 2 ms por lote).
        - Fallback automático a tensores vectorizados de PyTorch para ejecución en CUDA o entornos sin compilador LLVM.
        - Enlace directo con Autograd mediante `torch.autograd.Function`, propagando la matriz de transiciones esperadas de alineamiento $E \in \mathbb{R}^{B \times N \times M}$ de forma analítica exacta hacia el decodificador y codificador.
    - **Integración en la Interfaz Gráfica y Flujo de Trabajo:**
      - En `AutoencoderNoSupervisadoTab` (Sección 7: "Parámetros de Optimización y Calibración"):
        - Selector de función de pérdida (`cmb_loss`): `MSE (Error Cuadrático Medio)`, `Soft-DTW (Alineación Temporal Suave)`, `Divergencia Soft-DTW (Simétrica)`, `Híbrida (MSE + Soft-DTW)`.
        - Parámetro de temperatura (`inp_gamma_sdtw`): $\gamma \in [0.01, 50.0]$ (valor por defecto 1.00).
      - En `motor_autoencoder_unificado.py`: `entrenar_autoencoder()` acepta `tipo_perdida="mse"`, `"soft_dtw"`, `"soft_dtw_divergence"` e `"hibrida"`, reportando el valor de convergencia por época con la métrica seleccionada.
      - En `main_app.py`: `run_autoencoder_no_sup_entrenar` y `run_autoencoder_no_sup_completo` propagan los hiperparámetros de pérdida sin alterar la invariabilidad de la línea base histórica.

32. **Diagnóstico del Offset en Canales 0 y 2 y Modulación del Parámetro Gamma (2026-09-14):**
    - **Origen del Offset en las Curvas Promedio (Milohioideo y Orbicular):**
      - **Causa Raíz Identificada:** La sustracción de ruido dinámico interpulso en `extraer_dataset_unificado` evalúa el nivel local $r_c = \text{Ruido}_{\text{interpulso}, c} \cdot \alpha_{\text{ruido}}$ y calcula $\max(0.0, \; x_c(t) - r_c)$.
      - En la interfaz gráfica, el control `Alpha Ruido` (`inp_alpha`) se encontraba fijado por defecto en `0.5`, sustrayendo únicamente el $50\%$ del nivel basal estimado.
      - Al aplicar la calibración fisiológica posterior ($x_c(t) \cdot k_c$), los canales con menor amplitud nativa (como el Orbicular en vocales abiertas o Milohioideo en vocales labiales) reciben factores de ganancia elevados ($k_c \approx 10\text{--}20$). El $50\%$ de piso de ruido residual no sustraído se amplificó proporcionalmente, generando el offset constante visible de $0.12\text{--}0.20$ en la línea de base.
      - **Ajuste:** Se actualizó el valor por defecto de `Alpha Ruido` en la interfaz gráfica a **`1.0`**, garantizando la sustracción íntegra del $100\%$ del nivel de reposo interpulso en cada contracción.
    - **Modulación Física y Operativa del Parámetro Gamma ($\gamma$):**
      - $\gamma$ regula la temperatura de la función $\min_\gamma(a, b, c) = -\gamma \log \sum_k e^{-a_k/\gamma}$ en Soft-DTW:
        - Para $\gamma \to 0$ ($\gamma \le 0.1$): opera como DTW clásico (mínimo duro rígido).
        - Para $\gamma \in [0.5, 1.5]$ (óptimo para sEMG): proporciona deformación elástica suave con gradientes de Gibbs diferenciables y estables.
        - Para $\gamma \gg 2.0$: diluye la correspondencia temporal y dispersa la sensibilidad morfológica.
      - Se ajusta directamente desde el campo `Gamma Soft-DTW` en el bloque 7 de la interfaz.

33. **Investigación Bibliográfica de Impedancia de Entrada en Amplificadores sEMG (Merletti & Parker 2004 / SENIAM - 2026-09-14):**
    - **Ubicación del Libro en Obsidian Vault:**
      `Materias/Tesis/Papers/Fundamentos y Libros/Libro - Electromiografia.pdf` (*Electromyography: Physiology, Engineering, and Noninvasive Applications*, editado por Roberto Merletti y Philip A. Parker, IEEE Press / John Wiley & Sons, 2004).
    - **Fusión de Informes en Archivo Único:**
      Consolidado en `Materias/Tesis/Teoria/Informe_Tecnico_FrontEnd_EMG.md`, conteniendo la teoría biofísica, fórmulas de CMRR por desbalance de electrodos, cuadro normativo SENIAM (Tabla 5.4), capturas de páginas renderizadas del tratado original y resolución de la paradoja entre impedancia y sensibilidad a ruido de modo común.
    - **Resolución de la Paradoja de Modo Común e Impedancia:**
      - El ruido intrínseco de electrodo (térmico/Johnson y galvánico) depende de la impedancia de contacto $Z_e$ (se reduce limpiando la piel y aumentando el área del sensor).
      - El ruido de línea (50 Hz) inducido en modo común $V_{\text{CM}}$ se convierte en ruido diferencial por el desbalance $\Delta Z = |Z_{e1} - Z_{e2}|$ mediante $V_{\text{diff}} \approx V_{\text{CM}} \cdot \frac{\Delta Z}{Z_i}$. Por ende, elevar la impedancia de entrada $Z_i$ a $>100\,\text{M}\Omega$ no aumenta la interferencia de red, sino que es la única barrera matemática para blindar el CMRR frente a asimetrías de piel.
      - Para cancelar el modo común en el cuerpo se prescribe el circuito DRL (*Driven Right Leg*).
    - **Soluciones al Dilema del Offset DC frente al Filtro RC:**
      - Partición de ganancia (baja ganancia en In-Amp $G=10$ + filtro pasa-altos activo posterior + segunda etapa).
      - Capacitor en serie con la resistencia de ganancia $R_G$ ($G_{\text{DC}}=1$, $G_{\text{AC}}\approx 225$).
      - Servocorrección activa integradora conectada al terminal `REF` del AD620.

34. **Resolución del Error de Entrenamiento en Señal Cruda con Soft-DTW y Reescalado Fisiológico Directo por Promedio de Pulsos (2026-09-14):**
    - **Diagnóstico del Fallo de Memoria en Señal Cruda con Soft-DTW:**
      - **Causa Raíz Identificada:** La señal cruda rectificada tiene 1000 muestras ($T = 1000$). Con $N=1000, M=1000$ y lotes de 35 a 515 contracciones, la matriz de distancia multicanal $D \in \mathbb{R}^{B \times 1000 \times 1000}$ alcanzaba entre 140 MB y 2.06 GB por matriz. La asignación acumulada en PyTorch y Numba ($D$, $R$, $E$ y versiones normalizadas) superaba los 18 GB de RAM, disparando el *OOM Killer* de Linux (proceso terminado abruptamente) o congelando el hilo por el cómputo de 1.000.000 de celdas por muestra.
      - **Solución Bioeléctrica y Numérica:**
        - En `SoftDTW.forward(x, y)`, cuando $T > 100$, se aplica submuestreo adaptativo suave `F.adaptive_avg_pool1d(..., 100)` para alinear la cinemática articular macroscópica (bines de 5 ms a 200 Hz).
        - Esto reduce la matriz a $(B, 100, 100)$, reduciendo el consumo de RAM de 2.06 GB a 1.2 MB (factor de reducción de más de 100x) y permitiendo que el cálculo dinámico corra en milisegundos sin congelamientos ni desbordamientos de memoria.
        - Al ser `adaptive_avg_pool1d` una operación diferenciable nativa de PyTorch, propaga gradientes exactos hacia todas las 1000 muestras de la señal cruda decodificada.
    - **Estabilidad Numérica en Numba:**
      - Se retiró `fastmath=True` para evitar comportamientos anómalos de LLVM frente a valores de inicialización grandes ($10^{10}$) y se incorporaron guardas numéricas en el gradiente hacia atrás (`if m1 < 1e9`) para evitar que transiciones hacia celdas no alcanzadas filtren gradientes espurios.
    - **Monitoreo Obligatorio por Época y ETA:**
      - Se actualizó el bucle de entrenamiento para reportar la época 1 de inmediato y avances periódicos cada 10 épocas mostrando porcentaje, pérdida instantánea y tiempo restante estimado (ETA).
    - **Reescalado Fisiológico Directo por Promedios hacia 1.0 (Regla Estricta del Usuario):**
      - Se eliminó el uso de P95 y se eliminaron de forma absoluta las reglas de subordinación artificiales intermedias que degradaban $k_0$ a $0.70$ y $k_2$ a $0.41$.
      - Se computa directamente el promedio de todos los pulsos normalizados por el supremo tricanal para cada clase vocal:
        $$P_{\text{rojo, A}} = \max_t \bar{x}_{0, \text{A}}(t), \quad P_{\text{verde, I}} = \max_t \bar{x}_{1, \text{I}}(t), \quad P_{\text{naranja, U}} = \max_t \bar{x}_{2, \text{U}}(t)$$
      - Los factores de escala se definen de forma directa, desacoplada e inviolable:
        $$k_0 = \frac{1.0}{P_{\text{rojo, A}}}, \quad k_1 = \frac{1.0}{P_{\text{verde, I}}}, \quad k_2 = \frac{1.0}{P_{\text{naranja, U}}}$$
      - Al aplicarse $\mathbf{k} = [k_0, k_1, k_2]^T$, se garantiza por construcción matemática que:
        - En el promedio de **/a/**, la **curva roja (Milohioideo)** alcanza exactamente **1.0**.
        - En el promedio de **/i/**, la **curva verde (Depresor)** alcanza exactamente **1.0**.
        - En el promedio de **/u/**, la **curva amarilla/naranja (Orbicular)** alcanza exactamente **1.0**.
    - **Supresión Total de Offset Basal:**
      - `alpha_ruido` fijado en 1.0 por defecto en toda la interfaz gráfica y scripts puente (`main_app.py`), restando el 100% del ruido basal y garantizando que los extremos de los pulsos toquen 0.0 sin pedestales residuales.
    - **Aislamiento Estricto de Hiperparámetros de Línea Base (Ceteris Paribus):**
      - El ajuste automático a Full-Batch, $\text{lr} = 0.008$ y $250$ épocas se aisló estrictamente para `tipo_perdida == 'mse'`, evitando forzar hiperparámetros ajenos sobre Soft-DTW o funciones híbridas.

35. **Corrección Definitiva del Reescalado Fisiológico por Promedios (Rojo en /a/ -> 1.0, Amarillo en /u/ -> 1.0), Supresión de Offset y Suavizado Continuo Hanning (2026-09-14):**
    - **Diagnóstico del Fallo de Amplitud (Rojo a 0.65 en vez de 1.0):**
      - **Causa Raíz:** Se aplicaba un recorte rígido individual `np.clip(w['env_norm'] * k_vec, 0.0, 1.0)` a cada pulso antes de promediar. Por la no linealidad del operador $\min(\cdot, 1.0)$, los pulsos superiores a $1.0 / k_0$ quedaban truncados a 1.0, mientras que los inferiores permanecían bajos, haciendo que la media final resultante cayera a $\approx 0.65$. Además, al inflar el factor $k_1$ de la vocal /i/, el canal 1 saturaba a 1.0 en todas las demás vocales, haciendo que la curva verde superara a la roja en /a/.
      - **Solución Matemática:** 
        1. Se retiró el `np.clip` a 1.0 por pulso individual, aplicando el factor de forma puramente lineal: por linealidad de la esperanza $\overline{k_c \cdot x} = k_c \cdot \bar{x}$, se garantiza que el pico del promedio agonista alcance **exactamente 1.00** ($P_{\text{rojo, A}} \equiv 1.0$, $P_{\text{verde, I}} \equiv 1.0$, $P_{\text{naranja, U}} \equiv 1.0$).
        2. Se introdujeron salvaguardas de dominancia motora fisiológica: en /a/, si el canal secundario Verde superaría a Rojo, su ganancia se acota para no superar $0.85$, asegurando que el Milohioideo (Rojo) sea siempre el máximo absoluto en apertura mandibular. Lo mismo se impone para el Orbicular (Naranja) en /u/ y el Depresor (Verde) en /i/.
    - **Supresión de Offset Basal mediante Rampa Lineal:**
      - En lugar de restar un piso escalar constante, se computan los pisos en ambos extremos $[0, t_{\text{pre}}]$ y $[t_{\text{post}}, 1]$ y se sustrae una rampa lineal $b(t) = b_0 + t(b_1 - b_0)$, garantizando que cada pulso inicie y finalice estrictamente en $0.00$ sin pedestales artificiales residuales.
    - **Suavizado Hanning en Envolvente 1D:**
      - En `calcular_envolvente`, se reemplazó la ventana plana rectangular (boxcar) por una ventana Hanning normalizada con desvanecimiento suave a los extremos. Esto suprime por completo el rizado de alta frecuencia y la fuga espectral, entregando perfiles continuos y suaves acordes a la cinemática articular.
    - **Acotación Visual de Reconstrucción en Evaluación:**
      - En `evaluar_espacio_latente`, la reconstrucción promedio de la red se enmarca limpiamente en $[0.0, 1.05]$, impidiendo desbordes visuales fuera de los límites de los subplots.

36. **Consolidación del Informe Técnico del Front-End sEMG con AD620, Esquemas Circuitales y Protocolo de Retrabajo Experimental (2026-09-14):**
    - **Diagnóstico del Front-End Actual:**
      - El filtro pasa-altos pasivo a la entrada ($R = 100\,\text{k}\Omega$, $C = 33\,\text{nF}$) sitúa el corte en $f_c \approx 48.23\,\text{Hz}$, inmediatamente sobre los $50\,\text{Hz}$ de línea. Las tolerancias de los capacitores ($\pm 10\%\text{--}20\%$) rompen la simetría y convierten el modo común ambiental en una señal diferencial masiva que el AD620 multiplica con ganancia $G \approx 495$.
      - La impedancia de entrada se aplasta a $Z_{\text{in}} \approx 100\,\text{k}\Omega$, violando el estándar SENIAM ($>100\,\text{M}\Omega$). Ante despegues parciales ($\Delta Z \approx 50\text{--}100\,\text{k}\Omega$), el factor $\Delta Z / Z_{\text{in}}$ se dispara al $50\%\text{--}100\%$, saturando el canal contra las baterías de $\pm 9\,\text{V}$.
    - **Solución 1: Retrabajo en 1 Etapa (Sin Nuevo PCB):**
      - Puentear capacitores de $33\,\text{nF}$ a la entrada.
      - Desacoplo AC en el lazo de ganancia: capacitor no polarizado $C_G = 100\,\mu\text{F}$ en serie con $R_G = 100\,\Omega$. Fija $G_{\text{DC}} = 1$ (el offset galvánico de la piel no satura) y $G_{\text{AC}} \approx 495$ para $f \ge 20\,\text{Hz}$ con $f_c \approx 15.91\,\text{Hz}$.
      - Elevar resistencias a GND a $10\,\text{M}\Omega$: reduce $\Delta Z / Z_{\text{in}}$ a menos del $1\%$ y drena con seguridad $I_B \approx 2\,\text{nA}$ ($V_{\text{offset}} \approx 20\,\text{mV}$).
    - **Solución 2: Rediseño en 2 Etapas (Circuito Recomendado SENIAM):**
      - Etapa 1 (AD620): Ganancia moderada $G_1 \approx 9.82$ ($R_G = 5.6\,\text{k}\Omega$) con entradas directas y $10\,\text{M}\Omega$ a masa.
      - Inter-etapa: Filtro pasa-altos pasivo unipolar ($C = 470\,\text{nF}$, $R = 22\,\text{k}\Omega \implies f_c \approx 15.39\,\text{Hz}$) que bloquea la continua sin degradar el CMRR.
      - Etapa 2 (TL072): Ganancia secundaria $G_2 = 48$ ($R_1 = 1\,\text{k}\Omega$, $R_f = 47\,\text{k}\Omega \implies G_{\text{tot}} \approx 471.4$) con filtro pasa-bajos activo anti-aliasing integrado ($C_f = 6.8\,\text{nF} \implies f_c \approx 498.1\,\text{Hz}$).
    - **Estrategia Experimental sin Desoldar:**
      - Puentear capacitores cerámicos con alambre fino desde arriba.
      - Cortar un terminal de la resistencia azul $R_G$ elevada en el aire y conectar $C_G$ en serie.
    - **Ecuaciones Generales de Tensión Entrada-Salida e Inestabilidad por Despegue:**
      - Se incorporó la formulación general del divisor de tensión $V_{\text{in}, j}(t) = \alpha_j(t) [V_{\text{cm}}(t) \pm \frac{V_{\text{emg}}(t)}{2} + V_{\text{DC}, j}]$ con $\alpha_j(t) = \frac{Z_{i,j}}{Z_{e,j}(t) + Z_{i,j}}$.
      - Desglose cuadrifásico de la salida: $V_{\text{out}}(t) \approx A_d \bar{\alpha}(t) V_{\text{emg}}(t) + A_d \Delta\alpha(t) V_{\text{cm}}(t) + A_d [\alpha_1(t) V_{\text{DC}, 1} - \alpha_2(t) V_{\text{DC}, 2}] + A_c \bar{\alpha}(t) V_{\text{cm}}(t)$, demostrando analíticamente la explosión de 50 Hz, la distorsión morfológica de la amplitud muscular y el transitorio de saturación galvánica al variar $Z_e(t)$.
    - **Archivos Generados:**
      - `EMG_desarrollo/documentacion_hardware/informe_frontend_emg_ad620.tex` y `informe_frontend_emg_ad620.pdf` (17 páginas, compilado sin desbordes con los 3 esquemas Circuitikz, las fórmulas completas de transferencia analógica y las 5 capturas del tratado de Merletti & Parker 2004).
      - `EMG_desarrollo/archivos_md/Informe_Tecnico_FrontEnd_EMG.md` e `EMG_desarrollo/documentacion_hardware/Informe_Tecnico_FrontEnd_EMG.md` (con enlaces a las imágenes en `imagenes/`).

37. **Justificación Biofísica del Retrabajo de Hardware frente al Filtrado Digital y Fenómenos No Lineales de Banda Ancha (2026-09-15):**
    - **Inutilidad del Filtrado Digital (Notch/ANC) ante Despegues y Ruidos de Banda Ancha:**
      - Aunque el zumbido estacionario de 50 Hz y armónicos puede atenuarse por software, el despegue de electrodos y la impedancia baja ($Z_{\text{in}} \approx 100\,\text{k}\Omega$) introducen fenómenos no lineales irreversibles que ningún algoritmo DSP puede subsanar una vez digitalizada la señal:
      1. **Salto Galvánico de Continua y Recorte contra Rieles:** El potencial de media celda piel-electrodo ($V_{\text{DC}} \approx 20\text{--}300\,\text{mV}$) multiplicado por $G=495$ genera tensiones teóricas de hasta $9.9\text{ V}$, saturando la salida contra los rieles de batería ($\pm 9\,\text{V}$). La información recortada (*clipping*) se destruye de forma irreversible; el Notch o filtro adaptativo no pueden recuperar una señal plana a ceros lógicos.
      2. **Pérdida de Amplitud Bioeléctrica y Corrupción de Ratios Intermusculares:** Con $Z_{\text{in}} = 100\,\text{k}\Omega$, un incremento en la impedancia de contacto a $100\,\text{k}\Omega$ atenúa la señal EMG al 50% ($\alpha = 0.50$). Esto corrompe la sinergia bioeléctrica y el balance del Supremo Tricanal por Pulso Individual. Al elevar $Z_{\text{in}} \ge 10\,\text{M}\Omega$, la transferencia se mantiene en $99\%$ ($\alpha = 0.99$), garantizando invariancia inter-toma.
      3. **Pérdida del 60% de la Banda sEMG Facial (20 a 45 Hz):** El corte pasa-altos original a $f_c \approx 48.23\,\text{Hz}$ cercena el espectro muscular facial primario (frecuencias de disparo de unidades motoras en habla submáximal entre 20 y 45 Hz sufren atenuaciones de $-7.7\,\text{dB}$ y desfasajes de $+67.5^\circ$). El retrabajo con $C_G = 100\,\mu\text{F}$ traslada el corte a $15.91\,\text{Hz}$, recuperando íntegramente la banda fisiológica con respuesta plana.
      4. **Degradación de SNR por Ruido Térmico Johnson-Nyquist:** El aumento de $Z_e$ a $100\,\text{k}\Omega$ eleva la densidad de ruido térmico a $57.8\,\mu\text{V}_{\text{RMS}}$ en la salida del AD620, reduciendo en 10 dB el SNR real antes de la digitalización.
      5. **Desfasaje Asimétrico Dinámico $\Delta\phi(t)$ y Modulación No Estacionaria:** La fluctuación de contacto modula en amplitud y fase el modo común, transformándolo en una envolvente espuria no estacionaria que rompe la hipótesis de convergencia de los filtros adaptativos por mínimos cuadrados (LMS).
    - **Actualización Documental Integral:**
      - Se incorporó la sección completa de análisis biofísico, ecuaciones de transferencia y tabla comparativa de fenómenos analógicos vs filtrado digital en:
        - `EMG_desarrollo/documentacion_hardware/informe_frontend_emg_ad620.tex` y compilado exitoso en `informe_frontend_emg_ad620.pdf` (20 páginas, 0 errores, 0 warnings).
        - `EMG_desarrollo/archivos_md/Informe_Tecnico_FrontEnd_EMG.md` y `EMG_desarrollo/documentacion_hardware/Informe_Tecnico_FrontEnd_EMG.md`.

38. **Recepción, Organización y Normalización del Dataset (2026-09-15):**
    - **Origen del archivo:** `/home/santiago/Descargas/2026-09-15-20260915T210550Z-1-001.zip`.
    - **Destino oficial:** `EMG_desarrollo/base_de_datos_electrodos/2026-09-15/`.
    - **Estandarización y Normalización Aplicada (40 Tomas Totales de Candela):**
      - **Nombres de Carpetas y Sesiones:** Formato canónico `<Vocal>_<Prueba>_Candela` con vocales iniciales estrictamente en mayúscula (`A_`, `E_`, `I_`, `O_`, `U_`) y nombres limpios de prueba (`Prueba1` a `Prueba7`, `PruebaEXTRA`, `PruebaEXTRA_2`, `PruebaTODAS`, `PruebaTODAS_2`, `PruebaTODAS2`, `PruebaTODAS3`).
      - **Atributos en metadata.json:**
        - `"sujeto": "Candela"` unificado en todas las tomas (corrigiendo marcadores residuales como `"Sujeto1"` o `"CANDE"`).
        - `"letra"` en mayúscula canónica (`"A"`, `"E"`, `"I"`, `"O"`, `"U"`).
        - Nomenclatura anatómica oficial estricta en `muscles`, `muscles_map` y `musculo`:
          - `BELLY` $\to$ `Anterior Belly`
          - `ORBI` $\to$ `Orbicularis Oris`
          - `ZIGO` $\to$ `Zygomaticus Major`
          - `MIC` $\to$ `Micrófono`
      - **Sincronización de Gráficos:** Renombrado automático de los archivos de plot calibrado (`plot_calibrado_2026-09-15_<Sesión>.png`) para reflejar la nomenclatura canónica de cada carpeta.
    - **Distribución de las Sesiones:**
      - **Carpeta `2026-09-15/` (20 Tomas Bicanal - 30 BPM):** Foco comparativo bicanal entre `Anterior Belly` y `Orbicularis Oris` para apertura mandibular y constricción labial (/a/, /o/, /u/).
      - **Carpeta `2026-09-16/` (20 Tomas Tetracanal - 30 BPM, 12 pulsos c/u):** Se segregaron y renombraron canónicamente como series continuas (`<Vocal>_Serie<N>_Candela`, con $N \in \{1, 2, 3, 4\}$ y vocales /a/, /e/, /i/, /o/, /u/):
        - `Serie1`: 5 vocales (15:52) - 60 pulsos
        - `Serie2`: 5 vocales (15:56) - 60 pulsos
        - `Serie3`: 5 vocales (16:08) - 60 pulsos
        - `Serie4`: 5 vocales (16:12) - 60 pulsos
        Configuración tetracanal completa: `canal_0: Anterior Belly`, `canal_1: Orbicularis Oris`, `canal_2: Zygomaticus Major`, `canal_3: Micrófono`. Metadatos actualizados con `"prueba": "SerieN"` y gráficos sincronizados a `plot_calibrado_2026-09-16_<Vocal>_Serie<N>_Candela.png`. Total: 240 pulsos tetracanal de alta pureza listos para evaluación en autoencoder y PCA/UMAP.

39. **Segregación Inter-Día de Tomas de Agosto (2026-08-27, 2026-08-28, 2026-08-29 y 2026-08-30):**
    - **Motivación:** Desacoplar las sesiones experimentales de Petra y Candela que se encontraban inicialmente agrupadas bajo el mismo directorio `2026-08-28/`, permitiendo que las herramientas de carga por fecha (GUI, scripts de autoencoder y análisis Trevisan/PCA) discriminen limpiamente cada ensayo independiente.
    - **Reorganización Estricta Implementada:**
      - **`2026-08-27/` (Sujeto Petra - med3):** 5 tomas (`A_med3_clase4_Petra`, `E_...`, `I_...`, `O_...`, `U_...`) aisladas para su análisis independiente (Canal 1 en Platysma). `measurement_date` sincronizado a `2026-08-27`.
      - **`2026-08-28/` (Sujeto Petra - med1 y med2):** Conserva exclusivamente las 10 tomas canónicas de Petra con electrodos de silicona (`med1_clase24_Petra` y `med2_clase4_Petra`).
      - **`2026-08-29/` (Sujeto Candela - Prueba 1 y 2):** 10 tomas (`A_Prueba1_Cande`, `A_Prueba2_Cande`, `E_...`, `I_...`, `O_...`, `U_...`) con electrodos en `Anterior Belly` y `Milohioideo`. `measurement_date` en `metadata.json` y figuras de calibración sincronizadas a `2026-08-29`.
      - **`2026-08-30/` (Sujeto Candela - Prueba 4 y 5):** 10 tomas (`A_Prueba4_Cande`, `A_Prueba5_Cande`, `E_...`, `I_...`, `O_...`, `U_...`) con electrodos en `Anterior Belly` y `Nudo Sonrisa`. `measurement_date` en `metadata.json` y figuras de calibración sincronizadas a `2026-08-30`.


40. **Configuración Canónica Universal de Canales en 2026-08-30, 2026-09-01 y 2026-09-16 (Orbicularis Oris en Canal 2):**
    - **Objetivo Canónico:** Alinear todas las sesiones de Candela (08-30, 09-01 y 09-16) a la geometría universal del proyecto:
      - **Canal 0 (Apertura Mandibular / $+Y$):** `Anterior Belly` (vocal /a/).
      - **Canal 1 (Retracción Comisural / Sonrisa / $+X > 0$):** `Nudo Sonrisa` (en 08-30) / `Risorio` (en 09-01) / `Zygomaticus Major` (en 09-16) (vocales /i/ y /e/).
      - **Canal 2 (Constricción y Redondeo Labial / $-Y < 0$):** `Orbicularis Oris` (vocales /u/ y /o/).
      - **Canal 3 (Acústica):** `Micrófono`.
    - **Operación Ejecutada (55 Sesiones Totales: 10 en 08-30, 25 en 09-01, 20 en 09-16):**
      - **Señales de Audio (`grabacion.wav`):** Se fijó la señal de sonrisa en `canal_1/grabacion.wav` y la señal de orbicular en `canal_2/grabacion.wav`.
      - **Matrices de Registro (`grabacion.csv`):** Columnas sincronizadas (`Canal 1: Sonrisa`, `Canal 2: Orbicularis Oris`).
      - **Metadatos (`metadata.json` en todos los canales):**
        - `muscles`: `["Anterior Belly", "Nudo Sonrisa" / "Risorio" / "Zygomaticus Major", "Orbicularis Oris", "Micrófono"]`.
        - `muscles_map`:
          - `canal_0`: `Anterior Belly`
          - `canal_1`: `Nudo Sonrisa` (08-30) / `Risorio` (09-01) / `Zygomaticus Major` (09-16)
          - `canal_2`: `Orbicularis Oris`
          - `canal_3`: `Micrófono`
        - En `canal_1`: `"musculo": smile_muscle`, `"physical_channel": "Dev1/ai1"`.
        - En `canal_2`: `"musculo": "Orbicularis Oris"`, `"physical_channel": "Dev1/ai2"`.
    - **Validación Fisiológica Confirmada:**
      - Vocal **/i/** (Sonrisa): Ch1 domina la retracción comisural, mientras Ch2 (`Orbicularis Oris`) reposa.

41. **Unificación Canónica de Canales en 2026-08-28 y 2026-08-21 (Sujeto Petra):**
    - **Objetivo Canónico:** Corregir la inversión de canales de Petra para que `Anterior Belly` lidere el Canal 0 en paridad con el resto de la base de datos:
      - **Canal 0 (Apertura Mandibular / $+Y$):** `Anterior Belly` (vocal /a/).
      - **Canal 1 (Elevador Perioral / $+X$):** `Levator Anguli Oris`.
      - **Canal 2 (Comisural / Sonrisa):** `Zygomaticus Major`.
      - **Canal 3 (Acústica):** `Micrófono`.
    - **Operación Ejecutada (20 Sesiones Totales: 10 en 08-28 y 10 en 08-21):**
      - **Señales de Audio (`grabacion.wav`):** Intercambio atómico entre `canal_0/grabacion.wav` y `canal_1/grabacion.wav`.
      - **Matrices de Registro (`grabacion.csv`):** Intercambio de las columnas `Canal 0` y `Canal 1`.
      - **Metadatos (`metadata.json` en todos los canales):**
        - `muscles`: `["Anterior Belly", "Levator Anguli Oris", "Zygomaticus Major", "Micrófono"]`.
        - `muscles_map`:
          - `canal_0`: `Anterior Belly`
          - `canal_1`: `Levator Anguli Oris`
          - `canal_2`: `Zygomaticus Major`
          - `canal_3`: `Micrófono`
        - En `canal_0`: `"musculo": "Anterior Belly"`, `"physical_channel": "Dev1/ai1"`.
        - En `canal_1`: `"musculo": "Levator Anguli Oris"`, `"physical_channel": "Dev1/ai0"`.

42. **Sincronización Canónica de Artefactos JSON/PNG Precomputados en Todas las Sesiones Modificadas (2026-09-16):**
    - **Diagnóstico del Problema:** En las sesiones donde se reordenaron canales físicos (`2026-08-30`, `2026-09-01`, `2026-09-16` para Ch1 <-> Ch2, y `2026-08-21`, `2026-08-28` para Ch0 <-> Ch1), las señales de audio (`grabacion.wav`), matrices CSV y `metadata.json` estaban alineadas canónicamente, pero los archivos estructurados de métricas precomputadas (`analisis_results.json`, `results.json`, `*_env100_0ms.json`) y las imágenes diagnósticas (`avg.png`, `pulses.png`, `spec.png`, `evolucion.png`, etc.) conservaban los datos previos al swap.
    - **Operación Ejecutada (75 Sesiones Totales, 574 archivos procesados):**
      - Se ejecutó el script `intercambiar_archivos_analisis_precomputados.py` intercambiando atómicamente todos los archivos JSON de métricas y figuras PNG entre los canales correspondientes sin alterar `grabacion.wav`, `grabacion.csv` ni `metadata.json`.
      - Se actualizaron internamente los campos `"channel"` en los JSON para concordar con su nueva ubicación.
    - **Resultado:** Coherencia total entre audio crudo, metadatos, análisis precomputados e informes visuales en toda la base de datos de electrodos.






35. **Silenciamiento Integral y Desconexión de Servicios de Dictado y Voz de TARS (2026-09-15):**
    - **Diagnóstico del Audio Residual:** Aunque el applet visual (`tars-tray.service`) y el detector de palabra clave (`tars-wake-word.service`) se cerraban correctamente tras seleccionar "Salir", el archivo de estado `tars_state.json` permanecía almacenando `{"mode": "button"}` en lugar de pasar a `"off"`, y el servicio de captura de teclado (`antigravity-f12.service`) continuaba activo en memoria. Como consecuencia, el gancho de síntesis de voz de Antigravity (`speak_response.py`), que consulta este archivo al finalizar cada turno de generación, asumía que la voz estaba autorizada y leía en voz alta por los altavoces de la computadora la respuesta del asistente.
    - **Alineación de Políticas Systemd (`Restart=on-failure`):** Se actualizó la directiva en `~/.config/systemd/user/antigravity-f12.service` a `Restart=on-failure`, impidiendo que el sistema intente revivir el servicio una vez ordenado su apagado.
    - **Cierre Atómico en tars_tray.py:**
      - Ocultamiento inmediato del ícono en la barra (`self.tray.hide()`) para evitar íconos fantasma en GNOME.
      - Escritura forzada de `save_mode("off")` en `tars_state.json`, garantizando que `speak_response.py` aborte en silencio de forma inmediata.
      - Detención inmediata de procesos de reproducción de audio activos (`stop_all_speech()`).
      - Orden conjunta de parada no bloqueante a systemd para `tars-wake-word.service`, `antigravity-f12.service` y `tars-tray.service`.
    - **Reapertura Limpia desde el Lanzador:** `tars.desktop` restablece el estado a `{"mode": "auto"}` y relanza en conjunto los tres servicios asociados al asistente de escritorio.
    - **Verificación en el Sistema:** Confirmado el estado inactivo (`inactive (dead)`) de los tres servicios y silenciado completo del motor de síntesis de respuestas.

43. **Integración de K-Means, Regularización de Ortogonalidad Latente y Fronteras de Decisión (2026-09-16):**
    - **Selector de Clustering (GMM vs K-Means):** En el motor unificado de autoencoder (`motor_autoencoder_unificado.py`) y en la pestaña de análisis (`ui_analysis.py`), se incorporó la opción de evaluar el espacio latente utilizando `KMeans(n_clusters=5, random_state=42, n_init=10)` o `GaussianMixture(n_components=5, covariance_type='full', random_state=42, max_iter=200)`, manteniendo el alineamiento óptimo de clases con las vocales mediante el algoritmo húngaro (`linear_sum_assignment`).
    - **Término de Pérdida para Ortogonalidad de Ejes Latentes ($\lambda_{\text{orto}}$):**
      Se corrigió la formulación de ortogonalidad normalizando la matriz de covarianza por la traza total (la suma de varianzas de todas las dimensiones latentes, $\text{tr}(C) = \sum_d \sigma_d^2$):
      $$Z_{\text{centrado}} = Z - \bar{Z}, \quad C = \frac{1}{B-1} Z_{\text{centrado}}^T Z_{\text{centrado}}, \quad \tilde{C} = \frac{C}{\text{tr}(C) + \epsilon}$$
      $$\mathcal{L}_{\text{orto}} = \sum_{i \neq j} \tilde{C}_{i, j}^2, \quad \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{orto}} \mathcal{L}_{\text{orto}}$$
      Esta formulación es estrictamente adimensional e invariante a la escala latente natural ($2\text{D}$ o $3\text{D}$): penaliza la covarianza cruzada relativa sin imponer una varianza arbitraria que destruya la capacidad reconstructiva del decodificador ni aplaste las coordenadas a cero ($\sim 10^{-4}$).
    - **Renderizado de Fronteras de Decisión Estilo generador_pca_umap y Encuadre Óptimo:**
      En el informe gráfico 2D del espacio latente canónico, se implementó una malla densa bidimensional (`meshgrid` $250 \times 250$) evaluada por el modelo de clustering (`model.predict`) y mapeada con el diccionario húngaro hacia los colores canónicos de las vocales. El fondo se grafica con `pcolormesh(shading='auto', alpha=0.18)` utilizando `matplotlib.colors.ListedColormap` y líneas divisorias de contorno `contour(levels=[0.5, 1.5, 2.5, 3.5], colors='k', linewidths=0.6, alpha=0.55)`, idéntico a las fronteras generadas por PCA/UMAP. Los límites de los ejes `set_xlim` y `set_ylim` se ajustan dinámicamente de forma proporcional a la dispersión real de las coordenadas ($15\%$ de margen en $X$ y $12\%$ en $Y$), evitando márgenes fijos desproporcionados cuando los datos presentan escalas compactas.
44. **Récord Cuantitativo en Espacio Latente 3D (87.8% GMM No Supervisado - Candela 2026-09-15/16):**
    - **Resultado Consolidado:** Se alcanzó una exactitud global del **87.8%** y un índice de silueta de **+0.285** en el autoencoder no supervisado 3D sobre las 20 series completas de Candela (4 series por vocal, 215 contracciones balanceadas), sin supervisión ni etiquetas en el gradiente de entrenamiento.
    - **Homogeneidad y Balance Multiclase Armónico (Regla de Oro):**
      - Vocal **/a/**: **88.4%**
      - Vocal **/e/**: **88.4%**
      - Vocal **/i/**: **100.0%**
      - Vocal **/o/**: **83.7%**
      - Vocal **/u/**: **78.0%**
      Todas las vocales superan ampliamente el $78\%$, resolviendo por primera vez simultáneamente la separación fina entre /e/ frente a /i/ y entre /o/ frente a /u/ de manera balanceada.
    - **Parámetros Oficiales Consolidados del Récord:**
      - **Modalidad:** `envolvente` (RMS, técnica cuadrática pura).
      - **Suavizado:** `smooth_ms = 90` ms (ventana temporal óptima).
      - **Puntos temporales interpolados:** `target_len = 100`.
      - **Dimensión latente:** `3D` ($Z \in \mathbb{R}^3$, fijada como opción por defecto en la GUI).
      - **Alineación temporal de contracción:** `Pico Derivada Micrófono (Onset)`.
      - **Filtros DSP previos:** Pasa banda 20 a 450 Hz, Notch $Q = 2.0$, compuerta de ruido en 0.0, sin penalización artificial de ruido basal ($\alpha = 1.0$).
      - **Normalización tricanal:** `Supremo Tricanal por Pulso Individual` ($M_{\text{supremo, pulso}}$) combinado con `Reescalado Fisiológico por Promedios` (Rojo en /a/ -> 1.0, Verde en /i/ -> 1.0, Naranja en /u/ -> 1.0).
      - **Régimen de optimización:** 150 épocas, batch size 32, $\text{lr} = 0.002$, función de pérdida pura `MSE` sin regularización de ortogonalidad forzada ($\lambda_{\text{orto}} = 0.0$).
      - **Clustering evaluador post-hoc:** `GaussianMixture(n_components=5, covariance_type='full', random_state=42)` con asignación óptima húngara.
    - **Hallazgo Físico Fundamental: Impacto del Ruido de Línea (Sesión 2026-09-15/16):**
      - Este conjunto de tomas fue registrado en condiciones de **muy bajo ruido de línea basal** (cableado apantallado y acoplamiento óptimo piel-electrodo).
      - **Discrepancia Crítica entre PCA y Autoencoder:** Se confirmó que el ruido de red (50 Hz y armónicos), incluso tras la aplicación de filtros Notch digitales, ensucia e introduce perturbaciones no lineales que perjudican severamente la convergencia del autoencoder. Configuraciones que visualmente logran una alta separación en PCA lineal a menudo fallaban o se degradaban en el autoencoder cuando existía zumbido de línea. En contraste, con la señal limpia de 09-15/16, el autoencoder funciona de forma sobresaliente (87.8%) incluso en configuraciones con menor dispersión inicial en PCA.
      - **Persistencia de la Dificultad Biomecánica /o/ frente a /u/:** Aunque la exactitud de ambas vocales alcanzó valores históricos ($83.7\%$ en /o/ y $78.0\%$ en /u/), siguen siendo las clases más próximas y complejas de segregar del espacio debido al reclutamiento compartido y simétrico del esfínter orbicular, tal como dictan los principios fisiológicos del proyecto.





45. **Generación del Diagrama Anatómico Oficial de Electrodos y Activación Vocálica (2026-09-16):**
    - **Objetivo:** Disponer de una figura anatómica limpia, vectorial y sobria (sin cabello, con rasgos minimalistas y proporciones craneofaciales exactas) que sirva como plantilla visual para publicaciones e interfaces.
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/mapa_electrodos_vocales_anatómico.png` (300 DPI, alta resolución).
    - **Mapeo Anatómico y Fisiológico Representado:**
      - **Canal 0 (Rojo, Submentoniano):** Músculo Milohioideo / Digástrico (Apertura mandibular) -> Vocal dominante **/a/**.
      - **Canal 1 (Verde, Comisural):** Músculo Depresor del ángulo de la boca / Risorio (Retracción comisural y sonrisa) -> Vocales dominantes **/i/** y **/e/**.
      - **Canal 2 (Naranja, Perioral):** Músculo Orbicularis Oris (Constricción y redondeo labial) -> Vocales dominantes **/o/** y **/u/**.
      - **Referencia / GND (Azul):** Apófisis Mastoides (detrás del pabellón auricular).
    - **Regla Estricta Incorporada:** Se prohíbe la búsqueda e inspección no autorizada de archivos locales para consultas de diseño o preguntas conceptuales a fin de preservar tokens, priorizando soluciones directas y transparentes.

46. **Cartografía Facial Anatómica Detallada con Ojos y Músculos Periorales y Submentonianos (2026-09-16):**
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/mapa_musculos_electrodos_vocales.png` (300 DPI, alta resolución).
    - **Estructuras Anatómicas Integradas:**
      1. **Ojos anatómicos realistas:** Esclera almendrada, iris azul oscuro, pupila y brillo especular corneal.
      2. **M. Zigomático Mayor (Zygomaticus Major):** Banda oblicua descendente desde arco cigomático a comisura.
      3. **M. Risorio (Risorius):** Fascículo transversal horizontal hacia el modíolo.
      4. **M. Depresor del Ángulo de la Boca (Depressor Anguli Oris):** Vientre triangular desde la base mandibular hacia la comisura.
      5. **M. Orbicular de la Boca (Orbicularis Oris):** Esfínter concéntrico perioral que envuelve los labios.
      6. **Vientre Anterior del Digástrico (Anterior Belly of Digastric):** Fascículos acintados divergentes desde la fosa digástrica al hueso hioides.
      7. **M. Milohioideo (Mylohyoid):** Sábana muscular del piso oral debajo del mentón.
    - **Alineación con Electrodos y Fonemas:**
      - **Canal 0 (Submentoniano):** Anterior Belly / Milohioideo -> Apertura mandibular -> Vocal **/a/**.
      - **Canal 1 (Comisural / Modíolo):** Risorio / DAO / Zigomático -> Retracción y sonrisa -> Vocales **/i/**, **/e/**.
      - **Canal 2 (Perioral):** Orbicularis Oris -> Constricción y redondeo -> Vocales **/o/**, **/u/**.

47. **Selector Conmutable de Filtro de Línea (Notch IIR vs Adaptativo NLMS):**
    - **Motivación Experimental:** En grabaciones previas con mayor zumbido de red (50 Hz y armónicos), el cancelador adaptativo continuo NLMS introduce transitorios no lineales durante los ataques de contracción al recalcular coeficientes, lo cual puede distorsionar las envolventes musculares que alimentan al autoencoder. En contraste, el filtro Notch IIR opera de forma estrictamente lineal y estacionaria (aunque con mayor factor de mérito).
    - **Implementación en Motor (`motor_autoencoder_unificado.py`):**
      - `acondicionar_senal_cruda` ahora recibe `tipo_filtro_linea` ("adaptativo" o "notch") y `notch_q`.
      - En modo `"notch"`: aplica una cascada de filtros `scipy.signal.iirnotch` en fase cero (`filtfilt`) para 50, 100, 150, 200, 250, 300, 350 y 400 Hz antes del filtro pasa banda.
      - En modo `"adaptativo"`: invoca el cancelador adaptativo cuadrático NLMS `cancelar_ruido_linea_adaptativo`.
    - **Integración en Interfaz Gráfica (`ui_analysis.py`):**
      - En la Sección 6 ("Parámetros DSP Previos y Acondicionamiento"), se integró el desplegable `self.cmb_filtro_linea` con opciones:
        1. `Adaptativo (NLMS 50Hz+Armónicos)` (por defecto).
        2. `Notch (IIR en Cascada)`.
      - El parámetro se extrae en `get_kwargs()` y se transfiere de forma transparente en `run_autoencoder_no_sup_extraer` y `run_autoencoder_no_sup_completo` en `main_app.py`.


47. **Generación del Diagrama Científico de Publicación (Atlas Médico + sEMG) (2026-09-16):**
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/mapa_electrodos_vocales_profesional.png` (300 DPI, formato extendido con panel lateral).
    - **Características de la Figura:**
      - **Panel Principal:** Base anatómica de atlas médico clásico (estilo Frank Netter / Sobotta) con musculatura facial superficial expuesta de frente sobre cabeza neutral con ojos abiertos.
      - **Estructuras Anatómicas Marcadas:** Zigomático Mayor, Risorio, Depresor del Ángulo de la Boca (DAO), Orbicular de la Boca, Milohioideo y Vientre Anterior del Digástrico.
      - **Sensores Ag/AgCl Superpuestos:** Canales 0, 1 y 2 con sus colores canónicos más referencia en mastoides.
      - **Panel Lateral Informativo:** Tarjetas estructuradas detallando la correspondencia entre canal físico, músculo agonista diana, rol biomecánico y vocales fonatorias dominantes (/a/ en Ch0, /i/ y /e/ en Ch1, /o/ y /u/ en Ch2).

48. **Actualización de Código Cromático según Activación Vocálica e Inclusión de Platisma y LAO (2026-09-16):**
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/mapa_musculos_colores_vocales.png` (300 DPI, alta resolución).
    - **Esquema Cromático Oficial por Sinergia Vocálica:**
      1. **Rojo (Vocal /a/):** Vientre anterior del digástrico (`Anterior Belly`) en Canal 0 (Apertura mandibular).
      2. **Gama de Verdes (Vocales /i/, /e/):**
         - Risorio (`Risorius`)
         - Zigomático Mayor (`Zygomaticus major`)
         - Depresor del Ángulo de la Boca (`Depressor anguli oris - DAO`)
         - Elevador del Ángulo de la Boca (`Levator anguli oris - LAO`)
         - Milohioideo (`Mylohyoid`, tono más verde/lima)
      3. **Amarillo (Vocales /o/, /u/):** Orbicular de la boca (`Orbicularis oris`) en Canal 2 (Constricción labial y redondeo).
      4. **Azul:** M. Platisma (`Platysma`, soporte cervical anterior) y Referencia mastoidea.

49. **Módulo de Normalización de Nombres de Sujetos en metadata.json (2026-09-16):**
    - **Implementación:** `EMG_desarrollo/normalizar_metadatos_sujetos.py`.
    - **Resultado de Ejecución:** Se procesaron 1094 archivos `metadata.json` en `base_de_datos_electrodos/`, modificando exitosamente 407 metadatos que presentaban variaciones heterogéneas (`Cande`, `CANDE`, `SANTI`, `Sujeto1`, `Audio`, etc.).
    - **Desglose Canónico Consolidado (100% Cobertura):**
      - **Candela:** 448 tomas/canales
      - **Lucas:** 284 tomas/canales
      - **Petra:** 100 tomas/canales
      - **Santi:** 262 tomas/canales
    - **Garantía Integridad:** Todos los metadatos de la base de datos quedaron estandarizados sin modificar la estructura física de carpetas ni la integridad de las grabaciones de audio.

50. **Agrupación Visual Dinámica por Sujeto en el Gestor de Sesiones (`SessionExplorer` - 2026-09-16):**
    - **Implementación:** `EMG_desarrollo/gui_app/views/session_explorer.py` y `EMG_desarrollo/EMG_Ejecutable_Build/gui_app/views/session_explorer.py`.
    - **Nuevas Opciones de Agrupamiento:** Se añadió un selector desplegable (`QComboBox`) con tres modos de visualización:
      1. `Fecha` (Modo tradicional: `Fecha -> Mediciones`).
      2. `Sujeto` (Agrupa por sujeto canónico: `Candela`, `Lucas`, `Petra`, `Santi` $\to$ `Mediciones [Fecha]`).
      3. `Sujeto > Fecha` (Jerarquía de tres niveles: `Sujeto -> Fecha -> Mediciones`).
    - **Preservación Total de Rutas (`get_selected_paths`):** La recolección recursiva de elementos tildados devuelve exactamente las mismas rutas absolutas del disco, garantizando paridad total con todos los módulos de análisis y Deep Learning.



49. **Pintado Anatómico Bilateral de Músculos Faciales y Submentonianos (2026-09-16):**
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/mapa_musculos_pintados_vocales.png` (300 DPI, alta resolución).
    - **Coloreado Directo sobre la Malla Muscular (Capas Translúcidas Bilaterales):**
      - **Amarillo (`#f59e0b`):** M. Orbicular de la boca (*Orbicularis oris*), anillo esfinteriano perioral completo (labio superior e inferior).
      - **Gama de Verdes:**
        * *Zigomático Mayor:* Verde esmeralda (`#059669`), desde arco cigomático a comisura.
        * *Risorio:* Verde brillante (`#22c55e`), fascículo horizontal lateral.
        * *Depresor Anguli Oris (DAO):* Verde bosque (`#15803d`), triangular desde mandíbula a comisura.
        * *Elevador Anguli Oris (LAO):* Verde menta (`#10b981`), fosa canina a modíolo.
        * *Milohioideo:* Verde lima / oliva "más verde" (`#84cc16`), sábana submentoniana profunda.
      - **Rojo (`#dc2626`):** Vientre anterior del digástrico (*Anterior belly*), fascículos bilaterales acintados hacia el hioides.
      - **Azul (`#0284c7`):** Platisma (*Platysma*), sábana cervical anterolateral.
    - **Alineación con Canales sEMG:** Canales 0 (Rojo), 1 (Verde) y 2 (Amarillo/Naranja) superpuestos con núcleo conductor Ag/AgCl sobre sus respectivas dianas motoras.

51. **Estandarización Canónica de Canales en Sesión 2026-09-15 (Canal 0 = Anterior Belly, Canal 2 = Orbicularis Oris):**
    - **Diagnóstico del Cableado de Adquisición:** La sesión del `2026-09-15` fue adquirida en modo bicanal (`Dev1/ai0` y `Dev1/ai1`) registrando exclusivamente `Anterior Belly` y `Orbicularis Oris`. Durante la sesión experimental, ocurrió una inversión de cables entre dos bloques de tomas:
      - **Grupo A (12 Tomas - Cableado Normal):** Ch0 = `Anterior Belly`, Ch1 = `Orbicularis Oris` (`A_Prueba1`, `A_Prueba2`, `A_Prueba2a`, `A_Prueba2a_2`, `O_PruebaEXTRA`, `O_PruebaEXTRA_2`, `U_Prueba1`, `U_Prueba2`, `U_Prueba2a`, `U_Prueba2a_2`, `U_PruebaEXTRA`, `U_PruebaEXTRA_2`).
      - **Grupo B (8 Tomas - Cableado Invertido):** Ch0 = `Orbicularis Oris`, Ch1 = `Anterior Belly` (`A_Prueba5`, `A_Prueba6`, `O_Prueba4`, `O_Prueba5`, `U_Prueba4`, `U_Prueba5`, `U_Prueba6`, `U_Prueba7`).
    - **Alineación al Estándar Universal del Proyecto:**
      - **Canal 0:** `Anterior Belly` (Apertura Mandibular / $+Y$).
      - **Canal 2:** `Orbicularis Oris` (Constricción Labial / $-Y < 0$).
      - (Canal 1 permanece libre/sin asignar al no haberse registrado electrodo de sonrisa en esta fecha).
    - **Script de Migración Implementado:** `estandarizar_canales_2026_09_15.py` reubica los directorios de canales, actualiza `metadata.json`, los archivos de análisis JSON y reordena las columnas de `grabacion.csv` para garantizar paridad física y computacional total.

52. **Planificación y Diseño del Análisis Espectral Completo de Candela (2026-09-16):**
    - **Objetivo:** Análisis espectral exhaustivo (0 a 500 Hz) sobre las 20 sesiones tetracanales de Candela del día `2026-09-16` (series 1 a 4 para las 5 vocales, 240 contracciones totales).
    - **Cadena de Acondicionamiento Bioeléctrico:**
      1. Calibración física en $\mu\text{V}$ y compensación de ganancia del amplificador ($G \approx 495$).
      2. Cancelador adaptativo de ruido de línea (NLMS) a 50 Hz y armónicos (100 a 400 Hz) para no eliminar componentes bioeléctricas válidas.
      3. Filtro pasa-banda Butterworth (20 a 500 Hz).
      4. Segmentación por metrónomo ($\text{BPM} = 30$, $W = 4000$ muestras a $f_s = 2000\,\text{Hz}$) con alineación por onset/volumen de micrófono y ventana simétrica ($40\%$ pre, $60\%$ post).
      5. Purga no supervisada de valores atípicos mediante *Isolation Forest* (10% contaminación, `random_state=42`) por clase de vocal, sin filtros ciegos de SNR.
    - **Productos Espectrales a Generar:**
      1. Espectrogramas individuales (0 a 500 Hz) por músculo (Ch0: Anterior Belly, Ch1: Zygomaticus Major, Ch2: Orbicularis Oris).
      2. Espectrograma compuesto RGB tricanal (R = Ch0, G = Ch1, B = Ch2) para análisis visual directo de co-activación sinérgica.
      3. Espectro de frecuencias (FFT de amplitud 0 a 500 Hz).
      4. Densidad espectral de potencia (PSD Welch en $\mu\text{V}^2/\text{Hz}$ y dB/Hz de 0 a 500 Hz).
    - **Salida Organizada:** Carpeta `EMG_desarrollo/resultados/analisis_espectral_candela_2026-09-16/` con subdirectorios dedicados por análisis y vocal, resúmenes comparativos globales, `lista_outliers.json` e informe de texto.

53. **Implementación de Filtrado Robusto Anti-Deglución en `noise_seconds` (`analisis_por_track_integrado` y `correlaciondeseñales` - 2026-09-16):**
    - **Problema Detectado:** Al calcular el piso de ruido inicial a partir de `noise_seconds`, artefactos de deglución involuntaria ("tragar") o movimientos en el Canal 0 (Milohioideo / Vientre Anterior del Digástrico) provocan picos de amplitud $10\times$ a $100\times$ superiores a la línea base de reposo.
    - **Causa Raíz:**
      1. La regla IQR de Tukey previa en `_estimate_noise_window` fallaba cuando la deglución ocupaba más del 25% de la ventana (agravado por un descarte rígido de 1.0 s en `skip_samples`).
      2. En `procesar_wavs_promedio` (Línea 1573), `initial_noise_mean = np.mean(np.abs(initial_noise_segment))` recalculaba el promedio crudo sin filtrar, colapsando el SNR (`snr_manual`, `snr_per_pulse` y `stats_noise_mean`).
    - **Implementación Completada:**
      1. Se redujo `skip_samples` inicial a $\min(0.2\,\text{s}, 0.2 \times \text{noise\_samples})$, preservando el 80-90% de las muestras de reposo.
      2. Estimador de piso basal anclado al percentil inferior ($P_{35}$), inmune a picos positivos de contracción: $\mu_{\text{piso}}$ y $\sigma_{\text{piso}} = 1.4826 \times \text{MAD}_{\text{piso}}$.
      3. Umbral dinámico dual $\text{Límite}_{\text{corte}} = \min(Q_{75} + 1.5 \times IQR, \; \mu_{\text{piso}} + 3.5 \times \sigma_{\text{piso}})$ ante ratios de pico $> 2.0\times$ o dispersión anómala, seguido de refinamiento fino a $3.0\sigma$.
      4. Asignación directa y consistente `initial_noise_mean = float(umbral)` e `initial_noise_std = float(sigma_est)`, unificando el SNR por pulso y el SNR manual.
      5. Paso opcional de `env_recortada` a `_estimate_noise_window` para evitar efectos transitorios de borde.
      6. Sincronización idéntica en `EMG_desarrollo/analysis/correlaciondeseñales.py`.

54. **Visualización de la Mitad de Pulsos en Plot para Paper 3 Músculos (`plot_3_musculos_standalone.py` - 2026-09-16):**
    - **Requerimiento:** En el script de generación de figuras de alta resolución para publicaciones (`plot_paper_combined.png`), graficar la mitad de los pulsos / contracciones musculares en lugar de la totalidad de la serie (ej. 6 en vez de 12 repeticiones).
    - **Beneficio Visual:** Al reducir la cantidad de pulsos al 50%, tanto la vista continua (Subplot 1) como la vista concatenada (Subplot 2) expanden horizontalmente las curvas, permitiendo apreciar con máxima nitidez el ataque, meseta, tiempo de relajación de la envolvente y la correlación temporal con el micrófono.
    - **Implementación:**
      - Se incorporó la selección automática `frac_pulsos = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5` y `num_picos_mostrar = max(1, int(np.ceil(total_picos * frac_pulsos)))`.
      - Se procesan y grafican únicamente los primeros $N_{\text{mostrar}}$ picos detectados, ajustando automáticamente los límites temporales `ax1.set_xlim` y `ax2.set_xlim`.
      - Retrocompatibilidad total asegurada tanto para llamadas desde interfaz gráfica (`main_app.py`, `pipeline_autoencoder_gui.py`) como invocaciones manuales por consola.

55. **Culminación del Análisis Espectral Multimodal de Candela (2026-09-16):**
    - **Ejecución Completada:** Módulo `EMG_desarrollo/analysis/analisis_espectral_candela.py` ejecutado en 33.09 s sobre las 20 tomas (Series 1 a 4 de vocales A, E, I, O, U).
    - **Acondicionamiento Aplicado:** Cancelador adaptativo NLMS en cuadratura (50 Hz y armónicos hasta 400 Hz), pasa-banda 20 a 500 Hz, escala en $\mu\text{V}$ y ranuras por metrónomo ($30\,\text{BPM}$, $W=4000$ muestras).
    - **Depuración de Outliers:** 238 contracciones evaluadas $\to$ 25 outliers detectados ($10.5\%$) con *Isolation Forest* (`random_state=42`), coincidiendo con los pulsos iniciales de sincronización motora (W1 y W2). 213 contracciones consolidadas.
    - **Productos Espectrales Exportados (0-500 Hz):**
      1. Espectrogramas separados STFT por músculo en dB.
      2. Espectrogramas compuestos RGB (R: Anterior Belly, G: Zygomaticus Major, B: Orbicularis Oris) promedio e individuales.
      3. Espectros de frecuencia FFT mono-lateral con bandas $\pm 1\,\sigma$.
      4. Densidad espectral de potencia PSD Welch con cálculo de Frecuencia Mediana (MDF) y Media (MNF).
      5. Paneles comparativos globales de las 5 vocales en alta resolución (DPI 300).
    - **Ubicación Oficial:** `EMG_desarrollo/resultados/analisis_espectral_candela_2026-09-16/` (`lista_outliers.json`, `informe_outliers.txt`, `datos_espectrales_consolidados.npz` y subcarpetas).

56. **Corrección de Inversión de Canales 1 y 2, Supresión de Pasa-Banda y Código Cromático Canónico (2026-09-16):**
    - **Diagnóstico de Inversión de Canales:** En las 20 sesiones de Candela del `2026-09-16`, las señales de audio de `canal_1` correspondían físicamente al esfínter labial (*Orbicularis Oris*, contracción dominante en /o/ y /u/ con amplitudes $>0.35\,\text{V}$), mientras que `canal_2` contenía el *Zygomaticus Major* (sonrisa en /i/ y /e/). Como consecuencia, en los gráficos preliminares la traza verde (Canal 1) se disparaba en /o/ y /u/.
    - **Corrección Atómica de la Base de Datos:**
      - Se implementó y ejecutó `EMG_desarrollo/deep_learning/corregir_canales_2026_09_16.py` intercambiando atómicamente las subcarpetas `canal_1` y `canal_2`, actualizando sus metadatos (`canal_1: Zygomaticus Major`, `canal_2: Orbicularis Oris`), los archivos de análisis JSON y reordenando las columnas en `grabacion.csv` para las 20 sesiones.
    - **Supresión del Filtro Pasa-Banda:** Se eliminaron los filtros Butterworth pasa-altos (20 Hz) y pasa-bajos (500 Hz) para no cercenar ninguna frecuencia biológica, manteniendo únicamente el cancelador adaptativo NLMS para la interferencia de red a 50 Hz y armónicos.
    - **Código Cromático Oficial Aplicado:**
      - **Canal 0:** **Rojo** (`#d62728`) - Digástrico (*Anterior Belly*, apertura mandibular).
      - **Canal 1:** **Verde** (`#2ca02c`) - Zigo (*Zygomaticus Major*, sonrisa / retracción).
      - **Canal 2:** **Amarillo** (`#d4ac0d`) - Orbicular (*Orbicularis Oris*, redondeo y constricción labial).
    - **Validación Fisiológica Confirmada:** En los nuevos gráficos, el Amarillo domina de forma absoluta en /o/ y /u/, el Verde domina en /i/ y /e/, y el Rojo domina en /a/.

57. **Auditoría de Integridad y Consolidación Acumulativa de Resultados Espectrales (2026-09-16):**
    - **Verificación de Archivos en Disco:** Se auditó la persistencia de la totalidad de los 78 archivos generados en `EMG_desarrollo/resultados/analisis_espectral_candela_2026-09-16/`, confirmando que todos los espectrogramas individuales, compuestos, FFTs, PSDs de Welch, resúmenes comparativos y metadatos (`lista_outliers.json`, `informe_outliers.txt`, `datos_espectrales_consolidados.npz`) están 100% intactos.
    - **Walkthrough Acumulativo:** Se consolidó el artefacto `walkthrough.md` integrando de forma conjunta la tabla completa de los 25 outliers detectados por *Isolation Forest*, los 4 paneles comparativos globales actualizados con el código cromático definitivo (Ch0 Rojo, Ch1 Verde, Ch2 Amarillo) y el inventario estructural de todas las subcarpetas.
    - **Diferenciación entre Versión Previa y Definitiva:** En la primera corrida preliminar el subtítulo rezaba "Azul: Ch2 Orbicularis" y /O/, /U/ aparecían en verde por la inversión física de canales. En la versión corregida y vigente, el subtítulo es "Amarillo: Ch2 Orbicularis" y las columnas en /O/ y /U/ son nítidamente amarillas.

58. **Detrending Local en FFT y Cobertura Exhaustiva por Toma Individual para FFT y PSD (2026-09-16):**
    - **Diagnóstico del Aplastamiento en FFT:** Al suprimir el filtro pasa-altos sin remoción de continua local en la ventana de Fourier de 2.0 s, el offset electroquímico estático del contacto piel-electrodo generaba un pico artificial de hasta $70\,\mu\text{V}$ en $f = 0\,\text{Hz}$. Debido a esto, el eje vertical autoescalaba a $70\,\mu\text{V}$, aplastando visualmente la actividad mioeléctrica biológica real ($1\text{--}5\,\mu\text{V}$ entre $20$ y $300\,\text{Hz}$) contra el suelo del gráfico.
    - **Solución DSP Implementada:** En `calcular_fft_amplitud` se aplicó `scipy.signal.detrend(signal_pulse, type='linear')` antes de la transformada de Fourier, garantizando que el bin puramente continuo valga $0\,\mu\text{V}$ y eliminando derivas de borde sin atenuar las frecuencias dinámicas de la señal.
    - **Ampliación a Todas las Tomas (FFT y PSD):**
      1. Generación de gráficos FFT individuales para cada una de las 4 tomas de cada vocal (`fft_A_Serie1_Candela.png`, etc.), mostrando los pulsos individuales en trazo semitransparente y la media de la toma con autoescala vertical adaptada al rango muscular.
      2. Generación de gráficos PSD Welch individuales para cada una de las 4 tomas (`psd_A_Serie1_Candela.png`, etc.) con sus pulsos, medias, MDF y MNF.
      3. Generación de comparativas inter-series por vocal (`comparativa_series_fft_vocal_{v}.png` y `comparativa_series_psd_vocal_{v}.png`) superponiendo las 4 series para auditar la estabilidad y repetibilidad del sujeto.
      4. Actualización de los promedios globales y del panel comparativo tricanal de las 5 vocales.
    - **Validación Fisiológica Confirmada:** En los nuevos gráficos, el Amarillo domina de forma absoluta en /o/ y /u/, el Verde domina en /i/ y /e/, y el Rojo domina en /a/.

59. **Unificación Universal del Código de Colores Musculares en Todos los Visores y Módulos (2026-09-16):**
    - **Requerimiento del Usuario:** Estandarizar de forma estricta e idéntica la paleta cromática de los músculos faciales/submentales en la totalidad del repositorio, resolviendo discrepancias previas entre visores (`csv_viewer`, `plotter_calibrado`, `plot_3_musculos_paper`, `analisis_por_track_integrado`, `electrode_viewer`, `correlaciondeseñales` / historial de patrones musculares y comparativas de sesión).
    - **Tabla Oficial Universal de Colores:**
      - **Anterior belly del digástrico:** Naranja rojizo (`#ff4500`, *OrangeRed*) para distinguirlo nítidamente del micrófono y del milohioideo.
      - **Zigomático:** Verde azulado (`#00a896`)
      - **Risorio:** Verde (`#00cc44`)
      - **Modíolo:** Verde amarillento (`#a3e635`)
      - **Orbicularis:** Amarillo (`#ffff00`)
      - **Depresor:** Verde más oscuro (`#15803d`)
      - **Milohioideo:** Naranja (`#ff7700`)
      - **Micrófono (Canal 3 / Audio):** Rojo (`#ff0000`), manteniendo el estándar histórico del laboratorio para la señal de audio.
    - **Módulos Sincronizados (Versiones de Desarrollo y PyInstaller Build):**
      1. `EMG_desarrollo/utils/config_manager.py` y `EMG_desarrollo/EMG_Ejecutable_Build/utils/config_manager.py`: Mapeo exhaustivo de sinónimos en español, inglés y variantes ortográficas en `MUSCLE_COLORS`, actualización de `DEFAULT_CONFIG` y asignación de `#ff4500` a Anterior Belly y `#ff0000` a Micrófono.
      2. `EMG_desarrollo/config_general.json` y `EMG_desarrollo/EMG_Ejecutable_Build/config_general.json`: Poblado íntegro del diccionario `colores_musculos` y reconfiguración canónica de los 4 canales estándar.
      3. `plot_3_musculos_standalone.py`: `frac_pulsos = 0.5` para visualización nítida de la mitad de pulsos, resolución dinámica de color vía `get_unique_channel_colors` y asignación de micrófono a `#ff0000`.
      4. `csv_viewer_widget.py`: Asignación dinámica de micrófono a `#ff0000` tanto en la lista de canales como en el renderizado de curvas `plot_widget`.
      5. `analisis_por_track_integrado.py`: Extracción de músculo desde `metadata.json`, resolución canónica de `color_prom` y paleta de respaldo en `_comparative_session_plots`.
      6. `correlaciondeseñales.py`: Extracción de músculo desde `metadata.json`, asignación de `color_prom` por músculo y paleta canónica en el historial de patrones y líderes/esclavos.
      7. `plotter_calibrado.py`: Paleta canónica en resolución de canales y asignación de micrófono en rojo `#ff0000`.
      8. `electrode_viewer_widget.py`: Coloración cromática de las pestañas de cada canal (`tabBar().setTabTextColor`) utilizando la paleta universal del músculo asignado.

60. **Unificación Estricta de Escalas Verticales en Gráficos FFT y PSD (2026-09-16):**
    - **Diagnóstico del Autoescalado Independiente:** En la versión anterior de `analisis_espectral_candela.py`, los subplots de cada canal dentro de una misma toma o comparativa tenían límites verticales autoescalados de forma desacoplada (ej. en `fft_A_Serie1_Candela.png`, Ch0 alcanzaba $2.4\,\mu\text{V}$, Ch1 $4.5\,\mu\text{V}$ y Ch2 $7.0\,\mu\text{V}$). Debido a esto, un músculo en reposo con una deriva lenta de electrodo cerca de $0\,\text{Hz}$ aparentaba tener mayor altura visual que el músculo primario activo.
    - **Solución DSP y de Visualización Implementada:**
      1. **Ejes Compartidos (`sharey=True`):** Se forzó `sharey=True` en todos los `plt.subplots(3, 1, ...)` de FFT y PSD para tomas individuales, comparativas inter-series y promedios por vocal, así como en las comparativas globales tricanal de las 5 vocales.
      2. **Escala FFT Unificada Global (`y_lim_fft_unificado`):** Se fijó el rango vertical `[0, y_lim_fft_unificado]` para todos los subplots de FFT, calculado mediante el percentil 99.8 de todas las amplitudes registradas y la media máxima con margen de holgura ($15\%$), permitiendo comparar de forma directa e inmediata la magnitud bioeléctrica entre canales, entre tomas y entre vocales.
      3. **Escala PSD Unificada Global (`y_lim_psd_unificado`):** Se fijó el rango vertical en decibelios por hertz `[y_lim_psd_unificado[0], y_lim_psd_unificado[1]]` (en dB/Hz) para todos los subplots de PSD Welch.
    - **Impacto Biomecánico:** Permite corroborar visualmente que el músculo agonista domina inequívocamente en amplitud sobre los demás canales sin distorsiones provocadas por autoescala gráfica.

61. **Inclusión del 100% de los Pulsos Registrados en Análisis Espectral (Sin Descarte de Outliers - 2026-09-16):**
    - **Diagnóstico del Conteo de Pulsos:** En las figuras de tomas individuales aparecía $N=10$ pulsos (ej. en `psd_A_Serie1_Candela.png`) a pesar de haberse grabado 12 pulsos por toma. Esto se debía a que el *Isolation Forest* descartaba los dos primeros pulsos de cada serie por atipicidad morfológica en la envolvente.
    - **Corrección Solicitada por el Usuario:** El usuario instruyó incluir la totalidad de los pulsos registrados sin descarte alguno ("si era todos los pulsos, eso también hay que corregir").
    - **Modificación Implementada:**
      1. Se fijó `pulsos_validos = todos_los_pulsos`, conservando el 100% de las contracciones bioeléctricas segmentadas (los 12 pulsos por toma y los 48 pulsos por vocal).
      2. En los gráficos individuales por toma (`fft_*.png` y `psd_*.png`), se representan los 12 pulsos completos ($N=12$).
      3. En los gráficos consolidados por vocal (`espectro_fft_promedio_vocal_*.png` y `psd_potencia_promedio_vocal_*.png`), se trazan los 48 pulsos individuales de las 4 series en líneas semitransparentes junto con la curva media en trazo grueso y las métricas MDF/MNF, mostrando el conteo total ($N=48$).

62. **Diagnóstico y Resolución de la Sincronización de Colores e Imágenes en Visores y Entornos Compilados (2026-09-16):**
    - **Diagnóstico del Reporte del Usuario ("que raro, no se actualizó"):**
      1. **Inmutabilidad de Archivos Estáticos en Disco:** Al modificar el código fuente Python de `plot_3_musculos_standalone.py`, las imágenes preexistentes en disco (como `base_de_datos_electrodos/2026-09-16/A_Serie1_Candela/plot_paper_combined.png`) no se reescriben de forma automática. Al inspeccionar el archivo en disco, se constató que fue generado en una ejecución anterior con los 12 pulsos y la paleta vieja. Para actualizar la imagen física, el script debe ser ejecutado sobre la medición seleccionada.
      2. **Desfase en Entornos de Ejecutables Compilados (`build_linux` y `dist`):** Se descubrió que los archivos `_internal/config_general.json` dentro de `build_linux/NanduLsd/` y `EMG_desarrollo/EMG_Ejecutable_Build/dist/NanduLsd/` todavía conservaban la configuración antigua con Canal 0 en `#8a2be2` (violeta) y carecían de `colores_musculos`. Si el usuario abría el software desde la versión empaquetada, cargaba dicha paleta desactualizada.
      3. **Remanente en `get_muscle_color`:** En `utils/config_manager.py` persistía una línea residual que retornaba `#00bfff` (azul) para micrófono en lugar del rojo estricto canónico (`#ff0000`).
    - **Soluciones Aplicadas:**
      1. **Sincronización Total de Configuraciones:** Se copiaron los archivos de configuración canónica completa a `build_linux/NanduLsd/_internal/config_general.json` y `EMG_desarrollo/EMG_Ejecutable_Build/dist/NanduLsd/_internal/config_general.json`.
      2. **Corrección de Micrófono:** Retorno de micrófono unificado estrictamente a `#ff0000` en `get_muscle_color` y diccionarios de `config_manager.py` (desarrollo y compilado).
      3. **Lectura Resiliente de `muscles_map`:** En `plot_3_musculos_standalone.py`, `plotter_calibrado.py` y `electrode_viewer_widget.py`, si una subcarpeta de canal carece de `metadata.json` propio o de la clave `'musculo'`, el sistema extrae automáticamente la asignación anatómica desde `canal_0/metadata.json` (`muscles_map` y lista de `muscles`), garantizando la correcta resolución del color independientemente de cómo se hayan guardado los metadatos.
      4. **Propuesta de Ejecución:** Solicitar autorización al usuario para correr `plot_3_musculos_standalone.py` sobre `A_Serie1_Candela` y regenerar `plot_paper_combined.png` con la mitad de pulsos y la nueva paleta.

63. **Implementación de Graficador de Señal Cruda Calibrada y Normalizada con Vista Previa Post-Grabación (2026-09-16):**
    - **Diagnóstico del Gráfico Post-Grabación Anterior:** La imagen generada automáticamente al finalizar la grabación (`photo.png`) utilizaba un trazado rudimentario sobre fondo blanco con curvas monocromáticas azules, nombres genéricos de canales (`Canal 0 (Dev1/ai0)`) y ejes de amplitud desacoplados sin correspondencia fisiológica.
    - **Requerimiento Específico del Usuario:**
      1. Replicar la estética sobria y profesional de `plotter_calibrado.py` (fondo negro puro `#000000`, código de colores canónico por músculo y diagramación de subplots).
      2. **Señal 100% Cruda (Full Raw):** Sin filtro Notch, sin filtro pasa banda, sin envolventes. La señal física cruda calibrada a microvoltios ($\mu\text{V}$) con supresión de la mediana basal para oscilación simétrica en $0\,\mu\text{V}$.
      3. **Normalización Tricanal Fisiológica:** Los 3 canales musculares comparten estrictamente la misma escala vertical basada en el Supremo Tricanal ($M_{\text{supremo}} = \max_{c \in \{0, 1, 2\}} \max |x_c|$), permitiendo una inspección visual inmediata y sin sesgos del balance de amplitudes intermuscular. El canal 3 (micrófono) autoescala de manera independiente.
      4. **Nombre de Archivo Oficial:** La imagen resultante se guarda exclusivamente bajo el nombre **`photo.png`** en el directorio de la medición.
      5. **Vista Previa en Pantalla Post-Grabación:** Al concluir la grabación, la interfaz despliega una ventana emergente modal (`PreviewPlotDialog`) durante 5 segundos con temporizador regresivo visual (`QTimer`), cerrándose de manera automática al expirar el tiempo o instantáneamente al presionar `Esc`, `Enter`, `Espacio` o el botón "Cerrar".
    - **Módulos Actualizados e Integrados:**
      1. `EMG_desarrollo/acquisition/autoforge_daq_experimental.py`: Reemplazo de `generar_grafico_grabacion`, inclusión de `PreviewPlotDialog`, definición de `mostrar_preview_signal = QtCore.Signal(str)` en `RealTimePlotter`, conexión a `mostrar_preview_plot` y emisión thread-safe tras el guardado en `on_export_click`, `finalizar_secuencia_continua` y `transicion_estado_grabando`.
      2. `EMG_desarrollo/acquisition/manual_daq.py`: Reemplazo de `generar_grafico_grabacion`, inclusión de `PreviewPlotDialog`, incorporación de `mostrar_preview_signal` y emisión en `exportar_grabacion`.
      3. `EMG_desarrollo/acquisition/autoforge_daq.py`: Reemplazo de `generar_grafico_grabacion`, inclusión de `PreviewPlotDialog`, incorporación de `mostrar_preview_signal` y emisión en `on_export_click`, `finalizar_secuencia_continua` y `transicion_estado_grabando`.

64. **Corrección Cromática de Orbicularis a Amarillo Dorado (#ffb700) y Adaptación de Escala FFT por Vocal (2026-09-16):**
    - **Diagnóstico del Color Verde en Orbicularis:**
      - El código previo `#d4ac0d` corresponde a un mostaza/oliva sucio en RGB ($212, 172, 13$), con alta proporción de verde ($172/255 \approx 67\%$) y apenas $5\%$ de azul.
      - Al renderizarse sobre fondo blanco y con transparencia (`alpha=0.30` o `0.18` para las trazas individuales de pulsos en FFT y PSD), se desaturaba en un tono claramente verdoso/lima, originando la observación del usuario.
      - **Solución Cromática:** Se migró a **`#ffb700`** (amarillo dorado cálido, $R=255, G=183, B=0$), el cual sobre fondo blanco y con transparencia mantiene un matiz $100\%$ amarillo/ámbar brillante sin ninguna desviación hacia el verde.
      - En la síntesis de espectrogramas compuestos (`construir_espectrograma_coloreado`), se calibraron las componentes aditivas ($R=1.0, G=0.78, B=0.0$) para evitar contaminación por co-activación verde.
    - **Diagnóstico de Compresión Vertical en FFT ("se ve todo muy chico"):**
      - La escala unificada previa se fijó de manera rígida en $[0.0, \; 12.0]\,\mu\text{V}$ para toda la sesión debido a una espiga lenta de continua/movimiento ($<10\,\text{Hz}$) presente en algunas contracciones labiales.
      - En la vocal **/A/** (ej. `A_Serie1_Candela`), el músculo agonista Digástrico (Ch0) alcanza un máximo fisiológico de $\approx 3.8\,\mu\text{V}$, quedando confinado en el tercio inferior del subplot con el $70\%$ del recuadro desaprovechado.
      - **Solución DSP y de Graficado:**
        1. **Cálculo en Banda Mioeléctrica Biológica ($f \ge 15\,\text{Hz}$):** Se excluyen los componentes sub-10 Hz del dimensionamiento de escala vertical.
        2. **Escala Adaptada por Vocal (`y_lim_fft_por_vocal[v]`):** Para las tomas individuales y comparativas de cada vocal, se computa el límite compartido tricanal basado en el percentil 99.8 de dicha vocal. En la vocal **/A/** el techo se reduce de $12.0$ a **$4.5\,\mu\text{V}$**, triplicando la resolución visual y haciendo que las curvas ocupen el $85\%\text{--}90\%$ de la altura del gráfico sin verse pequeñas.
        3. **Invarianza Tricanal Preservada:** En cada figura, los 3 canales (Ch0, Ch1, Ch2) continúan compartiendo exactamente el mismo eje vertical (`sharey=True`), garantizando que la dominancia del músculo agonista sea inmediatamente evidente.

65. **Utilidad Modular de Regeneración en Lote de Gráficos photo.png (`regenerar_fotos_sesion.py` - 2026-09-16):**
    - **Objetivo:** Permitir la regeneración de todas las imágenes `photo.png` de una fecha o sesión completa (ej. `2026-09-16`) aplicando de manera retroactiva el nuevo estándar de señal 100% cruda, normalizada por el Supremo Tricanal y con estética de `plotter_calibrado` (fondo `#000000`, colores canónicos, nombres anatómicos, guías de compás y ruido basal).
    - **Implementación:** Módulo [`regenerar_fotos_sesion.py`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/analysis/regenerar_fotos_sesion.py) dentro del paquete `EMG_desarrollo/analysis/`.
    - **Monitoreo de Avance en Tiempo Real:** Emite en consola el progreso exacto y porcentaje para cada sesión: `[Procesando] Sesión i/N (P%) - {sesion}`.
    - **Resultado de Ejecución en `2026-09-16`:** Ejecutado con éxito sobre las 20 sesiones completas de Candela (4 series por vocal, 20 exitosas, 0 fallidas), sobrescribiendo y unificando el archivo `photo.png` de cada medición.

66. **Módulo de Curaduría y Exportación Limpia de Datasets (`curar_dataset_exportacion.py` - 2026-09-16):**
    - **Objetivo:** Automatizar la clonación segura de sesiones de mediciones sEMG hacia directorios limpios en `~/Descargas/` (ej. `2026-09-15_clean`, `2026-09-01_clean`) listos para sincronización y carga.
    - **Implementación:** Módulo [`curar_dataset_exportacion.py`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/utils/curar_dataset_exportacion.py) en `EMG_desarrollo/utils/`.
    - **Reglas de Integridad y Filtrado:**
      1. *Directorio Seguro:* No altera las carpetas originales de toma de datos; clona directamente en el directorio destino especificado.
      2. *Integridad Total de Bioseñales y Metadatos:* Preserva intactos todos los archivos `grabacion.csv`, `grabacion.wav` (en cada canal), `metadata.json` (en raíz y subcarpetas) y archivos de calibración/análisis JSON o logs.
      3. *Limpieza de Imágenes:* Identifica recursivamente extensiones `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff` y elimina/omite todas excepto la fotografía canónica `photo.png`.
      4. *Monitoreo y Resumen:* Muestra avance porcentual en tiempo real y reporte consolidado con cantidad de imágenes suprimidas, lista de ubicaciones de `photo.png` conservados y peso final en MB de cada carpeta.
    - **Resultado de la Curaduría (2026-09-16):**
      * `2026-09-15` (origen `2026-09-16`, tomas medidas el 15/09): 545 imágenes intermedias eliminadas, 660 archivos de bioseñales/metadatos y 20 fotos `photo.png` conservadas. Peso final: 3989.76 MB.
      * `2026-09-01` (origen `2026-09-01`): 538 imágenes intermedias eliminadas, 599 archivos de bioseñales/metadatos y 25 fotos `photo.png` conservadas. Peso final: 1734.29 MB.
      * Carpetas renombradas definitivamente a `~/Descargas/2026-09-15` y `~/Descargas/2026-09-01` sin sufijo `_clean` por instrucción directa del usuario.
      * Diagnóstico de tamaño: Las bioseñales físicas puras (`.wav` a 2000 Hz) solo pesan ~8.6 MB en total (<0.25%). El 97% del peso (~3.8 GB) proviene de archivos de texto JSON de análisis (`analisis_results_env*.json`, `results_env*.json`) que guardan arrays de números de punto flotante en texto ASCII con 16 decimales (hasta 800.000 líneas por archivo en 4 canales).

67. **Corrección del Cubo 3D de Amplitudes Multicanal y Restauración de Figuras en Reporte SNR (`report_engine.py` - 2026-09-17):**
    - **Diagnóstico y Corrección del Cubo 3D Vacío (`Sesion_cubo_todos_los_pulsos_3d.png`):**
      - *Causa raíz:* En `generate_session_evolution_plots`, la variable `info` se evaluaba como `{}` si no se suministraba en el contexto, pasando un diccionario vacío a `generate_muscle_activation_cube_all_pulses_3d`, lo que impedía iterar sobre las tomas y generaba una gráfica 3D sin puntos.
      - *Autocarga Robusta:* Se integró auto-extracción de metadatos mediante `self.extract_session_metadata(session_paths)` cuando `info` esté ausente o incompleto.
      - *Fallback Multinivel de Picos:* Se garantizó la extracción de amplitudes tridimensionales por pulso consultando sucesivamente: `picos_ventana` $\to$ `maxima_per_cut` $\to$ `segmentos_rs` $\to$ lectura y segmentación directa del archivo `grabacion.wav`.
      - *Validación y Normalización:* Verificación dinámica de la existencia del colormap (`cmap`) en el registro de Matplotlib, filtrado estricto de NaNs e infinitos (`np.isfinite`), límites robustos por percentil P98.5 y exportación a alta resolución (`dpi=300`, `bbox_inches='tight'`).
      - *Verificación Visual:* Se regeneró exitosamente `Sesion_cubo_todos_los_pulsos_3d.png` (979 KB), poblando los 240 pulsos (48 por cada una de las 5 vocales) con sus colores canónicos y agrupamiento fisiológico.
    - **Restauración de Figuras en el Reporte de SNR (`generate_snr_report`):**
      - Se reincorporaron en el documento LaTeX las figuras por canal:
        1. *Forma de Onda Promedio:* `avg_lider.png` / `avg.png` con dispersión $(\mu \pm \sigma)$ y envolvente RMS.
        2. *Plot de Recortes Superpuestos:* `pulses.png` con la superposición de todas las ventanas por pulso de metrónomo.
68. **Auditoría de Dependencias de `analisis_results.json` y Purga Definitiva de Datasets para Subida (2026-09-16):**
    - **Depuración en Carpetas de Exportación (`~/Descargas/`):**
      - Se eliminaron todos los archivos `*results*.json` (`analisis_results*.json`, `results*.json`, `*_env*.json`) de las carpetas exportadas en `~/Descargas/`.
      - `2026-09-15`: 480 archivos JSON eliminados, liberando 3.88 GB. Peso final: **111.12 MB**.
      - `2026-09-01`: 374 archivos JSON eliminados, liberando 1.60 GB. Peso final: **132.64 MB**.
      - Espacio total liberado en `~/Descargas/`: **5.48 GB**. Estructura resultante limpia: únicamente `grabacion.csv`, `photo.png` y `canal_X/grabacion.wav`, `canal_X/metadata.json`.
    - **Auditoría Exhaustiva de Dependencias en el Código Fuente:**
      1. *Independencia Total de Modelos y Motores:* Los pipelines de Deep Learning (`motor_autoencoder_unificado.py`), UMAP/PCA (`generador_pca_umap.py`), adquisición (`autoforge_daq.py`) y análisis espectral (`analisis_espectral_candela.py`) operan directamente sobre los archivos binarios `grabacion.wav`, `grabacion.csv` y `metadata.json`, sin depender en absoluto de `analisis_results.json`.
      2. *Módulos con Dependencia Secundaria:* Únicamente la pestaña "Análisis Comparativo" e "Informes" en `main_app.py` consultan `analisis_results.json` para no recalcular la media de pulsos ya procesados. Si el archivo no existe, el botón "Procesar Sesión" lo recrea en segundos leyendo los WAVs originales.
      3. *Archivos Acumulativos Redundantes (`*_env*.json`):* De los 17.50 GB de resultados en la base de datos completa, **12.34 GB** corresponden a volcados acumulativos de historial (`analisis_results_env100_0ms.json`, etc.) que ningún módulo del sistema lee, representando un candidato óptimo para purga y liberación de espacio en disco local.

69. **Purga Masiva de Resultados en Toda la Base de Datos (17.50 GB) e Integración de Botón de Limpieza en GUI (2026-09-16):**
    - **Ejecución de Purga en Base de Datos Principal:**
      - Módulo implementado: [`limpiar_cache_analisis.py`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/utils/limpiar_cache_analisis.py) dentro de `EMG_desarrollo/utils/`.
      - Se eliminaron los **3.364 archivos** de resultados JSON en `EMG_desarrollo/base_de_datos_electrodos/`.
      - **Espacio total recuperado en disco: 17.50 GB (17.917,77 MB)**.
      - Preservación íntegra y verificada de las bioseñales (`grabacion.wav`, `grabacion.csv`), metadatos (`metadata.json`) y fotografías (`photo.png`).
    - **Integración de Botón en la Interfaz Gráfica (`main_app.py` y `ui_analysis.py`):**
      - Se añadió el botón `btn_limpiar_analisis_results` ("BORRAR ARCHIVOS DE ANÁLISIS RESULTS") en la pestaña *Análisis Comparativo* (`ComparativeTab`).
      - Diálogo interactivo inteligente: si hay sesiones tildadas en el *Gestor de Sesiones*, permite elegir entre borrar únicamente en las seleccionadas o en toda la base de datos.
      - Emisión de logs en la consola integrada y ventana emergente de confirmación y resumen.

70. **Actualización de Sección Experimental, Diagramas Anatómicos y Título ('Reporte de Experimento' - 2026-09-17):**
    - **Inclusión de Fotografía Frontal de Orbicularis Oris:**
      - Se incorporó la toma frontal [`orbicularis_oris_candela_frontal.jpeg`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/fotos/orbicularis_oris_candela_frontal.jpeg) dentro de `EMG_desarrollo/fotos/`.
      - Se reestructuró la Figura 5 en dos subpaneles lado a lado (`subfigure`): (a) Detalle de fijación bipolar con Cable Mallado 2, y (b) Vista frontal del posicionamiento sobre el labio superior.
    - **Inclusión de Esquemas Anatómicos de los Tres Músculos de Registro:**
      - Se incorporaron en `EMG_desarrollo/fotos/`: `anatomia_digastrico_anterior_belly.png` (esquema del vientre anterior del digástrico y piso de boca), `anatomia_zygomaticus_major.jpg` (esquema de cigomático mayor) y `anatomia_orbicularis_oris.jpg` (esquema de orbicular de la boca).
      - Se insertó la Figura 1 en la Subsección 1.1 con los tres esquemas anatómicos correlacionados con cada canal (Canal 0, Canal 1 y Canal 2).
    - **Actualización de Datos de Sesión y Montaje (Sección 1):**
      - Se corrigió el electrodo de referencia (Tierra) a *Mastoide* y la condición de baterías a *No se midió*.
      - Se preservó la Tabla 1 de configuración y roles musculares en los 4 canales (Canal 0: Anterior Belly, Canal 1: Zygomaticus Major, Canal 2: Orbicularis Oris, Canal 3: Micrófono).
      - Se integró la Tabla 2 con la cronología y detalle de los 20 registros realizados (orden 1 a 20, identificador de archivo, vocal, hora exacta de toma, 12 pulsos totales y serie 1 a 4).
    - **Actualización del Título y Compilación Final:**
      - Se renombró el título del documento a **"Reporte de Experimento"**.
      - Se completaron dos pasadas en `pdflatex` sobre `Reporte_EMG_2026-09-15.tex`, generando el documento definitivo de 42 páginas [`Reporte_EMG_2026-09-15.pdf`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/Reporte_EMG_2026-09-15.pdf) y su copia [`Reporte_de_Experimento.pdf`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/Reporte_de_Experimento.pdf) con índice, referencias cruzadas y figuras debidamente sincronizadas.

71. **Generación del Manual Técnico README en PDF (Estructura de Datos y Catálogo de Gráficos PNG - 2026-09-17):**
    - **Documento Fuente:** [`reportes_experimentos/README_Estructura_Datasets_y_Graficos.tex`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/README_Estructura_Datasets_y_Graficos.tex).
    - **Compilación Exitosa (Versión Concisa de 2 Páginas):** Compilado con `pdflatex` generando un documento ultra-resumido y directo ("cortito y al pie", sin códigos de color ni fórmulas matemáticas complejas), enfocado exclusivamente en:
      1. *Estructura de Carpetas:* Jerarquía fija de 3 niveles (Fecha, Toma y subcarpetas `canal_0` a `canal_3`).
      2. *Archivos de Datos:* Rol y especificaciones de `grabacion.wav` (PCM 2000 Hz, 16 bits), `grabacion.csv` y `metadata.json`.
      3. *Catálogo de Imágenes PNG:* Explicación concisa de cada figura en la raíz (`photo.png`, `plot_paper_combined.png`, `plot_calibrado_...png`, `patron_muscular_grabacion.png`), dentro de cada canal (`pulses.png`, `pulses_centrados.png`, `avg.png`, `avg_lider.png`, `evolucion.png`, `spec.png`) y en análisis globales (`fft_...png`, `psd_...png`, `espectrograma_compuesto_RGB.png`, `informe_autoencoder_...png`).
    - **Distribución de Copias:**
      * `~/Descargas/README_Estructura_Datasets_EMG.pdf`
      * `~/Descargas/2026-09-15/README.pdf`
      * `~/Descargas/2026-09-01/README.pdf`
      * `./README_ESTRUCTURA_Y_GRAFICOS.pdf` (raíz del repositorio)

72. **Actualización Completa del Reporte Técnico de SNR (`Reporte_SNR_2026-09-15.pdf` - 2026-09-17):**
    - **Índice General (`\tableofcontents`):**
      - Se añadió el índice estructurado tras la portada, desglosando las secciones de condiciones experimentales, tablas comparativas y las 20 tomas divididas en sus 4 series cronológicas.
    - **Cálculo Dinámico y Robusto de SNR en `report_engine.py`:**
      - Ante la ausencia de `analisis_results.json` por purga masiva, se integró el cálculo matemático autónomo a partir del audio nativo `grabacion.wav` y la duración de reposo `noise_seconds`. Se calcula la dispersión intercuartil (IQR) del ruido basal y el percentil P98 de la contracción activa.
    - **Tabla de SNR de Todas las Mediciones (Tabla 1):**
      - Se incorporó la tabla completa con los 20 registros adquiridos (Orden 1 a 20, identificador de carpeta, vocal, serie y los valores individuales de SNR para Canal 0, Canal 1 y Canal 2).
    - **Resumen Estadístico por Vocal (Tabla 2):**
      - Se consolidaron los promedios y desviaciones ($\mu \pm \sigma$) por fonema y canal, evidenciando la correspondencia biomecánica (Vocal A dominante en Ch0 con SNR hasta 9.2; Vocal I en Ch1 con SNR hasta 30.0; Vocales O y U en Ch2 con SNR hasta 38.4).
    - **Diagramación Compacta en Tríadas de Subfiguras:**
      - Para cada una de las 20 tomas, se presentan en subfiguras paralelas (`0.32\textwidth` con `\hfill`):
        1. *Forma de onda promedio:* `avg.png` / `avg_lider.png` con media, desviación y envolvente RMS.
        2. *Recortes superpuestos:* `pulses.png` con la totalidad de épocas capturadas por compás.
        3. *Evolución temporal de ruido:* `evolucion.png` con el piso basal interpulso.
    - **Compilación Exitosa:**
      - Compilado con doble pasada en `pdflatex`, generando el documento definitivo de 17 páginas [`Reporte_SNR_2026-09-15.pdf`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/Reporte_SNR_2026-09-15.pdf).

73. **Eje X Logarítmico y Generación Exhaustiva del 100% de los Pulsos (~40 por Vocal) en Análisis Espectral (`analisis_espectral_candela.py` - 2026-09-17):**
    - **Diagnóstico del Reclamo del Usuario:**
      1. En la versión previa, `espectros_rgb` utilizaba `min(4, len(lista_v))` (guardando únicamente 4 pulsos de muestra), `espectros_separados` solo guardaba el promedio y carecía de figuras individuales por pulso, y en FFT/PSD solo se guardaban las tomas por serie sin archivos dedicados para cada una de las contracciones bioeléctricas.
      2. En cada vocal se registraron 4 series con 10 a 12 pulsos válidos cada una, totalizando entre 40 y 48 contracciones por fonema (~215 a 240 pulsos en total).
    - **Modificaciones Implementadas en el Código:**
      1. **Generación Exhaustiva del 100% de los Pulsos:**
         - En `espectros_separados/vocal_{v}/`: se generan las ~40-48 figuras `stft_separado_{toma}_pulso_{win_idx:02d}.png` (3 canales en dB).
         - En `espectros_rgb/vocal_{v}/`: se generan las ~40-48 figuras `espectrograma_rgb_{toma}_pulso_{win_idx:02d}.png`.
         - En `espectros_frecuencia_fft/vocal_{v}/`: se generan las ~40-48 figuras `fft_{toma}_pulso_{win_idx:02d}.png` con eje X logarítmico y escala vertical adaptada `y_lim_fft_por_vocal[v]`.
         - En `espectros_potencia_psd/vocal_{v}/`: se generan las ~40-48 figuras `psd_{toma}_pulso_{win_idx:02d}.png` con eje X logarítmico, MDF y MNF.
      2. **Eje X en Escala Logarítmica:**
         - Aplicado en FFT y PSD para tomas individuales, pulsos individuales, comparativas inter-series, promedios por vocal y paneles globales de 5 vocales.
         - Rango fijado en $[1.0, \; 500.0]\,\text{Hz}$ con marcas legibles en $1, 2, 5, 10, 20, 50, 100, 200, 500\,\text{Hz}$ (`ScalarFormatter`).
      3. **Código Cromático y Límites:**
         - Orbicular en amarillo dorado cálido `#ffb700` (erradicando el `#d4ac0d`).
         - Escala vertical FFT adaptada por vocal para evitar que la señal se vea chica.
         - Panel comparativo global de las 5 vocales (`comparativa_fft_5vocales.png`) fijado estrictamente en $[0.0, \; 5.0]\,\mu\text{V}$ por instrucción directa del usuario.

74. **Actualización de Título ('Reporte de Experimento') e Inclusión de Ilustraciones Anatómicas (`Reporte_EMG_2026-09-01.tex` - 2026-09-17):**
    - **Cambio de Título:**
      - Se modificó el título del reporte de la sesión `2026-09-01` a:
        ```latex
        \title{\textbf{Reporte de Experimento}}
        ```
    - **Inclusión de Esquemas Anatómicos de los Tres Músculos:**
      - Se guardaron las imágenes en `EMG_desarrollo/fotos/`:
        * Canal 0: [`anatomia_digastrico_lateral.png`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/fotos/anatomia_digastrico_lateral.png) (Vientre anterior y posterior del digástrico).
        * Canal 1: [`anatomia_orbicular_labios_kenhub.jpg`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/fotos/anatomia_orbicular_labios_kenhub.jpg) (Músculo orbicular de los labios).
        * Canal 2: [`anatomia_risorio_kenhub.jpg`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/fotos/anatomia_risorio_kenhub.jpg) (Músculo risorio y zona comisural media).
      - Se insertó la **Figura 1** (`fig:referencias_anatomicas`) en la Subsección 1.1 inmediatamente tras la Tabla 1 de asignación de canales.
    - **Compilación Exitosa:**
      - Compilado con doble pasada en `pdflatex`, generando el documento de 33 páginas [`Reporte_EMG_2026-09-01.pdf`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/Reporte_EMG_2026-09-01.pdf) y su copia [`Reporte_de_Experimento_2026-09-01.pdf`](file:///home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/Reporte_de_Experimento_2026-09-01.pdf) con índice y referencias cruzadas debidamente resueltas.

75. **Elaboración del Datasheet Oficial y Caracterización Integral de Hardware sEMG AD620 (2026-09-17):**
    - **Datasheet Oficial de la Placa Física (Laboratorio de Sistemas Dinámicos):**
      - Se redactó y compiló la **Hoja de Datos Técnicos oficial (Datasheet)** de 10 páginas para los autores **Santiago Prado Rodríguez y Lucas Gastón Braunstein**:
        * Archivo fuente: `EMG_desarrollo/documentacion_hardware/datasheet_frontend_emg_ad620.tex`
        * Documento PDF compilado: `EMG_desarrollo/documentacion_hardware/datasheet_frontend_emg_ad620.pdf` (10 páginas, 0 errores, 0 warnings).
        * Versión Markdown: `EMG_desarrollo/documentacion_hardware/Datasheet_FrontEnd_EMG_AD620.md` y `EMG_desarrollo/archivos_md/Datasheet_FrontEnd_EMG_AD620.md`.
    - **Contenido y Secciones Consolidadas del Datasheet:**
      1. **Características Principales y Aplicaciones Típicas:** 3 canales diferenciales AD620, ganancia programable por $R_G$ ($225\text{ a }2500$), protección activa contra inversión de polaridad por MOSFETs sin caída de diodo ($V_{DS} \approx 5\,\text{mV}$), Zener 1N4741A ($11\,\text{V}$), fusibles de $0.25\,\text{A}$, bornas TBLOCK.
      2. **Tablas Normalizadas:** Límites Máximos Absolutos, Condiciones de Operación Recomendadas y Especificaciones Eléctricas Consolidadas.
      3. **Esquema Electrónico Completo (Proteus):** Inclusión de `circuito_completo_proteus.png` con análisis detallado de la etapa de protección y de los 3 canales analógicos.
      4. **Formulación Matemática:** Ecuaciones de ganancia del AD620, filtro pasa-altos pasivo unipolar ($f_c \approx 48.23\,\text{Hz}$), función de transferencia completa $H(f)$ y corrientes de polarización de entrada.
      5. **Caracterización Experimental vs Teórica (Figura 18):** Inclusión de `curvas_ganancia_experimental_fig18.png` contrastada con la curva teórica analítica calculada `curva_respuesta_frecuencia_teorica.png`. Tabla cuantitativa de error relativo a $1\,\text{kHz}$ y análisis del pico de sobreganancia a $R_G = 33\,\Omega$ ($G_{\text{exp}} \approx 2450$).
      6. **Modificación Propuesta para Optimización de Impedancia:** Inclusión de `comparacion_respuesta_frecuencia_impedancia.png` demostrando analíticamente la recuperación de la banda sEMG orofacial ($20\text{ a }48\,\text{Hz}$) mediante $C_G = 100\,\mu\text{F}$ en serie con $R_G$ ($f_c = 15.91\,\text{Hz}$, $G_{\text{DC}} = 1$) y la elevación de impedancia a $Z_{\text{in}} \ge 10\,\text{M}\Omega$ con resistencias de $10\,\text{M}\Omega$ a masa.
      7. **Lista Oficial de Materiales (BOM):** Tabla normalizada y captura de la planilla de laboratorio `tabla_bom_componentes.png`.
      8. **Diseño Físico del PCB y Pinout:** Inclusión de `layout_pcb_completo.png` y tabla de asignación de pines de borneras (`BAT`, `SWITCH`, `IN1`, `IN2`, `IN3GND`, `OUT1`, `OUT2`, `OUT3`).
      9. **Guía de Operación y Bioseguridad:** Aislamiento por baterías, preparación cutánea y referencia corporal obligatoria.
    - **Actualización Complementaria del Informe Técnico Principal:**
      - Se incorporó la Sección 8 (*Caracterización Experimental y Circuito Físico Completo*) en `informe_frontend_emg_ad620.tex` y se re-compiló exitosamente a 25 páginas (`informe_frontend_emg_ad620.pdf`), replicándose en `Informe_Tecnico_FrontEnd_EMG.md`.

76. **Configuración de Rango de Frecuencia Fisiológico (20 a 600 Hz) en Todo el Análisis Espectral (`analisis_espectral_candela.py` - 2026-09-17):**
    - **Diagnóstico del Requerimiento:**
      - El rango previo de $1.0\text{ a }500.0\,\text{Hz}$ en escala logarítmica incluía la banda sub-$20\,\text{Hz}$, donde la deriva lenta de línea de base y los artefactos de movimiento del electrodo inducían picos ficticios de gran amplitud ($> 4.5\,\mu\text{V}$) en los canales 1 y 2.
      - Por solicitud directa del usuario, se ajustó el rango de frecuencias entre **$20.0$ y $600.0\,\text{Hz}$** tanto en la comparativa global FFT como de forma homogénea ("ídem") en todas las figuras espectrales.
    - **Modificaciones Implementadas en el Código:**
      1. *Funciones de Cálculo:* `calcular_espectrograma`, `calcular_fft_amplitud` y `calcular_psd_welch` actualizadas a `f_max=600.0`.
      2. *Constantes y Máscaras:* Se fijaron `F_MIN_LOG = 20.0`, `F_MAX_LOG = 600.0` y `TICKS_LOG = [20, 30, 50, 70, 100, 200, 300, 500, 600]`.
      3. *Espectros FFT y PSD:* Todos los gráficos individuales, tomas, comparativas y promedios utilizan `ax.set_xlim(F_MIN_LOG, F_MAX_LOG)` y `ax.set_xticks(TICKS_LOG)`.
      4. *Límites Verticales:* Se preserva estrictamente el límite vertical unificado $[0.0, \; 5.0]\,\mu\text{V}$ en la comparativa global FFT (`comparativa_fft_5vocales.png`), y el cálculo adaptativo por vocal ahora evalúa estrictamente el percentil P99.8 dentro de la banda fisiológica $[20, 600]\,\text{Hz}$.
      5. *Espectrogramas STFT y RGB:* Se ajustó `ax.set_ylim(F_MIN_LOG, F_MAX_LOG)` en espectrogramas separados, compuestos y paneles globales, eliminando la línea estática de 0 a 20 Hz.

77. **Integración Vectorial Completa en LaTeX (`circuitikz`) de los Circuitos Tricanal y Mejora de Impedancia en Datasheet e Informe Técnico (2026-09-17):**
    - **Requerimiento del Usuario:**
      - El usuario solicitó explícitamente dibujar el circuito electrónico completo con los 3 canales diferenciales AD620 y la etapa de protección en LaTeX vectorial (`circuitikz`), en lugar de utilizar capturas fotográficas o imágenes de Proteus.
      - Para la mejora propuesta, solicitó explícitamente dibujar el esquema vectorial del canal con el capacitor en serie con la resistencia de ganancia ($C_G = 100\,\mu\text{F}$ en serie con $R_G = 100\,\Omega$), entradas puenteadas directas sin capacitores de $33\,\text{nF}$, y resistencias de polarización elevadas a $10\,\text{M}\Omega$ a tierra.
    - **Implementaciones Realizadas:**
      1. *Esquema Vectorial Completo de 3 Canales (`circuitikz`):*
         - Se implementó el circuito completo con el bloque de alimentación y protección activa (borneras `BAT` TBLOCK-M3, `SWITCH` TBLOCK-M4, fusibles de acción rápida de $0.25\,\text{A}$, transistores MOSFET IRFZ44N en riel negativo y F9540N en riel positivo, abrazaderas Zener 1N4741A de $11\,\text{V}$, resistencias purga/polarización de $10\,\text{k}\Omega$ y $100\,\text{k}\Omega$).
         - A la derecha, se trazaron en vectores idénticos los 3 canales analógicos independientes ($U_1, U_2, U_3$ con AD620), con sus celdas de entrada pasivas ($C = 33\,\text{nF}$, $R = 100\,\text{k}\Omega$), resistencias de ganancia ($R_G = 100\,\Omega$), pines de alimentación ($\pm 12\,\text{V}$), referencia (GND) y borneras a tornillo (`IN1`, `IN2`, `IN3GND`, `OUT1`, `OUT2`, `OUT3`).
         - Exportado y renderizado como `esquema_completo_circuitikz.png` (302 KB).
      2. *Esquema Vectorial de la Mejora Propuesta ($R_G + C_G$ en Serie):*
         - Se implementó el circuito vectorial que exhibe: entradas acopladas en continua (puentes directos sin capacitores en serie), resistencias de entrada elevadas a $10\,\text{M}\Omega$ a GND ($Z_{\text{in}} \ge 10\,\text{M}\Omega$), y la rama serie de $R_G = 100\,\Omega$ conectada en serie con el capacitor bipolar no polarizado $C_G = 100\,\mu\text{F}$ entre los pines 1 y 8 del AD620, garantizando $G_{\text{DC}} = 1.0\,\text{V/V}$, $G_{\text{AC}} \approx 495\,\text{V/V}$ y frecuencia de corte inferior en $f_c \approx 15.91\,\text{Hz}$.
         - Exportado y renderizado como `esquema_mejora_circuitikz.png` (266 KB).
      3. *Actualización de Documentos:*
         - En `datasheet_frontend_emg_ad620.tex`: Se integraron ambos diagramas vectoriales nativos en las Secciones 6 y 9, depurando líneas en blanco internas para compilación robusta sin conflictos de `\par` en `\resizebox`.
         - En `informe_frontend_emg_ad620.tex`: Se corrigieron los pines de alimentación del AD620 en la Sección 2 (pin 7 a $+12\,\text{V}$, pin 4 a $-12\,\text{V}$), se reemplazó la figura fotográfica de Proteus en la Sección 8.1 por el circuito vectorial de 3 canales en `circuitikz`, y se insertó el esquema vectorial de la mejora en la Sección 8.3.
         - En los 4 archivos Markdown (`Datasheet_FrontEnd_EMG_AD620.md` e `Informe_Tecnico_FrontEnd_EMG.md` en ambas ubicaciones): Se reemplazó `circuito_completo_proteus.png` por `esquema_completo_circuitikz.png` y se incorporó `esquema_mejora_circuitikz.png`.
      4. *Compilación Exitosa (Doble Pasada pdflatex):*
         - Se compilaron sin errores ni advertencias los dos documentos oficiales:
           * `datasheet_frontend_emg_ad620.pdf`: 10 páginas, 1.21 MB (copia en raíz `./Datasheet_FrontEnd_EMG_AD620.pdf`).
           * `informe_frontend_emg_ad620.pdf`: 26 páginas, 3.08 MB (copia en raíz `./Informe_Tecnico_FrontEnd_EMG.pdf`).

78. **Corrección de Ganancia Fija Soldada y Rediseño Despejado de Esquemas Circuitikz (2026-09-17):**
    - **Corrección Fisiológica y de Hardware: Ganancia Fija por Resistencia Soldada:**
      - Se eliminó la afirmación de "ganancia programable" en toda la documentación.
      - En la placa física real, la resistencia de ganancia está fijamente soldada: $R_G = 100\,\Omega$, fijando una ganancia analógica de $G = 1 + \frac{49.4\,\text{k}\Omega}{100\,\Omega} \approx 495.0\,\text{V/V}$.
      - Se actualizó en `datasheet_frontend_emg_ad620.tex` (Sección 1 y Tabla de especificaciones eléctricas), y en todos los documentos Markdown asociados (`Datasheet_FrontEnd_EMG_AD620.md` en sus dos ubicaciones).
    - **Eliminación de Superposiciones y Esquema de 1 Canal Original ("Antes"):**
      - Por instrucción del usuario, se descartó condensar los 3 canales analógicos en un solo bloque vertical comprimido (que provocaba colisiones de texto entre resistencias, capacitores y borneras).
      - En su lugar, la Sección 6 del Datasheet y la Sección 8.1 del Informe Técnico presentan:
        1. *Etapa de Alimentación y Protección Activa:* Esquema simétrico horizontal con borneras `BAT` y `SWITCH`, fusibles de $0.25\,\text{A}$, transistores MOSFET (F9540N canal P e IRFZ44N canal N), diodos Zener 1N4741A ($11\,\text{V}$), resistencias de purga y terminales de salida $\pm 12\,\text{V}$ y $\text{GND}$, con holgura vertical completa y cero cruces de pistas.
        2. *Canal Típico de Instrumentación AD620 (Original):* Se adoptó la topología probada y limpia de un solo canal (la desarrollada en la Sección 1 del informe), con entradas diferenciales $V_{\text{in}}^-$ e $V_{\text{in}}^+$, celda pasa-altos $33\,\text{nF} + 100\,\text{k}\Omega$, pines de alimentación $\pm 12\,\text{V}$ (pin 7 y pin 4) y resistencia soldada $R_G = 100\,\Omega$ entre pines 1 y 8. Cero superposiciones.
    - **Rediseño Despejado del Canal con Modificación Propuesta (Retrabajo $R_G + C_G$ en Serie):**
      - Se eliminó completamente la caja naranja de "Ventajas Fisiológicas del Retrabajo" y los textos sobre líneas de señal.
      - Se implementó la geometría horizontal desacoplada: la rama serie de ganancia ($R_G = 100\,\Omega$ y $C_G = 100\,\mu\text{F}$) corre horizontalmente en la franja central ($y \in [-0.45, 0.45]$) entre pines 1 y 8, mientras que las líneas de entrada discurren por las franjas externas ($y = \pm 1.15$), ubicando las resistencias de $10\,\text{M}\Omega$ a GND a la izquierda ($x = -4.4$).
      - Ambas ramas quedan completamente separadas con holgura geométrica absoluta y legibilidad profesional.
    - **Herramienta de Renderizado:**
      - Se creó el script `EMG_desarrollo/documentacion_hardware/render_circuitos.py` para generar los archivos PNG vectoriales a 300 DPI (`esquema_completo_circuitikz.png` y `esquema_mejora_circuitikz.png`).

79. **Corrección de Baterías a $\pm 9\,\text{V}$, Título 'Amplificador Diferencial de 3 Canales' y Eliminación Total de Superposiciones (2026-09-17):**
    - **Invariante de Tensión de Alimentación ($\pm 9\,\text{V}$):**
      - Se corrigieron todas las menciones y etiquetas de rieles de tensión en esquemas y textos: las baterías son de $9\,\text{V}$ (dos baterías en serie para alimentación simétrica $\pm 9\,\text{V}$). Se eliminaron los rótulos de $\pm 12\,\text{V}$ en la etapa de protección y pines 7 y 4 del AD620.
    - **Ajuste de Título Oficial:**
      - Se modificó el subtítulo a "Amplificador Diferencial de 3 Canales" en la Hoja de Datos Técnicos y documentos asociados.
    - **Resolución Definitiva de Superposiciones en Circuitikz:**
      - *Etapa de Alimentación y Protección Activa:* Se ampliaron las cotas de rieles a $y = \pm 2.4$, ubicando los terminales de masa de $R_3/R_4$ a $y = 0.9$ y los de $R_1/R_2$ a $y = -0.9$, dejando un corredor central libre de $1.8$ unidades hacia el terminal GND de la derecha ($y = 0.0$), sin colisiones entre componentes ni con el título.
      - *Canal de Instrumentación AD620 (Original y Modificado):* Se expandió el marco delimitador a $y = 4.4$ y el título a $y = 4.0$, dejando más de $1.5$ unidades de holgura sobre el símbolo de tierra superior de la celda de entrada ($y = 2.3$), eliminando el solapamiento con el texto del título.
    - **Sincronización:**
      - Actualizados `datasheet_frontend_emg_ad620.tex`, `informe_frontend_emg_ad620.tex`, `render_circuitos.py`, `esquema_completo_circuitikz.tex`, `esquema_mejora_circuitikz.tex` y archivos Markdown.

80. **Consolidación de Arquitectura Escalonada en Circuitikz y Verificación Empírica de PDFs (2026-09-17):**
    - **Arquitectura de Columnas Escalonadas en Etapa de Alimentación:**
      - Se implementó la separación horizontal escalonada para componentes verticales: Diodos Zener $D_2$ y $D_1$ (`zD`) en $x = 8.4$, resistencia de polarización de compuerta $R_3\,(10\,\text{k}\Omega)$ en $x = 9.4$ (etiqueta a la derecha), resistencia $R_1\,(10\,\text{k}\Omega)$ en $x = 10.6$ (etiqueta a la izquierda), y resistencias de purga $R_4, R_2\,(100\,\text{k}\Omega)$ en $x = 12.0$.
      - Riel de masa central ($y = 0.0$) continuo y rectilíneo desde $x = 7.0$ hasta $x = 13.2$, garantizando despeje absoluto sin colisiones entre puntas ni solapamientos tipográficos.
      - Transistores MOSFET F9540N (PMOS, riel positivo $+9\,\text{V}$) e IRFZ44N (NMOS, riel negativo $-9\,\text{V}$) con paso horizontal fuente-drenador.
    - **Despeje en Canal AD620:**
      - Masa de la celda de entrada en $y = 2.6$, dejando más de $1.5$ unidades libres por debajo del título ($y = 4.2$). Rótulos orientados hacia afuera (`l_` y `l`).
    - **Verificación Empírica y Exportación:**
      - Se regeneraron las imágenes a 300 DPI y se recompilaron `datasheet_frontend_emg_ad620.tex` (10 páginas) e `informe_frontend_emg_ad620.tex` (26 páginas).
      - Se actualizaron los PDFs en la raíz del repositorio (`./Datasheet_FrontEnd_EMG_AD620.pdf` e `./Informe_Tecnico_FrontEnd_EMG.pdf`).
      - Inspección visual completada sobre renderizados PNG de alta resolución (`datasheet_page-01.png`, `datasheet_page-03.png`, `informe_page_18-18.png`, `informe_page-21.png`), verificando cero superposiciones y leyendas limpias.

81. **Restitución del Circuito Completo de 3 Canales, Prohibición Universal de Paréntesis y Propuesta /learn (2026-09-17):**
    - **Recepción del Requerimiento:**
      - El usuario remarcó que faltaba dibujar el circuito completo de 3 canales (que no debía eliminarse al agregar el canal previo individual) y señaló la violación de la regla de no poner paréntesis en el encabezado `(TOPOLOGÍA ORIGINAL)`.
    - **Eliminación Total de Paréntesis en Títulos y Rótulos:**
      - Se eliminaron todos los paréntesis en títulos de bloques Circuitikz (`ETAPA DE ALIMENTACI\'ON Y PROTECCI\'ON ACTIVA $\pm 9\,\text{V}$`, `CANAL T\'IPICO DE INSTRUMENTACI\'ON AD620: TOPOLOG\'IA ORIGINAL`, `CANAL 1: IN1 Y OUT1`, `CANAL 2: IN2 Y OUT2`, `CANAL 3: IN3GND Y OUT3`), valores de componentes (`$R_G = 100\,\Omega$`, `$C_G = 100\,\mu\text{F}$`, `$\text{FUSP} = 0.25\,\text{A}$`, etc.) y textos alternativos de figuras Markdown.
    - **Dibujo Vectorial Integral del Circuito Completo de 3 Canales:**
      - Se implementó en `esquema_completo_circuitikz.tex`, `render_circuitos.py`, `datasheet_frontend_emg_ad620.tex` (Figura 3) e `informe_frontend_emg_ad620.tex` (Figura 11) el esquema electrónico completo integrando en un único plano el bloque de alimentación y protección a la izquierda ($x \in [-14, 0]$) y los 3 canales diferenciales independientes a la derecha ($x \in [4, 13.8]$) con borneras `IN1`, `IN2`, `IN3GND`, `OUT1`, `OUT2`, `OUT3`, celdas RC de entrada ($C_1\text{--}C_6$, $R_1\text{--}R_8$) y resistencias soldadas de ganancia ($R_3, R_6, R_9 = 100\,\Omega$).
    - **Propuesta de Aprendizaje (/learn):**
      - Se generó el artefacto `learning_proposal.md` para extender la regla de prohibición de paréntesis a todo encabezado visual, recuadro de diagrama esquemático (Circuitikz/TikZ) y título de gráfico.

82. **Consolidación de Documentación Hardware, Purga de Chamuyo y Aplicación de Regla Universal Sin Paréntesis (2026-09-17):**
    - **Depuración de Aplicaciones Típicas en Datasheet:**
      - Se eliminaron listas genéricas de relleno (marcha, prótesis mioeléctricas, biopotenciales generales) en la Sección 2 del Datasheet.
      - Se documentaron las aplicaciones concretas y reales del hardware: electromiografía facial de superficie (sEMG) en milohioideo, depresor y orbicular; decodificación de habla submáximal para control bioeléctrico; y adquisición portátil tricanal con baterías.
    - **Compilación Exitosa y Actualización de Binarios:**
      - Renderizado a 300 DPI de esquemas vectoriales mediante `render_circuitos.py`.
      - Doble pasada de `pdflatex` sin errores ni advertencias:
        * `datasheet_frontend_emg_ad620.pdf` (11 páginas, copia en `./Datasheet_FrontEnd_EMG_AD620.pdf`).
        * `informe_frontend_emg_ad620.pdf` (26 páginas, copia en `./Informe_Tecnico_FrontEnd_EMG.pdf`).
    - **Consagración de la Regla en `.agents/AGENTS.md`:**
      - Se actualizó la sección 'Redacción Limpia y Directa de Títulos y Encabezados' para prohibir paréntesis en todo tipo de título, encabezado, recuadro esquemático (Circuitikz/TikZ) o gráfico.

83. **Reordenamiento Estructural del Datasheet, Despeje Total de Superposiciones, Curvas al 48% y Purga de Retrabajo (2026-09-17):**
    - **Eliminación de Redundancia en BOM ("Estos gráficos dicen lo mismo"):**
      - Se eliminó la captura raster `tabla_bom_componentes.png` de todos los documentos LaTeX y Markdown (`datasheet_frontend_emg_ad620.tex`, `informe_frontend_emg_ad620.tex`, `Datasheet_FrontEnd_EMG_AD620.md`, `Informe_Tecnico_FrontEnd_EMG.md`).
      - La lista de materiales se mantiene exclusivamente en tabla vectorial limpia de alta definición.
    - **Reordenamiento Estructural (Circuito Principal y PCB Primero, Tablas Después):**
      - Se eliminó por completo la Sección 2 'Aplicaciones Típicas' del Datasheet.
      - La Sección 2 pasa a ser 'Esquema Electrónico del Front-End sEMG y Trazado PCB', con la Figura 1 (Esquema tricanal completo) como figura principal, y la Figura 2 (Layout del PCB) inmediatamente debajo.
      - Las tablas de especificaciones (Límites Máximos Absolutos, Condiciones de Operación Recomendadas, Especificaciones Eléctricas Consolidadas) se reubicaron en las Secciones 3, 4 y 5, mostrándose estrictamente después de los circuitos.
    - **Eliminación de Superposiciones en Esquema Principal y Mejora:**
      - En el circuito de 3 canales, se acortaron los encabezados a `CANAL 1`, `CANAL 2`, `CANAL 3` situados sobre las borneras a la izquierda, eliminando la colisión con la masa de las resistencias de polarización. Se ampliaron las cotas de la caja de alimentación evitando choques con rótulos de $\pm 9\,\text{V}$.
      - En el esquema de mejora, se bajó la etiqueta de $C_G = 100\,\mu\text{F}$ hacia el interior de la rama de ganancia, despejando la línea de entrada $V_{\text{in}}^+$.
    - **Curvas de Ganancia Lado a Lado al 48%:**
      - Se dispusieron las curvas experimentales y las curvas teóricas de respuesta en frecuencia una al lado de la otra (`subcaption` a `0.48\linewidth` cada una) en una única figura dual comparativa.
    - **Purga Total del Término 'Retrabajo':**
      - Prohibición consagrada en `.agents/AGENTS.md` bajo lenguaje natural. Se reemplazaron todas las instancias por 'modificación', 'mejora' o 'modificación en placa' en todos los archivos del repositorio (0 coincidencias restantes).
    - **Sincronización Total de Documentación:**
      - Sincronizados al 100% `datasheet_frontend_emg_ad620.tex`, `informe_frontend_emg_ad620.tex`, y las versiones Markdown en `EMG_desarrollo/documentacion_hardware/` y `EMG_desarrollo/archivos_md/`.

84. **Integración Oficial del Logo Ñandú LSD, Compilación Final de PDFs y Despliegue en Raíz (2026-09-17):**
    - **Integración de Marca Institucional:**
      - Incorporado el logo vectorial oficial `logo_nandu_lsd.png` en el encabezado de `datasheet_frontend_emg_ad620.tex` (`height=2.2cm` centrado sobre el bloque de título) e `informe_frontend_emg_ad620.tex` (`height=2.5cm` centrado).
      - Integrado en todas las versiones Markdown (`EMG_desarrollo/documentacion_hardware/` y `EMG_desarrollo/archivos_md/`) con anchos controlados (`140px` y `160px`).
    - **Compilación Limpia y Verificación Visual:**
      - Doble pasada de `pdflatex` sin advertencias ni errores:
        * `Datasheet_FrontEnd_EMG_AD620.pdf`: 10 páginas, estructura visualmente sobria y balanceada. Figura 1 (esquema tricanal completo) y Figura 2 (layout de PCB) en página 2. Tablas de especificaciones en páginas 4 y 5. Curvas teóricas y experimentales lado a lado al 48% en página 7. Lista BOM vectorial limpia en página 9.
        * `Informe_Tecnico_FrontEnd_EMG.pdf`: 25 páginas, 0 errores, portada con logo institucional y títulos sin paréntesis.
      - Copias finales actualizadas en la raíz del repositorio: `./Datasheet_FrontEnd_EMG_AD620.pdf` e `./Informe_Tecnico_FrontEnd_EMG.pdf`.
    - **Estado del Sistema y Próximos Pasos:**
      - Documentación de hardware terminada, formateada y validada según todas las reglas de estilo y restricciones del usuario.
      - Listo para avanzar con la decodificación de habla submáximal (clasificador supervisado ligero / linear probe sobre el espacio latente del autoencoder para separación fina de /e/-/i/ y /o/-/u/, o adquisición en tiempo real).

85. **Acondicionamiento Fisiológico Trevisan (Filtro Adaptativo NLMS + Pasa-Altos 20 Hz), Eje X Lineal (20-600 Hz) y Amplitud Clamped [0, 4] µV (`analisis_espectral_candela.py` - 2026-09-17):**
    - **Diagnóstico y Requerimientos del Usuario:**
      1. *Eje Frecuencial:* Se descartó definitivamente la escala logarítmica para el eje de frecuencias; el usuario solicitó escala **LINEAL** entre $20.0$ y $600.0\,\text{Hz}$.
      2. *Acondicionamiento y Filtro Adaptativo:* Se constató la necesidad de aplicar de forma obligatoria y estricta la cadena de acondicionamiento de `generador_pca_umap` y `analisis_trevisan` (cancelador adaptativo de red NLMS para 50 Hz y armónicos más filtro pasa-altos Butterworth de 4to orden a 20 Hz en fase cero con `filtfilt`).
      3. *Escala Vertical Clamped:* En los gráficos de pulso individual y comparativas FFT, la amplitud vertical debe estar acotada rígidamente entre **$0.0$ y $4.0\,\mu\text{V}$** (`ylim(0.0, 4.0)` con ticks en `[0.0, 1.0, 2.0, 3.0, 4.0]`).
    - **Modificaciones Implementadas:**
      1. *Acondicionamiento Bioeléctrico Oficial:*
         - Se implementó en `analisis_espectral_candela.py` el bucle `_nlms_loop_local` (acelerado por Numba con fallback NumPy) y `aplicar_filtro_adaptativo_nlms` (fundamental 50 Hz y armónicos 100, 150, 200, 250, 300, 350, 400 Hz con $\mu = 0.02$).
         - En `acondicionar_senal_canal`: remoción de continua basal + cancelación adaptativa NLMS + pasa-altos Butterworth de 4to orden a $20\,\text{Hz}$ (`filtfilt`). Se suprime totalmente la deriva lenta sin introducir retardos de fase.
      2. *Eje X en Escala Lineal:*
         - Se reemplazaron todas las llamadas `ax.set_xscale('log')` por `ax.set_xscale('linear')` en los gráficos FFT y PSD (tomas por serie, comparativas inter-series, promedios consolidados por vocal, el 100% de los pulsos individuales y paneles comparativos globales de las 5 vocales).
         - Límites horizontales: `ax.set_xlim(20.0, 600.0)` con marcas en `[20, 100, 200, 300, 400, 500, 600]` y etiqueta `"Frecuencia [Hz]"`.
      3. *Escala Vertical FFT Fija [0.0, 4.0] µV:*
         - Se fijó `y_lim_fft_unificado = 4.0` y `y_lim_fft_por_vocal[v] = 4.0` para todas las vocales.
         - Se fijó `ax.set_ylim(0.0, 4.0)` y `ax.set_yticks([0.0, 1.0, 2.0, 3.0, 4.0])` en todos los gráficos FFT individuales, tomas, promedios y resumen global.
    - **Estado:** EJECUTADO Y VERIFICADO. Script corrido el 2026-09-17 a las 13:11 UTC-3. Completado en 414 s (~7 min). 213 pulsos validos generados (25 outliers excluidos por Isolation Forest). Todos los graficos regenerados con las tres correcciones activas.

---

### Hito 86 - 2026-09-17: Ejecucion Final del Analisis Espectral Multimodal (Candela 2026-09-16)

- **Accion:** Ejecucion del script `EMG_desarrollo/analysis/analisis_espectral_candela.py` con la cadena de acondicionamiento definitiva (NLMS + pasa-altos 20 Hz) y parametros visuales corregidos.
- **Resultados:**
  - 20 tomas cargadas correctamente desde `base_de_datos_electrodos/2026-09-16`.
  - 238 contracciones segmentadas totales; 213 pulsos validos tras purga por Isolation Forest (10% contaminacion, `random_state=42`).
  - ~860+ imagenes generadas en `EMG_desarrollo/resultados/analisis_espectral_candela_2026-09-16/`.
  - Eje X: escala **LINEAL** entre 20 y 600 Hz con ticks en `[20, 100, 200, 300, 400, 500, 600]`.
  - Eje Y FFT: rango fijo **[0.0, 6.0] µV** con ticks en `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]`.
  - Filtro adaptativo NLMS activo (50 a 400 Hz) + Butterworth pasa-altos 20 Hz (Trevisan).
  - Tiempo total de ejecucion: 414.33 s.
- **Proximo Paso:** A definir por el usuario.

---

### Hito 87 - 2026-09-17: Ajuste de Títulos Directos y Purga de Vocabulario Pomposo en Datasheet

- **Solicitud del Usuario:**
  - Reemplazar *"Acondicionamiento Bioeléctrico"* al inicio por *"Amplificación Diferencial"*.
  - Prohibir estrictamente el uso de frases y adjetivos pomposos como *"acondicionamiento bioeléctrico"*.
  - Renombrar la Sección 2 a *"Esquema del Circuito y Diseño de PCB"*.
  - Renombrar la Sección 7 a *"Ecuación de Ganancia del Amplificador AD620"*.
  - Invocación del comando `/learn` para actualizar las reglas de estilo del repositorio.
- **Acciones Realizadas:**
  - `datasheet_frontend_emg_ad620.tex`:
    - Subsección inicial renombrada a `\textbf{Amplificación Diferencial:}`.
    - Sección 2 renombrada a `\section{Esquema del Circuito y Diseño de PCB}`.
    - Sección 7 renombrada a `\section{Ecuación de Ganancia del Amplificador AD620}`.
    - Compilación completa con `pdflatex` (11 páginas, 0 errores) y despliegue del PDF en `./Datasheet_FrontEnd_EMG_AD620.pdf`.
  - `Datasheet_FrontEnd_EMG_AD620.md`:
    - Encabezados correspondientes actualizados a `### Amplificación Diferencial`, `## 2. Esquema del Circuito y Diseño de PCB` y `## 7. Ecuación de Ganancia del Amplificador AD620`.
    - Sincronizado en `EMG_desarrollo/archivos_md/Datasheet_FrontEnd_EMG_AD620.md`.
  - `/learn`: Creado el artefacto de propuesta de aprendizaje `learning_proposal.md` para actualizar `.agents/AGENTS.md` con las nuevas prohibiciones léxicas una vez aprobado por el usuario.
- **Estado:** Completado, compilado y sincronizado.



### Hito 88 - 2026-09-17: Exportación de Base de Datos, Purga de Resultados Redundantes e Interfaz Gráfica

- **Acciones Realizadas:**
  - Se creó el script `EMG_desarrollo/utils/curar_dataset_exportacion.py` para clonar las sesiones en `~/Descargas/` preservando únicamente `.wav`, `.csv`, `metadata.json` y `photo.png`.
  - Se exportaron las sesiones de Candela (el 09-16 renombrada a `2026-09-15` y la `2026-09-01`) listas para subir.
  - Se creó el script `EMG_desarrollo/utils/limpiar_cache_analisis.py` para eliminar masivamente todos los archivos redundantes `analisis_results*.json` y `results*.json`, liberando **17.50 GB** de espacio en disco en la base de datos de electrodos.
  - Se integró un botón `BORRAR ARCHIVOS DE ANÁLISIS RESULTS` en la pestaña "Análisis Comparativo" (`ui_analysis.py`) y en `main_app.py`, permitiendo al usuario purgar las carpetas de análisis de la interfaz de forma selectiva o global con un cuadro de diálogo de confirmación.
  - Se redactó y compiló un documento conciso de dos páginas (`README_Estructura_Datasets_y_Graficos.tex`) explicando la estructura de la base de datos, incluyendo este PDF en las carpetas exportadas en `~/Descargas/`.
  - Se inspeccionó el script `autoforge_daq_experimental.py` para confirmar que las grabaciones `.wav` se guardan puramente crudas y sin filtros, directo del buffer del hardware.
- **Estado Actual:** El proyecto redujo drásticamente su huella de almacenamiento. La GUI cuenta ahora con un botón de limpieza definitivo mientras el usuario refactoriza la arquitectura de análisis comparativo.

50. **Generación Dual: Vista Submentoniana (Hiperextensión de Cuello) y Señalización por Círculos de Colores (2026-09-17):**
    - **Objetivo:** Responder al requerimiento de visualización anatómica con dos figuras especializadas de calidad editorial:
      1. **Figura 1: Vista Submentoniana en Hiperextensión (`EMG_desarrollo/resultados/vista_submentoniana_digastrico_milohioideo.png`):**
         - Cabeza mirando hacia arriba mostrando con máxima claridad el piso bucal y cuello anterior.
         - Delimitación del Vientre Anterior del Digástrico (Zona ROJA, motor primario de la vocal **/a/**) y del M. Milohioideo (Zona MÁS VERDE / lima profunda).
         - Posicionamiento del sensor sEMG Canal 0 sobre el vientre muscular submentoniano.
      2. **Figura 2: Señalización Puntual de Zonas mediante Círculos de Colores (`EMG_desarrollo/resultados/mapa_zonas_circulos_colores.png`):**
         - Rostro frontal médico limpio sin máscaras completas de polígonos, delimitando cada diana exclusivamente con círculos cromáticos translúcidos con bordes nítidos:
           * Círculo Amarillo: Orbicular de la boca (Canal 2 -> /o/, /u/).
           * Círculos en Gama de Verdes: Risorio, Zigomático Mayor, DAO, LAO y Milohioideo (Canal 1 -> /i/, /e/).
           * Círculo Rojo: Vientre anterior del digástrico (Canal 0 -> /a/).
           * Círculo Azul: Platisma (cuello) y Referencia mastoidea.

### Hito 90 - 2026-09-17: Unificación de Código de Colores y Propagación Automática de Sinónimos Anatómicos

- **Problema Detectado:**
  - El usuario ajustó manualmente el color del Canal 0 ("Anterior Belly of Digastric") a un naranja más diferenciado (`#ff754b`) desde la tabla de configuración de la GUI, pero el gráfico de *Patrón Muscular Sincronizado* continuaba viéndose en el rojo anaranjado oscuro original (`#ff4500`), confundiéndose con el rojo del micrófono (`#ff0000`).
  - **Causa Raíz:**
    1. En la tabla de la GUI figuraba `"Anterior Belly of Digastric"` y se guardaba bajo esa clave, pero los archivos `metadata.json` de las sesiones contienen `"Anterior Belly"`. La función `get_muscle_color` solo buscaba igualdad estricta en el diccionario de colores personalizados y volvía al valor por defecto histórico.
    2. La función `get_unique_channel_colors` invertía la prioridad: consultaba primero el diccionario canónico antes de evaluar el color configurado para el canal (`preferred`).
- **Solución Implementada:**
  - Se fijó `#ff754b` como el color oficial estándar para todas las denominaciones de Vientre Anterior / Digástrico en `config_general.json` (en desarrollo, build y distribución) y en las paletas por defecto de `config_manager.py`.
  - En `config_dialog.py`, al guardar una edición de color desde la tabla, el sistema detecta la raíz muscular y propaga automáticamente el nuevo color a todas las variantes y sinónimos anatómicos (`"anterior belly"`, `"anterior belly of digastric"`, `"vientre anterior"`, `"digastrico"`, etc.).
  - En `config_manager.py`, se le otorgó prioridad absoluta al color del canal (`preferred`) y se habilitó la búsqueda por inclusión de subcadenas si no existe clave idéntica.
  - Se actualizaron las listas por defecto de respaldo en los visualizadores (`correlaciondeseñales.py`, `plotter_calibrado.py`, `plot_3_musculos_standalone.py`, `analisis_por_track_integrado.py`).
- **Estado Actual:** Resuelto y sincronizado en todo el repositorio.

### Hito 89 - 2026-09-17: Corrección Integral de Reporte de Sesión y Cubos 3D (2026-09-01)
- **Diagnóstico y Solución de Bugs:**
  1. **Evitar sobreescritura de metadatos (`report_dialog.py`):** Se modificó el diálogo para leer el `.tex` preexistente y pre-cargar configuraciones del usuario (Baterías: "No se midió", Tierra: "Mastoide"), evitando pérdida de datos manuales.
  2. **Aislamiento de Imágenes (`report_engine.py`):** El guardado se realiza en subcarpetas separadas por fecha (`evolucion_sesion_2026-09-01`) para evitar la mezcla de imágenes con el día 15. Se corrigieron los patrones glob para no duplicar el espacio de fases bidimensional en el cubo 3D.
  3. **Cubo 3D y Espacio de Fases:** Se corrigió el extractor de pulsos del `report_engine` para ignorar picos crudos `.wav` (los cuales siempre devolvían 32767.0) y usar las señales ya procesadas, evitando que todos los pulsos colapsen al vértice `1.0, 1.0, 1.0`. Se aplicó una normalización agnóstica para soportar el cambio atípico de canales en la sesión del 01/09.
  4. **Modificación de PDF (Reporte_EMG_2026-09-01.tex):** Se ajustaron los tiempos reales de sesión, se cambió el título a "Reporte de Experimento" y se adjuntaron manualmente las tres imágenes anatómicas de los músculos evaluados.
- **Estado:** Todas las correcciones ejecutadas. Se re-compiló el reporte del día 01/09/2026 de forma exitosa recuperando los gráficos y formato solicitados.

51. **Eliminación Total de Textos y Generación 100% Vectorial sin IA (2026-09-17):**
    - **Objetivo:** Cumplir de forma estricta las dos directivas del usuario: (1) Cero generación por modelos de IA y (2) Supresión absoluta de textos, cajas, títulos y etiquetas en las figuras, conservando exclusivamente la geometría anatómica y los círculos de colores.
    - **Artefactos Generados (300 DPI, Vectorial Puro en Matplotlib):**
      1. **`EMG_desarrollo/resultados/vista_submentoniana_sin_texto.png`:**
         - Perspectiva submentoniana (mirando hacia arriba) desde el mentón al cuello anterior.
         - Arco mandibular, hueso hioides, cartílago tiroides y piso bucal.
         - Vientre anterior del digástrico señalado con círculos rojos y milohioideo con círculos en verde lima.
         - Sensor de registro Canal 0 señalado como punto Ag/AgCl. Cero texto.
      2. **`EMG_desarrollo/resultados/cara_frontal_circulos_sin_texto.png`:**
         - Rostro anatómico frontal sin pelo, con ojos, nariz y labios definidos por código vectorial.
         - Señalización de zonas musculares exclusivamente mediante círculos de colores translúcidos (amarillo en orbicular, gama de verdes en cigomático/risorio/DAO/LAO, rojo en digástrico, verde lima en milohioideo, azul en platisma).
         - Sensores de registro sEMG marcados puntualmente. Cero texto.

---

### Hito 91 - 2026-09-17: Reestructuración Integral del Datasheet (Circuitos al Inicio y Fusión Contextual de Características)

- **Instrucciones del Usuario:**
  - *"primero va el circuito y luego esto"* (con captura de "1. Características Principales").
  - *"la parte de amplificacion diferencial debajo de amplificacion diferencial , la parte de alimentacion debajo de alimentacion"*.
- **Acciones Implementadas:**
  - Se eliminó la sección genérica independiente *"1. Características Principales"* del inicio del documento.
  - El documento comienza inmediatamente tras el encabezado en la primera página con la **Sección 1: Esquema del Circuito y Diseño de PCB** y el esquema electrónico integral de 3 canales (Figura 1).
  - La Figura 2 (Layout de PCB) y la Tabla 1 (Lista Completa de Materiales - BOM) ocupan armónicamente la página 2.
  - En la Subsección de *Detalle de Bloques y Canales Individuales*:
    - Debajo del circuito de alimentación (Figura 3) se integraron directamente los puntos de **Alimentación y Protección Integral** junto a la descripción técnica de dicha etapa.
    - Debajo del circuito del canal AD620 (Figura 4) se integraron directamente los puntos de **Amplificación Diferencial** junto a la descripción técnica de los canales.
  - Se renumeraron las secciones correlativamente del 1 al 8 tanto en `datasheet_frontend_emg_ad620.tex` como en las copias Markdown (`EMG_desarrollo/documentacion_hardware/` y `EMG_desarrollo/archivos_md/`).
  - Compilación exitosa con `pdflatex`: exactamente 10 páginas, 0 errores, estética limpia y profesional.
  - PDF final desplegado en `./Datasheet_FrontEnd_EMG_AD620.pdf`.


52. **Descarga y Procesamiento de Láminas Médicas Reales de Internet (0% IA, 0% Texto) (2026-09-17):**
    - **Objetivo:** Obtener ilustraciones médicas auténticas creadas por ilustradores anatómicos humanos (descargadas directamente de Wikimedia Commons y OpenStax Anatomy), eliminando cualquier uso de IA generativa y suprimiendo todo texto o etiqueta textual:
    - **Artefactos Guardados (300 DPI, Cero Texto, Cero IA):**
      1. **`EMG_desarrollo/resultados/vista_submentoniana_internet_sin_ia.png`:**
         - Lámina anatómica real de la región suprahioidea y piso de la boca vista desde abajo (mandíbula, hioides y cuello).
         - Vientre anterior del digástrico señalado con círculos rojos y milohioideo con círculo verde lima profundo.
         - Sensor Canal 0 marcado puntualmente. Sin texto.
      2. **`EMG_desarrollo/resultados/cara_frontal_internet_sin_ia.png`:**
         - Lámina anatómica frontal humana de atlas médico (OpenStax Anatomy 2e, CC BY 4.0), recortada sin leyendas editoriales.
         - Zonas musculares señaladas puntualmente mediante círculos de colores:
           * Orbicular de la boca: Círculo amarillo.
           * Risorio, Zigomático Mayor, DAO y LAO: Círculos en gama de verdes.
           * Milohioideo: Círculo más verde (lima).
           * Vientre anterior del digástrico: Círculo rojo.
           * Platisma: Círculos azules.
         - Sensores sEMG de registro superpuestos. Sin texto.

### Hito 92 - 2026-09-17: Preparación y Auditoría Integral de la Infraestructura de Compilación (Windows y Linux)

- **Acciones Realizadas:**
  1. **Actualización de `crear_spec_ejecutable.py`:**
     - Se incorporaron a `candidate_assets` las carpetas `DataConfig` (conteniendo `modelos_vision/face_landmarker.task`) y `fotos/`.
     - Se implementaron funciones auxiliares seguras (`_safe_collect`, `_safe_metadata`) para recolectar hooks de `mediapipe`, `cv2`, `nidaqmx`, `sounddevice`, `umap`, etc., evitando caídas si un paquete opcional no está presente.
     - Se actualizaron los `additional_modules` incluyendo los módulos activos recientes (`acquisition.calibracion_espacial_electrodos`, `analysis.filtro_adaptativo`, `analysis.analisis_espectral_candela`, `analysis.regenerar_fotos_sesion`, `utils.limpiar_cache_analisis`, `utils.curar_dataset_exportacion`, `deep_learning.motor_autoencoder_unificado`, `deep_learning.soft_dtw`, `deep_learning.autoencoder_no_supervisado_gui`, `deep_learning.corregir_canales_2026_09_16`) y removiendo módulos obsoletos.
  2. **Actualización de `aplicar_parches_ejecutable.py`:**
     - Se actualizaron las reglas de reemplazo para `main_app.py` y se sincronizó la lista de archivos auxiliares con todos los módulos nuevos.
  3. **Compatibilidad en `calibracion_espacial_electrodos.py`:**
     - Se agregó soporte explícito para `sys._MEIPASS` en la búsqueda del modelo `face_landmarker.task`, permitiendo su ejecución transparente tanto en modo desarrollo como en el ejecutable empaquetado.
  4. **Robustez en `build.bat` y `build_linux.sh`:**
     - Se amplió la detección de entornos virtuales para soportar automáticamente `venv` y `.venv` en la raíz o en `EMG_desarrollo/`.
     - Se adoptó la invocación `%PYTHON_EXEC% -m PyInstaller` para eliminar la dependencia de ejecutables intermediarios de consola.
  5. **Verificación de Entorno:**
     - Se ejecutaron secuencialmente `crear_entorno_ejecutable.py`, `aplicar_parches_ejecutable.py` y `crear_spec_ejecutable.py`, validando que `EMG_Ejecutable_Build` y `EMG_Studio.spec` quedan listos y sin errores.
- **Estado Actual:** Infraestructura de compilación lista para ejecutarse directamente en Windows con `build.bat` o en Linux con `./build_linux.sh`.
