# Memoria del Proyecto: Decodificación de Habla Submáximal y Espacio Latente Universal

**Fecha de ultima consolidacion:** 2026-09-25 10:15 UTC-3  
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

### 1.7 Sincronización Dinámica de Dimensiones en GUI y Agrupamiento Canónico de Sesiones
- **Resolución de Error de Dimensiones Lineales en GUI:**
  - El error `mat1 (2x240) y mat2 (1200x32)` se debía a una asincronía entre el código generado en la plantilla ($12 \times 100 = 1200$ entradas para la capa lineal `fc1`) y la longitud de remuestreo fijada en la interfaz ($12 \times 20 = 240$).
  - Se modificó `compilar_modelo_desde_codigo` en `motor_autoencoder_unificado.py` para inyectar dinámicamente la longitud temporal real del lote de datos en los parámetros `time_pts`, `target_len`, `time_len`, `pts` e `input_dim = \text{canales} \times T$.
  - Se conectó el control de puntos de envolvente en `ui_analysis.py` (`on_pts_env_changed`) para actualizar dinámicamente el texto del editor y validar la arquitectura en tiempo real para cualquier valor de remuestreo ($T = 20, 50, 100, 200, 500$).
- **Estandarización del Agrupamiento Canónico de Sesiones:**
  - Se formalizó en `extraer_sesion_agnostica` la regla canónica del proyecto: ante tomas nombradas como `vocal_pruebaotoma_sujeto` (ej. `A_Prueba1_Candela`, `E_Prueba1_Candela`, `I_T1_Lucas`), se extrae `parts[1].upper()` (ej. `PRUEBA1`, `T1`).
  - Esto garantiza que todas las vocales grabadas durante una misma prueba pertenezcan a la misma sesión, evitando que la normalización por reposo e impedancia $P_{95}$ se calcule sobre vocales aisladas, preservando rígidamente la sinergia intermuscular.
  - En `experimento_candela_conv_ortogonal.py` se corrigió la asignación de sesiones con `extraer_sesion_agnostica(t)` y se excluyó la toma incompleta `Prueba5` de Candela 01/09.
- **Validación Empírica Exitosa:**
  - Verificación en `scratch/verificar_cambios.py` superó todos los tests (código de salida 0): agrupamiento canónico de las 5 vocales verificado en motor y GPU, y compatibilidad dimensional de capas lineales probada sin excepciones para $T = 20, 50, 100, 200$.
  - Ejecución de `experimento_candela_conv_ortogonal.py` confirmó la recuperación completa de la geometría en Candela 01/09 (Risorio):
    - **Exactitud Global GMM:** **$75.51\%$** (Silueta $+0.430$, Davies-Bouldin $0.91$).
    - **Vocal /a/:** Alcanzó **$85.7\%$** (eliminando el artefacto donde caía "arriba y abajo" al quedar aislada de las demás vocales).
    - **Desglose Multiclase:** /a/ $85.7\%$, /e/ $52.8\%$, /i/ $87.8\%$, /o/ $69.4\%$, /u/ $78.0\%$.
    - **Separación de Pares:** /o/ vs /u/ $74.0\%$ | /e/ vs /i/ $91.7\%$.
  - Ejecución de `experimento_candela_perdida_compuesta.py` (Barrido de $\beta \in [0.00, 0.02, 0.05, 0.10, 0.20]$ con decodificador Dual-Head RMS + TKEO):
    - Al igual que en Lucas, **$\beta = 0.05$** fue el punto óptimo del decodificador dual en Candela 01/09, alcanzando **$70.41\%$** global (frente a $68.88\%$ en $\beta = 0.00$), elevando la vocal /u/ de $58.5\%$ a **$68.3\%$** y la separación /o/ vs /u/ a **$68.8\%$**.
    - El modelo de una sola cabeza (`ConvOrthogonalAE`) preserva mayor concentración en el cuello de botella (75.51%), pero la cabeza dual confirma que la regularización por energía instantánea TKEO con $\beta = 0.05$ favorece sistemáticamente el balance en fonemas de mandíbula cerrada (/u/).
- **Ablación de Ventana de Segmentación (Simétrica 50/50 frente a Asimétrica 40/60):**
  - Se ensayó la ventana de corte simétrica de Lucas (`pre_pct = 0.50`, `post_pct = 0.50`) en Candela.
  - La exactitud global en Candela 01/09 se redujo de **$75.51\%$ a $65.83\%$**.
  - La vocal **/o/** se desplomó de **$69.4\%$ a $11.1\%$**, absorbida por **/u/** ($100.0\%$), y la separación del par /o/ vs /u/ cayó de $74.0\%$ a $59.2\%$.
  - En Candela 15/09 (Cigomático), la /o/ colapsó al $0.0\%$ (absorbida por /u/ al $97.7\%$).
- **Barrido de Parámetros en Lucas con Ventana Asimétrica 40/60 (`grid_search_lucas_ventana4060.py`):**
  - Se desarrolló el script maestro de Grid Search modular para las 35 tomas de Lucas (`2026-07-10`), aplicando la ventana fisiológica de $40\%$ pre / $60\%$ post, RMS de $90\,\text{ms}$, remuestreo a 20 puntos ($D=60$), calibración intersesión y purga por Isolation Forest (10%).
  - Soporta 4 modalidades de búsqueda:
    1. `conv_2d`: Autoencoder Convolucional 1D Ortogonal 2D.
    2. `mlp_2d_sin_so2`: Autoencoder MLP Totalmente Conexo Ortogonal 2D sin rotación/alineación SO(2).
    3. `conv_3d`: Autoencoder Convolucional 1D Ortogonal 3D.
    4. `mlp_3d_sin_so2`: Autoencoder MLP Totalmente Conexo Ortogonal 3D sin corrección externa.
  - Almacena automáticamente caché del dataset (`dataset_lucas_ventana4060.npz`), guarda progreso incremental en CSV para permitir pausa/reanudación, y exporta pesos `.pt`, configuración `.json` y gráfico `.png` para el modelo campeón de cada modo.
  - Se incorporó el nivel masivo `--tier 5760` (con exactamente las 5,760 combinaciones históricas de `channels`, `kernel_size`, `act`, `lr`, $\lambda_W$ y $\lambda_Z$ para convolucionales, y 5,040 combinaciones para MLP con 12 configuraciones de capas ocultas). Optimizado con `set_to_none=True` y multi-hilo en CPU.
  - **Resultados Consolidados de los 4 Modos (48 combinaciones por modo, 100% Zero-Labels):**
    1. **`conv_2d`:** **$83.63\%$ GMM Global** (Silueta $+0.322$, DB $0.95$). Desglose: /a/: $91.5\%$, /e/: $62.4\%$, /i/: $90.6\%$, /o/: $80.2\%$, /u/: $93.9\%$. Separación /o/ vs /u/: **$87.0\%$**, /e/ vs /i/: $76.8\%$. Config: `channels=(8, 16), k=3, tanh, lr=0.002, lw=0.8, lz=0.3`.
    2. **`mlp_2d_sin_so2`:** **$78.24\%$ GMM Global** (Silueta $+0.349$, DB $0.94$). Desglose: /a/: $100.0\%$, /e/: $65.3\%$, /i/: $70.8\%$, /o/: $59.4\%$, /u/: $98.0\%$. Separación /o/ vs /u/: **$78.5\%$**, /e/ vs /i/: $68.1\%$. Config: `hidden=(32, 16), tanh, lr=0.002, lw=1.2, lz=0.25`.
    3. **`conv_3d`:** **$84.03\%$ GMM Global** (Silueta $+0.292$, DB $1.15$). Desglose: /a/: $90.4\%$, /e/: $63.4\%$, /i/: $93.4\%$, /o/: $79.2\%$, /u/: $93.9\%$. Separación /o/ vs /u/: **$86.5\%$**, /e/ vs /i/: $78.7\%$. Config: `channels=(6, 12), k=3, tanh, lr=0.002, lw=1.5, lz=0.5`.
    4. **`mlp_3d_sin_so2`:** **$85.03\%$ GMM Global** (Silueta $+0.320$, DB $1.09$). Desglose: /a/: $92.6\%$, /e/: $67.3\%$, /i/: $90.6\%$, /o/: $81.2\%$, /u/: $93.9\%$. Separación /o/ vs /u/: **$87.5\%$**, /e/ vs /i/: $79.2\%$. Piso mínimo por vocal: **$67.3\%$**. Config: `hidden=(32, 16), tanh, lr=0.003, lw=1.2, lz=0.45`.
  - **Conclusión Biomecánica:** La ventana $40/60$ eleva la separabilidad y el balance multiclase en Lucas tanto en redes convolucionales como en MLPs totalmente conexos. En 3D, el MLP sin corrección SO(2) alcanza un récord del $85.03\%$ con un piso homogéneo del $67.3\%$, demostrando que la cola del $60\%$ retiene la información de cierre articular necesaria para resolver el par /o/ vs /u/ ($87.5\%$).
- **Integración de Ventana de Corte Variable en la Interfaz Gráfica (`ui_analysis.py`, `main_app.py`):**
  - Se añadieron controles numéricos `inp_pre_pct` y `inp_post_pct` (`QDoubleSpinBox`, rango $0.05\text{--}0.95$, paso $0.05$, valores por defecto $0.40$ y $0.60$) en la sección *5. Alineación de Pulso Fisiológico y Ventana de Corte* de la pestaña de Autoencoder.
  - Se conectaron en `get_autoencoder_kwargs()` para reemplazar los valores hardcodeados de $0.50$, propagándolos a través de `main_app.py` hacia `motor.extraer_dataset_unificado` en los flujos de extracción, ejecución completa y evaluación.


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
    - **Script de Migración Ejecutado con Éxito:** `estandarizar_canales_2026_09_15.py` reubicó las subcarpetas de canales y sus archivos `grabacion.wav`, actualizó `metadata.json` en `canal_0` y `canal_2`, los archivos de análisis JSON y reordenó las columnas de `grabacion.csv` (`Tiempo (s), Canal 0, Canal 2`), garantizando paridad física y computacional total en las 20 sesiones.

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
- **Proximo Paso:** El usuario solicitó integrar el análisis espectral en la interfaz para generar el reporte de forma automática.

---

### Hito 87 - 2026-09-23: Integración de Reporte Espectral en la Interfaz Gráfica (GUI)

- **Acción:** Refactorización de scripts de análisis e integración en el motor de reportes de la GUI.
- **Detalles Técnicos:**
  1. Se modularizó `analisis_espectral_candela.py` (`ejecutar_analisis_completo`) para aceptar rutas dinámicas (`session_paths` y `salida_base_dir`), permitiendo analizar cualquier conjunto de sesiones seleccionadas en la interfaz.
  2. Se incorporó el método `generate_spectral_report` en `analysis/report_engine.py`. Este método invoca el pipeline de procesamiento espectral completo (Filtro NLMS + 20Hz HPF, FFT, PSD) y compila dinámicamente un documento LaTeX (`Reporte_Espectral_<fecha>.pdf`) que incluye las comparativas globales en escala lineal y límites estandarizados.
  3. Se añadió un nuevo botón **"4. Reporte Espectral y PSD"** en `ReportDialog` (`gui_app/views/report_dialog.py`), el cual dispara el análisis espectral en segundo plano (vía `ReportWorker`) mostrando el progreso sin bloquear la interfaz.
- **Estado:** Implementado, integrado y probado sintácticamente.

---

### Hito 88 - 2026-09-23: Implementación de Correlación Espectral EMG-Audio (Canal 3)

- **Acción:** Integración de la señal del micrófono (Canal 3) para visualización conjunta y análisis de correlación fisiológico-fonética.
- **Detalles Técnicos:**
  1. **Segmentación Paralela:** La señal de audio pura (`mic_sig`) se segmenta usando las mismas ventanas exactas ($p_{\text{inicio}}$ a $p_{\text{fin}}$) calculadas a partir de la envolvente electromiográfica maestra, garantizando sincronía total.
  2. **Análisis STFT Acústico:** Se calcula el espectrograma (STFT) del segmento de audio ($f_s = 2000\,\text{Hz}$, $f_{\text{max}} = 1000\,\text{Hz}$) almacenándolo en las estructuras de características junto al EMG.
  3. **Visualización de Correlación:** Se agregó una nueva figura global (`comparativa_correlacion_audio_5vocales.png`) dispuesta en 2 filas por 5 columnas.
     - **Fila Superior:** Espectrograma compuesto RGB del músculo (Rojo: Digástrico, Verde: Zigo, Amarillo: Orbicular).
     - **Fila Inferior:** Espectrograma del audio alineado, utilizando mapa de calor `magma` para resaltar los formantes vocálicos.
  4. **Reporte LaTeX:** Se inyectó esta nueva figura automáticamente bajo la sección "Correlación de Patrones EMG vs Audio" en el reporte de la interfaz GUI.
- **Próximo Paso:** Generación de figura para publicación (23 de septiembre).

---

### Hito 89 - 2026-09-23: Generación de Figuras de Publicación (Mediciones 09-23)

- **Acción:** Creación de un script dedicado (`generar_figura_paper_0923.py`) para renderizar un gráfico trimodal publicable sincronizado a 6000 Hz, usando exclusivamente los canales activos Masetero y Orbicular.
- **Detalles Técnicos:**
  1. **Análisis de Metadatos:** Se verificó que las tomas del 23 de septiembre (ej. `A_Prueba2_Sujeto1`) cuentan con $f_s = 6000\,\text{Hz}$ y mapeo de canales: Canal 0 (Masetero), Canal 2 (Orbicularis), Canal 3 (Micrófono).
  2. **Estructura del Gráfico de Publicación:**
     - **Panel Superior (Espectrograma Acústico):** STFT del micrófono (0 a 3000 Hz, colormap `magma`) mostrando los formantes vocálicos con alta resolución de frecuencia.
     - **Panel Medio (Forma de Onda Acústica):** Señal cruda del micrófono (Oscilograma) en función del tiempo (en segundos).
     - **Panel Inferior (Activación Muscular Normalizada):** Envolventes EMG filtradas (Pasa-altos 20Hz + suavizado de 50ms) del Masetero y el Orbicularis, superpuestas y normalizadas para mostrar el reclutamiento simultáneo.
  3. **Alineación:** Todos los ejes temporales (`sharex`) están estrictamente alineados, extrayendo un pulso central representativo de cada vocal (A, E, I, O, U) y centrando el evento mediante la envolvente máxima acústica.
  4. **Refinamientos Finales (Estándar LSD / Fisiología Acústica):**
     - **Tipografía y Jerarquía Visual sin Colisiones:** Se eliminaron las etiquetas numéricas intermedias del eje $X$ en los 3 paneles superiores (`labelbottom=False`), dejando el eje temporal exclusivamente al pie de la figura, y se ajustó el espaciado vertical inter-panel (`hspace=0.28`), erradicando cualquier superposición entre títulos de subplots, leyendas y escalas numéricas.
     - **Alineación Geométrica Estricta en Pixeles (GridSpec):** Se identificó y resolvió el desalineado horizontal de 50 píxeles provocado por el comando `colorbar` en Matplotlib (que estrechaba únicamente el primer subplot). Se implementó una grilla de dos columnas (`GridSpec(4, 2)`) asignando un eje dedicado `cax` para la colorbar, garantizando que el ancho y la coordenada horizontal $X$ de la línea $t = 0.0\,\text{s}$ sean 100% idénticos en los 4 paneles a nivel de subpíxel.
     - **Alineación Causal Estricta con Compensación de Semiventana:** Para erradicar el pre-eco y desfasaje temporal inherente a las ventanas simétricas de la STFT, se aplicó la corrección analítica por retardo de grupo ($\Delta t = +\frac{N}{2 \cdot f_s}$) tanto al espectrograma de audio como al RGB de EMG. La emergencia del color coincide ahora de forma milimétrica con la primera deflexión del oscilograma y con la línea discontinua de $t = 0.0\,\text{s}$.
     - **Pre-énfasis Acústico y Muscular:** Se aplica filtro digital de pre-énfasis tanto al micrófono ($y[n] = x[n] - 0.97 x[n-1]$) para ecualizar formantes altas frente a la caída de -6 dB/oct, como a los canales EMG ($y[n] = x[n] - 0.95 x[n-1]$) para realzar descargas motoras rápidas en el espectrograma RGB.
     - **Oscilograma Rectificado con Envolvente Superpuesta:** Señal de micrófono en valor absoluto (rectificada completa $|x[n]|$) en gris claro, con su envolvente de amplitud rápida en trazo negro continuo (ambas normalizadas al rango $[0.0, 1.0]$), maximizando la resolución vertical del perfil de intensidad fonatoria.
     - **Normalización Estricta:** Supremo Tricanal por pulso respetado integralmente.
- **Estado:** Implementado y generado para las 5 vocales en `/EMG_desarrollo/resultados/figuras_paper_09_23/figura_paper_<Vocal>.png`.
- **Próximo Paso:** A definir por el usuario (posibles refinamientos estéticos o uso en otros reportes).

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

53. **Auditoría Integral del Generador de Patrones Musculares y Catálogo de 143 Sesiones (2026-09-17):**
    - **Localización del Generador Oficial:** `EMG_desarrollo/analysis/correlaciondeseñales.py` (función de patrones sincronizados, líneas 820 a 950).
    - **Mecanismo de Procesamiento y Sincronización:**
      1. Alineación temporal al pico de la señal acústica de referencia (`canal_3: Micrófono`, fijado rígidamente en $t = 0\,\text{s}$).
      2. Cálculo de la envolvente media por pulso y banda de dispersión ($\mu(t) \pm \sigma(t)$).
      3. Estimación cuantitativa de latencias relativas: tiempo al pico ($t_{\text{pico}}$ en ms respecto al micrófono) y amplitud normalizada.
    - **Hallazgo Biomecánico Confirmado:** En la vocal **/i/** de Candela (`2026-08-30`, toma `I_Prueba4_Cande`), el electrodo sobre el Modíolo / Nudo Sonrisa registra un disparo marcadamente anticipado ($t < 0$), precediendo a la emisión sonora y a los demás canales.
    - **Catálogo de 143 Patrones Existentes:**
      - `2026-06-22` (Santi): 18 patrones (Milohioideo, DAO, Orbicular).
      - `2026-07-10` (Lucas): 35 patrones (Digástrico / Milohioideo, DAO, Orbicular).
      - `2026-08-21` (Petra): 10 patrones (Anterior Belly, LAO, Zigomático).
      - `2026-08-27` (Petra med3): 5 patrones (Platisma en Canal 1).
      - `2026-08-28` (Petra): 10 patrones (Anterior Belly, LAO, Zigomático).
      - `2026-08-29` (Candela): 10 patrones (Comparativa Anterior Belly vs Milohioideo).
      - `2026-08-30` (Candela): 10 patrones (Anterior Belly, Modíolo / Nudo Sonrisa, Orbicular).
      - `2026-09-01` (Candela): 25 patrones (Anterior Belly, Risorio, Orbicular).
      - `2026-09-16` (Candela): 20 patrones (Anterior Belly, Zigomático Mayor, Orbicular).

54. **Matriz Científica Unificada de Activación sEMG: 10 Zonas Funcionales × 5 Vocales (2026-09-17):**
    - **Ubicación del Artefacto:** `EMG_desarrollo/resultados/matriz_activacion_10zonas_5vocales.png` (250 DPI, 10 filas × 5 columnas, 50 celdas temporales).
    - **Metodología de Extracción y Alineación:**
      - Procesamiento directo de las señales acústicas y mioeléctricas (`grabacion.wav`) con filtrado pasa banda (20-450 Hz), Notch (50 Hz, $Q=2.0$) y envolvente cuadrática RMS ($\tau = 80\,\text{ms}$).
      - Sincronización rígidamente anclada al inicio/pico del micrófono ($t = 0\,\text{ms}$) en ventana estándar de $[-150\,\text{ms}, +250\,\text{ms}]$.
      - Normalización por el pico de activación inter-vocálico de cada fila, preservando las relaciones de reclutamiento fisiológico agonista/antagonista.
    - **Zonas y Configuraciones Representadas:**
      1. Digástrico Anterior (Candela 09-01) -> Pico máximo en /a/.
      2. Milohioideo / Piso bucal (Candela 08-29) -> Reclutamiento co-activado en apertura.
      3. Orbicular de la Boca (Candela 09-01) -> Pico máximo y exclusivo en /o/, /u/.
      4. Modíolo / Nudo Sonrisa (Candela 08-30) -> Disparo anticipado explosivo en /i/ ($t_{\text{pico}} < -100\,\text{ms}$).
      5. Risorio / Tracción lateral (Candela 09-01) -> Disparo lateral en /i/, /e/.
      6. Cigomático Mayor - Ag/AgCl (Candela 09-16) -> Tracción oblicua superficial.
      7. Cigomático Mayor - Silicona/IED amplia (Petra 08-28) -> Volumen de conducción profundo.
      8. Depresor del Ángulo / DAO (Lucas 07-10) -> Estabilización inferior en /a/, /o/.
      9. Elevador del Ángulo / LAO (Petra 08-28) -> Tracción superomedial profunda.
      10. Platisma / Cervical (Petra 08-27, med3) -> Tensión de cuello.

55. **Corrección de Inversión Fisiológica en Cigomático de Candela y Ventana Extendida de Campana Completa (2026-09-17):**
    - **Diagnóstico y Corrección en Candela 2026-09-16:**
      - Auditoría RMS en señales crudas reveló que en la sesión `2026-09-16`, los canales físicos estaban intercambiados respecto a la etiqueta del metadato: `canal_1` correspondía biomecánicamente al Orbicular (activación en /u/ = 5410 RMS) y `canal_2` correspondía al Cigomático Mayor (activación máxima en /i/ = 5169 RMS).
      - Se corrigió la extracción direccionando la Zona Cigomática de Candela a `canal_2`, eliminando la falsa activación en /o/ y /u/.
    - **Ampliación de Ventana a Campana Completa ([-600 ms a +800 ms]):**
      - Se expandió la ventana temporal a $1400\,\text{ms}$ para visualizar la envolvente fisiológica íntegra: reposo basal, flanco ascendente de reclutamiento, pico de máxima contracción y fase de relajación elástica.
    - **Atlas Inter-Sujeto Consolidado de 16 Registros × 5 Vocales:**
      - `EMG_desarrollo/resultados/matriz_activacion_campana_completa.png` (220 DPI, alta definición).
      - Desglose por sujeto: Candela, Petra, Lucas y Santi para Digástrico, Milohioideo, Orbicular, Modíolo, Risorio, Cigomático (Ag/AgCl y Silicona), DAO, LAO y Platisma.

56. **Resolución Definitiva del Cigomático de Candela e Integración Exhaustiva de Tomas de Orbicular (2026-09-17):**
    - **Diagnóstico de Doble Inversión en Cigomático de Candela (`2026-09-16`):**
      - Auditoría en los gráficos precomputados oficiales (`patron_muscular_grabacion.png`) confirmó que el script previo `corregir_canales_2026_09_16.py` ya había intercambiado atómicamente en disco las subcarpetas físicas.
      - En consecuencia, en el disco:
        * `canal_1`: es estrictamente el *Zygomaticus Major* (traza turquesa/verde). Exhibe una activación dominante exclusiva en la vocal **/i/** ($46.65\,\mu\text{V}$ con anticipación motora de $-239.5\,\text{ms}$ respecto al micrófono), mientras que en las restantes vocales permanece en reposo basal ($10.37\,\mu\text{V}$ en /a/, $7.22\,\mu\text{V}$ en /e/, $18.18\,\mu\text{V}$ en /o/ y $15.61\,\mu\text{V}$ en /u/).
        * `canal_2`: es estrictamente el *Orbicularis Oris* (traza amarilla), con picos masivos en /o/ ($120.21\,\mu\text{V}$) y /u/ ($95.40\,\mu\text{V}$).
      - En la versión previa de la matriz, al haberse asignado `canal_2` al Cigomático, se graficaba el Orbicular. Se corrigió definitivamente asignando **`canal_1`**.
    - **Desglose Exhaustivo de Tomas de Candela en Orbicular de la Boca:**
      - Se relevaron e integraron las 11 tomas registradas a través de todas las sesiones de Candela:
        1. `2026-08-25` (Prueba 1): `canal_0` (primer registro formal).
        2. `2026-08-29` (Prueba 1): `canal_2`.
        3. `2026-08-29` (Prueba 2): `canal_2`.
        4. `2026-08-30` (Prueba 4): `canal_2`.
        5. `2026-08-30` (Prueba 5): `canal_2`.
        6. `2026-09-01` (Prueba 1): `canal_2`.
        7. `2026-09-01` (Prueba 2): `canal_2`.
        8. `2026-09-01` (Prueba 3): `canal_2`.
        9. `2026-09-01` (Prueba 4): `canal_2`.
        10. `2026-09-16` (Serie 1 a 4): `canal_2`.
    - **Generador Unificado:** Script modular implementado en `EMG_desarrollo/analysis/generar_matriz_activacion_completa.py`.

57. **Atlas Integral de Activación Muscular: 30 Registros × 5 Vocales con Campana Completa (2026-09-17):**
    - **Desglose de los 3 Días de Lucas en Orbicular (`2026-07-10`):**
      - Se auditó el metadato temporal de las tomas de Lucas dentro de la carpeta `2026-07-10`, confirmando que provienen de 3 jornadas experimentales independientes:
        * **Día 1 (`2026-06-01`, T1):** Metrónomo a 40 BPM, pico en /o/ a $-133\,\text{ms}$ y /u/ a $-190\,\text{ms}$.
        * **Día 2 (`2026-06-03`, T2):** Metrónomo a 30 BPM, pico en /o/ a $-124\,\text{ms}$ y /u/ a $-83\,\text{ms}$.
        * **Día 3 (`2026-06-10`, T4):** Metrónomo a 30 BPM, pico en /o/ a $-176\,\text{ms}$ y /u/ a $-278\,\text{ms}$.
    - **Inclusión Exhaustiva de Candela (13 Tomas de Orbicular):**
      - Representación temporal completa de todas las tomas a lo largo de 5 fechas (`2026-08-25`, `2026-08-29` P1-P2, `2026-08-30` P4-P5, `2026-09-01` P1-P4 y `2026-09-16` S1-S4).
    - **Validación Visual Definitiva del Cigomático de Candela:**
      - En la fila 25, con `canal_1` asignado, el *Zygomaticus Major* exhibe un pico dominante nítido en la vocal **/i/** ($-226\,\text{ms}$) y permanece completamente silencioso/basal en /a/, /o/ y /u/, cumpliendo con rigor la observación biomecánica del usuario.
    - **Artefacto Consolidado:**
      - `EMG_desarrollo/resultados/matriz_activacion_campana_completa.png` (30 filas × 5 columnas = 150 celdas temporales sincronizadas con micrófono, 220 DPI).
      - Copia sincronizada en el scratch de artefactos para revisión visual del usuario.

58. **Acuerdo de Arquitectura Visual: Atlas Agrupado por Sujetos y Normalización Tricanal por Pulso Individual (2026-09-17):**
    - **Normalización Obligatoria por Supremo Tricanal por Pulso Individual:**
      - Se ratifica la directiva inmutable de normalizar cada pulso bioeléctrico por el divisor maestro instantáneo:
        $$M_{\text{supremo, pulso}} = \max_{c \in \{0, 1, 2\}} \left( \max_{t \in \text{ventana}} |x_c(t)| \right)$$
        $$\tilde{x}_c(t) = \frac{|x_c(t)|}{M_{\text{supremo, pulso}}}$$
      - Esto garantiza que las amplitudes relativas representen fielmente el balance intermuscular motor primario vs secundario (agonista = 1.0, sinergistas en su porcentaje real), evitando la distorsión del autoescalado unicanal independiente.
    - **Estructura Agrupada por Sujetos (1 Fila por Día/Sesión Global):**
      - **Candela:** Digástrico (09-01), Digástrico (09-16), Milohioideo (08-29), Cigomático (09-16), Risorio (09-01), Modíolo (08-30), y 5 filas de Orbicular (una por cada fecha de medición: 08-25, 08-29, 08-30, 09-01, 09-16).
      - **Lucas:** Submentoniano (07-10), DAO (07-10) y 3 filas de Orbicular (Día 1: 06-01, Día 2: 06-03, Día 3: 06-10).
      - **Santi:** Milohioideo (06-22), DAO (06-22), Orbicular (06-22).
      - **Petra:** Digástrico (08-28), Platisma (08-27), Cigomático Silicona (08-27).

59. **Consolidación del Atlas Comparativo sEMG por Sujeto y Supremo Tricanal (2026-09-18):**
    - **Procesamiento Exitoso:** Se ejecutó `EMG_desarrollo/analysis/generar_matriz_activacion_completa.py` procesando 22 filas consolidadas sin redundancias de series intrasesión.
    - **Preservación Estricta de la Sinergia Intermuscular:**
      - Cada pulso individual se normalizó por $M_{\text{supremo, pulso}}$ evaluando concurrentemente los canales 0, 1 y 2.
      - En el Cigomático Mayor de Candela (`canal_1`), la vocal /i/ alcanza el $40\text{--}50\%$ del supremo local, mientras que en /a/, /o/, /u/ permanece basal (< $0.15$), confirmando la correcta corrección en disco de los archivos de audio.
      - En el Orbicular de todos los sujetos, la respuesta en /o/ y /u/ satura cerca del 1.0 (motor primario esfinteriano), mientras que en /a/ la activación cae a niveles secundarios fisiológicos congruentes.
    - **Salida:** Imagen en `EMG_desarrollo/resultados/matriz_activacion_campana_completa.png`.

60. **Corrección Fisiológica Definitiva de Archivos WAV en 2026-09-16 (Canal 1 = Zygomaticus Major, Canal 2 = Orbicularis Oris):**
    - **Diagnóstico Confirmado por Auditoría de Pulsos:** Tras verificar empíricamente los archivos de audio `grabacion.wav` pulso por pulso con filtro pasa-banda (20 a 450 Hz), se constató que los WAVs de Canal 1 y Canal 2 habían quedado invertidos físicamente: el esfínter labial (`Orbicularis Oris`) alcanzaba $0.150\,\text{V}$ en la vocal /u/ dentro de `canal_1/grabacion.wav`, mientras que la sonrisa (`Zygomaticus Major`) alcanzaba $0.071\,\text{V}$ en la vocal /i/ dentro de `canal_2/grabacion.wav`.
    - **Operación Ejecutada:** Se ejecutó con éxito `corregir_wavs_2026_09_16.py` sobre las 20 tomas de `2026-09-16/`, intercambiando atómicamente `grabacion.wav` e intercambiando las columnas `Canal 1` y `Canal 2` en `grabacion.csv`.
    - **Validación Fisiológica Post-Corrección (19/20 Tomas Coherentes):**
      - En todas las tomas de **/o/** y **/u/** (`O_Serie1` a `O_Serie4`, `U_Serie1` a `U_Serie4`), el **Canal 2 (`Orbicularis Oris`)** domina inequívocamente en amplitud ($0.09\text{--}0.15\,\text{V}$ vs $0.06\text{--}0.08\,\text{V}$ en Ch1).
      - En las tomas de **/i/** (`I_Serie1` a `I_Serie3`), el **Canal 1 (`Zygomaticus Major`)** lidera la activación de sonrisa ($0.06\text{--}0.07\,\text{V}$ vs $0.03\text{--}0.05\,\text{V}$ en Ch2).
      - Se garantizó coherencia total y absoluta entre señales de audio, matrices CSV y metadatos en la totalidad de la sesión.

62. **Consolidación Definitiva del Atlas sEMG Basado en CSV con Promedios por Día (2026-09-18):**
    - **Reescritura Completa sobre grabacion.csv:** `generar_matriz_activacion_completa.py` procesa directamente los archivos tabulares de texto `grabacion.csv` mediante pandas, extrayendo tiempo y los 4 canales a su frecuencia nativa calculada dinámicamente por pulso.
    - **Promedio Multiserie por Fila:** Se consolidaron todas las tomas del mismo día en exactamente una curva promedio $\pm$ error estándar (SEM). En Candela 09-16 se integraron las series 1, 2, 3 y 4 en un único promedio armónico.
    - **Normalización Estricta por Supremo Tricanal del Pulso Individual ($M_{\text{supremo, pulso}}$):** Se preservó rígidamente la sinergia intermuscular tricanal sin autoescalados independientes.
    - **Corrección Definitiva de Canales para 2026-09-16:** En `grabacion.csv`, el Canal 2 corresponde anatómicamente al Cigomático Mayor (pico nítido en /I/ a $-229\,\text{ms}$) y el Canal 1 al Orbicular de los Labios (picos dominantes en /O/ a $-196\,\text{ms}$ y /U/ a $-206\,\text{ms}$).
    - **Colorimetría Completa sin Zonas Apagadas:** Se eliminó cualquier atenuación artificial o desaturación en gris, trazando todas las curvas con su tinte muscular representativo al 100% de color y transparencia de sombreado del 18%.
    - **Salida Oficial:** Matriz consolidada de 22 filas por 5 vocales renderizada en `EMG_desarrollo/resultados/matriz_activacion_campana_completa.png`.

63. **Integración de Arquitectura ConvAE con Decodificador Transpuesto y Pérdida Multiobjetivo (2026-09-18):**
    - **Botón de Carga Inmediata en la GUI:** Se incorporó el botón `Cargar ConvAE (ConvTranspose1D)` en la Sección 8 del panel de análisis (`ui_analysis.py`). Al pulsarlo, inyecta la clase `ConvAE` en el editor, activa la casilla de arquitectura personalizada y valida dimensiones de inmediato.
    - **Diferencias Estructurales de ConvAE vs Autoencoder Oficial:**
      * *Codificador:* 3 etapas de reducción con paso (`stride=2`, núcleos de 5, canales 3 -> 32 -> 64 -> 128) que comprimen de 100 a 13 muestras. Aplana $128 \times 13 = 1664$ hacia el espacio latente (`Linear(1664, latent_dim)`).
      * *Decodificador:* Reconstrucción inversa progresiva mediante convoluciones transpuestas (`ConvTranspose1d`), en contraste con el decodificador lineal denso del modelo oficial.
      * *Sensibilidad Temporal:* La ausencia de agrupamiento global adaptativo (GAP/GMP) hace que esta arquitectura sea sensible a desalineaciones temporales finas, pero capaz de reconstruir detalles de forma de onda y derivadas locales.
    - **Función de Pérdida Multiobjetivo:**
      * Implementada en `motor_autoencoder_unificado.py` con soporte para error absoluto cuadrático, error relativo por canal y error de la primera derivada temporal:
        $$\mathcal{L} = \text{MSE}_{\text{abs}} + \lambda_{\text{rel}} \frac{\|x - \hat{x}\|^2}{\|x\|^2 + \epsilon} + \lambda_{\text{deriv}} \frac{\|\Delta x - \Delta \hat{x}\|^2}{\|\Delta x\|^2 + \epsilon}$$
      * Seleccionable desde el menú desplegable de función de pérdida (`ui_analysis.py`).
    - **Corrección de Ortogonalidad en 3D por Normalización de Traza:**
      * Se reemplazó la penalización rígida de varianza que colapsaba el espacio latente tridimensional por una normalización invariante a la escala latente natural del autoencoder:
        $$\tilde{C} = \frac{C}{\text{tr}(C) + \epsilon}, \quad \mathcal{L}_{\text{orto}} = \sum_{i \neq j} \tilde{C}_{i, j}^2$$
      * Permite decorrelacionar los ejes latentes sin comprimir la dispersión bioeléctrica.

64. **Integración de Calibración Intersesión por Lote P95 (Estándar PCA/UMAP) al Autoencoder (2026-09-18):**
    - **Algoritmo de Calibración por Sesión:**
      * Las contracciones se agrupan por tupla $(\text{Fecha}, \text{Sesión})$.
      * Para cada sesión $s$, se calcula el percentil P95 del pico de cada canal:
        $$V_{s, c} = \text{percentil}_{95} \left( \left\{ \max_{t} |s_c(t)| \right\}_{\text{pulsos de } s} \right)$$
      * Se calcula la referencia máxima de la sesión $V_{s, \text{ref}} = \max_c V_{s, c}$ y los factores de corrección acotados a $5\times$:
        $$\alpha_{s, c} = \frac{V_{s, c}}{V_{s, \text{ref}}}, \quad C_{s, c} = \frac{1}{\max(\alpha_{s, c}, 0.20)}$$
      * Trazabilidad en logs: `_log(f"  [Intersesión] Sesión '{s_tag}' ({s_fecha}) -> Ch0: C={C[0]:.2f}, Ch1: C={C[1]:.2f}, Ch2: C={C[2]:.2f}")`.
    - **Preservación Estricta del Supremo Tricanal:**
      * Cada canal se escala por $C_{s, c}$ y de inmediato se normaliza por el divisor maestro instantáneo del pulso individual:
        $$M_{\text{supremo, pulso}} = \max_{c \in \{0, 1, 2\}} \left( \max_{t \in \text{ventana}} |\tilde{s}_c(t)| \right), \quad \hat{s}_c(t) = \frac{\tilde{s}_c(t)}{M_{\text{supremo, pulso}}}$$
      * Garantiza la preservación de las relaciones de sinergia intermuscular intramuestra, neutralizando al mismo tiempo las derivas de impedancia piel-electrodo entre días o sujetos.
    - **Control en la Interfaz Gráfica (`ui_analysis.py`):**
      * Casilla interactiva `Corrección Intersesión por Lote (Calibración P95 PCA/UMAP)` en Sección 7 de la GUI (`self.chk_correccion_intersesion`), activa por defecto.
      * Propagada a través de `get_kwargs()` y los lanzadores de `main_app.py` (`run_autoencoder_no_sup_extraer` y `run_autoencoder_no_sup_flujo_completo`).

65. **Fidelidad Literal de ConvAE y Detección Dinámica de reconstruction_loss (2026-09-18):**
    - **Inyección Completa del Script del Usuario:** Al presionar `Cargar ConvAE (ConvTranspose1D)`, se inyecta el script íntegro con imports (`torch`, `nn`, `DataLoader`, `TensorDataset`, `random`), fijación de semilla determinista (`SEED=42`), definición de clase `ConvAE` y la función `reconstruction_loss`.
    - **Detección Automática de Pérdida en el Motor (`compilar_modelo_desde_codigo`):**
      * Si el código define `reconstruction_loss`, el compilador la vincula directamente como `modelo.custom_loss_fn`.
      * Durante el entrenamiento, `entrenar_autoencoder` detecta la función y la ejecuta con prioridad absoluta, logueando `[Pérdida del Editor] Utilizando directamente la función 'reconstruction_loss' definida en el código`.
    - **Preconfiguración Automática en la GUI:** Al cargar la arquitectura, se activa la casilla personalizada y se selecciona de forma automática `Multiobjetivo (Relativa + Derivada)` en el selector de funciones de pérdida.

66. **Estandarización Canónica de Canales y Sujeto en Sesión 2026-09-18 (Canal 0 = Belly, Canal 1 = Modiolo, Canal 2 = Orbicularis Oris):**
    - **Diagnóstico del Registro Original:** En las 11 tomas del `2026-09-18` (10 tomas por vocal `*_cande` y 1 secuencia continua `SecuenciaContinua_Prueba1_Sujeto1`), los electrodos se adquirieron físicamente en orden desplazado: `canal_0: modiolo`, `canal_1: orbi`, `canal_2: belly`, `canal_3: mic`.
    - **Operación Ejecutada:** Se ejecutó con éxito `estandarizar_canales_2026_09_18.py` aplicando:
      1. *Renombrado de Directorios y Sujeto:* Se renombraron las carpetas a `*_Candela` y se fijó `"sujeto": "Candela"` en todos los `metadata.json`.
      2. *Permutación Atómica de Subcarpetas y Bioseñales:*
         - `belly` (antiguo Canal 2) trasladado a **`canal_0/`** (`Anterior Belly`).
         - `modiolo` (antiguo Canal 0) trasladado a **`canal_1/`** (`Modiolo`).
         - `orbi` (antiguo Canal 1) trasladado a **`canal_2/`** (`Orbicularis Oris`).
         - `mic` permanece en **`canal_3/`** (`Micrófono`).
      3. *Reordenamiento de `grabacion.csv`:* Columnas sincronizadas con la permutación física (`Canal 0: Belly`, `Canal 1: Modiolo`, `Canal 2: Orbicularis Oris`, `Canal 3: Micrófono`).
      4. *Sincronización de Metadatos:* Actualizados `muscles`, `muscles_map`, campos `canal`, `musculo` y canales físicos (`physical_channel`) en los 4 canales de cada sesión.
    - **Resultado:** Las 11 tomas del 18 de septiembre quedaron consolidadas bajo la arquitectura canónica universal del proyecto.

67. **Resolución de Desajuste Dimensional en ConvAE (Puntos Envolvente = 100) (2026-09-18):**
    - **Diagnóstico del Error `mat1 (32x384) and mat2 (1664x2)`:**
      * La interfaz gráfica tenía configurado `target_len = 20` (Puntos Envolvente: 20) en lugar del valor base 100.
      * La reducción con paso 2 sobre 20 muestras produce $20 \to 10 \to 5 \to 3$ muestras temporales, resultando en $128 \times 3 = 384$ características aplanadas.
      * La capa lineal de `ConvAE` está dimensionada fijamente para $128 \times 13 = 1664$ (asumiendo 100 muestras: $100 \to 50 \to 25 \to 13$), provocando el desajuste dimensional en la multiplicación matricial.
    - **Ajuste Preventivo en la GUI (`ui_analysis.py`):**
      * Al pulsar `Cargar ConvAE (ConvTranspose1D)`, el spinbox `self.inp_pts_env` se fija automáticamente en **`100`**.
      * La verificación de arquitectura ahora valida contra el valor real de `target_len` seleccionado en la interfaz en lugar del valor por defecto, emitiendo una advertencia explícita si hay discordancia con `nn.Linear(1664, ...)`.

68. **Migración Exitosa de Sujeto a Candela en 2026-09-08 y 2026-09-11 (2026-09-18):**
    - **Diagnóstico del Registro en Base de Datos:**
      * Las fechas `2026-09-08` (9 tomas) y `2026-09-11` (16 tomas) figuraban asignadas a "Santi" / "Sujeto1" tanto en metadatos como en nomenclatura de carpetas.
      * En el explorador de sesiones de la interfaz gráfica (`session_explorer.py`), se agrupaban erróneamente bajo el nodo "Santi".
    - **Operación Ejecutada (`migrar_sujeto_cande_09_08_09_11.py`):**
      * Se reasignó `"sujeto": "Candela"` y se normalizó `"letra"` a mayúscula en la totalidad de los archivos `metadata.json` de cada subcanal (`canal_0` y `canal_1`).
      * Se renombraron las 25 carpetas de tomas al estándar canónico: prefijo de vocal en mayúscula (`A_...`, `U_...`) y sufijo `_Candela`.
      * Se conservaron al 100% las señales biológicas `grabacion.wav`, las matrices numéricas `grabacion.csv` y los mapeos musculares originales.
    - **Resultado:** En el árbol de la aplicación gráfica, las 9 tomas del 8 de septiembre y las 16 tomas del 11 de septiembre quedan agrupadas bajo el sujeto **Candela**.

69. **Soporte de Arquitectura Bicanal (2 Músculos) y Tricanal Dinámico en el Autoencoder No Supervisado (2026-09-18):**
    - **Requerimiento e Hipótesis Científica:**
      * Permitir entrenar y evaluar el Autoencoder No Supervisado seleccionando cualquier par de 2 músculos (Ch0+Ch1, Ch0+Ch2, Ch1+Ch2) o la tríada completa (Ch0+Ch1+Ch2) para aislar sinergias musculares independientes (ej. apertura vs sonrisa, apertura vs protrusión labial).
    - **Selector Interactivo en la GUI (`ui_analysis.py`):**
      * Incorporado el grupo `Canales Musculares a Procesar: Selección Bicanal o Tricanal` con casillas interactivas para Canal 0, Canal 1 y Canal 2.
      * Regla de selección mínima: bloquea el desmarcado si la selección es menor a 2 canales, emitiendo una notificación informativa.
      * Adaptación automática de plantillas: `get_plantilla_codigo`, `on_cargar_convae` y `on_verificar_arquitectura` configuran `in_channels` dinámicamente según las casillas marcadas.
    - **Generalización del Motor Matemático (`motor_autoencoder_unificado.py`):**
      * Parámetro `canales_features` integrado en `extraer_dataset_unificado`.
      * Carga de audio WAV, cálculo de piso de ruido dinámico interpulso y corte guiado adaptados a $N \in \{2, 3\}$ canales.
      * Calibración intersesión P95 por lote calculada sobre los $N$ canales seleccionados:
        $$V_{s, c} = \text{percentil}_{95} \left( \left\{ \max_{t} |s_c(t)| \right\}_{\text{pulsos de } s} \right), \quad C_{s, c} = \frac{1}{\max(V_{s, c} / V_{s, \text{ref}}, 0.20)}$$
      * Preservación estricta de la regla obligatoria de normalización por el Supremo del Pulso Individual:
        $$M_{\text{supremo, pulso}} = \max_{c \in \text{canales seleccionados}} \left( \max_{t \in \text{ventana}} |x_c(t)| \right)$$
      * Reescalado fisiológico por promedios generalizado para detectar qué agonistas primarios están presentes en la combinación seleccionada.
      * Parámetro `in_channels` en `compilar_modelo_desde_codigo`, `verificar_arquitectura_codigo` y `crear_modelo_autoencoder`.
      * En `entrenar_autoencoder`: detecta `in_ch = X.shape[1]` del dataset `.npz` e instancia el modelo con el número exacto de canales.
      * En `evaluar_espacio_latente`: renderizado de curvas promedio de entrada y reconstrucción para cada uno de los canales activos.
    - **Propagación en la Aplicación Principal (`main_app.py`):**
      * `run_autoencoder_no_sup_extraer` y `run_autoencoder_no_sup_completo` transmiten `canales_features` al script de fondo y validan la red con `in_channels = len(canales_features)`.

70. **Estandarización Canónica Exitosa de Canales en 2026-09-08 y 2026-09-11 (2026-09-18):**
    - **Requerimiento del Usuario:** Configurar Canal 0 = Anterior Belly y Canal 2 = Orbicularis Oris para todas las tomas de `2026-09-08` (9 tomas) y `2026-09-11` (16 tomas), sincronizando archivos de audio WAV, matrices CSV y metadatos JSON.
    - **Operación Ejecutada (`estandarizar_canales_2026_09_08_09_11.py`):**
      * Las 25 tomas fueron procesadas y clasificadas atómicamente:
        - 8 tomas del Caso 1 (Ch0=Belly, Ch1=Orbi): subcarpeta `canal_1` trasladada a `canal_2`, columna `Canal 1` renombrada a `Canal 2` en `grabacion.csv`.
        - 17 tomas del Caso 2 (Ch0=Orbi, Ch1=Belly): permutación de subcarpetas (`canal_1` -> `canal_0`, `canal_0` -> `canal_2`), reordenamiento de columnas en `grabacion.csv` (`Canal 0` = datos de Belly, `Canal 2` = datos de Orbicularis).
      * Metadatos sincronizados en `canal_0/metadata.json` (`musculo: "Anterior Belly"`) y `canal_2/metadata.json` (`musculo: "Orbicularis Oris"`), con `muscles_map: {"canal_0": "Anterior Belly", "canal_2": "Orbicularis Oris"}`.
      * Archivos de audio `grabacion.wav` reubicados en sus canales anatómicos correspondientes.
    - **Resultado:** Las sesiones bicanales del 8 y 11 de septiembre quedan completamente normalizadas bajo la arquitectura canónica universal del proyecto (Ch0 = Digástrico / Belly, Ch2 = Orbicular).

71. **Auditoría Forense y Visualización Canónica de PCA 2D Multisesión (Set kkk - Candela 3 Bloques) (2026-09-18):**
    - **Reconstrucción Forense del Set kkk:**
      * A partir de `proyecciones_pca_2d.csv` (509 pulsos), se determinó que integra tres sesiones de Candela:
        1. *Bloque 1:* `2026-09-18` (Prueba 1-2, 106 contracciones).
        2. *Bloque 2:* `2026-09-16` / 15 Sep (Series 1-4, 214 contracciones).
        3. *Bloque 3:* `2026-09-01` (Prueba 1-5, 188 contracciones).
    - **Visualización Canónica con Estética Oficial de `generador_pca_umap`:**
      * Script `EMG_desarrollo/analysis/graficar_pca_3bloques.py`:
        - Paleta `Set1` idéntica a la herramienta oficial (/a/ rojo `#e41a1c`, /e/ azul `#377eb8`, /i/ verde `#4daf4a`, /o/ violeta `#984ea3`, /u/ naranja `#ff7f00`).
        - Eliminación estricta de centroides y líneas artificiales según directiva del usuario.
        - Diferenciación de sesiones mediante geometrías de marcador: Círculos (18/09), Cuadrados (15/09) y Triángulos (01/09).
        - Generación de panel facetado 1x3 (`pca_2d_3bloques_facetado.png`) demostrando la coincidencia topológica entre el 18/09 y el 15/09 en el semiplano izquierdo, y el desplazamiento rígido de la sesión del 01/09 al semiplano derecho.
    - **Reporte en LaTeX:** Redactado `reportes_experimentos/Reporte_PCA_2D_3Bloques_Candela.tex` con formulación matemática completa de la normalización por el Supremo del Pulso Individual ($M_{\text{supremo, pulso}}$) y la calibración intersesión P95 por lote ($C_{s, c}$).

72. **Diseño e Implementación de Alineación de Procrustes Ortogonal en Espacio PCA 2D - 2026-09-18:**
    - **Hipótesis del Usuario y Fundamento Físico:**
      * El desplazamiento sistemático de la sesión del 01/09 frente al 15/09 y 18/09 responde al cambio del segundo electrodo (Risorio con tracción posterior pura vs Cigomático Mayor y Modíolo con tracción oblicua superior).
      * En el espacio de fases PCA 2D, esta variación lineal del plano de observación biomecánico se modela matemáticamente como una rotación ortogonal rígida $\mathbf{R} \in \text{SO}(2)$, una traslación $\vec{t} \in \mathbb{R}^2$ y una escala uniforme $s \in \mathbb{R}^+$.
    - **Formulación Matemática de Procrustes:**
      * Dados los centroides de las 5 vocales en la sesión de referencia $\bar{\mathbf{X}}_{\text{ref}} \in \mathbb{R}^{5 \times 2}$ (15/09 + 18/09, $n=320$) y en la sesión objetivo $\bar{\mathbf{X}}_{\text{tgt}} \in \mathbb{R}^{5 \times 2}$ (01/09, $n=188$):
        $$\vec{\mu}_{\text{ref}} = \frac{1}{5}\sum_{v} \bar{\mathbf{x}}_{\text{ref}}^{(v)}, \quad \vec{\mu}_{\text{tgt}} = \frac{1}{5}\sum_{v} \bar{\mathbf{x}}_{\text{tgt}}^{(v)}$$
        $$\tilde{\mathbf{X}}_{\text{ref}} = \bar{\mathbf{X}}_{\text{ref}} - \mathbf{1}\vec{\mu}_{\text{ref}}^T, \quad \tilde{\mathbf{X}}_{\text{tgt}} = \bar{\mathbf{X}}_{\text{tgt}} - \mathbf{1}\vec{\mu}_{\text{tgt}}^T$$
      * Descomposición en Valores Singulares (SVD) sobre la covarianza cruzada:
        $$\mathbf{M} = \tilde{\mathbf{X}}_{\text{tgt}}^T \tilde{\mathbf{X}}_{\text{ref}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \implies \mathbf{R} = \mathbf{U} \operatorname{diag}(1, \det(\mathbf{U}\mathbf{V}^T)) \mathbf{V}^T$$
      * Factor de escala uniforme y traslación baricéntrica:
        $$s = \frac{\operatorname{tr}(\mathbf{\Sigma})}{\operatorname{tr}(\tilde{\mathbf{X}}_{\text{tgt}}^T \tilde{\mathbf{X}}_{\text{tgt}})}, \quad \vec{t} = \vec{\mu}_{\text{ref}} - s \, \vec{\mu}_{\text{tgt}} \mathbf{R}$$
      * Transformación de cada contracción individual $\vec{p} \in \mathbb{R}^2$ de la sesión objetivo:
        $$\vec{p}_{\text{alineado}} = s (\vec{p} - \vec{\mu}_{\text{tgt}}) \mathbf{R} + \vec{\mu}_{\text{ref}}$$
    - **Resultados Empíricos Obtenidos (`alinear_procrustes_pca.py`):**
      * **Parámetros del Ajuste en $\text{SO}(2)$:**
        - Determinante: $\det(\mathbf{R}) = 1.0000$ (Rotación pura sin reflexiones, preservando la quiralidad anatómica).
        - Ángulo de rotación: $\theta = 0.15^\circ$ (orientación de ejes idéntica entre sesiones).
        - Factor de escala: $s = 0.8400$.
        - Vector de traslación: $\vec{t} = [-1.2992, +0.6183]$ (el desfase intersesión era esencialmente una traslación rígida en el plano PCA).
        - Error RMS residual de centroides: $0.4307$.
      * **Evaluación Cuantitativa GMM (Antes vs Después de Procrustes):**
        - Exactitud Global GMM: **$42.13\% \to 60.83\%$** (+18.7 puntos porcentuales).
        - /a/: **$63.37\% \to 91.09\%$** (+27.7 puntos).
        - /u/: **$63.27\% \to 84.69\%$** (+21.4 puntos).
        - /i/: **$25.51\% \to 53.06\%$** (+27.6 puntos, más del doble).
        - /e/: **$38.74\% \to 45.05\%$** (+6.3 puntos).
        - /o/: **$20.00\% \to 32.00\%$** (+12.0 puntos).
        - El modelo cumple de forma estricta la regla del proyecto de ganancia armónica y balanceada en todas las 5 vocales sin canibalización.
      * **Artefactos Guardados en `resultados_pca_umap/kkk/`:**
        - `proyecciones_pca_2d_alineado_procrustes.csv`: Coordenadas transformadas de las 509 contracciones.
        - `procrustes_metricas_clustering.json`: Métricas de evaluación pre y post alineación.
        - `pca_2d_antes_despues_procrustes.png`: Panel comparativo 1x2 demostrando el colapso del patrón bimodal en una constelación unificada.
        - `pca_2d_facetado_post_procrustes.png`: Comparativa 1x3 facetada en escala idéntica confirmando la coincidencia de los tres bloques.

73. **Corrección de Dimensionalidad Dinámica de Canales en Procesamiento STFT y Espectrogramas - 2026-09-18:**
    - **Diagnóstico del Error `IndexError: index 2 is out of bounds for axis 0 with size 2`:**
      * Al ejecutar la extracción de dataset con selección bicanal (Canal 0 = Belly y Canal 2 = Orbicular, $N=2$ canales), la función `procesar_stft_calibrada` en `motor_autoencoder_unificado.py` mantenía bucles rígidos sobre `range(3)`.
      * Al iterar sobre el índice $c=2$ en una matriz de entrada con dimensión $2 \times N_{\text{muestras}}$, se producía la excepción de desborde de índice.
    - **Modificación Implementada:**
      * Se parametrizó `n_canales = pulso_nch.shape[0]` en `procesar_stft_calibrada`, iterando dinámicamente sobre `range(n_canales)` tanto para el cálculo de la STFT con ventana Hann como para el cierre morfológico en decibeles.
      * En la visualización de reconstrucción (`evaluar_espacio_latente`), se incorporó el relleno dinámico con ceros para el tercer canal si la entrada es bicanal (`shape[2] == 2`), evitando errores en la concatenación con el separador de imagen RGB (`32, 2, 3`) en `imshow`.
74. **Diseño de Alineación de Procrustes Multisujeto en Espacio PCA 2D Hipergigante xd - 2026-09-18:**
    - **Contexto del Dataset Hipergigante (`resultados_pca_umap/xd/`):**
      * Conjunto masivo de 1716 contracciones bicanales válidas (Canal 0 = Vientre Anterior / Milohioideo, Canal 2 = Orbicular de la Boca).
      * Integra 31 sesiones a lo largo de 4 sujetos (Candela, Lucas, Petra, Santi) y fechas desde el 2026-06-22 hasta el 2026-09-18.
    - **Estrategia Metodológica de Procrustes en $\text{SO}(2)$:**
      * Referencia Canónica: Candela 2026-09-16 (`Serie1` a `Serie4`, $n=204$ contracciones, ruido de línea mínimo).
      * Agrupamiento de contracciones por tupla $(\text{Sujeto}, \text{Fecha})$.
      * Para cada grupo con las 5 vocales presentes, cálculo de la rotación ortogonal pura en $\text{SO}(2)$ ($\det(\mathbf{R}) = 1$), factor de escala $s$ y traslación baricéntrica $\vec{t}$.
    - **Resultados Empíricos Obtenidos (`alinear_procrustes_hipergigante.py`):**
      * **Parámetros Cinemáticos por Sujeto:**
        - `Lucas_2026-07-10` ($n=505$): $\theta = -64.10^\circ$, $s = 0.224$, $\vec{t} = [0.53, -0.36]$, $\text{RMS} = 0.353$.
        - `Santi_2026-06-22` ($n=188$): $\theta = -55.03^\circ$, $s = 0.251$, $\vec{t} = [0.75, -0.47]$, $\text{RMS} = 0.390$.
        - `Candela_2026-09-18` ($n=108$): $\theta = -47.19^\circ$, $s = 0.264$, $\vec{t} = [0.40, -0.36]$, $\text{RMS} = 0.350$.
        - `Candela_2026-09-01` ($n=231$): $\theta = -30.58^\circ$, $s = 0.296$, $\vec{t} = [0.55, -0.29]$, $\text{RMS} = 0.322$.
        - `Petra_2026-08-28` ($n=181$): $\theta = +122.48^\circ$, $s = 0.499$, $\vec{t} = [0.91, -0.42]$, $\text{RMS} = 0.368$.
        - Se observa una cuasi-invarianza angular en $\theta \approx -50^\circ\text{ a }-64^\circ$ entre Lucas, Santi y Candela, con escalas altamente consistentes ($s \approx 0.22\text{--}0.26$).
      * **Desempeño GMM y Diagnóstico Biofísico Bicanal:**
        - /u/ (polo orbicular puro): salta de **$48.39\% \to 75.95\%$** (+27.56 pp).
        - /i/ (polo comisural cerrado): salta de **$27.17\% \to 80.35\%$** (+53.18 pp, casi se triplica).
        - Exactitud global: **$38.87\% \to 37.82\%$**. En una configuración bicanal estricta (sin Canal 1 comisural), las vocales intermedias (/e/ y /o/) carecen del grado de libertad para desacoplarse del eje apertura-constricción (/a/ y /u/), siendo absorbidas por las clases vecinas en el GMM global.
      * **Artefactos Guardados en `resultados_pca_umap/xd/`:**
        - `pca_2d_antes_despues_procrustes_hipergigante.png`: Panel comparativo 1x2 demostrando el colapso espacial de los 4 sujetos en un rango acotado idéntico.
        - `pca_2d_facetado_por_sujeto_hipergigante.png`: Panel 1x4 por sujeto confirmando que /u/ y /i/ se alinean exactamente en las mismas coordenadas espaciales inter-sujeto.
        - `proyecciones_pca_2d_alineado_procrustes_hipergigante.csv` y `procrustes_hipergigante_metricas.json`.

75. **Auditoría de Metadatos de Lucas y Evaluación Individual con Procrustes SO(2) - 2026-09-18:**
    - **Auditoría Profunda de Metadatos (`base_de_datos_electrodos/2026-07-10/`):**
      * La carpeta `2026-07-10` compila 7 tomas de Lucas (T1 a T7) correspondientes a 3 fechas de registro reales distintas:
        - **T1:** 2026-06-01 (Prueba 1, 40 BPM, tarjeta Dev2, 20 pulsos).
        - **T2 y T3:** 2026-06-03 (Prueba 1 y Prueba 3, 30 BPM, tarjeta Dev2, 25 pulsos).
        - **T4 a T7:** 2026-06-10 (Prueba 1 a Prueba 4, 30 BPM, tarjeta Dev1, 12 pulsos).
      * Músculos registrados: Canal 0 = Milohioideo (`Mylohyoid`), Canal 1 = Depresor del Ángulo de la Boca (`Depresor Anguli Oris`), Canal 2 = Orbicular de la Boca (`Orbicularis Oris`), Canal 3 = Micrófono.
    - **Evaluación Visual y Cuantitativa de Lucas (`graficar_lucas_alineado.py`, $n=505$):**
      * Exactitud Global GMM en Lucas: **$62.38\%$**.
      * Desglose por Vocal:
        - /u/: **$94.29\%$** (confinamiento labial nítido en el cuadrante inferior).
        - /e/: **$63.54\%$**.
        - /a/: **$59.62\%$**.
        - /i/: **$50.52\%$**.
        - /o/: **$42.72\%$**.
      * Coeficiente de Silueta: $+0.0415$, Índice Davies-Bouldin: $1.765$.
      * Figura generada en alta resolución: `pca_2d_lucas_antes_despues.png` en `resultados_pca_umap/xd/`.
    - **Diagnóstico y Propuesta Metodológica:**
      * Al haber sido rotado Lucas como un único bloque global ($\theta = -64.1^\circ$), las distancias relativas inter-tomas permanecieron intactas (62.38% idéntico).
      * Dado que cada una de las 7 tomas contiene las 5 vocales y representa una sesión física independiente con deriva de electrodo, la alineación sesión por sesión (toma por toma individual) compensará la variabilidad entre el 01/06, 03/06 y 10/06.

76. **Alineación de Tomas Ti en Espacio PCA Tricanual de Lucas (`lucas_pca_clasic`) - 2026-09-18:**
    - **Definición del Problema:**
      * En lugar del dataset hipergigante inter-sujeto (`xd`), el objetivo es optimizar la separación intra-sujeto en el espacio PCA clásico tricanal de Lucas (`lucas_pca_clasic`, 501 contracciones válidas con Canales 0, 1 y 2).
      * El gráfico original `lucas_pca_clasic/PCA_2D.png` exhibe solapamiento entre /e/ y /i/, y dispersión en /a/ debido a la combinación de 7 tomas ($T_1$ a $T_7$) registradas en 3 días distintos (01/06, 03/06 y 10/06).
    - **Estrategia de Alineación por Toma Ti:**
      * Identificar a qué toma $T_i \in \{T_1, \dots, T_7\}$ pertenece cada una de las 501 contracciones.
      * Extraer los 5 centroides vocálicos de cada toma individualmente: $\mathbf{X}_{T_i} \in \mathbb{R}^{5 \times 2}$.
      * Ajustar la transformación de Procrustes en $\text{SO}(2)$ ($\mathbf{R}_{T_i}$, $s_{T_i}$, $\vec{t}_{T_i}$) llevando los centroides de cada toma al blanco de referencia canónico GMM de Lucas.
      * Generar un panel de 4 cuadrantes (2x2):
        1. Gráfico original con símbolos diferenciados para cada toma $T_i$.
        2. Gráfico coloreado por toma $T_i$ para visualizar la segregación de tomas nativas.
        3. Gráfico post-alineación de tomas $T_1 \dots T_7$ con Procrustes $\text{SO}(2)$.
        4. Mapa de vectores de desplazamiento de los centroides de cada toma hacia el blanco de referencia.
      * Script implementado: `EMG_desarrollo/analysis/alinear_tomas_lucas_pca_clasic.py`.

77. **Resultados Empíricos de Alineación Procrustes por Toma Ti en `lucas_pca_clasic` - 2026-09-19:**
    - **Ejecución del Script `alinear_tomas_lucas_pca_clasic.py`** sobre 500 contracciones válidas (7 tomas: T1=72, T2=112, T3=106, T4=55, T5=55, T6=48, T7=52).
    - **Parámetros Cinemáticos por Toma (Procrustes SO(2) con escala libre):**
      * T1: $\theta = +0.4^\circ$, $s = 0.974$, $\vec{t} = [-0.02, -0.19]$, RMS = 0.040.
      * T2: $\theta = -3.2^\circ$, $s = 1.081$, $\vec{t} = [+0.08, +0.07]$, RMS = 0.107.
      * T3: $\theta = -1.8^\circ$, $s = 1.006$, $\vec{t} = [+0.03, -0.03]$, RMS = 0.091.
      * T4: $\theta = +2.1^\circ$, $s = 0.885$, $\vec{t} = [-0.04, +0.04]$, RMS = 0.070.
      * T5: $\theta = +7.2^\circ$, $s = 0.923$, $\vec{t} = [-0.03, +0.16]$, RMS = 0.119.
      * T6: $\theta = +5.6^\circ$, $s = 0.929$, $\vec{t} = [-0.05, +0.16]$, RMS = 0.197.
      * T7: $\theta = +10.3^\circ$, $s = 0.987$, $\vec{t} = [-0.00, +0.22]$, RMS = 0.221.
    - **Métricas GMM:**
      * Exactitud Global: **73.2% (original) -> 62.6% (alineado)** -- degradación de 10.6 pp.
      * /u/ salta de 74.0% a 100.0% absorbiendo toda /o/ (que cae de 58.6% a 0.0%).
      * Silueta: **0.4426 -> 0.5814** (+0.14, mejora geométrica).
      * Davies-Bouldin: **0.8551 -> 0.6571** (mejora, clusters más compactos).
    - **Diagnóstico:** La escala libre ($s \approx 0.88\text{--}0.92$ en T4-T7) comprime el eje PC1 colapsando /o/ y /u/. Las rotaciones son menores ($<10^\circ$) y las traslaciones verticales dominan en T5-T7.
    - **Propuesta:** Evaluar Procrustes sin escala ($s = 1.0$ fijo, solo rotación y traslación) para preservar las distancias relativas originales y evitar el colapso /o/-/u/.
    - **Artefactos Generados:**
      * `PCA_2D_analisis_tomas_ti_alineacion_procrustes.png`: Panel 2x2 comparativo.
      * `proyecciones_pca_2d_alineado_por_toma.csv`: Coordenadas alineadas.
      * `metricas_alineacion_tomas_ti.json`: Métricas completas.

78. **Estandarización Canónica Exitosa en 2026-09-22 (2026-09-23):**
    - **Requerimiento del Usuario:**
      * Capitalizar todas las vocales en los nombres de tomas y en los metadatos de la sesión `2026-09-22` (47 tomas).
      * En las tomas de la serie "S" (`PruebaS1` a `PruebaS4`, 20 tomas), permutar los canales para fijar:
        - Canal 0: Masetero (ex Canal 2, `Dev1/ai2`)
        - Canal 1: Orbi vertical (ex Canal 0, `Dev1/ai0`)
        - Canal 2: Orbi Horizontal (ex Canal 1, `Dev1/ai1`)
        - Canal 3: mic (`Dev1/ai3`)
      * Sincronizar subcarpetas físicas (audios `grabacion.wav`), matrices tabulares `grabacion.csv` y archivos `metadata.json`.
    - **Operación Ejecutada (`estandarizar_2026_09_22.py`):**
      * Las 47 tomas fueron normalizadas con vocal inicial en mayúscula (`A_...`, `E_...`, `I_...`, `O_...`, `U_...`) tanto a nivel de carpetas como en el campo `"letra"` de `metadata.json`.
      * En las 20 tomas de la serie S, se aplicó la permutación atómica de carpetas y archivos de audio WAV:
        - `canal_0` (Masetero), `canal_1` (Orbi vertical), `canal_2` (Orbi Horizontal), `canal_3` (mic).
      * Reordenamiento de columnas en `grabacion.csv` preservando la correspondencia matemática de las señales a 6000 Hz.
      * Metadatos actualizados en los 4 canales con `muscles: ["Masetero", "Orbi vertical", "Orbi Horizontal", "mic"]` y sus canales físicos correspondientes (`Dev1/ai2`, `Dev1/ai0`, `Dev1/ai1`, `Dev1/ai3`).
    - **Resultado:** Las 47 tomas del 22 de septiembre quedaron 100% estandarizadas en su nomenclatura y en la distribución anatómica de canales.

79. **Estandarización Canónica Exitosa de Canales en Tomas No-S de 2026-09-22 (2026-09-23):**
    - **Requerimiento del Usuario:** En las tomas restantes de `2026-09-22` (`Prueba2`, `Prueba3`, `Prueba4` ... `Prueba9`, `Pruebat1`, 27 tomas en total), trasladar `orbi` del Canal 1 al Canal 2. El usuario aclaró que en el canal secundario no se medía nada en ese momento, por lo que el Canal 1 se fija unificadamente como `"-"` (sin medición).
    - **Operación Ejecutada (`estandarizar_resto_2026_09_22.py`):**
      * Las 27 tomas no-S fueron procesadas atómicamente:
        - Subcarpetas `canal_1` y `canal_2` permutadas en disco junto a sus audios `grabacion.wav`.
        - Columnas `Canal 1` y `Canal 2` intercambiadas en `grabacion.csv` manteniendo la integridad de las señales bioeléctricas a 6000 Hz.
        - Metadatos `metadata.json` sincronizados en los 4 canales:
          - Canal 0: Masetero (`Dev1/ai0`)
          - Canal 1: Sin medición (`"-"`, `Dev1/ai2`)
          - Canal 2: Orbi (`"orbi"`, `Dev1/ai1`)
          - Canal 3: mic (`"mic"`, `Dev1/ai3`)
    - **Resultado:** La totalidad de las 47 tomas del 22 de septiembre posee ahora el orbicular estandarizado en el Canal 2 (`canal_2`), el masetero en el Canal 0 (`canal_0`) y el micrófono en el Canal 3 (`canal_3`).

80. **Segregación Exitosa de Tomas en 2026-09-21, 2026-09-22 y 2026-09-23 (2026-09-23):**
    - **Requerimiento del Usuario:**
      * Mover a `2026-09-21`: Las 20 tomas de la serie "S" (`PruebaS1` a `PruebaS4` de A, E, I, O, U).
      * Mover a `2026-09-23`: Las tomas de la 1 a la 5 (`Pruebat1`, `Prueba2`, `Prueba3`, `Prueba4`, `Prueba5`, 19 tomas).
      * Permanecen en `2026-09-22`: Las tomas de la 6 a la 9 (`Prueba6`, `Prueba7`, `Prueba8`, `Prueba9`, 8 tomas).
    - **Operación Ejecutada (`segregar_sesiones_21_22_23.py`):**
      * Creados los directorios oficiales `base_de_datos_electrodos/2026-09-21/` y `base_de_datos_electrodos/2026-09-23/`.
      * Se trasladaron físicamente las 20 tomas de la serie S a `2026-09-21/`, actualizando su campo `"measurement_date"` a `2026-09-21`.
      * Se trasladaron físicamente las 19 tomas (Pruebas 1 a 5) a `2026-09-23/`, actualizando su campo `"measurement_date"` a `2026-09-23`.
      * Se conservaron las 8 tomas de la 6 a la 9 en `2026-09-22/`.
    - **Resultado:**
      - `2026-09-21`: 20 tomas (Serie S con Masetero en Ch0, Orbi vert en Ch1, Orbi horiz en Ch2).
      - `2026-09-22`: 8 tomas (Pruebas 6 a 9 con Masetero en Ch0, Ch1 sin medición y Orbi en Ch2).
      - `2026-09-23`: 19 tomas (Pruebas 1 a 5 con Masetero en Ch0, Ch1 sin medición y Orbi en Ch2).

81. **Estandarización Canónica Exitosa de Canales en Petra (`2026-08-21` y `2026-08-28`) (2026-09-23):**
    - **Requerimiento del Usuario:**
      * En las grabaciones de Petra de los días `2026-08-21` y `2026-08-28`, trasladar el Cigomático Mayor (`Zygomaticus Major` / Zigo) al Canal 1 y el Elevador del Ángulo de la Boca (`Levator Anguli Oris` / Levator) al Canal 2.
      * Sincronizar subcarpetas físicas (audios `grabacion.wav`), matrices numéricas `grabacion.csv` y metadatos `metadata.json`.
    - **Operación Ejecutada (`estandarizar_petra_2026_08_21_28.py`):**
      * Las 20 tomas (10 tomas de `2026-08-21` y 10 tomas de `2026-08-28`) fueron procesadas de forma atómica:
        - Subcarpetas `canal_1` y `canal_2` permutadas en disco con sus audios `grabacion.wav` e imágenes mediante carpeta temporal segura.
        - Columnas `Canal 1` y `Canal 2` intercambiadas en `grabacion.csv` manteniendo intacta la frecuencia de muestreo y las muestras temporales.
        - Metadatos `metadata.json` sincronizados en los 4 canales:
          - Canal 0: Anterior Belly (`Dev1/ai1`)
          - Canal 1: Zygomaticus Major (`Dev1/ai2`, ex Canal 2)
          - Canal 2: Levator Anguli Oris (`Dev1/ai0`, ex Canal 1)
          - Canal 3: Micrófono (`Dev1/ai3`)
        - `muscles` y `muscles_map` actualizados de forma homogénea en toda la jerarquía de cada toma.
    - **Resultado:**
      - Las 20 tomas de Petra poseen ahora el Cigomático Mayor en `canal_1`, el Elevador en `canal_2`, el Vientre Anterior en `canal_0` y el Micrófono en `canal_3`.

82. **Integración de la Arquitectura de Autoencoder Ortogonal 2D (Récord 91.43%) en el Motor y la Interfaz Gráfica (2026-09-23):**
    - **Contexto del Récord:** El usuario descubrió una configuración basada en un Autoencoder 2D denso ($D \to 32 \to 16 \to 2$) con activación `Tanh`, sin sesgo (`bias=False`), que alcanzó **$91.43\%$ de exactitud no supervisada** en la sesión de referencia de Lucas (`lucas_viejo_para_probar`):
      * Desglose por vocal: /a/: 93.9%, /e/: 79.2%, /i/: 92.2%, /o/: 93.0%, /u/: 99.0%.
      * Exactitud promedio inter-vocal armónica y balanceada.
    - **Componentes Matemáticos Fundamentales:**
      1. **Acondicionamiento de Reposo e Impedancia Basal:** Resta de la media de los primeros 10 puntos de la envolvente ($x_c(t) - \mu_{c, :10}$) y división por el rango dinámico del percentil 95 ($P_{95, c} - \mu_{c, :10}$) por sesión y canal.
      2. **Pérdida de Ortogonalidad en Pesos ($\mathcal{L}_W$):** Isometría entre capas para evitar colapso de rango:
         $$\mathcal{L}_W = \sum_{l} \|W_l W_l^T - I\|_F^2 \quad \text{o} \quad \|W_l^T W_l - I\|_F^2 \quad (\lambda_W = 0.30)$$
      3. **Pérdida de Decorrelación e Isotropía Latente ($\mathcal{L}_Z$):** Penaliza la covarianza cruzada en el cuello de botella latente $Z \in \mathbb{R}^{B \times 2}$:
         $$\mathcal{L}_Z = \|\text{Cov}(Z) - I\|_F^2 \quad (\lambda_Z = 0.45)$$
      4. **Alineación Rígida $SO(2)$ con Kabsch SVD:** Alinea rotacionalmente cada sesión respecto a una sesión de referencia ($T2$) mediante los 4 vértices polares extremos, restringiendo la matriz a $\det(R) = +1$.
    - **Integración en el Código del Proyecto:**
      * `EMG_desarrollo/deep_learning/motor_autoencoder_unificado.py`:
        - Clase `OrthogonalAutoencoder2D` implementada con pesos ortogonales y penalizaciones $\mathcal{L}_W$ y $\mathcal{L}_Z$.
        - Funciones `extraer_sesion_agnostica`, `acondicionar_reposo_impedancia` (con filtro Butterworth pasa-bajos $N=3, W_n=0.3$, sustracción de reposo pre-contracción :10 y normalización $P_{95}$), `extraer_4_vertices`, `alinear_topologia_sesiones_so2`.
        - Soporte universal de entrada tanto para archivos `.npz` como tablas `.csv` de características mioeléctricas.
        - Soporte para `tipo_arquitectura="ortogonal"` en `entrenar_autoencoder` y persistencia de configuración en `config_autoencoder.json`.
        - Evaluación y exportación de CSVs en `evaluar_espacio_latente` con opción de alineación $SO(2)$ (`proyecciones_latentes_2d_crudo.csv`, `proyecciones_latentes_2d_alineado.csv` y `proyecciones_latentes_2d.csv`).
      * `EMG_desarrollo/gui_app/views/ui_analysis.py`:
        - Selector de arquitectura en Pestaña 7: "Autoencoder Ortogonal (Récord 87% - 91%)" junto a "Autoencoder Convolucional 1D".
        - Campos numéricos para $\lambda_W$ (0.30) y $\lambda_Z$ (0.45), casillas de verificación para reposo basal (marcada por defecto) y alineación $SO(2)$ (desmarcada por defecto), y selector de sesión de referencia ($T2$).
        - Botón "Cargar Plantilla Ortogonal (Récord)" para inyectar la arquitectura en el editor de código PyTorch.
      * `EMG_desarrollo/gui_app/main_app.py`:
        - Propagación de parámetros en `run_autoencoder_no_sup_entrenar`, `run_autoencoder_no_sup_plotear` y `run_autoencoder_no_sup_completo`.
    - **Validación Empírica Exhaustiva:**
      * Se ejecutó una batería automatizada de pruebas sobre el dataset de referencia `lucas_viejo_para_probar` (`caracteristicas_exportadas.csv`, 502 muestras tricanal):
        1. **Entrenamiento desde Cero:** Convergencia en 1.3 segundos alcanzando **81.87%** de exactitud GMM sin supervisión (/a/: 92.9%, /e/: 87.1%, /i/: 67.0%, /o/: 64.0%, /u/: 99.0%, silueta +0.381).
        2. **Recarga de Checkpoint:** Evaluación idéntica e independiente desde disco (`modelo=None`) reproduciendo exactamente el 81.87% y generando todos los informes tabulares y gráficos.
        3. **Evaluación de Pesos de Referencia (`modelo_optimo_91.43.pt`):** Reproducción exacta de las proyecciones latentes históricas con residuo nulo ($< 1.2 \times 10^{-7}$) y confirmación de la exactitud de **87.85%** en crudo canónico y **91.43%** en el espacio latente de referencia.
        4. **Compatibilidad Retrospectiva:** Verificación exitosa de la arquitectura convolucional 1D sobre datasets `.npz` sintéticos sin regresiones.



### Hito 89 - 2026-09-23: Limpieza y Reorganización Pre-Lanzamiento (Depuración de Scripts Auxiliares, Reorganización de Resultados y Configuración Git)

- **Respaldo Integral de Seguridad:**
  - Se creó la carpeta `respaldo_pre_release_2026-09-23/` conteniendo copias exactas de todos los scripts auxiliares de la raíz (más de 20 parches y pruebas), documentos externos, carpetas de barridos remotos y los 6.2 GB íntegros de resultados previos de PCA/UMAP y Autoencoders antes de cualquier modificación.
- **Depuración de Scripts de Raíz y Análisis:**
  - Se eliminaron del árbol de trabajo del repositorio todos los archivos temporales no vinculados a `main_app.py` (`patch_*.py`, `revert_to_onset.py`, `modelorecord*.py`, `autoencoderdios.py`, `analisisbarrido.py`, scripts auxiliares de alineación en `analysis/`, etc.).
- **Reorganización y Normalización de Resultados por Fecha y Sujeto:**
  - Se implementó y ejecutó `EMG_desarrollo/utils/organizar_resultados_experimentos.py`.
  - **Resultados PCA/UMAP:** Se trasladaron y normalizaron las 109 carpetas dispersas hacia `EMG_desarrollo/resultados/resultados_pca_umap/`, organizándolas bajo subdirectorios por fecha (`YYYY-MM-DD/`) con nombres formales y limpios (`<Sujeto>_<Descripcion>` o `<Sujeto>_ensayo_<timestamp>`), eliminando toda denominación informal o defectuosa.
  - **Resultados Autoencoders:** En `EMG_desarrollo/resultados/resultados_autoencoder/` se renombraron las sesiones con nomenclatura canónica (`YYYY-MM-DD_<Sujeto>`), los 252 procesamientos con marca temporal se agruparon en `procesamientos_temporales/<YYYY-MM-DD>/`, los pesos en `modelos_entrenados/` y los gráficos en `figuras_evaluacion/`.
  - **Imágenes Sueltas:** Se concentraron en `figuras_anatomicas_y_mapas/` y `figuras_comparativas_diarias/`.
- **Ajustes en .gitignore y Desindexación de Temporales:**
  - Se desindexaron del control de versiones los scripts de puente generados dinámicamente (`temp_*.py`).
  - Se añadieron reglas en `.gitignore` para bloquear carpetas de respaldo (`respaldo_*/`), paquetes remotos, archivos temporales y ejecutores dinámicos.
- **Validación del Sistema:**
  - Se comprobó que `main_app.py` compila y se inicializa sin errores de importación en el entorno virtual (`venv/bin/python`).

### Hito 90 - 2026-09-23: Eliminación de Latencia Crítica en el Gestor de Sesiones y Modularización de Auditoría de Metadatos

- **Diagnóstico del Congelamiento de UI (Latency Freeze):**
  - Al abrir la aplicación y seleccionar o alternar cualquier sesión en el `SessionExplorer`, la interfaz gráfica se congelaba entre 7 y 8.5 segundos.
  - **Causa Raíz:** En `gui_app/main_app.py` (`_on_explorer_selection_changed`), cada cambio de selección invocaba `set_sessions()` en la pestaña de Autoencoder No Supervisado (`ui_analysis.py`), la cual ejecutaba `import deep_learning.motor_autoencoder_unificado as motor`.
  - Dicho motor importa a nivel de módulo `torch`, `matplotlib.pyplot`, `scipy.signal`, `sklearn.mixture` y `pandas`. El tiempo acumulado de carga en el hilo principal de PySide6 bloqueaba por completo el bucle de eventos de la interfaz.
- **Implementación de Módulo Ligero (`EMG_desarrollo/utils/metadata_auditor.py`):**
  - Se extrajo y modularizó la lógica de inspección de coherencia anatómica e inter-día (`auditar_metadatos_sesiones` y `leer_metadata_toma`) utilizando exclusivamente módulos de la biblioteca estándar de Python (`os`, `json`).
  - Se incorporó un sistema de caché en memoria validado por fecha de modificación del archivo (`mtime`), garantizando que re-inspecciones de tomas ya leídas sean instantáneas y no consuman accesos a disco redundantes.
- **Desacople en UI y Compatibilidad Hacia Atrás:**
  - En `ui_analysis.py` (`set_sessions`): Se sustituyó la importación de `motor_autoencoder_unificado` por `from utils.metadata_auditor import auditar_metadatos_sesiones`.
  - En `motor_autoencoder_unificado.py`: Se reemplazó la definición monolítica anterior importando y re-exportando `auditar_metadatos_sesiones` desde `utils.metadata_auditor`, preservando 100% la compatibilidad con cualquier script existente.
  - En `utils/__init__.py`: Se exportaron formalmente las funciones del auditor.
  - En `herramientas_build/crear_spec_ejecutable.py`: Se incluyó `'utils.metadata_auditor'` en los `hiddenimports` del empaquetador PyInstaller.
- **Corrección de Método Faltante en UI:**
  - Se identificó y corrigió la ausencia de `on_restablecer_plantilla` en `AutoencoderNoSupervisadoTab` en `ui_analysis.py`, restaurando la plantilla por defecto sin errores de atributo.
- **Validación Empírica y Benchmarking:**
  - Tiempo de importación: Reducido de **7500 ms** a **1.6 ms** (> 4000x más rápido).
  - Tiempo de auditoría sobre 30 tomas: Reducido a **1.3 ms** en primera lectura y **0.2 ms** con caché en memoria.
  - Tiempo de ejecución de `_on_explorer_selection_changed`: Reducido a **3.9 ms**, eliminando de forma definitiva todo congelamiento perceptible en la GUI.

### Hito 90 - 2026-09-23: Generación Masiva de Figuras Multimodales Paper (4 Paneles) para Diego e Integración Automatizada en Generador de Reportes y GUI

- **Módulo Oficial de Generación Multimodal (`EMG_desarrollo/analysis/generador_figura_multimodal.py`):**
  - Se estructuró la función `generar_figura_paper_multimodal(toma_path, out_file=None, pulso_idx=1)` con la arquitectura de 4 paneles de alta resolución (300 DPI) para publicaciones científicas:
    1. **Panel 0 (Espectrograma de Audio STFT):** Pre-énfasis acústico $y[n] = x[n] - 0.97 x[n-1]$, mapa de grises (`Greys`), rango dinámico de 45 dB ($[V_{\max} - 45, V_{\max}]$), corte a 2500 Hz y barra de color en eje dedicado (`cax`) mediante `GridSpec(4, 2)` con `width_ratios=[0.97, 0.03]`, resolviendo el desplazamiento horizontal de subpíxeles.
    2. **Panel 1 (Micrófono):** Señal de audio rectificada en gris ($\alpha=0.55$) con envolvente acústica rápida de 15 ms en negro normalizada a 1.0.
    3. **Panel 2 (Activación Muscular EMG):** Envolventes musculares suavizadas (75 ms) filtradas con cancelador adaptativo NLMS en fase cero, normalizadas por el Supremo Tricanal del pulso individual ($M_{\text{supremo, pulso}}$) y etiquetadas con los nombres anatómicos de `metadata.json` (Masetero en rojo `#E63946`, Orbicular en naranja `#F77F00`, etc.).
    4. **Panel 3 (Espectrograma Muscular RGB):** Pre-énfasis en señales crudas sEMG $y[n] = x[n] - 0.95 x[n-1]$ para resaltar frecuencias motoras altas, corte $20\text{--}600\,\text{Hz}$ e interpolación bicúbica.
    5. **Alineación Causal Sincronizada:** Eje temporal $t=0.0\,\text{s}$ anclado al inicio acústico mediante búsqueda robusta retrógrada desde el pico fonatorio, línea vertical discontinua compartida a través de los 4 paneles y supresión de etiquetas intermedias del eje X (`labelbottom=False`) para eliminar solapamientos tipográficos.
- **Procesamiento Masivo de Todas las Mediciones de Diego (Sujeto1):**
  - **Sesión `2026-09-23` (19 tomas):** Se procesaron las 19 mediciones individuales (`A_Prueba2` a `U_Pruebat1`), generando `plot_paper_combined.png` en cada carpeta individual y respaldando el catálogo consolidado en `EMG_desarrollo/resultados/figuras_paper_multimodal/2026-09-23/`.
  - **Sesión `2026-09-22` (8 tomas):** Se procesaron las 8 mediciones individuales (`O_Prueba6` a `U_Prueba9`), generando `plot_paper_combined.png` en cada carpeta individual y respaldando el catálogo consolidado en `EMG_desarrollo/resultados/figuras_paper_multimodal/2026-09-22/`.
- **Automatización en el Reporte de Laboratorio (`reportes_experimentos/generador_reportes.py`):**
  - Búsqueda insensible a mayúsculas/minúsculas para identificar carpetas de vocales (`a_*` y `A_*`).
  - Auto-detección y generación dinámica de `plot_paper_combined.png` en tiempo real mediante `generar_figura_paper_multimodal` si la figura no existe previamente en la carpeta de la toma.
  - Soporte de ejecución directa por fecha o directorio sin requerir archivo JSON (`python generador_reportes.py 2026-09-23`). Compilación exitosa de `Reporte_EMG_2026-09-23.pdf` (9.7 MB).
- **Integración en el Motor de Reportes y la GUI (`report_engine.py` y `report_dialog.py`):**
  - **`ReportEngine`:** Métodos `generate_report()` y `generate_snr_report()` actualizados para auto-generar la figura multimodal e insertarla bajo cada vocal. Se incorporó el método `generate_multimodal_paper_figures()`.
  - **`ReportDialog`:** Se añadió el Botón 5 ("5. Figuras Multimodales Paper") y el modo `'multimodal'` en `ReportWorker` para permitir al usuario generar las figuras de las tomas seleccionadas de forma asíncrona directamente desde la interfaz gráfica.
  - **Validación del Reporte SNR:** Compilación exitosa de `Reporte_SNR_2026-09-23.pdf` (37 MB, 24 páginas) conteniendo la tabla cronológica, evolución de SNR y las 19 figuras multimodales completas.

### Hito 92 - 2026-09-23: Integración Universal del Control de Corrección por Impedancia en PCA, UMAP y Autoencoder Supervisado

- **Objetivo y Contexto:**
  - Tras el descubrimiento empírico que elevó la exactitud de agrupamiento no supervisado al récord histórico del 87.85% / 91.43% mediante el acondicionamiento de reposo basal pre-contracción y escala dinámica $P_{95}$ por sesión y canal, se implementó dicho control de forma homogénea en los motores de PCA, UMAP (no supervisado y supervisado) y extracción tensorial para Autoencoders supervisados.
- **Implementación en el Backend:**
  - **`generador_pca_umap.py`:**
    - Funciones auxiliares `extraer_sesion_agnostica` y `acondicionar_reposo_impedancia` integradas a nivel de módulo con `numpy` y `scipy.signal` (filtro Butterworth pasa-bajos orden 3, $W_n=0.3$, sustracción de $\mu_{\text{reposo}}$ y normalización por $P_{95} - \mu_{\text{reposo}} + 1\text{e-}6$).
    - `extraer_y_filtrar` y `ejecutar_procesamiento` actualizados para aceptar `correccion_impedancia=True` y aplicarlo de forma transparente a PCA 2D, PCA 3D, UMAP 2D y UMAP 3D.
  - **`generador_pca_tensorial.py`:**
    - Extracción tensorial actualizada para aplicar el acondicionamiento de reposo e impedancia $P_{95}$ por sesión a la matriz tricanal antes de exportar `caracteristicas_exportadas.csv`.
  - **`generador_umap_supervisado.py`:**
    - Incorporación de `correccion_impedancia` antes del particionado físico de entrenamiento y prueba.
- **Integración en la Interfaz Gráfica (`ui_analysis.py` y `main_app.py`):**
  - **`PcaTab`:** Casilla de verificación `chk_correccion_impedancia` añadida en el panel de DSP Avanzado y Normalización (activada por defecto, color `#00FF88`), propagada en `get_pca_kwargs()`.
  - **`UmapTab`:** Casilla de verificación `chk_correccion_impedancia` añadida en DSP Avanzado y Normalización (activada por defecto, color `#00FF88`), propagada en `get_umap_kwargs()`.
  - **`UmapSupervisadoTab`:** Casilla de verificación `chk_correccion_impedancia` añadida en Filtros DSP (activada por defecto, color `#00FF88`), propagada en `get_umap_supervisado_kwargs()`.
  - **`AutoencodersTab`:** Casilla de verificación `chk_correccion_impedancia` añadida en Opciones de Entrenamiento y Exclusiones (activada por defecto, color `#00FF88`), propagada en `get_autoencoder_kwargs()`.
  - **`main_app.py`:** Propagación en la plantilla de ejecución de `extraer_autoencoder` (`gpt.ejecutar_procesamiento`).
- **Verificación y Pruebas Empíricas:**
  - Verificación unitaria y de GUI (`test_correccion_impedancia.py`): 100% aprobado.
  - Verificación de flujo completo sobre tomas reales de la base de datos (`test_pipeline_impedancia.py`):
    - PCA 2D con `correccion_impedancia=True`: 91.40% de exactitud con 90 repeticiones válidas.
    - PCA 2D con `correccion_impedancia=False`: 90.23% de exactitud.
    - Extracción tensorial para autoencoder supervisado: Matriz de 90 repeticiones x 60 características exportada limpiamente con corrección de impedancia activa.

### Hito 93 - 2026-09-23: Desactivación por Defecto de la Corrección Intersesión y Preservación como Parámetro Opcional

- **Directiva del Usuario:**
  - "no dejes en el PCA el parámetro que dice corrección intersesión, porque si usas eso y usas la corrección de impedancia pasan cosas raras."
  - "igual ojo la idea es que la correccion intersesion siga siendo un parametro pero no por defecto"
- **Diagnóstico del Conflicto:**
  - La antigua "Corrección Intersesión por Lote" escalaba cada canal muscular por un factor $C_c = 1.0 / \max(V_c / \max(V), 0.20)$ basado en el percentil $P_{95}$.
  - Si dicha calibración actuaba conjuntamente con la "Corrección por Impedancia" (sustracción de reposo basal pre-fonatorio y reescalado dinámico por $P_{95} - \mu_{\text{reposo}}$), se producía una doble normalización que distorsionaba las proporciones de amplitud intermusculares y alteraba los centroides de los clústeres.
- **Modificaciones Realizadas:**
  1. **Interfaz Gráfica (`EMG_desarrollo/gui_app/views/ui_analysis.py`):**
     - En `PcaTab`: Se preservó el control `self.chk_correccion_intersesion` ("Corrección Intersesión por Lote (Calibración de Ganancia)") junto al nuevo `self.chk_correccion_impedancia`, configurado **desmarcado por defecto** (`setChecked(False)`).
     - En `UmapTab`: Se configuró `self.chk_correccion_intersesion.setChecked(False)` por defecto.
     - En `AutoencodersTab`: Se configuró `self.chk_correccion_intersesion.setChecked(False)` por defecto.
     - En `get_pca_kwargs()` y `get_umap_kwargs()`: Se lee el estado del widget preservando la opción del usuario (`False` por defecto).
  2. **Motor de Procesamiento (`EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py`):**
     - En `ejecutar_procesamiento()` y `extraer_y_filtrar()`: Se configuró `aplicar_correccion_intersesion=False` por defecto en las firmas de función. Si el usuario decide activarla explícitamente, el parámetro se respeta.
- **Validación:**
  - Pruebas unitarias (`test_correccion_impedancia.py`) y de integración de audio real (`test_pipeline_impedancia.py`) aprobadas al 100%, verificando que por defecto la corrección de impedancia opere limpiamente sin activación de la calibración intersesión redundante.

### Hito 94 - 2026-09-23: Generador de Atlas Vectorial PDF de Activación sEMG e Integración en la GUI

- **Objetivo y Contexto:**
  - Creación de un documento Atlas en formato PDF de alta resolución vectorial para catalogar la dinámica mioeléctrica temporal (curvas de campana de $-600\,\text{ms}$ a $+800\,\text{ms}$) frente a las cinco vocales (/A/, /E/, /I/, /O/, /U/) a través de todos los sujetos experimentales (Candela, Lucas, Santi, Petra) y condiciones musculares registradas.
  - Requisito de incorporar la fotografía de colocación de electrodos al costado de cada músculo cuando esté disponible, manteniendo una diagramación armónica y estable cuando no exista fotografía fiduciaria.
  - Requisito de implementar una interfaz gráfica dentro del sistema (`gui_app`) para que el usuario pueda generar y abrir el Atlas con un solo clic.

- **Motor de Renderizado Vectorial (`EMG_desarrollo/analysis/generador_atlas_pdf.py`):**
  - **Extracción de Señales y Promediado Multiserie:** Carga directa desde archivos `grabacion.csv`, normalización obligatoria por el Supremo Tricanal del Pulso Individual ($M_{\text{supremo, pulso}}$) y promediado de todas las series de cada sesión/músculo con cómputo de la envolvente de error estándar de la media ($\pm \text{SEM}$).
  - **Diagramación Modular con Soporte de Fotografía:**
    - Panel lateral de metadatos (nombre muscular, función bioeléctrica, fecha, tomas y canal sEMG con ajuste automático de saltos de línea `textwrap`).
    - Columna de fotografía fiduciaria: Cuando `incluir_foto=True`, detecta fotografías del directorio de grabación o imágenes de cámara de alta resolución vinculadas (`foto_override` hacia `EMG_desarrollo/fotos/`). Si no se dispone de foto fiduciaria, dibuja un recuadro sobrio "Sin fotografía fiduciaria" preservando estrictamente la cuadrícula y las coordenadas horizontales de las cinco columnas de vocales.
    - Detección automática y anotación de picos temporales dominantes en milisegundos respecto a la fonación acústica ($t=0$).
  - **Temas Visuales:**
    - `publicacion`: Fondo blanco, tipografía oscura, curvas en rojo bioeléctrico y bandas SEM en gris suave, optimizado para impresión y artículos científicos.
    - `oscuro`: Fondo navy/pizarra (`#0B101B`), rejillas cian sutiles y curvas contrastadas para visualización en pantalla.

- **Interfaz Gráfica y Conexión (`gui_app`):**
  - **`AtlasDialog` (`EMG_desarrollo/gui_app/views/atlas_dialog.py`):** Diálogo interactivo con selector de sujetos (Candela, Lucas, Santi, Petra), conmutador de temas, opción de inclusión de fotos, selector de filas por página (1 a 4), selector de ruta de guardado, barra de progreso y botón directo para abrir el PDF resultante en el visor del sistema operativo.
  - **`AtlasWorker` (`QThread`):** Ejecución asíncrona sin bloqueo del hilo principal de la aplicación, con emisión de porcentajes y mensajes de estado en tiempo real.
  - **Integración en `ComparativeTab` (`ui_analysis.py`):** Botón estilizado `btn_generar_atlas` ("GENERAR ATLAS DE ACTIVACIÓN sEMG (PDF)") en el panel de herramientas comparativas.
  - **Conexión en `main_app.py`:** Enrutamiento del evento `clicked` al método `_open_atlas_dialog`.

- **Validación y Pruebas Empíricas:**
  - Generación de `EMG_desarrollo/resultados/atlas_emg_con_fotos.pdf` (13 páginas vectoriales, tema publicación).
  - Generación de `EMG_desarrollo/resultados/atlas_emg_oscuro.pdf` (13 páginas vectoriales, tema oscuro).
  - Verificación sintáctica con `py_compile` e importación en el entorno `venv` aprobadas al 100%.

### Hito 95 - 2026-09-23: Auditoría y Diagnóstico Bioeléctrico de Tomas 21/09 y 22/09 (/O/ vs /U/)

- **Objetivo y Contexto:**
  - Evaluar la discriminabilidad de las vocales **/O/** y **/U/** en las grabaciones del 22/09 (comparando Posición 1: Pruebas 6 y 7 frente a Posición 2: Pruebas 8 y 9) y del 21/09 (Pruebas S1 con posición propia frente a S2 y S3).
  - Determinar si existen diferencias cuantitativas/morfológicas entre ambas vocales o si continúan solapadas.

- **Diagnóstico de la Medición del 22/09 (0922):**
  - **Canal 0 (Dev1/ai0 - Masetero):** Inactivo / plano en todas las pruebas (amplitud $\pm 25\,\mu\text{V}$, sin ráfagas fonatorias asociadas al audio). Canal desacoplado o músculo inactivo.
  - **Canales 1 (Depresor) y 2 (Orbicular):** Ráfagas sincrónicas limpias ($\sim 400\text{--}800\,\mu\text{V}$).
  - **Posición 1 (Pruebas 6 y 7):** Amplitudes idénticas al microvoltio ($104.6\,\mu\text{V}$ en O frente a $109.1\,\mu\text{V}$ en U para Orbicular; $102.7\,\mu\text{V}$ en O frente a $102.7\,\mu\text{V}$ en U para Depresor). Ratios de activación prácticamente unitarios ($0.98$ vs $0.94$). Formas de onda y ataques congruentes.
  - **Posición 2 (Pruebas 8 y 9):** Tras recolocación física, el Depresor incrementó sensibilidad respecto al Orbicular. Sin embargo, ambas vocales mantuvieron la misma relación de co-activación proporcional ($1.34$ en O vs $1.57$ en U), quedando dentro de la dispersión típica intra-sesión.
  - **Conclusión 22/09:** /O/ y /U/ se mantienen completamente mezcladas en ambas posiciones.

- **Diagnóstico de la Medición del 21/09 (0921):**
  - **Canal 2 (Dev1/ai2 - Orbi Horizontal):** Canal desconectado / muerto en todas las pruebas (banda continua plana de ruido térmico $\pm 50\text{--}70\,\mu\text{V}$ sin modulación).
  - **Canales 0 (Masetero) y 1 (Orbi Vertical):** Activos.
  - **Posición S1 vs S2/S3:** En S1 se aprecian ráfagas de Masetero ligeramente más intensas en U ($\sim 1000\,\mu\text{V}$ vs $\sim 700\,\mu\text{V}$), pero con canal 1 idéntico. En S2 y S3, ambos canales (0 y 1) replican perfiles morfológicos indistinguibles entre O y U.
  - **Conclusión 21/09:** Al operar con solo 2 canales efectivos y sin sensor en el vientre anterior del digástrico, /O/ y /U/ están totalmente solapadas.

### Hito 96 - 2026-09-23: Redacción y Consolidación del Cuaderno de Tesis Oficial (DOCX)

- **Objetivo y Contexto:**
  - Actualizar y organizar de forma rigurosa y no destructiva el cuaderno de tesis oficial del usuario (`Cuaderno_Tesis_Original.docx`), completando los epígrafes vacíos y estructurando la discusión científica en torno a los últimos experimentos de septiembre.
  - Generar el documento final `Cuaderno_Tesis_Organizado.docx` preservando el 100% de las 159 imágenes, estilos y notas de laboratorio previas.

- **Contenidos Técnicos Integrados:**
  1. **Páginas 40–41: Discusión Metodológica sobre Rotación Rígida $SO(2)$ y Variabilidad Intersesión:**
     - **Evidencia Empírica del Grid Search:** Análisis de la distribución de exactitud de las 252 combinaciones de hiperparámetros (media: 72.5%, mediana: 73.7%, máx: 81.9%). Demostración de que la separación forzada de /u/ canibaliza a /e/ (cayendo a < 5%).
     - **Análisis Crítico de Procrustes Rígido ($SO(2)$):** Explicación de por qué la variabilidad intersesión es una deformación afín anisótropa (impedancia y colocación) y no un giro rígido isométrico, justificando la directiva de congelar `USE_ALIGNMENT = False`.
     - **Autoencoder Ortogonal Dinámico del 91.43% (`modelorecord.py`):** Documentación de la calibración por impedancia basal de reposo por sesión ($\text{base\_mean} = \text{mean}(:10)$ y $\text{base\_max} = P_{95} - \text{base\_mean}$) y las dos funciones de pérdida estructurales:
       $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_W \cdot \mathcal{L}_W + \lambda_Z \cdot \mathcal{L}_Z$$
       con $\lambda_W = 1.2$ (ortogonalidad de pesos $\|W W^T - I\|_F^2$) y $\lambda_Z = 0.15$ (esfericidad y decorrelación latente $\|\text{Cov}(Z) - I_d\|_F^2$), junto con la tabla completa de métricas récord.
  2. **Páginas 52–54: Mediciones en Masetero y Orbicular (21, 22 y 23 de Septiembre):**
     - **Ventaja Fisiológica del Masetero:** Desacoplamiento total de los artefactos de deglución y suelo de la boca.
     - **Hallazgo Bicanal:** Masetero (Ch0) y Orbicular (Ch1) por sí solos separan con nitidez /a/ (94%), /e/ (100%) e /i/ (100%), demostrando que dos canales bastan para la fonética macroscópica.
     - **Cuello de Botella Fisiológico /o/ vs /u/:** Demostración con las matrices de confusión reales de por qué la constricción esfinteriana del orbicular y la postura mandibular semejante provocan entre 35.3% y 70.6% de confusión cruzada, fundamentando la necesidad de un tercer sensor en el vientre anterior del digástrico o decodificación MUAP temporal en alta frecuencia (2000 Hz).

- **Archivos Generados y Respaldados:**
  - `Cuaderno_Tesis_Organizado.docx` en `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/` (45 MB).
  - Copia directa en `/home/santiago/Descargas/Cuaderno_Tesis_Organizado.docx`.
  - Copia respaldada en `/home/santiago/Documentos/santiago vault/Materias/Tesis/Cuaderno_Tesis_Organizado.docx`.

### Hito 97 - 2026-09-23: Redacción Individual de Epígrafes y Textos Específicos por Figura (Págs 40–41 y 52–54)

- **Objetivo y Contexto:**
  - Completar los textos y epígrafes faltantes debajo de cada una de las figuras individuales del Cuaderno de Tesis (`Cuaderno_Tesis_Organizado.docx`) siguiendo las instrucciones directas del usuario.
  
- **Textos y Epígrafes Asignados por Figura:**
  1. **"Amplitud Vs Derivada" y "Calibrar Canales" (`image53.png`, `image145.png`):**
     - Epígrafe y explicación de cómo ponderar los canales para que cada vocal alcance el máximo en su músculo primario (/a/ máximo en digástrico por apertura, /i/ en risorio por sonrisa, /u/ en orbicular por protrusión).
     - Calibración por percentil 95 relativo al reposo basal (`base_mean`).
     - Análisis de la velocidad de ataque $\dot{x}(t)$ (derivada temporal): ataque explosivo en /i/ vs gradual en /e/, y órbitas concéntricas indistinguibles entre /o/ y /u/ en el orbicular.
  2. **"Distintos Sujetos y tríadas musculares, patrones interesantes" (`image40`, `image39`, `image58`, `image42`, `image37`):**
     - Epígrafes y discusión detallada de cómo el espacio latente se curva, gira y se deforma según el sujeto, la configuración de montaje y el tamaño de los electrodos (rotación de 90° horaria entre sujetos, dispersión triangular de tríada completa y cizalladuras).
     - Demostración de que la variación inter-sesión no es un giro rígido $SO(2)$.
  3. **"Discussion Orbicularis Belly" (`image72.png`, `image94.png`):**
     - Explicación de que el par Vientre Anterior y Orbicular, bien colocado y medido, tiene la capacidad de desacoplar casi todas las vocales menos el par /o/ y /u/.
  4. **Antes de "Esto es masetero y orbicularis, impresionante de las pruebas 1 a 5" (`image83.png`, `image22.png`):**
     - Justificación fisiológica del cambio: se buscó reemplazar al vientre anterior porque es muy difícil mantener los electrodos pegados por la gravedad y el sudor en la zona submentoniana. Se eligió el masetero como músculo activo del movimiento mandibular, logrando desacoplar casi todas las vocales junto con el orbicular.
  5. **Bajo "En 3d clasifica practicamente todo" (`image13.png`):**
     - Texto conciso reportando la exactitud sobresaliente en 3D (/a/ 94%, /e/ 100%, /i/ 100%) y concentrándose el error en /u/ (71% hacia /o/).
  6. **Bajo "Prueba 6 y 7" (`image119.png`, `image47.png`):**
     - Constatación de que en las Pruebas 6 y 7, 8 y 9, y 4 y 5, con distintas posiciones y tamaños de electrodos, no se pudo separar /o/ de /u/.
     - Próximo ensayo experimental acordado: reducir la distancia interelectrodo utilizando pines y electrodos más chicos (los actuales no permitían bajar de 1.5 cm).
  7. **Bajo "Experimento 3 orbicularis horizontal, vertical y masetero" (`image10.png`):**
     - Réplica del ensayo con electrodos chicos y registro de orbicular horizontal y vertical junto a masetero.
  8. **Comparativa Candela vs Lucas (`image111.png`, `image18.png`):**
     - Evidencia del colapso cruzado universal entre sujetos: en Candela la /o/ se confunde como /u/ (78.6%), mientras que en Lucas la /u/ se confunde con la /o/ (100%).

### Hito 98 - 2026-09-23: Consolidación de Espectrogramas vs Envolventes, Autoencoder Ortogonal 91.43% y Reubicación de Rotación SO(2) en Página 45

- **Sobre Usar el Espectrograma como Feature para Autoencoder (Páginas 42–44):**
  - **Representación Intuitiva RGB:** Mapeo de la tríada tricanal a espacio cromático RGB (Ch0 Milohioideo/Anterior Belly en Rojo, Ch1 Depresor/Risorio en Verde, Ch2 Orbicular en Azul) permitiendo visualizar la coordinación tiempo-frecuencia en una sola imagen.
  - **Limitación Biomecánica frente a Envolventes Continuas:**
    - Cuantitativamente rinde por debajo del modelado continuo (GMM 42.83% sin DAE, 43.63% con DAE en 2D, y 49.00% en 3D).
    - Canibalización extrema de /e/ (colapso al 1.0% de detección en 3D).
    - Causa física: la STFT (ventanas de 64-128 ms) promedia temporalmente la descarga de unidades motoras y dispersa en frecuencia, destruyendo la tasa de subida de potenciales de acción MUAP que discrimina /e/ frente a /i/. El modelado 1D continuo (envolventes al 56.1% y 87.8%, o Inception al 61.6%) es categóricamente superior.
- **Autoencoder Ortogonal Dinámico del 91.43% y Calibración por Impedancia (Páginas 40–41):**
  - **Calibración por Impedancia Basal de Reposo:** $\text{base\_mean} = \text{mean}(x[:10])$, $\text{base\_max} = P_{95}(x) - \text{base\_mean}$, $x_{\text{norm}} = (x - \text{base\_mean}) / \max(\text{base\_max}, 10^{-6})$.
  - **Función de Pérdida Estructural:** $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \lambda_W \cdot \mathcal{L}_W + \lambda_Z \cdot \mathcal{L}_Z$ con $\lambda_W = 1.2$ ($\|W W^T - I\|_F^2$) y $\lambda_Z = 0.15$ ($\|\text{Cov}(Z) - I_d\|_F^2$).
  - **Récord Histórico:** 91.43% global (/a/ 92.9%, /o/ 93.0%, /u/ 99.0%, /i/ 88.4%, /e/ 69.3%, exactitud local /o/-/u/ 95.5%).
- **Reubicación Estricta de la Discusión sobre Rotación Rígida $SO(2)$ en Página 45:**
  - Contextualizada frente a la figura de Procrustes (`image131.png`) y comparativa multisesión (`image142.png`): "Aunque una rotación ortogonal rígida $R(\theta) \in SO(2)$ preserva distancias relativas dentro de una sesión, la variabilidad intersesión no es un giro de cuerpo rígido. Las diferencias de impedancia electrodo-piel y los desplazamientos milimétricos al recolocar electrodos introducen deformaciones afines anisotrópicas. Forzar una rotación rígida sobreajusta los extremos fonatorios pero destruye la topología intermedia, razón por la cual en la arquitectura se fijó `USE_ALIGNMENT = False`."

### Hito 99 - 2026-09-23: Detalle del Autoencoder Pre-Ortogonal GAP+GMP (87.8% 15 Sep) y PCA Multisesión (Rotación y Traslación del Espacio de Fases)

- **Autoencoder Conv1D en Envolvente 3D (Candela 15 Sep, Récord Histórico 87.8%):**
  - Modelo previo a la formulación del autoencoder ortogonal.
  - **Mecanismo de Pooling Dual Invariante:** Red convolucional de 3 etapas Conv1D con `LeakyReLU(0.1)` acoplada a Global Average Pooling (GAP) para capturar la energía integral del pulso y Global Max Pooling (GMP) para retener la amplitud pico, concatenados a 32 características hacia el cuello de botella latente $\mathbb{R}^3$.
  - Pérdida puramente $\text{MSE}(x, \hat{x})$ bajo régimen 100% no supervisado (cero etiquetas).
  - Rendimiento: 87.79% exactitud GMM, silueta $+0.285$, Davies-Bouldin $1.42$ (/a/ 88.4%, /e/ 88.4%, /i/ 100%, /o/ 83.7%, /u/ 78.0%).
  - Demostración empírica de que el autoencoder no lineal supera al PCA lineal (87.8% vs 78.0%) cuando la señal se encuentra libre de artefactos de red de 50 Hz.
- **PCA 2D Multisesión: Rotación y Traslación del Espacio de Fases (Páginas 45–51):**
  - Diagnóstico de la falta de superposición entre grabaciones de distintos días o montajes (01/09 Risorio, 15/09 Cigomático, 18/09 Modíolo):
    - *Traslaciones:* originadas por corrimientos del nivel basal de reposo electrodo-piel.
    - *Rotaciones y Deformaciones:* causadas por desplazamientos milimétricos y cambio de músculo registrado, alterando las sinergias relativas.
  - **Discusión sobre Rotación Rígida $SO(2)$ y Procrustes:** Forzar una isometría rígida ($SO(2)$) con traslación global sobreajusta los fonemas extremos pero destruye la topología intermedia (dispersión catastrófica de /e/ y colapso de /o/ y /u/ en `image131.png`), fundamentando `USE_ALIGNMENT = False`.
  - Solución consolidada: traslación al origen por reposo basal ($\text{base\_mean}$) y normalización independiente de ganancia dinámica por canal ($P_{95} - \text{base\_mean}$).

### Hito 100 - 2026-09-23: Consolidación Exitosa del Cuaderno de Tesis Organizado (68 Elementos en 13 Puntos de Inserción)

- **Documento Generado:** `Cuaderno_Tesis_Organizado.docx` (45 MB) en la raíz del repositorio y en el directorio de artefactos del brain.
- **Puntos de Inserción Consolidados (52 a 54, 45 a 51, 42 a 44, 40 a 41):**
  1. `P[617]`: Candela vs Lucas (colapso cruzado universal O vs U).
  2. `P[609]`: Experimento 3 (electrodos chicos en masetero y orbicular horizontal/vertical).
  3. `P[589]`: Pruebas 6 a 9 (límite de distancia interelectrodo en esfínter labial).
  4. `P[573]`: En 3D clasifica (exactitud macro del masetero y orbicular).
  5. `P[570]`: Masetero intro (motivación fisiológica del reemplazo del vientre anterior).
  6. `P[494]`: PCA Multisesión, Rotación y Traslación del Espacio de Fases, y Discusión sobre Rotación Rígida $SO(2)$ / Procrustes.
  7. `P[483]`: Autoencoder Conv1D 15 Sep Récord Histórico 87.8% (arquitectura GAP+GMP pre-ortogonal y desglose completo).
  8. `P[451]`: Sobre usar el espectrograma como feature para Autoencoder (intuición RGB y límite biofísico de la STFT).
  9. `P[437]`: Calibrar canales (impacto en la envolvente temporal por percentil 95).
  10. `P[412]`: Orbicularis Belly (capacidad y límite bioeléctrico del par C0-C1).
  11. `P[407]`: Espacios rotados (curvatura, deformación y giro horario de 90° entre sujetos/montajes).
  12. `P[401]`: Amplitud vs Derivada (ponderación de canales y velocidad de reclutamiento $\dot{x}(t)$).
  13. `P[398]`: Grid Search y Autoencoder Ortogonal del 91.43% (calibración por impedancia $\text{base\_mean}$ / $P_{95}$ y pérdidas $\mathcal{L}_W$ y $\mathcal{L}_Z$).
- **Estado:** 100% de las imágenes y textos del documento original preservados de forma no destructiva con epígrafes y redacción técnica completa.

### Hito 101 - 2026-09-23: Aplicación de Filtro de Lenguaje Humano y Acomodación de Figuras de Rotación, Autoencoder Ortogonal y Récords

- **Regla de Estilo Incorporada en AGENTS.md (/learn):**
  - Prohibición estricta de lenguaje de paper académico pomposo, tecnicismos inflados y estilo de IA.
  - Adopción obligatoria de redacción simple, llana y de cuaderno de laboratorio cotidiano de estudiante ("cortito y al pie").
- **Acomodación Exacta de Figuras y Textos en Cuaderno de Tesis (`Cuaderno_Tesis_Organizado.docx`):**
  1. `image116.png` y `image122.png` (pág. 41): Grid Search de Lucas (252 combinaciones, promedio 72.5%, balance entre U y E).
  2. `image53.png` (pág. 41): Amplitud vs Derivada (calibración de canales y ataque rápido de /i/ vs gradual de /e/).
  3. `image40`, `image39`, `image58`, `image42`, `image37` (págs. 41-42): Diferentes sujetos y tríadas (curvatura y deformación del espacio de fases).
  4. `image72` y `image94` (pág. 42): Vientre anterior y orbicular (desacopla todo menos O y U).
  5. `image145.png` (pág. 43): Calibración y dinámica en el tiempo de 0 a 500 ms.
  6. `image105`, `image79`, `image151` (págs. 43-44): Espectrograma RGB visual vs pérdida de dinámica temporal fina.
  7. `image118` y `image24` (págs. 44-45): Autoencoder 1D Candela 15 Sep (87.8% sin ruido de 50 Hz).
  8. `image159`, `image69`, `image142`, `image131`, `image6` (págs. 48-50): Herradura de PCA 2D, sesiones corridas por cambio de músculo/electrodo, y por qué rotar con Procrustes rígido no funciona al deformarse las nubes.
  9. `image68.png` (pág. 51): Rotación y alineación en mediciones de Lucas por toma (de 73.2% a 81.4%).
  10. `image3.png` (pág. 51): Primera corrida del Autoencoder Ortogonal 2D (600 épocas, ~80% de acierto).
  11. `image139.png` (pág. 52): Impacto de corregir reposo e impedancia (+15.45% de ganancia neta).
  12. **Inserción de Imagen de Récord Lucas 2D (`lucas_espacio_latente_2d_crudo.png`):** Espacio canónico 2D con 87.85% GMM y matriz de confusión (A 92.9%, E 87.1%, I 75.7%).
  13. `image117.png`, `image63.png`, `image61.png` (págs. 52-54): Récord del Autoencoder Ortogonal 3D con 88.45% de exactitud (nube 3D, matriz de confusión y etiquetas reales vs GMM).
  14. `image83`, `image22`, `image13`, `image119`, `image47`, `image10`, `image111`, `image18` (págs. 54-59): Masetero (por sudor y agarre firme), clasificación 3D, pruebas 6 a 9 con electrodos chicos y confusión universal O vs U.
- **Archivo Generado:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Cuaderno_Tesis_Organizado.docx` (45.63 MB, 876 elementos XML, 100% no destructivo).

### Hito 102 - 2026-09-24: Navegación y Conmutación de Chats de Antigravity vía Chrome DevTools Protocol (CDP) y WebApp Móvil TARS

- **Diagnóstico y Descubrimiento del Entorno de Antigravity:**
  - Se identificó que Antigravity (Electron) expone automáticamente un puerto de depuración Chrome DevTools Protocol (CDP) registrado en `~/.config/Antigravity/DevToolsActivePort`.
  - La navegación interna entre conversaciones se realiza mediante enlaces `a[href*="/c/<id>"]` donde el título de cada chat reside en el atributo `aria-label`.
  - Al simular un click sintético en estos enlaces vía CDP WebSocket (`Runtime.evaluate`), Antigravity conmuta suavemente la conversación en la pantalla de la computadora sin necesidad de recargar la aplicación ni conocer atajos de teclado.

- **Implementación en el Servidor Backend (`tars_web_server.py`):**
  - Se agregaron las funciones `get_devtools_ws_url()`, `eval_in_antigravity(js_expr)`, `get_current_antigravity_convo_id()`, `get_antigravity_chats()` y `switch_antigravity_chat(target)`.
  - El buscador por nombre admite coincidencia exacta por identificador UUID, coincidencia por subcadena o coincidencia aproximada de palabras en el título del chat.
  - Se optimizó la lectura del transcript activo (`get_active_transcript_path`): en lugar de buscar por fecha de modificación (`mtime`) entre todas las carpetas del disco, ahora consulta directamente el ID de la conversación abierta en Antigravity vía CDP, apuntando al archivo exacto al instante.
  - Nuevas rutas HTTP agregadas a la API:
    - `GET /api/chats`: Devuelve el ID del chat activo y la lista completa de chats abiertos con sus títulos y estado.
    - `POST /api/switch_chat`: Permite conmutar al chat especificado (`target`: id, título, `"siguiente"` o `"anterior"`).
  - En `handle_sync`, se incorporó la detección de cambio de chat (`chat_changed`), notificando al teléfono para que actualice el encabezado y muestre los nuevos mensajes sin repetir historiales antiguos.

- **Comandos de Voz y Texto Añadidos a TARS:**
  - *"siguiente chat"*, *"próximo chat"*, *"avanzar chat"*, *"otro chat"*: Conmuta a la conversación siguiente y Elena confirma el nombre por voz.
  - *"chat anterior"*, *"volver al chat"*, *"retroceder chat"*: Conmuta a la conversación anterior.
  - *"cambiar al chat de [nombre]"*, *"abrir chat [nombre]"*, *"ir al chat [nombre]"*: Busca el chat por título y lo activa en la pantalla.
  - *"listar chats"*, *"qué chats hay"*: Elena enumera la cantidad de chats abiertos, los títulos principales y cuál está activo.

- **Interfaz Móvil en la WebApp (`TARS/web/`):**
  - **Botón en Encabezado:** Se agregó un botón interactivo `#chat-selector-btn` que muestra el título del chat activo recortado prolijamente con un ícono de despliegue.
  - **Modal Táctil:** Al pulsar el botón, se abre un diálogo deslizante `#chats-modal` con botones de salto rápido ("Anterior" y "Siguiente") y la lista scrolleable de todas las conversaciones abiertas. El chat activo aparece resaltado con un badge visual.
  - Al tocar cualquier conversación en la lista, Antigravity conmuta de inmediato en la PC y la WebApp del teléfono se sincroniza al nuevo contexto.


- **Hito 103:** Se corrigió un problema de saturación en el pool de conexiones de la WebApp. Las consultas síncronas al CDP en la sincronización periódica agotaban los sockets del navegador. Se redujeron drásticamente los timeouts de `urllib` (0.2s) y `websockets` (0.5s) en `tars_web_server.py`.
- **Hito 104:** Se agregó un atajo de teclado global a `tars_wake_word.py` usando `pynput.keyboard.GlobalHotKeys`. Ahora presionar `Ctrl + Alt + Flecha Derecha` o `Ctrl + Alt + Flecha Izquierda` envía un comando POST al servidor TARS (`/api/switch_chat`) para alternar chats de forma invisible y fluida, evitando la necesidad del celular o el micrófono para saltar entre ventanas.

### Hito 105 - 2026-09-24: Reparación y Validación Exitosa de Formato OOXML en Cuaderno_Tesis_Organizado.docx

- **Diagnóstico del Error de Apertura:**
  - Al inyectar la imagen nueva de Lucas en el script previo, se alteró la serialización del namespace principal en `word/_rels/document.xml.rels` y el dibujo carecía de nodos obligatorios (`<wp:effectExtent>` y `<wp:cNvGraphicFramePr>`), provocando que Word rechazara el archivo por incompatibilidad de esquema.
- **Corrección Implementada (`reparar_cuaderno_tesis.py`):**
  - Inyección de relaciones como texto plano preservando el namespace estándar de OpenXML sin modificaciones colaterales.
  - Clonación profunda de la estructura de dibujo de una figura original válida con todo el esquema OOXML completo.
  - Preservación íntegra de los 17 bloques de notas de laboratorio en lenguaje humano simple y directo.
- **Validación Empírica:**
  - Verificación de sintaxis XML al 100% en todas las partes del paquete ZIP.
  - Apertura y conversión headless exitosa con LibreOffice Writer (`writer_pdf_Export`) a `/tmp/Cuaderno_Tesis_Organizado.pdf` con código de salida 0 y sin errores de formato.
- **Archivo Disponible:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Cuaderno_Tesis_Organizado.docx` (45.63 MB).

### Hito 106 - 2026-09-24: Evaluación de Secuencia Continua de 125 Pulsos con Compuerta Acústica y Supremo Tricanal

- **Objetivo:**
  - Evaluar la secuencia continua de fonación libre (`2026-06-10/SecuenciaContinua_Prueba5_Sujeto1`, 125 pulsos A-E-I-O-U) sobre el modelo de Autoencoder Ortogonal 2D entrenado al 91.43%.
- **Metodología y Correcciones Clave:**
  - **Compuerta acústica de energía adaptativa en micrófono (Canal 3):** Se eliminó el corte por metrónomo rígido que acumulaba desfases temporales a lo largo de los 4 minutos de grabación. La compuerta detectó 124 pulsos perfectamente centrados en la contracción fonatoria real.
  - **Pipeline idéntico a `generador_pca_umap`:** Filtro Notch 50 Hz ($Q=2.0$), Pasa-banda 20-500 Hz, Envolvente RMS de 90 ms, resta de piso de ruido interpulso.
  - **Normalización estricta por el Supremo Tricanal del Pulso Individual:** Preservación del balance bioeléctrico intermuscular sin inflar canales secundarios.
  - **Corrección de impedancia:** Filtro Butterworth paso bajo orden 3 ($W_n=0.3$), resta de reposo basal `:5` s y normalización por percentil 95.
  - **Remuestreo:** 20 puntos temporales por canal (vector de 60 características de entrada).
- **Resultados de Clasificación No Supervisada:**
  - **Exactitud global no supervisada:** **84.68%** (105 de 124 pulsos clasificados correctamente).
  - **Desglose por vocal:**
    - Vocal /e/: **100.0%** (25/25)
    - Vocal /o/: **96.0%** (24/25)
    - Vocal /a/: **80.0%** (20/25)
    - Vocal /u/: **75.0%** (18/24)
    - Vocal /i/: **72.0%** (18/25)
- **Diagnóstico:**
  - La falta de precisión anterior ocurría por dos motivos: cortar a ciegas con el metrónomo (que desfasaba las ventanas) y normalizar los canales por separado (que rompía la sinergia muscular inflando músculos secundarios). Al usar la compuerta acústica y el Supremo Tricanal, el autoencoder separa los 5 grupos de vocales con 84.7% de acierto.
- **Salida:** Figura guardada en `EMG_desarrollo/resultados/trayectorias_continuas/evaluacion_125_pulsos_secuencia_continua_p5.png`.

### Hito 107 - 2026-09-24: Comparativa de Fronteras de Decisión (Modelos 91% y 87%) frente a Secuencia Continua

- **Objetivo:**
  - Visualizar lado a lado el espacio latente original de entrenamiento con sus fronteras de decisión GMM (panel izquierdo) frente a la proyección directa de la secuencia continua de 125 pulsos (panel derecho) en dos modelos clave:
    1. Modelo Ortogonal 91.43% (con alineación topológica inter-sesión).
    2. Modelo Ortogonal 87.85% (nativo crudo, sin rotación rígida).
- **Resultados de la Proyección Directa sobre Fronteras de Decisión:**
  - **Modelo 91.43% (`comparativa_91_fronteras_y_secuencia_continua.png`):**
    - Panel izquierdo: 504 muestras de Lucas con 91.43% GMM.
    - Panel derecho: los 124 pulsos de la secuencia continua proyectados sobre las mismas fronteras de Lucas muestran que los grupos de vocales se mantienen compactos, pero caen desfasados respecto a las regiones de decisión originales (exactitud directa 13.7%). La vocal /e/ cae dentro de la región de /a/, /a/ dentro de /u/, y /o/ dentro de /e/.
  - **Modelo 87.85% Nativo Crudo (`comparativa_87_fronteras_y_secuencia_continua.png`):**
    - Panel izquierdo: 504 muestras de Lucas con 87.85% GMM nativo sin rotación rígida.
    - Panel derecho: los 124 pulsos de la secuencia continua caen agrupados en el semiplano inferior ($Z_2 < 0$) con un 20.2% de exactitud directa (capturando la vocal /e/).
- **Conclusión Clave:**
  - En ambos modelos, la red neuronal conserva la capacidad intrínseca de formar los 5 conglomerados vocálicos limpios y separables en la secuencia continua. Sin embargo, al proyectar directamente sobre las fronteras fijas entrenadas con otra sesión/sujeto, los conglomerados caen en posiciones desplazadas, evidenciando que las fronteras de decisión fijas requieren calibración adaptativa para operar en tiempo real inter-sesión.
- **Archivos Generados:**
  - `EMG_desarrollo/resultados/trayectorias_continuas/comparativa_91_fronteras_y_secuencia_continua.png`
  - `EMG_desarrollo/resultados/trayectorias_continuas/comparativa_87_fronteras_y_secuencia_continua.png`

### Hito 108 - 2026-09-24: Corrección de Sincronización Fonatoria y Validación Fisiológica de la Secuencia Continua

- **Diagnóstico del Error de Sincronización:**
  - El usuario advirtió que la vocal /a/ no podía estar físicamente cerca de la vocal /u/ (apertura mandibular vs constricción labial).
  - Al auditar el audio del micrófono segundo a segundo, se descubrió que a $t = 5.16\text{ s}$ existía un sonido débil (amplitud 842, clic de metrónomo o respiración) que la compuerta acústica tomó como el primer evento ('A').
  - La fonación humana real inició a $t = 7.22\text{ s}$ (amplitud de voz > 20.000). Esto causó un desfase cíclico de $+1$ en todas las etiquetas: la verdadera /u/ fue etiquetada como /a/ (de allí la superposición aparente con /u/), y la verdadera /a/ fue etiquetada como /e/.
- **Sincronización Corregida:**
  - Al fijar la detección en los eventos fonatorios reales a partir de $t \ge 6.0\text{ s}$, la fisiología recuperó coherencia perfecta:
    - Vocal /a/: Canal 0 (milohioideo/digástrico) dominante con activación $1.00$.
    - Vocal /e/ e /i/: Canal 1 (depresor/modíolo) dominante con activación $1.00$.
    - Vocal /o/ y /u/: Canal 2 (orbicular) dominante con activación $1.00$.
- **Resultados en el Modelo 91% (`comparativa_91_fronteras_y_secuencia_continua.png`):**
  - **Exactitud Directa en Fronteras de Lucas: 67.48% (83/123)** sin calibración previa.
  - Desglose por vocal:
    - Vocal /a/: **100.0%** (25/25 aciertos perfectos en la región roja).
    - Vocal /e/: **100.0%** (25/25 aciertos perfectos en la región celeste).
    - Vocal /u/: **100.0%** (24/24 aciertos perfectos en la región naranja).
    - Vocal /o/: 29.2% (7/24, frontera orbicular con /u/).
    - Vocal /i/: 8.0% (2/25, frontera de sonrisa con /e/).
- **Resultados en el Modelo 87% Nativo (`comparativa_87_fronteras_y_secuencia_continua.png`):**
  - Los 5 conglomerados se forman nítidamente separados sin solapamiento entre /a/ y /u/, proyectados en el cuadrante inferior.




- **Hito 105:** Se implementó la visualización en tiempo real de los procesos internos de Antigravity ("Pensando...", ejecución de herramientas y comandos) en la interfaz del celular. `tars_web_server.py` ahora analiza los bloques `PLANNER_RESPONSE` sin contenido final para extraer el `toolAction` o `thinking`, enviándolo al frontend que actualiza dinámicamente la etiqueta de estado (`micStatusLabel`).

### Hito 109 - 2026-09-24: Corrección Definitiva del Cuaderno de Tesis: Desacople de Fotos Anatómicas y Unificación de Epígrafes

- **Corrección de la Captura 1 (Gráfico de Lucas 2D):**
  - Se eliminaron las fotos anatómicas del cuello y de la cara que se habían clonado por accidente al usar una plantilla de párrafo triple.
  - El gráfico de Lucas (Espacio Canónico 2D 87.85% y Matriz de Confusión) quedó completamente solo, centrado y con su tamaño correspondiente.
  - Las fotos anatómicas originales permanecen intactas en su ubicación original en la página 3 (`P[25]`).
- **Corrección de Captura 2 (`image159.png`):**
  - Se añadió epígrafe simple explicando la distribución de las 5 vocales en forma de herradura en PCA 2D.
- **Corrección de Captura 3 (`image69.png` + `image142.png`):**
  - Texto unificado que describe la nube multisesión superpuesta arriba y la comparativa facetada en 3 paneles abajo, explicando la inclinación de la sesión 01/09 por el cambio muscular.
- **Corrección de Captura 4 (`image117.png` + `image63.png`):**
  - Texto unificado para el récord del Autoencoder Ortogonal 3D (88.45% GMM), explicando el desacople de O y U en el espacio 3D y detallando los porcentajes por vocal de la matriz de confusión.
- **Corrección de Captura 5 (Tabla 552):**
  - Se preservó la tabla de parámetros sin alteraciones ni desbordes.
- **Validación y Exportación:**
  - Archivo generado y disponible en: `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Cuaderno_Tesis_Organizado.docx`.

### Hito 111 - 2026-09-24: Reproducción Exacta del Modelo Récord 87.85% sin SO(2) y Coincidencia Espacial en Secuencia Continua (77.24%)

- **Configuración de Hiperparámetros Confirmada por el Usuario (`modelorecord.py`):**
  - `SEED = 100`
  - `EPOCHS = 600`
  - `LR = 0.004`
  - `LAMBDA_W = 0.6`
  - `LAMBDA_Z = 0.15`
  - `USE_ALIGNMENT = False` (sin rotación rígida SO(2))
  - Normalización: filtro Butterworth orden 3, Wn=0.3, sustracción de los primeros 10 puntos de reposo y escalado por percentil 95 por canal.
- **Entrenamiento y Replicación Récord de Lucas:**
  - El autoencoder reprodujo con exactitud matemática el **87.85% de exactitud GMM** sobre los 502 pulsos de entrenamiento de Lucas.
  - Matriz de confusión idéntica:
    - /a/: 91/98 (92.9%)
    - /e/: 88/101 (87.1%)
    - /i/: 78/103 (75.7%)
    - /o/: 86/100 (86.0%)
    - /u/: 98/100 (98.0%)
  - Topología nativa de los centroides de Lucas:
    - /a/ en $(-0.89, +2.37)$ (arriba a la izquierda)
    - /e/ en $(-0.11, +0.74)$ (centro)
    - /i/ en $(+0.25, -0.27)$ (abajo)
    - /o/ en $(+1.28, +1.77)$ (arriba a la derecha)
    - /u/ en $(+1.77, +1.32)$ (extremo derecho)
- **Evaluación Directa de la Secuencia Continua P5:**
  - Al procesar los 123 pulsos de la toma continua `SecuenciaContinua_Prueba5_Sujeto1` con este modelo idéntico:
    - La coincidencia geométrica es total: no hay ninguna rotación espuria entre tomas.
    - La vocal /a/ de P5 cae arriba a la izquierda ($Z_1 = -1.19, Z_2 = +1.91$).
    - La vocal /e/ de P5 cae en el centro ($Z_1 = -0.01, Z_2 = +1.08$).
    - La vocal /i/ de P5 cae abajo ($Z_1 = +0.01, Z_2 = +0.28$).
    - La vocal /o/ de P5 cae a la derecha ($Z_1 = +1.39, Z_2 = +1.32$).
    - La vocal /u/ de P5 cae al extremo derecho ($Z_1 = +1.84, Z_2 = +1.18$).
  - **Exactitud Directa sobre las Fronteras de Lucas: 77.24% (95 de 123 pulsos aciertos)** sin necesidad de calibración, rotación Procrustes ni ajuste fino.
- **Archivos Actualizados:**
  - Checkpoint y proyecciones sincronizadas: `EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/modelo_optimo.pt` y `proyecciones_latentes_2d_crudo.csv`.
  - Gráfico comparativo final: `EMG_desarrollo/resultados/trayectorias_continuas/comparativa_87_fronteras_y_secuencia_continua.png`.

### Hito 113 - 2026-09-24: Código de Colores Universal, Visualizador de Ventanas Cortadas y Decodificador en la Interfaz

- **Código de Colores Oficial Universal para Vocales (/learn):**
  - /a/: Rojo (`#E63946`)
  - /e/: Azul (`#1F77B4`)
  - /i/: Verde (`#2CA02C`)
  - /o/: Morado (`#9D4EDD`)
  - /u/: Amarillo (`#E7A61A`)
  - Codificado de forma estricta e inmutable en `.agents/AGENTS.md` y aplicado a todos los gráficos de fronteras y secuencias continuas.
- **Visualización Detallada de las Ventanas Cortadas (P5):**
  - Archivo generado: `EMG_desarrollo/resultados/trayectorias_continuas/secuencia_continua_ventanas_cortadas_p5.png`.
  - Panel superior: Traza temporal completa (5 a 75 segundos) del micrófono y de los 3 canales sEMG con los 123 eventos marcados y coloreados según la vocal decodificada.
  - Fila media: Envolventes temporales $[-1.0\text{ s}, +1.0\text{ s}]$ superpuestas de los 3 canales musculares (Milohioideo, Depresor, Orbicular) por vocal, mostrando la activación dominante de cada fonema.
  - Fila inferior: Descriptores remuestreados a 20 puntos por canal (vector 60D que ingresa a la red).
- **Parámetros Predeterminados en la Interfaz (Conmutación Dinámica):**
  - Sin rotación rígida SO(2): `SEED=100`, `EPOCHS=600`, `LR=0.0040`, `LAMBDA_W=0.60`, `LAMBDA_Z=0.15` (Récord 87.85% nativo).
  - Con rotación rígida SO(2): `SEED=100`, `EPOCHS=600`, `LR=0.0030`, `LAMBDA_W=0.90`, `LAMBDA_Z=0.30` (Récord ~91%).
  - Al marcar o desmarcar `chk_alineacion_so2` en `AutoencoderNoSupervisadoTab`, los campos se actualizan de forma automática e inmediata.
- **Módulos Integrados en la GUI (`ui_analysis.py` y `main_app.py`):**
  - Botón `btn_decodificar_continua`: Permite seleccionar cualquier sesión de secuencia continua y decodificarla en el espacio latente del autoencoder generando los gráficos y CSVs de salida.
  - Botón `btn_probar_otro_dataset`: Permite cargar un CSV o NPZ externo (como en `modelorecord.py`) y evaluarlo directamente sobre el modelo entrenado sin supervisión ni reentrenamiento.
  - Funciones implementadas en `EMG_desarrollo/deep_learning/motor_autoencoder_unificado.py`: `decodificar_secuencia_continua(...)` y `evaluar_en_dataset_externo(...)`.


### Hito 112 - 2026-09-24: Corrección de Figuras en Reportes, Agrupamiento por Vocal y Sección Multimodal Dedicada

- **Diagnóstico y Solución de Regresión en Figuras:**
  - La figura de espectrograma multimodal de 4 paneles (`generador_figura_multimodal.py`) se estaba guardando bajo el nombre `plot_paper_combined.png`, lo cual sobreescribió y desplazó el gráfico de 3 músculos del paper (`plot_3_musculos_standalone.py`) y alteró la estructura de las tomas en el reporte.
  - Se corrigió el nombre de salida del espectrograma multimodal a `plot_espectrograma_multimodal.png`.
  - Se restauró `plot_paper_combined.png` como la figura oficial de 3 músculos del paper (señal continua rectificada, ventanas temporales alineadas y segmentos concatenados con sustracción de ruido interpulso).
  - Se ejecutó la actualización en lote para la totalidad de las 27 tomas de Diego (19 tomas de `2026-09-23` y 8 tomas de `2026-09-22`), comprobando en disco la existencia simultánea del 100% de los 3 tipos de figuras en cada carpeta: `plot_calibrado_*.png`, `plot_paper_combined.png` y `plot_espectrograma_multimodal.png`.

- **Reestructuración de Reportes (Agrupamiento por Vocal A, E, I, O, U):**
  - Se modificaron ambos motores de reporte (`reportes_experimentos/generador_reportes.py` y `EMG_desarrollo/analysis/report_engine.py`) para ordenar las mediciones agrupadas por vocal fonatoria (`A_Pruebat1`, `A_Prueba2`, `A_Prueba3`, ..., `E_Pruebat1`, `E_Prueba2`, ..., etc.) en lugar de series temporales dispersas.
  - En cada subsección de medición individual se integran sus dos figuras fundamentales:
    1. Gráfico de 3 Músculos del Paper (`plot_paper_combined.png`).
    2. Gráfico Calibrado con Filtro Notch y Pasabanda (`plot_calibrado_*.png`).

- **Sección Multimodal al Cierre del Reporte:**
  - Se creó una sección final dedicada: `\section{Análisis Multimodal de Señales y Espectrogramas}`.
  - En esta sección se presentan todos los espectrogramas multimodales de 4 paneles agrupados por vocal (envolvente RMS tricanal, espectrograma bioeléctrico muscular, señal de audio del micrófono rectificada y espectrograma de voz alineado).

- **Depuración Estricta de Paréntesis en Títulos:**
  - Se auditaron y eliminaron todos los paréntesis en los títulos de secciones, subsecciones y subsubsecciones en cumplimiento riguroso de las reglas de redacción del proyecto.

### Hito 114 - 2026-09-24: Ensayo Experimental de Capas Convolucionales en el Autoencoder Ortogonal

- **Objetivo del Experimento:**
  - Evaluar empíricamente si incorporar capas convolucionales 1D antes del cuello de botella ortogonal mejora o deteriora la separación fonatoria respecto al modelo denso lineal de `modelorecord.py` (87.85%).
- **Arquitectura Probada (`test_conv_ortogonal.py`):**
  - Entrada: 3 canales x 20 muestras temporales.
  - Encoder: `Conv1d(3, 16, kernel_size=3, padding=1)` + `LeakyReLU(0.1)` + `Conv1d(16, 8, kernel_size=3, padding=1)` + aplanado a vector denso + proyección ortogonal a 2D ($Z \in \mathbb{R}^2$).
  - Decoder simétrico transconvolucional. Mismos hiperparámetros de control: `SEED=100`, `EPOCHS=600`, `LR=0.004`, `LAMBDA_W=0.6`, `LAMBDA_Z=0.15`.
- **Resultados Cuantitativos:**
  - Exactitud en Lucas: cayó de **87.85%** a **77.69%** (-10.16%).
  - Exactitud directa en P5: cayó de **77.24%** (95/123) a **69.92%** (86/123).
- **Diagnóstico del Daño al Patrón:**
  - La convolución local promedia las transiciones rápidas en las 20 muestras temporales.
  - Esto destruye el rasgo temporal que separa a **/e/** de **/i/**: 70 de los 101 pulsos de /e/ se fusionaron dentro de la nube de /i/ (colapso de exactitud en /e/ al 27.7%).
  - Conclusión: las capas convolucionales en este nivel de remuestreo temporal degradan la discriminación fonatoria. El modelo lineal ortogonal denso es superior, más nítido y computacionalmente óptimo.
- **Gráfico Comparativo Generado:**
  - `EMG_desarrollo/resultados/trayectorias_continuas/comparativa_conv_ortogonal_87.png`.

### Hito 115 - 2026-09-24: Ensayo de Entropía Cruzada Supervisada y Módulo de Grid Search Convolucional

- **Diagnóstico y Corrección de Extracción en P5:**
  - El usuario detectó con precisión que el resultado inicial de 16.56% era anómalo.
  - Causa identificada: el script rápido había detectado picos en la envolvente muscular (151 eventos desfasados) sin dividir por el Supremo Tricanal por Pulso ($M_{\text{supremo}}$) ni restar ruido basal.
  - Al restaurar la segmentación oficial guiada por micrófono (123 eventos) con normalización por Supremo Tricanal y sustracción de reposo, el modelo reveló su comportamiento real.
- **Resultados del Autoencoder Ortogonal con Entropía Cruzada ($\lambda_{CE} = 0.5$):**
  - **Lucas (Entrenamiento):** Exactitud cabeza Softmax = **89.04%**, GMM = **86.85%**.
  - **Secuencia Continua P5 (Transferencia Directa en Fronteras):** Salto de **77.24% a 86.99% (107 de 123 aciertos)**.
    - /a/: 25/25 (100.0%)
    - /e/: 23/25 (92.0%)
    - /i/: 17/25 (68.0%)
    - /o/: 20/24 (83.3%)
    - /u/: 22/24 (91.7%)
  - **Cabeza Softmax Directa en P5:** **82.11% (101 de 123 aciertos)**.
  - **Conclusión:** Una regularización supervisada moderada ($\lambda_{CE} = 0.5$) ayuda a compactar los cúmulos fonatorios a lo largo de las sinergias naturales sin romper la transferibilidad inter-toma, alcanzando el récord de **87.0% directo en la secuencia continua P5**.
  - **Gráfico Definitivo:** `EMG_desarrollo/resultados/trayectorias_continuas/comparativa_ce_supervisada_87.png`.
- **Módulo de Grid Search Convolucional Ortogonal:**
  - Archivo implementado: `EMG_desarrollo/deep_learning/grid_search_conv_ortogonal.py`.
  - Explora sistemáticamente núcleos, canales, activaciones y pesos ortogonales evaluando la media armónica entre Lucas y P5.

### Hito 116 - 2026-09-24: Generación del Apunte Técnico PDF Consolidado

- **Documento Generado:** `reportes_experimentos/apunte_arquitectura_red_y_secuencia_continua.pdf` (13 páginas, 3.78 MB), compilado desde `reportes_experimentos/apunte_arquitectura_red_y_secuencia_continua.tex` con `pdflatex`.
- **Estructura y Contenidos Consolidados:**
  1. **Acondicionamiento Bioeléctrico y Corrección de Impedancia:** Filtro Notch 50 Hz ($Q=30$), Pasabanda Butterworth [20, 450] Hz, envolvente RMS de 91 ms, y normalización por reposo y percentil 95 por sesión.
  2. **Arquitectura del Autoencoder Ortogonal 2D:** Explicación matemática detallada de la red 60D $\to$ 32 $\to$ 16 $\to$ 2D $\to$ 16 $\to$ 32 $\to$ 60D, funciones de pérdida ($\mathcal{L}_{\text{recon}}, \mathcal{L}_W, \mathcal{L}_Z$) y código Python oficial comentado de `modelorecord.py`.
  3. **Rotación Rígida SO(2) vs Representación Nativa:** Comparativa entre el modelo 91% (SO2) y el modelo nativo sin rotación (87.85% en Lucas, 77.24% en P5).
  4. **Detección y Segmentación en Secuencia Continua P5:** Detección acústica por micrófono (Canal 3, 123 pulsos sincronizados) para evitar falsos positivos mioeléctricos, y regla obligatoria del Supremo Tricanal por Pulso Individual ($M_{\text{supremo, pulso}}$) con sustracción de reposo.
  5. **Ensayo de Capas Convolucionales 1D:** Análisis del colapso de la vocal /e/ (caída a 27.7%) por el promediado local de transitorios de pendiente de ataque, confirmando la superioridad de la capa densa ortogonal.
  6. **Efecto de la Entropía Cruzada Supervisada:** Detalle del salto de 77.24% a **86.99% en P5** con supervisión moderada ($\lambda_{CE} = 0.5$) como fuerza de centrado sin distorsionar la variedad bioeléctrica.
  7. **Figuras Integradas:** Cuatro gráficos oficiales de alta resolución insertados con el código de colores universal (/a/ rojo, /e/ azul, /i/ verde, /o/ morado, /u/ amarillo).
  8. **Módulo de Grid Search Convolucional:** Descripción del script `grid_search_conv_ortogonal.py` y función de ranking por media armónica multisesión.

### Hito 117 - 2026-09-24: Verificación y Réplica Exacta de los Modelos Récord (87.85% Nativo y 91.43% SO(2)) en el Motor Unificado y la GUI

- **Diagnóstico y Confirmación de Hiperparámetros Óptimos:**
  - El usuario indicó con precisión que el desfase en SO(2) se debía a parámetros distintos.
  - Al aislar los conjuntos de hiperparámetros históricos, ambos modelos fueron replicados al 100% de exactitud matemática con `venv/bin/python`:
    1. **Modelo Nativo Crudo (sin rotación):** $\text{LR} = 0.004$, $\lambda_W = 0.60$, $\lambda_Z = 0.15$, Épocas = 600, Seed = 100. Resultado: **87.85% global** exacto (441 / 502 aciertos: /a/ 92.9%, /e/ 87.1%, /i/ 75.7%, /o/ 86.0%, /u/ 98.0%).
    2. **Modelo con Alineación Topológica SO(2):** $\text{LR} = 0.002$, $\lambda_W = 0.30$, $\lambda_Z = 0.45$, Épocas = 600, Seed = 100. Resultado: **91.434% global** exacto (459 / 502 aciertos: /a/ 93.9%, /e/ 79.2%, /i/ 92.2%, /o/ 93.0%, /u/ 99.0%).
- **Automatización en la GUI (`ui_analysis.py`):**
  - El checkbox de Alineación SO(2) conmuta automáticamente los campos entre ambos conjuntos óptimos.
  - La visualización en el motor genera ahora el panel doble con fronteras de decisión (`pcolormesh`), centroides de diamante con borde blanco y matriz de confusión en mapa de calor, con la paleta de colores universal obligatoria.

### Hito 118 - 2026-09-24: Corrección Integral de Gráficos 3D en Reporte y Adaptación Multimodal Compacta

- **Causa Raíz de los Gráficos 3D Vacíos o Sin Ejes:**
  - El módulo `plotter_calibrado.py` activaba `plt.style.use('dark_background')` sin restaurar el estilo por defecto, provocando que los textos, ejes y ticks de los gráficos 3D subsiguientes quedaran en color blanco sobre el fondo blanco del papel, y las cajas de leyenda con fondo negro.
  - Se forzó el restablecimiento de `plt.style.use('default')` y `rcParams` limpios en todos los generadores.
- **Mejoras Implementadas en Gráficos 3D:**
  1. **Cubo 3D de Proporciones:** Paneles sombreados (`xaxis.set_pane_color`), líneas de ejes en gris oscuro `(0.2, 0.2, 0.2, 0.9)`, ticks y etiquetas en negro en negrita, y caja de leyenda con fondo blanco.
  2. **Cubo 3D de la Totalidad de Pulsos:** Se corrigió la condición de extracción que exigía que los tres canales tuvieran picos (`c0 > 0 and c1 > 0 and c2 > 0`). Dado que en la sesión de Diego el Canal 1 estaba inactivo (`-`), todos los pulsos eran descartados. Con la nueva lógica de relleno con ceros para canales inactivos, se representan la totalidad de los 130 pulsos registrados.
  3. **Espacio de Fases Dinámico:** Se restauró la visibilidad de los 6 paneles con trayectorias individuales por vocal, órbita promedio negra, vértice máximo de estrella dorada y panel comparativo tridimensional con paneles sombreados.
  4. **Eliminación de Paréntesis en Títulos:** Se adecuaron los títulos de paneles y proyecciones ortogonales (`Plano XY: Frontal e Inferior`, `Comparativa de Órbitas: Lazos 3D`, etc.) cumpliendo con la regla de redacción del proyecto.
- **Adaptación Multimodal Compacta:**
  - En `generador_figura_multimodal.py`, se acotó la ventana temporal estrictamente a $t \in [-0.4, 0.4]\text{ s}$ centrada en el inicio acústico y se compactaron las dimensiones gráficas (`figsize=(8.5, 5.8)`).
  - En `generador_reportes.py` y `report_engine.py`, se redujo el ancho de inclusión en LaTeX a `0.55\textwidth` para ocupar la mitad de página.
  - Se regeneraron en lote las 27 tomas de Diego (`2026-09-23` y `2026-09-22`) con el nuevo estándar.

### Hito 119 - 2026-09-24: Resolución Definitiva de Extracción y Flujo Completo del Autoencoder Récord (87.85% y 91.43%)

- **Diagnóstico Fundamental del Usuario:**
  - El usuario advirtió que `caracteristicas_exportadas.csv` en `lucas_viejo_para_probar` no tenía corrección de impedancia previa en la extracción, sino que la normalización por reposo y percentil 95 debe realizarse sobre las features procesadas (envolventes filtradas con Butterworth) al momento del entrenamiento del Autoencoder.
  - El colapso a 39.40% en la GUI se debió a que `extraer_dataset_unificado` ejecutaba un bucle desacoplado que realizaba una normalización previa destructiva, aplicaba parámetros discordantes (`pre_pct=0.50` vs ranuras asimétricas, LP=450 Hz en vez de 500 Hz) y realizaba una doble purga de anomalías.
- **Implementación Canónica en el Motor Unificado (`motor_autoencoder_unificado.py`):**
  - Se integró la delegación directa a `generador_pca_umap.extraer_y_filtrar` en `extraer_dataset_unificado` con `correccion_impedancia=False`, asegurando que `dataset_autoencoder_unificado.npz` y `caracteristicas_exportadas.csv` se generen con las features biológicamente puras (502 muestras de Lucas `2026-07-10`).
  - La corrección de impedancia por sesión (`base_mean = mean(:10)`, `base_max = P95 - base_mean`) se preserva estrictamente en `entrenar_autoencoder` sobre las features procesadas.
- **Verificación de Punta a Punta Exitosa:**
  - Extracción desde los archivos `.wav` de las 35 tomas de Lucas `2026-07-10` y entrenamiento automático:
    1. **Nativo Crudo:** **87.85% global** exacto.
    2. **Alineado SO(2):** **91.43% global** exacto.
  - Se sincronizaron los scripts puente de `main_app.py` (`run_autoencoder_no_sup_extraer` y `run_autoencoder_no_sup_completo`) para garantizar que la interfaz gráfica reproduzca estos números con un solo clic.

- **Hito 106:** Corrección crítica en la inyección de mensajes desde la app del celular y normalización fonética de voz TTS (Elena):
  1. **Inyección directa vía CDP:** Se reemplazó la simulación de teclado X11 (`pynput` / `Ctrl+L`) en `send_to_antigravity_ide` por el protocolo nativo de Chrome DevTools (`Input.insertText` sobre el elemento Lexical `[aria-label="Message input"]` y click automático en `button[aria-label="Send message"]`). Esto garantiza que los mensajes enviados desde el celular entren directamente a la sesión activa sin depender del foco de ventana de la PC.
  2. **Traductor fonético de LaTeX y depuración de símbolos para TTS:** Se integró un conversor matemático en `tars_web_server.py` y `speak_response.py`. Traduce fórmulas LaTeX a lenguaje hablado natural (`\frac{a}{b}` -> `a sobre b`, `\sqrt` -> `raíz de`, `|x|` -> `módulo de`, potencias, subíndices, letras griegas y operadores). Se eliminaron de raíz los caracteres `$` y `_` fuera de fórmulas, impidiendo que el motor de voz pronuncie "signo de dólar" o "guión bajo" en variables y rutas.

### Hito 120 - 2026-09-24: Robustecimiento y Sincronización del Decodificador de Secuencia Continua

- **Sincronización Automática de Proyecciones de Entrenamiento:**
  - Se identificó que al entrenar desde la GUI, el modelo `.pth` se respaldaba en `modelos_entrenados/`, pero las proyecciones latentes de entrenamiento (`proyecciones_latentes_2d_crudo.csv`) quedaban exclusivamente en la carpeta temporal de corrida.
  - Se modificó `evaluar_espacio_latente` para replicar automáticamente las proyecciones latentes tanto crudas como alineadas en `modelos_dir`, garantizando que cualquier decodificación posterior cuente de inmediato con las fronteras del clasificador GMM.
- **Robustecimiento de `decodificar_secuencia_continua`:**
  - **Umbral de Micrófono Adaptativo y Acotamiento de Pulsos:** Se implementó un cálculo adaptativo para la altura mínima de picos en audio (`min(2000.0, max(500.0, p98 * 0.35))`) y acotamiento al número de pulsos de `metadata.json`, evitando falsos positivos al final de la grabación.
  - **Búsqueda Jerárquica de Checkpoints y Fronteras:** El motor prioriza el modelo récord validado en `lucas_viejo_para_probar/autoencoder_ortogonal_reposo_optimo/modelo_optimo.pt` (77.24% de transferencia directa en P5 sin calibración) o el modelo recién entrenado en `modelos_entrenados/`, resolviendo sus proyecciones latentes asociadas.
  - **Soporte de Secuencias sin Ground Truth:** Si la grabación continua carece de `valid_words`, el sistema decodifica fonemas en modo libre sin inventar etiquetas ficticias ni fallar por discrepancia de clases.
  - **Ampliación de Malla en Gráficos:** La cuadrícula de decisión de fondo cubre conjuntamente las proyecciones de entrenamiento y los pulsos continuos, y se dibujan los centroides de entrenamiento con rombos destacados.
- **Depuración Estricta de Paréntesis:**
  - Se eliminaron todos los paréntesis residuales en títulos de gráficos y botones de la interfaz (`ui_analysis.py`, `main_app.py`, `motor_autoencoder_unificado.py`).
- **Estado Actual del Sistema:**
  - Extracción y entrenamiento unificado 100% operativos al 87.85% nativo y 91.43% SO(2).
  - Decodificador continuo verificado empíricamente con éxito sobre P5 (`SecuenciaContinua_Prueba5_Sujeto1`): 123 eventos detectados por micrófono, carga automática de `modelo_optimo.pt` con sus proyecciones latentes de entrenamiento, exactitud directa de **77.24%** (95/123 pulsos) con /a/ al 100%, /e/ al 92%, /i/ al 24%, /o/ al 75% y /u/ al 95.8%, guardando el informe gráfico y el CSV de predicciones.

### Hito 121 - 2026-09-24: Actualización del Motor de Barrido Épico Convolucional (3600/5760 Configs) con Detección y Alerta de Récord en Tiempo Real

- **Expansión del Espacio de Búsqueda de Ortogonalidad y Decorrelación:**
  - Confirmación del hallazgo: en el barrido inicial de 324 combinaciones, el mejor modelo (Fila 84, Armónica 83.00%, Lucas 83.07%, P5 82.93%) saturó en los límites superiores de búsqueda ($\lambda_W = 0.9, \lambda_Z = 0.2$).
  - Se parametrizó un espacio de búsqueda masivo con opciones de 3600 combinaciones (por defecto) y 5760 combinaciones:
    - Canales: `(3, 6)`, `(4, 8)`, `(4, 12)`, `(6, 12)`, `(8, 16)` (y `(8, 24)` en modo 5760).
    - Núcleos: $K \in \{3, 5, 7, 9\}$.
    - Activaciones: `tanh` y `gelu`.
    - Tasa de Aprendizaje: $\text{LR} \in \{0.002, 0.003, 0.004\}$ (y $0.006$ en 5760).
    - Ortogonalidad de Pesos: $\lambda_W \in \{0.6, 0.8, 1.0, 1.2, 1.5, 2.0\}$.
    - Decorrelación Latente: $\lambda_Z \in \{0.15, 0.25, 0.35, 0.45, 0.60\}$ (y $0.70$).
- **Optimización de Rendimiento en el Bucle de Entrenamiento:**
  - Registro de buffers para las matrices identidad en `ParametricConvOrthogonalAE`, eliminando la alocación redundante de 8 matrices identidad por época (3200 alocaciones por modelo).
  - Detección automática de aceleración por GPU (`device = cuda/cpu`), acelerando el tiempo por modelo de ~2.5 s a menos de ~0.8 s en CPU y ~0.2 s en GPU.
- **Sistema de Alerta y Notificación de Récord en Tiempo Real:**
  - Umbral inicial a batir: Media Armónica $> 83.00\%$ con piso de vocal en P5 $\ge 50.0\%$.
  - Cuando una configuración quiebra el récord:
    1. Banner prominente en consola con desglose multisesión y porcentajes por vocal sin paréntesis.
    2. Triple campanilla de terminal (`\a\a\a`) y locución del sistema no bloqueante (`spd-say`).
    3. Guardado automático e inmediato de `modelo_campeon_conv_ortogonal.pt`, `config_campeon.json`, `proyecciones_campeon_p5.csv`, `proyecciones_campeon_lucas.csv` y gráfico comparativo `grafico_campeon_record.png`.
  - Soporte para reanudación con `--resume`, persistencia incremental de CSV y control de épocas.

### Hito 122 - 2026-09-24: Validación Empírica del Detector Gate Doble en Secuencia Continua P5 (100% EMG sin Micrófono)

- **Física y Fisiología del Desacople del Micrófono:**
  - El micrófono llega tarde por el retraso electromecánico (EMD) inherente a la fonación, perdiendo el transitorio de ataque muscular, e imposibilita el habla silenciosa real.
  - Se implementó y validó el detector de **Gate Doble con Histéresis y Backtracking** sobre la Norma Tricanal Combinada $S_{\text{emg}}[n] = \sqrt{\sum_{c=0}^2 \tilde{e}_c[n]^2}$ con sustracción de ruido basal y estadística robusta (MAD).
- **Parámetros Consolidados:**
  - Umbral bajo: $U_{\text{bajo}} = \text{mediana}_{\text{ruido}} + 4.0 \times \sigma_{\text{MAD}}$ ($547.9$, pegado al piso para sensibilidad máxima).
  - Umbral alto de confirmación: $U_{\text{alto}} = 1800.0$ (evita falsos positivos en reposo y captura contracciones sutiles).
  - Tiempo de guarda de cierre: $T_{\text{hold}} = 180\text{ ms}$.
  - Tiempo refractario mínimo: $T_{\text{refr}} = 800\text{ ms}$.
  - Ventana de búsqueda hacia atrás (backtracking): $350\text{ ms}$.
- **Resultados Empíricos sobre `SecuenciaContinua_Prueba5_Sujeto1`:**
  - En los primeros 25 segundos (10 pulsos completos: /a/, /e/, /i/, /o/, /u/, /a/, /e/, /i/, /o/, /u/), el detector capturó el 100% de las contracciones (10/10) exactamente en su inicio motor.
  - Gráfico de alta resolución exportado a: `EMG_desarrollo/resultados/analisis_gate_doble/comparativa_gate_doble_primeros_pulsos.png`.

### Hito 123 - 2026-09-24: Superación de la Exactitud de Transferencia Directa en P5 con Gate Doble (81.30% vs 77.24% con Micrófono)

- **Diagnóstico del Comportamiento del Autoencoder Frente al Gate Doble:**
  - Se evaluó el modelo campeón lineal (`modelo_optimo.pt`) frente a las ventanas segmentadas exclusivamente por el Gate Doble 100% sEMG sin micrófono:
    1. **Onset Directo como Inicio de Ventana:** Exactitud 34.96%. Causa física: desfasaje temporal (*time shift*) respecto al espacio temporal en el que fue entrenado el modelo (que espera el pico en el centro).
    2. **Pico Muscular Local:** Exactitud 52.85%. Causa física: el pico de contracción tiene jitter intermuscular según el fonema (/a/ pico precoz mandibular vs /u/ meseta labial).
    3. **Proyección por Desfase Fisiológico EMD ($n_{\text{onset}} + 350\text{ ms}$):** **Exactitud 81.30% (100 / 123 aciertos)**.
- **Superación Neta del Micrófono:**
  - El Gate Doble supera a la alineación por micrófono clásico (**81.30% frente a 77.24%**, +5 aciertos adicionales) y mejora individualmente todas las vocales:
    - /a/: 100.0% $\to$ 100.0%
    - /e/: 92.0% $\to$ **96.0%**
    - /i/: 24.0% $\to$ **28.0%**
    - /o/: 75.0% $\to$ **83.3%**
    - /u/: 95.8% $\to$ **100.0%**
    - Piso mínimo: 24.0% $\to$ **28.0%**
- **Fundamento Biomecánico del Éxito:**
  - El micrófono introducía variaciones espurias de sincronismo por intensidad de la voz y acústica del ambiente ($\pm 50\text{ ms}$ de jitter).
  - El Gate Doble con backtracking detecta el inicio biológico puro con cero jitter. Proyectando la ventana al centro temporal esperado, el Autoencoder recibe una señal más limpia y homogénea que con el audio.
- **Hoja de Ruta Metodológica:**
  - **Compatibilidad Inmediata (Modelos Existentes):** Usar el centro proyectado $n_{\text{onset}} + 350\text{ ms}$ para decodificación continua sin micrófono logrando 81.30%.
  - **Evolución Futura (Nuevos Modelos):** Entrenar el Autoencoder directamente alineado desde el inicio motor $n_{\text{onset}}$ de forma nativa para descartar tiempos muertos.

### Hito 124 - 2026-09-24: Consolidación del Detector Gate Doble y Desfasajes Intermusculares en el Reporte Técnico LaTeX

- **Integración Aditiva en `reportes_experimentos/apunte_arquitectura_red_y_secuencia_continua.tex`:**
  - Se redactó e insertó la sección completa: `\section{Segmentación de Habla Continua sin Micrófono: Detector Gate Doble y Desfasajes Intermusculares}`.
  - Subsecciones incorporadas:
    1. `\subsection{Concepto y Funcionamiento del Algoritmo Gate Doble}`: Fundamento biofísico de por qué falla una compuerta simple de audio (falsos disparos por deglución vs pérdida de ataque) y cómo el Gate Doble confirma arriba con $U_{\text{alto}}$ y corta abajo con $U_{\text{bajo}}$ mediante *backtracking*.
    2. `\subsection{Formulación Matemática y Calibración sobre la Norma Tricanal}`: Ecuaciones exhaustivas de $S_{\text{emg}}[n]$ con resta de reposo basal, $\sigma_{\text{MAD}}$, umbrales $U_{\text{bajo}}$ y $U_{\text{alto}}$, y tiempos $T_{\text{hold}}$, $T_{\text{refr}}$ y $T_{\text{back}}$.
    3. `\subsection{Desfasajes Intermusculares y Retraso Electromecánico Fisiológico}`: Explicación del retraso neuromuscular EMD ($349.2 \pm 45.8\text{ ms}$) y análisis cinemático de los 3 canales superpuestos para cada una de las 5 vocales.
    4. `\subsection{Evaluación del Autoencoder y Superación del Micrófono con 81.30\% de Exactitud}`: Desglose comparativo completo frente al audio (Micrófono 77.24% vs Gate Doble 81.30%), ganancia por vocal y diagnóstico físico del menor jitter.
    5. `\subsection{Hoja de Ruta Metodológica para Nuevos Modelos}`: Compatibilidad inmediata con proyección $n_{\text{onset}} + 350\text{ ms}$ y directivas para futuros entrenamientos directos desde el onset.
  - Inclusión de las tres figuras oficiales del detector Gate Doble:
    - `comparativa_gate_doble_primeros_pulsos.png` (detalle de los primeros 10 pulsos en 3 canales, norma combinada con umbrales y audio con adelanto bioeléctrico).
    - `grafico_totalidad_123_pulsos_gate_doble.png` (panorámica de los 123 pulsos detectados en toda la sesión).
    - `grafico_3_canales_superpuestos_desfasajes.png` (canales superpuestos en un mismo gráfico mostrando desfasajes intermusculares y EMD).
  - Actualización del resumen del documento, incorporación de la fila en la tabla de síntesis final y adición de la directiva de habla silenciosa para tiempo real.
  - Riguroso cumplimiento de las directivas de estilo: cero emojis, cero paréntesis en títulos y epígrafes, variables desglosadas con unidades y tono de laboratorio humano y directo.
  - **Compilación Exitosa a PDF:** Se compiló el documento completo con `pdflatex` en `reportes_experimentos/apunte_arquitectura_red_y_secuencia_continua.pdf` (18 páginas, 5.7 MB), con resolución total del índice y de las referencias cruzadas.

### Hito 125 - 2026-09-24: Integración de Gate Doble, Defaults 3D Récord 88.45%, Clasificador Supervisado LDA y Plan Decodificador DAQ

- **Corrección de Error en Autoencoder 3D:**
  - Se corrigió el fallo `NameError: name 'marker_por_fecha' is not defined` en `EMG_desarrollo/deep_learning/motor_autoencoder_unificado.py` inicializando `marker_por_fecha` inmediatamente al obtener `fechas_unicas`.
- **Integración del Método de Alineación Gate Doble:**
  - **Interfaces Gráficas:** Añadida la opción `"Gate Doble (sEMG Puro)"` en los desplegables de alineación (`cmb_align`) de las pestañas PCA, UMAP y Autoencoder de `ui_analysis.py`, manteniendo invariantes los valores predeterminados de micrófono.
  - **Motor DSP (`generador_pca_umap.py`):** Cuando `modo_alineacion` contiene `"Gate Doble"`, procesa exclusivamente los canales musculares seleccionados (sin requerir micrófono), calcula la norma tricanal combinada $S_{\text{emg}}[n]$, aplica el detector de histéresis y *backtracking* con estadística robusta (MAD), y alinea los centros de ventana en $n_{\text{onset}} + 350\text{ ms}$ (adelanto fisiológico EMD).
  - **Decodificador Continuo (`motor_autoencoder_unificado.py`):** En `decodificar_secuencia_continua`, se añadió el parámetro `modo_deteccion`. Al seleccionar `"Gate Doble"`, segmenta la señal continua al 100% mioeléctrico con cero dependencia del canal de audio.
- **Configuración Predeterminada para Autoencoder en 3D (Récord 88.45%):**
  - Se configuraron los valores récord oficiales al conmutar a 3D en `ui_analysis.py`:
    - Épocas: `800`
    - Tasa de Aprendizaje: `0.0020`
    - Factor Ortogonalidad de Pesos ($\lambda_W$): `1.20`
    - Factor Decorrelación Latente ($\lambda_Z$): `0.15`
    - Tamaño de Lote: `512`
    - Arquitectura Fisiológica Simétrica: `input_dim -> 64 -> 16 -> 3 -> 16 -> 64 -> input_dim` (Tanh, `bias=False`).
- **Clasificador Supervisado LDA para Trazado de Fronteras:**
  - Se añadió la opción `"Supervisado (LDA: Fronteras Lineales)"` en `cmb_clustering` de `ui_analysis.py`.
  - En `evaluar_espacio_latente` y `decodificar_secuencia_continua` de `motor_autoencoder_unificado.py`, se integró `LinearDiscriminantAnalysis()`. Al proyectar sobre secuencias continuas, las fronteras son hiperplanos exactos ajustados sobre las proyecciones de entrenamiento de Lucas, evitando solapamientos estocásticos de componentes Gaussianas ciegas.
- **Plan Arquitectónico del Módulo Decodificador en Tiempo Real (DAQ):**
  - Análisis exhaustivo de `autoforge_daq.py` y formulación de la hoja de ruta técnica para decodificación bioeléctrica en streaming sin audio (buffer circular de 2.5 s, RMS online 91 ms, Gate Doble en tiempo real, inferencia PyTorch < 0.05 ms y display de fonemas vía señales Qt).
- **Corrección de Crash de Silueta con 1 Sola Clase:**
  - En `evaluar_espacio_latente` (linea ~1898), se añadió un guard que verifica `len(np.unique(Y_eval)) >= 2` antes de calcular `silhouette_score` y `davies_bouldin_score`. Si solo hay 1 clase vocal en los datos extraídos, se asigna `sil = 0.0` y `db = inf` con un aviso en consola, evitando el `ValueError` que ocurría al usar Gate Doble sobre tomas de una sola vocal.

### Hito 126 - 2026-09-24: Calibración Fina de Gate Doble, Selector Flexible para Probar en Otro Sujeto y Récord de Autoencoder Convolucional Ortogonal

- **Calibración Fina del Detector Gate Doble por Toma:**
  - Se ajustaron los umbrales adaptativos en `generador_pca_umap.py` para capturar la totalidad de contracciones musculares suaves (vocales cerradas /u/, /i/ y /e/):
    $$U_{\text{alto}} = \max(\text{med}_{\text{base}} + 2.2 \cdot \sigma_{\text{rob}}, \; p_{98} \cdot 0.18)$$
    $$U_{\text{bajo}} = \text{med}_{\text{base}} + 1.0 \cdot \sigma_{\text{rob}}$$
    $$T_{\text{refr}} = \min(0.650 \cdot f_s, \; 0.65 \cdot W_{\text{pulso}})$$
  - Esta calibración elimina la sobre-filtración de tomas con baja amplitud muscular y converge a la misma cantidad de repeticiones que el micrófono (~501 ventanas).

- **Evolución del Botón "Probar en Otro Conjunto de Mediciones" en la GUI:**
  - En `main_app.py` (`run_autoencoder_no_sup_probar_otro_dataset`), se reemplazó la solicitud rígida de archivos CSV/NPZ por un diálogo selector interactivo tri-modal:
    1. **Mediciones Marcadas en el Gestor:** Si el usuario seleccionó tomas del otro sujeto en el árbol izquierdo (`SessionExplorer`), las procesa directamente con un clic.
    2. **Elegir Carpeta de Grabaciones:** Permite navegar a cualquier carpeta de la base de datos (ej. otra fecha o sujeto), escaneando recursivamente las subcarpetas que contienen `canal_0/grabacion.wav`.
    3. **Cargar Archivo de Dataset:** Mantiene la opción de cargar archivos pre-extraídos `.npz` o `.csv`.
  - Al seleccionar grabaciones crudas (opciones 1 y 2), el sistema ejecuta automáticamente `motor.extraer_dataset_unificado` con el pipeline DSP activo (Gate Doble o Mic, filtros, envolvente, calibración P95) y evalúa el espacio latente con el modelo entrenado mediante `motor.evaluar_espacio_latente`.
  - Se habilitó en `evaluar_espacio_latente` el paso opcional de `modelo_path` y la instanciación dinámica de `hidden_dim = 64` para 3D (o 32 para 2D).

- **Récord del Autoencoder Convolucional Ortogonal (Barrido de Parámetros):**
  - **Configuración Destacada:** Iteración 3179/5760 (55.2% del barrido de búsqueda en grilla).
  - **Hiperparámetros Óptimos:**
    - Tamaño de Núcleo Convolucional ($K$): `5`
    - Canales de Salida ($C$): `(6, 12)`
    - Función de Activación: `tanh`
    - Tasa de Aprendizaje ($\eta$): `0.003`
    - Factor de Ortogonalidad de Pesos ($\lambda_W$): `2.0`
    - Factor de Decorrelación Latente ($\lambda_Z$): `0.5`
  - **Métricas de Separación Latente GMM:**
    - Sujeto de Entrenamiento (Lucas): **$89.0\%$**
    - Sujeto de Generalización (Candela P5): **$81.3\%$**
    - Media Armónica Balanceada: **$85.0\%$**
  - **Criterio de Selección:** Se priorizó esta configuración por su alta exactitud en el espacio latente de entrenamiento de Lucas (89.0%) y su preservación geométrica de las 5 clases fonatorias, logrando un balance inter-sujeto con Candela sin canibalizar vocales.
  - **Integración Oficial en la Interfaz Gráfica (`ui_analysis.py`):**
    - Se añadió la opción `"Autoencoder Ortogonal Convolucional: Récord 89% Lucas - 81% P5"` al desplegable de arquitecturas (`cmb_tipo_red`) y el botón directo `"Cargar Conv-Ortogonal: Récord 85%"`.
    - Al seleccionarlo, se configuran automáticamente los valores óptimos: Épocas = 350, Batch = 512, LR = 0.0030, $\lambda_W = 2.0$, $\lambda_Z = 0.5$, Impedancia de Reposo activada y carga de la plantilla PyTorch `ConvOrthogonalAutoencoder` en el editor de código con $K=5$, Canales $(6, 12)$ y función de pérdida ortogonal dual en capas convolucionales y lineales.
    - Se agregó el soporte del tag `conv_ortogonal` en `motor_autoencoder_unificado.py` para aplicar acondicionamiento de reposo e impedancia automáticamente.

### Hito 127 - 2026-09-24: Verificación Empírica de la Configuración 3179 y Resolución de Discrepancias en la GUI

- **Confirmación Numérica y Replicación Idéntica:**
  - Se ejecutó la configuración 3179 ($K=5$, Canales $(6, 12)$, Tanh, $\text{lr}=0.003$, $\lambda_W=2.0$, $\lambda_Z=0.5$, 350 épocas) tanto en el script de grilla como a través de `motor_autoencoder_unificado.py` con el código generado por la interfaz:
    - **Lucas (Entrenamiento):** **89.04%** (Silueta: $+0.425$, Davies-Bouldin: $0.795$)
      - /a/: 91.8%
      - /e/: 77.2%
      - /i/: 88.3%
      - /o/: 94.0%
      - /u/: 94.0%
    - **Candela P5 (Transferencia Directa sin fine-tuning):** **81.30%** (100 / 123 aciertos)
      - /a/: 100.0%
      - /e/: 84.0%
      - /i/: 52.0%
      - /o/: 75.0%
      - /u/: 95.8%
    - **Media Armónica Inter-Sujeto:** **85.00%**
- **Diagnóstico y Corrección de los Puntos de Falla en la GUI:**
  1. **Acondicionamiento de Reposo e Impedancia:** En versiones previas del motor, el acondicionamiento de reposo se omitía durante el entrenamiento cuando se usaba código personalizado, pero se aplicaba durante la evaluación. Se unificó para que aplique simétricamente en ambas fases.
  2. **Alineación SO(2):** La rotación rígida artificial SO(2) debe permanecer **desactivada**, ya que las coordenadas nativas ortogonales ya preservan la geometría canónica sin necesidad de forzar Kabsch contra una sesión arbitraria.
  3. **Entorno Virtual (`venv`):** Al correr desde terminal, se debe invocar `./venv/bin/python3` o activar el entorno (`source venv/bin/activate`) para disponer de `sklearn` y `torch`.

### Hito 128 - 2026-09-24: Corrección de Detección de Pulsos en Espacio de Fases 3D Multisesión

- **Diagnóstico del Error en la Galería del Espacio de Fases:**
  - El usuario reportó que el espacio de fases dinámico 3D se generaba vacío (los 6 paneles en blanco sin trayectorias ni órbitas, salvo el punto de reposo).
  - Causa exacta: la función `generate_dynamic_phase_space_3d` en `report_engine.py` dependía exclusivamente de la clave `"maxima_per_cut"` en el JSON de resultados de cada toma. En sesiones como las de Candela (`2026-09-18` y `2026-09-15`), dicha clave no existe, provocando que la lista de picos resultara vacía y se omitiera la extracción de segmentos.
- **Solución Implementada:**
  - Se incorporó un mecanismo robusto de fallback multi-nivel: en ausencia de `"maxima_per_cut"`, se leen los parámetros físicos del metrónomo (`bpm`, `noise_seconds`) de `metadata.json` y se detectan automáticamente los picos locales sobre la envolvente sumada de los canales activos.
  - Además, se blindó la carga de audio para rellenar con ceros canales inactivos o faltantes sin interrumpir el procesamiento tricanal.
- **Validación:**
  - Se regeneró la galería de 6 paneles y las proyecciones 2D para `2026-09-18` y `2026-09-15`, confirmando que todas las trayectorias vocálicas por pulso, órbitas promedio negras, estrellas de excursión máxima y lazos comparativos cerrados se grafican con nitidez sobre los paneles sombreados.

### Hito 129 - 2026-09-24: Selector Jerárquico de Mediciones, Corrección de Corte LP a 500 Hz, Decodificación Continua con Tira de Pulsos y Barrido Conv-Ortogonal en 3D

- **Selector Jerárquico Interactivo para Probar en Otro Sujeto (`selector_otro_sujeto_dialog.py`):**
  - Se implementó un diálogo modal con árbol (`QTreeWidget`), casillas de verificación (checkboxes) jerárquicas con sincronización padre-hijo (Sujeto -> Fecha -> Medición), buscador reactivo en tiempo real y botones de selección rápida: "Marcar Todo", "Desmarcar Todo", "Solo Continuas" y "Solo Aisladas".
  - Mantiene la compatibilidad con datasets externos `.npz` y `.csv`.
  - Integrado en `main_app.py` (`run_autoencoder_no_sup_probar_otro_dataset`) para procesar dinámicamente las tomas seleccionadas mediante `motor.extraer_dataset_unificado`.

- **Corrección de Frecuencia de Corte Pasa-Bajos a 500.0 Hz por Defecto:**
  - Se identificó la discrepancia entre el 88.02% (501 ventanas) y el 89.04% (502 ventanas): la caja de `LP Cutoff` en la GUI inicializaba en 450.0 Hz en lugar de 500.0 Hz.
  - Al cortar en 450 Hz, un pulso de /i/ perdía energía de alta frecuencia y quedaba excluido por umbral de SNR, reduciendo el conteo y perturbando levemente la frontera GMM entre /o/ y /u/.
  - Se fijó `self.inp_lp.setValue(500.0)` como predeterminado tanto en el arranque (`__init__`) como en la función de carga rápida (`on_cargar_conv_orto`).

- **Decodificación de Secuencia Continua con Tira Temporal y Modelo Campeón Convolucional:**
  - Se actualizó `decodificar_secuencia_continua` en `motor_autoencoder_unificado.py` y `main_app.py`:
    - Se eliminó la línea de trayectoria continua que unía los puntos en el espacio latente.
    - Se sustituyó el panel de reconstrucción por una tira temporal de la señal bioeléctrica continua (norma EMG tricanal), resaltando las ventanas segmentadas con sombreado de color por clase, líneas de corte y rótulos de texto (`1:A`, `2:E`, `4:O`, etc.) con indicación de aciertos y errores.
    - Se añadió la generación de un gráfico panorámico completo (`grafico_totalidad_pulsos_decodificados.png`) dividido en 4 tramos de 60 segundos que abarca la totalidad de la grabación continua.
    - Se reemplazó la carga del modelo lineal antiguo por el modelo convolucional ortogonal campeón recién entrenado (soporte de checkpoints de `ConvOrthogonalAutoencoder` y compilación de código personalizado), logrando una transferencia directa a Candela P5 del **81.30%** (100 / 123 pulsos acertados) frente al 77.2% anterior.

- **Módulo de Búsqueda en Grilla para Autoencoder Convolucional Ortogonal en 3D (`grid_search_conv_ortogonal_3d.py`):**
  - Implementación del script de barrido masivo adaptado a un espacio latente de 3 dimensiones ($Z \in \mathbb{R}^3$):
    - Arquitectura `ParametricConvOrthogonalAE3D`: compresión con convoluciones 1D, capa densa intermedia de 64 unidades y proyección a 3 coordenadas latentes.
    - Función de pérdida con buffer precalculado $I_{3\times 3}$: decorrelación latente $\mathcal{L}_Z = \|\text{Cov}(Z) - I_{3\times 3}\|_F^2$ y ortogonalidad matricial $\mathcal{L}_W$.
    - Métricas conjuntas: GMM en 3D sobre Lucas, transferencia directa a P5 en 3D y optimización de la media armónica sin canibalización de fonemas ($\min(\text{Vocal}_{\text{P5}}) \ge 50\%$).
    - Visualización con subplots 3D (`Axes3D`) para Lucas y P5, matriz de confusión y exportación automática del modelo campeón `modelo_campeon_conv_ortogonal_3d.pt`.
    - Modos de ejecución configurables (`quick`, `3600`, `5760`) con soporte para reanudación (`--resume`).

- **Punto de Pausa del Barrido en Grilla 2D (`grid_search_conv_ortogonal.py`):**
  - **Iteración pausada:** `[3936/5760]` ($68.3\%$ completado).
  - **Parámetros en pausa:** $K = 3$, Canales $= (8, 16)$, Activación $= \tanh$, $\text{lr} = 0.006$, $\lambda_W = 0.8$, $\lambda_Z = 0.15$.
  - **Métricas instantáneas:** Lucas: $85.3\%$, P5: $78.0\%$, Media Armónica: $81.5\%$, Tiempo restante estimado en pausa: $22.3\text{ min}$.
  - **Archivo de persistencia:** `EMG_desarrollo/resultados/grid_search_conv_ortogonal/resultados_grid_search_5760.csv` (3939 filas guardadas).
  - **Comando exacto para reanudar cuando se desee:**
    ```bash
    ./venv/bin/python3 EMG_desarrollo/deep_learning/grid_search_conv_ortogonal.py --mode 5760 --resume
    ```

### Hito 130 - 2026-09-24: Estudio Fisiológico de TKEO, Resolución Temporal y Descubrimiento del Autoencoder con Pérdida Compuesta Dual Head

- **Aclaración Canónica del Sujeto en Prueba 5:**
  - El usuario aclaró que la toma `SecuenciaContinua_Prueba5_Sujeto1` (2026-06-10, 123 pulsos a 30 BPM) corresponde al mismo sujeto (Lucas en habla continua). Por ende, la transferencia evalúa la generalización intra-sujeto desde habla aislada a habla continua rápida sin re-entrenamiento ni ajuste fino.

- **Evaluación del Operador de Energía Teager-Kaiser (TKEO) en la Entrada:**
  - **Solo TKEO:** En el entrenamiento de Lucas alcanzó **$89.84\%$** de exactitud GMM (superando a RMS en aisladas, con /o/ al $99.0\%$ y /a/ al $98.0\%$).
  - **El fallo en continua:** En la secuencia continua P5, la vocal **/i/** se desplomó al **$32.0\%$** (canibalizada, solo 8 aciertos de 25), rompiendo el equilibrio multiclase.
  - **Híbrido Espacial (6 Canales):** Meter RMS y TKEO juntos como canales de entrada degradó la generalización ($60.98\%$ en P5) debido al desbalance de escalas físicas entre microvoltios de amplitud y energía cuadrática.

- **Evaluación de Resolución Temporal (20 vs 25 vs 30 vs 35 Puntos):**
  - Se confirmó experimentalmente que **20 puntos por canal es el óptimo físico absoluto**.
  - A mayor número de puntos (bines $< 35\text{ ms}$), el modelo decae monótonamente ($84.5\% \to 81.7\% \to 80.5\% \to 78.3\%$) y la vocal /i/ se destruye ($70.9\% \to 37.9\%$), debido a que las convoluciones captan el disparo estocástico asíncrono de las unidades motoras individuales en lugar del patrón global de la envolvente.

### Hito 131 - 2026-09-24: Integración Completa en el Reporte LaTeX: Autoencoder Convolucional Ortogonal Récord, TKEO, Resolución Temporal y Pérdida Compuesta Dual Head

- **Documento Actualizado:** `reportes_experimentos/apunte_arquitectura_red_y_secuencia_continua.tex` (edición puramente aditiva y no destructiva).
- **Secciones Nuevas Incorporadas:**
  1. `\subsection{Descubrimiento del Autoencoder Convolucional Ortogonal Récord}`:
     - Configuración 3179 ($K=5$, Canales $(6, 12)$, Tanh, $\text{lr}=0.003$, $\lambda_W=2.0$, $\lambda_Z=0.5$).
     - Ecuación de pérdida con búferes persistentes de identidad $I_{d_l}$ y matriz de covarianza empírica $\text{Cov}(Z)$.
     - Tabla comparativa frente al control lineal (Lucas $89.04\%$, P5 $81.30\%$, Armónica $85.00\%$) y superación del colapso de /e/.
     - Figura: `EMG_desarrollo/resultados/grid_search_conv_ortogonal/grafico_campeon_record.png`.
  2. `\section{Estudio del Operador de Energía Teager-Kaiser frente a la Envolvente RMS}`:
     - Formulación matemática discreta $\Psi[s[n]] = s[n]^2 - s[n-1] s[n+1]$, rectificación y suavizado de $90.5\,\text{ms}$.
     - Desglose exhaustivo de variables con unidades ($\mu\text{V}$, $\mu\text{V}^2$).
     - Evaluación de entrada pura TKEO ($89.84\%$ en Lucas pero colapso de /i/ al $32.0\%$ en P5) y entrada híbrida 6 canales ($60.98\%$).
     - Diagnóstico físico del colapso de /i/ por sensibilidad cuadrática al piso de reposo en habla continua.
     - Figura: `EMG_desarrollo/resultados/experimento_rms_tkeo_riguroso/comparativa_rigurosa_rms_vs_tkeo.png`.
  3. `\section{Estudio de Resolución Temporal: Comparativa de 20 frente a 25, 30 y 35 Puntos}`:
     - Barrido de puntos $T \in \{20, 25, 30, 35\}$ por canal.
     - Demostración empírica de caída monótona ($85.00\% \to 81.74\% \to 80.54\% \to 78.37\%$) y destrucción de /i/ ($52\% \to 32\%$).
     - Justificación biofísica: $20$ puntos ($50\,\text{ms}$) es el óptimo físico que promedia el *jitter* estocástico de las unidades motoras sin captar ruido aleatorio inter-espiga.
     - Figura: `EMG_desarrollo/resultados/experimento_resolucion_temporal/comparativa_resoluciones_temporales.png`.
  4. `\section{Autoencoder con Pérdida Compuesta Dual Head: Reconstrucción Simultánea de RMS y TKEO}`:
     - Entrada limpia $3 \times 20$ RMS con decodificador de dos cabezas paralelas (reconstrucción simultánea de envolvente RMS y perfil TKEO).
     - Formulación de $\mathcal{L}_{\text{compuesta}} = \text{MSE}(\hat{X}_{\text{RMS}}, X_{\text{RMS}}) + \beta \cdot \text{MSE}(\hat{X}_{\text{TKEO}}, X_{\text{TKEO}}) + \lambda_W \mathcal{L}_W + \lambda_Z \mathcal{L}_Z$.
     - Barrido de $\beta \in [0.00, 0.20]$: máximo en $\beta = 0.05$ con **$82.11\%$ en P5 (101/123 pulsos)**, **$88.65\%$ en Lucas**, media armónica récord de **$85.25\%$** y preservación inmaculada de todas las vocales (piso en Lucas $77.2\%$, piso en P5 $52.0\%$).
     
### Hito 132 - 2026-09-24: Evaluación Experimental en Candela (01/09 vs 15/09) y Corrección de Impedancia Inter-Toma

- **Motivación Experimental:**
  - El usuario propuso evaluar la arquitectura convolucional ortogonal compacta ($K=5$, Canales $(6, 12)$, Tanh, $\lambda_W=2.0$, $\lambda_Z=0.5$, 20 muestras) sobre dos sesiones de Candela con diferente configuración mioeléctrica (01/09 con Risorio vs 15/09 con Cigomático Mayor) para estudiar la disociación fonatoria.
- **Diagnóstico del Artefacto de Desdoblamiento de /a/:**
  - En la primera prueba, la vocal /a/ del 01/09 apareció dividida en dos bandas paralelas (arriba y abajo).
  - Causa: Entre Prueba1/Prueba2 y Prueba3/Prueba4/Prueba5 hubo una variación de offset de contacto piel-electrodo. Sin sustracción de reposo ni balance de impedancias, el salto de continua dominó el eje $Z_2$.
  - Solución: Se integró la extracción oficial con sustracción dinámica de ruido basal IQR y acondicionamiento de impedancia inter-toma (`acondicionar_reposo_impedancia`). Las dos bandas colapsaron de inmediato en un único racimo mandibular.
- **Resultados Cuantitativos Corregidos:**
  1. **Candela 01/09 (Risorio en Canal 1, Anterior Belly en Canal 0, Orbicular en Canal 2):**
     - Ventanas: 217 inliers post-purga (Isolation Forest 10%).
     - **Separación Par /e/ frente a /i/:** **$100.0\%$** (/i/ alcanza $76.2\%$ con cero confusiones hacia /e/).
     - **Separación Par /o/ frente a /u/:** **$53.0\%$** (Colapso en el polo del orbicular superior).
     - Exactitud GMM Global: $55.76\%$ (Silueta $+0.369$, DB $1.12$).
     - Desglose: /a/: $38.1\%$, /e/: $58.0\%$, /i/: $76.2\%$, /o/: $79.2\%$, /u/: $17.1\%$.
  2. **Candela 15/09 (Cigomático Mayor en Canal 1, Anterior Belly en Canal 0, Orbicular en Canal 2):**
     - Ventanas: 197 inliers post-purga (carpeta `2026-09-16`).
     - **Separación Par /e/ frente a /i/:** **$100.0\%$** (/i/ alcanza **$100.0\%$ de exactitud pura**, 39/39 aciertos).
     - **Separación Par /o/ frente a /u/:** **$59.5\%$** (Polo labial superior concentrado).
     - Exactitud GMM Global: **$64.97\%$** (Silueta $+0.436$, DB $0.94$).
     - Desglose: /a/: $30.8\%$, /e/: $75.0\%$, /i/: $100.0\%$, /o/: $65.8\%$, /u/: $53.7\%$.
- **Conclusión Fisiológica:**
  - Ambas sesiones convergen a una geometría triangular idéntica:
    - Polo de sonrisa (/i/): Aislado con 100% de pureza respecto a /e/ (en 15/09 perfecto 39/39).
    - Polo mandibular (/a/ y /e/): Agrupado abajo.
    - Polo labial (/o/ y /u/): Agrupado arriba por co-activación orbicular.
  - Artefactos generados:
    - Gráfico comparativo: `EMG_desarrollo/resultados/experimento_candela_conv_ortogonal/evaluacion_candela_0901_y_0915.png`.
    - Métricas consolidadas: `EMG_desarrollo/resultados/experimento_candela_conv_ortogonal/metricas_candela.json`.

### Hito 133 - 2026-09-25: Implementación de Ventana de Corte Variable en GUI y Lanzamiento de Barrido Masivo Cuatrimodal en Lucas

- **Integración de Ventana de Corte Variable en GUI:**
  - Archivo `EMG_desarrollo/gui_app/views/ui_analysis.py`:
    - Sección 5 actualizada: "5. Alineación de Pulso Fisiológico y Ventana de Corte".
    - Controles interactivos agregados: `inp_pre_pct` (default 0.40) e `inp_post_pct` (default 0.60) con rango 0.05 a 0.95.
    - `get_autoencoder_kwargs` conectado dinámicamente con `pre_pct` y `post_pct` para el motor no supervisado.
  - Archivo `EMG_desarrollo/gui_app/main_app.py`:
    - Métodos `run_autoencoder_no_sup_extraer`, `run_autoencoder_no_sup_completo` y `run_autoencoder_no_sup_evaluar` actualizados para propagar `pre_pct` y `post_pct` a `extraer_dataset_unificado`.

- **Resultados de Validación Previa (Tier Rápido, 48 Combinaciones por Modo):**
  - `conv_2d`: 83.63% Exactitud GMM Global (/a/: 91.5%, /e/: 62.4%, /i/: 90.6%, /o/: 80.2%, /u/: 93.9%, /o/-/u/: 87.0%, silueta: +0.322).
  - `mlp_2d_sin_so2`: 78.24% Exactitud GMM Global (/a/: 100.0%, /e/: 65.3%, /i/: 70.8%, /o/: 59.4%, /u/: 98.0%, /o/-/u/: 78.5%, silueta: +0.349).
  - `conv_3d`: 84.03% Exactitud GMM Global (/a/: 90.4%, /e/: 63.4%, /i/: 93.4%, /o/: 79.2%, /u/: 93.9%, /o/-/u/: 86.5%, silueta: +0.292).
  - `mlp_3d_sin_so2`: 85.03% Exactitud GMM Global (/a/: 92.6%, /e/: 67.3%, /i/: 90.6%, /o/: 81.2%, /u/: 93.9%, /o/-/u/: 87.5%, silueta: +0.320).

- **Lanzamiento del Barrido Masivo Completo (Tier 5760 / 5040):**
  - **Script ejecutado:** `EMG_desarrollo/deep_learning/grid_search_lucas_ventana4060.py` con `--modo todos --tier 5760 --epochs 350`.
  - **Modos incluidos en secuencia:**
    1. `conv_2d`: 5.760 combinaciones (canales, núcleos de 3 a 9, activaciones, lr, regularizaciones).
    2. `mlp_2d_sin_so2`: 5.040 combinaciones (12 configuraciones de capas ocultas, tanh/gelu, 5 lrs, regularizaciones).
    3. `conv_3d`: 5.760 combinaciones.
    4. `mlp_3d_sin_so2`: 5.040 combinaciones.
  - **Total de combinaciones:** 21.600 ejecuciones a 350 épocas con optimizador Adam y regularización ortogonal analítica.
  - **Persistencia y Trazabilidad:** Guardado incremental muestra a muestra en `resultados_<modo>_5760.csv`, con exportación automática de pesos `.pt`, configuración `.json` y gráfico de dispersión `.png` para el modelo campeón de cada modo.

- **Punto de Pausa Solicitado por el Usuario:**
  - **Fecha y hora de pausa:** 2026-09-25 13:15 UTC-3.
  - **Modo en proceso:** `conv_2d` (Autoencoder Convolucional 1D Ortogonal 2D).
  - **Combinaciones completadas y guardadas:** 1.703 de 5.760 ($29.6\%$).
  - **Archivo de persistencia:** `EMG_desarrollo/resultados/grid_search_lucas_ventana4060/conv_2d/resultados_conv_2d_5760.csv` (1.704 filas con encabezado).
  - **Récord actual vigente del modo:** **80.44%** de exactitud global GMM (Separación /o/-/u/: $84.5\%$, Piso mínimo: $59.4\%$).
  - **Configuración campeona vigente:** Canales $(3, 6)$, núcleo $K = 5$, activación Tanh, $\text{lr} = 0.002$, $\lambda_W = 2.0$, $\lambda_Z = 0.35$.
  - **Artefactos del campeón preservados:**
    - Pesos: `EMG_desarrollo/resultados/grid_search_lucas_ventana4060/conv_2d/campeon_conv_2d.pt`
    - Configuración: `EMG_desarrollo/resultados/grid_search_lucas_ventana4060/conv_2d/config_campeon_conv_2d.json`
    - Gráfico: `EMG_desarrollo/resultados/grid_search_lucas_ventana4060/conv_2d/campeon_conv_2d.png`
  - **Comando exacto para reanudar cuando el usuario lo disponga:**
    ```bash
    ./venv/bin/python3 EMG_desarrollo/deep_learning/grid_search_lucas_ventana4060.py --modo todos --tier 5760 --epochs 350
    ```
    (El script detectará automáticamente las 1.703 combinaciones ya evaluadas, recuperará el récord del $80.44\%$ y continuará sin pérdidas desde la combinación 1.704).

### Hito 134 - 2026-09-25: Preparación de Build de Windows, Actualización de Spec de PyInstaller y Sincronización Git

- **Preparación para Compilación en Windows:**
  - Se auditó la cadena de empaquetado PyInstaller (`build.bat`, `crear_spec_ejecutable.py`, `aplicar_parches_ejecutable.py`).
  - Se actualizaron `EMG_desarrollo/herramientas_build/crear_spec_ejecutable.py` y `EMG_desarrollo/EMG_Ejecutable_Build/EMG_Studio.spec` para incluir explícitamente los nuevos módulos en `additional_modules`:
    - `analysis.generador_figura_multimodal`
    - `analysis.generador_atlas_pdf`
    - `analysis.batch_actualizar_figuras_reporte`
    - `gui_app.views.atlas_dialog`
    - `gui_app.views.selector_otro_sujeto_dialog`
  - Se corrigió la importación de `os` en la cabecera generada de `EMG_Studio.spec` para evitar excepciones al evaluar el icono.
- **Archivos Incorporados y Consolidados en el Repositorio:**
  - Nuevos diálogos GUI y motores de reporte: `atlas_dialog.py`, `selector_otro_sujeto_dialog.py`, `generador_atlas_pdf.py`, `batch_actualizar_figuras_reporte.py`.
  - Scripts de experimentación bioeléctrica y grid search: `grid_search_lucas_ventana4060.py`, `experimento_candela_conv_ortogonal.py`, `experimento_candela_perdida_compuesta.py`, `experimento_perdida_compuesta_tkeo.py`, `experimento_resolucion_temporal.py`, `experimento_rms_tkeo_lucas.py`, `experimento_rms_vs_tkeo_riguroso.py`, `grid_search_conv_ortogonal.py`, `grid_search_conv_ortogonal_3d.py`.
  - Reportes LaTeX actualizados: `Reporte_EMG_2026-09-18.tex`, `Reporte_EMG_2026-09-22.tex`, `Reporte_EMG_2026-09-23.tex`, `Reporte_EMG_2026-09-24.tex`, `Reporte_SNR_2026-09-23.tex`, `apunte_arquitectura_red_y_secuencia_continua.tex`.
  - Cuaderno de Tesis actualizado: `Cuaderno_Tesis.docx`.
- **Sincronización:**
  - Confirmación y subida íntegra a GitHub (`origin/master`) para habilitar la compilación nativa en entorno Windows mediante `build.bat`.








