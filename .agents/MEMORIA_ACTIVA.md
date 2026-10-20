# Memoria Activa del Proyecto (Ñandú EMG)

**Fecha de consolidación:** 2026-10-03  
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
  - **Récord Lucas MLP (Ventana 50/50):** **88.45%** en 3D (`(64, 16)`, Tanh, $\text{lr}=0.002$, $\lambda_W=1.2$, $\lambda_Z=0.15$) y **87.85%** en 2D nativo.
  - **Récord Lucas MLP (Ventana 40/60, 3D):** **87.43%** Lucas GMM y **87.21%** Media Armónica (`(48, 24)`, Tanh, $\text{lr}=0.0035$).
  - **Secuencia Continua P5:** Corresponde a habla continua del propio Lucas (`SecuenciaContinua_Prueba5_Sujeto1`), no de Candela.
  - **Resultados en Ventana 60/60 (Compensación de P5):** Media armónica 86.58% en 3D y 85.55% en 2D impulsada por P5 (87-90%), pero en Lucas GMM la ventana 60/60 rinde menos (83.4% en 2D, 85.4% en 3D) por solapamiento del metrónomo de 30 BPM.
  - **Presets Récord en GUI:** Implementado `cmb_preset_record` en `ui_analysis.py` con 6 configuraciones oficiales que asignan dimensiones, ventana, hiperparámetros y código PyTorch exactos.
  - **Decodificador en Tiempo Real (DAQ + Gate Doble):** En `realtime_decoder_engine.py` y `autoforge_daq_decodificador.py`, osciloscopio con 4 canales (CH0 a CH3), envolvente RMS en tiempo real por defecto (90 ms configurable), curva de norma tricanal $S_{\text{emg}}$ magenta, auto-escala dinámica sin techos artificiales y control interactivo de retardo electromecánico fisiológico EMD (default 350 ms). Integrado el modelo campeón de media armónica 87.21% (MLP 3D sin SO2 ventana 40/60) replicando con exactitud 87.43% en Lucas y 86.40-86.99% en P5 (108 de 125 fonaciones). Incorporado selector dual en el panel latente con **Vista 3D Interactiva** (Matplotlib Qt con rotación continua por mouse en 360°, zoom, centroides en diamante $\blacklozenge$, cruz en $(0,0,0)$ y preservación de ángulo en streaming) y **Vista 2D** con regiones pastel, centroides y selector de planos de proyección (Z1-Z2, Z1-Z3, Z2-Z3) para modelos 3D, con auto-conmutación según dimensión del modelo cargado.

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

## 5. Alineación con CCA, Tríadas Musculares y Hoja de Ruta
- **Límites de SO(d) y Marco de Bach y Jordan (2005):** La rotación rígida en SO(d) o Procrustes asume isometría pura y fracasa ante distorsiones afines (impedancia y colocación). Se adoptó CCA probabilístico ($x_i = W_i z + \epsilon_i$), reconociendo la naturaleza hipotética del comando motor común $z$ debido a la equivalencia motora de Bernstein.
- **Ángulos de Jordan y Validación de Tríada Canónica:** 
  - Risorio en Canal 1 (Candela 09-01) desfasaba /e/ a $75.1^\circ$.
  - Cigomático Mayor en Canal 1 (Candela 09-15/16) colapsó la divergencia de /e/ a solo $6.9^\circ$, con /u/ a $0.7^\circ$, /a/ a $19.0^\circ$ y /o/ a $20.6^\circ$. Correlación canónica principal $\rho_1 = 0.9824$ ($\theta_1 = 10.76^\circ$).
- **Hoja de Ruta de Ensayos:**
  1. **Experimento 1 (Petra 2026-08-21 vs 2026-08-28) [COMPLETADO]:** Estabilidad inter-día con tríada Anterior Belly, Zygomaticus Major y Levator Anguli Oris. Desvío angular mínimo: /a/ a $1.3^\circ$, /e/ a $2.4^\circ$, /i/ a $2.5^\circ$, /u/ a $3.3^\circ$, /o/ a $5.6^\circ$. Ángulo de Jordan $\theta_1 = 0.00^\circ$ ($\cos \theta_1 = 1.0000$), CCA inter-día $\rho_1 = 0.9960$ ($5.15^\circ$).
  2. **Experimento 2 (Alineación Inter-Sesión Intra-Sujeto: Lucas T1-T7, Petra 08-21 vs 08-28 y Candela 1-4) [COMPLETADO]:** Alineación CCA de variedades PCA intra-sujeto. Lucas T1 a T7 (502 muestras) $\rho_1 \ge 0.986$ ($\theta_1 < 4.5^\circ$), Petra inter-día 08-21 vs 08-28 (254 muestras) $\rho_1 = 1.0000$ ($\theta_1 = 0.48^\circ$), Candela pruebas 1 a 4 (191 muestras) $\rho_1 \ge 0.998$ ($\theta_1 < 2.8^\circ$). Superposición sin centroides ni rayos que obstruyan datos empíricos.
  3. **Experimento 3 (PCA lineal y Alineación de Sara Solla Tri-Sujeto) [COMPLETADO]:** Sesión única por sujeto (Lucas 07-10, Candela 09-15, Petra 08-28 Día 2). EVR 2D acumulada: Lucas 82.3%, Candela 88.6%, Petra 81.1%. CCA de Sara Solla: Candela vs Lucas $\rho_1 = 0.9996$ ($1.61^\circ$), Petra vs Lucas $\rho_1 = 0.9965$ ($4.82^\circ$) y $\rho_2 = 0.9635$ ($15.53^\circ$). Atractor /a/ coincide entre Lucas y Petra en $+Y$ con desvío de solo $0.01$ en $Z_1$, y la constricción /o/-/u/ colapsa en el mismo haz en $-X$.
  4. **Experimento 4 (Hipótesis de Impedancia y Variedad Motora Intrínseca Universal Tri-Sujeto: 947 muestras) [COMPLETADO]:**
     - **Lucas (502 m, 3 días, DAQ Dev1/Dev2):** Con impedancia + Sara Solla alcanza el récord de cohesión (Silueta $+0.4165$, DB $0.8212$). Sin impedancia, Sara Solla rescata la variedad ($+84.7\%$ en PCA, $+93.0\%$ en Autoencoder 2D récord). La corrección de impedancia realiza el trabajo sucio no lineal ecualizando ganancias y Sara Solla absorbe derivas angulares geométricas.
     - **Candela (191 m) y Petra (254 m):** Alineación intra-sujeto verificada con y sin impedancia.
     - **Variedad Motora Intrínseca Universal (947 muestras):** Alineación inter-sujeto con Sara Solla (Candela vs Lucas $\rho_1 = 0.9986$, Petra vs Lucas $\rho_1 = 0.9977$). Invarianza anatómica estricta: /a/ colapsa en $+Y$ (Ch0 depresor mandibular), /u/ y /o/ en $-X$ (Ch2 orbicular de labios) y /i/ en $(0,0)$. /e/ se desacopla ligeramente según Ch1 (DAO vs Risorio vs Cigomático) y es alineada afínmente.
     - **Separabilidad Tri-Sujeto (947 m):** GMM ciego da 57.66% por solapamiento en /o/-/u/, pero LDA supervisado alcanza **78.04% (77.83% CV)**, probando que la variedad latente unificada es fuertemente cuasi-linealmente separable (/a/ 94.6%, /i/ 83.3%, /u/ 82.9%, /o/ 74.1%).
  5. **Experimento 5 (Compensación Biofísica: Escala de Electrodos Pre-Encoder y Contracción Radial Latente) [COMPLETADO]:**
     - Se demostró que la brecha de transferencia hacia el decodificador de Lucas (91.63%) se debía exclusivamente a:
       1) Desbalance de ganancia de parches ($k_w$) antes del supremo tricanal (Candela $kw_1=0.5, kw_2=2.0 \implies 59.16\%$; Petra $kw_1=3.0, kw_2=1.0 \implies 64.17\%$).
       2) Dilatación de radio de giro latente ($R \approx 0.50$ vs $0.25$ de Lucas).
     - Al ecualizar la escala de dispersión radial ($\alpha = 0.40$), la exactitud sobre fronteras de Lucas salta a **92.67% en Candela** y **92.91% en Petra**, validando la universalidad total de la Cruz Latente sin reentrenamiento.

   6. **Experimento 6 (Cálculo Analítico de $k_w$ y Variedad Latente Intrínseca Universal por GPA) [COMPLETADO]:**
      - **Cálculo No Supervisado de $k_w$:** A partir de los ratios de amplitud relativa $P_{95}$ ($kw_c = \frac{P_{95}(x_c)/P_{95}(x_0)}{P_{95}^{\text{Lucas}}(x_c)/P_{95}^{\text{Lucas}}(x_0)}$), se eliminó la necesidad de barrido manual de hiperparámetros. Candela ($kw = [1.00, 0.44, 0.31]$) alcanza **93.19%** y Petra ($kw = [1.00, 0.30, 1.00]$) alcanza **92.91%** sobre el decodificador congelado de Lucas.
      - **Variedad Intrínseca Universal por GPA:** Generalized Procrustes Analysis sobre los centroides de Lucas, Candela y Petra converge en 15 iteraciones. Cruz canónica universal con /a/ a $90.0^\circ$ ($+Y$), /e/ a $28.0^\circ$, /i/ a $350.1^\circ$ ($+X$), /o/ a $213.2^\circ$ y /u/ a $226.3^\circ$ ($-Y$).
      - **Panel 2x3 con Fronteras GMM:** Generado en `EMG_desarrollo/resultados/grid_search_conv_ortogonal/variedad_intrinseca_tri_sujeto_completo.png` con regiones pastel, contornos negros y centroides diamante $\blacklozenge$, evidenciando el encastre de los tres sujetos en la misma variedad geométrica.
   7. **Experimento 7 (Evaluación Definitiva de Tríadas con Corrección por Impedancia Pura y Fronteras LDA de Lucas) [COMPLETADO]:**
      - **Pipeline Físico Puro:** 1) Corrección de impedancia basal por electrodo $(x_c - \mu)/P_{95}(c)$ + supremo tricanal. 2) Sara Solla intra-sujeto inter-sesión. 3) Sara Solla inter-sujeto hacia Lucas. 4) Clasificación directa en las fronteras fijas LDA de Lucas (sin $k_w$ forzado, sin $\alpha$ latente, sin líneas de contorno).
      - **Resultados de Transferencia Zero-Shot sobre Lucas (Azar = 20.0%):** Lucas (91.04%), Petra (63.78%, /a/: 100%), Candela 09-01 (61.78%, /i/: 79.5%), Candela 09-18 (56.54%), Santi 06-22 (56.40%) y Candela 08-30 (50.00%).
      - **Hallazgo Físico Fundamental:** La corrección por impedancia propia de cada sensor supera ampliamente a cualquier ecualización de ratios $k_w$ (+20% en Candela 09-01 y +9% en Petra). Al eliminar el reescalado latente $\alpha$, las fonaciones no se apiñan en el vértice central del LDA, preservando la separabilidad angular natural de los atractores.
      - **Validación Fisiológica de Santi:** En Santi 06-22, el Ch0 midió Milohioideo profundo en vez de Digástrico anterior; la /a/ (65%) y la /i/ (42%) se confunden en el origen, demostrando que sin sensor de apertura no hay decodificación biomecánica posible.
      - **Figuras Oficiales:** Generadas `panel_6_sujetos_triadas_fronteras_lda_lucas.png` y `deconstruccion_proceso_lda_lucas.png` limpias, sin líneas divisorias ni cortes de ejes.

   8. **Experimento 8 (Variedad No Lineal Sara Solla, Sweet Spot 50-70 pts y Procrustes 91% en Streaming) [COMPLETADO]:**
      - **Sweet Spot de Remuestreo Físico:** El barrido temporal de Isomap demostró que 50--70 puntos por canal es el óptimo físico de Shannon para sEMG facial (error residual geodésico mínimo en Lucas 2.42 y récord absoluto en Santi 06-23 de 86.38%). A 500 puntos ocurre colapso dimensional por hiperconcentración euclídea (error residual 22.59).
      - **Alineación de Procrustes sobre Modelo 91% Congelado:** Frente al colapso de Deep CCA (~20%), congelar el encoder del 91% y alinear con Procrustes O(2) preserva la topología intacta.
      - **Validación Causal en Tiempo Real:** En test out-of-sample estricto tras calibración rápida de 20 segundos (5 fonaciones por vocal), Santi 06-23 alcanza **71.56%** ($N=211$) y Petra 08-28 alcanza **62.88%** ($N=229$) sobre las fronteras fijas LDA de Lucas, con una latencia computacional $< 0.35\,\text{ms}$ por fonación.
   9. **Experimento 9 (Pseudo HD-sEMG Facial de 9 y 12 Canales en Candela: Conv 1D 3D) [COMPLETADO]:**
      - **Ensamble 9 Canales (630D):** 9 músculos ordenados anatómicamente. Con sustracción de ruido IQR y supremo tricanal, LDA directo da **89.33%** (azar = 20.0%), AE 2D rinde **78.00%**, PCA 2D: **78.00%**, Isomap 2D: **79.33%**.
      - **Expansión a 12 Canales (840D) y $\alpha=0.5$:** Se incorporaron sub-zonas del masetero y submental lateral. LDA directo salta a un impresionante **97.33%**, probando la ortogonalidad articulatoria completa de la musculatura orofacial.
      - **Autoencoder Convolucional 1D Ortogonal 3D:** Arquitectura récord adaptada a tensor $(N, 12, 70)$ con cuello de botella a $\mathbf{z} \in \mathbb{R}^3$ (MSE = $0.006710$). Separabilidad LDA en espacio latente $Z$ escala a **89.33%**, y PCA 3D alcanza **94.00%**.
      - **Paneles Oficiales:** Guardados en `panel_pseudo_hd_emg_candela_autoencoder.png` (9CH 2D) y `panel_pseudo_hd_emg_candela_12ch_3d.png` (12CH 3D).
   10. **Experimento 10 (Tríadas Faciales Risorio-Modiolo-Cigomático y Triángulo Vocálico Cuatri-Sujeto: 935 fonaciones) [COMPLETADO]:**
       - **Auditoría Biomecánica de Tríadas en Candela:** Risorio (09-01) y Modiolo (09-18) comparten la mecánica horizontal inferior ($\rho_1 = 0.9966, \rho_2 = 0.9772$, /i/, /u/, /o/ residual $d \le 0.14$). Cigomático Mayor (09-16) tracciona a $45^\circ$, generando un cruce en X que invierte la posición relativa entre /a/ y /e/ ($d \approx 0.50$), manteniendo idénticos los polos cardinales.
       - **Modelo Campeón Récord Conv Ortogonal 2D:** Replicado con exactitud sobre 502 muestras de Lucas (87.85% GMM nativo, 88.05% LDA nativo y 91.04% LDA con alineación afín inter-sesión de Sara Solla). Define el marco canónico con /i/ en $(0,0)$, /a/ en $(0.00, +2.74)$ y /u/ en $(-2.20, +0.89)$.
       - **Triángulo Vocálico de Hellwag Cuatri-Sujeto (935 fonaciones, 82.89% LDA Global):**
         1) Lucas canónico (5V): **88.33%** ($N=497$).
         2) Santi 2026-06-23 curado (/a/, /i/, /o/, /u/ con Digástrico $\times 3.5$ y reposo basal): **81.87%** ($N=171$).
         3) Petra cardinal (/a/, /i/, /u/ en 2 días): **79.33%** ($N=150$).
         4) Candela cardinal (/a/, /i/, /u/ con Cigomático): **65.81%** ($N=117$).
       - **Panel Oficial:** Publicado en `variedad_triangulo_vocalico_curado_cuatrisujeto.png` (2x2), demostrando que la topología del triángulo vocálico cardinal es un invariante fisiológico universal orofacial.

    11. **Experimento 11 (Quiralidad 3D en PC3, PPCA Generativo y Dataset Balanceado de 1298 Fonaciones) [COMPLETADO]:**
        - **Resolución 3D de Quiralidad (/a/ vs /e/):** En 2D, Risorio (09-01) y Cigomático (09-16) parecían invertirse y cruzarse en X. En $\mathbb{R}^3$, Procrustes confirma rotación rígida propia $\text{SO}(3)$ ($\det = +1$) con despegue vertical invariable en PC3 ($z_3 > 0$ en /e/ y $z_3 < 0$ en /a/, $\Delta z_3 = +1.04$). Las curvas son alabeadas y no se cortan en el espacio motor. Figura: `resolucion_quiralidad_3d_cruce_a_e.png`.
        - **PPCA Generativo e Imputación Fisiológica:** Transferencia bayesiana a posteriori del prior de Lucas preservando la covarianza articular y el piso de ruido tisular local $\sigma_k^2$ para sintetizar /e/, /i/ y /o/ faltantes en Santi, Petra y Candela sin interpolación manual. Figura: `ppca_imputacion_generativa_cuatrisujeto.png`.
        - **Optimización Temporal dPCA:** El 62.2% de la energía de activación es reloj común ($X_t$), concentrando el 90% de la información fonética en las muestras núcleo 8 a 13. La ventana intermedia 30/30 ($D=36$) retiene el 85.06% LDA con 40% de reducción dimensional.
        - **Dataset de Entrenamiento Oficial:** Guardado en `EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_de_entrenamiento.npz` con 1298 muestras balanceadas en las 5 vocales (1028 biológicas reales, 270 sintéticas PPCA) y bandera estricta `es_reconstruido` para evaluación blindada en reales.

    12. **Experimento 12 (Alineación Latente de Sara Solla Cuatri-Sujeto y Benchmark P5: 83.56% Real Global) [COMPLETADO]:**
        - **Fracaso de Pre-Alineación en Señal (NNLS en $X$):** Intentar forzar a Candela/Petra/Santi al espacio de señal de Lucas antes de la red degradó la exactitud global de 72.9% a 58.5% (Petra colapsó a 2.0%) debido a que la mezcla no negativa inyecta sesgos estáticos basales (+0.15 a +0.18) que destruyen las envolventes temporales y anulan canales activos.
        - **Éxito de Alineación Latente en $Z$ (Procrustes $O(2)$ sobre Modelo Campeón Congelado):** Con el modelo convolucional récord congelado (`modelo_campeon_conv_ortogonal.pt`, Lucas GMM 87.85% y P5 GMM 82.93%), la alineación ortogonal de Sara Solla absorbe las rotaciones físicas de electrodos (-95° en Candela) e inversiones de proyección 2D ($\det = -1.0$).
        - **Resultados Cuatri-Sujeto en Muestras Biológicas Reales ($N = 1028$):** Santi alcanza **88.37%** LDA (+20.15%), Petra **77.78%** LDA (+49.68%, Día 1: 85.07%, Día 2: 72.09%), Candela **75.41%** LDA (09-01 Risorio: 79.85%, 09-16 Cigomático: 70.00%). El corpus real completo escala de 68.48% a **83.56% LDA** (+15.08%), preservando la decodificación continua de habla de **82.93% GMM / 86.18% LDA en P5** (102 de 123 fonaciones). Figura oficial: `benchmark_campeon_cuatrisujeto_procrustes.png`.

     13. **Experimento 13 (Variedad 3D Pseudo-HD de 12 Canales, Bootstrap Monte Carlo N=1000 y Codo Geodésico de Tenenbaum) [COMPLETADO]:**
         - **12 Canales Oficiales y Vector 240D:** Depurado a 12 canales canónicos funcionales en Candela (3 depresores, 3 elevadores, 4 retractores, 2 esfínteres) con fechas reales de `metadata.json`. Normalizado a $T = 20$ puntos por canal ($D = 240$ features).
         - **Ensamble Bootstrap Monte Carlo ($N = 1000$):** Resuelve la disparidad muestral (8 a 45 pulsos por canal) muestreando con reemplazo condicionado por vocal (200 fonaciones por vocal, semilla 42) sobre un espacio de $\sim 1.2 \times 10^{13}$ combinaciones. Escala a $N = 1000 > D = 240$, estabilizando la covarianza. LDA 5-Fold en espacio ambiente 240D alcanza **99.60%**.
         - **Separabilidad GMM No Supervisada (Asignación Húngara):** PCA 2D: 55.70% (/o/ y /u/ colapsan en el mismo cluster plano), PCA 3D: **79.60%** (salto de $+23.90\%$ al liberar el eje perpendicular de protrusión), Isomap 2D ($k=35$): **72.10%** (desenreda la curvatura en el plano), Isomap 3D: **73.50%** (LDA: 82.00%).
         - **Codo Geodésico de Tenenbaum ($d = 3$):** Error residual $1 - R^2(D_G, D_Y)$ cae de 33.74% (1D) a 20.76% (2D) y entra en meseta estricta en $d=3$ (**18.54%**), confirmando que la variedad motora oral tiene **dimensión intrínseca 3**. PCA decae lentamente sin codo (31.17% en 3D, 58.31% de varianza no explicada). Prueba concluyente de redundancia muscular orofacial y saturación de separabilidad inter-canal.
         - **Artefactos Oficiales:** `isomap_2d_montecarlo_1000.png`, `pca_isomap_3d_montecarlo_1000.png`, `variedad_3d_interactiva_montecarlo_1000.html`, `curva_varianza_residual_pca_vs_isomap.png`, `curva_varianza_dimensionalidad_pca_isomap_1000m.png` e informe técnico compilado `informe_variedad_3d_pseudo_hd_candela.tex`.

     14. **Experimento 14 (Nueva Arquitectura de Autoencoder Tri-Sujeto: Invarianza de Gauge, Pozos Armónicos y Test Ciego Holdout) [COMPLETADO]:**
         - **Invarianza de Gauge SO(3) y Tronco Convolucional:** Adaptadores por sujeto $W_{\text{in}}^{(s)} \in \mathbb{R}^{3 \times 3}$ regularizados con $\|W W^T - I\|_F^2$ acoplados a un tronco común Conv1D (filtros 6 y 12, kernel 5, Tanh) y cuello de botella $\mathbf{z} \in \mathbb{R}^2$.
         - **Pozos de Potencial Armónicos:** Atractores canónicos (/a/ en $(0.0, 1.8)$, /e/ en $(1.2, 0.8)$, /i/ en $(1.8, -0.2)$, /o/ en $(-1.0, 0.8)$, /u/ en $(-1.8, -0.2)$) desacoplan la protrusión y las vocales intermedias sin supervisión de gradientes directos.
         - **Evaluación Ciega Out-of-Sample (718 pulsos):** Train (1125 pulsos: Lucas T1-T3, Santi P1-P2, Candela 4 sesiones) vs Test ciego (718 pulsos: Lucas T4-T7, Santi P3-P4, Candela día 2026-09-01 inédito).
         - **Monte Carlo Conservativo y Récord LDA:** Perturbaciones $SO(3)$ de electrodo ($\sigma = 0.06$ rad $\approx 3.5^\circ$), jitter tisular ($\sigma = 0.25$) y preservación de sinergias. En test ciego retenido, **LDA alcanza 66.71% global** (Lucas 69.23%, Santi 69.17%, Candela 61.89%, superando a GMM 59.05%). Documentado en `reporte_autoencoder_decodificacion_vocalica.tex` con figura `comparativa_clasificador_lda_test_holdout.png`.
         - **Extensión al Espacio Latente 3D y Generalización Inter-Día:** Geometría $\mathbf{z} \in \mathbb{R}^3$ con polos fisiológicos (/a/ en $Y$, /i/ en $X$, /u/ en $Z$, /e/ en plano $XY$ y /o/ en plano $YZ$). En test ciego retenido (718 pulsos), **LDA 3D alcanza 67.27% global**, $k$-NN ($k=7$) alcanza **69.78%**, la generalización en el día retenido de Candela sube a **66.80%** (+4.91% frente a 2D) y /o/ salta a **73.1%** global (91.1% en Lucas). Checkpoint: `modelo_trisujeto_3d_gauge.pt`, figura: `comparativa_autoencoder_3d_test_holdout.png`.
     15. **Experimento 15 (Prueba Extrema en Sujeto Inédito Petra y Desacople de Macro-Grupos Articulatorios) [COMPLETADO]:**
         - **Evaluación en Petra (254 pulsos en 2 días):** Zero-shot puro ($W=I$) colapsa a nivel de azar (20.87%) por rotación espacial de electrodos.
         - **Calibración Rápida de Gauge (0.73s):** Calibrando solo la rotación $3 \times 3$ con Día 1 (86 pulsos) y testeando en Día 2 ciego (168 pulsos, 1 semana después), alcanza **63.69% k-NN** y **61.31% LDA** (76.74% en D1).
         - **Desacople Absoluto de los 3 Macro-Grupos:** Cero confusiones inter-grupo. Apertura /a/ al **100.0% sostenido** (46 de 46). Sonrisa (/e/, /i/) y protrusión (/o/, /u/) aisladas al 100% de la apertura. Error residual puramente intra-grupo (/i/ confundida con /e/ y /u/ con /o/), dependiente del balance de ganancia relativa del Canal 0 de apertura mandibular. Figuras: `evaluacion_extrema_petra_3d.png` y `matrices_confusion_petra_dos_dias.png`.

      16. **Experimento 16 - Evaluación de Dimensionalidad 4D y Límite Físico del Rango Tricanal: Consolidación del Modelo 3D Campeón [COMPLETADO]:**
          - **Diseño Latente en R^4:** Cuello de botella en 4D ($Z_1$ sonrisa, $Z_2$ apertura, $Z_3$ protrusión, $Z_4$ tensión/altura vocálica relativa con /i/, /u/ en $+1.0$ y /e/, /o/ en $-1.0$) y regularización de covarianza cuádruple $\|\text{Cov}(\mathbf{Z}) - \mathbf{I}_4\|_F^2$.
          - **Test Ciego Fuera de Muestra (718 pulsos):** LDA cae a **66.02%** (-1.25% vs 3D) y $k$-NN cae a **66.85%** (-2.93% vs 3D). /e/ colapsa de 62.1% a 49.7% (-12.4%) y /o/ cae de 73.1% a 66.2% (-6.9%).
          - **Test Extremo en Petra (254 pulsos):** Calibración rápida Día 1 cae de 76.74% a 54.65% (-22.09%), y Día 2 ciego cae a 48.21% LDA / 48.81% $k$-NN (-13.10%). /e/ canibaliza a /i/ (0% para /i/) y protrusión se dispersa hacia sonrisa.
          - **Límite Físico y Teorema de Rango:** Con 3 canales físicos ($C=3$), el rango de fuentes musculares ortogonales está acotado por 3 ($\text{rank}(A) \le 3$), validando el Codo Geodésico de Tenenbaum ($d=3$, 18.54% error residual). La 4ª dimensión sufre la maldición de la dimensionalidad e inyecta ruido tisular.
          - **Consolidación Oficial:** El Autoencoder 3D (`modelo_trisujeto_3d_gauge.pt`, 67.27% LDA / 69.78% $k$-NN holdout y 63.69% en Petra) queda ratificado como el estándar oficial y definitivo del proyecto. Checkpoint 4D y script: `experimento_autoencoder_4d.py`, figura: `comparativa_autoencoder_4d_test_holdout.png`.

      17. **Hito 17 - Reorganización Documental, Libro de Experimentos IA y Paquetización de Scripts [COMPLETADO]:**
          - **Libro de Experimentos IA:** Documento maestro `libro_de_experimentos_ia.tex` (compilado en `libro_de_experimentos_ia.pdf`, 181 páginas) que unifica los 4 reportes de investigación en capítulos estructurados, con subcarpeta local `imagenes/` (104 figuras autónomas).
          - **Presentación Autocontenida:** `presentacion/presentacion_avances_nandu_emg.tex` (compilada en PDF, 18 slides 16:9) con su propia subcarpeta `imagenes/` (20 figuras locales).
          - **Paquete de Scripts Experimentales:** 61 scripts experimentales mudados desde `EMG_desarrollo/deep_learning/` hacia `reportes_experimentos/libro_experimentos_ia/scripts_experimentos/`, incorporando `__init__.py` y `README.md` catalogado por capítulos.
          - **Respaldo Histórico Comprimido:** Generado `reportes_experimentos/archivo_reportes_historicos.zip` (739.58 MB, 564 archivos respaldados con verificación `testzip` limpia), dejando la raíz de reportes completamente despejada.
          - **Regla en AGENTS.md:** Fijada regla obligatoria de contención que prohíbe scripts sueltos en producción y exige guardarlos en `scripts_experimentos/`.

      18. **Hito 18 - Purga de Residuos, Aislamiento de Códigos de Prueba Fuera del Repo y Reubicación de Tesis [COMPLETADO]:**
          - **Aislamiento de Códigos de Prueba:** Trasladados más de 105 scripts de prueba y 8 temporales (`temp_*.py`) fuera del repositorio a `/home/santiago/repositorios/codigos_de_prueba_nandu/` (subcarpetas `codigos_de_prueba_raiz`, `codigos_de_prueba_emg_desarrollo`, `codigos_de_prueba_deep_learning` y `temp_scripts_emg_desarrollo`).
          - **Reubicación de Documentos de Tesis:** Carpeta `documentos_tesis/` movida desde la raíz hacia `reportes_experimentos/documentos_tesis/`, centralizando toda la producción académica y cuadernos de tesis.
          - **Purga de Residuos y Respaldos Externos:** Eliminados `.pytest_cache/`, logos duplicados, imágenes sueltas de gráficos y la carpeta redundante `EMG_desarrollo/views/`. Respaldados `respaldo_pre_release_2026-09-23/` y el tarball de distribución Linux (`NanduLsd_Linux_x64.tar.gz`, 362 MB) en `/home/santiago/repositorios/Nandu_Backup_Local/`.
          - **Verificación de Inferencia y GUI:** Verificada la importación limpia y autónoma de `main_app.py` con código de salida 0.

      19. **Experimento 19 - Reconstrucción Biorthogonal y Desconvolución Espaciotemporal de Wiener de los 3 Ejes Musculares Puros desde CSVs Curados [COMPLETADO]:**
          - **Auditoría de Datasets CSV vs NPZ:** Se identificó que `Candela_2026-09-04` en el `.npz` histórico correspondía a `2026-09-04_Candela_Frankenstein` (`canal_0` donado de otra fecha). Al trabajar desde los 7 archivos `caracteristicas_exportadas.csv` curados ($N = 1,683$ pulsos reales de Lucas, Santi, Candela en 4 montajes y Petra en 2 días), cada sesión individual presenta separabilidad nativa entre 76.3% y 88.2% (media ponderada 83.8%).
          - **Teoría del Operador Sándwich y Desconvolución Espaciotemporal:** La inversión espacial estática $3 \times 3$ corrige la quiralidad ($\det = +1.000$) y centra las distancias baricéntricas ($\beta_E \to 0.493, \beta_O \to 0.488$), pero satura en 68% porque distintos músculos introducen desfasajes electromecánicos ($\Delta t = 3\text{--}4\text{ bines}$) y las vocales intermedias `/e/` y `/o/` requieren el modo bifásico ortogonal $g_1(t) = \dot{g}_0(t)$ (apertura mandibular seguida de retracción comisural o protrusión).
          - **Resultado en un Único Autoencoder Conv1D Ortogonal Unificado (100% Zero-Labels, $N = 1,683$):** Con Desconvolución Espaciotemporal de Wiener y blanqueo fraccionario ($\beta = 0.38$) *out-of-sample*, el único `ConvOrthogonalAutoencoder` salta de **47.71% GMM** (crudo) a **80.51% GMM Húngaro / 79.44%—82.89% KNN** sin canibalizar ninguna vocal (`/a/: 91%, /e/: 74%, /i/: 80%, /o/: 77%, /u/: 77%`). Script: `experimento_reconstruccion_ejes_puros_biorthogonal.py`, modelo: `autoencoder_conv1d_ortogonal_ejes_puros_biorthogonal.pt`, figura: `reconstruccion_biorthogonal_ejes_puros_multisensor.png`.

---

## 6. Infraestructura de Memoria y Tokens
- **Memoria de Trabajo:** `.agents/MEMORIA_ACTIVA.md` (~50 líneas, ~600 tokens al arrancar).
- **Historial Completo:** `.agents/historial/bitacora_completa_hitos_001_al_138.md` indexado por `.agents/INDICE_HITOS.md`.
- **Reglas del Proyecto:** `.agents/AGENTS.md` compactado a 5.3 KB (ahorro del 81.4% de tokens base por turno).
- **Auditoría Local:** Script ejecutable `.agents/auditar_tokens_sesion.py` para medir consumo de tokens en conversaciones.

