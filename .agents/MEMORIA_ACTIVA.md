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

---

## 6. Infraestructura de Memoria y Tokens
- **Memoria de Trabajo:** `.agents/MEMORIA_ACTIVA.md` (~50 líneas, ~600 tokens al arrancar).
- **Historial Completo:** `.agents/historial/bitacora_completa_hitos_001_al_138.md` indexado por `.agents/INDICE_HITOS.md`.
- **Reglas del Proyecto:** `.agents/AGENTS.md` compactado a 5.3 KB (ahorro del 81.4% de tokens base por turno).
- **Auditoría Local:** Script ejecutable `.agents/auditar_tokens_sesion.py` para medir consumo de tokens en conversaciones.

