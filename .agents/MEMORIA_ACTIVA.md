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

---

## 6. Infraestructura de Memoria y Tokens
- **Memoria de Trabajo:** `.agents/MEMORIA_ACTIVA.md` (~50 líneas, ~600 tokens al arrancar).
- **Historial Completo:** `.agents/historial/bitacora_completa_hitos_001_al_138.md` indexado por `.agents/INDICE_HITOS.md`.
- **Reglas del Proyecto:** `.agents/AGENTS.md` compactado a 5.3 KB (ahorro del 81.4% de tokens base por turno).
- **Auditoría Local:** Script ejecutable `.agents/auditar_tokens_sesion.py` para medir consumo de tokens en conversaciones.

