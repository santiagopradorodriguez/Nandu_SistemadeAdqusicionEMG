# Cuaderno de Laboratorio: Estudio de la Fase Pre-Fono, Geometría Muscular 3D y Descarte de Información Post-Fono

**Fecha de compilación:** 2026-11-04  
**Ubicación de figuras:** `reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono/figuras/`  
**Sujeto evaluado:** Candela (Mediciones: 2026-09-01, 2026-09-15, 2026-10-28)

---

## 1. Filosofía Metodológica: ¿Por qué No Creer en los Algoritmos Supervisados Complejos?

A los algoritmos supervisados complejos (redes neuronales profundas con etiquetas, SVM con kernels RBF no lineales) no se les cree para modelar la base motora porque **no son transferibles entre personas ni entre sesiones**:
* Un clasificador supervisado memoriza las fronteras exactas y los umbrales específicos de microvoltios de esa persona en esa sesión particular (afectado por impedancia de contacto, sudor, fatiga o desplazamiento milimétrico del parche).
* En cuanto se cambia de sujeto o de sesión, esas fronteras colapsan.

Por esta razón, en el proyecto Ñandú se establecen dos principios:
1. **Regla Cardinal Zero-Labels:** El aprendizaje de la variedad, la reducción de dimensión y el agrupamiento deben operar sin etiquetas, encontrando la estructura natural gobernada por la biomecánica muscular.
2. **Las etiquetas solo se usan al final para auditar:** Se emplean únicamente para calcular una métrica externa de verificación (GMM / K-Means con asignación húngara o LDA lineal de referencia).

---

## 2. El Espacio Físico Muscular Tridimensional (2026-09-15)

Al analizar la energía RMS pre-fonatoria en la Tríada Canónica (Ch0 Digástrico, Ch1 Cigomático Mayor a 45°, Ch2 Orbicular de los labios), se descubre que los datos forman naturalmente un trípode ortogonal sin necesidad de transformaciones estadísticas:

* **Eje X:** Ch0 Digástrico $\implies$ apertura mandibular activa $\rightarrow$ **/a/**.
* **Eje Y:** Ch1 Cigomático Mayor $\implies$ tracción comisural diagonal / sonrisa $\rightarrow$ **/i/**.
* **Eje Z:** Ch2 Orbicular $\implies$ constricción y protrusión labial $\rightarrow$ **/u/** y **/o/**.
* **Plano intermedio:** Coactivación equilibrada de Ch0 y Ch1 $\rightarrow$ **/e/**.

### Rendimiento del Clasificador Lineal Directo en Espacio Físico (Sin PCA):
* **Exactitud Directa (Global):** **86.21%** (frente al 85.22% del PCA 3D oficial del Atlas).
* **Exactitud en 5-Fold Cross Validation:** **84.73%** ($\pm 2.37\%$).
* **Desglose:**
  * **/a/**: 92.68%
  * **/e/**: 82.93%
  * **/i/**: 100.00%
  * **/o/**: 75.61%
  * **/u/**: 80.49%

**Figura asociada:**  
`figuras/03_energia_3d_pre_vs_post_2026_09_15.png`  
`figuras/02_energia_2d_pre_vs_post_2026_09_15.png`

---

## 3. Ensayo Zero-Labels Ciego vs Enfoque Geométrico (Sarrazolla–Derotari)

Al evaluar el espacio de energía física de forma 100% no supervisada con **GMM / K-Means + Asignación Húngara**:
* **Exactitud Zero-Labels:** **65.02%**.
* **El problema de la densidad ciega:** La **/i/** y la **/u/** se aíslan al **100.00%** por ser polos extremos. Sin embargo, **/o/** y **/u/** comparten el mismo músculo (Orbicular) y difieren en intensidad; el agrupamiento ciego por densidad las fusiona en una sola campana continua, canibalizando a la **/o/** (2.44%).

### Por qué el enfoque geométrico (Sarrazolla–Derotari) es superior:
En lugar de forzar a un algoritmo no supervisado a "adivinar" densidades o a un modelo supervisado a sobreajustar fronteras:
1. La base de 3 ejes ortogonales ya la da la anatomía.
2. Se trazan conos espaciales y planos angulares fijos derivados de la acción muscular:
   * Dominio de $X \implies$ **/a/**
   * Dominio de $Y \implies$ **/i/**
   * Dominio de $Z \implies$ **/u/** y **/o/** (diferenciadas por umbral radial de proyección labial)
   * Sector bisectriz $X$-$Y \implies$ **/e/**
3. Al normalizar cada pulso por el supremo tricanal ($M_{\text{supremo}}$), la escala de microvoltios queda calibrada y la geometría angular se mantiene invariante entre sujetos.

---

## 4. Comparativa con Medición de Risorio (2026-09-01)

Para comprobar si el triángulo era casualidad o biomecánica, se evaluó la sesión del 2026-09-01 (Microelectrodos 5×5 mm: Digástrico + Risorio + Orbicular):

| Métrica | 2026-09-15 (Cigomático 45°) | 2026-09-01 (Risorio) |
| :--- | :---: | :---: |
| **PCA 3D Oficial Atlas** | **85.22%** | **79.56%** |
| **Físico Directo 3D Pre-Fono** | **86.21%** | **73.33%** |
| **Varianza PC1 / PC2** | PC1: 43.6% / PC2: 42.4% (Simétrico) | PC1: 66.6% / PC2: 12.5% (Asimétrico) |

### Conclusión Anatómica:
* El Risorio es puramente horizontal y comparte el plano fascial perioral con el Orbicular.
* En 09-01, la **/e/** colapsa al **54.90%** en el espacio directo porque queda atrapada entre el fondo de Risorio y Orbicular.
* El triángulo perfecto es propiedad exclusiva del **Cigomático Mayor a 45°**, que tracciona en diagonal hacia el pómulo de forma verdaderamente desacoplada.

**Figuras asociadas:**  
`figuras/04_energia_3d_2026_09_01.png`  
`figuras/05_energia_2d_2026_09_01.png`

---

## 5. El Misterio de la Pérdida de Separabilidad (2026-10-28)

Se analizó la energía Pre-Fono en las sesiones consecutivas del 28 de octubre:

1. **Exp1: Habla Horizontal Natural (LDA Pre-Fono: 56.47%):**
   * Candela articuló la **/a/** retrayendo comisuras de forma relajada sin descender la mandíbula.
   * **En el gráfico 3D:** El Digástrico quedó casi en cero ($X \approx 0$) y toda la energía de la **/a/** se fue al Cigomático (Eje Y), solapándose con **/i/** y **/e/**.
   * La nube 3D colapsó a una línea diagonal.

2. **Exp2: Apertura Vertical Forzada (LDA Pre-Fono: 75.45%, 4 min después, mismos electrodos):**
   * Al exigir apertura mandibular, el Digástrico saltó a $X > 30\,\mu\text{V}$ y el Cigomático se apagó en la **/a/**.
   * La **/a/** recuperó su vértice inferior derecho y la exactitud subió casi **20 puntos en el acto**.

**Conclusión:** La pérdida de separabilidad no fue ruido ni falla instrumental; quedó grabada en la energía pre-fono como un cambio de patrón de reclutamiento motor por cambio dinámico en la rampa motora.

**Figura asociada:**  
`figuras/06_energia_3d_1028_exp1_vs_exp2.png`

---

## 6. Descarte de la Fase Post-Fono en PCA

Al confirmar que la fase posterior al sonido es relajación/rebote mecánico sin información fonética útil, se implementó el PCA recortado exclusivamente en la ventana pre-fono:

| Parámetro | PCA Pulso Completo | PCA Recorte Pre-Fono | Ganancia Neta |
| :--- | :---: | :---: | :---: |
| **Dimensión Temporal Entrada** | 60 features ($20 \times 3$) | 30 features ($10 \times 3$) | **-50% datos redundantes** |
| **Varianza Explicada (PC1 + PC2)** | 85.9% | **88.1%** | **+2.2 pp** |
| **Varianza Explicada 3D** | 91.7% | **93.4%** | **+1.7 pp** |
| **Exactitud LDA 2D** | 76.8% | **79.8%** | **+3.0 pp** |
| **Exactitud LDA 3D** | 85.2% | **85.7%** | **+0.5 pp** |

**Efecto visual:** Las nubes de **/o/** y **/u/** se compactan drásticamente al eliminar la dispersión temporal del cierre labial posterior.

**Figura asociada:**  
`figuras/07_pca_completo_vs_prefono_2026_09_15.png`

---

## 7. Índice de Figuras Generadas en la Carpeta
Todas las figuras quedaron guardadas en resolución publication-quality (250 dpi) en:  
`reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono/figuras/`

1. `01_shm_6paneles_2026_09_15.png`: Panel de 6 figuras Solla-Hopfield-Mindlin con extractor oficial y colores canónicos.
2. `02_energia_2d_pre_vs_post_2026_09_15.png`: Espacio físico 2D (Digástrico vs Cigomático, tamaño Orbicular) Pre vs Post Fono.
3. `03_energia_3d_pre_vs_post_2026_09_15.png`: Espacio físico 3D completo en Tríada Canónica Pre vs Post Fono.
4. `04_energia_3d_2026_09_01.png`: Espacio físico 3D con Risorio (microelectrodos 5×5 mm).
5. `05_energia_2d_2026_09_01.png`: Espacio físico 2D con Risorio.
6. `06_energia_3d_1028_exp1_vs_exp2.png`: Espacio físico 3D en 10-28 demostrando el colapso por habla horizontal y recuperación vertical.
7. `07_pca_completo_vs_prefono_2026_09_15.png`: Comparativa del PCA sobre pulso completo vs ventana exclusiva Pre-Fono.
