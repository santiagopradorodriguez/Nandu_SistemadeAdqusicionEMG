# Evaluación Experimental de la Fase Post-Fono: Mediciones Inseparables (10-28) y Efecto del Masetero (09-23)

**Preguntas de investigación:**
1. En mediciones donde la separabilidad se perdió (10-28 Exp1 Habla Horizontal), ¿se gana información en la fase Post-Fono o sigue degradada?
2. En mediciones con alto LDA donde el Masetero participa en el cierre mandibular post-emisión (09-23 Config1), ¿la información fonética sigue siendo mayor Pre-Fono que Post-Fono?

---

## 1. Tabla Comparativa General: Pre-Fono vs Post-Fono

| Medición Evaluada | N Pulsos | LDA Directo Pre-Fono | LDA Directo Post-Fono | Delta (Pre - Post) | 5-Fold CV Pre-Fono | 5-Fold CV Post-Fono |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **10-28 Exp1 (Habla Horizontal)** | 170 | **56.47%** | 45.88% | **+10.59 pp** | **53.53%** | 34.71% |
| **10-28 Exp2 (Apertura Vertical)** | 167 | **75.45%** | 64.67% | **+10.78 pp** | **73.64%** | 64.10% |
| **09-23 Config1 (Masetero + Doble Orbicular)** | 72 | **76.39%** | 59.72% | **+16.67 pp** | **72.19%** | 51.43% |
| **09-25 Bloque 4 (Masetero + Digástrico + Orbicular)** | 44 | **72.73%** | 63.64% | **+9.09 pp** | **63.61%** | 56.94% |

---

## 2. Hallazgos en Mediciones Inseparables (10-28 Exp1)

* **En Post-Fono no hay recuperación:** La exactitud cae de **56.47% (Pre-Fono)** a **45.88% (Post-Fono)**, y en validación cruzada 5-Fold se desploma al **34.71%** (apenas por encima del azar del 20%).
* **Explicación fisiológica:** Al haber articulado de forma horizontal con el Cigomático encendido en todas las vocales, la relajación posterior es un simple retorno elástico de las mejillas hacia el reposo. Todos los puntos colapsan al origen sin ninguna firma fonética residual.

---

## 3. Hallazgos en el Masetero (09-23 Config1)

El Masetero es un músculo masticatorio y oclusor: su activación natural máxima ocurre al **cerrar la mandíbula** tras pronunciar una vocal abierta:
* A pesar de tener un pico de contracción visible al final del pulso, **la información fonética útil sigue siendo mucho mayor Pre-Fono (+16.67 pp de ventaja)**:
  * **Pre-Fono:** **76.39%** (CV: 72.19%).
  * **Post-Fono:** **59.72%** (CV: 51.43%).
* **¿Por qué el pico del masetero post-fono no separa vocales?**  
  Porque el cierre de la boca ocurre para **todas** las vocales una vez que terminó la fonación. El masetero aprieta la mandíbula para regresar a la posición basal ocluyendo los molares sin importar si se dijo `/a/`, `/e/` o `/i/`. Por ende, ese pico es un **artefacto mecánico de oclusión general**, no una firma fonética vocalizada.

---

## 4. Figura Asociada en Alta Resolución
* `reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono/figuras/08_evaluacion_post_fono_1028_y_masetero.png`
