# Dinámica Pre-Fono vs Post-Fono en Secuencias Continuas De Bruijn (2026-10-30)

Evaluación en habla continua articulada bajo secuencias pseudoaleatorias De Bruijn con jittering temporal (ISI variable):
1. **Exp1 (Serie 1):** Jittering ±25% ($N=107$ pulsos limpios).
2. **Exp2 (Serie 2 - 40 Ciclos):** Jittering ±40% ($N=179$ pulsos limpios tras 700 fonaciones acumuladas).

---

## 1. Tabla Comparativa de Rendimiento

| Secuencia Evaluada | N Pulsos | LDA Directo Pre-Fono | LDA Directo Post-Fono | Caída Post-Fono | 5-Fold CV Pre-Fono | 5-Fold CV Post-Fono | Caída en Test (CV) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp1 (Serie 1 Jitter ±25%)** | 107 | **62.62%** | 47.66% | **-14.96 pp** | **59.74%** | 35.50% | **-24.24 pp** |
| **Exp2 (Serie 2 Jitter ±40%)** | 179 | **63.69%** | 33.52% | **-30.17 pp** | **59.17%** | 29.02% | **-30.15 pp** |

---

## 2. Fenómenos Fisiológicos Observados en el Espacio Físico 3D

### A. La Fase Pre-Fono Retiene la Intención Motora (~60% en Secuencia Continua):
* En una secuencia continua rápida, los fonemas se encadenan sin pausas. A pesar de la **co-articulación anticipatoria** (el músculo empieza a adoptar la forma del fonema siguiente cientos de milisegundos antes), la ventana **Pre-Fono** alcanza ~**60% de exactitud en prueba ciega (5-Fold CV)**.
* Los 3 ejes musculares (Digástrico en X, Cigomático en Y, Orbicular en Z) mantienen agrupamientos separados.

### B. El Colapso Catastrófico de la Fase Post-Fono (hasta 29% CV):
* En la **Serie 1**, el Post-Fono cae a **35.5%**.
* En la **Serie 2 (Jitter ±40%)**, el Post-Fono se desploma a **29.0% en Cross-Validation** (rozando el nivel de azar teórico del 20%).
* **¿Por qué ocurre este colapso extremo en secuencias continuas?**
  1. **Contaminación de Co-Articulación:** En habla continua no hay "descanso" post-fono; la cola de un fonema se superpone directamente con la fase preparatoria del fonema siguiente. La energía posterior no es el cierre de la vocal actual, sino la mezcla con la siguiente.
  2. **Efecto de la Fatiga Muscular:** En la Serie 2, tras 700 fonaciones, los músculos periorales no logran relajarse a tiempo y permanecen en un estado de co-contracción basal tónica indiferenciada, haciendo que toda la nube post-fono colapse en una masa confusa cerca del origen.

---

## 3. Conclusión para el Cuaderno de Laboratorio:
En habla continua, **la ventana Pre-Fono es todavía más crítica que en fonemas aislados**. Mientras que en Pre-Fono el cerebro comanda activamente la postura articulatoria requerida (logrando ~60% CV aún bajo co-articulación severa), la ventana Post-Fono en habla fluida es pura interferencia motora y decae por completo (hasta 29% CV).
