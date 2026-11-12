# Secuencias Continuas De Bruijn JUNTAS (2026-10-30: Serie 1 + Serie 2)

Evaluación de los dos experimentos continuos combinados en un único espacio físico 3D ($N=286$ pulsos limpios):
* **Serie 1 (Exp1):** Jittering ±25% ($N=107$).
* **Serie 2 (Exp2):** Jittering ±40%  ($N=179$).

---

## 1. Tabla Comparativa de Rendimiento Conjunto

| Fase Temporal | Exactitud LDA Directo | Exactitud 5-Fold Cross-Validation | Comportamiento en el Espacio Físico 3D |
| :--- | :---: | :---: | :--- |
| **PRE-FONO 3D** | **63.29%** | **61.89%** | **Estructura Conservada:** Ejes ortogonales poblados (/a/ en Digástrico, /i/ en Cigomático, /u/ y /o/ en Orbicular). |
| **POST-FONO 3D** | **39.86%** | **34.26%** | **Colapso Masivo:** Caída de **-27.6 pp en CV**. Nube apelmazada contra el origen por co-articulación continua. |
| **Diferencia Neta** | **+23.43 pp** | **+27.63 pp** | Pre-Fono casi duplica la exactitud del Post-Fono en prueba ciega. |

---

## 2. Hallazgos al Unificar Ambas Series:

1. **La consistencia de Pre-Fono (61.9% CV en N=286):**
   - Al unir ambas series, la cantidad de datos se casi triplica respecto a la Serie 1 sola.
   - Lejos de degradarse en la Serie 2, **la exactitud de Pre-Fono sube a 61.89% en Cross-Validation** (convergencia estadística robusta).
   - Esto demuestra que la orden motora previa al sonido es un atractor biomecánico estable a lo largo de toda la sesión continua.

2. **La degradación de Post-Fono se profundiza (34.3% CV):**
   - En el gráfico Post-Fono derecho, todas las vocales quedan comprimidas en un nudo cerca del origen.
   - En habla continua rápida, al terminar un fonema el aparato muscular no regresa a cero, sino que ya está ejecutando la co-articulación anticipatoria del siguiente fonema. El post-fono es interferencia acumulativa.

---

## 3. Figura Asociada
* `reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono/figuras/11_secuencias_continuas_1030_JUNTAS_pre_vs_post_fono.png`
