# Efecto del Descarte de la Fase Post-Fono en PCA (2026-09-15)

Comparativa lado a lado entre alimentar el PCA con la serie temporal completa vs recortar únicamente la fase preparatoria pre-fonatoria:
* **Izquierda (PCA Clásico Atlas):** Pulso completo de 60 puntos temporales ($20 \times 3$).
* **Derecha (PCA Pre-Fono):** Recorte exclusivo de la ventana previa a la fonación (30 puntos temporales, $10 \times 3$).

---

## Panel Comparativo PCA

![PCA Completo vs PCA Pre-Fono](/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/Comparativa_PCA_Completo_vs_PCA_PreFono_2026_09_15.png)

---

## Resultados y Conclusiones:

| Métrica | PCA Pulso Completo | PCA Recorte Pre-Fono | Diferencia |
| :--- | :---: | :---: | :---: |
| **Dimensión Temporal Entrada** | 60 features ($20 \times 3$) | 30 features ($10 \times 3$) | **-50% de datos redundantes** |
| **Varianza Explicada (PC1 + PC2)** | 85.9% | **88.1%** | **+2.2 pp de varianza capturada** |
| **Varianza Explicada 3D Total** | 91.7% | **93.4%** | **+1.7 pp** |
| **Exactitud LDA 2D** | 76.8% | **79.8%** | **+3.0 pp de ganancia neta en 2D** |
| **Exactitud LDA 3D** | 85.2% | **85.7%** | **+0.5 pp** |

### Qué revela la comparación:
1. **En 2D la ganancia es notable (+3.0 pp):** Al descartar la fase de relajación y cierre labial posterior (que es casi idéntica para todas las vocales y actúa como ruido correlacionado), los primeros dos componentes principales quedan mucho más "limpios" y separan mejor las nubes de puntos.
2. **Cúmulos más densos:** La dispersión intra-clase de la **/o/** (morado) y la **/u/** (amarillo) se reduce visiblemente en el panel derecho porque se elimina la variabilidad del tiempo que tarda cada labio en relajarse post-emisión.
