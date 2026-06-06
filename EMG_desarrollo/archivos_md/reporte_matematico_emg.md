# Reporte Matemático: Extracción de Características EMG y Pruebas de Hipótesis

Este documento detalla la justificación matemática y estadística detrás de las métricas utilizadas en la interfaz de análisis de señales electromiográficas (EMG) del proyecto **Ñandú LSD**, específicamente diseñadas para la preparación de datasets orientados a Machine Learning.

## 1. Métrica de Energía: MAV (Mean Absolute Value)

En el procesamiento de señales EMG orientadas al control mioeléctrico y reconocimiento de patrones (Pattern Recognition), la amplitud bruta (Pico Máximo) es altamente inestable y susceptible a artefactos de movimiento aislados. En su lugar, el **MAV** es considerado el estimador estándar del esfuerzo muscular en el dominio del tiempo.

**Fórmula Matemática:**
$$ \text{MAV} = \frac{1}{N} \sum_{i=1}^{N} |x_i| $$

Donde:
*   $N$: Es el número total de muestras discretas en la ventana de análisis (ej. 500 ms de señal).
*   $x_i$: Es la amplitud en microvoltios ($\mu V$) de la señal EMG en el instante $i$.

> [!NOTE]
> **Justificación Biológica:** El MAV representa matemáticamente el área rectificada bajo la curva de la señal en un intervalo dado. Al integrar la señal, el MAV es directamente proporcional al número de unidades motoras reclutadas y a su frecuencia de disparo promedio, ignorando picos de ruido aislados de alta frecuencia que no representan intención neuromuscular sostenida.

## 2. Prueba Estadística: Kruskal-Wallis H-Test

Antes de entrenar redes neuronales (como Redes Convolucionales en PyTorch), es imperativo demostrar la **separabilidad estadística** de las clases (en este caso, las 5 vocales: A, E, I, O, U). 

Aunque comúnmente se utiliza el Análisis de Varianza de una vía (ANOVA), este requiere que las distribuciones cumplan el supuesto de normalidad estricta. Las características EMG extraídas en el dominio del tiempo (como el MAV) están acotadas a valores positivos y suelen presentar asimetría (skewness) hacia la derecha. Por lo tanto, aplicar ANOVA sobre variables EMG suele incurrir en errores de Tipo I.

La solución óptima es el **Test de Kruskal-Wallis**, un análogo no paramétrico del ANOVA que evalúa las diferencias en las medianas operando sobre los rangos de los datos.

**Fórmula Matemática:**
$$ H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1) $$

Donde:
*   $N$: Número total de observaciones (pulsos EMG) en todas las clases.
*   $k$: Número de grupos independientes (5 vocales).
*   $n_j$: Número de observaciones en el grupo $j$.
*   $R_j$: La suma de los rangos para el grupo $j$ (tras ordenar todos los valores de menor a mayor).

> [!IMPORTANT]
> **Regla de Decisión:** Si el p-value asociado al estadístico $H$ es menor a $\alpha = 0.05$, se rechaza la Hipótesis Nula ($H_0$). Esto comprueba de manera irrefutable que el canal muscular posee diferencias estadísticamente significativas entre las vocales, garantizando la viabilidad del canal como input para el modelo de Machine Learning.

## 3. Visualización de Separabilidad: Boxplots (Gráficos de Caja)

El Boxplot es la herramienta gráfica definitiva para complementar pruebas no paramétricas, ya que visualiza los cuartiles matemáticos sin asumir normalidad.

Anatomía de la interpretación:
*   **Caja (Rango Intercuartílico - IQR):** Representa el 50% central de las contracciones (diferencia entre el 75º y 25º percentil).
*   **Línea Central:** Es la Mediana (Percentil 50). 
*   **Bigotes (Whiskers):** Definen los límites de los datos teóricamente esperados, típicamente calculados como $\text{Cuartil} \pm 1.5 \times \text{IQR}$.
*   **Puntos (Valores Atípicos / Outliers):** Representan mediciones específicas (pulsos aislados) que se escapan del patrón normal de varianza del músculo. En bioseñales, suelen corresponder a contracciones inusualmente bruscas o a errores de medición estocásticos.

---

## Referencias y Fuentes Científicas

Las decisiones arquitectónicas tomadas para la extracción de características y el diseño estadístico están respaldadas por la literatura fundamental del procesamiento mioeléctrico:

1.  **Phinyomark, A., Phukpattaranont, P., & Limsakul, C. (2012).** *Feature reduction and selection for EMG signal classification*. Expert Systems with Applications, 39(8), 7420-7431. 
    *(Referencia principal que demuestra estadísticamente que el MAV es la característica en el dominio temporal más robusta para el reconocimiento de patrones EMG).*
2.  **Oskoei, M. A., & Hu, H. (2007).** *Myoelectric control systems—A survey*. Biomedical Signal Processing and Control, 2(4), 275-294.
    *(Literatura estándar sobre el flujo universal de Machine Learning en bioseñales, destacando la necesidad de pruebas estadísticas para la validación de features).*
3.  **Merletti, R., & Parker, P. A. (2004).** *Electromyography: Physiology, Engineering, and Noninvasive Applications*. IEEE Press.
    *(Libro de texto base que justifica la relación fisiológica entre la integral de la amplitud y el reclutamiento de unidades motoras).*
4.  **Kruskal, W. H., & Wallis, W. A. (1952).** *Use of ranks in one-criterion variance analysis*. Journal of the American Statistical Association, 47(260), 583-621.
    *(El paper original que define el H-Test como alternativa superior al ANOVA para distribuciones sesgadas).*
