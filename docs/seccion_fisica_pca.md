# Teoría del PCA Aplicado a Señales EMG Faciales

## 1. Fundamentos Matemáticos del Análisis de Componentes Principales (PCA)

El Análisis de Componentes Principales (PCA) es una técnica de transformación lineal que proyecta un conjunto de datos multidimensional en un nuevo sistema de coordenadas ortogonal. En este nuevo espacio, las variables (los "componentes principales") no están correlacionadas y están ordenadas según la cantidad de varianza que explican de los datos originales.

Matemáticamente, dado un conjunto de señales centradas $X$ de dimensiones $N \times M$ (donde $N$ es el número de muestras y $M$ es el número de canales o variables), el primer paso del PCA consiste en calcular la **matriz de covarianza** $C$:

$$C = \frac{1}{N-1} X^T X$$

La matriz de covarianza captura cómo varían las distintas dimensiones (canales) de forma conjunta. Posteriormente, se realiza la descomposición en valores propios (eigendecomposition) de $C$:

$$C = V \Lambda V^T$$

Donde $\Lambda$ es una matriz diagonal que contiene los valores propios (eigenvalues, $\lambda_i$), y $V$ es la matriz cuyas columnas son los **vectores propios** (eigenvectors, $v_i$). Los valores propios indican la magnitud de la varianza capturada, mientras que los vectores propios definen las nuevas direcciones (ejes) en el espacio de los datos.

La proyección de las señales originales sobre el nuevo espacio de componentes principales se obtiene multiplicando los datos originales por la matriz de vectores propios:

$$Y = X V$$

En este espacio proyectado $Y$, el primer componente principal ($PC_1$) corresponde a la dirección en el espacio multidimensional original que maximiza la varianza de los datos.

## 2. Revelando la "Sinergia Vocal" a través del PCA en Envolventes Musculares

En el contexto de la adquisición de señales EMG (Electromiografía) faciales durante la fonación, contamos con múltiples canales que registran la actividad de diferentes grupos musculares simultáneamente. Al aplicar PCA sobre las envolventes temporales de estos canales, estamos analizando matemáticamente la correlación implícita entre ellos.

La articulación vocal es un proceso complejo que requiere la coordinación de múltiples músculos. Esta coordinación se refleja en la **Sinergia Vocal**: la co-activación estructurada y predecible de grupos musculares.

Cuando aplicamos PCA a la matriz formada por las envolventes de los canales EMG, la matriz de covarianza evalúa qué tan correlacionadas están las activaciones musculares en el tiempo. Si existe una fuerte sinergia vocal (una co-contracción coordinada de los músculos para producir un sonido específico, como una vocal), una gran parte de la varianza total de la señal será explicada por el primer componente principal ($PC_1$).

El eigenvector correspondiente a $PC_1$ proporciona los "pesos" o contribuciones relativas de cada canal muscular a este patrón de activación conjunta. Por lo tanto, el $PC_1$ extrae la dinámica temporal compartida subyacente a la fonación, purificando la señal de las variaciones o ruidos específicos y aislados de cada canal individual.

## 3. La Alineación "Master-Slave" como Requisito Matemático Obligatorio

Para que la matriz de covarianza refleje correctamente las correlaciones fisiológicas subyacentes, es un requisito matemático estricto que los eventos a analizar estén **alineados en fase** en el dominio temporal. 

El PCA asume que la varianza compartida ocurre simultáneamente en las distintas variables (canales). Si los pulsos de activación muscular correspondientes a un mismo evento fonético presentan desfasajes temporales entre repeticiones o canales (jitter), la matriz de covarianza se "difuminará". La varianza debida a la desalineación temporal dominará sobre la varianza debida a la amplitud y forma de la envolvente, lo que resultará en componentes principales que explican el error de fase en lugar de la sinergia muscular.

Para resolver este problema, se implementa la alineación **"Master-Slave"**:
1. **Master (Acústico)**: Se utiliza la señal del micrófono acústico, que tiene una delimitación temporal extremadamente precisa del evento sonoro, para detectar el inicio, máximo o centro del pulso fonético.
2. **Slave (EMG)**: Las envolventes de los canales musculares se segmentan y se centran temporalmente tomando como referencia exacta el evento detectado en el canal acústico.

Esta alineación garantiza que las envolturas musculares de todos los canales estén perfectamente superpuestas en el tiempo para el mismo evento articulatorio (fase cero relativa). Solo bajo esta condición de alineación temporal estricta, la covarianza calculada por el PCA refleja puramente la co-activación espacial y la sinergia de amplitud de los músculos faciales, permitiendo que la proyección en los vectores propios revele los verdaderos patrones biomecánicos subyacentes.
