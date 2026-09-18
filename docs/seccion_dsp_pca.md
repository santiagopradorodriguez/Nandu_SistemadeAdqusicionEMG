# Reporte Matemático: Construcción de la Matriz $X$ para PCA

En esta sección se detalla el procesamiento de señales digitales (DSP) empleado en el script `extractor_features.py` para construir la matriz de características $X$, la cual alimenta al algoritmo de Análisis de Componentes Principales (PCA). El proceso se realiza sobre ventanas de señal (pulsos) extraídas de 3 canales de electromiografía de superficie (sEMG).

Sea $s_i^{(c)}(t)$ la señal sEMG discreta correspondiente al pulso (o ventana) $i$-ésimo en el canal $c \in \{0, 1, 2\}$, de longitud $T$ muestras. A partir de cada señal, se puede extraer un vector representativo mediante uno de los siguientes tres métodos:

## 1. Amplitud Máxima (Max Amp)
Este método extrae el valor máximo de la amplitud de la señal (frecuentemente tras una normalización) dentro de la ventana de tiempo. Matemáticamente, el escalar representativo para el canal $c$ y pulso $i$ está dado por:

$$ v_i^{(c)} = \max_{t} \left| \frac{s_i^{(c)}(t)}{\| s_i^{(c)} \|_{\infty}} \right| $$

En este caso, la característica es un valor único, por lo que $v_i^{(c)} \in \mathbb{R}^1$. Este enfoque simplificado es útil para detectar la intensidad pico del pulso sEMG.

## 2. Envolvente (Envelope)
El método de envolvente busca capturar la variación de la amplitud de la señal en el tiempo, eliminando las oscilaciones de alta frecuencia. En DSP, esto se logra rectificando la señal y aplicando un filtro paso bajo (o un promedio móvil sin artefactos de borde). Para asegurar una dimensionalidad constante antes de entrar al modelo, la envolvente resultante se recorta, interpola o reduce (downsampling) a un número fijo de $M$ puntos (por ejemplo, $M = 100$).

Sea $E_i^{(c)}(t) = \text{LPF}(|s_i^{(c)}(t)|)$ la envolvente de la señal. El vector de características es la serie temporal de la envolvente convertida en un vector:

$$ v_i^{(c)} = \begin{bmatrix} E_i^{(c)}(t_1), E_i^{(c)}(t_2), \dots, E_i^{(c)}(t_M) \end{bmatrix}^T \in \mathbb{R}^M $$

Este método conserva la forma de onda de la activación muscular en el dominio del tiempo, y al mantener una longitud fija, asegura propiedades estacionarias requeridas en etapas posteriores.

## 3. Transformada de Fourier de Corto Alcance (STFT)
La STFT proporciona una representación tiempo-frecuencia del pulso sEMG. Se calcula la STFT de $s_i^{(c)}(t)$, obteniendo una matriz espectrograma $S_i^{(c)}(f, \tau)$, donde $f$ son los bines de frecuencia y $\tau$ los pasos de tiempo. 

Para utilizar esta representación bidimensional estructurada en PCA, se toma la magnitud (y a menudo se descartan las bandas como ruido de línea a 50Hz) y se aplana (operación `flatten()`) en un vector unidimensional. Si el espectrograma (o su magnitud) tiene dimensiones $F \times T'$, el vector resultante tendrá dimensión $K = F \times T'$.

$$ v_i^{(c)} = \text{flatten} \left( \left| S_i^{(c)}(f, \tau) \right| \right) \in \mathbb{R}^K $$

## Concatenación de Canales y Construcción de la Matriz $X$

Independientemente del método elegido (Max Amp, Envolvente o STFT), supongamos que el vector de características extraído para un canal $c$ y pulso $i$ tiene dimensión $d$ (donde $d=1$ para Max Amp, $d=M$ para Envolvente, y $d=K$ para STFT).

Dado que el sistema Ñandú adquiere señales simultáneamente de 3 canales, se concatena la información de los canales $c=0$, $c=1$ y $c=2$ para formar un único vector de características $x_i$ que describe completamente el pulso $i$:

$$ x_i = \begin{bmatrix} v_i^{(0)} \\ v_i^{(1)} \\ v_i^{(2)} \end{bmatrix} \in \mathbb{R}^D $$

donde la dimensión total del vector es $D = 3 \times d$. El vector $x_i$ representa una única observación en el espacio de características de alta dimensión.

Finalmente, si el dataset contiene un total de $N$ pulsos extraídos (ventanas), se apilan los vectores $x_i$ transpuestos para formar la matriz de diseño o matriz de datos $X$:

$$ X = \begin{bmatrix} x_1^T \\ x_2^T \\ \vdots \\ x_N^T \end{bmatrix} \in \mathbb{R}^{N \times D} $$

Esta matriz $X_{N \times D}$ es la entrada estándar para el algoritmo PCA. Antes del ajuste del modelo, las columnas de $X$ deben estar centradas en cero, garantizando que el dataset este normalizado y listo tanto para la extracción de componentes principales como para su posterior ingesta en modelos de Deep Learning en PyTorch (ej. Autoencoders).
