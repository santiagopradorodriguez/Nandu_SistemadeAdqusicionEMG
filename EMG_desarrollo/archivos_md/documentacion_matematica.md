# Documentación Matemática Profunda para el Procesamiento de sEMG

Este documento establece el marco teórico y matemático riguroso utilizado en el procesamiento de señales de Electromiografía de Superficie (sEMG) para el reconocimiento de habla silenciosa. Se asume un conocimiento avanzado en Procesamiento Digital de Señales (DSP) y análisis funcional.

## 1. Frecuencia de Muestreo y Teorema de Nyquist

El proceso de digitalización de la señal sEMG transforma una señal continua en el tiempo $x_c(t)$ a una secuencia discreta $x[n] = x_c(n T_s)$, donde $T_s$ es el periodo de muestreo y $f_s = 1/T_s$ la frecuencia de muestreo.

### Teorema de Muestreo de Nyquist-Shannon
Para que la señal analógica pueda ser reconstruida de forma exacta y sin solapamiento espectral (*aliasing*), la frecuencia de muestreo debe ser estrictamente mayor que el doble de la componente frecuencial máxima presente en la señal:
$$ f_s \ge 2 f_{max} $$

La señal sEMG de superficie posee la mayor parte de su energía útil concentrada en la banda de frecuencia entre $20\text{ Hz}$ y $400\text{ Hz} - 500\text{ Hz}$. En base al teorema de Nyquist, un muestreo a $1000\text{ Hz}$ sería teóricamente suficiente ($f_s = 2 \times 500\text{ Hz} = 1000\text{ Hz}$).

### Justificación del Sobremuestreo (2000 Hz o 6000 Hz)
En la práctica de sEMG de alta resolución, emplear frecuencias de $2000\text{ Hz}$ o $6000\text{ Hz}$ es un requisito por múltiples razones matemáticas y físicas:
1. **Relajación de los filtros anti-alias**: Un filtro analógico real no es ideal. Tiene una banda de transición finita. Un $f_s$ elevado permite situar el filtro de corte analógico muy por encima del límite útil de $500\text{ Hz}$, asegurando que la atenuación sea masiva en la frecuencia de doblez de Nyquist ($f_s/2$).
2. **Resolución temporal para la detección de *onsets***: Las tareas de reconocimiento de habla silenciosa exigen una precisión extrema en la localización del inicio temporal (onset) de la activación muscular. Una frecuencia de muestreo de $6000\text{ Hz}$ garantiza un espaciamiento temporal entre muestras de $\Delta t = 0.166\text{ ms}$, crucial para alineamientos finos.
3. **Distribución espectral de ruido de cuantización**: El sobremuestreo distribuye la varianza del error de cuantización $q^2/12$ en un ancho de banda mayor $[-f_s/2, f_s/2]$, lo que reduce la densidad espectral de ruido base y eleva la relación Señal a Ruido de Cuantificación (SQNR).

## 2. Filtro Notch (Rechazo de Ruido de Línea de 50 Hz)

La interferencia electromagnética de la red eléctrica es ubicua y se manifiesta primariamente como una componente aditiva a la frecuencia fundamental de la línea ($f_0 = 50\text{ Hz}$ en numerosas regiones).
El objetivo es anular la componente frecuencial en $\omega_0 = 2\pi (f_0 / f_s)$ rad/muestra, provocando la menor distorsión posible en las frecuencias adyacentes.

### Función de Transferencia en el Dominio $\mathcal{Z}$
Diseñamos un filtro Notch digital del tipo Respuesta Infinita al Impulso (IIR) con dos ceros complejos conjugados situados exactamente en la circunferencia unidad en el ángulo $\pm\omega_0$, y dos polos cercanos en el mismo ángulo.

Los ceros garantizan una atenuación de ganancia cero en la frecuencia exacta, y los polos empujan fuertemente la ganancia hacia $1$ fuera de la banda estrecha, controlando el factor de calidad $Q$.
$$ H(z) = \frac{(z - e^{j\omega_0})(z - e^{-j\omega_0})}{(z - r e^{j\omega_0})(z - r e^{-j\omega_0})} = \frac{1 - 2\cos(\omega_0)z^{-1} + z^{-2}}{1 - 2r\cos(\omega_0)z^{-1} + r^2 z^{-2}} $$

### Análisis de la Fase y el Polo $r$
El radio de los polos $r$ ($0 \ll r < 1$) define el ancho de banda a $-3\text{ dB}$, $\Delta\omega$, de la muesca:
$$ \Delta\omega \approx 2(1 - r) $$
Un valor típico de $r = 0.99$ proporciona un rechazo extremadamente angosto. No obstante, en la vecindad del corte se genera un artefacto de distorsión de fase no lineal. Dado que la linealidad de la fase es importante para preservar la morfología del pulso EMG, este filtro frecuentemente se aplica mediante una convolución bidireccional (*zero-phase filtering*, i.e., `filtfilt`), logrando:
$$ |H_{efectiva}(e^{j\omega})| = |H(e^{j\omega})|^2 \quad \text{y} \quad \angle H_{efectiva}(e^{j\omega}) = 0 $$

## 3. Filtrado Pasa-Banda de Butterworth

La sEMG está contaminada por derivas de línea base (movimiento del electrodo, variaciones en el contacto piel-electrodo, de $\approx 0 - 20\text{ Hz}$) y ruido de alta frecuencia que no se asocia al disparo de las unidades motoras. Para esto, se emplea un filtro pasa-banda, comúnmente Butterworth debido a su respuesta "máximamente plana" (*maximally flat*) en la banda de paso.

### Formulación Matemática
Para un filtro analógico paso-bajo de orden $N$, la magnitud al cuadrado de la respuesta en frecuencia es:
$$ |H(\Omega)|^2 = \frac{1}{1 + \left(\frac{\Omega}{\Omega_c}\right)^{2N}} $$
Los polos de la función de transferencia $H(s)H(-s)$ se encuentran equiespaciados en el círculo izquierdo del plano $s$:
$$ s_k = \Omega_c e^{j\left( \frac{\pi(2k+1)}{2N} + \frac{\pi}{2} \right)} \quad \text{para } k = 0, 1, \dots, 2N-1 $$

### Transformación Bilineal al Dominio Digital
El filtro continuo se mapea al dominio digital utilizando la transformación bilineal para evitar *aliasing* de fase y magnitud:
$$ s = \frac{2}{T_s} \frac{1 - z^{-1}}{1 + z^{-1}} $$
Este mapeo conforma toda la frecuencia $\Omega$ del plano $s$ ($-\infty$ a $\infty$) al círculo unidad en el plano $z$ ($-\pi$ a $\pi$), pero introduce una deformación no lineal del eje frecuencial (warping):
$$ \Omega = \frac{2}{T_s} \tan\left(\frac{\omega}{2}\right) $$
Para el diseño del filtro pasa-banda (ej. $20\text{ Hz} - 500\text{ Hz}$), las frecuencias de corte digitales discretas $\omega_{c1}$ y $\omega_{c2}$ son pre-warpadas, se sintetiza el prototipo analógico, se transforma a pasa-banda con mapeo espectral de frecuencia continua, y finalmente se aplica la transformación bilineal para deducir los coeficientes $b_k, a_k$ de la ecuación de diferencias diferencial:
$$ y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k] $$

## 4. Enventanado con Metrónomo (Alineación Temporal)

Para tareas de aprendizaje automático basadas en eventos predefinidos (como en experimentos sincronizados con un metrónomo o audios), las señales continuas $x[n]$ deben ser segmentadas en "épocas".

### Definición Matemática de Alineación
Supongamos un conjunto de marcas de tiempo de los "clicks" del metrónomo definidas como un conjunto de índices $\{N_1, N_2, \dots, N_K\}$. Para cada estímulo de disparo $N_i$, definimos una ventana de extracción asimétrica o simétrica, dada por un offset negativo $\Delta_{pre}$ y un offset positivo $\Delta_{post}$ (medidos en muestras).

La i-ésima época aislada $e_i[m]$ de longitud $L = \Delta_{pre} + \Delta_{post}$ se define formalmente como la función de correlación temporal truncada o simplemente mediante la ventana indicadora (pulso rectangular):
$$ w[n] = \Pi\left( \frac{n - (N_i - \Delta_{pre} + \frac{L}{2})}{L} \right) $$
$$ e_i[m] = x[N_i - \Delta_{pre} + m] \quad \text{para } m = 0, 1, \dots, L-1 $$

Esta extracción es crítica porque las redes Autoencoder/PyTorch asumen tensores matriciales fijos. Para que los perfiles morfodinámicos de la activación muscular coincidan latencia a latencia (salvo el *jitter* inherente humano o tiempo de reacción fisiológico), el origen de tiempos local $m=0$ siempre representa una fase constante de expectativa cognitiva con respecto al clic.

## 5. Rectificación y Envolvente RMS (Energía de la Señal)

La señal sEMG es estocástica, bipolar y de media nula ($\mathbb{E}[x[n]] \approx 0$). Para extraer la modulación de amplitud que correlaciona de manera isométrica o cinemática con la fuerza muscular, se extrae su envolvente.

### Rectificación de Onda Completa
La no-linealidad básica para transformar oscilaciones AC en una señal DC variante es tomar el valor absoluto:
$$ x_{rect}[n] = |x[n]| $$

### Envolvente del Valor Cuadrático Medio (RMS)
La técnica matemáticamente más estable para la estimación robusta de la potencia del reclutamiento de unidades motoras es la ventana móvil RMS. Dado un tamaño de ventana de $W$ muestras, el $\text{RMS}[n]$ se define como el estimador de varianza local:
$$ \text{RMS}[n] = \sqrt{ \frac{1}{W} \sum_{k=0}^{W-1} (x[n-k])^2 } $$
Esta operación equivale a pasar el operador al cuadrado de la señal a través de un filtro FIR Promediador Móvil (Filtro de media, que es óptimo en sentido de máxima verosimilitud bajo ruido blanco gaussiano estacionario) y luego aplicar una transformación de raíz cuadrada no-lineal:
$$ y[n] = x^2[n] * h_{MA}[n] $$
$$ \text{RMS}[n] = \sqrt{y[n]} $$
donde $h_{MA}[n] = \frac{1}{W} \text{ para } n \in [0, W-1]$. Las propiedades frecuenciales de $h_{MA}[n]$ exhiben una respuesta pasa-bajo en forma de seno cardinal discreto (*Dirichlet kernel*), que suaviza la señal para alimentar al clasificador.

## 6. Correlación Cruzada (Análisis de Similitud Morfológica)

En la detección de plantillas de activación muscular repetitivas o la identificación de *cross-talk* (interferencia mutua entre electrodos adyacentes), se emplea el análisis de correlación cruzada determinística discreta.

### Formulación Discreta de Energía Infinita / Épocas
Sean dos ventanas sEMG extraídas, $x[n]$ e $y[n]$, de longitud $L$. La correlación cruzada se define como:
$$ R_{xy}[m] = \sum_{n=0}^{L-1} x[n] y[n-m] $$
Donde $m$ representa el retraso (*lag*).

### Normalización (Coeficiente de Correlación de Pearson)
Para independizar la métrica de la escala de energía global o de la impedancia del tejido, se normaliza por el producto de las normas $\ell_2$:
$$ \hat{R}_{xy}[m] = \frac{R_{xy}[m]}{\sqrt{ \left(\sum_{n=0}^{L-1} x^2[n]\right) \left(\sum_{n=0}^{L-1} y^2[n-m]\right) }} $$
La función resultante se evalúa para $m \in [-M, M]$. El máximo de $\hat{R}_{xy}[m]$ proporciona el retraso temporal exacto entre ambas activaciones musculares musculares e indica de manera cuantitativa (donde $|\hat{R}_{xy}| \le 1$ por la desigualdad de Cauchy-Schwarz) la sincronía fisiológica del patrón motor a nivel de canales.

## 7. Cálculos de SNR (Relación Señal a Ruido) y Penalización de Ruido de Línea

La viabilidad de los modelos Deep Learning subsecuentes reside inexorablemente en la calidad intrínseca del tensor sEMG, siendo la Relación Señal a Ruido (SNR) el axioma cardinal de auditoría.

### Definición Base de Potencias
Sea un fragmento de señal clasificado como reposo (línea base) $x_{base}[n]$ de longitud $N_{base}$ y un fragmento de activación $x_{act}[n]$ de longitud $N_{act}$.
Estimadores insesgados de la potencia:
$$ P_{ruido} = \sigma_{base}^2 = \frac{1}{N_{base}-1} \sum_{n=0}^{N_{base}-1} \left(x_{base}[n] - \mu_{base}\right)^2 $$
$$ P_{total} = \frac{1}{N_{act}} \sum_{n=0}^{N_{act}-1} x_{act}^2[n] $$

Asumiendo aditividad ortogonal e incorrelación matemática entre el ruido basal y la señal mioeléctrica endógena, el teorema de Pitágoras estocástico postula:
$$ P_{total} \approx P_{se\tilde{n}al} + P_{ruido} \implies P_{se\tilde{n}al} = P_{total} - P_{ruido} $$

### SNR Acumulada
La métrica estandarizada se expresa en decibelios (dB) referidos a la potencia:
$$ \text{SNR}_{ac} = 10 \log_{10} \left( \frac{\max(0, P_{total} - P_{ruido})}{P_{ruido}} \right) $$

### Penalización Específica del Armónico 50 Hz
Para sancionar rigurosamente y auditar artefactos incontrolados de ruido eléctrico que escapan de un filtro Notch imperfecto, introducimos una penalización espectral.
Aplicando la Transformada Discreta de Fourier (DFT):
$$ X_{act}[k] = \sum_{n=0}^{N_{act}-1} x_{act}[n] e^{-j \frac{2\pi}{N_{act}} k n} $$
La densidad espectral local (Periodograma de Schuster) es $S_{act}[k] = \frac{1}{N_{act}} |X_{act}[k]|^2$. Identificamos el bin o vecindad correspondiente a $k_{50} = \lfloor N_{act} \cdot \frac{50}{f_s} \rceil$.
Extraemos la potencia espuria residual:
$$ P_{residual}^{50Hz} = \sum_{k \in \{ k_{50}-\delta, \dots, k_{50}+\delta \} } S_{act}[k] $$

Reestructuramos el modelo de la varianza considerando que una porción notable de $P_{total}$ puede deberse fraudulentamente al ruido inductivo. La SNR Auditora rigurosa descontará este espectro del numerador y lo confinará netamente al denominador:
$$ \text{SNR}_{robusto} = 10 \log_{10} \left( \frac{\max(0, P_{total} - P_{ruido} - P_{residual}^{50Hz})}{P_{ruido} + \alpha P_{residual}^{50Hz}} \right) $$
Donde $\alpha \ge 1$ sirve como multiplicador de penalización de Lagrange (factor de castigo), garantizando que las trazas de datos que exhiben pobre rechazo de línea base sean flageladas cuantitativamente de cara a la etapa de PyTorch.
