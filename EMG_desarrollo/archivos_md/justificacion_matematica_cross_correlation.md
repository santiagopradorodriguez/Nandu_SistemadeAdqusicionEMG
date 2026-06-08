# Justificación Matemática: Alineación de Señales EMG mediante Correlación Cruzada (Cross-Correlation)

## 1. El Problema de la Coarticulación y la Variabilidad Humana
Durante las sesiones de adquisición de datos para Machine Learning (específicamente en el protocolo "Secuencia Continua" guiado por el metrónomo de Nandu AutoForge), se espera que el sujeto sincronice su contracción muscular (fonema) exactamente con el "click" temporal del sistema. 

Sin embargo, debido a la fisiología del tiempo de reacción humano (latencia neuromotora) y a la coarticulación (entrelazamiento fluido de gestos sucesivos), la máxima densidad de energía mioeléctrica raras veces coincide de forma perfecta con el punto cero matemático del metrónomo. Esto genera "jitter" o desfasaje en los tensores de datos extraídos. Para una Red Neuronal (como un Autoencoder Convolucional 1D), este jitter temporal actúa como ruido posicional, dificultando la extracción de características morfológicas invariantes en el tiempo.

## 2. La Solución Matemática: Teorema de Correlación Cruzada
Para mitigar esto, hemos implementado una técnica de **Alineación Master-Slave** (presente en la Fase 4 del Pipeline DSP) basada en la Correlación Cruzada Discreta.

La correlación cruzada es una medida de similitud entre dos señales como función del desplazamiento temporal de una respecto a la otra. Matemáticamente, para dos secuencias discretas y reales \(x[n]\) (Master) e \(y[n]\) (Slave), se define como:

$$(x \star y)[k] = \sum_{n=-\infty}^{\infty} x[n] \cdot y[n+k]$$

Donde \(k\) es el "lag" o retardo. El valor máximo de esta función indica el desplazamiento temporal exacto en el cual las dos señales están matemáticamente mejor alineadas en términos de energía compartida.

## 3. Implementación en el Ecosistema Ñandú LSD
1. **Selección del Master**: Se establece un canal de referencia (comúnmente el Canal 0, de mayor amplitud, o el promedio inter-canal de una ventana de referencia perfecta) como la señal "Master" \(x[n]\).
2. **Aplicación de Envolvente**: En lugar de correlacionar la señal cruda oscilatoria (cuyas fases positivas y negativas de las unidades motoras cancelarían energía), se correlacionan las **Envolventes RMS**. Esto asegura que se compare exclusivamente la topología de activación de la contracción.
3. **Cálculo del Desfase (\(\tau\))**: 
   $$\tau_{max} = \arg\max_k \left( \text{Corr}(Env_{Master}, Env_{Slave}, k) \right)$$
   Utilizando `scipy.signal.correlate` con el modo de convolución rápida (FFT), obtenemos el \(\tau_{max}\) en O(N log N).
4. **Desplazamiento**: Se aplica un _roll_ circular (`np.roll`) o un re-corte desplazado a la señal original cruda del Slave (y a todos sus canales hermanos sincrónicamente) por \(\tau_{max}\) muestras. Dado que el recorte está rodeado de ruido basal por el protocolo AutoForge, los artefactos de borde (edge-artifacts) de un _roll_ circular son asintóticamente nulos.

## 4. Beneficios para Deep Learning
Al aplicar esta alineación en la Fase 4 (Pre-procesamiento):
- El Autoencoder 1D recibe tensores centrados. El "pico" de activación del fonema ocurrirá consistentemente en el centro del array (ej. muestra 250 de 500).
- La varianza inter-clase disminuye, facilitando la separación lineal en el espacio latente.
- Se corrige biológicamente el retraso cognitivo del paciente frente al estímulo visual/auditivo.
