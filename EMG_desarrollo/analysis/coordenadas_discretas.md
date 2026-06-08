# Coordenadas Motoras Discretas (Discrete Motor Coordinates)

Este módulo implementa el análisis computacional basado fuertemente en el paper **"Discrete Motor Coordinates for Vowel Production"** (Assaneo, Trevisan, Mindlin, 2013). Su objetivo es convertir series temporales biológicas (que varían continuamente) a un código binario o booleano que simplifique y aísle el estado de *activación* del músculo.

## ¿Por qué usar umbrales dinámicos (Piso de Ruido)?

En la metodología del paper original, cada articulador (lengua, labios, mandíbula) poseía sensores con propiedades intrínsecas diferentes, lo que ocasionaba *offsets* (valores basales) y escalas muy dispares. La misma situación se repite en la electromiografía (EMG): cada electrodo tiene una impedancia diferente con la piel, y cada músculo tiene un tamaño distinto, por lo que el "ruido de fondo" (*noise floor*) eléctrico nunca es cero perfecto ni es el mismo entre dos músculos.

Comparar todos los músculos usando un umbral fijo (e.g. 50% del máximo general) es falible porque el ruido eléctrico de un músculo mal conectado podría cruzar ese umbral accidentalmente, o un músculo débil pero activo podría quedarse siempre por debajo.

## Metodología del Código (`discrete_motor.py`)

Para solucionar esto, desarrollamos una aproximación estadística:

1. **Extracción del Ruido Inter-pulso (Offset):**
   El código toma automáticamente los bordes de cada "recorte" o pulso, los cuales corresponden fisiológicamente al momento en el que el músculo está descansando. De esa región se extraen la media $\mu$ (offset) y la desviación estándar $\sigma$ del ruido eléctrico.

2. **Supresión del Offset:**
   A la señal completa del pulso se le resta el valor $\mu$ del ruido basal. A partir de este momento, $0$ equivale a un "silencio electromiográfico" total, igualando el terreno para todos los canales.

3. **Cálculo del Umbral Estadístico Independiente:**
   En vez de un número estático, el umbral ($T$) se calcula dinámicamente para cada músculo como:
   $$ T_i = \mu_i + N \cdot \sigma_i $$
   Donde $N$ es el parámetro de **"Sensibilidad (N Std)"** que el usuario ingresa en la interfaz (por defecto: 3.0). Un valor de $N=3$ significa que cualquier punto de la señal que cruce el umbral está a 3 desviaciones estándar por encima del ruido. Estadísticamente, la probabilidad de que el ruido basal alcance esa amplitud por azar es menor al 0.3%, garantizando matemáticamente que se trata de actividad muscular.

4. **Binarización (El Espacio Discreto):**
   Si el valor pico del pulso sin offset supera al umbral dinámico $T_i$, ese canal entra al estado activo (`1`). Si se mantiene por debajo, es inactivo (`0`). Al cruzar el estado de todos los músculos seleccionados para ese pulso, el software extrae el "Código Binario" o la *coordenada motora* para esa iteración.

5. **Gráfica Resultante:**
   El código normaliza de manera cosmética la señal por su propio valor máximo absoluto solo a fines de que todas quepan perfectamente en un gráfico entre `0` y `1`. Se grafican en rojo los umbrales independientes y en azul la línea base de offset removido. Finalmente, en el centro de cada pulso se imprime el código binario extraído, y en el título se reporta la "coordenada" que apareció con mayor frecuencia a lo largo de toda la medición.
