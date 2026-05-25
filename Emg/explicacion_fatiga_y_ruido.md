# Métricas de Evaluación EMG: Caída SNR y Deriva de Ruido

Durante las mediciones repetitivas o sostenidas de electromiografía (EMG), no solo importa la amplitud máxima que el paciente puede alcanzar, sino también cómo se comporta el músculo y el hardware a lo largo del tiempo. 

Para medir esto objetivamente, el sistema calcula dos métricas que comparan el **inicio** de la medición contra el **final** de la misma: la **Caída de SNR** y la **Deriva de Ruido**.

Ambas métricas utilizan los **Cuartiles Extremos (25%)**. Es decir, si una prueba tiene 40 pulsos, el sistema compara el promedio de los primeros 10 pulsos contra el promedio de los últimos 10 pulsos.

---

## 1. Caída de SNR (Evaluación de Fatiga Muscular)

La **Caída de SNR** (Signal-to-Noise Ratio Drop) cuantifica la pérdida de fuerza o reclutamiento muscular a medida que avanza la prueba.

### ¿Cómo se calcula?
1. Se obtiene el valor SNR de cada pulso de la medición.
2. `snr_start` = Promedio del SNR en el primer 25% de los pulsos.
3. `snr_end` = Promedio del SNR en el último 25% de los pulsos.
4. **Fórmula:** `((snr_start - snr_end) / snr_start) * 100`

### Interpretación
* **Valores Positivos (ej. +15% a +30%):** Indican fatiga muscular normal. El paciente hizo un 15% a 30% menos de fuerza (en relación al ruido) al final de la prueba que al principio.
* **Valores Cercanos a 0%:** El paciente mantuvo una fuerza constante durante toda la prueba, sin mostrar signos de agotamiento.
* **Valores Negativos (ej. -10%):** Indican un efecto de "calentamiento" o aprendizaje. El paciente terminó contrayendo el músculo con más fuerza o de forma más eficiente al final de la prueba.

---

## 2. Deriva de Ruido (Evaluación de Hardware y Entorno)

La **Deriva de Ruido** (Noise Drift) cuantifica cómo cambia el piso de ruido (interferencia, estática, mal contacto) a lo largo de la prueba. Su objetivo principal es evaluar la calidad de los electrodos y el blindaje de los cables frente al movimiento.

### ¿Cómo se calcula?
1. Se mide el nivel RMS del ruido base en los espacios de descanso (inter-pulso) de la señal.
2. `noise_start` = Promedio del ruido durante el primer 25% de la prueba.
3. `noise_end` = Promedio del ruido durante el último 25% de la prueba.
4. **Fórmula:** `((noise_end / noise_start) - 1.0) * 100`

### Interpretación
* **Valores cercanos a 0% (ej. -5% a +5%):** Excelente estabilidad. Los cables tienen buen blindaje electromagnético y los electrodos se mantuvieron firmemente pegados a la piel a pesar del movimiento.
* **Valores altos positivos (ej. +50% a +200%):** Indican degradación de la señal. Posibles causas:
  * El cable actúa como antena y fue acumulando estática de los movimientos.
  * El sudor o el movimiento despegaron parcialmente los electrodos de la piel, aumentando la impedancia y capturando más interferencia.
  * Un cable no mallado que fue moviéndose cerca de una fuente de alimentación de 50Hz.

---

## Conclusión Analítica Conjunta

Al observar estas dos métricas juntas, podemos aislar variables. Si un paciente tiene una **Caída de SNR del 40%**, podríamos pensar que se fatigó mucho. Pero si al mismo tiempo la **Deriva de Ruido es del +150%**, sabemos que el SNR colapsó no porque el músculo fallara, sino porque el ruido (el divisor en la fórmula de SNR) creció desproporcionadamente debido a un problema de hardware.