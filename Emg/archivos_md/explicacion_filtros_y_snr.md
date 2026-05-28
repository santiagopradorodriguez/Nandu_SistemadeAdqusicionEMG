# Filtros, Envolventes y Cálculo de SNR en Señales EMG

Este documento explica el porqué del orden de los filtros en el procesamiento de señales electromiográficas (EMG) y cómo la presencia (o ausencia) del filtro Notch de 50 Hz afecta matemáticamente el cálculo de la Relación Señal-Ruido (SNR).

---

## 1. El Rol de los Filtros

En nuestro flujo de trabajo digital, los filtros se aplican a la **señal cruda** en el siguiente orden lógico:

1. **Filtro Pasa-Altos (ej. 20 Hz):** Elimina el "baseline wander" (desplazamientos de la línea base) causado por el movimiento mecánico de los cables, la piel o la respiración. Tras este filtro, la señal queda perfectamente centrada en el cero.
2. **Filtro Notch (50 Hz):** Elimina específicamente la interferencia electromagnética de la red eléctrica. 
3. **Filtro Pasa-Bajos (ej. 500 Hz):** Elimina el ruido estático de alta frecuencia y evita el aliasing.

---

## 2. El Problema del Offset Matemático (La Rectificación)

Para extraer la "silueta" de la fuerza muscular, se calcula la **envolvente**. El primer paso matemático de la envolvente es obtener el **valor absoluto** de la señal (`np.abs()`), lo que se conoce como rectificación.

### ¿Qué pasa si NO usamos el filtro Notch de 50 Hz?
Si la interferencia de 50 Hz de la pared llega hasta el paso de la rectificación:
1. La interferencia de 50 Hz es una onda senoidal enorme simétrica (oscila sobre y bajo el cero).
2. Al aplicarle el valor absoluto, **las partes negativas se voltean hacia arriba**.
3. Al volver todo positivo, acabas de crear matemáticamente un voltaje continuo (DC) que antes no existía.
4. **El resultado:** La envolvente de tu señal se monta sobre un "escalón" o "piso de ruido" artificialmente alto.

### ¿Por qué el filtro Pasa-Altos no elimina este offset?
Porque existe un **problema temporal en el procesamiento**:
* El Pasa-Altos de 20 Hz se aplica al principio. Como los 50 Hz de la red son mayores a 20 Hz, el filtro los deja pasar intactos.
* El "offset" nace *después*, durante la rectificación. Para cuando el offset se crea, el filtro Pasa-Altos ya hizo su trabajo y no puede volver a actuar.
* *Tampoco podemos aplicar un Pasa-Altos a la envolvente final*, porque la contracción muscular es muy lenta (1-5 Hz); si aplicamos un Pasa-Altos de 20 Hz a la envolvente, borraríamos la contracción del músculo.

**Conclusión:** El Pasa-Altos te salva de offsets *mecánicos*, pero el Notch es el único que te salva de los offsets *matemáticos* del ruido eléctrico.

---

## 3. Impacto en el Cálculo de SNR (Signal-to-Noise Ratio)

El cálculo clásico de SNR es: `SNR = Pico Máximo / Promedio del Ruido`.  
La decisión de usar o no el filtro Notch cambia por completo qué matemática debemos usar para que las comparaciones sean justas.

### Escenario A: Con Filtro Notch (El Método Clásico)
Al eliminar los 50 Hz, el único ruido que queda es "estática" aleatoria de bajo voltaje.
* El **promedio** de este ruido rectificado es un número real y muy bajito.
* **Fórmula recomendada:** Dividir la Amplitud Máxima por el Promedio del ruido. Funciona perfecto y da valores reales.

### Escenario B: Sin Filtro Notch (El Método Robusto / SNR Neto)
*Analogía:* Imagina que mides una **montaña** (músculo) que sobresale del **mar** (ruido). El **promedio** es el nivel de la marea, y la **desviación estándar** es el tamaño de las olas.

Al no usar Notch, los 50 Hz actúan como si **subieran la marea**. El promedio del ruido se vuelve un número gigantesco.
* Si usas la fórmula clásica y divides tu músculo por esa marea alta, el SNR se desploma y parece que el músculo no hizo fuerza (fisiológicamente falso).

**Fórmula recomendada (SNR Neto):**
1. **Amplitud Neta:** `(Pico Máximo) - (Promedio del Ruido)`.  
   Mides la montaña *desde la superficie del agua*, ignorando si la marea (los 50 Hz) está alta o baja.
2. **Fluctuación del Ruido:** En lugar de usar el promedio, divides por la *Desviación Estándar* del ruido (`sigma_est`). Estás midiendo únicamente el "oleaje".

`SNR Neto = Amplitud Neta / Desviación Estándar del Ruido`

Esto evalúa qué tan distinguible es el pico de la montaña por encima del oleaje aleatorio, permitiendo comparar de forma justa mediciones ruidosas contra mediciones limpias sin que el offset destruya tus estadísticas.

---

## Resumen de Mejores Prácticas
1. **Usar SIEMPRE el filtro Notch** si el objetivo principal es medir la amplitud máxima real y comparar la fuerza entre distintas sesiones.
2. Si por algún motivo experimental se debe trabajar sin Notch, utilizar el **cálculo de SNR Neto (Robusto)** apoyado en la desviación estándar para no penalizar artificialmente la señal por culpa del offset de la rectificación.