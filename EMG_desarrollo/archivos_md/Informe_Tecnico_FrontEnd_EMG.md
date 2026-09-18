<p align="center">
  <img src="imagenes/logo_nandu_lsd.png" alt="Logo Ñandú LSD" width="160">
</p>

# Informe Técnico: Optimización del Front-End sEMG con AD620

**Proyecto:** Ñandú sEMG -- Laboratorio de Sistemas Dinámicos  
**Fecha:** Septiembre 2026  
**Documento:** `Informe_Tecnico_FrontEnd_EMG.md`  

---

## 1. Resumen Ejecutivo y Diagnóstico del Circuito Actual

El sistema analógico de adquisición sEMG implementado originalmente utiliza un amplificador de instrumentación AD620 alimentado con baterías de $\pm 9\,\text{V}$, programado con una resistencia de ganancia nominal $R_G = 100\,\Omega$ ($G \approx 495$) y un filtro pasa-altos pasivo colocado inmediatamente en las entradas diferenciales (capacitores serie de $33\,\text{nF}$ y resistencias a tierra de $100\,\text{k}\Omega$).

Durante las mediciones se identificó una hipersensibilidad crítica: al producirse un despegue parcial de los parches faciales o al gesticular, el piso de ruido de línea a 50 Hz se dispara de forma abrupta hasta saturar los canales y deformar las envolventes musculares.

### 1.1 Proximidad de la Frecuencia de Corte a los 50 Hz de Línea
Con los valores comerciales de entrada ($R = 100\,\text{k}\Omega$, $C = 33\,\text{nF}$), la frecuencia de corte pasiva $f_c$ resulta:
$$f_c = \frac{1}{2\pi R C} = \frac{1}{2\pi \cdot (100 \times 10^3\,\Omega) \cdot (33 \times 10^{-9}\,\text{F})} \approx 48.23\,\text{Hz}$$

Estar situado a solo $1.8\,\text{Hz}$ de la red de $50\,\text{Hz}$ destruye la simetría del amplificador:
- En la vecindad de $f_c$, la rotación de fase alcanza $+45^\circ$ y la curva de transferencia presenta su pendiente más escarpada.
- Las tolerancias de los capacitores cerámicos ($\pm 10\%$ a $\pm 20\%$) provocan que una rama corte, por ejemplo, a $43.8\,\text{Hz}$ y la otra a $53.6\,\text{Hz}$.
- A $50\,\text{Hz}$, las dos entradas del AD620 reciben tensiones con amplitudes desiguales y fuerte desfase vectorial en el tiempo. Esta asimetría interna anula el rechazo de modo común (CMRR) y convierte el campo ambiental de red directamente en una señal diferencial espuria que el AD620 multiplica por su ganancia de $\approx 495$.

### 1.2 Aplastamiento de la Impedancia de Entrada ($Z_{\text{in}}$)
Aunque el chip AD620 posee una impedancia intrínseca de $Z_{\text{chip}} \approx 10\,\text{G}\Omega$, las resistencias de polarización a tierra conectadas en la placa quedan en paralelo con el integrado:
$$Z_{\text{in}} = R_{\text{ext}} \parallel Z_{\text{chip}} = \frac{100\,\text{k}\Omega \cdot 10\,\text{G}\Omega}{100\,\text{k}\Omega + 10\,\text{G}\Omega} \approx 100\,\text{k}\Omega$$

La norma internacional SENIAM (*Surface ElectroMyoGraphy for the Non-Invasive Assessment of Muscles*) dictamina:
- $Z_{\text{in}} > 100\,\text{M}\Omega$ para electrodos convencionales con gel conductor.
- $Z_{\text{in}} > 1000\,\text{M}\Omega$ ($1\,\text{G}\Omega$) para electrodos secos.

Operar con $100\,\text{k}\Omega$ sitúa al amplificador mil veces por debajo de la norma requerida para biopotenciales.

### 1.3 Mecanismo Físico de Despegue del Electrodo
La ecuación de Roberto Merletti y Philip Parker modela la conversión de modo común a diferencial:
$$\frac{V_{\text{out}}}{A_d} \approx V_{\text{cm}} \left( \frac{\Delta Z}{Z_i} + \frac{A_c}{A_d} \right)$$

donde:
- $V_{\text{cm}}$ es la tensión parásita de modo común inducida en el cuerpo por la red eléctrica.
- $Z_i$ es la impedancia de entrada del amplificador ($100\,\text{k}\Omega$).
- $\Delta Z = |Z_1 - Z_2|$ es el desbalance entre las impedancias de contacto piel-electrodo.

**Comportamiento:**
1. **Electrodo bien adherido:** La piel exfoliada y con gel conductor presenta $Z_1 \approx Z_2 \approx 5\text{--}10\,\text{k}\Omega$, con un desbalance residual de $\Delta Z \approx 1\,\text{k}\Omega$. Con $Z_i = 100\,\text{k}\Omega$, el cociente es $\frac{\Delta Z}{Z_i} = \frac{1\,\text{k}\Omega}{100\,\text{k}\Omega} = 0.01$ ($1\%$). Aunque se cuela algo de ruido, es residual.
2. **Electrodo con despegue parcial o tracción mecánica:** La impedancia del parche salta bruscamente a $50\text{--}100\,\text{k}\Omega$, provocando un desbalance $\Delta Z \approx 100\,\text{k}\Omega$. En ese instante:
   $$\frac{\Delta Z}{Z_i} \approx \frac{100\,\text{k}\Omega}{100\,\text{k}\Omega} = 1.0 \quad (100\%)$$
   Prácticamente la totalidad del voltio de ruido de línea presente en el paciente se transforma en señal diferencial, saturando la salida contra los rieles de $\pm 9\,\text{V}$.

### 1.4 Atenuación Inaceptable del Espectro Muscular
El rango bioeléctrico de interés para sEMG facial abarca de $20\,\text{Hz}$ a $500\,\text{Hz}$. Con el corte en $48.2\,\text{Hz}$, toda la banda lenta de potenciales de acción (de $20$ a $45\,\text{Hz}$) sufre una atenuación de entre $-6\,\text{dB}$ y $-12\,\text{dB}$, perdiendo más del $60\%$ de su potencia antes de ingresar al digitalizador.

---

## 2. Propuestas de Mejora

### 2.1 Propuesta 1: Modificación en una Etapa con Desacoplo de Alterna

Consiste en implementar el desacoplo de corriente alterna en el lazo de ganancia (Merletti \& Parker, Sección 5.3.2, Fig. 5.6b):

1. **Entradas Directas:** Puentear los dos capacitores cerámicos de $33\,\text{nF}$ en la entrada para que las bioseñales ingresen sin componentes reactivos desbalanceadores.
2. **Resistencias de Polarización de $10\,\text{M}\Omega$:** Reemplazar las resistencias de $100\,\text{k}\Omega$ a GND por valores de $10\,\text{M}\Omega$ (o mínimo $4.7\,\text{M}\Omega$). Esto reduce el cociente $\Delta Z / Z_{\text{in}}$ cien veces y suministra la ruta a masa obligatoria para las corrientes de polarización ($I_B \approx 2\,\text{nA}$).
3. **Desacoplo AC en Pines 1 y 8:**
   Intercalar en serie un capacitor bipolar de $C_G = 100\,\mu\text{F}$ con la resistencia de ganancia $R_G = 100\,\Omega$:
   - **En Corriente Continua ($f = 0\,\text{Hz}$):** $X_C \to \infty$, por lo que $Z_G \to \infty$. La ganancia estática resulta:
     $$G_{\text{DC}} = 1 + \frac{49.4\,\text{k}\Omega}{\infty} = 1$$
     El offset galvánico de la piel ($\pm 50\text{ a } \pm 100\,\text{mV}$) solo se multiplica por 1, evitando la saturación contra las baterías de $\pm 9\,\text{V}$.
   - **En Frecuencia de Corte ($f_c \approx 15.9\,\text{Hz}$):**
     $$f_c = \frac{1}{2\pi R_G C_G} = \frac{1}{2\pi \cdot 100\,\Omega \cdot 100\,\mu\text{F}} \approx 15.91\,\text{Hz}$$
   - **En Banda Muscular ($f \ge 20\,\text{Hz}$):** La reactancia capacitiva se vuelve despreciable ($31.8\,\Omega$ a $50\,\text{Hz}$, $15.9\,\Omega$ a $100\,\text{Hz}$, $3.2\,\Omega$ a $500\,\text{Hz}$), restableciendo la ganancia diferencial completa $G_{\text{AC}} \approx 495$.

---

### 2.2 Propuesta 2: Rediseño en Dos Etapas bajo Estándar SENIAM

Para la versión definitiva del hardware, la arquitectura óptima estandarizada por SENIAM distribuye la ganancia y el filtrado en dos etapas:

```
[Electrodos] ---> [Etapa 1: AD620] ---> [Filtro Inter-Etapa] ---> [Etapa 2: TL072] ---> [Salida DAQ]
                  (Entradas directas)   (Pasa-Altos Unipolar)     (Ganancia G2 = 48)
                  (G1 ≈ 9.8, 5.6k)      (fc ≈ 15.4 Hz, 470n/22k)  (Pasa-Bajos fc ≈ 498 Hz)
```

#### Bloque 1: Front-End AD620 con Baja Ganancia
- Entradas directas con resistencias de polarización de $10\,\text{M}\Omega$ a GND.
- Resistencia de ganancia $R_{G1} = 5.6\,\text{k}\Omega$:
  $$G_1 = 1 + \frac{49.4\,\text{k}\Omega}{5.6\,\text{k}\Omega} \approx 9.82$$
  Un offset DC severo de $100\,\text{mV}$ genera únicamente $0.98\,\text{V}$ a la salida, asegurando que el AD620 opere holgado sin saturar jamás los $\pm 9\,\text{V}$.

#### Bloque 2: Filtro Pasa-Altos Inter-Etapa
- Capacitor de acoplo $C_{\text{inter}} = 470\,\text{nF}$ en serie y resistencia $R_{\text{inter}} = 22\,\text{k}\Omega$ a masa:
  $$f_{c, \text{PA}} = \frac{1}{2\pi \cdot 22\,\text{k}\Omega \cdot 470\,\text{nF}} \approx 15.39\,\text{Hz}$$
- **Ventaja crítica:** Al estar referenciado a tierra (señal unipolar o *single-ended*), las tolerancias del capacitor no degradan el CMRR del sistema.

#### Bloque 3: Ganancia Secundaria y Pasa-Bajos Activo con TL072
- Amplificador no inversor con $R_1 = 1\,\text{k}\Omega$ y $R_f = 47\,\text{k}\Omega$:
  $$G_2 = 1 + \frac{R_f}{R_1} = 1 + \frac{47\,\text{k}\Omega}{1\,\text{k}\Omega} = 48$$
- Ganancia global del canal:
  $$G_{\text{tot}} = G_1 \cdot G_2 = 9.82 \cdot 48 \approx 471.4 \quad (\approx 495)$$
- Capacitor en paralelo con $R_f$ ($C_f = 6.8\,\text{nF}$) para filtro pasa-bajos activo anti-aliasing:
  $$f_{c, \text{PB}} = \frac{1}{2\pi R_f C_f} = \frac{1}{2\pi \cdot 47\,\text{k}\Omega \cdot 6.8\,\text{nF}} \approx 498.1\,\text{Hz}$$
  Atenúa armónicos de alta frecuencia e interferencias electromagnéticas por encima de $500\,\text{Hz}$ antes de la digitalización.

---

## 3. Cuadro Comparativo de Topologías

| Parámetro | 1. Original | 2. Modificado (1 Etapa) | 3. Recomendado (2 Etapas SENIAM) |
| :--- | :--- | :--- | :--- |
| **Topología** | In-Amp único ($G$ directa) | In-Amp con desacoplo AC en $R_G$ | In-Amp baja $G$ + Op-Amp activo |
| **Ganancia AC ($G_{\text{AC}}$)** | $\approx 495$ ($R_G = 100\,\Omega$) | $\approx 495$ ($R_G = 100\,\Omega$) | $\approx 471\text{--}495$ ($G_1 \approx 10$, $G_2 = 48$) |
| **Ganancia DC ($G_{\text{DC}}$)** | $495$ (Saturación contra rieles) | $1.0$ (Sin saturación) | $0$ (Bloqueado por filtro inter-etapa) |
| **Impedancia $Z_{\text{in}}$** | $\approx 100\,\text{k}\Omega$ | $\ge 10\,\text{M}\Omega$ | $\ge 10\,\text{M}\Omega$ a $1\,\text{G}\Omega$ |
| **Filtro Pasa-Altos** | $48.2\,\text{Hz}$ (Pérdida en sEMG) | $15.9\,\text{Hz}$ (Preserva banda útil) | $15.4\,\text{Hz}$ (Preserva banda útil) |
| **Filtro Pasa-Bajos** | Limitado por AD620 | Limitado por AD620 | $498\,\text{Hz}$ (Anti-aliasing activo) |
| **Sensibilidad al Despegue** | Crítica ($\Delta Z / Z_{\text{in}} \approx 1$) | Mínima ($\Delta Z / Z_{\text{in}} \le 0.01$) | Nula ($\Delta Z / Z_{\text{in}} \le 0.01$) |
| **Preservación de CMRR** | Destruida por caps. entrada | Máxima (entradas directas) | Máxima (entradas directas) |
| **Implementación** | PCB existente | Modificación inmediata en placa | Requiere fabricación de nuevo PCB |
| **Cumplimiento SENIAM** | No conforme | Aceptable para laboratorio | Cumplimiento total normativo |

---

## 4. Formulación Matemática Completa y Desglose de Parámetros

### 4.1 Ganancia Diferencial Estática del AD620
$$G = 1 + \frac{49.4\,\text{k}\Omega}{R_G}$$
- $G$: Ganancia diferencial de tensión en lazo cerrado (adimensional).
- $49.4\,\text{k}\Omega$: Resistencia interna equivalente de realimentación del AD620 ($24.7\,\text{k}\Omega + 24.7\,\text{k}\Omega$).
- $R_G$: Resistencia externa de ganancia entre pines 1 y 8 ($\Omega$).

### 4.2 Conversión de Modo Común a Diferencial por Desbalance de Merletti y Parker
$$\frac{V_{\text{out}}}{A_d} \approx V_{\text{cm}} \left( \frac{\Delta Z}{Z_i} + \frac{A_c}{A_d} \right)$$
- $V_{\text{out}}$: Tensión total presente en el pin 6 del amplificador ($\text{V}$).
- $A_d$: Ganancia diferencial del amplificador ($\text{V/V}$, adimensional).
- $A_c$: Ganancia en modo común del amplificador ($\text{V/V}$, adimensional).
- $A_c / A_d$: Factor de rechazo en modo común inverso ($\text{CMRR}^{-1}$).
- $V_{\text{cm}}$: Tensión en modo común acoplada al cuerpo por la red de $50\,\text{Hz}$ ($\text{V}$).
- $\Delta Z$: Desbalance entre las impedancias de contacto de los parches ($\Omega$).
- $Z_i$: Impedancia de entrada del amplificador ($\Omega$).

### 4.3 Frecuencia de Corte de Filtro Pasivo Pasa-Altos
$$f_c = \frac{1}{2\pi R C}$$
- $f_c$: Frecuencia de atenuación a $-3\,\text{dB}$ ($\text{Hz}$).
- $R$: Resistencia del filtro ($\Omega$).
- $C$: Capacidad del capacitor del filtro ($\text{F}$).

### 4.4 Impedancia Compleja en el Lazo de Ganancia
$$Z_G(f) = \sqrt{R_G^2 + X_C(f)^2} = \sqrt{R_G^2 + \left( \frac{1}{2\pi f C_G} \right)^2}$$
$$G(f) = 1 + \frac{49.4\,\text{k}\Omega}{Z_G(f)}$$
- $Z_G(f)$: Impedancia compleja neta entre los pines 1 y 8 ($\Omega$).
- $R_G$: Resistencia de ganancia en serie ($100\,\Omega$).
- $C_G$: Capacitor no polarizado de desacoplo ($100\,\mu\text{F}$).
- $f$: Frecuencia armónica de la componente bioeléctrica ($\text{Hz}$).
- $G(f)$: Ganancia efectiva dependiente de la frecuencia (adimensional).

### 4.5 Caída de Tensión por Corriente de Polarización
$$V_{\text{offset}, I_B} = I_B \cdot R_{\text{ext}}$$
- $V_{\text{offset}, I_B}$: Caída de continua en la entrada provocada por la corriente de polarización ($\text{V}$).
- $I_B$: Corriente de polarización del par de transistores de entrada ($I_B \approx 0.5\text{ a } 2.0\,\text{nA}$).
- $R_{\text{ext}}$: Resistencia de polarización conectada a tierra ($10\,\text{M}\Omega$).

### 4.6 Ganancia y Corte Anti-Aliasing en Etapa 2
$$G_{\text{tot}} = \left( 1 + \frac{49.4\,\text{k}\Omega}{R_{G1}} \right) \cdot \left( 1 + \frac{R_f}{R_1} \right)$$
$$f_{c, \text{PB}} = \frac{1}{2\pi R_f C_f}$$
- $G_{\text{tot}}$: Ganancia compuesta del acondicionamiento de señales (adimensional).
- $R_{G1}$: Resistencia de ganancia de la etapa 1 ($5.6\,\text{k}\Omega$).
- $R_1$: Resistencia de entrada inversora del TL072 a tierra ($1\,\text{k}\Omega$).
- $R_f$: Resistencia de realimentación del TL072 ($47\,\text{k}\Omega$).
- $C_f$: Capacitor de realimentación en paralelo ($6.8\,\text{nF}$).
- $f_{c, \text{PB}}$: Frecuencia de corte superior anti-aliasing ($\text{Hz}$).

### 4.7 Ecuación General de Transferencia de Tensión de Entrada a Salida y Desbalance Dinámico
La tensión total observable en la salida del amplificador ($V_{\text{out}}$) depende no solo del rechazo de modo común estático, sino de la transferencia analógica integral de cada rama de entrada considerando el acoplamiento bioeléctrico, el potencial galvánico y el divisor resistivo formado entre la impedancia de contacto piel-electrodo y la impedancia de entrada del instrumento.

Para cada electrodo de entrada ($j \in \{1, 2\}$), la tensión real presente en el nodo analógico del circuito integrado ($V_{\text{in}, j}$) resulta de la atenuación del divisor de tensión:
$$V_{\text{in}, j}(t) = \alpha_j(t) \cdot V_{\text{fuente}, j}(t) = \frac{Z_{i, j}}{Z_{e, j}(t) + Z_{i, j}} \cdot \left[ V_{\text{cm}}(t) \pm \frac{V_{\text{emg}}(t)}{2} + V_{\text{DC}, j}(t) \right]$$

donde:
- $V_{\text{in}, j}(t)$: Tensión instantánea que ingresa efectivamente a la entrada inversora o no inversora del chip ($\text{V}$).
- $\alpha_j(t)$: Factor de transferencia o atenuación del divisor de tensión de entrada para el canal $j$ (adimensional, $0 \le \alpha_j \le 1$).
- $Z_{e, j}(t)$: Impedancia de contacto de la interfaz piel-electrodo del canal $j$ ($\Omega$), dependiente de la adhesión mecánica, hidratación del gel y presión de contacto.
- $Z_{i, j}$: Impedancia de entrada a tierra de la rama correspondiente del amplificador ($\Omega$).
- $V_{\text{cm}}(t)$: Tensión parásita de modo común inducida en el cuerpo por el entorno electromagnético ($\text{V}$).
- $V_{\text{emg}}(t)$: Biopotencial de acción muscular diferencial genuino generado por la despolarización de las fibras musculares ($\text{V}$, típicamente $10\text{--}500\,\mu\text{V}$).
- $V_{\text{DC}, j}(t)$: Potencial galvánico continuo de media celda electroquímica en la interfaz metal-electrolito ($\text{V}$, típicamente $\pm 50\text{ a } \pm 300\,\text{mV}$).

La tensión total de salida del amplificador de instrumentación, considerando una ganancia diferencial $A_d$, una ganancia en modo común $A_c$ y una tensión de referencia $V_{\text{REF}}$, se modela formalmente como:
$$V_{\text{out}}(t) = A_d \left[ V_{\text{in}, 1}(t) - V_{\text{in}, 2}(t) \right] + A_c \left[ \frac{V_{\text{in}, 1}(t) + V_{\text{in}, 2}(t)}{2} \right] + V_{\text{REF}}$$

Sustituyendo la ecuación del divisor de entrada y expandiendo algebraicamente, la tensión de salida se desglosa en cuatro componentes físicos fundamentales:
$$V_{\text{out}}(t) \approx \underbrace{A_d \cdot \bar{\alpha}(t) \cdot V_{\text{emg}}(t)}_{\text{Señal sEMG amplificada}} + \underbrace{A_d \cdot \Delta\alpha(t) \cdot V_{\text{cm}}(t)}_{\text{Ruido de red por desbalance}} + \underbrace{A_d \cdot \left[ \alpha_1(t) V_{\text{DC}, 1} - \alpha_2(t) V_{\text{DC}, 2} \right]}_{\text{Salto y deriva de continua}} + \underbrace{A_c \bar{\alpha}(t) V_{\text{cm}}(t)}_{\text{Modo común residual}}$$

donde:
$$\bar{\alpha}(t) = \frac{\alpha_1(t) + \alpha_2(t)}{2} \approx \frac{Z_i}{\bar{Z}_e(t) + Z_i}$$
$$\Delta\alpha(t) = \alpha_1(t) - \alpha_2(t) = \frac{Z_i \cdot \left[ Z_{e, 2}(t) - Z_{e, 1}(t) \right]}{\left[ Z_{e, 1}(t) + Z_i \right] \cdot \left[ Z_{e, 2}(t) + Z_i \right]} \approx \frac{\Delta Z_e(t)}{Z_i} \quad (\text{para } Z_i \gg Z_e)$$

#### Mecanismo de Inestabilidad Dinámica por Despegue de Electrodos
Esta formulación matemática general revela las tres consecuencias simultáneas e inmediatas que se suscitan cuando un electrodo pierde contacto mecánico ($Z_{e, 1} \gg Z_{e, 2}$):
1. **Explosión del Ruido de Línea de 50 Hz:** El factor $\Delta\alpha(t)$ salta de $\approx 0.001$ a valores cercanos a $0.5\text{--}1.0$. La tensión de red ambiental $V_{\text{cm}}(t)$ se multiplica directamente por la ganancia diferencial completa $A_d \approx 495$, saturando instantáneamente los rieles de batería ($\pm 9\,\text{V}$).
2. **Pérdida de la Simetría y Distorsión de la Amplitud Muscular:** Al caer $\alpha_1(t)$ a valores inferiores a $0.5$, la señal muscular diferencial $V_{\text{emg}}(t)$ ya no se transfiere de manera balanceada. La contracción se atenúa artificialmente y se introduce una distorsión morfológica en la envolvente del pulso muscular.
3. **Salto Abrupto de Continua y Transitorio de Saturación:** La diferencia de potenciales galvánicos $(\alpha_1 V_{\text{DC}, 1} - \alpha_2 V_{\text{DC}, 2})$ experimenta un escalón instantáneo de tensión que se multiplica por $A_d$, desplazando violentamente la línea de base hacia los extremos del rango dinámico hasta que los lazos de acoplo logran reestabilizarse.

Al elevar la impedancia de entrada a $Z_i \ge 10\,\text{M}\Omega$ (o $> 100\,\text{M}\Omega$ como prescribe SENIAM), el denominador del factor $\Delta\alpha(t)$ se incrementa en varios órdenes de magnitud, garantizando que $\Delta\alpha(t) \approx 0$ e inmunizando al front-end frente a las variaciones dinámicas de la interfaz cutánea.

---

## 5. Análisis de Fenómenos No Lineales y Justificación de la Modificación Frente al Filtrado Digital

Una interrogante fundamental en el acondicionamiento de biopotenciales es si resulta estrictamente necesario modificar físicamente el circuito de la placa, dado que las interferencias de línea a $50\,\text{Hz}$ y sus armónicos ($100$, $150$, $250\,\text{Hz}$) pueden suprimirse mediante técnicas de procesamiento digital (como filtros Notch en cascada o canceladores adaptativos NLMS en cuadratura).

El análisis físico y circuital demuestra que el problema crítico ante el despegue de electrodos no radica en la componente periódica de $50\,\text{Hz}$, sino en fenómenos no lineales de banda ancha y distorsiones irreversibles que ningún algoritmo por software puede reconstruir una vez digitalizada la señal.

### 5.1 Salto Galvánico de Continua y Transitorio de Saturación
La interfaz metal-electrolito-piel constituye una celda electroquímica con un potencial continuo $V_{\text{DC}} \in [50, 300]\,\text{mV}$. Ante una gesticulación facial, vibración o despegue parcial del parche:
- La doble capa electroquímica se deforma bruscamente, modulando el potencial galvánico en el tiempo: $\Delta V_{\text{DC}}(t)$.
- Este transitorio no es una oscilación de $50\,\text{Hz}$, sino un escalón aperiódico de frecuencia ultralenta ($0.1\text{ a } 15\,\text{Hz}$).

En la topología actual sin desacoplo, el AD620 aplica su ganancia nominal directamente en corriente continua ($G_{\text{DC}} \approx 495$):
$$V_{\text{salida, DC}}(t) = 495 \cdot \Delta V_{\text{DC}}(t)$$
Un micro-despegue o tracción mecánica que altere el potencial en apenas $\Delta V_{\text{DC}} = 20\,\text{mV}$ intenta producir:
$$V_{\text{salida}} = 495 \times 0.02\,\text{V} = 9.9\,\text{V}$$
Al estar el sistema alimentado por baterías de $\pm 9\,\text{V}$, el amplificador colisiona inmediatamente contra el límite físico de saturación (*rail clipping*):
- La salida se congela en una meseta plana a $+8.5\,\text{V}$ o $-8.5\,\text{V}$.
- **Pérdida irrecuperable de información:** Durante el bloqueo por saturación, la derivada temporal se anula y la actividad muscular queda borrada. Ningún filtro digital (Notch, Butterworth o aprendizaje profundo) puede restituir una señal cuya amplitud fue recortada por saturación física del hardware.

Con la modificación propuesta (capacitor $C_G = 100\,\mu\text{F}$ en serie con $R_G$):
$$G_{\text{DC}} = 1 \implies V_{\text{salida, DC}} = 1 \times 20\,\text{mV} = 20\,\text{mV}$$
La línea base se desplaza únicamente $20\,\text{mV}$, evitando la saturación, manteniendo al amplificador en su zona lineal y preservando la señal muscular sin interrupciones.

### 5.2 Modulación Espuria de la Ganancia y Pérdida de Simetría Bioeléctrica
La tensión muscular genuina $V_{\text{emg}}$ transferida al circuito integrado depende del divisor resistivo formado por la impedancia de contacto del electrodo $Z_e$ y la impedancia de entrada $Z_i$:
$$V_{\text{in}} = V_{\text{emg}} \cdot \left( \frac{Z_i}{Z_e + Z_i} \right)$$

En la placa actual con $Z_i = 100\,\text{k}\Omega$:
- Con electrodo bien adherido ($Z_e \approx 5\,\text{k}\Omega$):
  $$\frac{100\,\text{k}\Omega}{5\,\text{k}\Omega + 100\,\text{k}\Omega} \approx 0.95 \quad (\text{se transfiere el } 95\% \text{ del potencial})$$
- Ante un despegue parcial o secado del gel ($Z_e \approx 100\,\text{k}\Omega$):
  $$\frac{100\,\text{k}\Omega}{100\,\text{k}\Omega + 100\,\text{k}\Omega} = 0.50 \quad (\text{se transfiere únicamente el } 50\% \text{ del potencial})$$

Esta caída del $50\%$ en la amplitud se origina por un fenómeno mecánico y no por relajación fisiológica. En esquemas de procesamiento multicanal basados en el balance de activación entre músculos (como los ratios entre milohioideo, digástrico y orbicular, o la normalización por el Supremo Tricanal $M_{\text{supremo}}$), este falseamiento destruye la coherencia de las envolventes. El software no puede distinguir si la amplitud decreció por fonación submáximal o por despegue del sensor.

Con la elevación de la impedancia a $Z_i \ge 10\,\text{M}\Omega$:
$$\frac{10\,\text{M}\Omega}{100\,\text{k}\Omega + 10\,\text{M}\Omega} \approx 0.99 \quad (99\%)$$
La amplitud bioeléctrica permanece invariable con un error inferior al $1\%$, independientemente de las fluctuaciones de contacto.

### 5.3 Ruido Térmico de Johnson y Ruido de Corriente de Banda Ancha
El despegue no solo introduce armónicos de red, sino ruido distribuido en todo el espectro sEMG ($20\text{ a }500\,\text{Hz}$):
1. **Ruido Térmico en la Piel:**  
    $$v_n = \sqrt{4 k_B T \cdot \text{Re}(Z_e) \cdot \Delta f}$$
    donde $k_B$ es la constante de Boltzmann ($1.38 \times 10^{-23}\,\text{J/K}$), $T$ es la temperatura absoluta en Kelvin ($310\,\text{K}$) y $\Delta f$ es el ancho de banda ($500\,\text{Hz}$). Al despegarse el electrodo, el área efectiva disminuye y $Z_e$ salta de $5\,\text{k}\Omega$ a $100\,\text{k}\Omega$, multiplicando por casi cinco veces el piso de ruido blanco térmico en toda la banda pasante.
2. **Ruido de Corriente del Amplificador ($i_n$):**  
    El AD620 presenta una densidad de corriente de ruido de $i_n \approx 100\,\text{fA}/\sqrt{\text{Hz}}$. Al circular por una asimetría de contacto $\Delta Z_e$, genera una tensión de ruido diferencial:
    $$v_{\text{ruido}} = i_n \cdot \Delta Z_e$$
    Este ruido es de naturaleza estocástica continua en el rango de $20\,\text{Hz}$ a $1\,\text{kHz}$, por lo que los filtros Notch centrados en $50\,\text{Hz}$ resultan totalmente ineficaces para eliminarlo.

### 5.4 Impacto del Filtro de Entrada en Cincuenta Hertz sobre la Bioseñal Facial
Mantener el filtro pasa-altos $RC$ de entrada ($C = 33\,\text{nF}$, $R = 100\,\text{k}\Omega$) con frecuencia de corte en $f_c \approx 48.2\,\text{Hz}$ introduce fallas intrínsecas graves:
1. **Atenuación Severa del Espectro sEMG Lento:**  
    La densidad espectral de potencia del electromiograma facial presenta componentes cardinales de reclutamiento motor entre $20\,\text{Hz}$ y $45\,\text{Hz}$. La curva de transferencia $|H(f)| = \frac{f/f_c}{\sqrt{1 + (f/f_c)^2}}$ arroja:
    - A $20\,\text{Hz}$: $|H(20)| \approx 0.38$ (pérdida del $62\%$ de la amplitud).
    - A $30\,\text{Hz}$: $|H(30)| \approx 0.53$ (pérdida del $47\%$ de la amplitud).
    - A $40\,\text{Hz}$: $|H(40)| \approx 0.64$ (pérdida del $36\%$ de la amplitud).  
    El filtro analógico previo suprime más de la mitad de la información biomecánica antes de la digitalización.
2. **Desfase Asimétrico y Generación de Tensión Diferencial:**  
    A $f_c$, la rotación de fase es de $+45^\circ$. Una disparidad de solo $\pm 10\%$ en la tolerancia entre capacitores desfasa una entrada respecto de la otra en $\Delta\phi \approx 8^\circ$. La diferencia vectorial convierte la tensión de modo común ambiental $V_{\text{cm}}$ en tensión diferencial:
    $$V_{\text{diff}}(t) \approx 2 V_0 \sin\left(\frac{\Delta\phi}{2}\right) \cos(\omega t) \approx 0.14 \cdot V_0$$
    Para $V_0 = 1\,\text{V}$, se generan $140\,\text{mV}$ diferenciales que, multiplicados por $G=495$, dan $69.3\,\text{V}$, saturando los rieles.
3. **Picos de Corriente Transitorios ($i = C \frac{dV}{dt}$):**  
    Ante un tirón mecánico del cable, la variación abrupta del potencial de contacto inyecta pulsos transitorios de corriente que despolarizan internamente el AD620 durante cientos de milisegundos.
4. **Impedancia de Entrada Reactiva:**  
    A $50\,\text{Hz}$, la reactancia capacitiva es $X_C \approx 96.5\,\text{k}\Omega$. El módulo neto $|Z_{\text{in}}| = \sqrt{R^2 + X_C^2} \approx 139\,\text{k}\Omega$ introduce desfases variables dependientes de la impedancia cutánea, deformando la morfología del pulso.

### 5.5 Cuadro Comparativo entre Filtrado Digital y Modificación de Hardware

| Fenómeno ante Despegue / Movimiento | Eficacia del Filtrado Digital (Notch / ANC) | Eficacia de la Modificación en Hardware ($R_G+C_G$, $10\,\text{M}\Omega$) |
| :--- | :--- | :--- |
| **Ruido de línea a 50 Hz** | **Alta** (elimina componentes senoidales) | **Alta** (elimina la conversión modo común a diferencial) |
| **Saturación contra rieles ($\pm 9\,\text{V}$)** | **Nula** (información destruida por clipping) | **Total** (fija $G_{\text{DC}}=1$, evitando saturación) |
| **Caída de amplitud del 50\% en contracción** | **Nula** (el software no distingue despegue de relajación) | **Total** (mantiene $Z_{\text{in}} \ge 10\,\text{M}\Omega$, transferencia al 99\%) |
| **Ruido térmico blanco (20–500 Hz)** | **Nula** (el Notch solo afecta 50 Hz) | **Alta** (reduce la inyección de corriente de ruido) |
| **Atenuación muscular lenta (20–45 Hz)** | **Nula** (energía ya eliminada por el filtro analógico) | **Total** (desplaza el corte a 15.9 Hz, recuperando la banda) |

---

## 6. Estrategia Experimental: Modificación en Placa sin Desoldar Componentes

Para validar de inmediato la corrección de 1 etapa de forma 100% reversible y sin dar vuelta la placa ni desoldar pistas:

### Paso 1: Puentear los Capacitores de Entrada desde la Cara Superior
- Tomar dos trozos cortos de alambre fino pelado de cobre.
- Enrollar firmemente un alambre uniendo los dos terminales expuestos de cada capacitor cerámico de $33\,\text{nF}$ (las dos lentejas naranjas sobre el AD620).
- **Verificación:** Con el multímetro en modo continuidad, medir entre el pad del conector de entrada y los pines 2 y 3 del chip; debe indicar $0\,\Omega$.

### Paso 2: Intercalar el Capacitor $C_G$ Cortando una Pata de $R_G$
En el montaje físico, la resistencia de ganancia ($R_G = 100\,\Omega$, azul) se encuentra montada con terminales largos por encima del zócalo:
- Con un alicate de corte al ras fino, cortar por la mitad una sola de las patas largas de $R_G$.
- Quedarán dos terminales rígidos expuestos mirando hacia arriba.
- Soldar o enroscar firmemente un capacitor no polarizado de $100\,\mu\text{F}$ (o dos electrolíticos de $220\,\mu\text{F}$ en antiserie) entre los dos extremos cortados.
- **Resultado:** El lazo queda conectado en serie ($R_G + C_G$) sin tocar el chip AD620 ni alterar el circuito impreso. Si se desea revertir, solo se retira el capacitor y se unen los dos alambres con una gota de estaño.

### Paso 3: Validación Inicial con las Resistencias Existentes de $100\,\text{k}\Omega$
**No es necesario desoldar las resistencias de $100\,\text{k}\Omega$ para la primera prueba de funcionamiento:**
- Las resistencias de $100\,\text{k}\Omega$ ya están conectadas a masa y cumplen la función de drenar $I_B$.
- Como el capacitor $C_G$ fija $G_{\text{DC}} = 1$, el offset galvánico de la piel ya no satura los rieles de $\pm 9\,\text{V}$.
- Encender la fuente y comprobar en el software que la línea base oscile libremente en reposo cerca de $0\,\text{V}$.

### Paso 4: Reemplazo por Resistencias de $10\,\text{M}\Omega$ para Blindaje Antidespegue
Una vez confirmado que la señal bioeléctrica entra viva y sin saturación:
- Cortar una sola pata de cada una de las resistencias de $100\,\text{k}\Omega$ a GND.
- Empalmar en su lugar resistencias axiales de $10\,\text{M}\Omega$ (o mínimo $4.7\,\text{M}\Omega$).
- Esto eleva $Z_{\text{in}}$ a más de $10\,\text{M}\Omega$, reduciendo el factor $\Delta Z / Z_{\text{in}}$ ante despegues en un $99\%$.

### Paso 5: Fijación Mecánica y Preparación de la Referencia
- **Alivio de Tensión Mecánica:** El despegue suele ser inducido por la palanca que ejerce el cable sobre el parche muscular. Fijar el cable trenzado con cinta médica adhesiva a la mandíbula o cuello, dejando los últimos $5\text{ a }8\,\text{cm}$ dóciles y sin tensión.
- **Preparación de la Referencia (Mastoides):** Limpiar rigurosamente con alcohol la zona ósea de la apófisis mastoides para asegurar una vía de drenaje de baja impedancia para el modo común corporal.

---

## 7. Evidencia Bibliográfica del Tratado de Merletti y Parker

A continuación se adjuntan las páginas pertinentes del libro oficial *Electromyography: Physiology, Engineering, and Noninvasive Applications* (Roberto Merletti y Philip A. Parker, IEEE Press / John Wiley \& Sons, 2004), extraídas directamente del tratado de referencia.

### 7.1 Requisitos de Entrada y Rechazo de Modo Común
La página 115 formaliza los requerimientos de la instrumentación de entrada: ultra-alta impedancia para evitar carga galvánica, bajo ruido referido a la entrada y alto CMRR para rechazar la interferencia ambiental de red.

![Página 115](imagenes/captura_merletti_sec5_5-136.png)

### 7.2 Justificación de la Impedancia Superior a Cien Megaohmios
La página 116 establece que la impedancia de entrada debe superar en al menos dos órdenes de magnitud la impedancia de la piel con gel ($> 100\,\text{M}\Omega$) y superar $1000\,\text{M}\Omega$ ($1\,\text{G}\Omega$) para electrodos secos, explicando el efecto del divisor capacitivo de entrada.

![Página 116](imagenes/captura_merletti_sec5_5-137.png)

### 7.3 Modelado de Acoplamiento y Redes de Entrada
La página 117 detalla la circuitería equivalente de acoplo de entrada, las capacidades parásitas de los cables y la necesidad de drenar las corrientes de polarización hacia masa sin degradar la impedancia vista por el electrodo.

![Página 117](imagenes/captura_merletti_sec5_5-138.png)

### 7.4 Desacoplo en el Lazo de Ganancia y Desbalance de Electrodos
La página 118 incluye la fundamental **Figura 5.6(b)**, donde se ilustra el desacoplo de alterna mediante capacitor serie en la rama de ganancia ($R_G + C_G$), así como la demostración analítica de la ecuación de interferencia de modo común por desbalance $\Delta Z / Z_i$.

![Página 118](imagenes/captura_merletti_sec5_5-139.png)

### 7.5 Cuadro de Recomendaciones Oficiales SENIAM
La página 127 presenta la **Tabla 5.4** con las recomendaciones oficiales consensuadas por el consorcio europeo SENIAM para sensores, amplificadores de instrumentación y convertidores A/D en electromiografía de superficie.

![Página 127](imagenes/captura_merletti_tabla_seniam-148.png)

---

## 8. Caracterización Experimental y Circuito Físico Completo

En esta sección se documenta el hardware completo desarrollado por Santiago Prado Rodríguez y Lucas Gastón Braunstein para el Laboratorio de Sistemas Dinámicos, integrando la etapa activa de protección de alimentación, el diseño del PCB, la nómina de componentes y la caracterización experimental de ganancia en función de la frecuencia.

### 8.1 Esquema Electrónico Integral y Etapa de Alimentación
![Esquema Electrónico Vectorial Completo: 3 Canales y Protección](imagenes/esquema_completo_circuitikz.png)

El sistema incorpora una etapa de protección activa contra inversión de polaridad que supera las limitaciones de los diodos rectificadores comunes:
- **Riel Negativo ($-9\,\text{V}$):** Utiliza un MOSFET de canal N (**IRFZ44N**) en el retorno. Con polaridad correcta, la compuerta se referencia a masa mediante $R_1 = 10\,\text{k}\Omega$, fijando $V_{GS} \approx +9\,\text{V}$ y saturando el canal con $R_{DS(\text{on})} \approx 17.5\,\text{m}\Omega$. La caída de tensión es de apenas $1.4\,\text{mV}$ a $50\,\text{mA}$, sin pérdidas térmicas.
- **Riel Positivo ($+9\,\text{V}$):** Utiliza un MOSFET de canal P (**F9540N**) en serie. Su compuerta referenciada a masa mediante $R_3 = 10\,\text{k}\Omega$ establece $V_{GS} \approx -9\,\text{V}$, conduciendo con $R_{DS(\text{on})} \approx 0.11\,\Omega$ y una caída de tensión de solo $5.5\,\text{mV}$.
- **Protección por Diodos Zener:** Los diodos $D_1$ y $D_2$ (1N4741A, $V_Z = 11\,\text{V}$) fijan la tensión máxima compuerta-fuente, protegiendo el dieléctrico de los transistores ante transitorios de encendido.
- **Fusibles de Acción Rápida:** Las líneas protegidas se conectan a fusibles de $0.25\,\text{A}$ ($FUSE_1$ y $FUSE_2$), previniendo fallos destructivos ante cortocircuitos accidentales.

### 8.2 Curvas de Amplificación en Función de la Resistencia de Ganancia (Figura 18)
![Curvas Experimentales de Ganancia vs Frecuencia](imagenes/curvas_ganancia_experimental_fig18.png)
*Figura 18: Curvas de amplificación en función de la resistencia $R_G$. En línea punteada el valor teórico según el fabricante ($G = 1 + 49.4\,\text{k}\Omega / R_G$).*

Para contrastar el comportamiento medido frente al modelo físico ideal, a continuación se presenta la respuesta en frecuencia teórica calculada con la formulación analítica completa del sistema:
$$|H(f)| = \frac{\frac{f}{f_L}}{\sqrt{1 + \left(\frac{f}{f_L}\right)^2}} \cdot \frac{1 + \frac{49.4\,\text{k}\Omega}{R_G}}{\sqrt{1 + \left(\frac{f}{f_H}\right)^2}}$$
con $f_L \approx 48.23\,\text{Hz}$ y $f_H \approx \frac{1.2 \times 10^7}{G}\,\text{Hz}$.

![Curvas Teóricas de Respuesta en Frecuencia](imagenes/curva_respuesta_frecuencia_teorica.png)

### 8.3 Comparación entre el Circuito Actual y la Modificación de Impedancia
Para subsanar el aplastamiento de impedancia ($Z_{\text{in}} \approx 100\,\text{k}\Omega$) y la pérdida de la banda sEMG orofacial por debajo de $48\,\text{Hz}$, la siguiente figura compara la respuesta del hardware actual frente a la modificación propuesta (puenteo de capacitores de entrada, elevación de resistencias a $10\,\text{M}\Omega$ y capacitor $C_G = 100\,\mu\text{F}$ en serie con $R_G$).

![Esquema Vectorial del Canal Modificado (Rg en serie con Cg)](imagenes/esquema_mejora_circuitikz.png)

![Comparación de Frecuencia e Impedancia](imagenes/comparacion_respuesta_frecuencia_impedancia.png)

### 8.4 Lista Oficial de Materiales (BOM) y Diseño del PCB

| Cant. | Componente | Descripción / Valor | Designador (Label) |
| :---: | :--- | :--- | :--- |
| 2 | Resistencia $10\,\text{k}\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ | $R_{1,3}$ (Alimentación) |
| 8 | Resistencia $100\,\text{k}\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ | $R_{1,2,5,4,7,8}$ (Canales) y $R_{2,4}$ (Alim.) |
| 3 | Resistencia $100\,\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ ($R_G$) | $R_{3,6,9}$ |
| 6 | Capacitor $33\,\text{nF}$ | Cerámico multicapa $50\,\text{V}$ | $C_{1,2,3,4,5,6}$ |
| 2 | Diodo Zener 1N4741A | Zener $11.0\,\text{V}$, $1.0\,\text{W}$ | $D_{1,2}$ |
| 1 | MOSFET IRFZ44N | Canal N, $55\,\text{V}$, $49\,\text{A}$, TO-220 | $\text{MOS N}$ |
| 1 | MOSFET F9540N | Canal P, $-100\,\text{V}$, $-19\,\text{A}$, TO-220 | $\text{MOS P}$ |
| 3 | Amplificador AD620 | Amplificador de instrumentación, DIP-8 | $U_{1,2,3}$ |
| 3 | Zócalo de 8 pines | Zócalo DIP-8 para circuito integrado | $U_{1,2,3}$ |
| 2 | Portafusible | Portafusible para PCB $5 \times 20\,\text{mm}$ | $\text{FUSN/P}_{1,2}$ |
| 2 | Fusible $0.25\,\text{A}$ | Fusible de vidrio de acción rápida $250\,\text{mA}$ | $\text{FUSE}_{1,2}$ |
| 1 | Tira pines macho | Tira de pines rectos paso $2.54\,\text{mm}$ | -- |
| 2 | Mini jumper | Jumper paso $2.54\,\text{mm}$ | -- |
| 1 | Placa virgen epoxi | Placa de cobre virgen $15 \times 15\,\text{cm}$ | Sustrato PCB |

![Layout del PCB Físico](imagenes/layout_pcb_completo.png)
*Figura: Trazado del circuito impreso (PCB Layout) con serigrafía del Laboratorio de Sistemas Dinámicos.*

---

## 9. Conclusiones y Próximos Pasos

1. La presencia de filtros pasa-altos $RC$ a la entrada de un amplificador de instrumentación es conceptualmente errónea en electromiografía: destruye el CMRR y reduce la impedancia del canal.
2. La modificación en 1 etapa mediante desacoplo AC en $R_G$ ($100\,\Omega + 100\,\mu\text{F}$) junto con resistencias de $10\,\text{M}\Omega$ a tierra soluciona de forma inmediata el desbalance de $50\,\text{Hz}$ y el riesgo de saturación sobre la placa ya armada.
3. Para la siguiente revisión de PCB, la arquitectura en dos etapas recomendada por SENIAM (AD620 a baja ganancia con filtro inter-etapa y segunda etapa activa TL072 con corte a $500\,\text{Hz}$) garantiza el cumplimiento normativo riguroso y la máxima robustez en mediciones sEMG submáximales.
