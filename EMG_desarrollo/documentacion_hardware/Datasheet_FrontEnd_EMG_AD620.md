<p align="center">
  <img src="imagenes/logo_nandu_lsd.png" alt="Logo Ñandú LSD" width="140">
</p>

# Hoja de Datos Técnicos (Datasheet)
## Amplificador Diferencial de 3 Canales
### Basado en Amplificador de Instrumentación AD620 y Protección Activa por MOSFETs

**Documento Oficial de Hardware — Revisión 1.0 (Septiembre 2026)**  
**Autores:** Santiago Prado Rodríguez — Lucas Gastón Braunstein  
**Filiación:** Laboratorio de Sistemas Dinámicos — Universidad de Buenos Aires  

---

## 1. Características Principales

### Amplificación Diferencial
- **3 canales analógicos diferenciales independientes** diseñados específicamente para electromiografía de superficie (sEMG) orofacial y biopotenciales de baja amplitud.
- **Amplificadores de instrumentación monolíticos AD620** montados en zócalos DIP-8 para reemplazo rápido y ensayo experimental.
- **Ganancia fija por resistencia soldada en placa:** $R_G = 100\,\Omega$ ($G \approx 495\,\text{V/V}$; ensayada experimentalmente entre $33\,\Omega$ y $220\,\Omega$).
- **Elevado rechazo de modo común:** $\text{CMRR} > 100\,\text{dB}$ a $50\,\text{Hz}$ para ganancias $G \ge 100$.
- **Filtro pasa-altos pasivo de entrada:** Celdas $RC$ unipolares ($C = 33\,\text{nF}$, $R = 100\,\text{k}\Omega$) con corte en $f_c \approx 48.23\,\text{Hz}$ y camino de retorno a masa para corrientes de polarización de entrada ($I_B$).
- **Conectores de entrada a tornillo (TBLOCK-M3):** Borneras triples por canal con terminal dedicado a referencia corporal en el canal 3 (`IN3GND`).

### Alimentación y Protección Integral
- **Alimentación simétrica bipolar por dos baterías externas de 9V:** $\pm 9\,\text{V}$ para garantizar aislamiento galvánico de red y bioseguridad absoluta.
- **Protección activa contra inversión de polaridad por MOSFETs:**
  - Riel positivo ($+9\,\text{V}$): MOSFET canal P modelo **F9540N** en serie ($R_{DS(\text{on})} \approx 0.11\,\Omega$).
  - Riel negativo ($-9\,\text{V}$): MOSFET canal N modelo **IRFZ44N** en retorno ($R_{DS(\text{on})} \approx 0.017\,\Omega$).
  - Caída de tensión despreciable ($V_{DS} \approx 5\,\text{mV}$ a $50\,\text{mA}$) que maximiza la vida útil de las baterías frente a la pérdida fija de $0.7\,\text{V}$ de diodos comunes.
- **Fijación de compuerta ($V_{GS}$):** Diodos Zener 1N4741A ($V_Z = 11.0\,\text{V}$) y divisores $10\,\text{k}\Omega / 100\,\text{k}\Omega$ protegen el óxido de compuerta contra sobretensiones y transitorios de conexión.
- **Fusibles de acción rápida:** Calibre de $0.25\,\text{A}$ ($250\,\text{mA}$) montados en portafusibles independientes por rama (`FUSP1/2` y `FUSN1/2`).
- **Interruptor general bipolar:** Bornera cuádruple (`SWITCH`) para corte simultáneo de ambos polos de alimentación.



## 2. Esquema del Circuito y Diseño de PCB

![Esquema Electrónico Vectorial Completo: 3 Canales y Protección](imagenes/esquema_completo_circuitikz.png)
*Figura: Esquema electrónico integral del front-end sEMG tricanal con etapa de alimentación y protección activa.*

![Layout del PCB Físico](imagenes/layout_pcb_completo.png)
*Figura: Trazado y ruteo físico del circuito impreso (PCB Layout) correspondiente al front-end sEMG tricanal.*

### Lista Completa de Materiales

| Cant. | Componente | Descripción / Valor | Designador (Label) |
| :---: | :--- | :--- | :--- |
| 2 | Resistencia $10\,\text{k}\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ | $R_{1,3}$ (Alimentación) |
| 8 | Resistencia $100\,\text{k}\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ | $R_{1,2,5,4,7,8}$ (Canales) y $R_{2,4}$ (Alim.) |
| 3 | Resistencia $100\,\Omega$ | Película metálica $1/4\,\text{W}$, $\pm 1\%$ ($R_G$) | $R_{3,6,9}$ |
| 6 | Capacitor $33\,\text{nF}$ | Cerámico multicapa $50\,\text{V}$ | $C_{1,2,3,4,5,6}$ |
| 2 | Diodo Zener 1N4741A | Zener $11.0\,\text{V}$, $1.0\,\text{W}$ | $D_{1,2}$ |
| 1 | MOSFET IRFZ44N | Canal N, $55\,\text{V}$, $49\,\text{A}$, $R_{DS}=17.5\,\text{m}\Omega$, TO-220 | $\text{MOS N}$ |
| 1 | MOSFET F9540N | Canal P, $-100\,\text{V}$, $-19\,\text{A}$, $R_{DS}=0.11\,\Omega$, TO-220 | $\text{MOS P}$ |
| 3 | Amplificador AD620 | Amplificador de instrumentación de precisión, DIP-8 | $U_{1,2,3}$ |
| 3 | Zócalo de 8 pines | Zócalo DIP-8 para circuito integrado | $U_{1,2,3}$ |
| 2 | Portafusible | Portafusible para circuito impreso 5x20 mm | $\text{FUSN/P}_{1,2}$ |
| 2 | Fusible $0.25\,\text{A}$ | Fusible de vidrio de acción rápida $250\,\text{mA}$ | $\text{FUSE}_{1,2}$ |
| 1 | Tira pines macho | Tira de pines rectos paso $2.54\,\text{mm}$ | -- |
| 2 | Mini jumper | Jumper de corto circuito paso $2.54\,\text{mm}$ | -- |
| 1 | Placa de cobre virgen | Placa de pertinax o epoxi $15 \times 15\,\text{cm}$ | Sustrato PCB |

### Análisis de los Bloques Circuitales
1. **Bornera de Alimentación (`BAT`):** Conector de tres contactos (Pin 1: $+9\,\text{V}$, Pin 2: $\text{GND}$, Pin 3: $-9\,\text{V}$).
2. **Bornera de Interruptor (`SWITCH`):** Conector cuádruple que interrumpe de forma simultánea los dos rieles de alimentación previo al ingreso a la etapa de protección.
3. **Protección contra Inversión de Polaridad por MOSFETs:**
   - **Riel Negativo ($-9\,\text{V}$):** Se intercala un MOSFET de canal N **IRFZ44N**. Su compuerta está referenciada a masa mediante $R_1 = 10\,\text{k}\Omega$. En polaridad correcta, la compuerta se encuentra $9\,\text{V}$ por encima de la fuente ($V_{GS} \approx +9\,\text{V}$), saturando el canal con $R_{DS(\text{on})} \approx 17.5\,\text{m}\Omega$. Si se invierten los cables de la batería, $V_{GS} \le 0\,\text{V}$ y el transistor se bloquea por completo. El diodo Zener $D_1$ (1N4741A, $11\,\text{V}$) fija el límite superior de $V_{GS}$ y $R_2 = 100\,\text{k}\Omega$ purga cargas residuales.
   - **Riel Positivo ($+9\,\text{V}$):** Se intercala un MOSFET de canal P **F9540N**. Su compuerta se referencia a masa a través de $R_3 = 10\,\text{k}\Omega$, logrando $V_{GS} \approx -9\,\text{V}$ en polaridad directa ($R_{DS(\text{on})} \approx 0.11\,\Omega$). El diodo Zener $D_2$ (1N4741A) actúa como abrazadera de protección.
   - **Fusibles FUSE1 y FUSE2:** Ubicados en serie tras los transistores, limitan la corriente a $250\,\text{mA}$.
4. **Canales de Instrumentación 1, 2 y 3:**
   - Cada canal dispone de celdas pasa-altos pasivas formadas por capacitores $C_1\text{--}C_6 = 33\,\text{nF}$ y resistencias $R_1, R_2, R_4, R_5, R_7, R_8 = 100\,\text{k}\Omega$.
   - Los amplificadores de instrumentación $U_1, U_2, U_3$ (AD620) toman la señal diferencial en sus pines 2 y 3.
   - La ganancia se fija por canal mediante $R_3, R_6, R_9$ ($R_G$).
   - El pin 5 de referencia (`REF`) está conectado sólidamente al plano de masa común ($0\,\text{V}$).
   - Las salidas analógicas se entregan en borneras a tornillo `OUT1`, `OUT2`, `OUT3`.

---

## 3. Asignación de Pines de Borneras

A continuación se detalla la correspondencia eléctrica de cada uno de los terminales a tornillo de la placa física mostrada en el diseño de circuito impreso (Figura 2):

| Bornera | Pin | Función | Descripción Técnica |
| :--- | :---: | :---: | :--- |
| `BAT` | Pin 1 | $+V_{\text{BAT}}$ | Entrada positiva de batería ($+9\,\text{V}$) |
| | Pin 2 | $\text{GND}$ | Punto medio de masa analógica común ($0\,\text{V}$) |
| | Pin 3 | $-V_{\text{BAT}}$ | Entrada negativa de batería ($-9\,\text{V}$) |
| `SWITCH` | Pin 1-2 | $\text{SW}_+$ | Contacto seco para corte del riel positivo |
| | Pin 3-4 | $\text{SW}_-$ | Contacto seco para corte del riel negativo |
| `IN1` | Pin 1 | $\text{GND}$ | Referencia de blindaje / plano de masa |
| | Pin 2 | $\text{IN1}_-$ | Entrada inversora del Canal 1 (Electrodo A) |
| | Pin 3 | $\text{IN1}_+$ | Entrada no inversora del Canal 1 (Electrodo B) |
| `IN2` | Pin 1 | $\text{GND}$ | Referencia de blindaje / plano de masa |
| | Pin 2 | $\text{IN2}_-$ | Entrada inversora del Canal 2 (Electrodo A) |
| | Pin 3 | $\text{IN2}_+$ | Entrada no inversora del Canal 2 (Electrodo B) |
| `IN3GND` | Pin 1 | $\text{REF}_{\text{body}}$ | Conexión al electrodo de referencia corporal (lóbulo / muñeca) |
| | Pin 2 | $\text{IN3}_-$ | Entrada inversora del Canal 3 |
| | Pin 3 | $\text{IN3}_+$ | Entrada no inversora del Canal 3 |
| `OUT1` | Pin 1 | $\text{GND}$ | Masa de referencia analógica hacia digitalizador |
| | Pin 2 | $\text{OUT1}$ | Salida sEMG amplificada del Canal 1 ($V_{\text{out1}}$) |
| `OUT2` | Pin 1 | $\text{GND}$ | Masa de referencia analógica hacia digitalizador |
| | Pin 2 | $\text{OUT2}$ | Salida sEMG amplificada del Canal 2 ($V_{\text{out2}}$) |
| `OUT3` | Pin 1 | $\text{GND}$ | Masa de referencia analógica hacia digitalizador |
| | Pin 2 | $\text{OUT3}$ | Salida sEMG amplificada del Canal 3 ($V_{\text{out3}}$) |

---

## 4. Límites Máximos Absolutos

| Parámetro | Símbolo | Valor Límite | Unidad |
| :--- | :---: | :---: | :---: |
| Tensión de alimentación positiva | $+V_S$ | $+18.0$ | $\text{V}$ |
| Tensión de alimentación negativa | $-V_S$ | $-18.0$ | $\text{V}$ |
| Tensión diferencial máxima de entrada | $V_{\text{in, dif}}$ | $\pm V_S$ | $\text{V}$ |
| Tensión de entrada en modo común | $V_{\text{cm, max}}$ | $\pm (V_S - 1.2)$ | $\text{V}$ |
| Corriente máxima admisible por riel | $I_{\text{max, riel}}$ | $0.25$ | $\text{A}$ |
| Duración admisible de cortocircuito en salida | $t_{\text{sc}}$ | Indefinida | -- |
| Temperatura de operación en laboratorio | $T_{\text{op}}$ | $0\text{ a }70$ | $^\circ\text{C}$ |
| Temperatura de almacenamiento | $T_{\text{stg}}$ | $-40\text{ a }+85$ | $^\circ\text{C}$ |

---

## 5. Condiciones de Operación Recomendadas

| Condición de Operación | Mínimo | Nominal | Máximo | Unidad |
| :--- | :---: | :---: | :---: | :---: |
| Tensión de alimentación simétrica ($\pm V_S$) | $\pm 6.0$ | $\pm 9.0\text{ ó }\pm 12.0$ | $\pm 15.0$ | $\text{V}$ |
| Amplitud esperada de la señal sEMG ($V_{\text{emg}}$) | $0.05$ | $0.2\text{ a }2.0$ | $5.0$ | $\text{mV}$ |
| Rango de resistencia de ganancia ($R_G$) | $33$ | $100$ | $220$ | $\Omega$ |
| Tensión de referencia de salida ($V_{\text{REF}}$) | $0.0$ | $0.0$ | $0.0$ | $\text{V}$ |
| Corriente consumida en reposo (por canal AD620) | $0.9$ | $1.3$ | $1.6$ | $\text{mA}$ |

---

## 6. Especificaciones Eléctricas Consolidadas

$V_S = \pm 12.0\,\text{V}$, $T_A = 25\,^\circ\text{C}$, $V_{\text{REF}} = 0\,\text{V}$, carga de $10\,\text{k}\Omega$, salvo indicación contraria.

| Parámetro | Condición de Ensayo | Mín. | Típ. | Máx. | Unidad |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Etapa de Ganancia y Amplificación (AD620)** | | | | | |
| Ganancia de tensión fijada en hardware ($G$) | $R_G = 100\,\Omega$ (soldada) | $480$ | $495$ | $510$ | $\text{V/V}$ |
| Error de ganancia respecto a fórmula | $R_G = 100\,\Omega$ | -- | $\pm 1.5$ | $\pm 3.0$ | $\%$ |
| Rango de excursión de salida ($V_{\text{out}}$) | $R_L \ge 10\,\text{k}\Omega$ | $\pm 10.5$ | $\pm 10.9$ | -- | $\text{V}$ |
| Rechazo de modo común ($\text{CMRR}$) | $f = 50\,\text{Hz}$, $G = 495$ | $100$ | $115$ | -- | $\text{dB}$ |
| **Respuesta en Frecuencia** | | | | | |
| Frecuencia de corte inferior ($f_L$) | Filtro de entrada $33\,\text{nF} + 100\,\text{k}\Omega$ | $42$ | $48.2$ | $54$ | $\text{Hz}$ |
| Frecuencia de corte superior ($f_H$) | $G = 495$, límite interno AD620 | $20$ | $24.2$ | $30$ | $\text{kHz}$ |
| Ancho de banda a $G = 225$ | $R_G = 220\,\Omega$ | $45$ | $53$ | -- | $\text{kHz}$ |
| **Impedancia y Corrientes de Entrada** | | | | | |
| Impedancia de entrada diferencial ($Z_{\text{in}}$) | A $f = 100\,\text{Hz}$ (red de bias) | $90$ | $100$ | $110$ | $\text{k}\Omega$ |
| Impedancia de entrada interna AD620 | Modo común / diferencial | -- | $10^{10} \parallel 2$ | -- | $\Omega \parallel \text{pF}$ |
| Corriente de polarización de entrada ($I_B$) | Terminales inverting / non-inverting | -- | $0.5$ | $2.0$ | $\text{nA}$ |
| Tensión de offset referida a la entrada | A temperatura ambiente | -- | $30$ | $125$ | $\mu\text{V}$ |
| Tensión de offset en reposo a la salida | $G = 495$, entrada aterrizada en DC | -- | $25$ | $95$ | $\text{mV}$ |
| **Etapa de Alimentación y Protección Activa** | | | | | |
| Caída de tensión en MOSFET positivo | F9540N a $I_{\text{carga}} = 50\,\text{mA}$ | -- | $5.5$ | $10$ | $\text{mV}$ |
| Caída de tensión en MOSFET negativo | IRFZ44N a $I_{\text{carga}} = 50\,\text{mA}$ | -- | $1.4$ | $3$ | $\text{mV}$ |
| Tensión de fijación Zener compuerta | Diodos 1N4741A | $10.5$ | $11.0$ | $11.5$ | $\text{V}$ |
| Calibre de los fusibles de protección | FUSE1, FUSE2 (acción rápida) | -- | $0.25$ | -- | $\text{A}$ |

---

## 7. Ecuación de Ganancia del Amplificador AD620

### Ganancia Diferencial del AD620
La ganancia de tensión en bucle cerrado viene dada por la ecuación del fabricante:
$$G = 1 + \frac{49.4\,\text{k}\Omega}{R_G}$$
donde:
- $G$: Ganancia diferencial de tensión ($[\text{V/V}]$).
- $R_G$: Resistencia externa de programación de ganancia ($[\Omega]$).
- $49.4\,\text{k}\Omega$: Suma de las dos resistencias internas de realimentación de precisión del AD620 ($2 \times 24.7\,\text{k}\Omega$).

Valores teóricos para las resistencias evaluadas:
- $R_G = 220\,\Omega \implies G = 225.55\,\text{V/V} \quad (47.06\,\text{dB})$
- $R_G = 100\,\Omega \implies G = 495.00\,\text{V/V} \quad (53.89\,\text{dB})$
- $R_G = 52\,\Omega \implies G = 951.00\,\text{V/V} \quad (59.56\,\text{dB})$
- $R_G = 47\,\Omega \implies G = 1052.06\,\text{V/V} \quad (60.44\,\text{dB})$
- $R_G = 33\,\Omega \implies G = 1497.97\,\text{V/V} \quad (63.51\,\text{dB})$

### Filtro Pasa-Altos Pasivo de Entrada
$$f_L = \frac{1}{2\pi \cdot R_{\text{in}} \cdot C_{\text{in}}} = \frac{1}{2\pi \cdot 100\,\text{k}\Omega \cdot 33\,\text{nF}} \approx 48.23\,\text{Hz}$$
donde:
- $f_L$: Frecuencia de corte inferior a $-3\,\text{dB}$ ($[\text{Hz}]$).
- $R_{\text{in}} = 100\,\text{k}\Omega$: Resistencia de retorno de polarización a masa ($[\Omega]$).
- $C_{\text{in}} = 33\,\text{nF}$: Capacitor cerámico de desacoplo ($[\text{F}]$).

### Función de Transferencia Completa del Canal
$$H(f) = \left( \frac{j \frac{f}{f_L}}{1 + j \frac{f}{f_L}} \right) \cdot \left( \frac{G}{1 + j \frac{f}{f_H(G)}} \right)$$
donde $f_H(G) \approx \frac{1.2 \times 10^7}{G}\,\text{Hz}$ representa el ancho de banda por producto ganancia-frecuencia del AD620.

---

## 8. Caracterización Experimental de Ganancia y Respuesta en Frecuencia

### Curvas Experimentales de Laboratorio
![Curvas Experimentales de Ganancia vs Frecuencia](imagenes/curvas_ganancia_experimental_fig18.png)
*Figura: Curvas de amplificación en función de la resistencia $R_G$. En línea punteada el valor teórico según el fabricante.*

### Curvas Teóricas de Respuesta en Frecuencia
![Curvas Teóricas de Respuesta en Frecuencia](imagenes/curva_respuesta_frecuencia_teorica.png)

### Comparación Cuantitativa a 1 kHz

| $R_G$ Nominal | $G$ Teórica Fabricante | $G$ Medida ($1\,\text{kHz}$) | Error Relativo | Frecuencia $f_H$ Medida |
| :---: | :---: | :---: | :---: | :---: |
| $220\,\Omega$ | $225.5$ | $220.0 \pm 8.5$ | $-2.4\%$ | $50\,\text{kHz}$ |
| $100\,\Omega$ | $495.0$ | $475.2 \pm 14.0$ | $-4.0\%$ | $24\,\text{kHz}$ |
| $52\,\Omega$  | $951.0$ | $1150.0 \pm 35.0$ | $+20.9\%$ | $12\,\text{kHz}$ |
| $47\,\Omega$  | $1052.1$ | $940.0 \pm 28.0$ | $-10.6\%$ | $11\,\text{kHz}$ |
| $33\,\Omega$  | $1498.0$ | $2450.0 \pm 85.0$ | $+63.5\%$ | $7\,\text{kHz}$ |

### Diagnóstico Físico
1. **Atenuación pronunciada bajo 50 Hz:** El corte pasivo en $f_L \approx 48.23\,\text{Hz}$ causa que a $10\,\text{Hz}$ la ganancia caiga a menos del $20\%$ de la meseta, verificando la supresión del contenido espectral bioeléctrico de baja frecuencia.
2. **Meseta estable en ganancias intermedias ($100\,\Omega$ y $220\,\Omega$):** Presentan una respuesta plana y homogénea entre $100\,\text{Hz}$ y $4\,\text{kHz}$, con excelente fidelidad frente al valor teórico del fabricante (error menor al $4\%$).
3. **Sobreelevación a ganancia extrema ($33\,\Omega$):** A $R_G = 33\,\Omega$, la ganancia experimental alcanza $\approx 2450$ (frente a $1498$ teórico). Esto se origina por efectos parásitos de inductancia en pistas del PCB y reducción del margen de fase interno del AD620 en el límite de bucle abierto, produciendo sobreelevación resonante (*peaking*).

---

## 9. Modificación Propuesta para Optimización de Impedancia y Despegues

### Diagnóstico de las Limitaciones Actuales
- **Baja impedancia de entrada ($Z_{\text{in}} \approx 100\,\text{k}\Omega$):** Ante despegues parciales del electrodo ($Z_e \approx 100\,\text{k}\Omega$), el divisor de tensión resistivo $\alpha = \frac{Z_{\text{in}}}{Z_{\text{in}} + Z_e}$ atenúa la señal bioeléctrica al $50\%$, destruyendo los ratios intermusculares fisiológicos.
- **Corte elevado a 48 Hz:** Cercena más del $60\%$ de la energía del sEMG orofacial (disparo de unidades motoras en habla entre $20\text{ y }45\,\text{Hz}$).

### Protocolo de Modificación no Destructiva
1. **Puentear los capacitores de entrada ($C_1\text{--}C_6 = 33\,\text{nF}$):** Cortocircuitar con alambre fino desde arriba. Conexión directa en continua hacia el AD620.
2. **Elevar resistencias de polarización a $10\,\text{M}\Omega$:** Reemplazar las resistencias de $100\,\text{k}\Omega$ por resistores de $10\,\text{M}\Omega$. La impedancia de entrada sube a $Z_{\text{in}} \ge 10\,\text{M}\Omega$, reduciendo la atenuación ante despegues a menos del $1\%$ ($\alpha \ge 0.99$).
3. **Desacoplo AC en el lazo de ganancia ($C_G = 100\,\mu\text{F}$):** Cortar un terminal de $R_G$ elevada en el aire y conectar en serie un capacitor no polarizado de $100\,\mu\text{F}$.
   $$G_{\text{DC}} = 1.0\,\text{V/V}, \quad G_{\text{AC}} \approx 495.0\,\text{V/V}, \quad f_c = \frac{1}{2\pi \cdot 100\,\Omega \cdot 100\,\mu\text{F}} \approx 15.91\,\text{Hz}$$
   Al fijar $G_{\text{DC}} = 1$, los saltos galvánicos de continua por despegue ($20\text{ a }50\,\text{mV}$) no saturan el amplificador contra los rieles de batería, y el corte en $15.91\,\text{Hz}$ recupera la banda muscular completa de $20\text{ a }48\,\text{Hz}$.

![Esquema Vectorial del Canal Modificado (Rg en serie con Cg)](imagenes/esquema_mejora_circuitikz.png)

![Comparación de Respuesta en Frecuencia e Impedancia](imagenes/comparacion_respuesta_frecuencia_impedancia.png)




