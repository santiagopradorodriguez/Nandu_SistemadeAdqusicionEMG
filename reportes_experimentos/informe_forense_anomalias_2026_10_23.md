# Informe Forense de Anomalias y Separabilidad: Sesion 2026-10-23 vs Sesiones Previas

## 1. Resumen Ejecutivo: Por que parecia que todo daba mal

Al auditar el codigo fuente, los metadatos JSON, las señales crudas `.wav` y los 5 videos grabados durante la sesion del `2026-10-23`, encontramos que la caida aparente de rendimiento no provino de una sola causa, sino de la combinacion de **tres bugs silenciosos en el codigo de PCA/Reportes** y **tres anomalias fisico-biomecanicas reales en la toma de datos**:

### A. Causas de Software (Ya corregidas en el repositorio)
1. **Bug 1 - `NameError` silencioso en `aplicar_pesos_canales` durante el Grid Search (`pca_analysis.py` y `generador_pca_umap.py`):**
   - La funcion `aplicar_pesos_canales` estaba definida unicamente como funcion local anidada dentro de `ejecutar_procesamiento`, pero era invocada desde `buscar_mejor_configuracion_pca` (linea 2107). Cada vez que el Grid Search intentaba evaluar una combinacion, caia en el bloque `except Exception` por `NameError: name 'aplicar_pesos_canales' is not defined`.
2. **Bug 2 - Indexacion posicional rota de `pesos_canales` en pares musculares (`C0-C2`):**
   - En el Experimento 2 (`pruebabelly`), los canales activos son `canal_0` (`orbi`) y `canal_2` (`belly`), mientras que `canal_1` esta desconectado (`nada`). Cuando se seleccionaba el par `["canal_0", "canal_2"]` con pesos `[1.0, 1.0, 2.5]`, `aplicar_pesos_canales` leia `pesos[0] = 1.0` y `pesos[1] = 1.0` (el peso del canal desconectado), ignorando por completo `pesos[2] = 2.5` del `belly`.
3. **Bug 3 - Normalizacion tricanal forzada sobre canales desconectados en Experimento 1 (`Prueba1`):**
   - En el Experimento 1 (monocanal `Masetero` en `canal_0`), `canal_1` y `canal_2` estaban abiertos (`nada`). En particular, `canal_2` tenia **546.5 uV constantes de ruido de 50 Hz** de cable flotante. El motor de reportes forzaba siempre `["canal_0", "canal_1", "canal_2"]`, dividiendo la señal del masetero por el ruido flotante del canal 2 y metiendo dos tercios de columnas de ruido puro al PCA. Al aislar `["canal_0"]` como PCA monocanal dedicado, la exactitud macro de 5 vocales con un unico electrodo en el masetero salto a **62.51% en 2D** (`/I/: 100.0%`, `/O/: 83.3%`, `/U/: 60.0%`, `/A/: 50.0%`).
4. **Bug 4 - Frecuencia de muestreo fija en `2000 Hz` frente a grabaciones a `6000 Hz` y calibracion intersesion activa:**
   - Mientras que `Prueba1` y `pruebabelly` se grabaron a `2000 Hz`, **`Pruebatricanal` se grabo a `6000 Hz`**. En `pca_analysis.py` y `generador_pca_umap.py`, el detector de ventana central y autocorrelacion tenia `2000` hardcodeado (`int(0.250 * 2000)` y `lags_50[0] / 2000.0`), achicando 3 veces la ventana temporal a `6000 Hz`. Ademas, `report_engine.py` dejaba `aplicar_correccion_intersesion=True` por defecto, tratando `Pruebatricanal`, `Pruebatricanal_2`, `Pruebatricanal_3` y `Pruebatricanal_4` como sesiones distintas y alterando las ganancias relativas entre series.

---

### B. Anomalias Fisicas y Biomecanicas Detectadas en Videos y Señales

1. **Desprendimiento mecanico del electrodo del Modiolo (`canal_1`) durante `Pruebatricanal` (Evidencia en video):**
   - En los videos `WIN_20261023_15_39_13_Pro.mp4` y `WIN_20261023_15_46_48_Pro.mp4` (grabados exactamente entre las 15:39 y las 16:02 hs durante `Pruebatricanal`), se observa que el electrodo recortado (`15 mm x 20 mm`) ubicado a `1.0 cm` de la comisura labial izquierda (**Modiolo, `canal_1`**) se despega repetidamente por el pliegue mecanico de la comisura al hablar.
   - En varias tomas, Candela tiene que sostener y presionar el electrodo del modiolo con el dedo indice mientras pronuncia las vocales. A mitad del video `WIN_20261023_15_46_48_Pro.mp4`, le colocan **curitas marrones** sobre el modiolo y sobre la referencia del cuello.
   - Esto genero explosiones de ruido de linea de base en `canal_1` especificamente en **4 tomas**:
     - `A_Pruebatricanal_2_Sujeto1`
     - `E_Pruebatricanal_Sujeto1`
     - `E_Pruebatricanal_2_Sujeto1`
     - `I_Pruebatricanal_4_Sujeto1`
   - **Prueba de impacto:** En `Pruebatricanal` con `Masetero x2.5`, al purgar unicamente esas 4 tomas donde el electrodo del modiolo se despego, la exactitud no supervisada GMM **sube de 53.72% a 58.40% en PCA 2D, de 50.19% a 62.03% en PCA 3D (+12.8 puntos) y a 68.66% en PCA+LDA**, logrando balance real en las 5 vocales (`A: 56.5%`, `E: 75.0%`, `I: 50.0%`, `O: 55.9%`, `U: 72.7%`).

2. **Por que el Masetero no separo `/a, e, i/` por amplitud como se esperaba de los reportes previos (`16/09` y `20/10`):**
   - En las sesiones historicas de alta separabilidad (`16/09/2025` con `79.3%` y `20/10/2026`), el canal que separaba `/i/` y `/e/` estaba colocado en el **puente submalar alto** (`3.0-3.5 cm` debajo del canto externo del ojo, captando el *Cigomatico Mayor* junto con fibras anteriores del masetero), mientras que `/a/` se separaba con el *Anterior Belly* submentoniano.
   - En cambio, el `2026-10-23` el Masetero (`canal_0`) se coloco en **disposicion vertical baja y posterior sobre la rama mandibular** (`2.5 cm` debajo del lobulo de la oreja hacia el gonion y a `6.0 cm` de la comisura, lejos del cigomatico).
   - En esa posicion posterior del masetero:
     - `/O/` y `/U/` tienen amplitud casi nula (`P95 = 13-16 uV`), por lo que **el Masetero SI separa perfectamente el bloque `{A, E, I}` del bloque `{O, U}`** (razon por la cual al multiplicar `Masetero x2.5` la vocal `/U/` sube del `18.2%` al `51.5%` en 2D y al `78.8%` en 3D).
     - Sin embargo, dentro de `{A, E, I}`, las tres vocales producen casi la misma amplitud pico en el masetero posterior (`P95 = 38-47 uV`): `/I/` y `/E/` por tension mandibular oclusal y `/A/` por el rebote de cierre mandibular.
     - **Que separa a `/A/`, `/E/` e `/I/` en el masetero posterior:** La **fase temporal respecto al microfono** y la derivada: en `/I/` la activacion es pre-fonatoria, mientras que en `/A/` es post-fonatoria (cierre). Al agregar el centro de masa temporal y la derivada sobre ventana extendida (`pre=0.5, post=0.7`), `/A/` e `/I/` suben ambas a **84.6% de exactitud** incluso usando solo el canal del masetero.

3. **Por que el Modiolo y el Orbicularis tuvieron solapamiento entre `/O/` y `/U/` en `Pruebatricanal`:**
   - Al colocar el Modiolo a solo `1.0 cm` de la comisura y fijarlo bajo la misma cinta/curita que cubria hacia el labio superior (`Orbicularis Oris`), ambos canales registraron co-activacion fuerte tanto en `/O/` (`C1 = 150 uV, C2 = 127 uV`) como en `/U/` (`C1 = 154 uV, C2 = 125 uV`).
   - En `Exp 2 (pruebabelly)`, donde `Orbicularis` (`C0`) y `Anterior Belly` (`C2`) estan anatomicamente lejos y sin acople mecanico de cinta, la separabilidad lineal (PCA+LDA) alcanza **77.50%** y `/O/` se clasifica al **100.0%** en GMM no supervisado.

4. **Rafaga espuria durante los primeros 3 segundos de `Ruido Basal` en el Experimento 1 (`Prueba1`):**
   - En las 13 tomas del Experimento 1 (`A_Prueba1` a `U_Prueba1`), el canal `Masetero` presenta una banda de alta energia (`100-200 uV`, ratio de explosion basal de `2.1x a 5.1x`) exactamente entre `t = 0 s` y `t = 3 s` (durante la ventana de `Ruido Basal`), que desaparece abruptamente en `t = 3.0 s` cuando comienza la secuencia de pulsos. Por eso fue clave usar el ruido interpulso dinamico IQR y **desactivar la correccion de impedancia por P95 del reposo basal** tal como indicaste.

---

## 2. Tabla Comparativa de Ablacion y Separabilidad (2026-10-23)

| Experimento | Configuracion Evaluada | PCA 2D GMM (%) | PCA 3D GMM (%) | PCA + LDA (%) | Observacion Principal |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp 1: Masetero (C0)** | Grid Search Oficial Monocanal (`smooth=50ms, pts=100, alpha=0.5`) | **62.51%** | **58.90%** | -- | `/I/: 100%`, `/O/: 83.3%`, `/U/: 60.0%`, `/A/: 50.0%` |
| **Exp 1: Masetero (C0)** | Base Monocanal (`pre=0.4, post=0.6`) | 51.72% | 51.91% | 55.89% | Separa `{A,I,O}` pero confunde `/E/` y `/U/` |
| **Exp 1: Masetero (C0)** | Ventana Extendida (`0.5/0.7`) + Fase Temporal y Derivada | **57.25%** | 49.56% | **62.09%** | Sube `/A/` a **84.6%** e `/I/` a **84.6%** por fase pre/post |
| **Exp 2: Belly + Orbi (C0, C2)** | Base Bicanal (`Pesos 1:1`) | 55.00% | **58.33%** | **77.50%** | En 3D logra `/I/: 100%`, `/O/: 100%`, `/U/: 50%` |
| **Exp 2: Belly + Orbi (C0, C2)** | Bicanal con `Belly x2.5` (`[1.0, 2.5]`) | 55.00% | 50.83% | 75.83% | En 2D logra `/O/: 100%`, `/I/: 83.3%`, `/U/: 50%` |
| **Exp 2: Belly + Orbi (C0, C2)** | Bicanal `Belly x2.5` + Diferencial Espacial + Fase | **56.67%** | 50.00% | 73.33% | Mejora separacion 2D de `/A/` (`33.3%`) y `/U/` (`50.0%`) |
| **Exp 3: Triada (C0, C1, C2)** | Base Triada (`Pesos 1:1:1`) | 47.98% | 49.17% | 60.03% | `/U/` cae a `18.2%` en 2D por baja amplitud de Masetero |
| **Exp 3: Triada (C0, C1, C2)** | Triada con `Masetero x2.5` (`[2.5, 1.0, 1.0]`) | **53.72%** | 50.19% | 56.28% | `/U/` salta de `18.2%` a **51.5%** en 2D y **78.8%** en 3D; `/A/` sube de `35.5%` a **54.8%** |
| **Exp 3: Triada (C0, C1, C2)** | Triada con `Masetero x3.0` (`[3.0, 1.0, 1.0]`) | 52.59% | 50.66% | 56.77% | `/A/` llega a **58.1%** en 2D, leve sobrepeso frente a `x2.5` |
| **Exp 3: Triada (C0, C1, C2)** | **Triada `Masetero x2.5` Purgando las 4 Tomas con Modiolo Despegado** | **58.40%** | **62.03%** | **68.66%** | **Salto de +12.8% en 3D y balance en las 5 vocales (`A:56.5%, E:75.0%, I:50.0%, O:55.9%, U:72.7%`)** |
| **Exp 3: Triada (C0, C1, C2)** | Triada `Masetero x2.5` + Purga + Diferencial `Orbi-Mod` + Fase | 56.75% | 55.39% | 62.90% | Lleva la vocal `/A/` al **82.6%** y `/O/` al **69.7%** |
