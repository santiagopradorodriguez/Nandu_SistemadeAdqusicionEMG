# Protocolo Experimental: Comparativa sEMG Tricanal entre Mímica Facial Superficial y Fuente Fonatoria Lingual

**Proyecto:** Ñandú EMG — Laboratorio de Señales y Dinámica - UBA  
**Sujeto:** Sujeto experimental  
**Fecha:** `____ / ____ / ________`  
**Operador:** `________________________`  

---

## 1. Justificación, Problema e Hipótesis

En campañas previas con montaje de Cigomático Mayor se observó una alta separabilidad geométrica en el espacio latente, atribuible a una posible sobrearticulación mímica. Al registrar habla natural a volumen normal, dicha separabilidad disminuyó.  
**Hipótesis de trabajo:** Determinar si la señal sEMG tricanal captura gestos mímicos superficiales producto de la sobrearticulación o la actividad neuromuscular intrínseca de la fuente fonatoria vocálica.  
**Diseño:** Ejecución secuencial obligatoria de **dos experimentos de montaje independientes** en la misma sesión para contrastar la superficie facial frente a la musculatura profunda del suelo de la boca.

---

## 2. Preparación Fiduciaria y Parámetros de Hardware

* **Preparación de piel:** Exclusivamente crema abrasiva, sin alcohol. Verificación de ruido basal `< 10 µV` en todos los canales antes de iniciar.
* **Electrodos:** Adhesivos pequeños para sEMG facial en configuración bipolar longitudinal con la mínima distancia inter-electrodo (`IED`) posible entre centros; reborde bermellón con electrodos chicos (`IED` mínima) y resto con electrodos de `1.5 cm`.
* **Adquisición y acondicionamiento:** Frecuencia de muestreo sEMG $f_s = 2000\text{ Hz}$ por canal. Amplificador de instrumentación diferencial con ganancia fija $G \approx 495\text{ V/V}$ ($R_G = 100\,\Omega$). Filtrado hardware/software: pasa-altos $20\text{ Hz}$, pasa-bajos $500\text{ Hz}$, Notch $50\text{ Hz}$.
* **Canales fijos en ambos experimentos:**
  * **Ch0:** Vientre anterior del digástrico — submentoniano medial, apertura mandibular.
  * **Ch2:** Orbicular de la boca superior — reborde bermellón superior, constricción y redondeo labial `/o, u/`.
  * **Ch3:** Micrófono acústico de sincronización — $f_s = 44.1\text{ kHz}$ de referencia.
  * **Referencia eléctrica:** Apófisis mastoides.

---

## 3. Procedimiento Experimental Secuencial

### Fase 1 — Experimento 1: Tríada Cigomático Mayor - Superficie y Gestos Faciales
**Objetivo:** Evaluar la mímica facial superficial y la tracción de comisuras labiales.  
**Canal diferencial Ch1:** Cigomático Mayor en mejilla, tracción de comisuras `/e, i/`. Colocación estricta según registro fotográfico fiduciario histórico de la primera campaña y de la penúltima sesión.

1. **Bloque 1A:** Vocales sostenidas fonadas a volumen natural (`/a, e, i, o, u/`).
2. **Bloque 1B:** Mímica silenciosa no fonada — gesto labial idéntico sin emisión fónica.
3. **Bloque 1C:** Habla natural continua.

### Fase 2 — Experimento 2: Tríada Milohioideo Lateral - Fuente Lingual y Suelo de Boca
**Objetivo:** Medir directamente la fuente motora generadora de las vocales en base de lengua y suelo bucal, eliminando la dependencia del gesto superficial.  
**Canal diferencial Ch1:** Milohioideo lateral / sensor lingual en región submentoniana lateral profunda, sensando raíz lingual, deglución y fonación.

1. **Bloque 2A:** Vocales sostenidas fonadas a volumen natural (`/a, e, i, o, u/`).
2. **Bloque 2B:** Mímica silenciosa no fonada — control sin activación lingual profunda.
3. **Bloque 2C:** Habla natural continua.
