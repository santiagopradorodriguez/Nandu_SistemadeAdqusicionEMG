# Registro de Laboratorio: Experimentos del 2026-10-23
**Sistema Ñandú EMG - Laboratorio de Sistemas Dinámicos (LSD, FCEN-UBA)**

---

## 1. Condiciones Ambientales y Estado Eléctrico
- **Fecha:** 2026-10-23
- **Temperatura:** 21.0 °C
- **Humedad Relativa:** 70.0% HR (18.0 °C y 93% HR en exploración previa matutina)
- **Ruido:** Bajo ruido eléctrico y acústico, presencia de personas en laboratorio.
- **Tensión de Baterías:**
  - Batería 1: -8.63 V
  - Batería 2: -8.57 V
  - Desbalance (|ΔV|): 0.06 V (ambas sobre umbral crítico de 8.5 V; simetría balanceada, sin riesgo de offset DC ni clipping).

---

## 2. Topología Anatómica y Geometría de Electrodos
### Canal 0 - Masetero (Posición Sub-Malar Alta Seleccionada)
- **Tipo de electrodos:** Parches de 24 mm de diámetro recortados a la mitad (semi-grandes) para reducir masa y evitar puentes salinos por humedad.
- **Separación entre electrodos:**
  - Distancia libre entre bordes: 0.5 cm (5 mm).
  - Distancia inter-electrodo (IED centro a centro): 2.9 cm (29 mm).
- **Posicionamiento métrico:**
  - Electrodo superior: centro a 2.5 cm del lóbulo de la oreja (hacia adelante y abajo, vientre alto).
  - Electrodo inferior: centro a 6.0 cm de la comisura labial (hacia atrás y abajo, despejando modíolo y risorio).
- **Dinámica articulatoria en /a/ vertical pura:** Descenso mandibular limpio (valle durante apertura y disparo de frenado oclusal tardío al ocluir mandíbula).
- **Dinámica articulatoria en /i/:** Disparo anticipado de estabilización mandibular pre-fonatorio.

### Canal 1 - Modíolo (Nodo Comisural)
- **Tipo de electrodos:** Recortados a 10 mm, IED = 15 mm, bordes libres >= 5 mm.
- **Incidencia de Campo Crítica:** Adhesivo inestable; sostenido con presión manual continua con el dedo durante toda la sesión.
- **Impacto Biofísico:**
  - Reducción de impedancia de contacto y aumento aparente de SNR.
  - Deriva lenta de amplitud por fluctuación involuntaria de la presión digital.
  - Absorción obligatoria mediante Normalización por Supremo Tricanal ($M_{\text{supremo, pulso}}$) en cada pulso.

### Canal 2 - Orbicularis Oris (Esfínter Labial)
- **Tipo de electrodos:** Recortados a 10 mm, IED = 15 mm. Borde labial superior/inferior cerca de comisura.

### Canal 3 - Micrófono Acústico
- **Función:** Sincronismo fonatorio (ai3).

### Referencia y Tierra
- **Ubicación:** Apófisis mastoides detrás de la oreja.

### Canales de Comparación Exploratoria Previa
- Vientre anterior del digástrico (Anterior Belly, 24 mm recortados) y Orbicularis Oris para contraste de apertura vertical.

---

## 2.1 Registro Fotográfico de Montajes y Materiales
- **Experimento 1 (Masetero Perfil):** `fotos_montaje/exp1_masetero_perfil.png` (disposición vertical de parches semi-grandes en mejilla, electrodo superior negro bajo arco cigomático, inferior rojo sobre masetero).
- **Experimento 2 (Belly + Orbicularis):** `fotos_montaje/exp2_belly_orbicularis_01.png` y `fotos_montaje/exp2_belly_orbicularis_02.png` (vista submentoniana de Anterior Belly en piso bucal y parches en labio superior Orbicularis Oris).
- **Experimento 3 (Tríada Masetero - Modíolo - Orbicularis):**
  - `fotos_montaje/exp3_triada_modiolo_orbi_perfil.png` (montaje del modíolo con cinta marrón y electrodos peribucales de Orbicularis y referencia de mastoides).
  - `fotos_montaje/exp3_triada_masetero_camara.png` (captura de cámara frontal/lateral durante la grabación con los 3 canales activos).
- **Materiales y Parches Utilizados:**
  - `fotos_montaje/electrodos_recortados_bandeja.png` (fotografía en bandeja quirúrgica de los parches Medi-Trace: electrodo de 24 mm, dos semi-grandes recortados para masetero/belly y dos micro-recortados a ~10 mm para modíolo y orbicularis).
  - `fotos_montaje/electrodos_recortados_tallas.png` (segunda toma en bandeja con gradación completa de tallas recortadas: 2 micro, 2 intermedias, 2 semi-grandes y 1 base).
- **Setup Experimental del Laboratorio:**
  - `fotos_montaje/setup_experimental_laboratorio.png` (mesa de trabajo, placa analógica AD620 con alimentación a baterías y salidas BNC, calibre vernier para medición milimétrica IED y distancias craneofaciales, micrófono direccional con antipop para canal 3 y blindaje de fondo).

---

## 3. Tomas Grabadas por Sujeto 1
### A. Tanda Discreta Vocálica - Pruebatricanal
- **Hora de inicio:** 16:09:11
- **Régimen:** 10 repeticiones por toma.
- **Volumen:** 20 tomas x 10 repeticiones = 200 pulsos (40 muestras exactas balanceadas por cada vocal /a/, /e/, /i/, /o/, /u/).
- **Secuencia ejecutada:**
  1. `/A/`
  2. `/O/`
  3. `/O/`
  4. `/U/`
  5. `/I/`
  6. `/U/`
  7. `/I/`
  8. `/U/`
  9. `/E/`
  10. `/E/`
  11. `/O/`
  12. `/E/`
  13. `/U/`
  14. `/A/`
  15. `/A/`
  16. `/O/`
  17. `/I/`
  18. `/I/`
  19. `/E/`
  20. `/A/`

### B. Tanda de Coarticulación y Sílabas CV - PruebaSILABASSEGUNDA
- **Hora de inicio:** 16:22:36 (13 min 25 s posteriores)
- **Régimen:** 7 repeticiones por toma.
- **Volumen:** 10 tomas x 7 repeticiones = 70 pulsos.
- **Secuencia ejecutada:**
  `/TA/` -> `/MA/` -> `/RA/` -> `/MA/` -> `/LA/` -> `/JA/` -> `/PA/` -> `/SA/` -> `/BA/` -> `/KA/`
  (14 repeticiones de `/MA/` y 7 repeticiones de cada una de las 8 sílabas restantes).

---

## 4. Criterio Metodológico
- Prohibición de uso de papers o fuentes externas no verificadas.
- Análisis basado exclusivamente en datos empíricos de adquisición directa, envolventes temporales, $P_{95}$, Supremo Tricanal y diferenciales espaciotemporales ($\tilde{x}_{\text{Orbi}} - \tilde{x}_{\text{Mod}}$ y $\dot{\tilde{x}}$).
