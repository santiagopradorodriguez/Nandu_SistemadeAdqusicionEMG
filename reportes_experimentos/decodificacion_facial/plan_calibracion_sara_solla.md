# Plan de Trabajo: Calibración por Variedades Latentes (Sara Solla / Procrustes / CCA)

**Fecha de registro:** 2026-11-11  
**Estado:** Listo para implementar en la próxima sesión.

---

## 1. Contexto y Diagnóstico
En la sesión del 2026-11-11 se probó el decodificador facial cinemático en vivo (`autoforge_daq_decodificador_facial.py`) con el modelo campeón convolucional ortogonal 4D (`campeon_geosup_conv_ortho_4d`).
- **Exactitud en vivo lograda:** **73.33% (11/15 aciertos)** sin pulsos perdidos.
- **Dificultad observada:** La vocal **/e/** natural tiene menor amplitud y es canibalizada por **/i/** (probabilidades ~41% vs ~57%).
- **Causa física/geométrica:** Desplazamiento rígido/rotación de la variedad motora en el espacio latente entre sesiones (por iluminación, encuadre de cámara, postura mandibular y variabilidad del día).
- **Datos guardados disponibles:** `auditoria_tiempo_real/ventanas_en_vivo/gate/` con 96 ventanas etiquetadas de hoy (17 /A/, 17 /E/, 21 /I/, 21 /O/, 20 /U/).

---

## 2. Fundamento Matemático (Sara Solla / Gallego et al., 2020)
En lugar de reentrenar la red neuronal convolucional completa:
1. **Centroides Canónicos de Referencia:** $\mu_{\text{ref}}^{(v)} \in \mathbb{R}^4$ del modelo entrenado.
2. **Centroides de la Sesión de Hoy:** $\bar{z}_{\text{hoy}}^{(v)} = \frac{1}{N_v} \sum f_{\theta}(X^{(v)}_i) \in \mathbb{R}^4$.
3. **Alineación de Procrustes Ortogonal (SVD):**
   $$H = \sum_{v \in \{A,E,I,O,U\}} (\bar{z}_{\text{hoy}}^{(v)} - \bar{Z}_{\text{hoy}})(\mu_{\text{ref}}^{(v)} - \bar{\mu}_{\text{ref}})^T = U S V^T \implies R = V U^T$$
   (Con $\det(R) = 1$ para evitar reflexiones espurias).
4. **Transformación en Tiempo Real:**
   Para cualquier nueva ventana entrante $x$:
   $$z = f_{\theta}(x), \quad z_{\text{alineado}} = R (z - \bar{Z}_{\text{hoy}}) + \bar{\mu}_{\text{ref}}$$
5. **Decodificación LDA:** Se evalúa $z_{\text{alineado}}$ sobre el LDA canónico ya validado.

---

## 3. Tareas para la Próxima Semana
1. **Motor de Decodificación (`realtime_facial_decoder_engine.py`):**
   - Agregar método `alinear_variedad_procrustes(ventanas_dict)` que calcula $R$ y los offsets.
   - En `predict_window()`, aplicar la transformación afín si la alineación está activa.
2. **Interfaz de Usuario (`autoforge_daq_decodificador_facial.py`):**
   - Agregar botón `[ ALINEAR VARIEDAD (SARA SOLLA) ]` en el panel de calibración/test.
   - Al presionarlo, cargar las ventanas de `auditoria_tiempo_real/ventanas_en_vivo/gate/` de la sesión actual, calcular $R$ y actualizar el motor con feedback visual (mostrando el error de Procrustes residual).
3. **Validación en Vivo:**
   - Correr nuevo test de 15 fonaciones con la variedad alineada y medir el salto de exactitud esperado (meta: 85% a 90%+).
