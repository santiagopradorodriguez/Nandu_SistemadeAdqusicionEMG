# Reglas del Proyecto (Ñandú EMG)

## 1. Reglas Cardinales de Interacción
- **Cero Emojis:** Prohibido usar emojis en código, documentación o respuestas.
- **Edición No Destructiva en Código y LaTeX:** Prohibido borrar, truncar o alterar código o texto existente al agregar nuevas secciones, tablas o imágenes, salvo instrucción explícita del usuario. En archivos `.tex`, toda edición debe ser estrictamente aditiva, auditando que `git diff --stat` registre cero eliminaciones.
- **Trazabilidad de Reportes Múltiples:** Identificar y explicitar siempre en qué archivo de reporte reside cada cuerpo de análisis (ej. `apunte_arquitectura_red_y_secuencia_continua.tex` para MLPs, Gate Doble y P5 vs `reporte_autoencoder_decodificacion_vocalica.tex` para variedades motoras y transferencia inter-sujeto) para evitar confusiones de contenido.
- **Prohibición de Ejecutar Código sin Permiso:** Prohibido correr comandos (`run_command`), scripts o tests sin confirmación previa del usuario.
- **Prohibición de Búsqueda Local sin Permiso:** Prohibido buscar o leer archivos `.py` del repositorio ante preguntas conceptuales o generales, para ahorrar tokens.
- **Estilo de Comunicación:** Cortito y al pie, lenguaje humano y cotidiano de cuaderno de laboratorio. Prohibido tono de paper inflado, palabras de relleno y conectores típicos de IA.
- **Corrección Literal:** Al aplicar correcciones del usuario ("correccion: ..."), no alterar la estructura ni inventar palabras; corregir solo ortografía y espacios.
- **Títulos Limpios:** Prohibido usar paréntesis `(...)` en títulos y encabezados de texto, diagramas o gráficos. Usar `:` o `-`.

## 2. Base de Datos y Metadatos sEMG
- **Estructura de Canales:** Los archivos residen obligatoriamente en subcarpetas `canal_0/`, `canal_1/`, `canal_2/`, `canal_3/` dentro de cada sesión. El `metadata.json` principal está en `canal_0/`.
- **Nomenclatura Anatómica Oficial:** Auditar `metadata.json` por toma. Para tomas estándar: Ch0 = Vientre anterior (`Anterior Belly`), Ch1 = Cigomático/Depresor (`Zygomaticus Major`/`Depresor Anguli Oris`), Ch2 = Orbicular (`Orbicularis Oris`), Ch3 = Micrófono.
- **Fecha Real de Medición:** Extraer siempre del campo `"measurement_date"` en `canal_0/metadata.json`, nunca del nombre de la carpeta superior.
- **Corte Guiado por Envolvente:** La señal cruda rectificada se segmenta usando los mismos límites temporales $[n_{\text{inicio}}, n_{\text{fin}}]$ determinados por la envolvente o micrófono.

## 3. Acondicionamiento de Señal y DSP
- **Normalización Obligatoria por Supremo Tricanal por Pulso Individual:**
  Prohibido normalizar canales independientes o por el máximo de la sesión. Normalizar cada pulso por su supremo intercanal:
  $$M_{\text{supremo, pulso}} = \max_{c \in \{0, 1, 2\}} \left( \max_{t \in \text{ventana}} |x_c(t)| \right), \quad \tilde{x}_c(t) = \frac{|x_c(t)|}{M_{\text{supremo, pulso}}}$$
- **Piso de Ruido Dinámico Interpulso:** Calcular el ruido local en los descansos pre y post pulso depurando espigas con rango intercuartil ($Q_3 + 1.5 \times \text{IQR}$). Prohibido descartar pulsos mediante umbrales fijos de SNR.
- **Purga de Artefactos:** Exclusivamente con *Isolation Forest* (10% contaminación máxima sobre envolvente suave).
- **Hardware Invariante:** Amplificador AD620 con resistencia soldada $R_G = 100\,\Omega \implies G \approx 495\,\text{V/V}$ fija. Prohibido describirla como programable.

## 4. Modelado y Deep Learning
- **Zero-Labels Estricto:** Autoencoders entrenados 100% sin supervisión. Prohibido usar etiquetas de vocales en funciones de pérdida o gradientes (`MSE(x, \hat{x})` y regularizaciones no supervisadas). Las etiquetas se usan únicamente para evaluación externa post-entrenamiento (GMM/K-Means con asignación húngara).
- **Semilla Fija Universal:** Todo script, entrenamiento o partición debe fijar `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`.
- **Balance Multiclase Obligatorio:** Evaluar armónicamente las 5 vocales (/a/, /e/, /i/, /o/, /u/). Un modelo solo es superador si sube el rendimiento global sin canibalizar ninguna vocal.
- **Rigor en Comparativas (Ceteris Paribus):** Al ensayar nuevas variantes, aislar estrictamente la variable y mantener idéntica la escala de entrada, tamaño de lote, tasa de aprendizaje ($\eta = 0.008$) y 250 épocas.
- **Monitoreo de Progreso:** Todo bucle largo o entrenamiento debe imprimir porcentaje, pérdida, ETA y tiempo por época. Estimar tiempos previamente al usuario.

## 5. Visualización y Convenciones
- **Anclaje Canónico de la Cruz Latente:** En gráficos 2D inter-sujeto, /a/ debe orientarse al semieje vertical positivo ($+Y$, $\theta = 90^\circ$, 12 en punto) y sonrisas (/i/, /e/) al semiplano derecho ($+X > 0$).
- **Código de Colores Universal:**
  - **/a/**: Rojo (`#E63946`)
  - **/e/**: Azul (`#1F77B4`)
  - **/i/**: Verde (`#2CA02C`)
  - **/o/**: Morado (`#9D4EDD`)
  - **/u/**: Amarillo (`#E7A61A`)
- **Reportes LaTeX (fancyhdr):** Configurar solo `\fancyhead[L]` y `\fancyhead[R]`. Dejar `\fancyhead[C]` vacío para evitar solapamientos.

## 6. Organización de Archivos y Memoria
- **Jerarquía Oficial de Resultados:** Guardar modelos en `EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/`, figuras en `figuras_evaluacion/` y datasets en `cache_datasets/`. Prohibido dispersar archivos sueltos en la raíz o en `/tmp/`.
- **Memoria del Proyecto:**
  - Al iniciar el chat, consultar `.agents/MEMORIA_ACTIVA.md` (resumen ejecutivo de ~50 líneas).
  - Actualizar `.agents/MEMORIA_ACTIVA.md` **únicamente al cerrar un hito, experimento o cambio de código real** (no en charlas o preguntas teóricas).
  - El historial detallado de hitos reside en `.agents/historial/bitacora_completa_hitos_001_al_138.md` con su índice en `.agents/INDICE_HITOS.md`.
