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
- **Reporte de Estado Cada 5 Minutos y Ahorro de Tokens:** Durante tareas largas o ejecuciones en segundo plano, reportar al usuario cada 5 minutos qué se está haciendo y el estado de avance (usando temporizadores `schedule` de 300 s en tareas de fondo). Prohibido encadenar decenas de lecturas o llamadas a herramientas en silencio sin avisar al usuario, para ahorrar tokens.
- **Alineación Previa Mediante Preguntas ante Trabajos Extensos:** Antes de iniciar tareas largas, complejas o multifase, está terminantemente prohibido lanzarse a ejecutar o explorar en solitario por el repositorio ("buscar solo por el mundo"). Es obligatorio detenerse, formular preguntas directas de clarificación al usuario y consensuar el paso a paso antes de procesar o correr cualquier tarea de gran envergadura.

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
- **Formato Estricto del Atlas de Activación Muscular sEMG:**
  - Encabezado: `Experimento YYYY-MM-DD: Nombre_Campana - Patrón de Activación Muscular`. Subtítulo: LSD - UBA y filtros (20-500 Hz + Notch 50 Hz, supremo tricanal por pulso).
  - Sección Superior (3 filas musculares):
    - Tarjeta muscular izquierda (`w_meta = 0.115`): Nombre anatómico en 2 líneas sin paréntesis, función fisiológica y canal sEMG asignado. Sin mención a ganancia o AD620.
    - Fotografía fiduciaria (`w_foto = 0.096`): Relación nativa portrait 3:4 sin barras negras laterales.
    - Tabla biométrica (`w_tabla = 0.370`, divisoria 0.59): Parámetro biométrico vs `VALOR / ESTIMACIÓN (± ERROR)`. Especificar siempre `Distancia Inter-electrodo (IED entre centros, estim.)` con su incertidumbre experimental ($\pm 3.5\,\text{mm}$), tamaño físico de electrodo ($5 \times 5\,\text{mm} \pm 0.02\,\text{mm}$) y distancias a puntos fiduciarios (Gnathion, Pronasale, Tragus, Cheilion) aclarando que son estimadas.
      - Invarianza Estética de la Tabla: Fondo uniforme `#0f172a` (prohibido usar fondos cebra o franjas alternadas). Caja de encabezado `#1e293b`. Líneas divisorias horizontales `axhline` (`#334155`, lw=0.8) entre todas las filas y vertical `axvline` (`#334155`, lw=0.9 en $x = 0.59$). Fila IED resaltada en ámbar (`#fbbf24`), valores restantes en blanco (`#ffffff`, 12 pt bold), etiquetas en `#cbd5e1` (10.5 pt bold, $x = 0.025$).
    - Curvas promedio de activación (5 columnas /a/, /e/, /i/, /o/, /u/): Ventana de $-600\,\text{ms}$ a $+800\,\text{ms}$, referencia acústica en $t=0$, media y sombreado $\pm 1\,\text{std}$, pico de contracción en ms.
  - Sección Media (2 paneles horizontales):
    - Cuaderno de sesión: Datos temporales, hardware (Fs, baterías, referencia mastoides) y registros (tomas grabadas, series, pulsos válidos).
    - Rendimiento clasificador lineal (LDA): Precisión global y macro en espacio PCA 3D con badges individuales por vocal en colores canónicos.
  - Sección Inferior (3 proyecciones ortogonales PCA 3D):
    - PC1 vs PC2, PC1 vs PC3, PC2 vs PC3 con `set_box_aspect(1)` (100% cuadradas).
    - Sin fronteras de decisión artificiales; etiquetas y títulos de subgráficos grandes (18 pt) y ticks legibles (14 pt).
  - Primera diapositiva: Índice cronológico general de campañas.

## 6. Organización de Archivos y Memoria
- **Contención de Códigos Experimentales en Libro de Experimentos IA:**
  Todo nuevo script de prueba, barrido de hiperparámetros o experimento puntual de modelado debe guardarse obligatoriamente dentro de `reportes_experimentos/libro_experimentos_ia/scripts_experimentos/`. Prohibido dispersar scripts experimentales sueltos en `EMG_desarrollo/deep_learning/` o en la raíz del repositorio, preservando limpio el código fuente de producción de la aplicación.
- **Jerarquía Oficial de Resultados:** Guardar modelos en `EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/`, figuras en `figuras_evaluacion/` y datasets en `cache_datasets/`. Prohibido dispersar archivos sueltos en la raíz o en `/tmp/`.
- **Nomenclatura Canónica de Montajes en Drive:**
  Dentro de `fotos_montaje/` de cada campaña, las imágenes deben organizarse sin ambigüedades:
  - Imágenes con electrodos anotados: `01_anotada_...png`, `02_anotada_...png`, `03_anotada_...png`.
  - Imágenes crudas originales: `01_raw_...jpeg`, `02_raw_...jpeg`, `03_raw_...jpeg`.
  Prohibido duplicar fotos con nombres cruzados o descriptores anatómicos invertidos.
- **Memoria del Proyecto:**
  - Al iniciar el chat, consultar `.agents/MEMORIA_ACTIVA.md` (resumen ejecutivo de ~50 líneas).
  - Actualizar `.agents/MEMORIA_ACTIVA.md` **únicamente al cerrar un hito, experimento o cambio de código real** (no en charlas o preguntas teóricas).
  - El historial detallado de hitos reside en `.agents/historial/bitacora_completa_hitos_001_al_138.md` con su índice en `.agents/INDICE_HITOS.md`.
