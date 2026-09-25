# Reglas del Proyecto (Ñandú EMG)

## Estructura de Subcarpetas de la Base de Datos EMG
Para evitar errores al leer los archivos de grabaciones (WAVs y JSONs), la estructura oficial y obligatoria de la base de datos de electrodos es la siguiente:

```
base_de_datos_electrodos/
└── <Fecha> (ej. 2026-06-10) /
    └── <Sesión> (ej. SecuenciaContinua_Prueba5_Sujeto1, A_T1_Lucas, etc.) /
        ├── canal_0/
        │   ├── grabacion.wav
        │   └── metadata.json
        ├── canal_1/
        │   ├── grabacion.wav
        │   └── metadata.json (Opcional)
        ├── canal_2/
        │   ├── grabacion.wav
        │   └── metadata.json (Opcional)
        └── canal_3/
            ├── grabacion.wav
            └── metadata.json (Opcional)
```

**Regla de Oro para el procesamiento de audio:**
Cualquier módulo DSP o de Machine Learning que necesite acceder a los datos, NO debe buscar los archivos `.wav` ni `metadata.json` en la raíz de la sesión (`<Sesión>/`), sino que **obligatoriamente debe iterar o acceder a las subcarpetas `canal_0`, `canal_1`, `canal_2` y `canal_3`**. El `metadata.json` principal (que contiene BPM, date, etc.) se encuentra siempre dentro de `canal_0`.

## Conservación Estricta de Contenido (Edición No Destructiva)
Cuando el usuario solicite "agregar", "insertar" o "poner" una nueva imagen, sección o bloque de texto, está **terminantemente prohibido** borrar, reemplazar o alterar de forma colateral cualquier otro texto, código o sección adyacente.
La edición debe ser puramente aditiva y conservadora, a menos que el usuario instruya explícitamente borrar o reemplazar contenido.

## Redacción Matemática y Exposición Pedagógica de Código
1. **Explicación Exhaustiva de Variables en Fórmulas:**
   Toda ecuación o fórmula matemática debe acompañarse inmediatamente de un desglose explícito y detallado de cada una de sus variables, parámetros y subíndices (ej. $t_{\text{ruido}}$, $f_s$, $n_{\text{inicio}}$, $W_{\text{ciclo}}$, $\text{RMS}[n]$, etc.), aclarando su significado físico, unidades (ej. Hz, s, muestras) y rol en el pipeline.

2. **Nomenclatura en Español y Términos Técnicos Universales:**
   Las ecuaciones, subíndices y textos explicativos deben redactarse en español (ej. $n_{\text{inicio, ruido}}$, $\tau_{\text{suavizado}}$, $|H_{\text{PA}}|$, $|H_{\text{PB}}|$, $\mu_{\text{ruido}}$, $\text{RIC}$, $\text{DAM}$), manteniendo en su denominación técnica universal en inglés únicamente aquellos conceptos estándar de la disciplina donde una traducción literal resulte antinatural o confusa (ej. $H_{\text{notch}}$, $\text{RMS}$, $\text{SNR}$, $\text{IQR}$, $\text{MAD}$).

3. **Intercalación Didáctica de Código por Pasos:**
   Al documentar algoritmos, secuencias de procesamiento o etapas de cálculo (ej. segmentación periódica, cascada de filtros DSP, extracción de envolventes), cada fragmento de código Python debe ubicarse inmediatamente debajo de la viñeta o paso explicativo correspondiente, evitando agrupar todo el código en bloques monolíticos al final de la sección.

## Corrección Literal de Textos del Usuario
Cuando el usuario proporcione un texto corregido (ej. "correccion: ..."), está **terminantemente prohibido inventar o agregar nuevas palabras o frases**. Únicamente se debe aplicar el texto exacto provisto por el usuario, corrigiendo exclusivamente la ortografía (tildes, caracteres tipográficos) y los espacios/formato LaTeX sin alterar la semántica ni la estructura elegida por el usuario.

## Redacción Limpia y Directa de Títulos y Encabezados
Aplica de forma estricta e inexcepcional a **TODO tipo de título, encabezado o rótulo principal**, incluyendo:
- Macros de jerarquía de texto (`\section`, `\subsection`, `\subsubsection`, `\paragraph`, `#`, `##`, etc.).
- Encabezados y títulos de recuadros en diagramas esquemáticos (bloques de `circuitikz`, `tikz`, diagramas de flujo).
- Títulos de gráficos generados por código (`plt.title`, `ax.set_title`, etc.).
- Rótulos de encabezado en figuras, tablas y cajas destacadas.

1. **Prohibición Absoluta de Paréntesis en Títulos:** Está terminantemente prohibido colocar paréntesis `(...)` en cualquier título o al final del mismo (ej. NO escribir `Título (Aclaración)`, `CANAL AD620 (TOPOLOGÍA ORIGINAL)`, `ETAPA DE ALIMENTACIÓN (±9V)`). El título debe contener exclusivamente el nombre directo y conciso del elemento o bloque. Si se requiere separar conceptos o clarificar, emplear dos puntos (`:`) o guión (`-`) sin paréntesis.
2. **Prohibición de Palabras de Relleno:** No utilizar adjetivos o términos de relleno artificiales (ej. NO agregar "Fisiológico", "Avanzado", "Estratégico", etc., salvo que el usuario lo solicite explícitamente). Los títulos deben ser sobrios, profesionales y directos.

## Lenguaje Humano, Natural y Prohibición de Tono Académico Rebuscado o de IA
Está **terminantemente prohibido** redactar textos, epígrafes, explicaciones o notas con tono de paper académico pomposo, tecnicismos inflados o estilo artificial de Inteligencia Artificial (ej. NO usar "deformaciones afines anisotrópicas", "isometría rígida", "cuello de botella latente", "canibalización de fonemas", "variedad bioeléctrica", "atenuación de artefactos gravitatorios").

1. **Estilo Real del Usuario (Cuaderno de Laboratorio Simple y Directo):**
   - Escribir en lenguaje humano, accesible y cotidiano, exactamente como un estudiante universitario explica sus pruebas a un compañero o en sus notas de laboratorio ("cortito y al pie").
   - Usar explicaciones prácticas y visuales:
     - En vez de *"deformaciones afines anisotrópicas por impedancia"*: decir *"el electrodo se movió un milímetro o cambió el contacto con la piel, así que los valores se estiran o cambian de escala entre un día y otro"*.
     - En vez de *"canibalización dimensional de clases fonatorias"*: decir *"al forzar la separación de una vocal terminamos mezclando y arruinando las otras"*.
     - En vez de *"atenuación de artefactos gravitatorios en el suelo de la boca"*: decir *"el electrodo abajo de la mandíbula se despega fácil por la gravedad y el sudor, en cambio en la mejilla (masetero) queda firme"*.
   - Las conclusiones deben ser directas: *"con esto la O y la U siguen saliendo mezcladas"*, *"en 3D clasifica casi todo perfecto menos tal caso"*, *"el electrodo chico ayudó a que no se toquen los cables"*.

2. **Cero Palabras de Relleno e Inteligencia Artificial:**
   - Evitar conectores típicos de IA como "es fundamental destacar que", "resulta imperativo señalar", "desempeña un rol crucial". Ir directo al hecho concreto sin introducciones decorativas.
   - Prohibido emplear palabras forzadas como 'retrabajo', 'pipeline', 'trade-off', 'framework', etc.

3. **Lenguaje Sobrio para Elementos Físicos:**
   - No usar adjetivos inflados ("amplificador bioeléctrico diferencial", "plataforma biopotencial").
   - Utilizar nombres simples y directos: "amplificador", "baterías", "cables", "electrodos", "placa", "medición".

## Obligatoriedad de Normalización por el Supremo Tricanal por Pulso Individual
Está **terminantemente prohibido** normalizar cada canal muscular de forma independiente dividiendo por su propio máximo ($x_c / \max(x_c)$) y está **terminantemente prohibido normalizar por el máximo global de la sesión** (`np.max(sesion)`).

- **Regla Obligatoria:** Todas las ventanas y tensores musculares deben normalizarse estrictamente por el **Supremo Tricanal del Pulso Individual ($M_{\text{supremo, pulso}}$)**:
  $$M_{\text{supremo, pulso}} = \max_{c \in \{0, 1, 2\}} \left( \max_{t \in \text{ventana}} |x_c(t)| \right)$$
  $$\tilde{x}_c(t) = \frac{|x_c(t)|}{M_{\text{supremo, pulso}}}$$

- **Justificación Fisiológica:**
  1. **Preservación de la Sinergia Intermuscular:** Mantiene intacto el balance bioeléctrico entre canales (el músculo motor primario alcanza $1.0$, y los secundarios quedan en sus ratios fisiológicos proporcionales).
  2. **Invarianza Frente a la Deriva Inter-Toma:** Normalizar por pulso neutraliza las variaciones de impedancia de contacto piel-electrodo y de intensidad fonatoria entre tomas distintas a lo largo de la sesión, impidiendo que contracciones válidas queden aplastadas por un único pulso atípico muy fuerte.

## Prohibición de Ejecutar Código sin Autorización Explícita
Está **terminantemente prohibido** ejecutar scripts, comandos en consola (`run_command`), tareas en segundo plano o pruebas automatizadas sin el consentimiento y autorización explícita previa del usuario.
- Ante cualquier necesidad de prueba, verificación o benchmarking, el asistente debe:
  1. Proponer la prueba explicando qué medirá y qué comando o script se utilizaría.
  2. Esperar la confirmación y permiso explícito del usuario antes de invocar cualquier herramienta de ejecución de código.

## Memoria del Proyecto y Estado Actual de la Investigación (Habla Submáximal)
Para asegurar la continuidad del trabajo entre sesiones, consultar el documento detallado en `.agents/MEMORIA_ESTADO_ACTUAL.md`.

**Puntos Clave Consolidados:**
1. **Invarianza de la Cruz Latente:** El autoencoder 1D no supervisado proyecta la actividad muscular en una geometría cruciforme universal (apertura mandibular en $+Y$, retracción comisural a la derecha y labial abajo).
2. **Anclaje Canónico Obligatorio:** En todo análisis y gráfico comparativo inter-sujeto, **el rayo de la vocal /a/ debe estar anclado estrictamente en el semieje vertical positivo ($+Y$, $\theta = 90^\circ$, 12 en punto)**, y las vocales de sonrisa (/i/, /e/) orientadas al semiplano derecho ($+X > 0$).
3. **Diagnóstico Anatómico de Tomas Previas (Candela 28/08):** La falta de separación de la vocal /a/ se debió al registro del músculo milohioideo en vez del vientre anterior del digástrico, y modíolo desplazado hacia abajo. Con electrodos bien posicionados (Candela 01/09), la cruz y la separación son nítidas.
4. **Próximo Paso Inmediato:** Extender la representación latente no supervisada conectando el encoder a una cabeza clasificadora supervisada ligera (Linear Probe / MLP) para resolver la discriminación fina entre /e/ frente a /i/, y /o/ frente a /u/.

## Monitoreo Obligatorio de Progreso, Tiempos y Estimaciones en Scripts y Tareas
Está **terminantemente prohibido** ejecutar o crear scripts que corran bucles largos (carga de tomas, épocas de entrenamiento, transformaciones pesadas) sin emitir información visual de progreso en tiempo real.

1. **Monitoreo en Carga y Procesamiento de Datos:**
   Todo bucle que itere sobre tomas, archivos o ventanas debe imprimir el porcentaje y avance exacto en consola:
   ```python
   print(f"[Carga] Toma {i+1}/{len(tomas)} ({((i+1)/len(tomas))*100:.1f}%) - {med_name}")
   ```

2. **Monitoreo en Épocas de Entrenamiento:**
   Todo bucle de entrenamiento (PyTorch, scikit-learn o similar) que supere 10 épocas debe imprimir periódicamente el avance (al menos cada 10, 20 o 25 épocas según la duración estimada), mostrando:
   * Época actual y total con porcentaje: `Época [50/250] (20.0%)`.
   * Pérdida instantánea (`MSE: 0.35120`).
   * Tiempo por época y estimación restante (*ETA*).
   * Al menos un `print` final al culminar el entrenamiento.

3. **Estimación Previa de Tiempo al Proponer Pruebas:**
   Al proponer cualquier prueba o script al usuario antes de pedir autorización, el asistente debe incluir obligatoriamente una estimación realista del tiempo que tomará la ejecución (ej. *"Tiempo estimado: ~2 minutos en CPU"* o *"Tiempo estimado: ~15 minutos por la profundidad de la red"*).

4. **Uso de Temporizadores en Tareas Asíncronas:**
   Cuando una tarea en segundo plano supere los 2 minutos estimados, el asistente debe establecer un temporizador o mecanismo de seguimiento (`schedule` / aviso proactivo) para no dejar al usuario en la incertidumbre.

## Regla Estricta de Memoria Persistente Inter-Turno (Lectura y Escritura Obligatoria)
Para garantizar la continuidad total y evitar cualquier pérdida de contexto o directivas:

1. **Lectura Obligatoria al Iniciar Cada Chat:**
   Al comenzar una conversación o procesar un requerimiento del usuario, el asistente debe inspeccionar obligatoriamente el archivo `.agents/MEMORIA_ESTADO_ACTUAL.md` antes de proponer soluciones o ejecutar tareas, repasando los últimos hitos, directivas inmutables y estado del código.

2. **Escritura Obligatoria en Cada Intervención:**
   Al concluir cada intervención, cambio de código, hito o análisis, el asistente debe actualizar de forma obligatoria e inmediata `.agents/MEMORIA_ESTADO_ACTUAL.md` registrando:
   - Lo último implementado o modificado.
   - Parámetros, fórmulas y decisiones acordadas.
   - Estado actual del sistema y próximos pasos inmediatos.

3. **Prohibición de Reiterar Enfoques Descartados:**
   Cualquier técnica descartada explícitamente queda catalogada como directiva inmutable y no podrá volverse a sugerir.

## Rigor en Pruebas Comparativas e Invarianza de la Línea Base
Al evaluar una nueva técnica, capa o variante arquitectónica (ej. BatchNorm, Dropout, nuevos tamaños de núcleo) frente a una línea base ya consolidada (ej. Autoencoder Conv1D al 56.1%), está **terminantemente prohibido** alterar de forma colateral cualquier otro parámetro de la línea base:

1. **Aislamiento Estricto de la Variable en Ensayo (Ceteris Paribus):**
   La única diferencia entre el modelo de control y el modelo experimental debe ser exclusivamente la técnica que se busca evaluar. Todo lo demás debe permanecer idéntico.

2. **Preservación Obligatoria de Hiperparámetros de Referencia:**
   - **Acondicionamiento y Normalización:** Mantener idéntica la escala de entrada (Supremo Tricanal computado por pulso individual $x_c(t) / M_{\text{supremo, pulso}}$).
   - **Régimen de Optimización:** Mantener el mismo tamaño de lote (*Full-Batch* si la referencia fue entrenada así), la misma tasa de aprendizaje ($\eta = 0.008$), optimizador y número de épocas ($250$).
   - **Arquitectura Base:** Mantener idénticas las capas del codificador y decodificador (salida lineal sin `ReLU` final de bloqueo).

3. **Verificación Previa del Modelo de Control:**
   Antes de contrastar o extraer conclusiones sobre la variante nueva, el asistente debe verificar que el modelo de control reproduzca con exactitud la métrica histórica de referencia ($56.1\%$). Si el control difiere de la referencia, el experimento se considera inválido y debe corregirse antes de continuar.

## Obligatoriedad de Estimación de Ruido Dinámico Interpulso y Prohibición de Filtros Ciegos de SNR
Está **terminantemente prohibido** estimar el piso de ruido basal utilizando únicamente los primeros segundos estáticos del archivo (ej. `noise_sec = 5.0` o `sig[:5*fs]`) y está **terminantemente prohibido descartar pulsos mediante umbrales rígidos de SNR** (`SNR > 0.5` o similares).

1. **Estimación Dinámica Interpulso Obligatoria:**
   El nivel de ruido basal de cada canal muscular debe calcularse estrictamente en el entorno local de cada contracción individual, evaluando los segmentos de reposo inmediatamente previo y posterior al pulso:
   $$\text{Ruido}_{\text{prom}, c} = \frac{\text{Ruido}_{\text{pre}, c} + \text{Ruido}_{\text{post}, c}}{2}$$
   donde cada intervalo se depura de espigas residuales mediante el rango intercuartil ($[\text{IQR} = Q_3 - Q_1]$, descartando valores superiores a $Q_3 + 1.5 \times \text{IQR}$).

2. **Prohibición de Filtro de Descarte por SNR:**
   No se deben eliminar pulsos mediante condiciones heurísticas de SNR. Las variaciones de energía entre vocales son fisiológicas (ej. fonemas de mandíbula cerrada tienen menor amplitud que los de apertura).

3. **Purga Única por Isolation Forest:**
   La eliminación de artefactos macroscópicos de electrodo se realiza exclusivamente mediante *Isolation Forest* con una contaminación máxima del 10% sobre la envolvente macroscópica suave, asegurando la conservación de los más de 500 pulsos balanceados de la sesión.

## Nomenclatura Anatómica Oficial de Canales sEMG
Está **terminantemente prohibido** inventar o asumir nombres musculares sin verificar los archivos `metadata.json` oficiales de la sesión.
- Para las sesiones de Lucas (`2026-07-10`), los canales registrados son estrictamente:
  - **Canal 0:** Milohioideo (`Mylohyoid`)
  - **Canal 1:** Depresor (`Depresor Anguli Oris`)
  - **Canal 2:** Orbicular (`Orbicularis Oris`)
  - **Canal 3:** Micrófono

## Prohibición de Reportes Sensacionalistas y Prioridad de Clasificación Equilibrada
Está **terminantemente prohibido** calificar como éxito, récord o avance un incremento aislado en la métrica de una sola clase o vocal si este ocurre a costa de degradar sustancialmente a las demás (ej. celebrar un aumento en `/e/` cuando `/a/` se desploma).

1. **Prioridad del Balance Multiclase:**
   En todo reporte, tabla o diagnóstico de modelos, el asistente debe evaluar y priorizar la clasificación armónica y equilibrada entre las 5 vocales (`/a/`, `/e/`, `/i/`, `/o/`, `/u/`). 
   Un modelo solo se considerará superador si eleva el rendimiento conjunto o mantiene un piso homogéneo sin canibalizar unas clases en favor de otras.

## Prohibición Absoluta de Supervisión en el Autoencoder (Zero-Labels Estricto)
Está **terminantemente prohibido** utilizar etiquetas de clases, vocales o fonemas durante el entrenamiento (cálculo de pérdida y propagación de gradientes `backward`) del autoencoder.
El descubrimiento de la variedad bioeléctrica latente debe ser **100% NO SUPERVISADO**.

1. **Cero Etiquetas en el Grafo de Gradientes:**
   La función de pérdida del modelo solo puede depender de la señal bioeléctrica observada $x$ y de su reconstrucción $\hat{x}$ (ej. $\text{MSE}(x, \hat{x})$, preservación de envolvente macroscópica $\text{MSE}(\text{Env}(x), \text{Env}(\hat{x}))$, regularizaciones de peso o divergencias de agrupamiento ciego auto-organizado sin etiquetas como DEC).

2. **Rol Exclusivo Post-Entrenamiento de las Etiquetas:**
   Las etiquetas de fonemas se reservan única y exclusivamente para la etapa final de evaluación diagnóstica externa (evaluar con GMM o K-Means qué porcentaje de los clusters naturales descubiertos por la red corresponden a cada vocal mediante asignación lineal húngara).

3. **Invalidez de Modelos Supervisados:**
   Cualquier resultado obtenido mediante guía de etiquetas o pérdidas de clasificación cruzada (`CrossEntropyLoss`, `CenterLoss` con etiquetas) queda declarado científicamente nulo e inválido para el objetivo de esta investigación.

## Organización Estricta del Código y Prohibición de Proliferación de Scripts
Está **terminantemente prohibido** generar scripts sueltos, temporales o descartables repartidos de forma desordenada por el repositorio (ej. `test_....py`, `temp_....py`, etc.).

1. **Desarrollo Modular y Unificado:**
   Toda funcionalidad nueva (extracción de características, arquitecturas de Deep Learning, evaluación o visualización) debe integrarse de forma estructurada, limpia y modular dentro de los paquetes oficiales del proyecto (`EMG_desarrollo/deep_learning/`, `gui_app/`, etc.), evitando la fragmentación del código.

2. **Centralización Estricta de Gráficos e Imágenes:**
   Está prohibido guardar figuras o imágenes en carpetas arbitrarias o temporales. Todas las salidas visuales deben dirigirse obligatoriamente a los directorios centralizados oficiales de resultados (`EMG_desarrollo/resultados/`, `resultados_autoencoder/`, etc.) con nombres descriptivos y organizados.

3. **Arquitectura Orientada a Interfaces Funcionales:**
   Cualquier batería de modelos o métodos de procesamiento debe converger en una interfaz unificada que permita seleccionar sesiones, conmutar modos de entrada (cruda, envolvente, espectrograma) y visualizar métricas consolidadas sin requerir la creación de nuevos scripts para cada ensayo.

4. **Limpieza Continua de Archivos de Prueba:**
   Tras validar una hipótesis experimental, el asistente debe consolidar el código definitivo en la herramienta principal y eliminar o archivar limpiamente cualquier script auxiliar intermedio para mantener el repositorio prístino.

## Auditoría Obligatoria de Metadatos y Coherencia Anatómica Inter-Día
Al cargar o procesar grabaciones de múltiples sesiones o diferentes días:

1. **Verificación Estricta de Músculos por Canal:**
   El sistema debe inspeccionar obligatoriamente el archivo `metadata.json` de cada sesión seleccionada y verificar la correspondencia de los músculos asignados a cada canal (`canal_0`, `canal_1`, `canal_2`, `canal_3`). Si se detectan inconsistencias anatómicas entre sesiones (ej. vientre anterior del digástrico vs milohioideo, o modíolo vs orbicular), debe emitirse una advertencia visual prominente en la interfaz alertando que se están combinando mediciones con diferente colocación de electrodos.

2. **Verificación de Metrónomo y Frecuencia de Muestreo:**
   Debe auditarse la concordancia de BPM (`bpm`) y tasa de muestreo (`fs`) entre todas las sesiones a procesar conjuntamente para garantizar que la segmentación de ciclos sea físicamente homogénea.

## Organización Estricta de Datos, Modelos y Carpetas de Salida
Está **terminantemente prohibido** dispersar datasets (`.npz`), checkpoints de pesos (`.pth`), archivos de métricas (`.csv`/`.json`) o gráficos en la raíz del repositorio o en directorios temporales (`/tmp/`).

1. **Jerarquía Oficial de Resultados:**
   Toda ejecución del motor de autoencoder o procesamiento de características debe estructurarse en subdirectorios limpios bajo `EMG_desarrollo/resultados/resultados_autoencoder/`:
   ```
   EMG_desarrollo/resultados/resultados_autoencoder/
   ├── cache_datasets/
   │   └── dataset_<modalidad>_<sujeto_o_sesion>.npz
   ├── modelos_entrenados/
   │   └── autoencoder_<modalidad>_<dim>d.pth
   └── figuras_evaluacion/
       └── informe_autoencoder_<modalidad>_<dim>d.png
   ```

2. **Persistencia Ordenada de Datasets:**
   Los archivos `.npz` deben incluir metadatos de proveniencia completos (`X_env`, `X_cruda`, `X_spec`, `Y`, `Tomas`, `Fechas`, `Musculos_Canales`, `bpm`, `fs`).

## Metodología Oficial de Extracción y Alineación Fisiológica (Estándar Trevisan / PCA-UMAP)
Toda extracción de pulsos y entrenamiento debe seguir estrictamente la cadena consolidada de `generador_pca_umap` y `analisis_trevisan`:

1. **Lectura Obligatoria de Metadatos por Toma:**
   Inspeccionar el archivo `metadata.json` de cada toma de forma independiente para obtener sus parámetros físicos reales (`bpm`, `noise_seconds`, `pulse_count`, `sample_rate`, `muscles`). Prohibido asumir valores globales fijos para toda la sesión.

2. **Corte Guiado por Envolvente para la Señal Cruda:**
   La señal cruda rectificada NO debe segmentarse con detectores de picos independientes en alta frecuencia. Debe cortarse utilizando exactamente los mismos límites temporales $[n_{\text{inicio}}, n_{\text{fin}}] = [p_{\text{idx}} - W_{\text{pre}}, \; p_{\text{idx}} + W_{\text{post}}]$ determinados a partir de la envolvente suavizada o el micrófono del canal maestro.

3. **Calibración Intersesión e Intercanal Acotada (Estándar PCA-UMAP):**
   La compensación de sensibilidad por canal debe calcularse por sesión mediante el percentil P95 de cada canal ($V_c = \text{percentile}_{95}(\max(s_c))$) acotando el factor de ganancia a un máximo de $5\times$ ($C_c = 1.0 / \max(V_c / \max(V), 0.20)$). Está terminantemente prohibido asumir qué músculo reside en qué canal sin auditar el archivo `metadata.json`.

4. **Trazabilidad Visual y Nombres Musculares en Informes:**
   - En los subplots de reconstrucción, cada curva debe indicar explícitamente en la leyenda el nombre anatómico del músculo correspondiente (`Ch0: Anterior Belly`, `Ch1: Orbicularis Oris`, `Ch2: Risorio`, etc.).

## Obligatoriedad de Semilla Fija Universal (Reproducibilidad Estricta)
Está **terminantemente prohibido** dejar librada la semilla pseudoaleatoria al azar en cualquier etapa de procesamiento, filtrado, extracción de datos, partición de entrenamiento o inicialización de redes neuronales (PyTorch, NumPy, scikit-learn, Python random):

1. **Fijación Previa Obligatoria:**
   Toda función de extracción, entrenamiento o evaluación latente debe invocar de forma obligatoria la fijación determinista de semilla (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` y `torch.cuda.manual_seed_all(42)` si aplica).
2. **Invarianza Matemática de Resultados:**
   Dos corridas idénticas sobre el mismo dataset deben generar exactamente las mismas coordenadas latentes, métricas de agrupamiento GMM y asignación bipartita de clusters.

## Extracción Estricta de Fecha Real desde metadata.json (measurement_date)
Está **terminantemente prohibido** asumir que la fecha de grabación de una toma corresponde al nombre de la carpeta contenedora o del directorio superior (ej. asumir ciegamente que toda toma dentro de `2026-07-10/` se grabó ese día):

1. **Lectura Obligatoria del Campo measurement_date:**
   Cada toma individual debe inspeccionar obligatoriamente el archivo `metadata.json` ubicado dentro de su subcarpeta `canal_0/` y extraer la fecha exacta del atributo `"measurement_date"` (ej. `"measurement_date": "2026-06-01T17:11:48.624352"` -> `"2026-06-01"`).
2. **Diferenciación y Trazabilidad Inter-Día:**
   En todos los gráficos latentes (2D y 3D) y reportes consolidados, los puntos de diferentes días reales deben diferenciarse visualmente mediante marcadores geométricos distintos y una leyenda con las fechas exactas encontradas.

## Prohibición de Búsqueda Local de Código sin Permiso
Está **terminantemente prohibido** buscar, leer o inspeccionar el código local (archivos `.py`, etc.) del repositorio cuando el usuario hace una pregunta o pide algo nuevo, a menos que el usuario otorgue permiso explícito previo. Esto es para evitar gastos innecesarios de tokens. El asistente debe ir directo al grano, priorizar el conocimiento general o, como máximo, buscar en internet si el usuario lo solicita.

## Registro Obligatorio de Parámetros de Extracción y Modelado (Trazabilidad Científica)
Al reportar o documentar resultados cuantitativos (ej. exactitud, silueta, separabilidad de vocales):
1. **Transparencia Completa de Configuración:** Debe explicarse explícitamente la combinación completa de parámetros utilizada para la obtención de dicho resultado (ej. tipo de envolvente, longitud de ventana temporal `target_len`, filtros pasa banda, normalización, algoritmo de clustering utilizado, peso de regularización $\lambda_{\text{orto}}$, etc.).
2. **Prohibición de Suposiciones Implícitas:** Está terminantemente prohibido asumir o dar por sentado que una métrica se obtuvo bajo los valores por defecto sin declararlo de forma clara.

## Estilo de Comunicación: Cortito y al Pie
Está **terminantemente prohibido** incluir introducciones largas, divagaciones, discursos o explicaciones teóricas no solicitadas ("chamuyo").
- Las respuestas deben ser breves, telegráficas, directas y enfocadas exclusivamente en la acción o resultado concreto solicitado por el usuario.

## Invariante de Hardware del Front-End sEMG (Ganancia Fija Soldada)
La ganancia del amplificador de instrumentación AD620 en la placa de adquisición física está **fijada por hardware mediante una resistencia soldada** $R_G = 100\,\Omega$ ($G = 1 + \frac{49.4\,\text{k}\Omega}{100\,\Omega} \approx 495\,\text{V/V}$).
- Está **terminantemente prohibido** describir la ganancia como "programable" o configurable.

## Formato de Encabezados LaTeX (fancyhdr)
En documentos técnicos con `fancyhdr`, configurar únicamente `\fancyhead[L]` y `\fancyhead[R]` sin texto largo en `\fancyhead[C]`, para evitar superposiciones tipográficas entre el nombre de la institución y el título del documento.

## Código de Colores Oficial Universal para Vocales
En todos los gráficos, diagramas, interfaces y visualizaciones del proyecto, el código de colores estricto y universal para las 5 vocales es:
- **/a/**: Rojo (`#E63946`)
- **/e/**: Azul (`#1F77B4`)
- **/i/**: Verde (`#2CA02C`)
- **/o/**: Morado (`#9D4EDD`)
- **/u/**: Amarillo (`#E7A61A`)
Está terminantemente prohibido alterar esta asignación cromática en cualquier módulo de visualización o reporte.
