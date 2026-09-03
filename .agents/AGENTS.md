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
Al redactar títulos de secciones, subsecciones o párrafos (`\section`, `\subsection`, `\subsubsection`, `\paragraph`, etc.):
1. **Prohibición de Paréntesis en Títulos:** Está terminantemente prohibido colocar paréntesis en los títulos o al final de ellos (ej. NO escribir `Título (Aclaración)`, `Paso 2: Remodelado (reshape)`, `Paso 3: Extracción (__getitem__)`). El título debe contener únicamente el nombre conciso del tema.
2. **Prohibición de Palabras de Relleno:** No utilizar adjetivos o términos de relleno artificiales (ej. NO agregar "Fisiológico", "Avanzado", "Estratégico", etc., salvo que el usuario lo solicite explícitamente). Los títulos deben ser sobrios, profesionales y directos.

## Prohibición del Término "Pipeline" y Anglicismos Redundantes
Está terminantemente prohibido utilizar el término **"pipeline"** o **"pipelines"** tanto en el cuerpo del texto como en pies de figuras, tablas o títulos.
- En su lugar, utilizar términos precisos y directos en español acordes al contexto:
  - *Procesamiento* / *Cadena de procesamiento*
  - *Acondicionamiento de señales*
  - *Flujo de datos* / *Flujo de trabajo*
  - *Secuencia de etapas* / *Esquema metodológico*

## Lenguaje Sobrio y Prohibición de Adjetivos Redundantes
Está prohibido utilizar adjetivos sobrecargados o términos técnicos redundantes para referirse a elementos estándar del sistema:
- **Evitar:** "amplificador bioeléctrico diferencial", "sistema bioeléctrico", "plataforma biopotencial", "instrumentación de precisión", etc.
- **Utilizar:** Términos directos, sobrios y naturales en español: *"amplificador"*, *"baterías"*, *"cables"*, *"electrodos"*, *"adquisición"*, *"medición"*.

## Obligatoriedad de Normalización por el Supremo Global (Prohibición de Normalización Independiente)
Está **terminantemente prohibido** normalizar cada canal muscular de forma independiente dividiendo por su propio máximo ($x_c / \max(x_c)$) o mediante escalados independientes (ej. StandardScaling/MinMax por canal separado).

- **Regla Obligatoria:** Todas las ventanas y tensores musculares deben normalizarse exclusivamente por el **Supremo Global Tricanal ($M_{\text{supremo}}$)** del pulso o de la sesión:
  $$M_{\text{supremo}} = \max_{c \in \{0, 1, 2\}} \left( \max_t |x_c(t)| \right)$$
  $$\tilde{x}_c(t) = \frac{x_c(t)}{M_{\text{supremo}}}$$
- **Justificación Fisiológica:** La información fonética del habla depende de la energía relativa intermuscular (cuál músculo es el dominante y cuáles son secundarios). Normalizar independientemente eleva el ruido o contracciones accesorias de $8\,\mu\text{V}$ a $1.0$, falseando la biomecánica de la articulación.

## Prohibición de Ejecutar Código sin Autorización Explícita
Está **terminantemente prohibido** ejecutar scripts, comandos en consola (`run_command`), tareas en segundo plano o pruebas automatizadas sin el consentimiento y autorización explícita previa del usuario.
- Ante cualquier necesidad de prueba, verificación o benchmarking, el asistente debe:
  1. Proponer la prueba explicando qué medirá y qué comando o script se utilizaría.
  2. Esperar la confirmación y permiso explícito del usuario antes de invocar cualquier herramienta de ejecución de código.






