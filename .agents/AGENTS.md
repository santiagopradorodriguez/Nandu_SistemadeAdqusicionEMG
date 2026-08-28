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
