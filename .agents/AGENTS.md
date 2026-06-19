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
