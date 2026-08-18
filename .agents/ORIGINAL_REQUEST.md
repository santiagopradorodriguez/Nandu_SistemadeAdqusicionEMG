# Original User Request

## 2026-08-17T17:11:57Z

# Teamwork Project Prompt — Draft

> Status: Ready for launch — awaiting user approval
> Goal: Craft prompt → get user approval → delegate to teamwork_preview

Curación, estabilización multiplataforma, documentación y pulido de visualizaciones del software Ñandú EMG en su propia UI, seguido de la redacción de un reporte académico no técnico en LaTeX.

Working directory: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo
Integrity mode: demo

## Requirements

### R1. Curaduría y Corrección Profunda (Arquitectura)
Revisar la arquitectura de la interfaz PySide6 y corregir bugs de inicialización o pasaje de parámetros. **No se debe alterar** la lógica matemática ni los pasos internos de los scripts de PCA, UMAP Supervisado ni Autoencoders. Leer el código existente para entender el comportamiento antes de modificar.

### R2. Empaquetado y Builds Multiplataforma
Mejorar y arreglar los scripts de configuración de empaquetado (ej. PyInstaller) que ya existen en el proyecto para asegurar que se puedan generar compilados funcionales tanto en Windows como en Linux.

### R3. Pulido de Visualizaciones en la Interfaz
En PCA 3D, mejorar el gráfico para que muestre proyecciones legibles. Asegurar que la "Galería de Resultados" dentro de la pestaña de Machine Learning pueda cargar y mostrar de manera amena y legible estos gráficos junto con otros datos vitales (como las matrices de confusión o los centroides).

### R4. Documentación del Software y Repositorio
Actualizar las instrucciones de uso generales en la interfaz, las "novedades", el `README.md` y el `CONTRIBUTING.md`. Generar un commit ordenado al finalizar.

### R5. Redacción del Reporte Académico (`software.tex`)
Escribir la sección `software.tex` (para un reporte de laboratorio). El texto no debe ser técnico (ignorar el "motor" y detalles de código), sino enfocarse en lo útil e importante que es el software. El estilo de redacción debe basarse en los informes de laboratorio previos ubicados en el vault del usuario (`/home/santiago/Documentos/santiago vault/`).

## Acceptance Criteria

### Funcionalidad y UI
- [ ] La UI principal inicia sin errores y los métodos de recolección de parámetros no generan excepciones.
- [ ] La Galería de Resultados en la UI carga exitosamente imágenes y tablas de métricas asociadas sin recortar el texto.
- [ ] El gráfico PCA 3D incluye proyecciones 2D en sus planos proyectados.

### Builds
- [ ] Los scripts de empaquetado para Linux y Windows completan su ejecución sin lanzar errores de dependencias faltantes.

### Documentación y Reporte
- [ ] `README.md` y `CONTRIBUTING.md` contienen las instrucciones actualizadas sin errores de sintaxis Markdown.
- [ ] El archivo `software.tex` compila correctamente (sin sintaxis inválida de LaTeX) y omite deliberadamente la explicación de arquitecturas de software.
- [ ] Se generó un commit en el repositorio con todos estos cambios aplicados.
