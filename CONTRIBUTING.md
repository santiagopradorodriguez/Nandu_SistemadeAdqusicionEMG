# Guia de Contribucion al Sistema de Adquisicion EMG (Nandu LSD)

Gracias por tu interes en contribuir al proyecto. Nandu LSD es una plataforma cientifica abierta, desarrollada por y para la comunidad del Laboratorio de Sistemas Dinamicos (LSD - FCEyN, UBA).

El objetivo es democratizar el acceso a herramientas de electromiografia de alta calidad, estandarizar la captura de senales mioelectricas y acelerar la construccion de datasets para Machine Learning y Deep Learning. Cualquier aporte —desde correccion de documentacion y optimizaciones de procesamiento digital de senales (DSP) hasta nuevas arquitecturas neuronales en PyTorch o mejoras de la interfaz de usuario en PySide6— es bienvenido.

---

## 1. Por Donde Empezar

El proyecto se encuentra en su version **v6.0**, estructurado de forma modular en torno al lanzador principal `EMG_desarrollo/gui_app/main_app.py`. A continuacion se detallan las areas principales de colaboracion:

### A. Interfaz de Usuario y Experiencia (PySide6 / PyQtGraph)
- **Visualizadores Interactivos:** Optimizar el rendimiento de renderizado en `csv_viewer_widget.py`, `calibrated_viewer_widget.py` y `electrode_viewer_widget.py` utilizando tecnicas de downsampling y actualizacion por regiones.
- **Responsividad:** Asegurar que los paneles se adapten correctamente a resoluciones variadas (desde pantallas de laboratorio de 1366x768 hasta monitores 4K).
- **Widgets de Analisis:** Desarrollar nuevos subpaneles para visualizacion de dinamicas motoras y representaciones graficas avanzadas.

### B. Optimizacion y Rendimiento
- **Concurrencia y Estabilidad de Hilos:** Fortalecer la comunicacion asincrona basada en `QThread` y `QTimer` para prevenir congelamientos de la interfaz grafica durante adquisiciones extensas o entrenamientos intensivos de Deep Learning.
- **Eficiencia de Memoria:** Garantizar que los buffers circulares de senal y la lectura de archivos CSV masivos empleen estructuras vectorizadas eficientes con NumPy y SciPy sin fugas de memoria (*memory leaks*).

### C. Procesamiento Digital de Senales (DSP)
- **Filtros de Fase Cero e IIR:** Validar el comportamiento de las etapas de filtrado (Notch 50 Hz, Butterworth pasabanda 20-500 Hz) y la continuidad del estado `zi` en capturas en tiempo real.
- **Algoritmos de Calidad y Fatiga:** Desarrollar y validar metricas cuantitativas de estimacion de la Relacion Senal-Ruido (SNR) y monitoreo de la relajacion inter-pulso.
- **Alineacion Master-Slave:** Refinar la deteccion de envolventes y metodos de correlacion cruzada para preservar los desfases fisiologicos inter-musculares.

### D. Machine Learning y Deep Learning (PyTorch)
- **Arquitecturas Neuronales:** Implementar y evaluar variantes de Autoencoders Convolucionales 1D, modelos recurrentes (LSTM/GRU) o Transformers para series temporales de sEMG.
- **Pipelines de Datos:** Mantener la compatibilidad del generador `dl_data_pipeline.py` y la clase `EMGDataset`, asegurando que la normalizacion y el remuestreo (500 dimensiones) operen sin sesgos ni fuga de datos (*data leakage*).
- **Clasificadores:** Optimizar modelos de XGBoost, tecnicas de binarizacion de unidades motoras (Metodo Trevisan) y decodificadores continuos.

### E. Documentacion y Calidad
- **Docstrings Estandarizados:** Documentar funciones y clases siguiendo los formatos de estilo NumPy o Google Docstrings.
- **Politica Estricta de No-Emojis:** Garantizar que todo el codigo fuente, comentarios, cadenas y archivos de documentacion permanezcan 100% libres de emojis.

---

## 2. Reglas de Oro del Repositorio

Para mantener la integridad y reproducibilidad del proyecto, todas las contribuciones deben respetar las siguientes reglas obligatorias:

1. **Estructura Oficial de la Base de Datos (AGENTS.md):**
   Toda grabacion y lectura de datos debe respetar la jerarquia:
   ```
   base_de_datos_electrodos/<Fecha>/<Sesión>/canal_0..3/
   ```
   donde el archivo maestro `metadata.json` y la grabacion primaria residen obligatoriamente en `canal_0`. Ningun algoritmo debe asumir archivos planos sueltos en la raiz de la sesion.

2. **Politica de Cero Emojis:**
   No se permite el uso de caracteres emoji en codigo Python, comentarios, cadenas de texto, mensajes de log, nombres de variables ni archivos de documentacion Markdown.

3. **Regla de Alcance (Scoping) en Componentes Qt:**
   Todo objeto visual, color o estado que deba persistir o ser accedido por metodos secundarios debe asociarse explicitamente a la instancia (`self.atributo`). Evitar variables locales en `__init__` que causen caidas por `NameError` en tiempo de ejecucion.

4. **Preservacion de Sinergia Fisiologica:**
   Al sincronizar multiples canales musculares, la alineacion temporal debe realizarse exclusivamente mediante el esquema Master-Slave (usando el canal de referencia para fijar la ventana temporal de todos los canales), evitando recortar canales de forma independiente.

---

## 3. Configuracion del Entorno de Desarrollo

Sigue estos pasos para preparar tu entorno de trabajo:

```bash
# 1. Clonar el repositorio
git clone https://github.com/santiagopradorodriguez/Nandu_SistemadeAdqusicionEMG.git
cd Nandu_SistemadeAdqusicionEMG

# 2. Crear y activar el entorno virtual
python -m venv venv

# En Windows:
.\venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate

# 3. Instalar dependencias del proyecto
# En Windows:
pip install -r requirements.txt
# En Linux:
pip install -r requirements_linux.txt

# 4. Modo Simulador (Sin Hardware NI)
# Puedes ejecutar el sistema completo activando la opcion "Usar Microfono" en la aplicacion
# para probar todo el pipeline con cualquier tarjeta de sonido convencional.
```

---

## 4. Flujo de Trabajo para Contribuciones (Pull Requests)

1. **Crear una rama (branch) tematica:**
   ```bash
   git checkout -b feature/nueva-arquitectura-autoencoder
   # o
   git checkout -b fix/alineacion-master-slave
   ```
2. **Implementar los cambios respetando las normas de estilo.**
3. **Ejecutar pruebas de regresion y verificacion:**
   - Verificar que el lanzador `python EMG_desarrollo/gui_app/main_app.py` inicie sin advertencias ni bloqueos de interfaz.
   - Ejecutar los scripts de prueba en `EMG_desarrollo/tests/`.
   - Verificar la ausencia de emojis en los archivos modificados.
4. **Crear commits semanticos:**
   ```bash
   git commit -m "feat(dsp): optimizar calculo de envolvente RMS con convolucion vectorizada"
   ```
5. **Enviar la rama y abrir un Pull Request:**
   ```bash
   git push origin feature/nueva-arquitectura-autoencoder
   ```
   Describe en el Pull Request la motivacion del cambio, los archivos modificados y el metodo empleado para verificar su correcto funcionamiento.

---

## 5. Estilo de Codigo y Convenciones

- **Estandar de Codigo:** Se sigue la guia de estilo **PEP 8** para codigo Python.
- **Idioma:** Se prioriza el idioma **espanol** para comentarios tecnicos, explicaciones cientificas y documentacion de funciones, facilitando la comprension de investigadores y estudiantes de habla hispana.
- **Tipado e Inferencia:** Se recomienda el uso de type hints en firmas de funciones criticas para mejorar la legibilidad y mantenimiento a largo plazo.

---

## 6. Lista de Tareas Pendientes y Roadmap (v6.0+)

Areas actualmente abiertas para colaboracion:
- [ ] **Visualizacion Anatomica de Electrodos:** Desarrollar un componente interactivo que despliegue esquemas graficos (`configuracion.jpg`) documentando la colocacion fisica de los electrodos en los grupos musculares.
- [ ] **Decodificador en Tiempo Real:** Integrar el modelo de autoencoder o clasificador continuo directamente con el buffer de entrada del DAQ para decodificacion fonatoria en vivo.
- [ ] **Boton de Pausa Segura:** Implementar control de pausa y reanudacion sincronizada en la maquina de estados de AutoForge.
- [ ] **Suite de Pruebas Unitarias:** Incrementar la cobertura automatizada de pruebas para modulos DSP, loaders de datos y vistas de visualizacion.

Agradecemos profundamente tu colaboracion para continuar fortaleciendo esta herramienta cientifica.
