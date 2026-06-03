# Contribuir al Sistema de Adquisición EMG

¡Gracias por tu interés en contribuir! Este proyecto es una herramienta científica abierta, desarrollada **por y para la comunidad**.

El objetivo es democratizar el acceso a herramientas de electromiografía de calidad, construyendo herramientas para acelerar la creación de Datasets para Machine Learning. Cualquier ayuda, desde corregir errores ortográficos hasta optimizar algoritmos de procesamiento de señales o mejorar la interfaz gráfica, es bienvenida.

## 📍 ¿Por dónde empezar?

Actualmente, el proyecto ha dado un gran salto hacia su **versión 4.0** (introduciendo `Nandu_AutoForge_DAQ.py` y la migración a PySide6). Tenemos identificadas varias áreas donde necesitamos ayuda:

### 🐛 Reporte de Bugs y Mejoras de UI (PySide6)
- **Nandu AutoForge:** Ayudar a refinar la máquina de estados del protocolo AutoForge. Proponer mejoras para la visualización del Peak-Hold y la integración del metrónomo visual.
- **Layout y Diseño:** Asegurar que la nueva interfaz `Nandu_AutoForge_DAQ.py` (con su estilo Cyberpunk) sea completamente *responsive* y se adapte sin problemas a resoluciones de pantallas pequeñas (laptops).
- **Herramientas Legadas:** Ayudar a migrar los scripts antiguos (como `editor_mediciones.py` o `visor_csv_interactivo.py`) desde PyQt5 al nuevo estándar de PySide6 para mantener consistencia.

### ⚡ Optimización y Rendimiento
- **Gestión de Hilos y Concurrencia:** El nuevo AutoForge depende de `subprocess` para el metrónomo y de `threading` para la adquisición de datos (Micrófono o placa NI). Ayúdanos a perfilar el código para garantizar que no haya *memory leaks* en sesiones de grabación intensivas.
- **Visor CSV Calibrado (`plotter_calibrado.py`):** Optimizar la lectura de datos (usando *chunking* con Pandas) y el renderizado rápido de filtros envolventes RMS sobre señales masivas.

### 🧪 Procesamiento de Señales 
- **Validación del Espectrograma:** En `Nandu_AutoForge_DAQ.py` calculamos una STFT rodante con la librería de SciPy. Necesitamos ayuda de expertos en DSP para validar que la escala de colores y el overlap de la ventana de Hamming sean científicamente óptimos para señales mioeléctricas.
- **Auto-Threshold Inteligente:** Mejorar los algoritmos de cálculo de SNR en tiempo real. Actualmente usamos la desviación estándar del ruido base, pero podríamos beneficiarnos de modelos predictivos o filtros adaptativos de ruido.

### 📚 Documentación
- **Comentarios en Código:** Agregar comentarios explicativos (Docstrings) dentro de las funciones críticas de la máquina de estados de AutoForge.
- **Entorno Virtual:** Ayudar a mantener actualizada la lista de `requirements.txt` y crear una guía de instalación específica para usuarios de Linux/Mac que quieran correr el "Modo Simulador".

### 💡 Lecciones de Desarrollo (Scoping en UI)
A la hora de desarrollar o arreglar bugs en las interfaces de usuario (PySide6 / Tkinter), recuerda esta regla de oro:
- **EVITA el uso de variables locales en `__init__` si van a ser leídas después.** Hemos tenido bugs fatales donde variables de color (como `bg_panel`) no eran accesibles fuera de `__init__`. **Siempre antepón `self.`** a cualquier configuración global de estilo o variable de estado de la ventana.

---

## 📋 Lista de Tareas Pendientes (TODO)

Actualmente, necesitamos colaboración urgente en las siguientes áreas:
- [ ] **Sistemas de Logs:** Reparar y estandarizar todos los `logging.info()` a lo largo del código. Actualmente, la observabilidad es irregular, y necesitamos un estándar de archivo rotativo o formato único.
- [ ] **Empaquetado EXE:** Depurar la compilación de PyInstaller para evitar pesos excesivos e inclusión de librerías innecesarias (ignorar carpetas de base de datos grandes en el `.spec`).

## 🛠️ Configuración del Entorno de Desarrollo

Para asegurar que todos trabajamos bajo las mismas condiciones, sigue estos pasos:

1.  **Fork del repositorio:** Crea tu propia copia del proyecto en GitHub.
2.  **Entorno Virtual:** Es altamente recomendable usar `venv` para no romper tu instalación local de Python.
    ```bash
    python -m venv venv
    # Activar en Windows:
    .\venv\Scripts\activate
    # Activar en Linux/Mac:
    source venv/bin/activate
    ```
3.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Drivers (Importante):** Si vas a trabajar con hardware físico, necesitas instalar el driver **NI-DAQmx**. Sin embargo, la nueva arquitectura permite usar el **"Modo Micrófono"** para testear el pipeline completo utilizando la placa de sonido de tu PC sin necesidad de drivers privativos.

---

## 🔄 Flujo de Trabajo (Pull Requests)

1.  Crea una nueva rama (branch) para tu contribución. Usa un nombre descriptivo:
    ```bash
    git checkout -b fix/autoforge-state-machine
    # o
    git checkout -b feature/pyside6-migration
    ```
2.  Realiza tus cambios.
3.  **Comentarios:** Si modificas lógica matemática compleja o la máquina de estados, por favor añade comentarios explicando el "por qué".
4.  Haz commit de tus cambios:
    ```bash
    git commit -m "Fix: Se corrige el escalado del eje Y en el modo AutoForge"
    ```
5.  Haz push a tu rama:
    ```bash
    git push origin fix/autoforge-state-machine
    ```
6.  Abre un **Pull Request** en este repositorio describiendo tus cambios.

## 📝 Estilo de Código e Idioma

- **Idioma:** Preferimos que los comentarios, la documentación y los nombres de variables (en lo posible) se mantengan en **español** para facilitar el acceso a la comunidad científica local de Argentina y Latinoamérica.
- **Estilo:** Intentamos seguir **PEP 8**. Mantén el código limpio y legible.

¡Esperamos tus aportes!
