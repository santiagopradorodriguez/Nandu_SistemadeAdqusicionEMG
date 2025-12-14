# Contribuir al Sistema de Adquisición EMG

¡Gracias por tu interés en contribuir! Este proyecto es una herramienta científica abierta, desarrollada **por y para la comunidad**.

El objetivo es democratizar el acceso a herramientas de electromiografía de calidad. Cualquier ayuda, desde corregir errores ortográficos hasta optimizar algoritmos de procesamiento de señales o mejorar la interfaz gráfica, es bienvenida.

## 📍 ¿Por dónde empezar?

Actualmente, el proyecto se encuentra en desarrollo activo y tenemos identificadas varias áreas donde necesitamos ayuda:

### 🐛 Reporte de Bugs y Mejoras de UI
- **Bug del Filtro Notch:** Al detener la adquisición en el `CodigoUnificador`, el checkbox del filtro Notch visualmente permanece "tildado" o no resetea su estado correctamente. Necesitamos asegurar la coherencia entre la UI y el estado interno.
- **Layout y Diseño:** Revisar la división de ventanas y la disposición de los widgets para asegurar que la interfaz sea usable en pantallas con resoluciones estándar (laptops).

### ⚡ Optimización y Rendimiento
- **Visor CSV (`visor_csv_interactivo.py`):** Este script experimenta lag o lentitud al cargar archivos de grabación muy largos. Se busca optimizar la lectura de datos (quizás usando *chunking* con Pandas) o el renderizado con `pyqtgraph`.
- **Análisis por Track:** Revisar la eficiencia en la generación de gráficos masivos.

### 🧪 Procesamiento de Señales 
- **Espectrogramas:** La generación de espectrogramas en `analisis_por_track_integrado.py` y en el código unificador requiere revisión para asegurar que los ejes y la escala de colores sean científicamente precisos.
- **Calibración de Resistencia:** Implementar una lógica de calibración específica para mediciones con resistencia de referencia de **100 Ohms**.

### 📚 Documentación
- **Comentarios en Código:** Agregar comentarios explicativos (Docstrings) dentro de las funciones críticas, especialmente en las secciones de cálculo matemático.
- **Entorno Virtual:** Ayudar a mantener actualizada la lista de `requirements.txt`.

---

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
4.  **Drivers (Importante):** Si vas a trabajar en el módulo de adquisición, necesitas tener instalado el driver **NI-DAQmx** de National Instruments, incluso si planeas usar el "Modo Prueba" (simulación), ya que la librería `nidaqmx` lo requiere para inicializarse.

---

## 🔄 Flujo de Trabajo (Pull Requests)

1.  Crea una nueva rama (branch) para tu contribución. Usa un nombre descriptivo:
    ```bash
    git checkout -b fix/bug-notch-filter
    # o
    git checkout -b feature/optimizacion-csv
    ```
2.  Realiza tus cambios.
3.  **Comentarios:** Si modificas lógica matemática compleja, por favor añade comentarios explicando el "por qué" de la fórmula.
4.  Haz commit de tus cambios:
    ```bash
    git commit -m "Fix: Se corrige el estado visual del checkbox Notch al detener grabación"
    ```
5.  Haz push a tu rama:
    ```bash
    git push origin fix/bug-notch-filter
    ```
6.  Abre un **Pull Request** en este repositorio describiendo tus cambios.

## 📝 Estilo de Código e Idioma

- **Idioma:** Preferimos que los comentarios, la documentación y los nombres de variables (en lo posible) se mantengan en **español** para facilitar el acceso a la comunidad científica local.
- **Estilo:** Intentamos seguir **PEP 8**. Mantén el código limpio y legible.

¡Esperamos tus aportes!
