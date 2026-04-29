Sistema de Adquisición y Análisis EMG
=======================================

Este documento proporciona las instrucciones para configurar y utilizar el conjunto de herramientas de software para la adquisición y análisis de señales electromiográficas (EMG).

Instalación Rápida (Recomendado)
---------------------------------
Para usuarios de Windows, se ha proporcionado un script que automatiza todo el proceso de configuración.

1.  **Ejecuta `install_dependencies.bat`**: Haz doble clic en el archivo `install_dependencies.bat`.
2.  **Sigue las instrucciones**: El script creará un entorno virtual, instalará todas las librerías de Python necesarias y te guiará en el proceso.
3.  **Activa el entorno**: Una vez finalizado, el script te indicará cómo activar el entorno virtual para poder ejecutar los programas. Generalmente, será ejecutando `.\venv\Scripts\activate` en tu terminal.

Instalación Manual
------------------
Si prefieres configurar el entorno manualmente o no estás en Windows, sigue estos pasos:

### Paso 1: Crear un Entorno Virtual

Es una buena práctica aislar las dependencias del proyecto.
```bash
# 1. Abre una terminal en la carpeta del proyecto.
# 2. Crea el entorno virtual:
python -m venv venv

# 3. Activa el entorno:
# En Windows:
.\venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### Paso 2: Instalar las Librerías de Python

Este proyecto utiliza un archivo `requirements.txt` que contiene la lista de todas las librerías necesarias. Con tu entorno virtual activado, ejecuta:

```bash
pip install -r requirements.txt
```

Esto instalará automáticamente las versiones compatibles de todas las librerías.

### Paso 3: Instalar el Driver de National Instruments

**¡IMPORTANTE!** La librería `nidaqmx` de Python es solo un conector. Para que el programa pueda comunicarse con el hardware de National Instruments (en modo real), necesitas instalar el driver **NI-DAQmx** en tu sistema.

-   **Descarga**: Puedes descargarlo desde el sitio web oficial de NI.
-   **Compatibilidad**: Asegúrate de descargar una versión del driver que sea compatible con tu sistema operativo y tu tarjeta de adquisición.

Sin este driver, el programa principal solo funcionará en "Modo Prueba".

Librerías Requeridas
--------------------
El archivo `requirements.txt` instalará las siguientes librerías:

-   **numpy**: Para manejo de arrays numéricos.
-   **pandas**: Para leer y manipular archivos .csv.
-   **matplotlib**: Para generar gráficos estáticos.
-   **scipy**: Para procesamiento de señales (filtros, espectrograma, etc.).
-   **soundfile**: Para leer y escribir archivos de audio .wav.
-   **pyqtgraph**: Para los gráficos en tiempo real de la interfaz principal.
-   **PyQt6**: El framework sobre el que corre `pyqtgraph`.
-   **Pillow**: Para el manejo de imágenes en el visor de electrodos.
-   **nidaqmx**: Para la comunicación con la tarjeta de adquisición de datos de National Instruments.
