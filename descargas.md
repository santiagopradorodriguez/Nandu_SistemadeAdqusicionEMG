# Descargas e Instalación de Ñandú LSD

Guía oficial de descargas de la suite **Ñandú LSD** para procesamiento de señales electromiográficas (EMG) y Deep Learning, junto con los controladores de hardware indispensables para la adquisición de datos en tiempo real.

---

## 1. Paquetes de Software Ñandú LSD

Los ejecutables empaquetados son autónomos y contienen todas las librerías científicas requeridas (PySide6, PyTorch, UMAP, SciPy, NumPy, sounddevice) sin necesidad de configurar entornos de Python adicionales.

| Sistema Operativo | Paquete Distribuible | Enlace de Descarga | Archivo de Ejecución |
| :--- | :--- | :--- | :--- |
| **Linux (x64)** | `NanduLsd_Linux_x64.tar.gz` | [Descargar de Google Drive](https://drive.google.com/file/d/1W7sNQCeWO6bzSvzh1CfXOYSdqgkZ4bJB/view?usp=drive_link) | `./run_nandu.sh` |
| **Windows (x64)** | `NanduLsd_Windows_x64.zip` | Compilación local con `build.bat` | `NanduLsd.exe` |

### Instrucciones de Ejecución en Linux
1. Descargar el paquete desde el siguiente enlace:
   - **Enlace de Descarga Directa:** [Ñandú LSD para Linux x64 (Google Drive)](https://drive.google.com/file/d/1W7sNQCeWO6bzSvzh1CfXOYSdqgkZ4bJB/view?usp=drive_link)
2. Descomprimir el archivo en tu carpeta de preferencia:
   ```bash
   tar -xzvf NanduLsd_Linux_x64.tar.gz
   cd NanduLsd
   ```
3. Dar permisos de ejecución e iniciar la aplicación:
   ```bash
   chmod +x run_nandu.sh
   ./run_nandu.sh
   ```

### Instrucciones de Ejecución en Windows
1. Descargar y descomprimir `NanduLsd_Windows_x64.zip`.
2. Ingresar a la carpeta descomprimida `NanduLsd`.
3. Ejecutar haciendo doble clic en `NanduLsd.exe`.

---

## 2. Controlador de Hardware National Instruments (NI-DAQmx)

Para realizar la captura física de señales EMG mediante placas de adquisición National Instruments conectadas por USB (por ejemplo, la placa multifunción **NI USB-6212**), es **estrictamente obligatorio** tener instalado en el sistema operativo el controlador oficial **NI-DAQmx**.

### ¿Por qué es necesario el driver?
La librería `nidaqmx` incluida en la aplicación actúa como un puente de alto nivel. Para abrir la comunicación en tiempo real con el convertidor analógico-digital (ADC), gestionar el búfer de muestreo a $f_s = 2000\text{ Hz}$ y sincronizar los canales de entrada analógica (`Dev1/ai0`, `Dev1/ai1`, `Dev1/ai2`, `Dev1/ai3`), el sistema operativo requiere las bibliotecas de bajo nivel y los controladores de dispositivo C provistos por National Instruments.

Si el controlador no está instalado o la placa no se encuentra conectada por USB, el sistema emitirá una advertencia y operará automáticamente en modo de prueba / micrófono acústico.

### Enlaces de Descarga Oficial del Driver

- **Descarga para Windows:**
  - [Instalador Online Oficial NI-DAQmx (National Instruments)](https://download.ni.com/support/nipkg/products/ni-d/ni-daqmx/25.8/online/ni-daqmx_25.8_online.exe)
  - [Página de Descargas de Controladores NI-DAQmx](https://www.ni.com/es/support/downloads/drivers/download.ni-daq-mx.html)

- **Descarga para Linux:**
  - [NI Linux Device Drivers](https://www.ni.com/es/support/downloads/drivers/download.ni-linux-device-drivers.html)

### Pasos de Instalación del Driver
1. Descargar el instalador de NI-DAQmx correspondiente a tu sistema operativo.
2. Ejecutar el asistente de instalación y seleccionar los componentes estándar de soporte para adquisición multifunción DAQ.
3. Reiniciar el equipo tras completar la instalación.
4. Conectar la placa NI USB al puerto de la computadora y verificar su detección abriendo la aplicación Ñandú LSD.
