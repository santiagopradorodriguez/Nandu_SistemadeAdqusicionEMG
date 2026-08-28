# Descargas y Compatibilidad

Este documento detalla la disponibilidad de los ejecutables pre-compilados del software **Ñandú LSD**, diseñados para utilizarse sin necesidad de instalar entornos de Python ni dependencias complejas, junto con los controladores de hardware indispensables para la adquisición de datos en tiempo real.

---

## 1. Windows (10 / 11)

El software ha sido completamente empaquetado para Windows en arquitectura de 64 bits.

**Estado:** Totalmente compatible y optimizado.

[![Descargar para Windows](https://img.shields.io/badge/Descargar_Nand%C3%BA_LSD_para_Windows-v5.0.0-0078D6?style=for-the-badge&logo=windows)](https://drive.google.com/drive/folders/1FtNJlB-4T-xKyZ0bZIJhjtzk2ltDawLK)

> **Instrucciones:**
> 1. Descarga el archivo comprimido desde el enlace de Google Drive.
> 2. Extrae la carpeta completa en tu computadora (ej. en el Escritorio o Documentos).
> 3. Entra a la carpeta y haz doble clic en `NanduLsd.exe`.

---

## 2. Linux (x64)

El paquete distribuible nativo para sistemas Linux (x86_64) ha sido compilado y probado con PySide6, PyTorch y aceleración gráfica.

**Estado:** Totalmente compatible y optimizado.

[![Descargar para Linux](https://img.shields.io/badge/Descargar_Nand%C3%BA_LSD_para_Linux-v5.0.0-FCC624?style=for-the-badge&logo=linux&logoColor=black)](https://drive.google.com/file/d/1W7sNQCeWO6bzSvzh1CfXOYSdqgkZ4bJB/view?usp=drive_link)

> **Instrucciones:**
> 1. Descarga el archivo comprimido `NanduLsd_Linux_x64.tar.gz` desde el enlace de Google Drive.
> 2. Extrae el paquete:
>    ```bash
>    tar -xzvf NanduLsd_Linux_x64.tar.gz
>    cd NanduLsd
>    ```
> 3. Concede permisos de ejecución al script lanzador e inicia el programa:
>    ```bash
>    chmod +x run_nandu.sh
>    ./run_nandu.sh
>    ```

---

## 3. Controlador de Hardware National Instruments (NI-DAQmx USB)

Para realizar la captura física de señales electromiográficas (EMG) mediante placas de adquisición National Instruments (por ejemplo, la placa multifunción **NI USB-6212**), es **estrictamente obligatorio instalar el driver oficial NI-DAQmx en la computadora**.

### ¿Por qué es necesario instalar el driver?
La librería `nidaqmx` incluida dentro del ejecutable es un conector de alto nivel en Python. Sin embargo, para comunicarse físicamente con el convertidor analógico-digital (ADC) por el puerto USB, gestionar el búfer de muestreo a $f_s = 2000\text{ Hz}$ y sincronizar los canales de entrada analógica (`Dev1/ai0`, `Dev1/ai1`, `Dev1/ai2`, `Dev1/ai3`), el sistema operativo necesita los controladores de bajo nivel y bibliotecas C oficiales de National Instruments.

> **Importante:** Si no se instala el driver o no se conecta la placa por USB, el programa no podrá realizar adquisiciones reales de hardware y conmutará automáticamente al modo de simulación / micrófono acústico.

### Enlaces Oficiales de Descarga del Driver NI-DAQmx

- **Para Windows (10 / 11):**
  - [Instalador Online Oficial NI-DAQmx para Windows (National Instruments)](https://download.ni.com/support/nipkg/products/ni-d/ni-daqmx/25.8/online/ni-daqmx_25.8_online.exe)
  - [Portal Oficial de Descargas de Controladores NI-DAQmx](https://www.ni.com/es/support/downloads/drivers/download.ni-daq-mx.html)

- **Para Linux:**
  - [Controladores NI Linux Device Drivers (NI-DAQmx)](https://www.ni.com/es/support/downloads/drivers/download.ni-linux-device-drivers.html)

### Pasos para la Puesta en Marcha
1. Descarga el instalador de NI-DAQmx correspondiente a tu sistema operativo.
2. Ejecuta el instalador y selecciona los paquetes de soporte para hardware multifunción DAQ.
3. Reinicia la computadora al finalizar la instalación.
4. Conecta el dispositivo NI USB y abre **Ñandú LSD** para comenzar a adquirir.
