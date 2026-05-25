# 🗺️ Roadmap EMG Analytics Studio (Futuras Versiones v4.0+)

Este documento contiene la lista de bugs conocidos pendientes de resolución y los objetivos arquitectónicos para las futuras actualizaciones.

## 🐛 Bugs Conocidos a Corregir
- **Ventana de Curación Híbrida:** Integrar completamente la ventana interactiva de curación (actualmente en Tkinter/Matplotlib) nativamente dentro del flujo de PySide6 para evitar que la ejecución dependa de scripts puente temporales (`temp_procesar.py`).
- **Modo Prueba (Simulador):** Arreglar el bug crítico que causa el cierre/cuelgue de la aplicación (`CodigoUnificador_integrado.py`) al presionar "Detener Adquisición" mientras se está corriendo el Modo Prueba con archivos pre-grabados.

## ✨ Nuevas Características (Features)
- **ElectrodeViewer Expandido:** Agregar soporte para visualizar no solo la señal principal, sino también los gráficos de "Evolución Temporal" y "Espectrogramas" directamente dentro del módulo ElectrodeViewer.
- **Visualización de Configuración de Electrodos:** Permitir que, si se coloca una imagen (ej. `configuracion.jpg`/`.png`) mostrando la disposición de los electrodos en la carpeta de la medición, esta se cargue y se muestre automáticamente en la interfaz del visor para contexto anatómico rápido.
- **Adquisición Dual (EMG + Audio):** Integrar un grabador de audio simultáneo que permita capturar audio de un micrófono sincronizado en paralelo con la placa EMG, permitiendo correlacionar comandos de voz o ruidos musculares mecánicos con la señal eléctrica.

## 🏗️ Refactorización de Arquitectura
- **Migración Total a PySide6:** Portar los scripts auxiliares/independientes restantes de Tkinter a la nueva arquitectura PySide6, específicamente:
  - `extractor_de_datos_procesados.py`
  - `editor_mediciones.py`
  - Cualquier herramienta *standalone* legada en la carpeta del proyecto.
