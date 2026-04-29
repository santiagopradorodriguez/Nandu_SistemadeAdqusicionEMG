# -*- coding: utf-8 -*-
"""
Este script muestra una ventana con las instrucciones de uso para todo el
sistema de adquisición y análisis de EMG.
"""

import tkinter as tk
from tkinter import font
from tkinter.scrolledtext import ScrolledText

INSTRUCTIONS_TEXT = """
¡Bienvenido al Sistema de Adquisición y Análisis de EMG!

Este es un conjunto de herramientas para la adquisición, procesamiento y análisis de señales electromiográficas. El flujo de trabajo recomendado se divide en los siguientes pasos.

PASO 1: Medir (CodigoUnificador_integrado.py)

Este es el programa principal para adquirir y grabar las señales de los electrodos.

Uso:
1.  Configuración: Antes de iniciar, ajusta los parámetros en el panel "Configuración de Adquisición".
    -   Dispositivo: Selecciona tu tarjeta NI-DAQ.
    -   Modo Prueba: Actívalo para usar señales simuladas sin hardware.
    -   Sample Rate: Frecuencia de muestreo (ej: 6000 S/s).
    -   Canales: Selecciona los canales físicos que vas a usar.

2.  Iniciar Adquisición: Presiona "Iniciar Adquisición" para ver las señales en tiempo real.

3.  Grabar Medición:
    -   Presiona "Empezar a Grabar". El sistema grabará una fase de "Ruido" y luego la "Señal".
    -   Si activaste el metrónomo, este guiará la fase de señal.
    -   Presiona "Detener Grabación" al finalizar.

4.  Exportar: Al detener, se te pedirá un nombre para la medición.
    -   El programa crea una carpeta en "base_de_datos_electrodos".
    -   Guarda los datos en formatos .wav (por canal), .csv (completo), .png (gráfico) y un metadata.json con los parámetros.

PASO 2: Análisis de Pulsos (analisis_por_track_integrado.py)

Procesa las mediciones para calcular el pulso promedio y la relación señal-ruido (SNR) de cada canal de forma independiente.

Uso:
1.  Ejecuta el script para abrir una ventana de selección.
2.  Elige las mediciones a analizar y los canales a procesar.
3.  Puedes ejecutarlo en modo interactivo para curar los datos, excluyendo ventanas (pulsos) de mala calidad.
4.  Resultados: Genera archivos de análisis (avg.png, pulses.png, results.json) en la carpeta de cada canal.
5.  Comparación: Si seleccionas varias mediciones, crea gráficos comparativos en la carpeta "analisis_comparativos".

PASO 3: Análisis de Correlación (correlaciondesenales.py)

Este es un script de análisis avanzado, diseñado para estudiar la coordinación entre diferentes músculos (canales).

Uso:
1.  Funciona de manera similar al análisis del PASO 2, pero implementa una estrategia "Master/Slave".
2.  Alineación: El canal 0 ("Master") se usa como referencia. El script calcula los desfasajes óptimos entre sus pulsos y luego aplica estos mismos desfasajes a los demás canales ("Slaves").
3.  Resultado Principal: Esto permite preservar la relación temporal entre las activaciones de los diferentes músculos. El resultado más importante es el gráfico "patron_muscular_....png", que superpone los pulsos promedio alineados de todos los canales, mostrando el patrón de activación coordinado.

Herramientas Adicionales y de Análisis
--------------------------------------

Extractor de Datos Procesados (extractor_de_datos_procesados.py)
Este script recolecta todos los pulsos individuales (ventanas) de los análisis y los organiza en una nueva base de datos.

-   Uso: Ejecútalo después de haber analizado tus mediciones.
-   Funcionamiento: Lee los archivos "analisis_results.json", extrae cada pulso, y lo guarda como un CSV individual en la carpeta "base_de_datos_letras", organizado por letra y canal.
-   Calibración: Calcula la amplitud real de cada pulso basándose en la resistencia del electrodo guardada en el metadata y genera un resumen en "amplitudes_maximas.csv".

Análisis Estadístico de Pulsos (analisis_estadistico_pulsos.py)
Realiza un análisis estadístico a partir de los datos extraídos por el script anterior.

-   Uso: Ejecútalo después del "Extractor de Datos".
-   Funcionamiento: Carga "amplitudes_maximas.csv", agrupa los pulsos por letra, y calcula estadísticas descriptivas (media, desviación estándar, etc.) para la amplitud real. Genera también un histograma comparativo.

Visor de Resultados (electrode_viewer_4.py)
Un explorador visual para navegar y comparar rápidamente los resultados de los análisis.

-   Uso: La aplicación carga y muestra una cuadrícula con miniaturas de los pulsos promedio de cada medición.
-   Detalles: Haz clic en una miniatura para ver todos los gráficos y metadatos asociados a esa medición y sus canales.
-   Comparación: Permite seleccionar varias mediciones para ver sus resultados lado a lado.

Plotter Calibrado (plotter_calibrado.py)
Herramienta para generar un gráfico de alta calidad de una medición específica a partir de sus datos crudos en CSV.

-   Uso: Al ejecutarlo, te pedirá que selecciones la carpeta de una medición.
-   Funcionamiento: Carga el .csv, aplica calibración de ganancia, convierte la señal a microvolts (µV) y aplica filtros (Notch y Pasabanda).
-   Resultado: Muestra y guarda un gráfico de alta calidad con cada canal en un subplot separado.

Editor de Mediciones (editor_mediciones.py)
Una utilidad para renombrar mediciones de formato "prueba" a un formato "formal" (Letra_Prueba_Sujeto).

-   Uso: Selecciona una medición de la lista y edita sus componentes (letra, prueba, sujeto).
-   Funcionamiento: El script renombra la carpeta de la medición y actualiza automáticamente los archivos "metadata.json" de todos sus canales internos para reflejar el nuevo nombre.

Actualizador de Metadata (actualizar_metadata.py)
Script para añadir o actualizar la resistencia de los electrodos en los metadatos de forma masiva.

-   Uso: Ejecútalo si necesitas añadir la resistencia a mediciones antiguas.
-   Funcionamiento: Escanea la `base_de_datos_electrodos`, extrae el valor de resistencia del nombre de la carpeta (ej: '...10ohm...'), y lo escribe en el campo "resistencia_ohm" de los archivos `metadata.json` correspondientes.
"""

def main():
    root = tk.Tk()
    root.title("Instrucciones de Uso")
    root.geometry("800x700")

    text_area = ScrolledText(root, wrap=tk.WORD, padx=10, pady=10, font=("Helvetica", 11))
    text_area.pack(expand=True, fill="both")

    # --- Definir estilos ---
    title_font = font.Font(family="Helvetica", size=16, weight="bold")
    h1_font = font.Font(family="Helvetica", size=14, weight="bold")
    h2_font = font.Font(family="Helvetica", size=12, weight="bold")
    bold_font = font.Font(family="Helvetica", size=11, weight="bold")

    text_area.tag_configure("title", font=title_font, spacing3=15)
    text_area.tag_configure("h1", font=h1_font, foreground="#00529B", spacing3=10)
    text_area.tag_configure("h2", font=h2_font, foreground="#333333", spacing3=5)
    text_area.tag_configure("bold", font=bold_font)

    # --- Insertar y formatear texto ---
    lines = INSTRUCTIONS_TEXT.strip().split('\n')
    
    is_first_line = True
    for line in lines:
        if is_first_line:
            text_area.insert(tk.END, line + '\n\n', "title")
            is_first_line = False
        elif line.startswith("PASO"):
            text_area.insert(tk.END, f"\n{line}\n", "h1")
        elif line.startswith("---"):
             text_area.insert(tk.END, f"\n{line}\n", "h1")
        elif line.startswith("Uso:") or line.startswith("Resultados:") or line.startswith("Comparación:") or line.startswith("Herramienta Adicional:") or (line.endswith(")") and "(" in line):
            text_area.insert(tk.END, f"\n{line}\n", "h2")
        else:
            text_area.insert(tk.END, line + '\n')

    text_area.config(state="disabled") # Hacer el texto de solo lectura
    root.mainloop()

if __name__ == "__main__":
    main()
