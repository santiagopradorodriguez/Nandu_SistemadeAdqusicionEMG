# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Realiza análisis estadístico sobre pulsos extraídos y genera métricas.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Realiza análisis estadístico sobre pulsos extraídos y genera métricas.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
analisis_estadistico_pulsos.py - v1.0

Este script realiza un análisis estadístico a partir de los datos de amplitud
previamente extraídos y guardados en 'amplitudes_maximas.csv'.

Funcionamiento:
1. Localiza automáticamente el archivo 'amplitudes_maximas.csv' dentro de la
   carpeta 'base_de_datos_letras'.
2. Carga los datos usando pandas.
3. Extrae la "letra" (el tipo de medición, ej: 'A', 'B') del nombre de cada pulso.
4. Agrupa los datos por letra y calcula estadísticas descriptivas para la
   columna 'Amplitud_Real' (media, desviación estándar, etc.).
5. Muestra los resultados en la consola.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    """Función principal del script de análisis estadístico."""
    print("--- Iniciando Análisis Estadístico desde 'base_de_datos_letras' v1.0 ---")

    # --- Localización robusta de archivos ---
    # Obtener el directorio donde se encuentra este script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir la ruta al archivo CSV relativo a la ubicación del script
    letras_dir = os.path.join(script_dir, 'base_de_datos_letras')
    path_csv = os.path.join(letras_dir, 'amplitudes_maximas.csv')

    print(f"Buscando y cargando datos de amplitud desde '{path_csv}'...")

    if not os.path.exists(path_csv):
        print("[ERROR] No se encontraron datos de amplitud para analizar.")
        print("   Asegúrate de haber ejecutado 'extractor_de_datos_procesados.py' primero.")
        return

    try:
        # Cargar los datos
        df = pd.read_csv(path_csv)

        # Extraer la letra del nombre del pulso (ej. 'A' de 'A_ampli1_a_10ohm...')
        df['Letra'] = df['nombre_pulso'].str[0]

        print("\n[INFO] Análisis de Amplitud Real por Letra:")
        
        # Agrupar por la nueva columna 'Letra' y calcular estadísticas para 'Amplitud_Real'
        stats = df.groupby('Letra')['Amplitud_Real'].describe()

        print(stats.to_string()) # Imprimir estadísticas en consola

        # --- NUEVO: Generar y guardar el histograma de amplitudes ---
        print("\n[INFO] Generando histograma de amplitudes reales...")
        path_histograma = os.path.join(letras_dir, 'histograma_amplitudes_reales.png')

        plt.figure(figsize=(12, 7))
        
        # Dibujar un histograma por cada letra para compararlos
        for letra, grupo in df.groupby('Letra'):
            # --- CORRECCIÓN: La amplitud ya se encuentra en microvolts (µV) ---
            amplitudes_microV = grupo['Amplitud_Real']
            plt.hist(amplitudes_microV, bins=20, alpha=0.7, label=f'Letra {letra}')

        plt.title('Ñandú LSD - Distribución de Amplitudes Reales por Letra')
        plt.xlabel('Amplitud Real (µV)')
        plt.ylabel('Frecuencia (Número de Pulsos)')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(path_histograma)
        
        print(f"   -> [OK] Histograma guardado en '{path_histograma}'")
        print("\n--- Análisis Finalizado ---")

    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante el análisis: {e}")

if __name__ == "__main__":
    main()