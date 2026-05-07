# -*- coding: utf-8 -*-
"""
extractor_de_datos_procesados.py - v1.3

Este script automatiza la extracción de pulsos individuales (ventanas recortadas)
de las mediciones ya procesadas y las organiza en una nueva base de datos
estructurada por letra y canal.

Funcionamiento:
1. Escanea la carpeta 'base_de_datos_electrodos'.
2. Identifica las mediciones "formales" que ya han sido analizadas,
   buscando carpetas cuyo nombre comience con una letra mayúscula seguida de
   un guion bajo (ej: 'A_Prueba1_Sujeto1').
3. Para cada medición formal, recorre sus subcarpetas de canal (ej: 'canal_0').
4. Lee el archivo 'analisis_results.json' para obtener los segmentos de pulso.
5. Lee el 'metadata.json' para obtener la resistencia del electrodo.
6. Calcula la ganancia y la amplitud real para cada pulso usando la fórmula:
   V_real = V_medida / (1 + R_fija / R_electrodo).
7. Acumula los datos en 'base_de_datos_letras', guardando cada pulso con un nombre único.
8. Genera un archivo 'amplitudes_maximas.csv' actualizado que ahora incluye
   amplitud medida, resistencia, ganancia y la amplitud real calibrada.
"""

import os
import json
import numpy as np
import re
import shutil
import pandas as pd # Usaremos pandas para escribir el CSV de resumen fácilmente

def calculate_real_amplitude(df):
    """Calcula Ganancia (G) para información. La amplitud ya viene calibrada."""
    R_fija = 49400.0 # 49.4 kΩ
    
    df_copy = df.copy() # Evitar SettingWithCopyWarning
    
    # Calcular G para mostrar en el CSV a modo de información
    df_copy['Ganancia (G)'] = 1 + (R_fija / df_copy['Resistencia'])
    
    # --- CORRECCIÓN ---
    # La amplitud que extraemos de analisis_results.json YA FUE CALIBRADA
    # a microvoltios en analisis_por_track_integrado.py. NO debemos dividirla de nuevo.
    df_copy['Amplitud_Real'] = df_copy['Amplitud_Medida']
    
    df_copy['Error_Amplitud_Real'] = df_copy['Error_Amplitud_Medida']
    
    return df_copy

def main():
    """Función principal del script de extracción."""
    
    # --- Configuración de directorios ---
    fuente_dir = "base_de_datos_electrodos" # Directorio de donde se leen los análisis
    destino_dir = "base_de_datos_letras"   # Directorio donde se guardan los pulsos
    R_FIJA = 49400.0                       # Resistencia fija del circuito en Ohms (49.4 kΩ)
    
    print(f"--- Iniciando Extractor de Datos Procesados v1.3 (con Calibración de Amplitud) ---")

    if not os.path.isdir(fuente_dir):
        print(f"❌ ERROR: El directorio fuente '{fuente_dir}' no existe. No hay nada que procesar.")
        return

    # Preguntar al usuario si desea limpiar la base de datos de destino
    if os.path.isdir(destino_dir):
        respuesta = input(f"⚠️  El directorio de destino '{destino_dir}' ya existe. ¿Desea limpiarlo antes de empezar? (s/n): ").lower()
        if respuesta == 's':
            try:
                shutil.rmtree(destino_dir)
                print(f"🗑️  Directorio '{destino_dir}' limpiado.")
            except Exception as e:
                print(f"❌ ERROR: No se pudo limpiar el directorio de destino. Error: {e}")
                return

    os.makedirs(destino_dir, exist_ok=True)
    print(f"📂 Directorio de destino asegurado: '{destino_dir}'")

    # Regex para identificar carpetas de mediciones formales (Letra_...)
    regex_formal = re.compile(r"^[A-Z]_")
    
    total_pulsos_extraidos = 0
    amplitudes_data = [] # Inicializar la lista para todos los datos de amplitud
    path_resumen_csv = os.path.join(destino_dir, "amplitudes_maximas.csv")


    # 1. Recorrer todas las mediciones en la base de datos de electrodos
    for nombre_medicion in sorted(os.listdir(fuente_dir)):
        path_medicion = os.path.join(fuente_dir, nombre_medicion)

        if os.path.isdir(path_medicion) and regex_formal.match(nombre_medicion):
            print(f"\n✅ Encontrada medición formal: '{nombre_medicion}'")
            
            letra = nombre_medicion[0]
            
            # 2. Recorrer los canales dentro de la medición
            for nombre_canal in sorted(os.listdir(path_medicion)):
                path_canal = os.path.join(path_medicion, nombre_canal)
                
                if os.path.isdir(path_canal) and nombre_canal.startswith("canal_"):
                    # 3. Buscar el archivo de resultados del análisis
                    # --- NUEVO: Leer la resistencia desde el metadata.json ---
                    path_metadata = os.path.join(path_canal, "metadata.json")
                    resistencia_ohm = None
                    if os.path.exists(path_metadata):
                        try:
                            with open(path_metadata, 'r', encoding='utf-8') as f_meta:
                                metadata = json.load(f_meta)
                                resistencia_ohm = metadata.get("resistencia_ohm")
                                if resistencia_ohm is None:
                                    print(f"    -> ⚠️  ADVERTENCIA: No se encontró 'resistencia_ohm' en '{path_metadata}'. No se podrá calibrar la amplitud para este canal.")
                                else:
                                    print(f"  - Resistencia detectada: {resistencia_ohm} Ω")
                        except Exception as e:
                            print(f"    -> ⚠️  ADVERTENCIA: Error al leer metadata.json en '{path_canal}': {e}")
                    else:
                        print(f"    -> ⚠️  ADVERTENCIA: No se encontró 'metadata.json' en '{path_canal}'. No se podrá calibrar la amplitud.")

                    path_json = os.path.join(path_canal, "analisis_results.json")
                    
                    if os.path.exists(path_json):
                        print(f"  - Procesando '{nombre_canal}'...")
                        try:
                            with open(path_json, 'r', encoding='utf-8') as f:
                                datos_analisis = json.load(f)
                            
                            segmentos = datos_analisis.get("segmentos_rs")

                            if not segmentos or not isinstance(segmentos, list):
                                print("    -> ⚠️  No se encontraron 'segmentos_rs' válidos en el JSON.")
                                continue

                            # 4. Crear la carpeta de destino final
                            path_destino_final = os.path.join(destino_dir, letra, nombre_canal)
                            os.makedirs(path_destino_final, exist_ok=True)

                            # 5. Guardar cada segmento como un CSV individual
                            for i, pulso in enumerate(segmentos):
                                # --- MODIFICADO: Nombre de archivo único para evitar sobrescrituras ---
                                nombre_medicion_base = os.path.splitext(nombre_medicion)[0]
                                nombre_archivo_pulso = f"{nombre_medicion_base}_pulso_{i+1:03d}.csv"
                                path_csv_pulso = os.path.join(path_destino_final, nombre_archivo_pulso)
                                np.savetxt(path_csv_pulso, np.array(pulso), delimiter=",", header="amplitud", comments="")
                                total_pulsos_extraidos += 1
                                
                                # --- MODIFICADO: Calcular y guardar todos los datos de amplitud ---
                                amplitud_medida = np.max(pulso)

                                amplitudes_data.append({
                                    "nombre_pulso": nombre_archivo_pulso,
                                    "Amplitud_Medida": amplitud_medida,
                                    "Resistencia": resistencia_ohm,
                                    # Placeholder para el error, ajustar si se tiene el dato
                                    "Error_Amplitud_Medida": 0.0 
                                })

                            print(f"    -> ✅ Se extrajeron y guardaron {len(segmentos)} pulsos desde '{nombre_medicion}'")

                        except Exception as e:
                            print(f"    -> ❌ ERROR al procesar el archivo JSON '{path_json}': {e}")
                    else:
                        print(f"  - Omitiendo '{nombre_canal}' (no se encontró 'analisis_results.json').")

    # --- MODIFICADO: Guardar el archivo de resumen de amplitudes actualizado al final ---
    if amplitudes_data:
        df_amplitudes = pd.DataFrame(amplitudes_data)
        df_final = calculate_real_amplitude(df_amplitudes)
        
        # Ordenar columnas para mayor claridad
        column_order = ["nombre_pulso", "Amplitud_Medida", "Error_Amplitud_Medida", "Resistencia", "Ganancia (G)", "Amplitud_Real", "Error_Amplitud_Real"]
        df_final = df_final[column_order]
        df_final.to_csv(path_resumen_csv, index=False, float_format='%.6f')
        print(f"\n✅ Resumen de amplitudes guardado en '{path_resumen_csv}'")

    print(f"\n--- ✨ Proceso Finalizado ---")
    print(f"Se extrajeron un total de {total_pulsos_extraidos} pulsos.")
    print(f"Los datos han sido organizados en la carpeta '{destino_dir}'.")

if __name__ == "__main__":
    main()