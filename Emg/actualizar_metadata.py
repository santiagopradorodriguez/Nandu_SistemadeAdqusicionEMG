# -*- coding: utf-8 -*-
"""
actualizar_metadata.py - v1.0

Este script automatiza la creación o actualización de archivos metadata.json
para cada canal dentro de las mediciones en 'base_de_datos_electrodos'.

Funcionamiento:
1. Escanea el directorio 'base_de_datos_electrodos'.
2. Para cada carpeta de medición, intenta extraer un valor de resistencia del
   nombre de la carpeta (buscando un patrón como '10ohm', '47ohm', '2.2ohm', etc.).
3. Si encuentra una resistencia, recorre las subcarpetas de canal ('canal_0', 'canal_1', ...).
4. En cada carpeta de canal:
   a. Si 'metadata.json' existe, lo lee, añade/actualiza el campo 'resistencia_ohm'
      y guarda los cambios.
   b. Si 'metadata.json' no existe, crea uno nuevo con el campo 'resistencia_ohm'.
"""

import os
import json
import re

def main():
    """Función principal del script."""
    
    base_dir = "base_de_datos_electrodos"
    print(f"--- Iniciando Actualizador de Metadata v1.0 en '{base_dir}' ---")

    if not os.path.isdir(base_dir):
        print(f"❌ ERROR: El directorio base '{base_dir}' no existe. No hay nada que procesar.")
        return

    # Regex para encontrar la resistencia en el nombre de la carpeta.
    # Busca un número (posiblemente con decimales usando '_') seguido de 'ohm'.
    # Ejemplos: '47ohm', '100ohm', '2_2ohm'
    regex_resistencia = re.compile(r'(\d+(_\d+)?)(?=ohm)', re.IGNORECASE)

    for nombre_medicion in sorted(os.listdir(base_dir)):
        path_medicion = os.path.join(base_dir, nombre_medicion)

        if os.path.isdir(path_medicion):
            match = regex_resistencia.search(nombre_medicion)
            if not match:
                print(f"\n- Omitiendo '{nombre_medicion}': No se encontró patrón de resistencia (ej: '47ohm').")
                continue

            # Extraer y convertir la resistencia a un número.
            resistencia_str = match.group(1).replace('_', '.')
            resistencia_val = float(resistencia_str)
            
            print(f"\n✅ Medición '{nombre_medicion}': Resistencia detectada = {resistencia_val} Ω")

            # Recorrer las carpetas de canal dentro de la medición
            for item in sorted(os.listdir(path_medicion)):
                path_canal = os.path.join(path_medicion, item)
                if os.path.isdir(path_canal) and item.startswith("canal_"):
                    meta_path = os.path.join(path_canal, 'metadata.json')
                    metadata = {}
                    
                    # Si el archivo ya existe, lo leemos primero
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r') as f:
                            metadata = json.load(f)
                    
                    # Añadimos o actualizamos el valor de la resistencia
                    metadata['resistencia_ohm'] = resistencia_val
                    
                    # Guardamos el archivo JSON actualizado
                    with open(meta_path, 'w') as f:
                        json.dump(metadata, f, indent=4)
                    print(f"  -> Metadata actualizado en '{os.path.join(item, 'metadata.json')}'")

    print("\n--- ✨ Proceso Finalizado ---")

if __name__ == "__main__":
    main()