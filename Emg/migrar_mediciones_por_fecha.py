# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Script de migración para organizar mediciones en carpetas por fecha.
# ==============================================================================

"""
migrar_mediciones_por_fecha.py

Script correspondiente a la Fase 1 del Plan de Reestructuración.
Organiza las mediciones existentes en la 'base_de_datos_electrodos' 
agrupándolas en subcarpetas por fecha (YYYY-MM-DD).
"""

import os
import json
import shutil
import re

def migrar_analisis_comparativos():
    """
    Organiza los archivos y carpetas en 'analisis_comparativos' en subcarpetas por fecha.
    Busca un bloque de 8 dígitos (YYYYMMDD) en cualquier parte del nombre.
    """
    comp_dir = "analisis_comparativos"
    
    # Regex para identificar si el nombre ya es una carpeta con formato de fecha (YYYY-MM-DD)
    date_folder_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    if not os.path.exists(comp_dir):
        print(f"⚠️ El directorio '{comp_dir}' no existe. Saltando...")
        return

    # Patrón para capturar YYYYMMDD (8 dígitos seguidos)
    date_pattern = re.compile(r"(\d{4})(\d{2})(\d{2})")
    
    # Listamos todo el contenido (archivos y carpetas) en la raíz de comparaciones
    items = os.listdir(comp_dir)
    movidos = 0

    print("\n--- Iniciando Migración de Análisis Comparativos ---")

    for item in items:
        # Ignorar si ya es una carpeta de fecha o archivos especiales
        if date_folder_pattern.match(item):
            continue
            
        ruta_origen = os.path.join(comp_dir, item)
        match = date_pattern.search(item)
        
        if match:
            año, mes, día = match.groups()
            fecha_folder = f"{año}-{mes}-{día}"
            
            # Evitar mover una carpeta dentro de sí misma
            if item == fecha_folder:
                continue

            ruta_destino_folder = os.path.join(comp_dir, fecha_folder)
            os.makedirs(ruta_destino_folder, exist_ok=True)
            
            try:
                print(f"📊 Moviendo: '{item}' -> '{fecha_folder}/'")
                # shutil.move funciona tanto para archivos como para directorios
                shutil.move(ruta_origen, os.path.join(ruta_destino_folder, item))
                movidos += 1
            except Exception as e:
                print(f"❌ Error moviendo '{item}': {e}")

    print(f"✅ Archivos de análisis migrados: {movidos}")

def migrar_base_datos():
    base_dir = "base_de_datos_electrodos"
    
    if not os.path.exists(base_dir):
        print(f"❌ El directorio '{base_dir}' no existe. Ejecuta el script desde la raíz del proyecto.")
        return

    # Regex para identificar si una carpeta ya es una carpeta de fecha (YYYY-MM-DD)
    date_folder_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    # Obtener lista de carpetas (ignorando archivos sueltos si los hay)
    carpetas = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    movidas = 0
    errores = 0

    print("--- Iniciando Migración Histórica de Mediciones ---")

    for carpeta in carpetas:
        # Si la carpeta ya tiene formato de fecha o es "Sin_Fecha", la ignoramos (ya fue procesada)
        if date_folder_pattern.match(carpeta) or carpeta == "Sin_Fecha":
            continue
            
        ruta_medicion = os.path.join(base_dir, carpeta)
        fecha_medicion = "Sin_Fecha" # Valor por defecto si falla la extracción
        
        # Buscar el archivo metadata.json (normalmente en canal_0)
        ruta_metadata = os.path.join(ruta_medicion, "canal_0", "metadata.json")
        
        if os.path.exists(ruta_metadata):
            try:
                with open(ruta_metadata, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    fecha_iso = metadata.get("measurement_date")
                    if fecha_iso:
                        # Extraer solo la parte YYYY-MM-DD (ej: "2023-10-27T10:00:00" -> "2023-10-27")
                        fecha_medicion = fecha_iso.split("T")[0]
            except Exception as e:
                print(f"⚠️ Error leyendo {ruta_metadata}: {e}")
        else:
            print(f"⚠️ No se encontró metadata en '{carpeta}'. Se moverá a 'Sin_Fecha'.")
            
        # Crear la carpeta de la fecha si no existe
        ruta_destino_fecha = os.path.join(base_dir, fecha_medicion)
        os.makedirs(ruta_destino_fecha, exist_ok=True)
        
        # Mover la medición a su nueva subcarpeta
        ruta_destino_final = os.path.join(ruta_destino_fecha, carpeta)
        
        try:
            print(f"📁 Moviendo: '{carpeta}' -> '{fecha_medicion}/{carpeta}'")
            shutil.move(ruta_medicion, ruta_destino_final)
            movidas += 1
        except Exception as e:
            print(f"❌ Error al mover '{carpeta}': {e}")
            errores += 1

    print("\n--- RESUMEN DE MIGRACIÓN ---")
    print(f"✅ Carpetas migradas exitosamente: {movidas}")
    print(f"❌ Errores: {errores}")

if __name__ == "__main__":
    migrar_base_datos()
    migrar_analisis_comparativos()