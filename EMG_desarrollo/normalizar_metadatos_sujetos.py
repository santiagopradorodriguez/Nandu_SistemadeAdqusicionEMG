# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Script de normalización de metadatos de sujetos en la base de datos.
# ==============================================================================

import os
import json

def normalizar_sujeto(nombre_raw, ruta_carpeta=""):
    """
    Normaliza el nombre del sujeto a uno de los 4 canónicos:
    - Candela
    - Lucas
    - Petra
    - Santi
    """
    texto = (str(nombre_raw or "") + " " + str(ruta_carpeta or "")).lower()
    
    if "cande" in texto:
        return "Candela"
    elif "lucas" in texto:
        return "Lucas"
    elif "petra" in texto:
        return "Petra"
    elif "santi" in texto or "sujeto1" in texto or "sujeto 1" in texto or "sujeto_1" in texto:
        return "Santi"
    
    return nombre_raw if nombre_raw else "Desconocido"

def ejecutar_normalizacion(db_path, solo_simulacion=False):
    if not os.path.exists(db_path):
        print(f"[ERROR] La ruta {db_path} no existe.")
        return

    modificados = 0
    totales = 0
    desglose = {"Candela": 0, "Lucas": 0, "Petra": 0, "Santi": 0, "Otros": 0}

    print("=" * 70)
    print(f"INICIANDO NORMALIZACIÓN DE METADATOS EN: {db_path}")
    print(f"Modo: {'SIMULACIÓN (sin escribir)' if solo_simulacion else 'ESCRITURA REAL'}")
    print("=" * 70)

    for root, dirs, files in os.walk(db_path):
        for file in files:
            if file == "metadata.json":
                totales += 1
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except Exception as e:
                    print(f"[ERROR] No se pudo leer {filepath}: {e}")
                    continue

                sujeto_orig = meta.get("sujeto", "")
                sujeto_norm = normalizar_sujeto(sujeto_orig, filepath)

                if sujeto_norm in desglose:
                    desglose[sujeto_norm] += 1
                else:
                    desglose["Otros"] += 1

                if sujeto_orig != sujeto_norm:
                    modificados += 1
                    rel_path = os.path.relpath(filepath, db_path)
                    print(f"[CAMBIO] {rel_path}")
                    print(f"         Anterior: '{sujeto_orig}' -> Nuevo: '{sujeto_norm}'")
                    
                    meta["sujeto"] = sujeto_norm
                    if not solo_simulacion:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=4, ensure_ascii=False)

    print("=" * 70)
    print("RESUMEN DE NORMALIZACIÓN:")
    print(f"Archivos metadata.json analizados: {totales}")
    print(f"Archivos modificados: {modificados}")
    print("Desglose por Sujeto Canónico:")
    for suj, cant in desglose.items():
        print(f"  - {suj}: {cant} tomas/canales")
    print("=" * 70)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "base_de_datos_electrodos")
    ejecutar_normalizacion(db_path, solo_simulacion=False)
