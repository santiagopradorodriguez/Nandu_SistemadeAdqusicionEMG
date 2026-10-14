# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Auditoría integral de tríadas musculares y asignación de canales
#              en la base de datos de electrodos.
# ==============================================================================

import os
import json
from collections import defaultdict

def clasificar_rol(nombre_musculo):
    if not nombre_musculo:
        return "Desconocido"
    m = nombre_musculo.lower()
    if any(k in m for k in ["belly", "anterior", "digastric", "mylohyoid", "milohioideo", "vientre"]):
        return "Apertura"
    elif any(k in m for k in ["orbicular", "orbicularis"]):
        return "Protrusion"
    elif any(k in m for k in ["zygomaticus", "cigomatico", "depresor", "dao", "risorio", "risorius", "smile"]):
        return "Sonrisa"
    elif any(k in m for k in ["mic", "audio"]):
        return "Microfono"
    return "Otro"

def auditar_base_datos():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path = os.path.join(repo_root, "EMG_desarrollo", "base_de_datos_electrodos")
    
    fechas_excluidas = {"2026-06-01", "2026-06-03", "2026-06-10", "2026-09-22", "2026-10-02"}
    
    if not os.path.exists(db_path):
        print(f"Error: No existe el directorio {db_path}")
        return

    carpetas = sorted(os.listdir(db_path))
    
    resultados_por_fecha = {}

    for c in carpetas:
        c_path = os.path.join(db_path, c)
        if not os.path.isdir(c_path):
            continue
        if c in fechas_excluidas:
            continue
        
        tomas = sorted([
            t for t in os.listdir(c_path)
            if os.path.isdir(os.path.join(c_path, t)) and any(t.startswith(f"{v}_") for v in ["A", "E", "I", "O", "U"])
        ])
        
        if not tomas:
            continue
        
        # Analizar tomas de la fecha
        sujetos = set()
        fechas_reales = set()
        configuraciones = defaultdict(int)
        detalles_tomas = []
        
        for t in tomas:
            meta_path = os.path.join(c_path, t, "canal_0", "metadata.json")
            if not os.path.exists(meta_path):
                continue
            
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                continue
            
            sujeto = meta.get("sujeto", "Desconocido")
            sujetos.add(sujeto)
            fecha_real = meta.get("measurement_date", "Sin fecha")
            fechas_reales.add(fecha_real.split("T")[0] if "T" in fecha_real else fecha_real)
            
            muscles_map = meta.get("muscles_map", {})
            muscles_list = meta.get("muscles", [])
            
            ch0 = muscles_map.get("canal_0") or (muscles_list[0] if len(muscles_list) > 0 else "N/A")
            ch1 = muscles_map.get("canal_1") or (muscles_list[1] if len(muscles_list) > 1 else "N/A")
            ch2 = muscles_map.get("canal_2") or (muscles_list[2] if len(muscles_list) > 2 else "N/A")
            ch3 = muscles_map.get("canal_3") or (muscles_list[3] if len(muscles_list) > 3 else "N/A")
            
            r0 = clasificar_rol(ch0)
            r1 = clasificar_rol(ch1)
            r2 = clasificar_rol(ch2)
            
            cfg_tuple = (f"Ch0:{ch0} ({r0})", f"Ch1:{ch1} ({r1})", f"Ch2:{ch2} ({r2})")
            configuraciones[cfg_tuple] += 1
            
            detalles_tomas.append({
                "toma": t,
                "vocal": t[0],
                "sujeto": sujeto,
                "cfg": cfg_tuple,
                "roles": (r0, r1, r2)
            })
            
        resultados_por_fecha[c] = {
            "carpeta": c,
            "fechas_reales": list(fechas_reales),
            "sujetos": list(sujetos),
            "total_tomas": len(detalles_tomas),
            "configuraciones": dict(configuraciones),
            "detalles": detalles_tomas
        }
    
    # Imprimir reporte general
    print("=" * 80)
    print("AUDITORIA DE TRIADAS MUSCULARES EN BASE DE DATOS sEMG")
    print(f"Fechas excluidas: {sorted(list(fechas_excluidas))}")
    print("=" * 80)
    
    canonica_roles = ("Apertura", "Sonrisa", "Protrusion")
    
    for c, data in resultados_por_fecha.items():
        suj_str = ", ".join(data["sujetos"])
        f_real_str = ", ".join(data["fechas_reales"])
        print(f"\nCarpeta: {c} | Sujeto: {suj_str} | Fecha Real: {f_real_str} | Total Tomas: {data['total_tomas']}")
        
        for cfg, count in data["configuraciones"].items():
            ch0_str, ch1_str, ch2_str = cfg
            r0 = ch0_str.split("(")[-1].rstrip(")")
            r1 = ch1_str.split("(")[-1].rstrip(")")
            r2 = ch2_str.split("(")[-1].rstrip(")")
            
            es_canonica = (r0, r1, r2) == canonica_roles
            estado = "CANONICA (Ch0:Ap, Ch1:Son, Ch2:Prot)" if es_canonica else f"VARIANTE O PERMUTADA ({r0}-{r1}-{r2})"
            
            print(f"  [{count} tomas] Estado: {estado}")
            print(f"    - {ch0_str}")
            print(f"    - {ch1_str}")
            print(f"    - {ch2_str}")
            
    print("\n" + "=" * 80)
    print("FIN DE LA AUDITORIA")
    print("=" * 80)

if __name__ == "__main__":
    auditar_base_datos()
