import os
import sys
import json
import subprocess
import time

DOC_ID = "1ebZMc3PY3vjjbICRaGE17ADTD1tpMuZ3vNjdEw-zroA"
BASE_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono"
TRACKING_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/estado_insercion_docs"
PLAN_PATH = os.path.join(TRACKING_DIR, "BLOQUES_PLAN.json")

def run_cmd(cmd_list):
    p = subprocess.run(cmd_list, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def paso_1_limpiar_primera_pagina():
    print("--- PASO 1: Verificando y limpiando imágenes erróneas en página 1 ---")
    code, out, err = run_cmd(["gdocs", "readonly", "dump-body", DOC_ID])
    if code != 0:
        print(f"Error al leer dump-body: {err}")
        return False
    data = json.loads(out)
    content = data.get("content", [])
    
    # Inspeccionar elemento 1
    elem1 = content[1]
    elements = elem1.get("paragraph", {}).get("elements", [])
    
    # Contar cuantas imagenes hay antes del texto
    img_count = 0
    for el in elements:
        if "inlineObjectElement" in el:
            img_count += 1
        elif "textRun" in el:
            txt = el["textRun"]["content"]
            print(f"Texto detectado inmediatamente después: {repr(txt[:30])}")
            break
            
    print(f"Total de imágenes detectadas en cabecera: {img_count}")
    if img_count == 0:
        print("No hay imágenes en la cabecera. Nada que limpiar.")
        return True
        
    start_del = 1
    end_del = 1 + img_count
    print(f"Eliminando rango [{start_del}, {end_del}) que contiene únicamente las {img_count} imágenes...")
    code, out, err = run_cmd(["gdocs", "mutate", "delete", DOC_ID, str(start_del), str(end_del), "--confirm"])
    if code != 0:
        print(f"Error al eliminar imágenes de cabecera: {err}")
        return False
    print("Cabecera de página 1 restaurada limpiamente a su estado original.")
    return True

def paso_2_insertar_imagenes_en_posicion():
    print("\n--- PASO 2: Insertando cada imagen en su posición exacta antes de cada epígrafe ---")
    with open(PLAN_PATH) as f:
        plan = json.load(f)
        
    for b in plan:
        fig_num = b["figura_num"]
        img_rel = b.get("image_file")
        if not img_rel:
            continue
        img_path = os.path.join(BASE_DIR, img_rel)
        if not os.path.exists(img_path):
            print(f"No existe el archivo de imagen: {img_path}")
            continue
            
        # Buscar offset del epigrafe
        patron_busqueda = f"{fig_num}:"
        code, out, err = run_cmd(["gdocs", "readonly", "find-offset", DOC_ID, patron_busqueda])
        if code != 0:
            print(f"Error buscando offset de {patron_busqueda}: {err}")
            continue
            
        # Extraer Start Index
        start_idx = None
        for line in out.split("\n"):
            if "Start Index:" in line:
                start_idx = int(line.split(":")[1].strip())
                break
                
        if start_idx is None:
            print(f"No se encontró Start Index para {patron_busqueda}")
            continue
            
        print(f"Insertando {fig_num} ({img_rel}) en offset {start_idx}...")
        code, out, err = run_cmd(["gdocs", "mutate", "insert-image-from-file", DOC_ID, img_path, "--index", str(start_idx)])
        if code != 0:
            print(f"Error al insertar {fig_num}: {err}")
        else:
            print(f"{fig_num} insertada con éxito.")
        time.sleep(2)

if __name__ == "__main__":
    ok = paso_1_limpiar_primera_pagina()
    if ok:
        time.sleep(2)
        paso_2_insertar_imagenes_en_posicion()
    print("\nProceso de corrección finalizado.")
