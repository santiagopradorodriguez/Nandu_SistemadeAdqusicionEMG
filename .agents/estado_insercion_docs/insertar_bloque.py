import sys
import os
import json
import subprocess
import time
import re

BASE_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono"
TRACKING_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/estado_insercion_docs"
ESTADO_PATH = os.path.join(TRACKING_DIR, "ESTADO.json")
PLAN_PATH = os.path.join(TRACKING_DIR, "BLOQUES_PLAN.json")

def clean_md_for_gdocs(text):
    lines = []
    for line in text.split("\n"):
        l = line.strip()
        # Omitir imagenes markdown inline
        if re.search(r"!\[.*\]\(.*\)", l):
            continue
        # Quitar titulos markdown
        if l.startswith("#"):
            l = re.sub(r"^#+\s*", "", l)
        # Separadores horizontales
        if l in ["---", "***", "___"]:
            continue
        # Convertir tablas markdown en lista legible
        if l.startswith("|"):
            if re.match(r"^\|[\s\-:]+\|$", l) or "---" in l:
                continue
            cells = [c.strip() for c in l.split("|")[1:-1]]
            if all(len(c) == 0 for c in cells):
                continue
            l = " - " + " | ".join(cells)
        # Quitar viñetas con asteriscos crudos para evitar error de gdocs markdown
        if l.startswith("* "):
            l = " - " + l[2:]
        # Quitar negritas y cursivas
        l = re.sub(r"\*\*([^*]+)\*\*", r"\1", l)
        l = re.sub(r"\*([^*]+)\*", r"\1", l)
        l = re.sub(r"`([^`]+)`", r"\1", l)
        l = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", l)
        lines.append(l)
    res = "\n".join(lines)
    res = re.sub(r"\n{3,}", "\n\n", res)
    return res.strip()

def run_cmd(cmd_list):
    p = subprocess.run(cmd_list, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def insertar_un_bloque():
    with open(ESTADO_PATH) as f:
        estado = json.load(f)

    with open(PLAN_PATH) as f:
        plan = json.load(f)

    doc_id = estado["doc_id"]
    bloque_idx = estado["bloque_actual"]

    if bloque_idx >= len(plan):
        print("FIN: Todos los bloques (12/12) han sido insertados exitosamente.")
        return False

    b = plan[bloque_idx]
    print(f"\n=======================================================")
    print(f"=== Procesando Bloque {b['id']}/{len(plan)}: {b['titulo']} ===")
    print(f"=======================================================")

    md_path = os.path.join(BASE_DIR, b["md_file"])
    with open(md_path) as f:
        raw_md = f.read()

    cuerpo_limpio = clean_md_for_gdocs(raw_md)

    # Encabezado formal de la seccion
    if b["id"] == 1:
        # En el bloque 1 el titulo principal ya habia sido insertado al probar
        texto_final = f"{cuerpo_limpio}\n\n"
    else:
        texto_header = f"{b['titulo']}\n\n"
        if b.get("subtitulo"):
            texto_header += f"{b['subtitulo']}\n\n"
        texto_final = f"{texto_header}{cuerpo_limpio}\n\n"

    print("1. Insertando texto...")
    code, out, err = run_cmd(["gdocs", "mutate", "append", doc_id, texto_final])
    if code != 0:
        print(f"ERROR al hacer append de texto: {err}")
        return False
    print("Texto insertado OK.")
    time.sleep(1.5)

    # 2. Insertar Imagen si corresponde
    img_rel = b.get("image_file")
    if img_rel:
        img_path = os.path.join(BASE_DIR, img_rel)
        if os.path.exists(img_path):
            print(f"2. Subiendo e insertando imagen {img_rel}...")
            # Insertar al final
            code, out, err = run_cmd(["gdocs", "mutate", "insert-image-from-file", doc_id, img_path])
            if code != 0:
                print(f"ERROR al insertar imagen: {err}")
                return False
            print("Imagen insertada OK.")
            time.sleep(2)

    # 3. Insertar epigrafe
    epigrafe = b.get("epigrafe")
    if epigrafe:
        print(f"3. Insertando epigrafe ({b['figura_num']})...")
        texto_epi = f"{epigrafe}\n\n\n"
        code, out, err = run_cmd(["gdocs", "mutate", "append", doc_id, texto_epi])
        if code != 0:
            print(f"ERROR al insertar epigrafe: {err}")
            return False
        print("Epigrafe insertado OK.")
        time.sleep(1)

    # 4. Actualizar estado
    estado["bloque_actual"] = bloque_idx + 1
    estado["ultimo_bloque_exitoso"] = b["id"]
    estado["historial"].append({
        "id": b["id"],
        "titulo": b["titulo"],
        "figura": b.get("figura_num"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })

    with open(ESTADO_PATH, "w") as f:
        json.dump(estado, f, indent=2)

    print(f"Bloque {b['id']} completado con exito. Estado guardado: {estado['bloque_actual']}/{len(plan)}.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--loop":
        while True:
            ok = insertar_un_bloque()
            if not ok:
                break
            time.sleep(2)
    else:
        insertar_un_bloque()
