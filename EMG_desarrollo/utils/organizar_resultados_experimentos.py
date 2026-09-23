# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Reorganiza y normaliza carpetas de resultados de PCA, UMAP y
#              Autoencoders por fecha y sujeto/nombre limpio.
# ==============================================================================

import os
import shutil
import glob
import time
import csv
import re

def sanitizar_nombre(nombre):
    palabras_informales = {
        "soy_un_boludo": "ensayo_general",
        "xd": "ensayo", "xdd": "ensayo", "xddç": "ensayo", "xxxdxdx": "ensayo", "xxxxx": "ensayo",
        "sdada": "ensayo", "sdf": "ensayo", "sdfsdf": "ensayo", "sssda": "ensayo", "sss": "ensayo",
        "a_veer": "ensayo", "aver": "ensayo", "a_verr": "ensayo", "averr": "ensayo", "a_verrrr": "ensayo", "veamos": "ensayo",
        "acasf": "ensayo", "again": "ensayo_reiterado", "otra_vez": "ensayo_reiterado",
        "bbb": "ensayo", "cc": "ensayo", "ccfcvv": "ensayo", "cxvv": "ensayo", "dasd": "ensayo",
        "ddd": "ensayo", "dddi": "ensayo", "dfgd": "ensayo", "ff": "ensayo", "ffff": "ensayo", "fianl": "ensayo_final",
        "ggg": "ensayo", "hhh": "ensayo", "hhhh": "ensayo", "jj": "ensayo", "jjj": "ensayo", "jjjj": "ensayo", "jjjjj": "ensayo", "jjjl": "ensayo",
        "k": "ensayo", "kk": "ensayo", "KK": "ensayo", "kkk": "ensayo", "kkkk": "ensayo", "kkkkxv": "ensayo",
        "ll": "ensayo", "lll": "ensayo", "llll": "ensayo", "mmjj": "ensayo", "noo": "ensayo", "ojo": "ensayo",
        "oo": "ensayo", "ooo": "ensayo", "pooo": "ensayo", "uuu": "ensayo", "vv": "ensayo", "vvvl": "ensayo",
        "´": "ensayo", "ÑÑÑ": "ensayo"
    }
    low = nombre.lower().strip()
    if low in palabras_informales:
        return palabras_informales[low]
    limpio = re.sub(r"[^a-zA-Z0-9_]+", "_", nombre).strip("_")
    return limpio if limpio else "ensayo"

def organizar_pca_umap(base_emg):
    origen_pca = os.path.join(base_emg, "deep_learning", "pca_umap_clustering", "resultados_pca_umap")
    destino_pca = os.path.join(base_emg, "resultados", "resultados_pca_umap")
    os.makedirs(destino_pca, exist_ok=True)
    
    if not os.path.exists(origen_pca):
        print(f"[PCA] Origen no existe: {origen_pca}")
        return

    carpetas = sorted([f for f in os.listdir(origen_pca) if os.path.isdir(os.path.join(origen_pca, f))])
    print(f"[PCA] Procesando {len(carpetas)} carpetas de resultados PCA/UMAP...")

    for f in carpetas:
        p_origen = os.path.join(origen_pca, f)
        csv_path = os.path.join(p_origen, "proyecciones_pca_2d.csv")
        mtime = os.path.getmtime(p_origen)
        date_str = time.strftime("%Y-%m-%d", time.localtime(mtime))
        time_str = time.strftime("%H%M%S", time.localtime(mtime))

        sujeto = "General"
        tomas = set()
        if os.path.exists(csv_path):
            try:
                with open(csv_path, encoding="utf-8", errors="ignore") as fp:
                    reader = csv.DictReader(fp)
                    for row in reader:
                        toma = row.get("Toma", "")
                        parts = toma.split("_")
                        if len(parts) > 2:
                            tomas.add(parts[-2])
            except Exception:
                pass
        
        if tomas:
            sujetos_list = sorted(list(tomas))
            if len(sujetos_list) == 1:
                sujeto = sujetos_list[0]
            else:
                sujeto = "_".join(sujetos_list[:3])
        elif "cande" in f.lower():
            sujeto = "Candela"
        elif "lucas" in f.lower():
            sujeto = "Lucas"
        elif "santi" in f.lower():
            sujeto = "Santi"
        elif "petra" in f.lower():
            sujeto = "Petra"

        desc_limpia = sanitizar_nombre(f)
        if desc_limpia.startswith("ensayo"):
            nombre_final = f"{sujeto}_{desc_limpia}_{time_str}"
        else:
            nombre_final = f"{sujeto}_{desc_limpia}"

        p_destino_fecha = os.path.join(destino_pca, date_str)
        os.makedirs(p_destino_fecha, exist_ok=True)
        p_destino_final = os.path.join(p_destino_fecha, nombre_final)

        # Si ya existe, añadir sufijo para no sobreescribir
        idx = 1
        ruta_definitiva = p_destino_final
        while os.path.exists(ruta_definitiva):
            ruta_definitiva = f"{p_destino_final}_{idx}"
            idx += 1

        shutil.move(p_origen, ruta_definitiva)

    print("[PCA] Reorganizacion de PCA/UMAP finalizada con exito.")

def organizar_autoencoders(base_emg):
    dir_ae = os.path.join(base_emg, "resultados", "resultados_autoencoder")
    if not os.path.exists(dir_ae):
        print(f"[AE] No existe: {dir_ae}")
        return

    # 1. Crear carpetas estándar
    dir_proc_temp = os.path.join(dir_ae, "procesamientos_temporales")
    dir_modelos = os.path.join(dir_ae, "modelos_entrenados")
    dir_figuras = os.path.join(dir_ae, "figuras_evaluacion")
    dir_cache = os.path.join(dir_ae, "cache_datasets")
    os.makedirs(dir_proc_temp, exist_ok=True)
    os.makedirs(dir_modelos, exist_ok=True)
    os.makedirs(dir_figuras, exist_ok=True)
    os.makedirs(dir_cache, exist_ok=True)

    # 2. Normalizar carpetas de sesiones con nombres no canónicos
    renombres_sesiones = {
        "Cande_2026-08-28_Prueba4_Prueba5": "2026-08-28_Candela_Prueba4_Prueba5",
        "Cande_Frankenstein_2026-09-04": "2026-09-04_Candela_Frankenstein",
        "Candela_Continuo_SinColapso_2026-09-01": "2026-09-01_Candela_Continuo",
        "Candela_DerivadaContinua_2026-09-01": "2026-09-01_Candela_DerivadaContinua",
        "Petra_Silicona_med1_med2": "2026-09-14_Petra_Silicona"
    }
    for old_n, new_n in renombres_sesiones.items():
        p_old = os.path.join(dir_ae, old_n)
        p_new = os.path.join(dir_ae, new_n)
        if os.path.exists(p_old):
            shutil.move(p_old, p_new)
            print(f"[AE] Sesion renombrada: {old_n} -> {new_n}")

    # 3. Mover procesamientos temporales a subcarpetas por fecha
    for item in os.listdir(dir_ae):
        p_item = os.path.join(dir_ae, item)
        if os.path.isdir(p_item) and item.startswith("procesamiento_"):
            # Extraer fecha YYYY-MM-DD
            m = re.search(r"procesamiento_(\d{4}-\d{2}-\d{2})", item)
            fecha = m.group(1) if m else "sin_fecha"
            dest_fecha = os.path.join(dir_proc_temp, fecha)
            os.makedirs(dest_fecha, exist_ok=True)
            shutil.move(p_item, os.path.join(dest_fecha, item))

    # 4. Mover archivos sueltos (.pth, .png, .json)
    for f in os.listdir(dir_ae):
        p_f = os.path.join(dir_ae, f)
        if os.path.isfile(p_f):
            if f.endswith(".pth"):
                shutil.move(p_f, os.path.join(dir_modelos, f))
            elif f.endswith(".png") or f.endswith(".json"):
                shutil.move(p_f, os.path.join(dir_figuras, f))

    print("[AE] Reorganizacion de Autoencoders completada con exito.")

def organizar_imagenes_sueltas(base_emg):
    dir_resultados = os.path.join(base_emg, "resultados")
    dir_anatomicas = os.path.join(dir_resultados, "figuras_anatomicas_y_mapas")
    dir_diarias = os.path.join(dir_resultados, "figuras_comparativas_diarias")
    os.makedirs(dir_anatomicas, exist_ok=True)
    os.makedirs(dir_diarias, exist_ok=True)

    # 1. Imagenes sueltas en EMG_desarrollo/resultados/
    imagenes_res = glob.glob(os.path.join(dir_resultados, "*.png"))
    for img in imagenes_res:
        nombre = os.path.basename(img)
        shutil.move(img, os.path.join(dir_anatomicas, nombre))

    # 2. Imagenes sueltas en EMG_desarrollo/
    imagenes_emg = glob.glob(os.path.join(base_emg, "*.png"))
    for img in imagenes_emg:
        nombre = os.path.basename(img)
        # Preservar el logo oficial
        if nombre == "logo_nandu_lsd.png":
            continue
        shutil.move(img, os.path.join(dir_diarias, nombre))

    print("[IMG] Imagenes sueltas organizadas en directorios dedicados.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"Iniciando organizacion de resultados en: {base_dir}")
    organizar_pca_umap(base_dir)
    organizar_autoencoders(base_dir)
    organizar_imagenes_sueltas(base_dir)
    print("Organizacion completa.")
