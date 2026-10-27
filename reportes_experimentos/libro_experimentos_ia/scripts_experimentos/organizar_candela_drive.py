#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script maestro de procesamiento y reorganización de la carpeta Candela para Drive.
1. Descomprime Candela-20261024T205221Z-1-001/002/003.zip.
2. Lee las primeras páginas de cada reporte para extraer configuración y parámetros.
3. Extrae las imágenes de los reportes PDF.
4. Normaliza los nombres de carpetas de experimentos y tomas de sEMG a formato canónico con 'Candela'.
5. Genera la estructura final lista para subir a Google Drive en 'Candela_Organizado_Drive/'.
"""

import os
import sys
import json
import shutil
import glob
import subprocess
import zipfile
from collections import OrderedDict

REPO_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
BD_DIR = os.path.join(REPO_ROOT, "EMG_desarrollo", "base_de_datos_electrodos")
REPORTES_DIR = os.path.join(REPO_ROOT, "reportes_experimentos")
DOWNLOADS_DIR = os.path.expanduser("~/Descargas")

STAGING_DIR = os.path.join(REPO_ROOT, "candela_staging")
OUTPUT_DRIVE_DIR = os.path.join(REPO_ROOT, "Candela_Organizado_Drive")

def log(msg):
    print(f"[{os.path.basename(__file__)}] {msg}", flush=True)

def paso1_descomprimir():
    log("Iniciando Paso 1: Descompresión de archivos zip de Candela...")
    zips = sorted(glob.glob(os.path.join(DOWNLOADS_DIR, "Candela-20261024T205221Z-*.zip")))
    if not zips:
        raise FileNotFoundError(f"No se encontraron archivos zip de Candela en {DOWNLOADS_DIR}")
    
    os.makedirs(STAGING_DIR, exist_ok=True)
    for z in zips:
        log(f"Extrayendo: {os.path.basename(z)} ({os.path.getsize(z)/(1024**2):.1f} MB)...")
        with zipfile.ZipFile(z, 'r') as zf:
            zf.extractall(STAGING_DIR)
        log(f"Extracción de {os.path.basename(z)} completada.")
    log("Paso 1 completado exitosamente.")

def extraer_texto_primeras_paginas(pdf_path, num_paginas=3):
    """Extrae texto de las primeras páginas del PDF usando pdftotext."""
    if not os.path.exists(pdf_path):
        return ""
    try:
        res = subprocess.run(
            ["pdftotext", "-f", "1", "-l", str(num_paginas), pdf_path, "-"],
            capture_output=True, text=True, check=True
        )
        return res.stdout
    except Exception as e:
        log(f"Aviso: error extrayendo texto de {pdf_path}: {e}")
        return ""

def extraer_imagenes_pdf(pdf_path, out_dir, max_paginas=4):
    """Extrae imágenes incrustadas de las primeras páginas del PDF."""
    if not os.path.exists(pdf_path):
        return []
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, "fig_reporte")
    try:
        subprocess.run(
            ["pdfimages", "-png", "-f", "1", "-l", str(max_paginas), pdf_path, prefix],
            capture_output=True, check=True
        )
        imgs = sorted(glob.glob(os.path.join(out_dir, "fig_reporte-*.png")))
        # Filtrar imágenes minúsculas (< 2 KB) que suelen ser íconos o líneas
        valid_imgs = []
        for im in imgs:
            if os.path.getsize(im) > 4096:
                valid_imgs.append(im)
            else:
                try:
                    os.remove(im)
                except OSError:
                    pass
        return valid_imgs
    except Exception as e:
        log(f"Aviso: error extrayendo imágenes de {pdf_path}: {e}")
        return []

def asegurar_metadata_candela(toma_path):
    """Asegura que el campo 'subject' en metadata.json sea 'Candela'."""
    meta_path = os.path.join(toma_path, "canal_0", "metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cambiado = False
            if data.get("subject") != "Candela":
                data["subject"] = "Candela"
                cambiado = True
            if data.get("sujeto") != "Candela":
                data["sujeto"] = "Candela"
                cambiado = True
            if cambiado:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            log(f"Aviso actualizando metadata en {meta_path}: {e}")

def link_o_copia_toma(src_toma, dst_toma):
    """Copia o enlaza una toma sEMG preservando su estructura."""
    os.makedirs(dst_toma, exist_ok=True)
    for root, dirs, files in os.walk(src_toma):
        rel = os.path.relpath(root, src_toma)
        target_dir = os.path.join(dst_toma, rel) if rel != "." else dst_toma
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            src_f = os.path.join(root, f)
            dst_f = os.path.join(target_dir, f)
            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)
                except Exception:
                    shutil.copy2(src_f, dst_f)
    asegurar_metadata_candela(dst_toma)

def normalizar_nombre_toma(toma_name):
    """
    Convierte nombres de toma al estándar: <Vocal>_<Prueba>_Candela
    Ejemplos:
    - a_pruebadigastrico2_Sujeto1 -> A_PruebaDigastrico2_Candela
    - A_PruebaS1_Sujeto1 -> A_PruebaS1_Candela
    - A_Pruebat1_Sujeto1 -> A_Pruebat1_Candela
    - A_pruebabelly_Sujeto1 -> A_PruebaBelly1_Candela
    - A_pruebabelly_2_Sujeto1 -> A_PruebaBelly2_Candela
    - A_Pruebatricanal_Sujeto1 -> A_PruebaTriada1_Candela
    - A_Pruebatricanal_2_Sujeto1 -> A_PruebaTriada2_Candela
    - A_Prueba1_cande -> A_Prueba1_Candela
    - A_Prueba1_Candela -> A_Prueba1_Candela
    """
    t = toma_name.strip()
    partes = t.split("_")
    if not partes:
        return t

    vocal = partes[0].upper()
    if vocal not in ["A", "E", "I", "O", "U", "BA", "DA", "JA", "KA", "LA", "MA", "PA", "RA", "SA", "TA"]:
        vocal = partes[0].capitalize()

    resto = "_".join(partes[1:])
    # Quitar cualquier sufijo de sujeto previo
    for suf in ["_Sujeto1", "_sujeto1", "_cande", "_Cande", "_Candela", "_candela"]:
        if resto.endswith(suf):
            resto = resto[:-len(suf)]

    # Limpiar y capitalizar nombre de prueba
    if resto.lower() == "pruebabelly":
        prueba = "PruebaBelly1"
    elif resto.lower() == "pruebabelly_2":
        prueba = "PruebaBelly2"
    elif resto.lower() == "pruebatricanal":
        prueba = "PruebaTriada1"
    elif resto.lower().startswith("pruebatricanal_"):
        num = resto.split("_")[-1]
        prueba = f"PruebaTriada{num}"
    elif resto.lower().startswith("pruebadigastrico"):
        prueba = "Prueba" + resto[len("prueba"):].capitalize()
        # capitalizar digastrico
        prueba = prueba.replace("digastrico", "Digastrico").replace("Digastrico2_2", "Digastrico2_Rep2")
    elif resto.lower().startswith("pruebamentoniano"):
        prueba = "Prueba" + resto[len("prueba"):].capitalize()
        prueba = prueba.replace("mentoniano", "Mentoniano")
    elif resto.lower().startswith("pruebamodiolo"):
        prueba = "Prueba" + resto[len("prueba"):].capitalize()
        prueba = prueba.replace("modiolo", "Modiolo")
    elif resto.startswith("Prueba") or resto.startswith("Serie"):
        prueba = resto
    else:
        prueba = resto.capitalize()

    return f"{vocal}_{prueba}_Candela"

def organizar_experimentos():
    candela_raw = os.path.join(STAGING_DIR, "Candela")
    if not os.path.isdir(candela_raw):
        raise FileNotFoundError(f"No se encontró el directorio {candela_raw}")

    os.makedirs(OUTPUT_DRIVE_DIR, exist_ok=True)
    log(f"Organizando experimentos en: {OUTPUT_DRIVE_DIR}")

    resumen_experimentos = []

    # =========================================================================
    # 1. 2026-09-01: Digástrico, Zona Risorio y Orbicularis
    # =========================================================================
    exp1_src = os.path.join(candela_raw, "Experimento_1 de Septiembre_Digastrico_Zona Risorio_Orbicularis")
    exp1_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-01_Digastrico_Risorio_Orbicularis_Candela")
    if os.path.isdir(exp1_src):
        os.makedirs(exp1_dst, exist_ok=True)
        # Copiar PDF
        pdf_1 = os.path.join(exp1_src, "Reporte_EMG_2026-09-01.pdf")
        if os.path.exists(pdf_1):
            shutil.copy2(pdf_1, os.path.join(exp1_dst, os.path.basename(pdf_1)))
            extraer_imagenes_pdf(pdf_1, os.path.join(exp1_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_1, 2)
            with open(os.path.join(exp1_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        # Tomas
        med_dst = os.path.join(exp1_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        med_src = os.path.join(exp1_src, "Mediciones")
        if os.path.isdir(med_src):
            for t in sorted(os.listdir(med_src)):
                p = os.path.join(med_src, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-01_Digastrico_Risorio_Orbicularis_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-01",
            "carpeta": "2026-09-01_Digastrico_Risorio_Orbicularis_Candela",
            "montaje": "Ch0: Digástrico (Anterior Belly), Ch1: Zona Risorio, Ch2: Orbicularis Oris",
            "reportes": ["Reporte_EMG_2026-09-01.pdf"]
        })

    # =========================================================================
    # 2. 2026-09-15: Digástrico, Cigomático y Orbicularis
    # =========================================================================
    exp15_src = os.path.join(candela_raw, "Experimento_15 de Septiembre_Digastrico_Zigomatico_Orbicularis")
    exp15_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela")
    if os.path.isdir(exp15_src):
        os.makedirs(exp15_dst, exist_ok=True)
        for pdf_n in ["Reporte_de_Experimento_2026-09-15.pdf", "Reporte_SNR_Y_Calidad_2026-09-15.pdf"]:
            pdf_p = os.path.join(exp15_src, pdf_n)
            if os.path.exists(pdf_p):
                shutil.copy2(pdf_p, os.path.join(exp15_dst, pdf_n))
        pdf_main = os.path.join(exp15_dst, "Reporte_de_Experimento_2026-09-15.pdf")
        if os.path.exists(pdf_main):
            extraer_imagenes_pdf(pdf_main, os.path.join(exp15_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_main, 2)
            with open(os.path.join(exp15_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        med_dst = os.path.join(exp15_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        med_src = os.path.join(exp15_src, "Mediciones")
        if os.path.isdir(med_src):
            for t in sorted(os.listdir(med_src)):
                p = os.path.join(med_src, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-15",
            "carpeta": "2026-09-15_Digastrico_Cigomatico_Orbicularis_Candela",
            "montaje": "Ch0: Anterior Belly, Ch1: Zygomaticus Major, Ch2: Orbicularis Oris",
            "reportes": ["Reporte_de_Experimento_2026-09-15.pdf", "Reporte_SNR_Y_Calidad_2026-09-15.pdf"]
        })

    # =========================================================================
    # 3. 2026-09-18: Digástrico, Modíolo y Orbicularis
    # =========================================================================
    exp18_src = os.path.join(candela_raw, "Experimento 18 de Septiembre")
    exp18_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-18_Digastrico_Modiolo_Orbicularis_Candela")
    if os.path.isdir(exp18_src):
        os.makedirs(exp18_dst, exist_ok=True)
        pdf_18 = os.path.join(exp18_src, "Reporte_EMG_2026-09-18.pdf")
        if os.path.exists(pdf_18):
            shutil.copy2(pdf_18, os.path.join(exp18_dst, os.path.basename(pdf_18)))
            extraer_imagenes_pdf(pdf_18, os.path.join(exp18_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_18, 2)
            with open(os.path.join(exp18_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        # Buscar tomas en '2026-09-18' o 'Mediciones'
        med_dst = os.path.join(exp18_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        tomas_src = os.path.join(exp18_src, "2026-09-18")
        if not os.path.isdir(tomas_src):
            tomas_src = os.path.join(exp18_src, "Mediciones")
        if os.path.isdir(tomas_src):
            for t in sorted(os.listdir(tomas_src)):
                p = os.path.join(tomas_src, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-18_Digastrico_Modiolo_Orbicularis_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-18",
            "carpeta": "2026-09-18_Digastrico_Modiolo_Orbicularis_Candela",
            "montaje": "Ch0: Anterior Belly, Ch1: Modiolo, Ch2: Orbicularis Oris",
            "reportes": ["Reporte_EMG_2026-09-18.pdf"]
        })

    # =========================================================================
    # 4. 2026-09-23 Configuración 1: Masetero y Orbiculares (Serie S1..S4)
    # =========================================================================
    c1_src = os.path.join(candela_raw, "2026-09-23_Config1_Masetero_y_Orbiculares")
    if not os.path.isdir(c1_src):
        c1_src = os.path.join(REPORTES_DIR, "para_subir_al_drive", "2026-09-23_Config1_Masetero_y_Orbiculares")
    c1_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-23_Config1_Masetero_y_Orbiculares_Candela")
    if os.path.isdir(c1_src):
        os.makedirs(c1_dst, exist_ok=True)
        pdf_c1 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-23_Config1_Masetero_Orbiculares.pdf")
        if os.path.exists(pdf_c1):
            shutil.copy2(pdf_c1, os.path.join(c1_dst, os.path.basename(pdf_c1)))
            extraer_imagenes_pdf(pdf_c1, os.path.join(c1_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_c1, 2)
            with open(os.path.join(c1_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        foto_crop = os.path.join(REPORTES_DIR, "foto_montaje_2026_09_21_s1_s4.png")
        if os.path.exists(foto_crop):
            shutil.copy2(foto_crop, os.path.join(c1_dst, os.path.basename(foto_crop)))
        med_dst = os.path.join(c1_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        med_src = os.path.join(c1_src, "mediciones")
        if not os.path.isdir(med_src):
            med_src = os.path.join(BD_DIR, "2026-09-21")
        if os.path.isdir(med_src):
            for t in sorted(os.listdir(med_src)):
                p = os.path.join(med_src, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-23_Config1_Masetero_y_Orbiculares_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-23 (Tarde 2)",
            "carpeta": "2026-09-23_Config1_Masetero_y_Orbiculares_Candela",
            "montaje": "Ch0: Masetero (Posición 1 alta), Ch1: Orbi vertical, Ch2: Orbi horizontal",
            "reportes": ["Reporte_EMG_2026-09-23_Config1_Masetero_Orbiculares.pdf"]
        })

    # =========================================================================
    # 5. 2026-09-23 Configuración 2: Masetero y Orbicularis (Pruebat1 y Prueba2..5)
    # =========================================================================
    c2_src = os.path.join(candela_raw, "2026-09-23_Config2_Masetero_y_Orbicularis")
    if not os.path.isdir(c2_src):
        c2_src = os.path.join(REPORTES_DIR, "para_subir_al_drive", "2026-09-23_Config2_Masetero_y_Orbicularis")
    c2_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-23_Config2_Masetero_y_Orbicularis_Candela")
    if os.path.isdir(c2_src):
        os.makedirs(c2_dst, exist_ok=True)
        pdf_c2 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-23_Config2_Masetero_Orbi.pdf")
        if os.path.exists(pdf_c2):
            shutil.copy2(pdf_c2, os.path.join(c2_dst, os.path.basename(pdf_c2)))
            extraer_imagenes_pdf(pdf_c2, os.path.join(c2_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_c2, 2)
            with open(os.path.join(c2_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        med_dst = os.path.join(c2_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        med_src = os.path.join(c2_src, "mediciones")
        if not os.path.isdir(med_src):
            med_src = os.path.join(BD_DIR, "2026-09-23")
        if os.path.isdir(med_src):
            for t in sorted(os.listdir(med_src)):
                p = os.path.join(med_src, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-23_Config2_Masetero_y_Orbicularis_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-23 (Tarde 1)",
            "carpeta": "2026-09-23_Config2_Masetero_y_Orbicularis_Candela",
            "montaje": "Ch0: Masetero (Posición 2), Ch1: Inactivo, Ch2: Orbicularis Oris",
            "reportes": ["Reporte_EMG_2026-09-23_Config2_Masetero_Orbi.pdf"]
        })

    # =========================================================================
    # 6. 2026-09-25: Exploración Modiolo, Digástrico y Mentoniano
    # =========================================================================
    exp25_src = os.path.join(candela_raw, "Experimento 25 de Septiembre")
    exp25_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela")
    if os.path.isdir(exp25_src):
        os.makedirs(exp25_dst, exist_ok=True)
        pdf_25 = os.path.join(exp25_src, "Reporte_Control_Experimental_2026-09-25.pdf")
        if os.path.exists(pdf_25):
            shutil.copy2(pdf_25, os.path.join(exp25_dst, os.path.basename(pdf_25)))
            extraer_imagenes_pdf(pdf_25, os.path.join(exp25_dst, "fotos_montaje_y_figuras"))
            txt_desc = extraer_texto_primeras_paginas(pdf_25, 2)
            with open(os.path.join(exp25_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
                f.write(txt_desc)
        # Fotografías y figuras de alta resolución que ya estaban en el zip
        img_25 = os.path.join(exp25_src, "imagenes_2026-09-25")
        if os.path.isdir(img_25):
            dst_img = os.path.join(exp25_dst, "fotos_montaje_originales")
            os.makedirs(dst_img, exist_ok=True)
            for f_im in sorted(os.listdir(img_25)):
                shutil.copy2(os.path.join(img_25, f_im), os.path.join(dst_img, f_im))
        # Mediciones organizadas por subconfiguraciones
        med_dst = os.path.join(exp25_dst, "mediciones")
        os.makedirs(med_dst, exist_ok=True)
        for cfg in ["Config_Digastrico", "Config_Mentoniano", "Config_Modiolo", "Config_digastrico2"]:
            cfg_p = os.path.join(exp25_src, cfg)
            if os.path.isdir(cfg_p):
                for t in sorted(os.listdir(cfg_p)):
                    p = os.path.join(cfg_p, t)
                    if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                        link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
        log("[OK] 2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela procesado.")
        resumen_experimentos.append({
            "fecha": "2026-09-25",
            "carpeta": "2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela",
            "montaje": "Exploración de Modiolo, Mentoniano y Digástrico con electrodos recortados",
            "reportes": ["Reporte_Control_Experimental_2026-09-25.pdf"]
        })

    # =========================================================================
    # 7. 2026-09-29: Modiolo y Orbicularis (Candela)
    # =========================================================================
    exp29_src = os.path.join(candela_raw, "Experimento 29 de Septiembre")
    exp29_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-09-29_Modiolo_y_Orbicularis_Candela")
    os.makedirs(exp29_dst, exist_ok=True)
    # Usar el PDF actualizado con PCA
    pdf_29 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-29.pdf")
    if os.path.exists(pdf_29):
        shutil.copy2(pdf_29, os.path.join(exp29_dst, os.path.basename(pdf_29)))
        extraer_imagenes_pdf(pdf_29, os.path.join(exp29_dst, "fotos_montaje_y_figuras"))
        txt_desc = extraer_texto_primeras_paginas(pdf_29, 2)
        with open(os.path.join(exp29_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
            f.write(txt_desc)
    med_dst = os.path.join(exp29_dst, "mediciones")
    os.makedirs(med_dst, exist_ok=True)
    tomas_29_src = os.path.join(exp29_src, "2026-09-29") if os.path.isdir(os.path.join(exp29_src, "2026-09-29")) else os.path.join(BD_DIR, "2026-09-29")
    if os.path.isdir(tomas_29_src):
        for t in sorted(os.listdir(tomas_29_src)):
            p = os.path.join(tomas_29_src, t)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                if "_Candela" in t:
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
    log("[OK] 2026-09-29_Modiolo_y_Orbicularis_Candela procesado.")
    resumen_experimentos.append({
        "fecha": "2026-09-29",
        "carpeta": "2026-09-29_Modiolo_y_Orbicularis_Candela",
        "montaje": "Ch0: Orbicularis Oris, Ch1: Modiolo (Retracción lateral)",
        "reportes": ["Reporte_EMG_2026-09-29.pdf"]
    })

    # =========================================================================
    # 8. 2026-10-20: Control Experimental Multi-Sujeto (Candela)
    # =========================================================================
    exp20_src = os.path.join(candela_raw, "Experimento 20 de Octubre")
    exp20_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-10-20_Control_Experimental_Candela")
    os.makedirs(exp20_dst, exist_ok=True)
    for pdf_n in ["reporte_control_experimental_2026_10_20.pdf", "sintesis_experimento_2026_10_20.pdf"]:
        pdf_p = os.path.join(exp20_src, pdf_n) if os.path.exists(os.path.join(exp20_src, pdf_n)) else os.path.join(REPORTES_DIR, pdf_n)
        if os.path.exists(pdf_p):
            shutil.copy2(pdf_p, os.path.join(exp20_dst, pdf_n))
    pdf_main = os.path.join(exp20_dst, "reporte_control_experimental_2026_10_20.pdf")
    if os.path.exists(pdf_main):
        extraer_imagenes_pdf(pdf_main, os.path.join(exp20_dst, "fotos_montaje_y_figuras"))
        txt_desc = extraer_texto_primeras_paginas(pdf_main, 2)
        with open(os.path.join(exp20_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
            f.write(txt_desc)
    # Incorporar tomas de Candela desde base_de_datos_electrodos/2026-10-20
    med_dst = os.path.join(exp20_dst, "mediciones")
    os.makedirs(med_dst, exist_ok=True)
    src_1020 = os.path.join(BD_DIR, "2026-10-20")
    if os.path.isdir(src_1020):
        for t in sorted(os.listdir(src_1020)):
            if "Candela" in t:
                p = os.path.join(src_1020, t)
                if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                    link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
    log("[OK] 2026-10-20_Control_Experimental_Candela procesado.")
    resumen_experimentos.append({
        "fecha": "2026-10-20",
        "carpeta": "2026-10-20_Control_Experimental_Candela",
        "montaje": "Control Experimental Multi-Sujeto: Exp1 y Exp2 en Candela",
        "reportes": ["reporte_control_experimental_2026_10_20.pdf", "sintesis_experimento_2026_10_20.pdf"]
    })

    # =========================================================================
    # 9, 10, 11: 2026-10-23 (Exp1, Exp2 y Exp3)
    # =========================================================================
    src_1023_drive = os.path.join(REPORTES_DIR, "para_subir_al_drive")

    # Exp 1 Masetero Monocanal
    e1_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-10-23_Exp1_Masetero_Monocanal_Candela")
    os.makedirs(e1_dst, exist_ok=True)
    pdf_e1 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp1_Masetero_Monocanal.pdf")
    if os.path.exists(pdf_e1):
        shutil.copy2(pdf_e1, os.path.join(e1_dst, os.path.basename(pdf_e1)))
        extraer_imagenes_pdf(pdf_e1, os.path.join(e1_dst, "fotos_montaje_y_figuras"))
        txt_desc = extraer_texto_primeras_paginas(pdf_e1, 2)
        with open(os.path.join(e1_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
            f.write(txt_desc)
    fotos_1023_orig = os.path.join(BD_DIR, "2026-10-23", "fotos_montaje")
    if os.path.isdir(fotos_1023_orig):
        shutil.copytree(fotos_1023_orig, os.path.join(e1_dst, "fotos_montaje_originales"), dirs_exist_ok=True)
    txt_e1 = os.path.join(BD_DIR, "2026-10-23", "registro_orden_autograbado_discreto_Prueba1_Sujeto1.txt")
    if os.path.exists(txt_e1):
        shutil.copy2(txt_e1, os.path.join(e1_dst, os.path.basename(txt_e1)))
    med_dst = os.path.join(e1_dst, "mediciones")
    os.makedirs(med_dst, exist_ok=True)
    med_src = os.path.join(src_1023_drive, "2026-10-23_Exp1_Masetero_Monocanal", "mediciones")
    if os.path.isdir(med_src):
        for t in sorted(os.listdir(med_src)):
            p = os.path.join(med_src, t)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
    log("[OK] 2026-10-23_Exp1_Masetero_Monocanal_Candela procesado.")
    resumen_experimentos.append({
        "fecha": "2026-10-23",
        "carpeta": "2026-10-23_Exp1_Masetero_Monocanal_Candela",
        "montaje": "Ch0: Masetero (Monocanal, búsqueda de posición)",
        "reportes": ["Reporte_EMG_2026-10-23_Exp1_Masetero_Monocanal.pdf"]
    })

    # Exp 2 Belly y Orbicularis
    e2_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-10-23_Exp2_Belly_y_Orbicularis_Candela")
    os.makedirs(e2_dst, exist_ok=True)
    pdf_e2 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp2_Belly_Orbi_x25.pdf")
    if os.path.exists(pdf_e2):
        shutil.copy2(pdf_e2, os.path.join(e2_dst, os.path.basename(pdf_e2)))
        extraer_imagenes_pdf(pdf_e2, os.path.join(e2_dst, "fotos_montaje_y_figuras"))
        txt_desc = extraer_texto_primeras_paginas(pdf_e2, 2)
        with open(os.path.join(e2_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
            f.write(txt_desc)
    if os.path.isdir(fotos_1023_orig):
        shutil.copytree(fotos_1023_orig, os.path.join(e2_dst, "fotos_montaje_originales"), dirs_exist_ok=True)
    txt_e2 = os.path.join(BD_DIR, "2026-10-23", "registro_orden_autograbado_discreto_pruebabelly_Sujeto1.txt")
    if os.path.exists(txt_e2):
        shutil.copy2(txt_e2, os.path.join(e2_dst, os.path.basename(txt_e2)))
    med_dst = os.path.join(e2_dst, "mediciones")
    os.makedirs(med_dst, exist_ok=True)
    med_src = os.path.join(src_1023_drive, "2026-10-23_Exp2_Belly_y_Orbicularis", "mediciones")
    if os.path.isdir(med_src):
        for t in sorted(os.listdir(med_src)):
            p = os.path.join(med_src, t)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
    log("[OK] 2026-10-23_Exp2_Belly_y_Orbicularis_Candela procesado.")
    resumen_experimentos.append({
        "fecha": "2026-10-23",
        "carpeta": "2026-10-23_Exp2_Belly_y_Orbicularis_Candela",
        "montaje": "Ch0: Orbicularis Oris, Ch2: Anterior Belly (Digástrico x2.5)",
        "reportes": ["Reporte_EMG_2026-10-23_Exp2_Belly_Orbi_x25.pdf"]
    })

    # Exp 3 Tríada Masetero, Modíolo y Orbicularis
    e3_dst = os.path.join(OUTPUT_DRIVE_DIR, "2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela")
    os.makedirs(e3_dst, exist_ok=True)
    pdf_e3 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp3_Triada_Masetero_x25.pdf")
    if os.path.exists(pdf_e3):
        shutil.copy2(pdf_e3, os.path.join(e3_dst, os.path.basename(pdf_e3)))
        extraer_imagenes_pdf(pdf_e3, os.path.join(e3_dst, "fotos_montaje_y_figuras"))
        txt_desc = extraer_texto_primeras_paginas(pdf_e3, 2)
        with open(os.path.join(e3_dst, "DESCRIPCION_EXPERIMENTO.txt"), "w", encoding="utf-8") as f:
            f.write(txt_desc)
    if os.path.isdir(fotos_1023_orig):
        shutil.copytree(fotos_1023_orig, os.path.join(e3_dst, "fotos_montaje_originales"), dirs_exist_ok=True)
    txt_e3 = os.path.join(BD_DIR, "2026-10-23", "registro_orden_autograbado_discreto_Pruebatricanal_Sujeto1.txt")
    if os.path.exists(txt_e3):
        shutil.copy2(txt_e3, os.path.join(e3_dst, os.path.basename(txt_e3)))
    med_dst = os.path.join(e3_dst, "mediciones")
    os.makedirs(med_dst, exist_ok=True)
    med_src = os.path.join(src_1023_drive, "2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis", "mediciones")
    if os.path.isdir(med_src):
        for t in sorted(os.listdir(med_src)):
            p = os.path.join(med_src, t)
            if os.path.isdir(p) and os.path.isdir(os.path.join(p, "canal_0")):
                link_o_copia_toma(p, os.path.join(med_dst, normalizar_nombre_toma(t)))
    log("[OK] 2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela procesado.")
    resumen_experimentos.append({
        "fecha": "2026-10-23",
        "carpeta": "2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis_Candela",
        "montaje": "Ch0: Masetero (x2.5), Ch1: Modiolo, Ch2: Orbicularis Oris",
        "reportes": ["Reporte_EMG_2026-10-23_Exp3_Triada_Masetero_x25.pdf"]
    })

    # =========================================================================
    # Generar README Índice Maestro
    # =========================================================================
    readme_path = os.path.join(OUTPUT_DRIVE_DIR, "README_INDICE_CAMPANAS_CANDELA.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# Catálogo Oficial de Campañas Experimentales: Sujeto Candela\n\n")
        f.write("Estructura consolidada y organizada para Google Drive. Cada carpeta contiene el reporte oficial en PDF, sus registros fotográficos y la subcarpeta `mediciones/` con la nomenclatura canónica `<Vocal>_<Prueba>_Candela`.\n\n")
        f.write("| Fecha | Carpeta de Campaña | Montaje Muscular / Configuración | Reporte PDF |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for exp in resumen_experimentos:
            reps = ", ".join(exp["reportes"])
            f.write(f"| {exp['fecha']} | `{exp['carpeta']}` | {exp['montaje']} | {reps} |\n")
        f.write("\n")

    log(f"README generado en: {readme_path}")
    log(f"Proceso finalizado. Total de carpetas organizadas: {len(resumen_experimentos)}")

if __name__ == "__main__":
    paso1_descomprimir()
    organizar_experimentos()
