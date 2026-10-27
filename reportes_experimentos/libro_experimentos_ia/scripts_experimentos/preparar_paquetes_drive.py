#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empaqueta las sesiones y reportes de investigación en una estructura
limpia y organizada en 'reportes_experimentos/para_subir_al_drive/',
utilizando enlaces duros (hardlinks) para no duplicar espacio en disco.
"""

import os
import sys
import shutil
import subprocess

REPO_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
BD_DIR = os.path.join(REPO_ROOT, "EMG_desarrollo", "base_de_datos_electrodos")
REPORTES_DIR = os.path.join(REPO_ROOT, "reportes_experimentos")
DEST_ROOT = os.path.join(REPORTES_DIR, "para_subir_al_drive")

def link_directory(src_dir, dst_dir):
    """Crea una estructura espejo en dst_dir usando cp -al para enlazar archivos por hardlinks."""
    os.makedirs(dst_dir, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target_dir = os.path.join(dst_dir, rel) if rel != "." else dst_dir
        os.makedirs(target_dir, exist_ok=True)
        for f in files:
            src_f = os.path.join(root, f)
            dst_f = os.path.join(target_dir, f)
            if not os.path.exists(dst_f):
                try:
                    os.link(src_f, dst_f)
                except Exception:
                    shutil.copy2(src_f, dst_f)

def armar_paquetes():
    os.makedirs(DEST_ROOT, exist_ok=True)
    print(f"Directorio de destino: {DEST_ROOT}")

    # =========================================================================
    # Paquete 1: 2026-09-23 Configuración 1 (Masetero y Orbiculares, Serie S1..S4)
    # =========================================================================
    p1_dir = os.path.join(DEST_ROOT, "2026-09-23_Config1_Masetero_y_Orbiculares")
    os.makedirs(p1_dir, exist_ok=True)
    pdf_p1 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-23_Config1_Masetero_Orbiculares.pdf")
    if os.path.exists(pdf_p1):
        shutil.copy2(pdf_p1, os.path.join(p1_dir, os.path.basename(pdf_p1)))
    foto_p1 = os.path.join(REPORTES_DIR, "foto_montaje_2026_09_21_s1_s4.png")
    if os.path.exists(foto_p1):
        shutil.copy2(foto_p1, os.path.join(p1_dir, os.path.basename(foto_p1)))

    med_p1 = os.path.join(p1_dir, "mediciones")
    os.makedirs(med_p1, exist_ok=True)
    src_0921 = os.path.join(BD_DIR, "2026-09-21")
    for toma in sorted(os.listdir(src_0921)):
        toma_src = os.path.join(src_0921, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            link_directory(toma_src, os.path.join(med_p1, toma))
    print("[OK] Paquete 1: 2026-09-23 Config1 armado.")

    # =========================================================================
    # Paquete 2: 2026-09-23 Configuración 2 (Masetero y Orbicularis, Pruebat1 y Prueba2..5)
    # =========================================================================
    p2_dir = os.path.join(DEST_ROOT, "2026-09-23_Config2_Masetero_y_Orbicularis")
    os.makedirs(p2_dir, exist_ok=True)
    pdf_p2 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-23_Config2_Masetero_Orbi.pdf")
    if os.path.exists(pdf_p2):
        shutil.copy2(pdf_p2, os.path.join(p2_dir, os.path.basename(pdf_p2)))

    med_p2 = os.path.join(p2_dir, "mediciones")
    os.makedirs(med_p2, exist_ok=True)
    src_0923 = os.path.join(BD_DIR, "2026-09-23")
    for toma in sorted(os.listdir(src_0923)):
        toma_src = os.path.join(src_0923, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            link_directory(toma_src, os.path.join(med_p2, toma))
    print("[OK] Paquete 2: 2026-09-23 Config2 armado.")

    # =========================================================================
    # Paquete 3: 2026-09-29 (Modiolo y Orbicularis, Sujeto Candela)
    # =========================================================================
    p3_dir = os.path.join(DEST_ROOT, "2026-09-29_Modiolo_y_Orbicularis_Candela")
    os.makedirs(p3_dir, exist_ok=True)
    pdf_p3 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-29.pdf")
    if os.path.exists(pdf_p3):
        shutil.copy2(pdf_p3, os.path.join(p3_dir, os.path.basename(pdf_p3)))

    med_p3 = os.path.join(p3_dir, "mediciones")
    os.makedirs(med_p3, exist_ok=True)
    src_0929 = os.path.join(BD_DIR, "2026-09-29")
    for toma in sorted(os.listdir(src_0929)):
        toma_src = os.path.join(src_0929, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            if "_Candela" in toma:
                link_directory(toma_src, os.path.join(med_p3, toma))
    print("[OK] Paquete 3: 2026-09-29 armado.")

    # =========================================================================
    # 2026-10-23: Separación en Exp 1, Exp 2 y Exp 3
    # =========================================================================
    src_1023 = os.path.join(BD_DIR, "2026-10-23")
    fotos_1023 = os.path.join(src_1023, "fotos_montaje")

    # Paquete 4: Exp 1 Masetero Monocanal
    p4_dir = os.path.join(DEST_ROOT, "2026-10-23_Exp1_Masetero_Monocanal")
    os.makedirs(p4_dir, exist_ok=True)
    pdf_p4 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp1_Masetero_Monocanal.pdf")
    if os.path.exists(pdf_p4):
        shutil.copy2(pdf_p4, os.path.join(p4_dir, os.path.basename(pdf_p4)))
    txt_p4 = os.path.join(src_1023, "registro_orden_autograbado_discreto_Prueba1_Sujeto1.txt")
    if os.path.exists(txt_p4):
        shutil.copy2(txt_p4, os.path.join(p4_dir, os.path.basename(txt_p4)))
    if os.path.isdir(fotos_1023):
        link_directory(fotos_1023, os.path.join(p4_dir, "fotos_montaje"))

    med_p4 = os.path.join(p4_dir, "mediciones")
    os.makedirs(med_p4, exist_ok=True)
    for toma in sorted(os.listdir(src_1023)):
        toma_src = os.path.join(src_1023, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            # Exp 1 tiene formato X_PruebaN_Sujeto1 (donde N es 1..5)
            partes = toma.split("_")
            if len(partes) >= 3 and partes[1].startswith("Prueba") and not "tricanal" in toma.lower() and not "belly" in toma.lower() and not "silaba" in toma.lower():
                link_directory(toma_src, os.path.join(med_p4, toma))
    print("[OK] Paquete 4: 2026-10-23 Exp1 armado.")

    # Paquete 5: Exp 2 Belly y Orbicularis
    p5_dir = os.path.join(DEST_ROOT, "2026-10-23_Exp2_Belly_y_Orbicularis")
    os.makedirs(p5_dir, exist_ok=True)
    pdf_p5 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp2_Belly_Orbi_x25.pdf")
    if os.path.exists(pdf_p5):
        shutil.copy2(pdf_p5, os.path.join(p5_dir, os.path.basename(pdf_p5)))
    txt_p5 = os.path.join(src_1023, "registro_orden_autograbado_discreto_pruebabelly_Sujeto1.txt")
    if os.path.exists(txt_p5):
        shutil.copy2(txt_p5, os.path.join(p5_dir, os.path.basename(txt_p5)))
    if os.path.isdir(fotos_1023):
        link_directory(fotos_1023, os.path.join(p5_dir, "fotos_montaje"))

    med_p5 = os.path.join(p5_dir, "mediciones")
    os.makedirs(med_p5, exist_ok=True)
    for toma in sorted(os.listdir(src_1023)):
        toma_src = os.path.join(src_1023, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            if "pruebabelly" in toma.lower():
                link_directory(toma_src, os.path.join(med_p5, toma))
    print("[OK] Paquete 5: 2026-10-23 Exp2 armado.")

    # Paquete 6: Exp 3 Tríada Masetero, Modíolo y Orbicularis
    p6_dir = os.path.join(DEST_ROOT, "2026-10-23_Exp3_Triada_Masetero_Modiolo_Orbicularis")
    os.makedirs(p6_dir, exist_ok=True)
    pdf_p6 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-10-23_Exp3_Triada_Masetero_x25.pdf")
    if os.path.exists(pdf_p6):
        shutil.copy2(pdf_p6, os.path.join(p6_dir, os.path.basename(pdf_p6)))
    txt_p6 = os.path.join(src_1023, "registro_orden_autograbado_discreto_Pruebatricanal_Sujeto1.txt")
    if os.path.exists(txt_p6):
        shutil.copy2(txt_p6, os.path.join(p6_dir, os.path.basename(txt_p6)))
    if os.path.isdir(fotos_1023):
        link_directory(fotos_1023, os.path.join(p6_dir, "fotos_montaje"))

    med_p6 = os.path.join(p6_dir, "mediciones")
    os.makedirs(med_p6, exist_ok=True)
    for toma in sorted(os.listdir(src_1023)):
        toma_src = os.path.join(src_1023, toma)
        if os.path.isdir(toma_src) and os.path.isdir(os.path.join(toma_src, "canal_0")):
            if "pruebatricanal" in toma.lower():
                link_directory(toma_src, os.path.join(med_p6, toma))
    print("[OK] Paquete 6: 2026-10-23 Exp3 armado.")

    print("\nTodos los paquetes fueron creados exitosamente en:")
    print(DEST_ROOT)

if __name__ == "__main__":
    armar_paquetes()
