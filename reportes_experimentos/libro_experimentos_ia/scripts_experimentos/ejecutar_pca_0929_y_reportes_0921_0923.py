#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paso 2: Ejecuta la cola cola_tareas_0921_0923.json para generar:
- Reporte_EMG_2026-09-21.pdf (PruebaS1..S4, Masetero + Orbi vertical + Orbi Horizontal)
  agregando la fotografía del montaje desde la presentación.
- Reporte_EMG_2026-09-23.pdf (Pruebat1 y Prueba2..5, Masetero + Orbi).
"""

import os
import sys
import shutil
import subprocess
from PIL import Image

REPO_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
EMG_DIR = os.path.join(REPO_ROOT, "EMG_desarrollo")
REPORTES_DIR = os.path.join(REPO_ROOT, "reportes_experimentos")

for p in [REPO_ROOT, EMG_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.report_queue import ReportQueueManager


def recortar_foto_presentacion_0921():
    panel_path = os.path.join(REPORTES_DIR, "presentacion", "imagenes", "masetero_panel_completo.png")
    out_crop = os.path.join(REPORTES_DIR, "foto_montaje_2026_09_21_s1_s4.png")
    if os.path.exists(panel_path) and not os.path.exists(out_crop):
        im = Image.open(panel_path)
        w, h = im.size
        crop = im.crop((int(w * 0.49), int(h * 0.04), int(w * 0.99), int(h * 0.48)))
        crop.save(out_crop)
        print(f"[OK] Foto del montaje 2026-09-21 recortada en: {out_crop}")
    return out_crop


def paso_2_ejecutar_cola_0921_0923():
    print("\n=======================================================")
    print("PASO 2: Ejecutando cola de reportes 2026-09-21 y 2026-09-23")
    print("=======================================================")
    foto_crop = recortar_foto_presentacion_0921()

    cola_json = os.path.join(REPORTES_DIR, "cola_tareas_0921_0923.json")
    mgr = ReportQueueManager(output_dir=REPORTES_DIR)
    mgr.load_from_json(cola_json)
    mgr.execute_queue(logger=lambda msg: print(msg, flush=True))

    # El generador con report_tag="2026_09_21" crea Reporte_EMG_2026-09-21_2026_09_21.tex
    for fecha_str, tag_str in [("2026-09-21", "2026_09_21"), ("2026-09-23", "2026_09_23")]:
        src_tex = os.path.join(REPORTES_DIR, f"Reporte_EMG_{fecha_str}_{tag_str}.tex")
        dst_tex = os.path.join(REPORTES_DIR, f"Reporte_EMG_{fecha_str}.tex")
        src_pdf = os.path.join(REPORTES_DIR, f"Reporte_EMG_{fecha_str}_{tag_str}.pdf")
        dst_pdf = os.path.join(REPORTES_DIR, f"Reporte_EMG_{fecha_str}.pdf")
        if os.path.exists(src_tex):
            shutil.copy2(src_tex, dst_tex)
        if os.path.exists(src_pdf):
            shutil.copy2(src_pdf, dst_pdf)

    tex_0921 = os.path.join(REPORTES_DIR, "Reporte_EMG_2026-09-21.tex")
    if os.path.exists(tex_0921) and os.path.exists(foto_crop):
        with open(tex_0921, "r", encoding="utf-8") as f:
            c21 = f.read()
        if "foto_montaje_2026_09_21_s1_s4.png" not in c21:
            target = "\\newpage\n\\tableofcontents"
            bloque_foto = f"""
\\subsection{{Registro Fotográfico del Montaje Experimental: Serie S1 a S4}}
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.78\\textwidth]{{{foto_crop.replace(chr(92), '/')}}}
\\caption{{Montaje experimental correspondiente a la serie S1 a S4 (Masetero en Canal 0, Orbi vertical en Canal 1 y Orbi Horizontal en Canal 2), extraído de la presentación de avances.}}
\\end{{figure}}
"""
            if target in c21:
                c21 = c21.replace(target, bloque_foto + "\n" + target, 1)
                with open(tex_0921, "w", encoding="utf-8") as f:
                    f.write(c21)
                for _ in range(2):
                    subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "Reporte_EMG_2026-09-21.tex"],
                        cwd=REPORTES_DIR,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                print("[OK] Fotografía agregada y Reporte_EMG_2026-09-21.pdf recompilado.")


if __name__ == "__main__":
    paso_2_ejecutar_cola_0921_0923()
