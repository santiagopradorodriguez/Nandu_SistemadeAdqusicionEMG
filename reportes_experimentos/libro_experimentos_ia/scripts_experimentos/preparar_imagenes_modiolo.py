"""
Prepara y optimiza las figuras de la exploración de Modiolo para la presentación de Beamer.
Copia las imágenes de control experimental y genera un panel compuesto con el montaje de electrodos.
"""
import os
import shutil
from PIL import Image

SRC_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/historicos_modiolo/imagenes_2026-09-25"
DST_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/presentacion/imagenes"
os.makedirs(DST_DIR, exist_ok=True)

# 1. Copiar imágenes clave
shutil.copy(os.path.join(SRC_DIR, "pca_3d_aciertos_errores_modiolo.png"),
            os.path.join(DST_DIR, "pca_3d_aciertos_errores_modiolo.png"))
shutil.copy(os.path.join(SRC_DIR, "pca_2d_scatter_modiolo.png"),
            os.path.join(DST_DIR, "pca_2d_scatter_modiolo.png"))
shutil.copy(os.path.join(SRC_DIR, "10_masetero_config_modiolo.png"),
            os.path.join(DST_DIR, "modiolo_foto_perfil.png"))
shutil.copy(os.path.join(SRC_DIR, "11_orbicular_config_modiolo.png"),
            os.path.join(DST_DIR, "modiolo_foto_frontal.png"))

# 2. Generar panel combinado de las dos fotos de electrodos (perfil + ángulo anterior)
im1 = Image.open(os.path.join(SRC_DIR, "10_masetero_config_modiolo.png"))
im2 = Image.open(os.path.join(SRC_DIR, "11_orbicular_config_modiolo.png"))

# Tienen la misma altura (397 px)
h = im1.height
w_total = im1.width + im2.width + 6 # 6 px de separación
composite = Image.new("RGB", (w_total, h), color=(255, 255, 255))
composite.paste(im1, (0, 0))
composite.paste(im2, (im1.width + 6, 0))

comp_path = os.path.join(DST_DIR, "modiolo_montaje_electrodos_panel.png")
composite.save(comp_path, dpi=(300, 300))
print(f"Panel de fotos de electrodos generado en: {comp_path} ({composite.size})")
