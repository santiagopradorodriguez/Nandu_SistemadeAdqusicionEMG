"""
Copia y acondiciona imágenes del experimento de Mentoniano (2026-09-27) para Beamer:
- Fotos del montaje frontal y desplazamiento 1.5 cm
- Gráfico de PCA 3D con aciertos y errores
"""
import os
import shutil
from PIL import Image

SRC_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/historicos_modiolo/imagenes_2026-09-25"
DST_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/presentacion/imagenes"
os.makedirs(DST_DIR, exist_ok=True)

# 1. Copiar imágenes
shutil.copy(os.path.join(SRC_DIR, "06_mentoniano_electrodos_chicos_frontal.png"),
            os.path.join(DST_DIR, "mentoniano_foto_frontal.png"))
shutil.copy(os.path.join(SRC_DIR, "pca_3d_aciertos_errores_mentoniano.png"),
            os.path.join(DST_DIR, "pca_3d_aciertos_errores_mentoniano.png"))
shutil.copy(os.path.join(SRC_DIR, "pca_2d_scatter_mentoniano.png"),
            os.path.join(DST_DIR, "pca_2d_scatter_mentoniano.png"))

# 2. Generar panel comparativo de posición inicial vs movido 1.5 cm
im_ini = Image.open(os.path.join(SRC_DIR, "08_mentoniano_posicion_inicial.png"))
im_mov = Image.open(os.path.join(SRC_DIR, "09_mentoniano_movido_1_5cm.png"))

# Combinar ambas lado a lado
w = im_ini.width + im_mov.width + 4
h = max(im_ini.height, im_mov.height)
comp = Image.new("RGB", (w, h), (255, 255, 255))
comp.paste(im_ini, (0, 0))
comp.paste(im_mov, (im_ini.width + 4, 0))
comp_path = os.path.join(DST_DIR, "mentoniano_ajuste_1_5cm_panel.png")
comp.save(comp_path, dpi=(300, 300))
print(f"Panel de desplazamiento mentoniano generado en: {comp_path} ({comp.size})")

# Verificar tamaños
for f in ["mentoniano_foto_frontal.png", "pca_3d_aciertos_errores_mentoniano.png"]:
    im = Image.open(os.path.join(DST_DIR, f))
    print(f"{f}: {im.size}")
