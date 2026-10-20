"""
Copia y acondiciona imágenes del experimento de Digástrico (2026-09-25) para Beamer:
- Foto del montaje en cuello con cable chico
- Gráfico de PCA 3D con aciertos y errores
"""
import os
import shutil
from PIL import Image

SRC_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/historicos_modiolo/imagenes_2026-09-25"
DST_DIR = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/presentacion/imagenes"
os.makedirs(DST_DIR, exist_ok=True)

# 1. Copiar imágenes
shutil.copy(os.path.join(SRC_DIR, "05_digastrico_cable_chico_electrodos.png"),
            os.path.join(DST_DIR, "digastrico_foto_cuello.png"))
shutil.copy(os.path.join(SRC_DIR, "pca_3d_aciertos_errores_digastrico_inicial.png"),
            os.path.join(DST_DIR, "pca_3d_aciertos_errores_digastrico_inicial.png"))
shutil.copy(os.path.join(SRC_DIR, "pca_2d_scatter_digastrico_inicial.png"),
            os.path.join(DST_DIR, "pca_2d_scatter_digastrico_inicial.png"))

print("Imágenes de digástrico copiadas exitosamente a:", DST_DIR)
