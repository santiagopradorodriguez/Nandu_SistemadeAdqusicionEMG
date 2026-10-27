"""
Generador del Atlas de Activación Muscular sEMG - Página Oficial por Sesión
Estructura rigurosa según especificación:
Día: Patrón de activación muscular (curvas de vocales)
Al lado: Las 3 fotos anatómicas curadas de los canales + foto de electrodos
Abajo: Posición y distancia predicha por IA (+- 3.5 mm)
Abajo izquierda: Imagen principal del PCA extraída del reporte oficial.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from PIL import Image

def generar_pagina_atlas_0901(output_pdf_path, output_png_path):
    fig = plt.figure(figsize=(16, 11), dpi=200, facecolor='#0f172a')
    
    # Grid principal: 2 columnas
    # Columna Izquierda: Activación temporal de vocales (arriba) + Proyección PCA Oficial (abajo)
    # Columna Derecha: Fotos anatómicas de los 3 canales + Foto electrodos + Métricas de distancia
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.1, 0.9], height_ratios=[1.0, 1.0], 
                           wspace=0.18, hspace=0.22, left=0.05, right=0.96, top=0.92, bottom=0.06)
    
    # Titulo General
    fig.suptitle("2026-09-01: Patrón de Activación Muscular sEMG - Candela", 
                 fontsize=20, fontweight='bold', color='#38bdf8', y=0.97)

    # ----------------------------------------------------
    # 1. ARRIBA IZQUIERDA: Curvas Promedio de Activación (Patrón Temporal)
    # ----------------------------------------------------
    # Usamos la imagen renderizada de patrones o graficamos directamente
    ax_patron = fig.add_subplot(gs[0, 0])
    ax_patron.set_facecolor('#1e293b')
    # Cargamos el patrón recortado del atlas previo
    im_patron = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_page-1.png")
    # Recortar solo el gráfico de curvas de atlas_page-1
    h_p, w_p, _ = im_patron.shape
    crop_p = im_patron[int(h_p*0.14):int(h_p*0.48), int(w_p*0.05):int(w_p*0.62)]
    ax_patron.imshow(crop_p)
    ax_patron.axis('off')
    ax_patron.set_title("Patrón Temporal Promedio por Vocal (/a/, /e/, /i/, /o/, /u/)", 
                        fontsize=12, fontweight='bold', color='#f8fafc', pad=8)

    # ----------------------------------------------------
    # 2. ABAJO IZQUIERDA: Proyección PCA Principal del Reporte
    # ----------------------------------------------------
    ax_pca = fig.add_subplot(gs[1, 0])
    ax_pca.set_facecolor('#1e293b')
    im_pca = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/pca_principal_fig38_2026_09_01.png")
    ax_pca.imshow(im_pca)
    ax_pca.axis('off')
    ax_pca.set_title("Proyección PCA Oficial de la Sesión: Tríada CH0, CH1, CH2", 
                     fontsize=12, fontweight='bold', color='#38bdf8', pad=8)

    # ----------------------------------------------------
    # 3. COLUMNA DERECHA: Las 3 fotos anatómicas curadas + Métricas
    # ----------------------------------------------------
    gs_derecha = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[:, 1], 
                                                  wspace=0.12, hspace=0.25)

    # Foto 1: CH0 Digástrico Submentoniana
    ax_f1 = fig.add_subplot(gs_derecha[0, 0])
    im_f1 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/02_render_submentoniana_digastrico.png")
    ax_f1.imshow(im_f1)
    ax_f1.axis('off')
    ax_f1.set_title("CH0: Anterior Belly\n(Submentoniana)", fontsize=10, fontweight='bold', color='#4ade80')

    # Foto 2: CH1 Orbicular Frontal
    ax_f2 = fig.add_subplot(gs_derecha[0, 1])
    im_f2 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/03_render_frontal_orbicularis.png")
    ax_f2.imshow(im_f2)
    ax_f2.axis('off')
    ax_f2.set_title("CH1: Orbicularis Oris\n(Frontal Labio Sup)", fontsize=10, fontweight='bold', color='#38bdf8')

    # Foto 3: CH2 Modiolo / Perfil
    ax_f3 = fig.add_subplot(gs_derecha[1, 0])
    im_f3 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/01_render_lateral_risorio.png")
    ax_f3.imshow(im_f3)
    ax_f3.axis('off')
    ax_f3.set_title("CH2: Modiolo / Risorio\n(Perfil Izquierdo)", fontsize=10, fontweight='bold', color='#f472b6')

    # Foto 4: Detalle / Referencia Mastoides
    ax_f4 = fig.add_subplot(gs_derecha[1, 1])
    # Recorte detalle de mastoides
    h_m, w_m, _ = im_f3.shape
    crop_mast = im_f3[int(h_m*0.35):int(h_m*0.65), int(w_m*0.7):int(w_m*0.98)]
    ax_f4.imshow(crop_mast)
    ax_f4.axis('off')
    ax_f4.set_title("REF: Apófisis Mastoides\n(Electrodo Referencia)", fontsize=10, fontweight='bold', color='#fbbf24')

    # Tabla / Tarjeta de Posición y Distancia Predicha por IA
    ax_card = fig.add_subplot(gs_derecha[2, :])
    ax_card.set_facecolor('#1e293b')
    ax_card.axis('off')

    texto_datos = (
        "PARÁMETROS EXPERIMENTALES Y DISTANCIAS PREDICHAS POR IA (±3.5 mm)\n"
        "• Sujeto: Candela | Fecha: 2026-09-01 | Frecuencia: 2000 Hz | Resistencia Soldada RG = 100 Ω (G ≈ 495)\n"
        "• Electrodos: Recortados a mano 5 x 5 mm (± 0.02 mm) con parche de gel superpuesto\n"
        "----------------------------------------------------------------------------------------------------\n"
        "• Canal 0 (Anterior Belly / Digástrico): Distancia predicha = 16.9 mm ± 3.5 mm  (Rango: 13.4 - 20.4 mm)\n"
        "• Canal 1 (Orbicularis Oris):            Distancia predicha = 25.4 mm ± 3.5 mm  (Rango: 21.9 - 28.9 mm)\n"
        "• Canal 2 (Modiolo / Comisura):         Distancia predicha = 19.1 mm ± 3.5 mm  (Rango: 15.6 - 22.6 mm)\n"
        "• Referencia: Apófisis mastoides izquierda | Separación nominal de laboratorio: 15.0 mm"
    )
    ax_card.text(0.03, 0.5, texto_datos, transform=ax_card.transAxes,
                 fontsize=9.5, fontfamily='monospace', color='#f1f5f9',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e293b', edgecolor='#475569', linewidth=1.5))

    # Guardar en PDF y PNG
    plt.savefig(output_pdf_path, format='pdf', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(output_png_path, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Página de Atlas generada:\n  PDF: {output_pdf_path}\n  PNG: {output_png_path}")

if __name__ == "__main__":
    pdf_out = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_candela_2026_09_01.pdf"
    png_out = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_candela_2026_09_01.png"
    generar_pagina_atlas_0901(pdf_out, png_out)
