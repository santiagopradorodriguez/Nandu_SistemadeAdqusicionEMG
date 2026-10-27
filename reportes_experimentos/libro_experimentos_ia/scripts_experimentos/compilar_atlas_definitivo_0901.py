"""
Compilador definitivo del Atlas Oficial sEMG por Sesión
Integra:
1. Día: Patrón de activación muscular (3 canales, 5 vocales, filtro 20-500 Hz, supremo tricanal)
2. PCA Oficial 2D con Elipses GMM generado con parámetros por defecto y solo reposo basal.
3. Las 3 fotos anatómicas curadas con sus marcas.
4. Foto de electrodos / referencia.
5. Tarjeta de instrumental y distancias predichas por IA (+- 3.5 mm).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

def compilar_atlas_candela_0901(pdf_path, png_path):
    fig = plt.figure(figsize=(18, 12), dpi=200, facecolor='#0f172a')
    
    # 2 Columnas principales
    gs = gridspec.GridSpec(2, 2, width_ratios=[1.05, 0.95], height_ratios=[1.0, 1.0], 
                           wspace=0.14, hspace=0.18, left=0.04, right=0.96, top=0.93, bottom=0.05)
    
    fig.suptitle("2026-09-01: Patrón de Activación Muscular sEMG - Candela", 
                 fontsize=22, fontweight='bold', color='#38bdf8', y=0.97)

    # 1. ARRIBA IZQUIERDA: Patrón de activación vectorial limpio
    ax_patron = fig.add_subplot(gs[0, 0])
    ax_patron.set_facecolor('#1e293b')
    im_patron = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/patron_activacion_vectorial_0901.png")
    ax_patron.imshow(im_patron)
    ax_patron.axis('off')
    ax_patron.set_title("Patrones Temporales de Activación Muscular (/a/, /e/, /i/, /o/, /u/)", 
                        fontsize=12, fontweight='bold', color='#f8fafc', pad=6)

    # 2. ABAJO IZQUIERDA: PCA 2D Oficial Generado con Parámetros por Defecto (500 Hz + Reposo Basal)
    ax_pca = fig.add_subplot(gs[1, 0])
    ax_pca.set_facecolor('#1e293b')
    im_pca = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/salida_pca_0901/PCA_2D.png")
    ax_pca.imshow(im_pca)
    ax_pca.axis('off')
    ax_pca.set_title("Proyección PCA 2D Oficial (Filtro 500 Hz + Reposo Basal + Elipses GMM)", 
                     fontsize=12, fontweight='bold', color='#38bdf8', pad=6)

    # 3. COLUMNA DERECHA: Fotos de colocación + Métricas
    gs_der = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs[:, 1], 
                                             wspace=0.10, hspace=0.20)

    # Foto 1: CH0 Digástrico Submentoniana
    ax_f1 = fig.add_subplot(gs_der[0, 0])
    im_f1 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/02_render_submentoniana_digastrico.png")
    ax_f1.imshow(im_f1)
    ax_f1.axis('off')
    ax_f1.set_title("CH0: Anterior Belly\n(Submentoniana Vertical)", fontsize=10, fontweight='bold', color='#4ade80')

    # Foto 2: CH1 Orbicular Frontal
    ax_f2 = fig.add_subplot(gs_der[0, 1])
    im_f2 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/03_render_frontal_orbicularis.png")
    ax_f2.imshow(im_f2)
    ax_f2.axis('off')
    ax_f2.set_title("CH1: Orbicularis Oris\n(Frontal Labio Superior)", fontsize=10, fontweight='bold', color='#38bdf8')

    # Foto 3: CH2 Modiolo Perfil Izquierdo
    ax_f3 = fig.add_subplot(gs_der[1, 0])
    im_f3 = mpimg.imread("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/01_render_lateral_risorio.png")
    ax_f3.imshow(im_f3)
    ax_f3.axis('off')
    ax_f3.set_title("CH2: Modiolo / Risorio\n(Perfil Lateral Izquierdo)", fontsize=10, fontweight='bold', color='#f472b6')

    # Foto 4: Referencia Mastoides
    ax_f4 = fig.add_subplot(gs_der[1, 1])
    h_m, w_m, _ = im_f3.shape
    crop_mast = im_f3[int(h_m*0.35):int(h_m*0.65), int(w_m*0.7):int(w_m*0.98)]
    ax_f4.imshow(crop_mast)
    ax_f4.axis('off')
    ax_f4.set_title("REF: Apófisis Mastoides\n(Electrodo de Referencia GND)", fontsize=10, fontweight='bold', color='#fbbf24')

    # Ficha Métricas
    ax_card = fig.add_subplot(gs_der[2, :])
    ax_card.set_facecolor('#1e293b')
    ax_card.axis('off')

    texto_datos = (
        "PARÁMETROS EXPERIMENTALES Y DISTANCIAS PREDICHAS POR IA (±3.5 mm)\n"
        "• Sujeto: Candela | Fecha: 2026-09-01 | Frecuencia de Muestreo: 2000 Hz\n"
        "• Acondicionamiento: AD620 (RG = 100 Ω, G ≈ 495 fijo) | Notch 50 Hz + Pasaaltos 20 Hz + Pasabajos 500 Hz\n"
        "• Electrodos: Recortados 5 x 5 mm (±0.02 mm) con parche de gel superpuesto | Separación nominal: 15.0 mm\n"
        "----------------------------------------------------------------------------------------------------\n"
        "• CH0 (Vientre anterior del digástrico): Distancia predicha IA = 16.9 mm ± 3.5 mm (Rango: 13.4 - 20.4 mm)\n"
        "• CH1 (Orbicularis Oris labio sup.):     Distancia predicha IA = 25.4 mm ± 3.5 mm (Rango: 21.9 - 28.9 mm)\n"
        "• CH2 (Modiolo / Comisura labial):      Distancia predicha IA = 19.1 mm ± 3.5 mm (Rango: 15.6 - 22.6 mm)\n"
        "• Referencia: Apófisis mastoides izquierda | Precisión Clustering No Supervisado (PCA 2D): 62.4%"
    )
    ax_card.text(0.02, 0.5, texto_datos, transform=ax_card.transAxes,
                 fontsize=9.5, fontfamily='monospace', color='#f1f5f9',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#1e293b', edgecolor='#475569', linewidth=1.5))

    plt.savefig(pdf_path, format='pdf', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(png_path, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Atlas consolidado generado con éxito:\n  {pdf_path}\n  {png_path}")

if __name__ == "__main__":
    pdf = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_candela_2026_09_01_definitivo.pdf"
    png = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_candela_2026_09_01_definitivo.png"
    compilar_atlas_candela_0901(pdf, png)
