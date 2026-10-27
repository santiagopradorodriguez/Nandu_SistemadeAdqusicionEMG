"""
Compilador Oficial del Atlas de Activación sEMG - Estándar Riguroso Ñandú
Ajustes solicitados:
1. Achicar un pelín el PCA (h_pca = 0.250 en lugar de 0.310).
2. Bajar los recuadros medios de sesión y LDA (y_banner = 0.295).
3. Agrandar sustancialmente la altura de las filas superiores (h_fila = 0.170 en lugar de 0.144).
4. Fotos mucho más grandes (w_foto y h_foto ampliadas).
5. Textos blancos de la tabla biométrica AL DOBLE DE TAMAÑO (14 pt / 15 pt) para que se lean de forma gigante y clara.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch

# Extractor oficial
sys.path.append("EMG_desarrollo/analysis")
from generador_atlas_pdf import GeneradorAtlasPDF, PRE_MS, POST_MS

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_COLS = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# Canales y colores originales
CANALES_CONFIG = [
    {
        'col_csv': 'Canal 0',
        'musculo_linea1': 'Anterior Belly',
        'musculo_linea2': 'Digástrico',
        'funcion': 'Apertura mandibular activa',
        'col': '#ef4444',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/02_render_submentoniana_digastrico.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '16.9 ± 3.5 mm',
        'dist_menton': '28.4 ± 3.5 mm',
        'dist_nariz': 'N/A (submentoniano)',
        'dist_trago': '78.5 ± 4.5 mm',
        'dist_comisura': '38.2 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 1',
        'musculo_linea1': 'Risorio',
        'musculo_linea2': 'Tracción lateral',
        'funcion': 'Sonrisa y retracción comisural',
        'col': '#22c55e',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/01_render_lateral_risorio.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '19.1 ± 3.5 mm',
        'dist_menton': '60.6 ± 4.0 mm',
        'dist_nariz': '92.0 ± 4.0 mm',
        'dist_trago': '116.8 ± 4.0 mm',
        'dist_comisura': '20.4 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 2',
        'musculo_linea1': 'Orbicular',
        'musculo_linea2': 'de los Labios',
        'funcion': 'Constricción y protrusión labial',
        'col': '#f59e0b',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/03_render_frontal_orbicularis.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '25.4 ± 3.5 mm',
        'dist_menton': '46.0 ± 3.5 mm',
        'dist_nariz': '16.3 ± 3.5 mm',
        'dist_trago': '71.1 ± 4.0 mm',
        'dist_comisura': '13.8 ± 3.5 mm'
    }
]

def compilar_atlas_redimensionado(pdf_out, png_out):
    ga = GeneradorAtlasPDF()
    ses_list = [
        {'A': 'A_Prueba1_Candela', 'E': 'E_Prueba1_Candela', 'I': 'I_Prueba1_Candela', 'O': 'O_Prueba1_Candela', 'U': 'U_Prueba1_Candela'},
        {'A': 'A_Prueba2_Candela', 'E': 'E_Prueba2_Candela', 'I': 'I_Prueba2_Candela', 'O': 'O_Prueba2_Candela', 'U': 'U_Prueba2_Candela'},
        {'A': 'A_Prueba3_Candela', 'E': 'E_Prueba3_Candela', 'I': 'I_Prueba3_Candela', 'O': 'O_Prueba3_Candela', 'U': 'U_Prueba3_Candela'},
        {'A': 'A_Prueba4_Candela', 'E': 'E_Prueba4_Candela', 'I': 'I_Prueba4_Candela', 'O': 'O_Prueba4_Candela', 'U': 'U_Prueba4_Candela'}
    ]

    curvas = {}
    for cfg in CANALES_CONFIG:
        col = cfg['col_csv']
        curvas[col] = {}
        for v in VOCALES:
            curvas[col][v] = ga.extraer_curva_promedio_series("2026-09-01", ses_list, v, col)

    df_feat = pd.read_csv("/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/salida_pca_0901_3d/caracteristicas_exportadas.csv")
    X = df_feat.iloc[:, 2:].values
    y = df_feat["Vocal"].values

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    var_ratios = pca.explained_variance_ratio_

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y)
    y_pred = lda.predict(X_pca)

    acc_lda_global = accuracy_score(y, y_pred) * 100.0
    acc_por_vocal = {}
    for v in VOCALES:
        idx_v = (y == v)
        acc_por_vocal[v] = accuracy_score(y[idx_v], y_pred[idx_v]) * 100.0

    fig = plt.figure(figsize=(26, 19.5), dpi=180, facecolor='#0a0f1d')
    
    c_card_bg = '#0f172a'
    c_card_border = '#334155'
    c_inner_box = '#1e293b'
    c_inner_border = '#475569'
    c_txt_main = '#f8fafc'
    c_txt_muted = '#94a3b8'
    c_axis_bg = '#151f32'
    c_grid = '#2d3b55'

    # 1. ENCABEZADO
    ax_head = fig.add_axes([0.025, 0.956, 0.95, 0.033])
    ax_head.set_facecolor('#151f32')
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    for sp in ax_head.spines.values():
        sp.set_color(c_card_border)

    ax_head.text(0.015, 0.65, "Experimento 2026-09-01: Digastrico_Risorio_Orbicularis_Candela - Patrón de Activación Muscular",
                 color='#38bdf8', fontsize=16.0, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG",
                 color='#94a3b8', fontsize=11, weight='bold', va='center')

    ax_head.text(0.985, 0.65, "Atlas Oficial de Activación Mioeléctrica",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
    ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Filtro: 20-500 Hz + Notch 50 Hz",
                 color='#38bdf8', fontsize=10.5, ha='right', va='center')

    # 2. SECCIÓN SUPERIOR OPTIMIZADA: FOTO CORRIDA A LA IZQUIERDA Y TABLA BIOMÉTRICA EXPANDIDA
    y_start = 0.942
    h_fila = 0.174
    gap_fila = 0.010

    w_meta = 0.115     # Compactado para permitir desplazar foto a la izquierda
    w_foto = 0.096     # Relación de aspecto nativa 3:4 (sin bandas negras vacías)
    w_tabla = 0.370    # Agrandada sustancialmente hacia la izquierda
    x_voc_start = 0.635
    w_voc_total = 0.350
    w_voc = (w_voc_total - 4 * 0.005) / 5.0

    for r_idx, cfg in enumerate(CANALES_CONFIG):
        y_base = y_start - (r_idx + 1) * h_fila - r_idx * gap_fila
        col = cfg['col_csv']
        col_m = cfg['col']

        # Metadatos del músculo
        ax_meta = fig.add_axes([0.025, y_base, w_meta, h_fila])
        ax_meta.set_facecolor(c_card_bg)
        ax_meta.set_xticks([])
        ax_meta.set_yticks([])
        for sp in ax_meta.spines.values():
            sp.set_color(c_card_border)

        ax_meta.text(0.07, 0.82, cfg['musculo_linea1'], color=col_m, fontsize=12.5, weight='bold', va='center')
        ax_meta.text(0.07, 0.67, cfg['musculo_linea2'], color=col_m, fontsize=11.5, weight='bold', va='center')
        ax_meta.text(0.07, 0.44, cfg['funcion'], color=c_txt_main, fontsize=10.0, va='center')
        ax_meta.text(0.07, 0.20, f"Canal sEMG: {col}", color='#38bdf8', fontsize=11.5, weight='bold', va='center')

        # FOTO FIDUCIARIA CORRIDA A LA IZQUIERDA
        ax_foto = fig.add_axes([0.025 + w_meta + 0.005, y_base, w_foto, h_fila])
        ax_foto.set_facecolor('#151f32')
        ax_foto.set_xticks([])
        ax_foto.set_yticks([])
        for sp in ax_foto.spines.values():
            sp.set_color(c_card_border)
        if os.path.exists(cfg['foto']):
            im_f = mpimg.imread(cfg['foto'])
            ax_foto.imshow(im_f)
            ax_foto.set_title("Colocación Anatómica", color='#cbd5e1', fontsize=11, weight='bold', pad=4)

        # TABLA BIOMÉTRICA AMPLIADA A LA IZQUIERDA CON COLUMNA ANCHA PARA FRASES COMPLETAS
        ax_tabla = fig.add_axes([0.025 + w_meta + 0.005 + w_foto + 0.006, y_base, w_tabla, h_fila])
        ax_tabla.set_facecolor(c_card_bg)
        ax_tabla.set_xticks([])
        ax_tabla.set_yticks([])
        for sp in ax_tabla.spines.values():
            sp.set_color(c_card_border)

        filas_tabla = [
            ("PARÁMETRO BIOMÉTRICO", "VALOR / ESTIMACIÓN (± ERROR)", True),
            ("Dimensión Electrodo", cfg['tam_electrodo'], False),
            ("Distancia Inter-electrodo (IED entre centros, estim.)", cfg['ied'], False),
            ("Dist. estimada a Mentón (Gnathion)", cfg['dist_menton'], False),
            ("Dist. estimada a Nariz (Pronasale)", cfg['dist_nariz'], False),
            ("Dist. estimada a Oreja (Tragus)", cfg['dist_trago'], False),
            ("Dist. estimada a Comisura Labial", cfg['dist_comisura'], False),
        ]

        n_filas = len(filas_tabla)
        h_row = 1.0 / n_filas

        for f_i, (lbl, val, is_header) in enumerate(filas_tabla):
            y_row = 1.0 - (f_i + 0.5) * h_row
            y_line = 1.0 - (f_i + 1) * h_row

            if is_header:
                p_head = FancyBboxPatch((0.005, 1.0 - h_row), 0.99, h_row, boxstyle="square,pad=0",
                                        facecolor='#1e293b', edgecolor='none')
                ax_tabla.add_patch(p_head)
                ax_tabla.text(0.025, y_row, lbl, color='#38bdf8', fontsize=11.0, weight='bold', va='center')
                ax_tabla.text(0.61, y_row, val, color='#38bdf8', fontsize=11.0, weight='bold', va='center')
            else:
                col_val = '#fbbf24' if "Inter-electrodo" in lbl else '#ffffff'
                ax_tabla.text(0.025, y_row, lbl, color='#cbd5e1', fontsize=10.5, va='center', weight='bold')
                ax_tabla.text(0.61, y_row, val, color=col_val, fontsize=12.0, weight='bold', va='center')

            if f_i < n_filas - 1:
                ax_tabla.axhline(y_line, color='#334155', linewidth=0.8, xmin=0.01, xmax=0.99)

        ax_tabla.axvline(0.59, color='#334155', linewidth=0.9, ymin=0.01, ymax=0.99)

        # 5 Columnas de Vocales
        for v_idx, v in enumerate(VOCALES):
            xv = x_voc_start + v_idx * (w_voc + 0.005)
            ax_v = fig.add_axes([xv, y_base, w_voc, h_fila])
            ax_v.set_facecolor(c_axis_bg)
            ax_v.grid(True, linestyle=':', alpha=0.35, color=c_grid)

            ax_v.axvline(0, color='#f87171', linestyle='--', linewidth=1.2, alpha=0.85)

            c = curvas[col].get(v)
            if c is not None:
                t = c['t']
                y_c = c['mean']
                y_err = c['std']

                ax_v.plot(t, y_c, color=col_m, linewidth=2.3, alpha=0.95)
                ax_v.fill_between(t, np.maximum(0, y_c - y_err), y_c + y_err, color=col_m, alpha=0.22)

                if np.max(y_c) > 0.15:
                    idx_p = np.argmax(y_c)
                    t_p = t[idx_p]
                    val_p = y_c[idx_p]
                    ax_v.plot(t_p, val_p, marker='o', markersize=4.2, color=col_m)
                    ax_v.text(t_p, min(1.10, val_p + 0.05), f"{int(t_p)}ms",
                              color='#ffffff', fontsize=9.5, ha='center', va='bottom', weight='bold')

            ax_v.set_xlim(-PRE_MS, POST_MS)
            ax_v.set_ylim(-0.05, 1.20)
            ax_v.tick_params(colors=c_txt_muted, labelsize=9)
            for sp in ax_v.spines.values():
                sp.set_color(c_card_border)

            if r_idx == 0:
                ax_v.set_title(f"Vocal /{v.lower()}/", color=VOCAL_COLS[v], fontsize=13.5, weight='bold', pad=4)
            if r_idx == len(CANALES_CONFIG) - 1:
                ax_v.set_xlabel("Tiempo al mic [ms]", color=c_txt_muted, fontsize=9.5, labelpad=3)

    # 3. SECCIÓN MEDIA: BANNERS AMPLIADOS HACIA ARRIBA (h_banner = 0.082)
    y_banner = 0.300
    h_banner = 0.082

    # Banner 1: Registro de Sesión
    ax_b1 = fig.add_axes([0.025, y_banner, 0.465, h_banner])
    ax_b1.set_facecolor(c_card_bg)
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    for sp in ax_b1.spines.values():
        sp.set_color(c_card_border)
    ax_b1.text(0.03, 0.86, "REGISTRO DE LA SESIÓN EXPERIMENTAL (CUADERNO)", color='#38bdf8', fontsize=12.0, weight='bold', va='center')

    p_t = FancyBboxPatch((0.02, 0.08), 0.30, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_t)
    ax_b1.text(0.04, 0.64, "TEMPORAL", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.04, 0.45, "• Candela (09-01)", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.28, "• 16:12 a 16:48", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.11, "• 36.2 min total", color='#ffffff', fontsize=11.5, weight='bold')

    p_a = FancyBboxPatch((0.34, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_a)
    ax_b1.text(0.36, 0.64, "HARDWARE", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.36, 0.45, "• Fs: 2000 Hz", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.28, "• Baterías: 8.3V / 7.7V", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.11, "• Ref: Mastoides izq.", color='#ffffff', fontsize=11.5, weight='bold')

    p_p = FancyBboxPatch((0.67, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_p)
    ax_b1.text(0.69, 0.64, "REGISTROS", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.69, 0.45, "• 22 tomas grabadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.28, "• 4 series analizadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.11, "• 222 pulsos válidos", color='#ffffff', fontsize=11.5, weight='bold')

    # Banner 2: LDA con Badges de Vocales
    ax_b2 = fig.add_axes([0.505, y_banner, 0.470, h_banner])
    ax_b2.set_facecolor(c_card_bg)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    for sp in ax_b2.spines.values():
        sp.set_color(c_card_border)
    ax_b2.text(0.03, 0.86, f"RENDIMIENTO CLASIFICADOR LINEAL (LDA) EN PCA 3D  |  GLOBAL: {acc_lda_global:.1f}%  (MACRO: {np.mean(list(acc_por_vocal.values())):.1f}%)", 
               color='#4ade80', fontsize=12.0, weight='bold', va='center')

    w_badge = 0.18
    gap_badge = 0.016
    for idx_b, v in enumerate(VOCALES):
        xb = 0.02 + idx_b * (w_badge + gap_badge)
        color_v = VOCAL_COLS[v]
        p_v = FancyBboxPatch((xb, 0.08), w_badge, 0.70, boxstyle="round,pad=0.015", 
                             facecolor=c_inner_box, edgecolor=color_v, lw=2.0)
        ax_b2.add_patch(p_v)
        
        ax_b2.text(xb + w_badge*0.5, 0.58, f"Vocal /{v.lower()}/", color=color_v, fontsize=12.0, weight='bold', ha='center', va='center')
        ax_b2.text(xb + w_badge*0.5, 0.26, f"{acc_por_vocal[v]:.1f}%", color='#ffffff', fontsize=14.5, weight='bold', ha='center', va='center')

    # 4. SECCIÓN INFERIOR: PCA 3D CUADRADOS CON LABELS Y TÍTULOS AL DOBLE
    y_pca = 0.038
    h_pca = 0.230

    projs = [(0, 1), (0, 2), (1, 2)]
    titulos_projs = [
        f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 2 ({var_ratios[1]*100:.1f}%)",
        f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)",
        f"Componente 2 ({var_ratios[1]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)"
    ]

    w_proj = 0.270
    gap_proj = (0.950 - 3 * w_proj) / 2.0  # Espaciado regular

    for p_idx, (c_x, c_y) in enumerate(projs):
        xp = 0.025 + p_idx * (w_proj + gap_proj)
        ax_p = fig.add_axes([xp, y_pca, w_proj, h_pca])
        ax_p.set_facecolor(c_axis_bg)
        ax_p.set_box_aspect(1)  # Garantiza que quede 100% CUADRADO
        ax_p.grid(True, linestyle='--', alpha=0.35, color=c_grid)

        for v in VOCALES:
            idx_v = (y == v)
            ax_p.scatter(X_pca[idx_v, c_x], X_pca[idx_v, c_y],
                         label=f"/{v.lower()}/", color=VOCAL_COLS[v],
                         s=80, alpha=0.92, edgecolors='#0a0f1d', linewidths=0.9, zorder=3)

        # LABELS Y TÍTULOS AL DOBLE DE TAMAÑO (18 - 20 pt)
        ax_p.set_xlabel(f"CP {c_x+1} ({var_ratios[c_x]*100:.1f}% var)", 
                        color='#cbd5e1', fontsize=18, fontweight='bold', labelpad=6)
        ax_p.set_ylabel(f"CP {c_y+1} ({var_ratios[c_y]*100:.1f}% var)", 
                        color='#cbd5e1', fontsize=18, fontweight='bold', labelpad=6)
        ax_p.set_title(titulos_projs[p_idx], color='#38bdf8', fontsize=18, weight='bold', pad=10)
        ax_p.tick_params(colors=c_txt_muted, labelsize=14)
        for sp in ax_p.spines.values():
            sp.set_color(c_card_border)

        if p_idx == 0:
            ax_p.legend(loc='upper right', frameon=True, facecolor=c_card_bg, edgecolor=c_card_border,
                        labelcolor='#f8fafc', fontsize=13, ncol=2)

    plt.savefig(pdf_out, format='pdf', facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(png_out, format='png', dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Atlas Redimensionado Generado con Éxito:\n  {pdf_out}\n  {png_out}")

if __name__ == "__main__":
    pdf = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_oficial_2026_09_01_completo.pdf"
    png = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/atlas_oficial_2026_09_01_completo.png"
    compilar_atlas_redimensionado(pdf, png)
    
    # Asegurar que la diapositiva 1 del atlas sea idéntica a la plantilla aprobada
    import shutil
    shutil.copyfile(png, "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/diapositiva_01_2026-09-01.png")
