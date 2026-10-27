# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Compilador de la Diapositiva 05 (2026-09-25 Bloque 1) del Atlas Oficial sEMG
#              - Campaña: 2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela
#              - Bloque 1: PruebaDigastrico
#              - Fila 1: Masetero (Rojo #ef4444) -> Canal 0
#              - Fila 2: Anterior Belly Digástrico Horizontal (Verde #059669) -> Canal 1
#              - Fila 3: Orbicular Labio Superior (Amarillo #f59e0b) -> Canal 2
#              - Extracción y filtrado de outliers: generador_pca_umap con defaults de GUI
#              - Lienzo exacto: figsize=(26, 19.5), dpi=200 (5200x3900 px)
# ==============================================================================

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

# Importar motor canónico de curvas
sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/analysis")
from generador_atlas_pdf import GeneradorAtlasPDF

# Importar extractor y filtro de outliers oficial de generador_pca_umap
sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/deep_learning/pca_umap_clustering")
from generador_pca_umap import extraer_y_filtrar

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_COLS = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

CANALES_CONFIG = [
    {
        'col_csv': 'Canal 0',
        'musculo_linea1': 'Masetero',
        'musculo_linea2': 'Cuerpo mandibular',
        'funcion': 'Elevación mandibular y oclusión',
        'col': '#059669',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/01_render_0925_bloque1_masetero_3x4.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '20.0 ± 3.5 mm',
        'dist_menton': '65.0 ± 4.0 mm',
        'dist_nariz': '70.0 ± 4.0 mm',
        'dist_trago': '44.0 ± 3.5 mm',
        'dist_comisura': '42.0 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 1',
        'musculo_linea1': 'Anterior Belly',
        'musculo_linea2': 'Digástrico Horizontal',
        'funcion': 'Apertura mandibular activa',
        'col': '#ef4444',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/02_render_0925_bloque1_digastrico_3x4.png',
        'tam_electrodo': '10 mm (disco ag/agcl)',
        'ied': '25.0 ± 3.5 mm',
        'dist_menton': '22.0 ± 3.5 mm',
        'dist_nariz': 'N/A (submentoniano)',
        'dist_trago': '75.0 ± 4.5 mm',
        'dist_comisura': '35.0 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 2',
        'musculo_linea1': 'Orbicular',
        'musculo_linea2': 'Labio Superior',
        'funcion': 'Protrusión y esfínter oral',
        'col': '#f59e0b',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/03_render_0925_bloque1_orbicular_3x4.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '24.0 ± 3.5 mm',
        'dist_menton': '45.0 ± 3.5 mm',
        'dist_nariz': '12.0 ± 3.0 mm',
        'dist_trago': '72.0 ± 4.0 mm',
        'dist_comisura': '15.0 ± 3.5 mm'
    }
]

def compilar_diapositiva_05(png_out, pdf_out):
    base_dir_camp = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Drive/2026-09-25_Exploracion_Modiolo_Digastrico_Mentoniano_Candela'
    ga = GeneradorAtlasPDF(base_db=base_dir_camp)
    ses_list = [
        {'A': 'A_PruebaDigastrico_Candela', 'E': 'E_PruebaDigastrico_Candela', 'I': 'I_PruebaDigastrico_Candela', 'O': 'O_PruebaDigastrico_Candela', 'U': 'U_PruebaDigastrico_Candela'},
        {'A': 'A_PruebaDigastrico_2_Candela', 'E': 'E_PruebaDigastrico_2_Candela', 'I': 'I_PruebaDigastrico_2_Candela', 'O': 'O_PruebaDigastrico_2_Candela', 'U': 'U_PruebaDigastrico_2_Candela'}
    ]

    # Extraer curvas promedio oficiales
    curvas = {}
    for cfg in CANALES_CONFIG:
        col = cfg['col_csv']
        curvas[col] = {}
        for v in VOCALES:
            curvas[col][v] = ga.extraer_curva_promedio_series('mediciones', ses_list, v, col)

    # Extracción y filtrado oficial de outliers con generador_pca_umap (defaults de la interfaz GUI)
    params_default = {
        'alpha_ruido': 0.5,
        'smooth_ms': 125,
        'target_length': 20,
        'snr_threshold': 0.5,
        'outlier_contamination': 0.10,
        'gate_ratio_ruido': 0.0,
        'tipo_filtro_ruido': 'notch',
        'notch_q': 30.0,
        'highpass_cutoff_hz': 20.0,
        'lowpass_cutoff_hz': 500.0,
        'aplicar_correccion_intersesion': False
    }

    base_dir_pca = os.path.join(base_dir_camp, 'mediciones')
    import json
    mediciones_pca = []
    for d in sorted(os.listdir(base_dir_pca)):
        if ('Digastrico' in d or 'astrico' in d) and os.path.isdir(os.path.join(base_dir_pca, d)):
            p = os.path.join(base_dir_pca, d, 'canal_0', 'metadata.json')
            if os.path.exists(p):
                with open(p) as f_m:
                    if json.load(f_m).get('muscles_map', {}).get('canal_1') == 'belly 1':
                        mediciones_pca.append(d)

    X_clean, y_clean, Tomas_clean, descartados = extraer_y_filtrar(
        mediciones=mediciones_pca,
        base_dir=base_dir_pca,
        params=params_default,
        aplicar_trevisan=False,
        modo_alineacion='Pico Volumen Micrófono',
        pre_pct=0.4,
        post_pct=0.6,
        canales_features=['canal_0', 'canal_1', 'canal_2'],
        verbose=True
    )

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_clean)
    var_ratios = pca.explained_variance_ratio_

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y_clean)
    y_pred = lda.predict(X_pca)

    acc_lda_global = accuracy_score(y_clean, y_pred) * 100.0
    acc_lda_macro = np.mean([accuracy_score(y_clean[y_clean == v], y_pred[y_clean == v]) * 100.0 for v in VOCALES if np.sum(y_clean == v) > 0])
    acc_por_vocal = {v: (accuracy_score(y_clean[y_clean == v], y_pred[y_clean == v]) * 100.0 if np.sum(y_clean == v) > 0 else 0.0) for v in VOCALES}

    # Construcción del Lienzo Exacto
    fig = plt.figure(figsize=(26, 19.5), dpi=200, facecolor='#0a0f1d')

    c_card_bg = '#0f172a'
    c_card_border = '#334155'
    c_inner_box = '#1e293b'
    c_inner_border = '#475569'
    c_txt_main = '#f8fafc'
    c_txt_muted = '#94a3b8'
    c_axis_bg = '#151f32'
    c_grid = '#2d3b55'

    # 1. Encabezado
    ax_head = fig.add_axes([0.025, 0.956, 0.95, 0.033])
    ax_head.set_facecolor('#151f32')
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    for sp in ax_head.spines.values():
        sp.set_color(c_card_border)

    ax_head.text(0.015, 0.65, "Experimento 2026-09-25: Bloque 1 PruebaDigastrico - Patrón de Activación Muscular",
                 color='#38bdf8', fontsize=16.0, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG",
                 color='#94a3b8', fontsize=11, weight='bold', va='center')

    ax_head.text(0.985, 0.65, "Atlas Oficial de Activación Mioeléctrica",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
    ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Filtro: 20-500 Hz + Notch 50 Hz",
                 color='#38bdf8', fontsize=10.5, ha='right', va='center')

    # 2. Filas de Músculos (Sección Superior)
    y_start = 0.940
    h_fila = 0.174
    gap_fila = 0.010

    w_meta = 0.115
    w_foto = 0.096
    w_tabla = 0.370
    x_voc_start = 0.635
    w_voc_total = 0.350
    w_voc = (w_voc_total - 4 * 0.005) / 5.0

    PRE_MS = 600
    POST_MS = 800

    for r_idx, cfg in enumerate(CANALES_CONFIG):
        y_base = y_start - (r_idx + 1) * h_fila - r_idx * gap_fila
        col = cfg['col_csv']
        col_m = cfg['col']

        # Tarjeta muscular izquierda
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

        # Foto fiduciaria de colocación (3:4 portrait)
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

        # Tabla biométrica invariante
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

        # 5 Gráficos de activación vocal (/a/, /e/, /i/, /o/, /u/) con COLOR DEL MÚSCULO
        for v_idx, v in enumerate(VOCALES):
            xv = x_voc_start + v_idx * (w_voc + 0.005)
            ax_v = fig.add_axes([xv, y_base, w_voc, h_fila])
            ax_v.set_facecolor(c_axis_bg)
            ax_v.grid(True, linestyle=':', alpha=0.35, color=c_grid)

            ax_v.axvline(0, color='#f87171', linestyle='--', linewidth=1.2, alpha=0.85)

            c_dict = curvas.get(col, {}).get(v)
            if c_dict is not None and len(c_dict.get('mean', [])) > 0:
                t = c_dict['t']
                y_c = c_dict['mean']
                s_c = c_dict['std']

                ax_v.plot(t, y_c, color=col_m, linewidth=2.3, alpha=0.95)
                ax_v.fill_between(t, np.maximum(0, y_c - s_c), y_c + s_c, color=col_m, alpha=0.20)

                i_pico = np.argmax(y_c)
                t_pico = t[i_pico]
                y_pico = y_c[i_pico]
                ax_v.plot(t_pico, y_pico, marker='o', markersize=4.5, color='#ffffff', zorder=4)
                ax_v.annotate(f"{t_pico:.0f} ms", (t_pico, y_pico),
                              textcoords="offset points", xytext=(0, 5),
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

    # 3. Sección Media: Banners a y_banner = 0.300
    y_banner = 0.300
    h_banner = 0.082

    # Banner 1: Registro de Sesión
    ax_b1 = fig.add_axes([0.025, y_banner, 0.465, h_banner])
    ax_b1.set_facecolor(c_card_bg)
    ax_b1.set_xticks([])
    ax_b1.set_yticks([])
    for sp in ax_b1.spines.values():
        sp.set_color(c_card_border)

    p_b1_title = FancyBboxPatch((0.005, 0.68), 0.99, 0.30, boxstyle="square,pad=0",
                                facecolor=c_inner_box, edgecolor='none')
    ax_b1.add_patch(p_b1_title)
    ax_b1.text(0.025, 0.83, "REGISTRO DE LA SESIÓN EXPERIMENTAL (CUADERNO)",
               color='#38bdf8', fontsize=11.5, weight='bold', va='center')

    col_w = 0.31
    for c_i, (b_title, b_items) in enumerate([
        ("TEMPORAL", [
            "• Candela (09-25)",
            "• 14:07 a 14:14",
            "• 6.5 min total"
        ]),
        ("HARDWARE", [
            "• Fs: 6000 Hz",
            "• Ref: Frente",
            "• Ganancia: 495 V/V"
        ]),
        ("REGISTROS", [
            "• 10 tomas grabadas",
            "• 2 series analizadas",
            f"• {len(y_clean)} pulsos válidos"
        ])
    ]):
        xc = 0.025 + c_i * (col_w + 0.02)
        ax_b1.text(xc, 0.52, b_title, color='#38bdf8', fontsize=10.0, weight='bold')
        for it_i, it_txt in enumerate(b_items):
            ax_b1.text(xc, 0.34 - it_i * 0.14, it_txt, color=c_txt_main, fontsize=9.5)

    # Banner 2: Rendimiento LDA
    ax_b2 = fig.add_axes([0.505, y_banner, 0.470, h_banner])
    ax_b2.set_facecolor(c_card_bg)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    for sp in ax_b2.spines.values():
        sp.set_color(c_card_border)

    p_b2_title = FancyBboxPatch((0.005, 0.68), 0.99, 0.30, boxstyle="square,pad=0",
                                facecolor=c_inner_box, edgecolor='none')
    ax_b2.add_patch(p_b2_title)
    ax_b2.text(0.025, 0.83,
               f"RENDIMIENTO CLASIFICADOR LINEAL (LDA) EN PCA 3D  |  GLOBAL: {acc_lda_global:.1f}% (MACRO: {acc_lda_macro:.1f}%)",
               color='#4ade80', fontsize=11.5, weight='bold', va='center')

    w_box = 0.18
    gap_box = 0.018
    h_box = 0.54
    y_box = 0.09

    for v_i, v in enumerate(VOCALES):
        xb = 0.015 + v_i * (w_box + gap_box)
        p_v = FancyBboxPatch((xb, y_box), w_box, h_box, boxstyle="square,pad=0",
                             facecolor=c_axis_bg, edgecolor=VOCAL_COLS[v], linewidth=1.4)
        ax_b2.add_patch(p_v)
        ax_b2.text(xb + w_box / 2.0, y_box + 0.36, f"Vocal /{v.lower()}/",
                   color=VOCAL_COLS[v], fontsize=10.5, weight='bold', ha='center', va='center')
        ax_b2.text(xb + w_box / 2.0, y_box + 0.15, f"{acc_por_vocal[v]:.1f}%",
                   color='#ffffff', fontsize=13.0, weight='bold', ha='center', va='center')

    # 4. Sección Inferior: 3 Proyecciones Ortogonales PCA 3D
    y_pca = 0.038
    h_pca = 0.230
    w_proj = 0.270
    gap_pca = (0.950 - 3 * w_proj) / 2.0

    proys = [
        (0, 1, 'PC1', 'PC2'),
        (0, 2, 'PC1', 'PC3'),
        (1, 2, 'PC2', 'PC3')
    ]

    for p_idx, (idx1, idx2, lbl1, lbl2) in enumerate(proys):
        xp = 0.025 + p_idx * (w_proj + gap_pca)
        ax_p = fig.add_axes([xp, y_pca, w_proj, h_pca])
        ax_p.set_facecolor(c_axis_bg)
        ax_p.set_box_aspect(1)
        ax_p.grid(True, linestyle=':', alpha=0.35, color=c_grid)

        for sp in ax_p.spines.values():
            sp.set_color(c_card_border)

        ax_p.set_title(f"{lbl1} vs {lbl2}", color=c_txt_main, fontsize=18, weight='bold', pad=8)

        v1_pct = var_ratios[idx1] * 100.0
        v2_pct = var_ratios[idx2] * 100.0
        ax_p.set_xlabel(f"{lbl1} ({v1_pct:.1f}%)", color=c_txt_main, fontsize=15, weight='bold', labelpad=4)
        ax_p.set_ylabel(f"{lbl2} ({v2_pct:.1f}%)", color=c_txt_main, fontsize=15, weight='bold', labelpad=4)

        for v in VOCALES:
            mask = (y_clean == v)
            if np.sum(mask) > 0:
                ax_p.scatter(X_pca[mask, idx1], X_pca[mask, idx2],
                             color=VOCAL_COLS[v], label=f"/{v.lower()}/",
                             s=55, alpha=0.88, edgecolors='#ffffff', linewidths=0.5, zorder=3)

        ax_p.tick_params(colors=c_txt_muted, labelsize=14)

        if p_idx == 0:
            leg = ax_p.legend(facecolor=c_card_bg, edgecolor=c_card_border,
                              fontsize=13, loc='upper right', framealpha=0.92)
            for text in leg.get_texts():
                text.set_color(c_txt_main)
                text.set_weight('bold')

    print(f"Guardando imagen en {png_out} (DPI=200)...")
    plt.savefig(png_out, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Guardando PDF en {pdf_out}...")
    plt.savefig(pdf_out, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("[OK] Diapositiva 05 generada exitosamente con geometría exacta e invarianza estética.")

if __name__ == '__main__':
    out_dir = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29'
    png_path = os.path.join(out_dir, 'diapositiva_05_2026-09-25_bloque1.png')
    pdf_path = os.path.join(out_dir, 'diapositiva_05_2026-09-25_bloque1.pdf')
    compilar_diapositiva_05(png_path, pdf_path)
