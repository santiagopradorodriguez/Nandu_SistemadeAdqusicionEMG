# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Compilador de la Diapositiva 02 (2026-09-15) del Atlas Oficial sEMG
#              - Fila 1: Apertura (Rojo #ef4444) -> Anterior Belly Digástrico (Canal 0)
#              - Fila 2: Sonrisa (Verde #22c55e) -> Cigomático Mayor (Canal 2)
#              - Fila 3: Protrusión (Amarillo #f59e0b) -> Orbicular de los Labios (Canal 1)
#              - PCA/LDA extraído con generador_pca_umap.extraer_y_filtrar (IsolationForest 10%)
#              - Geometría de ejes idéntica a la Diapositiva 01 (cero solapamientos)
# ==============================================================================

import os
import sys
import json
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
        'musculo_linea1': 'Anterior Belly',
        'musculo_linea2': 'Digástrico',
        'funcion': 'Apertura mandibular activa',
        'col': '#ef4444',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/01_render_0915_submentoniana.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '17.4 ± 3.5 mm',
        'dist_menton': '29.1 ± 3.5 mm',
        'dist_nariz': 'N/A (submentoniano)',
        'dist_trago': '79.2 ± 4.5 mm',
        'dist_comisura': '39.0 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 2',
        'musculo_linea1': 'Cigomático Mayor',
        'musculo_linea2': 'Elevador labial',
        'funcion': 'Elevación comisural y sonrisa',
        'col': '#22c55e',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/02_render_0915_cigomatico_corregido.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '20.0 ± 3.5 mm',
        'dist_menton': '60.9 ± 4.0 mm',
        'dist_nariz': '39.5 ± 3.5 mm',
        'dist_trago': '104.6 ± 4.0 mm',
        'dist_comisura': '19.2 ± 3.5 mm'
    },
    {
        'col_csv': 'Canal 1',
        'musculo_linea1': 'Orbicular',
        'musculo_linea2': 'de los Labios',
        'funcion': 'Constricción y protrusión labial',
        'col': '#f59e0b',
        'foto': '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/04_render_0915_frontal_orbicularis.png',
        'tam_electrodo': '5 x 5 mm (±0.02 mm)',
        'ied': '22.0 ± 3.5 mm',
        'dist_menton': '45.2 ± 3.5 mm',
        'dist_nariz': '18.1 ± 3.5 mm',
        'dist_trago': '74.0 ± 4.0 mm',
        'dist_comisura': '14.5 ± 3.5 mm'
    }
]

def compilar_diapositiva_02(png_out, pdf_out):
    ga = GeneradorAtlasPDF()
    ses_list = [
        {'A': 'A_Serie1_Candela', 'E': 'E_Serie1_Candela', 'I': 'I_Serie1_Candela', 'O': 'O_Serie1_Candela', 'U': 'U_Serie1_Candela'},
        {'A': 'A_Serie2_Candela', 'E': 'E_Serie2_Candela', 'I': 'I_Serie2_Candela', 'O': 'O_Serie2_Candela', 'U': 'U_Serie2_Candela'},
        {'A': 'A_Serie3_Candela', 'E': 'E_Serie3_Candela', 'I': 'I_Serie3_Candela', 'O': 'O_Serie3_Candela', 'U': 'U_Serie3_Candela'},
        {'A': 'A_Serie4_Candela', 'E': 'E_Serie4_Candela', 'I': 'I_Serie4_Candela', 'O': 'O_Serie4_Candela', 'U': 'U_Serie4_Candela'}
    ]

    # Extraer curvas promedio oficiales
    curvas = {}
    for cfg in CANALES_CONFIG:
        col = cfg['col_csv']
        curvas[col] = {}
        for v in VOCALES:
            curvas[col][v] = ga.extraer_curva_promedio_series("2026-09-16", ses_list, v, col)

    # 1. Extracción y filtrado oficial de outliers con generador_pca_umap usando PARÁMETROS POR DEFECTO DE LA INTERFAZ
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

    base_dir_pca = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos/2026-09-16'
    mediciones_pca = sorted([d for d in os.listdir(base_dir_pca) if os.path.isdir(os.path.join(base_dir_pca, d))])

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
    acc_por_vocal = {}
    for v in VOCALES:
        idx_v = (y_clean == v)
        acc_por_vocal[v] = accuracy_score(y_clean[idx_v], y_pred[idx_v]) * 100.0
    acc_macro = np.mean(list(acc_por_vocal.values()))

    # Configuración de lienzo EXACTA de Diapositiva 01: figsize=(26, 19.5), dpi=200
    fig = plt.figure(figsize=(26, 19.5), dpi=200, facecolor='#0a0f1d')
    
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

    ax_head.text(0.015, 0.65, "Experimento 2026-09-15: Digastrico_Cigomatico_Orbicularis_Candela - Patrón de Activación Muscular",
                 color='#38bdf8', fontsize=16.0, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG",
                 color='#94a3b8', fontsize=11, weight='bold', va='center')

    ax_head.text(0.985, 0.65, "Atlas Oficial de Activación Mioeléctrica",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
    ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Filtro: 20-500 Hz + Notch 50 Hz",
                 color='#38bdf8', fontsize=10.5, ha='right', va='center')

    # 2. SECCIÓN SUPERIOR: FILAS MUSCULARES
    y_start = 0.942
    h_fila = 0.174
    gap_fila = 0.010

    w_meta = 0.115
    w_foto = 0.096
    w_tabla = 0.370
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

        # Foto fiduciaria de colocación
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

        # Tabla biométrica
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

        # 5 Gráficos de activación vocal (/a/, /e/, /i/, /o/, /u/)
        for v_idx, v in enumerate(VOCALES):
            xv = x_voc_start + v_idx * (w_voc + 0.005)
            ax_v = fig.add_axes([xv, y_base, w_voc, h_fila])
            ax_v.set_facecolor(c_axis_bg)
            ax_v.grid(True, linestyle=':', alpha=0.35, color=c_grid)

            ax_v.axvline(0, color='#f87171', linestyle='--', linewidth=1.2, alpha=0.85)

            c_data = curvas[col][v]
            if c_data is not None:
                t = c_data['t']
                y_c = c_data['mean']
                y_err = c_data['std']

                # COLOR DEL MÚSCULO POR FILA (Rojo para Apertura, Verde para Sonrisa, Amarillo para Protrusión)
                ax_v.plot(t, y_c, color=col_m, linewidth=2.3, alpha=0.95)
                ax_v.fill_between(t, np.maximum(0, y_c - y_err), y_c + y_err, color=col_m, alpha=0.22)

                if np.max(y_c) > 0.15:
                    idx_p = np.argmax(y_c)
                    t_p = t[idx_p]
                    val_p = y_c[idx_p]
                    ax_v.plot(t_p, val_p, marker='o', markersize=4.2, color=col_m)
                    ax_v.text(t_p, min(1.10, val_p + 0.05), f"{int(t_p)}ms",
                              color='#ffffff', fontsize=9.5, ha='center', va='bottom', weight='bold')

            ax_v.set_xlim(-600, 800)
            ax_v.set_ylim(-0.05, 1.20)
            ax_v.tick_params(colors=c_txt_muted, labelsize=9)
            for sp in ax_v.spines.values():
                sp.set_color(c_card_border)

            if r_idx == 0:
                ax_v.set_title(f"Vocal /{v.lower()}/", color=VOCAL_COLS[v], fontsize=13.5, weight='bold', pad=4)
            if r_idx == len(CANALES_CONFIG) - 1:
                ax_v.set_xlabel("Tiempo al mic [ms]", color=c_txt_muted, fontsize=9.5, labelpad=3)
            else:
                ax_v.set_xticklabels([])
            if v_idx > 0:
                ax_v.set_yticklabels([])

    # 3. SECCIÓN MEDIA: BANNERS IDÉNTICOS A DIAPOSITIVA 01 (y_banner = 0.300, h_banner = 0.082)
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
    ax_b1.text(0.04, 0.45, "• Candela (09-15)", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.28, "• 15:52 a 16:35", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.11, "• 43.1 min total", color='#ffffff', fontsize=11.5, weight='bold')

    p_a = FancyBboxPatch((0.34, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_a)
    ax_b1.text(0.36, 0.64, "HARDWARE", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.36, 0.45, "• Fs: 2000 Hz", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.28, "• Baterías: 8.4V / 7.8V", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.11, "• Ref: Mastoides izq.", color='#ffffff', fontsize=11.5, weight='bold')

    p_p = FancyBboxPatch((0.67, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_p)
    ax_b1.text(0.69, 0.64, "REGISTROS", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.69, 0.45, "• 20 tomas grabadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.28, "• 4 series analizadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.11, f"• {len(y_clean)} pulsos válidos", color='#ffffff', fontsize=11.5, weight='bold')

    # Banner 2: LDA con Badges de Vocales
    ax_b2 = fig.add_axes([0.505, y_banner, 0.470, h_banner])
    ax_b2.set_facecolor(c_card_bg)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    for sp in ax_b2.spines.values():
        sp.set_color(c_card_border)
    ax_b2.text(0.03, 0.86, f"RENDIMIENTO CLASIFICADOR LINEAL (LDA) EN PCA 3D  |  GLOBAL: {acc_lda_global:.1f}%  (MACRO: {acc_macro:.1f}%)", 
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

    # 4. SECCIÓN INFERIOR: PCA 3D CUADRADOS IDÉNTICOS A DIAPOSITIVA 01 (y_pca = 0.038, h_pca = 0.230)
    y_pca = 0.038
    h_pca = 0.230
    w_proj = 0.270
    gap_proj = (0.950 - 3 * w_proj) / 2.0

    projs = [(0, 1), (0, 2), (1, 2)]
    titulos_projs = [
        f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 2 ({var_ratios[1]*100:.1f}%)",
        f"Componente 1 ({var_ratios[0]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)",
        f"Componente 2 ({var_ratios[1]*100:.1f}%) vs Componente 3 ({var_ratios[2]*100:.1f}%)"
    ]

    for p_idx, (c_x, c_y) in enumerate(projs):
        xp = 0.025 + p_idx * (w_proj + gap_proj)
        ax_p = fig.add_axes([xp, y_pca, w_proj, h_pca])
        ax_p.set_facecolor(c_axis_bg)
        ax_p.set_box_aspect(1)
        for sp in ax_p.spines.values():
            sp.set_color(c_card_border)
            sp.set_linewidth(1.5)

        ax_p.grid(True, linestyle=':', alpha=0.35, color=c_grid)

        for v in VOCALES:
            idx_v = (y_clean == v)
            if np.sum(idx_v) > 0:
                ax_p.scatter(X_pca[idx_v, c_x], X_pca[idx_v, c_y],
                             c=VOCAL_COLS[v], label=f"/{v.lower()}/",
                             s=55, alpha=0.85, edgecolors='none')

        ax_p.set_title(titulos_projs[p_idx], color='#f8fafc', fontsize=18, weight='bold', pad=10)
        ax_p.set_xlabel(f"PC{c_x+1} ({var_ratios[c_x]*100:.1f}% var)", color=c_txt_main, fontsize=18, weight='bold')
        ax_p.set_ylabel(f"PC{c_y+1} ({var_ratios[c_y]*100:.1f}% var)", color=c_txt_main, fontsize=18, weight='bold')
        ax_p.tick_params(colors=c_txt_muted, labelsize=14)

        if p_idx == 0:
            leg = ax_p.legend(facecolor='#1e293b', edgecolor=c_card_border, labelcolor=c_txt_main,
                              fontsize=13, loc='upper right', framealpha=0.9)
            for text, v in zip(leg.get_texts(), VOCALES):
                text.set_color(VOCAL_COLS[v])
                text.set_weight('bold')

    print(f"Guardando imagen en {png_out} (DPI=200)...")
    plt.savefig(png_out, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Guardando PDF en {pdf_out}...")
    plt.savefig(pdf_out, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print("[OK] Diapositiva 02 generada exitosamente con geometría exacta e invarianza estética.")

if __name__ == "__main__":
    png_path = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/diapositiva_02_2026-09-15.png"
    pdf_path = "/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/diapositiva_02_2026-09-15.pdf"
    compilar_diapositiva_02(png_path, pdf_path)
