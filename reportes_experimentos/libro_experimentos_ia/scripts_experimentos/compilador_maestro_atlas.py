# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor Maestro Canónico Parametrizado del Atlas Oficial sEMG
#              - Unifica el 100% de la estética, DSP y cálculo PCA en una sola función
#              - Soporta procesamiento en lote para todos los experimentos
#              - Deja placeholders pulcros en fotos y biometría pendientes
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

sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/analysis")
from generador_atlas_pdf import GeneradorAtlasPDF

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

def compilar_diapositiva_maestra(
    titulo_experimento: str,
    base_dir_camp: str,
    canales_cfg: list,
    ses_list_curvas: list,
    filtro_mediciones_pca: callable,
    info_cuaderno: dict,
    png_out: str,
    pdf_out: str,
    subfolder_mediciones: str = 'mediciones'
):
    print(f"\n=======================================================")
    print(f"Procesando: {titulo_experimento}")
    print(f"=======================================================")

    ga = GeneradorAtlasPDF(base_db=base_dir_camp)

    canales_activos = [cfg for cfg in canales_cfg if cfg.get('activo', True)]
    n_activos = len(canales_activos)

    # 1. Extraer curvas promedio oficiales
    curvas = {}
    for cfg in canales_activos:
        col = cfg['col_csv']
        curvas[col] = {}
        for v in VOCALES:
            if len(ses_list_curvas) > 0:
                try:
                    curvas[col][v] = ga.extraer_curva_promedio_series(subfolder_mediciones, ses_list_curvas, v, col)
                except Exception as e:
                    curvas[col][v] = {'t': np.linspace(-600, 800, 100), 'mean': np.zeros(100), 'std': np.zeros(100)}
            else:
                curvas[col][v] = {'t': np.linspace(-600, 800, 100), 'mean': np.zeros(100), 'std': np.zeros(100)}

    # 2. PCA 3D y LDA con generador_pca_umap canónico
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

    base_dir_pca = os.path.join(base_dir_camp, subfolder_mediciones) if subfolder_mediciones else base_dir_camp
    todas_meds = sorted([d for d in os.listdir(base_dir_pca) if os.path.isdir(os.path.join(base_dir_pca, d)) and not 'Secuencia' in d])
    mediciones_filtradas = [d for d in todas_meds if filtro_mediciones_pca(d, os.path.join(base_dir_pca, d))]

    canales_features = [cfg['col_csv'].lower().replace(' ', '_') for cfg in canales_cfg if cfg.get('activo', True)]
    if not canales_features:
        canales_features = ['canal_0', 'canal_1', 'canal_2']

    X_clean, y_clean, Tomas_clean, descartados = extraer_y_filtrar(
        mediciones=mediciones_filtradas,
        base_dir=base_dir_pca,
        params=params_default,
        aplicar_trevisan=False,
        modo_alineacion='Pico Volumen Micrófono',
        pre_pct=0.4,
        post_pct=0.6,
        canales_features=canales_features,
        verbose=False
    )

    n_comp = min(3, X_clean.shape[1], len(X_clean))
    if n_comp < 3:
        # Fallback padding si es monocanal
        X_padded = np.zeros((len(X_clean), 3))
        X_padded[:, :X_clean.shape[1]] = X_clean[:, :min(3, X_clean.shape[1])]
        X_clean = X_padded

    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_clean)
    var_ratios = pca.explained_variance_ratio_

    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_pca, y_clean)
        y_pred = lda.predict(X_pca)
        acc_lda_global = accuracy_score(y_clean, y_pred) * 100.0
        acc_lda_macro = np.mean([accuracy_score(y_clean[y_clean == v], y_pred[y_clean == v]) * 100.0 for v in VOCALES if np.sum(y_clean == v) > 0])
        acc_por_vocal = {v: (accuracy_score(y_clean[y_clean == v], y_pred[y_clean == v]) * 100.0 if np.sum(y_clean == v) > 0 else 0.0) for v in VOCALES}
    except Exception:
        acc_lda_global = 0.0
        acc_lda_macro = 0.0
        acc_por_vocal = {v: 0.0 for v in VOCALES}

    # 3. Construcción del Lienzo Exacto Invariante (26 x 19.5 pulgadas, 5200x3900 px)
    fig = plt.figure(figsize=(26, 19.5), dpi=200, facecolor='#0a0f1d')

    c_card_bg = '#0f172a'
    c_card_border = '#334155'
    c_inner_box = '#1e293b'
    c_inner_border = '#475569'
    c_txt_main = '#f8fafc'
    c_txt_muted = '#94a3b8'
    c_axis_bg = '#151f32'
    c_grid = '#2d3b55'

    # ENCABEZADO
    ax_head = fig.add_axes([0.025, 0.956, 0.95, 0.033])
    ax_head.set_facecolor('#151f32')
    ax_head.set_xticks([])
    ax_head.set_yticks([])
    for sp in ax_head.spines.values():
        sp.set_color(c_card_border)

    ax_head.text(0.015, 0.65, f"{titulo_experimento} - Patrón de Activación Muscular",
                 color='#38bdf8', fontsize=16.0, weight='bold', va='center')
    ax_head.text(0.015, 0.25, "Laboratorio de Sistemas Dinámicos (LSD - UBA) | Sistema Ñandú EMG",
                 color='#94a3b8', fontsize=11, weight='bold', va='center')

    ax_head.text(0.985, 0.65, "Atlas Oficial de Activación Mioeléctrica",
                 color='#f8fafc', fontsize=14, weight='bold', ha='right', va='center')
    ax_head.text(0.985, 0.25, "Normalización: Supremo Tricanal por Pulso  |  Filtro: 20-500 Hz + Notch 50 Hz",
                 color='#38bdf8', fontsize=10.5, ha='right', va='center')

    # SECCIÓN SUPERIOR: FILAS MUSCULARES DINÁMICAS (1, 2 o 3 canales activos)
    if n_activos == 1:
        y_start = 0.880
        h_fila = 0.350
        gap_fila = 0.0
    elif n_activos == 2:
        y_start = 0.940
        h_fila = 0.255
        gap_fila = 0.025
    else:
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

    for r_idx, cfg in enumerate(canales_activos):
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

        # Foto fiduciaria de colocación
        ax_foto = fig.add_axes([0.025 + w_meta + 0.005, y_base, w_foto, h_fila])
        ax_foto.set_facecolor('#151f32')
        ax_foto.set_xticks([])
        ax_foto.set_yticks([])
        for sp in ax_foto.spines.values():
            sp.set_color(c_card_border)
        ax_foto.set_title("Colocación Anatómica", color='#cbd5e1', fontsize=11, weight='bold', pad=4)

        foto_path = cfg.get('foto', '')
        if foto_path in ('FOTO_PERDIDA', 'imagen_perdida', 'perdida'):
            ax_foto.text(0.5, 0.58, "REGISTRO\nFOTOGRÁFICO", color='#f59e0b', fontsize=11, weight='bold', ha='center', va='center')
            ax_foto.text(0.5, 0.32, "NO DISPONIBLE", color='#94a3b8', fontsize=9.5, weight='bold', ha='center', va='center')
            ax_foto.text(0.5, 0.18, "(Imagen perdida)", color='#64748b', fontsize=8, weight='bold', ha='center', va='center')
        elif foto_path and os.path.exists(foto_path):
            im_f = mpimg.imread(foto_path)
            ax_foto.imshow(im_f)
        else:
            # Placeholder pulcro
            ax_foto.text(0.5, 0.55, "FOTO\nPENDIENTE", color='#64748b', fontsize=11, weight='bold', ha='center', va='center')
            ax_foto.text(0.5, 0.30, "DE CURACIÓN", color='#475569', fontsize=9, weight='bold', ha='center', va='center')

        # Tabla biométrica invariante
        ax_tabla = fig.add_axes([0.025 + w_meta + 0.005 + w_foto + 0.006, y_base, w_tabla, h_fila])
        ax_tabla.set_facecolor(c_card_bg)
        ax_tabla.set_xticks([])
        ax_tabla.set_yticks([])
        for sp in ax_tabla.spines.values():
            sp.set_color(c_card_border)

        filas_tabla = [
            ("PARÁMETRO BIOMÉTRICO", "VALOR / ESTIMACIÓN (± ERROR)", True),
            ("Dimensión Electrodo", cfg.get('tam_electrodo', '5 x 5 mm (±0.02 mm)'), False),
            ("Distancia Inter-electrodo (IED entre centros, estim.)", cfg.get('ied', 'Pendiente de curación'), False),
            ("Dist. estimada a Mentón (Gnathion)", cfg.get('dist_menton', 'Pendiente de curación'), False),
            ("Dist. estimada a Nariz (Pronasale)", cfg.get('dist_nariz', 'Pendiente de curación'), False),
            ("Dist. estimada a Oreja (Tragus)", cfg.get('dist_trago', 'Pendiente de curación'), False),
            ("Dist. estimada a Comisura Labial", cfg.get('dist_comisura', 'Pendiente de curación'), False),
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
                if "Pendiente" in val:
                    col_val = '#64748b'
                    w_val = 'normal'
                elif "Inter-electrodo" in lbl:
                    col_val = '#fbbf24'
                    w_val = 'bold'
                else:
                    col_val = '#ffffff'
                    w_val = 'bold'
                ax_tabla.text(0.025, y_row, lbl, color='#cbd5e1', fontsize=10.5, va='center', weight='bold')
                ax_tabla.text(0.61, y_row, val, color=col_val, fontsize=12.0 if w_val=='bold' else 10.5, weight=w_val, va='center')

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

            c_dict = curvas.get(col, {}).get(v)
            if c_dict is not None and len(c_dict.get('mean', [])) > 0:
                t = c_dict['t']
                y_c = c_dict['mean']
                s_c = c_dict['std']

                ax_v.plot(t, y_c, color=col_m, linewidth=2.3, alpha=0.95)
                ax_v.fill_between(t, np.maximum(0, y_c - s_c), y_c + s_c, color=col_m, alpha=0.20)

                idx_p = np.argmax(y_c)
                t_p = t[idx_p]
                val_p = y_c[idx_p]
                if val_p > 0.08:
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
            if r_idx == n_activos - 1:
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
    ax_b1.text(0.04, 0.45, f"• {info_cuaderno.get('sujeto', 'Candela')}", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.28, f"• {info_cuaderno.get('horario', 'N/A')}", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.04, 0.11, f"• {info_cuaderno.get('duracion', 'N/A')}", color='#ffffff', fontsize=11.5, weight='bold')

    p_a = FancyBboxPatch((0.34, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_a)
    ax_b1.text(0.36, 0.64, "HARDWARE", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.36, 0.45, f"• Fs: {info_cuaderno.get('fs', '6000 Hz')}", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.28, "• Baterías: 8.3V / 7.7V", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.36, 0.11, f"• Ref: {info_cuaderno.get('ref', 'Mastoides')}", color='#ffffff', fontsize=11.5, weight='bold')

    p_p = FancyBboxPatch((0.67, 0.08), 0.31, 0.70, boxstyle="round,pad=0.015", facecolor=c_inner_box, edgecolor=c_inner_border, lw=1.2)
    ax_b1.add_patch(p_p)
    ax_b1.text(0.69, 0.64, "REGISTROS", color='#38bdf8', fontsize=10.5, weight='bold')
    ax_b1.text(0.69, 0.45, f"• {len(mediciones_filtradas)} tomas grabadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.28, f"• {info_cuaderno.get('series_n', 'N/A')} series analizadas", color='#ffffff', fontsize=11.5, weight='bold')
    ax_b1.text(0.69, 0.11, f"• {len(y_clean)} pulsos válidos", color='#ffffff', fontsize=11.5, weight='bold')

    # Banner 2: LDA con Badges de Vocales
    ax_b2 = fig.add_axes([0.505, y_banner, 0.470, h_banner])
    ax_b2.set_facecolor(c_card_bg)
    ax_b2.set_xticks([])
    ax_b2.set_yticks([])
    for sp in ax_b2.spines.values():
        sp.set_color(c_card_border)
    ax_b2.text(0.03, 0.86, f"RENDIMIENTO CLASIFICADOR LINEAL (LDA) EN PCA 3D  |  GLOBAL: {acc_lda_global:.1f}%  (MACRO: {acc_lda_macro:.1f}%)", 
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
            idx_v = (y_clean == v)
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

    print(f"Guardando imagen en {png_out} (DPI=200)...")
    plt.savefig(png_out, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Guardando PDF en {pdf_out}...")
    plt.savefig(pdf_out, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Diapositiva generada exitosamente.")
