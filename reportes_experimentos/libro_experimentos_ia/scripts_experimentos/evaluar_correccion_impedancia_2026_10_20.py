# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación comparativa del impacto de la corrección de impedancia
#              (Con vs Sin corrección por reposo basal y P95) en PCA 2D/3D e Isomap 2D
#              sobre las mediciones del 2026-10-20.
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Semilla universal obligatoria
np.random.seed(42)

# Rutas del repositorio
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

base_dir = os.path.join(emg_desarrollo, "base_de_datos_electrodos")
pca_root = os.path.join(emg_desarrollo, "deep_learning", "pca_umap_clustering", "resultados_pca_umap")
out_fig_dir = os.path.join(emg_desarrollo, "resultados", "resultados_autoencoder", "figuras_evaluacion")
os.makedirs(out_fig_dir, exist_ok=True)

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

mediciones_candela_exp1 = [
    "2026-10-20/A_Exp1_Prueba1_Candela", "2026-10-20/A_Exp1_Prueba2_Candela", "2026-10-20/A_Exp1_Prueba3_Candela",
    "2026-10-20/E_Exp1_Prueba1_Candela", "2026-10-20/E_Exp1_Prueba2_Candela", "2026-10-20/E_Exp1_Prueba3_Candela",
    "2026-10-20/I_Exp1_Prueba1_Candela", "2026-10-20/I_Exp1_Prueba2_Candela", "2026-10-20/I_Exp1_Prueba3_Candela",
    "2026-10-20/O_Exp1_Prueba1_Candela", "2026-10-20/O_Exp1_Prueba2_Candela", "2026-10-20/O_Exp1_Prueba3_Candela", "2026-10-20/O_Exp1_Prueba4_Candela",
    "2026-10-20/U_Exp1_Prueba1_Candela", "2026-10-20/U_Exp1_Prueba2_Candela", "2026-10-20/U_Exp1_Prueba3_Candela", "2026-10-20/U_Exp1_Prueba4_Candela"
]

mediciones_candela_exp2 = [
    "2026-10-20/A_Exp2_Prueba1_Candela", "2026-10-20/A_Exp2_Prueba2_Candela", "2026-10-20/A_Exp2_Prueba3_Candela", "2026-10-20/A_Exp2_Prueba4_Candela",
    "2026-10-20/E_Exp2_Prueba1_Candela", "2026-10-20/E_Exp2_Prueba2_Candela", "2026-10-20/E_Exp2_Prueba3_Candela", "2026-10-20/E_Exp2_Prueba4_Candela",
    "2026-10-20/I_Exp2_Prueba1_Candela", "2026-10-20/I_Exp2_Prueba2_Candela", "2026-10-20/I_Exp2_Prueba3_Candela", "2026-10-20/I_Exp2_Prueba4_Candela",
    "2026-10-20/O_Exp2_Prueba1_Candela", "2026-10-20/O_Exp2_Prueba2_Candela", "2026-10-20/O_Exp2_Prueba3_Candela", "2026-10-20/O_Exp2_Prueba4_Candela", "2026-10-20/O_Exp2_Prueba5_Candela",
    "2026-10-20/U_Exp2_Prueba1_Candela", "2026-10-20/U_Exp2_Prueba2_Candela", "2026-10-20/U_Exp2_Prueba3_Candela", "2026-10-20/U_Exp2_Prueba4_Candela"
]

mediciones_santi = [
    # Excluidas por falla técnica / señal nula (< 16.5 uV):
    # "2026-10-20/A_Prueba1_Santi", "2026-10-20/A_Prueba2_Santi",
    "2026-10-20/A_Prueba3_Santi", "2026-10-20/A_Prueba4_Santi", "2026-10-20/A_Prueba5_Santi",
    "2026-10-20/E_Prueba1_Santi", "2026-10-20/E_Prueba2_Santi", "2026-10-20/E_Prueba3_Santi", "2026-10-20/E_Prueba4_Santi",
    "2026-10-20/I_Prueba1_Santi", "2026-10-20/I_Prueba2_Santi", "2026-10-20/I_Prueba3_Santi", "2026-10-20/I_Prueba4_Santi",
    "2026-10-20/O_Prueba1_Santi", "2026-10-20/O_Prueba2_Santi", "2026-10-20/O_Prueba3_Santi", "2026-10-20/O_Prueba4_Santi", "2026-10-20/O_Prueba5_Santi",
    "2026-10-20/U_Prueba1_Santi", "2026-10-20/U_Prueba2_Santi", "2026-10-20/U_Prueba3_Santi", "2026-10-20/U_Prueba4_Santi",
    "2026-10-20/ma_Prueba1_Santi", "2026-10-20/pa_Prueba1_Santi"
]

casos = [
    ("Candela_Exp1", mediciones_candela_exp1),
    ("Candela_Exp2", mediciones_candela_exp2),
    ("Santi_Exploracion", mediciones_santi)
]

def evaluar_modelo(Z, y_true):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=5)
    pred_raw = gmm.fit_predict(Z)
    contingency = np.zeros((5, 5))
    for i, vl in enumerate(VOCALES):
        for j in range(5):
            contingency[j, i] = np.sum((y_true == vl) & (pred_raw == j))
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    pred_idx = np.array([cluster_to_vocal[c] for c in pred_raw])
    y_idx = np.array([VOCAL_TO_IDX[v] for v in y_true])
    acc_gmm = accuracy_score(y_idx, pred_idx) * 100.0
    desg_gmm = {v: float(np.mean(pred_idx[y_true == v] == VOCAL_TO_IDX[v]) * 100.0) for v in VOCALES}
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preds_lda = np.zeros_like(y_idx)
    for tr, te in skf.split(Z, y_idx):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Z[tr], y_idx[tr])
        preds_lda[te] = lda.predict(Z[te])
    acc_lda = accuracy_score(y_idx, preds_lda) * 100.0
    desg_lda = {v: float(np.mean(preds_lda[y_true == v] == VOCAL_TO_IDX[v]) * 100.0) for v in VOCALES}
    return acc_gmm, acc_lda, desg_gmm, desg_lda

# 1. Ejecutar procesamiento para correccion_impedancia = False
for tag_sesion, meds in casos:
    dir_out_sin = os.path.join(pca_root, f"2026-10-20_{tag_sesion}_sin_impedancia")
    os.makedirs(dir_out_sin, exist_ok=True)
    
    cfg = {
        "proc_pca_2d": True,
        "proc_pca_3d": True,
        "proc_umap_2d": False,
        "proc_umap_3d": False,
        "ocultar_leyenda": False,
        "tipo_filtro_ruido": "notch",
        "notch_q": 2.0,
        "highpass_cutoff_hz": 20.0,
        "lowpass_cutoff_hz": 300.0,
        "params_2d": {
            "alpha_ruido": 0.5,
            "gate_ratio_ruido": 0.0,
            "smooth_ms": 90,
            "target_length": 20,
            "snr_threshold": 0.5,
            "outlier_contamination": 0.05,
            "notch_q": 2.0,
            "tipo_filtro_ruido": "notch",
            "highpass_cutoff_hz": 20.0,
            "lowpass_cutoff_hz": 300.0,
            "comp_x": "PC1",
            "comp_y": "PC2",
            "pesos_canales": [1.0, 1.0, 1.0]
        },
        "params_3d": {
            "alpha_ruido": 0.5,
            "gate_ratio_ruido": 0.0,
            "smooth_ms": 125,
            "target_length": 20,
            "snr_threshold": 0.5,
            "outlier_contamination": 0.1,
            "notch_q": 2.0,
            "tipo_filtro_ruido": "notch",
            "highpass_cutoff_hz": 20.0,
            "lowpass_cutoff_hz": 300.0,
            "comp_x": "PC1",
            "comp_y": "PC2",
            "comp_z": "PC3",
            "pesos_canales": [1.0, 1.0, 1.0]
        },
        "algoritmo_clustering_pca": "GMM",
        "algoritmo_clustering_umap": "K-Means",
        "aplicar_trevisan": False,
        "aplicar_correccion_intersesion": False,
        "correccion_impedancia": False,
        "ignorar_ventana_cero": False,
        "pre_pct": 0.4,
        "post_pct": 0.6,
        "modo_alineacion": "Pico Volumen Micrófono",
        "estilo_visual": "Fronteras",
        "canales_features": ["canal_1", "canal_2", "canal_0"] if "Santi" in tag_sesion else ["canal_0", "canal_1", "canal_2"]
    }
    
    csv_test = os.path.join(dir_out_sin, "caracteristicas_exportadas.csv")
    if not os.path.exists(csv_test):
        print(f"Procesando {tag_sesion} SIN corrección de impedancia...")
        generador.ejecutar_procesamiento(mediciones=meds, base_dir=base_dir, out_dir=dir_out_sin, **cfg)
    else:
        print(f"[CACHE] {csv_test} ya existe. Omitiendo reprocesamiento.")

# 2. Extraer datos y comparar con la sesión base (CON impedancia)
resultados_comparativa = {}

for tag_sesion, _ in casos:
    resultados_comparativa[tag_sesion] = {}
    
    # Ruta CON impedancia (original)
    csv_con = os.path.join(pca_root, f"2026-10-20_{tag_sesion}", "caracteristicas_exportadas.csv")
    # Ruta SIN impedancia
    csv_sin = os.path.join(pca_root, f"2026-10-20_{tag_sesion}_sin_impedancia", "caracteristicas_exportadas.csv")
    
    for modo, csv_path in [("con_impedancia", csv_con), ("sin_impedancia", csv_sin)]:
        df = pd.read_csv(csv_path)
        vcol = 'Vocal' if 'Vocal' in df.columns else 'vocal'
        tcol = 'Toma' if 'Toma' in df.columns else 'toma'
        y = df[vcol].values
        feat_cols = [c for c in df.columns if c not in [vcol, 'sesion', tcol]]
        X = df[feat_cols].values
        
        # PCA 2D
        pca2 = PCA(n_components=2, random_state=42)
        Z_pca2 = pca2.fit_transform(X)
        gmm_pca2, lda_pca2, desg_gmm_pca2, desg_lda_pca2 = evaluar_modelo(Z_pca2, y)
        
        # PCA 3D
        pca3 = PCA(n_components=3, random_state=42)
        Z_pca3 = pca3.fit_transform(X)
        gmm_pca3, lda_pca3, desg_gmm_pca3, desg_lda_pca3 = evaluar_modelo(Z_pca3, y)
        
        # Isomap 2D (k=20)
        iso = Isomap(n_neighbors=20, n_components=2)
        Z_iso = iso.fit_transform(X)
        gmm_iso, lda_iso, desg_gmm_iso, desg_lda_iso = evaluar_modelo(Z_iso, y)
        
        resultados_comparativa[tag_sesion][modo] = {
            "N": len(y),
            "PCA_2D_GMM": gmm_pca2,
            "PCA_2D_LDA": lda_pca2,
            "PCA_3D_GMM": gmm_pca3,
            "PCA_3D_LDA": lda_pca3,
            "Isomap_GMM": gmm_iso,
            "Isomap_LDA": lda_iso,
            "Desglose_PCA_2D_GMM": desg_gmm_pca2,
            "Desglose_PCA_2D_LDA": desg_lda_pca2
        }

# Guardar JSON con métricas
json_out = os.path.join(out_fig_dir, "metricas_correccion_impedancia_2026_10_20.json")
with open(json_out, "w") as f:
    json.dump(resultados_comparativa, f, indent=4)
print(f"Métricas guardadas en: {json_out}")

# 3. Graficar Comparativa Visual 3x2 (Filas: Candela Exp 1, Exp 2, Santi | Columnas: Sin Impedancia vs Con Impedancia)
fig, axes = plt.subplots(3, 2, figsize=(14, 18), dpi=200)

nombres_filas = [
    "Candela Exp 1 (Masetero 2.4 mm)",
    "Candela Exp 2 (Masetero 23 mm)",
    "Santiago Exploración (Auto-colocación)"
]

for row_idx, (tag_sesion, _) in enumerate(casos):
    csv_sin = os.path.join(pca_root, f"2026-10-20_{tag_sesion}_sin_impedancia", "caracteristicas_exportadas.csv")
    csv_con = os.path.join(pca_root, f"2026-10-20_{tag_sesion}", "caracteristicas_exportadas.csv")
    
    for col_idx, (modo_label, csv_p) in enumerate([("Sin Corrección de Impedancia", csv_sin), ("Con Corrección de Impedancia (P95/Basal)", csv_con)]):
        ax = axes[row_idx, col_idx]
        df = pd.read_csv(csv_p)
        vcol = 'Vocal' if 'Vocal' in df.columns else 'vocal'
        tcol = 'Toma' if 'Toma' in df.columns else 'toma'
        y = df[vcol].values
        feat_cols = [c for c in df.columns if c not in [vcol, 'sesion', tcol]]
        X = df[feat_cols].values
        
        pca2 = PCA(n_components=2, random_state=42)
        Z = pca2.fit_transform(X)
        
        modo_key = "sin_impedancia" if col_idx == 0 else "con_impedancia"
        met = resultados_comparativa[tag_sesion][modo_key]
        
        for vl in VOCALES:
            mask = (y == vl)
            if np.sum(mask) > 0:
                ax.scatter(Z[mask, 0], Z[mask, 1], c=COLORES_VOCALES[vl], label=f"/{vl.lower()}/", alpha=0.75, s=30, edgecolors='none')
        
        ax.set_title(f"{nombres_filas[row_idx]} - {modo_label}\nGMM: {met['PCA_2D_GMM']:.1f}% | LDA CV: {met['PCA_2D_LDA']:.1f}%", fontsize=11, fontweight='bold')
        ax.set_xlabel("PC1", fontsize=9)
        ax.set_ylabel("PC2", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.3)
        if row_idx == 0 and col_idx == 1:
            ax.legend(loc='upper right', frameon=True, fontsize=8)

plt.suptitle("Impacto de la Corrección de Impedancia Basal y P95 en PCA 2D", fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
fig_out = os.path.join(out_fig_dir, "comparativa_correccion_impedancia_2026_10_20.png")
plt.savefig(fig_out, bbox_inches='tight')
plt.close()
print(f"Figura comparativa guardada en: {fig_out}")
