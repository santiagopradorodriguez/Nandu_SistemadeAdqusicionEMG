# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Barrido comparativo de sustracción de ruido basal (alpha = 0.0 vs 0.5 vs 1.0)
#              en PCA 2D/3D e Isomap 2D sobre las mediciones del 2026-10-20.
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Semilla Universal
np.random.seed(42)

# Rutas
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
    "2026-10-20/U_Prueba1_Santi", "2026-10-20/U_Prueba2_Santi", "2026-10-20/U_Prueba3_Santi", "2026-10-20/U_Prueba4_Santi"
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

resultados_alpha = {}

for alpha_val in [0.0, 1.0]:
    alpha_tag = f"alpha_{alpha_val:.1f}".replace(".", "_")
    print(f"\n==========================================================================")
    print(f"PROCESANDO SESIONES CON ALPHA DE RUIDO = {alpha_val}")
    print(f"==========================================================================")
    
    for tag_sesion, meds in casos:
        dir_out = os.path.join(pca_root, f"2026-10-20_{tag_sesion}_{alpha_tag}")
        os.makedirs(dir_out, exist_ok=True)
        
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
                "alpha_ruido": alpha_val,
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
                "alpha_ruido": alpha_val,
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
            "correccion_impedancia": True,
            "ignorar_ventana_cero": False,
            "pre_pct": 0.4,
            "post_pct": 0.6,
            "modo_alineacion": "Pico Volumen Micrófono",
            "estilo_visual": "Fronteras",
            "canales_features": ["canal_1", "canal_2", "canal_0"] if "Santi" in tag_sesion else ["canal_0", "canal_1", "canal_2"]
        }
        
        try:
            generador.ejecutar_procesamiento(mediciones=meds, base_dir=base_dir, out_dir=dir_out, **cfg)
            print(f"[OK] {tag_sesion} procesado para alpha={alpha_val}")
        except Exception as e:
            print(f"[ERROR] Error procesando {tag_sesion} para alpha={alpha_val}: {e}")

# Ahora evaluar métricas cuantitativas para alpha 0.0, 0.5 y 1.0
print("\n" + "=" * 80)
print("EVALUACIÓN COMPARATIVA GLOBAL DE ALPHA (0.0 vs 0.5 vs 1.0)")
print("=" * 80)

res_final = {}
for tag_sesion, _ in casos:
    res_final[tag_sesion] = {}
    for a in [0.0, 0.5, 1.0]:
        a_key = f"alpha_{a:.1f}"
        if a == 0.5:
            d_path = os.path.join(pca_root, f"2026-10-20_{tag_sesion}")
        else:
            d_path = os.path.join(pca_root, f"2026-10-20_{tag_sesion}_alpha_{a:.1f}".replace(".", "_"))
            
        csv_file = os.path.join(d_path, "caracteristicas_exportadas.csv")
        if not os.path.exists(csv_file):
            print(f"Falta CSV en {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        df = df[df['Vocal'].isin(VOCALES)].copy()
        y = df['Vocal'].values
        feat_cols = [c for c in df.columns if c.startswith('Ch')]
        X = df[feat_cols].to_numpy().copy()
        
        # 1. Base (Masetero x1.0)
        pca2 = PCA(n_components=2, random_state=42)
        Z_pca2 = pca2.fit_transform(X)
        gmm_p2, lda_p2, desg_p2, _ = evaluar_modelo(Z_pca2, y)
        
        pca3 = PCA(n_components=3, random_state=42)
        Z_pca3 = pca3.fit_transform(X)
        gmm_p3, lda_p3, desg_p3, _ = evaluar_modelo(Z_pca3, y)
        
        iso = Isomap(n_neighbors=20, n_components=2)
        Z_iso = iso.fit_transform(X)
        gmm_iso, lda_iso, desg_iso, _ = evaluar_modelo(Z_iso, y)
        
        # 2. Masetero x2.0
        X_x2 = X.copy()
        X_x2[:, 40:60] *= 2.0
        Z_p2_x2 = PCA(n_components=2, random_state=42).fit_transform(X_x2)
        gmm_p2_x2, lda_p2_x2, _, _ = evaluar_modelo(Z_p2_x2, y)
        
        Z_p3_x2 = PCA(n_components=3, random_state=42).fit_transform(X_x2)
        gmm_p3_x2, lda_p3_x2, desg_p3_x2, _ = evaluar_modelo(Z_p3_x2, y)
        
        Z_iso_x2 = Isomap(n_neighbors=20, n_components=2).fit_transform(X_x2)
        gmm_iso_x2, lda_iso_x2, _, _ = evaluar_modelo(Z_iso_x2, y)
        
        res_final[tag_sesion][a_key] = {
            "N": len(df),
            "PCA_2D_GMM": gmm_p2, "PCA_2D_LDA": lda_p2,
            "PCA_3D_GMM": gmm_p3, "PCA_3D_LDA": lda_p3,
            "Isomap_GMM": gmm_iso, "Isomap_LDA": lda_iso,
            "PCA_2D_GMM_Masetero_x2": gmm_p2_x2,
            "PCA_3D_GMM_Masetero_x2": gmm_p3_x2,
            "PCA_3D_LDA_Masetero_x2": lda_p3_x2,
            "Isomap_LDA_Masetero_x2": lda_iso_x2,
            "Desglose_PCA_3D_x2": desg_p3_x2
        }

json_alpha = os.path.join(out_fig_dir, "metricas_barrido_alpha_2026_10_20.json")
with open(json_alpha, "w", encoding="utf-8") as f:
    json.dump(res_final, f, indent=4, ensure_ascii=False)

print("\nMétricas exportadas a:", json_alpha)
print("BARRIDO DE ALPHA COMPLETADO CON ÉXITO.")
