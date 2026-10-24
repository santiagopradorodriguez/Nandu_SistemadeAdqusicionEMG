# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación experimental de PCA incorporando la feature de energía
#              tricanal para desacoplar y separar los atractores de /e/ e /i/.
# ==============================================================================

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Semilla fija universal
np.random.seed(42)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

import deep_learning.pca_umap_clustering.generador_pca_umap as gpu

base_dir = os.path.join(emg_desarrollo, "base_de_datos_electrodos")
out_fig_dir = os.path.join(emg_desarrollo, "resultados", "resultados_autoencoder", "figuras_evaluacion")
os.makedirs(out_fig_dir, exist_ok=True)

COLORES_VOCALES = {
    'A': '#E63946', # Rojo
    'E': '#1F77B4', # Azul
    'I': '#2CA02C', # Verde
    'O': '#9D4EDD', # Morado
    'U': '#E7A61A'  # Amarillo
}

sesiones = [
    {
        "id": "Candela_Exp1",
        "nombre": "Candela - Exp 1: Masetero 24 mm",
        "tomas": [
            "2026-10-20/A_Exp1_Prueba1_Candela", "2026-10-20/A_Exp1_Prueba2_Candela", "2026-10-20/A_Exp1_Prueba3_Candela",
            "2026-10-20/E_Exp1_Prueba1_Candela", "2026-10-20/E_Exp1_Prueba2_Candela", "2026-10-20/E_Exp1_Prueba3_Candela",
            "2026-10-20/I_Exp1_Prueba1_Candela", "2026-10-20/I_Exp1_Prueba2_Candela", "2026-10-20/I_Exp1_Prueba3_Candela",
            "2026-10-20/O_Exp1_Prueba1_Candela", "2026-10-20/O_Exp1_Prueba2_Candela", "2026-10-20/O_Exp1_Prueba3_Candela", "2026-10-20/O_Exp1_Prueba4_Candela",
            "2026-10-20/U_Exp1_Prueba1_Candela", "2026-10-20/U_Exp1_Prueba2_Candela", "2026-10-20/U_Exp1_Prueba3_Candela", "2026-10-20/U_Exp1_Prueba4_Candela"
        ]
    },
    {
        "id": "Candela_Exp2",
        "nombre": "Candela - Exp 2: Masetero 23 mm",
        "tomas": [
            "2026-10-20/A_Exp2_Prueba1_Candela", "2026-10-20/A_Exp2_Prueba2_Candela", "2026-10-20/A_Exp2_Prueba3_Candela", "2026-10-20/A_Exp2_Prueba4_Candela",
            "2026-10-20/E_Exp2_Prueba1_Candela", "2026-10-20/E_Exp2_Prueba2_Candela", "2026-10-20/E_Exp2_Prueba3_Candela", "2026-10-20/E_Exp2_Prueba4_Candela",
            "2026-10-20/I_Exp2_Prueba1_Candela", "2026-10-20/I_Exp2_Prueba2_Candela", "2026-10-20/I_Exp2_Prueba3_Candela", "2026-10-20/I_Exp2_Prueba4_Candela",
            "2026-10-20/O_Exp2_Prueba1_Candela", "2026-10-20/O_Exp2_Prueba2_Candela", "2026-10-20/O_Exp2_Prueba3_Candela", "2026-10-20/O_Exp2_Prueba4_Candela", "2026-10-20/O_Exp2_Prueba5_Candela",
            "2026-10-20/U_Exp2_Prueba1_Candela", "2026-10-20/U_Exp2_Prueba2_Candela", "2026-10-20/U_Exp2_Prueba3_Candela", "2026-10-20/U_Exp2_Prueba4_Candela"
        ]
    }
]

def evaluar_condicion(X, Y):
    vocales = sorted(list(set(Y)))
    v2i = {v: i for i, v in enumerate(vocales)}
    y_true = np.array([v2i[v] for v in Y])
    
    # PCA 2D
    pca2 = PCA(n_components=2, random_state=42)
    X_2d = pca2.fit_transform(X)
    evr2 = pca2.explained_variance_ratio_ * 100.0
    
    # GMM 2D
    gmm2 = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    y_gmm2 = gmm2.fit_predict(X_2d)
    cm2 = confusion_matrix(y_true, y_gmm2)
    r2, c2 = linear_sum_assignment(-cm2)
    acc_gmm2 = float(cm2[r2, c2].sum() / len(y_true) * 100.0)
    
    # LDA 2D
    lda2 = LDA()
    lda2.fit(X_2d, y_true)
    acc_lda2 = float(lda2.score(X_2d, y_true) * 100.0)
    
    # PCA 3D
    pca3 = PCA(n_components=3, random_state=42)
    X_3d = pca3.fit_transform(X)
    evr3 = pca3.explained_variance_ratio_ * 100.0
    
    # GMM 3D
    gmm3 = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
    y_gmm3 = gmm3.fit_predict(X_3d)
    cm3 = confusion_matrix(y_true, y_gmm3)
    r3, c3 = linear_sum_assignment(-cm3)
    acc_gmm3 = float(cm3[r3, c3].sum() / len(y_true) * 100.0)
    
    # LDA 3D
    lda3 = LDA()
    lda3.fit(X_3d, y_true)
    acc_lda3 = float(lda3.score(X_3d, y_true) * 100.0)
    
    # Métricas específicas /e/ vs /i/
    idx_e, idx_i = v2i['E'], v2i['I']
    c_e_2d = np.mean(X_2d[y_true == idx_e], axis=0)
    c_i_2d = np.mean(X_2d[y_true == idx_i], axis=0)
    dist_ei_2d = float(np.linalg.norm(c_e_2d - c_i_2d))
    
    c_e_3d = np.mean(X_3d[y_true == idx_e], axis=0)
    c_i_3d = np.mean(X_3d[y_true == idx_i], axis=0)
    dist_ei_3d = float(np.linalg.norm(c_e_3d - c_i_3d))
    
    mask_ei = (y_true == idx_e) | (y_true == idx_i)
    lda_ei_2d = LDA().fit(X_2d[mask_ei], y_true[mask_ei])
    acc_ei_2d = float(lda_ei_2d.score(X_2d[mask_ei], y_true[mask_ei]) * 100.0)
    
    lda_ei_3d = LDA().fit(X_3d[mask_ei], y_true[mask_ei])
    acc_ei_3d = float(lda_ei_3d.score(X_3d[mask_ei], y_true[mask_ei]) * 100.0)
    
    return {
        "X_2d": X_2d,
        "evr2": evr2,
        "acc_gmm2": acc_gmm2,
        "acc_lda2": acc_lda2,
        "X_3d": X_3d,
        "evr3": evr3,
        "acc_gmm3": acc_gmm3,
        "acc_lda3": acc_lda3,
        "dist_ei_2d": dist_ei_2d,
        "dist_ei_3d": dist_ei_3d,
        "acc_ei_2d": acc_ei_2d,
        "acc_ei_3d": acc_ei_3d,
        "centroide_e_2d": c_e_2d.tolist(),
        "centroide_i_2d": c_i_2d.tolist()
    }

resultados_experimento = {}
datos_para_graficar = []

for ses_idx, ses in enumerate(sesiones):
    s_id = ses["id"]
    s_nom = ses["nombre"]
    tomas = ses["tomas"]
    
    print(f"\n=======================================================")
    print(f"Extrayendo y evaluando {s_nom}")
    print(f"=======================================================")
    
    X_raw, Y_list, Tomas_list, SNRs = gpu.extraer_features_concatenadas(
        base_dir, tomas,
        alpha_ruido=0.5,
        smooth_ms=90,
        notch_q=2.0,
        target_len=20,
        return_raw_cache=True,
        canales_features=['canal_0', 'canal_1', 'canal_2']
    )
    
    X_norm_list, E_log_list, M_sup_list = [], [], []
    for segs in X_raw:
        e_tot = sum(np.sum(s**2) for s in segs)
        m_sup = max(np.max(s) for s in segs) if segs else 1e-9
        vec = []
        for s in segs:
            s_norm = s / (m_sup + 1e-9)
            s_rs = gpu.resample(s_norm, 20)
            s_rs[s_rs < 0] = 0.0
            vec.append(s_rs)
        X_norm_list.append(np.concatenate(vec))
        E_log_list.append(np.log10(e_tot + 1e-9))
        M_sup_list.append(m_sup)
        
    X_base = np.array(X_norm_list) # (N, 60)
    E_feat = np.array(E_log_list)[:, None]
    # Escalar para equilibrar varianza relativa en el espacio de PCA
    E_scaled = (E_feat - np.mean(E_feat)) / (np.std(E_feat) + 1e-9) * np.std(X_base)
    X_with_energy = np.hstack([X_base, E_scaled])
    Y = np.array(Y_list)
    
    # 1. Evaluar Base (Sin Energía)
    res_base = evaluar_condicion(X_base, Y)
    
    # 2. Evaluar Con Energía Tricanal
    res_energy = evaluar_condicion(X_with_energy, Y)
    
    # Desglose de energía por vocal
    energia_stats = {}
    for v in ['A', 'E', 'I', 'O', 'U']:
        mask = Y == v
        energia_stats[v] = {
            "n_pulsos": int(np.sum(mask)),
            "e_media": float(np.mean(10**np.array(E_log_list)[mask])),
            "m_supremo_media": float(np.mean(np.array(M_sup_list)[mask])),
            "m_supremo_std": float(np.std(np.array(M_sup_list)[mask]))
        }
        
    resultados_experimento[s_id] = {
        "nombre": s_nom,
        "n_pulsos": len(Y),
        "energia_por_vocal": energia_stats,
        "base_sin_energia": {
            "gmm_2d": res_base["acc_gmm2"],
            "lda_2d": res_base["acc_lda2"],
            "gmm_3d": res_base["acc_gmm3"],
            "lda_3d": res_base["acc_lda3"],
            "distancia_ei_2d": res_base["dist_ei_2d"],
            "separacion_ei_lda_2d": res_base["acc_ei_2d"],
            "separacion_ei_lda_3d": res_base["acc_ei_3d"]
        },
        "con_feature_energia": {
            "gmm_2d": res_energy["acc_gmm2"],
            "lda_2d": res_energy["acc_lda2"],
            "gmm_3d": res_energy["acc_gmm3"],
            "lda_3d": res_energy["acc_lda3"],
            "distancia_ei_2d": res_energy["dist_ei_2d"],
            "separacion_ei_lda_2d": res_energy["acc_ei_2d"],
            "separacion_ei_lda_3d": res_energy["acc_ei_3d"]
        }
    }
    
    datos_para_graficar.append((s_nom, res_base, res_energy, Y))

# Creación explícita de la figura posterior a las extracciones (evita que plt.close() interno la destruya)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
plt.subplots_adjust(hspace=0.28, wspace=0.22)

for ses_idx, (s_nom, res_base, res_energy, Y) in enumerate(datos_para_graficar):
    # Graficar Columna 0: Base (Sin Energía)
    ax_b = axes[ses_idx, 0]
    X_2d_b = res_base["X_2d"]
    evr_b = res_base["evr2"]
    for v in ['A', 'O', 'U', 'E', 'I']:
        m = Y == v
        alpha_val = 0.85 if v in ['E', 'I'] else 0.35
        size_val = 45 if v in ['E', 'I'] else 25
        edge_c = 'black' if v in ['E', 'I'] else 'none'
        ax_b.scatter(X_2d_b[m, 0], X_2d_b[m, 1], c=COLORES_VOCALES[v], label=f"/{v.lower()}/", alpha=alpha_val, s=size_val, edgecolors=edge_c, linewidth=0.6)
    
    # Línea entre centroides /e/ e /i/
    ce_b = res_base["centroide_e_2d"]
    ci_b = res_base["centroide_i_2d"]
    ax_b.plot([ce_b[0], ci_b[0]], [ce_b[1], ci_b[1]], 'k--', linewidth=2.0, alpha=0.9, label=f"d(e,i) = {res_base['dist_ei_2d']:.3f}")
    ax_b.scatter([ce_b[0], ci_b[0]], [ce_b[1], ci_b[1]], c=['#1F77B4', '#2CA02C'], s=120, edgecolors='black', linewidth=1.5, zorder=5)
    
    ax_b.set_title(f"{s_nom} - Base: Sin Energía (GMM: {res_base['acc_gmm2']:.1f}% | LDA: {res_base['acc_lda2']:.1f}%)", fontsize=11, fontweight="bold")
    ax_b.set_xlabel(f"PC1 ({evr_b[0]:.1f}%)", fontsize=10)
    ax_b.set_ylabel(f"PC2 ({evr_b[1]:.1f}%)", fontsize=10)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)
    
    # Graficar Columna 1: Con Feature de Energía Tricanal
    ax_e = axes[ses_idx, 1]
    X_2d_e = res_energy["X_2d"]
    evr_e = res_energy["evr2"]
    for v in ['A', 'O', 'U', 'E', 'I']:
        m = Y == v
        alpha_val = 0.85 if v in ['E', 'I'] else 0.35
        size_val = 45 if v in ['E', 'I'] else 25
        edge_c = 'black' if v in ['E', 'I'] else 'none'
        ax_e.scatter(X_2d_e[m, 0], X_2d_e[m, 1], c=COLORES_VOCALES[v], label=f"/{v.lower()}/", alpha=alpha_val, s=size_val, edgecolors=edge_c, linewidth=0.6)
        
    ce_e = res_energy["centroide_e_2d"]
    ci_e = res_energy["centroide_i_2d"]
    ax_e.plot([ce_e[0], ci_e[0]], [ce_e[1], ci_e[1]], 'k--', linewidth=2.0, alpha=0.9, label=f"d(e,i) = {res_energy['dist_ei_2d']:.3f} (+{(res_energy['dist_ei_2d']/res_base['dist_ei_2d']-1)*100:.1f}%)")
    ax_e.scatter([ce_e[0], ci_e[0]], [ce_e[1], ci_e[1]], c=['#1F77B4', '#2CA02C'], s=120, edgecolors='black', linewidth=1.5, zorder=5)
    
    ax_e.set_title(f"{s_nom} - Con Energía Tricanal (GMM: {res_energy['acc_gmm2']:.1f}% | LDA: {res_energy['acc_lda2']:.1f}%)", fontsize=11, fontweight="bold")
    ax_e.set_xlabel(f"PC1 ({evr_e[0]:.1f}%)", fontsize=10)
    ax_e.set_ylabel(f"PC2 ({evr_e[1]:.1f}%)", fontsize=10)
    ax_e.grid(True, linestyle=":", alpha=0.6)
    ax_e.legend(loc="upper right", fontsize=8, framealpha=0.9)

fig_out = os.path.join(out_fig_dir, "comparativa_pca_feature_energia_tricanal_2026_10_20.png")
fig.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close(fig)

json_out = os.path.join(out_fig_dir, "metricas_pca_feature_energia_tricanal_2026_10_20.json")
with open(json_out, "w") as f:
    json.dump(resultados_experimento, f, indent=4)

print(f"\n[OK] Figura guardada en: {fig_out}")
print(f"[OK] Métricas guardadas en: {json_out}")
