# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación del impacto de amplificar el canal del Masetero (Ch2 x2 y x3)
#              en PCA 2D e Isomap 2D sobre las mediciones del 2026-10-20.
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Semilla Universal
np.random.seed(42)

# Colores Oficiales
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}
VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}

# Rutas
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
pca_root = os.path.join(emg_desarrollo, "deep_learning", "pca_umap_clustering", "resultados_pca_umap")
out_fig_dir = os.path.join(emg_desarrollo, "resultados", "resultados_autoencoder", "figuras_evaluacion")
os.makedirs(out_fig_dir, exist_ok=True)

def evaluar_gmm_hungaro(Z, y_true):
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
    
    acc_global = accuracy_score(y_idx, pred_idx) * 100.0
    desglose = {}
    for v in VOCALES:
        mask = (y_true == v)
        if np.sum(mask) > 0:
            desglose[v] = np.mean(pred_idx[mask] == VOCAL_TO_IDX[v]) * 100.0
        else:
            desglose[v] = 0.0
    return acc_global, desglose, pred_raw

def evaluar_lda_cv(Z, y_true):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_idx = np.array([VOCAL_TO_IDX[v] for v in y_true])
    preds = np.zeros_like(y_idx)
    
    for train_idx, test_idx in skf.split(Z, y_idx):
        lda = LinearDiscriminantAnalysis()
        lda.fit(Z[train_idx], y_idx[train_idx])
        preds[test_idx] = lda.predict(Z[test_idx])
        
    acc = accuracy_score(y_idx, preds) * 100.0
    desglose = {}
    for v in VOCALES:
        mask = (y_true == v)
        desglose[v] = np.mean(preds[mask] == VOCAL_TO_IDX[v]) * 100.0
    return acc, desglose

sesiones = [
    ("Candela_Exp1", "2026-10-20_Candela_Exp1", "Candela - Exp 1 (Masetero 2.4 mm)"),
    ("Candela_Exp2", "2026-10-20_Candela_Exp2", "Candela - Exp 2 (Masetero 23 mm)"),
    ("Santi_Exploracion", "2026-10-20_Santi_Exploracion", "Santi - Exploración (Auto-colocación)")
]

multiplicadores = [1.0, 2.0, 3.0]
k_fijo = 20

resultados_amplificacion = {}

print("==========================================================================")
print("EVALUANDO AMPLIFICACIÓN DE MASETERO (Ch2 x1.0, x2.0, x3.0) EN PCA E ISOMAP")
print(f"Isomap con k fijo = {k_fijo}")
print("==========================================================================")

# Figura comparativa: 3 sesiones x 3 multiplicadores (mostrando Isomap con k fijo)
fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=300)
fig.patch.set_facecolor('#0B0C10')

for row_idx, (tag, folder_name, titulo_sesion) in enumerate(sesiones):
    csv_path = os.path.join(pca_root, folder_name, "caracteristicas_exportadas.csv")
    df = pd.read_csv(csv_path)
    df = df[df['Vocal'].isin(VOCALES)].copy()
    
    # Excluir tomas fallidas de Santi si existieran
    if "Santi" in tag and 'Toma' in df.columns:
        df = df[~df['Toma'].astype(str).str.contains('A_Prueba1_Santi|A_Prueba2_Santi|A_Prueba1_santi|A_Prueba2_santi', case=False, na=False)].copy()
        
    y_labels = df['Vocal'].values
    feat_cols = [c for c in df.columns if c.startswith('Ch')]
    X_base = df[feat_cols].values # (N, 60) -> 0:20 Ch0, 20:40 Ch1, 40:60 Ch2 (Masetero)
    
    # Asegurar permutación [1, 2, 0] para restituir orden canónico [orbi, nudo, masetero]
    if "Santi" in tag:
        param_file = os.path.join(pca_root, folder_name, "parametros.json")
        ya_permutado = False
        if os.path.exists(param_file):
            try:
                with open(param_file, 'r', encoding='utf-8') as pf:
                    p_data = json.load(pf)
                    if p_data.get("canales_features") == ["canal_1", "canal_2", "canal_0"]:
                        ya_permutado = True
            except Exception:
                pass
        if not ya_permutado:
            X_base = np.concatenate([X_base[:, 20:40], X_base[:, 40:60], X_base[:, 0:20]], axis=1)
    
    resultados_amplificacion[tag] = {}
    
    for col_idx, mult in enumerate(multiplicadores):
        X_mod = X_base.copy()
        # Canal 2 es Masetero (índices 40 a 60)
        X_mod[:, 40:60] *= mult
        
        # 1. PCA 2D
        pca = PCA(n_components=2, random_state=42)
        Z_pca = pca.fit_transform(X_mod)
        acc_pca_gmm, desg_pca_gmm, _ = evaluar_gmm_hungaro(Z_pca, y_labels)
        acc_pca_lda, desg_pca_lda = evaluar_lda_cv(Z_pca, y_labels)
        
        # 2. Isomap 2D (con mismo k)
        iso = Isomap(n_neighbors=k_fijo, n_components=2)
        Z_iso = iso.fit_transform(X_mod)
        acc_iso_gmm, desg_iso_gmm, _ = evaluar_gmm_hungaro(Z_iso, y_labels)
        acc_iso_lda, desg_iso_lda = evaluar_lda_cv(Z_iso, y_labels)
        
        resultados_amplificacion[tag][f"x{mult:.1f}"] = {
            "PCA_GMM": acc_pca_gmm,
            "PCA_LDA": acc_pca_lda,
            "Isomap_GMM": acc_iso_gmm,
            "Isomap_LDA": acc_iso_lda,
            "Desglose_PCA": desg_pca_gmm,
            "Desglose_Isomap": desg_iso_gmm
        }
        
        # Graficar proyección Isomap (k=20) para cada multiplicador
        ax = axes[row_idx, col_idx]
        ax.set_facecolor('#1F2833')
        for v in VOCALES:
            m = (y_labels == v)
            ax.scatter(Z_iso[m, 0], Z_iso[m, 1], c=COLORES_VOCALES[v], label=f'/{v.lower()}/', alpha=0.75, edgecolors='none', s=35)
            
        ax.set_title(
            f"{titulo_sesion} | Masetero x{mult:.0f}\nPCA: {acc_pca_gmm:.1f}% | Isomap (k={k_fijo}): {acc_iso_gmm:.1f}%",
            color='white', fontsize=11, fontweight='bold'
        )
        ax.grid(True, linestyle='--', alpha=0.3, color='#C5C6C7')
        ax.tick_params(colors='white')

axes[0, 0].legend(loc='upper right', framealpha=0.8, facecolor='#0B0C10', labelcolor='white')
plt.tight_layout()

fig_path = os.path.join(out_fig_dir, "comparativa_masetero_amplificado_2026_10_20.png")
plt.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# Exportar JSON de métricas
json_path = os.path.join(out_fig_dir, "metricas_masetero_amplificado_2026_10_20.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultados_amplificacion, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 80)
print(f"RESUMEN DETALLADO: IMPACTO DE AMPLIFICAR MASETERO (k={k_fijo} en Isomap)")
print("=" * 80)
for tag, mult_dict in resultados_amplificacion.items():
    print(f"\n[{tag}]:")
    for mult_key, m in mult_dict.items():
        print(f"  {mult_key}: PCA 2D (GMM={m['PCA_GMM']:.2f}%, LDA={m['PCA_LDA']:.2f}%) | Isomap 2D (GMM={m['Isomap_GMM']:.2f}%, LDA={m['Isomap_LDA']:.2f}%)")

print("\nFigura guardada en:", fig_path)
print("Métricas guardadas en:", json_path)
