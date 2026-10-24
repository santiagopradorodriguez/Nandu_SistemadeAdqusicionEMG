# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Evaluación de no linealidad (Isomap 2D/3D y Autoencoder Conv Ortogonal 2D)
#              sobre las mediciones del 2026-10-20 (Candela Exp 1, Exp 2 y Santi).
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

# Fijar semilla universal
torch.manual_seed(42)
np.random.seed(42)

# Colores Oficiales del Proyecto
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}
VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}

# Rutas del proyecto
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from grid_search_conv_ortogonal import ParametricConvOrthogonalAE

# Directorios de datos
pca_root = os.path.join(emg_desarrollo, "deep_learning", "pca_umap_clustering", "resultados_pca_umap")
out_fig_dir = os.path.join(emg_desarrollo, "resultados", "resultados_autoencoder", "figuras_evaluacion")
os.makedirs(out_fig_dir, exist_ok=True)

# Cargar Modelo Campeón Convolucional Ortogonal 2D
p_ckpt = os.path.join(emg_desarrollo, "resultados", "grid_search_conv_ortogonal", "modelo_campeon_conv_ortogonal.pt")
ckpt = torch.load(p_ckpt, map_location='cpu', weights_only=False)
model_conv = ParametricConvOrthogonalAE(
    in_channels=3, time_pts=20, conv_channels=(6, 12),
    kernel_size=5, latent_dim=2, act_name='tanh'
)
model_conv.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt, strict=False)
model_conv.eval()

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

resultados_totales = {}

fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=300)
fig.patch.set_facecolor('#0B0C10')

for row_idx, (tag, folder_name, titulo_sesion) in enumerate(sesiones):
    csv_path = os.path.join(pca_root, folder_name, "caracteristicas_exportadas.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] No se encontró {csv_path}")
        continue
        
    df = pd.read_csv(csv_path)
    # Filtrar solo vocales estándar
    df = df[df['Vocal'].isin(VOCALES)].copy()
    
    # Excluir tomas fallidas de Santi si existieran en el CSV
    if "Santi" in tag and 'Toma' in df.columns:
        df = df[~df['Toma'].astype(str).str.contains('A_Prueba1_Santi|A_Prueba2_Santi|A_Prueba1_santi|A_Prueba2_santi', case=False, na=False)].copy()
        
    y_labels = df['Vocal'].values
    
    feat_cols = [c for c in df.columns if c.startswith('Ch')]
    X_raw = df[feat_cols].values # (N, 60)
    
    # Verificar si el CSV requiere permutación canónica para Santi [1, 2, 0]
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
            # Permutación [1, 2, 0]: nuevo Ch0=orbi (col 20:40), Ch1=nudo (col 40:60), Ch2=mast (col 0:20)
            X_raw = np.concatenate([X_raw[:, 20:40], X_raw[:, 40:60], X_raw[:, 0:20]], axis=1)
    
    # 1. PCA Baseline (de las características proyectadas)
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2, random_state=42)
    Z_pca = pca_2d.fit_transform(X_raw)
    acc_pca_gmm, desg_pca_gmm, _ = evaluar_gmm_hungaro(Z_pca, y_labels)
    acc_pca_lda, desg_pca_lda = evaluar_lda_cv(Z_pca, y_labels)
    
    # 2. Isomap 2D No Lineal (barrido de k vecinos)
    best_k = 15
    best_iso_acc = -1
    best_Z_iso = None
    for k in [10, 15, 20, 25, 30]:
        iso = Isomap(n_neighbors=k, n_components=2)
        Z_iso_cand = iso.fit_transform(X_raw)
        acc_cand, _, _ = evaluar_gmm_hungaro(Z_iso_cand, y_labels)
        if acc_cand > best_iso_acc:
            best_iso_acc = acc_cand
            best_k = k
            best_Z_iso = Z_iso_cand
            
    acc_iso_gmm, desg_iso_gmm, _ = evaluar_gmm_hungaro(best_Z_iso, y_labels)
    acc_iso_lda, desg_iso_lda = evaluar_lda_cv(best_Z_iso, y_labels)
    
    # 3. Autoencoder Convolucional Ortogonal 2D Récord
    # Tensor (N, 3, 20)
    X_tensor = np.stack([X_raw[:, 0:20], X_raw[:, 20:40], X_raw[:, 40:60]], axis=1)
    X_t = torch.tensor(X_tensor, dtype=torch.float32)
    with torch.no_grad():
        Z_conv = model_conv.encode(X_t).numpy()
        
    acc_conv_gmm, desg_conv_gmm, _ = evaluar_gmm_hungaro(Z_conv, y_labels)
    acc_conv_lda, desg_conv_lda = evaluar_lda_cv(Z_conv, y_labels)
    
    resultados_totales[tag] = {
        "N": len(df),
        "PCA_2D_GMM": acc_pca_gmm,
        "PCA_2D_LDA": acc_pca_lda,
        "Isomap_k": best_k,
        "Isomap_2D_GMM": acc_iso_gmm,
        "Isomap_2D_LDA": acc_iso_lda,
        "Conv_2D_GMM": acc_conv_gmm,
        "Conv_2D_LDA": acc_conv_lda,
        "Desglose_Isomap": desg_iso_gmm,
        "Desglose_Conv": desg_conv_gmm
    }
    
    # Graficar Columna 1: PCA 2D
    ax_pca = axes[row_idx, 0]
    ax_pca.set_facecolor('#1F2833')
    for v in VOCALES:
        m = (y_labels == v)
        ax_pca.scatter(Z_pca[m, 0], Z_pca[m, 1], c=COLORES_VOCALES[v], label=f'/{v.lower()}/', alpha=0.75, edgecolors='none', s=35)
    ax_pca.set_title(f"{titulo_sesion}\nPCA 2D - GMM: {acc_pca_gmm:.1f}% | LDA: {acc_pca_lda:.1f}%", color='white', fontsize=11, fontweight='bold')
    ax_pca.grid(True, linestyle='--', alpha=0.3, color='#C5C6C7')
    ax_pca.tick_params(colors='white')
    
    # Graficar Columna 2: Isomap 2D
    ax_iso = axes[row_idx, 1]
    ax_iso.set_facecolor('#1F2833')
    for v in VOCALES:
        m = (y_labels == v)
        ax_iso.scatter(best_Z_iso[m, 0], best_Z_iso[m, 1], c=COLORES_VOCALES[v], label=f'/{v.lower()}/', alpha=0.75, edgecolors='none', s=35)
    ax_iso.set_title(f"Isomap 2D (k={best_k})\nGMM: {acc_iso_gmm:.1f}% | LDA: {acc_iso_lda:.1f}%", color='white', fontsize=11, fontweight='bold')
    ax_iso.grid(True, linestyle='--', alpha=0.3, color='#C5C6C7')
    ax_iso.tick_params(colors='white')
    
    # Graficar Columna 3: Autoencoder Conv Ortogonal 2D
    ax_conv = axes[row_idx, 2]
    ax_conv.set_facecolor('#1F2833')
    for v in VOCALES:
        m = (y_labels == v)
        ax_conv.scatter(Z_conv[m, 0], Z_conv[m, 1], c=COLORES_VOCALES[v], label=f'/{v.lower()}/', alpha=0.75, edgecolors='none', s=35)
    ax_conv.set_title(f"Autoencoder Conv Ortogonal 2D Récord\nGMM: {acc_conv_gmm:.1f}% | LDA: {acc_conv_lda:.1f}%", color='white', fontsize=11, fontweight='bold')
    ax_conv.grid(True, linestyle='--', alpha=0.3, color='#C5C6C7')
    ax_conv.tick_params(colors='white')

axes[0, 0].legend(loc='upper right', framealpha=0.8, facecolor='#0B0C10', labelcolor='white')
plt.tight_layout()
fig_path = os.path.join(out_fig_dir, "comparativa_no_linealidad_sesiones_2026_10_20.png")
plt.savefig(fig_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

# Exportar métricas a JSON
json_path = os.path.join(out_fig_dir, "metricas_no_linealidad_2026_10_20.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(resultados_totales, f, indent=4, ensure_ascii=False)

print("\n" + "=" * 75)
print("RESUMEN COMPARATIVO DE NO LINEALIDAD (PCA vs ISOMAP vs CONV ORTOGONAL 2D)")
print("=" * 75)
for tag, data in resultados_totales.items():
    print(f"\n[{tag}] N={data['N']} muestras:")
    print(f"  - Lineal PCA 2D          : GMM = {data['PCA_2D_GMM']:.2f}% | LDA CV = {data['PCA_2D_LDA']:.2f}%")
    print(f"  - No Lineal Isomap (k={data['Isomap_k']}) : GMM = {data['Isomap_2D_GMM']:.2f}% | LDA CV = {data['Isomap_2D_LDA']:.2f}%")
    print(f"  - Conv Ortogonal 2D Lucas: GMM = {data['Conv_2D_GMM']:.2f}% | LDA CV = {data['Conv_2D_LDA']:.2f}%")

print("\nFigura guardada en:", fig_path)
print("Métricas guardadas en:", json_path)
