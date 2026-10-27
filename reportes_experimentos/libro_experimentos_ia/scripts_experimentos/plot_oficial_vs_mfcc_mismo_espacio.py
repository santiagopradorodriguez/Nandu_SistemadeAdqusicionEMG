import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# 1. Cargar las 171 muestras oficiales exactas
df_feat = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/caracteristicas_exportadas.csv")
vocales = df_feat['Vocal'].values
tomas = df_feat['Toma'].values
X_raw = df_feat.iloc[:, 2:].values

# Pesos oficiales [2.5, 1.0, 1.0]
pesos = [2.5, 1.0, 1.0]
pts = 60
X_proc = X_raw.copy()
for c, w in enumerate(pesos):
    X_proc[:, c*pts:(c+1)*pts] *= w

# Tu PCA oficial exacto (error con proyecciones guardadas = 0)
pca_oficial = PCA(n_components=2)
X_pca_oficial = pca_oficial.fit_transform(X_proc)

# 2. Cargar MFCCs calculados para exactamente esas tomas
import extraer_mfcc_estricto_oficial as ext_mfcc
X_mfcc = ext_mfcc.X_mfcc_arr
Y_mfcc = ext_mfcc.Y_arr
pca_mfcc = PCA(n_components=2).fit_transform(X_mfcc)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
voc_list = ['A', 'E', 'I', 'O', 'U']

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: TU PCA Oficial (idéntico pixel a pixel a PCA_2D.png)
for v in voc_list:
    m = (vocales == v)
    axes[0].scatter(X_pca_oficial[m, 0], X_pca_oficial[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

axes[0].set_title("Tu PCA 2D Oficial Exacto (Envolvente sEMG)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("PC 1", fontweight='bold')
axes[0].set_ylabel("PC 2", fontweight='bold')
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='best', frameon=True)

# Panel 2: MFCC sEMG puro
for v in voc_list:
    m = (Y_mfcc == v)
    axes[1].scatter(pca_mfcc[m, 0], pca_mfcc[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

axes[1].set_title("MFCC sEMG puro (Mismas Mediciones)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("PC 1", fontweight='bold')
axes[1].set_ylabel("PC 2", fontweight='bold')
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/tu_pca_oficial_vs_mfcc.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Comparativa oficial guardada en:", out_f)
