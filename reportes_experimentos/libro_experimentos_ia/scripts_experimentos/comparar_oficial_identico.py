import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# 1. Cargar la matriz exacta oficial ya procesada por generador_pca_umap
df_feat = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/caracteristicas_exportadas.csv")
vocales = df_feat['Vocal'].values
tomas = df_feat['Toma'].values
X_raw = df_feat.iloc[:, 2:].values # 180 cols (60 pts por canal)

# Aplicar pesos exactos del JSON [2.5, 1.0, 1.0]
pesos = [2.5, 1.0, 1.0]
pts = 60
X_proc = X_raw.copy()
for c, w in enumerate(pesos):
    X_proc[:, c*pts:(c+1)*pts] *= w

# PCA exacto oficial
pca_oficial = PCA(n_components=2)
X_pca_oficial = pca_oficial.fit_transform(X_proc)

# Cargar también las proyecciones que generó tu script
df_proj = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/proyecciones_pca_2d.csv")
X_pca_guardado = df_proj[['PC1', 'PC2']].values

print("Diferencia maxima absoluta entre PCA reproducido y proyecciones guardadas:")
print(np.max(np.abs(np.abs(X_pca_oficial) - np.abs(X_pca_guardado))))

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
voc_list = ['A', 'E', 'I', 'O', 'U']

fig, ax = plt.subplots(figsize=(8, 7))
for v in voc_list:
    m = (vocales == v)
    ax.scatter(X_pca_oficial[m, 0], X_pca_oficial[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.8, edgecolors='black', linewidths=0.5)

ax.set_title("Tu PCA 2D Oficial Exacto (171 pulsos)", fontweight='bold')
ax.set_xlabel("PC 1", fontweight='bold')
ax.set_ylabel("PC 2", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/tu_pca_oficial_exacto.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Tu figura oficial reproducida exactamente en:", out_f)
