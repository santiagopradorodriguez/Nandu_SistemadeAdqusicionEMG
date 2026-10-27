import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('EMG_desarrollo/deep_learning/pca_umap_clustering')
import generador_pca_umap as gpu
from sklearn.decomposition import PCA

base_dir = "EMG_desarrollo/base_de_datos_electrodos"
mediciones = [
    os.path.join("2026-10-23", f"{v}_Pruebatricanal_{suf}")
    for v in ['A', 'E', 'I', 'O', 'U']
    for suf in ['Sujeto1', '2_Sujeto1', '3_Sujeto1', '4_Sujeto1']
    if os.path.isdir(os.path.join(base_dir, "2026-10-23", f"{v}_Pruebatricanal_{suf}"))
]

with open("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/parametros_optimos_pca_2d.json") as f:
    params = json.load(f)

# Correr la funcion oficial de extraccion y filtrado de generador_pca_umap
X_clean, Y_clean, Tomas_clean, desc = gpu.extraer_y_filtrar(
    mediciones=mediciones,
    base_dir=base_dir,
    params=params,
    aplicar_trevisan=False,
    modo_alineacion="Pico Volumen Micrófono",
    pre_pct=0.4,
    post_pct=0.6,
    canales_features=["canal_0", "canal_1", "canal_2"],
    ignorar_ventana_cero=False,
    aplicar_correccion_intersesion=False,
    correccion_impedancia=False
)

# Aplicar pesos de canales
pesos = params.get('pesos_canales', [2.5, 1.0, 1.0])
X_proc = gpu.aplicar_pesos_canales(X_clean, ["canal_0", "canal_1", "canal_2"], pesos)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_proc)

COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}
voc_list = ['A', 'E', 'I', 'O', 'U']

fig, ax = plt.subplots(figsize=(9, 8))
for v in voc_list:
    m = (Y_clean == v)
    ax.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={np.sum(m)})", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

var = pca.explained_variance_ratio_ * 100.0
ax.set_title(f"PCA 2D Replicado con generador_pca_umap desde Cero\nVar: PC1={var[0]:.1f}%, PC2={var[1]:.1f}% (Total: {np.sum(var):.1f}%)", fontweight='bold')
ax.set_xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold')
ax.set_ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_replicado_desde_cero.png"
plt.savefig(out_f, dpi=300)
plt.close()

# Comparar con el oficial guardado
df_guardado = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/proyecciones_pca_2d.csv")
print(f"Muestras replicadas: {len(X_pca)} vs Guardadas: {len(df_guardado)}")
print(f"Guardado en: {out_f}")
