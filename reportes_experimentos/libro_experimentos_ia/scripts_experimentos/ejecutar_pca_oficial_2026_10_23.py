import os
import sys
import numpy as np
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

params = {
    'alpha_ruido': 1.0,
    'gate_ratio_ruido': 0.0,
    'smooth_ms': 250,
    'notch_q': 2.0,
    'target_length': 100,
    'tipo_filtro_ruido': 'notch',
    'highpass_cutoff_hz': 20.0,
    'lowpass_cutoff_hz': 300.0,
    'tipo_envolvente': 'rms',
    'snr_min': 0.0,
    'outlier_contamination': 0.05
}

X_clean, Y_clean, Tomas_clean, _ = gpu.extraer_y_filtrar(
    mediciones=mediciones,
    base_dir=base_dir,
    params=params,
    aplicar_trevisan=False,
    modo_alineacion="Pico Volumen Micrófono",
    pre_pct=0.4,
    post_pct=0.6,
    canales_features=["canal_0", "canal_1", "canal_2"],
    ignorar_ventana_cero=True,
    correccion_impedancia=True
)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_clean)

COLOR_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

plt.figure(figsize=(9, 8))
for v in ['A', 'E', 'I', 'O', 'U']:
    m = (Y_clean == v)
    plt.scatter(X_pca[m, 0], X_pca[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=50, alpha=0.8, edgecolors='black', linewidths=0.5)

plt.title("PCA 2D Oficial (Pipeline Completo generador_pca_umap)", fontweight='bold')
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()

out_path = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_oficial_pipeline.png"
plt.savefig(out_path, dpi=300)
plt.close()
print("Guardado en:", out_path)
