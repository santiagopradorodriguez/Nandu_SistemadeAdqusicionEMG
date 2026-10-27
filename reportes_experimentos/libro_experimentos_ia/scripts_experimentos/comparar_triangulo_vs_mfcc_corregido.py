import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.optimize import linear_sum_assignment

# Cargar envolvente oficial
import os, sys
sys.path.append('EMG_desarrollo/deep_learning/pca_umap_clustering')
import generador_pca_umap as gpu

base_dir = 'EMG_desarrollo/base_de_datos_electrodos'
mediciones = [
    os.path.join('2026-07-10', f'{v}_T{t}_Lucas')
    for v in ['A', 'E', 'I', 'O', 'U']
    for t in range(1, 8)
    if os.path.isdir(os.path.join(base_dir, '2026-07-10', f'{v}_T{t}_Lucas'))
]

params = {
    'smooth_ms': 250,
    'target_length': 100,
    'alpha_ruido': 1.0,
    'notch_q': 2.0,
    'tipo_filtro_ruido': 'notch',
    'highpass_cutoff_hz': 20.0,
    'lowpass_cutoff_hz': 300.0,
    'tipo_envolvente': 'rms',
    'snr_threshold': 0.0,
    'gate_ratio_ruido': 0.0,
    'outlier_contamination': 0.05
}

X_env_clean, Y_clean, _, _ = gpu.extraer_y_filtrar(
    mediciones=mediciones, base_dir=base_dir, params=params,
    aplicar_trevisan=False, modo_alineacion='Pico Volumen Micrófono',
    pre_pct=0.4, post_pct=0.6, canales_features=['canal_0', 'canal_1', 'canal_2'],
    ignorar_ventana_cero=True, correccion_impedancia=False
)

# Cargar MFCC crudo oscilatorio de esas mismas tomas
import mfcc_lucas_desde_wavs_crudos as base_m
X_mfcc_crudo = base_m.X_mfcc[:len(X_env_clean)] # alineado
Y_mfcc = base_m.Y[:len(X_env_clean)]

pca_env = PCA(n_components=2).fit_transform(X_env_clean)
pca_mfcc = PCA(n_components=2).fit_transform(X_mfcc_crudo)

# Hibrido
gamma = np.sqrt(np.mean(np.var(X_env_clean, axis=0)) / (np.mean(np.var(X_mfcc_crudo, axis=0)) + 1e-9))
X_hib = np.hstack([X_env_clean, X_mfcc_crudo * gamma])
pca_hib = PCA(n_components=2).fit_transform(X_hib)

vocales = ['A', 'E', 'I', 'O', 'U']
def eval_km(X_p, Y_t):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    for i, v in enumerate(vocales):
        for j in range(5): cost[i, j] = -np.sum((Y_t == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: vocales[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_t) * 100

mask_ou = np.isin(Y_clean, ['O', 'U'])
km_env = eval_km(pca_env, Y_clean)
km_mfcc = eval_km(pca_mfcc, Y_mfcc)
km_hib = eval_km(pca_hib, Y_clean)

lda_ou_env = LinearDiscriminantAnalysis().fit(pca_env[mask_ou], Y_clean[mask_ou]).score(pca_env[mask_ou], Y_clean[mask_ou]) * 100
lda_ou_mfcc = LinearDiscriminantAnalysis().fit(pca_mfcc[mask_ou], Y_mfcc[mask_ou]).score(pca_mfcc[mask_ou], Y_mfcc[mask_ou]) * 100
lda_ou_hib = LinearDiscriminantAnalysis().fit(pca_hib[mask_ou], Y_clean[mask_ou]).score(pca_hib[mask_ou], Y_clean[mask_ou]) * 100

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

paneles = [
    (axes[0], pca_env, Y_clean, "1. Envolvente Oficial (Triángulo Canónico)", km_env, lda_ou_env),
    (axes[1], pca_mfcc, Y_mfcc, "2. Mel Muscular sEMG (Cruda AC)", km_mfcc, lda_ou_mfcc),
    (axes[2], pca_hib, Y_clean, "3. Híbrido Equilibrado (Tiempo + Mel AC)", km_hib, lda_ou_hib)
]

for ax, X_p, Y_p, tit, km_v, ou_v in paneles:
    for v in vocales:
        m = (Y_p == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means: {km_v:.1f}% | Sep. O vs U: {ou_v:.1f}%", fontweight='bold', fontsize=11)
    ax.set_xlabel("PC 1", fontweight='bold')
    ax.set_ylabel("PC 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lucas_triangulo_vs_mfcc_corregido.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
