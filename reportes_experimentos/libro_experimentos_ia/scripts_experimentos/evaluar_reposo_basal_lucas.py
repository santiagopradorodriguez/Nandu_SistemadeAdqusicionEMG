import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import linear_sum_assignment

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

# 1. Extraccion con CORRECCION DE REPOSO BASAL / IMPEDANCIA ACTIVA
print("Extrayendo con correccion_impedancia=True...")
X_env_imp, Y_imp, Tomas_imp, _ = gpu.extraer_y_filtrar(
    mediciones=mediciones, base_dir=base_dir, params=params,
    aplicar_trevisan=False, modo_alineacion='Pico Volumen Micrófono',
    pre_pct=0.4, post_pct=0.6, canales_features=['canal_0', 'canal_1', 'canal_2'],
    ignorar_ventana_cero=True, correccion_impedancia=True
)

# 2. Extraccion SIN correccion
print("Extrayendo con correccion_impedancia=False...")
X_env_noimp, Y_noimp, Tomas_noimp, _ = gpu.extraer_y_filtrar(
    mediciones=mediciones, base_dir=base_dir, params=params,
    aplicar_trevisan=False, modo_alineacion='Pico Volumen Micrófono',
    pre_pct=0.4, post_pct=0.6, canales_features=['canal_0', 'canal_1', 'canal_2'],
    ignorar_ventana_cero=True, correccion_impedancia=False
)

# Traemos el MFCC crudo alineado
import mfcc_lucas_desde_wavs_crudos as base_m
X_mfcc = base_m.X_mfcc[:len(X_env_imp)]

# Factores de equilibrado
g_imp = np.sqrt(np.mean(np.var(X_env_imp, axis=0)) / (np.mean(np.var(X_mfcc, axis=0)) + 1e-9))
X_hib_imp = np.hstack([X_env_imp, X_mfcc * g_imp])

g_noimp = np.sqrt(np.mean(np.var(X_env_noimp, axis=0)) / (np.mean(np.var(X_mfcc, axis=0)) + 1e-9))
X_hib_noimp = np.hstack([X_env_noimp, X_mfcc * g_noimp])

# Evaluacion LDA y CV-5
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
mask_ou = np.isin(Y_imp, ['O', 'U'])
Y_ou = Y_imp[mask_ou]

modelos = {
    "1. Envolvente SIN Reposo Basal": X_env_noimp,
    "2. Envolvente CON Reposo Basal": X_env_imp,
    "3. Híbrido SIN Reposo Basal": X_hib_noimp,
    "4. Híbrido CON Reposo Basal (Tiempo + Mel AC)": X_hib_imp,
}

print("\n=== EFECTO DEL REPOSO BASAL EN CLASIFICACION DIRECTA (LDA CV-5) ===")
print(f"{'Modelo':<48} | {'Exactitud 5v (CV-5)':<20} | {'O vs U (CV-5)':<12}")
print("-" * 85)

for nombre, X_mat in modelos.items():
    sc_5v = []
    for tr, ts in cv.split(X_mat, Y_imp):
        lda = LinearDiscriminantAnalysis().fit(X_mat[tr], Y_imp[tr])
        sc_5v.append(lda.score(X_mat[ts], Y_imp[ts]))
        
    X_ou = X_mat[mask_ou]
    sc_ou = []
    for tr, ts in cv.split(X_ou, Y_ou):
        lda = LinearDiscriminantAnalysis().fit(X_ou[tr], Y_ou[tr])
        sc_ou.append(lda.score(X_ou[ts], Y_ou[ts]))
        
    print(f"{nombre:<48} | {np.mean(sc_5v)*100:5.1f}% (+/- {np.std(sc_5v)*100:4.1f}%)    | {np.mean(sc_ou)*100:5.1f}%")

# Proyecciones Isomap 2D
print("\nCalculando Isomap 2D...")
iso_env_imp = Isomap(n_neighbors=15, n_components=2).fit_transform(X_env_imp)
iso_hib_imp = Isomap(n_neighbors=15, n_components=2).fit_transform(X_hib_imp)

# Separabilidad O vs U en Isomap 2D con Reposo Basal
lda_iso_env_ou = LinearDiscriminantAnalysis().fit(iso_env_imp[mask_ou], Y_ou).score(iso_env_imp[mask_ou], Y_ou) * 100
lda_iso_hib_ou = LinearDiscriminantAnalysis().fit(iso_hib_imp[mask_ou], Y_ou).score(iso_hib_imp[mask_ou], Y_ou) * 100

def eval_km(X_p):
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_p)
    cost = np.zeros((5, 5))
    voc = ['A', 'E', 'I', 'O', 'U']
    for i, v in enumerate(voc):
        for j in range(5): cost[i, j] = -np.sum((Y_imp == v) & (km.labels_ == j))
    r, c = linear_sum_assignment(cost)
    map_c = {c[i]: voc[r[i]] for i in range(5)}
    preds = np.array([map_c[cl] for cl in km.labels_])
    return np.mean(preds == Y_imp) * 100

km_env_imp = eval_km(iso_env_imp)
km_hib_imp = eval_km(iso_hib_imp)

print(f"Isomap 2D Envolvente CON Reposo -> K-Means: {km_env_imp:.1f}% | O vs U (LDA): {lda_iso_env_ou:.1f}%")
print(f"Isomap 2D Híbrido CON Reposo    -> K-Means: {km_hib_imp:.1f}% | O vs U (LDA): {lda_iso_hib_ou:.1f}%")

# Plot comparativo
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

for ax, X_p, tit, km_v, ou_v in [
    (axes[0], iso_env_imp, "Isomap 2D - Envolvente CON Reposo Basal", km_env_imp, lda_iso_env_ou),
    (axes[1], iso_hib_imp, "Isomap 2D - Híbrido CON Reposo Basal (Tiempo + Mel AC)", km_hib_imp, lda_iso_hib_ou)
]:
    for v in ['A', 'E', 'I', 'O', 'U']:
        m = (Y_imp == v)
        ax.scatter(X_p[m, 0], X_p[m, 1], c=COLOR_VOCALES[v], label=f"/{v.lower()}/", s=35, alpha=0.85, edgecolors='black', linewidths=0.5)
    ax.set_title(f"{tit}\nK-Means: {km_v:.1f}% | Sep. Lineal /o/ vs /u/: {ou_v:.1f}%", fontweight='bold', fontsize=11)
    ax.set_xlabel("Iso 1", fontweight='bold')
    ax.set_ylabel("Iso 2", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best', frameon=True)

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lucas_isomap_reposo_basal.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
