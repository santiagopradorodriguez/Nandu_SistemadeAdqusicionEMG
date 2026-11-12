import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.abspath("."))
from EMG_desarrollo.deep_learning.generador_pca_6canales import procesar_sesion_6canales, VOCALES_CANONICAS

# Código de colores universal canónico del proyecto:
# /a/: Rojo (#E63946)
# /e/: Azul (#1F77B4)
# /i/: Verde (#2CA02C)
# /o/: Morado (#9D4EDD)
# /u/: Amarillo (#E7A61A)
COLORES_CANONICOS = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

session_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1"

X, y, pulsos, curvas_dict, t_curvas = procesar_sesion_6canales(
    session_dir,
    rms_ms=90,
    outlier_contamination=0.08
)

# Convertir etiquetas a mayúsculas si vinieran en minúsculas
y = np.array([str(val).upper() for val in y])

pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X)
var = pca.explained_variance_ratio_ * 100

lda = LinearDiscriminantAnalysis()
lda.fit(X_pca, y)
preds_lda = lda.predict(X_pca)
acc_lda = np.mean(preds_lda == y) * 100.0

acc_por_vocal = {}
for v in VOCALES_CANONICAS:
    mask_v = (y == v)
    acc_por_vocal[v] = np.mean(preds_lda[mask_v] == v) * 100.0 if np.any(mask_v) else 0.0

kmeans = KMeans(n_clusters=len(np.unique(y)), random_state=42, n_init=20)
clusters = kmeans.fit_predict(X_pca)
vocales_u = sorted(list(set(y)))
y_idx = np.array([vocales_u.index(v) for v in y])
mat_conf = np.zeros((len(vocales_u), len(vocales_u)))
for c, r in zip(clusters, y_idx):
    mat_conf[c, r] += 1
row_ind, col_ind = linear_sum_assignment(-mat_conf)
acc_kmeans = (mat_conf[row_ind, col_ind].sum() / len(y)) * 100.0

def renderizar_figura_3d(X_pca, y, var, acc_lda, acc_por_vocal, acc_kmeans, output_png):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), facecolor='#ffffff')
    plt.subplots_adjust(top=0.86, bottom=0.12, left=0.05, right=0.98, wspace=0.28)
    
    proyecciones = [
        (0, 1, f"PC1 ({var[0]:.1f}%) vs PC2 ({var[1]:.1f}%)", axes[0]),
        (0, 2, f"PC1 ({var[0]:.1f}%) vs PC3 ({var[2]:.1f}%)", axes[1]),
        (1, 2, f"PC2 ({var[1]:.1f}%) vs PC3 ({var[2]:.1f}%)", axes[2])
    ]
    
    fig.suptitle(
        f"PCA 3D Cinemático Facial: LDA = {acc_lda:.1f}% | K-Means No Supervisado = {acc_kmeans:.1f}% (Varianza 3D = {np.sum(var):.1f}%)",
        fontsize=14, fontweight='bold', color='#111827'
    )
    
    for idx_x, idx_y, title, ax in proyecciones:
        ax.set_facecolor('#ffffff')
        ax.set_box_aspect(1)
        ax.grid(True, linestyle='--', alpha=0.35, color='#94a3b8')
        ax.axhline(0, color='#64748b', linestyle=':', linewidth=0.8)
        ax.axvline(0, color='#64748b', linestyle=':', linewidth=0.8)
        
        for v in ['A', 'E', 'I', 'O', 'U']:
            mask = (y == v)
            if not np.any(mask):
                continue
            color = COLORES_CANONICOS[v]
            ax.scatter(
                X_pca[mask, idx_x], X_pca[mask, idx_y],
                c=color, s=55, alpha=0.85, edgecolors='none',
                label=f"/{v.lower()}/ ({acc_por_vocal.get(v, 0):.0f}%)"
            )
            
            centroid_x = np.mean(X_pca[mask, idx_x])
            centroid_y = np.mean(X_pca[mask, idx_y])
            ax.scatter(
                [centroid_x], [centroid_y],
                c=color, s=260, edgecolors='#0f172a', linewidths=2.0, zorder=5
            )
            ax.text(
                centroid_x, centroid_y, f"/{v.lower()}/",
                color='#ffffff', fontsize=10, fontweight='bold',
                ha='center', va='center', zorder=6
            )
            
        ax.set_title(title, fontsize=12, fontweight='bold', color='#1e293b')
        ax.set_xlabel(f"PC{idx_x+1}", fontsize=11, fontweight='bold', color='#1e293b')
        ax.set_ylabel(f"PC{idx_y+1}", fontsize=11, fontweight='bold', color='#1e293b')
        ax.tick_params(colors='#1e293b')
        if idx_x == 0 and idx_y == 1:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9, facecolor='#ffffff', edgecolor='#cbd5e1')
            
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=200, facecolor='#ffffff')
    plt.close()
    print(f"[OK] Guardado en color: {output_png}")

renderizar_figura_3d(X_pca, y, var, acc_lda, acc_por_vocal, acc_kmeans, "reportes_experimentos/decodificacion_facial/pca_cinetico_oficial.png")
renderizar_figura_3d(X_pca, y, var, acc_lda, acc_por_vocal, acc_kmeans, "reportes_experimentos/decodificacion_facial/pca_6canales_oficial.png")
