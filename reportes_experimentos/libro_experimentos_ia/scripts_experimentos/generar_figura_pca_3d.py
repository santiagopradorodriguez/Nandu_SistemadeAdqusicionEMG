# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Descripcion: Generacion de visualizacion PCA 3D sin StandardScaler
#              sobre la triada masetero-modiolo-orbicularis (2026-10-23)
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

sys.path.append('reportes_experimentos/libro_experimentos_ia/scripts_experimentos')
import experimento_mfcc_pca_2026_10_23 as exp

COLOR_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

def main():
    muestras = exp.extraer_datos_triada()
    Y = np.array([m['vocal'] for m in muestras])
    X = np.array([m['feat_env'] for m in muestras])
    
    # PCA directo sin StandardScaler (tal como en generador_pca_umap)
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X)
    
    var = pca.explained_variance_ratio_ * 100.0
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    for v in ['A', 'E', 'I', 'O', 'U']:
        mask = (Y == v)
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2],
            c=COLOR_VOCALES[v],
            label=f"/{v.lower()}/ (N={np.sum(mask)})",
            s=55, alpha=0.85, edgecolors='black', linewidths=0.5
        )
        
    ax.set_title(
        f"PCA 3D sEMG - Tríada Masetero-Modiolo-Orbicularis (2026-10-23)\n"
        f"Varianza Explicada: PC1={var[0]:.1f}%, PC2={var[1]:.1f}%, PC3={var[2]:.1f}% (Total: {np.sum(var):.1f}%)",
        fontsize=12, fontweight='bold', pad=15
    )
    ax.set_xlabel(f"PC 1 ({var[0]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_ylabel(f"PC 2 ({var[1]:.1f}%)", fontweight='bold', labelpad=10)
    ax.set_zlabel(f"PC 3 ({var[2]:.1f}%)", fontweight='bold', labelpad=10)
    
    ax.view_init(elev=25, azim=135)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True, edgecolor='black', facecolor='white')
    
    plt.tight_layout()
    out_dir = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos"
    out_path = os.path.join(out_dir, "pca_3d_triada_2026_10_23_sin_scaler.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Figura guardada en: {out_path}")

if __name__ == "__main__":
    main()
