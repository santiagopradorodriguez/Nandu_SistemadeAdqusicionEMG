import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import analisis_silabas_2026_10_23 as base

X = base.X
Y = base.Y

# Mapear a punto de articulacion:
# Bilabiales: BA, PA, MA (activan fuerte Orbicular Ch2)
# Coronales / Dentales: TA, DA, SA, LA, RA (lengua y dientes, actividad masetero/modiolo)
# Velares / Dorsales: KA, JA (lengua posterior, apertura amplia)
def get_punto_articulacion(s):
    if s in ['BA', 'PA', 'MA']: return 'Bilabial (B, P, M)'
    elif s in ['TA', 'SA', 'LA', 'RA']: return 'Coronal/Alveolar (T, S, L, R)'
    elif s in ['KA', 'JA']: return 'Velar/Dorsal (K, J)'
    return 'Otro'

Y_art = np.array([get_punto_articulacion(s) for s in Y])

lda_art = LinearDiscriminantAnalysis().fit(X, Y_art)
acc_art = lda_art.score(X, Y_art) * 100

print(f"Exactitud LDA por Punto de Articulación (3 macro-familias): {acc_art:.1f}%")

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(9, 8))
colores_art = {
    'Bilabial (B, P, M)': '#9D4EDD',       # Morado labial
    'Coronal/Alveolar (T, S, L, R)': '#1F77B4', # Azul modiolo/sonrisa
    'Velar/Dorsal (K, J)': '#E63946'        # Rojo masetero/apertura
}

for art, col in colores_art.items():
    m = (Y_art == art)
    plt.scatter(X_pca[m, 0], X_pca[m, 1], c=col, label=f"{art} (N={np.sum(m)})", s=60, alpha=0.85, edgecolors='black', linewidths=0.5)

# Anotar texto de cada punto
for i, s in enumerate(Y):
    plt.annotate(s.lower(), (X_pca[i, 0]+0.05, X_pca[i, 1]+0.05), fontsize=8, alpha=0.7)

plt.title(f"PCA 2D de Sílabas por Punto de Articulación Motor\nExactitud Separabilidad Macro-Familias: {acc_art:.1f}%", fontweight='bold')
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_silabas_punto_articulacion.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
