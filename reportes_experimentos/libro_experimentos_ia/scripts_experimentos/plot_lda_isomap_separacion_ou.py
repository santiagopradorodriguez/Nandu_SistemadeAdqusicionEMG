import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import evaluar_reposo_basal_lucas as base

# Proyecciones Isomap 2D calculadas en el script anterior
iso_env_imp = base.iso_env_imp
Y = base.Y_imp
mask_ou = np.isin(Y, ['O', 'U'])

X_ou = iso_env_imp[mask_ou]
Y_ou = Y[mask_ou]

# LDA sobre el Isomap 2D
lda = LinearDiscriminantAnalysis()
lda.fit(X_ou, Y_ou)
acc = lda.score(X_ou, Y_ou) * 100
preds = lda.predict(X_ou)

# Matriz de confusion binaria
n_o_total = np.sum(Y_ou == 'O')
n_u_total = np.sum(Y_ou == 'U')
o_aciertos = np.sum((Y_ou == 'O') & (preds == 'O'))
u_aciertos = np.sum((Y_ou == 'U') & (preds == 'U'))

print(f"Total /o/: {n_o_total} (Aciertos: {o_aciertos} -> {o_aciertos/n_o_total*100:.1f}%)")
print(f"Total /u/: {n_u_total} (Aciertos: {u_aciertos} -> {u_aciertos/n_u_total*100:.1f}%)")
print(f"Exactitud LDA directa en Isomap 2D: {acc:.1f}%")

# Plot con frontera de decision LDA
plt.figure(figsize=(9, 8))
# Puntos O y U
plt.scatter(X_ou[Y_ou == 'O', 0], X_ou[Y_ou == 'O', 1], c='#9D4EDD', label=f"/o/ ({o_aciertos}/{n_o_total})", s=45, alpha=0.85, edgecolors='black', linewidths=0.5)
plt.scatter(X_ou[Y_ou == 'U', 0], X_ou[Y_ou == 'U', 1], c='#E7A61A', label=f"/u/ ({u_aciertos}/{n_u_total})", s=45, alpha=0.85, edgecolors='black', linewidths=0.5)

# Dibujar recta de separacion LDA
x_min, x_max = X_ou[:, 0].min() - 0.5, X_ou[:, 0].max() + 0.5
y_min, y_max = X_ou[:, 1].min() - 0.5, X_ou[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
Z = lda.predict(np.c_[xx.ravel(), yy.ravel()])
Z = np.array([1 if z == 'U' else 0 for z in Z]).reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2.0)

plt.title(f"Frontera LDA Binaria /o/ vs /u/ sobre Isomap 2D (Reposo Basal)\nExactitud: {acc:.1f}% (Aciertos: {o_aciertos + u_aciertos}/{len(Y_ou)})", fontweight='bold', fontsize=12)
plt.xlabel("Iso 1 (Geodésico)", fontweight='bold')
plt.ylabel("Iso 2 (Geodésico)", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, fontsize=11)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/lda_isomap_frontera_o_vs_u.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
