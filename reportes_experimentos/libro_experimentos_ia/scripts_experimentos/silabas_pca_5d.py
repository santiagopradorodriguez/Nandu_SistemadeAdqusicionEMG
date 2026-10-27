import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

import analisis_silabas_2026_10_23 as base

X = base.X     # (57, 180)
Y = base.Y     # 9 silabas
silabas = sorted(list(np.unique(Y)))

pca5 = PCA(n_components=5, random_state=42)
X_5d = pca5.fit_transform(X)
var5 = pca5.explained_variance_ratio_ * 100

lda5 = LinearDiscriminantAnalysis().fit(X_5d, Y)
preds5 = lda5.predict(X_5d)
acc_5d = np.mean(preds5 == Y) * 100

cm_5d = confusion_matrix(Y, preds5, labels=silabas)
cm_5d_pct = cm_5d.astype('float') / cm_5d.sum(axis=1)[:, np.newaxis] * 100.0

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
sc = []
for tr, ts in cv.split(X_5d, Y):
    clf = LinearDiscriminantAnalysis().fit(X_5d[tr], Y[tr])
    sc.append(clf.score(X_5d[ts], Y[ts]))
acc_cv = np.mean(sc) * 100

print(f"=== PCA 5D SÍLABAS ===")
print(f"Varianza Explicada Acumulada: {np.sum(var5):.1f}% (PC1: {var5[0]:.1f}%, PC2: {var5[1]:.1f}%, PC3: {var5[2]:.1f}%, PC4: {var5[3]:.1f}%, PC5: {var5[4]:.1f}%)")
print(f"Exactitud LDA Directa: {acc_5d:.1f}%")
print(f"Exactitud LDA CV-3:     {acc_cv:.1f}% (+/- {np.std(sc)*100:.1f}%)")

print("\nMatriz de Confusión en Porcentajes (PCA 5D):")
print("Real \\ Pred | " + "  ".join([f"{s:>4}" for s in silabas]) + " | Acierto")
print("-" * 70)
for i, s in enumerate(silabas):
    tot = np.sum(Y == s)
    ac = cm_5d[i, i]
    fila = "  ".join([f"{cm_5d_pct[i, j]:4.0f}%" if cm_5d_pct[i, j] > 0 else "   - " for j in range(len(silabas))])
    print(f"    /{s.lower()}/   | {fila} | {cm_5d_pct[i, i]:5.1f}% ({ac}/{tot})")

# Graficar Heatmap 5D
plt.figure(figsize=(9, 8))
im = plt.imshow(cm_5d_pct, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
plt.title(f"Matriz de Confusión PCA 5D - Normalizada por Clase\nExactitud Global: {acc_5d:.1f}% (Var: {np.sum(var5):.1f}%)", fontweight='bold', fontsize=12)
plt.xticks(range(len(silabas)), [f"/{s.lower()}/" for s in silabas], fontweight='bold')
plt.yticks(range(len(silabas)), [f"/{s.lower()}/" for s in silabas], fontweight='bold')
plt.xlabel("Predicción", fontweight='bold')
plt.ylabel("Real", fontweight='bold')

for i in range(len(silabas)):
    for j in range(len(silabas)):
        val = cm_5d_pct[i, j]
        col = "white" if val > 50 else "black"
        txt = f"{val:.0f}%" if val > 0 else "-"
        plt.text(j, i, txt, ha="center", va="center", color=col, fontweight='bold')

cbar = plt.colorbar(im, shrink=0.8, pad=0.04)
cbar.set_label('Porcentaje de Clasificación (%)', fontweight='bold')
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/heatmap_pca_5d_silabas.png"
plt.savefig(out_f, dpi=300)
plt.close()
print(f"Heatmap guardado en: {out_f}")
