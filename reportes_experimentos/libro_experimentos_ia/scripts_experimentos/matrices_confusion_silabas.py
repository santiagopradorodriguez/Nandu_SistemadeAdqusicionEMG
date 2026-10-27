import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

import analisis_silabas_2026_10_23 as base

X = base.X     # (57, 180)
Y = base.Y     # 9 silabas
silabas = sorted(list(np.unique(Y)))

# 1. PCA 4D con LDA
pca4 = PCA(n_components=4, random_state=42)
X_4d = pca4.fit_transform(X)
lda4 = LinearDiscriminantAnalysis().fit(X_4d, Y)
preds_4d = lda4.predict(X_4d)
cm_4d = confusion_matrix(Y, preds_4d, labels=silabas)
acc_4d = np.mean(preds_4d == Y) * 100

# 2. PCA 8D con LDA
pca8 = PCA(n_components=8, random_state=42)
X_8d = pca8.fit_transform(X)
lda8 = LinearDiscriminantAnalysis().fit(X_8d, Y)
preds_8d = lda8.predict(X_8d)
cm_8d = confusion_matrix(Y, preds_8d, labels=silabas)
acc_8d = np.mean(preds_8d == Y) * 100

def imprimir_matriz(cm, nombre, acc):
    print(f"\n=== MATRIZ DE CONFUSION: {nombre} (Acc: {acc:.1f}%) ===")
    print("Real \\ Pred | " + "  ".join([f"{s:>3}" for s in silabas]) + " | Acierto")
    print("-" * 65)
    for i, s in enumerate(silabas):
        tot = np.sum(Y == s)
        ac = cm[i, i]
        pct = ac / tot * 100
        fila = "  ".join([f"{cm[i, j]:3d}" for j in range(len(silabas))])
        print(f"    /{s.lower()}/   | {fila} | {pct:5.1f}% ({ac}/{tot})")

imprimir_matriz(cm_4d, "PCA 4D (LDA)", acc_4d)
imprimir_matriz(cm_8d, "PCA 8D (LDA)", acc_8d)

# Generar heatmaps lado a lado
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for ax, cm, tit, acc in [(axes[0], cm_4d, "PCA 4D", acc_4d), (axes[1], cm_8d, "PCA 8D", acc_8d)]:
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_title(f"Matriz de Confusión {tit} (Acc: {acc:.1f}%)", fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(silabas)))
    ax.set_yticks(range(len(silabas)))
    ax.set_xticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_yticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_xlabel("Predicción", fontweight='bold')
    ax.set_ylabel("Real", fontweight='bold')
    
    # Numeros adentro
    for i in range(len(silabas)):
        for j in range(len(silabas)):
            val = cm[i, j]
            col = "white" if val > np.max(cm)/2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=col, fontweight='bold')

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/heatmaps_confusion_silabas.png"
plt.savefig(out_f, dpi=300)
plt.close()
print(f"\nHeatmap guardado en: {out_f}")
