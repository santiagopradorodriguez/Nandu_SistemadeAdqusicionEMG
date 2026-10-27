import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

# Cargamos el Isomap 2D y 3D ya calculados con reposo basal
import evaluar_reposo_basal_lucas as base

X_iso2d = base.iso_env_imp
Y = base.Y_imp
vocales = ['A', 'E', 'I', 'O', 'U']

# 1. LDA Multiclase (5 vocales) DIRECTO (Ajuste sobre todo el espacio)
lda_5v = LinearDiscriminantAnalysis()
lda_5v.fit(X_iso2d, Y)
preds_5v = lda_5v.predict(X_iso2d)
acc_directa = lda_5v.score(X_iso2d, Y) * 100
cm_directa = confusion_matrix(Y, preds_5v, labels=vocales)

# 2. LDA Multiclase con VALIDACION CRUZADA REAL (5-Fold CV)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_cv = []
for tr, ts in cv.split(X_iso2d, Y):
    clf = LinearDiscriminantAnalysis().fit(X_iso2d[tr], Y[tr])
    scores_cv.append(clf.score(X_iso2d[ts], Y[ts]))
acc_cv = np.mean(scores_cv) * 100
std_cv = np.std(scores_cv) * 100

print(f"=== EXACTITUD GENERAL MULTICLASE (5 VOCALES) - LDA SOBRE ISOMAP 2D ===")
print(f"Exactitud Directa (Resubstitución): {acc_directa:.1f}%")
print(f"Exactitud Real CV-5 (Test):          {acc_cv:.1f}% (+/- {std_cv:.1f}%)")
print("\nMatriz de Confusión LDA Multiclase:")
print("Real \\ Pred |  /a/   /e/   /i/   /o/   /u/  | Acierto por vocal")
print("-" * 65)
for i, v in enumerate(vocales):
    tot = np.sum(Y == v)
    ac = cm_directa[i, i]
    pct = ac / tot * 100
    fila = "  ".join([f"{cm_directa[i, j]:4d}" for j in range(5)])
    print(f"    /{v.lower()}/    | {fila}  | {pct:5.1f}% ({ac}/{tot})")
