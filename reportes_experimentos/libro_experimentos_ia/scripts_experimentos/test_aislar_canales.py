import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Importar procesar_sesion_6canales del codigo original
sys.path.insert(0, os.path.abspath("."))
from EMG_desarrollo.deep_learning.generador_pca_6canales import procesar_sesion_6canales, VOCALES_CANONICAS

session_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1"

X, y, pulsos, curvas_dict, t_curvas = procesar_sesion_6canales(
    session_dir,
    rms_ms=90,
    outlier_contamination=0.08
)

# X tiene shape (230, 150) -> 6 canales x 25 puntos
# Ch1: Apertura (0:25)
# Ch2: Sonrisa (25:50)
# Ch3: Orbicular (50:75)
# Ch4: Mejilla (75:100)
# Ch5: Malar (100:125)
# Ch6: Masetero (125:150)

def evaluar(X_mat, y_labels, desc=""):
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_mat)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y_labels)
    acc = np.mean(lda.predict(X_pca) == y_labels) * 100.0
    var = np.sum(pca.explained_variance_ratio_ * 100.0)
    print(f"{desc:<35} | Var 3D: {var:5.1f}% | LDA: {acc:5.1f}%")

print("Evaluación de subconjuntos de canales sobre el código oficial:")
evaluar(X, y, "6 Canales Oficiales")
evaluar(X[:, :75], y, "Triada Canónica (Ap + Sonr + Orb)")
evaluar(X[:, np.r_[0:75, 75:100]], y, "Triada + Mejilla (4ch)")
evaluar(X[:, np.r_[0:75, 100:125]], y, "Triada + Malar (4ch)")
evaluar(X[:, np.r_[0:75, 125:150]], y, "Triada + Masetero (4ch)")
evaluar(X[:, np.r_[0:75, 75:100, 125:150]], y, "Triada + Mejilla + Masetero (5ch)")
