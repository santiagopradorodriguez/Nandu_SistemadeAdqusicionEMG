import os, sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

sys.path.insert(0, os.path.abspath("."))
from EMG_desarrollo.deep_learning.generador_pca_6canales import procesar_sesion_6canales

session_dir = "EMG_desarrollo/base_de_datos_electrodos/2026-11-09/SecuenciaContinua_Pruebaconnuevoslandrmaksfacial_Sujeto1"

X, y, pulsos, curvas_dict, t_curvas = procesar_sesion_6canales(
    session_dir,
    rms_ms=90,
    outlier_contamination=0.08
)

# Evaluar diferentes niveles de gate de ruido sobre cada canal (silenciando cuando la actividad del canal es espuria)
for gate in [0.0, 0.10, 0.15, 0.20, 0.30, 0.40]:
    X_g = X.copy()
    if gate > 0:
        for i in range(X_g.shape[0]):
            for ch_i in range(6):
                seg = X_g[i, ch_i*25:(ch_i+1)*25]
                # Si la amplitud máxima en este canal durante el gesto no supera el umbral de ruido, poner en 0
                if np.max(seg) < gate:
                    X_g[i, ch_i*25:(ch_i+1)*25] = 0.0
                    
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_g)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_pca, y)
    acc = np.mean(lda.predict(X_pca) == y) * 100.0
    var = np.sum(pca.explained_variance_ratio_ * 100.0)
    
    # Por vocal
    acc_v = {}
    for v in ['A', 'E', 'I', 'O', 'U']:
        m = (y == v)
        acc_v[v] = np.mean(lda.predict(X_pca)[m] == v) * 100.0
    print(f"Gate {gate:4.2f} mm | Var 3D: {var:5.1f}% | LDA Global: {acc:5.1f}% | /a/: {acc_v['A']:.1f}% | /e/: {acc_v['E']:.1f}% | /i/: {acc_v['I']:.1f}% | /o/: {acc_v['O']:.1f}% | /u/: {acc_v['U']:.1f}%")
