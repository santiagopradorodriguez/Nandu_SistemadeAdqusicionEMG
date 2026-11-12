#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG
# Autores: Lucas Braunstein y Santiago Prado
# Descripción: Evaluación de PCA en dimensión 4D (PC1 a PC4) para las dos
#              secuencias continuas De Bruijn combinadas (2026-10-30: Exp1 + Exp2)
#              con pulso completo tradicional (60 features temporales).
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# Fijar semillas universales
np.random.seed(42)

sys.path.append('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/deep_learning/pca_umap_clustering')
sys.path.append('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/deep_learning/binarizacion')
from generador_pca_umap import extraer_y_filtrar

# Paleta oficial universal Ñandú
PALETA_OFICIAL = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}
VOCALES = ['A', 'E', 'I', 'O', 'U']

params = {
    'alpha_ruido': 0.5,
    'gate_ratio_ruido': 0.0,
    'snr_threshold': 0.5,
    'outlier_contamination': 0.10,
    'smooth_ms': 125,
    'target_length': 20,
    'tipo_filtro_ruido': 'notch',
    'notch_q': 30.0,
    'highpass_cutoff_hz': 20.0,
    'lowpass_cutoff_hz': 500.0,
    'aplicar_correccion_intersesion': False
}

def cargar_serie(subpath):
    base_dir = os.path.join('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Nuevo', subpath, 'datos_crudos_y_pulsos')
    meds = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    res = extraer_y_filtrar(meds, base_dir, params, False, 'Pico Volumen Micrófono', 0.4, 0.6, ['canal_0', 'canal_1', 'canal_2'], verbose=False)
    X = np.array(res[0])
    y = np.array(res[1])
    return X, y

print("Cargando Serie 1 (Exp1 Jittering 25%)...")
X1, y1 = cargar_serie('2026-10-30_Exp1_SecuenciaDeBruijn_Jittering_Serie1')
print("Cargando Serie 2 (Exp2 Jittering 40%)...")
X2, y2 = cargar_serie('2026-10-30_Exp2_SecuenciaDeBruijn_Jittering_Serie2')

X = np.vstack([X1, X2])
y = np.concatenate([y1, y2])
N, D = X.shape
print(f"Dataset combinado: N={N} pulsos, D={D} variables temporales.")

# Ajuste PCA
pca = PCA(n_components=6, random_state=42)
X_pca = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100
var_cum = np.cumsum(var_exp)

print("\n--- Desglose de Varianza Explicada ---")
for i in range(6):
    print(f"PC{i+1}: {var_exp[i]:.2f}% (Acumulada: {var_cum[i]:.2f}%)")

# Métricas comparativas 3D vs 4D
def evaluar_lda(X_sub, y_labels):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_sub, y_labels)
    y_pred_direct = lda.predict(X_sub)
    acc_direct = np.mean(y_pred_direct == y_labels) * 100
    
    # Por vocal directo
    acc_vocal_direct = {}
    for v in VOCALES:
        m = (y_labels == v)
        acc_vocal_direct[v] = np.mean(y_pred_direct[m] == v) * 100 if np.sum(m) > 0 else 0.0
        
    # 5-Fold Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_cv = np.empty_like(y_labels)
    for train_idx, test_idx in cv.split(X_sub, y_labels):
        m_cv = LinearDiscriminantAnalysis()
        m_cv.fit(X_sub[train_idx], y_labels[train_idx])
        y_pred_cv[test_idx] = m_cv.predict(X_sub[test_idx])
        
    acc_cv = np.mean(y_pred_cv == y_labels) * 100
    acc_vocal_cv = {}
    for v in VOCALES:
        m = (y_labels == v)
        acc_vocal_cv[v] = np.mean(y_pred_cv[m] == v) * 100 if np.sum(m) > 0 else 0.0
        
    cm_cv = confusion_matrix(y_labels, y_pred_cv, labels=VOCALES)
    return acc_direct, acc_cv, acc_vocal_direct, acc_vocal_cv, cm_cv

acc_dir_3d, acc_cv_3d, v_dir_3d, v_cv_3d, cm_3d = evaluar_lda(X_pca[:, :3], y)
acc_dir_4d, acc_cv_4d, v_dir_4d, v_cv_4d, cm_4d = evaluar_lda(X_pca[:, :4], y)

print("\n=======================================================")
print(f"DIMENSION 3D (PC1-PC3, Varianza {var_cum[2]:.2f}%):")
print(f"  LDA Directo: {acc_dir_3d:.2f}% | 5-Fold CV: {acc_cv_3d:.2f}%")
print("  Por vocal en CV:")
for v in VOCALES:
    print(f"    /{v}/: {v_cv_3d[v]:.2f}%")

print("\n=======================================================")
print(f"DIMENSION 4D (PC1-PC4, Varianza {var_cum[3]:.2f}%):")
print(f"  LDA Directo: {acc_dir_4d:.2f}% | 5-Fold CV: {acc_cv_4d:.2f}%")
print("  Por vocal en CV:")
for v in VOCALES:
    print(f"    /{v}/: {v_cv_4d[v]:.2f}% (Diferencia respecto a 3D: {v_cv_4d[v] - v_cv_3d[v]:+.2f} pp)")

print("\nMatriz de Confusión 4D (5-Fold CV):")
print("Filas: Real, Columnas: Predicho [A, E, I, O, U]")
print(cm_4d)

# Generar Figura 2x3 con las 6 proyecciones 2D en 4D
fig, axes = plt.subplots(2, 3, figsize=(18, 12), dpi=250)
fig.patch.set_facecolor('#FFFFFF')

pares_proyeccion = [
    (0, 1, 'PC1', 'PC2'),
    (0, 2, 'PC1', 'PC3'),
    (0, 3, 'PC1', 'PC4'),
    (1, 2, 'PC2', 'PC3'),
    (1, 3, 'PC2', 'PC4'),
    (2, 3, 'PC3', 'PC4')
]

for idx, (p_x, p_y, name_x, name_y) in enumerate(pares_proyeccion):
    r = idx // 3
    c = idx % 3
    ax = axes[r, c]
    ax.set_facecolor('#FFFFFF')
    
    # Graficar puntos de cada vocal
    for v in VOCALES:
        mask = (y == v)
        ax.scatter(
            X_pca[mask, p_x], X_pca[mask, p_y],
            c=PALETA_OFICIAL[v],
            label=f'/{v.lower()}/',
            alpha=0.75,
            s=45,
            edgecolors='black',
            linewidth=0.4
        )
        # Centroide
        cx = np.mean(X_pca[mask, p_x])
        cy = np.mean(X_pca[mask, p_y])
        ax.scatter(cx, cy, c=PALETA_OFICIAL[v], s=160, marker='D', edgecolors='#000000', linewidth=1.5, zorder=5)

    ax.set_title(f'Proyección {name_x} vs {name_y}', fontsize=12, fontweight='bold', pad=8)
    ax.set_xlabel(f'{name_x} ({var_exp[p_x]:.2f}% var)', fontsize=10)
    ax.set_ylabel(f'{name_y} ({var_exp[p_y]:.2f}% var)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.axhline(0, color='#94a3b8', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='#94a3b8', linestyle=':', linewidth=0.8)
    if idx == 0:
        ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

# Subtitulo y encabezado sin parentesis
header_text = (
    f"Secuencias Continuas De Bruijn 10-30 Juntas - Proyecciones PCA 4D (N={N})\n"
    f"Varianza Acumulada 4D: {var_cum[3]:.2f}% (PC1: {var_exp[0]:.1f}%, PC2: {var_exp[1]:.1f}%, PC3: {var_exp[2]:.1f}%, PC4: {var_exp[3]:.1f}%)\n"
    f"Rendimiento LDA 4D: Directo {acc_dir_4d:.2f}% | 5-Fold CV {acc_cv_4d:.2f}%  [En 3D: Directo {acc_dir_3d:.2f}%, CV {acc_cv_3d:.2f}%]"
)
fig.suptitle(header_text, fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])

out_fig_dir = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/reportes_experimentos/libro_experimentos_ia/cuaderno_laboratorio_pre_fono/figuras'
out_fig_path = os.path.join(out_fig_dir, '13_secuencias_continuas_1030_PCA_4D_proyecciones.png')
plt.savefig(out_fig_path, dpi=250, bbox_inches='tight')
plt.close()
print(f"\nFigura guardada en: {out_fig_path}")

# Copiar al directorio de artifacts
artifact_path = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29/13_secuencias_continuas_1030_PCA_4D_proyecciones.png'
os.system(f'cp "{out_fig_path}" "{artifact_path}"')
print(f"Copiada a artifacts: {artifact_path}")
