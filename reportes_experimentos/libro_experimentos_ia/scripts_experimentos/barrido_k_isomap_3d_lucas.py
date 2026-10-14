#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Modulo: Barrido de Parametro de Vecinos (k) en Isomap 3D - Sujeto Lucas
# Objetivo: Evaluar la variedad no lineal en 3D para k = [6, 12, 20, 35, 60],
#           calculando Exactitud GMM 3D, centroides y orden relativo frente a PCA 3D.
# ==============================================================================

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Semilla fija universal
np.random.seed(42)

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Cargar dataset de Lucas
npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana5050/dataset_lucas_ventana5050.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_lucas_ventana4060/dataset_lucas_ventana4060.npz")

print(f"[Dataset] Cargando mediciones de Lucas desde:\n          {npz_path}")
data = np.load(npz_path, allow_pickle=True)
X_raw = data['X_clean'] if 'X_clean' in data else data['X']
Y_raw = data['Y_clean'] if 'Y_clean' in data else data['Y']

N = len(X_raw)
X = X_raw.reshape(N, -1).astype(np.float64)
Y = np.array([str(v).upper().strip() for v in Y_raw])

vocales = ['A', 'E', 'I', 'O', 'U']
vocal_to_idx = {v: i for i, v in enumerate(vocales)}
y_true_int = np.array([vocal_to_idx[v] for v in Y])

colores_oficiales = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# Evaluacion GMM 3D no supervisada
def evaluar_gmm_3d(Z, y_true):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=20)
    preds = gmm.fit_predict(Z)
    cm = confusion_matrix(y_true, preds)
    r_ind, c_ind = linear_sum_assignment(-cm)
    map_dict = {c: r for r, c in zip(r_ind, c_ind)}
    mapped_preds = np.array([map_dict[p] for p in preds])
    acc = np.mean(mapped_preds == y_true) * 100.0
    return acc, gmm

# 2. PCA 3D Referencia
pca_3d = PCA(n_components=3, random_state=42)
Z_pca_3d = pca_3d.fit_transform(X)
var_pca_3d = np.sum(pca_3d.explained_variance_ratio_) * 100.0
acc_pca_3d, _ = evaluar_gmm_3d(Z_pca_3d, y_true_int)

# Orientar /a/ hacia +Z, /i/ hacia +X
if np.mean(Z_pca_3d[Y == 'A', 2]) < 0:
    Z_pca_3d[:, 2] = -Z_pca_3d[:, 2]
if np.mean(Z_pca_3d[Y == 'I', 0]) < 0:
    Z_pca_3d[:, 0] = -Z_pca_3d[:, 0]

# 3. Barrido de k en Isomap 3D
k_vals = [6, 12, 20, 35, 60]
isomap_3d_results = {}

for k in k_vals:
    print(f"Calculando Isomap 3D con k={k}...")
    iso = Isomap(n_neighbors=k, n_components=3)
    Z_k = iso.fit_transform(X)
    
    # Orientar /a/ hacia +Z, /i/ hacia +X
    if np.mean(Z_k[Y == 'A', 2]) < 0:
        Z_k[:, 2] = -Z_k[:, 2]
    if np.mean(Z_k[Y == 'I', 0]) < 0:
        Z_k[:, 0] = -Z_k[:, 0]
        
    acc_k, _ = evaluar_gmm_3d(Z_k, y_true_int)
    
    # Centroides
    c_o = np.mean(Z_k[Y == 'O'], axis=0)
    c_u = np.mean(Z_k[Y == 'U'], axis=0)
    
    # Distancia al origen en el plano X-Y (magnitud radial en plano basal)
    r_u = np.linalg.norm(c_u[:2])
    r_o = np.linalg.norm(c_o[:2])
    
    # Distancia 3D al centroide de /a/
    c_a = np.mean(Z_k[Y == 'A'], axis=0)
    d_a_o = np.linalg.norm(c_o - c_a)
    d_a_u = np.linalg.norm(c_u - c_a)
    
    # Quiralidad / orden relativo respecto a /a/:
    # En la fisiologia vocal, /o/ esta mas cerca de /a/ que /u/ (d(a, o) < d(a, u))
    quiralidad_orden = (d_a_o < d_a_u)
    
    isomap_3d_results[k] = {
        'Z': Z_k,
        'acc': acc_k,
        'c_o': c_o,
        'c_u': c_u,
        'c_a': c_a,
        'r_u': r_u,
        'r_o': r_o,
        'd_a_o': d_a_o,
        'd_a_u': d_a_u,
        'quiralidad_orden': quiralidad_orden
    }

# 4. Tabla Cuantitativa en Consola
print("\n" + "=" * 90)
print(f"{'Método':^14} | {'k':^6} | {'GMM 3D Acc':^12} | {'d(a, o)':^10} | {'d(a, u)':^10} | {'r_u (Plano XY)':^14} | {'r_o (Plano XY)':^14} | {'Orden Fisiol.':^14}")
print("=" * 90)

c_a_pca = np.mean(Z_pca_3d[Y == 'A'], axis=0)
c_o_pca = np.mean(Z_pca_3d[Y == 'O'], axis=0)
c_u_pca = np.mean(Z_pca_3d[Y == 'U'], axis=0)
d_ao_pca = np.linalg.norm(c_o_pca - c_a_pca)
d_au_pca = np.linalg.norm(c_u_pca - c_a_pca)
r_u_pca = np.linalg.norm(c_u_pca[:2])
r_o_pca = np.linalg.norm(c_o_pca[:2])
pca_orden_ok = (d_ao_pca < d_au_pca)

print(f"{'PCA 3D':^14} | {'Global':^6} | {acc_pca_3d:^12.2f}% | {d_ao_pca:^10.2f} | {d_au_pca:^10.2f} | {r_u_pca:^14.2f} | {r_o_pca:^14.2f} | {'Correcto' if pca_orden_ok else 'Invertido':^14}")

for k in k_vals:
    res = isomap_3d_results[k]
    o_str = "Correcto (d(a,o)<d(a,u))" if res['quiralidad_orden'] else "Invertido"
    print(f"{'Isomap 3D':^14} | {k:^6} | {res['acc']:^12.2f}% | {res['d_a_o']:^10.2f} | {res['d_a_u']:^10.2f} | {res['r_u']:^14.2f} | {res['r_o']:^14.2f} | {o_str:^14}")
print("=" * 90)

# 5. Generacion del Grafico Multipanel 3D (2 filas x 3 columnas)
out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/analisis_dimensionalidad")
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, "barrido_k_isomap_3d_lucas.png")

fig = plt.figure(figsize=(20, 12), facecolor='white')

# Panel 0: PCA 3D
ax0 = fig.add_subplot(2, 3, 1, projection='3d')
ax0.set_facecolor('white')
for v in vocales:
    mask = (Y == v)
    c = colores_oficiales[v]
    ax0.scatter(Z_pca_3d[mask, 0], Z_pca_3d[mask, 1], Z_pca_3d[mask, 2], c=c, alpha=0.55, s=25, edgecolors='none')
    cent = np.mean(Z_pca_3d[mask], axis=0)
    ax0.scatter(cent[0], cent[1], cent[2], c=c, marker='D', s=100, edgecolors='black', linewidths=1.2)
    ax0.text(cent[0], cent[1], cent[2], f" /{v.lower()}/", fontsize=10, fontweight='bold')

ax0.set_title(f"PCA 3D: Referencia (Var: {var_pca_3d:.1f}%)\nGMM 3D: {acc_pca_3d:.2f}%", fontsize=11, fontweight='bold')
ax0.set_xlabel("PC 1", fontsize=9)
ax0.set_ylabel("PC 2", fontsize=9)
ax0.set_zlabel("PC 3", fontsize=9)
ax0.view_init(elev=22, azim=45)

# Paneles 1 a 5: Isomap 3D
for idx, k in enumerate(k_vals):
    ax = fig.add_subplot(2, 3, idx + 2, projection='3d')
    ax.set_facecolor('white')
    res = isomap_3d_results[k]
    Z_k = res['Z']
    
    for v in vocales:
        mask = (Y == v)
        c = colores_oficiales[v]
        ax.scatter(Z_k[mask, 0], Z_k[mask, 1], Z_k[mask, 2], c=c, alpha=0.55, s=25, edgecolors='none')
        cent = np.mean(Z_k[mask], axis=0)
        ax.scatter(cent[0], cent[1], cent[2], c=c, marker='D', s=100, edgecolors='black', linewidths=1.2)
        ax.text(cent[0], cent[1], cent[2], f" /{v.lower()}/", fontsize=10, fontweight='bold')
        
    orden_label = "Física OK" if res['quiralidad_orden'] else "Invertida"
    ax.set_title(f"Isomap 3D: k = {k}\nGMM 3D: {res['acc']:.2f}% | {orden_label}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Dim 1", fontsize=9)
    ax.set_ylabel("Dim 2", fontsize=9)
    ax.set_zlabel("Dim 3", fontsize=9)
    ax.view_init(elev=22, azim=45)

plt.tight_layout()
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n[OK] Panel comparativo 3D guardado exitosamente en:\n     {out_fig}")
