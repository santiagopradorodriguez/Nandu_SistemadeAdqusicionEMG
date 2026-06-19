import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import umap
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generador_pca_umap import extraer_features_concatenadas, evaluar_clustering_no_supervisado

base_dir = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\base_de_datos_electrodos'
out_dir = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\analysis\resultados_pca_umap'
os.makedirs(out_dir, exist_ok=True)

print("==== BARRIDO FINO DE HIPERPARÁMETROS (DSP) ====")
print("Escaneando directorio de mediciones...")
mediciones = []
dias_dir = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
for dia in dias_dir:
    dia_path = os.path.join(base_dir, dia)
    for m in os.listdir(dia_path):
        if os.path.isdir(os.path.join(dia_path, m)):
            mediciones.append(os.path.join(dia, m).replace('\\', '/'))
print(f"-> Se encontraron {len(mediciones)} tomas en total.")

snr_threshold = 0.5
outlier_contamination = 0.05
notch_q = 2.0
alpha_ruido = 1.0

# Espacio de búsqueda fino (MÁS GRANDE para dejar toda la noche)
smooth_ms_list = [40, 60, 80, 100, 120, 140, 160, 180, 200, 250]
target_len_list = [10, 15, 20, 25, 30, 40, 50, 75, 100]
resultados_acc = np.zeros((len(smooth_ms_list), len(target_len_list)))

total_iters = len(smooth_ms_list) * len(target_len_list)
print(f"\nIniciando Grid Search 2D: {total_iters} combinaciones en total.")

with tqdm(total=total_iters, desc="Optimizando") as pbar:
    for i, smooth_ms in enumerate(smooth_ms_list):
        for j, target_len in enumerate(target_len_list):
            try:
                X, Y, Tomas, _ = extraer_features_concatenadas(
                    base_dir=base_dir, mediciones=mediciones, alpha_ruido=alpha_ruido, 
                    smooth_ms=smooth_ms, notch_q=notch_q, target_len=target_len
                )
                
                # Limpieza de NaNs
                X_clean, Y_clean = [], []
                for idx in range(len(X)):
                    if not np.isnan(X[idx]).any():
                        X_clean.append(X[idx])
                        Y_clean.append(Y[idx])
                X_clean, Y_clean = np.array(X_clean), np.array(Y_clean)
                
                if len(X_clean) > 50:
                    clf = IsolationForest(contamination=outlier_contamination, random_state=42)
                    y_outliers = clf.fit_predict(X_clean)
                    mask = y_outliers == 1
                    X_clean, Y_clean = X_clean[mask], Y_clean[mask]
                    
                if len(X_clean) < 10:
                    resultados_acc[i, j] = 0
                else:
                    from sklearn.preprocessing import StandardScaler
                    X_scaled = StandardScaler().fit_transform(X_clean)
                    
                    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=3, metric='manhattan', random_state=42)
                    X_umap = reducer.fit_transform(X_scaled)
                    
                    acc, _, _, _, _ = evaluar_clustering_no_supervisado(X_umap, Y_clean, "UMAP")
                    resultados_acc[i, j] = acc
            except Exception as e:
                resultados_acc[i, j] = 0
                
            pbar.update(1)

# Guardar Resultados
df_res = pd.DataFrame(resultados_acc, index=[f"{ms}ms" for ms in smooth_ms_list], columns=[f"{pt}pts" for pt in target_len_list])
csv_path = os.path.join(out_dir, 'resultados_grid_fino.csv')
df_res.to_csv(csv_path)

# Generar Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_res, annot=True, fmt=".1f", cmap="magma", cbar_kws={'label': 'Accuracy UMAP (%)'})
plt.title("Grid Search Fino 2D: Accuracy Topológica\nAcoplamiento Envolvente vs Puntos de Remuestreo", pad=20, fontsize=14)
plt.ylabel("Envolvente (smooth_ms)", fontsize=12)
plt.xlabel("Puntos de Remuestreo (target_length)", fontsize=12)
plt.tight_layout()

img_path = os.path.join(out_dir, 'grid_search_fino_heatmap.png')
plt.savefig(img_path, dpi=300)
plt.close()

print(f"\n[!] BARRIDO FINALIZADO.")
print(f" -> Datos crudos guardados en: {csv_path}")
print(f" -> Gráfico Heatmap guardado en: {img_path}")
