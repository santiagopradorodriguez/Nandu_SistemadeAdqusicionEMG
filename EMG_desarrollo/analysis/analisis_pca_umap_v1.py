import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import os

def plot_scatter(X, labels, title, filename):
    vocales = ['A', 'E', 'I', 'O', 'U']
    colores = {'A': 'red', 'E': 'green', 'I': 'blue', 'O': 'orange', 'U': 'purple'}
    
    plt.figure(figsize=(10, 8))
    for v in vocales:
        mask = labels == v
        plt.scatter(X[mask, 0], X[mask, 1], c=colores[v], label=v, alpha=0.7, edgecolors='k')
        
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    data_dir = r"C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\base_de_datos_letras"
    csv_file = os.path.join(data_dir, "dataset_features_v1.csv")
    
    if not os.path.exists(csv_file):
        print(f"No se encontro {csv_file}")
        return

    df = pd.read_csv(csv_file)
    print(f"Dataset cargado: {len(df)} muestras.")
    
    features = ['Max_Ch0', 'Max_Ch1', 'Max_Ch2']
    X = df[features].values
    y = df['Vocal'].values
    
    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n--- Analisis PCA (2D) ---")
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)
    sil_pca = silhouette_score(X_pca, y)
    print(f"Silhouette Score (PCA): {sil_pca:.3f}")
    
    # Calcular Distancias PCA
    vocales_unicas = np.unique(y)
    centroids_pca = {v: np.mean(X_pca[y == v], axis=0) for v in vocales_unicas}
    print("Matriz de Distancia entre Centroides (Euclidiana) - PCA:")
    df_dist_pca = pd.DataFrame(index=vocales_unicas, columns=vocales_unicas)
    for v1 in vocales_unicas:
        for v2 in vocales_unicas:
            df_dist_pca.loc[v1, v2] = np.linalg.norm(centroids_pca[v1] - centroids_pca[v2])
    pd.options.display.float_format = '{:.2f}'.format
    print(df_dist_pca)
    
    plot_scatter(X_pca, y, f"PCA de Pulsos EMG (Silhouette: {sil_pca:.3f})", os.path.join(data_dir, "pca_2d_v1.png"))
    
    print("\n--- Analisis UMAP (2D) ---")
    reducer = umap.UMAP(random_state=42)
    X_umap = reducer.fit_transform(X_scaled)
    sil_umap = silhouette_score(X_umap, y)
    print(f"Silhouette Score (UMAP): {sil_umap:.3f}")
    
    # Calcular Distancias UMAP
    centroids_umap = {v: np.mean(X_umap[y == v], axis=0) for v in vocales_unicas}
    print("Matriz de Distancia entre Centroides (Euclidiana) - UMAP:")
    df_dist_umap = pd.DataFrame(index=vocales_unicas, columns=vocales_unicas)
    for v1 in vocales_unicas:
        for v2 in vocales_unicas:
            df_dist_umap.loc[v1, v2] = np.linalg.norm(centroids_umap[v1] - centroids_umap[v2])
    print(df_dist_umap)
    
    plot_scatter(X_umap, y, f"UMAP de Pulsos EMG (Silhouette: {sil_umap:.3f})", os.path.join(data_dir, "umap_2d_v1.png"))
    
    print("\n¡Analisis completado! Graficos V1 guardados en base_de_datos_letras.")

if __name__ == "__main__":
    main()
