import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import umap
import sys
def main():
    print("Iniciando análisis de clustering PCA/UMAP...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, "..", "base_de_datos_letras"))
    
    # Parse arguments
    method = "max_amp"
    use_scaler = True
    center_only = False
    for arg in sys.argv[1:]:
        if arg == "--no-scaler":
            use_scaler = False
        elif arg == "--center-only":
            center_only = True
        else:
            method = arg
    
    if method == "env":
        csv_path = os.path.join(data_dir, "dataset_features_env.csv")
        print("Usando dataset Envolvente Temporal.")
    elif method == "env_plus":
        csv_path = os.path.join(data_dir, "dataset_features_env_plus.csv")
        print("Usando dataset Envolvente Temporal PLUS (Lags + Stats + BIN).")
    elif method == "hierro":
        csv_path = os.path.join(data_dir, "dataset_features_hierro.csv")
        print("Usando dataset Hierro (Features estadísticas puras sin serie temporal).")
    elif method == "stft":
        csv_path = os.path.join(data_dir, "dataset_features_stft.csv")
        print("Usando dataset STFT.")
    elif method == "mav_corr":
        csv_path = os.path.join(data_dir, "dataset_features_mav_corr.csv")
        print("Usando dataset Híbrido 6D (MAV + Pearson).")
    elif method == "dios":
        csv_path = os.path.join(data_dir, "dataset_features_dios.csv")
        print("Usando dataset Modo Dios (Multidominio Avanzado).")
    elif method == "custom":
        csv_path = os.path.join(data_dir, "dataset_features_custom.csv")
        print("Usando dataset Custom (Features Cinemáticas y Físicas).")
    else:
        # Por defecto method == "max_amp"
        csv_path = os.path.join(data_dir, "dataset_features_max_amp.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_dir, "dataset_features.csv") # Compatibilidad hacia atrás
        print("Usando dataset Amplitud Clásica (Fuerza + Latencia).")
        
    if not os.path.exists(csv_path):
        print(f"Error: No se encontró el dataset {csv_path}. ¡Por favor haz click en 'Extraer Dataset' primero!")
        return
        
    # Cargar datos
    df = pd.read_csv(csv_path)
    print(f"Dataset cargado: {len(df)} ventanas fonéticas.")
    
    features = [col for col in df.columns if col not in ['Toma', 'Vocal'] and not col.endswith('_ZCR') and not col.endswith('_PeakLat')]
    
    # Rellenar cualquier valor vacío con 0.0 (por si hay ventanas de distinto largo por cambios de BPM)
    df[features] = df[features].fillna(0.0)
    
    X_raw = df[features].values
    y = df['Vocal'].values
    
    print(f"Dimensiones de entrada: {X_raw.shape[1]} features por muestra.")
    
    if center_only:
        print("Centrando en cero las columnas (sin modificar la varianza)...")
        scaler = StandardScaler(with_std=False)
        X = scaler.fit_transform(X_raw)
    elif use_scaler:
        print("Aplicando StandardScaler (centrado y varianza 1) a los datos...")
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
    else:
        print("Saltando StandardScaler (usando datos crudos).")
        X = X_raw
        
    # Aplicar PESO MODERADO a la Binarización de Trevisan (Desempate suave en PCA)
    bin_cols = [c for c in features if 'BIN' in c]
    if len(bin_cols) > 0:
        print(f"Ajustando peso de {len(bin_cols)} columnas BIN (Peso x2.0 para evitar colapso de clusters)...")
        bin_indices = [features.index(c) for c in bin_cols]
        X[:, bin_indices] *= 2.0
        
    # --- DIAGNÓSTICOS RECOMENDADOS (DEEPSEEK) ---
    print("\n--- DIAGNÓSTICO DE DATOS ---")
    print("Balance de clases (Vocales):")
    print(df['Vocal'].value_counts().to_string())
    
    # Evaluar Varianza 0
    varianzas = np.var(X, axis=0)
    print(f"Rango de varianzas de las features (Min: {varianzas.min():.4f}, Max: {varianzas.max():.4f})")
    
    # Calcular Silhouette en Alta Dimensión (95% Varianza)
    pca_full = PCA(n_components=0.95)
    try:
        X_pca_full = pca_full.fit_transform(X)
        if X_pca_full.shape[1] > 1:
            sil_full = silhouette_score(X_pca_full, y)
            print(f"Silhouette Score en ALTA DIMENSIÓN ({X_pca_full.shape[1]} componentes reteniendo 95% varianza): {sil_full:.3f}")
        else:
            print("No se pudo calcular Silhouette en alta dimensión (solo quedó 1 componente).")
    except Exception as e:
        print(f"Nota: No se pudo calcular PCA 95% varianza ({e})")
    print("----------------------------\n")

    def calcular_metricas(X_proj, labels, nombre):
        if len(np.unique(labels)) < 2: return 0.0
        # Silhouette Score
        sil = silhouette_score(X_proj, labels)
        print(f"\n[{nombre}] Silhouette Score: {sil:.3f} (1=Perfecto, 0=Superpuesto, -1=Incorrecto)")
        
        # Calcular centroides
        vocales_unicas = np.unique(labels)
        centroids = {}
        for vocal in vocales_unicas:
            centroids[vocal] = np.mean(X_proj[labels == vocal], axis=0)
            
        # Calcular e imprimir matriz de distancias euclidianas
        print(f"[{nombre}] Matriz de Distancia entre Centroides (Euclidiana):")
        df_dist = pd.DataFrame(index=vocales_unicas, columns=vocales_unicas)
        for v1 in vocales_unicas:
            for v2 in vocales_unicas:
                if v1 == v2:
                    df_dist.loc[v1, v2] = 0.0
                else:
                    dist = np.linalg.norm(centroids[v1] - centroids[v2])
                    df_dist.loc[v1, v2] = float(dist)
                    
        # Formatear la matriz para que se imprima limpia
        pd.options.display.float_format = '{:.2f}'.format
        print(df_dist)
        
        return sil
    
    # Configuración visual
    plt.rcParams.update({'font.size': 12})
    palette = sns.color_palette("Set1", n_colors=len(np.unique(y)))
    
    # ---------------------------------------------------------
    # 1. PCA (Principal Component Analysis) en 3D
    # ---------------------------------------------------------
    print("\nCalculando PCA 3D...")
    # Al ser 3 canales, podemos usar n_components=3
    n_comp_3d = min(3, X.shape[1])
    pca = PCA(n_components=n_comp_3d)
    X_pca = pca.fit_transform(X)
    
    if n_comp_3d == 3:
        sil_pca3d = calcular_metricas(X_pca, y, "PCA 3D")
    
    cols = [f'PC{i+1}' for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(data=X_pca, columns=cols)
    df_pca['Vocal'] = y
    
    fig = plt.figure(figsize=(10, 8))
    
    # Solo hacer grafico 3D si hay 3 componentes
    if X_pca.shape[1] >= 3:
        ax = fig.add_subplot(111, projection='3d')
        
        unique_vocales = np.unique(y)
        for i, vocal in enumerate(unique_vocales):
            subset = df_pca[df_pca['Vocal'] == vocal]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
                       c=[palette[i]], label=vocal, s=60, alpha=0.8)
                       
        ax.set_title(f'PCA de Vocales (Proyección 3D)\nSilhouette Score: {sil_pca3d:.3f}')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Varianza)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Varianza)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} Varianza)')
        ax.legend(title='Vocal', loc='best')
        
        pca_out = os.path.join(data_dir, "cluster_pca.png")
        plt.savefig(pca_out, dpi=300, bbox_inches='tight')
        
        # Ángulo alternativo 1
        ax.view_init(elev=30, azim=45)
        pca_out_alt1 = os.path.join(data_dir, "cluster_pca_ang1.png")
        plt.savefig(pca_out_alt1, dpi=300, bbox_inches='tight')
        
        # Ángulo alternativo 2
        ax.view_init(elev=20, azim=120)
        pca_out_alt2 = os.path.join(data_dir, "cluster_pca_ang2.png")
        plt.savefig(pca_out_alt2, dpi=300, bbox_inches='tight')
        
        plt.close()
        print(f"Gráficos PCA 3D guardados en la carpeta (3 ángulos diferentes)")
    else:
        plt.close()
        print("No hay suficientes componentes para gráfico 3D.")
    
    # (Código duplicado eliminado)
    # ---------------------------------------------------------
    # 1.5 PCA en 2D
    # ---------------------------------------------------------
    print("\nCalculando PCA 2D...")
    sil_pca2d = calcular_metricas(X_pca[:, :2], y, "PCA 2D")
    
    fig2d, ax2d = plt.subplots(figsize=(10, 8))
    for i, vocal in enumerate(unique_vocales):
        subset = df_pca[df_pca['Vocal'] == vocal]
        ax2d.scatter(subset['PC1'], subset['PC2'], 
                     c=[palette[i]], label=vocal, s=60, alpha=0.8)
                     
    ax2d.set_title(f'PCA de Vocales (Proyección 2D)\nSilhouette Score: {sil_pca2d:.3f}')
    ax2d.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} Varianza)')
    ax2d.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} Varianza)')
    ax2d.legend(title='Vocal', loc='best')
    ax2d.grid(True, linestyle='--', alpha=0.5)
    
    pca2d_out = os.path.join(data_dir, "cluster_pca_2d.png")
    plt.savefig(pca2d_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico PCA 2D guardado en {pca2d_out}")
    
    # ---------------------------------------------------------
    # 2. UMAP (Uniform Manifold Approximation and Projection)
    # ---------------------------------------------------------
    print("\nCalculando UMAP 2D...")
    # Ajustar n_neighbors basado en el tamaño del dataset
    n_neighbors = min(15, len(X) - 1)
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42, # Para reproducibilidad
        verbose=True
    )
    X_umap = reducer.fit_transform(X)
    sil_umap2d = calcular_metricas(X_umap, y, "UMAP 2D")
    
    df_umap = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2'])
    df_umap['Vocal'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='UMAP1', y='UMAP2',
        hue='Vocal',
        palette=palette,
        data=df_umap,
        legend="full",
        alpha=0.7,
        s=60
    )
    plt.title(f'UMAP de Vocales 2D\nSilhouette Score: {sil_umap2d:.3f}')
    plt.xlabel('UMAP Dimensión 1')
    plt.ylabel('UMAP Dimensión 2')
    plt.grid(True, alpha=0.3)
    
    umap_out = os.path.join(data_dir, "cluster_umap.png")
    plt.savefig(umap_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico UMAP guardado en {umap_out}")
    
    # ---------------------------------------------------------
    # 3. UMAP en 3D
    # ---------------------------------------------------------
    print("\nCalculando UMAP 3D...")
    reducer_3d = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_components=3,
        metric='euclidean',
        random_state=42,
        verbose=False
    )
    X_umap_3d = reducer_3d.fit_transform(X)
    sil_umap3d = calcular_metricas(X_umap_3d, y, "UMAP 3D")
    
    df_umap_3d = pd.DataFrame(data=X_umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
    df_umap_3d['Vocal'] = y
    
    fig_umap3d = plt.figure(figsize=(10, 8))
    ax_umap3d = fig_umap3d.add_subplot(111, projection='3d')
    
    for i, vocal in enumerate(unique_vocales):
        subset = df_umap_3d[df_umap_3d['Vocal'] == vocal]
        ax_umap3d.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], 
                          c=[palette[i]], label=vocal, s=60, alpha=0.8)
                          
    ax_umap3d.set_title(f'UMAP de Vocales (Proyección 3D)\nSilhouette Score: {sil_umap3d:.3f}')
    ax_umap3d.set_xlabel('UMAP 1')
    ax_umap3d.set_ylabel('UMAP 2')
    ax_umap3d.set_zlabel('UMAP 3')
    ax_umap3d.legend(title='Vocal', loc='best')
    
    umap3d_out = os.path.join(data_dir, "cluster_umap_3d.png")
    plt.savefig(umap3d_out, dpi=300, bbox_inches='tight')
    
    ax_umap3d.view_init(elev=30, azim=45)
    plt.savefig(os.path.join(data_dir, "cluster_umap_3d_ang1.png"), dpi=300, bbox_inches='tight')
    
    ax_umap3d.view_init(elev=20, azim=120)
    plt.savefig(os.path.join(data_dir, "cluster_umap_3d_ang2.png"), dpi=300, bbox_inches='tight')
    
    plt.close()
    print("Gráficos UMAP 3D guardados en la carpeta (3 ángulos diferentes).")
    
    print("\n¡Análisis de Clustering completado con éxito!")

if __name__ == "__main__":
    main()
