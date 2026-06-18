import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap

def generar_graficos_y_scores():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", "base_de_datos_letras"))
    
    datasets = {}
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("dataset_features") and file.endswith(".csv"):
                rel_path = os.path.relpath(root, base_dir)
                if rel_path == ".":
                    method_name = file.replace(".csv", "")
                else:
                    method_name = rel_path
                
                # Solo evaluar los nuestros
                if method_name.startswith("EXP_"):
                    datasets[method_name] = os.path.join(root, file)
                    
    if not datasets:
        print("No hay datasets experimentales.")
        return
        
    resultados = []
    
    for nombre, csv_path in datasets.items():
        print(f"Generando gráficos para: {nombre}")
        out_dir = os.path.dirname(csv_path)
        
        df = pd.read_csv(csv_path)
        df_vocales = df[df['Vocal'] != '0'].copy()
        
        if df_vocales.empty:
            continue
            
        features = [col for col in df_vocales.columns if col not in ['Toma', 'Vocal']]
        X_raw = df_vocales[features].values
        y = df_vocales['Vocal'].values
        
        # OBLIGATORIO: Escalar los datos para PCA/UMAP
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)
        
        plt.rcParams.update({'font.size': 12})
        unique_vocales = np.unique(y)
        palette = sns.color_palette("Set1", n_colors=len(unique_vocales))
        
        # PCA
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X)
        score_pca = silhouette_score(X_pca, y)
        
        df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
        df_pca['Vocal'] = y
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, vocal in enumerate(unique_vocales):
            subset = df_pca[df_pca['Vocal'] == vocal]
            ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], 
                       c=[palette[i]], label=vocal, s=60, alpha=0.8)
        
        ax.set_title(f'PCA de Vocales (Silhouette: {score_pca:.3f})')
        ax.set_xlabel(f'PC1')
        ax.set_ylabel(f'PC2')
        ax.set_zlabel(f'PC3')
        ax.legend(title='Vocal', loc='best')
        plt.savefig(os.path.join(out_dir, "cluster_pca.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # PCA 2D
        fig2d = plt.figure(figsize=(10, 8))
        ax2d = fig2d.add_subplot(111)
        for i, vocal in enumerate(unique_vocales):
            subset = df_pca[df_pca['Vocal'] == vocal]
            ax2d.scatter(subset['PC1'], subset['PC2'], 
                       c=[palette[i]], label=vocal, s=60, alpha=0.8)
        ax2d.set_title(f'PCA 2D de Vocales')
        ax2d.set_xlabel('PC1')
        ax2d.set_ylabel('PC2')
        ax2d.legend(title='Vocal', loc='best')
        plt.savefig(os.path.join(out_dir, "cluster_pca_2d.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # UMAP 3D
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reducer = umap.UMAP(n_components=3, random_state=42)
            X_umap = reducer.fit_transform(X)
        score_umap = silhouette_score(X_umap, y)
        
        df_umap = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2', 'UMAP3'])
        df_umap['Vocal'] = y
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        for i, vocal in enumerate(unique_vocales):
            subset = df_umap[df_umap['Vocal'] == vocal]
            ax.scatter(subset['UMAP1'], subset['UMAP2'], subset['UMAP3'], 
                       c=[palette[i]], label=vocal, s=60, alpha=0.8)
                       
        ax.set_title(f'UMAP de Vocales (Silhouette: {score_umap:.3f})')
        ax.set_xlabel('UMAP1')
        ax.set_ylabel('UMAP2')
        ax.set_zlabel('UMAP3')
        ax.legend(title='Vocal', loc='best')
        plt.savefig(os.path.join(out_dir, "cluster_umap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # UMAP 2D
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            reducer_2d = umap.UMAP(n_components=2, random_state=42)
            X_umap_2d = reducer_2d.fit_transform(X)
            
        df_umap_2d = pd.DataFrame(data=X_umap_2d, columns=['UMAP1', 'UMAP2'])
        df_umap_2d['Vocal'] = y
        
        fig2d_u = plt.figure(figsize=(10, 8))
        ax2d_u = fig2d_u.add_subplot(111)
        for i, vocal in enumerate(unique_vocales):
            subset = df_umap_2d[df_umap_2d['Vocal'] == vocal]
            ax2d_u.scatter(subset['UMAP1'], subset['UMAP2'], 
                       c=[palette[i]], label=vocal, s=60, alpha=0.8)
        ax2d_u.set_title(f'UMAP 2D de Vocales')
        ax2d_u.set_xlabel('UMAP1')
        ax2d_u.set_ylabel('UMAP2')
        ax2d_u.legend(title='Vocal', loc='best')
        plt.savefig(os.path.join(out_dir, "cluster_umap_2d.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Cálculo de distancias entre centroides en UMAP y PCA
        centroids_umap = {}
        centroids_pca = {}
        for vocal in unique_vocales:
            centroids_umap[vocal] = df_umap[df_umap['Vocal'] == vocal][['UMAP1', 'UMAP2', 'UMAP3']].mean().values
            centroids_pca[vocal] = df_pca[df_pca['Vocal'] == vocal][['PC1', 'PC2', 'PC3']].mean().values
            
        import itertools
        from scipy.spatial.distance import euclidean
        distancias_umap = {}
        distancias_pca = {}
        for v1, v2 in itertools.combinations(unique_vocales, 2):
            dist_u = euclidean(centroids_umap[v1], centroids_umap[v2])
            dist_p = euclidean(centroids_pca[v1], centroids_pca[v2])
            distancias_umap[f"{v1}-{v2}"] = dist_u
            distancias_pca[f"{v1}-{v2}"] = dist_p
            
        resultados.append({
            "Configuración": nombre,
            "Silhouette PCA": score_pca,
            "Silhouette UMAP": score_umap,
            "Distancias UMAP": distancias_umap,
            "Distancias PCA": distancias_pca
        })
        
    # Ranking
    # Ranking Formateado como Tabla Markdown
    print("\n# RANKING FINAL DE SEPARACIÓN DE VOCALES\n")
    resultados.sort(key=lambda x: x["Silhouette UMAP"], reverse=True)
    
    # Encabezados de la tabla (incluyendo las comparaciones E-I y A-E como ejemplos clave)
    print("| Configuración | Sil. UMAP | Sil. PCA | Dist. UMAP (E-I) | Dist. PCA (E-I) | Dist. UMAP (Peor Par) |")
    print("|--------------|-----------|----------|------------------|-----------------|-----------------------|")
    
    table_data = []
    columns = ["Configuración", "Sil. UMAP", "Sil. PCA", "Dist. UMAP (E-I)", "Dist. PCA (E-I)", "Peor Par (UMAP)"]
    
    for r in resultados:
        dists_umap = r["Distancias UMAP"]
        dists_pca = r["Distancias PCA"]
        
        # Obtener el par más confundido en UMAP
        peor_par_umap = min(dists_umap.items(), key=lambda x: x[1])
        
        # Extraer específicamente E-I si existe, sino 0
        ei_umap = dists_umap.get("E-I", dists_umap.get("I-E", 0.0))
        ei_pca = dists_pca.get("E-I", dists_pca.get("I-E", 0.0))
        
        conf = r['Configuración'].replace('EXP_', '')
        print(f"| {conf} | {r['Silhouette UMAP']:.4f} | {r['Silhouette PCA']:.4f} | **{ei_umap:.2f}** | {ei_pca:.2f} | {peor_par_umap[0]} ({peor_par_umap[1]:.2f}) |")
        
        table_data.append([
            conf, 
            f"{r['Silhouette UMAP']:.4f}", 
            f"{r['Silhouette PCA']:.4f}", 
            f"{ei_umap:.2f}", 
            f"{ei_pca:.2f}", 
            f"{peor_par_umap[0]} ({peor_par_umap[1]:.2f})"
        ])
        
    print("\n*Nota: Distancias mayores a 3.0 en UMAP generalmente indican una buena separación visual. En PCA dependen de la escala.*")
    
    # ---------------------------------------------------------
    # Generar Tabla PNG Bonita
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, max(2, len(table_data) * 0.6 + 1)))
    ax.axis('off')
    ax.axis('tight')
    
    table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)
    
    # Estilizar la tabla
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1F2833')
        else:
            if col == 3: # Columna Dist. UMAP (E-I)
                val = float(table_data[row-1][col])
                if val < 1.5:
                    cell.set_facecolor('#ffcccc') # Rojo (malo)
                elif val >= 3.0:
                    cell.set_facecolor('#ccffcc') # Verde (bueno)
                else:
                    cell.set_facecolor('#ffffcc') # Amarillo (regular)

    plt.title("Ranking Final de Separación de Vocales (UMAP/PCA)", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    out_img = os.path.join(base_dir, "ranking_tabla.png")
    plt.savefig(out_img, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n¡Se ha generado una tabla visual hermosa en:\n{out_img}")

if __name__ == "__main__":
    generar_graficos_y_scores()
