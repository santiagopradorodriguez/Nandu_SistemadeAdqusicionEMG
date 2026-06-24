import sys

def modify_pca():
    with open('EMG_desarrollo/analysis/pca_motor.py', 'r') as f:
        content = f.read()

    # 1. Update plot_pca_results signature
    old_sig = "def plot_pca_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, variance_ratio, n_components):"
    new_sig = "def plot_pca_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, variance_ratio, n_components, sil_score_2d=float('nan'), sil_score_3d=float('nan')):"
    content = content.replace(old_sig, new_sig)

    # 2. Update latex generation in plot_pca_results
    old_latex = """    latex_code += f"    \\\\item \\\\textbf{{Total de Pulsos (N):}} {len(labels)}\\n"
    latex_code += "\\\\end{itemize}\\n\\n\""""
    new_latex = """    latex_code += f"    \\\\item \\\\textbf{{Total de Pulsos (N):}} {len(labels)}\\n"
    sil_str = f"{sil_score_2d:.4f}" if not np.isnan(sil_score_2d) else "N/A"
    sil_str_3d = f"{sil_score_3d:.4f}" if not np.isnan(sil_score_3d) else "N/A"
    latex_code += f"    \\\\item \\\\textbf{{Silhouette Score (2D):}} {sil_str}\\n"
    latex_code += f"    \\\\item \\\\textbf{{Silhouette Score (3D):}} {sil_str_3d}\\n"
    latex_code += "\\\\end{itemize}\\n\\n\""""
    content = content.replace(old_latex, new_latex)

    # 3. Calculate sil_score_3d
    old_sil = """    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
    except:
        sil_score_2d = float('nan')"""
    new_sil = """    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
        sil_score_3d = silhouette_score(embedding_3d, y)
        logger(f"[INFO] Silhouette Score (3D): {sil_score_3d:.4f}")
    except:
        sil_score_2d = float('nan')
        sil_score_3d = float('nan')"""
    content = content.replace(old_sil, new_sil)

    # 4. Update plot_pca_results call
    old_call = """    plot_pca_results(
        embedding_2d, embedding_3d, y, Roles, 
        out_dir_base, folder_name, variance_ratio, n_components
    )"""
    new_call = """    plot_pca_results(
        embedding_2d, embedding_3d, y, Roles, 
        out_dir_base, folder_name, variance_ratio, n_components,
        sil_score_2d, sil_score_3d
    )"""
    content = content.replace(old_call, new_call)

    # 5. K-Means 3D Matriz Confusion
    # find where 2D matrix is saved
    old_kmeans_end = """        cm_path = os.path.join(out_dir_final, "matriz_confusion_pca.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")"""
    
    new_kmeans_end = """        cm_path = os.path.join(out_dir_final, "matriz_confusion_pca.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")

        # K-Means 3D
        kmeans_3d = KMeans(n_clusters=k, random_state=42)
        kmeans_3d.fit(embedding_3d[Roles == 'train'])
        test_preds_3d = kmeans_3d.predict(embedding_3d[Roles == 'test'])
        
        train_preds_3d = kmeans_3d.labels_
        cluster_to_vocal_3d = {}
        for cluster_id in range(k):
            vocales_in_cluster = y_train_str[train_preds_3d == cluster_id]
            if len(vocales_in_cluster) > 0:
                cluster_to_vocal_3d[cluster_id] = pd.Series(vocales_in_cluster).mode()[0]
            else:
                cluster_to_vocal_3d[cluster_id] = "Desconocido"
                
        test_preds_vocales_3d = [cluster_to_vocal_3d.get(c, "Desconocido") for c in test_preds_3d]
        
        cm_3d = confusion_matrix(y_test_true, test_preds_vocales_3d, labels=unique_vocales)
        cm_pct_3d = np.zeros_like(cm_3d, dtype=float)
        row_sums_3d = cm_3d.sum(axis=1)
        for i in range(cm_3d.shape[0]):
            if row_sums_3d[i] > 0:
                cm_pct_3d[i] = (cm_3d[i] / row_sums_3d[i]) * 100
                
        annot_data_3d = np.empty_like(cm_3d, dtype=object)
        for i in range(cm_3d.shape[0]):
            for j in range(cm_3d.shape[1]):
                annot_data_3d[i, j] = f"{cm_pct_3d[i, j]:.1f}%\\n({cm_3d[i, j]})"
                
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_3d, annot=annot_data_3d, fmt='', cmap='Blues', xticklabels=unique_vocales, yticklabels=unique_vocales)
        plt.title('Matriz de Confusión (Test Data) - K-Means sobre PCA 3D')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción (K-Means)')
        cm_path_3d = os.path.join(out_dir_final, "matriz_confusion_pca_3d.png")
        plt.savefig(cm_path_3d, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión 3D generada: {cm_path_3d}")"""
        
    content = content.replace(old_kmeans_end, new_kmeans_end)
    
    with open('EMG_desarrollo/analysis/pca_motor.py', 'w') as f:
        f.write(content)

def modify_umap():
    with open('EMG_desarrollo/analysis/umap_motor.py', 'r') as f:
        content = f.read()

    # 1. Update plot_umap_results signature
    old_sig = "def plot_umap_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, mediciones_aceptadas, mediciones_rechazadas, n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d):"
    new_sig = "def plot_umap_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, mediciones_aceptadas, mediciones_rechazadas, n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d=float('nan'), sil_score_3d=float('nan')):"
    content = content.replace(old_sig, new_sig)

    # 2. Update latex generation in plot_umap_results
    old_latex = """    sil_str = f"{sil_score_2d:.4f}" if not np.isnan(sil_score_2d) else "N/A"
    latex_code += f"    \\\\item \\\\textbf{{Silhouette Score (2D):}} {sil_str}\\n"
    latex_code += "\\\\end{itemize}\\n\\n\""""
    new_latex = """    sil_str = f"{sil_score_2d:.4f}" if not np.isnan(sil_score_2d) else "N/A"
    sil_str_3d = f"{sil_score_3d:.4f}" if not np.isnan(sil_score_3d) else "N/A"
    latex_code += f"    \\\\item \\\\textbf{{Silhouette Score (2D):}} {sil_str}\\n"
    latex_code += f"    \\\\item \\\\textbf{{Silhouette Score (3D):}} {sil_str_3d}\\n"
    latex_code += "\\\\end{itemize}\\n\\n\""""
    content = content.replace(old_latex, new_latex)

    # 3. Calculate sil_score_3d
    old_sil = """    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
    except:
        sil_score_2d = float('nan')"""
    new_sil = """    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
        sil_score_3d = silhouette_score(embedding_3d, y)
        logger(f"[INFO] Silhouette Score (3D): {sil_score_3d:.4f}")
    except:
        sil_score_2d = float('nan')
        sil_score_3d = float('nan')"""
    content = content.replace(old_sil, new_sil)

    # 4. Update plot_umap_results call
    old_call = """    img_path_2d, img_path_3d, tex_path = plot_umap_results(
        embedding_2d, embedding_3d, y, roles, out_dir_final, folder_name, 
        mediciones_aceptadas, mediciones_rechazadas, 
        n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d
    )"""
    new_call = """    img_path_2d, img_path_3d, tex_path = plot_umap_results(
        embedding_2d, embedding_3d, y, roles, out_dir_final, folder_name, 
        mediciones_aceptadas, mediciones_rechazadas, 
        n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d, sil_score_3d
    )"""
    content = content.replace(old_call, new_call)

    # 5. K-Means 3D Matriz Confusion
    old_kmeans_end = """        cm_path = os.path.join(out_dir_final, "matriz_confusion_umap.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")"""
    
    new_kmeans_end = """        cm_path = os.path.join(out_dir_final, "matriz_confusion_umap.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")

        # K-Means 3D
        kmeans_3d = KMeans(n_clusters=k, random_state=42)
        kmeans_3d.fit(embedding_3d[roles == 'train'])
        test_preds_3d = kmeans_3d.predict(embedding_3d[roles == 'test'])
        
        train_preds_3d = kmeans_3d.labels_
        cluster_to_vocal_3d = {}
        for cluster_id in range(k):
            vocales_in_cluster = y_train_str[train_preds_3d == cluster_id]
            if len(vocales_in_cluster) > 0:
                cluster_to_vocal_3d[cluster_id] = pd.Series(vocales_in_cluster).mode()[0]
            else:
                cluster_to_vocal_3d[cluster_id] = "Desconocido"
                
        test_preds_vocales_3d = [cluster_to_vocal_3d.get(c, "Desconocido") for c in test_preds_3d]
        
        cm_3d = confusion_matrix(y_test_true, test_preds_vocales_3d, labels=unique_vocales)
        cm_pct_3d = np.zeros_like(cm_3d, dtype=float)
        row_sums_3d = cm_3d.sum(axis=1)
        for i in range(cm_3d.shape[0]):
            if row_sums_3d[i] > 0:
                cm_pct_3d[i] = (cm_3d[i] / row_sums_3d[i]) * 100
                
        annot_data_3d = np.empty_like(cm_3d, dtype=object)
        for i in range(cm_3d.shape[0]):
            for j in range(cm_3d.shape[1]):
                annot_data_3d[i, j] = f"{cm_pct_3d[i, j]:.1f}%\\n({cm_3d[i, j]})"
                
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_3d, annot=annot_data_3d, fmt='', cmap='Blues', xticklabels=unique_vocales, yticklabels=unique_vocales)
        plt.title('Matriz de Confusión (Test Data) - K-Means sobre UMAP 3D')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción (K-Means)')
        cm_path_3d = os.path.join(out_dir_final, "matriz_confusion_umap_3d.png")
        plt.savefig(cm_path_3d, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión 3D generada: {cm_path_3d}")"""
        
    content = content.replace(old_kmeans_end, new_kmeans_end)
    
    with open('EMG_desarrollo/analysis/umap_motor.py', 'w') as f:
        f.write(content)

modify_pca()
modify_umap()
