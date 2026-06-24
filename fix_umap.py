import sys

with open('EMG_desarrollo/analysis/umap_motor.py', 'r') as f:
    content = f.read()

# We need to replace `ejecutar_umap` entirely.
# Let's find the start of `def ejecutar_umap`
start_idx = content.find("def ejecutar_umap(asignaciones_vocales")
if start_idx == -1:
    print("Could not find ejecutar_umap")
    sys.exit(1)

new_ejecutar_umap = """def ejecutar_umap(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, vector_mode, n_neighbors, min_dist, is_supervised, run_kmeans, import_pca, pca_csv_path, logger=print):
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    import os
    import csv
    import json
    import umap
    
    logger("\\n" + "="*50)
    logger(f" INICIANDO ANÁLISIS UMAP ({'SUPERVISADO' if is_supervised else 'NO SUPERVISADO'})")
    logger("="*50)
    
    mediciones_aceptadas = []
    mediciones_rechazadas = []
    
    out_dir_general = list(asignaciones_vocales.keys())[0] if asignaciones_vocales else ""
    if out_dir_general:
        session_dir = os.path.dirname(out_dir_general)
        base_umap_dir = os.path.join(session_dir, "UMAP")
        os.makedirs(base_umap_dir, exist_ok=True)
        
    if import_pca and pca_csv_path and os.path.exists(pca_csv_path):
        logger(f"Importando datos directamente desde PCA: {os.path.basename(pca_csv_path)}")
        df = pd.read_csv(pca_csv_path)
        cols_to_drop = ['Toma', 'Rol', 'Vocal']
        X = df.drop(columns=[c for c in cols_to_drop if c in df.columns]).values
        y = df['Vocal'].values
        roles = df['Rol'].values if 'Rol' in df.columns else np.array(['train']*len(y))
        
        vector_mode = "PCA"
        for v in sorted(list(set(y))):
            c = np.sum(y == v)
            mediciones_aceptadas.append(("Importado_PCA", v, c, c))
    else:
        logger("Utilizando Motor Unificado de Extracción (PCA/UMAP)...")
        from analysis.pca_motor import build_pca_features
        X_full, y, roles, Tomas, med_acc_dict, med_rej_dict, info_pulsos = build_pca_features(
            asignaciones_vocales, canales_seleccionados, mapped_names, logger,
            filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo
        )
        
        if len(X_full) < 5:
            raise ValueError("No hay suficientes datos válidos para ejecutar UMAP (menos de 5 pulsos). Ajuste el filtro SNR.")
            
        # Adaptar matriz segun vector_mode
        if vector_mode == "Completa":
            X = X_full
            logger("Vector Mode: Completa (Matriz cruda 100 muestras/canal preservada)")
        elif vector_mode == "Picos":
            # Reshape a (N, num_canales, 100) y calcular max
            num_canales = len(canales_seleccionados)
            X_reshaped = X_full.reshape(X_full.shape[0], num_canales, 100)
            X = np.max(np.abs(X_reshaped), axis=2)
            logger("Vector Mode: Picos (Matriz reducida al pico máximo absoluto de cada canal tras filtrado robusto)")
        else:
            X = X_full
            
        # Generar formato para reporte
        for v, pulsos in med_acc_dict.items():
            c = len(pulsos)
            mediciones_aceptadas.append(("Pipeline_Unificado", v, c, c))
            
        for v, pulsos in med_rej_dict.items():
            for p in pulsos:
                mediciones_rechazadas.append((p, v, "SNR_Unified", 0.0))
                
    logger(f"\\n[INFO] Matriz de características X lista: {X.shape[0]} pulsos, {X.shape[1]} dimensiones.")
    
    # Preparar sufijo único y subcarpeta
    snr_lim_str = f"{filtro_snr_limite:.1f}".replace('.', '-')
    dist_str = f"{min_dist:.1f}".replace('.', '-')
    base_folder_name = f"UMAP_{vector_mode}_SNR{snr_lim_str}_N{n_neighbors}_D{dist_str}"
    
    mediciones_usadas_set = set([m[0] for m in mediciones_aceptadas])
    sufijo_folder = ""
    contador = 1
    
    while True:
        folder_name = f"{base_folder_name}{sufijo_folder}"
        folder_path = os.path.join(base_umap_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            break
        sufijo_folder = f"_{contador}"
        contador += 1
        
    out_dir_final = folder_path
        
    try:
        csv_path = os.path.join(out_dir_final, "UMAP_features.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f"Feature_{i}" for i in range(X.shape[1])] + ["Rol", "Label_Vocal"]
            writer.writerow(header)
            for i in range(X.shape[0]):
                row = list(X[i]) + [roles[i], y[i]]
                writer.writerow(row)
        logger(f"[INFO] Matriz exportada a: {csv_path}")
    except Exception as e:
        logger(f"[WARN] No se pudo exportar el CSV: {e}")
        
    logger(f"[INFO] Entrenando UMAP 2D y 3D (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    unique_vocales = sorted(list(set(y)))
    vocal_to_int = {v: i for i, v in enumerate(unique_vocales)}
    
    X_train = X[roles == 'train']
    y_train_str = y[roles == 'train']
    y_train = np.array([vocal_to_int[v] for v in y_train_str]) if is_supervised else None
    
    if len(X_train) < 2:
        logger("❌ Error: No hay suficientes pulsos de 'Entrenamiento' para UMAP.")
        return
        
    reducer_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    reducer_2d.fit(X_train, y=y_train)
    embedding_2d = reducer_2d.transform(X)
    
    reducer_3d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=42)
    reducer_3d.fit(X_train, y=y_train)
    embedding_3d = reducer_3d.transform(X)
    
    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
    except:
        sil_score_2d = float('nan')
    
    logger("[INFO] Proyecciones completadas. Generando reportes gráficos...")
    img_path_2d, img_path_3d, tex_path = plot_umap_results(
        embedding_2d, embedding_3d, y, roles, out_dir_final, folder_name, 
        mediciones_aceptadas, mediciones_rechazadas, 
        n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d
    )
    
    if run_kmeans and 'test' in roles:
        logger("Ejecutando K-Means en el espacio UMAP y construyendo Matriz de Confusión...")
        k = len(unique_vocales)
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        kmeans.fit(embedding_2d[roles == 'train'])
        test_preds = kmeans.predict(embedding_2d[roles == 'test'])
        y_test_true = y[roles == 'test']
        
        train_preds = kmeans.labels_
        cluster_to_vocal = {}
        for cluster_id in range(k):
            vocales_in_cluster = y_train_str[train_preds == cluster_id]
            if len(vocales_in_cluster) > 0:
                cluster_to_vocal[cluster_id] = pd.Series(vocales_in_cluster).mode()[0]
            else:
                cluster_to_vocal[cluster_id] = "Desconocido"
                
        test_preds_vocales = [cluster_to_vocal.get(c, "Desconocido") for c in test_preds]
        
        cm = confusion_matrix(y_test_true, test_preds_vocales, labels=unique_vocales)
        cm_pct = np.zeros_like(cm, dtype=float)
        row_sums = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_pct[i] = (cm[i] / row_sums[i]) * 100
                
        annot_data = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_data[i, j] = f"{cm_pct[i, j]:.1f}%\\n({cm[i, j]})"
                
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=annot_data, fmt='', cmap='Blues', xticklabels=unique_vocales, yticklabels=unique_vocales)
        plt.title('Matriz de Confusión (Test Data) - K-Means sobre UMAP 2D')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción (K-Means)')
        cm_path = os.path.join(out_dir_final, "matriz_confusion_umap.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")
    
    logger(f"[+] UMAP Finalizado exitosamente. Archivos generados:\\n  - {img_path_2d}\\n  - {img_path_3d}\\n  - {tex_path}")
"""

final_content = content[:start_idx] + new_ejecutar_umap

with open('EMG_desarrollo/analysis/umap_motor.py', 'w') as f:
    f.write(final_content)

