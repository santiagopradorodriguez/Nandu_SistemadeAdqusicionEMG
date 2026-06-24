import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import umap

from sklearn.metrics import silhouette_score, confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd

import warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")

def plot_umap_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, mediciones_aceptadas, mediciones_rechazadas, n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d=float('nan'), sil_score_3d=float('nan')):
    """
    Genera el Scatter Plot 2D y 3D de UMAP y el reporte LaTeX correspondiente.
    """
    unique_labels = sorted(list(set(labels)))
    custom_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange']
    
    # --- Plot 2D ---
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    for i, vocal in enumerate(unique_labels):
        c = custom_colors[i % len(custom_colors)]
        
        idx_train = (np.array(labels) == vocal) & (np.array(roles) == 'train')
        if np.any(idx_train):
            ax_2d.scatter(embedding_2d[idx_train, 0], embedding_2d[idx_train, 1], label=f'Vocal {vocal} (Train)', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60, marker='o')
            
        idx_test = (np.array(labels) == vocal) & (np.array(roles) == 'test')
        if np.any(idx_test):
            ax_2d.scatter(embedding_2d[idx_test, 0], embedding_2d[idx_test, 1], label=f'Vocal {vocal} (Test)', color=c, alpha=0.9, edgecolors='k', linewidth=1.5, s=150, marker='*')
        
    ax_2d.set_title(f"Proyección UMAP 2D ({'SUMAP' if is_supervised else 'No Supervisado'})\nVector: {vector_mode} | vecinos: {n_neighbors} | dist\\_min: {min_dist}", fontweight='bold', pad=15)
    ax_2d.set_xlabel("UMAP Dimensión 1")
    ax_2d.set_ylabel("UMAP Dimensión 2")
    ax_2d.legend(title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_2d.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    img_filename_2d = f"UMAP_Scatter_2D_{sufijo}.png"
    out_file_2d = os.path.join(out_dir, img_filename_2d)
    fig_2d.savefig(out_file_2d, dpi=300, bbox_inches='tight')
    plt.close(fig_2d)
    
    # --- Plot 3D ---
    fig_3d = plt.figure(figsize=(10, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    for i, vocal in enumerate(unique_labels):
        c = custom_colors[i % len(custom_colors)]
        
        idx_train = (np.array(labels) == vocal) & (np.array(roles) == 'train')
        if np.any(idx_train):
            ax_3d.scatter(embedding_3d[idx_train, 0], embedding_3d[idx_train, 1], embedding_3d[idx_train, 2], label=f'Vocal {vocal} (Train)', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60, marker='o')
            
        idx_test = (np.array(labels) == vocal) & (np.array(roles) == 'test')
        if np.any(idx_test):
            ax_3d.scatter(embedding_3d[idx_test, 0], embedding_3d[idx_test, 1], embedding_3d[idx_test, 2], label=f'Vocal {vocal} (Test)', color=c, alpha=0.9, edgecolors='k', linewidth=1.5, s=150, marker='*')
        
    ax_3d.set_title(f"Proyección UMAP 3D ({'SUMAP' if is_supervised else 'No Supervisado'})", fontweight='bold', pad=15)
    ax_3d.set_xlabel("UMAP Dim 1")
    ax_3d.set_ylabel("UMAP Dim 2")
    ax_3d.set_zlabel("UMAP Dim 3")
    ax_3d.legend(title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    img_filename_3d = f"UMAP_Scatter_3D_{sufijo}.png"
    out_file_3d = os.path.join(out_dir, img_filename_3d)
    fig_3d.savefig(out_file_3d, dpi=300, bbox_inches='tight')
    plt.close(fig_3d)
    
    # --- Generar Reporte LaTeX ---
    tex_filename = f"umap_report_{sufijo}.tex"
    tex_filepath = os.path.join(out_dir, tex_filename)
    
    vocales_str = ", ".join(unique_labels)
    modo_str = "Supervisado (SUMAP)" if is_supervised else "No Supervisado"
    
    latex_code = "\\documentclass[12pt]{article}\n"
    latex_code += "\\usepackage[utf8]{inputenc}\n"
    latex_code += "\\usepackage{graphicx}\n"
    latex_code += "\\usepackage[a4paper, margin=2cm]{geometry}\n"
    latex_code += "\\begin{document}\n\n"
    
    latex_code += "\\section*{Reporte de Clustering: UMAP}\n"
    latex_code += "\\begin{itemize}\n"
    latex_code += f"    \\item \\textbf{{Vocales Analizadas:}} {vocales_str}\n"
    latex_code += f"    \\item \\textbf{{Modo de Entrenamiento:}} {modo_str}\n"
    latex_code += f"    \\item \\textbf{{Vectorización:}} {vector_mode}\n"
    latex_code += f"    \\item \\textbf{{Hiperparámetros:}} n\\_neighbors = {n_neighbors}, min\\_dist = {min_dist}\n"
    latex_code += f"    \\item \\textbf{{Total de Pulsos (N):}} {len(labels)}\n"
    sil_str = f"{sil_score_2d:.4f}" if not np.isnan(sil_score_2d) else "N/A"
    sil_str_3d = f"{sil_score_3d:.4f}" if not np.isnan(sil_score_3d) else "N/A"
    latex_code += f"    \\item \\textbf{{Silhouette Score (2D):}} {sil_str}\n"
    latex_code += f"    \\item \\textbf{{Silhouette Score (3D):}} {sil_str_3d}\n"
    latex_code += "\\end{itemize}\n\n"
    
    latex_code += "\\begin{figure}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += f"\\includegraphics[width=\\textwidth]{{{img_filename_2d}}}\n"
    latex_code += f"\\caption{{Proyección 2D del espacio de características EMG usando UMAP.}}\n"
    latex_code += "\\end{figure}\n\n"
    
    latex_code += "\\newpage\n"
    latex_code += "\\begin{figure}[h!]\n"
    latex_code += "\\centering\n"
    latex_code += f"\\includegraphics[width=\\textwidth]{{{img_filename_3d}}}\n"
    latex_code += f"\\caption{{Perspectiva 3D del espacio topológico UMAP.}}\n"
    latex_code += "\\end{figure}\n\n"
    
    latex_code += "\\newpage\n"
    latex_code += "\\section*{Mediciones Utilizadas y Filtradas}\n"
    
    latex_code += "\\subsection*{Mediciones Aprobadas}\n"
    latex_code += "\\begin{itemize}\n"
    for path, v, count, total in mediciones_aceptadas:
        parts = path.split(os.sep)
        try:
            idx = parts.index("base_de_datos_electrodos")
            rel_path = os.sep.join(parts[idx:])
        except ValueError:
            rel_path = os.path.basename(path)
        rel_path_tex = rel_path.replace('_', '\\_')
        latex_code += f"\\item {rel_path_tex} (Vocal: {v}) - \\textbf{{({count}/{total}) pulsos válidos}}\n"
    latex_code += "\\end{itemize}\n"
    
    if mediciones_rechazadas:
        latex_code += "\\subsection*{Mediciones Rechazadas por Ruido (SNR)}\n"
        latex_code += "\\begin{itemize}\n"
        for path, v, causa, snr_val in mediciones_rechazadas:
            parts = path.split(os.sep)
            try:
                idx = parts.index("base_de_datos_electrodos")
                rel_path = os.sep.join(parts[idx:])
            except ValueError:
                rel_path = os.path.basename(path)
            rel_path_tex = rel_path.replace('_', '\\_')
            snr_str = f"{snr_val:.2f}" if snr_val != float('inf') else "N/A"
            latex_code += f"\\item {rel_path_tex} (Vocal: {v}) - Rechazada por SNR {causa}: \\textbf{{{snr_str}}}\n"
        latex_code += "\\end{itemize}\n"
        
    latex_code += "\\end{document}\n"
    
    with open(tex_filepath, 'w') as f:
        f.write(latex_code)
        
    return out_file_2d, out_file_3d, tex_filepath

def ejecutar_umap(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, vector_mode, n_neighbors, min_dist, is_supervised, run_kmeans, import_pca, pca_csv_path, logger=print):
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import numpy as np
    import os
    import csv
    import json
    import umap
    
    logger("\n" + "="*50)
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
                
    logger(f"\n[INFO] Matriz de características X lista: {X.shape[0]} pulsos, {X.shape[1]} dimensiones.")
    
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
        sil_score_3d = silhouette_score(embedding_3d, y)
        logger(f"[INFO] Silhouette Score (3D): {sil_score_3d:.4f}")
    except:
        sil_score_2d = float('nan')
        sil_score_3d = float('nan')
    
    logger("[INFO] Proyecciones completadas. Generando reportes gráficos...")
    img_path_2d, img_path_3d, tex_path = plot_umap_results(
        embedding_2d, embedding_3d, y, roles, out_dir_final, folder_name, 
        mediciones_aceptadas, mediciones_rechazadas, 
        n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d, sil_score_3d
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
                annot_data[i, j] = f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})"
                
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
                annot_data_3d[i, j] = f"{cm_pct_3d[i, j]:.1f}%\n({cm_3d[i, j]})"
                
        plt.figure(figsize=(8,6))
        sns.heatmap(cm_3d, annot=annot_data_3d, fmt='', cmap='Blues', xticklabels=unique_vocales, yticklabels=unique_vocales)
        plt.title('Matriz de Confusión (Test Data) - K-Means sobre UMAP 3D')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción (K-Means)')
        cm_path_3d = os.path.join(out_dir_final, "matriz_confusion_umap_3d.png")
        plt.savefig(cm_path_3d, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión 3D generada: {cm_path_3d}")
    
    logger(f"[+] UMAP Finalizado exitosamente. Archivos generados:\n  - {img_path_2d}\n  - {img_path_3d}\n  - {tex_path}")
