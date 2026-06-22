import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import umap

import warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden to 1 by setting random_state")

def plot_umap_results(embedding_2d, embedding_3d, labels, out_dir, sufijo, mediciones_aceptadas, mediciones_rechazadas, n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d):
    """
    Genera el Scatter Plot 2D y 3D de UMAP y el reporte LaTeX correspondiente.
    """
    unique_labels = sorted(list(set(labels)))
    custom_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange']
    
    # --- Plot 2D ---
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    for i, vocal in enumerate(unique_labels):
        idx = np.array(labels) == vocal
        c = custom_colors[i % len(custom_colors)]
        ax_2d.scatter(embedding_2d[idx, 0], embedding_2d[idx, 1], label=f'Vocal {vocal}', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60)
        
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
        idx = np.array(labels) == vocal
        c = custom_colors[i % len(custom_colors)]
        ax_3d.scatter(embedding_3d[idx, 0], embedding_3d[idx, 1], embedding_3d[idx, 2], label=f'Vocal {vocal}', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60)
        
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
    latex_code += f"    \\item \\textbf{{Silhouette Score (2D):}} {sil_str}\n"
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

def ejecutar_umap(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, vector_mode, n_neighbors, min_dist, is_supervised, logger=print):
    logger("\n" + "="*50)
    logger(f" INICIANDO ANÁLISIS UMAP ({'SUPERVISADO' if is_supervised else 'NO SUPERVISADO'})")
    logger("="*50)
    
    X_data = []
    Y_labels = []
    ventana_referencia = None
    
    mediciones_aceptadas = []
    mediciones_rechazadas = []
    
    out_dir_general = list(asignaciones_vocales.keys())[0] if asignaciones_vocales else ""
    if out_dir_general:
        session_dir = os.path.dirname(out_dir_general)
        base_umap_dir = os.path.join(session_dir, "UMAP")
        os.makedirs(base_umap_dir, exist_ok=True)
        
    for path, vocal in asignaciones_vocales.items():
        canales_medicion = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('canal_')]
        canales = [ch for ch in canales_seleccionados if ch in canales_medicion]
        
        if not canales: continue
        
        json_path_0 = os.path.join(path, canales[0], "analisis_results.json")
        if not os.path.exists(json_path_0): continue
            
        with open(json_path_0, 'r') as f: data = json.load(f)
            
        env_ms = data.get('smooth_ms', None)
        if env_ms is None:
            for k, v in data.items():
                if isinstance(v, dict) and 'smooth_ms' in v:
                    env_ms = v['smooth_ms']
                    break
                    
        if ventana_referencia is None:
            ventana_referencia = env_ms
            logger(f"  [i] Ventana de referencia: {ventana_referencia} ms")
        elif env_ms != ventana_referencia:
            raise ValueError(f"Inconsistencia de ventanas en {os.path.basename(path)}")
            
        recortes_medicion = {}
        min_snr_global = float('inf')
        
        for ch in canales:
            ventana_str = str(ventana_referencia).replace('.', '_')
            jpath = os.path.join(path, ch, f"analisis_results_env{ventana_str}ms.json")
            if not os.path.exists(jpath): jpath = os.path.join(path, ch, f"analisis_results_env{ventana_referencia}ms.json")
            if not os.path.exists(jpath): jpath = os.path.join(path, ch, "analisis_results.json")
                
            if os.path.exists(jpath):
                with open(jpath, 'r') as f: d = json.load(f)
                
                ch_snr_mean = d.get('snr_mean', 0)
                if ch_snr_mean < min_snr_global: min_snr_global = ch_snr_mean
                
                segs = d.get('segmentos_rs', None)
                if segs is None:
                    for k, v in d.items():
                        if isinstance(v, dict) and 'segmentos_rs' in v:
                            segs = v['segmentos_rs']
                            break
                if segs: recortes_medicion[ch] = segs
                    
        aplica_global = "Global" in filtro_snr_tipo or "Ambos" in filtro_snr_tipo
        if filtro_snr_activo and aplica_global and min_snr_global < filtro_snr_limite:
            logger(f"  [-] Omitida (SNR global {min_snr_global:.2f} < {filtro_snr_limite}): {os.path.basename(path)}")
            mediciones_rechazadas.append((path, vocal, "Global", min_snr_global))
            continue
                
        if not recortes_medicion: continue
            
        num_pulsos = min([len(s) for s in recortes_medicion.values()])
        canales_validos = sorted(list(recortes_medicion.keys()))
        pulsos_validos_count = 0
        
        for i in range(num_pulsos):
            recorte_limpio = {}
            max_global = 0.0
            min_snr_pulso = float('inf')
            
            for ch in canales_validos:
                arr = np.array(recortes_medicion[ch][i])
                
                edge_len = max(1, int(len(arr) * 0.15))
                orillas = np.concatenate([arr[:edge_len], arr[-edge_len:]])
                
                ruido_basal = np.mean(orillas)
                ruido_std = np.std(orillas) if np.std(orillas) > 0 else 1e-6
                
                arr_sin_offset = arr - ruido_basal
                recorte_limpio[ch] = arr_sin_offset
                
                m = np.max(np.abs(arr_sin_offset))
                if m > max_global: max_global = m
                
                snr_ch = m / ruido_std
                if snr_ch < min_snr_pulso: min_snr_pulso = snr_ch
                
            aplica_ventana = "Ventana" in filtro_snr_tipo or "Ambos" in filtro_snr_tipo
            if filtro_snr_activo and aplica_ventana and min_snr_pulso < filtro_snr_limite:
                continue
                
            if max_global == 0: max_global = 1.0
            
            # Feature Vector Extraction
            if vector_mode == "Picos":
                vector_i = []
                for ch in canales_validos:
                    arr_norm = recorte_limpio[ch] / max_global
                    vector_i.append(np.max(arr_norm))
                X_data.append(vector_i)
                Y_labels.append(vocal)
                pulsos_validos_count += 1
                
            elif vector_mode == "Completa":
                vector_i = []
                for ch in canales_validos:
                    arr_norm = recorte_limpio[ch] / max_global
                    vector_i.extend(arr_norm.tolist())
                X_data.append(vector_i)
                Y_labels.append(vocal)
                pulsos_validos_count += 1
                
        if pulsos_validos_count > 0:
            logger(f"  [+] Añadida {os.path.basename(path)} ({pulsos_validos_count}/{num_pulsos} pulsos).")
            mediciones_aceptadas.append((path, vocal, pulsos_validos_count, num_pulsos))
        else:
            logger(f"  [-] Omitida (Todos los pulsos filtrados por SNR Ventana): {os.path.basename(path)}")
            mediciones_rechazadas.append((path, vocal, "Ventana", min_snr_pulso))

    if len(X_data) < 5:
        raise ValueError("No hay suficientes datos válidos para ejecutar UMAP (menos de 5 pulsos). Ajuste el filtro SNR.")
        
    X = np.array(X_data)
    y = np.array(Y_labels)
    logger(f"\n[INFO] Matriz de características X construida: {X.shape[0]} pulsos, {X.shape[1]} dimensiones.")
    
    # Preparar sufijo único y subcarpeta inteligente
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
        else:
            meta_path = os.path.join(folder_path, "mediciones.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if set(meta) == mediciones_usadas_set:
                        break
                except:
                    pass
            sufijo_folder = f"_{contador}"
            contador += 1
            
    with open(os.path.join(folder_path, "mediciones.json"), "w") as f:
        json.dump(list(mediciones_usadas_set), f)
        
    sufijo_unico = ""
    out_dir_final = folder_path
        
    # Exportar Data a CSV usando librería estándar
    try:
        import csv
        csv_path = os.path.join(out_dir_final, "UMAP_features.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = [f"Feature_{i}" for i in range(X.shape[1])] + ["Label_Vocal"]
            writer.writerow(header)
            for i in range(X.shape[0]):
                row = list(X[i]) + [y[i]]
                writer.writerow(row)
        logger(f"[INFO] Matriz de características exportada a: {csv_path}")
    except Exception as e:
        logger(f"[WARN] No se pudo exportar el CSV: {e}")
        
    # Ejecutar UMAP
    logger(f"[INFO] Entrenando UMAP 2D y 3D (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    # Mapear vocales a enteros para y_train si es supervisado
    unique_vocales = sorted(list(set(y)))
    vocal_to_int = {v: i for i, v in enumerate(unique_vocales)}
    y_train = np.array([vocal_to_int[v] for v in y]) if is_supervised else None
    
    # Modelo 2D
    reducer_2d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, random_state=42)
    embedding_2d = reducer_2d.fit_transform(X, y=y_train)
    
    # Modelo 3D
    reducer_3d = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3, random_state=42)
    embedding_3d = reducer_3d.fit_transform(X, y=y_train)
    
    # Calcular Silhouette Score
    try:
        from sklearn.metrics import silhouette_score
        sil_score_2d = silhouette_score(embedding_2d, y)
        logger(f"[INFO] Silhouette Score (2D): {sil_score_2d:.4f}")
    except Exception as e:
        sil_score_2d = float('nan')
        logger(f"[WARN] No se pudo calcular el Silhouette Score: {e}")
    
    logger("[INFO] Proyecciones completadas. Generando reportes gráficos...")
    img_path_2d, img_path_3d, tex_path = plot_umap_results(
        embedding_2d, embedding_3d, y, out_dir_final, sufijo_unico, 
        mediciones_aceptadas, mediciones_rechazadas, 
        n_neighbors, min_dist, vector_mode, is_supervised, sil_score_2d
    )
    
    logger(f"[+] UMAP Finalizado exitosamente. Archivos generados:\n  - {img_path_2d}\n  - {img_path_3d}\n  - {tex_path}")
