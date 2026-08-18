# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor de procesamiento y cálculo PCA para análisis comparativo de sesiones EMG.
# ==============================================================================

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.signal import resample, find_peaks, correlate, correlation_lags
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances, confusion_matrix
from sklearn.cluster import KMeans
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import sys
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir_abs))

def get_interpulse_noise(processed_segment, initial_noise):
    if len(processed_segment) < 10:
        return initial_noise
        
    abs_noise = np.abs(processed_segment)
    q1 = np.percentile(abs_noise, 25)
    q3 = np.percentile(abs_noise, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    valid_noise = abs_noise[abs_noise <= upper_bound]
    if len(valid_noise) < 3:
        valid_noise = abs_noise
        
    curr_mean = np.mean(valid_noise)
    if initial_noise > 0 and (curr_mean / initial_noise) > 5.0:
        return initial_noise
        
    return curr_mean

def build_pca_features(asignaciones_vocales, canales_seleccionados, mapped_names, logger, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo):
    X = []
    Y = []
    Tomas = []
    
    mediciones_aceptadas = defaultdict(list)
    mediciones_rechazadas = defaultdict(list)
    
    totales_brutos = 0
    filtrados_snr = 0
    resultantes = 0
    
    TARGET_LEN = 100 # Resolucion estandar por canal
    
    Roles = []
    
    for path, vocal_data in asignaciones_vocales.items():
        if isinstance(vocal_data, dict):
            vocal = vocal_data.get('vocal', '').upper()
            rol = vocal_data.get('rol', 'train').lower()
        else:
            vocal = vocal_data.upper()
            rol = 'train'
            
        if vocal == 'IGNORAR': continue
        
        # Cargar todos los canales (incluido mic si existe)
        all_chans_in_dir = [d for d in os.listdir(path) if d.startswith('canal_')]
        canales_data = {}
        for ch in all_chans_in_dir:
            json_files = [f for f in os.listdir(os.path.join(path, ch)) if f.startswith('analisis_results') and f.endswith('.json')]
            if not json_files:
                res_path = os.path.join(path, ch, 'results.json')
                if os.path.exists(res_path):
                    try:
                        with open(res_path, 'r') as f: res_data = json.load(f)
                        c_path = os.path.join(path, ch, "metadata.json")
                        meta = json.load(open(c_path)) if os.path.exists(c_path) else {}
                        bpm = meta.get('bpm', 40)
                        canales_data[ch] = {
                            'picos_segundos': res_data.get('picos_ventana', []),
                            'meta': meta,
                            'muestras_pulso': int((60.0 / bpm) * 1000),
                            'noise': meta.get('noise_seconds', 2.0)
                        }
                    except Exception as e:
                        logger(f"    [!] Error al cargar results.json para {ch}: {e}")
                continue
                
            json_path = os.path.join(path, ch, json_files[0]) # Toma el primero que encuentre (ej: analisis_results.json)
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    
                env = np.array(data['env_recortada'])
                
                c_path = os.path.join(path, ch, "metadata.json")
                if os.path.exists(c_path):
                    with open(c_path, 'r') as f:
                        meta = json.load(f)
                else:
                    meta = {}
                    
                canales_data[ch] = {
                    'env': env,
                    'meta': meta,
                    'muestras_pulso': data.get('muestras_pulso', meta.get('samples_per_pulse', 1500)),
                    'noise': meta.get('noise_seconds', 2.0)
                }
            except Exception as e:
                logger(f"    [!] Error al cargar datos para {ch}: {e}")
                continue
                
        # Canal microfono (asumimos canal_3 o el de mayor indice si no hay)
        mic_ch = "canal_3"
        if mic_ch not in canales_data:
            logger(f"  [!] Advertencia: No se encontró micrófono ({mic_ch}) en {os.path.basename(path)}. Saltando...")
            continue
            
        if 'env' in canales_data[mic_ch]:
            env_mic = canales_data[mic_ch]['env']
            muestras_pulso = canales_data[mic_ch]['muestras_pulso']
            
            dist_samples = int(0.8 * muestras_pulso)
            min_height = np.max(env_mic) * 0.2
            picos_mic, _ = find_peaks(env_mic, distance=dist_samples, height=min_height)
        elif 'picos_segundos' in canales_data[mic_ch]:
            picos_mic = [int(p * 1000) for p in canales_data[mic_ch]['picos_segundos']]
            muestras_pulso = canales_data[mic_ch]['muestras_pulso']
        else:
            logger(f"  [!] Advertencia: Canal micrófono sin datos útiles en {os.path.basename(path)}. Saltando...")
            continue
        
        pre_samples = int(muestras_pulso * 0.4)
        post_samples = int(muestras_pulso * 0.6)
        
        pulsos_validos = 0
        totales_brutos += len(picos_mic)
        
        for win_idx, pico in enumerate(picos_mic):
            real_cut_start = pico - pre_samples
            real_cut_end = pico + post_samples
            
            if real_cut_start < 0 or real_cut_end > len(env_mic):
                continue
                
            valido = True
            segs_brutos = []
            max_supremo = 1e-9
            ruido_acumulado_window = 0.0
            
            for ch in canales_seleccionados:
                if ch not in canales_data:
                    valido = False; break
                
                env_ch_raw = canales_data[ch]['env']
                if real_cut_end > len(env_ch_raw):
                    valido = False; break
                    
                segmento_ch = env_ch_raw[real_cut_start:real_cut_end].copy()
                
                initial_noise = canales_data[ch]['noise']
                noise_win_samples = max(3, int(muestras_pulso / 4.0))
                
                # Ruido PRE
                noise_start_pre = max(0, int(pico - 0.5 * muestras_pulso - noise_win_samples))
                noise_end_pre = min(len(env_ch_raw), noise_start_pre + noise_win_samples)
                ruido_pre = initial_noise
                if noise_end_pre > noise_start_pre:
                    ruido_pre = get_interpulse_noise(env_ch_raw[noise_start_pre:noise_end_pre], initial_noise)
                    
                # Ruido POST
                noise_start_post = min(len(env_ch_raw), int(pico + 0.5 * muestras_pulso))
                noise_end_post = min(len(env_ch_raw), noise_start_post + noise_win_samples)
                ruido_post = ruido_pre
                if noise_end_post > noise_start_post:
                    ruido_post = get_interpulse_noise(env_ch_raw[noise_start_post:noise_end_post], initial_noise)
                    
                ruido_promedio = (ruido_pre + ruido_post) / 2.0
                ruido_acumulado_window += ruido_promedio
                
                # Resta agresiva (alpha=1.0)
                segmento_ch = np.maximum(segmento_ch - ruido_promedio, 0)
                
                m_val = np.max(segmento_ch)
                if m_val > max_supremo:
                    max_supremo = m_val
                    
                segs_brutos.append(segmento_ch)
                
            if not valido:
                continue
                
            ruido_promedio_total = ruido_acumulado_window / len(canales_seleccionados)
            snr_win = max_supremo / (ruido_promedio_total + 1e-9)
            
            # Filtro SNR
            if filtro_snr_activo:
                if filtro_snr_tipo in ["Por Ventana (Individual)", "Ambos (Global + Ventana)"]:
                    if snr_win < filtro_snr_limite:
                        mediciones_rechazadas[vocal].append(f"{os.path.basename(path)}_Win{win_idx} (SNR={snr_win:.1f})")
                        filtrados_snr += 1
                        continue
                        
            # Construir vector final concatenando resamples
            vector_concatenado = []
            for seg in segs_brutos:
                seg_norm = seg / max_supremo
                seg_rs = resample(seg_norm, TARGET_LEN)
                seg_rs[seg_rs < 0] = 0.0
                vector_concatenado.append(seg_rs)
                
            flat_vector = np.concatenate(vector_concatenado)
            X.append(flat_vector)
            Y.append(vocal)
            Roles.append(rol)
            Tomas.append(f"{os.path.basename(path)}_W{win_idx}")
            pulsos_validos += 1
            resultantes += 1
            mediciones_aceptadas[vocal].append(f"{os.path.basename(path)}_Win{win_idx} (SNR={snr_win:.1f})")
            
        if pulsos_validos > 0:
            logger(f"  [+] Añadida {os.path.basename(path)} ({pulsos_validos}/{len(picos_mic)} pulsos).")
        else:
            logger(f"  [-] Omitida (Todos los pulsos filtrados): {os.path.basename(path)}")
    info_pulsos = {
        'totales_brutos': totales_brutos,
        'filtrados_snr': filtrados_snr,
        'resultantes': resultantes
    }
            
    return np.array(X), np.array(Y), np.array(Roles), Tomas, mediciones_aceptadas, mediciones_rechazadas, info_pulsos

def plot_pca_results(embedding_2d, embedding_3d, labels, roles, out_dir, sufijo, variance_ratio, n_components, sil_score_2d=float('nan'), sil_score_3d=float('nan')):
    unique_labels = sorted(list(set(labels)))
    custom_colors = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:orange']
    
    # 2D Plot
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    for i, vocal in enumerate(unique_labels):
        c = custom_colors[i % len(custom_colors)]
        
        # Train points
        idx_train = (np.array(labels) == vocal) & (np.array(roles) == 'train')
        if np.any(idx_train):
            ax_2d.scatter(embedding_2d[idx_train, 0], embedding_2d[idx_train, 1], label=f'Vocal {vocal} (Train)', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60, marker='o')
            
        # Test points
        idx_test = (np.array(labels) == vocal) & (np.array(roles) == 'test')
        if np.any(idx_test):
            ax_2d.scatter(embedding_2d[idx_test, 0], embedding_2d[idx_test, 1], label=f'Vocal {vocal} (Test)', color=c, alpha=0.9, edgecolors='k', linewidth=1.5, s=150, marker='*')
        
    ax_2d.set_title(f"PCA 2D (Retención Varianza: {np.sum(variance_ratio[:2])*100:.1f}%)", fontweight='bold', pad=15)
    ax_2d.set_xlabel(f"PC1 ({variance_ratio[0]*100:.1f}%)")
    ax_2d.set_ylabel(f"PC2 ({variance_ratio[1]*100:.1f}%)")
    ax_2d.legend(title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_2d.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig_2d.savefig(os.path.join(out_dir, f"PCA_Scatter_2D_{sufijo}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_2d)
    
    # 3D Plot
    if embedding_3d is not None and len(variance_ratio) >= 3:
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        for i, vocal in enumerate(unique_labels):
            c = custom_colors[i % len(custom_colors)]
            
            # Train points
            idx_train = (np.array(labels) == vocal) & (np.array(roles) == 'train')
            if np.any(idx_train):
                ax_3d.scatter(embedding_3d[idx_train, 0], embedding_3d[idx_train, 1], embedding_3d[idx_train, 2], label=f'Vocal {vocal} (Train)', color=c, alpha=0.8, edgecolors='k', linewidth=0.5, s=60, marker='o')
                
            # Test points
            idx_test = (np.array(labels) == vocal) & (np.array(roles) == 'test')
            if np.any(idx_test):
                ax_3d.scatter(embedding_3d[idx_test, 0], embedding_3d[idx_test, 1], embedding_3d[idx_test, 2], label=f'Vocal {vocal} (Test)', color=c, alpha=0.9, edgecolors='k', linewidth=1.5, s=150, marker='*')
            
        x_min, x_max = ax_3d.get_xlim()
        y_min, y_max = ax_3d.get_ylim()
        z_min, z_max = ax_3d.get_zlim()
        for i, vocal in enumerate(unique_labels):
            c = custom_colors[i % len(custom_colors)]
            idx_vocal = (np.array(labels) == vocal)
            pts = embedding_3d[idx_vocal]
            if len(pts) > 0:
                ax_3d.scatter(pts[:, 0], pts[:, 1], zs=z_min, zdir='z',
                              color=c, s=25, alpha=0.20, edgecolors='none', depthshade=False)
                ax_3d.scatter(pts[:, 0], pts[:, 2], zs=y_max, zdir='y',
                              color=c, s=20, alpha=0.15, edgecolors='none', depthshade=False)
                ax_3d.scatter(pts[:, 1], pts[:, 2], zs=x_min, zdir='x',
                              color=c, s=20, alpha=0.15, edgecolors='none', depthshade=False)
        ax_3d.set_xlim(x_min, x_max)
        ax_3d.set_ylim(y_min, y_max)
        ax_3d.set_zlim(z_min, z_max)

        ax_3d.set_title(f"PCA 3D (Retención Varianza: {np.sum(variance_ratio[:3])*100:.1f}%)", fontweight='bold', pad=15)
        ax_3d.set_xlabel(f"PC1 ({variance_ratio[0]*100:.1f}%)")
        ax_3d.set_ylabel(f"PC2 ({variance_ratio[1]*100:.1f}%)")
        ax_3d.set_zlabel(f"PC3 ({variance_ratio[2]*100:.1f}%)")
        ax_3d.legend(title="Clases", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig_3d.savefig(os.path.join(out_dir, f"PCA_Scatter_3D_{sufijo}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_3d)

def ejecutar_pca(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, is_supervised, use_umap, n_components, run_kmeans, logger):
    logger("Iniciando construcción del DataFrame PCA y reducción de dimensiones...")
    
    base_dir = os.path.dirname(list(asignaciones_vocales.keys())[0]) # 2026-06-10 directory
    date_folder = os.path.basename(os.path.dirname(list(asignaciones_vocales.keys())[0]))
    out_dir_base = os.path.join(os.path.dirname(base_dir), date_folder, "PCA")
    
    num_mediciones = len(asignaciones_vocales)
    base_sufijo = f"M{num_mediciones}_Comp{n_components}_SNR{str(filtro_snr_limite).replace('.','-')}"
    if is_supervised: base_sufijo += "_Sup"
    if use_umap: base_sufijo += "_ForUMAP"
    
    sufijo_idx = ""
    contador = 1
    mediciones_usadas_set = set(asignaciones_vocales.keys())
    
    while True:
        sufijo = f"{base_sufijo}{sufijo_idx}"
        out_dir = os.path.join(out_dir_base, f"PCA_{sufijo}")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            break
        else:
            meta_path = os.path.join(out_dir, "mediciones.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    if set(meta) == mediciones_usadas_set:
                        break
                except:
                    pass
            sufijo_idx = f"_{contador}"
            contador += 1
            
    with open(os.path.join(out_dir, "mediciones.json"), "w") as f:
        json.dump(list(mediciones_usadas_set), f)
    
    # Check if mediciones.json exists and we can skip building?
    # Para ser robustos, siempre extraemos y validamos por si cambió algún canal seleccionado
    logger("Construyendo matriz estricta 100 muestras/canal (Alineación por Micrófono)")
    X, Y, Roles, Tomas, med_acc, med_rej, info_pulsos = build_pca_features(
        asignaciones_vocales, canales_seleccionados, mapped_names, logger,
        filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo
    )
    
    if len(X) < 2:
        logger("[ERROR] No hay suficientes pulsos válidos para hacer PCA.")
        return
        
    logger(f"Vector Maestro generado: {X.shape[0]} repeticiones de {X.shape[1]} features.")
    
    # PCA Calculation
    X_train = X[Roles == 'train']
    if len(X_train) < 2:
        logger("[ERROR] No hay suficientes pulsos de 'Entrenamiento' para hacer PCA.")
        return
        
    pca = PCA(n_components=min(n_components, X_train.shape[0], X_train.shape[1]))
    pca.fit(X_train)
    X_pca = pca.transform(X)
    variance_ratios = pca.explained_variance_ratio_
    
    X_pca_2d = X_pca[:, :2]
    X_pca_3d = X_pca[:, :3] if X_pca.shape[1] >= 3 else None
    
    # Exportar DataFrame para Vector Auditor (Vector Completo)
    cols = []
    for ch in canales_seleccionados:
        cols.extend([f"{ch}_T{i}" for i in range(100)])
    
    df_full = pd.DataFrame(X, columns=cols)
    df_full.insert(0, 'Toma', Tomas)
    df_full.insert(1, 'Rol', Roles)
    df_full.insert(2, 'Vocal', Y)
    csv_path = os.path.join(out_dir, "vector_maestro_300d.csv")
    df_full.to_csv(csv_path, index=False)
    logger(f"CSV de alta dimensionalidad exportado a: {csv_path}")
    
    if use_umap:
        # Exportar CSV de PCA reducida para consumo de UMAP
        df_reduced = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
        df_reduced.insert(0, 'Toma', Tomas)
        df_reduced.insert(1, 'Rol', Roles)
        df_reduced.insert(2, 'Vocal', Y)
        reduced_csv = os.path.join(out_dir, f"pca_reduced_{n_components}comp.csv")
        df_reduced.to_csv(reduced_csv, index=False)
        logger(f"CSV Reducido (PCA->UMAP) exportado a: {reduced_csv}")
        
    if run_kmeans and 'test' in Roles:
        logger("Ejecutando K-Means en el espacio PCA y construyendo Matriz de Confusión...")
        unique_vocales = sorted(list(set(Y)))
        k = len(unique_vocales)
        kmeans = KMeans(n_clusters=k, random_state=42)
        
        # Fit kmeans on Train
        kmeans.fit(X_pca_2d[Roles == 'train'])
        
        # Predict on Test
        test_preds = kmeans.predict(X_pca_2d[Roles == 'test'])
        y_test_true = Y[Roles == 'test']
        y_train_true = Y[Roles == 'train']
        train_preds = kmeans.labels_
        
        # Bautizar clusters usando Train
        cluster_to_vocal = {}
        for cluster_id in range(k):
            vocales_in_cluster = y_train_true[train_preds == cluster_id]
            if len(vocales_in_cluster) > 0:
                vocal_mayoritaria = pd.Series(vocales_in_cluster).mode()[0]
                cluster_to_vocal[cluster_id] = vocal_mayoritaria
            else:
                cluster_to_vocal[cluster_id] = "Desconocido"
                
        # Traducir predicciones de test
        test_preds_vocales = [cluster_to_vocal.get(c, "Desconocido") for c in test_preds]
        
        # Generar Matriz
        cm = confusion_matrix(y_test_true, test_preds_vocales, labels=unique_vocales)
        
        # Calculate percentages
        cm_pct = np.zeros_like(cm, dtype=float)
        row_sums = cm.sum(axis=1)
        for i in range(cm.shape[0]):
            if row_sums[i] > 0:
                cm_pct[i] = (cm[i] / row_sums[i]) * 100
                
        annot_data = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_data[i, j] = f"{cm_pct[i, j]:.1f}%\n({cm[i, j]})"
                
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=annot_data, fmt='', cmap='Blues', xticklabels=unique_vocales, yticklabels=unique_vocales)
        plt.title('Matriz de Confusión (Test Data) - K-Means sobre PCA 2D')
        plt.ylabel('Verdadero')
        plt.xlabel('Predicción (K-Means)')
        cm_path = os.path.join(out_dir, "matriz_confusion_pca.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger(f"Matriz de Confusión generada: {cm_path}")
    
    # Calcular metricas si hay múltiples vocales
    sil_score_2d = float('nan')
    sil_score_full = float('nan')
    if len(set(Y)) > 1:
        sil_score_2d = silhouette_score(X_pca_2d, Y, metric='euclidean')
        sil_score_full = silhouette_score(X, Y, metric='euclidean')
        
    logger(f"Generando gráficos PCA (Varianza retenida 2D: {np.sum(variance_ratios[:2])*100:.1f}%)")
    plot_pca_results(X_pca_2d, X_pca_3d, Y, Roles, out_dir, sufijo, variance_ratios, n_components)
    
    # Metricas .txt
    with open(os.path.join(out_dir, "metricas.txt"), "w") as f:
        f.write(f"Silhouette Score (Full {X.shape[1]}D): {sil_score_full:.4f}\n")
        f.write(f"Silhouette Score (PCA 2D): {sil_score_2d:.4f}\n")
        f.write(f"Varianza Explicada Acumulada ({n_components} comp): {np.sum(variance_ratios)*100:.2f}%\n")
        
    # Reporte Extraccion
    with open(os.path.join(out_dir, "reporte_extraccion.txt"), "w") as f:
        f.write("=== REPORTE DE EXTRACCIÓN PCA ===\n")
        f.write(f"Pulsos Totales Detectados (Brutos): {info_pulsos['totales_brutos']}\n")
        f.write(f"Pulsos Filtrados por SNR: {info_pulsos['filtrados_snr']}\n")
        f.write(f"Pulsos Resultantes (Válidos): {info_pulsos['resultantes']}\n")
        f.write("\n=== MEDICIONES ACEPTADAS ===\n")
        for v, records in med_acc.items():
            f.write(f"Vocal {v}:\n")
            for r in records: f.write(f"  - {r}\n")
        f.write("\n=== MEDICIONES RECHAZADAS ===\n")
        for v, records in med_rej.items():
            f.write(f"Vocal {v}:\n")
            for r in records: f.write(f"  - {r}\n")
            
    logger(f"Reporte guardado en: {os.path.join(out_dir, 'reporte_extraccion.txt')}")
    logger("Pipeline PCA completado exitosamente.")
