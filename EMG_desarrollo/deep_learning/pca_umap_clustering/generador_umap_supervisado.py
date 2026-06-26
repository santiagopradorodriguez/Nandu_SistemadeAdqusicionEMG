import os
import sys
import io
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.signal import resample, find_peaks
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import umap

import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root

# Importamos las utilidades de analisis_trevisan
import analisis_trevisan as at


def procesar_mediciones(base_dir):
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    mediciones_list = []
    
    if not os.path.isdir(base_dir):
        print(f"Error: No se encontró el directorio {base_dir}")
        return []
        
    for date_folder in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date_folder)
        if os.path.isdir(date_path) and date_pattern.match(date_folder):
            if date_folder != "2026-07-10":
                continue
            for med_folder in sorted(os.listdir(date_path)):
                med_path = os.path.join(date_path, med_folder)
                if os.path.isdir(med_path):
                    if any(f.startswith("canal_") for f in os.listdir(med_path) if os.path.isdir(os.path.join(med_path, f))):
                        mediciones_list.append(os.path.join(date_folder, med_folder))
                        
    return mediciones_list

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

def extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=1.0, smooth_ms=250, notch_q=30.0, target_len=100, return_raw_cache=False):
    """
    Extrae y alinea las ventanas de los canales 0, 1 y 2.
    Devuelve X (matriz de features), Y (labels/vocales) y Tomas (nombres de las mediciones).
    """
    X = []
    Y = []
    Tomas = []
    SNRs = []
    
    canales_features = ["canal_0", "canal_1", "canal_2"]
    canales_procesar = ["canal_0", "canal_1", "canal_2", "canal_3"]
    
    total_mediciones = len(mediciones)
    
    for idx, rel_path in enumerate(mediciones):
        med_path = os.path.join(base_dir, rel_path)
        med_name = os.path.basename(rel_path)
        partes = med_name.split('_')
        if len(partes) < 1:
            continue
            
        vocal = partes[0].upper()
        if vocal not in ['A', 'E', 'I', 'O', 'U']:
            continue
            
        progreso = (idx / total_mediciones) * 100
        print(f"\n[{progreso:.1f}%] Procesando: {med_name} (Vocal: {vocal})")
        
        # Leemos archivo manual de exclusiones si existe
        excluded_windows = []
        exclude_path = os.path.join(med_path, 'excluded_windows.json')
        if os.path.exists(exclude_path):
            with open(exclude_path, 'r') as f:
                data_excl = json.load(f)
                excluded_windows = data_excl.get("excluded_windows", [])
                
        canales_data = {}
        
        for ch in canales_procesar:
            carpeta = os.path.join(med_path, ch)
            if not os.path.exists(carpeta):
                print(f"  Advertencia: {ch} no encontrado en {med_name}")
                continue
                
            bpm_u, noise_u, pulsos_u = 50, 2.0, None
            meta_path = os.path.join(carpeta, 'metadata.json')
            try:
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                        bpm_u = meta.get('bpm', bpm_u)
                        noise_u = meta.get('noise_seconds', noise_u)
                        pulsos_u = meta.get('pulse_count', pulsos_u)
            except: pass
            
            # Usar la función original para procesar, silenciando sus prints excesivos
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                res_final = at.procesar_wavs_promedio(
                    carpeta=carpeta, output_root=carpeta,
                    bpm=bpm_u, mostrar_recortes=False,
                    noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
                    excluded_windows=excluded_windows,
                    show_interactive_plot=False,
                    notch_q_factor=notch_q,
                    tipo_envolvente="rms", smooth_ms=smooth_ms
                )
            finally:
                sys.stdout = old_stdout
            
            if res_final:
                fname = list(res_final.keys())[0]
                canales_data[ch] = res_final[fname]
                
        if len(canales_data) < 4:
            print(f"  -> Se omite porque no están los 4 canales válidos (0, 1, 2 y 3).")
            continue
            
        # Alinear ventanas usando canal 3 como maestro (find_peaks global)
        muestras_pulso = canales_data["canal_3"]['muestras_pulso']
        env_mic_raw = canales_data["canal_3"]['env_recortada']
        
        # Encontrar picos en el micrófono como en el script interactivo
        # distance: 80% del período esperado
        # height: 20% del máximo del micrófono
        dist_samples = int(0.8 * muestras_pulso)
        min_height = np.max(env_mic_raw) * 0.2
        
        picos_mic, _ = find_peaks(env_mic_raw, distance=dist_samples, height=min_height)
        
        TARGET_LEN = target_len
        
        for win_idx, pico in enumerate(picos_mic):
            # Definir ventana física simétrica basada en el pico del micrófono
            pre_samples = int(muestras_pulso * 0.4)
            post_samples = int(muestras_pulso * 0.6)
            
            real_cut_start = pico - pre_samples
            real_cut_end = pico + post_samples
            
            # Verificar límites
            if real_cut_start < 0 or real_cut_end > len(env_mic_raw):
                continue
                
            valido = True
            segs_brutos = []
            max_supremo = 1e-9
            ruido_acumulado_window = 0.0
            
            # 1. Extraer los 3 canales y buscar el máximo supremo
            for ch in canales_features:
                env_ch_raw = canales_data[ch]['env_recortada']
                
                if real_cut_end > len(env_ch_raw):
                    valido = False
                    break
                    
                segmento_ch = env_ch_raw[real_cut_start:real_cut_end].copy()
                
                # Cálculo de ruido inter-pulso dinámico (Pre y Post)
                initial_noise = canales_data[ch].get('noise_levels', [0])[0] if len(canales_data[ch].get('noise_levels', [])) > 0 else 0
                noise_win_samples = max(3, int(muestras_pulso / 4.0))
                
                # Ruido PRE-pulso
                noise_start_pre = max(0, int(pico - 0.5 * muestras_pulso - noise_win_samples))
                noise_end_pre = min(len(env_ch_raw), noise_start_pre + noise_win_samples)
                ruido_pre = initial_noise
                if noise_end_pre > noise_start_pre:
                    ruido_pre = get_interpulse_noise(env_ch_raw[noise_start_pre:noise_end_pre], initial_noise)
                    
                # Ruido POST-pulso
                noise_start_post = min(len(env_ch_raw), int(pico + 0.5 * muestras_pulso))
                noise_end_post = min(len(env_ch_raw), noise_start_post + noise_win_samples)
                ruido_post = ruido_pre # default
                if noise_end_post > noise_start_post:
                    ruido_post = get_interpulse_noise(env_ch_raw[noise_start_post:noise_end_post], initial_noise)
                    
                # Promedio y atenuación
                ruido_promedio = (ruido_pre + ruido_post) / 2.0
                ruido_acumulado_window += ruido_promedio
                agresividad = alpha_ruido
                ruido_a_restar = ruido_promedio * agresividad
                
                segmento_ch = np.maximum(segmento_ch - ruido_a_restar, 0)
                
                m_val = np.max(segmento_ch)
                if m_val > max_supremo:
                    max_supremo = m_val
                    
                segs_brutos.append(segmento_ch)
                
            if not valido:
                continue
            if return_raw_cache:
                if valido:
                    X.append(segs_brutos)
                    Y.append(vocal)
                    Tomas.append(f"{med_name}_Win{win_idx}")
                    ruido_promedio_total = ruido_acumulado_window / 3.0
                    snr = max_supremo / (ruido_promedio_total + 1e-9)
                    SNRs.append(snr)
                continue
                
            # 2. Normalizar, remuestrear y concatenar
            vector_concatenado = []
            for seg in segs_brutos:
                # Normalizar por pulso (max_supremo) ANTES de FFT
                seg_norm = seg / max_supremo
                
                # Remuestreo por FFT
                seg_rs = resample(seg_norm, TARGET_LEN)
                seg_rs[seg_rs < 0] = 0.0
                
                vector_concatenado.append(seg_rs)
                
            if valido:
                vector_concatenado = np.concatenate(vector_concatenado)
                X.append(vector_concatenado)
                Y.append(vocal)
                Tomas.append(f"{med_name}_Win{win_idx}")
                ruido_promedio_total = ruido_acumulado_window / 3.0
                snr = max_supremo / (ruido_promedio_total + 1e-9)
                SNRs.append(snr)
                
    if return_raw_cache:
        return X, Y, Tomas, SNRs
    return np.array(X), np.array(Y), np.array(Tomas), np.array(SNRs)

def evaluate_classifier(X_proj, Y, name):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X_proj, Y):
        clf = SVC(kernel='rbf')
        clf.fit(X_proj[train_idx], Y[train_idx])
        scores.append(accuracy_score(Y[test_idx], clf.predict(X_proj[test_idx])))
    print(f"Accuracy {name} (5-fold): {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

def evaluar_clustering_no_supervisado(X, Y, nombre):
    # Obtener etiquetas reales y cantidad de clases
    vocales_unicas = sorted(list(set(Y)))
    n_clases = len(vocales_unicas)
    
    if n_clases < 2:
        print("Aviso: Solo hay una clase seleccionada. El clustering no tiene sentido.")
        n_clases = 2 # Evitar crash del K-Means

    # Encontrar K-Means ciego
    kmeans = KMeans(n_clusters=n_clases, random_state=42, n_init=10)
    y_pred_kmeans = kmeans.fit_predict(X)
    
    y_true_int = np.array([vocales_unicas.index(v) for v in Y])
    
    # Armar matriz de contingencia/confusión entre reales y falsas
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_int, y_pred_kmeans)
    
    # Algoritmo Húngaro (linear_sum_assignment maximiza coincidencia si pasamos -cm)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Calcular accuracy final basado en el mapeo óptimo
    total_correctos = cm[row_ind, col_ind].sum()
    accuracy = (total_correctos / len(Y)) * 100
    
    # --- INFO OCULTA PARA EL PROFESOR ---
    import pandas as pd
    df_cm_bruta = pd.DataFrame(cm[:len(vocales_unicas), :n_clases], index=vocales_unicas, columns=[f"Clúster {i}" for i in range(n_clases)])
    
    mapeo_str = []
    for real_idx, kmeans_idx in zip(row_ind, col_ind):
        vocal = vocales_unicas[real_idx]
        mapeo_str.append(f"Clúster {kmeans_idx} -> Vocal {vocal}")
        
    print(f"\n=== INFO OCULTA K-MEANS ({nombre}) ===")
    print("Matriz bruta (Sin etiquetas):")
    print(df_cm_bruta.to_string())
    print("\nMapeo óptimo descubierto por el Algoritmo Húngaro:")
    print(" | ".join(mapeo_str))
    print("=======================================")
    
    # --- INFO DETALLADA POR VOCAL ---
    # Reordenar las columnas de la matriz de confusión usando el mapeo del Húngaro
    cm_optima = cm[:, col_ind]
    acc_por_vocal = cm_optima.diagonal() / cm_optima.sum(axis=1) * 100
    
    print(f"\n>> Desglose Accuracy Final para {nombre}:")
    for i, vocal in enumerate(vocales_unicas):
        print(f"   Vocal {vocal}: {acc_por_vocal[i]:.1f}%")
        
    return accuracy, acc_por_vocal, vocales_unicas, df_cm_bruta, mapeo_str

def plot_scatter(X_proj, Y, title, output_path, is_3d=False, variance_ratios=None, connect_points=False):
    fig = plt.figure(figsize=(10, 8))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    vocales = sorted(list(set(Y)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    plot_points = []
    
    for i, vocal in enumerate(vocales):
        idx = Y == vocal
        # Para cuando dibujamos solo los centroides (1 solo punto por vocal)
        if type(idx) == np.bool_ and idx == False:
            idx = np.array([True if y == vocal else False for y in Y])
            
        # Fix for direct arrays
        if isinstance(idx, bool):
            idx = Y == vocal
            
        if is_3d:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], label=vocal, color=palette[i], alpha=0.9, s=80)
            if connect_points and len(X_proj[idx]) > 0:
                plot_points.append(X_proj[idx][0])
        else:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], label=vocal, color=palette[i], alpha=0.9, s=80)
            if connect_points and len(X_proj[idx]) > 0:
                plot_points.append(X_proj[idx][0])
                
    if connect_points and len(plot_points) > 1:
        # Cerrar el polígono
        plot_points.append(plot_points[0])
        pts = np.array(plot_points)
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', linestyle='--', alpha=0.7)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color='gray', linestyle='--', alpha=0.7)
            
    ax.set_title(title)
    if is_3d:
        x_label = f'Componente 1 ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else 'Componente 1'
        y_label = f'Componente 2 ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else 'Componente 2'
        z_label = f'Componente 3 ({variance_ratios[2]*100:.1f}%)' if variance_ratios is not None else 'Componente 3'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)
    else:
        x_label = f'Componente 1 ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else 'Componente 1'
        y_label = f'Componente 2 ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else 'Componente 2'
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
def calcular_centroides_y_distancias(X_proj, Y):
    vocales = sorted(list(set(Y)))
    centroides = {}
    for vocal in vocales:
        idx = Y == vocal
        centroides[vocal] = np.mean(X_proj[idx], axis=0)
        
    n = len(vocales)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(centroides[vocales[i]] - centroides[vocales[j]])
            
    return centroides, dist_matrix, vocales

def ejecutar_procesamiento(
    mediciones, 
    base_dir, 
    alpha_ruido, 
    snr_threshold, 
    outlier_contamination, 
    smooth_ms, 
    target_length, 
    notch_q,
    umap_n_neighbors,
    umap_min_dist,
    umap_metric,
    umap_supervised=False
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "resultados_umap_supervisado")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n2. Extracción y concatenación de características de {len(mediciones)} mediciones...")
    X, Y, Tomas, SNRs = extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=alpha_ruido, smooth_ms=smooth_ms, notch_q=notch_q, target_len=target_length)
    
    if len(X) == 0:
        print("Error: No se obtuvieron datos válidos para procesar.")
        return
        
    print(f"\nDatos recolectados: {X.shape[0]} repeticiones (pulsos), Dimensión de cada feature: {X.shape[1]}")
    X = np.array(X)
    
    # Guardamos los originales SIN FILTRAR para exportarlos al final
    X_orig = X.copy()
    Y_orig = np.array(Y).copy()
    Tomas_orig = np.array(Tomas).copy()
    
    # ------------------ Filtro de Outliers ------------------
    print("\n3. Aplicando filtro de Outliers (Isolation Forest)...")
    from sklearn.ensemble import IsolationForest
    
    X_clean = []
    Y_clean = []
    Tomas_clean = []
    outliers_detectados = 0
    descartados = []
    
    # Filtramos por clase (vocal) para no eliminar varianza válida inter-clase
    for vocal in np.unique(Y):
        mask = Y == vocal
        X_vocal = X[mask]
        Tomas_vocal = Tomas[mask]
        SNRs_vocal = SNRs[mask]
        
        # 1. Filtro duro por SNR
        valid_snr_mask = SNRs_vocal >= snr_threshold
        for i, is_valid in enumerate(valid_snr_mask):
            if not is_valid:
                outliers_detectados += 1
                razon = f"SNR muy bajo (<{snr_threshold})"
                print(f"  [!] Descartado por {razon}: {Tomas_vocal[i]} (Vocal {vocal}) | SNR: {SNRs_vocal[i]:.2f}")
                descartados.append({"Toma": Tomas_vocal[i], "Vocal": vocal, "SNR": SNRs_vocal[i], "Motivo": razon})
                
        # Quedarse solo con los que pasaron el filtro SNR
        X_vocal_snr = X_vocal[valid_snr_mask]
        Tomas_vocal_snr = Tomas_vocal[valid_snr_mask]
        SNRs_vocal_snr = SNRs_vocal[valid_snr_mask]
        
        # 2. Filtro estadístico (Isolation Forest)
        # Necesitamos un mínimo de muestras para aislar
        if len(X_vocal_snr) > 5 and outlier_contamination > 0:
            # Porcentaje de contaminación esperada (outliers)
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            preds = iso.fit_predict(X_vocal_snr)
            
            for i, is_inlier in enumerate(preds):
                if is_inlier == 1:
                    X_clean.append(X_vocal_snr[i])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_vocal_snr[i])
                else:
                    outliers_detectados += 1
                    razon = "Outlier estadístico (IsolationForest)"
                    print(f"  [!] {razon} removido: {Tomas_vocal_snr[i]} (Vocal {vocal}) | SNR: {SNRs_vocal_snr[i]:.2f}")
                    descartados.append({"Toma": Tomas_vocal_snr[i], "Vocal": vocal, "SNR": SNRs_vocal_snr[i], "Motivo": razon})
        else:
            for i in range(len(X_vocal_snr)):
                X_clean.append(X_vocal_snr[i])
                Y_clean.append(vocal)
                Tomas_clean.append(Tomas_vocal_snr[i])
                
    X = np.array(X_clean)
    Y = np.array(Y_clean)
    Tomas = np.array(Tomas_clean)
    print(f"  -> Total outliers removidos: {outliers_detectados}")
    print(f"  -> Repeticiones finales válidas: {len(X)}")
    Y = np.array(Y)
    
    # ------------------ GRÁFICO DE DESCARTES POR SESIÓN ------------------
    if len(descartados) > 0:
        print("\n[+] Generando Gráfico de Mediciones Descartadas...")
        df_desc = pd.DataFrame(descartados)
        # Extraer Session ID de la toma
        def parse_session_desc(toma_str):
            parts = toma_str.split('_')
            if len(parts) >= 3:
                return f"{parts[1]}_{parts[2]}" 
            return toma_str.split('_Win')[0]
            
        df_desc['Sesion'] = df_desc['Toma'].apply(parse_session_desc)
        conteo_sesiones = df_desc.groupby('Sesion').size().sort_values(ascending=True)
        
        plt.figure(figsize=(10, max(6, len(conteo_sesiones)*0.4)))
        conteo_sesiones.plot(kind='barh', color='salmon', edgecolor='black')
        plt.title('Ventanas Descartadas por Sesión (Filtros SNR + IsolationForest)')
        plt.xlabel('Cantidad de Ventanas Rechazadas')
        plt.ylabel('Sesión de Grabación')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Descartados_por_Sesion.png"), dpi=300)
        plt.close()
        print("  -> Gráfico guardado en 'Descartados_por_Sesion.png'")
    
    # La normalización por pulso ya se aplicó dentro del bucle
    X_scaled = X

    # ------------------ Train / Test Split Físico ------------------
    print("\n4. Aplicando Group Shuffle Split (Train/Test) por Sesión Física...")
    
    def get_session_id(toma_str):
        parts = toma_str.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}" 
        return toma_str.split('_Win')[0]

    sesiones_base = [get_session_id(toma) for toma in Tomas]
    sesiones_unicas = list(set(sesiones_base))
    sesiones_unicas.sort()
    
    np.random.seed(42)
    np.random.shuffle(sesiones_unicas)
    
    train_sesiones_size = int(0.8 * len(sesiones_unicas))
    train_sesiones = set(sesiones_unicas[:train_sesiones_size])
    val_sesiones = set(sesiones_unicas[train_sesiones_size:])
    
    train_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in train_sesiones]
    test_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in val_sesiones]
    
    print(f"  -> Total de Sesiones Físicas: {len(sesiones_unicas)} | Train: {len(train_sesiones)} | Test: {len(val_sesiones)}")
    print(f"  -> Sesiones asignadas a TRAIN: {sorted(list(train_sesiones))}")
    print(f"  -> Sesiones asignadas a TEST: {sorted(list(val_sesiones))}")
    print(f"  -> Ventanas Train: {len(train_indices)} | Ventanas Test: {len(test_indices)}")
    
    if len(test_indices) == 0 or len(train_indices) == 0:
        print("Error: No hay suficientes datos para hacer un split. Intenta seleccionar más mediciones.")
        return
        
    X_train = X_scaled[train_indices]
    Y_train = Y[train_indices]
    Tomas_train = Tomas[train_indices]
    
    X_test = X_scaled[test_indices]
    Y_test = Y[test_indices]
    Tomas_test = Tomas[test_indices]
    
    # ------------------ UMAP SUPERVISADO ------------------
    print(f"\n5. Aplicando UMAP Supervisado (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric={umap_metric})...")
    
    if umap_n_neighbors >= len(X_train):
        umap_n_neighbors = max(2, len(X_train) - 1)
        print(f"  [Aviso] n_neighbors ajustado a {umap_n_neighbors} por falta de muestras en Train.")
        
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    Y_train_encoded = le.fit_transform(Y_train)
    
    # Entrenar modelo UMAP
    umap_2d = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, n_components=2, random_state=42)
    X_train_umap_2d = umap_2d.fit_transform(X_train, y=Y_train_encoded)
    
    umap_3d = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, n_components=3, random_state=42)
    X_train_umap_3d = umap_3d.fit_transform(X_train, y=Y_train_encoded)
    
    # Transformar el set de test ciegamente
    print("  -> Transformando Set de Prueba (Blind Transform)...")
    X_test_umap_2d = umap_2d.transform(X_test)
    X_test_umap_3d = umap_3d.transform(X_test)
    
    # --- LIMPIEZA DE OUTLIERS ESPACIALES EN TRAIN ---
    print("\n[+] Detectando Outliers Espaciales en UMAP 3D (Train)...")
    from sklearn.neighbors import LocalOutlierFactor
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03) # 3% más extremo
    inlier_mask = lof.fit_predict(X_train_umap_3d) == 1
    
    outliers_train_indices = np.where(~inlier_mask)[0]
    if len(outliers_train_indices) > 0:
        print(f"  -> Se identificaron {len(outliers_train_indices)} puntos aislados en el espacio de entrenamiento.")
        with open(os.path.join(out_dir, "Outliers_Espaciales_Train.txt"), "w") as f:
            f.write("LISTA DE VENTANAS AISLADAS EN EL UMAP 3D DE ENTRENAMIENTO:\n")
            f.write("Elimine estos archivos físicos para mejorar la calidad del modelo:\n\n")
            for idx in outliers_train_indices:
                f.write(f"- {Tomas_train[idx]} (Vocal {Y_train[idx]})\n")
        print("  -> Nombres exactos guardados en 'Outliers_Espaciales_Train.txt'")
    else:
        print("  -> No se detectaron outliers espaciales.")
        
    X_train_umap_3d = X_train_umap_3d[inlier_mask]
    X_train_umap_2d = X_train_umap_2d[inlier_mask]
    Y_train = Y_train[inlier_mask]
    Tomas_train = Tomas_train[inlier_mask]

    # Graficar
    plot_scatter(X_train_umap_2d, Y_train, "UMAP 2D (Train Set Limpio) - Supervisado", os.path.join(out_dir, "UMAP_2D_Train.png"), is_3d=False)
    plot_scatter(X_test_umap_2d, Y_test, "UMAP 2D (Test Set Ciego) - Supervisado", os.path.join(out_dir, "UMAP_2D_Test.png"), is_3d=False)
    
    plot_scatter(X_train_umap_3d, Y_train, "UMAP 3D (Train Set Limpio) - Supervisado", os.path.join(out_dir, "UMAP_3D_Train.png"), is_3d=True)
    plot_scatter(X_test_umap_3d, Y_test, "UMAP 3D (Test Set Ciego) - Supervisado", os.path.join(out_dir, "UMAP_3D_Test.png"), is_3d=True)
    
    # ------------------ MÉTRICAS EN TEST SET ------------------
    print("\n6. Calculando distancias (Euclidiana) y Silhouette Scores sobre el TEST SET...")
    sil_umap_2d_test = silhouette_score(X_test_umap_2d, Y_test, metric='euclidean')
    sil_umap_3d_test = silhouette_score(X_test_umap_3d, Y_test, metric='euclidean')
    
    print(f"Silhouette Score (UMAP 2D Test): {sil_umap_2d_test:.4f}")
    print(f"Silhouette Score (UMAP 3D Test): {sil_umap_3d_test:.4f}")
    
    print("\n--- Distancias entre centroides (UMAP 3D Test) ---")
    cent, dist_mat, vocales = calcular_centroides_y_distancias(X_test_umap_3d, Y_test)
    
    df_dist = pd.DataFrame(dist_mat, index=vocales, columns=vocales)
    print(df_dist.to_string())
    
    # ------------------ CLASIFICADOR KNN EN ESPACIO UMAP ------------------
    print("\n--- Evaluando Precisión (Accuracy) con K-Nearest Neighbors ---")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import confusion_matrix
    
    # Entrenar KNN en el espacio UMAP LIMPIO del Train Set (3D)
    knn_3d = KNeighborsClassifier(n_neighbors=5)
    knn_3d.fit(X_train_umap_3d, Y_train)
    Y_pred_3d = knn_3d.predict(X_test_umap_3d)
    
    # Entrenar KNN en el espacio UMAP LIMPIO del Train Set (2D)
    knn_2d = KNeighborsClassifier(n_neighbors=5)
    knn_2d.fit(X_train_umap_2d, Y_train)
    Y_pred_2d = knn_2d.predict(X_test_umap_2d)
    
    # --- GRÁFICO DE ERRORES DE PREDICCIÓN DETALLADO (VENTANAS) ---
    errores = []
    for i in range(len(Y_test)):
        if Y_test[i] != Y_pred_3d[i]:
            toma_str = Tomas_test[i]
            # Acortar un poco el nombre para que quepa en el gráfico
            label = f"{toma_str} (R:{Y_test[i]}->P:{Y_pred_3d[i]})"
            errores.append({'TomaLabel': label})
            
    if len(errores) > 0:
        print(f"\n[!] Se encontraron {len(errores)} predicciones incorrectas en el Test Set.")
        df_err = pd.DataFrame(errores)
        
        # Guardar archivo de texto con el detalle
        with open(os.path.join(out_dir, "Errores_Detalle_Ventanas.txt"), "w") as f:
            f.write("LISTA DE VENTANAS MAL CLASIFICADAS:\n")
            for e in errores:
                f.write(f"- {e['TomaLabel']}\n")
                
        conteo_err = df_err.groupby('TomaLabel').size().sort_index(ascending=False)
        
        plt.figure(figsize=(10, max(6, len(conteo_err)*0.3)))
        conteo_err.plot(kind='barh', color='orange', edgecolor='black')
        plt.title('Ventanas Específicas Mal Clasificadas por KNN (Test Set)')
        plt.xlabel('Ocurrencia (1=Error)')
        plt.ylabel('Ventana de Grabación (Real -> Predicción)')
        plt.xticks([0, 1])
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Errores_Prediccion_Ventanas.png"), dpi=300)
        plt.close()
        print("  -> Gráfico de ventanas guardado en 'Errores_Prediccion_Ventanas.png'")
        print("  -> Lista en texto guardada en 'Errores_Detalle_Ventanas.txt'")
    else:
        print("\n[!] ¡PERFECTO! 0 errores de predicción en el Test Set.")
    acc_knn_3d = accuracy_score(Y_test, Y_pred_3d) * 100
    acc_knn_2d = accuracy_score(Y_test, Y_pred_2d) * 100
    print(f"=> Accuracy KNN (UMAP 3D Test): {acc_knn_3d:.2f}%")
    print(f"=> Accuracy KNN (UMAP 2D Test): {acc_knn_2d:.2f}%")
    
    # Matriz de Confusión Normalizada (Porcentajes) 3D
    vocales_ordenadas = sorted(list(set(Y)))
    cm_3d = confusion_matrix(Y_test, Y_pred_3d, labels=vocales_ordenadas, normalize='true')
    cm_percent_3d = cm_3d * 100
    df_cm_3d = pd.DataFrame(cm_percent_3d, index=vocales_ordenadas, columns=vocales_ordenadas)
    
    # Matriz de Confusión Normalizada (Porcentajes) 2D
    cm_2d = confusion_matrix(Y_test, Y_pred_2d, labels=vocales_ordenadas, normalize='true')
    cm_percent_2d = cm_2d * 100
    df_cm_2d = pd.DataFrame(cm_percent_2d, index=vocales_ordenadas, columns=vocales_ordenadas)
    
    print("\nMatriz de Confusión en % (3D):")
    print(df_cm_3d.round(2).to_string())
    
    # Guardar métricas
    with open(os.path.join(out_dir, "metricas.txt"), "w") as f:
        f.write("========================================================\n")
        f.write("      INFO METRICAS TEST SET (UMAP SUPERVISADO)\n")
        f.write("========================================================\n\n")
        
        f.write(f"Accuracy Clasificación KNN (Test 3D): {acc_knn_3d:.2f}%\n")
        f.write(f"Accuracy Clasificación KNN (Test 2D): {acc_knn_2d:.2f}%\n")
        f.write(f"Silhouette Score (UMAP 3D Test): {sil_umap_3d_test:.4f}\n")
        f.write(f"Silhouette Score (UMAP 2D Test): {sil_umap_2d_test:.4f}\n\n")
        
        f.write("Matriz de Confusión KNN (%) (UMAP 3D Test):\n")
        f.write(df_cm_3d.round(2).to_string() + "\n\n")
        
        f.write("Matriz de Confusión KNN (%) (UMAP 2D Test):\n")
        f.write(df_cm_2d.round(2).to_string() + "\n\n")
        
        f.write("Matriz de Distancias de Centroides (UMAP 3D Test):\n")
        f.write(df_dist.to_string() + "\n\n")
        
    # Guardar reporte de mediciones descartadas
    if descartados:
        df_desc = pd.DataFrame(descartados)
        desc_path = os.path.join(out_dir, "reporte_mediciones_descartadas.csv")
        df_desc.to_csv(desc_path, index=False)
        print(f"\n[!] Guardado reporte de {len(descartados)} mediciones descartadas en {desc_path}")
        
    # Guardar reporte de mediciones PROCESADAS exitosamente
    if Tomas_clean:
        df_procesadas = pd.DataFrame({'Toma': Tomas_clean, 'Vocal': Y_clean})
        proc_path = os.path.join(out_dir, "reporte_mediciones_procesadas.csv")
        df_procesadas.to_csv(proc_path, index=False)
        print(f"[!] Guardado reporte de {len(Tomas_clean)} mediciones exitosamente procesadas en {proc_path}")
        
    # --- NUEVO: GUARDAR TABLAS COMO IMAGEN ---
    print("\n[INFO] Generando tablas en formato Imagen (.png) para el cuaderno...")
    def guardar_tabla_imagen(df, title, filepath, col_width=2.5, row_height=0.625, font_size=12):
        # Crear figura
        fig, ax = plt.subplots(figsize=(df.shape[1]*col_width, (df.shape[0]+1)*row_height))
        ax.axis('off')
        ax.axis('tight')
        
        # Redondear si son floats
        df_str = df.round(2) if df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).any() else df
        
        table = ax.table(cellText=df_str.values, colLabels=df_str.columns, rowLabels=df_str.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.5)
        
        plt.title(title, pad=20, fontsize=font_size+2, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    try:
        # Guardar Tabla de Parámetros
        parametros = {
            "Parámetro": [
                "Alpha Ruido", "SNR Mínimo", "Contaminación Outliers",
                "Ventana RMS (ms)", "Remuestreo (target_len)", "Filtro Notch (Q)",
                "UMAP n_neighbors", "UMAP min_dist", "UMAP métrica"
            ],
            "Valor": [
                alpha_ruido, snr_threshold, outlier_contamination,
                smooth_ms, target_length, notch_q,
                umap_n_neighbors, umap_min_dist, umap_metric
            ]
        }
        df_params = pd.DataFrame(parametros).set_index("Parámetro")
        guardar_tabla_imagen(df_params, "Configuración del Experimento", os.path.join(out_dir, "tabla_parametros.png"), col_width=4.0)
        
        # Guardar Tablas de Distancia
        guardar_tabla_imagen(df_dist, "Matriz de Distancias - UMAP 3D Test", os.path.join(out_dir, "tabla_distancias_umap.png"))
        
        # Mapa de Calor de Matriz de Confusión 3D
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm_3d, annot=True, fmt=".1f", cmap="Blues", cbar=True, vmin=0, vmax=100)
        plt.title(f"Matriz de Confusión KNN - UMAP 3D\n(Accuracy Global: {acc_knn_3d:.2f}%)", pad=15, fontweight='bold', fontsize=14)
        plt.ylabel("Vocal Real", fontweight='bold')
        plt.xlabel("Vocal Predicha (Por KNN)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap_confusion_knn_3D.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Mapa de Calor de Matriz de Confusión 2D
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_cm_2d, annot=True, fmt=".1f", cmap="Blues", cbar=True, vmin=0, vmax=100)
        plt.title(f"Matriz de Confusión KNN - UMAP 2D\n(Accuracy Global: {acc_knn_2d:.2f}%)", pad=15, fontweight='bold', fontsize=14)
        plt.ylabel("Vocal Real", fontweight='bold')
        plt.xlabel("Vocal Predicha (Por KNN)", fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "heatmap_confusion_knn_2D.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de Barras de Aciertos por Vocal 3D
        plt.figure(figsize=(8, 5))
        aciertos_por_vocal_3d = np.diag(df_cm_3d)
        sns.barplot(x=df_cm_3d.index, y=aciertos_por_vocal_3d, palette="viridis")
        plt.ylim(0, 110)
        plt.title("Porcentaje de Acierto por Vocal - UMAP 3D (Test Set)", pad=15, fontweight='bold', fontsize=14)
        plt.ylabel("Accuracy (%)", fontweight='bold')
        plt.xlabel("Vocal", fontweight='bold')
        for i, v in enumerate(aciertos_por_vocal_3d):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "barplot_accuracy_vocal_3D.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gráfico de Barras de Aciertos por Vocal 2D
        plt.figure(figsize=(8, 5))
        aciertos_por_vocal_2d = np.diag(df_cm_2d)
        sns.barplot(x=df_cm_2d.index, y=aciertos_por_vocal_2d, palette="viridis")
        plt.ylim(0, 110)
        plt.title("Porcentaje de Acierto por Vocal - UMAP 2D (Test Set)", pad=15, fontweight='bold', fontsize=14)
        plt.ylabel("Accuracy (%)", fontweight='bold')
        plt.xlabel("Vocal", fontweight='bold')
        for i, v in enumerate(aciertos_por_vocal_2d):
            plt.text(i, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "barplot_accuracy_vocal_2D.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Guardar Resumen de Procesamiento
        if descartados:
            df_resumen_desc = df_desc.groupby('Vocal').size().to_frame('Cant. Descartada')
        else:
            vocales_presentes = sorted(list(set(Y_orig)))
            df_resumen_desc = pd.DataFrame({'Cant. Descartada': [0]*len(vocales_presentes)}, index=vocales_presentes)
            
        df_resumen_proc = df_procesadas.groupby('Vocal').size().to_frame('Cant. Procesada')
        df_resumen = df_resumen_proc.join(df_resumen_desc, how='outer').fillna(0).astype(int)
        df_resumen['Total'] = df_resumen['Cant. Procesada'] + df_resumen['Cant. Descartada']
        
        guardar_tabla_imagen(df_resumen, "Resumen de Mediciones por Vocal", os.path.join(out_dir, "tabla_resumen_mediciones.png"))
        
        # --- TABLA DETALLADA DE MEDICIONES ---
        detalle_procesadas = df_procesadas.copy()
        detalle_procesadas['Estado'] = 'Procesada'
        detalle_procesadas['Motivo / SNR'] = '-'
        
        if descartados:
            detalle_descartadas = df_desc.copy()
            detalle_descartadas['Estado'] = 'Descartada'
            detalle_descartadas['Motivo / SNR'] = detalle_descartadas['Motivo'] + " (SNR: " + detalle_descartadas['SNR'].round(2).astype(str) + ")"
            detalle_descartadas = detalle_descartadas[['Toma', 'Vocal', 'Estado', 'Motivo / SNR']]
        else:
            detalle_descartadas = pd.DataFrame(columns=['Toma', 'Vocal', 'Estado', 'Motivo / SNR'])
            
        df_detalle = pd.concat([detalle_procesadas, detalle_descartadas], ignore_index=True)
        # Ordenar alfabéticamente por Toma para que queden A_T1, A_T2...
        df_detalle = df_detalle.sort_values(by=['Vocal', 'Toma']).set_index('Toma')
        
        print("  -> ¡Tablas en imagen guardadas exitosamente!")
    except Exception as e:
        print(f"  -> Error al generar imágenes de tablas: {e}")

    # Exportar DataFrame de características para visor_features.py
    print("\n6. Exportando características (SIN FILTRAR) a CSV para auditoría visual...")
    n_features = X_orig.shape[1]
    cols = []
    # Asumimos que los features están en orden: Ch0 (100 pts), Ch1 (100 pts), Ch2 (100 pts)
    puntos_por_canal = n_features // 3
    for ch in range(3):
        for t in range(puntos_por_canal):
            cols.append(f"Ch{ch}_T{t}")
            
    # Exportamos las limpias (después de SNR y Outliers) para que el autoencoder no entrene con basura
    df_export = pd.DataFrame(X_clean, columns=cols)
    df_export.insert(0, 'Toma', Tomas_clean)
    df_export.insert(0, 'Vocal', Y_clean)
    
    csv_out_path = os.path.join(out_dir, "caracteristicas_exportadas.csv")
    df_export.to_csv(csv_out_path, index=False)
    print(f"Dataset LIMPIO exportado exitosamente a: {csv_out_path}")
    
    # Exportamos también las sin filtrar
    df_sucio = pd.DataFrame(X_orig, columns=cols)
    df_sucio.insert(0, 'Toma', Tomas_orig)
    df_sucio.insert(0, 'Vocal', Y_orig)
    
    csv_sucio_path = os.path.join(out_dir, "caracteristicas_sin_filtrar.csv")
    df_sucio.to_csv(csv_sucio_path, index=False)
        
    print(f"\nProceso completado. Resultados guardados en {out_dir}")

class GeneradorUMAPSupervisadoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador UMAP Supervisado (Train/Test Split)")
        self.root.geometry("600x850")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "base_de_datos_electrodos")
        
        main_frame = tk.Frame(root, padx=15, pady=15, bg="#0B0C10")
        main_frame.pack(fill="both", expand=True)
        
        # --- Listado de Mediciones ---
        meas_frame = tk.LabelFrame(main_frame, text="Seleccionar Mediciones para PCA/UMAP", padx=10, pady=10, bg="#1F2833", fg="#66FCF1")
        meas_frame.pack(fill="both", expand=True, pady=(0,10))
        
        self.listbox_mediciones = tk.Listbox(meas_frame, selectmode=tk.EXTENDED, bg="#0B0C10", fg="#66FCF1")
        self.listbox_mediciones.pack(side="left", fill="both", expand=True)
        
        sb = tk.Scrollbar(meas_frame, orient="vertical", command=self.listbox_mediciones.yview)
        sb.pack(side="right", fill="y")
        self.listbox_mediciones.config(yscrollcommand=sb.set)
        
        # --- Parámetros Configurables ---
        params_frame = tk.LabelFrame(main_frame, text="Parámetros DSP y Limpieza", padx=10, pady=10, bg="#1F2833", fg="#66FCF1")
        params_frame.pack(fill="x", pady=(0,15))
        
        # 1. Alpha Ruido
        f1 = tk.Frame(params_frame, bg="#1F2833")
        f1.pack(fill="x", pady=2)
        tk.Label(f1, text="Agresividad Resta de Ruido (Alpha):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_alpha = tk.Entry(f1, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_alpha.pack(side="left")
        self.ent_alpha.insert(0, "1.0")
        
        # 2. SNR Threshold
        f2 = tk.Frame(params_frame, bg="#1F2833")
        f2.pack(fill="x", pady=2)
        tk.Label(f2, text="Filtro Duro SNR Mínimo (ej: 0.5):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_snr = tk.Entry(f2, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_snr.pack(side="left")
        self.ent_snr.insert(0, "0.5")
        
        # 3. Contaminacion IsolationForest
        f3 = tk.Frame(params_frame, bg="#1F2833")
        f3.pack(fill="x", pady=2)
        tk.Label(f3, text="Tasa de Outliers Estadísticos (0.05=5%):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_outliers = tk.Entry(f3, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_outliers.pack(side="left")
        self.ent_outliers.insert(0, "0.05")
        
        # 4. Suavizado RMS (ms)
        f4 = tk.Frame(params_frame, bg="#1F2833")
        f4.pack(fill="x", pady=2)
        tk.Label(f4, text="Ventana Envolvente RMS (ms):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_smooth = tk.Entry(f4, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth.pack(side="left")
        self.ent_smooth.insert(0, "75")
        
        # 4.b Longitud Objetivo (Remuestreo)
        f4b = tk.Frame(params_frame, bg="#1F2833")
        f4b.pack(fill="x", pady=2)
        tk.Label(f4b, text="Puntos de Remuestreo (target_len):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_target_len = tk.Entry(f4b, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_target_len.pack(side="left")
        self.ent_target_len.insert(0, "20")
        
        # 5. Filtro Notch (Q)
        f5 = tk.Frame(params_frame, bg="#1F2833")
        f5.pack(fill="x", pady=2)
        tk.Label(f5, text="Filtro Notch Q Factor:", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_notch = tk.Entry(f5, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_notch.pack(side="left")
        self.ent_notch.insert(0, "2.0")
        
        # --- Parámetros UMAP ---
        umap_frame = tk.LabelFrame(main_frame, text="Parámetros de Proyección UMAP", padx=10, pady=10, bg="#1F2833", fg="#66FCF1")
        umap_frame.pack(fill="x", pady=(0,15))
        
        # UMAP n_neighbors
        fu1 = tk.Frame(umap_frame, bg="#1F2833")
        fu1.pack(fill="x", pady=2)
        tk.Label(fu1, text="n_neighbors (Local vs Global):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_umap_nn = tk.Entry(fu1, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_nn.pack(side="left")
        self.ent_umap_nn.insert(0, "20")
        
        # UMAP min_dist
        fu2 = tk.Frame(umap_frame, bg="#1F2833")
        fu2.pack(fill="x", pady=2)
        tk.Label(fu2, text="min_dist (Densidad de clúster):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_umap_md = tk.Entry(fu2, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_md.pack(side="left")
        self.ent_umap_md.insert(0, "0.1")
        
        # UMAP metric
        fu3 = tk.Frame(umap_frame, bg="#1F2833")
        fu3.pack(fill="x", pady=2)
        tk.Label(fu3, text="Métrica de distancia:", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.combo_metric = ttk.Combobox(fu3, values=["euclidean", "cosine", "manhattan", "correlation"], width=15)
        self.combo_metric.pack(side="left")
        self.combo_metric.set("euclidean")
        
        # --- Botón Procesar ---
        self.btn_procesar = tk.Button(main_frame, text="Generar Test/Train y Visualizar", command=self.iniciar_procesamiento, bg="#45A29E", fg="white", font=("Arial", 12, "bold"))
        self.btn_procesar.pack(fill="x", pady=10)
        
        self.cargar_mediciones()

    def cargar_mediciones(self):
        self.listbox_mediciones.delete(0, tk.END)
        mediciones = procesar_mediciones(self.base_dir)
        for med in mediciones:
            self.listbox_mediciones.insert(tk.END, med)

    def iniciar_procesamiento(self):
        seleccionadas = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        if not seleccionadas:
            messagebox.showwarning("Advertencia", "Debe seleccionar al menos una medición.")
            return
            
        try:
            val_alpha = float(self.ent_alpha.get())
            val_snr = float(self.ent_snr.get())
            val_outliers = float(self.ent_outliers.get())
            val_smooth = int(self.ent_smooth.get())
            val_target_len = int(self.ent_target_len.get())
            val_notch_q = float(self.ent_notch.get())
            
            val_umap_nn = int(self.ent_umap_nn.get())
            val_umap_md = float(self.ent_umap_md.get())
            val_umap_metric = self.combo_metric.get()
            
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos")
            return
            
        self.root.destroy()
        ejecutar_procesamiento(
            seleccionadas, 
            self.base_dir, 
            val_alpha, 
            val_snr, 
            val_outliers, 
            val_smooth, 
            val_target_len, 
            notch_q=val_notch_q,
            umap_n_neighbors=val_umap_nn,
            umap_min_dist=val_umap_md,
            umap_metric=val_umap_metric
        )

def main():
    root = tk.Tk()
    app = GeneradorUMAPSupervisadoGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
