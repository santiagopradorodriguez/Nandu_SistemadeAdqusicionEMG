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

def extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=1.0, smooth_ms=250, notch_q=30.0, target_len=100, return_raw_cache=False, aplicar_trevisan=False, modo_alineacion="Pico Volumen Micrófono", pre_pct=0.4, post_pct=0.6, canales_features=["canal_0", "canal_1", "canal_2"]):
    """
    Extrae y alinea las ventanas de los canales solicitados.
    Devuelve X (matriz de features), Y (labels/vocales) y Tomas (nombres de las mediciones).
    """
    X = []
    Y = []
    Tomas = []
    SNRs = []
    
    canales_procesar = list(set(canales_features + ["canal_3"]))
    
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
                    tipo_envolvente="rms", smooth_ms=smooth_ms,
                    pre_pct=pre_pct, post_pct=post_pct
                )
            finally:
                sys.stdout = old_stdout
            
            if res_final:
                fname = list(res_final.keys())[0]
                canales_data[ch] = res_final[fname]
                
        if len(canales_data) < len(canales_procesar):
            print(f"  -> Se omite porque no se pudieron cargar los {len(canales_procesar)} canales solicitados.")
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
        
        # --- ALINEACIÓN POR DERIVADA (NUEVO) ---
        if modo_alineacion == "Pico Derivada Micrófono (Onset)":
            periodo_sec = 60.0 / bpm_u
            sr_aprox = int(muestras_pulso / periodo_sec)
            
            deriv_mic = np.gradient(env_mic_raw)
            win_size = max(1, int(sr_aprox * 0.25))
            deriv_mic = np.convolve(deriv_mic, np.ones(win_size)/win_size, mode='same')
            
            picos_deriv = []
            for p_amp in picos_mic:
                rango_inicio = max(0, int(p_amp - pre_pct * muestras_pulso))
                if rango_inicio < p_amp:
                    idx_rel = np.argmax(deriv_mic[rango_inicio:p_amp])
                    picos_deriv.append(rango_inicio + idx_rel)
                else:
                    picos_deriv.append(p_amp)
            picos_mic = np.array(picos_deriv)
        
        TARGET_LEN = target_len
        
        # Almacenamiento temporal para esta medición
        ventanas_medicion = []
        picos_medicion = [] # para guardar el máximo de cada canal en la ventana
        
        for win_idx, pico in enumerate(picos_mic):
            # Definir ventana física simétrica basada en el pico del micrófono
            pre_samples = int(muestras_pulso * pre_pct)
            post_samples = int(muestras_pulso * post_pct)
            
            real_cut_start = pico - pre_samples
            real_cut_end = pico + post_samples
            
            # Verificar límites
            if real_cut_start < 0 or real_cut_end > len(env_mic_raw):
                continue
                
            valido = True
            segs_brutos = []
            max_supremo = 1e-9
            ruido_acumulado_window = 0.0
            picos_canales = []
            
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
                picos_canales.append(m_val)
                
            if not valido:
                continue
                
            ventanas_medicion.append({
                'win_idx': win_idx,
                'segs_brutos': segs_brutos,
                'max_supremo': max_supremo,
                'ruido_acumulado': ruido_acumulado_window
            })
            picos_medicion.append(picos_canales)
            
        # Aplicar Corrección Trevisan si corresponde
        if aplicar_trevisan and len(picos_medicion) > 0:
            import pandas as pd
            picos_matrix = np.array(picos_medicion)
            df_picos = pd.DataFrame(picos_matrix)
            picos_matrix_suavizados = df_picos.rolling(window=15, center=True, min_periods=1).median().values
            
            picos_detrended = picos_matrix_suavizados.copy()
            x_idx = np.arange(len(picos_detrended))
            for c_idx in range(picos_detrended.shape[1]):
                y_vals = picos_detrended[:, c_idx]
                if len(y_vals) > 1:
                    slope, intercept = np.polyfit(x_idx, y_vals, 1)
                    trend = slope * x_idx + intercept
                    picos_detrended[:, c_idx] = np.maximum(y_vals - trend + np.mean(y_vals), 0.0)
            
            max_pico_ventana = np.max(picos_detrended, axis=1) + 1e-9
            picos_norm = picos_detrended / max_pico_ventana[:, np.newaxis]
        else:
            picos_norm = None
            
        # Empaquetar y resamplear
        for i, v_data in enumerate(ventanas_medicion):
            win_idx = v_data['win_idx']
            segs_brutos = v_data['segs_brutos']
            max_supremo = v_data['max_supremo']
            ruido_acumulado_window = v_data['ruido_acumulado']
            
            if return_raw_cache:
                X.append(segs_brutos)
                Y.append(vocal)
                Tomas.append(f"{med_name}_Win{win_idx}")
                ruido_promedio_total = ruido_acumulado_window / 3.0
                snr = max_supremo / (ruido_promedio_total + 1e-9)
                SNRs.append(snr)
                continue
                
            vector_concatenado = []
            for c_idx, seg in enumerate(segs_brutos):
                if picos_norm is not None:
                    # Corrección Trevisan: el pico del segmento debe igualar a picos_norm[i, c_idx]
                    pico_actual = np.max(seg)
                    if pico_actual > 1e-9:
                        factor = picos_norm[i, c_idx] / pico_actual
                        seg_norm = seg * factor
                    else:
                        seg_norm = seg * 0.0
                else:
                    # Normalización clásica (max_supremo)
                    seg_norm = seg / (max_supremo + 1e-9)
                
                # Remuestreo por FFT
                seg_rs = resample(seg_norm, TARGET_LEN)
                seg_rs[seg_rs < 0] = 0.0
                vector_concatenado.append(seg_rs)
                
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

def evaluar_clustering_no_supervisado(X, Y, nombre, algoritmo="K-Means"):
    # Obtener etiquetas reales y cantidad de clases
    vocales_unicas = sorted(list(set(Y)))
    n_clases = len(vocales_unicas)
    
    if n_clases < 2:
        print("Aviso: Solo hay una clase seleccionada. El clustering no tiene sentido.")
        n_clases = 2 # Evitar crash del modelo

    # Encontrar clústeres ciegos
    if algoritmo == "GMM":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42, n_init=5)
    else:
        model = KMeans(n_clusters=n_clases, random_state=42, n_init=10)
        
    y_pred_kmeans = model.fit_predict(X)
    
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

def plot_distance_heatmap(dist_matrix, vocales, title, output_path):
    plt.figure(figsize=(6, 5))
    plt.style.use('dark_background')
    sns.heatmap(dist_matrix, annot=True, cmap="YlGnBu", xticklabels=vocales, yticklabels=vocales, fmt=".2f", cbar_kws={'label': 'Distancia Euclidiana'})
    plt.title(title, color="white", pad=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='#0B0C10')
    plt.close()

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
    pca_comps=[1, 2, 3],
    aplicar_trevisan=False,
    algoritmo_clustering="K-Means",
    modo_alineacion="Pico Volumen Micrófono",
    pre_pct=0.4,
    post_pct=0.6,
    canales_features=["canal_0", "canal_1", "canal_2"]
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "resultados_pca_umap")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n2. Extracción y concatenación de características de {len(mediciones)} mediciones (Trevisan={aplicar_trevisan}, Alineación={modo_alineacion})...")
    print(f"   -> Canales incluidos en PCA: {', '.join(canales_features)}")
    print(f"   -> Recorte de ventana configurado: Pre={pre_pct*100:.1f}%, Post={post_pct*100:.1f}% del período.")
    X, Y, Tomas, SNRs = extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=alpha_ruido, smooth_ms=smooth_ms, notch_q=notch_q, target_len=target_length, aplicar_trevisan=aplicar_trevisan, modo_alineacion=modo_alineacion, pre_pct=pre_pct, post_pct=post_pct, canales_features=canales_features)
    
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
    
    # La normalización por pulso ya se aplicó dentro del bucle
    X_scaled = X

    print(f"\nAplicando PCA y UMAP...")
    
    # ------------------ PCA ------------------
    try:
        req_comps_0 = [c - 1 for c in pca_comps]
        max_c = max(req_comps_0)
        
        # Calcular PCA base con suficientes componentes
        pca_base = PCA(n_components=max_c + 1)
        X_pca_base = pca_base.fit_transform(X_scaled)
        
        # --- PCA N-Dimensional Completo (para clustering y UMAP) ---
        X_pca_selected = X_pca_base[:, req_comps_0]
        
        # --- PCA 2D (Para Graficar) ---
        if len(req_comps_0) >= 2:
            idx_2d = req_comps_0[:2]
        else:
            idx_2d = req_comps_0 + [0] * (2 - len(req_comps_0))
            
        X_pca_2d = X_pca_base[:, idx_2d]
        var_ratios_2d = pca_base.explained_variance_ratio_[idx_2d]
        
        # --- PCA 3D (Para Graficar) ---
        if len(req_comps_0) >= 3:
            idx_3d = req_comps_0[:3]
        else:
            idx_3d = req_comps_0 + [0] * (3 - len(req_comps_0))
            
        X_pca_3d = X_pca_base[:, idx_3d]
        var_ratios_3d = pca_base.explained_variance_ratio_[idx_3d]
        
        var_exp_total = np.sum(pca_base.explained_variance_ratio_[req_comps_0])
        print(f"Varianza explicada por las componentes PCA seleccionadas {pca_comps}: {var_exp_total*100:.2f}%")
        
        plot_scatter(X_pca_2d, Y, f"PCA 2D (Comps: {pca_comps[:2]}) - Vocales EMG", os.path.join(out_dir, "PCA_2D.png"), is_3d=False, variance_ratios=var_ratios_2d)
        plot_scatter(X_pca_3d, Y, f"PCA 3D (Comps: {pca_comps[:3]}) - Vocales EMG", os.path.join(out_dir, "PCA_3D.png"), is_3d=True, variance_ratios=var_ratios_3d)
        
    except Exception as e:
        print(f"Error en el filtrado de componentes PCA: {e}")
        return
    
    # --- PCA 2D Centroides ---
    cent_pca_2d, _, vocales_pca_2d = calcular_centroides_y_distancias(X_pca_2d, Y)
    X_cent_2d = np.array([cent_pca_2d[v] for v in vocales_pca_2d])
    Y_cent_2d = np.array(vocales_pca_2d)
    plot_scatter(X_cent_2d, Y_cent_2d, f"PCA 2D (Comps: {pca_comps[:2]}) - Promedio de Vocales", os.path.join(out_dir, "PCA_2D_Centroides.png"), is_3d=False, variance_ratios=var_ratios_2d, connect_points=True)
    
    # --- PCA 3D Centroides ---
    cent_pca, _, vocales_pca = calcular_centroides_y_distancias(X_pca_3d, Y)
    X_cent = np.array([cent_pca[v] for v in vocales_pca])
    Y_cent = np.array(vocales_pca)
    plot_scatter(X_cent, Y_cent, f"PCA 3D (Comps: {pca_comps[:3]}) - Promedio de Vocales", os.path.join(out_dir, "PCA_3D_Centroides.png"), is_3d=True, variance_ratios=var_ratios_3d, connect_points=True)
    
    # ------------------ UMAP ------------------
    print(f"\n5. Aplicando UMAP (n_neighbors={umap_n_neighbors}, min_dist={umap_min_dist}, metric={umap_metric})...")
    
    # Asegurar que n_neighbors no sea mayor que el número de muestras
    if umap_n_neighbors >= len(X):
        umap_n_neighbors = max(2, len(X) - 1)
        print(f"  [Aviso] n_neighbors ajustado a {umap_n_neighbors} por falta de muestras.")
        
    print("  [INFO] Alimentando UMAP con los datos originales (crudos).")
        
    np.random.seed(42)
    umap_2d = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_scaled)
    
    np.random.seed(42)
    umap_3d = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist, metric=umap_metric, n_components=3, random_state=42)
    X_umap_3d = umap_3d.fit_transform(X_scaled)
    
    plot_scatter(X_umap_2d, Y, "UMAP 2D (Señal Cruda) - Vocales EMG", os.path.join(out_dir, "UMAP_2D.png"), is_3d=False)
    plot_scatter(X_umap_3d, Y, "UMAP 3D (Señal Cruda) - Vocales EMG", os.path.join(out_dir, "UMAP_3D.png"), is_3d=True)
    
    # ------------------ MÉTRICAS ------------------
    print("\n5. Calculando distancias (Euclidiana) y Silhouette Scores...")
    sil_pca_2d = silhouette_score(X_pca_2d, Y, metric='euclidean')
    sil_pca_nd = silhouette_score(X_pca_selected, Y, metric='euclidean')
    
    sil_umap_2d = silhouette_score(X_umap_2d, Y, metric='euclidean')
    sil_umap_3d = silhouette_score(X_umap_3d, Y, metric='euclidean')
    
    print(f"Silhouette Score (PCA 2D): {sil_pca_2d:.4f}")
    print(f"Silhouette Score (PCA {len(pca_comps)}D): {sil_pca_nd:.4f}")
    print(f"Silhouette Score (UMAP 2D): {sil_umap_2d:.4f}")
    print(f"Silhouette Score (UMAP 3D): {sil_umap_3d:.4f}")
    
    print(f"\n--- Distancias entre centroides (PCA {len(pca_comps)}D) ---")
    cent_pca, dist_mat_pca, vocales_pca = calcular_centroides_y_distancias(X_pca_selected, Y)
    df_dist_pca = pd.DataFrame(dist_mat_pca, index=vocales_pca, columns=vocales_pca)
    print(df_dist_pca.to_string())
    
    print("\n--- Distancias entre centroides (UMAP 3D) ---")
    cent, dist_mat, vocales = calcular_centroides_y_distancias(X_umap_3d, Y)
    
    df_dist = pd.DataFrame(dist_mat, index=vocales, columns=vocales)
    print(df_dist.to_string())
    
    plot_distance_heatmap(dist_mat_pca, vocales_pca, f"Matriz de Distancias - Centroides (PCA {len(pca_comps)}D)", os.path.join(out_dir, "heatmap_distancias_pca.png"))
    plot_distance_heatmap(dist_mat, vocales, "Matriz de Distancias - Centroides (UMAP 3D)", os.path.join(out_dir, "heatmap_distancias_umap.png"))
    
    # ------------------ CLUSTERING NO SUPERVISADO ------------------
    print(f"\n--- Evaluando Clustering No Supervisado ({algoritmo_clustering} + Húngaro) ---")
    
    pca_name = f"PCA {len(pca_comps)}D"
    acc_pca_2d, acc_vocales_pca_2d, voc_pca_2d, df_cm_pca_2d, mapeo_pca_2d = evaluar_clustering_no_supervisado(X_pca_2d, Y, "PCA 2D", algoritmo_clustering)
    acc_pca_nd, acc_vocales_pca, voc_pca, df_cm_pca, mapeo_pca = evaluar_clustering_no_supervisado(X_pca_selected, Y, pca_name, algoritmo_clustering)
    
    acc_umap_2d, acc_vocales_umap_2d, voc_umap_2d, df_cm_umap_2d, mapeo_umap_2d = evaluar_clustering_no_supervisado(X_umap_2d, Y, "UMAP 2D", algoritmo_clustering)
    acc_umap_3d, acc_vocales_umap, voc_umap, df_cm_umap, mapeo_umap = evaluar_clustering_no_supervisado(X_umap_3d, Y, "UMAP 3D", algoritmo_clustering)
    
    print(f"\n=> TOTAL Accuracy Clustering No Supervisado (PCA 2D) : {acc_pca_2d:.2f}%")
    print(f"=> TOTAL Accuracy Clustering No Supervisado ({pca_name}) : {acc_pca_nd:.2f}%")
    print(f"=> TOTAL Accuracy Clustering No Supervisado (UMAP 2D): {acc_umap_2d:.2f}%")
    print(f"=> TOTAL Accuracy Clustering No Supervisado (UMAP 3D): {acc_umap_3d:.2f}%")
    
    # Guardar métricas
    with open(os.path.join(out_dir, "metricas.txt"), "w") as f:
        f.write("========================================================\n")
        f.write("      INFO OCULTA DE CLUSTERING (PARA EL PROFESOR)\n")
        f.write("========================================================\n\n")
        f.write(f"--- MATRIZ BRUTA {pca_name} ---\n")
        f.write(df_cm_pca.to_string() + "\n")
        f.write("Mapeo Húngaro: " + " | ".join(mapeo_pca) + "\n\n")
        
        f.write("--- MATRIZ BRUTA UMAP 3D ---\n")
        f.write(df_cm_umap.to_string() + "\n")
        f.write("Mapeo Húngaro: " + " | ".join(mapeo_umap) + "\n\n")
        f.write("========================================================\n\n")
        
        f.write(f"Silhouette Score (PCA 2D): {sil_pca_2d:.4f}\n")
        f.write(f"Silhouette Score ({pca_name}): {sil_pca_nd:.4f}\n")
        f.write(f"Silhouette Score (UMAP 2D): {sil_umap_2d:.4f}\n")
        f.write(f"Silhouette Score (UMAP 3D): {sil_umap_3d:.4f}\n\n")
        
        f.write(f"Accuracy No Supervisado (PCA 2D): {acc_pca_2d:.2f}%\n")
        f.write(f"Accuracy No Supervisado ({pca_name}): {acc_pca_nd:.2f}%\n")
        for i, v in enumerate(voc_pca):
            f.write(f"  - Vocal {v}: {acc_vocales_pca[i]:.2f}%\n")
            
        f.write(f"\nAccuracy No Supervisado (UMAP 2D): {acc_umap_2d:.2f}%\n")
        f.write(f"Accuracy No Supervisado (UMAP 3D): {acc_umap_3d:.2f}%\n")
        for i, v in enumerate(voc_umap):
            f.write(f"  - Vocal {v}: {acc_vocales_umap[i]:.2f}%\n")
            
        f.write(f"\nMatriz de Distancias ({pca_name}):\n")
        f.write(df_dist_pca.to_string() + "\n\n")
        f.write("Matriz de Distancias (UMAP 3D):\n")
        f.write(df_dist.to_string())
        
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
        
        # Redondear correctamente solo columnas numéricas sin fallar por tipos StringDtype
        df_str = df.copy()
        num_cols = df_str.select_dtypes(include=['number']).columns
        if len(num_cols) > 0:
            df_str[num_cols] = df_str[num_cols].round(2)
        
        table = ax.table(cellText=df_str.values, colLabels=df_str.columns, rowLabels=df_str.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1, 1.5)
        
        plt.title(title, pad=20, fontsize=font_size+2, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

    try:
        # Guardar Tablas de Distancia
        guardar_tabla_imagen(df_dist_pca, "Matriz de Distancias - PCA 3D", os.path.join(out_dir, "tabla_distancias_pca.png"))
        guardar_tabla_imagen(df_dist, "Matriz de Distancias - UMAP 3D", os.path.join(out_dir, "tabla_distancias_umap.png"))
        
        # Guardar Tablas de Confusión (Mapeo Húngaro)
        guardar_tabla_imagen(df_cm_pca, "Matriz de Clustering (K-Means vs Real) - PCA 3D", os.path.join(out_dir, "tabla_clustering_pca.png"))
        guardar_tabla_imagen(df_cm_umap, "Matriz de Clustering (K-Means vs Real) - UMAP 3D", os.path.join(out_dir, "tabla_clustering_umap.png"))
        
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
        
        # Al ser una tabla larga (35 filas), achicamos un poco la altura de fila para que entre en la imagen sin que sea gigante
        guardar_tabla_imagen(df_detalle, "Detalle Exacto por Toma", os.path.join(out_dir, "tabla_detalle_mediciones.png"), col_width=3.0, row_height=0.35, font_size=9)
        
        # --- TABLAS DE PARÁMETROS ---
        df_params_dsp = pd.DataFrame({
            "Parámetro": ["Agresividad Resta Ruido (Alpha)", "Filtro Notch (Q)", "Envolvente (Smooth ms)", "Remuestreo (Longitud)", "Filtro de SNR", "Filtro Isolation Forest"],
            "Valor": [str(alpha_ruido), str(notch_q), f"{smooth_ms} ms", f"{target_length} pts", f">= {snr_threshold}", f"{outlier_contamination*100}% outliers"]
        }).set_index("Parámetro")
        
        df_params_umap = pd.DataFrame({
            "Parámetro": ["Nº Vecinos (n_neighbors)", "Distancia Mín. (min_dist)", "Métrica de Distancia", "Dimensiones UMAP"],
            "Valor": [str(umap_n_neighbors), str(umap_min_dist), str(umap_metric).capitalize(), "3D"]
        }).set_index("Parámetro")
        
        guardar_tabla_imagen(df_params_dsp, "Parámetros de Filtrado y DSP", os.path.join(out_dir, "tabla_parametros_dsp.png"), col_width=4.0)
        guardar_tabla_imagen(df_params_umap, "Hiperparámetros UMAP Topológico", os.path.join(out_dir, "tabla_parametros_umap.png"), col_width=4.0)
        
        # --- TABLA DE ACCURACY COMPARATIVA ---
        metricas_nombres = ["Silhouette Score", "Accuracy Global"] + [f"Accuracy Vocal {v}" for v in voc_pca]
        
        pca_2d_vals = [f"{sil_pca_2d:.4f}", f"{acc_pca_2d:.2f}%"] + [f"{acc:.2f}%" for acc in acc_vocales_pca_2d]
        pca_nd_vals = [f"{sil_pca_nd:.4f}", f"{acc_pca_nd:.2f}%"] + [f"{acc:.2f}%" for acc in acc_vocales_pca]
        umap_2d_vals = [f"{sil_umap_2d:.4f}", f"{acc_umap_2d:.2f}%"] + [f"{acc:.2f}%" for acc in acc_vocales_umap_2d]
        umap_3d_vals = [f"{sil_umap_3d:.4f}", f"{acc_umap_3d:.2f}%"] + [f"{acc:.2f}%" for acc in acc_vocales_umap]
        
        df_accuracy = pd.DataFrame({
            "Métrica": metricas_nombres,
            "PCA 2D": pca_2d_vals,
            f"{pca_name}": pca_nd_vals,
            "UMAP 2D": umap_2d_vals,
            "UMAP 3D": umap_3d_vals
        }).set_index("Métrica")
        
        guardar_tabla_imagen(df_accuracy, "Comparativa de Precisión (Accuracy) y Silhouette", os.path.join(out_dir, "tabla_accuracy_comparativa.png"), col_width=2.5, row_height=0.45, font_size=11)
        
        print("  -> ¡Tablas en imagen guardadas exitosamente!")
    except Exception as e:
        print(f"  -> Error al generar imágenes de tablas: {e}")

    # Exportar DataFrame de características para visor_features.py
    print("\n6. Exportando características (SIN FILTRAR) a CSV para auditoría visual...")
    n_features = X_orig.shape[1]
    cols = []
    
    num_canales = len(canales_features)
    puntos_por_canal = n_features // num_canales
    
    for ch_name in canales_features:
        ch_idx = ch_name.split('_')[-1]
        for t in range(puntos_por_canal):
            cols.append(f"Ch{ch_idx}_T{t}")
            
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

class GeneradorPCAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador PCA/UMAP")
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
        
        # --- Selección de Canales ---
        ch_frame = tk.LabelFrame(main_frame, text="Canales EMG a incluir en PCA", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        ch_frame.pack(fill="x", pady=(0,5))
        
        self.var_ch0 = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="Canal 0 (Masetero)", variable=self.var_ch0, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        self.var_ch1 = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="Canal 1 (Orbicular)", variable=self.var_ch1, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        self.var_ch2 = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="Canal 2 (Tiroaritenoideo)", variable=self.var_ch2, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        # --- Parámetros Configurables ---
        params_frame = tk.LabelFrame(main_frame, text="Parámetros DSP y Limpieza", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        params_frame.pack(fill="x", pady=(0,5))
        
        # Row 0: Alpha y SNR
        tk.Label(params_frame, text="Agresividad Ruido (Alpha):", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=0, padx=2, pady=2)
        self.ent_alpha = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_alpha.grid(row=0, column=1, padx=2, pady=2)
        self.ent_alpha.insert(0, "1.0")
        
        tk.Label(params_frame, text="Filtro SNR Mínimo:", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=2, padx=2, pady=2)
        self.ent_snr = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_snr.grid(row=0, column=3, padx=2, pady=2)
        self.ent_snr.insert(0, "0.5")
        
        # Row 1: Outliers y Smooth
        tk.Label(params_frame, text="Outliers (0.05=5%):", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=0, padx=2, pady=2)
        self.ent_outliers = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_outliers.grid(row=1, column=1, padx=2, pady=2)
        self.ent_outliers.insert(0, "0.05")
        
        tk.Label(params_frame, text="Suavizado Env RMS (ms):", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=2, padx=2, pady=2)
        self.ent_smooth = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth.grid(row=1, column=3, padx=2, pady=2)
        self.ent_smooth.insert(0, "90")
        
        # Row 2: Target Len y Notch
        tk.Label(params_frame, text="Pts Remuestreo (LEN):", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=2, column=0, padx=2, pady=2)
        self.ent_target_len = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_target_len.grid(row=2, column=1, padx=2, pady=2)
        self.ent_target_len.insert(0, "20")
        
        tk.Label(params_frame, text="Filtro Notch Q Factor:", width=22, anchor="w", bg="#1F2833", fg="white").grid(row=2, column=2, padx=2, pady=2)
        self.ent_notch = tk.Entry(params_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_notch.grid(row=2, column=3, padx=2, pady=2)
        self.ent_notch.insert(0, "2.0")
        
        # --- Parámetros PCA ---
        pca_frame = tk.LabelFrame(main_frame, text="Parámetros PCA", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        pca_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(pca_frame, text="Componentes a retener (ej: 1,2,3):", width=30, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_pca_comps = tk.Entry(pca_frame, width=15, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_pca_comps.pack(side="left")
        self.ent_pca_comps.insert(0, "1,2,3")
        
        # --- Parámetros UMAP ---
        umap_frame = tk.LabelFrame(main_frame, text="Parámetros UMAP", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        umap_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(umap_frame, text="n_neighbors:", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=0, padx=2, pady=2)
        self.ent_umap_nn = tk.Entry(umap_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_nn.grid(row=0, column=1, padx=2, pady=2)
        self.ent_umap_nn.insert(0, "10")
        
        tk.Label(umap_frame, text="min_dist:", width=10, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=2, padx=2, pady=2)
        self.ent_umap_md = tk.Entry(umap_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_md.grid(row=0, column=3, padx=2, pady=2)
        self.ent_umap_md.insert(0, "0.1")
        
        tk.Label(umap_frame, text="Métrica:", width=10, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=4, padx=2, pady=2)
        self.combo_metric = ttk.Combobox(umap_frame, values=["euclidean", "cosine", "manhattan", "correlation"], width=12)
        self.combo_metric.grid(row=0, column=5, padx=2, pady=2)
        self.combo_metric.set("cosine")
        
        # --- Clustering ---
        cluster_frame = tk.LabelFrame(main_frame, text="Algoritmo de Agrupamiento", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        cluster_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(cluster_frame, text="Seleccionar algoritmo:", width=25, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.combo_cluster = ttk.Combobox(cluster_frame, values=["K-Means", "GMM"], width=15)
        self.combo_cluster.pack(side="left")
        self.combo_cluster.set("K-Means")
        
        # --- Normalización Avanzada (DSP y Trevisan) ---
        trev_frame = tk.LabelFrame(main_frame, text="DSP Avanzado y Normalización", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        trev_frame.pack(fill="x", pady=(0,5))
        
        self.var_aplicar_trevisan = tk.BooleanVar(value=True)
        tk.Checkbutton(trev_frame, text="Aplicar Corrección Trevisan (Mediana Móvil + Detrending)", variable=self.var_aplicar_trevisan, bg="#1F2833", fg="white", selectcolor="#0B0C10").grid(row=0, column=0, columnspan=2, sticky="w")
        
        tk.Label(trev_frame, text="Pre-Ventana (%):", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=0, padx=2, pady=2)
        self.ent_pre_pct = tk.Entry(trev_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_pre_pct.grid(row=1, column=1, padx=2, pady=2)
        self.ent_pre_pct.insert(0, "0.4")
        
        tk.Label(trev_frame, text="Post-Ventana (%):", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=2, padx=2, pady=2)
        self.ent_post_pct = tk.Entry(trev_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_post_pct.grid(row=1, column=3, padx=2, pady=2)
        self.ent_post_pct.insert(0, "0.6")
        
        # --- Alineación Temporal ---
        align_frame = tk.LabelFrame(main_frame, text="Alineación Temporal", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        align_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(align_frame, text="Centrar ventana en:", width=20, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.combo_align = ttk.Combobox(align_frame, values=["Pico Volumen Micrófono", "Pico Derivada Micrófono (Onset)"], width=30)
        self.combo_align.pack(side="left")
        self.combo_align.set("Pico Volumen Micrófono")
        
        # --- Botón Procesar ---
        self.btn_procesar = tk.Button(main_frame, text="Generar Dataset y Visualizar", command=self.iniciar_procesamiento, bg="#45A29E", fg="white", font=("Arial", 12, "bold"))
        self.btn_procesar.pack(fill="x", pady=5)
        
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
            
            pca_comps_str = self.ent_pca_comps.get()
            val_pca_comps = [int(x.strip()) for x in pca_comps_str.split(',')]
            if len(val_pca_comps) < 2:
                messagebox.showwarning("Advertencia", "Debe ingresar al menos 2 componentes PCA.")
                return
            
            val_trevisan = self.var_aplicar_trevisan.get()
            val_pre_pct = float(self.ent_pre_pct.get())
            val_post_pct = float(self.ent_post_pct.get())
            val_algoritmo = self.combo_cluster.get()
            val_align = self.combo_align.get()
            
            canales_sel = []
            if self.var_ch0.get(): canales_sel.append("canal_0")
            if self.var_ch1.get(): canales_sel.append("canal_1")
            if self.var_ch2.get(): canales_sel.append("canal_2")
            
            if not canales_sel:
                messagebox.showwarning("Advertencia", "Debe seleccionar al menos 1 canal muscular.")
                return
            
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
            umap_metric=val_umap_metric,
            pca_comps=val_pca_comps,
            aplicar_trevisan=val_trevisan,
            algoritmo_clustering=val_algoritmo,
            modo_alineacion=val_align,
            pre_pct=val_pre_pct,
            post_pct=val_post_pct,
            canales_features=canales_sel
        )

def main():
    root = tk.Tk()
    app = GeneradorPCAGUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[!] Programa cerrado por el usuario (Ctrl+C).")
        import sys
        sys.exit(0)

if __name__ == "__main__":
    main()
