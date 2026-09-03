# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generador de proyecciones y clustering PCA/UMAP con visualizaciones 2D/3D avanzadas.
# ==============================================================================

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

# ==========================================
# CONFIGURACIÓN GLOBAL DE ESTÉTICA
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.5
plt.rcParams['grid.color'] = '#b0b0b0'
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'white'
# ==========================================

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

def export_cluster_data(output_path, algoritmo, model, centroids, vocales_unicas, cluster_to_vocal_idx, is_3d=False):
    import json
    json_path = output_path.replace('.png', f'_centroides_{algoritmo}.json')
    
    data = {
        "algoritmo": algoritmo,
        "n_dimensiones": 3 if is_3d else 2,
        "vocales": vocales_unicas,
        "centroides_mapeados": {}
    }
    
    for kmeans_idx, real_idx in cluster_to_vocal_idx.items():
        vocal = vocales_unicas[real_idx]
        centroid = centroids[kmeans_idx].tolist()
        data["centroides_mapeados"][vocal] = centroid
        
    if algoritmo == "GMM" and hasattr(model, 'covariances_'):
        data["covarianzas_mapeadas"] = {}
        for kmeans_idx, real_idx in cluster_to_vocal_idx.items():
            vocal = vocales_unicas[real_idx]
            cov = model.covariances_[kmeans_idx].tolist()
            data["covarianzas_mapeadas"][vocal] = cov
            
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"  -> Datos exportados a {os.path.basename(json_path)}")

def procesar_mediciones(base_dir):
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    
    mediciones_list = []
    
    if not os.path.isdir(base_dir):
        print(f"Error: No se encontró el directorio {base_dir}")
        return []
        
    for date_folder in sorted(os.listdir(base_dir), reverse=True):
        date_path = os.path.join(base_dir, date_folder)
        if os.path.isdir(date_path) and date_pattern.match(date_folder):
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

def extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=1.0, gate_ratio_ruido=8.0, smooth_ms=250, notch_q=2.0, target_len=100, return_raw_cache=False, aplicar_trevisan=False, modo_alineacion="Pico Volumen Micrófono", pre_pct=0.4, post_pct=0.6, canales_features=["canal_0", "canal_1", "canal_2"], ignorar_ventana_cero=False, cache_canales_data=None, aplicar_correccion_intersesion=False):
    """
    Extrae y alinea las ventanas de los canales solicitados.
    Devuelve X (matriz de features), Y (labels/vocales) y Tomas (nombres de las mediciones).
    """
    X = []
    Y = []
    Tomas = []
    SNRs = []
    mediciones_procesadas = []
    
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
            
        # Leemos archivo manual de exclusiones si existe
        excluded_windows = []
        exclude_path = os.path.join(med_path, 'excluded_windows.json')
        if os.path.exists(exclude_path):
            with open(exclude_path, 'r') as f:
                data_excl = json.load(f)
                excluded_windows = data_excl.get("excluded_windows", [])
                
        cache_key = f"{med_name}_{smooth_ms}_{notch_q}"
        if cache_canales_data is not None and cache_key in cache_canales_data:
            canales_data = cache_canales_data[cache_key]
        else:
            canales_data = {}
            for ch in canales_procesar:
                carpeta = os.path.join(med_path, ch)
                if not os.path.exists(carpeta):
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
                    
            if cache_canales_data is not None and len(canales_data) == len(canales_procesar):
                cache_canales_data[cache_key] = canales_data
                
        if len(canales_data) < len(canales_procesar):
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
            if ignorar_ventana_cero and win_idx == 0:
                continue
                
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
                
                # 1. Detector de Ruido Puro Multivariable (Rel_IQR * Ratio_Centro * Tau_Norm)
                p75_ch, p25_ch = np.percentile(segmento_ch, 75), np.percentile(segmento_ch, 25)
                iqr_ch = p75_ch - p25_ch
                rel_iqr_ch = (np.max(segmento_ch) - np.median(segmento_ch)) / (iqr_ch + 1e-9)
                
                centro_samples = int(0.250 * 2000)
                idx_max_ch = np.argmax(segmento_ch)
                c_i = max(0, idx_max_ch - centro_samples // 2)
                c_f = min(len(segmento_ch), idx_max_ch + centro_samples // 2)
                energia_centro = np.mean(segmento_ch[c_i:c_f]**2)
                mascara_bordes = np.ones(len(segmento_ch), dtype=bool)
                mascara_bordes[c_i:c_f] = False
                energia_bordes = np.mean(segmento_ch[mascara_bordes]**2) if np.sum(mascara_bordes) > 0 else 1.0
                ratio_energia_centro = energia_centro / (energia_bordes + 1e-9)
                
                seg_zero_mean = segmento_ch - np.mean(segmento_ch)
                if np.std(seg_zero_mean) > 0:
                    from scipy.signal import correlate
                    acorr = correlate(seg_zero_mean, seg_zero_mean, mode="full")
                    acorr = acorr[len(acorr)//2:]
                    acorr = acorr / (acorr[0] + 1e-9)
                    lags_50 = np.where(acorr < 0.5)[0]
                    tau_50_ms = (lags_50[0] / 2000.0) * 1000.0 if len(lags_50) > 0 else 0.0
                else:
                    tau_50_ms = 0.0
                    
                score_activacion = rel_iqr_ch * ratio_energia_centro * (tau_50_ms / 50.0)
                
                # 2. Resta del piso de ruido interpulso
                segmento_ch = np.maximum(segmento_ch - ruido_a_restar, 0.0)
                
                # 3. Compuerta de Ruido Puro: si el score no supera el umbral, mandar a 0.0
                if gate_ratio_ruido > 0 and (score_activacion < gate_ratio_ruido):
                    segmento_ch = np.zeros_like(segmento_ch)
                
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
            
        mediciones_procesadas.append({
            'rel_path': rel_path,
            'med_name': med_name,
            'vocal': vocal,
            'ventanas_medicion': ventanas_medicion,
            'picos_norm': picos_norm
        })

    # -------------------------------------------------------------
    # CALIBRACIÓN INTERSESIÓN POR LOTE (SI ESTÁ ACTIVA)
    # -------------------------------------------------------------
    session_factors = {}
    if aplicar_correccion_intersesion:
        sessions_dict = {}
        for m_data in mediciones_procesadas:
            r_path = m_data['rel_path']
            date_folder = os.path.dirname(r_path)
            m_base = os.path.basename(r_path)
            partes_s = m_base.split('_')
            session_tag = '_'.join(partes_s[1:]) if len(partes_s) > 1 else m_base
            s_key = (date_folder, session_tag)
            sessions_dict.setdefault(s_key, []).append(m_data)

        for s_key, meds_sesion in sessions_dict.items():
            all_ventanas = [w for m in meds_sesion for w in m['ventanas_medicion']]
            if not all_ventanas:
                continue
            V = []
            for c_idx in range(len(canales_features)):
                picos_canal = [w['segs_brutos'][c_idx].max() for w in all_ventanas if len(w['segs_brutos']) > c_idx]
                p95 = float(np.percentile(picos_canal, 95)) if picos_canal else 1.0
                V.append(p95)
            V = np.array(V)
            V_ref = float(np.max(V))
            if V_ref > 1e-9:
                alpha = V / V_ref
                # alpha_piso = 0.20 garantiza no amplificar más de 5.0x
                C = 1.0 / np.maximum(alpha, 0.20)
            else:
                C = np.ones(len(canales_features))
            session_factors[s_key] = C
            s_date, s_tag = s_key
            ch_str = ", ".join([f"{canales_features[c]}: C={C[c]:.2f} (p95={V[c]:.2f})" for c in range(len(canales_features))])
            print(f"[Intersesión] Sesión '{s_tag}' ({s_date}) -> {ch_str}")

    # -------------------------------------------------------------
    # EMPAQUETADO, ESCALADO Y NORMALIZACIÓN TRICANAL
    # -------------------------------------------------------------
    for m_data in mediciones_procesadas:
        r_path = m_data['rel_path']
        med_name = m_data['med_name']
        vocal = m_data['vocal']
        picos_norm = m_data['picos_norm']
        
        date_folder = os.path.dirname(r_path)
        partes_s = med_name.split('_')
        session_tag = '_'.join(partes_s[1:]) if len(partes_s) > 1 else med_name
        s_key = (date_folder, session_tag)
        
        C = session_factors.get(s_key, np.ones(len(canales_features))) if aplicar_correccion_intersesion else np.ones(len(canales_features))
        
        for i, v_data in enumerate(m_data['ventanas_medicion']):
            win_idx = v_data['win_idx']
            segs_brutos = v_data['segs_brutos']
            ruido_acumulado_window = v_data['ruido_acumulado']
            
            if aplicar_correccion_intersesion:
                segs_proc = [segs_brutos[c_idx] * C[c_idx] for c_idx in range(len(segs_brutos))]
                max_supremo = max([np.max(s) for s in segs_proc]) if segs_proc else 1e-9
            else:
                segs_proc = segs_brutos
                max_supremo = v_data['max_supremo']
                
            if return_raw_cache:
                X.append(segs_proc)
                Y.append(vocal)
                Tomas.append(f"{med_name}_Win{win_idx}")
                ruido_promedio_total = ruido_acumulado_window / float(len(canales_features))
                snr = max_supremo / (ruido_promedio_total + 1e-9)
                SNRs.append(snr)
                continue
                
            vector_concatenado = []
            for c_idx, seg in enumerate(segs_proc):
                if picos_norm is not None:
                    pico_actual = np.max(seg)
                    if pico_actual > 1e-9:
                        factor = picos_norm[i, c_idx] / pico_actual
                        seg_norm = seg * factor
                    else:
                        seg_norm = seg * 0.0
                else:
                    seg_norm = seg / (max_supremo + 1e-9)
                    
                seg_rs = resample(seg_norm, TARGET_LEN)
                seg_rs[seg_rs < 0] = 0.0
                vector_concatenado.append(seg_rs)
                
            vector_concatenado = np.concatenate(vector_concatenado)
            X.append(vector_concatenado)
            Y.append(vocal)
            Tomas.append(f"{med_name}_Win{win_idx}")
            ruido_promedio_total = ruido_acumulado_window / float(len(canales_features))
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

def evaluar_clustering_no_supervisado(X, Y, nombre, algoritmo="K-Means", verbose=True):
    # Obtener etiquetas reales y cantidad de clases
    vocales_unicas = sorted(list(set(Y)))
    n_clases = len(vocales_unicas)
    
    if n_clases < 2:
        if verbose: print("Aviso: Solo hay una clase seleccionada. El clustering no tiene sentido.")
        n_clases = 2 # Evitar crash del modelo

    # Encontrar clústeres ciegos
    if algoritmo == "GMM":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42, n_init=100, max_iter=500)
    else:
        model = KMeans(n_clusters=n_clases, random_state=42, n_init=100)
        
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
        
    if verbose:
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
    macro_accuracy = float(np.mean(acc_por_vocal))
    
    # --- MATRIZ ÓPTIMA EN PORCENTAJES ---
    # Convertimos cada fila en porcentajes del total de esa clase
    cm_optima_pct = (cm_optima / cm_optima.sum(axis=1)[:, np.newaxis]) * 100
    df_cm_optima = pd.DataFrame(cm_optima_pct, index=vocales_unicas, columns=vocales_unicas)
    
    if verbose:
        print(f"\n>> Desglose Accuracy Final para {nombre}:")
        for i, vocal in enumerate(vocales_unicas):
            print(f"   Vocal {vocal}: {acc_por_vocal[i]:.1f}%")
        print(f"   Exactitud Macro Promedio: {macro_accuracy:.2f}% (Micro: {accuracy:.2f}%)")
        
    return macro_accuracy, acc_por_vocal, vocales_unicas, df_cm_optima, mapeo_str

def plot_scatter(X_proj, Y, title, output_path, is_3d=False, variance_ratios=None, connect_points=False, **kwargs):
    # Estilo académico (Paper)
    plt.style.use('default')
    fig = plt.figure(figsize=(9, 7), facecolor='white')
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        # Grilla 3D sutil
        ax.grid(color='#d3d3d3', linestyle='-', linewidth=0.5)
    else:
        ax = fig.add_subplot(111)
        ax.set_facecolor('white')
        ax.grid(color='#e0e0e0', linestyle='--', linewidth=0.7, alpha=0.7)
        # Removiendo bordes innecesarios (Top y Right)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.tick_params(colors='black')
        
    vocales = sorted(list(set(Y)))
    # Paleta estándar, clara y amigable para publicaciones (Set1 o tab10)
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    plot_points = []
    
    for i, vocal in enumerate(vocales):
        idx = Y == vocal
        if type(idx) == np.bool_ and idx == False:
            idx = np.array([True if y == vocal else False for y in Y])
        if isinstance(idx, bool):
            idx = Y == vocal
            
        color = palette[i % len(palette)]
            
        if is_3d:
            # Puntos sólidos con bordes oscuros sutiles para mejorar legibilidad
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], label=vocal, color=color, alpha=1.0, s=60, edgecolors='black', linewidth=0.3, depthshade=False)
            if connect_points and len(X_proj[idx]) > 0:
                plot_points.append(X_proj[idx][0])
        else:
            # Puntos sólidos
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], label=vocal, color=color, alpha=1.0, s=60, edgecolors='black', linewidth=0.3)
            if connect_points and len(X_proj[idx]) > 0:
                plot_points.append(X_proj[idx][0])
                
    if is_3d:
        # Calcular límites espaciales y añadir planos de proyección 2D
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        z_min, z_max = ax.get_zlim()
        if 'xlim' in kwargs:
            x_min, x_max = kwargs['xlim']
        if 'ylim' in kwargs:
            y_min, y_max = kwargs['ylim']
        if 'zlim' in kwargs:
            z_min, z_max = kwargs['zlim']
            
        for i, vocal in enumerate(vocales):
            idx = Y == vocal
            if type(idx) == np.bool_ and idx == False:
                idx = np.array([True if y == vocal else False for y in Y])
            if isinstance(idx, bool):
                idx = Y == vocal
            color = palette[i % len(palette)]
            pts = X_proj[idx]
            if len(pts) > 0:
                # Sombra en piso (plano XY en z_min)
                ax.scatter(pts[:, 0], pts[:, 1], zs=z_min, zdir='z',
                           color=color, s=25, alpha=0.20, edgecolors='none', depthshade=False, zorder=1)
                # Sombra en pared trasera (plano XZ en y_max)
                ax.scatter(pts[:, 0], pts[:, 2], zs=y_max, zdir='y',
                           color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)
                # Sombra en pared lateral (plano YZ en x_min)
                ax.scatter(pts[:, 1], pts[:, 2], zs=x_min, zdir='x',
                           color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    if connect_points and len(plot_points) > 1:
        plot_points.append(plot_points[0])
        pts = np.array(plot_points)
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color='gray', linestyle='--', alpha=0.8, linewidth=1.5)
            
    ax.set_title(title, color='black', fontsize=28, fontweight='bold', pad=15)
    ax.tick_params(labelsize=22)
    
    comp_labels = kwargs.get('axis_labels', ['Componente 1', 'Componente 2', 'Componente 3'] if is_3d else ['Componente 1', 'Componente 2'])
    if is_3d:
        x_label = f'{comp_labels[0]} ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else comp_labels[0]
        y_label = f'{comp_labels[1]} ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else comp_labels[1]
        z_label = f'{comp_labels[2]} ({variance_ratios[2]*100:.1f}%)' if variance_ratios is not None else comp_labels[2]
        ax.set_xlabel(x_label, color='black', fontsize=16, labelpad=10)
        ax.set_ylabel(y_label, color='black', fontsize=16, labelpad=10)
        ax.set_zlabel(z_label, color='black', fontsize=16, labelpad=10)
    else:
        x_label = f'{comp_labels[0]} ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else comp_labels[0]
        y_label = f'{comp_labels[1]} ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else comp_labels[1]
        ax.set_xlabel(x_label, color='black', fontsize=24)
        ax.set_ylabel(y_label, color='black', fontsize=24)
        
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
        
    if not kwargs.get('ocultar_leyenda', False):
        legend = ax.legend(frameon=True, facecolor='white', edgecolor='gray', labelcolor='black', fontsize=24)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

def plot_analisis_errores_3d_proyecciones_2d(X_proj, Y, title, output_path, variance_ratios=None, algoritmo="GMM", **kwargs):
    plt.style.use('default')
    fig = plt.figure(figsize=(26, 8.5), facecolor='white')
    
    vocales = sorted(list(set(Y)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    import matplotlib.colors as mc
    import colorsys
    cmap_bg = mc.ListedColormap([palette[i] for i in range(len(vocales))])
    
    axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1)
        ax.set_facecolor('white')
        ax.grid(color='#d3d3d3', linestyle='-', linewidth=0.5)
        axes.append(ax)
        
    comp_labels = kwargs.get('axis_labels', ['Componente 1', 'Componente 2', 'Componente 3'])
    projections = [(0, 1), (0, 2), (1, 2)]
    labels = [
        (f'{comp_labels[0]} ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else comp_labels[0],
         f'{comp_labels[1]} ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else comp_labels[1]),
        (f'{comp_labels[0]} ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else comp_labels[0],
         f'{comp_labels[2]} ({variance_ratios[2]*100:.1f}%)' if variance_ratios is not None else comp_labels[2]),
        (f'{comp_labels[1]} ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else comp_labels[1],
         f'{comp_labels[2]} ({variance_ratios[2]*100:.1f}%)' if variance_ratios is not None else comp_labels[2])
    ]
    
    for ax, proj, (xl, yl) in zip(axes, projections, labels):
        X_2d = X_proj[:, [proj[0], proj[1]]]
        
        # Entrenar un modelo 2D para esta cara para dibujar el fondo
        n_clases = len(vocales)
        if algoritmo == "GMM":
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42)
        else:
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clases, random_state=42, n_init=10)
        
        model.fit(X_2d)
        
        x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
        y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        Z = model.predict(grid)
        
        # Mapeo de clusters a vocales para alinear colores con Y_true (Húngaro)
        y_pred = model.predict(X_2d)
        y_true_int = np.array([vocales.index(v) for v in Y])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_int, y_pred)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-cm)
        cluster_to_vocal = {col: row for row, col in zip(row_ind, col_ind)}
        
        Z_mapped = np.vectorize(cluster_to_vocal.get)(Z)
        Z_mapped = Z_mapped.reshape(xx.shape)
        
        ax.pcolormesh(xx, yy, Z_mapped, cmap=cmap_bg, alpha=0.2, zorder=0)
        ax.contour(xx, yy, Z_mapped, colors='k', linewidths=0.5, alpha=0.5, zorder=1)
        
        for i, vocal in enumerate(vocales):
            idx = np.where(np.array(Y) == vocal)[0]
            color = palette[i]
            ax.scatter(X_2d[idx, 0], X_2d[idx, 1], label=vocal, color=color, alpha=1.0, s=80, edgecolors='black', linewidth=0.5, zorder=4)
            
        ax.set_xlabel(xl, color='black', fontsize=16)
        ax.set_ylabel(yl, color='black', fontsize=16)
            
    fig.suptitle(title, color='black', fontsize=24, fontweight='bold', y=0.98)
    
    if not kwargs.get('ocultar_leyenda', False):
        axes[2].legend(frameon=True, facecolor='white', edgecolor='gray', labelcolor='black', fontsize=15, loc='upper right')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()


def plot_scatter_3d_multi_angle(X_proj, Y, title, output_path, variance_ratios=None, **kwargs):
    plt.style.use('default')
    fig = plt.figure(figsize=(26, 8.5), facecolor='white')
    
    vocales = sorted(list(set(Y)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    axes = []
    for i in range(3):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_facecolor('white')
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.grid(color='#d3d3d3', linestyle='-', linewidth=0.5)
        axes.append(ax)
        
    for i, vocal in enumerate(vocales):
        idx = np.where(np.array(Y) == vocal)[0]
        color = palette[i % len(palette)]
        for ax in axes:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], label=vocal, color=color, alpha=1.0, s=60, edgecolors='black', linewidth=0.3, depthshade=False)
            
    # Proyecciones 2D en los planos de cada vista
    for ax in axes:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        z_min, z_max = ax.get_zlim()
        for i, vocal in enumerate(vocales):
            idx = np.where(np.array(Y) == vocal)[0]
            color = palette[i % len(palette)]
            pts = X_proj[idx]
            if len(pts) > 0:
                # Sombra en piso (plano XY en z_min)
                ax.scatter(pts[:, 0], pts[:, 1], zs=z_min, zdir='z',
                           color=color, s=25, alpha=0.20, edgecolors='none', depthshade=False, zorder=1)
                # Sombra en pared trasera (plano XZ en y_max)
                ax.scatter(pts[:, 0], pts[:, 2], zs=y_max, zdir='y',
                           color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)
                # Sombra en pared lateral (plano YZ en x_min)
                ax.scatter(pts[:, 1], pts[:, 2], zs=x_min, zdir='x',
                           color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    fig.suptitle(title, color='black', fontsize=28, fontweight='bold', y=0.98)
    
    comp_labels = kwargs.get('axis_labels', ['PC1', 'PC2', 'PC3'])
    for ax in axes:
        x_label = f'{comp_labels[0]} ({variance_ratios[0]*100:.1f}%)' if variance_ratios is not None else comp_labels[0]
        y_label = f'{comp_labels[1]} ({variance_ratios[1]*100:.1f}%)' if variance_ratios is not None else comp_labels[1]
        z_label = f'{comp_labels[2]} ({variance_ratios[2]*100:.1f}%)' if variance_ratios is not None else comp_labels[2]
        ax.set_xlabel(x_label, color='black', fontsize=16, labelpad=12)
        ax.set_ylabel(y_label, color='black', fontsize=16, labelpad=12)
        ax.set_zlabel(z_label, color='black', fontsize=16, labelpad=12)
    
    # Set angles
    axes[0].view_init(elev=20, azim=-60)  # Default Frontal
    axes[1].view_init(elev=20, azim=30)   # Rotated +90 degrees
    axes[2].view_init(elev=20, azim=120)  # Rotated +180 degrees
    
    axes[0].set_title("Vista Frontal", fontsize=19, fontweight='bold', pad=25)
    axes[1].set_title("Vista Lateral (+90°)", fontsize=19, fontweight='bold', pad=25)
    axes[2].set_title("Vista Posterior (+180°)", fontsize=19, fontweight='bold', pad=25)
    
    # Legend
    if not kwargs.get('ocultar_leyenda', False):
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=len(vocales), frameon=True, facecolor='white', edgecolor='gray', title="Vocales", title_fontsize=24, fontsize=18)
    
    plt.subplots_adjust(hspace=0.1)
    plt.tight_layout(rect=[0, 0, 1, 0.92], h_pad=2.0) 
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()


def plot_analisis_errores_2d(X, Y, Tomas, title, output_path, variance_ratios=None, algoritmo="K-Means", is_umap=False, ocultar_leyenda=False, estilo_visual="Elipses", **kwargs):
    from scipy.optimize import linear_sum_assignment
    import matplotlib.patches as patches
    import matplotlib.patheffects as pe
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    ax = fig.add_subplot(111)
    
    ax.grid(color='#f0f0f0', linestyle='-', linewidth=1.5, alpha=0.8, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    vocales_unicas = sorted(list(set(Y)))
    n_clases = len(vocales_unicas)
    import collections
    print(f"\nDistribución real de vocales en los datos ({len(Y)} total):")
    for k, v in sorted(collections.Counter(Y).items()):
        print(f"  {k}: {v} muestras")

    if algoritmo == "GMM":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42, n_init=100, max_iter=500)
        y_pred_kmeans = model.fit_predict(X)
        centroids = model.means_
    else:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clases, random_state=42, n_init=100)
        y_pred_kmeans = model.fit_predict(X)
        centroids = model.cluster_centers_
    
    y_true_int = np.array([vocales_unicas.index(v) for v in Y])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_int, y_pred_kmeans)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    cluster_to_vocal_idx = {kmeans_idx: real_idx for real_idx, kmeans_idx in zip(row_ind, col_ind)}
    y_pred_mapped_idx = np.array([cluster_to_vocal_idx.get(c, 0) for c in y_pred_kmeans])
    
    acc_global = np.sum(y_pred_mapped_idx == y_true_int) / len(y_true_int) * 100
    
    export_cluster_data(output_path, algoritmo, model, centroids, vocales_unicas, cluster_to_vocal_idx, is_3d=False)
    
    import matplotlib.colors as mc
    import colorsys

    def adjust_lightness(color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

    palette = sns.color_palette("Set1", n_colors=n_clases)
    
    # --- RENDERIZADO AVANZADO (Fondo) ---
    if algoritmo == "GMM" and estilo_visual in ["Sombreado", "Fronteras"]:
        x_min, x_max = X[:, 0].min() - (X[:, 0].max() - X[:, 0].min())*0.1, X[:, 0].max() + (X[:, 0].max() - X[:, 0].min())*0.1
        y_min, y_max = X[:, 1].min() - (X[:, 1].max() - X[:, 1].min())*0.1, X[:, 1].max() + (X[:, 1].max() - X[:, 1].min())*0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 800), np.linspace(y_min, y_max, 800))
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        if estilo_visual == "Fronteras":
            Z_cluster = model.predict(grid)
            Z_mapped = np.array([cluster_to_vocal_idx.get(c, 0) for c in Z_cluster])
            Z_mapped = Z_mapped.reshape(xx.shape)
            cmap = mc.ListedColormap([palette[i] for i in range(n_clases)])
            ax.pcolormesh(xx, yy, Z_mapped, cmap=cmap, alpha=0.3, zorder=0, shading='auto', vmin=0, vmax=n_clases-1)
            ax.contour(xx, yy, Z_mapped, levels=np.arange(0.5, n_clases - 0.5, 1), colors='k', linewidths=0.5, alpha=0.5, zorder=0, antialiased=True)
            
        elif estilo_visual == "Sombreado":
            from scipy.stats import multivariate_normal
            for cluster_id in range(n_clases):
                real_idx = cluster_to_vocal_idx.get(cluster_id, 0)
                color = palette[real_idx]
                cov = model.covariances_[cluster_id]
                mean = model.means_[cluster_id]
                rv = multivariate_normal(mean, cov)
                Z = rv.pdf(grid).reshape(xx.shape)
                if Z.max() > 0:
                    Z = Z / Z.max()
                cmap_custom = mc.LinearSegmentedColormap.from_list("", [(1,1,1,0), mc.to_rgba(color, 0.4)])
                ax.contourf(xx, yy, Z, levels=12, cmap=cmap_custom, zorder=0)
    
    print("\n--- DETALLE DE MEDICIONES MAL CLASIFICADAS (PCA 2D) ---")
    errores_encontrados = False
    legend_elements = []
    
    for real_idx, vocal in enumerate(vocales_unicas):
        color = palette[real_idx]
        
        idx_true = (y_true_int == real_idx)
        X_true = X[idx_true]
        # Determinar si el punto fue clasificado correctamente
        Y_pred_for_this = y_pred_mapped_idx[idx_true]
        idx_error = (Y_pred_for_this != real_idx)
        idx_correct = ~idx_error

        # Puntos Correctos (Círculos sólidos del color original)
        if np.any(idx_correct):
            ax.scatter(X_true[idx_correct, 0], X_true[idx_correct, 1], 
                       c=[color], marker='o', s=80, edgecolors='black', linewidth=0.5, alpha=1.0, zorder=4)
        
        # Puntos Incorrectos (ya no se sombrean)
        if np.any(idx_error):
            ax.scatter(X_true[idx_error, 0], X_true[idx_error, 1], 
                       c=[color], marker='o', s=80, edgecolors='black', linewidth=0.5, alpha=1.0, zorder=3)
        
        # Centroide y Elipse de confianza (asociados a la vocal predecida por el cluster)
        cluster_id = col_ind[real_idx]
        centroid = centroids[cluster_id]
        
        dark_color = adjust_lightness(color, 0.65)
        ax.scatter(centroid[0], centroid[1], c=[dark_color], marker='D', s=250, edgecolors='black', linewidth=1.5, zorder=5,
                   path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])
        
        if estilo_visual == "Elipses":
            puntos_en_cluster = X[y_pred_kmeans == cluster_id]
            if len(puntos_en_cluster) > 2:
                cov = np.cov(puntos_en_cluster.T)
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
                width, height = 2 * np.sqrt(eigenvalues) * 2 
                ellipse = patches.Ellipse(xy=centroid, width=width, height=height, angle=angle, 
                                          edgecolor=color, facecolor=color, alpha=0.15, linewidth=2, linestyle='-', zorder=1)
                ax.add_patch(ellipse)
            
        # Loggear errores en consola (ya no se dibujan líneas ni cruces)
        Y_pred_for_this = y_pred_mapped_idx[idx_true]
        idx_error = (Y_pred_for_this != real_idx)
        
        for point_idx, is_error in enumerate(idx_error):
            if is_error:
                global_idx = np.where(idx_true)[0][point_idx]
                toma_fallida = Tomas[global_idx]
                wrong_cluster_id = y_pred_kmeans[global_idx]
                vocal_predicha = vocales_unicas[cluster_to_vocal_idx[wrong_cluster_id]]
                
            
    # ax.set_title(title, color='black', fontsize=24, fontweight='bold', pad=15)
    ax.tick_params(labelsize=18)
    comp_labels = kwargs.get('axis_labels', ['UMAP1', 'UMAP2'] if is_umap else ['PC1', 'PC2'])
    x_label = comp_labels[0]
    y_label = comp_labels[1]
    ax.set_xlabel(x_label, color='black', fontsize=22, fontweight='bold', labelpad=10)
    ax.set_ylabel(y_label, color='black', fontsize=22, fontweight='bold', labelpad=10)
    
    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X, y_pred_kmeans)
    
    for i, v in enumerate(vocales_unicas):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], 
                                          markersize=12, markeredgecolor='black', markeredgewidth=0.5, label=f'{v}'))
    
    # Explicación de formas y tonos usando un color neutro (Gris)
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=12, markeredgecolor='black', label='Centroide'))
    
    if estilo_visual == "Elipses":
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.15, markersize=16, markeredgecolor='gray', markeredgewidth=2, label=r'Elipse (2$\sigma$)'))
    elif estilo_visual == "Sombreado":
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', alpha=0.4, markersize=16, markeredgecolor='none', label='Densidad GMM'))
    elif estilo_visual == "Fronteras":
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', alpha=0.4, markersize=16, markeredgecolor='black', markeredgewidth=1, label='Fronteras de Decisión'))
        
    if not ocultar_leyenda:
        legend = ax.legend(handles=legend_elements, loc='best', 
                           title="Vocal", borderaxespad=0., fontsize=16, title_fontsize=20,
                           frameon=True, edgecolor='#dddddd', facecolor='white')
    
    plt.tight_layout()
    if output_path.endswith('.pdf'):
        plt.savefig(output_path[:-4] + '.png', dpi=300, facecolor='white', bbox_inches='tight')
    elif output_path.endswith('.png'):
        plt.savefig(output_path[:-4] + '.pdf', dpi=300, facecolor='white', bbox_inches='tight')
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_analisis_errores_3d(X, Y, Tomas, title, output_path, variance_ratios=None, algoritmo="K-Means", is_umap=False, ocultar_leyenda=False, **kwargs):
    from scipy.optimize import linear_sum_assignment
    import matplotlib.patheffects as pe
    import matplotlib.patches as patches
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    fig = plt.figure(figsize=(12, 9), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('white')
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.grid(color='#f0f0f0', linestyle='-', linewidth=1.5, alpha=0.8)
    
    import matplotlib.colors as mc
    import colorsys

    def adjust_lightness(color, amount=0.5):
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    
    vocales_unicas = sorted(list(set(Y)))
    n_clases = len(vocales_unicas)
    
    if algoritmo == "GMM":
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clases, covariance_type='full', random_state=42, n_init=100, max_iter=500)
        y_pred_kmeans = model.fit_predict(X)
        centroids = model.means_
    else:
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=n_clases, random_state=42, n_init=100)
        y_pred_kmeans = model.fit_predict(X)
        centroids = model.cluster_centers_
    
    y_true_int = np.array([vocales_unicas.index(v) for v in Y])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_int, y_pred_kmeans)
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    cluster_to_vocal_idx = {kmeans_idx: real_idx for real_idx, kmeans_idx in zip(row_ind, col_ind)}
    y_pred_mapped_idx = np.array([cluster_to_vocal_idx.get(c, 0) for c in y_pred_kmeans])
    acc_global = np.sum(y_pred_mapped_idx == y_true_int) / len(y_true_int) * 100
    
    palette = sns.color_palette("Set1", n_colors=n_clases)
    
    print("\n--- DETALLE DE MEDICIONES MAL CLASIFICADAS (PCA 3D) ---")
    errores_encontrados = False
    legend_elements = []
    
    for real_idx, vocal in enumerate(vocales_unicas):
        color = palette[real_idx]
        
        idx_true = (y_true_int == real_idx)
        X_true = X[idx_true]
        
        Y_pred_for_this = y_pred_mapped_idx[idx_true]
        idx_error = (Y_pred_for_this != real_idx)
        
        # Puntos Reales Correctos
        if np.any(~idx_error):
            idx_correct = ~idx_error
            ax.scatter(X_true[idx_correct, 0], X_true[idx_correct, 1], X_true[idx_correct, 2],
                       c=[color], marker='o', s=80, edgecolors='white', linewidth=1.2, alpha=1.0, zorder=3)
        
        # Centroide (Predicción)
        cluster_id = col_ind[real_idx]
        centroid = centroids[cluster_id]
        ax.scatter(centroid[0], centroid[1], centroid[2], 
                   c=[color], marker='D', s=250, edgecolors='black', linewidth=1.5, zorder=5,
                   path_effects=[pe.withStroke(linewidth=4, foreground="white", alpha=0.8)])
                   
        # Elipsoide 3D (3 Desviaciones Estándar)
        try:
            cov = np.cov(X_true, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            idx_sort = evals.argsort()[::-1]
            evals = evals[idx_sort]
            evecs = evecs[:, idx_sort]
            radii = np.sqrt(evals) * 3
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = np.outer(np.cos(u), np.sin(v))
            y_sphere = np.outer(np.sin(u), np.sin(v))
            z_sphere = np.outer(np.ones_like(u), np.cos(v))
            x_sphere *= radii[0]
            y_sphere *= radii[1]
            z_sphere *= radii[2]
            points_sphere = np.vstack((x_sphere.flatten(), y_sphere.flatten(), z_sphere.flatten()))
            points_rot = evecs @ points_sphere
            x_ell = points_rot[0, :].reshape(x_sphere.shape) + centroid[0]
            y_ell = points_rot[1, :].reshape(y_sphere.shape) + centroid[1]
            z_ell = points_rot[2, :].reshape(z_sphere.shape) + centroid[2]
            ax.plot_wireframe(x_ell, y_ell, z_ell, color=color, alpha=0.15, linewidth=0.6, zorder=2)
        except Exception as e:
            pass
                   
        Y_pred_for_this = y_pred_mapped_idx[idx_true]
        idx_error = (Y_pred_for_this != real_idx)
        
        # Puntos incorrectos
        if np.any(idx_error):
            ax.scatter(X_true[idx_error, 0], X_true[idx_error, 1], X_true[idx_error, 2],
                       c=[color], marker='o', s=80, edgecolors='white', linewidth=0.5, alpha=1.0, zorder=2)
        
        for point_idx, is_error in enumerate(idx_error):
            if is_error:
                global_idx = np.where(idx_true)[0][point_idx]
                toma_fallida = Tomas[global_idx]
                wrong_cluster_id = y_pred_kmeans[global_idx]
                vocal_predicha = vocales_unicas[cluster_to_vocal_idx[wrong_cluster_id]]
                
                print(f"[ERROR 3D] Toma '{toma_fallida}': Pronuncio '{vocal}' pero cayo cerca del centroide de '{vocal_predicha}'.")
                errores_encontrados = True
                
    # Proyecciones 2D en paredes y piso (sombras, líneas de caída, elipses 2D proyectadas)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    for real_idx, vocal in enumerate(vocales_unicas):
        color = palette[real_idx]
        idx_true = (y_true_int == real_idx)
        X_true = X[idx_true]
        cluster_id = col_ind[real_idx]
        centroid = centroids[cluster_id]

        if len(X_true) > 0:
            # Sombra en piso (plano XY en z_min)
            ax.scatter(X_true[:, 0], X_true[:, 1], zs=z_min, zdir='z',
                       color=color, s=25, alpha=0.20, edgecolors='none', depthshade=False, zorder=1)
            # Sombra en pared trasera (plano XZ en y_max)
            ax.scatter(X_true[:, 0], X_true[:, 2], zs=y_max, zdir='y',
                       color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)
            # Sombra en pared lateral (plano YZ en x_min)
            ax.scatter(X_true[:, 1], X_true[:, 2], zs=x_min, zdir='x',
                       color=color, s=20, alpha=0.15, edgecolors='none', depthshade=False, zorder=1)

            # Línea de caída vertical desde el centroide al piso
            ax.plot([centroid[0], centroid[0]], [centroid[1], centroid[1]], [z_min, centroid[2]],
                    color=color, linestyle=':', linewidth=1.2, alpha=0.7, zorder=2)

            # Centroide proyectado en piso
            ax.scatter([centroid[0]], [centroid[1]], zs=z_min, zdir='z',
                       marker='D', s=80, color=color, alpha=0.4, edgecolors='black', linewidth=0.8, zorder=2)

            # Elipse 2D proyectada en el piso (XY en z_min)
            try:
                cov_2d = np.cov(X_true[:, :2], rowvar=False)
                evals_2d, evecs_2d = np.linalg.eigh(cov_2d)
                idx_sort_2d = evals_2d.argsort()[::-1]
                evals_2d = evals_2d[idx_sort_2d]
                evecs_2d = evecs_2d[:, idx_sort_2d]
                radii_2d = np.sqrt(np.maximum(evals_2d, 1e-9)) * 3
                theta = np.linspace(0, 2 * np.pi, 60)
                ell_pts = np.array([radii_2d[0] * np.cos(theta), radii_2d[1] * np.sin(theta)])
                ell_rot = evecs_2d @ ell_pts
                x_ell_floor = ell_rot[0, :] + centroid[0]
                y_ell_floor = ell_rot[1, :] + centroid[1]
                ax.plot(x_ell_floor, y_ell_floor, zs=z_min, zdir='z',
                        color=color, alpha=0.35, linestyle='--', linewidth=1.0, zorder=2)
            except Exception:
                pass

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    if not errores_encontrados:
        print("[OK] Clasificacion 3D perfecta: Ningun error en esta prueba.")
    print("-------------------------------------------------------")
            
    # ax.set_title(title, color='#2c3e50', fontsize=22, fontweight='900', fontfamily='sans-serif', pad=20)
    ax.set_xlabel('PC1', color='#34495e', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('PC2', color='#34495e', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_zlabel('PC3', color='#34495e', fontsize=15, fontweight='bold', labelpad=10)
    
    for i, v in enumerate(vocales_unicas):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[i], 
                                          markersize=9, markeredgecolor='white', label=f'{v}'))
    
    # Explicación de formas y tonos usando un color neutro (Gris)
    legend_elements.append(plt.Line2D([0], [0], marker='', color='w', label='')) # Espaciador
    legend_elements.append(plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=9, markeredgecolor='black', label='Centroide'))
    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.15, markersize=14, markeredgecolor='gray', markeredgewidth=2, label=r'Elipse (3$\sigma$)'))
        
    if not ocultar_leyenda:
        legend = ax.legend(handles=legend_elements, loc='best', 
                           title="Vocal", borderaxespad=0., fontsize=15, title_fontsize=20,
                           frameon=True, edgecolor='#dddddd', facecolor='white')
              
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()


def plot_recognition_rates_bar_chart(accuracies, labels, title, output_path, acc_vocales_list=None, vocales=None, **kwargs):
    plt.style.use('default')
    
    if acc_vocales_list is not None and vocales is not None:
        fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
        n_groups = len(labels)
        n_vocales = len(vocales)
        total_bars = 1 + n_vocales
        width = 0.8 / total_bars
        x = np.arange(n_groups)
        
        palette = sns.color_palette("Set1", n_colors=n_vocales)
        
        # Global bar (dark gray/blue)
        offset_global = - (total_bars / 2) * width + width / 2
        bars_global = ax.bar(x + offset_global, accuracies, width, color='#34495e', edgecolor='black', linewidth=1.0, label='Global')
        for bar in bars_global:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=13, fontweight='bold', color='black', rotation=90)
            
        for i, vocal in enumerate(vocales):
            vocal_accs = [acc_voc[i] for acc_voc in acc_vocales_list]
            offset = offset_global + (i + 1) * width
            bars_vocal = ax.bar(x + offset, vocal_accs, width, color=palette[i], edgecolor='black', linewidth=1.0, label=f'Vocal {vocal}')
            for bar in bars_vocal:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.0f}", ha='center', va='bottom', fontsize=12, color='black', rotation=90)
                
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, fontweight='bold', color='black')
    if not kwargs.get('ocultar_leyenda', False):
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Precisión", title_fontsize=20, fontsize=15, frameon=True, edgecolor='gray')
    else:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        x = np.arange(len(labels))
        width = 0.6
        bars = ax.bar(x, accuracies, width, color='#4A90E2', edgecolor='black', linewidth=1.2)
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontsize=15, fontweight='bold', color='black')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=16, fontweight='bold', color='black')
        
    ax.set_ylabel('Tasa de reconocimiento [%]', fontsize=17, fontweight='bold', color='black')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=15, color='black')
    ax.set_ylim(0, 119) # Extra space for texts
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='gray')
    
    ax.set_facecolor('white')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
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
    plt.style.use('default')
    fig = plt.figure(figsize=(7, 6), facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    
    # Paleta clásica para publicaciones (YlGnBu o Blues)
    sns.heatmap(dist_matrix, annot=True, cmap="YlGnBu", xticklabels=vocales, yticklabels=vocales, 
                fmt=".2f", cbar_kws={'label': 'Distancia Euclidiana'}, ax=ax,
                linewidths=0.5, linecolor='gray', 
                annot_kws={"size": 11, "color": "black"})
                
    ax.set_title(title, color="black", pad=15, fontsize=17, fontweight='bold')
    ax.tick_params(colors='black', labelsize=11)
    
    # Colorbar tweaks
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.ax.yaxis.set_tick_params(labelcolor='black')
    cbar.set_label('Distancia Euclidiana', color='black', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def calcular_y_guardar_silhouette_por_vocal(X_proj, Y_true, title, filepath):
    from sklearn.metrics import silhouette_samples, silhouette_score
    import pandas as pd
    
    sil_global = silhouette_score(X_proj, Y_true)
    sil_samples = silhouette_samples(X_proj, Y_true)
    
    vocales = sorted(list(set(Y_true)))
    resultados = []
    
    for vocal in vocales:
        idx = np.array(Y_true) == vocal
        sil_mean = np.mean(sil_samples[idx])
        resultados.append({'Vocal': vocal, 'Silhouette Score': f"{sil_mean:.3f}"})
        
    resultados.append({'Vocal': 'Global', 'Silhouette Score': f"{sil_global:.3f}"})
    
    df = pd.DataFrame(resultados)
    
    latex_code = df.to_latex(index=False, column_format='lc', escape=False)
    latex_code = latex_code.replace('\\toprule', '\\toprule\n\\rowcolor{col01}')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(latex_code)
        
    return sil_global

def guardar_tabla_imagen(df, title, filepath, col_width=2.4, row_height=0.65, font_size=11):
    plt.style.use('default')
    max_col_len = max([len(str(c)) for c in df.columns] + [10])
    max_row_len = max([len(str(r)) for r in df.index] + [10])
    col_w = max(col_width, max_col_len * 0.16)
    extra_left = max(0.5, max_row_len * 0.12)
    fig_w = max(7.0, df.shape[1] * col_w + extra_left)
    fig_h = max(2.5, (df.shape[0] + 2) * row_height)
    
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor='white')
    ax.axis('off')
    ax.axis('tight')
    df_str = df.copy()
    num_cols = df_str.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        df_str[num_cols] = df_str[num_cols].round(2)
    table = ax.table(cellText=df_str.values, colLabels=df_str.columns, rowLabels=df_str.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.4)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.visible_edges = 'horizontal' 
        if row == 0 or col == -1:
            cell.set_facecolor('#f2f2f2')
            cell.get_text().set_fontweight('bold')
        else:
            cell.set_facecolor('#ffffff' if row % 2 == 0 else '#fafafa')
        if row == 0 or row == len(df_str):
            cell.set_linewidth(1.5)
    plt.title(title, pad=20, fontsize=font_size+2, fontweight='bold', color='black')
    plt.subplots_adjust(top=0.85, bottom=0.08, left=0.10, right=0.95)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def guardar_matriz_latex(df, title, filepath):
    latex_code = [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        " & \\multicolumn{5}{c}{\\textbf{Predicción}} \\\\",
        " & \\textbf{A} & \\textbf{E} & \\textbf{I} & \\textbf{O} & \\textbf{U} \\\\",
        "\\midrule"
    ]
    for i, row_name in enumerate(df.index):
        line_parts = [f"\\textbf{{Real {str(row_name).replace('Real ', '')}}}"]
        for val in df.iloc[i]:
            intensity = val / 100.0
            r = 1.0 - 0.7 * intensity
            g = 1.0 - 0.7 * intensity
            b = 1.0
            text_color = "\\color{white}" if val >= 50 else ""
            line_parts.append(f"\\cellcolor[rgb]{{{r:.2f},{g:.2f},{b:.2f}}} {text_color} {val:.0f}\\%")
        latex_code.append(" & ".join(line_parts) + " \\\\")
        
    latex_code.extend([
        "\\bottomrule",
        "\\end{tabular}"
    ])
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_code))

def plot_confusion_matrix_heatmap(df_cm, title, filepath):
    plt.style.use('default')
    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor='white')
    
    annot = np.array([[f"{v:.0f}%" for v in row] for row in df_cm.values])
    cmap = sns.light_palette("blue", as_cmap=True)
    sns.heatmap(df_cm, annot=annot, fmt="", cmap=cmap, cbar=False, ax=ax, vmin=0, vmax=100, annot_kws={"fontsize": 12, "fontweight": "normal"})
    
    ax.set_yticklabels([f"Real {str(v).replace('Real ', '')}" for v in df_cm.index], rotation=0, fontweight='bold', fontsize=16)
    ax.set_xticklabels([str(c).replace('Real ', '') for c in df_cm.columns], fontweight='bold', fontsize=16)
    ax.xaxis.tick_top()
    
    ax.set_ylabel("Real", fontsize=15, fontweight='bold', labelpad=10)
    ax.set_xlabel("Predicción", fontsize=15, fontweight='bold', labelpad=15)
    ax.xaxis.set_label_position('top')
    
    for _, spine in ax.spines.items():
        spine.set_visible(False)
    ax.tick_params(left=False, top=False)
    
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(len(df_cm), color='black', linewidth=2)
    ax.plot([0, 1], [1.13, 1.13], transform=ax.transAxes, color='black', linewidth=2, clip_on=False)
    
    plt.title(title, pad=35, fontsize=16, fontweight='bold')
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.18, right=0.92)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def extraer_y_filtrar(mediciones, base_dir, params, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=False, cache_canales_data=None, verbose=True, aplicar_correccion_intersesion=False):
    X, Y, Tomas, SNRs = extraer_features_concatenadas(
        base_dir, mediciones, 
        alpha_ruido=params['alpha_ruido'], 
        gate_ratio_ruido=params.get('gate_ratio_ruido', 0.0),
        smooth_ms=params['smooth_ms'], 
        notch_q=params['notch_q'], 
        target_len=params['target_length'], 
        aplicar_trevisan=aplicar_trevisan, 
        modo_alineacion=modo_alineacion, 
        pre_pct=pre_pct, 
        post_pct=post_pct, 
        canales_features=canales_features,
        ignorar_ventana_cero=ignorar_ventana_cero,
        cache_canales_data=cache_canales_data,
        aplicar_correccion_intersesion=aplicar_correccion_intersesion
    )
    
    if len(X) == 0:
        return [], [], [], []
        
    X = np.array(X)
    from sklearn.ensemble import IsolationForest
    
    X_clean = []
    Y_clean = []
    Tomas_clean = []
    outliers_detectados = 0
    descartados = []
    
    for vocal in np.unique(Y):
        mask = Y == vocal
        X_vocal = X[mask]
        Tomas_vocal = np.array(Tomas)[mask]
        SNRs_vocal = np.array(SNRs)[mask]
        
        valid_snr_mask = SNRs_vocal >= params['snr_threshold']
        for i, is_valid in enumerate(valid_snr_mask):
            if not is_valid:
                outliers_detectados += 1
                descartados.append({"Toma": Tomas_vocal[i], "Vocal": vocal, "SNR": SNRs_vocal[i], "Motivo": "SNR muy bajo"})
                
        X_vocal_snr = X_vocal[valid_snr_mask]
        Tomas_vocal_snr = Tomas_vocal[valid_snr_mask]
        SNRs_vocal_snr = SNRs_vocal[valid_snr_mask]
        
        if len(X_vocal_snr) > 5 and params['outlier_contamination'] > 0:
            iso = IsolationForest(contamination=params['outlier_contamination'], random_state=42)
            preds = iso.fit_predict(X_vocal_snr)
            
            for i, is_inlier in enumerate(preds):
                if is_inlier == 1:
                    X_clean.append(X_vocal_snr[i])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_vocal_snr[i])
                else:
                    outliers_detectados += 1
                    descartados.append({"Toma": Tomas_vocal_snr[i], "Vocal": vocal, "SNR": SNRs_vocal_snr[i], "Motivo": "Outlier estadístico"})
        else:
            for i in range(len(X_vocal_snr)):
                X_clean.append(X_vocal_snr[i])
                Y_clean.append(vocal)
                Tomas_clean.append(Tomas_vocal_snr[i])
                
    if verbose:
        print(f"    Total outliers/SNR removidos: {outliers_detectados}")
        print(f"    Repeticiones finales válidas: {len(X_clean)}")
    
    return np.array(X_clean), np.array(Y_clean), np.array(Tomas_clean), descartados

def ejecutar_procesamiento(
    mediciones, 
    base_dir, 
    params_2d=None, 
    params_3d=None, 
    params_umap=None, 
    proc_pca_2d=False, 
    proc_pca_3d=False, 
    proc_umap_2d=False, 
    proc_umap_3d=False, 
    umap_n_neighbors=15, 
    umap_min_dist=0.1, 
    umap_metric='euclidean',
    aplicar_trevisan=False,
    algoritmo_clustering_pca="K-Means",
    algoritmo_clustering_umap="GMM",
    modo_alineacion="Pico Volumen Micrófono",
    pre_pct=0.4,
    post_pct=0.6,
    canales_features=["canal_0", "canal_1", "canal_2"],
    ocultar_leyenda=False,
    estilo_visual="Elipses",
    ignorar_ventana_cero=False,
    out_dir=None,
    aplicar_correccion_intersesion=True
):
    if out_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(script_dir, "resultados_pca_umap")
    os.makedirs(out_dir, exist_ok=True)
    
    import numpy as np
    from sklearn.decomposition import PCA
    import umap
    from sklearn.metrics import silhouette_score
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"\nIniciando Procesamiento. Canales: {', '.join(canales_features)}")
    
    def aplicar_pesos_canales(X, canales_list, pesos):
        if pesos is None or not isinstance(pesos, (list, tuple)) or len(pesos) == 0:
            return X
        if all(float(w) == 1.0 for w in pesos):
            return X
        X_out = np.array(X, copy=True)
        n_feat = X_out.shape[1]
        n_ch = len(canales_list) if len(canales_list) > 0 else 3
        pts_per_ch = n_feat // n_ch
        for c_idx in range(min(n_ch, len(pesos))):
            w = float(pesos[c_idx])
            if w != 1.0:
                X_out[:, c_idx*pts_per_ch : (c_idx+1)*pts_per_ch] *= w
        return X_out

    resultados_ejecucion = {}
    if proc_pca_2d:
        print("\n=== PROCESANDO PCA 2D ===")
        X_2d, Y_2d, Tomas_2d, desc_2d = extraer_y_filtrar(mediciones, base_dir, params_2d, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero, aplicar_correccion_intersesion=aplicar_correccion_intersesion)
        resultados_ejecucion['X_2d'] = X_2d
        resultados_ejecucion['Y_2d'] = Y_2d
        if len(X_2d) > 0:
            pesos_2d = params_2d.get('pesos_canales', [1.0, 1.0, 1.0])
            X_2d_proc = aplicar_pesos_canales(X_2d, canales_features, pesos_2d)
            
            comp_x_2d_str = str(params_2d.get('comp_x', 'PC1'))
            comp_y_2d_str = str(params_2d.get('comp_y', 'PC2'))
            idx_x_2d = int(comp_x_2d_str.replace('PC', '')) - 1
            idx_y_2d = int(comp_y_2d_str.replace('PC', '')) - 1
            n_comps_2d = max(idx_x_2d, idx_y_2d, 1) + 1
            
            pca_2d = PCA(n_components=n_comps_2d)
            X_pca_full_2d = pca_2d.fit_transform(X_2d_proc)
            X_pca_2d = X_pca_full_2d[:, [idx_x_2d, idx_y_2d]]
            var_ratios_2d = [pca_2d.explained_variance_ratio_[idx_x_2d], pca_2d.explained_variance_ratio_[idx_y_2d]]
            
            plot_scatter(X_pca_2d, Y_2d, f"PCA 2D ({comp_x_2d_str} vs {comp_y_2d_str}) - Vocales EMG", os.path.join(out_dir, "PCA_2D.png"), is_3d=False, variance_ratios=var_ratios_2d, ocultar_leyenda=ocultar_leyenda, axis_labels=[comp_x_2d_str, comp_y_2d_str])
            plot_analisis_errores_2d(X_pca_2d, Y_2d, Tomas_2d, f"Análisis de Aciertos y Errores (PCA 2D: {comp_x_2d_str}-{comp_y_2d_str}) - {algoritmo_clustering_pca}", os.path.join(out_dir, "PCA_2D_Analisis_Errores.png"), variance_ratios=var_ratios_2d, algoritmo=algoritmo_clustering_pca, is_umap=False, ocultar_leyenda=ocultar_leyenda, estilo_visual=estilo_visual, axis_labels=[comp_x_2d_str, comp_y_2d_str])
            acc_pca_2d, acc_vocales_pca_2d, voc_pca_2d, df_cm_pca_2d, mapeo_pca_2d = evaluar_clustering_no_supervisado(X_pca_2d, Y_2d, "PCA 2D", algoritmo_clustering_pca)
            print(f"=> TOTAL Accuracy Clustering No Supervisado (PCA 2D) : {acc_pca_2d:.2f}%")
            plot_confusion_matrix_heatmap(df_cm_pca_2d, "Matriz de Confusión - PCA 2D", os.path.join(out_dir, "heatmap_confusion_pca_2d.png"))
            guardar_matriz_latex(df_cm_pca_2d, "Matriz de Confusión - PCA 2D", os.path.join(out_dir, "matriz_confusion_pca_2d.tex"))
            
            cent_pca_2d, dist_mat_pca_2d, vocales_pca_2d = calcular_centroides_y_distancias(X_pca_2d, Y_2d)
            df_dist_pca_2d = pd.DataFrame(dist_mat_pca_2d, index=vocales_pca_2d, columns=vocales_pca_2d)
            df_dist_pca_2d.to_csv(os.path.join(out_dir, "matriz_distancias_pca_2d.csv"))
            df_cm_pca_2d.to_csv(os.path.join(out_dir, "matriz_confusion_pca_2d.csv"))
            # Add distance matrix latex export
            try:
                latex_dist_2d = df_dist_pca_2d.to_latex(index=True, column_format='l' + 'c'*len(vocales_pca_2d), float_format="%.2f")
                latex_dist_2d = latex_dist_2d.replace('\\toprule', '\\toprule\n\\rowcolor{col01}')
                with open(os.path.join(out_dir, "matriz_distancias_pca_2d.tex"), 'w', encoding='utf-8') as f:
                    f.write(latex_dist_2d)
            except Exception as e:
                print(f"Advertencia exportando matriz_distancias_pca_2d.tex: {e}")
                
            calcular_y_guardar_silhouette_por_vocal(X_pca_2d, Y_2d, "Silhouette Score - PCA 2D", os.path.join(out_dir, "silhouette_pca_2d.tex"))

            if desc_2d:
                pd.DataFrame(desc_2d).to_csv(os.path.join(out_dir, "reporte_mediciones_descartadas_PCA_2D.csv"), index=False)
                
            # Exportar datos proyectados
            df_proy_2d = pd.DataFrame(X_pca_2d, columns=[comp_x_2d_str, comp_y_2d_str])
            df_proy_2d.insert(0, 'Vocal', Y_2d)
            df_proy_2d.insert(1, 'Toma', Tomas_2d)
            df_proy_2d.to_csv(os.path.join(out_dir, "proyecciones_pca_2d.csv"), index=False)

            # Exportar características crudas/filtradas para Visor de Features
            n_features_2d = X_2d.shape[1]
            num_canales_2d = len(canales_features) if len(canales_features) > 0 else 3
            puntos_por_canal_2d = n_features_2d // num_canales_2d
            cols_feat_2d = []
            for ch_name in canales_features:
                ch_idx = ch_name.replace('canal_', '')
                for t in range(puntos_por_canal_2d):
                    cols_feat_2d.append(f"Ch{ch_idx}_T{t}")
            if len(cols_feat_2d) == n_features_2d:
                df_feat_2d = pd.DataFrame(X_2d, columns=cols_feat_2d)
            else:
                df_feat_2d = pd.DataFrame(X_2d, columns=[f"Feature_{i}" for i in range(n_features_2d)])
            df_feat_2d.insert(0, 'Toma', Tomas_2d)
            df_feat_2d.insert(0, 'Vocal', Y_2d)
            csv_feat_path = os.path.join(out_dir, "caracteristicas_exportadas.csv")
            df_feat_2d.to_csv(csv_feat_path, index=False)
            print(f"=> Features exportadas a CSV: {csv_feat_path}")

    if proc_umap_2d:
        print("\n=== PROCESANDO UMAP 2D ===")
        X_2d_u, Y_2d_u, Tomas_2d_u, desc_2d_u = extraer_y_filtrar(mediciones, base_dir, params_umap, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero, aplicar_correccion_intersesion=aplicar_correccion_intersesion)
        if len(X_2d_u) > 0:
            umap_2d = umap.UMAP(n_neighbors=min(umap_n_neighbors, len(X_2d_u)-1), min_dist=umap_min_dist, metric=umap_metric, n_components=2, random_state=42)
            X_umap_2d = umap_2d.fit_transform(X_2d_u)
            plot_scatter(X_umap_2d, Y_2d_u, "UMAP 2D - Vocales EMG", os.path.join(out_dir, "UMAP_2D.png"), is_3d=False)
            plot_analisis_errores_2d(X_umap_2d, Y_2d_u, Tomas_2d_u, f"Análisis de Aciertos y Errores (UMAP 2D) - {algoritmo_clustering_umap}", os.path.join(out_dir, "UMAP_2D_Analisis_Errores.png"), variance_ratios=None, algoritmo=algoritmo_clustering_umap, is_umap=True, estilo_visual=estilo_visual)
            acc_umap_2d, acc_vocales_umap_2d, voc_umap_2d, df_cm_umap_2d, mapeo_umap_2d = evaluar_clustering_no_supervisado(X_umap_2d, Y_2d_u, "UMAP 2D", algoritmo_clustering_umap)
            print(f"=> TOTAL Accuracy Clustering No Supervisado (UMAP 2D): {acc_umap_2d:.2f}%")
            plot_confusion_matrix_heatmap(df_cm_umap_2d, "Matriz de Confusión - UMAP 2D", os.path.join(out_dir, "heatmap_confusion_umap_2d.png"))
            guardar_matriz_latex(df_cm_umap_2d, "Matriz de Confusión - UMAP 2D", os.path.join(out_dir, "matriz_confusion_umap_2d.tex"))
            df_cm_umap_2d.to_csv(os.path.join(out_dir, "matriz_confusion_umap_2d.csv"))
            df_proy_u2d = pd.DataFrame(X_umap_2d, columns=['UMAP1', 'UMAP2'])
            df_proy_u2d.insert(0, 'Vocal', Y_2d_u)
            df_proy_u2d.insert(1, 'Toma', Tomas_2d_u)
            df_proy_u2d.to_csv(os.path.join(out_dir, "proyecciones_umap_2d.csv"), index=False)
            if desc_2d_u:
                pd.DataFrame(desc_2d_u).to_csv(os.path.join(out_dir, "reporte_mediciones_descartadas_UMAP_2D.csv"), index=False)

    if proc_pca_3d:
        print("\n=== PROCESANDO PCA 3D ===")
        X_3d, Y_3d, Tomas_3d, desc_3d = extraer_y_filtrar(mediciones, base_dir, params_3d, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero, aplicar_correccion_intersesion=aplicar_correccion_intersesion)
        resultados_ejecucion['X_3d'] = X_3d
        resultados_ejecucion['Y_3d'] = Y_3d
        if len(X_3d) > 0:
            pesos_3d = params_3d.get('pesos_canales', [1.0, 1.0, 1.0])
            X_3d_proc = aplicar_pesos_canales(X_3d, canales_features, pesos_3d)
            
            comp_x_3d_str = str(params_3d.get('comp_x', 'PC1'))
            comp_y_3d_str = str(params_3d.get('comp_y', 'PC2'))
            comp_z_3d_str = str(params_3d.get('comp_z', 'PC3'))
            idx_x_3d = int(comp_x_3d_str.replace('PC', '')) - 1
            idx_y_3d = int(comp_y_3d_str.replace('PC', '')) - 1
            idx_z_3d = int(comp_z_3d_str.replace('PC', '')) - 1
            n_comps_3d = max(idx_x_3d, idx_y_3d, idx_z_3d, 2) + 1
            
            pca_3d = PCA(n_components=n_comps_3d)
            X_pca_full_3d = pca_3d.fit_transform(X_3d_proc)
            X_pca_3d = X_pca_full_3d[:, [idx_x_3d, idx_y_3d, idx_z_3d]]
            var_ratios_3d = [pca_3d.explained_variance_ratio_[idx_x_3d], pca_3d.explained_variance_ratio_[idx_y_3d], pca_3d.explained_variance_ratio_[idx_z_3d]]
            
            plot_scatter_3d_multi_angle(X_pca_3d, Y_3d, f"PCA 3D ({comp_x_3d_str}, {comp_y_3d_str}, {comp_z_3d_str}) - Vocales EMG", os.path.join(out_dir, "PCA_3D.png"), variance_ratios=var_ratios_3d, axis_labels=[comp_x_3d_str, comp_y_3d_str, comp_z_3d_str])
            plot_analisis_errores_3d_proyecciones_2d(X_pca_3d, Y_3d, f"Análisis de Aciertos y Errores (PCA 3D: {comp_x_3d_str}-{comp_y_3d_str}-{comp_z_3d_str}) - {algoritmo_clustering_pca}", os.path.join(out_dir, "PCA_3D_Analisis_Errores.png"), variance_ratios=var_ratios_3d, algoritmo=algoritmo_clustering_pca, axis_labels=[comp_x_3d_str, comp_y_3d_str, comp_z_3d_str])
            acc_pca_3d, acc_vocales_pca_3d, voc_pca_3d, df_cm_pca_3d, mapeo_pca_3d = evaluar_clustering_no_supervisado(X_pca_3d, Y_3d, "PCA 3D", algoritmo_clustering_pca)
            print(f"=> TOTAL Accuracy Clustering No Supervisado (PCA 3D) : {acc_pca_3d:.2f}%")
            plot_confusion_matrix_heatmap(df_cm_pca_3d, "Matriz de Confusión - PCA 3D", os.path.join(out_dir, "heatmap_confusion_pca_3d.png"))
            guardar_matriz_latex(df_cm_pca_3d, "Matriz de Confusión - PCA 3D", os.path.join(out_dir, "matriz_confusion_pca_3d.tex"))
            df_cm_pca_3d.to_csv(os.path.join(out_dir, "matriz_confusion_pca_3d.csv"))
            
            cent_pca_3d, dist_mat_pca_3d, vocales_pca_3d = calcular_centroides_y_distancias(X_pca_3d, Y_3d)
            df_dist_pca_3d = pd.DataFrame(dist_mat_pca_3d, index=vocales_pca_3d, columns=vocales_pca_3d)
            df_dist_pca_3d.to_csv(os.path.join(out_dir, "matriz_distancias_pca_3d.csv"))
            guardar_tabla_imagen(df_dist_pca_3d, "Matriz de Distancias - PCA 3D", os.path.join(out_dir, "tabla_distancias_pca_3d.png"))
            
            # Add distance matrix latex export
            try:
                latex_dist_3d = df_dist_pca_3d.to_latex(index=True, column_format='l' + 'c'*len(vocales_pca_3d), float_format="%.2f")
                latex_dist_3d = latex_dist_3d.replace('\\toprule', '\\toprule\n\\rowcolor{col01}')
                with open(os.path.join(out_dir, "matriz_distancias_pca_3d.tex"), 'w', encoding='utf-8') as f:
                    f.write(latex_dist_3d)
            except Exception as e:
                print(f"Advertencia exportando matriz_distancias_pca_3d.tex: {e}")
                
            calcular_y_guardar_silhouette_por_vocal(X_pca_3d, Y_3d, "Silhouette Score - PCA 3D", os.path.join(out_dir, "silhouette_pca_3d.tex"))

            if desc_3d:
                pd.DataFrame(desc_3d).to_csv(os.path.join(out_dir, "reporte_mediciones_descartadas_PCA_3D.csv"), index=False)

            # Exportar datos proyectados
            df_proy_3d = pd.DataFrame(X_pca_3d, columns=[comp_x_3d_str, comp_y_3d_str, comp_z_3d_str])
            df_proy_3d.insert(0, 'Vocal', Y_3d)
            df_proy_3d.insert(1, 'Toma', Tomas_3d)
            df_proy_3d.to_csv(os.path.join(out_dir, "proyecciones_pca_3d.csv"), index=False)

            # Exportar características crudas/filtradas (3D)
            n_features_3d = X_3d.shape[1]
            num_canales_3d = len(canales_features) if len(canales_features) > 0 else 3
            puntos_por_canal_3d = n_features_3d // num_canales_3d
            cols_feat_3d = []
            for ch_name in canales_features:
                ch_idx = ch_name.replace('canal_', '')
                for t in range(puntos_por_canal_3d):
                    cols_feat_3d.append(f"Ch{ch_idx}_T{t}")
            if len(cols_feat_3d) == n_features_3d:
                df_feat_3d = pd.DataFrame(X_3d, columns=cols_feat_3d)
            else:
                df_feat_3d = pd.DataFrame(X_3d, columns=[f"Feature_{i}" for i in range(n_features_3d)])
            df_feat_3d.insert(0, 'Toma', Tomas_3d)
            df_feat_3d.insert(0, 'Vocal', Y_3d)
            df_feat_3d.to_csv(os.path.join(out_dir, "caracteristicas_exportadas_3d.csv"), index=False)
            if not proc_pca_2d:
                df_feat_3d.to_csv(os.path.join(out_dir, "caracteristicas_exportadas.csv"), index=False)

    if proc_umap_3d:
        print("\n=== PROCESANDO UMAP 3D ===")
        X_3d_u, Y_3d_u, Tomas_3d_u, desc_3d_u = extraer_y_filtrar(mediciones, base_dir, params_umap, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero, aplicar_correccion_intersesion=aplicar_correccion_intersesion)
        if len(X_3d_u) > 0:
            umap_3d = umap.UMAP(n_neighbors=min(umap_n_neighbors, len(X_3d_u)-1), min_dist=umap_min_dist, metric=umap_metric, n_components=3, random_state=42)
            X_umap_3d = umap_3d.fit_transform(X_3d_u)
            plot_scatter(X_umap_3d, Y_3d_u, "UMAP 3D - Vocales EMG", os.path.join(out_dir, "UMAP_3D.png"), is_3d=True)
            plot_analisis_errores_3d(X_umap_3d, Y_3d_u, Tomas_3d_u, f"Análisis de Aciertos y Errores (UMAP 3D) - {algoritmo_clustering_umap}", os.path.join(out_dir, "UMAP_3D_Analisis_Errores.png"), variance_ratios=None, algoritmo=algoritmo_clustering_umap, is_umap=True)
            acc_umap_3d, acc_vocales_umap_3d, voc_umap_3d, df_cm_umap_3d, mapeo_umap_3d = evaluar_clustering_no_supervisado(X_umap_3d, Y_3d_u, "UMAP 3D", algoritmo_clustering_umap)
            print(f"=> TOTAL Accuracy Clustering No Supervisado (UMAP 3D): {acc_umap_3d:.2f}%")
            plot_confusion_matrix_heatmap(df_cm_umap_3d, "Matriz de Confusión - UMAP 3D", os.path.join(out_dir, "heatmap_confusion_umap_3d.png"))
            guardar_matriz_latex(df_cm_umap_3d, "Matriz de Confusión - UMAP 3D", os.path.join(out_dir, "matriz_confusion_umap_3d.tex"))
            df_cm_umap_3d.to_csv(os.path.join(out_dir, "matriz_confusion_umap_3d.csv"))
            
            cent_umap_3d, dist_mat_umap_3d, vocales_umap_3d = calcular_centroides_y_distancias(X_umap_3d, Y_3d_u)
            df_dist_umap_3d = pd.DataFrame(dist_mat_umap_3d, index=vocales_umap_3d, columns=vocales_umap_3d)
            df_dist_umap_3d.to_csv(os.path.join(out_dir, "matriz_distancias_umap_3d.csv"))
            guardar_tabla_imagen(df_dist_umap_3d, "Matriz de Distancias - UMAP 3D", os.path.join(out_dir, "tabla_distancias_umap_3d.png"))
            df_proy_u3d = pd.DataFrame(X_umap_3d, columns=['UMAP1', 'UMAP2', 'UMAP3'])
            df_proy_u3d.insert(0, 'Vocal', Y_3d_u)
            df_proy_u3d.insert(1, 'Toma', Tomas_3d_u)
            df_proy_u3d.to_csv(os.path.join(out_dir, "proyecciones_umap_3d.csv"), index=False)
            if desc_3d_u:
                pd.DataFrame(desc_3d_u).to_csv(os.path.join(out_dir, "reporte_mediciones_descartadas_UMAP_3D.csv"), index=False)

    import subprocess
    plots_to_open = [
        os.path.join(out_dir, "PCA_2D.png"),
        os.path.join(out_dir, "PCA_3D.png"),
        os.path.join(out_dir, "PCA_2D_Analisis_Errores.png"),
        os.path.join(out_dir, "PCA_3D_Analisis_Errores.png"),
        os.path.join(out_dir, "UMAP_2D.png"),
        os.path.join(out_dir, "UMAP_3D.png")
    ]
    for p in plots_to_open:
        if os.path.exists(p):
            try:
                subprocess.Popen(["xdg-open", p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass

    return resultados_ejecucion

def generar_grafico_distribucion_accuracy(df_hist, output_img_path, n_components=2, algoritmo="GMM"):
    """Genera un gráfico de distribución de densidad y probabilidad del Accuracy durante el barrido de hiperparámetros."""
    if df_hist is None or df_hist.empty:
        return None

    col_acc = 'raw_accuracy' if 'raw_accuracy' in df_hist.columns else 'accuracy_clasificacion'
    acc_vals = df_hist[col_acc].dropna()
    acc_vals = acc_vals[acc_vals >= 0].values

    if len(acc_vals) < 3:
        return None

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    fig.patch.set_facecolor('#0B0C10')
    ax.set_facecolor('#1F2833')

    # Histograma de densidad con estimación KDE
    sns.histplot(
        acc_vals,
        kde=True,
        stat="density",
        color="#66FCF1",
        bins=20,
        edgecolor="#1F2833",
        alpha=0.6,
        ax=ax,
        line_kws={'linewidth': 2.0, 'color': '#45A29E'}
    )

    media = np.mean(acc_vals)
    mediana = np.median(acc_vals)
    maximo = np.max(acc_vals)
    desv = np.std(acc_vals)

    ax.axvline(media, color="#FF5722", linestyle="--", linewidth=1.8, label=f"Media: {media:.1f}%")
    ax.axvline(mediana, color="#FFEB3B", linestyle=":", linewidth=1.8, label=f"Mediana: {mediana:.1f}%")
    ax.axvline(maximo, color="#4CAF50", linestyle="-.", linewidth=2.0, label=f"Máximo: {maximo:.1f}%")

    # Probabilidades acumuladas de clasificacion
    prob_gt_60 = (np.sum(acc_vals >= 60.0) / len(acc_vals)) * 100.0
    prob_gt_70 = (np.sum(acc_vals >= 70.0) / len(acc_vals)) * 100.0
    prob_gt_80 = (np.sum(acc_vals >= 80.0) / len(acc_vals)) * 100.0

    info_text = (
        f"Combinaciones: {len(acc_vals)}\n"
        f"Desv. Estándar: ±{desv:.1f}%\n"
        f"P(Exactitud ≥ 60%): {prob_gt_60:.1f}%\n"
        f"P(Exactitud ≥ 70%): {prob_gt_70:.1f}%\n"
        f"P(Exactitud ≥ 80%): {prob_gt_80:.1f}%"
    )
    ax.text(
        0.03, 0.95, info_text,
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#0B0C10', edgecolor='#45A29E', alpha=0.9)
    )

    ax.set_title(f"Distribución de Probabilidad de Exactitud - Grid Search PCA {n_components}D ({algoritmo})", fontsize=12, fontweight="bold", color="white", pad=12)
    ax.set_xlabel("Porcentaje de Exactitud (%)", fontsize=10, color="white")
    ax.set_ylabel("Densidad de Probabilidad", fontsize=10, color="white")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#45A29E")

    ax.grid(True, linestyle="--", alpha=0.3, color="#C5C6C7")
    legend = ax.legend(loc="upper right", frameon=True, facecolor="#0B0C10", edgecolor="#45A29E")
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.savefig(output_img_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    return output_img_path

def buscar_mejor_configuracion_pca(mediciones, base_dir, params_base, aplicar_trevisan, modo_alineacion, pre_pct, post_pct, canales_features, ignorar_ventana_cero=False, algoritmo_clustering="GMM", smooth_grid=None, target_len_grid=None, alpha_grid=None, notch_q_grid=None, logger=print, n_jobs=-1, n_components=2, aplicar_correccion_intersesion=True, tag_nombre=None, output_dir=None, ejecutar_ganador_con_graficos=True, estilo_visual="Fronteras"):
    if smooth_grid is None:
        smooth_grid = [50, 80, 125, 150, 180, 220, 250]
    if target_len_grid is None:
        target_len_grid = [10, 20, 40, 60, 80, 100]
    if alpha_grid is None:
        alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    if notch_q_grid is None:
        notch_q_grid = [2.0]

    num_workers = os.cpu_count() or 4 if n_jobs in [-1, None] else n_jobs
    combis = [(s, t, a, q) for s in smooth_grid for t in target_len_grid for a in alpha_grid for q in notch_q_grid]
    total_comb = len(combis)

    gate_ruido_val = params_base.get('gate_ratio_ruido', 0.0)
    logger(f"\n[GRID SEARCH PCA] Iniciando proceso (Barrido de {total_comb} combinaciones con {num_workers} hilos de CPU | Notch Q = 2.0 | Gate Ruido = {gate_ruido_val})...")
    logger("  - [Paso 1/2] Cargando audios y aplicando pre-filtros en memoria RAM...")

    cache_canales = {}
    old_stdout_init = sys.stdout
    sys.stdout = io.StringIO()
    try:
        unique_sq = list(set((s, q) for s, t, a, q in combis))
        total_sq = len(unique_sq)
        for i, (s_ms, q_val) in enumerate(unique_sq):
            p_temp = params_base.copy()
            p_temp['smooth_ms'] = s_ms
            p_temp['notch_q'] = q_val
            p_temp['gate_ratio_ruido'] = gate_ruido_val
            extraer_y_filtrar(
                mediciones, base_dir, p_temp, aplicar_trevisan, modo_alineacion,
                pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero,
                cache_canales_data=cache_canales, verbose=False,
                aplicar_correccion_intersesion=aplicar_correccion_intersesion
            )
            pct = ((i + 1) / total_sq) * 100
            sys.stdout = old_stdout_init
            print(f"    -> Progreso Paso 1: {pct:.1f}% ({i+1}/{total_sq} pares Envolvente/Notch)", end='\r', flush=True)
            sys.stdout = io.StringIO()
        sys.stdout = old_stdout_init
        print()
    finally:
        sys.stdout = old_stdout_init

    logger(f"  - [Paso 2/2] Carga en RAM finalizada ({len(cache_canales)} combinaciones). Evaluando combinaciones en paralelo:\n")

    def _eval_comb(c):
        s_ms, t_len, a_ruido, q_val = c
        p = params_base.copy()
        p['smooth_ms'] = s_ms
        p['target_length'] = t_len
        p['alpha_ruido'] = a_ruido
        p['notch_q'] = q_val
        p['gate_ratio_ruido'] = gate_ruido_val

        raw_acc = -1.0
        motivo_descarte = ""
        vocal_acc_dict = {}
        try:
            X, Y, _, _ = extraer_y_filtrar(
                mediciones, base_dir, p, aplicar_trevisan, modo_alineacion,
                pre_pct, post_pct, canales_features, ignorar_ventana_cero=ignorar_ventana_cero,
                cache_canales_data=cache_canales, verbose=False,
                aplicar_correccion_intersesion=aplicar_correccion_intersesion
            )

            if len(X) > 5 and len(np.unique(Y)) > 1:
                pca = PCA(n_components=n_components)
                X_pca = pca.fit_transform(X)
                sil_score = silhouette_score(X_pca, Y)
                acc_score, acc_por_vocal, vocales_unicas, _, _ = evaluar_clustering_no_supervisado(X_pca, Y, f"Grid PCA {n_components}D", algoritmo=algoritmo_clustering, verbose=False)
                raw_acc = acc_score
                vocal_acc_dict = {str(v): round(float(a), 2) for v, a in zip(vocales_unicas, acc_por_vocal)}
                
                # Penalizar configuraciones que maten completamente una vocal (desbalanceadas)
                if min(acc_por_vocal) < 1.0:
                    acc_score = -1.0
                    motivo_descarte = "Vocal al 0%"
            else:
                sil_score = -1.0
                acc_score = -1.0
                motivo_descarte = "Muestras insuficientes"
        except Exception:
            sil_score = -1.0
            acc_score = -1.0
            motivo_descarte = "Error numérico"

        res_dict = {
            "smooth_ms": s_ms,
            "target_length": t_len,
            "alpha_ruido": a_ruido,
            "notch_q": q_val,
            "gate_ratio_ruido": gate_ruido_val,
            "accuracy_clasificacion": acc_score,
            "raw_accuracy": raw_acc,
            "motivo_descarte": motivo_descarte,
            "silhouette_score": sil_score,
            "porcentaje_por_vocal": vocal_acc_dict
        }
        for v_name, v_acc in vocal_acc_dict.items():
            res_dict[f"acc_{v_name}_pct"] = v_acc
        return res_dict

    best_acc = -1.0
    best_sil = -1.0
    best_config = None
    best_vocal_acc = {}
    historial = []
    curr = 0

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_eval_comb, c) for c in combis]
        for f in as_completed(futures):
            curr += 1
            res = f.result()
            historial.append(res)

            s_ms = res["smooth_ms"]
            t_len = res["target_length"]
            a_ruido = res["alpha_ruido"]
            q_val = res["notch_q"]
            acc_score = res["accuracy_clasificacion"]
            raw_acc = res.get("raw_accuracy", -1.0)
            motivo = res.get("motivo_descarte", "")
            sil_score = res["silhouette_score"]
            vocal_acc = res.get("porcentaje_por_vocal", {})

            pct = (curr / total_comb) * 100
            bar_len = 20
            filled = int(bar_len * curr // total_comb)
            bar = '#' * filled + '-' * (bar_len - filled)

            is_best = False
            if (acc_score > best_acc) or (abs(acc_score - best_acc) < 1e-4 and sil_score > best_sil):
                best_acc = acc_score
                best_sil = sil_score
                best_vocal_acc = vocal_acc
                best_config = (s_ms, t_len, a_ruido, q_val)
                is_best = True

            tag = " ¡NUEVO ÓPTIMO!" if is_best else ""
            vocal_str = ", ".join([f"{v}:{acc:.0f}%" for v, acc in vocal_acc.items()]) if vocal_acc else ""
            vocal_info = f" [{vocal_str}]" if vocal_str else ""

            if acc_score >= 0:
                score_str = f"Clasificación: {acc_score:.2f}%{vocal_info}, Silhouette: {sil_score:.4f}"
            elif raw_acc >= 0:
                score_str = f"Clasificación: {raw_acc:.2f}% (Descartado: {motivo}), Silhouette: {sil_score:.4f}"
            else:
                score_str = f"Clasificación: N/A ({motivo or 'Sobre-suavizado/SNR'})"

            logger(f"[{bar}] {pct:5.1f}% ({curr:4d}/{total_comb}) Smooth: {s_ms:3d}ms | Pts: {t_len:3d} | Alpha: {a_ruido:.2f} | Notch Q: {q_val:.1f} -> {score_str}{tag}")

    if best_config is not None:
        vocal_summary = " | ".join([f"Vocal {v}: {acc:.1f}%" for v, acc in best_vocal_acc.items()])
        logger(f"\n[GRID SEARCH PCA] Búsqueda finalizada al 100%.")
        logger(f"  -> Configuración óptima: Smooth={best_config[0]}ms, Pts={best_config[1]}, Alpha={best_config[2]}, Notch Q={best_config[3]}")
        logger(f"  -> Clasificación Global: {best_acc:.2f}% (Silhouette: {best_sil:.4f})")
        if vocal_summary:
            logger(f"  -> Desglose por Vocal: {vocal_summary}")

    from datetime import datetime
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root_dl = os.path.dirname(os.path.abspath(__file__))

    if output_dir is None:
        tag_clean = f"_{tag_nombre}" if tag_nombre else ""
        carpeta_salida = os.path.join(root_dl, "resultados_grid_search", f"grid_{n_components}D{tag_clean}_{timestamp_str}")
    else:
        carpeta_salida = output_dir
    os.makedirs(carpeta_salida, exist_ok=True)

    if historial:
        try:
            df_hist = pd.DataFrame(historial)
            df_hist.sort_values(by=["accuracy_clasificacion", "silhouette_score"], ascending=False, inplace=True)
            
            # Guardado versionado
            csv_versionado = os.path.join(carpeta_salida, "resultados_grid_search.csv")
            df_hist.to_csv(csv_versionado, index=False)
            logger(f"  [+] Tabla de combinaciones guardada en: {csv_versionado}")

            # Guardado legacy en raíz
            csv_legacy = os.path.join(root_dl, "resultados_grid_search_pca.csv")
            df_hist.to_csv(csv_legacy, index=False)

            # Generación de la curva de densidad y probabilidades
            grafico_dist_path = os.path.join(carpeta_salida, "distribucion_accuracy_grid_search.png")
            generar_grafico_distribucion_accuracy(df_hist, grafico_dist_path, n_components=n_components, algoritmo=algoritmo_clustering)
            logger(f"  [+] Gráfico de distribución de probabilidad guardado en: {grafico_dist_path}")
        except Exception as e_hist:
            logger(f"  [!] Error al exportar métricas de Grid Search: {e_hist}")

    if best_config is not None:
        resumen_optimo = {
            "smooth_ms": int(best_config[0]),
            "target_length": int(best_config[1]),
            "alpha_ruido": float(best_config[2]),
            "notch_q": float(best_config[3]) if len(best_config) > 3 else 2.0,
            "gate_ratio_ruido": float(params_base.get("gate_ratio_ruido", 0.0)),
            "snr_threshold": float(params_base.get("snr_threshold", 0.5)),
            "outlier_contamination": float(params_base.get("outlier_contamination", 0.10)),
            "accuracy_clasificacion": float(best_acc),
            "porcentaje_por_vocal": best_vocal_acc,
            "silhouette_score": float(best_sil),
            "n_components": n_components,
            "aplicar_correccion_intersesion": aplicar_correccion_intersesion,
            "fecha_ejecucion": timestamp_str
        }
        try:
            json_versionado = os.path.join(carpeta_salida, "parametros_optimos.json")
            with open(json_versionado, "w", encoding="utf-8") as f:
                json.dump(resumen_optimo, f, indent=4)
            json_legacy = os.path.join(root_dl, "parametros_optimos_pca.json")
            with open(json_legacy, "w", encoding="utf-8") as f:
                json.dump(resumen_optimo, f, indent=4)
        except Exception:
            pass

        # Ejecución automática de la configuración ganadora con generación completa de gráficos
        if ejecutar_ganador_con_graficos:
            try:
                logger(f"\n[+] Ejecutando proyección PCA {n_components}D con la configuración ganadora...")
                params_ganador = params_base.copy()
                params_ganador["smooth_ms"] = int(best_config[0])
                params_ganador["target_length"] = int(best_config[1])
                params_ganador["alpha_ruido"] = float(best_config[2])
                params_ganador["notch_q"] = float(best_config[3]) if len(best_config) > 3 else 2.0
                params_ganador["gate_ratio_ruido"] = float(params_base.get("gate_ratio_ruido", 0.0))
                params_ganador["snr_threshold"] = float(params_base.get("snr_threshold", 0.5))
                params_ganador["outlier_contamination"] = float(params_base.get("outlier_contamination", 0.10))
                if n_components == 2:
                    params_ganador["comp_x"] = "PC1"
                    params_ganador["comp_y"] = "PC2"
                    p2d, p3d = params_ganador, None
                else:
                    params_ganador["comp_x"] = "PC1"
                    params_ganador["comp_y"] = "PC2"
                    params_ganador["comp_z"] = "PC3"
                    p2d, p3d = None, params_ganador

                ejecutar_procesamiento(
                    mediciones=mediciones,
                    base_dir=base_dir,
                    params_2d=p2d,
                    params_3d=p3d,
                    proc_pca_2d=(n_components == 2),
                    proc_pca_3d=(n_components == 3),
                    algoritmo_clustering_pca=algoritmo_clustering,
                    aplicar_trevisan=aplicar_trevisan,
                    modo_alineacion=modo_alineacion,
                    pre_pct=pre_pct,
                    post_pct=post_pct,
                    canales_features=canales_features,
                    ignorar_ventana_cero=ignorar_ventana_cero,
                    out_dir=carpeta_salida,
                    aplicar_correccion_intersesion=aplicar_correccion_intersesion,
                    estilo_visual=estilo_visual
                )
                logger(f"  [✓] Proyección PCA {n_components}D y gráficos generados exitosamente en:\n      {carpeta_salida}")
            except Exception as e_proc:
                logger(f"  [!] No se pudo completar la ejecución del ganador: {e_proc}")

    return best_config, best_acc, best_sil, best_vocal_acc, historial, carpeta_salida

class GeneradorPCAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador PCA/UMAP")
        self.root.geometry("600x850")
        
        if getattr(sys, 'frozen', False):
            root_p = os.path.dirname(os.path.abspath(sys.executable))
            if os.path.basename(root_p) == "_internal":
                root_p = os.path.dirname(root_p)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_p = os.path.dirname(os.path.dirname(script_dir))
        self.base_dir = os.path.join(root_p, "base_de_datos_electrodos")
        
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
        tk.Checkbutton(ch_frame, text="Canal 0 (Milohioideo)", variable=self.var_ch0, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        self.var_ch1 = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="Canal 1 (Depresor)", variable=self.var_ch1, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        self.var_ch2 = tk.BooleanVar(value=True)
        tk.Checkbutton(ch_frame, text="Canal 2 (Orbicular)", variable=self.var_ch2, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=10)
        
        # --- Modos de Procesamiento ---
        modos_frame = tk.LabelFrame(main_frame, text="Procesar Gráficos", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        modos_frame.pack(fill="x", pady=(0,5))
        
        self.var_proc_pca_2d = tk.BooleanVar(value=False)
        tk.Checkbutton(modos_frame, text="PCA 2D", variable=self.var_proc_pca_2d, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)
        
        self.var_proc_pca_3d = tk.BooleanVar(value=False)
        tk.Checkbutton(modos_frame, text="PCA 3D", variable=self.var_proc_pca_3d, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)

        self.var_proc_umap_2d = tk.BooleanVar(value=False)
        tk.Checkbutton(modos_frame, text="UMAP 2D", variable=self.var_proc_umap_2d, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)

        self.var_proc_umap_3d = tk.BooleanVar(value=False)
        tk.Checkbutton(modos_frame, text="UMAP 3D", variable=self.var_proc_umap_3d, bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)
        
        self.var_ocultar_leyenda = tk.BooleanVar(value=False)
        tk.Checkbutton(modos_frame, text="Ocultar Leyenda", variable=self.var_ocultar_leyenda, bg="#1F2833", fg="#FF4C4C", selectcolor="#0B0C10").pack(side="left", padx=5)
        
        # --- Parámetros Configurables 2D ---
        params_2d_frame = tk.LabelFrame(main_frame, text="Parámetros DSP y Limpieza (2D)", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        params_2d_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(params_2d_frame, text="Alpha:", width=5, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=0, padx=2, pady=2)
        self.ent_alpha_2d = tk.Entry(params_2d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_alpha_2d.grid(row=0, column=1, padx=2, pady=2)
        self.ent_alpha_2d.insert(0, "0.5")
        
        tk.Label(params_2d_frame, text="Smooth:", width=8, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=2, padx=2, pady=2)
        self.ent_smooth_2d = tk.Entry(params_2d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth_2d.grid(row=0, column=3, padx=2, pady=2)
        self.ent_smooth_2d.insert(0, "90")
        
        tk.Label(params_2d_frame, text="Pts:", width=4, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=4, padx=2, pady=2)
        self.ent_target_len_2d = tk.Entry(params_2d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_target_len_2d.grid(row=0, column=5, padx=2, pady=2)
        self.ent_target_len_2d.insert(0, "20")
        
        tk.Label(params_2d_frame, text="SNR:", width=4, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=6, padx=2, pady=2)
        self.ent_snr_2d = tk.Entry(params_2d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_snr_2d.grid(row=0, column=7, padx=2, pady=2)
        self.ent_snr_2d.insert(0, "0.5")
        
        tk.Label(params_2d_frame, text="Outliers:", width=7, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=8, padx=2, pady=2)
        self.ent_outliers_2d = tk.Entry(params_2d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_outliers_2d.grid(row=0, column=9, padx=2, pady=2)
        self.ent_outliers_2d.insert(0, "0.10")
        
        # --- Parámetros Configurables 3D ---
        params_3d_frame = tk.LabelFrame(main_frame, text="Parámetros DSP y Limpieza (3D)", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        params_3d_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(params_3d_frame, text="Alpha:", width=5, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=0, padx=2, pady=2)
        self.ent_alpha_3d = tk.Entry(params_3d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_alpha_3d.grid(row=0, column=1, padx=2, pady=2)
        self.ent_alpha_3d.insert(0, "0.5")
        
        tk.Label(params_3d_frame, text="Smooth:", width=8, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=2, padx=2, pady=2)
        self.ent_smooth_3d = tk.Entry(params_3d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth_3d.grid(row=0, column=3, padx=2, pady=2)
        self.ent_smooth_3d.insert(0, "125")
        
        tk.Label(params_3d_frame, text="Pts:", width=4, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=4, padx=2, pady=2)
        self.ent_target_len_3d = tk.Entry(params_3d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_target_len_3d.grid(row=0, column=5, padx=2, pady=2)
        self.ent_target_len_3d.insert(0, "20")
        
        tk.Label(params_3d_frame, text="SNR:", width=4, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=6, padx=2, pady=2)
        self.ent_snr_3d = tk.Entry(params_3d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_snr_3d.grid(row=0, column=7, padx=2, pady=2)
        self.ent_snr_3d.insert(0, "0.5")
        
        tk.Label(params_3d_frame, text="Outliers:", width=7, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=8, padx=2, pady=2)
        self.ent_outliers_3d = tk.Entry(params_3d_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_outliers_3d.grid(row=0, column=9, padx=2, pady=2)
        self.ent_outliers_3d.insert(0, "0.10")
        
        # --- Parámetros UMAP ---
        umap_frame = tk.LabelFrame(main_frame, text="Parámetros UMAP", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        umap_frame.pack(fill="x", pady=(0,5))
        
        # ROW 0: Algoritmo UMAP Specs
        tk.Label(umap_frame, text="Alpha:", width=6, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=0, padx=2, pady=2)
        self.ent_alpha_umap = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_alpha_umap.grid(row=0, column=1, padx=2, pady=2)
        self.ent_alpha_umap.insert(0, "1.0")

        tk.Label(umap_frame, text="n_neighbors:", width=12, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=2, padx=2, pady=2)
        self.ent_umap_nn = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_nn.grid(row=0, column=3, padx=2, pady=2)
        self.ent_umap_nn.insert(0, "10")
        
        tk.Label(umap_frame, text="min_dist:", width=10, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=4, padx=2, pady=2)
        self.ent_umap_md = tk.Entry(umap_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_umap_md.grid(row=0, column=5, padx=2, pady=2)
        self.ent_umap_md.insert(0, "0.1")
        
        tk.Label(umap_frame, text="Métrica:", width=10, anchor="w", bg="#1F2833", fg="white").grid(row=0, column=6, padx=2, pady=2)
        self.combo_metric = ttk.Combobox(umap_frame, values=["euclidean", "cosine", "manhattan", "correlation"], width=12)
        self.combo_metric.grid(row=0, column=7, padx=2, pady=2)
        self.combo_metric.set("cosine")
        
        # ROW 1: DSP Specs for UMAP
        tk.Label(umap_frame, text="Smooth:", width=7, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=0, padx=2, pady=2)
        self.ent_smooth_umap = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth_umap.grid(row=1, column=1, padx=2, pady=2)
        self.ent_smooth_umap.insert(0, "125")
        
        tk.Label(umap_frame, text="Pts:", width=4, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=2, padx=2, pady=2)
        self.ent_target_len_umap = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_target_len_umap.grid(row=1, column=3, padx=2, pady=2)
        self.ent_target_len_umap.insert(0, "20")
        
        tk.Label(umap_frame, text="SNR:", width=5, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=4, padx=2, pady=2)
        self.ent_snr_umap = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_snr_umap.grid(row=1, column=5, padx=2, pady=2)
        self.ent_snr_umap.insert(0, "0.5")
        
        tk.Label(umap_frame, text="Outliers:", width=8, anchor="w", bg="#1F2833", fg="white").grid(row=1, column=6, padx=2, pady=2)
        self.ent_outliers_umap = tk.Entry(umap_frame, width=5, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_outliers_umap.grid(row=1, column=7, padx=2, pady=2)
        self.ent_outliers_umap.insert(0, "0.10")

        self.combo_metric.set("cosine")
        
        # --- Clustering ---
        cluster_frame = tk.LabelFrame(main_frame, text="Algoritmo de Agrupamiento", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        cluster_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(cluster_frame, text="Evaluar PCA:", anchor="w", bg="#1F2833", fg="white").pack(side="left", padx=(0,5))
        self.combo_cluster_pca = ttk.Combobox(cluster_frame, values=["K-Means", "GMM"], width=10)
        self.combo_cluster_pca.pack(side="left", padx=(0, 15))
        self.combo_cluster_pca.set("GMM")
        
        tk.Label(cluster_frame, text="Evaluar UMAP:", anchor="w", bg="#1F2833", fg="white").pack(side="left", padx=(0,5))
        self.combo_cluster_umap = ttk.Combobox(cluster_frame, values=["K-Means", "GMM"], width=10)
        self.combo_cluster_umap.pack(side="left")
        self.combo_cluster_umap.set("K-Means")
        
        # --- Normalización Avanzada (DSP y Trevisan) ---
        trev_frame = tk.LabelFrame(main_frame, text="DSP Avanzado y Normalización", padx=5, pady=5, bg="#1F2833", fg="white")
        trev_frame.pack(fill="x", pady=(0,5))
        
        self.var_aplicar_trevisan = tk.BooleanVar(value=False)
        tk.Checkbutton(trev_frame, text="Aplicar Corrección Trevisan (Mediana Móvil + Detrending)", variable=self.var_aplicar_trevisan, bg="#1F2833", fg="white", selectcolor="#0B0C10").grid(row=0, column=0, columnspan=2, sticky="w")
        
        self.var_ignorar_win0 = tk.BooleanVar(value=False)
        tk.Checkbutton(trev_frame, text="Ignorar Ventana 0 (Artefactos)", variable=self.var_ignorar_win0, bg="#1F2833", fg="white", selectcolor="#0B0C10").grid(row=0, column=2, columnspan=2, sticky="w")
        
        self.var_correccion_intersesion = tk.BooleanVar(value=True)
        tk.Checkbutton(trev_frame, text="Corrección Intersesión por Lote (Calibración de Ganancia)", variable=self.var_correccion_intersesion, bg="#1F2833", fg="#66FCF1", selectcolor="#0B0C10").grid(row=1, column=0, columnspan=4, sticky="w")
        
        tk.Label(trev_frame, text="Pre-Ventana (%):", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=2, column=0, padx=2, pady=2)
        self.ent_pre_pct = tk.Entry(trev_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_pre_pct.grid(row=2, column=1, padx=2, pady=2)
        self.ent_pre_pct.insert(0, "0.4")
        
        tk.Label(trev_frame, text="Post-Ventana (%):", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=2, column=2, padx=2, pady=2)
        self.ent_post_pct = tk.Entry(trev_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_post_pct.grid(row=2, column=3, padx=2, pady=2)
        self.ent_post_pct.insert(0, "0.6")

        tk.Label(trev_frame, text="Gate Ratio Ruido:", width=15, anchor="w", bg="#1F2833", fg="white").grid(row=3, column=0, padx=2, pady=2)
        self.ent_gate = tk.Entry(trev_frame, width=8, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_gate.grid(row=3, column=1, padx=2, pady=2)
        self.ent_gate.insert(0, "0.0")

        self.ent_gate_2d = self.ent_gate
        self.ent_gate_3d = self.ent_gate
        self.ent_gate_umap = self.ent_gate
        
        # --- Alineación Temporal ---
        align_frame = tk.LabelFrame(main_frame, text="Alineación Temporal", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        align_frame.pack(fill="x", pady=(0,5))
        
        tk.Label(align_frame, text="Centrar ventana en:", width=20, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.combo_align = ttk.Combobox(align_frame, values=["Pico Volumen Micrófono", "Pico Derivada Micrófono (Onset)"], width=30)
        self.combo_align.pack(side="left")
        self.combo_align.set("Pico Volumen Micrófono")
        
        # --- Estilo Visual GMM ---
        visual_frame = tk.LabelFrame(main_frame, text="Estilo Visual (Solo GMM)", padx=5, pady=5, bg="#1F2833", fg="#66FCF1")
        visual_frame.pack(fill="x", pady=(0,5))
        
        self.var_estilo_visual = tk.StringVar(value="Elipses")
        tk.Radiobutton(visual_frame, text="Elipses", variable=self.var_estilo_visual, value="Elipses", bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)
        tk.Radiobutton(visual_frame, text="Sombreado", variable=self.var_estilo_visual, value="Sombreado", bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)
        tk.Radiobutton(visual_frame, text="Fronteras", variable=self.var_estilo_visual, value="Fronteras", bg="#1F2833", fg="white", selectcolor="#0B0C10").pack(side="left", padx=5)
        
        # --- Botones Procesar y Grid Search ---
        btn_frame = tk.Frame(main_frame, bg="#0B0C10")
        btn_frame.pack(fill="x", pady=5)

        tag_frame = tk.Frame(btn_frame, bg="#0B0C10")
        tag_frame.pack(fill="x", pady=(0, 4))
        tk.Label(tag_frame, text="Etiqueta / Nombre Set:", width=20, anchor="w", bg="#0B0C10", fg="white").pack(side="left")
        self.ent_tag_grid = tk.Entry(tag_frame, bg="#1F2833", fg="white", insertbackground="white")
        self.ent_tag_grid.pack(side="left", fill="x", expand=True)

        grid_frame = tk.Frame(btn_frame, bg="#0B0C10")
        grid_frame.pack(fill="x", pady=(0, 5))
        
        self.btn_grid_search_2d = tk.Button(
            grid_frame, text="Grid Search (Optimizar 2D)",
            command=lambda: self.ejecutar_grid_search_ui(n_components=2), bg="#66FCF1", fg="#0B0C10",
            font=("Arial", 11, "bold")
        )
        self.btn_grid_search_2d.pack(side="left", fill="x", expand=True, padx=(0, 2))

        self.btn_grid_search_3d = tk.Button(
            grid_frame, text="Grid Search (Optimizar 3D)",
            command=lambda: self.ejecutar_grid_search_ui(n_components=3), bg="#45A29E", fg="#0B0C10",
            font=("Arial", 11, "bold")
        )
        self.btn_grid_search_3d.pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.btn_procesar = tk.Button(
            btn_frame, text="Generar Dataset y Visualizar",
            command=self.iniciar_procesamiento, bg="#45A29E", fg="white",
            font=("Arial", 12, "bold")
        )
        self.btn_procesar.pack(fill="x")
        
        self.cargar_mediciones()

    def ejecutar_grid_search_ui(self, n_components=2):
        seleccionadas = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        if not seleccionadas:
            messagebox.showwarning("Advertencia", "Debe seleccionar al menos una medición para realizar el Grid Search.")
            return

        canales_sel = []
        if self.var_ch0.get(): canales_sel.append("canal_0")
        if self.var_ch1.get(): canales_sel.append("canal_1")
        if self.var_ch2.get(): canales_sel.append("canal_2")

        if not canales_sel:
            messagebox.showwarning("Advertencia", "Debe seleccionar al menos 1 canal muscular.")
            return

        try:
            snr_val = float(self.ent_snr_2d.get()) if n_components == 2 else float(self.ent_snr_3d.get())
            outlier_val = float(self.ent_outliers_2d.get()) if n_components == 2 else float(self.ent_outliers_3d.get())
            gate_val = float(self.ent_gate.get()) if hasattr(self, 'ent_gate') else 0.0
            
            params_base = {
                "alpha_ruido": 0.5,
                "snr_threshold": snr_val,
                "outlier_contamination": outlier_val,
                "gate_ratio_ruido": gate_val,
                "smooth_ms": 90,
                "target_length": 20,
                "notch_q": float(self.ent_notch.get()) if hasattr(self, 'ent_notch') and isinstance(self.ent_notch, tk.Entry) else 2.0
            }
            val_trevisan = self.var_aplicar_trevisan.get()
            val_pre_pct = float(self.ent_pre_pct.get())
            val_post_pct = float(self.ent_post_pct.get())
            val_align = self.combo_align.get()
            ignorar_win0 = self.var_ignorar_win0.get()
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos en la interfaz.")
            return

        tag_val = self.ent_tag_grid.get().strip() if hasattr(self, 'ent_tag_grid') else None

        messagebox.showinfo(
            "Iniciando Grid Search",
            "Comenzando la búsqueda de hiperparámetros óptimos para PCA.\n"
            "Se evaluarán las combinaciones de Envolvente (Smooth), Remuestreo (Puntos) y Alfa de ruido.\n"
            "Por favor aguarde unos instantes..."
        )

        res = buscar_mejor_configuracion_pca(
            seleccionadas, self.base_dir, params_base,
            aplicar_trevisan=val_trevisan, modo_alineacion=val_align,
            pre_pct=val_pre_pct, post_pct=val_post_pct,
            canales_features=canales_sel, ignorar_ventana_cero=ignorar_win0,
            algoritmo_clustering=self.combo_cluster_pca.get() if hasattr(self, 'combo_cluster_pca') else "GMM",
            n_components=n_components,
            aplicar_correccion_intersesion=self.var_correccion_intersesion.get(),
            tag_nombre=tag_val,
            ejecutar_ganador_con_graficos=True,
            estilo_visual=self.var_estilo_visual.get() if hasattr(self, 'var_estilo_visual') else "Fronteras"
        )

        best_config = res[0]
        best_score = res[1]
        carpeta_salida = res[5] if len(res) > 5 else "resultados_grid_search"

        if best_config is None or best_score <= -1.0:
            messagebox.showerror("Error Grid Search", "No se pudo encontrar una configuración válida.")
            return

        if len(best_config) == 4:
            best_smooth, best_pts, best_alpha, best_notch = best_config
        else:
            best_smooth, best_pts, best_alpha = best_config[:3]
            best_notch = 2.0

        if n_components == 2:
            self.ent_alpha_2d.delete(0, tk.END)
            self.ent_alpha_2d.insert(0, str(best_alpha))
            self.ent_smooth_2d.delete(0, tk.END)
            self.ent_smooth_2d.insert(0, str(best_smooth))
            self.ent_target_len_2d.delete(0, tk.END)
            self.ent_target_len_2d.insert(0, str(best_pts))
        elif n_components == 3:
            self.ent_alpha_3d.delete(0, tk.END)
            self.ent_alpha_3d.insert(0, str(best_alpha))
            self.ent_smooth_3d.delete(0, tk.END)
            self.ent_smooth_3d.insert(0, str(best_smooth))
            self.ent_target_len_3d.delete(0, tk.END)
            self.ent_target_len_3d.insert(0, str(best_pts))

        # Actualizar campos UMAP (compartido)
        self.ent_alpha_umap.delete(0, tk.END)
        self.ent_alpha_umap.insert(0, str(best_alpha))
        self.ent_smooth_umap.delete(0, tk.END)
        self.ent_smooth_umap.insert(0, str(best_smooth))
        self.ent_target_len_umap.delete(0, tk.END)
        self.ent_target_len_umap.insert(0, str(best_pts))
        if hasattr(self, 'ent_notch') and isinstance(self.ent_notch, tk.Entry):
            self.ent_notch.delete(0, tk.END)
            self.ent_notch.insert(0, str(best_notch))

        messagebox.showinfo(
            "Grid Search Finalizado",
            f"¡Configuración Óptima Encontrada!\n\n"
            f"- Envolvente (Smooth): {best_smooth} ms\n"
            f"- Puntos Remuestreo: {best_pts}\n"
            f"- Alfa Ruido: {best_alpha}\n"
            f"- Notch Q: {best_notch}\n\n"
            f"Clasificación PCA ({n_components}D): {best_score:.2f}%\n\n"
            f"Se ejecutó el PCA con la configuración ganadora.\n"
            f"Resultados, gráficos y distribución archivados en:\n{carpeta_salida}"
        )

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
            proc_pca_2d = self.var_proc_pca_2d.get()
            proc_pca_3d = self.var_proc_pca_3d.get()
            proc_umap_2d = self.var_proc_umap_2d.get()
            proc_umap_3d = self.var_proc_umap_3d.get()
            
            if not (proc_pca_2d or proc_pca_3d or proc_umap_2d or proc_umap_3d):
                messagebox.showwarning("Advertencia", "Debe seleccionar al menos un procesamiento.")
                return
                
            params_2d = {
                "alpha_ruido": float(self.ent_alpha_2d.get()),
                "gate_ratio_ruido": float(self.ent_gate_2d.get()) if hasattr(self, 'ent_gate_2d') else 0.0,
                "snr_threshold": float(self.ent_snr_2d.get()),
                "outlier_contamination": float(self.ent_outliers_2d.get()),
                "smooth_ms": int(self.ent_smooth_2d.get()),
                "target_length": int(self.ent_target_len_2d.get()),
                "notch_q": float(self.ent_notch.get()) if hasattr(self, 'ent_notch') and isinstance(self.ent_notch, tk.Entry) else 2.0
            }
            
            params_3d = {
                "alpha_ruido": float(self.ent_alpha_3d.get()),
                "gate_ratio_ruido": float(self.ent_gate_3d.get()) if hasattr(self, 'ent_gate_3d') else 0.0,
                "snr_threshold": float(self.ent_snr_3d.get()),
                "outlier_contamination": float(self.ent_outliers_3d.get()),
                "smooth_ms": int(self.ent_smooth_3d.get()),
                "target_length": int(self.ent_target_len_3d.get()),
                "notch_q": float(self.ent_notch.get()) if hasattr(self, 'ent_notch') and isinstance(self.ent_notch, tk.Entry) else 2.0
            }
            
            params_umap = {
                "alpha_ruido": float(self.ent_alpha_umap.get()),
                "gate_ratio_ruido": float(self.ent_gate_umap.get()) if hasattr(self, 'ent_gate_umap') else 0.0,
                "snr_threshold": float(self.ent_snr_umap.get()),
                "outlier_contamination": float(self.ent_outliers_umap.get()),
                "smooth_ms": int(self.ent_smooth_umap.get()),
                "target_length": int(self.ent_target_len_umap.get()),
                "notch_q": float(self.ent_notch.get()) if hasattr(self, 'ent_notch') and isinstance(self.ent_notch, tk.Entry) else 2.0
            }
            
            val_umap_nn = int(self.ent_umap_nn.get())
            val_umap_md = float(self.ent_umap_md.get())
            val_umap_metric = self.combo_metric.get()
            
            val_trevisan = self.var_aplicar_trevisan.get()
            val_pre_pct = float(self.ent_pre_pct.get())
            val_post_pct = float(self.ent_post_pct.get())
            val_algoritmo_pca = self.combo_cluster_pca.get()
            val_algoritmo_umap = self.combo_cluster_umap.get()
            val_align = self.combo_align.get()
            val_estilo_visual = self.var_estilo_visual.get() if hasattr(self, 'var_estilo_visual') else "Elipses"
            
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
            params_2d=params_2d,
            params_3d=params_3d,
            params_umap=params_umap,
            proc_pca_2d=proc_pca_2d,
            proc_pca_3d=proc_pca_3d,
            proc_umap_2d=proc_umap_2d,
            proc_umap_3d=proc_umap_3d,
            umap_n_neighbors=val_umap_nn,
            umap_min_dist=val_umap_md,
            umap_metric=val_umap_metric,
            aplicar_trevisan=val_trevisan,
            algoritmo_clustering_pca=val_algoritmo_pca,
            algoritmo_clustering_umap=val_algoritmo_umap,
            modo_alineacion=val_align,
            pre_pct=val_pre_pct,
            post_pct=val_post_pct,
            canales_features=canales_sel,
            ocultar_leyenda=self.var_ocultar_leyenda.get(),
            estilo_visual=val_estilo_visual,
            ignorar_ventana_cero=self.var_ignorar_win0.get(),
            aplicar_correccion_intersesion=self.var_correccion_intersesion.get()
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
