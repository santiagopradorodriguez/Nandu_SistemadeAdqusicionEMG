import os
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
from sklearn.metrics import silhouette_score, pairwise_distances
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

def extraer_features_concatenadas(base_dir, mediciones, alpha_ruido=1.0, smooth_ms=120, notch_q=2.0, target_length=100, use_manual_exclusions=True, verbose=True):
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
    for rel_path in mediciones:
        med_path = os.path.join(base_dir, rel_path)
        med_name = os.path.basename(rel_path)
        partes = med_name.split('_')
        if len(partes) < 1:
            continue
            
        vocal = partes[0].upper()
        if vocal not in ['A', 'E', 'I', 'O', 'U']:
            continue
        if verbose:
            print(f"\nProcesando: {med_name} (Vocal: {vocal})")
        
        # Leemos las exclusiones directamente del metadata.json del canal_0 si la opción está activada
        excluded_windows = []
        if use_manual_exclusions:
            meta_path_ch0 = os.path.join(med_path, "canal_0", "metadata.json")
            if os.path.exists(meta_path_ch0):
                try:
                    with open(meta_path_ch0, 'r') as f:
                        meta_ch0 = json.load(f)
                        excluded_windows = meta_ch0.get("excluded_windows", [])
                except: pass
                
        canales_data = {}
        
        for ch in canales_procesar:
            carpeta = os.path.join(med_path, ch)
            if not os.path.exists(carpeta):
                if verbose:
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
                        # Por si acaso, si el canal_i tiene exclusions las sumamos
                        if use_manual_exclusions and "excluded_windows" in meta:
                            excluded_windows = list(set(excluded_windows + meta.get("excluded_windows", [])))
            except: pass
            
            # Usar la función original para procesar
            res_final = at.procesar_wavs_promedio(
                carpeta=carpeta, output_root=carpeta,
                bpm=bpm_u, mostrar_recortes=False,
                noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
                excluded_windows=excluded_windows,
                show_interactive_plot=False,
                notch_q_factor=notch_q,
                tipo_envolvente="rms", smooth_ms=smooth_ms,
                verbose=verbose
            )
            
            if res_final:
                fname = list(res_final.keys())[0]
                canales_data[ch] = res_final[fname]
                
        if len(canales_data) < 4:
            if verbose:
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
        
        TARGET_LEN = target_length
        
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
                tensor_sample = np.stack(vector_concatenado) # Shape (3, 1500)
                X.append(tensor_sample)
                Y.append(vocal)
                Tomas.append(f"{med_name}_Win{win_idx}")
                ruido_promedio_total = ruido_acumulado_window / 3.0
                snr = max_supremo / (ruido_promedio_total + 1e-9)
                SNRs.append(snr)
                
    return np.array(X), np.array(Y), np.array(Tomas), np.array(SNRs)

def plot_scatter(X_proj, Y, title, output_path, is_3d=False, variance_ratios=None):
    fig = plt.figure(figsize=(10, 8))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    vocales = sorted(list(set(Y)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
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
        else:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], label=vocal, color=palette[i], alpha=0.9, s=80)
            
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

def ejecutar_procesamiento(mediciones, alpha_ruido=1.0, snr_threshold=0.5, outlier_contamination=0.05, smooth_ms=120, notch_q=2.0, target_length=100, use_manual_exclusions=True, verbose=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "base_de_datos_electrodos")
    out_dir = os.path.join(script_dir, "resultados_pca_umap")
    os.makedirs(out_dir, exist_ok=True)
    
    if verbose:
        print(f"\n2. Extracción y concatenación de características de {len(mediciones)} mediciones...")
        X, Y, Tomas, SNRs = extraer_features_concatenadas(
            base_dir, mediciones, alpha_ruido=alpha_ruido, smooth_ms=smooth_ms, notch_q=notch_q, target_length=target_length, use_manual_exclusions=use_manual_exclusions, verbose=verbose)
    
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
                print(f"  [!] Descartado por SNR muy bajo (<{snr_threshold}): {Tomas_vocal[i]} (Vocal {vocal}) | SNR: {SNRs_vocal[i]:.2f}")
                
        # Quedarse solo con los que pasaron el filtro SNR
        X_vocal_snr = X_vocal[valid_snr_mask]
        Tomas_vocal_snr = Tomas_vocal[valid_snr_mask]
        SNRs_vocal_snr = SNRs_vocal[valid_snr_mask]
        
        # 2. Filtro estadístico (Isolation Forest)
        # Necesitamos un mínimo de muestras para aislar
        if len(X_vocal_snr) > 5 and outlier_contamination > 0:
            # Porcentaje de contaminación esperada (outliers)
            iso = IsolationForest(contamination=outlier_contamination, random_state=42)
            # Aplanar para IsolationForest
            X_flat_snr = X_vocal_snr.reshape(X_vocal_snr.shape[0], -1)
            preds = iso.fit_predict(X_flat_snr)
            
            for i, is_inlier in enumerate(preds):
                if is_inlier == 1:
                    X_clean.append(X_vocal_snr[i])
                    Y_clean.append(vocal)
                    Tomas_clean.append(Tomas_vocal_snr[i])
                else:
                    outliers_detectados += 1
                    print(f"  [!] Outlier estadístico removido: {Tomas_vocal_snr[i]} (Vocal {vocal}) | SNR: {SNRs_vocal_snr[i]:.2f}")
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
    
    base_repo_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    out_dir = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial")
    os.makedirs(out_dir, exist_ok=True)
    
    # ------------------ Exportar Dataset para Autoencoder ------------------
    print("\n4. Exportando matriz de características limpias...")
    
    X_clean_flat = X.reshape(X.shape[0], -1)
    n_features = X_clean_flat.shape[1]
    cols = []
    puntos_por_canal = n_features // 3
    for ch in range(3):
        for t in range(puntos_por_canal):
            cols.append(f"Ch{ch}_T{t}")
            
    csv_export_path = os.path.join(out_dir, "caracteristicas_exportadas.csv")
    df = pd.DataFrame(X_clean_flat, columns=cols)
    df.insert(0, 'Toma', Tomas)
    df.insert(0, 'Vocal', Y)
    df.to_csv(csv_export_path, index=False)
    
    print(f"Dataset LIMPIO exportado exitosamente a: {csv_export_path}")
    print(f"Dimensiones de entrenamiento: {df.shape[0]} instancias x {n_features} variables ({puntos_por_canal} puntos x 3 canales)")
    
    # Exportar dataset sin filtrar como referencia
    csv_sucio_path = os.path.join(out_dir, "caracteristicas_sin_filtrar.csv")
    df_sucio = pd.DataFrame(X_orig.reshape(X_orig.shape[0], -1), columns=cols)
    df_sucio.insert(0, 'Toma', Tomas_orig)
    df_sucio.insert(0, 'Vocal', Y_orig)
    df_sucio.to_csv(csv_sucio_path, index=False)
        
    print(f"\nExtracción finalizada con éxito. Datos listos para entrenar el Autoencoder.")

class GeneradorPCAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador PCA/UMAP")
        self.root.geometry("600x650")
        
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
        
        self.listbox_mediciones.bind("<<ListboxSelect>>", self.on_selection_change)
        
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
        
        # 4. Envolvente (smooth_ms)
        f4 = tk.Frame(params_frame, bg="#1F2833")
        f4.pack(fill="x", pady=2)
        tk.Label(f4, text="Suavizado Envolvente RMS (ms):", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_smooth = tk.Entry(f4, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_smooth.pack(side="left")
        self.ent_smooth.insert(0, "120")
        
        # 5. Notch Q Factor
        f5 = tk.Frame(params_frame, bg="#1F2833")
        f5.pack(fill="x", pady=2)
        tk.Label(f5, text="Filtro Notch Q Factor:", width=35, anchor="w", bg="#1F2833", fg="white").pack(side="left")
        self.ent_notch_q = tk.Entry(f5, width=10, bg="#0B0C10", fg="white", insertbackground="white")
        self.ent_notch_q.pack(side="left")
        self.ent_notch_q.insert(0, "2.0")
        
        # --- Botón Procesar ---
        act_frame = tk.Frame(main_frame, pady=10, bg="#0B0C10")
        act_frame.pack(fill="x", side="bottom")
        
        self.btn_procesar = tk.Button(act_frame, text="EJECUTAR PCA/UMAP...", command=self.iniciar_procesamiento,
                                      state="disabled", bg="#111111", fg="#00FF00")
        self.btn_procesar.pack(fill="x", ipady=5, pady=(0,10))
        
        self.cargar_mediciones()

    def cargar_mediciones(self):
        self.listbox_mediciones.delete(0, tk.END)
        mediciones = procesar_mediciones(self.base_dir)
        for med in mediciones:
            self.listbox_mediciones.insert(tk.END, med)

    def on_selection_change(self, event=None):
        if len(self.listbox_mediciones.curselection()) > 0:
            self.btn_procesar.config(state="normal")
        else:
            self.btn_procesar.config(state="disabled")

    def iniciar_procesamiento(self):
        seleccionadas = [self.listbox_mediciones.get(i) for i in self.listbox_mediciones.curselection()]
        if not seleccionadas:
            messagebox.showwarning("Advertencia", "Debe seleccionar al menos una medición.")
            return
            
        try:
            val_alpha = float(self.ent_alpha.get())
            val_snr = float(self.ent_snr.get())
            val_out = float(self.ent_outliers.get())
            val_smooth = int(self.ent_smooth.get())
            val_notch_q = float(self.ent_notch_q.get())
        except ValueError:
            messagebox.showerror("Error", "Por favor ingrese valores numéricos válidos en los parámetros.")
            return
            
        self.root.destroy()
        ejecutar_procesamiento(
            seleccionadas, 
            alpha_ruido=val_alpha, 
            snr_threshold=val_snr, 
            outlier_contamination=val_out, 
            smooth_ms=val_smooth,
            notch_q=val_notch_q
        )

def main():
    root = tk.Tk()
    app = GeneradorPCAGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
