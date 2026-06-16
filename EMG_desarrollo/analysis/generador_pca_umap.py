import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from scipy.signal import resample, find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances
import umap

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

def extraer_features_concatenadas(base_dir, mediciones):
    """
    Extrae y alinea las ventanas de los canales 0, 1 y 2.
    Devuelve X (matriz de features) e Y (labels/vocales).
    """
    X = []
    Y = []
    
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
            
        print(f"\nProcesando: {med_name} (Vocal: {vocal})")
        
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
            
            # Usar la función original para procesar
            res_final = at.procesar_wavs_promedio(
                carpeta=carpeta, output_root=carpeta,
                bpm=bpm_u, mostrar_recortes=False,
                noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
                excluded_windows=excluded_windows,
                show_interactive_plot=False,
                tipo_envolvente="rms", smooth_ms=250
            )
            
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
        
        TARGET_LEN = 100
        
        for pico in picos_mic:
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
            
            # 1. Extraer los 3 canales y buscar el máximo supremo
            for ch in canales_features:
                env_ch_raw = canales_data[ch]['env_recortada']
                
                if real_cut_end > len(env_ch_raw):
                    valido = False
                    break
                    
                segmento_ch = env_ch_raw[real_cut_start:real_cut_end].copy()
                
                # Restar ruido específico aproximado
                # Como no iteramos sobre 'i', tomamos el ruido promedio o del primer bloque
                ruido_ch = canales_data[ch].get('noise_levels', [0])[0] if len(canales_data[ch].get('noise_levels', [])) > 0 else 0
                segmento_ch = np.maximum(segmento_ch - ruido_ch, 0)
                
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
                vector_concatenado = np.concatenate(vector_concatenado)
                X.append(vector_concatenado)
                Y.append(vocal)
                
    return np.array(X), np.array(Y)

def plot_scatter(X_proj, Y, title, output_path, is_3d=False):
    fig = plt.figure(figsize=(10, 8))
    
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)
        
    vocales = sorted(list(set(Y)))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, vocal in enumerate(vocales):
        idx = Y == vocal
        if is_3d:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], X_proj[idx, 2], label=vocal, c=colors[i%len(colors)], alpha=0.7, s=50)
        else:
            ax.scatter(X_proj[idx, 0], X_proj[idx, 1], label=vocal, c=colors[i%len(colors)], alpha=0.7, s=50)
            
    ax.set_title(title)
    if is_3d:
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        ax.set_zlabel('Componente 3')
    else:
        ax.set_xlabel('Componente 1')
        ax.set_ylabel('Componente 2')
        
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

def ejecutar_procesamiento(mediciones):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_electrodos")
    out_dir = os.path.join(script_dir, "resultados_pca_umap")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n2. Extracción y concatenación de características de {len(mediciones)} mediciones...")
    X, Y = extraer_features_concatenadas(base_dir, mediciones)
    
    if len(X) == 0:
        print("Error: No se obtuvieron datos válidos para procesar.")
        return
        
    print(f"\nDatos recolectados: {X.shape[0]} repeticiones (pulsos), Dimensión de cada feature: {X.shape[1]}")
    X = np.array(X)
    Y = np.array(Y)
    
    # La normalización por pulso ya se aplicó dentro del bucle
    X_scaled = X

    print(f"\nAplicando PCA y UMAP...")
    
    # ------------------ PCA ------------------
    # --- PCA 2D ---
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    
    # --- PCA 3D ---
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    
    var_exp = np.sum(pca_3d.explained_variance_ratio_)
    print(f"Varianza explicada por PCA 3D: {var_exp*100:.2f}%")
    
    plot_scatter(X_pca_2d, Y, "PCA 2D - Vocales EMG", os.path.join(out_dir, "PCA_2D.png"), is_3d=False)
    plot_scatter(X_pca_3d, Y, "PCA 3D - Vocales EMG", os.path.join(out_dir, "PCA_3D.png"), is_3d=True)
    
    # ------------------ UMAP ------------------
    print("\n5. Aplicando UMAP...")
    umap_2d = umap.UMAP(n_components=2, random_state=42)
    X_umap_2d = umap_2d.fit_transform(X_scaled)
    
    umap_3d = umap.UMAP(n_components=3, random_state=42)
    X_umap_3d = umap_3d.fit_transform(X_scaled)
    
    plot_scatter(X_umap_2d, Y, "UMAP 2D - Vocales EMG", os.path.join(out_dir, "UMAP_2D.png"), is_3d=False)
    plot_scatter(X_umap_3d, Y, "UMAP 3D - Vocales EMG", os.path.join(out_dir, "UMAP_3D.png"), is_3d=True)
    
    # ------------------ MÉTRICAS ------------------
    print("\n5. Calculando distancias (Euclidiana) y Silhouette Scores...")
    sil_pca_2d = silhouette_score(X_pca_2d, Y, metric='euclidean')
    sil_pca_3d = silhouette_score(X_pca_3d, Y, metric='euclidean')
    
    sil_umap_2d = silhouette_score(X_umap_2d, Y, metric='euclidean')
    sil_umap_3d = silhouette_score(X_umap_3d, Y, metric='euclidean')
    
    print(f"Silhouette Score (PCA 2D): {sil_pca_2d:.4f}")
    print(f"Silhouette Score (PCA 3D): {sil_pca_3d:.4f}")
    print(f"Silhouette Score (UMAP 2D): {sil_umap_2d:.4f}")
    print(f"Silhouette Score (UMAP 3D): {sil_umap_3d:.4f}")
    
    print("\n--- Distancias entre centroides (UMAP 3D) ---")
    cent, dist_mat, vocales = calcular_centroides_y_distancias(X_umap_3d, Y)
    
    df_dist = pd.DataFrame(dist_mat, index=vocales, columns=vocales)
    print(df_dist.to_string())
    
    # Guardar métricas
    with open(os.path.join(out_dir, "metricas.txt"), "w") as f:
        f.write(f"Silhouette Score (PCA 3D): {sil_pca:.4f}\n")
        f.write(f"Silhouette Score (UMAP 3D): {sil_umap:.4f}\n\n")
        f.write("Matriz de Distancias (UMAP 3D):\n")
        f.write(df_dist.to_string())
        
    print(f"\nProceso completado. Resultados guardados en {out_dir}")

class GeneradorPCAGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Generador PCA/UMAP")
        self.root.geometry("500x400")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_electrodos")
        
        main_frame = tk.Frame(root, padx=15, pady=15, bg="#0B0C10")
        main_frame.pack(fill="both", expand=True)
        
        meas_frame = tk.LabelFrame(main_frame, text="Seleccionar Mediciones para PCA/UMAP", padx=10, pady=10, bg="#1F2833", fg="#66FCF1")
        meas_frame.pack(fill="both", expand=True, pady=(0,15))
        
        self.listbox_mediciones = tk.Listbox(meas_frame, selectmode=tk.EXTENDED, bg="#0B0C10", fg="#66FCF1")
        self.listbox_mediciones.pack(side="left", fill="both", expand=True)
        
        sb = tk.Scrollbar(meas_frame, orient="vertical", command=self.listbox_mediciones.yview)
        sb.pack(side="right", fill="y")
        self.listbox_mediciones.config(yscrollcommand=sb.set)
        
        self.listbox_mediciones.bind("<<ListboxSelect>>", self.on_selection_change)
        
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
            
        self.root.destroy()
        ejecutar_procesamiento(seleccionadas)

def main():
    root = tk.Tk()
    app = GeneradorPCAGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
