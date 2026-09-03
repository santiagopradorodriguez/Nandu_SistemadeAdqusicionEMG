# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Reducción de dimensionalidad y clustering con Encoder Convolucional 1D
#              Configuración por defecto: smooth_ms=90, target_len=20, alpha_ruido=1.0
# ==============================================================================

import os
import sys
import io
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed=42):
    """Fija todas las semillas aleatorias para reproducibilidad absoluta."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Fijar semilla global al cargar el módulo
set_seed(42)

# Ajuste de rutas para importar utilidades del proyecto
script_dir = os.path.dirname(os.path.abspath(__file__))
emg_desarrollo_dir = os.path.dirname(script_dir)
if emg_desarrollo_dir not in sys.path:
    sys.path.append(emg_desarrollo_dir)
    sys.path.append(os.path.join(script_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(script_dir, "binarizacion"))

try:
    import analisis_trevisan as at
except ImportError:
    at = None

# ==========================================
# 1. ARQUITECTURA: DenseAutoencoder3D (Non-linear PCA)
# ==========================================
class ConvAutoencoder1D(nn.Module):
    # Mantenemos el nombre de la clase para no romper el resto del script
    def __init__(self, latent_dim=3, target_length=20, kernel_size=None):
        super(ConvAutoencoder1D, self).__init__()
        self.target_length = target_length
        self.latent_dim = latent_dim
        
        input_features = 3 * target_length  # 3 canales x 20 muestras = 60
        
        # Encoder puramente denso (MLP) - Ve la foto global igual que PCA
        self.encoder = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder puramente denso
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, input_features),
            nn.ReLU()  # Envolventes positivas
        )
        
        # Clasificador para forzar separación (opcional pero ayuda al 3D)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 5)
        )

    def encode(self, x):
        x_flat = x.view(x.size(0), -1)
        latent = self.encoder(x_flat)
        return latent

    def decode(self, latent):
        x_rec = self.decoder(latent)
        x_rec = x_rec.view(x_rec.size(0), 3, self.target_length)
        return x_rec

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        logits = self.classifier(latent)
        return reconstruction, latent, logits

# ==========================================
# 2. CARGA Y EXTRACCIÓN DE BIOPOTENCIALES
# ==========================================
def get_interpulse_noise(sub_signal, initial_noise):
    if len(sub_signal) == 0:
        return initial_noise
    m = np.mean(sub_signal)
    if initial_noise > 0 and (m / initial_noise) > 5.0:
        return initial_noise
    return m

def cargar_y_procesar_dataset(
    db_root="base_de_datos_electrodos",
    smooth_ms=90,          # <-- Envolvente por defecto 90 ms
    target_len=20,         # <-- Remuestreo por defecto 20 puntos
    alpha_ruido=1.0,       # <-- Supresión de ruido basal alpha=1.0
    pre_pct=0.40,          # <-- 40% pre-pico
    post_pct=0.60          # <-- 60% post-pico
):
    print(f"\n[INFO] Leyendo datos desde: {os.path.abspath(db_root)}")
    print(f"[INFO] Parámetros: smooth_ms={smooth_ms} ms | target_len={target_len} puntos | alpha_ruido={alpha_ruido}")
    
    canales_features = ["canal_0", "canal_1", "canal_2"]
    canales_procesar = canales_features + ["canal_3"]
    
    X_tensors, Y_labels, Sesiones = [], [], []
    
    carpetas_sesion = []
    for root, dirs, files in os.walk(db_root):
        if all(ch in dirs for ch in canales_procesar):
            carpetas_sesion.append(root)
            
    if not carpetas_sesion:
        print("[ADVERTENCIA] No se encontraron carpetas con la estructura oficial canal_0..3.")
        return np.array([]), np.array([]), []

    for carpeta_sesion in sorted(carpetas_sesion):
        med_name = os.path.basename(carpeta_sesion)
        vocal_detectada = med_name.split('_')[0].upper()
        if vocal_detectada not in ["A", "E", "I", "O", "U"]:
            continue
            
        canales_data = {}
        for ch in canales_procesar:
            ch_dir = os.path.join(carpeta_sesion, ch)
            meta_path = os.path.join(carpeta_sesion, "canal_0", "metadata.json")
            bpm_u, noise_u, pulsos_u = 30, 2.0, None
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        bpm_u = meta.get("bpm", 30)
                        noise_u = meta.get("noise_seconds", 2.0)
                        pulsos_u = meta.get("pulse_count", None)
                except:
                    pass
            
            if at is not None:
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    res_final = at.procesar_wavs_promedio(
                        carpeta=ch_dir, output_root=ch_dir,
                        bpm=bpm_u, mostrar_recortes=False,
                        noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
                        excluded_windows=[], show_interactive_plot=False,
                        notch_q_factor=2.0, tipo_envolvente="rms",
                        smooth_ms=smooth_ms, pre_pct=pre_pct, post_pct=post_pct
                    )
                finally:
                    sys.stdout = old_stdout
                    
                if res_final:
                    fname = list(res_final.keys())[0]
                    canales_data[ch] = res_final[fname]
                    
        if len(canales_data) < len(canales_procesar):
            continue
            
        # Sincronización acústica con Canal 3 (Micrófono)
        muestras_pulso = canales_data["canal_3"]["muestras_pulso"]
        env_mic_raw = canales_data["canal_3"]["env_recortada"]
        dist_samples = int(0.8 * muestras_pulso)
        min_height = np.max(env_mic_raw) * 0.2
        picos_mic, _ = find_peaks(env_mic_raw, distance=dist_samples, height=min_height)
        
        pre_samples = int(muestras_pulso * pre_pct)
        post_samples = int(muestras_pulso * post_pct)
        
        for win_idx, pico in enumerate(picos_mic):
            if win_idx == 0:
                continue
                
            real_cut_start = pico - pre_samples
            real_cut_end = pico + post_samples
            
            if real_cut_start < 0 or real_cut_end > len(env_mic_raw):
                continue
                
            segs_canales = []
            max_supremo = 1e-9
            valido = True
            
            for ch in canales_features:
                env_ch = canales_data[ch]["env_recortada"]
                if real_cut_end > len(env_ch):
                    valido = False
                    break
                
                segmento = env_ch[real_cut_start:real_cut_end].copy()
                
                # Sustracción de ruido interpulso
                initial_noise = canales_data[ch].get("noise_levels", [0])[0] if len(canales_data[ch].get("noise_levels", [])) > 0 else 0
                noise_win = max(3, int(muestras_pulso / 4.0))
                
                n_pre_start = max(0, int(pico - 0.5 * muestras_pulso - noise_win))
                n_pre_end = min(len(env_ch), n_pre_start + noise_win)
                r_pre = get_interpulse_noise(env_ch[n_pre_start:n_pre_end], initial_noise)
                
                n_post_start = min(len(env_ch), int(pico + 0.5 * muestras_pulso))
                n_post_end = min(len(env_ch), n_post_start + noise_win)
                r_post = get_interpulse_noise(env_ch[n_post_start:n_post_end], initial_noise)
                
                r_prom = (r_pre + r_post) / 2.0
                segmento_limpio = np.maximum(segmento - r_prom * alpha_ruido, 0.0)
                
                m_val = np.max(segmento_limpio)
                if m_val > max_supremo:
                    max_supremo = m_val
                segs_canales.append(segmento_limpio)
                
            if not valido or max_supremo <= 1e-6:
                continue
                
            # Normalización relativa por supremo y remuestreo FFT a target_len
            canales_rs = []
            for seg in segs_canales:
                seg_norm = seg / max_supremo
                seg_rs = resample(seg_norm, target_len)
                seg_rs[seg_rs < 0] = 0.0
                canales_rs.append(seg_rs)
                
            tensor_3ch = np.stack(canales_rs, axis=0)  # (3, target_len)
            X_tensors.append(tensor_3ch)
            Y_labels.append(vocal_detectada)
            Sesiones.append(med_name)

    X_arr = np.array(X_tensors, dtype=np.float32)
    Y_arr = np.array(Y_labels)
    print(f"[EXITO] Tensor final construido: {X_arr.shape[0]} muestras con dimensiones {X_arr.shape[1:]}")
    return X_arr, Y_arr, Sesiones


# ==========================================
# 3. REDUCCIÓN CON ENCODER CONVOLUCIONAL 1D
# ==========================================
def reducir_con_encoder(
    X_tensors, Y_labels,
    latent_dim=2,
    target_len=20,
    epochs=60,
    batch_size=16,
    lr=1e-3,
    modelo_guardado=None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvAutoencoder1D(latent_dim=latent_dim, target_length=target_len).to(device)
    
    vocal_to_idx = {"A": 0, "E": 1, "I": 2, "O": 3, "U": 4}
    y_indices = np.array([vocal_to_idx[v] for v in Y_labels], dtype=np.int64)
    
    t_X = torch.tensor(X_tensors, dtype=torch.float32)
    t_Y = torch.tensor(y_indices, dtype=torch.long)
    
    if modelo_guardado and os.path.exists(modelo_guardado):
        print(f"[INFO] Cargando checkpoint: {modelo_guardado}")
        model.load_state_dict(torch.load(modelo_guardado, map_location=device))
    else:
        print(f"[INFO] Optimizando VAE ({latent_dim}D) con Restricciones Físicas durante {epochs} épocas...")
        dataset = TensorDataset(t_X, t_Y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        criterion_mse = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for bx, by in loader:
                bx = bx.to(device)
                
                # Forward pass
                rec, lat, _ = model(bx)
                
                # Pérdida 100% no supervisada (Física)
                loss = criterion_mse(rec, bx)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                print(f"  Época [{epoch+1}/{epochs}] - Loss Total: {total_loss/len(loader):.4f}")

    # Extracción de coordenadas latentes
    model.eval()
    with torch.no_grad():
        t_X_dev = t_X.to(device)
        Z_latent = model.encode(t_X_dev).cpu().numpy()
        
    return Z_latent, model


# ==========================================
# 4. VISUALIZACIÓN Y CLUSTERING
# ==========================================
def graficar_proyeccion(Z_latent, Y_labels, latent_dim=2, output_img="proyeccion_encoder.png"):
    palette = {"A": "#e41a1c", "E": "#377eb8", "I": "#4daf4a", "O": "#984ea3", "U": "#ff7f00"}
    vocales_unicas = sorted(list(set(Y_labels)))
    
    # Clustering K-Means y asignación húngara
    n_clusters = len(vocales_unicas)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_preds = kmeans.fit_predict(Z_latent)
    
    vocal_to_idx = {v: i for i, v in enumerate(vocales_unicas)}
    real_indices = np.array([vocal_to_idx[v] for v in Y_labels])
    cost_matrix = np.zeros((n_clusters, n_clusters))
    for c_i in range(n_clusters):
        for v_j in range(n_clusters):
            cost_matrix[c_i, v_j] = -np.sum((cluster_preds == c_i) & (real_indices == v_j))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {row: col for row, col in zip(row_ind, col_ind)}
    mapped_preds = np.array([cluster_to_vocal[c] for c in cluster_preds])
    acc = accuracy_score(real_indices, mapped_preds) * 100.0
    
    if latent_dim == 2:
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        for v in vocales_unicas:
            mask = (Y_labels == v)
            ax.scatter(
                Z_latent[mask, 0], Z_latent[mask, 1],
                label=f"Vocal /{v.lower()}/",
                color=palette.get(v, "#333333"),
                alpha=0.8, edgecolors="none", s=55
            )
            
        centroids = kmeans.cluster_centers_
        for c_idx, cent in enumerate(centroids):
            v_name = vocales_unicas[cluster_to_vocal[c_idx]]
            ax.scatter(cent[0], cent[1], c="black", marker="X", s=130, linewidths=1.5, zorder=5)
            ax.annotate(f"Centr. /{v_name.lower()}/", (cent[0], cent[1]), textcoords="offset points", xytext=(0, 6), ha="center", fontweight="bold", fontsize=9)
            
        ax.set_title(f"Reducción de Dimensionalidad 2D con Encoder Convolucional 1D\nExactitud K-Means: {acc:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dimensión Latente 1 ($z_1$)", fontsize=10)
        ax.set_ylabel("Dimensión Latente 2 ($z_2$)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", frameon=True)
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        print(f"[EXITO] Gráfico guardado en: {output_img}")
        
    elif latent_dim == 3:
        fig = plt.figure(figsize=(9.5, 7.5))
        ax = fig.add_subplot(111, projection="3d")
        for v in vocales_unicas:
            mask = (Y_labels == v)
            ax.scatter(
                Z_latent[mask, 0], Z_latent[mask, 1], Z_latent[mask, 2],
                label=f"Vocal /{v.lower()}/",
                color=palette.get(v, "#333333"),
                alpha=0.8, s=45
            )
        ax.set_title(f"Espacio Latente 3D con Encoder Convolucional 1D (Acc K-Means: {acc:.1f}%)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Dimensión Latente 1 ($z_1$)")
        ax.set_ylabel("Dimensión Latente 2 ($z_2$)")
        ax.set_zlabel("Dimensión Latente 3 ($z_3$)")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_img, dpi=300)
        print(f"[EXITO] Gráfico guardado en: {output_img}")


# ==========================================
# 5. EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    SMOOTH_MS = 90
    TARGET_LEN = 20
    ALPHA_RUIDO = 1.0
    LATENT_DIM = 3  # Reducir directo a 3D con Autoencoder Denso + Supervisión
    
    # Ruta a las mediciones del 07 (2026-07-10) o la pasada por argumento
    if len(sys.argv) > 1:
        ruta_datos = sys.argv[1]
    else:
        ruta_datos = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos", "2026-07-10")
        if not os.path.exists(ruta_datos):
            ruta_datos = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos")
            
    print(f"[NANDU LSD] Ejecutando reducción de dimensionalidad con Encoder Convolucional 1D")
    print(f"            Ruta seleccionada: {os.path.abspath(ruta_datos)}")
    
    # 1. Carga y preprocesamiento de bioseñales idéntico a analisis_trevisan
    X, Y, sesiones = cargar_y_procesar_dataset(
        db_root=ruta_datos,
        smooth_ms=SMOOTH_MS,
        target_len=TARGET_LEN,
        alpha_ruido=ALPHA_RUIDO,
        pre_pct=0.40,
        post_pct=0.60
    )
    
    if len(X) > 0:
        # 2. Reducción latente con el Encoder
        Z_lat, model = reducir_con_encoder(
            X, Y,
            latent_dim=LATENT_DIM,
            target_len=TARGET_LEN,
            epochs=60
        )
        
        # 3. Graficado y evaluación K-Means
        out_file = f"proyeccion_encoder_{LATENT_DIM}d_2026-07-10.png"
        graficar_proyeccion(
            Z_lat, Y,
            latent_dim=LATENT_DIM,
            output_img=out_file
        )
        print(f"[LISTO] Gráfico generado con éxito en: {os.path.abspath(out_file)}")
    else:
        print("[ERROR] No se pudieron cargar datos válidos para procesar.")
