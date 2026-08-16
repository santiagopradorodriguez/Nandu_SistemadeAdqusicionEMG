import os
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score

from dataset_emg import EMGDataset
from modelos import ConvAutoencoder1D

def plot_latent_space(csv_path, model_path, latent_dim=16):
    print(f"==================================================")
    print(f"Generando Espacio Latente y Métricas...")
    print(f"Usando archivo de evaluación: {os.path.abspath(csv_path)}")
    print(f"==================================================")
    print("Cargando modelo y extrayendo espacio latente...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # El modelo fue entrenado con el pipeline comprimido: target_length=20 (20 pts/canal, 60D total)
    target_length = 20
    
    # Cargar dataset (sin augmentation para evaluar las muestras reales)
    dataset = EMGDataset(csv_path, target_length=target_length, apply_augmentation=False)
    
    model = ConvAutoencoder1D(latent_dim=latent_dim, target_length=target_length).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    latents = []
    labels_true = []
    labels_pred = []
    
    # Mapeo de índices a vocales (necesario para traducir la predicción)
    vocales_unicas = sorted(list(set(dataset.labels)))
    idx_to_vocal = {i: v for i, v in enumerate(vocales_unicas)}
    
    with torch.no_grad():
        for i in range(len(dataset)):
            x, y_idx, y_str = dataset[i]
            x = x.unsqueeze(0).to(device) # Añadir batch dim
            
            reconstruction, latent, logits = model(x)
            
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_str = idx_to_vocal[pred_idx]
            
            latents.append(latent.cpu().numpy().squeeze())
            labels_true.append(y_str)
            labels_pred.append(pred_str)
            
    latents = np.array(latents) # (N, 16)
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # ---------------------------------------------------------
    # RECREAR EL MISMO SPLIT DE TRAIN/TEST (Semilla 42)
    # ---------------------------------------------------------
    todas_las_tomas = dataset.tomas
    def get_session_id(toma_str):
        parts = toma_str.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        return toma_str.split('_Win')[0]
        
    sesiones_base = [get_session_id(toma) for toma in todas_las_tomas]
    sesiones_unicas = list(set(sesiones_base))
    sesiones_unicas.sort()
    
    np.random.seed(42)
    np.random.shuffle(sesiones_unicas)
    
    train_sesiones_size = int(0.8 * len(sesiones_unicas))
    train_sesiones = set(sesiones_unicas[:train_sesiones_size])
    val_sesiones = set(sesiones_unicas[train_sesiones_size:])
    
    train_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in train_sesiones]
    test_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in val_sesiones]
    
    latents_train = latents[train_indices]
    labels_train = labels_true[train_indices]
    
    latents_test = latents[test_indices]
    labels_test = labels_true[test_indices]
    
    print(f"Total: {len(latents)} | Train: {len(latents_train)} | Test: {len(latents_test)}")
    
    # Calcular Silhouette sobre el espacio latente de 16D (Solo Train para ser justos)
    sil_score_train = silhouette_score(latents_train, labels_train, metric='euclidean')
    print(f"Silhouette Score Train (Latent Space {latent_dim}D): {sil_score_train:.4f}")
    
    # Reducir a 3D con UMAP para visualización (Solo fit en Train)
    print("Aplicando UMAP para reducción a 3D (Solo en Train)...")
    reducer = umap.UMAP(n_components=3, random_state=42, min_dist=0.1)
    latents_train_3d = reducer.fit_transform(latents_train)
    
    print("Transformando el set de Test ciegamente...")
    latents_test_3d = reducer.transform(latents_test)
    
    sil_umap_3d_train = silhouette_score(latents_train_3d, labels_train, metric='euclidean')
    sil_umap_3d_test = silhouette_score(latents_test_3d, labels_test, metric='euclidean')
    print(f"Silhouette Score Train (UMAP 3D): {sil_umap_3d_train:.4f}")
    print(f"Silhouette Score Test (UMAP 3D): {sil_umap_3d_test:.4f}")
    
    # Plotear en 3D (Dos gráficos: Train vs Test)
    fig = plt.figure(figsize=(18, 8))
    
    vocales = sorted(list(set(labels_true)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    # Subplot Train
    ax1 = fig.add_subplot(121, projection='3d')
    for i, vocal in enumerate(vocales):
        idx = labels_train == vocal
        ax1.scatter(
            latents_train_3d[idx, 0], latents_train_3d[idx, 1], latents_train_3d[idx, 2], 
            label=vocal, color=palette[i], s=40, edgecolor='k', alpha=0.8
        )
    ax1.set_title(f"UMAP 3D (Train Set)\nSil Train: {sil_umap_3d_train:.4f}")
    
    # Subplot Test
    ax2 = fig.add_subplot(122, projection='3d')
    for i, vocal in enumerate(vocales):
        idx = labels_test == vocal
        ax2.scatter(
            latents_test_3d[idx, 0], latents_test_3d[idx, 1], latents_test_3d[idx, 2], 
            label=vocal, color=palette[i], s=40, edgecolor='k', alpha=0.8
        )
    ax2.set_title(f"UMAP 3D (Test Set Ciego)\nSil Test: {sil_umap_3d_test:.4f}")
    
    plt.legend()
    
    base_repo_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    out_dir = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder")
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"latent_space_umap_3d_{latent_dim}d.png")
    plt.savefig(plot_path)
    print(f"Gráfico guardado en {plot_path}")
    
    # --- GRÁFICOS DE PRECISIÓN (Accuracy por Vocal) ---
    print("\nGenerando gráficos de precisión...")
    
    # Matriz de confusión SOLO SOBRE EL TEST SET
    labels_test_true = labels_true[test_indices]
    labels_test_pred = labels_pred[test_indices]
    cm = confusion_matrix(labels_test_true, labels_test_pred, labels=vocales)
    
    # Accuracy por vocal
    acc_por_vocal = cm.diagonal() / cm.sum(axis=1) * 100
    acc_global = accuracy_score(labels_test_true, labels_test_pred) * 100
    print(f"Accuracy Global en Test Set: {acc_global:.2f}%")
    
    fig2, (ax_cm, ax_bar) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Matriz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vocales, yticklabels=vocales, ax=ax_cm)
    ax_cm.set_title('Matriz de Confusión')
    ax_cm.set_xlabel('Predicción')
    ax_cm.set_ylabel('Real')
    
    # Plot Barras
    bars = ax_bar.bar(vocales, acc_por_vocal, color=palette)
    ax_bar.set_title('Accuracy por Vocal (%)')
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel('Precisión (%)')
    
    # Añadir números sobre las barras con 2 decimales
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold')
                
    plt.tight_layout()
    acc_plot_path = os.path.join(out_dir, f"accuracy_por_vocal_{latent_dim}d.png")
    plt.savefig(acc_plot_path)
    print(f"Gráficos de precisión guardados.")
    
    # Cerrar las figuras de matplotlib para liberar memoria
    plt.close('all')
    
    # Abrir los PNG generados con el visor del sistema (no bloqueante)
    import subprocess
    for img_path in [plot_path, acc_plot_path]:
        if os.path.exists(img_path):
            subprocess.Popen(["xdg-open", img_path])
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "..", "analysis", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
    model_file = os.path.join(base_dir, "autoencoder_emg_16d.pth")
    
    if not os.path.exists(model_file):
        print(f"No se encontró el modelo entrenado en {model_file}")
        print("Por favor corré train_autoencoder.py primero.")
    else:
        plot_latent_space(csv_file, model_file, latent_dim=16)
