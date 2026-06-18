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
    
    # Cargar dataset (sin augmentation para evaluar las muestras reales)
    dataset = EMGDataset(csv_path, target_length=100, apply_augmentation=False)
    
    model = ConvAutoencoder1D(latent_dim=latent_dim).to(device)
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
    
    # Calcular Silhouette sobre el espacio latente de 16D
    sil_score = silhouette_score(latents, labels_true, metric='euclidean')
    print(f"Silhouette Score (Latent Space {latent_dim}D): {sil_score:.4f}")
    
    # Reducir a 3D con UMAP para visualización
    print("Aplicando UMAP para reducción a 3D...")
    reducer = umap.UMAP(n_components=3, random_state=42, min_dist=0.1)
    latents_3d = reducer.fit_transform(latents)
    
    sil_umap_3d = silhouette_score(latents_3d, labels_true, metric='euclidean')
    print(f"Silhouette Score (Latent Space -> UMAP 3D): {sil_umap_3d:.4f}")
    
    # Plotear en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    vocales = sorted(list(set(labels_true)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    for i, vocal in enumerate(vocales):
        idx = labels_true == vocal
        ax.scatter(
            latents_3d[idx, 0], 
            latents_3d[idx, 1], 
            latents_3d[idx, 2], 
            label=vocal, 
            color=palette[i], 
            s=40, 
            edgecolor='k', 
            alpha=0.8
        )
        
    ax.set_title(f"Espacio Latente (Autoencoder) -> UMAP 3D\nSilhouette {latent_dim}D: {sil_score:.4f}")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
    plt.legend()
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_path = os.path.join(out_dir, "latent_space_umap_3d.png")
    plt.savefig(plot_path)
    print(f"Gráfico guardado en {plot_path}")
    
    # --- GRÁFICOS DE PRECISIÓN (Accuracy por Vocal) ---
    print("\nGenerando gráficos de precisión...")
    
    # Matriz de confusión
    cm = confusion_matrix(labels_true, labels_pred, labels=vocales)
    
    # Accuracy por vocal
    acc_por_vocal = cm.diagonal() / cm.sum(axis=1) * 100
    
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
    
    # Añadir números sobre las barras
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
                
    acc_path = os.path.join(out_dir, "accuracy_por_vocal.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    print(f"Gráfico de precisión guardado en {acc_path}")
    
    # Mostrar todos los gráficos juntos
    plt.show()
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "..", "analysis", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
    model_file = os.path.join(base_dir, "autoencoder_emg_16d.pth")
    
    if not os.path.exists(model_file):
        print(f"No se encontró el modelo entrenado en {model_file}")
        print("Por favor corré train_autoencoder.py primero.")
    else:
        plot_latent_space(csv_file, model_file, latent_dim=16)
