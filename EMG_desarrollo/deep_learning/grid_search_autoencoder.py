import os
import sys
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Añadir directorios de deep_learning al sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
base_repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
if base_repo_dir not in sys.path:
    sys.path.insert(0, base_repo_dir)

from dataset_emg import EMGDataset
from modelos import ConvAutoencoder1D

def _set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_grid_search(
    csv_path=None, 
    latent_dims=None, 
    kernel_sizes=None, 
    batch_sizes=None,
    alphas=None, 
    epochs=80, 
    lr=1e-3, 
    verbose=True, 
    progress_callback=None
):
    """
    Ejecuta una búsqueda sistemática por grilla (Grid Search) sobre el espacio
    de hiperparámetros del Autoencoder Convolucional 1D para señales EMG.
    
    Por defecto recorre 36 combinaciones de alto impacto:
      - 4 dimensiones latentes: [2, 4, 8, 16]
      - 3 tamaños de kernel temporal: [5, 7, 9] (incorporando k=9 para dinámicas musculares largas)
      - 3 tamaños de batch: [8, 16, 32] (frecuencia de gradiente estocástico)
      - 1 factor alpha calibrado: [0.5]
    """
    _set_seed(42)
    
    if latent_dims is None:
        latent_dims = [2, 4, 8, 16]
    if kernel_sizes is None:
        kernel_sizes = [5, 7, 9]
    if batch_sizes is None:
        batch_sizes = [8, 16, 32]
    if alphas is None:
        alphas = [0.5]
        
    total_combinaciones = len(latent_dims) * len(kernel_sizes) * len(batch_sizes) * len(alphas)
    
    # 1. Localizar dataset si no se proveyó
    if csv_path is None or not os.path.exists(csv_path):
        candidates = [
            os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv"),
            os.path.join(base_repo_dir, "resultados", "resultados_pca_umap", "caracteristicas_exportadas.csv"),
            os.path.join(script_dir, "caracteristicas_exportadas.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break
                
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("No se encontró el dataset 'caracteristicas_exportadas.csv'. Ejecute primero la extracción de datos.")
        
    if verbose:
        print("=" * 75)
        print("INICIANDO GRID SEARCH DE AUTOENCODER (36 COMBINACIONES AVANZADAS)")
        print(f"Dataset: {os.path.abspath(csv_path)}")
        print(f"Dimensiones Latentes: {latent_dims}")
        print(f"Tamaños de Kernel:   {kernel_sizes}")
        print(f"Batch Sizes:         {batch_sizes}")
        print(f"Factores Alpha Loss: {alphas}")
        print(f"Épocas por corrida:  {epochs} | Learning Rate: {lr}")
        print(f"Total Combinaciones: {total_combinaciones}")
        print("=" * 75)

    # 2. Cargar Dataset con split determinista por sesión física (prevenir data leakage)
    dataset_train = EMGDataset(csv_path, apply_augmentation=True)
    dataset_val = EMGDataset(csv_path, apply_augmentation=False)
    inferred_target_length = dataset_train.tensors.shape[2]
    
    def get_session_id(toma_str):
        parts = toma_str.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}"
        return toma_str.split('_Win')[0]
        
    sesiones_base = [get_session_id(toma) for toma in dataset_train.tomas]
    sesiones_unicas = sorted(list(set(sesiones_base)))
    
    _set_seed(42)
    np.random.shuffle(sesiones_unicas)
    
    train_sesiones_size = max(1, int(0.8 * len(sesiones_unicas)))
    train_sesiones = set(sesiones_unicas[:train_sesiones_size])
    val_sesiones = set(sesiones_unicas[train_sesiones_size:])
    if not val_sesiones:
        val_sesiones = train_sesiones
        
    train_indices = [i for i, s in enumerate(sesiones_base) if s in train_sesiones]
    val_indices = [i for i, s in enumerate(sesiones_base) if s in val_sesiones]
    
    train_subset = torch.utils.data.Subset(dataset_train, train_indices)
    val_subset = torch.utils.data.Subset(dataset_val, val_indices)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Bucle de Grid Search
    resultados = []
    best_overall_acc = -1.0
    best_overall_loss = float('inf')
    best_model_state = None
    best_combo_info = None
    
    combo_idx = 0
    
    for l_dim in latent_dims:
        for k_size in kernel_sizes:
            for b_size in batch_sizes:
                for alpha in alphas:
                    combo_idx += 1
                    _set_seed(42) # Semilla idéntica para equidad absoluta
                    
                    train_loader = DataLoader(train_subset, batch_size=b_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=b_size, shuffle=False)
                    
                    if verbose:
                        print(f"\n[{combo_idx:2d}/{total_combinaciones}] LatentDim={l_dim:2d} | Kernel={k_size} | Batch={b_size:2d} | Alpha={alpha:.1f} ...", end=" ", flush=True)
                        
                    model = ConvAutoencoder1D(latent_dim=l_dim, target_length=inferred_target_length, kernel_size=k_size).to(device)
                    criterion_mse = nn.MSELoss()
                    criterion_ce = nn.CrossEntropyLoss()
                    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
                    
                    best_val_acc_combo = 0.0
                    best_val_loss_combo = float('inf')
                    best_epoch_combo = 0
                    best_weights_combo = None
                    
                    for epoch in range(1, epochs + 1):
                        model.train()
                        running_loss = 0.0
                        correct_train = 0
                        total_train = 0
                        
                        for batch_x, batch_y_idx, _ in train_loader:
                            batch_x = batch_x.to(device)
                            batch_y_idx = batch_y_idx.to(device)
                            
                            optimizer.zero_grad()
                            recon, latent, logits = model(batch_x)
                            
                            loss_recon = criterion_mse(recon, batch_x)
                            loss_cls = criterion_ce(logits, batch_y_idx)
                            loss_total = (1.0 - alpha) * loss_recon + alpha * loss_cls
                            
                            loss_total.backward()
                            optimizer.step()
                            
                            running_loss += loss_total.item() * batch_x.size(0)
                            preds = torch.argmax(logits, dim=1)
                            correct_train += (preds == batch_y_idx).sum().item()
                            total_train += batch_x.size(0)
                            
                        epoch_train_loss = running_loss / len(train_loader.dataset)
                        train_acc = 100.0 * correct_train / total_train
                        
                        # Validación
                        model.eval()
                        val_loss = 0.0
                        correct_val = 0
                        total_val = 0
                        with torch.no_grad():
                            for batch_x, batch_y_idx, _ in val_loader:
                                batch_x = batch_x.to(device)
                                batch_y_idx = batch_y_idx.to(device)
                                
                                recon, latent, logits = model(batch_x)
                                l_rec = criterion_mse(recon, batch_x)
                                l_c = criterion_ce(logits, batch_y_idx)
                                l_tot = (1.0 - alpha) * l_rec + alpha * l_c
                                
                                val_loss += l_tot.item() * batch_x.size(0)
                                preds = torch.argmax(logits, dim=1)
                                correct_val += (preds == batch_y_idx).sum().item()
                                total_val += batch_x.size(0)
                                
                        epoch_val_loss = val_loss / len(val_loader.dataset)
                        val_acc = 100.0 * correct_val / total_val
                        scheduler.step(epoch_val_loss)
                        
                        if val_acc > best_val_acc_combo:
                            best_val_acc_combo = val_acc
                            best_val_loss_combo = epoch_val_loss
                            best_epoch_combo = epoch
                            best_weights_combo = copy.deepcopy(model.state_dict())
                            
                    # Cargar mejores pesos del combo para calcular Silhouette Score
                    if best_weights_combo is not None:
                        model.load_state_dict(best_weights_combo)
                    model.eval()
                    
                    # Extraer representaciones latentes del validation set
                    latents_val = []
                    labels_val_list = []
                    with torch.no_grad():
                        for idx in val_indices:
                            bx, by_idx, by_str = dataset_val[idx]
                            bx = bx.unsqueeze(0).to(device)
                            _, lat, _ = model(bx)
                            latents_val.append(lat.cpu().numpy().squeeze())
                            labels_val_list.append(by_str)
                            
                    latents_val = np.array(latents_val)
                    if latents_val.ndim == 1:
                        latents_val = latents_val.reshape(-1, 1)
                        
                    if len(set(labels_val_list)) > 1 and latents_val.shape[1] > 1:
                        sil_val = silhouette_score(latents_val, labels_val_list, metric='euclidean')
                    else:
                        sil_val = 0.0
                        
                    if verbose:
                        print(f"Val Acc: {best_val_acc_combo:5.2f}% | Val Loss: {best_val_loss_combo:.4f} | Silhouette: {sil_val:+.4f} (Época {best_epoch_combo})")
                        
                    registro = {
                        "combo_id": combo_idx,
                        "latent_dim": l_dim,
                        "kernel_size": k_size,
                        "batch_size": b_size,
                        "alpha": alpha,
                        "val_acc": round(best_val_acc_combo, 2),
                        "val_loss": round(best_val_loss_combo, 4),
                        "silhouette": round(sil_val, 4),
                        "best_epoch": best_epoch_combo
                    }
                    resultados.append(registro)
                    
                    if progress_callback:
                        progress_callback(combo_idx, total_combinaciones, registro)
                        
                    # Evaluar si es el nuevo Campeón Global (Prioridad: mayor Val Acc, desempate: mayor Silhouette)
                    es_mejor = False
                    if best_val_acc_combo > best_overall_acc:
                        es_mejor = True
                    elif abs(best_val_acc_combo - best_overall_acc) < 1e-4:
                        if sil_val > (best_combo_info["silhouette"] if best_combo_info else -1):
                            es_mejor = True
                            
                    if es_mejor:
                        best_overall_acc = best_val_acc_combo
                        best_overall_loss = best_val_loss_combo
                        best_model_state = copy.deepcopy(best_weights_combo)
                        best_combo_info = registro
                        
    # 4. Exportar Resultados y Gráficos
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados.sort_values(by=["val_acc", "silhouette"], ascending=[False, False]).reset_index(drop=True)
    
    out_dir = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_out_path = os.path.join(out_dir, "grid_search_resultados.csv")
    df_resultados.to_csv(csv_out_path, index=False)
    
    # Guardar modelo campeón
    if best_model_state is not None:
        campeon_path = os.path.join(out_dir, "autoencoder_campeon.pth")
        torch.save(best_model_state, campeon_path)
        alias_path = os.path.join(out_dir, "autoencoder_emg.pth")
        torch.save(best_model_state, alias_path)
        
        # También guardar con nombre dimensional
        dim_path = os.path.join(out_dir, f"autoencoder_emg_{best_combo_info['latent_dim']}d.pth")
        torch.save(best_model_state, dim_path)

    # 5. Generar Mapa de Calor Comparativo (Heatmap)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    # Crear columna combinada (Kernel, Batch) para el eje X
    df_plot = df_resultados.copy()
    if len(alphas) > 1:
        df_plot["config_cnn"] = df_plot.apply(lambda r: f"K={int(r['kernel_size'])} | B={int(r['batch_size'])} | a={r['alpha']:.1f}", axis=1)
    else:
        df_plot["config_cnn"] = df_plot.apply(lambda r: f"K={int(r['kernel_size'])} | B={int(r['batch_size'])}", axis=1)
    
    pivot_acc = df_plot.pivot(index="latent_dim", columns="config_cnn", values="val_acc")
    pivot_sil = df_plot.pivot(index="latent_dim", columns="config_cnn", values="silhouette")
    
    sns.heatmap(pivot_acc, annot=True, fmt=".1f", cmap="YlGnBu", ax=axes[0], cbar_kws={'label': 'Validation Accuracy (%)'})
    axes[0].set_title("Grid Search: Validation Accuracy (%)", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Dimensión Latente ($d_{latente}$)")
    axes[0].set_xlabel("Configuración CNN (Kernel y Batch Size)")
    
    sns.heatmap(pivot_sil, annot=True, fmt=".3f", cmap="magma", ax=axes[1], cbar_kws={'label': 'Silhouette Score'})
    axes[1].set_title("Grid Search: Silhouette Score (Espacio Latente)", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Dimensión Latente ($d_{latente}$)")
    axes[1].set_xlabel("Configuración CNN (Kernel y Batch Size)")
    
    plt.tight_layout()
    heatmap_path = os.path.join(out_dir, "grid_search_heatmap.png")
    plt.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    
    if verbose:
        print("\n" + "=" * 75)
        print("GRID SEARCH FINALIZADO CON ÉXITO")
        print("=" * 75)
        print(f"Top 5 Mejores Configuraciones:")
        print(df_resultados.head(5).to_string(index=False))
        print("-" * 75)
        print(f"MODELO CAMPEÓN SELECCIONADO:")
        print(f"  - Latent Dim:   {best_combo_info['latent_dim']}D")
        print(f"  - Kernel Size:  {best_combo_info['kernel_size']}")
        print(f"  - Batch Size:   {best_combo_info['batch_size']}")
        print(f"  - Alpha Loss:   {best_combo_info['alpha']}")
        print(f"  - Val Accuracy: {best_combo_info['val_acc']:.2f}%")
        print(f"  - Silhouette:   {best_combo_info['silhouette']:+.4f}")
        print(f"Archivos guardados en:")
        print(f"  - CSV:     {csv_out_path}")
        print(f"  - Heatmap: {heatmap_path}")
        print(f"  - Modelo:  {campeon_path}")
        print("=" * 75)
        
    return df_resultados, best_combo_info

if __name__ == "__main__":
    run_grid_search(epochs=60)
