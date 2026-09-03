import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Forzar reproducibilidad absoluta
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

try:
    from dataset_emg import EMGDataset
    from modelos import ConvAutoencoder1D
except ImportError:
    from deep_learning.dataset_emg import EMGDataset
    from deep_learning.modelos import ConvAutoencoder1D

def train_autoencoder(csv_path, epochs=80, batch_size=16, lr=1e-3, latent_dim=8, kernel_size=5, force_epochs=False, alpha=0.5, verbose=True, save_model=True, train_sessions=None, test_sessions=None, out_dir=None):
    def _set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    _set_seed(42)
        
    if verbose:
        print(f"==================================================")
        print(f"Iniciando entrenamiento del Autoencoder Convolucional...")
        print(f"Usando archivo de entrenamiento: {os.path.abspath(csv_path)}")
        print(f"Parametros: Epochs={epochs}, BatchSize={batch_size}, LR={lr}, LatentDim={latent_dim}, KernelSize={kernel_size}, Alpha={alpha}")
        print(f"==================================================")
    
    dataset_train = EMGDataset(csv_path, apply_augmentation=True)
    dataset_val = EMGDataset(csv_path, apply_augmentation=False)
    
    # ---------------------------------------------------------
    # PARTICIÓN DE SESIONES FÍSICAS (TRAIN / TEST)
    # ---------------------------------------------------------
    todas_las_tomas = dataset_train.tomas
    train_sessions_names = [os.path.basename(s).strip() for s in (train_sessions or []) if s and str(s).strip()]
    test_sessions_names = [os.path.basename(s).strip() for s in (test_sessions or []) if s and str(s).strip()]
    
    train_indices = []
    val_indices = []
    
    if train_sessions_names or test_sessions_names:
        if verbose:
            print(f"Aplicando partición manual definida por el usuario...")
            print(f"  -> Sesiones asignadas a TRAIN: {train_sessions_names}")
            print(f"  -> Sesiones asignadas a TEST:  {test_sessions_names}")
            
        for i, toma in enumerate(todas_las_tomas):
            asignado = False
            toma_clean = toma.replace(" ", "").lower()
            for s_name in train_sessions_names:
                if s_name.replace(" ", "").lower() in toma_clean:
                    train_indices.append(i)
                    asignado = True
                    break
            if not asignado:
                for s_name in test_sessions_names:
                    if s_name.replace(" ", "").lower() in toma_clean:
                        val_indices.append(i)
                        asignado = True
                        break
                        
        if not train_indices:
            print("[Advertencia] No hubo coincidencia para Train con los nombres dados. Se usarán todas las muestras para Train.")
            train_indices = list(range(len(todas_las_tomas)))
            
        if not val_indices:
            print("[Aviso] No se asignaron sesiones a Test. Se usará el conjunto de Train para validación.")
            val_indices = train_indices
            
        train_sesiones = sorted(list(set(train_sessions_names))) if train_sessions_names else ["Todas"]
        val_sesiones = sorted(list(set(test_sessions_names))) if test_sessions_names else train_sesiones
    else:
        # Fallback: Split determinista por Sesión Física Real (Prueba1_Candela, Prueba2_Candela, etc.)
        def get_session_id(toma_str):
            base = toma_str.rsplit('_Win', 1)[0]
            parts = base.split('_')
            if len(parts) > 1 and parts[0].upper() in ['A', 'E', 'I', 'O', 'U']:
                return '_'.join(parts[1:])
            return base
            
        sesiones_base = [get_session_id(toma) for toma in todas_las_tomas]
        sesiones_unicas = sorted(list(set(sesiones_base)))
        
        _set_seed(42)
        np.random.shuffle(sesiones_unicas)
        
        train_sesiones_size = max(1, int(0.8 * len(sesiones_unicas)))
        train_sesiones = set(sesiones_unicas[:train_sesiones_size])
        val_sesiones = set(sesiones_unicas[train_sesiones_size:])
        if not val_sesiones:
            val_sesiones = train_sesiones
        
        train_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in train_sesiones]
        val_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in val_sesiones]
    
    print(f"Total de Sesiones Físicas: {len(train_sesiones) + len(val_sesiones)} | Sesiones Train: {len(train_sesiones)} | Sesiones Val: {len(val_sesiones)}")
    print(f"--- SESIONES FÍSICAS EN ENTRENAMIENTO ---")
    for t in sorted(list(train_sesiones)):
        print(f"  - {t}")
    print(f"--- SESIONES FÍSICAS EN VALIDACIÓN ---")
    for t in sorted(list(val_sesiones)):
        print(f"  - {t}")
    print(f"--------------------------------------------------")
    print(f"Ventanas Train: {len(train_indices)} | Ventanas Val: {len(val_indices)}")
    
    train_dataset = torch.utils.data.Subset(dataset_train, train_indices)
    val_dataset = torch.utils.data.Subset(dataset_val, val_indices)
    
    _set_seed(42)
    drop_last_train = (len(train_dataset) > batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last_train)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Inicializar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    inferred_target_length = dataset_train.tensors.shape[2]
    model = ConvAutoencoder1D(latent_dim=latent_dim, target_length=inferred_target_length, kernel_size=kernel_size).to(device)
    
    # Criterio y Optimizador
    criterion_mse = nn.MSELoss() # Reconstrucción
    criterion_ce = nn.CrossEntropyLoss() # Clasificación
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    
    # Historial para graficar
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # alpha ya viene como parámetro
    
    best_val_loss = float('inf')
    best_model_wts = None
    best_epoch = 0
    best_val_acc = 0.0
    
    # 3. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels_idx, labels_str in train_loader:
            if inputs.size(0) <= 1:
                continue
            inputs = inputs.to(device)
            labels_idx = labels_idx.to(device)
            
            optimizer.zero_grad()
            reconstruction, latent, logits = model(inputs)
            
            loss_rec = criterion_mse(reconstruction, inputs)
            loss_cls = criterion_ce(logits, labels_idx)
            loss = (1 - alpha) * loss_rec + alpha * loss_cls
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Accuracy
            _, predicted = torch.max(logits.data, 1)
            total_train += labels_idx.size(0)
            correct_train += (predicted == labels_idx).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accs.append(train_acc)
        
        # Validación
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels_idx, labels_str in val_loader:
                inputs = inputs.to(device)
                labels_idx = labels_idx.to(device)
                
                reconstruction, latent, logits = model(inputs)
                
                loss_rec = criterion_mse(reconstruction, inputs)
                loss_cls = criterion_ce(logits, labels_idx)
                loss = (1 - alpha) * loss_rec + alpha * loss_cls
                
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(logits.data, 1)
                total_val += labels_idx.size(0)
                correct_val += (predicted == labels_idx).sum().item()
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(val_acc)
        scheduler.step(epoch_val_loss)
        
        if verbose and (epoch % 10 == 0 or epoch == epochs):
            print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
            
        # Guardar el mejor modelo (Model Checkpointing basado en Validation Accuracy)
        if val_acc > best_val_acc:
            best_val_loss = epoch_val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            import copy
            best_model_wts = copy.deepcopy(model.state_dict())
            
    if verbose:
        print(f"\nEntrenamiento finalizado. Mejor Validation Accuracy logrado: {best_val_acc:.2f}% en la Época {best_epoch} (Loss: {best_val_loss:.4f})")
    
    if force_epochs:
        if verbose: print(f"FORZANDO ÉPOCAS: Se guardarán los pesos de la última época ({epochs}) ignorando el Checkpointing.")
    elif best_model_wts is not None:
        if verbose: print(f"Restaurando los pesos del modelo a la época con mejor Accuracy ({best_epoch}) (Cancelando Overfitting de Clasificación)...")
        model.load_state_dict(best_model_wts)
            
    # 4. Guardar resultados
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
    central_dir = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder")
    
    if out_dir is None:
        sujeto_detectado = "General"
        for t in dataset_train.tomas:
            parts = t.rsplit('_Win', 1)[0].split('_')
            if len(parts) >= 3:
                sujeto_detectado = parts[-1]
                break
        import datetime
        fecha_detectada = datetime.datetime.now().strftime("%Y-%m-%d")
        out_dir = os.path.join(central_dir, f"{fecha_detectada}_{sujeto_detectado}")
        
    if save_model:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(central_dir, exist_ok=True)
        
        model_path = os.path.join(out_dir, f"autoencoder_emg_{latent_dim}d.pth")
        weights_to_save = best_model_wts if best_model_wts and not force_epochs else model.state_dict()
        torch.save(weights_to_save, model_path)
        
        # Guardar alias global en out_dir y en central_dir
        alias_path = os.path.join(out_dir, "autoencoder_emg.pth")
        torch.save(weights_to_save, alias_path)
        torch.save(weights_to_save, os.path.join(central_dir, f"autoencoder_emg_{latent_dim}d.pth"))
        torch.save(weights_to_save, os.path.join(central_dir, "autoencoder_emg.pth"))
        
        # Guardar configuración de partición para ploteo y evaluación
        import json
        split_cfg = {
            "train_sessions": sorted(list(set(train_sesiones))),
            "test_sessions": sorted(list(set(val_sesiones))),
            "latent_dim": latent_dim
        }
        with open(os.path.join(out_dir, "split_config.json"), "w") as f:
            json.dump(split_cfg, f, indent=4)
        with open(os.path.join(central_dir, "split_config.json"), "w") as f:
            json.dump(split_cfg, f, indent=4)
            
        if verbose: 
            print(f"Resultados y modelo guardados en carpeta:")
            print(f"  -> Carpeta Sujeto/Fecha: {out_dir}")
            print(f"  -> Galería Central:      {central_dir}")
    
    # Graficar curva de Loss y Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (MSE + CE)')
    ax1.set_title('Curva de Aprendizaje (Pérdida Total)')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(train_accs, label='Train Accuracy', color='green')
    ax2.plot(val_accs, label='Validation Accuracy', color='orange')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    plt.title(f'Autoencoder Training Curve\nBest Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})')
    plt.legend()
    
    if save_model:
        plot_path = os.path.join(out_dir, "loss_curve.png")
        plt.savefig(plot_path)
        if out_dir != central_dir:
            plt.savefig(os.path.join(central_dir, "loss_curve.png"))
    plt.close(fig)
    
    return best_val_loss, best_val_acc

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_file = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró el dataset en {csv_file}")
        print("Por favor corré el generador_pca_tensorial.py primero.")
    else:
        train_autoencoder(csv_file, epochs=400, batch_size=16, latent_dim=16)
