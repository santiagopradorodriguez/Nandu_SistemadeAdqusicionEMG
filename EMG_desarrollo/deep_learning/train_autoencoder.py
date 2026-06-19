import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from dataset_emg import EMGDataset
from modelos import ConvAutoencoder1D

def train_autoencoder(csv_path, epochs=150, batch_size=32, lr=1e-3, latent_dim=16):
    print(f"==================================================")
    print(f"Iniciando entrenamiento del Autoencoder Convolucional...")
    print(f"Usando archivo de entrenamiento: {os.path.abspath(csv_path)}")
    print(f"==================================================")
    
    dataset_train = EMGDataset(csv_path, target_length=100, apply_augmentation=True)
    dataset_val = EMGDataset(csv_path, target_length=100, apply_augmentation=False)
    
    # ---------------------------------------------------------
    # FIX DATA LEAKAGE: Split por 'Toma' en lugar de ventanas
    # ---------------------------------------------------------
    import numpy as np
    
    todas_las_tomas = dataset_train.tomas
    # Extraer el identificador físico de la sesión (ej. 'T1_Lucas' de 'A_T1_Lucas_Win0')
    # El formato es {Vocal}_{Toma}_{Paciente}_Win{X}
    def get_session_id(toma_str):
        parts = toma_str.split('_')
        if len(parts) >= 3:
            return f"{parts[1]}_{parts[2]}" # Ej: T1_Lucas
        return toma_str.split('_Win')[0]
        
    sesiones_base = [get_session_id(toma) for toma in todas_las_tomas]
    sesiones_unicas = list(set(sesiones_base))
    
    # Ordenar y fijar semilla para reproducibilidad
    sesiones_unicas.sort()
    np.random.seed(42)
    np.random.shuffle(sesiones_unicas)
    
    # 80% Sesiones para Train, 20% Sesiones para Validación
    train_sesiones_size = int(0.8 * len(sesiones_unicas))
    train_sesiones = set(sesiones_unicas[:train_sesiones_size])
    val_sesiones = set(sesiones_unicas[train_sesiones_size:])
    
    train_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in train_sesiones]
    val_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in val_sesiones]
    
    print(f"Total de Sesiones Físicas: {len(sesiones_unicas)} | Sesiones Train: {len(train_sesiones)} | Sesiones Val: {len(val_sesiones)}")
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 2. Inicializar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de entrenamiento: {device}")
    
    model = ConvAutoencoder1D(latent_dim=latent_dim).to(device)
    
    # Criterio y Optimizador
    criterion_mse = nn.MSELoss() # Reconstrucción
    criterion_ce = nn.CrossEntropyLoss() # Clasificación
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Historial para graficar
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    alpha = 0.5 # Factor de balance (50% Reconstruccion, 50% Clasificacion)
    
    # 3. Bucle de Entrenamiento
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels_idx, labels_str in train_loader:
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
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:3d}/{epochs}] - Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")
            
    # 4. Guardar resultados
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
    out_dir = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder")
    os.makedirs(out_dir, exist_ok=True)
    
    model_path = os.path.join(out_dir, f"autoencoder_emg_{latent_dim}d.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modelo guardado en {model_path}")
    
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
    ax2.set_title('Curva de Clasificación de Vocales')
    ax2.legend()
    ax2.grid(True)
    
    plot_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(plot_path)
    # Mostrar curva de entrenamiento
    plt.show()
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
    csv_file = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
    
    if not os.path.exists(csv_file):
        print(f"Error: No se encontró el dataset en {csv_file}")
        print("Por favor corré el generador_pca_tensorial.py primero.")
    else:
        train_autoencoder(csv_file, epochs=400, batch_size=16, latent_dim=16)
