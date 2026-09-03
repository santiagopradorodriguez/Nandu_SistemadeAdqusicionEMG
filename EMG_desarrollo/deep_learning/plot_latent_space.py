import os
import sys
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from dataset_emg import EMGDataset
    from modelos import ConvAutoencoder1D
except ImportError:
    from deep_learning.dataset_emg import EMGDataset
    from deep_learning.modelos import ConvAutoencoder1D

def plot_latent_space(csv_path, model_path, latent_dim=16, train_sessions=None, test_sessions=None, out_dir=None):
    print(f"==================================================")
    print(f"Generando Espacio Latente y Métricas...")
    print(f"Usando archivo de evaluación: {os.path.abspath(csv_path)}")
    print(f"==================================================")
    print("Cargando modelo y extrayendo espacio latente...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar dataset (sin augmentation para evaluar las muestras reales)
    dataset = EMGDataset(csv_path, target_length=None, apply_augmentation=False)
    inferred_target_length = dataset.tensors.shape[2]
    
    # Cargar pesos del modelo y deducir dimensiones reales del archivo .pth
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint if isinstance(checkpoint, dict) else checkpoint.state_dict()
    
    if 'encoder_fc.3.weight' in state_dict:
        latent_dim = state_dict['encoder_fc.3.weight'].shape[0]
    if 'encoder_fc.0.weight' in state_dict:
        target_length = state_dict['encoder_fc.0.weight'].shape[1] // 32
    else:
        target_length = inferred_target_length
        
    kernel_size = state_dict['encoder_cnn.0.weight'].shape[2] if 'encoder_cnn.0.weight' in state_dict else 5
        
    print(f"Dimensiones activas -> Latent Dim: {latent_dim}D, Target Length por Canal: {target_length}, Kernel Size: {kernel_size}")
    
    # Si la dimensión del dataset no coincide con el target_length del modelo, recargar con ese target
    if dataset.tensors.shape[2] != target_length:
        dataset = EMGDataset(csv_path, target_length=target_length, apply_augmentation=False)
    
    model = ConvAutoencoder1D(latent_dim=latent_dim, target_length=target_length, kernel_size=kernel_size).to(device)
    model.load_state_dict(state_dict)
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
            pred_str = idx_to_vocal.get(pred_idx, "Unknown")
            
            latents.append(latent.cpu().numpy().squeeze())
            labels_true.append(y_str)
            labels_pred.append(pred_str)
            
    latents = np.array(latents)
    if latents.ndim == 1:
        latents = latents.reshape(-1, 1)
    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    
    # ---------------------------------------------------------
    # PARTICIÓN DE TRAIN/TEST (Manual o split_config.json)
    # ---------------------------------------------------------
    todas_las_tomas = dataset.tomas
    
    train_sessions_names = [os.path.basename(s).strip() for s in (train_sessions or []) if s and str(s).strip()]
    test_sessions_names = [os.path.basename(s).strip() for s in (test_sessions or []) if s and str(s).strip()]
    
    if not train_sessions_names and not test_sessions_names:
        model_dir = os.path.dirname(os.path.abspath(model_path))
        cfg_file = os.path.join(model_dir, "split_config.json")
        if os.path.exists(cfg_file):
            try:
                import json
                with open(cfg_file, "r") as f:
                    cfg = json.load(f)
                train_sessions_names = [os.path.basename(s).strip() for s in cfg.get("train_sessions", []) if s and str(s).strip()]
                test_sessions_names = [os.path.basename(s).strip() for s in cfg.get("test_sessions", []) if s and str(s).strip()]
                print(f"Cargada partición desde: {cfg_file}")
            except Exception as e:
                print(f"Aviso: No se pudo leer {cfg_file}: {e}")

    train_indices = []
    test_indices = []

    if train_sessions_names or test_sessions_names:
        print(f"Aplicando partición específica de sesiones:")
        print(f"  -> Sesiones Train: {train_sessions_names}")
        print(f"  -> Sesiones Test:  {test_sessions_names}")
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
                        test_indices.append(i)
                        asignado = True
                        break
        if not train_indices:
            print("[Advertencia] No se encontraron muestras de Train con las sesiones dadas. Se asignan todas a Train.")
            train_indices = list(range(len(todas_las_tomas)))
        if not test_indices:
            print("[Aviso] No hay sesiones en Test. Se usarán las de Train para evaluación.")
            test_indices = train_indices
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
        
        np.random.seed(42)
        np.random.shuffle(sesiones_unicas)
        
        train_sesiones_size = max(1, int(0.8 * len(sesiones_unicas)))
        train_sesiones = set(sesiones_unicas[:train_sesiones_size])
        val_sesiones = set(sesiones_unicas[train_sesiones_size:])
        if not val_sesiones:
            val_sesiones = train_sesiones
            
        train_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in train_sesiones]
        test_indices = [i for i, sesion in enumerate(sesiones_base) if sesion in val_sesiones]
    
    latents_train = latents[train_indices]
    labels_train = labels_true[train_indices]
    
    latents_test = latents[test_indices]
    labels_test = labels_true[test_indices]
    
    print(f"Total: {len(latents)} | Train: {len(latents_train)} | Test: {len(latents_test)}")
    
    # Calcular Silhouette sobre el espacio latente completo
    if len(set(labels_train)) > 1 and latents_train.shape[1] > 1:
        sil_score_train = silhouette_score(latents_train, labels_train, metric='euclidean')
        print(f"Silhouette Score Train (Latent Space {latent_dim}D): {sil_score_train:.4f}")
    else:
        sil_score_train = 0.0
    
    # Proyección adaptativa: Si es menor a 4D (2D o 3D) NO se usa UMAP
    actual_latent_dim = latents.shape[1]
    if actual_latent_dim < 4:
        print(f"Dimensión latente {actual_latent_dim}D (< 4D): Ploteo directo de coordenadas sin UMAP.")
        latents_train_proj = latents_train
        latents_test_proj = latents_test
        if actual_latent_dim == 2:
            is_3d_plot = False
            proj_name = "Espacio Latente 2D Directo (Sin UMAP)"
        elif actual_latent_dim == 3:
            is_3d_plot = True
            proj_name = "Espacio Latente 3D Directo (Sin UMAP)"
        else:
            is_3d_plot = False
            proj_name = f"Espacio Latente {actual_latent_dim}D (Sin UMAP)"
    else:
        print(f"Aplicando UMAP para reducción a 3D de espacio {actual_latent_dim}D (>= 4D)...")
        n_comp = min(3, len(latents_train)-1) if len(latents_train) > 3 else 2
        reducer = umap.UMAP(n_components=n_comp, random_state=42, min_dist=0.1)
        latents_train_proj = reducer.fit_transform(latents_train)
        latents_test_proj = reducer.transform(latents_test)
        is_3d_plot = (n_comp == 3)
        proj_name = f"UMAP {n_comp}D ({actual_latent_dim}D)"
        
    sil_proj_train = silhouette_score(latents_train_proj, labels_train, metric='euclidean') if len(set(labels_train)) > 1 and latents_train_proj.shape[1] > 1 else 0.0
    sil_proj_test = silhouette_score(latents_test_proj, labels_test, metric='euclidean') if len(set(labels_test)) > 1 and latents_test_proj.shape[1] > 1 else 0.0
    print(f"Silhouette Score Train ({proj_name}): {sil_proj_train:.4f}")
    print(f"Silhouette Score Test ({proj_name}): {sil_proj_test:.4f}")
    
    # Plotear (Dos gráficos: Train vs Test)
    fig = plt.figure(figsize=(18, 8))
    
    vocales = sorted(list(set(labels_true)))
    palette = sns.color_palette("Set1", n_colors=len(vocales))
    
    if is_3d_plot:
        ax1 = fig.add_subplot(121, projection='3d')
        for i, vocal in enumerate(vocales):
            idx = labels_train == vocal
            ax1.scatter(
                latents_train_proj[idx, 0], latents_train_proj[idx, 1], latents_train_proj[idx, 2], 
                label=vocal, color=palette[i], s=40, edgecolor='k', alpha=0.8
            )
        ax1.set_title(f"{proj_name} (Train Set)\nSil Train: {sil_proj_train:.4f}")
        ax1.set_xlabel('$z_1$' if actual_latent_dim == 3 else 'UMAP 1')
        ax1.set_ylabel('$z_2$' if actual_latent_dim == 3 else 'UMAP 2')
        ax1.set_zlabel('$z_3$' if actual_latent_dim == 3 else 'UMAP 3')
        
        ax2 = fig.add_subplot(122, projection='3d')
        for i, vocal in enumerate(vocales):
            idx = labels_test == vocal
            ax2.scatter(
                latents_test_proj[idx, 0], latents_test_proj[idx, 1], latents_test_proj[idx, 2], 
                label=vocal, color=palette[i], s=40, edgecolor='k', alpha=0.8
            )
        ax2.set_title(f"{proj_name} (Test Set Ciego)\nSil Test: {sil_proj_test:.4f}")
        ax2.set_xlabel('$z_1$' if actual_latent_dim == 3 else 'UMAP 1')
        ax2.set_ylabel('$z_2$' if actual_latent_dim == 3 else 'UMAP 2')
        ax2.set_zlabel('$z_3$' if actual_latent_dim == 3 else 'UMAP 3')
    else:
        ax1 = fig.add_subplot(121)
        for i, vocal in enumerate(vocales):
            idx = labels_train == vocal
            ax1.scatter(
                latents_train_proj[idx, 0], latents_train_proj[idx, 1], 
                label=vocal, color=palette[i], s=50, edgecolor='k', alpha=0.8
            )
        ax1.set_title(f"{proj_name} (Train Set)\nSil Train: {sil_proj_train:.4f}")
        ax1.set_xlabel('$z_1$')
        ax1.set_ylabel('$z_2$')
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(122)
        for i, vocal in enumerate(vocales):
            idx = labels_test == vocal
            ax2.scatter(
                latents_test_proj[idx, 0], latents_test_proj[idx, 1], 
                label=vocal, color=palette[i], s=50, edgecolor='k', alpha=0.8
            )
        ax2.set_title(f"{proj_name} (Test Set Ciego)\nSil Test: {sil_proj_test:.4f}")
        ax2.set_xlabel('$z_1$')
        ax2.set_ylabel('$z_2$')
        ax2.grid(True, alpha=0.3)
        
    plt.legend()
    
    base_repo_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    central_dir = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder")
    
    if out_dir is None:
        sujeto_detectado = "General"
        for t in dataset.tomas:
            parts = t.rsplit('_Win', 1)[0].split('_')
            if len(parts) >= 3:
                sujeto_detectado = parts[-1]
                break
        import datetime
        fecha_detectada = datetime.datetime.now().strftime("%Y-%m-%d")
        out_dir = os.path.join(central_dir, f"{fecha_detectada}_{sujeto_detectado}")
        
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(central_dir, exist_ok=True)
    
    plot_path = os.path.join(out_dir, f"latent_space_{actual_latent_dim}d.png")
    plt.savefig(plot_path)
    if out_dir != central_dir:
        plt.savefig(os.path.join(central_dir, f"latent_space_{actual_latent_dim}d.png"))
    print(f"Gráfico guardado en:\n  -> {plot_path}")
    
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
    acc_plot_path = os.path.join(out_dir, f"accuracy_por_vocal_{actual_latent_dim}d.png")
    plt.savefig(acc_plot_path)
    if out_dir != central_dir:
        plt.savefig(os.path.join(central_dir, f"accuracy_por_vocal_{actual_latent_dim}d.png"))
    print(f"Gráficos de precisión guardados en: {acc_plot_path}")
    
    # Guardar reporte de métricas en JSON
    import json
    metricas_dict = {
        "dimension_latente": int(actual_latent_dim),
        "accuracy_global_test_pct": float(acc_global),
        "accuracy_por_vocal": {str(v): float(acc) for v, acc in zip(vocales, acc_por_vocal)},
        "silhouette_train": float(sil_proj_train),
        "silhouette_test": float(sil_proj_test),
        "matriz_confusion": cm.tolist()
    }
    json_path = os.path.join(out_dir, f"metricas_evaluacion_{actual_latent_dim}d.json")
    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(metricas_dict, f, indent=4)
    if out_dir != central_dir:
        with open(os.path.join(central_dir, f"metricas_evaluacion_{actual_latent_dim}d.json"), "w", encoding='utf-8') as f:
            json.dump(metricas_dict, f, indent=4)
    print(f"Métricas guardadas en: {json_path}")
    
    # Cerrar las figuras de matplotlib para liberar memoria
    plt.close('all')
    
    # Abrir los PNG generados con el visor del sistema (no bloqueante)
    import subprocess
    for img_path in [plot_path, acc_plot_path]:
        if os.path.exists(img_path):
            try:
                if os.name == 'nt':
                    os.startfile(img_path)
                else:
                    subprocess.Popen(["xdg-open", img_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
    
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "..", "analysis", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
    model_file = os.path.join(base_dir, "autoencoder_emg_16d.pth")
    
    if not os.path.exists(model_file):
        print(f"No se encontró el modelo entrenado en {model_file}")
        print("Por favor corré train_autoencoder.py primero.")
    else:
        plot_latent_space(csv_file, model_file, latent_dim=16)
