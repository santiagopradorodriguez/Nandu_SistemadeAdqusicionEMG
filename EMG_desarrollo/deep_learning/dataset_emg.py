import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EMGDataset(Dataset):
    """
    Carga el CSV de características exportadas (aplanadas a N x 300) y 
    las devuelve como tensores de PyTorch de forma (3, 100).
    """
    def __init__(self, csv_path, target_length=100, apply_augmentation=False):
        self.data = pd.read_csv(csv_path)
        print(f"[Dataset EMG] Archivo cargado exitosamente desde: {os.path.abspath(csv_path)}")
        print(f"[Dataset EMG] Dimensiones leídas (Filas, Columnas): {self.data.shape}")
        
        # Las columnas 0 y 1 son 'Vocal' y 'Toma'
        self.labels = self.data['Vocal'].values
        self.tomas = self.data['Toma'].values
        
        # El resto son las características (300 columnas)
        features = self.data.iloc[:, 2:].values
        
        # Remodelamos a (N, 3, 100)
        self.tensors = features.reshape(-1, 3, target_length)
        print(f"[Dataset EMG] Dataset convertido exitosamente a Tensor de forma: {self.tensors.shape}")
        
        # Mapeo de vocales a enteros (opcional, para si queremos clasificar después)
        vocales_unicas = sorted(list(set(self.labels)))
        self.label_to_idx = {v: i for i, v in enumerate(vocales_unicas)}
        
        self.apply_augmentation = apply_augmentation

    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        x = self.tensors[idx].astype(np.float32)
        y_str = self.labels[idx]
        y_idx = self.label_to_idx[y_str]
        
        # Data Augmentation (Validado por el Agente Físico - Mundo Real)
        if self.apply_augmentation:
            # 1. Escalamiento Proporcional (Variaciones de fuerza natural)
            escala = np.random.uniform(0.85, 1.15)
            x = x * escala
            
            # 2. Ruido Gaussiano (Simula ruido de piso del amplificador / piel)
            # Acotado pero presente para forzar a la red a no memorizar valores exactos
            noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
            x = x + noise
            
            # 3. Time Masking Abrupto (Simulación de fallas de contacto/ADC)
            # Validado físicamente: enseña a la red a interpolar la inercia real del músculo
            # frente a un dropout de telemetría o falso contacto.
            if np.random.rand() > 0.5:
                mask_len = np.random.randint(2, 6)
                mask_start = np.random.randint(0, x.shape[1] - mask_len)
                x[:, mask_start:mask_start+mask_len] = 0.0
            
            x = np.clip(x, 0, 3.0) # Clampeamos seguridad extrema
            
        return torch.tensor(x), torch.tensor(y_idx), y_str
