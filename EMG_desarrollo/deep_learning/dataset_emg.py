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
    def __init__(self, csv_path, target_length=None, apply_augmentation=False):
        self.data = pd.read_csv(csv_path)
        print(f"[Dataset EMG] Archivo cargado exitosamente desde: {os.path.abspath(csv_path)}")
        print(f"[Dataset EMG] Dimensiones leídas (Filas, Columnas): {self.data.shape}")
        
        # Las columnas 0 y 1 son 'Vocal' y 'Toma'
        self.labels = self.data['Vocal'].values
        self.tomas = self.data['Toma'].values
        
        # El resto son las características (300 columnas)
        features = self.data.iloc[:, 2:].values
        
        # Inferir target_length si no se pasa
        if target_length is None:
            target_length = features.shape[1] // 3
            
        # Remodelamos a (N, 3, target_length)
        raw_tensors = features.reshape(-1, 3, target_length)
        
        # Acondicionamiento Fisiológico del Modelo Récord (Butterworth + Resta Reposo + P95)
        self.tensors = self._acondicionar_tensores_record(raw_tensors)
        print(f"[Dataset EMG] Dataset acondicionado según norma récord a Tensor: {self.tensors.shape}")
        
        # Mapeo de vocales a enteros
        vocales_unicas = sorted(list(set(self.labels)))
        self.label_to_idx = {v: i for i, v in enumerate(vocales_unicas)}
        
        self.apply_augmentation = apply_augmentation

    def _acondicionar_tensores_record(self, tensors):
        """Aplica el filtrado Butterworth 3 (Wn=0.3), sustracción de base inicial y división por P95 del récord."""
        from scipy.signal import butter, filtfilt
        N, n_ch, n_pts = tensors.shape
        b_bw, a_bw = butter(3, 0.3, btype='low')
        X_filt = np.zeros_like(tensors)
        for i in range(N):
            for c in range(n_ch):
                X_filt[i, c, :] = filtfilt(b_bw, a_bw, tensors[i, c, :])

        X_norm = np.zeros_like(X_filt)
        n_base = max(2, min(10, n_pts // 2))
        for i in range(N):
            for c in range(n_ch):
                base_mean = np.mean(X_filt[i, c, :n_base])
                p95 = np.percentile(X_filt[i, c, :], 95)
                scale = p95 - base_mean
                if scale < 1e-4:
                    scale = np.max(X_filt[i, c, :]) - base_mean
                if scale < 1e-6:
                    scale = 1.0
                X_norm[i, c, :] = (X_filt[i, c, :] - base_mean) / scale

        return X_norm

    def __len__(self):
        return len(self.tensors)
        
    def __getitem__(self, idx):
        x = self.tensors[idx].astype(np.float32)
        y_str = self.labels[idx]
        y_idx = self.label_to_idx[y_str]
        
        # Data Augmentation suave (preserva la continuidad de atractores sin deformar el manifold)
        if self.apply_augmentation:
            # 1. Escalamiento Proporcional suave (0.92 a 1.08)
            escala = np.random.uniform(0.92, 1.08)
            x = x * escala
            
            # 2. Ruido Gaussiano sutil
            noise = np.random.normal(0, 0.005, x.shape).astype(np.float32)
            x = x + noise
            
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y_idx, dtype=torch.long), y_str
