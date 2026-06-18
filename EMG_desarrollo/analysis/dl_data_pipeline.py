# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Pipeline Batch Processing de señales sEMG para Deep Learning.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Pipeline Batch Processing de señales sEMG para Deep Learning.
# ==============================================================================

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
import torch
from torch.utils.data import Dataset

# ------------------------------------------------------------------
# CONFIGURACIÓN DSP (Heredada de analisis_por_track_integrado.py)
# ------------------------------------------------------------------
FS = 10000  # Frecuencia de muestreo por defecto
TARGET_SAMPLES = 500  # Para la estandarización tensorial

def apply_bandpass_filter(data, lowcut=20.0, highcut=500.0, fs=FS, order=4):
    """ Filtro Pasa-Banda Butterworth (Fase cero para evitar desfase de picos) """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def apply_notch_filter(data, w0=50.0, Q=2.0, fs=FS):
    """ Filtro Notch para eliminar ruido de línea de 50Hz """
    b, a = iirnotch(w0, Q, fs)
    return filtfilt(b, a, data)

def calculate_rms_envelope(data, window_ms=50, fs=FS):
    """ Cálculo de envolvente RMS mediante convolución """
    window_length = int((window_ms / 1000.0) * fs)
    # Elevar al cuadrado, media móvil, raíz cuadrada
    squared = np.power(data, 2)
    window = np.ones(window_length) / float(window_length)
    rms = np.sqrt(np.convolve(squared, window, 'same'))
    return rms

# ------------------------------------------------------------------
# CLASE: Pipeline de Preprocesamiento EMG para ML
# ------------------------------------------------------------------
class EMG_DeepLearning_Pipeline:
    def __init__(self, base_dir, output_dir="datasets_ml"):
        self.base_dir = base_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Diccionario para mapear Sílabas a IDs numéricos
        self.labels_map = {}
        self.current_label_id = 0
        self.dataset_index = []

    def get_label_id(self, silaba):
        if silaba not in self.labels_map:
            self.labels_map[silaba] = self.current_label_id
            self.current_label_id += 1
        return self.labels_map[silaba]

    def process_all_subjects(self):
        print(f"🔄 Iniciando Ingestión Batch en: {self.base_dir}")
        
        carpetas_sujetos = [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
        
        for sujeto in carpetas_sujetos:
            sujeto_path = os.path.join(self.base_dir, sujeto)
            archivos_csv = glob.glob(os.path.join(sujeto_path, "*.csv"))
            
            for csv_file in archivos_csv:
                # Ignorar archivos auxiliares como etiquetas.csv
                if "etiquetas" in os.path.basename(csv_file).lower():
                    continue
                    
                self._process_single_recording(csv_file)
                
        # Guardar metadatos e index
        self._save_metadata()

    def _process_single_recording(self, csv_file):
        """ Ingestión, Limpieza y Alineación de un registro EMG completo """
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"⚠️ Error leyendo {csv_file}: {e}")
            return

        # Asumimos columnas: Time, Ch1, Ch2, Ch3 (Ajustar según hardware real)
        cols = [c for c in df.columns if 'ch' in c.lower()]
        if len(cols) < 3:
            return # Se necesitan los 3 canales
            
        # Extraer etiqueta del nombre del archivo (Sujeto_SÍLABA_tomaX)
        filename = os.path.basename(csv_file)
        parts = filename.split('_')
        if len(parts) >= 2:
            silaba = parts[1]
        else:
            silaba = "UNKNOWN"
            
        label_id = self.get_label_id(silaba)

        # 1. LIMPIEZA ACÚSTICA
        channels_clean = []
        for c in cols[:3]:
            raw_signal = df[c].values
            
            # Filtros
            bandpassed = apply_bandpass_filter(raw_signal)
            notched = apply_notch_filter(bandpassed)
            
            # Envolvente RMS
            rms = calculate_rms_envelope(notched)
            channels_clean.append(rms)

        # 2. SEGMENTACIÓN Y ALINEACIÓN (Master-Slave simplificado para batch)
        # En una implementación completa de Fase 2, acá leeríamos el etiquetas.csv
        # Para el proof-of-concept del pipeline tensorial, cortamos la ventana alrededor del pico máximo del Master
        
        master_ch = channels_clean[0] # Ej: Ch1 como master
        peak_idx = np.argmax(master_ch)
        
        # Tomar ventana de 1 segundo alrededor del pico (fs=10000 -> 10000 muestras)
        window_size = int(FS * 1.0) 
        half_window = window_size // 2
        
        start_idx = max(0, peak_idx - half_window)
        end_idx = min(len(master_ch), peak_idx + half_window)
        
        aligned_channels = []
        for ch in channels_clean:
            segment = ch[start_idx:end_idx]
            aligned_channels.append(segment)
            
        # 3. ESTANDARIZACIÓN TENSORIAL
        # Resampling a 500 muestras
        tensor_channels = []
        for seg in aligned_channels:
            if len(seg) > 0:
                resampled = signal.resample(seg, TARGET_SAMPLES)
                # Normalización Min-Max [0, 1]
                min_val = np.min(resampled)
                max_val = np.max(resampled)
                if max_val - min_val > 0:
                    normalized = (resampled - min_val) / (max_val - min_val)
                else:
                    normalized = resampled
                tensor_channels.append(normalized)
            else:
                tensor_channels.append(np.zeros(TARGET_SAMPLES))

        # Empaquetado en Tensor (3, 500)
        emg_tensor = torch.tensor(np.array(tensor_channels), dtype=torch.float32)
        
        # 4. EXPORTACIÓN
        tensor_filename = filename.replace('.csv', '.pt')
        tensor_path = os.path.join(self.output_dir, tensor_filename)
        torch.save(emg_tensor, tensor_path)
        
        # Registrar en index
        self.dataset_index.append({
            "file": tensor_filename,
            "label_id": label_id,
            "silaba": silaba
        })
        
    def _save_metadata(self):
        index_path = os.path.join(self.output_dir, "dataset_index.json")
        with open(index_path, 'w') as f:
            json.dump({
                "labels_map": self.labels_map,
                "data": self.dataset_index
            }, f, indent=4)
        print(f"✅ Pipeline Finalizado. Índices guardados en {index_path}")

# ------------------------------------------------------------------
# DATASET DE PYTORCH
# ------------------------------------------------------------------
class EMGDataset(Dataset):
    def __init__(self, index_file_path):
        with open(index_file_path, 'r') as f:
            meta = json.load(f)
            
        self.data_dir = os.path.dirname(index_file_path)
        self.samples = meta['data']
        self.labels_map = meta['labels_map']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        file_path = os.path.join(self.data_dir, item['file'])
        
        # Cargar el tensor de PyTorch directamente
        tensor = torch.load(file_path)
        
        label = torch.tensor(item['label_id'], dtype=torch.long)
        return tensor, label

if __name__ == "__main__":
    # Prueba del script
    # pipeline = EMG_DeepLearning_Pipeline(base_dir="../base_de_datos_electrodos")
    # pipeline.process_all_subjects()
    pass
