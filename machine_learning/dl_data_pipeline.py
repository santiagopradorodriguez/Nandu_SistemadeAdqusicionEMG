import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt, correlate, correlation_lags, iirnotch
from scipy.interpolate import interp1d

class EMGDSPPipeline:
    def __init__(self, fs=2000, lowcut=20.0, highcut=500.0, rms_window_ms=50.0, target_length=500, apply_notch=True):
        """
        Pipeline de Pre-Procesamiento DSP para señales sEMG.
        
        Args:
            fs (int): Frecuencia de muestreo (Hz).
            lowcut (float): Frecuencia de corte inferior pasa-banda (Hz).
            highcut (float): Frecuencia de corte superior pasa-banda (Hz).
            rms_window_ms (float): Tamaño de la ventana de envolvente RMS (ms).
            target_length (int): Longitud constante para el tensor resultante.
            apply_notch (bool): Si es True, aplica filtro Notch 50Hz para ruido de línea eléctrica.
        """
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.rms_window_ms = rms_window_ms
        self.target_length = target_length
        self.apply_notch = apply_notch

    def subtract_offset(self, signal):
        """1. Resta del Offset: Calcular la media estática de la señal y restarla para centrar en cero."""
        return signal - np.mean(signal)

    def apply_notch_filter(self, signal, freq=50.0, q=30.0):
        """Filtro Notch para penalizar ruido de línea (50Hz) y evitar artefactos."""
        b, a = iirnotch(freq, q, self.fs)
        padlen = min(len(signal) - 1, int(self.fs * 0.1))
        if padlen > 3:
            return filtfilt(b, a, signal, padlen=padlen)
        return filtfilt(b, a, signal)

    def bandpass_filter(self, signal, order=4):
        """2. Filtro Pasa-banda: 20-500 Hz (Butterworth)."""
        nyq = 0.5 * self.fs
        low = self.lowcut / nyq
        high = self.highcut / nyq
        if high >= 1.0:
            high = 0.99
            
        b, a = butter(order, [low, high], btype='band')
        padlen = min(len(signal) - 1, int(self.fs * 0.1))
        if padlen > 3:
            return filtfilt(b, a, signal, padlen=padlen)
        return filtfilt(b, a, signal)

    def rms_envelope(self, signal):
        """3. Envolvente RMS: Ventana de 50ms (por defecto)."""
        win_len = int((self.rms_window_ms / 1000.0) * self.fs)
        if win_len < 1:
            win_len = 1
            
        sig_sq = signal ** 2
        window = np.ones(win_len) / float(win_len)
        # Se utiliza mode='same' para no generar artefactos en los bordes y conservar el largo
        rms_sq = np.convolve(sig_sq, window, mode='same')
        
        # Evitar valores negativos residuales por precisión de punto flotante antes de np.sqrt
        return np.sqrt(np.maximum(rms_sq, 0.0))

    def normalize_by_snr(self, signal, noise_rms=None):
        """
        4. Normalización por SNR Inter-pulso: 
        Normalizar la señal usando la métrica de ruido base RMS.
        """
        if noise_rms is None:
            # Si no se provee el ruido inter-pulso explícitamente, se estima del 10% más bajo de amplitud local
            sorted_sig = np.sort(np.abs(signal))
            noise_len = max(1, int(0.1 * len(sorted_sig)))
            noise_rms = np.sqrt(np.mean(sorted_sig[:noise_len]**2))
            
        if noise_rms > 1e-8:
            return signal / noise_rms
        return signal

    def align_master_slave(self, slave_signal, master_signal):
        """
        5. Alineación de Fase Master-Slave: 
        Utilizar Cross-Correlation para alinear temporalmente el pico máximo.
        Basado en logic de 'correlaciondeseñales.py'.
        """
        if master_signal is None:
            return slave_signal
            
        corr = correlate(slave_signal, master_signal, mode='same')
        lags = correlation_lags(len(slave_signal), len(master_signal), mode='same')
        lag = lags[np.argmax(corr)]
        shift_val = -int(lag)
        
        aligned_signal = np.roll(slave_signal, shift_val)
        return aligned_signal

    def tensorize_and_normalize(self, signal):
        """
        6. Tensorización: 
        Resampling a longitud constante (500 muestras por defecto) y normalización Min-Max (0.0 a 1.0).
        Esto deja los tensores normalizados y estacionarios para el Autoencoder/modelos PyTorch.
        """
        if len(signal) != self.target_length:
            old_indices = np.linspace(0, 1, len(signal))
            new_indices = np.linspace(0, 1, self.target_length)
            f = interp1d(old_indices, signal, kind='linear', fill_value="extrapolate")
            signal = f(new_indices)
            
        min_val = np.min(signal)
        max_val = np.max(signal)
        
        if max_val > min_val:
            normalized_signal = (signal - min_val) / (max_val - min_val)
        else:
            normalized_signal = signal - min_val
            
        # Añadimos dimension (1, target_length) asumiendo canal único (1D feature map)
        tensor_out = torch.tensor(normalized_signal, dtype=torch.float32).unsqueeze(0)
        return tensor_out

    def process(self, signal, master_signal=None, noise_rms=None):
        """Ejecuta el pipeline completo secuencialmente sobre un numpy array."""
        # 1. Offset
        sig = self.subtract_offset(signal)
        
        # 1b. Notch (opcional para ruido línea)
        if self.apply_notch:
            sig = self.apply_notch_filter(sig)
            
        # 2. Bandpass
        sig = self.bandpass_filter(sig)
        
        # 3. Envolvente RMS
        sig = self.rms_envelope(sig)
        
        # 4. Normalización por SNR
        sig = self.normalize_by_snr(sig, noise_rms)
        
        # 5. Alineación de Fase Master-Slave
        sig = self.align_master_slave(sig, master_signal)
        
        # 6. Tensorización y Normalización Min-Max
        tensor_sig = self.tensorize_and_normalize(sig)
        
        return tensor_sig


class EMGDataset(Dataset):
    def __init__(self, data_paths, target_labels=None, pipeline=None, master_paths=None, noise_rms_list=None):
        """
        Dataset en PyTorch que hereda de torch.utils.data.Dataset para cargar los arrays .npy,
        pasarlos por la pipeline y asignar el target numérico de vocales.
        
        Args:
            data_paths (list): Lista de rutas a los archivos .npy con los arrays de señal.
            target_labels (list): Lista de etiquetas correspondientes (ej: 'A', 'E', 'I', 'O', 'U').
            pipeline (EMGDSPPipeline): Instancia del pipeline de procesamiento DSP.
            master_paths (list): Opcional. Rutas a los arrays .npy de las señales master.
            noise_rms_list (list): Opcional. Valores RMS del ruido base inter-pulso para SNR.
        """
        self.data_paths = data_paths
        self.target_labels = target_labels
        self.pipeline = pipeline if pipeline is not None else EMGDSPPipeline()
        self.master_paths = master_paths
        self.noise_rms_list = noise_rms_list
        
        # Target mapping explícito solicitado por el usuario
        self.label_map = {'A': 1, 'E': 2, 'I': 3, 'O': 4, 'U': 5}

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        signal = np.load(path)
        
        # Procesamiento en línea de la señal Master para cross-correlación
        master_sig = None
        if self.master_paths and self.master_paths[idx]:
            master_sig_raw = np.load(self.master_paths[idx])
            # La señal master necesita procesamiento de envolvente para compararla correctamente
            master_sig = self.pipeline.subtract_offset(master_sig_raw)
            if self.pipeline.apply_notch:
                master_sig = self.pipeline.apply_notch_filter(master_sig)
            master_sig = self.pipeline.bandpass_filter(master_sig)
            master_sig = self.pipeline.rms_envelope(master_sig)
            
        # Nivel de ruido opcional provisto por fase 3
        noise_rms = None
        if self.noise_rms_list is not None and self.noise_rms_list[idx] is not None:
            noise_rms = self.noise_rms_list[idx]

        # Aplicar el DSP
        tensor_sig = self.pipeline.process(signal, master_signal=master_sig, noise_rms=noise_rms)
        
        # Asignar target
        if self.target_labels is not None:
            label_str = str(self.target_labels[idx]).upper().strip()
            target = self.label_map.get(label_str, 0)
            return tensor_sig, torch.tensor(target, dtype=torch.long)
        
        return tensor_sig
