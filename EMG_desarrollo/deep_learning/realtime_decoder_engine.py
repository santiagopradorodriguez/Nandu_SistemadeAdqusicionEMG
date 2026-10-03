# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor de decodificación bioeléctrica en streaming en tiempo real
#              con detector Gate Doble, proyección latente y fronteras de decisión.
# ==============================================================================

import os
import sys
import json
import time
import numpy as np
from scipy import signal
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_A_COLOR_HEX = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}

# Colores en formato RGBA normalizado (0.0 a 1.0)
VOCAL_A_RGBA = {
    'A': (0.902, 0.224, 0.275, 1.0),
    'E': (0.122, 0.467, 0.706, 1.0),
    'I': (0.173, 0.627, 0.173, 1.0),
    'O': (0.616, 0.306, 0.867, 1.0),
    'U': (0.906, 0.651, 0.102, 1.0)
}

# ==============================================================================
# 1. ARQUITECTURA DE AUTOENCODER CONVOLUCIONAL ORTOGONAL (PARAMETRIC)
# ==============================================================================
class ParametricConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2, act_name='tanh'):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)

        if act_name == 'tanh':
            self.act = nn.Tanh()
        elif act_name == 'gelu':
            self.act = nn.GELU()
        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()

        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)

        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

    def encode(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        return self.fc2(h3)

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.time_pts)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        z = self.fc2(h3)

        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        recon_3d = self.deconv2(dh3)
        recon_flat = recon_3d.view(recon_3d.shape[0], -1)
        return recon_flat, z


class OrthogonalAutoencoder2DSemi(nn.Module):
    """
    Autoencoder Ortogonal 2D con regularización de Entropía Cruzada (Hito 115 / Apunte LaTeX).
    Densa 60 -> 32 -> 16 -> 2 -> 16 -> 32 -> 60 con cabeza de clasificación Softmax auxiliar.
    """
    def __init__(self, input_dim=60, hidden_dim=32, latent_dim=2, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 16, bias=False)
        self.fc3 = nn.Linear(16, latent_dim, bias=False)
        self.act = nn.Tanh()
        self.dfc1 = nn.Linear(latent_dim, 16, bias=False)
        self.dfc2 = nn.Linear(16, hidden_dim, bias=False)
        self.dfc3 = nn.Linear(hidden_dim, input_dim, bias=False)
        self.classifier = nn.Linear(latent_dim, num_classes, bias=True)

    def encode(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x):
        h1 = self.act(self.fc1(x))
        h2 = self.act(self.fc2(h1))
        z = self.fc3(h2)
        logits = self.classifier(z)
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        recon = self.dfc3(dh2)
        return recon, z, logits


class ParametricMLPOrthogonalAE(nn.Module):
    """
    Autoencoder MLP Ortogonal Simétrico sin sesgos (D -> h1 -> h2 -> latent_dim -> h2 -> h1 -> D).
    """
    def __init__(self, input_dim=60, hidden_dims=(48, 24), latent_dim=3, act_name='tanh'):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1, bias=False)
        self.fc2 = nn.Linear(h1, h2, bias=False)
        self.fc3 = nn.Linear(h2, latent_dim, bias=False)
        if act_name == 'tanh':
            self.act = nn.Tanh()
        elif act_name == 'gelu':
            self.act = nn.GELU()
        elif act_name == 'leaky':
            self.act = nn.LeakyReLU(0.1)
        else:
            self.act = nn.ReLU()
        self.dfc1 = nn.Linear(latent_dim, h2, bias=False)
        self.dfc2 = nn.Linear(h2, h1, bias=False)
        self.dfc3 = nn.Linear(h1, input_dim, bias=False)

    def encode(self, x):
        if x.dim() == 3:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        h1 = self.act(self.fc1(x_flat))
        h2 = self.act(self.fc2(h1))
        return self.fc3(h2)

    def forward(self, x):
        if x.dim() == 3:
            x_flat = x.view(x.shape[0], -1)
        else:
            x_flat = x
        h1 = self.act(self.fc1(x_flat))
        h2 = self.act(self.fc2(h1))
        z = self.fc3(h2)
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1))
        recon_flat = self.dfc3(dh2)
        return recon_flat, z


# ==============================================================================
# 2. MOTOR DE DECODIFICACION EN TIEMPO REAL
SESSION_CALIBRATIONS = {
    'P5': {
        'base_mean': np.array([0.08416305, 0.13086277, 0.08983488], dtype=np.float32),
        'scale': np.array([0.30263156, 0.54462415, 0.57315177], dtype=np.float32)
    },
    'LUCAS': {
        'base_mean': np.array([0.03652162, 0.11596428, 0.08700229], dtype=np.float32),
        'scale': np.array([0.42615682, 0.39703524, 0.3273475], dtype=np.float32)
    }
}

# ==============================================================================
class RealtimeDecoderEngine:
    """
    Motor desacoplado de procesamiento bioeléctrico en tiempo real:
    - Buffers circulares de audio/EMG.
    - Filtrado IIR causal continuo (Notch 50 Hz + Pasa-Banda/Pasa-Bajos).
    - Envolvente RMS en línea (90 ms) y Norma Tricanal Combinada S_emg.
    - Máquina de estados Gate Doble con histéresis y backtracking hacia atrás.
    - Sincronización acústica opcional por micrófono (Canal 3) para grabaciones de prueba.
    - Extracción de ventana centrada por retardo electromecánico (EMD) o pico de audio.
    - Normalización obligatoria por Supremo Tricanal por Pulso Individual.
    - Inferencia PyTorch (< 0.1 ms) + Clasificación probabilística GMM.
    - Cálculo de mapa de regiones de decisión 2D.
    """

    def __init__(
        self,
        model_path=None,
        proyecciones_path=None,
        sample_rate=2000,
        buffer_sec=8.0,
        target_len=20,
        emd_delay_ms=350,
        pre_win_ms=800,
        post_win_ms=1200,
        device='cpu'
    ):
        self.fs = sample_rate
        self.target_len = target_len
        self.device = torch.device(device)
        self.n_canales = 3

        # Factores de estandarización por canal esperados por el modelo ortogonal
        # (Referencia de Lucas: compensación de impedancia y amplitud entre canales)
        self.channel_base_mean = np.array([0.03652162, 0.11596428, 0.08700229], dtype=np.float32)
        self.channel_scale = np.array([0.42615682, 0.39703524, 0.3273475], dtype=np.float32)

        # Parámetros temporales en muestras (Ventana 40/60 por defecto: 800 ms pre / 1200 ms post)
        self.buffer_len = int(buffer_sec * self.fs)
        self.emd_delay_ms = float(emd_delay_ms)
        self.emd_samples = int((self.emd_delay_ms / 1000.0) * self.fs)
        self.pre_samples = int((pre_win_ms / 1000.0) * self.fs)
        self.post_samples = int((post_win_ms / 1000.0) * self.fs)
        self.total_win_samples = self.pre_samples + self.post_samples
        self.rms_win_len = int(0.090 * self.fs)  # 90 ms
        self.t_refr_samples = int(0.800 * self.fs)  # 800 ms
        self.max_lookback_samples = int(0.350 * self.fs)  # 350 ms

        # Modelo y Clasificador
        self.model = None
        self.gmm = None
        self.cluster_to_vocal = {}
        self.cfg = {}
        self.latent_dim = 2

        # Puntos de referencia de Lucas
        self.ref_z1 = []
        self.ref_z2 = []
        self.ref_vocales = []

        # Estado de filtros IIR causales y zero-phase
        self._init_filters()

        # Buffers circulares y estado Gate Doble
        self.reset_state()

        # Cargar modelo si se proporcionó
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        # Cargar proyecciones de referencia si se proporcionó
        if proyecciones_path and os.path.exists(proyecciones_path):
            self.load_reference_projections(proyecciones_path)

    def _init_filters(self):
        """Inicializa coeficientes SOS y condiciones iniciales zi para filtrado IIR continuo."""
        # 1. Notch 50 Hz con Q=2.0
        b_notch, a_notch = signal.iirnotch(w0=50.0, Q=2.0, fs=self.fs)
        self.notch_sos = signal.tf2sos(b_notch, a_notch)
        self.notch_zi = np.zeros((self.notch_sos.shape[0], 2, self.n_canales))
        self.b_notch = b_notch
        self.a_notch = a_notch

        # 2. Pasa-banda Butterworth 20-500 Hz orden 2
        self.bp_sos = signal.butter(N=2, Wn=[20.0, 500.0], btype='bandpass', fs=self.fs, output='sos')
        self.bp_zi = np.zeros((self.bp_sos.shape[0], 2, self.n_canales))
        b_band, a_band = signal.butter(N=2, Wn=[20.0, 500.0], btype='bandpass', fs=self.fs)
        self.b_band = b_band
        self.a_band = a_band

        # 3. Kernel para envolvente RMS
        self.rms_kernel = np.ones(self.rms_win_len, dtype=np.float32) / self.rms_win_len

        # 4. Filtro pasa-bajos temporal (Butterworth orden 3, Wn=0.3) para los 20 puntos remuestreados
        self.b_temporal, self.a_temporal = signal.butter(N=3, Wn=0.3, btype='low')
        self.filt_initialized = False

        # 5. Kernel para detección de envolvente de micrófono (50 ms)
        self.win_mic = int(0.050 * self.fs)
        self.mic_kernel = np.ones(self.win_mic, dtype=np.float32) / self.win_mic

    def reset_state(self):
        """Reinicia los buffers de datos y la máquina de estados."""
        # Buffers de señal cruda y filtrada
        self.raw_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.filt_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.env_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.s_emg_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self.mic_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self.mic_sync_enabled = True
        self.ultimo_pico_mic_sample = -self.t_refr_samples

        # Contador global de muestras recibidas
        self.total_samples_received = 0
        self.filt_initialized = False

        # Umbrales bioeléctricos de Gate Doble
        self.u_bajo = 120.0
        self.u_alto = 650.0
        self.med_base = 60.0
        self.ruido_base_canales = np.zeros((self.n_canales, 1), dtype=np.float32)
        self.calibrado = False

        # Máquina de estados: 'CALIBRANDO', 'ARMADO', 'ESPERANDO_FIN_VENTANA', 'REFRACTARIO'
        self.estado_gate = 'CALIBRANDO'
        self.ultimo_onset_sample = -self.t_refr_samples
        self.onset_sample_actual = -1
        self.center_sample_actual = -1
        self.fin_sample_actual = -1

    def recalibrar_ruido(self, duracion_sec=5.0):
        """
        Calcula el piso de ruido basal, dispersión robusta (MAD) y umbrales a partir del buffer actual.
        Usa zero-phase filtfilt sobre el buffer crudo para garantizar máxima precisión bioeléctrica.
        """
        n_samples_calib = min(int(duracion_sec * self.fs), self.buffer_len)
        if self.total_samples_received < n_samples_calib:
            return False

        raw_seg = self.raw_buffer[:, -n_samples_calib:]
        try:
            seg_f = np.zeros_like(raw_seg)
            for c in range(self.n_canales):
                s_n = signal.filtfilt(self.b_notch, self.a_notch, raw_seg[c])
                seg_f[c] = signal.filtfilt(self.b_band, self.a_band, s_n)
            seg_e = np.zeros_like(seg_f)
            for c in range(self.n_canales):
                seg_e[c] = np.sqrt(np.maximum(0, signal.convolve(seg_f[c] ** 2, self.rms_kernel, mode='same')))
            self.ruido_base_canales = np.median(seg_e, axis=1, keepdims=True)
            env_seg = seg_e
        except Exception:
            env_seg = self.env_buffer[:, -n_samples_calib:]
            self.ruido_base_canales = np.median(env_seg, axis=1, keepdims=True)

        # 2. Recalcular la envolvente limpia y la norma S_emg en este segmento de calibración
        env_seg_clean = np.maximum(env_seg - self.ruido_base_canales, 0.0)
        s_seg = np.sqrt(np.sum(env_seg_clean ** 2, axis=0))
        self.s_emg_buffer[-n_samples_calib:] = s_seg

        # 3. Estadísticas basales robustas
        self.med_base = float(np.median(s_seg))
        mad_base = float(np.median(np.abs(s_seg - self.med_base)))
        sigma_rob = 1.4826 * mad_base + 1e-6

        # 4. Umbrales bioeléctricos de Gate Doble
        self.u_bajo = max(100.0, self.med_base + 3.0 * sigma_rob)
        self.u_alto = max(1100.0, min(1600.0, self.u_bajo * 2.2))

        self.calibrado = True
        if self.estado_gate == 'CALIBRANDO':
            self.estado_gate = 'ARMADO'
        return True

    def set_emd_delay_ms(self, emd_ms: float):
        """
        Actualiza dinámicamente el retardo fisiológico EMD en milisegundos.
        Alinea el centro de la ventana de análisis con respecto al inicio motor:
        n_centro = n_onset + EMD_lag
        """
        self.emd_delay_ms = max(0.0, float(emd_ms))
        self.emd_samples = int((self.emd_delay_ms / 1000.0) * self.fs)

    def set_rms_window_ms(self, win_ms: float):
        """
        Actualiza en tiempo real el tamaño de ventana para el suavizado de la envolvente RMS y S_emg.
        Permite alinear de forma idéntica el tiempo de respuesta temporal con el micrófono.
        """
        self.rms_win_ms = max(10.0, float(win_ms))
        self.rms_win_len = max(1, int((self.rms_win_ms / 1000.0) * self.fs))
        self.rms_kernel = np.ones(self.rms_win_len, dtype=np.float32) / self.rms_win_len

    def set_session_calibration(self, base_mean, scale):
        """Ajusta los factores de estandarización por canal para la sesión actual."""
        self.channel_base_mean = np.array(base_mean, dtype=np.float32)
        self.channel_scale = np.array(scale, dtype=np.float32)

    def set_calibration_preset(self, preset_name='P5'):
        """Aplica el perfil de calibración predeterminado ('P5' o 'LUCAS')."""
        key = str(preset_name).upper()
        if key in SESSION_CALIBRATIONS:
            p = SESSION_CALIBRATIONS[key]
            self.set_session_calibration(p['base_mean'], p['scale'])

    def set_mic_sync_enabled(self, enabled: bool):
        """Habilita o deshabilita la sincronización acústica por micrófono si Canal 3 está presente."""
        self.mic_sync_enabled = bool(enabled)

    def load_model(self, pt_path):
        """Carga el modelo PyTorch entrenado y reconstruye el clasificador GMM."""
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"No existe el archivo de modelo: {pt_path}")

        ckpt = torch.load(pt_path, map_location=self.device, weights_only=False)
        self.cfg = ckpt.get('cfg', {})

        # Factores de estandarización por canal si vienen en el checkpoint
        if 'channel_base_mean' in ckpt:
            self.channel_base_mean = np.array(ckpt['channel_base_mean'], dtype=np.float32)
        if 'channel_scale' in ckpt:
            self.channel_scale = np.array(ckpt['channel_scale'], dtype=np.float32)

        # Ajuste de ventana si viene en el checkpoint (ej. 40/60 o 50/50)
        if 'pre_pct' in self.cfg and 'post_pct' in self.cfg:
            total_dur_ms = 2000.0
            self.pre_samples = int(float(self.cfg['pre_pct']) * (total_dur_ms / 1000.0) * self.fs)
            self.post_samples = int(float(self.cfg['post_pct']) * (total_dur_ms / 1000.0) * self.fs)
            self.total_win_samples = self.pre_samples + self.post_samples

        self.latent_dim = int(self.cfg.get('latent_dim', 2))
        act = str(self.cfg.get('act', 'tanh'))
        tipo = str(self.cfg.get('tipo', ''))

        # Instanciar arquitectura correspondiente
        state_dict_raw = ckpt.get('model_state_dict', ckpt)
        state_keys = state_dict_raw.keys() if hasattr(state_dict_raw, 'keys') else []

        if 'classifier.weight' in state_keys or 'semisupervisado' in tipo:
            self.model = OrthogonalAutoencoder2DSemi(
                input_dim=self.n_canales * self.target_len,
                hidden_dim=int(self.cfg.get('hidden_dim', 32)),
                latent_dim=self.latent_dim,
                num_classes=5
            ).to(self.device)
        elif 'hidden_dims' in self.cfg or ('fc1.weight' in state_keys and 'fc2.weight' in state_keys and 'fc3.weight' in state_keys and 'conv1.weight' not in state_keys):
            hidden_dims = self.cfg.get('hidden_dims', (48, 24))
            if isinstance(hidden_dims, str):
                import ast
                hidden_dims = ast.literal_eval(hidden_dims)
            elif 'fc1.weight' in state_dict_raw and 'fc2.weight' in state_dict_raw:
                hidden_dims = (state_dict_raw['fc1.weight'].shape[0], state_dict_raw['fc2.weight'].shape[0])
            if 'fc3.weight' in state_dict_raw:
                self.latent_dim = int(state_dict_raw['fc3.weight'].shape[0])
            self.model = ParametricMLPOrthogonalAE(
                input_dim=self.n_canales * self.target_len,
                hidden_dims=hidden_dims,
                latent_dim=self.latent_dim,
                act_name=act
            ).to(self.device)
        else:
            channels = self.cfg.get('channels', (6, 12))
            if isinstance(channels, str):
                import ast
                channels = ast.literal_eval(channels)
            kernel_size = int(self.cfg.get('kernel_size', 5))
            self.model = ParametricConvOrthogonalAE(
                in_channels=3,
                time_pts=self.target_len,
                conv_channels=channels,
                kernel_size=kernel_size,
                latent_dim=self.latent_dim,
                act_name=act
            ).to(self.device)

        # Cargar pesos
        if 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
        else:
            self.model.load_state_dict(ckpt, strict=False)
        self.model.eval()

        # Reconstruir clasificador GMM
        if 'gmm_weights' in ckpt and 'gmm_means' in ckpt and 'gmm_covariances' in ckpt:
            self.gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42)
            self.gmm.weights_ = ckpt['gmm_weights']
            self.gmm.means_ = ckpt['gmm_means']
            self.gmm.covariances_ = ckpt['gmm_covariances']
            self.gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(self.gmm.covariances_))
            self.cluster_to_vocal = {int(k): int(v) for k, v in ckpt.get('cluster_to_vocal', {}).items()}
        else:
            # Buscar proyecciones latentes asociadas en el directorio del checkpoint
            dir_ckpt = os.path.dirname(pt_path)
            candidatos_csv = [
                os.path.join(dir_ckpt, "proyecciones_latentes_2d_crudo.csv"),
                os.path.join(dir_ckpt, "proyecciones_latentes_2d.csv")
            ]
            csv_path = next((c for c in candidatos_csv if os.path.exists(c)), None)
            if csv_path:
                import pandas as pd
                from scipy.optimize import linear_sum_assignment
                df_ent = pd.read_csv(csv_path)
                cols_z = [c for c in ['Z1', 'Z2', 'Z3'] if c in df_ent.columns]
                z_ent = df_ent[cols_z].values
                y_ent = df_ent['Vocal'].values
                self.gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
                pred_ent = self.gmm.fit_predict(z_ent)
                contingency = np.zeros((5, 5))
                for i_v, vl in enumerate(VOCALES):
                    for j_c in range(5):
                        contingency[j_c, i_v] = np.sum((y_ent == vl) & (pred_ent == j_c))
                row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
                self.cluster_to_vocal = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
            else:
                self.gmm = None
                self.cluster_to_vocal = {}

        return True

    def load_reference_projections(self, csv_path):
        """Carga proyecciones latentes de referencia de Lucas para graficar de fondo."""
        if not os.path.exists(csv_path):
            return False
        import pandas as pd
        df = pd.read_csv(csv_path)
        self.ref_z1 = df['Z1'].values.tolist()
        self.ref_z2 = df['Z2'].values.tolist()
        self.ref_vocales = df['Vocal'].values.tolist()
        return True

    def compute_decision_grid(self, x_range=(-3.5, 3.5), y_range=(-3.5, 3.5), resolution=120, proj_axes=(0, 1)):
        """
        Calcula la grilla de decisión 2D del clasificador GMM sobre el plano latente seleccionado.
        proj_axes: tupla de 2 índices (e.g. (0, 1) para Z1-Z2, (0, 2) para Z1-Z3, (1, 2) para Z2-Z3).
        Devuelve la matriz de colores RGBA lista para ser mostrada en PyQtGraph o Matplotlib.
        """
        if self.gmm is None or len(self.cluster_to_vocal) == 0:
            return None, None, None

        ax_x, ax_y = proj_axes
        x_lin = np.linspace(x_range[0], x_range[1], resolution)
        y_lin = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x_lin, y_lin)

        if self.latent_dim == 3:
            grid_pts = np.zeros((len(xx.ravel()), 3), dtype=np.float32)
            grid_pts[:, ax_x] = xx.ravel()
            grid_pts[:, ax_y] = yy.ravel()
            rem_axis = (set([0, 1, 2]) - set([ax_x, ax_y])).pop()
            if hasattr(self.gmm, 'means_') and self.gmm.means_ is not None:
                grid_pts[:, rem_axis] = float(np.mean(self.gmm.means_[:, rem_axis]))
        else:
            grid_pts = np.c_[xx.ravel(), yy.ravel()]

        # Predecir clusters GMM
        clusters = self.gmm.predict(grid_pts)
        probs = self.gmm.predict_proba(grid_pts)
        max_prob = np.max(probs, axis=1)

        # Mapear cluster a índice de vocal (0:A, 1:E, 2:I, 3:O, 4:U)
        vocal_indices = np.array([self.cluster_to_vocal.get(c, 0) for c in clusters])

        # Matriz RGBA
        rgba_img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        for v_idx, v_name in enumerate(VOCALES):
            mask = (vocal_indices == v_idx)
            color_norm = VOCAL_A_RGBA[v_name]
            r = int(color_norm[0] * 255)
            g = int(color_norm[1] * 255)
            b = int(color_norm[2] * 255)

            # Opacidad base suave (pastel) modulada ligeramente por la confianza
            alpha = (60 + 50 * max_prob[mask]).astype(np.uint8)
            grid_v_idx = np.where(mask)[0]
            for idx in grid_v_idx:
                row = idx // resolution
                col = idx % resolution
                rgba_img[row, col, 0] = r
                rgba_img[row, col, 1] = g
                rgba_img[row, col, 2] = b
                rgba_img[row, col, 3] = alpha[np.where(grid_v_idx == idx)[0][0]]

        return rgba_img, x_range, y_range

    def push_chunk(self, chunk, mic_chunk=None):
        """
        Ingesta de un nuevo lote de muestras crudas en streaming.
        chunk: array de forma (n_canales, n_muestras) en microvoltios o voltios.
        mic_chunk: array opcional con muestras del canal de micrófono (Canal 3).
        Retorna:
          eventos_decodificados: lista de eventos fonatorios detectados y decodificados.
          s_emg_nuevo: vector reciente de norma tricanal para visualización.
          estado_actual: dict con la telemetría del Gate Doble.
        """
        n_chunk = chunk.shape[1]
        if n_chunk == 0:
            return [], np.array([]), self.get_telemetry()

        t_inicio_proc = time.perf_counter()

        # 1. Filtrado IIR Causal (Notch 50 Hz + Pasa-Banda 20-500 Hz) con memoria zi
        if not self.filt_initialized:
            for c in range(self.n_canales):
                val0 = float(chunk[c, 0])
                self.notch_zi[:, :, c] = signal.sosfilt_zi(self.notch_sos) * val0
                self.bp_zi[:, :, c] = signal.sosfilt_zi(self.bp_sos) * val0
            self.filt_initialized = True

        chunk_notch = np.zeros_like(chunk)
        for c in range(self.n_canales):
            chunk_notch[c], self.notch_zi[:, :, c] = signal.sosfilt(
                self.notch_sos, chunk[c], zi=self.notch_zi[:, :, c]
            )

        chunk_filt = np.zeros_like(chunk)
        for c in range(self.n_canales):
            chunk_filt[c], self.bp_zi[:, :, c] = signal.sosfilt(
                self.bp_sos, chunk_notch[c], zi=self.bp_zi[:, :, c]
            )

        # 2. Desplazar buffers circulares y alojar datos nuevos
        self.raw_buffer = np.roll(self.raw_buffer, -n_chunk, axis=1)
        self.raw_buffer[:, -n_chunk:] = chunk

        self.filt_buffer = np.roll(self.filt_buffer, -n_chunk, axis=1)
        self.filt_buffer[:, -n_chunk:] = chunk_filt

        # Actualizar buffer de micrófono si se proporciona
        if mic_chunk is not None:
            if np.ndim(mic_chunk) > 1:
                mic_flat = mic_chunk.flatten()
            else:
                mic_flat = mic_chunk
            self.mic_buffer = np.roll(self.mic_buffer, -n_chunk)
            self.mic_buffer[-n_chunk:] = mic_flat

        # 3. Envolvente RMS en línea (ventana 90 ms)
        needed = min(self.buffer_len, self.rms_win_len + n_chunk)
        tail_filt = self.filt_buffer[:, -needed:]
        env_tail = np.zeros((self.n_canales, n_chunk), dtype=np.float32)
        for c in range(self.n_canales):
            sq = tail_filt[c] ** 2
            conv = np.convolve(sq, self.rms_kernel, mode='same')
            if len(conv) >= n_chunk:
                env_tail[c] = np.sqrt(np.maximum(conv[-n_chunk:], 0.0))
            else:
                env_tail[c, -len(conv):] = np.sqrt(np.maximum(conv, 0.0))

        self.env_buffer = np.roll(self.env_buffer, -n_chunk, axis=1)
        self.env_buffer[:, -n_chunk:] = env_tail

        # 4. Sustracción de piso de ruido y Norma Tricanal Combinada S_emg
        samples_base = self.total_samples_received

        # Si todavía está calibrando y ya acumuló 5.0s, recalibrar el piso de ruido primero
        if self.estado_gate == 'CALIBRANDO' and (samples_base + n_chunk) >= int(5.0 * self.fs):
            self.recalibrar_ruido(5.0)
            self.ultimo_onset_sample = samples_base + n_chunk

        env_clean = np.maximum(env_tail - self.ruido_base_canales, 0.0)
        s_emg_nuevo = np.sqrt(np.sum(env_clean ** 2, axis=0))

        self.s_emg_buffer = np.roll(self.s_emg_buffer, -n_chunk)
        self.s_emg_buffer[-n_chunk:] = s_emg_nuevo

        eventos = []

        # Sincronización acústica si Canal 3 (Micrófono) está activo y habilitado
        if mic_chunk is not None and self.mic_sync_enabled and (samples_base + n_chunk) >= int(6.0 * self.fs):
            recent_samples = min(int(4.5 * self.fs), self.buffer_len)
            sub_mic = self.mic_buffer[-recent_samples:]
            sub_env = np.convolve(np.abs(sub_mic), self.mic_kernel, mode='same')
            pks, _ = signal.find_peaks(sub_env, distance=int(1.2 * self.fs), height=2000)
            margin_samples = int(1.0 * self.fs)
            wait_samples = self.post_samples + margin_samples

            for p_sub in pks:
                p_sample = (samples_base + n_chunk) - recent_samples + p_sub
                if (p_sub <= recent_samples - wait_samples) and p_sample >= int(6.0 * self.fs) and (p_sample - self.ultimo_pico_mic_sample) >= int(1.1 * self.fs):
                    self.ultimo_pico_mic_sample = p_sample
                    evento = self._extraer_y_decodificar(
                        onset_s=p_sample - self.emd_samples,
                        center_s=p_sample,
                        curr_s=(samples_base + n_chunk),
                        t_inicio=t_inicio_proc
                    )
                    if evento is not None:
                        eventos.append(evento)
                        self.estado_gate = 'REFRACTARIO'
                        self.ultimo_onset_sample = p_sample - self.emd_samples

        # 5. Máquina de Estados de Gate Doble (activa cuando no hay mic sync o como respaldo)
        if self.estado_gate == 'CALIBRANDO':
            self.total_samples_received += n_chunk
            return eventos, s_emg_nuevo, self.get_telemetry()

        if not (mic_chunk is not None and self.mic_sync_enabled):
            for i in range(n_chunk):
                curr_sample = samples_base + i
                val_s = s_emg_nuevo[i]

                # Estado ARMADO: esperando que S_emg cruce el umbral alto de confirmación
                if self.estado_gate == 'ARMADO':
                    if val_s >= self.u_alto and (curr_sample - self.ultimo_onset_sample) >= self.t_refr_samples:
                        buffer_idx_curr = self.buffer_len - n_chunk + i
                        lookback_st = max(0, buffer_idx_curr - self.max_lookback_samples)
                        sub_s = self.s_emg_buffer[lookback_st:buffer_idx_curr]
                        cruces_bajo = np.where(sub_s <= self.u_bajo)[0]

                        if len(cruces_bajo) > 0:
                            offset_bajo = (buffer_idx_curr - (lookback_st + cruces_bajo[-1]))
                        else:
                            offset_bajo = min(self.max_lookback_samples, buffer_idx_curr - lookback_st)

                        onset_sample = curr_sample - offset_bajo
                        center_sample = onset_sample + self.emd_samples
                        fin_sample = center_sample + self.post_samples

                        self.onset_sample_actual = onset_sample
                        self.center_sample_actual = center_sample
                        self.fin_sample_actual = fin_sample
                        self.ultimo_onset_sample = onset_sample
                        self.estado_gate = 'ESPERANDO_FIN_VENTANA'

                # Estado ESPERANDO_FIN_VENTANA: acumulando muestras hasta completar la ventana post
                elif self.estado_gate == 'ESPERANDO_FIN_VENTANA':
                    if curr_sample >= self.fin_sample_actual:
                        evento = self._extraer_y_decodificar(
                            onset_s=self.onset_sample_actual,
                            center_s=self.center_sample_actual,
                            curr_s=curr_sample,
                            t_inicio=t_inicio_proc
                        )
                        if evento is not None:
                            eventos.append(evento)
                        self.estado_gate = 'REFRACTARIO'

                # Estado REFRACTARIO: esperando que concluya el período refractario biológico
                elif self.estado_gate == 'REFRACTARIO':
                    if (curr_sample - self.ultimo_onset_sample) >= self.t_refr_samples:
                        self.estado_gate = 'ARMADO'
        else:
            # En modo mic sync, mantener armado el gate cuando concluya el refractario
            if self.estado_gate == 'REFRACTARIO':
                if ((samples_base + n_chunk) - self.ultimo_onset_sample) >= self.t_refr_samples:
                    self.estado_gate = 'ARMADO'

        self.total_samples_received += n_chunk
        return eventos, s_emg_nuevo, self.get_telemetry()

    def flush_pending_events(self):
        """
        Procesa cualquier evento pendiente en el buffer al finalizar la reproducción o adquisición.
        Garantiza que el último evento de una grabación de prueba sea decodificado.
        """
        eventos = []
        if self.mic_sync_enabled and self.total_samples_received >= int(6.0 * self.fs):
            recent_samples = min(int(4.5 * self.fs), self.buffer_len)
            sub_mic = self.mic_buffer[-recent_samples:]
            sub_env = np.convolve(np.abs(sub_mic), self.mic_kernel, mode='same')
            pks, _ = signal.find_peaks(sub_env, distance=int(1.2 * self.fs), height=2000)
            for p_sub in pks:
                p_sample = self.total_samples_received - recent_samples + p_sub
                if (p_sub <= recent_samples - self.post_samples) and p_sample >= int(6.0 * self.fs) and (p_sample - self.ultimo_pico_mic_sample) >= int(1.1 * self.fs):
                    self.ultimo_pico_mic_sample = p_sample
                    evento = self._extraer_y_decodificar(
                        onset_s=p_sample - self.emd_samples,
                        center_s=p_sample,
                        curr_s=self.total_samples_received,
                        t_inicio=time.perf_counter()
                    )
                    if evento is not None:
                        eventos.append(evento)
        return eventos

    def _extraer_y_decodificar(self, onset_s, center_s, curr_s, t_inicio):
        """
        Corta la ventana centrada por EMD o acústica, acondiciona con zero-phase filtfilt,
        normaliza por Supremo Tricanal y ejecuta la inferencia de PyTorch + GMM.
        """
        distancia_al_fin = curr_s - center_s
        idx_centro_buf = self.buffer_len - distancia_al_fin

        idx_ini = idx_centro_buf - self.pre_samples
        idx_fin = idx_centro_buf + self.post_samples

        if idx_ini < 0 or idx_fin > self.buffer_len:
            return None

        # Margen de seguridad temporal (1.0 s) a cada lado para que los transitorios
        # de borde de filtfilt (Notch 50 Hz y Pasabanda) decaigan fuera de la ventana útil
        margin_samples = int(1.0 * self.fs)
        idx_ini_pad = max(0, idx_ini - margin_samples)
        idx_fin_pad = min(self.buffer_len, idx_fin + margin_samples)

        seg_raw = self.raw_buffer[:, idx_ini_pad:idx_fin_pad]
        if seg_raw.shape[1] < self.total_win_samples:
            return None

        # 2. Filtrado zero-phase filtfilt sobre el segmento con margen
        seg_filt = np.zeros_like(seg_raw)
        for c in range(self.n_canales):
            s_n = signal.filtfilt(self.b_notch, self.a_notch, seg_raw[c])
            seg_filt[c] = signal.filtfilt(self.b_band, self.a_band, s_n)

        # 3. Envolvente RMS simétrica
        seg_env = np.zeros_like(seg_filt)
        for c in range(self.n_canales):
            seg_env[c] = np.sqrt(np.maximum(0, np.convolve(seg_filt[c] ** 2, self.rms_kernel, mode='same')))

        # 4. Recortar exactamente la ventana útil [idx_ini, idx_fin]
        offset_ini = idx_ini - idx_ini_pad
        offset_fin = offset_ini + self.total_win_samples
        win_env = seg_env[:, offset_ini:offset_fin]

        # 5. Sustracción de piso de ruido basal por canal
        win_clean = np.maximum(win_env - self.ruido_base_canales, 0.0)

        # 5. Normalización OBLIGATORIA por Supremo Tricanal por Pulso Individual
        supremo = float(np.max(win_clean))
        if supremo < 1e-6:
            supremo = 1.0
        win_norm = win_clean / supremo

        # 6. Remuestreo lineal a target_len (20 puntos por canal)
        win_resampled = np.zeros((self.n_canales, self.target_len), dtype=np.float32)
        x_orig = np.linspace(0, 1, self.total_win_samples)
        x_target = np.linspace(0, 1, self.target_len)
        for c in range(self.n_canales):
            win_resampled[c] = np.interp(x_target, x_orig, win_norm[c])

        # 7. Suavizado temporal Butterworth (orden 3, Wn=0.3) idéntico al entrenamiento
        win_smooth = np.zeros_like(win_resampled)
        for c in range(self.n_canales):
            win_smooth[c] = signal.filtfilt(self.b_temporal, self.a_temporal, win_resampled[c])

        # 8. Estandarización de impedancia / amplitud esperada por la red ortogonal
        win_final = np.zeros_like(win_smooth)
        for c in range(self.n_canales):
            win_final[c] = (win_smooth[c] - self.channel_base_mean[c]) / self.channel_scale[c]

        # 8. Preparar tensor de entrada PyTorch (1, 60)
        x_flat = win_final.reshape(1, -1)
        x_tensor = torch.tensor(x_flat, dtype=torch.float32, device=self.device)

        # 9. Inferencia PyTorch (Encoder)
        t_infer_st = time.perf_counter()
        with torch.no_grad():
            if self.model is not None:
                z_tensor = self.model.encode(x_tensor)
                z_np = z_tensor.cpu().numpy()[0]
            else:
                z_np = np.zeros(self.latent_dim, dtype=np.float32)
        t_infer_end = time.perf_counter()

        # 10. Clasificación GMM y probabilidades
        vocal_ganadora = 'A'
        confianza_pct = 0.0
        probs_dict = {v: 0.0 for v in VOCALES}

        if self.gmm is not None and len(self.cluster_to_vocal) > 0:
            cluster_pred = int(self.gmm.predict(z_np.reshape(1, -1))[0])
            probs_gmm = self.gmm.predict_proba(z_np.reshape(1, -1))[0]

            v_idx = self.cluster_to_vocal.get(cluster_pred, 0)
            vocal_ganadora = VOCALES[v_idx]

            # Probabilidades mapeadas a cada vocal
            for c_id, v_id in self.cluster_to_vocal.items():
                if c_id < len(probs_gmm):
                    v_name = VOCALES[v_id]
                    probs_dict[v_name] += float(probs_gmm[c_id]) * 100.0

            confianza_pct = probs_dict[vocal_ganadora]
        else:
            # Fallback simple
            vocal_ganadora = 'A'
            confianza_pct = 80.0
            probs_dict['A'] = 80.0

        latencia_total_ms = (time.perf_counter() - t_inicio) * 1000.0
        latencia_infer_ms = (t_infer_end - t_infer_st) * 1000.0

        return {
            'vocal': vocal_ganadora,
            'color': VOCAL_A_COLOR_HEX[vocal_ganadora],
            'confianza': confianza_pct,
            'z1': float(z_np[0]),
            'z2': float(z_np[1]) if len(z_np) > 1 else 0.0,
            'z3': float(z_np[2]) if len(z_np) > 2 else 0.0,
            'probs': probs_dict,
            'onset_sample': onset_s,
            'center_sample': center_s,
            'supremo_uv': supremo,
            'latencia_total_ms': latencia_total_ms,
            'latencia_infer_ms': latencia_infer_ms,
            'timestamp': time.time()
        }

    def get_telemetry(self):
        """Devuelve el estado instantáneo de la telemetría del Gate Doble."""
        return {
            'estado': self.estado_gate,
            'u_bajo': self.u_bajo,
            'u_alto': self.u_alto,
            'med_base': self.med_base,
            'calibrado': self.calibrado,
            'emd_delay_ms': getattr(self, 'emd_delay_ms', 350.0),
            'total_muestras': self.total_samples_received
        }
