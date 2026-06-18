import os
import json
import csv
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import soundfile as sf
from scipy.signal import resample, find_peaks, butter, filtfilt, iirnotch, stft, welch

try:
    import pywt
except ImportError:
    pywt = None

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def apply_dsp_pipeline(sig, sr, method, noise_seconds=0.15, smooth_ms=75.0, notch_q=30.0):
    nyq = 0.5 * sr
    
    cutoff_hp = min(20.0, nyq * 0.99)
    b, a = butter(4, cutoff_hp / nyq, btype='high', analog=False)
    sig_filt = filtfilt(b, a, sig)
    
    f0 = 50.0
    if f0 < nyq:
        b, a = iirnotch(f0, notch_q, sr)
        sig_filt = filtfilt(b, a, sig_filt)
        
    cutoff_lp = min(500.0, nyq * 0.99)
    b, a = butter(4, cutoff_lp / nyq, btype='low', analog=False)
    sig_filt = filtfilt(b, a, sig_filt)
    
    if method in ["env", "env_plus"]:
        window_size = int(smooth_ms / 1000.0 * sr)
        if window_size == 0: window_size = 1
        # Verdadera Envolvente RMS (Root Mean Square)
        squared = sig_filt ** 2
        mean_squared = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        processed = np.sqrt(np.maximum(mean_squared, 0))
    elif method in ["stft", "dios"]:
        # Para STFT y dios, necesitamos la señal oscilatoria intacta (sin rectificar ni promediar)
        processed = sig_filt
    else: # amplitud
        processed = np.abs(sig_filt)
        
    initial_noise_mean = 0.0
    noise_samples = int(noise_seconds * sr)
    if noise_samples > 0 and noise_samples < len(processed):
        skip = min(int(0.1 * sr), noise_samples // 2)
        initial_noise_mean = np.mean(np.abs(processed[skip:noise_samples]))
        
    return sig_filt, processed, initial_noise_mean

def get_interpulse_noise(processed_segment, initial_noise):
    if len(processed_segment) < 3:
        return initial_noise
        
    abs_noise = np.abs(processed_segment)
    q1 = np.percentile(abs_noise, 25)
    q3 = np.percentile(abs_noise, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    
    valid_noise = abs_noise[abs_noise <= upper_bound]
    if len(valid_noise) < 3:
        valid_noise = abs_noise
        
    curr_mean = np.mean(valid_noise)
    
    if initial_noise > 0 and (curr_mean / initial_noise) > 5.0:
        return initial_noise
    return curr_mean

class ExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ñandú LSD - Extractor Multimodelo Interactivo")
        self.root.geometry("750x900")
        
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.fg_text = "#C5C6C7"
        
        self.root.configure(bg=self.bg_dark)
        self.BASE_DIR = os.path.abspath(os.path.join(current_dir, "..", "base_de_datos_electrodos"))
        
        main_frame = tk.Frame(self.root, padx=15, pady=15, bg=self.bg_dark)
        main_frame.pack(fill="both", expand=True)

        lbl = tk.Label(main_frame, text="1. Seleccionar mediciones a extraer:", bg=self.bg_dark, fg=self.cyan_neon, font=("Arial", 11, "bold"))
        lbl.pack(anchor="w", pady=(0, 5))

        list_frame = tk.Frame(main_frame, bg=self.bg_dark)
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED, yscrollcommand=scrollbar.set,
                                  bg=self.bg_panel, fg="white", selectbackground=self.cyan_neon,
                                  selectforeground="black", borderwidth=0, highlightthickness=1)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)

        btn_frame = tk.Frame(main_frame, bg=self.bg_dark)
        btn_frame.pack(fill="x", pady=5)
        
        tk.Button(btn_frame, text="Seleccionar Todas", command=self.select_all, bg=self.bg_panel, fg=self.cyan_neon).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Deseleccionar Todas", command=self.deselect_all, bg=self.bg_panel, fg=self.cyan_neon).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Actualizar Lista", command=self.populate_listbox, bg=self.bg_panel, fg=self.cyan_neon).pack(side="right", padx=5)

        lbl_settings = tk.Label(main_frame, text="2. Configuración de Extracción:", bg=self.bg_dark, fg=self.cyan_neon, font=("Arial", 11, "bold"))
        lbl_settings.pack(anchor="w", pady=(15, 5))

        # --- MODO DEFINITIVO (ENV) ---
        self.extraction_method = tk.StringVar(value="env")
        
        # Eliminados los RadioButtons y el frame Custom a petición del usuario.

        # Panel de opciones exclusivas de la Envolvente
        self.env_opt_frame = tk.Frame(main_frame, bg=self.bg_panel, padx=10, pady=10)
        # Se muestra u oculta en toggle_env_options
        
        tk.Label(self.env_opt_frame, text="Suavizado Envolvente (ms):", bg=self.bg_panel, fg="white").grid(row=0, column=0, sticky="w", pady=5)
        self.env_smooth_var = tk.DoubleVar(value=250.0)
        tk.Spinbox(self.env_opt_frame, from_=5.0, to=500.0, increment=5.0, textvariable=self.env_smooth_var, width=8).grid(row=0, column=1, sticky="w", padx=10)

        tk.Label(self.env_opt_frame, text="Filtro Notch Q Factor:", bg=self.bg_panel, fg="white").grid(row=0, column=2, sticky="w", pady=5)
        self.notch_q_var = tk.DoubleVar(value=30.0)
        tk.Spinbox(self.env_opt_frame, from_=0.1, to=100.0, increment=0.5, textvariable=self.notch_q_var, width=8).grid(row=0, column=3, sticky="w", padx=10)

        self.use_master_slave_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.env_opt_frame, text="Modo Master-Slave (Anclar al Músculo Más Fuerte/Canal 3)", variable=self.use_master_slave_var, bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        tk.Label(self.env_opt_frame, text="Suavizado Maestro (ms):", bg=self.bg_panel, fg="white").grid(row=2, column=0, sticky="w", pady=5)
        self.master_smooth_var = tk.DoubleVar(value=250.0)
        tk.Spinbox(self.env_opt_frame, from_=50.0, to=1000.0, increment=10.0, textvariable=self.master_smooth_var, width=8).grid(row=2, column=1, sticky="w", padx=10)

        self.use_fine_alignment_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.env_opt_frame, text="Alineación Fina (Buscar pico local de cada músculo tras anclar)", variable=self.use_fine_alignment_var, bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=3, column=0, columnspan=2, sticky="w", pady=5)
        
        tk.Label(self.env_opt_frame, text="Normalización de Ventana:", bg=self.bg_panel, fg="white").grid(row=4, column=0, sticky="w", pady=5)
        self.norm_mode_var = tk.StringVar(value="global_max")
        tk.Radiobutton(self.env_opt_frame, text="Max Global (Conserva proporción muscular)", variable=self.norm_mode_var, value="global_max", bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=4, column=1, sticky="w")
        tk.Radiobutton(self.env_opt_frame, text="Max Local (Normaliza cada canal por su propio pico)", variable=self.norm_mode_var, value="local_max", bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=5, column=1, sticky="w")
        tk.Radiobutton(self.env_opt_frame, text="Norma L1 Horizontal (Suma abs = 1)", variable=self.norm_mode_var, value="l1", bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=6, column=1, sticky="w")
        tk.Radiobutton(self.env_opt_frame, text="Norma L2 Horizontal (Suma cuadrados = 1)", variable=self.norm_mode_var, value="l2", bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=7, column=1, sticky="w")
        tk.Radiobutton(self.env_opt_frame, text="Sin Normalizar (Datos crudos en mV)", variable=self.norm_mode_var, value="none", bg=self.bg_panel, fg="white", selectcolor=self.bg_dark, command=lambda: self.scaler_mode_var.set("StandardScaler (Completo)")).grid(row=8, column=1, sticky="w")
        
        tk.Label(self.env_opt_frame, text="Factor Pre-Pico (x Periodo, ej 0.4):", bg=self.bg_panel, fg="white").grid(row=9, column=0, sticky="w", pady=5)
        self.pre_peak_factor_var = tk.DoubleVar(value=0.4)
        tk.Spinbox(self.env_opt_frame, from_=0.1, to=1.5, increment=0.05, textvariable=self.pre_peak_factor_var, width=8).grid(row=9, column=1, sticky="w", padx=10)

        tk.Label(self.env_opt_frame, text="Factor Post-Pico (x Periodo, ej 0.6):", bg=self.bg_panel, fg="white").grid(row=10, column=0, sticky="w", pady=5)
        self.post_peak_factor_var = tk.DoubleVar(value=0.6)
        tk.Spinbox(self.env_opt_frame, from_=0.1, to=1.5, increment=0.05, textvariable=self.post_peak_factor_var, width=8).grid(row=10, column=1, sticky="w", padx=10)

        tk.Label(self.env_opt_frame, text="Puntos de Resampling (PCA Dim):", bg=self.bg_panel, fg="white").grid(row=11, column=0, sticky="w", pady=5)
        self.resample_points_var = tk.IntVar(value=100)
        tk.Spinbox(self.env_opt_frame, from_=50, to=1000, increment=50, textvariable=self.resample_points_var, width=8).grid(row=11, column=1, sticky="w", padx=10)

        self.use_binarization_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.env_opt_frame, text="Incluir Desempate Binario (Trevisan)", variable=self.use_binarization_var, bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=12, column=0, columnspan=2, sticky="w", pady=5)

        self.use_log_transform_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.env_opt_frame, text="Transformada Logarítmica (Truco DSP Ratio->Resta)", variable=self.use_log_transform_var, bg=self.bg_panel, fg="white", selectcolor=self.bg_dark).grid(row=13, column=0, columnspan=2, sticky="w", pady=5)

        tk.Label(self.env_opt_frame, text="Modo Scaler (PCA):", bg=self.bg_panel, fg="white").grid(row=14, column=0, sticky="w", pady=5)
        self.scaler_mode_var = tk.StringVar(value="--center-only")
        scaler_options = {
            "Solo Centrar Media (--center-only)": "--center-only",
            "Sin Scaler (Datos Crudos)": "--no-scaler",
            "StandardScaler (Completo)": ""
        }
        self.scaler_dropdown = ttk.Combobox(self.env_opt_frame, textvariable=self.scaler_mode_var, values=list(scaler_options.keys()), state="readonly", width=25)
        self.scaler_dropdown.grid(row=14, column=1, sticky="w", padx=10)
        self.scaler_options_map = scaler_options

        self.btn_extract = tk.Button(main_frame, text="3. Extraer Features (y correr PCA/UMAP)", command=self.run_extraction_thread, bg=self.cyan_neon, fg="black", font=("Arial", 12, "bold"), pady=10)
        self.btn_extract.pack(fill="x", pady=10)

        self.btn_xgboost = tk.Button(main_frame, text="💥 4. Opción Nuclear (Clasificar con XGBoost)", command=self.run_xgboost, bg="#ff4d4d", fg="white", font=("Arial", 12, "bold"), pady=10)
        self.btn_xgboost.pack(fill="x", pady=(0, 15))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x")
        
        self.lbl_status = tk.Label(main_frame, text="Listo.", bg=self.bg_dark, fg=self.fg_text)
        self.lbl_status.pack(pady=5)
        
        self.log_text = tk.Text(main_frame, height=8, bg=self.bg_panel, fg=self.cyan_neon, font=("Courier", 9))
        self.log_text.pack(fill="both", expand=True, pady=5)
        self.log_text.insert(tk.END, "Esperando extracción...")
        self.log_text.config(state=tk.DISABLED)
        
        self.populate_listbox()
        self.toggle_env_options()

    def log(self, text):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(text)

    def toggle_env_options(self):
        self.env_opt_frame.pack(fill="x", pady=5, before=self.btn_extract)

    def populate_listbox(self):
        self.listbox.delete(0, tk.END)
        if not os.path.exists(self.BASE_DIR):
            return
        valid_folders = []
        for root, dirs, files in os.walk(self.BASE_DIR):
            if 'canal_0' in dirs:
                rel_path = os.path.relpath(root, self.BASE_DIR)
                valid_folders.append(rel_path)
        valid_folders.sort()
        for folder in valid_folders:
            self.listbox.insert(tk.END, folder)

    def select_all(self):
        self.listbox.select_set(0, tk.END)

    def deselect_all(self):
        self.listbox.selection_clear(0, tk.END)

    def run_xgboost(self):
        method = self.extraction_method.get()
        script_path = os.path.join(current_dir, "analisis_xgboost.py")
        self.log(f"Ejecutando Opción Nuclear XGBoost con dataset {method}...")
        subprocess.Popen([sys.executable, script_path, method])

    def run_extraction_thread(self):
        seleccionados = self.listbox.curselection()
        if not seleccionados:
            messagebox.showwarning("Advertencia", "Selecciona al menos una toma.")
            return
            
        seleccion = [self.listbox.get(i) for i in seleccionados]
        self.btn_extract.config(state=tk.DISABLED)
        self.progress_var.set(0)
        
        # Obtener valores de la UI
        method = self.extraction_method.get()
        env_smooth_ms = self.env_smooth_var.get()
        use_master_slave = self.use_master_slave_var.get()
        master_smooth_ms = self.master_smooth_var.get()
        use_fine_alignment = self.use_fine_alignment_var.get()
        norm_mode = self.norm_mode_var.get()
        pre_peak_factor = self.pre_peak_factor_var.get()
        post_peak_factor = self.post_peak_factor_var.get()
        resample_points = self.resample_points_var.get()
        use_binarization = self.use_binarization_var.get()
        use_log_transform = self.use_log_transform_var.get()
        
        self.log(f"Iniciando extracción masiva...")
        self.log(f"Método seleccionado: {method}")
        
        if method == "dios" and pywt is None:
            messagebox.showerror("Dependencia Faltante", "El 'Modo Dios' requiere la librería PyWavelets.\nInstálala usando: pip install PyWavelets")
            self.btn_extract.config(state=tk.NORMAL)
            return

        thread = threading.Thread(target=self.extraer_dataset_features, args=(seleccion, method, env_smooth_ms, use_master_slave, master_smooth_ms, use_fine_alignment, norm_mode, pre_peak_factor, post_peak_factor, resample_points, use_binarization, use_log_transform))
        thread.start()

    def extraer_dataset_features(self, seleccion, method, env_smooth_ms, use_master_slave, master_smooth_ms, use_fine_alignment, norm_mode, pre_peak_factor, post_peak_factor, resample_points, use_binarization, use_log_transform):
        self.lbl_status.config(text=f"Iniciando extracción con método {method.upper()}...")
        
        out_dir = os.path.abspath(os.path.join(current_dir, "..", "base_de_datos_letras"))
        os.makedirs(out_dir, exist_ok=True)
            
        all_rows = []
        all_rows_debug = [] # Nuevo: Para guardar las ventanas raw y verificarlas visualmente en Modo Dios
        dynamic_fieldnames = []
        dynamic_fieldnames_debug = []
        dynamic_fieldnames_generated = False
        ventanas_validas_procesadas = 0

        total = len(seleccion)
        for i, folder in enumerate(seleccion):
            self.lbl_status.config(text=f"[{i+1}/{total}] Procesando: {folder}")
            med_path = os.path.join(self.BASE_DIR, folder)
            
            bpm_u = 30
            noise_u = 2.0
            meta_path = os.path.join(med_path, 'canal_0', 'metadata.json')
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta_data = json.load(f)
                        bpm_u = meta_data.get('bpm', bpm_u)
                        noise_u = meta_data.get('noise_seconds', noise_u)
                except:
                    pass
                    
            periodo = 60.0 / bpm_u
            
            ch3_dir = os.path.join(med_path, "canal_3")
            if not os.path.exists(ch3_dir):
                continue
            wavs3 = [f for f in os.listdir(ch3_dir) if f.endswith(".wav")]
            if not wavs3:
                continue
                
            sig3, sr = sf.read(os.path.join(ch3_dir, wavs3[0]))
            if sig3.ndim > 1: sig3 = sig3[:, 0]
            
            dist_samples = int(0.8 * periodo * sr)
            start_search = int(noise_u * sr)
            
            if use_master_slave:
                _, env3, _ = apply_dsp_pipeline(sig3, sr, "env", noise_u, smooth_ms=master_smooth_ms, notch_q=self.notch_q_var.get())
            else:
                _, env3, _ = apply_dsp_pipeline(sig3, sr, "env", noise_u, smooth_ms=env_smooth_ms, notch_q=self.notch_q_var.get())
                
            picos, _ = find_peaks(env3[start_search:], distance=dist_samples, height=np.max(env3[start_search:])*0.2)
            picos = picos + start_search

            envs = []
            noises = []
            for ch in ['canal_0', 'canal_1', 'canal_2']:
                ch_dir = os.path.join(med_path, ch)
                if not os.path.exists(ch_dir) or not [f for f in os.listdir(ch_dir) if f.endswith(".wav")]:
                    envs.append(np.zeros_like(env3))
                    noises.append(0.0)
                    continue
                wavs = [f for f in os.listdir(ch_dir) if f.endswith(".wav")]
                sig, _ = sf.read(os.path.join(ch_dir, wavs[0]))
                if sig.ndim > 1: sig = sig[:, 0]
                
                _, env_ch, ini_noise = apply_dsp_pipeline(sig, sr, method, noise_u, smooth_ms=env_smooth_ms, notch_q=self.notch_q_var.get())
                envs.append(env_ch)
                noises.append(ini_noise)

            vocal_label = "Desconocida"
            basename = os.path.basename(folder).upper()
            if basename.startswith("A_"): vocal_label = "A"
            elif basename.startswith("E_"): vocal_label = "E"
            elif basename.startswith("I_"): vocal_label = "I"
            elif basename.startswith("O_"): vocal_label = "O"
            elif basename.startswith("U_"): vocal_label = "U"

            periodo_samples = int(periodo * sr)
            noise_win_samples = max(3, int(periodo_samples / 4.0))

            for j, pico in enumerate(picos):
                if pico < (0.5 * periodo * sr) or pico > (len(sig3) - 0.5 * periodo * sr):
                    continue
                    
                if j == 0:
                    noise_start = max(0, int(pico - 0.5 * periodo_samples - noise_win_samples))
                else:
                    midpoint = (picos[j-1] + pico) // 2
                    noise_start = max(0, int(midpoint - noise_win_samples // 2))
                    
                noise_end = min(len(envs[0]), noise_start + noise_win_samples)
                
                interpulse_noise = [0.0, 0.0, 0.0]
                if noise_end > noise_start:
                    for ch_idx in range(3):
                        noise_seg = envs[ch_idx][noise_start:noise_end]
                        interpulse_noise[ch_idx] = get_interpulse_noise(noise_seg, noises[ch_idx])
                        
                # ---- MASTER-SLAVE FINE ALIGNMENT (CROSS-CORRELATION) ----
                t_pico_master = pico / sr
                window_radius_sec = 0.5 * periodo
                
                if use_fine_alignment:
                    import scipy.signal
                    idx_start_mic = max(0, int((t_pico_master - window_radius_sec) * sr))
                    idx_end_mic = min(len(env3), int((t_pico_master + window_radius_sec) * sr))
                    if idx_end_mic > idx_start_mic:
                        mic_window = env3[idx_start_mic:idx_end_mic]
                        if len(mic_window) > 0:
                            pulso_ideal = scipy.signal.windows.hann(len(mic_window))
                            correlacion = np.correlate(mic_window, pulso_ideal, mode='full')
                            lag_samples = np.argmax(correlacion) - (len(mic_window) - 1)
                            t_pico_master += (lag_samples / sr)
                # ---------------------------------------------------------
                        
                segs_env = []
                for ch_idx in range(3):
                    t_pico_ch = t_pico_master
                    
                    # Recorte asimétrico usando Factores Independientes
                    t_start = t_pico_ch - (periodo * pre_peak_factor)
                    t_end = t_pico_ch + (periodo * post_peak_factor)
                    
                    idx_start = int(t_start * sr)
                    idx_end = int(t_end * sr)
                    
                    seg_ch = np.zeros(idx_end - idx_start)
                    sig_start = max(0, idx_start)
                    sig_end = min(len(envs[ch_idx]), idx_end)
                    
                    if sig_end > sig_start:
                        valid_data = envs[ch_idx][sig_start:sig_end]
                        insert_start = sig_start - idx_start
                        seg_ch[insert_start : insert_start + len(valid_data)] = valid_data
                        
                    if method not in ["stft", "dios"]:
                        seg_ch = seg_ch - interpulse_noise[ch_idx]
                        seg_ch[seg_ch < 0] = 0.0
                    segs_env.append(seg_ch)
                    
                row_data = {
                    'Toma': folder,
                    'Vocal': vocal_label
                }
                
                if method == "dios":
                    # --- Novedad: Guardar también la ventana raw para validación visual ---
                    row_data_debug = {
                        'Toma': folder,
                        'Vocal': vocal_label
                    }
                    
                    # Normalización Modo Dios: encontrar Pico Máximo de amplitud cruda entre los 3 canales
                    max_pico_amplitud = 1e-9
                    for e in segs_env:
                        if len(e) > 0:
                            m_val = np.max(np.abs(e))
                            if m_val > max_pico_amplitud:
                                max_pico_amplitud = m_val
                                
                    for ch_idx in range(3):
                        x = segs_env[ch_idx]
                        if len(x) == 0:
                            x = np.zeros(10)
                            
                        # 1. Dominio Temporal
                        rms = np.sqrt(np.mean(x**2))
                        var = np.var(x)
                        z_cross = np.where(np.diff(np.signbit(x)))[0]
                        zcr = len(z_cross) / len(x) if len(x) > 0 else 0
                        
                        # 2. Dominio Espectral (Welch)
                        mnf = 0.0
                        mdf = 0.0
                        nperseg = min(256, len(x))
                        if nperseg > 0:
                            frequencies, psd = welch(x, fs=sr, nperseg=nperseg)
                            psd_sum = np.sum(psd)
                            if psd_sum > 0:
                                mnf = np.sum(frequencies * psd) / psd_sum
                                cumulative_sum = np.cumsum(psd)
                                mdf = frequencies[np.searchsorted(cumulative_sum, psd_sum / 2.0)]
                                
                        # 3. Dominio Tiempo-Frecuencia (Wavelet)
                        wavelet_energies = []
                        wavelet_entropies = []
                        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet('db4').dec_len)
                        level = min(4, max_level)
                        if level > 0:
                            coeffs = pywt.wavedec(x, 'db4', level=level)
                            for c in coeffs:
                                energy = np.sum(c**2)
                                p = (c**2) / (energy + 1e-9)
                                p = p[p > 0]
                                entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0.0
                                wavelet_energies.append(energy)
                                wavelet_entropies.append(entropy)
                            while len(wavelet_energies) < 5:
                                wavelet_energies.append(0.0)
                                wavelet_entropies.append(0.0)
                        else:
                            wavelet_energies = [0.0]*5
                            wavelet_entropies = [0.0]*5
                            
                        # Normalización: "divide únicamente las características basadas en amplitud/energía por ese Pico Máximo"
                        rms_norm = rms / max_pico_amplitud
                        var_norm = var / max_pico_amplitud
                        wavelet_energies_norm = [e / max_pico_amplitud for e in wavelet_energies]
                        
                        # Assemblaje
                        features = [
                            rms_norm, var_norm, zcr,
                            mnf, mdf,
                            *wavelet_energies_norm,
                            *wavelet_entropies
                        ]
                        
                        feat_names = [
                            "RMS", "Var", "ZCR", "MNF", "MDF",
                            "E_cA4", "E_cD4", "E_cD3", "E_cD2", "E_cD1",
                            "S_cA4", "S_cD4", "S_cD3", "S_cD2", "S_cD1"
                        ]
                        
                        for feat_i, (fname, fval) in enumerate(zip(feat_names, features)):
                            col_name = f'Ch{ch_idx}_{fname}'
                            row_data[col_name] = f"{fval:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                                dynamic_fieldnames.append(col_name)
                                
                        # --- Guardar versión downsampled (400 pts) de la ventana raw para el Visor ---
                        if len(x) > 0:
                            x_resampled = resample(x, 400)
                            for t_i, val in enumerate(x_resampled):
                                col_name_debug = f'Ch{ch_idx}_T{t_i}'
                                row_data_debug[col_name_debug] = f"{val:.6f}"
                                if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                                    dynamic_fieldnames_debug.append(col_name_debug)
                                    
                    all_rows_debug.append(row_data_debug)
                else:
                    norm_envs = []
                    if norm_mode == "l1":
                        sum_abs = 0.0
                        for e in segs_env:
                            if len(e) > 0: sum_abs += np.sum(np.abs(e))
                        if sum_abs == 0: sum_abs = 1e-9
                        for ch_idx in range(3):
                            norm_envs.append(segs_env[ch_idx] / sum_abs)
                            
                    elif norm_mode == "l2":
                        sum_sq = 0.0
                        for e in segs_env:
                            if len(e) > 0: sum_sq += np.sum(e**2)
                        l2_norm = np.sqrt(sum_sq)
                        if l2_norm == 0: l2_norm = 1e-9
                        for ch_idx in range(3):
                            norm_envs.append(segs_env[ch_idx] / l2_norm)
                            
                    elif norm_mode == "local_max":
                        for e in segs_env:
                            m_val = np.max(e) if len(e) > 0 else 0
                            if m_val == 0: m_val = 1e-9
                            norm_envs.append(e / m_val)
                            
                    elif norm_mode == "none":
                        # Sin normalizar: datos crudos en mV
                        for e in segs_env:
                            norm_envs.append(e)
                            
                    else: # global_max
                        max_supremo = 1e-9
                        for e in segs_env:
                            if len(e) > 0:
                                m_val = np.max(e)
                                if m_val > max_supremo:
                                    max_supremo = m_val
                        for ch_idx in range(3):
                            norm_envs.append(segs_env[ch_idx] / max_supremo)
                    
                    # Mantener la resolución original exacta
                    target_length = int((window_radius_sec * 2) * sr)
                    
                    
                    if method != "hierro":
                        for ch_idx in range(3):
                            shape_raw = norm_envs[ch_idx]
                            
                            if len(shape_raw) > target_length:
                                shape_raw = shape_raw[:target_length]
                            elif len(shape_raw) < target_length:
                                shape_raw = np.pad(shape_raw, (0, target_length - len(shape_raw)), 'constant')
                                
                            # Aplicar STFT si es el modo seleccionado (método del Paper)
                            if method == "stft":
                                stft_window = int(0.25 * sr)
                                stft_overlap = int(0.125 * sr)
                                
                                f, t, Zxx = stft(shape_raw, fs=sr, nperseg=stft_window, noverlap=stft_overlap)
                                shape_raw = np.abs(Zxx).flatten()
                            elif method in ["env", "env_plus"]:
                                shape_raw[shape_raw < 0] = 0.0
                                shape_raw = resample(shape_raw, resample_points)
                                shape_raw[shape_raw < 0] = 0.0
                                
                                if use_log_transform:
                                    shape_raw = np.log1p(shape_raw)
                            else:
                                shape_raw[shape_raw < 0] = 0.0
                            
                            for t_i, val in enumerate(shape_raw):
                                col_name = f'Ch{ch_idx}_T{t_i}'
                                row_data[col_name] = f"{val:.6f}"
                                if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                                    dynamic_fieldnames.append(col_name)
                                    
                if method in ["env", "env_plus", "hierro"] and use_binarization:
                    global_max_raw = max([np.max(e) if len(e)>0 else 0 for e in segs_env]) if len(segs_env)>0 else 1e-9
                    # Umbrales óptimos derivados del análisis de sensibilidad: Ch0: 0.1, Ch1: 0.0, Ch2: 0.6
                    thresholds = [0.10, 0.00, 0.60]
                    for ch_idx in range(3):
                        e_raw = segs_env[ch_idx]
                        bin_val = 1.0 if (len(e_raw) > 0 and np.max(e_raw) > thresholds[ch_idx] * global_max_raw) else 0.0
                        
                        col_name = f'Ch{ch_idx}_BIN'
                        row_data[col_name] = f"{bin_val:.6f}"
                        if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                            dynamic_fieldnames.append(col_name)
                            
                if method in ["env_plus", "hierro"]:
                    for ch_idx in range(3):
                        e = norm_envs[ch_idx]
                        wl = np.sum(np.abs(np.diff(e))) if len(e)>1 else 0.0
                        var = np.var(e) if len(e)>0 else 0.0
                        mav = np.mean(np.abs(e)) if len(e)>0 else 0.0
                        
                        for feat_name, val in [('WL', wl), ('VAR', var), ('MAV', mav)]:
                            col_name = f'Ch{ch_idx}_{feat_name}'
                            row_data[col_name] = f"{val:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                                dynamic_fieldnames.append(col_name)
                    
                    for ch_a, ch_b in [(0, 1), (0, 2), (1, 2)]:
                        ea = norm_envs[ch_a]
                        eb = norm_envs[ch_b]
                        if len(ea) > 0 and len(eb) > 0:
                            corr = np.correlate(ea, eb, mode='full')
                            lag_samples = np.argmax(corr) - (len(eb) - 1)
                            lag_ms = (lag_samples / sr) * 1000.0
                        else:
                            lag_ms = 0.0
                        col_name = f'Lag_Ch{ch_a}_Ch{ch_b}'
                        row_data[col_name] = f"{lag_ms:.3f}"
                        if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0:
                            dynamic_fieldnames.append(col_name)

                if method == "custom":
                    # Extraer opciones de la UI
                    use_max = self.custom_vars["max"].get()
                    use_t_pico = self.custom_vars["tiempo_pico"].get()
                    use_area = self.custom_vars["area"].get()
                    use_ancho = self.custom_vars["ancho"].get()
                    use_ratios = self.custom_vars["ratios"].get()
                    use_lags = self.custom_vars["lags"].get()
                    use_deriv = self.custom_vars["derivada"].get()
                    use_accel = self.custom_vars["aceleracion"].get()
                    
                    max_vals = []
                    
                    for ch_idx in range(3):
                        e = norm_envs[ch_idx]
                        if len(e) == 0:
                            e = np.zeros(10) # Fallback seguro
                        
                        max_val = np.max(e)
                        max_vals.append(max_val)
                        
                        if use_max:
                            col = f'Ch{ch_idx}_Max'
                            row_data[col] = f"{max_val:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                        if use_t_pico:
                            t_pico = np.argmax(e) / sr * 1000.0
                            col = f'Ch{ch_idx}_TiempoPico'
                            row_data[col] = f"{t_pico:.3f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                        if use_area:
                            area = np.sum(e)
                            col = f'Ch{ch_idx}_Area'
                            row_data[col] = f"{area:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                        if use_ancho:
                            # FWHM (Full Width at Half Maximum)
                            half_max = max_val / 2.0
                            above_half = np.where(e >= half_max)[0]
                            ancho = len(above_half) / sr * 1000.0 if len(above_half) > 0 else 0.0
                            col = f'Ch{ch_idx}_Ancho'
                            row_data[col] = f"{ancho:.3f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                        if use_deriv:
                            vel = np.max(np.abs(np.gradient(e)))
                            col = f'Ch{ch_idx}_Derivada'
                            row_data[col] = f"{vel:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                        if use_accel:
                            accel = np.max(np.abs(np.gradient(np.gradient(e))))
                            col = f'Ch{ch_idx}_Acel'
                            row_data[col] = f"{accel:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                    if use_ratios:
                        for ch_a, ch_b in [(0, 1), (0, 2), (1, 2)]:
                            ratio = max_vals[ch_a] / max_vals[ch_b] if max_vals[ch_b] > 1e-6 else 0.0
                            col = f'Ratio_Ch{ch_a}_Ch{ch_b}'
                            row_data[col] = f"{ratio:.6f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)
                            
                    if use_lags:
                        from scipy.signal import correlate
                        for ch_a, ch_b in [(0, 1), (0, 2), (1, 2)]:
                            ea, eb = norm_envs[ch_a], norm_envs[ch_b]
                            if len(ea) > 0 and len(eb) > 0:
                                corr = correlate(ea, eb, mode='full', method='fft')
                                lag_samples = np.argmax(corr) - (len(eb) - 1)
                                lag_ms = (lag_samples / sr) * 1000.0
                            else:
                                lag_ms = 0.0
                            col = f'Lag_Ch{ch_a}_Ch{ch_b}'
                            row_data[col] = f"{lag_ms:.3f}"
                            if not dynamic_fieldnames_generated and ventanas_validas_procesadas == 0: dynamic_fieldnames.append(col)

                all_rows.append(row_data)
                ventanas_validas_procesadas += 1

            if not dynamic_fieldnames_generated and ventanas_validas_procesadas > 0:
                dynamic_fieldnames_generated = True

            self.progress_var.set((i + 1) / total * 100)
            self.root.update_idletasks()

        if all_rows:
            csv_path = os.path.join(out_dir, f"dataset_features_{method}.csv")
            fieldnames = ['Toma', 'Vocal'] + dynamic_fieldnames
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in all_rows:
                    writer.writerow(r)
            self.lbl_status.config(text=f"Éxito. Guardado en dataset_features_{method}.csv ({len(all_rows)} ventanas)")
            
            # --- Guardar archivo debug si es Modo Dios ---
            if method == "dios" and all_rows_debug:
                csv_debug_path = os.path.join(out_dir, "dataset_features_dios_ventanas.csv")
                fieldnames_debug = ['Toma', 'Vocal'] + dynamic_fieldnames_debug
                with open(csv_debug_path, 'w', newline='', encoding='utf-8') as f_dbg:
                    writer_dbg = csv.DictWriter(f_dbg, fieldnames=fieldnames_debug)
                    writer_dbg.writeheader()
                    for r in all_rows_debug:
                        writer_dbg.writerow(r)
                self.log(f"Dataset de validación visual guardado en: {csv_debug_path}")
            
            self.lbl_status.config(text="Calculando PCA y UMAP (Silhouette Score)...")
            self.root.update_idletasks()
            
            try:
                import subprocess
                pca_script = os.path.join(current_dir, "analisis_pca_umap.py")
                if os.path.exists(pca_script):
                    selected_key = self.scaler_mode_var.get()
                    scaler_flag = self.scaler_options_map.get(selected_key, "--center-only")
                    
                    cmd = [sys.executable, pca_script, method]
                    if scaler_flag:
                        cmd.append(scaler_flag)
                        
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    output = result.stdout
                    self.root.after(0, self.update_log, output)
                    self.lbl_status.config(text="Proceso de Extracción y Análisis COMPLETADO.")
                else:
                    self.lbl_status.config(text="No se encontró analisis_pca_umap.py")
            except Exception as e:
                self.lbl_status.config(text=f"Error al ejecutar PCA: {e}")
                
        else:
            self.lbl_status.config(text="No se generaron datos.")
            
        self.btn_extract.config(state=tk.NORMAL)

    def update_log(self, text):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, text)
        self.log_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = ExtractorGUI(root)
    root.mainloop()
