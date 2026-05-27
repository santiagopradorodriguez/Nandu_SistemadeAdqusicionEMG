import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import signal
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QListWidget, QLabel, QPushButton, QMessageBox, 
    QAbstractItemView, QLineEdit
)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def compute_env(sig, fs, smooth_ms=50):
    """Calcula la envolvente usando Hilbert y Media Móvil (SMA)"""
    sig_abs = np.abs(sig)
    env = np.abs(signal.hilbert(sig_abs))
    if smooth_ms > 0:
        win_len = int(max(1, round(smooth_ms * fs / 1000.0)))
        if win_len > 1:
            window = np.ones(win_len, dtype=float) / float(win_len)
            env = np.convolve(env, window, mode='same')
    return env

def apply_filters(sig, fs, use_notch=True):
    """Aplica Pasa-altos, Notch (opcional) y Pasa-bajos"""
    nyq = 0.5 * fs
    # Pasa-Altos (20 Hz)
    b, a = signal.butter(4, 20/nyq, btype='high')
    sig_filt = signal.filtfilt(b, a, sig)
    
    # Notch (50 Hz)
    if use_notch:
        # Se usa Q=2.0 según la recomendación en los archivos de proyecto para mayor estabilidad
        b, a = signal.iirnotch(50.0, 2.0, fs)
        sig_filt = signal.filtfilt(b, a, sig_filt)
        
    # Pasa-Bajos (500 Hz)
    limite = min(500/nyq, 0.99)
    b, a = signal.butter(4, limite, btype='low')
    sig_filt = signal.filtfilt(b, a, sig_filt)
    
    return sig_filt

def process_measurement(folder_path):
    csv_path = os.path.join(folder_path, 'grabacion.csv')
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error leyendo {csv_path}: {e}")
        return None
        
    col_tiempo = df.columns[0]
    cols_canales = [col for col in df.columns[1:] if 'Canal' in col]
    if not cols_canales:
        return None
    
    canal = cols_canales[0]
    raw_sig = df[canal].values
    tiempo = df[col_tiempo].values
    
    try:
        fs = 1 / (tiempo[1] - tiempo[0])
    except:
        fs = 2000.0
        
    # Buscar noise_seconds y resistencia_ohm en metadata
    noise_seconds = 2.0 
    ganancia = 495.0 # Ganancia por defecto
    try:
        for item in os.listdir(folder_path):
            if os.path.isdir(os.path.join(folder_path, item)) and item.startswith("canal_"):
                meta_path = os.path.join(folder_path, item, 'metadata.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        md = json.load(f)
                        if 'noise_seconds' in md and md['noise_seconds']:
                            noise_seconds = float(md['noise_seconds'])
                        if 'resistencia_ohm' in md and md['resistencia_ohm']:
                            res_ohm = float(md['resistencia_ohm'])
                            ganancia = 1.0 + (49400.0 / res_ohm)
                    break
    except:
        pass

    # Aplicar calibración a microvoltios (µV)
    raw_sig = (raw_sig / ganancia) * 1e6

    noise_samples = int(noise_seconds * fs)
    if noise_samples == 0 or noise_samples > len(raw_sig):
        noise_samples = min(int(2.0 * fs), len(raw_sig))
        
    def calc_metrics(env_signal, n_samples):
        # Ruido Inicial: Promedio y SEM de la ventana inicial
        ini_env = env_signal[:n_samples]
        chunk_size = int(fs * 0.25) # bloques de 250ms
        if len(ini_env) >= chunk_size * 2:
            chunks = [np.mean(ini_env[i:i+chunk_size]) for i in range(0, len(ini_env), chunk_size) if len(ini_env[i:i+chunk_size]) == chunk_size]
            ini_mean = np.mean(chunks)
            ini_err = np.std(chunks, ddof=1) / np.sqrt(len(chunks))
        else:
            ini_mean = np.mean(ini_env)
            ini_err = np.std(ini_env) / np.sqrt(len(ini_env)) if len(ini_env) > 0 else 0.0
            
        # Ruido Inter-pulso: Aprox global (Std global dividida en bloques de 1 seg)
        chunk_size = int(fs * 1.0)
        if len(env_signal) >= chunk_size * 2:
            chunks_std = [np.std(env_signal[i:i+chunk_size]) for i in range(0, len(env_signal), chunk_size) if len(env_signal[i:i+chunk_size]) == chunk_size]
            inter_mean = np.mean(chunks_std)
            inter_err = np.std(chunks_std, ddof=1) / np.sqrt(len(chunks_std))
        else:
            inter_mean = np.std(env_signal)
            inter_err = 0.0
            
        return ini_mean, ini_err, inter_mean, inter_err

    # Rama 1: Sin Notch
    sig_no_notch = apply_filters(raw_sig, fs, use_notch=False)
    env_no_notch = compute_env(sig_no_notch, fs)
    r_ini_nn, err_ini_nn, r_int_nn, err_int_nn = calc_metrics(env_no_notch, noise_samples)
    
    # Rama 2: Con Notch
    sig_notch = apply_filters(raw_sig, fs, use_notch=True)
    env_notch = compute_env(sig_notch, fs)
    r_ini_n, err_ini_n, r_int_n, err_int_n = calc_metrics(env_notch, noise_samples)
    
    # Obtenemos ruta absoluta para display
    abs_csv_path = os.path.abspath(csv_path)
    
    return {
        'nombre_csv': abs_csv_path,
        'ruido_inicial_no_notch': r_ini_nn,
        'err_inicial_no_notch': err_ini_nn,
        'ruido_inter_no_notch': r_int_nn,
        'err_inter_no_notch': err_int_nn,
        'ruido_inicial_notch': r_ini_n,
        'err_inicial_notch': err_ini_n,
        'ruido_inter_notch': r_int_n,
        'err_inter_notch': err_int_n
    }

class SelectorDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Análisis Comparativo - Selección de Carpetas")
        self.resize(800, 600)
        self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")
        self.seleccionadas = []
        self.prefijo_graficos = "comparativa"
        
        layout = QVBoxLayout(self)
        
        # Campo para nombre de gráficos
        h_layout = QHBoxLayout()
        h_layout.addWidget(QLabel("Prefijo para guardar gráficos:"))
        self.txt_prefijo = QLineEdit("comparativa_exp1")
        self.txt_prefijo.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #333;")
        h_layout.addWidget(self.txt_prefijo)
        layout.addLayout(h_layout)
        
        lbl = QLabel("Seleccione las carpetas a comparar (use Ctrl/Shift para selección múltiple):")
        layout.addWidget(lbl)
        
        self.listbox = QListWidget()
        self.listbox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.listbox.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #333;")
        
        self.base_dir_abs = "/home/lbraun/Repos/Nandu_SistemadeAdqusicionEMG/Emg/base_de_datos_electrodos/"
        if os.path.exists(self.base_dir_abs):
            for root, dirs, files in os.walk(self.base_dir_abs):
                if 'grabacion.csv' in files:
                    rel_path = os.path.relpath(root, self.base_dir_abs)
                    self.listbox.addItem(rel_path)
        else:
            QMessageBox.warning(self, "Atención", f"No se encontró el directorio base: {self.base_dir_abs}")
                    
        layout.addWidget(self.listbox)
        
        btn = QPushButton("Analizar Seleccionadas")
        btn.setStyleSheet("background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px;")
        btn.clicked.connect(self.confirmar)
        layout.addWidget(btn)
        
    def confirmar(self):
        items = self.listbox.selectedItems()
        if not items:
            QMessageBox.warning(self, "Error", "Debe seleccionar al menos una carpeta para comparar.")
            return
        
        pref = self.txt_prefijo.text().strip()
        if pref:
            self.prefijo_graficos = pref
            
        self.seleccionadas = [os.path.join(self.base_dir_abs, item.text()) for item in items]
        self.accept()

def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
        
    dialog = SelectorDialog()
    if dialog.exec() != QDialog.Accepted:
        return
        
    carpetas = dialog.seleccionadas
    prefijo = dialog.prefijo_graficos
    resultados = []
    
    print("\n[+] Iniciando análisis comparativo...")
    for f in carpetas:
        print(f"    Analizando: {f}")
        res = process_measurement(f)
        if res:
            resultados.append(res)
            
    if not resultados:
        print("\n[!] No se obtuvieron resultados válidos.")
        return
        
    # Ordenar resultados de menor a mayor ruido inicial (Con Notch)
    resultados.sort(key=lambda x: x['ruido_inicial_notch'])
    
    print("\n" + "="*80)
    print("RESULTADOS ORDENADOS DE MENOR A MAYOR RUIDO (CON NOTCH)")
    print("="*80)
    for i, r in enumerate(resultados):
        print(f"[{i+1}] Archivo: {r['nombre_csv']}")
        print(f"    Ruido Inicial (Con Notch):      {r['ruido_inicial_notch']:7.2f} ± {r['err_inicial_notch']:5.2f} µV")
        print(f"    Ruido Inter-pulso (Con Notch):  {r['ruido_inter_notch']:7.2f} ± {r['err_inter_notch']:5.2f} µV")
        print(f"    Ruido Inicial (Sin Notch):      {r['ruido_inicial_no_notch']:7.2f} ± {r['err_inicial_no_notch']:5.2f} µV")
        print(f"    Ruido Inter-pulso (Sin Notch):  {r['ruido_inter_no_notch']:7.2f} ± {r['err_inter_no_notch']:5.2f} µV")
        print("-"*80)
        
    # Gráficos
    nombres = [os.path.basename(os.path.dirname(r['nombre_csv'])) for r in resultados]
    
    r_ini_n = [r['ruido_inicial_notch'] for r in resultados]
    e_ini_n = [r['err_inicial_notch'] for r in resultados]
    r_ini_nn = [r['ruido_inicial_no_notch'] for r in resultados]
    e_ini_nn = [r['err_inicial_no_notch'] for r in resultados]
    
    r_int_n = [r['ruido_inter_notch'] for r in resultados]
    e_int_n = [r['err_inter_notch'] for r in resultados]
    r_int_nn = [r['ruido_inter_no_notch'] for r in resultados]
    e_int_nn = [r['err_inter_no_notch'] for r in resultados]
    
    x = np.arange(len(nombres))
    width = 0.35
    
    # 1. Gráfico de Ruido Inicial
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    rects1_a = ax1.bar(x - width/2, r_ini_n, width, yerr=e_ini_n, capsize=5, label='Con Notch (50Hz)', color='#1f77b4', ecolor='black')
    rects1_b = ax1.bar(x + width/2, r_ini_nn, width, yerr=e_ini_nn, capsize=5, label='Sin Notch', color='#ff7f0e', ecolor='black')
    
    ax1.set_ylabel('Ruido Inicial Promedio (µV)', fontsize=14)
    ax1.set_title('Comparativa de Ruido Inicial por Medición\n(Ordenado de Menor a Mayor)', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(nombres, rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    fig1.tight_layout()
    fig1.savefig(f'{prefijo}_ruido_inicial.png', dpi=150)
    
    # 2. Gráfico de Ruido Inter-pulso
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    rects2_a = ax2.bar(x - width/2, r_int_n, width, yerr=e_int_n, capsize=5, label='Con Notch (50Hz)', color='#2ca02c', ecolor='black')
    rects2_b = ax2.bar(x + width/2, r_int_nn, width, yerr=e_int_nn, capsize=5, label='Sin Notch', color='#d62728', ecolor='black')
    
    ax2.set_ylabel('Ruido Inter-pulso RMS/Std (µV)', fontsize=14)
    ax2.set_title('Comparativa de Ruido Inter-pulso por Medición\n(Ordenado de Menor a Mayor)', fontsize=16, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(nombres, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    fig2.tight_layout()
    fig2.savefig(f'{prefijo}_ruido_interpulso.png', dpi=150)
    
    print(f"\n[+] Gráficos guardados en el directorio actual como '{prefijo}_ruido_inicial.png' y '{prefijo}_ruido_interpulso.png'")
    plt.show()

if __name__ == "__main__":
    main()
