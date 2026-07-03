import os
import sys
import json
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, CheckButtons, Slider
from scipy.signal import find_peaks, butter, filtfilt, iirnotch

script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir))

def apply_dsp_pipeline(sig, sr, noise_seconds, smooth_ms=250.0, highpass_30hz=False, extreme_smooth=True):
    nyq = 0.5 * sr
    cutoff_hp = 30.0 if highpass_30hz else min(20.0, nyq * 0.99)
    b, a = butter(4, cutoff_hp / nyq, btype='high', analog=False)
    sig_filt = filtfilt(b, a, sig)
    f0 = 50.0
    if f0 < nyq:
        b, a = iirnotch(f0, 2.0, sr)
        sig_filt = filtfilt(b, a, sig_filt)
    cutoff_lp = min(500.0, nyq * 0.99)
    b, a = butter(4, cutoff_lp / nyq, btype='low', analog=False)
    sig_filt = filtfilt(b, a, sig_filt)
    window_size = int(smooth_ms / 1000.0 * sr)
    if window_size == 0: window_size = 1
    # Envolvente RMS (sin raíz o con media móvil, asumiendo lo que ya había en tu script)
    # Volviste a media móvil simple:
    env = np.convolve(np.abs(sig_filt), np.ones(window_size)/window_size, mode='same')
    
    if extreme_smooth:
        b_lp, a_lp = butter(2, 5.0 / nyq, btype='low', analog=False)
        env = filtfilt(b_lp, a_lp, env)
        env[env < 0] = 0
        
    initial_noise_mean = 0.0
    noise_samples = int(noise_seconds * sr)
    if noise_samples > 0 and noise_samples < len(env):
        skip = min(int(0.1 * sr), noise_samples // 2)
        initial_noise_mean = np.mean(env[skip:noise_samples])
    return env, initial_noise_mean

def smooth_derivative(deriv, sr):
    window_size_deriv = max(1, int(sr * 0.25)) 
    return np.convolve(deriv, np.ones(window_size_deriv)/window_size_deriv, mode='same')

def select_measurement_gui(base_dir):
    import tkinter as tk
    from tkinter import ttk
    import re
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    mediciones = []
    if os.path.exists(base_dir):
        for date_folder in sorted(os.listdir(base_dir), reverse=True):
            date_path = os.path.join(base_dir, date_folder)
            if os.path.isdir(date_path) and date_pattern.match(date_folder):
                for med_folder in sorted(os.listdir(date_path)):
                    med_path_full = os.path.join(date_path, med_folder)
                    if os.path.isdir(med_path_full):
                        mediciones.append(f"{date_folder}/{med_folder}")
    if not mediciones: return None
    root = tk.Tk()
    root.title("Selector de Tomas")
    root.geometry("400x150")
    root.eval('tk::PlaceWindow . center')
    selected_path = [None]
    ttk.Label(root, text="Seleccione la toma a visualizar:", font=("Arial", 11)).pack(pady=10)
    combo = ttk.Combobox(root, values=mediciones, state="readonly", width=45)
    combo.pack(pady=5)
    if mediciones: combo.current(0)
    def on_ok():
        selected_path[0] = os.path.join(base_dir, combo.get())
        root.destroy()
    ttk.Button(root, text="Graficar Derivadas", command=on_ok).pack(pady=10)
    root.mainloop()
    return selected_path[0]

class InteractiveViewer:
    def __init__(self, med_path, med_rel_path, raw_sig3, raw_sigs, noise, sr, periodo):
        self.med_path = med_path
        self.med_rel_path = med_rel_path
        self.raw_sig3 = raw_sig3
        self.raw_sigs = raw_sigs
        self.noise = noise
        self.sr = sr
        self.periodo = periodo
        
        self.highpass_30hz = False
        self.extreme_smooth = True
        self.pre_pct = 0.4
        self.post_pct = 0.6
        
        self.pulses_per_page = 5
        self.current_page = 0
        
        self.recompute_dsp()
        self.align_mode = 'amp'
        self.excluded_windows = []
        self.exclude_path = os.path.join(med_path, 'excluded_windows.json')
        if os.path.exists(self.exclude_path):
            try:
                with open(self.exclude_path, 'r') as f:
                    self.excluded_windows = json.load(f).get("excluded_windows", [])
            except: pass
        self.channels_visible = [True, True, True, True]
        self.setup_ui()
        self.draw_pulse()
        plt.show()

    def recompute_dsp(self):
        # Reprocesar micrófono
        self.env3, _ = apply_dsp_pipeline(self.raw_sig3, self.sr, self.noise, smooth_ms=250.0, highpass_30hz=self.highpass_30hz, extreme_smooth=self.extreme_smooth)
        
        dist = int(0.8 * self.periodo * self.sr)
        st = int(self.noise * self.sr)
        picos_amp, _ = find_peaks(self.env3[st:], distance=dist, height=np.max(self.env3[st:])*0.2)
        self.picos_amp = picos_amp + st
        
        deriv_env3 = smooth_derivative(np.gradient(self.env3), self.sr)
        self.picos_deriv = []
        p_samp = int(self.periodo * self.sr)
        for p in self.picos_amp:
            r = max(0, int(p - 0.4 * p_samp))
            if r < p: self.picos_deriv.append(r + np.argmax(deriv_env3[r:p]))
            else: self.picos_deriv.append(p)
            
        # Reprocesar músculos
        self.sigs = []
        for raw_sig in self.raw_sigs:
            env, _ = apply_dsp_pipeline(raw_sig, self.sr, self.noise, smooth_ms=250.0, highpass_30hz=self.highpass_30hz, extreme_smooth=self.extreme_smooth)
            self.sigs.append(env)
            
        self.total_pulses = len(self.picos_amp)
        import math
        self.total_pages = math.ceil(self.total_pulses / self.pulses_per_page)

    def get_current_picos(self):
        return self.picos_amp if self.align_mode == 'amp' else self.picos_deriv

    def setup_ui(self):
        self.fig, (self.ax_sig, self.ax_deriv) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        self.fig.subplots_adjust(bottom=0.25, right=0.85)
        for ax in [self.ax_sig, self.ax_deriv]: ax.set_facecolor('#1c1c1c')
        self.fig.patch.set_facecolor('#1c1c1c')
        
        # Botones de Navegación por Páginas
        ax_prev = plt.axes([0.25, 0.05, 0.08, 0.05])
        self.b_prev = Button(ax_prev, '< Página')
        self.b_prev.on_clicked(self.prev_page)
        
        ax_next = plt.axes([0.34, 0.05, 0.08, 0.05])
        self.b_next = Button(ax_next, 'Página >')
        self.b_next.on_clicked(self.next_page)
        
        # Botón Guardar
        ax_save = plt.axes([0.45, 0.05, 0.2, 0.05])
        self.b_save = Button(ax_save, 'Guardar Exclusiones en JSON')
        self.b_save.on_clicked(self.save_exclusions)
        
        # Opciones DSP
        ax_dsp = plt.axes([0.7, 0.02, 0.15, 0.12], facecolor='#2c3e50')
        self.check_dsp = CheckButtons(ax_dsp, ('High-Pass 30Hz', 'Suavizado Extremo (5Hz)'), (self.highpass_30hz, self.extreme_smooth))
        for label in self.check_dsp.labels: label.set_color('white')
        self.check_dsp.on_clicked(self.toggle_dsp)
        
        # Sliders Ventana
        ax_pre = plt.axes([0.1, 0.15, 0.2, 0.02], facecolor='#2c3e50')
        self.slider_pre = Slider(ax_pre, 'Pre-Ventana (%)', 0.1, 0.8, valinit=self.pre_pct, valstep=0.05)
        self.slider_pre.on_changed(self.update_window)
        
        ax_post = plt.axes([0.1, 0.1, 0.2, 0.02], facecolor='#2c3e50')
        self.slider_post = Slider(ax_post, 'Post-Ventana (%)', 0.1, 0.9, valinit=self.post_pct, valstep=0.05)
        self.slider_post.on_changed(self.update_window)
        
        # RadioButtons Alineación
        ax_radio = plt.axes([0.87, 0.5, 0.12, 0.15], facecolor='#2c3e50')
        self.radio = RadioButtons(ax_radio, ('Pico Mic', 'Derivada Mic'), active=0)
        for label in self.radio.labels: label.set_color('white')
        self.radio.on_clicked(self.change_alignment)
        
        # CheckButtons Canales
        ax_check = plt.axes([0.87, 0.7, 0.12, 0.2], facecolor='#2c3e50')
        self.check = CheckButtons(ax_check, ('Miloioide', 'Depresor', 'Orbicularis', 'Micrófono'), (True, True, True, True))
        for label in self.check.labels: label.set_color('white')
        self.check.on_clicked(self.toggle_channels)
        
        # Conectar clic en el gráfico para excluir
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def toggle_channels(self, label):
        idx = ['Miloioide', 'Depresor', 'Orbicularis', 'Micrófono'].index(label)
        self.channels_visible[idx] = not self.channels_visible[idx]
        self.draw_pulse()
        
    def toggle_dsp(self, label):
        if label == 'High-Pass 30Hz': self.highpass_30hz = not self.highpass_30hz
        if label == 'Suavizado Extremo (5Hz)': self.extreme_smooth = not self.extreme_smooth
        self.recompute_dsp()
        self.draw_pulse()
        
    def update_window(self, val):
        self.pre_pct = self.slider_pre.val
        self.post_pct = self.slider_post.val
        self.draw_pulse()

    def change_alignment(self, label):
        self.align_mode = 'amp' if label == 'Pico Mic' else 'deriv'
        self.draw_pulse()

    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.draw_pulse()

    def next_page(self, event):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.draw_pulse()

    def on_click(self, event):
        if event.inaxes in [self.ax_sig, self.ax_deriv]:
            if event.xdata is not None:
                # Cada pulso ocupa exactamente self.periodo segundos en el eje X
                idx_clicked = int(event.xdata // self.periodo)
                if 0 <= idx_clicked < self.total_pulses:
                    if idx_clicked in self.excluded_windows:
                        self.excluded_windows.remove(idx_clicked)
                    else:
                        self.excluded_windows.append(idx_clicked)
                    self.draw_pulse()

    def save_exclusions(self, event):
        with open(self.exclude_path, 'w') as f:
            json.dump({"excluded_windows": sorted(list(self.excluded_windows))}, f, indent=4)
        print(f"Exclusiones guardadas en {self.exclude_path}")
        self.b_save.color = '#27ae60'
        self.fig.canvas.draw_idle()

    def draw_pulse(self):
        self.ax_sig.clear()
        self.ax_deriv.clear()
        
        picos = self.get_current_picos()
        colors = ['#39ff14', '#8a2be2', '#ffd700']
        labels = ['Miloioide', 'Depresor', 'Orbicularis']
        
        start_idx = self.current_page * self.pulses_per_page
        end_idx = min(self.total_pulses, start_idx + self.pulses_per_page)
        
        for i in range(start_idx, end_idx):
            pico = picos[i]
            t_pico = pico / self.sr
            t_start = t_pico - 0.5 * self.periodo
            t_end = t_pico + 0.5 * self.periodo
            idx_start = max(0, int(t_start * self.sr))
            idx_end = min(len(self.env3), int(t_end * self.sr))
            
            if idx_end <= idx_start: continue
            
            # Tiempo relativo del segmento mapeado al eje X global (concatenado)
            t_local = np.linspace(-0.5 * self.periodo, 0.5 * self.periodo, idx_end - idx_start)
            t_global = t_local + (i * self.periodo) + (0.5 * self.periodo)
            
            if i in self.excluded_windows:
                self.ax_sig.axvspan(i * self.periodo, (i+1) * self.periodo, color='#4a0000', alpha=0.5)
                self.ax_deriv.axvspan(i * self.periodo, (i+1) * self.periodo, color='#4a0000', alpha=0.5)
                
            # Micrófono
            if self.channels_visible[3]:
                mic_seg = self.env3[idx_start:idx_end].copy()
                if np.max(mic_seg) > 0: mic_seg = mic_seg / np.max(mic_seg)
                self.ax_sig.plot(t_global, mic_seg, color='red', label='Micrófono' if i==0 else "", linewidth=2.5, alpha=0.3)
                
                deriv_mic = smooth_derivative(np.gradient(mic_seg), self.sr)
                if np.max(np.abs(deriv_mic)) > 0: deriv_mic = deriv_mic / np.max(np.abs(deriv_mic))
                self.ax_deriv.plot(t_global, deriv_mic, color='red', linewidth=2.5, alpha=0.4, label='Deriv. Mic' if i==0 else "")
                
                idx_peak_mic = np.argmax(deriv_mic)
                self.ax_deriv.plot(t_global[idx_peak_mic], deriv_mic[idx_peak_mic], 'o', color='red', markersize=8)
                self.ax_sig.axvline(t_global[idx_peak_mic], color='red', linestyle=':', alpha=0.4)

            # Músculos
            for ch in range(3):
                if self.channels_visible[ch]:
                    seg = self.sigs[ch][idx_start:idx_end].copy()
                    if np.max(seg) > 0: seg = seg / np.max(seg)
                    self.ax_sig.plot(t_global, seg, color=colors[ch], label=labels[ch] if i==0 else "", linewidth=2, alpha=0.9)
                    
                    deriv_ch = smooth_derivative(np.gradient(seg), self.sr)
                    if np.max(np.abs(deriv_ch)) > 0: deriv_ch = deriv_ch / np.max(np.abs(deriv_ch))
                    self.ax_deriv.plot(t_global, deriv_ch, color=colors[ch], linewidth=1.5, alpha=0.8, label=f'Deriv. {labels[ch]}' if i==0 else "")
                    
                    idx_peak_ch = np.argmax(deriv_ch)
                    self.ax_deriv.plot(t_global[idx_peak_ch], deriv_ch[idx_peak_ch], 'o', color=colors[ch], markersize=6)
                    self.ax_sig.axvline(t_global[idx_peak_ch], color=colors[ch], linestyle=':', alpha=0.5)
                    
            # Centro y separador
            centro_global = (i * self.periodo) + (0.5 * self.periodo)
            self.ax_sig.axvline(centro_global, color='white', linestyle='-', linewidth=2, alpha=0.8, label='Centro' if i==0 else "")
            self.ax_deriv.axvline(centro_global, color='white', linestyle='-', linewidth=2, alpha=0.8)
            
            # Límites de la ventana de extracción PCA
            start_pca = centro_global - self.pre_pct * self.periodo
            end_pca = centro_global + self.post_pct * self.periodo
            pulse_start = i * self.periodo
            pulse_end = (i + 1) * self.periodo
            
            # Máscaras oscuras para tapar lo que queda afuera de la ventana
            self.ax_sig.axvspan(pulse_start, start_pca, color='#1c1c1c', alpha=0.9, zorder=10)
            self.ax_deriv.axvspan(pulse_start, start_pca, color='#1c1c1c', alpha=0.9, zorder=10)
            
            self.ax_sig.axvspan(end_pca, pulse_end, color='#1c1c1c', alpha=0.9, zorder=10)
            self.ax_deriv.axvspan(end_pca, pulse_end, color='#1c1c1c', alpha=0.9, zorder=10)
            
            # Líneas de los bordes
            self.ax_sig.axvline(start_pca, color='#3498db', linestyle='-', linewidth=2, alpha=0.9, zorder=11, label='Inicio Ventana' if i==0 else "")
            self.ax_sig.axvline(end_pca, color='#e74c3c', linestyle='-', linewidth=2, alpha=0.9, zorder=11, label='Fin Ventana' if i==0 else "")
            
            self.ax_sig.axvline((i+1) * self.periodo, color='#bdc3c7', linestyle='--', linewidth=1, alpha=0.5)
            self.ax_deriv.axvline((i+1) * self.periodo, color='#bdc3c7', linestyle='--', linewidth=1, alpha=0.5)
            
            # Texto indicador de pulso
            self.ax_sig.text(centro_global, 1.05, f"P {i+1}", color='white', ha='center', fontsize=10, fontweight='bold')

        self.ax_deriv.axhline(0, color='gray', linestyle=':', alpha=0.5)
        self.ax_sig.set_xlim(start_idx * self.periodo, (start_idx + self.pulses_per_page) * self.periodo)
        
        excluidas_str = ", ".join([str(x+1) for x in sorted(self.excluded_windows)])
        texto_excl = f" | Excluidas totales: [{excluidas_str}]" if self.excluded_windows else ""
        self.ax_sig.set_title(f"Toma: {self.med_rel_path} | Página {self.current_page + 1} de {self.total_pages} (Clic en un pulso para excluirlo){texto_excl}", color='white', fontsize=14)
        
        for ax in [self.ax_sig, self.ax_deriv]:
            ax.grid(True, linestyle=':', alpha=0.3, color='#bdc3c7')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#7f8c8d')
            ax.spines['bottom'].set_color('#7f8c8d')
            ax.tick_params(colors='#ecf0f1')
            ax.legend(loc='upper right', frameon=True, facecolor='#2c3e50', edgecolor='#34495e', labelcolor='white')

        self.ax_sig.set_ylabel("Amplitud Envolvente", color='white')
        self.ax_deriv.set_ylabel("Derivada", color='white')
        self.ax_deriv.set_xlabel("Tiempo Absoluto (segundos)", color='white')
        self.fig.canvas.draw_idle()

def main():
    if len(sys.argv) < 2:
        base_dir = os.path.abspath(os.path.join(deep_learning_dir, "..", "base_de_datos_electrodos"))
        med_path = select_measurement_gui(base_dir)
        if not med_path:
            print("No se seleccionó ninguna medición.")
            sys.exit(0)
    else: med_path = sys.argv[1]
    med_rel_path = os.path.basename(med_path)
    
    bpm, noise = 30, 2.0
    meta_path = os.path.join(med_path, 'canal_0', 'metadata.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                data = json.load(f)
                bpm = data.get('bpm', bpm)
                noise = data.get('noise_seconds', noise)
        except: pass

    periodo = 60.0 / bpm
    ch3_dir = os.path.join(med_path, "canal_3")
    wavs3 = [f for f in os.listdir(ch3_dir) if f.endswith(".wav")] if os.path.exists(ch3_dir) else []
    if not wavs3:
        print(f"Falta canal_3 en: {med_path}")
        sys.exit(1)
        
    raw_sig3, sr = sf.read(os.path.join(ch3_dir, wavs3[0]))
    if raw_sig3.ndim > 1: raw_sig3 = raw_sig3[:, 0]
    
    raw_sigs = []
    for ch in ['canal_0', 'canal_1', 'canal_2']:
        ch_dir = os.path.join(med_path, ch)
        wavs = [f for f in os.listdir(ch_dir) if f.endswith(".wav")] if os.path.exists(ch_dir) else []
        if not wavs:
            raw_sigs.append(np.zeros_like(raw_sig3))
            continue
        sig, sr_ch = sf.read(os.path.join(ch_dir, wavs[0]))
        if sig.ndim > 1: sig = sig[:, 0]
        raw_sigs.append(sig)
        
    viewer = InteractiveViewer(med_path, med_rel_path, raw_sig3, raw_sigs, noise, sr, periodo)

if __name__ == "__main__":
    main()
