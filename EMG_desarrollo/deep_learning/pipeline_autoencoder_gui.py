import tkinter as tk
from tkinter import ttk, messagebox
import os
import sys
import os

# Agregamos rutas para poder importar los módulos
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
if script_dir_abs not in sys.path:
    sys.path.append(script_dir_abs)
if os.path.join(script_dir_abs, "dataset_tools") not in sys.path:
    sys.path.append(os.path.join(script_dir_abs, "dataset_tools"))

import threading
import subprocess

# Para invocar lógicas existentes
import generador_pca_tensorial as gpt
import train_autoencoder as ta
import plot_latent_space as pls

class PipelineAutoencoderGUI:
    def __init__(self, root, rutas_preseleccionadas=None):
        self.root = root
        self.root.title("Pipeline Maestro de Autoencoder & Deep Learning")
        self.root.geometry("800x850")
        self.root.configure(bg="#0B0C10")
        
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.cyan_dim = "#45A29E"
        self.fg_text = "#C5C6C7"
        self.green_neon = "#00FF00"
        
        self.rutas_preseleccionadas = rutas_preseleccionadas or []
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_electrodos")
        
        self.setup_ui()
        
    def setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TLabel", background=self.bg_panel, foreground=self.fg_text, font=("Arial", 10))
        style.configure("TFrame", background=self.bg_panel)
        style.configure("TButton", font=("Arial", 10, "bold"), background=self.cyan_dim, foreground="black")
        
        main_frame = tk.Frame(self.root, bg=self.bg_dark, padx=20, pady=20)
        main_frame.pack(fill="both", expand=True)
        
        lbl_title = tk.Label(main_frame, text="PIPELINE MAESTRO: AUTOENCODER 1D", bg=self.bg_dark, fg=self.cyan_neon, font=("Arial", 16, "bold"))
        lbl_title.pack(pady=(0, 15))
        
        # --- PANEL DE MEDICIONES ---
        frame_med = tk.LabelFrame(main_frame, text=" 1. Selección de Mediciones ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_med.pack(fill="x", pady=5)
        
        scrollbar = tk.Scrollbar(frame_med)
        scrollbar.pack(side="right", fill="y")
        
        self.listbox_med = tk.Listbox(frame_med, selectmode=tk.MULTIPLE, height=6, bg="#111111", fg="white", yscrollcommand=scrollbar.set)
        self.listbox_med.pack(side="left", fill="x", expand=True)
        scrollbar.config(command=self.listbox_med.yview)
        
        # --- PANEL DE EXTRACCIÓN TENSORIAL ---
        frame_dsp = tk.LabelFrame(main_frame, text=" 2. Hiperparámetros de Extracción DSP (Tensorial) ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_dsp.pack(fill="x", pady=5)
        
        grid_dsp = tk.Frame(frame_dsp, bg=self.bg_panel)
        grid_dsp.pack(fill="x")
        
        params_dsp = [
            ("Alpha Ruido:", "0.4", "ent_alpha"),
            ("SNR Min:", "2.0", "ent_snr"),
            ("Outliers (%):", "0.05", "ent_outliers"),
            ("Smooth (ms):", "50", "ent_smooth"),
            ("Notch Q:", "30", "ent_notch_q")
        ]
        
        for i, (label_text, default_val, attr_name) in enumerate(params_dsp):
            row = i // 3
            col = (i % 3) * 2
            tk.Label(grid_dsp, text=label_text, bg=self.bg_panel, fg=self.fg_text).grid(row=row, column=col, sticky="e", padx=5, pady=5)
            ent = tk.Entry(grid_dsp, width=8, bg="#111111", fg="white", insertbackground="white")
            ent.insert(0, default_val)
            ent.grid(row=row, column=col+1, sticky="w", padx=5, pady=5)
            setattr(self, attr_name, ent)
            
        # --- PANEL DE AUTOENCODER ---
        frame_nn = tk.LabelFrame(main_frame, text=" 3. Hiperparámetros de Red Neuronal ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_nn.pack(fill="x", pady=5)
        
        grid_nn = tk.Frame(frame_nn, bg=self.bg_panel)
        grid_nn.pack(fill="x")
        
        params_nn = [
            ("Épocas:", "150", "ent_epochs"),
            ("Batch Size:", "32", "ent_batch"),
            ("Latent Dim:", "16", "ent_latent")
        ]
        
        for i, (label_text, default_val, attr_name) in enumerate(params_nn):
            tk.Label(grid_nn, text=label_text, bg=self.bg_panel, fg=self.fg_text).grid(row=0, column=i*2, sticky="e", padx=5, pady=5)
            ent = tk.Entry(grid_nn, width=8, bg="#111111", fg="white", insertbackground="white")
            ent.insert(0, default_val)
            ent.grid(row=0, column=i*2+1, sticky="w", padx=5, pady=5)
            setattr(self, attr_name, ent)
            
        # --- BOTON DE EJECUCION ---
        self.btn_ejecutar = tk.Button(main_frame, text="EJECUTAR PIPELINE COMPLETO", font=("Arial", 14, "bold"), bg=self.green_neon, fg="black", command=self.ejecutar_pipeline)
        self.btn_ejecutar.pack(fill="x", pady=15, ipady=10)
        
        # --- LOG CONSOLE ---
        self.log_text = tk.Text(main_frame, height=8, bg="#111111", fg="#00FF00", font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        # --- HERRAMIENTAS ADICIONALES ---
        frame_tools = tk.Frame(main_frame, bg=self.bg_dark)
        frame_tools.pack(fill="x", pady=5)
        
        tk.Button(frame_tools, text="Visualizador de Features", bg="#333333", fg="white", command=self.lanzar_visor).pack(side="left", fill="x", expand=True, padx=2)
        tk.Button(frame_tools, text="Plot 3 Músculos (Paper)", bg="#333333", fg="white", command=self.lanzar_plot_musculos).pack(side="right", fill="x", expand=True, padx=2)
        
        self.cargar_mediciones()

    def log(self, mensaje):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, mensaje + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")
        self.root.update_idletasks()

    def cargar_mediciones(self):
        mediciones = gpt.procesar_mediciones(self.base_dir)
        for i, med in enumerate(mediciones):
            self.listbox_med.insert(tk.END, med)
            if self.rutas_preseleccionadas:
                # Si viene preseleccionado desde main_app
                if any(med.replace("/", "\\") in rp.replace("/", "\\") for rp in self.rutas_preseleccionadas):
                    self.listbox_med.selection_set(i)
        
        if not self.rutas_preseleccionadas and mediciones:
            # Seleccionar todas por defecto
            self.listbox_med.select_set(0, tk.END)

    def ejecutar_pipeline(self):
        seleccionadas = [self.listbox_med.get(i) for i in self.listbox_med.curselection()]
        if not seleccionadas:
            messagebox.showwarning("Advertencia", "Seleccione al menos una medición.")
            return
            
        try:
            val_alpha = float(self.ent_alpha.get())
            val_snr = float(self.ent_snr.get())
            val_out = float(self.ent_outliers.get())
            val_smooth = int(self.ent_smooth.get())
            val_notch_q = float(self.ent_notch_q.get())
            
            v_epochs = int(self.ent_epochs.get())
            v_batch = int(self.ent_batch.get())
            v_latent = int(self.ent_latent.get())
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos.")
            return
            
        self.btn_ejecutar.config(state="disabled")
        self.log("=========================================")
        self.log("INICIANDO PIPELINE DE AUTOENCODER")
        self.log("=========================================")
        
        # Ejecutamos en un thread para no bloquear la GUI
        threading.Thread(target=self._pipeline_thread, args=(seleccionadas, val_alpha, val_snr, val_out, val_smooth, val_notch_q, v_epochs, v_batch, v_latent)).start()

    def _pipeline_thread(self, seleccionadas, val_alpha, val_snr, val_out, val_smooth, val_notch_q, v_epochs, v_batch, v_latent):
        import traceback
        try:
            self.log("[1/3] Extracción Tensorial de Features...")
            # Sobrescribimos el print temporalmente para que salga en la consola
            old_stdout = sys.stdout
            class LogWriter:
                def __init__(self, log_func): self.log_func = log_func
                def write(self, t): 
                    if t.strip(): self.log_func(t.strip())
                def flush(self): pass
            
            sys.stdout = LogWriter(self.log)
            
            # 1. Extracción Tensorial
            gpt.ejecutar_procesamiento(
                seleccionadas,
                alpha_ruido=val_alpha,
                snr_threshold=val_snr,
                outlier_contamination=val_out,
                smooth_ms=val_smooth,
                notch_q=val_notch_q
            )
            
            # 2. Entrenamiento
            self.log("\n[2/3] Entrenando Autoencoder Conv1D...")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
            csv_file = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
            
            if not os.path.exists(csv_file):
                raise Exception(f"No se encontró el dataset en {csv_file}")
                
            ta.train_autoencoder(csv_file, epochs=v_epochs, batch_size=v_batch, latent_dim=v_latent)
            
            # 3. Ploteo de Espacio Latente
            self.log("\n[3/3] Generando mapa UMAP 3D del espacio latente...")
            model_path = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder", f"autoencoder_emg_{v_latent}d.pth")
            if not os.path.exists(model_path):
                raise Exception(f"No se encontró el modelo entrenado en {model_path}")
                
            pls.plot_latent_space(csv_file, model_path, latent_dim=v_latent)
            
            sys.stdout = old_stdout
            self.log("\n>>> PIPELINE COMPLETADO EXITOSAMENTE <<<")
            messagebox.showinfo("Éxito", "El Pipeline del Autoencoder se ejecutó correctamente.")
        except Exception as e:
            sys.stdout = old_stdout
            self.log(f"\n[ERROR] El pipeline falló: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error de Pipeline", f"Ocurrió un error:\n{e}")
        finally:
            self.btn_ejecutar.config(state="normal")
            
    def lanzar_visor(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visor_path = os.path.join(script_dir, "dataset_tools", "visor_features.py")
        subprocess.Popen([sys.executable, visor_path])
        
    def lanzar_plot_musculos(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, "dataset_tools", "plot_3_musculos_standalone.py")
        subprocess.Popen([sys.executable, plot_path])

if __name__ == "__main__":
    root = tk.Tk()
    rutas = sys.argv[1:] if len(sys.argv) > 1 else None
    app = PipelineAutoencoderGUI(root, rutas_preseleccionadas=rutas)
    root.mainloop()
