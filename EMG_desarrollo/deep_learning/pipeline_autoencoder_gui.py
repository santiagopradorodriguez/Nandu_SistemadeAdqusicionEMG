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
        self.root.title("Autoencoder")
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
        
        lbl_title = tk.Label(main_frame, text="AUTOENCODER 1D", bg=self.bg_dark, fg=self.cyan_neon, font=("Arial", 16, "bold"))
        lbl_title.pack(pady=(0, 15))
        
        # --- PANEL DE MEDICIONES ---
        frame_med = tk.LabelFrame(main_frame, text=" 1. Selección de Mediciones ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_med.pack(fill="x", pady=5)
        
        scrollbar = tk.Scrollbar(frame_med)
        scrollbar.pack(side="right", fill="y")
        
        self.listbox_med = tk.Listbox(frame_med, selectmode=tk.EXTENDED, height=6, bg="#111111", fg="white", yscrollcommand=scrollbar.set)
        self.listbox_med.pack(side="left", fill="x", expand=True)
        scrollbar.config(command=self.listbox_med.yview)
        
        # --- PANEL DE EXTRACCIÓN TENSORIAL ---
        frame_dsp = tk.LabelFrame(main_frame, text=" 2. Parámetros de Extracción DSP (Tensorial) ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_dsp.pack(fill="x", pady=5)
        
        grid_dsp = tk.Frame(frame_dsp, bg=self.bg_panel)
        grid_dsp.pack(fill="x")
        
        params_dsp = [
            ("Alpha Ruido:", "1.0", "ent_alpha"),
            ("SNR Min:", "0.5", "ent_snr"),
            ("Outliers (%):", "0.05", "ent_outliers"),
            ("Smooth (ms):", "150", "ent_smooth"),
            ("Target Len:", "100", "ent_target"),
            ("Notch Q:", "2.0", "ent_notch_q")
        ]
        
        for i, (label_text, default_val, attr_name) in enumerate(params_dsp):
            row = i // 3
            col = (i % 3) * 2
            tk.Label(grid_dsp, text=label_text, bg=self.bg_panel, fg=self.fg_text).grid(row=row, column=col, sticky="e", padx=5, pady=5)
            ent = tk.Entry(grid_dsp, width=8, bg="#111111", fg="white", insertbackground="white")
            ent.insert(0, default_val)
            ent.grid(row=row, column=col+1, sticky="w", padx=5, pady=5)
            setattr(self, attr_name, ent)
            
        # Checkbox exclusiones manuales
        self.var_manual_excl = tk.BooleanVar(value=True)
        self.chk_manual_excl = tk.Checkbutton(grid_dsp, text="Aplicar Exclusiones Manuales (metadata.json)", variable=self.var_manual_excl, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark)
        self.chk_manual_excl.grid(row=2, column=0, columnspan=6, sticky="w", padx=5, pady=5)
            
        # --- PANEL DE AUTOENCODER ---
        frame_nn = tk.LabelFrame(main_frame, text=" 3. Parámetros de Red Neuronal ", bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 11, "bold"), padx=10, pady=10)
        frame_nn.pack(fill="x", pady=5)
        
        grid_nn = tk.Frame(frame_nn, bg=self.bg_panel)
        grid_nn.pack(fill="x")
        
        params_nn = [
            ("Épocas:", "150", "ent_epochs"),
            ("Batch Size:", "32", "ent_batch"),
            ("Latent Dim:", "16", "ent_latent"),
            ("Alpha Loss:", "0.5", "ent_alpha_loss")
        ]
        
        for i, (label_text, default_val, attr_name) in enumerate(params_nn):
            tk.Label(grid_nn, text=label_text, bg=self.bg_panel, fg=self.fg_text).grid(row=0, column=i*2, sticky="e", padx=5, pady=5)
            ent = tk.Entry(grid_nn, width=8, bg="#111111", fg="white", insertbackground="white")
            ent.insert(0, default_val)
            ent.grid(row=0, column=i*2+1, sticky="w", padx=5, pady=5)
            setattr(self, attr_name, ent)
            
        # Checkbox forzar épocas
        self.var_force_epochs = tk.BooleanVar(value=False)
        self.chk_force_epochs = tk.Checkbutton(grid_nn, text="Forzar Épocas (Ignorar Checkpoint)", variable=self.var_force_epochs, bg=self.bg_panel, fg=self.fg_text, selectcolor=self.bg_dark)
        self.chk_force_epochs.grid(row=1, column=0, columnspan=6, sticky="w", padx=5, pady=5)
            
        # --- BOTONES DE EJECUCION ---
        frame_btns = tk.Frame(main_frame, bg=self.bg_dark)
        frame_btns.pack(fill="x", pady=15)
        
        self.btn_extraer = tk.Button(frame_btns, text="EXTRAER DATASET", font=("Arial", 11, "bold"), bg=self.cyan_dim, fg="black", command=self.ejecutar_extraccion)
        self.btn_extraer.pack(side="left", fill="x", expand=True, padx=2, ipady=8)
        
        self.btn_entrenar = tk.Button(frame_btns, text="ENTRENAR AUTOENCODER", font=("Arial", 11, "bold"), bg=self.cyan_dim, fg="black", command=self.ejecutar_entrenamiento)
        self.btn_entrenar.pack(side="left", fill="x", expand=True, padx=2, ipady=8)
        
        self.btn_plotear = tk.Button(frame_btns, text="PLOTEAR ESPACIO LATENTE", font=("Arial", 11, "bold"), bg=self.green_neon, fg="black", command=self.ejecutar_ploteo)
        self.btn_plotear.pack(side="left", fill="x", expand=True, padx=2, ipady=8)
        
        # --- LOG CONSOLE ---
        self.log_text = tk.Text(main_frame, height=8, bg="#111111", fg="#00FF00", font=("Consolas", 9), state="disabled")
        self.log_text.pack(fill="both", expand=True, pady=5)
        
        # --- HERRAMIENTAS ADICIONALES ---
        frame_tools = tk.Frame(main_frame, bg=self.bg_dark)
        frame_tools.pack(fill="x", pady=15)
        
        tk.Button(frame_tools, text="VISUALIZADOR DE FEATURES", bg="#333333", fg="white", font=("Arial", 10, "bold"), command=self.lanzar_visor).pack(fill="x", expand=True, padx=2, pady=2, ipady=5)
        tk.Button(frame_tools, text="DECODIFICAR SECUENCIA CONTINUA", bg="#333333", fg="#00FFFF", font=("Arial", 10, "bold"), command=self.lanzar_decodificador_continuo).pack(fill="x", expand=True, padx=2, pady=2, ipady=5)
        
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

    def get_params_dsp(self):
        return (
            float(self.ent_alpha.get()),
            float(self.ent_snr.get()),
            float(self.ent_outliers.get()),
            int(self.ent_smooth.get()),
            int(self.ent_target.get()),
            float(self.ent_notch_q.get()),
            self.var_manual_excl.get()
        )
        
    def get_params_nn(self):
        return (
            int(self.ent_epochs.get()),
            int(self.ent_batch.get()),
            int(self.ent_latent.get()),
            float(self.ent_alpha_loss.get()),
            self.var_force_epochs.get()
        )
        
    def toggle_buttons(self, state):
        self.btn_extraer.config(state=state)
        self.btn_entrenar.config(state=state)
        self.btn_plotear.config(state=state)

    def ejecutar_extraccion(self):
        seleccionadas = [self.listbox_med.get(i) for i in self.listbox_med.curselection()]
        if not seleccionadas:
            messagebox.showwarning("Advertencia", "Seleccione al menos una medición.")
            return
            
        try:
            params = self.get_params_dsp()
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos en DSP.")
            return
            
        self.toggle_buttons("disabled")
        self.log("=========================================")
        self.log("INICIANDO EXTRACCIÓN DE DATASET")
        self.log("=========================================")
        threading.Thread(target=self._extraccion_thread, args=(seleccionadas, *params)).start()

    def _extraccion_thread(self, seleccionadas, val_alpha, val_snr, val_out, val_smooth, val_target, val_notch_q, use_manual_excl):
        import traceback
        try:
            old_stdout = sys.stdout
            class LogWriter:
                def __init__(self, log_func): self.log_func = log_func
                def write(self, t): 
                    if t.strip(): self.log_func(t.strip())
                def flush(self): pass
            sys.stdout = LogWriter(self.log)
            
            gpt.ejecutar_procesamiento(
                seleccionadas,
                alpha_ruido=val_alpha,
                snr_threshold=val_snr,
                outlier_contamination=val_out,
                smooth_ms=val_smooth,
                target_length=val_target,
                notch_q=val_notch_q,
                use_manual_exclusions=use_manual_excl
            )
            
            sys.stdout = old_stdout
            self.log("\n>>> EXTRACCIÓN COMPLETADA <<<")
        except Exception as e:
            sys.stdout = old_stdout
            self.log(f"\n[ERROR] El proceso falló: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
        finally:
            self.toggle_buttons("normal")

    def ejecutar_entrenamiento(self):
        try:
            params = self.get_params_nn()
        except ValueError:
            messagebox.showerror("Error", "Parámetros numéricos inválidos en Red Neuronal.")
            return
            
        self.toggle_buttons("disabled")
        self.log("=========================================")
        self.log("INICIANDO ENTRENAMIENTO AUTOENCODER")
        self.log("=========================================")
        threading.Thread(target=self._entrenamiento_thread, args=params).start()

    def _entrenamiento_thread(self, v_epochs, v_batch, v_latent, v_alpha_loss, v_force):
        import traceback
        try:
            old_stdout = sys.stdout
            class LogWriter:
                def __init__(self, log_func): self.log_func = log_func
                def write(self, t): 
                    if t.strip(): self.log_func(t.strip())
                def flush(self): pass
            sys.stdout = LogWriter(self.log)
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
            csv_file = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
            
            if not os.path.exists(csv_file):
                raise Exception(f"No se encontró el dataset en {csv_file}\nDebes extraer el dataset primero.")
                
            ta.train_autoencoder(csv_file, epochs=v_epochs, batch_size=v_batch, latent_dim=v_latent, force_epochs=v_force, alpha=v_alpha_loss)
            
            sys.stdout = old_stdout
            self.log("\n>>> ENTRENAMIENTO COMPLETADO <<<")
        except Exception as e:
            sys.stdout = old_stdout
            self.log(f"\n[ERROR] El proceso falló: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
        finally:
            self.toggle_buttons("normal")

    def ejecutar_ploteo(self):
        try:
            _, _, v_latent, _, _ = self.get_params_nn()
        except ValueError:
            messagebox.showerror("Error", "Latent Dim inválido.")
            return
            
        self.toggle_buttons("disabled")
        self.log("=========================================")
        self.log("INICIANDO PLOTEO ESPACIO LATENTE")
        self.log("=========================================")
        threading.Thread(target=self._ploteo_thread, args=(v_latent,)).start()

    def _ploteo_thread(self, v_latent):
        import traceback
        try:
            old_stdout = sys.stdout
            class LogWriter:
                def __init__(self, log_func): self.log_func = log_func
                def write(self, t): 
                    if t.strip(): self.log_func(t.strip())
                def flush(self): pass
            sys.stdout = LogWriter(self.log)
            
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_repo_dir = os.path.abspath(os.path.join(script_dir, ".."))
            csv_file = os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv")
            model_path = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder", f"autoencoder_emg_{v_latent}d.pth")
            
            if not os.path.exists(csv_file) or not os.path.exists(model_path):
                raise Exception("Faltan archivos.\nDebes extraer el dataset y entrenar el modelo primero.")
                
            pls.plot_latent_space(csv_file, model_path, latent_dim=v_latent)
            
            sys.stdout = old_stdout
            self.log("\n>>> PLOTEO COMPLETADO <<<")
        except Exception as e:
            sys.stdout = old_stdout
            self.log(f"\n[ERROR] El proceso falló: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
        finally:
            self.toggle_buttons("normal")
            
    def lanzar_decodificador_continuo(self):
        from tkinter import filedialog
        carpeta = filedialog.askdirectory(initialdir=self.base_dir, title="Seleccione la carpeta de Secuencia Continua")
        if not carpeta:
            return
            
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decodificador_continuo.py")
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"No se encontró {script_path}")
            return
            
        self.log(f"Lanzando Decodificador Continuo para: {os.path.basename(carpeta)}...")
        self.toggle_buttons("disabled")
        
        # Ejecutar en un hilo para mostrar prints en el log de la GUI
        threading.Thread(target=self._decodificador_thread, args=(carpeta,)).start()

    def _decodificador_thread(self, carpeta):
        import traceback
        try:
            old_stdout = sys.stdout
            class LogWriter:
                def __init__(self, log_func): self.log_func = log_func
                def write(self, t): 
                    if t.strip(): self.log_func(t.strip())
                def flush(self): pass
            sys.stdout = LogWriter(self.log)
            
            # Importar e invocar la funcion directamente en lugar de subproceso
            import deep_learning.decodificador_continuo as dc
            modelo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados", "resultados_autoencoder", "autoencoder_emg_16d.pth")
            
            # Traer parametros de ruido/notch actuales de la GUI si se quiere
            val_alpha, _, _, val_smooth, val_target, val_notch_q, val_manual = self.get_params_dsp()
            
            dc.decodificar_secuencia(carpeta, modelo_path, alpha_ruido=val_alpha, smooth_ms=val_smooth, notch_q=val_notch_q, use_manual_exclusions=val_manual, target_length=val_target)
            
            sys.stdout = old_stdout
            self.log("\n>>> DECODIFICACIÓN COMPLETADA <<<")
        except Exception as e:
            sys.stdout = old_stdout
            self.log(f"\n[ERROR] El decodificador falló: {e}")
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
        finally:
            self.toggle_buttons("normal")

    def lanzar_visor(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        visor_path = os.path.join(script_dir, "dataset_tools", "visor_features.py")
        if os.path.exists(visor_path):
            import subprocess
            subprocess.Popen([sys.executable, visor_path])
        else:
            messagebox.showerror("Error", f"No se encontró {visor_path}")
            
    def lanzar_plot_musculos(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(script_dir, "dataset_tools", "plot_3_musculos_standalone.py")
        import subprocess
        subprocess.Popen([sys.executable, plot_path])

if __name__ == "__main__":
    root = tk.Tk()
    rutas = sys.argv[1:] if len(sys.argv) > 1 else None
    app = PipelineAutoencoderGUI(root, rutas_preseleccionadas=rutas)
    root.mainloop()
