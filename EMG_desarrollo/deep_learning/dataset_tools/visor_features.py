import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os

import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root


class FeatureViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visor de Features (Data PCA/UMAP)")
        self.root.geometry("1100x700")
        
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.fg_text = "#C5C6C7"
        
        self.root.configure(bg=self.bg_dark)
        
        self.data = []  # Lista de diccionarios con la data
        self.tomas = [] # Lista de nombres de tomas
        self.visible_indices = [] # Lista de índices reales correspondientes al listbox
        
        self.setup_ui()
        self.auto_load_default()
        
    def setup_ui(self):
        # Panel izquierdo (Controles y Lista)
        left_panel = tk.Frame(self.root, width=300, bg=self.bg_dark, padx=10, pady=10)
        left_panel.pack(side="left", fill="y", expand=False)
        
        btn_load = tk.Button(left_panel, text="Cargar CSV de Features", command=self.load_csv, bg=self.bg_panel, fg=self.cyan_neon, font=("Arial", 10, "bold"))
        btn_load.pack(fill="x", pady=(0, 5))
        
        # Selector de Datasets
        self.dataset_var = tk.StringVar()
        self.dataset_selector = ttk.Combobox(left_panel, textvariable=self.dataset_var, state="readonly")
        self.dataset_selector.pack(fill="x", pady=(0, 10))
        self.dataset_selector.bind("<<ComboboxSelected>>", self.on_dataset_selected)
        
        self.lbl_info = tk.Label(left_panel, text="No hay archivo cargado", bg=self.bg_dark, fg=self.fg_text, wraplength=280)
        self.lbl_info.pack(fill="x", pady=(0, 10))
        
        lbl_list = tk.Label(left_panel, text="Tomas (Instancias extraídas):", bg=self.bg_dark, fg=self.cyan_neon, anchor="w")
        lbl_list.pack(fill="x")
        
        # Filtro
        filter_frame = tk.Frame(left_panel, bg=self.bg_dark)
        filter_frame.pack(fill="x", pady=5)
        tk.Label(filter_frame, text="Filtrar por Vocal:", bg=self.bg_dark, fg="white").pack(side="left")
        self.vocal_filter_var = tk.StringVar(value="Todas")
        self.vocal_filter = ttk.Combobox(filter_frame, textvariable=self.vocal_filter_var, values=["Todas", "A", "E", "I", "O", "U"], width=8, state="readonly")
        self.vocal_filter.pack(side="right")
        self.vocal_filter.bind("<<ComboboxSelected>>", self.apply_filter)
        
        list_frame = tk.Frame(left_panel, bg=self.bg_dark)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side="right", fill="y")
        
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, bg=self.bg_panel, fg="white", 
                                  selectbackground=self.cyan_neon, selectforeground="black", highlightthickness=0)
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        
        # Panel derecho (Gráfico)
        self.right_panel = tk.Frame(self.root, bg="white")
        self.right_panel.pack(side="right", fill="both", expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor(self.bg_dark)
        self.ax.set_facecolor(self.bg_panel)
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color(self.cyan_neon)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.fg_text)
            
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.ax.set_title("Selecciona una toma de la lista para visualizar las 300 variables")

    def auto_load_default(self):
        # Busca cualquier archivo dataset_features*.csv en la carpeta
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base_de_datos_letras"))
        if not os.path.exists(self.base_dir):
            return
            
        import glob
        pattern = os.path.join(self.base_dir, "dataset_features*.csv")
        files = glob.glob(pattern)
        
        if not files:
            return
            
        # Ordenar por fecha de modificación (el más reciente primero)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Llenar el combobox con los nombres de archivo
        filenames = [os.path.basename(f) for f in files]
        self.dataset_selector['values'] = filenames
        self.dataset_selector.current(0) # Seleccionar el más reciente
        
        # Cargar el archivo seleccionado
        self.process_csv(files[0])

    def on_dataset_selected(self, event):
        filename = self.dataset_var.get()
        if filename:
            path = os.path.join(self.base_dir, filename)
            if os.path.exists(path):
                self.process_csv(path)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Seleccionar CSV de Features",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if file_path:
            self.process_csv(file_path)

    def process_csv(self, file_path):
        self.data = []
        self.tomas = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.data.append(row)
                    self.tomas.append(f"{row['Vocal']} - {row['Toma']}")
            
            self.lbl_info.config(text=f"Archivo: {os.path.basename(file_path)}\nTotal instancias: {len(self.data)}")
            self.apply_filter()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo CSV:\n{e}")

    def apply_filter(self, event=None):
        self.listbox.delete(0, tk.END)
        filtro = self.vocal_filter_var.get()
        self.visible_indices = []
        
        for i, row in enumerate(self.data):
            if filtro == "Todas" or row["Vocal"] == filtro:
                display_text = f"{row['Vocal']} - {os.path.basename(row['Toma'])}"
                self.listbox.insert(tk.END, display_text)
                self.visible_indices.append(i)

    def on_select(self, event):
        selection = self.listbox.curselection()
        if not selection:
            return
            
        index_in_listbox = selection[0]
        real_idx = self.visible_indices[index_in_listbox]
        self.plot_data(real_idx)

    def plot_data(self, idx):
        row = self.data[idx]
        
        ch0_vals = []
        ch1_vals = []
        ch2_vals = []
        
        # Auto-detectar la resolución de los datos (ej: 100, 250, 500 puntos)
        t_max = 0
        while f"Ch0_T{t_max}" in row:
            t_max += 1
            
        if t_max == 0:
            messagebox.showwarning("Aviso", "No se encontraron columnas de tiempo (Ch0_TX).")
            return
            
        try:
            for t in range(t_max):
                v0 = row.get(f"Ch0_T{t}", "0.0")
                v1 = row.get(f"Ch1_T{t}", "0.0")
                v2 = row.get(f"Ch2_T{t}", "0.0")
                
                if v0 == "": v0 = "0.0"
                if v1 == "": v1 = "0.0"
                if v2 == "": v2 = "0.0"
                
                ch0_vals.append(float(v0))
                ch1_vals.append(float(v1))
                ch2_vals.append(float(v2))
        except ValueError:
            messagebox.showwarning("Aviso", "Error al parsear los datos numéricos.")
            return

        self.fig.clear()
        is_stft = "stft" in self.dataset_var.get().lower()
        
        if is_stft:
            ax0 = self.fig.add_subplot(311)
            ax1 = self.fig.add_subplot(312)
            ax2 = self.fig.add_subplot(313)
            
            for ax in [ax0, ax1, ax2]:
                ax.set_facecolor(self.bg_panel)
                ax.tick_params(colors="white")
                for spine in ax.spines.values():
                    spine.set_edgecolor(self.fg_text)
                    
            # Intentar deducir la resolución de frecuencia (f_bins) basándose en la longitud del array aplanado
            f_bins_options = [251, 51] # 251 bins = ventana 250ms (mejor para EMG), 51 bins = ventana 50ms
            f_bins = None
            t_bins = None
            
            for possible_f in f_bins_options:
                if t_max % possible_f == 0:
                    f_bins = possible_f
                    t_bins = t_max // f_bins
                    break
                    
            if f_bins is not None:
                img0 = np.array(ch0_vals).reshape(f_bins, t_bins)
                img1 = np.array(ch1_vals).reshape(f_bins, t_bins)
                img2 = np.array(ch2_vals).reshape(f_bins, t_bins)
                
                # Para mayor visibilidad, aplicamos un poco de contraste
                vmax = np.max([img0.max(), img1.max(), img2.max()]) * 0.8
                
                ax0.imshow(img0, aspect='auto', origin='lower', cmap='magma', vmax=vmax)
                ax1.imshow(img1, aspect='auto', origin='lower', cmap='magma', vmax=vmax)
                ax2.imshow(img2, aspect='auto', origin='lower', cmap='magma', vmax=vmax)
                
                ax0.set_title(f"Canal 0 (Masetero) - STFT ({t_bins}x{f_bins})", color=self.cyan_neon)
                ax1.set_title("Canal 1 (Orbicular) - STFT", color=self.cyan_neon)
                ax2.set_title("Canal 2 (Cigomático) - STFT", color=self.cyan_neon)
                
                ax2.set_xlabel("Ventanas temporales", color="white")
                ax1.set_ylabel("Frecuencia (bins)", color="white")
            else:
                ax0.set_title(f"Error STFT: {t_max} no es divisible por {f_bins} bins.", color="red")
                
            self.fig.tight_layout()
            self.canvas.draw()
            return

        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.bg_panel)
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color(self.cyan_neon)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.fg_text)
            
        time_axis = np.linspace(-50, 50, t_max) # Asumiendo ventana centrada en %
        
        self.ax.plot(time_axis, ch0_vals, label='Canal 0 (Masetero)', color='#45B7D1', linewidth=2)
        self.ax.plot(time_axis, ch1_vals, label='Canal 1 (Orbicular)', color='#FF6B6B', linewidth=2)
        self.ax.plot(time_axis, ch2_vals, label='Canal 2 (Cigomático)', color='#C5C6C7', linewidth=2)
        
        self.ax.axvline(x=0, color='#F3E94C', linestyle='--', linewidth=2, alpha=0.8, label='Pico Micrófono (Ancla)')
        
        self.ax.set_title(f"Características Dinámicas - Toma: {row['Toma']}", color=self.cyan_neon)
        self.ax.set_xlabel('Tiempo relativo al pico del micrófono (%)')
        self.ax.set_ylabel('Amplitud Normalizada')
        self.ax.legend(facecolor=self.bg_dark, edgecolor=self.cyan_neon, labelcolor='white', loc='upper right')
        self.ax.grid(True, color=self.fg_text, alpha=0.2)
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureViewerApp(root)
    root.mainloop()
