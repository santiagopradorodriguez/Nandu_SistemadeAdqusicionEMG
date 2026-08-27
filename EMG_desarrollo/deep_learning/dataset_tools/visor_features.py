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
        
        # Selector de Fuente de Datos
        self.fuente_var = tk.StringVar(value="PCA/UMAP")
        self.fuente_selector = ttk.Combobox(left_panel, textvariable=self.fuente_var, values=["PCA/UMAP", "Tensorial (Autoencoder)"], state="readonly")
        self.fuente_selector.pack(fill="x", pady=(0, 5))
        self.fuente_selector.bind("<<ComboboxSelected>>", self.on_fuente_selected)
        
        # Selector de Sets / Corridas detectadas
        self.available_sets = {}
        self.set_var = tk.StringVar()
        self.set_selector = ttk.Combobox(left_panel, textvariable=self.set_var, state="readonly")
        self.set_selector.pack(fill="x", pady=(0, 10))
        self.set_selector.bind("<<ComboboxSelected>>", self.on_set_selected)
        
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
        self.ax.set_title("Selecciona una toma de la lista para visualizar las variables")

    def auto_load_default(self):
        if len(sys.argv) > 1 and os.path.exists(sys.argv[1]) and sys.argv[1].endswith(".csv"):
            self.process_csv(sys.argv[1])
        else:
            self.on_fuente_selected(None)

    def on_fuente_selected(self, event):
        fuente = self.fuente_var.get()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_repo_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
        
        search_dirs = []
        if fuente == "PCA/UMAP":
            search_dirs = [
                os.path.join(base_repo_dir, "deep_learning", "pca_umap_clustering", "resultados_pca_umap"),
                os.path.join(base_repo_dir, "resultados", "resultados_pca_umap"),
                os.path.join(base_repo_dir, "resultados_pca_umap"),
            ]
        else:
            search_dirs = [
                os.path.join(base_repo_dir, "resultados", "resultados_pca_tensorial"),
                os.path.join(base_repo_dir, "deep_learning", "resultados_pca_tensorial"),
                os.path.join(base_repo_dir, "resultados_pca_tensorial"),
            ]
            
        self.available_sets = {}
        for s_dir in search_dirs:
            if not os.path.exists(s_dir):
                continue
            # Archivo en la raiz del directorio de resultados
            root_csv = os.path.join(s_dir, "caracteristicas_exportadas.csv")
            if os.path.exists(root_csv):
                self.available_sets["(Último / Raíz)"] = root_csv
            # Buscar en subcarpetas (sets de experimentos nombrados)
            for item in os.listdir(s_dir):
                subpath = os.path.join(s_dir, item)
                if os.path.isdir(subpath):
                    cand_csv = os.path.join(subpath, "caracteristicas_exportadas.csv")
                    if os.path.exists(cand_csv):
                        mtime = os.path.getmtime(cand_csv)
                        self.available_sets[item] = cand_csv

        if self.available_sets:
            # Ordenar por fecha de modificación más reciente
            sorted_sets = sorted(self.available_sets.keys(), key=lambda k: os.path.getmtime(self.available_sets[k]), reverse=True)
            self.set_selector['values'] = sorted_sets
            first_set = sorted_sets[0]
            self.set_var.set(first_set)
            self.process_csv(self.available_sets[first_set])
        else:
            self.set_selector['values'] = []
            self.set_var.set("")
            self.data = []
            self.tomas = []
            self.listbox.delete(0, tk.END)
            self.lbl_info.config(text="No se encontraron archivos 'caracteristicas_exportadas.csv'.\nEjecuta PCA o carga un CSV manualmente.")
            self.ax.clear()
            self.canvas.draw()

    def on_set_selected(self, event):
        selected = self.set_var.get()
        if selected in self.available_sets:
            self.process_csv(self.available_sets[selected])

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
        
        # Auto-detectar canales presentes
        channels_found = []
        for ch_idx in range(8):
            if f"Ch{ch_idx}_T0" in row:
                channels_found.append(ch_idx)
                
        if not channels_found:
            messagebox.showwarning("Aviso", "No se encontraron columnas de canales (ChX_T0).")
            return
            
        # Auto-detectar la resolución de los datos (t_max)
        t_max = 0
        first_ch = channels_found[0]
        while f"Ch{first_ch}_T{t_max}" in row:
            t_max += 1
            
        if t_max == 0:
            messagebox.showwarning("Aviso", "No se encontraron puntos temporales.")
            return

        channel_data = {}
        try:
            for ch in channels_found:
                vals = []
                for t in range(t_max):
                    v = row.get(f"Ch{ch}_T{t}", "0.0")
                    if v == "" or v is None: v = "0.0"
                    vals.append(float(v))
                channel_data[ch] = vals
        except ValueError:
            messagebox.showwarning("Aviso", "Error al parsear los datos numéricos.")
            return

        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.bg_panel)
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color(self.cyan_neon)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.fg_text)
            
        time_axis = np.linspace(-50, 50, t_max) # Ventana normalizada (%)
        
        channel_meta = {
            0: {"name": "Canal 0 (DAO)", "color": "#7000FF"},
            1: {"name": "Canal 1 (Milohioideo)", "color": "#00FF88"},
            2: {"name": "Canal 2 (Orbicular)", "color": "#FFE600"},
            3: {"name": "Canal 3 (Micrófono)", "color": "#FF003C"},
        }
        
        default_colors = ["#7000FF", "#00FF88", "#FFE600", "#FF003C", "#00FFFF", "#FF00FF"]
        
        # Encontrar y graficar los picos de las derivadas
        def plot_peak(ch_np, color):
            if len(ch_np) == 0 or np.max(ch_np) == 0: return
            grad = np.gradient(ch_np)
            win = max(1, len(ch_np) // 10)
            if win > 1: grad = np.convolve(grad, np.ones(win)/win, mode='same')
            idx_pico = np.argmax(grad)
            self.ax.plot(time_axis[idx_pico], ch_np[idx_pico], 'o', color=color, markersize=7)
            self.ax.axvline(time_axis[idx_pico], color=color, linestyle=':', alpha=0.4)

        for ch in channels_found:
            meta = channel_meta.get(ch, {"name": f"Canal {ch}", "color": default_colors[ch % len(default_colors)]})
            c_name = meta["name"]
            c_color = meta["color"]
            c_vals = channel_data[ch]
            self.ax.plot(time_axis, c_vals, label=c_name, color=c_color, linewidth=2)
            plot_peak(np.array(c_vals), c_color)
        
        self.ax.axvline(x=0, color='#66FCF1', linestyle='--', linewidth=2, alpha=0.8, label='Centro Ventana (Onset)')
        
        self.ax.set_title(f"Características Dinámicas - Toma: {row.get('Toma', '')} [Vocal: {row.get('Vocal', '')}]", color=self.cyan_neon)
        self.ax.set_xlabel('Tiempo relativo al Onset (%)')
        self.ax.set_ylabel('Amplitud Normalizada')
        self.ax.legend(facecolor=self.bg_dark, edgecolor=self.cyan_neon, labelcolor='white', loc='upper right')
        self.ax.grid(True, color=self.fg_text, alpha=0.2)
        
        self.fig.tight_layout()
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = FeatureViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
