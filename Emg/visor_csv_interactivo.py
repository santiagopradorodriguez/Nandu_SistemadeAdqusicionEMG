# -*- coding: utf-8 -*-
"""
Visor CSV Interactivo v4.3 (Exportación en Alta Calidad)
# Esta es la última versión funcional conocida.

Una herramienta para cargar y explorar archivos CSV generados por el sistema de adquisición.
Permite:
- Seleccionar un archivo CSV.
- Visualizar los datos en un gráfico de Matplotlib.
- Seleccionar qué canales mostrar.
- Navegar fluidamente por la grabación usando un deslizador de tiempo.
- Controlar el zoom de los ejes X e Y con deslizadores.
- Exportar la vista actual del gráfico a un archivo PNG.
"""
import tkinter as tk
from tkinter import filedialog, messagebox, font, ttk
import pandas as pd
import numpy as np
import os

# --- NUEVO: Import para filtros ---
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import threading

MAX_POINTS_TO_PLOT = 75_000 # Límite de puntos a graficar antes de aplicar downsampling

def downsample_lttb_fast(x, y, threshold):
    """
    Largest-Triangle-Three-Buckets (LTTB) downsampling algorithm.
    This is a fast, correct implementation that preserves visual features.
    """
    if len(x) <= threshold:
        return x, y

    # 1. Bucket size
    every = (len(x) - 2) / (threshold - 2)

    # 2. Prep work
    sampled_x = np.zeros(threshold, dtype=np.float64)
    sampled_y = np.zeros(threshold, dtype=np.float64)

    # 3. Always add the first point
    sampled_x[0] = x[0]
    sampled_y[0] = y[0]

    a = 0  # Index of the last selected point

    for i in range(threshold - 2):
        # 4. Calculate the range for the next bucket
        start = int(np.floor((i + 1) * every)) + 1
        end = int(np.floor((i + 2) * every)) + 1
        end = min(end, len(x))

        if start >= end:
            continue

        # 5. Calculate the area of the triangle for each point in the bucket
        # and select the one that forms the largest triangle.
        areas = 0.5 * np.abs(
            (x[a] - x[start:end]) * (y[start:end] - y[a]) -
            (x[a] - x[start:end]) * (y[a] - y[start:end])
        )
        
        max_area_idx = np.argmax(areas)
        a = start + max_area_idx

        sampled_x[i + 1] = x[a]
        sampled_y[i + 1] = y[a]

    # 6. Always add the last point
    sampled_x[-1] = x[-1]
    sampled_y[-1] = y[-1]

    return sampled_x, sampled_y

class CSVViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visor CSV Interactivo v4.3")
        self.root.geometry("1200x850")
        self.root.configure(bg="#f0f0f0")

        self.BASE_DIR = "base_de_datos_electrodos"
        self.mediciones_disponibles = []
        self.var_medicion_seleccionada = tk.StringVar()
        self.var_medicion_seleccionada.trace_add("write", self.on_measurement_selected)

        # --- Control para evitar bucles de actualización de sliders ---
        self._is_programmatically_updating_sliders = False

        self.font_label = font.Font(family="Helvetica", size=10)
        self.font_bold = font.Font(family="Helvetica", size=10, weight="bold")

        # --- Atributos para carga en hilo ---
        self.loading_thread = None
        self.loaded_df = None

        self.min_y_global = -1.0
        self.max_y_global = 1.0
        self.last_loaded_filepath = None

        # --- Atributos de datos ---
        self.df = None
        self.time_col = None
        self.channel_cols = []
        self.channel_vars = {}

        # --- Estructura de la GUI ---
        top_frame = tk.Frame(root, padx=10, pady=5, bg="#f0f0f0")
        top_frame.pack(fill="x", side="top")

        # --- REESTRUCTURADO: Layout principal con panel de control a la derecha ---
        main_content_frame = tk.Frame(root)
        main_content_frame.pack(fill="both", expand=True)

        plot_frame = tk.Frame(main_content_frame, bg="#f0f0f0")
        plot_frame.pack(side="left", fill="both", expand=True)

        # --- Controles Superiores ---
        tk.Label(top_frame, text="Seleccionar Medición:", font=self.font_bold, bg="#f0f0f0").pack(side="left", padx=(5, 2))
        self.measurement_menu = tk.OptionMenu(top_frame, self.var_medicion_seleccionada, "")
        self.measurement_menu.config(width=30)
        self.measurement_menu.pack(side="left", padx=5)

        self.btn_refresh = tk.Button(top_frame, text="Refrescar Lista", command=self.cargar_mediciones, font=self.font_label)
        self.btn_refresh.pack(side="left", padx=5)

        # Se actualiza al cargar
        self.lbl_file = tk.Label(top_frame, text="Seleccione una medición.", fg="#333", wraplength=400, font=self.font_label, bg="#f0f0f0")
        self.lbl_file.pack(side="left", padx=10)

        # --- NUEVO: Mover el botón de autoscale y opciones al frame superior ---
        self.btn_autoscale = tk.Button(top_frame, text="Ajuste Automático", command=self.autoscale_view, state="disabled", font=self.font_label)
        self.btn_autoscale.pack(side="right", padx=(10, 5))

        options_frame = tk.LabelFrame(top_frame, text="Opciones", padx=5, pady=2, font=self.font_bold, bg="#f0f0f0")
        options_frame.pack(side="right", padx=5)

        self.btn_export = tk.Button(options_frame, text="Exportar PNG", command=self.export_png, state="disabled", font=self.font_label)
        self.btn_export.pack(side="left", padx=5)

        # --- Gráfico Matplotlib ---
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax = self.fig.add_subplot(111)
        # --- CORRECCIÓN: Restaurar la conexión del callback para el eje Y ---
        self.ax.callbacks.connect('ylim_changed', self.on_ax_ylim_changed)
        self.last_ylim = None
        self.ax.grid(True)
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Amplitud (V)")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # --- NUEVO: Panel de control lateral derecho ---
        right_panel = tk.Frame(main_content_frame, bg="#f0f0f0", width=200)
        right_panel.pack(side="right", fill="y", padx=10, pady=10)

        # --- NUEVO: Panel de selección de canales (movido al panel derecho) ---
        self.channels_frame = tk.LabelFrame(right_panel, text="Canales", font=self.font_bold, bg="#f0f0f0", padx=10, pady=5)
        self.channels_frame.pack(fill="x", pady=(0, 10))
        # El contenido (checkboxes) se añade dinámicamente en `setup_channel_checkboxes`

        # --- Controles de Navegación (movidos al panel derecho) ---
        nav_frame = tk.LabelFrame(right_panel, text="Navegación", font=self.font_bold, bg="#f0f0f0", padx=10, pady=10)
        nav_frame.pack(fill="x", pady=(0, 10))

        # Deslizador de tiempo (posición)
        self.time_slider = tk.Scale(nav_frame, from_=0, to=100, orient="horizontal", relief="flat",
                                    label="Posición (s)", length=400, command=self.on_pan_control_changed, font=self.font_label, bg="#f0f0f0", highlightthickness=0, resolution=0.01)
        self.time_slider.pack(fill="x", expand=True, pady=(0, 10))
        self.time_slider.set(0)

        # Deslizador de posición Y
        self.y_pan_slider = tk.Scale(nav_frame, from_=100, to=0, orient="horizontal", label="Posición Y", command=self.on_y_pan_changed)
        self.y_pan_slider.pack(fill="x", expand=True)
        self._is_programmatically_updating_y_scroll = False

        # --- Controles de Zoom (movidos al panel derecho) ---
        zoom_frame = tk.LabelFrame(right_panel, text="Zoom", font=self.font_bold, bg="#f0f0f0", padx=10, pady=10)
        zoom_frame.pack(fill="x", pady=10)

        # NUEVO: Deslizador de zoom en X (Duración)
        self.x_zoom_slider = tk.Scale(zoom_frame, from_=30, to=0.1, resolution=0.1, orient="horizontal", label="Duración (s)", command=self.on_zoom_control_changed)
        self.x_zoom_slider.set(5.0) # Un valor inicial más amplio
        self.x_zoom_slider.pack(fill="x", expand=True, pady=(0, 10))

        # --- Controles Eje Y ---
        self.y_zoom_slider = tk.Scale(zoom_frame, from_=100, to=1, orient="horizontal", label="Amplitud (%)", command=self.update_y_zoom)
        self.y_zoom_slider.set(100) # 100% = sin zoom
        self.y_zoom_slider.pack(fill="x", expand=True)

        # --- NUEVO: Controles de Filtro ---
        filter_frame = tk.LabelFrame(right_panel, text="Filtros", font=self.font_bold, bg="#f0f0f0", padx=10, pady=10)
        filter_frame.pack(fill="x", pady=10)

        self.var_notch_filter = tk.BooleanVar(value=False)
        self.chk_notch = tk.Checkbutton(filter_frame, text="Notch 50 Hz", variable=self.var_notch_filter, command=self.on_channel_toggle, bg="#f0f0f0", font=self.font_label)
        self.chk_notch.pack(anchor="w")

        self.var_bandpass_filter = tk.BooleanVar(value=False)
        self.chk_bandpass = tk.Checkbutton(filter_frame, text="Pasa-Banda", variable=self.var_bandpass_filter, command=self.on_channel_toggle, bg="#f0f0f0", font=self.font_label)
        self.chk_bandpass.pack(anchor="w")

        bandpass_options_frame = tk.Frame(filter_frame, bg="#f0f0f0")
        bandpass_options_frame.pack(fill="x", padx=(20, 0))

        tk.Label(bandpass_options_frame, text="Baja (Hz):", font=self.font_label, bg="#f0f0f0").pack(side="left")
        self.spin_low_cut = tk.Spinbox(bandpass_options_frame, from_=1, to=20000, width=6, command=self.on_channel_toggle)
        self.spin_low_cut.delete(0, "end")
        self.spin_low_cut.insert(0, "20")
        self.spin_low_cut.pack(side="left", padx=5)

        tk.Label(bandpass_options_frame, text="Alta (Hz):", font=self.font_label, bg="#f0f0f0").pack(side="left")
        self.spin_high_cut = tk.Spinbox(bandpass_options_frame, from_=2, to=22000, width=6, command=self.on_channel_toggle)
        self.spin_high_cut.delete(0, "end")
        self.spin_high_cut.insert(0, "500")
        self.spin_high_cut.pack(side="left", padx=5)

        if not SCIPY_AVAILABLE:
            for widget in [self.chk_notch, self.chk_bandpass, self.spin_low_cut, self.spin_high_cut]:
                widget.config(state="disabled")
            tk.Label(filter_frame, text="Scipy no instalado.", fg="red", bg="#f0f0f0").pack(anchor="w")

        # Carga inicial
        self.cargar_mediciones()

    def cargar_mediciones(self):
        """Carga la lista de mediciones (carpetas) desde el directorio base."""
        self.mediciones_disponibles = []
        menu = self.measurement_menu["menu"]
        menu.delete(0, "end")

        try:
            if os.path.isdir(self.BASE_DIR):
                # Carpetas de medición que contienen directamente un archivo .csv
                carpetas_validas = [
                    item for item in sorted(os.listdir(self.BASE_DIR))
                    if os.path.isdir(os.path.join(self.BASE_DIR, item)) and any(f.lower().endswith('.csv') for f in os.listdir(os.path.join(self.BASE_DIR, item)))
                ]
                self.mediciones_disponibles = carpetas_validas
                if carpetas_validas:
                    for medicion in carpetas_validas:
                        menu.add_command(label=medicion, command=tk._setit(self.var_medicion_seleccionada, medicion))
                    self.var_medicion_seleccionada.set(carpetas_validas[0]) # Esto disparará el trace
                else:
                    self.var_medicion_seleccionada.set("") # Limpia la variable si no hay mediciones
                    self.lbl_file.config(text=f"No se encontraron mediciones con CSV en '{self.BASE_DIR}'")
        except FileNotFoundError:
            messagebox.showerror("Error", f"No se encontró el directorio base: '{self.BASE_DIR}'")

    def on_measurement_selected(self, *args):
        """Se llama al seleccionar una medición del menú. Busca y carga el CSV."""
        selection = self.var_medicion_seleccionada.get()
        if not selection: return

        measurement_path = os.path.join(self.BASE_DIR, selection)
        csv_file = None
        for f in sorted(os.listdir(measurement_path)):
            if f.lower().endswith('.csv'):
                csv_file = os.path.join(measurement_path, f)
                break
        if csv_file:
            # Evitar recargas innecesarias si el archivo ya está cargado
            if csv_file != self.last_loaded_filepath:
                self.start_loading_csv(csv_file)

    def start_loading_csv(self, filepath):
        """Inicia la carga del CSV en un hilo para no congelar la GUI."""
        self.progress_win = tk.Toplevel(self.root)
        self.progress_win.title("Cargando...")
        self.progress_win.geometry("300x80")
        self.progress_win.transient(self.root)
        self.progress_win.grab_set()
        
        tk.Label(self.progress_win, text="Cargando datos, por favor espere...", pady=10, font=self.font_label).pack()
        progress_bar = ttk.Progressbar(self.progress_win, orient="horizontal", length=250, mode="indeterminate")
        progress_bar.pack(pady=5)
        progress_bar.start(10)

        self.loading_thread = threading.Thread(target=self._load_csv_threaded, args=(filepath,), daemon=True)
        self.loading_thread.start()
        self.root.after(100, self.check_loading_thread)

    def _load_csv_threaded(self, filepath):
        """Función que se ejecuta en el hilo para cargar el CSV."""
        try:
            self.loaded_df = pd.read_csv(filepath)
        except Exception as e:
            self.loaded_df = e # Pasar la excepción para mostrarla en el hilo principal

    def check_loading_thread(self):
        """Verifica si el hilo de carga ha terminado."""
        if self.loading_thread.is_alive():
            self.root.after(100, self.check_loading_thread)
        else:
            self.progress_win.destroy()
            if isinstance(self.loaded_df, Exception):
                messagebox.showerror("Error al cargar", f"No se pudo leer el archivo CSV.\nError: {self.loaded_df}")
                self.df = None
                self.lbl_file.config(text="Error al cargar archivo.")
                self.btn_export.config(state="disabled")
                self.btn_autoscale.config(state="disabled")
            else:
                # El nombre del hilo no se puede pasar directamente, usamos una variable
                filepath = self.loading_thread.name
                self.df = self.loaded_df
                self.finish_loading_csv(filepath)

    def finish_loading_csv(self, filepath):
        """Se ejecuta en el hilo principal después de que el CSV se ha cargado."""
        if not filepath:
            return

        try:
            # Calcular Fs y actualizar label
            fs = 1 / (self.df[self.df.columns[0]].iloc[1] - self.df[self.df.columns[0]].iloc[0])
            self.lbl_file.config(text=f"Archivo: {os.path.basename(filepath)} (Fs: {fs:.2f} Hz)")
            self.btn_export.config(state="normal")
            self.btn_autoscale.config(state="normal")
            self.last_loaded_filepath = filepath

            # Identificar columnas
            self.time_col = self.df.columns[0]
            self.channel_cols = self.df.columns[1:]

            # Configurar controles
            max_time = self.df[self.time_col].iloc[-1]
            self.time_slider.config(to=max_time)
            # Ajustar el máximo del slider de duración si la grabación es corta
            if max_time < self.x_zoom_slider.cget("to"):
                self.x_zoom_slider.config(to=max_time)

            self.time_slider.set(0)

            # Calcular límites globales para el eje Y
            y_min = self.df[self.channel_cols].min().min()
            y_max = self.df[self.channel_cols].max().max()
            y_range = y_max - y_min
            self.min_y_global = y_min - 0.1 * y_range
            self.max_y_global = y_max + 0.1 * y_range

            self.setup_channel_checkboxes()
            self.plot_full_data() # Dibuja los datos una vez
            self.update_y_scrollbar() # Actualiza el scrollbar con los nuevos límites
            self.update_view_from_controls() # Ajusta la vista inicial

        except Exception as e:
            messagebox.showerror("Error de Procesamiento", f"Error al procesar el archivo CSV después de cargarlo.\nError: {e}")
            self.df = None
            self.lbl_file.config(text="Error al cargar archivo.")
            self.btn_export.config(state="disabled")
            self.btn_autoscale.config(state="disabled")

    def setup_channel_checkboxes(self):
        """Crea los checkboxes para cada canal detectado."""
        # Limpiar checkboxes anteriores
        # --- CORRECCIÓN: Destruir todos los widgets hijos del frame de canales ---
        for widget in self.channels_frame.winfo_children(): widget.destroy()

        # --- NUEVO: Crear un frame interno para alinear los checkboxes ---
        inner_frame = tk.Frame(self.channels_frame, bg="#f0f0f0")
        inner_frame.pack(fill="x")
        self.channel_vars = {}
        for i, col_name in enumerate(self.channel_cols):
            var = tk.BooleanVar(value=True)
            # Al cambiar un checkbox, se redibuja todo y luego se ajusta la vista
            display_name = col_name.strip() # Eliminar espacios extra del header del CSV
            chk = tk.Checkbutton(inner_frame, text=display_name, variable=var, command=self.on_channel_toggle, bg="#f0f0f0", font=self.font_label)
            # --- CORRECCIÓN: Usar grid para asegurar que todos los checkboxes sean visibles ---
            row = i // 2
            col = i % 2
            chk.grid(row=row, column=col, sticky="w", padx=2)
            self.channel_vars[col_name] = var

    def plot_full_data(self):
        """Limpia y dibuja los datos completos de los canales seleccionados. Es lento."""
        if self.df is None:
            return
        
        # Guardar límites actuales para restaurarlos después de redibujar
        current_xlim = self.ax.get_xlim()
        current_ylim = self.last_ylim if self.last_ylim is not None else self.ax.get_ylim()
        is_zoomed_x = self.ax.get_xlim() != (0.0, 1.0) # Heurística para saber si el usuario hizo zoom manual
        
        # Limpiar y redibujar
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("Tiempo (s)")
        self.ax.set_ylabel("Amplitud (V)")
        
        time_data = self.df[self.time_col].values

        # --- NUEVO: Lógica de filtrado ---
        fs = 1 / (time_data[1] - time_data[0]) if len(time_data) > 1 else 2000 # Fs por defecto
        apply_notch = self.var_notch_filter.get()
        apply_bandpass = self.var_bandpass_filter.get()

        for col_name in self.channel_cols:
            if self.channel_vars.get(col_name) and self.channel_vars[col_name].get():
                y_data_original = self.df[col_name].values
                y_data_processed = y_data_original

                # Aplicar Notch si está habilitado
                if apply_notch and SCIPY_AVAILABLE:
                    try:
                        f0 = 50.0
                        Q = 30.0
                        b, a = signal.iirnotch(f0, Q, fs)
                        y_data_processed = signal.filtfilt(b, a, y_data_processed)
                    except Exception as e:
                        print(f"Error aplicando filtro Notch: {e}")

                # Aplicar Pasa-Banda si está habilitado
                if apply_bandpass and SCIPY_AVAILABLE:
                    try:
                        low_cut = float(self.spin_low_cut.get())
                        high_cut = float(self.spin_high_cut.get())
                        if low_cut < high_cut:
                            nyquist = 0.5 * fs
                            low = low_cut / nyquist
                            high = high_cut / nyquist
                            sos = signal.butter(4, [low, high], btype='band', output='sos')
                            y_data_processed = signal.sosfiltfilt(sos, y_data_processed)
                    except (ValueError, TypeError) as e:
                        print(f"Error en valores de filtro Pasa-Banda: {e}")

                # Submuestreo inteligente sobre los datos procesados
                x_data, y_data = downsample_lttb_fast(time_data, y_data_processed, MAX_POINTS_TO_PLOT)
                self.ax.plot(x_data, y_data, label=col_name, lw=1.2)
        
        if self.ax.has_data():
            # Limpiar y re-crear la leyenda para evitar duplicados
            if self.ax.get_legend() is not None: self.ax.get_legend().remove()
            self.ax.legend(loc='upper right') # <-- SOLUCIÓN: Posición fija para evitar cálculo lento.
        
        # Restaurar zoom si el usuario había hecho zoom manual
        if is_zoomed_x:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)

        self.canvas.draw()

    def on_ax_ylim_changed(self, ax):
        """Se activa cuando los límites del eje Y del gráfico cambian (por zoom, etc.)."""
        self.last_ylim = ax.get_ylim()
        # Actualiza la posición del scrollbar para que coincida con la vista
        self.update_y_scrollbar()

    def on_channel_toggle(self):
        self.plot_full_data()
        self.update_view_from_controls()


    def _update_plot_view_from_pan(self):
        """Actualiza la vista solo por paneo (movimiento del slider de tiempo)."""
        if self.df is None or not self.ax.has_data():
            return
        try:
            start_time = float(self.time_slider.get())
            duration = float(self.x_zoom_slider.get())
        except (ValueError, TypeError):
            return

        end_time = start_time + duration
        self.ax.set_xlim(start_time, end_time)
        self.canvas.draw()

    def on_pan_control_changed(self, *args):
        """Se llama cuando se mueve el slider de posición (paneo)."""
        if self._is_programmatically_updating_sliders:
            return
        self._update_plot_view_from_pan()

    def on_zoom_control_changed(self, *args):
        """Se llama cuando se mueve el slider de duración (zoom). Implementa zoom centrado."""
        if self._is_programmatically_updating_sliders:
            return
        if self.df is None or not self.ax.has_data():
            return

        try:
            new_duration = float(self.x_zoom_slider.get())
            if new_duration <= 0:
                return
        except (ValueError, TypeError):
            return

        current_start, current_end = self.ax.get_xlim()
        center_time = (current_start + current_end) / 2.0

        new_start = center_time - (new_duration / 2.0)
        new_end = center_time + (new_duration / 2.0)

        # Corregir si se sale de los límites
        max_time = self.time_slider.cget("to")
        if new_start < 0:
            new_start = 0
            new_end = new_duration
        if new_end > max_time:
            new_end = max_time
            new_start = max_time - new_duration

        self.ax.set_xlim(new_start, new_end)
        self._is_programmatically_updating_sliders = True
        self.time_slider.set(new_start)
        self._is_programmatically_updating_sliders = False
        self.canvas.draw()

    def update_view_from_controls(self, *args):
        """Función genérica para actualizar la vista, llamada al inicio."""
        self._update_plot_view_from_pan()
        self.canvas.draw()

    def on_y_pan_changed(self, scroll_value_str):
        """Se activa cuando el usuario mueve el deslizador de posición vertical."""
        if self.df is None or not self.ax.has_data() or self._is_programmatically_updating_y_scroll:
            return

        # El valor del slider va de 100 (arriba) a 0 (abajo). Lo invertimos y normalizamos a 0-1.
        scroll_value = float(scroll_value_str)
        normalized_value = (100 - scroll_value) / 100.0

        # Mapear el valor normalizado (0.0 a 1.0) al rango de datos global
        y_range_global = self.max_y_global - self.min_y_global if self.max_y_global > self.min_y_global else 1
        new_center_y = self.min_y_global + normalized_value * y_range_global

        # Obtener la altura de la vista actual para mantener el nivel de zoom
        current_ymin, current_ymax = self.ax.get_ylim()
        view_height = current_ymax - current_ymin
        
        # Establecer los nuevos límites centrados en la posición del scroll
        self.ax.set_ylim(new_center_y - view_height / 2, new_center_y + view_height / 2)
        self.canvas.draw()

    def update_y_zoom(self, *args):
        """Ajusta el zoom del eje Y basado en el slider vertical."""
        if self.df is None or not self.ax.has_data():
            return
        
        zoom_percentage = float(self.y_zoom_slider.get()) / 100.0
        
        current_ymin, current_ymax = self.ax.get_ylim()
        y_center = (current_ymax + current_ymin) / 2.0
        
        global_range = self.max_y_global - self.min_y_global
        visible_range = global_range * zoom_percentage
        
        new_min_y = y_center - visible_range / 2.0
        new_max_y = y_center + visible_range / 2.0
        self.ax.set_ylim(new_min_y, new_max_y)
        
        # No es necesario llamar a self.update_y_scrollbar() aquí porque
        # set_ylim dispara el callback on_ax_ylim_changed, que ya lo hace.
        self.canvas.draw()

    def update_y_scrollbar(self):
        """Actualiza la posición y el tamaño del 'pulgar' del scrollbar vertical."""
        if self.df is None or self._is_programmatically_updating_y_scroll:
            return

        self._is_programmatically_updating_y_scroll = True
        
        current_ymin, current_ymax = self.ax.get_ylim()
        view_center = (current_ymin + current_ymax) / 2.0
        
        y_range_global = self.max_y_global - self.min_y_global

        if y_range_global > 0:
            # Mapear la posición central de la vista al rango del slider (0-100)
            normalized_center = (view_center - self.min_y_global) / y_range_global
            # Invertir para que el máximo de la señal esté arriba (valor 100)
            slider_value = 100 - (normalized_center * 100)
            self.y_pan_slider.set(slider_value)
            
        self._is_programmatically_updating_y_scroll = False

    def autoscale_view(self):
        """Ajusta los ejes X e Y para mostrar todos los datos y resetea los sliders."""
        if self.df is None or not self.ax.has_data():
            return

        # 1. Ajustar los límites de los ejes para mostrar todo
        max_time = self.time_slider.cget("to")
        self.ax.set_xlim(0, max_time)
        self.ax.set_ylim(self.min_y_global, self.max_y_global)

        # 2. Resetear los sliders a sus valores por defecto
        # Usar la bandera para evitar que los sliders disparen sus propios eventos de actualización
        self._is_programmatically_updating_sliders = True
        
        self.time_slider.set(0)
        self.x_zoom_slider.set(max_time) # El zoom X es la duración total
        self.y_zoom_slider.set(100) # El zoom Y es el 100% (sin zoom)
        
        self._is_programmatically_updating_sliders = False

        # El slider de posición Y se actualizará automáticamente porque set_ylim
        # dispara el evento 'on_ax_ylim_changed', que a su vez llama a 'update_y_scrollbar'.
        self.canvas.draw()

    def export_png(self):
        """Exporta la vista actual del gráfico a un archivo PNG."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")],
            title="Guardar vista como PNG"
        )
        if not filepath:
            return
        
        if self.df is None:
            messagebox.showerror("Error", "No hay datos cargados para exportar.")
            return

        # --- NUEVO: Lógica de exportación en alta calidad ---
        # 1. Obtener los límites actuales de la vista
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()

        # 2. Filtrar el DataFrame original para obtener solo los datos visibles
        visible_df = self.df[(self.df[self.time_col] >= current_xlim[0]) & (self.df[self.time_col] <= current_xlim[1])]

        # 3. Crear una nueva figura temporal para exportar
        export_fig, export_ax = plt.subplots(figsize=(15, 8), dpi=300)
        export_ax.grid(True)
        export_ax.set_xlabel("Tiempo (s)")
        export_ax.set_ylabel("Amplitud (V)")
        export_ax.set_title(f"Exportación de Vista: {os.path.basename(self.last_loaded_filepath)}")
        
        # 4. Dibujar los datos de alta calidad en la nueva figura
        time_data = visible_df[self.time_col]
        for col_name in self.channel_cols:
            if self.channel_vars.get(col_name) and self.channel_vars[col_name].get():
                export_ax.plot(time_data, visible_df[col_name], label=col_name, lw=1.2)

        # 5. Aplicar los mismos límites de zoom y leyenda
        export_ax.set_xlim(current_xlim)
        export_ax.set_ylim(current_ylim)
        if export_ax.has_data():
            export_ax.legend(loc='upper right')

        # 6. Guardar la figura de alta calidad
        try:
            export_fig.savefig(filepath, bbox_inches='tight')
            messagebox.showinfo("Exportado", f"Gráfico guardado en:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error al exportar", f"No se pudo guardar el archivo.\nError: {e}")
        finally:
            plt.close(export_fig) # Liberar memoria


if __name__ == "__main__":
    root = tk.Tk()
    app = CSVViewerApp(root)
    root.mainloop()