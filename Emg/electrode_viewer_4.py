#!/usr/bin/env python3
"""
electrode_viewer.py - v5.0

Reestructurado para una vista por Medición en lugar de por Canal.

Requisitos: pillow, matplotlib, numpy
"""
import os, io, json
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from tkinter import ttk # Usar ttk para las pestañas

# -------------------------
# Config
# -------------------------
VERSION = "5.0"

THUMB_SIZE = (220, 140)
DETAIL_THUMB = (220, 120)
DEFAULT_COLUMNS = 5
MAX_SELECT = 3

PHOTO_NAMES = ["thumbnail.png", "thumbnail.jpg", "photo.png", "photo.jpg"]
PULSES_NAMES = ["pulses.png", "pulses_centrados.png"]
AVG_NAMES = ["avg.png", "avg_lider.png"]
SPEC_NAMES = ["spec.png", "spec_lider.png"]

# --- NUEVO: Nombres de los archivos de análisis adicionales ---
ADDITIONAL_ANALYSIS_FILES = [
    "analisis_comparativo.png",
    "analisis_comparativo_overlay.png",
    "analisis_comparativo_snr_grouped.png",
    "analisis_comparativo_amplitud_max_bar.png"
]

# -------------------------
# Helpers
# -------------------------
def find_file_with_any(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.isfile(p):
            return p
    return None

def parse_date_flexible(s):
    if not s: return None
    if isinstance(s, datetime): return s
    s = s.strip()
    try: return datetime.fromisoformat(s)
    except: pass
    for fm in ("%Y-%m-%d","%d-%m-%Y","%d/%m/%Y","%Y/%m/%d","%d %b %Y","%d %B %Y"):
        try: return datetime.strptime(s, fm)
        except: pass
    import re
    m = re.search(r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})", s)
    if m:
        y,mo,d = m.groups()
        try: return datetime(int(y), int(mo), int(d))
        except: return None
    return None

def read_metadata(folder):
    out = {"snr": None, "measurement_date": None, "origin_date": None}
    p = os.path.join(folder, "metadata.json")
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                j = json.load(fh)
            if "snr" in j:
                try: out["snr"] = float(j["snr"])
                except: out["snr"] = None
            md = j.get("measurement_date") or j.get("date") or j.get("measured_on")
            od = j.get("origin_date") or j.get("origin") or j.get("created")
            out["measurement_date"] = parse_date_flexible(md) if md else None
            out["origin_date"] = parse_date_flexible(od) if od else None
        except Exception:
            pass
    if out["snr"] is None:
        p2 = os.path.join(folder, "snr.txt")
        if os.path.isfile(p2):
            try:
                with open(p2, "r", encoding="utf-8") as fh:
                    t = fh.read().strip()
                    for token in t.replace(",", " ").split():
                        try:
                            out["snr"] = float(token); break
                        except: continue
            except:
                pass
    return out

def read_channel_metadata(channel_folder):
    """Lee y parsea metadata.json de una carpeta de canal."""
    meta_path = os.path.join(channel_folder, "metadata.json")
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def read_analysis_results(channel_folder):
    """
    Busca y lee el archivo 'results.json' directamente dentro de la carpeta del canal.
    """
    json_path = os.path.join(channel_folder, "results.json")
    if os.path.isfile(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if data.get("snr_manual") is not None:
                    return data.get("snr_manual"), data.get("snr_uncertainty")
        except Exception:
            pass # Ignora errores de parseo o de archivo
    return None, None

def pil_image_to_tk_photoimage(pil_image, size):
    im = pil_image.convert("RGBA")
    im = ImageOps.fit(im, size, Image.LANCZOS)
    b = io.BytesIO(); im.save(b, format="PNG"); b.seek(0)
    return tk.PhotoImage(data=b.read())

# -------------------------
# Model
# -------------------------
class Channel:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.analysis_snr, self.analysis_snr_uncertainty = read_analysis_results(path)
        self.metadata = read_channel_metadata(path)

        # Diccionario para almacenar los gráficos encontrados
        self.plots = {
            "Promedio (Ajustado Por Correlación)": find_file_with_any(self.path, ["avg_lider.png"]),
            "Pico Maximo Promedio ": find_file_with_any(self.path, ["avg.png"]),
            "Recortes (Original)": find_file_with_any(self.path, ["pulses.png"]),
            "Espectrograma de Promedio por Correlacion": find_file_with_any(self.path, ["spec_lider.png"]),
            "Espectrograma Pico Maximo promedio": find_file_with_any(self.path, ["spec.png"]),
        }
        # Filtrar los que no se encontraron
        self.plots = {k: v for k, v in self.plots.items() if v}

class Measurement:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)

        # Leer metadatos de la medición
        meta = read_metadata(path)
        self.snr = meta.get("snr")
        self.origin_date = meta.get("origin_date")
        self.measurement_date = meta.get("measurement_date")

        # Primero, encuentra todos los canales y carga sus metadatos
        self.channels = []
        for item in sorted(os.listdir(path)):
            channel_path = os.path.join(path, item)
            if os.path.isdir(channel_path) and item.startswith("canal_"):
                self.channels.append(Channel(channel_path))

        # Ahora, si no se encontró fecha de medición en la raíz,
        # intenta obtenerla desde el primer canal como fallback.
        if not self.measurement_date and self.channels:
            # La fecha es la misma para todos los canales, así que tomamos la del primero.
            date_str = self.channels[0].metadata.get("measurement_date")
            self.measurement_date = parse_date_flexible(date_str)

        # Busca una imagen representativa para la miniatura
        self.thumbnail_plot_path = self.get_representative_plot()
        # Busca una foto real como fallback o para la vista de detalle
        self.photo_path = find_file_with_any(self.path, PHOTO_NAMES)

        # Path al gráfico calibrado
        calibrated_plot_name = f"plot_calibrado_{self.name}.png"
        calibrated_plot_path = os.path.join(self.path, calibrated_plot_name)
        self.calibrated_plot_path = calibrated_plot_path if os.path.isfile(calibrated_plot_path) else None

        # Definir paths a plots representativos para la comparación
        self.pulses_path = None
        self.avg_path = None
        if self.channels:
            # Prioriza canal_0, si no, usa el primero que encuentre
            channel_to_check = next((ch for ch in self.channels if ch.name == "canal_0"), self.channels[0])
            self.pulses_path = find_file_with_any(channel_to_check.path, PULSES_NAMES)
            self.avg_path = find_file_with_any(channel_to_check.path, AVG_NAMES)

    def get_representative_plot(self):
        """
        Busca un gráfico representativo para la miniatura.
        Prioridad:
        1. Gráfico calibrado en la raíz de la medición.
        2. Gráfico de pulso promedio ('avg.png') en el primer canal (o canal_0).
        """
        # 1. Buscar el gráfico calibrado en la raíz de la medición.
        calibrated_plot_name = f"plot_calibrado_{self.name}.png"
        calibrated_plot_path = os.path.join(self.path, calibrated_plot_name)
        if os.path.isfile(calibrated_plot_path):
            return calibrated_plot_path

        # 2. Fallback: buscar el gráfico de pulso promedio en los canales.
        if not self.channels:
            return None
        
        # Prioriza canal_0, si no, usa el primero que encuentre
        channel_to_check = next((ch for ch in self.channels if ch.name == "canal_0"), self.channels[0])
        
        return find_file_with_any(channel_to_check.path, AVG_NAMES)

# -------------------------
# Magnifier helper
# -------------------------
class Magnifier:
    def __init__(self, master_tk, fig_canvas, ax, image_path):
        self.master = master_tk
        self.canvas = fig_canvas
        self.ax = ax
        self.image_path = image_path
        self.zoom_factor = 3.0
        self.size_px = 280
        self._connected = False
        self.zoom_win = None
        self._image_arr = None
        self._load_image(image_path)

    def _load_image(self, image_path):
        if image_path and os.path.isfile(image_path):
            try:
                im = Image.open(image_path).convert("RGB")
                self._pil = im
                self._image_arr = np.array(im)
            except:
                self._pil = None
                self._image_arr = None
        else:
            self._pil = None
            self._image_arr = None

    def set_zoom(self, z):
        try:
            v = float(z); self.zoom_factor = max(1.0, v)
        except: pass

    def enable(self):
        if self._image_arr is None:
            messagebox.showinfo("Magnifier", "Imagen no disponible para la lupa.")
            return
        if not self._connected:
            self.cid = self.canvas.mpl_connect("motion_notify_event", self._on_motion)
            self._connected = True
            if self.zoom_win is None or not tk.Toplevel.winfo_exists(self.zoom_win):
                self.zoom_win = tk.Toplevel(self.master)
                self.zoom_win.wm_title("Magnifier")
                self.zoom_label = tk.Label(self.zoom_win)
                self.zoom_label.pack()
                self.zoom_win.geometry(f"{self.size_px}x{self.size_px}")
                self.zoom_win.attributes("-topmost", True)

    def disable(self):
        if self._connected:
            try: self.canvas.mpl_disconnect(self.cid)
            except: pass
            self._connected = False
        if self.zoom_win is not None and tk.Toplevel.winfo_exists(self.zoom_win):
            try: self.zoom_win.destroy()
            except: pass
            self.zoom_win = None

    def update_image(self, image_path):
        self.image_path = image_path
        self._load_image(image_path)

    def _on_motion(self, event):
        if event.inaxes is None or event.inaxes != self.ax: return
        if self._image_arr is None: return
        x = int(event.xdata) if event.xdata is not None else None
        y = int(event.ydata) if event.ydata is not None else None
        if x is None or y is None: return
        h, w = self._image_arr.shape[0], self._image_arr.shape[1]
        half_w = int((self.size_px / self.zoom_factor) / 2)
        half_h = int((self.size_px / self.zoom_factor) / 2)
        x0 = max(0, x - half_w); x1 = min(w, x + half_w)
        y0 = max(0, y - half_h); y1 = min(h, y + half_h)
        crop = self._image_arr[y0:y1, x0:x1]
        if crop.size == 0: return
        pil = Image.fromarray(crop).resize((self.size_px, self.size_px), Image.LANCZOS)
        tkim = pil_image_to_tk_photoimage(pil, (self.size_px, self.size_px))
        if self.zoom_win and tk.Toplevel.winfo_exists(self.zoom_win):
            self.zoom_label.configure(image=tkim); self.zoom_label.image = tkim

# -------------------------
# App
# -------------------------
class App:
    def __init__(self, root):
        self.root = root
        root.title(f"Electrode Viewer - v{VERSION}")
        self.base_folder = "base_de_datos_electrodos" # <-- CARPETA POR DEFECTO
        self.measurements = []
        self.thumb_cache = {}
        self.selected = {}
        self.columns = DEFAULT_COLUMNS
        self.sort_by = tk.StringVar(value="SNR")

        top = tk.Frame(root); top.pack(fill="x", padx=6, pady=6)
        tk.Button(top, text="Abrir otra carpeta...", command=self.open_folder).pack(side="left")
        tk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=4)
        tk.Label(top, text="Columns:").pack(side="left", padx=(12,2))
        self.spin_cols = tk.Spinbox(top, from_=1, to=8, width=3, command=self.on_columns_change)
        self.spin_cols.delete(0,"end"); self.spin_cols.insert(0, str(DEFAULT_COLUMNS)); self.spin_cols.pack(side="left")
        tk.Label(top, text="Sort by:").pack(side="left", padx=(12,2))
        self.sort_menu = tk.OptionMenu(top, self.sort_by, "SNR", "Measurement date"); self.sort_menu.pack(side="left")
        tk.Button(top, text="Apply sort", command=self.build_grid).pack(side="left", padx=6)

        # --- NUEVO: Botón para ver comparaciones ---
        tk.Button(top, text="Ver Análisis Comparativos", command=self.open_comparisons_folder, bg="#ffc107").pack(side="left", padx=20)

        tk.Button(top, text="Compare Selected", command=self.compare_selected).pack(side="right")
        self.info_label = tk.Label(top, text=f"Mostrando: {os.path.abspath(self.base_folder)}"); self.info_label.pack(side="left", padx=8)

        self.canvas = tk.Canvas(root, height=700)
        self.v_scroll = tk.Scrollbar(root, orient="vertical", command=self.canvas.yview)
        self.grid_container = tk.Frame(self.canvas)
        self.grid_container.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0,0), window=self.grid_container, anchor="nw")
        self.canvas.configure(yscrollcommand=self.v_scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.v_scroll.pack(side="right", fill="y")

        # --- Carga inicial ---
        self.refresh()

    def open_folder(self):
        folder = filedialog.askdirectory(title="Seleccionar carpeta base de electrodos")
        if not folder: return
        self.base_folder = folder
        self.info_label.config(text=f"Mostrando: {os.path.abspath(self.base_folder)}")
        self.load_measurements(); self.build_grid()

    def open_comparisons_folder(self):
        """Abre la carpeta donde se guardan los análisis comparativos."""
        comp_dir = "analisis_comparativos"
        abs_path = os.path.abspath(comp_dir)
        if not os.path.isdir(abs_path):
            messagebox.showinfo("Información", f"La carpeta de análisis comparativos no existe aún.\n({abs_path})")
            return
        try:
            print(f"Abriendo carpeta de comparaciones: {abs_path}")
            os.startfile(abs_path)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la carpeta:\n{e}")

    def refresh(self):
        if not self.base_folder:
            messagebox.showinfo("Info", "No se ha configurado una carpeta base."); return
        self.load_measurements(); self.build_grid()

    def load_measurements(self):
        self.measurements = []
        if not self.base_folder: return
        for name in sorted(os.listdir(self.base_folder)):
            measurement_path = os.path.join(self.base_folder, name)
            if os.path.isdir(measurement_path):
                self.measurements.append(Measurement(measurement_path))
        self.apply_sorting()

    def apply_sorting(self):
        key = self.sort_by.get()
        if key == "Measurement date":
            def mk(m): return m.measurement_date.timestamp() if m.measurement_date else -1e12
            self.measurements.sort(key=lambda x: mk(x), reverse=True)
        else: # Por defecto, ordena por nombre (SNR no está en Measurement)
            self.measurements.sort(key=lambda x: x.name)

    def on_columns_change(self):
        try: v = int(self.spin_cols.get()); self.columns = max(1, min(8, v))
        except: self.columns = DEFAULT_COLUMNS
        self.build_grid()

    def build_grid(self):
        self.apply_sorting()
        for child in self.grid_container.winfo_children(): child.destroy()
        self.thumb_cache.clear(); self.selected.clear()
        cols = self.columns
        pad = 8
        r = c = 0
        for m in self.measurements:
            frame = tk.Frame(self.grid_container, bd=1, relief="groove", padx=6, pady=6)
            frame.grid(row=r, column=c, padx=pad, pady=pad, sticky="n")
            
            # Usa el gráfico del pulso como miniatura, con la foto como fallback
            thumb_path = m.thumbnail_plot_path if m.thumbnail_plot_path else m.photo_path
            thumb_img = self.make_thumbnail(thumb_path, m.path, THUMB_SIZE)
            lbl = tk.Label(frame, image=thumb_img, cursor="hand2"); lbl.image = thumb_img; lbl.pack()
            lbl.bind("<Button-1>", lambda ev, measurement=m: self.open_detail(measurement))
            
            name_lbl = tk.Label(frame, text=m.name, font=("Arial", 11, "bold"), wraplength=THUMB_SIZE[0])
            name_lbl.pack(anchor="w", pady=(6,0))
            
            md_text = f'Fecha: {m.measurement_date.strftime("%Y-%m-%d")}' if m.measurement_date else "Fecha: N/A"
            info_lbl = tk.Label(frame, text=md_text, font=("Arial", 9), justify="left"); info_lbl.pack(anchor="w", pady=(4,0))
            var = tk.IntVar(value=0); chk = tk.Checkbutton(frame, text="Select", variable=var); chk.pack(anchor="w", pady=(4,0))
            self.selected[m.name] = (var, m)
            c += 1
            if c >= cols: c = 0; r += 1

    def make_thumbnail(self, image_path, cache_key, size=THUMB_SIZE):
        if cache_key in self.thumb_cache: return self.thumb_cache[cache_key]
        if image_path and os.path.isfile(image_path):
            try: pil = Image.open(image_path)
            except: pil = Image.new("RGB", size, (220,220,220))
        else: pil = Image.new("RGB", size, (240,240,240)) # Fallback a un cuadro gris
        try: tkimg = pil_image_to_tk_photoimage(pil, size)
        except: tkimg = tk.PhotoImage(width=size[0], height=size[1])
        self.thumb_cache[cache_key] = tkimg; return tkimg

    # ---------------- Detail window ----------------
    def open_detail(self, measurement: Measurement):
        win = tk.Toplevel(self.root)
        win.geometry("1000x750")

        # Panel superior con información general
        top_frame = tk.Frame(win, bd=1, relief="solid", padx=5, pady=5)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        md_text = f'Fecha: {measurement.measurement_date.strftime("%Y-%m-%d %H:%M:%S")}' if measurement.measurement_date else "Fecha: N/A"
        win.title(f"Detalle - {measurement.name} | {md_text}")
        tk.Label(top_frame, text=f"Medición: {measurement.name}", font=("Arial", 12, "bold")).pack(side="left")
        tk.Label(top_frame, text=f" | {md_text}", font=("Arial", 11)).pack(side="left", padx=10)

        # --- Botón para ver análisis adicionales ---
        def open_additional_analysis():
            found_any = False
            for fname in ADDITIONAL_ANALYSIS_FILES:
                fpath = os.path.join(measurement.path, fname)
                if os.path.exists(fpath):
                    try:
                        os.startfile(fpath)
                        found_any = True
                    except Exception as e:
                        print(f"Error al abrir '{fpath}': {e}")
            if not found_any:
                messagebox.showinfo("Información", "No se encontraron archivos de análisis adicionales en esta carpeta.")
        tk.Button(top_frame, text="Ver Análisis Adicional", command=open_additional_analysis).pack(side="right", padx=10)
        # Notebook (pestañas) para cada canal
        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def show_on_main(ax, canvas, image_path, title):
            ax.clear()
            if image_path and os.path.isfile(image_path):
                try:
                    img = plt.imread(image_path)
                    ax.imshow(img)
                    ax.set_title(title)
                except Exception as e:
                    ax.text(0.5, 0.5, f"No se pudo cargar la imagen:\n{e}", ha="center", va="center", wrap=True)
            else:
                ax.text(0.5, 0.5, "Archivo no encontrado", ha="center", va="center")
            ax.axis("off")
            canvas.draw()

        # --- NUEVO: Pestaña para el gráfico combinado (plot_calibrado_*) ---
        combined_plot_path = os.path.join(measurement.path, f"plot_calibrado_{measurement.name}.png")
        if os.path.isfile(combined_plot_path):
            tab_comb = tk.Frame(notebook)
            notebook.add(tab_comb, text="  Señales Musculares  ")
            
            fig_comb = plt.Figure()
            ax_comb = fig_comb.add_subplot(111)
            
            canvas_comb = FigureCanvasTkAgg(fig_comb, master=tab_comb)
            widget_comb = canvas_comb.get_tk_widget()
            widget_comb.pack(side="top", fill="both", expand=True)

            toolbar_frame_comb = tk.Frame(tab_comb)
            toolbar_frame_comb.pack(fill="x", side="bottom")
            toolbar_comb = NavigationToolbar2Tk(canvas_comb, toolbar_frame_comb)
            toolbar_comb.update()
            
            show_on_main(ax_comb, canvas_comb, combined_plot_path, "Señales Musculares")

        if not measurement.channels:
            # Si no hay canales, y tampoco plot combinado, mostrar mensaje.
            # Si hay plot combinado, se mostrará y la función terminará.
            if not os.path.isfile(combined_plot_path):
                tk.Label(notebook, text="No se encontraron carpetas de canal analizadas ni plot calibrado.").pack(pady=50)
            return

        for channel in measurement.channels:
            self.create_channel_tab(notebook, channel, show_on_main)
            
    # ---------------- Compare window ----------------
    def compare_selected(self):
        selected_list = []
        for name,(var,e) in self.selected.items():
            if var.get(): selected_list.append(e)
        if not selected_list:
            messagebox.showinfo("Info", "Seleccioná 1-3 electrodos (casillas) para comparar."); return
        if len(selected_list) > MAX_SELECT:
            messagebox.showwarning("Warning", f"Seleccioná hasta {MAX_SELECT} electrodos."); return
        self.open_compare_window(selected_list)

    def create_channel_tab(self, notebook, channel: Channel, show_on_main_func):
        tab = tk.Frame(notebook)
        notebook.add(tab, text=f"  {channel.name.upper()}  ")

        left = tk.Frame(tab); left.pack(side="left", fill="y", padx=10, pady=10)
        right = tk.Frame(tab); right.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        info_text = ""
        if channel.analysis_snr is not None:
            snr_val = f"{channel.analysis_snr:.2f}"
            unc_val = f" ± {channel.analysis_snr_uncertainty:.2f}" if channel.analysis_snr_uncertainty is not None else ""
            info_text += f"SNR (Análisis): {snr_val}{unc_val}\n"
        tk.Label(left, text=info_text, font=("Arial", 11, "bold"), justify="left").pack(anchor="w", pady=5)

        main_fig = plt.Figure(figsize=(7, 6)); main_ax = main_fig.add_subplot(111); main_ax.axis("off")
        main_canvas = FigureCanvasTkAgg(main_fig, master=right); main_canvas.draw()
        main_widget = main_canvas.get_tk_widget(); main_widget.pack(fill="both", expand=True)
        toolbar_frame = tk.Frame(right); toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(main_canvas, toolbar_frame); toolbar.update()

        # Botones para seleccionar qué gráfico ver
        for plot_name, plot_path in channel.plots.items():
            tk.Button(left, text=plot_name, command=lambda p=plot_path, t=plot_name: show_on_main_func(main_ax, main_canvas, p, t)).pack(fill="x", pady=2)

        # --- NUEVO: Mostrar metadata del canal ---
        meta_frame = tk.LabelFrame(left, text="Metadata", padx=5, pady=5)
        meta_frame.pack(fill="x", pady=(15, 0), expand=False)

        if channel.metadata:
            # Usar un Text widget para poder scrollear si hay muchos datos
            meta_text_widget = tk.Text(meta_frame, height=12, width=35, wrap="none", font=("Courier New", 8), relief="flat", bg=left.cget('bg'))
            scrollbar_y = tk.Scrollbar(meta_frame, command=meta_text_widget.yview, orient="vertical")
            scrollbar_x = tk.Scrollbar(meta_frame, command=meta_text_widget.xview, orient="horizontal")
            meta_text_widget.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
            
            pretty_meta = json.dumps(channel.metadata, indent=2, default=str)
            meta_text_widget.insert(tk.END, pretty_meta)
            meta_text_widget.config(state="disabled")
            
            scrollbar_y.pack(side="right", fill="y")
            scrollbar_x.pack(side="bottom", fill="x")
            meta_text_widget.pack(side="left", fill="both", expand=True)
        else:
            tk.Label(meta_frame, text="No se encontró metadata.json").pack()

    def open_compare_window(self, electrodes):
        n = len(electrodes)
        win = tk.Toplevel(self.root); win.title("Compare - " + ", ".join([e.name for e in electrodes]))
        top = tk.Frame(win); top.pack(fill="x", padx=6, pady=6)
        tk.Label(top, text="Show:").pack(side="left")
        view_choice = tk.StringVar(value="Señal Calibrada")
        for val in ("Señal Calibrada", "Photo", "Pulses", "Average", "All"):
            tk.Radiobutton(top, text=val, variable=view_choice, value=val, indicatoron=0).pack(side="left", padx=4)
        tk.Label(top, text="   Date shown:").pack(side="left", padx=(12,2))
        date_choice = tk.StringVar(value="Medición")
        tk.OptionMenu(top, date_choice, "Medición", "Origen").pack(side="left")
        tk.Button(top, text="Apply", command=lambda: render()).pack(side="right")

        container = tk.Frame(win); container.pack(fill="both", expand=True)

        def date_text(e):
            if date_choice.get() == "Medición":
                return f"Medición: {e.measurement_date.strftime('%Y-%m-%d')}" if e.measurement_date else "Medición: N/A"
            else:
                return f"Origen: {e.origin_date.strftime('%Y-%m-%d')}" if e.origin_date else "Origen: N/A"

        def render():
            for ch in container.winfo_children(): ch.destroy()
            choice = view_choice.get()
            if choice == "All":
                rows = 3; cols = n
                fig = plt.Figure(figsize=(5*cols, 4*rows)); axs=[]
                for r in range(rows):
                    row_axes=[]
                    for c in range(cols):
                        ax = fig.add_subplot(rows, cols, r*cols + c + 1); ax.axis("off"); row_axes.append(ax)
                    axs.append(row_axes)
                for idx,e in enumerate(electrodes):
                    ax_calibrado = axs[0][idx]
                    if e.calibrated_plot_path and os.path.isfile(e.calibrated_plot_path):
                        try: ax_calibrado.imshow(plt.imread(e.calibrated_plot_path))
                        except: ax_calibrado.text(0.5, 0.5, "Cannot load plot", ha="center")
                    else: ax_calibrado.text(0.5, 0.5, "No calibrated plot", ha="center")
                    snr_txt = f"{e.name}\nSNR: {e.snr:.2f}" if e.snr is not None else f"{e.name}\nSNR: N/A"
                    ax_calibrado.set_title(f"{snr_txt}\n{date_text(e)}", fontsize=9)
                    ax_p = axs[1][idx]
                    if e.pulses_path and os.path.isfile(e.pulses_path):
                        try: ax_p.imshow(plt.imread(e.pulses_path))
                        except: ax_p.text(0.5,0.5,"Cannot load pulses",ha="center")
                    else: ax_p.text(0.5,0.5,"No pulses file",ha="center")
                    ax_p.set_title("Pulses", fontsize=9)
                    ax_a = axs[2][idx]
                    if e.avg_path and os.path.isfile(e.avg_path):
                        try: ax_a.imshow(plt.imread(e.avg_path))
                        except: ax_a.text(0.5,0.5,"Cannot load avg",ha="center")
                    else: ax_a.text(0.5,0.5,"No avg file",ha="center")
                    ax_a.set_title("Average", fontsize=9)
                canvas = FigureCanvasTkAgg(fig, master=container); canvas.draw(); canvas.get_tk_widget().pack(fill="both", expand=True)
            else:
                # For each electrode, create its own Figure + Canvas + Toolbar + Magnifier controls
                # arrange them horizontally in container
                col_frame = tk.Frame(container); col_frame.pack(fill="both", expand=True)
                for idx,e in enumerate(electrodes):
                    cell = tk.Frame(col_frame, bd=1, relief="flat")
                    cell.pack(side="left", fill="both", expand=True, padx=4, pady=4)

                    fig = plt.Figure(figsize=(6,6))
                    ax = fig.add_subplot(111); ax.axis("off")
                    # load image path depending on choice
                    path = None
                    if choice == "Señal Calibrada": path = e.calibrated_plot_path
                    elif choice == "Photo": path = e.photo_path
                    elif choice == "Pulses": path = e.pulses_path
                    elif choice == "Average": path = e.avg_path

                    if path and os.path.isfile(path):
                        try: ax.imshow(plt.imread(path))
                        except: ax.text(0.5,0.5,"Cannot load",ha="center")
                    else:
                        ax.text(0.5,0.5,"No file",ha="center")
                    title = f"{e.name}\nSNR: {e.snr:.2f}" if e.snr is not None else f"{e.name}\nSNR: N/A"
                    title = title + "\n" + date_text(e)
                    ax.set_title(title, fontsize=10)

                    # canvas and toolbar inside this cell
                    canvas = FigureCanvasTkAgg(fig, master=cell)
                    canvas.draw()
                    widget = canvas.get_tk_widget(); widget.pack(fill="both", expand=True)
                    tbf = tk.Frame(cell); tbf.pack(fill="x")
                    toolbar = NavigationToolbar2Tk(canvas, tbf); toolbar.update()

                    # magnifier controls for THIS column (affects this axis)
                    mgf = tk.Frame(cell); mgf.pack(fill="x", pady=(4,4))
                    mag_var = tk.IntVar(value=0)
                    mag_chk = tk.Checkbutton(mgf, text="Magnifier", variable=mag_var)
                    mag_chk.pack(side="left", padx=(4,6))
                    tk.Label(mgf, text="Zoom:").pack(side="left")
                    mag_scale = tk.Scale(mgf, from_=1.5, to=8.0, resolution=0.5, orient="horizontal", length=140)
                    mag_scale.set(3.0); mag_scale.pack(side="left", padx=(4,6))

                    # instantiate magnifier for this axis
                    sample_path = path
                    mag = Magnifier(self.root, canvas, ax, sample_path)
                    def make_toggle(mag_obj, var_obj, scale_obj):
                        def toggle():
                            if var_obj.get():
                                mag_obj.set_zoom(scale_obj.get()); mag_obj.enable()
                            else:
                                mag_obj.disable()
                        return toggle
                    mag_chk.configure(command=make_toggle(mag, mag_var, mag_scale))
                    mag_scale.configure(command=lambda v, m=mag: m.set_zoom(v))

        render()

# -------------------------
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
