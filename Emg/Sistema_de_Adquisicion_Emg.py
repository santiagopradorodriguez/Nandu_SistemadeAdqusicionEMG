import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os
from pathlib import Path

# --- NUEVO: Import para manejar imágenes ---
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
# Esta es la última versión funcional conocida.

# --- Versión del lanzador ---
VERSION = "3.1"

# =============================================================
# CONFIGURACIÓN
# =============================================================
BASE_DIR = Path(__file__).resolve().parent
ICON_DIR = BASE_DIR / "icons"

# --- Lista de scripts a lanzar ---
# Formato: (Nombre para mostrar, archivo.py, nombre_del_icono.png, color_hex)
MAIN_SCRIPTS = [
    ("1. Medir Señal", "CodigoUnificador_integrado.py", "measure.png", "#007BFF"),
    ("2. Ver Señales Crudas", "visor_csv_interactivo.py", "view_raw.png", "#17A2B8"),
    ("3. Análisis de Señal Ruido", "analisis_por_track_integrado.py", "analyze.png", "#28A745"),
    ("4. Ver Resultados", "electrode_viewer_4.py", "results.png", "#6C757D"),
]

UTILITY_SCRIPTS = [
    ("Instrucciones", "instrucciones_uso.py", "instructions.png", "#ffc107", "black"),
    ("Metrónomo", "metronomo_visual.py", "metronome.png", "#FD7E14", "white"),
    ("Editar Medición(En Desarrollo)", "editor_mediciones.py", "edit_measurement.png", "#E83E8C", "white"),
    ("Extraer Datos(En Desarrollo)", "extractor_de_datos_procesados.py", "extract.png", "#6610f2", "white"),
    ("Graficador", "plotter_calibrado.py", "plot_calibrated.png", "#20c997", "white"),
    ("Análisis Correlación(En Desarrollo)", "correlaciondeseñales.py", "correlation.png", "#d63384", "white"),
]

# =============================================================
# HELPERS
# =============================================================
def launch_script(script_path_rel):
    """
    Lanza un script de Python en un nuevo proceso.
    """
    full_path = BASE_DIR / script_path_rel
    if not full_path.exists():
        messagebox.showerror("Error", f"No se encontró el archivo:\n{full_path}")
        return

    try:
        python_executable = sys.executable
        print(f"Lanzando: {python_executable} {full_path}")
        # --- MODIFICACIÓN PARA DEPURACIÓN ---
        # En Windows, para poder ver los errores en los scripts que se lanzan,
        # abrimos el script en una nueva ventana de comandos que no se cierra
        # automáticamente al terminar. El comando 'cmd /k' ejecuta el script y
        # mantiene la ventana abierta.
        if sys.platform == "win32":
            subprocess.Popen(['cmd', '/c', 'start', 'cmd', '/k', python_executable, str(full_path)])
        else:
            # Comportamiento original para otros sistemas operativos (Linux, macOS)
            subprocess.Popen([python_executable, str(full_path)])
    except Exception as e:
        messagebox.showerror("Error de Lanzamiento", str(e))

def open_script_in_editor(script_path_rel):
    """Abre el archivo de script en el editor de texto predeterminado."""
    full_path = BASE_DIR / script_path_rel
    if not full_path.exists():
        messagebox.showerror("Error", f"No se encontró el archivo:\n{full_path}")
        return
    try:
        os.startfile(full_path)  # Solo para Windows
    except AttributeError:
        # Fallback para Linux/macOS
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, str(full_path)])
    except Exception as e:
        messagebox.showerror("Error al abrir", f"No se pudo abrir el archivo:\n{e}")

def load_icon(name, size=(24, 24)):
    """Carga un icono desde la carpeta de iconos."""
    if not PIL_AVAILABLE:
        return None
    icon_path = ICON_DIR / name
    if not icon_path.exists():
        print(f"Advertencia: No se encontró el icono '{name}'")
        return None
    try:
        img = Image.open(icon_path).resize(size, Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print(f"Error al cargar el icono '{name}': {e}")
        return None

def main():
    root = tk.Tk()
    root.title(f"Sistema de Adquisición EMG - v{VERSION}") # Título actualizado
    root.geometry("550x750") # Más alto para el logo
    root.resizable(False, False)

    # --- Estilos con ttk ---
    style = ttk.Style(root)
    style.theme_use('clam') # Usar un tema que permita personalizar colores de fondo
    style.configure("TFrame", background="#f0f2f5")
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), background="#f0f2f5")
    
    # --- MEJORA: Estilos de color para tarjetas y botones ---
    for i, (_, _, _, color) in enumerate(MAIN_SCRIPTS):
        style.configure(f"Card{i}.TFrame", background=color, relief="raised", borderwidth=1)
        style.configure(f"CardTitle{i}.TLabel", font=("Segoe UI", 12, "bold"), foreground="white", background=color)
        # Estilo para el hover (cuando el mouse está encima)
        style.map(f"Card{i}.TFrame", background=[('active', '#ffffff')])

    for i, (_, _, _, color, fg_color) in enumerate(UTILITY_SCRIPTS):
        style.configure(f"Util{i}.TButton", font=("Segoe UI", 9, "bold"), background=color, foreground=fg_color)
        style.map(f"Util{i}.TButton",
                  background=[('active', '#ffffff'), ('!disabled', color)],
                  foreground=[('active', 'black')])

    main_frame = ttk.Frame(root, padding=15, style="TFrame")
    main_frame.pack(expand=True, fill="both")

    # --- MEJORA: Cargar y mostrar el logo ---
    if PIL_AVAILABLE:
        try:
            logo_path = None
            pictures_dir = BASE_DIR / "DataConfig" / "Pictures"
            if pictures_dir.is_dir():
                for filename in os.listdir(pictures_dir):
                    if filename.lower().startswith("logo"):
                        logo_path = pictures_dir / filename
                        break
            
            if logo_path and logo_path.exists():
                img = Image.open(logo_path)
                img.thumbnail((400, 200)) # Tamaño máximo del logo
                photo = ImageTk.PhotoImage(img)
                logo_label = ttk.Label(main_frame, image=photo, background="#f0f2f5")
                logo_label.image = photo # Guardar referencia
                logo_label.pack(pady=(0, 15))
        except Exception as e:
            print(f"Error al cargar el logo: {e}")

    # --- Cargar iconos una vez ---
    root.icons = {
        "edit_script": load_icon("edit_script.png", size=(18, 18))
    }
    # --- CORRECCIÓN: El bucle debe manejar tuplas de diferente longitud. ---
    # Usamos '*' para capturar los elementos restantes que no necesitamos en este bucle.
    for _, _, icon_name, *_ in MAIN_SCRIPTS + UTILITY_SCRIPTS:
        root.icons[icon_name] = load_icon(icon_name)

    ttk.Label(main_frame, text="Sistema de Adquisición EMG", style="Title.TLabel").pack(pady=(0, 20))

    # --- Contenedor de tarjetas para scripts principales ---
    scripts_grid = ttk.Frame(main_frame, style="TFrame")
    scripts_grid.pack(fill="x", pady=10)

    for i, (label, script_file, icon_name, color) in enumerate(MAIN_SCRIPTS):
        card = ttk.Frame(scripts_grid, padding=15, style=f"Card{i}.TFrame", cursor="hand2")
        card.grid(row=i, column=0, sticky="ew", pady=5)
        card.columnconfigure(1, weight=1)

        # --- CORRECCIÓN: La función bind() pasa un objeto 'event' que debemos ignorar. ---
        # La lambda ahora acepta 'event' como primer argumento, pero usa 's' (que tiene el valor por defecto de script_file).
        action = lambda event, s=script_file: launch_script(s)
        card.bind("<Button-1>", action)

        icon_label = ttk.Label(card, image=root.icons.get(icon_name), background=color)
        icon_label.grid(row=0, column=0, rowspan=2, padx=(0, 10))
        icon_label.bind("<Button-1>", action)

        title_label = ttk.Label(card, text=label, style=f"CardTitle{i}.TLabel")
        title_label.grid(row=0, column=1, sticky="w")
        title_label.bind("<Button-1>", action)
        
        btn_edit = ttk.Button(card, image=root.icons.get("edit_script"), style="Toolbutton", command=lambda s=script_file: open_script_in_editor(s))
        btn_edit.grid(row=0, column=2, rowspan=2, sticky="e", padx=(10, 0))

    # --- Contenedor para utilidades ---
    utils_frame = ttk.LabelFrame(main_frame, text="Utilidades", padding=10)
    utils_frame.pack(fill="x", pady=20)

    num_util_cols = 3  # Máximo de botones por fila
    for i, (label, script_file, icon_name, color, fg_color) in enumerate(UTILITY_SCRIPTS):
        # Re-configurar el estilo para cada botón de utilidad con su color específico
        style.configure(f"Util{i}.TButton", font=("Segoe UI", 9, "bold"), background=color, foreground=fg_color)
        style.map(f"Util{i}.TButton", background=[('active', '#dddddd')])

        btn = ttk.Button(utils_frame, text=label, image=root.icons.get(icon_name), compound="left", style=f"Util{i}.TButton", command=lambda s=script_file: launch_script(s))
        
        row = i // num_util_cols
        col = i % num_util_cols
        
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
        utils_frame.columnconfigure(col, weight=1)

    # --- Etiqueta de versión ---
    version_label = ttk.Label(main_frame, text=f"v{VERSION}", font=("Segoe UI", 8), foreground="grey", background="#f0f2f5")
    version_label.pack(side="bottom", anchor="se", pady=(10, 0))

    root.mainloop()

if __name__ == "__main__":
    main()