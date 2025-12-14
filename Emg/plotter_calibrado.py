# -*- coding: utf-8 -*-
"""
plotter_calibrado_secuencial_final.py

Script de visualización EMG MULTI-ARCHIVO.
Guarda automáticamente en la carpeta de origen con formato: plot_calibrado_{nombre}.png
"""

import os
import pandas as pd
import numpy as np
import json
from scipy import signal

# --- IMPORTANTE: Forzar backend TkAgg antes de importar pyplot ---
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Imports de Tkinter
from tkinter import Tk, Label, Button, Frame, Checkbutton, Radiobutton, Entry, StringVar, BooleanVar, Toplevel, Listbox, Scrollbar, MULTIPLE, END

# --- 1. CONFIGURACIÓN GENERAL ---

BASE_DIR = "base_de_datos_electrodos"

FACTORES_G = {
    'Canal 0': 495,
    'Canal 1': 495,
    'Canal 2': 495,
}

NOMBRES_CANALES_MAP = {
    'Canal 0': 'Depresor Anguli Oris',
    'Canal 1': 'Orbicularis Oris',
    'Canal 2': 'Mylohyoid',
}

# Parámetros Fijos
FREQ_NOTCH = 50.0            
Q_FACTOR_NOTCH = 30.0        
FREQ_PASABANDA = [20, 1000]  
ORDEN_PASABANDA = 4          
RMS_WINDOW_MS = 75           

# --- 2. CLASES DE INTERFAZ (GUI) ---

class VentanaSeleccion:
    def __init__(self, master, base_dir):
        self.master = master
        self.master.title("Selección de Mediciones")
        self.seleccionadas = []
        self.base_dir = base_dir

        Label(master, text="Seleccione las mediciones a procesar:", font=("Arial", 11, "bold")).pack(pady=10)
        Label(master, text="(Use Ctrl o Shift para selección múltiple)", font=("Arial", 8, "italic")).pack(pady=(0,5))

        frame_lista = Frame(master)
        frame_lista.pack(fill="both", expand=True, padx=10)

        scrollbar = Scrollbar(frame_lista)
        scrollbar.pack(side="right", fill="y")

        self.lista_items = Listbox(frame_lista, selectmode=MULTIPLE, width=50, height=15, yscrollcommand=scrollbar.set)
        self.lista_items.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.lista_items.yview)

        # Poblar lista
        self.items = self.obtener_carpetas()
        for item in self.items:
            self.lista_items.insert(END, item)

        Button(master, text="Continuar a Configuración >>", command=self.confirmar, bg="#DDDDDD", font=("Arial", 10, "bold")).pack(pady=15)

    def obtener_carpetas(self):
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            return []
        return [d for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]

    def confirmar(self):
        indices = self.lista_items.curselection()
        self.seleccionadas = [self.items[i] for i in indices]
        self.master.destroy()

class VentanaConfiguracion:
    def __init__(self, master):
        self.master = master
        self.master.title("Configuración de Procesamiento")
        self.resultado = None

        self.usar_notch = BooleanVar(value=True)
        self.usar_bandpass = BooleanVar(value=True)
        self.tipo_envolvente = StringVar(value="ninguna") 
        self.t_inicio = StringVar(value="") 
        self.t_fin = StringVar(value="")    

        # Filtros
        frame_filtros = Frame(master, padx=10, pady=10, borderwidth=1, relief="groove")
        frame_filtros.pack(fill="x", padx=10, pady=5)
        Label(frame_filtros, text="Filtros Digitales", font=("Arial", 10, "bold")).pack(anchor="w")
        Checkbutton(frame_filtros, text=f"Filtro Notch ({int(FREQ_NOTCH)} Hz)", variable=self.usar_notch).pack(anchor="w")
        Checkbutton(frame_filtros, text=f"Filtro Pasabanda ({FREQ_PASABANDA[0]}-{FREQ_PASABANDA[1]} Hz)", variable=self.usar_bandpass).pack(anchor="w")

        # Envolvente
        frame_env = Frame(master, padx=10, pady=10, borderwidth=1, relief="groove")
        frame_env.pack(fill="x", padx=10, pady=5)
        Label(frame_env, text="Procesamiento / Envolvente", font=("Arial", 10, "bold")).pack(anchor="w")
        Radiobutton(frame_env, text="Solo Señal Filtrada (Sin envolvente)", variable=self.tipo_envolvente, value="ninguna").pack(anchor="w")
        Radiobutton(frame_env, text="Envolvente de Hilbert (Analítica)", variable=self.tipo_envolvente, value="hilbert").pack(anchor="w")
        Radiobutton(frame_env, text=f"Envolvente RMS (Ventana {RMS_WINDOW_MS}ms)", variable=self.tipo_envolvente, value="rms").pack(anchor="w")

        # Tiempo
        frame_tiempo = Frame(master, padx=10, pady=10, borderwidth=1, relief="groove")
        frame_tiempo.pack(fill="x", padx=10, pady=5)
        Label(frame_tiempo, text="Intervalo de Tiempo (s)", font=("Arial", 10, "bold")).pack(anchor="w")
        Label(frame_tiempo, text="Dejar en blanco para graficar todo.", font=("Arial", 8, "italic")).pack(anchor="w", pady=(0, 5))

        frame_inputs = Frame(frame_tiempo)
        frame_inputs.pack(fill="x")
        Label(frame_inputs, text="Inicio:").pack(side="left")
        Entry(frame_inputs, textvariable=self.t_inicio, width=8).pack(side="left", padx=5)
        Label(frame_inputs, text="Fin:").pack(side="left")
        Entry(frame_inputs, textvariable=self.t_fin, width=8).pack(side="left", padx=5)

        Button(master, text="Empezar Secuencia", command=self.confirmar, bg="#DDDDDD", font=("Arial", 10, "bold")).pack(pady=15)

    def confirmar(self):
        start, end = None, None
        try:
            if self.t_inicio.get().strip(): start = float(self.t_inicio.get())
            if self.t_fin.get().strip(): end = float(self.t_fin.get())
        except ValueError: pass
        
        self.resultado = {
            "notch": self.usar_notch.get(),
            "bandpass": self.usar_bandpass.get(),
            "tipo_env": self.tipo_envolvente.get(),
            "start_time": start,
            "end_time": end
        }
        self.master.destroy()

# --- 3. FUNCIONES DE PROCESAMIENTO ---

def calcular_rms(senal, fs, window_ms):
    window_samples = int(fs * (window_ms / 1000.0))
    if window_samples < 1: window_samples = 1
    s = pd.Series(senal)
    rms = s.pow(2).rolling(window=window_samples, center=True).mean().apply(np.sqrt)
    return rms.fillna(0).values

def plotear_medicion_secuencial(nombre_medicion, config):
    """
    Procesa, GUARDA en la carpeta origen y muestra la gráfica.
    """
    print(f"\n>>> Procesando: {nombre_medicion}...")
    
    aplicar_notch = config["notch"]
    aplicar_pasabanda = config["bandpass"]
    tipo_envolvente = config["tipo_env"]
    start_time = config["start_time"]
    end_time = config["end_time"]

    # 1. Cargar CSV
    path_medicion = os.path.join(BASE_DIR, nombre_medicion)
    archivo_csv = next((os.path.join(path_medicion, f) for f in os.listdir(path_medicion) if f.lower().endswith('.csv')), None)
    
    if not archivo_csv:
        print(f"❌ Saltando {nombre_medicion}: No hay CSV.")
        return

    try:
        df = pd.read_csv(archivo_csv)
    except Exception as e:
        print(f"❌ Error leyendo CSV: {e}")
        return

    # 2. Filtrar Tiempo
    col_tiempo = df.columns[0]
    rango_str = "Completa"
    if start_time is not None and end_time is not None:
        if start_time < end_time:
            df = df[(df[col_tiempo] >= start_time) & (df[col_tiempo] <= end_time)]
            rango_str = f"{start_time}s - {end_time}s"

    if df.empty: return

    cols_canales = [col for col in df.columns[1:] if col.strip() in FACTORES_G]
    if not cols_canales: return

    # Fs
    try: fs = 1 / (df[col_tiempo].iloc[1] - df[col_tiempo].iloc[0])
    except: fs = 1000.0

    # Metadata
    bpm, noise_seconds = None, None
    try:
        for item in os.listdir(path_medicion):
            if os.path.isdir(os.path.join(path_medicion, item)) and item.startswith("canal_"):
                meta_path = os.path.join(path_medicion, item, 'metadata.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        md = json.load(f)
                        bpm, noise_seconds = md.get('bpm'), md.get('noise_seconds')
                    break
    except: pass

    # 3. Graficar
    num_canales = len(cols_canales)
    fig, axs = plt.subplots(num_canales, 1, figsize=(16, 5 * num_canales), sharex=True, squeeze=False)
    
    #fig.suptitle(f"Medición: {nombre_medicion}\n({rango_str})", fontsize=18, fontweight='bold')
    colores = plt.cm.viridis(np.linspace(0, 1, num_canales))

    for i, nombre_canal in enumerate(cols_canales):
        ax = axs[i, 0]
        nom_limpio = nombre_canal.strip()
        
        raw = df[nombre_canal].values
        ganancia = FACTORES_G.get(nom_limpio, 1.0)
        sig = (raw / ganancia) * 1e6 

        # --- NUEVO: Restar offset DC antes de filtrar (solo para modos con envolvente) ---
        dc_offset_removido = False
        if tipo_envolvente in ['hilbert', 'rms']:
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        dc_offset = np.mean(sig[:noise_end_idx])
                        sig = sig - dc_offset
                        dc_offset_removido = True

        info_filtros = []
        if aplicar_notch:
            b, a = signal.iirnotch(FREQ_NOTCH, Q_FACTOR_NOTCH, fs)
            sig = signal.filtfilt(b, a, sig)
            info_filtros.append("Notch")
        
        if aplicar_pasabanda:
            nyq = 0.5 * fs
            low, high = FREQ_PASABANDA[0]/nyq, min(FREQ_PASABANDA[1]/nyq, 0.99)
            b, a = signal.butter(ORDEN_PASABANDA, [low, high], btype='band')
            sig = signal.filtfilt(b, a, sig)
            info_filtros.append("BP")

        etiqueta_env = ""
        if tipo_envolvente == 'hilbert':
            env = np.abs(signal.hilbert(sig))
            etiqueta_env = " | Env. Hilbert"

            # Restar ruido si está disponible
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.mean(env[:noise_end_idx])
                        env = np.maximum(0, env - noise_level)
                        etiqueta_env += " (ruido restado)"
            
            ax.plot(df[col_tiempo], env, color=colores[i], lw=1.2)

        elif tipo_envolvente == 'rms':
            # Se usa un colormap distinto para la señal y su envolvente RMS para mayor claridad
            colores_rms = plt.cm.tab10(np.linspace(0, 1, 10))
            color_actual = colores_rms[i % 10]

            ax.plot(df[col_tiempo], sig, color=color_actual, alpha=0.4, lw=1, label='Señal Cruda')
            env_rms = calcular_rms(sig, fs, RMS_WINDOW_MS)
            etiqueta_env = " | Env. RMS"

            # Restar ruido si está disponible
            if noise_seconds is not None and noise_seconds > 0:
                tiempo_actual = df[col_tiempo].values
                if len(tiempo_actual) > 0 and tiempo_actual[0] < noise_seconds:
                    noise_end_idx = np.searchsorted(tiempo_actual, noise_seconds, side='right')
                    if noise_end_idx > 0:
                        noise_level = np.nanmean(env_rms[:noise_end_idx])
                        if not np.isnan(noise_level):
                            env_rms = np.maximum(0, env_rms - noise_level)
                            etiqueta_env += " (ruido restado)"

            ax.plot(df[col_tiempo], env_rms, color=color_actual, lw=1.5, label='RMS')
            ax.legend(loc='upper right', fontsize=20)
            max_rms = np.nanmax(env_rms)
            if max_rms > 0: ax.set_ylim(-5, max_rms * 2)
        else:
            ax.plot(df[col_tiempo], sig, color=colores[i], lw=0.8)
        
        if bpm and noise_seconds is not None:
            tau = 60.0/bpm
            t_max = df[col_tiempo].iloc[-1]
            k = 0
            while True:
               #la ventana esta entre tau y menos tau sobre 2
                line_t = noise_seconds + k*tau + tau/2
                if line_t > t_max: break
                if line_t >= df[col_tiempo].iloc[0]:
                    ax.axvline(x=line_t, color='black', ls='--', lw=1, alpha=0.4)
                k += 1

        tit = NOMBRES_CANALES_MAP.get(nom_limpio, nom_limpio)
        if info_filtros: tit += f" | {', '.join(info_filtros)}"
        tit += etiqueta_env
        ax.set_title(tit, fontsize=25)
        ax.set_ylabel("Amplitud (µV)", fontsize=27)
        ax.grid(True, alpha=0.5, ls='--')
        ax.tick_params(axis='both', which='major', labelsize=20)

    axs[-1, 0].set_xlabel("Tiempo (s)", fontsize=27)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # --- GUARDADO AUTOMÁTICO (SOLICITUD DE USUARIO) ---
    # Formato: plot_calibrado_{nombre_medicion}.png
    # Ubicación: Dentro de la misma carpeta de la medición
    nombre_archivo = f"plot_calibrado_{nombre_medicion}.png"
    ruta_guardado = os.path.join(path_medicion, nombre_archivo)
    
    plt.savefig(ruta_guardado, dpi=100)
    print(f"✅ Guardado en: {ruta_guardado}")
    
    # --- VISUALIZACIÓN BLOQUEANTE ---
    print(f"👁️ Visualizando {nombre_medicion}. Cierra la ventana del gráfico para continuar...")
    plt.show()      
    plt.close('all') 
    print(f"⏭️ Pasando a la siguiente...\n")

def flujo_principal():
    root = Tk()
    root.withdraw() 
    
    # 1. Selección
    ventana_sel = Toplevel(root)
    app_sel = VentanaSeleccion(ventana_sel, BASE_DIR)
    root.wait_window(ventana_sel)
    
    mediciones = app_sel.seleccionadas
    if not mediciones:
        print("No se seleccionaron mediciones.")
        root.destroy()
        return

    # 2. Configuración
    ventana_conf = Toplevel(root)
    app_conf = VentanaConfiguracion(ventana_conf)
    root.wait_window(ventana_conf)
    
    config = app_conf.resultado
    root.destroy() 

    if not config: return

    # 3. Bucle Secuencial
    total = len(mediciones)
    print(f"--- Iniciando secuencia de {total} mediciones ---")
    
    for i, nombre_medicion in enumerate(mediciones):
        print(f"[{i+1}/{total}] Cargando datos...")
        plotear_medicion_secuencial(nombre_medicion, config)

    print("--- Todas las mediciones procesadas ---")

if __name__ == "__main__":
    flujo_principal()