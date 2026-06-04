# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Metrónomo visual y sonoro para guiar las pruebas de adquisición.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
# Esta es la última versión funcional conocida.
Metrónomo Visual y Sonoro v1.0

Una herramienta simple que proporciona una señal visual (parpadeo) y auditiva (beep)
a un ritmo configurable en BPM (Beats Per Minute).

Se puede lanzar desde el panel de control principal.
"""
import tkinter as tk
import threading
import sys
import json
import os
from tkinter import font

# --- NUEVO: Import para sonido de metrónomo ---
try:
    import winsound
except ImportError:
    winsound = None # winsound solo está disponible en Windows

class MetronomeApp:
    def __init__(self, root, start_x=None, start_y=None, start_w=None, start_h=None):
        self.root = root
        self.root.title("Ñandú LSD - Metrónomo Cyberpunk")
        
        # Obtener resolución de pantalla para anclar a la derecha
        screen_w = self.root.winfo_screenwidth()
        window_w = start_w if start_w is not None else 230
        window_h = start_h if start_h is not None else 380
        x_pos = start_x if start_x is not None else screen_w - window_w - 20 # 20px de margen derecho
        y_pos = start_y if start_y is not None else 50 # 50px de margen superior
        self.root.geometry(f"{window_w}x{window_h}+{x_pos}+{y_pos}")
        
        self.root.configure(bg="#050505")
        self.root.resizable(False, False)
        
        # --- NUEVO: Mantener la ventana del metrónomo siempre al frente ---
        self.root.attributes("-topmost", True)

        # Asegura que save_config se llama al cerrar la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # --- Estado del metrónomo ---
        self.is_running = False
        self.is_counting = False # <-- NUEVO: Controla si se debe contar o no
        self.is_muted = False # <-- NUEVO: Controla si el beep suena o no
        # --- MODIFICADO: Cargar BPM desde el archivo de configuración ---
        self.bpm = tk.IntVar(value=60)
        self.subdivisions = tk.IntVar(value=4) # <-- NUEVO: Para sub-pulsos
        self.timer_id = None
        # --- NUEVO: Variable para el contador de pulsos ---
        self.beat_count = tk.StringVar(value="0")

        # --- NUEVO: Hilo para escuchar comandos ---
        self.command_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self.command_thread.start()
        # --- NUEVO: Índice para el ciclo de sub-pulsos ---
        self.beat_cycle_index = 0
        # --- Colores para el pulso visual ---
        self.COLOR_BEAT = "#00FFFF"  # Cian Neón para el pulso
        self.COLOR_IDLE = "#111111"  # Negro profundo cuando está en reposo

        # --- Fuentes ---
        title_font = font.Font(family="Helvetica", size=11, weight="bold")
        value_font = font.Font(family="Helvetica", size=24, weight="bold")
        button_font = font.Font(family="Helvetica", size=9)
        # --- NUEVO: Fuente para el contador ---
        counter_font = font.Font(family="Helvetica", size=32, weight="bold")

        # --- Elemento visual para el pulso ---
        self.pulse_frame = tk.Frame(root, bg=self.COLOR_IDLE, height=60)
        self.pulse_frame.pack(fill="x", padx=20, pady=20)

        # --- NUEVO: Display del contador de pulsos ---
        pulse_container = tk.Frame(self.root, bg="#050505")
        pulse_container.pack(expand=True, fill="both")

        # Título dinámico "INICIANDO" / "PULSO"
        self.lbl_title_var = tk.StringVar(value="PULSO")
        self.lbl_title = tk.Label(pulse_container, textvariable=self.lbl_title_var, font=title_font, fg="#00FFFF", bg="#050505")
        self.lbl_title.pack(pady=(5, 5))

        # Contador de Pulsos gigante
        self.counter_label = tk.Label(pulse_container, textvariable=self.beat_count, font=counter_font, fg="#00FFFF", bg="#050505")
        self.counter_label.pack(pady=(0, 10))

        # --- Controles ---
        controls_frame = tk.Frame(root, bg="#050505")
        controls_frame.pack(fill="both", expand=True, padx=20)

        # Display de BPM
        tk.Label(controls_frame, text="BPM", font=title_font, fg="#00FF00", bg="#050505").pack()
        self.bpm_label = tk.Label(controls_frame, textvariable=self.bpm, font=value_font, fg="#00FF00", bg="#050505")
        self.bpm_label.pack(pady=(5, 15))

        # Slider para ajustar BPM
        self.bpm_slider = tk.Scale(
            controls_frame,
            from_=20,
            to=200,
            orient="horizontal",
            variable=self.bpm,
            showvalue=0,
            length=200,
            bg="#050505",
            fg="#00FF00",
            highlightthickness=0,
            troughcolor="#222222"
        )
        self.bpm_slider.pack(pady=10)

        # --- NUEVO: Control de Subdivisiones ---
        tk.Label(controls_frame, text="Subdivisiones por Pulso", font=title_font, fg="#00FF00", bg="#050505").pack()
        self.subdiv_spinbox = tk.Spinbox(
            controls_frame,
            from_=1,
            to=8,
            textvariable=self.subdivisions,
            width=5,
            font=button_font,
            bg="#111111",
            fg="#00FFFF",
            state="readonly"
        )
        self.subdiv_spinbox.pack(pady=5)

        # Botones de Start/Stop
        button_container = tk.Frame(controls_frame, bg="#050505")
        button_container.pack(pady=20)

        self.btn_start = tk.Button(button_container, text="INICIAR", command=self.start, font=button_font, width=10, bg="#AA0000", fg="white", activebackground="#FF0000")
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = tk.Button(button_container, text="DETENER", command=self.stop, font=button_font, width=10, bg="#550000", fg="white", state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        
        # --- NUEVO: Botón de Reset ---
        self.btn_reset = tk.Button(button_container, text="RESET", command=self.reset_counter, font=button_font, width=8, bg="#333333", fg="white")
        self.btn_reset.pack(side="left", padx=10)

        # --- NUEVO: Cargar la última configuración guardada ---
        self.load_config()

    def load_config(self):
        """Carga la última configuración guardada desde metronome_config.json."""
        if os.path.exists('metronome_config.json'):
            try:
                with open('metronome_config.json', 'r') as f:
                    config = json.load(f)
                    self.bpm.set(config.get('last_bpm', 60))
                    self.subdivisions.set(config.get('subdivisions', 4))
                    print(f"Configuración de metrónomo cargada: BPM={self.bpm.get()}")
            except Exception as e:
                print(f"Error al cargar config del metrónomo, usando valores por defecto. Error: {e}")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.beat_cycle_index = 0 # Iniciar siempre desde el pulso principal
        self.reset_counter() # Reinicia el contador al iniciar
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.bpm_slider.config(state="disabled")
        self.subdiv_spinbox.config(state="disabled")
        self.beat()

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self.timer_id:
            self.root.after_cancel(self.timer_id)
        self.pulse_frame.config(bg=self.COLOR_IDLE)
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.count_in_remaining = 0
        self.beat_count.set("0")
        self.beat_cycle_index = 0
        self.bpm_slider.config(state="normal")
        self.subdiv_spinbox.config(state="readonly")

    def _listen_for_commands(self):
        """Escucha comandos desde stdin en un hilo separado."""
        for line in sys.stdin:
            command = line.strip()
            if command == "START_COUNTING":
                print("[Metrónomo] Recibido comando START_COUNTING.")
                # Usamos after para asegurar que la GUI se actualice desde el hilo principal
                self.root.after(0, self.start_counting)
            elif command == "STOP_APP":
                print("[Metrónomo] Recibido comando STOP_APP.")
                self.root.after(0, self.on_closing)
            elif command == "MUTE":
                print("[Metrónomo] Recibido comando MUTE.")
                self.is_muted = True
            elif command == "UNMUTE":
                print("[Metrónomo] Recibido comando UNMUTE.")
                self.is_muted = False

    def start_counting(self):
        """Activa el contador y lo resetea."""
        print("[Metrónomo] Comando recibido. Iniciando conteo.")
        self.is_counting = True
        self.reset_counter()
        self.beat() # FORZAR EL BEEP INMEDIATAMENTE

    def save_config(self):
        """Guarda la configuración del metrónomo en un archivo JSON."""
        config = {
            "last_bpm": self.bpm.get(),
            "last_beat_count": self.beat_count.get(),
            "subdivisions": self.subdivisions.get()
        }
        try:
            with open('metronome_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                print(f"Configuración de metrónomo guardada: {config} (beat_count={self.beat_count.get()})")
        except Exception as e:
            print(f"Error al guardar config del metrónomo: {e}")
    def reset_counter(self):
        """Reinicia el contador de pulsos a cero."""
        self.beat_count.set("0")

    def on_closing(self):
        """Maneja el cierre de la ventana, guardando la configuración."""
        self.save_config()
        self.root.destroy()

    def beat(self):
        if not self.is_running:
            return # Si no está corriendo, no hace nada (ni cuenta ni suena)

        num_subdivs = self.subdivisions.get()
        is_main_beat = (self.beat_cycle_index == 0)

        if is_main_beat:
            # --- LÓGICA DEL PULSO PRINCIPAL (EL "1") ---
            self.pulse_frame.config(bg=self.COLOR_BEAT)
            count_in = getattr(self, 'count_in_remaining', 0)
            
            # Actualización de la GUI y del número de conteo
            if count_in > 1:
                # Fase preparatoria (Rojo, cuenta regresiva)
                if hasattr(self, 'lbl_title_var'):
                    self.lbl_title_var.set("INICIANDO")
                    self.lbl_title.config(fg="#FF0000")
                if hasattr(self, 'counter_label'):
                    self.counter_label.config(fg="#FF0000")
                self.beat_count.set(str(count_in - 1))
                self.count_in_remaining -= 1
            else:
                # Fase normal o pulso de arranque (GO)
                if hasattr(self, 'lbl_title_var'):
                    self.lbl_title_var.set("PULSO")
                    self.lbl_title.config(fg="#00FFFF")
                if hasattr(self, 'counter_label'):
                    self.counter_label.config(fg="#00FFFF")
                
                if self.count_in_remaining > 0:
                    self.beat_count.set("1") # Muestra '1' en el pulso de GO
                    self.count_in_remaining = 0
                else:
                    try:
                        current = int(self.beat_count.get())
                    except ValueError:
                        current = 0
                    self.beat_count.set(str(current + 1))
                        
            # Sonidos de metrónomo (Graves para cuenta atrás, Agudo para GO, Normal el resto)
            if winsound and not self.is_muted:
                try:
                    if count_in > 1:
                        threading.Thread(target=winsound.Beep, args=(800, 200), daemon=True).start()
                    elif count_in == 1:
                        threading.Thread(target=winsound.Beep, args=(1200, 500), daemon=True).start()
                    else:
                        threading.Thread(target=winsound.Beep, args=(1000, 100), daemon=True).start()
                except Exception as e:
                    print(f"Error al reproducir sonido con winsound: {e}")

            self.root.after(50, lambda: self.pulse_frame.config(bg=self.COLOR_IDLE))

        elif getattr(self, 'count_in_remaining', 0) == 0:
            # --- LÓGICA DEL SUB-PULSO (EL "2, 3, 4...") ---
            self.pulse_frame.config(bg="#00AAAA") # Cian más oscuro
            self.root.after(50, lambda: self.pulse_frame.config(bg=self.COLOR_IDLE))
            if winsound and not self.is_muted:
                try:
                    threading.Thread(target=winsound.Beep, args=(1600, 50), daemon=True).start()
                except Exception as e:
                    print(f"Error al reproducir sonido de sub-beat: {e}")

        # --- LÓGICA DE TEMPORIZACIÓN (COMÚN A AMBOS) ---
        self.beat_cycle_index = (self.beat_cycle_index + 1) % num_subdivs
        interval_ms = int(60000 / self.bpm.get())
        sub_interval_ms = interval_ms // num_subdivs if num_subdivs > 1 else interval_ms
        self.timer_id = self.root.after(sub_interval_ms, self.beat)

def main():
    # --- NUEVO: Lógica para autostart ---
    autostart = '--autostart' in sys.argv
    start_muted = '--mute' in sys.argv
    
    target_word = ""
    bpm_arg = None
    count_in_arg = 0
    x_arg = None
    y_arg = None
    w_arg = None
    h_arg = None
    
    for arg in sys.argv:
        if arg.startswith("--word="):
            target_word = arg.split("=")[1]
        elif arg.startswith("--bpm="):
            try:
                bpm_arg = int(arg.split("=")[1])
            except ValueError:
                pass
        elif arg.startswith("--count-in="):
            try:
                count_in_arg = int(arg.split("=")[1])
            except ValueError:
                pass
        elif arg.startswith("--x="):
            try: x_arg = int(arg.split("=")[1])
            except ValueError: pass
        elif arg.startswith("--y="):
            try: y_arg = int(arg.split("=")[1])
            except ValueError: pass
        elif arg.startswith("--width="):
            try: w_arg = int(arg.split("=")[1])
            except ValueError: pass
        elif arg.startswith("--height="):
            try: h_arg = int(arg.split("=")[1])
            except ValueError: pass

    root = tk.Tk()
    app = MetronomeApp(root, start_x=x_arg, start_y=y_arg, start_w=w_arg, start_h=h_arg)
    app.count_in_remaining = count_in_arg
    
    if start_muted:
        app.is_muted = True
        print("Metrónomo iniciado en modo MUTE.")

    if target_word:
        pass # La palabra ahora se muestra en ventana_palabras.py independiente
        
    if bpm_arg is not None:
        app.bpm.set(bpm_arg)
        
    if autostart:
        print("Metrónomo iniciado con autostart.")
        app.is_counting = '--count' in sys.argv # Activa el conteo automáticamente si se pasa el flag
        app.start()

    root.mainloop()

if __name__ == "__main__":
    main()