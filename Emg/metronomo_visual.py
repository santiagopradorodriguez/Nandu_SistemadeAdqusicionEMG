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
    def __init__(self, root):
        self.root = root
        self.root.title("Metrónomo Visual")
        self.root.geometry("350x500") # Aumentar altura para el contador
        self.root.configure(bg="#2E2E2E")
        self.root.resizable(False, False)

        # Asegura que save_config se llama al cerrar la ventana
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # --- Estado del metrónomo ---
        self.is_running = False
        self.is_counting = False # <-- NUEVO: Controla si se debe contar o no
        # --- MODIFICADO: Cargar BPM desde el archivo de configuración ---
        self.bpm = tk.IntVar(value=60)
        self.timer_id = None
        # --- NUEVO: Variable para el contador de pulsos ---
        self.beat_count = tk.IntVar(value=0)

        # --- NUEVO: Hilo para escuchar comandos ---
        self.command_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        self.command_thread.start()
        # --- Colores para el pulso visual ---
        self.COLOR_BEAT = "#FFFFFF"  # Blanco para el pulso
        self.COLOR_IDLE = "#3C3C3C"  # Gris oscuro cuando está en reposo

        # --- Fuentes ---
        title_font = font.Font(family="Helvetica", size=14, weight="bold")
        value_font = font.Font(family="Helvetica", size=36, weight="bold")
        button_font = font.Font(family="Helvetica", size=12)
        # --- NUEVO: Fuente para el contador ---
        counter_font = font.Font(family="Helvetica", size=48, weight="bold")

        # --- Elemento visual para el pulso ---
        self.pulse_frame = tk.Frame(root, bg=self.COLOR_IDLE, height=100)
        self.pulse_frame.pack(fill="x", padx=20, pady=20)

        # --- Controles ---
        controls_frame = tk.Frame(root, bg="#2E2E2E")
        controls_frame.pack(fill="both", expand=True, padx=20)

        # --- NUEVO: Display del contador de pulsos ---
        tk.Label(controls_frame, text="Pulso Actual", font=title_font, fg="white", bg="#2E2E2E").pack()
        self.counter_label = tk.Label(controls_frame, textvariable=self.beat_count, font=counter_font, fg=self.COLOR_BEAT, bg="#2E2E2E")
        self.counter_label.pack(pady=(5, 15))


        # Display de BPM
        tk.Label(controls_frame, text="BPM", font=title_font, fg="white", bg="#2E2E2E").pack()
        self.bpm_label = tk.Label(controls_frame, textvariable=self.bpm, font=value_font, fg="white", bg="#2E2E2E")
        self.bpm_label.pack(pady=(5, 15))

        # Slider para ajustar BPM
        self.bpm_slider = tk.Scale(
            controls_frame,
            from_=20,
            to=200,
            orient="horizontal",
            variable=self.bpm,
            showvalue=0,
            length=300,
            bg="#2E2E2E",
            fg="white",
            highlightthickness=0,
            troughcolor="#555555"
        )
        self.bpm_slider.pack(pady=10)

        # Botones de Start/Stop
        button_container = tk.Frame(controls_frame, bg="#2E2E2E")
        button_container.pack(pady=20)

        self.btn_start = tk.Button(button_container, text="Iniciar", command=self.start, font=button_font, width=10, bg="#28A745", fg="white")
        self.btn_start.pack(side="left", padx=10)

        self.btn_stop = tk.Button(button_container, text="Detener", command=self.stop, font=button_font, width=10, bg="#DC3545", fg="white", state="disabled")
        self.btn_stop.pack(side="left", padx=10)
        
        # --- NUEVO: Botón de Reset ---
        self.btn_reset = tk.Button(button_container, text="Reset", command=self.reset_counter, font=button_font, width=8, bg="#6C757D", fg="white")
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
                    print(f"Configuración de metrónomo cargada: BPM={self.bpm.get()}")
            except Exception as e:
                print(f"Error al cargar config del metrónomo, usando valores por defecto. Error: {e}")

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self.reset_counter() # Reinicia el contador al iniciar
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.bpm_slider.config(state="disabled")
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
        self.bpm_slider.config(state="normal")

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

    def start_counting(self):
        """Activa el contador y lo resetea."""
        print("[Metrónomo] Comando recibido. Iniciando conteo.")
        self.is_counting = True
        self.reset_counter()

    def save_config(self):
        """Guarda la configuración del metrónomo en un archivo JSON."""
        config = {
            "last_bpm": self.bpm.get(),
            "last_beat_count": self.beat_count.get()
        }
        try:
            with open('metronome_config.json', 'w') as f:
                json.dump(config, f, indent=4)
                print(f"Configuración de metrónomo guardada: {config} (beat_count={self.beat_count.get()})")
        except Exception as e:
            print(f"Error al guardar config del metrónomo: {e}")
    def reset_counter(self):
        """Reinicia el contador de pulsos a cero."""
        self.beat_count.set(0)

    def on_closing(self):
        """Maneja el cierre de la ventana, guardando la configuración."""
        self.save_config()
        self.root.destroy()

    def beat(self):
        if not self.is_running:
            return # Si no está corriendo, no hace nada (ni cuenta ni suena)

        # Pulso visual y sonoro
        self.pulse_frame.config(bg=self.COLOR_BEAT)
        if self.is_counting: # <-- SOLO cuenta si está habilitado
            self.beat_count.set(self.beat_count.get() + 1) # Incrementar contador
        if winsound:
            try:
                winsound.Beep(1000, 50) # Frecuencia de 1000 Hz, duración de 50 ms
            except Exception as e:
                print(f"Error al reproducir sonido con winsound: {e}")

        # Apagar el pulso visual después de un corto tiempo
        self.root.after(50, lambda: self.pulse_frame.config(bg=self.COLOR_IDLE))

        # Programar el siguiente pulso
        interval_ms = int(60000 / self.bpm.get())
        self.timer_id = self.root.after(interval_ms, self.beat)

if __name__ == "__main__":
    # --- NUEVO: Lógica para autostart ---
    autostart = '--autostart' in sys.argv

    root = tk.Tk()
    app = MetronomeApp(root)
    if autostart:
        print("Metrónomo iniciado con autostart.")
        # --- MODIFICADO: Con autostart, solo empieza a sonar, no a contar ---
        app.is_running = True
        app.reset_counter()  # Asegura que el contador empieza en 0
        app.beat()

    root.mainloop()