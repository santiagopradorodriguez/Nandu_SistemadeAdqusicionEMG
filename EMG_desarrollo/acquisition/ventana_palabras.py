# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Interfaz de ventana para mostrar palabras clave durante la adquisición.
# ==============================================================================

import sys
import tkinter as tk
from tkinter import font
import threading

def main():
    target_word = "ESPERANDO..."
    x_arg = None
    y_arg = None
    w_arg = None
    h_arg = None
    
    for arg in sys.argv:
        if arg.startswith("--word="):
            target_word = arg.split("=")[1]
            target_word = target_word.replace("\\n", "\n")
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
    root.title("Ñandú LSD - AutoForge - Palabra Actual")
    
    # Obtener resolución para anclar a la derecha debajo del metrónomo
    screen_w = root.winfo_screenwidth()
    window_w = w_arg if w_arg is not None else 600
    window_h = h_arg if h_arg is not None else 250
    x_pos = x_arg if x_arg is not None else screen_w - window_w - 20 # 20px de margen derecho
    y_pos = y_arg if y_arg is not None else 480 # Debajo del metrónomo
    root.geometry(f"{window_w}x{window_h}+{x_pos}+{y_pos}")
    
    root.configure(bg="#050505")
    
    # Mantener siempre al frente
    root.attributes("-topmost", True)
    
    # Fuente reducida a la mitad
    word_font = font.Font(family="Helvetica", size=45, weight="bold")
    
    label = tk.Label(root, text=target_word.upper(), font=word_font, fg="#00FFFF", bg="#050505", justify="center", anchor="center")
    label.pack(expand=True, fill="both")
    
    def listen_for_commands():
        for line in sys.stdin:
            line = line.strip()
            if line.startswith("RESIZE"):
                try:
                    parts = line.split()
                    w, h, x, y = parts[1], parts[2], parts[3], parts[4]
                    root.after(0, lambda w_=w, h_=h, x_=x, y_=y: root.geometry(f"{w_}x{h_}+{x_}+{y_}"))
                except Exception as e:
                    pass
            elif line:
                word = line.replace("\\n", "\n")
                root.after(0, lambda w=word: label.config(text=w.upper()))
                
    t = threading.Thread(target=listen_for_commands, daemon=True)
    t.start()
    
    root.mainloop()

if __name__ == "__main__":
    main()
