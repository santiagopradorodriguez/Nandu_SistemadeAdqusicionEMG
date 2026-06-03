# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Interfaz de ventana para mostrar palabras clave durante la adquisición.
# ==============================================================================

import sys
import tkinter as tk
from tkinter import font

def main():
    target_word = "ESPERANDO..."
    for arg in sys.argv:
        if arg.startswith("--word="):
            target_word = arg.split("=")[1]
            
    root = tk.Tk()
    root.title("Ñandú LSD - AutoForge - Palabra Actual")
    
    # Obtener resolución para anclar a la derecha debajo del metrónomo
    screen_w = root.winfo_screenwidth()
    window_w = 600
    window_h = 250
    x_pos = screen_w - window_w - 20 # 20px de margen derecho
    y_pos = 480 # Debajo del metrónomo (que mide 400 + 50 de margen superior = 450)
    root.geometry(f"{window_w}x{window_h}+{x_pos}+{y_pos}")
    
    root.configure(bg="#050505")
    
    # Mantener siempre al frente
    root.attributes("-topmost", True)
    
    # Fuente reducida a la mitad
    word_font = font.Font(family="Helvetica", size=45, weight="bold")
    
    label = tk.Label(root, text=target_word.upper(), font=word_font, fg="#00FFFF", bg="#050505")
    label.pack(expand=True, fill="both")
    
    root.mainloop()

if __name__ == "__main__":
    main()
