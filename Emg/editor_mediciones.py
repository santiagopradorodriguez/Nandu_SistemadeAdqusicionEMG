# -*- coding: utf-8 -*-
"""
editor_mediciones.py - v1.0

Herramienta para renombrar mediciones y actualizar sus metadatos.

Permite seleccionar una medición existente de la base de datos,
cambiar su nombre de formato "prueba" a "formal" (o editar uno ya formal),
y aplica los cambios tanto al nombre de la carpeta como a los
archivos 'metadata.json' internos de cada canal.
"""
import os
import json
import tkinter as tk
from tkinter import ttk, messagebox, font

class MeasurementEditorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Mediciones v1.0")
        self.root.geometry("700x450")
        self.root.configure(bg="#f0f0f0")

        self.BASE_DIR = "base_de_datos_electrodos"
        self.selected_measurement = None

        # --- Estilos ---
        self.font_bold = font.Font(family="Helvetica", size=10, weight="bold")
        self.font_normal = font.Font(family="Helvetica", size=10)

        # --- Layout Principal ---
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # --- Panel Izquierdo: Lista de Mediciones ---
        list_frame = ttk.LabelFrame(main_frame, text="1. Seleccionar Medición", padding="10")
        list_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.listbox = tk.Listbox(list_frame, exportselection=False, font=self.font_normal)
        self.listbox.pack(side="left", fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        self.listbox.bind("<<ListboxSelect>>", self.on_selection_change)

        # --- Panel Derecho: Editor de Detalles ---
        editor_frame = ttk.LabelFrame(main_frame, text="2. Editar Detalles", padding="15")
        editor_frame.pack(side="left", fill="y", ipadx=10)

        self.lbl_current_name = ttk.Label(editor_frame, text="Nombre Actual: (ninguno)", font=self.font_bold, wraplength=250)
        self.lbl_current_name.pack(anchor="w", pady=(0, 15))

        form_frame = ttk.Frame(editor_frame)
        form_frame.pack(fill="x")

        ttk.Label(form_frame, text="Letra:", font=self.font_normal).grid(row=0, column=0, sticky="w", pady=5)
        self.var_letra = tk.StringVar()
        self.entry_letra = ttk.Entry(form_frame, textvariable=self.var_letra, width=30, font=self.font_normal)
        self.entry_letra.grid(row=0, column=1, sticky="ew", pady=5)

        ttk.Label(form_frame, text="Prueba:", font=self.font_normal).grid(row=1, column=0, sticky="w", pady=5)
        self.var_prueba = tk.StringVar()
        self.entry_prueba = ttk.Entry(form_frame, textvariable=self.var_prueba, width=30, font=self.font_normal)
        self.entry_prueba.grid(row=1, column=1, sticky="ew", pady=5)

        ttk.Label(form_frame, text="Sujeto:", font=self.font_normal).grid(row=2, column=0, sticky="w", pady=5)
        self.var_sujeto = tk.StringVar()
        self.entry_sujeto = ttk.Entry(form_frame, textvariable=self.var_sujeto, width=30, font=self.font_normal)
        self.entry_sujeto.grid(row=2, column=1, sticky="ew", pady=5)

        # --- Botones de Acción ---
        action_frame = ttk.Frame(editor_frame)
        action_frame.pack(fill="x", pady=(20, 0), side="bottom")

        self.btn_save = ttk.Button(action_frame, text="Guardar Cambios", command=self.save_changes, state="disabled")
        self.btn_save.pack(fill="x", ipady=5, pady=(0, 5))

        self.btn_refresh = ttk.Button(action_frame, text="Refrescar Lista", command=self.load_measurements)
        self.btn_refresh.pack(fill="x", ipady=5)

        # Carga inicial
        self.load_measurements()
        self.set_editor_state("disabled")

    def set_editor_state(self, state):
        """Habilita o deshabilita los campos de edición."""
        self.entry_letra.config(state=state)
        self.entry_prueba.config(state=state)
        self.entry_sujeto.config(state=state)
        self.btn_save.config(state=state)

    def load_measurements(self):
        """Escanea la carpeta base y carga los nombres de las mediciones en la lista."""
        self.listbox.delete(0, tk.END)
        try:
            if os.path.isdir(self.BASE_DIR):
                for item in sorted(os.listdir(self.BASE_DIR)):
                    if os.path.isdir(os.path.join(self.BASE_DIR, item)):
                        self.listbox.insert(tk.END, item)
            else:
                messagebox.showerror("Error", f"El directorio base '{self.BASE_DIR}' no existe.")
        except Exception as e:
            messagebox.showerror("Error de Lectura", f"No se pudo leer el directorio de mediciones.\nError: {e}")

    def on_selection_change(self, event=None):
        """Se activa al seleccionar un ítem. Carga sus datos en el editor."""
        selection_indices = self.listbox.curselection()
        if not selection_indices:
            self.selected_measurement = None
            self.set_editor_state("disabled")
            return

        self.selected_measurement = self.listbox.get(selection_indices[0])
        self.lbl_current_name.config(text=f"Nombre Actual: {self.selected_measurement}")

        # --- MEJORA: Rellenar automáticamente los campos al seleccionar ---
        # Primero, intentar parsear el nombre de la carpeta seleccionada.
        parts = self.selected_measurement.split('_')
        if len(parts) >= 3:
            # Asumir formato Letra_Prueba_Sujeto
            self.var_letra.set(parts[0])
            self.var_prueba.set(parts[1])
            self.var_sujeto.set("_".join(parts[2:])) # Unir el resto por si el sujeto tiene guiones bajos
        else:
            # Si no es formato formal, intentar leer el metadata.json
            measurement_path = os.path.join(self.BASE_DIR, self.selected_measurement)
            first_channel_path = None
            for item in sorted(os.listdir(measurement_path)):
                if item.startswith("canal_"):
                    first_channel_path = os.path.join(measurement_path, item)
                    break
            
            if first_channel_path and os.path.exists(os.path.join(first_channel_path, "metadata.json")):
                meta_path = os.path.join(first_channel_path, "metadata.json")
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                self.var_letra.set(meta.get("letra", "A"))
                # Si es una medición de prueba, usar el nombre de la carpeta como base para el campo "Prueba"
                self.var_prueba.set(self.selected_measurement if not meta.get("is_formal") else meta.get("prueba", "Prueba1"))
                self.var_sujeto.set(meta.get("sujeto", "Sujeto1"))
            else:
                # Fallback si no hay JSON: rellenar con valores por defecto y el nombre de la carpeta
                self.var_letra.set("A")
                self.var_prueba.set(self.selected_measurement)
                self.var_sujeto.set("Sujeto1")
        
        self.set_editor_state("normal")

    def save_changes(self):
        """Renombra la carpeta y actualiza todos los archivos JSON internos."""
        if not self.selected_measurement:
            return

        # 1. Obtener los nuevos valores y construir el nuevo nombre
        new_letra = self.var_letra.get().strip()
        new_prueba = self.var_prueba.get().strip()
        new_sujeto = self.var_sujeto.get().strip()

        if not all([new_letra, new_prueba, new_sujeto]):
            messagebox.showerror("Error", "Todos los campos (Sujeto, Letra, Prueba) son obligatorios.")
            return

        new_folder_name = f"{new_letra}_{new_prueba}_{new_sujeto}"
        old_path = os.path.join(self.BASE_DIR, self.selected_measurement)
        new_path = os.path.join(self.BASE_DIR, new_folder_name)

        if old_path == new_path:
            messagebox.showinfo("Información", "El nombre no ha cambiado. No se realizaron acciones.")
            return

        if os.path.exists(new_path):
            messagebox.showerror("Error", f"Ya existe una carpeta con el nombre '{new_folder_name}'.\nPor favor, elige un nombre único.")
            return

        # 2. Confirmación del usuario (¡muy importante!)
        if not messagebox.askyesno("Confirmar Cambios",
            f"¿Estás seguro de que quieres renombrar:\n\n"
            f"'{self.selected_measurement}'\n\na\n\n"
            f"'{new_folder_name}'?\n\n"
            "Esta acción modificará la carpeta y sus archivos internos."):
            return

        try:
            # 3. Renombrar la carpeta principal
            os.rename(old_path, new_path)
            print(f"Carpeta renombrada a: '{new_path}'")

            # 4. Iterar sobre las subcarpetas de canal y actualizar los JSON
            for item in sorted(os.listdir(new_path)):
                channel_path = os.path.join(new_path, item)
                if os.path.isdir(channel_path) and item.startswith("canal_"):
                    meta_path = os.path.join(channel_path, "metadata.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r+', encoding='utf-8') as f:
                                meta = json.load(f)
                                # Actualizar los campos
                                meta['letra'] = new_letra
                                meta['prueba'] = new_prueba
                                meta['sujeto'] = new_sujeto
                                meta['is_formal'] = True
                                # Volver al inicio del archivo para sobreescribir
                                f.seek(0)
                                json.dump(meta, f, indent=4)
                                f.truncate() # Cortar el archivo si el nuevo contenido es más corto
                            print(f"  -> Actualizado: {meta_path}")
                        except Exception as e:
                            print(f"  -> ADVERTENCIA: No se pudo actualizar '{meta_path}'. Error: {e}")
            
            messagebox.showinfo("Éxito", "La medición ha sido renombrada y actualizada correctamente.")

        except Exception as e:
            messagebox.showerror("Error Crítico", f"Ocurrió un error durante el proceso de renombrado.\n\nError: {e}")
            # Intentar revertir si es posible (puede fallar si el error fue a mitad de camino)
            if not os.path.exists(old_path) and os.path.exists(new_path):
                os.rename(new_path, old_path)
        finally:
            # 5. Refrescar la lista para mostrar el cambio
            self.load_measurements()
            self.set_editor_state("disabled")
            self.lbl_current_name.config(text="Nombre Actual: (ninguno)")

if __name__ == "__main__":
    root = tk.Tk()
    app = MeasurementEditorApp(root)
    root.mainloop()