# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Herramienta de utilidad para editar detalles de mediciones existentes.
# ==============================================================================

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
import sys
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QListWidget, QLabel, QLineEdit, QPushButton, 
    QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt

class MeasurementEditorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Editor de Mediciones v2.0 (PySide6)")
        self.resize(700, 450)
        self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")

        _current_dir = os.path.dirname(os.path.abspath(__file__))
        self.BASE_DIR = os.path.join(os.path.dirname(_current_dir), "base_de_datos_electrodos")
        self.selected_measurement = None

        # --- Layout Principal ---
        main_layout = QHBoxLayout(self)

        # --- Panel Izquierdo: Lista de Mediciones ---
        list_group = QGroupBox("1. Seleccionar Medición")
        list_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        list_layout = QVBoxLayout(list_group)
        
        self.listbox = QListWidget()
        self.listbox.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #333;")
        self.listbox.itemSelectionChanged.connect(self.on_selection_change)
        list_layout.addWidget(self.listbox)
        main_layout.addWidget(list_group, stretch=1)

        # --- Panel Derecho: Editor de Detalles ---
        editor_group = QGroupBox("2. Editar Detalles")
        editor_group.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        editor_layout = QVBoxLayout(editor_group)

        self.lbl_current_name = QLabel("Nombre Actual: (ninguno)")
        self.lbl_current_name.setStyleSheet("font-weight: bold; color: #ff003c;")
        self.lbl_current_name.setWordWrap(True)
        editor_layout.addWidget(self.lbl_current_name)

        form_layout = QFormLayout()
        
        self.entry_letra = QLineEdit()
        self.entry_letra.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444; padding: 5px;")
        form_layout.addRow("Letra:", self.entry_letra)
        
        self.entry_prueba = QLineEdit()
        self.entry_prueba.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444; padding: 5px;")
        form_layout.addRow("Prueba:", self.entry_prueba)
        
        self.entry_sujeto = QLineEdit()
        self.entry_sujeto.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #444; padding: 5px;")
        form_layout.addRow("Sujeto:", self.entry_sujeto)
        
        editor_layout.addLayout(form_layout)
        editor_layout.addStretch()

        # --- Botones de Acción ---
        self.btn_save = QPushButton("Guardar Cambios")
        self.btn_save.setStyleSheet("QPushButton { background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px; border-radius: 3px; } QPushButton:disabled { background-color: #333; color: #666; }")
        self.btn_save.clicked.connect(self.save_changes)
        editor_layout.addWidget(self.btn_save)

        self.btn_refresh = QPushButton("Refrescar Lista")
        self.btn_refresh.setStyleSheet("QPushButton { background-color: #222; color: #fff; padding: 10px; border: 1px solid #00ffcc; border-radius: 3px; }")
        self.btn_refresh.clicked.connect(self.load_measurements)
        editor_layout.addWidget(self.btn_refresh)

        main_layout.addWidget(editor_group, stretch=1)

        # Carga inicial
        self.load_measurements()
        self.set_editor_state(False)

    def set_editor_state(self, state):
        self.entry_letra.setEnabled(state)
        self.entry_prueba.setEnabled(state)
        self.entry_sujeto.setEnabled(state)
        self.btn_save.setEnabled(state)

    def load_measurements(self):
        self.listbox.clear()
        try:
            if os.path.isdir(self.BASE_DIR):
                for fecha in sorted(os.listdir(self.BASE_DIR)):
                    fecha_path = os.path.join(self.BASE_DIR, fecha)
                    if os.path.isdir(fecha_path):
                        for medicion in sorted(os.listdir(fecha_path)):
                            medicion_path = os.path.join(fecha_path, medicion)
                            if os.path.isdir(medicion_path):
                                # Guardamos el path relativo usando slash común
                                self.listbox.addItem(f"{fecha}/{medicion}")
            else:
                QMessageBox.critical(self, "Error", f"El directorio base '{self.BASE_DIR}' no existe.")
        except Exception as e:
            QMessageBox.critical(self, "Error de Lectura", f"No se pudo leer el directorio de mediciones.\nError: {e}")

    def on_selection_change(self):
        selected_items = self.listbox.selectedItems()
        if not selected_items:
            self.selected_measurement = None
            self.set_editor_state(False)
            return

        self.selected_measurement = selected_items[0].text()
        
        # El nombre puro de la medición es la última parte
        medicion_name = self.selected_measurement.split('/')[-1]
        self.lbl_current_name.setText(f"Nombre Actual: {medicion_name}")

        parts = medicion_name.split('_')
        if len(parts) >= 3:
            self.entry_letra.setText(parts[0])
            self.entry_prueba.setText(parts[1])
            self.entry_sujeto.setText("_".join(parts[2:]))
        else:
            measurement_path = os.path.join(self.BASE_DIR, self.selected_measurement)
            first_channel_path = None
            if os.path.exists(measurement_path):
                for item in sorted(os.listdir(measurement_path)):
                    if item.startswith("canal_"):
                        first_channel_path = os.path.join(measurement_path, item)
                        break
            
            meta_loaded = False
            if first_channel_path and os.path.exists(os.path.join(first_channel_path, "metadata.json")):
                meta_path = os.path.join(first_channel_path, "metadata.json")
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    self.entry_letra.setText(meta.get("letra", "A"))
                    self.entry_prueba.setText(medicion_name if not meta.get("is_formal") else meta.get("prueba", "Prueba1"))
                    self.entry_sujeto.setText(meta.get("sujeto", "Sujeto1"))
                    meta_loaded = True
                except:
                    pass
                    
            if not meta_loaded:
                self.entry_letra.setText("A")
                self.entry_prueba.setText(medicion_name)
                self.entry_sujeto.setText("Sujeto1")
        
        self.set_editor_state(True)

    def save_changes(self):
        if not self.selected_measurement:
            return

        new_letra = self.entry_letra.text().strip()
        new_prueba = self.entry_prueba.text().strip()
        new_sujeto = self.entry_sujeto.text().strip()

        if not all([new_letra, new_prueba, new_sujeto]):
            QMessageBox.critical(self, "Error", "Todos los campos (Sujeto, Letra, Prueba) son obligatorios.")
            return

        new_folder_name = f"{new_letra}_{new_prueba}_{new_sujeto}"
        old_path = os.path.join(self.BASE_DIR, self.selected_measurement)
        
        fecha_name = self.selected_measurement.split('/')[0]
        new_path = os.path.join(self.BASE_DIR, fecha_name, new_folder_name)

        if old_path == new_path:
            QMessageBox.information(self, "Información", "El nombre no ha cambiado. No se realizaron acciones.")
            return

        if os.path.exists(new_path):
            QMessageBox.critical(self, "Error", f"Ya existe una carpeta con el nombre '{new_folder_name}'.\nPor favor, elige un nombre único.")
            return

        reply = QMessageBox.question(
            self, "Confirmar Cambios",
            f"¿Estás seguro de que quieres renombrar:\n\n'{self.selected_measurement}'\n\na\n\n'{fecha_name}/{new_folder_name}'?\n\nEsta acción modificará la carpeta y sus archivos internos.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.No:
            return

        try:
            os.rename(old_path, new_path)
            for item in sorted(os.listdir(new_path)):
                channel_path = os.path.join(new_path, item)
                if os.path.isdir(channel_path) and item.startswith("canal_"):
                    meta_path = os.path.join(channel_path, "metadata.json")
                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path, 'r+', encoding='utf-8') as f:
                                meta = json.load(f)
                                meta['letra'] = new_letra
                                meta['prueba'] = new_prueba
                                meta['sujeto'] = new_sujeto
                                meta['is_formal'] = True
                                f.seek(0)
                                json.dump(meta, f, indent=4)
                                f.truncate()
                        except Exception as e:
                            print(f"ADVERTENCIA: No se pudo actualizar '{meta_path}'. Error: {e}")
            
            QMessageBox.information(self, "Éxito", "La medición ha sido renombrada y actualizada correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error Crítico", f"Ocurrió un error durante el proceso de renombrado.\n\nError: {e}")
            if not os.path.exists(old_path) and os.path.exists(new_path):
                os.rename(new_path, old_path)
        finally:
            self.load_measurements()
            self.set_editor_state(False)
            self.lbl_current_name.setText("Nombre Actual: (ninguno)")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = MeasurementEditorDialog()
    dialog.exec()