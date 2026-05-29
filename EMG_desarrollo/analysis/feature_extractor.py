# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Extracción de características y pulsos desde mediciones procesadas.
# ==============================================================================

# -*- coding: utf-8 -*-
"""
extractor_de_datos_procesados.py - v1.3

Este script automatiza la extracción de pulsos individuales (ventanas recortadas)
de las mediciones ya procesadas y las organiza en una nueva base de datos
estructurada por letra y canal.

Funcionamiento:
1. Escanea la carpeta 'base_de_datos_electrodos'.
2. Identifica las mediciones "formales" que ya han sido analizadas,
   buscando carpetas cuyo nombre comience con una letra mayúscula seguida de
   un guion bajo (ej: 'A_Prueba1_Sujeto1').
3. Para cada medición formal, recorre sus subcarpetas de canal (ej: 'canal_0').
4. Lee el archivo 'analisis_results.json' para obtener los segmentos de pulso.
5. Lee el 'metadata.json' para obtener la resistencia del electrodo.
6. Calcula la ganancia y la amplitud real para cada pulso usando la fórmula:
   V_real = V_medida / (1 + R_fija / R_electrodo).
7. Acumula los datos en 'base_de_datos_letras', guardando cada pulso con un nombre único.
8. Genera un archivo 'amplitudes_maximas.csv' actualizado que ahora incluye
   amplitud medida, resistencia, ganancia y la amplitud real calibrada.
"""

import os
import json
import numpy as np
import scipy.signal
import re
import shutil
import pandas as pd
import sys
from PySide6.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal

def calculate_real_amplitude(df):
    R_fija = 49400.0
    df_copy = df.copy()
    df_copy['Ganancia (G)'] = 1 + (R_fija / df_copy['Resistencia'])
    df_copy['Amplitud_Real'] = df_copy['Amplitud_Medida']
    df_copy['Error_Amplitud_Real'] = df_copy['Error_Amplitud_Medida']
    return df_copy

class ExtractorThread(QThread):
    log_signal = Signal(str)
    progress_signal = Signal(int)
    finished_signal = Signal(bool, str)

    def __init__(self, fuente_dir, destino_dir, clean_dest):
        super().__init__()
        self.fuente_dir = fuente_dir
        self.destino_dir = destino_dir
        self.clean_dest = clean_dest

    def log(self, msg):
        print(msg)
        self.log_signal.emit(msg)

    def run(self):
        try:
            self.log("--- Iniciando Extractor de Datos Procesados v1.3 ---")
            
            if not os.path.isdir(self.fuente_dir):
                self.log(f"ERROR: El directorio fuente '{self.fuente_dir}' no existe.")
                self.finished_signal.emit(False, "Error: Directorio fuente no existe.")
                return

            if self.clean_dest and os.path.isdir(self.destino_dir):
                try:
                    shutil.rmtree(self.destino_dir)
                    self.log(f"Directorio '{self.destino_dir}' limpiado.")
                except Exception as e:
                    self.log(f"ERROR: No se pudo limpiar el directorio de destino. {e}")
                    self.finished_signal.emit(False, f"Error al limpiar: {e}")
                    return

            os.makedirs(self.destino_dir, exist_ok=True)
            
            regex_formal = re.compile(r"^[A-Z]_")
            total_pulsos_extraidos = 0
            amplitudes_data = []
            path_resumen_csv = os.path.join(self.destino_dir, "amplitudes_maximas.csv")
            
            mediciones = [m for m in sorted(os.listdir(self.fuente_dir)) if os.path.isdir(os.path.join(self.fuente_dir, m)) and regex_formal.match(m)]
            total_mediciones = len(mediciones)
            
            if total_mediciones == 0:
                self.log("No se encontraron mediciones formales.")
                self.finished_signal.emit(True, "No se encontraron mediciones.")
                return

            for idx, nombre_medicion in enumerate(mediciones):
                path_medicion = os.path.join(self.fuente_dir, nombre_medicion)
                self.log(f"Procesando: {nombre_medicion}")
                letra = nombre_medicion[0]
                
                canales = [c for c in sorted(os.listdir(path_medicion)) if os.path.isdir(os.path.join(path_medicion, c)) and c.startswith("canal_")]
                for nombre_canal in canales:
                    path_canal = os.path.join(path_medicion, nombre_canal)
                    path_metadata = os.path.join(path_canal, "metadata.json")
                    resistencia_ohm = None
                    
                    if os.path.exists(path_metadata):
                        try:
                            with open(path_metadata, 'r', encoding='utf-8') as f_meta:
                                metadata = json.load(f_meta)
                                resistencia_ohm = metadata.get("resistencia_ohm")
                        except: pass
                        
                    path_json = os.path.join(path_canal, "analisis_results.json")
                    if os.path.exists(path_json):
                        try:
                            with open(path_json, 'r', encoding='utf-8') as f:
                                datos_analisis = json.load(f)
                            segmentos = datos_analisis.get("segmentos_rs")
                            if segmentos and isinstance(segmentos, list):
                                path_destino_final = os.path.join(self.destino_dir, letra, nombre_canal)
                                os.makedirs(path_destino_final, exist_ok=True)
                                
                                for i, pulso in enumerate(segmentos):
                                    nombre_medicion_base = os.path.splitext(nombre_medicion)[0]
                                    nombre_archivo_pulso = f"{nombre_medicion_base}_pulso_{i+1:03d}.npy"
                                    path_npy_pulso = os.path.join(path_destino_final, nombre_archivo_pulso)
                                    
                                    pulso_arr = np.array(pulso)
                                    # Estandarización Tensorial para ML
                                    resampled = scipy.signal.resample(pulso_arr, 500)
                                    min_val = np.min(resampled)
                                    max_val = np.max(resampled)
                                    if max_val - min_val > 0:
                                        normalized = (resampled - min_val) / (max_val - min_val)
                                    else:
                                        normalized = resampled
                                    
                                    np.save(path_npy_pulso, normalized)
                                    total_pulsos_extraidos += 1
                                    
                                    amplitudes_data.append({
                                        "nombre_pulso": nombre_archivo_pulso,
                                        "Amplitud_Medida": np.max(pulso),
                                        "Resistencia": resistencia_ohm,
                                        "Error_Amplitud_Medida": 0.0 
                                    })
                                self.log(f"  -> Extraídos {len(segmentos)} pulsos de {nombre_canal}")
                        except Exception as e:
                            self.log(f"  -> ERROR en {path_json}: {e}")
                
                # Actualizar barra de progreso
                progreso = int(((idx + 1) / total_mediciones) * 100)
                self.progress_signal.emit(progreso)

            if amplitudes_data:
                df_amplitudes = pd.DataFrame(amplitudes_data)
                df_final = calculate_real_amplitude(df_amplitudes)
                column_order = ["nombre_pulso", "Amplitud_Medida", "Error_Amplitud_Medida", "Resistencia", "Ganancia (G)", "Amplitud_Real", "Error_Amplitud_Real"]
                df_final = df_final[column_order]
                df_final.to_csv(path_resumen_csv, index=False, float_format='%.6f')
                self.log(f"\nResumen de amplitudes guardado en '{path_resumen_csv}'")

            self.log(f"\n--- Proceso Finalizado ---")
            self.log_signal.emit(f"Total extraídos: {total_pulsos_extraidos} pulsos.")
            self.finished_signal.emit(True, f"Extracción completada. {total_pulsos_extraidos} pulsos.")
            
        except Exception as e:
            self.log_signal.emit(f"ERROR FATAL: {e}")
            self.finished_signal.emit(False, str(e))

class ExtractorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ñandú LSD - Extractor de Datos Procesados (ML)")
        self.resize(600, 400)
        self.setStyleSheet("background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace;")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.fuente_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_electrodos")
        self.destino_dir = os.path.join(os.path.dirname(script_dir), "base_de_datos_letras")
        
        layout = QVBoxLayout(self)
        
        lbl_info = QLabel("Extrae pulsos individuales (ventanas recortadas) de las mediciones formales ya procesadas y las organiza por letras para Machine Learning.")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #aaa; margin-bottom: 10px;")
        layout.addWidget(lbl_info)
        
        self.text_log = QTextEdit()
        self.text_log.setReadOnly(True)
        self.text_log.setStyleSheet("background-color: #111; color: #fff; border: 1px solid #333;")
        layout.addWidget(self.text_log)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("QProgressBar { border: 1px solid #333; border-radius: 3px; text-align: center; color: white; } QProgressBar::chunk { background-color: #00ffcc; }")
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        btn_layout = QHBoxLayout()
        
        self.btn_extract = QPushButton("Iniciar Extracción")
        self.btn_extract.setStyleSheet("background-color: #00ffcc; color: #000; font-weight: bold; padding: 10px;")
        self.btn_extract.clicked.connect(lambda: self.start_extraction(clean=False))
        btn_layout.addWidget(self.btn_extract)
        
        self.btn_extract_clean = QPushButton("Limpiar Destino e Iniciar")
        self.btn_extract_clean.setStyleSheet("background-color: #ff003c; color: #fff; font-weight: bold; padding: 10px;")
        self.btn_extract_clean.clicked.connect(lambda: self.start_extraction(clean=True))
        btn_layout.addWidget(self.btn_extract_clean)
        
        layout.addLayout(btn_layout)
        
        self.thread = None

    def start_extraction(self, clean):
        if clean:
            reply = QMessageBox.question(self, "Confirmar limpieza", f"¿Estás seguro de que quieres borrar completamente la carpeta '{self.destino_dir}' antes de extraer?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
                
        self.btn_extract.setEnabled(False)
        self.btn_extract_clean.setEnabled(False)
        self.text_log.clear()
        self.progress_bar.setValue(0)
        
        self.thread = ExtractorThread(self.fuente_dir, self.destino_dir, clean)
        self.thread.log_signal.connect(self.append_log)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def append_log(self, text):
        self.text_log.append(text)
        self.text_log.verticalScrollBar().setValue(self.text_log.verticalScrollBar().maximum())

    def on_finished(self, success, msg):
        self.btn_extract.setEnabled(True)
        self.btn_extract_clean.setEnabled(True)
        if success:
            QMessageBox.information(self, "Éxito", msg)
        else:
            QMessageBox.critical(self, "Error", msg)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = ExtractorDialog()
    dialog.exec()