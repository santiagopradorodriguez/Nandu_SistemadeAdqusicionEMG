import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, 
    QHBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox, QListWidget, QSplitter
)
from PySide6.QtCore import Qt

class AuditorVectorialPCA(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Visor de Features (Data PCA/UMAP)")
        self.resize(1100, 700)
        self.setWindowFlags(self.windowFlags() | Qt.Dialog)
        
        self.bg_dark = "#0B0C10"
        self.bg_panel = "#1F2833"
        self.cyan_neon = "#66FCF1"
        self.fg_text = "#C5C6C7"
        
        self.setStyleSheet(f"background-color: {self.bg_dark}; color: {self.fg_text};")
        
        self.df = None
        self.visible_indices = []
        
        main_layout = QHBoxLayout(self)
        
        # Panel Izquierdo
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Buscar carpetas PCA
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_dir = os.path.join(self.base_dir, "base_de_datos_electrodos")
        
        self.combo_box = QComboBox()
        self.cargar_opciones()
        self.combo_box.currentIndexChanged.connect(self.on_fuente_selected)
        
        self.lbl_info = QLabel("No hay archivo cargado")
        self.lbl_info.setWordWrap(True)
        
        self.vocal_filter = QComboBox()
        self.vocal_filter.addItems(["Todas", "A", "E", "I", "O", "U"])
        self.vocal_filter.currentIndexChanged.connect(self.apply_filter)
        
        self.listbox = QListWidget()
        self.listbox.setStyleSheet(f"background-color: {self.bg_panel}; color: white; selection-background-color: {self.cyan_neon}; selection-color: black;")
        self.listbox.itemSelectionChanged.connect(self.on_select)
        
        left_layout.addWidget(QLabel("Seleccionar Matriz 300D:"))
        left_layout.addWidget(self.combo_box)
        left_layout.addWidget(self.lbl_info)
        left_layout.addWidget(QLabel("Filtrar por Vocal:"))
        left_layout.addWidget(self.vocal_filter)
        left_layout.addWidget(QLabel("Tomas (Instancias extraídas):"))
        left_layout.addWidget(self.listbox)
        
        # Panel Derecho (Gráfico)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.fig.patch.set_facecolor(self.bg_dark)
        self.ax.set_facecolor(self.bg_panel)
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color(self.cyan_neon)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.fg_text)
            
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)
        
        self.on_fuente_selected()
        
    def cargar_opciones(self):
        opciones = []
        if os.path.exists(self.db_dir):
            for fecha in os.listdir(self.db_dir):
                pca_path = os.path.join(self.db_dir, fecha, "PCA")
                if os.path.isdir(pca_path):
                    for sub in os.listdir(pca_path):
                        csv_path = os.path.join(pca_path, sub, "vector_maestro_300d.csv")
                        if os.path.isfile(csv_path):
                            opciones.append((f"{fecha} -> {sub}", csv_path))
                            
        for nombre, path in opciones:
            self.combo_box.addItem(nombre, path)

    def on_fuente_selected(self):
        csv_path = self.combo_box.currentData()
        if not csv_path or not os.path.exists(csv_path):
            self.df = None
            self.lbl_info.setText("No hay archivo cargado")
            self.listbox.clear()
            return
            
        try:
            self.df = pd.read_csv(csv_path)
            self.lbl_info.setText(f"Archivo: {os.path.basename(csv_path)}\nTotal instancias: {len(self.df)}")
            self.apply_filter()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el archivo:\n{e}")

    def apply_filter(self):
        self.listbox.clear()
        self.visible_indices = []
        if self.df is None: return
        
        filtro = self.vocal_filter.currentText()
        
        for i, row in self.df.iterrows():
            vocal = row.get("Vocal", "?")
            toma = row.get("Toma", "?")
            if filtro == "Todas" or vocal == filtro:
                self.listbox.addItem(f"{vocal} - {os.path.basename(str(toma))}")
                self.visible_indices.append(i)

    def on_select(self):
        selected = self.listbox.selectedItems()
        if not selected: return
        
        idx_in_list = self.listbox.row(selected[0])
        real_idx = self.visible_indices[idx_in_list]
        self.plot_data(real_idx)

    def plot_data(self, idx):
        row = self.df.iloc[idx]
        
        ch0_vals = []
        ch1_vals = []
        ch2_vals = []
        
        # Identificar longitud (TARGET_LEN)
        t_max = 0
        while f"canal_0_T{t_max}" in row or f"Ch0_T{t_max}" in row:
            t_max += 1
            
        if t_max == 0:
            QMessageBox.warning(self, "Aviso", "No se encontraron columnas de tiempo.")
            return
            
        for t in range(t_max):
            # Soporta tanto canal_0_T0 como Ch0_T0
            v0 = row.get(f"canal_0_T{t}", row.get(f"Ch0_T{t}", 0.0))
            v1 = row.get(f"canal_1_T{t}", row.get(f"Ch1_T{t}", 0.0))
            v2 = row.get(f"canal_2_T{t}", row.get(f"Ch2_T{t}", 0.0))
            ch0_vals.append(float(v0))
            ch1_vals.append(float(v1))
            ch2_vals.append(float(v2))
            
        self.ax.clear()
        self.ax.set_facecolor(self.bg_panel)
        self.ax.tick_params(colors="white")
        self.ax.xaxis.label.set_color("white")
        self.ax.yaxis.label.set_color("white")
        self.ax.title.set_color(self.cyan_neon)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.fg_text)
            
        time_axis = np.linspace(-50, 50, t_max)
        
        self.ax.plot(time_axis, ch0_vals, label='Canal 0 (Mylohyoid)', color='#45B7D1', linewidth=2)
        self.ax.plot(time_axis, ch1_vals, label='Canal 1 (Depressor Anguli Oris)', color='#FF6B6B', linewidth=2)
        self.ax.plot(time_axis, ch2_vals, label='Canal 2 (Orbicularis Oris)', color='#C5C6C7', linewidth=2)
        
        self.ax.axvline(x=0, color='#F3E94C', linestyle='--', linewidth=2, alpha=0.8, label='Pico Micrófono (Ancla)')
        
        self.ax.set_title(f"Características Dinámicas - Toma: {row.get('Toma', '?')}", color=self.cyan_neon)
        self.ax.set_xlabel('Tiempo relativo al pico del micrófono (%)')
        self.ax.set_ylabel('Amplitud Normalizada')
        self.ax.legend(facecolor=self.bg_dark, edgecolor=self.cyan_neon, labelcolor='white', loc='upper right')
        self.ax.grid(True, color=self.fg_text, alpha=0.2)
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ventana = AuditorVectorialPCA()
    ventana.show()
    sys.exit(app.exec())
