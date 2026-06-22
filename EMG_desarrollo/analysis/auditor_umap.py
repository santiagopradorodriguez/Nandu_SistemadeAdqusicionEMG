import sys
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, 
    QHBoxLayout, QComboBox, QPushButton, QLabel, QMessageBox
)
from PySide6.QtCore import Qt

class AuditorUMAP(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Auditor de Vectores UMAP")
        self.resize(500, 150)
        self.setWindowFlags(self.windowFlags() | Qt.Dialog)
        
        layout = QVBoxLayout(self)
        
        # Buscar carpetas UMAP
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.db_dir = os.path.join(self.base_dir, "base_de_datos_electrodos")
        
        self.combo_box = QComboBox()
        self.cargar_opciones()
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel("Seleccionar Sesión UMAP:"))
        top_layout.addWidget(self.combo_box)
        
        btn_graficar = QPushButton("Graficar Vector Resultante")
        btn_graficar.clicked.connect(self.graficar)
        
        layout.addLayout(top_layout)
        layout.addWidget(btn_graficar)
        
    def cargar_opciones(self):
        opciones = []
        if os.path.exists(self.db_dir):
            for fecha in os.listdir(self.db_dir):
                umap_path = os.path.join(self.db_dir, fecha, "UMAP")
                if os.path.isdir(umap_path):
                    for sub in os.listdir(umap_path):
                        csv_path = os.path.join(umap_path, sub, "UMAP_features.csv")
                        if os.path.isfile(csv_path):
                            opciones.append((f"{fecha} -> {sub}", csv_path))
                            
        raiz_umap = os.path.join(self.base_dir, "UMAP")
        if os.path.exists(raiz_umap):
            for arch in os.listdir(raiz_umap):
                if arch.endswith(".csv"):
                    opciones.append((f"Raíz -> {arch}", os.path.join(raiz_umap, arch)))
                    
        for nombre, path in opciones:
            self.combo_box.addItem(nombre, path)
            
    def graficar(self):
        csv_path = self.combo_box.currentData()
        if not csv_path:
            QMessageBox.warning(self, "Aviso", "No hay ninguna matriz seleccionada.")
            return
            
        try:
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                if 'Label_Vocal' not in header:
                    QMessageBox.critical(self, "Error", "El CSV no tiene 'Label_Vocal'.")
                    return
                label_idx = header.index('Label_Vocal')
                
                data_features = []
                data_labels = []
                for row in reader:
                    if not row: continue
                    data_labels.append(row[label_idx])
                    feats = [float(x) for i, x in enumerate(row) if i != label_idx]
                    data_features.append(feats)
                    
            features = np.array(data_features)
            labels = np.array(data_labels)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al cargar:\n{e}")
            return
            
        vocales = np.unique(labels)
        is_picos = features.shape[1] < 15
        
        # Restauramos el estilo clásico y claro por defecto de matplotlib
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if is_picos:
            ax.set_title("Vectores UMAP - Modo PICOS")
            ax.set_xlabel("Índice de Canal")
            ax.set_ylabel("Amplitud Máxima Normalizada")
            for v in sorted(vocales):
                subset = features[labels == v]
                ax.plot(np.mean(subset, axis=0), marker='o', label=f'Promedio Vocal {v}')
        else:
            ax.set_title("Vectores UMAP - Modo COMPLETA (Onda Plana Concatenada)")
            ax.set_xlabel("Índice de Muestra Concatenada (Canal 0 -> Canal 1 -> ...)")
            ax.set_ylabel("Amplitud Normalizada Global")
            
            for v in sorted(vocales):
                subset = features[labels == v]
                mean_vec = np.mean(subset, axis=0)
                ax.plot(mean_vec, label=f'Promedio Vocal {v}')
                
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show(block=False)

def main():
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    
    ventana = AuditorUMAP()
    ventana.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
