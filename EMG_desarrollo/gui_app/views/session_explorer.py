# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para navegar y explorar sesiones de medición.
# ==============================================================================

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
import os
import re

class SessionExplorer(QWidget):
    medicion_seleccionada = Signal(str)
    selection_changed = Signal()

    def __init__(self, root_path, parent=None):
        super().__init__(parent)
        self.root_path = root_path
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Etiqueta de Título
        lbl_title = QLabel("Base de Datos de Electrodos")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF4444;")
        self.layout.addWidget(lbl_title)
        
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #111; color: white; border: 1px solid #444; font-size: 13px;
            }
            QTreeWidget::item:hover { background-color: #333; }
        """)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.layout.addWidget(self.tree)
        
        # Botón Refrescar
        btn_refresh = QPushButton("Refrescar Directorio")
        btn_refresh.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        btn_refresh.clicked.connect(self.cargar_arbol)
        self.layout.addWidget(btn_refresh)
        
        self.setMinimumWidth(250)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.cargar_arbol()
        
    def cargar_arbol(self):
        self.tree.clear()
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not os.path.exists(self.root_path):
            return
            
        items = sorted(os.listdir(self.root_path), reverse=True)
        fechas = [d for d in items if os.path.isdir(os.path.join(self.root_path, d)) and date_pattern.match(d)]
        
        for fecha in fechas:
            fecha_path = os.path.join(self.root_path, fecha)
            fecha_item = QTreeWidgetItem(self.tree, [fecha])
            fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
            fecha_item.setCheckState(0, Qt.Unchecked)
            fecha_item.setData(0, Qt.UserRole, fecha_path)
            # Marcar que es fecha (padre)
            fecha_item.setData(0, Qt.UserRole + 1, "fecha")
            
            mediciones = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
            for med in mediciones:
                med_path = os.path.join(fecha_path, med)
                med_item = QTreeWidgetItem(fecha_item, [med])
                med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                med_item.setCheckState(0, Qt.Unchecked)
                med_item.setData(0, Qt.UserRole, med_path)
                # Marcar que es medición (hijo)
                med_item.setData(0, Qt.UserRole + 1, "medicion")

    def _on_item_clicked(self, item, column):
        tipo = item.data(0, Qt.UserRole + 1)
        if tipo == "medicion":
            ruta = item.data(0, Qt.UserRole)
            self.medicion_seleccionada.emit(ruta)

    def _on_item_changed(self, item, column):
        """Propagar estado de checkbox de padre a hijos"""
        self.tree.blockSignals(True)
        estado = item.checkState(column)
        for i in range(item.childCount()):
            item.child(i).setCheckState(column, estado)
        self.tree.blockSignals(False)
        self.selection_changed.emit()

    def get_selected_paths(self):
        """Devuelve una LISTA de rutas absolutas (las mediciones que están tildadas)"""
        rutas = []
        for i in range(self.tree.topLevelItemCount()):
            fecha_item = self.tree.topLevelItem(i)
            for j in range(fecha_item.childCount()):
                med_item = fecha_item.child(j)
                if med_item.checkState(0) == Qt.Checked:
                    rutas.append(med_item.data(0, Qt.UserRole))
        return rutas
