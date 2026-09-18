# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para navegar y explorar sesiones de medición.
# ==============================================================================

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSizePolicy, QComboBox
)
from PySide6.QtCore import Qt, Signal
import os
import re
import json

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

        # Control de Agrupamiento
        layout_modo = QHBoxLayout()
        lbl_modo = QLabel("Agrupar por:")
        lbl_modo.setStyleSheet("color: #ccc; font-size: 12px;")
        
        self.cmb_agrupar = QComboBox()
        self.cmb_agrupar.addItems(["Fecha", "Sujeto", "Sujeto > Fecha"])
        self.cmb_agrupar.setStyleSheet("""
            QComboBox {
                background-color: #222;
                color: #00ffcc;
                border: 1px solid #444;
                padding: 4px;
                border-radius: 3px;
                font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background-color: #222;
                color: #ffffff;
                selection-background-color: #004d3a;
            }
        """)
        self.cmb_agrupar.currentIndexChanged.connect(self.cargar_arbol)
        layout_modo.addWidget(lbl_modo)
        layout_modo.addWidget(self.cmb_agrupar)
        self.layout.addLayout(layout_modo)
        
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
        
        # Acceso directo al Autoencoder No Supervisado vinculado al Gestor
        self.btn_autoencoder_no_sup = QPushButton("Autoencoder No Supervisado")
        self.btn_autoencoder_no_sup.setStyleSheet("""
            QPushButton {
                background-color: #002b20;
                color: #00ffcc;
                border: 1px solid #00ffcc;
                font-weight: bold;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #004d3a;
                color: #ffffff;
            }
        """)
        self.layout.addWidget(self.btn_autoencoder_no_sup)
        
        self.setMinimumWidth(250)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        
        self.cargar_arbol()

    def _obtener_sujeto(self, med_path):
        meta_path = os.path.join(med_path, "canal_0", "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    sujeto = meta.get("sujeto")
                    if sujeto:
                        return str(sujeto).strip()
            except Exception:
                pass
        
        # Fallback desde nombre de carpeta
        name = os.path.basename(med_path).lower()
        if "cande" in name:
            return "Candela"
        elif "lucas" in name:
            return "Lucas"
        elif "petra" in name:
            return "Petra"
        elif "santi" in name or "sujeto1" in name:
            return "Santi"
        return "Desconocido"

    def cargar_arbol(self):
        self.tree.clear()
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not os.path.exists(self.root_path):
            try:
                os.makedirs(self.root_path, exist_ok=True)
            except Exception:
                return
            
        items = sorted(os.listdir(self.root_path), reverse=True)
        fechas = [d for d in items if os.path.isdir(os.path.join(self.root_path, d)) and date_pattern.match(d)]
        
        modo = self.cmb_agrupar.currentText()

        if modo == "Fecha":
            for fecha in fechas:
                fecha_path = os.path.join(self.root_path, fecha)
                fecha_item = QTreeWidgetItem(self.tree, [fecha])
                fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                fecha_item.setCheckState(0, Qt.Unchecked)
                fecha_item.setData(0, Qt.UserRole, fecha_path)
                fecha_item.setData(0, Qt.UserRole + 1, "grupo")
                
                mediciones = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
                for med in mediciones:
                    med_path = os.path.join(fecha_path, med)
                    med_item = QTreeWidgetItem(fecha_item, [med])
                    med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                    med_item.setCheckState(0, Qt.Unchecked)
                    med_item.setData(0, Qt.UserRole, med_path)
                    med_item.setData(0, Qt.UserRole + 1, "medicion")

        elif modo == "Sujeto":
            # Agrupar tomas por Sujeto
            sujetos_map = {}
            for fecha in fechas:
                fecha_path = os.path.join(self.root_path, fecha)
                mediciones = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
                for med in mediciones:
                    med_path = os.path.join(fecha_path, med)
                    sujeto = self._obtener_sujeto(med_path)
                    if sujeto not in sujetos_map:
                        sujetos_map[sujeto] = []
                    sujetos_map[sujeto].append((med, fecha, med_path))

            for sujeto in sorted(sujetos_map.keys()):
                sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({len(sujetos_map[sujeto])})"])
                sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                sujeto_item.setCheckState(0, Qt.Unchecked)
                sujeto_item.setData(0, Qt.UserRole + 1, "grupo")

                for med, fecha, med_path in sorted(sujetos_map[sujeto], key=lambda x: (x[1], x[0]), reverse=True):
                    med_label = f"{med}  [{fecha}]"
                    med_item = QTreeWidgetItem(sujeto_item, [med_label])
                    med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                    med_item.setCheckState(0, Qt.Unchecked)
                    med_item.setData(0, Qt.UserRole, med_path)
                    med_item.setData(0, Qt.UserRole + 1, "medicion")

        elif modo == "Sujeto > Fecha":
            # Agrupar por Sujeto -> Fecha -> Medicion
            sujetos_map = {}
            for fecha in fechas:
                fecha_path = os.path.join(self.root_path, fecha)
                mediciones = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
                for med in mediciones:
                    med_path = os.path.join(fecha_path, med)
                    sujeto = self._obtener_sujeto(med_path)
                    if sujeto not in sujetos_map:
                        sujetos_map[sujeto] = {}
                    if fecha not in sujetos_map[sujeto]:
                        sujetos_map[sujeto][fecha] = []
                    sujetos_map[sujeto][fecha].append((med, med_path))

            for sujeto in sorted(sujetos_map.keys()):
                total_tomas = sum(len(tomas) for tomas in sujetos_map[sujeto].values())
                sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({total_tomas})"])
                sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                sujeto_item.setCheckState(0, Qt.Unchecked)
                sujeto_item.setData(0, Qt.UserRole + 1, "grupo")

                for fecha in sorted(sujetos_map[sujeto].keys(), reverse=True):
                    fecha_item = QTreeWidgetItem(sujeto_item, [f"{fecha} ({len(sujetos_map[sujeto][fecha])})"])
                    fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                    fecha_item.setCheckState(0, Qt.Unchecked)
                    fecha_item.setData(0, Qt.UserRole + 1, "grupo")

                    for med, med_path in sorted(sujetos_map[sujeto][fecha], key=lambda x: x[0]):
                        med_item = QTreeWidgetItem(fecha_item, [med])
                        med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                        med_item.setCheckState(0, Qt.Unchecked)
                        med_item.setData(0, Qt.UserRole, med_path)
                        med_item.setData(0, Qt.UserRole + 1, "medicion")

    def _on_item_clicked(self, item, column):
        tipo = item.data(0, Qt.UserRole + 1)
        if tipo == "medicion":
            ruta = item.data(0, Qt.UserRole)
            self.medicion_seleccionada.emit(ruta)

    def _on_item_changed(self, item, column):
        """Propagar estado de checkbox de padre a hijos recursivamente"""
        self.tree.blockSignals(True)
        estado = item.checkState(column)
        def _propagar(padre, st):
            for i in range(padre.childCount()):
                hijo = padre.child(i)
                hijo.setCheckState(column, st)
                _propagar(hijo, st)
        _propagar(item, estado)
        self.tree.blockSignals(False)
        self.selection_changed.emit()

    def get_selected_paths(self):
        """Devuelve una LISTA de rutas absolutas (las mediciones que están tildadas)"""
        rutas = []
        def _recolectar(item):
            tipo = item.data(0, Qt.UserRole + 1)
            if tipo == "medicion" and item.checkState(0) == Qt.Checked:
                ruta = item.data(0, Qt.UserRole)
                if ruta:
                    rutas.append(ruta)
            for i in range(item.childCount()):
                _recolectar(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _recolectar(self.tree.topLevelItem(i))
        return list(dict.fromkeys(rutas))
