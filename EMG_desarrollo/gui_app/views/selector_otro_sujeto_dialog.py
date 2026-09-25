# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo selector de mediciones con casillas de verificación para
#              probar modelos en otros sujetos o datasets.
# ==============================================================================

import os
import re
import json
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTreeWidget, QTreeWidgetItem, QComboBox,
    QFileDialog, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt
from utils.path_utils import get_database_path, get_project_root


class SelectorMedicionesOtroSujetoDialog(QDialog):
    """
    Diálogo interactivo para seleccionar quirúrgicamente las grabaciones de
    otro sujeto a evaluar con el modelo de autoencoder entrenado.
    Permite marcar con casillas de verificación (checkboxes) sujetos enteros,
    fechas o mediciones individuales.
    """
    def __init__(self, parent=None, rutas_preseleccionadas=None):
        super().__init__(parent)
        self.setWindowTitle("Selector de Mediciones: Probar en Otro Sujeto")
        self.setMinimumSize(780, 560)
        self.resize(840, 620)

        self.root_path = get_database_path()
        self.rutas_preseleccionadas = set(os.path.abspath(r) for r in (rutas_preseleccionadas or []))
        self.archivo_externo = None

        self.setStyleSheet("""
            QDialog {
                background-color: #0c0c0c;
                color: #e0e0e0;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 12px;
            }
            QLineEdit {
                background-color: #1a1a1a;
                color: #66FCF1;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 6px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #66FCF1;
            }
            QComboBox {
                background-color: #1a1a1a;
                color: #66FCF1;
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 5px;
                font-size: 12px;
            }
            QTreeWidget {
                background-color: #121212;
                color: #d0d0d0;
                border: 1px solid #2a2a2a;
                border-radius: 4px;
                font-size: 12px;
            }
            QTreeWidget::item {
                padding: 4px;
                border-radius: 3px;
            }
            QTreeWidget::item:hover {
                background-color: #1f2833;
                color: #ffffff;
            }
            QTreeWidget::item:selected {
                background-color: #003322;
                color: #00FF88;
            }
            QPushButton {
                background-color: #1f2833;
                color: #c5c6c7;
                border: 1px solid #45a29e;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #45a29e;
                color: #0b0c10;
            }
            QPushButton#btnEvaluar {
                background-color: #004d26;
                color: #00FF88;
                border: 2px solid #00FF88;
                font-size: 12px;
                padding: 8px 18px;
            }
            QPushButton#btnEvaluar:hover {
                background-color: #00FF88;
                color: #000000;
            }
            QPushButton#btnEvaluar:disabled {
                background-color: #1a1a1a;
                color: #555555;
                border: 1px solid #333333;
            }
            QPushButton#btnArchivo {
                background-color: #2b2b1a;
                color: #FFE600;
                border: 1px solid #FFE600;
            }
            QPushButton#btnArchivo:hover {
                background-color: #FFE600;
                color: #000000;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # 1. Cabecera explicativa
        lbl_info = QLabel("Seleccione las mediciones del sujeto a evaluar marcando sus casillas de verificación:")
        lbl_info.setStyleSheet("color: #66FCF1; font-weight: bold; font-size: 13px;")
        layout.addWidget(lbl_info)

        # 2. Barra de Control: Agrupación y Búsqueda
        bar_top = QHBoxLayout()
        bar_top.setSpacing(8)

        bar_top.addWidget(QLabel("Agrupar por:"))
        self.cmb_agrupar = QComboBox()
        self.cmb_agrupar.addItems(["Sujeto / Fecha", "Sujeto", "Fecha"])
        self.cmb_agrupar.currentIndexChanged.connect(self.cargar_arbol)
        bar_top.addWidget(self.cmb_agrupar)

        bar_top.addSpacing(10)
        bar_top.addWidget(QLabel("Filtrar:"))
        self.inp_filtro = QLineEdit()
        self.inp_filtro.setPlaceholderText("Buscar sujeto, fecha o medición (ej. Candela, Prueba5)...")
        self.inp_filtro.textChanged.connect(self._aplicar_filtro)
        bar_top.addWidget(self.inp_filtro)

        layout.addLayout(bar_top)

        # 3. Botones de Selección Rápida
        bar_quick = QHBoxLayout()
        bar_quick.setSpacing(6)

        self.btn_marcar_todo = QPushButton("Marcar Todo")
        self.btn_marcar_todo.clicked.connect(lambda: self._set_all_check(True))
        bar_quick.addWidget(self.btn_marcar_todo)

        self.btn_desmarcar_todo = QPushButton("Desmarcar Todo")
        self.btn_desmarcar_todo.clicked.connect(lambda: self._set_all_check(False))
        bar_quick.addWidget(self.btn_desmarcar_todo)

        self.btn_solo_continuas = QPushButton("Solo Secuencias Continuas")
        self.btn_solo_continuas.clicked.connect(self._seleccionar_solo_continuas)
        bar_quick.addWidget(self.btn_solo_continuas)

        self.btn_solo_aisladas = QPushButton("Solo Vocales Aisladas")
        self.btn_solo_aisladas.clicked.connect(self._seleccionar_solo_aisladas)
        bar_quick.addWidget(self.btn_solo_aisladas)

        bar_quick.addStretch()
        layout.addLayout(bar_quick)

        # 4. Árbol de Mediciones con Checkboxes
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Medición / Sesión", "Sujeto", "Fecha", "Canales"])
        self.tree.setColumnWidth(0, 380)
        self.tree.setColumnWidth(1, 130)
        self.tree.setColumnWidth(2, 110)
        self.tree.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.tree)

        # 5. Barra Inferior de Estado y Acciones
        bar_bottom = QHBoxLayout()
        bar_bottom.setSpacing(10)

        self.lbl_contador = QLabel("Mediciones seleccionadas: 0 tomas")
        self.lbl_contador.setStyleSheet("color: #00FF88; font-weight: bold; font-size: 13px;")
        bar_bottom.addWidget(self.lbl_contador)

        bar_bottom.addStretch()

        self.btn_cargar_archivo = QPushButton("Cargar Archivo Externo (.npz / .csv)...")
        self.btn_cargar_archivo.setObjectName("btnArchivo")
        self.btn_cargar_archivo.clicked.connect(self._on_cargar_archivo_externo)
        bar_bottom.addWidget(self.btn_cargar_archivo)

        self.btn_cancelar = QPushButton("Cancelar")
        self.btn_cancelar.clicked.connect(self.reject)
        bar_bottom.addWidget(self.btn_cancelar)

        self.btn_evaluar = QPushButton("EVALUAR SELECCIÓN EN MODELO")
        self.btn_evaluar.setObjectName("btnEvaluar")
        self.btn_evaluar.setEnabled(False)
        self.btn_evaluar.clicked.connect(self.accept)
        bar_bottom.addWidget(self.btn_evaluar)

        layout.addLayout(bar_bottom)

        # Cargar los datos de la base de datos
        self.cargar_arbol()

    def _obtener_metadatos_medicion(self, med_path):
        meta_path = os.path.join(med_path, "canal_0", "metadata.json")
        sujeto = "Desconocido"
        canales_count = 0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    s = meta.get("sujeto")
                    if s:
                        sujeto = str(s).strip()
            except Exception:
                pass
        
        # Conteo de canales
        for ch in range(4):
            if os.path.exists(os.path.join(med_path, f"canal_{ch}", "grabacion.wav")):
                canales_count += 1

        if sujeto == "Desconocido":
            name = os.path.basename(med_path).lower()
            if "cande" in name:
                sujeto = "Candela"
            elif "lucas" in name:
                sujeto = "Lucas"
            elif "petra" in name:
                sujeto = "Petra"
            elif "santi" in name or "sujeto1" in name:
                sujeto = "Santi"

        return sujeto, canales_count

    def cargar_arbol(self):
        self.tree.blockSignals(True)
        self.tree.clear()

        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not os.path.exists(self.root_path):
            self.tree.blockSignals(False)
            return

        items = sorted(os.listdir(self.root_path), reverse=True)
        fechas = [d for d in items if os.path.isdir(os.path.join(self.root_path, d)) and date_pattern.match(d)]
        modo = self.cmb_agrupar.currentText()

        # Recolectar todas las tomas
        tomas_info = []
        for fecha in fechas:
            fecha_path = os.path.join(self.root_path, fecha)
            mediciones = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
            for med in mediciones:
                med_path = os.path.join(fecha_path, med)
                if not os.path.exists(os.path.join(med_path, "canal_0", "grabacion.wav")):
                    continue
                sujeto, ch_cnt = self._obtener_metadatos_medicion(med_path)
                tomas_info.append({
                    'nombre': med,
                    'fecha': fecha,
                    'sujeto': sujeto,
                    'path': med_path,
                    'canales': f"{ch_cnt} canales"
                })

        if modo == "Sujeto / Fecha":
            # Sujeto -> Fecha -> Tomas
            sujetos_map = {}
            for t in tomas_info:
                s = t['sujeto']
                f = t['fecha']
                sujetos_map.setdefault(s, {}).setdefault(f, []).append(t)

            for sujeto in sorted(sujetos_map.keys()):
                total_s = sum(len(sujetos_map[sujeto][f]) for f in sujetos_map[sujeto])
                sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({total_s} tomas)", sujeto, "", ""])
                sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                sujeto_item.setCheckState(0, Qt.Unchecked)
                sujeto_item.setData(0, Qt.UserRole + 1, "grupo")
                sujeto_item.setExpanded(True)

                for fecha in sorted(sujetos_map[sujeto].keys(), reverse=True):
                    tomas_f = sujetos_map[sujeto][fecha]
                    fecha_item = QTreeWidgetItem(sujeto_item, [f"{fecha} ({len(tomas_f)} tomas)", sujeto, fecha, ""])
                    fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                    fecha_item.setCheckState(0, Qt.Unchecked)
                    fecha_item.setData(0, Qt.UserRole + 1, "grupo")
                    fecha_item.setExpanded(True)

                    for t in tomas_f:
                        med_item = QTreeWidgetItem(fecha_item, [t['nombre'], t['sujeto'], t['fecha'], t['canales']])
                        med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                        pre_check = Qt.Checked if os.path.abspath(t['path']) in self.rutas_preseleccionadas else Qt.Unchecked
                        med_item.setCheckState(0, pre_check)
                        med_item.setData(0, Qt.UserRole, t['path'])
                        med_item.setData(0, Qt.UserRole + 1, "medicion")

        elif modo == "Sujeto":
            # Sujeto -> Tomas
            sujetos_map = {}
            for t in tomas_info:
                sujetos_map.setdefault(t['sujeto'], []).append(t)

            for sujeto in sorted(sujetos_map.keys()):
                tomas_s = sujetos_map[sujeto]
                sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({len(tomas_s)} tomas)", sujeto, "", ""])
                sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                sujeto_item.setCheckState(0, Qt.Unchecked)
                sujeto_item.setData(0, Qt.UserRole + 1, "grupo")
                sujeto_item.setExpanded(True)

                for t in sorted(tomas_s, key=lambda x: (x['fecha'], x['nombre']), reverse=True):
                    med_item = QTreeWidgetItem(sujeto_item, [t['nombre'], t['sujeto'], t['fecha'], t['canales']])
                    med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                    pre_check = Qt.Checked if os.path.abspath(t['path']) in self.rutas_preseleccionadas else Qt.Unchecked
                    med_item.setCheckState(0, pre_check)
                    med_item.setData(0, Qt.UserRole, t['path'])
                    med_item.setData(0, Qt.UserRole + 1, "medicion")

        elif modo == "Fecha":
            # Fecha -> Tomas
            fechas_map = {}
            for t in tomas_info:
                fechas_map.setdefault(t['fecha'], []).append(t)

            for fecha in sorted(fechas_map.keys(), reverse=True):
                tomas_f = fechas_map[fecha]
                fecha_item = QTreeWidgetItem(self.tree, [f"{fecha} ({len(tomas_f)} tomas)", "", fecha, ""])
                fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                fecha_item.setCheckState(0, Qt.Unchecked)
                fecha_item.setData(0, Qt.UserRole + 1, "grupo")
                fecha_item.setExpanded(True)

                for t in tomas_f:
                    med_item = QTreeWidgetItem(fecha_item, [t['nombre'], t['sujeto'], t['fecha'], t['canales']])
                    med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                    pre_check = Qt.Checked if os.path.abspath(t['path']) in self.rutas_preseleccionadas else Qt.Unchecked
                    med_item.setCheckState(0, pre_check)
                    med_item.setData(0, Qt.UserRole, t['path'])
                    med_item.setData(0, Qt.UserRole + 1, "medicion")

        self.tree.blockSignals(False)
        self._actualizar_contador()

    def _on_item_changed(self, item, column):
        if column != 0:
            return
        tipo = item.data(0, Qt.UserRole + 1)
        estado = item.checkState(0)

        self.tree.blockSignals(True)
        if tipo == "grupo":
            def _propagar_hacia_abajo(padre, st):
                for i in range(padre.childCount()):
                    hijo = padre.child(i)
                    hijo.setCheckState(0, st)
                    _propagar_hacia_abajo(hijo, st)
            _propagar_hacia_abajo(item, estado)

        self.tree.blockSignals(False)
        self._actualizar_contador()

    def _set_all_check(self, checked):
        state = Qt.Checked if checked else Qt.Unchecked
        self.tree.blockSignals(True)
        def _set_check(item):
            item.setCheckState(0, state)
            for i in range(item.childCount()):
                _set_check(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _set_check(self.tree.topLevelItem(i))
        self.tree.blockSignals(False)
        self._actualizar_contador()

    def _seleccionar_solo_continuas(self):
        self.tree.blockSignals(True)
        def _check_continua(item):
            tipo = item.data(0, Qt.UserRole + 1)
            if tipo == "medicion":
                nombre = item.text(0).lower()
                es_continua = ("continua" in nombre or "prueba5" in nombre or "p5" in nombre)
                item.setCheckState(0, Qt.Checked if es_continua else Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.Unchecked)
            for i in range(item.childCount()):
                _check_continua(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _check_continua(self.tree.topLevelItem(i))
        self.tree.blockSignals(False)
        self._actualizar_contador()

    def _seleccionar_solo_aisladas(self):
        self.tree.blockSignals(True)
        def _check_aislada(item):
            tipo = item.data(0, Qt.UserRole + 1)
            if tipo == "medicion":
                nombre = item.text(0).lower()
                es_aislada = any(nombre.startswith(f"{v.lower()}_") for v in ['a', 'e', 'i', 'o', 'u'])
                item.setCheckState(0, Qt.Checked if es_aislada else Qt.Unchecked)
            else:
                item.setCheckState(0, Qt.Unchecked)
            for i in range(item.childCount()):
                _check_aislada(item.child(i))

        for i in range(self.tree.topLevelItemCount()):
            _check_aislada(self.tree.topLevelItem(i))
        self.tree.blockSignals(False)
        self._actualizar_contador()

    def _aplicar_filtro(self, texto):
        filtro = texto.lower().strip()
        def _filtrar_item(item):
            coincide_propio = any(filtro in item.text(c).lower() for c in range(item.columnCount()))
            coincide_hijo = False
            for i in range(item.childCount()):
                if _filtrar_item(item.child(i)):
                    coincide_hijo = True
            visible = coincide_propio or coincide_hijo or (filtro == "")
            item.setHidden(not visible)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            _filtrar_item(self.tree.topLevelItem(i))

    def _actualizar_contador(self):
        rutas = self.get_selected_paths()
        n = len(rutas)
        if self.archivo_externo:
            nom_arch = os.path.basename(self.archivo_externo)
            self.lbl_contador.setText(f"Archivo cargado: {nom_arch}")
            self.btn_evaluar.setEnabled(True)
        else:
            self.lbl_contador.setText(f"Mediciones seleccionadas: {n} tomas")
            self.btn_evaluar.setEnabled(n > 0)

    def _on_cargar_archivo_externo(self):
        root_dir = get_project_root()
        base_dir = os.path.join(root_dir, "resultados")
        archivo_sel, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccione el Archivo CSV o NPZ del Dataset Externo",
            base_dir,
            "Datasets (*.csv *.npz);;Todos los archivos (*.*)"
        )
        if archivo_sel:
            self.archivo_externo = archivo_sel
            self._actualizar_contador()

    def get_selected_paths(self):
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

    def get_selected_file(self):
        return self.archivo_externo
