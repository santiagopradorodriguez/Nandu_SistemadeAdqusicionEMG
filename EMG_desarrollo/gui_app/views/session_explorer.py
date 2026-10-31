# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz optimizado (Zero-Lag) con caché en memoria y
#              disco, escaneo vía os.scandir y filtro instantáneo de sesiones.
# ==============================================================================

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTreeWidget, QTreeWidgetItem, QPushButton, QSizePolicy, QComboBox,
    QMenu, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QUrl, QTimer
from PySide6.QtGui import QDesktopServices
import os
import sys
import subprocess
import re
import json


class SessionExplorer(QWidget):
    medicion_seleccionada = Signal(str)
    selection_changed = Signal()

    def __init__(self, root_path, parent=None):
        super().__init__(parent)
        self.root_path = root_path

        # Caché en memoria y persistente en disco para evitar re-leer metadata.json en cada refresco
        self._sujeto_cache = {}
        self._cache_dirty = False
        self._cache_file = self._resolve_cache_file()
        self._load_disk_cache()

        # Debounce para emisión de cambios de selección masivos
        self._selection_emit_timer = QTimer(self)
        self._selection_emit_timer.setSingleShot(True)
        self._selection_emit_timer.setInterval(80)
        self._selection_emit_timer.timeout.connect(self.selection_changed.emit)

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(8)

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

        # Barra de búsqueda rápida sin reconstruir el árbol
        self.inp_filtro = QLineEdit()
        self.inp_filtro.setPlaceholderText("Filtrar medición, vocal o sujeto...")
        self.inp_filtro.setClearButtonEnabled(True)
        self.inp_filtro.setStyleSheet("""
            QLineEdit {
                background-color: #161616;
                color: #00ffcc;
                border: 1px solid #333;
                padding: 4px 6px;
                border-radius: 3px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border: 1px solid #00ffcc;
            }
        """)
        self.inp_filtro.textChanged.connect(self._filtrar_arbol_en_vivo)
        self.layout.addWidget(self.inp_filtro)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)  # Optimización crítica de renderizado en Qt para árboles grandes
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._mostrar_menu_contextual)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #111; color: white; border: 1px solid #444; font-size: 13px;
            }
            QTreeWidget::item:hover { background-color: #333; }
        """)
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.layout.addWidget(self.tree)

        # Fila de controles de directorio
        layout_dirs = QHBoxLayout()
        layout_dirs.setContentsMargins(0, 0, 0, 0)
        layout_dirs.setSpacing(6)

        btn_refresh = QPushButton("Refrescar Directorio")
        btn_refresh.setStyleSheet("background-color: #333; color: white; padding: 5px; border-radius: 3px;")
        btn_refresh.clicked.connect(self.cargar_arbol)
        layout_dirs.addWidget(btn_refresh)

        self.btn_open_dir = QPushButton("Abrir Directorio")
        self.btn_open_dir.setStyleSheet("""
            QPushButton {
                background-color: #14261f;
                color: #00ffcc;
                border: 1px solid #00ffcc;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1b4332;
                color: #ffffff;
            }
        """)
        self.btn_open_dir.clicked.connect(self.abrir_directorio_seleccionado)
        layout_dirs.addWidget(self.btn_open_dir)

        self.layout.addLayout(layout_dirs)

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

    def _resolve_cache_file(self):
        try:
            cache_dir = os.path.join(os.path.dirname(self.root_path), "cache_datos")
            os.makedirs(cache_dir, exist_ok=True)
            return os.path.join(cache_dir, "_sujeto_explorer_cache.json")
        except Exception:
            return None

    def _load_disk_cache(self):
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._sujeto_cache = data
            except Exception:
                self._sujeto_cache = {}

    def _save_disk_cache(self):
        if not self._cache_dirty or not self._cache_file:
            return
        try:
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(self._sujeto_cache, f, ensure_ascii=False)
            self._cache_dirty = False
        except Exception:
            pass

    def _inferir_sujeto_por_nombre(self, folder_name):
        name = folder_name.lower()
        if "cande" in name:
            return "Candela"
        elif "lucas" in name:
            return "Lucas"
        elif "petra" in name:
            return "Petra"
        elif "santi" in name or "sujeto1" in name:
            return "Santi"
        # Si tiene formato Vocal_Prueba_Sujeto o SecuenciaContinua_Prueba_Sujeto_Hora
        parts = folder_name.split("_")
        if len(parts) >= 3:
            if parts[0].upper() in ("A", "E", "I", "O", "U") and not parts[-1].isdigit():
                return parts[-1]
            elif parts[0].startswith("SecuenciaContinua") and len(parts) >= 4:
                return parts[2]
        return None

    def _obtener_sujeto(self, med_path):
        cached = self._sujeto_cache.get(med_path)
        if cached:
            return cached

        # 1. Intentar inferencia rápida O(1) por nombre de carpeta sin tocar el disco
        fast_guess = self._inferir_sujeto_por_nombre(os.path.basename(med_path))
        if fast_guess:
            self._sujeto_cache[med_path] = fast_guess
            self._cache_dirty = True
            return fast_guess

        # 2. Leer metadata.json una única vez y guardar en caché
        for rel_meta in (os.path.join("canal_0", "metadata.json"), "metadata.json"):
            meta_path = os.path.join(med_path, rel_meta)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    sujeto = meta.get("sujeto") or (meta.get("identificacion", {}) or {}).get("sujeto")
                    if sujeto:
                        val = str(sujeto).strip()
                        self._sujeto_cache[med_path] = val
                        self._cache_dirty = True
                        return val
                except Exception:
                    pass

        self._sujeto_cache[med_path] = "Desconocido"
        self._cache_dirty = True
        return "Desconocido"

    def _escanear_fechas_y_mediciones(self):
        """Escaneo ultrarrápido con os.scandir evitando llamadas redundantes a stat()."""
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if not os.path.exists(self.root_path):
            try:
                os.makedirs(self.root_path, exist_ok=True)
            except Exception:
                return []

        fechas_entries = []
        try:
            with os.scandir(self.root_path) as it:
                for entry in it:
                    if entry.is_dir() and date_pattern.match(entry.name):
                        fechas_entries.append((entry.name, entry.path))
        except Exception:
            return []

        fechas_entries.sort(key=lambda x: x[0], reverse=True)
        resultado = []
        for fecha_name, fecha_path in fechas_entries:
            meds = []
            try:
                with os.scandir(fecha_path) as mit:
                    for m_entry in mit:
                        if m_entry.is_dir() and m_entry.name != "UMBRALES":
                            meds.append((m_entry.name, m_entry.path))
            except Exception:
                continue
            meds.sort(key=lambda x: x[0])
            resultado.append((fecha_name, fecha_path, meds))
        return resultado

    def cargar_arbol(self):
        # Preservar selección previa antes de recargar
        prev_selected = set(self.get_selected_paths())

        self.tree.setUpdatesEnabled(False)
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            estructura = self._escanear_fechas_y_mediciones()
            modo = self.cmb_agrupar.currentText()

            if modo == "Fecha":
                for fecha, fecha_path, mediciones in estructura:
                    fecha_item = QTreeWidgetItem(self.tree, [f"{fecha} ({len(mediciones)})"])
                    fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                    fecha_item.setCheckState(0, Qt.Unchecked)
                    fecha_item.setData(0, Qt.UserRole, fecha_path)
                    fecha_item.setData(0, Qt.UserRole + 1, "grupo")

                    for med, med_path in mediciones:
                        med_item = QTreeWidgetItem(fecha_item, [med])
                        med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                        med_item.setCheckState(0, Qt.Checked if med_path in prev_selected else Qt.Unchecked)
                        med_item.setData(0, Qt.UserRole, med_path)
                        med_item.setData(0, Qt.UserRole + 1, "medicion")

            elif modo == "Sujeto":
                sujetos_map = {}
                for fecha, fecha_path, mediciones in estructura:
                    for med, med_path in mediciones:
                        sujeto = self._obtener_sujeto(med_path)
                        sujetos_map.setdefault(sujeto, []).append((med, fecha, med_path))

                for sujeto in sorted(sujetos_map.keys()):
                    lista_tomas = sujetos_map[sujeto]
                    sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({len(lista_tomas)})"])
                    sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                    sujeto_item.setCheckState(0, Qt.Unchecked)
                    sujeto_item.setData(0, Qt.UserRole + 1, "grupo")

                    for med, fecha, med_path in sorted(lista_tomas, key=lambda x: (x[1], x[0]), reverse=True):
                        med_label = f"{med}  [{fecha}]"
                        med_item = QTreeWidgetItem(sujeto_item, [med_label])
                        med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                        med_item.setCheckState(0, Qt.Checked if med_path in prev_selected else Qt.Unchecked)
                        med_item.setData(0, Qt.UserRole, med_path)
                        med_item.setData(0, Qt.UserRole + 1, "medicion")

            elif modo == "Sujeto > Fecha":
                sujetos_map = {}
                for fecha, fecha_path, mediciones in estructura:
                    for med, med_path in mediciones:
                        sujeto = self._obtener_sujeto(med_path)
                        sujetos_map.setdefault(sujeto, {}).setdefault(fecha, []).append((med, med_path))

                for sujeto in sorted(sujetos_map.keys()):
                    total_tomas = sum(len(tomas) for tomas in sujetos_map[sujeto].values())
                    sujeto_item = QTreeWidgetItem(self.tree, [f"{sujeto} ({total_tomas})"])
                    sujeto_item.setFlags(sujeto_item.flags() | Qt.ItemIsUserCheckable)
                    sujeto_item.setCheckState(0, Qt.Unchecked)
                    sujeto_item.setData(0, Qt.UserRole + 1, "grupo")

                    for fecha in sorted(sujetos_map[sujeto].keys(), reverse=True):
                        tomas_fecha = sujetos_map[sujeto][fecha]
                        fecha_item = QTreeWidgetItem(sujeto_item, [f"{fecha} ({len(tomas_fecha)})"])
                        fecha_item.setFlags(fecha_item.flags() | Qt.ItemIsUserCheckable)
                        fecha_item.setCheckState(0, Qt.Unchecked)
                        fecha_item.setData(0, Qt.UserRole + 1, "grupo")

                        for med, med_path in tomas_fecha:
                            med_item = QTreeWidgetItem(fecha_item, [med])
                            med_item.setFlags(med_item.flags() | Qt.ItemIsUserCheckable)
                            med_item.setCheckState(0, Qt.Checked if med_path in prev_selected else Qt.Unchecked)
                            med_item.setData(0, Qt.UserRole, med_path)
                            med_item.setData(0, Qt.UserRole + 1, "medicion")

            if hasattr(self, "inp_filtro") and self.inp_filtro.text().strip():
                self._filtrar_arbol_en_vivo(self.inp_filtro.text())
        finally:
            self.tree.blockSignals(False)
            self.tree.setUpdatesEnabled(True)
            self._save_disk_cache()

    def _filtrar_arbol_en_vivo(self, texto):
        filtro = texto.strip().lower()
        self.tree.setUpdatesEnabled(False)
        try:
            def _filtrar_item(item):
                es_medicion = (item.data(0, Qt.UserRole + 1) == "medicion")
                propio_match = (not filtro) or (filtro in item.text(0).lower())
                if es_medicion:
                    item.setHidden(not propio_match)
                    return propio_match
                alguna_hija_visible = False
                for i in range(item.childCount()):
                    if _filtrar_item(item.child(i)):
                        alguna_hija_visible = True
                visible = propio_match or alguna_hija_visible
                item.setHidden(not visible)
                if filtro and alguna_hija_visible:
                    item.setExpanded(True)
                return visible

            for i in range(self.tree.topLevelItemCount()):
                _filtrar_item(self.tree.topLevelItem(i))
        finally:
            self.tree.setUpdatesEnabled(True)

    def _on_item_clicked(self, item, column):
        tipo = item.data(0, Qt.UserRole + 1)
        if tipo == "medicion":
            ruta = item.data(0, Qt.UserRole)
            self.medicion_seleccionada.emit(ruta)

    def _on_item_changed(self, item, column):
        """Propagar estado de checkbox de padre a hijos sin repintados intermedios y con debounce."""
        self.tree.setUpdatesEnabled(False)
        self.tree.blockSignals(True)
        try:
            estado = item.checkState(column)
            def _propagar(padre, st):
                for i in range(padre.childCount()):
                    hijo = padre.child(i)
                    if not hijo.isHidden():
                        hijo.setCheckState(column, st)
                        _propagar(hijo, st)
            _propagar(item, estado)
        finally:
            self.tree.blockSignals(False)
            self.tree.setUpdatesEnabled(True)
        self._selection_emit_timer.start()

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

    def _mostrar_menu_contextual(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a1a;
                color: #ffffff;
                border: 1px solid #444;
                padding: 4px;
            }
            QMenu::item:selected {
                background-color: #004d3a;
                color: #00ffcc;
            }
        """)
        action_open = menu.addAction("Abrir Directorio en Explorador del Sistema")
        action_open.triggered.connect(self.abrir_directorio_seleccionado)
        menu.exec(self.tree.viewport().mapToGlobal(pos))

    def abrir_directorio_seleccionado(self):
        target = None
        selected_items = self.tree.selectedItems()
        if selected_items:
            item = selected_items[0]
            ruta = item.data(0, Qt.UserRole)
            if ruta and os.path.exists(ruta):
                target = ruta
            else:
                for i in range(item.childCount()):
                    c_ruta = item.child(i).data(0, Qt.UserRole)
                    if c_ruta and os.path.exists(c_ruta):
                        target = os.path.dirname(c_ruta)
                        break

        if not target or not os.path.exists(target):
            target = self.root_path

        abs_target = os.path.abspath(target)
        if not os.path.exists(abs_target):
            try:
                os.makedirs(abs_target, exist_ok=True)
            except Exception:
                pass

        try:
            opened = QDesktopServices.openUrl(QUrl.fromLocalFile(abs_target))
            if opened:
                return
        except Exception:
            pass

        try:
            if sys.platform.startswith("win"):
                os.startfile(abs_target)
            elif sys.platform.startswith("darwin"):
                subprocess.Popen(["open", abs_target])
            else:
                subprocess.Popen(["xdg-open", abs_target])
        except Exception as e:
            print(f"[Aviso] No se pudo abrir directorio '{abs_target}': {e}")
