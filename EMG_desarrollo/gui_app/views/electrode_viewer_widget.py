# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para visualizar señales por electrodo.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para visualizar señales por electrodo.
# ==============================================================================

import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSplitter, 
    QListWidget, QListWidgetItem, QTabWidget, QScrollArea,
    QPushButton, QHBoxLayout, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage, QCursor
from PySide6.QtWidgets import QDialog, QVBoxLayout, QWidget

class ClickableImage(QLabel):
    def __init__(self, img_path, max_width=400, parent=None):
        super().__init__(parent)
        self.img_path = img_path
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self.setStyleSheet("background-color: #000; border: 1px solid #333; padding: 5px; border-radius: 4px;")
        
        self.pix = QPixmap(img_path)
        if self.pix.width() > max_width:
            self.setPixmap(self.pix.scaledToWidth(max_width, Qt.SmoothTransformation))
        else:
            self.setPixmap(self.pix)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.show_fullscreen_image()
            
    def show_fullscreen_image(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Ñandú LSD - Visor de Imagen")
        dialog.setStyleSheet("background-color: #050505;")
        
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        
        lbl = QLabel()
        lbl.setAlignment(Qt.AlignCenter)
        
        # Escalar al 90% de la pantalla actual para que quepa bien
        screen = self.screen().availableGeometry()
        max_w = int(screen.width() * 0.9)
        max_h = int(screen.height() * 0.9)
        
        if self.pix.width() > max_w or self.pix.height() > max_h:
            lbl.setPixmap(self.pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            lbl.setPixmap(self.pix)
            
        scroll.setWidget(lbl)
        layout.addWidget(scroll)
        
        dialog.resize(min(self.pix.width() + 40, max_w), min(self.pix.height() + 40, max_h))
        dialog.exec()

class ThumbnailLoader(QThread):
    finished = Signal(str, str, QPixmap) # path, name, pixmap

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        for path in self.paths:
            name = os.path.basename(path)
            # Find representative plot or photo
            thumb_path = None
            calibrated_plot = os.path.join(path, f"plot_calibrado_{name}.png")
            if os.path.exists(calibrated_plot):
                thumb_path = calibrated_plot
            else:
                # search in canal_0
                c0 = os.path.join(path, "canal_0")
                if os.path.exists(c0):
                    if hasattr(self, 'suffix') and self.suffix and self.suffix != "Antiguo":
                        avg = os.path.join(c0, f"avg_{self.suffix}.png")
                    else:
                        avg = os.path.join(c0, "avg.png")
                        
                    if os.path.exists(avg):
                        thumb_path = avg
            
            pixmap = QPixmap()
            if thumb_path and os.path.exists(thumb_path):
                # Load scaled
                img = QImage(thumb_path)
                if not img.isNull():
                    pixmap = QPixmap.fromImage(img).scaled(220, 140, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            else:
                # empty pixmap
                pixmap = QPixmap(220, 140)
                pixmap.fill(Qt.darkGray)

            self.finished.emit(path, name, pixmap)

class ChannelViewerWidget(QWidget):
    def __init__(self, canal_path, parent=None):
        super().__init__(parent)
        self.canal_path = canal_path
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.current_suffix = None
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab { font-size: 12px; padding: 5px 15px; background: #111; color: #888; border: 1px solid #222; }
            QTabBar::tab:selected { background: #333; color: #fff; border-color: #555; font-weight: bold; }
            QTabBar::tab:hover { background: #222; color: #aaa; }
            QTabWidget::pane { border: 1px solid #333; background: #0a0a0a; }
        """)
        self.layout.addWidget(self.tabs)
        
        self.image_labels = {}
        self._setup_ui()
        self._scan_configs()

    def _setup_ui(self):
        # Metadata
        meta_path = os.path.join(self.canal_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                try:
                    md = json.load(f)
                    scroll_meta = QScrollArea()
                    scroll_meta.setWidgetResizable(True)
                    lbl_meta = QLabel(json.dumps(md, indent=2))
                    lbl_meta.setStyleSheet("color: #00ff00; background-color: #0c0c0c; border: 1px solid #333; padding: 15px; font-family: monospace; font-size: 13px;")
                    lbl_meta.setAlignment(Qt.AlignTop | Qt.AlignLeft)
                    scroll_meta.setWidget(lbl_meta)
                    self.tabs.addTab(scroll_meta, "📝 Metadata")
                except: pass

        # Prefix -> Tab Name
        self.img_categories = {
            "avg": "⚡ Promedio",
            "pulses": "✂️ Recortes",
            "spec": "🌈 Espectrograma",
            "evolucion": "📈 Evolución"
        }
        
        for prefix, tab_name in self.img_categories.items():
            scroll_img = QScrollArea()
            scroll_img.setWidgetResizable(True)
            scroll_img.setAlignment(Qt.AlignCenter)
            
            container = QWidget()
            lyt_container = QVBoxLayout(container)
            lyt_container.setAlignment(Qt.AlignCenter)
            
            lbl_hint = QLabel("🔍 Haz clic en la imagen para ampliar")
            lbl_hint.setStyleSheet("color: #888; margin-bottom: 5px;")
            lbl_hint.setAlignment(Qt.AlignCenter)
            
            # Place holder for clickable image
            clickable_img = ClickableImage("", max_width=500)
            self.image_labels[prefix] = clickable_img
            
            lyt_container.addStretch()
            lyt_container.addWidget(lbl_hint)
            lyt_container.addWidget(clickable_img)
            lyt_container.addStretch()
            
            scroll_img.setWidget(container)
            self.tabs.addTab(scroll_img, tab_name)

    def _scan_configs(self):
        pass # Movido a nivel global en ElectrodeViewerWidget

    def set_suffix(self, suffix):
        self.current_suffix = suffix
        for prefix, img_label in self.image_labels.items():
            if suffix == "Antiguo" or suffix is None:
                filename = f"{prefix}.png"
            else:
                filename = f"{prefix}_{suffix}.png"
                
            img_path = os.path.join(self.canal_path, filename)
            if os.path.exists(img_path):
                img_label.img_path = img_path
                img_label.pix = QPixmap(img_path)
                if img_label.pix.width() > 500:
                    img_label.setPixmap(img_label.pix.scaledToWidth(500, Qt.SmoothTransformation))
                else:
                    img_label.setPixmap(img_label.pix)
                img_label.show()
            else:
                # No se encontró esta imagen para esta configuración
                img_label.hide()

    def set_sub_tab(self, tab_text):
        for j in range(self.tabs.count()):
            if self.tabs.tabText(j) == tab_text:
                self.tabs.setCurrentIndex(j)
                break


class ElectrodeViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Header toolbar
        self.toolbar = QHBoxLayout()
        self.lbl_info = QLabel("Mostrando 0 electrodos seleccionados.")
        self.lbl_info.setStyleSheet("color: #00ffaa; font-family: 'Courier New', monospace; font-size: 14px; font-weight: bold;")
        self.btn_refresh = QPushButton("🔄 Sincronizar con Sesiones Marcadas")
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #880000; 
                color: white; 
                padding: 8px 15px; 
                font-weight: bold;
                border-radius: 5px;
                border: 1px solid #ff4444;
            }
            QPushButton:hover {
                background-color: #aa0000;
            }
            QPushButton:pressed {
                background-color: #550000;
            }
        """)
        self.toolbar.addWidget(self.btn_refresh)
        
        self.lbl_info = QLabel("Mostrando 0 electrodos seleccionados.")
        self.lbl_info.setStyleSheet("color: #00ffaa; font-family: 'Courier New', monospace; font-size: 14px; font-weight: bold;")
        self.toolbar.addWidget(self.lbl_info)
        
        self.toolbar.addStretch()
        
        self.lbl_global_config = QLabel("Configuración de Procesamiento:")
        self.lbl_global_config.setStyleSheet("color: #888; font-weight: bold;")
        self.cmb_global_config = QComboBox()
        self.cmb_global_config.setStyleSheet("""
            QComboBox { background-color: #1a1a1a; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 2px 10px; }
            QComboBox::drop-down { border-left: 1px solid #444; }
            QComboBox QAbstractItemView { background-color: #1a1a1a; color: #fff; selection-background-color: #880000; }
        """)
        self.cmb_global_config.currentIndexChanged.connect(self._on_global_config_changed)
        
        self.toolbar.addWidget(self.lbl_global_config)
        self.toolbar.addWidget(self.cmb_global_config)
        
        self.layout.addLayout(self.toolbar)

        self.splitter = QSplitter(Qt.Horizontal)
        
        # Grid
        self.list_widget = QListWidget()
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setIconSize(QSize(220, 140))
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(15)
        self.list_widget.setStyleSheet("""
            QListWidget { 
                background-color: #0a0a0a; 
                border: 1px solid #333; 
                border-radius: 8px;
                padding: 10px;
                outline: none;
            } 
            QListWidget::item { 
                color: #ddd; 
                background-color: #1a1a1a;
                border: 1px solid #222;
                border-radius: 8px;
                padding: 5px;
            }
            QListWidget::item:hover {
                background-color: #2a2a2a;
                border: 1px solid #FF4444;
                color: white;
            }
            QListWidget::item:selected {
                background-color: #440000;
                border: 2px solid #FF0000;
                color: #FFaaaa;
            }
            QScrollBar:vertical {
                border: none;
                background: #111;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background: #FF4444;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.itemClicked.connect(self._on_item_double_clicked) # Click simple también abre
        
        self.splitter.addWidget(self.list_widget)
        
        # Detail Pane
        self.detail_widget = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_widget)
        self.detail_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_detail_title = QLabel("Haz clic en una miniatura para ver los detalles.")
        self.lbl_detail_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF4444; padding: 5px; background-color: #111; border-radius: 4px; border: 1px solid #333;")
        self.detail_layout.addWidget(self.lbl_detail_title)
        
        self.tabs_channels = QTabWidget()
        self.tabs_channels.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #444;
                background: #0f0f0f;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #1a1a1a;
                color: #888;
                padding: 8px 16px;
                border: 1px solid #333;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:hover {
                background: #2a2a2a;
                color: #fff;
            }
            QTabBar::tab:selected {
                background: #880000;
                color: #fff;
                border: 1px solid #ff4444;
                border-bottom: none;
            }
        """)
        self.detail_layout.addWidget(self.tabs_channels)
        
        self.splitter.addWidget(self.detail_widget)
        self.splitter.setSizes([500, 700])
        
        self.layout.addWidget(self.splitter, 1) # STRETCH=1 para evitar espacio vacío arriba
        
        self.current_loader = None
        self.current_paths = []
        self._is_updating_configs = False

    def _update_global_configs(self, paths):
        if not paths: return
        # Escanear el primer path para buscar configuraciones
        self._is_updating_configs = True
        self.cmb_global_config.clear()
        
        path = paths[0]
        c0 = os.path.join(path, "canal_0")
        suffixes = set()
        if os.path.exists(c0):
            for f in os.listdir(c0):
                for prefix in ["avg", "pulses", "spec", "evolucion"]:
                    if f.startswith(prefix + "_") and f.endswith(".png"):
                        suffix = f[len(prefix)+1:-4]
                        suffixes.add(suffix)
                    elif f == prefix + ".png":
                        suffixes.add("Antiguo")
                        
        for s in sorted(list(suffixes)):
            display_name = s.replace('_', ' ').title() if s != "Antiguo" else "Por Defecto (Antiguo)"
            self.cmb_global_config.addItem(display_name, s)
            
        self._is_updating_configs = False
        
    def _on_global_config_changed(self, index):
        if self._is_updating_configs or index < 0: return
        suffix = self.cmb_global_config.itemData(index)
        
        # 1. Update thumbnails
        if hasattr(self, 'current_paths') and self.current_paths:
            self._reload_thumbnails(self.current_paths, suffix)
            
        # 2. Update all active ChannelViewerWidgets in tabs
        for i in range(self.tabs_channels.count()):
            widget = self.tabs_channels.widget(i)
            if isinstance(widget, ChannelViewerWidget):
                widget.set_suffix(suffix)

    def _reload_thumbnails(self, paths, suffix):
        self.list_widget.clear()
        
        if not hasattr(self, '_active_threads'):
            self._active_threads = []
        self._active_threads = [t for t in self._active_threads if t.isRunning()]
            
        self.current_loader = ThumbnailLoader(paths)
        self.current_loader.suffix = suffix
        self._active_threads.append(self.current_loader)
        
        self.current_loader.finished.connect(lambda p, n, px, loader=self.current_loader: self._add_thumbnail(p, n, px, loader))
        self.current_loader.start()

    def load_measurements(self, paths):
        self.list_widget.clear()
        self.tabs_channels.clear()
        self.current_paths = paths
        self.lbl_info.setText(f"Mostrando {len(paths)} electrodos seleccionados.")
        self.lbl_detail_title.setText("Haz clic en una miniatura para ver los detalles.")
        
        self._update_global_configs(paths)
        
        # Load thumbnails with current suffix
        suffix = self.cmb_global_config.itemData(self.cmb_global_config.currentIndex()) if self.cmb_global_config.count() > 0 else None
        self._reload_thumbnails(paths, suffix)

    def _add_thumbnail(self, path, name, pixmap, loader=None):
        if loader is not None and loader is not getattr(self, 'current_loader', None):
            return # Ignorar hilos viejos que terminaron tarde
            
        item = QListWidgetItem()
        item.setIcon(QIcon(pixmap))
        item.setText(name)
        item.setData(Qt.UserRole, path)
        # Centrar el texto debajo del ícono
        item.setTextAlignment(Qt.AlignCenter)
        self.list_widget.addItem(item)

    def _on_item_double_clicked(self, item):
        path = item.data(Qt.UserRole)
        name = item.text()
        self.lbl_detail_title.setText(f"Medición: {name}")
        
        # 0. Guardar estado actual de las pestañas
        current_main_tab_text = None
        current_sub_tab_text = getattr(self, '_last_sub_tab_text', None)
        
        idx_main = self.tabs_channels.currentIndex()
        if idx_main >= 0:
            current_main_tab_text = self.tabs_channels.tabText(idx_main)
            widget_main = self.tabs_channels.widget(idx_main)
            if isinstance(widget_main, QTabWidget):
                idx_sub = widget_main.currentIndex()
                if idx_sub >= 0:
                    current_sub_tab_text = widget_main.tabText(idx_sub)
                    self._last_sub_tab_text = current_sub_tab_text

        self.tabs_channels.clear()
        
        # 1. Pestaña Señales Musculares (General)
        calibrated_plot = os.path.join(path, f"plot_calibrado_{name}.png")
        if os.path.exists(calibrated_plot):
            tab = QWidget()
            t_layout = QVBoxLayout(tab)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            s_content = QWidget()
            s_layout = QVBoxLayout(s_content)
            
            lbl_img = QLabel()
            pix = QPixmap(calibrated_plot)
            lbl_img.setPixmap(pix.scaledToWidth(800, Qt.SmoothTransformation))
            s_layout.addWidget(lbl_img)
            s_layout.addStretch()
            scroll.setWidget(s_content)
            t_layout.addWidget(scroll)
            self.tabs_channels.addTab(tab, "Señales Musculares")
            
        # 2. Pestañas por canal
        canales = sorted([d for d in os.listdir(path) if d.startswith("canal_") and os.path.isdir(os.path.join(path, d))])
        for canal in canales:
            canal_path = os.path.join(path, canal)
            viewer_widget = ChannelViewerWidget(canal_path)
            # Aplicar sufijo global
            global_suffix = self.cmb_global_config.itemData(self.cmb_global_config.currentIndex())
            if global_suffix:
                viewer_widget.set_suffix(global_suffix)
            
            if current_sub_tab_text:
                viewer_widget.set_sub_tab(current_sub_tab_text)
                
            self.tabs_channels.addTab(viewer_widget, canal.upper())
            
        # Restaurar la pestaña principal
        if current_main_tab_text:
            for i in range(self.tabs_channels.count()):
                if self.tabs_channels.tabText(i) == current_main_tab_text:
                    self.tabs_channels.setCurrentIndex(i)
                    break
