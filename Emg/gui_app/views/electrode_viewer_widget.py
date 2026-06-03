import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSplitter, 
    QListWidget, QListWidgetItem, QTabWidget, QScrollArea,
    QPushButton, QHBoxLayout
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
        dialog.setWindowTitle("Visor de Imagen")
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

class ElectrodeViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        
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
        self.toolbar.addWidget(self.lbl_info)
        self.toolbar.addStretch()
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
        
        self.layout.addWidget(self.splitter)
        
        self.current_loader = None

    def load_measurements(self, paths):
        self.list_widget.clear()
        self.tabs_channels.clear()
        self.lbl_info.setText(f"Mostrando {len(paths)} electrodos seleccionados.")
        self.lbl_detail_title.setText("Haz clic en una miniatura para ver los detalles.")
        
        if self.current_loader and self.current_loader.isRunning():
            self.current_loader.terminate()
            
        self.current_loader = ThumbnailLoader(paths)
        self.current_loader.finished.connect(self._add_thumbnail)
        self.current_loader.start()

    def _add_thumbnail(self, path, name, pixmap):
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
            canal_tab = QTabWidget()
            canal_tab.setStyleSheet("""
                QTabBar::tab { font-size: 12px; padding: 5px 15px; background: #111; color: #888; border: 1px solid #222; }
                QTabBar::tab:selected { background: #333; color: #fff; border-color: #555; font-weight: bold; }
                QTabBar::tab:hover { background: #222; color: #aaa; }
                QTabWidget::pane { border: 1px solid #333; background: #0a0a0a; }
            """)
            
            # Read metadata
            meta_path = os.path.join(canal_path, "metadata.json")
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
                        canal_tab.addTab(scroll_meta, "📝 Metadata")
                    except: pass
                    
            # Nombres bonitos para subpestañas
            img_tabs = {
                "avg.png": "⚡ Promedio",
                "pulses.png": "✂️ Recortes",
                "spec.png": "🌈 Espectrograma",
                "evolucion.png": "📈 Evolución"
            }
            
            # Load images into sub-tabs
            for img_name, tab_name in img_tabs.items():
                img_path = os.path.join(canal_path, img_name)
                if os.path.exists(img_path):
                    scroll_img = QScrollArea()
                    scroll_img.setWidgetResizable(True)
                    scroll_img.setAlignment(Qt.AlignCenter)
                    
                    # Hint label
                    lbl_hint = QLabel("🔍 Haz clic en la imagen para ampliar")
                    lbl_hint.setStyleSheet("color: #888; margin-bottom: 5px;")
                    lbl_hint.setAlignment(Qt.AlignCenter)
                    
                    # Contenedor centralizado
                    container = QWidget()
                    lyt_container = QVBoxLayout(container)
                    lyt_container.setAlignment(Qt.AlignCenter)
                    
                    # Custom Clickable Image (max_width 500 para hacerla chica inicialmente)
                    clickable_img = ClickableImage(img_path, max_width=500)
                    
                    lyt_container.addStretch()
                    lyt_container.addWidget(lbl_hint)
                    lyt_container.addWidget(clickable_img)
                    lyt_container.addStretch()
                    
                    scroll_img.setWidget(container)
                    canal_tab.addTab(scroll_img, tab_name)
                    
            # Forzar sub-pestaña si se había guardado el estado
            if current_sub_tab_text:
                for j in range(canal_tab.count()):
                    if canal_tab.tabText(j) == current_sub_tab_text:
                        canal_tab.setCurrentIndex(j)
                        break
                        
            self.tabs_channels.addTab(canal_tab, canal.upper())
            
        # Restaurar la pestaña principal
        if current_main_tab_text:
            for i in range(self.tabs_channels.count()):
                if self.tabs_channels.tabText(i) == current_main_tab_text:
                    self.tabs_channels.setCurrentIndex(i)
                    break
