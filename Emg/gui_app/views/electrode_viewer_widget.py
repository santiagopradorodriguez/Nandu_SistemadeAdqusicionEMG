import os
import json
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSplitter, 
    QListWidget, QListWidgetItem, QTabWidget, QScrollArea,
    QPushButton, QHBoxLayout
)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QIcon, QPixmap, QImage

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
        self.lbl_info.setStyleSheet("color: #888; font-family: monospace;")
        self.btn_refresh = QPushButton("🔄 Sincronizar con Sesiones Marcadas")
        self.btn_refresh.setStyleSheet("background-color: #333; color: white; padding: 5px 15px; font-weight: bold;")
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
        self.list_widget.setSpacing(10)
        self.list_widget.setStyleSheet("QListWidget { background-color: #111; border: 1px solid #333; } QListWidget::item { color: white; }")
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.list_widget.itemClicked.connect(self._on_item_double_clicked) # Click simple también abre
        
        self.splitter.addWidget(self.list_widget)
        
        # Detail Pane
        self.detail_widget = QWidget()
        self.detail_layout = QVBoxLayout(self.detail_widget)
        self.lbl_detail_title = QLabel("Haz clic en una miniatura para ver los detalles.")
        self.lbl_detail_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF4444;")
        self.detail_layout.addWidget(self.lbl_detail_title)
        
        self.tabs_channels = QTabWidget()
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
            canal_tab.setStyleSheet("QTabBar::tab { font-size: 12px; padding: 5px 10px; }")
            
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
                    
                    lbl_img = QLabel()
                    lbl_img.setAlignment(Qt.AlignCenter)
                    pix = QPixmap(img_path)
                    
                    # Cargar a tamaño decente, el scrollarea hara el resto si es muy grande
                    if pix.width() > 900:
                        lbl_img.setPixmap(pix.scaledToWidth(900, Qt.SmoothTransformation))
                    else:
                        lbl_img.setPixmap(pix)
                        
                    scroll_img.setWidget(lbl_img)
                    canal_tab.addTab(scroll_img, tab_name)
                    
            self.tabs_channels.addTab(canal_tab, canal.upper())
