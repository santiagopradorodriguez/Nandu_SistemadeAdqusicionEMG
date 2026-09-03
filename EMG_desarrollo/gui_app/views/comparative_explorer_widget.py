# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para exploración comparativa de sesiones.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para exploración comparativa de sesiones.
# ==============================================================================

import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSizePolicy,
    QSplitter, QScrollArea, QTabWidget, QDialog
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QCursor

class ClickableImage(QLabel):
    def __init__(self, img_path, max_width=500, parent=None):
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

class ComparativeViewerWidget(QWidget):
    def __init__(self, root_path, parent=None):
        super().__init__(parent)
        self.root_path = root_path
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)
        
        # Toolbar
        self.toolbar = QHBoxLayout()
        lbl_title = QLabel(" Explorador de Análisis Comparativos")
        lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #FF4444;")
        
        btn_refresh = QPushButton(" Actualizar Directorio")
        btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #880000; color: white; padding: 5px 15px; font-weight: bold; border-radius: 4px; border: 1px solid #ff4444;
            }
            QPushButton:hover { background-color: #aa0000; }
        """)
        btn_refresh.clicked.connect(self.cargar_arbol)

        btn_export_pdf = QPushButton(" Exportar Reporte PDF")
        btn_export_pdf.setStyleSheet("""
            QPushButton {
                background-color: #005533; color: #00ff88; padding: 5px 15px; font-weight: bold; border-radius: 4px; border: 1px solid #00ff88;
            }
            QPushButton:hover { background-color: #007744; }
        """)
        btn_export_pdf.clicked.connect(self._on_export_pdf)
        
        self.toolbar.addWidget(lbl_title)
        self.toolbar.addStretch()
        self.toolbar.addWidget(btn_refresh)
        self.toolbar.addWidget(btn_export_pdf)
        self.layout.addLayout(self.toolbar)
        
        # Splitter principal
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Panel Izquierdo: Árbol de Experimentos
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setStyleSheet("""
            QTreeWidget {
                background-color: #0a0a0a; color: white; border: 1px solid #333; font-size: 13px; border-radius: 8px; padding: 5px; outline: none;
            }
            QTreeWidget::item { padding: 5px; border-radius: 4px; }
            QTreeWidget::item:hover { background-color: #1a1a1a; border: 1px solid #555; }
            QTreeWidget::item:selected { background-color: #440000; color: #ffaaaa; border: 1px solid #FF0000; }
            QTreeWidget::branch { background-color: transparent; }
        """)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.splitter.addWidget(self.tree)
        
        # Panel Derecho: Visor de Contenido
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_experimento_title = QLabel("Selecciona un experimento del panel izquierdo para ver los resultados.")
        self.lbl_experimento_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffaa; padding: 10px; background-color: #111; border: 1px solid #333; border-radius: 5px;")
        self.content_layout.addWidget(self.lbl_experimento_title)
        
        self.tabs_graficos = QTabWidget()
        self.tabs_graficos.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; background: #0f0f0f; border-radius: 4px; }
            QTabBar::tab { background: #1a1a1a; color: #888; padding: 8px 16px; border: 1px solid #333; border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px; margin-right: 2px; font-weight: bold; }
            QTabBar::tab:hover { background: #2a2a2a; color: #fff; }
            QTabBar::tab:selected { background: #880000; color: #fff; border: 1px solid #ff4444; border-bottom: none; }
        """)
        self.content_layout.addWidget(self.tabs_graficos)
        
        self.splitter.addWidget(self.content_widget)
        self.splitter.setSizes([250, 950])
        self.layout.addWidget(self.splitter)
        
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
            fecha_item = QTreeWidgetItem(self.tree, [f" {fecha}"])
            fecha_item.setData(0, Qt.UserRole, fecha_path)
            fecha_item.setData(0, Qt.UserRole + 1, "fecha")
            fecha_item.setExpanded(True)
            
            experimentos = sorted([d for d in os.listdir(fecha_path) if os.path.isdir(os.path.join(fecha_path, d))])
            for exp in experimentos:
                exp_path = os.path.join(fecha_path, exp)
                exp_item = QTreeWidgetItem(fecha_item, [f" {exp}"])
                exp_item.setData(0, Qt.UserRole, exp_path)
                exp_item.setData(0, Qt.UserRole + 1, "experimento")
                exp_item.setExpanded(True)
                
                # Check for subdirectories inside experiments
                sub_items = sorted([d for d in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, d))])
                for sub in sub_items:
                     sub_path = os.path.join(exp_path, sub)
                     sub_item = QTreeWidgetItem(exp_item, [f" {sub}"])
                     sub_item.setData(0, Qt.UserRole, sub_path)
                     sub_item.setData(0, Qt.UserRole + 1, "experimento")

    def _on_item_clicked(self, item, column):
        tipo = item.data(0, Qt.UserRole + 1)
        ruta = item.data(0, Qt.UserRole)
        
        if tipo == "experimento":
            self.lbl_experimento_title.setText(f"Resultados: {item.text(0)}")
            self.cargar_experimento(ruta)
        elif tipo == "fecha":
            self.lbl_experimento_title.setText(f"Carpeta de fecha: {item.text(0)} (Selecciona un experimento)")
            self.tabs_graficos.clear()
            
    def cargar_experimento(self, path):
        self.tabs_graficos.clear()
        
        if not os.path.exists(path):
            return
            
        archivos = os.listdir(path)
        imagenes = [f for f in archivos if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        textos = [f for f in archivos if f.lower().endswith(('.csv', '.tex', '.json', '.txt'))]
        
        if imagenes:
            tab_imgs = QWidget()
            lyt_imgs = QVBoxLayout(tab_imgs)
            
            scroll_imgs = QScrollArea()
            scroll_imgs.setWidgetResizable(True)
            scroll_imgs.setStyleSheet("QScrollArea { border: none; background-color: #0c0c0c; }")
            
            container_imgs = QWidget()
            lyt_container = QVBoxLayout(container_imgs)
            lyt_container.setAlignment(Qt.AlignTop | Qt.AlignCenter)
            
            for img_name in sorted(imagenes):
                img_path = os.path.join(path, img_name)
                
                lbl_title = QLabel(f" {img_name}")
                lbl_title.setStyleSheet("color: #ffaa00; font-weight: bold; margin-top: 15px; font-size: 14px;")
                lbl_title.setAlignment(Qt.AlignCenter)
                lyt_container.addWidget(lbl_title)
                
                lbl_hint = QLabel(" Haz clic en la imagen para ampliar")
                lbl_hint.setStyleSheet("color: #888; margin-bottom: 5px;")
                lbl_hint.setAlignment(Qt.AlignCenter)
                lyt_container.addWidget(lbl_hint)
                
                clickable_img = ClickableImage(img_path, max_width=500)
                lyt_container.addWidget(clickable_img)
                
            lyt_container.addStretch()
            scroll_imgs.setWidget(container_imgs)
            lyt_imgs.addWidget(scroll_imgs)
            
            self.tabs_graficos.addTab(tab_imgs, " Gráficos Generados")
            
        if textos:
            tab_txt = QWidget()
            lyt_txt = QVBoxLayout(tab_txt)
            
            scroll_txt = QScrollArea()
            scroll_txt.setWidgetResizable(True)
            scroll_txt.setStyleSheet("QScrollArea { border: none; background-color: #0c0c0c; }")
            
            container_txt = QWidget()
            lyt_container_txt = QVBoxLayout(container_txt)
            lyt_container_txt.setAlignment(Qt.AlignTop)
            
            for txt_name in sorted(textos):
                txt_path = os.path.join(path, txt_name)
                
                lbl_title = QLabel(f" {txt_name}")
                lbl_title.setStyleSheet("color: #00ffaa; font-weight: bold; margin-top: 15px; font-size: 14px;")
                lyt_container_txt.addWidget(lbl_title)
                
                lbl_content = QLabel()
                lbl_content.setStyleSheet("background-color: #111; color: #ddd; padding: 15px; border: 1px solid #333; font-family: 'Courier New', monospace; border-radius: 4px;")
                lbl_content.setWordWrap(True)
                lbl_content.setTextInteractionFlags(Qt.TextSelectableByMouse)
                
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        contenido = f.read()
                        if len(contenido) > 3000:
                            contenido = contenido[:3000] + "\n... [Mostrando solo los primeros 3000 caracteres]"
                        lbl_content.setText(contenido)
                except Exception as e:
                    lbl_content.setText(f"No se pudo leer el archivo: {e}")
                    
                lyt_container_txt.addWidget(lbl_content)
                
            lyt_container_txt.addStretch()
            scroll_txt.setWidget(container_txt)
            lyt_txt.addWidget(scroll_txt)
            
            self.tabs_graficos.addTab(tab_txt, " Datos (CSV/Tex)")

    def _on_export_pdf(self):
        item = self.tree.currentItem()
        if not item:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Sin Selección", "Selecciona una fecha o experimento en el árbol para exportar el reporte.")
            return
            
        ruta = item.data(0, Qt.UserRole)
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', str(ruta))
        if not date_match:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Fecha no detectada", "No se pudo identificar la fecha de la sesión seleccionada.")
            return
            
        fecha_str = date_match.group(0)
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base_datos_fecha = os.path.join(repo_root, "base_de_datos_electrodos", fecha_str)
        
        session_paths = []
        if os.path.isdir(base_datos_fecha):
            session_paths = [
                os.path.join(base_datos_fecha, d) 
                for d in sorted(os.listdir(base_datos_fecha)) 
                if os.path.isdir(os.path.join(base_datos_fecha, d))
            ]
            
        if not session_paths:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, 
                "Sin Mediciones", 
                f"No se encontraron carpetas de mediciones en base_de_datos_electrodos/{fecha_str}."
            )
            return
            
        from views.report_dialog import ReportDialog
        dialog = ReportDialog(session_paths, parent=self)
        dialog.exec()
