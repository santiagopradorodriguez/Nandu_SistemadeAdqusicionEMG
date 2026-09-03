# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo para la configuración y generación modular de reportes integrales en PDF.
# ==============================================================================

import os
import sys
import subprocess
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QTextEdit, QPushButton, QCheckBox,
    QMessageBox, QScrollArea, QWidget, QFileDialog, QListWidget,
    QProgressBar
)
from PySide6.QtCore import Qt, QThread, Signal

# Importar el motor de reportes
script_dir = os.path.dirname(os.path.abspath(__file__))
gui_app_dir = os.path.dirname(script_dir)
emg_root = os.path.dirname(gui_app_dir)
if emg_root not in sys.path:
    sys.path.append(emg_root)

from analysis.report_engine import ReportEngine

class ReportWorker(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(dict)

    def __init__(self, engine, session_paths, notes_dict, mode='main'):
        super().__init__()
        self.engine = engine
        self.session_paths = session_paths
        self.notes_dict = notes_dict
        self.mode = mode

    def run(self):
        try:
            import importlib
            import analysis.report_engine as r_mod
            importlib.reload(r_mod)
            engine = r_mod.ReportEngine()
            if self.mode == 'snr':
                result = engine.generate_snr_report(
                    self.session_paths, 
                    self.notes_dict,
                    logger=lambda msg: self.progress_signal.emit(str(msg))
                )
            else:
                result = engine.generate_report(
                    self.session_paths, 
                    self.notes_dict,
                    logger=lambda msg: self.progress_signal.emit(str(msg))
                )
            self.finished_signal.emit(result)
        except Exception as e:
            self.finished_signal.emit({
                'status': 'error',
                'tex_path': None,
                'pdf_path': None,
                'error': str(e)
            })

class ReportDialog(QDialog):
    def __init__(self, session_paths, parent=None):
        super().__init__(parent)
        self.session_paths = session_paths
        self.engine = ReportEngine()
        self.meta_info = self.engine.extract_session_metadata(session_paths)
        self.fotos_seleccionadas = []
        
        self.setWindowTitle("Generador de Reporte Integral de Sesión (LaTeX / PDF)")
        self.setMinimumSize(720, 750)
        self.resize(760, 800)
        
        self.setStyleSheet("""
            QDialog {
                background-color: #121212;
                color: #e0e0e0;
                font-family: sans-serif;
            }
            QGroupBox {
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
                color: #00ffcc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit, QTextEdit, QListWidget {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border: 1px solid #00ffcc;
            }
            QLabel {
                color: #cccccc;
            }
            QPushButton {
                background-color: #222222;
                color: #ffffff;
                border: 1px solid #555555;
                padding: 7px 14px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #333333;
                border-color: #00ffcc;
            }
        """)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # Encabezado informativo
        lbl_title = QLabel("Reporte Técnico Integral de Adquisición, Señales y PCA")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffcc;")
        main_layout.addWidget(lbl_title)

        med_count = len(self.session_paths)
        lbl_sub = QLabel(f"Mediciones seleccionadas: {med_count} | Fecha: {self.meta_info['fecha']} | Sujeto: {self.meta_info['sujeto']}")
        lbl_sub.setStyleSheet("color: #888888; font-size: 12px; margin-bottom: 5px;")
        main_layout.addWidget(lbl_sub)

        # Scroll Area para el formulario
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(12)

        # 1. Configuración de Hardware
        gb_hw = QGroupBox("1. Hardware y Referencia")
        l_hw = QFormLayout(gb_hw)
        l_hw.setLabelAlignment(Qt.AlignRight)
        
        self.inp_fecha = QLineEdit(self.meta_info['fecha'])
        self.inp_sujeto = QLineEdit(self.meta_info['sujeto'])
        self.inp_baterias = QLineEdit("8.30 V y 8.20 V")
        self.inp_tierra = QLineEdit("Frente")
        
        l_hw.addRow("Fecha:", self.inp_fecha)
        l_hw.addRow("Sujeto:", self.inp_sujeto)
        l_hw.addRow("Tensión Baterías:", self.inp_baterias)
        l_hw.addRow("Electrodo de Tierra:", self.inp_tierra)
        form_layout.addWidget(gb_hw)

        # 2. Canales y Músculos Medidos
        gb_ch = QGroupBox("2. Mapeo de Canales y Músculos Medidos")
        l_ch = QFormLayout(gb_ch)
        l_ch.setLabelAlignment(Qt.AlignRight)
        
        self.inp_ch0 = QLineEdit(self.meta_info['canales'].get(0, "Canal 0"))
        self.inp_ch1 = QLineEdit(self.meta_info['canales'].get(1, "Canal 1"))
        self.inp_ch2 = QLineEdit(self.meta_info['canales'].get(2, "Canal 2"))
        self.inp_ch3 = QLineEdit(self.meta_info['canales'].get(3, "Canal 3 (Mic/Ref)"))
        
        l_ch.addRow("Canal 0:", self.inp_ch0)
        l_ch.addRow("Canal 1:", self.inp_ch1)
        l_ch.addRow("Canal 2:", self.inp_ch2)
        l_ch.addRow("Canal 3 (Mic):", self.inp_ch3)
        form_layout.addWidget(gb_ch)

        # 3. Fotografías de la Sesión
        gb_photos = QGroupBox("3. Fotografías de la Sesión y Montaje (Opcional)")
        l_photos = QVBoxLayout(gb_photos)
        
        self.list_photos = QListWidget()
        self.list_photos.setFixedHeight(65)
        l_photos.addWidget(self.list_photos)
        
        h_btn_photos = QHBoxLayout()
        btn_add_photo = QPushButton("Adjuntar Fotografías...")
        btn_add_photo.clicked.connect(self.adjuntar_fotos)
        btn_clear_photos = QPushButton("Limpiar")
        btn_clear_photos.clicked.connect(self.limpiar_fotos)
        h_btn_photos.addWidget(btn_add_photo)
        h_btn_photos.addWidget(btn_clear_photos)
        h_btn_photos.addStretch()
        l_photos.addLayout(h_btn_photos)
        form_layout.addWidget(gb_photos)

        # 4. Textos y Notas de Sesión
        gb_notes = QGroupBox("4. Notas de Montaje y Protocolo")
        l_notes = QVBoxLayout(gb_notes)
        
        l_notes.addWidget(QLabel("Armado y Fijación de Electrodos:"))
        self.txt_electrodos = QTextEdit()
        self.txt_electrodos.setPlaceholderText("Ej: Electrodos recortados con plancha de gel y doble cinta de refuerzo...")
        self.txt_electrodos.setFixedHeight(50)
        l_notes.addWidget(self.txt_electrodos)
        
        l_notes.addWidget(QLabel("Secuencia y Protocolo:"))
        self.txt_secuencia = QTextEdit()
        self.txt_secuencia.setPlaceholderText("Ej: Protocolo de repeticiones por vocal (A, E, I, O, U) en series consecutivas...")
        self.txt_secuencia.setFixedHeight(50)
        l_notes.addWidget(self.txt_secuencia)
        
        l_notes.addWidget(QLabel("Observaciones y Registro de Artefactos:"))
        self.txt_observaciones = QTextEdit()
        self.txt_observaciones.setPlaceholderText("Ej: Picos involuntarios por deglución...")
        self.txt_observaciones.setFixedHeight(50)
        l_notes.addWidget(self.txt_observaciones)
        form_layout.addWidget(gb_notes)

        scroll.setWidget(form_container)
        main_layout.addWidget(scroll)

        # Consola de estado / progreso
        self.lbl_status = QLabel("Listo para generar.")
        self.lbl_status.setStyleSheet("color: #00ffcc; font-weight: bold; font-family: monospace;")
        main_layout.addWidget(self.lbl_status)

        # Botones inferiores modulares
        btn_layout = QHBoxLayout()

        self.btn_cancel = QPushButton("Cerrar")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_open_pdf = QPushButton("Abrir PDF")
        self.btn_open_pdf.setEnabled(False)
        self.btn_open_pdf.setStyleSheet("""
            QPushButton {
                background-color: #004444; color: #00ffff; border: 1px solid #00ffff;
                padding: 7px 14px;
            }
            QPushButton:hover { background-color: #006666; }
        """)
        self.btn_open_pdf.clicked.connect(self.abrir_pdf)
        btn_layout.addWidget(self.btn_open_pdf)

        btn_layout.addStretch()

        # Botón 1: Generar Reporte Base (Rápido)
        self.btn_base = QPushButton("1. Generar Reporte Base (Rápido)")
        self.btn_base.setToolTip("Compila instantáneamente las Secciones 1 a 4 (Metadatos, Señales, Evolución y Patrones) para previsualizar el documento.")
        self.btn_base.setStyleSheet("""
            QPushButton {
                background-color: #1a3a5a; color: #66FCF1; border: 1px solid #66FCF1;
                padding: 8px 14px; font-size: 12px;
            }
            QPushButton:hover { background-color: #2a5a8a; }
        """)
        self.btn_base.clicked.connect(lambda: self.lanzar_generacion(incluir_pca=False))
        btn_layout.addWidget(self.btn_base)

        # Botón 2: Generar/Anexar Sección PCA
        self.btn_pca = QPushButton("2. Generar y Anexar Sección PCA")
        self.btn_pca.setToolTip("Ejecuta el Grid Search 2D/3D completo con Fronteras de Decisión y anexa la Sección 5 de PCA al PDF.")
        self.btn_pca.setStyleSheet("""
            QPushButton {
                background-color: #006644; color: #ffffff; border: 1px solid #00ff88;
                padding: 8px 16px; font-size: 12px;
            }
            QPushButton:hover { background-color: #008855; }
        """)
        self.btn_pca.clicked.connect(lambda: self.lanzar_generacion(incluir_pca=True))
        btn_layout.addWidget(self.btn_pca)

        # Botón 3: Generar Reporte de SNR y Calidad (Separado)
        self.btn_snr = QPushButton("3. Reporte de SNR y Calidad")
        self.btn_snr.setToolTip("Genera un documento PDF independiente enfocado exclusivamente en métricas de SNR y ruido interpulso.")
        self.btn_snr.setStyleSheet("""
            QPushButton {
                background-color: #4a2a00; color: #ffbb66; border: 1px solid #ffbb66;
                padding: 8px 14px; font-size: 12px;
            }
            QPushButton:hover { background-color: #6a3a00; }
        """)
        self.btn_snr.clicked.connect(self.lanzar_generacion_snr)
        btn_layout.addWidget(self.btn_snr)

        main_layout.addLayout(btn_layout)
        self.generated_pdf_path = None

    def adjuntar_fotos(self):
        archivos, _ = QFileDialog.getOpenFileNames(
            self, 
            "Seleccionar Fotografías de la Sesión", 
            "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        if archivos:
            for f in archivos:
                if f not in self.fotos_seleccionadas:
                    self.fotos_seleccionadas.append(f)
                    self.list_photos.addItem(os.path.basename(f))

    def limpiar_fotos(self):
        self.fotos_seleccionadas.clear()
        self.list_photos.clear()

    def lanzar_generacion(self, incluir_pca=False):
        notes_dict = {
            'fecha': self.inp_fecha.text().strip(),
            'sujeto': self.inp_sujeto.text().strip(),
            'baterias': self.inp_baterias.text().strip(),
            'tierra': self.inp_tierra.text().strip(),
            'electrodos_nota': self.txt_electrodos.toPlainText().strip(),
            'secuencia': self.txt_secuencia.toPlainText().strip(),
            'notas': self.txt_observaciones.toPlainText().strip(),
            'fotos': self.fotos_seleccionadas,
            'incluir_pca': incluir_pca,
            'ejecutar_grid': True,
            'canales': {
                0: self.inp_ch0.text().strip(),
                1: self.inp_ch1.text().strip(),
                2: self.inp_ch2.text().strip(),
                3: self.inp_ch3.text().strip()
            }
        }

        self.btn_base.setEnabled(False)
        self.btn_pca.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        if hasattr(self, 'btn_snr'): self.btn_snr.setEnabled(False)
        
        tipo_str = "Reporte Completo con PCA (Grid Search)" if incluir_pca else "Reporte Base (Señales y Evolución)"
        self.lbl_status.setText(f"Iniciando compilación: {tipo_str}...")

        self.worker = ReportWorker(self.engine, self.session_paths, notes_dict, mode='main')
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def lanzar_generacion_snr(self):
        notes_dict = {
            'fecha': self.inp_fecha.text().strip(),
            'sujeto': self.inp_sujeto.text().strip(),
            'canales': {
                0: self.inp_ch0.text().strip(),
                1: self.inp_ch1.text().strip(),
                2: self.inp_ch2.text().strip(),
                3: self.inp_ch3.text().strip()
            }
        }

        self.btn_base.setEnabled(False)
        self.btn_pca.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        if hasattr(self, 'btn_snr'): self.btn_snr.setEnabled(False)

        self.lbl_status.setText("Iniciando compilación: Reporte de SNR y Calidad...")

        self.worker = ReportWorker(self.engine, self.session_paths, notes_dict, mode='snr')
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_generation_finished)
        self.worker.start()

    def on_progress(self, msg):
        self.lbl_status.setText(msg[:95] + ("..." if len(msg) > 95 else ""))

    def on_generation_finished(self, result):
        self.btn_base.setEnabled(True)
        self.btn_pca.setEnabled(True)
        self.btn_cancel.setEnabled(True)
        if hasattr(self, 'btn_snr'): self.btn_snr.setEnabled(True)

        if result['status'] == 'success':
            self.generated_pdf_path = result['pdf_path']
            self.lbl_status.setText(f"PDF generado: {os.path.basename(self.generated_pdf_path)}")
            self.btn_open_pdf.setEnabled(True)
            QMessageBox.information(
                self, 
                "Reporte Generado con Éxito", 
                f"El reporte PDF fue generado y compilado en:\n\n{self.generated_pdf_path}"
            )
        else:
            self.lbl_status.setText("Error en la compilación del reporte.")
            err_msg = result.get('error', 'Error desconocido al invocar pdflatex.')
            tex_file = result.get('tex_path', 'N/A')
            QMessageBox.critical(
                self, 
                "Error al Generar Reporte", 
                f"Ocurrió un error al compilar el PDF con LaTeX:\n\n{err_msg}\n\nEl archivo .tex se encuentra en:\n{tex_file}"
            )

    def abrir_pdf(self):
        if self.generated_pdf_path and os.path.exists(self.generated_pdf_path):
            if sys.platform.startswith('linux'):
                subprocess.Popen(['xdg-open', self.generated_pdf_path])
            elif sys.platform == 'win32':
                os.startfile(self.generated_pdf_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.generated_pdf_path])
