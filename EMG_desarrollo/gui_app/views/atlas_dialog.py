# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Diálogo interactivo para configurar y compilar el Atlas sEMG en PDF.
# ==============================================================================

import os
import sys
import subprocess
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QCheckBox, QComboBox, QSpinBox,
    QProgressBar, QMessageBox, QFileDialog, QWidget
)
from PySide6.QtCore import Qt, QThread, Signal

# Asegurar importación del backend
script_dir = os.path.dirname(os.path.abspath(__file__))
gui_app_dir = os.path.dirname(script_dir)
emg_root = os.path.dirname(gui_app_dir)
if emg_root not in sys.path:
    sys.path.insert(0, emg_root)

from analysis.generador_atlas_pdf import GeneradorAtlasPDF, CATALOGO_SUJETOS

class AtlasWorker(QThread):
    progress_signal = Signal(str, int)
    finished_signal = Signal(dict)

    def __init__(self, ruta_pdf, incluir_foto, tema, sujetos, filas_por_pagina, base_db=None):
        super().__init__()
        self.ruta_pdf = ruta_pdf
        self.incluir_foto = incluir_foto
        self.tema = tema
        self.sujetos = sujetos
        self.filas_por_pagina = filas_por_pagina
        self.base_db = base_db

    def run(self):
        try:
            generador = GeneradorAtlasPDF(base_db=self.base_db)
            def cb(msg, pct):
                self.progress_signal.emit(msg, pct)

            pdf_generado = generador.generar_atlas(
                ruta_pdf=self.ruta_pdf,
                incluir_foto=self.incluir_foto,
                tema=self.tema,
                sujetos_filtro=self.sujetos,
                filas_por_pagina=self.filas_por_pagina,
                progress_cb=cb
            )
            self.finished_signal.emit({
                'success': True,
                'pdf_path': pdf_generado,
                'error': None
            })
        except Exception as e:
            self.finished_signal.emit({
                'success': False,
                'pdf_path': None,
                'error': str(e)
            })

class AtlasDialog(QDialog):
    def __init__(self, base_db=None, parent=None):
        super().__init__(parent)
        self.base_db = base_db
        self.ultimo_pdf = None

        self.setWindowTitle("Generador del Atlas de Activación sEMG (PDF)")
        self.setMinimumSize(680, 560)
        self.resize(720, 600)

        self.setStyleSheet("""
            QDialog {
                background-color: #0c0f17;
                color: #e2e8f0;
                font-family: sans-serif;
            }
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 14px;
                font-weight: bold;
                color: #00ffcc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QLineEdit, QComboBox, QSpinBox {
                background-color: #151d2a;
                color: #ffffff;
                border: 1px solid #334155;
                padding: 6px;
                border-radius: 4px;
            }
            QCheckBox {
                color: #f1f5f9;
                font-size: 13px;
                spacing: 6px;
            }
            QCheckBox::indicator {
                border: 1px solid #38bdf8;
                width: 16px;
                height: 16px;
                background: #0f172a;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background: #00ffcc;
            }
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #151d2a;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #00ffcc;
                border-radius: 3px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # 1. OPCIONES DE DISEÑO Y CONTENIDO
        g_opciones = QGroupBox("1. Opciones de Formato y Presentación")
        l_opciones = QVBoxLayout(g_opciones)

        self.chk_incluir_foto = QCheckBox("Incluir fotografía de colocación de electrodos (en el lateral si está disponible)")
        self.chk_incluir_foto.setChecked(True)
        l_opciones.addWidget(self.chk_incluir_foto)

        h_params = QHBoxLayout()
        h_params.addWidget(QLabel("Estilo visual:"))
        self.cmb_tema = QComboBox()
        self.cmb_tema.addItem("Publicación Científica (Fondo Blanco)", "publicacion")
        self.cmb_tema.addItem("Modo Oscuro (Cyberpunk)", "oscuro")
        h_params.addWidget(self.cmb_tema, stretch=1)

        h_params.addWidget(QLabel("Filas por página:"))
        self.spn_filas = QSpinBox()
        self.spn_filas.setRange(1, 4)
        self.spn_filas.setValue(2)
        h_params.addWidget(self.spn_filas)
        l_opciones.addLayout(h_params)
        layout.addWidget(g_opciones)

        # 2. SELECCIÓN DE SUJETOS
        g_sujetos = QGroupBox("2. Sujetos a Incluir en el Atlas")
        l_sujetos = QVBoxLayout(g_sujetos)

        h_chks = QHBoxLayout()
        self.chks_sujetos = {}
        todos_sujetos = [b['sujeto'] for b in CATALOGO_SUJETOS]
        for s in todos_sujetos:
            chk = QCheckBox(s)
            chk.setChecked(True)
            self.chks_sujetos[s] = chk
            h_chks.addWidget(chk)

        l_sujetos.addLayout(h_chks)

        h_btn_sel = QHBoxLayout()
        btn_sel_todos = QPushButton("Seleccionar Todos")
        btn_sel_todos.setStyleSheet("background-color: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 4px 8px; border-radius: 3px;")
        btn_sel_todos.clicked.connect(lambda: [c.setChecked(True) for c in self.chks_sujetos.values()])
        h_btn_sel.addWidget(btn_sel_todos)

        btn_desel = QPushButton("Deseleccionar Todos")
        btn_desel.setStyleSheet("background-color: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 4px 8px; border-radius: 3px;")
        btn_desel.clicked.connect(lambda: [c.setChecked(False) for c in self.chks_sujetos.values()])
        h_btn_sel.addWidget(btn_desel)
        h_btn_sel.addStretch()
        l_sujetos.addLayout(h_btn_sel)
        layout.addWidget(g_sujetos)

        # 3. DESTINO DEL ARCHIVO PDF
        g_salida = QGroupBox("3. Archivo PDF de Salida")
        l_salida = QHBoxLayout(g_salida)

        def_dir = os.path.abspath(os.path.join(emg_root, "resultados"))
        os.makedirs(def_dir, exist_ok=True)
        def_path = os.path.join(def_dir, "Atlas_Activacion_Muscular_sEMG.pdf")

        self.txt_ruta_pdf = QLineEdit(def_path)
        btn_examinar = QPushButton("Examinar...")
        btn_examinar.setStyleSheet("background-color: #1e293b; color: #f1f5f9; border: 1px solid #334155; padding: 6px 12px; border-radius: 4px;")
        btn_examinar.clicked.connect(self._examinar_ruta)

        l_salida.addWidget(self.txt_ruta_pdf, stretch=1)
        l_salida.addWidget(btn_examinar)
        layout.addWidget(g_salida)

        # 4. MONITOR DE ESTADO Y PROGRESO
        self.lbl_estado = QLabel("Listo para generar el documento.")
        self.lbl_estado.setStyleSheet("color: #94a3b8; font-size: 12px;")
        layout.addWidget(self.lbl_estado)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # 5. BOTONES DE ACCIÓN
        h_acciones = QHBoxLayout()

        self.btn_generar = QPushButton("GENERAR ATLAS (PDF)")
        self.btn_generar.setFixedHeight(45)
        self.btn_generar.setCursor(Qt.PointingHandCursor)
        self.btn_generar.setStyleSheet("""
            QPushButton {
                background-color: #00ffcc;
                color: #000000;
                font-weight: bold;
                font-size: 14px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #38bdf8;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
        """)
        self.btn_generar.clicked.connect(self._iniciar_generacion)
        h_acciones.addWidget(self.btn_generar, stretch=2)

        self.btn_abrir_pdf = QPushButton("ABRIR PDF GENERADO")
        self.btn_abrir_pdf.setFixedHeight(45)
        self.btn_abrir_pdf.setEnabled(False)
        self.btn_abrir_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_abrir_pdf.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: #ffffff;
                font-weight: bold;
                font-size: 13px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:disabled {
                background-color: #1e293b;
                color: #475569;
            }
        """)
        self.btn_abrir_pdf.clicked.connect(self._abrir_pdf)
        h_acciones.addWidget(self.btn_abrir_pdf, stretch=1)

        layout.addLayout(h_acciones)

    def _examinar_ruta(self):
        sug, _ = QFileDialog.getSaveFileName(
            self,
            "Seleccionar ubicación del Atlas PDF",
            self.txt_ruta_pdf.text().strip(),
            "Archivos PDF (*.pdf)"
        )
        if sug:
            if not sug.lower().endswith(".pdf"):
                sug += ".pdf"
            self.txt_ruta_pdf.setText(sug)

    def _iniciar_generacion(self):
        sujetos_sel = [s for s, chk in self.chks_sujetos.items() if chk.isChecked()]
        if not sujetos_sel:
            QMessageBox.warning(self, "Sin Sujetos", "Por favor seleccione al menos un sujeto para generar el atlas.")
            return

        ruta_pdf = self.txt_ruta_pdf.text().strip()
        if not ruta_pdf:
            QMessageBox.warning(self, "Ruta Inválida", "Especifique una ruta válida para el archivo PDF.")
            return

        self.btn_generar.setEnabled(False)
        self.btn_abrir_pdf.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_estado.setText("Iniciando compilación del Atlas sEMG...")

        tema = self.cmb_tema.currentData()
        filas = self.spn_filas.value()
        incluir_foto = self.chk_incluir_foto.isChecked()

        self.worker = AtlasWorker(
            ruta_pdf=ruta_pdf,
            incluir_foto=incluir_foto,
            tema=tema,
            sujetos=sujetos_sel,
            filas_por_pagina=filas,
            base_db=self.base_db
        )
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, msg, pct):
        self.lbl_estado.setText(msg)
        self.progress_bar.setValue(pct)

    def _on_finished(self, res):
        self.btn_generar.setEnabled(True)
        if res['success']:
            self.ultimo_pdf = res['pdf_path']
            self.btn_abrir_pdf.setEnabled(True)
            self.lbl_estado.setText(f"Completado exitosamente. Archivo: {os.path.basename(self.ultimo_pdf)}")
            QMessageBox.information(
                self,
                "Atlas sEMG Compilado",
                f"El Atlas de Activación Muscular sEMG se ha generado exitosamente en:\n\n{self.ultimo_pdf}"
            )
        else:
            self.lbl_estado.setText("Error durante la generación.")
            QMessageBox.critical(
                self,
                "Error al Generar Atlas",
                f"Ocurrió un error durante la generación del Atlas:\n\n{res['error']}"
            )

    def _abrir_pdf(self):
        if self.ultimo_pdf and os.path.exists(self.ultimo_pdf):
            try:
                subprocess.Popen(["xdg-open", self.ultimo_pdf])
            except Exception as e:
                QMessageBox.warning(self, "Aviso", f"No se pudo abrir el archivo automáticamente: {e}")
