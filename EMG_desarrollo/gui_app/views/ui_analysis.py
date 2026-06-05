# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Definiciones de interfaz de usuario para módulos de análisis.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Definiciones de interfaz de usuario para módulos de análisis.
# ==============================================================================

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QTabWidget,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLineEdit, QComboBox
)
from PySide6.QtCore import Qt

class ProcessingTab(QWidget):
    """Pestaña 1: Reemplaza al ProcessingOptionsDialog original de Tkinter"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)

        # 0. SELECCIÓN DE CANALES (Faltaba esta "configuración")
        self.g_canales_procesar = QGroupBox("1. Seleccionar Canales a Procesar (Global)")
        self.lyt_canales_procesar = QVBoxLayout()
        self.lbl_canales_warn = QLabel("Selecciona mediciones en el Gestor para ver los canales...")
        self.lbl_canales_warn.setStyleSheet("color: #FFA500;")
        self.lyt_canales_procesar.addWidget(self.lbl_canales_warn)
        self.g_canales_procesar.setLayout(self.lyt_canales_procesar)
        self.layout.addWidget(self.g_canales_procesar)

        # 1. OPCIONES DE ANÁLISIS INDIVIDUAL Y FILTROS
        g_ind = QGroupBox("2. Opciones de Análisis Individual")
        l_ind = QVBoxLayout()
        
        self.chk_recortes = QCheckBox("Generar gráfico de recortes (pulses.png)")
        self.chk_recortes.setChecked(True)
        l_ind.addWidget(self.chk_recortes)

        self.chk_cruda = QCheckBox("Incluir señal cruda en recortes")
        self.chk_cruda.setChecked(False)
        l_ind.addWidget(self.chk_cruda)

        self.chk_espectrograma = QCheckBox("Generar espectrograma (spec.png)")
        self.chk_espectrograma.setChecked(False)
        l_ind.addWidget(self.chk_espectrograma)

        self.chk_notch = QCheckBox("Aplicar filtro Notch 50 Hz (ruido de línea)")
        self.chk_notch.setChecked(True) # En original el var dice value=False pero el chk dice por defecto activado en sus comentarios? Original decia value=False
        l_ind.addWidget(self.chk_notch)

        # Evolucion temporal
        row_evol = QHBoxLayout()
        self.chk_evolucion = QCheckBox("Gráfico Evolución Temporal SNR y Ruido")
        self.chk_evolucion.setChecked(True)
        row_evol.addWidget(self.chk_evolucion)
        
        row_evol.addWidget(QLabel(" | Inicio(s):"))
        self.inp_evol_start = QLineEdit("10")
        self.inp_evol_start.setMaximumWidth(50)
        row_evol.addWidget(self.inp_evol_start)
        
        row_evol.addWidget(QLabel("Fin(s):"))
        self.inp_evol_end = QLineEdit("1000")
        self.inp_evol_end.setMaximumWidth(50)
        row_evol.addWidget(self.inp_evol_end)
        l_ind.addLayout(row_evol)

        # Excluir ventanas
        row_excl = QHBoxLayout()
        row_excl.addWidget(QLabel("Excluir ventanas (ej: 1,24):"))
        self.inp_excluded = QLineEdit("")
        row_excl.addWidget(self.inp_excluded)
        l_ind.addLayout(row_excl)

        # Suavizado Envolvente
        row_smooth = QHBoxLayout()
        row_smooth.addWidget(QLabel("Envolvente:"))
        self.cmb_tipo_env = QComboBox()
        self.cmb_tipo_env.addItems(["media_movil", "rms", "ninguna"])
        row_smooth.addWidget(self.cmb_tipo_env)
        
        row_smooth.addWidget(QLabel("Suavizado (ms):"))
        self.inp_smooth = QLineEdit("50")
        row_smooth.addWidget(self.inp_smooth)
        l_ind.addLayout(row_smooth)

        # Filtro Pasa-Altos
        row_hp = QHBoxLayout()
        row_hp.addWidget(QLabel("Filtro Pasa-Altos (Hz, 0 para desactivar):"))
        self.inp_hp = QLineEdit("20")
        row_hp.addWidget(self.inp_hp)
        l_ind.addLayout(row_hp)

        # Filtro Pasa-Bajos
        row_lp = QHBoxLayout()
        row_lp.addWidget(QLabel("Filtro Pasa-Bajos (Hz, 0 para desactivar):"))
        self.inp_lp = QLineEdit("500")
        row_lp.addWidget(self.inp_lp)
        l_ind.addLayout(row_lp)

        g_ind.setLayout(l_ind)
        self.layout.addWidget(g_ind)
        
        self.layout.addStretch()

        # Botones Lanzar
        btn_layout = QHBoxLayout()
        self.btn_run_procesar = QPushButton("🧠 PROCESAR Y CURAR INDIVIDUALES")
        self.btn_run_procesar.setFixedHeight(50)
        self.btn_run_procesar.setCursor(Qt.PointingHandCursor)
        self.btn_run_procesar.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #00ffcc; border: 2px solid #00ffcc; border-radius: 5px;
            }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        
        self.btn_run_rapido = QPushButton("⚡ REPROCESAR RÁPIDO")
        self.btn_run_rapido.setFixedHeight(50)
        self.btn_run_rapido.setCursor(Qt.PointingHandCursor)
        self.btn_run_rapido.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #ff0033; border: 2px solid #ff0033; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ff0033; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        
        btn_layout.addWidget(self.btn_run_procesar)
        btn_layout.addWidget(self.btn_run_rapido)
        self.layout.addLayout(btn_layout)


class ComparativeTab(QWidget):
    """Pestaña 2: Reemplaza al ComparativeOptionsDialog original de Tkinter"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(15)

        # 1. NOMBRE DE SET
        g_nom = QGroupBox("1. Nombre del Set de Análisis")
        l_nom = QVBoxLayout()
        self.inp_nombre_analisis = QLineEdit()
        self.inp_nombre_analisis.setPlaceholderText("(Opcional) Ej: Comparacion_Sujeto1_Post")
        l_nom.addWidget(self.inp_nombre_analisis)
        g_nom.setLayout(l_nom)
        self.layout.addWidget(g_nom)

        # 2. SELECCIÓN DE CANAL
        g_canal = QGroupBox("2. Comparar datos del Canal:")
        l_canal = QVBoxLayout()
        self.cmb_canal_comun = QComboBox()
        self.cmb_canal_comun.setEnabled(False) # Se habilita dinámicamente desde el Orquestador
        l_canal.addWidget(self.cmb_canal_comun)
        self.lbl_warning_canal = QLabel("Selecciona al menos 2 mediciones en el Gestor de Sesiones.")
        self.lbl_warning_canal.setStyleSheet("color: #FFA500;")
        l_canal.addWidget(self.lbl_warning_canal)
        g_canal.setLayout(l_canal)
        self.layout.addWidget(g_canal)

        # 3. OPCIONES GRÁFICOS
        g_graf = QGroupBox("3. Opciones de Gráficos Comparativos")
        l_graf = QVBoxLayout()
        
        self.chk_overlay = QCheckBox("Generar Overlay de Pulsos")
        self.chk_overlay.setChecked(True)
        l_graf.addWidget(self.chk_overlay)
        
        self.chk_snr = QCheckBox("Generar Gráfico SNR (Amplitud)")
        self.chk_snr.setChecked(True)
        l_graf.addWidget(self.chk_snr)

        self.chk_amp = QCheckBox("Generar Gráfico Amplitud Máxima")
        self.chk_amp.setChecked(True)
        l_graf.addWidget(self.chk_amp)

        self.chk_snr_time = QCheckBox("Generar Gráfico SNR vs Tiempo")
        self.chk_snr_time.setChecked(True)
        l_graf.addWidget(self.chk_snr_time)

        self.chk_amp_time = QCheckBox("Generar Gráfico Amplitud vs Tiempo")
        self.chk_amp_time.setChecked(True)
        l_graf.addWidget(self.chk_amp_time)

        self.chk_table = QCheckBox("Generar Tabla de Resultados (CSV y PNG)")
        self.chk_table.setChecked(True)
        l_graf.addWidget(self.chk_table)

        g_graf.setLayout(l_graf)
        self.layout.addWidget(g_graf)
        
        self.layout.addStretch()

        # Botón Lanzar
        self.btn_run_comparativo = QPushButton("📊 LANZAR ANÁLISIS COMPARATIVO")
        self.btn_run_comparativo.setFixedHeight(50)
        self.btn_run_comparativo.setCursor(Qt.PointingHandCursor)
        self.btn_run_comparativo.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #ffff00; border: 2px solid #ffff00; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ffff00; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        self.layout.addWidget(self.btn_run_comparativo)


class AnalysisPanel(QWidget):
    """Contenedor Principal que alberga las Pestañas de Análisis y emite los kwargs"""
    def __init__(self):
        super().__init__()
        # Estilo oscuro
        self.setStyleSheet("""
            QWidget { background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace; }
            QGroupBox {
                border: 1px solid #00ffcc; border-radius: 4px;
                margin-top: 10px; padding-top: 10px; font-weight: bold; color: #ff0033;
            }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            QLabel { color: #00ffcc; }
            QCheckBox { color: #00ffcc; }
            QCheckBox::indicator { border: 1px solid #00ffcc; width: 15px; height: 15px; background: #111; }
            QCheckBox::indicator:checked { background: #00ffcc; }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #111; color: #ff0033; border: 1px solid #ff0033; padding: 4px;
            }
            QTabWidget::pane { border: 2px solid #ff0033; border-radius: 4px; background: #050505; }
            QTabBar::tab {
                background: #111; color: #00ffcc; padding: 8px 15px; border: 1px solid #ff0033;
                border-bottom: none; border-top-left-radius: 4px; border-top-right-radius: 4px;
            }
            QTabBar::tab:selected { background: #050505; color: #ff0033; font-weight: bold; border: 2px solid #ff0033; border-bottom: none; }
        """)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        self.tabs = QTabWidget()
        
        # Pestaña Individual
        self.tab_procesamiento = ProcessingTab()
        self.tabs.addTab(self.tab_procesamiento, "⚙️ Procesamiento Individual")
        
        # Pestaña Comparativa
        self.tab_comparativo = ComparativeTab()
        self.tabs.addTab(self.tab_comparativo, "📊 Análisis Comparativo")

        main_layout.addWidget(self.tabs)

    def get_processing_kwargs(self):
        """Devuelve los kwargs basados exclusivamente en lo que ofrece la UI original de Tkinter"""
        # Parsear exclusiones
        excl_str = self.tab_procesamiento.inp_excluded.text().strip()
        excluded = []
        if excl_str:
            try: excluded = [int(x.strip()) for x in excl_str.split(',') if x.strip()]
            except ValueError: pass

        # Parsear numéricos con fallbacks seguros
        try: smooth = float(self.tab_procesamiento.inp_smooth.text())
        except ValueError: smooth = 50.0

        try: hp = float(self.tab_procesamiento.inp_hp.text())
        except ValueError: hp = 20.0

        try: lp = float(self.tab_procesamiento.inp_lp.text())
        except ValueError: lp = 500.0

        try: ev_start = float(self.tab_procesamiento.inp_evol_start.text())
        except ValueError: ev_start = 10.0

        try: ev_end = float(self.tab_procesamiento.inp_evol_end.text())
        except ValueError: ev_end = 1000.0
        
        tipo_env = self.tab_procesamiento.cmb_tipo_env.currentText()

        return {
            'smooth_ms': smooth,
            'tipo_envolvente': tipo_env,
            'apply_notch_filter': self.tab_procesamiento.chk_notch.isChecked(),
            'highpass_cutoff_hz': hp,
            'lowpass_cutoff_hz': lp,
            'mostrar_recortes': self.tab_procesamiento.chk_recortes.isChecked(),
            'mostrar_senal_cruda': self.tab_procesamiento.chk_cruda.isChecked(),
            'mostrar_espectrograma': self.tab_procesamiento.chk_espectrograma.isChecked(),
            'mostrar_evolucion': self.tab_procesamiento.chk_evolucion.isChecked(),
            'evol_t_start': ev_start,
            'evol_t_end': ev_end,
            'excluded_windows_list': excluded
        }

    def get_comparative_kwargs(self):
        """Devuelve los booleanos para _comparative_plots"""
        return {
            'show_overlay': self.tab_comparativo.chk_overlay.isChecked(),
            'show_snr': self.tab_comparativo.chk_snr.isChecked(),
            'show_amplitude': self.tab_comparativo.chk_amp.isChecked(),
            'show_table': self.tab_comparativo.chk_table.isChecked(),
            'show_snr_time': self.tab_comparativo.chk_snr_time.isChecked(),
            'show_amp_time': self.tab_comparativo.chk_amp_time.isChecked()
        }
