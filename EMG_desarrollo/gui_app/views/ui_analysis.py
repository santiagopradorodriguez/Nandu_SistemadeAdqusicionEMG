# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Definiciones de interfaz de usuario para módulos de análisis.
# ==============================================================================

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QTabWidget,
    QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QPushButton, QLineEdit, QComboBox,
    QScrollArea, QRadioButton, QGridLayout, QDialog, QListWidget, QPlainTextEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

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

        self.chk_cyberpunk = QCheckBox("Tema Cyberpunk (Gráficos oscuros y neón)")
        self.chk_cyberpunk.setChecked(False) # Por defecto apagado para usar estética normal
        l_ind.addWidget(self.chk_cyberpunk)

        self.chk_espectrograma = QCheckBox("Generar Espectrograma Señal Completa (Estilo Praat)")
        self.chk_espectrograma.setChecked(False)
        row_spec = QHBoxLayout()
        row_spec.addWidget(self.chk_espectrograma)

        row_spec.addWidget(QLabel("Freq. Máx (Hz):"))
        self.inp_spec_fmax = QLineEdit("5000")
        self.inp_spec_fmax.setFixedWidth(60)
        row_spec.addWidget(self.inp_spec_fmax)
        row_spec.addStretch()
        l_ind.addLayout(row_spec)

        row_notch = QHBoxLayout()
        self.chk_notch = QCheckBox("Aplicar filtro Notch 50 Hz (ruido de línea)")
        self.chk_notch.setChecked(True)
        row_notch.addWidget(self.chk_notch)

        row_notch.addWidget(QLabel("Factor Q:"))
        self.inp_notch_q = QLineEdit("2.0")
        self.inp_notch_q.setFixedWidth(40)
        row_notch.addWidget(self.inp_notch_q)
        row_notch.addStretch()
        l_ind.addLayout(row_notch)

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
        self.btn_run_procesar = QPushButton(" PROCESAR Y CURAR INDIVIDUALES")
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
        
        self.btn_run_rapido = QPushButton("REPROCESAR RAPIDO")
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

    def get_processing_kwargs(self):
        """Recolecta todos los parámetros de procesamiento de la pestaña individual."""
        excl_raw = self.inp_excluded.text().strip()
        excl_list = []
        if excl_raw:
            for part in excl_raw.split(','):
                part = part.strip()
                if part.isdigit():
                    excl_list.append(int(part))
        return {
            'mostrar_recortes': self.chk_recortes.isChecked(),
            'mostrar_senal_cruda': self.chk_cruda.isChecked(),
            'tema_cyberpunk': self.chk_cyberpunk.isChecked(),
            'mostrar_espectrograma': self.chk_espectrograma.isChecked(),
            'frecuenciamaxima': self.inp_spec_fmax.text().strip() or "5000",
            'apply_notch_filter': self.chk_notch.isChecked(),
            'notch_q_factor': self.inp_notch_q.text().strip() or "2.0",
            'mostrar_evolucion': self.chk_evolucion.isChecked(),
            'evol_t_start': self.inp_evol_start.text().strip() or "10",
            'evol_t_end': self.inp_evol_end.text().strip() or "1000",
            'excluded_windows_list': excl_list,
            'tipo_envolvente': self.cmb_tipo_env.currentText(),
            'smooth_ms': self.inp_smooth.text().strip() or "50",
            'highpass_cutoff_hz': self.inp_hp.text().strip() or "20",
            'lowpass_cutoff_hz': self.inp_lp.text().strip() or "500"
        }


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

        # Botones Lanzar
        btn_layout = QHBoxLayout()
        
        self.btn_run_comparativo = QPushButton(" LANZAR ANÁLISIS COMPARATIVO")
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
        btn_layout.addWidget(self.btn_run_comparativo)

        self.btn_run_sesion = QPushButton("LANZAR EVOLUCIÓN DE SESIÓN")
        self.btn_run_sesion.setFixedHeight(50)
        self.btn_run_sesion.setCursor(Qt.PointingHandCursor)
        self.btn_run_sesion.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #00ffff; border: 2px solid #00ffff; border-radius: 5px;
            }
            QPushButton:hover { background-color: #00ffff; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        btn_layout.addWidget(self.btn_run_sesion)
        
        self.layout.addLayout(btn_layout)

        self.btn_generar_reporte = QPushButton("GENERAR REPORTE INTEGRAL (PDF)")
        self.btn_generar_reporte.setFixedHeight(45)
        self.btn_generar_reporte.setCursor(Qt.PointingHandCursor)
        self.btn_generar_reporte.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 13px;
                background-color: transparent; color: #00ff88; border: 2px solid #00ff88; border-radius: 5px;
                margin-top: 5px;
            }
            QPushButton:hover { background-color: #00ff88; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        self.layout.addWidget(self.btn_generar_reporte)

        self.btn_limpiar_analisis_results = QPushButton("BORRAR ARCHIVOS DE ANÁLISIS RESULTS (LIBERAR ESPACIO)")
        self.btn_limpiar_analisis_results.setFixedHeight(40)
        self.btn_limpiar_analisis_results.setCursor(Qt.PointingHandCursor)
        self.btn_limpiar_analisis_results.setToolTip("Elimina los archivos JSON de análisis intermedio para recuperar espacio en disco.")
        self.btn_limpiar_analisis_results.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 12px;
                background-color: transparent; color: #ff5555; border: 2px solid #ff5555; border-radius: 5px;
                margin-top: 5px;
            }
            QPushButton:hover { background-color: #ff5555; color: #ffffff; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        self.layout.addWidget(self.btn_limpiar_analisis_results)


class DiscreteMotorTab(QWidget):
    """Pestaña para Análisis de Coordenadas Discretas (Assaneo et al. 2013)"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 1. Selector de Método (QTabWidget)
        self.method_tabs = QTabWidget()
        
        # 1A. Pestaña Estadístico
        tab_stat = QWidget()
        l_stat = QFormLayout(tab_stat)
        self.inp_std_multiplier = QDoubleSpinBox()
        self.inp_std_multiplier.setRange(1.0, 10.0)
        self.inp_std_multiplier.setSingleStep(0.5)
        self.inp_std_multiplier.setValue(3.0)
        l_stat.addRow("Sensibilidad (N Std):", self.inp_std_multiplier)
        self.method_tabs.addTab(tab_stat, " Umbral Estadístico (Ruido)")
        
        # 1B. Pestaña Manual
        tab_man = QWidget()
        l_man = QVBoxLayout(tab_man)
        l_man.addWidget(QLabel("Umbrales absolutos (0.01 a 1.0) sobre la máxima amplitud global del pulso."))
        
        try:
            from utils.config_manager import ConfigManager
            cm = ConfigManager()
            c_config = cm.get("canales") or {}
        except Exception:
            c_config = {}
        
        self.manual_thresholds = {}
        form_man = QFormLayout()
        for i in range(8):  # Soportamos hasta 8 canales por defecto
            c_key = f"canal_{i}"
            nombre = c_config.get(f"Canal {i}", {}).get("musculo", c_key)
            sp = QDoubleSpinBox()
            sp.setRange(0.01, 1.0)
            sp.setSingleStep(0.05)
            sp.setValue(0.5)
            form_man.addRow(f"Umbral {nombre}:", sp)
            self.manual_thresholds[c_key] = sp
            
        l_man.addLayout(form_man)
        self.method_tabs.addTab(tab_man, "Umbral Manual por Canal")
        
        self.layout.addWidget(self.method_tabs)
        
        # 2. Anotación de Vocales
        g_voc = QGroupBox("Anotación de Secuencia de Vocales")
        g_voc.setCheckable(True)
        g_voc.setChecked(False)
        self.g_voc = g_voc
        l_voc = QFormLayout()
        
        self.cmb_vocal_orden = QComboBox()
        self.cmb_vocal_orden.addItems(["Normal (a, e, i, o, u)", "Inverso (u, o, i, e, a)"])
        l_voc.addRow("Orden:", self.cmb_vocal_orden)
        
        self.cmb_vocal_inicio = QComboBox()
        self.cmb_vocal_inicio.addItems(["a", "e", "i", "o", "u"])
        l_voc.addRow("Primera vocal del registro:", self.cmb_vocal_inicio)
        
        g_voc.setLayout(l_voc)
        self.layout.addWidget(g_voc)
        
        self.layout.addStretch()
        
        btn_layout = QHBoxLayout()
        self.btn_run_motor = QPushButton(" LANZAR COORDENADAS DISCRETAS")
        self.btn_run_motor.setFixedHeight(50)
        self.btn_run_motor.setCursor(Qt.PointingHandCursor)
        self.btn_run_motor.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #ff00ff; border: 2px solid #ff00ff; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ff00ff; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        btn_layout.addWidget(self.btn_run_motor)
        
        self.layout.addLayout(btn_layout)

class TrainingMotorTab(QWidget):
    """Pestaña para Entrenamiento de Umbrales Óptimos (Barrido de Colisiones)"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 1. Opciones de Filtro y Pre-procesamiento
        g_filtro = QGroupBox("1. Limpieza de Datos (Pre-procesamiento)")
        l_filtro = QFormLayout()
        
        self.chk_snr = QCheckBox("Descartar si el SNR es menor a:")
        self.chk_snr.setChecked(True)
        
        self.inp_snr_limit = QDoubleSpinBox()
        self.inp_snr_limit.setRange(0.1, 50.0)
        self.inp_snr_limit.setSingleStep(0.5)
        self.inp_snr_limit.setValue(4.0)
        
        self.cmb_snr_tipo = QComboBox()
        self.cmb_snr_tipo.addItems(["Por Ventana (Individual)", "Global (Toda la medición)", "Ambos (Global + Ventana)"])
        
        row_lyt = QHBoxLayout()
        row_lyt.addWidget(self.chk_snr)
        row_lyt.addWidget(self.inp_snr_limit)
        row_lyt.addWidget(self.cmb_snr_tipo)
        row_lyt.addStretch()
        
        l_filtro.addRow(row_lyt)
        g_filtro.setLayout(l_filtro)
        self.layout.addWidget(g_filtro)
        
        # 2. Metodología de Discretización
        g_metodo = QGroupBox("2. Metodología de Discretización")
        l_metodo = QFormLayout()
        
        self.cmb_tipo_barrido = QComboBox()
        self.cmb_tipo_barrido.addItems([
            "Umbral Común (Único para todos los canales)",
            "Umbral por Canal (Búsqueda de Intervalos Óptimos)"
        ])
        
        self.inp_paso_barrido = QDoubleSpinBox()
        self.inp_paso_barrido.setRange(0.01, 0.20)
        self.inp_paso_barrido.setSingleStep(0.01)
        self.inp_paso_barrido.setValue(0.05)
        self.inp_paso_barrido.setToolTip("Paso de iteración para buscar intervalos.")
        
        l_metodo.addRow(QLabel("Tipo de Búsqueda:"), self.cmb_tipo_barrido)
        l_metodo.addRow(QLabel("Resolución del Barrido:"), self.inp_paso_barrido)
        
        g_metodo.setLayout(l_metodo)
        self.layout.addWidget(g_metodo)
        
        # --- BOTON DE EJECUCION ---
        btn_layout = QHBoxLayout()
        self.btn_run_training = QPushButton("ENTRENAR UMBRALES (TRAIN)")
        self.btn_run_training.setFixedHeight(50)
        self.btn_run_training.setCursor(Qt.PointingHandCursor)
        self.btn_run_training.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #00ffcc; border: 2px solid #00ffcc; border-radius: 5px;
            }
            QPushButton:hover { background-color: #00ffcc; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        btn_layout.addWidget(self.btn_run_training)
        
        self.layout.addLayout(btn_layout)

class PcaTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        
        g_canales = QGroupBox("Canales EMG a incluir en PCA")
        l_canales = QHBoxLayout()
        self.chk_canal_0 = QCheckBox("Canal 0 (Milohioideo)")
        self.chk_canal_0.setChecked(True)
        self.chk_canal_1 = QCheckBox("Canal 1 (Depresor)")
        self.chk_canal_1.setChecked(True)
        self.chk_canal_2 = QCheckBox("Canal 2 (Orbicular)")
        self.chk_canal_2.setChecked(True)
        l_canales.addWidget(self.chk_canal_0)
        l_canales.addWidget(self.chk_canal_1)
        l_canales.addWidget(self.chk_canal_2)
        g_canales.setLayout(l_canales)
        lay.addWidget(g_canales)
        
        g_graficos = QGroupBox("Procesar Gráficos")
        l_graficos = QHBoxLayout()
        self.chk_pca_2d = QCheckBox("PCA 2D")
        self.chk_pca_2d.setChecked(True)
        self.chk_pca_3d = QCheckBox("PCA 3D")
        self.chk_pca_3d.setChecked(True)
        self.chk_ocultar_leyenda = QCheckBox("Ocultar Leyenda")
        self.chk_ocultar_leyenda.setChecked(False)
        l_graficos.addWidget(self.chk_pca_2d)
        l_graficos.addWidget(self.chk_pca_3d)
        l_graficos.addWidget(self.chk_ocultar_leyenda)
        g_graficos.setLayout(l_graficos)
        lay.addWidget(g_graficos)
        
        def create_dsp_row(title, default_smooth, default_alpha, is_3d=False):
            g = QGroupBox(title)
            l = QGridLayout()
            l.setContentsMargins(5, 5, 5, 5)
            l.setSpacing(5)
            
            # Fila 0: Parámetros DSP
            l.addWidget(QLabel("Alpha:"), 0, 0)
            inp_alpha = QDoubleSpinBox()
            inp_alpha.setRange(0.01, 10.0)
            inp_alpha.setSingleStep(0.1)
            inp_alpha.setValue(default_alpha)
            inp_alpha.setFixedWidth(55)
            l.addWidget(inp_alpha, 0, 1)
            
            l.addWidget(QLabel("Smooth:"), 0, 2)
            inp_smooth = QSpinBox()
            inp_smooth.setRange(0, 1000)
            inp_smooth.setValue(default_smooth)
            inp_smooth.setFixedWidth(55)
            l.addWidget(inp_smooth, 0, 3)
            
            l.addWidget(QLabel("Pts:"), 0, 4)
            inp_pts = QSpinBox()
            inp_pts.setRange(1, 1000)
            inp_pts.setValue(20)
            inp_pts.setFixedWidth(45)
            l.addWidget(inp_pts, 0, 5)
            
            l.addWidget(QLabel("SNR:"), 0, 6)
            inp_snr = QDoubleSpinBox()
            inp_snr.setRange(0.0, 100.0)
            inp_snr.setSingleStep(0.05)
            inp_snr.setValue(0.5)
            inp_snr.setFixedWidth(55)
            l.addWidget(inp_snr, 0, 7)
            
            l.addWidget(QLabel("Outliers:"), 0, 8)
            inp_outliers = QDoubleSpinBox()
            inp_outliers.setRange(0.0, 0.99)
            inp_outliers.setSingleStep(0.05)
            inp_outliers.setValue(0.10)
            inp_outliers.setFixedWidth(55)
            l.addWidget(inp_outliers, 0, 9)
            
            l.addWidget(QLabel("Notch Q:"), 0, 10)
            inp_notch = QDoubleSpinBox()
            inp_notch.setRange(0.1, 100.0)
            inp_notch.setSingleStep(0.5)
            inp_notch.setValue(2.0)
            inp_notch.setFixedWidth(45)
            l.addWidget(inp_notch, 0, 11)
            
            l.addWidget(QLabel("Gate:"), 0, 12)
            inp_gate = QDoubleSpinBox()
            inp_gate.setRange(0.0, 100.0)
            inp_gate.setSingleStep(0.5)
            inp_gate.setValue(0.0)
            inp_gate.setFixedWidth(45)
            l.addWidget(inp_gate, 0, 13)

            # Fila 1: Componentes y Pesos de Canales
            pcs = [f"PC{i}" for i in range(1, 7)]
            l.addWidget(QLabel("Eje X:"), 1, 0)
            cmb_pc_x = QComboBox()
            cmb_pc_x.addItems(pcs)
            cmb_pc_x.setCurrentText("PC1")
            cmb_pc_x.setFixedWidth(55)
            l.addWidget(cmb_pc_x, 1, 1)

            l.addWidget(QLabel("Eje Y:"), 1, 2)
            cmb_pc_y = QComboBox()
            cmb_pc_y.addItems(pcs)
            cmb_pc_y.setCurrentText("PC2")
            cmb_pc_y.setFixedWidth(55)
            l.addWidget(cmb_pc_y, 1, 3)

            if is_3d:
                l.addWidget(QLabel("Eje Z:"), 1, 4)
                cmb_pc_z = QComboBox()
                cmb_pc_z.addItems(pcs)
                cmb_pc_z.setCurrentText("PC3")
                cmb_pc_z.setFixedWidth(50)
                l.addWidget(cmb_pc_z, 1, 5)
            else:
                cmb_pc_z = None

            # Pesos de canales W0, W1, W2
            w_col = 6 if is_3d else 4
            l.addWidget(QLabel("W Ch0:"), 1, w_col)
            inp_w0 = QDoubleSpinBox()
            inp_w0.setRange(0.0, 20.0)
            inp_w0.setSingleStep(0.2)
            inp_w0.setValue(1.0)
            inp_w0.setFixedWidth(50)
            l.addWidget(inp_w0, 1, w_col+1)

            l.addWidget(QLabel("W Ch1:"), 1, w_col+2)
            inp_w1 = QDoubleSpinBox()
            inp_w1.setRange(0.0, 20.0)
            inp_w1.setSingleStep(0.2)
            inp_w1.setValue(1.0)
            inp_w1.setFixedWidth(50)
            l.addWidget(inp_w1, 1, w_col+3)

            l.addWidget(QLabel("W Ch2:"), 1, w_col+4)
            inp_w2 = QDoubleSpinBox()
            inp_w2.setRange(0.0, 20.0)
            inp_w2.setSingleStep(0.2)
            inp_w2.setValue(1.0)
            inp_w2.setFixedWidth(50)
            l.addWidget(inp_w2, 1, w_col+5)

            g.setLayout(l)
            return (g, inp_alpha, inp_smooth, inp_pts, inp_snr, inp_outliers, inp_notch, inp_gate,
                    cmb_pc_x, cmb_pc_y, cmb_pc_z, inp_w0, inp_w1, inp_w2)

        (g_dsp_2d, self.inp_alpha_2d, self.inp_smooth_2d, self.inp_pts_2d, self.inp_snr_2d, 
         self.inp_outliers_2d, self.inp_notch_2d, self.inp_gate_2d,
         self.cmb_pc_x_2d, self.cmb_pc_y_2d, _, self.inp_w0_2d, self.inp_w1_2d, self.inp_w2_2d) = create_dsp_row(
             "Parámetros DSP, Componentes y Ponderación (2D)", 90, 0.5, is_3d=False)
        lay.addWidget(g_dsp_2d)
        
        (g_dsp_3d, self.inp_alpha_3d, self.inp_smooth_3d, self.inp_pts_3d, self.inp_snr_3d, 
         self.inp_outliers_3d, self.inp_notch_3d, self.inp_gate_3d,
         self.cmb_pc_x_3d, self.cmb_pc_y_3d, self.cmb_pc_z_3d, self.inp_w0_3d, self.inp_w1_3d, self.inp_w2_3d) = create_dsp_row(
             "Parámetros DSP, Componentes y Ponderación (3D)", 125, 0.5, is_3d=True)
        lay.addWidget(g_dsp_3d)

        g_cluster = QGroupBox("Algoritmo de Agrupamiento")
        l_cluster = QHBoxLayout()
        l_cluster.addWidget(QLabel("Evaluar PCA:"))
        self.cmb_cluster = QComboBox()
        self.cmb_cluster.addItems(["GMM", "GMM Jerárquico (PCA Local)", "K-Means"])
        l_cluster.addWidget(self.cmb_cluster)
        g_cluster.setLayout(l_cluster)
        lay.addWidget(g_cluster)

        g_adv = QGroupBox("DSP Avanzado y Normalización")
        l_adv = QGridLayout()
        self.chk_trevisan = QCheckBox("Aplicar Corrección Trevisan (Mediana Móvil + Detrending)")
        self.chk_trevisan.setChecked(False)
        l_adv.addWidget(self.chk_trevisan, 0, 0, 1, 2)
        
        self.chk_ignorar_cero = QCheckBox("Ignorar Ventana 0 (Artefactos)")
        self.chk_ignorar_cero.setChecked(False)
        l_adv.addWidget(self.chk_ignorar_cero, 0, 2, 1, 2)
        
        self.chk_correccion_intersesion = QCheckBox("Corrección Intersesión por Lote (Calibración de Ganancia)")
        self.chk_correccion_intersesion.setChecked(True)
        l_adv.addWidget(self.chk_correccion_intersesion, 1, 0, 1, 4)
        
        l_adv.addWidget(QLabel("Pre-Ventana (%):"), 2, 0)
        self.inp_pre_pct = QDoubleSpinBox()
        self.inp_pre_pct.setRange(0.0, 1.0)
        self.inp_pre_pct.setSingleStep(0.1)
        self.inp_pre_pct.setValue(0.4)
        l_adv.addWidget(self.inp_pre_pct, 2, 1)
        
        l_adv.addWidget(QLabel("Post-Ventana (%):"), 2, 2)
        self.inp_post_pct = QDoubleSpinBox()
        self.inp_post_pct.setRange(0.0, 1.0)
        self.inp_post_pct.setSingleStep(0.1)
        self.inp_post_pct.setValue(0.6)
        l_adv.addWidget(self.inp_post_pct, 2, 3)
        
        l_adv.addWidget(QLabel("Gate Ratio Ruido:"), 3, 0)
        self.inp_gate = QDoubleSpinBox()
        self.inp_gate.setRange(0.0, 100.0)
        self.inp_gate.setSingleStep(0.5)
        self.inp_gate.setValue(0.0)
        self.inp_gate.setFixedWidth(60)
        l_adv.addWidget(self.inp_gate, 3, 1)

        l_adv.addWidget(QLabel("Filtro Ruido Línea:"), 3, 2)
        self.cmb_noise_filter = QComboBox()
        self.cmb_noise_filter.addItems(["Filtro Notch (IIR)", "Filtro Adaptativo (NLMS)", "Desactivado"])
        self.cmb_noise_filter.setCurrentIndex(0)
        self.cmb_noise_filter.setToolTip("Selecciona el método de supresión de interferencia de línea eléctrica (50 Hz y armónicos):\n- Filtro Notch (IIR): Notch convencional de 50 Hz con factor de calidad Q.\n- Filtro Adaptativo (NLMS): Cancelación adaptativa NLMS con armónicos sintéticos (50, 100, 150, 200 Hz).\n- Desactivado: Sin supresión de línea.")
        l_adv.addWidget(self.cmb_noise_filter, 3, 3)

        self.lbl_notch_q = QLabel("Notch Factor Q:")
        l_adv.addWidget(self.lbl_notch_q, 4, 0)
        self.inp_notch_q = QDoubleSpinBox()
        self.inp_notch_q.setRange(0.1, 100.0)
        self.inp_notch_q.setSingleStep(0.5)
        self.inp_notch_q.setValue(2.0)
        self.inp_notch_q.setFixedWidth(60)
        self.inp_notch_q.setToolTip("Factor de calidad Q para el filtro Notch IIR (un valor más bajo ensancha el ancho de rechazo).")
        l_adv.addWidget(self.inp_notch_q, 4, 1)

        l_adv.addWidget(QLabel("Filtro Pasaltos:"), 4, 2)
        self.inp_highpass = QDoubleSpinBox()
        self.inp_highpass.setRange(0.0, 500.0)
        self.inp_highpass.setSingleStep(1.0)
        self.inp_highpass.setValue(20.0)
        self.inp_highpass.setSuffix(" Hz")
        self.inp_highpass.setToolTip("Frecuencia de corte inferior pasaltos (Butterworth orden 4).\n0.0 Hz desactiva el filtro pasaltos.")
        l_adv.addWidget(self.inp_highpass, 4, 3)

        l_adv.addWidget(QLabel("Filtro Pasabajos:"), 5, 0)
        self.cmb_lowpass = QComboBox()
        self.cmb_lowpass.addItems(["300 Hz (Rangayyan)", "500 Hz (Previo)", "Desactivado"])
        self.cmb_lowpass.setCurrentIndex(0)
        self.cmb_lowpass.setToolTip("Selecciona el filtrado pasabajos antes de extraer envolventes:\n- 300 Hz: Butterworth orden 6 de Rangayyan (recomendado).\n- 500 Hz: Butterworth orden 4 estándar previo.\n- Desactivado: Sin filtro pasabajos.")
        l_adv.addWidget(self.cmb_lowpass, 5, 1, 1, 3)
        
        g_adv.setLayout(l_adv)
        lay.addWidget(g_adv)

        def _on_noise_filter_changed_pca(idx):
            is_notch = (idx == 0)
            self.lbl_notch_q.setEnabled(is_notch)
            self.inp_notch_q.setEnabled(is_notch)
            if hasattr(self, 'inp_notch_2d'):
                self.inp_notch_2d.setEnabled(is_notch)
            if hasattr(self, 'inp_notch_3d'):
                self.inp_notch_3d.setEnabled(is_notch)
        self.cmb_noise_filter.currentIndexChanged.connect(_on_noise_filter_changed_pca)

        # Sincronización bidireccional entre Notch Q general y Notch Q 2D / 3D
        if hasattr(self, 'inp_notch_2d') and hasattr(self, 'inp_notch_3d'):
            self.inp_notch_q.valueChanged.connect(lambda v: (self.inp_notch_2d.blockSignals(True), self.inp_notch_2d.setValue(v), self.inp_notch_2d.blockSignals(False)))
            self.inp_notch_q.valueChanged.connect(lambda v: (self.inp_notch_3d.blockSignals(True), self.inp_notch_3d.setValue(v), self.inp_notch_3d.blockSignals(False)))
            self.inp_notch_2d.valueChanged.connect(lambda v: (self.inp_notch_q.blockSignals(True), self.inp_notch_q.setValue(v), self.inp_notch_q.blockSignals(False)))
            self.inp_notch_3d.valueChanged.connect(lambda v: (self.inp_notch_q.blockSignals(True), self.inp_notch_q.setValue(v), self.inp_notch_q.blockSignals(False)))

        g_align = QGroupBox("Alineación Temporal")
        l_align = QHBoxLayout()
        l_align.addWidget(QLabel("Centrar ventana en:"))
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["Pico Volumen Micrófono", "Pico Derivada Micrófono (Onset)", "Pico Canal 0", "Pico Canal 1", "Pico Canal 2"])
        l_align.addWidget(self.cmb_align)
        g_align.setLayout(l_align)
        lay.addWidget(g_align)

        g_visual = QGroupBox("Estilo Visual (Solo GMM)")
        l_visual = QHBoxLayout()
        self.rb_fronteras = QRadioButton("Fronteras")
        self.rb_fronteras.setChecked(True)
        self.rb_sombreado = QRadioButton("Sombreado")
        self.rb_elipses = QRadioButton("Elipses")
        l_visual.addWidget(self.rb_fronteras)
        l_visual.addWidget(self.rb_sombreado)
        l_visual.addWidget(self.rb_elipses)
        g_visual.setLayout(l_visual)
        lay.addWidget(g_visual)

        lay.addStretch()
        scroll.setWidget(content)
        self.layout.addWidget(scroll)
        
        l_grid_btns = QHBoxLayout()
        self.btn_grid_search_2d = QPushButton(" GRID SEARCH (2D)")
        self.btn_grid_search_2d.setStyleSheet("background-color: #66FCF1; color: black; font-weight: bold; padding: 10px; margin-bottom: 5px;")
        
        self.btn_grid_search_3d = QPushButton(" GRID SEARCH (3D)")
        self.btn_grid_search_3d.setStyleSheet("background-color: #45A29E; color: black; font-weight: bold; padding: 10px; margin-bottom: 5px;")
        
        l_grid_btns.addWidget(self.btn_grid_search_2d)
        l_grid_btns.addWidget(self.btn_grid_search_3d)
        self.layout.addLayout(l_grid_btns)

        self.btn_run = QPushButton(" 1. LANZAR PCA (COMP. PRINCIPALES)")
        self.btn_run.setStyleSheet("background-color: #00ffcc; color: black; font-weight: bold; padding: 10px;")
        self.layout.addWidget(self.btn_run)

        self.btn_visor_features = QPushButton(" VISUALIZADOR DE FEATURES (PCA / UMAP)")
        self.btn_visor_features.setStyleSheet("background-color: #2b0938; color: #e879f9; font-weight: bold; border: 1px solid #e879f9; padding: 10px; margin-top: 4px;")
        self.layout.addWidget(self.btn_visor_features)

class UmapTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        
        g_canales = QGroupBox("Canales EMG a incluir en UMAP")
        l_canales = QHBoxLayout()
        self.chk_canal_0 = QCheckBox("Canal 0 (Milohioideo)")
        self.chk_canal_0.setChecked(True)
        self.chk_canal_1 = QCheckBox("Canal 1 (Depresor)")
        self.chk_canal_1.setChecked(True)
        self.chk_canal_2 = QCheckBox("Canal 2 (Orbicular)")
        self.chk_canal_2.setChecked(True)
        l_canales.addWidget(self.chk_canal_0)
        l_canales.addWidget(self.chk_canal_1)
        l_canales.addWidget(self.chk_canal_2)
        g_canales.setLayout(l_canales)
        lay.addWidget(g_canales)

        g_graficos = QGroupBox("Procesar Gráficos")
        l_graficos = QHBoxLayout()
        self.chk_umap_2d = QCheckBox("UMAP 2D")
        self.chk_umap_2d.setChecked(True)
        self.chk_umap_3d = QCheckBox("UMAP 3D")
        self.chk_umap_3d.setChecked(True)
        self.chk_ocultar_leyenda = QCheckBox("Ocultar Leyenda")
        self.chk_ocultar_leyenda.setChecked(False)
        l_graficos.addWidget(self.chk_umap_2d)
        l_graficos.addWidget(self.chk_umap_3d)
        l_graficos.addWidget(self.chk_ocultar_leyenda)
        g_graficos.setLayout(l_graficos)
        lay.addWidget(g_graficos)

        g_umap = QGroupBox("Hiperparámetros Topológicos UMAP")
        l_umap = QGridLayout()
        l_umap.setContentsMargins(5, 5, 5, 5)
        l_umap.setSpacing(5)

        l_umap.addWidget(QLabel("Alpha:"), 0, 0)
        self.inp_alpha_u = QDoubleSpinBox()
        self.inp_alpha_u.setRange(0.01, 10.0)
        self.inp_alpha_u.setSingleStep(0.1)
        self.inp_alpha_u.setValue(1.0)
        self.inp_alpha_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_alpha_u, 0, 1)

        l_umap.addWidget(QLabel("n_neighbors:"), 0, 2)
        self.inp_n_neighbors = QSpinBox()
        self.inp_n_neighbors.setRange(2, 500)
        self.inp_n_neighbors.setValue(10)
        self.inp_n_neighbors.setFixedWidth(60)
        l_umap.addWidget(self.inp_n_neighbors, 0, 3)

        l_umap.addWidget(QLabel("min_dist:"), 0, 4)
        self.inp_min_dist = QDoubleSpinBox()
        self.inp_min_dist.setRange(0.0, 1.0)
        self.inp_min_dist.setSingleStep(0.05)
        self.inp_min_dist.setValue(0.05)
        self.inp_min_dist.setFixedWidth(60)
        l_umap.addWidget(self.inp_min_dist, 0, 5)

        l_umap.addWidget(QLabel("Métrica:"), 0, 6)
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(["euclidean", "manhattan", "chebyshev", "minkowski", "cosine"])
        self.cmb_metric.setFixedWidth(90)
        l_umap.addWidget(self.cmb_metric, 0, 7)

        l_umap.addWidget(QLabel("Smooth:"), 1, 0)
        self.inp_smooth_u = QSpinBox()
        self.inp_smooth_u.setRange(0, 1000)
        self.inp_smooth_u.setValue(125)
        self.inp_smooth_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_smooth_u, 1, 1)

        l_umap.addWidget(QLabel("Pts:"), 1, 2)
        self.inp_pts_u = QSpinBox()
        self.inp_pts_u.setRange(1, 1000)
        self.inp_pts_u.setValue(20)
        self.inp_pts_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_pts_u, 1, 3)

        l_umap.addWidget(QLabel("SNR:"), 1, 4)
        self.inp_snr_u = QDoubleSpinBox()
        self.inp_snr_u.setRange(0.0, 100.0)
        self.inp_snr_u.setSingleStep(0.05)
        self.inp_snr_u.setValue(0.5)
        self.inp_snr_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_snr_u, 1, 5)

        l_umap.addWidget(QLabel("Outliers:"), 1, 6)
        self.inp_outliers_u = QDoubleSpinBox()
        self.inp_outliers_u.setRange(0.0, 0.99)
        self.inp_outliers_u.setSingleStep(0.05)
        self.inp_outliers_u.setValue(0.10)
        self.inp_outliers_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_outliers_u, 1, 7)

        l_umap.addWidget(QLabel("Notch Q:"), 2, 0)
        self.inp_notch_u = QDoubleSpinBox()
        self.inp_notch_u.setRange(0.1, 100.0)
        self.inp_notch_u.setSingleStep(0.5)
        self.inp_notch_u.setValue(2.0)
        self.inp_notch_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_notch_u, 2, 1)

        l_umap.addWidget(QLabel("Gate:"), 2, 2)
        self.inp_gate_u = QDoubleSpinBox()
        self.inp_gate_u.setRange(0.0, 100.0)
        self.inp_gate_u.setSingleStep(0.5)
        self.inp_gate_u.setValue(8.0)
        self.inp_gate_u.setFixedWidth(60)
        l_umap.addWidget(self.inp_gate_u, 2, 3)

        g_umap.setLayout(l_umap)
        lay.addWidget(g_umap)

        g_cluster = QGroupBox("Algoritmo de Agrupamiento")
        l_cluster = QHBoxLayout()
        l_cluster.addWidget(QLabel("Evaluar UMAP:"))
        self.cmb_cluster = QComboBox()
        self.cmb_cluster.addItems(["K-Means", "GMM"])
        l_cluster.addWidget(self.cmb_cluster)
        g_cluster.setLayout(l_cluster)
        lay.addWidget(g_cluster)

        g_adv = QGroupBox("DSP Avanzado y Normalización")
        l_adv = QGridLayout()
        self.chk_trevisan = QCheckBox("Aplicar Corrección Trevisan")
        self.chk_trevisan.setChecked(False)
        l_adv.addWidget(self.chk_trevisan, 0, 0, 1, 2)
        self.chk_ignorar_cero = QCheckBox("Ignorar Ventana 0")
        self.chk_ignorar_cero.setChecked(False)
        l_adv.addWidget(self.chk_ignorar_cero, 0, 2, 1, 2)
        self.chk_correccion_intersesion = QCheckBox("Corrección Intersesión por Lote (Calibración de Ganancia)")
        self.chk_correccion_intersesion.setChecked(True)
        l_adv.addWidget(self.chk_correccion_intersesion, 1, 0, 1, 4)
        l_adv.addWidget(QLabel("Pre-Ventana (%):"), 2, 0)
        self.inp_pre_pct = QDoubleSpinBox()
        self.inp_pre_pct.setRange(0.0, 1.0)
        self.inp_pre_pct.setSingleStep(0.1)
        self.inp_pre_pct.setValue(0.4)
        l_adv.addWidget(self.inp_pre_pct, 2, 1)
        l_adv.addWidget(QLabel("Post-Ventana (%):"), 2, 2)
        self.inp_post_pct = QDoubleSpinBox()
        self.inp_post_pct.setRange(0.0, 1.0)
        self.inp_post_pct.setSingleStep(0.1)
        self.inp_post_pct.setValue(0.6)
        l_adv.addWidget(self.inp_post_pct, 2, 3)
        
        l_adv.addWidget(QLabel("Gate Ratio Ruido:"), 3, 0)
        self.inp_gate = QDoubleSpinBox()
        self.inp_gate.setRange(0.0, 100.0)
        self.inp_gate.setSingleStep(0.5)
        self.inp_gate.setValue(0.0)
        self.inp_gate.setFixedWidth(60)
        l_adv.addWidget(self.inp_gate, 3, 1)

        l_adv.addWidget(QLabel("Filtro Ruido Línea:"), 3, 2)
        self.cmb_noise_filter = QComboBox()
        self.cmb_noise_filter.addItems(["Filtro Notch (IIR)", "Filtro Adaptativo (NLMS)", "Desactivado"])
        self.cmb_noise_filter.setCurrentIndex(0)
        self.cmb_noise_filter.setToolTip("Selecciona el método de supresión de interferencia de línea eléctrica (50 Hz y armónicos):\n- Filtro Notch (IIR): Notch convencional de 50 Hz con factor de calidad Q.\n- Filtro Adaptativo (NLMS): Cancelación adaptativa NLMS con armónicos sintéticos (50, 100, 150, 200 Hz).\n- Desactivado: Sin supresión de línea.")
        l_adv.addWidget(self.cmb_noise_filter, 3, 3)

        self.lbl_notch_q = QLabel("Notch Factor Q:")
        l_adv.addWidget(self.lbl_notch_q, 4, 0)
        self.inp_notch_q = QDoubleSpinBox()
        self.inp_notch_q.setRange(0.1, 100.0)
        self.inp_notch_q.setSingleStep(0.5)
        self.inp_notch_q.setValue(2.0)
        self.inp_notch_q.setFixedWidth(60)
        self.inp_notch_q.setToolTip("Factor de calidad Q para el filtro Notch IIR.")
        l_adv.addWidget(self.inp_notch_q, 4, 1)

        l_adv.addWidget(QLabel("Filtro Pasaltos:"), 4, 2)
        self.inp_highpass = QDoubleSpinBox()
        self.inp_highpass.setRange(0.0, 500.0)
        self.inp_highpass.setSingleStep(1.0)
        self.inp_highpass.setValue(20.0)
        self.inp_highpass.setSuffix(" Hz")
        self.inp_highpass.setToolTip("Frecuencia de corte inferior pasaltos (Butterworth orden 4).\n0.0 Hz desactiva el filtro pasaltos.")
        l_adv.addWidget(self.inp_highpass, 4, 3)

        l_adv.addWidget(QLabel("Filtro Pasabajos:"), 5, 0)
        self.cmb_lowpass = QComboBox()
        self.cmb_lowpass.addItems(["300 Hz (Rangayyan)", "500 Hz (Previo)", "Desactivado"])
        self.cmb_lowpass.setCurrentIndex(0)
        self.cmb_lowpass.setToolTip("Selecciona el filtrado pasabajos antes de extraer envolventes:\n- 300 Hz: Butterworth orden 6 de Rangayyan (recomendado).\n- 500 Hz: Butterworth orden 4 estándar previo.\n- Desactivado: Sin filtro pasabajos.")
        l_adv.addWidget(self.cmb_lowpass, 5, 1, 1, 3)
        
        g_adv.setLayout(l_adv)
        lay.addWidget(g_adv)

        def _on_noise_filter_changed_umap(idx):
            is_notch = (idx == 0)
            self.lbl_notch_q.setEnabled(is_notch)
            self.inp_notch_q.setEnabled(is_notch)
            if hasattr(self, 'inp_notch_u'):
                self.inp_notch_u.setEnabled(is_notch)
        self.cmb_noise_filter.currentIndexChanged.connect(_on_noise_filter_changed_umap)

        if hasattr(self, 'inp_notch_u'):
            self.inp_notch_q.valueChanged.connect(lambda v: (self.inp_notch_u.blockSignals(True), self.inp_notch_u.setValue(v), self.inp_notch_u.blockSignals(False)))
            self.inp_notch_u.valueChanged.connect(lambda v: (self.inp_notch_q.blockSignals(True), self.inp_notch_q.setValue(v), self.inp_notch_q.blockSignals(False)))

        g_align = QGroupBox("Alineación Temporal")
        l_align = QHBoxLayout()
        l_align.addWidget(QLabel("Centrar ventana en:"))
        self.cmb_align = QComboBox()
        self.cmb_align.addItems(["Pico Volumen Micrófono", "Pico Derivada Micrófono (Onset)", "Pico Canal 0", "Pico Canal 1", "Pico Canal 2"])
        l_align.addWidget(self.cmb_align)
        g_align.setLayout(l_align)
        lay.addWidget(g_align)

        lay.addStretch()
        scroll.setWidget(content)
        self.layout.addWidget(scroll)
        
        self.btn_run = QPushButton(" 2. LANZAR UMAP (NO LINEAL)")
        self.btn_run.setStyleSheet("background-color: #ff00ff; color: white; font-weight: bold; padding: 10px;")
        self.layout.addWidget(self.btn_run)

class UmapSupervisadoTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)
        
        g_dsp = QGroupBox("Filtros DSP (Extracción Base)")
        l_dsp = QGridLayout()
        
        l_dsp.addWidget(QLabel("Alpha:"), 0, 0)
        self.inp_alpha = QDoubleSpinBox()
        self.inp_alpha.setRange(0.01, 10.0)
        self.inp_alpha.setSingleStep(0.1)
        self.inp_alpha.setValue(1.0)
        l_dsp.addWidget(self.inp_alpha, 0, 1)
        
        l_dsp.addWidget(QLabel("Smooth (ms):"), 0, 2)
        self.inp_smooth = QSpinBox()
        self.inp_smooth.setRange(0, 1000)
        self.inp_smooth.setValue(125)
        l_dsp.addWidget(self.inp_smooth, 0, 3)
        
        l_dsp.addWidget(QLabel("Target Len:"), 0, 4)
        self.inp_target_len = QSpinBox()
        self.inp_target_len.setRange(1, 1000)
        self.inp_target_len.setValue(40)
        l_dsp.addWidget(self.inp_target_len, 0, 5)

        l_dsp.addWidget(QLabel("SNR Thresh:"), 1, 0)
        self.inp_snr = QDoubleSpinBox()
        self.inp_snr.setRange(0.0, 100.0)
        self.inp_snr.setValue(3.0)
        l_dsp.addWidget(self.inp_snr, 1, 1)

        l_dsp.addWidget(QLabel("Outliers:"), 1, 2)
        self.inp_outliers = QDoubleSpinBox()
        self.inp_outliers.setRange(0.0, 0.99)
        self.inp_outliers.setSingleStep(0.05)
        self.inp_outliers.setValue(0.10)
        l_dsp.addWidget(self.inp_outliers, 1, 3)
        
        l_dsp.addWidget(QLabel("Notch Q:"), 1, 4)
        self.inp_notch = QDoubleSpinBox()
        self.inp_notch.setRange(0.1, 50.0)
        self.inp_notch.setSingleStep(0.5)
        self.inp_notch.setValue(2.0)
        l_dsp.addWidget(self.inp_notch, 1, 5)

        g_dsp.setLayout(l_dsp)
        lay.addWidget(g_dsp)
        
        g_umap = QGroupBox("Hiperparámetros de Embedding UMAP")
        l_umap = QGridLayout()
        
        l_umap.addWidget(QLabel("n_neighbors:"), 0, 0)
        self.inp_umap_nn = QSpinBox()
        self.inp_umap_nn.setRange(2, 500)
        self.inp_umap_nn.setValue(5)
        l_umap.addWidget(self.inp_umap_nn, 0, 1)
        
        l_umap.addWidget(QLabel("min_dist:"), 0, 2)
        self.inp_umap_md = QDoubleSpinBox()
        self.inp_umap_md.setRange(0.0, 1.0)
        self.inp_umap_md.setSingleStep(0.05)
        self.inp_umap_md.setValue(0.8)
        l_umap.addWidget(self.inp_umap_md, 0, 3)
        
        l_umap.addWidget(QLabel("Métrica:"), 1, 0)
        self.cmb_metric = QComboBox()
        self.cmb_metric.addItems(["euclidean", "cosine", "manhattan", "correlation"])
        l_umap.addWidget(self.cmb_metric, 1, 1)
        
        l_umap.addWidget(QLabel("target_weight:"), 1, 2)
        self.inp_umap_tw = QDoubleSpinBox()
        self.inp_umap_tw.setRange(0.0, 1.0)
        self.inp_umap_tw.setSingleStep(0.1)
        self.inp_umap_tw.setValue(0.8)
        l_umap.addWidget(self.inp_umap_tw, 1, 3)
        
        self.chk_outliers_train = QCheckBox("Eliminar Outliers Espaciales del Train Set")
        self.chk_outliers_train.setChecked(False)
        l_umap.addWidget(self.chk_outliers_train, 2, 0, 1, 4)
        
        g_umap.setLayout(l_umap)
        lay.addWidget(g_umap)
        
        lay.addStretch()
        scroll.setWidget(content)
        self.layout.addWidget(scroll)
        
        self.btn_run = QPushButton(" 3. GENERAR UMAP SUPERVISADO (TRAIN/TEST)")
        self.btn_run.setStyleSheet("background-color: #45A29E; color: white; font-weight: bold; padding: 10px;")
        self.layout.addWidget(self.btn_run)

class AutoencodersTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        lay = QVBoxLayout(content)

        # 1. Parámetros DSP y Extracción Tensorial
        g_dsp = QGroupBox("1. Parámetros DSP y Extracción Tensorial")
        l_dsp = QGridLayout()
        
        l_dsp.addWidget(QLabel("Alpha:"), 0, 0)
        self.inp_alpha = QDoubleSpinBox()
        self.inp_alpha.setRange(0.0, 5.0)
        self.inp_alpha.setSingleStep(0.1)
        self.inp_alpha.setValue(1.0)
        self.inp_alpha.setFixedWidth(60)
        l_dsp.addWidget(self.inp_alpha, 0, 1)

        l_dsp.addWidget(QLabel("Smooth:"), 0, 2)
        self.inp_smooth = QSpinBox()
        self.inp_smooth.setRange(10, 1000)
        self.inp_smooth.setSingleStep(10)
        self.inp_smooth.setValue(150)
        self.inp_smooth.setFixedWidth(60)
        l_dsp.addWidget(self.inp_smooth, 0, 3)

        l_dsp.addWidget(QLabel("Pts:"), 0, 4)
        self.inp_pts = QSpinBox()
        self.inp_pts.setRange(10, 500)
        self.inp_pts.setSingleStep(10)
        self.inp_pts.setValue(100)
        self.inp_pts.setFixedWidth(60)
        l_dsp.addWidget(self.inp_pts, 0, 5)

        l_dsp.addWidget(QLabel("SNR:"), 0, 6)
        self.inp_snr = QDoubleSpinBox()
        self.inp_snr.setRange(0.0, 50.0)
        self.inp_snr.setSingleStep(0.1)
        self.inp_snr.setValue(0.5)
        self.inp_snr.setFixedWidth(60)
        l_dsp.addWidget(self.inp_snr, 0, 7)

        l_dsp.addWidget(QLabel("Outliers:"), 0, 8)
        self.inp_outliers = QDoubleSpinBox()
        self.inp_outliers.setRange(0.0, 0.99)
        self.inp_outliers.setSingleStep(0.01)
        self.inp_outliers.setValue(0.05)
        self.inp_outliers.setFixedWidth(60)
        l_dsp.addWidget(self.inp_outliers, 0, 9)

        l_dsp.addWidget(QLabel("Notch Q:"), 0, 10)
        self.inp_notch = QDoubleSpinBox()
        self.inp_notch.setRange(0.1, 100.0)
        self.inp_notch.setSingleStep(0.5)
        self.inp_notch.setValue(2.0)
        self.inp_notch.setFixedWidth(50)
        l_dsp.addWidget(self.inp_notch, 0, 11)

        g_dsp.setLayout(l_dsp)
        lay.addWidget(g_dsp)

        # 2. Parámetros de Red Neuronal
        g_nn = QGroupBox("2. Parámetros de Red Neuronal (Autoencoder 1D)")
        l_nn = QGridLayout()

        l_nn.addWidget(QLabel("Épocas:"), 0, 0)
        self.inp_epochs = QSpinBox()
        self.inp_epochs.setRange(1, 1000)
        self.inp_epochs.setValue(80)
        self.inp_epochs.setFixedWidth(60)
        l_nn.addWidget(self.inp_epochs, 0, 1)

        l_nn.addWidget(QLabel("Batch Size:"), 0, 2)
        self.inp_batch = QSpinBox()
        self.inp_batch.setRange(1, 512)
        self.inp_batch.setValue(16)
        self.inp_batch.setFixedWidth(60)
        l_nn.addWidget(self.inp_batch, 0, 3)

        l_nn.addWidget(QLabel("Latent Dim:"), 0, 4)
        self.inp_latent = QSpinBox()
        self.inp_latent.setRange(1, 256)
        self.inp_latent.setValue(8)
        self.inp_latent.setFixedWidth(60)
        l_nn.addWidget(self.inp_latent, 0, 5)

        l_nn.addWidget(QLabel("Kernel Size:"), 0, 6)
        self.inp_kernel = QSpinBox()
        self.inp_kernel.setRange(1, 31)
        self.inp_kernel.setSingleStep(2)
        self.inp_kernel.setValue(5)
        self.inp_kernel.setFixedWidth(60)
        l_nn.addWidget(self.inp_kernel, 0, 7)

        l_nn.addWidget(QLabel("Alpha Loss:"), 0, 8)
        self.inp_alpha_loss = QDoubleSpinBox()
        self.inp_alpha_loss.setRange(0.0, 1.0)
        self.inp_alpha_loss.setSingleStep(0.05)
        self.inp_alpha_loss.setValue(0.5)
        self.inp_alpha_loss.setFixedWidth(60)
        l_nn.addWidget(self.inp_alpha_loss, 0, 9)

        g_nn.setLayout(l_nn)
        lay.addWidget(g_nn)

        # 3. Opciones y Exclusiones
        g_opt = QGroupBox("3. Opciones de Entrenamiento y Exclusiones")
        l_opt = QHBoxLayout()
        self.chk_manual_excl = QCheckBox("Aplicar Exclusiones Manuales (metadata.json)")
        self.chk_manual_excl.setChecked(True)
        l_opt.addWidget(self.chk_manual_excl)

        self.chk_force_epochs = QCheckBox("Forzar Épocas (Ignorar Checkpoint)")
        self.chk_force_epochs.setChecked(False)
        l_opt.addWidget(self.chk_force_epochs)
        g_opt.setLayout(l_opt)
        lay.addWidget(g_opt)

        # 4. Partición de Sesiones Físicas (Train / Test)
        g_split = QGroupBox("4. Partición de Sesiones Físicas (Train / Test)")
        l_split = QVBoxLayout()

        l_split_top = QHBoxLayout()
        self.btn_sync_sesiones = QPushButton(" Cargar Selección del Explorador")
        self.btn_sync_sesiones.setStyleSheet("background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; font-weight: bold; padding: 5px;")
        l_split_top.addWidget(self.btn_sync_sesiones)

        self.btn_auto_split = QPushButton(" Partición Rápida (80/20)")
        self.btn_auto_split.setStyleSheet("background-color: #1F2833; color: #FFE600; border: 1px solid #FFE600; font-weight: bold; padding: 5px;")
        self.btn_auto_split.clicked.connect(self._auto_split)
        l_split_top.addWidget(self.btn_auto_split)

        self.btn_limpiar_sesiones = QPushButton(" Limpiar")
        self.btn_limpiar_sesiones.setStyleSheet("background-color: #1F2833; color: #ff0055; border: 1px solid #ff0055; font-weight: bold; padding: 5px;")
        self.btn_limpiar_sesiones.clicked.connect(self._clear_sessions)
        l_split_top.addWidget(self.btn_limpiar_sesiones)

        self.lbl_split_status = QLabel("Total: 0 | Train: 0 | Test: 0")
        self.lbl_split_status.setStyleSheet("color: #00FF00; font-weight: bold; padding-left: 10px;")
        l_split_top.addWidget(self.lbl_split_status)
        l_split_top.addStretch()
        l_split.addLayout(l_split_top)

        l_lists = QHBoxLayout()

        v_train = QVBoxLayout()
        v_train.addWidget(QLabel("Sesiones de Entrenamiento (Train):"))
        self.lst_train = QListWidget()
        self.lst_train.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst_train.setFixedHeight(130)
        self.lst_train.setStyleSheet("background-color: #0c0c0c; color: #66FCF1; border: 1px solid #45A29E;")
        v_train.addWidget(self.lst_train)
        l_lists.addLayout(v_train)

        v_arrows = QVBoxLayout()
        v_arrows.addStretch()
        self.btn_to_test = QPushButton(">>")
        self.btn_to_test.setToolTip("Mover sesiones seleccionadas a Test")
        self.btn_to_test.setStyleSheet("background-color: #45A29E; color: black; font-weight: bold; padding: 6px 12px;")
        self.btn_to_test.clicked.connect(self._move_to_test)

        self.btn_to_train = QPushButton("<<")
        self.btn_to_train.setToolTip("Mover sesiones seleccionadas a Train")
        self.btn_to_train.setStyleSheet("background-color: #45A29E; color: black; font-weight: bold; padding: 6px 12px;")
        self.btn_to_train.clicked.connect(self._move_to_train)

        v_arrows.addWidget(self.btn_to_test)
        v_arrows.addWidget(self.btn_to_train)
        v_arrows.addStretch()
        l_lists.addLayout(v_arrows)

        v_test = QVBoxLayout()
        v_test.addWidget(QLabel("Sesiones de Testeo Ciego (Test):"))
        self.lst_test = QListWidget()
        self.lst_test.setSelectionMode(QListWidget.ExtendedSelection)
        self.lst_test.setFixedHeight(130)
        self.lst_test.setStyleSheet("background-color: #0c0c0c; color: #FFE600; border: 1px solid #FFE600;")
        v_test.addWidget(self.lst_test)
        l_lists.addLayout(v_test)

        l_split.addLayout(l_lists)
        g_split.setLayout(l_split)
        lay.addWidget(g_split)

        lay.addStretch()
        scroll.setWidget(content)
        self.layout.addWidget(scroll)

        # Botones de Acción (Estilo Cyberpunk)
        self.btn_grid_search = QPushButton(" GRID SEARCH AUTOENCODER (36 COMBINACIONES)")
        self.btn_grid_search.setStyleSheet("background-color: #ffe600; color: black; font-weight: bold; padding: 10px; margin-bottom: 3px;")
        self.layout.addWidget(self.btn_grid_search)

        l_main_btns = QHBoxLayout()
        self.btn_extraer = QPushButton("1. EXTRAER DATASET")
        self.btn_extraer.setStyleSheet("background-color: #45A29E; color: black; font-weight: bold; padding: 10px;")
        self.btn_entrenar = QPushButton("2. ENTRENAR AUTOENCODER")
        self.btn_entrenar.setStyleSheet("background-color: #66FCF1; color: black; font-weight: bold; padding: 10px;")
        self.btn_plotear = QPushButton("3. PLOTEAR ESPACIO LATENTE")
        self.btn_plotear.setStyleSheet("background-color: #00FF00; color: black; font-weight: bold; padding: 10px;")
        l_main_btns.addWidget(self.btn_extraer)
        l_main_btns.addWidget(self.btn_entrenar)
        l_main_btns.addWidget(self.btn_plotear)
        self.layout.addLayout(l_main_btns)

        l_sub_btns = QHBoxLayout()
        self.btn_decodificador = QPushButton(" DECODIFICAR SECUENCIA CONTINUA")
        self.btn_decodificador.setStyleSheet("background-color: #003344; color: #00FFFF; border: 1px solid #00FFFF; font-weight: bold; padding: 9px;")
        self.btn_abrir_carpeta = QPushButton(" ABRIR CARPETA DE RESULTADOS")
        self.btn_abrir_carpeta.setStyleSheet("background-color: #1F2833; color: #FFE600; border: 1px solid #FFE600; font-weight: bold; padding: 9px;")
        self.btn_visor_features = QPushButton(" VISUALIZADOR DE FEATURES")
        self.btn_visor_features.setStyleSheet("background-color: #2b0938; color: #e879f9; border: 1px solid #e879f9; font-weight: bold; padding: 9px;")
        self.btn_pipeline_gui = QPushButton(" FLUJO DE TRABAJO (GUI)")
        self.btn_pipeline_gui.setStyleSheet("background-color: #1a1a1a; color: #66FCF1; border: 1px solid #66FCF1; font-weight: bold; padding: 9px;")
        l_sub_btns.addWidget(self.btn_decodificador)
        l_sub_btns.addWidget(self.btn_abrir_carpeta)
        l_sub_btns.addWidget(self.btn_visor_features)
        l_sub_btns.addWidget(self.btn_pipeline_gui)
        self.layout.addLayout(l_sub_btns)

    def _move_to_test(self):
        items = self.lst_train.selectedItems()
        for item in items:
            self.lst_train.takeItem(self.lst_train.row(item))
            self.lst_test.addItem(item.text())
        self._update_status()

    def _move_to_train(self):
        items = self.lst_test.selectedItems()
        for item in items:
            self.lst_test.takeItem(self.lst_test.row(item))
            self.lst_train.addItem(item.text())
        self._update_status()

    def _clear_sessions(self):
        self.lst_train.clear()
        self.lst_test.clear()
        self._update_status()

    def _auto_split(self):
        all_s = [self.lst_train.item(i).text() for i in range(self.lst_train.count())] + \
                [self.lst_test.item(i).text() for i in range(self.lst_test.count())]
        if not all_s:
            return
        all_s = sorted(list(set(all_s)))
        import random
        random.seed(42)
        shuffled = list(all_s)
        random.shuffle(shuffled)
        
        # 80% de las sesiones a Train, 20% a Test (ej: Prueba1, Prueba2, Prueba3 -> Train, Prueba4 -> Test)
        n_train = max(1, int(0.8 * len(shuffled)))
        train_s = shuffled[:n_train]
        test_s = shuffled[n_train:]
        if not test_s and len(train_s) > 1:
            test_s = [train_s.pop()]
            
        self.set_sessions(sorted(train_s), sorted(test_s))

    def auto_split(self):
        self._auto_split()

    def set_sessions(self, train_list, test_list=None):
        self.lst_train.clear()
        self.lst_test.clear()
        for s in train_list:
            self.lst_train.addItem(s)
        if test_list:
            for s in test_list:
                self.lst_test.addItem(s)
        self._update_status()

    def _update_status(self):
        n_tr = self.lst_train.count()
        n_te = self.lst_test.count()
        tot = n_tr + n_te
        self.lbl_split_status.setText(f"Total: {tot} | Train: {n_tr} | Test: {n_te}")

    def get_train_sessions(self):
        return [self.lst_train.item(i).text() for i in range(self.lst_train.count())]

    def get_test_sessions(self):
        return [self.lst_test.item(i).text() for i in range(self.lst_test.count())]

class AutoencoderNoSupervisadoTab(QWidget):
    """Pestaña dedicada al Autoencoder 100% No Supervisado (Cero Etiquetas) vinculado al Gestor de Sesiones"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rutas_actuales = []
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        content = QWidget()
        self.layout = QVBoxLayout(content)
        self.layout.setContentsMargins(12, 12, 12, 12)
        self.layout.setSpacing(10)

        # Encabezado
        lbl_title = QLabel("ESTUDIO DE AUTOENCODERS NO SUPERVISADOS (CERO ETIQUETAS)")
        lbl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #66FCF1;")
        self.layout.addWidget(lbl_title)

        lbl_desc = QLabel(
            "Plataforma unificada para descubrimiento de variedades latentes y geometrías mioeléctricas.\n"
            "• Sincronización directa con el Gestor de Sesiones de la izquierda.\n"
            "• Modalidades: Envolvente 1D (200 Hz), Señal Cruda 1D (2000 Hz, GAP+GMP), Espectrogramas 2D y 3D (STFT).\n"
            "• Auditoría de Metadatos: Detección estricta de discrepancias anatómicas (músculos por canal) y BPM.\n"
            "• Normalización obligatoria por el Supremo Tricanal del pulso individual.\n"
            "• Cero Etiquetas: Función de pérdida sin etiquetas y evaluación diagnóstica ciega con GMM canónico."
        )
        lbl_desc.setStyleSheet("color: #C5C6C7; font-size: 11px;")
        self.layout.addWidget(lbl_desc)

        # 1. Sesiones del Gestor y Auditoría Anatómica
        g_ses = QGroupBox("1. Mediciones Seleccionadas del Gestor de Sesiones y Auditoría Inter-Día")
        l_ses = QVBoxLayout()
        
        bar_sync = QHBoxLayout()
        self.btn_sync_sesiones = QPushButton("Sincronizar Selección del Gestor")
        self.btn_sync_sesiones.setStyleSheet(
            "background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; font-weight: bold; padding: 6px;"
        )
        bar_sync.addWidget(self.btn_sync_sesiones)
        
        self.lbl_sesiones_count = QLabel("Total seleccionadas: 0 mediciones")
        self.lbl_sesiones_count.setStyleSheet("color: #00FF88; font-weight: bold; font-size: 11px;")
        bar_sync.addWidget(self.lbl_sesiones_count)
        bar_sync.addStretch()
        l_ses.addLayout(bar_sync)
        
        self.lst_sesiones = QListWidget()
        self.lst_sesiones.setFixedHeight(95)
        self.lst_sesiones.setStyleSheet("background-color: #0c0c0c; color: #66FCF1; border: 1px solid #333333; font-family: monospace; font-size: 11px;")
        l_ses.addWidget(self.lst_sesiones)
        
        self.lbl_alerta_audit = QLabel("[AUDITORIA]: Seleccione sesiones en el Gestor de Sesiones de la izquierda.")
        self.lbl_alerta_audit.setWordWrap(True)
        self.lbl_alerta_audit.setStyleSheet("background-color: #111111; color: #45A29E; border: 1px solid #333333; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")
        l_ses.addWidget(self.lbl_alerta_audit)
        
        g_ses.setLayout(l_ses)
        self.layout.addWidget(g_ses)

        # ----------------------------------------------------------------------
        # 2. MODALIDAD 1: ENVOLVENTE 1D
        # ----------------------------------------------------------------------
        g_env = QGroupBox("2. Modalidad: Envolvente 1D (RMS / Contorno Cinemático)")
        l_env = QGridLayout()
        l_env.setContentsMargins(6, 6, 6, 6)
        l_env.setSpacing(6)

        self.rb_env = QRadioButton("Activar Envolvente 1D")
        self.rb_env.setChecked(True)
        self.rb_env.setStyleSheet("color: #00FF88; font-weight: bold;")
        l_env.addWidget(self.rb_env, 0, 0, 1, 2)

        l_env.addWidget(QLabel("Técnica:"), 0, 2)
        self.cmb_tipo_env = QComboBox()
        self.cmb_tipo_env.addItems(["RMS", "Media Móvil", "TKEO (Teager-Kaiser)", "Hilbert"])
        self.cmb_tipo_env.setStyleSheet("background-color: #1F2833; color: #00FF88; border: 1px solid #00AA55; padding: 3px; font-weight: bold;")
        l_env.addWidget(self.cmb_tipo_env, 0, 3)

        l_env.addWidget(QLabel("Smooth:"), 0, 4)
        self.inp_smooth_ms = QSpinBox()
        self.inp_smooth_ms.setRange(10, 1000)
        self.inp_smooth_ms.setValue(90)
        self.inp_smooth_ms.setSuffix(" ms")
        self.inp_smooth_ms.setFixedWidth(70)
        l_env.addWidget(self.inp_smooth_ms, 0, 5)

        l_env.addWidget(QLabel("Puntos:"), 0, 6)
        self.inp_pts_env = QSpinBox()
        self.inp_pts_env.setRange(10, 5000)
        self.inp_pts_env.setValue(100)
        self.inp_pts_env.setFixedWidth(65)
        l_env.addWidget(self.inp_pts_env, 0, 7)

        l_env.addWidget(QLabel("Dimensión Latente:"), 1, 0)
        self.rb_dim_2d_env = QRadioButton("2D (Plano Canónico)")
        self.rb_dim_2d_env.setChecked(False)
        self.rb_dim_2d_env.setStyleSheet("color: #00FF88; font-weight: bold;")
        self.rb_dim_3d_env = QRadioButton("3D (Espacio Tridimensional)")
        self.rb_dim_3d_env.setChecked(True)
        self.rb_dim_3d_env.setStyleSheet("color: #66FCF1; font-weight: bold;")
        row_dim_env = QHBoxLayout()
        row_dim_env.addWidget(self.rb_dim_2d_env)
        row_dim_env.addWidget(self.rb_dim_3d_env)
        row_dim_env.addStretch()
        l_env.addLayout(row_dim_env, 1, 1, 1, 7)

        g_env.setLayout(l_env)
        self.layout.addWidget(g_env)

        # ----------------------------------------------------------------------
        # 3. MODALIDAD 2: SEÑAL CRUDA 1D
        # ----------------------------------------------------------------------
        g_cruda = QGroupBox("3. Modalidad: Señal Cruda 1D (2000 Hz / GAP+GMP)")
        l_cruda = QGridLayout()
        l_cruda.setContentsMargins(6, 6, 6, 6)
        l_cruda.setSpacing(6)

        self.rb_cruda = QRadioButton("Activar Señal Cruda 1D")
        self.rb_cruda.setStyleSheet("color: #66FCF1; font-weight: bold;")
        l_cruda.addWidget(self.rb_cruda, 0, 0, 1, 2)

        l_cruda.addWidget(QLabel("Puntos Temporales:"), 0, 2)
        self.inp_pts_cruda = QSpinBox()
        self.inp_pts_cruda.setRange(10, 5000)
        self.inp_pts_cruda.setValue(1000)
        self.inp_pts_cruda.setFixedWidth(70)
        l_cruda.addWidget(self.inp_pts_cruda, 0, 3)

        lbl_gap_desc = QLabel("Pooling: Dual GAP + GMP (Invarianza Temporal)")
        lbl_gap_desc.setStyleSheet("color: #45A29E; font-size: 11px;")
        l_cruda.addWidget(lbl_gap_desc, 0, 4, 1, 4)

        l_cruda.addWidget(QLabel("Dimensión Latente:"), 1, 0)
        self.rb_dim_2d_cruda = QRadioButton("2D (Plano Canónico)")
        self.rb_dim_2d_cruda.setChecked(True)
        self.rb_dim_2d_cruda.setStyleSheet("color: #00FF88; font-weight: bold;")
        self.rb_dim_3d_cruda = QRadioButton("3D (Espacio Tridimensional)")
        self.rb_dim_3d_cruda.setStyleSheet("color: #66FCF1; font-weight: bold;")
        row_dim_cruda = QHBoxLayout()
        row_dim_cruda.addWidget(self.rb_dim_2d_cruda)
        row_dim_cruda.addWidget(self.rb_dim_3d_cruda)
        row_dim_cruda.addStretch()
        l_cruda.addLayout(row_dim_cruda, 1, 1, 1, 7)

        g_cruda.setLayout(l_cruda)
        self.layout.addWidget(g_cruda)

        # ----------------------------------------------------------------------
        # 4. MODALIDAD 3: ESPECTROGRAMA (STFT)
        # ----------------------------------------------------------------------
        g_spec = QGroupBox("4. Modalidad: Espectrograma (STFT Calibrada)")
        l_spec = QGridLayout()
        l_spec.setContentsMargins(6, 6, 6, 6)
        l_spec.setSpacing(6)

        self.rb_spec = QRadioButton("Activar Espectrograma")
        self.rb_spec.setStyleSheet("color: #FFE600; font-weight: bold;")
        l_spec.addWidget(self.rb_spec, 0, 0, 1, 2)

        lbl_spec_desc = QLabel("Resolución: 32 frecuencias x 64 tiempos (Escala Logarítmica en dB)")
        lbl_spec_desc.setStyleSheet("color: #45A29E; font-size: 11px;")
        l_spec.addWidget(lbl_spec_desc, 0, 2, 1, 6)

        l_spec.addWidget(QLabel("Dimensión Latente:"), 1, 0)
        self.rb_dim_2d_spec = QRadioButton("2D (Plano Canónico)")
        self.rb_dim_2d_spec.setChecked(True)
        self.rb_dim_2d_spec.setStyleSheet("color: #00FF88; font-weight: bold;")
        self.rb_dim_3d_spec = QRadioButton("3D (Espacio Tridimensional)")
        self.rb_dim_3d_spec.setStyleSheet("color: #66FCF1; font-weight: bold;")
        row_dim_spec = QHBoxLayout()
        row_dim_spec.addWidget(self.rb_dim_2d_spec)
        row_dim_spec.addWidget(self.rb_dim_3d_spec)
        row_dim_spec.addStretch()
        l_spec.addLayout(row_dim_spec, 1, 1, 1, 7)

        g_spec.setLayout(l_spec)
        self.layout.addWidget(g_spec)

        # Agrupador exclusivo de modalidades
        from PySide6.QtWidgets import QButtonGroup
        self.grp_modalidad = QButtonGroup(self)
        self.grp_modalidad.addButton(self.rb_env)
        self.grp_modalidad.addButton(self.rb_cruda)
        self.grp_modalidad.addButton(self.rb_spec)

        self.grp_dim_env = QButtonGroup(self)
        self.grp_dim_env.addButton(self.rb_dim_2d_env)
        self.grp_dim_env.addButton(self.rb_dim_3d_env)

        self.grp_dim_cruda = QButtonGroup(self)
        self.grp_dim_cruda.addButton(self.rb_dim_2d_cruda)
        self.grp_dim_cruda.addButton(self.rb_dim_3d_cruda)

        self.grp_dim_spec = QButtonGroup(self)
        self.grp_dim_spec.addButton(self.rb_dim_2d_spec)
        self.grp_dim_spec.addButton(self.rb_dim_3d_spec)

        # ----------------------------------------------------------------------
        # 5. ALINEACIÓN DE PULSO FISIOLÓGICO
        # ----------------------------------------------------------------------
        g_align = QGroupBox("5. Alineación de Pulso Fisiológico")
        l_align = QHBoxLayout()
        l_align.setContentsMargins(6, 6, 6, 6)
        l_align.addWidget(QLabel("Método de Alineación:"))
        self.cmb_align = QComboBox()
        self.cmb_align.addItems([
            "Pico Volumen Micrófono",
            "Pico Derivada Micrófono (Onset)",
            "Pico Canal 0",
            "Pico Canal 1",
            "Pico Canal 2",
            "Pico Envolvente Muscular (Supremo)"
        ])
        self.cmb_align.setCurrentText("Pico Derivada Micrófono (Onset)")
        self.cmb_align.setStyleSheet("background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; padding: 4px; font-weight: bold;")
        l_align.addWidget(self.cmb_align)
        l_align.addStretch()
        g_align.setLayout(l_align)
        self.layout.addWidget(g_align)

        # ----------------------------------------------------------------------
        # 6. PARÁMETROS DSP PREVIOS Y ACONDICIONAMIENTO
        # ----------------------------------------------------------------------
        g_dsp = QGroupBox("6. Parámetros DSP Previos y Acondicionamiento")
        l_dsp = QGridLayout()
        l_dsp.setContentsMargins(6, 6, 6, 6)
        l_dsp.setSpacing(6)

        # Fila 0: Filtro Pasa-banda y Notch
        l_dsp.addWidget(QLabel("HP Cutoff:"), 0, 0)
        self.inp_hp = QDoubleSpinBox()
        self.inp_hp.setRange(1.0, 100.0)
        self.inp_hp.setValue(20.0)
        self.inp_hp.setSuffix(" Hz")
        self.inp_hp.setFixedWidth(65)
        l_dsp.addWidget(self.inp_hp, 0, 1)

        l_dsp.addWidget(QLabel("LP Cutoff:"), 0, 2)
        self.inp_lp = QDoubleSpinBox()
        self.inp_lp.setRange(100.0, 950.0)
        self.inp_lp.setValue(450.0)
        self.inp_lp.setSuffix(" Hz")
        self.inp_lp.setFixedWidth(70)
        l_dsp.addWidget(self.inp_lp, 0, 3)

        l_dsp.addWidget(QLabel("Notch Q:"), 0, 4)
        self.inp_notch = QDoubleSpinBox()
        self.inp_notch.setRange(0.1, 100.0)
        self.inp_notch.setValue(2.0)
        self.inp_notch.setFixedWidth(55)
        l_dsp.addWidget(self.inp_notch, 0, 5)

        l_dsp.addWidget(QLabel("Alpha Ruido:"), 0, 6)
        self.inp_alpha = QDoubleSpinBox()
        self.inp_alpha.setRange(0.0, 10.0)
        self.inp_alpha.setValue(1.0)
        self.inp_alpha.setFixedWidth(55)
        l_dsp.addWidget(self.inp_alpha, 0, 7)

        # Fila 1: Filtro de Línea (Notch vs Adaptativo), Compuerta, SNR, Outliers
        l_dsp.addWidget(QLabel("Filtro Línea:"), 1, 0)
        self.cmb_filtro_linea = QComboBox()
        self.cmb_filtro_linea.addItems([
            "Adaptativo (NLMS 50Hz+Armónicos)",
            "Notch (IIR en Cascada)"
        ])
        self.cmb_filtro_linea.setCurrentText("Adaptativo (NLMS 50Hz+Armónicos)")
        self.cmb_filtro_linea.setStyleSheet("background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; padding: 2px; font-weight: bold;")
        l_dsp.addWidget(self.cmb_filtro_linea, 1, 1, 1, 2)

        l_dsp.addWidget(QLabel("Gate Ruido:"), 1, 3)
        self.inp_gate = QDoubleSpinBox()
        self.inp_gate.setRange(0.0, 100.0)
        self.inp_gate.setValue(0.0)
        self.inp_gate.setFixedWidth(50)
        l_dsp.addWidget(self.inp_gate, 1, 4)

        l_dsp.addWidget(QLabel("SNR:"), 1, 5)
        self.inp_snr = QDoubleSpinBox()
        self.inp_snr.setRange(0.0, 100.0)
        self.inp_snr.setValue(0.5)
        self.inp_snr.setFixedWidth(50)
        l_dsp.addWidget(self.inp_snr, 1, 6)

        l_dsp.addWidget(QLabel("Outliers:"), 1, 7)
        self.inp_outliers = QDoubleSpinBox()
        self.inp_outliers.setRange(0.0, 0.50)
        self.inp_outliers.setValue(0.10)
        self.inp_outliers.setFixedWidth(50)
        l_dsp.addWidget(self.inp_outliers, 1, 8)

        # Fila 2: Ponderación de Canales
        l_dsp.addWidget(QLabel("W Ch0:"), 2, 0)
        self.inp_w0 = QDoubleSpinBox()
        self.inp_w0.setRange(0.0, 20.0)
        self.inp_w0.setValue(1.0)
        self.inp_w0.setFixedWidth(55)
        l_dsp.addWidget(self.inp_w0, 2, 1)

        l_dsp.addWidget(QLabel("W Ch1:"), 2, 2)
        self.inp_w1 = QDoubleSpinBox()
        self.inp_w1.setRange(0.0, 20.0)
        self.inp_w1.setValue(1.0)
        self.inp_w1.setFixedWidth(55)
        l_dsp.addWidget(self.inp_w1, 2, 3)

        l_dsp.addWidget(QLabel("W Ch2:"), 2, 4)
        self.inp_w2 = QDoubleSpinBox()
        self.inp_w2.setRange(0.0, 20.0)
        self.inp_w2.setValue(1.0)
        self.inp_w2.setFixedWidth(55)
        l_dsp.addWidget(self.inp_w2, 2, 5)

        g_dsp.setLayout(l_dsp)
        self.layout.addWidget(g_dsp)

        # 7. Parámetros de Optimización y Calibración
        g_par = QGroupBox("7. Parámetros de Optimización y Calibración")
        l_par = QGridLayout()
        
        # Fila 0: Parámetros numéricos base
        l_par.addWidget(QLabel("Épocas:"), 0, 0)
        self.inp_epochs = QSpinBox()
        self.inp_epochs.setRange(1, 1000)
        self.inp_epochs.setValue(150)
        self.inp_epochs.setFixedWidth(65)
        l_par.addWidget(self.inp_epochs, 0, 1)
        
        l_par.addWidget(QLabel("Batch Size:"), 0, 2)
        self.inp_batch = QSpinBox()
        self.inp_batch.setRange(1, 512)
        self.inp_batch.setValue(32)
        self.inp_batch.setFixedWidth(65)
        l_par.addWidget(self.inp_batch, 0, 3)
        
        l_par.addWidget(QLabel("Learning Rate:"), 0, 4)
        self.inp_lr = QDoubleSpinBox()
        self.inp_lr.setRange(0.0001, 0.1)
        self.inp_lr.setSingleStep(0.001)
        self.inp_lr.setDecimals(4)
        self.inp_lr.setValue(0.0020)
        self.inp_lr.setFixedWidth(75)
        l_par.addWidget(self.inp_lr, 0, 5)

        # Fila 1: Función de Pérdida (MSE vs Soft-DTW) y parámetro Gamma
        l_par.addWidget(QLabel("Función Pérdida:"), 1, 0)
        self.cmb_loss = QComboBox()
        self.cmb_loss.addItems([
            "MSE (Error Cuadrático Medio)",
            "Soft-DTW (Alineación Temporal Suave)",
            "Divergencia Soft-DTW (Simétrica)",
            "Híbrida (MSE + Soft-DTW)"
        ])
        self.cmb_loss.setCurrentText("MSE (Error Cuadrático Medio)")
        self.cmb_loss.setStyleSheet("background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; padding: 3px; font-weight: bold;")
        l_par.addWidget(self.cmb_loss, 1, 1, 1, 2)

        l_par.addWidget(QLabel("Gamma Soft-DTW:"), 1, 3)
        self.inp_gamma_sdtw = QDoubleSpinBox()
        self.inp_gamma_sdtw.setRange(0.01, 50.0)
        self.inp_gamma_sdtw.setValue(1.00)
        self.inp_gamma_sdtw.setSingleStep(0.1)
        self.inp_gamma_sdtw.setDecimals(2)
        self.inp_gamma_sdtw.setFixedWidth(65)
        l_par.addWidget(self.inp_gamma_sdtw, 1, 4)

        # Fila 2: Algoritmo de Clustering y Regularización de Ortogonalidad Latente
        l_par.addWidget(QLabel("Clustering:"), 2, 0)
        self.cmb_clustering = QComboBox()
        self.cmb_clustering.addItems([
            "GMM (Gaussian Mixture)",
            "K-Means (k-medias)"
        ])
        self.cmb_clustering.setCurrentText("GMM (Gaussian Mixture)")
        self.cmb_clustering.setStyleSheet("background-color: #1F2833; color: #66FCF1; border: 1px solid #45A29E; padding: 3px; font-weight: bold;")
        l_par.addWidget(self.cmb_clustering, 2, 1, 1, 2)

        l_par.addWidget(QLabel("Ortogonalidad Latente (\u03bb):"), 2, 3)
        self.inp_lambda_orto = QDoubleSpinBox()
        self.inp_lambda_orto.setRange(0.0, 50.0)
        self.inp_lambda_orto.setValue(0.0)
        self.inp_lambda_orto.setSingleStep(0.1)
        self.inp_lambda_orto.setDecimals(2)
        self.inp_lambda_orto.setFixedWidth(65)
        self.inp_lambda_orto.setToolTip("Penaliza la covarianza no diagonal en el espacio latente para forzar que los ejes sean ortogonales.")
        l_par.addWidget(self.inp_lambda_orto, 2, 4)

        # Fila 3: Reescalado Fisiológico Directo por Promedios
        self.chk_p95 = QCheckBox("Reescalado Fisiológico por Promedios (Rojo en /a/ -> 1.0, Verde en /i/ -> 1.0, Amarillo en /u/ -> 1.0)")
        self.chk_p95.setChecked(True)
        self.chk_p95.setStyleSheet("color: #66FCF1; font-weight: bold;")
        l_par.addWidget(self.chk_p95, 3, 0, 1, 6)
        
        g_par.setLayout(l_par)
        self.layout.addWidget(g_par)

        # 8. Arquitectura de Red Neuronal (Editor de Código PyTorch)
        g_arch = QGroupBox("8. Arquitectura de Red Neuronal (Editor de Código PyTorch)")
        l_arch = QVBoxLayout()

        bar_arch = QHBoxLayout()
        self.chk_usar_custom = QCheckBox("Usar Arquitectura Personalizada del Editor (PyTorch nn.Module)")
        self.chk_usar_custom.setChecked(False)
        self.chk_usar_custom.setStyleSheet("color: #FFE600; font-weight: bold;")
        bar_arch.addWidget(self.chk_usar_custom)

        self.btn_restablecer_arch = QPushButton("Restablecer Plantilla Oficial")
        self.btn_restablecer_arch.setStyleSheet("background-color: #1F2833; color: #45A29E; border: 1px solid #45A29E; font-size: 11px; padding: 4px;")
        bar_arch.addWidget(self.btn_restablecer_arch)

        self.btn_verificar_arch = QPushButton("Verificar Sintaxis y Capas")
        self.btn_verificar_arch.setStyleSheet("background-color: #1F2833; color: #00FF88; border: 1px solid #00AA55; font-size: 11px; font-weight: bold; padding: 4px;")
        bar_arch.addWidget(self.btn_verificar_arch)
        bar_arch.addStretch()
        l_arch.addLayout(bar_arch)

        self.txt_codigo_arch = QPlainTextEdit()
        self.txt_codigo_arch.setFixedHeight(230)
        self.txt_codigo_arch.setStyleSheet(
            "background-color: #0c0c0c; color: #66FCF1; border: 1px solid #333333; "
            "font-family: Consolas, Courier New, monospace; font-size: 11px; padding: 6px;"
        )
        self.txt_codigo_arch.setPlainText(self.get_plantilla_codigo("envolvente"))
        l_arch.addWidget(self.txt_codigo_arch)

        self.lbl_arch_status = QLabel("[INFO]: Arquitectura oficial pre-cargada con Invarianza Temporal (GAP + GMP).")
        self.lbl_arch_status.setStyleSheet("background-color: #111111; color: #45A29E; border: 1px solid #333333; padding: 5px; font-family: monospace; font-size: 10px; border-radius: 4px;")
        l_arch.addWidget(self.lbl_arch_status)

        g_arch.setLayout(l_arch)
        self.layout.addWidget(g_arch)

        # Conectar eventos de la arquitectura
        self.btn_restablecer_arch.clicked.connect(self.on_restablecer_plantilla)
        self.btn_verificar_arch.clicked.connect(self.on_verificar_arquitectura)
        self.rb_env.toggled.connect(self.on_modalidad_toggled)
        self.rb_cruda.toggled.connect(self.on_modalidad_toggled)
        self.rb_spec.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_2d_env.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_3d_env.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_2d_cruda.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_3d_cruda.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_2d_spec.toggled.connect(self.on_modalidad_toggled)
        self.rb_dim_3d_spec.toggled.connect(self.on_modalidad_toggled)

        # 9. Acciones de Procesamiento y Entrenamiento
        g_act = QGroupBox("9. Acciones de Procesamiento y Entrenamiento")
        l_act = QVBoxLayout()
        
        row_steps = QHBoxLayout()
        self.btn_extraer = QPushButton("1. Extraer Dataset")
        self.btn_extraer.setStyleSheet("background-color: #1F2833; color: #45A29E; font-weight: bold; padding: 8px; border: 1px solid #45A29E;")
        row_steps.addWidget(self.btn_extraer)
        
        self.btn_entrenar = QPushButton("2. Entrenar Autoencoder")
        self.btn_entrenar.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 8px; border: 1px solid #66FCF1;")
        row_steps.addWidget(self.btn_entrenar)
        
        self.btn_plotear = QPushButton("3. Plotear Espacio Latente")
        self.btn_plotear.setStyleSheet("background-color: #1F2833; color: #00FF88; font-weight: bold; padding: 8px; border: 1px solid #00FF88;")
        row_steps.addWidget(self.btn_plotear)
        l_act.addLayout(row_steps)
        
        self.btn_flujo_completo = QPushButton("FLUJO COMPLETO: EXTRACCION -> ENTRENAMIENTO -> EVALUACION (1-CLICK)")
        self.btn_flujo_completo.setStyleSheet("background-color: #004d33; color: #00FF88; font-weight: bold; font-size: 12px; padding: 12px; border: 2px solid #00FF88; border-radius: 4px;")
        l_act.addWidget(self.btn_flujo_completo)
        
        row_aux = QHBoxLayout()
        self.btn_lanzar_estudio = QPushButton("Abrir Estudio Gráfico Completo (GUI Externa)")
        self.btn_lanzar_estudio.setStyleSheet("background-color: #1F2833; color: #66FCF1; font-weight: bold; padding: 7px; border: 1px solid #45A29E;")
        row_aux.addWidget(self.btn_lanzar_estudio)

        self.btn_ver_ultimo_grafico = QPushButton("Visualizar Último Gráfico (PNG)")
        self.btn_ver_ultimo_grafico.setStyleSheet("background-color: #1F2833; color: #00FF88; font-weight: bold; padding: 7px; border: 1px solid #00AA55;")
        row_aux.addWidget(self.btn_ver_ultimo_grafico)
        
        self.btn_abrir_resultados = QPushButton("Abrir Carpeta de Resultados")
        self.btn_abrir_resultados.setStyleSheet("background-color: #1F2833; color: #FFE600; font-weight: bold; padding: 7px; border: 1px solid #FFE600;")
        row_aux.addWidget(self.btn_abrir_resultados)
        l_act.addLayout(row_aux)
        
        g_act.setLayout(l_act)
        self.layout.addWidget(g_act)

        self.layout.addStretch()
        scroll.setWidget(content)
        main_layout.addWidget(scroll)

    def get_plantilla_codigo(self, modalidad=None):
        if modalidad is None:
            if self.rb_cruda.isChecked():
                modalidad = "cruda"
            elif self.rb_spec.isChecked():
                modalidad = "espectrograma"
            else:
                modalidad = "envolvente"

        latent_dim = 3 if self.rb_dim_3d.isChecked() else 2

        if modalidad == "cruda":
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderPersonalizado(nn.Module):
    """
    Autoencoder Convolucional 1D para Señal Cruda con Invarianza Temporal (Línea Base Histórica 54.3%).
    Utiliza núcleos amplios con ReLU para rectificación aprendida y pooling dual (GAP + GMP).
    """
    def __init__(self, in_channels=3, latent_dim={latent_dim}, target_len=1000):
        super().__init__()
        self.target_len = target_len
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=31, padding=15),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=15, padding=7),
            nn.ReLU()
        )
        # Cálculo dinámico de canales para el pooling (GAP + GMP)
        with torch.no_grad():
            num_features = self.conv(torch.zeros(1, in_channels, self.target_len)).shape[1]

        # Invarianza Temporal: Global Average Pooling + Global Max Pooling (GAP + GMP)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc_enc = nn.Sequential(
            nn.Linear(num_features * 2, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, in_channels * target_len)
        )

    def encode(self, x):
        h = self.conv(x)
        avg_f = self.gap(h).squeeze(-1)
        max_f = self.gmp(h).squeeze(-1)
        z = self.fc_enc(torch.cat([avg_f, max_f], dim=1))
        return z

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decoder(z).view(-1, self.in_channels, self.target_len)
        if x.shape[-1] != self.target_len:
            x_rec = F.interpolate(x_rec, size=x.shape[-1], mode='linear', align_corners=False)
        return x_rec, z
'''
        elif modalidad.startswith("espectrograma"):
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderPersonalizado(nn.Module):
    """Autoencoder 2D para espectrogramas calibrados en dB (3x32x64)"""
    def __init__(self, in_channels=3, latent_dim={latent_dim}):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        act = lambda: nn.LeakyReLU(0.1)

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), act(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), act(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 64), act(),
            nn.Linear(64, latent_dim)
        )
        self.fc_dec = nn.Sequential(
            nn.Linear(latent_dim, 64), act(),
            nn.Linear(64, 64 * 4 * 8), act()
        )
        self.up3 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), act())
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), act())
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), act())
        self.out_conv = nn.Conv2d(16, in_channels, 3, padding=1)

    def forward(self, x):
        h = self.enc(x)
        z = self.fc_enc(h)
        h_dec = self.fc_dec(z).view(z.shape[0], 64, 4, 8)
        d = self.dec3(self.up3(h_dec))
        d = self.dec2(self.up2(d))
        d = self.dec1(self.up1(d))
        x_rec = self.out_conv(d)
        return x_rec, z
'''
        else:
            return f'''import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoencoderPersonalizado(nn.Module):
    """
    Autoencoder Convolucional 1D para envolventes continuas con Invarianza Temporal.
    Utiliza pooling dual (GAP + GMP) para capturar la integral de activación y el pico
    de contracción de forma completamente independiente de la longitud temporal.
    """
    def __init__(self, in_channels=3, latent_dim={latent_dim}, target_len=100):
        super().__init__()
        self.in_channels = in_channels
        self.target_len = target_len
        self.latent_dim = latent_dim
        act = lambda: nn.LeakyReLU(0.1)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, padding=3), act(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), act(),
            nn.Conv1d(32, 16, kernel_size=5, padding=2), act()
        )
        # Cálculo dinámico de canales para el pooling (GAP + GMP)
        with torch.no_grad():
            num_features = self.conv(torch.zeros(1, in_channels, self.target_len)).shape[1]

        # Reducción Global Independiente del Tiempo (GAP + GMP)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.fc_enc = nn.Sequential(
            nn.Linear(num_features * 2, 32),
            act(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            act(),
            nn.Linear(32, in_channels * target_len)
        )

    def encode(self, x):
        h = self.conv(x)
        avg_f = self.gap(h).squeeze(-1)
        max_f = self.gmp(h).squeeze(-1)
        z = self.fc_enc(torch.cat([avg_f, max_f], dim=1))
        return z

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decoder(z).view(-1, self.in_channels, self.target_len)
        if x.shape[-1] != self.target_len:
            x_rec = F.interpolate(x_rec, size=x.shape[-1], mode='linear', align_corners=False)
        return x_rec, z
'''

    @property
    def rb_dim_2d(self):
        class _D2Proxy:
            def __init__(proxy_self, parent):
                proxy_self.parent = parent
            def isChecked(proxy_self):
                if proxy_self.parent.rb_cruda.isChecked():
                    return proxy_self.parent.rb_dim_2d_cruda.isChecked()
                elif proxy_self.parent.rb_spec.isChecked():
                    return proxy_self.parent.rb_dim_2d_spec.isChecked()
                return proxy_self.parent.rb_dim_2d_env.isChecked()
        return _D2Proxy(self)

    @property
    def rb_dim_3d(self):
        class _D3Proxy:
            def __init__(proxy_self, parent):
                proxy_self.parent = parent
            def isChecked(proxy_self):
                if proxy_self.parent.rb_cruda.isChecked():
                    return proxy_self.parent.rb_dim_3d_cruda.isChecked()
                elif proxy_self.parent.rb_spec.isChecked():
                    return proxy_self.parent.rb_dim_3d_spec.isChecked()
                return proxy_self.parent.rb_dim_3d_env.isChecked()
        return _D3Proxy(self)

    @property
    def inp_pts(self):
        if hasattr(self, 'rb_cruda') and self.rb_cruda.isChecked():
            return self.inp_pts_cruda
        return self.inp_pts_env

    def on_modalidad_toggled(self):
        if not self.chk_usar_custom.isChecked():
            self.txt_codigo_arch.setPlainText(self.get_plantilla_codigo())
            self.lbl_arch_status.setText("[INFO]: Plantilla actualizada para la modalidad seleccionada.")
            self.lbl_arch_status.setStyleSheet("background-color: #111111; color: #45A29E; border: 1px solid #333333; padding: 5px; font-family: monospace; font-size: 10px; border-radius: 4px;")

    def on_restablecer_plantilla(self):
        self.txt_codigo_arch.setPlainText(self.get_plantilla_codigo())
        self.lbl_arch_status.setText("[INFO]: Plantilla oficial restablecida.")
        self.lbl_arch_status.setStyleSheet("background-color: #111111; color: #45A29E; border: 1px solid #333333; padding: 5px; font-family: monospace; font-size: 10px; border-radius: 4px;")

    def on_verificar_arquitectura(self):
        codigo = self.txt_codigo_arch.toPlainText()
        modalidad = "cruda" if self.rb_cruda.isChecked() else ("espectrograma" if self.rb_spec.isChecked() else "envolvente")
        latent_dim = 3 if self.rb_dim_3d.isChecked() else 2
        try:
            import deep_learning.motor_autoencoder_unificado as motor
            ok, msg = motor.verificar_arquitectura_codigo(codigo, modalidad=modalidad, latent_dim=latent_dim)
            if ok:
                self.lbl_arch_status.setText(f"[VALIDACION OK]: {msg}")
                self.lbl_arch_status.setStyleSheet("background-color: #002211; color: #00FF88; border: 1px solid #00AA55; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")
            else:
                self.lbl_arch_status.setText(f"[ERROR]: {msg}")
                self.lbl_arch_status.setStyleSheet("background-color: #330011; color: #FF4444; border: 1px solid #FF0055; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px; font-weight: bold;")
        except Exception as e:
            self.lbl_arch_status.setText(f"[ERROR]: {e}")
            self.lbl_arch_status.setStyleSheet("background-color: #330011; color: #FF4444; border: 1px solid #FF0055; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")

    def set_sessions(self, rutas):
        self.rutas_actuales = rutas or []
        n = len(self.rutas_actuales)
        self.lbl_sesiones_count.setText(f"Total seleccionadas: {n} mediciones")
        
        self.lst_sesiones.clear()
        for r in self.rutas_actuales:
            nombre = os.path.basename(r)
            padre = os.path.basename(os.path.dirname(r))
            self.lst_sesiones.addItem(f"[{padre}] {nombre}")
            
        if n == 0:
            self.lbl_alerta_audit.setText("[AUDITORIA]: No hay sesiones seleccionadas en el Gestor de Sesiones.")
            self.lbl_alerta_audit.setStyleSheet("background-color: #111111; color: #888888; border: 1px solid #333333; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")
            self.btn_extraer.setEnabled(False)
            self.btn_flujo_completo.setEnabled(False)
            return

        self.btn_extraer.setEnabled(True)
        self.btn_flujo_completo.setEnabled(True)

        try:
            import deep_learning.motor_autoencoder_unificado as motor
            res = motor.auditar_metadatos_sesiones(self.rutas_actuales)
            if not res['compatible']:
                adv_txt = " | ".join(res['advertencias'][:2])
                self.lbl_alerta_audit.setText(f"[ALERTA ANATOMICA INTER-DIA]: {adv_txt}")
                self.lbl_alerta_audit.setStyleSheet("background-color: #330011; color: #FF4444; border: 1px solid #FF0055; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px; font-weight: bold;")
            else:
                m_res = res.get('musculos_resumen', {})
                bpm_val = res.get('bpm_comun', 'N/D') or "N/D"
                fechas_d = res.get('fechas_detectadas', [])
                fechas_txt = f" | Fechas: {', '.join(fechas_d)}" if fechas_d else ""
                self.lbl_alerta_audit.setText(f"[COHERENCIA ANATOMICA OK] {n} tomas validadas{fechas_txt} | Ch0: {m_res.get('canal_0', 'N/D')}, Ch1: {m_res.get('canal_1', 'N/D')}, Ch2: {m_res.get('canal_2', 'N/D')} | Metrónomo: {bpm_val} BPM")
                self.lbl_alerta_audit.setStyleSheet("background-color: #002211; color: #00FF88; border: 1px solid #00AA55; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")
        except Exception as e:
            self.lbl_alerta_audit.setText(f"[AUDITORIA]: Error al inspeccionar metadatos: {e}")
            self.lbl_alerta_audit.setStyleSheet("background-color: #221100; color: #FFAA00; border: 1px solid #FFAA00; padding: 6px; font-family: monospace; font-size: 10px; border-radius: 4px;")

    def get_kwargs(self):
        if self.rb_cruda.isChecked():
            modalidad = "cruda"
            latent_dim = 3 if self.rb_dim_3d_cruda.isChecked() else 2
            target_len = self.inp_pts_cruda.value()
        elif self.rb_spec.isChecked():
            modalidad = "espectrograma"
            latent_dim = 3 if self.rb_dim_3d_spec.isChecked() else 2
            target_len = 1000
        else:
            modalidad = "envolvente"
            latent_dim = 3 if self.rb_dim_3d_env.isChecked() else 2
            target_len = self.inp_pts_env.value()

        env_text = self.cmb_tipo_env.currentText().lower()
        if "media" in env_text:
            tipo_env = "media_movil"
        elif "tkeo" in env_text:
            tipo_env = "tkeo"
        elif "hilbert" in env_text:
            tipo_env = "hilbert"
        else:
            tipo_env = "rms"

        loss_text = self.cmb_loss.currentText().lower()
        if "divergencia" in loss_text:
            tipo_perdida = "soft_dtw_divergence"
        elif "híbrida" in loss_text or "hibrida" in loss_text:
            tipo_perdida = "hibrida"
        elif "soft-dtw" in loss_text or "soft_dtw" in loss_text:
            tipo_perdida = "soft_dtw"
        else:
            tipo_perdida = "mse"

        gamma_sdtw = float(self.inp_gamma_sdtw.value())
        clustering_text = self.cmb_clustering.currentText().lower()
        algoritmo_clustering = "kmeans" if "k-means" in clustering_text or "kmeans" in clustering_text else "gmm"
        lambda_orto = float(self.inp_lambda_orto.value())
        filtro_linea_text = self.cmb_filtro_linea.currentText().lower()
        tipo_filtro_linea = "notch" if "notch" in filtro_linea_text else "adaptativo"
            
        return {
            'modalidad': modalidad,
            'latent_dim': latent_dim,
            'epochs': self.inp_epochs.value(),
            'batch_size': self.inp_batch.value(),
            'lr': float(self.inp_lr.value()),
            'tipo_perdida': tipo_perdida,
            'gamma_sdtw': gamma_sdtw,
            'algoritmo_clustering': algoritmo_clustering,
            'lambda_orto': lambda_orto,
            'tipo_filtro_linea': tipo_filtro_linea,
            'usar_calibracion_p95': self.chk_p95.isChecked(),
            'usar_custom_arch': self.chk_usar_custom.isChecked(),
            'codigo_custom_arch': self.txt_codigo_arch.toPlainText(),
            'modo_alineacion': self.cmb_align.currentText(),
            'tipo_envolvente': tipo_env,
            'smooth_ms': self.inp_smooth_ms.value(),
            'alpha_ruido': float(self.inp_alpha.value()),
            'target_len': target_len,
            'highpass_cutoff_hz': float(self.inp_hp.value()),
            'lowpass_cutoff_hz': float(self.inp_lp.value()),
            'snr_min': float(self.inp_snr.value()),
            'outlier_contamination': float(self.inp_outliers.value()),
            'notch_q': float(self.inp_notch.value()),
            'gate_ratio_ruido': float(self.inp_gate.value()),
            'w_canales': [float(self.inp_w0.value()), float(self.inp_w1.value()), float(self.inp_w2.value())],
            'rutas': list(self.rutas_actuales)
        }

class MachineLearningTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        self.tabs = QTabWidget()
        
        self.tab_pca = PcaTab()
        self.tab_umap = UmapTab()
        self.tab_umap_sup = UmapSupervisadoTab()
        self.tab_autoencoders = AutoencodersTab()
        
        self.tabs.addTab(self.tab_pca, "PCA")
        self.tabs.addTab(self.tab_umap, "UMAP No-Lineal")
        self.tabs.addTab(self.tab_umap_sup, "UMAP Supervisado")
        self.tabs.addTab(self.tab_autoencoders, "Autoencoders")
        
        layout.addWidget(self.tabs)

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
                background: #111; color: #aaa; border: 2px solid #333; padding: 10px; font-weight: bold;
            }
            QTabBar::tab:selected { background: #ff0033; color: #fff; border-color: #ff0033; }
        """)
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # Pestaña Individual
        self.tab_procesamiento = ProcessingTab()
        self.tabs.addTab(self.tab_procesamiento, " Procesamiento Individual")
        
        # Pestaña Comparativa
        self.tab_comparativo = ComparativeTab()
        self.tabs.addTab(self.tab_comparativo, " Análisis Comparativo")
        
        self.layout.addWidget(self.tabs)
        
    def get_processing_kwargs(self):
        """Obtiene los parámetros de procesamiento delegando a la pestaña de procesamiento."""
        return self.tab_procesamiento.get_processing_kwargs()

    def get_trevisan_kwargs(self):
        t = self.tab_procesamiento
        smooth_val = 50.0
        try:
            smooth_val = float(t.inp_smooth.text().strip() or 50.0)
        except (ValueError, TypeError):
            smooth_val = 50.0
        return {
            'alpha_ruido': 1.0,
            'snr_threshold': 3.0,
            'smooth_ms': smooth_val,
            'n_pts_window': 100
        }

    def get_comparative_kwargs(self):
        t = self.tab_comparativo
        return {
            'show_overlay': t.chk_overlay.isChecked(),
            'show_snr': t.chk_snr.isChecked(),
            'show_amplitude': t.chk_amp.isChecked(),
            'show_snr_time': t.chk_snr_time.isChecked(),
            'show_amp_time': t.chk_amp_time.isChecked(),
            'show_table': t.chk_table.isChecked()
        }
        

    
class MachineLearningPanel(QWidget):
    """Contenedor para la sección de Deep Learning / Machine Learning"""
    def __init__(self):
        super().__init__()
        self.setStyleSheet(
            "QWidget { background-color: #050505; color: #00ffcc; font-family: 'Courier New', monospace; }"
            "QTabWidget::pane { border: 2px solid #ff00ff; border-radius: 4px; background: #050505; }"
            "QTabBar::tab { background: #111; color: #aaa; border: 2px solid #333; padding: 10px; font-weight: bold; }"
            "QTabBar::tab:selected { background: #ff00ff; color: #fff; border-color: #ff00ff; }"
            "QGroupBox { border: 1px solid #ff00ff; border-radius: 4px; margin-top: 10px; padding-top: 10px; font-weight: bold; color: #00ffcc; }"
            "QLabel { color: #00ffcc; }"
            "QCheckBox { color: #00ffcc; }"
            "QCheckBox::indicator:checked { background: #ff00ff; }"
            "QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { background-color: #111; color: #00ffcc; border: 1px solid #00ffcc; padding: 4px; }"
        )
        
        self.layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # 1. Umbrales (Coordenadas + Entrenamiento)
        self.tab_umbrales = QWidget()
        lyt_umbrales = QVBoxLayout(self.tab_umbrales)
        self.subtabs_umbrales = QTabWidget()
        self.tab_motor = DiscreteMotorTab()
        self.tab_training = TrainingMotorTab()
        self.subtabs_umbrales.addTab(self.tab_motor, " Coordenadas Discretas")
        self.subtabs_umbrales.addTab(self.tab_training, " Entrenamiento de Umbrales")
        lyt_umbrales.addWidget(self.subtabs_umbrales)
        self.tabs.addTab(self.tab_umbrales, "Umbrales")
        
        # 2, 3, 4. PCA, UMAP, UMAP Supervisado
        self.tab_pca = PcaTab()
        self.tabs.addTab(self.tab_pca, "PCA")
        
        self.tab_umap = UmapTab()
        self.tabs.addTab(self.tab_umap, "UMAP No-Lineal")
        
        self.tab_umap_sup = UmapSupervisadoTab()
        self.tabs.addTab(self.tab_umap_sup, "UMAP Supervisado")
        
        # 5. Autoencoders (Supervisado)
        self.tab_autoencoders = AutoencodersTab()
        self.tabs.addTab(self.tab_autoencoders, "Autoencoders (Supervisado)")
        
        # 6. Autoencoder No Supervisado (Cero Etiquetas)
        self.tab_autoencoder_no_sup = AutoencoderNoSupervisadoTab()
        self.tabs.addTab(self.tab_autoencoder_no_sup, "Autoencoder No Supervisado (Cero Etiquetas)")
        
        # 6. Otros Clasificadores y Herramientas
        self.tab_otros = QWidget()
        lyt_otros = QVBoxLayout(self.tab_otros)
        self.btn_trevisan = QPushButton("Análisis de Binarización (Trevisan)")
        self.btn_trevisan.setStyleSheet("padding: 15px; font-size: 14px; background-color: #001a33; color: #00ffff; border: 1px solid #00ffff;")
        self.btn_visor_features = QPushButton("Visualizador de Features (PCA/UMAP)")
        self.btn_visor_features.setStyleSheet("padding: 15px; font-size: 14px; background-color: #330033; color: #ff00ff; border: 1px solid #ff00ff;")
        
        lyt_otros.addWidget(self.btn_trevisan)
        lyt_otros.addWidget(self.btn_visor_features)
        lyt_otros.addStretch()
        # self.tabs.addTab(self.tab_otros, "Herramientas Extra (Trevisan, Visor)")
        
        self.layout.addWidget(self.tabs)

    def get_discrete_kwargs(self):
        return {
            'n_std': self.tab_motor.inp_std_multiplier.value(),
            'vocal_orden': self.tab_motor.cmb_vocal_orden.currentText(),
            'vocal_inicio': self.tab_motor.cmb_vocal_inicio.currentText()
        }

    def get_training_kwargs(self):
        return {
            'chk_snr': self.tab_training.chk_snr.isChecked(),
            'snr_limit': self.tab_training.inp_snr_limit.value(),
            'snr_tipo': self.tab_training.cmb_snr_tipo.currentText(),
            'tipo_barrido': self.tab_training.cmb_tipo_barrido.currentText(),
            'paso_barrido': self.tab_training.inp_paso_barrido.value()
        }

    def get_pca_kwargs(self):
        t = self.tab_pca
        canales = []
        if t.chk_canal_0.isChecked(): canales.append("canal_0")
        if t.chk_canal_1.isChecked(): canales.append("canal_1")
        if t.chk_canal_2.isChecked(): canales.append("canal_2")
        
        estilo_visual = "Fronteras"
        if t.rb_elipses.isChecked(): estilo_visual = "Elipses"
        elif t.rb_sombreado.isChecked(): estilo_visual = "Sombreado"
        
        lowpass_txt = t.cmb_lowpass.currentText() if hasattr(t, 'cmb_lowpass') else "300 Hz (Rangayyan)"
        if "300" in lowpass_txt:
            lowpass_val = 300.0
        elif "500" in lowpass_txt:
            lowpass_val = 500.0
        else:
            lowpass_val = 0.0
        
        noise_txt = t.cmb_noise_filter.currentText() if hasattr(t, 'cmb_noise_filter') else "Notch (IIR)"
        if "adaptativo" in noise_txt.lower() or "nlms" in noise_txt.lower():
            tipo_filtro_ruido = "adaptativo"
        elif "desactivado" in noise_txt.lower():
            tipo_filtro_ruido = "desactivado"
        else:
            tipo_filtro_ruido = "notch"

        highpass_val = t.inp_highpass.value() if hasattr(t, 'inp_highpass') else 20.0
        notch_q_val = t.inp_notch_q.value() if hasattr(t, 'inp_notch_q') else 2.0
        
        return {
            'proc_pca_2d': t.chk_pca_2d.isChecked(),
            'proc_pca_3d': t.chk_pca_3d.isChecked(),
            'proc_umap_2d': False,
            'proc_umap_3d': False,
            'ocultar_leyenda': t.chk_ocultar_leyenda.isChecked(),
            'tipo_filtro_ruido': tipo_filtro_ruido,
            'notch_q': notch_q_val,
            'highpass_cutoff_hz': highpass_val,
            'lowpass_cutoff_hz': lowpass_val,
            'params_2d': {
                'alpha_ruido': t.inp_alpha_2d.value(),
                'gate_ratio_ruido': t.inp_gate.value() if hasattr(t, 'inp_gate') else (t.inp_gate_2d.value() if hasattr(t, 'inp_gate_2d') else 0.0),
                'smooth_ms': t.inp_smooth_2d.value(),
                'target_length': t.inp_pts_2d.value(),
                'snr_threshold': t.inp_snr_2d.value(),
                'outlier_contamination': t.inp_outliers_2d.value(),
                'notch_q': t.inp_notch_2d.value() if hasattr(t, 'inp_notch_2d') else notch_q_val,
                'tipo_filtro_ruido': tipo_filtro_ruido,
                'highpass_cutoff_hz': highpass_val,
                'lowpass_cutoff_hz': lowpass_val,
                'comp_x': t.cmb_pc_x_2d.currentText() if hasattr(t, 'cmb_pc_x_2d') else 'PC1',
                'comp_y': t.cmb_pc_y_2d.currentText() if hasattr(t, 'cmb_pc_y_2d') else 'PC2',
                'pesos_canales': [
                    t.inp_w0_2d.value() if hasattr(t, 'inp_w0_2d') else 1.0,
                    t.inp_w1_2d.value() if hasattr(t, 'inp_w1_2d') else 1.0,
                    t.inp_w2_2d.value() if hasattr(t, 'inp_w2_2d') else 1.0
                ]
            },
            'params_3d': {
                'alpha_ruido': t.inp_alpha_3d.value(),
                'gate_ratio_ruido': t.inp_gate.value() if hasattr(t, 'inp_gate') else (t.inp_gate_3d.value() if hasattr(t, 'inp_gate_3d') else 0.0),
                'smooth_ms': t.inp_smooth_3d.value(),
                'target_length': t.inp_pts_3d.value(),
                'snr_threshold': t.inp_snr_3d.value(),
                'outlier_contamination': t.inp_outliers_3d.value(),
                'notch_q': t.inp_notch_3d.value() if hasattr(t, 'inp_notch_3d') else notch_q_val,
                'tipo_filtro_ruido': tipo_filtro_ruido,
                'highpass_cutoff_hz': highpass_val,
                'lowpass_cutoff_hz': lowpass_val,
                'comp_x': t.cmb_pc_x_3d.currentText() if hasattr(t, 'cmb_pc_x_3d') else 'PC1',
                'comp_y': t.cmb_pc_y_3d.currentText() if hasattr(t, 'cmb_pc_y_3d') else 'PC2',
                'comp_z': t.cmb_pc_z_3d.currentText() if hasattr(t, 'cmb_pc_z_3d') else 'PC3',
                'pesos_canales': [
                    t.inp_w0_3d.value() if hasattr(t, 'inp_w0_3d') else 1.0,
                    t.inp_w1_3d.value() if hasattr(t, 'inp_w1_3d') else 1.0,
                    t.inp_w2_3d.value() if hasattr(t, 'inp_w2_3d') else 1.0
                ]
            },
            'params_umap': {},
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'umap_metric': 'euclidean',
            'algoritmo_clustering_pca': t.cmb_cluster.currentText(),
            'algoritmo_clustering_umap': 'K-Means',
            'aplicar_trevisan': t.chk_trevisan.isChecked(),
            'aplicar_correccion_intersesion': t.chk_correccion_intersesion.isChecked() if hasattr(t, 'chk_correccion_intersesion') else True,
            'ignorar_ventana_cero': t.chk_ignorar_cero.isChecked(),
            'pre_pct': t.inp_pre_pct.value(),
            'post_pct': t.inp_post_pct.value(),
            'modo_alineacion': t.cmb_align.currentText(),
            'estilo_visual': estilo_visual,
            'canales_features': canales
        }

    def get_umap_kwargs(self):
        t = self.tab_umap
        canales = []
        if t.chk_canal_0.isChecked(): canales.append("canal_0")
        if t.chk_canal_1.isChecked(): canales.append("canal_1")
        if t.chk_canal_2.isChecked(): canales.append("canal_2")
        
        lowpass_txt = t.cmb_lowpass.currentText() if hasattr(t, 'cmb_lowpass') else "300 Hz (Rangayyan)"
        if "300" in lowpass_txt:
            lowpass_val = 300.0
        elif "500" in lowpass_txt:
            lowpass_val = 500.0
        else:
            lowpass_val = 0.0
        
        noise_txt = t.cmb_noise_filter.currentText() if hasattr(t, 'cmb_noise_filter') else "Notch (IIR)"
        if "adaptativo" in noise_txt.lower() or "nlms" in noise_txt.lower():
            tipo_filtro_ruido = "adaptativo"
        elif "desactivado" in noise_txt.lower():
            tipo_filtro_ruido = "desactivado"
        else:
            tipo_filtro_ruido = "notch"

        highpass_val = t.inp_highpass.value() if hasattr(t, 'inp_highpass') else 20.0
        notch_q_val = t.inp_notch_q.value() if hasattr(t, 'inp_notch_q') else 2.0

        return {
            'proc_pca_2d': False,
            'proc_pca_3d': False,
            'proc_umap_2d': t.chk_umap_2d.isChecked(),
            'proc_umap_3d': t.chk_umap_3d.isChecked(),
            'ocultar_leyenda': t.chk_ocultar_leyenda.isChecked(),
            'tipo_filtro_ruido': tipo_filtro_ruido,
            'notch_q': notch_q_val,
            'highpass_cutoff_hz': highpass_val,
            'lowpass_cutoff_hz': lowpass_val,
            'params_2d': {},
            'params_3d': {},
            'params_umap': {
                'alpha_ruido': t.inp_alpha_u.value(),
                'gate_ratio_ruido': t.inp_gate.value() if hasattr(t, 'inp_gate') else (t.inp_gate_u.value() if hasattr(t, 'inp_gate_u') else 0.0),
                'smooth_ms': t.inp_smooth_u.value(),
                'target_length': t.inp_pts_u.value(),
                'snr_threshold': t.inp_snr_u.value(),
                'outlier_contamination': t.inp_outliers_u.value(),
                'notch_q': t.inp_notch_u.value() if hasattr(t, 'inp_notch_u') else notch_q_val,
                'tipo_filtro_ruido': tipo_filtro_ruido,
                'highpass_cutoff_hz': highpass_val,
                'lowpass_cutoff_hz': lowpass_val
            },
            'umap_n_neighbors': t.inp_n_neighbors.value(),
            'umap_min_dist': t.inp_min_dist.value(),
            'umap_metric': t.cmb_metric.currentText(),
            'algoritmo_clustering_pca': 'K-Means',
            'algoritmo_clustering_umap': t.cmb_cluster.currentText(),
            'aplicar_trevisan': t.chk_trevisan.isChecked(),
            'aplicar_correccion_intersesion': t.chk_correccion_intersesion.isChecked() if hasattr(t, 'chk_correccion_intersesion') else True,
            'ignorar_ventana_cero': t.chk_ignorar_cero.isChecked(),
            'pre_pct': t.inp_pre_pct.value(),
            'post_pct': t.inp_post_pct.value(),
            'modo_alineacion': t.cmb_align.currentText(),
            'estilo_visual': 'Elipses',
            'canales_features': canales
        }

    def get_umap_supervisado_kwargs(self):
        t = self.tab_umap_sup
        t_umap = self.tab_umap
        canales = []
        if t_umap.chk_canal_0.isChecked(): canales.append("canal_0")
        if t_umap.chk_canal_1.isChecked(): canales.append("canal_1")
        if t_umap.chk_canal_2.isChecked(): canales.append("canal_2")

        noise_txt = t_umap.cmb_noise_filter.currentText() if hasattr(t_umap, 'cmb_noise_filter') else "Notch (IIR)"
        if "adaptativo" in noise_txt.lower() or "nlms" in noise_txt.lower():
            tipo_filtro_ruido = "adaptativo"
        elif "desactivado" in noise_txt.lower():
            tipo_filtro_ruido = "desactivado"
        else:
            tipo_filtro_ruido = "notch"

        highpass_val = t_umap.inp_highpass.value() if hasattr(t_umap, 'inp_highpass') else 20.0
        lowpass_txt = t_umap.cmb_lowpass.currentText() if hasattr(t_umap, 'cmb_lowpass') else "300 Hz (Rangayyan)"
        if "300" in lowpass_txt:
            lowpass_val = 300.0
        elif "500" in lowpass_txt:
            lowpass_val = 500.0
        else:
            lowpass_val = 0.0

        return {
            'alpha_ruido': t.inp_alpha.value(),
            'smooth_ms': t.inp_smooth.value(),
            'target_length': t.inp_target_len.value(),
            'snr_threshold': t.inp_snr.value(),
            'outlier_contamination': t.inp_outliers.value(),
            'notch_q': t.inp_notch.value(),
            'tipo_filtro_ruido': tipo_filtro_ruido,
            'highpass_cutoff_hz': highpass_val,
            'lowpass_cutoff_hz': lowpass_val,
            'umap_n_neighbors': t.inp_umap_nn.value(),
            'umap_min_dist': t.inp_umap_md.value(),
            'umap_metric': t.cmb_metric.currentText(),
            'target_weight': t.inp_umap_tw.value(),
            'eliminar_outliers_train': t.chk_outliers_train.isChecked(),
            'aplicar_trevisan': t_umap.chk_trevisan.isChecked(),
            'ignorar_ventana_cero': t_umap.chk_ignorar_cero.isChecked(),
            'pre_pct': t_umap.inp_pre_pct.value(),
            'post_pct': t_umap.inp_post_pct.value(),
            'modo_alineacion': t_umap.cmb_align.currentText(),
            'canales_features': canales
        }

    def get_autoencoder_kwargs(self):
        t = self.tab_autoencoders
        return {
            'alpha_ruido': t.inp_alpha.value(),
            'smooth_ms': t.inp_smooth.value(),
            'target_length': t.inp_pts.value(),
            'snr_min': t.inp_snr.value(),
            'outliers_pct': t.inp_outliers.value(),
            'notch_q': t.inp_notch.value(),
            'use_manual_exclusions': t.chk_manual_excl.isChecked(),
            'epochs': t.inp_epochs.value(),
            'batch_size': t.inp_batch.value(),
            'latent_dim': t.inp_latent.value(),
            'kernel_size': t.inp_kernel.value(),
            'alpha_loss': t.inp_alpha_loss.value(),
            'force_epochs': t.chk_force_epochs.isChecked(),
            'train_sessions': t.get_train_sessions(),
            'test_sessions': t.get_test_sessions()
        }

class TrainTestSplitDialog(QDialog):
    def __init__(self, sesiones, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración de UMAP Supervisado")
        self.setMinimumSize(600, 450)
        self.setStyleSheet("""
            QDialog { background-color: #0c0c0c; color: #e0e0e0; font-family: 'Consolas', 'Courier New', monospace; }
            QLabel { color: #ffffff; font-weight: bold; }
            QLineEdit { background-color: #1e1e1e; color: #00ffcc; border: 1px solid #333; padding: 5px; }
            QListWidget { background-color: #1e1e1e; color: #00ffcc; border: 1px solid #333; }
            QPushButton { background-color: #333333; color: white; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #555555; }
            QPushButton#btnConfirm { background-color: #0066cc; }
            QPushButton#btnConfirm:hover { background-color: #0088ff; }
        """)

        layout = QVBoxLayout(self)

        # Nombre del Set
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("Nombre del Set de Mediciones:"))
        self.inp_nombre = QLineEdit()
        self.inp_nombre.setPlaceholderText("Ej: sujeto_lucas_prueba_1")
        h_name.addWidget(self.inp_nombre)
        layout.addLayout(h_name)

        # Listas de Train y Test
        h_lists = QHBoxLayout()
        
        v_train = QVBoxLayout()
        v_train.addWidget(QLabel("Entrenamiento (Train)"))
        self.lst_train = QListWidget()
        self.lst_train.setSelectionMode(QListWidget.ExtendedSelection)
        # By default, add all sessions to Train
        for s in sesiones:
            self.lst_train.addItem(s)
        v_train.addWidget(self.lst_train)
        h_lists.addLayout(v_train)

        # Botones de flechas
        v_arrows = QVBoxLayout()
        v_arrows.addStretch()
        self.btn_to_test = QPushButton(">>")
        self.btn_to_train = QPushButton("<<")
        self.btn_to_test.clicked.connect(self._move_to_test)
        self.btn_to_train.clicked.connect(self._move_to_train)
        v_arrows.addWidget(self.btn_to_test)
        v_arrows.addWidget(self.btn_to_train)
        v_arrows.addStretch()
        h_lists.addLayout(v_arrows)

        v_test = QVBoxLayout()
        v_test.addWidget(QLabel("Validación (Test)"))
        self.lst_test = QListWidget()
        self.lst_test.setSelectionMode(QListWidget.ExtendedSelection)
        v_test.addWidget(self.lst_test)
        h_lists.addLayout(v_test)

        layout.addLayout(h_lists)

        # Botones de Acción
        h_action = QHBoxLayout()
        h_action.addStretch()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_confirm = QPushButton("Confirmar")
        self.btn_confirm.setObjectName("btnConfirm")
        
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.clicked.connect(self.accept)
        
        h_action.addWidget(self.btn_cancel)
        h_action.addWidget(self.btn_confirm)
        layout.addLayout(h_action)

    def _move_to_test(self):
        items = self.lst_train.selectedItems()
        for item in items:
            self.lst_train.takeItem(self.lst_train.row(item))
            self.lst_test.addItem(item)

    def _move_to_train(self):
        items = self.lst_test.selectedItems()
        for item in items:
            self.lst_test.takeItem(self.lst_test.row(item))
            self.lst_train.addItem(item)
            
    def get_results(self):
        nombre = self.inp_nombre.text().strip().replace(" ", "_")
        train_sessions = [self.lst_train.item(i).text() for i in range(self.lst_train.count())]
        test_sessions = [self.lst_test.item(i).text() for i in range(self.lst_test.count())]
        return nombre, train_sessions, test_sessions


class ThresholdTrainingDialog(QDialog):
    def __init__(self, mediciones, mapped_names=None, available_channels=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuracion de Entrenamiento de Umbrales")
        self.setMinimumSize(620, 520)
        self.setStyleSheet("""
            QDialog { background-color: #0c0c0c; color: #e0e0e0; font-family: 'Consolas', 'Courier New', monospace; }
            QLabel { color: #ffffff; font-weight: bold; }
            QLineEdit { background-color: #1e1e1e; color: #00ffcc; border: 1px solid #333; padding: 5px; }
            QComboBox { background-color: #1e1e1e; color: #00ffcc; border: 1px solid #333; padding: 4px; }
            QCheckBox { color: #00ffcc; font-weight: bold; }
            QCheckBox::indicator:checked { background-color: #ff00ff; border: 1px solid #ff00ff; }
            QScrollArea { border: 1px solid #333; background-color: #111; }
            QPushButton { background-color: #333333; color: white; border-radius: 4px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #555555; }
            QPushButton#btnConfirm { background-color: #00887a; color: #ffffff; }
            QPushButton#btnConfirm:hover { background-color: #00b39f; }
        """)

        layout = QVBoxLayout(self)

        # 1. Identificador del Set de Entrenamiento
        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("Nombre del Set (Opcional):"))
        self.inp_nombre = QLineEdit()
        self.inp_nombre.setPlaceholderText("Ej: entrenamiento_sujeto1")
        h_name.addWidget(self.inp_nombre)
        layout.addLayout(h_name)

        # 2. Seleccion de Canales Musculares
        g_canales = QGroupBox("1. Canales Musculares a Entrenar (Excluir canal microfono)")
        g_canales.setStyleSheet("QGroupBox { border: 1px solid #00ffcc; border-radius: 4px; margin-top: 8px; padding-top: 10px; font-weight: bold; color: #00ffcc; }")
        lyt_chans = QHBoxLayout(g_canales)
        
        if not available_channels:
            available_channels = ["canal_0", "canal_1", "canal_2"]
        if mapped_names is None:
            mapped_names = {}
            
        self.channel_checkboxes = {}
        for ch in available_channels:
            if ch.lower() == "canal_3":
                continue
            musculo = mapped_names.get(ch, ch)
            chk = QCheckBox(f"{ch} ({musculo})")
            chk.setChecked(True)
            lyt_chans.addWidget(chk)
            self.channel_checkboxes[ch] = chk
            
        if not self.channel_checkboxes:
            for ch in ["canal_0", "canal_1", "canal_2"]:
                chk = QCheckBox(ch)
                chk.setChecked(True)
                lyt_chans.addWidget(chk)
                self.channel_checkboxes[ch] = chk

        layout.addWidget(g_canales)

        # 3. Asignacion de Vocales por Medicion
        layout.addWidget(QLabel("2. Asignacion de Vocal por Medicion:"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        self.combos_vocales = {}
        for med in mediciones:
            row = QHBoxLayout()
            med_name = os.path.basename(med)
            lbl = QLabel(med_name)
            lbl.setStyleSheet("color: #aaa; font-weight: normal;")
            cmb = QComboBox()
            cmb.addItems(["A", "E", "I", "O", "U", "Ignorar"])
            
            detected_vocal = None
            tokens = med_name.replace('-', '_').split('_')
            for t in tokens:
                t_up = t.upper()
                if t_up in ["A", "E", "I", "O", "U"]:
                    detected_vocal = t_up
                    break
                    
            if detected_vocal:
                cmb.setCurrentText(detected_vocal)
            else:
                cmb.setCurrentIndex(0)
                
            row.addWidget(lbl, stretch=2)
            row.addWidget(cmb, stretch=1)
            scroll_layout.addLayout(row)
            self.combos_vocales[med] = cmb

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Botones de Accion
        h_action = QHBoxLayout()
        h_action.addStretch()
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_confirm = QPushButton("Confirmar e Iniciar Entrenamiento")
        self.btn_confirm.setObjectName("btnConfirm")

        self.btn_cancel.clicked.connect(self.reject)
        self.btn_confirm.clicked.connect(self._validate_and_accept)

        h_action.addWidget(self.btn_cancel)
        h_action.addWidget(self.btn_confirm)
        layout.addLayout(h_action)

    def _validate_and_accept(self):
        from PySide6.QtWidgets import QMessageBox
        chans = [ch for ch, chk in self.channel_checkboxes.items() if chk.isChecked()]
        if not chans:
            QMessageBox.warning(self, "Atencion", "Debe seleccionar al menos un canal muscular para el entrenamiento.")
            return
            
        asignaciones = {p: c.currentText() for p, c in self.combos_vocales.items() if c.currentText() != "Ignorar"}
        if not asignaciones:
            QMessageBox.warning(self, "Atencion", "Debe asignar al menos una medicion con una vocal valida (distinta de 'Ignorar').")
            return
            
        self.accept()

    def get_results(self):
        nombre = self.inp_nombre.text().strip().replace(" ", "_")
        selected_channels = [ch for ch, chk in self.channel_checkboxes.items() if chk.isChecked()]
        asignaciones_vocales = {p: c.currentText() for p, c in self.combos_vocales.items() if c.currentText() != "Ignorar"}
        return nombre, selected_channels, asignaciones_vocales

