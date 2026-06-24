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
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QTabWidget,
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

        self.chk_cyberpunk = QCheckBox("Tema Cyberpunk (Gráficos oscuros y neón)")
        self.chk_cyberpunk.setChecked(False) # Por defecto apagado para usar estética normal
        l_ind.addWidget(self.chk_cyberpunk)

        self.chk_espectrograma = QCheckBox("Generar Espectrograma Señal Completa (Estilo Praat)")
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

        # Botones Lanzar
        btn_layout = QHBoxLayout()
        
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
        btn_layout.addWidget(self.btn_run_comparativo)

        self.btn_run_sesion = QPushButton("📈 LANZAR EVOLUCIÓN DE SESIÓN")
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
        self.method_tabs.addTab(tab_stat, "📊 Umbral Estadístico (Ruido)")
        
        # 1B. Pestaña Manual
        tab_man = QWidget()
        l_man = QVBoxLayout(tab_man)
        l_man.addWidget(QLabel("Umbrales absolutos (0.01 a 1.0) sobre la máxima amplitud global del pulso."))
        
        from utils.config_manager import ConfigManager
        cm = ConfigManager()
        c_config = cm.get("canales") or {}
        
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
        self.method_tabs.addTab(tab_man, "✍️ Umbral Manual por Canal")
        
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
        self.btn_run_motor = QPushButton("🧠 LANZAR COORDENADAS DISCRETAS")
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
        self.btn_run_training = QPushButton("🏋️ ENTRENAR UMBRALES (TRAIN)")
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

class PcaMotorTab(QWidget):
    """Pestaña para Análisis PCA y Reducción de Dimensionalidad"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 1. Configuración del PCA
        g_pca = QGroupBox("1. Hiperparámetros y Pre-procesamiento PCA")
        l_pca = QFormLayout()
        
        self.chk_supervised = QCheckBox("PCA Supervisado (Agrupar/colorear por vocal)")
        self.chk_supervised.setChecked(True)
        l_pca.addRow(self.chk_supervised)
        
        # Integración con UMAP
        self.chk_use_umap = QCheckBox("Utilizar resultados limpios en UMAP")
        self.chk_use_umap.setChecked(True)
        self.chk_use_umap.toggled.connect(self.toggle_umap_options)
        l_pca.addRow(self.chk_use_umap)
        
        self.inp_n_components = QSpinBox()
        self.inp_n_components.setRange(2, 300)
        self.inp_n_components.setValue(15)
        self.inp_n_components.setToolTip("Número de componentes a retener para UMAP (ej: 15)")
        l_pca.addRow("N° Componentes Intermedias:", self.inp_n_components)
        
        self.chk_kmeans = QCheckBox("Ejecutar Clustering K-Means (Evaluación y Matriz de Confusión)")
        self.chk_kmeans.setChecked(False)
        l_pca.addRow(self.chk_kmeans)
        
        g_pca.setLayout(l_pca)
        self.layout.addWidget(g_pca)
        
        # 2. Opciones de Filtro (Igual que Training/UMAP)
        g_filtro = QGroupBox("2. Limpieza de Datos (SNR)")
        l_filtro = QFormLayout()
        
        self.chk_snr = QCheckBox("Descartar pulsos con SNR menor a:")
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
        
        # Botón Lanzar
        btn_layout = QHBoxLayout()
        self.btn_run_pca = QPushButton("🧩 LANZAR ANÁLISIS PCA")
        self.btn_run_pca.setFixedHeight(50)
        self.btn_run_pca.setCursor(Qt.PointingHandCursor)
        self.btn_run_pca.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #ffaa00; border: 2px solid #ffaa00; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ffaa00; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        btn_layout.addWidget(self.btn_run_pca)
        
        self.layout.addLayout(btn_layout)

    def toggle_umap_options(self, checked):
        self.inp_n_components.setEnabled(checked)

class UmapMotorTab(QWidget):
    """Pestaña para Análisis de Clustering con UMAP/SUMAP"""
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        
        # 1. Configuración de Vectorización
        g_vector = QGroupBox("1. Modo de Vectorización")
        l_vector = QFormLayout()
        
        self.cmb_vector_mode = QComboBox()
        self.cmb_vector_mode.addItems(["Completa", "Picos"])
        l_vector.addRow("Características (Features):", self.cmb_vector_mode)
        
        self.chk_import_pca = QCheckBox("Importar desde resultado PCA")
        self.chk_import_pca.setChecked(False)
        
        self.cmb_pca_results = QComboBox()
        self.cmb_pca_results.setEnabled(False)
        self.chk_import_pca.toggled.connect(self.cmb_pca_results.setEnabled)
        
        l_vector.addRow(self.chk_import_pca)
        l_vector.addRow("Resultados PCA:", self.cmb_pca_results)
        
        g_vector.setLayout(l_vector)
        self.layout.addWidget(g_vector)
        
        # 2. Configuración del Modelo UMAP
        g_umap = QGroupBox("2. Hiperparámetros UMAP")
        l_umap = QFormLayout()
        
        self.chk_supervised = QCheckBox("Forzar Separación por Vocales (SUMAP Supervisado)")
        self.chk_supervised.setChecked(False)
        l_umap.addRow(self.chk_supervised)
        
        self.chk_kmeans = QCheckBox("Ejecutar Clustering K-Means (Evaluación y Matriz de Confusión)")
        self.chk_kmeans.setChecked(False)
        l_umap.addRow(self.chk_kmeans)
        
        self.inp_n_neighbors = QSpinBox()
        self.inp_n_neighbors.setRange(2, 500)
        self.inp_n_neighbors.setValue(15)
        l_umap.addRow("Vecinos (n_neighbors):", self.inp_n_neighbors)
        
        self.inp_min_dist = QDoubleSpinBox()
        self.inp_min_dist.setRange(0.0, 1.0)
        self.inp_min_dist.setSingleStep(0.1)
        self.inp_min_dist.setValue(0.1)
        l_umap.addRow("Distancia Mínima (min_dist):", self.inp_min_dist)
        
        g_umap.setLayout(l_umap)
        self.layout.addWidget(g_umap)
        
        # 3. Opciones de Filtro (Igual que Training)
        g_filtro = QGroupBox("3. Limpieza de Datos (SNR)")
        l_filtro = QFormLayout()
        
        self.chk_snr = QCheckBox("Descartar pulsos con SNR menor a:")
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
        
        # Botón Lanzar
        btn_layout = QHBoxLayout()
        self.btn_run_umap = QPushButton("🌌 LANZAR CLUSTERING UMAP")
        self.btn_run_umap.setFixedHeight(50)
        self.btn_run_umap.setCursor(Qt.PointingHandCursor)
        self.btn_run_umap.setStyleSheet("""
            QPushButton {
                font-weight: bold; font-size: 14px;
                background-color: transparent; color: #ff00ff; border: 2px solid #ff00ff; border-radius: 5px;
            }
            QPushButton:hover { background-color: #ff00ff; color: #000; }
            QPushButton:disabled { border: 2px solid #555; color: #555; }
        """)
        btn_layout.addWidget(self.btn_run_umap)
        
        self.layout.addLayout(btn_layout)

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

        # Pestaña Motor Discreto
        self.tab_motor = DiscreteMotorTab()
        self.tabs.addTab(self.tab_motor, "🧠 Coordenadas Discretas")

        # Pestaña Entrenamiento de Umbrales
        self.tab_training = TrainingMotorTab()
        self.tabs.addTab(self.tab_training, "🎯 Entrenamiento de Umbrales")

        # Pestaña PCA
        self.tab_pca = PcaMotorTab()
        self.tabs.addTab(self.tab_pca, "🧩 Análisis PCA")

        # Pestaña UMAP
        self.tab_umap = UmapMotorTab()
        self.tabs.addTab(self.tab_umap, "🌌 Clustering UMAP")

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
            'excluded_windows_list': excluded,
            'tema_cyberpunk': self.tab_procesamiento.chk_cyberpunk.isChecked()
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
