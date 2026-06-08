# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Widget de interfaz para visualizar datos crudos desde archivos CSV.
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QGroupBox, QFileDialog, QScrollArea, QSplitter, 
    QSpinBox, QDoubleSpinBox, QSlider, QComboBox
)
from PySide6.QtCore import Qt, QThread, Signal
import pyqtgraph as pg
import pyqtgraph.exporters

# --- OPTIMIZACIÓN EXTREMA DE FLUIDEZ ---
# Antialiasing se ve "suave" pero devora la CPU al hacer zoom con cientos de miles de puntos.
pg.setConfigOptions(antialias=False) 
pg.setConfigOption('background', '#050505')
pg.setConfigOption('foreground', '#d3d3d3')

MAX_POINTS_TO_PLOT = 25_000 # Límite optimizado para LTTB (muy fluido y preserva picos)

def downsample_lttb_fast(x, y, threshold):
    if len(x) <= threshold: return x, y
    n_blocks = threshold // 2
    block_size = len(x) // n_blocks
    trunc_len = n_blocks * block_size
    x_trunc = x[:trunc_len]
    y_trunc = y[:trunc_len]
    x_blocks = x_trunc.reshape(n_blocks, block_size)
    y_blocks = y_trunc.reshape(n_blocks, block_size)
    min_idxs = np.argmin(y_blocks, axis=1)
    max_idxs = np.argmax(y_blocks, axis=1)
    offsets = np.arange(n_blocks) * block_size
    abs_min_idxs = min_idxs + offsets
    abs_max_idxs = max_idxs + offsets
    all_idxs = np.concatenate((abs_min_idxs, abs_max_idxs))
    all_idxs.sort()
    if all_idxs[0] != 0: all_idxs = np.insert(all_idxs, 0, 0)
    if all_idxs[-1] != len(x) - 1: all_idxs = np.append(all_idxs, len(x) - 1)
    return x[all_idxs], y[all_idxs]

class DataLoaderThread(QThread):
    finished_loading = Signal(object, list, str, float) # df, cols_canales, time_col, global_max_y
    error_loading = Signal(str)

    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            df = pd.read_csv(self.filepath)
            df.columns = df.columns.str.strip()
            time_col = None
            for c in ['Time', 'Tiempo', 'Time(s)']:
                if c in df.columns:
                    time_col = c; break
            if not time_col and len(df.columns) > 0: time_col = df.columns[0]
            
            canales = [c for c in df.columns if c != time_col]
            
            # Calibración a microvoltios
            measurement_dir = os.path.dirname(self.filepath)
            meta_path = os.path.join(measurement_dir, 'metadata.json')
            resistencia_ohm = 100.0
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        if 'resistencia_ohm' in meta:
                            resistencia_ohm = float(meta['resistencia_ohm'])
                except: pass
            
            r_fija = 49400.0
            ganancia = 1.0 + (r_fija / resistencia_ohm)
            
            # Convertir todos los canales (asumiendo que están en voltios si no están calibrados, o usar ganancia)
            # En la versión de tkinter se divide por ganancia y * 1e6
            for c in canales:
                df[c] = (df[c] / ganancia) * 1e6

            global_max_y = 1.0
            if len(canales) > 0:
                global_max_y = np.max(np.abs(df[canales].values))
                if np.isnan(global_max_y) or global_max_y == 0: global_max_y = 1.0

            self.finished_loading.emit(df, canales, time_col, global_max_y)
        except Exception as e:
            self.error_loading.emit(str(e))

class CsvViewerWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df = None
        self.canales_originales = {}
        self.time_data = None
        self.global_max_y = 100.0
        self._updating_sliders = False
        
        self.channel_colors = ['#FF4136', '#0074D9', '#2ECC40', '#FF851B', '#B10DC9', '#39CCCC']
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        self.lbl_status = QLabel("Listo. Seleccione una medición en el Gestor de Sesiones (Izquierda).")
        self.lbl_status.setStyleSheet("color: #888; font-family: monospace; padding: 5px;")
        self.layout.addWidget(self.lbl_status)
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Gráfico
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setClipToView(True) # OPTIMIZACIÓN: Solo dibuja los puntos en pantalla
        self.plot_widget.setLabel('bottom', 'Tiempo', units='s')
        self.plot_widget.setLabel('left', 'Amplitud', units='µV')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.addLegend()
        self.plot_widget.sigXRangeChanged.connect(self._on_plot_xrange_changed)
        splitter.addWidget(self.plot_widget)
        
        # Panel Derecho
        self.ctrl_panel = QWidget()
        ctrl_layout = QVBoxLayout(self.ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        scroll_content = QWidget()
        self.slayout = QVBoxLayout(scroll_content)
        self.slayout.setSpacing(15)
        
        # Canales
        self.grp_channels = QGroupBox("Canales")
        self.grp_channels.setStyleSheet("font-weight: bold; color: #aaa;")
        self.lyt_channels = QVBoxLayout(self.grp_channels)
        self.slayout.addWidget(self.grp_channels)
        
        # Navegación (Sliders como el original)
        grp_nav = QGroupBox("Navegación")
        grp_nav.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_nav = QVBoxLayout(grp_nav)
        
        lyt_nav.addWidget(QLabel("Posición (s):"))
        self.slider_time = QSlider(Qt.Horizontal)
        self.slider_time.setRange(0, 100)
        self.slider_time.valueChanged.connect(self._on_time_slider_changed)
        lyt_nav.addWidget(self.slider_time)
        
        lyt_nav.addWidget(QLabel("Posición Y:"))
        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(0, 100)
        self.slider_y.setValue(50)
        self.slider_y.valueChanged.connect(self._on_y_slider_changed)
        lyt_nav.addWidget(self.slider_y)
        self.slayout.addWidget(grp_nav)
        
        # Zoom
        grp_zoom = QGroupBox("Zoom")
        grp_zoom.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_zoom = QVBoxLayout(grp_zoom)
        
        lyt_zoom.addWidget(QLabel("Duración en Pantalla (s):"))
        self.spin_zoom_x = QDoubleSpinBox()
        self.spin_zoom_x.setRange(0.1, 1000)
        self.spin_zoom_x.setValue(5.0)
        self.spin_zoom_x.valueChanged.connect(self._apply_view_ranges)
        lyt_zoom.addWidget(self.spin_zoom_x)
        
        lyt_zoom.addWidget(QLabel("Amplitud (%):"))
        self.slider_zoom_y = QSlider(Qt.Horizontal)
        self.slider_zoom_y.setRange(1, 100)
        self.slider_zoom_y.setValue(100)
        self.slider_zoom_y.valueChanged.connect(self._apply_view_ranges)
        lyt_zoom.addWidget(self.slider_zoom_y)
        self.btn_autoscale = QPushButton("Ajuste Automático")
        self.btn_autoscale.clicked.connect(self._autoscale)
        lyt_zoom.addWidget(self.btn_autoscale)
        self.slayout.addWidget(grp_zoom)
        
        # Filtros
        self.grp_filters = QGroupBox("Filtros")
        self.grp_filters.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_filt = QVBoxLayout(self.grp_filters)
        self.chk_notch = QCheckBox("Notch 50Hz")
        self.chk_notch.stateChanged.connect(self.update_plot)
        lyt_filt.addWidget(self.chk_notch)
        
        self.chk_bandpass = QCheckBox("Pasa-Banda")
        self.chk_bandpass.setChecked(True)
        self.chk_bandpass.stateChanged.connect(self.update_plot)
        lyt_filt.addWidget(self.chk_bandpass)
        
        row_hp = QHBoxLayout()
        row_hp.addWidget(QLabel("Highpass (Hz):"))
        self.spin_hp = QDoubleSpinBox()
        self.spin_hp.setRange(0, 1000)
        self.spin_hp.setValue(20) # Por defecto 20Hz
        row_hp.addWidget(self.spin_hp)
        lyt_filt.addLayout(row_hp)
        
        row_lp = QHBoxLayout()
        row_lp.addWidget(QLabel("Lowpass (Hz):"))
        self.spin_lp = QDoubleSpinBox()
        self.spin_lp.setRange(0, 10000)
        self.spin_lp.setValue(500) # Por defecto 500Hz
        row_lp.addWidget(self.spin_lp)
        lyt_filt.addLayout(row_lp)
        
        self.spin_lp.editingFinished.connect(self.update_plot)
        self.spin_hp.editingFinished.connect(self.update_plot)
        self.slayout.addWidget(self.grp_filters)
        
        # Envolvente
        grp_env = QGroupBox("Envolvente")
        grp_env.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_env = QVBoxLayout(grp_env)
        self.cmb_env = QComboBox()
        self.cmb_env.addItems(["ninguna", "media_movil", "rms"])
        self.cmb_env.currentTextChanged.connect(self.update_plot)
        lyt_env.addWidget(self.cmb_env)
        row_env = QHBoxLayout()
        row_env.addWidget(QLabel("Ventana (ms):"))
        self.spin_env = QSpinBox()
        self.spin_env.setRange(1, 5000)
        self.spin_env.setValue(50)
        self.spin_env.editingFinished.connect(self.update_plot)
        row_env.addWidget(self.spin_env)
        lyt_env.addLayout(row_env)
        self.slayout.addWidget(grp_env)
        
        # Extras
        grp_opts = QGroupBox("Extras")
        grp_opts.setStyleSheet("font-weight: bold; color: #aaa;")
        lyt_opts = QVBoxLayout(grp_opts)
        self.btn_export = QPushButton("📸 Exportar PNG")
        self.btn_export.clicked.connect(self.export_png)
        lyt_opts.addWidget(self.btn_export)
        self.slayout.addWidget(grp_opts)
        
        self.slayout.addStretch()
        scroll.setWidget(scroll_content)
        ctrl_layout.addWidget(scroll)
        
        splitter.addWidget(self.ctrl_panel)
        splitter.setSizes([800, 250])
        self.layout.addWidget(splitter, stretch=1)
        self.channel_checkboxes = {}

    def load_csv(self, filepath):
        if not os.path.exists(filepath):
            self.lbl_status.setText(f"Error: No se encuentra el archivo {filepath}")
            return
        self.lbl_status.setText(f"Cargando archivo: {os.path.basename(filepath)} ...")
        self.loader = DataLoaderThread(filepath)
        self.loader.finished_loading.connect(self._on_csv_loaded)
        self.loader.error_loading.connect(lambda err: self.lbl_status.setText(f"Error: {err}"))
        self.loader.start()

    def _on_csv_loaded(self, df, canales, time_col, max_y):
        self.df = df
        self.time_data = self.df[time_col].values
        self.global_max_y = max_y
        
        fs = 1.0
        if len(self.time_data) > 1:
            fs = 1.0 / (self.time_data[1] - self.time_data[0])
            
        self.lbl_status.setText(f"Archivo cargado. {len(self.df)} ptos. Fs: {fs:.1f}Hz. Tiempo: {self.time_data[-1]:.2f}s")
        
        # Configurar Sliders de Navegación
        t_max = self.time_data[-1]
        self._updating_sliders = True
        self.slider_time.setRange(0, int(t_max * 100))
        self.slider_time.setValue(0)
        self.spin_zoom_x.setValue(min(5.0, t_max))
        self.slider_y.setValue(50)
        self.slider_zoom_y.setValue(100)
        self._updating_sliders = False
        
        # Limpiar canales previos
        for chk in self.channel_checkboxes.values():
            self.lyt_channels.removeWidget(chk)
            chk.deleteLater()
        self.channel_checkboxes.clear()
        
        # Guardar data original y crear checkboxes
        self.canales_originales.clear()
        for idx, canal in enumerate(canales):
            self.canales_originales[canal] = self.df[canal].values
            chk = QCheckBox(canal)
            chk.setChecked(True)
            chk.setStyleSheet(f"color: {self.channel_colors[idx % len(self.channel_colors)]}; font-weight: bold;")
            chk.stateChanged.connect(self.update_plot)
            self.lyt_channels.addWidget(chk)
            self.channel_checkboxes[canal] = chk
            
        self.update_plot()
        self._apply_view_ranges()

    def update_plot(self):
        if self.df is None: return
        self.plot_widget.clear()
        
        # Limpiar la leyenda para evitar que se acumulen nombres viejos (canales "fantasma")
        if getattr(self.plot_widget.plotItem, 'legend', None) is not None:
            self.plot_widget.plotItem.legend.clear()
        
        fs = 1.0
        if len(self.time_data) > 1:
            fs = 1.0 / (self.time_data[1] - self.time_data[0])
            
        notch = self.chk_notch.isChecked()
        lp = self.spin_lp.value()
        hp = self.spin_hp.value()
        tipo_env = self.cmb_env.currentText()
        env_window = int((self.spin_env.value() / 1000.0) * fs)
        if env_window < 1: env_window = 1
        
        for idx_canal, (canal, chk) in enumerate(self.channel_checkboxes.items()):
            if chk.isChecked():
                y_data = self.canales_originales[canal].copy()
                
                # Filtros
                if notch and fs > 110:
                    b, a = signal.iirnotch(50.0, 30.0, fs)
                    y_data = signal.filtfilt(b, a, y_data)
                
                if self.chk_bandpass.isChecked():
                    if hp > 0 and hp < fs/2:
                        b, a = signal.butter(4, hp / (0.5 * fs), btype='high')
                        y_data = signal.filtfilt(b, a, y_data)
                    if lp > 0 and lp < fs/2:
                        b, a = signal.butter(4, lp / (0.5 * fs), btype='low')
                        y_data = signal.filtfilt(b, a, y_data)
                
                # Envolvente
                if tipo_env != "ninguna":
                    if tipo_env == "rms":
                        y_data_sq = y_data**2
                        kernel = np.ones(env_window) / env_window
                        y_data = np.sqrt(np.convolve(y_data_sq, kernel, mode='same'))
                    else: # media_movil
                        y_data = np.abs(y_data)
                        kernel = np.ones(env_window) / env_window
                        y_data = np.convolve(y_data, kernel, mode='same')
                    
                # Downsampling
                x_plot, y_plot = downsample_lttb_fast(self.time_data, y_data, MAX_POINTS_TO_PLOT)
                
                color = self.channel_colors[idx_canal % len(self.channel_colors)]
                self.plot_widget.plot(x_plot, y_plot, name=canal, pen=pg.mkPen(color, width=1.5))

    def _autoscale(self):
        if self.df is None: return
        self._updating_sliders = True
        self.slider_zoom_y.setValue(100)
        self.slider_y.setValue(50)
        self._updating_sliders = False
        self.plot_widget.autoRange()

    def _apply_view_ranges(self):
        if self.df is None or self._updating_sliders: return
        
        t_start = self.slider_time.value() / 100.0
        duracion = self.spin_zoom_x.value()
        self.plot_widget.setXRange(t_start, t_start + duracion, padding=0)
        
        amp_pct = self.slider_zoom_y.value() / 100.0
        y_span = (self.global_max_y * 2.5) * amp_pct
        
        y_center_pct = self.slider_y.value() / 100.0 # 0 a 1
        y_min_limit = -self.global_max_y * 1.5
        y_max_limit = self.global_max_y * 1.5
        y_center = y_min_limit + (y_max_limit - y_min_limit) * y_center_pct
        
        self.plot_widget.setYRange(y_center - y_span/2, y_center + y_span/2, padding=0)

    def _on_time_slider_changed(self):
        self._apply_view_ranges()

    def _on_y_slider_changed(self):
        self._apply_view_ranges()

    def _on_plot_xrange_changed(self, _, range_tuple):
        """Sincronizar slider cuando el usuario arrastra el gráfico con el mouse"""
        if self._updating_sliders or self.df is None: return
        self._updating_sliders = True
        self.slider_time.setValue(int(range_tuple[0] * 100))
        duracion = range_tuple[1] - range_tuple[0]
        self.spin_zoom_x.setValue(max(0.1, duracion))
        self._updating_sliders = False

    def export_png(self):
        filepath, _ = QFileDialog.getSaveFileName(self, "Exportar Gráfico", "export.png", "PNG (*.png)")
        if filepath:
            exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
            exporter.export(filepath)
            self.lbl_status.setText(f"Exportado a: {filepath}")
