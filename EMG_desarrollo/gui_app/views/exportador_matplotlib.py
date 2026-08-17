import sys
import os
import pandas as pd
import numpy as np
from scipy import signal
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QCheckBox, QGroupBox, QFileDialog, 
    QSpinBox, QDoubleSpinBox, QComboBox, QRadioButton, QButtonGroup,
    QScrollArea, QSplitter, QMessageBox, QSlider, QLineEdit
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.config_manager import ConfigManager
config_mgr = ConfigManager()

class ExportadorMatplotlib(QMainWindow):
    def __init__(self, filepath=None):
        super().__init__()
        self.filepath = filepath
        self.df = None
        
        # Banderas para que sea una ventana flotante (ideal para Sway / Tiling WMs)
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("Exportador Gráfico Avanzado (Paper)")
        self.resize(1200, 800)
        
        self.canales_conf = config_mgr.get("canales") or {}
        
        self._init_ui()
        
        if self.filepath and os.path.exists(self.filepath):
            self.load_data()
        else:
            self._select_file()
            
    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        
        # Panel izquierdo (Controles)
        ctrl_panel = QScrollArea()
        ctrl_panel.setFixedWidth(350)
        ctrl_panel.setWidgetResizable(True)
        ctrl_content = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_content)
        
        # --- SELECCION DE ARCHIVO ---
        self.lbl_file = QLabel("Archivo: Ninguno")
        self.lbl_file.setWordWrap(True)
        btn_open = QPushButton("Abrir Otro CSV")
        btn_open.clicked.connect(self._select_file)
        ctrl_layout.addWidget(self.lbl_file)
        ctrl_layout.addWidget(btn_open)
        
        # --- PROCESAMIENTO ---
        grp_proc = QGroupBox("Procesamiento de Señal")
        lyt_proc = QVBoxLayout(grp_proc)
        
        self.chk_notch = QCheckBox("Filtro Notch (50Hz)")
        self.chk_notch.stateChanged.connect(self.update_plot)
        lyt_proc.addWidget(self.chk_notch)
        
        self.chk_bp = QCheckBox("Filtro Pasa-bandas")
        self.chk_bp.setChecked(True)
        self.chk_bp.stateChanged.connect(self.update_plot)
        lyt_proc.addWidget(self.chk_bp)
        
        bp_h = QHBoxLayout()
        self.spin_hp = QDoubleSpinBox()
        self.spin_hp.setRange(0.1, 1000)
        self.spin_hp.setValue(20.0)
        self.spin_hp.editingFinished.connect(self.update_plot)
        self.spin_lp = QDoubleSpinBox()
        self.spin_lp.setRange(0.1, 2000)
        self.spin_lp.setValue(500.0)
        self.spin_lp.editingFinished.connect(self.update_plot)
        bp_h.addWidget(QLabel("HP:"))
        bp_h.addWidget(self.spin_hp)
        bp_h.addWidget(QLabel("LP:"))
        bp_h.addWidget(self.spin_lp)
        lyt_proc.addLayout(bp_h)
        
        lyt_proc.addWidget(QLabel("Envolvente:"))
        self.cmb_env = QComboBox()
        self.cmb_env.addItems(["ninguna", "rms", "media_movil"])
        self.cmb_env.currentTextChanged.connect(self.update_plot)
        lyt_proc.addWidget(self.cmb_env)
        
        env_h = QHBoxLayout()
        env_h.addWidget(QLabel("Ventana (ms):"))
        self.spin_env = QSpinBox()
        self.spin_env.setRange(1, 2000)
        self.spin_env.setValue(50)
        self.spin_env.editingFinished.connect(self.update_plot)
        env_h.addWidget(self.spin_env)
        lyt_proc.addLayout(env_h)
        
        ctrl_layout.addWidget(grp_proc)
        
        # --- ESTETICA ---
        grp_est = QGroupBox("Estética y Visualización")
        lyt_est = QVBoxLayout(grp_est)
        
        self.rad_normal = QRadioButton("Normal (Fondo Blanco)")
        self.rad_cyber = QRadioButton("Cyberpunk (Fondo Oscuro)")
        self.rad_normal.setChecked(True)
        self.bg_group = QButtonGroup()
        self.bg_group.addButton(self.rad_normal)
        self.bg_group.addButton(self.rad_cyber)
        self.rad_normal.toggled.connect(self.update_plot)
        
        lyt_est.addWidget(self.rad_normal)
        lyt_est.addWidget(self.rad_cyber)
        
        self.chk_grid = QCheckBox("Mostrar Grilla")
        self.chk_grid.setChecked(True)
        self.chk_grid.stateChanged.connect(self.update_plot)
        lyt_est.addWidget(self.chk_grid)
        
        lyt_est.addWidget(QLabel("Opacidad Grilla:"))
        self.slider_grid = QSlider(Qt.Horizontal)
        self.slider_grid.setRange(10, 100)
        self.slider_grid.setValue(50)
        self.slider_grid.valueChanged.connect(self.update_plot)
        lyt_est.addWidget(self.slider_grid)
        
        fh_line = QHBoxLayout()
        fh_line.addWidget(QLabel("Grosor de Línea:"))
        self.spin_lw = QDoubleSpinBox()
        self.spin_lw.setRange(0.1, 10.0)
        self.spin_lw.setValue(1.5)
        self.spin_lw.setSingleStep(0.1)
        self.spin_lw.editingFinished.connect(self.update_plot)
        fh_line.addWidget(self.spin_lw)
        lyt_est.addLayout(fh_line)
        
        ctrl_layout.addWidget(grp_est)
        
        # --- CANALES ---
        self.grp_chan = QGroupBox("Canales a Mostrar")
        self.lyt_chan = QVBoxLayout(self.grp_chan)
        self.chk_canales = []
        # Se llenará en load_data
        ctrl_layout.addWidget(self.grp_chan)
        
        # --- MODO DE APILAMIENTO ---
        grp_stack = QGroupBox("Modo de Gráfico")
        lyt_stack = QVBoxLayout(grp_stack)
        
        self.rad_sup = QRadioButton("Superpuestos (0 Central)")
        self.rad_sep = QRadioButton("Separados (Stacked)")
        self.rad_sep.setChecked(True)
        self.stack_group = QButtonGroup()
        self.stack_group.addButton(self.rad_sup)
        self.stack_group.addButton(self.rad_sep)
        self.rad_sup.toggled.connect(self.update_plot)
        
        lyt_stack.addWidget(self.rad_sup)
        lyt_stack.addWidget(self.rad_sep)
        
        self.chk_auto_sep = QCheckBox("Separación Y Automática")
        self.chk_auto_sep.setChecked(True)
        self.chk_auto_sep.stateChanged.connect(self._toggle_auto_sep)
        lyt_stack.addWidget(self.chk_auto_sep)
        
        lyt_stack.addWidget(QLabel("Distancia Y (µV):"))
        self.spin_sep_y = QSpinBox()
        self.spin_sep_y.setRange(0, 50000)
        self.spin_sep_y.setValue(100)
        self.spin_sep_y.setSingleStep(50)
        self.spin_sep_y.setEnabled(False)
        self.spin_sep_y.editingFinished.connect(self.update_plot)
        lyt_stack.addWidget(self.spin_sep_y)
        
        lyt_stack.addWidget(QLabel("Valor Scale Bar (µV):"))
        self.spin_scale = QSpinBox()
        self.spin_scale.setRange(10, 5000)
        self.spin_scale.setValue(100)
        self.spin_scale.setSingleStep(50)
        self.spin_scale.editingFinished.connect(self.update_plot)
        lyt_stack.addWidget(self.spin_scale)
        
        ctrl_layout.addWidget(grp_stack)
        
        # --- TIPOGRAFIA ---
        grp_font = QGroupBox("Tipografía (Exportación)")
        lyt_font = QVBoxLayout(grp_font)
        
        fh_axes = QHBoxLayout()
        fh_axes.addWidget(QLabel("Títulos (X/Y):"))
        self.spin_f_title = QSpinBox()
        self.spin_f_title.setValue(14)
        self.spin_f_title.editingFinished.connect(self.update_plot)
        fh_axes.addWidget(self.spin_f_title)
        lyt_font.addLayout(fh_axes)
        
        fh_ticks = QHBoxLayout()
        fh_ticks.addWidget(QLabel("Números (Ticks):"))
        self.spin_f_tick = QSpinBox()
        self.spin_f_tick.setValue(12)
        self.spin_f_tick.editingFinished.connect(self.update_plot)
        fh_ticks.addWidget(self.spin_f_tick)
        lyt_font.addLayout(fh_ticks)
        
        fh_leg = QHBoxLayout()
        fh_leg.addWidget(QLabel("Leyenda:"))
        self.spin_f_leg = QSpinBox()
        self.spin_f_leg.setValue(12)
        self.spin_f_leg.editingFinished.connect(self.update_plot)
        fh_leg.addWidget(self.spin_f_leg)
        lyt_font.addLayout(fh_leg)
        
        lyt_font.addWidget(QLabel("Leyenda Pos X (%)"))
        self.slider_leg_x = QSlider(Qt.Horizontal)
        self.slider_leg_x.setRange(0, 100)
        self.slider_leg_x.setValue(100)
        self.slider_leg_x.valueChanged.connect(self.update_plot)
        lyt_font.addWidget(self.slider_leg_x)
        
        lyt_font.addWidget(QLabel("Leyenda Pos Y (%)"))
        self.slider_leg_y = QSlider(Qt.Horizontal)
        self.slider_leg_y.setRange(0, 100)
        self.slider_leg_y.setValue(100)
        self.slider_leg_y.valueChanged.connect(self.update_plot)
        lyt_font.addWidget(self.slider_leg_y)
        
        self.chk_show_leg = QCheckBox("Mostrar Leyenda")
        self.chk_show_leg.setChecked(True)
        self.chk_show_leg.stateChanged.connect(self.update_plot)
        lyt_font.addWidget(self.chk_show_leg)
        
        ctrl_layout.addWidget(grp_font)
        
        # --- EXPORTACION ---
        btn_png = QPushButton("📸 Exportar PNG")
        btn_png.clicked.connect(lambda: self.export_fig('png'))
        btn_pdf = QPushButton("📄 Exportar PDF Vectorial")
        btn_pdf.clicked.connect(lambda: self.export_fig('pdf'))
        
        ctrl_layout.addWidget(btn_png)
        ctrl_layout.addWidget(btn_pdf)
        
        ctrl_layout.addStretch()
        ctrl_panel.setWidget(ctrl_content)
        
        # Panel derecho (Matplotlib Canvas)
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        plot_layout = QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        
        plot_widget = QWidget()
        plot_widget.setLayout(plot_layout)
        
        layout.addWidget(ctrl_panel)
        layout.addWidget(plot_widget)
        
        self._load_aesthetics_config()

    def _load_aesthetics_config(self):
        try:
            from utils.config_manager import ConfigManager
            cm = ConfigManager()
            est = cm.get("estetica_exportacion") or {}
            
            if "f_title" in est: self.spin_f_title.setValue(est["f_title"])
            if "f_tick" in est: self.spin_f_tick.setValue(est["f_tick"])
            if "f_leg" in est: self.spin_f_leg.setValue(est["f_leg"])
            if "lw" in est: self.spin_lw.setValue(est["lw"])
            if "show_leg" in est: self.chk_show_leg.setChecked(est["show_leg"])
            if "leg_x" in est: self.slider_leg_x.setValue(est["leg_x"])
            if "leg_y" in est: self.slider_leg_y.setValue(est["leg_y"])
        except Exception as e:
            print("Error cargando estetica:", e)

    def _save_aesthetics_config(self):
        try:
            from utils.config_manager import ConfigManager
            cm = ConfigManager()
            est = cm.get("estetica_exportacion") or {}
            
            est["f_title"] = self.spin_f_title.value()
            est["f_tick"] = self.spin_f_tick.value()
            est["f_leg"] = self.spin_f_leg.value()
            est["lw"] = self.spin_lw.value()
            est["show_leg"] = self.chk_show_leg.isChecked()
            est["leg_x"] = self.slider_leg_x.value()
            est["leg_y"] = self.slider_leg_y.value()
            
            cm.set("estetica_exportacion", est)
        except Exception as e:
            print("Error guardando estetica:", e)

    def _toggle_auto_sep(self):
        is_auto = self.chk_auto_sep.isChecked()
        self.spin_sep_y.setEnabled(not is_auto)
        if is_auto:
            self.update_plot()

    def _select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar CSV", "", "CSV Files (*.csv)")
        if path:
            self.filepath = path
            self.load_data()
            
    def load_data(self):
        try:
            if os.path.isdir(self.filepath):
                self.filepath = os.path.join(self.filepath, "grabacion.csv")
                
            self.df = pd.read_csv(self.filepath)
            
            medicion_name = os.path.basename(os.path.dirname(self.filepath))
            self.lbl_file.setText(f"Medición: {medicion_name}")
            
            # Limpiar controles viejos
            for item in self.chk_canales:
                item['chk'].deleteLater()
                item['edit'].deleteLater()
                item['layout'].deleteLater()
            self.chk_canales.clear()
            
            # Limpiar layout principal de canales
            while self.lyt_chan.count():
                child = self.lyt_chan.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout():
                    pass
            
            self.df.columns = self.df.columns.str.strip()
            time_col = None
            for c in ['Time', 'Tiempo', 'Time(s)']:
                if c in self.df.columns:
                    time_col = c; break
            if not time_col and len(self.df.columns) > 0: time_col = self.df.columns[0]
            self.time_col = time_col
            
            self.signal_cols = [c for c in self.df.columns if c != time_col]
            
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
            
            for c in self.signal_cols:
                self.df[c] = (self.df[c] / ganancia) * 1e6
                
            # Crear checkboxes por canal encontrado
            for col in self.signal_cols:
                # Buscar nombre en config
                c_idx = col.split('_')[-1] if '_' in col else col
                conf = self.canales_conf.get(f"Canal {c_idx}", {})
                musculo = conf.get("musculo", col)
                
                h_lyt = QHBoxLayout()
                chk = QCheckBox(col)
                chk.setChecked(True)
                chk.stateChanged.connect(self.update_plot)
                
                edit = QLineEdit(musculo)
                edit.editingFinished.connect(self.update_plot)
                
                h_lyt.addWidget(chk)
                h_lyt.addWidget(edit)
                
                # Crear un widget contenedor para el layout horizontal
                row_widget = QWidget()
                row_widget.setLayout(h_lyt)
                self.lyt_chan.addWidget(row_widget)
                
                self.chk_canales.append({
                    'chk': chk,
                    'edit': edit,
                    'layout': row_widget,
                    'col_name': col
                })
                
            self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"No se pudo cargar el CSV:\n{e}")

    def procesar_senal(self, data, fs=1000.0):
        y = data.copy()
        
        if self.chk_notch.isChecked() and fs > 110:
            b, a = signal.iirnotch(50.0, 30.0, fs)
            y = signal.filtfilt(b, a, y)
            
        if self.chk_bp.isChecked():
            hp = self.spin_hp.value()
            lp = self.spin_lp.value()
            if hp > 0 and hp < fs/2:
                b, a = signal.butter(4, hp / (0.5 * fs), btype='high')
                y = signal.filtfilt(b, a, y)
            if lp > 0 and lp < fs/2:
                b, a = signal.butter(4, lp / (0.5 * fs), btype='low')
                y = signal.filtfilt(b, a, y)
                
        env_mode = self.cmb_env.currentText()
        if env_mode != "ninguna":
            window_ms = self.spin_env.value()
            env_window = int((window_ms / 1000.0) * fs)
            if env_window < 1: env_window = 1
            
            if env_mode == "rms":
                y_sq = y**2
                kernel = np.ones(env_window) / env_window
                y = np.sqrt(np.convolve(y_sq, kernel, mode='same'))
            elif env_mode == "media_movil":
                y = np.abs(y)
                kernel = np.ones(env_window) / env_window
                y = np.convolve(y, kernel, mode='same')
                
        return y

    def update_plot(self):
        if self.df is None: return
        
        self.ax.clear()
        
        is_cyber = self.rad_cyber.isChecked()
        is_sep = self.rad_sep.isChecked()
        
        bg_color = '#0B0C10' if is_cyber else '#FFFFFF'
        fg_color = '#C5C6C7' if is_cyber else '#000000'
        grid_color = '#1F2833' if is_cyber else '#DDDDDD'
        
        self.fig.patch.set_facecolor(bg_color)
        self.ax.set_facecolor(bg_color)
        
        self.ax.spines['bottom'].set_color(fg_color)
        self.ax.spines['top'].set_color(bg_color) 
        self.ax.spines['right'].set_color(bg_color)
        self.ax.spines['left'].set_color(fg_color)
        
        self.ax.tick_params(axis='x', colors=fg_color)
        self.ax.tick_params(axis='y', colors=fg_color)
        
        self.ax.xaxis.label.set_color(fg_color)
        self.ax.yaxis.label.set_color(fg_color)
        self.ax.title.set_color(fg_color)
        
        # Colores
        cyber_colors = ['#00ffcc', '#ff00ff', '#ffff00', '#00ff00']
        light_colors = ['#1f77b4', '#800080', '#ff7f0e', '#2ca02c']
        colors = cyber_colors if is_cyber else light_colors
        
        fs = 1000.0 # Valor seguro
        
        f_title = self.spin_f_title.value()
        f_tick = self.spin_f_tick.value()
        f_leg = self.spin_f_leg.value()
        
        if hasattr(self, 'time_col') and self.time_col in self.df.columns:
            t = self.df[self.time_col].values
            self.ax.set_xlabel("Tiempo (s)", fontsize=f_title)
            if len(t) > 1:
                fs = 1.0 / (t[1] - t[0])
        else:
            t = np.arange(len(self.df))
            self.ax.set_xlabel("Muestras", fontsize=f_title)
            
        self.ax.tick_params(axis='both', which='major', labelsize=f_tick)
            
        active_cols = []
        labels_map = {}
        for item in self.chk_canales:
            if item['chk'].isChecked():
                col = item['col_name']
                active_cols.append(col)
                labels_map[col] = item['edit'].text()
        
        offset = 0
        offset_step = 0
        
        # Calcular offset basándonos en la amplitud máxima
        if is_sep and len(active_cols) > 0:
            if self.chk_auto_sep.isChecked():
                max_amps = []
                for col in active_cols:
                    y = self.procesar_senal(self.df[col].values, fs)
                    max_amps.append(np.max(np.abs(y)))
                offset_step = np.max(max_amps) * 1.1 if max_amps else 100
                self.spin_sep_y.blockSignals(True)
                self.spin_sep_y.setValue(int(offset_step))
                self.spin_sep_y.blockSignals(False)
            else:
                offset_step = self.spin_sep_y.value()
        
        lines = []
        for i, col in enumerate(active_cols):
            y = self.procesar_senal(self.df[col].values, fs)
            
            musculo = labels_map.get(col, col)
            
            c = colors[i % len(colors)]
            
            if is_sep:
                y = y - (i * offset_step)
                
            line, = self.ax.plot(t, y, color=c, label=musculo, linewidth=self.spin_lw.value())
            lines.append(line)
            
        if self.chk_grid.isChecked():
            alpha = self.slider_grid.value() / 100.0
            self.ax.grid(True, color=grid_color, alpha=alpha, linestyle='--')
        else:
            self.ax.grid(False)
            
        if is_sep:
            # Ocultar ticks Y y poner Scale Bar
            self.ax.set_yticks([])
            self.ax.spines['left'].set_visible(False)
            self.ax.set_ylabel("")
            
            if len(active_cols) > 0:
                sb_val = self.spin_scale.value()
                # Posición scale bar: esquina inferior derecha
                x_min, x_max = self.ax.get_xlim()
                y_min, y_max = self.ax.get_ylim()
                
                sb_x = x_max - (x_max - x_min) * 0.05
                sb_y1 = y_min + (y_max - y_min) * 0.05
                sb_y2 = sb_y1 + sb_val
                
                self.ax.plot([sb_x, sb_x], [sb_y1, sb_y2], color=fg_color, linewidth=3)
                self.ax.text(sb_x + (x_max - x_min)*0.01, (sb_y1+sb_y2)/2, f"{sb_val} µV", 
                             color=fg_color, va='center', fontsize=f_tick, fontweight='bold')
        else:
            self.ax.spines['left'].set_visible(True)
            self.ax.set_ylabel("Amplitud (µV)", fontsize=f_title)
            
        if lines and self.chk_show_leg.isChecked():
            pos_x = self.slider_leg_x.value() / 100.0
            pos_y = self.slider_leg_y.value() / 100.0
            
            leg = self.ax.legend(loc='lower left', bbox_to_anchor=(pos_x, pos_y), 
                                 frameon=True, facecolor=bg_color, edgecolor=fg_color, fontsize=f_leg)
            for text in leg.get_texts():
                text.set_color(fg_color)
                
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._save_aesthetics_config()

    def export_fig(self, fmt='png'):
        filepath, _ = QFileDialog.getSaveFileName(self, f"Exportar {fmt.upper()}", f"grafico_paper.{fmt}", f"{fmt.upper()} (*.{fmt})")
        if filepath:
            try:
                self.fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Éxito", f"Gráfico exportado a:\n{filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"No se pudo guardar:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    win = ExportadorMatplotlib(path)
    win.show()
    sys.exit(app.exec())
