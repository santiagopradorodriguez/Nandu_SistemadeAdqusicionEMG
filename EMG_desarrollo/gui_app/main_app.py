# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Punto de entrada principal para la aplicación gráfica (GUI).
# ==============================================================================

import sys
import os

# Añadir carpeta base al path para imports relativos al proyecto SOLO si NO está congelado por PyInstaller
if not getattr(sys, 'frozen', False):
    root_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root_project_dir)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- MULTIPLEXOR PARA PYINSTALLER (SINGLE EXECUTABLE) ---
if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and (sys.argv[1].endswith('.py') or sys.argv[1].endswith('.pyw')):
  raw_arg = sys.argv[1]

  # 1. Si el archivo existe físicamente en el disco (ej. temp_run_pca.py, temp_comparativo.py, scripts temporales)
  if os.path.exists(raw_arg):
    import runpy
    script_abs = os.path.abspath(raw_arg)
    sys.argv = [script_abs] + sys.argv[2:]
    script_dir = os.path.dirname(script_abs)
    if script_dir not in sys.path:
      sys.path.insert(0, script_dir)
    try:
      runpy.run_path(script_abs, run_name="__main__")
    except Exception as e:
      import traceback
      print(f"Error al ejecutar script en disco ({script_abs}): {e}\n{traceback.format_exc()}")
      input("\nPresiona ENTER para cerrar esta ventana...")
    sys.exit(0)

  # 2. Si es un módulo empaquetado interno (ej. 'acquisition/manual_daq.py' o 'acquisition.manual_daq')
  sys.argv = [raw_arg] + sys.argv[2:]
  clean_name = os.path.normpath(raw_arg).replace('\\', '/')
  if clean_name.endswith('.py') or clean_name.endswith('.pyw'):
    clean_name = os.path.splitext(clean_name)[0]
  module_name = clean_name.replace('/', '.')
  while module_name.startswith('.'):
    module_name = module_name[1:]
  for prefix in ['EMG_desarrollo.', 'EMG_Ejecutable_Build.']:
    if module_name.startswith(prefix):
      module_name = module_name[len(prefix):]

  try:
    if module_name == 'acquisition.manual_daq':
      import acquisition.manual_daq as module
    elif module_name == 'acquisition.autoforge_daq':
      import acquisition.autoforge_daq as module
    elif module_name == 'acquisition.autoforge_daq_experimental':
      import acquisition.autoforge_daq_experimental as module
    elif module_name == 'acquisition.metronomo_visual':
      import acquisition.metronomo_visual as module
    elif module_name == 'acquisition.ventana_palabras':
      import acquisition.ventana_palabras as module
    elif module_name == 'acquisition.modulo_de_entrenamiento':
      import acquisition.modulo_de_entrenamiento as module
    elif module_name == 'analysis.reproductor_canal3':
      import analysis.reproductor_canal3 as module
    elif module_name == 'analysis.segmentador_secuencias':
      import analysis.segmentador_secuencias as module
    elif module_name == 'analysis.plotter_calibrado':
      import analysis.plotter_calibrado as module
    elif module_name in ('analysis.correlaciondeseñales', 'analysis.correlaciondesenales'):
      import analysis.correlaciondeseñales as module
    elif module_name == 'analysis.electrode_viewer_4':
      import analysis.electrode_viewer_4 as module
    elif module_name == 'analysis.analisis_estadistico_pulsos':
      import analysis.analisis_estadistico_pulsos as module
    elif module_name == 'analysis.discrete_motor':
      import analysis.discrete_motor as module
    elif module_name == 'analysis.pca_motor':
      import analysis.pca_motor as module
    elif module_name == 'analysis.training_motor':
      import analysis.training_motor as module
    elif module_name == 'analysis.umap_motor':
      import analysis.umap_motor as module
    elif module_name == 'analysis.generar_graficos_y_ranking':
      import analysis.generar_graficos_y_ranking as module
    elif module_name == 'analysis.plot_metricas_tesis':
      import analysis.plot_metricas_tesis as module
    elif module_name == 'instrucciones_uso':
      import instrucciones_uso as module
    elif module_name == 'utils.editor_mediciones':
      import utils.editor_mediciones as module
    elif module_name == 'utils.actualizar_metadata':
      import utils.actualizar_metadata as module
    elif module_name == 'utils.migrar_mediciones_por_fecha':
      import utils.migrar_mediciones_por_fecha as module
    elif module_name == 'deep_learning.pca_umap_clustering.generador_pca_umap':
      import deep_learning.pca_umap_clustering.generador_pca_umap as module
    elif module_name == 'deep_learning.binarizacion.analisis_trevisan':
      import deep_learning.binarizacion.analisis_trevisan as module
    elif module_name == 'deep_learning.binarizacion.analisis_trevisan_bandas':
      import deep_learning.binarizacion.analisis_trevisan_bandas as module
    elif module_name == 'deep_learning.binarizacion.analisis_binario':
      import deep_learning.binarizacion.analisis_binario as module
    elif module_name == 'deep_learning.pipeline_autoencoder_gui':
      import deep_learning.pipeline_autoencoder_gui as module
    elif module_name == 'deep_learning.dataset_tools.visor_features':
      import deep_learning.dataset_tools.visor_features as module
    elif module_name == 'deep_learning.dataset_tools.plot_3_musculos_standalone':
      import deep_learning.dataset_tools.plot_3_musculos_standalone as module
    elif module_name == 'deep_learning.dataset_tools.plot_derivadas_standalone':
      import deep_learning.dataset_tools.plot_derivadas_standalone as module
    elif module_name == 'deep_learning.dataset_tools.generador_pca_tensorial':
      import deep_learning.dataset_tools.generador_pca_tensorial as module
    elif module_name == 'deep_learning.generador_umap_supervisado':
      import deep_learning.generador_umap_supervisado as module
    elif module_name == 'deep_learning.pca_analysis':
      import deep_learning.pca_analysis as module
    elif module_name == 'deep_learning.umap_analysis':
      import deep_learning.umap_analysis as module
    elif module_name == 'deep_learning.experimento_grid_search_3_autoencoder':
      import deep_learning.experimento_grid_search_3_autoencoder as module
    else:
      import importlib
      module = importlib.import_module(module_name)

    if hasattr(module, 'main'):
      module.main()
    elif hasattr(module, 'flujo_principal'):
      module.flujo_principal()
    else:
      import runpy
      runpy.run_module(module_name, run_name="__main__")
  except Exception as e:
    import traceback
    error_msg = f"Error al ejecutar {module_name}: {e}\n{traceback.format_exc()}"
    print(error_msg)
    try:
      with open("multiplexer_error.log", "w", encoding="utf-8") as f:
        f.write(error_msg)
    except:
      pass
    print("\nPresiona ENTER para cerrar esta ventana...")
    input()
  sys.exit(0)
# --------------------------------------------------------



import subprocess
try:
  matplotlib.use('TkAgg') # Forzar TkAgg para que las ventanas de curación de Matplotlib pausen el script correctamente en PySide6
except Exception:
  pass
from PySide6.QtWidgets import (
  QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
  QDockWidget, QTextEdit, QLabel, QTreeView, QTabWidget,
  QToolBar, QPushButton, QSizePolicy, QMessageBox, QComboBox,
  QTableWidget, QTableWidgetItem, QScrollArea, QDialog
)
from PySide6.QtCore import Qt, QThreadPool, QSize, QTimer
from PySide6.QtGui import QFont, QColor, QTextCursor, QAction, QPixmap, QIcon, QCursor

app = QApplication.instance()
if not app:
  app = QApplication(sys.argv)



from core.threads import EmittingStream, Worker
from views.session_explorer import SessionExplorer
from views.ui_analysis import AnalysisPanel, MachineLearningPanel
from utils.path_utils import (
    get_project_root,
    get_database_path,
    get_comparative_path,
    get_session_analysis_path,
    get_resource_path,
)

# Importar la lógica de negocio original
try:
  from analysis.analisis_por_track_integrado import procesar_wavs_promedio
except ImportError:
  procesar_wavs_promedio = None

try:
  import qdarkstyle
except ImportError:
  qdarkstyle = None



class ImageLabel(QLabel):
  """QLabel especial que mantiene la relación de aspecto de la imagen al redimensionar la ventana."""
  def __init__(self, text="", parent=None):
    super().__init__(text, parent)
    self.setAlignment(Qt.AlignCenter)
    self._pixmap = None
    self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    self.setMinimumSize(100, 100)

  def setPixmap(self, pixmap):
    self._pixmap = pixmap
    self._update_pixmap()

  def resizeEvent(self, event):
    self._update_pixmap()
    super().resizeEvent(event)

  def _update_pixmap(self):
    if self._pixmap is not None and not self._pixmap.isNull():
      if self.width() > 0 and self.height() > 0:
        super().setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class ZoomableImageWidget(QWidget):
  """Widget de visualización de imágenes de alta resolución con zoom interactivo, scroll y modal de pantalla completa."""
  def __init__(self, placeholder="[Sin Imagen]", parent=None):
    super().__init__(parent)
    self.placeholder_text = placeholder
    self._pixmap = None
    self._filepath = None
    self._scale_factor = 1.0
    self._fit_mode = True

    # Debounce timer para evitar loops de resize con el window manager
    self._resize_timer = QTimer(self)
    self._resize_timer.setSingleShot(True)
    self._resize_timer.timeout.connect(self._on_debounced_resize)

    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)

    # Controls toolbar
    self.toolbar = QHBoxLayout()
    self.toolbar.setSpacing(4)

    self.btn_zoom_in = QPushButton("+ Zoom")
    self.btn_zoom_out = QPushButton("- Zoom")
    self.btn_zoom_fit = QPushButton("Ajustar")
    self.btn_zoom_100 = QPushButton("100% (1:1)")
    self.btn_fullscreen = QPushButton("Pantalla Completa")

    btn_style = """
      QPushButton {
        background-color: #151515; color: #00ffcc; border: 1px solid #333;
        padding: 3px 8px; font-size: 11px; font-weight: bold; border-radius: 3px;
      }
      QPushButton:hover { background-color: #00ffcc; color: #000; }
    """
    for btn in [self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_fit, self.btn_zoom_100, self.btn_fullscreen]:
      btn.setStyleSheet(btn_style)
      self.toolbar.addWidget(btn)

    self.toolbar.addStretch()
    self.lbl_zoom = QLabel("Ajustado")
    self.lbl_zoom.setStyleSheet("color: #888; font-size: 11px; padding-right: 4px;")
    self.toolbar.addWidget(self.lbl_zoom)
    layout.addLayout(self.toolbar)

    # Scroll area containing the image label
    self.scroll_area = QScrollArea()
    self.scroll_area.setWidgetResizable(True)
    self.scroll_area.setAlignment(Qt.AlignCenter)
    self.scroll_area.setStyleSheet("background-color: #0c0c0c; border: 1px solid #222;")
    self.scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    self.scroll_area.setMinimumSize(100, 100)

    self.img_label = QLabel(self.placeholder_text)
    self.img_label.setAlignment(Qt.AlignCenter)
    self.img_label.setStyleSheet("background-color: #0c0c0c; color: #666; font-size: 13px;")
    self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    self.img_label.setScaledContents(False)
    self.img_label.setCursor(Qt.PointingHandCursor)
    self.scroll_area.setWidget(self.img_label)
    layout.addWidget(self.scroll_area, stretch=1)

    self.btn_zoom_in.clicked.connect(self.zoom_in)
    self.btn_zoom_out.clicked.connect(self.zoom_out)
    self.btn_zoom_fit.clicked.connect(self.fit_to_view)
    self.btn_zoom_100.clicked.connect(self.reset_zoom)
    self.btn_fullscreen.clicked.connect(self.show_fullscreen)
    self.img_label.mouseDoubleClickEvent = lambda e: self.show_fullscreen()

  def setPixmap(self, pixmap, filepath=None):
    self._pixmap = pixmap
    self._filepath = filepath
    if self._pixmap is not None and not self._pixmap.isNull():
      self.img_label.setText("")
      if self._fit_mode:
        self.fit_to_view()
      else:
        self._apply_zoom()
    else:
      self.img_label.setPixmap(QPixmap())
      self.img_label.setText(self.placeholder_text)

  def setText(self, text):
    self._pixmap = None
    self._filepath = None
    self.img_label.setPixmap(QPixmap())
    self.img_label.setText(text)
    self.lbl_zoom.setText("-")

  def size(self):
    return self.scroll_area.viewport().size()

  def zoom_in(self):
    if self._pixmap is None or self._pixmap.isNull(): return
    self._fit_mode = False
    self._scale_factor = min(self._scale_factor * 1.25, 8.0)
    self._apply_zoom()

  def zoom_out(self):
    if self._pixmap is None or self._pixmap.isNull(): return
    self._fit_mode = False
    self._scale_factor = max(self._scale_factor / 1.25, 0.1)
    self._apply_zoom()

  def reset_zoom(self):
    if self._pixmap is None or self._pixmap.isNull(): return
    self._fit_mode = False
    self._scale_factor = 1.0
    self._apply_zoom()

  def fit_to_view(self):
    if self._pixmap is None or self._pixmap.isNull(): return
    self._fit_mode = True
    self.scroll_area.setWidgetResizable(True)
    self.img_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    vp_size = self.scroll_area.viewport().size()
    scaled_pix = self._pixmap.scaled(vp_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    self.img_label.setPixmap(scaled_pix)
    self.lbl_zoom.setText("Ajustado")

  def _apply_zoom(self, fit_text=False):
    if self._pixmap is None or self._pixmap.isNull(): return
    if self._fit_mode:
      self.fit_to_view()
      return
    self.scroll_area.setWidgetResizable(False)
    self.img_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    target_w = max(10, int(self._pixmap.width() * self._scale_factor))
    target_h = max(10, int(self._pixmap.height() * self._scale_factor))
    target_size = QSize(target_w, target_h)
    scaled_pix = self._pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    self.img_label.setPixmap(scaled_pix)
    self.img_label.resize(target_size)
    if fit_text:
      self.lbl_zoom.setText("Ajustado")
    else:
      self.lbl_zoom.setText(f"{int(self._scale_factor * 100)}%")

  def resizeEvent(self, event):
    super().resizeEvent(event)
    if self._fit_mode and self._pixmap is not None and not self._pixmap.isNull():
      self._resize_timer.start(50)

  def _on_debounced_resize(self):
    if self._fit_mode and self._pixmap is not None and not self._pixmap.isNull():
      self.fit_to_view()

  def show_fullscreen(self):
    if self._pixmap is None or self._pixmap.isNull(): return
    dialog = QDialog(self)
    dialog.setWindowTitle("NANDU LSD - Visor de Grafico de Alta Resolucion")
    dialog.setStyleSheet("background-color: #050505; color: #fff;")
    dlg_lyt = QVBoxLayout(dialog)
    dlg_lyt.setContentsMargins(5, 5, 5, 5)
    
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setStyleSheet("border: none; background: #000;")
    
    lbl = QLabel()
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setPixmap(self._pixmap)
    scroll.setWidget(lbl)
    dlg_lyt.addWidget(scroll)
    
    screen = self.screen().availableGeometry()
    max_w = int(screen.width() * 0.95)
    max_h = int(screen.height() * 0.95)
    dialog.resize(min(self._pixmap.width() + 40, max_w), min(self._pixmap.height() + 40, max_h))
    dialog.exec()

  open_fullscreen = show_fullscreen

class PatronMuscularViewerWidget(QWidget):
  def __init__(self):
    super().__init__()
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
    
    top_bar = QHBoxLayout()
    self.btn_correlacion = QPushButton("Graficar Patrón Muscular Desfasajes")
    self.btn_correlacion.setStyleSheet("""
        QPushButton {
            background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc;
            padding: 5px 15px; font-weight: bold; border-radius: 4px; margin: 5px;
        }
        QPushButton:hover { background-color: #00ffcc; color: #000; }
    """)
    top_bar.addWidget(self.btn_correlacion)
    top_bar.addStretch()
    layout.addLayout(top_bar)
    
    self.img_label = ImageLabel("[Visor de Patrón Muscular]\nAún no hay medición seleccionada")
    self.img_label.setStyleSheet("background-color: #0c0c0c; border: 1px solid #333; color: #555;")
    layout.addWidget(self.img_label)

  def load_patron(self, path):
    import os
    import glob
    from PySide6.QtGui import QPixmap
    search_pattern = os.path.join(path, "patron_muscular_*.png")
    files = glob.glob(search_pattern)
    if files:
      pix = QPixmap(files[0])
      self.img_label.clear()
      self.img_label.setPixmap(pix)
    else:
      self.img_label.clear()
      self.img_label._pixmap = None
      self.img_label.setText("[No se encontró gráfico de Patrón Muscular en esta medición]")
      self.img_label.update()

class ReaperStyleHub(QMainWindow):
  """
  Clase ReaperStyleHub.

  Representa y gestiona las operaciones relacionadas con ReaperStyleHub.
  """
  def __init__(self):
    """
    Ejecuta la funcionalidad de __init__.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    super().__init__()
    self.setWindowTitle("Ñandú LSD - EMG Analytics Studio (DAW Edition)")
    
    # Modo de acoplamiento múltiple para los paneles
    self.setDockOptions(QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks)
    
    # Área Central (Donde irían los gráficos grandes / Osciloscopio)
    self.setWindowTitle("Ñandú LSD - Estación Biomédica (EMG)")
    self.setGeometry(100, 100, 1400, 800)
    
    # Crear la Barra de Herramientas Superior
    self._create_main_toolbar()
    
    # Crear el Workspace Central (Pestañas)
    self.tabs = QTabWidget()
    self.tabs.setStyleSheet("QTabBar::tab { height: 40px; width: 250px; font-weight: bold; font-size: 14px; }")
    self.setCentralWidget(self.tabs)
    
    self._setup_tabs()
    
    # Crear los paneles flotantes (Docks) Globales
    self._create_dock_explorer()
    self._create_dock_terminal()
    
    self._setup_styles()
    
    # --- MOTOR DE HILOS Y CONSOLA INTERNA ---
    self.threadpool = QThreadPool()
    
    # Redirigir consola de Windows a nuestra UI
    sys.stdout = EmittingStream()
    sys.stdout.signal_obj.new_text.connect(self._write_to_console)
    
    print(f"> Ñandu LSD Initialized...")
    print(f"> Motor de hilos listo. CPU Threads: {self.threadpool.maxThreadCount()}")

  def _write_to_console(self, text):
    """
    Ejecuta la funcionalidad de _write_to_console.

    Args:
      text (Any): Argumento posicional text.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.log_console.moveCursor(QTextCursor.End)
    self.log_console.insertPlainText(text)
    self.log_console.ensureCursorVisible()

  def _launch_external(self, script_name, args=None):
    """
    Ejecuta la funcionalidad de _launch_external.

    Args:
      script_name (Any): Argumento posicional script_name.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    import subprocess
    import os
    if args is None:
        args = []
    
    if getattr(sys, 'frozen', False):
      try:
        if sys.platform == "win32":
          subprocess.Popen([sys.executable, script_name] + args, creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
          import shutil
          terminals = ['xterm', 'konsole', 'gnome-terminal', 'xfce4-terminal']
          term_cmd = None
          for t in terminals:
              if shutil.which(t):
                  if t == 'xterm': term_cmd = [t, '-e', sys.executable, script_name] + args
                  elif t == 'konsole': term_cmd = [t, '-e', sys.executable, script_name] + args
                  elif t == 'gnome-terminal': term_cmd = [t, '--', sys.executable, script_name] + args
                  else: term_cmd = [t, '-e', sys.executable, script_name] + args
                  break
          if term_cmd:
              subprocess.Popen(term_cmd)
          else:
              subprocess.Popen([sys.executable, script_name] + args)
      except Exception as e:
        QMessageBox.critical(self, "Error Crítico", f"Error al abrir {script_name}:\n{e}")
      return

    root_dir = get_project_root()
    script_path = os.path.join(root_dir, script_name)
    
    if not os.path.exists(script_path):
      QMessageBox.critical(self, "Error", f"No se encontró el script: {script_name} en {script_path}")
      return
      
    try:
      if sys.platform == "win32":
        subprocess.Popen([sys.executable, script_path] + args, creationflags=subprocess.CREATE_NEW_CONSOLE)
      else:
        import shutil
        terminals = ['xterm', 'konsole', 'gnome-terminal', 'xfce4-terminal']
        term_cmd = None
        for t in terminals:
            if shutil.which(t):
                if t == 'xterm': term_cmd = [t, '-e', sys.executable, script_path] + args
                elif t == 'konsole': term_cmd = [t, '-e', sys.executable, script_path] + args
                elif t == 'gnome-terminal': term_cmd = [t, '--', sys.executable, script_path] + args
                else: term_cmd = [t, '-e', sys.executable, script_path] + args
                break
        if term_cmd:
            subprocess.Popen(term_cmd)
        else:
            subprocess.Popen([sys.executable, script_path] + args)
    except Exception as e:
      QMessageBox.critical(self, "Error Crítico", f"Error al abrir {script_name}:\n{e}")

  def _run_reproductor_audio(self, *args):
    """Ejecuta el reproductor de audios pasándole la medición seleccionada"""
    selected_paths = self.explorer_widget.get_selected_paths()
    if selected_paths:
        self._launch_external("analysis/reproductor_canal3.py", args=[selected_paths[0]])
    else:
        self._launch_external("analysis/reproductor_canal3.py")

  def _create_main_toolbar(self):
    """
    Ejecuta la funcionalidad de _create_main_toolbar.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    toolbar = QToolBar("Herramientas Adicionales")
    toolbar.setIconSize(QSize(16, 16))
    toolbar.setMovable(False)
    toolbar.setFloatable(False)
    self.setCorner(Qt.TopLeftCorner, Qt.LeftDockWidgetArea)
    self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
    self.addToolBar(Qt.TopToolBarArea, toolbar)
    from PySide6.QtWidgets import QToolButton, QMenu
    
    def create_menu_button(title, items):
        btn = QToolButton()
        btn.setText(title)
        btn.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn)
        for name, script in items:
            action = QAction(name, self)
            if script == "_internal_config":
                action.triggered.connect(self._open_config_dialog)
            elif script == "analysis/plotter_calibrado.py":
                action.triggered.connect(self._run_plotter_calibrado)
            elif script == "analysis/correlaciondeseñales.py":
                action.triggered.connect(self._run_correlacion_nativo)
            elif script == "analysis/reproductor_canal3.py":
                action.triggered.connect(self._run_reproductor_audio)
            else:
                action.triggered.connect(lambda checked=False, s=script: self._launch_external(s))
            menu.addAction(action)
        btn.setMenu(menu)
        toolbar.addWidget(btn)
        
    create_menu_button(" Configuración & Ayuda", [
        (" Configuración General", "_internal_config"),
        ("Instrucciones y Créditos", "instrucciones_uso.py")
    ])

    create_menu_button(" Utilidades", [
        ("Entrenamiento AutoForge", "acquisition/modulo_de_entrenamiento.py"),
        ("Metrónomo", "acquisition/metronomo_visual.py"),
        ("Segmentador de Secuencias Continuas", "analysis/segmentador_secuencias.py"),
        ("Editar Medición", "utils/editor_mediciones.py"),
        ("Reproductor de Audios", "analysis/reproductor_canal3.py")
    ])

    create_menu_button(" Deep Learning", [
        ("Visualizador de Features", "deep_learning/dataset_tools/visor_features.py")
    ])

  def _open_config_dialog(self):
    """
    Ejecuta la funcionalidad de _open_config_dialog.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    from views.config_dialog import ConfiguracionDialog
    dialog = ConfiguracionDialog(self)
    dialog.exec()


  def _setup_tabs(self):
    # --- TAB 1: BIENVENIDA Y ADQUISICIÓN (DAQ) ---
    """
    Ejecuta la funcionalidad de _setup_tabs.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.tab_daq = QWidget()
    lyt_daq = QHBoxLayout(self.tab_daq)
    
    import os
    from pathlib import Path
    
    # --- Cargar Logo del Programa ---
    logo_path = None
    try:
      from utils.path_utils import get_resource_path, get_project_root
      candidate_paths = [
        get_resource_path("logo_nandu_lsd.png"),
        get_resource_path("logo.png"),
        os.path.join(get_project_root(), "logo_nandu_lsd.png"),
        os.path.join(get_project_root(), "EMG_desarrollo", "logo_nandu_lsd.png"),
        os.path.join(get_project_root(), "EMG_desarrollo", "gui_app", "assets", "logo_nandu_lsd.png"),
        os.path.join(get_project_root(), "EMG_desarrollo", "gui_app", "logo_nandu_lsd.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo_nandu_lsd.png"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo_nandu_lsd.png"),
      ]
      if hasattr(sys, '_MEIPASS'):
        candidate_paths.insert(0, os.path.join(sys._MEIPASS, "logo_nandu_lsd.png"))
        candidate_paths.insert(1, os.path.join(sys._MEIPASS, "logo.png"))
        
      for cp in candidate_paths:
        if cp and os.path.exists(cp):
          logo_path = cp
          break
          
      if not logo_path:
        root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        gui_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = gui_dir / "assets"
        pictures_dir = Path.home() / "Pictures"
        search_dirs = [assets_dir, gui_dir, root_dir, Path(get_project_root()), pictures_dir]
        if hasattr(sys, '_MEIPASS'):
          search_dirs.insert(0, Path(sys._MEIPASS))
        for search_dir in search_dirs:
          if search_dir.exists():
            for filename in os.listdir(search_dir):
              if filename.lower().startswith("logo") and filename.lower().endswith((".png", ".jpg", ".jpeg")):
                logo_path = str(search_dir / filename)
                break
          if logo_path:
            break
    except Exception as e:
      print(f"> Error buscando logo: {e}")

    # Contenedor izquierdo para logo e info
    vbox_info = QVBoxLayout()
    
    # Etiqueta para el Logo
    lbl_logo = QLabel()
    lbl_logo.setAlignment(Qt.AlignCenter)
    if logo_path and os.path.exists(logo_path):
      from PySide6.QtGui import QPixmap
      pix = QPixmap(logo_path)
      pix = pix.scaled(480, 180, Qt.KeepAspectRatio, Qt.SmoothTransformation)
      lbl_logo.setPixmap(pix)
      lbl_logo.setStyleSheet("padding: 10px; background-color: #050505; border-radius: 8px; border: 1px solid #222;")
    else:
      lbl_logo.setText("<h2>[ÑANDÚ LSD LOGO]</h2><p style='color:#888;'>Coloca un archivo 'logo_nandu_lsd.png' en la carpeta raíz</p>")
      lbl_logo.setStyleSheet("color: #FF0000; background-color: #111; border: 1px dashed #FF4444; padding: 20px; border-radius: 8px;")
    
    vbox_info.addWidget(lbl_logo)
    
    md_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "archivos_md", "documentacion_matematica.md"))
    file_url = f"file:///{md_path.replace(chr(92), '/')}"
    
    html_intro = f"""
    <div style='padding: 20px;'>
      <h2 style='color:#00ffff;'>Plataforma de Investigación EMG v6.1</h2>
      <p>Bienvenido al hub centralizado para adquisición en tiempo real, curación, procesamiento DSP y análisis de Deep Learning de señales electromiográficas (sEMG).</p>
      
      <h3 style='color:#00ffaa;'>Novedades Actualización v6.1 (Deep Learning, Topología & Arquitectura):</h3>
      <ul>
        <li><b>Sistema de Colores y Mapeo Anatómico por Músculo:</b> Convención estandarizada fija por canal: <i>Depresor Anguli Oris</i> (violeta), <i>Mylohyoid</i> (verde), <i>Orbicularis Oris</i> (amarillo) y <i>Micrófono/Canal 3</i> (rojo permanente). Diálogo interactivo al iniciar para confirmar la topografía de electrodos.</li>
        <li><b>Metadatos y Trazabilidad Enriquecida:</b> Registro unificado de <code>muscles_map</code>, <code>muscles</code> y <code>timestamp</code> en <code>metadata.json</code>, propagado automáticamente en recortes y segmentaciones.</li>
        <li><b>Motores Analíticos Desacoplados:</b> Nuevos motores independientes para análisis discreto (<code>discrete_motor</code>), barrido paramétrico y entrenamiento (<code>training_motor</code>), reducción dimensional PCA 2D/3D con siluetas y distancias inter-vocálicas (<code>pca_motor</code>), proyecciones UMAP supervisadas y no supervisadas (<code>umap_motor</code>) y generador automático de rankings de experimentos (<code>generar_graficos_y_ranking</code>).</li>
        <li><b>Autoencoders Convolucionales 1D (PyTorch):</b> Redes neuronales convolucionales profundas para compresión no lineal a espacios latentes 2D/3D y decodificación continua de potenciales mioeléctricos.</li>
        <li><b>Resolución Centralizada de Rutas (<code>path_utils</code>):</b> Desacople total que garantiza que <code>base_de_datos_electrodos</code> y reportes comparativos residan junto al ejecutable tanto en desarrollo como en distribución compilada.</li>
        <li><b>Optimización y Depuración:</b> Eliminación de librerías obsoletas (XGBoost) para un sistema más rápido, liviano y enfocado.</li>
      </ul>

      <h3 style='color:#ffaa00;'>Instrucciones Rápidas:</h3>
      <ul>
        <li><b>1. Adquisición:</b> Haz clic en el botón de la derecha para capturar nuevas mediciones con metrónomo y cuenta regresiva, confirmando los músculos en el diálogo inicial.</li>
        <li><b>2. Visualización:</b> Inspecciona las curvas en el Explorador CSV y visores de electrodos con identificación por color de cada músculo.</li>
        <li><b>3. Curación y Análisis:</b> Selecciona mediciones en el gestor de sesiones, verifica los canales deseados y haz clic en <i>PROCESAR</i> en la pestaña 3.</li>
        <li><b>4. Machine Learning & Deep Learning:</b> Accede a la Pestaña 4 para entrenar Autoencoders 1D, calcular proyecciones topológicas PCA/UMAP o binarizar patrones mediante Método Trevisan.</li>
      </ul>
      
      <h3 style='color:#00ff00;'>Fundamento Teórico:</h3>
      <p>Para revisar las ecuaciones utilizadas en la extracción de ruido (RMS, P-P) y las justificaciones del procesamiento, haz clic en el siguiente enlace:</p>
      <p> <a href='{file_url}' style='color:#00aaff; font-size: 14px;'><b>Abrir documentacion_matematica.md</b></a></p>
    </div>
    """
    
    lbl_intro = QLabel(html_intro)
    lbl_intro.setWordWrap(True)
    lbl_intro.setTextFormat(Qt.RichText)
    lbl_intro.setTextInteractionFlags(Qt.TextBrowserInteraction)
    lbl_intro.setOpenExternalLinks(True)
    lbl_intro.setStyleSheet("QLabel { background-color: #111111; color: white; border: 1px solid #333333; border-radius: 8px; margin-top: 10px; }")
    
    btn_daq = QPushButton(" INICIAR ADQUISICIÓN\nDE SEÑALES (DAQ)")
    btn_daq.setStyleSheet("""
      QPushButton {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 24px; 
        font-weight: 900; 
        background-color: #0d0d1a; 
        color: #00ffcc; 
        padding: 40px 20px;
        border-radius: 4px;
        border: 2px solid #00ffcc;
        border-right: 8px solid #ff003c;
        border-bottom: 8px solid #ff003c;
      }
      QPushButton:hover {
        background-color: #00ffcc;
        color: #0d0d1a;
        border: 2px solid #ff003c;
        border-right: 8px solid #ff003c;
        border-bottom: 8px solid #ff003c;
      }
      QPushButton:pressed {
        background-color: #ff003c;
        color: #ffffff;
        border: 2px solid #ffffff;
      }
    """)
    btn_daq.clicked.connect(lambda: self._launch_external("acquisition/manual_daq.py"))
    
    vbox_info.addWidget(lbl_intro, stretch=2)
    lyt_daq.addLayout(vbox_info, stretch=2)
    lyt_daq.addSpacing(20)
    
    vbox_btn = QVBoxLayout()
    vbox_btn.addStretch()
    vbox_btn.addWidget(btn_daq)
    
    btn_autoforge_staging = QPushButton("AUTOGRABADO")
    btn_autoforge_staging.setStyleSheet("""
      QPushButton {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 20px; 
        font-weight: 900; 
        background-color: #1a0d00; 
        color: #FF8800; 
        padding: 30px 15px;
        border-radius: 4px;
        border: 2px solid #FF8800;
        border-right: 8px solid #FFFF00;
        border-bottom: 8px solid #FFFF00;
        margin-top: 8px;
      }
      QPushButton:hover {
        background-color: #FF8800;
        color: #0a0000;
        border: 2px solid #FFFF00;
        border-right: 8px solid #FFFF00;
        border-bottom: 8px solid #FFFF00;
      }
      QPushButton:pressed {
        background-color: #FFFF00;
        color: #000000;
        border: 2px solid #ffffff;
      }
    """)
    btn_autoforge_staging.clicked.connect(lambda: self._launch_external("acquisition/autoforge_daq_experimental.py"))
    vbox_btn.addWidget(btn_autoforge_staging)
    
    vbox_btn.addStretch()
    
    lyt_daq.addLayout(vbox_btn, stretch=1)
    self.tabs.addTab(self.tab_daq, "1. INICIO Y ADQUISICIÓN")
    
    # --- TAB 2: VISUALIZACIÓN NATIVA ---
    self.tab_view = QWidget()
    lyt_view = QVBoxLayout(self.tab_view)
    self.tabs_viz = QTabWidget()
    from views.csv_viewer_widget import CsvViewerWidget
    self.csv_viewer = CsvViewerWidget()
    self.tabs_viz.addTab(self.csv_viewer, " Explorador de Señales (CSV)")
    from views.calibrated_viewer_widget import CalibratedViewerWidget
    self.calibrated_viewer = CalibratedViewerWidget()
    self.calibrated_viewer.request_generate_plots.connect(self._start_plot_generation)
    self.calibrated_viewer.btn_plot_3m.clicked.connect(self._run_plot_3m)
    self.tabs_viz.addTab(self.calibrated_viewer, " Historial Gráficos Musculares")
    from views.electrode_viewer_widget import ElectrodeViewerWidget
    self.electrode_viewer = ElectrodeViewerWidget()
    self.electrode_viewer.btn_refresh.clicked.connect(self._sync_electrode_viewer)
    self.tabs_viz.addTab(self.electrode_viewer, " Visor de Electrodos (Grilla)")
    self.patron_viewer = PatronMuscularViewerWidget()
    self.patron_viewer.btn_correlacion.clicked.connect(self._run_correlacion_nativo)
    self.tabs_viz.addTab(self.patron_viewer, " Historial Patrón Muscular")
    lyt_view.addWidget(self.tabs_viz)
    self.tabs.addTab(self.tab_view, "2. VISUALIZACIÓN")

    # --- TAB 3: ANÁLISIS Y EXTRACCIÓN ---
    self.tab_analysis = QWidget()
    lyt_analysis = QHBoxLayout(self.tab_analysis)
    self.analysis_panel = AnalysisPanel()
    self.analysis_panel.tab_procesamiento.btn_run_procesar.clicked.connect(lambda: self._run_analysis(interactivo=True))
    self.analysis_panel.tab_procesamiento.btn_run_rapido.clicked.connect(lambda: self._run_analysis(interactivo=False))
    self.analysis_panel.tab_comparativo.btn_run_comparativo.clicked.connect(self.run_analisis_comparativo_nativo)
    self.analysis_panel.tab_comparativo.btn_run_sesion.clicked.connect(self.run_analisis_sesion_nativo)
    lyt_analysis.addWidget(self.analysis_panel, stretch=1)

    self.panel_visor = QWidget()
    lyt_visor = QVBoxLayout(self.panel_visor)
    lyt_visor.setContentsMargins(5, 5, 5, 5)
    lyt_visor.setSpacing(5)
    
    h_visor = QHBoxLayout()
    self.cmb_resultados = QComboBox()
    self.btn_refrescar_visor = QPushButton("Refrescar")
    self.btn_refrescar_visor.setStyleSheet("""
        QPushButton {
            background-color: #1a1a1a; color: #00ffcc; border: 1px solid #00ffcc;
            padding: 4px 12px; font-weight: bold; border-radius: 3px;
        }
        QPushButton:hover { background-color: #00ffcc; color: #000; }
    """)
    h_visor.addWidget(QLabel("Archivo / Resultado:"))
    h_visor.addWidget(self.cmb_resultados, stretch=1)
    h_visor.addWidget(self.btn_refrescar_visor)
    lyt_visor.addLayout(h_visor)
    
    # Sub-pestañas para visualización de gráficos vs métricas y tablas estructuradas
    self.visor_subtabs = QTabWidget()
    self.visor_subtabs.setStyleSheet("""
        QTabWidget::pane { border: 1px solid #333; background: #050505; }
        QTabBar::tab { background: #111; color: #888; border: 1px solid #333; padding: 6px 12px; font-weight: bold; }
        QTabBar::tab:selected { background: #222; color: #00ffcc; border-color: #00ffcc; }
    """)
    
    # Sub-pestaña 1: Visor interactivo de imágenes con zoom
    self.img_viewer = ZoomableImageWidget("[Visor de Resultados Integrado]\nAquí aparecerán los gráficos generados")
    self.visor_subtabs.addTab(self.img_viewer, "Gráfico / Imagen")
    
    # Sub-pestaña 2: Métricas y Tablas (CSV / LaTeX / JSON / TXT)
    self.tab_metricas_visor = QWidget()
    lyt_metricas = QVBoxLayout(self.tab_metricas_visor)
    lyt_metricas.setContentsMargins(4, 4, 4, 4)
    
    self.tbl_metricas_visor = QTableWidget()
    self.tbl_metricas_visor.setStyleSheet("""
        QTableWidget {
            background-color: #0c0c0c; color: #eee; gridline-color: #333;
            border: 1px solid #333; font-family: monospace; font-size: 11px;
        }
        QHeaderView::section {
            background-color: #1a1a1a; color: #00ffcc; font-weight: bold;
            padding: 4px; border: 1px solid #333;
        }
        QTableWidget::item:selected { background-color: #004433; color: #00ffcc; }
    """)
    self.txt_metricas_visor = QTextEdit()
    self.txt_metricas_visor.setReadOnly(True)
    self.txt_metricas_visor.setStyleSheet("""
        QTextEdit {
            background-color: #0c0c0c; color: #00ffcc; font-family: 'Courier New', monospace;
            font-size: 11px; border: 1px solid #333; padding: 6px;
        }
    """)
    self.txt_metricas_visor.hide()
    
    lyt_metricas.addWidget(self.tbl_metricas_visor)
    lyt_metricas.addWidget(self.txt_metricas_visor)
    self.visor_subtabs.addTab(self.tab_metricas_visor, "Métricas y Tablas")
    
    lyt_visor.addWidget(self.visor_subtabs, stretch=1)
    self.panel_visor.setMinimumWidth(500)
    # Remove lyt_analysis.addWidget(self.panel_visor) since it moves to TAB 4
    
    self.tabs.addTab(self.tab_analysis, "3. ANÁLISIS Y EXTRACCIÓN")

    # --- TAB 4: MACHINE LEARNING ---
    self.tab_dl_ml = MachineLearningPanel()
    # Insert panel_visor as a tab inside MachineLearningPanel
    self.tab_dl_ml.tabs.addTab(self.panel_visor, "Galería de Resultados")
    
    self.btn_refrescar_visor.clicked.connect(self._refrescar_visor_imagenes)
    self.cmb_resultados.currentIndexChanged.connect(self._cargar_imagen_visor)
    self._refrescar_visor_imagenes()

    # Connect the ML tab buttons
    self.tab_dl_ml.tab_pca.btn_grid_search_2d.clicked.connect(lambda checked=False: self.run_pca_grid_search_nativo(n_components=2))
    self.tab_dl_ml.tab_pca.btn_grid_search_3d.clicked.connect(lambda checked=False: self.run_pca_grid_search_nativo(n_components=3))
    self.tab_dl_ml.tab_pca.btn_run.clicked.connect(self.run_pca_nativo)
    self.tab_dl_ml.tab_pca.btn_visor_features.clicked.connect(lambda: self._launch_external("deep_learning/dataset_tools/visor_features.py"))
    self.tab_dl_ml.tab_umap.btn_run.clicked.connect(self.run_umap_nativo)
    self.tab_dl_ml.tab_umap_sup.btn_run.clicked.connect(self.run_umap_supervisado_nativo)
    # (Botones btn_run_motor y btn_run_training se conectarán en un futuro cuando sus respectivos scripts nativos estén implementados)
    
    # Conectar los clasificadores (Trevisan, Autoencoders, Visor)
    self.tab_dl_ml.btn_trevisan.clicked.connect(lambda: self._launch_dl_ml_script("deep_learning/binarizacion/analisis_trevisan.py"))
    if hasattr(self.tab_dl_ml, 'btn_autoencoders'):
      self.tab_dl_ml.btn_autoencoders.clicked.connect(lambda: self._launch_dl_ml_script("deep_learning/pipeline_autoencoder_gui.py"))
    
    # Conexiones nativas de la pestaña Autoencoders
    if hasattr(self.tab_dl_ml, 'tab_autoencoders'):
      ae = self.tab_dl_ml.tab_autoencoders
      ae.btn_grid_search.clicked.connect(self.run_autoencoder_grid_search_nativo)
      ae.btn_extraer.clicked.connect(self.run_autoencoder_extraer_nativo)
      ae.btn_entrenar.clicked.connect(self.run_autoencoder_entrenar_nativo)
      ae.btn_plotear.clicked.connect(self.run_autoencoder_plotear_nativo)
      ae.btn_decodificador.clicked.connect(self.run_autoencoder_decodificador_nativo)
      ae.btn_visor_features.clicked.connect(lambda: self._launch_external("deep_learning/dataset_tools/visor_features.py"))
      ae.btn_pipeline_gui.clicked.connect(lambda: self._launch_dl_ml_script("deep_learning/pipeline_autoencoder_gui.py"))
    
    if hasattr(self.tab_dl_ml, 'btn_visor_features'):
      self.tab_dl_ml.btn_visor_features.clicked.connect(lambda: self._launch_external("deep_learning/dataset_tools/visor_features.py"))
    
    self.tabs.addTab(self.tab_dl_ml, "4. MACHINE LEARNING")

    # --- TAB 5: HISTORIAL DE RESULTADOS ---
    self.tab_historial = QWidget()
    lyt_historial = QVBoxLayout(self.tab_historial)
    self.tabs_historial = QTabWidget()
    
    import os
    from views.comparative_explorer_widget import ComparativeViewerWidget
    comparative_path = get_comparative_path()
    self.comparative_viewer = ComparativeViewerWidget(root_path=comparative_path)
    self.tabs_historial.addTab(self.comparative_viewer, "Historial de Comparativas")
    
    session_path = get_session_analysis_path()
    if not os.path.exists(session_path):
        os.makedirs(session_path)
    self.session_viewer = ComparativeViewerWidget(root_path=session_path)
    self.tabs_historial.addTab(self.session_viewer, "Historial de Sesión")
    
    lyt_historial.addWidget(self.tabs_historial)
    self.tabs.addTab(self.tab_historial, "5. HISTORIAL DE RESULTADOS")

  def _create_dock_explorer(self):
    """Panel tipo 'Media Explorer' o 'Gestor de Sesiones'"""
    self.dock_explorer = QDockWidget("Gestor de Sesiones", self)
    self.dock_explorer.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    
    # Le pasamos la ruta absoluta apuntando directo a base_de_datos_electrodos
    db_path = get_database_path()
    
    self.explorer_widget = SessionExplorer(root_path=db_path)
    self.explorer_widget.medicion_seleccionada.connect(self._on_medicion_selected_for_csv)
    self.explorer_widget.selection_changed.connect(self._on_explorer_selection_changed)
    
    self.dock_explorer.setWidget(self.explorer_widget)
    self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_explorer)

  def _on_medicion_selected_for_csv(self, path):
    """Carga la medición seleccionada en el visor de CSV y en el visor Calibrado."""
    if hasattr(self, 'calibrated_viewer'):
      self.calibrated_viewer.load_calibrated_plot(path)
      
    if hasattr(self, 'patron_viewer'):
      self.patron_viewer.load_patron(path)
      
    if hasattr(self, 'csv_viewer') and hasattr(self.csv_viewer, 'load_csv'):
      csv_path = None
      try:
        if os.path.isdir(path):
          for file in os.listdir(path):
            if file.lower().endswith('.csv'):
              csv_path = os.path.join(path, file)
              break
      except Exception as e:
        print(f"Error al buscar CSV: {e}")
      if csv_path:
        self.csv_viewer.load_csv(csv_path)
      else:
        print(f"> No se encontró archivo CSV en la medición: {path}")
    else:
      print(f"> Medición seleccionada para el CSV Viewer: {path}")

  def _on_explorer_selection_changed(self):
    """Reacciona a los checkboxes del SessionExplorer para habilitar botones y calcular canales comunes"""
    rutas = self.explorer_widget.get_selected_paths()
    n = len(rutas)
    
    # 1. Habilitar/Deshabilitar botones
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(n > 0)
    self.analysis_panel.tab_comparativo.btn_run_comparativo.setEnabled(n > 1)
    self.analysis_panel.tab_comparativo.btn_run_sesion.setEnabled(n > 1)
    
    # 2. Calcular Canales Totales (Procesamiento) y Comunes (Comparativo)
    cmb = self.analysis_panel.tab_comparativo.cmb_canal_comun
    lbl = self.analysis_panel.tab_comparativo.lbl_warning_canal
    cmb.clear()
    
    canales_totales = set()
    canales_comunes = None
    
    for path_medicion in rutas:
      canales_actuales = set()
      try:
        for item in os.listdir(path_medicion):
          channel_dir = os.path.join(path_medicion, item)
          if os.path.isdir(channel_dir) and item.startswith("canal_"):
            # Todos los canales van para el total (Procesamiento Individual)
            canales_totales.add(item)
            # Solo los procesados van para comunes (Análisis Comparativo)
            if os.path.exists(os.path.join(channel_dir, 'analisis_results.json')):
              canales_actuales.add(item)
      except Exception: continue
      
      if canales_comunes is None:
        canales_comunes = canales_actuales
      else:
        canales_comunes.intersection_update(canales_actuales)
      
    if n > 1:
      cmb.setEnabled(True)
      lbl.setText("Canales detectados listos para comparar.")
      lbl.setStyleSheet("color: #00FF00;")
    else:
      cmb.setEnabled(False)
      lbl.setText("Selecciona al menos 2 mediciones en el Gestor.")
      lbl.setStyleSheet("color: #FFA500;")
      
    # Llenar ComboBox (Comunes)
    if canales_comunes:
      sorted_canales_comunes = sorted(list(canales_comunes), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
      cmb.addItems(sorted_canales_comunes)
    elif n > 1:
      lbl.setText("Error: No hay canales comunes procesados.")
      lbl.setStyleSheet("color: #FF4444;")

    # 3. Llenar Checkboxes de Procesamiento Individual (Todos)
    tab_proc = self.analysis_panel.tab_procesamiento
    # Limpiar Layout
    while tab_proc.lyt_canales_procesar.count():
      child = tab_proc.lyt_canales_procesar.takeAt(0)
      if child.widget():
        child.widget().deleteLater()
        
    if n > 0:
      if canales_totales:
        sorted_canales_totales = sorted(list(canales_totales), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        tab_proc.checkboxes_canales = {}
        for canal in sorted_canales_totales:
          from PySide6.QtWidgets import QCheckBox
          musc_name = None
          ch_idx_str = canal.replace('canal_', '')
          for p in rutas:
            # 1. Chequear metadata específico de canal
            p_meta = os.path.join(p, canal, "metadata.json")
            if os.path.exists(p_meta):
              try:
                with open(p_meta, 'r', encoding='utf-8') as fm:
                  md_chk = json.load(fm)
                  if 'musculo' in md_chk and md_chk['musculo']:
                    musc_name = md_chk['musculo']
                    break
              except Exception: pass
            # 2. Chequear canal_0/metadata.json centralizado
            p_meta0 = os.path.join(p, "canal_0", "metadata.json")
            if os.path.exists(p_meta0) and not musc_name:
              try:
                with open(p_meta0, 'r', encoding='utf-8') as fm:
                  md0 = json.load(fm)
                  m_map = md0.get('muscles_map', {})
                  if ch_idx_str in m_map:
                    musc_name = m_map[ch_idx_str]
                    break
                  elif canal in m_map:
                    musc_name = m_map[canal]
                    break
                  elif 'muscles' in md0 and ch_idx_str.isdigit() and int(ch_idx_str) < len(md0['muscles']):
                    musc_name = md0['muscles'][int(ch_idx_str)]
                    break
              except Exception: pass
          if not musc_name:
            try:
              from utils.config_manager import ConfigManager
              cm = ConfigManager()
              c_conf = cm.get("canales") or {}
              musc_name = c_conf.get(f"Canal {ch_idx_str}", {}).get("musculo", "")
            except Exception: pass
          
          label_chk = f"{canal} - {musc_name}" if musc_name else canal
          chk = QCheckBox(label_chk)
          chk.setChecked(True)
          tab_proc.checkboxes_canales[canal] = chk
          tab_proc.lyt_canales_procesar.addWidget(chk)
      else:
        from PySide6.QtWidgets import QLabel
        lbl = QLabel("No se detectaron subcarpetas de canal en estas mediciones.")
        lbl.setStyleSheet("color: #FFA500;")
        tab_proc.lyt_canales_procesar.addWidget(lbl)
    else:
      from PySide6.QtWidgets import QLabel
      lbl = QLabel("Selecciona mediciones en el Gestor para ver los canales...")
      lbl.setStyleSheet("color: #FFA500;")
      tab_proc.lyt_canales_procesar.addWidget(lbl)

  def _run_plotter_calibrado(self, *args):
    """Ejecuta el plotter calibrado con la medición seleccionada"""
    selected_paths = self.explorer_widget.get_selected_paths()
    if not selected_paths:
      print(" Selecciona al menos una medición en el Gestor de Sesiones.")
      return
      
    import sys
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
      sys.path.append(root_dir)
      
    from analysis.plotter_calibrado import PlotterConfigDialog, plotear_medicion_secuencial
    
    # Le pasamos la lista de carpetas para que se visualicen en el dialog
    mediciones_nombres = [os.path.relpath(p, self.explorer_widget.root_path) for p in selected_paths]
    
    dialog = PlotterConfigDialog()
    # Pre-poblar el listbox con las seleccionadas para mantener compatibilidad
    dialog.listbox.clear()
    for nom in mediciones_nombres:
      dialog.listbox.addItem(nom)
      
    if dialog.exec():
      config = dialog.resultado
      limits_cache = {}
      for path in selected_paths:
        print(f" Generando gráfico calibrado para: {path}")
        # En plotter_calibrado, se usa on_close para capturar límites, pero aquí pasamos directo.
        # Pasamos mostrar_plot=True para que el usuario vea la ventana emergente 
        # y pueda interactuar/cerrarla antes de pasar a la siguiente (capturando los límites de zoom).
        plotear_medicion_secuencial(path, config, limits_cache, mostrar_plot=True)
        
      # Cambiamos automáticamente al Visor de Calibrados para que vea los resultados
      self.tabs.setCurrentWidget(self.tab_view)
      self.tabs_viz.setCurrentWidget(self.calibrated_viewer)
      # Cargamos la primera medición seleccionada
      self.calibrated_viewer.load_calibrated_plot(selected_paths[0])
      self.img_viewer.setText(f" ¡Gráficos musculares generados con éxito para {len(selected_paths)} mediciones!\nVe a la pestaña 3. VISUALIZACIÓN.")

  def _start_plot_generation(self, config):
    selected_paths = self.explorer_widget.get_selected_paths()
    if not selected_paths:
      from PySide6.QtWidgets import QMessageBox
      QMessageBox.warning(self, "Atención", "Por favor, seleccione al menos una medición en el Gestor de Sesiones.")
      return

    from PySide6.QtWidgets import QProgressDialog
    self.plot_progress = QProgressDialog("Iniciando procesamiento...", "Cancelar", 0, len(selected_paths), self)
    self.plot_progress.setWindowTitle("Generando Gráficos")
    self.plot_progress.setWindowModality(Qt.WindowModal)
    
    # Importar QThread localmente para evitar problemas de imports globales
    from PySide6.QtCore import QThread, Signal
    import os
    import traceback
    
    class PlotterThread(QThread):
        progress_update = Signal(int, int, str)
        finished_all = Signal(list)
        error_occurred = Signal(str)

        def __init__(self, paths, cfg, parent=None):
            super().__init__(parent)
            self.paths = paths
            self.cfg = cfg

        def run(self):
            try:
                from analysis.plotter_calibrado import plotear_medicion_secuencial
                limits_cache = {}
                total = len(self.paths)
                for i, path in enumerate(self.paths):
                    if self.isInterruptionRequested():
                        break
                    self.progress_update.emit(i, total, os.path.basename(path))
                    plotear_medicion_secuencial(path, self.cfg, limits_cache, mostrar_plot=False)
                self.finished_all.emit(self.paths)
            except Exception as e:
                self.error_occurred.emit(f"{e}\n{traceback.format_exc()}")

    self.plot_thread = PlotterThread(selected_paths, config, self)
    self.plot_progress.canceled.connect(self.plot_thread.requestInterruption)
    
    def on_progress(curr, tot, name):
        self.plot_progress.setLabelText(f"Procesando {curr+1} de {tot}:\n{name}")
        self.plot_progress.setValue(curr)
        
    def on_finished(paths):
        self.plot_progress.setValue(len(paths))
        if paths and hasattr(self, 'calibrated_viewer'):
            self.calibrated_viewer.load_calibrated_plot(paths[0])
            if hasattr(self, 'img_viewer'):
                self.img_viewer.setText(f" ¡Gráfico generado exitosamente!\nMostrando {os.path.basename(paths[0])}.")
            
    def on_error(err_msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", f"Fallo al graficar:\n{err_msg}")
        self.plot_progress.close()
        
    self.plot_thread.progress_update.connect(on_progress)
    self.plot_thread.finished_all.connect(on_finished)
    self.plot_thread.error_occurred.connect(on_error)
    
    self.plot_progress.show()
    self.plot_thread.start()

  def _run_plot_3m(self):
    selected_paths = self.explorer_widget.get_selected_paths()
    if not selected_paths:
      from PySide6.QtWidgets import QMessageBox
      QMessageBox.warning(self, "Atención", "Por favor, seleccione al menos una medición en el Gestor de Sesiones.")
      return
    theme = "dark" if self.calibrated_viewer.chk_oscuro.isChecked() else "light"
    
    try:
        smooth_ms = float(self.calibrated_viewer.inp_smooth.text())
    except:
        smooth_ms = 250.0
        
    self._launch_external("deep_learning/dataset_tools/plot_3_musculos_standalone.py", args=[selected_paths[0], theme, str(smooth_ms)])

  def _run_correlacion_nativo(self, *args):
    """Ejecuta el script de correlación con la medición seleccionada"""
    selected_paths = self.explorer_widget.get_selected_paths()
    if not selected_paths:
      print(" Selecciona al menos una medición en el Gestor de Sesiones.")
      return
      
    import sys
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root_dir not in sys.path:
      sys.path.append(root_dir)
      
    from analysis.correlaciondeseñales import main as run_correlacion
    
    print(f" Abriendo configuración de Correlación para {len(selected_paths)} mediciones...")
    run_correlacion(mediciones_dirs=selected_paths)
    print(" Correlación Finalizada.")

  def run_analisis_comparativo_nativo(self):
    """
    Ejecuta la funcionalidad de run_analisis_comparativo_nativo.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    rutas = self.explorer_widget.get_selected_paths()
    canal = self.analysis_panel.tab_comparativo.cmb_canal_comun.currentText()
    if not canal or len(rutas) < 2: return
    
    nombre_custom = self.analysis_panel.tab_comparativo.inp_nombre_analisis.text().strip()
    kwargs = self.analysis_panel.get_comparative_kwargs()
    
    self.log_console.append("\n" + "="*45)
    self.log_console.append("> LEVANTANDO ANÁLISIS COMPARATIVO (Nativo PySide6)")
    self.log_console.append("="*45)
    
    # Como _comparative_plots tiene Matplotlib, debemos aislarlo también
    import subprocess
    import sys
    
    # Recolectar nombres RELATIVOS (Fecha/Medicion)
    base_dir = os.path.dirname(os.path.dirname(rutas[0])) # Ej: base_de_datos_electrodos
    nombres_medicion = [f"{os.path.basename(os.path.dirname(r))}/{os.path.basename(r)}" for r in rutas]
    emg_root = os.path.dirname(base_dir)
    
    emg_root_escaped = emg_root.replace('\\', '/')
    base_dir_escaped = base_dir.replace('\\', '/')
    
    bridge_script = f"""
import sys
import os

# --- INYECCIÓN DE SEGURIDAD PARA RESOLUCIÓN DE MÓDULOS (PyInstaller / Nativo) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
# -------------------------------------------------------------------------------

from pathlib import Path
import json
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg') # Garantizar ventana comparativa en Windows
sys.path.append("{emg_root_escaped}")
import analysis.analisis_por_track_integrado as api

mediciones = {nombres_medicion}
base_dir = "{base_dir_escaped}"
canal = "{canal}"
nombre_custom = "{nombre_custom}"

try:
  resultados_globales = {{}}
  for med in mediciones:
    clave = f"{{med}}-{{canal}}"
    path = os.path.join(base_dir, med, canal, 'analisis_results.json')
    try:
      with open(path, 'r', encoding='utf-8') as f:
        res = json.load(f)
        
      # Intentar metadata
      meta_path = os.path.join(base_dir, med, canal, 'metadata.json')
      if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as fm:
          md = json.load(fm)
          if 'measurement_date' not in res: res['measurement_date'] = md.get('measurement_date','')
          if 'comentario' not in res: res['comentario'] = md.get('comentario','')
          
      res['file'] = clave
      resultados_globales[clave] = res
    except Exception as e:
      print(f"Error cargando {{path}}: {{e}}")

  if len(resultados_globales) > 1:
    promedios_globales = [res['mean_pulse'] for res in resultados_globales.values() if 'mean_pulse' in res]
    tiempos_globales = [res['pulse_time'] for res in resultados_globales.values() if 'pulse_time' in res]
    nombres_globales = [res['file'] for res in resultados_globales.values() if 'file' in res]
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    nombre_carpeta = nombre_custom if nombre_custom else f"comparacion_{{timestamp}}"
    
    out_dir = os.path.join("{emg_root_escaped}", "analisis_comparativos", today_str, nombre_carpeta)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "comparativa.png")
    
    api._comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados_globales, out_png, **{repr(kwargs)})
    print(f"ANÁLISIS COMPARATIVO FINALIZADO. Guardado en: {{out_dir}}")
  else:
    print("No hay suficientes resultados válidos cargados.")

except Exception as e:
  import traceback
  print("\\n" + "="*50)
  print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL ANÁLISIS")
  print("="*50)
  traceback.print_exc()
finally:
  input("\\nPresione ENTER para cerrar esta ventana...")
"""
    script_path = os.path.join(emg_root, "gui_app", "temp_comparativo.py")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
      f.write(bridge_script)
      
    # Ejecutar de forma asíncrona pero nativa con QProcess para detectar cuándo termina
    from PySide6.QtCore import QProcess, QProcessEnvironment
    from PySide6.QtGui import QPixmap
    from PySide6.QtCore import Qt
    
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QPixmap
    from PySide6.QtCore import Qt
    
    class ComparativeRunner(QThread):
      """
      Clase ComparativeRunner.

      Representa y gestiona las operaciones relacionadas con ComparativeRunner.
      """
      finished_signal = Signal(object)
      def __init__(self, spath):
        """
        Ejecuta la funcionalidad de __init__.

        Args:
          spath (Any): Argumento posicional spath.

        Returns:
          Any: Resultado de la ejecución de la función.
        """
        super().__init__()
        self.spath = spath
      def run(self):
        """
        Ejecuta la funcionalidad de run.

        Returns:
          Any: Resultado de la ejecución de la función.
        """
        import subprocess
        import sys
        import shutil
        if sys.platform == "win32":
          p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
          terminals = ['xterm', 'konsole', 'gnome-terminal', 'xfce4-terminal']
          term_cmd = None
          for t in terminals:
              if shutil.which(t):
                  if t == 'xterm': term_cmd = [t, '-e', sys.executable, self.spath]
                  elif t == 'konsole': term_cmd = [t, '--nofork', '-e', sys.executable, self.spath]
                  elif t == 'gnome-terminal': term_cmd = [t, '--wait', '--', sys.executable, self.spath]
                  else: term_cmd = [t, '-e', sys.executable, self.spath]
                  break
          if term_cmd:
              p = subprocess.run(term_cmd)
          else:
              p = subprocess.run([sys.executable, self.spath])
        self.finished_signal.emit(getattr(p, 'returncode', 0))
        
    self.comparative_thread = ComparativeRunner(script_path)
    
    def on_comparative_finished(exit_code):
      """
      Ejecuta la funcionalidad de on_comparative_finished.

      Args:
        exit_code (Any): Argumento posicional exit_code.

      Returns:
        Any: Resultado de la ejecución de la función.
      """
      self.analysis_panel.tab_comparativo.btn_run_comparativo.setText("LANZAR ANÁLISIS COMPARATIVO")
      self.analysis_panel.tab_comparativo.btn_run_comparativo.setEnabled(True)
      self.log_console.append(f"> Análisis comparativo finalizado en terminal nativa.")
      
      # Intentar cargar resultado_promedio_overlay.png
      out_dir = os.path.join(emg_root, "analisis_comparativos", "estadisticas_globales")
      img_path = os.path.join(out_dir, "resultado_promedio_overlay.png")
      if os.path.exists(img_path):
        pix = QPixmap(img_path)
        pix = pix.scaled(self.img_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_viewer.setPixmap(pix)
      else:
        self.img_viewer.setText("[Proceso Comparativo Finalizado: No se encontraron gráficos de overlay]")
        
    self.comparative_thread.finished_signal.connect(on_comparative_finished)
    self.comparative_thread.start()
    
    self.log_console.append(f"> Tarea enviada exitosamente. Abriendo terminal CMD nativa...")

  def run_analisis_sesion_nativo(self):
    rutas = self.explorer_widget.get_selected_paths()
    if len(rutas) < 2: return
    
    nombre_custom = self.analysis_panel.tab_comparativo.inp_nombre_analisis.text().strip()
    
    self.log_console.append("\n" + "="*45)
    self.log_console.append("> LEVANTANDO EVOLUCIÓN CONTINUA DE SESIÓN")
    self.log_console.append("="*45)
    
    import subprocess
    import sys
    
    base_dir = os.path.dirname(os.path.dirname(rutas[0]))
    nombres_medicion = [f"{os.path.basename(os.path.dirname(r))}/{os.path.basename(r)}" for r in rutas]
    emg_root = os.path.dirname(base_dir)
    
    emg_root_escaped = emg_root.replace('\\', '/')
    base_dir_escaped = base_dir.replace('\\', '/')
    
    bridge_script = f"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)

from pathlib import Path
import json
import re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
sys.path.append("{emg_root_escaped}")
import analysis.analisis_por_track_integrado as api

mediciones_a_comparar = {nombres_medicion}
base_dir = "{base_dir_escaped}"
nombre_custom = "{nombre_custom}"

try:
    mediciones_data = []
    for nombre_medicion in mediciones_a_comparar:
        path_medicion = os.path.join(base_dir, nombre_medicion)
        folder_name = os.path.basename(path_medicion)
        
        letra_match = re.match(r'^([AEIOUaeiou])_', folder_name)
        letra = letra_match.group(1).upper() if letra_match else '?'
        
        dt_obj = None
        hora_str = ""
        pulse_count = 0
        canales_data = {{}}
        muscles_map = {{}}
        meta0_path = os.path.join(path_medicion, 'canal_0', 'metadata.json')
        if not os.path.exists(meta0_path):
            meta0_path = os.path.join(path_medicion, 'metadata.json')
        if os.path.exists(meta0_path):
            try:
                with open(meta0_path, 'r', encoding='utf-8') as f0:
                    m0 = json.load(f0)
                    if 'muscles_map' in m0:
                        muscles_map = m0['muscles_map']
                    elif 'muscles' in m0 and isinstance(m0['muscles'], list):
                        muscles_map = {{i: m for i, m in enumerate(m0['muscles'])}}
            except Exception:
                pass
        
        for ch_idx in [0, 1, 2]:
            ch_key = f'canal_{{ch_idx}}'
            ch_path = os.path.join(path_medicion, ch_key)
            if not os.path.exists(ch_path): continue
            
            res_path = os.path.join(ch_path, 'analisis_results.json')
            meta_path = os.path.join(ch_path, 'metadata.json')
            
            if not os.path.exists(res_path): continue
            
            with open(res_path, 'r') as f:
                res = json.load(f)
                
            ch_musculo = ""
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    ch_musculo = meta.get('musculo', '')
                    if dt_obj is None:
                        mdate = meta.get('measurement_date', '')
                        if mdate:
                            try:
                                dt_obj = datetime.fromisoformat(mdate)
                                hora_str = dt_obj.strftime("%H:%M:%S")
                            except:
                                pass
                                
                    if pulse_count == 0:
                        pulse_count = meta.get('pulse_count', 0)
            if not ch_musculo and muscles_map:
                ch_musculo = muscles_map.get(str(ch_idx), muscles_map.get(ch_idx, ''))
            
            snr_per_pulse = []
            segmentos_rs = res.get('segmentos_rs', [])
            if not isinstance(segmentos_rs, list):
                segmentos_rs = []
                
            umbral = res.get('umbral', None)
            picos_ventana = res.get('picos_ventana', [])
            
            amp_per_pulse = []
            if isinstance(segmentos_rs, list) and len(segmentos_rs) > 0:
                for p in segmentos_rs:
                    if isinstance(p, list) and len(p) > 0:
                        p_arr = np.array(p)
                        # Resta de offset basal robusto como en plotter_calibrado
                        q25, q75 = np.percentile(p_arr, [25, 75])
                        iqr = q75 - q25
                        clean_base = p_arr[p_arr <= q75 + 1.5 * iqr]
                        p_base = np.percentile(clean_base, 10) if len(clean_base) >= 5 else np.min(p_arr)
                        p_clean = np.maximum(0.0, p_arr - p_base)
                        
                        amp_val = float(np.max(p_clean))
                        mav_val = float(np.mean(p_clean))
                        amp_per_pulse.append(amp_val)
                        if umbral and umbral > 0:
                            snr_per_pulse.append(mav_val / umbral)
                        else:
                            snr_per_pulse.append(np.nan)
                    else:
                        amp_per_pulse.append(np.nan)
                        snr_per_pulse.append(np.nan)
            elif isinstance(picos_ventana, list) and len(picos_ventana) > 0:
                for pv in picos_ventana:
                    if pv is not None and not np.isnan(pv):
                        amp_per_pulse.append(float(pv))
                        if umbral and umbral > 0:
                            snr_per_pulse.append(float(pv) / umbral)
                        else:
                            snr_per_pulse.append(np.nan)
                    else:
                        amp_per_pulse.append(np.nan)
                        snr_per_pulse.append(np.nan)
            else:
                amp_per_pulse = [np.nan] * len(snr_per_pulse)
                
            canales_data[ch_key] = {{
                'snr': snr_per_pulse,
                'amp': amp_per_pulse,
                'musculo': ch_musculo
            }}
            
        if dt_obj is None:
            dt_obj = datetime.now()
            hora_str = "??.??"
            
        mediciones_data.append({{
            'folder_name': folder_name,
            'letra': letra,
            'dt_obj': dt_obj,
            'hora_str': hora_str,
            'pulse_count': pulse_count,
            'muscles_map': muscles_map,
            'canales': canales_data
        }})
        
    mediciones_data.sort(key=lambda x: x['dt_obj'])
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    
    nombre_carpeta = nombre_custom if nombre_custom else f"Sesion_{{timestamp}}"
    output_comp_dir = os.path.join("{emg_root_escaped}", "analisis_de_sesiones", today_str, nombre_carpeta)
    os.makedirs(output_comp_dir, exist_ok=True)
    
    nombre_salida_base = os.path.join(output_comp_dir, "Sesion")
    
    api._comparative_session_plots(mediciones_data, nombre_salida_base)

except Exception as e:
    import traceback
    print("\\n" + "="*50)
    print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL ANÁLISIS DE SESIÓN")
    print("="*50)
    traceback.print_exc()
finally:
    input("\\nPresione ENTER para cerrar esta ventana...")
"""
    script_path = os.path.join(emg_root, "gui_app", "temp_sesion.py")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
      f.write(bridge_script)
      
    from PySide6.QtCore import QThread, Signal
    class SessionRunner(QThread):
      finished_signal = Signal(object)
      def __init__(self, spath):
        super().__init__()
        self.spath = spath
      def run(self):
        import subprocess
        import sys
        import shutil
        if sys.platform == "win32":
          p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
          terminals = ['xterm', 'konsole', 'gnome-terminal', 'xfce4-terminal']
          term_cmd = None
          for t in terminals:
              if shutil.which(t):
                  if t == 'xterm': term_cmd = [t, '-e', sys.executable, self.spath]
                  elif t == 'konsole': term_cmd = [t, '--nofork', '-e', sys.executable, self.spath]
                  elif t == 'gnome-terminal': term_cmd = [t, '--wait', '--', sys.executable, self.spath]
                  else: term_cmd = [t, '-e', sys.executable, self.spath]
                  break
          if term_cmd:
              p = subprocess.run(term_cmd)
          else:
              p = subprocess.run([sys.executable, self.spath])
        self.finished_signal.emit(getattr(p, 'returncode', 0))
        
    self.sesion_thread = SessionRunner(script_path)
    
    def on_sesion_finished(exit_code):
      self.analysis_panel.tab_comparativo.btn_run_sesion.setText("LANZAR EVOLUCIÓN DE SESIÓN")
      self.analysis_panel.tab_comparativo.btn_run_sesion.setEnabled(True)
      self.log_console.append(f"> Evolución Continua de Sesión finalizada en terminal nativa.")
    # Start thread
    self.log_console.append(f"Ejecutando sesión: python '{script_path}' ...\n")
    self.sesion_thread.start()

  def _refrescar_visor_imagenes(self):
    self.cmb_resultados.clear()
    import os
    from pathlib import Path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    paths_to_check = [
        os.path.join(root_dir, "resultados"),
        os.path.join(root_dir, "deep_learning", "pca_umap_clustering", "resultados_pca_umap"),
        os.path.join(root_dir, "deep_learning", "resultados_umap_supervisado"),
        os.path.join(root_dir, "analisis_comparativos")
    ]
    
    found_files = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.csv', '.tex', '.json', '.txt')
    
    for base_path in paths_to_check:
        if os.path.exists(base_path):
            for root, _, files in os.walk(base_path):
                for f in files:
                    if f.lower().endswith(valid_exts):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, base_path)
                        try:
                            mtime = os.path.getmtime(full_path)
                        except OSError:
                            mtime = 0
                        category = os.path.basename(base_path)
                        display_label = f"[{category}] {rel_path}"
                        found_files.append((mtime, display_label, full_path))
                        
    # Ordenar por fecha de modificación (los más recientes primero)
    found_files.sort(key=lambda x: x[0], reverse=True)
    for _, label, path in found_files:
        self.cmb_resultados.addItem(label, path)
        
    if self.cmb_resultados.count() > 0:
        self.cmb_resultados.setCurrentIndex(0)

  def _cargar_imagen_visor(self, index):
    if index < 0: return
    filepath = self.cmb_resultados.itemData(index)
    if not filepath or not os.path.exists(filepath):
        return
        
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext in ('.png', '.jpg', '.jpeg'):
        from PySide6.QtGui import QPixmap, QPixmapCache, QImage
        QPixmapCache.clear()
        img = QImage(filepath)
        pixmap = QPixmap.fromImage(img)
        self.img_viewer.setPixmap(pixmap, filepath=filepath)
        self.visor_subtabs.setCurrentIndex(0)
    elif ext == '.csv':
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            self.tbl_metricas_visor.clear()
            self.tbl_metricas_visor.setRowCount(df.shape[0])
            self.tbl_metricas_visor.setColumnCount(df.shape[1])
            self.tbl_metricas_visor.setHorizontalHeaderLabels([str(c) for c in df.columns])
            for r in range(df.shape[0]):
                for c in range(df.shape[1]):
                    val = df.iat[r, c]
                    item = QTableWidgetItem(f"{val:.3f}" if isinstance(val, float) else str(val))
                    item.setTextAlignment(Qt.AlignCenter)
                    self.tbl_metricas_visor.setItem(r, c, item)
            self.tbl_metricas_visor.resizeColumnsToContents()
            self.tbl_metricas_visor.show()
            self.txt_metricas_visor.hide()
            self.visor_subtabs.setCurrentIndex(1)
        except Exception as e:
            self.txt_metricas_visor.setPlainText(f"Error al leer archivo CSV:\n{e}")
            self.tbl_metricas_visor.hide()
            self.txt_metricas_visor.show()
            self.visor_subtabs.setCurrentIndex(1)
    elif ext in ('.tex', '.json', '.txt'):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            self.txt_metricas_visor.setPlainText(content)
            self.tbl_metricas_visor.hide()
            self.txt_metricas_visor.show()
            self.visor_subtabs.setCurrentIndex(1)
        except Exception as e:
            self.txt_metricas_visor.setPlainText(f"Error al leer archivo de texto:\n{e}")
            self.tbl_metricas_visor.hide()
            self.txt_metricas_visor.show()
            self.visor_subtabs.setCurrentIndex(1)

  def _generar_base_dir_y_mediciones(self):
    rutas = self.explorer_widget.get_selected_paths()
    if not rutas:
        self.log_console.append("\n[Error] Debe seleccionar al menos una carpeta.")
        return None, None
        
    from pathlib import Path
    import os
    mediciones_lista = []
    base_dir_global = ""
    for ruta in rutas:
        p = Path(ruta)
        parts = p.parts
        if "base_de_datos_electrodos" in parts:
            idx = parts.index("base_de_datos_electrodos")
            base_dir_global = os.path.join(*parts[:idx+1])
            if len(parts) > idx + 2:
                med_str = os.path.join(parts[idx+1], parts[idx+2])
                if med_str not in mediciones_lista:
                    mediciones_lista.append(med_str)
    return base_dir_global, mediciones_lista

  def _launch_bridge_script(self, name, title, kwargs, bridge_template, suffix=".py"):
    base_dir, mediciones = self._generar_base_dir_y_mediciones()
    if not base_dir: return

    self.log_console.append("\n" + "="*45)
    self.log_console.append(f"> INICIANDO {title}")
    self.log_console.append("="*45)

    import os, json, tempfile
    temp_json = tempfile.mktemp(suffix=".json")
    with open(temp_json, "w") as f:
        json.dump(kwargs, f)
        
    script_path = os.path.join(os.getcwd(), f"temp_{name}{suffix}")
    root_project_dir = get_project_root()
    
    bridge_script = bridge_template.replace("{TEMP_JSON}", temp_json)\
                                   .replace("{BASE_DIR}", base_dir)\
                                   .replace("{MEDICIONES}", str(mediciones))\
                                   .replace("{ROOT_PROJECT_DIR}", root_project_dir)
    
    with open(script_path, "w") as f:
        f.write(bridge_script)
        
    self.log_console.append(f"> SCRIPT GENERADO: {script_path}")
    
    import subprocess
    import shutil

    if sys.platform == "win32" or os.name == 'nt':
        # En Windows lanzamos en una consola dedicada para ver los logs
        try:
            cmd_win = ['cmd.exe', '/c', 'start', f'Ñandú LSD - {title}', 'cmd.exe', '/k', sys.executable, script_path]
            subprocess.Popen(cmd_win, shell=True)
            self.log_console.append("> SCRIPT LANZADO EN CONSOLA WINDOWS")
        except Exception as e:
            self.log_console.append(f"> [Error] Falló el lanzamiento en consola: {str(e)}")
            subprocess.Popen([sys.executable, script_path], creationflags=getattr(subprocess, 'CREATE_NEW_CONSOLE', 0))
    else:
        terminals = ['konsole', 'gnome-terminal', 'xfce4-terminal', 'mate-terminal', 'xterm']
        term_cmd = None
        
        for t in terminals:
            if shutil.which(t):
                if t == 'gnome-terminal':
                    term_cmd = [t, '--', 'bash', '-c', f"{sys.executable} '{script_path}'; echo '\nProceso finalizado. Presiona Enter para salir...'; read"]
                else:
                    term_cmd = [t, '-e', f"bash -c \"{sys.executable} '{script_path}'; echo '\nProceso finalizado. Presiona Enter para salir...'; read\""]
                break
                
        if term_cmd:
            try:
                subprocess.Popen(term_cmd)
                self.log_console.append(f"> SCRIPT LANZADO EN TERMINAL ({term_cmd[0]})")
            except Exception as e:
                self.log_console.append(f"> [Error] Falló el lanzamiento en terminal: {str(e)}")
                subprocess.Popen([sys.executable, script_path])
        else:
            self.log_console.append("> [Aviso] No se encontró terminal gráfica, lanzando en background...")
            subprocess.Popen([sys.executable, script_path])

  def run_pca_grid_search_nativo(self, n_components=2):
    base_dir, mediciones = self._generar_base_dir_y_mediciones()
    if not base_dir or not mediciones:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Advertencia", "Debe seleccionar al menos una medición en el Gestor de Sesiones.")
        return

    kwargs = self.tab_dl_ml.get_pca_kwargs()
    params_key = "params_2d" if n_components == 2 else "params_3d"

    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

mediciones = {MEDICIONES}
base_dir = r'{BASE_DIR}'

import deep_learning.pca_analysis as pca_ana

print("=========================================================")
print("     INICIANDO BÚSQUEDA DE PARÁMETROS ÓPTIMOS (PCA)     ")
print("=========================================================")

res = pca_ana.buscar_mejor_configuracion_pca(
    mediciones=mediciones,
    base_dir=base_dir,
    params_base=kwargs.get("{PARAMS_KEY}", {}),
    aplicar_trevisan=kwargs.get("aplicar_trevisan", False),
    modo_alineacion=kwargs.get("modo_alineacion", "Pico Volumen Micrófono"),
    pre_pct=kwargs.get("pre_pct", 0.4),
    post_pct=kwargs.get("post_pct", 0.6),
    canales_features=kwargs.get("canales_features", ["canal_0", "canal_1", "canal_2"]),
    ignorar_ventana_cero=kwargs.get("ignorar_ventana_cero", False),
    algoritmo_clustering=kwargs.get("algoritmo_clustering_pca", "GMM"),
    logger=print,
    n_components={N_COMPONENTS}
)

if isinstance(res, tuple) and len(res) >= 3:
    best_config = res[0]
    best_acc = res[1]
    best_sil = res[2]
    best_vocal_acc = res[3] if len(res) >= 4 and isinstance(res[3], dict) else {}

if best_config:
    if len(best_config) == 4:
        best_smooth, best_pts, best_alpha, best_notch = best_config
    else:
        best_smooth, best_pts, best_alpha = best_config[:3]
        best_notch = 2.0
    out_file = os.path.join(project_root, "deep_learning", "parametros_optimos_pca.json")
    with open(out_file, "w") as f:
        json.dump({
            "smooth_ms": best_smooth,
            "target_length": best_pts,
            "alpha_ruido": best_alpha,
            "notch_q": best_notch,
            "accuracy_clasificacion": best_acc,
            "porcentaje_por_vocal": best_vocal_acc,
            "silhouette_score": best_sil
        }, f, indent=4)
    print("")
    print("---------------------------------------------------------")
    print("¡CONFIGURACIÓN ÓPTIMA HALLADA (MAX CLASIFICACIÓN)! ")
    print("  - Smooth (Envolvente): " + str(best_smooth) + " ms")
    print("  - Remuestreo (Pts):    " + str(best_pts))
    print("  - Alfa Ruido:          " + str(best_alpha))
    print("  - Notch Q:             " + str(best_notch))
    print("  - Clasificación (%):   " + str(best_acc) + " %")
    if best_vocal_acc:
        print("  - Desglose por Vocal:")
        for v_name, v_pct in best_vocal_acc.items():
            print(f"      * Vocal {v_name}: {v_pct:.1f}%")
    print("  - Silhouette Score:    " + str(best_sil))
    print("---------------------------------------------------------")
    print("Se cargaron los resultados automáticamente.")
"""
    template = template.replace("{PARAMS_KEY}", params_key).replace("{N_COMPONENTS}", str(n_components))
    self._launch_bridge_script("run_pca_grid", f"GRID SEARCH PCA ({n_components}D)", kwargs, template)

    from PySide6.QtCore import QTimer
    import os, json
    
    def cargar_parametros_optimos():
        out_file = os.path.join(self.root_dir, "deep_learning", "parametros_optimos_pca.json")
        if os.path.exists(out_file):
            try:
                with open(out_file, "r") as f:
                    data = json.load(f)
                best_smooth = data.get("smooth_ms", 90)
                best_pts = data.get("target_length", 20)
                best_alpha = data.get("alpha_ruido", 0.5)
                best_notch = data.get("notch_q", 2.0)
                best_acc = data.get("accuracy_clasificacion", 0.0)
                best_vocal_acc = data.get("porcentaje_por_vocal", {})
                best_sil = data.get("silhouette_score", 0.0)

                pca_tab = self.tab_dl_ml.tab_pca
                if n_components == 2:
                    pca_tab.inp_alpha_2d.setValue(best_alpha)
                    pca_tab.inp_smooth_2d.setValue(best_smooth)
                    pca_tab.inp_pts_2d.setValue(best_pts)
                    pca_tab.inp_notch_2d.setValue(best_notch)
                elif n_components == 3:
                    pca_tab.inp_alpha_3d.setValue(best_alpha)
                    pca_tab.inp_smooth_3d.setValue(best_smooth)
                    pca_tab.inp_pts_3d.setValue(best_pts)
                    pca_tab.inp_notch_3d.setValue(best_notch)

                vocal_str = "\n".join([f"  • Vocal {v}: {acc:.1f}%" for v, acc in best_vocal_acc.items()])
                vocal_msg = f"\nDesglose por Vocal:\n{vocal_str}\n" if vocal_str else ""

                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Grid Search Finalizado",
                    f"¡Configuración Óptima Encontrada!\n\n"
                    f"- Envolvente (Smooth): {best_smooth} ms\n"
                    f"- Puntos Remuestreo: {best_pts}\n"
                    f"- Alfa Ruido: {best_alpha}\n"
                    f"- Notch Q: {best_notch}\n\n"
                    f"Precisión Clasificación (%): {best_acc:.2f}%\n"
                    f"{vocal_msg}"
                    f"Silhouette Score (PCA): {best_sil:.4f}\n\n"
                    f"Se han cargado automáticamente los parámetros en la interfaz."
                )
                self.timer_check_opt.stop()
                os.remove(out_file)
            except Exception as e:
                pass

    self.timer_check_opt = QTimer(self)
    self.timer_check_opt.setInterval(2000)
    self.timer_check_opt.timeout.connect(cargar_parametros_optimos)
    self.timer_check_opt.start()

  def run_pca_nativo(self):
    from PySide6.QtWidgets import QInputDialog
    nombre_set, ok = QInputDialog.getText(self, "Nombre del Set (PCA)", "Introduce un nombre para identificar este set de resultados:")
    if not ok or not nombre_set.strip():
        self.log_console.append("> [Aviso] Ejecución de PCA cancelada.")
        return
    nombre_set = nombre_set.strip().replace(" ", "_")

    kwargs = self.tab_dl_ml.get_pca_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

mediciones = {MEDICIONES}
base_dir = r'{BASE_DIR}'

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

# Define explicit out_dir based on user input
pca_umap_dir = os.path.join(project_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", \"""" + nombre_set + """\")
os.makedirs(pca_umap_dir, exist_ok=True)

# Save the kwargs into the folder
with open(os.path.join(pca_umap_dir, "parametros.json"), 'w') as f:
    json.dump(kwargs, f, indent=4)

generador.ejecutar_procesamiento(mediciones=mediciones, base_dir=base_dir, out_dir=pca_umap_dir, **kwargs)
"""
    self._launch_bridge_script("run_pca", "PCA (COMPONENTES PRINCIPALES)", kwargs, template)

  def run_umap_nativo(self):
    from PySide6.QtWidgets import QInputDialog
    nombre_set, ok = QInputDialog.getText(self, "Nombre del Set (UMAP)", "Introduce un nombre para identificar este set de resultados:")
    if not ok or not nombre_set.strip():
        self.log_console.append("> [Aviso] Ejecución de UMAP cancelada.")
        return
    nombre_set = nombre_set.strip().replace(" ", "_")

    kwargs = self.tab_dl_ml.get_umap_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

mediciones = {MEDICIONES}
base_dir = r'{BASE_DIR}'

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

pca_umap_dir = os.path.join(project_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", \"""" + nombre_set + """\")
os.makedirs(pca_umap_dir, exist_ok=True)

with open(os.path.join(pca_umap_dir, "parametros.json"), 'w') as f:
    json.dump(kwargs, f, indent=4)

generador.ejecutar_procesamiento(mediciones=mediciones, base_dir=base_dir, out_dir=pca_umap_dir, **kwargs)
"""
    self._launch_bridge_script("run_umap", "UMAP (NO LINEAL)", kwargs, template)

  def run_umap_supervisado_nativo(self):
    base_dir, mediciones = self._generar_base_dir_y_mediciones()
    if not base_dir: return

    # Extraer nombres de sesiones únicas (formato: "fecha/sesion")
    sesiones_unicas = list(set(mediciones))
    sesiones_unicas.sort()

    from views.ui_analysis import TrainTestSplitDialog
    dialog = TrainTestSplitDialog(sesiones_unicas, self)
    if dialog.exec() != TrainTestSplitDialog.Accepted:
        self.log_console.append("> [Aviso] Ejecución de UMAP Supervisado cancelada.")
        return
        
    nombre_set, train_sessions, test_sessions = dialog.get_results()
    if not nombre_set:
        nombre_set = "umap_supervisado_run"

    kwargs = self.tab_dl_ml.get_umap_supervisado_kwargs()
    # Inyectar train_sessions y test_sessions a kwargs para pasarlo al script
    kwargs["train_sessions"] = train_sessions
    kwargs["test_sessions"] = test_sessions

    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

mediciones = {MEDICIONES}
base_dir = r'{BASE_DIR}'

import deep_learning.generador_umap_supervisado as gen_sup

out_dir = os.path.join(project_root, "deep_learning", "resultados_umap_supervisado", \"""" + nombre_set + """\")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(out_dir, "parametros.json"), 'w') as f:
    json.dump(kwargs, f, indent=4)

gen_sup.ejecutar_procesamiento(
    mediciones=mediciones,
    base_dir=base_dir,
    out_dir=out_dir,
    **kwargs
)
"""
    self._launch_bridge_script("run_umap_sup", "UMAP SUPERVISADO", kwargs, template)

  def run_autoencoder_extraer_nativo(self):
    rutas = self.explorer_widget.get_selected_paths()
    if not rutas:
      self.log_console.append("> ERROR: Selecciona al menos una medición en el Gestor de Sesiones para extraer el dataset del Autoencoder.\n")
      return

    kwargs = self.tab_dl_ml.get_autoencoder_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

mediciones = {MEDICIONES}
base_dir = r'{BASE_DIR}'

import deep_learning.dataset_tools.generador_pca_tensorial as gpt

print("==================================================")
print("EXTRAYENDO DATASET TENSORIAL PARA AUTOENCODER...")
print(f"Mediciones seleccionadas: {len(mediciones)}")
print("==================================================")

X, Y, Tomas = gpt.extraer_features_concatenadas(
    base_dir=base_dir,
    mediciones=mediciones,
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    smooth_ms=kwargs.get('smooth_ms', 150),
    notch_q=kwargs.get('notch_q', 2.0),
    target_length=kwargs.get('target_length', 100),
    use_manual_exclusions=kwargs.get('use_manual_exclusions', True),
    verbose=True
)

out_dir = os.path.join(project_root, "resultados", "resultados_pca_tensorial")
os.makedirs(out_dir, exist_ok=True)
csv_out = os.path.join(out_dir, "caracteristicas_exportadas.csv")

import pandas as pd
df = pd.DataFrame(X)
df.insert(0, 'Toma', Tomas)
df.insert(0, 'Vocal', Y)
df.to_csv(csv_out, index=False)
print(f"\\n>>> DATASET EXPORTADO EXITOSAMENTE ({len(df)} muestras) EN: {csv_out} <<<")
"""
    self._launch_bridge_script("extraer_autoencoder", "EXTRACCION DATASET AUTOENCODER", kwargs, template)

  def run_autoencoder_entrenar_nativo(self):
    kwargs = self.tab_dl_ml.get_autoencoder_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

csv_candidates = [
    os.path.join(project_root, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "resultados", "resultados_pca_umap", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "deep_learning", "caracteristicas_exportadas.csv"),
]
csv_file = None
for c in csv_candidates:
    if os.path.exists(c):
        csv_file = c
        break

if not csv_file:
    print("> ERROR: No se encontró 'caracteristicas_exportadas.csv'. Ejecuta primero '1. EXTRAER DATASET'.")
    sys.exit(1)

import deep_learning.train_autoencoder as ta
ta.train_autoencoder(
    csv_path=csv_file,
    epochs=kwargs.get('epochs', 80),
    batch_size=kwargs.get('batch_size', 16),
    latent_dim=kwargs.get('latent_dim', 8),
    kernel_size=kwargs.get('kernel_size', 5),
    force_epochs=kwargs.get('force_epochs', False),
    alpha=kwargs.get('alpha_loss', 0.5),
    verbose=True
)
"""
    self._launch_bridge_script("entrenar_autoencoder", "ENTRENAMIENTO AUTOENCODER", kwargs, template)

  def run_autoencoder_plotear_nativo(self):
    kwargs = self.tab_dl_ml.get_autoencoder_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

csv_candidates = [
    os.path.join(project_root, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "resultados", "resultados_pca_umap", "caracteristicas_exportadas.csv"),
]
csv_file = next((c for c in csv_candidates if os.path.exists(c)), None)
if not csv_file:
    print("> ERROR: No se encontró 'caracteristicas_exportadas.csv'.")
    sys.exit(1)

out_dir = os.path.join(project_root, "resultados", "resultados_autoencoder")
l_dim = kwargs.get('latent_dim', 8)
model_candidates = [
    os.path.join(out_dir, f"autoencoder_emg_{l_dim}d.pth"),
    os.path.join(out_dir, "autoencoder_campeon.pth"),
    os.path.join(out_dir, "autoencoder_emg.pth"),
]
model_path = next((m for m in model_candidates if os.path.exists(m)), None)
if not model_path:
    print("> ERROR: No se encontró ningún modelo (.pth) entrenado.")
    sys.exit(1)

import deep_learning.plot_latent_space as pls
pls.plot_latent_space(csv_file, model_path, latent_dim=l_dim)
"""
    self._launch_bridge_script("plotear_latente", "PLOTEO ESPACIO LATENTE", kwargs, template)

  def run_autoencoder_grid_search_nativo(self):
    kwargs = self.tab_dl_ml.get_autoencoder_kwargs()
    template = """
import json
import sys
import os

project_root = r'{ROOT_PROJECT_DIR}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'{TEMP_JSON}', 'r') as f:
    kwargs = json.load(f)

import deep_learning.grid_search_autoencoder as gsa
df_res, campeon = gsa.run_grid_search(epochs=min(kwargs.get('epochs', 80), 80))
"""
    self._launch_bridge_script("grid_search_ae", "GRID SEARCH AUTOENCODER (36 COMB.)", kwargs, template)

  def run_autoencoder_decodificador_nativo(self):
    rutas = self.explorer_widget.get_selected_paths()
    carpeta = rutas[0] if rutas else None
    if not carpeta or not os.path.isdir(carpeta):
        from PySide6.QtWidgets import QFileDialog
        root_dir = get_project_root()
        base_dir = os.path.join(root_dir, "base_de_datos_electrodos")
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccione la carpeta de Secuencia Continua", base_dir)
        if not carpeta:
            return

    kwargs = self.tab_dl_ml.get_autoencoder_kwargs()
    carpeta_clean = carpeta.replace("\\\\", "/")
    template = f'''
import json
import sys
import os

project_root = r'{{ROOT_PROJECT_DIR}}'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'{{TEMP_JSON}}', 'r') as f:
    kwargs = json.load(f)

carpeta_secuencia = r"{carpeta_clean}"

out_dir = os.path.join(project_root, "resultados", "resultados_autoencoder")
l_dim = kwargs.get('latent_dim', 8)
model_candidates = [
    os.path.join(out_dir, f"autoencoder_emg_{{l_dim}}d.pth"),
    os.path.join(out_dir, "autoencoder_campeon.pth"),
    os.path.join(out_dir, "autoencoder_emg.pth"),
]
model_path = next((m for m in model_candidates if os.path.exists(m)), None)
if not model_path:
    print("> ERROR: No se encontró ningún modelo (.pth) entrenado.")
    sys.exit(1)

import deep_learning.decodificador_continuo as dc
dc.decodificar_secuencia(
    carpeta_secuencia=carpeta_secuencia,
    modelo_path=model_path,
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    smooth_ms=kwargs.get('smooth_ms', 150),
    notch_q=kwargs.get('notch_q', 2.0),
    use_manual_exclusions=kwargs.get('use_manual_exclusions', True),
    target_length=kwargs.get('target_length', 100)
)
'''
    self._launch_bridge_script("decodificador_continuo", f"DECODIFICADOR CONTINUO ({os.path.basename(carpeta)})", kwargs, template)

  def _sync_electrode_viewer(self):
    """Sincroniza el visor de electrodos con las mediciones seleccionadas en el gestor."""
    paths = self.explorer_widget.get_selected_paths()
    if hasattr(self, 'electrode_viewer'):
      self.electrode_viewer.load_measurements(paths)

  def _create_dock_terminal(self):
    """Panel inferior para logs (Consola Integrada)"""
    self.dock_terminal = QDockWidget("Consola de Orquestador Multihilo", self)
    self.dock_terminal.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
    
    self.log_console = QTextEdit()
    self.log_console.setReadOnly(True)
    self.log_console.setStyleSheet("""
      QTextEdit {
        background-color: #050505;
        color: #00ff00;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 12px;
        border: 1px solid #222;
      }
    """)
    self.dock_terminal.setWidget(self.log_console)
    self.addDockWidget(Qt.BottomDockWidgetArea, self.dock_terminal)

  def _run_analysis(self, interactivo=True):
    """
    Ejecuta la funcionalidad de _run_analysis.

    Args:
      interactivo (Any): Argumento posicional interactivo.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    rutas = self.explorer_widget.get_selected_paths()
    if not rutas:
      self.log_console.append("> ERROR: Por favor selecciona una Fecha/Medición válida en el Gestor de Sesiones.\\n")
      return

    self.log_console.append("\\n> INICIANDO PROCESAMIENTO BATCH MULTIHILO...")
    self.log_console.append(f"> {len(rutas)} Mediciones en cola.")

    # Obtener los nuevos parámetros del formulario PySide6
    kwargs = self.analysis_panel.get_processing_kwargs()
    
    # Extraer canales seleccionados en la UI
    tab_proc = self.analysis_panel.tab_procesamiento
    canales_elegidos = []
    if hasattr(tab_proc, 'checkboxes_canales') and tab_proc.checkboxes_canales:
      canales_elegidos = [c for c, chk in tab_proc.checkboxes_canales.items() if chk.isChecked()]
      
    if not canales_elegidos:
      # Si los checkboxes no estaban inicializados o quedaron vacíos, auto-detectar canales de las mediciones
      canales_detectados = set()
      for r in rutas:
        if os.path.isdir(r):
          for item in os.listdir(r):
            if item.startswith("canal_") and os.path.isdir(os.path.join(r, item)):
              canales_detectados.add(item)
      if canales_detectados:
        canales_elegidos = sorted(list(canales_detectados), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        
    if not canales_elegidos:
      self.log_console.append("> ERROR: Selecciona al menos un canal a procesar (Configuración 1).\n")
      return
      
    # Bloquear botones temporalmente para no apilar llamadas
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(False)
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setText("PROCESANDO...")
    
    import subprocess
    import sys
    import os
    from utils.path_utils import get_project_root, get_database_path
    
    base_dir = get_database_path()
    nombres_medicion = [os.path.relpath(r, base_dir).replace('\\', '/') for r in rutas]
    emg_root = get_project_root()
    
    bridge_script = f"""
import sys
import os
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
sys.path.append(r"{emg_root}")
import analysis.analisis_por_track_integrado as api

mediciones = {nombres_medicion}
base_dir = r"{base_dir}"

try:
  root = tk.Tk()
  root.withdraw()
  dialog = api.ProcessingOptionsDialog(root)
  dialog.populate_channels(base_dir, mediciones)

  # Trasplantar la selección de canales de PySide6 a Tkinter
  canales_elegidos = {canales_elegidos}
  for canal_key, var in dialog.canales_seleccionados.items():
    var.set(canal_key in canales_elegidos)

  # Inyectar los parámetros de nuestra GUI PySide6 a su GUI Tkinter
  dialog.var_mostrar_recortes.set({kwargs['mostrar_recortes']})
  dialog.var_mostrar_senal_cruda.set({kwargs['mostrar_senal_cruda']})
  dialog.var_mostrar_espectrograma.set({kwargs['mostrar_espectrograma']})
  dialog.var_frecuenciamaxima.set("{kwargs.get('frecuenciamaxima', 5000)}")
  dialog.var_notch_filter.set({kwargs['apply_notch_filter']})
  dialog.var_notch_q_factor.set("{kwargs.get('notch_q_factor', 2.0)}")
  dialog.var_mostrar_evolucion.set({kwargs['mostrar_evolucion']})
  dialog.var_evol_t_start.set("{kwargs['evol_t_start']}")
  dialog.var_evol_t_end.set("{kwargs['evol_t_end']}")
  dialog.var_smooth_ms.set("{kwargs['smooth_ms']}")
  dialog.var_tipo_env.set("{kwargs.get('tipo_envolvente', 'media_movil')}")
  dialog.var_highpass_cutoff.set("{kwargs['highpass_cutoff_hz']}")
  dialog.var_lowpass_cutoff.set("{kwargs['lowpass_cutoff_hz']}")
  if hasattr(dialog, 'var_cyberpunk'):
    dialog.var_cyberpunk.set({kwargs.get('tema_cyberpunk', False)})

  excl_list = {kwargs['excluded_windows_list']}
  excl_str = ",".join(map(str, excl_list)) if excl_list else ""
  dialog.var_excluded_windows.set(excl_str)

  print("\\n> Orquestador Tkinter Aislado Inicializado. Ejecutando Rutina original de ProcessingOptionsDialog...")
  # Ejecutar su propia rutina que ya maneja pop-ups, metadatos y curación
  dialog.procesar(interactivo={interactivo})

except Exception as e:
  import traceback
  print("\\n" + "="*50)
  print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL PROCESAMIENTO")
  print("="*50)
  traceback.print_exc()
finally:
  input("\\nPresione ENTER para cerrar esta ventana...")
"""
    script_path = os.path.join(emg_root, "gui_app", "temp_procesar.py")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w", encoding="utf-8") as f:
      f.write(bridge_script)
      
    from PySide6.QtCore import QThread, Signal
    from PySide6.QtGui import QPixmap
    from PySide6.QtCore import Qt
    
    class ProcessRunner(QThread):
      """
      Clase ProcessRunner.

      Representa y gestiona las operaciones relacionadas con ProcessRunner.
      """
      finished_signal = Signal(object)
      def __init__(self, spath):
        """
        Ejecuta la funcionalidad de __init__.

        Args:
          spath (Any): Argumento posicional spath.

        Returns:
          Any: Resultado de la ejecución de la función.
        """
        super().__init__()
        self.spath = spath
        
      def run(self):
        """
        Ejecuta la funcionalidad de run.

        Returns:
          Any: Resultado de la ejecución de la función.
        """
        import subprocess
        import sys
        import shutil
        # Run in new console!
        if sys.platform == "win32":
          p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
          terminals = ['xterm', 'konsole', 'gnome-terminal', 'xfce4-terminal']
          term_cmd = None
          for t in terminals:
              if shutil.which(t):
                  if t == 'xterm': term_cmd = [t, '-e', sys.executable, self.spath]
                  elif t == 'konsole': term_cmd = [t, '--nofork', '-e', sys.executable, self.spath]
                  elif t == 'gnome-terminal': term_cmd = [t, '--wait', '--', sys.executable, self.spath]
                  else: term_cmd = [t, '-e', sys.executable, self.spath]
                  break
          if term_cmd:
              p = subprocess.run(term_cmd)
          else:
              p = subprocess.run([sys.executable, self.spath])
        self.finished_signal.emit(getattr(p, 'returncode', 0))
        
    self.procesador_thread = ProcessRunner(script_path)
    
    def on_procesador_finished(exit_code):
      """
      Ejecuta la funcionalidad de on_procesador_finished.

      Args:
        exit_code (Any): Argumento posicional exit_code.

      Returns:
        Any: Resultado de la ejecución de la función.
      """
      self.analysis_panel.tab_procesamiento.btn_run_procesar.setText(" PROCESAR Y CURAR INDIVIDUALES")
      self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(True)
      self.log_console.append(f"> Tarea de curación finalizada en terminal nativa.")
      
      # Intentar cargar pulses.png de la primera medición seleccionada
      img_path = os.path.join(rutas[0], canales_elegidos[0], "pulses.png")
      if not os.path.exists(img_path):
        img_path = os.path.join(rutas[0], canales_elegidos[0], "avg.png")
        
      if os.path.exists(img_path):
        pix = QPixmap(img_path)
        pix = pix.scaled(self.img_viewer.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_viewer.setPixmap(pix)
      else:
        self.img_viewer.setText("[Proceso Completado: Sin Imágenes Encontradas (pulses.png)]")
        
    self.procesador_thread.finished_signal.connect(on_procesador_finished)
    self.procesador_thread.start()
    
    self.log_console.append(f"> Tarea enviada exitosamente. Abriendo terminal CMD nativa...")

  def _on_analysis_result(self, result):
    """
    Ejecuta la funcionalidad de _on_analysis_result.

    Args:
      result (Any): Argumento posicional result.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if isinstance(result, list) and len(result) > 0:
      last_img = result[-1] # Mostrar la última imagen generada (la del canal_N o paciente final)
      pix = QPixmap(last_img)
      self.img_viewer.setPixmap(pix)
    else:
      self.img_viewer.setText("[Proceso Completado: Sin Imágenes Encontradas]")

  def _on_analysis_finished(self):
    """
    Ejecuta la funcionalidad de _on_analysis_finished.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(True)
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setText("PROCESAR")
    print("\n> BATCH PROCESSING FINALIZADO CON ÉXITO.")
    
  def _on_analysis_error(self, err_tuple):
    """
    Ejecuta la funcionalidad de _on_analysis_error.

    Args:
      err_tuple (Any): Argumento posicional err_tuple.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    exctype, value, tb = err_tuple
    self.img_viewer.setText("[Error en el Procesamiento]")
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(True)
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setText("PROCESAR")
    print("\n> ERROR DURANTE EL PROCESAMIENTO:")
    print(str(value))

  def _setup_styles(self):
    # Aplicamos estrictamente la estética del web_viewer (Negro, Rojo, Courier)
    """
    Ejecuta la funcionalidad de _setup_styles.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self.setStyleSheet("""
      QMainWindow, QWidget {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Courier New', Courier, monospace;
      }
      QDockWidget {
        font-weight: bold;
        color: #ffffff;
      }
      QDockWidget::title {
        background: #111111;
        text-align: center;
        padding: 6px;
        border: 1px solid #FF0000;
        color: #FF0000;
        font-size: 14px;
        letter-spacing: 1px;
      }
      QTabWidget::pane {
        border: 2px solid #FF0000;
        background-color: #0a0a0a;
      }
      QTabBar::tab {
        background-color: transparent;
        color: #888888;
        border: 1px solid #333333;
        padding: 8px 20px;
        font-weight: bold;
        font-size: 14px;
      }
      QTabBar::tab:hover {
        color: #ffffff;
        border-color: #777777;
      }
      QTabBar::tab:selected {
        background-color: #FF0000;
        color: #000000;
        border-color: #FF0000;
      }
      QTextEdit {
        background-color: #0c0c0c;
        color: #00ff00;
        border: 1px solid #333333;
      }
      QLabel {
        background-color: transparent;
      }
    """)

  def _launch_dl_ml_script(self, script_rel_path):
    rutas = self.explorer_widget.get_selected_paths()
    if not rutas:
      self.log_console.append(f"> ERROR: Selecciona al menos una medición para lanzar {script_rel_path}\n")
      return
    
    import os
    import sys
    import subprocess
    
    if getattr(sys, 'frozen', False):
      cmd = [sys.executable, script_rel_path] + rutas
    else:
      root_dir = get_project_root()
      script_abs_path = os.path.join(root_dir, script_rel_path.replace("/", os.sep))
      
      if not os.path.exists(script_abs_path):
        self.log_console.append(f"> ERROR: Script no encontrado: {script_abs_path}\n")
        return
      cmd = [sys.executable, script_abs_path] + rutas
      
    self.log_console.append(f"> INICIANDO SCRIPT: {script_rel_path}")
    for r in rutas:
      self.log_console.append(f"  - {r}")
    
    if os.name == 'nt':
        terminal_cmd = ['cmd.exe', '/c', 'start', 'cmd.exe', '/k'] + cmd
    else:
        # En Linux buscamos la terminal instalada iterando sobre las más populares
        import shlex
        import shutil
        cmd_str = " ".join(shlex.quote(c) for c in cmd)
        bash_cmd = f"{cmd_str}; echo ''; read -p 'Presiona Enter para cerrar la terminal...'"
        
        terminales = ['konsole', 'gnome-terminal', 'xfce4-terminal', 'mate-terminal', 'lxterminal', 'x-terminal-emulator', 'xterm']
        terminal_elegida = None
        for term in terminales:
            if shutil.which(term):
                terminal_elegida = term
                break
                
        if terminal_elegida:
            if terminal_elegida == 'gnome-terminal':
                terminal_cmd = [terminal_elegida, '--', 'bash', '-c', bash_cmd]
            else:
                terminal_cmd = [terminal_elegida, '-e', 'bash', '-c', bash_cmd]
        else:
            # Fallback extremo si no encuentra absolutamente nada
            terminal_cmd = ['bash', '-c', bash_cmd]
        
    try:
      subprocess.Popen(terminal_cmd)
      self.log_console.append("> Script lanzado exitosamente en segundo plano.\n")
    except Exception as e:
      self.log_console.append(f"> ERROR al lanzar script: {e}\n")

def main():
  from PySide6.QtWidgets import QSplashScreen
  from PySide6.QtGui import QPixmap, QColor
  from pathlib import Path
  import time
  import ctypes
  
  # --- Solución para el ícono en la barra de tareas de Windows ---
  if os.name == 'nt':
    try:
      myappid = 'nandu.lsd.emg_studio.5.0.0'
      ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
      pass
  
  # El ejecutable pone el icono en la raíz (sys._MEIPASS), no en gui_app/
  if getattr(sys, 'frozen', False):
    icon_path = os.path.join(sys._MEIPASS, 'icono.ico')
  else:
    icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'icono.ico')
    
  if os.path.exists(icon_path):
    app.setWindowIcon(QIcon(icon_path))
  # -----------------------------------------------------------
  
  # 1. Buscar logo para el Splash Screen
  logo_path = None
  try:
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gui_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    assets_dir = gui_dir / "assets"
    pictures_dir = Path.home() / "Pictures"
    
    search_dirs = [assets_dir, gui_dir, root_dir, Path(get_project_root()), pictures_dir]
    if hasattr(sys, '_MEIPASS'):
      search_dirs.insert(0, Path(sys._MEIPASS))
    for search_dir in search_dirs:
      if search_dir.exists():
        for filename in os.listdir(search_dir):
          if filename.lower().startswith("logo") and filename.lower().endswith((".png", ".jpg", ".jpeg")):
            logo_path = str(search_dir / filename)
            break
      if logo_path:
        break
  except Exception:
    pass

  splash = None
  if logo_path and os.path.exists(logo_path):
    pixmap = QPixmap(logo_path)
    if pixmap.width() > 800:
      pixmap = pixmap.scaledToWidth(800, Qt.SmoothTransformation)
    from PySide6.QtWidgets import QProgressBar
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    
    splash_progress = QProgressBar(splash)
    splash_progress.setGeometry(15, pixmap.height() - 28, pixmap.width() - 30, 16)
    splash_progress.setStyleSheet("QProgressBar { border: 1px solid #555555; border-radius: 4px; text-align: center; color: white; font-weight: bold; background-color: #111111; } QProgressBar::chunk { background-color: #00ffcc; border-radius: 3px; }")
    splash_progress.setValue(35)
    splash_progress.show()
    
    splash.show()
    splash.showMessage("Iniciando Ñandú LSD EMG Analytics...", Qt.AlignBottom | Qt.AlignCenter, QColor("white"))
    app.processEvents()
    time.sleep(0.8)
    
  if qdarkstyle:
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    
  if splash:
    if 'splash_progress' in locals() and splash_progress:
      splash_progress.setValue(75)
    splash.showMessage("Cargando módulos y base de datos...", Qt.AlignBottom | Qt.AlignCenter, QColor("white"))
    app.processEvents()
    
  window = ReaperStyleHub()
  # Modificar stylesheet base DESPUÉS del qdarkstyle
  window._setup_styles()
  
  if splash:
    if 'splash_progress' in locals() and splash_progress:
      splash_progress.setValue(100)
      app.processEvents()
    splash.finish(window)
    
  window.show()
  sys.exit(app.exec())

if __name__ == "__main__":
  main()
