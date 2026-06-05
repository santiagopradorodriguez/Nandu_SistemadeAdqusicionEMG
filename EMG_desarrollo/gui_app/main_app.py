# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Punto de entrada principal para la aplicación gráfica (GUI).
# ==============================================================================

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
if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
  script_name = sys.argv[1]
  # Remover el nombre del script de los argumentos
  sys.argv = [script_name] + sys.argv[2:]
  
  # Transformar 'acquisition/manual_daq.py' -> 'acquisition.manual_daq'
  module_name = script_name.replace('\\', '/').replace('.py', '').replace('/', '.')
  
  try:
    # Importaciones explícitas para forzar a PyInstaller a empaquetarlas en el PYZ.
    # Si se usa importlib dinámico, PyInstaller puede ignorarlas.
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
    elif module_name == 'analysis.reproductor_canal3':
      import analysis.reproductor_canal3 as module
    elif module_name == 'utils.editor_mediciones':
      import utils.editor_mediciones as module
    elif module_name == 'utils.actualizar_metadata':
      import utils.actualizar_metadata as module
    elif module_name == 'utils.migrar_mediciones_por_fecha':
      import utils.migrar_mediciones_por_fecha as module
    else:
      import importlib
      module = importlib.import_module(module_name)
      
    if hasattr(module, 'main'):
      module.main()
    else:
      print(f"Error: {module_name} no tiene una función main().")
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
import matplotlib
matplotlib.use('TkAgg') # Forzar TkAgg para que las ventanas de curación de Matplotlib pausen el script correctamente en PySide6
from PySide6.QtWidgets import (
  QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
  QDockWidget, QTextEdit, QLabel, QTreeView, QTabWidget,
  QToolBar, QPushButton, QSizePolicy, QMessageBox
)
from PySide6.QtCore import Qt, QThreadPool, QSize
from PySide6.QtGui import QFont, QColor, QTextCursor, QAction, QPixmap, QIcon

app = QApplication.instance()
if not app:
  app = QApplication(sys.argv)



from core.threads import EmittingStream, Worker
from views.session_explorer import SessionExplorer
from views.ui_analysis import AnalysisPanel

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
    """
    Ejecuta la funcionalidad de __init__.

    Args:
      text (Any): Argumento posicional text.
      parent (Any): Argumento posicional parent.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    super().__init__(text, parent)
    self.setAlignment(Qt.AlignCenter)
    self._pixmap = None
    self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
    self.setMinimumSize(100, 100)

  def setPixmap(self, pixmap):
    """
    Ejecuta la funcionalidad de setPixmap.

    Args:
      pixmap (Any): Argumento posicional pixmap.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self._pixmap = pixmap
    self._update_pixmap()

  def resizeEvent(self, event):
    """
    Ejecuta la funcionalidad de resizeEvent.

    Args:
      event (Any): Argumento posicional event.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    self._update_pixmap()
    super().resizeEvent(event)

  def _update_pixmap(self):
    """
    Ejecuta la funcionalidad de _update_pixmap.

    Returns:
      Any: Resultado de la ejecución de la función.
    """
    if self._pixmap is not None and not self._pixmap.isNull():
      if self.width() > 0 and self.height() > 0:
        super().setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

class PatronMuscularViewerWidget(QWidget):
  def __init__(self):
    super().__init__()
    layout = QVBoxLayout(self)
    layout.setContentsMargins(0, 0, 0, 0)
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
          subprocess.Popen([sys.executable, script_name] + args)
      except Exception as e:
        QMessageBox.critical(self, "Error Crítico", f"Error al abrir {script_name}:\n{e}")
      return

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root_dir, script_name)
    
    if not os.path.exists(script_path):
      QMessageBox.critical(self, "Error", f"No se encontró el script: {script_name} en {script_path}")
      return
      
    try:
      if sys.platform == "win32":
        subprocess.Popen([sys.executable, script_path] + args, creationflags=subprocess.CREATE_NEW_CONSOLE)
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
    self.addToolBar(Qt.TopToolBarArea, toolbar)
    
    utils = [
      ("⚙ Configuración General", "_internal_config"),
      ("Instrucciones y Créditos", "instrucciones_uso.py"),
      ("Metrónomo", "acquisition/metronomo_visual.py"),
      ("Entrenamiento AutoForge", "acquisition/modulo_de_entrenamiento.py"),
      ("Editar Medición", "utils/editor_mediciones.py"),
      ("Extraer Datos ML", "analysis/feature_extractor.py"),
      ("Graficador", "analysis/plotter_calibrado.py"),
      ("Análisis Correlación", "analysis/correlaciondeseñales.py"),
      ("Reproductor de Audios", "analysis/reproductor_canal3.py")
    ]
    
    for name, script in utils:
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
      toolbar.addAction(action)

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
    
    # --- NUEVO: Cargar Logo del Programa ---
    logo_path = None
    try:
      # Buscar en varias ubicaciones posibles
      root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
      gui_dir = Path(os.path.dirname(os.path.abspath(__file__)))
      assets_dir = gui_dir / "assets"
      pictures_dir = Path.home() / "Pictures"
      
      search_dirs = [assets_dir, gui_dir, root_dir, pictures_dir]
      
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
      pix = pix.scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
      lbl_logo.setPixmap(pix)
      lbl_logo.setStyleSheet("padding: 10px; background-color: #050505; border-radius: 8px; border: 1px solid #222;")
    else:
      lbl_logo.setText("<h2>[ÑANDÚ LSD LOGO]</h2><p style='color:#888;'>Coloca un archivo 'logo.png' en la carpeta Imágenes</p>")
      lbl_logo.setStyleSheet("color: #FF0000; background-color: #111; border: 1px dashed #FF4444; padding: 20px; border-radius: 8px;")
    
    vbox_info.addWidget(lbl_logo)
    
    md_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "archivos_md", "documentacion_matematica.md"))
    file_url = f"file:///{md_path.replace(chr(92), '/')}"
    
    html_intro = f"""
    <div style='padding: 20px;'>
      <h2 style='color:#00ffff;'> Plataforma de Investigación EMG v4.0</h2>
      <p>Bienvenido al hub centralizado para adquisición en tiempo real, curación y análisis comparativo de señales electromiográficas.</p>
      
      <h3 style='color:#00ffaa;'>Novedades Actualización v4.0 (Secuencia Continua & AutoForge):</h3>
      <ul>
        <li><b>AutoForge Secuencia Continua:</b> Captura el diccionario entero de forma cíclica en un solo click, autogenerando las etiquetas correctas de Machine Learning (valid_words) para cada pulso en los metadatos.</li>
        <li><b>Cálculo de Ruido y SNR Dinámico:</b> Análisis automático del ruido de fondo previo a cada estímulo, con offset centrado en base al primer octavo del pulso promedio para lograr gráficos de overlay precisos.</li>
        <li><b>Sincronización Perfecta:</b> La geometría de búsqueda de pulsos se centra dinámicamente usando como referencia la ventana exacta del beat del metrónomo.</li>
        <li><b>Framework Moderno PySide6:</b> Estabilidad extrema, estética Cyberpunk, colores persistentes y prevención de caídas de UI frente a errores internos.</li>
      </ul>

      <h3 style='color:#ffaa00;'> Instrucciones Rápidas:</h3>
      <ul>
        <li><b>Adquisición:</b> Haz clic en el botón rojo de la derecha para grabar nuevas mediciones o correr el simulador.</li>
        <li><b>Curación Individual:</b> Selecciona mediciones en el árbol izquierdo, marca los canales deseados, configura los filtros Notch/Pasabanda y haz clic en Procesar en la pestaña 'Análisis'. Las gráficas aparecerán automáticamente.</li>
      </ul>
      
      <h3 style='color:#00ff00;'> Fundamento Teórico:</h3>
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
    
    btn_autoforge = QPushButton("AUTOGRABADO")
    btn_autoforge.setStyleSheet("""
      QPushButton {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 20px; 
        font-weight: 900; 
        background-color: #1a001a; 
        color: #ff00ff; 
        padding: 30px 15px;
        border-radius: 4px;
        border: 2px solid #ff00ff;
        border-right: 8px solid #00ffff;
        border-bottom: 8px solid #00ffff;
        margin-top: 20px;
      }
      QPushButton:hover {
        background-color: #ff00ff;
        color: #1a001a;
        border: 2px solid #00ffff;
        border-right: 8px solid #00ffff;
        border-bottom: 8px solid #00ffff;
        margin-top: 20px;
      }
      QPushButton:pressed {
        background-color: #00ffff;
        color: #000000;
        border: 2px solid #ffffff;
        margin-top: 20px;
      }
    """)
    btn_autoforge.clicked.connect(lambda: self._launch_external("acquisition/autoforge_daq.py"))
    vbox_btn.addWidget(btn_autoforge)

    btn_autoforge_staging = QPushButton("⚡ AUTOGRABADO\nSTAGING")
    btn_autoforge_staging.setStyleSheet("""
      QPushButton {
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 16px; 
        font-weight: 900; 
        background-color: #1a0d00; 
        color: #FF8800; 
        padding: 18px 15px;
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
    
    # --- TAB 2: ANÁLISIS ---
    self.tab_analysis = QWidget()
    lyt_analysis = QHBoxLayout(self.tab_analysis)
    self.analysis_panel = AnalysisPanel()
    self.analysis_panel.tab_procesamiento.btn_run_procesar.clicked.connect(lambda: self._run_analysis(interactivo=True))
    self.analysis_panel.tab_procesamiento.btn_run_rapido.clicked.connect(lambda: self._run_analysis(interactivo=False))
    self.analysis_panel.tab_comparativo.btn_run_comparativo.clicked.connect(self.run_analisis_comparativo_nativo)
    lyt_analysis.addWidget(self.analysis_panel, stretch=1)
    
    # Visor de Imágenes Integrado
    self.img_viewer = ImageLabel("[Visor de Resultados Integrado]\nAquí aparecerán los gráficos avg.png / pulses.png generados")
    self.img_viewer.setStyleSheet("background-color: #0c0c0c; border: 1px solid #333; color: #555;")
    self.img_viewer.setMinimumWidth(500)
    lyt_analysis.addWidget(self.img_viewer, stretch=2)
    
    self.tabs.addTab(self.tab_analysis, "2. ANÁLISIS Y EXTRACCIÓN")
    
    # --- TAB 3: VISUALIZACIÓN NATIVA ---
    self.tab_view = QWidget()
    lyt_view = QVBoxLayout(self.tab_view)
    
    # Sub-pestañas para visualización
    self.tabs_viz = QTabWidget()
    
    # Sub-pestaña: Visor CSV (Natívo PyQtGraph)
    from views.csv_viewer_widget import CsvViewerWidget
    self.csv_viewer = CsvViewerWidget()
    self.tabs_viz.addTab(self.csv_viewer, " Explorador de Señales (CSV)")
    
    # Sub-pestaña: Visor de Gráficos Calibrados
    from views.calibrated_viewer_widget import CalibratedViewerWidget
    self.calibrated_viewer = CalibratedViewerWidget()
    self.tabs_viz.addTab(self.calibrated_viewer, " Historial Gráficos Musculares")
    
    # Sub-pestaña: Electrode Viewer (Nativo PySide6)
    from views.electrode_viewer_widget import ElectrodeViewerWidget
    self.electrode_viewer = ElectrodeViewerWidget()
    self.electrode_viewer.btn_refresh.clicked.connect(self._sync_electrode_viewer)
    self.tabs_viz.addTab(self.electrode_viewer, " Visor de Electrodos (Grilla)")
    
    # Sub-pestaña: Visor de Patrón Muscular
    self.patron_viewer = PatronMuscularViewerWidget()
    self.tabs_viz.addTab(self.patron_viewer, " Historial Patrón Muscular")
    
    lyt_view.addWidget(self.tabs_viz)
    self.tabs.addTab(self.tab_view, "3. VISUALIZACIÓN")
    
    # --- TAB 4: HISTORIAL DE COMPARATIVAS ---
    from views.comparative_explorer_widget import ComparativeViewerWidget
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comparative_path = os.path.join(root_dir, "analisis_comparativos")
    
    self.comparative_viewer = ComparativeViewerWidget(root_path=comparative_path)
    self.tabs.addTab(self.comparative_viewer, "4. HISTORIAL DE COMPARATIVAS")

  def _create_dock_explorer(self):
    """Panel tipo 'Media Explorer' o 'Gestor de Sesiones'"""
    self.dock_explorer = QDockWidget("Gestor de Sesiones", self)
    self.dock_explorer.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
    
    # Le pasamos la ruta absoluta apuntando directo a base_de_datos_electrodos
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(root_dir, "base_de_datos_electrodos")
    
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
          chk = QCheckBox(canal)
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
        p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.finished_signal.emit(p.returncode)
        
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
    if hasattr(tab_proc, 'checkboxes_canales'):
      canales_elegidos = [c for c, chk in tab_proc.checkboxes_canales.items() if chk.isChecked()]
      
    if not canales_elegidos:
      self.log_console.append("> ERROR: Selecciona al menos un canal a procesar (Configuración 1).\n")
      return
      
    # Bloquear botones temporalmente para no apilar llamadas
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setEnabled(False)
    self.analysis_panel.tab_procesamiento.btn_run_procesar.setText("PROCESANDO...")
    
    import subprocess
    import sys
    import os
    
    # Rutas relativas para compatibilidad exacta con su Tkinter (Ej: 2026-05-22/Medicion)
    base_dir = os.path.dirname(os.path.dirname(rutas[0]))
    nombres_medicion = [f"{os.path.basename(os.path.dirname(r))}/{os.path.basename(r)}" for r in rutas]
    emg_root = os.path.dirname(base_dir)
    
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
  dialog.var_notch_filter.set({kwargs['apply_notch_filter']})
  dialog.var_mostrar_evolucion.set({kwargs['mostrar_evolucion']})
  dialog.var_evol_t_start.set("{kwargs['evol_t_start']}")
  dialog.var_evol_t_end.set("{kwargs['evol_t_end']}")
  dialog.var_smooth_ms.set("{kwargs['smooth_ms']}")
  dialog.var_tipo_env.set("{kwargs.get('tipo_envolvente', 'media_movil')}")
  dialog.var_highpass_cutoff.set("{kwargs['highpass_cutoff_hz']}")
  dialog.var_lowpass_cutoff.set("{kwargs['lowpass_cutoff_hz']}")

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
        # Run in new console!
        p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
        self.finished_signal.emit(p.returncode)
        
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
    
    search_dirs = [assets_dir, gui_dir, root_dir, pictures_dir]
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
    splash = QSplashScreen(pixmap, Qt.WindowStaysOnTopHint)
    splash.show()
    splash.showMessage("Iniciando Ñandú LSD EMG Analytics...", Qt.AlignBottom | Qt.AlignCenter, QColor("white"))
    app.processEvents()
    time.sleep(1.0) # Delay para asegurar que el usuario vea el logo (estilo software pro)
    
  if qdarkstyle:
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyside6'))
    
  if splash:
    splash.showMessage("Cargando módulos y base de datos...", Qt.AlignBottom | Qt.AlignCenter, QColor("white"))
    app.processEvents()
    
  window = ReaperStyleHub()
  # Modificar stylesheet base DESPUÉS del qdarkstyle
  window._setup_styles()
  
  if splash:
    splash.finish(window)
    
  window.show()
  sys.exit(app.exec())

if __name__ == "__main__":
  main()
