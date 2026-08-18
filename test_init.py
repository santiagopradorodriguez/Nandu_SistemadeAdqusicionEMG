import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'EMG_desarrollo'))

from PySide6.QtWidgets import QApplication

app = QApplication.instance() or QApplication(sys.argv)

try:
    from EMG_desarrollo.gui_app.main_app import ReaperStyleHub
    hub = ReaperStyleHub()
    
    # Try calling kwargs methods
    dl_panel = hub.tab_dl_ml
    dl_panel.get_pca_kwargs()
    dl_panel.get_umap_kwargs()
    dl_panel.get_umap_supervisado_kwargs()
    dl_panel.get_discrete_kwargs()
    dl_panel.get_training_kwargs()
    
    analysis_panel = hub.analysis_panel
    analysis_panel.get_comparative_kwargs()
    
    with open('test_init_result.txt', 'w') as f:
        f.write("SUCCESS\n")
except Exception as e:
    import traceback
    with open('test_init_result.txt', 'w') as f:
        f.write("ERROR:\n" + traceback.format_exc() + "\n")

import sys
sys.exit(0)
