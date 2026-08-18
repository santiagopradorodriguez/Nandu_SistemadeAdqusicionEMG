import sys
import os
sys.path.append(os.path.abspath('EMG_desarrollo'))
from gui_app.views.ui_analysis import AnalysisPanel
from PySide6.QtWidgets import QApplication

app = QApplication([])
panel = AnalysisPanel()
kwargs = panel.get_pca_umap_kwargs()

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

mediciones = []  # No carpetas por ahora
try:
    generador.ejecutar_procesamiento(
        mediciones,
        "/tmp",
        **kwargs
    )
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
