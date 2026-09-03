import sys
import traceback
from PySide6.QtWidgets import QApplication

try:
    from analysis.plotter_calibrado import plotear_medicion_secuencial
    config = {
        "notch": True,
        "bandpass": True,
        "tipo_env": "ninguna",
        "start_time": 0,
        "end_time": 10,
        "graficar_fft": False,
        "tema_oscuro": True
    }
    app = QApplication(sys.argv)
    print("Testing plotear_medicion_secuencial...")
    plotear_medicion_secuencial("2026-06-10/A_Prueba1_Sujeto1", config, {}, mostrar_plot=False)
    print("DONE")
except Exception as e:
    traceback.print_exc()
