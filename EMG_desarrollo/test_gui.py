import sys
import traceback
from PySide6.QtWidgets import QApplication

try:
    from analysis.plotter_calibrado import flujo_principal
    sys.argv = ["analysis/plotter_calibrado.py", "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos/2026-06-10/A_Prueba1_Sujeto1"]
    flujo_principal()
except Exception as e:
    traceback.print_exc()
