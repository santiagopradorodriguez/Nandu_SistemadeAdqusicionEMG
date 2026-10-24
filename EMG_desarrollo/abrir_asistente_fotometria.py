#!/usr/bin/env python3
import sys
import os

# Usar el framework Qt nativo de NANDU LSD (PySide6)
from PySide6 import QtWidgets, QtCore

# Asegurar path al modulo
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from EMG_desarrollo.acquisition.fotometria_webcam import WebcamPhotometryManager

def main():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    manager = WebcamPhotometryManager(parent=None, camera_index=0)
    
    # Canales sEMG facial de NANDU LSD
    nombres_canales = [
        "Masetero Der",
        "Cigomatico Der",
        "Depresor Anguli Oris",
        "Orbicularis Oris",
    ]
    indices_canales = [0, 1, 2, 3]

    print("[INFO] Abriendo Asistente de Posicionamiento Facial 3D y Fotometria...")
    print("[INFO] Framework Qt: PySide6")
    print("[INFO] Dispositivo: /dev/video0")
    print("[INFO] Canales configurados:", nombres_canales)
    
    ok = manager.open_alignment_dialog(
        parent_widget=None,
        nombres_musculos=nombres_canales,
        active_channel_indices=indices_canales,
        sujeto="Santiago",
        prueba="Calibracion_sEMG",
    )

    if ok:
        print("[OK] Calibracion confirmada por el usuario.")
        calib = manager.calibracion_inicial_data
        if calib:
            n_electrodos = len(calib.get("electrodos", []))
            n_pares = len(calib.get("pares_diferenciales", []))
            print(f"[OK] Total electrodos fisicos triangulados: {n_electrodos}")
            print(f"[OK] Total pares diferenciales bipolares: {n_pares}")
    else:
        print("[AVISO] El usuario cerro o cancelo el asistente.")

    manager.stop_worker()

if __name__ == "__main__":
    main()
