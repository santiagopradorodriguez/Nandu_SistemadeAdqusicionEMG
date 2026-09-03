import sys
import os

# Simulando lo que hace main_app.py
args = ["/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos/2026-06-10/Medicion1"]
root_dir = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo"
BASE_DIR = os.path.join(root_dir, "base_de_datos_electrodos")

print("Argumentos recibidos:", args)
pre_selected = [os.path.relpath(p, BASE_DIR).replace('\\', '/') for p in args]
print("Paths relativos generados:", pre_selected)
