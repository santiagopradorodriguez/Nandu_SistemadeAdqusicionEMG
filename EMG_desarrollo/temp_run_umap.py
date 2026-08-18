
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'/tmp/tmp0kkibaxj.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas', '2026-07-10/A_T3_Lucas', '2026-07-10/A_T4_Lucas', '2026-07-10/A_T5_Lucas', '2026-07-10/A_T6_Lucas', '2026-07-10/A_T7_Lucas', '2026-07-10/E_T1_Lucas', '2026-07-10/E_T2_Lucas', '2026-07-10/E_T3_Lucas', '2026-07-10/E_T4_Lucas', '2026-07-10/E_T5_Lucas', '2026-07-10/E_T6_Lucas', '2026-07-10/E_T7_Lucas', '2026-07-10/I_T1_Lucas', '2026-07-10/I_T2_Lucas', '2026-07-10/I_T3_Lucas', '2026-07-10/I_T4_Lucas', '2026-07-10/I_T5_Lucas', '2026-07-10/I_T6_Lucas', '2026-07-10/I_T7_Lucas', '2026-07-10/O_T1_Lucas', '2026-07-10/O_T2_Lucas', '2026-07-10/O_T3_Lucas', '2026-07-10/O_T4_Lucas', '2026-07-10/O_T5_Lucas', '2026-07-10/O_T6_Lucas', '2026-07-10/O_T7_Lucas', '2026-07-10/U_T1_Lucas', '2026-07-10/U_T2_Lucas', '2026-07-10/U_T3_Lucas', '2026-07-10/U_T4_Lucas', '2026-07-10/U_T5_Lucas', '2026-07-10/U_T6_Lucas', '2026-07-10/U_T7_Lucas']
base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

import deep_learning.pca_umap_clustering.generador_pca_umap as generador
generador.ejecutar_procesamiento(mediciones=mediciones, base_dir=base_dir, **kwargs)
