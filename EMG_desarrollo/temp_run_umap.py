
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'/tmp/tmp185qzd4s.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-09-16/A_Serie1_Candela', '2026-09-16/A_Serie2_Candela', '2026-09-16/A_Serie3_Candela', '2026-09-16/A_Serie4_Candela', '2026-09-16/E_Serie1_Candela', '2026-09-16/E_Serie2_Candela', '2026-09-16/E_Serie3_Candela', '2026-09-16/E_Serie4_Candela', '2026-09-16/I_Serie1_Candela', '2026-09-16/I_Serie2_Candela', '2026-09-16/I_Serie3_Candela', '2026-09-16/I_Serie4_Candela', '2026-09-16/O_Serie1_Candela', '2026-09-16/O_Serie2_Candela', '2026-09-16/O_Serie3_Candela', '2026-09-16/O_Serie4_Candela', '2026-09-16/U_Serie1_Candela', '2026-09-16/U_Serie2_Candela', '2026-09-16/U_Serie3_Candela', '2026-09-16/U_Serie4_Candela']
base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

pca_umap_dir = os.path.join(project_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", "ÑÑÑ")
os.makedirs(pca_umap_dir, exist_ok=True)

with open(os.path.join(pca_umap_dir, "parametros.json"), 'w') as f:
    json.dump(kwargs, f, indent=4)

generador.ejecutar_procesamiento(mediciones=mediciones, base_dir=base_dir, out_dir=pca_umap_dir, **kwargs)
