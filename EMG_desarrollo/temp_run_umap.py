
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'/tmp/tmpf2uyuii3.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-08-21/A_Prueba1silicona_petra', '2026-08-21/A_silicona_aeiou_petra', '2026-08-21/E_Prueba1silicona_petra', '2026-08-21/E_silicona_aeiou_petra', '2026-08-21/I_Prueba1silicona_petra', '2026-08-21/I_silicona_aeiou_petra', '2026-08-21/O_Prueba1silicona_petra', '2026-08-21/O_silicona_aeiou_petra', '2026-08-21/U_Prueba1silicona_petra', '2026-08-21/U_silicona_aeiou_petra']
base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

pca_umap_dir = os.path.join(project_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", "umpa_petra")
os.makedirs(pca_umap_dir, exist_ok=True)

with open(os.path.join(pca_umap_dir, "parametros.json"), 'w') as f:
    json.dump(kwargs, f, indent=4)

generador.ejecutar_procesamiento(mediciones=mediciones, base_dir=base_dir, out_dir=pca_umap_dir, **kwargs)
