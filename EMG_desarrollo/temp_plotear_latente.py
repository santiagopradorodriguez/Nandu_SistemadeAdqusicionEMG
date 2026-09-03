
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
dl_dir = os.path.join(project_root, "deep_learning")
if dl_dir not in sys.path:
    sys.path.insert(0, dl_dir)

with open(r'/tmp/tmp5c9zjhp8.json', 'r') as f:
    kwargs = json.load(f)

csv_candidates = [
    os.path.join(project_root, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "resultados", "resultados_pca_umap", "caracteristicas_exportadas.csv"),
]
csv_file = next((c for c in csv_candidates if os.path.exists(c)), None)
if not csv_file:
    print("> ERROR: No se encontró 'caracteristicas_exportadas.csv'.")
    sys.exit(1)

out_dir = os.path.join(project_root, "resultados", "resultados_autoencoder")
l_dim = kwargs.get('latent_dim', 8)
model_candidates = [
    os.path.join(out_dir, f"autoencoder_emg_{l_dim}d.pth"),
    os.path.join(out_dir, "autoencoder_campeon.pth"),
    os.path.join(out_dir, "autoencoder_emg.pth"),
]
model_path = next((m for m in model_candidates if os.path.exists(m)), None)
if not model_path:
    print("> ERROR: No se encontró ningún modelo (.pth) entrenado.")
    sys.exit(1)

import deep_learning.plot_latent_space as pls
pls.plot_latent_space(
    csv_file, 
    model_path, 
    latent_dim=l_dim,
    train_sessions=kwargs.get('train_sessions', []),
    test_sessions=kwargs.get('test_sessions', [])
)
