
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
dl_dir = os.path.join(project_root, "deep_learning")
if dl_dir not in sys.path:
    sys.path.insert(0, dl_dir)

with open(r'/tmp/tmpp9w9od8p.json', 'r') as f:
    kwargs = json.load(f)

csv_candidates = [
    os.path.join(project_root, "resultados", "resultados_pca_tensorial", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "resultados", "resultados_pca_umap", "caracteristicas_exportadas.csv"),
    os.path.join(project_root, "deep_learning", "caracteristicas_exportadas.csv"),
]
csv_file = None
for c in csv_candidates:
    if os.path.exists(c):
        csv_file = c
        break

if not csv_file:
    print("> ERROR: No se encontró 'caracteristicas_exportadas.csv'. Ejecuta primero '1. EXTRAER DATASET'.")
    sys.exit(1)

import deep_learning.train_autoencoder as ta
ta.train_autoencoder(
    csv_path=csv_file,
    epochs=kwargs.get('epochs', 80),
    batch_size=kwargs.get('batch_size', 16),
    latent_dim=kwargs.get('latent_dim', 8),
    kernel_size=kwargs.get('kernel_size', 5),
    force_epochs=kwargs.get('force_epochs', False),
    alpha=kwargs.get('alpha_loss', 0.5),
    verbose=True,
    train_sessions=kwargs.get('train_sessions', []),
    test_sessions=kwargs.get('test_sessions', [])
)
