
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
dl_dir = os.path.join(project_root, "deep_learning")
if dl_dir not in sys.path:
    sys.path.insert(0, dl_dir)

with open(r'/tmp/tmpdvp1v1u2.json', 'r') as f:
    kwargs = json.load(f)

user_dim = kwargs.get('latent_dim', 2)
dims_to_test = sorted(list(set([2, 4, 8, 16, user_dim])))

import deep_learning.grid_search_autoencoder as gsa
df_res, campeon = gsa.run_grid_search(
    epochs=min(kwargs.get('epochs', 80), 80),
    latent_dims=dims_to_test,
    train_sessions=kwargs.get('train_sessions', []),
    test_sessions=kwargs.get('test_sessions', [])
)
