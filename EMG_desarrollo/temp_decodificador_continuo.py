
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)
dl_dir = os.path.join(project_root, "deep_learning")
if dl_dir not in sys.path:
    sys.path.insert(0, dl_dir)

with open(r'/tmp/tmpb_y08t7g.json', 'r') as f:
    kwargs = json.load(f)

carpeta_secuencia = r"/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos/2026-06-22/SecuenciaContinua_Prueba5_SANTI"

out_dir = os.path.join(project_root, "resultados", "resultados_autoencoder")
l_dim = kwargs.get('latent_dim', 2)
model_candidates = [
    os.path.join(out_dir, f"autoencoder_emg_{l_dim}d.pth"),
    os.path.join(out_dir, "autoencoder_campeon.pth"),
    os.path.join(out_dir, "autoencoder_emg.pth"),
    os.path.join(out_dir, "autoencoder_emg_2d.pth"),
    os.path.join(out_dir, "autoencoder_emg_8d.pth"),
]
model_path = next((m for m in model_candidates if os.path.exists(m)), None)
if not model_path:
    print("> ERROR: No se encontró ningún modelo (.pth) entrenado.")
    sys.exit(1)

import deep_learning.decodificador_continuo as dc
dc.decodificar_secuencia(
    carpeta_secuencia=carpeta_secuencia,
    modelo_path=model_path,
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    smooth_ms=kwargs.get('smooth_ms', 150),
    notch_q=kwargs.get('notch_q', 2.0),
    use_manual_exclusions=kwargs.get('use_manual_exclusions', True),
    target_length=kwargs.get('target_length', 100)
)
