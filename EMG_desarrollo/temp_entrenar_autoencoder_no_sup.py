
import json
import sys
import os
import glob

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmp37x79w7g.json', 'r') as f:
    kwargs = json.load(f)

import deep_learning.motor_autoencoder_unificado as motor

modalidad = kwargs.get('modalidad', 'envolvente')
latent_dim = kwargs.get('latent_dim', 2)
epochs = kwargs.get('epochs', 150)
batch_size = kwargs.get('batch_size', 32)
lr = kwargs.get('lr', 0.002)
tipo_perdida = kwargs.get('tipo_perdida', 'mse')
gamma_sdtw = kwargs.get('gamma_sdtw', 1.0)

proc_dirs = sorted(glob.glob(os.path.join(project_root, "resultados", "resultados_autoencoder", "procesamiento_*")))
target_folder = proc_dirs[-1] if proc_dirs else os.path.join(project_root, "resultados", "resultados_autoencoder")

npz_path = os.path.join(target_folder, "dataset_autoencoder_unificado.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(project_root, "cache_datos", "dataset_autoencoder_unificado.npz")
if not os.path.exists(npz_path):
    print(f"ERROR: No se encontro el dataset en {npz_path}. Ejecuta la extraccion primero.")
    sys.exit(1)

print("\n" + "="*70)
print(f"  ENTRENAMIENTO AUTOENCODER NO SUPERVISADO ({modalidad.upper()} - {latent_dim}D)")
print(f"  Zero-Labels: 100% No Supervisado | Pérdida: {tipo_perdida.upper()} (gamma: {gamma_sdtw})")
print(f"  Carpeta de procesamiento: {target_folder}")
print("="*70 + "\n")

modelo, pth = motor.entrenar_autoencoder(
    archivo_npz=npz_path,
    modalidad=modalidad,
    latent_dim=latent_dim,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    tipo_perdida=tipo_perdida,
    gamma_sdtw=gamma_sdtw,
    carpeta_salida=target_folder,
    usar_custom_arch=kwargs.get('usar_custom_arch', False),
    codigo_custom_arch=kwargs.get('codigo_custom_arch', None),
    callback_log=print
)
print(f"\nEntrenamiento culminado con exito: {pth}")
