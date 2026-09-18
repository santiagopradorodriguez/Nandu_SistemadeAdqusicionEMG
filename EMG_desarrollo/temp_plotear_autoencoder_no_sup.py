
import json
import sys
import os
import glob
import torch

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmpiz_u4pq3.json', 'r') as f:
    kwargs = json.load(f)

import deep_learning.motor_autoencoder_unificado as motor

modalidad = kwargs.get('modalidad', 'envolvente')
latent_dim = kwargs.get('latent_dim', 2)
algoritmo_clustering = kwargs.get('algoritmo_clustering', 'gmm')

proc_dirs = sorted(glob.glob(os.path.join(project_root, "resultados", "resultados_autoencoder", "procesamiento_*")))
target_folder = proc_dirs[-1] if proc_dirs else os.path.join(project_root, "resultados", "resultados_autoencoder")

npz_path = os.path.join(target_folder, "dataset_autoencoder_unificado.npz")
if not os.path.exists(npz_path):
    npz_path = os.path.join(project_root, "cache_datos", "dataset_autoencoder_unificado.npz")

if not os.path.exists(npz_path):
    print(f"ERROR: No existe el dataset {npz_path}. Ejecuta la extracción primero (Paso 1).")
    sys.exit(1)

print("\n" + "="*70)
print(f"  EVALUACION DEL ESPACIO LATENTE ({modalidad.upper()} - {latent_dim}D)")
print(f"  Alineacion Canonica (/a/ en +Y, sonrisa en +X) | Metricas y Fronteras: {algoritmo_clustering.upper()}")
print(f"  Carpeta de salida: {target_folder}")
print("="*70 + "\n")

metricas = motor.evaluar_espacio_latente(
    archivo_npz=npz_path,
    modelo=None,
    modalidad=modalidad,
    latent_dim=latent_dim,
    carpeta_salida=target_folder,
    usar_custom_arch=kwargs.get('usar_custom_arch', False),
    codigo_custom_arch=kwargs.get('codigo_custom_arch', None),
    mostrar_grafico=True,
    algoritmo_clustering=algoritmo_clustering,
    callback_log=print
)
print(f"\nEvaluacion completada. Exactitud {metricas.get('algoritmo_clustering', 'Clustering')}: {metricas['cluster_acc']:.2f}%")
print(f"Informe grafico guardado en: {metricas['fig_path']}")
if metricas and metricas.get('fig_path') and os.path.exists(metricas['fig_path']):
    motor.abrir_imagen_en_visor(metricas['fig_path'])
