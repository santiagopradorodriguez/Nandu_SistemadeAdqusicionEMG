
import json
import sys
import os
from datetime import datetime

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmplqbq0vs0.json', 'r') as f:
    kwargs = json.load(f)

base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'
mediciones = ['2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas', '2026-07-10/A_T3_Lucas', '2026-07-10/A_T4_Lucas', '2026-07-10/A_T5_Lucas', '2026-07-10/A_T6_Lucas', '2026-07-10/A_T7_Lucas', '2026-07-10/E_T1_Lucas', '2026-07-10/E_T2_Lucas', '2026-07-10/E_T3_Lucas', '2026-07-10/E_T4_Lucas', '2026-07-10/E_T5_Lucas', '2026-07-10/E_T6_Lucas', '2026-07-10/E_T7_Lucas', '2026-07-10/I_T1_Lucas', '2026-07-10/I_T2_Lucas', '2026-07-10/I_T3_Lucas', '2026-07-10/I_T4_Lucas', '2026-07-10/I_T5_Lucas', '2026-07-10/I_T6_Lucas', '2026-07-10/I_T7_Lucas', '2026-07-10/O_T1_Lucas', '2026-07-10/O_T2_Lucas', '2026-07-10/O_T3_Lucas', '2026-07-10/O_T4_Lucas', '2026-07-10/O_T5_Lucas', '2026-07-10/O_T6_Lucas', '2026-07-10/O_T7_Lucas', '2026-07-10/UMBRALES', '2026-07-10/U_T1_Lucas', '2026-07-10/U_T2_Lucas', '2026-07-10/U_T3_Lucas', '2026-07-10/U_T4_Lucas', '2026-07-10/U_T5_Lucas', '2026-07-10/U_T6_Lucas', '2026-07-10/U_T7_Lucas']
rutas = [os.path.join(base_dir, m) for m in mediciones]

import deep_learning.motor_autoencoder_unificado as motor

modalidad = kwargs.get('modalidad', 'envolvente')
latent_dim = kwargs.get('latent_dim', 2)
ts_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
carpeta_exp = os.path.join(project_root, "resultados", "resultados_autoencoder", f"procesamiento_{ts_str}_{modalidad}_{latent_dim}d")
os.makedirs(carpeta_exp, exist_ok=True)

print("\n" + "="*70)
print("  EXTRACCION DATASET UNIFICADO (AUTOENCODER NO SUPERVISADO)")
print("  Supremo Tricanal | Ruido IQR Dinamico | Calibracion Fisiologica")
print(f"  Carpeta de procesamiento: {carpeta_exp}")
print("="*70 + "\n")

npz_path, n_pulsos = motor.extraer_dataset_unificado(
    rutas,
    usar_calibracion_p95=kwargs.get('usar_calibracion_p95', True),
    modo_alineacion=kwargs.get('modo_alineacion', 'Pico Volumen Micrófono'),
    carpeta_salida=carpeta_exp,
    tipo_envolvente=kwargs.get('tipo_envolvente', 'rms'),
    smooth_ms=kwargs.get('smooth_ms', 90),
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    target_len=kwargs.get('target_len', 100),
    outlier_contamination=kwargs.get('outlier_contamination', 0.10),
    w_canales=kwargs.get('w_canales', [1.0, 1.0, 1.0]),
    callback_log=print
)
print(f"\nExtraccion culminada con exito: {n_pulsos} pulsos extraidos.")
print(f"Dataset guardado en: {npz_path}")
