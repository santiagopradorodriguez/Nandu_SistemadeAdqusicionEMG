
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmpqa4sub7d.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela', '2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas', '2026-07-10/A_T3_Lucas', '2026-07-10/A_T4_Lucas', '2026-07-10/A_T5_Lucas', '2026-07-10/A_T6_Lucas', '2026-07-10/A_T7_Lucas', '2026-07-10/E_T1_Lucas', '2026-07-10/E_T2_Lucas', '2026-07-10/E_T3_Lucas', '2026-07-10/E_T4_Lucas', '2026-07-10/E_T5_Lucas', '2026-07-10/E_T6_Lucas', '2026-07-10/E_T7_Lucas', '2026-07-10/I_T1_Lucas', '2026-07-10/I_T2_Lucas', '2026-07-10/I_T3_Lucas', '2026-07-10/I_T4_Lucas', '2026-07-10/I_T5_Lucas', '2026-07-10/I_T6_Lucas', '2026-07-10/I_T7_Lucas', '2026-07-10/O_T1_Lucas', '2026-07-10/O_T2_Lucas', '2026-07-10/O_T3_Lucas', '2026-07-10/O_T4_Lucas', '2026-07-10/O_T5_Lucas', '2026-07-10/O_T6_Lucas', '2026-07-10/O_T7_Lucas', '2026-07-10/U_T1_Lucas', '2026-07-10/U_T2_Lucas', '2026-07-10/U_T3_Lucas', '2026-07-10/U_T4_Lucas', '2026-07-10/U_T5_Lucas', '2026-07-10/U_T6_Lucas', '2026-07-10/U_T7_Lucas', '2026-06-22/A_Prueba1_SANTI', '2026-06-22/A_Prueba2_SANTI', '2026-06-22/E_Prueba1_SANTI', '2026-06-22/E_Prueba2_SANTI', '2026-06-22/O_Prueba1_SANTI', '2026-06-22/O_Prueba2_SANTI', '2026-06-22/U_Prueba1_SANTI', '2026-06-22/U_Prueba2_SANTI']
base_dir = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos'

train_s = kwargs.get('train_sessions', [])
test_s = kwargs.get('test_sessions', [])
if train_s or test_s:
    all_session_names = set(train_s + test_s)
    matched_mediciones = []
    for root, dirs, files in os.walk(base_dir):
        for d in dirs:
            parts = d.split('_')
            if len(parts) > 1 and parts[0].upper() in ['A', 'E', 'I', 'O', 'U']:
                s_id = '_'.join(parts[1:])
                if s_id in all_session_names or d in all_session_names:
                    rel = os.path.relpath(os.path.join(root, d), base_dir)
                    if rel not in matched_mediciones:
                        matched_mediciones.append(rel)
    if matched_mediciones:
        mediciones = matched_mediciones

import deep_learning.dataset_tools.generador_pca_tensorial as gpt

print("==================================================")
print("EXTRAYENDO DATASET TENSORIAL PARA AUTOENCODER...")
print(f"Mediciones seleccionadas: {len(mediciones)}")
print("==================================================")

gpt.ejecutar_procesamiento(
    mediciones=mediciones,
    alpha_ruido=kwargs.get('alpha_ruido', 1.0),
    snr_threshold=kwargs.get('snr_min', 0.5),
    outlier_contamination=kwargs.get('outliers_pct', 0.05),
    smooth_ms=kwargs.get('smooth_ms', 150),
    target_length=kwargs.get('target_length', 100),
    notch_q=kwargs.get('notch_q', 2.0),
    use_manual_exclusions=kwargs.get('use_manual_exclusions', True),
    verbose=True
)
