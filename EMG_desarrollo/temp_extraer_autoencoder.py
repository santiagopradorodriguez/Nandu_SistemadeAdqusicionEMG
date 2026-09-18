
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

with open(r'/tmp/tmpl7xw7guy.json', 'r') as f:
    kwargs = json.load(f)

mediciones = ['2026-09-16/A_Serie1_Candela', '2026-09-16/A_Serie2_Candela', '2026-09-16/A_Serie3_Candela', '2026-09-16/A_Serie4_Candela', '2026-09-16/E_Serie1_Candela', '2026-09-16/E_Serie2_Candela', '2026-09-16/E_Serie3_Candela', '2026-09-16/E_Serie4_Candela', '2026-09-16/I_Serie1_Candela', '2026-09-16/I_Serie2_Candela', '2026-09-16/I_Serie3_Candela', '2026-09-16/I_Serie4_Candela', '2026-09-16/O_Serie1_Candela', '2026-09-16/O_Serie2_Candela', '2026-09-16/O_Serie3_Candela', '2026-09-16/O_Serie4_Candela', '2026-09-16/U_Serie1_Candela', '2026-09-16/U_Serie2_Candela', '2026-09-16/U_Serie3_Candela', '2026-09-16/U_Serie4_Candela', '2026-09-04/A_Prueba4_Cande', '2026-09-04/A_Prueba5_Cande', '2026-09-04/E_Prueba4_Cande', '2026-09-04/E_Prueba5_Cande', '2026-09-04/I_Prueba4_Cande', '2026-09-04/I_Prueba5_Cande', '2026-09-04/O_Prueba4_Cande', '2026-09-04/O_Prueba5_Cande', '2026-09-04/U_Prueba4_Cande', '2026-09-04/U_Prueba5_Cande', '2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/SecuenciaContinua_Prueba1_Candela', '2026-09-01/SecuenciaContinua_Prueba2_Candela', '2026-09-01/SecuenciaContinua_Prueba3_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela', '2026-08-30/A_Prueba4_Cande', '2026-08-30/A_Prueba5_Cande', '2026-08-30/E_Prueba4_Cande', '2026-08-30/E_Prueba5_Cande', '2026-08-30/I_Prueba4_Cande', '2026-08-30/I_Prueba5_Cande', '2026-08-30/O_Prueba4_Cande', '2026-08-30/O_Prueba5_Cande', '2026-08-30/U_Prueba4_Cande', '2026-08-30/U_Prueba5_Cande', '2026-08-28/A_med1_clase24_Petra', '2026-08-28/A_med2_clase4_Petra', '2026-08-28/E_med1_clase24_Petra', '2026-08-28/E_med2_clase4_Petra', '2026-08-28/I_med1_clase24_Petra', '2026-08-28/I_med2_clase4_Petra', '2026-08-28/O_med1_clase24_Petra', '2026-08-28/O_med2_clase4_Petra', '2026-08-28/U_med1_clase24_Petra', '2026-08-28/U_med2_clase4_Petra', '2026-08-21/A_Prueba1silicona_Petra', '2026-08-21/A_silicona_aeiou_Petra', '2026-08-21/E_Prueba1silicona_Petra', '2026-08-21/E_silicona_aeiou_Petra', '2026-08-21/I_Prueba1silicona_Petra', '2026-08-21/I_silicona_aeiou_Petra', '2026-08-21/O_Prueba1silicona_Petra', '2026-08-21/O_silicona_aeiou_Petra', '2026-08-21/U_Prueba1silicona_Petra', '2026-08-21/U_silicona_aeiou_Petra', '2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas', '2026-07-10/A_T3_Lucas', '2026-07-10/A_T4_Lucas', '2026-07-10/A_T5_Lucas', '2026-07-10/A_T6_Lucas', '2026-07-10/A_T7_Lucas', '2026-07-10/E_T1_Lucas', '2026-07-10/E_T2_Lucas', '2026-07-10/E_T3_Lucas', '2026-07-10/E_T4_Lucas', '2026-07-10/E_T5_Lucas', '2026-07-10/E_T6_Lucas', '2026-07-10/E_T7_Lucas', '2026-07-10/I_T1_Lucas', '2026-07-10/I_T2_Lucas', '2026-07-10/I_T3_Lucas', '2026-07-10/I_T4_Lucas', '2026-07-10/I_T5_Lucas', '2026-07-10/I_T6_Lucas', '2026-07-10/I_T7_Lucas', '2026-07-10/O_T1_Lucas', '2026-07-10/O_T2_Lucas', '2026-07-10/O_T3_Lucas', '2026-07-10/O_T4_Lucas', '2026-07-10/O_T5_Lucas', '2026-07-10/O_T6_Lucas', '2026-07-10/O_T7_Lucas', '2026-07-10/UMBRALES', '2026-07-10/U_T1_Lucas', '2026-07-10/U_T2_Lucas', '2026-07-10/U_T3_Lucas', '2026-07-10/U_T4_Lucas', '2026-07-10/U_T5_Lucas', '2026-07-10/U_T6_Lucas', '2026-07-10/U_T7_Lucas', '2026-06-22/A_Prueba1_SANTI', '2026-06-22/A_Prueba2_SANTI', '2026-06-22/A_Prueba3_SANTI', '2026-06-22/A_Prueba4_SANTI', '2026-06-22/E_Prueba1_SANTI', '2026-06-22/E_Prueba2_SANTI', '2026-06-22/E_Prueba3_SANTI', '2026-06-22/E_Prueba4_SANTI', '2026-06-22/I_Prueba1_SANTI', '2026-06-22/I_Prueba2_SANTI', '2026-06-22/I_Prueba3_SANTI', '2026-06-22/I_Prueba4_SANTI', '2026-06-22/O_Prueba1_SANTI', '2026-06-22/O_Prueba2_SANTI', '2026-06-22/O_Prueba3_SANTI', '2026-06-22/O_Prueba4_SANTI', '2026-06-22/SecuenciaContinua_Prueba5_SANTI', '2026-06-22/SecuenciaContinua_Prueba6_SANTI', '2026-06-22/U_Prueba1_SANTI', '2026-06-22/U_Prueba2_SANTI', '2026-06-22/U_Prueba3_SANTI', '2026-06-22/U_Prueba4_SANTI']
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
