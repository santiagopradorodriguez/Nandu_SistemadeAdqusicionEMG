import os
import sys

# Agregar el directorio al path
script_dir = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\analysis'
sys.path.append(script_dir)

from generador_pca_umap import ejecutar_procesamiento

import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root


base_dir = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\base_de_datos_electrodos'

# Buscar mediciones
mediciones = []
dias_dir = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
for dia in dias_dir:
    dia_path = os.path.join(base_dir, dia)
    for m in os.listdir(dia_path):
        if os.path.isdir(os.path.join(dia_path, m)):
            mediciones.append((dia, m))

# Llamar directamente a la función de procesamiento con los defaults
ejecutar_procesamiento(
    base_dir=base_dir,
    mediciones=meds if 'meds' in locals() else mediciones,
    snr_threshold=0.5,
    outlier_contamination=0.05,
    alpha_ruido=1.0,
    smooth_ms=120,
    target_length=20,
    notch_q=2.0,
    umap_n_neighbors=15,
    umap_min_dist=0.1,
    umap_metric='manhattan'
)
