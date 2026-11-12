#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG
# Autores: Lucas Braunstein y Santiago Prado
# Descripción: Generador oficial de datasets .npz separados por día:
#              - 2026-10-28 (Día 1: Habla Horizontal Natural vs Vertical Forzada)
#              - 2026-10-30 (Día 2: Secuencias Continuas De Bruijn Jittering)
#              Incluye tensores 3D temporales (N, 3, 20), vectores 2D (N, 60),
#              envolventes pre-fono (N, 3, 10), etiquetas y metadatos.
# ==============================================================================

import os
import sys
import numpy as np

# Semilla fija
np.random.seed(42)

sys.path.append('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/deep_learning/pca_umap_clustering')
sys.path.append('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/deep_learning/binarizacion')
from generador_pca_umap import extraer_y_filtrar

OUT_DIR = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets'
os.makedirs(OUT_DIR, exist_ok=True)

params = {
    'alpha_ruido': 0.5,
    'gate_ratio_ruido': 0.0,
    'snr_threshold': 0.5,
    'outlier_contamination': 0.10,
    'smooth_ms': 125,
    'target_length': 20,
    'tipo_filtro_ruido': 'notch',
    'notch_q': 30.0,
    'highpass_cutoff_hz': 20.0,
    'lowpass_cutoff_hz': 500.0,
    'aplicar_correccion_intersesion': False
}

CANALES = ['canal_0', 'canal_1', 'canal_2']
MUSCULOS = np.array(['Digástrico (Ch0)', 'Cigomático (Ch1)', 'Orbicular (Ch2)'])

def extraer_bloque(subpath, sesion_id, fecha_str):
    base_dir = os.path.join('/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/Candela_Organizado_Nuevo', subpath, 'datos_crudos_y_pulsos')
    meds = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d != 'fotos_montaje'])
    print(f"Extrayendo {sesion_id} ({len(meds)} tomas)...")
    res = extraer_y_filtrar(meds, base_dir, params, False, 'Pico Volumen Micrófono', 0.4, 0.6, CANALES, verbose=False)
    X_raw = np.array(res[0], dtype=np.float32)
    y_raw = np.array(res[1])
    tomas_raw = np.array(res[2])
    
    N, D = X_raw.shape
    T = D // 3
    X_3d = X_raw.reshape(N, 3, T)
    
    # Recorte pre-fono
    idx_corte = max(6, min(T - 2, int(round(T * 0.58))))
    idx_ini_pre = max(0, int(round(T * 0.12)))
    X_pre_3d = X_3d[:, :, idx_ini_pre:idx_corte]
    
    fechas = np.array([fecha_str] * N)
    sesiones = np.array([sesion_id] * N)
    
    print(f"  -> Extraídos {N} pulsos limpios. Clases: {np.unique(y_raw, return_counts=True)}")
    return {
        'X_flat_60': X_raw,
        'X_3d_20': X_3d,
        'X_pre_3d': X_pre_3d,
        'Y': y_raw,
        'Tomas': tomas_raw,
        'Fechas': fechas,
        'Sesion': sesiones
    }

print("=== 1. Procesando mediciones del 2026-10-28 ===")
b_1028_exp1 = extraer_bloque('2026-10-28_Exp1_HablaHorizontalNatural_Digastrico_Cigomatico_Orbicular', '1028_Exp1_Horizontal', '2026-10-28')
b_1028_exp2 = extraer_bloque('2026-10-28_Exp2_AperturaVerticalForzada_MismosElectrodos_4minDespues', '1028_Exp2_Vertical', '2026-10-28')

# Dataset 1: Día 2026-10-28 completo (con ambas variantes etiquetadas)
X_1028_flat = np.vstack([b_1028_exp1['X_flat_60'], b_1028_exp2['X_flat_60']])
X_1028_3d = np.concatenate([b_1028_exp1['X_3d_20'], b_1028_exp2['X_3d_20']], axis=0)
X_1028_pre = np.concatenate([b_1028_exp1['X_pre_3d'], b_1028_exp2['X_pre_3d']], axis=0)
Y_1028 = np.concatenate([b_1028_exp1['Y'], b_1028_exp2['Y']])
Tomas_1028 = np.concatenate([b_1028_exp1['Tomas'], b_1028_exp2['Tomas']])
Fechas_1028 = np.concatenate([b_1028_exp1['Fechas'], b_1028_exp2['Fechas']])
Sesiones_1028 = np.concatenate([b_1028_exp1['Sesion'], b_1028_exp2['Sesion']])

path_npz_1028 = os.path.join(OUT_DIR, 'dataset_candela_2026_10_28_separado.npz')
np.savez_compressed(
    path_npz_1028,
    X_flat_60=X_1028_flat,
    X_3d_20=X_1028_3d,
    X_pre_3d=X_1028_pre,
    Y=Y_1028,
    Tomas=Tomas_1028,
    Fechas=Fechas_1028,
    Sesiones=Sesiones_1028,
    Musculos_Canales=MUSCULOS,
    X_exp1_horizontal_flat=b_1028_exp1['X_flat_60'],
    X_exp1_horizontal_3d=b_1028_exp1['X_3d_20'],
    Y_exp1=b_1028_exp1['Y'],
    X_exp2_vertical_flat=b_1028_exp2['X_flat_60'],
    X_exp2_vertical_3d=b_1028_exp2['X_3d_20'],
    Y_exp2=b_1028_exp2['Y']
)
print(f"Guardado: {path_npz_1028} (N={len(Y_1028)} pulsos: {len(b_1028_exp1['Y'])} Horizontal + {len(b_1028_exp2['Y'])} Vertical)")

print("\n=== 2. Procesando mediciones del 2026-10-30 ===")
b_1030_exp1 = extraer_bloque('2026-10-30_Exp1_SecuenciaDeBruijn_Jittering_Serie1', '1030_Exp1_Serie1', '2026-10-30')
b_1030_exp2 = extraer_bloque('2026-10-30_Exp2_SecuenciaDeBruijn_Jittering_Serie2', '1030_Exp2_Serie2', '2026-10-30')

X_1030_flat = np.vstack([b_1030_exp1['X_flat_60'], b_1030_exp2['X_flat_60']])
X_1030_3d = np.concatenate([b_1030_exp1['X_3d_20'], b_1030_exp2['X_3d_20']], axis=0)
X_1030_pre = np.concatenate([b_1030_exp1['X_pre_3d'], b_1030_exp2['X_pre_3d']], axis=0)
Y_1030 = np.concatenate([b_1030_exp1['Y'], b_1030_exp2['Y']])
Tomas_1030 = np.concatenate([b_1030_exp1['Tomas'], b_1030_exp2['Tomas']])
Fechas_1030 = np.concatenate([b_1030_exp1['Fechas'], b_1030_exp2['Fechas']])
Sesiones_1030 = np.concatenate([b_1030_exp1['Sesion'], b_1030_exp2['Sesion']])

path_npz_1030 = os.path.join(OUT_DIR, 'dataset_candela_2026_10_30_separado.npz')
np.savez_compressed(
    path_npz_1030,
    X_flat_60=X_1030_flat,
    X_3d_20=X_1030_3d,
    X_pre_3d=X_1030_pre,
    Y=Y_1030,
    Tomas=Tomas_1030,
    Fechas=Fechas_1030,
    Sesiones=Sesiones_1030,
    Musculos_Canales=MUSCULOS,
    X_exp1_serie1_flat=b_1030_exp1['X_flat_60'],
    X_exp1_serie1_3d=b_1030_exp1['X_3d_20'],
    Y_exp1=b_1030_exp1['Y'],
    X_exp2_serie2_flat=b_1030_exp2['X_flat_60'],
    X_exp2_serie2_3d=b_1030_exp2['X_3d_20'],
    Y_exp2=b_1030_exp2['Y']
)
print(f"Guardado: {path_npz_1030} (N={len(Y_1030)} pulsos: {len(b_1030_exp1['Y'])} Serie1 + {len(b_1030_exp2['Y'])} Serie2)")

# Copiar también al directorio de artifacts para descarga o inspección directa
artifact_dir = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29'
os.system(f'cp "{path_npz_1028}" "{artifact_dir}/dataset_candela_2026_10_28_separado.npz"')
os.system(f'cp "{path_npz_1030}" "{artifact_dir}/dataset_candela_2026_10_30_separado.npz"')
print(f"\nArchivos copiados a artifacts: {artifact_dir}")
