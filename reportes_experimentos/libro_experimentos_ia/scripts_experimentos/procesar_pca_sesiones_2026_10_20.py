# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Procesamiento automático de PCA 2D y 3D para las mediciones del 2026-10-20
#              (Candela Experimento 1, Candela Experimento 2 y Sesión Exploratoria Santi).
# ==============================================================================

import os
import sys
import json

# Rutas del proyecto
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
emg_desarrollo = os.path.join(repo_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

import deep_learning.pca_umap_clustering.generador_pca_umap as generador

base_dir = os.path.join(emg_desarrollo, "base_de_datos_electrodos")
resultados_root = os.path.join(emg_desarrollo, "deep_learning", "pca_umap_clustering", "resultados_pca_umap")

# Parámetros estándar de acondicionamiento y PCA
params_2d = {
    "alpha_ruido": 0.5,
    "gate_ratio_ruido": 0.0,
    "smooth_ms": 90,
    "target_length": 20,
    "snr_threshold": 0.5,
    "outlier_contamination": 0.05,
    "notch_q": 2.0,
    "tipo_filtro_ruido": "notch",
    "highpass_cutoff_hz": 20.0,
    "lowpass_cutoff_hz": 300.0,
    "comp_x": "PC1",
    "comp_y": "PC2",
    "pesos_canales": [1.0, 1.0, 1.0]
}

params_3d = {
    "alpha_ruido": 0.5,
    "gate_ratio_ruido": 0.0,
    "smooth_ms": 125,
    "target_length": 20,
    "snr_threshold": 0.5,
    "outlier_contamination": 0.1,
    "notch_q": 2.0,
    "tipo_filtro_ruido": "notch",
    "highpass_cutoff_hz": 20.0,
    "lowpass_cutoff_hz": 300.0,
    "comp_x": "PC1",
    "comp_y": "PC2",
    "comp_z": "PC3",
    "pesos_canales": [1.0, 1.0, 1.0]
}

config_base = {
    "proc_pca_2d": True,
    "proc_pca_3d": True,
    "proc_umap_2d": False,
    "proc_umap_3d": False,
    "ocultar_leyenda": False,
    "tipo_filtro_ruido": "notch",
    "notch_q": 2.0,
    "highpass_cutoff_hz": 20.0,
    "lowpass_cutoff_hz": 300.0,
    "params_2d": params_2d,
    "params_3d": params_3d,
    "algoritmo_clustering_pca": "GMM",
    "algoritmo_clustering_umap": "K-Means",
    "aplicar_trevisan": False,
    "aplicar_correccion_intersesion": False,
    "correccion_impedancia": True,
    "ignorar_ventana_cero": False,
    "pre_pct": 0.4,
    "post_pct": 0.6,
    "modo_alineacion": "Pico Volumen Micrófono",
    "estilo_visual": "Fronteras",
    "canales_features": ["canal_0", "canal_1", "canal_2"]
}

# 1. Definición de sets de tomas
mediciones_candela_exp1 = [
    "2026-10-20/A_Exp1_Prueba1_Candela",
    "2026-10-20/A_Exp1_Prueba2_Candela",
    "2026-10-20/A_Exp1_Prueba3_Candela",
    "2026-10-20/E_Exp1_Prueba1_Candela",
    "2026-10-20/E_Exp1_Prueba2_Candela",
    "2026-10-20/E_Exp1_Prueba3_Candela",
    "2026-10-20/I_Exp1_Prueba1_Candela",
    "2026-10-20/I_Exp1_Prueba2_Candela",
    "2026-10-20/I_Exp1_Prueba3_Candela",
    "2026-10-20/O_Exp1_Prueba1_Candela",
    "2026-10-20/O_Exp1_Prueba2_Candela",
    "2026-10-20/O_Exp1_Prueba3_Candela",
    "2026-10-20/O_Exp1_Prueba4_Candela",
    "2026-10-20/U_Exp1_Prueba1_Candela",
    "2026-10-20/U_Exp1_Prueba2_Candela",
    "2026-10-20/U_Exp1_Prueba3_Candela",
    "2026-10-20/U_Exp1_Prueba4_Candela"
]

mediciones_candela_exp2 = [
    "2026-10-20/A_Exp2_Prueba1_Candela",
    "2026-10-20/A_Exp2_Prueba2_Candela",
    "2026-10-20/A_Exp2_Prueba3_Candela",
    "2026-10-20/A_Exp2_Prueba4_Candela",
    "2026-10-20/E_Exp2_Prueba1_Candela",
    "2026-10-20/E_Exp2_Prueba2_Candela",
    "2026-10-20/E_Exp2_Prueba3_Candela",
    "2026-10-20/E_Exp2_Prueba4_Candela",
    "2026-10-20/I_Exp2_Prueba1_Candela",
    "2026-10-20/I_Exp2_Prueba2_Candela",
    "2026-10-20/I_Exp2_Prueba3_Candela",
    "2026-10-20/I_Exp2_Prueba4_Candela",
    "2026-10-20/O_Exp2_Prueba1_Candela",
    "2026-10-20/O_Exp2_Prueba2_Candela",
    "2026-10-20/O_Exp2_Prueba3_Candela",
    "2026-10-20/O_Exp2_Prueba4_Candela",
    "2026-10-20/O_Exp2_Prueba5_Candela",
    "2026-10-20/U_Exp2_Prueba1_Candela",
    "2026-10-20/U_Exp2_Prueba2_Candela",
    "2026-10-20/U_Exp2_Prueba3_Candela",
    "2026-10-20/U_Exp2_Prueba4_Candela"
]

mediciones_santi = [
    # Excluidas por falla técnica / señal nula (< 16.5 uV):
    # "2026-10-20/A_Prueba1_Santi",
    # "2026-10-20/A_Prueba2_Santi",
    "2026-10-20/A_Prueba3_Santi",
    "2026-10-20/A_Prueba4_Santi",
    "2026-10-20/A_Prueba5_Santi",
    "2026-10-20/E_Prueba1_Santi",
    "2026-10-20/E_Prueba2_Santi",
    "2026-10-20/E_Prueba3_Santi",
    "2026-10-20/E_Prueba4_Santi",
    "2026-10-20/I_Prueba1_Santi",
    "2026-10-20/I_Prueba2_Santi",
    "2026-10-20/I_Prueba3_Santi",
    "2026-10-20/I_Prueba4_Santi",
    "2026-10-20/O_Prueba1_Santi",
    "2026-10-20/O_Prueba2_Santi",
    "2026-10-20/O_Prueba3_Santi",
    "2026-10-20/O_Prueba4_Santi",
    "2026-10-20/O_Prueba5_Santi",
    "2026-10-20/U_Prueba1_Santi",
    "2026-10-20/U_Prueba2_Santi",
    "2026-10-20/U_Prueba3_Santi",
    "2026-10-20/U_Prueba4_Santi"
]

jobs = [
    ("2026-10-20_Candela_Exp1", mediciones_candela_exp1),
    ("2026-10-20_Candela_Exp2", mediciones_candela_exp2),
    ("2026-10-20_Santi_Exploracion", mediciones_santi)
]

print("======================================================================")
print("INICIANDO PROCESAMIENTO PCA 2D/3D (2026-10-20)")
print("======================================================================")

for nombre_set, meds in jobs:
    out_dir = os.path.join(resultados_root, nombre_set)
    os.makedirs(out_dir, exist_ok=True)
    
    cfg_job = dict(config_base)
    if "Santi" in nombre_set:
        # En Santiago desde 15:37:22 (Prueba2 a Prueba5):
        # Hardware: [Ch0=mast, Ch1=orbi, Ch2=nudo].
        # Permutación [1, 2, 0] para restituir orden canónico [Ch0=orbi, Ch1=modiolo/nudo, Ch2=masetero]:
        cfg_job["canales_features"] = ["canal_1", "canal_2", "canal_0"]
    
    print(f"\n[PROCESANDO] {nombre_set} ({len(meds)} tomas)")
    print(f"Directorio de salida: {out_dir}")
    print(f"Canales configurados: {cfg_job['canales_features']}")
    
    with open(os.path.join(out_dir, "parametros.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_job, f, indent=4, ensure_ascii=False)
        
    try:
        generador.ejecutar_procesamiento(
            mediciones=meds,
            base_dir=base_dir,
            out_dir=out_dir,
            **cfg_job
        )
        print(f"[OK] {nombre_set} finalizado exitosamente.")
    except Exception as e:
        print(f"[ERROR] Falló procesamiento en {nombre_set}: {e}")
        import traceback
        traceback.print_exc()

print("\n======================================================================")
print("TODOS LOS PROCESAMIENTOS DE PCA HAN CONCLUIDO.")
print("======================================================================")
