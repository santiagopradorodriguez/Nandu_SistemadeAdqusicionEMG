#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG
# Descripción: Exportador de pulsos crudos sEMG segmentados a CSV y tablas
#              consolidadas para Argon.
# ==============================================================================

import os
import sys
import numpy as np
import pandas as pd

OUT_DIR = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/csv_crudos_por_dia'
os.makedirs(OUT_DIR, exist_ok=True)

NPZ_DIR = '/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets'

def exportar_npz_a_csv(npz_name, prefijo_salida):
    npz_path = os.path.join(NPZ_DIR, npz_name)
    data = np.load(npz_path)
    
    X_3d = data['X_3d_20']       # (N, 3, 20)
    X_flat = data['X_flat_60']   # (N, 60)
    Y = data['Y']                # (N,)
    Tomas = data['Tomas']        # (N,)
    Fechas = data['Fechas']      # (N,)
    Sesiones = data['Sesiones']  # (N,)
    N = len(Y)
    
    print(f"Exportando {prefijo_salida} (N={N} pulsos)...")
    
    # 1. CSV Consolidado con features temporales (60 columnas) + metadatos
    cols_feat = []
    canales = ['Ch0_Digastrico', 'Ch1_Cigomatico', 'Ch2_Orbicular']
    for ch in canales:
        for t in range(20):
            cols_feat.append(f"{ch}_t{t:02d}")
            
    df_features = pd.DataFrame(X_flat, columns=cols_feat)
    df_features.insert(0, 'pulso_id', np.arange(1, N + 1))
    df_features.insert(1, 'vocal', Y)
    df_features.insert(2, 'sesion', Sesiones)
    df_features.insert(3, 'toma', Tomas)
    df_features.insert(4, 'fecha', Fechas)
    
    csv_feat_path = os.path.join(OUT_DIR, f"{prefijo_salida}_features_60d.csv")
    df_features.to_csv(csv_feat_path, index=False)
    print(f"  -> Guardado: {csv_feat_path}")
    
    # 2. CSV Formato Largo (Tidy) para modelos secuenciales/crudos:
    # pulso_id, vocal, sesion, toma, step (0..19), Ch0_Digastrico, Ch1_Cigomatico, Ch2_Orbicular
    rows_tidy = []
    for i in range(N):
        pid = i + 1
        voc = Y[i]
        ses = Sesiones[i]
        tom = Tomas[i]
        fec = Fechas[i]
        for t in range(20):
            rows_tidy.append({
                'pulso_id': pid,
                'vocal': voc,
                'sesion': ses,
                'toma': tom,
                'fecha': fec,
                'time_step': t,
                'Ch0_Digastrico': X_3d[i, 0, t],
                'Ch1_Cigomatico': X_3d[i, 1, t],
                'Ch2_Orbicular': X_3d[i, 2, t]
            })
    df_tidy = pd.DataFrame(rows_tidy)
    csv_tidy_path = os.path.join(OUT_DIR, f"{prefijo_salida}_series_temporales_crudo_3ch.csv")
    df_tidy.to_csv(csv_tidy_path, index=False)
    print(f"  -> Guardado: {csv_tidy_path}")

print("=== Exportando CSVs de 2026-10-28 ===")
exportar_npz_a_csv('dataset_candela_2026_10_28_separado.npz', 'candela_2026_10_28')

print("\n=== Exportando CSVs de 2026-10-30 ===")
exportar_npz_a_csv('dataset_candela_2026_10_30_separado.npz', 'candela_2026_10_30')

# Copiar también a artifacts
art_dir = '/home/santiago/.gemini/antigravity/brain/39ced7b1-5efb-4d1f-b72b-7de07df1fc29'
os.system(f'cp {OUT_DIR}/*.csv "{art_dir}/"')
print(f"\nTodos los CSVs copiados a artifacts: {art_dir}")
