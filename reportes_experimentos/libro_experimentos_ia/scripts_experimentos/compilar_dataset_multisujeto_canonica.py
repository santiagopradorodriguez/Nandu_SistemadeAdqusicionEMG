# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Módulo: Compilador de Dataset Multi-Sujeto Canónico (Fase 1: Lucas, Santi, Candela)
# Tríada: Canal 0 = Apertura, Canal 1 = Sonrisa, Canal 2 = Protrusión
# Normalización: Supremo Tricanal por Pulso Individual + Isolation Forest 10%
# ==============================================================================

import os
import sys
import re
import json
import numpy as np
import torch
from sklearn.ensemble import IsolationForest

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from EMG_desarrollo.deep_learning.pca_umap_clustering import generador_pca_umap as gpu

VOCALES = ['A', 'E', 'I', 'O', 'U']

# Sesiones de oro que cumplen estrictamente la tríada canónica:
SESIONES_CANONICAS = [
    # Lucas
    {"carpeta": "2026-07-10", "sujeto": "Lucas"},
    # Santi
    {"carpeta": "2026-06-22", "sujeto": "Santi"},
    {"carpeta": "2026-06-23", "sujeto": "Santi"},
    # Candela
    {"carpeta": "2026-08-30", "sujeto": "Candela"},
    {"carpeta": "2026-09-01", "sujeto": "Candela"},
    {"carpeta": "2026-09-04", "sujeto": "Candela"},
    {"carpeta": "2026-09-16", "sujeto": "Candela"},
    {"carpeta": "2026-09-18", "sujeto": "Candela"}
]

def filtrar_tomas_validas(base_path):
    todas = sorted(os.listdir(base_path))
    validas = []
    for d in todas:
        d_path = os.path.join(base_path, d)
        if not os.path.isdir(d_path):
            continue
        if d in ["UMBRALES", "SecuenciaContinua", "scratch"]:
            continue
        if "secuenciacontinua" in d.lower():
            continue
        partes = d.split('_')
        if len(partes) > 0 and partes[0].upper() in VOCALES:
            meta0 = os.path.join(d_path, "canal_0", "metadata.json")
            if os.path.exists(meta0):
                validas.append(d)
    return validas

def compilar_dataset():
    db_base = os.path.join(project_root, "EMG_desarrollo", "base_de_datos_electrodos")
    out_dir = os.path.join(project_root, "EMG_desarrollo", "resultados", "resultados_autoencoder", "cache_datasets")
    os.makedirs(out_dir, exist_ok=True)
    out_npz = os.path.join(out_dir, "dataset_multisujeto_canonica.npz")
    
    print("=" * 80)
    print("COMPILANDO DATASET MULTI-SUJETO CANONICO (FASE 1)")
    print(f"Destino: {out_npz}")
    print("=" * 80)
    
    X_list = []
    Y_list = []
    sujetos_list = []
    fechas_list = []
    tomas_list = []
    
    for ses in SESIONES_CANONICAS:
        c_name = ses["carpeta"]
        suj_name = ses["sujeto"]
        c_path = os.path.join(db_base, c_name)
        
        if not os.path.exists(c_path):
            print(f"[Aviso] No existe la carpeta {c_path}. Omitiendo.")
            continue
            
        tomas = filtrar_tomas_validas(c_path)
        print(f"\n[Extrayendo {suj_name} - {c_name}] {len(tomas)} tomas válidas encontradas...")
        
        if len(tomas) == 0:
            continue
            
        X_sub, Y_sub, tomas_wins, _ = gpu.extraer_features_concatenadas(
            base_dir=c_path,
            mediciones=tomas,
            alpha_ruido=1.0,
            gate_ratio_ruido=0.0,
            smooth_ms=90,
            notch_q=2.0,
            target_len=20,
            modo_alineacion="Pico Volumen Micrófono",
            pre_pct=0.50,
            post_pct=0.50,
            canales_features=["canal_0", "canal_1", "canal_2"],
            aplicar_correccion_intersesion=True,
            tipo_envolvente="rms",
            lowpass_cutoff_hz=500.0,
            tipo_filtro_ruido="notch",
            highpass_cutoff_hz=20.0
        )
        
        n_pulses = len(Y_sub)
        print(f"  -> Pulsos extraídos: {n_pulses}")
        
        if n_pulses == 0:
            continue
            
        X_arr_sub = np.array(X_sub, dtype=np.float32).reshape(n_pulses, 3, 20)
        Y_arr_sub = np.array(Y_sub)
        
        X_list.append(X_arr_sub)
        Y_list.append(Y_arr_sub)
        sujetos_list.append(np.array([suj_name] * n_pulses))
        fechas_list.append(np.array([c_name] * n_pulses))
        tomas_list.append(np.array(tomas_wins))
        
    if not X_list:
        print("[Error] No se extrajo ningún pulso.")
        return
        
    X_raw = np.concatenate(X_list, axis=0)
    Y_raw = np.concatenate(Y_list, axis=0)
    sujetos_raw = np.concatenate(sujetos_list, axis=0)
    fechas_raw = np.concatenate(fechas_list, axis=0)
    tomas_raw = np.concatenate(tomas_list, axis=0)
    
    total_raw = len(Y_raw)
    print("\n" + "=" * 80)
    print(f"EXTRACCION CRUDA COMPLETADA: {total_raw} pulsos totales")
    print("=" * 80)
    for suj in np.unique(sujetos_raw):
        m = (sujetos_raw == suj)
        print(f"  {suj:<10}: {np.sum(m)} pulsos")
        
    # Purga de artefactos por sujeto con Isolation Forest (10% contaminación)
    print("\n[Purga de Artefactos] Aplicando Isolation Forest (10%) por sujeto...")
    inliers_mask = np.zeros(total_raw, dtype=bool)
    
    for suj in np.unique(sujetos_raw):
        idx_suj = np.where(sujetos_raw == suj)[0]
        X_suj_flat = X_raw[idx_suj].reshape(len(idx_suj), -1)
        
        iso = IsolationForest(contamination=0.10, random_state=42)
        preds = iso.fit_predict(X_suj_flat)
        inliers_suj = (preds == 1)
        inliers_mask[idx_suj[inliers_suj]] = True
        
        print(f"  {suj:<10}: {np.sum(inliers_suj)} inliers / {len(idx_suj)} ({np.mean(inliers_suj)*100:.1f}%)")
        
    X_clean = X_raw[inliers_mask]
    Y_clean = Y_raw[inliers_mask]
    sujetos_clean = sujetos_raw[inliers_mask]
    fechas_clean = fechas_raw[inliers_mask]
    tomas_clean = tomas_raw[inliers_mask]
    
    total_clean = len(Y_clean)
    print("\n" + "=" * 80)
    print(f"DATASET LIMPIO CONSOLIDADO: {total_clean} pulsos ({total_clean/total_raw*100:.1f}%)")
    print("=" * 80)
    
    # Desglose por sujeto y vocal
    print(f"{'Sujeto':<10} | " + " | ".join([f"/{v.lower()}/" for v in VOCALES]) + " | Total")
    print("-" * 55)
    for suj in np.unique(sujetos_clean):
        m_s = (sujetos_clean == suj)
        v_counts = [np.sum(m_s & (Y_clean == v)) for v in VOCALES]
        row_str = " | ".join([f"{c:>4}" for c in v_counts])
        print(f"{suj:<10} | {row_str} | {np.sum(m_s):>5}")
    print("-" * 55)
    tot_v = [np.sum(Y_clean == v) for v in VOCALES]
    row_tot = " | ".join([f"{c:>4}" for c in tot_v])
    print(f"{'TOTAL':<10} | {row_tot} | {total_clean:>5}")
    
    # Guardar en archivo npz
    np.savez_compressed(
        out_npz,
        X_clean=X_clean,
        Y_clean=Y_clean,
        sujetos_clean=sujetos_clean,
        fechas_clean=fechas_clean,
        tomas_clean=tomas_clean,
        X_raw=X_raw,
        Y_raw=Y_raw,
        sujetos_raw=sujetos_raw,
        fechas_raw=fechas_raw,
        tomas_raw=tomas_raw
    )
    
    print(f"\n[OK] Dataset guardado exitosamente en:\n     {out_npz}")
    print(f"     Dimensiones X_clean: {X_clean.shape}")
    print(f"     Dimensiones Y_clean: {Y_clean.shape}")
    print("=" * 80)

if __name__ == "__main__":
    compilar_dataset()
