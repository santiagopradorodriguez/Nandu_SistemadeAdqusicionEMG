# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Script temporal/de soporte para la vista comparativa.
# ==============================================================================

import sys
import os
import json
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg') # Garantizar ventana comparativa en Windows
sys.path.append("c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/Emg")
import analisis_por_track_integrado as api

mediciones = ['2026-05-20/_E2_1_TRENZADOMALLADOGND_Sujeto1', '2026-05-20/_E2_2_MALLADOGND_Sujeto1', '2026-05-20/_E2_3_MALLADOGND_Sujeto1', '2026-05-20/_E2_4_TRENZADOMALLADOGND_Sujeto1']
base_dir = "c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/Emg/base_de_datos_electrodos"
canal = "canal_0"
nombre_custom = "EXPERIMENTO2"

resultados_globales = {}
for med in mediciones:
    clave = f"{med}-{canal}"
    path = os.path.join(base_dir, med, canal, 'analisis_results.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            res = json.load(f)
            
        # Intentar metadata
        meta_path = os.path.join(base_dir, med, canal, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as fm:
                md = json.load(fm)
                if 'measurement_date' not in res: res['measurement_date'] = md.get('measurement_date','')
                if 'comentario' not in res: res['comentario'] = md.get('comentario','')
                
        res['file'] = clave
        resultados_globales[clave] = res
    except Exception as e:
        print(f"Error cargando {path}: {e}")

if len(resultados_globales) > 1:
    promedios_globales = [res['mean_pulse'] for res in resultados_globales.values() if 'mean_pulse' in res]
    tiempos_globales = [res['pulse_time'] for res in resultados_globales.values() if 'pulse_time' in res]
    nombres_globales = [res['file'] for res in resultados_globales.values() if 'file' in res]
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    nombre_carpeta = nombre_custom if nombre_custom else f"comparacion_{timestamp}"
    
    out_dir = os.path.join("c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/Emg", "analisis_comparativos", today_str, nombre_carpeta)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "comparativa.png")
    
    api._comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados_globales, out_png, **{'show_overlay': True, 'show_snr': True, 'show_amplitude': True, 'show_table': True, 'show_snr_time': True, 'show_amp_time': True})
    print(f"ANÁLISIS COMPARATIVO FINALIZADO. Guardado en: {out_dir}")
else:
    print("No hay suficientes resultados válidos cargados.")
