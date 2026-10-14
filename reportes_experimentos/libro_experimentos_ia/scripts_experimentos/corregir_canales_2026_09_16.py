# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Corrección e intercambio de Canal 1 (Zygomaticus) y Canal 2 (Orbicularis)
#              en las 20 sesiones de Candela del 2026-09-16.
# ==============================================================================

import os
import shutil
import json
import pandas as pd
import numpy as np

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "base_de_datos_electrodos", "2026-09-16"))

sesiones = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and any(d.startswith(f"{v}_") for v in ['A', 'E', 'I', 'O', 'U'])
])

print(f"Iniciando intercambio de Canal 1 y Canal 2 en {len(sesiones)} sesiones de 2026-09-16...")

for s in sesiones:
    ses_path = os.path.join(base_dir, s)
    ch1_path = os.path.join(ses_path, "canal_1")
    ch2_path = os.path.join(ses_path, "canal_2")
    temp_path = os.path.join(ses_path, "canal_temp_swap")

    if not (os.path.exists(ch1_path) and os.path.exists(ch2_path)):
        print(f"Aviso: Omitiendo {s}, faltan subcarpetas.")
        continue

    # 1. Swap carpetas atómicamente
    os.rename(ch1_path, temp_path)
    os.rename(ch2_path, ch1_path)
    os.rename(temp_path, ch2_path)

    # 2. Actualizar metadata.json en el nuevo canal_1
    meta1_file = os.path.join(ch1_path, "metadata.json")
    if os.path.exists(meta1_file):
        with open(meta1_file, 'r', encoding='utf-8') as f:
            m1 = json.load(f)
        m1["canal"] = "canal_1"
        m1["musculo"] = "Zygomaticus Major"
        m1["physical_channel"] = "Dev1/ai1"
        if "muscles_map" in m1:
            m1["muscles_map"]["canal_1"] = "Zygomaticus Major"
            m1["muscles_map"]["canal_2"] = "Orbicularis Oris"
        with open(meta1_file, 'w', encoding='utf-8') as f:
            json.dump(m1, f, indent=4, ensure_ascii=False)

    # 3. Actualizar metadata.json en el nuevo canal_2
    meta2_file = os.path.join(ch2_path, "metadata.json")
    if os.path.exists(meta2_file):
        with open(meta2_file, 'r', encoding='utf-8') as f:
            m2 = json.load(f)
        m2["canal"] = "canal_2"
        m2["musculo"] = "Orbicularis Oris"
        m2["physical_channel"] = "Dev1/ai2"
        if "muscles_map" in m2:
            m2["muscles_map"]["canal_1"] = "Zygomaticus Major"
            m2["muscles_map"]["canal_2"] = "Orbicularis Oris"
        with open(meta2_file, 'w', encoding='utf-8') as f:
            json.dump(m2, f, indent=4, ensure_ascii=False)

    # 4. Actualizar metadata.json en canal_0 y canal_3
    for ch_other in ["canal_0", "canal_3"]:
        meta_other = os.path.join(ses_path, ch_other, "metadata.json")
        if os.path.exists(meta_other):
            with open(meta_other, 'r', encoding='utf-8') as f:
                mo = json.load(f)
            if "muscles_map" in mo:
                mo["muscles_map"]["canal_1"] = "Zygomaticus Major"
                mo["muscles_map"]["canal_2"] = "Orbicularis Oris"
            if "muscles" in mo:
                mo["muscles"] = ["Anterior Belly", "Zygomaticus Major", "Orbicularis Oris", "Micrófono"]
            with open(meta_other, 'w', encoding='utf-8') as f:
                json.dump(mo, f, indent=4, ensure_ascii=False)

    # 5. Swap de columnas en grabacion.csv
    csv_file = os.path.join(ses_path, "grabacion.csv")
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            if 'Canal 1' in df.columns and 'Canal 2' in df.columns:
                col1_data = df['Canal 1'].copy()
                col2_data = df['Canal 2'].copy()
                df['Canal 1'] = col2_data
                df['Canal 2'] = col1_data
                df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error actualizando grabacion.csv en {s}: {e}")

    # 6. Actualizar campo channel en archivos json precomputados
    for ch_dir, new_ch_num in [(ch1_path, 1), (ch2_path, 2)]:
        for jf in [f for f in os.listdir(ch_dir) if f.endswith('.json') and f != 'metadata.json']:
            j_path = os.path.join(ch_dir, jf)
            try:
                with open(j_path, 'r', encoding='utf-8') as f:
                    jd = json.load(f)
                if isinstance(jd, dict) and "channel" in jd:
                    jd["channel"] = new_ch_num
                    with open(j_path, 'w', encoding='utf-8') as f:
                        json.dump(jd, f, indent=4)
            except Exception:
                pass

    print(f"  [OK] {s}: Canal 1 y Canal 2 intercambiados exitosamente.")

print("Intercambio de canales completado.")
