import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    bd_electrodos = base_dir / "base_de_datos_electrodos"
    bd_letras = base_dir / "base_de_datos_letras"
    
    if not bd_electrodos.exists():
        print(f"No existe la carpeta {bd_electrodos}")
        return
        
    bd_letras.mkdir(exist_ok=True)
    
    # Encontrar todas las mediciones (secuencias antiguas y nuevas monocategóricas)
    seq_folders = []
    for f in bd_electrodos.iterdir():
        if f.is_dir() and (f.name.startswith("SECUENCIA_PRUEBA_") or f.name.startswith("SecuenciaContinua_") or "_SC_" in f.name or f.name.startswith("SC_")):
            seq_folders.append(f)
            
    if not seq_folders:
        print("No se encontraron carpetas de secuencias en base_de_datos_electrodos.")
        return
        
    for seq_folder in seq_folders:
        print(f"\nProcesando secuencia: {seq_folder.name}")
        
        # Buscar metadata.json (en raíz o en canal_0)
        meta_path = seq_folder / "metadata.json"
        if not meta_path.exists():
            meta_path = seq_folder / "canal_0" / "metadata.json"
            
        if not meta_path.exists():
            print(f"  [ERROR] No se encontró metadata.json en {seq_folder.name}")
            continue
            
        with open(meta_path, "r", encoding="utf-8") as f:
            try:
                meta_data = json.load(f)
            except Exception as e:
                print(f"  [ERROR] No se pudo leer metadata.json: {e}")
                continue
                
        valid_words = meta_data.get("valid_words", [])
        bpm = meta_data.get("bpm", None)
        noise_seconds = meta_data.get("noise_seconds", 5.0)
        
        # Soportar carpetas monocategóricas generadas por el segmentador
        letra_unica = meta_data.get("letra", None)
        pulse_count = meta_data.get("pulse_count", 0)
        
        if valid_words:
            letters_sequence = valid_words
        elif letra_unica and pulse_count > 0:
            letters_sequence = [letra_unica] * pulse_count
        else:
            print(f"  [ERROR] Faltan 'valid_words' o 'letra' en metadata.json para {seq_folder.name}")
            continue
            
        if not bpm:
            print(f"  [ERROR] Falta 'bpm' en metadata.json para {seq_folder.name}")
            continue
            
        csv_path = seq_folder / "grabacion.csv"
        if not csv_path.exists():
            print(f"  [ERROR] No se encontró grabacion.csv en {seq_folder.name}")
            continue
            
        print(f"  Leyendo {csv_path.name}...")
        df = pd.read_csv(csv_path)
        
        # Verificar canales
        canales_req = ["Canal 0", "Canal 1", "Canal 2"]
        missing_cols = [c for c in canales_req if c not in df.columns]
        if missing_cols:
            print(f"  [ERROR] Faltan las columnas {missing_cols} en el CSV.")
            continue
            
        # Calcular parámetros de recortes
        if "Tiempo (s)" in df.columns and len(df) > 1:
            dt = df["Tiempo (s)"].iloc[1] - df["Tiempo (s)"].iloc[0]
            sample_rate = 1.0 / dt
        else:
            sample_rate = meta_data.get("sample_rate", 2000.0)
            
        muestras_pulso = int(round(sample_rate * 60.0 / bpm))
        start_sample = int(round(noise_seconds * sample_rate))
        
        # Leer SNR de cada canal desde results.json
        snr_canales = {}
        for idx_canal in range(len(canales_req)):
            snr_canales[idx_canal] = None
            res_path = seq_folder / f"canal_{idx_canal}" / "results.json"
            if res_path.exists():
                try:
                    with open(res_path, "r") as f:
                        res = json.load(f)
                        snr_val = res.get("snr_manual")
                        if snr_val is None:
                            snr_val = res.get("snr_mean")
                        snr_canales[idx_canal] = float(snr_val) if snr_val is not None else None
                except Exception as e:
                    pass
        
        # Recortar pulsos
        extracciones = 0
        for i, letter in enumerate(letters_sequence):
            c_start = start_sample + i * muestras_pulso
            c_end = c_start + muestras_pulso
            
            if c_end > len(df):
                print(f"  [ADVERTENCIA] El CSV termina antes de poder extraer el pulso {i} (letra: {letter})")
                break
                
            for idx_canal, canal_col in enumerate(canales_req):
                # FILTRO DE SNR: Si el SNR es menor a 4, ignorar este canal para este pulso
                snr_val = snr_canales[idx_canal]
                if snr_val is not None and snr_val < 4.0:
                    continue
                    
                ventana_canal = df[canal_col].iloc[c_start:c_end].values
                
                # Directorio de salida
                out_dir = bd_letras / str(letter) / f"canal_{idx_canal}"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Nombre de archivo
                out_name = f"{seq_folder.name}_pulso_{i:03d}.npy"
                out_path = out_dir / out_name
                
                np.save(out_path, ventana_canal)
            extracciones += 1
                
        print(f"  -> Extracción completa para {seq_folder.name}: {extracciones} pulsos guardados.")

if __name__ == "__main__":
    main()
