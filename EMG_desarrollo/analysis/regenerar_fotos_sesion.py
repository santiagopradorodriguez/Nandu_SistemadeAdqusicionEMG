# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Utilidad para regenerar en lote los gráficos photo.png con señal cruda normalizada y estética de plotter calibrado.
# ==============================================================================

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Asegurar que EMG_desarrollo y raíz estén en sys.path
CURRENT_DIR = Path(__file__).resolve().parent
DEV_DIR = CURRENT_DIR.parent
ROOT_DIR = DEV_DIR.parent
if str(DEV_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from acquisition.autoforge_daq_experimental import generar_grafico_grabacion

def regenerar_fotos_sesiones(fecha: str = "2026-09-16", base_db: str = None):
    """
    Regenera los archivos photo.png para todas las sesiones grabadas en la fecha indicada,
    aplicando la señal 100% cruda, normalizada por supremo tricanal y con estética de plotter_calibrado.
    """
    np.random.seed(42)
    if base_db is None:
        base_db = DEV_DIR / "base_de_datos_electrodos"
    else:
        base_db = Path(base_db)

    fecha_dir = base_db / fecha
    if not fecha_dir.exists():
        print(f"Error: El directorio de fecha {fecha_dir} no existe.")
        return False

    sesiones = sorted([d for d in os.listdir(fecha_dir) if (fecha_dir / d).is_dir()])
    total = len(sesiones)
    if total == 0:
        print(f"No se encontraron sesiones en {fecha_dir}")
        return False

    print(f"Iniciando regeneración de photo.png para {total} sesiones de la fecha {fecha}...")
    exitos = 0
    errores = 0

    for idx, sesion in enumerate(sesiones):
        sesion_path = fecha_dir / sesion
        csv_path = sesion_path / "grabacion.csv"
        
        print(f"[Procesando] Sesión {idx+1}/{total} ({((idx+1)/total)*100:.1f}%) - {sesion}")

        if not csv_path.exists():
            print(f"  [Aviso] No se encontró grabacion.csv en {sesion}, buscando archivos WAV por canal...")
            # Si no hay CSV, intentar leer WAVs
            wav_channels = []
            fs_val = 2000.0
            for c_idx in range(4):
                ch_wav = sesion_path / f"canal_{c_idx}" / "grabacion.wav"
                if ch_wav.exists():
                    try:
                        from scipy.io import wavfile
                        fs_val, data_w = wavfile.read(str(ch_wav))
                        wav_channels.append(data_w.astype(np.float64))
                    except Exception as e_w:
                        print(f"  [Error] No se pudo leer {ch_wav}: {e_w}")
                        break
            if len(wav_channels) >= 3:
                datos_arr = np.array(wav_channels)
                cols_canales = [f"Canal {i}" for i in range(len(wav_channels))]
                res = generar_grafico_grabacion(
                    datos_completos=datos_arr,
                    sample_rate=fs_val,
                    output_dir=str(sesion_path),
                    num_canales=len(wav_channels),
                    canales_daq=cols_canales,
                    base_name="photo"
                )
                if res:
                    exitos += 1
                else:
                    errores += 1
            else:
                print(f"  [Error] No hay datos válidos para {sesion}")
                errores += 1
            continue

        try:
            df = pd.read_csv(csv_path)
            col_tiempo = df.columns[0]
            cols_canales = [col for col in df.columns if "Canal" in col or "Dev" in col]
            
            if not cols_canales:
                print(f"  [Error] No se encontraron columnas de canales en {csv_path}")
                errores += 1
                continue

            # Frecuencia de muestreo
            try:
                fs_val = 1.0 / float(df[col_tiempo].iloc[1] - df[col_tiempo].iloc[0])
            except Exception:
                fs_val = 2000.0

            # Matriz de datos crudos en Volts (num_canales, num_muestras)
            datos_canales = np.array([df[c].values for c in cols_canales], dtype=np.float64)

            # Generar gráfico photo.png
            res = generar_grafico_grabacion(
                datos_completos=datos_canales,
                sample_rate=fs_val,
                output_dir=str(sesion_path),
                num_canales=len(cols_canales),
                canales_daq=cols_canales,
                base_name="photo"
            )

            if res:
                exitos += 1
            else:
                errores += 1

        except Exception as e_proc:
            print(f"  [Error] Fallo al procesar {sesion}: {e_proc}")
            errores += 1

    print(f"\nRegeneración finalizada: {exitos} exitosas, {errores} fallidas de un total de {total} sesiones.")
    return exitos > 0

if __name__ == "__main__":
    fecha_target = sys.argv[1] if len(sys.argv) > 1 else "2026-09-16"
    regenerar_fotos_sesiones(fecha_target)
