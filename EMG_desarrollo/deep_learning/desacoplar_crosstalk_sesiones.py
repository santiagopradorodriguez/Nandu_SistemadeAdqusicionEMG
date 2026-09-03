# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Desacople de crosstalk electromiográfico (Orbicular -> Cigomático)
#              Genera una carpeta de base de datos deslinealizada compatible con la GUI
# ==============================================================================

import os
import sys
import shutil
import json
import numpy as np
from scipy.io import wavfile

def estimar_beta_crosstalk(base_dir, fecha="2026-09-01"):
    """
    Estima el factor beta de fuga del Orbicular (Ch1) al Cigomático (Ch2)
    utilizando las tomas de la vocal /u/, donde el cigomático permanece pasivo.
    """
    ruta_fecha = os.path.join(base_dir, fecha)
    if not os.path.exists(ruta_fecha):
        print(f"[ERROR] No existe la ruta: {ruta_fecha}")
        return 0.25

    ratios = []
    for sesion in sorted(os.listdir(ruta_fecha)):
        if any(k in sesion for k in ["Secuencia", "secuencia", "Continua", "continua"]):
            continue
        if not sesion.startswith("U_") and not sesion.startswith("u_"):
            continue
        
        path_ch1 = os.path.join(ruta_fecha, sesion, "canal_1", "grabacion.wav")
        path_ch2 = os.path.join(ruta_fecha, sesion, "canal_2", "grabacion.wav")
        
        if os.path.exists(path_ch1) and os.path.exists(path_ch2):
            try:
                fs1, data1 = wavfile.read(path_ch1)
                fs2, data2 = wavfile.read(path_ch2)
                
                # Normalizar a float
                d1 = data1.astype(np.float64) / (32768.0 if data1.dtype == np.int16 else 1.0)
                d2 = data2.astype(np.float64) / (32768.0 if data2.dtype == np.int16 else 1.0)
                
                min_len = min(len(d1), len(d2))
                d1, d2 = d1[:min_len], d2[:min_len]
                
                rms1 = np.sqrt(np.mean(d1**2))
                rms2 = np.sqrt(np.mean(d2**2))
                
                if rms1 > 1e-5:
                    ratio = rms2 / rms1
                    ratios.append(ratio)
                    print(f"  [Calibración /u/] {sesion:30s} -> RMS Ch1: {rms1:.5f} | RMS Ch2: {rms2:.5f} | Beta: {ratio:.4f}")
            except Exception as e:
                print(f"  [Advertencia] Error leyendo {sesion}: {e}")

    if len(ratios) > 0:
        beta_opt = float(np.median(ratios))
        print(f"\n=> Beta mediano estimado en /u/: {beta_opt:.4f} (media: {np.mean(ratios):.4f})")
        return beta_opt
    else:
        print("\n=> No se encontraron tomas de /u/, usando valor por defecto beta=0.25")
        return 0.25

def procesar_desacople_dataset(base_dir, fecha_origen="2026-09-01", fecha_destino="2026-09-01_deslinealizadas", beta=None):
    ruta_origen = os.path.join(base_dir, fecha_origen)
    ruta_destino = os.path.join(base_dir, fecha_destino)
    
    if not os.path.exists(ruta_origen):
        print(f"[ERROR] Ruta de origen no encontrada: {ruta_origen}")
        return

    if beta is None:
        print(f"=== PASO 1: ESTIMACIÓN DE BETA EN TOMAS /U/ ({fecha_origen}) ===")
        beta = estimar_beta_crosstalk(base_dir, fecha_origen)

    print(f"\n=== PASO 2: DESACOPLANDO CANAL 2 CON BETA = {beta:.4f} ===")
    os.makedirs(ruta_destino, exist_ok=True)

    sesiones_procesadas = 0
    for sesion in sorted(os.listdir(ruta_origen)):
        dir_sesion_in = os.path.join(ruta_origen, sesion)
        if not os.path.isdir(dir_sesion_in):
            continue
        if any(k in sesion for k in ["Secuencia", "secuencia", "Continua", "continua"]):
            continue

        dir_sesion_out = os.path.join(ruta_destino, f"{sesion}_deslinealizada")
        os.makedirs(dir_sesion_out, exist_ok=True)

        # Copiar y procesar cada canal
        for ch in ["canal_0", "canal_1", "canal_2", "canal_3"]:
            dir_ch_in = os.path.join(dir_sesion_in, ch)
            dir_ch_out = os.path.join(dir_sesion_out, ch)
            os.makedirs(dir_ch_out, exist_ok=True)

            # Copiar metadata.json si existe
            meta_in = os.path.join(dir_ch_in, "metadata.json")
            if os.path.exists(meta_in):
                meta_out = os.path.join(dir_ch_out, "metadata.json")
                shutil.copy2(meta_in, meta_out)

        # Procesar audios
        path_wav0_in = os.path.join(dir_sesion_in, "canal_0", "grabacion.wav")
        path_wav1_in = os.path.join(dir_sesion_in, "canal_1", "grabacion.wav")
        path_wav2_in = os.path.join(dir_sesion_in, "canal_2", "grabacion.wav")
        path_wav3_in = os.path.join(dir_sesion_in, "canal_3", "grabacion.wav")

        # Canal 0 y Canal 3 se copian intactos
        if os.path.exists(path_wav0_in):
            shutil.copy2(path_wav0_in, os.path.join(dir_sesion_out, "canal_0", "grabacion.wav"))
        if os.path.exists(path_wav1_in):
            shutil.copy2(path_wav1_in, os.path.join(dir_sesion_out, "canal_1", "grabacion.wav"))
        if os.path.exists(path_wav3_in):
            shutil.copy2(path_wav3_in, os.path.join(dir_sesion_out, "canal_3", "grabacion.wav"))

        # Canal 2 se desacopla: s2_clean = s2 - beta * s1
        if os.path.exists(path_wav1_in) and os.path.exists(path_wav2_in):
            try:
                fs1, data1 = wavfile.read(path_wav1_in)
                fs2, data2 = wavfile.read(path_wav2_in)
                
                min_len = min(len(data1), len(data2))
                d1 = data1[:min_len].astype(np.float64)
                d2 = data2[:min_len].astype(np.float64)
                
                # Resta de crosstalk
                d2_clean = d2 - beta * d1
                
                # Preservar formato original (int16 o float32)
                if data2.dtype == np.int16:
                    d2_clean = np.clip(d2_clean, -32768, 32767).astype(np.int16)
                else:
                    d2_clean = d2_clean.astype(data2.dtype)
                    
                wavfile.write(os.path.join(dir_sesion_out, "canal_2", "grabacion.wav"), fs2, d2_clean)
            except Exception as e:
                print(f"  [ERROR] Procesando audio Ch2 en {sesion}: {e}")
        elif os.path.exists(path_wav2_in):
            shutil.copy2(path_wav2_in, os.path.join(dir_sesion_out, "canal_2", "grabacion.wav"))

        sesiones_procesadas += 1
        print(f"  [PROCESADA] {sesion} -> {sesion}_deslinealizada")

    print(f"\n[EXITO] Total sesiones deslinealizadas creadas: {sesiones_procesadas}")
    print(f"[UBICACIÓN] {os.path.abspath(ruta_destino)}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emg_desarrollo_dir = os.path.dirname(script_dir)
    db_base = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos")
    
    fecha_src = sys.argv[1] if len(sys.argv) > 1 else "2026-09-01"
    fecha_dst = sys.argv[2] if len(sys.argv) > 2 else f"{fecha_src}_deslinealizadas"
    
    procesar_desacople_dataset(db_base, fecha_src, fecha_dst)
