
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)

from pathlib import Path
import json
import re
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
sys.path.append("c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo")
import analysis.analisis_por_track_integrado as api

mediciones_a_comparar = ['2026-06-03/SC_VOCALESAUTK1_Lucas', '2026-06-03/SC_VOCALESAUTK2_Lucas', '2026-06-03/SC_VOCALESAUTK3_Lucas', '2026-06-03/SC_VOCALESAUTK4_Lucas', '2026-06-03/SC_VOCALESAUTK5_Lucas']
base_dir = "c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo/base_de_datos_electrodos"
nombre_custom = ""

try:
    mediciones_data = []
    for nombre_medicion in mediciones_a_comparar:
        path_medicion = os.path.join(base_dir, nombre_medicion)
        folder_name = os.path.basename(path_medicion)
        
        letra_match = re.match(r'^([AEIOUaeiou])_', folder_name)
        letra = letra_match.group(1).upper() if letra_match else '?'
        
        dt_obj = None
        hora_str = ""
        pulse_count = 0
        canales_data = {}
        
        for ch_idx in [0, 1, 2]:
            ch_key = f'canal_{ch_idx}'
            ch_path = os.path.join(path_medicion, ch_key)
            if not os.path.exists(ch_path): continue
            
            res_path = os.path.join(ch_path, 'analisis_results.json')
            meta_path = os.path.join(ch_path, 'metadata.json')
            
            if not os.path.exists(res_path): continue
            
            with open(res_path, 'r') as f:
                res = json.load(f)
                
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                    if dt_obj is None:
                        mdate = meta.get('measurement_date', '')
                        if mdate:
                            try:
                                dt_obj = datetime.fromisoformat(mdate)
                                hora_str = dt_obj.strftime("%H:%M:%S")
                            except:
                                pass
                                
                    if pulse_count == 0:
                        pulse_count = meta.get('pulse_count', 0)
            
            snr_per_pulse = []
            segmentos_rs = res.get('segmentos_rs', [])
            if not isinstance(segmentos_rs, list):
                segmentos_rs = []
                
            umbral = res.get('umbral', None)
            
            amp_per_pulse = []
            if isinstance(segmentos_rs, list) and len(segmentos_rs) > 0:
                for p in segmentos_rs:
                    if isinstance(p, list) and len(p) > 0:
                        mav_val = float(np.mean(np.abs(p)))
                        amp_per_pulse.append(mav_val)
                        if umbral and umbral > 0:
                            snr_per_pulse.append(mav_val / umbral)
                        else:
                            snr_per_pulse.append(np.nan)
                    else:
                        amp_per_pulse.append(np.nan)
                        snr_per_pulse.append(np.nan)
            else:
                amp_per_pulse = [np.nan] * len(snr_per_pulse)
                
            canales_data[ch_key] = {
                'snr': snr_per_pulse,
                'amp': amp_per_pulse
            }
            
        if dt_obj is None:
            dt_obj = datetime.now()
            hora_str = "??.??"
            
        mediciones_data.append({
            'folder_name': folder_name,
            'letra': letra,
            'dt_obj': dt_obj,
            'hora_str': hora_str,
            'pulse_count': pulse_count,
            'canales': canales_data
        })
        
    mediciones_data.sort(key=lambda x: x['dt_obj'])
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    
    nombre_carpeta = nombre_custom if nombre_custom else f"Sesion_{timestamp}"
    output_comp_dir = os.path.join("c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo", "analisis_de_sesiones", today_str, nombre_carpeta)
    os.makedirs(output_comp_dir, exist_ok=True)
    
    nombre_salida_base = os.path.join(output_comp_dir, "Sesion")
    
    api._comparative_session_plots(mediciones_data, nombre_salida_base)

except Exception as e:
    import traceback
    print("\n" + "="*50)
    print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL ANÁLISIS DE SESIÓN")
    print("="*50)
    traceback.print_exc()
finally:
    input("\nPresione ENTER para cerrar esta ventana...")
