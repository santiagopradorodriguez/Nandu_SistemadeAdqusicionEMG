
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
sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo")
import analysis.analisis_por_track_integrado as api

mediciones_a_comparar = ['2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela']
base_dir = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos"
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
        muscles_map = {}
        meta0_path = os.path.join(path_medicion, 'canal_0', 'metadata.json')
        if not os.path.exists(meta0_path):
            meta0_path = os.path.join(path_medicion, 'metadata.json')
        if os.path.exists(meta0_path):
            try:
                with open(meta0_path, 'r', encoding='utf-8') as f0:
                    m0 = json.load(f0)
                    if 'muscles_map' in m0:
                        muscles_map = m0['muscles_map']
                    elif 'muscles' in m0 and isinstance(m0['muscles'], list):
                        muscles_map = {i: m for i, m in enumerate(m0['muscles'])}
            except Exception:
                pass
        
        for ch_idx in [0, 1, 2]:
            ch_key = f'canal_{ch_idx}'
            ch_path = os.path.join(path_medicion, ch_key)
            if not os.path.exists(ch_path): continue
            
            res_path = os.path.join(ch_path, 'analisis_results.json')
            meta_path = os.path.join(ch_path, 'metadata.json')
            
            if not os.path.exists(res_path): continue
            
            try:
                with open(res_path, 'r') as f:
                    raw_text = f.read()
                try:
                    res = json.loads(raw_text)
                except json.JSONDecodeError:
                    # Archivo con datos extra al final: intentar decodear solo el primer objeto JSON
                    decoder = json.JSONDecoder()
                    res, _ = decoder.raw_decode(raw_text)
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo leer {res_path}: {e}. Omitiendo canal.")
                continue
                
            ch_musculo = ""
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    ch_musculo = meta.get('musculo', '')
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
            if not ch_musculo and muscles_map:
                ch_musculo = muscles_map.get(str(ch_idx), muscles_map.get(ch_idx, ''))
            
            snr_per_pulse = []
            segmentos_rs = res.get('segmentos_rs', [])
            if not isinstance(segmentos_rs, list):
                segmentos_rs = []
                
            umbral = res.get('umbral', None)
            picos_ventana = res.get('picos_ventana', [])
            
            amp_per_pulse = []
            if isinstance(segmentos_rs, list) and len(segmentos_rs) > 0:
                for p in segmentos_rs:
                    if isinstance(p, list) and len(p) > 0:
                        p_arr = np.array(p)
                        # Resta de offset basal robusto como en plotter_calibrado
                        q25, q75 = np.percentile(p_arr, [25, 75])
                        iqr = q75 - q25
                        clean_base = p_arr[p_arr <= q75 + 1.5 * iqr]
                        p_base = np.percentile(clean_base, 10) if len(clean_base) >= 5 else np.min(p_arr)
                        p_clean = np.maximum(0.0, p_arr - p_base)
                        
                        amp_val = float(np.max(p_clean))
                        mav_val = float(np.mean(p_clean))
                        amp_per_pulse.append(amp_val)
                        if umbral and umbral > 0:
                            snr_per_pulse.append(mav_val / umbral)
                        else:
                            snr_per_pulse.append(np.nan)
                    else:
                        amp_per_pulse.append(np.nan)
                        snr_per_pulse.append(np.nan)
            elif isinstance(picos_ventana, list) and len(picos_ventana) > 0:
                for pv in picos_ventana:
                    if pv is not None and not np.isnan(pv):
                        amp_per_pulse.append(float(pv))
                        if umbral and umbral > 0:
                            snr_per_pulse.append(float(pv) / umbral)
                        else:
                            snr_per_pulse.append(np.nan)
                    else:
                        amp_per_pulse.append(np.nan)
                        snr_per_pulse.append(np.nan)
            else:
                amp_per_pulse = [np.nan] * len(snr_per_pulse)
                
            canales_data[ch_key] = {
                'snr': snr_per_pulse,
                'amp': amp_per_pulse,
                'musculo': ch_musculo
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
            'muscles_map': muscles_map,
            'canales': canales_data
        })
        
    mediciones_data.sort(key=lambda x: x['dt_obj'])
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H%M%S")
    
    nombre_carpeta = nombre_custom if nombre_custom else f"Sesion_{timestamp}"
    output_comp_dir = os.path.join("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo", "analisis_de_sesiones", today_str, nombre_carpeta)
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
