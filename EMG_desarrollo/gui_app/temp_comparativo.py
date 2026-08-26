
import sys
import os

# --- INYECCIÓN DE SEGURIDAD PARA RESOLUCIÓN DE MÓDULOS (PyInstaller / Nativo) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
  sys.path.insert(0, current_dir)
# -------------------------------------------------------------------------------

from pathlib import Path
import json
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg') # Garantizar ventana comparativa en Windows
sys.path.append("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo")
import analysis.analisis_por_track_integrado as api

mediciones = ['2026-07-10/A_T1_Lucas', '2026-07-10/A_T2_Lucas']
base_dir = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos"
canal = "canal_1"
nombre_custom = "comparacion"

try:
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
    
    out_dir = os.path.join("/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo", "analisis_comparativos", today_str, nombre_carpeta)
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "comparativa.png")
    
    api._comparative_plots(promedios_globales, tiempos_globales, nombres_globales, resultados_globales, out_png, **{'show_overlay': True, 'show_snr': True, 'show_amplitude': True, 'show_snr_time': True, 'show_amp_time': True, 'show_table': True})
    print(f"ANÁLISIS COMPARATIVO FINALIZADO. Guardado en: {out_dir}")
  else:
    print("No hay suficientes resultados válidos cargados.")

except Exception as e:
  import traceback
  print("\n" + "="*50)
  print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL ANÁLISIS")
  print("="*50)
  traceback.print_exc()
finally:
  input("\nPresione ENTER para cerrar esta ventana...")
