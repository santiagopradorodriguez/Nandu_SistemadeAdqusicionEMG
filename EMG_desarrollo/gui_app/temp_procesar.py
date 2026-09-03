
import sys
import os
import tkinter as tk
import matplotlib
matplotlib.use('TkAgg')
sys.path.append(r"/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo")
import analysis.analisis_por_track_integrado as api

mediciones = ['2026-09-01/A_Prueba1_Candela', '2026-09-01/A_Prueba2_Candela', '2026-09-01/A_Prueba3_Candela', '2026-09-01/A_Prueba4_Candela', '2026-09-01/E_Prueba1_Candela', '2026-09-01/E_Prueba2_Candela', '2026-09-01/E_Prueba3_Candela', '2026-09-01/E_Prueba4_Candela', '2026-09-01/E_Prueba5_Candela', '2026-09-01/I_Prueba1_Candela', '2026-09-01/I_Prueba2_Candela', '2026-09-01/I_Prueba3_Candela', '2026-09-01/I_Prueba4_Candela', '2026-09-01/O_Prueba1_Candela', '2026-09-01/O_Prueba2_Candela', '2026-09-01/O_Prueba3_Candela', '2026-09-01/O_Prueba4_Candela', '2026-09-01/O_Prueba5_Candela', '2026-09-01/U_Prueba1_Candela', '2026-09-01/U_Prueba2_Candela', '2026-09-01/U_Prueba3_Candela', '2026-09-01/U_Prueba4_Candela']
base_dir = r"/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/base_de_datos_electrodos"

try:
  root = tk.Tk()
  root.withdraw()
  dialog = api.ProcessingOptionsDialog(root)
  dialog.populate_channels(base_dir, mediciones)

  # Trasplantar la selección de canales de PySide6 a Tkinter
  canales_elegidos = ['canal_0', 'canal_1', 'canal_2', 'canal_3']
  for canal_key, var in dialog.canales_seleccionados.items():
    var.set(canal_key in canales_elegidos)

  # Inyectar los parámetros de nuestra GUI PySide6 a su GUI Tkinter
  dialog.var_mostrar_recortes.set(True)
  dialog.var_mostrar_senal_cruda.set(False)
  dialog.var_mostrar_espectrograma.set(False)
  dialog.var_frecuenciamaxima.set("5000")
  dialog.var_notch_filter.set(True)
  dialog.var_notch_q_factor.set("2.0")
  dialog.var_mostrar_evolucion.set(True)
  dialog.var_evol_t_start.set("10")
  dialog.var_evol_t_end.set("1000")
  dialog.var_smooth_ms.set("75")
  dialog.var_tipo_env.set("rms")
  dialog.var_highpass_cutoff.set("20")
  dialog.var_lowpass_cutoff.set("500")
  if hasattr(dialog, 'var_cyberpunk'):
    dialog.var_cyberpunk.set(True)

  excl_list = []
  excl_str = ",".join(map(str, excl_list)) if excl_list else ""
  dialog.var_excluded_windows.set(excl_str)

  print("\n> Orquestador Tkinter Aislado Inicializado. Ejecutando Rutina original de ProcessingOptionsDialog...")
  # Ejecutar su propia rutina que ya maneja pop-ups, metadatos y curación
  dialog.procesar(interactivo=False)

except Exception as e:
  import traceback
  print("\n" + "="*50)
  print(" OCURRIÓ UN ERROR CRÍTICO DURANTE EL PROCESAMIENTO")
  print("="*50)
  traceback.print_exc()
finally:
  input("\nPresione ENTER para cerrar esta ventana...")
