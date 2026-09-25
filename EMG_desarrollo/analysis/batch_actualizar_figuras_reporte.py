import os
import sys
import shutil

# Asegurar path
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
emg_desarrollo_dir = os.path.join(repo_root, "EMG_desarrollo")
if emg_desarrollo_dir not in sys.path:
    sys.path.insert(0, emg_desarrollo_dir)

from deep_learning.dataset_tools.plot_3_musculos_standalone import generar_plot_3_musculos
from analysis.plotter_calibrado import plotear_medicion_secuencial
from analysis.generador_figura_multimodal import generar_figura_paper_multimodal

def procesar_sesiones(fechas=["2026-09-23", "2026-09-22"]):
    base_datos = os.path.join(emg_desarrollo_dir, "base_de_datos_electrodos")
    resultados_centrales = os.path.join(emg_desarrollo_dir, "resultados", "figuras_paper_multimodal")
    
    cfg_calib = {
        'notch': True,
        'bandpass': True,
        'tipo_env': 'rms',
        'start_time': None,
        'end_time': None,
        'tema_oscuro': False,
        'graficar_fft': False
    }

    for fecha in fechas:
        sesion_dir = os.path.join(base_datos, fecha)
        if not os.path.isdir(sesion_dir):
            print(f"[Aviso] No existe la sesión: {sesion_dir}")
            continue
            
        tomas = sorted([
            d for d in os.listdir(sesion_dir)
            if os.path.isdir(os.path.join(sesion_dir, d)) and not d.startswith('.') and 'secuencia' not in d.lower()
        ])
        
        print(f"\n=======================================================")
        print(f"Iniciando actualización de figuras para sesión {fecha} ({len(tomas)} tomas)")
        print(f"=======================================================")
        
        for i, toma_name in enumerate(tomas):
            pct = ((i + 1) / len(tomas)) * 100.0
            print(f"\n[Procesando] Toma {i+1}/{len(tomas)} ({pct:.1f}%) - {toma_name}")
            toma_path = os.path.join(sesion_dir, toma_name)
            
            # 1. Regenerar plot_espectrograma_multimodal.png (compacto y ventana [-0.4, 0.4]s)
            dest_spec = os.path.join(toma_path, "plot_espectrograma_multimodal.png")
            try:
                generar_figura_paper_multimodal(toma_path, out_file=dest_spec, pulso_idx=1)
                print(f"  [Espectrograma] Actualizado (compacto [-0.4, 0.4]s): plot_espectrograma_multimodal.png")
            except Exception as e:
                print(f"  [Error Espectrograma] Falló en {toma_name}: {e}")
                
            # 2. Generar plot_paper_combined.png (3 músculos) si no existe
            curr_paper = os.path.join(toma_path, "plot_paper_combined.png")
            if not os.path.exists(curr_paper):
                try:
                    generar_plot_3_musculos(toma_path, theme="light", smooth_ms_val=250.0, frac_pulsos=0.5, mostrar=False)
                    print(f"  [3 Músculos Paper] Generado: plot_paper_combined.png")
                except Exception as e:
                    print(f"  [Error 3 Músculos] Falló en {toma_name}: {e}")
                
            # 3. Generar plot_calibrado_*.png
            try:
                rel_path = f"{fecha}/{toma_name}"
                plotear_medicion_secuencial(rel_path, cfg_calib, mostrar_plot=False)
                print(f"  [Calibrado] Generado plot_calibrado")
            except Exception as e:
                print(f"  [Error Calibrado] Falló en {toma_name}: {e}")

    print("\n[Completado] Actualización de todas las figuras finalizada exitosamente.")

if __name__ == "__main__":
    fechas = sys.argv[1:] if len(sys.argv) > 1 else ["2026-09-23", "2026-09-22"]
    procesar_sesiones(fechas)
