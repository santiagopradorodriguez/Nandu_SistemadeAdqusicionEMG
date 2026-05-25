import re
import os

filepath = r"c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\Emg\analisis_por_track_integrado_experimental.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Renombrar procesar_wavs_promedio a _procesar_un_smooth
# y añadir un parámetro skip_plots=True
content = content.replace("def procesar_wavs_promedio(", "def _procesar_un_smooth(")
# Buscar donde hace _plot_pulse_full y condicionarlo a skip_plots
content = re.sub(r"(# --- GRAFICO: pulsos individuales y promedio.*?_plot_pulse_full\()", 
                 r"if not skip_plots:\n            \1", content, flags=re.DOTALL)
# Buscar donde hace _plot_espectro_and_spectrogram
content = re.sub(r"(if mostrar_espectrograma:\n\s+_plot_espectro_and_spectrogram\()", 
                 r"if mostrar_espectrograma and not skip_plots:\n            _plot_espectro_and_spectrogram(", content)
# Buscar donde hace _plot_recortes
content = re.sub(r"(interactive_excluded = excluded_windows\n\s+if mostrar_recortes:\n\s+interactive_excluded = _plot_recortes\()", 
                 r"interactive_excluded = excluded_windows\n        if mostrar_recortes and not skip_plots:\n            interactive_excluded = _plot_recortes(", content)
# Buscar donde hace _plot_evolucion_temporal
content = re.sub(r"(if mostrar_evolucion:\n\s+_plot_evolucion_temporal\()", 
                 r"if mostrar_evolucion and not skip_plots:\n            _plot_evolucion_temporal(", content)
# Añadir el argumento skip_plots en _procesar_un_smooth
content = content.replace("evol_t_end=100.0\n):", "evol_t_end=100.0,\n    skip_plots=False\n):")

# 2. Agregar las nuevas funciones de plot y el nuevo procesar_wavs_promedio
nuevas_funciones = """
import matplotlib.pyplot as plt

def _plot_pulse_full_experimental(dict_resultados, filename, out_prom, show_plot=False):
    plt.figure(figsize=(12, 8))
    colors = {5: 'blue', 25: 'green', 50: 'red'}
    
    for smooth, res in dict_resultados.items():
        t_pulso = res['pulse_time']
        pulso_promedio = res['mean_pulse']
        pulso_err = res.get('std_pulse', 0) / np.sqrt(len(res.get('segmentos_norm', [1])))
        snr = res.get('snr_manual', 0)
        
        plt.plot(t_pulso, pulso_promedio, color=colors.get(smooth, 'black'), linewidth=2,
                 label=f"Promedio {smooth}ms (SNR={snr:.2f})")
        plt.fill_between(t_pulso,
                         pulso_promedio - pulso_err,
                         pulso_promedio + pulso_err,
                         color=colors.get(smooth, 'black'),
                         alpha=0.15)
                         
    plt.title(f"Pulso promedio comparativo - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [µV]")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(out_prom, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()

def _plot_recortes_experimental(dict_resultados, filename, out_rec, show_plot=False):
    plt.figure(figsize=(12, 6))
    colors = {5: 'blue', 25: 'green', 50: 'red'}
    
    first_res = list(dict_resultados.values())[0]
    t_recortada = first_res['t_recortada']
    signal_recortada = first_res['signal_recortada']
    
    plt.plot(t_recortada, signal_recortada, color="black", linewidth=1.0, alpha=0.4, label="Señal procesada")
    
    for smooth, res in dict_resultados.items():
        env = res['env_recortada']
        plt.plot(t_recortada, env, color=colors.get(smooth, 'black'), linewidth=1.5, alpha=0.8, label=f"Envolvente {smooth}ms")
        
    plt.title(f"Envolventes comparativas - {filename}")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud [µV]")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')
    plt.savefig(out_rec, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()

def procesar_wavs_promedio(
    carpeta, bpm=50, colorgrafico="blue", tiempoinicial=0, tiempofinal=25,
    nombre_salida="resultado_promedio.png", mostrar_individuales=True,
    mostrar_recortes=True, mostrar_espectrograma=True, frecuenciamaxima=1000,
    frecuenciaminima=0, colores_aleatorios=False, seed=None,
    espectrograma_db=False, calcular_umbral=True, metodo_umbral='outside_windows',
    factor_umbral=6, mostrar_umbral=True, mostrar_tabla=True,
    usar_picos=True, peak_prominence=None, peak_height=None,
    peak_distance_sec=0.4, pre_window_sec=None, post_window_sec=None,
    normalize_by='rms', resample_len=None, one_max_per_cut=True,
    n_pulsos_manual=None, fixed_umbral_abs=0.5, apply_envelope=True,
    smooth_ms=50, noise_seconds=2, excluded_windows=None,
    peak_search_threshold=0.25, plot_mode='mean', individual_alpha=0.25,
    lowpass_cutoff_hz=500.0, highpass_cutoff_hz=20.0,
    output_root="", display_name_for_plot="", show_interactive_plot=False,
    show_average_plot=False, apply_notch_filter=False,
    mostrar_evolucion=False, evol_t_start=25.0, evol_t_end=100.0
):
    # La nueva funcion principal iterara por 3 valores de smooth
    smooths = [5, 25, 50]
    
    # Obtener lista de archivos
    archivos = [f for f in os.listdir(carpeta) if f.lower().endswith(".wav")]
    if not archivos: return {}
    
    for filename in archivos:
        dict_resultados_smooth = {}
        for s in smooths:
            print(f"\\n--- Procesando {filename} con filtro {s}ms ---")
            res_s = _procesar_un_smooth(
                carpeta=carpeta, bpm=bpm, colorgrafico=colorgrafico,
                tiempoinicial=tiempoinicial, tiempofinal=tiempofinal,
                nombre_salida=nombre_salida, mostrar_individuales=mostrar_individuales,
                mostrar_recortes=mostrar_recortes, mostrar_espectrograma=mostrar_espectrograma,
                frecuenciamaxima=frecuenciamaxima, frecuenciaminima=frecuenciaminima,
                colores_aleatorios=colores_aleatorios, seed=seed,
                espectrograma_db=espectrograma_db, calcular_umbral=calcular_umbral,
                metodo_umbral=metodo_umbral, factor_umbral=factor_umbral,
                mostrar_umbral=mostrar_umbral, mostrar_tabla=mostrar_tabla,
                usar_picos=usar_picos, peak_prominence=peak_prominence,
                peak_height=peak_height, peak_distance_sec=peak_distance_sec,
                pre_window_sec=pre_window_sec, post_window_sec=post_window_sec,
                normalize_by=normalize_by, resample_len=resample_len,
                one_max_per_cut=one_max_per_cut, n_pulsos_manual=n_pulsos_manual,
                fixed_umbral_abs=fixed_umbral_abs, apply_envelope=apply_envelope,
                smooth_ms=s, noise_seconds=noise_seconds, excluded_windows=excluded_windows,
                peak_search_threshold=peak_search_threshold, plot_mode=plot_mode,
                individual_alpha=individual_alpha, lowpass_cutoff_hz=lowpass_cutoff_hz,
                highpass_cutoff_hz=highpass_cutoff_hz, output_root=output_root,
                display_name_for_plot=display_name_for_plot, show_interactive_plot=False,
                show_average_plot=False, apply_notch_filter=apply_notch_filter,
                mostrar_evolucion=mostrar_evolucion, evol_t_start=evol_t_start,
                evol_t_end=evol_t_end, skip_plots=True
            )
            # Recuperar el diccionario de resultados de este archivo (devuelve un dict de dicts)
            if filename in res_s:
                dict_resultados_smooth[s] = res_s[filename]
                
        if dict_resultados_smooth:
            # Ahora graficamos los 3
            out_dir = output_root
            final_plot_title = display_name_for_plot or filename
            out_prom = os.path.join(out_dir, "avg_experimental.png")
            out_rec = os.path.join(out_dir, "pulses_experimental.png")
            
            _plot_pulse_full_experimental(dict_resultados_smooth, final_plot_title, out_prom, show_plot=show_average_plot)
            if mostrar_recortes:
                _plot_recortes_experimental(dict_resultados_smooth, final_plot_title, out_rec, show_plot=False)
                
    return {}
"""

content = content.replace("class ProcessingOptionsDialog(tk.Toplevel):", nuevas_funciones + "\n\nclass ProcessingOptionsDialog(tk.Toplevel):")

# 3. Remover entrada de smooth_ms en el GUI porque ahora es automático
content = re.sub(r"self\.var_smooth_ms = tk\.StringVar\(value=\"50\"\)\n", "", content)
content = re.sub(r"smooth_frame = tk\.Frame\(individual_plots_frame\).*?tk\.Entry\(smooth_frame, textvariable=self\.var_smooth_ms.*?\)\.pack\(side=\"left\", padx=\(5,0\)\)\n", "", content, flags=re.DOTALL)
# Quitarlo tb en try except
content = re.sub(r"smooth_val = float\(self\.var_smooth_ms\.get\(\)\)\n", "smooth_val = 50.0\n", content)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Patch aplicado correctamente")
