import re
import os

filepath = r"c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\Emg\analisis_por_track_integrado_experimental.py"

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

nuevas_funciones_2 = """
def _plot_evolucion_temporal_experimental(dict_resultados, filename, out_path, t_start, t_end, show_plot=False):
    import numpy as np
    plt.figure(figsize=(15, 6))
    colors = {5: 'blue', 25: 'green', 50: 'red'}
    
    plt.suptitle(f"Evolución Temporal Comparativa: SNR y Ruido - {filename}", fontsize=14)
    
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    for smooth, res in dict_resultados.items():
        if 'stats_time' not in res or 'stats_snr' not in res:
            continue
        t_arr = np.array(res['stats_time'])
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        if not np.any(mask): mask = np.ones_like(t_arr, dtype=bool)
        
        t_plot = t_arr[mask]
        snr_plot = np.array(res['stats_snr'])[mask]
        noise_mean_plot = np.array(res['stats_noise_mean'])[mask]
        
        ax1.plot(t_plot, snr_plot, marker='o', linestyle='-', color=colors.get(smooth, 'black'), label=f"SNR {smooth}ms", linewidth=2, alpha=0.7)
        ax2.plot(t_plot, noise_mean_plot, marker='x', linestyle='--', color=colors.get(smooth, 'black'), label=f"Ruido {smooth}ms", linewidth=2, alpha=0.7)
        
    ax1.set_title("Evolución SNR Promedio")
    ax1.set_xlabel("Tiempo de Señal (s)")
    ax1.set_ylabel("SNR")
    ax1.grid(True, alpha=0.5)
    ax1.legend(loc='best')
    
    ax2.axhline(100.0, color='gray', linestyle='--', alpha=0.7, label='Línea Base (100%)')
    ax2.set_title("Evolución Ruido Inter-pulso")
    ax2.set_xlabel("Tiempo de Señal (s)")
    ax2.set_ylabel("Nivel de Ruido (%)")
    ax2.grid(True, alpha=0.5)
    ax2.legend(loc='best')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()

def _plot_espectro_and_spectrogram_experimental(dict_resultados, filename, out_spec, show_plot=False):
    import numpy as np
    plt.figure(figsize=(10, 5))
    colors = {5: 'blue', 25: 'green', 50: 'red'}
    
    for smooth, res in dict_resultados.items():
        pulso_promedio = res['mean_pulse']
        target_len = len(pulso_promedio)
        # Aproximar fs_seg
        duration = 1.0 # Aproximación si no tenemos pre_w+post_w a mano
        if 'periodo' in res: duration = res['periodo']
        
        freqs = np.fft.rfftfreq(len(pulso_promedio), d=duration/float(len(pulso_promedio)))
        spec = np.abs(np.fft.rfft(pulso_promedio))
        spec_db = 20.0 * np.log10(spec / (np.max(spec) + 1e-20) + 1e-20)
        
        mask_freq = (freqs >= 0) & (freqs <= min(1000, len(pulso_promedio)/duration/2.0))
        plt.plot(freqs[mask_freq], np.abs(spec_db[mask_freq]), color=colors.get(smooth, 'black'), label=f"Espectro {smooth}ms", alpha=0.8)
        
    plt.title(f"Espectro de Frecuencias Comparativo - {filename}")
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud [dB rel.]')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(out_spec, dpi=300, bbox_inches='tight')
    if show_plot: plt.show()
    else: plt.close()
"""

content = content.replace("def _plot_pulse_full_experimental", nuevas_funciones_2 + "\ndef _plot_pulse_full_experimental")

# Reemplazar dentro de la nueva procesar_wavs_promedio para llamarlas y guardar stats para la temporal
content = content.replace("res_s = _procesar_un_smooth(", "res_s = _procesar_un_smooth(")
# Ya habíamos sustituido _procesar_un_smooth para que devuelva dict, pero la original no devolvía stats_time ni stats_snr
# Modificar la original _procesar_un_smooth para que guarde stats_time, stats_snr, stats_noise_mean en resultados
content = content.replace("'noise_rms_from_noise_window': noise_rms_from_noise_window,", "'noise_rms_from_noise_window': noise_rms_from_noise_window,\n            'stats_time': stats_time,\n            'stats_snr': stats_snr,\n            'stats_noise_mean': stats_noise_mean,\n            'stats_noise_std': stats_noise_std,")

# Modificar procesar_wavs_promedio para llamar a _plot_evolucion_temporal_experimental y _plot_espectro_and_spectrogram_experimental
llamadas_nuevas = """            out_evol = os.path.join(out_dir, "evolucion_experimental.png")
            out_spec = os.path.join(out_dir, "spec_experimental.png")
            
            _plot_pulse_full_experimental(dict_resultados_smooth, final_plot_title, out_prom, show_plot=show_average_plot)
            if mostrar_recortes:
                _plot_recortes_experimental(dict_resultados_smooth, final_plot_title, out_rec, show_plot=False)
            if mostrar_evolucion:
                _plot_evolucion_temporal_experimental(dict_resultados_smooth, final_plot_title, out_evol, evol_t_start, evol_t_end, show_plot=False)
            if mostrar_espectrograma:
                _plot_espectro_and_spectrogram_experimental(dict_resultados_smooth, final_plot_title, out_spec, show_plot=False)"""

content = re.sub(r"out_dir = output_root.*?_plot_recortes_experimental\(dict_resultados_smooth, final_plot_title, out_rec, show_plot=False\)", llamadas_nuevas, content, flags=re.DOTALL)


with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Parche 2 aplicado correctamente")
