import os
import numpy as np

file1 = r'c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\gui_app\main_app.py'
with open(file1, 'r', encoding='utf-8') as f:
    content1 = f.read()

target1 = '''            amp_per_pulse = []
            if isinstance(segmentos_rs, list) and len(segmentos_rs) > 0:
                for p in segmentos_rs:
                    if isinstance(p, list) and len(p) > 0:
                        max_p = float(np.max(p))
                        amp_per_pulse.append(max_p)
                        if not res.get('snr_per_pulse') and umbral and umbral > 0:
                            snr_per_pulse.append(max_p / umbral)
                    else:
                        amp_per_pulse.append(np.nan)
                        if not res.get('snr_per_pulse') and umbral and umbral > 0:
                            snr_per_pulse.append(np.nan)'''

replace1 = '''            amp_per_pulse = []
            if isinstance(segmentos_rs, list) and len(segmentos_rs) > 0:
                for p in segmentos_rs:
                    if isinstance(p, list) and len(p) > 0:
                        mav_val = float(np.mean(np.abs(p)))
                        amp_per_pulse.append(mav_val)
                        if not res.get('snr_per_pulse') and umbral and umbral > 0:
                            snr_per_pulse.append(mav_val / umbral)
                    else:
                        amp_per_pulse.append(np.nan)
                        if not res.get('snr_per_pulse') and umbral and umbral > 0:
                            snr_per_pulse.append(np.nan)'''

if target1 in content1:
    content1 = content1.replace(target1, replace1)
    with open(file1, 'w', encoding='utf-8') as f:
        f.write(content1)
    print('main_app.py updated successfully.')
else:
    print('target1 not found in main_app.py')

file2 = r'c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\analysis\analisis_por_track_integrado.py'
with open(file2, 'r', encoding='utf-8') as f:
    content2 = f.read()

target2 = '''        for letra, vals in hist_data_amp[ch_idx].items():
            if len(vals) > 0:
                ax_hamp.hist(vals, bins=10, alpha=0.5, label=letra, color=letter_colors.get(letra, 'gray'), density=True)
        ax_hamp.set_title(f"Distribución de Amplitud - {labels[ch_idx]}")
        ax_hamp.set_xlabel("Amplitud Máxima [µV]")
        ax_hamp.set_ylabel("Densidad")
        ax_hamp.legend()'''

replace2 = '''        import scipy.stats as stats
        data_to_plot = []
        labels_plot = []
        vowels = ['A', 'E', 'I', 'O', 'U']
        letras_presentes = list(hist_data_amp[ch_idx].keys())
        orden_letras = [v for v in vowels if v in letras_presentes] + sorted([l for l in letras_presentes if l not in vowels])
        
        for letra in orden_letras:
            vals = hist_data_amp[ch_idx][letra]
            if len(vals) > 0:
                data_to_plot.append(vals)
                labels_plot.append(letra)
                
        if len(data_to_plot) > 0:
            bp = ax_hamp.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
            for i, patch in enumerate(bp['boxes']):
                letra = labels_plot[i]
                patch.set_facecolor(letter_colors.get(letra, 'gray'))
                patch.set_alpha(0.6)
                
            ax_hamp.set_title(f"Distribución de MAV - {labels[ch_idx]}")
            ax_hamp.set_xlabel("Vocal")
            ax_hamp.set_ylabel("MAV [µV]")
            
            if len(data_to_plot) > 1:
                try:
                    f_stat, p_val = stats.kruskal(*data_to_plot)
                    ax_hamp.text(0.05, 0.95, f"Kruskal-Wallis p-value: {p_val:.2e}", transform=ax_hamp.transAxes, 
                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
                except Exception as e:
                    pass'''

if target2 in content2:
    content2 = content2.replace(target2, replace2)
    with open(file2, 'w', encoding='utf-8') as f:
        f.write(content2)
    print('analisis_por_track_integrado.py updated successfully.')
else:
    print('target2 not found in analisis_por_track_integrado.py')
