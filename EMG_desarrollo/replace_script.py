import os

file_path = r'c:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\analysis\analisis_por_track_integrado.py'

with open(file_path, 'rb') as f:
    content = f.read()

target = b'''        for letra, vals in hist_data_amp[ch_idx].items():
            if len(vals) > 0:
                ax_hamp.hist(vals, bins=10, alpha=0.5, label=letra, color=letter_colors.get(letra, 'gray'), density=True)
        ax_hamp.set_title(f"Distribuci\xc3\xb3n de Amplitud - {labels[ch_idx]}")
        ax_hamp.set_xlabel("Amplitud M\xc3\xa1xima [\xc2\xb5V]")
        ax_hamp.set_ylabel("Densidad")
        ax_hamp.legend()'''

target = target.replace(b'\r\n', b'\n')
content = content.replace(b'\r\n', b'\n')

replace = b'''        import scipy.stats as stats
        
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
                
            ax_hamp.set_title(f"Distribuci\xc3\xb3n de MAV - {labels[ch_idx]}")
            ax_hamp.set_xlabel("Vocal")
            ax_hamp.set_ylabel("MAV [\xc2\xb5V]")
            
            if len(data_to_plot) > 1:
                try:
                    f_stat, p_val = stats.kruskal(*data_to_plot)
                    ax_hamp.text(0.05, 0.95, f"Kruskal-Wallis p-value: {p_val:.2e}", transform=ax_hamp.transAxes, 
                                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none'))
                except Exception as e:
                    pass'''

if target in content:
    content = content.replace(target, replace)
    with open(file_path, 'wb') as f:
        f.write(content)
    print('analisis_por_track_integrado.py updated successfully using bytes!')
else:
    print('target not found in analisis_por_track_integrado.py. Try partial match.')
