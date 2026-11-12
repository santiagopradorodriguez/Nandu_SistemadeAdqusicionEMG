import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

csv_path = "reportes_experimentos/decodificacion_facial/formantes_generador_pca_6canales.csv"
df = pd.read_csv(csv_path)

COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)
fig.patch.set_facecolor('#0b0f19')

for ax, is_log, title in [
    (ax1, False, "Escala Lineal Estándar (Hz)"),
    (ax2, True, "Escala Logarítmica (Hz)")
]:
    ax.set_facecolor('#0f172a')
    if is_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([900, 1100, 1300, 1500, 1700])
        ax.set_yticks([300, 400, 500, 600, 700, 800, 900])
    else:
        ax.set_xticks(np.arange(900, 1800, 200))
        ax.set_yticks(np.arange(300, 1000, 100))
        
    for voc in ['A', 'E', 'I', 'O', 'U']:
        sub = df[df['vocal'] == voc]
        f1 = sub['F1_Hz'].values
        f2 = sub['F2_Hz'].values
        color = COLORES_VOCALES[voc]
        
        ax.scatter(f2, f1, c=color, alpha=0.75, s=40, edgecolors='#ffffff', linewidths=0.4, label=f"/{voc.lower()}/")
        f1_m, f2_m = np.mean(f1), np.mean(f2)
        ax.scatter([f2_m], [f1_m], c=color, s=180, edgecolors='#ffffff', linewidths=2.0, zorder=5)
        ax.text(f2_m, f1_m - (20 if not is_log else f1_m*0.05), f"/{voc.lower()}/", color='#ffffff', fontsize=13, fontweight='bold', ha='center', va='top', zorder=6)

    ax.set_xlim(1800, 850)
    ax.set_ylim(950, 250)
    ax.grid(True, which='both' if is_log else 'major', color='#334155', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.set_xlabel(f"F2 [Hz] ({title}) - Invertido", fontsize=11, fontweight='bold', color='#cbd5e1')
    ax.set_ylabel(f"F1 [Hz] ({title}) - Invertido", fontsize=11, fontweight='bold', color='#cbd5e1')
    ax.set_title(title, fontsize=13, fontweight='bold', color='#ffffff', pad=12)
    leg = ax.legend(loc='lower left', frameon=True, facecolor='#1e293b', edgecolor='#475569', fontsize=10)
    for text in leg.get_texts():
        text.set_color('#ffffff')

fig.suptitle("Comparativa Acústica Directa: Escala Lineal vs Escala Logarítmica", fontsize=15, fontweight='bold', color='#ffffff', y=0.98)
out_png = "reportes_experimentos/decodificacion_facial/comparativa_formantes_lineal_vs_log.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print(f"[OK] Comparativa guardada en: {out_png}")
