import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Cargar datos canónicos extraídos
csv_path = "reportes_experimentos/decodificacion_facial/formantes_generador_pca_6canales.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró {csv_path}")

df = pd.read_csv(csv_path)

# Paleta canónica del proyecto
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
fig.patch.set_facecolor('#0b0f19')
ax.set_facecolor('#0f172a')

# Convención fonética acústica internacional:
# Eje X: F2 (invertido, escala logarítmica)
# Eje Y: F1 (invertido, escala logarítmica)
# Mayor apertura mandibular abajo, articulación anterior a la izquierda.

ax.set_xscale('log')
ax.set_yscale('log')

# Graficar dispersión y centroides
for voc in ['A', 'E', 'I', 'O', 'U']:
    sub = df[df['vocal'] == voc]
    f1 = sub['F1_Hz'].values
    f2 = sub['F2_Hz'].values
    color = COLORES_VOCALES[voc]
    
    # Nube individual
    ax.scatter(
        f2, f1, 
        c=color, 
        alpha=0.75, 
        s=45, 
        edgecolors='#ffffff', 
        linewidths=0.5,
        label=f"/{voc.lower()}/ (N={len(sub)})"
    )
    
    # Centroide con elipse / dispersión
    f1_mean, f2_mean = np.mean(f1), np.mean(f2)
    f1_std, f2_std = np.std(f1), np.std(f2)
    
    ax.scatter(
        [f2_mean], [f1_mean],
        c=color,
        s=220,
        edgecolors='#ffffff',
        linewidths=2.0,
        zorder=5
    )
    ax.text(
        f2_mean, f1_mean * 0.94,
        f"/{voc.lower()}/",
        color='#ffffff',
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top',
        zorder=6
    )

# Configuración de límites e inversión de ejes fonéticos
# Rango F2: ~800 Hz a ~2000 Hz (invertido: 2000 a 800)
# Rango F1: ~250 Hz a ~1000 Hz (invertido: 1000 a 250)
ax.set_xlim(1900, 850)
ax.set_ylim(1000, 280)

# Formato de ticks legibles en escala logarítmica
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
ax.set_xticks([900, 1000, 1200, 1400, 1600, 1800])
ax.set_yticks([300, 400, 500, 600, 700, 800, 900])

ax.grid(True, which='both', color='#334155', linestyle='--', linewidth=0.6, alpha=0.6)
ax.set_xlabel("Frecuencia del Segundo Formante F2 [Hz] - Escala Logarítmica (Invertido)", fontsize=12, fontweight='bold', color='#cbd5e1', labelpad=10)
ax.set_ylabel("Frecuencia del Primer Formante F1 [Hz] - Escala Logarítmica (Invertido)", fontsize=12, fontweight='bold', color='#cbd5e1', labelpad=10)

ax.set_title("Plano Acústico de Formantes F1-F2 en Escala Logarítmica - Sesión Continua", fontsize=14, fontweight='bold', color='#ffffff', pad=15)

leg = ax.legend(loc='lower left', frameon=True, facecolor='#1e293b', edgecolor='#475569', fontsize=11)
for text in leg.get_texts():
    text.set_color('#ffffff')

# Guardar figura
out_png = "reportes_experimentos/decodificacion_facial/plano_formantes_logaritmico.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print(f"[OK] Plano acústico logarítmico generado en: {out_png}")
