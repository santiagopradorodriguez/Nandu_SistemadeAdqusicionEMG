import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = "reportes_experimentos/decodificacion_facial/formantes_generador_pca_6canales.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"No se encontró {csv_path}")

df = pd.read_csv(csv_path)

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

# Escala lineal estándar
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
    
    # Centroide
    f1_mean, f2_mean = np.mean(f1), np.mean(f2)
    ax.scatter(
        [f2_mean], [f1_mean],
        c=color,
        s=220,
        edgecolors='#ffffff',
        linewidths=2.0,
        zorder=5
    )
    ax.text(
        f2_mean, f1_mean - 18,
        f"/{voc.lower()}/",
        color='#ffffff',
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='top',
        zorder=6
    )

# En fonética acústica, los ejes F2 y F1 se suelen invertir para reflejar la posición anatómica lingual/mandibular
# F2 invertido (de anterior a posterior): 1700 a 850 Hz
# F1 invertido (de cerrada a abierta): 250 a 950 Hz
ax.set_xlim(1750, 850)
ax.set_ylim(950, 250)

ax.grid(True, color='#334155', linestyle='--', linewidth=0.6, alpha=0.6)
ax.set_xlabel("Frecuencia del Segundo Formante F2 [Hz] - Escala Lineal (Invertido)", fontsize=12, fontweight='bold', color='#cbd5e1', labelpad=10)
ax.set_ylabel("Frecuencia del Primer Formante F1 [Hz] - Escala Lineal (Invertido)", fontsize=12, fontweight='bold', color='#cbd5e1', labelpad=10)

ax.set_title("Plano Acústico de Formantes F1-F2 en Escala Lineal - Sesión Continua", fontsize=14, fontweight='bold', color='#ffffff', pad=15)

leg = ax.legend(loc='lower left', frameon=True, facecolor='#1e293b', edgecolor='#475569', fontsize=11)
for text in leg.get_texts():
    text.set_color('#ffffff')

out_png = "reportes_experimentos/decodificacion_facial/plano_formantes_lineal.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
plt.close()
print(f"[OK] Plano lineal generado en: {out_png}")
