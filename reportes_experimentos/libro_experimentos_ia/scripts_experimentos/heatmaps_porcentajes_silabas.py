import numpy as np
import matplotlib.pyplot as plt

import matrices_confusion_silabas as base

cm_4d = base.cm_4d
cm_8d = base.cm_8d
silabas = base.silabas

# Normalizar por filas para obtener porcentajes exactos por clase real
cm_4d_pct = cm_4d.astype('float') / cm_4d.sum(axis=1)[:, np.newaxis] * 100.0
cm_8d_pct = cm_8d.astype('float') / cm_8d.sum(axis=1)[:, np.newaxis] * 100.0

fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))

paneles = [
    (axes[0], cm_4d_pct, "PCA 4D (LDA)", base.acc_4d),
    (axes[1], cm_8d_pct, "PCA 8D (LDA)", base.acc_8d)
]

for ax, cm_pct, tit, acc in paneles:
    im = ax.imshow(cm_pct, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
    ax.set_title(f"Matriz de Confusión {tit} - Normalizada por Clase\nExactitud Global: {acc:.1f}%", fontweight='bold', fontsize=12)
    ax.set_xticks(range(len(silabas)))
    ax.set_yticks(range(len(silabas)))
    ax.set_xticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_yticklabels([f"/{s.lower()}/" for s in silabas], fontweight='bold')
    ax.set_xlabel("Predicción", fontweight='bold')
    ax.set_ylabel("Real", fontweight='bold')
    
    # Texto porcentual adentro de cada celda
    for i in range(len(silabas)):
        for j in range(len(silabas)):
            val = cm_pct[i, j]
            col = "white" if val > 50 else "black"
            texto = f"{val:.0f}%" if val > 0 else "-"
            ax.text(j, i, texto, ha="center", va="center", color=col, fontweight='bold', fontsize=10)

# Barra de color común
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.03)
cbar.set_label('Porcentaje de Clasificación (%)', fontweight='bold')

plt.tight_layout()
out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/heatmaps_porcentajes_silabas.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
