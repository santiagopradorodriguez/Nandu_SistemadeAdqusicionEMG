import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("reportes_experimentos/analisis_pca_generado_Exp3_Triada_Masetero_x25/triada/proyecciones_pca_2d.csv")
COLOR_VOCALES = {'A': '#E63946', 'E': '#1F77B4', 'I': '#2CA02C', 'O': '#9D4EDD', 'U': '#E7A61A'}

plt.figure(figsize=(9, 8))
for v in ['A', 'E', 'I', 'O', 'U']:
    sub = df[df['Vocal'] == v]
    plt.scatter(sub['PC1'], sub['PC2'], c=COLOR_VOCALES[v], label=f"/{v.lower()}/ (N={len(sub)})", s=50, alpha=0.85, edgecolors='black', linewidths=0.5)

plt.title("PCA 2D Oficial Original (Leído de proyecciones_pca_2d.csv)", fontweight='bold')
plt.xlabel("PC 1", fontweight='bold')
plt.ylabel("PC 2", fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(loc='best', frameon=True)
plt.tight_layout()

out_f = "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/pca_2d_oficial_ploteado_directo.png"
plt.savefig(out_f, dpi=300)
plt.close()
print("Guardado en:", out_f)
