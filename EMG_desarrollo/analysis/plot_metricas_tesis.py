import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

res_dir = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo\analysis\resultados_pca_umap'
metricas_path = os.path.join(res_dir, 'metricas.txt')
csv_desc_path = os.path.join(res_dir, 'reporte_mediciones_descartadas.csv')

# 1. Leer metricas.txt
with open(metricas_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Función helper para extraer matrices
def extract_matrix(lines, start_marker, end_marker=None):
    start_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i + 1
            break
    if start_idx == -1: return None
    
    mat_lines = []
    for line in lines[start_idx:]:
        if line.strip() == '' or (end_marker and end_marker in line):
            break
        mat_lines.append(line.split())
        
    # Asumiendo formato pd.to_string()
    headers = mat_lines[0]
    data = []
    index = []
    for row in mat_lines[1:]:
        index.append(row[0])
        data.append([float(x) for x in row[1:]])
    return pd.DataFrame(data, index=index, columns=headers)

# Extraer matrices
df_cm_pca = extract_matrix(lines, '--- MATRIZ BRUTA PCA 3D ---', 'Mapeo Húngaro:')
df_cm_umap = extract_matrix(lines, '--- MATRIZ BRUTA UMAP 3D ---', 'Mapeo Húngaro:')
df_dist_pca = extract_matrix(lines, 'Matriz de Distancias (PCA 3D):')
df_dist_umap = extract_matrix(lines, 'Matriz de Distancias (UMAP 3D):')

# Extraer Accuracy por Vocal
acc_pca = {}
acc_umap = {}
current_mode = None
for line in lines:
    if 'Accuracy No Supervisado (PCA 3D)' in line: current_mode = 'PCA'
    elif 'Accuracy No Supervisado (UMAP 3D)' in line: current_mode = 'UMAP'
    elif 'Vocal' in line and '%' in line:
        parts = line.strip().split(':')
        vocal = parts[0].replace('- Vocal ', '').strip()
        val = float(parts[1].replace('%', '').strip())
        if current_mode == 'PCA': acc_pca[vocal] = val
        elif current_mode == 'UMAP': acc_umap[vocal] = val

sns.set_theme(style='whitegrid')

# ----- PLOT 1: Confusion Matrices -----
if df_cm_pca is not None and df_cm_umap is not None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(df_cm_pca, annot=True, fmt='.0f', cmap='Blues', ax=axes[0], cbar=False)
    axes[0].set_title('K-Means Bruto sobre PCA 3D')
    axes[0].set_ylabel('Vocal Real')
    axes[0].set_xlabel('Clúster Asignado por K-Means')
    
    sns.heatmap(df_cm_umap, annot=True, fmt='.0f', cmap='Oranges', ax=axes[1], cbar=False)
    axes[1].set_title('K-Means Bruto sobre UMAP 3D')
    axes[1].set_ylabel('Vocal Real')
    axes[1].set_xlabel('Clúster Asignado por K-Means')
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'grafico_matrices_brutas.png'), dpi=300)
    plt.close()

# ----- PLOT 2: Accuracy per Vocal -----
if acc_pca and acc_umap:
    vocales = list(acc_pca.keys())
    x = np.arange(len(vocales))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, [acc_pca[v] for v in vocales], width, label='PCA 3D', color='skyblue')
    ax.bar(x + width/2, [acc_umap[v] for v in vocales], width, label='UMAP 3D', color='coral')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Precisión de Clustering No Supervisado por Vocal')
    ax.set_xticks(x)
    ax.set_xticklabels(vocales)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # Añadir valores
    for i, v in enumerate(vocales):
        ax.text(i - width/2, acc_pca[v] + 1, f"{acc_pca[v]:.1f}%", ha='center')
        ax.text(i + width/2, acc_umap[v] + 1, f"{acc_umap[v]:.1f}%", ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'grafico_accuracy_vocales.png'), dpi=300)
    plt.close()

# ----- PLOT 3: Mediciones Descartadas -----
if os.path.exists(csv_desc_path):
    df_desc = pd.read_csv(csv_desc_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Por Vocal
    sns.countplot(data=df_desc, x='Vocal', order=['A', 'E', 'I', 'O', 'U'], palette='Set2', ax=axes[0])
    axes[0].set_title('Pulsos Descartados por Vocal')
    axes[0].set_ylabel('Cantidad de Pulsos Eliminados')
    
    # Por Motivo
    sns.countplot(data=df_desc, y='Motivo', palette='Pastel1', ax=axes[1])
    axes[1].set_title('Motivos de Exclusión')
    axes[1].set_xlabel('Cantidad')
    
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'grafico_descartados.png'), dpi=300)
    plt.close()
    print("Gráficos de Descartes generados.")

print("Todos los gráficos fueron generados en resultados_pca_umap/")
