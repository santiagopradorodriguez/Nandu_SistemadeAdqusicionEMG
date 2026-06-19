import os
import glob

files = {
    r'deep_learning\pca_umap_clustering\generador_pca_umap.py': ('"resultados_pca_umap"', '"resultados", "resultados_pca_umap"'),
    r'deep_learning\pca_umap_clustering\experimentos_grid_search.py': ('"resultados_experimentos"', '"resultados", "resultados_experimentos"'),
    r'deep_learning\dataset_tools\generador_pca_tensorial.py': ('"resultados_pca_tensorial"', '"resultados", "resultados_pca_tensorial"'),
    r'deep_learning\binarizacion\analisis_trevisan.py': ('"resultados_binarizacion"', '"resultados", "resultados_binarizacion"'),
}

for filepath, (old, new) in files.items():
    if not os.path.exists(filepath):
        print(f"Not found: {filepath}")
        continue
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # In generador_pca_umap, experimentos_grid_search, and generador_pca_tensorial, they use os.path.dirname(script_dir).
    # Since they were moved inside deep_learning/xxx, they need an extra dirname to reach the root.
    if 'os.path.join(os.path.dirname(script_dir), ' + old in content:
        content = content.replace(
            'os.path.join(os.path.dirname(script_dir), ' + old,
            'os.path.join(os.path.dirname(os.path.dirname(script_dir)), ' + new
        )
    elif 'os.path.join(os.path.dirname(os.path.dirname(script_dir)), ' + old in content:
        content = content.replace(
            'os.path.join(os.path.dirname(os.path.dirname(script_dir)), ' + old,
            'os.path.join(os.path.dirname(os.path.dirname(script_dir)), ' + new
        )
        
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Updated {filepath}")

