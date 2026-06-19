import os
import shutil
import glob

root = r'C:\Users\MSI\OneDrive\Documentos\DOCUMENTOS SANTIAGO\santiago-prado-repositorio\EMG_desarrollo'
resultados_dir = os.path.join(root, 'resultados')

os.makedirs(resultados_dir, exist_ok=True)

# Move the 4 folders
folders_to_move = [
    (os.path.join(root, 'analysis', 'resultados_experimentos'), os.path.join(resultados_dir, 'resultados_experimentos')),
    (os.path.join(root, 'analysis', 'resultados_pca_tensorial'), os.path.join(resultados_dir, 'resultados_pca_tensorial')),
    (os.path.join(root, 'analysis', 'resultados_pca_umap'), os.path.join(resultados_dir, 'resultados_pca_umap')),
    (os.path.join(root, 'resultados_binarizacion'), os.path.join(resultados_dir, 'resultados_binarizacion'))
]

for src, dst in folders_to_move:
    if os.path.exists(src):
        # We use git mv if possible, but since these might not be tracked or tracked partially, 
        # let's just use Python shutil.move. Then we can 'git add -A' later.
        try:
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Failed to move {src}: {e}")

