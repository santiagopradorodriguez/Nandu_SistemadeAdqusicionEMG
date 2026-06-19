import os
import glob

# For all files in deep_learning/binarizacion, deep_learning/pca_umap_clustering, deep_learning/dataset_tools
# We will inject sys.path code right after 'import sys' or 'import os'

sys_path_code = '''
import sys
import os
script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) == "deep_learning":
    sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
    sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
    sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
    sys.path.append(os.path.dirname(deep_learning_dir)) # EMG_desarrollo root
'''

files_to_check = glob.glob('deep_learning/**/*.py', recursive=True)
for file in files_to_check:
    # Skip standard deep learning files that weren't moved
    if file in ['deep_learning\\dataset_emg.py', 'deep_learning\\modelos.py', 'deep_learning\\train_autoencoder.py', 'deep_learning\\plot_latent_space.py']:
        continue
        
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'script_dir_abs =' not in content:
        # inject it at the top, after the imports
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_idx = i + 1
        
        lines.insert(insert_idx, sys_path_code)
        
        with open(file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Injected sys.path to {file}")

