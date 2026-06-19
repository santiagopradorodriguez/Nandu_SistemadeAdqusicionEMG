import os
import glob

# For all files in deep_learning, we want to make sure they can import from each other
# Since we moved files around, we can inject the paths.
files_to_check = glob.glob('deep_learning/**/*.py', recursive=True)
for file in files_to_check:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'sys.path.append(os.path.dirname(os.path.abspath(__file__)))' in content:
        # replace it with appending the new specific dirs
        new_sys_path = '''script_dir_abs = os.path.dirname(os.path.abspath(__file__))
deep_learning_dir = os.path.dirname(script_dir_abs)
if os.path.basename(deep_learning_dir) != "deep_learning":
    deep_learning_dir = script_dir_abs # fallback
sys.path.append(os.path.join(deep_learning_dir, "pca_umap_clustering"))
sys.path.append(os.path.join(deep_learning_dir, "dataset_tools"))
sys.path.append(os.path.join(deep_learning_dir, "binarizacion"))
sys.path.append(script_dir_abs)'''
        content = content.replace('sys.path.append(os.path.dirname(os.path.abspath(__file__)))', new_sys_path)
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated sys.path in {file}")

