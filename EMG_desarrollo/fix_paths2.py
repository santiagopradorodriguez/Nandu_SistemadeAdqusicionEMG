import os
import glob

files_to_check = glob.glob('deep_learning/**/*.py', recursive=True)
for file in files_to_check:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for "analysis" paths
    if '"analysis"' in content or "'analysis'" in content:
        content = content.replace('"analysis"', '""') # wait, that's dangerous. Let's just print them first to inspect.
        print(f"Warning: hardcoded 'analysis' found in {file}")

