import os
import glob

files_to_check = glob.glob('deep_learning/**/*.py', recursive=True)
for file in files_to_check:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'os.path.dirname(script_dir)' in content:
        content = content.replace('os.path.dirname(script_dir)', 'os.path.dirname(os.path.dirname(script_dir))')
        with open(file, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated paths in {file}")

    # Also fix sys.path imports!
    # If they were appending os.path.dirname(os.path.abspath(__file__)) it's fine.
    # But if they did sys.path.append(os.path.dirname(script_dir)), it needs an extra dirname too.
