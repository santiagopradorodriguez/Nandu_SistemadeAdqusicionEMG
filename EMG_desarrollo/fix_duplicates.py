import os, glob, re
for f in glob.glob('**/*.py', recursive=True):
    if 'env' in f or 'venv' in f or 'EMG_Ejecutable_Build' in f or 'fix_duplicates.py' in f: continue
    with open(f, 'r', encoding='utf-8') as file: content = file.read()
    
    # We want to find the header block
    block_pattern = r'# ==============================================================================\n# Proyecto: NANDU LSD - Sistema de Adquisici.n EMG y Deep Learning\n# Autores: Lucas Braunstein y Santiago Prado\n# Instituci.n: Laboratorio de Sistemas Din.micos \(LSD\) - FCEyN, UBA\n# Descripci.n:[^\n]+\n# ==============================================================================\n'
    
    matches = list(re.finditer(block_pattern, content))
    if len(matches) > 1:
        # Keep the first match, remove the others
        first_match = matches[0]
        # Remove everything matching the pattern and any whitespace after it
        content_no_headers = re.sub(block_pattern + r'\s*', '', content)
        new_content = first_match.group(0) + '\n' + content_no_headers
        with open(f, 'w', encoding='utf-8') as file: file.write(new_content)
        print(f'Fixed {f}')
