# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo rename_titles.py del sistema NANDU LSD.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo rename_titles.py del sistema NANDU LSD.
# ==============================================================================

import os, re
pattern1 = re.compile(r'(setWindowTitle\([\'\"])(.*?)([\'\"]\))')
pattern2 = re.compile(r'(\.title\([\'\"])(.*?)([\'\"]\))')

for root, _, files in os.walk('.'):
    if 'EMG_Ejecutable_Build' in root or '.agents' in root or '.git' in root: continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            def repl(m):
                old_title = m.group(2)
                prefix = 'Ñandú LSD - '
                
                # if it already has the prefix or something similar
                if 'Ñandú LSD - ' in old_title or 'Ñandu LSD' in old_title or 'Nandu LSD' in old_title:
                    if old_title.startswith('Ñandú LSD - '): 
                        return m.group(0) # already fine
                    # clean up old prefix
                    clean_title = old_title.replace('Ñandu LSD - ', '').replace('Nandu LSD - ', '').replace('Ñandú LSD - ', '').replace('Ñandu LSD ', '')
                    if clean_title == '' or clean_title.startswith('-'):
                        clean_title = clean_title.lstrip('- ')
                    new_title = prefix + clean_title
                    if new_title == prefix:
                        new_title = prefix + "Módulo"
                    return m.group(1) + new_title + m.group(3)
                
                # Some exceptions where title has f-string or variables inside (not perfectly handled by regex, but we try)
                # But our regex only captures string literals so far. If the literal starts with f", it captured the f" too!
                # Wait, m.group(1) captures `setWindowTitle("` or `setWindowTitle(f"`. 
                # Let's just prepend the prefix inside the quote.
                new_title = prefix + old_title
                return m.group(1) + new_title + m.group(3)

            new_content = pattern1.sub(repl, content)
            new_content = pattern2.sub(repl, new_content)
            
            # Special case for f-strings with variables
            # e.g. setWindowTitle(f"Visor y Grabador...")
            # the regex captures:
            # group 1: setWindowTitle(f"
            # group 2: Visor y Grabador...
            # group 3: ")
            
            if new_content != content:
                print(f'Modifying {path}')
                with open(path, 'w', encoding='utf-8') as file:
                    file.write(new_content)
