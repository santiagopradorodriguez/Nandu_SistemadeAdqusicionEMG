import re

file_path = 'c:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/Emg/Nandu_AutoForge_DAQ.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i in range(2063, 2143): # 2064 to 2143 in 1-indexed is 2063 to 2142 in 0-indexed
    line = lines[i]
    if line.startswith('    '): # Unindent 4 spaces
        lines[i] = line[4:]

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print('Indentation fixed.')
