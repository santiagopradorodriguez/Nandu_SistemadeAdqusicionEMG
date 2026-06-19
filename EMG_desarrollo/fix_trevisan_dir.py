import os
import re

file_path = r'deep_learning\binarizacion\analisis_trevisan.py'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # Add OUT_DIR initialization
    if 'self.BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "base_de_datos_electrodos")' in line:
        indent = line[:len(line) - len(line.lstrip())]
        lines.insert(i+1, indent + 'self.OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(script_dir)), "resultados_binarizacion")\n')
        lines.insert(i+2, indent + 'os.makedirs(self.OUT_DIR, exist_ok=True)\n')
        break

# Now replace usages for saving files
for i, line in enumerate(lines):
    if '"separabilidad_picos_boxplot.png"' in line or \
       '"umbrales_boxplot.png"' in line or \
       '"porcentajes_vocales.png"' in line or \
       '"sensibilidad_umbrales.png"' in line or \
       '"espacio_motor_3d.png"' in line:
        lines[i] = line.replace('self.BASE_DIR', 'self.OUT_DIR')
        
    if '"senal_corregida_combinada.png"' in line and 'os.path.join' in line:
        indent = line[:len(line) - len(line.lstrip())]
        # Replace: out = os.path.join(self.BASE_DIR, med_name, "senal_corregida_combinada.png")
        # With: out_med = os.path.join(self.OUT_DIR, med_name); os.makedirs(out_med, exist_ok=True); out = os.path.join(out_med, "senal_corregida_combinada.png")
        new_lines = [
            indent + 'out_med = os.path.join(self.OUT_DIR, med_name)\n',
            indent + 'os.makedirs(out_med, exist_ok=True)\n',
            line.replace('self.BASE_DIR, med_name', 'out_med')
        ]
        lines[i] = "".join(new_lines)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(lines)
print("analisis_trevisan.py updated to use OUT_DIR")
