import sys

with open('EMG_desarrollo/analysis/umap_motor.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    # The loop starts at 183 (`else:`) -> 184 (`for path...`)
    # We want to indent from 192 (canales_medicion = ...) up to line 304 (mediciones_rechazadas.append(...))
    # Let's dynamically find it.
    pass

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if "if vocal == 'IGNORAR': continue" in line:
        start_idx = i + 1
        break

for i in range(start_idx, len(lines)):
    if "if not import_pca:" in lines[i]:
        end_idx = i
        break

print(f"Indenting lines {start_idx} to {end_idx - 1}")

for i in range(start_idx, end_idx):
    if lines[i].strip() == "":
        continue
    # Add 4 spaces
    lines[i] = "    " + lines[i]

with open('EMG_desarrollo/analysis/umap_motor.py', 'w') as f:
    f.writelines(lines)
