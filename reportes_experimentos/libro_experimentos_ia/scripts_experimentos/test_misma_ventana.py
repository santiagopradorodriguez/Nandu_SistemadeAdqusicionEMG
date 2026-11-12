import re

with open("reportes_experimentos/libro_experimentos_ia/scripts_experimentos/extraer_formantes_canonico.py") as f:
    code_formantes = f.read()

print("Extracción de ventana en script de formantes:")
for line in code_formantes.splitlines():
    if "t_mic_wav" in line or "offset_audio_daq" in line or "idx_centro" in line or "win" in line:
        print("  ", line.strip())
