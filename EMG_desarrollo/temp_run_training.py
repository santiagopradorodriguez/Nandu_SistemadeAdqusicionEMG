
import json
import sys
import os

project_root = r'/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

with open(r'/tmp/tmp0x1pbde2.json', 'r') as f:
    cfg = json.load(f)

asignaciones_vocales = cfg['asignaciones_vocales']
canales_seleccionados = cfg['canales_seleccionados']
mapped_names = cfg['mapped_names']
filtro_snr_activo = cfg['filtro_snr_activo']
filtro_snr_limite = cfg['filtro_snr_limite']
filtro_snr_tipo = cfg['filtro_snr_tipo']
tipo_barrido = cfg['tipo_barrido']
paso_barrido = cfg['paso_barrido']
nombre_set = cfg.get('nombre_set', '')

import analysis.training_motor as tm

print("=========================================================")
print("      INICIANDO ENTRENAMIENTO DE UMBRALES (TRAIN)       ")
print("=========================================================")
print(f"Mediciones seleccionadas: {len(asignaciones_vocales)}")
print(f"Canales seleccionados: {canales_seleccionados}")
print(f"Tipo de barrido: {tipo_barrido} (Paso: {paso_barrido})")
print(f"Filtro SNR: {filtro_snr_tipo} > {filtro_snr_limite} (Activo: {filtro_snr_activo})")
if nombre_set:
    print(f"Identificador del set: {nombre_set}")
print("=========================================================\n")

tm.ejecutar_entrenamiento(
    asignaciones_vocales=asignaciones_vocales,
    canales_seleccionados=canales_seleccionados,
    mapped_names=mapped_names,
    filtro_snr_activo=filtro_snr_activo,
    filtro_snr_limite=filtro_snr_limite,
    filtro_snr_tipo=filtro_snr_tipo,
    tipo_barrido=tipo_barrido,
    paso_barrido=paso_barrido,
    nombre_set=nombre_set,
    logger=print
)
