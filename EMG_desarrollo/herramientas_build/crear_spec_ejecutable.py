# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Genera el archivo spec de PyInstaller para la construcción.
# ==============================================================================

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

def generar_spec():
    # Al moverse a 'herramientas_build', el directorio raíz del proyecto es el padre
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")
    
    if not os.path.exists(build_dir):
        print("❌ Carpeta 'EMG_Ejecutable_Build' no encontrada.")
        return

    # Validar qué carpetas de assets existen para empaquetarlas
    datas_tuples = []
    for resource, dest in [
        ('icons', 'icons'),
        ('DataConfig/Pictures', 'DataConfig/Pictures'),
        ('gui_app/assets', 'gui_app/assets'),
        ('justificacion_matematica.md', '.'),
        ('palabras.txt', '.'),
        ('icono.ico', '.')
    ]:
        if os.path.exists(os.path.join(build_dir, os.path.normpath(resource))):
            datas_tuples.append(f"('{resource}', '{dest}')")
            
    datas_str = ",\n        ".join(datas_tuples)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all, copy_metadata

nidaqmx_datas, nidaqmx_binaries, nidaqmx_hiddenimports = collect_all('nidaqmx')
sd_datas, sd_binaries, sd_hiddenimports = collect_all('sounddevice')
sf_datas, sf_binaries, sf_hiddenimports = collect_all('soundfile')
nitypes_metadata = copy_metadata('nitypes')

block_cipher = None

# Lista de módulos adicionales a empaquetar en el ejecutable principal
additional_modules = [
    'acquisition.manual_daq',
    'acquisition.autoforge_daq',
    'acquisition.metronomo_visual',
    'analysis.analisis_por_track_integrado',
    'analysis.electrode_viewer_4',
    'analysis.feature_extractor',
    'analysis.plotter_calibrado',
    'analysis.correlaciondeseñales',
    'analysis.analisis_estadistico_pulsos',
    'utils.editor_mediciones',
    'utils.actualizar_metadata',
    'utils.migrar_mediciones_por_fecha',
    'instrucciones_uso',
    'acquisition.ventana_palabras',
    'views.config_dialog'
]

# Librerías que PyInstaller a veces no detecta automáticamente
hidden_imports = [
    'scipy.signal', 'scipy.special', 'scipy.io.wavfile',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_qt5agg',
    'nidaqmx', 'sounddevice', 'soundfile', 'pyqtgraph', 'pandas', 'PIL',
    'PySide6', 'qdarkstyle', 'utils', 'acquisition', 'analysis', 'core', 'views'
] + additional_modules + nidaqmx_hiddenimports + sd_hiddenimports + sf_hiddenimports

datas = [
        {datas_str}
] + nidaqmx_datas + sd_datas + sf_datas + nitypes_metadata

binaries = nidaqmx_binaries + sd_binaries + sf_binaries

a = Analysis(
    ['gui_app/main_app.py'],
    pathex=['.', 'gui_app'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'torchaudio'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NanduLsd',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True, 
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../icono.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NanduLsd'
)
"""
    spec_path = os.path.join(build_dir, "EMG_Studio.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"[OK] Archivo Multipaquete .spec generado exitosamente en: {spec_path}")

if __name__ == "__main__":
    generar_spec()