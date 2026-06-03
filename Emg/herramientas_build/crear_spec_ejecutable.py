import os

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
        ('justificacion_matematica.md', '.')
    ]:
        if os.path.exists(os.path.join(build_dir, os.path.normpath(resource))):
            datas_tuples.append(f"('{resource}', '{dest}')")
            
    datas_str = ",\n        ".join(datas_tuples)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Lista de TODOS los programas que lanzamos (serán convertidos a .exe independientes)
scripts = [
    'Sistema_de_Adquisicion_Emg',
    'CodigoUnificador_integrado',
    'Nandu_AutoForge_DAQ',
    'visor_csv_interactivo',
    'analisis_por_track_integrado',
    'analisis_por_track_integrado_experimental',
    'electrode_viewer_4',
    'instrucciones_uso',
    'metronomo_visual',
    'ventana_palabras',
    'editor_mediciones',
    'extractor_de_datos_procesados',
    'plotter_calibrado',
    'correlaciondeseñales',
    'actualizar_metadata',
    'gui_app/main_app'
]

analyses = []
pyzs = []
exes = []

# Librerías que PyInstaller a veces no detecta automáticamente
hidden_imports = [
    'scipy.signal', 'scipy.special', 'scipy.io.wavfile',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_qt5agg',
    'nidaqmx', 'sounddevice', 'soundfile', 'pyqtgraph', 'pandas', 'PIL',
    'PySide6', 'qdarkstyle'
]

datas = [
        {datas_str}
]

import os

for script_path in scripts:
    script_file = script_path + '.py'
    exe_name = script_path.split('/')[-1]
    
    a = Analysis(
        [script_file],
        pathex=[],
        binaries=[],
        datas=datas,
        hiddenimports=hidden_imports,
        hookspath=[],
        hooksconfig={{}},
        runtime_hooks=[],
        excludes=[],
        win_no_prefer_redirects=False,
        win_private_assemblies=False,
        cipher=block_cipher,
        noarchive=False,
    )
    analyses.append(a)
    pyzs.append(PYZ(a.pure, a.zipped_data, cipher=block_cipher))
    
    # console=True: Mantener la consola negra en esta fase para ver si hay errores (Fase 3)
    exes.append(EXE(pyzs[-1], a.scripts, [], exclude_binaries=True, name=exe_name, debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True, disable_windowed_traceback=False, argv_emulation=False, target_arch=None, codesign_identity=None, entitlements_file=None))

collect_args = []
for exe, a in zip(exes, analyses):
    collect_args.extend([exe, a.binaries, a.zipfiles, a.datas])

coll = COLLECT(*collect_args, strip=False, upx=True, upx_exclude=[], name='EMG_Studio_App')
"""
    spec_path = os.path.join(build_dir, "EMG_Studio.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"✅ Archivo Multipaquete .spec generado exitosamente en: {spec_path}")

if __name__ == "__main__":
    generar_spec()