# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Genera el archivo spec de PyInstaller para la construcción.
# ==============================================================================

import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

def generar_spec():
    # Al moverse a 'herramientas_build', el directorio raíz del proyecto es el padre
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")
    repo_root = os.path.dirname(base_dir)
    
    if not os.path.exists(build_dir):
        print("[ERROR] Carpeta 'EMG_Ejecutable_Build' no encontrada.")
        return

    # Validar qué recursos y assets existen para empaquetarlos
    candidate_assets = [
        ('icono.ico', '.'),
        ('config_general.json', '.'),
        ('metronome_config.json', '.'),
        ('palabras.txt', '.'),
        ('palabras.txt', 'acquisition'),
        ('archivos_md', 'archivos_md'),
        ('papers', 'papers'),
        ('logo_nandu_lsd.png', '.'),
        ('usb-621x-manual.pdf', '.'),
    ]
    
    datas_tuples = []
    for src, dst in candidate_assets:
        local_path = os.path.join(build_dir, os.path.normpath(src))
        base_path = os.path.join(base_dir, os.path.normpath(src))
        root_path = os.path.join(repo_root, os.path.normpath(src))
        if os.path.exists(local_path):
            datas_tuples.append(f"('{src}', '{dst}')")
        elif os.path.exists(base_path):
            datas_tuples.append(f"('{base_path}', '{dst}')")
        elif os.path.exists(root_path):
            datas_tuples.append(f"('{root_path}', '{dst}')")
            
    datas_str = ",\n        ".join(datas_tuples)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata

nidaqmx_datas, nidaqmx_binaries, nidaqmx_hiddenimports = collect_all('nidaqmx')
sd_datas, sd_binaries, sd_hiddenimports = collect_all('sounddevice')
sf_datas, sf_binaries, sf_hiddenimports = collect_all('soundfile')
umap_datas, umap_binaries, umap_hiddenimports = collect_all('umap')
sns_datas, sns_binaries, sns_hiddenimports = collect_all('seaborn')
tly_datas, tly_binaries, tly_hiddenimports = collect_all('tensorly')
numba_datas, numba_binaries, numba_hiddenimports = collect_all('numba')
pynndescent_datas, pynndescent_binaries, pynndescent_hiddenimports = collect_all('pynndescent')
tqdm_datas, tqdm_binaries, tqdm_hiddenimports = collect_all('tqdm')
nitypes_metadata = copy_metadata('nitypes')

block_cipher = None

# Lista de módulos adicionales a empaquetar en el ejecutable principal
additional_modules = [
    'acquisition.manual_daq',
    'acquisition.autoforge_daq',
    'acquisition.autoforge_daq_experimental',
    'acquisition.metronomo_visual',
    'acquisition.modulo_de_entrenamiento',
    'acquisition.ventana_palabras',
    'analysis.analisis_por_track_integrado',
    'analysis.segmentador_secuencias',
    'analysis.electrode_viewer_4',
    'analysis.plotter_calibrado',
    'analysis.correlaciondeseñales',
    'analysis.analisis_estadistico_pulsos',
    'analysis.reproductor_canal3',
    'analysis.discrete_motor',
    'analysis.pca_motor',
    'analysis.training_motor',
    'analysis.umap_motor',
    'analysis.generar_graficos_y_ranking',
    'utils.editor_mediciones',
    'utils.actualizar_metadata',
    'utils.migrar_mediciones_por_fecha',
    'utils.config_manager',
    'utils.path_utils',
    'utils.logger',
    'instrucciones_uso',
    'views.config_dialog',
    'gui_app.core.threads',
    'gui_app.views.calibrated_viewer_widget',
    'gui_app.views.comparative_explorer_widget',
    'gui_app.views.config_dialog',
    'gui_app.views.csv_viewer_widget',
    'gui_app.views.electrode_viewer_widget',
    'gui_app.views.session_explorer',
    'gui_app.views.ui_analysis',
    'deep_learning.pipeline_autoencoder_gui',
    'deep_learning.pca_umap_clustering.generador_pca_umap',
    'deep_learning.binarizacion.analisis_trevisan',
    'deep_learning.binarizacion.analisis_binario',
    'deep_learning.binarizacion.analisis_trevisan_bandas',
    'deep_learning.dataset_tools.visor_features',
    'deep_learning.dataset_tools.plot_3_musculos_standalone',
    'deep_learning.dataset_tools.plot_derivadas_standalone',
    'deep_learning.dataset_tools.generador_pca_tensorial',
    'deep_learning.modelos',
    'deep_learning.dataset_emg',
    'deep_learning.train_autoencoder',
    'deep_learning.plot_latent_space',
    'deep_learning.decodificador_continuo',
    'deep_learning.generador_umap_supervisado',
    'deep_learning.pca_analysis',
    'deep_learning.umap_analysis',
]

# Librerías y módulos que PyInstaller requiere explícitamente
hidden_imports = [
    'scipy.signal', 'scipy.special', 'scipy.io.wavfile', 'scipy.ndimage',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_qt5agg',
    'nidaqmx', 'sounddevice', 'soundfile', 'pyqtgraph', 'pandas', 'PIL',
    'PySide6', 'qdarkstyle', 'utils', 'acquisition', 'analysis', 'core', 'views',
    'deep_learning', 'tkinter', 'numba', 'tqdm', 'decouple', 'python-decouple', 'requests',
    'tzlocal', 'hightime', 'sklearn', 'sklearn.utils._typedefs',
    'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils', 'pynndescent',
] + additional_modules + nidaqmx_hiddenimports + sd_hiddenimports + sf_hiddenimports + umap_hiddenimports + sns_hiddenimports + tly_hiddenimports + numba_hiddenimports + pynndescent_hiddenimports + tqdm_hiddenimports

datas = [
        {datas_str}
] + nidaqmx_datas + sd_datas + sf_datas + umap_datas + sns_datas + tly_datas + numba_datas + pynndescent_datas + tqdm_datas + nitypes_metadata

binaries = nidaqmx_binaries + sd_binaries + sf_binaries + umap_binaries + sns_binaries + tly_binaries + numba_binaries + pynndescent_binaries + tqdm_binaries

a = Analysis(
    ['gui_app/main_app.py'],
    pathex=[
        '.',
        'gui_app',
        'deep_learning',
        'deep_learning/binarizacion',
        'deep_learning/dataset_tools',
        'deep_learning/pca_umap_clustering',
        'deep_learning/machine_learning',
        'acquisition',
        'analysis',
        'utils',
        'views'
    ],
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
    name='NanduLsd_Core',
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
    icon='icono.ico' if os.path.exists('icono.ico') else None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NanduLsd_Core'
)
"""
    spec_path = os.path.join(build_dir, "EMG_Studio.spec")
    with open(spec_path, "w", encoding="utf-8") as f:
        f.write(spec_content)
    print(f"[OK] Archivo Multipaquete .spec generado exitosamente en: {spec_path}")

if __name__ == "__main__":
    generar_spec()