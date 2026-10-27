# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files, copy_metadata

def _safe_collect(pkg):
    try:
        return collect_all(pkg)
    except Exception:
        return [], [], []

def _safe_collect_data(pkg):
    try:
        return collect_data_files(pkg)
    except Exception:
        return []

def _safe_metadata(pkg):
    try:
        return copy_metadata(pkg)
    except Exception:
        return []

nidaqmx_datas, nidaqmx_binaries, nidaqmx_hiddenimports = _safe_collect('nidaqmx')
sd_datas, sd_binaries, sd_hiddenimports = _safe_collect('sounddevice')
sf_datas, sf_binaries, sf_hiddenimports = _safe_collect('soundfile')
umap_datas, umap_binaries, umap_hiddenimports = _safe_collect('umap')
sns_datas, sns_binaries, sns_hiddenimports = _safe_collect('seaborn')
tly_datas, tly_binaries, tly_hiddenimports = _safe_collect('tensorly')
numba_datas, numba_binaries, numba_hiddenimports = _safe_collect('numba')
pynndescent_datas, pynndescent_binaries, pynndescent_hiddenimports = _safe_collect('pynndescent')
tqdm_datas, tqdm_binaries, tqdm_hiddenimports = _safe_collect('tqdm')
mp_datas, mp_binaries, mp_hiddenimports = _safe_collect('mediapipe')
try:
    mp_extra_datas = collect_data_files('mediapipe')
except Exception:
    mp_extra_datas = []
cv2_datas, cv2_binaries, cv2_hiddenimports = _safe_collect('cv2')
torch_datas, torch_binaries, torch_hiddenimports = _safe_collect('torch')
mne_datas, mne_binaries, mne_hiddenimports = _safe_collect('mne')
llvmlite_datas, llvmlite_binaries, llvmlite_hiddenimports = _safe_collect('llvmlite')
sklearn_datas, sklearn_binaries, sklearn_hiddenimports = _safe_collect('sklearn')
nitypes_metadata = _safe_metadata('nitypes')

block_cipher = None

# Lista de módulos adicionales a empaquetar en el ejecutable principal
additional_modules = [
    'acquisition.manual_daq',
    'acquisition.autoforge_daq',
    'acquisition.autoforge_daq_experimental',
    'acquisition.autoforge_daq_decodificador',
    'acquisition.calibracion_espacial_electrodos',
    'acquisition.fotometria_webcam',
    'EMG_desarrollo.acquisition.calibracion_espacial_electrodos',
    'EMG_desarrollo.acquisition.fotometria_webcam',
    'calibracion_espacial_electrodos',
    'fotometria_webcam',
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
    'analysis.filtro_adaptativo',
    'analysis.analisis_espectral_candela',
    'analysis.regenerar_fotos_sesion',
    'analysis.generar_graficos_y_ranking',
    'analysis.plot_metricas_tesis',
    'analysis.report_engine',
    'analysis.generador_figura_multimodal',
    'analysis.generador_atlas_pdf',
    'analysis.batch_actualizar_figuras_reporte',
    'utils.editor_mediciones',
    'utils.actualizar_metadata',
    'utils.migrar_mediciones_por_fecha',
    'utils.config_manager',
    'utils.path_utils',
    'utils.sound_utils',
    'utils.logger',
    'utils.limpiar_cache_analisis',
    'utils.curar_dataset_exportacion',
    'utils.metadata_auditor',
    'instrucciones_uso',
    'gui_app.core.threads',
    'gui_app.views.calibrated_viewer_widget',
    'gui_app.views.comparative_explorer_widget',
    'gui_app.views.config_dialog',
    'gui_app.views.csv_viewer_widget',
    'gui_app.views.electrode_viewer_widget',
    'gui_app.views.session_explorer',
    'gui_app.views.ui_analysis',
    'gui_app.views.report_dialog',
    'gui_app.views.atlas_dialog',
    'gui_app.views.selector_otro_sujeto_dialog',
    'gui_app.temp_comparativo',
    'gui_app.temp_procesar',
    'gui_app.temp_sesion',
    'deep_learning.realtime_decoder_engine',
    'deep_learning.pipeline_autoencoder_gui',
    'deep_learning.grid_search_autoencoder',
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
    'deep_learning.pca_analysis',
    'deep_learning.umap_analysis',
    'deep_learning.motor_autoencoder_unificado',
    'deep_learning.soft_dtw',
    'deep_learning.autoencoder_no_supervisado_gui',
]

# Librerías y módulos que PyInstaller requiere explícitamente
hidden_imports = [
    'scipy.signal', 'scipy.special', 'scipy.io.wavfile', 'scipy.ndimage',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qtagg',
    'nidaqmx', 'sounddevice', 'soundfile', 'pyqtgraph', 'pandas', 'PIL',
    'PySide6', 'qdarkstyle', 'utils', 'acquisition', 'analysis', 'core', 'views',
    'deep_learning', 'tkinter', 'numba', 'llvmlite', 'tqdm', 'decouple', 'requests',
    'tzlocal', 'hightime', 'sklearn', 'sklearn.utils._typedefs',
    'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils', 'umap', 'pynndescent', 'seaborn', 'mne', 'cv2', 'mediapipe', 'torch',
    'EMG_desarrollo.acquisition.calibracion_espacial_electrodos',
    'EMG_desarrollo.acquisition.fotometria_webcam',
    'calibracion_espacial_electrodos',
    'fotometria_webcam',
] + additional_modules + nidaqmx_hiddenimports + sd_hiddenimports + sf_hiddenimports + umap_hiddenimports + sns_hiddenimports + tly_hiddenimports + numba_hiddenimports + pynndescent_hiddenimports + tqdm_hiddenimports + mp_hiddenimports + cv2_hiddenimports + torch_hiddenimports + mne_hiddenimports + llvmlite_hiddenimports + sklearn_hiddenimports

datas = [
        ('icono.ico', '.'),
        ('config_general.json', '.'),
        ('metronome_config.json', '.'),
        ('palabras.txt', '.'),
        ('palabras.txt', 'acquisition'),
        ('archivos_md', 'archivos_md'),
        ('papers', 'papers'),
        ('DataConfig', 'DataConfig'),
        ('DataConfig/modelos_vision', 'DataConfig/modelos_vision'),
        ('DataConfig/modelos_vision/face_landmarker.task', 'DataConfig/modelos_vision'),
        ('C:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo/DataConfig/modelos_vision/face_landmarker.task', 'DataConfig/modelos_vision'),
        ('acquisition', 'acquisition'),
        ('analysis', 'analysis'),
        ('deep_learning', 'deep_learning'),
        ('acquisition/calibracion_espacial_electrodos.py', 'acquisition'),
        ('acquisition/fotometria_webcam.py', 'acquisition'),
        ('C:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo/acquisition/calibracion_espacial_electrodos.py', 'acquisition'),
        ('C:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo/acquisition/fotometria_webcam.py', 'acquisition'),
        ('fotos', 'fotos'),
        ('logo_nandu_lsd.png', '.'),
        ('logo_nandu_lsd.png', 'gui_app/assets'),
        ('logo_nandu_lsd.png', 'gui_app'),
        ('usb-621x-manual.pdf', '.'),
        ('modelos', 'modelos'),
        ('resultados', 'resultados'),
        ('C:/Users/MSI/OneDrive/Documentos/DOCUMENTOS SANTIAGO/santiago-prado-repositorio/EMG_desarrollo/resultados', 'EMG_desarrollo/resultados')
] + nidaqmx_datas + sd_datas + sf_datas + umap_datas + sns_datas + tly_datas + numba_datas + pynndescent_datas + tqdm_datas + mp_datas + mp_extra_datas + cv2_datas + torch_datas + mne_datas + llvmlite_datas + sklearn_datas + nitypes_metadata

qt_flat_binaries = []
try:
    import PySide6
    import shiboken6
    for _pkg_dir in (os.path.dirname(PySide6.__file__), os.path.dirname(shiboken6.__file__)):
        for _name in os.listdir(_pkg_dir):
            if _name.lower().endswith(('.dll', '.pyd')):
                qt_flat_binaries.append((os.path.join(_pkg_dir, _name), '.'))
except Exception:
    pass

binaries = nidaqmx_binaries + sd_binaries + sf_binaries + umap_binaries + sns_binaries + tly_binaries + numba_binaries + pynndescent_binaries + tqdm_binaries + mp_binaries + cv2_binaries + torch_binaries + mne_binaries + llvmlite_binaries + sklearn_binaries + qt_flat_binaries

# Un UNICO script de entrada: el bootloader de PyInstaller NO soporta --run-script
# (los scripts de la lista se ejecutarian EN CADENA al cerrarse cada uno).
# Los scripts externos (autoforge_daq_experimental, manual_daq, decodificador,
# metronomo, visores, etc.) se empaquetan como DATA y el dispatcher inyectado en
# main_app los lanza con runpy cuando recibe '--run-script <script>'.
a_scripts = ['gui_app/main_app.py']

a = Analysis(
    a_scripts,
    pathex=[
        '.',
        'gui_app',
        'gui_app/views',
        'deep_learning',
        'deep_learning/binarizacion',
        'deep_learning/dataset_tools',
        'deep_learning/pca_umap_clustering',
        'acquisition',
        'analysis',
        'utils',
        'views'
    ],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['C:\\Users\\MSI\\OneDrive\\Documentos\\DOCUMENTOS SANTIAGO\\santiago-prado-repositorio\\EMG_desarrollo\\herramientas_build\\runtime_hook_qt_paths.py'],
    excludes=['torchvision', 'torchaudio'],
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
