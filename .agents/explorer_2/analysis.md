# Multi-Platform Packaging & Build Analysis (Requirement R2)

## 1. Executive Summary

This report documents the architectural review and empirical investigation into the build and packaging systems of the **Nandu LSD (Sistema de Adquisición EMG)** repository across Linux and Windows platforms.

The system uses PyInstaller (version 6.21.0 on Python 3.12) to compile a multi-module Python application featuring PySide6 (Qt6), PyQtGraph, SciPy/NumPy/Pandas, NI-DAQmx, sounddevice/soundfile, and machine learning toolkits (scikit-learn, UMAP, XGBoost, TensorLy, and PyTorch).

Key findings:
1. The existing build pipeline in `EMG_desarrollo/herramientas_build/` relies on automated spec generation (`crear_spec_ejecutable.py`) and source file patching (`aplicar_parches_ejecutable.py`).
2. Several critical data files (`config_general.json`, `metronome_config.json`, `logo_nandu_lsd.png`, `archivos_md/` documentation) are omitted from the PyInstaller `datas` bundle.
3. Essential hidden imports and C-extensions (`seaborn`, `xgboost`, `tensorly`, `umap`, `numba`, `tqdm`, `pynndescent`) were missing or incomplete, directly causing runtime crashes (documented in `multiplexer_error.log`).
4. Two obsolete module paths (`analysis.analisis_por_track_integrado_experimental` and `analysis.feature_extractor`) in `additional_modules` triggered module import failures.
5. In-app subprocess script launching in frozen mode (`_launch_dl_ml_script` in `main_app.py`) checked absolute filesystem paths that do not exist inside frozen bundles, preventing auxiliary ML/DL scripts from launching.
6. The repository root was missing root-level convenience wrappers (`build_linux.sh`, `build.bat`), and root directories `build_linux/` and `build_windows/` remained unpopulated.

---

## 2. Build System & Script Inventory

### 2.1 File Locations & Roles

| Path | Purpose / Description | Status / Issues |
|---|---|---|
| `EMG_desarrollo/build_linux.sh` | Main bash build script for Linux | Functional, but targets only `EMG_Ejecutable_Build/dist/NanduLsd` and lacks argument forwarding in `run_nandu.sh` |
| `EMG_desarrollo/build.bat` | Main batch build script for Windows | Functional with .NET `csc.exe` C# launcher compilation, but relies on brittle spec/patch generator |
| `EMG_desarrollo/herramientas_build/crear_entorno_ejecutable.py` | Copies `EMG_desarrollo/` into staging directory `EMG_Ejecutable_Build/` | Working; excludes database and venv directories |
| `EMG_desarrollo/herramientas_build/aplicar_parches_ejecutable.py` | Injects PyInstaller path resolution helpers (`resource_path`, `user_data_path`, `lanzar_script`) | Outdated string matches (e.g. `viejo_launch` and `md_path`) fail to match updated source code |
| `EMG_desarrollo/herramientas_build/crear_spec_ejecutable.py` | Generates `EMG_Studio.spec` dynamically | Missing datas (`config_general.json`, `metronome_config.json`, `logo_nandu_lsd.png`, `archivos_md`), missing `collect_all` hooks for XGBoost, UMAP, TensorLy, Seaborn |
| `EMG_desarrollo/herramientas_build/launcher.cs` | Native C# launcher for Windows with splash screen | Compiles via `csc.exe` /target:winexe; functional |
| `EMG_desarrollo/EMG_Ejecutable_Build/EMG_Studio.spec` | Generated spec file used by PyInstaller | Contains broken module references and incomplete datas/hiddenimports |
| `build_linux/` (repo root) | Target distribution folder for Linux builds | Currently empty |
| `build_windows/` (repo root) | Target distribution folder for Windows builds | Currently empty |
| `build_linux.sh` (repo root) | Root wrapper for Linux build | Missing at repo root |
| `build.bat` (repo root) | Root wrapper for Windows build | Missing at repo root |

---

## 3. Dependency & Requirements Audit

### 3.1 Comparison of Requirements Files

| Requirement / Package | `requirements.txt` (Root) | `requirements_linux.txt` (Root) | `EMG_desarrollo/requirements.txt` | Used in Codebase? |
|---|---|---|---|---|
| `numpy` | `numpy<=1.24.3` | `numpy` | `numpy<=1.24.3` | Yes (33 files) |
| `pandas` | `pandas<=2.0.3` | `pandas` | `pandas<=2.0.3` | Yes (22 files) |
| `matplotlib` | `matplotlib<=3.7.3` | `matplotlib` | `matplotlib<=3.7.3` | Yes (31 files) |
| `scipy` | `scipy<=1.10.1` | `scipy` | `scipy<=1.10.1` | Yes (21 files) |
| `soundfile` | `soundfile` | `soundfile` | `soundfile` | Yes (7 files) |
| `sounddevice` | `sounddevice` | `sounddevice` | `sounddevice` | Yes (2 files) |
| `pyqtgraph` | `pyqtgraph` | `pyqtgraph` | `pyqtgraph` | Yes (5 files) |
| `PySide6` | `PySide6<=6.5.3` | `PySide6` | `PySide6<=6.5.3` | Yes (20 files) |
| `Pillow` | `Pillow<=10.0.1` | `Pillow` | `Pillow<=10.0.1` | Yes (1 file) |
| `nidaqmx` | `nidaqmx` | `nidaqmx` | `nidaqmx` | Yes (3 files) |
| `qdarkstyle` | `qdarkstyle` | `qdarkstyle` | `qdarkstyle` | Yes (1 file) |
| `torch` / `torchaudio` / `torchvision` | `torch<=2.0.1` | `torch` | `torch<=2.0.1` | Yes (5 files in `deep_learning/`) |
| `pyinstaller` | `pyinstaller==5.13.2` | `pyinstaller` | `pyinstaller==5.13.2` | Build tool |
| `scikit-learn` | `scikit-learn` | `scikit-learn` | `scikit-learn` | Yes (9 files) |
| `umap-learn` | `umap-learn` | `umap-learn` | `umap-learn` | Yes (7 files) |
| `seaborn` | **MISSING** | `seaborn` | `seaborn` | Yes (10 files) |
| `xgboost` | **MISSING** | `xgboost` | `xgboost` | Yes (1 file: `analisis_xgboost.py`) |
| `tensorly` | **MISSING** | **MISSING** | `tensorly` | Yes (1 file: `generador_pca_tensorial.py`) |
| `numba` | **MISSING** | **MISSING** | `numba` | Yes (1 file: `autoforge_daq_experimental.py`) |
| `tqdm` | **MISSING** | **MISSING** | **MISSING** | Yes (1 file: `experimento_grid_search_3_autoencoder.py`) |
| `python-decouple` | **MISSING** | **MISSING** | `python-decouple` | Runtime utility |
| `requests` | **MISSING** | **MISSING** | `requests` | Runtime utility |
| `tzlocal` | **MISSING** | **MISSING** | `tzlocal` | Required by nidaqmx |
| `hightime` | **MISSING** | **MISSING** | `hightime` | Required by nidaqmx |

### 3.2 Root Cause of Dependency Version Conflicts
The strict pin `numpy<=1.24.3`, `PySide6<=6.5.3`, `pyinstaller==5.13.2`, `torch<=2.0.1` was originally added for legacy Windows 7 / Python 3.8 environments. On modern Python (3.11, 3.12+), installing `numpy<=1.24.3` or `PySide6<=6.5.3` fails with wheel compilation errors.

**Recommendation:**
Consolidate requirements into standard modern packages with optional constraints:
```text
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
soundfile>=0.12.0
sounddevice>=0.4.0
pyqtgraph>=0.13.0
PySide6>=6.5.0
Pillow>=9.5.0
nidaqmx>=0.8.0
qdarkstyle>=3.1.0
scikit-learn>=1.2.0
umap-learn>=0.5.3
seaborn>=0.12.0
xgboost>=1.7.0
tensorly>=0.8.0
numba>=0.57.0
tqdm>=4.65.0
python-decouple>=3.8
requests>=2.28.0
tzlocal>=5.0
hightime>=0.2.0
pyinstaller>=6.0.0
```

---

## 4. PyInstaller Spec & Packaging Discrepancies

### 4.1 Missing Data Files (`datas`)
In `crear_spec_ejecutable.py` (lines 28-40), the current data bundle search list is:
```python
    for resource, dest in [
        ('icons', 'icons'),
        ('DataConfig/Pictures', 'DataConfig/Pictures'),
        ('gui_app/assets', 'gui_app/assets'),
        ('justificacion_matematica.md', '.'),
        ('palabras.txt', '.'),
        ('icono.ico', '.')
    ]:
```
When evaluated against the actual directory structure:
- `icons`: Directory does not exist (`EMG_desarrollo/icons` is absent).
- `DataConfig/Pictures`: Directory does not exist (`EMG_desarrollo/DataConfig/Pictures` is absent).
- `gui_app/assets`: Directory does not exist (`EMG_desarrollo/gui_app/assets` is absent).
- `justificacion_matematica.md`: Does not exist at root of `EMG_desarrollo`. It is located at `archivos_md/justificacion_matematica.md`.
- `palabras.txt`: Exists and is included.
- `icono.ico`: Exists and is included.

**Critical Omissions:**
1. `config_general.json`: The core JSON configuration file used by `utils/config_manager.py` and `views/config_dialog.py` is not bundled.
2. `metronome_config.json`: The metronome configuration file used by `acquisition/metronomo_visual.py` and `acquisition/autoforge_daq.py` is not bundled.
3. `logo_nandu_lsd.png`: The application logo displayed on the splash screen and welcome tab is not bundled.
4. `archivos_md/`: The markdown documentation directory (containing `documentacion_matematica.md`, `justificacion_matematica.md`, `informe_cientifico_emg.md`, `calculo_snr_explicacion.md`, etc.) is not bundled.
5. `papers/`: Reference documentation files (`usb-621x-manual.pdf`) are not bundled.

### 4.2 Hidden Imports & C-Extension Binaries
In `crear_spec_ejecutable.py`, PyInstaller hooks are collected for `nidaqmx`, `sounddevice`, and `soundfile`. However, dynamic/compiled libraries for several machine learning packages are missing:
1. `xgboost`: Contains compiled binary shared objects (`libxgboost.so` / `xgboost.dll`). Must use `collect_all('xgboost')`.
2. `umap`: Uses `numba` JIT and `pynndescent`. Must use `collect_all('umap')`.
3. `seaborn`: Dynamic plotting styles and palettes. Must use `collect_all('seaborn')`.
4. `tensorly`: Tensor decomposition backends. Must use `collect_all('tensorly')`.
5. `scikit-learn`: Cython extensions in `sklearn.utils._typedefs`, `sklearn.neighbors._typedefs`, `sklearn.neighbors._quad_tree`, `sklearn.tree._utils`.
6. `tkinter`: Interactive TkAgg backend for Matplotlib manual signal curation.

### 4.3 Outdated Module Paths in `additional_modules`
In `crear_spec_ejecutable.py` (lines 54-81):
- `'analysis.analisis_por_track_integrado_experimental'`: File was deleted/renamed; does not exist.
- `'analysis.feature_extractor'`: File was renamed to `deep_learning/dataset_tools/visor_features.py`; does not exist at `analysis.feature_extractor`.

### 4.4 Unprefixed Internal Imports & `pathex` Search Paths
Several modules in `deep_learning/` import sibling modules without the package prefix:
- `deep_learning/plot_latent_space.py`: `from modelos import ConvAutoencoder1D`, `from dataset_emg import EMGDataset`
- `deep_learning/train_autoencoder.py`: `from modelos import ConvAutoencoder1D`, `from dataset_emg import EMGDataset`
- `deep_learning/pca_umap_clustering/generador_pca_umap.py`: `import analisis_trevisan as at`
- `deep_learning/pca_analysis.py`: `import analisis_trevisan as at`
- `deep_learning/umap_analysis.py`: `import analisis_trevisan as at`
- `deep_learning/dataset_tools/generador_pca_tensorial.py`: `import analisis_trevisan as at`

In the spec file, `pathex` is currently set only to `['.', 'gui_app']`. To ensure PyInstaller resolves these imports during analysis, `pathex` must include:
```python
pathex = [
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
]
```

---

## 5. In-App Multiplexer & Subprocess Execution Analysis

### 5.1 Architecture of the Multiplexer
The application runs as a single compiled executable (`NanduLsd_Core` / `NanduLsd.exe`). When launching auxiliary tools (e.g. AutoForge DAQ, Metronome, XGBoost analysis, Visualizer), the main app spawns another instance of `sys.executable` passing the relative script path as `sys.argv[1]`:
```bash
NanduLsd_Core deep_learning/machine_learning/analisis_xgboost.py <arg1> <arg2>
```
In `gui_app/main_app.py` (lines 25-79):
```python
if getattr(sys, 'frozen', False) and len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
    script_name = sys.argv[1]
    sys.argv = [script_name] + sys.argv[2:]
    module_name = script_name.replace('\\', '/').replace('.py', '').replace('/', '.')
    ...
    if hasattr(module, 'main'):
        module.main()
    sys.exit(0)
```

### 5.2 Subprocess Launching Bugs in Frozen Mode
In `gui_app/main_app.py` (lines 1711-1735), `_launch_dl_ml_script` contains a bug when frozen:
```python
  def _launch_dl_ml_script(self, script_rel_path):
    ...
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_abs_path = os.path.join(root_dir, script_rel_path.replace("/", os.sep))
    
    if not os.path.exists(script_abs_path):
      self.log_console.append(f"> ERROR: Script no encontrado: {script_abs_path}\n")
      return
      
    cmd = [sys.executable, script_abs_path] + rutas
```
**Why this breaks in the frozen executable:**
1. In frozen mode, loose `.py` files do not exist inside `root_dir` (which points to `_internal`). Thus `os.path.exists(script_abs_path)` returns `False`, aborting the launch immediately.
2. Even if bypassed, passing `script_abs_path` (an absolute path) causes `sys.argv[1]` in the child process to be `/tmp/.../deep_learning/machine_learning/analisis_xgboost.py`. The child process attempts `module_name = sys.argv[1].replace('/', '.')`, generating an invalid Python module name like `tmp._MEIxxxx.deep_learning...` and failing to load.

**Fix:**
In `_launch_dl_ml_script`, when `getattr(sys, 'frozen', False)`:
1. Skip `os.path.exists(script_abs_path)`.
2. Pass `script_rel_path` directly: `cmd = [sys.executable, script_rel_path] + rutas`.

---

## 6. Step-by-Step Fix Recommendations

### 6.1 Unified `crear_spec_ejecutable.py` Template
Replace the spec generator to dynamically locate and bundle all assets and C-extensions:

```python
# EMG_desarrollo/herramientas_build/crear_spec_ejecutable.py
import os
from PyInstaller.utils.hooks import collect_all, copy_metadata

def generar_spec():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(base_dir, "EMG_Ejecutable_Build")
    repo_root = os.path.dirname(base_dir)
    
    if not os.path.exists(build_dir):
        print("Carpeta 'EMG_Ejecutable_Build' no encontrada.")
        return

    # Dynamic asset detection
    datas_tuples = []
    
    candidate_assets = [
        ('icono.ico', '.'),
        ('config_general.json', '.'),
        ('metronome_config.json', '.'),
        ('palabras.txt', '.'),
        ('palabras.txt', 'acquisition'),
        ('archivos_md', 'archivos_md'),
        ('papers', 'papers'),
        ('logo_nandu_lsd.png', '.'),
    ]
    
    for src, dst in candidate_assets:
        local_path = os.path.join(build_dir, os.path.normpath(src))
        root_path = os.path.join(repo_root, os.path.normpath(src))
        if os.path.exists(local_path):
            datas_tuples.append(f"('{src}', '{dst}')")
        elif os.path.exists(root_path):
            datas_tuples.append(f"('{root_path}', '{dst}')")
            
    datas_str = ",\n        ".join(datas_tuples)

    spec_content = f"""# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_all, copy_metadata

nidaqmx_datas, nidaqmx_binaries, nidaqmx_hiddenimports = collect_all('nidaqmx')
sd_datas, sd_binaries, sd_hiddenimports = collect_all('sounddevice')
sf_datas, sf_binaries, sf_hiddenimports = collect_all('soundfile')
xgb_datas, xgb_binaries, xgb_hiddenimports = collect_all('xgboost')
umap_datas, umap_binaries, umap_hiddenimports = collect_all('umap')
sns_datas, sns_binaries, sns_hiddenimports = collect_all('seaborn')
tly_datas, tly_binaries, tly_hiddenimports = collect_all('tensorly')
nitypes_metadata = copy_metadata('nitypes')

block_cipher = None

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
    'deep_learning.machine_learning.analisis_xgboost',
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

hidden_imports = [
    'scipy.signal', 'scipy.special', 'scipy.io.wavfile', 'scipy.ndimage',
    'matplotlib.backends.backend_tkagg', 'matplotlib.backends.backend_qt5agg',
    'nidaqmx', 'sounddevice', 'soundfile', 'pyqtgraph', 'pandas', 'PIL',
    'PySide6', 'qdarkstyle', 'utils', 'acquisition', 'analysis', 'core', 'views',
    'deep_learning', 'tkinter', 'numba', 'tqdm', 'decouple', 'requests',
    'tzlocal', 'hightime', 'sklearn', 'sklearn.utils._typedefs',
    'sklearn.neighbors._typedefs', 'sklearn.neighbors._quad_tree',
    'sklearn.tree._utils', 'pynndescent',
] + additional_modules + nidaqmx_hiddenimports + sd_hiddenimports + sf_hiddenimports + xgb_hiddenimports + umap_hiddenimports + sns_hiddenimports + tly_hiddenimports

datas = [
    {datas_str}
] + nidaqmx_datas + sd_datas + sf_datas + xgb_datas + umap_datas + sns_datas + tly_datas + nitypes_metadata

binaries = nidaqmx_binaries + sd_binaries + sf_binaries + xgb_binaries + umap_binaries + sns_binaries + tly_binaries

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
```

### 6.2 Root-Level `build_linux.sh`
Provide a repository-root `build_linux.sh` script:
```bash
#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR/EMG_desarrollo"

echo "===================================================="
echo "NANDU LSD - Compilacion Multiplataforma (LINUX)"
echo "===================================================="

# Determine Python interpreter
if [ -f "$ROOT_DIR/venv/bin/python" ]; then
    PYTHON_EXEC="$ROOT_DIR/venv/bin/python"
    PYINSTALLER_EXEC="$ROOT_DIR/venv/bin/pyinstaller"
else
    PYTHON_EXEC="python3"
    PYINSTALLER_EXEC="pyinstaller"
fi

echo "[1/4] Creando entorno de compilacion temporal..."
$PYTHON_EXEC herramientas_build/crear_entorno_ejecutable.py

echo "[2/4] Aplicando parches de compatibilidad..."
$PYTHON_EXEC herramientas_build/aplicar_parches_ejecutable.py

echo "[3/4] Generando archivo .spec..."
$PYTHON_EXEC herramientas_build/crear_spec_ejecutable.py

echo "[4/4] Ejecutando PyInstaller..."
cd EMG_Ejecutable_Build
$PYINSTALLER_EXEC EMG_Studio.spec --noconfirm --clean
cd ..

echo "Finalizando estructura de distribucion..."
DIST_DIR="EMG_Ejecutable_Build/dist"
if [ -d "$DIST_DIR/NanduLsd" ]; then
    rm -rf "$DIST_DIR/NanduLsd"
fi
if [ -d "$DIST_DIR/NanduLsd_Core" ]; then
    mv "$DIST_DIR/NanduLsd_Core" "$DIST_DIR/NanduLsd"
fi

# Create bash launcher with argument forwarding
cat << 'EOF' > "$DIST_DIR/NanduLsd/run_nandu.sh"
#!/bin/bash
cd "$(dirname "$0")"
./NanduLsd_Core "$@"
EOF
chmod +x "$DIST_DIR/NanduLsd/run_nandu.sh"

# Sync to root build_linux directory
if [ -d "$ROOT_DIR/build_linux" ]; then
    rm -rf "$ROOT_DIR/build_linux/NanduLsd"
    cp -r "$DIST_DIR/NanduLsd" "$ROOT_DIR/build_linux/"
fi

echo "===================================================="
echo "BUILD COMPLETADO EXITOSAMENTE."
echo "Ejecutable listo en: $ROOT_DIR/build_linux/NanduLsd/run_nandu.sh"
echo "===================================================="
```

### 6.3 Root-Level `build.bat`
Provide a repository-root `build.bat` script:
```batch
@echo off
setlocal enabledelayedexpansion

set ROOT_DIR=%~dp0
cd /d "%ROOT_DIR%EMG_desarrollo"

echo ====================================================
echo NANDU LSD - Compilacion Multiplataforma (WINDOWS)
echo ====================================================

where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python no se encuentra en el PATH.
    exit /b 1
)

echo [1/4] Creando entorno de compilacion temporal...
python herramientas_build\crear_entorno_ejecutable.py

echo [2/4] Aplicando parches de compatibilidad...
python herramientas_build\aplicar_parches_ejecutable.py

echo [3/4] Generando archivo .spec...
python herramientas_build\crear_spec_ejecutable.py

echo [4/4] Ejecutando PyInstaller...
cd EMG_Ejecutable_Build
pyinstaller EMG_Studio.spec --noconfirm --clean
cd ..

set CSC_PATH=C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe
if exist "%CSC_PATH%" (
    echo Compilando Launcher C# nativo...
    "%CSC_PATH%" /nologo /target:winexe /out:EMG_Ejecutable_Build\dist\NanduLsd_Core\NanduLsd.exe /win32icon:icono.ico herramientas_build\launcher.cs
)

echo Finalizando estructura de distribucion...
if exist EMG_Ejecutable_Build\dist\NanduLsd rmdir /s /q EMG_Ejecutable_Build\dist\NanduLsd
if exist EMG_Ejecutable_Build\dist\NanduLsd_Core rename EMG_Ejecutable_Build\dist\NanduLsd_Core NanduLsd

if exist "%ROOT_DIR%build_windows" (
    if exist "%ROOT_DIR%build_windows\NanduLsd" rmdir /s /q "%ROOT_DIR%build_windows\NanduLsd"
    xcopy /E /I /Y "EMG_Ejecutable_Build\dist\NanduLsd" "%ROOT_DIR%build_windows\NanduLsd"
)

echo ====================================================
echo BUILD COMPLETADO EXITOSAMENTE.
echo Ejecutable listo en: %ROOT_DIR%build_windows\NanduLsd\NanduLsd.exe
echo ====================================================
```
