# Handoff Report: Requirement R2 (Multi-Platform Packaging & Builds)

## 1. Observation

Direct observations from codebase inspection, tool executions, and empirical AST/import analysis:

1. **Build Error in Existing Dist**:
   File `EMG_desarrollo/EMG_Ejecutable_Build/dist/NanduLsd/multiplexer_error.log` (lines 1-11):
   ```text
   Error al ejecutar deep_learning.machine_learning.analisis_xgboost: No module named 'seaborn'
   Traceback (most recent call last):
     File "main_app.py", line 111, in <module>
     File "importlib/__init__.py", line 90, in import_module
     File "deep_learning/machine_learning/analisis_xgboost.py", line 55, in <module>
   ModuleNotFoundError: No module named 'seaborn'
   ```

2. **Missing Spec Assets & Incomplete Datas**:
   `EMG_desarrollo/herramientas_build/crear_spec_ejecutable.py` (lines 28-39):
   - Resources `icons`, `DataConfig/Pictures`, and `gui_app/assets` do not exist in the filesystem.
   - `justificacion_matematica.md` is looked up at root instead of `archivos_md/justificacion_matematica.md`.
   - `config_general.json`, `metronome_config.json`, `logo_nandu_lsd.png`, `archivos_md/`, and `papers/` are completely omitted from `datas`.

3. **Obsolete / Broken Modules in Spec**:
   Testing module imports in `crear_spec_ejecutable.py` `additional_modules`:
   - `analysis.analisis_por_track_integrado_experimental`: `ModuleNotFoundError: No module named 'analysis.analisis_por_track_integrado_experimental'`
   - `analysis.feature_extractor`: `ModuleNotFoundError: No module named 'analysis.feature_extractor'`

4. **Missing Hidden Imports & Cython / C-Extensions**:
   Inspection of `crear_spec_ejecutable.py` and `requirements.txt`:
   - `seaborn`, `xgboost`, `tensorly`, `umap` (`umap-learn`), `numba`, `tqdm`, `pynndescent` are imported in the codebase but missing from `hidden_imports` and `collect_all` declarations.
   - `pathex` in `crear_spec_ejecutable.py` only contained `['.', 'gui_app']`, causing unprefixed imports (`import modelos`, `import dataset_emg`, `import analisis_trevisan`) to fail resolution during build.

5. **In-App Subprocess Launching Bug in Frozen Mode**:
   `EMG_desarrollo/gui_app/main_app.py` (lines 1720-1731):
   - `_launch_dl_ml_script` executes `os.path.exists(script_abs_path)`. When running frozen from `_internal`, loose `.py` files do not exist on disk, causing the function to abort with `> ERROR: Script no encontrado`.
   - Furthermore, passing absolute path `script_abs_path` prevents the multiplexer (lines 25-33) from parsing the correct Python module name.

6. **Missing Root Build Wrappers and Empty Distribution Folders**:
   - `build_linux.sh` and `build.bat` do not exist at the repository root `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/`.
   - Root folders `build_linux/` and `build_windows/` are empty.

---

## 2. Logic Chain

1. From Observation 1, the compiled frozen binary crashed when executing `analisis_xgboost.py` because `seaborn` was missing from the packaged bundle.
2. From Observation 4, `seaborn`, `xgboost`, `tensorly`, and `umap` require their C-extensions, data files, and submodules to be collected via `collect_all` hooks in `crear_spec_ejecutable.py`.
3. From Observation 3, `additional_modules` in the spec generator included non-existent module names (`analysis.analisis_por_track_integrado_experimental` and `analysis.feature_extractor`), leading to import failures during multiplexer execution.
4. From Observation 2, critical JSON configuration files (`config_general.json`, `metronome_config.json`), graphical assets (`logo_nandu_lsd.png`), and markdown documentation (`archivos_md/`) are needed at runtime by `config_manager.py`, `metronomo_visual.py`, `main_app.py`, and `instrucciones_uso.py`. Because they were omitted from `datas`, frozen applications fail to load custom configurations or help documents.
5. From Observation 5, auxiliary machine learning scripts cannot be launched from the GUI in frozen mode because `_launch_dl_ml_script` attempts an absolute filesystem existence check on unpacked `.py` files and passes absolute paths to the multiplexer rather than relative module descriptors.
6. From Observation 6, developer ergonomics and automated CI/builds require unified entry points at both the repository root and `EMG_desarrollo/`.

---

## 3. Caveats

1. **NI-DAQmx Hardware Drivers on Linux**: The `nidaqmx` Python package collects successfully on Linux, but communicating with physical National Instruments hardware requires proprietary NI-DAQmx C drivers installed on the host system. The software correctly falls back to microphone / simulated mode when hardware drivers are absent.
2. **PyTorch Packaging Size vs Standalone Autoencoder**: `torch`, `torchvision`, and `torchaudio` are currently excluded (`excludes=['torch', 'torchvision', 'torchaudio']`) to keep the distributable size around 150-250MB. If PyTorch is included, the package expands to ~1.5GB. The GUI pipeline should gracefully inform users if PyTorch training is requested without torch installed.

---

## 4. Conclusion

Requirement R2 (Multi-platform packaging and builds) requires the following actionable changes:
1. Update `crear_spec_ejecutable.py` to:
   - Include `collect_all` for `nidaqmx`, `sounddevice`, `soundfile`, `xgboost`, `umap`, `seaborn`, and `tensorly`.
   - Clean up `additional_modules` by removing invalid paths and adding all valid analysis, utility, view, and deep learning modules.
   - Expand `pathex` to include all submodule roots (`deep_learning`, `acquisition`, `analysis`, `utils`, `views`, etc.).
   - Include all required runtime data files (`config_general.json`, `metronome_config.json`, `palabras.txt`, `icono.ico`, `logo_nandu_lsd.png`, `archivos_md/`, `papers/`).
2. Update `_launch_dl_ml_script` in `gui_app/main_app.py` to support frozen execution without absolute path checks on disk.
3. Update `requirements.txt` and `requirements_linux.txt` to include `seaborn`, `xgboost`, `tensorly`, `numba`, `tqdm`, `python-decouple`, `requests`, `tzlocal`, `hightime`.
4. Add repository-root `build_linux.sh` and `build.bat` scripts that wrap the build pipeline and deploy cleanly into `build_linux/NanduLsd` and `build_windows/NanduLsd`.

---

## 5. Verification Method

1. **Spec Generation Verification**:
   Execute:
   ```bash
   ./venv/bin/python EMG_desarrollo/herramientas_build/crear_spec_ejecutable.py
   ```
   Inspect `EMG_desarrollo/EMG_Ejecutable_Build/EMG_Studio.spec` to confirm all `datas`, `binaries`, `hiddenimports`, and `pathex` entries are present.

2. **Clean Linux Build Verification**:
   Execute:
   ```bash
   ./build_linux.sh
   ```
   Verify that:
   - Build completes without error.
   - `build_linux/NanduLsd/NanduLsd_Core` and `build_linux/NanduLsd/run_nandu.sh` exist and have execution permissions.
   - Running `./build_linux/NanduLsd/run_nandu.sh deep_learning/machine_learning/analisis_xgboost.py` invokes the multiplexer cleanly without `ModuleNotFoundError: No module named 'seaborn'`.

3. **Configuration & Data Files Verification**:
   Inspect `build_linux/NanduLsd/_internal/` or `EMG_Ejecutable_Build/dist/NanduLsd/_internal/` to verify presence of `config_general.json`, `metronome_config.json`, `palabras.txt`, `icono.ico`, and `archivos_md/`.
