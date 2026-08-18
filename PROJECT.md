# Project: Nandu EMG Acquisition System (Curation, Packaging, Polishing, & Reporting)

## Architecture
- **GUI Framework**: PySide6 (Qt) in `EMG_desarrollo/gui_app/`, `EMG_desarrollo/views/`
- **Acquisition & DSP**: `EMG_desarrollo/acquisition/`, `EMG_desarrollo/analysis/`
- **Machine Learning & Deep Learning**: `EMG_desarrollo/deep_learning/` (PCA, Supervised UMAP, Autoencoders)
- **Database Layout**: `base_de_datos_electrodos/<Date>/<Session>/canal_{0,1,2,3}/`
- **Packaging / Builds**: PyInstaller spec and build scripts in `EMG_desarrollo/herramientas_build/`, `build_linux.sh`, `build.bat`
- **Documentation & Report**: `README.md`, `CONTRIBUTING.md`, `software.tex`

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | UI Architecture Curation & Visual Polishing (R1 & R3) | Fix UI parameter collection, Linux subprocesses, 3D PCA projections, and ML Results Gallery | none | DONE |
| M2 | Multi-platform Build & Packaging (R2) | Improve PyInstaller specs, dependencies, datas, and root build scripts for Linux and Windows | none | DONE |
| M3 | Software & Repository Documentation (R4) | Modernize README.md, sanitize emojis in CONTRIBUTING.md, update in-app instrucciones_uso.py | M1 | DONE |
| M4 | Academic Report software.tex (R5) | Author non-technical academic section software.tex matching user vault style; verify pdflatex compilation | none | DONE |
| M5 | Final E2E Verification & Git Commit (R1-R5) | Full acceptance criteria test suite (Tiers 1-4), adversarial tests, clean git commit | M1, M2, M3, M4 | IN_PROGRESS |

## Interface Contracts
### UI ↔ DSP/ML Pipeline
- `AnalysisPanel.get_processing_kwargs()` must return valid dict matching `ProcessingTab` widgets.
- `subprocess.run` calls must guard `creationflags=subprocess.CREATE_NEW_CONSOLE` with `if sys.platform == 'win32'`.
- 3D PCA visualizations must add 2D shadow projections onto $z_{\min}, y_{\max}, x_{\min}$ planes and centroid drop lines without modifying `PCA.fit_transform` or any mathematical decomposition.
- Results gallery must recursively discover results in `resultados_pca_umap/<set_name>/` and provide zoom/scroll for figures and structured tables for metrics.

### Packaging & Builds
- `crear_spec_ejecutable.py` bundles `seaborn`, `xgboost`, `umap`, `tensorly`, `nidaqmx`, `sounddevice`, `soundfile` via `collect_all`.
- All runtime assets (`config_general.json`, `metronome_config.json`, `palabras.txt`, `icono.ico`, `logo_nandu_lsd.png`, `archivos_md/`, `papers/`) included in `datas`.
- Root `build_linux.sh` and `build.bat` build to `build_linux/NanduLsd` and `build_windows/NanduLsd`.

## Code Layout
- Main app: `EMG_desarrollo/`
- Packaging: `herramientas_build/`, `build_linux.sh`, `build.bat`
- Documentation: `README.md`, `CONTRIBUTING.md`, `EMG_desarrollo/instrucciones_uso.py`
- Report: `software.tex`
