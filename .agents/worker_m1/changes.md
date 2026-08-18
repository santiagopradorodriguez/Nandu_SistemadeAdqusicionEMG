# Changes Log - Worker 1 (UI Architecture & Visual Polishing Specialist)

## 1. UI Parameter Collection (`EMG_desarrollo/gui_app/views/ui_analysis.py`)
- **Implemented `ProcessingTab.get_processing_kwargs()`**: Recovers all 15 parameters configured across the individual analysis tab (filter toggles, spectrogram frequency cutoff, notch Q factor, evolution range, envelope type, smoothing window, highpass and lowpass cutoffs, and excluded window list).
- **Implemented `AnalysisPanel.get_processing_kwargs()`**: Delegates directly to `self.tab_procesamiento.get_processing_kwargs()`.
- **Fixed `AnalysisPanel.get_trevisan_kwargs()`**: Replaced invalid `.value()` calls on `QLineEdit` widgets with safe float parsing and fallbacks (`alpha_ruido`, `snr_threshold`, `smooth_ms`, `n_pts_window`).
- **Cleaned Button Labels**: Removed emoji characters from GUI buttons (`btn_run_rapido`, `btn_run_sesion`, `btn_run_training`).

## 2. Linux Subprocess Compatibility (`EMG_desarrollo/gui_app/main_app.py`)
- **Guarded `CREATE_NEW_CONSOLE`**: Wrapped `creationflags=subprocess.CREATE_NEW_CONSOLE` with `if sys.platform == "win32":` in `ComparativeRunner.run()`, `SessionRunner.run()`, and `ProcessRunner.run()`, preventing `AttributeError` crashes on Linux.
- **Defensive Backend Setting**: Wrapped `matplotlib.use('TkAgg')` in a safe `try...except` block to avoid crashes in headless or non-Tk environments.

## 3. Results Gallery & Interactive Multi-Format Viewer (`EMG_desarrollo/gui_app/main_app.py`)
- **Recursive File Discovery**: Updated `_refrescar_visor_imagenes()` to recursively scan `resultados`, `deep_learning/pca_umap_clustering/resultados_pca_umap`, `resultados_umap_supervisado`, and `analisis_comparativos` for `.png`, `.jpg`, `.jpeg`, `.csv`, `.tex`, `.json`, and `.txt` files, sorted by newest modification time.
- **Interactive `ZoomableImageWidget`**: Created a zoomable viewer supporting zoom in (+), zoom out (-), reset (100% 1:1), fit to view, mouse double-click, and modal fullscreen dialog.
- **Dual-Tab Gallery**: Implemented a two-tab viewer (Tab 1: "Gráfico / Imagen" for visual plots; Tab 2: "Métricas y Tablas" with a `QTableWidget` for CSV datasets and a monospace `QTextEdit` for LaTeX, JSON, and text metrics).

## 4. 3D PCA Visual Projection Enhancements (`EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py`, `EMG_desarrollo/deep_learning/pca_analysis.py`, `EMG_desarrollo/deep_learning/umap_analysis.py`, `EMG_desarrollo/deep_learning/generador_umap_supervisado.py`, `EMG_desarrollo/analysis/pca_motor.py`)
- **2D Shadow Projections**: In `plot_scatter` (3D mode) and `plot_scatter_3d_multi_angle`, rendered 2D projected shadows on the bounding planes (floor XY at `z_min`, back wall XZ at `y_max`, side wall YZ at `x_min`) with subtle alpha transparency, preserving spatial orientation across all view angles.
- **Centroid Floor Drop Lines & Projected Ellipses**: In `plot_analisis_errores_3d`, added vertical dotted drop lines from 3D cluster centroids to the floor plane (`z_min`), diamond centroid markers on the floor, and 2D covariance confidence ellipses projected onto the floor.
- **Margin & Padding Adjustments**: In `guardar_tabla_imagen` and `plot_confusion_matrix_heatmap`, calculated dynamic canvas dimensions and applied `subplots_adjust` so that top column headers and titles are never clipped.
- **Strict Mathematical Integrity**: Kept all dimensionality reduction algorithms (`PCA`, `UMAP`, `KMeans`, `GaussianMixture`, `IsolationForest`) 100% genuine and unaltered.
- **License Headers & Emojis**: Added official NANDU LSD license headers to all modified files and replaced all emojis with text representations.
