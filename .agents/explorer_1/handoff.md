# Handoff Report: Explorer 1 (UI Architecture & Visual Polishing)

Date: 2026-08-17
From: Explorer 1 (UI Architecture & Visual Polishing Specialist)
To: Orchestrator / Implementer Agents
Scope: Requirements R1 and R3 for Nandu EMG Acquisition System (`EMG_desarrollo`)

---

## 1. Observation

### 1.1 UI Initialization and Parameter Collection
- **Observation 1.1.1**: In `EMG_desarrollo/gui_app/main_app.py:1453`, `_run_analysis()` invokes:
  ```python
  kwargs = self.analysis_panel.get_processing_kwargs()
  ```
  Inspection of `EMG_desarrollo/gui_app/views/ui_analysis.py` (lines 825-885) shows that `AnalysisPanel` defines only `__init__`, `get_trevisan_kwargs`, and `get_comparative_kwargs`. Method `get_processing_kwargs` is completely absent.
- **Observation 1.1.2**: In `EMG_desarrollo/gui_app/views/ui_analysis.py:864-870`, `AnalysisPanel.get_trevisan_kwargs()` contains:
  ```python
  def get_trevisan_kwargs(self):
      return {
          'alpha_ruido': self.tab_procesamiento.inp_alpha.value(),
          'snr_threshold': self.tab_procesamiento.inp_snr.value(),
          'smooth_ms': self.tab_procesamiento.inp_smooth.value(),
          'n_pts_window': self.tab_procesamiento.inp_n_pts.value()
      }
  ```
  `ProcessingTab` (lines 23-164) contains no `inp_alpha`, `inp_snr`, or `inp_n_pts` attributes, and `inp_smooth` is a `QLineEdit` without a `.value()` method.
- **Observation 1.1.3**: In `EMG_desarrollo/gui_app/main_app.py`, lines 956, 1150, and 1572:
  - Line 956 (`ComparativeRunner.run`): `p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)`
  - Line 1150 (`SessionRunner.run`): `p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)`
  - Line 1572 (`ProcessRunner.run`): `p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)`
  On Linux systems, `subprocess.CREATE_NEW_CONSOLE` does not exist in Python's standard library, causing an immediate runtime `AttributeError`.

### 1.2 3D PCA Visualization and Projection Planes
- **Observation 1.2.1**: In `EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py`:
  - `plot_scatter()` (lines 453-536) renders 3D scatter plots when `is_3d=True` on `Axes3D`.
  - `plot_scatter_3d_multi_angle()` (lines 619-660) renders 3 distinct 3D subplots for different camera azimuths (`azim=-60`, `azim=30`, `azim=120`).
  - `plot_analisis_errores_3d()` (lines 856-1001) renders 3D centroids, data points, and 3-sigma wireframe ellipsoids on `Axes3D`.
  - `plot_analisis_errores_3d_proyecciones_2d()` (lines 538-616) creates 3 separate 2D subplot panels (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3) instead of rendering projections onto the 3D bounding box planes.
- **Observation 1.2.2**: In all 3D plots, the PCA decomposition is computed via `sklearn.decomposition.PCA(n_components=3).fit_transform(X)` (e.g. line 1375 of `generador_pca_umap.py`). Coordinates $X_{\text{pca}} \in \mathbb{R}^{N \times 3}$ are passed directly to `Axes3D.scatter(X[:,0], X[:,1], X[:,2])`.

### 1.3 Machine Learning Results Gallery
- **Observation 1.3.1**: In `EMG_desarrollo/gui_app/main_app.py:1176-1181`, `_refrescar_visor_imagenes()` searches:
  ```python
  for path in paths_to_check:
      if os.path.exists(path):
          for file in glob.glob(os.path.join(path, "*.png")):
              basename = os.path.basename(file)
              self.cmb_resultados.addItem(f"{Path(path).name} / {basename}", file)
  ```
  However, in lines 1299, 1340, and 1394, results are saved into set subdirectories:
  `pca_umap_dir = os.path.join(project_root, "deep_learning", "pca_umap_clustering", "resultados_pca_umap", nombre_set)`
  The shallow `glob.glob("*.png")` fails to discover files within `nombre_set` subdirectories.
- **Observation 1.3.2**: In `EMG_desarrollo/gui_app/main_app.py:118-173`, `ImageLabel` scales images to fit within its current widget geometry:
  `super().setPixmap(self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))`
  Large figures (e.g. 26x8.5 inch 300 DPI figures) scaled to ~500x400 px become unreadable, with no zoom, scroll, or fullscreen view.
- **Observation 1.3.3**: In `generador_pca_umap.py:1285-1321`, `plot_confusion_matrix_heatmap()` applies `ax.xaxis.tick_top()`, `ax.xaxis.set_label_position('top')`, and `plt.title(title, pad=50)` together with `plt.tight_layout()`. `tight_layout` does not allocate adequate top margin for top-positioned labels and titles, producing title/label collisions or edge clipping.

---

## 2. Logic Chain

1. **Failure to Launch Individual Processing**:
   - Step 1: User clicks "PROCESAR Y CURAR INDIVIDUALES" in UI Analysis tab.
   - Step 2: Signal triggers `main_app.py:1434` (`_run_analysis()`).
   - Step 3: Line 1453 executes `self.analysis_panel.get_processing_kwargs()`.
   - Step 4: Python attempts to lookup `get_processing_kwargs` on `AnalysisPanel` instance.
   - Step 5: Since `AnalysisPanel` does not define `get_processing_kwargs`, Python raises `AttributeError: 'AnalysisPanel' object has no attribute 'get_processing_kwargs'`, crashing the processing handler.
   - Conclusion: `get_processing_kwargs` must be implemented on `AnalysisPanel` to extract all 15 parameters from `ProcessingTab`.

2. **Linux Subprocess Incompatibility**:
   - Step 1: User launches comparative analysis, session analysis, or batch processing on Linux.
   - Step 2: `ComparativeRunner`, `SessionRunner`, or `ProcessRunner` executes `p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)`.
   - Step 3: On Linux, `subprocess.CREATE_NEW_CONSOLE` is non-existent.
   - Step 4: Python raises `AttributeError: module 'subprocess' has no attribute 'CREATE_NEW_CONSOLE'`.
   - Conclusion: `creationflags` must only be passed when `sys.platform == "win32"`.

3. **2D Projection Planes in 3D PCA Visualizations**:
   - Step 1: PCA mathematical reduction generates an $N \times 3$ matrix of projected coordinates $[PC1, PC2, PC3]$.
   - Step 2: Matplotlib `Axes3D` defines spatial bounding limits: $x_{\min}, x_{\max}, y_{\min}, y_{\max}, z_{\min}, z_{\max}$.
   - Step 3: Plotting auxiliary 2D scatter calls with `zs=z_min, zdir='z'` (floor shadow), `zs=y_max, zdir='y'` (back wall shadow), and `zs=x_min, zdir='x'` (side wall shadow) places visual 2D projected shadows on the box planes.
   - Step 4: Adding vertical lines from centroids $(c_x, c_y, c_z)$ to $(c_x, c_y, z_{\min})$ provides unambiguous height/depth cues.
   - Step 5: Projecting 2D covariance ellipses onto $z = z_{\min}$ creates flat confidence contours on the floor.
   - Conclusion: This visual enhancement operates strictly in the Matplotlib rendering pass, leaving the mathematical algorithms of PCA, variance ratios, and cluster evaluations 100% intact.

4. **Empty Dropdown in Results Gallery**:
   - Step 1: ML routines save files in `deep_learning/.../resultados_pca_umap/<nombre_set>/*.png`.
   - Step 2: `_refrescar_visor_imagenes` scans `resultados_pca_umap/*.png`.
   - Step 3: No files match the shallow pattern at the parent level.
   - Step 4: `cmb_resultados` remains empty or fails to display newly completed runs.
   - Conclusion: `_refrescar_visor_imagenes` must recursively scan subdirectories (`os.walk` or `**/*.png`).

---

## 3. Caveats

1. **Tkinter Dialog Compatibility**: The processing engine uses Tkinter dialogs executed in child Python subprocesses (`temp_procesar.py`). This exploration focused on PySide6 parameter collection and bridge script generation; Tkinter internal event loops were not modified.
2. **Audio Database Layout**: Database recordings follow the mandatory structure `base_de_datos_electrodos/<Date>/<Session>/canal_{0..3}/grabacion.wav` with `metadata.json` in `canal_0`. Exploration verified that `SessionExplorer` and `_generar_base_dir_y_mediciones()` correctly parse this directory hierarchy.
3. **No Code Modification**: In accordance with the read-only exploration constraint, no source files in `EMG_desarrollo` were modified during this investigation.

---

## 4. Conclusion

The investigation successfully identified all critical architectural bottlenecks and visual polishing requirements:

1. **PySide6 UI Parameter Collection**:
   - Missing `AnalysisPanel.get_processing_kwargs()` in `EMG_desarrollo/gui_app/views/ui_analysis.py` is the primary blocker for individual processing.
   - Unconditional Windows-specific flags in `main_app.py` must be guarded with `if sys.platform == "win32"`.
2. **3D PCA 2D Projection Planes**:
   - 2D shadow projections on XY, XZ, YZ planes, centroid drop lines, and 2D floor ellipses can be added directly inside `plot_scatter()`, `plot_scatter_3d_multi_angle()`, and `plot_analisis_errores_3d()` in `generador_pca_umap.py` and `pca_analysis.py` without altering any PCA math.
3. **Machine Learning Results Gallery**:
   - `_refrescar_visor_imagenes()` must use recursive discovery to populate named experiment runs.
   - `panel_visor` should be upgraded with zoom/scroll controls and a dual-tab layout (Visualizations + Structured Metric Tables for `.csv`, `.tex`, `.json` data).
   - Matplotlib figure generation functions (`guardar_tabla_imagen`, `plot_confusion_matrix_heatmap`, `plot_recognition_rates_bar_chart`) require dynamic padding and constrained layout to prevent text clipping.

---

## 5. Verification Method

### 5.1 Verification Commands
To verify the findings independently:

1. **Verify missing method in `ui_analysis.py`**:
   ```bash
   python3 -c "
   import sys
   sys.path.insert(0, 'EMG_desarrollo/gui_app')
   from views.ui_analysis import AnalysisPanel
   from PySide6.QtWidgets import QApplication
   app = QApplication.instance() or QApplication(sys.argv)
   panel = AnalysisPanel()
   print('Has get_processing_kwargs:', hasattr(panel, 'get_processing_kwargs'))
   "
   ```
   *Expected result*: `Has get_processing_kwargs: False` (Confirms Observation 1.1.1).

2. **Verify Linux subprocess creationflags crash**:
   ```bash
   python3 -c "
   import subprocess
   try:
       flags = subprocess.CREATE_NEW_CONSOLE
       print('CREATE_NEW_CONSOLE exists')
   except AttributeError as e:
       print('AttributeError as predicted on Linux:', e)
   "
   ```
   *Expected result*: `AttributeError: module 'subprocess' has no attribute 'CREATE_NEW_CONSOLE'` on Linux.

3. **Verify shallow glob failure in results directory**:
   ```bash
   python3 -c "
   import os, glob
   root = 'EMG_desarrollo/deep_learning/pca_umap_clustering/resultados_pca_umap'
   shallow = glob.glob(os.path.join(root, '*.png'))
   recursive = glob.glob(os.path.join(root, '**', '*.png'), recursive=True)
   print('Shallow count:', len(shallow), 'Recursive count:', len(recursive))
   "
   ```

### 5.2 Files to Inspect
- `EMG_desarrollo/gui_app/main_app.py` (lines 571-597, 956, 1150, 1163-1195, 1453, 1572)
- `EMG_desarrollo/gui_app/views/ui_analysis.py` (lines 23-164, 825-885, 886-1081)
- `EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py` (lines 453-660, 856-1054, 1231-1350)
- `EMG_desarrollo/.agents/explorer_1/analysis.md` (comprehensive technical analysis report)
