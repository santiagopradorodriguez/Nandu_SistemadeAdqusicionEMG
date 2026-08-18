# Technical Analysis: UI Architecture & Visual Polishing (Requirements R1 & R3)

Date: 2026-08-17
Investigator: Explorer 1 (UI Architecture & Visual Polishing Specialist)
Scope: PySide6 UI Initialization, 3D PCA Visualization with 2D Projection Planes, and ML Results Gallery Formatting

---

## 1. Executive Summary

This investigation analyzed the PySide6 UI architecture, 3D PCA visual rendering, and the Machine Learning Results Gallery in the Nandu EMG acquisition system (`EMG_desarrollo`). 

Key findings:
1. **Critical UI Initialization & Parameter Collection Bugs (Requirement R1)**:
   - `AnalysisPanel` in `EMG_desarrollo/gui_app/views/ui_analysis.py` is missing the `get_processing_kwargs()` method entirely, causing an immediate `AttributeError` when triggering processing from `main_app.py:1453`.
   - `AnalysisPanel.get_trevisan_kwargs()` references non-existent widgets on `ProcessingTab` (`inp_alpha`, `inp_snr`, `inp_n_pts`), which would also raise `AttributeError`.
   - Cross-platform runtime crashes exist in `main_app.py` lines 956, 1150, and 1572 where `creationflags=subprocess.CREATE_NEW_CONSOLE` is passed unconditionally on Linux/macOS, causing `AttributeError: module 'subprocess' has no attribute 'CREATE_NEW_CONSOLE'`.
   - Module shadowing exists between `EMG_desarrollo/views/config_dialog.py` (legacy) and `EMG_desarrollo/gui_app/views/config_dialog.py`.

2. **3D PCA Visualization & 2D Projection Planes (Requirement R3)**:
   - 3D PCA scatter plots are rendered via Matplotlib `Axes3D` in `generador_pca_umap.py`, `pca_analysis.py`, and `analysis/pca_motor.py`.
   - Adding 2D projection planes (shadows on XY floor, XZ back wall, YZ side wall, drop lines from points/centroids to floor, and projected 2D covariance ellipses) can be implemented purely in the rendering layer using `Axes3D.scatter(..., zs=..., zdir=...)` and `Axes3D.plot(...)` without modifying any PCA mathematical decomposition (`sklearn.decomposition.PCA`).

3. **Machine Learning Results Gallery & Metric Presentation (Requirement R3)**:
   - The dropdown refresh function `_refrescar_visor_imagenes` in `main_app.py:1178` uses shallow non-recursive `glob.glob("*.png")`, failing to locate results saved in named set subdirectories (`resultados_pca_umap/<set_name>/`).
   - ImageLabel scales down large figures (like 3-panel 26x8.5 inch figures or wide metric tables) without scroll or zoom capabilities, rendering text unreadable.
   - Text/metric files (`.csv`, `.tex`, `.json`, `.txt`) are not loaded or visible in the ML Gallery.
   - Matplotlib table generation in `guardar_tabla_imagen()` and `plot_confusion_matrix_heatmap()` has layout clipping due to `tight_layout()` colliding with top-positioned axis labels (`tick_top()`) and legends outside axis boundaries.

---

## 2. Investigation 1: PySide6 UI Initialization & Parameter Collection

### 2.1 Missing `get_processing_kwargs()` in `AnalysisPanel`
- **Location**: `EMG_desarrollo/gui_app/views/ui_analysis.py:826-885`
- **Call site**: `EMG_desarrollo/gui_app/main_app.py:1453` (`kwargs = self.analysis_panel.get_processing_kwargs()`)
- **Direct Observation**:
  In `main_app.py:1453`:
  ```python
  kwargs = self.analysis_panel.get_processing_kwargs()
  ```
  In `ui_analysis.py`, `AnalysisPanel` only defines:
  - `__init__(self)` (lines 827-863)
  - `get_trevisan_kwargs(self)` (lines 864-870)
  - `get_comparative_kwargs(self)` (lines 872-882)
  There is NO `get_processing_kwargs(self)` method in `AnalysisPanel` or in `ProcessingTab`.
- **Expected Keys**:
  In `main_app.py:1502-1520`, the bridge script expects the following dictionary structure:
  ```python
  {
      'mostrar_recortes': bool,           # from tab_procesamiento.chk_recortes.isChecked()
      'mostrar_senal_cruda': bool,        # from tab_procesamiento.chk_cruda.isChecked()
      'tema_cyberpunk': bool,             # from tab_procesamiento.chk_cyberpunk.isChecked()
      'mostrar_espectrograma': bool,      # from tab_procesamiento.chk_espectrograma.isChecked()
      'frecuenciamaxima': str or float,   # from tab_procesamiento.inp_spec_fmax.text()
      'apply_notch_filter': bool,         # from tab_procesamiento.chk_notch.isChecked()
      'notch_q_factor': str or float,     # from tab_procesamiento.inp_notch_q.text()
      'mostrar_evolucion': bool,          # from tab_procesamiento.chk_evolucion.isChecked()
      'evol_t_start': str or float,       # from tab_procesamiento.inp_evol_start.text()
      'evol_t_end': str or float,         # from tab_procesamiento.inp_evol_end.text()
      'excluded_windows_list': list[int], # parsed from tab_procesamiento.inp_excluded.text()
      'tipo_envolvente': str,             # from tab_procesamiento.cmb_tipo_env.currentText()
      'smooth_ms': str or float,          # from tab_procesamiento.inp_smooth.text()
      'highpass_cutoff_hz': str or float, # from tab_procesamiento.inp_hp.text()
      'lowpass_cutoff_hz': str or float   # from tab_procesamiento.inp_lp.text()
  }
  ```
- **Recommended Fix Strategy**:
  Implement `get_processing_kwargs(self)` on `AnalysisPanel` (or delegate from `ProcessingTab`):
  ```python
  def get_processing_kwargs(self):
      t = self.tab_procesamiento
      excl_raw = t.inp_excluded.text().strip()
      excl_list = []
      if excl_raw:
          for part in excl_raw.split(','):
              part = part.strip()
              if part.isdigit():
                  excl_list.append(int(part))
      return {
          'mostrar_recortes': t.chk_recortes.isChecked(),
          'mostrar_senal_cruda': t.chk_cruda.isChecked(),
          'tema_cyberpunk': t.chk_cyberpunk.isChecked(),
          'mostrar_espectrograma': t.chk_espectrograma.isChecked(),
          'frecuenciamaxima': t.inp_spec_fmax.text().strip() or "5000",
          'apply_notch_filter': t.chk_notch.isChecked(),
          'notch_q_factor': t.inp_notch_q.text().strip() or "2.0",
          'mostrar_evolucion': t.chk_evolucion.isChecked(),
          'evol_t_start': t.inp_evol_start.text().strip() or "10",
          'evol_t_end': t.inp_evol_end.text().strip() or "1000",
          'excluded_windows_list': excl_list,
          'tipo_envolvente': t.cmb_tipo_env.currentText(),
          'smooth_ms': t.inp_smooth.text().strip() or "50",
          'highpass_cutoff_hz': t.inp_hp.text().strip() or "20",
          'lowpass_cutoff_hz': t.inp_lp.text().strip() or "500"
      }
  ```

### 2.2 Broken `get_trevisan_kwargs()` in `AnalysisPanel`
- **Location**: `EMG_desarrollo/gui_app/views/ui_analysis.py:864-870`
- **Direct Observation**:
  ```python
  def get_trevisan_kwargs(self):
      return {
          'alpha_ruido': self.tab_procesamiento.inp_alpha.value(),
          'snr_threshold': self.tab_procesamiento.inp_snr.value(),
          'smooth_ms': self.tab_procesamiento.inp_smooth.value(),
          'n_pts_window': self.tab_procesamiento.inp_n_pts.value()
      }
  ```
  `ProcessingTab` does not contain `inp_alpha`, `inp_snr`, or `inp_n_pts`. `inp_smooth` is a `QLineEdit` without a `.value()` method.
- **Recommended Fix**: Update `get_trevisan_kwargs()` to point to the correct tab or use default fallback values if called.

### 2.3 Cross-Platform Subprocess Crash on Linux (`subprocess.CREATE_NEW_CONSOLE`)
- **Locations**:
  1. `EMG_desarrollo/gui_app/main_app.py:956` in `ComparativeRunner.run()`
  2. `EMG_desarrollo/gui_app/main_app.py:1150` in `SessionRunner.run()`
  3. `EMG_desarrollo/gui_app/main_app.py:1572` in `ProcessRunner.run()`
- **Direct Observation**:
  ```python
  p = subprocess.run([sys.executable, self.spath], creationflags=subprocess.CREATE_NEW_CONSOLE)
  ```
  On Linux, `subprocess.CREATE_NEW_CONSOLE` is undefined in Python's standard library. When running on Linux, executing comparative plots, session evolution, or batch processing raises `AttributeError: module 'subprocess' has no attribute 'CREATE_NEW_CONSOLE'`.
- **Recommended Fix**:
  Guard with platform check across all runner threads:
  ```python
  run_kwargs = {}
  if sys.platform == "win32":
      run_kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
  p = subprocess.run([sys.executable, self.spath], **run_kwargs)
  ```

### 2.4 Duplicate / Shadowing of `views/config_dialog.py`
- **Locations**:
  - Legacy: `EMG_desarrollo/views/config_dialog.py` (212 lines)
  - Current: `EMG_desarrollo/gui_app/views/config_dialog.py` (257 lines)
- **Direct Observation**:
  In `main_app.py:20-22`:
  ```python
  root_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  sys.path.insert(0, root_project_dir)
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  ```
  Because both `root_project_dir` and `gui_app` are added to `sys.path`, `from views.config_dialog import ConfiguracionDialog` in `main_app.py:373` relies on path insertion order. Clean resolution requires standardizing imports on `views.config_dialog` from `gui_app/views/` and archiving or removing redundant root `views/config_dialog.py`.

---

## 3. Investigation 2: 3D PCA Visualization & 2D Projection Planes

### 3.1 3D PCA Rendering Architecture
- **Primary Generator**: `EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py`
- **Functions Generating 3D Visualizations**:
  1. `plot_scatter(X_proj, Y, title, output_path, is_3d=True, variance_ratios=...)` (lines 453-536): Standard 3D scatter plot.
  2. `plot_scatter_3d_multi_angle(X_proj, Y, title, output_path, variance_ratios=...)` (lines 619-660): 3 subplots from 3 camera angles (Frontal azim=-60, Lateral azim=30, Posterior azim=120).
  3. `plot_analisis_errores_3d(X, Y, Tomas, title, output_path, variance_ratios=...)` (lines 856-1001): 3D scatter with centroids, 3-sigma wireframe ellipsoids, misclassified points.
  4. `plot_analisis_errores_3d_proyecciones_2d(X_proj, Y, title, output_path, variance_ratios=...)` (lines 538-616): 3 separate 2D planar projection subplots (PC1 vs PC2, PC1 vs PC3, PC2 vs PC3) with decision boundaries.
- **Secondary Scripts**:
  - `EMG_desarrollo/deep_learning/pca_analysis.py` (lines 427-510, 593-634, 829-974)
  - `EMG_desarrollo/analysis/pca_motor.py` (lines 264-289)

### 3.2 Strategy to Add 2D Projection Planes (Shadows on XY, XZ, YZ)
To provide depth cues and cluster disambiguation without modifying PCA math:

1. **Floor Projection (XY Plane at $z = z_{\min}$)**:
   Extract spatial limits:
   ```python
   x_min, x_max = ax.get_xlim()
   y_min, y_max = ax.get_ylim()
   z_min, z_max = ax.get_zlim()
   ```
   For each class/vocal $k$:
   ```python
   # Shadow on floor (XY plane)
   ax.scatter(
       X_proj[idx, 0], X_proj[idx, 1],
       zs=z_min, zdir='z',
       color=color, s=25, alpha=0.20,
       edgecolors='none', depthshade=False, zorder=1
   )
   ```

2. **Back Wall Projection (XZ Plane at $y = y_{\max}$)**:
   ```python
   # Shadow on back wall (XZ plane)
   ax.scatter(
       X_proj[idx, 0], X_proj[idx, 2],
       zs=y_max, zdir='y',
       color=color, s=25, alpha=0.15,
       edgecolors='none', depthshade=False, zorder=1
   )
   ```

3. **Side Wall Projection (YZ Plane at $x = x_{\min}$)**:
   ```python
   # Shadow on side wall (YZ plane)
   ax.scatter(
       X_proj[idx, 1], X_proj[idx, 2],
       zs=x_min, zdir='x',
       color=color, s=25, alpha=0.15,
       edgecolors='none', depthshade=False, zorder=1
   )
   ```

4. **Vertical Drop Lines for Centroids**:
   In `plot_analisis_errores_3d()`:
   ```python
   # Drop line from 3D centroid to floor
   ax.plot(
       [centroid[0], centroid[0]],
       [centroid[1], centroid[1]],
       [z_min, centroid[2]],
       color=color, linestyle=':', linewidth=1.0, alpha=0.6, zorder=2
   )
   # Shadow centroid on floor
   ax.scatter(
       [centroid[0]], [centroid[1]],
       zs=z_min, zdir='z',
       marker='D', s=70, color=color, alpha=0.35, edgecolors='gray', zorder=2
   )
   ```

5. **Floor-Projected 2D Covariance Ellipses**:
   For each cluster, project the 2D ellipse onto the floor $z = z_{\min}$:
   ```python
   # Calculate 2D covariance on (PC1, PC2)
   cov_2d = np.cov(X_true[:, :2], rowvar=False)
   evals_2d, evecs_2d = np.linalg.eigh(cov_2d)
   radii_2d = np.sqrt(np.maximum(evals_2d, 1e-9)) * 3
   theta = np.linspace(0, 2 * np.pi, 50)
   ell_pts = np.array([radii_2d[0] * np.cos(theta), radii_2d[1] * np.sin(theta)])
   ell_rot = evecs_2d @ ell_pts
   x_ell_floor = ell_rot[0, :] + centroid[0]
   y_ell_floor = ell_rot[1, :] + centroid[1]
   ax.plot(
       x_ell_floor, y_ell_floor,
       zs=z_min, zdir='z',
       color=color, alpha=0.3, linestyle='--', linewidth=0.8, zorder=2
   )
   ```

### 3.3 Confirmation of Mathematical Invariance
- `sklearn.decomposition.PCA(n_components=3)` is unchanged.
- `fit_transform()` computation is unchanged.
- Explained variance ratios, singular values, coordinate tensors, and Hungarian cluster mappings are untouched.
- Only visual presentation artists (`Axes3D.scatter`, `Axes3D.plot`) receive secondary 2D projected coordinates with fixed plane offsets (`zs=z_min`, `zs=y_max`, `zs=x_min`).

---

## 4. Investigation 3: Machine Learning Results Gallery & Metric Display

### 4.1 Root Causes of Gallery & Metric Display Issues

1. **Subfolder Search Limitation in `_refrescar_visor_imagenes()`**:
   - Location: `EMG_desarrollo/gui_app/main_app.py:1163-1184`
   - Code:
     ```python
     for path in paths_to_check:
         if os.path.exists(path):
             for file in glob.glob(os.path.join(path, "*.png")):
                 basename = os.path.basename(file)
                 self.cmb_resultados.addItem(f"{Path(path).name} / {basename}", file)
     ```
   - When PCA or UMAP runs from GUI (`main_app.py:1299, 1340, 1394`), results are saved in:
     `deep_learning/pca_umap_clustering/resultados_pca_umap/<nombre_set>/`
     `deep_learning/resultados_umap_supervisado/<nombre_set>/`
   - `glob.glob(os.path.join(path, "*.png"))` is shallow and ignores all subdirectories.
   - Consequently, newly generated results never appear in the dropdown!

2. **Table & Metric Image Truncation / Overlap during Generation**:
   - In `guardar_tabla_imagen()` (`generador_pca_umap.py:1231-1258`):
     - `table.scale(1, 1.5)` expands the table vertically without adjusting figure bounding box.
     - `plt.title(title, pad=15)` collides with table headers when `ax.axis('tight')` is active.
     - `bbox_inches='tight'` clips side row labels if column width multiplier is insufficient for long strings.
   - In `plot_confusion_matrix_heatmap()` (`generador_pca_umap.py:1285-1321`):
     - `ax.xaxis.tick_top()` + `ax.xaxis.set_label_position('top')` + `plt.title(title, pad=50)` + `plt.tight_layout()`.
     - Matplotlib's `tight_layout()` does not calculate top padding correctly when top ticks, top x-label, and large title padding are combined, causing the title to overlap the "Prediccion" label or get clipped at the image border.
   - In `plot_recognition_rates_bar_chart()` (`generador_pca_umap.py:1003-1054`):
     - `ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))` places legend outside the plot box without `bbox_extra_artists=(legend,)`, resulting in clipped legend text on export.

3. **Single ImageLabel vs Rich Multi-Format Results Display**:
   - `panel_visor` (`main_app.py:571-597`) consists only of a `QComboBox`, a "Refrescar" button, and an `ImageLabel`.
   - It cannot display non-image outputs generated by the ML pipeline:
     - `reporte_mediciones_descartadas_PCA_3D.csv`
     - `matriz_confusion_pca_3d.tex`
     - `matriz_distancias_pca_3d.tex`
     - `silhouette_pca_3d.tex`
     - `parametros.json`
   - `ImageLabel` does not support interactive zooming (+, -, 1:1), panning with scrollbars, or modal fullscreen expansion (unlike `ClickableImage` in `comparative_explorer_widget.py`).

### 4.2 Recommended Architecture for the Results Gallery

1. **Recursive Subfolder Discovery in `_refrescar_visor_imagenes()`**:
   ```python
   def _refrescar_visor_imagenes(self):
       self.cmb_resultados.clear()
       root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
       paths_to_check = [
           os.path.join(root_dir, "resultados"),
           os.path.join(root_dir, "deep_learning", "pca_umap_clustering", "resultados_pca_umap"),
           os.path.join(root_dir, "deep_learning", "resultados_umap_supervisado")
       ]
       found_files = []
       for base_path in paths_to_check:
           if os.path.exists(base_path):
               for root, _, files in os.walk(base_path):
                   for f in files:
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.csv', '.tex', '.json', '.txt')):
                           full_path = os.path.join(root, f)
                           rel_path = os.path.relpath(full_path, base_path)
                           mtime = os.path.getmtime(full_path)
                           category = os.path.basename(base_path)
                           found_files.append((mtime, f"[{category}] {rel_path}", full_path))
       
       # Sort newest first
       found_files.sort(key=lambda x: x[0], reverse=True)
       for _, label, path in found_files:
           self.cmb_resultados.addItem(label, path)
           
       if self.cmb_resultados.count() > 0:
           self.cmb_resultados.setCurrentIndex(0)
   ```

2. **Dual-Tab Gallery (Visualizations + Structured Metrics)**:
   Refactor `panel_visor` to contain a `QTabWidget`:
   - **Tab 1: "Gráficos"**:
     - `QScrollArea` containing an enhanced `ClickableImage` / `ZoomableImageWidget`.
     - Controls: Zoom In (+), Zoom Out (-), Reset (1:1), Fit to Window, Fullscreen on double click.
   - **Tab 2: "Métricas y Tablas (CSV / LaTeX / JSON)"**:
     - `QTableWidget` for CSV files (interactive sorting, clear cell borders, zebra striping).
     - `QTextEdit` with syntax highlighting / monospace font for LaTeX and JSON files.

3. **Matplotlib Table and Heatmap Layout Fixes**:
   - In `plot_confusion_matrix_heatmap`:
     Use `fig.subplots_adjust(top=0.82, bottom=0.12, left=0.18, right=0.92)` instead of `plt.tight_layout()`, or use constrained layout `fig = plt.figure(figsize=(7.5, 6), layout='constrained')`.
   - In `guardar_tabla_imagen`:
     Compute dynamic figure dimensions based on max text length and number of rows:
     ```python
     max_col_len = max(max(len(str(c)) for c in df.columns), max(len(str(v)) for v in df.values.flatten()))
     col_w = max(1.8, max_col_len * 0.18)
     fig, ax = plt.subplots(figsize=(df.shape[1] * col_w + 1.5, (df.shape[0] + 2) * 0.65), facecolor='white')
     ```
   - In `plot_recognition_rates_bar_chart`:
     Pass `bbox_inches='tight'` with `bbox_extra_artists` or reserve space with `subplots_adjust(right=0.75)`.

---

## 5. Synthesis & Priority Action Plan

| Priority | Component | File | Issue | Recommended Action |
|---|---|---|---|---|
| P0 (Blocker) | Parameter Collection | `EMG_desarrollo/gui_app/views/ui_analysis.py` | `AnalysisPanel.get_processing_kwargs()` missing | Implement `get_processing_kwargs()` on `AnalysisPanel` mapping all 15 GUI controls. |
| P0 (Blocker) | Parameter Collection | `EMG_desarrollo/gui_app/views/ui_analysis.py` | `AnalysisPanel.get_trevisan_kwargs()` invalid widget references | Fix widget references in `get_trevisan_kwargs()`. |
| P0 (Blocker) | Process Execution | `EMG_desarrollo/gui_app/main_app.py` | `subprocess.CREATE_NEW_CONSOLE` crashes on Linux (lines 956, 1150, 1572) | Wrap with `if sys.platform == "win32"` check. |
| P1 (High) | ML Gallery | `EMG_desarrollo/gui_app/main_app.py` | `_refrescar_visor_imagenes` non-recursive glob (line 1178) | Use `os.walk` or recursive glob `**/*.png` to discover named experiment folders. |
| P1 (High) | 3D Visualization | `EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py` | Lack of 2D projection planes in 3D scatter plots | Add floor (XY at $z_{\min}$), back wall (XZ at $y_{\max}$), side wall (YZ at $x_{\min}$) scatter projections, centroid drop lines, and projected 2D covariance ellipses. |
| P2 (Medium) | Metric Layout | `generador_pca_umap.py` / `generador_umap_supervisado.py` | Confusion matrix and table title/header truncation with `tight_layout` | Use constrained layout or explicit `subplots_adjust` with dynamic padding. |
| P2 (Medium) | ML Gallery UX | `EMG_desarrollo/gui_app/main_app.py` | ImageLabel lacks zoom/scroll; non-image metric files (.csv, .tex) unsupported | Upgrade `panel_visor` with zoom controls, QScrollArea, and dual-tab view (Images + Metric Tables). |
