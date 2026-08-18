# Empirical Challenge Report - Challenger 2 (Milestone M1)

## Challenge Summary

**Overall risk assessment**: LOW

Empirical testing and adversarial stress verification were performed against Requirements R1 (Results Gallery recursive discovery and preview engine) and R3 (3D PCA plot rendering, floor/wall projections, centroid drop lines, floor ellipses, confusion matrix heatmap, and metric table exports).

All 43 automated unit and stress tests executed in the Python 3 virtual environment passed with 100% success rate without crashes or data corruption.

---

## Challenges

### [Low] Challenge 1: Corrupted or Malformed Files in Results Gallery
- **Assumption challenged**: The gallery assumes discovered files (`.png`, `.csv`, `.tex`, `.json`, `.txt`) are well-formed and valid for reading/parsing.
- **Attack scenario**: Injected 0-byte image files, syntactically broken CSVs (unclosed quotes, ragged row lengths), malformed JSON strings, and non-UTF-8 binary text files into the gallery search path.
- **Blast radius**: Potential UI freeze, unhandled exception, or application crash when user selects a corrupted artifact from the gallery combobox.
- **Observed behavior**: The gallery gracefully handled all invalid inputs:
  - 0-byte/corrupted PNGs resulted in null pixmaps with placeholder display.
  - Corrupted CSVs were caught by `try...except` and rendered diagnostic error text in `txt_metricas_visor` instead of crashing.
  - Non-UTF-8 text was loaded using `errors='replace'` without raising `UnicodeDecodeError`.
- **Mitigation**: Existing fallback exception blocks in `_cargar_imagen_visor()` are robust and prevent GUI crashes.

### [Low] Challenge 2: Degenerate 3D Point Distributions and Singular Covariance Matrices
- **Assumption challenged**: 3D PCA error analysis plots calculate spatial confidence ellipsoids and 2D floor ellipses via covariance eigen-decomposition (`np.linalg.eigh(cov)`), which assumes non-singular covariance matrices.
- **Attack scenario**: Injected collinear data (rank-1 covariance matrix), planar data ($z=0$ constant across all points), single-sample clusters, and extreme coordinates ($\pm 10^6$).
- **Blast radius**: Numerical singularity, division by zero in radius scaling, or crash during wireframe/ellipse plotting.
- **Observed behavior**: The implementation wraps 3D wireframe ellipsoid calculation and 2D floor ellipse calculation in guarded `try...except` blocks and uses `np.maximum(evals_2d, 1e-9)` to clamp non-positive eigenvalues. For collinear rank-1 data, a minor `RuntimeWarning: invalid value encountered in sqrt` is emitted internally on line 1004 of `generador_pca_umap.py` and cleanly caught without interrupting plot generation or crashing the script.
- **Mitigation**: Plot generation finishes cleanly and writes complete 300 DPI image files.

### [Low] Challenge 3: Label Clipping on Large or High-Dimension Metric Tables
- **Assumption challenged**: `guardar_tabla_imagen` and `plot_confusion_matrix_heatmap` must accommodate arbitrarily long column/row names and titles without clipping or truncation.
- **Attack scenario**: Rendered tables with 60+ character column headers, 60+ character row labels, 100+ character titles, and dense 15x8 tables containing special characters (`$`, `_`, `%`, `&`, `#`, `<` `>`).
- **Blast radius**: Truncated metric labels in published thesis figures or cut-off headers in exported PNGs.
- **Observed behavior**: The dynamic width/height calculation (`col_w = max(col_width, max_col_len * 0.16)`, `fig_w = max(7.0, df.shape[1] * col_w + extra_left)`) dynamically scaled the canvas from default dimensions up to 2000+ pixels, preventing any label truncation or title overlap.

---

## Stress Test Results

| Test ID | Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|---|
| ST-01 | Recursive Gallery Discovery: nested subfolders with allowed vs. disallowed extensions | Index only `.png`, `.jpg`, `.jpeg`, `.csv`, `.tex`, `.json`, `.txt`; ignore `.bin`, `.wav`, `.py` | Exactly 6 valid files indexed across 4 base directories; disallowed files ignored | PASS |
| ST-02 | Gallery File Loading: `.png`, `.csv`, `.tex`, `.json`, `.txt` | Correct tab switching and widget population | Subtab index switches to 0 for images and 1 for text/tables; data loaded correctly | PASS |
| ST-03 | ZoomableImageWidget Controls | Zoom in/out, fit to view, 1:1 reset, and text placeholder | All zoom transformations adjust scale factors accurately; fit-to-view calculates viewport ratios | PASS |
| ST-04 | 3D PCA Floor and Wall Projections (`plot_scatter`) | Render XY floor ($z_{\min}$), XZ back wall ($y_{\max}$), YZ side wall ($x_{\min}$) shadow projections | All 3 projection planes rendered with subtle alpha and saved to 300 DPI image | PASS |
| ST-05 | 3D PCA Multi-Angle Views (`plot_scatter_3d_multi_angle`) | Render Frontal, Lateral (+90°), and Posterior (+180°) views with projections | High-resolution multi-view image generated (>2000px width) with correct orientation | PASS |
| ST-06 | 3D Error Analysis (`plot_analisis_errores_3d`) | Render centroid diamonds, vertical drop lines, floor projected centroid, and 2D floor ellipse | Drop lines (`:`), centroid markers, and floor ellipses rendered for K-Means and GMM | PASS |
| ST-07 | 3D PCA Degenerate Topologies | Graceful handling of collinear points, planar data, and 2-cluster distributions | Plots generated without unhandled exceptions; output PNGs created | PASS |
| ST-08 | Metric Table Export with Long Labels (`guardar_tabla_imagen`) | Dynamic canvas expansion for 60+ char headers, row labels, and 100+ char titles | Canvas auto-expanded (>1500px width); zero label clipping; clean alternating rows | PASS |
| ST-09 | Confusion Matrix Heatmap with Long Labels (`plot_confusion_matrix_heatmap`) | Render 5x5 matrix with extended class names and top title positioning | Figure saved with high DPI, top axis labels, and visible title | PASS |
| ST-10 | LaTeX Matrix Export (`guardar_matriz_latex`) | Valid LaTeX tabular syntax with RGB cell colors and percentage formatting | Valid LaTeX output generated with `\begin{tabular}`, `\cellcolor`, and header rows | PASS |
| ST-11 | Corrupted/Malformed File Injection in Gallery | Graceful handling of 0-byte images, bad CSVs, bad JSON, non-UTF-8 TXT | No crashes; diagnostic text rendered for corrupted CSV/JSON/TXT; null pixmap handled | PASS |
| ST-12 | Planar and Extreme Coordinate 3D Scatter | Render 3D scatter with $z=0$ constant and coordinates up to $\pm 10^6$ | Projections and main scatter rendered without math overflow or plotting errors | PASS |
| ST-13 | Table Rendering with Math & Special Characters | Handle `$`, `%`, `_`, `&`, `#`, `/` in headers and row labels | Matplotlib rendered table without LaTeX parse errors or bounding box overflow | PASS |
| ST-14 | Zero and Perfect Confusion Matrices | Heatmap rendering for all-zeros matrix and 100% diagonal matrix | Annotations ('0%', '100%') and color mappings rendered accurately | PASS |

---

## Unchallenged Areas

- **Native Windows Process Spawning**: `creationflags=subprocess.CREATE_NEW_CONSOLE` was verified via static code inspection and conditional platform guarding (`if sys.platform == "win32"`), but live process execution was tested exclusively under Linux x86_64 environment.
- **Physical NI-DAQ Hardware Interface**: DAQ hardware acquisition was mocked/out of scope as Milestone M1 focuses on UI architecture and offline visual analytics.

---

## Empirical Verdict

**VERDICT: PASS**

All visual rendering components, 3D PCA projection mathematics, results gallery discovery mechanisms, and export formatters meet requirements R1 and R3 with high empirical stability.
