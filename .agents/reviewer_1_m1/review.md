# Quality and Adversarial Review Report - Milestone M1

## Review Summary

**Verdict**: APPROVE (PASS)

Milestone M1 changes (Requirements R1 and R3) have been independently inspected, tested, and stress-tested. All GUI parameter extractions, cross-platform subprocess executions, gallery navigation, and 3D visual projection enhancements meet quality, robustness, and mathematical integrity criteria. No emojis were used in the review artifacts.

---

## 1. Findings and Assessment

### Task 1: UI Parameter Collection (`EMG_desarrollo/gui_app/views/ui_analysis.py`)
- **Observation**:
  - `ProcessingTab.get_processing_kwargs()` cleanly collects all 15 GUI parameters (filtering cutoffs, notch toggles and Q factors, Praat spectrogram cutoffs, temporal evolution ranges, window exclusions, smoothing ms, and envelope types).
  - `AnalysisPanel.get_processing_kwargs()` delegates directly to `ProcessingTab.get_processing_kwargs()`.
  - `AnalysisPanel.get_trevisan_kwargs()` replaces unsafe `.value()` calls with robust `float()` conversion wrapped in `try...except (ValueError, TypeError)` with default fallback `50.0`.
  - Excluded window parsing (`inp_excluded`) handles comma-separated values, whitespace, and non-numeric characters gracefully via `part.isdigit()`.
- **Assessment**: Correct and robust.

### Task 2: Subprocess Portability & Results Gallery (`EMG_desarrollo/gui_app/main_app.py`)
- **Observation**:
  - `creationflags=subprocess.CREATE_NEW_CONSOLE` is properly guarded behind `if sys.platform == "win32":` across all execution paths (`_launch_external`, `ComparativeRunner.run()`, `SessionRunner.run()`, `ProcessRunner.run()`). On Linux/macOS, standard `subprocess.Popen` / `subprocess.run` is invoked without Windows-only flags, resolving `AttributeError`.
  - `_refrescar_visor_imagenes()` recursively scans target directories (`resultados`, `resultados_pca_umap`, `resultados_umap_supervisado`, `analisis_comparativos`) using `os.walk` for extensions `.png`, `.jpg`, `.jpeg`, `.csv`, `.tex`, `.json`, and `.txt`, ordered by modification timestamp descending.
  - `ZoomableImageWidget` implements smooth zoom in (+), zoom out (-), reset (1:1), fit to view, aspect ratio preservation, bounded scaling (0.1x to 8.0x), and modal fullscreen display.
  - `visor_subtabs` provides a dual-tab viewer: Tab 0 ("Gráfico / Imagen") displays visual plots; Tab 1 ("Métricas y Tablas") dynamically switches between `QTableWidget` for CSV data and monospace `QTextEdit` for LaTeX, JSON, and text metrics.
- **Assessment**: Fully compliant with cross-platform requirements and responsive UI design.

### Task 3: 3D PCA 2D Projections & Floor Shadows (`generador_pca_umap.py` & `pca_analysis.py`)
- **Observation**:
  - `plot_scatter` (3D mode) and `plot_scatter_3d_multi_angle` project 2D point shadows onto the bounding floor (XY at `z_min`), back wall (XZ at `y_max`), and side wall (YZ at `x_min`) with subtle alpha transparency (0.15 - 0.20).
  - `plot_analisis_errores_3d` renders vertical dotted drop lines (`:`) from cluster centroids to `z_min`, diamond markers (`D`) on the floor, and 2D covariance confidence ellipses projected onto the floor plane XY at `z_min`.
  - Dynamic figure dimensions and `subplots_adjust` in `guardar_tabla_imagen` and `plot_confusion_matrix_heatmap` eliminate title and header truncation on high-DPI outputs.
  - All mathematical decomposition algorithms (`PCA`, `UMAP`, `KMeans`, `GaussianMixture`, `IsolationForest`) remain 100% genuine with no hardcoded values or shortcuts.
- **Assessment**: High visual quality and complete preservation of mathematical integrity.

---

## 2. Verified Claims

- `AnalysisPanel.get_processing_kwargs()` returns all 15 expected keys with proper defaults -> Verified via automated unit tests in `test_m1` -> PASS
- `AnalysisPanel.get_trevisan_kwargs()` handles invalid/empty strings safely -> Verified via automated unit tests in `test_m1` -> PASS
- `main_app.py` runs without `CREATE_NEW_CONSOLE` crashes on Linux -> Verified via automated unit tests in `test_m1` -> PASS
- `ZoomableImageWidget` enforces zoom limits and fit-to-view -> Verified via automated unit tests in `test_m1` -> PASS
- Gallery handles images, CSVs, TeX, and JSON files correctly -> Verified via automated unit tests in `test_m1` -> PASS
- 3D PCA projections and floor drop lines render valid image files -> Verified via automated Matplotlib rendering tests in `test_m1` -> PASS

---

## 3. Adversarial Stress-Testing and Edge Cases

| Scenario | Expected Behavior | Actual Behavior | Result |
|---|---|---|---|
| Malformed excluded window input (`" 1, 2, invalid, -3, 10 "`) | Ignore non-digits, return `[1, 2, 10]` | Correctly filtered to `[1, 2, 10]` | PASS |
| Non-numeric input in `inp_smooth` | Return default float `50.0` without throwing exception | Caught by `try...except`, returns `50.0` | PASS |
| Empty results directory | Display placeholder, no unhandled exception | Handled cleanly by `os.path.exists` and empty combo | PASS |
| Singular covariance matrix in 3D floor ellipse projection | Skip ellipse rendering or clamp eigenvalues safely without crash | Handled cleanly in `try...except` and `np.maximum(evals_2d, 1e-9)` | PASS |
| Zooming beyond 8.0x or below 0.1x | Clamp scaling factor within bounds `[0.1, 8.0]` | Clamped correctly | PASS |

---

## 4. Integrity and Compliance Checklist

- [x] NO EMOJIS in review files or outputs
- [x] Mathematical logic of PCA, UMAP, and Autoencoders NOT modified
- [x] No hardcoded test results or mock shortcuts
- [x] Full test suite executed with project virtual environment
- [x] Official NANDU LSD license headers present
