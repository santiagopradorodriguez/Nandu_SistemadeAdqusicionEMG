# Handoff Report - Challenger 2 (Milestone M1)

## 1. Observation
- **Test Executions**:
  - Executed test suite `EMG_desarrollo/tests/test_gallery_and_3d_pca.py` (10 unit test cases) via `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/venv/bin/python`:
    - Result: `Ran 10 tests in 11.004s. OK`.
  - Executed adversarial stress test suite `EMG_desarrollo/tests/test_adversarial_stress_m1.py` (4 adversarial stress test cases) via `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/venv/bin/python`:
    - Result: `Ran 4 tests in 6.801s. OK`.
  - Executed full test suite discovery `python -m unittest discover -s EMG_desarrollo/tests -p "test_*.py"` (43 test cases total including DSP, Gallery, 3D PCA, and adversarial stress):
    - Result: `Ran 43 tests in 32.636s. OK`.
- **Target Areas Verified**:
  - **Results Gallery (R1)**:
    - Recursive directory discovery with `os.walk` across `resultados`, `deep_learning/pca_umap_clustering/resultados_pca_umap`, `deep_learning/resultados_umap_supervisado`, and `analisis_comparativos`.
    - File extension filtering ensuring only `.png`, `.jpg`, `.jpeg`, `.csv`, `.tex`, `.json`, and `.txt` are indexed while `.bin`, `.wav`, `.py` are omitted.
    - Sorting by modification timestamp (`os.path.getmtime`).
    - Robust file loading in `_cargar_imagen_visor()`: switching subtabs to Image Viewer (tab 0) for images and Metric/Text Viewer (tab 1) for structured tables (`.csv`) and text (`.tex`, `.json`, `.txt`).
    - `ZoomableImageWidget` functionality: zoom in, zoom out, fit to view, 1:1 reset, fullscreen modal, and placeholder handling.
  - **3D PCA & Visual Polishing (R3)**:
    - 3D PCA scatter rendering in `generador_pca_umap.py` and `pca_analysis.py` with multi-plane projections: floor projection on XY plane ($z = z_{\min}$), rear wall projection on XZ plane ($y = y_{\max}$), and lateral wall projection on YZ plane ($x = x_{\min}$).
    - Multi-angle 3D view generator `plot_scatter_3d_multi_angle()` rendering Frontal, Lateral (+90°), and Posterior (+180°) views with plane shadows.
    - 3D error analysis `plot_analisis_errores_3d()` with cluster centroid markers, vertical dotted drop lines from centroids to floor, projected floor centroids, and projected 2D floor confidence ellipses ($3\sigma$).
    - Dynamic sizing in `guardar_tabla_imagen()` preventing header and title clipping for long labels and titles.
    - `plot_confusion_matrix_heatmap()` and `guardar_matriz_latex()` formatting and rendering without edge clipping.

## 2. Logic Chain
1. Empirical test cases were designed to verify both functional and edge-case behavior of Requirements R1 and R3.
2. In R1, the gallery discovery logic was evaluated against deep directory hierarchies and non-standard file formats. The implementation successfully isolated valid artifacts, sorted them chronologically, and formatted combobox labels as `[category] rel_path`.
3. In R1 file loading, injecting corrupted and non-UTF-8 files confirmed that exception handlers prevent Qt UI crashes and present informative diagnostic output to the user.
4. In R3, 3D PCA rendering functions were tested with both synthetic clustered data and degenerate spatial topologies (planar, collinear, extreme coordinate values, single-point clusters). The floor and wall projections, vertical drop lines, and 2D floor ellipses rendered consistently into high-resolution 300 DPI image files.
5. In R3 table and matrix exports, testing with 60+ character labels verified that the dynamic canvas dimension calculations in `guardar_tabla_imagen` and `plot_confusion_matrix_heatmap` eliminate text truncation.

## 3. Caveats
- All UI and visualization tests were executed using Qt offscreen platform (`QT_QPA_PLATFORM=offscreen`) and Matplotlib non-interactive Agg backend (`matplotlib.use('Agg')`), which is appropriate for headless testing environments.
- Windows-specific console creation flags (`creationflags=subprocess.CREATE_NEW_CONSOLE`) are guarded conditionally in code (`if sys.platform == "win32"`), tested statically for platform isolation.

## 4. Conclusion
- Requirements R1 (Results Gallery) and R3 (3D PCA Floor/Wall Projections & Visual Polishing) are empirically validated.
- All 43 test cases passed with zero failures or errors.
- Empirical Verdict: **PASS**.

## 5. Verification Method
To independently reproduce and verify these findings, run:
```bash
/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/venv/bin/python -m unittest discover -s /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/tests -p "test_*.py"
```
- **Expected result**: All 43 tests pass (`OK`).
- **Invalidation condition**: Any assertion error, unhandled exception, or failing test in `test_gallery_and_3d_pca.py` or `test_adversarial_stress_m1.py`.
