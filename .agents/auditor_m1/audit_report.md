# Forensic Audit Report - Milestone M1 (UI Architecture & Visual Polishing)

**Work Product**: EMG_desarrollo (Milestone M1 changes: ui_analysis.py, main_app.py, generador_pca_umap.py, generador_umap_supervisado.py, analisis_por_track_integrado.py)
**Profile**: General Project
**Verdict**: INTEGRITY VIOLATION

---

## 1. Executive Summary

A comprehensive forensic audit was conducted on Milestone M1 implementations covering Requirements R1 (UI Architecture & Dynamic Parameter Extraction) and R3 (Visual Polishing, 3D PCA/UMAP Projections, and Metrics Gallery).

The mathematical decomposition pipelines (PCA, UMAP, Supervised UMAP, Autoencoders, DSP filters) and UI parameter binding logic were verified empirically and found to be genuine, complete, and functional.

However, an explicit constraint violation was detected:
- The presence of Unicode emoji character `\u270d` ("Writing Hand" `✍`) in modified file `EMG_desarrollo/gui_app/views/ui_analysis.py` at line 321.
- Untracked motor modules (`pca_motor.py` lines 362, 370 and `umap_motor.py` line 267) also contain Unicode emoji character `\u274c` ("Cross Mark" `❌`).

Per the Forensic Audit Mandate, because a mandatory check failed, the binary verdict is **INTEGRITY VIOLATION**.

---

## 2. Forensic Phase Results

### Phase 1: Static Analysis & Facade Detection
- **Hardcoded Test Results**: PASS — Zero instances of hardcoded metrics, canned accuracies, or mock clustering results found.
- **Facade Implementations**: PASS — All classes and methods (`AnalysisPanel.get_processing_kwargs()`, `AnalysisPanel.get_trevisan_kwargs()`, `MachineLearningPanel.get_pca_kwargs()`, `MachineLearningPanel.get_umap_kwargs()`, `MachineLearningPanel.get_umap_supervisado_kwargs()`, `ZoomableImageWidget`, `_refrescar_visor_imagenes()`) contain genuine business logic.
- **Pre-populated Artifacts**: PASS — No synthetic or fabricated test verification logs exist.

### Phase 2: Mathematics Audit
- **PCA Decomposition**: PASS — Uses `sklearn.decomposition.PCA(n_components=2/3).fit_transform(X)` directly on empirical feature tensors. Variance ratios and loadings are computed strictly from real covariance matrices.
- **UMAP & Supervised UMAP**: PASS — Uses `umap.UMAP` with genuine hyperparameters (`n_neighbors`, `min_dist`, `metric`, `target_weight`). Supervised projection performs true `fit_transform(X_train, y=Y_train_encoded)` and blind `transform(X_test)`.
- **Autoencoders**: PASS — Convolutional 1D neural network architectures in PyTorch remain authentic with real gradient optimization.
- **DSP Filters**: PASS — IIR notch filters (50 Hz), Butterworth bandpass/lowpass filters, and RMS / Moving Average envelopes in `analisis_trevisan.py` and `analisis_por_track_integrado.py` remain untouched.
- **3D Spatial Projections**: PASS — Bounding plane projections (`zs=z_min, zdir='z'`, `zs=y_max, zdir='y'`, `zs=x_min, zdir='x'`), vertical centroid drop lines, and floor confidence ellipses are purely visual layer renderings in Matplotlib and do NOT mutate or distort feature coordinate data.

### Phase 3: Code Authenticity & Cross-Platform Execution
- **UI Parameter Recovery (R1)**: PASS — `ProcessingTab.get_processing_kwargs()` dynamically extracts all 15 parameters (`mostrar_recortes`, `mostrar_senal_cruda`, `tema_cyberpunk`, `mostrar_espectrograma`, `frecuenciamaxima`, `apply_notch_filter`, `notch_q_factor`, `mostrar_evolucion`, `evol_t_start`, `evol_t_end`, `excluded_windows_list`, `tipo_envolvente`, `smooth_ms`, `highpass_cutoff_hz`, `lowpass_cutoff_hz`).
- **Subprocess Safety (R1)**: PASS — `creationflags=subprocess.CREATE_NEW_CONSOLE` is properly guarded with `if sys.platform == "win32":`. Cross-platform terminal launcher detects available Linux terminals (`konsole`, `gnome-terminal`, `xfce4-terminal`, `xterm`).
- **Gallery & Zoom Inspection (R3)**: PASS — `_refrescar_visor_imagenes()` recursively scans `resultados`, `resultados_pca_umap`, `resultados_umap_supervisado`, and `analisis_comparativos` via `os.walk`, indexing images (`.png`, `.jpg`), tables (`.csv`), and LaTeX/text reports (`.tex`, `.json`, `.txt`). `ZoomableImageWidget` provides zoom in/out, fit to view, 1:1, and modal dialog.

### Phase 4: Constraint & Emoji Audit
- **Emoji Scan**: FAIL —
  - `EMG_desarrollo/gui_app/views/ui_analysis.py:321`: `self.method_tabs.addTab(tab_man, "✍ Umbral Manual por Canal")` contains `✍` (`\u270d`).
  - `EMG_desarrollo/analysis/pca_motor.py:362`: `logger("❌ Error: No hay suficientes pulsos válidos para hacer PCA.")` contains `❌` (`\u274c`).
  - `EMG_desarrollo/analysis/pca_motor.py:370`: `logger("❌ Error: No hay suficientes pulsos de 'Entrenamiento' para hacer PCA.")` contains `❌` (`\u274c`).
  - `EMG_desarrollo/analysis/umap_motor.py:267`: `logger("❌ Error: No hay suficientes pulsos de 'Entrenamiento' para UMAP.")` contains `❌` (`\u274c`).

---

## 3. Empirical Verification Evidence

### Test Execution Output
```
Processing kwargs count: 15
PCA kwargs: ['proc_pca_2d', 'proc_pca_3d', 'proc_umap_2d', 'proc_umap_3d', 'ocultar_leyenda', 'params_2d', 'params_3d', 'params_umap', 'umap_n_neighbors', 'umap_min_dist', 'umap_metric', 'algoritmo_clustering_pca', 'algoritmo_clustering_umap', 'aplicar_trevisan', 'ignorar_ventana_cero', 'pre_pct', 'post_pct', 'modo_alineacion', 'estilo_visual', 'canales_features']
UMAP kwargs: ['proc_pca_2d', 'proc_pca_3d', 'proc_umap_2d', 'proc_umap_3d', 'ocultar_leyenda', 'params_2d', 'params_3d', 'params_umap', 'umap_n_neighbors', 'umap_min_dist', 'umap_metric', 'algoritmo_clustering_pca', 'algoritmo_clustering_umap', 'aplicar_trevisan', 'ignorar_ventana_cero', 'pre_pct', 'post_pct', 'modo_alineacion', 'estilo_visual', 'canales_features']
Supervised UMAP kwargs: ['alpha_ruido', 'smooth_ms', 'target_length', 'snr_threshold', 'outlier_contamination', 'notch_q', 'umap_n_neighbors', 'umap_min_dist', 'umap_metric', 'target_weight', 'eliminar_outliers_train', 'aplicar_trevisan', 'ignorar_ventana_cero', 'pre_pct', 'post_pct', 'modo_alineacion', 'canales_features']
Found gallery items in combo: 40
TEST EXECUTION COMPLETE: ALL ASSERTIONS PASSED.
```

### Emoji Scan Evidence
```
DIFF EMOJI in diff --git a/EMG_desarrollo/gui_app/views/ui_analysis.py b/EMG_desarrollo/gui_app/views/ui_analysis.py:
Line 321: ✍ (0x270d) -> + self.method_tabs.addTab(tab_man, "✍ Umbral Manual por Canal")
```

---

## 4. Remediation Required

To achieve a CLEAN verdict:
1. In `EMG_desarrollo/gui_app/views/ui_analysis.py` line 321:
   Replace `"✍ Umbral Manual por Canal"` with `"Umbral Manual por Canal"`.
2. In `EMG_desarrollo/analysis/pca_motor.py` lines 362 and 370:
   Replace `"❌ Error:"` with `"Error:"` or `"[ERROR]"`.
3. In `EMG_desarrollo/analysis/umap_motor.py` line 267:
   Replace `"❌ Error:"` with `"Error:"` or `"[ERROR]"`.
