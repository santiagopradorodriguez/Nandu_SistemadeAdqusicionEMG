# BRIEFING — 2026-08-17T17:17:00Z

## Mission
Investigate UI Architecture and Visual Polishing for Requirements R1 and R3 in EMG_desarrollo (PySide6 UI initialization, 3D PCA visualization with 2D projection planes, and Galeria de Resultados metric display/formatting).

## 🔒 My Identity
- Archetype: explorer
- Roles: UI Architecture and Visual Polishing Specialist
- Working directory: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/explorer_1
- Original parent: 3a24731f-f6e3-4e37-8895-e772f89af223
- Milestone: Investigation and Analysis of UI Initialization, 3D PCA projections, and ML Results Gallery

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- NO EMOJIS in any output or file
- DO NOT alter or recommend altering the mathematical logic or internal steps of PCA, Supervised UMAP, or Autoencoders
- Database recordings reside in subfolders canal_0, canal_1, canal_2, canal_3 under each session folder; metadata.json in canal_0

## Current Parent
- Conversation ID: 3a24731f-f6e3-4e37-8895-e772f89af223
- Updated: 2026-08-17T17:17:00Z

## Investigation State
- **Explored paths**: EMG_desarrollo/gui_app/main_app.py, EMG_desarrollo/gui_app/views/ui_analysis.py, EMG_desarrollo/gui_app/views/config_dialog.py, EMG_desarrollo/gui_app/views/session_explorer.py, EMG_desarrollo/gui_app/views/comparative_explorer_widget.py, EMG_desarrollo/gui_app/views/calibrated_viewer_widget.py, EMG_desarrollo/views/config_dialog.py, EMG_desarrollo/deep_learning/pca_umap_clustering/generador_pca_umap.py, EMG_desarrollo/deep_learning/pca_analysis.py, EMG_desarrollo/deep_learning/generador_umap_supervisado.py, EMG_desarrollo/analysis/pca_motor.py, EMG_desarrollo/analysis/plot_metricas_tesis.py.
- **Key findings**:
  1. `AnalysisPanel.get_processing_kwargs()` is missing in `ui_analysis.py`, crashing processing calls.
  2. `AnalysisPanel.get_trevisan_kwargs()` contains invalid widget attributes.
  3. `subprocess.CREATE_NEW_CONSOLE` in `main_app.py` lines 956, 1150, 1572 crashes on Linux.
  4. 3D PCA plots (`Axes3D`) can render 2D shadow projections on XY, XZ, YZ planes, centroid drop lines, and 2D floor covariance ellipses purely in Matplotlib rendering without touching PCA math.
  5. `_refrescar_visor_imagenes` uses shallow non-recursive glob and misses named set subdirectories.
  6. Table and heatmap rendering has title/label collision with `tight_layout()`.
- **Unexplored areas**: None within the scope of Requirements R1 and R3.

## Key Decisions Made
- Fully documented all root causes, line numbers, and concrete fix strategies in `analysis.md` and `handoff.md`.

## Artifact Index
- ORIGINAL_REQUEST.md — Original user request
- BRIEFING.md — Working memory
- progress.md — Liveness heartbeat
- analysis.md — Detailed investigation findings
- handoff.md — Self-contained 5-component handoff report
