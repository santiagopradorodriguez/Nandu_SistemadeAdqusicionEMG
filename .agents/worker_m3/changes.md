# Changes Summary — Worker M3 (Documentation Specialist)

**Date:** 2026-08-17
**Scope:** Requirement R4 - Software & Repository Documentation (Nandu EMG v6.0)

## 1. Files Modified

### 1. `README.md`
- **Location:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/README.md`
- **Changes:**
  - Upgraded project title, badges, and overview to Ñandú EMG v6.0 (Laboratorio de Sistemas Dinámicos - FCEyN UBA).
  - Documented the official multi-channel database hierarchy strictly adhering to `AGENTS.md`: `base_de_datos_electrodos/<Fecha>/<Sesión>/canal_0..3/` with `metadata.json` and primary WAV in `canal_0/`.
  - Added a detailed architectural breakdown of the 5 main tabs in `gui_app/main_app.py` (Inicio y Adquisición, Visualización, Análisis y Extracción, Machine Learning, Historial de Resultados) plus auxiliary tools (Mini-DAW, Configuración General, Extractor Deep Learning).
  - Embedded a comprehensive Mermaid flowchart diagramming the 5 phases of the scientific workflow (Adquisición, DSP, Tensorización, Machine Learning/Deep Learning, Exploración).
  - Documented DSP filtering (Notch 50 Hz, Butterworth pasabanda 20-500 Hz, RMS envelope, Master-Slave cross-correlation alignment), Machine Learning (PCA, UMAP, XGBoost, Trevisan binarization, continuous decoders), and Deep Learning (1D Convolutional Autoencoders in PyTorch).
  - Updated quickstart instructions (`python EMG_desarrollo/gui_app/main_app.py`) and compilation steps for Windows (`build.bat`) and Linux (`build_linux.sh`).
  - Enforced 100% zero-emoji policy across all sections.

### 2. `CONTRIBUTING.md`
- **Location:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/CONTRIBUTING.md`
- **Changes:**
  - Removed all emoji characters from section headings and text body.
  - Synchronized architectural references with v6.0 (modular organization across `acquisition/`, `analysis/`, `deep_learning/`, `gui_app/`, `utils/`).
  - Documented core repository rules: AGENTS.md database compliance, zero-emoji policy, Qt scoping rules (`self.`), and physiological synergy preservation via Master-Slave alignment.
  - Added clear developer contribution instructions, virtual environment setup, PyTorch tensor formatting guidelines, and pull request workflow.
  - Updated the active roadmap to reflect v6.0+ objectives (anatomical visualizer, real-time silent speech decoder, safe pause button).

### 3. `EMG_desarrollo/instrucciones_uso.py`
- **Location:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/instrucciones_uso.py`
- **Changes:**
  - Updated in-app HTML user guide to version 6.0 covering all 5 main tabs:
    * Pestaña 1: Inicio y Adquisición (Modo Manual y AutoForge con metrónomo audiovisual, calibración de ruido basal, secuencia continua con `valid_words`, cálculo SNR adaptativo y envolvente RMS en vivo).
    * Pestaña 2: Visualización (Explorador CSV PyQtGraph, Historial de Gráficos Musculares, Visor de Electrodos en Grilla, Historial de Patrón Muscular).
    * Pestaña 3: Análisis y Extracción (Procesamiento interactivo de pulsos, filtros de fase cero, envolvente RMS, alineación Master-Slave por cross-correlation, análisis estadístico de sesión).
    * Pestaña 4: Machine Learning y Deep Learning (PCA, UMAP no supervisado y supervisado, Autoencoders Convolucionales 1D en PyTorch, clasificador XGBoost, binarización de Trevisan, Galería de Resultados integrada con visor de gráficos con zoom y tablas de métricas).
    * Pestaña 5: Historial de Resultados (Historial de Comparativas e Historial de Sesión).
    * Herramientas Auxiliares: Mini-DAW Canal 3, Configuración General, Extractor de Tensores para PyTorch, Editor de Mediciones.
  - Enforced 100% zero-emoji policy across UI strings and HTML content.
  - Verified compilation with `py_compile`.

### 4. `DESCARGAS.md`
- **Location:** `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/DESCARGAS.md`
- **Changes:**
  - Removed decorative emojis to maintain project-wide zero-emoji standard.
  - Updated download notes and alternative source build instructions.

## 2. Verification
- Developed and ran automated verification script checking 20 target files across the repository for unicode emoji sequences (`0x1F000-0x1FFFF`, `0x2600-0x27BF`, `0x2300-0x23FF`, `0x2B50-0x2B55`, `0xFE0E-0xFE0F`).
- Confirmed zero emoji occurrences across all audited files.
- Confirmed all required architectural keywords and database structure patterns in `README.md`.
