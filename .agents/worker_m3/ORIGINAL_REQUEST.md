## 2026-08-17T21:13:36Z
You are Worker M3 (Software & Repository Documentation Specialist) for Requirement R4 of the Nandu EMG Acquisition System.

Working Directory: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/worker_m3
Reference Analysis: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/explorer_3/analysis.md
Reference Handoff: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/explorer_3/handoff.md
Project Rule AGENTS.md: /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/AGENTS.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A Forensic Auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Key Constraints:
1. NO EMOJIS: Never use emojis anywhere in any code, comments, strings, documentation, or output. Clean all existing emojis in documentation files.
2. The database structure MUST strictly follow AGENTS.md: `base_de_datos_electrodos/<Fecha>/<Sesión>/canal_0..3/` with `metadata.json` in `canal_0`.
3. Keep code and markdown cleanly formatted.

Tasks:
1. Initialize your workspace in /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/worker_m3 (create BRIEFING.md and maintain progress.md).
2. Load skill /home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/.agents/skills/repo-manager/SKILL.md.
3. Update `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/README.md`:
   - Update project description for Ñandú EMG v6.0.
   - Document the correct multi-channel database layout (`base_de_datos_electrodos/<Fecha>/<Sesión>/canal_0..3/`).
   - Document the complete workflow: AutoForge metronome recording, Master-Slave alignment, DSP filtering, Machine Learning (PCA/UMAP), Deep Learning (1D Autoencoders, XGBoost), and Results Gallery.
   - Update quickstart instructions for running and building.
   - Ensure ZERO emojis in `README.md`.
4. Update `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/CONTRIBUTING.md`:
   - Remove ALL emojis from headers and body text.
   - Update roadmap and pending tasks to reflect current v6.0 architecture.
   - Ensure clean markdown syntax and clear developer contribution steps.
5. Update `/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/instrucciones_uso.py` and any documentation under `EMG_desarrollo/archivos_md/`:
   - Update in-app user guide to version 6.0 covering all 6 tabs and novelties.
   - Ensure ZERO emojis in `instrucciones_uso.py` and `.md` files.
6. Verify with an automated script that ZERO emojis exist across `README.md`, `CONTRIBUTING.md`, `EMG_desarrollo/instrucciones_uso.py`, and `EMG_desarrollo/archivos_md/`.
7. Document changes in `changes.md` and generate `handoff.md`.
8. Send a message to orchestrator with your completion report.
