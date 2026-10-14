# ==============================================================================
# Paquete de Scripts de Experimentación - Libro de Experimentos IA (Ñandú sEMG)
# ==============================================================================
import os
import sys

# Inyectar rutas del proyecto para ejecución autónoma
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_PKG_DIR, "../../../"))
_DL_CORE = os.path.join(_REPO_ROOT, "EMG_desarrollo", "deep_learning")

for _p in [_PKG_DIR, _REPO_ROOT, _DL_CORE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
