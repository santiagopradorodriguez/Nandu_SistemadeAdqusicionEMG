# ============================================================================
# NANDU LSD - Runtime hook para PyInstaller
# Asegura que las DLL de PySide6 y shiboken6 se encuentren en Windows.
# ============================================================================

import os
import sys
import ctypes


if sys.platform.startswith("win") and getattr(sys, "frozen", False):
    _root = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    for _folder in (
        _root,
        os.path.join(_root, "PySide6"),
        os.path.join(_root, "shiboken6"),
    ):
        if os.path.isdir(_folder):
            try:
                os.add_dll_directory(_folder)
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = _folder + os.pathsep + os.environ.get("PATH", "")

    # Some scientific packages bundle older MSVC runtimes.  If those are
    # found first, shiboken6 can fail with WinError 127 even though all Qt
    # DLLs are present.  Preload the Windows runtimes that match Python/Qt.
    _system32 = os.path.join(os.environ.get("SystemRoot", r"C:\\Windows"), "System32")
    for _runtime in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"):
        try:
            ctypes.WinDLL(os.path.join(_system32, _runtime))
        except OSError:
            pass

    # PySide6 on Windows expects Shiboken to be loaded before the Qt
    # extension modules.  Do this explicitly in the frozen application to
    # avoid QtWidgets.pyd failing with WinError 127.
    try:
        import shiboken6  # noqa: F401
        from shiboken6 import Shiboken  # noqa: F401
    except Exception:
        pass
