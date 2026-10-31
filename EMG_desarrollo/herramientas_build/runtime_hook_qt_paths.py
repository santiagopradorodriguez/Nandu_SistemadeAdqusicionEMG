# ============================================================================
# NANDU LSD v6.2 - Runtime hook para PyInstaller
# Asegura que las DLL de PySide6, shiboken6 y pyzmq/libsodium se encuentren
# tanto en Windows nativo como en VM Windows para el puente DAQ.
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
        os.path.join(_root, "zmq"),
        os.path.join(_root, "pyzmq.libs"),
    ):
        if os.path.isdir(_folder):
            try:
                os.add_dll_directory(_folder)
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = _folder + os.pathsep + os.environ.get("PATH", "")

    _system32 = os.path.join(os.environ.get("SystemRoot", r"C:\\Windows"), "System32")
    for _runtime in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "concrt140.dll"):
        try:
            ctypes.WinDLL(os.path.join(_system32, _runtime))
        except OSError:
            pass

    try:
        import shiboken6  # noqa: F401
        from shiboken6 import Shiboken  # noqa: F401
    except Exception:
        pass
