# ==============================================================================
# Archivo: EMG_desarrollo/core/process_runner.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Descripción: Ejecutor QProcess sin archivos temporales temp_*.py.
# ==============================================================================

from __future__ import annotations
import sys
import json
from typing import Dict, Any, Optional
from PySide6.QtCore import QObject, QProcess, Signal


class ModuleTaskRunner(QObject):
    log_received = Signal(str)
    task_finished = Signal(bool, dict)

    RESULT_PREFIX = "__NANDU_JSON_RESULT__:"

    def __init__(self, project_root: str, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.project_root = project_root
        self.process: Optional[QProcess] = None
        self._last_result: Dict[str, Any] = {}

    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QProcess.NotRunning

    def run_module_function(self, module_path: str, function_name: str, kwargs: Dict[str, Any]) -> None:
        if self.is_running():
            self.log_received.emit("> [Aviso] Ya hay una tarea ejecutándose en segundo plano.\n")
            return

        self._last_result = {}
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_ready_read)
        self.process.finished.connect(self._on_finished)

        bootstrap_code = (
            "import sys, json, importlib, traceback\n"
            f"sys.path.insert(0, {repr(self.project_root)})\n"
            "payload = json.loads(sys.stdin.read())\n"
            "try:\n"
            "    mod = importlib.import_module(payload['module'])\n"
            "    fn = getattr(mod, payload['function'])\n"
            "    res = fn(**payload['kwargs'])\n"
            f"    print('{self.RESULT_PREFIX}' + json.dumps({{'status': 'ok', 'data': str(res)}}), flush=True)\n"
            "except Exception:\n"
            "    traceback.print_exc()\n"
            "    sys.exit(1)\n"
        )

        self.process.start(sys.executable, ["-u", "-c", bootstrap_code])
        payload_bytes = json.dumps({
            "module": module_path,
            "function": function_name,
            "kwargs": kwargs,
        }).encode("utf-8")
        self.process.write(payload_bytes)
        self.process.closeWriteChannel()

    def terminate_task(self) -> None:
        if self.is_running() and self.process is not None:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()

    def _on_ready_read(self) -> None:
        if self.process is None:
            return
        raw_text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in raw_text.splitlines(keepends=True):
            if line.startswith(self.RESULT_PREFIX):
                try:
                    self._last_result = json.loads(line[len(self.RESULT_PREFIX):].strip())
                except Exception:
                    pass
            else:
                self.log_received.emit(line)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        success = (exit_code == 0 and exit_status == QProcess.NormalExit)
        self.task_finished.emit(success, self._last_result)
