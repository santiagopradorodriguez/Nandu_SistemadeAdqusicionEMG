# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Manejo de hilos para procesos en segundo plano en la GUI.
# ==============================================================================

import sys
import traceback
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

class StreamSignal(QObject):
    """Objeto necesario para emitir señales desde un stream de texto normal"""
    new_text = Signal(str)

class EmittingStream(object):
    """Redirige el flujo de texto (como sys.stdout) a una señal Qt"""
    def __init__(self):
        super().__init__()
        self.signal_obj = StreamSignal()
        
    def write(self, text):
        if text.strip() or text == '\\n': # Enviar también los saltos de línea intencionales
            self.signal_obj.new_text.emit(str(text))
            
    def flush(self):
        pass # Requerido para compatibilidad con sys.stdout

class WorkerSignals(QObject):
    """
    Define las señales disponibles desde el Worker (hilo en background).
    """
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)

class Worker(QRunnable):
    """
    Clase genérica de Worker para ejecutar funciones pesadas en background
    sin bloquear la UI.
    """
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        
        # Añadimos callback de progreso si la función lo acepta/soporta
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        """Ejecuta la función interceptando errores"""
        try:
            # Eliminamos el progress_callback si la función objetivo no lo acepta
            import inspect
            sig = inspect.signature(self.fn)
            if 'progress_callback' not in sig.parameters:
                self.kwargs.pop('progress_callback', None)
                
            result = self.fn(*self.args, **self.kwargs)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
