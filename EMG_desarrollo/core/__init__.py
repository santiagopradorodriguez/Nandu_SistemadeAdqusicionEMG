# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo central de procesamiento y arquitectura (core).
# ==============================================================================

from .dsp_engine import (
    calculate_rms_envelope,
    calculate_rms_envelope_2d,
    decimate_min_max,
    calcular_ruido_linea_50hz,
)
from .preroll_and_filters import (
    PreRollRecordingBuffer,
    ConfigurableFilterChain,
)
from .daq_engine import (
    AcquisitionSource,
    RingBuffer,
    UnifiedAcquisitionWorker,
)
from .protocol_fsm import (
    ProtocolState,
    ProtocolMode,
    ProtocolConfig,
    AcquisitionProtocolFSM,
)
from .process_runner import ModuleTaskRunner
from .latent_bci_utils import LatentSpaceBCIGuard, LatentKalmanTracker

__all__ = [
    "calculate_rms_envelope",
    "calculate_rms_envelope_2d",
    "decimate_min_max",
    "calcular_ruido_linea_50hz",
    "PreRollRecordingBuffer",
    "ConfigurableFilterChain",
    "AcquisitionSource",
    "RingBuffer",
    "UnifiedAcquisitionWorker",
    "ProtocolState",
    "ProtocolMode",
    "ProtocolConfig",
    "AcquisitionProtocolFSM",
    "ModuleTaskRunner",
    "LatentSpaceBCIGuard",
    "LatentKalmanTracker",
]
