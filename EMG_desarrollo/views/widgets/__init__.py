# ==============================================================================
# Archivo: EMG_desarrollo/views/widgets/__init__.py
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# ==============================================================================

from EMG_desarrollo.views.widgets.session_teleprompter_hud import SessionTeleprompterHUD
from EMG_desarrollo.views.widgets.reaper_oscilloscope import (
    ReaperOscilloscopeWidget,
    TrackControlHeader,
)

__all__ = [
    "SessionTeleprompterHUD",
    "ReaperOscilloscopeWidget",
    "TrackControlHeader",
]
