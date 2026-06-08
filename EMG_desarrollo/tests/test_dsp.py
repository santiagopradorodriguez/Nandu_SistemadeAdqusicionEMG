# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo test_dsp.py del sistema NANDU LSD.
# ==============================================================================

# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Módulo test_dsp.py del sistema NANDU LSD.
# ==============================================================================

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analysis.analisis_por_track_integrado import rms, _resample_to

def test_rms():
    """Prueba el cálculo de RMS de una señal."""
    # Señal constante
    signal = np.full(100, 3.0)
    rms_val = rms(signal)
    assert np.isclose(rms_val, 3.0), "El RMS de una constante debe ser la constante."
    
    # Señal senoidal con amplitud A, el RMS es A / sqrt(2)
    A = 5.0
    t = np.linspace(0, 2*np.pi, 1000)
    signal_sin = A * np.sin(t)
    rms_sin = rms(signal_sin)
    assert np.isclose(rms_sin, A / np.sqrt(2), atol=0.1), "El RMS de una senoidal es A/sqrt(2)."

def test_resample_to():
    """Prueba el remuestreo de señales usando scipy.signal.resample."""
    original_signal = np.sin(np.linspace(0, 10, 500))
    resampled = _resample_to(original_signal, 250)
    
    assert len(resampled) == 250, "El tamaño resampleado no es el correcto."
    # El valor máximo debería conservarse razonablemente bien
    assert np.isclose(np.max(original_signal), np.max(resampled), atol=0.05)
