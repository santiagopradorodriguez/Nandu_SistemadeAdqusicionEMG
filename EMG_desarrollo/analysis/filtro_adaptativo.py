# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Cancelador adaptativo de interferencia de línea de 50 Hz (LMS/NLMS).
# ==============================================================================

# -*- coding: utf-8 -*-

"""
Módulo de Procesamiento Adaptativo de Señales Bioeléctricas (ANC - Adaptive Noise Cancellation)

Fundamentación Matemática:
En la adquisición de señales de electromiografía de superficie (sEMG), la interferencia
proveniente de la red de suministro eléctrico a 50 Hz (y sus armónicos pares e impares)
constituye una de las fuentes de contaminación más persistentes.

A diferencia de un filtro Notch de muesca IIR convencional, que atenúa de manera fija e
irreversible una banda espectral alrededor de 50 Hz e introduce distorsiones transitorias
(ringing) y retardo de fase, el cancelador adaptativo de Widrow sintetiza internamente
vectores de referencia ortogonales en cuadratura y ajusta sus pesos en tiempo real para
restar exclusivamente la componente coherente de la interferencia:

1. Señal primaria observada:
   d[n] = s[n] + n_0[n]
   donde:
     - d[n]: Señal biopotencial medida en el canal muscular (µV).
     - s[n]: Actividad mioeléctrica fisiológica genuina (µV).
     - n_0[n]: Interferencia sinusoidal de red con amplitud A y fase phi desconocidas.
     - n: Índice discreto de tiempo (muestra n).
     - fs: Frecuencia de muestreo del sistema (Hz).
     - f_0: Frecuencia nominal de la red (50.0 Hz).

2. Generación sintética de referencia en cuadratura:
   x_1[n] = cos(2 * pi * f_0 * n / fs)
   x_2[n] = sin(2 * pi * f_0 * n / fs)
   Para incluir el primer armónico (100 Hz):
   x_3[n] = cos(4 * pi * f_0 * n / fs)
   x_4[n] = sin(4 * pi * f_0 * n / fs)

3. Estimación adaptativa del ruido de línea:
   y_est[n] = sum_{k} w_k[n] * x_k[n]

4. Señal mioeléctrica recuperada (error de predicción):
   e[n] = d[n] - y_est[n]

5. Regla de adaptación de pesos:
   - Modo LMS Estándar:
     w_k[n+1] = w_k[n] + 2 * mu * e[n] * x_k[n]
   - Modo NLMS (Normalizado):
     w_k[n+1] = w_k[n] + (mu_norm / (||x[n]||^2 + epsilon)) * e[n] * x_k[n]
   donde mu es la tasa de aprendizaje o paso de adaptación, y epsilon = 1e-7.
"""

import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# ==============================================================================
# Núcleo de Procesamiento Acelerado con Numba (JIT)
# ==============================================================================
if HAS_NUMBA:
    @njit(fastmath=True)
    def _lms_cuadratura_core_numba(d, x_ref, mu, normalizado):
        N, K = x_ref.shape
        e = np.empty(N, dtype=np.float64)
        y_est = np.empty(N, dtype=np.float64)
        w = np.zeros(K, dtype=np.float64)
        eps = 1e-7

        for n in range(N):
            # 1. Estimación del ruido actual como combinación lineal
            y_val = 0.0
            for k in range(K):
                y_val += w[k] * x_ref[n, k]
            y_est[n] = y_val

            # 2. Señal limpia resultante (error de predicción)
            err = d[n] - y_val
            e[n] = err

            # 3. Factor de escala del paso de adaptación
            if normalizado:
                norm_sq = 0.0
                for k in range(K):
                    norm_sq += x_ref[n, k] * x_ref[n, k]
                paso = mu / (norm_sq + eps)
            else:
                paso = 2.0 * mu

            # 4. Actualización estocástica de pesos
            for k in range(K):
                w[k] += paso * err * x_ref[n, k]

        return e, y_est, w


# ==============================================================================
# Núcleo de Procesamiento con Respaldo en NumPy Estándar
# ==============================================================================
def _lms_cuadratura_core_numpy(d, x_ref, mu, normalizado):
    N, K = x_ref.shape
    e = np.empty(N, dtype=np.float64)
    y_est = np.empty(N, dtype=np.float64)
    w = np.zeros(K, dtype=np.float64)
    eps = 1e-7

    for n in range(N):
        # 1. Estimación del ruido actual
        y_val = np.dot(w, x_ref[n])
        y_est[n] = y_val

        # 2. Señal limpia resultante
        err = d[n] - y_val
        e[n] = err

        # 3. Factor de adaptación
        if normalizado:
            norm_sq = np.dot(x_ref[n], x_ref[n])
            paso = mu / (norm_sq + eps)
        else:
            paso = 2.0 * mu

        # 4. Actualización de pesos
        w += paso * err * x_ref[n]

    return e, y_est, w


# ==============================================================================
# Interfaz Pública del Filtro Adaptativo
# ==============================================================================
def cancelar_ruido_linea_adaptativo(
    signal_array,
    fs,
    f0=50.0,
    mu=0.005,
    incluir_armonico=False,
    normalizado=False,
    armonicos=None
):
    """
    Aplica cancelación adaptativa de ruido (ANC) de 50 Hz a una señal temporal.

    Parámetros:
      signal_array: ndarray unidimensional con la señal en microvoltios (µV).
      fs: Frecuencia de muestreo en Hz (ej. 2000.0).
      f0: Frecuencia fundamental de red a cancelar en Hz (predeterminado: 50.0 Hz).
      mu: Paso de adaptación o tasa de aprendizaje (predeterminado: 0.005).
      incluir_armonico: Booleano para incluir la componente de 100 Hz (primer armónico).
      normalizado: Booleano para activar el modo NLMS (LMS normalizado).
      armonicos: Tupla o lista opcional de frecuencias armónicas a cancelar (ej. (50, 100, 150, 200, ...)).

    Retorna:
      e: ndarray unidimensional con la señal filtrada y acondicionada (µV).
      y_est: ndarray unidimensional con la estimación en tiempo real del ruido de 50 Hz (µV).
      w: ndarray con los coeficientes finales aprendidos por el filtro.
    """
    if signal_array is None or len(signal_array) == 0:
        return np.array([]), np.array([]), np.zeros(2)

    d = np.asarray(signal_array, dtype=np.float64)
    N = len(d)

    if fs is None or fs <= 0.0 or N < 2:
        return d.copy(), np.zeros_like(d), np.zeros(2)

    # Construcción del vector de tiempo discreto
    t = np.arange(N, dtype=np.float64) / float(fs)

    # Construcción de referencias sintéticas ortogonales en cuadratura
    if armonicos is not None:
        cols = []
        for fh in armonicos:
            if fh < (fs * 0.5):
                omega_h = 2.0 * np.pi * fh
                cols.append(np.cos(omega_h * t))
                cols.append(np.sin(omega_h * t))
        x_ref = np.column_stack(cols) if len(cols) > 0 else np.zeros((N, 2))
    elif incluir_armonico:
        omega_0 = 2.0 * np.pi * f0
        ref_cos_1 = np.cos(omega_0 * t)
        ref_sin_1 = np.sin(omega_0 * t)
        ref_cos_2 = np.cos(2.0 * omega_0 * t)
        ref_sin_2 = np.sin(2.0 * omega_0 * t)
        x_ref = np.column_stack((ref_cos_1, ref_sin_1, ref_cos_2, ref_sin_2))
    else:
        omega_0 = 2.0 * np.pi * f0
        ref_cos_1 = np.cos(omega_0 * t)
        ref_sin_1 = np.sin(omega_0 * t)
        x_ref = np.column_stack((ref_cos_1, ref_sin_1))

    # Selección del motor de ejecución (Numba optimizado o NumPy)
    if HAS_NUMBA:
        e, y_est, w = _lms_cuadratura_core_numba(d, x_ref, float(mu), bool(normalizado))
    else:
        e, y_est, w = _lms_cuadratura_core_numpy(d, x_ref, float(mu), bool(normalizado))

    return e, y_est, w


def inyectar_ruido_linea(
    signal_array,
    fs,
    f0=50.0,
    amplitud_uv=50.0,
    fase_rad=0.0,
    incluir_armonico=False,
    amplitud_armonico_uv=15.0
):
    """
    Inyecta sintéticamente una interferencia de red de 50 Hz para pruebas de laboratorio.

    Parámetros:
      signal_array: ndarray con la señal biopotencial original (µV).
      fs: Frecuencia de muestreo en Hz.
      f0: Frecuencia de la interferencia en Hz (predeterminado: 50.0 Hz).
      amplitud_uv: Amplitud pico de la onda senoidal de 50 Hz en microvoltios (µV).
      fase_rad: Desfase angular inicial en radianes.
      incluir_armonico: Si es True, añade además una componente armónica a 100 Hz.
      amplitud_armonico_uv: Amplitud pico del segundo armónico (µV).

    Retorna:
      signal_contaminada: ndarray con la señal resultante tras sumar el ruido de prueba.
      ruido_puro: ndarray con la perturbación aislada inyectada.
    """
    if signal_array is None or len(signal_array) == 0:
        return np.array([]), np.array([])

    d = np.asarray(signal_array, dtype=np.float64)
    N = len(d)

    if fs is None or fs <= 0.0 or N < 2:
        return d.copy(), np.zeros_like(d)

    t = np.arange(N, dtype=np.float64) / float(fs)
    omega_0 = 2.0 * np.pi * f0

    ruido_puro = amplitud_uv * np.sin(omega_0 * t + fase_rad)

    if incluir_armonico:
        ruido_puro += amplitud_armonico_uv * np.sin(2.0 * omega_0 * t + 2.0 * fase_rad)

    signal_contaminada = d + ruido_puro
    return signal_contaminada, ruido_puro
