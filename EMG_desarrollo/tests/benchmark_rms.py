import time
import numpy as np
import scipy.ndimage
import scipy.signal

# Parámetros del benchmark
SAMPLE_RATE = 2000
CHUNK_SIZE = 50       # Tamaño del chunk de hardware (ej: cada 25 ms)
WINDOW_SIZE = 4000    # Tamaño total del buffer a graficar (2 segundos)
NUM_CANALES = 4

# Simular datos
buffer_datos = np.random.randn(NUM_CANALES, WINDOW_SIZE)
nuevo_chunk = np.random.randn(NUM_CANALES, CHUNK_SIZE)

ITERACIONES = 1000

print(f"--- BENCHMARK DE ENVUELTES RMS (Frecuencia de refresco, {ITERACIONES} frames) ---")

# 1. MÉTODO VIEJO (dsp-auditor): np.convolve iterativo sobre ventana completa
start = time.perf_counter()
window = np.ones(int(SAMPLE_RATE * 0.1)) / int(SAMPLE_RATE * 0.1)
for _ in range(ITERACIONES):
    rms_completo = np.zeros_like(buffer_datos)
    for i in range(NUM_CANALES):
        sq = np.square(buffer_datos[i])
        conv = np.convolve(sq, window, mode='same')
        rms_completo[i] = np.sqrt(np.maximum(conv, 0))
t1 = time.perf_counter() - start
print(f"1. np.convolve (Ventana completa, iterativo): {t1:.4f} seg -> {(ITERACIONES/t1):.1f} FPS")

# 2. MÉTODO 2: scipy.ndimage.uniform_filter1d sobre ventana completa
start = time.perf_counter()
ws = int(SAMPLE_RATE * 0.1)
for _ in range(ITERACIONES):
    sq = np.square(buffer_datos)
    ms = scipy.ndimage.uniform_filter1d(sq, size=ws, axis=1)
    rms_vectorizado = np.sqrt(np.maximum(ms, 0))
t2 = time.perf_counter() - start
print(f"2. scipy uniform_filter1d (Ventana completa, vectorizado): {t2:.4f} seg -> {(ITERACIONES/t2):.1f} FPS")

# 3. MÉTODO 3: Envolvente IIR sobre CHUNK (Antigravity v5.1)
b, a = scipy.signal.butter(2, 5.0, btype='low', fs=SAMPLE_RATE)
sos = scipy.signal.tf2sos(b, a)
zi_base = scipy.signal.sosfilt_zi(sos)
zi = np.stack([zi_base] * NUM_CANALES, axis=-1)

start = time.perf_counter()
env_buffer = np.zeros_like(buffer_datos)
for _ in range(ITERACIONES):
    # Procesar SOLO el chunk
    rectified = np.abs(nuevo_chunk)
    env_chunk = np.zeros_like(rectified)
    for i in range(NUM_CANALES):
        env_chunk[i, :], zi[:, :, i] = scipy.signal.sosfilt(sos, rectified[i, :], zi=zi[:, :, i])
    
    # Actualizar buffer
    env_buffer[:, :-CHUNK_SIZE] = env_buffer[:, CHUNK_SIZE:]
    env_buffer[:, -CHUNK_SIZE:] = env_chunk
t3 = time.perf_counter() - start
print(f"3. Envolvente IIR (Solo sobre el chunk nuevo): {t3:.4f} seg -> {(ITERACIONES/t3):.1f} FPS")

print(f"\nConclusión: El Método 3 es {t1/t3:.1f} veces más rápido que el método original de dsp-auditor.")
