import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample, find_peaks

# Añadir el root al sys.path para importar analisis_trevisan
current_dir = os.path.dirname(os.path.abspath(__file__))
base_repo_dir = os.path.abspath(os.path.join(current_dir, ".."))
if base_repo_dir not in sys.path:
    sys.path.append(base_repo_dir)

import analisis_trevisan as at
from deep_learning.modelos import ConvAutoencoder1D

def get_interpulse_noise(env_segment, fallback_noise):
    if len(env_segment) < 3: return fallback_noise
    return np.median(env_segment)

def decodificar_secuencia(carpeta_secuencia, modelo_path, alpha_ruido=1.0, smooth_ms=150, notch_q=2.0):
    print(f"\n==================================================")
    print(f"DECODIFICANDO SECUENCIA CONTINUA")
    print(f"Carpeta: {carpeta_secuencia}")
    print(f"==================================================")
    
    if not os.path.exists(carpeta_secuencia):
        print(f"Error: No existe la carpeta {carpeta_secuencia}")
        return
        
    if not os.path.exists(modelo_path):
        print(f"Error: No se encontró el modelo entrenado en {modelo_path}")
        return

    # 1. Leer metadata o inferir BPM
    bpm_u = 40
    noise_u = 5.0
    pulsos_u = 20
    
    # Buscar metadata en canal_0
    meta_path = os.path.join(carpeta_secuencia, "canal_0", "metadata.json")
    if os.path.exists(meta_path):
        import json
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                bpm_u = meta.get('bpm', bpm_u)
                noise_u = meta.get('noise_seconds', noise_u)
                pulsos_u = meta.get('pulse_count', pulsos_u)
        except:
            pass

    print(f"Configuración DSP -> BPM: {bpm_u}, Smooth: {smooth_ms}ms, Notch Q: {notch_q}, Alpha Ruido: {alpha_ruido}")

    # 2. Procesar con Trevisan canal por canal
    canales_features = ["canal_0", "canal_1", "canal_2"]
    canales_procesar = ["canal_0", "canal_1", "canal_2", "canal_3"]
    data = {}
    
    for ch in canales_procesar:
        ch_dir = os.path.join(carpeta_secuencia, ch)
        if not os.path.exists(ch_dir):
            print(f"Error: Falta la carpeta del {ch}.")
            return
            
        res_ch = at.procesar_wavs_promedio(
            carpeta=ch_dir, output_root=ch_dir,
            bpm=bpm_u, mostrar_recortes=False,
            noise_seconds=noise_u, n_pulsos_manual=pulsos_u,
            excluded_windows=[], show_interactive_plot=False,
            notch_q_factor=notch_q, tipo_envolvente="rms", smooth_ms=smooth_ms
        )
        
        if not res_ch:
            print(f"Error: Falló el procesamiento en {ch}.")
            return
            
        fname = list(res_ch.keys())[0]
        data[ch] = res_ch[fname]

    # 3. Alineación y Ventaneo (Canal 3 maestro)
    muestras_pulso = data["canal_3"]['muestras_pulso']
    env_mic_raw = data["canal_3"]['env_recortada']
    
    dist_samples = int(0.8 * muestras_pulso)
    min_height = np.max(env_mic_raw) * 0.2
    
    picos_mic, _ = find_peaks(env_mic_raw, distance=dist_samples, height=min_height)
    print(f"Se detectaron {len(picos_mic)} ventanas/vocalizaciones en la secuencia.")
    
    if len(picos_mic) == 0:
        print("No se detectaron picos en el micrófono.")
        return

    TARGET_LEN = 100
    X_tensores = []
    ventanas_validas_grafico = [] # Guardar las formas de onda crudas para plotear
    
    for win_idx, pico in enumerate(picos_mic):
        pre_samples = int(muestras_pulso * 0.4)
        post_samples = int(muestras_pulso * 0.6)
        
        real_cut_start = pico - pre_samples
        real_cut_end = pico + post_samples
        
        if real_cut_start < 0 or real_cut_end > len(env_mic_raw):
            continue
            
        valido = True
        segs_brutos = []
        max_supremo = 1e-9
        
        for ch in canales_features:
            env_ch_raw = data[ch]['env_recortada']
            if real_cut_end > len(env_ch_raw):
                valido = False
                break
                
            segmento_ch = env_ch_raw[real_cut_start:real_cut_end].copy()
            
            initial_noise = data[ch].get('noise_levels', [0])[0] if len(data[ch].get('noise_levels', [])) > 0 else 0
            noise_win_samples = max(3, int(muestras_pulso / 4.0))
            
            noise_start_pre = max(0, int(pico - 0.5 * muestras_pulso - noise_win_samples))
            noise_end_pre = min(len(env_ch_raw), noise_start_pre + noise_win_samples)
            ruido_pre = initial_noise
            if noise_end_pre > noise_start_pre:
                ruido_pre = get_interpulse_noise(env_ch_raw[noise_start_pre:noise_end_pre], initial_noise)
                
            noise_start_post = min(len(env_ch_raw), int(pico + 0.5 * muestras_pulso))
            noise_end_post = min(len(env_ch_raw), noise_start_post + noise_win_samples)
            ruido_post = ruido_pre
            if noise_end_post > noise_start_post:
                ruido_post = get_interpulse_noise(env_ch_raw[noise_start_post:noise_end_post], initial_noise)
                
            ruido_promedio = (ruido_pre + ruido_post) / 2.0
            ruido_a_restar = ruido_promedio * alpha_ruido
            
            segmento_ch = np.maximum(segmento_ch - ruido_a_restar, 0)
            
            m_val = np.max(segmento_ch)
            if m_val > max_supremo:
                max_supremo = m_val
                
            segs_brutos.append(segmento_ch)
            
        if not valido:
            continue
            
        vector_concatenado = []
        for seg in segs_brutos:
            seg_norm = seg / max_supremo
            seg_rs = resample(seg_norm, TARGET_LEN)
            seg_rs[seg_rs < 0] = 0.0
            vector_concatenado.append(seg_rs)
            
        tensor_sample = np.stack(vector_concatenado) # (3, 100)
        X_tensores.append(tensor_sample)
        ventanas_validas_grafico.append(vector_concatenado) # Las 3 señales remuestreadas para plotear

    if len(X_tensores) == 0:
        print("Ninguna ventana fue válida tras la extracción.")
        return

    # 4. Inferencia con la Red Neuronal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cargando modelo en dispositivo: {device}")
    
    # Asumimos latent_dim=16 como entrenamos por defecto
    model = ConvAutoencoder1D(latent_dim=16).to(device)
    model.load_state_dict(torch.load(modelo_path, map_location=device, weights_only=True))
    model.eval()
    
    X_torch = torch.tensor(np.array(X_tensores), dtype=torch.float32).to(device)
    
    mapa_vocales = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}
    predicciones = []
    
    with torch.no_grad():
        _, _, logits = model(X_torch)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        for p in preds:
            predicciones.append(mapa_vocales[p])
            
    # Imprimir predicciones como texto
    print("\n--- SECUENCIA DE VOCALES PREDICHAS ---")
    print(" -> ".join(predicciones))
    print("--------------------------------------\n")

    # 5. Graficar resultados ventana por ventana
    n_wins = len(ventanas_validas_grafico)
    cols = 5
    rows = int(np.ceil(n_wins / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2.5))
    fig.suptitle(f"Decodificación de Secuencia Continua: {os.path.basename(carpeta_secuencia)}", fontsize=16)
    
    # Asegurar que axes sea iterable y plano
    if n_wins == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
        
    for idx in range(len(axes)):
        ax = axes[idx]
        if idx < n_wins:
            sigs = ventanas_validas_grafico[idx]
            vocal_pred = predicciones[idx]
            
            ax.plot(sigs[0], label="C0 (Masetero)", color='blue')
            ax.plot(sigs[1], label="C1 (Digástrico)", color='red')
            ax.plot(sigs[2], label="C2 (Risorio)", color='green')
            
            ax.set_title(f"Win {idx} -> {vocal_pred}", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.legend(fontsize=8)
        else:
            ax.axis('off') # Ocultar subplots sobrantes
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        carpeta = sys.argv[1]
    else:
        print("Uso: python decodificador_continuo.py <ruta_carpeta_secuencia>")
        sys.exit(1)
        
    modelo_path = os.path.join(base_repo_dir, "resultados", "resultados_autoencoder", "autoencoder_emg_16d.pth")
    decodificar_secuencia(carpeta, modelo_path)
