import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

def calcular_coordenadas_discretas(paths_mediciones, canales_seleccionados, mapped_names, 
                                   mode='estadistico', thresh_stats=3.0, thresh_manual=None, 
                                   vocales_config=None, out_dir=None):
    """
    Modo Estadístico: Umbrales independientes basados en ruido inter-pulso.
    Modo Manual: Umbrales definidos por canal basados en el pico máximo global del pulso.
    Anotación de Vocales: Superpone la secuencia de vocales si se activa.
    """
    
    for medicion_path in paths_mediciones:
        nombre_medicion = os.path.basename(medicion_path)
        print(f"\n[+] Procesando Coordenadas Discretas ({mode}) para: {nombre_medicion}")
        
        data_por_canal = {}
        bpm = 60.0 # Default fallback
        meta_path = os.path.join(medicion_path, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    bpm = meta.get('bpm', 60.0)
            except Exception:
                pass
                
        for ch in canales_seleccionados:
            json_path = os.path.join(medicion_path, ch, "analisis_results.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    segmentos = data.get('segmentos_rs', None)
                    if segmentos is None:
                        for k, v in data.items():
                            if isinstance(v, dict) and 'segmentos_rs' in v:
                                segmentos = v['segmentos_rs']
                                break
                    if segmentos and len(segmentos) > 0:
                        data_por_canal[ch] = segmentos
                except Exception as e:
                    print(f"    - Error leyendo {ch}: {e}")
                    
        if not data_por_canal:
            continue
            
        num_pulsos = min([len(segs) for segs in data_por_canal.values()])
        if num_pulsos == 0: continue
        
        # 0. Preparar secuencia de vocales si aplica
        secuencia_vocales = []
        if vocales_config and vocales_config.get('enabled'):
            base_seq = ['a', 'e', 'i', 'o', 'u']
            if vocales_config.get('secuencia') == 'inverso':
                base_seq = ['u', 'o', 'i', 'e', 'a']
            
            primera = vocales_config.get('primera', 'a')
            if primera in base_seq:
                idx = base_seq.index(primera)
                base_seq = base_seq[idx:] + base_seq[:idx]
                
            # Repetir secuencia hasta cubrir todos los pulsos
            secuencia_vocales = [base_seq[i % len(base_seq)] for i in range(num_pulsos)]
        
        senales_concatenadas = {ch: [] for ch in canales_seleccionados if ch in data_por_canal}
        umbrales_visuales = {}
        codigos_binarios = []
        
        canales_validos = [ch for ch in canales_seleccionados if ch in data_por_canal]
        
        # ----- LOGICA MODO ESTADISTICO -----
        if mode == 'estadistico':
            ruido_stats = {}
            for ch in canales_validos:
                segs = data_por_canal[ch]
                ruido_vals = []
                for i in range(num_pulsos):
                    arr = np.array(segs[i])
                    edge_len = max(1, int(len(arr) * 0.15))
                    ruido_vals.extend(arr[:edge_len])
                    ruido_vals.extend(arr[-edge_len:])
                    
                ruido_mean = np.mean(ruido_vals)
                ruido_std = np.std(ruido_vals)
                thresh_abs = thresh_stats * ruido_std 
                
                max_canal = 0.0
                for i in range(num_pulsos):
                    arr = np.array(segs[i]) - ruido_mean
                    m = np.max(np.abs(arr))
                    if m > max_canal: max_canal = m
                if max_canal == 0: max_canal = 1.0
                
                ruido_stats[ch] = {
                    'mean': ruido_mean,
                    'thresh_abs': thresh_abs,
                    'max_canal': max_canal
                }
                umbrales_visuales[ch] = thresh_abs / max_canal
                
            for i in range(num_pulsos):
                codigo_pulso = {}
                for ch in canales_validos:
                    arr = np.array(data_por_canal[ch][i])
                    stats = ruido_stats[ch]
                    arr_sin_offset = arr - stats['mean']
                    
                    estado = 1 if np.max(arr_sin_offset) >= stats['thresh_abs'] else 0
                    codigo_pulso[ch] = estado
                    
                    arr_norm = arr_sin_offset / stats['max_canal']
                    senales_concatenadas[ch].extend(arr_norm.tolist())
                    
                codigos_binarios.append(codigo_pulso)

        # ----- LOGICA MODO MANUAL (Global por pulso) -----
        else:
            for ch in canales_validos:
                umbrales_visuales[ch] = thresh_manual.get(ch, 0.5)
                
            for i in range(num_pulsos):
                # Encontrar el máximo global para este pulso i (sin offsets, pura amplitud bruta)
                recortes_i = {ch: np.array(data_por_canal[ch][i]) for ch in canales_validos}
                max_global = max([np.max(np.abs(arr)) for arr in recortes_i.values()] + [0.0001])
                
                codigo_pulso = {}
                for ch in canales_validos:
                    norm_pulse = recortes_i[ch] / max_global
                    senales_concatenadas[ch].extend(norm_pulse.tolist())
                    
                    umbral_ch = thresh_manual.get(ch, 0.5)
                    estado = 1 if np.max(norm_pulse) >= umbral_ch else 0
                    codigo_pulso[ch] = estado
                    
                codigos_binarios.append(codigo_pulso)
                
        # --- ANALISIS E IMPRESION ---
        tuplas_codigos = [tuple(c[ch] for ch in canales_validos) for c in codigos_binarios]
        conteo = Counter(tuplas_codigos)
        codigo_mas_frecuente, frec = conteo.most_common(1)[0]
        porcentaje = (frec / num_pulsos) * 100
        str_codigo_frecuente = f"Código más frecuente: {codigo_mas_frecuente} ({porcentaje:.1f}% de los pulsos)"
        print(f"    - {str_codigo_frecuente}")
        
        # --- GRAFICA ---
        fig, axes = plt.subplots(len(canales_validos), 1, figsize=(12, 3 * len(canales_validos)), sharex=True)
        if len(canales_validos) == 1: axes = [axes]
        
        tiempo_por_pulso = 60.0 / bpm
        
        for idx, ch in enumerate(canales_validos):
            ax = axes[idx]
            sig = np.array(senales_concatenadas[ch])
            
            # Eje X en tiempo (segundos) basado en los BPM y cantidad de pulsos
            total_duration = num_pulsos * tiempo_por_pulso
            time_axis = np.linspace(0, total_duration, len(sig))
            
            ax.plot(time_axis, sig, color='black', alpha=0.8, linewidth=1.0)
            
            thresh_viz = umbrales_visuales[ch]
            lbl_umbral = f'Umbral Estadístico (N={thresh_stats})' if mode == 'estadistico' else f'Umbral Manual ({thresh_viz})'
            ax.axhline(thresh_viz, color='red', linestyle='--', alpha=0.8, label=lbl_umbral)
            
            if mode == 'estadistico':
                ax.axhline(0, color='blue', linestyle=':', alpha=0.5, label='Línea Base')
                
            ax.set_ylabel('Amplitud Relativa')
            ax.set_ylim(min(-0.1, np.min(sig)-0.1), 1.1)
            ax.set_title(f"{mapped_names.get(ch, ch)}", fontweight='bold')
            
            for i in range(num_pulsos):
                t_inicio = i * tiempo_por_pulso
                t_centro = t_inicio + (tiempo_por_pulso / 2.0)
                
                ax.axvline(t_inicio, color='gray', linestyle=':', alpha=0.5)
                
                # Letra de la vocal si está activada
                if secuencia_vocales:
                    ax.text(t_centro, 1.0, f"'{secuencia_vocales[i]}'", color='blue', 
                            fontsize=12, fontweight='bold', ha='center', va='bottom')
                    
                # Bit binario
                bit = codigos_binarios[i][ch]
                ax.text(t_centro, 0.85 if secuencia_vocales else 0.9, f"{bit}", 
                        color='green' if bit==1 else 'red', fontsize=12, fontweight='bold', ha='center', va='center')
                
            if idx == 0: ax.legend(loc='upper right')
                
        axes[-1].set_xlabel('Tiempo (s)')
        str_modo = "Umbrales Estadísticos" if mode == 'estadistico' else "Umbrales Manuales"
        fig.suptitle(f"Discrete Motor Coordinates ({str_modo})\nMedición: {nombre_medicion}\n{str_codigo_frecuente}", fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        save_dir = out_dir if out_dir else medicion_path
        os.makedirs(save_dir, exist_ok=True)
        
        # Diferenciar el nombre del archivo según el modo
        sufijo = "estadistico" if mode == "estadistico" else "manual"
        out_file = os.path.join(save_dir, f"discrete_motor_{sufijo}_{nombre_medicion}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    -> Gráfico exportado: {out_file}")
