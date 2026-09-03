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
                
        # Buscar picos del canal maestro (microfono canal_3 si existe)
        mic_picos = None
        mic_path = os.path.join(medicion_path, "canal_3", "analisis_results.json")
        if os.path.exists(mic_path):
            try:
                with open(mic_path, 'r') as f:
                    d_mic = json.load(f)
                mic_picos = d_mic.get('maxima_per_cut', None)
            except Exception:
                pass

        noise_levels_por_canal = {}
        for ch in canales_seleccionados:
            json_path = os.path.join(medicion_path, ch, "analisis_results.json")
            if not os.path.exists(json_path):
                json_path = os.path.join(medicion_path, ch, "results.json")

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    segmentos = data.get('segmentos_rs', None)
                    if segmentos is None or len(segmentos) == 0:
                        for k, v in data.items():
                            if isinstance(v, dict) and 'segmentos_rs' in v and v['segmentos_rs']:
                                segmentos = v['segmentos_rs']
                                break
                                
                    # Si no está pre-segmentado, extraer directamente de env_recortada
                    if (segmentos is None or len(segmentos) == 0) and 'env_recortada' in data:
                        env = np.array(data['env_recortada'])
                        muestras_pulso = int(data.get('muestras_pulso', 4000))
                        picos = mic_picos if (mic_picos and len(mic_picos) > 0) else data.get('maxima_per_cut', [])
                        pre_s = int(muestras_pulso * 0.4)
                        post_s = int(muestras_pulso * 0.6)
                        
                        segs_ext = []
                        for p in picos:
                            s_start = int(p - pre_s)
                            s_end = int(p + post_s)
                            if s_start >= 0 and s_end <= len(env):
                                segs_ext.append(env[s_start:s_end].tolist())
                        if len(segs_ext) > 0:
                            segmentos = segs_ext

                    if segmentos and len(segmentos) > 0:
                        data_por_canal[ch] = segmentos
                        noise_levels_por_canal[ch] = data.get('noise_levels', [])
                except Exception as e:
                    print(f"    - Error leyendo {ch}: {e}")
                    
        if not data_por_canal:
            print(f"    [!] No se encontraron segmentos válidos para procesar en {nombre_medicion}.")
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
        
        # ----- LOGICA MODO ESTADISTICO (Ruido Interpulso Trevisan + Supremo Global Tricanal) -----
        if mode == 'estadistico':
            ruido_stats = {}
            for ch in canales_validos:
                # Usar noise_levels exacto de Trevisan si existe en analisis_results.json
                nl = noise_levels_por_canal.get(ch, [])
                if nl and len(nl) >= num_pulsos:
                    noise_vals = [float(x) for x in nl[:num_pulsos]]
                else:
                    # Si no hay noise_levels precalculado, estimar del segmento basal o recortes
                    segs = data_por_canal[ch]
                    noise_vals = [float(np.min(np.abs(segs[i]))) for i in range(num_pulsos)]
                    
                ruido_mean = float(np.mean(noise_vals)) if len(noise_vals) > 0 else 0.0
                ruido_std = float(np.std(noise_vals)) if len(noise_vals) > 0 else 1.0
                if ruido_std <= 0 or np.isnan(ruido_std):
                    ruido_std = max(1e-6, ruido_mean * 0.1)
                    
                thresh_abs = thresh_stats * ruido_std
                ruido_stats[ch] = {
                    'mean': ruido_mean,
                    'std': ruido_std,
                    'thresh_abs': thresh_abs
                }
            
            # Calcular señales normalizadas por el Supremo Global Tricanal de cada pulso
            max_supremos = []
            for i in range(num_pulsos):
                recortes_i = {ch: np.array(data_por_canal[ch][i]) for ch in canales_validos}
                arr_sin_offset = {ch: np.maximum(recortes_i[ch] - ruido_stats[ch]['mean'], 0.0) for ch in canales_validos}
                m_sup = max([float(np.max(arr_sin_offset[ch])) for ch in canales_validos] + [1e-6])
                max_supremos.append(m_sup)
                
                codigo_pulso = {}
                for ch in canales_validos:
                    # Normalización obligatoria por el Supremo Global Tricanal
                    arr_norm = arr_sin_offset[ch] / m_sup
                    senales_concatenadas[ch].extend(arr_norm.tolist())
                    
                    estado = 1 if np.max(arr_sin_offset[ch]) >= ruido_stats[ch]['thresh_abs'] else 0
                    codigo_pulso[ch] = estado
                    
                codigos_binarios.append(codigo_pulso)
                
            prom_supremo = float(np.mean(max_supremos)) if len(max_supremos) > 0 else 1.0
            for ch in canales_validos:
                umbrales_visuales[ch] = min(1.0, max(0.01, ruido_stats[ch]['thresh_abs'] / (prom_supremo + 1e-6)))

        # ----- LOGICA MODO MANUAL (Supremo Global Tricanal por pulso) -----
        else:
            for ch in canales_validos:
                th_val = thresh_manual.get(ch, 0.5) if isinstance(thresh_manual, dict) else float(thresh_manual or 0.5)
                umbrales_visuales[ch] = th_val
                
            for i in range(num_pulsos):
                recortes_i = {ch: np.array(data_por_canal[ch][i]) for ch in canales_validos}
                # Supremo Global Tricanal del pulso i
                max_supremo_i = max([float(np.max(np.abs(arr))) for arr in recortes_i.values()] + [1e-6])
                
                codigo_pulso = {}
                for ch in canales_validos:
                    norm_pulse = recortes_i[ch] / max_supremo_i
                    senales_concatenadas[ch].extend(norm_pulse.tolist())
                    
                    umbral_ch = thresh_manual.get(ch, 0.5) if isinstance(thresh_manual, dict) else float(thresh_manual or 0.5)
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

        # Replicar en resultados centrales para el visor
        import shutil
        try:
            cur = save_dir
            found_root = None
            while cur and cur != os.path.dirname(cur):
                if os.path.exists(os.path.join(cur, "resultados")) or os.path.exists(os.path.join(cur, "base_de_datos_electrodos")):
                    found_root = cur
                    break
                cur = os.path.dirname(cur)
            if found_root:
                c_dir = os.path.join(found_root, "resultados", "resultados_coordenadas_discretas")
                os.makedirs(c_dir, exist_ok=True)
                shutil.copy2(out_file, os.path.join(c_dir, os.path.basename(out_file)))
        except Exception as e:
            pass
