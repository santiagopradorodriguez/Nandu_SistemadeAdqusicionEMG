import os
import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_normalized_debug(recortes_normalizados, canales, mapped_names, path_medicion, vocal, bpm):
    """
    Grafica los recortes concatenados normalizados. El eje X se convierte a tiempo real.
    Colorea en rojo el 15% de los bordes de cada recorte (donde se estimó el ruido basal).
    """
    num_pulsos = len(recortes_normalizados[canales[0]])
    fig, axes = plt.subplots(len(canales), 1, figsize=(12, 3 * len(canales)), sharex=True)
    if len(canales) == 1: axes = [axes]
    
    tiempo_por_pulso = 60.0 / bpm
    total_duration = num_pulsos * tiempo_por_pulso
    
    for idx, ch in enumerate(canales):
        ax = axes[idx]
        
        # Mapear el eje X para que coincida con el tiempo teórico de BPM
        sig_concat = []
        is_noise_concat = []
        
        for i in range(num_pulsos):
            arr = recortes_normalizados[ch][i]
            edge_len = max(1, int(len(arr) * 0.15))
            sig_concat.extend(arr.tolist())
            
            # Crear un mask booleano para colorear el ruido
            mask = np.zeros(len(arr), dtype=bool)
            mask[:edge_len] = True
            mask[-edge_len:] = True
            is_noise_concat.extend(mask.tolist())
            
        sig = np.array(sig_concat)
        is_noise = np.array(is_noise_concat)
        time_axis = np.linspace(0, total_duration, len(sig))
        
        # Graficar señal completa en negro
        ax.plot(time_axis, sig, color='black', alpha=0.8, linewidth=1.0)
        
        # Superponer en rojo las zonas de ruido
        # Usamos scatter o ploteamos solo los segmentos usando máscaras con NaN
        sig_ruido = sig.copy()
        sig_ruido[~is_noise] = np.nan
        ax.plot(time_axis, sig_ruido, color='red', alpha=0.9, linewidth=1.5, label='Ruido (Muestra)')
        
        ax.axhline(0, color='blue', linestyle=':', alpha=0.5, label='0 Real (Sin Baseline)')
        ax.set_ylabel('Amplitud Normalizada')
        ax.set_ylim(-0.1, 1.1)
        
        nombre_musculo = mapped_names.get(ch, ch)
        ax.set_title(f"Músculo: {nombre_musculo} ({ch}) | Vocal: {vocal}", fontweight='bold')
        
        for i in range(num_pulsos):
            t_inicio = i * tiempo_por_pulso
            ax.axvline(t_inicio, color='gray', linestyle=':', alpha=0.5)
            
        if idx == 0: ax.legend(loc='upper right')
        
    axes[-1].set_xlabel('Tiempo (s)')
    fig.suptitle(f"Debug: Señal Normalizada - {os.path.basename(path_medicion)}", fontweight='bold')
    plt.tight_layout()
    out_file = os.path.join(path_medicion, f"training_debug_norm_vocal_{vocal}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_results_table(resultados_tabla, umbral_optimo, out_dir, filtro_snr_tipo, filtro_snr_limite, asignaciones_vocales, sufijo):
    """
    Genera un gráfico tipo tabla con los resultados finales del barrido.
    """
    # Preparar string corto para reporte
    if "Ambos" in filtro_snr_tipo:
        tipo_corto = "Ambos"
    elif "Global" in filtro_snr_tipo:
        tipo_corto = "Global"
    else:
        tipo_corto = "Ventana"
        
    # Si umbral_optimo es un dict, significa que son intervalos por canal
    is_intervalo = isinstance(umbral_optimo, dict)
    
    fig, ax = plt.subplots(figsize=(14, min(8, max(4, len(resultados_tabla)))))
    ax.axis('tight')
    ax.axis('off')
    
    col_labels = ["Vocal", "Total\n(Pulsos)", "Moda Global", "Vectores que aparecieron", "Frecuencia\nRelativa (%)"]
    table_data = []
    
    for r in resultados_tabla:
        # Formatear el vector y frecuencia
        str_vectores = "\n".join([str(v) for v in r['vectores']])
        str_frecs = "\n".join([f"{f:.1f}%" for f in r['frecuencias']])
        row = [
            r['vocal'],
            str(r['total_pulsos']),
            str(r['moda_global']),
            str_vectores,
            str_frecs
        ]
        table_data.append(row)
        
    # --- CREAR LA IMAGEN DE LA TABLA ---
    # Hacer el figsize dinámico para evitar solapamientos (altura depende del max de lineas)
    max_lineas = max([len(r['vectores']) for r in resultados_tabla])
    alto_celda = 0.5 * max_lineas
    alto_total = max(4, len(resultados_tabla) * alto_celda + 2)
    fig, ax = plt.subplots(figsize=(14, alto_total))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, alto_celda * 2) # Escala de celdas
    
    # Ajustar estilos
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif col in [3, 4]:
            cell.set_text_props(verticalalignment='center')
            
    # Titulo descriptivo
    str_umbral = ""
    if is_intervalo:
        str_umbral = "\n".join([f"{ch}: [{vmin:.2f} - {vmax:.2f}]" for ch, (vmin, vmax) in umbral_optimo.items()])
    else:
        str_umbral = f"{umbral_optimo:.2f}"
        
    vocales_usadas = sorted(list(set([r['vocal'] for r in resultados_tabla])))
    vocales_str = ", ".join(vocales_usadas)
        
    titulo = f"Resultados de Calibración Discreta\nVocales Analizadas: {vocales_str}\n(Umbral seleccionado: {str_umbral} | Filtro: {filtro_snr_tipo} > {filtro_snr_limite:.1f})"
    plt.title(titulo, fontweight='bold', pad=20)
    plt.tight_layout()
    
    out_file = os.path.join(out_dir, f"training_results_table_{sufijo}.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- CREAR TABLA EN LATEX ---
    latex_code = "\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n\\begin{document}\n"
    latex_code += "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
    latex_code += "\\textbf{Vocal} & \\textbf{Total (Pulsos)} & \\textbf{Moda Global} & \\textbf{Vectores} & \\textbf{Frecuencia (\\%)} \\\\\n\\hline\n"
    
    for r in resultados_tabla:
        vocal = r['vocal']
        total = r['total_pulsos']
        moda = str(r['moda_global'])
        vectores_str = "\\begin{tabular}{@{}c@{}}" + " \\\\ ".join([str(v) for v in r['vectores']]) + "\\end{tabular}"
        frec_str = "\\begin{tabular}{@{}c@{}}" + " \\\\ ".join([f"{f:.1f}\\%" for f in r['frecuencias']]) + "\\end{tabular}"
        latex_code += f"{vocal} & {total} & {moda} & {vectores_str} & {frec_str} \\\\\n\\hline\n"
        
    str_umbral_tex = str_umbral.replace('\n', ', ')
    latex_code += "\\end{tabular}\n"
    latex_code += f"\\caption{{Resultados de Calibración Discreta. Vocales: {vocales_str}. Umbral: {str_umbral_tex}. Filtro SNR {tipo_corto}: $>$ {filtro_snr_limite:.1f}}}\n"
    latex_code += "\\label{tab:coordenadas_discretas}\n\\end{table}\n"
    
    # Agregar lista de mediciones en página nueva
    latex_code += "\\newpage\n"
    latex_code += "\\section*{Mediciones utilizadas en el entrenamiento}\n"
    latex_code += "\\begin{itemize}\n"
    
    # Procesar paths para mostrar desde 'base_de_datos_electrodos'
    for path in sorted(asignaciones_vocales.keys()):
        # Extraer desde base_de_datos_electrodos si existe
        parts = path.split(os.sep)
        try:
            idx = parts.index("base_de_datos_electrodos")
            rel_path = os.sep.join(parts[idx:])
        except ValueError:
            rel_path = os.path.basename(path)
            
        # Escapar caracteres de latex como guiones bajos
        rel_path_tex = rel_path.replace('_', '\\_')
        vocal = asignaciones_vocales[path]
        latex_code += f"\\item {rel_path_tex} (Vocal: {vocal})\n"
        
    latex_code += "\\end{itemize}\n"
    latex_code += "\\end{document}\n"
    
    tex_file = os.path.join(out_dir, f"training_results_table_{sufijo}.tex")
    with open(tex_file, 'w') as f:
        f.write(latex_code)
        
    print(f"\n[+] Tabla exportada como Imagen: {out_file}")
    print(f"[+] Tabla exportada en LaTeX: {tex_file}")

def ejecutar_entrenamiento(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, tipo_barrido, paso_barrido, logger=print):
    import itertools
    logger("\n" + "="*50)
    logger(" INICIANDO CALIBRACIÓN DE UMBRALES DISCRETOS")
    logger("="*50)
    
    datos_por_vocal = defaultdict(list)
    ventana_referencia = None
    mediciones_para_verificacion = {}
    
    # Para la tabla de reporte final necesitamos todos los códigos que aparecieron para cada pulso
    # estructura: codigos_crudos_vocal[vocal] = [picos_pulso_1, picos_pulso_2, ...]
    
    out_dir_general = list(asignaciones_vocales.keys())[0] if asignaciones_vocales else ""
    if out_dir_general:
        out_dir_general = os.path.dirname(out_dir_general) # carpeta padre
    
    for path, vocal in asignaciones_vocales.items():
        # Obtener canales intersectados (los que existen en la medición y fueron seleccionados en UI)
        canales_medicion = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)) and d.startswith('canal_')]
        canales = [ch for ch in canales_seleccionados if ch in canales_medicion]
        
        if not canales: continue
        
        # Extraer BPM del metadata.json para el eje de tiempo
        bpm = 60.0
        meta_path = os.path.join(path, "metadata.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    bpm = json.load(f).get('bpm', 60.0)
            except: pass
            
        json_path_0 = os.path.join(path, canales[0], "analisis_results.json")
        if not os.path.exists(json_path_0): continue
            
        with open(json_path_0, 'r') as f: data = json.load(f)
            
        env_ms = data.get('smooth_ms', None)
        if env_ms is None:
            for k, v in data.items():
                if isinstance(v, dict) and 'smooth_ms' in v:
                    env_ms = v['smooth_ms']
                    break
                    
        if ventana_referencia is None:
            ventana_referencia = env_ms
            logger(f"  [i] Ventana de envolvente de referencia fijada en: {ventana_referencia} ms")
        elif env_ms != ventana_referencia:
            raise ValueError(f"Inconsistencia: {os.path.basename(path)} tiene ventana de {env_ms}ms, pero se esperaba {ventana_referencia}ms.")
            
        # Extraer SNR global de cada canal y los segmentos de la ventana especifica
        recortes_medicion = {}
        min_snr_global = float('inf')
        
        for ch in canales:
            # Buscamos especificamente el archivo de la ventana de referencia
            ventana_str = str(ventana_referencia).replace('.', '_')
            jpath = os.path.join(path, ch, f"analisis_results_env{ventana_str}ms.json")
            if not os.path.exists(jpath):
                # Fallback con punto para compatibilidad con datos viejos
                jpath = os.path.join(path, ch, f"analisis_results_env{ventana_referencia}ms.json")
            if not os.path.exists(jpath):
                # Fallback al results generico
                jpath = os.path.join(path, ch, "analisis_results.json")
                
            if os.path.exists(jpath):
                with open(jpath, 'r') as f: d = json.load(f)
                
                # SNR Global de este canal (buscamos el peor canal)
                ch_snr_mean = d.get('snr_mean', 0)
                if ch_snr_mean < min_snr_global: min_snr_global = ch_snr_mean
                
                # Segmentos
                segs = d.get('segmentos_rs', None)
                if segs is None:
                    for k, v in d.items():
                        if isinstance(v, dict) and 'segmentos_rs' in v:
                            segs = v['segmentos_rs']
                            break
                if segs: recortes_medicion[ch] = segs
                    
        # Filtrado SNR previo (Global de la medición)
        aplica_global = "Global" in filtro_snr_tipo or "Ambos" in filtro_snr_tipo
        if filtro_snr_activo and aplica_global:
            if min_snr_global < filtro_snr_limite:
                logger(f"  [-] Medicion {os.path.basename(path)} descartada entera (SNR global mínimo {min_snr_global:.2f} < {filtro_snr_limite:.1f})")
                continue
                
        if not recortes_medicion: continue
            
        num_pulsos = min([len(s) for s in recortes_medicion.values()])
        canales_validos = sorted(list(recortes_medicion.keys()))
        
        recortes_limpios_debug = {ch: [] for ch in canales_validos}
        pulsos_validos_count = 0
        
        for i in range(num_pulsos):
            recorte_limpio = {}
            max_global = 0.0
            min_snr_pulso = float('inf') # Guardará el SNR del peor canal en este pulso
            
            for ch in canales_validos:
                arr = np.array(recortes_medicion[ch][i])
                edge_len = max(1, int(len(arr) * 0.15))
                orillas = np.concatenate([arr[:edge_len], arr[-edge_len:]])
                
                ruido_basal = np.mean(orillas)
                ruido_std = np.std(orillas)
                if ruido_std == 0: ruido_std = 1e-6
                
                arr_sin_offset = arr - ruido_basal
                recorte_limpio[ch] = arr_sin_offset
                
                m = np.max(np.abs(arr_sin_offset))
                if m > max_global: max_global = m
                
                # SNR individual de amplitud para este canal
                snr_ch = m / ruido_std
                if snr_ch < min_snr_pulso: min_snr_pulso = snr_ch
                
            # Filtro SNR por segmento
            aplica_ventana = "Ventana" in filtro_snr_tipo or "Ambos" in filtro_snr_tipo
            if filtro_snr_activo and aplica_ventana and min_snr_pulso < filtro_snr_limite:
                # El peor canal de este pulso no supera el ruido limite. Descartamos el pulso completo.
                continue
                
            if max_global == 0: max_global = 1.0
            
            picos_normalizados = []
            for ch in canales_validos:
                arr_norm = recorte_limpio[ch] / max_global
                recortes_limpios_debug[ch].append(arr_norm)
                pico_ch = np.max(arr_norm)
                picos_normalizados.append(pico_ch)
                
            datos_por_vocal[vocal].append(picos_normalizados)
            pulsos_validos_count += 1
            
        if pulsos_validos_count == 0:
            logger(f"  [-] Medicion {os.path.basename(path)} omitida: Ningún pulso superó el SNR > {filtro_snr_limite}")
            continue
            
        # Generar gráfico debug
        plot_normalized_debug(recortes_limpios_debug, canales_validos, mapped_names, path, vocal, bpm)
        logger(f"  [+] Añadida {os.path.basename(path)} a vocal '{vocal}' ({pulsos_validos_count}/{num_pulsos} pulsos válidos). Debug gráfico creado.")
        
        # Guardar para ploteo de verificación final
        mediciones_para_verificacion[path] = {
            'recortes': recortes_limpios_debug,
            'canales': canales_validos,
            'mapped_names': mapped_names,
            'vocal': vocal,
            'bpm': bpm,
            'num_pulsos': pulsos_validos_count
        }

    if not datos_por_vocal:
        raise ValueError("Ninguna medición superó los filtros.")

    # --- PASO 5: Barrido de Umbrales Multidimensional ---
    logger("\n" + "-"*50)
    logger(f" EJECUTANDO BARRIDO DE UMBRALES ({tipo_barrido})")
    logger("-" * 50)
    
    if "Canal" in tipo_barrido:
        # Barrido por canal independiente
        eje_umbrales = np.arange(paso_barrido, 1.00, paso_barrido)
        combinaciones = list(itertools.product(eje_umbrales, repeat=len(canales_validos)))
    else:
        # Barrido comun
        eje_umbrales = np.arange(0.01, 1.00, 0.01)
        combinaciones = [(u,) * len(canales_validos) for u in eje_umbrales]
        
    logger(f"  [i] Evaluando {len(combinaciones)} combinaciones de umbrales...")
    
    resultados_barrido = []
    
    for combo_th in combinaciones:
        coordenadas_por_vocal = {}
        score_acumulado = 0.0
        
        for vocal, lista_picos in datos_por_vocal.items():
            # Discretizamos usando el umbral correspondiente a cada canal
            codigos = []
            for picos in lista_picos:
                cod = tuple([1 if picos[i] >= combo_th[i] else 0 for i in range(len(canales_validos))])
                codigos.append(cod)
                
            c = Counter(codigos)
            coordenada_ganadora = c.most_common(1)[0][0]
            frecuencia_ganadora = c.most_common(1)[0][1] / len(codigos)
            
            coordenadas_por_vocal[vocal] = coordenada_ganadora
            score_acumulado += frecuencia_ganadora
            
        valores_unicos = set(coordenadas_por_vocal.values())
        colisiones = len(coordenadas_por_vocal) - len(valores_unicos)
        score_promedio = score_acumulado / len(datos_por_vocal)
        
        resultados_barrido.append({
            'umbral': combo_th,
            'colisiones': colisiones,
            'score': score_promedio,
            'coord_ganadoras': coordenadas_por_vocal
        })
        
    # Encontrar el mejor umbral (menor numero de colisiones y mayor score)
    min_colisiones = min([x['colisiones'] for x in resultados_barrido])
    candidatos_sin_colisiones = [x for x in resultados_barrido if x['colisiones'] == min_colisiones]
    
    max_score = max([x['score'] for x in candidatos_sin_colisiones])
    mejores_candidatos = [x for x in candidatos_sin_colisiones if x['score'] >= max_score * 0.99]
    
    if "Canal" in tipo_barrido:
        # Extraer intervalos para cada canal
        intervalos_optimos = {}
        for idx, ch in enumerate(canales_validos):
            valores_ch = [c['umbral'][idx] for c in mejores_candidatos]
            intervalos_optimos[ch] = (min(valores_ch), max(valores_ch))
            
        umbral_optimo_reporte = intervalos_optimos
        combo_final = mejores_candidatos[len(mejores_candidatos)//2]['umbral'] # Para generar la tabla usamos el punto medio
        str_rep = ", ".join([f"{ch}: [{vmin:.2f} - {vmax:.2f}]" for ch, (vmin, vmax) in intervalos_optimos.items()])
        logger(f"\n[INFO] Intervalos óptimos: {str_rep}")
        logger(f"  -> Score máximo logrado: {max_score*100:.1f}% de frecuencia modal promedio (Colisiones: {min_colisiones}).")
        
    else:
        umbral_optimo_reporte = mejores_candidatos[len(mejores_candidatos)//2]['umbral'][0]
        combo_final = mejores_candidatos[len(mejores_candidatos)//2]['umbral']
        logger(f"\n[INFO] Umbral óptimo seleccionado: {umbral_optimo_reporte:.2f} con {min_colisiones} colisiones de vocales.")
        logger(f"  -> Score máximo logrado: {max_score*100:.1f}% de frecuencia modal promedio.")
    
    # --- Generación de reporte final de Tabla ---
    resultados_tabla = []
    
    for vocal, lista_picos in datos_por_vocal.items():
        codigos = []
        for picos in lista_picos:
            cod = tuple([1 if picos[i] >= combo_final[i] else 0 for i in range(len(canales_validos))])
            codigos.append(cod)
            
        total_pulsos = len(codigos)
        c = Counter(codigos)
        
        moda_global = c.most_common(1)[0][0]
        
        vectores = []
        frecuencias = []
        for vec, count in c.most_common():
            vectores.append(vec)
            frecuencias.append((count / total_pulsos) * 100.0)
            
        resultados_tabla.append({
            'vocal': vocal,
            'total_pulsos': total_pulsos,
            'moda_global': moda_global,
            'vectores': vectores,
            'frecuencias': frecuencias
        })
        
    # --- Generación de Sufijo Único ---
    if "Ambos" in filtro_snr_tipo:
        tipo_corto = "Ambos"
    elif "Global" in filtro_snr_tipo:
        tipo_corto = "Global"
    else:
        tipo_corto = "Ventana"
        
    snr_lim_str = f"{filtro_snr_limite:.1f}".replace('.', '_')
    base_sufijo = f"Filtro{tipo_corto}_SNR{snr_lim_str}"
    
    sufijo_unico = base_sufijo
    contador = 1
    while os.path.exists(os.path.join(out_dir_general, f"training_results_table_{sufijo_unico}.tex")):
        sufijo_unico = f"{base_sufijo}_{contador}"
        contador += 1
        
    plot_results_table(resultados_tabla, umbral_optimo_reporte, out_dir_general, filtro_snr_tipo, filtro_snr_limite, asignaciones_vocales, sufijo_unico)
    
    # Generar gráficos de verificación
    logger("\n[INFO] Generando gráficos de Verificación de Entrenamiento...")
    _plot_training_verification(mediciones_para_verificacion, umbral_optimo_reporte, sufijo_unico)
    
    logger("\nAlgoritmo de calibración finalizado exitosamente.")

def _plot_training_verification(mediciones_para_verificacion, umbral_optimo, sufijo):
    """
    Genera un gráfico por medición mostrando los canales, el umbral elegido, 
    sombrea el área sobre el umbral y anota el valor binario resultante (1 o 0).
    """
    for path, data in mediciones_para_verificacion.items():
        recortes = data['recortes']
        canales = data['canales']
        vocal = data['vocal']
        bpm = data['bpm']
        num_pulsos = data['num_pulsos']
        
        fig, axes = plt.subplots(len(canales), 1, figsize=(12, 3 * len(canales)), sharex=True)
        if len(canales) == 1: axes = [axes]
        
        tiempo_por_pulso = 60.0 / bpm
        total_duration = num_pulsos * tiempo_por_pulso
        
        for idx, ch in enumerate(canales):
            ax = axes[idx]
            sig_concat = []
            
            for i in range(num_pulsos):
                sig_concat.extend(recortes[ch][i].tolist())
                
            sig = np.array(sig_concat)
            time_axis = np.linspace(0, total_duration, len(sig))
            
            # Obtener umbral específico para este canal
            if isinstance(umbral_optimo, dict):
                # Es un barrido por canal. Tomamos el promedio del intervalo
                vmin, vmax = umbral_optimo.get(ch, (0.5, 0.5))
                th = (vmin + vmax) / 2.0
            else:
                th = umbral_optimo
                
            # Graficar señal
            ax.plot(time_axis, sig, color='black', alpha=0.8, linewidth=1.0)
            
            # Sombrear área sobre el umbral
            ax.fill_between(time_axis, th, sig, where=(sig >= th), color='red', alpha=0.3, interpolate=True)
            
            # Dibujar línea de umbral
            ax.axhline(th, color='red', linestyle='--', alpha=0.8, label=f'Umbral Final ({th:.2f})')
            
            ax.set_ylabel('Amplitud Normalizada')
            ax.set_ylim(-0.1, 1.1)
            
            nombre_musculo = data['mapped_names'].get(ch, ch)
            ax.set_title(f"Músculo: {nombre_musculo} ({ch})", fontweight='bold')
            
            # Analizar cada pulso para escribir [1] o [0]
            for i in range(num_pulsos):
                t_inicio = i * tiempo_por_pulso
                t_fin = (i + 1) * tiempo_por_pulso
                ax.axvline(t_inicio, color='gray', linestyle=':', alpha=0.5)
                
                # Encontrar el pico de este pulso
                arr_pulso = recortes[ch][i]
                max_val = np.max(arr_pulso)
                
                # Determinar si pasó el umbral
                val_binario = 1 if max_val >= th else 0
                
                # Anotar en el gráfico, centrado en el pulso
                t_centro = t_inicio + (tiempo_por_pulso / 2.0)
                ax.text(t_centro, 1.02, f"[{val_binario}]", 
                        horizontalalignment='center', verticalalignment='bottom', 
                        color='red' if val_binario == 1 else 'gray',
                        fontweight='bold', fontsize=10)
                
            if idx == 0: ax.legend(loc='upper right')
            
        axes[-1].set_xlabel('Tiempo (s)')
        fig.suptitle(f"Training Verification - {os.path.basename(path)} (Vocal: {vocal})", fontweight='bold')
        plt.tight_layout()
        
        out_file = os.path.join(path, f"Training_Verification_vocal_{vocal}_{sufijo}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
