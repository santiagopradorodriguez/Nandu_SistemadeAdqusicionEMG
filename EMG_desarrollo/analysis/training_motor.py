import os
import json
import numpy as np
from collections import defaultdict, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_normalized_debug(recortes_normalizados, canales, mapped_names, path_medicion, vocal, out_dir):
    """
    Grafica los recortes concatenados normalizados (Señales ya purgadas de ruido basal).
    """
    num_pulsos = len(recortes_normalizados[canales[0]])
    fig, axes = plt.subplots(len(canales), 1, figsize=(12, 3 * len(canales)), sharex=True)
    if len(canales) == 1: axes = [axes]
    
    for idx, ch in enumerate(canales):
        ax = axes[idx]
        
        sig_concat = []
        for i in range(num_pulsos):
            sig_concat.extend(recortes_normalizados[ch][i].tolist())
            
        sig = np.array(sig_concat)
        time_axis = np.linspace(0, num_pulsos, len(sig)) # Eje x relativo en "pulsos"
        
        ax.plot(time_axis, sig, color='black', alpha=0.8, linewidth=1.0)
        ax.axhline(0, color='blue', linestyle=':', alpha=0.5, label='Línea Base (Purgada)')
        ax.set_ylabel('Amplitud Normalizada')
        ax.set_ylim(-0.1, 1.1)
        
        nombre_musculo = mapped_names.get(ch, ch)
        ax.set_title(f"Músculo: {nombre_musculo} ({ch}) | Vocal: {vocal}", fontweight='bold')
        
        for i in range(num_pulsos + 1):
            ax.axvline(i, color='gray', linestyle='--', alpha=0.3)
            
        if idx == 0:
            ax.legend(loc='upper right')
            
    axes[-1].set_xlabel('Pulsos (Ventanas concatenadas)')
    fig.suptitle(f"Señal Normalizada y Purgada - {os.path.basename(path_medicion)}", fontweight='bold')
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"signal_debug_{os.path.basename(path_medicion)}.png")
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
        
    # Si umbral_optimo es un dict, significa que son valores por canal
    is_intervalo = isinstance(umbral_optimo, dict)
    
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
    max_lineas = max([len(r['vectores']) for r in resultados_tabla]) if resultados_tabla else 1
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
        str_umbral = "\n".join([f"{ch}: {val:.2f}" if not isinstance(val, (tuple, list)) else f"{ch}: [{val[0]:.2f} - {val[1]:.2f}]" for ch, val in umbral_optimo.items()])
    else:
        str_umbral = f"{float(umbral_optimo):.2f}"
        
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

def ejecutar_entrenamiento(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, tipo_barrido, paso_barrido, nombre_set=None, estetica_config=None, logger=print):
    import os
    import json
    import numpy as np
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from analysis.pca_motor import build_pca_features

    logger("\n" + "="*50)
    logger(" INICIANDO ENTRENAMIENTO DE UMBRALES (MOTOR UNIFICADO)")
    logger("="*50)

    out_dir_general = list(asignaciones_vocales.keys())[0] if asignaciones_vocales else ""
    if not out_dir_general:
        raise ValueError("No se enviaron mediciones para entrenar.")
        
    session_dir = os.path.dirname(out_dir_general)
    base_umbrales_dir = os.path.join(session_dir, "UMBRALES")
    os.makedirs(base_umbrales_dir, exist_ok=True)
    
    # Carpeta unica
    snr_lim_str = f"{filtro_snr_limite:.1f}".replace('.', '-')
    if nombre_set and str(nombre_set).strip():
        clean_set = str(nombre_set).strip().replace(' ', '_')
        base_folder_name = f"UMBRALES_{clean_set}_SNR{snr_lim_str}"
    else:
        base_folder_name = f"UMBRALES_SNR{snr_lim_str}"
    
    sufijo_folder = ""
    contador = 1
    while True:
        folder_name = f"{base_folder_name}{sufijo_folder}"
        folder_path = os.path.join(base_umbrales_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            break
        sufijo_folder = f"_{contador}"
        contador += 1
        
    out_dir_final = folder_path
    
    # Motor unificado
    logger(f"[INFO] Extrayendo características de {len(asignaciones_vocales)} mediciones...")
    X_full, Y_arr, Roles_arr, Tomas, med_acc_dict, med_rej_dict, info_pulsos = build_pca_features(
        asignaciones_vocales, canales_seleccionados, mapped_names, logger,
        filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo
    )
    
    logger(f"\n[INFO] Resumen de Pulsos (Filtro SNR > {filtro_snr_limite}):")
    logger(f"  - Totales Brutos: {info_pulsos['totales_brutos']}")
    logger(f"  - Filtrados SNR: {info_pulsos['filtrados_snr']}")
    logger(f"  - Aprobados: {info_pulsos['resultantes']}")
    
    if len(X_full) == 0:
        raise ValueError("No hay suficientes pulsos válidos para entrenar. Ajuste el filtro SNR.")
        
    # Reconstruir datos para el algoritmo
    datos_por_vocal = defaultdict(list)
    mediciones_para_verificacion = {}
    
    num_canales = len(canales_seleccionados)
    X_reshaped = X_full.reshape(X_full.shape[0], num_canales, 100)
    
    for i in range(len(X_full)):
        vocal = Y_arr[i]
        toma = Tomas[i] 
        base_toma = toma.rsplit('_W', 1)[0]
        
        # Encontrar path original
        path = next((p for p in asignaciones_vocales.keys() if os.path.basename(p) == base_toma), base_toma)
        
        # Pico maximo por canal bruto
        picos_brutos = np.max(X_reshaped[i], axis=1)
        
        if path not in mediciones_para_verificacion:
            mediciones_para_verificacion[path] = {
                'recortes': {ch: [] for ch in canales_seleccionados},
                'picos_secuencia': [], # Para guardar y detrendear luego
                'vocal_list': [],
                'canales': canales_seleccionados,
                'mapped_names': mapped_names,
                'vocal': vocal,
                'num_pulsos': 0
            }
            
        mediciones_para_verificacion[path]['picos_secuencia'].append(picos_brutos)
        mediciones_para_verificacion[path]['vocal_list'].append(vocal)
        for c_idx, ch in enumerate(canales_seleccionados):
            mediciones_para_verificacion[path]['recortes'][ch].append(X_reshaped[i, c_idx])
            
        mediciones_para_verificacion[path]['num_pulsos'] += 1

    # --- APLICAR DETRENDING POR MEDICIÓN ---
    for path, data in mediciones_para_verificacion.items():
        picos_matrix = np.array(data['picos_secuencia']) # shape (n_pulsos, n_canales)
        picos_detrended = picos_matrix.copy()
        
        # Detrending lineal (Corrección de fatiga/electrodos)
        x_idx = np.arange(len(picos_detrended))
        if len(picos_detrended) > 1:
            for c_idx in range(picos_detrended.shape[1]):
                y_vals = picos_detrended[:, c_idx]
                coefs = np.polyfit(x_idx, y_vals, 1)
                trend = np.polyval(coefs, x_idx)
                picos_detrended[:, c_idx] = np.maximum(y_vals - trend + np.mean(y_vals), 0.0)
                
        # Normalizar tras el detrending
        max_pico_ventana = np.max(picos_detrended, axis=1) + 1e-9
        picos_norm = picos_detrended / max_pico_ventana[:, np.newaxis]
        
        # Sincronizar recortes de señal para que los gráficos de verificación coincidan al 100%
        for idx_p in range(len(picos_norm)):
            for c_idx, ch in enumerate(data['canales']):
                orig_peak = picos_matrix[idx_p, c_idx] + 1e-9
                detrended_peak = picos_detrended[idx_p, c_idx]
                scale_factor = detrended_peak / orig_peak
                seg_scaled = data['recortes'][ch][idx_p] * scale_factor
                data['recortes'][ch][idx_p] = seg_scaled / max_pico_ventana[idx_p]
        
        # Guardar en datos_por_vocal y continuar
        for idx_p, p_norm in enumerate(picos_norm):
            v = data['vocal_list'][idx_p]
            datos_por_vocal[v].append(p_norm)
        
    # Graficar debugs
    for path, data in mediciones_para_verificacion.items():
        plot_normalized_debug(data['recortes'], data['canales'], data['mapped_names'], path, data['vocal'], out_dir_final)
        
    # Reporte de Trazabilidad
    reporte_path = os.path.join(out_dir_final, "reporte_trazabilidad.txt")
    with open(reporte_path, 'w') as f:
        f.write("REPORTE DE TRAZABILIDAD - ENTRENAMIENTO DE UMBRALES\n")
        f.write("="*50 + "\n\n")
        f.write(f"Filtro SNR: {filtro_snr_tipo} > {filtro_snr_limite}\n")
        f.write(f"Totales Brutos: {info_pulsos['totales_brutos']}\n")
        f.write(f"Filtrados SNR: {info_pulsos['filtrados_snr']}\n")
        f.write(f"Aprobados: {info_pulsos['resultantes']}\n\n")
        f.write("MEDICIONES ACEPTADAS:\n")
        for v, pulsos in med_acc_dict.items():
            f.write(f"  Vocal {v}: {len(pulsos)} pulsos aportados.\n")
        f.write("\nMEDICIONES RECHAZADAS:\n")
        for v, pulsos in med_rej_dict.items():
            for p in pulsos:
                f.write(f"  - {p} (Vocal {v})\n")
                
    logger(f"[INFO] Reporte de trazabilidad guardado en {reporte_path}")
    
    # --- BARRIDO DE UMBRALES ---
    logger("\n" + "-"*50)
    logger(f" EJECUTANDO BARRIDO DE UMBRALES ({tipo_barrido})")
    logger("-" * 50)
    
    import itertools
    from collections import Counter
    
    def calcular_score_separabilidad(resultados):
        total_vocales = len(resultados)
        if total_vocales == 0:
            return -999999.0, 0, 0.0

        modas = []
        frecs = []
        for r in resultados:
            if r['moda_global'] is not None and len(r['frecuencias']) > 0:
                modas.append(r['moda_global'])
                frecs.append(r['frecuencias'][0])
            else:
                modas.append(None)
                frecs.append(0.0)

        degen_zeros = tuple([0] * num_canales)
        degen_ones = tuple([1] * num_canales)

        valid_modas = [m for m in modas if m is not None]
        conteo_modas = Counter(valid_modas)

        vocales_aisladas = 0
        colisiones = 0
        for m in valid_modas:
            if m in (degen_zeros, degen_ones):
                colisiones += 3
            elif conteo_modas[m] == 1:
                vocales_aisladas += 1
            else:
                colisiones += 1

        vectores_unicos = len(set(valid_modas) - {degen_zeros, degen_ones})
        frec_media = float(np.mean(frecs)) if len(frecs) > 0 else 0.0

        score = (vocales_aisladas * 1000.0) + (vectores_unicos * 100.0) + frec_media - (colisiones * 50.0)
        return score, vocales_aisladas, frec_media

    # Evitar umbrales triviales extremos (buscar en 0.05 a 0.95)
    paso_seguro = max(0.01, float(paso_barrido))
    umbrales_base = np.arange(0.05, 0.96, paso_seguro)
    mejor_score = -999999.0
    mejor_aisladas = 0
    mejor_frec_media = 0.0
    mejores_resultados = None
    
    if "Canal" in tipo_barrido:
        logger("  [i] Estrategia: Búsqueda del umbral óptimo independiente por canal")
        logger(f"  [i] Evaluando {len(umbrales_base)**num_canales} combinaciones posibles...")
        
        combinaciones = list(itertools.product(umbrales_base, repeat=num_canales))
        umbral_optimo_general = {ch: 0.5 for ch in canales_seleccionados}
        
        for comb in combinaciones:
            comb_vals = [round(float(x), 3) for x in comb]
            res = evaluar_umbral(datos_por_vocal, num_canales, comb_vals)
            score, aisladas, frec_media = calcular_score_separabilidad(res)
            
            if score > mejor_score:
                mejor_score = score
                mejor_aisladas = aisladas
                mejor_frec_media = frec_media
                mejores_resultados = res
                umbral_optimo_general = {ch: comb_vals[i] for i, ch in enumerate(canales_seleccionados)}
                
        umbral_final = umbral_optimo_general
        logger(f"  [+] Mejor distinción lograda: {mejor_aisladas} vocales aisladas con vector exclusivo (Consistencia media: {mejor_frec_media:.1f}%).")
        
    else:
        logger("  [i] Estrategia: Búsqueda de umbral común para todos los canales")
        umbral_optimo_general = 0.5
        
        for umbral in umbrales_base:
            th_val = round(float(umbral), 3)
            res = evaluar_umbral(datos_por_vocal, num_canales, th_val)
            score, aisladas, frec_media = calcular_score_separabilidad(res)
            
            if score > mejor_score:
                mejor_score = score
                mejor_aisladas = aisladas
                mejor_frec_media = frec_media
                mejores_resultados = res
                umbral_optimo_general = th_val
                
        if mejores_resultados is None:
            mejores_resultados = evaluar_umbral(datos_por_vocal, num_canales, 0.5)
            umbral_optimo_general = 0.5
            
        umbral_final = umbral_optimo_general
        logger(f"  [+] Mejor distinción lograda: {mejor_aisladas} vocales aisladas con vector exclusivo (Umbral: {umbral_final:.2f}, Consistencia media: {mejor_frec_media:.1f}%).")
        
    plot_results_table(mejores_resultados, umbral_final, out_dir_final, filtro_snr_tipo, filtro_snr_limite, asignaciones_vocales, folder_name)
    _plot_training_verification(mediciones_para_verificacion, umbral_final, out_dir_final, folder_name)
    
    # Guardar JSON
    out_json = os.path.join(out_dir_final, "umbrales_optimos.json")
    res_dict = {
        "estrategia": tipo_barrido,
        "umbral_optimo": umbral_final,
        "resultados_detallados": mejores_resultados
    }
    with open(out_json, 'w') as f:
        json.dump(res_dict, f, indent=4)

    # Replicar resultados en galeria central para el visor de la interfaz
    import shutil
    try:
        cur = out_dir_final
        found_root = None
        while cur and cur != os.path.dirname(cur):
            if os.path.exists(os.path.join(cur, "resultados")) or os.path.exists(os.path.join(cur, "base_de_datos_electrodos")):
                found_root = cur
                break
            cur = os.path.dirname(cur)
        if found_root:
            central_dir = os.path.join(found_root, "resultados", "resultados_umbrales", folder_name)
            os.makedirs(central_dir, exist_ok=True)
            for item in os.listdir(out_dir_final):
                s_item = os.path.join(out_dir_final, item)
                d_item = os.path.join(central_dir, item)
                if os.path.isfile(s_item):
                    shutil.copy2(s_item, d_item)
            logger(f"[+] Resultados replicados en galeria central: {central_dir}")
    except Exception as e:
        logger(f"[!] Aviso al replicar en galeria central: {e}")
        
    logger(f"\n[+] Entrenamiento Finalizado. Graficos y JSON guardados en: {out_dir_final}")

def evaluar_umbral(datos_por_vocal, num_canales, umbral):
    from collections import Counter
    resultados = []
    is_list = isinstance(umbral, (list, tuple, np.ndarray))
    
    for vocal, lista_pulsos in datos_por_vocal.items():
        vectores_binarios = []
        for pulso in lista_pulsos:
            if is_list:
                binario = tuple([1 if val > th else 0 for val, th in zip(pulso, umbral)])
            else:
                binario = tuple([1 if val > umbral else 0 for val in pulso])
            vectores_binarios.append(binario)
            
        conteo = Counter(vectores_binarios)
        total = len(vectores_binarios)
        
        vectores_str = []
        frecuencias = []
        
        for vec, count in conteo.most_common():
            frec = (count / total) * 100
            if frec > 5:
                vectores_str.append(vec)
                frecuencias.append(frec)
                
        moda = conteo.most_common(1)[0][0] if conteo else None
        
        resultados.append({
            'vocal': vocal,
            'total_pulsos': total,
            'moda_global': moda,
            'vectores': vectores_str,
            'frecuencias': frecuencias
        })
    return resultados

def _plot_training_verification(mediciones_para_verificacion, umbral_optimo, out_dir, sufijo):
    """
    Genera un gráfico por medición mostrando los canales, el umbral elegido, 
    sombrea el área sobre el umbral y anota el valor binario resultante (1 o 0).
    """
    for path, data in mediciones_para_verificacion.items():
        recortes = data['recortes']
        canales = data['canales']
        vocal = data['vocal']
        num_pulsos = data['num_pulsos']
        
        fig, axes = plt.subplots(len(canales), 1, figsize=(12, 3 * len(canales)), sharex=True)
        if len(canales) == 1: axes = [axes]
        
        for idx, ch in enumerate(canales):
            ax = axes[idx]
            sig_concat = []
            
            for i in range(num_pulsos):
                sig_concat.extend(recortes[ch][i].tolist())
                
            sig = np.array(sig_concat)
            time_axis = np.linspace(0, num_pulsos, len(sig))
            
            # Obtener umbral específico para este canal
            if isinstance(umbral_optimo, dict):
                val = umbral_optimo.get(ch, 0.5)
                th = (val[0] + val[1]) / 2.0 if isinstance(val, (tuple, list)) else float(val)
            else:
                th = float(umbral_optimo)
                
            # Graficar señal
            ax.plot(time_axis, sig, color='black', alpha=0.8, linewidth=1.0)
            
            # Sombrear área sobre el umbral
            ax.fill_between(time_axis, th, sig, where=(sig > th), color='green', alpha=0.3, label='Señal > Umbral (1)')
            ax.fill_between(time_axis, 0, sig, where=(sig <= th), color='red', alpha=0.1, label='Señal < Umbral (0)')
            
            ax.axhline(th, color='purple', linestyle='-', linewidth=2, label=f'Umbral Final: {th:.2f}')
            
            # Anotar valor binario por pulso
            for p in range(num_pulsos):
                inicio = p
                fin = p + 1
                segment_mask = (time_axis >= inicio) & (time_axis < fin)
                segment_sig = sig[segment_mask]
                
                if len(segment_sig) > 0:
                    max_val = np.max(segment_sig)
                    binario = 1 if max_val > th else 0
                    
                    color_txt = 'green' if binario == 1 else 'red'
                    ax.text(p + 0.5, 1.05, str(binario), color=color_txt, fontweight='bold', fontsize=12, ha='center')
                    
            ax.set_ylabel('Amplitud Norm.')
            ax.set_ylim(-0.1, 1.2)
            
            nombre_musculo = data['mapped_names'].get(ch, ch)
            ax.set_title(f"Músculo: {nombre_musculo} ({ch}) | Vocal: {vocal}", fontweight='bold')
            
            for i in range(num_pulsos + 1):
                ax.axvline(i, color='gray', linestyle='--', alpha=0.3)
                
            ax.legend(loc='upper right')
                
        axes[-1].set_xlabel('Pulsos')
        fig.suptitle(f"Validación de Umbrales - {os.path.basename(path)}", fontweight='bold')
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"verificacion_{os.path.basename(path)}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
