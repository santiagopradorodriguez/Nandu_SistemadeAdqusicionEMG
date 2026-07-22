import sys

with open('EMG_desarrollo/analysis/training_motor.py', 'r') as f:
    content = f.read()

# Fix 1: _plot_training_verification legend for all channels
old_legend = """            if idx == 0:
                ax.legend(loc='upper right')"""
new_legend = """            ax.legend(loc='upper right')"""
content = content.replace(old_legend, new_legend)

# Fix 2: Rewrite the threshold sweep logic completely
start_sweep = content.find("    # --- BARRIDO DE UMBRALES ---")
end_sweep = content.find("def evaluar_umbral(")

new_sweep = """    # --- BARRIDO DE UMBRALES ---
    logger("\\n" + "-"*50)
    logger(f" EJECUTANDO BARRIDO DE UMBRALES ({tipo_barrido})")
    logger("-" * 50)
    
    import itertools
    umbrales_base = np.arange(0.0, 1.01, paso_barrido)
    mejores_resultados = None
    max_distincion = -1
    mejor_frec_media = -1
    
    if "Canal" in tipo_barrido:
        logger("  [i] Estrategia: Búsqueda del umbral óptimo independiente por canal")
        logger(f"  [i] Evaluando {len(umbrales_base)**num_canales} combinaciones posibles...")
        
        combinaciones = list(itertools.product(umbrales_base, repeat=num_canales))
        umbral_optimo_general = {ch: 0.5 for ch in canales_seleccionados}
        
        for comb in combinaciones:
            res = evaluar_umbral(datos_por_vocal, num_canales, comb)
            modas = [r['moda_global'] for r in res if len(r['vectores']) >= 1 and r['frecuencias'][0] > 70]
            # Validar que no se superpongan vectores modales
            modas_unicas = set(modas)
            # Solo contamos si cada vocal dominante tiene un vector distinto
            distincion = len(modas_unicas)
            if distincion < len(modas):
                distincion = 0 # Castigamos si dos vocales caen en el mismo vector modal
                
            frec_media = np.mean([r['frecuencias'][0] for r in res if len(r['frecuencias']) > 0]) if distincion > 0 else 0
            
            if distincion > max_distincion or (distincion == max_distincion and frec_media > mejor_frec_media):
                max_distincion = distincion
                mejor_frec_media = frec_media
                mejores_resultados = res
                umbral_optimo_general = {ch: (comb[i]-0.01, comb[i]+0.01) for i, ch in enumerate(canales_seleccionados)}
                
        umbral_final = umbral_optimo_general
        logger(f"  [+] Mejor distinción lograda: {max_distincion} vocales aisladas.")
        
    else:
        logger("  [i] Estrategia: Búsqueda de umbral común para todos los canales")
        umbral_optimo_general = 0.5
        
        for umbral in umbrales_base:
            res = evaluar_umbral(datos_por_vocal, num_canales, umbral)
            modas = [r['moda_global'] for r in res if len(r['vectores']) >= 1 and r['frecuencias'][0] > 70]
            modas_unicas = set(modas)
            distincion = len(modas_unicas)
            if distincion < len(modas):
                distincion = 0
                
            frec_media = np.mean([r['frecuencias'][0] for r in res if len(r['frecuencias']) > 0]) if distincion > 0 else 0
            
            if distincion > max_distincion or (distincion == max_distincion and frec_media > mejor_frec_media):
                max_distincion = distincion
                mejor_frec_media = frec_media
                mejores_resultados = res
                umbral_optimo_general = umbral
                
        if mejores_resultados is None:
            mejores_resultados = evaluar_umbral(datos_por_vocal, num_canales, 0.5)
            umbral_optimo_general = 0.5
            
        umbral_final = umbral_optimo_general
        logger(f"  [+] Mejor distinción lograda: {max_distincion} vocales aisladas (Umbral: {umbral_final:.2f}).")
        
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
        
    logger(f"\\n[+] Entrenamiento Finalizado. Gráficos y JSON guardados en: {out_dir_final}")

"""

content = content[:start_sweep] + new_sweep + content[end_sweep:]

# Fix 3: Modify evaluar_umbral to handle a tuple of thresholds
old_evaluar = """def evaluar_umbral(datos_por_vocal, num_canales, umbral):
    # Función auxiliar intacta
    from collections import Counter
    resultados = []
    for vocal, lista_pulsos in datos_por_vocal.items():
        vectores_binarios = []
        for pulso in lista_pulsos:
            binario = tuple([1 if val > umbral else 0 for val in pulso])
            vectores_binarios.append(binario)"""
            
new_evaluar = """def evaluar_umbral(datos_por_vocal, num_canales, umbral):
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
            vectores_binarios.append(binario)"""

content = content.replace(old_evaluar, new_evaluar)

with open('EMG_desarrollo/analysis/training_motor.py', 'w') as f:
    f.write(content)
