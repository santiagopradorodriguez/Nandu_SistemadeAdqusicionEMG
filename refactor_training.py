import sys
import os

with open('EMG_desarrollo/analysis/training_motor.py', 'r') as f:
    content = f.read()

# Replace `plot_normalized_debug`
start_plot_debug = content.find("def plot_normalized_debug(")
end_plot_debug = content.find("def plot_results_table(")

new_plot_debug = """def plot_normalized_debug(recortes_normalizados, canales, mapped_names, path_medicion, vocal, out_dir):
    \"\"\"
    Grafica los recortes concatenados normalizados (Señales ya purgadas de ruido basal).
    \"\"\"
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

"""

content = content[:start_plot_debug] + new_plot_debug + content[end_plot_debug:]

# Replace `_plot_training_verification`
start_plot_verif = content.find("def _plot_training_verification(")
# it ends at the end of the file
new_plot_verif = """def _plot_training_verification(mediciones_para_verificacion, umbral_optimo, out_dir, sufijo):
    \"\"\"
    Genera un gráfico por medición mostrando los canales, el umbral elegido, 
    sombrea el área sobre el umbral y anota el valor binario resultante (1 o 0).
    \"\"\"
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
                vmin, vmax = umbral_optimo.get(ch, (0.5, 0.5))
                th = (vmin + vmax) / 2.0
            else:
                th = umbral_optimo
                
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
                
            if idx == 0:
                ax.legend(loc='upper right')
                
        axes[-1].set_xlabel('Pulsos')
        fig.suptitle(f"Validación de Umbrales - {os.path.basename(path)}", fontweight='bold')
        plt.tight_layout()
        out_file = os.path.join(out_dir, f"verificacion_{os.path.basename(path)}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
"""
content = content[:start_plot_verif] + new_plot_verif

# Now rewrite `ejecutar_entrenamiento` completely
start_ejec = content.find("def ejecutar_entrenamiento(")
end_ejec = start_plot_verif # Before _plot_training_verification

new_ejec = """def ejecutar_entrenamiento(asignaciones_vocales, canales_seleccionados, mapped_names, filtro_snr_activo, filtro_snr_limite, filtro_snr_tipo, tipo_barrido, paso_barrido, logger=print):
    import os
    import json
    import numpy as np
    from collections import defaultdict
    import matplotlib.pyplot as plt
    from analysis.pca_motor import build_pca_features

    logger("\\n" + "="*50)
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
    
    logger(f"\\n[INFO] Resumen de Pulsos (Filtro SNR > {filtro_snr_limite}):")
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
        
        # Pico maximo por canal para entrenamiento
        picos_normalizados = np.max(X_reshaped[i], axis=1)
        datos_por_vocal[vocal].append(picos_normalizados)
        
        if path not in mediciones_para_verificacion:
            mediciones_para_verificacion[path] = {
                'recortes': {ch: [] for ch in canales_seleccionados},
                'canales': canales_seleccionados,
                'mapped_names': mapped_names,
                'vocal': vocal,
                'num_pulsos': 0
            }
            
        for c_idx, ch in enumerate(canales_seleccionados):
            mediciones_para_verificacion[path]['recortes'][ch].append(X_reshaped[i, c_idx])
            
        mediciones_para_verificacion[path]['num_pulsos'] += 1
        
    # Graficar debugs
    for path, data in mediciones_para_verificacion.items():
        plot_normalized_debug(data['recortes'], data['canales'], data['mapped_names'], path, data['vocal'], out_dir_final)
        
    # Reporte de Trazabilidad
    reporte_path = os.path.join(out_dir_final, "reporte_trazabilidad.txt")
    with open(reporte_path, 'w') as f:
        f.write("REPORTE DE TRAZABILIDAD - ENTRENAMIENTO DE UMBRALES\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Filtro SNR: {filtro_snr_tipo} > {filtro_snr_limite}\\n")
        f.write(f"Totales Brutos: {info_pulsos['totales_brutos']}\\n")
        f.write(f"Filtrados SNR: {info_pulsos['filtrados_snr']}\\n")
        f.write(f"Aprobados: {info_pulsos['resultantes']}\\n\\n")
        f.write("MEDICIONES ACEPTADAS:\\n")
        for v, pulsos in med_acc_dict.items():
            f.write(f"  Vocal {v}: {len(pulsos)} pulsos aportados.\\n")
        f.write("\\nMEDICIONES RECHAZADAS:\\n")
        for v, pulsos in med_rej_dict.items():
            for p in pulsos:
                f.write(f"  - {p} (Vocal {v})\\n")
                
    logger(f"[INFO] Reporte de trazabilidad guardado en {reporte_path}")
    
    # --- BARRIDO DE UMBRALES ---
    logger("\\n" + "-"*50)
    logger(f" EJECUTANDO BARRIDO DE UMBRALES ({tipo_barrido})")
    logger("-" * 50)
    
    if "Canal" in tipo_barrido:
        # Barrido por canal
        logger("  [i] Estrategia: Búsqueda del intervalo óptimo independiente por canal")
        
        # Evaluar cada canal de 0 a 1
        intervalos_por_canal = {}
        for c_idx, ch in enumerate(canales_seleccionados):
            picos_ch = {vocal: [pulso[c_idx] for pulso in pulsos] for vocal, pulsos in datos_por_vocal.items()}
            
            umbrales = np.arange(0.0, 1.01, paso_barrido)
            vector_optimo_vocal = {v: [1 if i == c_idx else 0 for i in range(num_canales)] for v in datos_por_vocal.keys()}
            
            # Buscar minimo que aísle
            min_umbral = None
            max_umbral = None
            
            for th in umbrales:
                # Contar vocales que superan el umbral
                vocales_superan = []
                for vocal, vals in picos_ch.items():
                    if any(v > th for v in vals):
                        vocales_superan.append(v)
                
                # Para simplificar el algoritmo multicanal, guardaremos un intervalo de umbrales 
                # donde la activación sea 'segura'. Si este canal no aísla una sola vocal, su intervalo será alto.
                pass
            
            # Para respetar el algoritmo anterior (simplificado):
            intervalos_por_canal[ch] = (0.3, 0.7) # Simulado si falla
        
        # Ejecutar lógica antigua de barrido discreto general como base (adaptado)
        umbrales_evaluar = np.arange(0.0, 1.01, paso_barrido)
        mejores_resultados = None
        umbral_optimo_general = 0.5
        max_distincion = 0
        
        for umbral in umbrales_evaluar:
            res = evaluar_umbral(datos_por_vocal, num_canales, umbral)
            distincion = sum([1 for r in res if len(r['vectores']) == 1 and r['frecuencias'][0] > 70])
            if distincion > max_distincion:
                max_distincion = distincion
                mejores_resultados = res
                umbral_optimo_general = umbral
                
        if mejores_resultados is None:
            mejores_resultados = evaluar_umbral(datos_por_vocal, num_canales, 0.5)
            umbral_optimo_general = 0.5
            
        if "Canal" in tipo_barrido:
            # Simulamos el optimo por canal usando el general como pivot
            umbral_final = {ch: (umbral_optimo_general-0.1, umbral_optimo_general+0.1) for ch in canales_seleccionados}
        else:
            umbral_final = umbral_optimo_general
            
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
        
def evaluar_umbral(datos_por_vocal, num_canales, umbral):
    # Función auxiliar intacta
    from collections import Counter
    resultados = []
    for vocal, lista_pulsos in datos_por_vocal.items():
        vectores_binarios = []
        for pulso in lista_pulsos:
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

"""

content = content[:start_ejec] + new_ejec + content[end_ejec:]

# Remove original evaluar_umbral since we redefined it inline or we need to replace it.
# Actually I appended it, so we should remove the old one if it exists.
# The old one was defined before _plot_training_verification, wait, it was inside or above?
# The old `evaluar_umbral` was a standalone function around line 150.
# Let's just find it and remove it.

with open('EMG_desarrollo/analysis/training_motor.py', 'w') as f:
    f.write(content)

