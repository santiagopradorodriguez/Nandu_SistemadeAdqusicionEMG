import os
import glob
import subprocess
import json
import argparse
import re
import sys
from datetime import datetime

# Asegurar path para módulos de EMG_desarrollo
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path_target = os.path.join(repo_root, "EMG_desarrollo")
if sys_path_target not in sys.path:
    sys.path.insert(0, sys_path_target)

def escape_latex(text):
    """Escapa caracteres reservados de LaTeX."""
    if not text:
        return ""
    text = str(text)
    text = text.replace('\\', '')
    text = text.replace('_', r'\_')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')
    text = text.replace('$', r'\$')
    return text

def parse_prueba_sort_key(folder_name):
    """Devuelve una clave de ordenamiento natural para nombres como A_Pruebat1, A_Prueba2, etc."""
    nums = re.findall(r'\d+', folder_name)
    val = int(nums[0]) if nums else 999
    return (val, folder_name)

def find_image_in_folder(vocal_folder, img_type):
    """Busca dinámicamente la ruta de la imagen en la carpeta según el tipo de gráfico."""
    if not os.path.isdir(vocal_folder):
        return None
        
    imgs = []
    if img_type == "patron":
        imgs = glob.glob(os.path.join(vocal_folder, "patron_muscular_grabacion.png"))
    elif img_type == "paper":
        imgs = glob.glob(os.path.join(vocal_folder, "plot_paper_combined.png"))
        if not imgs:
            try:
                from deep_learning.dataset_tools.plot_3_musculos_standalone import generar_plot_3_musculos
                gen_p = generar_plot_3_musculos(vocal_folder, mostrar=False)
                if gen_p and os.path.exists(gen_p):
                    imgs = [gen_p]
            except Exception as err:
                print(f"[Aviso] No se pudo autogenerar plot 3 músculos para {vocal_folder}: {err}")
    elif img_type == "calib":
        imgs = glob.glob(os.path.join(vocal_folder, "plot_calibrado_*.png"))
        if not imgs:
            try:
                from analysis.plotter_calibrado import plotear_medicion_secuencial
                fecha = os.path.basename(os.path.dirname(vocal_folder))
                toma = os.path.basename(vocal_folder)
                cfg_calib = {
                    'notch': True, 'bandpass': True, 'tipo_env': 'rms',
                    'start_time': None, 'end_time': None, 'tema_oscuro': False, 'graficar_fft': False
                }
                plotear_medicion_secuencial(f"{fecha}/{toma}", cfg_calib, mostrar_plot=False)
                imgs = glob.glob(os.path.join(vocal_folder, "plot_calibrado_*.png"))
            except Exception as err:
                print(f"[Aviso] No se pudo autogenerar plot calibrado para {vocal_folder}: {err}")
    elif img_type == "multimodal":
        imgs = glob.glob(os.path.join(vocal_folder, "plot_espectrograma_multimodal.png"))
        if not imgs:
            try:
                from analysis.generador_figura_multimodal import generar_figura_paper_multimodal
                gen_p = generar_figura_paper_multimodal(vocal_folder)
                if gen_p and os.path.exists(gen_p):
                    imgs = [gen_p]
            except Exception as err:
                print(f"[Aviso] No se pudo autogenerar figura multimodal para {vocal_folder}: {err}")
                
    if imgs and os.path.exists(imgs[0]):
        return os.path.abspath(imgs[0]).replace('\\', '/')
    return None

def generar_reporte(config_input):
    config = {}
    config_input_str = str(config_input)
    
    if os.path.isfile(config_input_str) and config_input_str.endswith('.json'):
        with open(config_input_str, 'r', encoding='utf-8') as f:
            config = json.load(f)
        fecha = config.get("fecha", datetime.now().strftime("%Y-%m-%d"))
        base_dir = config.get("directorio_base", f"../EMG_desarrollo/base_de_datos_electrodos/{fecha}")
    else:
        # Se ingresó una fecha o una ruta directa a una sesión
        if os.path.isdir(config_input_str):
            base_dir = config_input_str
            fecha = os.path.basename(base_dir)
        else:
            fecha = config_input_str
            base_dir = os.path.join(repo_root, "EMG_desarrollo", "base_de_datos_electrodos", fecha)
            
        # Extraer metadatos automáticos desde la primera toma encontrada
        canales = {"0": "Canal 0", "1": "Canal 1", "2": "Canal 2"}
        sujeto = "Sujeto"
        for d in sorted(os.listdir(base_dir)) if os.path.isdir(base_dir) else []:
            m_file = os.path.join(base_dir, d, "canal_0", "metadata.json")
            if os.path.exists(m_file):
                try:
                    with open(m_file, 'r', encoding='utf-8') as mf:
                        m_data = json.load(mf)
                        sujeto = m_data.get('sujeto', sujeto)
                        m_map = m_data.get('muscles_map', {})
                        if m_map:
                            canales["0"] = m_map.get("canal_0", canales["0"])
                            canales["1"] = m_map.get("canal_1", canales["1"])
                            canales["2"] = m_map.get("canal_2", canales["2"])
                        break
                except Exception:
                    pass
                    
        config = {
            "fecha": fecha,
            "baterias": "Alimentación por baterías de 9V (bajo ruido)",
            "tierra": "Referencia GND en apófisis mastoides",
            "electrodos_nota": f"Registro de superficie sEMG submáximal ({escape_latex(sujeto)})",
            "canales": canales,
            "musculos_nota": f"Canal 0: {escape_latex(canales['0'])}, Canal 1: {escape_latex(canales['1'])}, Canal 2: {escape_latex(canales['2'])}",
            "secuencia": "Secuencia periódica de fonación vocálica guiada por metrónomo a 30 BPM",
            "notas": "Reporte técnico integral compilado de forma automatizada por el sistema."
        }
        
    base_dir = os.path.abspath(base_dir)
    
    # 1. Cabecera y Configuración
    doc = []
    doc.append(r"\documentclass[11pt,a4paper]{article}")
    doc.append(r"\usepackage[utf8]{inputenc}")
    doc.append(r"\usepackage[spanish]{babel}")
    doc.append(r"\usepackage{graphicx}")
    doc.append(r"\usepackage{geometry}")
    doc.append(r"\usepackage{caption}")
    doc.append(r"\usepackage{subcaption}")
    doc.append(r"\usepackage{hyperref}")
    doc.append(r"\usepackage{float}")
    doc.append(r"\geometry{top=2cm, bottom=2cm, left=2.5cm, right=2.5cm}")
    doc.append("")
    doc.append(r"\title{Reporte de Experimento EMG}")
    doc.append(r"\author{Ñandú - Sistema de Adquisición EMG}")
    doc.append(f"\\date{{{escape_latex(fecha)}}}")
    doc.append("")
    doc.append(r"\begin{document}")
    doc.append("")
    doc.append(r"\maketitle")
    doc.append("")
    doc.append(r"\section{Configuración y Hardware}")
    doc.append(f"\\textbf{{Baterías:}} {escape_latex(config.get('baterias', 'N/A'))} \\\\")
    doc.append(f"\\textbf{{Tierra:}} {escape_latex(config.get('tierra', 'N/A'))}")
    doc.append("")
    doc.append(r"\section{Armado y Electrodos}")
    doc.append(escape_latex(config.get('electrodos_nota', '')))
    doc.append("")
    doc.append(r"\section{Ubicación de los Músculos}")
    doc.append(r"\begin{itemize}")
    canales = config.get("canales", {})
    doc.append(f"    \\item \\textbf{{Canal 0:}} {escape_latex(canales.get('0', 'N/A'))}")
    doc.append(f"    \\item \\textbf{{Canal 1:}} {escape_latex(canales.get('1', 'N/A'))}")
    doc.append(f"    \\item \\textbf{{Canal 2:}} {escape_latex(canales.get('2', 'N/A'))}")
    doc.append(r"\end{itemize}")
    doc.append(escape_latex(config.get('musculos_nota', '')))
    doc.append("")
    doc.append(r"\section{Protocolo y Secuencia}")
    doc.append(escape_latex(config.get('secuencia', '')))
    doc.append("")
    doc.append(r"\section{Notas y Observaciones}")
    doc.append(escape_latex(config.get('notas', '')))
    doc.append("")
    
    # 2. Patrones Musculares Comparativa (si existen)
    doc.append(r"\newpage")
    doc.append(r"\section{Patrones Musculares Promedio}")
    doc.append("Comparativa de los patrones musculares promedio suavizados para las 5 vocales fonatorias:")
    doc.append("")
    
    patron_imgs = {}
    for v in ["A", "E", "I", "O", "U"]:
        v_folders = sorted(glob.glob(os.path.join(base_dir, f"{v}_*")) + glob.glob(os.path.join(base_dir, f"{v.lower()}_*")))
        v_folders = [f for f in v_folders if os.path.isdir(f) and 'secuencia' not in os.path.basename(f).lower()]
        if v_folders:
            p_img = find_image_in_folder(v_folders[0], "patron")
            if p_img:
                patron_imgs[v] = p_img
                
    if patron_imgs:
        doc.append(r"\begin{figure}[H]")
        doc.append(r"\centering")
        v_keys = [k for k in ["A", "E", "I", "O", "U"] if k in patron_imgs]
        for idx, vk in enumerate(v_keys):
            doc.append(r"\begin{subfigure}{0.31\textwidth}")
            doc.append(f"    \\includegraphics[width=\\textwidth]{{{patron_imgs[vk]}}}")
            doc.append(f"    \\caption{{Vocal {vk}}}")
            doc.append(r"\end{subfigure}")
            if idx == 2:
                doc.append(r"\vspace{0.4cm}")
            elif idx < len(v_keys) - 1:
                doc.append(r"\hfill")
        doc.append(r"\caption{Comparativa de patrones musculares promedio suavizados.}")
        doc.append(r"\end{figure}")
        doc.append("")

    # 3. Recopilar carpetas por vocal
    vocales = ["A", "E", "I", "O", "U"]
    tomas_por_vocal = {}
    for v in vocales:
        v_folders = glob.glob(os.path.join(base_dir, f"{v}_*")) + glob.glob(os.path.join(base_dir, f"{v.lower()}_*"))
        v_folders = [f for f in v_folders if os.path.isdir(f) and 'secuencia' not in os.path.basename(f).lower()]
        v_folders.sort(key=parse_prueba_sort_key)
        if v_folders:
            tomas_por_vocal[v] = v_folders

    # 4. Sección de Análisis de Señales por Vocal: Plot Calibrado y Plot 3 Músculos Paper
    doc.append(r"\newpage")
    doc.append(r"\section{Análisis de Señales por Vocal}")
    doc.append("A continuación se presentan los registros de 3 músculos del paper y las señales calibradas agrupadas vocal por vocal:")
    doc.append("")

    for v in vocales:
        if v not in tomas_por_vocal:
            continue
            
        doc.append(f"\\subsection{{Vocal {v}}}")
        doc.append(f"Se presentan a continuación las tomas registradas para la vocal \\textbf{{{v}}}, ordenadas cronológicamente por número de prueba:")
        doc.append("")
        
        for folder in tomas_por_vocal[v]:
            toma_name = os.path.basename(folder)
            doc.append(f"\\subsubsection{{Medición: {escape_latex(toma_name)}}}")
            
            paper_img = find_image_in_folder(folder, "paper")
            calib_img = find_image_in_folder(folder, "calib")
            
            if paper_img:
                doc.append(r"\begin{figure}[H]")
                doc.append(r"\centering")
                doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{paper_img}}}")
                doc.append(f"\\caption{{Registro de 3 músculos de la vocal {v} - {escape_latex(toma_name)}: línea temporal de ventanas y concatenación simétrica con ruido restado.}}")
                doc.append(r"\end{figure}")
                doc.append("")
                
            if calib_img:
                doc.append(r"\begin{figure}[H]")
                doc.append(r"\centering")
                doc.append(f"\\includegraphics[width=0.88\\textwidth]{{{calib_img}}}")
                doc.append(f"\\caption{{Señales calibradas de la vocal {v} - {escape_latex(toma_name)}: filtro Notch en 50 Hz, pasabanda y envolvente RMS.}}")
                doc.append(r"\end{figure}")
                doc.append("")

    # 5. Sección Final: Análisis Multimodal de Señales y Espectrogramas (4 paneles)
    doc.append(r"\newpage")
    doc.append(r"\section{Análisis Multimodal de Señales y Espectrogramas}")
    doc.append("A continuación se presentan los registros multimodales de 4 paneles para las mediciones de la sesión, agrupados vocal por vocal. Cada figura integra el espectrograma acústico STFT con pre-énfasis, el oscilograma del micrófono rectificado con su envolvente acústica, la activación EMG normalizada por el Supremo Tricanal del pulso y el espectrograma muscular RGB:")
    doc.append("")

    for v in vocales:
        if v not in tomas_por_vocal:
            continue
            
        doc.append(f"\\subsection{{Vocal {v}: Espectrogramas y Registro Multimodal}}")
        doc.append("")
        
        for folder in tomas_por_vocal[v]:
            toma_name = os.path.basename(folder)
            spec_img = find_image_in_folder(folder, "multimodal")
            
            if spec_img:
                doc.append(r"\begin{figure}[H]")
                doc.append(r"\centering")
                doc.append(f"\\includegraphics[width=0.55\\textwidth]{{{spec_img}}}")
                doc.append(f"\\caption{{Análisis multimodal de 4 paneles para la vocal {v} - {escape_latex(toma_name)}: espectrograma acústico con pre-énfasis, señal de micrófono rectificada, activación muscular sEMG por Supremo Tricanal y espectrograma RGB.}}")
                doc.append(r"\end{figure}")
                doc.append("")

    doc.append(r"\end{document}")
    
    tex_content = "\n".join(doc)
    
    # Guardar archivo .tex
    report_name = f"Reporte_EMG_{fecha}"
    out_dir = os.path.abspath(os.path.dirname(__file__))
    tex_path = os.path.join(out_dir, f"{report_name}.tex")
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
        
    print(f"Generado archivo TeX en: {tex_path}")
    
    # Compilar con pdflatex
    print("Compilando PDF...")
    try:
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{report_name}.tex"], cwd=out_dir, check=True, stdout=subprocess.DEVNULL)
        # Compilar dos veces para asegurar referencias
        subprocess.run(["pdflatex", "-interaction=nonstopmode", f"{report_name}.tex"], cwd=out_dir, check=True, stdout=subprocess.DEVNULL)
        pdf_path = os.path.join(out_dir, report_name + '.pdf')
        print(f"PDF generado exitosamente: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print("Error al compilar el PDF. Verifica si LaTeX está instalado correctamente y los paths a las imágenes no tienen caracteres raros.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de reportes en LaTeX para experimentos EMG")
    parser.add_argument("config", nargs="?", default="2026-09-23", help="Ruta al archivo JSON de configuración o fecha de sesión (ej. 2026-09-23)")
    parser.add_argument("--sesion", "--fecha", dest="sesion", help="Fecha o carpeta de la sesión (ej. 2026-09-23)")
    args = parser.parse_args()
    
    target = args.sesion if args.sesion else args.config
    generar_reporte(target)
