import os
import glob
import subprocess
import json
import argparse
from datetime import datetime

LATEX_TEMPLATE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{hyperref}
\geometry{top=2cm, bottom=2cm, left=2.5cm, right=2.5cm}

\title{Reporte de Experimento EMG}
\author{Ñandú - Sistema de Adquisición EMG}
\date{<<FECHA>>}

\begin{document}

\maketitle

\section{Configuración y Hardware}
\textbf{Baterías:} <<BATERIAS>> \\
\textbf{Tierra:} <<TIERRA>>

\section{Armado y Electrodos}
<<ELECTRODOS_NOTA>>

\section{Ubicación de los Músculos}
\begin{itemize}
    \item \textbf{Canal 0:} <<CANAL_0>>
    \item \textbf{Canal 1:} <<CANAL_1>>
    \item \textbf{Canal 2:} <<CANAL_2>>
\end{itemize}
<<MUSCULOS_NOTA>>

\section{Protocolo y Secuencia}
<<SECUENCIA>>

\section{Notas y Observaciones}
<<NOTAS>>

\newpage
\section{Análisis de Señales}

\subsection{Patrones Musculares (Comparativa de Vocales)}
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PATRON_A>>}
        \caption{Vocal A}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PATRON_E>>}
        \caption{Vocal E}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PATRON_I>>}
        \caption{Vocal I}
    \end{subfigure}
    
    \vspace{0.5cm}
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PATRON_O>>}
        \caption{Vocal O}
    \end{subfigure}
    \hspace{1cm}
    \begin{subfigure}{0.32\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PATRON_U>>}
        \caption{Vocal U}
    \end{subfigure}
    \caption{Comparativa de los patrones musculares promedio suavizados para las 5 vocales.}
\end{figure}

\newpage
\subsection{Vocal A}
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.8\textwidth]{<<IMG_PAPER_A>>}
    \caption{Registro combinado de la vocal A. Señal con ruido restado, alineada y normalizada.}
\end{figure}
\begin{figure}[h!]
    \centering
    \includegraphics[width=0.8\textwidth]{<<IMG_CALIB_A>>}
    \caption{Señales calibradas de la vocal A (filtro notch, pasabanda 20-500 Hz, envolvente RMS 75 ms).}
\end{figure}

\newpage
\subsection{Vocales E e I}
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PAPER_E>>}
        \caption{Registro E}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PAPER_I>>}
        \caption{Registro I}
    \end{subfigure}
    \caption{Registro combinado de las vocales E e I.}
\end{figure}
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_CALIB_E>>}
        \caption{Calibrado E}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_CALIB_I>>}
        \caption{Calibrado I}
    \end{subfigure}
    \caption{Señales calibradas comparativas de las vocales E e I.}
\end{figure}

\newpage
\subsection{Vocales O y U}
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PAPER_O>>}
        \caption{Registro O}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_PAPER_U>>}
        \caption{Registro U}
    \end{subfigure}
    \caption{Registro combinado de las vocales O y U.}
\end{figure}
\begin{figure}[h!]
    \centering
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_CALIB_O>>}
        \caption{Calibrado O}
    \end{subfigure}
    \hfill
    \begin{subfigure}{0.48\textwidth}
        \includegraphics[width=\textwidth]{<<IMG_CALIB_U>>}
        \caption{Calibrado U}
    \end{subfigure}
    \caption{Señales calibradas comparativas de las vocales O y U.}
\end{figure}

\end{document}
"""

def find_image(base_dir, vocal, img_type):
    """Busca dinámicamente la ruta de la imagen según la vocal y el tipo de gráfico."""
    # Buscar la subcarpeta de la vocal (soporta minúsculas 'a_*' y mayúsculas 'A_*')
    folders = glob.glob(os.path.join(base_dir, f"{vocal.lower()}_*"))
    if not folders:
        folders = glob.glob(os.path.join(base_dir, f"{vocal.upper()}_*"))
    if not folders:
        return "example-image" # Fallback de LaTeX si no existe
    
    vocal_folder = folders[0]
    
    if img_type == "patron":
        pattern = "patron_muscular_grabacion.png"
    elif img_type == "paper":
        pattern = "plot_paper_combined.png"
    elif img_type == "calib":
        pattern = "plot_calibrado_*.png"
    
    img_search = os.path.join(vocal_folder, pattern)
    imgs = glob.glob(img_search)
    
    # Auto-generación de la figura multimodal paper si no existe
    if not imgs and img_type == "paper":
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys_path_target = os.path.join(repo_root, "EMG_desarrollo")
            if sys_path_target not in sys.path:
                sys.path.insert(0, sys_path_target)
            from analysis.generador_figura_multimodal import generar_figura_paper_multimodal
            print(f"[Reporte Lab] Generando figura multimodal paper para {os.path.basename(vocal_folder)}...")
            gen_path = generar_figura_paper_multimodal(vocal_folder)
            if os.path.exists(gen_path):
                imgs = [gen_path]
        except Exception as err:
            print(f"[Aviso] No se pudo autogenerar figura paper para {vocal_folder}: {err}")
    
    if imgs:
        # Reemplazar barras invertidas por normales para LaTeX
        return imgs[0].replace('\\', '/')
    return "example-image"

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
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isdir(config_input_str):
            base_dir = config_input_str
            fecha = os.path.basename(base_dir)
        else:
            fecha = config_input_str
            base_dir = os.path.join(repo_root, "EMG_desarrollo", "base_de_datos_electrodos", fecha)
            
        # Extraer metadatos automáticos desde la primera toma encontrada
        canales = {"0": "Canal 0", "1": "Canal 1", "2": "Canal 2"}
        sujeto = "Sujeto"
        for d in os.listdir(base_dir) if os.path.isdir(base_dir) else []:
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
            "electrodos_nota": f"Registro de superficie sEMG submáximal ({sujeto})",
            "canales": canales,
            "musculos_nota": f"Canal 0: {canales['0']}, Canal 1: {canales['1']}, Canal 2: {canales['2']}",
            "secuencia": "Secuencia periódica de fonación vocálica guiada por metrónomo a 30 BPM",
            "notas": "Reporte técnico integral compilado de forma automatizada por el sistema."
        }
        
    base_dir = os.path.abspath(base_dir)
    
    tex_content = LATEX_TEMPLATE
    tex_content = tex_content.replace("<<FECHA>>", fecha)
    tex_content = tex_content.replace("<<BATERIAS>>", config.get("baterias", "N/A"))
    tex_content = tex_content.replace("<<TIERRA>>", config.get("tierra", "N/A"))
    tex_content = tex_content.replace("<<ELECTRODOS_NOTA>>", config.get("electrodos_nota", ""))
    
    canales = config.get("canales", {})
    tex_content = tex_content.replace("<<CANAL_0>>", canales.get("0", "N/A"))
    tex_content = tex_content.replace("<<CANAL_1>>", canales.get("1", "N/A"))
    tex_content = tex_content.replace("<<CANAL_2>>", canales.get("2", "N/A"))
    tex_content = tex_content.replace("<<MUSCULOS_NOTA>>", config.get("musculos_nota", ""))
    
    tex_content = tex_content.replace("<<SECUENCIA>>", config.get("secuencia", ""))
    tex_content = tex_content.replace("<<NOTAS>>", config.get("notas", ""))
    
    # Remplazar imágenes
    vocales = ["a", "e", "i", "o", "u"]
    for v in vocales:
        v_upper = v.upper()
        tex_content = tex_content.replace(f"<<IMG_PATRON_{v_upper}>>", find_image(base_dir, v, "patron"))
        tex_content = tex_content.replace(f"<<IMG_PAPER_{v_upper}>>", find_image(base_dir, v, "paper"))
        tex_content = tex_content.replace(f"<<IMG_CALIB_{v_upper}>>", find_image(base_dir, v, "calib"))
        
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
        print(f"PDF generado exitosamente: {os.path.join(out_dir, report_name + '.pdf')}")
    except subprocess.CalledProcessError as e:
        print("Error al compilar el PDF. Verifica si LaTeX está instalado correctamente y los paths a las imágenes no tienen caracteres raros.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de reportes en LaTeX para experimentos EMG")
    parser.add_argument("config", nargs="?", default="2026-09-23", help="Ruta al archivo JSON de configuración o fecha de sesión (ej. 2026-09-23)")
    parser.add_argument("--sesion", "--fecha", dest="sesion", help="Fecha o carpeta de la sesión (ej. 2026-09-23)")
    args = parser.parse_args()
    
    target = args.sesion if args.sesion else args.config
    generar_reporte(target)
