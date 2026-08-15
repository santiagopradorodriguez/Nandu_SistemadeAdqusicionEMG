import re

with open("sections/01_hardware.tex", "r") as f:
    content = f.read()

# Define the old block to be replaced
old_block_pattern = r"\\subsubsection\{Comparación de ruidos iniciales en los últimos experimentos\}.*?\\textbf\{cables trenzados y mallados con conexión a GND\} para el sistema de adquisición\."

new_block = r"""\subsubsection{Comparación de ruidos iniciales en los últimos experimentos}

El objetivo de esta sección final de análisis es estudiar en detalle qué cable presentó el nivel de ruido más bajo, según los experimentos que se realizaron.

Para el Experimento 0 y 1, tenemos los siguientes gráficos, correspondientes ambos a un cable trenzado mallado.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image179.png}\\[2ex]
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image549.png}
\caption{Ruidos iniciales para el Experimento 0.}
\label{fig:ruido_exp0}
\end{figure}

Lo que estudiamos como \enquote{Ruido Inter-pulso}, sería la amplitud de la señal entre pulsos. Puede dar valores raros, se puede ver en los datos crudos. Lo asociamos a qué \enquote{tan bien} realizamos el movimiento. Es una señal muscular entre activación, se puede tener un espasmo intermedio. Que den todos los valores coincidentes, habla de que se hicieron bien las mediciones.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image458.png}
\caption{Ruidos iniciales para el Experimento 1.}
\label{fig:ruido_exp1}
\end{figure}

La diferencia en el E0 tiene que ver seguro con que no apoyamos la mano en la compu y no matamos la señal de \qty{50}{\hertz}. Apoyamos la mano en la compu porque no queremos lidiar con ruidos raros, solo con ver qué cable elegimos.

Vamos con la comparación del Experimento 2:

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image129.png}\\[2ex]
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image111.png}
\caption{Ruidos iniciales para el Experimento 2. Cable Trenzado Mallado (arriba) vs. Cable Mallado sin trenzar (abajo).}
\label{fig:ruido_exp2}
\end{figure}

Este fue el experimento en que la última medición (la 4) se desoldó justo después de tomar la medición, por lo tanto, es probable que esta medición no aporte mucho a la cuestión. Pero lo que se ve, es que el sin trenzar dio mejor.

\clearpage

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image442.png}\\[2ex]
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image302.png}
\caption{Ruidos iniciales para el Experimento 3 (con sujeto alternativo). Cable Trenzado Mallado (arriba) vs. Cable Mallado sin trenzar (abajo).}
\label{fig:ruido_exp3}
\end{figure}

Este experimento se hizo sobre Santi. Acá sí se podría decir que el trenzado redujo el ruido. Principalmente en la toma 4.

Vamos con el Experimento 4, quizás el más revelador:

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image272.png}
\caption{Comparación de ruidos iniciales en el Experimento 4. Cable Trenzado Mallado vs. None.}
\label{fig:ruido_exp4}
\end{figure}

Vamos a fijarnos solo en los sin Notch. Como se ve, la primera toma del trenzado aumentó el ruido, la segunda lo redujo y le \enquote{ganó} al cable sin nada.

Por ahora, vamos con cable trenzado mallado. Si bien no obtuvimos resultados absolutamente contundentes en cuanto a si es mejor el cable mallado sin trenzar o el trenzado en todos los escenarios, sabemos de entrada (y confirmamos con estos indicios empíricos) que debería ser mejor el \textbf{cables trenzados y mallados con conexión a GND} para el sistema de adquisición."""

if re.search(old_block_pattern, content, flags=re.DOTALL):
    new_content = re.sub(old_block_pattern, lambda m: new_block, content, flags=re.DOTALL)
    with open("sections/01_hardware.tex", "w") as f:
        f.write(new_content)
    print("Replaced successfully.")
else:
    print("Pattern not found!")

