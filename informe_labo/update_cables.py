import re

with open("sections/01_hardware.tex", "r") as f:
    content = f.read()

# Find the start of the Cables subsection
match = re.search(r"\\subsection\{Cables\}", content)
if match:
    base_content = content[:match.start()]
else:
    base_content = content

cables_section = r"""\clearpage

\subsection{Cables}

\subsubsection{Cable de referencia original}

En las primeras iteraciones para las pruebas del dispositivo, se construyó un cable de referencia empleando una ficha plug metálica (de \qty{6}{\milli\meter}) y una ficha mini-canon con dos botones a presión, correspondientes a los electrodos comerciales. Estos botones a presión metálicos tienen un diámetro de \qty{11}{\milli\meter}.

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image136.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image171.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image433.jpg}
\end{subfigure}
\caption{Construcción de los primeros cables con botones a presión de \qty{11}{\milli\meter} y conectores plug.}
\label{fig:cables_iniciales}
\end{figure}

\clearpage

\subsubsection{Experimentación y selección de cables}

Durante varias semanas se llevó a cabo un proceso de prueba y medición caótico para determinar qué tipo de cables y electrodos utilizar (mallados, trenzados, sin trenzar, conectados a masa, etc.), y cómo estos afectaban la Relación Señal a Ruido (SNR). Si bien esta fase fue desordenada, se rescataron importantes observaciones empíricas:

\begin{itemize}
    \item \textbf{Desgaste de los broches metálicos:} Se descubrió que la presión necesaria para abrochar los cables dependía de un \enquote{resorte} interno. En conectores nuevos, la separación entre resortes era de \qty{2.7}{\milli\meter}, mientras que en los usados se ensanchaba a \qty{3}{\milli\meter}. Esta diferencia provocaba que, al aplicar fuerza para conectar los broches sobre el músculo, el gel conductivo del electrodo se desplazara, arruinando la medición temporalmente al introducir ruido por mal contacto.
    \item \textbf{Ruido de línea y configuración de la placa NIDAQ:} Se observó que el sistema presentaba un ruido de \qty{50}{\hertz} persistente. Tras sospechar del hardware, se descubrió que la NIDAQ estaba configurada por defecto en modo diferencial. Cambiar la configuración de lectura por software al modo \enquote{RSE} (Referenced Single-Ended) permitió referenciar las señales correctamente a la tierra del circuito, aunque no erradicó el ruido por completo.
    \item \textbf{Descarga a tierra y ficha T:} Durante el desarrollo, se observó que al tocar la computadora con la mano de la referencia (o al desconectar la PC de la fuente), el ruido de \qty{50}{\hertz} disminuía notablemente. También se propuso utilizar una ficha T en la entrada de la NIDAQ, confirmando la vulnerabilidad del sistema ante bucles de masa y el ruido de línea.
    \item \textbf{Filtros Notch y artefactos de Gibbs:} Se debatió extensamente el uso de filtros Notch para eliminar los \qty{50}{\hertz}, versus el uso de un simple filtro pasabanda. La literatura científica advertía que los filtros Notch con bandas de rechazo muy angostas pueden inducir el \enquote{fenómeno de Gibbs}, generando distorsiones y oscilaciones artificiales que alteran la morfología real del pico muscular en el dominio del tiempo.
\end{itemize}

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image308.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[height=5cm]{insumos_ia/cuaderno_de_labo/images/image138.jpg}
\end{subfigure}
\caption{Izquierda: distancia entre los resortes internos (\enquote{r}) del broche a presión. Derecha: Electrodos de la marca Cardinal Health empleados para las pruebas.}
\label{fig:tips_cables}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[height=4cm]{insumos_ia/cuaderno_de_labo/images/image456.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[height=4cm]{insumos_ia/cuaderno_de_labo/images/image422.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.3\textwidth}
    \centering
    \includegraphics[height=4cm]{insumos_ia/cuaderno_de_labo/images/image504.jpg}
\end{subfigure}
\caption{Imágenes de la fase exploratoria. Electrodo de referencia, conexión de cables mallados al paciente y setup completo hacia la NIDAQ.}
\label{fig:caotico_setup}
\end{figure}

A partir del análisis de datos de esta primera etapa, se discutió arduamente sobre el uso del filtro Notch. Debido a desconfianzas, se separó el análisis de la SNR evaluando los datos con y sin dicho filtro:

\textbf{SIN filtro NOTCH:}

Según las mediciones, lo primero que notamos es que las mejores SNR se dieron antes de un intervalo de descanso, siendo la destacada la del cable mallado \textit{sin} conectar a tierra. Sin embargo, en la inspección visual (y recordando las experiencias en vivo) parecía haber problemas con la adhesión de los electrodos que ensuciaban los datos. A continuación, la mejor y peor medición según nuestro criterio para este análisis sin filtro Notch:

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image299.png}
\caption{Cable mallado sin tierra toma 2 (la mejor medición sin Notch).}
\label{fig:mejor_sin_notch}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image209.png}
\caption{Cable mallado con tierra toma 2 (la peor medición sin Notch).}
\label{fig:peor_sin_notch}
\end{figure}

\textbf{CON filtro NOTCH:}

Según nuestro criterio de que un SNR $\geq$ 4 califica como buena señal, en esta oportunidad 4 de las mediciones superan el valor. Más allá de eso, el único cambio significativo con el filtro Notch es que ahora la medición del \textit{cable sin mallar} es la que gana, pero el análisis en profundidad seguía siendo poco claro. 

A continuación se observan los pulsos individuales procesados sin filtro Notch y luego con filtro Notch, para contrastarlos visualmente:

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image523.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image312.png}
\end{subfigure}
\caption{Sin filtro Notch: Cable sin mallar toma 1 (izq) vs. Cable mallado sin tierra toma 2 (der). A simple vista, la medición de la derecha tuvo mejor SNR.}
\label{fig:comparativa_sin_notch}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image104.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image311.png}
\end{subfigure}
\caption{Con filtro Notch: Cable sin mallar toma 1 (izq) vs. Cable mallado sin tierra toma 2 (der). Aquí pareciera que la medición de la derecha dio mejor, a diferencia del análisis numérico. Esto se debió a que el ruido se tomaba al principio y en el cable sin mallar (izq) el ruido aumentó durante los pulsos falseando el SNR total.}
\label{fig:comparativa_con_notch}
\end{figure}

\clearpage

\subsubsection{Protocolo definitivo para la determinación de cables}

Tras organizar los criterios de medición para no confiar ciegamente en datos caóticos, se diseñó una batería de experimentos formales enfocados en caracterizar el ruido y la pérdida de señal midiendo la SNR y su evolución en el tiempo. A partir de aquí, \textbf{todas las mallas de los cables fueron conectadas a GND}. El análisis definió variables como el \enquote{Ruido basal} (en reposo), el \enquote{Ruido Inter-pulso}, la \enquote{Amplitud Máxima} y la evolución del \enquote{SNR promedio acumulado}. 

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.18\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image379.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.18\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image364.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.18\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image18.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.18\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image71.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.18\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image325.jpg}
\end{subfigure}
\caption{Set experimental utilizado en las pruebas formales para evaluar exhaustivamente el rendimiento de los cables.}
\label{fig:set_experimental_cables}
\end{figure}

\textbf{Experimento 0 (Temporalidad inicial):} El objetivo era evaluar si la primera medición siempre resultaba muy alta y decaía. Se realizaron 5 mediciones de 20 pulsos en intervalos de 1 minuto usando el cable Trenzado Mallado con GND. En esta primera instancia, hubo un error de medición: no se apoyó la mano sobre la PC, por lo que hubo mucho ruido de línea.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image140.png}
\caption{Archivos de referencia generados para el Experimento 0.}
\label{fig:exp0_archivos}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image267.png}
\caption{Overlay de pulsos promedio de cada medición para el Experimento 0.}
\label{fig:exp0_overlay}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image508.png}
\caption{Gráfico de barras de las amplitudes en orden cronológico (Experimento 0).}
\label{fig:exp0_barras}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image538.png}\\
\vspace{2mm}
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image207.png}
\caption{Evolución del Ruido y el SNR a lo largo de la sesión para el Experimento 0.}
\label{fig:exp0_evolucion}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image322.png}
\caption{Tabla de resultados del Experimento 0, ordenados según SNR. Fueron aproximadamente 7 minutos de medición: la amplitud promedio decayó, pero el SNR se mantuvo estable ya que el ruido también disminuyó.}
\label{fig:exp0_tabla_snr}
\end{figure}

\clearpage

\textbf{Experimento 1 (Temporalidad rigurosa):} Se evaluó cómo se reproducía la medición con el paso del tiempo utilizando el mismo cable Trenzado Mallado GND, midiendo a $t = 0, 5, 10, 15$ minutos (con descansos intercalados), tomando 40 pulsos por iteración.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image282.png}
\caption{Archivos de referencia generados para el Experimento 1.}
\label{fig:exp1_archivos}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image397.png}
\caption{Overlay de pulsos promedio de cada medición para el Experimento 1.}
\label{fig:exp1_overlay}
\end{figure}

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image264.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image409.png}
\end{subfigure}
\caption{Gráficos de barras de amplitud y evolución de ruidos para el Experimento 1.}
\label{fig:exp1_graficos}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image36.png}
\caption{Evolución del SNR en el tiempo para el Experimento 1 (15 minutos totales).}
\label{fig:exp1_snr_evolucion}
\end{figure}

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{insumos_ia/cuaderno_de_labo/images/image356.png}
\caption{Tabla de SNR ordenado para el Experimento 1. Las amplitudes decaen temporalmente tal como se observaba anteriormente.}
\label{fig:exp1_tabla}
\end{figure}

\clearpage

\textbf{Experimentos 2, 3 y 4 (Comparación directa):} Posteriormente, se realizaron los experimentos comparando "Trenzado Mallado GND" contra "Mallado GND" (Experimento 2), repitiendo lo mismo con un sujeto alternativo (Experimento 3, debido a que en el 2 se desoldó un pin del cable trenzado) y luego contra un cable "Sin trenzar ni mallar - None" (Experimento 4). Las imágenes del setup experimental con el nuevo sujeto se muestran a continuación:

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.4\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image524.jpg}
\end{subfigure}\hfill
\begin{subfigure}{0.4\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image328.jpg}
\end{subfigure}
\caption{Imágenes del set experimental utilizando al sujeto alternativo para los experimentos 3 y 4.}
\label{fig:exp3_setup}
\end{figure}

\subsubsection{Comparación de ruidos iniciales en los últimos experimentos}

El objetivo de esta sección final de análisis es estudiar en detalle qué cable presentó el nivel de ruido más bajo. Para los Experimentos 0 y 1 (ambos realizados con cable Trenzado Mallado), se estudió el \enquote{Ruido Inter-pulso} (la amplitud de la señal entre pulsos, asociada a la inestabilidad muscular o pequeños espasmos). 

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image179.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image549.png}
\end{subfigure}
\vspace{2mm}
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image458.png}
\end{subfigure}
\caption{Ruidos iniciales para el Experimento 0 (arriba) y Experimento 1 (abajo). La clara diferencia en el Experimento 0 provino de no haber apoyado la mano en la computadora para suprimir el ruido de línea.}
\label{fig:ruido_exp01}
\end{figure}

Al evaluar la comparación del Experimento 2 (Trenzado Mallado vs. Mallado sin trenzar), las mediciones apuntaron a que el cable \textit{sin} trenzar se comportaba mejor, aunque, como se documentó, una de las soldaduras temporales falló, invalidando la última toma.

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image129.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image111.png}
\end{subfigure}
\caption{Ruidos iniciales para el Experimento 2. Se observa que el cable \textit{sin} trenzar dio un mejor resultado parcial.}
\label{fig:ruido_exp2}
\end{figure}

Se repitió la prueba bajo el título de Experimento 3 con el segundo sujeto. En este escenario, sí se comprobó empíricamente que el trenzado ayudaba a reducir el ruido (particularmente notorio en la toma 4).

\begin{figure}[htbp]
\centering
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image442.png}
\end{subfigure}\hfill
\begin{subfigure}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{insumos_ia/cuaderno_de_labo/images/image302.png}
\end{subfigure}
\caption{Ruidos iniciales para el Experimento 3. Aquí el cable trenzado redujo el nivel de ruido eficazmente.}
\label{fig:ruido_exp3}
\end{figure}

Finalmente, el Experimento 4, que resultó ser el más revelador de la sesión, enfrentó el cable Trenzado Mallado contra un cable ordinario (None) que no poseía ni trenzado ni malla. 

\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{insumos_ia/cuaderno_de_labo/images/image272.png}
\caption{Comparación de ruidos iniciales en el Experimento 4. Se destacan las señales sin filtro Notch.}
\label{fig:ruido_exp4}
\end{figure}

Al observar detalladamente las señales sin filtro Notch, se denota que si bien la primera toma del cable trenzado inyectó sorpresivamente más ruido, en la segunda iteración lo redujo drásticamente, \enquote{ganándole} en desempeño al cable común sin nada. Por lo tanto, si bien los resultados no fueron abrumadoramente contundentes en todos los escenarios posibles, basándose en la teoría electromagnética y los indicios rescatados en estos gráficos, se determinó definitivamente fabricar e implementar \textbf{cables trenzados y mallados con conexión a GND} para el sistema de adquisición.

"""

with open("sections/01_hardware.tex", "w") as f:
    f.write(base_content + cables_section)

print("Done replacing.")
