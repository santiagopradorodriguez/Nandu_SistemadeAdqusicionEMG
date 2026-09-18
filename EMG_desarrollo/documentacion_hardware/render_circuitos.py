import os
import subprocess

DIR_HW = os.path.dirname(os.path.abspath(__file__))
DIR_IMG = os.path.join(DIR_HW, "imagenes")
os.makedirs(DIR_IMG, exist_ok=True)

# 1. Esquema Completo: Etapa de Alimentación y Protección Activa (Izquierda) + 3 Canales AD620 (Derecha)
TEX_COMPLETO = r"""\documentclass[border=10pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[siunitx]{circuitikz}
\usepackage{xcolor}

\begin{document}
\begin{circuitikz}[american, thick, font=\sffamily]

  % ====================================================
  % BLOQUE 1: ALIMENTACION Y PROTECCION ACTIVA (IZQUIERDA)
  % ====================================================
  \begin{scope}[xshift=-14cm, yshift=0cm]
    \draw[dashed, draw=blue!60, fill=blue!2, rounded corners] (-2.8, 9.2) rectangle (13.8, -9.2);
    \node[anchor=north west, font=\bfseries\color{blue!70!black}] at (-2.5, 8.8) {ETAPA DE ALIMENTACI\'ON Y PROTECCI\'ON ACTIVA $\pm 9\,\text{V}$};

    % Bornera BAT TBLOCK-M3
    \draw (-2.2, 5.0) rectangle (0.4, -5.0);
    \node[font=\bfseries] at (-0.9, 4.4) {BAT};
    \node[font=\scriptsize, anchor=west] at (-2.0, 3.4) {Pin 1: +9V};
    \node[font=\scriptsize, anchor=west] at (-2.0, 0.0) {Pin 2: GND};
    \node[font=\scriptsize, anchor=west] at (-2.0, -3.4) {Pin 3: -9V};
    \draw (0.4, 3.4) -- (1.6, 3.4);
    \draw (0.4, 0.0) -- (1.0, 0.0) node[ground]{};
    \draw (0.4, -3.4) -- (1.6, -3.4);

    % Bornera SWITCH TBLOCK-M4
    \draw (1.6, 5.0) rectangle (3.8, -5.0);
    \node[font=\bfseries] at (2.7, 4.4) {SWITCH};
    \node[font=\scriptsize] at (2.0, 3.4) {1}; \node[font=\scriptsize] at (3.4, 3.4) {2};
    \node[font=\scriptsize] at (2.0, -3.4) {3}; \node[font=\scriptsize] at (3.4, -3.4) {4};
    \draw[very thick] (2.2, 3.4) -- (3.2, 3.4);
    \draw[very thick] (2.2, -3.4) -- (3.2, -3.4);

    % Fusibles
    \draw (3.8, 3.4) to[fuse, *-*, l={\small $\text{FUSP} = 0.25\,\text{A}$}] (6.0, 3.4);
    \draw (3.8, -3.4) to[fuse, *-*, l_={\small $\text{FUSN} = 0.25\,\text{A}$}] (6.0, -3.4);

    % --- RIEL POSITIVO (+9V) ---
    \draw (6.0, 3.4) -- (7.2, 3.4) coordinate (s_pos)
          node[pmos, rotate=90, yscale=-1, anchor=S] (QP) {};
    \node[above=6pt, font=\small\bfseries] at (7.2, 3.7) {F9540N};
    \draw (QP.D) -- (13.0, 3.4) node[circle, fill, inner sep=2pt]{} 
          node[right, font=\bfseries\color{red!70!black}] {+9V};

    \draw (QP.G) -- (7.2, 2.0) coordinate (g_pos) -- (9.2, 2.0);
    \draw (8.2, 3.4) to[zD, *-, l_={\small $D_2$}] (8.2, 2.0) node[circ]{};
    \draw (9.2, 2.0) to[R, *-, l={\small $R_3 = 10\,\text{k}\Omega$}] (9.2, 0.0);

    % --- RIEL NEGATIVO (-9V) ---
    \draw (6.0, -3.4) -- (7.2, -3.4) coordinate (s_neg)
          node[nmos, rotate=-90, yscale=-1, anchor=S] (QN) {};
    \node[below=6pt, font=\small\bfseries] at (7.2, -3.7) {IRFZ44N};
    \draw (QN.D) -- (13.0, -3.4) node[circle, fill, inner sep=2pt]{} 
          node[right, font=\bfseries\color{blue!70!black}] {-9V};

    \draw (QN.G) -- (7.2, -2.0) coordinate (g_neg) -- (10.4, -2.0);
    \draw (8.2, -3.4) to[zD, *-, l={\small $D_1$}] (8.2, -2.0) node[circ]{};
    \draw (10.4, -2.0) to[R, *-, l={\small $R_1 = 10\,\text{k}\Omega$}] (10.4, 0.0);

    % Resistencias Bleeder
    \draw (11.8, 3.4) to[R, *-, l={\small $R_4 = 100\,\text{k}\Omega$}] (11.8, 0.0);
    \draw (11.8, -3.4) to[R, *-, l_={\small $R_2 = 100\,\text{k}\Omega$}] (11.8, 0.0);

    % Masa GND
    \draw (6.8, 0.0) node[ground]{} -- (13.0, 0.0) node[circle, fill, inner sep=2pt]{} 
          node[right, font=\bfseries] {GND};
    \draw (9.2, 0.0) node[circ]{};
    \draw (10.4, 0.0) node[circ]{};
    \draw (11.8, 0.0) node[circ]{};
  \end{scope}

  % ====================================================
  % BLOQUE 2: CANAL 1 (U1 - AD620)
  % ====================================================
  \begin{scope}[xshift=4cm, yshift=6.0cm]
    \draw[dashed, draw=blue!60, fill=blue!2, rounded corners] (-3.8, 2.7) rectangle (9.8, -2.7);
    \node[anchor=north west, font=\bfseries\color{blue!70!black}] at (-3.5, 2.4) {CANAL 1: IN1 Y OUT1};

    % Bornera IN1
    \draw (-3.4, 1.3) rectangle (-2.0, -1.3);
    \node[font=\bfseries\scriptsize] at (-2.7, 0.9) {IN1};
    \node[font=\tiny] at (-3.1, 0.45) {1}; \draw (-2.0, 0.45) -- (-1.6, 0.45) node[ground]{};
    \node[font=\tiny] at (-3.1, 0.0) {2};  \draw (-2.0, 0.0) -- (-1.0, 0.0) -- (-1.0, 0.9) coordinate (in1_neg_start);
    \node[font=\tiny] at (-3.1, -0.45) {3}; \draw (-2.0, -0.45) -- (-1.0, -0.45) -- (-1.0, -0.9) coordinate (in1_pos_start);

    % Filtros de entrada
    \draw (in1_neg_start) to[C, l={\scriptsize $C_2 = 33\,\text{nF}$}] (0.8, 0.9) coordinate (junc1_neg) -- (3.0, 0.9);
    \draw (junc1_neg) to[R, *-, l_={\scriptsize $R_2 = 100\,\text{k}\Omega$}] (0.8, 1.9) node[ground, yscale=-1]{};

    \draw (in1_pos_start) to[C, l_={\scriptsize $C_1 = 33\,\text{nF}$}] (0.8, -0.9) coordinate (junc1_pos) -- (3.0, -0.9);
    \draw (junc1_pos) to[R, *-, l={\scriptsize $R_1 = 100\,\text{k}\Omega$}] (0.8, -1.9) node[ground]{};

    % AD620 (U1)
    \draw[thick] (3.0, 1.4) -- (5.4, 0.0) -- (3.0, -1.4) -- cycle;
    \node at (4.1, 0.0) {\small AD620};
    \node[below=4pt, font=\tiny] at (4.1, 0.0) {U1};
    \node at (3.3, 0.9) {\scriptsize \(-\)};
    \node at (3.3, -0.9) {\scriptsize \(+\)};

    % Resistencia de ganancia RG (Pin 1 y 8)
    \draw (3.0, 0.4) node[right=1pt, font=\tiny] {1} -- (2.2, 0.4) coordinate (rg1_top);
    \draw (3.0, -0.4) node[right=1pt, font=\tiny] {8} -- (2.2, -0.4) coordinate (rg1_bot);
    \draw (rg1_top) to[R, *-*, l_={\scriptsize $R_3 = 100\,\Omega$}] (rg1_bot);

    % Alimentación U1
    \draw (3.8, 0.9) -- ++(0, 0.7) node[above, font=\tiny\bfseries\color{red!70!black}] {+9V};
    \draw (3.8, -0.9) -- ++(0, -0.7) node[below, font=\tiny\bfseries\color{blue!70!black}] {-9V};
    \draw (4.6, -0.5) -- ++(0, -0.7) node[ground]{};

    % Salida y Bornera OUT1
    \draw (5.4, 0.0) -- (7.6, 0.0) coordinate (out1_wire);
    \draw (7.6, 1.0) rectangle (9.2, -1.0);
    \node[font=\bfseries\scriptsize] at (8.4, 0.6) {OUT1};
    \node[font=\tiny] at (8.0, 0.0) {2}; \draw (out1_wire) -- (7.6, 0.0);
    \node[font=\tiny] at (8.0, -0.5) {1}; \draw (9.2, -0.5) -- ++(0.3, 0) node[ground]{};
  \end{scope}

  % ====================================================
  % BLOQUE 3: CANAL 2 (U2 - AD620)
  % ====================================================
  \begin{scope}[xshift=4cm, yshift=0.0cm]
    \draw[dashed, draw=blue!60, fill=blue!2, rounded corners] (-3.8, 2.7) rectangle (9.8, -2.7);
    \node[anchor=north west, font=\bfseries\color{blue!70!black}] at (-3.5, 2.4) {CANAL 2: IN2 Y OUT2};

    % Bornera IN2
    \draw (-3.4, 1.3) rectangle (-2.0, -1.3);
    \node[font=\bfseries\scriptsize] at (-2.7, 0.9) {IN2};
    \node[font=\tiny] at (-3.1, 0.45) {1}; \draw (-2.0, 0.45) -- (-1.6, 0.45) node[ground]{};
    \node[font=\tiny] at (-3.1, 0.0) {2};  \draw (-2.0, 0.0) -- (-1.0, 0.0) -- (-1.0, 0.9) coordinate (in2_neg_start);
    \node[font=\tiny] at (-3.1, -0.45) {3}; \draw (-2.0, -0.45) -- (-1.0, -0.45) -- (-1.0, -0.9) coordinate (in2_pos_start);

    % Filtros de entrada
    \draw (in2_neg_start) to[C, l={\scriptsize $C_4 = 33\,\text{nF}$}] (0.8, 0.9) coordinate (junc2_neg) -- (3.0, 0.9);
    \draw (junc2_neg) to[R, *-, l_={\scriptsize $R_4 = 100\,\text{k}\Omega$}] (0.8, 1.9) node[ground, yscale=-1]{};

    \draw (in2_pos_start) to[C, l_={\scriptsize $C_3 = 33\,\text{nF}$}] (0.8, -0.9) coordinate (junc2_pos) -- (3.0, -0.9);
    \draw (junc2_pos) to[R, *-, l={\scriptsize $R_5 = 100\,\text{k}\Omega$}] (0.8, -1.9) node[ground]{};

    % AD620 (U2)
    \draw[thick] (3.0, 1.4) -- (5.4, 0.0) -- (3.0, -1.4) -- cycle;
    \node at (4.1, 0.0) {\small AD620};
    \node[below=4pt, font=\tiny] at (4.1, 0.0) {U2};
    \node at (3.3, 0.9) {\scriptsize \(-\)};
    \node at (3.3, -0.9) {\scriptsize \(+\)};

    % Resistencia de ganancia RG (Pin 1 y 8)
    \draw (3.0, 0.4) node[right=1pt, font=\tiny] {1} -- (2.2, 0.4) coordinate (rg2_top);
    \draw (3.0, -0.4) node[right=1pt, font=\tiny] {8} -- (2.2, -0.4) coordinate (rg2_bot);
    \draw (rg2_top) to[R, *-*, l_={\scriptsize $R_6 = 100\,\Omega$}] (rg2_bot);

    % Alimentación U2
    \draw (3.8, 0.9) -- ++(0, 0.7) node[above, font=\tiny\bfseries\color{red!70!black}] {+9V};
    \draw (3.8, -0.9) -- ++(0, -0.7) node[below, font=\tiny\bfseries\color{blue!70!black}] {-9V};
    \draw (4.6, -0.5) -- ++(0, -0.7) node[ground]{};

    % Salida y Bornera OUT2
    \draw (5.4, 0.0) -- (7.6, 0.0) coordinate (out2_wire);
    \draw (7.6, 1.0) rectangle (9.2, -1.0);
    \node[font=\bfseries\scriptsize] at (8.4, 0.6) {OUT2};
    \node[font=\tiny] at (8.0, 0.0) {2}; \draw (out2_wire) -- (7.6, 0.0);
    \node[font=\tiny] at (8.0, -0.5) {1}; \draw (9.2, -0.5) -- ++(0.3, 0) node[ground]{};
  \end{scope}

  % ====================================================
  % BLOQUE 4: CANAL 3 (U3 - AD620)
  % ====================================================
  \begin{scope}[xshift=4cm, yshift=-6.0cm]
    \draw[dashed, draw=blue!60, fill=blue!2, rounded corners] (-3.8, 2.7) rectangle (9.8, -2.7);
    \node[anchor=north west, font=\bfseries\color{blue!70!black}] at (-3.5, 2.4) {CANAL 3: IN3GND Y OUT3};

    % Bornera IN3GND
    \draw (-3.4, 1.3) rectangle (-2.0, -1.3);
    \node[font=\bfseries\scriptsize] at (-2.7, 0.9) {IN3GND};
    \node[font=\tiny] at (-3.1, 0.45) {1}; \draw (-2.0, 0.45) -- (-1.4, 0.45) node[ground]{} node[right=2pt, font=\tiny\bfseries] {REF};
    \node[font=\tiny] at (-3.1, 0.0) {2};  \draw (-2.0, 0.0) -- (-1.0, 0.0) -- (-1.0, 0.9) coordinate (in3_neg_start);
    \node[font=\tiny] at (-3.1, -0.45) {3}; \draw (-2.0, -0.45) -- (-1.0, -0.45) -- (-1.0, -0.9) coordinate (in3_pos_start);

    % Filtros de entrada
    \draw (in3_neg_start) to[C, l={\scriptsize $C_6 = 33\,\text{nF}$}] (0.8, 0.9) coordinate (junc3_neg) -- (3.0, 0.9);
    \draw (junc3_neg) to[R, *-, l_={\scriptsize $R_7 = 100\,\text{k}\Omega$}] (0.8, 1.9) node[ground, yscale=-1]{};

    \draw (in3_pos_start) to[C, l_={\scriptsize $C_5 = 33\,\text{nF}$}] (0.8, -0.9) coordinate (junc3_pos) -- (3.0, -0.9);
    \draw (junc3_pos) to[R, *-, l={\scriptsize $R_8 = 100\,\text{k}\Omega$}] (0.8, -1.9) node[ground]{};

    % AD620 (U3)
    \draw[thick] (3.0, 1.4) -- (5.4, 0.0) -- (3.0, -1.4) -- cycle;
    \node at (4.1, 0.0) {\small AD620};
    \node[below=4pt, font=\tiny] at (4.1, 0.0) {U3};
    \node at (3.3, 0.9) {\scriptsize \(-\)};
    \node at (3.3, -0.9) {\scriptsize \(+\)};

    % Resistencia de ganancia RG (Pin 1 y 8)
    \draw (3.0, 0.4) node[right=1pt, font=\tiny] {1} -- (2.2, 0.4) coordinate (rg3_top);
    \draw (3.0, -0.4) node[right=1pt, font=\tiny] {8} -- (2.2, -0.4) coordinate (rg3_bot);
    \draw (rg3_top) to[R, *-*, l_={\scriptsize $R_9 = 100\,\Omega$}] (rg3_bot);

    % Alimentación U3
    \draw (3.8, 0.9) -- ++(0, 0.7) node[above, font=\tiny\bfseries\color{red!70!black}] {+9V};
    \draw (3.8, -0.9) -- ++(0, -0.7) node[below, font=\tiny\bfseries\color{blue!70!black}] {-9V};
    \draw (4.6, -0.5) -- ++(0, -0.7) node[ground]{};

    % Salida y Bornera OUT3
    \draw (5.4, 0.0) -- (7.6, 0.0) coordinate (out3_wire);
    \draw (7.6, 1.0) rectangle (9.2, -1.0);
    \node[font=\bfseries\scriptsize] at (8.4, 0.6) {OUT3};
    \node[font=\tiny] at (8.0, 0.0) {2}; \draw (out3_wire) -- (7.6, 0.0);
    \node[font=\tiny] at (8.0, -0.5) {1}; \draw (9.2, -0.5) -- ++(0.3, 0) node[ground]{};
  \end{scope}

\end{circuitikz}
\end{document}
"""

# 2. Esquema Mejora: Canal con Desacoplo AC en Lazo de Ganancia (Sin superposiciones, sin caja naranja)
TEX_MEJORA = r"""\documentclass[border=10pt]{standalone}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage[siunitx]{circuitikz}
\usepackage{xcolor}

\begin{document}
\begin{circuitikz}[american, thick, font=\sffamily]

    % Marco delimitador con amplio margen vertical
    \draw[dashed, draw=green!60!black, fill=green!2, rounded corners] (-6.8, 4.4) rectangle (4.8, -3.8);
    \node[anchor=north west, font=\bfseries\color{green!50!black}] at (-6.5, 4.0) {CANAL CON MODIFICACI\'ON PROPUESTA: DESACOPLO AC EN LAZO DE GANANCIA};

    % AD620
    \draw[thick] (-0.6, 1.8) -- (2.6, 0) -- (-0.6, -1.8) -- cycle;
    \node at (0.9, 0) {\Large AD620};
    \node at (-0.2, 1.2) {\Large \(-\)};
    \node at (-0.2, -1.2) {\Large \(+\)};

    % Pines de Alimentación (±9V) y Referencia
    \draw[-latex] (0.4, 1.2) -- ++(0, 0.8) node[above, font=\small\bfseries\color{red!70!black}] {+9V};
    \node[left=1pt, font=\scriptsize] at (0.4, 1.4) {7};

    \draw[-latex] (0.4, -1.2) -- ++(0, -0.8) node[below, font=\small\bfseries\color{blue!70!black}] {-9V};
    \node[left=1pt, font=\scriptsize] at (0.4, -1.4) {4};

    \draw (1.4, -0.7) -- ++(0, -0.9) node[ground]{};
    \node[right=2pt, font=\scriptsize] at (1.4, -0.7) {5};

    \draw (2.6, 0) -- ++(1.4, 0) node[ocirc]{} node[pos=0.6, above, font=\scriptsize] {6} node[right, font=\small\bfseries] {$V_{\text{out}}$};

    % Lazo de ganancia horizontal (RG = 100 Ohm y CG = 100 uF para fc ≈ 16 Hz)
    \draw (-0.6, 0.5) node[right=2pt, font=\scriptsize] {1} 
          to[R, l_={\small $R_G = 100\,\Omega$}, *-*] (-2.6, 0.5) coordinate (rg_left);
    \draw (-0.6, -0.5) node[right=2pt, font=\scriptsize] {8} 
          to[C, l={\small $C_G = 100\,\mu\text{F}$}, *-*] (-2.6, -0.5) coordinate (cg_left);
    \draw (rg_left) -- (cg_left);

    % Entradas directas acopladas en continua con alta impedancia (10 MOhm a GND)
    \draw (-0.6, 1.3) -- (-1.6, 1.3) -- (-4.4, 1.3) coordinate (juncA);
    \draw (juncA) to[R, l_={\small $10\text{ M}\Omega$}] ++(0, 1.1) node[ground, yscale=-1]{};
    \draw (juncA) -- ++(-1.4, 0) node[left, font=\small\bfseries] {\(V_{\mathrm{in}}^-\)};

    \draw (-0.6, -1.3) -- (-1.6, -1.3) -- (-4.4, -1.3) coordinate (juncB);
    \draw (juncB) to[R, l={\small $10\text{ M}\Omega$}] ++(0, -1.1) node[ground]{};
    \draw (juncB) -- ++(-1.4, 0) node[left, font=\small\bfseries] {\(V_{\mathrm{in}}^+\)};

\end{circuitikz}
\end{document}
"""

def compilar_y_convertir(base_name):
    tex_path = os.path.join(DIR_HW, f"{base_name}.tex")
    pdf_path = os.path.join(DIR_HW, f"{base_name}.pdf")
    png_path = os.path.join(DIR_IMG, f"{base_name}.png")

    print(f"[Compilación] Generando PDF para {base_name}...")
    res = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
        cwd=DIR_HW,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if res.returncode != 0:
        print(f"[ERROR] Falló pdflatex para {base_name}:")
        print(res.stdout[-1000:])
        return False

    print(f"[Render] Convirtiendo {base_name}.pdf a PNG (300 DPI)...")
    res_conv = subprocess.run(
        ["pdftoppm", "-png", "-r", "300", "-singlefile", f"{base_name}.pdf", os.path.join(DIR_IMG, base_name)],
        cwd=DIR_HW,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if res_conv.returncode == 0:
        print(f"[OK] Imagen guardada en: {png_path}")
        return True
    else:
        print(f"[ERROR] Falló pdftoppm para {base_name}: {res_conv.stderr}")
        return False

if __name__ == "__main__":
    print("=== RENDERIZADOR DE ESQUEMAS CIRCUITIKZ ===")
    ok1 = compilar_y_convertir("esquema_completo_circuitikz")
    ok2 = compilar_y_convertir("esquema_mejora_circuitikz")
    print(f"Resultado final: Completo={ok1}, Mejora={ok2}")
