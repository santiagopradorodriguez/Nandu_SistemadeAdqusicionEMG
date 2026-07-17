---
name: formato-latex-siunitx
description: Reglas para formatear texto en código LaTeX utilizando siunitx y booktabs.
---
# Skill: Formato LaTeX Especializado
Tu objetivo es traducir texto redactado a código fuente LaTeX listo para ser insertado.
**Reglas Estrictas de Formato:**
1. **Sin Preámbulo:** El código resultante será incluido con un comando `\input{}`. NO incluyas `\documentclass`, `\begin{document}`, ni paquetes. Solo entrega el
cuerpo del texto.
2. **Magnitudes y Unidades (siunitx):** Es de uso obligatorio el paquete `siunitx` para CUALQUIER valor numérico con unidades (ej: `\qty{5}{\volt}`,
`\SI{10}{\kilo\ohm}`). No escribas unidades en modo matemático convencional.
3. **Tablas (booktabs):** Todas las tablas (como la de componentes) deben estar maquetadas con `booktabs` usando exclusivamente `\toprule`, `\midrule` y `\bottomrule`.
Queda prohibido el uso de líneas verticales (`|`).
4. **Imágenes (figure):** Reemplaza cualquier marcador de tipo `[INSERTAR IMAGEN: ...]` por un entorno figure de LaTeX estructurado de la siguiente forma:
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{rutas/pendientes.png}
\caption{Descripción extraída del marcador}
\label{fig:etiqueta_descriptiva}
\end{figure}
