#!/usr/bin/env python3
"""
Genera el mapa conceptual de las 3 coordenadas articulatorias:
Cinematica interna (Trevisan 2013) vs. Generadores sEMG (Nandu LSD)
y el quiebre de la biyeccion lengua-sonrisa.
Tema: Blanco / Editorial impreso para Tesis en LaTeX.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.style.use('default')
fig, ax = plt.subplots(figsize=(13.5, 7.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Fondo blanco limpio editorial
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# Titulo principal arriba
ax.text(50, 96, "MAPA CONCEPTUAL: COORDENADAS ARTICULATORIAS Y OBSERVABILIDAD sEMG",
        color='#0f172a', fontsize=13.5, fontweight='bold', ha='center', va='center')
ax.text(50, 92.5, "De la cinematica interna invasiva (Trevisan 2013) al sensado mioelectrico periferico (Nandu LSD)",
        color='#475569', fontsize=10, ha='center', va='center')

# Funcion para dibujar cajas redondeadas en tema blanco
def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", items=None, title_color='#0f172a'):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.8,rounding_size=1.5",
                                 facecolor=bg_color, edgecolor=border_color, linewidth=1.5)
    ax.add_patch(box)
    curr_y = y + h - 2.5
    ax.text(x + w/2, curr_y, title, color=title_color, fontsize=10.5, fontweight='bold', ha='center', va='top')
    if subtitle:
        curr_y -= 3.2
        ax.text(x + w/2, curr_y, subtitle, color='#64748b', fontsize=8.5, ha='center', va='top', style='italic')
    if items:
        curr_y -= 3.5
        for it in items:
            ax.text(x + 2.0, curr_y, it, color='#1e293b', fontsize=8.2, ha='left', va='top')
            curr_y -= 3.0

# Encabezados de Columna
ax.text(17, 86, "1. GRADO DE LIBERTAD FISICO\n(Tracto Vocal Humano)", 
        color='#0369a1', fontsize=11, fontweight='bold', ha='center', va='center', multialignment='center')

ax.text(50, 86, "2. CINEMATICA INTERNA (HALL)\n(Assaneo, Trevisan & Mindlin 2013)", 
        color='#6d28d9', fontsize=11, fontweight='bold', ha='center', va='center', multialignment='center')

ax.text(83, 86, "3. ACTUADORES sEMG\n(Sistema Nandu LSD)", 
        color='#047857', fontsize=11, fontweight='bold', ha='center', va='center', multialignment='center')

# --- FILA 1: APERTURA MANDIBULAR ---
# GD 1
draw_box(4, 58, 26, 21, '#f0f9ff', '#0284c7', "Apertura Mandibular", "Control del 1er Formante (F1)",
         ["• Descenso vertical del maxilar", "• Depresion de la mandibula", "• Clave acustica de la vocal /a/"],
         title_color='#0369a1')

# Hall 1
draw_box(37, 58, 26, 21, '#f5f3ff', '#7c3aed', "Sensor Jaw (Mandibula)", "Molares superiores e inferiores",
         ["• Distancia interdental molar", "• 0 = Abierta (deprimida en /a/)", "• 1 = Ocluida / Cerrada en /i/, /u/"],
         title_color='#6d28d9')

# sEMG 1
draw_box(70, 58, 26, 21, '#ecfdf5', '#059669', "Vientre Ant. Digastrico", "Region submentoniana (CH0)",
         ["• Motor primario de descenso", "• Relacion BIYECTIVA y causal", "• Disparo masivo en /a/"],
         title_color='#047857')

# Flechas Fila 1
ax.annotate("", xy=(37, 68.5), xytext=(30, 68.5),
            arrowprops=dict(arrowstyle="->", color='#0284c7', lw=2.0))
ax.annotate("", xy=(70, 68.5), xytext=(63, 68.5),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=2.0))
ax.text(66.5, 70.8, "Causal", color='#047857', fontsize=8.5, fontweight='bold', ha='center')

# --- FILA 2: CONSTRICCION / PROTRUSION LABIAL ---
# GD 2
draw_box(4, 31, 26, 22, '#f0f9ff', '#0284c7', "Constriccion Labial", "Elongacion acustica y area labial",
         ["• Variacion de longitud ΔL", "• Reduccion del area Ab", "• Abatimiento simultaneo F1 y F2"],
         title_color='#0369a1')

# Hall 2
draw_box(37, 31, 26, 22, '#f5f3ff', '#7c3aed', "Sensor Lips (Labios)", "Labio inferior vs. Incisivos inf.",
         ["• Mide DESPEGUE vertical", "• 1 = Labio retraido / abierto", "• 0 = Pasivo / cerrado en /o/, /u/"],
         title_color='#6d28d9')

# sEMG 2
draw_box(70, 31, 26, 22, '#ecfdf5', '#059669', "Orbicularis Oris", "Esfinter perioral labial (CH2)",
         ["• Esfinter concentrico obligatorio", "• Relacion BIYECTIVA y causal", "• Disparo selectivo en /o/ y /u/"],
         title_color='#047857')

# Flechas Fila 2
ax.annotate("", xy=(37, 42), xytext=(30, 42),
            arrowprops=dict(arrowstyle="->", color='#0284c7', lw=2.0))
ax.annotate("", xy=(70, 42), xytext=(63, 42),
            arrowprops=dict(arrowstyle="->", color='#059669', lw=2.0))
ax.text(66.5, 44.3, "Causal", color='#047857', fontsize=8.5, fontweight='bold', ha='center')

# --- FILA 3: POSICION LINGUAL Y EL QUIEBRE DE LA BIYECCION ---
# GD 3
draw_box(4, 4, 26, 22, '#f0f9ff', '#0284c7', "Constriccion Lingual", "Avance anteroposterior del dorso",
         ["• Elevacion lingual palatal", "• Modula el 2do Formante (F2)", "• Maxima separacion en /i/ y /e/"],
         title_color='#0369a1')

# Hall 3
draw_box(37, 4, 26, 22, '#f5f3ff', '#7c3aed', "Sensor Tongue (Lengua)", "Dorso de la lengua vs. Paladar",
         ["• Iman a 1.5 cm del apice", "• Medicion INTRAORAL directa", "• 1 = Lengua alta palatal (/i/, /e/)"],
         title_color='#6d28d9')

# sEMG 3 (Alerta: Quiebre de Biyeccion - caja rojiza suave)
draw_box(70, 4, 26, 22, '#fff1f2', '#e11d48', "Cigomatico / Mejilla", "Sinergia de 'Sonrisa' (CH1)",
         ["• Retraccion comisural facultativa", "• NO ES CAUSA OBLIGATORIA", "• Quiebre: /i/ sin sonrisa colapsa", "• Equivalencia motora lingual"],
         title_color='#be123c')

# Flechas Fila 3
ax.annotate("", xy=(37, 15), xytext=(30, 15),
            arrowprops=dict(arrowstyle="->", color='#0284c7', lw=2.0))

# Flecha punteada y alerta de quiebre
ax.annotate("", xy=(70, 15), xytext=(63, 15),
            arrowprops=dict(arrowstyle="->", color='#e11d48', lw=2.0, ls='--'))
ax.text(66.5, 17.5, "NO biyectivo", color='#be123c', fontsize=8.5, fontweight='bold', ha='center')
ax.text(66.5, 13.0, "(Sinergia)", color='#e11d48', fontsize=8, ha='center', style='italic')

plt.tight_layout()
out_path = "reportes_experimentos/tesis_modular_latex/tesis_licenciatura_modular/figuras/mapa_conceptual_coordenadas_trevisan_vs_emg.png"
plt.savefig(out_path, dpi=300, facecolor='#ffffff', edgecolor='none')
plt.close()
print("Regenerado corregido con exito:", out_path)
