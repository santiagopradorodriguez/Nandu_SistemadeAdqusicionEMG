# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Generador de informe visual y numérico de triangulación 3D de electrodos sEMG sobre fotografía frontal de Lucas.
# ==============================================================================

import os
import sys
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Importar agente de triangulación
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "acquisition")))
from calibracion_espacial_electrodos import AntigravityAgent, INDICE_NARIZ_PUNTA, INDICE_IRIS_CENTRO_IZQ, INDICE_IRIS_CENTRO_DER


def generar_informe_completo() -> None:
    ruta_img3 = "/home/santiago/.gemini/antigravity/brain/fd13144b-12f0-469f-a648-c112c1446237/.user_uploaded/media_1789364207033.png"
    img3 = cv2.imread(ruta_img3)
    if img3 is None:
        print(f"Error: No se encontro {ruta_img3}")
        return

    # Extraer panel 1 (vista frontal con gafas de pasta negra)
    # x: 45 a 280, y: 150 a 355
    panel_frontal = img3[150:355, 45:280].copy()
    h_crop, w_crop, _ = panel_frontal.shape
    print(f"Dimensiones del panel frontal: {w_crop} x {h_crop} px")

    # Inicializar agente
    agente = AntigravityAgent(refine_landmarks=True)
    fiduciarios = agente.extraer_malla_facial_3d(panel_frontal)
    if fiduciarios is None:
        print("Error: No se detecto el rostro en el panel frontal.")
        return

    puntos_3d = fiduciarios["puntos_3d"]
    puntos_2d_px = fiduciarios["puntos_2d_px"]
    nariz_3d = fiduciarios["punta_nariz_3d"]
    nariz_px = fiduciarios["punta_nariz_px"]
    dist_ip_3d = fiduciarios["distancia_interpupilar_3d"]
    dist_ip_px = fiduciarios["distancia_interpupilar_px"]
    ojo_izq_px = fiduciarios["ojo_izq_px"]
    ojo_der_px = fiduciarios["ojo_der_px"]

    print(f"Distancia interpupilar 3D: {dist_ip_3d:.6f} | En pixeles: {dist_ip_px:.2f} px")
    print(f"Origen nasal (px): {nariz_px[0]:.1f}, {nariz_px[1]:.1f}")

    # Coordenadas exactas de los 6 broches metálicos identificados visualmente en la imagen
    # En el sistema coordenado de panel_frontal (con origen en y=150 de la imagen original)
    # y_panel = y_original - 150
    electrodos_definidos = [
        {
            "id": 0,
            "canal": "Canal 0 (Ant.)",
            "musculo": "Mylohyoid (Anterior)",
            "cx": 92.0,
            "cy": 293.0 - 150.0, # 143.0
            "color": "#00FF00", # Verde brillante
        },
        {
            "id": 1,
            "canal": "Canal 0 (Post.)",
            "musculo": "Mylohyoid (Posterior)",
            "cx": 92.0,
            "cy": 315.0 - 150.0, # 165.0
            "color": "#1E90FF", # Azul dodger
        },
        {
            "id": 2,
            "canal": "Canal 1 (Sup.)",
            "musculo": "Risorius / Depresor (Superior)",
            "cx": 60.0,
            "cy": 238.0 - 150.0, # 88.0
            "color": "#FFA500", # Naranja
        },
        {
            "id": 3,
            "canal": "Canal 1 (Inf.)",
            "musculo": "Risorius / Depresor (Inferior)",
            "cx": 58.0,
            "cy": 266.0 - 150.0, # 116.0
            "color": "#FF4500", # Naranja rojizo
        },
        {
            "id": 4,
            "canal": "Canal 2 (Sup.)",
            "musculo": "Modiolus / Orbicularis (Superior)",
            "cx": 129.5,
            "cy": 246.0 - 150.0, # 96.0
            "color": "#FF00FF", # Magenta
        },
        {
            "id": 5,
            "canal": "Canal 2 (Inf.)",
            "musculo": "Modiolus / Orbicularis (Inferior)",
            "cx": 129.5,
            "cy": 268.0 - 150.0, # 118.0
            "color": "#8A2BE2", # Azul violeta
        },
    ]

    # Triangulación fiduciaria para cada broche
    tabla_resultados = []
    for el in electrodos_definidos:
        cx, cy = el["cx"], el["cy"]
        dist_2d = np.hypot(puntos_2d_px[:, 0] - cx, puntos_2d_px[:, 1] - cy)
        v_cercano = int(np.argmin(dist_2d))
        error_px = float(dist_2d[v_cercano])

        vertice_3d = puntos_3d[v_cercano]
        delta_3d = vertice_3d - nariz_3d
        d_3d_abs = float(np.linalg.norm(delta_3d))
        d_3d_norm = float(d_3d_abs / dist_ip_3d)
        delta_3d_norm = (delta_3d / dist_ip_3d).tolist()

        el["v_cercano"] = v_cercano
        el["error_px"] = error_px
        el["vx_px"] = float(puntos_2d_px[v_cercano, 0])
        el["vy_px"] = float(puntos_2d_px[v_cercano, 1])
        el["d_3d_abs"] = d_3d_abs
        el["d_3d_norm"] = d_3d_norm
        el["vector_norm"] = delta_3d_norm
        tabla_resultados.append(el)

        print(
            f"[{el['canal']}] {el['musculo']}: v_3d={v_cercano}, "
            f"d_norm={d_3d_norm:.4f}, delta_norm=({delta_3d_norm[0]:.3f}, {delta_3d_norm[1]:.3f}, {delta_3d_norm[2]:.3f})"
        )

    # --------------------------------------------------------------------------
    # GENERACIÓN DE FIGURA GRÁFICA MULTIPANEL CON MATPLOTLIB
    # --------------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 9), facecolor="#0F1117")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 1.0, 1.2], height_ratios=[1.0, 1.0])

    # Subplot 1: Imagen con triangulación y vectores
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor("#161B22")
    rgb_crop = cv2.cvtColor(panel_frontal, cv2.COLOR_BGR2RGB)
    ax1.imshow(rgb_crop)

    # Dibujar malla facial tenue
    ax1.scatter(
        puntos_2d_px[:, 0],
        puntos_2d_px[:, 1],
        s=1.0,
        c="#38BDF8",
        alpha=0.35,
        label="Vértices MediaPipe (478 pts)",
    )

    # Línea interpupilar
    ax1.plot(
        [ojo_izq_px[0], ojo_der_px[0]],
        [ojo_izq_px[1], ojo_der_px[1]],
        color="#FACC15",
        linewidth=2.5,
        linestyle="--",
        label=f"D. Interpupilar ({dist_ip_px:.1f} px)",
    )
    ax1.scatter([ojo_izq_px[0], ojo_der_px[0]], [ojo_izq_px[1], ojo_der_px[1]], s=35, c="#FACC15", edgecolors="#FFFFFF")

    # Origen nasal
    ax1.scatter(nariz_px[0], nariz_px[1], s=90, c="#EF4444", edgecolors="#FFFFFF", zorder=5, label="Origen (Pronasale)")
    ax1.text(
        nariz_px[0] + 5,
        nariz_px[1] - 5,
        "Origen (Nariz)",
        color="#EF4444",
        fontsize=9,
        fontweight="bold",
    )

    # Dibujar electrodos y vectores
    for el in tabla_resultados:
        cx, cy = el["cx"], el["cy"]
        vx, vy = el["vx_px"], el["vy_px"]
        color = el["color"]

        # Vértice de la malla
        ax1.scatter(vx, vy, s=25, c="#38BDF8", edgecolors="#FFFFFF", zorder=4)

        # Centroide del electrodo
        ax1.scatter(cx, cy, s=70, c=color, edgecolors="#FFFFFF", linewidths=1.5, zorder=5)

        # Vector desde el origen a la proyección del electrodo
        ax1.annotate(
            "",
            xy=(cx, cy),
            xytext=(nariz_px[0], nariz_px[1]),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.8, mutation_scale=12),
        )

        # Etiqueta
        d_val = el["d_3d_norm"]
        ax1.text(
            cx + 6,
            cy + 2,
            f"{el['canal']}\n$\\tilde{{d}}={d_val:.3f}$",
            color=color,
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#000000", alpha=0.7, edgecolor=color),
        )

    ax1.set_title("Triangulación Fiduciaria 3D sobre Rostro (Vista Frontal)", color="#FFFFFF", fontsize=12, pad=10)
    ax1.axis("off")
    ax1.legend(loc="lower left", facecolor="#161B22", edgecolor="#30363D", labelcolor="#FFFFFF", fontsize=8)

    # Subplot 2: Diagrama de Dispersión Coordenadas 3D Normalizadas (Plano X-Y)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161B22")
    ax2.grid(True, linestyle=":", color="#30363D", alpha=0.7)
    ax2.axhline(0, color="#EF4444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.axvline(0, color="#EF4444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax2.scatter(0, 0, s=100, c="#EF4444", edgecolors="#FFFFFF", zorder=5, label="Origen (0, 0)")

    for el in tabla_resultados:
        dx, dy, dz = el["vector_norm"]
        # En coordenadas MediaPipe: X hacia la derecha de la imagen, Y hacia abajo
        # Invertimos Y para que apertura/submentoniano quede abajo según convención anatómica
        ax2.scatter(dx, -dy, s=90, c=el["color"], edgecolors="#FFFFFF", zorder=4)
        ax2.plot([0, dx], [0, -dy], color=el["color"], linestyle=":", alpha=0.6)
        ax2.text(
            dx + 0.03,
            -dy + 0.02,
            f"{el['canal']}\n($\\tilde{{d}}={el['d_3d_norm']:.2f}$)",
            color=el["color"],
            fontsize=8,
            fontweight="bold",
        )

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.6, 0.4)
    ax2.set_xlabel("Eje Horizontal $\\tilde{X}$ (Normalizado a D. Interpupilar)", color="#C9D1D9", fontsize=9)
    ax2.set_ylabel("Eje Vertical $-\\tilde{Y}$ (Caudal / Apertura)", color="#C9D1D9", fontsize=9)
    ax2.tick_params(colors="#8B949E", labelsize=8)
    ax2.set_title("Coordenadas 3D Relativas al Origen Nasal $(\\tilde{X}, -\\tilde{Y})$", color="#FFFFFF", fontsize=11)

    # Subplot 3: Desglose de Distancias Euclidianas Normalizadas por Canal
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#161B22")
    ax3.grid(True, linestyle=":", color="#30363D", alpha=0.7, axis="x")

    nombres_canales = [el["canal"] for el in tabla_resultados]
    distancias_norm = [el["d_3d_norm"] for el in tabla_resultados]
    colores_barras = [el["color"] for el in tabla_resultados]

    barras = ax3.barh(nombres_canales, distancias_norm, color=colores_barras, height=0.55, edgecolor="#FFFFFF", alpha=0.85)
    for b, v in zip(barras, distancias_norm):
        ax3.text(
            v + 0.03,
            b.get_y() + b.get_height() / 2,
            f"{v:.4f} $D_{{\\text{{IP}}}}$",
            va="center",
            color="#FFFFFF",
            fontsize=9,
            fontweight="bold",
        )

    ax3.set_xlim(0, max(distancias_norm) * 1.25)
    ax3.set_xlabel("Distancia Euclidiana 3D Normalizada $\\tilde{d} = d_{3\\text{D}} / D_{\\text{IP}}$", color="#C9D1D9", fontsize=9)
    ax3.tick_params(colors="#8B949E", labelsize=8)
    ax3.set_title("Métrica Invariante de Separación al Origen", color="#FFFFFF", fontsize=11)

    # Subplot 4: Panel Técnico y Metadatos JSON Estructurados
    ax4 = fig.add_subplot(gs[:, 2])
    ax4.set_facecolor("#161B22")
    ax4.axis("off")

    lineas_resumen = [
        "CALIBRACION GEOMETRICA ESPACIAL 3D sEMG",
        "Proyecto: NANDU LSD - FCEyN, UBA",
        f"Sujeto: Lucas (2026-09-14) | Fotografia Frontal",
        "-------------------------------------------------------------",
        f"Distancia Interpupilar 3D (D_IP):  {dist_ip_3d:.6f}",
        f"Distancia Interpupilar en Imagen:  {dist_ip_px:.1f} px",
        f"Punta de la Nariz (Pronasale):    Landmark {INDICE_NARIZ_PUNTA}",
        f"Coordenadas Origen (px):          ({nariz_px[0]:.1f}, {nariz_px[1]:.1f})",
        "-------------------------------------------------------------",
        "METRICAS EXACTAS POR ELECTRODO:",
    ]

    for el in tabla_resultados:
        dx, dy, dz = el["vector_norm"]
        lineas_resumen.append(
            f"  * {el['canal']:<15} | Vertice 3D: {el['v_cercano']:<3}\n"
            f"    Musculo:  {el['musculo']}\n"
            f"    Dist. 3D: {el['d_3d_norm']:.4f} D_IP  (Abs: {el['d_3d_abs']:.4f})\n"
            f"    Vector:   [X={dx:+.3f}, Y={dy:+.3f}, Z={dz:+.3f}]\n"
            f"    Error 2D: {el['error_px']:.1f} px"
        )

    lineas_resumen.append("-------------------------------------------------------------")
    lineas_resumen.append("Inyeccion en Cabecera CSV: DISPONIBLE")
    lineas_resumen.append("Compatibilidad: NumPy savetxt / pandas read_csv(comment='#')")

    texto_panel = "\n".join(lineas_resumen)
    ax4.text(
        0.05,
        0.95,
        texto_panel,
        transform=ax4.transAxes,
        color="#F0F6FC",
        fontsize=8.5,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#0D1117", edgecolor="#30363D", linewidth=1.5),
    )

    plt.tight_layout()

    # Guardar en resultados y en el directorio de artefactos
    ruta_salida_resultados = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/resultados/calibracion_espacial_pruebas/informe_calibracion_electrodos_lucas.png"
    os.makedirs(os.path.dirname(ruta_salida_resultados), exist_ok=True)
    plt.savefig(ruta_salida_resultados, dpi=300, bbox_inches="tight")

    ruta_salida_brain = "/home/santiago/.gemini/antigravity/brain/fd13144b-12f0-469f-a648-c112c1446237/informe_calibracion_electrodos_lucas.png"
    plt.savefig(ruta_salida_brain, dpi=300, bbox_inches="tight")

    # Guardar también el JSON de metadatos exactos
    metadatos_export = {
        "fecha": "2026-09-14",
        "sujeto": "Lucas",
        "base_normalizacion": "distancia_interpupilar_3d",
        "distancia_interpupilar_3d": float(dist_ip_3d),
        "distancia_interpupilar_px": float(dist_ip_px),
        "origen_nasal": {
            "indice_mediapipe": INDICE_NARIZ_PUNTA,
            "coordenadas_3d": [float(v) for v in nariz_3d],
            "coordenadas_px": [float(nariz_px[0]), float(nariz_px[1])],
        },
        "electrodos": [
            {
                "id": el["id"],
                "canal": el["canal"],
                "musculo": el["musculo"],
                "vertice_malla_3d": el["v_cercano"],
                "distancia_3d_normalizada": round(el["d_3d_norm"], 6),
                "distancia_3d_absoluta": round(el["d_3d_abs"], 6),
                "vector_3d_normalizado": [round(v, 6) for v in el["vector_norm"]],
                "error_reproyeccion_px": round(el["error_px"], 2),
            }
            for el in tabla_resultados
        ],
    }

    ruta_json = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG/EMG_desarrollo/resultados/calibracion_espacial_pruebas/calibracion_exacta_lucas.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadatos_export, f, indent=2)

    print(f"Informe visual guardado en:")
    print(f"  {ruta_salida_resultados}")
    print(f"  {ruta_salida_brain}")
    print(f"Metadatos JSON guardados en: {ruta_json}")


if __name__ == "__main__":
    generar_informe_completo()
