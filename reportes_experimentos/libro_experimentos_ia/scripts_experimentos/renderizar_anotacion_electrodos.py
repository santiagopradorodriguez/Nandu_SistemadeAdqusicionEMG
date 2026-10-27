"""
Renderizador de detecciones de electrodos sobre fotos de montaje de Candela.
Replica con fidelidad la estética visual del informe:
- Cartela negra translúcida con estadísticas (IED px, mm, coordenadas).
- Círculos de electrodos con centro y aro exterior.
- Líneas de dipolo bipolar.
- Etiquetas de canal y coordenadas (X, Y).
"""

import os
import sys
import csv
from typing import Dict, List, Any
import numpy as np
import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CSV_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/electrodos_anotados_agente.csv")
BASE_DRIVE = os.path.join(REPO_ROOT, "Candela_Organizado_Drive")

COLORES = {
    "CH0": (0, 255, 0),      # Verde
    "CH1": (255, 191, 0),    # Cyan/Celeste (BGR: B=255, G=191, R=0)
    "CH2": (214, 0, 255),    # Magenta/Rosa (BGR: B=214, G=0, R=255)
    "REF": (0, 255, 255),    # Amarillo
}

COLOR_NAMES = {
    "CH0": "#00FF00",
    "CH1": "#00BFFF",
    "CH2": "#D600FF",
    "REF": "#FFFF00",
}


def resolver_ruta_foto(campania: str, archivo_foto: str) -> str:
    # 1. Búsqueda directa
    p1 = os.path.join(BASE_DRIVE, campania, "fotos_montaje", archivo_foto)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DRIVE, campania, archivo_foto)
    if os.path.exists(p2):
        return p2

    # 2. Búsqueda por prefijo numérico en fotos_montaje
    montaje_dir = os.path.join(BASE_DRIVE, campania, "fotos_montaje")
    if os.path.exists(montaje_dir):
        archivos = sorted(os.listdir(montaje_dir))
        # Si archivo_foto empieza con '01_', '02_', etc.
        prefijo = archivo_foto[:3]
        for f in archivos:
            if f.startswith(prefijo) and f.lower().endswith((".png", ".jpeg", ".jpg")):
                return os.path.join(montaje_dir, f)
            # Buscar por palabras clave
            if "frontal" in archivo_foto.lower() and "frontal" in f.lower():
                return os.path.join(montaje_dir, f)
            if "cuello" in archivo_foto.lower() and ("cuello" in f.lower() or "cigomatico" in f.lower()):
                return os.path.join(montaje_dir, f)
            if "lateral" in archivo_foto.lower() and "lateral" in f.lower():
                return os.path.join(montaje_dir, f)

    return ""


def renderizar_foto(campania: str, archivo_foto: str, salida_png: str) -> bool:
    ruta_img = resolver_ruta_foto(campania, archivo_foto)
    if not ruta_img or not os.path.exists(ruta_img):
        print(f"Error: No se encontró imagen para {campania} -> {archivo_foto}")
        return False

    img = cv2.imread(ruta_img)
    if img is None:
        print(f"Error al leer imagen: {ruta_img}")
        return False

    alto, ancho = img.shape[:2]
    canvas = img.copy()

    # Cargar datos del CSV
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filas = [r for r in reader if r["Campaña"] == campania and r["Archivo_Foto"] == archivo_foto]

    if not filas:
        print(f"No hay registros en CSV para {campania} -> {archivo_foto}")
        return False

    # Agrupar por canal
    canales: Dict[str, List[Dict[str, Any]]] = {}
    for r in filas:
        ch = r["Canal"]
        canales.setdefault(ch, []).append(r)

    # 1. Dibujar líneas de dipolo y círculos
    escala_radio = max(int(min(ancho, alto) * 0.015), 6)

    for ch, items in canales.items():
        color_bgr = COLORES.get(ch, (255, 255, 255))

        # Si hay 2 polos (bipolar), dibujar línea conectora
        if len(items) >= 2:
            p1 = (int(float(items[0]["X_px"])), int(float(items[0]["Y_px"])))
            p2 = (int(float(items[1]["X_px"])), int(float(items[1]["Y_px"])))
            cv2.line(canvas, p1, p2, color_bgr, 2, cv2.LINE_AA)

        for it in items:
            px = int(float(it["X_px"]))
            py = int(float(it["Y_px"]))
            id_el = it["ID_Electrodo"]

            # Círculo externo
            cv2.circle(canvas, (px, py), escala_radio, color_bgr, 2, cv2.LINE_AA)
            # Círculo blanco intermedio
            cv2.circle(canvas, (px, py), max(escala_radio - 3, 2), (255, 255, 255), 1, cv2.LINE_AA)
            # Punto central sólido
            cv2.circle(canvas, (px, py), max(escala_radio - 5, 2), (0, 0, 255) if "red" in it["Tipo_Deteccion"].lower() else (0, 0, 0), -1, cv2.LINE_AA)

            # Texto al lado del electrodo
            txt = f"{id_el} ({px},{py})"
            # Desplazar texto para no tapar el electrodo
            offset_x = escala_radio + 4
            offset_y = 4
            if px > ancho * 0.7:
                offset_x = - (len(txt) * 9 + escala_radio + 4)

            cv2.putText(canvas, txt, (px + offset_x, py + offset_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_bgr, 1, cv2.LINE_AA)

    # 2. Dibujar cartela negra translúcida superior derecha
    overlay = canvas.copy()
    box_w = min(int(ancho * 0.45), 450)
    box_w = max(box_w, 300)
    margin = 15

    # Contar líneas necesarias
    lineas_texto = [
        (f"Deteccion de Electrodos - {campania.split('_')[0]}", (240, 240, 240), 0.50, 2),
    ]

    for ch in ["CH0", "CH1", "CH2", "REF"]:
        if ch not in canales:
            continue
        items = canales[ch]
        musculo_base = items[0]["Musculo_Polo"].split("(")[0].strip()
        ied_txt = ""
        if items[0].get("IED_Bipolar_px"):
            ied_px = float(items[0]["IED_Bipolar_px"])
            ied_txt = f" - IED: {ied_px:.1f} px"

        col_ch = COLORES.get(ch, (255, 255, 255))
        lineas_texto.append((f"{ch} {musculo_base}{ied_txt}", col_ch, 0.44, 1))

        if len(items) == 2:
            txt_e1 = f"  E1: ({items[0]['X_px']}, {items[0]['Y_px']}) px"
            txt_e2 = f"  E2: ({items[1]['X_px']}, {items[1]['Y_px']}) px"
            lineas_texto.append((f"{txt_e1} | {txt_e2}", (200, 200, 200), 0.38, 1))
        elif len(items) == 1:
            txt_e1 = f"  E1: ({items[0]['X_px']}, {items[0]['Y_px']}) px - {items[0]['Musculo_Polo']}"
            lineas_texto.append((txt_e1, (200, 200, 200), 0.38, 1))

    line_h = 22
    box_h = len(lineas_texto) * line_h + 20
    x1 = ancho - box_w - margin
    y1 = margin
    x2 = ancho - margin
    y2 = y1 + box_h

    # Fondo oscuro
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    alpha = 0.85
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    # Borde sutil
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), 1)

    # Renderizar textos en la cartela
    curr_y = y1 + 22
    for texto, color, f_scale, f_thick in lineas_texto:
        cv2.putText(canvas, texto, (x1 + 12, curr_y), cv2.FONT_HERSHEY_SIMPLEX, f_scale, color, f_thick, cv2.LINE_AA)
        curr_y += line_h

    os.makedirs(os.path.dirname(salida_png), exist_ok=True)
    cv2.imwrite(salida_png, canvas)
    print(f"[OK] Imagen renderizada guardada en: {salida_png}")
    return True


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        camp = sys.argv[1]
        foto = sys.argv[2]
        out = sys.argv[3]
        renderizar_foto(camp, foto, out)
    else:
        print("Uso: python renderizar_anotacion_electrodos.py <campania> <archivo_foto> <salida_png>")
