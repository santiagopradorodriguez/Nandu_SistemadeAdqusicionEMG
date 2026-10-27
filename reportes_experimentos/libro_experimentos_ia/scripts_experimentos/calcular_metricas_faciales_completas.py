"""
Proyecto NANDU LSD - Sistema de Adquisición sEMG
Cálculo de Biometría Facial 3D y Distancias Métricas Anatómicas (Ojos, Oreja, Nariz, Jaw)
Sujeto Fijo: Candela (D_IP = 61.0 mm).
"""

import os
import sys
import csv
import json
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from EMG_desarrollo.acquisition.calibracion_espacial_electrodos import (
    AntigravityAgent,
    INDICE_NARIZ_PUNTA,
    INDICE_MENTON_GNATHION,
    INDICE_COMISURA_IZQ,
    INDICE_COMISURA_DER,
    INDICE_TRAGO_OREJA_IZQ,
    INDICE_TRAGO_OREJA_DER,
)

CSV_PATH = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/electrodos_anotados_agente.csv")
BASE_DRIVE = os.path.join(REPO_ROOT, "Candela_Organizado_Drive")
OUTPUT_METRIC_CSV = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/metadatos_espaciales_candela_metricos.csv")
OUTPUT_JSON = os.path.join(REPO_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos/resumen_metricas_candela.json")

CANDELA_D_IP_MM = 61.0


def resolver_ruta_foto(campania: str, archivo_foto: str) -> str:
    p1 = os.path.join(BASE_DRIVE, campania, "fotos_montaje", archivo_foto)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_DRIVE, campania, archivo_foto)
    if os.path.exists(p2):
        return p2

    montaje_dir = os.path.join(BASE_DRIVE, campania, "fotos_montaje")
    if os.path.exists(montaje_dir):
        archivos = sorted(os.listdir(montaje_dir))
        prefijo = archivo_foto[:3]
        for f in archivos:
            if f.startswith(prefijo) and f.lower().endswith((".png", ".jpeg", ".jpg")):
                return os.path.join(montaje_dir, f)
            if "frontal" in archivo_foto.lower() and "frontal" in f.lower():
                return os.path.join(montaje_dir, f)
            if "cuello" in archivo_foto.lower() and ("cuello" in f.lower() or "cigomatico" in f.lower()):
                return os.path.join(montaje_dir, f)
            if "lateral" in archivo_foto.lower() and "lateral" in f.lower():
                return os.path.join(montaje_dir, f)
    return ""


def ejecutar_analisis_metrico() -> None:
    agent = AntigravityAgent(refine_landmarks=True)

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        filas = list(reader)

    # Agrupar por foto
    fotos: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in filas:
        key = (r["Campaña"], r["Archivo_Foto"])
        fotos.setdefault(key, []).append(r)

    # Cache de fiduciarias por sesión para fotos sin ojos visibles
    escalas_sesion: Dict[str, float] = {}

    # Primera pasada: obtener escalas de fotos frontales con ojos detectados
    for (camp, foto), rows in fotos.items():
        img_p = resolver_ruta_foto(camp, foto)
        if not img_p or not os.path.exists(img_p):
            continue
        img = cv2.imread(img_p)
        if img is None:
            continue
        fids = agent.extraer_malla_facial_3d(img)
        if fids and "distancia_interpupilar_px" in fids:
            dip_px = float(fids["distancia_interpupilar_px"])
            if dip_px > 10.0:
                esc = CANDELA_D_IP_MM / dip_px
                # Normalizar escala si la resolución es comparable
                escalas_sesion[camp] = esc

    filas_metricas: List[Dict[str, Any]] = []
    resumen_json: Dict[str, Any] = {
        "sujeto": "Candela",
        "d_ip_nominal_mm": CANDELA_D_IP_MM,
        "campañas": {},
    }

    print(f"Iniciando calculo metrico para {len(fotos)} fotos...")

    for (camp, foto), rows in sorted(fotos.items()):
        img_p = resolver_ruta_foto(camp, foto)
        if not img_p or not os.path.exists(img_p):
            print(f"Foto no encontrada: {camp} -> {foto}")
            continue

        img = cv2.imread(img_p)
        if img is None:
            continue

        alto, ancho = img.shape[:2]
        fids = agent.extraer_malla_facial_3d(img)
        has_face = fids is not None

        escala_mm_px: float = 0.25
        metodo_escala = "referencia_hardware"

        if has_face and fids and "distancia_interpupilar_px" in fids:
            dip_px = float(fids["distancia_interpupilar_px"])
            if dip_px > 10.0:
                escala_mm_px = CANDELA_D_IP_MM / dip_px
                metodo_escala = "distancia_interpupilar_ojos_61mm"
        elif camp in escalas_sesion:
            escala_mm_px = escalas_sesion[camp]
            metodo_escala = "escala_sesion_candela"
        else:
            # Sesiones recortadas vs estándar
            if any(k in camp for k in ["25", "29", "23"]):
                escala_mm_px = 0.20  # escala aproximada para recortados
            else:
                escala_mm_px = 0.25

        # Puntos anatómicos faciales de referencia
        pt_nariz = None
        pt_jaw = None
        pt_comisura_izq = None
        pt_comisura_der = None
        pt_trago_izq = None
        pt_trago_der = None
        pose_bio = {}

        if has_face and fids and "puntos_2d_px" in fids:
            pts_2d = fids["puntos_2d_px"]
            pt_nariz = (float(pts_2d[INDICE_NARIZ_PUNTA, 0]), float(pts_2d[INDICE_NARIZ_PUNTA, 1]))
            pt_jaw = (float(pts_2d[INDICE_MENTON_GNATHION, 0]), float(pts_2d[INDICE_MENTON_GNATHION, 1]))
            pt_comisura_izq = (float(pts_2d[INDICE_COMISURA_IZQ, 0]), float(pts_2d[INDICE_COMISURA_IZQ, 1]))
            pt_comisura_der = (float(pts_2d[INDICE_COMISURA_DER, 0]), float(pts_2d[INDICE_COMISURA_DER, 1]))
            pt_trago_izq = (float(pts_2d[INDICE_TRAGO_OREJA_IZQ, 0]), float(pts_2d[INDICE_TRAGO_OREJA_IZQ, 1]))
            pt_trago_der = (float(pts_2d[INDICE_TRAGO_OREJA_DER, 0]), float(pts_2d[INDICE_TRAGO_OREJA_DER, 1]))

            pose_bio = agent.calcular_pose_y_biometria_facial(
                fids, (alto, ancho), d_ip_real_cm=CANDELA_D_IP_MM / 10.0, escala_mm_por_px=escala_mm_px
            )

        # Agrupar por canal para IED
        canales_dict: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            canales_dict.setdefault(r["Canal"], []).append(r)

        # Calcular IED en mm por canal
        ied_mm_por_canal: Dict[str, float] = {}
        angulo_por_canal: Dict[str, float] = {}

        for ch, ch_rows in canales_dict.items():
            if len(ch_rows) == 2:
                x1, y1 = float(ch_rows[0]["X_px"]), float(ch_rows[0]["Y_px"])
                x2, y2 = float(ch_rows[1]["X_px"]), float(ch_rows[1]["Y_px"])
                ied_px = float(np.hypot(x2 - x1, y2 - y1))
                ied_mm = ied_px * escala_mm_px
                ang_deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                ied_mm_por_canal[ch] = ied_mm
                angulo_por_canal[ch] = ang_deg

        resumen_foto: Dict[str, Any] = {
            "archivo": foto,
            "ruta_disco": img_p,
            "resolucion": f"{ancho}x{alto}",
            "rostro_3d_detectado": has_face,
            "metodo_escala": metodo_escala,
            "escala_mm_px": round(escala_mm_px, 4),
            "pose_biometria": pose_bio,
            "electrodos": [],
        }

        for r in rows:
            x_px = float(r["X_px"])
            y_px = float(r["Y_px"])
            ch = r["Canal"]

            # Distancias a puntos anatómicos faciales (en mm)
            d_nariz_mm = round(float(np.hypot(x_px - pt_nariz[0], y_px - pt_nariz[1]) * escala_mm_px), 2) if pt_nariz else None
            d_jaw_mm = round(float(np.hypot(x_px - pt_jaw[0], y_px - pt_jaw[1]) * escala_mm_px), 2) if pt_jaw else None

            # Determinar si el electrodo está en lado izquierdo o derecho de la cara
            es_izq = (x_px <= pt_nariz[0]) if pt_nariz else True
            pt_comisura = pt_comisura_izq if es_izq else pt_comisura_der
            pt_trago = pt_trago_izq if es_izq else pt_trago_der

            d_comisura_mm = round(float(np.hypot(x_px - pt_comisura[0], y_px - pt_comisura[1]) * escala_mm_px), 2) if pt_comisura else None
            d_oreja_trago_mm = round(float(np.hypot(x_px - pt_trago[0], y_px - pt_trago[1]) * escala_mm_px), 2) if pt_trago else None

            ied_mm_val = round(ied_mm_por_canal.get(ch, 0.0), 2) if ch in ied_mm_por_canal else None
            ang_val = round(angulo_por_canal.get(ch, 0.0), 2) if ch in angulo_por_canal else None

            fila_out = {
                "Campaña": camp,
                "Archivo_Foto": foto,
                "Resolucion_WxH": f"{ancho}x{alto}",
                "Canal": ch,
                "ID_Electrodo": r["ID_Electrodo"],
                "Musculo_Polo": r["Musculo_Polo"],
                "X_px": round(x_px, 1),
                "Y_px": round(y_px, 1),
                "Escala_mm_px": round(escala_mm_px, 4),
                "Metodo_Escala": metodo_escala,
                "IED_mm": ied_mm_val if ied_mm_val is not None else "",
                "Angulo_Dipolo_deg": ang_val if ang_val is not None else "",
                "Dist_Nariz_Pronasale_mm": d_nariz_mm if d_nariz_mm is not None else "",
                "Dist_Jaw_Menton_mm": d_jaw_mm if d_jaw_mm is not None else "",
                "Dist_Comisura_Labial_mm": d_comisura_mm if d_comisura_mm is not None else "",
                "Dist_Oreja_Trago_mm": d_oreja_trago_mm if d_oreja_trago_mm is not None else "",
            }
            filas_metricas.append(fila_out)
            resumen_foto["electrodos"].append(fila_out)

        camp_key = camp
        if camp_key not in resumen_json["campañas"]:
            resumen_json["campañas"][camp_key] = []
        resumen_json["campañas"][camp_key].append(resumen_foto)

    # Guardar CSV métrico
    fieldnames = [
        "Campaña", "Archivo_Foto", "Resolucion_WxH", "Canal", "ID_Electrodo",
        "Musculo_Polo", "X_px", "Y_px", "Escala_mm_px", "Metodo_Escala",
        "IED_mm", "Angulo_Dipolo_deg", "Dist_Nariz_Pronasale_mm", "Dist_Jaw_Menton_mm",
        "Dist_Comisura_Labial_mm", "Dist_Oreja_Trago_mm"
    ]
    with open(OUTPUT_METRIC_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filas_metricas)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(resumen_json, f, indent=2, ensure_ascii=False)

    print(f"[OK] Generado CSV metrico: {OUTPUT_METRIC_CSV} ({len(filas_metricas)} electrodos procesados)")
    print(f"[OK] Generado JSON resumen: {OUTPUT_JSON}")


if __name__ == "__main__":
    ejecutar_analisis_metrico()
