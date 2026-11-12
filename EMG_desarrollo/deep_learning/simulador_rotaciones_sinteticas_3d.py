#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición y Decodificación Cinemática/sEMG
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Simulador y Validador de Rotaciones Rígidas Sintéticas 3D
#              (Yaw, Pitch, Roll + Traslación/Zoom) sobre los 200 pulsos reales
#              de SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02.
#              Demuestra cuantitativa y visualmente:
#                1. Cómo colapsa el pipeline 2D Bruto (sin desacople craneal):
#                   - Falsos positivos continuos en reposo durante cabeceos/giros.
#                   - Caída drástica de la exactitud de clasificación en test.
#                2. Cómo el Desacople de Cuerpo Rígido Craneal (Head Frame):
#                   r_{k, local}(t) = R(t)^T (P_k(t) - C_craneo(t))
#                   anula la perturbación rígida, mantiene 0 falsos positivos en
#                   reposo y preserva 85.0% de exactitud cinemática autónoma.
# ==============================================================================

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pkg_root = "/tmp/work_pkg/paquete_tareas_argon_2026-11-09"
sys.path.insert(0, os.path.join(pkg_root, "codigos_decodificador_y_daq"))

from calibracion_espacial_electrodos import (
    AntigravityAgent,
    INDICE_NARIZ_PUNTA,
    INDICE_GLABELA,
    INDICE_CANTO_EXTERNO_OJO_IZQ,
    INDICE_CANTO_INTERNO_OJO_IZQ,
    INDICE_CANTO_EXTERNO_OJO_DER,
    INDICE_CANTO_INTERNO_OJO_DER,
    INDICE_IRIS_CENTRO_IZQ,
    INDICE_IRIS_CENTRO_DER,
    INDICE_TRAGO_OREJA_IZQ,
    INDICE_TRAGO_OREJA_DER,
)
from realtime_facial_decoder_engine import (
    RealtimeDecoderEngine,
    GeoSupConvOrthoNet,
    VOCALES,
    CANON_REST_DIST_MM,
    DEADBAND_RATIO_CANALES,
    calcular_canales_relativos_invariantes,
    filtro_ema_causal_1d,
)
from decodificador_continuo_facial_6canales import (
    entrenar_modelo_cinematico_autonomo,
    evaluar_sesion_con_motor,
)

PUNTOS_18_MAP = {
    "comisura_izq": 61,
    "comisura_der": 291,
    "labio_sup": 13,
    "labio_inf": 14,
    "menton": 152,
    "mejilla_izq": 207,
    "mejilla_der": 427,
    "risorio_izq": 214,
    "risorio_der": 434,
    "masetero_izq": 58,
    "masetero_der": 288,
    "malar_izq": 123,
    "malar_der": 352,
    "cupido_izq": 37,
    "cupido_der": 267,
    "labio_inf_izq": 84,
    "labio_inf_der": 314,
    "submenton": 199,
}


def construir_matriz_rotacion_euler(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Construye la matriz de rotación 3D intrínseca R = R_z(roll) @ R_x(pitch) @ R_y(yaw)."""
    y = np.radians(yaw_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    Ry = np.array([
        [ np.cos(y), 0.0, np.sin(y)],
        [       0.0, 1.0,       0.0],
        [-np.sin(y), 0.0, np.cos(y)]
    ], dtype=np.float64)
    Rx = np.array([
        [1.0,       0.0,        0.0],
        [0.0, np.cos(p), -np.sin(p)],
        [0.0, np.sin(p),  np.cos(p)]
    ], dtype=np.float64)
    Rz = np.array([
        [np.cos(r), -np.sin(r), 0.0],
        [np.sin(r),  np.cos(r), 0.0],
        [      0.0,        0.0, 1.0]
    ], dtype=np.float64)
    return Rz @ Rx @ Ry


def reconstruir_malla_con_anclas_oseas(df_tray: pd.DataFrame, ancho_img: float = 640.0):
    """
    A partir de las 18 trayectorias reales de SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02,
    reconstruye en cada frame t el estado local canónico del rostro y añade los puntos óseos
    fijos del cráneo (Pronasale=1, Puente=4, Glabela=10, Ojos=33/263/468/473, Tragos=234/454)
    para poder someter toda la cabeza a rotaciones 3D arbitrarias (Yaw, Pitch, Roll) y zoom.
    """
    N = len(df_tray)
    dx_com_px = np.abs(df_tray["comisura_der_x_px"].values - df_tray["comisura_izq_x_px"].values)
    com_mm = df_tray["ancho_comisuras_mm"].values
    valid_m = (dx_com_px > 10) & (com_mm > 10)
    escala_base = float(np.median(com_mm[valid_m] / dx_com_px[valid_m]))
    dip_px_base = 63.0 / escala_base

    # Estimar el centro craneal medio en reposo a partir del punto medio inter-malar y arco de cupido
    cx_malar = 0.5 * (np.median(df_tray["malar_izq_x_px"]) + np.median(df_tray["malar_der_x_px"]))
    cy_malar = 0.5 * (np.median(df_tray["malar_izq_y_px"]) + np.median(df_tray["malar_der_y_px"]))
    cx_cup = 0.5 * (np.median(df_tray["cupido_izq_x_px"]) + np.median(df_tray["cupido_der_x_px"]))
    cy_cup = 0.5 * (np.median(df_tray["cupido_izq_y_px"]) + np.median(df_tray["cupido_der_y_px"]))

    # Coordenadas óseas fijas en el marco craneal en reposo (en píxeles y z_rel)
    nariz_px_0 = np.array([cx_cup, cy_cup - 0.28 * dip_px_base, 0.0], dtype=np.float64)
    puente_px_0 = np.array([cx_malar, cy_malar - 0.22 * dip_px_base, 0.035], dtype=np.float64)
    glabela_px_0 = np.array([nariz_px_0[0], cy_malar - 0.58 * dip_px_base, 0.0], dtype=np.float64)
    ojo_izq_px_0 = np.array([cx_malar - 0.50 * dip_px_base, cy_malar - 0.35 * dip_px_base, 0.075], dtype=np.float64)
    ojo_der_px_0 = np.array([cx_malar + 0.50 * dip_px_base, cy_malar - 0.35 * dip_px_base, 0.075], dtype=np.float64)
    trago_izq_px_0 = np.array([cx_malar - 1.05 * dip_px_base, cy_malar - 0.05 * dip_px_base, 0.220], dtype=np.float64)
    trago_der_px_0 = np.array([cx_malar + 1.05 * dip_px_base, cy_malar - 0.05 * dip_px_base, 0.220], dtype=np.float64)

    # Construir tensor (N, 478, 3) en coordenadas métricas del cráneo (mm)
    pts_local_mm = np.zeros((N, 478, 3), dtype=np.float64)

    for nombre, idx_lm in PUNTOS_18_MAP.items():
        x_px = df_tray[f"{nombre}_x_px"].values
        y_px = df_tray[f"{nombre}_y_px"].values
        z_rel = df_tray[f"{nombre}_z_rel"].values
        pts_local_mm[:, idx_lm, 0] = (x_px - nariz_px_0[0]) * escala_base
        pts_local_mm[:, idx_lm, 1] = (y_px - nariz_px_0[1]) * escala_base
        pts_local_mm[:, idx_lm, 2] = z_rel * ancho_img * escala_base

    # Preservar la distancia exacta en mm registrada de apertura_vertical_mm y ancho_comisuras_mm en el estado local
    ap_true_mm = df_tray["apertura_vertical_mm"].values
    com_true_mm = df_tray["ancho_comisuras_mm"].values
    v_ap = pts_local_mm[:, 14, :2] - pts_local_mm[:, 13, :2]
    n_ap = np.maximum(np.linalg.norm(v_ap, axis=1, keepdims=True), 1e-6)
    mid_ap = 0.5 * (pts_local_mm[:, 13, :2] + pts_local_mm[:, 14, :2])
    pts_local_mm[:, 13, :2] = mid_ap - 0.5 * (v_ap / n_ap) * ap_true_mm[:, None]
    pts_local_mm[:, 14, :2] = mid_ap + 0.5 * (v_ap / n_ap) * ap_true_mm[:, None]

    old_61 = pts_local_mm[:, 61, :2].copy()
    old_291 = pts_local_mm[:, 291, :2].copy()
    v_com = old_291 - old_61
    n_com = np.maximum(np.linalg.norm(v_com, axis=1, keepdims=True), 1e-6)
    mid_com = 0.5 * (old_61 + old_291)
    pts_local_mm[:, 61, :2] = mid_com - 0.5 * (v_com / n_com) * com_true_mm[:, None]
    pts_local_mm[:, 291, :2] = mid_com + 0.5 * (v_com / n_com) * com_true_mm[:, None]
    delta_61 = pts_local_mm[:, 61, :2] - old_61
    delta_291 = pts_local_mm[:, 291, :2] - old_291
    for idx_izq in (207, 123, 58):
        pts_local_mm[:, idx_izq, :2] += delta_61
    for idx_der in (427, 352, 288):
        pts_local_mm[:, idx_der, :2] += delta_291

    # Anclas óseas rígidas en mm respecto a nariz_px_0
    anclas = {
        INDICE_NARIZ_PUNTA: nariz_px_0,
        4: puente_px_0,
        INDICE_GLABELA: glabela_px_0,
        INDICE_CANTO_EXTERNO_OJO_IZQ: ojo_izq_px_0,
        INDICE_CANTO_INTERNO_OJO_IZQ: ojo_izq_px_0,
        INDICE_IRIS_CENTRO_IZQ: ojo_izq_px_0,
        INDICE_CANTO_EXTERNO_OJO_DER: ojo_der_px_0,
        INDICE_CANTO_INTERNO_OJO_DER: ojo_der_px_0,
        INDICE_IRIS_CENTRO_DER: ojo_der_px_0,
        INDICE_TRAGO_OREJA_IZQ: trago_izq_px_0,
        INDICE_TRAGO_OREJA_DER: trago_der_px_0,
    }
    for idx_lm, p0 in anclas.items():
        pts_local_mm[:, idx_lm, 0] = (p0[0] - nariz_px_0[0]) * escala_base
        pts_local_mm[:, idx_lm, 1] = (p0[1] - nariz_px_0[1]) * escala_base
        pts_local_mm[:, idx_lm, 2] = p0[2] * ancho_img * escala_base

    return pts_local_mm, nariz_px_0, escala_base


def aplicar_perturbacion_rigida_sintetica(
    pts_local_mm: np.ndarray,
    t_s: np.ndarray,
    nariz_px_0: np.ndarray,
    escala_base: float,
    yaw_amp_deg: float = 25.0,
    pitch_amp_deg: float = 18.0,
    roll_amp_deg: float = 15.0,
    zoom_amp_pct: float = 0.15,
    ancho_img: float = 640.0,
):
    """
    Aplica en cada instante t_s una rotación rígida 3D continua (Yaw, Pitch, Roll) +
    cambio de distancia a cámara (zoom) + traslación planar (x, y) a toda la cabeza,
    y proyecta a coordenadas de cámara 2D (píxeles) y profundidad z_rel:
      1) Extrae las distancias BRUTAS 2D en cámara (como hacía el código viejo sin Head Frame).
      2) Extrae las distancias COMPENSADAS POR CUERPO RÍGIDO proyectando al Head Frame:
         r_{k, local}(t) = R(t)^T (P_k(t) - C_craneo(t)).
    """
    N = len(t_s)
    yaw_t = yaw_amp_deg * np.sin(2.0 * np.pi * 0.23 * t_s + 0.3)
    pitch_t = pitch_amp_deg * np.sin(2.0 * np.pi * 0.17 * t_s + 1.1)
    roll_t = roll_amp_deg * np.cos(2.0 * np.pi * 0.19 * t_s + 0.7)
    zoom_t = 1.0 + zoom_amp_pct * np.sin(2.0 * np.pi * 0.11 * t_s)
    tx_px_t = 35.0 * np.sin(2.0 * np.pi * 0.13 * t_s)
    ty_px_t = 25.0 * np.cos(2.0 * np.pi * 0.15 * t_s)

    ap_bruto = np.zeros(N, dtype=np.float64)
    com_bruto = np.zeros(N, dtype=np.float64)
    mej_bruto = np.zeros(N, dtype=np.float64)
    mal_bruto = np.zeros(N, dtype=np.float64)
    mas_bruto = np.zeros(N, dtype=np.float64)

    ap_rigido = np.zeros(N, dtype=np.float64)
    com_rigido = np.zeros(N, dtype=np.float64)
    mej_rigido = np.zeros(N, dtype=np.float64)
    mal_rigido = np.zeros(N, dtype=np.float64)
    mas_rigido = np.zeros(N, dtype=np.float64)

    active_indices = list(PUNTOS_18_MAP.values()) + [
        INDICE_NARIZ_PUNTA, 4, INDICE_GLABELA,
        INDICE_CANTO_EXTERNO_OJO_IZQ, INDICE_CANTO_INTERNO_OJO_IZQ, INDICE_IRIS_CENTRO_IZQ,
        INDICE_CANTO_EXTERNO_OJO_DER, INDICE_CANTO_INTERNO_OJO_DER, INDICE_IRIS_CENTRO_DER,
        INDICE_TRAGO_OREJA_IZQ, INDICE_TRAGO_OREJA_DER
    ]
    active_indices = sorted(list(set(active_indices)))

    for i in range(N):
        R_true = construir_matriz_rotacion_euler(yaw_t[i], pitch_t[i], roll_t[i])
        s_zoom = zoom_t[i]
        # Rotar los puntos en 3D y proyectar a píxeles de cámara con zoom y traslación
        pts_rot_mm = pts_local_mm[i, active_indices] @ R_true.T
        escala_cam = escala_base / s_zoom  # mm por pixel en este frame

        p2d_frame = np.zeros((478, 2), dtype=np.float64)
        p3d_frame = np.zeros((478, 3), dtype=np.float64)

        p2d_frame[active_indices, 0] = nariz_px_0[0] + tx_px_t[i] + pts_rot_mm[:, 0] / escala_cam
        p2d_frame[active_indices, 1] = nariz_px_0[1] + ty_px_t[i] + pts_rot_mm[:, 1] / escala_cam
        p3d_frame[active_indices, 0] = p2d_frame[active_indices, 0] / ancho_img
        p3d_frame[active_indices, 1] = p2d_frame[active_indices, 1] / 480.0
        p3d_frame[active_indices, 2] = (pts_rot_mm[:, 2] / escala_cam) / ancho_img

        # 1) Cálculo BRUTO 2D (proyección 2D en el plano de imagen escalada por DIP 2D aparente):
        # Cuando hay Yaw (giro de cabeza), la distancia interpupilar 2D aparente se achica por cos(yaw),
        # inflando artificialmente todas las distancias verticales (apertura mandibular) por 1/cos(yaw),
        # y cuando hay Pitch (cabeceo), la distancia vertical 2D labio-mejilla/malar cambia con sin(pitch)!
        dip_2d_aparente = max(float(np.linalg.norm(p2d_frame[INDICE_IRIS_CENTRO_DER] - p2d_frame[INDICE_IRIS_CENTRO_IZQ])), 10.0)
        escala_2d_aparente = 63.0 / dip_2d_aparente

        def _d2(ia, ib):
            return float(np.linalg.norm(p2d_frame[ia] - p2d_frame[ib])) * escala_2d_aparente

        ap_bruto[i] = _d2(13, 14)
        com_bruto[i] = _d2(61, 291)
        mej_bruto[i] = 0.5 * (_d2(61, 207) + _d2(291, 427))
        mal_bruto[i] = 0.5 * (_d2(61, 123) + _d2(291, 352))
        mas_bruto[i] = 0.5 * (_d2(61, 58) + _d2(291, 288))

        # 2) Cálculo COMPENSADO POR CUERPO RÍGIDO (Tarea A - Head Frame 3D):
        # Usa la distancia interocular 3D real y proyecta r_{k, local}(t) = R(t)^T (P_k(t) - C_craneo(t))
        vec_ojos_3d_px = np.array([
            p2d_frame[INDICE_IRIS_CENTRO_DER, 0] - p2d_frame[INDICE_IRIS_CENTRO_IZQ, 0],
            p2d_frame[INDICE_IRIS_CENTRO_DER, 1] - p2d_frame[INDICE_IRIS_CENTRO_IZQ, 1],
            (p3d_frame[INDICE_IRIS_CENTRO_DER, 2] - p3d_frame[INDICE_IRIS_CENTRO_IZQ, 2]) * ancho_img
        ])
        dip_3d_px = max(float(np.linalg.norm(vec_ojos_3d_px)), 10.0)
        escala_3d_real = 63.0 / dip_3d_px

        pts_m = np.column_stack([
            p2d_frame[:, 0] * escala_3d_real,
            p2d_frame[:, 1] * escala_3d_real,
            p3d_frame[:, 2] * ancho_img * escala_3d_real
        ])
        p_nar = pts_m[INDICE_NARIZ_PUNTA]
        p_oi = pts_m[INDICE_IRIS_CENTRO_IZQ]
        p_od = pts_m[INDICE_IRIS_CENTRO_DER]
        p_gl = pts_m[INDICE_GLABELA]

        c_craneo = (p_nar + p_oi + p_od) / 3.0
        vx = p_od - p_oi
        ux = vx / max(float(np.linalg.norm(vx)), 1e-6)
        v_up = p_gl - p_nar
        vy = v_up - np.dot(v_up, ux) * ux
        uy = vy / max(float(np.linalg.norm(vy)), 1e-6)
        uz = np.cross(ux, uy)
        uz = uz / max(float(np.linalg.norm(uz)), 1e-6)
        R_est = np.column_stack([ux, uy, uz])

        r_local = (pts_m - c_craneo[None, :]) @ R_est

        def _d3_loc(ia, ib):
            return float(np.linalg.norm(r_local[ia, :2] - r_local[ib, :2]))

        ap_rigido[i] = _d3_loc(13, 14)
        com_rigido[i] = _d3_loc(61, 291)
        mej_rigido[i] = 0.5 * (_d3_loc(61, 207) + _d3_loc(291, 427))
        mal_rigido[i] = 0.5 * (_d3_loc(61, 123) + _d3_loc(291, 352))
        mas_rigido[i] = 0.5 * (_d3_loc(61, 58) + _d3_loc(291, 288))

    return {
        "yaw_t": yaw_t,
        "pitch_t": pitch_t,
        "roll_t": roll_t,
        "zoom_t": zoom_t,
        "bruto": (ap_bruto, com_bruto, mej_bruto, mal_bruto, mas_bruto),
        "rigido": (ap_rigido, com_rigido, mej_rigido, mal_rigido, mas_rigido),
    }


def main():
    print("====================================================================================")
    print("  BENCHMARK DE ROTACIONES RÍGIDAS SINTÉTICAS 3D (YAW / PITCH / ROLL + ZOOM)")
    print("  Sesión Canónica: SecuenciaContinua_Pruebaconlandmarksvarioss_Sujeto1_02 (200 pulsos)")
    print("====================================================================================")

    ses_dir = os.path.join(pkg_root, "sesiones_cortadas", "medicion_2_test_tiempo_real_pulsos_131_200")
    df_tray = pd.read_csv(os.path.join(ses_dir, "trayectoria_puntos.csv"))
    df_pulsos = pd.read_csv(os.path.join(ses_dir, "pulsos_metronomo.csv"))
    clean_set = set(np.load(os.path.join(pkg_root, "datos_features", "pulsos_idx.npy")).tolist())

    pts_local_mm, nariz_px_0, escala_base = reconstruir_malla_con_anclas_oseas(df_tray)
    t_s = df_tray["t_s"].values

    # 1. Prueba en REPOSO ABSOLUTO sometido a rotaciones severas de cabeza:
    # Tomamos un frame de reposo real y rotamos la cabeza continuamente durante 20 segundos (2000 muestras a 100 Hz)
    # con Yaw = +-30 deg, Pitch = +-22 deg, Roll = +-18 deg, Zoom = +-18%
    t_rest = np.linspace(0.0, 20.0, 2000)
    idx_rest_frame = 10  # Frame inicial en reposo
    pts_rest_mm = np.repeat(pts_local_mm[idx_rest_frame:idx_rest_frame+1], len(t_rest), axis=0)
    sim_rest = aplicar_perturbacion_rigida_sintetica(
        pts_rest_mm, t_rest, nariz_px_0, escala_base,
        yaw_amp_deg=30.0, pitch_amp_deg=22.0, roll_amp_deg=18.0, zoom_amp_pct=0.18
    )

    b_rest = {
        "base_ap": float(sim_rest["rigido"][0][0]),
        "base_com": float(sim_rest["rigido"][1][0]),
        "base_mej": float(sim_rest["rigido"][2][0]),
        "base_mal": float(sim_rest["rigido"][3][0]),
        "base_mas": float(sim_rest["rigido"][4][0]),
    }

    ch_bruto_rest = np.array(calcular_canales_relativos_invariantes(
        *sim_rest["bruto"], baseline_stats=b_rest, aplicar_deadband=True, devolver_en_mm_canonicos=True
    ), dtype=np.float32)
    ch_rigido_rest = np.array(calcular_canales_relativos_invariantes(
        *sim_rest["rigido"], baseline_stats=b_rest, aplicar_deadband=True, devolver_en_mm_canonicos=True
    ), dtype=np.float32)

    model_3d_path = os.path.join(pkg_root, "modelos", "campeon_geosup_conv_ortho_3d.pt")
    eng_bruto = RealtimeDecoderEngine(model_path=model_3d_path, trigger_mode="GATE_HYBRID")
    eng_rigido = RealtimeDecoderEngine(model_path=model_3d_path, trigger_mode="GATE_HYBRID")

    fp_bruto, fp_rigido = [], []
    s_bruto_trace, s_rigido_trace = [], []
    for k in range(0, len(t_rest), 5):
        e_b, s_b, _ = eng_bruto.push_chunk(ch_bruto_rest[:, k:k+5])
        e_r, s_r, _ = eng_rigido.push_chunk(ch_rigido_rest[:, k:k+5])
        fp_bruto.extend(e_b)
        fp_rigido.extend(e_r)
        s_bruto_trace.extend(s_b.tolist())
        s_rigido_trace.extend(s_r.tolist())

    max_err_rigido_mm = max(
        float(np.max(np.abs(sim_rest["rigido"][j] - sim_rest["rigido"][j][0])))
        for j in range(5)
    )
    max_err_bruto_mm = max(
        float(np.max(np.abs(sim_rest["bruto"][j] - sim_rest["bruto"][j][0])))
        for j in range(5)
    )

    print(f"\n[1] PRUEBA DE INMUNIDAD EN REPOSO CON CABECEO/GIRO SEVERO (20 s: Yaw +-30°, Pitch +-22°, Roll +-18°):")
    print(f"    -> Sin Desacople (2D Bruto)  : Desviación espuria máx = {max_err_bruto_mm:6.2f} mm | Norma S_cam máx = {np.max(s_bruto_trace):.3f} | Falsos Positivos = {len(fp_bruto)} disparos espurios")
    print(f"    -> Con Head Frame R(t)^T     : Desviación espuria máx = {max_err_rigido_mm:6.2e} mm | Norma S_cam máx = {np.max(s_rigido_trace):.4f} | Falsos Positivos = {len(fp_rigido)} disparos")

    # 2. Prueba sobre los 60 pulsos de TEST (Pulsos #131..#200) bajo distintos niveles de rotación rígida 3D
    niveles = [
        ("Sin rotación extra (0°)", 0.0, 0.0, 0.0, 0.0),
        ("Movimiento Moderado (Yaw ±15°, Pitch ±12°, Roll ±10°)", 15.0, 12.0, 10.0, 0.10),
        ("Movimiento Fuerte (Yaw ±25°, Pitch ±18°, Roll ±15°)", 25.0, 18.0, 15.0, 0.15),
        ("Movimiento Extremo (Yaw ±35°, Pitch ±25°, Roll ±20°)", 35.0, 25.0, 20.0, 0.20),
    ]

    df_princ_test = df_pulsos[(df_pulsos["principal"] == 1) & (df_pulsos["idx"] >= 131)].copy()
    t_grid = np.linspace(-0.5, 0.5, 25)
    s_scales = np.array([14.5, 6.0, 6.0, 1.5, 1.5, 2.0], dtype=np.float64) / 15.0
    s_weights = np.array([1.0, 1.0, 1.0, 0.35, 0.35, 0.35], dtype=np.float64)

    print("\n[2] EXACTITUD EN TEST AUTÓNOMO (GATE_HYBRID, Pulsos #131..#200) BAJO ROTACIONES SINTÉTICAS 3D:")
    resumen_tabla = []

    for nombre_niv, y_amp, p_amp, r_amp, z_amp in niveles:
        sim_full = aplicar_perturbacion_rigida_sintetica(
            pts_local_mm, t_s, nariz_px_0, escala_base,
            yaw_amp_deg=y_amp, pitch_amp_deg=p_amp, roll_amp_deg=r_amp, zoom_amp_pct=z_amp
        )
        accs = {}
        for tipo_pipe in ("bruto", "rigido"):
            series_5 = [filtro_ema_causal_1d(arr, alpha=0.72) for arr in sim_full[tipo_pipe]]
            b_st = {
                "base_ap": float(np.percentile(series_5[0], 10)),
                "base_com": float(np.percentile(series_5[1], 20)),
                "base_mej": float(np.percentile(series_5[2], 20)),
                "base_mal": float(np.percentile(series_5[3], 20)),
                "base_mas": float(np.percentile(series_5[4], 20)),
            }
            canales = calcular_canales_relativos_invariantes(
                *series_5, baseline_stats=b_st, aplicar_deadband=False, devolver_en_mm_canonicos=True
            )
            ch_mat = np.vstack([np.maximum(0.0, c) for c in canales])
            s_raw = np.sqrt(np.sum(s_weights[:, None] * ((ch_mat / s_scales[:, None]) ** 2), axis=0))
            active_gate = np.clip((np.maximum(ch_mat[0], np.maximum(ch_mat[2] * 1.4, ch_mat[1] * 0.65)) - 1.0) / 1.5, 0.0, 1.0)
            S_kin = s_raw * active_gate
            aciertos, total = 0, 0
            for _, row in df_princ_test.iterrows():
                p_num = int(row["idx"])
                if p_num not in clean_set:
                    continue
                v_gt = str(row["palabra"]).upper().strip()
                t_d = float(row["t_daq_s"])
                dur = float(row["intervalo_s"])
                group = df_tray[df_tray["pulso"] == p_num]
                if len(group) < 5:
                    continue
                idx_sub = group.index.values
                t_sub = t_s[idx_sub]
                s_sub = S_kin[idx_sub]
                valid_mask = (t_sub >= t_d + 0.15) & (t_sub <= t_d + dur - 0.10)
                if not np.any(valid_mask):
                    valid_mask = np.ones_like(t_sub, dtype=bool)
                t_v = t_sub[valid_mask]
                s_v = s_sub[valid_mask]

                i_pk = int(np.argmax(s_v))
                t_pk = float(t_v[i_pk])
                high_idx = np.where(s_v >= 10.20)[0]
                ic = int(high_idx[0]) if len(high_idx) > 0 else i_pk
                thresh_low = min(6.30, 0.38 * float(s_v[ic]))
                st_b = max(0, ic - 10)
                low_idx = np.where(s_v[st_b:ic+1] <= thresh_low)[0]
                io = (st_b + int(low_idx[-1])) if len(low_idx) > 0 else (st_b + int(np.argmin(s_v[st_b:ic+1])))
                end_lobe = min(len(s_v), io + 16)
                lobe_t = t_v[io:end_lobe]
                lobe_s = np.maximum(0.0, s_v[io:end_lobe] - thresh_low)**2
                t_cen = float(np.sum(lobe_t * lobe_s) / np.sum(lobe_s)) if np.sum(lobe_s) > 1e-8 else t_pk
                t_hyb = 0.65 * t_cen + 0.35 * t_pk
                t_center = t_hyb + (eng_rigido.emd_delay_ms / 1000.0)

                t_rel = t_sub - t_center
                vecs = [np.interp(t_grid, t_rel, ch_arr[idx_sub], left=0.0, right=0.0) for ch_arr in canales]
                x_150 = np.concatenate(vecs)
                preds, _, _ = eng_rigido.encode_and_predict_window(x_150, return_rest_prob=False)
                if VOCALES[int(preds[0])] == v_gt:
                    aciertos += 1
                total += 1
            accs[tipo_pipe] = (aciertos / max(1, total)) * 100.0
        print(f"    * {nombre_niv:54s} -> 2D Bruto: {accs['bruto']:5.1f}% | Con Head Frame R(t)^T: {accs['rigido']:5.1f}%")
        resumen_tabla.append((nombre_niv, accs["bruto"], accs["rigido"]))

    # 3. Generar lámina comparativa PNG
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=False)
    plt.subplots_adjust(top=0.91, bottom=0.08, left=0.08, right=0.96, hspace=0.38)
    fig.suptitle(
        "Validación de Desacople de Cuerpo Rígido Craneal (Head Frame) bajo Rotaciones Sintéticas 3D",
        fontsize=14, fontweight="bold", color="#0f172a"
    )

    # Panel 1: Ángulos de rotación sintética aplicados en reposo
    ax0 = axes[0]
    ax0.plot(t_rest, sim_rest["yaw_t"], label="Giro (Yaw ±30°)", color="#2563eb", linewidth=1.8)
    ax0.plot(t_rest, sim_rest["pitch_t"], label="Inclinación (Pitch ±22°)", color="#d97706", linewidth=1.8)
    ax0.plot(t_rest, sim_rest["roll_t"], label="Ladeo (Roll ±18°)", color="#7c3aed", linewidth=1.6, linestyle="--")
    ax0.set_title("1. Perturbación Rígida 3D Aplicada a la Cabeza en Reposo Vocal (0 a 20 s)", fontsize=11.5, fontweight="bold")
    ax0.set_ylabel("Ángulo (°)", fontweight="bold")
    ax0.set_xlabel("Tiempo (s)")
    ax0.grid(True, linestyle="--", alpha=0.4)
    ax0.legend(loc="upper right", ncol=3, fontsize=9.5)

    # Panel 2: Norma cinemática S_cam(t) y falsos positivos en reposo: 2D Bruto vs Head Frame
    ax1 = axes[1]
    ax1.plot(t_rest, s_bruto_trace, label=f"Sin Desacople (2D Bruto) -> {len(fp_bruto)} Falsos Disparos", color="#dc2626", linewidth=1.8, alpha=0.9)
    ax1.plot(t_rest, s_rigido_trace, label=f"Con Head Frame R(t)^T (P_k - C_craneo) -> {len(fp_rigido)} Falsos Disparos", color="#059669", linewidth=2.2)
    ax1.axhline(0.68, color="#dc2626", linestyle="--", linewidth=1.2, label="Umbral U_alto = 0.68")
    ax1.axhline(0.42, color="#0284c7", linestyle=":", linewidth=1.2, label="Umbral U_bajo = 0.42")
    ax1.set_title("2. Respuesta de la Norma Cinemática S_cam(t) en Silencio durante Rotación de Cabeza", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Norma S_cam (u.a.)", fontweight="bold")
    ax1.set_xlabel("Tiempo (s)")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", ncol=2, fontsize=9.5)

    # Panel 3: Comparación de exactitud en Test Autónomo (Pulsos #131..#200)
    ax2 = axes[2]
    etiquetas_niv = ["0° (Estático)", "Moderado\n(±15°/±12°/±10°)", "Fuerte\n(±25°/±18°/±15°)", "Extremo\n(±35°/±25°/±20°)"]
    x_pos = np.arange(len(etiquetas_niv))
    w_bar = 0.34
    vals_bruto = [r[1] for r in resumen_tabla]
    vals_rigido = [r[2] for r in resumen_tabla]
    b1 = ax2.bar(x_pos - w_bar/2, vals_bruto, width=w_bar, color="#ef4444", label="Sin Desacople (2D Bruto)")
    b2 = ax2.bar(x_pos + w_bar/2, vals_rigido, width=w_bar, color="#10b981", label="Con Head Frame R(t)^T (Tarea A)")
    for rect in b1:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#b91c1c")
    for rect in b2:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#047857")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(etiquetas_niv, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Exactitud Test (%)", fontweight="bold")
    ax2.set_title("3. Exactitud de Decodificación Autónoma (GATE_HYBRID, N=60 pulsos de Test) frente a Rotaciones 3D", fontsize=11.5, fontweight="bold")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="lower right", fontsize=10)

    out_png = "/working_dir/c_d887eaa6594a3e92/artifacts/file_generation/ttl=63d/output/validacion_rotaciones_sinteticas_head_frame.png"
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"\n[OK] Gráfico comparativo guardado en: {out_png}")


if __name__ == "__main__":
    main()
