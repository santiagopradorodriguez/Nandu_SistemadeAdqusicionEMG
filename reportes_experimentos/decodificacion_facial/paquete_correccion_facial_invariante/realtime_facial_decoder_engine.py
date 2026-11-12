#!/usr/bin/env python3
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición y Decodificación Cinemática/sEMG
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Descripción: Motor de decodificación en tiempo real para 6 Canales Faciales
#              (y retrocompatible con 3 Canales sEMG) con:
#              - Estabilización adaptativa One-Euro Filter 3D (f_c,min=0.85 Hz, beta=0.055)
#                sobre coordenadas crudas de MediaPipe + Ajuste Rígido Procrustes Kabsch
#                ponderado sobre la cresta ósea nasal-frontal [1, 4, 6, 168, 10, 33, 263]
#              - Norma Cinemática Ponderada S(t) con realce específico del Canal 2
#                (Constricción / Protrusión Orbicular de /o/ y /u/, w_2=2.25, s_2=3.6/15)
#              - Segmentación de Habla Continua por Valles de Inflexión Cinemática
#                (\dot{S}(t) = 0) sin exigir retorno al reposo basal entre fonemas
#              - Espacio Latente 4D/3D/2D Ortogonal a Postura Craneal (Corr = 0.000)
#                con Excursión Diferencial Pre-Onset y Refinamiento por Margen Focal.
# ==============================================================================

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import scipy.ndimage
from scipy import signal, linalg, spatial
from scipy.sparse.csgraph import shortest_path

try:
    import torch
    import torch.nn as nn
    TORCH_DISPONIBLE = True
except ImportError:
    torch = None
    TORCH_DISPONIBLE = False
    class _DummyModule:
        def __init__(self, *args, **kwargs): pass
        def to(self, *args, **kwargs): return self
        def eval(self, *args, **kwargs): return self
    class _DummyNN:
        Module = _DummyModule
    nn = _DummyNN()

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCALES_LOW = ['a', 'e', 'i', 'o', 'u']
CLASE_REPOSO = '---'

VOCAL_A_COLOR_HEX = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A',  # Amarillo
    '---': '#66FCF1' # Reposo / Silencio
}

VOCAL_A_RGBA = {
    'A': (0.902, 0.224, 0.275, 1.0),
    'E': (0.122, 0.467, 0.706, 1.0),
    'I': (0.173, 0.627, 0.173, 1.0),
    'O': (0.616, 0.306, 0.867, 1.0),
    'U': (0.906, 0.651, 0.102, 1.0)
}

NOMBRES_6CANALES_FACIALES = [
    "CH0: Apertura Mandibular (Digástrico)",
    "CH1: Tracción Sonrisa (Cigomático Mayor)",
    "CH2: Constricción / Protrusión Esfínter (Orbicular /o/, /u/)",
    "CH3: Retracción Bucal (Buccinador/Risorio)",
    "CH4: Ascenso Malar (Cigomático Menor)",
    "CH5: Compresión Maseterina (Masetero)"
]

# Distancias anatómicas canónicas de reposo (mm) para proyectar ratios relativos
# (\Delta d / d_{reposo, sujeto}) a unidades biomecánicas invariantes inter-sujeto
CANON_REST_DIST_MM = np.array([
    52.06666667,  # CH0: referencia ancho comisuras en reposo (mm)
    52.06666667,  # CH1: referencia ancho comisuras en reposo (mm)
    52.06666667,  # CH2: referencia ancho comisuras en reposo (mm)
    24.02434085,  # CH3: referencia distancia comisura-mejilla en reposo (mm)
    47.91229808,  # CH4: referencia distancia comisura-malar en reposo (mm)
    34.64579414   # CH5: referencia distancia comisura-masetero en reposo (mm)
], dtype=np.float64)

# Banda muerta (deadband) relativa al reposo (\Delta d / d_{reposo}):
# CH2 tiene umbral reducido (0.007 ~ 0.36 mm) para no recortar la protrusión/redondeo labial de /o/ y /u/
DEADBAND_RATIO_CANALES = np.array([
    0.014,  # CH0: ~0.72 mm eq (apertura labial mínima voluntaria)
    0.011,  # CH1: ~0.57 mm eq (tracción comisural)
    0.007,  # CH2: ~0.36 mm eq (constricción/protrusión orbicular sensible para /o/ y /u/)
    0.011,  # CH3: ~0.26 mm eq (retracción mejilla)
    0.010,  # CH4: ~0.48 mm eq (ascenso malar)
    0.011   # CH5: ~0.38 mm eq (compresión masetero)
], dtype=np.float64)

# Escalas y pesos biomecánicos calibrados de la Norma Cinemática S(t):
# Compensa la menor excursión horizontal 2D de la protrusión labial en /o/ y /u/ (CH2: w=2.25, s=3.6/15)
NORM_S_SCALES_6CH = np.array([12.5, 5.5, 3.6, 1.8, 1.8, 2.2], dtype=np.float64) / 15.0
NORM_S_WEIGHTS_6CH = np.array([1.00, 1.15, 2.25, 0.30, 0.30, 0.30], dtype=np.float64)


# ==============================================================================
# 0. ESTABILIZACIÓN DE LANDMARKS: ONE-EURO FILTER 3D + PROCRUSTES KABSCH ÓSEO
# ==============================================================================
class OneEuroLandmarkFilter3D:
    """
    Filtro adaptativo One-Euro (Casiez et al.) vectorizado sobre la malla 3D/2D de MediaPipe:
      f_c(t) = f_{c,min} + beta * ||dx/dt||
    - En reposo (||dx/dt|| ~ 0): aplica frecuencia de corte ultra-baja (f_{c,min} = 0.85 Hz),
      eliminando el temblor de 2 a 5 px de MediaPipe y clavando los puntos en la piel.
    - En articulación o giro rápido: abre instantáneamente el ancho de banda (> 18 Hz)
      para preservar transitorios rápidos (60-100 ms) sin retardo de fase.
    """
    def __init__(self, f_min=0.85, beta=0.055, f_d=1.0):
        self.f_min = float(f_min)
        self.beta = float(beta)
        self.f_d = float(f_d)
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    @staticmethod
    def _alpha(dt, f_c):
        tau = 1.0 / (2.0 * np.pi * np.maximum(f_c, 1e-4))
        return 1.0 / (1.0 + tau / max(float(dt), 1e-4))

    def filter(self, pts, t_now=None):
        x = np.asarray(pts, dtype=np.float64)
        if t_now is None:
            t_now = time.perf_counter()
        if self._x_prev is None or self._x_prev.shape != x.shape:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev = float(t_now)
            return x.copy()

        dt = float(t_now - self._t_prev)
        if dt <= 1e-5 or dt > 1.0:
            dt = 1.0 / 30.0
        self._t_prev = float(t_now)

        a_d = self._alpha(dt, self.f_d)
        dx_raw = (x - self._x_prev) / dt
        dx_hat = a_d * dx_raw + (1.0 - a_d) * self._dx_prev

        f_c = self.f_min + self.beta * np.abs(dx_hat)
        a_x = self._alpha(dt, f_c)
        x_hat = a_x * x + (1.0 - a_x) * self._x_prev

        self._x_prev = x_hat.copy()
        self._dx_prev = dx_hat.copy()
        return x_hat


class ProcrustesKabschHeadFrame:
    """
    Estabilizador de Cuerpo Rígido Craneal mediante Procrustes Ortogonal Ponderado (SVD Kabsch)
    sobre la cresta ósea nasal-frontal [1, 4, 6, 168, 10] (peso 1.0, inmune a lentes y a oclusión
    lateral) y cantos oculares [33, 133, 362, 263] (peso 0.35).
    Calcula además la protrusión labial anterior 3D (\Delta z_{labios}) para reforzar /o/ y /u/.
    """
    INDICES_ANCLAJE_OSEO = [1, 4, 6, 168, 10, 33, 133, 362, 263]
    PESOS_ANCLAJE_OSEO = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.35, 0.45, 0.45, 0.35], dtype=np.float64)

    def __init__(self):
        self.ref_template_mm = None
        self.ref_lip_z_mm = None
        self._dip_filter = OneEuroLandmarkFilter3D(f_min=0.25, beta=0.008, f_d=0.5)

    def reset(self):
        self.ref_template_mm = None
        self.ref_lip_z_mm = None
        self._dip_filter.reset()

    def project(self, puntos_2d_px, puntos_3d=None, d_ip_nominal_mm=63.0, ancho_img=640.0, t_now=None):
        p2d = np.asarray(puntos_2d_px, dtype=np.float64)
        n_pts = len(p2d)
        if n_pts >= 478:
            o_izq = p2d[468]
            o_der = p2d[473]
        else:
            o_izq = 0.5 * (p2d[33] + p2d[133])
            o_der = 0.5 * (p2d[263] + p2d[362])

        dip_2d = float(np.linalg.norm(o_der - o_izq))
        if puntos_3d is not None and len(puntos_3d) == n_pts:
            p3d = np.asarray(puntos_3d, dtype=np.float64)
            idx_r = 473 if n_pts >= 478 else 263
            idx_l = 468 if n_pts >= 478 else 33
            dz_px = float((p3d[idx_r, 2] - p3d[idx_l, 2]) * ancho_img)
        else:
            p3d = None
            dz_px = 0.0

        dip_raw = max(float(np.hypot(dip_2d, dz_px)), 10.0)
        dip_smooth = float(self._dip_filter.filter(np.array([dip_raw]), t_now=t_now)[0])
        escala = float(d_ip_nominal_mm) / max(dip_smooth, 10.0)

        if p3d is not None:
            z_mm = p3d[:, 2] * float(ancho_img) * escala
        else:
            z_mm = np.zeros(n_pts, dtype=np.float64)

        pts_m = np.column_stack([p2d[:, 0] * escala, p2d[:, 1] * escala, z_mm])

        # Marco ortonormal base centrado en la cresta ósea
        valid_idx = [idx for idx in self.INDICES_ANCLAJE_OSEO if idx < n_pts]
        if len(valid_idx) >= 5:
            w = self.PESOS_ANCLAJE_OSEO[:len(valid_idx)]
            w = w / np.sum(w)
            B_curr = pts_m[valid_idx]
            c_craneo = np.sum(B_curr * w[:, None], axis=0)
            B_centered = B_curr - c_craneo[None, :]

            # Construir orientación canónica inicial si no hay plantilla previa
            vx = o_der * escala - o_izq * escala
            ux = np.array([vx[0], vx[1], dz_px * escala], dtype=np.float64)
            ux /= max(float(np.linalg.norm(ux)), 1e-6)
            p_glab = pts_m[10] if n_pts > 10 else 0.5 * (pts_m[33] + pts_m[263])
            v_up = p_glab - pts_m[1]
            vy = v_up - np.dot(v_up, ux) * ux
            uy = vy / max(float(np.linalg.norm(vy)), 1e-6)
            uz = np.cross(ux, uy)
            uz /= max(float(np.linalg.norm(uz)), 1e-6)
            R_triad = np.column_stack([ux, uy, uz])

            if self.ref_template_mm is None or len(self.ref_template_mm) != len(valid_idx):
                self.ref_template_mm = B_centered @ R_triad
                R = R_triad
            else:
                # Ajuste Rígido Procrustes Ortogonal Ponderado (Kabsch SVD)
                H = (B_centered * w[:, None]).T @ self.ref_template_mm
                U, _, Vt = linalg.svd(H)
                d_sign = 1.0 if linalg.det(U @ Vt) >= 0 else -1.0
                S_corr = np.diag([1.0, 1.0, d_sign])
                R = U @ S_corr @ Vt
                # Adaptar plantilla suavemente en reposo para evitar deriva secular
                self.ref_template_mm = 0.995 * self.ref_template_mm + 0.005 * (B_centered @ R)
        else:
            c_craneo = pts_m[1] if n_pts > 1 else np.mean(pts_m, axis=0)
            R = np.eye(3, dtype=np.float64)

        r_local = (pts_m - c_craneo[None, :]) @ R

        # Protrusión labial 3D relativa al pronasale (en /o/ y /u/ los labios avanzan hacia -Z / +protrusión)
        prot_labial_mm = 0.0
        if n_pts > 14:
            lip_z_rel = 0.5 * (r_local[13, 2] + r_local[14, 2]) - r_local[1, 2]
            if self.ref_lip_z_mm is None:
                self.ref_lip_z_mm = float(lip_z_rel)
            ap_est = float(np.linalg.norm(r_local[13, :2] - r_local[14, :2]))
            if ap_est < 2.2:
                self.ref_lip_z_mm = 0.98 * self.ref_lip_z_mm + 0.02 * float(lip_z_rel)
            prot_labial_mm = max(0.0, float(self.ref_lip_z_mm - lip_z_rel))

        return {
            "r_local_mm": r_local,
            "c_craneo_mm": c_craneo,
            "R_craneo": R,
            "escala_mm_px": escala,
            "dip_px": dip_smooth,
            "protrusion_labial_mm": prot_labial_mm
        }


def filtro_ema_causal_1d(arr, alpha=0.70):
    """
    Filtro exponencial causal ligero (EMA con alpha >= 0.65) para preservar transitorios
    articulatorios rápidos (60-100 ms) sin el retardo de 100-150 ms de los filtros uniformes.
    """
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 0 or len(x) <= 1:
        return x
    a = float(np.clip(alpha, 0.65, 0.98))
    b_coeff = np.array([a], dtype=np.float64)
    a_coeff = np.array([1.0, -(1.0 - a)], dtype=np.float64)
    zi = signal.lfilter_zi(b_coeff, a_coeff) * x[0]
    y, _ = signal.lfilter(b_coeff, a_coeff, x, zi=zi)
    return y


def velocidad_diferencias_centrales_1d(arr, dt=0.01):
    """Calcula la velocidad instantánea d/dt r(t) mediante diferencias centrales de orden 2."""
    x = np.asarray(arr, dtype=np.float64)
    if x.ndim == 0 or len(x) < 3:
        return np.zeros_like(x)
    return np.gradient(x, float(dt), axis=-1)


def calcular_norma_cinematica_ponderada(canales_6ch):
    """
    Calcula la Norma Cinemática Ponderada S(t) sobre los 6 canales faciales con realce
    específico en CH2 (Constricción/Protrusión Orbicular de /o/ y /u/, w_2 = 2.25, s_2 = 3.6/15),
    compensando que en /o/ y /u/ los canales de mejilla/malar (CH3..CH5) permanecen en cero.
    """
    ch_mat = np.vstack([np.maximum(0.0, np.asarray(c, dtype=np.float64)) for c in canales_6ch])
    n_ch = ch_mat.shape[0]
    scales = NORM_S_SCALES_6CH[:n_ch, None]
    weights = NORM_S_WEIGHTS_6CH[:n_ch, None]
    s_raw = np.sqrt(np.sum(weights * ((ch_mat / scales) ** 2), axis=0))
    if n_ch >= 3:
        prim_act = np.maximum(ch_mat[0], np.maximum(ch_mat[2] * 2.10, ch_mat[1] * 1.15))
        active_gate = np.clip((prim_act - 0.45) / 0.85, 0.0, 1.0)
        return s_raw * active_gate
    return s_raw


def desacoplar_postura_craneal_en_reposo(ap_mm, com_mm, mej_mm, mal_mm, mas_mm, yaw_deg, pitch_deg, roll_deg):
    """
    Desacopla por regresión Ridge sobre los cuadros de reposo mandibular (ap_mm <= P25)
    cualquier fluctuación residual inducida por rotaciones de cabeza (yaw, pitch, roll).
    """
    ap_a = np.asarray(ap_mm, dtype=np.float64)
    com_a = np.asarray(com_mm, dtype=np.float64)
    mej_a = np.asarray(mej_mm, dtype=np.float64)
    mal_a = np.asarray(mal_mm, dtype=np.float64)
    mas_a = np.asarray(mas_mm, dtype=np.float64)
    poses = np.column_stack([
        np.asarray(yaw_deg, dtype=np.float64),
        np.asarray(pitch_deg, dtype=np.float64),
        np.asarray(roll_deg, dtype=np.float64)
    ])
    rest_mask = ap_a <= np.percentile(ap_a, 25)
    if np.sum(rest_mask) < 10:
        rest_mask = np.ones(len(ap_a), dtype=bool)
    pose_med = np.median(poses[rest_mask], axis=0, keepdims=True)
    P_c = poses - pose_med
    P_rest = P_c[rest_mask]
    G_pose = linalg.solve(P_rest.T @ P_rest + 10.0 * np.eye(3), P_rest.T)

    betas = {}
    out_arrs = []
    for name, arr in [('ap', ap_a), ('com', com_a), ('mej', mej_a), ('mal', mal_a), ('mas', mas_a)]:
        y_r = arr[rest_mask] - np.median(arr[rest_mask])
        beta = G_pose @ y_r
        betas[name] = beta
        out_arrs.append(arr - (P_c @ beta))

    return out_arrs[0], out_arrs[1], out_arrs[2], out_arrs[3], out_arrs[4], {
        'rest_mask': rest_mask,
        'pose_med': pose_med,
        'P_centered': P_c,
        'betas': betas
    }


def calcular_canales_relativos_invariantes(
    ap_mm, com_mm, d_mej_mm, d_mal_mm, d_mas_mm,
    baseline_stats=None, aplicar_deadband=True, devolver_en_mm_canonicos=True,
    protrusion_labial_mm=None
):
    """
    Transforma las 5 distancias anatómicas faciales (ya proyectadas al marco craneal local)
    en 6 canales articulatorios invariantes a la fisionomía absoluta del sujeto mediante
    ratios relativos al propio reposo del usuario (\Delta d / d_{reposo}).
    Si se dispone de protrusion_labial_mm (avance 3D de los labios respecto al pronasale),
    refuerza el Canal 2 (Constricción/Protrusión Orbicular) para /o/ y /u/.
    """
    ap_arr = np.asarray(ap_mm, dtype=np.float64)
    com_arr = np.asarray(com_mm, dtype=np.float64)
    mej_arr = np.asarray(d_mej_mm, dtype=np.float64)
    mal_arr = np.asarray(d_mal_mm, dtype=np.float64)
    mas_arr = np.asarray(d_mas_mm, dtype=np.float64)

    es_escalar = (ap_arr.ndim == 0)
    if baseline_stats is None or not isinstance(baseline_stats, dict):
        if es_escalar:
            b_ap, b_com, b_mej, b_mal, b_mas = float(ap_arr), float(com_arr), float(mej_arr), float(mal_arr), float(mas_arr)
        else:
            rest_m = ap_arr <= np.percentile(ap_arr, 25)
            if not np.any(rest_m):
                rest_m = np.ones_like(ap_arr, dtype=bool)
            b_ap = float(np.percentile(ap_arr, 10))
            b_com = float(np.median(com_arr[rest_m]))
            b_mej = float(np.median(mej_arr[rest_m]))
            b_mal = float(np.median(mal_arr[rest_m]))
            b_mas = float(np.median(mas_arr[rest_m]))
    else:
        if not es_escalar:
            rest_m = ap_arr <= np.percentile(ap_arr, 25)
            if not np.any(rest_m):
                rest_m = np.ones_like(ap_arr, dtype=bool)
        b_ap = float(baseline_stats.get('base_ap', np.percentile(ap_arr, 10) if not es_escalar else float(ap_arr)))
        b_com = float(baseline_stats.get('base_com', np.median(com_arr[rest_m]) if not es_escalar else float(com_arr)))
        b_mej = float(baseline_stats.get('base_mej', np.median(mej_arr[rest_m]) if not es_escalar else float(mej_arr)))
        b_mal = float(baseline_stats.get('base_mal', np.median(mal_arr[rest_m]) if not es_escalar else float(mal_arr)))
        b_mas = float(baseline_stats.get('base_mas', np.median(mas_arr[rest_m]) if not es_escalar else float(mas_arr)))

    ref_com = max(abs(b_com), 25.0)
    ref_mej = max(abs(b_mej), 10.0)
    ref_mal = max(abs(b_mal), 20.0)
    ref_mas = max(abs(b_mas), 15.0)

    r1 = np.maximum(0.0, ap_arr - b_ap) / ref_com
    com_rel = (com_arr - b_com) / ref_com
    r2 = np.maximum(0.0, com_rel)
    prot_term = (0.55 * np.maximum(0.0, np.asarray(protrusion_labial_mm, dtype=np.float64)) / ref_com) if protrusion_labial_mm is not None else 0.0
    r3 = np.maximum(0.0, -com_rel) + prot_term
    r4 = np.maximum(0.0, -(mej_arr - b_mej)) / ref_mej
    r5 = np.maximum(0.0, -(mal_arr - b_mal)) / ref_mal
    r6 = np.maximum(0.0, -(mas_arr - b_mas)) / ref_mas

    ratios = [r1, r2, r3, r4, r5, r6]
    if aplicar_deadband:
        ratios = [
            np.maximum(0.0, r - DEADBAND_RATIO_CANALES[idx])
            for idx, r in enumerate(ratios)
        ]

    if devolver_en_mm_canonicos:
        canales = [ratios[idx] * CANON_REST_DIST_MM[idx] for idx in range(6)]
    else:
        canales = ratios

    if es_escalar:
        return np.array([float(c) for c in canales], dtype=np.float32)
    return [np.asarray(c, dtype=np.float64) for c in canales]


# ==============================================================================
# 1. ARQUITECTURAS PYTORCH Y GEOMÉTRICAS ORTOGONALES (6 CANALES: 2D, 3D Y 4D)
# ==============================================================================
class GeoSupConvOrthoNet(nn.Module):
    """
    Red Récord Supervisada con Fronteras Geométricas Lineales, Desacople de Coarticulación
    y Proyección Ortogonal a la Postura Craneal (GeoSupConvOrthoAE 2D / 3D / 4D).
    """
    def __init__(self, in_channels=6, time_pts=25, latent_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        if TORCH_DISPONIBLE and torch is not None:
            self.conv_pos = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            self.conv_vel = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
            self.conv_acc = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)

    @staticmethod
    def build_multiscale_features_np(X_mat, ch_mean, ch_std, X_iso_anchor=None, k_iso=16, iso_cache=None, feature_version=3):
        N = X_mat.shape[0]
        X_3d = X_mat.reshape(N, 6, 25).astype(np.float64)
        X_flat_scaled = X_3d.reshape(N, -1)
        X_norm = (X_3d - ch_mean) / ch_std
        k_pos = np.array([0.2, 0.6, 0.2])
        k_vel = np.array([-0.5, 0.0, 0.5])
        k_acc = np.array([0.25, -0.5, 0.25])
        pos = scipy.ndimage.convolve1d(X_norm, k_pos, axis=2, mode='nearest')
        vel = scipy.ndimage.convolve1d(X_norm, k_vel, axis=2, mode='nearest') * 2.2
        acc = scipy.ndimage.convolve1d(X_norm, k_acc, axis=2, mode='nearest') * 2.8
        c_max = np.max(pos, axis=2)
        c_p90 = np.percentile(pos, 90, axis=2)
        c_mean = np.mean(pos[:, :, 5:20], axis=2)
        c_energy = np.sqrt(np.mean(pos[:, :, 4:21]**2, axis=2))
        c_vel_pk = np.max(np.abs(vel[:, :, 2:22]), axis=2)

        fv = int(feature_version)
        if fv >= 3:
            edge_base = 0.5 * (np.percentile(pos[:, :, :5], 25, axis=2) + np.percentile(pos[:, :, 20:], 25, axis=2))
            onset_base = np.mean(pos[:, :, :5], axis=2)
            exc_core = np.max(pos[:, :, 6:19], axis=2) - edge_base
            mean_core = np.mean(pos[:, :, 7:18], axis=2) - edge_base
            delta_onset = np.mean(pos[:, :, 8:17], axis=2) - onset_base
            vel_attack = np.mean(vel[:, :, 5:12], axis=2) - np.mean(vel[:, :, 14:21], axis=2)

            raw_core = np.maximum(0.0, X_3d[:, :, 5:20])
            ch_peak_core = np.max(raw_core, axis=2)
            ch_mean_core = np.mean(raw_core[:, :, 2:13], axis=2)
            sup_core = np.max(ch_peak_core, axis=1, keepdims=True) + 0.35
            sup_ratios = ch_peak_core / sup_core
            sup_mean_ratios = ch_mean_core / (np.max(ch_mean_core, axis=1, keepdims=True) + 0.35)

            rc = np.mean(np.maximum(0.0, X_3d[:, :, 7:18]), axis=2)
            r_10 = rc[:, 1:2] / (rc[:, 0:1] + 1.2)
            r_20 = rc[:, 2:3] / (rc[:, 0:1] + 1.2)
            r_21 = (rc[:, 2:3] - rc[:, 1:2]) / (rc[:, 1:2] + rc[:, 2:3] + 1.0)

            base_feats = np.hstack([
                pos.reshape(N, -1), vel.reshape(N, -1), acc.reshape(N, -1),
                c_max * 2.5, c_p90 * 2.5, c_mean * 2.5, c_energy * 2.5, c_vel_pk * 1.5,
                exc_core * 2.4, mean_core * 2.4, delta_onset * 1.8, vel_attack * 1.6,
                r_10 * 2.5, r_20 * 2.8, r_21 * 3.2,
                sup_ratios * 3.5, sup_mean_ratios * 2.5
            ])
            iso_w = 0.25
        elif fv == 2:
            edge_base = 0.5 * (np.percentile(pos[:, :, :5], 25, axis=2) + np.percentile(pos[:, :, 20:], 25, axis=2))
            exc_core = np.max(pos[:, :, 6:19], axis=2) - edge_base
            mean_core = np.mean(pos[:, :, 7:18], axis=2) - edge_base
            raw_core = np.maximum(0.0, X_3d[:, :, 5:20])
            ch_peak_core = np.max(raw_core, axis=2)
            sup_core = np.max(ch_peak_core, axis=1, keepdims=True) + 0.35
            sup_ratios = ch_peak_core / sup_core
            rc = np.mean(np.maximum(0.0, X_3d[:, :, 7:18]), axis=2)
            r_10 = rc[:, 1:2] / (rc[:, 0:1] + 1.2)
            r_20 = rc[:, 2:3] / (rc[:, 0:1] + 1.2)
            r_21 = (rc[:, 2:3] - rc[:, 1:2]) / (rc[:, 1:2] + rc[:, 2:3] + 1.0)
            base_feats = np.hstack([
                pos.reshape(N, -1), vel.reshape(N, -1), acc.reshape(N, -1),
                c_max * 2.5, c_p90 * 2.5, c_mean * 2.5, c_energy * 2.5, c_vel_pk * 1.5,
                exc_core * 2.2, mean_core * 2.2, r_10 * 2.5, r_20 * 2.5, r_21 * 3.0,
                sup_ratios * 3.5
            ])
            iso_w = 0.25
        else:
            base_feats = np.hstack([
                pos.reshape(N, -1), vel.reshape(N, -1), acc.reshape(N, -1),
                c_max * 2.5, c_p90 * 2.5, c_mean * 2.5, c_energy * 2.5, c_vel_pk * 1.5
            ])
            iso_w = 0.15

        if iso_cache is not None:
            D_q = spatial.distance.cdist(X_flat_scaled, iso_cache['X_anchor'], metric='euclidean')
            Mq = len(X_flat_scaled)
            N_a = len(iso_cache['X_anchor'])
            D_geo_q = np.zeros((Mq, N_a))
            for i in range(Mq):
                idx_k = np.argsort(D_q[i])[:k_iso]
                d_via = D_q[i, idx_k][:, None] + iso_cache['D_geo'][idx_k, :]
                D_geo_q[i] = np.min(d_via, axis=0)
            D2_q = D_geo_q ** 2
            q_mean = np.mean(D2_q, axis=1, keepdims=True)
            B_q = -0.5 * (D2_q - q_mean - iso_cache['col_RowMean'][None, :] + iso_cache['grand_mean'])
            z_iso = B_q @ iso_cache['proj_vecs']
            return np.hstack([base_feats, z_iso * iso_w])
        return base_feats


# ==============================================================================
# 2. MOTOR DE DECODIFICACIÓN FACIAL DE 6 CANALES EN TIEMPO REAL
# ==============================================================================
class RealtimeDecoderEngine:
    """
    Motor unificado de decodificación cinemática facial en tiempo real (6 canales + Mic opcional)
    con soporte para:
      - Espacio latente 4D (desacople /e/ vs /o/ y ortogonalidad a yaw/pitch/roll), 3D y 2D
      - Norma Cinemática Ponderada S(t) con realce de Constricción/Protrusión Labial (CH2)
        para que /o/ y /u/ disparen ventanas con la misma sensibilidad que /a/, /e/, /i/
      - Segmentación por Valles de Inflexión Cinemática (\dot{S}(t) = 0) para habla continua
        coarticulada sin retorno a reposo entre fonemas
      - Estado explícito de Silencio/Reposo ('---') para colapsar ruido residual.
    """

    def __init__(
        self,
        model_path=None,
        proyecciones_path=None,
        sample_rate=100,
        buffer_sec=8.0,
        target_len=25,
        n_canales=6,
        emd_delay_ms=71.0,
        trigger_mode='GATE_HYBRID',
        pre_win_ms=500,
        post_win_ms=500,
        ema_alpha=0.72,
        device='cpu'
    ):
        self.fs = float(sample_rate)
        self.target_len = int(target_len)
        self.n_canales = int(n_canales)
        self.device = device
        self.trigger_mode = str(trigger_mode).upper()
        self.ema_alpha = max(0.65, float(ema_alpha))
        self.input_is_ratio = False

        self.default_shifts_ms = {
            'GATE_HYBRID': 70.9,
            'GATE_CENTROID': 64.5,
            'GATE_CFD55': 268.5,
            'GATE_ONSET': 416.5,
            'GATE_NOSHIFT': 0.0,
            'MIC_SYNC': 0.0
        }
        self.emd_delay_ms = float(emd_delay_ms)
        self.emd_samples = int(round((self.emd_delay_ms / 1000.0) * self.fs))

        self.buffer_len = int(buffer_sec * self.fs)
        self.pre_samples = int((pre_win_ms / 1000.0) * self.fs)
        self.post_samples = int((post_win_ms / 1000.0) * self.fs)
        self.total_win_samples = self.pre_samples + self.post_samples
        self.t_refr_samples = int(1.15 * self.fs)
        self.max_lookback_samples = int(0.42 * self.fs)

        # Escalas y pesos biomecánicos con realce del Canal 2 (Constricción/Protrusión Orbicular /o/ y /u/)
        self.s_scales = NORM_S_SCALES_6CH[:self.n_canales].astype(np.float32)
        self.s_weights = NORM_S_WEIGHTS_6CH[:self.n_canales].astype(np.float32)
        self.auto_calibrate_gate = False
        self.allow_mic_monitor = False

        self.ckpt = None
        self.model = None
        self.iso_cache = None
        self.latent_dim = 4
        self.feature_version = 3
        self.cluster_to_vocal = {i: i for i in range(5)}
        self.centroids = None
        self.inv_pooled_cov = None
        self.pooled_cov = None
        self.rest_centroid = np.zeros(4, dtype=np.float64)
        self.gmm = None

        self.ref_z1 = []
        self.ref_z2 = []
        self.ref_z3 = []
        self.ref_z4 = []
        self.ref_vocales = []

        self.reset_state()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        self.set_trigger_mode(self.trigger_mode, auto_update_shift=True)
        if proyecciones_path and os.path.exists(proyecciones_path):
            self.load_reference_projections(proyecciones_path)

    def reset_state(self):
        """Reinicia los buffers circulares y la máquina de estados del Gate Doble + Detector de Valles."""
        self.raw_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.filt_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.env_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.vel_buffer = np.zeros((self.n_canales, self.buffer_len), dtype=np.float32)
        self.s_emg_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self.s_vel_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self.mic_buffer = np.zeros(self.buffer_len, dtype=np.float32)
        self._ema_prev = np.zeros(self.n_canales, dtype=np.float32)

        self.mic_sync_enabled = (self.trigger_mode == 'MIC_SYNC')
        self.ultimo_pico_mic_sample = -self.t_refr_samples
        self.total_samples_received = 0

        self.u_bajo = 4.20
        self.u_alto = 9.20
        self.med_base = 0.05
        self.sig_base = 0.15
        self.ruido_base_canales = np.zeros((self.n_canales, 1), dtype=np.float32)
        self.calibrado = True
        self.auto_calibrate_gate = False

        self.estado_gate = 'ESPERANDO_REPOSO'
        self.ultimo_onset_sample = -self.t_refr_samples
        self.ultimo_center_sample = -self.t_refr_samples
        self.onset_sample_actual = -1
        self.confirm_sample_actual = -1
        self.center_sample_actual = -1
        self.close_sample_actual = -1
        self.fin_sample_actual = -1
        self._peak_s_actual = 0.0
        self._min_post_peak_s = 999.0
        self._min_post_peak_sample = -1
        self._thresh_low_actual = 4.20
        self._anchor_computed = False
        self.last_window_info = None

    def set_calibration_preset(self, preset_name: str):
        self.set_trigger_mode('GATE_HYBRID', auto_update_shift=True)

    def set_trigger_mode(self, mode: str, auto_update_shift: bool = True):
        m = str(mode).upper()
        self.trigger_mode = m
        self.mic_sync_enabled = (m == 'MIC_SYNC')
        if auto_update_shift and m in self.default_shifts_ms:
            self.set_emd_delay_ms(self.default_shifts_ms[m])

    def set_emd_delay_ms(self, emd_ms: float):
        self.emd_delay_ms = float(emd_ms)
        self.emd_samples = int(round((self.emd_delay_ms / 1000.0) * self.fs))

    def set_mic_sync_enabled(self, enabled: bool):
        self.mic_sync_enabled = bool(enabled)
        if enabled:
            self.trigger_mode = 'MIC_SYNC'
        elif self.trigger_mode == 'MIC_SYNC':
            self.trigger_mode = 'GATE_HYBRID'

    def set_rms_window_ms(self, win_ms: float):
        pass

    def recalibrar_ruido(self, duracion_sec=5.0):
        n_s = min(int(duracion_sec * self.fs), self.buffer_len, max(10, self.total_samples_received))
        sub_s = self.s_emg_buffer[-n_s:]
        self.med_base = float(np.clip(np.percentile(sub_s, 30), 0.0, 1.5))
        peak_p95 = float(np.percentile(sub_s, 95))
        low_mask = sub_s <= np.percentile(sub_s, 65)
        sub_low = sub_s[low_mask] if np.any(low_mask) else sub_s
        mad = float(np.median(np.abs(sub_low - np.median(sub_low))))
        self.sig_base = 1.4826 * mad + 1e-4
        if peak_p95 >= 8.0:
            self.u_bajo = float(np.clip(self.med_base + 0.18 * (peak_p95 - self.med_base), 3.0, 6.2))
            self.u_alto = float(np.clip(self.med_base + 0.35 * (peak_p95 - self.med_base), 6.2, 12.0))
        else:
            self.u_bajo = float(np.clip(max(3.2, self.med_base + 3.0 * self.sig_base), 3.0, 5.5))
            self.u_alto = float(np.clip(max(7.0, self.med_base + 6.0 * self.sig_base), 6.2, 11.0))
        self.calibrado = True
        self.estado_gate = 'ARMADO'
        return True

    def ajustar_umbral_por_falso_positivo(self, factor_incremento=1.08):
        self.u_alto = min(18.0, float(self.u_alto * factor_incremento))
        self.u_bajo = min(9.5, float(self.u_bajo * (0.5 + 0.5 * factor_incremento)))
        return self.u_bajo, self.u_alto

    def _build_isomap_cache(self, X_anchor, k_iso=16):
        D_tr = spatial.distance.cdist(X_anchor, X_anchor, metric='euclidean')
        N = len(X_anchor)
        G = np.full((N, N), np.inf)
        for i in range(N):
            idx_k = np.argsort(D_tr[i])[1:k_iso+1]
            for j in idx_k:
                G[i, j] = D_tr[i, j]
                G[j, i] = D_tr[j, i]
        D_geo = shortest_path(G, method='D', directed=False)
        max_d = np.max(D_geo[np.isfinite(D_geo)])
        D_geo[~np.isfinite(D_geo)] = max_d * 1.5
        D2 = D_geo ** 2
        H = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * H @ D2 @ H
        evals, evecs = linalg.eigh(B)
        idx_top = np.argsort(evals)[::-1][:10]
        pos_evals = np.maximum(evals[idx_top], 1e-6)
        return {
            'X_anchor': X_anchor,
            'D_geo': D_geo,
            'col_RowMean': np.mean(D2, axis=1),
            'grand_mean': np.mean(D2),
            'proj_vecs': evecs[:, idx_top] / np.sqrt(pos_evals)
        }

    def load_model(self, pt_path):
        if not pt_path or not os.path.exists(pt_path):
            return False
        try:
            if pt_path.endswith('.npz'):
                data = np.load(pt_path, allow_pickle=True)
                ckpt = {k: data[k].item() if data[k].ndim == 0 else data[k] for k in data.files}
            else:
                with open(pt_path, 'rb') as f:
                    ckpt = pickle.load(f)
        except Exception as e:
            print(f"[ERROR RealtimeDecoderEngine.load_model] {e}")
            return False

        self.ckpt = ckpt
        self.latent_dim = int(ckpt.get('latent_dim', 3))
        w_rows = int(np.asarray(ckpt['W_enc']).shape[0])
        if 'feature_version' in ckpt:
            self.feature_version = int(ckpt['feature_version'])
        else:
            self.feature_version = 3 if w_rows >= 530 else (2 if w_rows >= 500 else 1)

        self.centroids = np.array(ckpt['gmm_means'], dtype=np.float64)
        self.inv_pooled_cov = np.array(ckpt['inv_pooled_cov'], dtype=np.float64)
        self.pooled_cov = np.array(ckpt['gmm_covariances'][0], dtype=np.float64)
        self.cluster_to_vocal = {int(k): int(v) for k, v in ckpt.get('cluster_to_vocal', {i: i for i in range(5)}).items()}

        class MockGMM:
            def __init__(self, means):
                self.means_ = means
        self.gmm = MockGMM(self.centroids)

        if 'shift_hybrid_ms' in ckpt:
            self.default_shifts_ms['GATE_HYBRID'] = float(ckpt['shift_hybrid_ms'])
            self.default_shifts_ms['GATE_CFD55'] = float(ckpt.get('shift_cfd55_ms', 268.5))
            self.default_shifts_ms['GATE_ONSET'] = float(ckpt.get('shift_onset_ms', 416.5))
            self.default_shifts_ms['GATE_CENTROID'] = float(ckpt.get('shift_centroid_ms', 64.5))

        if 'X_train_iso_anchor' in ckpt:
            self.iso_cache = self._build_isomap_cache(
                np.array(ckpt['X_train_iso_anchor'], dtype=np.float64),
                k_iso=int(ckpt.get('k_iso', 16))
            )

        try:
            F_zero = GeoSupConvOrthoNet.build_multiscale_features_np(
                np.zeros((1, 150), dtype=np.float64),
                ch_mean=self.ckpt['ch_mean'],
                ch_std=self.ckpt['ch_std'],
                k_iso=int(self.ckpt.get('k_iso', 16)),
                iso_cache=self.iso_cache,
                feature_version=self.feature_version
            )
            self.rest_centroid = ((F_zero - self.ckpt['mu']) @ self.ckpt['W_enc'])[0]
        except Exception:
            self.rest_centroid = np.zeros(self.latent_dim, dtype=np.float64)

        self.model = True

        dir_m = os.path.dirname(pt_path)
        nom_m = os.path.splitext(os.path.basename(pt_path))[0]
        csv_cand = os.path.join(dir_m, f"proyecciones_{nom_m}.csv")
        if os.path.exists(csv_cand):
            self.load_reference_projections(csv_cand)
        return True

    def load_reference_projections(self, csv_path):
        if not os.path.exists(csv_path):
            return False
        df = pd.read_csv(csv_path)
        self.ref_z1 = df['Z1'].values.tolist()
        self.ref_z2 = df['Z2'].values.tolist()
        self.ref_z3 = df['Z3'].values.tolist() if 'Z3' in df.columns else [0.0]*len(df)
        self.ref_z4 = df['Z4'].values.tolist() if 'Z4' in df.columns else [0.0]*len(df)
        self.ref_vocales = df['Vocal'].values.tolist()
        return True

    def _preprocess_window_anti_bias_i(self, X_3d_in):
        w_out = np.maximum(0.0, X_3d_in.astype(np.float64)).copy()
        norm_ch0 = w_out / self.s_scales[None, :, None]
        s_win0 = np.sqrt(np.sum(self.s_weights[None, :, None] * (norm_ch0 ** 2), axis=1))
        pk_s0 = np.max(s_win0, axis=1)

        if getattr(self, 'feature_version', 3) >= 2:
            gain_v2 = np.where((pk_s0 >= 4.2) & (pk_s0 < 12.0), np.clip(14.5 / np.maximum(pk_s0, 4.2), 1.0, 2.15), 1.0)[:, None, None]
            return (w_out * gain_v2).reshape(len(w_out), -1), pk_s0

        pk0 = np.max(w_out[:, 0, :], axis=1)
        pk1 = np.max(w_out[:, 1, :], axis=1)
        pk2 = np.max(w_out[:, 2, :], axis=1)
        ratio_10 = pk1 / (pk0 + 1.0)
        pk34 = np.max(w_out[:, 3:5, :], axis=(1, 2))
        for idx in range(len(w_out)):
            if pk2[idx] < 0.6 and pk0[idx] >= 4.0:
                if ratio_10[idx] < 0.30 and (pk0[idx] < 13.5 or pk34[idx] > 0.25):
                    w_out[idx, 3:5, :] = np.minimum(w_out[idx, 3:5, :], 0.08)
                    sc = max(1.0, 18.2 / max(pk0[idx], 4.0))
                    w_out[idx, 0:2, :] *= sc
                elif pk_s0[idx] < 15.5 and 0.30 <= ratio_10[idx] < 0.44 and pk0[idx] < 15.0:
                    w_out[idx, 3:5, :] = np.minimum(w_out[idx, 3:5, :], 0.12)
                    sc = max(1.0, 15.2 / max(pk0[idx], 4.0))
                    w_out[idx, 0:2, :] *= sc

        norm_ch = w_out / self.s_scales[None, :, None]
        s_win = np.sqrt(np.sum(self.s_weights[None, :, None] * (norm_ch ** 2), axis=1))
        pk_s = np.max(s_win, axis=1)
        gain = np.where((pk_s >= 5.0) & (pk_s < 16.5), np.clip(19.5 / np.maximum(pk_s, 5.0), 1.0, 2.1), 1.0)[:, None, None]
        return (w_out * gain).reshape(len(w_out), -1), pk_s0

    def encode_and_predict_window(self, x_150_vec, return_rest_prob=False):
        if x_150_vec.ndim == 1:
            X_in = x_150_vec.reshape(1, -1)
        else:
            X_in = x_150_vec

        if self.ckpt is None:
            Z_zero = np.zeros((len(X_in), self.latent_dim))
            probs = np.ones((len(X_in), 5)) / 5.0
            return np.zeros(len(X_in), dtype=int), probs, Z_zero

        X_3d = X_in.reshape(len(X_in), 6, 25)
        X_proc, peak_s = self._preprocess_window_anti_bias_i(X_3d)

        F_feat = GeoSupConvOrthoNet.build_multiscale_features_np(
            X_proc,
            ch_mean=self.ckpt['ch_mean'],
            ch_std=self.ckpt['ch_std'],
            k_iso=int(self.ckpt.get('k_iso', 16)),
            iso_cache=self.iso_cache,
            feature_version=self.feature_version
        )
        Z = ((F_feat - self.ckpt['mu']) @ self.ckpt['W_enc']).copy()

        gate_gain = np.clip((peak_s - 3.0) / 4.2, 0.0, 1.0)[:, None]
        Z = Z * gate_gain

        scores = np.column_stack([
            -0.5 * np.sum(((Z - self.centroids[c]) @ self.inv_pooled_cov) * (Z - self.centroids[c]), axis=1)
            for c in range(5)
        ])
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)

        if return_rest_prob:
            rest_mask = peak_s < max(4.2, self.u_bajo * 0.85)
            preds = np.where(rest_mask, -1, preds)
        return preds, probs, Z

    def compute_decision_grid(self, x_range=(-3.2, 3.2), y_range=(-3.2, 3.2), resolution=160, proj_axes=(0, 1)):
        if self.centroids is None or self.inv_pooled_cov is None:
            return None, None, None

        ax_x = min(int(proj_axes[0]), self.latent_dim - 1)
        ax_y = min(int(proj_axes[1]), self.latent_dim - 1)
        if ax_x == ax_y:
            ax_y = min(1, self.latent_dim - 1)

        x_lin = np.linspace(x_range[0], x_range[1], resolution)
        y_lin = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(x_lin, y_lin)
        pts_2d = np.column_stack([xx.ravel(), yy.ravel()])

        if self.pooled_cov is not None and self.pooled_cov.shape[0] > max(ax_x, ax_y):
            sub_cov = self.pooled_cov[np.ix_([ax_x, ax_y], [ax_x, ax_y])] + 0.02 * np.eye(2)
            inv_cov_2d = linalg.inv(sub_cov)
        else:
            inv_cov_2d = np.eye(2)

        cen_2d = self.centroids[:, [ax_x, ax_y]]
        scores = np.column_stack([
            -0.5 * np.sum(((pts_2d - cen_2d[c]) @ inv_cov_2d) * (pts_2d - cen_2d[c]), axis=1)
            for c in range(5)
        ])
        probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        clusters = np.argmax(probs, axis=1)
        max_prob = np.max(probs, axis=1)

        rgba_img = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        for v_idx, v_name in enumerate(VOCALES):
            mask = (clusters == v_idx)
            color_norm = VOCAL_A_RGBA[v_name]
            r, g, b = int(color_norm[0]*255), int(color_norm[1]*255), int(color_norm[2]*255)
            alpha = (55 + 55 * max_prob[mask]).astype(np.uint8)
            grid_v_idx = np.where(mask)[0]
            rows = grid_v_idx // resolution
            cols = grid_v_idx % resolution
            rgba_img[rows, cols, 0] = r
            rgba_img[rows, cols, 1] = g
            rgba_img[rows, cols, 2] = b
            rgba_img[rows, cols, 3] = alpha

        return rgba_img, x_range, y_range

    def push_chunk(self, chunk, mic_chunk=None):
        """
        Ingesta en streaming de los 6 canales cinemáticos (6, n_chunk) con:
          - Norma Cinemática Ponderada S(t) con realce en CH2 (w_2=2.25) para /o/ y /u/
          - Gate Doble con soporte para segmentación por Valles de Inflexión Cinemática
            (\dot{S}(t) = 0) en habla continua coarticulada sin retorno a reposo.
        """
        n_chunk = chunk.shape[1]
        if n_chunk == 0:
            return [], np.array([]), self.get_telemetry()

        t_inicio = time.perf_counter()
        ch_in = chunk[:self.n_canales].astype(np.float32)

        if getattr(self, "input_is_ratio", False):
            ch_in = ch_in * CANON_REST_DIST_MM[:self.n_canales, None].astype(np.float32)

        alpha_ema = float(self.ema_alpha)
        ch_ema = np.empty_like(ch_in)
        for k_s in range(n_chunk):
            self._ema_prev = alpha_ema * ch_in[:, k_s] + (1.0 - alpha_ema) * self._ema_prev
            ch_ema[:, k_s] = self._ema_prev

        self.raw_buffer = np.roll(self.raw_buffer, -n_chunk, axis=1)
        self.raw_buffer[:, -n_chunk:] = ch_in
        self.filt_buffer = np.roll(self.filt_buffer, -n_chunk, axis=1)
        self.filt_buffer[:, -n_chunk:] = ch_ema
        self.env_buffer = np.roll(self.env_buffer, -n_chunk, axis=1)
        self.env_buffer[:, -n_chunk:] = np.maximum(0.0, ch_ema)

        tail_len = min(self.buffer_len, n_chunk + 4)
        vel_tail = velocidad_diferencias_centrales_1d(self.env_buffer[:, -tail_len:], dt=1.0 / self.fs)
        self.vel_buffer = np.roll(self.vel_buffer, -n_chunk, axis=1)
        self.vel_buffer[:, -n_chunk:] = vel_tail[:, -n_chunk:]

        if (self.mic_sync_enabled or getattr(self, "allow_mic_monitor", False)) and mic_chunk is not None:
            mic_flat = np.asarray(mic_chunk, dtype=np.float32).flatten()
            self.mic_buffer = np.roll(self.mic_buffer, -n_chunk)
            self.mic_buffer[-n_chunk:] = mic_flat
        else:
            self.mic_buffer = np.roll(self.mic_buffer, -n_chunk)
            self.mic_buffer[-n_chunk:] = 0.0

        # Estimación adaptativa del piso de reposo por canal cuando mandíbula y esfínter están quietos
        n_hist = min(self.buffer_len, self.total_samples_received + n_chunk)
        if n_hist >= 4:
            sub_env = self.env_buffer[:, -n_hist:]
            rest_m = (sub_env[0] <= max(1.2, float(np.percentile(sub_env[0], 30)))) & (sub_env[2] <= max(1.0, float(np.percentile(sub_env[2], 35))))
            if np.any(rest_m):
                max_f = np.array([0.45, 1.8, 0.35, 0.45, 0.45, 0.45], dtype=np.float32)[:self.n_canales, None]
                fl_est = np.array([np.percentile(sub_env[c, rest_m], 60) for c in range(self.n_canales)], dtype=np.float32)[:, None]
                self.ruido_base_canales = np.minimum(fl_est, max_f)

        ch_clean = np.maximum(0.0, ch_ema - self.ruido_base_canales)
        norm_ch = ch_clean / self.s_scales[:, None]
        s_pos = np.sqrt(np.sum(self.s_weights[:, None] * (norm_ch ** 2), axis=0))
        # Compuerta proporcional con realce 2.1x en CH2 (protrusión/constricción labial de /o/ y /u/)
        prim_act = np.maximum(ch_clean[0], np.maximum(ch_clean[2] * 2.10, ch_clean[1] * 1.15))
        active_gate = np.clip((prim_act - 0.45) / 0.85, 0.0, 1.0)
        s_nuevo = s_pos * active_gate

        self.s_emg_buffer = np.roll(self.s_emg_buffer, -n_chunk)
        self.s_emg_buffer[-n_chunk:] = s_nuevo
        s_tail_vel = velocidad_diferencias_centrales_1d(self.s_emg_buffer[-tail_len:], dt=1.0 / self.fs)
        self.s_vel_buffer = np.roll(self.s_vel_buffer, -n_chunk)
        self.s_vel_buffer[-n_chunk:] = s_tail_vel[-n_chunk:]

        if getattr(self, "auto_calibrate_gate", False) and n_hist >= int(3.0 * self.fs):
            sub_s_all = self.s_emg_buffer[-n_hist:]
            p95_s = float(np.percentile(sub_s_all, 95))
            p25_s = float(np.clip(np.percentile(sub_s_all, 25), 0.0, 1.5))
            self.med_base = p25_s
            if 6.0 <= p95_s < 16.0:
                tgt_u_alto = float(np.clip(p25_s + 0.40 * (p95_s - p25_s), 4.5, 9.2))
                tgt_u_bajo = float(np.clip(p25_s + 0.20 * (p95_s - p25_s), 2.2, 4.2))
                self.u_alto = 0.94 * self.u_alto + 0.06 * tgt_u_alto
                self.u_bajo = 0.94 * self.u_bajo + 0.06 * tgt_u_bajo

        samples_base = self.total_samples_received
        end_stream_s = samples_base + n_chunk
        eventos = []

        # MODO 1: Sincronización por Micrófono (MIC_SYNC)
        if self.mic_sync_enabled and mic_chunk is not None and end_stream_s >= int(1.5 * self.fs):
            recent_s = min(int(3.0 * self.fs), self.buffer_len)
            sub_mic = self.mic_buffer[-recent_s:]
            if float(np.max(sub_mic)) > 0.08:
                th_mic = max(0.08, np.percentile(sub_mic, 80) * 0.55)
                pks, _ = signal.find_peaks(sub_mic, distance=int(0.95 * self.fs), height=th_mic)
                wait_s = self.post_samples + int(0.10 * self.fs)
                for p_sub in pks:
                    p_sample = end_stream_s - recent_s + p_sub
                    if (p_sub <= recent_s - wait_s) and (p_sample - self.ultimo_pico_mic_sample) >= int(0.85 * self.fs):
                        self.ultimo_pico_mic_sample = p_sample
                        ev = self._extraer_y_decodificar(
                            onset_s=p_sample - int(0.35 * self.fs),
                            center_s=p_sample,
                            close_s=p_sample + int(0.35 * self.fs),
                            end_stream_s=end_stream_s,
                            t_inicio=t_inicio
                        )
                        if ev is not None:
                            eventos.append(ev)

        # MODO 2: Detección Gobernada por Norma Ponderada S(t) + Valles de Inflexión Coarticulatoria (\dot{S}=0)
        if not self.mic_sync_enabled:
            lobe_samples = int(0.50 * self.fs)
            min_center_sep = int(0.78 * self.fs)
            for i in range(n_chunk):
                curr_sample = samples_base + i
                val_s = float(s_nuevo[i])

                if self.estado_gate == 'ESPERANDO_REPOSO':
                    if val_s <= self.u_bajo:
                        self.estado_gate = 'ARMADO'

                elif self.estado_gate == 'ARMADO':
                    if (
                        val_s >= self.u_alto
                        and (curr_sample - self.ultimo_onset_sample) >= int(0.78 * self.fs)
                        and (curr_sample - self.ultimo_center_sample) >= min_center_sep
                    ):
                        buf_idx = self.buffer_len - n_chunk + i
                        lb_st = max(0, buf_idx - int(0.35 * self.fs))
                        sub_s = self.s_emg_buffer[lb_st:buf_idx+1]
                        min_pre = float(np.min(sub_s)) if len(sub_s) > 0 else val_s
                        prominencia = val_s - min_pre

                        # Aceptar disparo si viene de reposo basal O de un valle de coarticulación con prominencia >= 3.0
                        viene_de_reposo = (min_pre <= self.u_bajo * 1.45) and (prominencia >= 1.8)
                        viene_de_valle_coart = (prominencia >= 3.2) and (min_pre <= 0.68 * val_s)
                        if not (viene_de_reposo or viene_de_valle_coart):
                            continue

                        th_low = max(self.u_bajo, min_pre + 0.25 * prominencia) if not viene_de_reposo else min(self.u_bajo, 0.38 * val_s)
                        cruces = np.where(sub_s <= th_low)[0]
                        if len(cruces) > 0:
                            offset_bajo = (buf_idx - (lb_st + int(cruces[-1])))
                        else:
                            offset_bajo = (buf_idx - (lb_st + int(np.argmin(sub_s))))

                        onset_s = curr_sample - offset_bajo
                        self.onset_sample_actual = onset_s
                        self.confirm_sample_actual = curr_sample
                        self.close_sample_actual = -1
                        self._peak_s_actual = val_s
                        self._min_post_peak_s = val_s
                        self._min_post_peak_sample = curr_sample
                        self._thresh_low_actual = th_low
                        self._anchor_computed = False
                        self.fin_sample_actual = onset_s + lobe_samples
                        self.ultimo_onset_sample = onset_s
                        self.estado_gate = 'ESPERANDO_FIN_VENTANA'

                elif self.estado_gate == 'ESPERANDO_FIN_VENTANA':
                    if val_s > self._peak_s_actual:
                        self._peak_s_actual = val_s
                        self._min_post_peak_s = val_s
                        self._min_post_peak_sample = curr_sample
                    elif val_s < self._min_post_peak_s:
                        self._min_post_peak_s = val_s
                        self._min_post_peak_sample = curr_sample

                    if self.close_sample_actual < 0 and curr_sample > (self.confirm_sample_actual + int(0.14 * self.fs)):
                        # Cierre por umbral basal O por valle de inflexión coarticulatoria (\dot{S}=0)
                        cayo_a_reposo = (val_s <= self.u_bajo)
                        valle_inflexion = (
                            curr_sample >= self.onset_sample_actual + int(0.24 * self.fs)
                            and self._min_post_peak_s <= 0.62 * self._peak_s_actual
                            and (val_s - self._min_post_peak_s) >= 2.0
                        )
                        if cayo_a_reposo:
                            self.close_sample_actual = curr_sample
                        elif valle_inflexion:
                            self.close_sample_actual = self._min_post_peak_sample

                    pulso_cerrado = (self.close_sample_actual > 0 and curr_sample >= self.close_sample_actual + int(0.05 * self.fs))
                    pulso_timeout = (curr_sample >= self.onset_sample_actual + int(1.20 * self.fs))

                    if not self._anchor_computed and (pulso_cerrado or pulso_timeout) and curr_sample >= self.fin_sample_actual:
                        dist_on = end_stream_s - self.onset_sample_actual
                        idx_on_buf = max(0, self.buffer_len - dist_on)
                        fin_lobe_s = self.close_sample_actual if self.close_sample_actual > self.onset_sample_actual else curr_sample
                        dur_lobe_samples = max(lobe_samples, min(int(1.20 * self.fs), fin_lobe_s - self.onset_sample_actual + int(0.05 * self.fs)))
                        idx_end_lobe = min(self.buffer_len, idx_on_buf + dur_lobe_samples)
                        lobe_s = self.s_emg_buffer[idx_on_buf:idx_end_lobe]

                        puntos_activos = np.where(lobe_s >= self.u_bajo)[0]
                        puntos_altos = np.where(lobe_s >= self.u_alto)[0]
                        if len(puntos_activos) < 7 or len(puntos_altos) < 3:
                            self.estado_gate = 'ARMADO'
                            continue

                        if len(lobe_s) > 3 and self.trigger_mode in ('GATE_HYBRID', 'GATE_CENTROID', 'GATE_CFD55'):
                            i_pk_rel = int(np.argmax(lobe_s))
                            s_min = float(np.min(lobe_s[:i_pk_rel+1]))
                            if self.trigger_mode == 'GATE_CFD55':
                                th_cfd = s_min + 0.55 * (float(lobe_s[i_pk_rel]) - s_min)
                                cfd_pts = np.where(lobe_s[:i_pk_rel+1] <= th_cfd)[0]
                                i_anchor = int(cfd_pts[-1]) if len(cfd_pts) > 0 else i_pk_rel
                            else:
                                high_pts = np.where(lobe_s >= self.u_alto)[0]
                                ic_rel = int(high_pts[0]) if len(high_pts) > 0 else i_pk_rel
                                th_base = min(self.u_bajo, 0.38 * float(lobe_s[ic_rel]))
                                w_s = np.maximum(0.0, lobe_s - th_base) ** 2
                                i_cen = float(np.sum(np.arange(len(lobe_s)) * w_s) / (np.sum(w_s) + 1e-8))
                                if self.trigger_mode == 'GATE_CENTROID':
                                    i_anchor = int(round(i_cen))
                                else:
                                    i_anchor = int(round(0.50 * i_cen + 0.50 * i_pk_rel))
                            anchor_sample = self.onset_sample_actual + i_anchor
                        else:
                            anchor_sample = self.onset_sample_actual

                        if self.close_sample_actual < 0 and len(lobe_s) > 3:
                            i_pk_l = int(np.argmax(lobe_s))
                            post_low = np.where(lobe_s[i_pk_l:] <= self.u_bajo)[0]
                            if len(post_low) > 0:
                                self.close_sample_actual = self.onset_sample_actual + i_pk_l + int(post_low[0])
                            else:
                                self.close_sample_actual = self.onset_sample_actual + len(lobe_s)

                        self.center_sample_actual = anchor_sample + self.emd_samples
                        self.ultimo_center_sample = self.center_sample_actual
                        self._anchor_computed = True
                        self.fin_sample_actual = max(curr_sample, self.center_sample_actual + self.post_samples)

                    if self._anchor_computed and end_stream_s >= (self.center_sample_actual + self.post_samples):
                        ev = self._extraer_y_decodificar(
                            onset_s=self.onset_sample_actual,
                            center_s=self.center_sample_actual,
                            close_s=self.close_sample_actual if self.close_sample_actual > self.onset_sample_actual else (self.center_sample_actual + int(0.35 * self.fs)),
                            end_stream_s=end_stream_s,
                            t_inicio=t_inicio
                        )
                        if ev is not None:
                            eventos.append(ev)
                        self.estado_gate = 'REFRACTARIO'
                        self._min_post_peak_s = val_s

                elif self.estado_gate == 'REFRACTARIO':
                    if val_s < getattr(self, '_min_post_peak_s', val_s):
                        self._min_post_peak_s = val_s
                    # Re-armar por caída bajo U_bajo O por valle de coarticulación (<= 0.62 del pico anterior)
                    if val_s <= self.u_bajo or val_s <= 0.62 * max(self.u_alto, getattr(self, '_peak_s_actual', self.u_alto)):
                        self.estado_gate = 'ARMADO'

        self.total_samples_received += n_chunk
        return eventos, s_nuevo, self.get_telemetry()

    def _extraer_y_decodificar(self, onset_s, center_s, close_s, end_stream_s, t_inicio):
        dist_al_fin = end_stream_s - center_s
        idx_centro = self.buffer_len - dist_al_fin
        idx_ini = idx_centro - self.pre_samples
        idx_fin = idx_centro + self.post_samples

        if idx_ini < 0 or idx_fin > self.buffer_len:
            return None

        seg = self.env_buffer[:, idx_ini:idx_fin]
        if seg.shape[1] < 3:
            return None

        s_seg = self.s_emg_buffer[idx_ini:idx_fin]
        active_idxs = np.where(s_seg >= (self.u_bajo * 0.85))[0]
        if len(active_idxs) >= 4:
            rel_ini_s = (int(active_idxs[0]) - self.pre_samples) / self.fs
            rel_fin_s = (int(active_idxs[-1]) - self.pre_samples) / self.fs
            rel_ini_s = min(-0.18, max(-0.50, rel_ini_s - 0.04))
            rel_fin_s = max(0.22, min(0.50, rel_fin_s + 0.05))
        else:
            rel_ini_s = -0.25
            rel_fin_s = 0.35

        win_start_sample = int(round(center_s + rel_ini_s * self.fs))
        win_end_sample = int(round(center_s + rel_fin_s * self.fs))
        duracion_ventana_ms = float((win_end_sample - win_start_sample) / self.fs * 1000.0)

        t_orig = np.linspace(-0.5, 0.5, seg.shape[1])
        t_grid = np.linspace(-0.5, 0.5, self.target_len)
        vecs = [np.interp(t_grid, t_orig, seg[c]) for c in range(self.n_canales)]
        x_150 = np.concatenate(vecs)

        t_inf0 = time.perf_counter()
        preds, probs, Z = self.encode_and_predict_window(x_150, return_rest_prob=True)
        t_inf1 = time.perf_counter()

        v_idx = int(preds[0])
        if v_idx < 0:
            return None

        vocal_ganadora = VOCALES[v_idx]
        probs_v = probs[0]
        probs_dict = {VOCALES[i]: float(probs_v[i] * 100.0) for i in range(5)}
        confianza_pct = probs_dict[vocal_ganadora]
        z_np = Z[0]

        ev_dict = {
            'vocal': vocal_ganadora,
            'color': VOCAL_A_COLOR_HEX[vocal_ganadora],
            'confianza': confianza_pct,
            'z1': float(z_np[0]),
            'z2': float(z_np[1]) if len(z_np) > 1 else 0.0,
            'z3': float(z_np[2]) if len(z_np) > 2 else 0.0,
            'z4': float(z_np[3]) if len(z_np) > 3 else 0.0,
            'probs': probs_dict,
            'onset_sample': int(onset_s),
            'center_sample': int(center_s),
            'close_sample': int(close_s),
            'win_start_sample': int(win_start_sample),
            'win_end_sample': int(win_end_sample),
            'win_rel_ini_s': float(rel_ini_s),
            'win_rel_fin_s': float(rel_fin_s),
            'duracion_ventana_ms': float(duracion_ventana_ms),
            'x_150': x_150.copy(),
            'raw_window_6ch': seg.copy(),
            's_window': s_seg.copy(),
            'trigger_mode': self.trigger_mode,
            'corrimiento_ms': self.emd_delay_ms,
            'supremo_uv': float(np.max(x_150)),
            'latencia_total_ms': (time.perf_counter() - t_inicio) * 1000.0,
            'latencia_infer_ms': (t_inf1 - t_inf0) * 1000.0,
            'timestamp': time.time()
        }
        self.last_window_info = ev_dict
        return ev_dict

    def flush_pending_events(self):
        if self.estado_gate == 'ESPERANDO_FIN_VENTANA' and self.onset_sample_actual >= 0:
            t_inicio = time.perf_counter()
            end_stream_s = self.total_samples_received
            center_s = self.center_sample_actual if self._anchor_computed else (self.onset_sample_actual + self.emd_samples)
            center_s = min(center_s, end_stream_s - self.post_samples)
            ev = self._extraer_y_decodificar(
                onset_s=self.onset_sample_actual,
                center_s=center_s,
                close_s=self.close_sample_actual if self.close_sample_actual > 0 else (center_s + int(0.30 * self.fs)),
                end_stream_s=end_stream_s,
                t_inicio=t_inicio
            )
            self.estado_gate = 'ARMADO'
            if ev is not None:
                return [ev]
        return []

    def get_telemetry(self):
        return {
            'estado': self.estado_gate,
            'trigger_mode': self.trigger_mode,
            'u_bajo': self.u_bajo,
            'u_alto': self.u_alto,
            'med_base': self.med_base,
            'calibrado': self.calibrado,
            'emd_delay_ms': self.emd_delay_ms,
            'ema_alpha': self.ema_alpha,
            'total_muestras': self.total_samples_received,
            'last_window': self.last_window_info
        }


RealtimeFacialDecoderEngine = RealtimeDecoderEngine
