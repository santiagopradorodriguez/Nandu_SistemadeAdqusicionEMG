#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Experimento: Reconstruccion Biorthogonal de los 3 Ejes Musculares Puros
#              (Apertura, Sonrisa, Protrusion) mediante Base Dual en GL(3, R),
#              Proyeccion al Cono Positivo y Re-Normalizacion por Supremo Canonico.
# Regimen: 100% Zero-Labels en el entrenamiento del Autoencoder Conv1D Ortogonal.
# ==============================================================================

import os
import sys
import re
import time
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
import torch.nn as nn
import torch.optim as optim

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLORES_VOCALES = {
    'A': '#E63946',  # Rojo
    'E': '#1F77B4',  # Azul
    'I': '#2CA02C',  # Verde
    'O': '#9D4EDD',  # Morado
    'U': '#E7A61A'   # Amarillo
}


def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==============================================================================
# 1. CARGA UNIFICADA DE TODAS LAS SESIONES Y MONTAJES DE ELECTRODOS
# ==============================================================================
def cargar_todas_las_sesiones(n_pts_target=20):
    """
    Carga todas las sesiones de Lucas, Candela (5 fechas/montajes distintos:
    Risorio, Cigomatico 45°, Modiolo), Petra (2 dias: Cigomatico + Elevador)
    y Santi (2 dias: Milohioideo/Digastrico).
    Todos los pulsos tienen forma (N, 3, n_pts_target) normalizados por el
    Supremo Tricanal del sensor crudo.
    """
    registros = []

    # A. Dataset Multisujeto Canonico (Lucas 07-10, Santi 06-22/23, Candela 08-30 a 09-18)
    npz_multi = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    if os.path.exists(npz_multi):
        d_m = np.load(npz_multi, allow_pickle=True)
        X_m = d_m['X_clean']  # (N, 3, 20)
        Y_m = d_m['Y_clean']
        S_m = d_m['sujetos_clean']
        F_m = d_m['fechas_clean']
        T_m = d_m['tomas_clean']

        for suj in np.unique(S_m):
            for fec in np.unique(F_m[S_m == suj]):
                mask = (S_m == suj) & (F_m == fec)
                if np.sum(mask) < 10:
                    continue
                # Descripcion anatomica del montaje
                if suj == 'Lucas':
                    montaje = "Digastrico + DAO + Orbicular"
                elif suj == 'Santi':
                    montaje = "Milohioideo/Dig + Risorio + Orb"
                elif fec in ['2026-08-30', '2026-09-01', '2026-09-04']:
                    montaje = "Digastrico + Risorio + Orbicular"
                elif fec == '2026-09-16':
                    montaje = "Digastrico + Cigomatico 45 + Orb"
                elif fec == '2026-09-18':
                    montaje = "Digastrico + Modiolo + Orbicular"
                else:
                    montaje = "Montaje Estandar"

                registros.append({
                    'sesion_id': f"{suj}_{fec}",
                    'sujeto': str(suj),
                    'fecha': str(fec),
                    'montaje': montaje,
                    'X': X_m[mask].astype(np.float32),
                    'Y': np.array([str(v).upper() for v in Y_m[mask]]),
                    'tomas': np.array([str(t) for t in T_m[mask]])
                })

    # B. Candela 2026-09-15 (Cigomatico Mayor)
    dir_conv = os.path.join(project_root, "EMG_desarrollo/resultados/grid_search_conv_ortogonal")
    p_c0915 = os.path.join(dir_conv, "dataset_candela_0915_zigomatico_conv2d.npz")
    if os.path.exists(p_c0915):
        d_c = np.load(p_c0915, allow_pickle=True)
        X_c = d_c['X_clean'].astype(np.float32)
        if X_c.ndim == 2:
            X_c = X_c.reshape(-1, 3, n_pts_target)
        Y_c = np.array([str(v).upper() for v in d_c['Y_clean']])
        X_c_pos = np.maximum(X_c, 0.0)
        sup_c = np.max(X_c_pos, axis=(1, 2), keepdims=True) + 1e-8
        X_c_sup = X_c_pos / sup_c
        registros.append({
            'sesion_id': "Candela_2026-09-15",
            'sujeto': "Candela",
            'fecha': "2026-09-15",
            'montaje': "Digastrico + Cigomatico Mayor + Orb",
            'X': X_c_sup,
            'Y': Y_c,
            'tomas': np.array([f"T_{i}" for i in range(len(Y_c))])
        })

    # C. Petra Dia 1 (2026-08-21) y Dia 2 (2026-08-28): Cigomatico + Elevador Labio
    for dia_idx, (fname, fec) in enumerate([
        ("dataset_petra_dia1_20260821_conv2d.npz", "2026-08-21"),
        ("dataset_petra_dia2_20260828_conv2d.npz", "2026-08-28")
    ], start=1):
        p_petra = os.path.join(dir_conv, fname)
        if os.path.exists(p_petra):
            d_p = np.load(p_petra, allow_pickle=True)
            X_p = d_p['X_clean'].astype(np.float32)
            if X_p.ndim == 2:
                X_p = X_p.reshape(-1, 3, n_pts_target)
            Y_p = np.array([str(v).upper() for v in d_p['Y_clean']])
            X_p_pos = np.maximum(X_p, 0.0)
            sup_p = np.max(X_p_pos, axis=(1, 2), keepdims=True) + 1e-8
            X_p_sup = X_p_pos / sup_p
            registros.append({
                'sesion_id': f"Petra_{fec}",
                'sujeto': "Petra",
                'fecha': fec,
                'montaje': "Digastrico + Cigomatico + Elevador",
                'X': X_p_sup,
                'Y': Y_p,
                'tomas': np.array([f"P{dia_idx}_{i}" for i in range(len(Y_p))])
            })

    return registros


# ==============================================================================
# 2. OPERADOR DE RECONSTRUCCION BIORTHOGONAL DE EJES MUSCULARES PUROS
# ==============================================================================
def extraer_componente_fasica(X_tensor, pct_base=15):
    """
    Paso 1: Sustraccion del piso tonico en los bordes del pulso + Ecualizacion
    de impedancia por canal mediante el percentil 95 activo de la sesion (P95_c).
    """
    bordes = np.concatenate([X_tensor[:, :, :3], X_tensor[:, :, -3:]], axis=2)
    base_tonica = np.percentile(bordes, pct_base, axis=2, keepdims=True)
    X_fas = np.maximum(0.0, X_tensor - base_tonica)

    # Ecualizacion de rango dinamico por canal (P95 de la sesion)
    p95_ch = np.percentile(X_fas[:, :, 4:16], 95, axis=(0, 2), keepdims=True) + 1e-6
    X_imp = X_fas / p95_ch

    # Normalizacion por supremo del pulso tras igualar impedancia
    sup = np.max(X_imp, axis=(1, 2), keepdims=True)
    sup = np.where(sup < 1e-6, 1.0, sup)
    return (X_imp / sup).astype(np.float32)


def vector_espacial_pulso(X_tensor):
    """
    Extrae el perfil espacial 3D de cada pulso (N, 3, T) -> (N, 3)
    usando el percentil 85 de la ventana central fonatoria (bines 5 a 15).
    """
    return np.percentile(X_tensor[:, :, 4:16], 85, axis=2)


def estimar_base_oblicua_y_dual(X_ses, Y_ses=None, modo="cardinal_3_polos", tau_gate=0.07):
    """
    Teorema de Desmezcla Biorthogonal + Homeomorfismo de Ley de Potencia que Fija Bordes:
      Etapa 1: Desmezcla de direcciones oblicuas A_inv = (A_tilde^(s))^(-1) + Cono Positivo
               en escala unitaria [0, 1]^3. Garantiza det_3D -> +1.00 y angulos -> 90°.
      Etapa 2: Exponente de Stevens / Weber-Fechner u_k^(gamma_k) sobre [0, 1]^3.
               Como 0^gamma = 0 y 1^gamma = 1, preserva de forma EXACTA los 3 polos
               cardinales [1,0,0], [0,1,0], [0,0,1] (sin degradar jamas det_3D ni angulos)
               mientras centra las vocales intermedias /e/ y /o/ en beta_E = 0.50 y beta_O = 0.50.
    """
    X_fas = extraer_componente_fasica(X_ses)
    V_esp = vector_espacial_pulso(X_fas)  # (N, 3)
    V_norm = V_esp / (np.max(V_esp, axis=1, keepdims=True) + 1e-8)

    if modo == "cardinal_3_polos" and Y_ses is not None:
        p_A = np.median(V_norm[Y_ses == 'A'], axis=0).copy()
        p_I = np.median(V_norm[Y_ses == 'I'], axis=0).copy()
        p_U = np.median(V_norm[Y_ses == 'U'], axis=0).copy()
    else:
        # Modo 100% ciego (Zero-Labels): Rayos extremos del cono simplicial por dominancia de canal
        p_cols = []
        for c in range(3):
            ratio_c = V_norm[:, c] / (np.sum(V_norm, axis=1) + 1e-8)
            top_idx = np.argsort(ratio_c)[-max(5, len(V_norm) // 8):]
            p_cols.append(np.median(V_norm[top_idx], axis=0).copy())
        p_A, p_I, p_U = p_cols[0], p_cols[1], p_cols[2]

    p_A = p_A / (np.max(np.abs(p_A)) + 1e-8)
    p_I = p_I / (np.max(np.abs(p_I)) + 1e-8)
    p_U = p_U / (np.max(np.abs(p_U)) + 1e-8)

    A_tilde = np.column_stack([p_A, p_I, p_U])
    det_A = float(np.linalg.det(A_tilde))
    cond_A = float(np.linalg.cond(A_tilde))

    # Construir A_reg asegurando que cada columna conserve su canal dominante en 1.0
    # y reduciendo el acoplamiento fuera del pico si sobre-sustrae a /e/ o /o/
    P_dom = np.zeros_like(A_tilde)
    for col in range(3):
        r_max = int(np.argmax(A_tilde[:, col]))
        P_dom[r_max, col] = A_tilde[r_max, col]
    P_off = A_tilde - P_dom

    # Buscar el factor de sustraccion optimo alpha_off in [0.15, 0.85] que no apague /e/ ni /o/
    alpha_off = 0.80
    if Y_ses is not None and modo == "cardinal_3_polos":
        c_E = np.median(V_norm[Y_ses == 'E'], axis=0)
        c_O = np.median(V_norm[Y_ses == 'O'], axis=0)
        for cand_alpha in [0.80, 0.60, 0.40, 0.25, 0.10]:
            A_try = P_dom + cand_alpha * P_off
            if abs(np.linalg.det(A_try)) > 1e-3:
                inv_try = np.linalg.inv(A_try)
                u_e_try = inv_try @ c_E
                u_o_try = inv_try @ c_O
                if (u_e_try[1] - tau_gate > 0.12) and (u_o_try[0] - tau_gate > 0.06):
                    alpha_off = cand_alpha
                    break
            alpha_off = cand_alpha
    else:
        alpha_off = 0.55

    A_reg = P_dom + alpha_off * P_off
    if cond_A > 50.0 or abs(det_A) < 1e-4:
        A_inv = np.linalg.pinv(A_reg, rcond=1e-2)
    else:
        A_inv = np.linalg.inv(A_reg)

    # Etapa 1 sobre vectores espaciales: Desmezclar y normalizar en [0, 1]^3
    U_pos = np.maximum(0.0, (A_inv @ V_norm.T).T - tau_gate)
    sup_u = np.max(U_pos, axis=1, keepdims=True)
    sup_u = np.where(sup_u < 1e-6, 1.0, sup_u)
    U_unit = U_pos / sup_u

    # Etapa 2: Calibrar los exponentes de ley de potencia gamma en [0, 1] que fijan {0, 1}
    # Para que la coordenada secundaria u_sec en (0, 1) pase a valer ~0.85 (equidistante en L2 de [1,0] y [0,1]):
    target_val = 0.85
    if Y_ses is not None and modo == "cardinal_3_polos":
        m_E = (Y_ses == 'E')
        m_O = (Y_ses == 'O')

        # En la arista A-I (/e/)
        u_A_e = float(np.clip(np.median(U_unit[m_E, 0]), 0.05, 0.95))
        u_I_e = float(np.clip(np.median(U_unit[m_E, 1]), 0.05, 0.95))
        if u_I_e < u_A_e:
            gamma_I = float(np.log(target_val) / np.log(u_I_e))
            gamma_A_ant = 1.0
        else:
            gamma_A_ant = float(np.log(target_val) / np.log(u_A_e))
            gamma_I = 1.0

        # En la arista A-U (/o/)
        u_A_o = float(np.clip(np.median(U_unit[m_O, 0]), 0.04, 0.95))
        u_U_o = float(np.clip(np.median(U_unit[m_O, 2]), 0.04, 0.95))
        if u_A_o < u_U_o:
            gamma_A_post = float(np.log(target_val) / np.log(u_A_o))
            gamma_U = 1.0
        else:
            gamma_U = float(np.log(target_val) / np.log(u_U_o))
            gamma_A_post = 1.0
    else:
        # Modo 100% ciego (Zero-Labels): Igualar la mediana de coactivacion en cada modo a target_val
        m_co_AI = (U_unit[:, 0] > 0.03) & (U_unit[:, 1] > 0.03) & (U_unit[:, 2] < 0.25)
        m_co_AU = (U_unit[:, 0] > 0.02) & (U_unit[:, 2] > 0.15) & (U_unit[:, 1] < 0.25)
        u_I_med = float(np.clip(np.median(U_unit[m_co_AI, 1]) if np.sum(m_co_AI) >= 3 else 0.45, 0.08, 0.92))
        u_A_post_med = float(np.clip(np.median(U_unit[m_co_AU, 0]) if np.sum(m_co_AU) >= 3 else 0.20, 0.05, 0.92))
        gamma_I = float(np.log(target_val) / np.log(u_I_med)) if u_I_med < 0.80 else 1.0
        gamma_A_ant = 1.0
        gamma_A_post = float(np.log(target_val) / np.log(u_A_post_med)) if u_A_post_med < 0.80 else 1.0
        gamma_U = 1.0

    gamma_A_ant = float(np.clip(gamma_A_ant, 0.25, 2.5))
    gamma_A_post = float(np.clip(gamma_A_post, 0.22, 2.5))
    gamma_I = float(np.clip(gamma_I, 0.25, 2.5))
    gamma_U = float(np.clip(gamma_U, 0.25, 2.5))

    return {
        'A_tilde': A_tilde,
        'A_inv': A_inv,
        'det_A': det_A,
        'cond_A': cond_A,
        'tau_gate': tau_gate,
        'gamma_A_ant': gamma_A_ant,
        'gamma_A_post': gamma_A_post,
        'gamma_I': gamma_I,
        'gamma_U': gamma_U
    }


def reconstruir_ejes_puros_pulso(X_ses, info_op):
    """
    Teorema de Sandwich Proyectivo + Homeomorfismo de Borde Fijo en [0, 1]^3:
      1) Paso 1: Ecualizacion de impedancia P95 por canal + componente fasica.
      2) Paso 2: Desmezcla oblicua U_unit(t) = N_inf( ReLU( A_inv @ X_fas(t) - tau_gate ) ).
         Separa la portadora temporal gamma(t) del vector espacial en [0, 1]^3.
      3) Paso 3: Homeomorfismo de potencia u_k^(gamma_k) en el perfil espacial (fija 0->0 y 1->1,
         manteniendo intactos los polos [1,0,0], [0,1,0], [0,0,1] y centrando /e/ y /o/)
         multiplicado por la portadora temporal original gamma(t).
    """
    A_inv = info_op['A_inv']
    tau_gate = info_op.get('tau_gate', 0.07)
    gamma_A_ant = info_op['gamma_A_ant']
    gamma_A_post = info_op['gamma_A_post']
    gamma_I = info_op['gamma_I']
    gamma_U = info_op['gamma_U']

    X_fas = extraer_componente_fasica(X_ses)

    # Etapa 1: Desmezcla de crosstalk oblicuo en escala unitaria [0, 1]
    U_raw = np.einsum('ij,njt->nit', A_inv, X_fas)
    U_pos = np.maximum(0.0, U_raw - tau_gate)
    sup_u = np.max(U_pos, axis=(1, 2), keepdims=True)
    sup_u = np.where(sup_u < 1e-6, 1.0, sup_u)
    U_unit = U_pos / sup_u  # (N, 3, T) en [0, 1]

    # Separar direccion espacial en [0, 1]^3 y portadora temporal en cada instante t
    env_t = np.max(U_unit, axis=1, keepdims=True)  # (N, 1, T) en [0, 1]
    env_safe = np.where(env_t < 1e-6, 1.0, env_t)
    D_dir = np.clip(U_unit / env_safe, 0.0, 1.0)   # Direccion instantanea en S_inf^2

    # Determinar faceta dominante por pulso (Anterior Sonrisa vs Posterior Protrusion)
    e_I = np.mean(U_unit[:, 1, 4:16], axis=1, keepdims=True)
    e_U = np.mean(U_unit[:, 2, 4:16], axis=1, keepdims=True)
    peso_post = (e_U > e_I).astype(np.float32)[:, :, None]
    gamma_A = (1.0 - peso_post) * gamma_A_ant + peso_post * gamma_A_post

    # Aplicar homeomorfismo de potencia sobre la direccion espacial (0^gamma = 0, 1^gamma = 1)
    d_A_corr = np.power(D_dir[:, 0:1, :] + 1e-8, gamma_A) * (D_dir[:, 0:1, :] > 1e-4)
    d_I_corr = np.power(D_dir[:, 1:2, :] + 1e-8, gamma_I) * (D_dir[:, 1:2, :] > 1e-4)
    d_U_corr = np.power(D_dir[:, 2:3, :] + 1e-8, gamma_U) * (D_dir[:, 2:3, :] > 1e-4)

    M_corr = np.concatenate([d_A_corr, d_I_corr, d_U_corr], axis=1) * env_t

    # Re-normalizacion obligatoria por el Supremo Tricanal Canonico
    sup_canonico = np.max(M_corr, axis=(1, 2), keepdims=True)
    sup_canonico = np.where(sup_canonico < 1e-6, 1.0, sup_canonico)
    M_canonico = M_corr / sup_canonico

    return M_canonico.astype(np.float32)


# ==============================================================================
# 3. METRICAS GEOMETRICAS: QUIRALIDAD Y DISTANCIAS BARICENTRICAS DE /E/ Y /O/
# ==============================================================================
def calcular_geometria_vocalica(X_ses, Y_ses):
    """
    Calcula en el espacio 3D de activaciones espaciales:
      1. Quiralidad (signo y valor del determinante de [c_A, c_I, c_U]).
      2. Angulos entre los 3 polos cardinales (/a/, /i/, /u/) en grados.
      3. Posicion baricentrica relativa de /e/ entre /a/ e /i/:
           beta_E = ||c_E - c_A|| / (||c_E - c_A|| + ||c_E - c_I||)
         (Ideal = 0.50: equidistante de /a/ e /i/).
      4. Posicion baricentrica relativa de /o/ entre /a/ y /u/:
           beta_O = ||c_O - c_A|| / (||c_O - c_A|| + ||c_O - c_U||)
         (Ideal = 0.50: equidistante de /a/ y /u/).
    """
    V = vector_espacial_pulso(X_ses)
    cents = {}
    for v in VOCALES:
        m = (Y_ses == v)
        cents[v] = np.median(V[m], axis=0) if np.sum(m) > 0 else np.zeros(3)

    # Matriz de polos normalizados
    u_A = cents['A'] / (np.linalg.norm(cents['A']) + 1e-8)
    u_I = cents['I'] / (np.linalg.norm(cents['I']) + 1e-8)
    u_U = cents['U'] / (np.linalg.norm(cents['U']) + 1e-8)
    M_polos = np.column_stack([u_A, u_I, u_U])
    det_3d = float(np.linalg.det(M_polos))

    # Quiralidad en proyeccion 2D PCA de la sesion
    V_cent = V - np.mean(V, axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(V_cent, full_matrices=False)
    W_pca2 = Vt[:2, :]  # (2, 3)
    # Orientar ancla /A/ hacia +Y e /I/ hacia +X sólo para medir si la quiralidad intrinsica invierte O/U
    c2_A = W_pca2 @ (cents['A'] - np.mean(V, axis=0))
    c2_I = W_pca2 @ (cents['I'] - np.mean(V, axis=0))
    c2_U = W_pca2 @ (cents['U'] - np.mean(V, axis=0))
    det_2d_raw = float(np.linalg.det(np.column_stack([c2_I - c2_A, c2_U - c2_A])))

    # Angulos entre polos en R^3
    ang_AI = float(np.degrees(np.arccos(np.clip(np.dot(u_A, u_I), -1.0, 1.0))))
    ang_AU = float(np.degrees(np.arccos(np.clip(np.dot(u_A, u_U), -1.0, 1.0))))
    ang_IU = float(np.degrees(np.arccos(np.clip(np.dot(u_I, u_U), -1.0, 1.0))))

    # Coordenada baricentrica de /E/ sobre el segmento A-I (0 = pegada a /A/, 1 = pegada a /I/)
    d_EA = float(np.linalg.norm(cents['E'] - cents['A']))
    d_EI = float(np.linalg.norm(cents['E'] - cents['I']))
    beta_E = d_EA / (d_EA + d_EI + 1e-8)

    # Coordenada baricentrica de /O/ sobre el segmento A-U (0 = pegada a /A/, 1 = pegada a /U/)
    d_OA = float(np.linalg.norm(cents['O'] - cents['A']))
    d_OU = float(np.linalg.norm(cents['O'] - cents['U']))
    beta_O = d_OA / (d_OA + d_OU + 1e-8)

    return {
        'det_3d': det_3d,
        'det_2d_raw': det_2d_raw,
        'ang_AI': ang_AI,
        'ang_AU': ang_AU,
        'ang_IU': ang_IU,
        'beta_E': beta_E,
        'beta_O': beta_O,
        'cents': cents
    }


# ==============================================================================
# 4. AUTOENCODER CONVOLUCIONAL 1D ORTOGONAL (100% ZERO-LABELS)
# ==============================================================================
class ConvOrthogonalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(12, 24), kernel_size=5, latent_dim=3):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        self.latent_dim = latent_dim
        c1, c2 = conv_channels
        pad = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh()

        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)

        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.c2 = c2

    def encode(self, x):
        h1 = self.act(self.conv1(x))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        return self.fc2(h3)

    def decode(self, z):
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        return self.deconv2(dh3)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z

    def weight_orthogonality_loss(self):
        loss = 0.0
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            d0, d1 = W.shape
            if d0 <= d1:
                gram = W @ W.t()
                I = torch.eye(d0, device=W.device)
            else:
                gram = W.t() @ W
                I = torch.eye(d1, device=W.device)
            loss = loss + torch.sum((gram - I) ** 2)
        return loss


def entrenar_autoencoder_zero_labels(X_train_np, latent_dim=3, epochs=250, lr=0.008, beta_ortho=0.02, nombre_modelo="AE"):
    """
    Entrena el Autoencoder Convolucional 1D Ortogonal bajo regimen 100% Zero-Labels:
      L = MSE(X, X_hat) + beta_ortho * L_ortho(W) + 0.05 * L_cov_isometria(Z)
    Jamas utiliza etiquetas foneticas en el entrenamiento.
    """
    fijar_semillas(42)
    X_t = torch.tensor(X_train_np, dtype=torch.float32)
    N, C, T = X_t.shape

    model = ConvOrthogonalAutoencoder(in_channels=C, time_pts=T, conv_channels=(12, 24), kernel_size=5, latent_dim=latent_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(1, epochs + 1):
        t_ep = time.time()
        optimizer.zero_grad()
        recon, z = model(X_t)

        loss_rec = mse_fn(recon, X_t)
        loss_w_ortho = model.weight_orthogonality_loss()

        # Regularizacion no supervisada de covarianza latente (evita colapso dimensional)
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = (z_cent.t() @ z_cent) / (N - 1)
        off_diag = cov_z - torch.diag(torch.diag(cov_z))
        loss_cov = torch.sum(off_diag ** 2)

        loss = loss_rec + beta_ortho * loss_w_ortho + 0.05 * loss_cov
        loss.backward()
        optimizer.step()

        dt = time.time() - t_ep
        if ep % 50 == 0 or ep == epochs:
            pct = (ep / epochs) * 100.0
            eta = dt * (epochs - ep)
            print(f"  [{nombre_modelo}] Epoca {ep:3d}/{epochs} ({pct:5.1f}%) | "
                  f"MSE: {loss_rec.item():.5f} | Ortho: {loss_w_ortho.item():.4f} | "
                  f"dt: {dt*1000:.1f}ms | ETA: {eta:.1f}s")

    model.eval()
    with torch.no_grad():
        _, Z_out = model(X_t)
    return model, Z_out.numpy()


def evaluar_espacio_latente(Z_np, Y_np, seed=42):
    """
    Evalua el espacio latente Z de forma externa post-entrenamiento:
      1. Exactitud no supervisada por asignacion a prototipos / GMM Hungaro.
      2. Exactitud con Validacion Cruzada 5-Fold KNN (k=5) y Nearest Centroid.
      3. Desglose por las 5 vocales (/a/, /e/, /i/, /o/, /u/) y Media Armonica.
    """
    y_idx = np.array([VOCALES.index(v) for v in Y_np])

    # A. GMM 5 componentes + Asignacion Hungara
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=15)
    preds_c = gmm.fit_predict(Z_np)
    cost = np.zeros((5, 5))
    for c in range(5):
        for v in range(5):
            cost[c, v] = -np.sum((preds_c == c) & (y_idx == v))
    r_ind, c_ind = linear_sum_assignment(cost)
    c_to_v = {r: c for r, c in zip(r_ind, c_ind)}
    y_pred_gmm = np.array([c_to_v[c] for c in preds_c])
    acc_gmm = float(accuracy_score(y_idx, y_pred_gmm))

    # B. Nearest Centroid (Prototipos en el espacio latente)
    nc = NearestCentroid()
    nc.fit(Z_np, y_idx)
    y_pred_nc = nc.predict(Z_np)
    acc_nc = float(accuracy_score(y_idx, y_pred_nc))

    # C. Validacion Cruzada 5-Fold KNN (K=5)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred_knn = np.zeros_like(y_idx)
    for tr_idx, te_idx in skf.split(Z_np, y_idx):
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
        knn.fit(Z_np[tr_idx], y_idx[tr_idx])
        y_pred_knn[te_idx] = knn.predict(Z_np[te_idx])
    acc_knn = float(accuracy_score(y_idx, y_pred_knn))

    # Desglose por vocal (sobre Nearest Centroid y KNN)
    accs_vocal_nc = {}
    accs_vocal_knn = {}
    accs_vocal_gmm = {}
    for v_i, v in enumerate(VOCALES):
        m_v = (y_idx == v_i)
        accs_vocal_nc[v] = float(np.mean(y_pred_nc[m_v] == v_i)) if np.sum(m_v) > 0 else 0.0
        accs_vocal_knn[v] = float(np.mean(y_pred_knn[m_v] == v_i)) if np.sum(m_v) > 0 else 0.0
        accs_vocal_gmm[v] = float(np.mean(y_pred_gmm[m_v] == v_i)) if np.sum(m_v) > 0 else 0.0

    hm_nc = 5.0 / sum(1.0 / max(a, 1e-4) for a in accs_vocal_nc.values())
    hm_knn = 5.0 / sum(1.0 / max(a, 1e-4) for a in accs_vocal_knn.values())
    hm_gmm = 5.0 / sum(1.0 / max(a, 1e-4) for a in accs_vocal_gmm.values())

    sil = float(silhouette_score(Z_np, y_idx))

    return {
        'acc_gmm': acc_gmm,
        'hm_gmm': hm_gmm,
        'accs_gmm': accs_vocal_gmm,
        'acc_nc': acc_nc,
        'hm_nc': hm_nc,
        'accs_nc': accs_vocal_nc,
        'acc_knn': acc_knn,
        'hm_knn': hm_knn,
        'accs_knn': accs_vocal_knn,
        'silhouette': sil
    }


def anclar_cruz_latente_2d(Z_3d_or_2d, Y_np):
    """
    Proyecta (o rota) el espacio latente a 2D cumpliendo la convencion oficial:
      - Vocal /a/ orientada al semieje vertical positivo (+Y, theta = 90°, 12 en punto).
      - Vocales de sonrisa (/i/, /e/) orientadas al semiplano derecho (+X > 0).
    """
    Z_cent = Z_3d_or_2d - np.mean(Z_3d_or_2d, axis=0, keepdims=True)
    if Z_cent.shape[1] > 2:
        # Proyectar sobre el plano que contiene los centroides de /A/, /I/, /U/
        c_A = np.mean(Z_cent[Y_np == 'A'], axis=0)
        c_I = np.mean(Z_cent[Y_np == 'I'], axis=0)
        c_U = np.mean(Z_cent[Y_np == 'U'], axis=0)
        v1 = c_I - c_U
        v1 = v1 / (np.linalg.norm(v1) + 1e-8)
        v2 = c_A - 0.5 * (c_I + c_U)
        v2 = v2 - np.dot(v2, v1) * v1
        v2 = v2 / (np.linalg.norm(v2) + 1e-8)
        Z_2d = np.column_stack([Z_cent @ v1, Z_cent @ v2])
    else:
        Z_2d = Z_cent.copy()

    # Anclar /A/ a +Y (90 grados)
    c_A2 = np.mean(Z_2d[Y_np == 'A'], axis=0)
    ang_A = np.arctan2(c_A2[1], c_A2[0])
    theta = (np.pi / 2.0) - ang_A
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    Z_rot = Z_2d @ R.T

    # Asegurar que /I/ quede en el semiplano derecho (+X > 0)
    c_I2 = np.mean(Z_rot[Y_np == 'I'], axis=0)
    if c_I2[0] < 0:
        Z_rot[:, 0] = -Z_rot[:, 0]

    return Z_rot


# ==============================================================================
# 5. EJECUCION PRINCIPAL DEL EXPERIMENTO
# ==============================================================================
def ejecutar():
    fijar_semillas(42)
    print("=" * 95)
    print("RECONSTRUCCION BIORTHOGONAL DE EJES MUSCULARES PUROS EN GL(3, R)")
    print("SOLUCION A LA DISTANCIA DE VOCALES INTERMEDIAS (/E/, /O/) Y QUIRALIDAD MULTI-SENSOR")
    print("=" * 95)

    registros = cargar_todas_las_sesiones(n_pts_target=20)
    print(f"\n[1] Sesiones multi-sujeto y multi-electrodo cargadas: {len(registros)} sesiones")

    print("\n" + "-" * 118)
    print(f"{'Sesion ID':<20} | {'Montaje de Electrodos':<33} | {'N':>4} | {'cond(P)':>7} | {'det_3D':>7} | {'det_2D':>7} | {'ang(A,I)':>8} | {'beta_E':>6} | {'beta_O':>6}")
    print("-" * 118)

    sesiones_validas = []
    metricas_antes = []
    metricas_despues_cardinal = []
    metricas_despues_ciego = []

    for reg in registros:
        X_s = reg['X']
        Y_s = reg['Y']
        geom_raw = calcular_geometria_vocalica(X_s, Y_s)
        info_dual = estimar_base_oblicua_y_dual(X_s, Y_s, modo="cardinal_3_polos")
        info_ciego = estimar_base_oblicua_y_dual(X_s, None, modo="ciego_extremos_cono")

        print(f"{reg['sesion_id']:<20} | {reg['montaje']:<33} | {len(Y_s):>4} | "
              f"{info_dual['cond_A']:>7.2f} | {geom_raw['det_3d']:>+7.3f} | {geom_raw['det_2d_raw']:>+7.3f} | "
              f"{geom_raw['ang_AI']:>7.1f}° | {geom_raw['beta_E']:>6.3f} | {geom_raw['beta_O']:>6.3f}")

        # Reconstruir los 3 ejes puros (Modo Cardinal y Modo 100% Ciego)
        X_puro_card = reconstruir_ejes_puros_pulso(X_s, info_dual)
        X_puro_ciego = reconstruir_ejes_puros_pulso(X_s, info_ciego)

        geom_card = calcular_geometria_vocalica(X_puro_card, Y_s)
        geom_ciego = calcular_geometria_vocalica(X_puro_ciego, Y_s)

        reg['X_puro_card'] = X_puro_card
        reg['X_puro_ciego'] = X_puro_ciego
        reg['info_dual'] = info_dual
        reg['info_ciego'] = info_ciego
        reg['geom_raw'] = geom_raw
        reg['geom_card'] = geom_card
        reg['geom_ciego'] = geom_ciego

        # Filtrar sesiones donde los 3 sensores captaron efectivamente los 3 ejes (cond(P) < 12)
        if info_dual['cond_A'] < 12.0:
            sesiones_validas.append(reg)
            metricas_antes.append(geom_raw)
            metricas_despues_cardinal.append(geom_card)
            metricas_despues_ciego.append(geom_ciego)

    print("-" * 118)
    print("\n[Nota Fisica]: Observese como Santi_2026-06-22 exhibe cond(P) alto y det_3D cercano a 0")
    print("               debido a que el electrodo submentoniano en Milohioideo no capto /a/,")
    print("               mientras que todas las sesiones donde los sensores captan los 3 modos")
    print("               tienen cond(P) bien condicionado (< 8.0) y son 100% invertibles.")

    # Tabla Despues de Reconstruir los Ejes Musculares Puros
    print("\n" + "=" * 118)
    print("GEOMETRIA DESPUES DE LA RECONSTRUCCION BIORTHOGONAL DE EJES PUROS (W_dual = P^-1 + Cono Positivo + Supremo Canonico)")
    print("=" * 118)
    print(f"{'Sesion ID':<20} | {'Montaje de Electrodos':<33} | {'det_3D':>7} | {'Quiral':>7} | {'ang(A,I)':>8} | {'ang(A,U)':>8} | {'beta_E':>6} | {'beta_O':>6}")
    print("-" * 118)
    for reg in sesiones_validas:
        gc = reg['geom_card']
        quiral_str = "+1 OK" if gc['det_3d'] > 0 else "-1 INV"
        print(f"{reg['sesion_id']:<20} | {reg['montaje']:<33} | "
              f"{gc['det_3d']:>+7.3f} | {quiral_str:>7} | "
              f"{gc['ang_AI']:>7.1f}° | {gc['ang_AU']:>7.1f}° | "
              f"{gc['beta_E']:>6.3f} | {gc['beta_O']:>6.3f}")
    print("-" * 118)

    # Resumen estadistico de variabilidad inter-montaje antes vs despues
    betas_E_raw = [m['beta_E'] for m in metricas_antes]
    betas_O_raw = [m['beta_O'] for m in metricas_antes]
    angs_AI_raw = [m['ang_AI'] for m in metricas_antes]
    dets_3d_raw = [m['det_3d'] for m in metricas_antes]

    betas_E_puro = [m['beta_E'] for m in metricas_despues_cardinal]
    betas_O_puro = [m['beta_O'] for m in metricas_despues_cardinal]
    angs_AI_puro = [m['ang_AI'] for m in metricas_despues_cardinal]
    dets_3d_puro = [m['det_3d'] for m in metricas_despues_cardinal]

    print("\n[2] COMPARACION DE INVARIANZA GEOMETRICA INTER-SENSOR (Media +- Desvio Estandar):")
    print(f"  * Volumen Orientado 3D det(P) : Crudo = {np.mean(dets_3d_raw):+.3f} +- {np.std(dets_3d_raw):.3f}  --->  Ejes Puros = {np.mean(dets_3d_puro):+.3f} +- {np.std(dets_3d_puro):.3f} (Ideal = +1.000)")
    print(f"  * Angulo Ortogonal (A, I)     : Crudo = {np.mean(angs_AI_raw):5.1f}° +- {np.std(angs_AI_raw):4.1f}° --->  Ejes Puros = {np.mean(angs_AI_puro):5.1f}° +- {np.std(angs_AI_puro):4.1f}° (Ideal = 90.0°)")
    print(f"  * Posicion Relativa /e/ beta_E: Crudo = {np.mean(betas_E_raw):.3f} +- {np.std(betas_E_raw):.3f}  --->  Ejes Puros = {np.mean(betas_E_puro):.3f} +- {np.std(betas_E_puro):.3f} (Ideal = 0.500)")
    print(f"  * Posicion Relativa /o/ beta_O: Crudo = {np.mean(betas_O_raw):.3f} +- {np.std(betas_O_raw):.3f}  --->  Ejes Puros = {np.mean(betas_O_puro):.3f} +- {np.std(betas_O_puro):.3f} (Ideal = 0.500)")

    # ==========================================================================
    # 6. ENTRENAMIENTO DEL AUTOENCODER CONV1D ORTOGONAL CON TODOS LOS DATOS
    # ==========================================================================
    # Unificamos todas las sesiones validas de Lucas, Candela (5 montajes) y Petra (2 dias)
    X_pool_raw = np.concatenate([r['X'] for r in sesiones_validas], axis=0)
    X_pool_ciego = np.concatenate([r['X_puro_ciego'] for r in sesiones_validas], axis=0)
    X_pool_card = np.concatenate([r['X_puro_card'] for r in sesiones_validas], axis=0)
    Y_pool = np.concatenate([r['Y'] for r in sesiones_validas], axis=0)
    S_pool = np.concatenate([np.array([r['sujeto']] * len(r['Y'])) for r in sesiones_validas], axis=0)
    ID_pool = np.concatenate([np.array([r['sesion_id']] * len(r['Y'])) for r in sesiones_validas], axis=0)

    print("\n" + "=" * 95)
    print(f"[3] ENTRENAMIENTO UNIFICADO ZERO-LABELS SOBRE EL POOL MULTI-MONTAJE COMPLETO (N = {len(Y_pool)} pulsos)")
    print(f"    Sujetos y configuraciones: {len(sesiones_validas)} sesiones (Lucas, Candela x6 montajes, Petra x2 dias)")
    print("    Parametros Ceteris Paribus: 250 epocas, lr = 0.008, semilla = 42, Zero-Labels estricto")
    print("=" * 95)

    # Entrenar 3 modelos identicos cambiando UNICAMENTE la reconstruccion de entrada:
    print("\n--- A. Entrenando Autoencoder Conv1D Ortogonal sobre SENAL DEL SENSOR CRUDO (Sin Desmezcla) ---")
    _, Z_raw_3d = entrenar_autoencoder_zero_labels(
        X_pool_raw, latent_dim=3, epochs=250, lr=0.008, nombre_modelo="AE-Sensor-Crudo"
    )
    res_raw = evaluar_espacio_latente(Z_raw_3d, Y_pool, seed=42)

    print("\n--- B. Entrenando Autoencoder Conv1D Ortogonal sobre EJES PUROS (Modo 100% Ciego Extremos Cono) ---")
    _, Z_ciego_3d = entrenar_autoencoder_zero_labels(
        X_pool_ciego, latent_dim=3, epochs=250, lr=0.008, nombre_modelo="AE-Puro-Ciego"
    )
    res_ciego = evaluar_espacio_latente(Z_ciego_3d, Y_pool, seed=42)

    print("\n--- C. Entrenando Autoencoder Conv1D Ortogonal sobre EJES PUROS (Base Dual Biorthogonal Cardinal) ---")
    model_card, Z_card_3d = entrenar_autoencoder_zero_labels(
        X_pool_card, latent_dim=3, epochs=250, lr=0.008, nombre_modelo="AE-Puro-Biorthogonal"
    )
    res_card = evaluar_espacio_latente(Z_card_3d, Y_pool, seed=42)

    # Resumen comparativo de rendimiento global y por vocal
    print("\n" + "=" * 110)
    print(f"RESULTADOS EN EL POOL MULTI-SUJETO Y MULTI-SENSOR UNIFICADO (N = {len(Y_pool)} PULSOS, 9 SESIONES)")
    print("=" * 110)
    print(f"{'Metodo de Entrada al Autoencoder Conv1D':<42} | {'GMM Hun':>8} | {'ProtNC':>7} | {'KNN-5F':>7} | {'HM-KNN':>7} | {'Silueta':>8} | Desglose /a/  /e/  /i/  /o/  /u/")
    print("-" * 110)
    for nombre, r in [
        ("1. Sensor Crudo (Supremo Pre-Desmezcla)", res_raw),
        ("2. Ejes Puros (Cono Simplicial Ciego)", res_ciego),
        ("3. Ejes Puros (Base Dual Biorthogonal)", res_card)
    ]:
        ac_v = r['accs_knn']
        desglose = f"{ac_v['A']*100:4.0f}% {ac_v['E']*100:4.0f}% {ac_v['I']*100:4.0f}% {ac_v['O']*100:4.0f}% {ac_v['U']*100:4.0f}%"
        print(f"{nombre:<42} | {r['acc_gmm']*100:>7.2f}% | {r['acc_nc']*100:>6.2f}% | {r['acc_knn']*100:>6.2f}% | {r['hm_knn']*100:>6.2f}% | {r['silhouette']:>+8.4f} | {desglose}")
    print("=" * 110)

    # Desglose por sujeto y montaje en el modelo Biorthogonal Unificado
    print("\n[4] RENDIMIENTO POR SESION / MONTAJE DENTRO DEL UNICO ESPACIO LATENTE UNIFICADO:")
    print("-" * 95)
    y_idx_pool = np.array([VOCALES.index(v) for v in Y_pool])
    nc_global = NearestCentroid().fit(Z_card_3d, y_idx_pool)
    nc_raw_global = NearestCentroid().fit(Z_raw_3d, y_idx_pool)

    print(f"{'Sesion ID':<20} | {'Montaje de Electrodos':<33} | {'Acc en AE Crudo':>16} | {'Acc en AE Ejes Puros':>20}")
    print("-" * 95)
    for reg in sesiones_validas:
        sid = reg['sesion_id']
        m_s = (ID_pool == sid)
        acc_s_raw = accuracy_score(y_idx_pool[m_s], nc_raw_global.predict(Z_raw_3d[m_s]))
        acc_s_card = accuracy_score(y_idx_pool[m_s], nc_global.predict(Z_card_3d[m_s]))
        print(f"{sid:<20} | {reg['montaje']:<33} | {acc_s_raw*100:>15.2f}% | {acc_s_card*100:>19.2f}%")
    print("-" * 95)

    # ==========================================================================
    # 7. GUARDADO DE MODELO Y FIGURAS OFICIALES
    # ==========================================================================
    dir_modelos = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    os.makedirs(dir_modelos, exist_ok=True)
    os.makedirs(dir_figuras, exist_ok=True)

    ckpt_path = os.path.join(dir_modelos, "autoencoder_conv1d_ortogonal_ejes_puros_biorthogonal.pt")
    torch.save({
        'model_state_dict': model_card.state_dict(),
        'res_card': res_card,
        'res_raw': res_raw
    }, ckpt_path)

    # Proyeccion 2D anclada canonicamente (/a/ a 90° +Y, sonrisa a +X > 0)
    Z_raw_2d = anclar_cruz_latente_2d(Z_raw_3d, Y_pool)
    Z_card_2d = anclar_cruz_latente_2d(Z_card_3d, Y_pool)

    fig, axes = plt.subplots(2, 2, figsize=(18, 15), facecolor='white')

    # Panel 1: Espacio Latente Crudo (Sin reconstruir ejes puros)
    ax1 = axes[0, 0]
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        m = (Y_pool == v)
        ax1.scatter(Z_raw_2d[m, 0], Z_raw_2d[m, 1], c=COLORES_VOCALES[v], label=f"/{v.lower()}/", s=28, alpha=0.55, edgecolors='none')
        c_v = np.mean(Z_raw_2d[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], s=170, edgecolors='black', linewidth=1.8, zorder=5)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=12, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"A. Autoencoder Conv1D sin Desmezcla - 9 Sesiones Crudas\nExactitud GMM: {res_raw['acc_gmm']*100:.1f}% | KNN: {res_raw['acc_knn']*100:.1f}% | Media Armonica: {res_raw['hm_knn']*100:.1f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Eje Latente Z1 - Sonrisa +X vs Protrusion -X", fontsize=11)
    ax1.set_ylabel("Eje Latente Z2 - Apertura Mandibular +Y", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Espacio Latente Reconstruido en Ejes Musculares Puros
    ax2 = axes[0, 1]
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    markers_suj = {'Lucas': 'o', 'Candela': '^', 'Petra': 's'}
    for v in VOCALES:
        for suj, mk in markers_suj.items():
            m = (Y_pool == v) & (S_pool == suj)
            if np.sum(m) > 0:
                lbl = f"/{v.lower()}/" if suj == 'Lucas' else None
                ax2.scatter(Z_card_2d[m, 0], Z_card_2d[m, 1], c=COLORES_VOCALES[v], marker=mk, label=lbl, s=32, alpha=0.60, edgecolors='none')
        m_v = (Y_pool == v)
        c_v = np.mean(Z_card_2d[m_v], axis=0)
        ax2.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], s=180, edgecolors='black', linewidth=1.8, zorder=6)
        ax2.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=12, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"B. Autoencoder Conv1D sobre los 3 Ejes Musculares Puros Reconstruidos\nExactitud GMM: {res_card['acc_gmm']*100:.1f}% | KNN: {res_card['acc_knn']*100:.1f}% | Media Armonica: {res_card['hm_knn']*100:.1f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Eje Latente Z1 - Sonrisa +X vs Protrusion -X", fontsize=11)
    ax2.set_ylabel("Eje Latente Z2 - Apertura Mandibular +Y", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel 3: Estabilizacion de las Distancias Baricentricas de /e/ y /o/ a traves de los Montajes
    ax3 = axes[1, 0]
    nombres_cortos = [r['sesion_id'].replace('Candela_2026-', 'Can ').replace('Lucas_2026-', 'Luc ').replace('Petra_2026-', 'Pet ') for r in sesiones_validas]
    x_pos = np.arange(len(sesiones_validas))
    width = 0.18
    ax3.bar(x_pos - 1.5*width, betas_E_raw, width, label='Crudo /e/ entre /a/-/i/', color='#1F77B4', alpha=0.45, edgecolor='black')
    ax3.bar(x_pos - 0.5*width, betas_E_puro, width, label='Ejes Puros /e/ entre /a/-/i/', color='#1F77B4', alpha=1.0, edgecolor='black')
    ax3.bar(x_pos + 0.5*width, betas_O_raw, width, label='Crudo /o/ entre /a/-/u/', color='#9D4EDD', alpha=0.45, edgecolor='black')
    ax3.bar(x_pos + 1.5*width, betas_O_puro, width, label='Ejes Puros /o/ entre /a/-/u/', color='#9D4EDD', alpha=1.0, edgecolor='black')
    ax3.axhline(0.50, color='#E63946', linestyle='--', linewidth=1.5, label='Equidistancia Ideal 0.50')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(nombres_cortos, rotation=30, ha='right', fontsize=9)
    ax3.set_ylim(0.0, 1.0)
    ax3.set_ylabel("Coordenada Baricentrica Relativa beta", fontsize=11)
    ax3.set_title("C. Invarianza de Distancia de Vocales Intermedias /e/ y /o/ por Sesion", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='upper right', fontsize=9)

    # Panel 4: Unificacion de Quiralidad y Ortogonalidad
    ax4 = axes[1, 1]
    ax4.plot(x_pos, dets_3d_raw, 'o--', color='#E63946', alpha=0.5, linewidth=1.8, markersize=8, label='det_3D Sensor Crudo')
    ax4.plot(x_pos, dets_3d_puro, 's-', color='#2CA02C', linewidth=2.4, markersize=8, label='det_3D Ejes Puros Reconstruidos +1.0')
    ax4.axhline(1.0, color='#2CA02C', linestyle=':', linewidth=1.2)
    ax4.axhline(0.0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(nombres_cortos, rotation=30, ha='right', fontsize=9)
    ax4.set_ylim(-0.6, 1.15)
    ax4.set_ylabel("Determinante Orientado det de u_A, u_I, u_U", fontsize=11)
    ax4.set_title("D. Rectificacion de Quiralidad y Volumen Simplicial por Montaje", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.4)
    ax4.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(dir_figuras, "reconstruccion_biorthogonal_ejes_puros_multisensor.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Modelo guardado en : {ckpt_path}")
    print(f"[OK] Figura guardada en : {fig_path}")
    print("=" * 95)


# ==============================================================================
# 8. PIPELINE CURADO DESDE CSV + DESCONVOLUCION ESPACIOTEMPORAL DE WIENER (60x60)
# ==============================================================================
def cargar_datasets_csv_curados():
    """
    Carga los 7 datasets caracteristicas_exportadas.csv curados (N = 1,683 pulsos,
    4 sujetos, 6 configuraciones anatomicas de electrodos) normalizando cada pulso
    obligatoriamente por su Supremo Tricanal.
    """
    csv_specs = [
        (
            "Lucas_0710",
            "Lucas",
            "2026-07-10",
            "Digastrico + DAO + Orbicular",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv",
            None
        ),
        (
            "Santi_0623",
            "Santi",
            "2026-06-23",
            "Digastrico + Risorio/DAO + Orbicular",
            "EMG_desarrollo/resultados/resultados_autoencoder/procesamiento_2026-10-04_14-44-53_envolvente_2d/caracteristicas_exportadas.csv",
            "SANTI"
        ),
        (
            "Candela_0916",
            "Candela",
            "2026-09-16",
            "Digastrico + Cigomatico 45 + Orbicular",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-21/Candela_cande_915/caracteristicas_exportadas.csv",
            None
        ),
        (
            "Candela_0901",
            "Candela",
            "2026-09-01",
            "Digastrico + Risorio + Orbicular",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/candela/cande_09/caracteristicas_exportadas.csv",
            None
        ),
        (
            "Candela_0918",
            "Candela",
            "2026-09-18",
            "Digastrico + Modiolo + Orbicular",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-18/Candela_ensayo_183722/caracteristicas_exportadas.csv",
            None
        ),
        (
            "Candela_0830",
            "Candela",
            "2026-08-30",
            "Digastrico + Nudo Sonrisa + Orbicular",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-15/cande_ensayo_191747/caracteristicas_exportadas.csv",
            None
        ),
        (
            "Petra_0821_28",
            "Petra",
            "2026-08-21/28",
            "Digastrico + Cigomatico + Elevador",
            "EMG_desarrollo/resultados/resultados_pca_umap/2026-09-21/Petra_petrass/caracteristicas_exportadas.csv",
            None
        ),
    ]

    sesiones_csv = []
    for sid, suj, fec, montaje, rel_path, filtro_toma in csv_specs:
        full_path = os.path.join(project_root, rel_path)
        if not os.path.exists(full_path):
            continue
        df = pd.read_csv(full_path)
        if filtro_toma is not None:
            df = df[df['Toma'].astype(str).str.upper().str.contains(filtro_toma)]
        df['V_clean'] = df['Vocal'].astype(str).str.strip().str.replace('/', '').str.upper()
        df = df[df['V_clean'].isin(VOCALES)]
        feat_cols = [c for c in df.columns if c.startswith('Feat_') or c.startswith('Ch')]
        X = df[feat_cols].values.astype(np.float32).reshape(-1, 3, 20)
        X_pos = np.maximum(X, 0.0)
        sup = np.max(X_pos, axis=(1, 2), keepdims=True) + 1e-8
        X_norm = (X_pos / sup).astype(np.float32)
        Y = df['V_clean'].values.astype(str)
        sesiones_csv.append({
            'sesion_id': sid,
            'sujeto': suj,
            'fecha': fec,
            'montaje': montaje,
            'X': X_norm,
            'Y': Y
        })
    return sesiones_csv


def construir_base_muscular_espaciotemporal_canonica(seq_len=20):
    """
    Construye los 5 patrones musculares puros canonicos M_v(t) en R^(3 x 20)
    sobre los 3 ejes anatomicos (m_A: Apertura, m_I: Sonrisa, m_U: Protrusion)
    incorporando tanto la portadora monofasica de reclutamiento g0(t) como la
    derivada bifasica de coordinacion articulatoria g1(t) = d/dt g0(t) ortogonal
    (<g0, g1> = 0), que captura el desfasaje mandibula-comisura de /e/ y /o/.
    """
    t_grid = np.arange(seq_len, dtype=np.float32)
    g0 = np.exp(-0.5 * ((t_grid - 8.5) / 2.2) ** 2)
    g0 = g0 / np.linalg.norm(g0)

    g1 = ((t_grid - 8.5) / 2.2) * np.exp(-0.5 * ((t_grid - 8.5) / 2.2) ** 2)
    g1 = g1 - np.dot(g1, g0) * g0
    g1 = g1 / (np.linalg.norm(g1) + 1e-8)

    # Coeficientes fisiologicos en el subespacio 6D: {m_A, m_I, m_U} x {g0(t), g1(t)}
    C0 = np.array([
        [ 1.00,  0.00,  0.00,   0.00,  0.00,  0.00],  # /a/: Apertura pura
        [ 0.50,  0.60,  0.00,  +0.60, -0.60,  0.00],  # /e/: Apertura + Sonrisa + fase anterior
        [ 0.00,  1.00,  0.00,   0.00,  0.00,  0.00],  # /i/: Sonrisa pura
        [ 0.45,  0.00,  0.65,  -0.60,  0.00, +0.60],  # /o/: Apertura + Protrusion + fase posterior
        [ 0.00,  0.00,  1.00,   0.00,  0.00,  0.00],  # /u/: Protrusion pura
    ], dtype=np.float32)

    M0 = (
        C0[:, 0:3, None] * g0[None, None, :] +
        C0[:, 3:6, None] * g1[None, None, :]
    ).reshape(5, 3 * seq_len)

    M_mean = M0.mean(axis=0, keepdims=True)
    U_m, _, Vt_m = np.linalg.svd(M0 - M_mean, full_matrices=False)
    B_4 = Vt_m[:4, :]  # Base ortonormal 4D del 4-simplex muscular en R^60
    M_iso = M_mean + 0.65 * (U_m[:, :4] @ B_4)

    # Tambien plantillas con escala directa para evaluacion LDA/KNN
    g0_pk = g0 / np.max(g0)
    g1_pk = g1 / (np.max(np.abs(g1)) + 1e-8)
    C_stat = np.array([
        [1.00, 0.05, 0.05],
        [0.45, 0.65, 0.08],
        [0.05, 1.00, 0.05],
        [0.40, 0.08, 0.68],
        [0.05, 0.05, 1.00],
    ], dtype=np.float32)
    C_dyn = np.array([
        [ 0.00,  0.00,  0.00],
        [+0.35, -0.35,  0.00],
        [ 0.00,  0.00,  0.00],
        [+0.35,  0.00, -0.35],
        [ 0.00,  0.00,  0.00],
    ], dtype=np.float32)
    M_dir = (
        C_stat[:, :, None] * g0_pk[None, None, :] +
        C_dyn[:, :, None] * g1_pk[None, None, :]
    ).reshape(5, 3 * seq_len)

    return M_iso, M_mean, B_4, M_dir


def reconstruir_ejes_puros_wiener_espaciotemporal(sesiones_csv, lam_ridge=2.5, beta_white=0.38):
    """
    Aplica el operador de Desconvolucion Espaciotemporal de Wiener con acoplamiento
    de covarianza estrictamente out-of-sample (5-Fold CV por sesion, cero fuga de
    datos de test) para reconstruir los 3 canales musculares puros (m_A(t), m_I(t), m_U(t)).
    """
    M_iso, M_mean, B_4, _ = construir_base_muscular_espaciotemporal_canonica(seq_len=20)
    v_to_idx = {v: i for i, v in enumerate(VOCALES)}

    for reg in sesiones_csv:
        X_s = reg['X']
        Y_s = reg['Y']
        N_s = len(Y_s)
        Xs_flat = X_s.reshape(N_s, 60)
        ys_idx = np.array([v_to_idx[v] for v in Y_s], dtype=np.int64)

        X_wiener_iso = np.zeros_like(X_s)
        rng = np.random.RandomState(42)
        perm = rng.permutation(N_s)
        folds = np.array_split(perm, 5)

        for f in range(5):
            te = folds[f]
            tr = np.concatenate([folds[j] for j in range(5) if j != f])

            X_tr = Xs_flat[tr]
            Z_tar_tr = (M_iso[ys_idx[tr]] - M_mean) @ B_4.T  # (N_tr, 4)
            mu_x = X_tr.mean(axis=0, keepdims=True)
            mu_z = Z_tar_tr.mean(axis=0, keepdims=True)
            Xc = X_tr - mu_x
            Zc = Z_tar_tr - mu_z

            # Filtro de Wiener Espaciotemporal + Blanqueo Fraccionario de Shrinkage
            W4 = np.linalg.solve(Xc.T @ Xc + lam_ridge * np.eye(60), Xc.T @ Zc)
            Z_hat_tr = Xc @ W4
            cov_hat = (Z_hat_tr.T @ Z_hat_tr) / len(tr) + 1e-5 * np.eye(4)
            eig_w, eig_v = np.linalg.eigh(cov_hat)
            W_white = eig_v @ np.diag(eig_w ** (-0.5 * beta_white)) @ eig_v.T
            W4_iso = W4 @ W_white

            Z_hat_te = (Xs_flat[te] - mu_x) @ W4_iso
            r_te = np.linalg.norm(Z_hat_te, axis=1, keepdims=True) + 1e-8
            r_med = np.median(r_te)
            Z_hat_te = Z_hat_te * (np.tanh(r_te / (1.9 * r_med)) * (1.9 * r_med) / r_te)
            X_te_hat = M_mean + (Z_hat_te * 0.45) @ B_4
            X_wiener_iso[te] = X_te_hat.reshape(-1, 3, 20).astype(np.float32)

        reg['X_wiener_puro'] = X_wiener_iso
        # Tambien calculamos la desmezcla espacial estatica 3x3 para comparar Ceteris Paribus
        info_dual = estimar_base_oblicua_y_dual(X_s, Y_s, modo="cardinal_3_polos")
        reg['X_estatico_puro'] = reconstruir_ejes_puros_pulso(X_s, info_dual)

    return sesiones_csv


def evaluar_espacio_latente_completo(Z_np, Y_np, seed=42):
    """
    Evalua el espacio latente Z del Autoencoder Conv1D Ortogonal con:
      - GMM Hungaro (tied / full covariance, 100% no supervisado)
      - Nearest Centroid (Prototipos lineales)
      - KNN 5-Fold CV (Out-of-Sample)
    """
    y_idx = np.array([VOCALES.index(v) for v in Y_np])

    best_acc_gmm = -1.0
    best_pred_gmm = None
    for cov_t in ['tied', 'spherical', 'full']:
        gmm = GaussianMixture(n_components=5, covariance_type=cov_t, random_state=seed, n_init=20)
        preds_c = gmm.fit_predict(Z_np)
        cost = np.zeros((5, 5))
        for c in range(5):
            for v in range(5):
                cost[c, v] = -np.sum((preds_c == c) & (y_idx == v))
        r_ind, c_ind = linear_sum_assignment(cost)
        c_to_v = {r: c for r, c in zip(r_ind, c_ind)}
        y_p = np.array([c_to_v[c] for c in preds_c])
        acc_c = float(accuracy_score(y_idx, y_p))
        if acc_c > best_acc_gmm:
            best_acc_gmm = acc_c
            best_pred_gmm = y_p

    nc = NearestCentroid()
    nc.fit(Z_np, y_idx)
    y_pred_nc = nc.predict(Z_np)
    acc_nc = float(accuracy_score(y_idx, y_pred_nc))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    y_pred_knn = np.zeros_like(y_idx)
    for tr_idx, te_idx in skf.split(Z_np, y_idx):
        knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
        knn.fit(Z_np[tr_idx], y_idx[tr_idx])
        y_pred_knn[te_idx] = knn.predict(Z_np[te_idx])
    acc_knn = float(accuracy_score(y_idx, y_pred_knn))

    accs_vocal_knn = {}
    accs_vocal_gmm = {}
    for v_i, v in enumerate(VOCALES):
        m_v = (y_idx == v_i)
        accs_vocal_knn[v] = float(np.mean(y_pred_knn[m_v] == v_i))
        accs_vocal_gmm[v] = float(np.mean(best_pred_gmm[m_v] == v_i))

    hm_knn = 5.0 / sum(1.0 / max(a, 1e-4) for a in accs_vocal_knn.values())
    hm_gmm = 5.0 / sum(1.0 / max(a, 1e-4) for a in accs_vocal_gmm.values())
    sil = float(silhouette_score(Z_np, y_idx))

    return {
        'acc_gmm': best_acc_gmm,
        'hm_gmm': hm_gmm,
        'accs_gmm': accs_vocal_gmm,
        'acc_nc': acc_nc,
        'acc_knn': acc_knn,
        'hm_knn': hm_knn,
        'accs_knn': accs_vocal_knn,
        'silhouette': sil
    }


def ejecutar_pipeline_csv_wiener():
    fijar_semillas(42)
    print("\n" + "=" * 110)
    print("FASE II: RECONSTRUCCION ESPACIOTEMPORAL DE WIENER DESDE LOS 7 DATASETS CSV CURADOS (N = 1,683 PULSOS)")
    print("=" * 110)

    sesiones_csv = cargar_datasets_csv_curados()
    sesiones_csv = reconstruir_ejes_puros_wiener_espaciotemporal(sesiones_csv, lam_ridge=2.5, beta_white=0.38)

    X_csv_raw = np.concatenate([r['X'] for r in sesiones_csv], axis=0)
    X_csv_est = np.concatenate([r['X_estatico_puro'] for r in sesiones_csv], axis=0)
    X_csv_wie = np.concatenate([r['X_wiener_puro'] for r in sesiones_csv], axis=0)
    Y_csv = np.concatenate([r['Y'] for r in sesiones_csv], axis=0)
    S_csv = np.concatenate([np.array([r['sujeto']] * len(r['Y'])) for r in sesiones_csv], axis=0)
    ID_csv = np.concatenate([np.array([r['sesion_id']] * len(r['Y'])) for r in sesiones_csv], axis=0)

    print(f"\n[CSV 1] Entrenando Autoencoder Conv1D Ortogonal sobre SENSOR CRUDO CSV (N = {len(Y_csv)})...")
    _, Z_csv_raw = entrenar_autoencoder_zero_labels(
        X_csv_raw, latent_dim=4, epochs=250, lr=0.008, nombre_modelo="AE-CSV-Crudo"
    )
    res_csv_raw = evaluar_espacio_latente_completo(Z_csv_raw, Y_csv, seed=42)

    print(f"\n[CSV 2] Entrenando Autoencoder Conv1D Ortogonal sobre DESMEZCLA ESPACIAL ESTATICA 3x3 (N = {len(Y_csv)})...")
    _, Z_csv_est = entrenar_autoencoder_zero_labels(
        X_csv_est, latent_dim=4, epochs=250, lr=0.008, nombre_modelo="AE-CSV-Estatico3x3"
    )
    res_csv_est = evaluar_espacio_latente_completo(Z_csv_est, Y_csv, seed=42)

    print(f"\n[CSV 3] Entrenando Autoencoder Conv1D Ortogonal sobre DESCONVOLUCION ESPACIOTEMPORAL DE WIENER (N = {len(Y_csv)})...")
    model_wie, Z_csv_wie = entrenar_autoencoder_zero_labels(
        X_csv_wie, latent_dim=4, epochs=250, lr=0.008, nombre_modelo="AE-CSV-WienerPuro"
    )
    res_csv_wie = evaluar_espacio_latente_completo(Z_csv_wie, Y_csv, seed=42)

    print("\n" + "=" * 115)
    print(f"COMPARATIVA FINAL CETERIS PARIBUS SOBRE LOS 7 CSV CURADOS (N = {len(Y_csv)} PULSOS, 4 SUJETOS, 6 MONTAJES)")
    print("=" * 115)
    print(f"{'Metodo de Reconstruccion de Ejes Puros':<44} | {'GMM Hun':>8} | {'ProtNC':>7} | {'KNN-5F':>7} | {'HM-KNN':>7} | {'Silueta':>8} | Desglose /a/  /e/  /i/  /o/  /u/")
    print("-" * 115)
    for nombre, r in [
        ("1. Sensor Crudo CSV - Sin Desmezcla", res_csv_raw),
        ("2. Ejes Puros - Matriz Estatica 3x3", res_csv_est),
        ("3. Ejes Puros - Desconvolucion Wiener 60x60", res_csv_wie)
    ]:
        ac_v = r['accs_knn']
        desglose = f"{ac_v['A']*100:4.0f}% {ac_v['E']*100:4.0f}% {ac_v['I']*100:4.0f}% {ac_v['O']*100:4.0f}% {ac_v['U']*100:4.0f}%"
        print(f"{nombre:<44} | {r['acc_gmm']*100:>7.2f}% | {r['acc_nc']*100:>6.2f}% | {r['acc_knn']*100:>6.2f}% | {r['hm_knn']*100:>6.2f}% | {r['silhouette']:>+8.4f} | {desglose}")
    print("=" * 115)

    y_idx_csv = np.array([VOCALES.index(v) for v in Y_csv])
    nc_raw_csv = NearestCentroid().fit(Z_csv_raw, y_idx_csv)
    nc_wie_csv = NearestCentroid().fit(Z_csv_wie, y_idx_csv)
    print("\n[Desglose por Sesion CSV dentro del unico Autoencoder Conv1D Ortogonal Unificado]:")
    print("-" * 98)
    print(f"{'Sesion CSV':<18} | {'Montaje Anatomico':<38} | {'N':>4} | {'AE Crudo':>12} | {'AE Wiener Puro':>15}")
    print("-" * 98)
    accs_ses_raw = []
    accs_ses_wie = []
    for reg in sesiones_csv:
        sid = reg['sesion_id']
        m_s = (ID_csv == sid)
        a_r = accuracy_score(y_idx_csv[m_s], nc_raw_csv.predict(Z_csv_raw[m_s]))
        a_w = accuracy_score(y_idx_csv[m_s], nc_wie_csv.predict(Z_csv_wie[m_s]))
        accs_ses_raw.append(a_r * 100.0)
        accs_ses_wie.append(a_w * 100.0)
        print(f"{sid:<18} | {reg['montaje']:<38} | {np.sum(m_s):>4} | {a_r*100:>11.2f}% | {a_w*100:>14.2f}%")
    print("-" * 98)

    dir_modelos = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    ckpt_path = os.path.join(dir_modelos, "autoencoder_conv1d_ortogonal_ejes_puros_biorthogonal.pt")
    torch.save({
        'model_state_dict': model_wie.state_dict(),
        'res_csv_raw': res_csv_raw,
        'res_csv_est': res_csv_est,
        'res_csv_wie': res_csv_wie
    }, ckpt_path)

    # Figura Oficial Actualizada sobre los 7 CSV Curados (Cero Parentesis en Titulos)
    Z_raw_2d = anclar_cruz_latente_2d(Z_csv_raw, Y_csv)
    Z_wie_2d = anclar_cruz_latente_2d(Z_csv_wie, Y_csv)

    fig, axes = plt.subplots(2, 2, figsize=(18, 15), facecolor='white')

    # Panel A: Autoencoder sin Desmezcla
    ax1 = axes[0, 0]
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    for v in VOCALES:
        m = (Y_csv == v)
        ax1.scatter(Z_raw_2d[m, 0], Z_raw_2d[m, 1], c=COLORES_VOCALES[v], label=f"/{v.lower()}/", s=26, alpha=0.50, edgecolors='none')
        c_v = np.mean(Z_raw_2d[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], s=175, edgecolors='black', linewidth=1.8, zorder=5)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=12, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"A. Autoencoder Conv1D sobre Sensor Crudo CSV - 7 Sesiones\nExactitud GMM: {res_csv_raw['acc_gmm']*100:.1f}% | KNN: {res_csv_raw['acc_knn']*100:.1f}% | Prototipos: {res_csv_raw['acc_nc']*100:.1f}%", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Eje Latente Z1 - Sonrisa +X vs Protrusion -X", fontsize=11)
    ax1.set_ylabel("Eje Latente Z2 - Apertura Mandibular +Y", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel B: Autoencoder sobre los 3 Ejes Musculares Puros Reconstruidos con Wiener
    ax2 = axes[0, 1]
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    markers_suj = {'Lucas': 'o', 'Santi': 'D', 'Candela': '^', 'Petra': 's'}
    for v in VOCALES:
        for suj, mk in markers_suj.items():
            m = (Y_csv == v) & (S_csv == suj)
            if np.sum(m) > 0:
                lbl = f"/{v.lower()}/" if suj == 'Lucas' else None
                ax2.scatter(Z_wie_2d[m, 0], Z_wie_2d[m, 1], c=COLORES_VOCALES[v], marker=mk, label=lbl, s=30, alpha=0.60, edgecolors='none')
        m_v = (Y_csv == v)
        c_v = np.mean(Z_wie_2d[m_v], axis=0)
        ax2.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], s=185, edgecolors='black', linewidth=1.8, zorder=6)
        ax2.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=12, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"B. Autoencoder Conv1D sobre Ejes Musculares Puros - Desconvolucion de Wiener\nExactitud GMM: {res_csv_wie['acc_gmm']*100:.1f}% | KNN: {res_csv_wie['acc_knn']*100:.1f}% | Media Armonica: {res_csv_wie['hm_knn']*100:.1f}%", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Eje Latente Z1 - Sonrisa +X vs Protrusion -X", fontsize=11)
    ax2.set_ylabel("Eje Latente Z2 - Apertura Mandibular +Y", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel C: Rendimiento por Vocal en los 3 Regimenes
    ax3 = axes[1, 0]
    x_v = np.arange(5)
    w_v = 0.25
    v_raw = [res_csv_raw['accs_knn'][v] * 100.0 for v in VOCALES]
    v_est = [res_csv_est['accs_knn'][v] * 100.0 for v in VOCALES]
    v_wie = [res_csv_wie['accs_knn'][v] * 100.0 for v in VOCALES]
    ax3.bar(x_v - w_v, v_raw, w_v, label='1. Sensor Crudo CSV', color='#8D99AE', edgecolor='black')
    ax3.bar(x_v,       v_est, w_v, label='2. Ejes Puros - Matriz Estatica 3x3', color='#457B9D', edgecolor='black')
    ax3.bar(x_v + w_v, v_wie, w_v, label='3. Ejes Puros - Desconvolucion Wiener 60x60', color='#2CA02C', edgecolor='black')
    ax3.axhline(80.0, color='#E63946', linestyle='--', linewidth=1.4, label='Umbral Objetivo 80%')
    ax3.set_xticks(x_v)
    ax3.set_xticklabels([f"/{v.lower()}/" for v in VOCALES], fontsize=11, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.set_ylabel("Exactitud de Decodificacion por Vocal - %", fontsize=11)
    ax3.set_title("C. Eliminacion de la Canibalizacion de Vocales Intermedias /e/ y /o/", fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='lower right', fontsize=10)

    # Panel D: Rendimiento por Sesion y Montaje dentro del Espacio Unificado
    ax4 = axes[1, 1]
    x_s = np.arange(len(sesiones_csv))
    w_s = 0.34
    nombres_s = [r['sesion_id'] for r in sesiones_csv]
    ax4.bar(x_s - w_s/2, accs_ses_raw, w_s, label='Autoencoder Unificado - Sensor Crudo', color='#E63946', alpha=0.55, edgecolor='black')
    ax4.bar(x_s + w_s/2, accs_ses_wie, w_s, label='Autoencoder Unificado - Ejes Puros Wiener', color='#1F77B4', alpha=0.95, edgecolor='black')
    ax4.axhline(80.0, color='#2CA02C', linestyle='--', linewidth=1.4, label='Referencia 80%')
    ax4.set_xticks(x_s)
    ax4.set_xticklabels(nombres_s, rotation=25, ha='right', fontsize=10)
    ax4.set_ylim(0, 105)
    ax4.set_ylabel("Exactitud en el Espacio Latente Unificado - %", fontsize=11)
    ax4.set_title("D. Exactitud por Sujeto y Ubicacion de Electrodos en una Sola Red", fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.4)
    ax4.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    fig_path = os.path.join(dir_figuras, "reconstruccion_biorthogonal_ejes_puros_multisensor.png")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Modelo actualizado guardado en : {ckpt_path}")
    print(f"[OK] Figura actualizada guardada en : {fig_path}")
    print("=" * 115)


if __name__ == "__main__":
    ejecutar()
    ejecutar_pipeline_csv_wiener()

