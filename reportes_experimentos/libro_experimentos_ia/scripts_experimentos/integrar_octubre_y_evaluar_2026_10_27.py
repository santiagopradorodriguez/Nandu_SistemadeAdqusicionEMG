#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Experimento: Integracion de Sesiones de Octubre (2026-10-20, 10-23, 10-28,
#              10-29, 10-30) al Dataset Unificado con Reconstruccion 12D de Ejes
#              Musculares Puros y Evaluacion Ciega sobre 2026-10-27 Aislado.
# ==============================================================================

import os
import sys
import re
import time
import random
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import fractional_matrix_power
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

PROJECT_ROOT = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
EMG_DIR = os.path.join(PROJECT_ROOT, "EMG_desarrollo")
SCRIPTS_EXP_DIR = os.path.join(PROJECT_ROOT, "reportes_experimentos/libro_experimentos_ia/scripts_experimentos")

for p in [PROJECT_ROOT, EMG_DIR, SCRIPTS_EXP_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from deep_learning.motor_autoencoder_unificado import extraer_dataset_unificado

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}


def fijar_semillas(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ConvOrthogonalAutoencoder(nn.Module):
    def __init__(self, in_channels=3, seq_len=20, latent_dim=4):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.enc_conv = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.flat_dim = 32 * (seq_len // 4)
        self.fc_enc = nn.Linear(self.flat_dim, latent_dim, bias=False)
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, in_channels, kernel_size=5, padding=2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc_conv(x).reshape(x.size(0), -1)
        return self.fc_enc(h)

    def forward(self, x):
        z = self.encode(x)
        h_rec = self.fc_dec(z).reshape(x.size(0), 32, self.seq_len // 4)
        x_rec = self.dec_conv(h_rec)
        return x_rec, z

    def orthogonal_loss(self):
        W = self.fc_enc.weight
        WTW = torch.matmul(W, W.t())
        I = torch.eye(self.latent_dim, device=W.device)
        return torch.norm(WTW - I, p='fro') ** 2


def entrenar_autoencoder_zero_labels(X_np, latent_dim=4, epochs=250, lr=0.008, nombre_modelo="AE"):
    fijar_semillas(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvOrthogonalAutoencoder(in_channels=3, seq_len=20, latent_dim=latent_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
    dataset = torch.utils.data.TensorDataset(X_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_mse, ep_ort = 0.0, 0.0
        for (bx,) in loader:
            optimizer.zero_grad()
            x_rec, z = model(bx)
            loss_mse = nn.functional.mse_loss(x_rec, bx)
            loss_ort = model.orthogonal_loss()
            z_c = z - z.mean(dim=0, keepdim=True)
            cov_z = (z_c.t() @ z_c) / max(len(bx) - 1, 1)
            off_diag = cov_z - torch.diag(torch.diag(cov_z))
            loss_cov = torch.norm(off_diag, p='fro') ** 2
            loss = loss_mse + 0.05 * loss_ort + 0.02 * loss_cov
            loss.backward()
            optimizer.step()
            ep_mse += loss_mse.item() * len(bx)
            ep_ort += loss_ort.item() * len(bx)
        scheduler.step()
        if ep in [1, 50, 100, 150, 200, epochs]:
            dt = (time.time() - t0) / ep
            eta = dt * (epochs - ep)
            print(f"  [{nombre_modelo}] Epoca {ep:3d}/{epochs} ({ep/epochs*100:5.1f}%) | MSE: {ep_mse/len(X_np):.5f} | Ortho: {ep_ort/len(X_np):.4f} | ETA: {eta:.1f}s", flush=True)

    model.eval()
    with torch.no_grad():
        Z_out = model.encode(X_t).cpu().numpy()
    return model.cpu(), Z_out

BASE_DATOS_DIR = os.path.join(EMG_DIR, "base_de_datos_electrodos")
CACHE_OUT_DIR = os.path.join(EMG_DIR, "resultados/resultados_autoencoder/cache_datasets")
MODELOS_OUT_DIR = os.path.join(EMG_DIR, "resultados/resultados_autoencoder/modelos_entrenados")
FIGURAS_OUT_DIR = os.path.join(EMG_DIR, "resultados/resultados_autoencoder/figuras_evaluacion")
LIBRO_IMG_DIR = os.path.join(PROJECT_ROOT, "reportes_experimentos/libro_experimentos_ia/imagenes")

for d in [CACHE_OUT_DIR, MODELOS_OUT_DIR, FIGURAS_OUT_DIR, LIBRO_IMG_DIR]:
    os.makedirs(d, exist_ok=True)


# ==============================================================================
# 1. BASE CANONICA MUSCULAR Y CALIBRADOR FISICO 12D (HERMITE-GAUSS + WIENER)
# ==============================================================================
t_grid = np.arange(20, dtype=np.float64)
g0 = np.exp(-0.5 * ((t_grid - 8.5) / 2.8) ** 2)
g0 = g0 / np.max(g0)
g1 = -(t_grid - 8.5) / 2.8 * np.exp(-0.5 * ((t_grid - 8.5) / 2.8) ** 2)
g1 = g1 / np.max(np.abs(g1))

P_SPACE = {
    "A": np.array([1.00, 0.04, 0.04]),
    "E": np.array([0.68, 0.72, 0.04]),
    "I": np.array([0.04, 1.00, 0.04]),
    "O": np.array([0.68, 0.04, 0.72]),
    "U": np.array([0.04, 0.04, 1.00]),
}
P_BIPHASIC = {
    "A": np.array([0.00,  0.00,  0.00]),
    "E": np.array([0.25, -0.25,  0.00]),
    "I": np.array([0.00, -0.12,  0.00]),
    "O": np.array([0.25,  0.00, -0.25]),
    "U": np.array([0.00,  0.00, -0.12]),
}
T_CANON = {}
for v in VOCALES:
    M = np.outer(P_SPACE[v], g0) + np.outer(P_BIPHASIC[v], g1)
    M = np.clip(M, 0.0, None)
    M = M / (np.max(M) + 1e-12)
    T_CANON[v] = M

T_mat = np.array([T_CANON[v].reshape(-1) for v in VOCALES])
mu_T = np.mean(T_mat, axis=0, keepdims=True)
U_t, S_t, Vt_t = np.linalg.svd(T_mat - mu_T, full_matrices=False)
B_4 = Vt_t[:4]
Z_canon = {
    v: ((T_CANON[v].reshape(1, -1) - mu_T) @ B_4.T / (S_t[:4] + 1e-12)).reshape(-1)
    for v in VOCALES
}


def get_hermite_basis(t0, sigma=3.0, K=4):
    u = (t_grid - t0) / sigma
    Phi = np.zeros((K, 20))
    Phi[0] = np.exp(-0.5 * u**2)
    Phi[1] = u * np.exp(-0.5 * u**2)
    Phi[2] = (u**2 - 1.0) * np.exp(-0.5 * u**2)
    Phi[3] = (u**3 - 3.0 * u) * np.exp(-0.5 * u**2)
    Q, _ = np.linalg.qr(Phi.T)
    return Q.T[:K]


def extract_temporal_reduced_features(X_3x20, Phi):
    N = len(X_3x20)
    K = Phi.shape[0]
    F = np.zeros((N, 3 * K))
    for i in range(N):
        F[i] = (X_3x20[i] @ Phi.T).reshape(-1)
    return F


def fit_calibrator_12d(X_cal, y_cal, lam=1.5, beta=0.35):
    prof = np.mean(X_cal, axis=(0, 1))
    t0 = float(np.sum(t_grid * prof) / (np.sum(prof) + 1e-12))
    Phi = get_hermite_basis(t0, sigma=3.0, K=4)
    F_cal = extract_temporal_reduced_features(X_cal, Phi)
    mu_F = np.mean(F_cal, axis=0, keepdims=True)
    std_F = np.std(F_cal, axis=0, keepdims=True) + 1e-3
    Fc_cal = (F_cal - mu_F) / std_F
    Z_tr = np.array([Z_canon[v] for v in y_cal])
    D = F_cal.shape[1]
    G = Fc_cal.T @ Fc_cal
    lam_eff = lam * (np.trace(G) / D)
    W_4 = np.linalg.solve(G + lam_eff * np.eye(D), Fc_cal.T @ Z_tr)
    Z_hat_cal = Fc_cal @ W_4
    cov_Z = (Z_hat_cal.T @ Z_hat_cal) / max(len(Fc_cal), 1) + 0.05 * np.eye(4)
    W_whiten = np.real(fractional_matrix_power(cov_Z, -beta / 2.0))
    W_iso = W_4 @ W_whiten

    Z_cal_iso = Fc_cal @ W_iso
    X_cal_rec = np.clip((Z_cal_iso * S_t[:4]) @ B_4 + mu_T, 0.0, None).reshape(-1, 3, 20)
    med = np.zeros((1, 3, 1))
    for c, v_pole in enumerate(["A", "I", "U"]):
        mask = (y_cal == v_pole)
        med[0, c, 0] = np.median(np.max(X_cal_rec[mask, c, :], axis=1)) if np.any(mask) else 1.0

    return {
        "t0": t0,
        "Phi": Phi,
        "mu_F": mu_F,
        "std_F": std_F,
        "W_iso": W_iso,
        "med": med
    }


def apply_calibrator_12d(X_in, cal_params):
    F_in = extract_temporal_reduced_features(X_in, cal_params["Phi"])
    std_F = cal_params.get("std_F", 1.0)
    Fc_in = (F_in - cal_params["mu_F"]) / std_F
    Z_hat = Fc_in @ cal_params["W_iso"]
    X_rec = np.clip((Z_hat * S_t[:4]) @ B_4 + mu_T, 0.0, None).reshape(-1, 3, 20)
    X_s = X_rec / (cal_params["med"] + 1e-12)
    X_s = np.tanh(1.3 * X_s)
    sup = np.max(X_s, axis=(1, 2), keepdims=True) + 1e-12
    return (X_s / sup).astype(np.float32)


def calibrate_session_causal(X_s, y_s, k_per_vowel=7):
    cal_idx, test_idx = [], []
    for v in VOCALES:
        v_idx = np.where(y_s == v)[0]
        k_use = min(k_per_vowel, max(2, len(v_idx) // 2))
        cal_idx.extend(v_idx[:k_use])
        test_idx.extend(v_idx[k_use:])
    cal_idx = np.array(cal_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    cal_params = fit_calibrator_12d(X_s[cal_idx], y_s[cal_idx])
    X_all_rec = apply_calibrator_12d(X_s, cal_params)
    return X_all_rec, cal_idx, test_idx, cal_params


# ==============================================================================
# 2. EXTRACCION Y CARGA DE SESIONES HISTORICAS + OCTUBRE
# ==============================================================================
def cargar_csv_curado(csv_path, sesion_id, sujeto, fecha, montaje, filtro_toma=None):
    df = pd.read_csv(csv_path)
    if filtro_toma is not None:
        df = df[df["Toma"].astype(str).str.upper().str.contains(filtro_toma.upper())].reset_index(drop=True)
    df["V_clean"] = df["Vocal"].astype(str).str.strip().str.replace("/", "").str.upper()
    df = df[df["V_clean"].isin(VOCALES)].reset_index(drop=True)
    df["win_num"] = df["Toma"].apply(
        lambda s: int(re.search(r"Win(\d+)", str(s)).group(1)) if re.search(r"Win(\d+)", str(s)) else 0
    )
    df["serie_num"] = df["Toma"].apply(
        lambda s: 2 if ("_2_" in str(s) or "Prueba2" in str(s)) else 1
    )
    feat_cols = [c for c in df.columns if c.startswith("Feat_") or c.startswith("Ch")]
    df = df.sort_values(["serie_num", "win_num"]).reset_index(drop=True)
    X_3x20 = np.maximum(df[feat_cols].values.astype(np.float64).reshape(-1, 3, 20), 0.0)
    sup = np.max(X_3x20, axis=(1, 2), keepdims=True) + 1e-12
    return {
        "sesion_id": sesion_id,
        "sujeto": sujeto,
        "fecha": fecha,
        "montaje": montaje,
        "X": (X_3x20 / sup).astype(np.float32),
        "Y": df["V_clean"].values,
        "Tomas": df["Toma"].values
    }


def extraer_bloque_crudo_octubre(rutas_tomas, sesion_id, sujeto, fecha, montaje):
    out_tmp = os.path.join(CACHE_OUT_DIR, f"tmp_{sesion_id}")
    os.makedirs(out_tmp, exist_ok=True)
    csv_tmp = os.path.join(out_tmp, "caracteristicas_exportadas.csv")
    need_extract = not os.path.exists(csv_tmp)
    if not need_extract:
        try:
            df_chk = pd.read_csv(csv_tmp)
            if len(df_chk) < 50:
                need_extract = True
        except Exception:
            need_extract = True

    if need_extract:
        modo_alin = "Pico Envolvente EMG" if "sonido" in sesion_id.lower() else "Pico Volumen Micrófono"
        print(f"  -> Extrayendo bloque crudo [{sesion_id}] ({len(rutas_tomas)} tomas, alineacion={modo_alin})...")
        extraer_dataset_unificado(
            rutas_tomas=rutas_tomas,
            aplicar_correccion_intersesion=False,
            modo_alineacion=modo_alin,
            carpeta_salida=out_tmp,
            tipo_envolvente="rms",
            smooth_ms=100,
            target_len=20,
            outlier_contamination=0.10,
            canales_features=("canal_0", "canal_1", "canal_2")
        )
        if os.path.exists(csv_tmp) and len(pd.read_csv(csv_tmp)) < 50 and modo_alin != "Pico Envolvente EMG":
            print(f"  -> Fallback a Pico Envolvente EMG para [{sesion_id}]...")
            extraer_dataset_unificado(
                rutas_tomas=rutas_tomas,
                aplicar_correccion_intersesion=False,
                modo_alineacion="Pico Envolvente EMG",
                carpeta_salida=out_tmp,
                tipo_envolvente="rms",
                smooth_ms=100,
                target_len=20,
                outlier_contamination=0.10,
                canales_features=("canal_0", "canal_1", "canal_2")
            )
    return cargar_csv_curado(csv_tmp, sesion_id, sujeto, fecha, montaje)


def construir_corpus_completo():
    bloques_train = []

    # A. Las 7 sesiones historicas curadas (desglosando Lucas T1..T7 y Petra Dia1/Dia2)
    csv_lucas = os.path.join(EMG_DIR, "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/lucas/lucas_viejo_para_probar/caracteristicas_exportadas.csv")
    df_l = pd.read_csv(csv_lucas)
    df_l["V_clean"] = df_l["Vocal"].astype(str).str.strip().str.replace("/", "").str.upper()
    df_l = df_l[df_l["V_clean"].isin(VOCALES)].reset_index(drop=True)
    df_l["win_num"] = df_l["Toma"].apply(lambda s: int(re.search(r"Win(\d+)", str(s)).group(1)) if re.search(r"Win(\d+)", str(s)) else 0)
    df_l["t_id"] = df_l["Toma"].apply(lambda s: re.search(r"_(T\d+)_", str(s)).group(1) if re.search(r"_(T\d+)_", str(s)) else "T1")
    feat_cols_l = [c for c in df_l.columns if c.startswith("Feat_") or c.startswith("Ch")]
    for tid in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        sub = df_l[df_l["t_id"] == tid].sort_values(["win_num"]).reset_index(drop=True)
        X_3x20 = np.maximum(sub[feat_cols_l].values.astype(np.float64).reshape(-1, 3, 20), 0.0)
        sup = np.max(X_3x20, axis=(1, 2), keepdims=True) + 1e-12
        bloques_train.append({
            "sesion_id": f"Lucas_0710_{tid}",
            "sujeto": "Lucas",
            "fecha": "2026-07-10",
            "montaje": f"Digastrico + DAO + Orbicular ({tid})",
            "X": (X_3x20 / sup).astype(np.float32),
            "Y": sub["V_clean"].values,
            "Tomas": sub["Toma"].values,
            "k_cal": 4
        })

    # Petra Dia 1 y Dia 2
    csv_petra = os.path.join(EMG_DIR, "resultados/resultados_pca_umap/2026-09-21/Petra_petrass/caracteristicas_exportadas.csv")
    df_p = pd.read_csv(csv_petra)
    df_p["V_clean"] = df_p["Vocal"].astype(str).str.strip().str.replace("/", "").str.upper()
    df_p = df_p[df_p["V_clean"].isin(VOCALES)].reset_index(drop=True)
    df_p["win_num"] = df_p["Toma"].apply(lambda s: int(re.search(r"Win(\d+)", str(s)).group(1)) if re.search(r"Win(\d+)", str(s)) else 0)
    df_p["sub_day"] = df_p["Toma"].apply(lambda s: "Dia1" if "silicona" in str(s).lower() else "Dia2")
    feat_cols_p = [c for c in df_p.columns if c.startswith("Feat_") or c.startswith("Ch")]
    for sday, fec_p in [("Dia1", "2026-08-21"), ("Dia2", "2026-08-28")]:
        sub = df_p[df_p["sub_day"] == sday].sort_values(["win_num"]).reset_index(drop=True)
        X_3x20 = np.maximum(sub[feat_cols_p].values.astype(np.float64).reshape(-1, 3, 20), 0.0)
        sup = np.max(X_3x20, axis=(1, 2), keepdims=True) + 1e-12
        bloques_train.append({
            "sesion_id": f"Petra_{sday}",
            "sujeto": "Petra",
            "fecha": fec_p,
            "montaje": f"Digastrico + Cigomatico + Elevador ({sday})",
            "X": (X_3x20 / sup).astype(np.float32),
            "Y": sub["V_clean"].values,
            "Tomas": sub["Toma"].values,
            "k_cal": 7
        })

    # Santi 0623 y las 4 Candelas de ago/sep
    for sid, suj, fec, mont, rel_p, flt in [
        ("Santi_0623", "Santi", "2026-06-23", "Digastrico + Risorio/DAO + Orbicular",
         "resultados/resultados_autoencoder/procesamiento_2026-10-04_14-44-53_envolvente_2d/caracteristicas_exportadas.csv", "SANTI"),
        ("Candela_0830", "Candela", "2026-08-30", "Digastrico + Nudo Sonrisa + Orbicular",
         "resultados/resultados_pca_umap/2026-09-15/cande_ensayo_191747/caracteristicas_exportadas.csv", None),
        ("Candela_0901", "Candela", "2026-09-01", "Digastrico + Risorio + Orbicular",
         "resultados/resultados_pca_umap/2026-09-12/General_por_sujeto/candela/cande_09/caracteristicas_exportadas.csv", None),
        ("Candela_0916", "Candela", "2026-09-16", "Digastrico + Cigomatico 45 + Orbicular",
         "resultados/resultados_pca_umap/2026-09-21/Candela_cande_915/caracteristicas_exportadas.csv", None),
        ("Candela_0918", "Candela", "2026-09-18", "Digastrico + Modiolo + Orbicular",
         "resultados/resultados_pca_umap/2026-09-18/Candela_ensayo_183722/caracteristicas_exportadas.csv", None),
    ]:
        b = cargar_csv_curado(os.path.join(EMG_DIR, rel_p), sid, suj, fec, mont, flt)
        b["k_cal"] = 7
        bloques_train.append(b)

    # B. Sesiones Nuevas de Octubre para Train:
    # 1) 2026-10-23: Pruebatricanal_Candela (ya exportado en procesamiento_2026-10-24_00-04-25_envolvente_2d)
    csv_1023 = os.path.join(EMG_DIR, "resultados/resultados_autoencoder/procesamiento_2026-10-24_00-04-25_envolvente_2d/caracteristicas_exportadas.csv")
    if os.path.exists(csv_1023):
        b_1023 = cargar_csv_curado(csv_1023, "Candela_1023_Tricanal", "Candela", "2026-10-23", "Masetero + Modiolo + Orbicular")
        b_1023["k_cal"] = 7
        bloques_train.append(b_1023)

    # 2) 2026-10-20: Exp1_Candela, Exp2_Candela y Prueba_Santi
    dir_1020 = os.path.join(BASE_DATOS_DIR, "2026-10-20")
    if os.path.exists(dir_1020):
        all_1020 = sorted([os.path.join(dir_1020, d) for d in os.listdir(dir_1020) if os.path.isdir(os.path.join(dir_1020, d))])
        r_exp1 = [p for p in all_1020 if "Exp1_" in os.path.basename(p)]
        r_exp2 = [p for p in all_1020 if "Exp2_" in os.path.basename(p)]
        r_santi = [p for p in all_1020 if "Santi" in os.path.basename(p)]
        for r_list, sid, suj, mont in [
            (r_exp1, "Candela_1020_Exp1", "Candela", "Control Experimental 1 - 2026-10-20"),
            (r_exp2, "Candela_1020_Exp2", "Candela", "Control Experimental 2 - 2026-10-20"),
            (r_santi, "Santi_1020", "Santi", "Control Santi - 2026-10-20"),
        ]:
            if len(r_list) >= 5:
                b = extraer_bloque_crudo_octubre(r_list, sid, suj, "2026-10-20", mont)
                b["k_cal"] = 7
                bloques_train.append(b)

    # 3) 2026-10-28 (Prueba1/2_Candela) y 2026-10-29 (Pruebavertical_Candela)
    for folder_d, sid, mont in [
        ("2026-10-28", "Candela_1028_VocalesDistinto", "Config 27/10 - Prueba 1 y 2 Vocales Distinto"),
        ("2026-10-29", "Candela_1029_Vertical", "Config 27/10 - Prueba Vertical Vocales Distinto"),
    ]:
        dir_p = os.path.join(BASE_DATOS_DIR, folder_d)
        if os.path.exists(dir_p):
            r_list = sorted([
                os.path.join(dir_p, d) for d in os.listdir(dir_p)
                if os.path.isdir(os.path.join(dir_p, d)) and d.split("_")[0].upper() in VOCALES
            ])
            if len(r_list) >= 5:
                b = extraer_bloque_crudo_octubre(r_list, sid, "Candela", "2026-10-27", mont)
                b["k_cal"] = 6
                bloques_train.append(b)

    # 4) 2026-10-30: Solo las 5 tomas de vocales aisladas (A/E/I/O/U_Pruebainicial_Candela)
    dir_1030 = os.path.join(BASE_DATOS_DIR, "2026-10-30")
    if os.path.exists(dir_1030):
        r_1030 = sorted([
            os.path.join(dir_1030, d) for d in os.listdir(dir_1030)
            if os.path.isdir(os.path.join(dir_1030, d))
            and d.split("_")[0].upper() in VOCALES
            and "SecuenciaContinua" not in d
        ])
        if len(r_1030) >= 5:
            b = extraer_bloque_crudo_octubre(r_1030, "Candela_1030_Inicial", "Candela", "2026-10-30", "Prueba Inicial Vocales Aisladas - 2026-10-30")
            b["k_cal"] = 4
            bloques_train.append(b)

    # C. TEST SET AISLADO: Toda la carpeta 2026-10-27 (Pruebadatoslucas + Pruebasiversealespejo + pruebasonido)
    bloques_test_1027 = []
    dir_1027 = os.path.join(BASE_DATOS_DIR, "2026-10-27")
    all_1027 = sorted([os.path.join(dir_1027, d) for d in os.listdir(dir_1027) if os.path.isdir(os.path.join(dir_1027, d))])
    r_lucas = [p for p in all_1027 if "datoslucas" in os.path.basename(p).lower()]
    r_espejo = [p for p in all_1027 if "espejo" in os.path.basename(p).lower()]
    r_sonido = [p for p in all_1027 if "sonido" in os.path.basename(p).lower()]

    for r_list, sid, mont in [
        (r_lucas, "Test_1027_DatosLucas", "Candela 27/10 - Prueba Datos Lucas - Anomalia"),
        (r_espejo, "Test_1027_SinEspejo", "Candela 27/10 - Sin Verse al Espejo - Anomalia"),
        (r_sonido, "Test_1027_PruebaSonido", "Candela 27/10 - Prueba Sonido - Anomalia"),
    ]:
        if len(r_list) >= 5:
            b = extraer_bloque_crudo_octubre(r_list, sid, "Candela", "2026-10-27", mont)
            b["k_cal"] = 8
            bloques_test_1027.append(b)

    return bloques_train, bloques_test_1027


def main():
    fijar_semillas(42)
    print("=" * 105)
    print("INTEGRACION DE MEDICIONES DE OCTUBRE Y EVALUACION CIEGA EN 2026-10-27 (PRUEBA LUCAS / ESPEJO / SONIDO)")
    print("=" * 105)

    bloques_train, bloques_test = construir_corpus_completo()

    # 1. Calibrar cada bloque de entrenamiento (12D)
    X_tr_raw_list, X_tr_puro_list, Y_tr_list, Tomas_tr_list, Fechas_tr_list, Ses_tr_list = [], [], [], [], [], []
    calibradores_por_sesion = {}

    print("\n[1] BLOQUES EN EL POOL DE ENTRENAMIENTO:")
    print("-" * 95)
    for b in bloques_train:
        X_rec, cal_idx, test_idx, cal_p = calibrate_session_causal(b["X"], b["Y"], k_per_vowel=b["k_cal"])
        b["X_puro"] = X_rec
        b["cal_idx"] = cal_idx
        b["test_idx"] = test_idx
        calibradores_por_sesion[b["sesion_id"]] = cal_p

        X_tr_raw_list.append(b["X"])
        X_tr_puro_list.append(X_rec)
        Y_tr_list.append(b["Y"])
        Tomas_tr_list.append(b["Tomas"])
        Fechas_tr_list.append(np.array([b["fecha"]] * len(b["Y"])))
        Ses_tr_list.append(np.array([b["sesion_id"]] * len(b["Y"])))
        print(f"  + {b['sesion_id']:<30} | N={len(b['Y']):4d} pulsos | Fecha: {b['fecha']} | {b['montaje']}")

    X_tr_raw = np.concatenate(X_tr_raw_list, axis=0)
    X_tr_puro = np.concatenate(X_tr_puro_list, axis=0)
    Y_tr = np.concatenate(Y_tr_list, axis=0)
    Tomas_tr = np.concatenate(Tomas_tr_list, axis=0)
    Fechas_tr = np.concatenate(Fechas_tr_list, axis=0)
    Ses_tr = np.concatenate(Ses_tr_list, axis=0)

    print("-" * 95)
    print(f"TOTAL ENTRENAMIENTO: N_train = {len(Y_tr)} pulsos en {len(bloques_train)} bloques.")

    # 2. Entrenar el ConvOrthogonalAutoencoder (100% Zero-Labels) SOLO con el Pool de Entrenamiento
    print("\n[2] ENTRENANDO ConvOrthogonalAutoencoder (250 epocas, lr=0.008, Zero-Labels) SIN 2026-10-27...")
    model_ae, Z_tr_4d = entrenar_autoencoder_zero_labels(
        X_tr_puro, latent_dim=4, epochs=250, lr=0.008, nombre_modelo="AE-Unificado-Octubre-12D"
    )
    model_ae.eval()

    # Proyeccion 2D canonica ajustada UNICAMENTE en el conjunto de entrenamiento
    lda_2d = LinearDiscriminantAnalysis(n_components=2)
    Z_tr_2d = lda_2d.fit_transform(Z_tr_4d, Y_tr)
    c_tr_mean = np.mean(Z_tr_2d, axis=0)
    Z_tr_2d = Z_tr_2d - c_tr_mean
    vA = np.mean(Z_tr_2d[Y_tr == "A"], axis=0)
    ang = np.pi / 2.0 - np.arctan2(vA[1], vA[0])
    R_rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    Z_tr_2d = Z_tr_2d @ R_rot.T
    sign_x = -1.0 if np.mean(Z_tr_2d[Y_tr == "I", 0]) < np.mean(Z_tr_2d[Y_tr == "U", 0]) else 1.0
    Z_tr_2d[:, 0] *= sign_x

    def project_4d_to_2d(Z4):
        Z2 = (lda_2d.transform(Z4) - c_tr_mean) @ R_rot.T
        Z2[:, 0] *= sign_x
        return Z2

    clf_lda_4d = LinearDiscriminantAnalysis().fit(Z_tr_4d, Y_tr)
    clf_lda_2d = LinearDiscriminantAnalysis().fit(Z_tr_2d, Y_tr)
    clf_nc_4d = NearestCentroid().fit(Z_tr_4d, Y_tr)
    clf_knn_4d = KNeighborsClassifier(n_neighbors=9, weights="distance").fit(Z_tr_4d, Y_tr)

    acc_tr_lda = accuracy_score(Y_tr, clf_lda_4d.predict(Z_tr_4d)) * 100.0
    print(f"  -> Exactitud sobre el Pool de Entrenamiento ({len(Y_tr)} pulsos): {acc_tr_lda:.2f}%")

    # 3. Evaluar en frio el Test Set Aislado 2026-10-27 (por sub-bloque y en conjunto)
    print("\n[3] EVALUACION CAUSAL SOBRE EL TEST SET AISLADO 2026-10-27 (30 TOMAS MEZCLADAS):")
    print("-" * 105)
    print(f"{'Sub-Sesion 2026-10-27 (Aislada de la Red)':<38} | {'N_cal':>5} | {'N_test':>6} | {'1. Crudo Zero-Shot':>18} | {'2. Ejes Puros 12D':>18} | Desglose /a/ /e/ /i/ /o/ /u/")
    print("-" * 105)

    X_te_raw_list, X_te_puro_list, Y_te_list, Tomas_te_list, Fechas_te_list, Ses_te_list = [], [], [], [], [], []
    test_results_plot = []
    tot_raw_ok, tot_puro_ok, tot_n_test = 0, 0, 0
    all_y_test_fut, all_pred_test_fut = [], []

    for b in bloques_test:
        X_s, y_s = b["X"], b["Y"]
        X_rec, cal_idx, test_idx, cal_p = calibrate_session_causal(X_s, y_s, k_per_vowel=b["k_cal"])
        b["X_puro"] = X_rec
        calibradores_por_sesion[b["sesion_id"]] = cal_p

        X_te_raw_list.append(X_s)
        X_te_puro_list.append(X_rec)
        Y_te_list.append(y_s)
        Tomas_te_list.append(b["Tomas"])
        Fechas_te_list.append(np.array([b["fecha"]] * len(y_s)))
        Ses_te_list.append(np.array([b["sesion_id"]] * len(y_s)))

        with torch.no_grad():
            Z_raw_te_4d = model_ae.encode(torch.tensor(X_s[test_idx], dtype=torch.float32)).cpu().numpy()
            Z_puro_cal_4d = model_ae.encode(torch.tensor(X_rec[cal_idx], dtype=torch.float32)).cpu().numpy()
            Z_puro_te_4d = model_ae.encode(torch.tensor(X_rec[test_idx], dtype=torch.float32)).cpu().numpy()

        y_te_fut = y_s[test_idx]
        y_cal_init = y_s[cal_idx]

        pred_raw = clf_lda_4d.predict(Z_raw_te_4d)
        pred_puro = clf_lda_4d.predict(Z_puro_te_4d)

        acc_raw = accuracy_score(y_te_fut, pred_raw) * 100.0
        acc_puro = accuracy_score(y_te_fut, pred_puro) * 100.0

        tot_raw_ok += int(np.sum(pred_raw == y_te_fut))
        tot_puro_ok += int(np.sum(pred_puro == y_te_fut))
        tot_n_test += len(y_te_fut)
        all_y_test_fut.append(y_te_fut)
        all_pred_test_fut.append(pred_puro)

        v_accs = []
        for v in VOCALES:
            mv = (y_te_fut == v)
            va = np.mean(pred_puro[mv] == v) * 100.0 if np.any(mv) else 0.0
            v_accs.append(f"{va:3.0f}%")
        desglose_str = " ".join(v_accs)

        print(f"{b['montaje'][:38]:<38} | {len(cal_idx):>5d} | {len(test_idx):>6d} | {acc_raw:>17.2f}% | {acc_puro:>17.2f}% | {desglose_str}")

        test_results_plot.append({
            "sid": b["sesion_id"],
            "title": b["montaje"],
            "Z_cal_2d": project_4d_to_2d(Z_puro_cal_4d),
            "y_cal": y_cal_init,
            "Z_te_2d": project_4d_to_2d(Z_puro_te_4d),
            "Z_raw_2d": project_4d_to_2d(Z_raw_te_4d),
            "y_te": y_te_fut,
            "preds": pred_puro,
            "acc_raw": acc_raw,
            "acc_puro": acc_puro
        })

    print("-" * 105)
    g_raw = tot_raw_ok / max(tot_n_test, 1) * 100.0
    g_puro = tot_puro_ok / max(tot_n_test, 1) * 100.0
    all_y_test_fut = np.concatenate(all_y_test_fut)
    all_pred_test_fut = np.concatenate(all_pred_test_fut)
    v_glob = " ".join([f"{np.mean(all_pred_test_fut[all_y_test_fut == v] == v)*100:3.0f}%" for v in VOCALES])
    print(f"{'GLOBAL TEST 2026-10-27 (30 TOMAS)':<38} | {sum(len(r['y_cal']) for r in test_results_plot):>5d} | {tot_n_test:>6d} | {g_raw:>17.2f}% | {g_puro:>17.2f}% | {v_glob}")
    print("=" * 105)

    # 4. Guardar el Dataset Unificado Completo (Train + Test identificado) y el Modelo .pt
    X_all_raw = np.concatenate([X_tr_raw] + X_te_raw_list, axis=0)
    X_all_puro = np.concatenate([X_tr_puro] + X_te_puro_list, axis=0)
    Y_all = np.concatenate([Y_tr] + Y_te_list, axis=0)
    Tomas_all = np.concatenate([Tomas_tr] + Tomas_te_list, axis=0)
    Fechas_all = np.concatenate([Fechas_tr] + Fechas_te_list, axis=0)
    Ses_all = np.concatenate([Ses_tr] + Ses_te_list, axis=0)
    Split_all = np.array(["TRAIN"] * len(Y_tr) + ["TEST_1027"] * (len(Y_all) - len(Y_tr)))

    npz_unificado_path = os.path.join(CACHE_OUT_DIR, "dataset_unificado_con_octubre_12d.npz")
    np.savez_compressed(
        npz_unificado_path,
        X_cruda=X_all_raw,
        X_env_raw=X_all_raw,
        X_env=X_all_puro,
        X_spec=X_all_puro,
        Y=Y_all,
        Tomas=Tomas_all,
        Fechas=Fechas_all,
        Sesiones=Ses_all,
        Split=Split_all,
        Musculos_Canales=np.array(["Eje 0 Mandibula Pura", "Eje 1 Sonrisa Pura", "Eje 2 Trompita Pura"]),
        bpm=40,
        fs=2000,
        tipo_envolvente="rms_12d_ejes_puros",
        smooth_ms=100,
        canales_features=np.array(["canal_0", "canal_1", "canal_2"])
    )

    # Exportar tambien en CSV para que cualquier herramienta del repo lo pueda abrir directo
    cols_feat = [f"Ch{c}_T{t}" for c in range(3) for t in range(20)]
    df_export = pd.DataFrame(X_all_puro.reshape(len(Y_all), 60), columns=cols_feat)
    df_export.insert(0, "Split", Split_all)
    df_export.insert(0, "Sesion", Ses_all)
    df_export.insert(0, "Fecha", Fechas_all)
    df_export.insert(0, "Toma", Tomas_all)
    df_export.insert(0, "Vocal", Y_all)
    csv_unificado_path = os.path.join(CACHE_OUT_DIR, "caracteristicas_exportadas_unificado_octubre_12d.csv")
    df_export.to_csv(csv_unificado_path, index=False)

    # Guardar checkpoint completo del modelo con parametros de calibracion 12D y clasificadores
    ckpt_path = os.path.join(MODELOS_OUT_DIR, "autoencoder_conv1d_ortogonal_unificado_octubre_12d.pt")
    torch.save({
        "model_state_dict": model_ae.state_dict(),
        "architecture": "ConvOrthogonalAutoencoder_12D_PureAxes",
        "latent_dim": 4,
        "seq_len": 20,
        "in_channels": 3,
        "T_CANON": T_CANON,
        "mu_T": mu_T,
        "S_t": S_t,
        "B_4": B_4,
        "Z_canon": Z_canon,
        "lda_2d": lda_2d,
        "c_tr_mean": c_tr_mean,
        "R_rot": R_rot,
        "sign_x": sign_x,
        "clf_lda_4d": clf_lda_4d,
        "clf_lda_2d": clf_lda_2d,
        "clf_nc_4d": clf_nc_4d,
        "clf_knn_4d": clf_knn_4d,
        "Z_tr_2d_centroids": {v: np.mean(Z_tr_2d[Y_tr == v], axis=0) for v in VOCALES},
        "Z_tr_4d_centroids": {v: np.mean(Z_tr_4d[Y_tr == v], axis=0) for v in VOCALES},
        "calibradores_por_sesion": calibradores_por_sesion,
        "n_train": len(Y_tr),
        "n_test_1027": len(Y_all) - len(Y_tr),
        "acc_train_lda": acc_tr_lda,
        "acc_test_1027_raw": g_raw,
        "acc_test_1027_12d": g_puro
    }, ckpt_path)

    print(f"\n[OK] Dataset NPZ guardado en : {npz_unificado_path} (N_total = {len(Y_all)} pulsos)")
    print(f"[OK] Dataset CSV guardado en : {csv_unificado_path}")
    print(f"[OK] Modelo .pt guardado en  : {ckpt_path}")

    # 5. Generar Figura de 4 Paneles: Pool Train + Los 3 Bloques de Test 2026-10-27
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), dpi=200)
    axes = axes.flatten()
    cmap_bg = ListedColormap([COLORES_VOCALES[v] for v in VOCALES])

    all_pts_ref = np.vstack([Z_tr_2d] + [r["Z_te_2d"] for r in test_results_plot])
    x_min, x_max = np.percentile(all_pts_ref[:, 0], 1) - 0.5, np.percentile(all_pts_ref[:, 0], 99) + 0.5
    y_min, y_max = np.percentile(all_pts_ref[:, 1], 1) - 0.5, np.percentile(all_pts_ref[:, 1], 99) + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid_preds = clf_lda_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    grid_idx = np.array([VOCALES.index(v) for v in grid_preds]).reshape(xx.shape)

    # Panel 0: Pool de Entrenamiento Unificado (Historico + Octubre sin 10-27)
    ax0 = axes[0]
    ax0.contourf(xx, yy, grid_idx, levels=np.arange(-0.5, 5.5, 1), cmap=cmap_bg, alpha=0.15)
    ax0.contour(xx, yy, grid_idx, levels=np.arange(-0.5, 5.5, 1), colors="gray", linewidths=0.6, alpha=0.4)
    for v in VOCALES:
        m = (Y_tr == v)
        ax0.scatter(Z_tr_2d[m, 0], Z_tr_2d[m, 1], c=COLORES_VOCALES[v], s=18, alpha=0.35, edgecolors="none", label=f"/{v.lower()}/")
        c_v = np.mean(Z_tr_2d[m], axis=0)
        ax0.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], marker="D", s=110, edgecolors="black", linewidths=1.5, zorder=5)
    ax0.set_xlim(x_min, x_max); ax0.set_ylim(y_min, y_max)
    ax0.set_title(f"A. Pool de Entrenamiento Unificado con Octubre - N={len(Y_tr)} Pulsos\n19 Bloques en Train | Exactitud LDA: {acc_tr_lda:.1f}%", fontsize=11.5, fontweight="bold")
    ax0.set_xlabel("Latente Z1 - Sonrisa +X vs Trompita -X")
    ax0.set_ylabel("Latente Z2 - Apertura Mandibular +Y")
    ax0.legend(loc="upper right")
    ax0.grid(True, alpha=0.22)

    # Paneles 1, 2, 3: Los 3 bloques del Test Aislado 2026-10-27
    for idx_r, r in enumerate(test_results_plot[:3]):
        ax = axes[idx_r + 1]
        ax.contourf(xx, yy, grid_idx, levels=np.arange(-0.5, 5.5, 1), cmap=cmap_bg, alpha=0.16)
        ax.contour(xx, yy, grid_idx, levels=np.arange(-0.5, 5.5, 1), colors="gray", linewidths=0.6, alpha=0.4)
        for v in VOCALES:
            c_v = np.mean(Z_tr_2d[Y_tr == v], axis=0)
            ax.scatter(c_v[0], c_v[1], c=COLORES_VOCALES[v], marker="D", s=100, edgecolors="black", linewidths=1.4, zorder=5)
            mc = (r["y_cal"] == v)
            if np.any(mc):
                ax.scatter(r["Z_cal_2d"][mc, 0], r["Z_cal_2d"][mc, 1], facecolors="none", edgecolors=COLORES_VOCALES[v], marker="^", s=42, linewidths=1.2, alpha=0.75, zorder=3)
            mt = (r["y_te"] == v)
            if np.any(mt):
                cmask = mt & (r["preds"] == r["y_te"])
                wmask = mt & (r["preds"] != r["y_te"])
                if np.any(cmask):
                    ax.scatter(r["Z_te_2d"][cmask, 0], r["Z_te_2d"][cmask, 1], c=COLORES_VOCALES[v], marker="o", s=52, edgecolors="white", linewidths=0.7, alpha=0.92, zorder=4)
                if np.any(wmask):
                    ax.scatter(r["Z_te_2d"][wmask, 0], r["Z_te_2d"][wmask, 1], c=COLORES_VOCALES[v], marker="o", s=52, edgecolors="black", linewidths=1.3, alpha=0.82, zorder=4)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        clean_title = r["title"].replace("(", "- ").replace(")", "")
        ax.set_title(
            f"{chr(66 + idx_r)}. {clean_title}\nCrudo Zero-Shot: {r['acc_raw']:.1f}% -> Calibracion 12D: {r['acc_puro']:.1f}% | N_test={len(r['y_te'])}",
            fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Latente Z1 - Red Congelada")
        ax.set_ylabel("Latente Z2 - Red Congelada")
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        f"Integracion de Sesiones de Octubre y Test Aislado en 2026-10-27 - Crudo: {g_raw:.1f}% a Calibracion 12D: {g_puro:.1f}%",
        fontsize=13.5, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.94])

    for fig_p in [
        os.path.join(FIGURAS_OUT_DIR, "evaluacion_integracion_octubre_test_1027_12d.png"),
        os.path.join(LIBRO_IMG_DIR, "evaluacion_integracion_octubre_test_1027_12d.png"),
        "/home/santiago/.gemini/antigravity/brain/52fca87e-95a9-4554-a8c6-efe5340287d7/evaluacion_integracion_octubre_test_1027_12d.png"
    ]:
        plt.savefig(fig_p, dpi=200)
    plt.close(fig)
    print("[OK] Figura guardada en evaluacion_integracion_octubre_test_1027_12d.png")


if __name__ == "__main__":
    main()
