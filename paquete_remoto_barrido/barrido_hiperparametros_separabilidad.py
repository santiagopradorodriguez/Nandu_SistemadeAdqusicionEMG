# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisicion EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institucion: Laboratorio de Sistemas Dinamicos (LSD) - FCEyN, UBA
# Descripcion: Barrido exhaustivo y sistematico de hiperparametros (Grid Search)
#              sobre Autoencoders Convolucionales 1D en senal cruda rectificada
#              a 2000 Hz, orientado a maximizar la separabilidad insupervisada,
#              la pureza de agrupamiento GMM y el coeficiente de silueta latente.
#              Incluye:
#              - Purga robusta de valores atipicos en el espacio latente (IQR 2.0).
#              - Deteccion automatica y penalizacion de colapso latente (Dying ReLU/1D).
#              - Monitoreo en tiempo real de avance, perdidas y tiempos estimados.
#              - Persistencia incremental continua en CSV con vaciado forzado a disco.
#              - Exportacion automatica del mejor modelo (.pth) y graficado canónico.
# ==============================================================================

import os
import sys
import time
import argparse
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.optimize import linear_sum_assignment

# Fijar semillas base para reproducibilidad
torch.manual_seed(42)
np.random.seed(42)

# Rutas del entorno (compatibles tanto con el repositorio como con carpeta standalone en PC remota)
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(script_dir, "cache_datos")):
    cache_dir = os.path.join(script_dir, "cache_datos")
    resultados_dir = os.path.join(script_dir, "resultados_barrido")
else:
    cache_dir = os.path.abspath(os.path.join(script_dir, "..", "cache_datos"))
    resultados_dir = os.path.abspath(os.path.join(script_dir, "..", "resultados_barrido"))

os.makedirs(resultados_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)

archivo_cache_default = os.path.join(cache_dir, "dataset_lucas_20260710_503pulsos.npz")
csv_resultados_path = os.path.join(resultados_dir, "resultados_barrido_separabilidad.csv")
mejor_modelo_path = os.path.join(resultados_dir, "mejor_autoencoder_separabilidad.pth")
figura_mejor_path = os.path.join(resultados_dir, "mejor_espacio_latente_barrido.png")

# Nombres anatomicos oficiales de canales (Lucas 2026-07-10)
CANALES_NOMBRES = ["Milohioideo", "Depresor", "Orbicular"]
COLORES_VOCALES = {
    'A': '#d62728',  # Rojo
    'E': '#1f77b4',  # Azul
    'I': '#2ca02c',  # Verde
    'O': '#9467bd',  # Violeta
    'U': '#d4ac0d'   # Amarillo
}

# ==============================================================================
# 1. ARQUITECTURA MODULAR DEL AUTOENCODER 1D
# ==============================================================================

class AutoencoderConvModular(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_filters_1=32,
        num_filters_2=16,
        kernel_1=31,
        kernel_2=15,
        pooling_type='dual',   # 'dual' (GAP+GMP), 'gap' (GAP puro), 'gmp' (GMP puro)
        activation_type='leaky_0.1', # 'leaky_0.1', 'leaky_0.2', 'elu'
        fc_hidden=32,
        latent_dim=2,
        target_len=1000
    ):
        super(AutoencoderConvModular, self).__init__()
        self.target_len = target_len
        self.pooling_type = pooling_type

        # Seleccion de activacion no saturada
        if activation_type == 'leaky_0.2':
            act_fn = lambda: nn.LeakyReLU(0.2)
        elif activation_type == 'elu':
            act_fn = lambda: nn.ELU(alpha=1.0)
        else:
            act_fn = lambda: nn.LeakyReLU(0.1)

        pad_1 = kernel_1 // 2
        pad_2 = kernel_2 // 2

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters_1, kernel_size=kernel_1, padding=pad_1),
            act_fn(),
            nn.Conv1d(num_filters_1, num_filters_2, kernel_size=kernel_2, padding=pad_2),
            act_fn()
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        pool_factor = 2 if pooling_type == 'dual' else 1
        dense_in = num_filters_2 * pool_factor

        self.fc_enc = nn.Sequential(
            nn.Linear(dense_in, fc_hidden),
            act_fn(),
            nn.Linear(fc_hidden, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, fc_hidden * 2),
            act_fn(),
            nn.Linear(fc_hidden * 2, in_channels * target_len)
        )

    def encode(self, x):
        h = self.conv(x)
        if self.pooling_type == 'dual':
            f_avg = self.gap(h).squeeze(-1)
            f_max = self.gmp(h).squeeze(-1)
            v = torch.cat([f_avg, f_max], dim=1)
        elif self.pooling_type == 'gmp':
            v = self.gmp(h).squeeze(-1)
        else: # 'gap'
            v = self.gap(h).squeeze(-1)
        z = self.fc_enc(v)
        return z

    def decode(self, z):
        x_rec = self.decoder(z).view(-1, 3, self.target_len)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

# ==============================================================================
# 2. EVALUACION DE SEPARABILIDAD Y PURGA DE OUTLIERS LATENTES
# ==============================================================================

def purgar_outliers_latentes(Z, Y, factor_iqr=2.0):
    """
    Descarta puntos atipicos aislados en el plano latente mediante distancia euclidea
    al centroide de su clase o mediana global, evitando que deformen las elipses GMM.
    """
    mascara_valida = np.ones(len(Y), dtype=bool)
    vocales_unicas = np.unique(Y)

    for v in vocales_unicas:
        idx_v = np.where(Y == v)[0]
        if len(idx_v) < 8:
            continue
        Z_v = Z[idx_v]
        centroide_v = np.median(Z_v, axis=0)
        distancias = np.linalg.norm(Z_v - centroide_v, axis=1)

        q1 = np.percentile(distancias, 25)
        q3 = np.percentile(distancias, 75)
        iqr = q3 - q1
        limite_superior = q3 + factor_iqr * iqr

        outliers_locales = idx_v[distancias > limite_superior]
        mascara_valida[outliers_locales] = False

    return mascara_valida

def evaluar_separabilidad_latente(Z, Y, purgar_outliers=True):
    """
    Calcula exactitud de agrupamiento GMM (Hungarian algorithm), Coeficiente de Silueta,
    Davies-Bouldin, y separabilidad individual de pares criticos (/o/ vs /u/, /e/ vs /i/).
    Detecta si ocurrio colapso dimensional.
    """
    # Deteccion temprana de colapso de varianza (Dying ReLU o proyeccion a un punto)
    var_z = np.var(Z, axis=0)
    if np.any(np.isnan(Z)) or np.any(var_z < 1e-7):
        return {
            'colapso': True,
            'gmm_acc': 20.0, # Nivel de azar puro
            'silhouette': -1.0,
            'calinski': 0.0,
            'davies_bouldin': 99.0,
            'acc_ou': 50.0,
            'acc_ei': 50.0,
            'acc_a': 20.0,
            'acc_e': 20.0,
            'acc_i': 20.0,
            'acc_o': 20.0,
            'acc_u': 20.0,
            'puntos_evaluados': len(Y)
        }

    # Descarte de valores atipicos latentes
    if purgar_outliers:
        mascara = purgar_outliers_latentes(Z, Y, factor_iqr=2.0)
        Z_eval = Z[mascara]
        Y_eval = Y[mascara]
    else:
        Z_eval = Z
        Y_eval = Y

    if len(Z_eval) < 50:
        Z_eval = Z
        Y_eval = Y

    vocales_unicas = sorted(list(set(Y_eval)))
    mapa_vocales = {v: i for i, v in enumerate(vocales_unicas)}
    y_true = np.array([mapa_vocales[v] for v in Y_eval])
    n_classes = len(vocales_unicas)

    try:
        gmm = GaussianMixture(n_components=n_classes, covariance_type='full', random_state=42, n_init=5)
        clusters = gmm.fit_predict(Z_eval)

        # Asignacion hungara optima
        matriz_costo = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(n_classes):
                matriz_costo[i, j] = -np.sum((clusters == i) & (y_true == j))

        row_ind, col_ind = linear_sum_assignment(matriz_costo)
        clusters_map = np.zeros_like(clusters)
        for c_cluster, c_vocal in zip(row_ind, col_ind):
            clusters_map[clusters == c_cluster] = c_vocal

        gmm_acc = accuracy_score(y_true, clusters_map) * 100.0
    except Exception:
        gmm_acc = 20.0

    try:
        sil = float(silhouette_score(Z_eval, Y_eval))
    except Exception:
        sil = -1.0

    try:
        cal = float(calinski_harabasz_score(Z_eval, Y_eval))
    except Exception:
        cal = 0.0

    try:
        db = float(davies_bouldin_score(Z_eval, Y_eval))
    except Exception:
        db = 99.0

    # Separabilidad especifica /o/ vs /u/
    mask_ou = np.isin(Y_eval, ['O', 'U'])
    if np.sum(mask_ou) > 10:
        z_ou = Z_eval[mask_ou]
        y_ou = (Y_eval[mask_ou] == 'U').astype(int)
        try:
            gmm_ou = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=3)
            preds_ou = gmm_ou.fit_predict(z_ou)
            acc_1 = accuracy_score(y_ou, preds_ou)
            acc_2 = accuracy_score(y_ou, 1 - preds_ou)
            acc_ou = max(acc_1, acc_2) * 100.0
        except Exception:
            acc_ou = 50.0
    else:
        acc_ou = 50.0

    # Separabilidad especifica /e/ vs /i/
    mask_ei = np.isin(Y_eval, ['E', 'I'])
    if np.sum(mask_ei) > 10:
        z_ei = Z_eval[mask_ei]
        y_ei = (Y_eval[mask_ei] == 'I').astype(int)
        try:
            gmm_ei = GaussianMixture(n_components=2, covariance_type='full', random_state=42, n_init=3)
            preds_ei = gmm_ei.fit_predict(z_ei)
            acc_1 = accuracy_score(y_ei, preds_ei)
            acc_2 = accuracy_score(y_ei, 1 - preds_ei)
            acc_ei = max(acc_1, acc_2) * 100.0
        except Exception:
            acc_ei = 50.0
    else:
        acc_ei = 50.0

    # Desglose de clasificacion por cada vocal individual
    acc_vocales = {}
    for idx_c, v_nom in enumerate(vocales_unicas):
        mask_v = (y_true == idx_c)
        if np.sum(mask_v) > 0:
            acc_v = np.mean(clusters_map[mask_v] == idx_c) * 100.0
            acc_vocales[f"acc_{v_nom.lower()}"] = round(float(acc_v), 1)
        else:
            acc_vocales[f"acc_{v_nom.lower()}"] = 0.0

    colapso = (sil < -0.1) or (gmm_acc < 25.0)

    resultado = {
        'colapso': colapso,
        'gmm_acc': round(gmm_acc, 2),
        'silhouette': round(sil, 4),
        'calinski': round(cal, 2),
        'davies_bouldin': round(db, 3),
        'acc_ou': round(acc_ou, 2),
        'acc_ei': round(acc_ei, 2),
        'puntos_evaluados': len(Y_eval)
    }
    resultado.update(acc_vocales)
    return resultado

# ==============================================================================
# 3. ENTRENAMIENTO DE UNA COMBINACION
# ==============================================================================

def entrenar_combinacion(
    tensor_x,
    labels_y,
    cfg,
    epochs=250,
    device='cpu',
    verbose_epochs=False,
    combo_idx=1,
    total_combos=1
):
    modelo = AutoencoderConvModular(
        in_channels=3,
        num_filters_1=cfg['num_filters_1'],
        num_filters_2=cfg['num_filters_2'],
        kernel_1=cfg['kernel_1'],
        kernel_2=cfg['kernel_2'],
        pooling_type=cfg['pooling_type'],
        activation_type=cfg['activation_type'],
        fc_hidden=cfg['fc_hidden'],
        latent_dim=2,
        target_len=tensor_x.shape[-1]
    ).to(device)

    criterio = nn.MSELoss()
    if cfg.get('optimizer', 'adam') == 'adamw':
        optimizador = optim.AdamW(modelo.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        optimizador = optim.Adam(modelo.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    t_in = tensor_x.to(device)
    batch_size = cfg.get('batch_size', None) # None = Full Batch
    n_muestras = len(tensor_x)

    t_start = time.time()
    modelo.train()

    for ep in range(1, epochs + 1):
        if batch_size is None or batch_size >= n_muestras:
            optimizador.zero_grad()
            rec, _ = modelo(t_in)
            loss = criterio(rec, t_in)
            loss.backward()
            optimizador.step()
            loss_val = loss.item()
        else:
            perm = torch.randperm(n_muestras)
            ep_loss = 0.0
            n_batches = 0
            for b_i in range(0, n_muestras, batch_size):
                b_idx = perm[b_i:b_i + batch_size]
                b_x = t_in[b_idx]
                optimizador.zero_grad()
                b_rec, _ = modelo(b_x)
                b_loss = criterio(b_rec, b_x)
                b_loss.backward()
                optimizador.step()
                ep_loss += b_loss.item()
                n_batches += 1
            loss_val = ep_loss / max(1, n_batches)

        # Monitoreo periodico en consola
        if verbose_epochs and (ep == 1 or ep % 50 == 0 or ep == epochs):
            pct = (ep / epochs) * 100.0
            t_elap = time.time() - t_start
            eta_ep = (t_elap / ep) * (epochs - ep)
            print(f"      Epoca [{ep:3d}/{epochs}] ({pct:4.1f}%) | MSE: {loss_val:.5f} | ETA epoca: {eta_ep:.1f}s")

    duracion = time.time() - t_start

    # Extraccion de representacion latente y reconstruccion final
    modelo.eval()
    with torch.no_grad():
        x_rec_t, z_t = modelo(t_in)
        z = z_t.cpu().numpy()
        x_rec = x_rec_t.cpu().numpy()

    metricas = evaluar_separabilidad_latente(z, labels_y, purgar_outliers=True)
    metricas['mse_reconstruccion'] = round(float(loss_val), 6)
    metricas['tiempo_entrenamiento_s'] = round(duracion, 2)

    return modelo, z, x_rec, metricas

# ==============================================================================
# 4. ALINEACION CANONICA Y GRAFICACION DE RESULTADOS
# ==============================================================================

def alinear_canonicamente(z_lat, labels):
    """
    Ancla /a/ estrictamente en el semieje vertical positivo (+Y, 90 deg)
    y fuerza que las vocales de sonrisa (/i/, /e/) queden a la derecha (+X > 0).
    """
    z_rot = z_lat.copy()
    idx_a = np.where(labels == 'A')[0]
    if len(idx_a) > 0:
        c_a = np.mean(z_rot[idx_a], axis=0)
        theta_a = np.arctan2(c_a[1], c_a[0])
        rot_ang = (np.pi / 2.0) - theta_a
        c, s = np.cos(rot_ang), np.sin(rot_ang)
        R = np.array([[c, -s], [s, c]])
        z_rot = np.dot(z_rot, R.T)

    idx_sonrisa = np.where(np.isin(labels, ['I', 'E']))[0]
    if len(idx_sonrisa) > 0:
        if np.median(z_rot[idx_sonrisa, 0]) < 0:
            z_rot[:, 0] *= -1.0

    return z_rot

def generar_grafico_mejor_modelo(z_lat, x_orig, x_rec, y_labels, info_mejor, save_path):
    """
    Genera figura de diagnostico de 2 filas:
    Fila 1: Espacio latente 2D alineado con sectores canónicos y elipses GMM.
    Fila 2: Cinco paneles de reconstruccion tricanal por vocal.
    """
    z_can = alinear_canonicamente(z_lat, y_labels)
    mascara_limpia = purgar_outliers_latentes(z_can, y_labels, factor_iqr=2.0)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.2, 1.0])

    # 1. Panel de Espacio Latente Canónico
    ax_lat = fig.add_subplot(gs[0, :3])
    ax_resumen = fig.add_subplot(gs[0, 3:])

    for v in ['A', 'E', 'I', 'O', 'U']:
        idx = np.where((y_labels == v) & mascara_limpia)[0]
        if len(idx) > 0:
            ax_lat.scatter(
                z_can[idx, 0], z_can[idx, 1],
                c=COLORES_VOCALES[v],
                label=f"Vocal /{v.lower()}/ (N={len(idx)})",
                alpha=0.75, edgecolors='black', linewidth=0.5, s=45
            )
            c = np.median(z_can[idx], axis=0)
            ax_lat.plot(c[0], c[1], marker='X', color='black', markersize=10, markeredgecolor='white', markeredgewidth=1.5)

    ax_lat.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_lat.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax_lat.grid(True, alpha=0.25)
    ax_lat.set_title(
        f"Mejor Espacio Latente 2D (Canónico) | GMM Acc: {info_mejor['gmm_acc']}%\n"
        f"Silueta: {info_mejor['silhouette']:+.4f} | Calinski-Harabasz: {info_mejor['calinski']} | Davies-Bouldin: {info_mejor['davies_bouldin']}",
        fontsize=12, fontweight='bold'
    )
    ax_lat.set_xlabel("Eje X Canónico (Retracción Comisural / Sonrisa ->)", fontsize=10)
    ax_lat.set_ylabel("Eje Y Canónico (Apertura Mandibular /a/ ^)", fontsize=10)
    ax_lat.legend(loc='best', fontsize=9, framealpha=0.85)

    # 2. Panel de Resumen de Metricas del Mejor Modelo
    ax_resumen.axis('off')
    texto_resumen = (
        f"MEJOR CONFIGURACION DE BARRIDO\n"
        f"--------------------------------------------------\n"
        f"• Exactitud GMM:         {info_mejor['gmm_acc']} %\n"
        f"• Coeficiente de Silueta: {info_mejor['silhouette']:+.4f}\n"
        f"• Desglose por Vocal:\n"
        f"    /a/: {info_mejor.get('acc_a', 0):.1f}% | /e/: {info_mejor.get('acc_e', 0):.1f}% | /i/: {info_mejor.get('acc_i', 0):.1f}%\n"
        f"    /o/: {info_mejor.get('acc_o', 0):.1f}% | /u/: {info_mejor.get('acc_u', 0):.1f}%\n"
        f"• Separabilidad /o/ vs /u/: {info_mejor['acc_ou']} %\n"
        f"• Separabilidad /e/ vs /i/: {info_mejor['acc_ei']} %\n"
        f"• Perdida Reconstruccion: {info_mejor['mse_reconstruccion']:.6f}\n"
        f"• Tiempo de corrida:     {info_mejor['tiempo_entrenamiento_s']} s\n"
        f"--------------------------------------------------\n"
        f"HIPERPARAMETROS ARQUITECTONICOS:\n"
        f"• Filtros (C1, C2):      ({info_mejor['num_filters_1']}, {info_mejor['num_filters_2']})\n"
        f"• Kernels (K1, K2):      ({info_mejor['kernel_1']}, {info_mejor['kernel_2']})\n"
        f"• Agrupamiento (Pool):   {info_mejor['pooling_type'].upper()}\n"
        f"• Activacion:            {info_mejor['activation_type']}\n"
        f"• Capa Oculta Densa:     {info_mejor['fc_hidden']} neuronas\n"
        f"• Tasa de Aprendizaje:   {info_mejor['lr']} ({info_mejor['optimizer']})\n"
        f"• Weight Decay:          {info_mejor['weight_decay']}\n"
        f"• Batch Size:            {info_mejor.get('batch_size', 'Full Batch')}\n"
        f"• Epocas:                {info_mejor['epochs']}\n"
    )
    ax_resumen.text(0.05, 0.95, texto_resumen, fontsize=10.0, family='monospace', verticalalignment='top')

    # 3. Paneles Inferiores: Reconstruccion Exclusiva por Vocal
    tiempo = np.linspace(0.0, 1.5, x_orig.shape[-1])
    colores_canales = ['#d62728', '#1f77b4', '#2ca02c']

    for i, v in enumerate(['A', 'E', 'I', 'O', 'U']):
        ax_v = fig.add_subplot(gs[1, i])
        idx_v = np.where(y_labels == v)[0]

        if len(idx_v) > 0:
            rec_v = x_rec[idx_v]
            mse_v = np.mean((x_orig[idx_v] - rec_v) ** 2)

            for c_idx in range(3):
                rec_mean = np.mean(rec_v[:, c_idx, :], axis=0)
                lbl = f"Rec {CANALES_NOMBRES[c_idx]}" if i == 0 else None
                ax_v.plot(tiempo, rec_mean, label=lbl, color=colores_canales[c_idx], lw=1.8)

            acc_v_val = info_mejor.get(f'acc_{v.lower()}', 0.0)
            ax_v.set_title(f"Vocal /{v.lower()}/ (N={len(idx_v)})\nAcc GMM: {acc_v_val:.1f}% | MSE: {mse_v:.4f}", fontsize=10.5, fontweight='bold')
        else:
            ax_v.set_title(f"Vocal /{v.lower()}/ (Sin datos)", fontsize=11)

        ax_v.set_xlabel("Tiempo en ciclo 40/60 (s)", fontsize=9)
        ax_v.set_ylim(-0.02, 1.05)
        ax_v.grid(True, alpha=0.3)
        if i == 0:
            ax_v.set_ylabel("Amplitud Normalizada (Supremo)", fontsize=9)
            ax_v.legend(loc='upper right', fontsize=8, framealpha=0.85)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

# ==============================================================================
# 5. GENERACION DEL ESPACIO DE BUSQUEDA
# ==============================================================================

def generar_espacio_hiperparametros(modo='exhaustivo'):
    """
    Genera la lista de diccionarios con las combinaciones a evaluar.
    - 'control': 1 corrida identica a la linea base historica (reproduce 54.27%).
    - 'rapido': 12 combinaciones representativas para validacion veloz.
    - 'estandar': 36 combinaciones estrategicas (variando kernels, filtros, lr).
    - 'exhaustivo': 72 combinaciones cubriendo todos los grados de libertad.
    """
    if modo == 'control':
        return [{
            'num_filters_1': 32,
            'num_filters_2': 16,
            'kernel_1': 31,
            'kernel_2': 15,
            'pooling_type': 'dual',
            'activation_type': 'leaky_0.1',
            'fc_hidden': 32,
            'lr': 0.008,
            'weight_decay': 1e-5,
            'optimizer': 'adam',
            'batch_size': None
        }]

    if modo == 'rapido':
        combos = []
        for k1, k2 in [(31, 15), (15, 7), (63, 31)]:
            for f1, f2 in [(32, 16), (16, 16)]:
                for pool in ['dual', 'gap']:
                    combos.append({
                        'num_filters_1': f1,
                        'num_filters_2': f2,
                        'kernel_1': k1,
                        'kernel_2': k2,
                        'pooling_type': pool,
                        'activation_type': 'leaky_0.1',
                        'fc_hidden': 32,
                        'lr': 0.008,
                        'weight_decay': 1e-5,
                        'optimizer': 'adam',
                        'batch_size': None
                    })
        return combos

    if modo == 'estandar':
        combos = []
        for k1, k2 in [(31, 15), (15, 7), (63, 31)]:
            for f1, f2 in [(32, 16), (16, 16), (64, 32)]:
                for pool in ['dual', 'gap']:
                    for lr in [0.004, 0.008]:
                        combos.append({
                            'num_filters_1': f1,
                            'num_filters_2': f2,
                            'kernel_1': k1,
                            'kernel_2': k2,
                            'pooling_type': pool,
                            'activation_type': 'leaky_0.1',
                            'fc_hidden': 32,
                            'lr': lr,
                            'weight_decay': 1e-5,
                            'optimizer': 'adam',
                            'batch_size': None
                        })
        return combos

    # Modo exhaustivo ("toda la tarde"): 72 combinaciones ricas
    combos = []
    for k1, k2 in [(31, 15), (15, 7), (63, 31)]:
        for f1, f2 in [(32, 16), (16, 16), (64, 32)]:
            for pool in ['dual', 'gap']:
                for act in ['leaky_0.1', 'leaky_0.2']:
                    for lr in [0.004, 0.008]:
                        combos.append({
                            'num_filters_1': f1,
                            'num_filters_2': f2,
                            'kernel_1': k1,
                            'kernel_2': k2,
                            'pooling_type': pool,
                            'activation_type': act,
                            'fc_hidden': 32,
                            'lr': lr,
                            'weight_decay': 1e-5,
                            'optimizer': 'adam',
                            'batch_size': None
                        })
    return combos

# ==============================================================================
# 6. BUCLE PRINCIPAL DE BARRIDO
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Barrido de Hiperparametros para Autoencoder Conv1D EMG")
    parser.add_argument('--modo', type=str, default='exhaustivo', choices=['control', 'rapido', 'estandar', 'exhaustivo'],
                        help="Modo del barrido: control (1 corrida), rapido (12), estandar (36), exhaustivo (72)")
    parser.add_argument('--epochs', type=int, default=250, help="Numero de epocas por combinacion (defecto: 250)")
    parser.add_argument('--cache', type=str, default=archivo_cache_default, help="Ruta al archivo .npz de datos")
    parser.add_argument('--verbose_epochs', action='store_true', help="Imprimir avance detallado de epocas")
    args = parser.parse_args()

    print("=" * 80)
    print("BARRIDO SISTEMATICO DE HIPERPARAMETROS - MAXIMIZACION DE SEPARABILIDAD")
    print(f"Modo: {args.modo.upper()} | Epocas por corrida: {args.epochs}")
    print(f"Archivo de cache: {args.cache}")
    print("=" * 80)

    # 1. Cargar archivo de datos en cache
    if not os.path.exists(args.cache):
        print(f"ERROR: No se encuentra el archivo de cache: {args.cache}")
        print("Por favor genere primero la cache con: python extraer_cache_pulsos_lucas_07.py")
        sys.exit(1)

    print("Cargando dataset desde cache comprimida...")
    datos = np.load(args.cache)
    X_rect = datos['X_rect'] # [N, 3, 1000]
    Y_labels = datos['Y']
    print(f"Dataset cargado exitosamente: {len(Y_labels)} pulsos limpios.")
    for v in ['A', 'E', 'I', 'O', 'U']:
        print(f"  Vocal /{v.lower()}/: {np.sum(Y_labels == v)} contracciones")

    tensor_X = torch.tensor(X_rect, dtype=torch.float32)

    # 2. Generar lista de combinaciones
    combinaciones = generar_espacio_hiperparametros(args.modo)
    total_combos = len(combinaciones)
    tiempo_est_min = (total_combos * 2.2 * (args.epochs / 250)) / 60.0

    print(f"\nTotal de combinaciones a evaluar: {total_combos}")
    print(f"Tiempo total estimado: ~{tiempo_est_min:.1f} minutos ({tiempo_est_min / 60.0:.2f} horas)")
    print(f"Resultados se registraran continuamente en: {csv_resultados_path}")
    print("-" * 80)

    # Inicializar o cargar CSV persistente
    columnas_csv = [
        'id', 'num_filters_1', 'num_filters_2', 'kernel_1', 'kernel_2',
        'pooling_type', 'activation_type', 'fc_hidden', 'lr', 'weight_decay',
        'optimizer', 'batch_size', 'epochs', 'colapso', 'gmm_acc', 'silhouette',
        'acc_a', 'acc_e', 'acc_i', 'acc_o', 'acc_u',
        'calinski', 'davies_bouldin', 'acc_ou', 'acc_ei', 'mse_reconstruccion',
        'tiempo_entrenamiento_s'
    ]

    # Si no existe, crear cabecera
    if not os.path.exists(csv_resultados_path):
        pd.DataFrame(columns=columnas_csv).to_csv(csv_resultados_path, index=False)

    mejor_acc = -1.0
    mejor_sil = -2.0
    mejor_modelo_state = None
    mejor_z = None
    mejor_rec = None
    mejor_info = None

    t_inicio_barrido = time.time()

    for idx, cfg in enumerate(combinaciones, start=1):
        cfg['epochs'] = args.epochs
        pct_global = (idx / total_combos) * 100.0
        t_transcurrido = time.time() - t_inicio_barrido
        t_prom_combo = t_transcurrido / idx if idx > 1 else 2.2
        eta_total_s = t_prom_combo * (total_combos - idx + 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_total_s))

        print(
            f"\n[{idx:3d}/{total_combos}] ({pct_global:5.1f}%) | "
            f"K:({cfg['kernel_1']},{cfg['kernel_2']}) F:({cfg['num_filters_1']},{cfg['num_filters_2']}) "
            f"P:{cfg['pooling_type']} Act:{cfg['activation_type']} lr:{cfg['lr']} | ETA: {eta_str}"
        )

        modelo, z, x_rec, met = entrenar_combinacion(
            tensor_X, Y_labels, cfg,
            epochs=args.epochs,
            verbose_epochs=args.verbose_epochs,
            combo_idx=idx,
            total_combos=total_combos
        )

        # Imprimir resumen de la corrida con desglose individual por vocal
        flag_colapso = "[COLAPSO]" if met['colapso'] else "[OK]"
        vocales_str = f"/a/:{met.get('acc_a', 0):.1f}% /e/:{met.get('acc_e', 0):.1f}% /i/:{met.get('acc_i', 0):.1f}% /o/:{met.get('acc_o', 0):.1f}% /u/:{met.get('acc_u', 0):.1f}%"
        print(
            f"      -> {flag_colapso} GMM: {met['gmm_acc']:5.2f}% | Silueta: {met['silhouette']:+6.4f}\n"
            f"         Vocales: [{vocales_str}] | O/U: {met['acc_ou']:5.1f}% | E/I: {met['acc_ei']:5.1f}% | "
            f"MSE: {met['mse_reconstruccion']:.6f} ({met['tiempo_entrenamiento_s']:.1f}s)"
        )

        # Registro persistente incremental a disco con flush
        fila = {
            'id': idx,
            'num_filters_1': cfg['num_filters_1'],
            'num_filters_2': cfg['num_filters_2'],
            'kernel_1': cfg['kernel_1'],
            'kernel_2': cfg['kernel_2'],
            'pooling_type': cfg['pooling_type'],
            'activation_type': cfg['activation_type'],
            'fc_hidden': cfg['fc_hidden'],
            'lr': cfg['lr'],
            'weight_decay': cfg['weight_decay'],
            'optimizer': cfg['optimizer'],
            'batch_size': cfg['batch_size'] if cfg['batch_size'] is not None else 'Full',
            'epochs': cfg['epochs'],
            'colapso': met['colapso'],
            'gmm_acc': met['gmm_acc'],
            'silhouette': met['silhouette'],
            'acc_a': met.get('acc_a', 0.0),
            'acc_e': met.get('acc_e', 0.0),
            'acc_i': met.get('acc_i', 0.0),
            'acc_o': met.get('acc_o', 0.0),
            'acc_u': met.get('acc_u', 0.0),
            'calinski': met['calinski'],
            'davies_bouldin': met['davies_bouldin'],
            'acc_ou': met['acc_ou'],
            'acc_ei': met['acc_ei'],
            'mse_reconstruccion': met['mse_reconstruccion'],
            'tiempo_entrenamiento_s': met['tiempo_entrenamiento_s']
        }
        pd.DataFrame([fila]).to_csv(csv_resultados_path, mode='a', header=False, index=False)

        # Criterio compuesto para mejor modelo: maximizar GMM Acc, desempatar con silueta
        criterio_score = met['gmm_acc'] + (10.0 * met['silhouette'] if met['silhouette'] > 0 else -100.0)
        criterio_mejor = mejor_acc + (10.0 * mejor_sil if mejor_sil > 0 else -100.0)

        if not met['colapso'] and criterio_score > criterio_mejor:
            mejor_acc = met['gmm_acc']
            mejor_sil = met['silhouette']
            mejor_modelo_state = modelo.state_dict()
            mejor_z = z
            mejor_rec = x_rec
            mejor_info = {**cfg, **met}

            # Guardar inmediatamente checkpoint de pesos PyTorch
            torch.save({
                'config': cfg,
                'metricas': met,
                'state_dict': mejor_modelo_state
            }, mejor_modelo_path)

            # Generar figura diagnostica actualizada
            generar_grafico_mejor_modelo(mejor_z, X_rect, mejor_rec, Y_labels, mejor_info, figura_mejor_path)
            print(f"      *** NUEVO RECORD REGISTRADO: GMM {mejor_acc:.2f}% | Silueta {mejor_sil:+.4f} (Guardado) ***")

    tiempo_total_m = (time.time() - t_inicio_barrido) / 60.0
    print("\n" + "=" * 80)
    print(f"BARRIDO COMPLETADO CON EXITO EN {tiempo_total_m:.2f} MINUTOS")
    print(f"Total combinaciones evaluadas: {total_combos}")
    if mejor_info is not None:
        print(f"MEJOR EXACTITUD GMM:     {mejor_info['gmm_acc']:.2f}%")
        print(f"MEJOR SILUETA LATENTE:   {mejor_info['silhouette']:+.4f}")
        print(f"SEPARABILIDAD /O/ vs /U/: {mejor_info['acc_ou']:.2f}%")
        print(f"SEPARABILIDAD /E/ vs /I/: {mejor_info['acc_ei']:.2f}%")
        print(f"DESGLOSE POR VOCAL DEL MEJOR MODELO:")
        print(f"  Vocal /a/: {mejor_info.get('acc_a', 0):.1f}%")
        print(f"  Vocal /e/: {mejor_info.get('acc_e', 0):.1f}%")
        print(f"  Vocal /i/: {mejor_info.get('acc_i', 0):.1f}%")
        print(f"  Vocal /o/: {mejor_info.get('acc_o', 0):.1f}%")
        print(f"  Vocal /u/: {mejor_info.get('acc_u', 0):.1f}%")
        print(f"CONFIGURACION GANADORA:")
        print(f"  Kernels: ({mejor_info['kernel_1']}, {mejor_info['kernel_2']})")
        print(f"  Filtros: ({mejor_info['num_filters_1']}, {mejor_info['num_filters_2']})")
        print(f"  Pooling: {mejor_info['pooling_type'].upper()} | Activacion: {mejor_info['activation_type']}")
        print(f"  LR: {mejor_info['lr']} | Optimizador: {mejor_info['optimizer']}")
        print(f"Archivo de pesos: {mejor_modelo_path}")
        print(f"Figura canónica: {figura_mejor_path}")
    else:
        print("AVISO: Todas las combinaciones evaluadas mostraron advertencia de colapso.")
    print("=" * 80)

if __name__ == "__main__":
    main()
