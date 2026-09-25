#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluación del Autoencoder Convolucional Ortogonal en Candela:
Sesiones 2026-09-01 (Risorio) y 2026-09-15/16 (Cigomático Mayor).

Utiliza la cadena consolidada estándar de generador_pca_umap:
- Sustracción dinámica del piso de ruido basal interpulso (IQR).
- Calibración de impedancia intersesión P95 por canal (aplicar_correccion_intersesion=True).
- Normalización por Supremo Tricanal del pulso.
- Arquitectura convolucional ortogonal compacta (K=5, Canales 6 y 12, Tanh,
  lambda_W=2.0, lambda_Z=0.5, 20 puntos por canal).
"""

import os
import sys
import time
import glob
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
emg_desarrollo = os.path.join(project_root, "EMG_desarrollo")
if emg_desarrollo not in sys.path:
    sys.path.insert(0, emg_desarrollo)

from deep_learning.pca_umap_clustering import generador_pca_umap as gpu
from deep_learning.motor_autoencoder_unificado import acondicionar_reposo_impedancia, extraer_sesion_agnostica

VOCALES = ['A', 'E', 'I', 'O', 'U']
VOCAL_TO_IDX = {v: i for i, v in enumerate(VOCALES)}
COLORES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}

class ConvOrthogonalAE(nn.Module):
    def __init__(self, in_channels=3, channels=(6, 12), kernel_size=5, latent_dim=2, time_len=20):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.time_len = time_len
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, channels[0], kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=kernel_size, padding=padding, bias=False)
        self.flatten_dim = channels[1] * time_len
        self.fc_enc1 = nn.Linear(self.flatten_dim, 64, bias=False)
        self.fc_enc2 = nn.Linear(64, latent_dim, bias=False)
        self.act = nn.Tanh()

        self.fc_dec1 = nn.Linear(latent_dim, 64, bias=False)
        self.fc_dec2 = nn.Linear(64, self.flatten_dim, bias=False)
        self.deconv1 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=kernel_size, padding=padding, bias=False)
        self.deconv2 = nn.ConvTranspose1d(channels[0], in_channels, kernel_size=kernel_size, padding=padding, bias=False)

        # Búferes precalculados para ortogonalidad
        self.register_buffer("eye_c1", torch.eye(channels[0]))
        self.register_buffer("eye_c2", torch.eye(channels[1]))
        self.register_buffer("eye_fc1", torch.eye(64))
        self.register_buffer("eye_fc2", torch.eye(latent_dim))
        self.register_buffer("eye_z", torch.eye(latent_dim))

    def encode(self, x):
        h = self.act(self.conv1(x))
        h = self.act(self.conv2(h))
        h = h.view(h.size(0), -1)
        h = self.act(self.fc_enc1(h))
        z = self.fc_enc2(h)
        return z

    def decode(self, z):
        h = self.act(self.fc_dec1(z))
        h = self.act(self.fc_dec2(h))
        h = h.view(h.size(0), self.channels[1], self.time_len)
        h = self.act(self.deconv1(h))
        x_rec = self.deconv2(h)
        return x_rec

    def forward(self, x):
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def calc_loss(self, x, x_rec, z, lambda_w=2.0, lambda_z=0.5):
        mse = nn.functional.mse_loss(x_rec, x)

        # Ortogonalidad de pesos
        w_c1 = self.conv1.weight.view(self.conv1.weight.size(0), -1)
        w_c2 = self.conv2.weight.view(self.conv2.weight.size(0), -1)
        l_w = (torch.norm(w_c1 @ w_c1.T - self.eye_c1, p='fro')**2 +
               torch.norm(w_c2 @ w_c2.T - self.eye_c2, p='fro')**2 +
               torch.norm(self.fc_enc1.weight @ self.fc_enc1.weight.T - self.eye_fc1, p='fro')**2 +
               torch.norm(self.fc_enc2.weight @ self.fc_enc2.weight.T - self.eye_fc2, p='fro')**2)

        # Decorrelación latente
        z_cent = z - z.mean(dim=0, keepdim=True)
        cov_z = (z_cent.T @ z_cent) / max(1, z.size(0) - 1)
        l_z = torch.norm(cov_z - self.eye_z, p='fro')**2

        total = mse + lambda_w * l_w + lambda_z * l_z
        return total, mse.item(), l_w.item(), l_z.item()

def extraer_con_cadena_oficial(base_dir, tomas_relativas, nombre_sesion):
    print(f"\n[Extracción Oficial GPU] Procesando {len(tomas_relativas)} tomas para {nombre_sesion}...")
    X_raw, Y_raw, tomas_wins, _ = gpu.extraer_features_concatenadas(
        base_dir=base_dir,
        mediciones=tomas_relativas,
        alpha_ruido=1.0,
        gate_ratio_ruido=0.0,
        smooth_ms=90,
        notch_q=2.0,
        target_len=20,
        modo_alineacion="Pico Volumen Micrófono",
        pre_pct=0.4,
        post_pct=0.6,
        canales_features=["canal_0", "canal_1", "canal_2"],
        aplicar_correccion_intersesion=True,
        tipo_envolvente="rms",
        lowpass_cutoff_hz=500.0,
        tipo_filtro_ruido="notch",
        highpass_cutoff_hz=20.0
    )
    
    X_arr = np.array(X_raw, dtype=np.float32).reshape(-1, 3, 20)
    Y_arr = np.array(Y_raw)
    sesiones_arr = np.array([extraer_sesion_agnostica(t) for t in tomas_wins])
    
    # Acondicionamiento de reposo e impedancia (sustracción de nivel basal por sesión completa)
    print(f"  [Acondicionamiento] Aplicando corrección de reposo basal e impedancia inter-sesión ({len(np.unique(sesiones_arr))} sesiones)...")
    X_arr = acondicionar_reposo_impedancia(X_arr, sesiones_arr, n_canales=3, n_pts_reposo=6)

    # Purga de valores atípicos con Isolation Forest (10%)
    print(f"  [Purga] Aplicando Isolation Forest (10%) sobre {len(X_arr)} ventanas...")
    X_flat = X_arr.reshape(len(X_arr), -1)
    iso = IsolationForest(contamination=0.10, random_state=42)
    mask_inliers = (iso.fit_predict(X_flat) == 1)
    X_clean = X_arr[mask_inliers]
    Y_clean = Y_arr[mask_inliers]
    print(f"  [OK] Ventanas válidas post-purga: {len(X_clean)} / {len(X_arr)}")

    # Conteo por vocal
    counts = {v: int(np.sum(Y_clean == v)) for v in VOCALES}
    print(f"  [Desglose] " + " | ".join([f"/{v.lower()}/: {counts[v]}" for v in VOCALES]))

    return X_clean, Y_clean

def entrenar_y_evaluar(X, Y, nombre_sesion, epochs=350, lr=0.003, device='cpu', seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ConvOrthogonalAE(in_channels=3, channels=(6, 12), kernel_size=5, latent_dim=2, time_len=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    x_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    print(f"\n[Entrenamiento] Iniciando {epochs} épocas para {nombre_sesion}...")
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        x_rec, z = model(x_tensor)
        loss, mse_val, lw_val, lz_val = model.calc_loss(x_tensor, x_rec, z, lambda_w=2.0, lambda_z=0.5)
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0 or (ep + 1) == epochs:
            elapsed = time.time() - t_start
            eta = (elapsed / (ep + 1)) * (epochs - (ep + 1))
            pct = ((ep + 1) / epochs) * 100.0
            print(f"  Época [{ep+1}/{epochs}] ({pct:.1f}%) - Loss: {loss.item():.4f} (MSE: {mse_val:.4f}, L_W: {lw_val:.4f}, L_Z: {lz_val:.4f}) - ETA: {eta:.1f}s")

    # Inferencia en el espacio latente
    model.eval()
    with torch.no_grad():
        _, z_eval = model(x_tensor)
        Z = z_eval.cpu().numpy()

    # Evaluación GMM no supervisada
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42, n_init=10)
    pred_clusters = gmm.fit_predict(Z)

    # Asignación lineal húngara de etiquetas diagnósticas
    contingency = np.zeros((5, 5))
    for i_v, v in enumerate(VOCALES):
        for j_c in range(5):
            contingency[j_c, i_v] = np.sum((Y == v) & (pred_clusters == j_c))

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    c2v = {row_ind[k]: col_ind[k] for k in range(len(row_ind))}
    pred_vocal_idx = np.array([c2v[c] for c in pred_clusters])
    y_true_idx = np.array([VOCAL_TO_IDX[v] for v in Y])

    acc_global = accuracy_score(y_true_idx, pred_vocal_idx) * 100.0
    cm = confusion_matrix(y_true_idx, pred_vocal_idx, labels=range(5))

    acc_por_vocal = {}
    for i_v, v in enumerate(VOCALES):
        tot = np.sum(y_true_idx == i_v)
        ok = cm[i_v, i_v]
        acc_por_vocal[v] = (ok / tot * 100.0) if tot > 0 else 0.0

    sil = silhouette_score(Z, pred_vocal_idx)
    db = davies_bouldin_score(Z, pred_vocal_idx)

    # Métricas de separación de pares críticos
    # Par 1: /o/ vs /u/
    mask_ou = np.isin(y_true_idx, [VOCAL_TO_IDX['O'], VOCAL_TO_IDX['U']])
    cm_ou = confusion_matrix(y_true_idx[mask_ou], pred_vocal_idx[mask_ou], labels=[VOCAL_TO_IDX['O'], VOCAL_TO_IDX['U']])
    acc_ou = np.trace(cm_ou) / max(1, np.sum(cm_ou)) * 100.0

    # Par 2: /e/ vs /i/
    mask_ei = np.isin(y_true_idx, [VOCAL_TO_IDX['E'], VOCAL_TO_IDX['I']])
    cm_ei = confusion_matrix(y_true_idx[mask_ei], pred_vocal_idx[mask_ei], labels=[VOCAL_TO_IDX['E'], VOCAL_TO_IDX['I']])
    acc_ei = np.trace(cm_ei) / max(1, np.sum(cm_ei)) * 100.0

    print(f"\n=== Resultados {nombre_sesion} ===")
    print(f"  Exactitud GMM Global: {acc_global:.2f}%")
    print(f"  Silueta: {sil:+.3f} | Davies-Bouldin: {db:.2f}")
    print(f"  Desglose: " + " | ".join([f"/{v.lower()}/: {acc_por_vocal[v]:.1f}%" for v in VOCALES]))
    print(f"  Separación Par /o/ vs /u/: {acc_ou:.1f}%")
    print(f"  Separación Par /e/ vs /i/: {acc_ei:.1f}%")

    res = {
        'Z': Z,
        'Y': Y,
        'y_true_idx': y_true_idx,
        'pred_vocal_idx': pred_vocal_idx,
        'acc_global': acc_global,
        'acc_por_vocal': acc_por_vocal,
        'acc_ou': acc_ou,
        'acc_ei': acc_ei,
        'sil': sil,
        'db': db,
        'cm': cm,
        'model': model
    }
    return res

def main():
    base_dir = os.path.join(project_root, "EMG_desarrollo/base_de_datos_electrodos")
    out_dir = os.path.join(project_root, "EMG_desarrollo/resultados/experimento_candela_conv_ortogonal")
    os.makedirs(out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo de cálculo: {device}")

    # 1. Obtener tomas de calibración de Candela 2026-09-01 (excluyendo continuas e incompletas)
    tomas_0901 = sorted([
        os.path.relpath(p, base_dir) 
        for p in glob.glob(os.path.join(base_dir, "2026-09-01", "*_Candela")) 
        if os.path.isdir(p) and not os.path.basename(p).startswith("SecuenciaContinua") and "Prueba5" not in os.path.basename(p)
    ])

    # 2. Obtener tomas de Candela 2026-09-15 (carpeta 2026-09-16)
    tomas_0915 = sorted([
        os.path.relpath(p, base_dir) 
        for p in glob.glob(os.path.join(base_dir, "2026-09-16", "*_Candela")) 
        if os.path.isdir(p)
    ])

    # 3. Extracción oficial con sustracción de ruido e impedancia
    X_0901, Y_0901 = extraer_con_cadena_oficial(base_dir, tomas_0901, "Candela 2026-09-01 - Risorio")
    res_0901 = entrenar_y_evaluar(X_0901, Y_0901, "Candela 2026-09-01 - Risorio", epochs=350, lr=0.003, device=device)

    X_0915, Y_0915 = extraer_con_cadena_oficial(base_dir, tomas_0915, "Candela 15/09 - Cigomático")
    res_0915 = entrenar_y_evaluar(X_0915, Y_0915, "Candela 15/09 - Cigomático", epochs=350, lr=0.003, device=device)

    # 4. Generar Panel Comparativo de 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    # Panel 0,0: Espacio Latente Candela 01/09
    ax0 = axes[0, 0]
    Z_01 = res_0901['Z']
    Y_01 = res_0901['Y']
    for v in VOCALES:
        mask = (Y_01 == v)
        ax0.scatter(Z_01[mask, 0], Z_01[mask, 1], c=COLORES[v], label=f"Vocal /{v.lower()}/", s=35, alpha=0.8)
    ax0.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax0.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax0.set_title(f"Candela 01/09 - Risorio: Exactitud {res_0901['acc_global']:.1f}%\n/o/ vs /u/: {res_0901['acc_ou']:.1f}% | /e/ vs /i/: {res_0901['acc_ei']:.1f}%", fontsize=10, fontweight='bold')
    ax0.set_xlabel("Coordenada Z1")
    ax0.set_ylabel("Coordenada Z2")
    ax0.grid(True, linestyle=':', alpha=0.5)
    ax0.legend(loc='upper right', fontsize=8)

    # Panel 0,1: Matriz de Confusión Candela 01/09
    ax1 = axes[0, 1]
    cm_01 = res_0901['cm']
    cm_norm_01 = cm_01.astype(float) / (cm_01.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0
    im0 = ax1.imshow(cm_norm_01, cmap=plt.cm.Blues, vmin=0, vmax=100)
    fig.colorbar(im0, ax=ax1, fraction=0.046, pad=0.04)
    ax1.set(xticks=range(5), yticks=range(5),
            xticklabels=[f"/{v.lower()}/" for v in VOCALES],
            yticklabels=[f"/{v.lower()}/" for v in VOCALES],
            xlabel="Predicción GMM", ylabel="Vocal Real",
            title=f"Matriz de Confusión: Candela 01/09\nSilueta: {res_0901['sil']:+.3f}")
    for r in range(5):
        for c in range(5):
            val = cm_01[r, c]
            pct = cm_norm_01[r, c]
            col = "white" if pct > 50 else "black"
            ax1.text(c, r, f"{val}\n{pct:.0f}%", ha='center', va='center', color=col, fontsize=8)

    # Panel 1,0: Espacio Latente Candela 15/09
    ax2 = axes[1, 0]
    Z_15 = res_0915['Z']
    Y_15 = res_0915['Y']
    for v in VOCALES:
        mask = (Y_15 == v)
        ax2.scatter(Z_15[mask, 0], Z_15[mask, 1], c=COLORES[v], label=f"Vocal /{v.lower()}/", s=35, alpha=0.8)
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_title(f"Candela 15/09 - Cigomático: Exactitud {res_0915['acc_global']:.1f}%\n/o/ vs /u/: {res_0915['acc_ou']:.1f}% | /e/ vs /i/: {res_0915['acc_ei']:.1f}%", fontsize=10, fontweight='bold')
    ax2.set_xlabel("Coordenada Z1")
    ax2.set_ylabel("Coordenada Z2")
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=8)

    # Panel 1,1: Matriz de Confusión Candela 15/09
    ax3 = axes[1, 1]
    cm_15 = res_0915['cm']
    cm_norm_15 = cm_15.astype(float) / (cm_15.sum(axis=1)[:, np.newaxis] + 1e-9) * 100.0
    im1 = ax3.imshow(cm_norm_15, cmap=plt.cm.Blues, vmin=0, vmax=100)
    fig.colorbar(im1, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set(xticks=range(5), yticks=range(5),
            xticklabels=[f"/{v.lower()}/" for v in VOCALES],
            yticklabels=[f"/{v.lower()}/" for v in VOCALES],
            xlabel="Predicción GMM", ylabel="Vocal Real",
            title=f"Matriz de Confusión: Candela 15/09\nSilueta: {res_0915['sil']:+.3f}")
    for r in range(5):
        for c in range(5):
            val = cm_15[r, c]
            pct = cm_norm_15[r, c]
            col = "white" if pct > 50 else "black"
            ax3.text(c, r, f"{val}\n{pct:.0f}%", ha='center', va='center', color=col, fontsize=8)

    out_png = os.path.join(out_dir, "evaluacion_candela_0901_y_0915.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"\n[OK] Gráfico comparativo generado exitosamente en: {out_png}")

    # Copiar también al directorio de artefactos para visualización inmediata
    art_dir = "/home/santiago/.gemini/antigravity/brain/73bff312-38b6-4d25-bb9c-a8e6cff36da5"
    if os.path.exists(art_dir):
        import shutil
        shutil.copy(out_png, os.path.join(art_dir, "evaluacion_candela_0901_y_0915.png"))

    # Guardar métricas en JSON
    json_path = os.path.join(out_dir, "metricas_candela.json")
    metricas = {
        'Candela_0901': {
            'acc_global': res_0901['acc_global'],
            'acc_por_vocal': res_0901['acc_por_vocal'],
            'acc_ou': res_0901['acc_ou'],
            'acc_ei': res_0901['acc_ei'],
            'silueta': float(res_0901['sil']),
            'davies_bouldin': float(res_0901['db'])
        },
        'Candela_0915': {
            'acc_global': res_0915['acc_global'],
            'acc_por_vocal': res_0915['acc_por_vocal'],
            'acc_ou': res_0915['acc_ou'],
            'acc_ei': res_0915['acc_ei'],
            'silueta': float(res_0915['sil']),
            'davies_bouldin': float(res_0915['db'])
        }
    }
    with open(json_path, "w") as f:
        json.dump(metricas, f, indent=2)
    print(f"[OK] Métricas consolidadas en: {json_path}")

if __name__ == "__main__":
    main()
