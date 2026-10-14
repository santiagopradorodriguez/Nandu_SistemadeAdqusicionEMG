# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Experimento: Tronco Canónico Congelado (Lucas) + Read-in Adaptativo (Santi y Candela)
# Metodología: Transferencia asimétrica rigurosa de variedad motora inter-sujeto
# ==============================================================================

import os
import sys
import json
import time
import numpy as np
import matplotlib
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix

project_root = "/home/santiago/repositorios/Nandu_SistemadeAdqusicionEMG"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Fijar semilla universal
torch.manual_seed(42)
np.random.seed(42)

VOCALES = ['A', 'E', 'I', 'O', 'U']
COLORES_VOCALES = {
    'A': '#E63946',
    'E': '#1F77B4',
    'I': '#2CA02C',
    'O': '#9D4EDD',
    'U': '#E7A61A'
}
MARKERS_SUJETO = {
    'Lucas': 'o',
    'Santi': 's',
    'Candela': '^'
}

# ==============================================================================
# 1. ARQUITECTURA: TRONCO ORTOGONAL CONVOLUCIONAL
# ==============================================================================

class ConvTrunkEncoder(nn.Module):
    def __init__(self, in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh()
        
        self.fc1 = nn.Linear(c2 * time_pts, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)
        nn.init.orthogonal_(self.fc2.weight)

    def forward(self, x):
        h1 = self.act(self.conv1(x))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        z = self.fc2(h3)
        return z

class ConvTrunkDecoder(nn.Module):
    def __init__(self, out_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2):
        super().__init__()
        self.out_channels = out_channels
        self.time_pts = time_pts
        c1, c2 = conv_channels
        pad = kernel_size // 2
        self.c2 = c2
        
        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * time_pts, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, out_channels, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh()

    def forward(self, z):
        dh1 = self.act(self.dfc1(z))
        dh2 = self.act(self.dfc2(dh1)).view(dh1.shape[0], self.c2, self.time_pts)
        dh3 = self.act(self.deconv1(dh2))
        x_rec = self.deconv2(dh3)
        return x_rec

class SubjectAdapter(nn.Module):
    """Adaptador de Read-in y Read-out 3x3 por sujeto"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.read_in = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        self.read_out = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
        with torch.no_grad():
            self.read_in.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_in.bias.zero_()
            self.read_out.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
            self.read_out.bias.zero_()

    def adapt_in(self, x):
        return self.read_in(x)

    def adapt_out(self, x_rec):
        return self.read_out(x_rec)

    def ortho_loss(self):
        W = self.read_in.weight.squeeze(-1)
        diff = W @ W.T - torch.eye(3, device=W.device)
        return torch.norm(diff, p='fro') ** 2

# ==============================================================================
# 2. EVALUACION GMM HUNGARA
# ==============================================================================

def evaluar_gmm_hungaro(Z_np, Y_np, seed=42):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=10)
    cluster_preds = gmm.fit_predict(Z_np)
    
    cost_matrix = np.zeros((5, 5))
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    
    y_pred_idx = np.array([cluster_to_vocal[c] for c in cluster_preds])
    acc_global = accuracy_score(y_idx, y_pred_idx)
    return acc_global, y_pred_idx, gmm

def evaluar_con_gmm_fijo(gmm, Z_np, Y_np):
    """Evalúa nuevos puntos sobre un GMM ya ajustado en Lucas"""
    cluster_preds = gmm.predict(Z_np)
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    
    cost_matrix = np.zeros((5, 5))
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    
    y_pred_idx = np.array([cluster_to_vocal.get(c, 0) for c in cluster_preds])
    acc = accuracy_score(y_idx, y_pred_idx)
    return acc, y_pred_idx

# ==============================================================================
# 3. EXPERIMENTO DE TRANSFERENCIA ASIMETRICA
# ==============================================================================

def ejecutar():
    npz_path = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    data = np.load(npz_path, allow_pickle=True)
    X_clean = data['X_clean'] # (1658, 3, 20)
    Y_clean = data['Y_clean'] # (1658,)
    sujetos = data['sujetos_clean'] # (1658,)

    mask_L = (sujetos == 'Lucas')
    mask_S = (sujetos == 'Santi')
    mask_C = (sujetos == 'Candela')

    X_L = torch.tensor(X_clean[mask_L], dtype=torch.float32)
    Y_L = Y_clean[mask_L]
    X_S = torch.tensor(X_clean[mask_S], dtype=torch.float32)
    Y_S = Y_clean[mask_S]
    X_C = torch.tensor(X_clean[mask_C], dtype=torch.float32)
    Y_C = Y_clean[mask_C]

    print("=" * 80)
    print("EXPERIMENTO: TRONCO CANONICO CONGELADO (LUCAS) + READ-IN ADAPTATIVO")
    print(f"Pulsos: Lucas = {len(Y_L)} | Santi = {len(Y_S)} | Candela = {len(Y_C)}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # FASE 1: Entrenar Tronco Canónico con Lucas
    # --------------------------------------------------------------------------
    print("\n[FASE 1] Entrenando Tronco Canónico con Lucas (250 épocas)...")
    torch.manual_seed(42)
    encoder_trunk = ConvTrunkEncoder(in_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2)
    decoder_trunk = ConvTrunkDecoder(out_channels=3, time_pts=20, conv_channels=(6, 12), kernel_size=5, latent_dim=2)
    
    opt_trunk = optim.Adam(list(encoder_trunk.parameters()) + list(decoder_trunk.parameters()), lr=0.008)
    mse = nn.MSELoss()

    for epoch in range(1, 251):
        opt_trunk.zero_grad()
        z_L = encoder_trunk(X_L)
        rec_L = decoder_trunk(z_L)
        loss_rec = mse(rec_L, X_L)
        
        # Regularización ortogonal
        loss_ortho = 0.0
        for m in [encoder_trunk.fc1, encoder_trunk.fc2, decoder_trunk.dfc1, decoder_trunk.dfc2]:
            W = m.weight
            d0, d1 = W.shape
            diff = (W @ W.T - torch.eye(d0)) if d0 < d1 else (W.T @ W - torch.eye(d1))
            loss_ortho += torch.norm(diff, p='fro') ** 2
            
        total_loss = loss_rec + 0.01 * loss_ortho
        total_loss.backward()
        opt_trunk.step()

        if epoch % 50 == 0 or epoch == 250:
            print(f"  Época {epoch:3d}/250 | Loss Rec: {loss_rec.item():.5f} | Ortho: {loss_ortho.item():.5f}")

    encoder_trunk.eval()
    decoder_trunk.eval()
    with torch.no_grad():
        Z_L_np = encoder_trunk(X_L).numpy()
    acc_L, _, gmm_L = evaluar_gmm_hungaro(Z_L_np, Y_L, seed=42)
    print(f"\n[OK] Tronco Canónico Entrenado en Lucas -> Exactitud GMM: {acc_L*100:.2f}%")

    # --------------------------------------------------------------------------
    # FASE 2: Congelar Tronco Canónico
    # --------------------------------------------------------------------------
    print("\n[FASE 2] Congelando Tronco Canónico (requires_grad = False)...")
    for p in encoder_trunk.parameters():
        p.requires_grad = False
    for p in decoder_trunk.parameters():
        p.requires_grad = False

    # --------------------------------------------------------------------------
    # FASE 3: Entrenar Adaptadores Read-in de Santi y Candela
    # --------------------------------------------------------------------------
    adaptador_Santi = SubjectAdapter(in_channels=3)
    adaptador_Candela = SubjectAdapter(in_channels=3)

    print("\n[FASE 3A] Ajustando Read-in de Santi sobre el Tronco Congelado (250 épocas)...")
    opt_S = optim.Adam(adaptador_Santi.parameters(), lr=0.008)
    for epoch in range(1, 251):
        opt_S.zero_grad()
        x_proj_S = adaptador_Santi.adapt_in(X_S)
        z_S = encoder_trunk(x_proj_S)
        x_rec_trunk_S = decoder_trunk(z_S)
        x_rec_S = adaptador_Santi.adapt_out(x_rec_trunk_S)
        
        loss_S = mse(x_rec_S, X_S) + 0.05 * adaptador_Santi.ortho_loss()
        loss_S.backward()
        opt_S.step()
        if epoch % 100 == 0 or epoch == 250:
            print(f"  Santi | Época {epoch:3d}/250 | Loss Rec: {loss_S.item():.5f}")

    print("\n[FASE 3B] Ajustando Read-in de Candela sobre el Tronco Congelado (250 épocas)...")
    opt_C = optim.Adam(adaptador_Candela.parameters(), lr=0.008)
    for epoch in range(1, 251):
        opt_C.zero_grad()
        x_proj_C = adaptador_Candela.adapt_in(X_C)
        z_C = encoder_trunk(x_proj_C)
        x_rec_trunk_C = decoder_trunk(z_C)
        x_rec_C = adaptador_Candela.adapt_out(x_rec_trunk_C)
        
        loss_C = mse(x_rec_C, X_C) + 0.05 * adaptador_Candela.ortho_loss()
        loss_C.backward()
        opt_C.step()
        if epoch % 100 == 0 or epoch == 250:
            print(f"  Candela | Época {epoch:3d}/250 | Loss Rec: {loss_C.item():.5f}")

    # --------------------------------------------------------------------------
    # FASE 4: Proyecciones Finales y Métricas
    # --------------------------------------------------------------------------
    adaptador_Santi.eval()
    adaptador_Candela.eval()
    with torch.no_grad():
        Z_S_np = encoder_trunk(adaptador_Santi.adapt_in(X_S)).numpy()
        Z_C_np = encoder_trunk(adaptador_Candela.adapt_in(X_C)).numpy()

    # Evaluar Santi y Candela directamente contra el GMM fijado de Lucas
    acc_S_on_L, _ = evaluar_con_gmm_fijo(gmm_L, Z_S_np, Y_S)
    acc_C_on_L, _ = evaluar_con_gmm_fijo(gmm_L, Z_C_np, Y_C)

    # Evaluar el espacio latente unificado total (1658 puntos)
    Z_total = np.vstack([Z_L_np, Z_S_np, Z_C_np])
    Y_total = np.concatenate([Y_L, Y_S, Y_C])
    acc_global_total, _, _ = evaluar_gmm_hungaro(Z_total, Y_total, seed=42)

    print("\n" + "=" * 80)
    print("RESULTADOS DE LA TRANSFERENCIA ASIMETRICA CON TRONCO CONGELADO")
    print("=" * 80)
    print(f"Lucas Canónico (Tronco Base)          : {acc_L*100:.2f}%")
    print(f"Santi Proyectado sobre GMM de Lucas   : {acc_S_on_L*100:.2f}%")
    print(f"Candela Proyectada sobre GMM de Lucas : {acc_C_on_L*100:.2f}%")
    print(f"Exactitud Global Unificada (1658 pts) : {acc_global_total*100:.2f}%")

    # Matrices Read-in aprendidas
    for nombre, adapt in [('Santi', adaptador_Santi), ('Candela', adaptador_Candela)]:
        W_rin = adapt.read_in.weight.squeeze(-1).detach().numpy()
        b_rin = adapt.read_in.bias.detach().numpy()
        print(f"\nMatriz Read-in {nombre}:")
        for row in W_rin:
            print("  [ " + "  ".join([f"{val:+.3f}" for val in row]) + " ]")
        print("  Sesgo b: [ " + "  ".join([f"{val:+.3f}" for val in b_rin]) + " ]")

    # --------------------------------------------------------------------------
    # FASE 5: Graficado de 4 Paneles
    # --------------------------------------------------------------------------
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    fig_path = os.path.join(dir_figuras, "transferencia_tronco_congelado_lucas_vs_santi_cande.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')

    # Panel 1: Lucas Canónico (Fijo)
    ax1 = axes[0, 0]
    ax1.set_facecolor('white')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax1.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_L == v)
        ax1.scatter(Z_L_np[m, 0], Z_L_np[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='none')
        c_v = np.mean(Z_L_np[m], axis=0)
        ax1.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax1.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax1.set_title(f"Lucas Canónico - Tronco Fijo (GMM Acc: {acc_L*100:.1f}%)", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Z1", fontsize=11)
    ax1.set_ylabel("Z2", fontsize=11)
    ax1.grid(True, linestyle=':', alpha=0.4)
    ax1.legend(loc='upper right')

    # Panel 2: Santi proyectado con Read-in
    ax2 = axes[0, 1]
    ax2.set_facecolor('white')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax2.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_S == v)
        ax2.scatter(Z_S_np[m, 0], Z_S_np[m, 1], c=[COLORES_VOCALES[v]], marker='s', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_S_np[m], axis=0)
        ax2.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='s', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax2.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax2.set_title(f"Santi Proyectado con Read-in (GMM Lucas: {acc_S_on_L*100:.1f}%)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Z1", fontsize=11)
    ax2.set_ylabel("Z2", fontsize=11)
    ax2.grid(True, linestyle=':', alpha=0.4)
    ax2.legend(loc='upper right')

    # Panel 3: Candela proyectada con Read-in
    ax3 = axes[1, 0]
    ax3.set_facecolor('white')
    ax3.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax3.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_C == v)
        ax3.scatter(Z_C_np[m, 0], Z_C_np[m, 1], c=[COLORES_VOCALES[v]], marker='^', label=f"/{v.lower()}/", s=40, alpha=0.7, edgecolors='black', linewidth=0.3)
        c_v = np.mean(Z_C_np[m], axis=0)
        ax3.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], marker='^', s=150, edgecolors='black', linewidth=1.5, zorder=5)
        ax3.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax3.set_title(f"Candela Proyectada con Read-in (GMM Lucas: {acc_C_on_L*100:.1f}%)", fontsize=13, fontweight='bold')
    ax3.set_xlabel("Z1", fontsize=11)
    ax3.set_ylabel("Z2", fontsize=11)
    ax3.grid(True, linestyle=':', alpha=0.4)
    ax3.legend(loc='upper right')

    # Panel 4: Superposición Tri-Sujeto en Variedad Unificada
    ax4 = axes[1, 1]
    ax4.set_facecolor('white')
    ax4.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax4.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m_L = (Y_L == v)
        m_S = (Y_S == v)
        m_C = (Y_C == v)
        ax4.scatter(Z_L_np[m_L, 0], Z_L_np[m_L, 1], c=[COLORES_VOCALES[v]], marker='o', s=25, alpha=0.5, edgecolors='none')
        ax4.scatter(Z_S_np[m_S, 0], Z_S_np[m_S, 1], c=[COLORES_VOCALES[v]], marker='s', s=25, alpha=0.5, edgecolors='none')
        ax4.scatter(Z_C_np[m_C, 0], Z_C_np[m_C, 1], c=[COLORES_VOCALES[v]], marker='^', s=25, alpha=0.5, edgecolors='none')
    
    # Dibujar centroides globales de las 5 vocales
    for v in VOCALES:
        m_all = (Y_total == v)
        c_tot = np.mean(Z_total[m_all], axis=0)
        ax4.scatter(c_tot[0], c_tot[1], c=[COLORES_VOCALES[v]], s=160, edgecolors='black', linewidth=1.8, zorder=6)
        ax4.annotate(f"/{v.lower()}/", (c_tot[0], c_tot[1]), fontsize=11, fontweight='bold',
                     path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    ax4.set_title(f"Variedad Unificada Tri-Sujeto (Global Acc: {acc_global_total*100:.1f}%)", fontsize=13, fontweight='bold')
    ax4.set_xlabel("Z1", fontsize=11)
    ax4.set_ylabel("Z2", fontsize=11)
    ax4.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n[Guardado] Gráfico de Transferencia: {fig_path}")

    # Guardar Checkpoint
    p_ckpt = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados/modelo_readin_tronco_congelado.pt")
    torch.save({
        'encoder_trunk': encoder_trunk.state_dict(),
        'decoder_trunk': decoder_trunk.state_dict(),
        'read_in_Santi': adaptador_Santi.read_in.state_dict(),
        'read_in_Candela': adaptador_Candela.read_in.state_dict(),
        'acc_global': acc_global_total
    }, p_ckpt)
    print(f"[Guardado] Checkpoint del Modelo: {p_ckpt}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar()
