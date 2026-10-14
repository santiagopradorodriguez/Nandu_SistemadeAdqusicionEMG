# ==============================================================================
# Proyecto: NANDU LSD - Sistema de Adquisición EMG y Deep Learning
# Autores: Lucas Braunstein y Santiago Prado
# Institución: Laboratorio de Sistemas Dinámicos (LSD) - FCEyN, UBA
# Experimento: Capas de Entrada Específicas (Read-in Layers) + Tronco Latente Compartido
# Comparativa Rigurosa Ceteris Paribus: Baseline Pooled vs Multi-Subject Read-in
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
# 1. ARQUITECTURAS: TRONCO COMPARTIDO Y READ-IN ESPECIFICO
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
        
        # Inicialización ortogonal
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

class BaselinePooledAE(nn.Module):
    """Autoencoder tradicional sin adaptadores de sujeto"""
    def __init__(self, in_channels=3, time_pts=20, latent_dim=2):
        super().__init__()
        self.encoder = ConvTrunkEncoder(in_channels, time_pts, conv_channels=(6, 12), kernel_size=5, latent_dim=latent_dim)
        self.decoder = ConvTrunkDecoder(in_channels, time_pts, conv_channels=(6, 12), kernel_size=5, latent_dim=latent_dim)

    def forward(self, x, sujetos=None):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

    def encode(self, x, sujetos=None):
        return self.encoder(x)

    def weight_orthogonality_loss(self):
        loss = 0.0
        for m in [self.encoder.fc1, self.encoder.fc2, self.decoder.dfc1, self.decoder.dfc2]:
            W = m.weight
            d0, d1 = W.shape
            if d0 < d1:
                diff = W @ W.T - torch.eye(d0, device=W.device)
            else:
                diff = W.T @ W - torch.eye(d1, device=W.device)
            loss += torch.norm(diff, p='fro') ** 2
        return loss

class MultiSubjectReadInAE(nn.Module):
    """
    Autoencoder Multi-Sujeto:
    - Capas Read-in específicas por sujeto (Conv1D 1x1 intercanal: 3x3)
    - Tronco convolucional compartido
    - Capas Read-out específicas por sujeto
    """
    def __init__(self, lista_sujetos, in_channels=3, time_pts=20, latent_dim=2):
        super().__init__()
        self.lista_sujetos = lista_sujetos
        self.encoder = ConvTrunkEncoder(in_channels, time_pts, conv_channels=(6, 12), kernel_size=5, latent_dim=latent_dim)
        self.decoder = ConvTrunkDecoder(in_channels, time_pts, conv_channels=(6, 12), kernel_size=5, latent_dim=latent_dim)
        
        # Diccionarios de adaptadores por sujeto
        self.read_in = nn.ModuleDict()
        self.read_out = nn.ModuleDict()
        
        for s in lista_sujetos:
            # Conv1D 1x1 que actúa como transformación lineal 3x3 entre canales
            rin = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
            rout = nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=True)
            
            # Inicialización estrictamente en la identidad para estabilidad inicial
            with torch.no_grad():
                rin.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
                rin.bias.zero_()
                rout.weight.copy_(torch.eye(in_channels).unsqueeze(-1))
                rout.bias.zero_()
                
            self.read_in[s] = rin
            self.read_out[s] = rout

    def encode(self, x, sujetos):
        # x: (N, 3, 20), sujetos: array de strings
        z_out = torch.zeros(x.shape[0], self.encoder.fc2.out_features, device=x.device)
        for s in self.lista_sujetos:
            mask = (sujetos == s)
            if np.sum(mask) == 0:
                continue
            idx = np.where(mask)[0]
            x_s = x[idx]
            x_adapt = self.read_in[s](x_s)
            z_s = self.encoder(x_adapt)
            z_out[idx] = z_s
        return z_out

    def forward(self, x, sujetos):
        x_rec_out = torch.zeros_like(x)
        z_out = torch.zeros(x.shape[0], self.encoder.fc2.out_features, device=x.device)
        
        for s in self.lista_sujetos:
            mask = (sujetos == s)
            if np.sum(mask) == 0:
                continue
            idx = np.where(mask)[0]
            x_s = x[idx]
            
            # 1. Read-in específico
            x_adapt = self.read_in[s](x_s)
            # 2. Tronco compartido
            z_s = self.encoder(x_adapt)
            x_tronco = self.decoder(z_s)
            # 3. Read-out específico
            x_rec_s = self.read_out[s](x_tronco)
            
            x_rec_out[idx] = x_rec_s
            z_out[idx] = z_s
            
        return x_rec_out, z_out

    def weight_orthogonality_loss(self):
        loss = 0.0
        # Ortogonalidad del tronco
        for m in [self.encoder.fc1, self.encoder.fc2, self.decoder.dfc1, self.decoder.dfc2]:
            W = m.weight
            d0, d1 = W.shape
            if d0 < d1:
                diff = W @ W.T - torch.eye(d0, device=W.device)
            else:
                diff = W.T @ W - torch.eye(d1, device=W.device)
            loss += torch.norm(diff, p='fro') ** 2
            
        # Regularización ortogonal suave en el read-in para evitar colapso de canales
        for s in self.lista_sujetos:
            W_rin = self.read_in[s].weight.squeeze(-1) # (3, 3)
            diff_rin = W_rin @ W_rin.T - torch.eye(3, device=W_rin.device)
            loss += 0.1 * (torch.norm(diff_rin, p='fro') ** 2)
            
        return loss

# ==============================================================================
# 2. EVALUACION GMM NO SUPERVISADA CON ASIGNACION HUNGARA
# ==============================================================================

def evaluar_gmm_hungaro(Z_np, Y_np, seed=42):
    gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=seed, n_init=10)
    cluster_preds = gmm.fit_predict(Z_np)
    
    # Matriz de costos para asignación húngara
    cost_matrix = np.zeros((5, 5))
    y_idx = np.array([VOCALES.index(v) for v in Y_np])
    
    for c in range(5):
        for v in range(5):
            cost_matrix[c, v] = -np.sum((cluster_preds == c) & (y_idx == v))
            
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    cluster_to_vocal = {r: c for r, c in zip(row_ind, col_ind)}
    
    y_pred_idx = np.array([cluster_to_vocal[c] for c in cluster_preds])
    acc_global = accuracy_score(y_idx, y_pred_idx)
    
    # Desglose por vocal
    cm = confusion_matrix(y_idx, y_pred_idx, labels=list(range(5)))
    recalls = [cm[i, i] / max(1, np.sum(cm[i, :])) for i in range(5)]
    
    return acc_global, recalls, y_pred_idx, gmm

# ==============================================================================
# 3. BUCLE DE ENTRENAMIENTO CETERIS PARIBUS
# ==============================================================================

def entrenar_modelo(modelo, X_tensor, sujetos_np, epochs=250, lr=0.008):
    optimizer = optim.Adam(modelo.parameters(), lr=lr)
    mse_crit = nn.MSELoss()
    
    print(f"  Entrenando durante {epochs} épocas (lr={lr}, Adam)...")
    t0 = time.time()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        x_rec, z = modelo(X_tensor, sujetos_np)
        
        loss_rec = mse_crit(x_rec, X_tensor)
        loss_ortho = 0.01 * modelo.weight_orthogonality_loss()
        total_loss = loss_rec + loss_ortho
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"    Época {epoch:3d}/{epochs} | Loss Rec: {loss_rec.item():.5f} | Ortho: {loss_ortho.item():.5f} | Tiempo: {elapsed:.1f}s")
            
    return modelo

# ==============================================================================
# 4. EXPERIMENTO PRINCIPAL
# ==============================================================================

def ejecutar_experimento():
    npz_path = os.path.join(
        project_root,
        "EMG_desarrollo/resultados/resultados_autoencoder/cache_datasets/dataset_multisujeto_canonica.npz"
    )
    if not os.path.exists(npz_path):
        print(f"[Error] No se encontró el dataset en {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    X_clean = data['X_clean'] # (1658, 3, 20)
    Y_clean = data['Y_clean'] # (1658,)
    sujetos = data['sujetos_clean'] # (1658,)
    
    N = len(Y_clean)
    print("=" * 80)
    print("EXPERIMENTO: CAPAS READ-IN VS POOLED EN DATASET CANONICO MULTI-SUJETO")
    print(f"Total pulsos: {N} | Sujetos: Lucas, Santi, Candela | Ventana: 50/50")
    print("=" * 80)

    X_t = torch.tensor(X_clean, dtype=torch.float32)
    lista_suj = ['Lucas', 'Santi', 'Candela']

    # --------------------------------------------------------------------------
    # MODELO A: Baseline Pooled (Sin Read-in)
    # --------------------------------------------------------------------------
    print("\n[1/2] Entrenando MODELO A: Baseline Pooled (Tronco único sin adaptadores)...")
    torch.manual_seed(42)
    model_pooled = BaselinePooledAE(in_channels=3, time_pts=20, latent_dim=2)
    model_pooled = entrenar_modelo(model_pooled, X_t, sujetos, epochs=250, lr=0.008)

    model_pooled.eval()
    with torch.no_grad():
        Z_pooled = model_pooled.encode(X_t).numpy()

    acc_pooled_global, rec_pooled, y_pred_p, _ = evaluar_gmm_hungaro(Z_pooled, Y_clean, seed=42)
    print(f"  -> Exactitud Global GMM (Pooled): {acc_pooled_global*100:.2f}%")
    for s in lista_suj:
        m_s = (sujetos == s)
        acc_s = accuracy_score(
            np.array([VOCALES.index(v) for v in Y_clean[m_s]]),
            y_pred_p[m_s]
        )
        print(f"     * {s:<10}: {acc_s*100:.2f}%")

    # --------------------------------------------------------------------------
    # MODELO B: Multi-Subject con Read-in Layers
    # --------------------------------------------------------------------------
    print("\n[2/2] Entrenando MODELO B: Multi-Subject Read-in (Capas 3x3 por sujeto + Tronco)...")
    torch.manual_seed(42)
    model_readin = MultiSubjectReadInAE(lista_suj, in_channels=3, time_pts=20, latent_dim=2)
    model_readin = entrenar_modelo(model_readin, X_t, sujetos, epochs=250, lr=0.008)

    model_readin.eval()
    with torch.no_grad():
        Z_readin = model_readin.encode(X_t, sujetos).numpy()

    acc_readin_global, rec_readin, y_pred_r, _ = evaluar_gmm_hungaro(Z_readin, Y_clean, seed=42)
    print(f"  -> Exactitud Global GMM (Read-in): {acc_readin_global*100:.2f}%")
    for s in lista_suj:
        m_s = (sujetos == s)
        acc_s = accuracy_score(
            np.array([VOCALES.index(v) for v in Y_clean[m_s]]),
            y_pred_r[m_s]
        )
        print(f"     * {s:<10}: {acc_s*100:.2f}%")

    # --------------------------------------------------------------------------
    # Inspección de Matrices Read-in aprendidas
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MATRICES DE MEZCLA READ-IN APRENDIDAS (Canales Virtuales vs Canales Físicos)")
    print("=" * 80)
    for s in lista_suj:
        W_rin = model_readin.read_in[s].weight.squeeze(-1).detach().numpy()
        b_rin = model_readin.read_in[s].bias.detach().numpy()
        print(f"\nSujeto: {s}")
        print("  W (3x3):")
        for row in W_rin:
            print("    [ " + "  ".join([f"{val:+.3f}" for val in row]) + " ]")
        print("  b (sesgo): [ " + "  ".join([f"{val:+.3f}" for val in b_rin]) + " ]")

    # --------------------------------------------------------------------------
    # Guardar Checkpoints
    # --------------------------------------------------------------------------
    dir_modelos = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/modelos_entrenados")
    os.makedirs(dir_modelos, exist_ok=True)
    p_ckpt = os.path.join(dir_modelos, "modelo_multisujeto_readin_conv2d.pt")
    torch.save({
        'model_state_dict': model_readin.state_dict(),
        'acc_readin_global': acc_readin_global,
        'sujetos': lista_suj
    }, p_ckpt)
    print(f"\n[Guardado] Modelo Read-in: {p_ckpt}")

    # --------------------------------------------------------------------------
    # Graficado Comparativo de 4 Paneles
    # --------------------------------------------------------------------------
    dir_figuras = os.path.join(project_root, "EMG_desarrollo/resultados/resultados_autoencoder/figuras_evaluacion")
    os.makedirs(dir_figuras, exist_ok=True)
    fig_path = os.path.join(dir_figuras, "comparativa_latente_pooled_vs_readin_2d.png")

    fig, axes = plt.subplots(2, 2, figsize=(18, 16), facecolor='white')
    
    # Panel 1: Pooled coloreado por Vocal
    ax = axes[0, 0]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_clean == v)
        ax.scatter(Z_pooled[m, 0], Z_pooled[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=35, alpha=0.6, edgecolors='none')
        c_v = np.mean(Z_pooled[m], axis=0)
        ax.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=140, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax.set_title(f"Baseline Pooled - Vocales (GMM Acc: {acc_pooled_global*100:.1f}%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Z1", fontsize=11)
    ax.set_ylabel("Z2", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(frameon=True, loc='upper right')

    # Panel 2: Pooled coloreado por Sujeto
    ax = axes[0, 1]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    colores_suj = {'Lucas': '#2b5c8f', 'Santi': '#2a9d8f', 'Candela': '#e76f51'}
    for s in lista_suj:
        m = (sujetos == s)
        ax.scatter(Z_pooled[m, 0], Z_pooled[m, 1], c=[colores_suj[s]], marker=MARKERS_SUJETO[s], label=s, s=35, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.set_title("Baseline Pooled - Distribución por Sujeto", fontsize=13, fontweight='bold')
    ax.set_xlabel("Z1", fontsize=11)
    ax.set_ylabel("Z2", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(frameon=True, loc='upper right')

    # Panel 3: Read-in coloreado por Vocal
    ax = axes[1, 0]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for v in VOCALES:
        m = (Y_clean == v)
        ax.scatter(Z_readin[m, 0], Z_readin[m, 1], c=[COLORES_VOCALES[v]], label=f"/{v.lower()}/", s=35, alpha=0.6, edgecolors='none')
        c_v = np.mean(Z_readin[m], axis=0)
        ax.scatter(c_v[0], c_v[1], c=[COLORES_VOCALES[v]], s=140, edgecolors='black', linewidth=1.5, zorder=5)
        ax.annotate(f"/{v.lower()}/", (c_v[0], c_v[1]), fontsize=11, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    ax.set_title(f"Multi-Subject Read-in - Vocales (GMM Acc: {acc_readin_global*100:.1f}%)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Z1", fontsize=11)
    ax.set_ylabel("Z2", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(frameon=True, loc='upper right')

    # Panel 4: Read-in coloreado por Sujeto
    ax = axes[1, 1]
    ax.set_facecolor('white')
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4)
    for s in lista_suj:
        m = (sujetos == s)
        ax.scatter(Z_readin[m, 0], Z_readin[m, 1], c=[colores_suj[s]], marker=MARKERS_SUJETO[s], label=s, s=35, alpha=0.6, edgecolors='black', linewidth=0.3)
    ax.set_title("Multi-Subject Read-in - Distribución por Sujeto", fontsize=13, fontweight='bold')
    ax.set_xlabel("Z1", fontsize=11)
    ax.set_ylabel("Z2", fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(frameon=True, loc='upper right')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[Guardado] Gráfico comparativo de 4 paneles:\n     {fig_path}")

    # Guardar métricas en JSON
    metrics_path = os.path.join(dir_figuras, "metricas_comparativa_readin.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'acc_pooled_global': float(acc_pooled_global),
            'acc_readin_global': float(acc_readin_global),
            'recalls_pooled': [float(r) for r in rec_pooled],
            'recalls_readin': [float(r) for r in rec_readin]
        }, f, indent=4)
    print(f"[Guardado] Métricas JSON: {metrics_path}")
    print("=" * 80)

if __name__ == "__main__":
    ejecutar_experimento()
