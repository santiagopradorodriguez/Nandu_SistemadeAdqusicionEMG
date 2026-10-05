import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    """
    Autoencoder Convolucional 1D Ortogonal y Semisupervisado (Generación Récord).
    Basado en la arquitectura campeona de 89.04% (Lucas GMM / P5 Continua):
    - Convoluciones 1D con canales (6, 12), kernel 5, sin bias y padding simétrico.
    - Activación Tanh continua para preservar la variedad motora suave y simétrica.
    - Regularización ortogonal estricta de pesos (Weight Orthogonality Loss).
    - Regularización de dispersión latente (Latent Covariance Loss).
    - Cabezal de clasificación semisupervisado regularizado directamente conectado a latent_dim sin BatchNorms espurios.
    """
    def __init__(self, in_channels=3, latent_dim=2, target_length=20, kernel_size=5, conv_channels=(6, 12), num_classes=5, act_name='tanh'):
        super(ConvAutoencoder1D, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.target_length = target_length
        self.kernel_size = kernel_size
        self.conv_channels = conv_channels
        c1, c2 = conv_channels
        pad = kernel_size // 2
        
        # --- ENCODER CONVOLUCIONAL ORTOGONAL ---
        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=kernel_size, padding=pad, bias=False)
        self.act = nn.Tanh() if act_name.lower() == 'tanh' else nn.ReLU()
            
        self.fc1 = nn.Linear(c2 * target_length, 32, bias=False)
        self.fc2 = nn.Linear(32, latent_dim, bias=False)
        
        # --- DECODER CONVOLUCIONAL ORTOGONAL ---
        self.dfc1 = nn.Linear(latent_dim, 32, bias=False)
        self.dfc2 = nn.Linear(32, c2 * target_length, bias=False)
        self.deconv1 = nn.ConvTranspose1d(c2, c1, kernel_size=kernel_size, padding=pad, bias=False)
        self.deconv2 = nn.ConvTranspose1d(c1, in_channels, kernel_size=kernel_size, padding=pad, bias=False)
        
        # --- CLASIFICADOR SEMISUPERVISADO REGULARIZADO ---
        # Cabezal compacto sin BatchNorm para evitar desfases estadísticos en tiempo real / continua
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16, num_classes)
        )

    def encode(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.target_length)
        else:
            x_3d = x
        h1 = self.act(self.conv1(x_3d))
        h2 = self.act(self.conv2(h1))
        h_flat = h2.view(h2.shape[0], -1)
        h3 = self.act(self.fc1(h_flat))
        latent = self.fc2(h3)
        return latent

    def decode(self, latent):
        c2 = self.conv_channels[1]
        dh1 = self.act(self.dfc1(latent))
        dh2 = self.act(self.dfc2(dh1))
        dh2_3d = dh2.view(dh2.shape[0], c2, self.target_length)
        dh3 = self.act(self.deconv1(dh2_3d))
        recon_3d = self.deconv2(dh3)
        return recon_3d

    def forward(self, x):
        if x.dim() == 2:
            x_3d = x.view(x.shape[0], self.in_channels, self.target_length)
        else:
            x_3d = x
            
        latent = self.encode(x_3d)
        reconstruction = self.decode(latent)
        logits = self.classifier(latent)
        
        # Si la entrada era aplanada 2D, devolver la reconstrucción aplanada para mantener compatibilidad
        if x.dim() == 2:
            reconstruction = reconstruction.view(reconstruction.shape[0], -1)
            
        return reconstruction, latent, logits

    def weight_orthogonality_loss(self):
        """Calcula la penalización ortogonal W W^T = I para todas las capas lineales y convolucionales."""
        loss = torch.tensor(0.0, device=self.conv1.weight.device)
        for layer in [self.fc1, self.fc2, self.dfc1, self.dfc2]:
            W = layer.weight
            d0, d1 = W.shape
            gram = torch.mm(W, W.t()) if d0 < d1 else torch.mm(W.t(), W)
            I = torch.eye(min(d0, d1), device=W.device)
            loss = loss + torch.sum((gram - I) ** 2)
            
        for conv in [self.conv1, self.conv2, self.deconv1, self.deconv2]:
            W = conv.weight.view(conv.weight.shape[0], -1)
            d0, d1 = W.shape
            gram = torch.mm(W, W.t()) if d0 < d1 else torch.mm(W.t(), W)
            I = torch.eye(min(d0, d1), device=W.device)
            loss = loss + torch.sum((gram - I) ** 2)
            
        return loss

    def latent_covariance_loss(self, z):
        """Penaliza la correlación entre dimensiones latentes para forzar dispersión ortogonal Cov(Z) = I."""
        N = z.shape[0]
        if N <= 1:
            return torch.tensor(0.0, device=z.device)
        z_cent = z - torch.mean(z, dim=0, keepdim=True)
        cov_z = torch.mm(z_cent.t(), z_cent) / (N - 1)
        I_d = torch.eye(z.shape[1], device=z.device)
        return torch.sum((cov_z - I_d) ** 2)

