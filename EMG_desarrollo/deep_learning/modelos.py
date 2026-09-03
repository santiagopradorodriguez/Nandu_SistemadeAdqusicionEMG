import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    def __init__(self, latent_dim=8, target_length=100, kernel_size=5):
        super(ConvAutoencoder1D, self).__init__()
        self.target_length = target_length
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Entrada: (Batch, 3, target_length)
        # Usamos stride=1 y padding=(k//2) para preservar exactamente la resolución temporal
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=kernel_size, stride=1, padding=padding), # -> (B, 16, target_length)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=1, padding=padding), # -> (B, 32, target_length)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # Al mantener target_length puntos de tiempo, aplanamos 32 canales * target_length
        self.encoder_fc = nn.Sequential(
            nn.Linear(32 * target_length, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )
        
        # --- DECODER ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32 * target_length),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=kernel_size, stride=1, padding=padding), # -> (B, 16, target_length)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(16, 3, kernel_size=kernel_size, stride=1, padding=padding), # -> (B, 3, target_length)
            nn.ReLU()
        )
        
        # --- CLASSIFIER (Camino B) ---
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 5) # 5 Vocales
        )

    def encode(self, x):
        is_single = (x.size(0) == 1)
        was_training = self.training
        if is_single and was_training:
            self.eval()
        try:
            x = self.encoder_cnn(x)
            x = x.view(x.size(0), -1) # Flatten
            latent = self.encoder_fc(x)
        finally:
            if is_single and was_training:
                self.train()
        return latent

    def decode(self, latent):
        is_single = (latent.size(0) == 1)
        was_training = self.training
        if is_single and was_training:
            self.eval()
        try:
            x = self.decoder_fc(latent)
            x = x.view(x.size(0), 32, self.target_length) # Reshape
            x = self.decoder_cnn(x)
        finally:
            if is_single and was_training:
                self.train()
        return x

    def forward(self, x):
        is_single = (x.size(0) == 1)
        was_training = self.training
        if is_single and was_training:
            self.eval()
        try:
            latent = self.encode(x)
            reconstruction = self.decode(latent)
            logits = self.classifier(latent)
        finally:
            if is_single and was_training:
                self.train()
        return reconstruction, latent, logits
