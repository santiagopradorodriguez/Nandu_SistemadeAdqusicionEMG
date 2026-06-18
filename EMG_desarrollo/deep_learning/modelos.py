import torch
import torch.nn as nn

class ConvAutoencoder1D(nn.Module):
    def __init__(self, latent_dim=32):
        super(ConvAutoencoder1D, self).__init__()
        
        # Entrada: (Batch, 3, 100)
        # Usamos stride=1 para NO destruir la resolución temporal de 20ms
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=5, stride=1, padding=2), # -> (B, 16, 100)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2), # -> (B, 32, 100)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # Al mantener 100 puntos de tiempo, aplanamos 32 canales * 100
        self.encoder_fc = nn.Sequential(
            nn.Linear(32 * 100, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, latent_dim)
        )
        
        # --- DECODER ---
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 32 * 100),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=1, padding=2), # -> (B, 16, 100)
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(16, 3, kernel_size=5, stride=1, padding=2), # -> (B, 3, 100)
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
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1) # Flatten
        latent = self.encoder_fc(x)
        return latent

    def decode(self, latent):
        x = self.decoder_fc(latent)
        x = x.view(x.size(0), 32, 100) # Reshape
        x = self.decoder_cnn(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        logits = self.classifier(latent)
        return reconstruction, latent, logits
