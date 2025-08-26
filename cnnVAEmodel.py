import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dimension = 32
shape_of_image = 64
beta = 10

class Encoder(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # Bx1x64x64 -> Bx16x32x32 (z.B. für 64x64 input)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1) # Bx16x32x32 -> Bx32x16x16
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1) # Bx32x16x16 -> Bx64x8x8
        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.fc_dec = nn.Linear(latent_dim, 64*8*8)#256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 4, 2, 1)  # 8x8 -> 16x16
        self.deconv2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 16x16 -> 32x32
        self.deconv3 = nn.ConvTranspose2d(16, 1, 4, 2, 1)  # 32x32 -> 64x64
    def forward(self, z):
        x = self.fc_dec(z)
        x = x.view(-1, 64, 8, 8)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x_recon = torch.sigmoid(self.deconv3(x))  # Sigmoid für Werte 0..1
        return x_recon

class cnnVAE(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def cnn_vae_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

print(cnnVAE)