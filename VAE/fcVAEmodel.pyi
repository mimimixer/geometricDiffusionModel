import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dimension = 32
shape_of_image = 64
beta = 0.01

class VAE(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: von Bild → Mittelwert + Standardabweichung
        self.encoder = nn.Sequential(
            nn.Flatten(),                                                   # transforms to onedimentional tensor
            nn.Linear(shape_of_image*shape_of_image, latent_dimension*latent_dimension),   # linear tranform. to input data, input and output features: defines network
            nn.ReLU()                                                       # activation
        )
        self.fc_mu = nn.Linear(latent_dimension*latent_dimension, latent_dim) # nn calculates mean of latent distribution
        self.fc_logvar = nn.Linear(latent_dimension*latent_dimension, latent_dim) # nn calculates log-variance of latent distribution

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dimension * latent_dimension),
            nn.ReLU(),
            nn.Linear(latent_dimension * latent_dimension, shape_of_image * shape_of_image),
            nn.Sigmoid()  # für Werte zwischen 0 und 1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z ~ N(mu, sigma^2)

    def forward(self, x):
        x_flat = self.encoder(x)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z).view(-1, 1, 28, 28)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta*kl_div