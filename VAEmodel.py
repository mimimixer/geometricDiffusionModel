import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dimension = 32
shape_of_image = 64
beta = 5

class Encoder(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: von Bild → Mittelwert + Standardabweichung
        self.fc1 = nn.Linear(shape_of_image*shape_of_image, latent_dimension*latent_dimension)
        self.fc2 = nn.Linear(shape_of_image*shape_of_image, latent_dimension*latent_dimension)
        self.fc3 = nn.Linear(shape_of_image*shape_of_image, latent_dimension*latent_dimension)
        self.fc_mu = nn.Linear(latent_dimension*latent_dimension, latent_dim) # nn calculates mean of latent distribution
        self.fc_logvar = nn.Linear(latent_dimension*latent_dimension, latent_dim) # nn calculates log-variance of latent distribution

    def forward(self, x):
        x_flat = torch.flatten(x, start_dim=1)
        h = torch.relu(self.fc1(x_flat))
        h = torch.relu(self.fc2(x_flat))
        h = torch.relu(self.fc3(x_flat))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.latent_dim = latent_dim
        # Decoder: von z → rekonstruiertes Bild
        self.fc1 = nn.Linear(latent_dim, latent_dimension * latent_dimension)
        self.fc2 = nn.Linear(latent_dimension * latent_dimension, shape_of_image*shape_of_image)
        self.fc3 = nn.Linear(latent_dimension * latent_dimension, shape_of_image*shape_of_image)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(h))
        x = torch.sigmoid(self.fc3(h))
        return x.view(-1, 1, shape_of_image, shape_of_image)

class VAE(nn.Module):
    def __init__(self, latent_dim=latent_dimension):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # z ~ N(mu, sigma^2)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta*kl_div

