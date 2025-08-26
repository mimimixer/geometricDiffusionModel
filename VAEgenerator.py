import os
from datetime import datetime

import matplotlib.pyplot as plt
import torch
import VAEmodel

filename = ("VAEmodelTrained131351-ell_ol_c-32e150b5L364-3fc.pt")
model = VAEmodel.VAE(latent_dim=VAEmodel.latent_dimension)
model.load_state_dict(torch.load(filename))
loss_info = os.path.splitext(filename)[0].split("-")[-1]

current_time = datetime.now().strftime("%d%H%M")
plot_name = (f"VAEplot{current_time}.png")

def generate_images(model, latent_dim=VAEmodel.latent_dimension, n=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, latent_dim)  # zufällige Punkte im Latent Space
        samples = model.decoder(z).view(-1, 1, VAEmodel.shape_of_image, VAEmodel.shape_of_image)
        grid = torch.cat([img for img in samples], dim=2)  # nebeneinander
        plt.imshow(grid.squeeze().numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"Generated Images from VAE - training loss {loss_info}")
        plt.savefig(plot_name, bbox_inches='tight')
        plt.show()


generate_images(model)