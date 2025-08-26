from datetime import datetime
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils

import cnnVAEmodel

filename = "VAE/friday/cnnVAEmodelTrained231524-all_shapes_dir-dataset_straigt_centered_outlined-32e150-b7.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = cnnVAEmodel.cnnVAE(latent_dim=cnnVAEmodel.latent_dimension).to(device)
model.load_state_dict(torch.load(filename))

current_time = datetime.now().strftime("%d%H%M")


def generate_images(model, latent_dim=cnnVAEmodel.latent_dimension, n=10, nrow=5, name = filename):
    model.eval()
    plot_name = (f"plot_{name}_{current_time}.png")
    with torch.no_grad():
        z = torch.randn(n, latent_dim).to(device) # zufällige Punkte im Latent Space
        samples = model.decoder(z)#.view(-1, 1, cnnVAEmodel.shape_of_image, cnnVAEmodel.shape_of_image)
        #grid = torch.cat([img for img in samples], dim=2)  # nebeneinander
        #plt.imshow(grid.squeeze().numpy(), cmap='gray')
        grid = vutils.make_grid(samples, nrow=nrow, normalize=True, pad_value=1)
        # Plotten
        plt.figure(figsize=(10, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap='gray')

        plt.axis('off')
        plt.title("Generated Images from cnnVAE")
        plt.savefig(plot_name, bbox_inches='tight')
        plt.show()

#generate_images(model)