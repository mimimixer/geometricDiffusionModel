from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from sklearn.manifold import TSNE
import umap
import re
import numpy as np


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#model_path = "VAE/bestResults/cnnVAEmodelTrained220107-all_shapes_dir-dataset_straight_centered_filled2500-32e150-b10.pt"
#folder_path = "../generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"
save_path = "cnnVAEvisualisation"

class FolderImageDataset(Dataset):
    def __init__(self, folder_path, image_size):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x > 0.5).float())  # optional threshold
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        img = Image.open(img_path).convert("L")  # grayscale
        img = self.transform(img)

        # Extract the first number from the filename
        filename = self.image_files[idx]
        match = re.search(r'\d+', filename)
        label = int(match.group()) if match else 0

        return img, label

def get_latents_from_vae(vae_model, dataloader):
    zs, labels = [], []
    vae_model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, tuple):
                imgs, y = batch
            else:
                imgs = batch
                y = [0]*len(batch)  # Dummy-Labels

            # Channel-Dimension sicherstellen
            if imgs.dim() == 3:  # [B, H, W]
                imgs = imgs.unsqueeze(1)  # -> [B, 1, H, W]

            imgs = imgs.to(device).float()
            mu, logvar = vae_model.encoder(imgs)
            z = vae_model.reparameterize(mu, logvar)
            zs.append(z.cpu())

            # Labels in 1D Tensor konvertieren
            if isinstance(y, int):
                y = torch.tensor([y], dtype=torch.long)
            elif isinstance(y, list):
                y = torch.tensor(y, dtype=torch.long)
            elif isinstance(y, torch.Tensor) and y.dim() == 0:
                y = y.unsqueeze(0)
            labels.append(y)
    return torch.cat(zs).numpy(), torch.cat(labels).numpy()

# ---- t-SNE Visualisierung ----
def visualize_tsne(zs, labels, save_path):
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30, random_state=42)
    zs_2d = tsne.fit_transform(zs)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for ul in unique_labels:
        idxs = labels == ul
        plt.scatter(zs_2d[idxs, 0], zs_2d[idxs, 1], label=str(ul), s=10)
    plt.legend(title="Labels")
    plt.title("cnnVAE Latent space visualization with t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(f"{save_path}tSNE", dpi=300)
    plt.show()

# ---- UMAP Visualisierung ----
def visualize_umap(zs, labels, save_path):
    reducer = umap.UMAP(n_components=2, random_state=42)
    zs_2d = reducer.fit_transform(zs)

    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for ul in unique_labels:
        idxs = labels == ul
        plt.scatter(zs_2d[idxs, 0], zs_2d[idxs, 1], label=str(ul), s=10)
    plt.legend(title="Labels")
    plt.title("Latent space visualization with UMAP")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig(f"{save_path}UMAP", dpi=300)
    plt.show()

"""
    # Dataset erstellen
    image_size = cnnVAEmodel.shape_of_image
    
    dataset = FolderImageDataset(folder_path, image_size)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # VAE laden
    vae = cnnVAEmodel.cnnVAE(latent_dim=32).to(device)
    vae.load_state_dict(torch.load(model_path, map_location=device))
    vae.eval()
    
    # Latents extrahieren
    zs, labels = get_latents_from_vae(vae, dataloader)
    
    # Visualisierung
    visualize_tsne(zs, labels)
    visualize_umap(zs, labels)
"""
