import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from PIL import Image

import cnnVAEmodel

# ⚡ Device MPS
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- Encoder-Klasse wieder so definieren wie beim Training! ----
class Encoder(torch.nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 4, 2, 1)  # Bx1x64x64 -> Bx16x32x32 (z.B. für 64x64 input)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1) # Bx16x32x32 -> Bx32x16x16
        self.dropout1 = nn.Dropout(0.2)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1) # Bx32x16x16 -> Bx64x8x8

        self.fc_mu = nn.Linear(64*8*8, latent_dim)
        self.fc_logvar = nn.Linear(64*8*8, latent_dim)

    def forward(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.dropout1(x)
        x = nn.relu(self.conv3(x))
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        return mu, logvar

# ---- Hilfsfunktion: Embeddings aus Encoder extrahieren ----
def get_latents(model_path, dataloader, latent_dim=32):
    encoder = Encoder(latent_dim).to(device)
    encoder.load_state_dict(torch.load(model_path, map_location=device))
    encoder.eval()

    zs, labels = [], []
    with torch.no_grad():
        for imgs, y in dataloader:
            imgs = imgs.to(device)
            mu, logvar = encoder(imgs)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            zs.append(z.cpu())
            labels.append(y)

    return torch.cat(zs).numpy(), torch.cat(labels).numpy()

# ---- t-SNE Visualisierung ----
def visualize_tsne(zs, labels):
    tsne = TSNE(n_components=2, learning_rate="auto", init="random", perplexity=30)
    zs_2d = tsne.fit_transform(zs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="Class label")
    plt.title("Latent space visualization with t-SNE")
    plt.show()

# ---- UMAP Visualisierung ----
def visualize_umap(zs, labels):
    reducer = umap.UMAP(n_components=2, random_state=42)
    zs_2d = reducer.fit_transform(zs)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(zs_2d[:, 0], zs_2d[:, 1], c=labels, cmap="tab10", s=10)
    plt.colorbar(scatter, label="Class label")
    plt.title("Latent space visualization with UMAP")
    plt.show()


image_folder_path = "../generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"
image_size = cnnVAEmodel.shape_of_image

def create_dataset_with_labels(folder_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    image_tensors = []
    labels = []

    for f in os.listdir(folder_path):
        if f.endswith(".png"):
            img = Image.open(os.path.join(folder_path, f)).convert("L")
            img = transform(img)
            img = (img > 0.5).float()  # optional thresholding
            image_tensors.append(img)
            labels.append(0)  # Dummy-Label, falls nötig

    # ---- In einen Tensor stapeln ----
    images = torch.stack(image_tensors).to(device)
    labels = torch.tensor(labels)

    # ---- In Batches für den Encoder ----
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]

#current encoder
model_path = "/cnnVAEmodelTrained211142-all2500_fi_c-32e100-b3.pt"
dataset = create_dataset_with_labels(image_folder_path, image_size)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
zs, labels = get_latents(model_path, dataset, latent_dim=32)
visualize_tsne(zs, labels)
visualize_umap(zs, labels)