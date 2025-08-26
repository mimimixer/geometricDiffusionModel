# Unconditional Diffusion Model for 64x64 Geometric Shapes
# Author: ChatGPT | User: [You]

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.datasets as dsets
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


my_data = "../generateShapes/dataset_straight_centered/dataset_straight_centered_filled/"

# ----------------------------
# 1. Parameters
# ----------------------------
IMG_SIZE = 64
BATCH_SIZE = 64
CHANNELS = 1
EPOCHS = 20
LR = 2e-4
TIMESTEPS = 1000

# ----------------------------
# 2. Dataset & Transform
# ----------------------------
data_transform = T.Compose([
    T.Grayscale(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Lambda(lambda t: (t * 2) - 1)  # scale to [-1, 1]
])

dataset = dsets.ImageFolder(root='my_data', transform=data_transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# 3. Noise Schedule
# ----------------------------
betas = torch.linspace(1e-4, 0.02, TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Helper to add noise at timestep t
def add_noise(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

# ----------------------------
# 4. U-Net Definition
# ----------------------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(CHANNELS + 1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2))

        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2))

        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())

        self.final = nn.Conv2d(64, CHANNELS, 1)

    def forward(self, x, t):
        # Append timestep as extra channel
        t = t[:, None, None, None].float() / TIMESTEPS
        t = t.expand(-1, 1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([x, t], dim=1)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.middle(x2)
        x = self.up2(x3)
        x = self.up1(x)
        return self.final(x)

# ----------------------------
# 5. Training Loop
# ----------------------------
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, _ in pbar:
        images = images.to(device)
        t = torch.randint(0, TIMESTEPS, (images.size(0),), device=device)
        noise = torch.randn_like(images)
        x_noisy = add_noise(images, t, noise)
        predicted_noise = model(x_noisy, t)

        loss = F.mse_loss(predicted_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    # Save model & sample
    torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pt")
