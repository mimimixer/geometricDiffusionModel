import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

data = "../generateShapes/dataset_straight_centered/dataset_straight_centered_filled2500/"
number_of_epochs = 1000
number_of_timesteps = 50

# -------------------------------
# Logging Setup
# -------------------------------
logging.basicConfig(filename="../diffusion3/train.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------
# Device (MPS if available)
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# Dummy UNet (kleines Modell für 64x64)
# -------------------------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)

        self.dec1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        # Zeitinformation simpel einbetten
        t = t.float().view(-1, 1, 1, 1) / 1000.0
        x1 = F.relu(self.enc1(x) + t)
        x2 = F.relu(self.enc2(x1))
        x3 = F.relu(self.enc3(x2))
        x4 = F.relu(self.dec1(x3))
        x5 = F.relu(self.dec2(x4))
        return self.out(x5)

# -------------------------------
# Noise Schedule
# -------------------------------
#def get_beta_schedule(T, start=1e-4, end=0.02):
#    return torch.linspace(start, end, T)
def cosine_beta_schedule(T, s=0.008):
    steps = torch.arange(T+1, dtype=torch.float64)
    alphas_cumprod = torch.cos(((steps/T) + s) / (1+s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    #return torch.clip(torch.tensor(betas, dtype=torch.float32), 0.0001, 0.9999)
    return torch.clip(betas.clone().detach().float(), 0.0001, 0.9999)
# -------------------------------
# Training Step
# -------------------------------
def diffusion_loss(model, x0, t, betas, device):
    noise = torch.randn_like(x0)
    alpha = torch.cumprod(1 - betas, dim=0).to(device)
    sqrt_alpha = alpha[t].sqrt().view(-1, 1, 1, 1)
    sqrt_one_minus = (1 - alpha[t]).sqrt().view(-1, 1, 1, 1)
    xt = sqrt_alpha * x0 + sqrt_one_minus * noise
    pred_noise = model(xt, t)
    return F.mse_loss(pred_noise, noise)

# -------------------------------
# Inference / Sampling
# -------------------------------
@torch.no_grad()
def sample(model, betas, num_samples=5, timesteps=number_of_timesteps):
    x = torch.randn(num_samples, 1, 64, 64, device=device)
    alpha = torch.cumprod(1 - betas, dim=0).to(device)
    for t in reversed(range(timesteps)):
        z = torch.randn_like(x) if t > 0 else 0
        pred_noise = model(x, torch.tensor([t]*num_samples, device=device))
        coef = 1 / torch.sqrt(1 - betas[t])
        mean = (1 / torch.sqrt(1 - betas[t])) * (
            x - (betas[t] / (1 - alpha[t]).sqrt()) * pred_noise
        )
        x = mean + (betas[t].sqrt() * z)
    return x

# -------------------------------
# Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)  # [-1,1]
])

dataset = datasets.ImageFolder(data, transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------
# Training Loop
# -------------------------------
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
T = 1000
betas = cosine_beta_schedule(T, s=0.008).to(device) #get_beta_schedule(T).to(device)

epochs = number_of_epochs
plot_interval = max(1, epochs // 10)

for epoch in range(1, epochs + 1):
    start = time.time()
    for x, _ in loader:
        x = x.to(device)
        t = torch.randint(0, T, (x.size(0),), device=device).long()
        loss = diffusion_loss(model, x, t, betas, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    duration = time.time() - start
    mem = torch.mps.current_allocated_memory() if device.type == "mps" else 0
    log_msg = f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Time: {duration:.2f}s, Mem: {mem/1e6:.2f}MB"
    print(log_msg)
    logging.info(log_msg)

    # alle 10% Epochen Samples plotten
    if epoch % plot_interval == 0:
        imgs = sample(model, betas, num_samples=5, timesteps=number_of_timesteps)
        imgs = (imgs.clamp(-1, 1) + 1) / 2  # [0,1]
        """grid = torch.cat([img for img in imgs], dim=2).squeeze().cpu()
        plt.imshow(grid, cmap="gray")
        plt.axis("off")
        plt.title(f"Samples at Epoch {epoch}")
        plt.savefig(f"samples_epoch{epoch}.png")
        plt.close()"""
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axes):
            ax.imshow(imgs[i].squeeze().cpu(), cmap="gray")
            ax.axis("off")
        plt.suptitle(f"Samples at Epoch {epoch}")
        plt.savefig(f"samples_epoch{epoch}.png")
        plt.close()

# Speichern
torch.save(model.state_dict(), "../diffusion3/diffusion_shapes.pth")

