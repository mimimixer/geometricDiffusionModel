import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -------------------------------
# Device
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
number_of_timesteps = 30

# -------------------------------
# UNet (muss identisch zum Trainingsmodell sein!)
# -------------------------------
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x, t):
        t = t.float().view(-1, 1, 1, 1) / 1000.0
        x = F.relu(self.enc1(x) + t)
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        return self.out(x)

# -------------------------------
# Noise Schedule
# -------------------------------
def get_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

# -------------------------------
# Sampling
# -------------------------------
@torch.no_grad()
def sample(model, betas, num_samples=5, timesteps=1000):
    x = torch.randn(num_samples, 1, 64, 64, device=device)
    alpha = torch.cumprod(1 - betas, dim=0).to(device)

    for t in reversed(range(timesteps)):
        z = torch.randn_like(x) if t > 0 else 0
        pred_noise = model(x, torch.tensor([t]*num_samples, device=device))
        mean = (1 / torch.sqrt(1 - betas[t])) * (
            x - (betas[t] / (1 - alpha[t]).sqrt()) * pred_noise
        )
        x = mean + (betas[t].sqrt() * z)
    return x

# -------------------------------
# Main: Laden + Generieren
# -------------------------------
if __name__ == "__main__":
    T = 1000
    betas = get_beta_schedule(T).to(device)

    # Modell laden
    model = SimpleUNet().to(device)
    model.load_state_dict(torch.load("../diffusion3/diffusion_shapes.pth", map_location=device))
    model.eval()

    # 5 Bilder generieren
    imgs = sample(model, betas, num_samples=5, timesteps=number_of_timesteps)
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # [0,1]

    # Plotten
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i].squeeze().cpu(), cmap="gray")
        ax.axis("off")
    plt.suptitle("Generated Shapes")
    plt.savefig("generated_samples.png")
    plt.show()