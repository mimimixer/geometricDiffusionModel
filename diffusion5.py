"""
Simple diffusion training script (no residual U-Net) for 64x64 grayscale shape images.
- Labels are parsed from filename first digit (1..5)
- Simple Conv encoder/decoder (no residual connections)
- DDPM-style training (model predicts noise)
- Runs on MPS if available (mac), otherwise CUDA/CPU
- Logs to CSV: total params, total_train_time_sec, max_cpu_mem_mb, and per-epoch stats
- Saves sample grid of shape (10 checkpoints x 5 classes)
- Saves t-SNE and UMAP plots of encoder bottleneck

Usage: edit CONFIG at top and run `python simple_diffusion_noresidual_backbone.py`

Requirements:
  torch torchvision numpy pillow psutil scikit-learn umap-learn matplotlib

"""

from __future__ import annotations
import os
import re
import time
import math
import csv
from pathlib import Path
from typing import List

import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import psutil
import matplotlib.pyplot as plt

# ----------------------
# CONFIG (edit here)
# ----------------------
DATA_DIR = "./generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"       # path to PNGs
my_data = os.path.basename(DATA_DIR)
LOG_DIR = "./run_simple_diff"          # output dir
EPOCHS = 750
base_channels = 32
BATCH_SIZE = 64
LR = 2e-4
TIMESTEPS = 100
SEED = 42
NUM_WORKERS = 4
SAMPLES_PER_CLASS = 1            # images per class at each checkpoint
MAX_LATENT_SAMPLES = 1000
IMG_SIZE = 64
NUM_CLASSES = 5

# ----------------------
# Utilities
# ----------------------
CLASS_MAP = {1: "triangle", 2: "rectangle", 3: "square", 4: "circle", 5: "ellipse"}

def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def get_device():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------
# Dataset
# ----------------------
class ShapesDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = sorted([p for p in self.root.rglob("*.png")])
        if not self.files:
            raise FileNotFoundError(f"No PNG files found under {self.root}")
        self.labels = []
        digit_re = re.compile(r"^(\d)")
        for p in self.files:
            m = digit_re.match(p.name)
            if not m:
                raise ValueError(f"Filename must start with label digit 1..5: {p.name}")
            d = int(m.group(1))
            if d not in CLASS_MAP:
                raise ValueError(f"Label digit must be 1..5 in filename: {p.name}")
            self.labels.append(d - 1)  # 0..4

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        arr = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        x = x * 2 - 1  # [-1,1]
        return x, label


# ----------------------
# Linear beta schedule
# ----------------------
class LinearBetaSchedule:
    def __init__(self, timesteps: int = 200, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


# ----------------------
# Simple Conv backbone (no residuals)
# ----------------------
class SimpleConvBackbone(nn.Module):
    def __init__(self, img_ch=1, base_ch=base_channels, cond_dim=128, num_classes=5):
        super().__init__()
        self.cond_emb = nn.Embedding(num_classes, cond_dim)
        # encoder
        self.down1 = nn.Conv2d(img_ch, base_ch, 4, 2, 1)       # 64 -> 32
        self.down2 = nn.Conv2d(base_ch, base_ch*2, 4, 2, 1)    # 32 -> 16
        self.down3 = nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1)  # 16 -> 8
        # mid
        self.mid = nn.Conv2d(base_ch*4, base_ch*4, 3, 1, 1)
        # decoder
        self.up3 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, 2, 1)  # 8 -> 16
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, 2, 1)    # 16 -> 32
        self.up1 = nn.ConvTranspose2d(base_ch, img_ch, 4, 2, 1)       # 32 -> 64
        # small time embedding MLP
        self.time_mlp = nn.Sequential(nn.Linear(1, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.act = nn.SiLU()

    def forward(self, x, t, y):
        # x: (B,1,64,64), t: (B,) long, y: (B,) long
        B = x.size(0)
        # prepare conditioning
        t = t.float().unsqueeze(1)  # (B,1)
        t_emb = self.time_mlp(t)    # (B,cond_dim)
        y_emb = self.cond_emb(y)    # (B,cond_dim)
        cond = t_emb + y_emb        # (B,cond_dim)
        # inject cond by adding to channels (broadcast)

        d1 = self.act(self.down1(x))      # (B, base, 32, 32)
        d2 = self.act(self.down2(d1))     # (B, base*2, 16, 16)
        d3 = self.act(self.down3(d2))     # (B, base*4, 8, 8)

        # inject cond into mid by projecting to feature map
        cond_map = cond.view(B, -1, 1, 1)
        cond_map = cond_map.expand(-1, -1, d3.size(2), d3.size(3))
        mid = self.act(self.mid(d3) + cond_map[:, :d3.size(1), ...])

        u3 = self.act(self.up3(mid))
        u2 = self.act(self.up2(u3))
        out = self.up1(u2)
        return out  # predicts noise in same shape as x

    def get_bottleneck(self, x, y):
        # returns flattened bottleneck features for visualization
        with torch.no_grad():
            d1 = self.act(self.down1(x))
            d2 = self.act(self.down2(d1))
            d3 = self.act(self.down3(d2))
        # global average pool and flatten
        pooled = F.adaptive_avg_pool2d(d3, (1,1)).view(d3.size(0), -1)
        return pooled


# ----------------------
# Diffusion trainer utilities
# ----------------------
class DiffusionTrainer:
    def __init__(self, model: nn.Module, scheduler: LinearBetaSchedule, device: torch.device):
        self.model = model
        self.sch = scheduler
        self.device = device

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.sch.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        b = self.sch.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return a * x0 + b * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, shape: tuple, y: torch.Tensor, steps: int | None = None):
        # Default: full T steps
        T = self.sch.T if steps is None else steps
        B = shape[0]
        img = torch.randn(shape, device=self.device)
        for i in reversed(range(T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps_theta = self.model(img, t, y)
            beta_t = self.sch.betas[t].view(-1,1,1,1)
            sqrt_one_minus = self.sch.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
            sqrt_recip = self.sch.sqrt_recip_alphas[t].view(-1,1,1,1)
            mean = sqrt_recip * (img - beta_t / sqrt_one_minus * eps_theta)
            if i > 0:
                var = self.sch.posterior_variance[t].view(-1,1,1,1)
                noise = torch.randn_like(img)
                img = mean + torch.sqrt(var) * noise
            else:
                img = mean
        return img.clamp(-1,1)


# ----------------------
# Logging
# ----------------------
class CSVLogger:
    def __init__(self, out_dir: Path, total_params: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / f"train_log_for_{my_data}_e{EPOCHS}.csv"
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.max_cpu_mem = 0.0
        # write header
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["total_params", total_params])
            w.writerow(["metric", "value"])  # placeholder for totals
            w.writerow(["epoch", "epochs", "epoch_time_sec", "loss", "cpu_mem_mb"])

    def _mem_snapshot(self):
        rss = self.process.memory_info().rss / (1024**2)
        self.max_cpu_mem = max(self.max_cpu_mem, rss)
        return rss

    def log_epoch(self, epoch: int, epochs: int, epoch_time: float, loss: float):
        cpu_mb = self._mem_snapshot()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch,"/", f"{epochs:<5}", f"{epoch_time:>8.3f}", f"{loss:>8.6f}", f"{cpu_mb:>8.1f}"])

    def finalize(self):
        total_time = time.time() - self.start_time
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["total_train_time_sec", f"{total_time:.3f}"])
            w.writerow(["max_cpu_mem_mb", f"{self.max_cpu_mem:.1f}"])
        return total_time, self.max_cpu_mem


# ----------------------
# Helpers: save grid, visualize
# ----------------------

def save_image_grid(rows: List[torch.Tensor], out_path: Path):
    # rows: list of tensors shape (NUM_CLASSES,1,64,64)
    num_rows = len(rows)
    num_cols = NUM_CLASSES
    cell = IMG_SIZE
    grid = Image.new("L", (num_cols*cell, num_rows*cell), color=255)
    for r, row in enumerate(rows):
        for c in range(num_cols):
            img = row[c]
            img = ((img + 1) / 2).clamp(0,1)
            img_np = (img.squeeze(0).cpu().numpy()*255).astype(np.uint8)
            pil = Image.fromarray(img_np, mode="L")
            grid.paste(pil, (c*cell, r*cell))
    grid.save(out_path)


def visualize_latents(model: SimpleConvBackbone, loader: DataLoader, device: torch.device, out_dir: Path, max_samples: int = 1000):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            h = model.get_bottleneck(x, y)
            feats.append(h.cpu())
            labels.append(y.cpu())
            if sum(b.size(0) for b in labels) >= max_samples:
                break
    if not feats:
        return
    X = torch.cat(feats, dim=0).numpy()
    Y = torch.cat(labels, dim=0).numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", max_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X)
    plt.figure(figsize=(6,5))
    for cls in range(NUM_CLASSES):
        idx = Y == cls
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], s=8, label=CLASS_MAP[cls+1])
    plt.legend(markerscale=2)
    plt.title("Latent space t-SNE (bottleneck)")
    plt.tight_layout()
    plt.savefig(out_dir / f"latents_tsne_for_{my_data}_e{EPOCHS}.png", dpi=200)
    plt.close()

    # UMAP
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean")
        X_umap = reducer.fit_transform(X)
        plt.figure(figsize=(6,5))
        for cls in range(NUM_CLASSES):
            idx = Y == cls
            plt.scatter(X_umap[idx,0], X_umap[idx,1], s=8, label=CLASS_MAP[cls+1])
        plt.legend(markerscale=2)
        plt.title("Latent space UMAP (bottleneck)")
        plt.tight_layout()
        plt.savefig(out_dir / f"latents_umap_for_{my_data}_e{EPOCHS}.png", dpi=200)
        plt.close()
    else:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")


# ----------------------
# Main training routine
# ----------------------

def main():
    seed_all(SEED)
    device = get_device()
    #device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    ds = ShapesDataset(DATA_DIR)
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    vis_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SimpleConvBackbone(img_ch=1, base_ch=base_channels, cond_dim=128, num_classes=NUM_CLASSES).to(device)
    sch = LinearBetaSchedule(timesteps=TIMESTEPS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params}")

    logger = CSVLogger(LOG_DIR, total_params)
    trainer = DiffusionTrainer(model, sch, device)

    # checkpoints (10 rows)
    checkpoints = sorted(set(max(1, int(round(EPOCHS * p))) for p in np.linspace(0.1, 1.0, 10)))
    sample_rows: List[torch.Tensor] = []

    model.train()
    global_start = time.time()
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)
            t = torch.randint(0, sch.T, (B,), device=device).long()
            x_noisy, noise = trainer.q_sample(x, t)
            opt.zero_grad()
            pred_noise = model(x_noisy, t, y)
            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        logger.log_epoch(epoch, EPOCHS, epoch_time, avg_loss)
        print(f"Epoch {epoch}/{EPOCHS} - loss {avg_loss:.6f} - {epoch_time:.2f}s")

        # sampling checkpoint
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                imgs_per_class = []
                for cls in range(NUM_CLASSES):
                    y_cls = torch.full((SAMPLES_PER_CLASS,), cls, dtype=torch.long, device=device)
                    gen = trainer.p_sample_loop((SAMPLES_PER_CLASS, 1, IMG_SIZE, IMG_SIZE), y_cls)
                    imgs_per_class.append(gen[0:1])
                row = torch.cat(imgs_per_class, dim=0)  # (NUM_CLASSES,1,H,W)
                sample_rows.append(row.detach().cpu())
            model.train()

    torch.save(model.state_dict(), "not_used/model.pt")
    print("✅ Modell gespeichert als model.pt")

    total_time, max_cpu = logger.finalize()
    print(f"Total train time: {total_time:.1f}s | Max CPU RSS: {max_cpu:.1f} MB | Params: {total_params}")

    out_dir = Path(LOG_DIR)
    # save grid
    if len(sample_rows) > 0:
        grid_path = out_dir / f"samples_grid_10x5_for_{my_data}_e{EPOCHS}.png"
        save_image_grid(sample_rows, grid_path)
        print(f"Saved sample grid to {grid_path}")

    # save final per-class samples (one each)
    model.eval()
    with torch.no_grad():
        imgs_per_class = []
        for cls in range(NUM_CLASSES):
            y_cls = torch.full((1,), cls, dtype=torch.long, device=device)
            gen = trainer.p_sample_loop((1, 1, IMG_SIZE, IMG_SIZE), y_cls)
            imgs_per_class.append(gen)
        final_row = torch.cat(imgs_per_class, dim=0)
        save_image_grid([final_row.cpu()], out_dir / "final_samples_1x5.png")

    # visualize latents
    visualize_latents(model, vis_loader, device, out_dir, max_samples=MAX_LATENT_SAMPLES)


if __name__ == "__main__":
    main()
