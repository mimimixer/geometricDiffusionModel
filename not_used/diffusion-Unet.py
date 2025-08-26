"""
Simple class-conditional diffusion (DDPM-style) for 64x64 grayscale PNGs
with 5 shape classes labeled in filenames (first digit 1..5):
  1=triangle, 2=rectangle, 3=square, 4=circle, 5=ellipse

Features:
- U-Net backbone: 3 down blocks, 1 mid block, 3 up blocks, residual skips.
- Class conditioning via label embeddings (+ time embeddings) injected in each block.
- Trains on Mac using MPS if available (also supports CPU/CUDA).
- Dataloader parses labels from filename prefix; expects 2500 imgs per class (but not required).
- Logging to CSV: total_params, total_train_time_sec, max_cpu_mem_mb, and per-epoch: epoch, epochs,
  epoch_time_sec, loss, cpu_mem_mb, cuda_mem_mb (if CUDA), mps_mem_mb (if MPS).
- After every 10% of epochs, generates 1 sample per class (5 total) and stores rows; at end saves a
  10x5 grid (rows = checkpoints, cols = classes) as PNG.
- Extracts latents from encoder bottleneck on a subset and visualizes with t-SNE and UMAP.

Usage (example):
    python simple_diffusion_shapes.py \
        --data_dir /path/to/pngs \
        --epochs 50 \
        --batch_size 64 \
        --lr 2e-4 \
        --log_dir ./logs_run1 \
        --num_workers 4

Notes:
- Requires: torch, torchvision, numpy, pillow, psutil, scikit-learn, umap-learn, matplotlib
- UMAP is optional; if umap-learn is missing, the script will skip UMAP with a warning.
- This script keeps things intentionally simple for clarity (no EMA, no v-pred, linear betas, T=200).
"""
from __future__ import annotations
import os
import re
import time
import math
import csv
import random
from pathlib import Path
from typing import List, Tuple, Optional

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

# ------------------------------
# Utilities
# ------------------------------
CLASS_MAP = {1: "triangle", 2: "rectangle", 3: "square", 4: "circle", 5: "ellipse"}
CLASS_IDS = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = 5


def seed_all(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------
# Dataset
# ------------------------------
class ShapesDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.files = [p for p in self.root.rglob("*.png")]
        if not self.files:
            raise FileNotFoundError(f"No PNG files found under {self.root}")
        # Pre-parse labels from filenames: first digit 1..5
        self.labels = []
        digit_re = re.compile(r"^(\d)")
        for p in self.files:
            m = digit_re.match(p.name)
            if not m:
                raise ValueError(f"Filename must start with label digit 1..5: {p.name}")
            d = int(m.group(1))
            if d not in CLASS_MAP:
                raise ValueError(f"Label digit must be 1..5 in filename: {p.name}")
            self.labels.append(d - 1)  # store 0..4

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]
        img = Image.open(path).convert("L")  # grayscale
        img = img.resize((64, 64), Image.NEAREST)  # ensure size
        x = torch.from_numpy(np.array(img)).float() / 255.0
        x = x.unsqueeze(0)  # (1, 64, 64)
        x = x * 2 - 1  # [-1, 1]
        return x, label


# ------------------------------
# Diffusion scheduler (DDPM-style)
# ------------------------------
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
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def to(self, device):
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(device))
        return self


# ------------------------------
# Embeddings
# ------------------------------
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor):
        # timesteps: (B,) int or float in [0, T)
        device = timesteps.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb  # (B, dim)


# ------------------------------
# U-Net blocks
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        # conditioning projects (time + class)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch)
        )

    def forward(self, x, cond_vec):
        h = self.conv1(x)
        h = self.gn1(h)
        h = h + self.cond(cond_vec).unsqueeze(-1).unsqueeze(-2)
        h = self.act(h)
        h = self.conv2(h)
        h = self.gn2(h)
        return self.act(h + self.skip(x))


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.res1 = ResidualBlock(in_ch, out_ch, cond_dim)
        self.res2 = ResidualBlock(out_ch, out_ch, cond_dim)
        self.down = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, cond_vec):
        x = self.res1(x, cond_vec)
        x = self.res2(x, cond_vec)
        skip = x
        x = self.down(x)
        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res1 = ResidualBlock(out_ch * 2, out_ch, cond_dim)  # concat skip
        self.res2 = ResidualBlock(out_ch, out_ch, cond_dim)

    def forward(self, x, skip, cond_vec):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, cond_vec)
        x = self.res2(x, cond_vec)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, ch_mults=(1,2,4), cond_dim=256, num_classes=5):
        super().__init__()
        c1, c2, c3 = [base_ch*m for m in ch_mults]
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch, c1, cond_dim)
        self.down2 = DownBlock(c1, c2, cond_dim)
        self.down3 = DownBlock(c2, c3, cond_dim)

        self.mid1 = ResidualBlock(c3, c3, cond_dim)
        self.mid2 = ResidualBlock(c3, c3, cond_dim)

        self.up3 = UpBlock(c3, c3, c2, cond_dim)
        self.up2 = UpBlock(c2, c3, c1, cond_dim)
        self.up1 = UpBlock(c1, c2, base_ch, cond_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

        # embeddings
        self.time_emb = SinusoidalPositionEmbeddings(cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim*4), nn.SiLU(), nn.Linear(cond_dim*4, cond_dim))
        self.label_emb = nn.Embedding(num_classes, cond_dim)
        self.label_mlp = nn.Sequential(nn.Linear(cond_dim, cond_dim*4), nn.SiLU(), nn.Linear(cond_dim*4, cond_dim))

        self._last_latents = None  # store bottleneck for visualization

    def cond_vec(self, t, y):
        t_emb = self.time_mlp(self.time_emb(t))
        y_emb = self.label_mlp(self.label_emb(y))
        return t_emb + y_emb

    def forward(self, x, t, y):
        # x in [-1,1]
        cond = self.cond_vec(t, y)
        x = self.in_conv(x)

        x, s1 = self.down1(x, cond)
        x, s2 = self.down2(x, cond)
        x, s3 = self.down3(x, cond)

        x = self.mid1(x, cond)
        self._last_latents = x.detach()  # (B, C, H, W) at bottleneck
        x = self.mid2(x, cond)

        x = self.up3(x, s3, cond)
        x = self.up2(x, s2, cond)
        x = self.up1(x, s1, cond)

        x = self.out_norm(x)
        x = self.out_act(x)
        return self.out_conv(x)

    @torch.no_grad()
    def get_bottleneck_latents(self, x, y, t_zero: bool = True):
        # Run forward until mid1 to collect latents (no noise prediction needed)
        device = x.device
        B = x.size(0)
        t = torch.zeros(B, dtype=torch.long, device=device) if t_zero else torch.randint(0, 1, (B,), device=device)
        cond = self.cond_vec(t, y)
        h = self.in_conv(x)
        h, s1 = self.down1(h, cond)
        h, s2 = self.down2(h, cond)
        h, s3 = self.down3(h, cond)
        h = self.mid1(h, cond)
        return h  # (B, C, H, W)


# ------------------------------
# Training & sampling helpers
# ------------------------------
class DiffusionTrainer:
    def __init__(self, model: UNet, scheduler: LinearBetaSchedule, device):
        self.model = model
        self.sch = scheduler
        self.device = device

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sch.sqrt_alphas_cumprod[t].view(-1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.sch.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def p_sample_loop(self, shape: Tuple[int,int,int,int], y: torch.Tensor):
        B = shape[0]
        img = torch.randn(shape, device=self.device)
        for i in reversed(range(self.sch.T)):
            t = torch.full((B,), i, device=self.device, dtype=torch.long)
            eps_theta = self.model(img, t, y)
            beta_t = self.sch.betas[t].view(-1,1,1,1)
            sqrt_one_minus_alphas_cumprod_t = self.sch.sqrt_one_minus_alphas_cumprod[t].view(-1,1,1,1)
            sqrt_recip_alphas_t = self.sch.sqrt_recip_alphas[t].view(-1,1,1,1)
            # Estimate x0
            x0 = (img - sqrt_one_minus_alphas_cumprod_t * eps_theta) / self.sch.sqrt_alphas_cumprod[t].view(-1,1,1,1)
            # Predicted mean
            mean = sqrt_recip_alphas_t * (img - beta_t / sqrt_one_minus_alphas_cumprod_t * eps_theta)
            if i > 0:
                noise = torch.randn_like(img)
                var = self.sch.posterior_variance[t].view(-1,1,1,1)
                img = mean + torch.sqrt(var) * noise
            else:
                img = mean
        return img.clamp(-1, 1)


# ------------------------------
# Monitoring helpers
# ------------------------------
class Logger:
    def __init__(self, log_dir: str | Path, total_params: int):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / "train_log.csv"
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())
        self.max_cpu_mem = 0.0
        # init CSV with header
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["total_params", total_params])
            w.writerow(["metric", "value"])  # placeholder for totals after training
            w.writerow(["epoch", "epochs", "epoch_time_sec", "loss", "cpu_mem_mb", "cuda_mem_mb", "mps_mem_mb"])  # per-epoch header

    def _mem_snapshot(self, device: torch.device):
        rss = self.process.memory_info().rss / (1024**2)
        self.max_cpu_mem = max(self.max_cpu_mem, rss)
        cuda_mb = 0.0
        mps_mb = 0.0
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                cuda_mb = torch.cuda.max_memory_allocated() / (1024**2)
            except Exception:
                cuda_mb = 0.0
        if device.type == "mps" and torch.backends.mps.is_available():
            # PyTorch lacks precise MPS memory API; we approximate with current rss delta
            mps_mb = 0.0  # left as 0; macOS does not expose per-process VRAM easily
        return rss, cuda_mb, mps_mb

    def log_epoch(self, epoch: int, epochs: int, epoch_time: float, loss: float, device: torch.device):
        cpu_mb, cuda_mb, mps_mb = self._mem_snapshot(device)
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, epochs, f"{epoch_time:.3f}", f"{loss:.6f}", f"{cpu_mb:.1f}", f"{cuda_mb:.1f}", f"{mps_mb:.1f}"])

    def finalize(self):
        total_time = time.time() - self.start_time
        # write totals row
        with open(self.csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(["total_train_time_sec", f"{total_time:.3f}"])
            w.writerow(["max_cpu_mem_mb", f"{self.max_cpu_mem:.1f}"])
        return total_time, self.max_cpu_mem


# ------------------------------
# Main training
# ------------------------------
import argparse

def save_image_grid(rows: List[torch.Tensor], out_path: Path):
    # rows: list of tensors shape (5,1,64,64) in [-1,1]
    num_rows = len(rows)
    num_cols = 5
    cell = 64
    grid = Image.new("L", (num_cols*cell, num_rows*cell), color=255)
    for r, row in enumerate(rows):
        for c in range(num_cols):
            img = row[c]
            img = ((img + 1) / 2).clamp(0,1)
            img_np = (img.squeeze(0).cpu().numpy()*255).astype(np.uint8)
            pil = Image.fromarray(img_np, mode="L")
            grid.paste(pil, (c*cell, r*cell))
    grid.save(out_path)


def visualize_latents(model: UNet, loader: DataLoader, device: torch.device, out_dir: Path, max_samples: int = 1000):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            h = model.get_bottleneck_latents(x, y, t_zero=True)  # (B, C, H, W)
            h = F.adaptive_avg_pool2d(h, (4,4))
            h = h.flatten(1)
            feats.append(h.cpu())
            labels.append(y.cpu())
            if sum(len(b) for b in labels) >= max_samples:
                break
    if not feats:
        return
    X = torch.cat(feats, dim=0).numpy()
    Y = torch.cat(labels, dim=0).numpy()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_iter=1000, verbose=1)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    for cls in range(NUM_CLASSES):
        idx = Y == cls
        plt.scatter(X_tsne[idx,0], X_tsne[idx,1], s=8, label=CLASS_MAP[cls+1])
    plt.legend(markerscale=2)
    plt.title("Latent space t-SNE (bottleneck)")
    plt.tight_layout()
    plt.savefig(out_dir / "latents_tsne.png", dpi=200)
    plt.close()

    # UMAP (optional)
    if HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean")
        X_umap = reducer.fit_transform(X)
        plt.figure()
        for cls in range(NUM_CLASSES):
            idx = Y == cls
            plt.scatter(X_umap[idx,0], X_umap[idx,1], s=8, label=CLASS_MAP[cls+1])
        plt.legend(markerscale=2)
        plt.title("Latent space UMAP (bottleneck)")
        plt.tight_layout()
        plt.savefig(out_dir / "latents_umap.png", dpi=200)
        plt.close()
    else:
        print("[WARN] umap-learn not installed; skipping UMAP plot.")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    class Args: pass
    args = Args()
    args.data_dir = "../generateShapes/dataset_straight_centered/dataset_straight_centered_filled/dataset_straight_centered_filled-all_shapes_dir"
    args.epochs = 50
    args.batch_size = 64
    args.lr = 2e-4
    args.timesteps = 200
    args.log_dir = "../run_logs"
    args.num_workers = 4
    args.seed = 42
    args.samples_per_class = 1
    args.max_latent_samples = 1000

    seed_all(args.seed)
    device = get_device()
    print(f"Device: {device}")

    ds = ShapesDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    vis_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = UNet(in_ch=1, base_ch=64, ch_mults=(1,2,4), cond_dim=256, num_classes=NUM_CLASSES).to(device)
    sch = LinearBetaSchedule(timesteps=args.timesteps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_params = count_parameters(model)
    logger = Logger(args.log_dir, total_params)

    trainer = DiffusionTrainer(model, sch, device)

    # checkpoints for sampling: 10% steps
    checkpoints = sorted(set(max(1, int(round(args.epochs * p))) for p in np.linspace(0.1, 1.0, 10)))
    sample_rows: List[torch.Tensor] = []

    model.train()
    global_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            B = x.size(0)
            t = torch.randint(0, sch.T, (B,), device=device).long()
            x_noisy, noise = trainer.q_sample(x, t)
            pred_noise = model(x_noisy, t, y)
            loss = F.mse_loss(pred_noise, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        epoch_time = time.time() - t0
        avg_loss = epoch_loss / max(1, n_batches)
        logger.log_epoch(epoch, args.epochs, epoch_time, avg_loss, device)
        print(f"Epoch {epoch}/{args.epochs} - loss {avg_loss:.4f} - {epoch_time:.2f}s")

        # sampling checkpoint
        if epoch in checkpoints:
            model.eval()
            with torch.no_grad():
                imgs_per_class = []
                for cls in range(NUM_CLASSES):
                    y_cls = torch.full((args.samples_per_class,), cls, dtype=torch.long, device=device)
                    gen = trainer.p_sample_loop((args.samples_per_class, 1, 64, 64), y_cls)
                    # take the first image of each class for the grid row
                    imgs_per_class.append(gen[0:1])
                row = torch.cat(imgs_per_class, dim=0)  # (5,1,64,64)
                sample_rows.append(row.detach().cpu())
            model.train()

    total_time, max_cpu = logger.finalize()
    print(f"Total train time: {total_time:.1f}s | Max CPU RSS: {max_cpu:.1f} MB | Params: {total_params}")

    out_dir = Path(args.log_dir)
    # save grid
    if len(sample_rows) > 0:
        grid_path = out_dir / "samples_grid_10x5.png"
        save_image_grid(sample_rows, grid_path)
        print(f"Saved sample grid to {grid_path}")

    # save final per-class samples (one each)
    model.eval()
    with torch.no_grad():
        imgs_per_class = []
        for cls in range(NUM_CLASSES):
            y_cls = torch.full((1,), cls, dtype=torch.long, device=device)
            gen = trainer.p_sample_loop((1, 1, 64, 64), y_cls)
            imgs_per_class.append(gen)
        final_row = torch.cat(imgs_per_class, dim=0)
        save_image_grid([final_row.cpu()], out_dir / "final_samples_1x5.png")

    # visualize latents
    visualize_latents(model, vis_loader, device, out_dir, max_samples=args.max_latent_samples)


if __name__ == "__main__":
    main()
