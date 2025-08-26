#!/usr/bin/env python3
"""
DCGAN for generating simple black-and-white geometric shapes (64x64).

Features
- Works with an unlabeled folder of images (any directory structure)
- Handles grayscale or RGB inputs (converted to 1 channel)
- Logs per-epoch training time and total wall time
- Saves sample grids every 20 epochs (exactly 5 images)
- Saves CSV log and model checkpoints

Usage
------
python dcgan_shapes.py \
  --data_dir /path/to/images \
  --epochs 200 \
  --batch_size 128 \
  --lr 0.0002 \
  --beta1 0.5 \
  --nz 64

Outputs are written under ./outputs by default.
"""

my_data = "../generateShapes/dataset_straight_centered/dataset_straight_centered_filled/"

import torch
import argparse
import os
import math
import time
from datetime import timedelta

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils


# ----------------------
# Model definitions
# ----------------------
class Generator(nn.Module):
    def __init__(self, nz=64, ngf=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # input Z: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=False),
            # state size: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=False),
            # state size: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=False),
            # state size: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=False),
            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # output: (nc) x 64 x 64 in [-1, 1]
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    def __init__(self, ndf=32, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            # input: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=False),
            # 32x32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=False),
            # 16x16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.Dropout(0.3),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=False),
            # 8x8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=False),
            # 4x4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.main(x).view(-1)


# ----------------------
# Utilities
# ----------------------

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias.data)


def make_dataloaders(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # [-1, 1]
    ])

    # ImageFolder expects subfolders; if your images are all in one folder, it's fine: they go under one class.
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(len(dataset))
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    return dl, len(dataset)


def save_samples(netG, fixed_noise, out_dir, epoch, device):
    netG.eval()
    with torch.no_grad():
        fake = netG(fixed_noise.to(device)).cpu()
    # Select exactly 5 images (fixed_noise should have 5 rows)
    grid = vutils.make_grid(fake, nrow=fake.size(0), normalize=True, value_range=(-1, 1))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"samples_epoch_{epoch:04d}.png")
    vutils.save_image(grid, path)
    return path


def write_csv_header(path):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('epoch, d_loss, g_loss, epoch_time_sec, total_elapsed_sec\n')


def append_csv_row(path, epoch, d_loss, g_loss, epoch_time, total_elapsed):
    with open(path, 'a') as f:
        f.write(f"epoch: {epoch}, d_loss:{d_loss:.3f}, g_loss:{g_loss:.3f}, epoch_time_sec:{epoch_time:.2f}, total_elapsed_sec:{total_elapsed:.2f}\n")


# ----------------------
# Training
# ----------------------

def train(args):
    #device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    device = torch.device("mps" if torch.backends.mps.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    dl, n_images = make_dataloaders(args.data_dir, args.batch_size)
    print(f"Loaded {n_images} images")

    nc = 1
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=nc).to(device)
    netD = Discriminator(ndf=args.ndf, nc=nc).to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion = nn.BCEWithLogitsLoss()

    fixed_noise = torch.randn(5, args.nz, 1, 1)  # for periodic snapshots (exactly 5)

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    real_label = 0.9
    fake_label = 0.0

    os.makedirs(args.out_dir, exist_ok=True)
    log_csv = os.path.join(args.out_dir, 'training_log.csv')
    write_csv_header(log_csv)

    start_time = time.time()
    total_elapsed = 0.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        netG.train()
        netD.train()

        running_d, running_g = 0.0, 0.0
        num_batches = 0

        for real, _ in dl:
            num_batches += 1
            real = real.to(device)
            b_size = real.size(0)

            # -------------------
            # Update D: maximize log(D(x)) + log(1 - D(G(z)))
            # -------------------
            netD.zero_grad(set_to_none=True)

            # Real
            label = torch.full((b_size,), real_label, device=device)
            out_real = netD(real)
            errD_real = criterion(out_real, label)

            # Fake
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise).detach()
            #label.fill_(fake_label)
            label_fake = torch.full((b_size,), fake_label, device=device)
            out_fake = netD(fake)
            errD_fake = criterion(out_fake, label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optimizerD.step()

            # -------------------
            # Update G: maximize log(D(G(z)))
            # -------------------
            netG.zero_grad(set_to_none=True)
            #label.fill_(real_label)
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label_gen = torch.full((b_size,), real_label, device=device)
            out = netD(fake)
            errG = criterion(out, label_gen)
            errG.backward()
            optimizerG.step()

            running_d += errD.item()
            running_g += errG.item()

        epoch_time = time.time() - epoch_start
        total_elapsed = time.time() - start_time
        d_loss = running_d / max(1, num_batches)
        g_loss = running_g / max(1, num_batches)

        append_csv_row(log_csv, epoch, d_loss, g_loss, epoch_time, total_elapsed)

        # Save checkpoints
        if epoch % args.ckpt_every == 0 or epoch == args.epochs:
            torch.save({'epoch': epoch, 'netG': netG.state_dict()}, os.path.join(args.out_dir, f'netG_epoch_{epoch:04d}.pt'))
            torch.save({'epoch': epoch, 'netD': netD.state_dict()}, os.path.join(args.out_dir, f'netD_epoch_{epoch:04d}.pt'))

        # Visual samples every 20 epochs
        if epoch % 20 == 0 or epoch == 1:
            sample_path = save_samples(netG, fixed_noise, args.out_dir, epoch, device)
            print(f"[Epoch {epoch}] Samples saved to: {sample_path}")

        print(f"Epoch {epoch:4d}/{args.epochs} | D_loss: {d_loss:.4f} | G_loss: {g_loss:.4f} | epoch: {timedelta(seconds=round(epoch_time))} | total: {timedelta(seconds=round(total_elapsed))}")

    total_time = time.time() - start_time
    print("\nTraining finished.")
    print(f"Total training time: {timedelta(seconds=round(total_time))}")


# ----------------------
# CLI
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="DCGAN for black-and-white square shapes (64x64)")
    #parser.add_argument('--data_dir', type=str, required=True, help='Root folder containing images (ImageFolder-style).')
    parser.add_argument('--data_dir', type=str, default = my_data, help='Root folder containing images (ImageFolder-style).')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Directory to save outputs (samples, logs, checkpoints).')
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--batch_size', type=int, default=128)#128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--nz', type=int, default=64, help='Latent vector size')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=32)
    parser.add_argument('--ckpt_every', type=int, default=50)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train(args)


if __name__ == '__main__':
    main()
