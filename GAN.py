import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import number
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import umap
from sklearn.manifold import TSNE
import psutil

# -----------------------------
# 0. Presets
# -----------------------------

latent_dim = 100
number_of_image_channels=1
num_classes=5
feature_maps=64
epochs = 150
size_of_batch = 64
size_of_image = 64
log_interval = epochs // 10  # visualize every 20 epochs
criterion = nn.BCELoss()
lr = 0.0001
beta1 = 0.5
number_of_samples_for_visualisation = 2500
#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cpu")
print(f"Using device: {device}")
my_data_folder = "/Users/sunny/Documents/aiai2/geoDiff1/generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"
my_data = os.path.basename(my_data_folder)

# -----------------------------
# 1. Dataset
# -----------------------------
class ShapesDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.folder_path, fname)
        img = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        label = int(fname[0]) - 1  # 0-indexed: 0=triangle ... 4=ellipse
        return img, label

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = ShapesDataset(folder_path=my_data_folder, transform=transform)
dataloader = DataLoader(dataset, batch_size=size_of_batch, shuffle=True, num_workers=0)

# -----------------------------
# 2. DCGAN models
# -----------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=latent_dim, img_channels=number_of_image_channels, num_classes=num_classes, feature_maps=feature_maps):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # input = z_dim + num_classes
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + num_classes, feature_maps * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_maps, img_channels, 4, 4, 0, bias=False),
            #nn.Sigmoid()  # since images are B/W in [0,1]
            nn.Tanh()
        )


    def forward(self, z, labels):
        z = z.view(z.size(0), z.size(1), 1, 1)
        # labels: [B] integers
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()   # [B, num_classes]
        one_hot = one_hot.unsqueeze(2).unsqueeze(3)  # [B, num_classes, 1, 1]
        # Concatenate noise + one-hot
        x = torch.cat([z, one_hot], dim=1)       # [B, z_dim+num_classes, 1, 1]
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=number_of_image_channels, num_classes=num_classes, feature_maps=feature_maps):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        # input channels = image channels + num_classes
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, feature_maps, 4, 4, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.2),

            nn.Conv2d(feature_maps * 2, 1, 8, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        one_hot = one_hot.unsqueeze(2).unsqueeze(3)  # [B, num_classes, 1, 1]
        one_hot = one_hot.expand(-1, -1, img.size(2), img.size(3))  # [B, num_classes, H, W]
        # Concatenate image + label map
        x = torch.cat([img, one_hot], dim=1)  # [B, 1+num_classes, H, W]
        return self.net(x).view(-1, 1)

#generator = Generator(latent_dim, num_classes).to(device)
#discriminator = Discriminator(num_classes).to(device)
generator = Generator(latent_dim=latent_dim, img_channels=number_of_image_channels, num_classes=num_classes, feature_maps=feature_maps).to(device)
discriminator = Discriminator(img_channels=number_of_image_channels, num_classes=num_classes, feature_maps=feature_maps).to(device)

# -----------------------------
# 3. Optimizers and loss
# -----------------------------
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*2, betas=(beta1, 0.999))

# -----------------------------
# 4. Training
# -----------------------------
generated_epoch_images = []  # each row per logged epoch
log = {"epoch":[], "epoch_time":[], "d_loss":[], "g_loss":[], "cpu_memory_MB":[]}
max_cpu_mem_used = 0
min_cpu_mem_used = 0

num_params_G = sum(p.numel() for p in generator.parameters())
num_params_D = sum(p.numel() for p in discriminator.parameters())

start_total = time.time()
for epoch in range(1, epochs+1):
    start = time.time()
    for imgs, labels in dataloader:
        image_size_cur = imgs.size(0)
        real_imgs = imgs.to(device)
        labels = labels.to(device)

        # ---- Train Discriminator ----
        optimizer_D.zero_grad()
        real_targets = torch.full((image_size_cur, 1), 0.9, device=device) #discriminaror too strong
        fake_targets = torch.zeros(image_size_cur, 1, device=device)

        outputs_real = discriminator(real_imgs, labels)
        loss_real = criterion(outputs_real, real_targets)

        z = torch.randn(image_size_cur, latent_dim, device=device)
        random_labels = torch.randint(0, num_classes, (image_size_cur,), device=device)
        fake_imgs = generator(z, random_labels)
        outputs_fake = discriminator(fake_imgs.detach(), random_labels)
        loss_fake = criterion(outputs_fake, fake_targets)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # ---- Train Generator ----
        optimizer_G.zero_grad()
        z = torch.randn(image_size_cur, latent_dim, device=device)
        random_labels = torch.randint(0, num_classes, (image_size_cur,), device=device)
        fake_imgs = generator(z, random_labels)
        outputs = discriminator(fake_imgs, random_labels)
        loss_G = criterion(outputs, real_targets)
        loss_G.backward()
        optimizer_G.step()

    epoch_time = time.time() - start

    log["epoch"].append(epoch)
    log["epoch_time"].append(epoch_time)
    log["d_loss"].append(loss_D.item())
    log["g_loss"].append(loss_G.item())
    memory_MB = psutil.Process().memory_info().rss / 1024 ** 2
    if memory_MB > max_cpu_mem_used:
        max_cpu_mem_used = memory_MB
    if min_cpu_mem_used > memory_MB:
        min_cpu_mem_used = memory_MB
    log["cpu_memory_MB"].append(memory_MB)
    print(f"Epoch [{epoch}/{epochs}] | Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | Time: {epoch_time:.2f}s | Memory_used: {memory_MB:.2f}MB")

    # ---- Periodic visualization ----
    if epoch % log_interval == 0:
        generator.eval()
        with torch.no_grad():
            row_imgs = []
            for lbl in range(num_classes):
                z = torch.randn(1, latent_dim, 1, 1, device=device)
                lbl_tensor = torch.tensor([lbl], device=device, dtype=torch.long)
                gen_imgs = generator(z, lbl_tensor)
                #concat_label = torch.cat([gen_imgs[i] for i in range(10)], dim=2)
                #row_imgs.append(concat_label)
                row_imgs.append(gen_imgs.squeeze(0))
            final_row = torch.cat(row_imgs, dim=2)
            generated_epoch_images.append(final_row.cpu())
        generator.train()

torch.save(generator.state_dict(), f"generator_{my_data}_e{epochs}.pt")
torch.save(discriminator.state_dict(), f"discriminator_{my_data}_e{epochs}.pt")

end_total = time.time()
logfile_path = f"training_log_{my_data}.txt"
with open(logfile_path, "w") as f:
    f.write(f"Number of parameters - Generator: {num_params_G}, Discriminator: {num_params_D}\n")
    f.write(f"Maximum cpu-memory used: {max_cpu_mem_used:.2f} MB, Minimum cpu-memory used: {min_cpu_mem_used:.2f} MB\n")
    f.write(f"Total training time: {end_total - start_total:.2f} seconds\n\n")
    f.write("Epoch\tEpoch_time(s)\tD_loss\tG_loss\tcpu_memory_used\n")
    for i, epoch in enumerate(log["epoch"]):
        f.write(f"{epoch:<5}\t{log['epoch_time'][i]:>8.2f}\t{log['d_loss'][i]:>8.4f}\t{log['g_loss'][i]:>8.4f}\t{log['cpu_memory_MB'][i]:>8.2f}\n")

# ---- Save final grid image ----
grid_img = torch.cat(generated_epoch_images, dim=1)  # rows = epochs, columns = labels
plt.figure(figsize=(20, 20))
plt.imshow(grid_img.squeeze(), cmap='gray')
plt.axis('off')
plt.title("Generated images evolution per epoch and label")
plt.savefig(f"training_images_for_{my_data}.png", dpi=150, bbox_inches='tight')
plt.close()
"""
if generated_epoch_images:
    # Konkateniere alle Zeilen vertikal
    grid_img = torch.cat(generated_epoch_images, dim=2)  # dim=2 für vertikale Konkatenation
    grid_img = grid_img.squeeze(0)  # Entferne die Batch-Dimension

    plt.figure(figsize=(10, 20))  # 5 Spalten x 10 Zeilen
    plt.imshow(grid_img.squeeze(), cmap='gray')
    plt.axis('off')
    plt.title(f"Generated images for {my_data}: Rows=Epochs, Columns=Labels (0-4)")
    plt.savefig(f"training_images_for_{my_data}.png", dpi=150, bbox_inches='tight')
    plt.close()
"""
# ---- Latent space visualization ----
generator.eval()
num_samples = number_of_samples_for_visualisation
latent_vectors = torch.randn(num_samples, latent_dim, device=device)
labels = torch.randint(0, num_classes, (num_samples,), device=device)
with torch.no_grad():
    gen_imgs = generator(latent_vectors, labels).view(num_samples, -1).cpu().numpy()

# UMAP
reducer = umap.UMAP()
embedding_umap = reducer.fit_transform(gen_imgs)
plt.figure()
for lbl in range(num_classes):
    plt.scatter(embedding_umap[labels.cpu()==lbl,0], embedding_umap[labels.cpu()==lbl,1], label=str(lbl))
plt.legend()
plt.title(f"UMAP of generated images_for_{my_data}")
plt.savefig(f"visualisation_umap_{my_data}.png")
plt.close()

# tSNE
embedding_tsne = TSNE(n_components=2).fit_transform(gen_imgs)
plt.figure()
for lbl in range(num_classes):
    plt.scatter(embedding_tsne[labels.cpu()==lbl,0], embedding_tsne[labels.cpu()==lbl,1], label=str(lbl))
plt.legend()
plt.title(f"tSNE of generated images for_{my_data}")
plt.savefig(f"visualisation_tsne_{my_data}.png")
plt.close()

print("Training complete. Logs, grid images, and latent visualizations saved.")