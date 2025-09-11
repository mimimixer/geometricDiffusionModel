import os, time, psutil, torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
from PIL import Image
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.manifold import TSNE
import umap
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from pathlib import Path
import time


print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("Torch version:", torch.__version__)


# --- SETTINGS ---
my_data = "/Users/sunny/Documents/aiai2/geoDiff1/generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"
timestp = time.strftime("%Y%m%d%H%M%S")
number_of_epochs = 25
batch_size = 16
number_of_images_per_label = 2500
log_file = f"training_log_{number_of_images_per_label}_e{number_of_epochs}_ro{timestp}.txt"
inference_plot = f"all_samples_{number_of_images_per_label}_e{number_of_epochs}_ro{timestp}.png"
modelname = f"model_500_{number_of_images_per_label}_e{number_of_epochs}_ro{timestp}.pt"
save_tsne_name = f"diffsion_tsne_{number_of_images_per_label}_e{number_of_epochs}_ro{timestp}.png"
save_umap_name = f"diffsion_umap_{number_of_images_per_label}_e{number_of_epochs}_ro{timestp}.png"
number_of_labels = 5
number_of_out_channels = 1 #greyscale, for RGB its 3
CLASS_MAP = {1: "triangle", 2: "rectangle", 3: "square", 4: "circle", 5: "ellipse"}
IMG_SIZE = 64
num_train_timesteps = 250
lr = 1e-4

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = "cpu"
print("Using device:", device)


# read dataset class
class ShapesDataset(Dataset):
    def __init__(self, root: str | Path, img_size: int = IMG_SIZE):
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
        self.img_size = img_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        label = self.labels[idx]
        #image
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.NEAREST)
        arr = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        x = x * 2 - 1  # [-1,1]
        # one-hot label map
        one_hot = torch.zeros(len(CLASS_MAP), dtype=torch.float32)
        one_hot[label] = 1
        cond_map = one_hot.view(len(CLASS_MAP), 1, 1).expand(len(CLASS_MAP), self.img_size, self.img_size)
        #put together
        x = torch.cat([x, cond_map], dim=0)
        return x, label

# model definition
model = UNet2DModel(
    sample_size=64,
    in_channels=number_of_out_channels+number_of_labels,
    out_channels=number_of_out_channels,
    layers_per_block=1,
    block_out_channels=(64, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
)

# scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule="squaredcos_cap_v2")

#inferece with label
def generate_with_label(unet, scheduler, label, num_inference_steps, device, img_size=64):
    # create conditioned input
    one_hot = torch.zeros(number_of_labels, dtype=torch.float32, device=device)
    one_hot[label] = 1
    cond_map = one_hot.view(number_of_labels, 1, 1).expand(number_of_labels, img_size, img_size)

    # start from Gaussian noise
    x = torch.randn(1, 1, img_size, img_size, device=device)
    x = torch.cat([x, cond_map.unsqueeze(0)], dim=1)  # (1, 6, H, W)

    scheduler.set_timesteps(num_inference_steps)

    # denoise loop
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = unet(x, t).sample
        x = scheduler.step(noise_pred, t, x).prev_sample

    # final image is the *first channel* (the grayscale image)
    img = x[0, 0].cpu().clamp(-1, 1)
    img = ((img + 1) / 2 * 255).numpy().astype("uint8")
    return Image.fromarray(img)

# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
model.to(device)

"""
# print & save architecture
input_images = (batch_size, number_of_out_channels+number_of_labels, IMG_SIZE, IMG_SIZE) # (B, C, H, W)
dummy_timestep = torch.tensor([16])
arch_str = summary(
    model,
    input_size=(input_images, dummy_timestep),
    depth=2,     # how deep to expand layers (higher = more detail)
    col_names=("input_size", "output_size", "num_params"),
    verbose=0
)
# Filter lines: only show layers with params or meaningful operations
arch_summary = []
arch_summary.append(
    f"Layer (type:depth-idx)                        Input Shape               Output Shape              Param #")
for layer in arch_str.summary_list:
    if layer.input_size != [] and layer.num_params >0:
        arch_summary.append(f"{layer.get_layer_name(True, 1):<40} {str(layer.input_size):<25} {str(layer.output_size):<25} {layer.num_params}")

# Convert to string table
#header = "="*100 + "\n"
#header += f"{'Layer (type:depth-idx)':<40} {'Input Shape':<25} {'Output Shape':<25} {'Param #'}\n"
#header += "="*100 + "\n"

#table_str = header + "\n".join(arch_summary) + "\n" + "="*100

with open(f"architektur{timestp}.txt", "w") as f:
    for i in arch_summary:
        print(f"{i}\n")
        f.write(f"{i}\n")
"""
# Logging setup
total_params = sum(p.numel() for p in model.parameters())
start_time = time.time()
log_lines = [f"Total parameters: {total_params}\nTotal epochs: {number_of_epochs}\n"]

# when inference for visualisation?
def should_visualize(epoch: int) -> bool:
    if epoch <= 3:
        return True                   # first 3 epochs
    elif 4 <= epoch <= 11:
        return (epoch % 2 == 1)       # every 2nd epoch
    elif 12 <= epoch <= 20:
        return (epoch % 3 == 2)       # every 3rd epoch
    elif 20 < epoch:
        return (epoch % 5 == 0)       # every 5th epoch
    else:
        return False

# import data
dataset = ShapesDataset(my_data)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# training loop
epoch_loss = 0
losses = []
samples = []
for epoch in range(1, number_of_epochs + 1):
    for batch, labels in tqdm(dataloader):
        # get images, noise,and trainingsteps, put together with scheduler
        clean_images = batch.to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch.size(0),), device=device
        ).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        #let model predict noise and calculate loss
        noise_pred = model(noisy_images, timesteps).sample
        loss = F.mse_loss(noise_pred[:,0:1], noise[:,0:1])
        #backpropagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch}: loss={epoch_loss:.4f}")

    # visualizing after 10% epochs: infere what it has learned so far
    if should_visualize(epoch): #epoch % max(1, number_of_epochs // 10) == 0:
        #pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
        #pipeline.to(device)
        label_images = []
        for label in range(number_of_labels):
            #images = pipeline(num_inference_steps=num_train_timesteps//5).images
            images = generate_with_label(model, noise_scheduler, label, num_train_timesteps // 5, device)
            label_images.append(images)#images[0])
            images.show()#images[0].show()
        samples.append((epoch, label_images))

# save model
torch.save(model.state_dict(), modelname)

# write logs
end_time = time.time()
total_time = end_time - start_time
memory_used = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
with open(log_file, "w") as f:
    f.write(f"Total parameters: {total_params}\n")
    f.write(f"Total epochs: {number_of_epochs}\n")
    f.write(f"Training time (s): {total_time:.2f}\n")
    f.write(f"Memory used (MB): {memory_used:.2f}\n")
    for i, l in enumerate(losses, 1):
        f.write(f"Epoch {i}: loss={l:.4f}\n")

# print samples of generated images during training
rows = len(samples)
cols = number_of_labels
fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
for r, (epoch, imgs) in enumerate(samples):
    for c, img in enumerate(imgs):
        axes[r][c].imshow(img, cmap="gray")
        axes[r][c].axis("off")
        if r == 0:
            axes[r][c].set_title(CLASS_MAP[c+1], fontsize = 14)#(f"Label {c+1}")
    #axes[r][0].set_ylabel(f"Epoch {epoch}", rotation=90, size="large", labelpad=20)
    fig.text(
        0.02,  # x-position (left margin)
        (rows - r - 0.5) / rows,  # y-position (centered on row)
        f"Epoch {epoch}",
        va="center", ha="left",
        fontsize=16, weight="bold"
    )

plt.tight_layout(rect=[0.05, 0, 1, 1])  # leave space for epoch text on left
#plt.tight_layout()
plt.savefig(inference_plot, dpi=150)
plt.close()

# latent space visualization
def get_latents(model, dataloader, device):
    latents, labels = [], []
    with torch.no_grad():
        for imgs, labs in dataloader:
            imgs = imgs.to(device)
            # Extract features from the first conv layer
            feats = model.conv_in(imgs)  # shape: (batch, 64, H, W)
            # Flatten to vectors
            latents.append(feats.mean(dim=[2, 3]).cpu())  # global average pooling
            labels.append(labs)
    return torch.cat(latents), torch.cat(labels)

latents, labels = get_latents(model, dataloader, device)
latents += 1e-6 * torch.randn_like(latents)

# t-SNE
tsne_emb = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto").fit_transform(latents)
plt.figure(figsize=(6, 6))
tsne_scatter = plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=labels, cmap="tab10", s=5)
handles, _ = tsne_scatter.legend_elements()
plt.legend(handles, [CLASS_MAP[i+1] for i in range(len(CLASS_MAP))], title="Labels")
plt.title("t-SNE diffusion Latent Space")
plt.savefig(save_tsne_name)
plt.close()

# UMAP
umap_emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean").fit_transform(latents)
plt.figure(figsize=(6, 6))
umap_scatter = plt.scatter(umap_emb[:, 0], umap_emb[:, 1], c=labels, cmap="tab10", s=5)
handles, _ = umap_scatter.legend_elements()
plt.legend(handles, [CLASS_MAP[i+1] for i in range(len(CLASS_MAP))], title="Labels")
plt.title("UMAP diffusion Latent Space")
plt.savefig(save_umap_name)
plt.close()

print("✅ Training finished. Logs saved.")