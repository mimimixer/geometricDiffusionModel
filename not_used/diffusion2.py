from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from tqdm import tqdm

my_data = "./generateShapes/dataset_straight_centered/dataset_straight_centered_filled2500"


# --- Custom Dataset ---
class ShapesDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        for folder in os.listdir(root_dir):
            for fname in os.listdir(os.path.join(root_dir, folder)):
                self.paths.append(os.path.join(root_dir, folder, fname))
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),  # converts to [0, 1]
            lambda x: 2 * x - 1,  # scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("L")
        return self.transform(image)


# --- UNet Model ---
model = UNet2DModel(
    sample_size=64,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(64, 128),
    down_block_types=("DownBlock2D", "DownBlock2D"),
    up_block_types=("UpBlock2D", "UpBlock2D"),
)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# --- Training Loop ---
dataset = ShapesDataset(my_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
model.to(device)

for epoch in range(20):
    for batch in tqdm(dataloader):
        clean_images = batch.to(device)
        noise = torch.randn_like(clean_images)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps).sample
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}: loss={loss.item():.4f}")

# --- Inference ---
pipeline = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline.to(device)
images = pipeline(num_inference_steps=50).images

# Save one example
images[0].save("sample.png")