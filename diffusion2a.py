from cv2 import transform
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


# --- Custom Dataset ---
class ShapesDataset(Dataset):
    def __init__(self, file):
        self.paths = []
        self.paths.append(file)
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),  # converts to [0, 1]
            lambda x: 2 * x - 1,  # scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert('L')
        return image, 'mein label'
        #return self.transform(image)

# --- Training Loop ---
dataset = ShapesDataset("generateShapes/dataset rotated cropped/dataset_outlined_rotated1/triangle/triangle_000.png")
dataloader = DataLoader(dataset)

TIMESTEPS = 19

# 3. Noise Schedule
betas = torch.linspace(1e-4, 0.02, TIMESTEPS)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

def add_noise(start_img, t):
    noise = torch.randn_like(start_img)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    return sqrt_alphas_cumprod * start_img + sqrt_one_minus * noise
    #return transform(img)

print(len(dataloader))
figure = plt.figure(figsize=(64, 64))
img, label = dataloader.dataset[0]
figure.add_subplot(5, 5, 1)
plt.title(label)
plt.imshow(img, cmap = 'gray')

tensor_transformation_function = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),  # converts to [0, 1]
            lambda x: 2 * x - 1,  # scale to [-1, 1]
        ])
tensor = tensor_transformation_function(img)
for i in range(TIMESTEPS):
    figure.add_subplot(5, 5, i+2)
    t = torch.randint(0, 1, (1,))
    x_noisy = add_noise(tensor, t)
    tensor = x_noisy
    print(x_noisy.type())
    print(x_noisy.shape)
    noisy_image = T.ToPILImage()(x_noisy[-1,:,:,:])
    plt.imshow(noisy_image, cmap = 'gray')

plt.show()
plt.imsave('output_11.png', tensor, cmap = 'gray')