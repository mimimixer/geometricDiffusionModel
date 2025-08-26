import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import time
import os

import VAEmodel
import VAEdataloader

number_of_training_epochs = 150
batch_size = 16

latent_dimension = VAEmodel.latent_dimension
shape_of_image = VAEmodel.shape_of_image
current_time = time.strftime("%d%H%M")

# load dataset
folder_path = "generateShapes/dataset_straight_centered/dataset_straigt_centered_outlined/5ellipse"
kind_of_shapes = os.path.basename(folder_path)

transform = transforms.Compose([
    transforms.Resize((shape_of_image, shape_of_image)),
    transforms.ToTensor()
])
#transform = transforms.ToTensor()
#train_data = VAEdataloader.ShapeDataset(folder_path, transform=transform)
dataset = VAEdataloader.ShapeDataset(folder_path, transform=transform)

# Split: 80% Train, 20% Test
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_data, test_data = random_split(dataset, [train_size, test_size])

#train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Modell, Optimierer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#device = torch.device("cpu")
print(f"using device: {device} for vae training")

model = VAEmodel.VAE(latent_dim=latent_dimension).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#images = next(iter(train_loader))
#print("Shape: ",images.shape, "Dtype: ", images.dtype,
#        "Min value: ", images.min().item(),"Max value: ", images.max().item())
#plt.imshow(images[0].squeeze(), cmap='gray')
#plt.title("Sample from Dataset")
#plt.show()

total_start = time.time()

# Training
for epoch in range(number_of_training_epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    for images in train_loader:
        images = images.to(device).float()
        x_recon, mu, logvar = model(images)
        loss = VAEmodel.vae_loss(x_recon, images, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    epoch_time = time.time() - epoch_start

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device).float()
            x_recon, mu, logvar = model(images)
            loss = VAEmodel.vae_loss(x_recon, images, mu, logvar)
            test_loss += loss.item()

    print(f"Epoch {epoch+1}/{number_of_training_epochs}, "
            f"Train Loss: {total_loss / len(train_loader.dataset):.4f}, "
            f"Test Loss: {test_loss / len(test_loader.dataset):.4f}), "
            f"Time: {epoch_time:.2f} sec")
            #f"Mean: {model.encoder.fc_mu}")

total_time = time.time() - total_start

print(f"Total training time: {total_time:.0f} sec")

trained_model_name = (f"VAEmodelTrained{current_time}-{kind_of_shapes}-{latent_dimension}e{number_of_training_epochs}b{VAEmodel.beta}L{total_loss / len(train_loader.dataset):.0f}.pt")
torch.save(model.state_dict(), trained_model_name)