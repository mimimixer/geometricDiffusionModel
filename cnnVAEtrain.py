import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import time
import os

import cnnVAEmodel
import VAEdataloader
from cnnVAEgenerator import generate_images
from visualizeVAE import visualize_tsne, visualize_umap, get_latents_from_vae

number_of_training_epochs = 150
batch_size = 32
number_of_generated_images = 10

latent_dimension = cnnVAEmodel.latent_dimension
shape_of_image = cnnVAEmodel.shape_of_image

# load dataset
folder_path = "generateShapes/dataset_straight_centered/all_shapes_dir-dataset_straight_centered_filled2500"
kind_of_shapes = os.path.basename(folder_path)
current_time = time.strftime("%d%H%M")
trained_model_name = (f"cnnVAEmodelTrained{current_time}-{kind_of_shapes}-{latent_dimension}e{number_of_training_epochs}-b{cnnVAEmodel.beta}")
log_file = f"training_log_{trained_model_name}.txt"
with open(log_file, "w") as f:
    f.write("Epoch,TrainLoss,TestLoss,TimePerEpoch(sec)\n")

total_start = time.time()

transform = transforms.Compose([
    transforms.Resize((shape_of_image, shape_of_image)),
    transforms.ToTensor()
])
#transform = transforms.ToTensor()
#train_data = VAEdataloader.ShapeDataset(folder_path, transform=transform)
dataset_wih_labels = VAEdataloader.ShapeDataset(folder_path, transform=transform)
print(folder_path)
dataset = [dataset_wih_labels[i][0] for i in range(len(dataset_wih_labels))]


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
print(f"using device: {device} for cnnVAE training")

model = cnnVAEmodel.cnnVAE(latent_dim=latent_dimension).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params}")

images = next(iter(train_loader))
"""
print("Shape: ",images.shape, "Dtype: ", images.dtype,
        "Min value: ", images.min().item(),"Max value: ", images.max().item())
plt.imshow(images[0].squeeze(), cmap='gray')
plt.title("Sample from Dataset")
plt.show()
"""
# Training
for epoch in range(number_of_training_epochs):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    for images in train_loader:
        images = images.to(device).float()
        recon, mu, logvar = model(images)
        loss = cnnVAEmodel.cnn_vae_loss(recon, images, mu, logvar)

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
            recon, mu, logvar = model(images)
            loss = cnnVAEmodel.cnn_vae_loss(recon, images, mu, logvar)
            test_loss += loss.item()

    print(f"Epoch {epoch+1}/{number_of_training_epochs}, "
            f"Train Loss: {total_loss / len(train_loader.dataset):.4f}, "
            f"Test Loss: {test_loss / len(test_loader.dataset):.4f}), "
            f"Time: {epoch_time:.2f} sec")
            #f"Mean: {model.encoder.fc_mu}")

    with open(log_file, "a") as f:
        f.write(f"{epoch+1},"
                f"{total_loss / len(train_loader.dataset):.4f},"
                f"{test_loss / len(test_loader.dataset):.4f},"
                f"{epoch_time:.2f}\n")

total_time = time.time() - total_start
print(f"Total training time: {total_time:.2f} sec")

with open(log_file, "a") as f:
    f.write("\n")
    f.write(f"Device: {device}\n")
    f.write(f"Total parameters: {num_params}\n")
    f.write(f"Total training time: {total_time:.2f} sec\n")

torch.save(model.state_dict(), f"{trained_model_name}.pt")
generate_images(model=model, latent_dim=latent_dimension, n=number_of_generated_images, name=trained_model_name)
zs, labels = get_latents_from_vae(model, dataset_wih_labels)

# Visualisierung
save_path = f"visualisation"
visualize_tsne(zs, labels, f"{save_path}_tSNE_{trained_model_name}")
visualize_umap(zs, labels, f"{save_path}_UMAP_{trained_model_name}")