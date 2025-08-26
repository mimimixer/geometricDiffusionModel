import os
import time
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.io import imread_collection, imsave
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.pyplot as plt
from random import randrange

# --- Load dataset ---
data_dir = "../generateShapes/dataset_rotated_cropped/dataset_filled_rotated1/circle"
shape_name = os.path.basename(data_dir)
print(shape_name)
img_size = (64, 64)   # resize all images for consistency
n_components = 20              # number of eigenshapes
output_dir = f"results_PCA_{shape_name}" # where to save results
os.makedirs(output_dir, exist_ok=True)

start_time = time.time()
images = []

for file in os.listdir(data_dir):
    if file.endswith(".png") or file.endswith(".jpg"):
        img = imread_collection([os.path.join(data_dir, file)])[0]
        if img.shape[-1] == 4:
            img = img[..., :3]
        img_gray = rgb2gray(img)
        img_resized = resize(img_gray, img_size)
        images.append(img_resized.flatten())


images = np.array(images)
print("Dataset shape:", images.shape)  # (2500, 4096) if 64x64

# --- PCA ---
#scaler = StandardScaler(with_mean=True, with_std=False)  # mean-center only
#X_centered = scaler.fit_transform(images)

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
X_pca = pca.fit_transform(images)#X_centered)


# --- Show some eigenshapes ---
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.ravel()):
    ax.imshow(pca.components_[i].reshape(img_size), cmap="gray")
    ax.set_title(f"Eigenshape {i+1}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"eigenshapes_for_{shape_name}.png"))
plt.show()


# --- Save one reconstruction example ---
idx = randrange(images.shape[0])  # pick first image
proj = X_pca[idx]
recon = pca.inverse_transform(proj)
recon_img = recon.reshape(img_size)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(images[idx].reshape(img_size), cmap="gray")
ax1.set_title("Original")
ax2.imshow(recon_img, cmap="gray")
ax2.set_title("Reconstruction")
plt.savefig(os.path.join(output_dir, f"reconstruction_{shape_name}.png"))
plt.show()


print("Eigenvectors shape:", pca.components_.shape)
print("Projection (faceprints) shape:", X_pca.shape)


# --- Logging ---
end_time = time.time()
elapsed_time = end_time - start_time

logfile = os.path.join(output_dir, "log.txt")
with open(logfile, "w") as f:
    f.write(f"Dataset shape: {images.shape}\n")
    f.write(f"Eigenvector shape: {pca.components_.shape}\n")
    f.write(f"Projections shape: {X_pca.shape}\n")
    f.write(f"Time elapsed: {elapsed_time:.2f} seconds\n")

print("Done! Results saved in:", output_dir)