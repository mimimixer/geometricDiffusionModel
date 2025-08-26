from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import re

class ShapeDataset(Dataset):
    def __init__(self, folder_path, transform=None, treshold=True, return_labels = True):
        self.paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
        self.transform = transform
        self.treshold = treshold
        self.return_labels = return_labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(self.paths[idx]).convert('L')  # Graustufen
        if self.transform:
            image = self.transform(image)
        if self.treshold:
            image = (image > 0.5).float()
        if self.return_labels:
            # Label aus Dateiname extrahieren (erste Zahl im Namen)
            filename = os.path.basename(img_path)
            match = re.search(r'\d+', filename)
            label = int(match.group()) if match else 0
            return image, label
        else:
            return image

