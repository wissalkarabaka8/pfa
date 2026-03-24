import os
from PIL import Image
import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        for file in os.listdir(root_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                self.image_paths.append(os.path.join(root_dir, file))

    def __len__(self):
        return len(self.image_paths)

    def nombre_images(self):
        print("Nombre total d'images dans le dataset:", len(self.image_paths))

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
    def taille_image(self):
        sample = Image.open(self.image_paths[random.randint(0, len(self.image_paths)-1)]).convert("RGB")
        print("size of a random image :", sample.size)

def return_data(folder_path, batch_size=32, num_workers=1, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    dataset = MyDataset(folder_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader

if __name__ == '__main__':
    folder_path = r'folder_path' # Remplace par le chemin de ton dossier d'images

    # Crée le DataLoader avec ta fonction return_data
    loader = return_data(folder_path, batch_size=32, num_workers=1, image_size=64)

    # Récupère le premier batch d'images
    images = next(iter(loader))
    
    # Vérifie les images
    print("Shape du batch:", images.shape)
