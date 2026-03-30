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
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                self.image_paths.append(os.path.join(root_dir, file))

        # ✅ Avertissement si dossier vide
        if len(self.image_paths) == 0:
            print(f"[WARNING] Aucune image trouvée dans : {root_dir}")
        else:
            print(f"[Dataset] {len(self.image_paths)} images chargées depuis : {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def nombre_images(self):
        print("Nombre total d'images dans le dataset:", len(self.image_paths))

    def taille_image(self):
        if len(self.image_paths) == 0:
            print("[WARNING] Aucune image disponible.")
            return
        sample = Image.open(
            self.image_paths[random.randint(0, len(self.image_paths) - 1)]
        ).convert("RGB")
        print("Taille d'une image aléatoire :", sample.size)


def return_data(folder_path, batch_size=32, shuffle=True,
                num_workers=0, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = MyDataset(folder_path, transform=transform)

    # ✅ Dossier vide → erreur claire
    if len(dataset) == 0:
        raise ValueError(f"❌ Dossier vide ou sans images valides : {folder_path}")

    # ✅ Ne pas dépasser le nombre d'images disponibles
    actual_batch = min(batch_size, len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=actual_batch,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    return loader


if __name__ == '__main__':
    folder_path = r'./data/train'
    loader = return_data(folder_path, batch_size=32)
    images = next(iter(loader))
    print("Shape du batch:", images.shape)