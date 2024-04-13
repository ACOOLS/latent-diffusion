import os
import numpy as np
import albumentations
from torch.utils.data import Dataset
import random

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path



class CustomDataset(Dataset):
    def __init__(self, image_folder, size, transform=None):
        self.data = None
        self.size = size
        self.image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.transform = transform

        


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        # Charger l'image
        image = Image.open(self.image_paths[i])
        to_tensor = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((self.size, self.size)),  # Correction ici
            transforms.ToTensor()
        ])
        image = to_tensor(image)

        return {"image": image }



class CustomDatasetCrop(Dataset):
    def __init__(self, size, image_folder, transform=None, random_crops=0):
        self.image_paths = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.transform = transform
        self.random_crops = random_crops
        self.taille_crop = size // 16
        self.num_crop = 16*16

    def __getitem__(self, index):
        # Charger l'image
        image = Image.open(self.image_paths[index])
        
        # Cropping fixe
        crops = [image.crop((i%16 * self.taille_crop, i//16 * self.taille_crop, (i%16 + 1) * self.taille_crop, (i//16 + 1) * self.taille_crop)) for i in range(self.num_crop)]
  
        to_tensor = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        ])

        crops = [to_tensor(crop) for crop in crops]

        # n = 1  # Le nombre de crops que vous souhaitez sélectionner
        # selected_crops = random.sample(crops, n)

        crops_tensor = torch.stack([crop for crop in crops])

        return {"image": crops_tensor }

    def __len__(self):
        return len(self.image_paths)




class CustomTrain(CustomDataset):
    def __init__(self, image_folder, size, transform=None):
        super().__init__(image_folder,size, transform)
        
        for root, _, files in os.walk(image_folder):
            for file in files:

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))



class CustomTest(CustomDataset):
    def __init__(self, image_folder, size,  transform=None):

        super().__init__(image_folder, size, transform)
        
        for root, _, files in os.walk(image_folder):
            for file in files:

                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))



class CustomTrain_crop(CustomDatasetCrop):
    def __init__(self, size, image_folder, transform=None, random_crops=0):
        super().__init__(size, image_folder, transform, random_crops)
        
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))

class CustomTest_crop(CustomDatasetCrop):
    def __init__(self, size, image_folder, transform=None, random_crops=0):

        super().__init__(size, image_folder, transform, random_crops)
   
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                    self.image_paths.append(os.path.join(root, file))