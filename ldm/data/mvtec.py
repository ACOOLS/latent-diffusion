import os, yaml, pickle, shutil, tarfile, glob
import cv2
import albumentations
import PIL
import numpy as np
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from functools import partial
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, Subset
import torch
import taming.data.utils as tdu
from taming.data.base import ImagePaths

from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light
import torchvision.transforms as transforms
import random

from PIL import Image
import albumentations


class MVTecDataset(Dataset):
    def __init__(self, data_root, process_images=False):
        self.data_root = data_root
        self.process_images = process_images
        self.image_paths = [os.path.join(data_root, fname) for fname in os.listdir(data_root)]

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # image = Image.open(img_path).convert('RGB')  # Convertir en RGB pour la cohérence

        # if self.process_images:
        #     pass

        return img_path

    def __len__(self):
        return len(self.image_paths)


class MVTecSR(Dataset):
    def __init__(self, data_root, size=None,
                 degradation=None, downscale_f=4, min_crop_f=0.5, max_crop_f=1.,
                 random_crop=True):
        self.data_root = data_root
        self.transform = None
        self.random_crops = 5
        self.taille_crop = size // 16
        self.num_crop = 16 * 16 
        self.image_paths = [os.path.join(data_root, file_name) for file_name in os.listdir(data_root)]
    
        assert size
        assert (size / downscale_f).is_integer()
        self.size = size
        print("size ", size)
        self.LR_size = int(size / downscale_f)
        self.min_crop_f = min_crop_f
        self.max_crop_f = max_crop_f
        assert(max_crop_f <= 1.)
        self.center_crop = not random_crop

        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size//8, interpolation=cv2.INTER_AREA)

        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            self.pil_interpolation = degradation.startswith("pil_")

            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize, size=self.LR_size, interpolation=interpolation_fn)

            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def get_random_crops(image, num_crops=1, crop_size=(256, 256), crop_range=(192, 832)):
        crops = []
        min_x, max_x = crop_range  # Plage pour la largeur
        min_y, max_y = crop_range  # Plage pour la hauteur

        for _ in range(num_crops):
            # Choisir un point de départ aléatoire dans la plage spécifiée
            x = random.randint(min_x, max_x - crop_size[0])
            y = random.randint(min_y, max_y - crop_size[1])

            # Extraire et ajouter le crop à la liste
            crop = image.crop((x, y, x + crop_size[0], y + crop_size[1]))
            crops.append(crop)

        return crops
    
    @staticmethod
    def fixed_crops(image):
        # Définir les points de départ et de fin pour la découpe
        x_debut, y_debut = 0, 0
        x_fin, y_fin = 1024, 1024
        crops = []
        # Taille des crops
        taille_crop = 64
        nb_crops = 5

        max_x_debut = image.width - taille_crop * nb_crops
        max_y_debut = image.height - taille_crop * nb_crops

        x_debut = random.randint(64, max_x_debut)
        y_debut = random.randint(64, max_y_debut)

        crops = []

        # Itérer sur la zone de découpe en pas de 64 pixels
        for i in range(nb_crops):
            for j in range(nb_crops):
                x = x_debut + j * taille_crop
                y = y_debut + i * taille_crop
                # Définir les coordonnées du coin supérieur gauche et inférieur droit du crop
                box = (x, y, x + taille_crop, y + taille_crop)
                
                # Effectuer la découpe
                crop_image = image.crop(box)
                crops.append(crop_image)

        return crops

    def __getitem__(self, idx):
        if self.base is None:
            raise ValueError("Base dataset is not initialized.")

        example = self.base[idx]
        # Charger et prétraiter l'image ici
        #print("ICIC ???? ", self.image_paths[idx])
        image = Image.open(self.image_paths[idx])
        
        # Utiliser get_random_crops pour extraire des crops aléatoires
        #crops = self.get_random_crops(image, num_crops=100, crop_size=(64, 64), crop_range=(128, 896))
        #crops = self.fixed_crops(image)
        
        to_tensor = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
        ])

        # Extraction de crops fixes
        crops = [image.crop((i % 16 * self.taille_crop, i // 16 * self.taille_crop, 
                             (i % 16 + 1) * self.taille_crop, (i // 16 + 1) * self.taille_crop))
                 for i in range(self.num_crop)]
        
        # Ajout des crops aléatoires
        for _ in range(self.random_crops):
            x = random.randint(0, image.width - self.taille_crop)
            y = random.randint(0, image.height - self.taille_crop)
            random_crop = image.crop((x, y, x + self.taille_crop, y + self.taille_crop))
            crops.append(random_crop)

        # Transformation des crops
        if self.transform:
            crops = [self.transform(crop) for crop in crops]
        else:
            to_tensor = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
            crops = [to_tensor(crop) for crop in crops]

        # image = np.array(image).astype(np.uint8)
    
        # #crops = [to_tensor(crop) for crop in crops]

        # image = crops[0]
        
        
        
        # # if not image.mode == "RGB":
        # #     image = image.convert("RGB")

        # image = np.array(image).astype(np.uint8)

        # min_side_len = min(image.shape[:2])
        # crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        # crop_side_len = int(crop_side_len)

        # if self.center_crop:
        #     self.cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)

        # else:
        #     self.cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)

        # image = self.cropper(image=image)["image"]
        # image = self.image_rescaler(image=image)["image"]

        # if self.pil_interpolation:
        #     image_pil = PIL.Image.fromarray(image)
        #     LR_image_pil = self.degradation_process(image_pil)
        #     LR_image = np.array(LR_image_pil).astype(np.uint8)

        # else:
        #     LR_image = self.degradation_process(image=image)["image"]
        
        # image_pil = PIL.Image.fromarray(image)
        # image= to_tensor(image_pil)
        # LR_image= to_tensor(LR_image_pil)
        # image_np = np.array(image).astype(np.float32)
        # LR_image_np = np.array(LR_image).astype(np.float32)
        # #image_np = (image_np/127.5 - 1.0)
        # #LR_image_np = (LR_image_np/127.5 - 1.0)

        # example = {"image": image_np, "LR_image": LR_image_np}

        # return example

        # Initialiser les listes pour stocker les résultats de tous les crops
        images_np = []
        LR_images_np = []

        # Traiter chaque crop
        for crop in crops:
            crop = np.array(crop).astype(np.uint8)
            min_side_len = min(crop.shape[:2])
         
            crop_side_len = min_side_len #* np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
            crop_side_len = int(crop_side_len)

            # Appliquer des opérations de crop supplémentaires si nécessaire
            if self.center_crop:
                cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len)
            else:
                cropper = albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
            
            cropped_image = cropper(image=crop)["image"]
            resized_image = self.image_rescaler(image=cropped_image)["image"]

            # Traiter pour obtenir l'image LR
            if self.pil_interpolation:
                image_pil = PIL.Image.fromarray(resized_image)
                LR_image_pil = self.degradation_process(image_pil)
                LR_image = np.array(LR_image_pil).astype(np.uint8)
            else:
                LR_image = self.degradation_process(image=resized_image)["image"]

            # Convertir en tenseurs et ajouter aux listes
            images_np.append(to_tensor(PIL.Image.fromarray(resized_image)))
            LR_images_np.append(to_tensor(PIL.Image.fromarray(LR_image)))

        # Empiler les images de tous les crops pour créer un batch
        images_tensor = torch.stack(images_np)
        LR_images_tensor = torch.stack(LR_images_np)

        # Les images sont maintenant des tenseurs de la forme [num_crops, 1, H, W]
        return {"image": images_tensor, "LR_image": LR_images_tensor}
       

class MVTecSRTrain(MVTecSR):
    def __init__(self, data_root, subset_size=None, **kwargs):
        super().__init__(data_root, **kwargs)
        self.data_root_train = data_root
        self.subset_size = subset_size
        print("subset_size :", subset_size)

        self.image_paths = []  # Initialisation avec une liste vide
        # Chargement des chemins d'image
        self.base = self.get_base()

    def __len__(self):
        if self.image_paths is None:
            print("Erreur: self.image_paths est None")
            return 0
        return len(self.image_paths)

    def get_base(self):
        
        # for root, _, files in os.walk(self.data_root):
        #     for file in files:
        #         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        #             self.image_paths.append(os.path.join(root, file))
        
        mvtec_dataset = MVTecDataset(data_root=self.data_root_train, process_images=False)
        print("len(self.image_paths) ", len(mvtec_dataset))
        #if self.subset_size is not None and 0 < self.subset_size < len(self.image_paths):
        selected_indices = random.sample(range(len(mvtec_dataset)), self.subset_size)
        self.image_paths = [mvtec_dataset[i] for i in selected_indices]
        mvtec_dataset = Subset(mvtec_dataset, selected_indices)
        

        return mvtec_dataset
    

class MVTecSRValidation(MVTecSR):
    def __init__(self, data_root, subset_size=None, **kwargs):
        super().__init__(data_root, **kwargs)
        self.data_root_val = data_root
        self.subset_size = subset_size
        self.image_paths = []  # Initialisation avec une liste vide
        # Chargement des chemins d'image
        self.base = self.get_base()

    def __len__(self):
        if self.image_paths is None:
            print("Erreur: self.image_paths est None")
            return 0
        return len(self.image_paths)

    def get_base(self):
        # self.image_paths = []
        # for root, _, files in os.walk(self.data_root_val):
        #     for file in files:
        #         if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
        #             self.image_paths.append(os.path.join(root, file))

        mvtec_dataset = MVTecDataset(data_root=self.data_root_val, process_images=False)
        print("len(self.image_paths) ", len(mvtec_dataset))
        #if self.subset_size is not None and 0 < self.subset_size < len(self.image_paths):
        selected_indices = random.sample(range(len(mvtec_dataset)), self.subset_size)
        self.image_paths = [mvtec_dataset[i] for i in selected_indices]
        mvtec_dataset = Subset(mvtec_dataset, selected_indices)
        
        return mvtec_dataset