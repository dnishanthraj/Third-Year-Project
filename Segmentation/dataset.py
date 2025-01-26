import os
import numpy as np
import glob

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF
import torchvision
from torchvision import transforms


import albumentations as albu
from albumentations.pytorch import ToTensorV2

class MyLidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS,Albumentation=False):
        """
        IMAGES_PATHS: list of images paths ['./Images/0001_01_images.npy','./Images/0001_02_images.npy']
        MASKS_PATHS: list of masks paths ['./Masks/0001_01_masks.npy','./Masks/0001_02_masks.npy']
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths= MASK_PATHS
        self.albumentation = Albumentation


        self.albu_transformations = albu.Compose([
            albu.ElasticTransform(alpha=1.0, alpha_affine=0.5, sigma=4, p=0.2),
            albu.HorizontalFlip(p=0.2),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            albu.RandomGamma(gamma_limit=(80, 120), p=0.2),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            albu.GaussianBlur(blur_limit=(3, 5), p=0.2),
            # albu.Rotate(limit=15, p=0.2),
            # albu.PadIfNeeded(min_height=256, min_width=256, p=0.2),
            # albu.CLAHE(clip_limit=2.0, p=0.2),
            ToTensorV2()
        ])

        self.transformations = transforms.Compose([transforms.ToTensor()])

    def transform(self, image, mask):
        # Normalize image to [0, 1] range
        image = image - image.min()  # Shift minimum value to 0
        if image.max() > 0:  # Avoid division by zero
            image = image / image.max()

        if self.albumentation:
            # Ensure Albumentations expects correct shapes and types
            image = image.reshape(512, 512, 1).astype('float32')
            mask = mask.reshape(512, 512, 1).astype('uint8')

            # Apply Albumentations
            augmented = self.albu_transformations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.reshape([1, 512, 512])
        else:
            image = self.transformations(image)
            mask = self.transformations(mask)

        # Convert to PyTorch tensors
        image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image, mask


    def __getitem__(self, index):
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])
        image,mask = self.transform(image,mask)
        return image,mask

    def __len__(self):
        return len(self.image_paths)
