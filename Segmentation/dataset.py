# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# Custom Dataset Class for LIDC-IDRI Lung Nodule Segmentation
# ---------------------------------------------------------
class MyLidcDataset(Dataset):
    def __init__(self, IMAGES_PATHS, MASK_PATHS, Albumentation=False, return_pid=False):
        """
        Args:
            IMAGES_PATHS (list): List of paths to image .npy files.
            MASK_PATHS (list): List of paths to corresponding mask .npy files.
            Albumentation (bool): Whether to apply Albumentations-based data augmentation.
            return_pid (bool): Whether to return patient IDs with each sample.
        """
        self.image_paths = IMAGES_PATHS
        self.mask_paths = MASK_PATHS
        self.albumentation = Albumentation
        self.return_pid = return_pid

        # Extract patient IDs from filenames
        self.patient_ids = []
        for path in self.image_paths:
            filename = os.path.basename(path)      # e.g., "0001_01_images.npy"
            pid = filename.split('_')[0]            # Extract "0001" as patient ID
            self.patient_ids.append(pid)

        # Define Albumentations-based transformations (advanced augmentations) - decide which augmentations you would like to keep, and comment out the unused ones.
        self.albu_transformations = albu.Compose([
            albu.ElasticTransform(alpha=1.0, alpha_affine=0.5, sigma=4, p=0.2),
            albu.HorizontalFlip(p=0.2),
            albu.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
            albu.RandomGamma(gamma_limit=(80, 120), p=0.2), 
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2),
            albu.GaussianBlur(blur_limit=(3, 5), p=0.2),
            ToTensorV2()   # Convert to PyTorch tensors
        ])

        # Define basic PyTorch transformations (for cases without augmentation)
        self.transformations = transforms.Compose([
            transforms.ToTensor()
        ])

    def transform(self, image, mask):
        """
        Apply normalization and augmentation to input image and mask.

        Args:
            image (np.ndarray): Raw input image.
            mask (np.ndarray): Corresponding binary mask.

        Returns:
            tuple: (image, mask) as torch.FloatTensors
        """
        # Normalize image to [0, 1] range
        image = image - image.min() 
        if image.max() > 0:
            image = image / image.max()

        if self.albumentation:
            # Prepare image and mask for Albumentations (shape: H x W x C)
            image = image.reshape(512, 512, 1).astype('float32')
            mask = mask.reshape(512, 512, 1).astype('uint8')

            # Apply Albumentations transformations
            augmented = self.albu_transformations(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

            # Reshape mask back to [1, H, W] format
            mask = mask.reshape([1, 512, 512])
        else:
            # Apply basic PyTorch ToTensor() transformation
            image = self.transformations(image)
            mask = self.transformations(mask)

        # Ensure both image and mask are FloatTensors
        image, mask = image.type(torch.FloatTensor), mask.type(torch.FloatTensor)
        return image, mask

    def __getitem__(self, index):
        """
        Retrieves a single sample (image and mask) by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) if return_pid is False,
                   (image, mask, patient_id) if return_pid is True
        """
        # Load image and mask from .npy files
        image = np.load(self.image_paths[index])
        mask = np.load(self.mask_paths[index])

        # Apply normalization and (optional) augmentation
        image, mask = self.transform(image, mask)

        if self.return_pid:
            # Return image, mask, and patient ID if needed
            pid = self.patient_ids[index]
            return image, mask, pid
        else:
            # Otherwise, return just the image and mask
            return image, mask

    def __len__(self):
        """
        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.image_paths)
