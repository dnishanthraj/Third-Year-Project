# -----------------------------------------------
# Utility functions for LIDC-IDRI Dataset Processing
# -----------------------------------------------

import argparse
import os
import numpy as np

from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from skimage import measure, morphology
import scipy.ndimage as ndimage
from sklearn.cluster import KMeans

# -------------------------------
# Check if input string is a directory
# -------------------------------
def is_dir_path(string):
    """
    Checks if the given string is a valid directory path.
    
    Args:
        string (str): Directory path to check.
        
    Returns:
        str: The input string if it is a valid directory.

    Raises:
        NotADirectoryError: If the path is not a directory.
    """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


# -------------------------------
# Lung Region Segmentation
# -------------------------------
def segment_lung(img):
    """
    Segments the lung regions from a CT slice.

    The function applies standardisation, filtering, and morphological
    operations to isolate the lung areas, removing non-lung regions.

    Args:
        img (numpy.ndarray): A single 2D CT slice.

    Returns:
        numpy.ndarray: A 2D segmented lung mask applied to the original image.
    """
    # Standardise pixel intensities (zero mean, unit variance)
    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    # Focus on the central region to reduce noise
    middle = img[100:400, 100:400]
    mean_middle = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)

    # Remove extreme pixel values (replace with mean)
    img[img == max_val] = mean_middle
    img[img == min_val] = mean_middle

    # Apply median filter to denoise
    img = median_filter(img, size=3)

    # Apply anisotropic diffusion to enhance edges while smoothing noise
    img = anisotropic_diffusion(img)

    # -------------------------------
    # Thresholding via KMeans
    # -------------------------------
    kmeans = KMeans(n_clusters=2).fit(middle.reshape(-1, 1))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # Binary thresholding

    # Morphological operations
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))

    # Label connected regions
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)

    # Filter regions to keep only good lung areas
    good_labels = []
    for prop in regions:
        B = prop.bbox  # bounding box (min_row, min_col, max_row, max_col)
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)

    # Create empty mask
    mask = np.zeros((512, 512), dtype=np.int8)

    # Build final lung mask
    for N in good_labels:
        mask += np.where(labels == N, 1, 0)

    # Final dilation to make lung mask smoother and close gaps
    mask = morphology.dilation(mask, np.ones([10, 10]))

    # Apply mask to the original image (preserving lung regions only)
    return mask * img


# -------------------------------
# Count Trainable Parameters
# -------------------------------
def count_params(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        int: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
