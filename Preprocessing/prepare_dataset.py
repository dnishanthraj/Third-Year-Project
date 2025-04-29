# -----------------------------------------------
# LIDC-IDRI Dataset Preprocessing Script
# -----------------------------------------------
# This script extracts nodules and clean slices, segments the lung regions,
# saves images and masks, and generates metadata for later training/validation.
# -----------------------------------------------

import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import math
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high

from utils import is_dir_path, segment_lung
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# -------------------------------
# Load configuration
# -------------------------------
parser = ConfigParser()
parser.read('lung.conf')

# Directories from configuration
DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset', 'MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset', 'IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset', 'META_PATH'))

# Hyperparameters
mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')
confidence_level = parser.getfloat('pylidc', 'confidence_level')
padding = parser.getint('pylidc', 'padding_size')


# -------------------------------
# Dataset Preparer Class
# -------------------------------
class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK, META_DIR, mask_threshold, padding, confidence_level=0.5):
        # Initialization
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]

        # Metadata DataFrame
        self.meta = pd.DataFrame(columns=[
            'patient_id', 'nodule_no', 'slice_no', 'original_image', 'mask_image',
            'malignancy', 'subtlety', 'texture', 'sphericity', 'margin', 'is_cancer', 'is_clean'
        ])

    def calculate_malignancy(self, nodule):
        """Calculate the malignancy score for a nodule based on multiple annotations."""
        malignancy_scores = [annotation.malignancy for annotation in nodule]
        malignancy = median_high(malignancy_scores)

        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def calculate_nodule_characteristics(self, nodule):
        """Extract subtlety, texture, sphericity, margin scores."""
        subtlety = median_high([ann.subtlety for ann in nodule])
        texture = median_high([ann.texture for ann in nodule])
        sphericity = median_high([ann.sphericity for ann in nodule])
        margin = median_high([ann.margin for ann in nodule])
        return subtlety, texture, sphericity, margin

    def save_meta(self, meta_list):
        """Append a new row to the metadata."""
        tmp = pd.Series(meta_list, index=self.meta.columns)
        self.meta = pd.concat([self.meta, tmp.to_frame().T], ignore_index=True)

    def prepare_dataset(self):
        """Main function to prepare and save dataset."""

        prefix = [str(x).zfill(3) for x in range(1000)]  # 000, 001, 002...

        # Ensure necessary directories exist
        for path in [self.img_path, self.mask_path, self.clean_path_img, self.clean_path_mask, self.meta_path]:
            os.makedirs(path, exist_ok=True)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient  # Example: LIDC-IDRI-0001
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()

            print(f"Patient ID: {pid} | DICOM Shape: {vol.shape} | Annotated Nodules: {len(nodules_annotation)}")

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            patient_image_dir.mkdir(parents=True, exist_ok=True)
            patient_mask_dir.mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Case 1: Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    lung_np_array = vol[cbbox]

                    if lung_np_array.shape[2] == 0:
                        print(f"Skipping empty nodule {nodule_idx} for patient {pid}.")
                        continue

                    malignancy, cancer_label = self.calculate_malignancy(nodule)
                    subtlety, texture, sphericity, margin = self.calculate_nodule_characteristics(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                            continue  # Skip tiny masks

                        # Segment lung area
                        lung_segmented = segment_lung(lung_np_array[:, :, nodule_slice])
                        lung_segmented[lung_segmented == -0] = 0

                        # Save image and mask
                        nodule_name = f"{pid[-4:]}_NI{prefix[nodule_idx]}_slice{prefix[nodule_slice]}"
                        mask_name = f"{pid[-4:]}_MA{prefix[nodule_idx]}_slice{prefix[nodule_slice]}"

                        nodule_meta_path = f"{pid}/{nodule_name}"
                        mask_meta_path = f"{pid}/{mask_name}"

                        meta_list = [
                            pid[-4:], nodule_idx, prefix[nodule_slice],
                            nodule_meta_path, mask_meta_path,
                            malignancy, subtlety, texture, sphericity, margin,
                            cancer_label, False
                        ]

                        self.save_meta(meta_list)
                        np.save(patient_image_dir / nodule_name, lung_segmented)
                        np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])

            else:
                # Case 2: Clean patients (no nodules)
                print(f"Clean Dataset: {pid}")
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                patient_clean_dir_image.mkdir(parents=True, exist_ok=True)
                patient_clean_dir_mask.mkdir(parents=True, exist_ok=True)

                for slice_idx in range(min(vol.shape[2], 50)):  # Limit to first 50 slices
                    lung_segmented = segment_lung(vol[:, :, slice_idx])
                    lung_segmented[lung_segmented == -0] = 0
                    lung_mask = np.zeros_like(lung_segmented)

                    nodule_name = f"{pid[-4:]}_CN001_slice{prefix[slice_idx]}"
                    mask_name = f"{pid[-4:]}_CM001_slice{prefix[slice_idx]}"
                    nodule_meta_path = f"{pid}/{nodule_name}"
                    mask_meta_path = f"{pid}/{mask_name}"

                    meta_list = [
                        pid[-4:], slice_idx, prefix[slice_idx],
                        nodule_meta_path, mask_meta_path,
                        0, 0, 0, 0, 0, False, True
                    ]

                    self.save_meta(meta_list)
                    np.save(patient_clean_dir_image / nodule_name, lung_segmented)
                    np.save(patient_clean_dir_mask / mask_name, lung_mask)

            # Save updated metadata after each patient
            print(f"Saved metadata for patient {pid}")
            self.meta.to_csv(self.meta_path + 'meta_info.csv', index=False)


# -------------------------------
# Main Script Entry
# -------------------------------
if __name__ == '__main__':
    # Exclude hidden files
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()

    # Start from the first patient (can be modified easily)
    start_index = LIDC_IDRI_list.index('LIDC-IDRI-0001')
    LIDC_IDRI_list = LIDC_IDRI_list[start_index:]

    # Create and run dataset preparer
    dataset_maker = MakeDataSet(
        LIDC_IDRI_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        mask_threshold,
        padding,
        confidence_level
    )
    dataset_maker.prepare_dataset()
