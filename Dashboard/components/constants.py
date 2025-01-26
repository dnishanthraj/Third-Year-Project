import os

# Adjust ROOT_DIR to point to the actual project root directory
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Define paths relative to the corrected project root
OUTPUT_MASK_DIR = os.path.join(ROOT_DIR, "Segmentation/model_outputs/NestedUNET_with_augmentation_20250124_153410_92030151/Segmentation_output/NestedUNET_with_augmentation")
GRAD_CAM_DIR = os.path.join(ROOT_DIR, "Segmentation/model_outputs/NestedUNET_with_augmentation_20250124_153410_92030151/Grad_CAM_output/NestedUNET_with_augmentation")
METRICS_DIR = os.path.join(ROOT_DIR, "Segmentation/model_outputs/NestedUNET_with_augmentation_20250124_153410_92030151/metrics")
IMAGE_DIR = os.path.join(ROOT_DIR, "Preprocessing/data/Image")
MASK_DIR = os.path.join(ROOT_DIR, "Preprocessing/data/Mask")
