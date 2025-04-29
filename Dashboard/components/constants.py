# -------------------------------------------------------
# constants.py - Directory Paths for the Lung Nodule Dashboard
# -------------------------------------------------------

import os

# -------------------------------------------------------
# Root Directory (2 levels up from this file)
# -------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# -------------------------------------------------------
# Base Directory for Model Outputs (predictions, grad-cam, metrics)
# -------------------------------------------------------
MODEL_OUTPUTS_BASE_DIR = os.path.join(ROOT_DIR, "Segmentation/model_outputs")

# -------------------------------------------------------
# Dynamic Paths (set later during runtime)
# -------------------------------------------------------
OUTPUT_MASK_DIR = ""   # Path where predicted masks will be loaded from
GRAD_CAM_DIR = ""      # Path where Grad-CAM visualizations will be loaded from
METRICS_DIR = ""       # Path where model evaluation metrics will be loaded from

# -------------------------------------------------------
# Fixed Data Paths (preprocessed images and masks)
# -------------------------------------------------------
IMAGE_DIR = os.path.join(ROOT_DIR, "Preprocessing/data/Image")  # Directory containing original CT images
MASK_DIR = os.path.join(ROOT_DIR, "Preprocessing/data/Mask")    # Directory containing ground-truth masks
