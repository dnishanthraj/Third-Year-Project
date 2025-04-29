# ---------------------------------------------------------
# Configuration File Creator for Lung Nodule Segmentation
# ---------------------------------------------------------

from configparser import ConfigParser

if __name__ == "__main__":
    # This script generates a configuration (.conf) file for the project.
    # Update the directories below to match your application needs.

    config = ConfigParser()

    # -------------------------------
    # Section: prepare_dataset
    # -------------------------------
    config['prepare_dataset'] = {
        # Path to the original LIDC-IDRI DICOM dataset
        'LIDC_DICOM_PATH': './LIDC-IDRI',

        # Directories to save processed outputs
        'MASK_PATH': './data/Mask',             # Processed mask directory
        'IMAGE_PATH': './data/Image',            # Processed image directory

        # Directories to save clean (no nodule) images and masks
        'CLEAN_PATH_IMAGE': './data/Clean/Image',
        'CLEAN_PATH_MASK': './data/Clean/Mask',

        # Directory to save metadata (CSV files with annotations, malignancy scores, splits, etc.)
        'META_PATH': './data/Meta/',

        # Threshold to filter masks based on size (sum of mask pixels)
        # Small masks below this threshold (e.g., 8 pixels) are discarded to reduce noise/outliers
        'Mask_Threshold': 8
    }

    # -------------------------------
    # Section: pylidc (library settings)
    # -------------------------------
    config['pylidc'] = {
        # Confidence level for combining annotations (overlap among radiologists)
        'confidence_level': 0.5,

        # Image padding size (final image size after preprocessing)
        'padding_size': 512
    }

    # -------------------------------
    # Write Configuration File
    # -------------------------------
    # The configuration is saved to 'lung.conf'
    with open('./lung.conf', 'w') as f:
        config.write(f)
