# -------------------------------------------------------
# components/image_utils.py - Utility Functions for Images
# -------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------------------------------
# Load a .npy file (either raw array or dictionary)
# -------------------------------------------------------
def load_npy(filepath):
    """Load a .npy file and handle raw arrays or dictionaries."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        st.error(f"Error loading .npy file: {e}")
        return None

# -------------------------------------------------------
# Combine base image with optional overlay (mask or Grad-CAM)
# -------------------------------------------------------
def combine_images(base_image, overlay=None, overlay_type=None):
    """
    Combine a base CT slice with an overlay (mask or Grad-CAM).
    
    Args:
        base_image: Grayscale lung slice (numpy array)
        overlay: Mask or Grad-CAM heatmap (numpy array)
        overlay_type: Type of overlay ("MA" = Ground Truth, "PD" = Predicted Mask, "GC" = Grad-CAM)

    Returns:
        RGB combined image (numpy array)
    """

    # Normalize the base image to [0, 1]
    base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())

    # Convert grayscale to RGB
    combined_image = np.stack([base_image] * 3, axis=-1)

    # If overlay is provided
    if overlay is not None:
        # Ensure overlay is float type for processing
        overlay = overlay.astype(float) if overlay.dtype == bool else overlay
        if overlay.max() > overlay.min():
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())

        # -------------------------------------------------------
        # Apply Color Maps depending on overlay_type
        # -------------------------------------------------------
        if overlay_type == "GC":
            # Grad-CAM heatmap using 'jet' colormap
            overlay_colored = plt.cm.jet(overlay)[:, :, :3]
        elif overlay_type == "MA":
            # Ground Truth Mask in Green
            overlay_colored = np.stack([np.zeros_like(overlay), overlay, np.zeros_like(overlay)], axis=-1)
        elif overlay_type == "PD":
            # Predicted Mask in Red
            overlay_colored = np.stack([overlay, np.zeros_like(overlay), np.zeros_like(overlay)], axis=-1)
        else:
            # Default (no overlay)
            overlay_colored = np.zeros_like(combined_image)

        # Blend the overlay into the base image (50% strength)
        combined_image += 0.5 * overlay_colored
        combined_image = np.clip(combined_image, 0, 1)

    # Convert to uint8 format for visualization
    return (combined_image * 255).astype(np.uint8)
