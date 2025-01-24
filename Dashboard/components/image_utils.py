# components/image_utils.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def load_npy(filepath):
    """Load a .npy file and handle raw arrays or dictionaries."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        st.error(f"Error loading .npy file: {e}")
        return None

def combine_images(base_image, overlay=None, overlay_type=None):
    """Combine base image with overlay using color schemes."""
    base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())  # Normalize base image
    combined_image = np.stack([base_image] * 3, axis=-1)  # Convert grayscale to RGB

    if overlay is not None:
        overlay = overlay.astype(float) if overlay.dtype == bool else overlay
        if overlay.max() > overlay.min():
            overlay = (overlay - overlay.min()) / (overlay.max() - overlay.min())

        # Apply overlay type color schemes
        if overlay_type == "GC":  # Grad-CAM Heatmap
            overlay_colored = plt.cm.jet(overlay)[:, :, :3]
        elif overlay_type == "MA":  # Ground Truth Mask (Green)
            overlay_colored = np.stack([np.zeros_like(overlay), overlay, np.zeros_like(overlay)], axis=-1)
        elif overlay_type == "PD":  # Predicted Mask (Red)
            overlay_colored = np.stack([overlay, np.zeros_like(overlay), np.zeros_like(overlay)], axis=-1)
        else:
            overlay_colored = np.zeros_like(combined_image)

        # Combine base image and overlay
        combined_image += 0.5 * overlay_colored
        combined_image = np.clip(combined_image, 0, 1)

    return (combined_image * 255).astype(np.uint8)