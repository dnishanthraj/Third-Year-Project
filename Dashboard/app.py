import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Paths for demonstration purposes
OUTPUT_MASK_DIR = "../Segmentation/model_outputs/NestedUNET_with_augmentation/Segmentation_output/NestedUNET_with_augmentation"
GRAD_CAM_DIR = "../Segmentation/model_outputs/NestedUNET_with_augmentation/Grad_CAM_output/NestedUNET_with_augmentation"
IMAGE_DIR = "../Preprocessing/data/Image"
MASK_DIR = "../Preprocessing/data/Mask"

def load_npy(filepath):
    """Load a .npy file and handle raw arrays or dictionaries."""
    try:
        data = np.load(filepath, allow_pickle=True)
        return data
    except Exception as e:
        st.error(f"Error loading .npy file: {e}")
        return None

def display_overlay_streamlit(base_image, overlay=None, overlay_type=None):
    """Display overlay on the base image using distinct color schemes."""
    plt.figure(figsize=(6, 6))
    plt.imshow(base_image, cmap="gray", interpolation="none")  # Grayscale for the base image

    # Apply overlay with specific color schemes
    if overlay is not None:
        if overlay_type == "GC":  # Grad-CAM Heatmap
            plt.imshow(overlay, cmap="jet", alpha=0.5, interpolation="none")
        elif overlay_type == "MA":  # Ground Truth Mask (Green)
            green_overlay = np.zeros((*overlay.shape, 4))  # RGBA
            green_overlay[:, :, 1] = overlay  # Green channel
            green_overlay[:, :, 3] = overlay  # Alpha channel
            plt.imshow(green_overlay, alpha=0.5, interpolation="none")
        elif overlay_type == "PD":  # Predicted Mask (Red)
            red_overlay = np.zeros((*overlay.shape, 4))  # RGBA
            red_overlay[:, :, 0] = overlay  # Red channel
            red_overlay[:, :, 3] = overlay  # Alpha channel
            plt.imshow(red_overlay, alpha=0.5, interpolation="none")

    plt.axis("off")
    st.pyplot(plt)

def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for the correct .npy file within the patient-specific subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    return file_path if os.path.exists(file_path) else None

def display_overlay(patient_id, region_id, slice_name, overlay_type):
    """Display an overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask) on the Original Image."""
    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), original_file_name)
    overlay_path = (
        find_file_in_subfolder(MASK_DIR, int(patient_id), overlay_file_name) if overlay_type == "MA"
        else os.path.join(GRAD_CAM_DIR if overlay_type == "GC" else OUTPUT_MASK_DIR, overlay_file_name)
    )

    original_image = load_npy(original_path) if original_path else None
    overlay_image = load_npy(overlay_path) if overlay_path and os.path.exists(overlay_path) else None

    if original_image is not None:
        display_overlay_streamlit(original_image, overlay_image, overlay_type)
    else:
        st.warning("Original image not found.")

def parse_filenames(files, prefix):
    """Parse filenames to group by Patient ID, Region ID, and Slices."""
    patients = {}
    for file in files:
        if prefix in file:
            parts = file.split("_")
            patient_id = parts[0]
            region_id = parts[1].replace("PD", "").replace("GC", "").replace("MA", "").replace("NI", "")
            slice_info = parts[-1].replace(".npy", "").replace("slice", "Slice ")
            patients.setdefault(patient_id, {}).setdefault(region_id, []).append((file, slice_info))
    return patients

def sort_patients(patients):
    """Sort patients, regions, and slices numerically."""
    sorted_patients = {}
    for patient_id, regions in sorted(patients.items(), key=lambda x: int(x[0])):
        sorted_regions = {region_id: sorted(slices, key=lambda x: int(x[1].split()[-1]))
                          for region_id, slices in regions.items()}
        sorted_patients[patient_id] = sorted_regions
    return sorted_patients

# Static Header Title
st.title("Explainable XAI for Lung Nodule Segmentation Dashboard")

# Collect `.npy` files dynamically
output_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('.npy')]
grad_cam_files = [f for f in os.listdir(GRAD_CAM_DIR) if f.endswith('.npy')]

# Parse and sort filenames
output_masks_by_patient = sort_patients(parse_filenames(output_mask_files, prefix="PD"))
grad_cams_by_patient = sort_patients(parse_filenames(grad_cam_files, prefix="GC"))

# Sidebar for selecting patient, region, slice, and overlay type
st.sidebar.title("Available Files")
selected_patient = st.sidebar.selectbox("Select Patient", list(output_masks_by_patient.keys()))
selected_region = st.sidebar.selectbox("Select Region", list(output_masks_by_patient[selected_patient].keys()))
selected_slice = st.sidebar.selectbox("Select Slice", output_masks_by_patient[selected_patient][selected_region], format_func=lambda x: x[1])
overlay_type = st.sidebar.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"])

# Display the selected overlay
if selected_patient and selected_region and selected_slice:
    slice_index = selected_slice[1].split()[-1]
    st.header(f"Patient {int(selected_patient)} | Region {int(selected_region)} | Slice {slice_index}")
    if overlay_type != "None":
        overlay_code = "GC" if overlay_type == "Grad-CAM Heatmap" else "MA" if overlay_type == "Ground Truth Mask" else "PD"
        display_overlay(selected_patient, selected_region, slice_index, overlay_code)
    else:
        file_name = f"{selected_patient}_NI{selected_region}_slice{slice_index}.npy"
        original_path = find_file_in_subfolder(IMAGE_DIR, int(selected_patient), file_name)
        original_image = load_npy(original_path) if original_path else None
        if original_image is not None:
            display_overlay_streamlit(original_image)
