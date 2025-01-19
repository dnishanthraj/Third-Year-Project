import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt

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

def overlay_images(base_image, overlay, alpha=0.5, cmap_overlay=None):
    """Overlay the non-zero values of the overlay image on the base image."""
    base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())  # Normalize base image

    # Ensure overlay is converted to integers if it's boolean
    if overlay.dtype == bool:
        overlay = overlay.astype(int)

    # Normalize the overlay
    overlay_normalized = (overlay - overlay.min()) / (overlay.max() - overlay.min()) if overlay.max() > 0 else overlay

    # Create a colored overlay
    if cmap_overlay:
        overlay_colored = plt.cm.get_cmap(cmap_overlay)(overlay_normalized)[:, :, :3]
    else:
        overlay_colored = np.stack([overlay_normalized] * 3, axis=-1)

    # Apply overlay where non-zero
    combined_image = base_image.copy()
    for i in range(3):  # RGB channels
        combined_image += alpha * overlay_colored[:, :, i] * (overlay > 0)

    # Clip to ensure valid image range
    combined_image = np.clip(combined_image, 0, 1)
    return combined_image


def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for the correct .npy file within the patient-specific subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def display_overlay(patient_id, region_id, slice_name, overlay_type):
    """Display an overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask) on the Original Image."""
    # File paths
    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), original_file_name)
    overlay_path = (
        find_file_in_subfolder(MASK_DIR, int(patient_id), overlay_file_name)  # For Ground Truth
        if overlay_type == "MA"
        else os.path.join(
            GRAD_CAM_DIR if overlay_type == "GC" else OUTPUT_MASK_DIR, overlay_file_name
        )
    )

    # Load images
    original_image = load_npy(original_path) if original_path else None
    overlay_image = load_npy(overlay_path) if overlay_path and os.path.exists(overlay_path) else None

    # Display overlay
    if original_image is not None:
        if overlay_image is not None:
            combined_image = overlay_images(
                original_image, overlay_image, alpha=0.5, cmap_overlay=("jet" if overlay_type == "GC" else "cividis")
            )

            display_image(combined_image, caption=f"Original + {overlay_type} Overlay")
        else:
            display_image(original_image, caption="Original Image (No Overlay)")
    else:
        st.warning("Original image not found.")

def display_image(data, caption, cmap=None):
    """Display the data with optional colormap for Grad-CAM or masks."""
    if isinstance(data, np.ndarray):
        if data.ndim == 2:  # Ensure 2D data
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data, cmap=cmap)
            ax.axis('off')
            ax.set_title(caption)
            st.pyplot(fig)
        else:
            st.warning(f"Data has unsupported shape {data.shape} for visualization.")
    else:
        st.warning("Data is not a NumPy array.")

def parse_filenames(files, prefix):
    """
    Parse filenames to group by Patient ID, Region ID, and Slices.
    Expected format: `0008_PD001_slice001`
    """
    patients = {}
    for file in files:
        if prefix in file:  # Ensure the file matches the correct type
            parts = file.split("_")
            patient_id = parts[0]
            region_id = parts[1].replace("PD", "").replace("GC", "").replace("MA", "").replace("NI", "")  # Extract middle number
            slice_info = parts[-1].replace(".npy", "").replace("slice", "Slice ")
            
            if patient_id not in patients:
                patients[patient_id] = {}
            if region_id not in patients[patient_id]:
                patients[patient_id][region_id] = []
            
            patients[patient_id][region_id].append((file, slice_info))
    return patients

def sort_patients(patients):
    """Sort patients, regions, and slices numerically."""
    sorted_patients = {}
    for patient_id, regions in sorted(patients.items(), key=lambda x: int(x[0])):  # Sort by Patient ID
        sorted_regions = {}
        for region_id, slices in sorted(regions.items(), key=lambda x: int(x[0])):  # Sort by Region ID
            sorted_regions[region_id] = sorted(slices, key=lambda x: int(x[1].split()[-1]))  # Sort by Slice Number
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
selected_patient = st.sidebar.selectbox(
    "Select Patient", 
    list(output_masks_by_patient.keys()), 
    format_func=lambda x: f"Patient {int(x)}"
)

selected_region = st.sidebar.selectbox(
    "Select Region", 
    list(output_masks_by_patient[selected_patient].keys()), 
    format_func=lambda x: f"Region {int(x)}"
)

selected_slice = st.sidebar.selectbox(
    "Select Slice", 
    output_masks_by_patient[selected_patient][selected_region], 
    format_func=lambda x: x[1]
)

overlay_type = st.sidebar.radio(
    "Select Overlay", 
    ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"]
)

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
            display_image(original_image, caption="Original Image")
