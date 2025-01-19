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

def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for the correct .npy file within the patient-specific subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    if os.path.exists(file_path):
        return file_path
    else:
        return None

def display_selected_category(patient_id, region_id, slice_name, display_type):
    """Display the selected category (Original Image, Ground Truth Mask, Predicted Mask, Grad-CAM)."""
    if display_type == "Original Image":
        base_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
        file_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), base_file_name)
        st.subheader("Original Image")
        data = load_npy(file_path) if file_path else None
        if data is not None:
            display_image(data, caption="Original Image", cmap="gray")
    elif display_type == "Ground Truth Mask":
        base_file_name = f"{patient_id}_MA{region_id}_slice{slice_name}.npy"
        file_path = find_file_in_subfolder(MASK_DIR, int(patient_id), base_file_name)
        st.subheader("Ground Truth Mask")
        data = load_npy(file_path) if file_path else None
        if data is not None:
            display_image(data, caption="Ground Truth Mask", cmap="cividis")
    elif display_type == "Predicted Mask":
        file_name = f"{patient_id}_PD{region_id}_slice{slice_name}.npy"
        file_path = os.path.join(OUTPUT_MASK_DIR, file_name)
        st.subheader("Predicted Mask")
        data = load_npy(file_path) if os.path.exists(file_path) else None
        if data is not None:
            display_image(data, caption="Predicted Mask", cmap="cividis")
    elif display_type == "Grad-CAM Heatmap":
        file_name = f"{patient_id}_GC{region_id}_slice{slice_name}.npy"
        file_path = os.path.join(GRAD_CAM_DIR, file_name)
        st.subheader("Grad-CAM Heatmap")
        data = load_npy(file_path) if os.path.exists(file_path) else None
        if data is not None:
            display_image(data, caption="Grad-CAM Heatmap", cmap="jet")

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

# Sidebar for selecting patient, region, slice, and category
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

display_type = st.sidebar.radio(
    "Select Display Category", 
    ["Original Image", "Ground Truth Mask", "Predicted Mask", "Grad-CAM Heatmap"]
)

# Display the selected category
if selected_patient and selected_region and selected_slice:
    slice_index = selected_slice[1].split()[-1]
    st.header(f"Patient {int(selected_patient)} | Region {int(selected_region)} | Slice {slice_index}")
    display_selected_category(selected_patient, selected_region, slice_index, display_type)
