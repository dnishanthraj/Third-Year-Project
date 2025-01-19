import streamlit as st
import os
import numpy as np
from PIL import Image
from streamlit_image_zoom import image_zoom
import matplotlib.pyplot as plt
import io

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

def export_file(data, file_type, file_name):
    """Export data as a file for download."""
    if file_type == "npy":
        buffer = io.BytesIO()
        np.save(buffer, data)
        buffer.seek(0)
        st.download_button(
            label="Download as .npy",
            data=buffer,
            file_name=f"{file_name}.npy",
            mime="application/octet-stream",
        )
    elif file_type == "png":
        image = Image.fromarray(data)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        st.download_button(
            label="Download as .png",
            data=buffer,
            file_name=f"{file_name}.png",
            mime="image/png",
        )

def display_zoomable_image(base_image, overlay=None, overlay_type=None, zoom_factor=2.0, file_name="exported_image"):
    """Display an image with zoom functionality using streamlit-image-zoom and export options."""
    # Combine images
    combined_image = combine_images(base_image, overlay, overlay_type)

    # Convert to PIL Image
    pil_image = Image.fromarray(combined_image)

    # Display image with zoom
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)  # Center the image
    image_zoom(pil_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
    st.write("</div>", unsafe_allow_html=True)

    # Add export buttons
    st.subheader("Export Options")
    export_file(combined_image, "png", file_name)
    export_file(base_image, "npy", f"{file_name}_original")

def find_file_in_subfolder(base_dir, patient_id, file_name):
    """Search for the correct .npy file within the patient-specific subfolder."""
    subfolder = os.path.join(base_dir, f"LIDC-IDRI-{patient_id:04d}")
    file_path = os.path.join(subfolder, file_name)
    return file_path if os.path.exists(file_path) else None

def display_overlay(patient_id, region_id, slice_name, overlay_type, zoom_factor):
    """Display an overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask) on the Original Image."""
    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), original_file_name)
    overlay_path = (
        find_file_in_subfolder(MASK_DIR, int(patient_id), overlay_file_name) if overlay_type == "MA"
        else os.path.join(GRAD_CAM_DIR if overlay_type == "GC" else OUTPUT_MASK_DIR, overlay_file_name)
    )

    original_image = load_npy(original_path)
    overlay_image = load_npy(overlay_path) if overlay_path and os.path.exists(overlay_path) else None

    if original_image is not None:
        display_zoomable_image(original_image, overlay_image, overlay_type, zoom_factor, f"{patient_id}_region{region_id}_slice{slice_name}")
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

# Sidebar for selecting patient, region, slice, overlay type, and zoom factor
st.sidebar.title("Available Files")

# Searchable dropdown for patients
selected_patient = st.sidebar.multiselect(
    "Search Patient", 
    sorted(output_masks_by_patient.keys(), key=lambda x: int(x)),
    default=[sorted(output_masks_by_patient.keys(), key=lambda x: int(x))[0]],
    max_selections=1,  # Only one patient can be selected
)

# Check if a patient is selected
if selected_patient:
    selected_patient = selected_patient[0]
    
    # Display region and slice dropdowns
    sorted_regions = sorted(output_masks_by_patient[selected_patient].keys(), key=lambda x: int(x))
    selected_region = st.sidebar.selectbox("Select Region", sorted_regions)
    
    sorted_slices = sorted(output_masks_by_patient[selected_patient][selected_region], key=lambda x: int(x[1].split()[-1]))
    selected_slice = st.sidebar.selectbox("Select Slice", sorted_slices, format_func=lambda x: x[1])

    # Add overlay and zoom options
    overlay_type = st.sidebar.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"])
    zoom_factor = st.sidebar.slider("Zoom Factor", min_value=1.0, max_value=10.0, step=0.1, value=2.0)

    # Display the selected overlay
    slice_index = selected_slice[1].split()[-1]
    st.header(f"Patient {int(selected_patient)} | Region {int(selected_region)} | Slice {slice_index}")
    if overlay_type != "None":
        overlay_code = "GC" if overlay_type == "Grad-CAM Heatmap" else "MA" if overlay_type == "Ground Truth Mask" else "PD"
        display_overlay(selected_patient, selected_region, slice_index, overlay_code, zoom_factor)
    else:
        file_name = f"{selected_patient}_NI{selected_region}_slice{slice_index}.npy"
        original_path = find_file_in_subfolder(IMAGE_DIR, int(selected_patient), file_name)
        original_image = load_npy(original_path) if original_path else None
        if original_image is not None:
            display_zoomable_image(original_image, zoom_factor=zoom_factor, file_name=file_name)
else:
    st.sidebar.warning("Please select a patient to proceed.")
