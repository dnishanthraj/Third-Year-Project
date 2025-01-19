import streamlit as st
import os
from PIL import Image
from streamlit_image_zoom import image_zoom

# Paths for demonstration purposes
OUTPUT_MASK_DIR = "../Segmentation/model_outputs/NestedUNET_with_augmentation/Segmentation_output/NestedUNET_with_augmentation"
GRAD_CAM_DIR = "../Segmentation/model_outputs/NestedUNET_with_augmentation/Grad_CAM_output/NestedUNET_with_augmentation"

def load_image(filepath):
    """Load a PNG image."""
    return Image.open(filepath)

def parse_filenames(files, prefix):
    """Parse filenames to group by patient and slices, with a specific prefix (PD or GC)."""
    patients = {}
    for file in files:
        if prefix in file:  # Ensure the file matches the correct type
            parts = file.split("_")
            patient_id = parts[0]
            slice_info = parts[-1].replace(".png", "").replace("slice", "Slice ")
            if patient_id not in patients:
                patients[patient_id] = []
            patients[patient_id].append((file, slice_info))
    return patients

def sort_patients(patients):
    """Sort patients by patient number and their slices by slice number."""
    sorted_patients = {}
    for patient_id, slices in sorted(patients.items(), key=lambda x: int(x[0])):
        sorted_patients[patient_id] = sorted(slices, key=lambda x: int(x[1].split()[-1]))
    return sorted_patients

# Static Header Title
st.title("Explainable XAI for Lung Nodule Segmentation Dashboard")

# Collect PNG files dynamically
output_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('.png')]
grad_cam_files = [f for f in os.listdir(GRAD_CAM_DIR) if f.endswith('.png')]

# Parse and sort filenames
output_masks_by_patient = sort_patients(parse_filenames(output_mask_files, prefix="PD"))
grad_cams_by_patient = sort_patients(parse_filenames(grad_cam_files, prefix="GC"))

# Sidebar for selecting patient and slice
st.sidebar.title("Available Files")
tab = st.sidebar.radio("Select File Category", ("Predicted Masks", "Grad-CAM"))

# Helper function to handle dropdowns dynamically
def select_patient_and_slice(files_by_patient, dir_path, file_prefix):
    # Select a patient
    selected_patient = st.sidebar.selectbox(
        "Select Patient", 
        list(files_by_patient.keys()), 
        format_func=lambda x: f"Patient {int(x)}",
        key=f"{file_prefix}_patient_selector"
    )
    
    # Select a slice for the selected patient
    selected_slice = st.sidebar.selectbox(
        "Select Slice", 
        files_by_patient[selected_patient], 
        format_func=lambda x: x[1],  # Show slice info only
        key=f"{file_prefix}_slice_selector"
    )
    
    # Construct the file path and return
    file_name = selected_slice[0]
    image_path = os.path.join(dir_path, file_name)
    return selected_patient, selected_slice[1], file_name, image_path

# Get the currently selected patient and slice
if tab == "Predicted Masks":
    if output_masks_by_patient:
        selected_patient, selected_slice, selected_file, image_path = select_patient_and_slice(
            output_masks_by_patient, OUTPUT_MASK_DIR, file_prefix="PD"
        )
        if os.path.exists(image_path):
            st.header(f"Patient {int(selected_patient)} | {selected_slice}")
            st.image(load_image(image_path), caption=f"File: {selected_file}", use_column_width=True)
            zoomable_image = image_zoom(image_path)
            st.write("Zoom or pan the image above.")
    else:
        st.warning("No predicted masks found.")

elif tab == "Grad-CAM":
    if grad_cams_by_patient:
        # Use the same patient and slice if already selected in the other tab
        selected_patient, selected_slice, selected_file, image_path = select_patient_and_slice(
            grad_cams_by_patient, GRAD_CAM_DIR, file_prefix="GC"
        )
        if os.path.exists(image_path):
            st.header(f"Patient {int(selected_patient)} | {selected_slice}")
            st.image(load_image(image_path), caption=f"File: {selected_file}", use_column_width=True)
            zoomable_image = image_zoom(image_path)
            st.write("Zoom or pan the image above.")
    else:
        st.warning("No Grad-CAM files found.")
