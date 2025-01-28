import streamlit as st
import os
import numpy as np
from components import *

# Static Header Title
st.title("Explainable XAI for Lung Nodule Segmentation - Slice Results")

# Dynamically resolve the paths to the `model_outputs` folder
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "Project", "Segmentation", "model_outputs")

# List all folders under `model_outputs`
available_folders = [f for f in os.listdir(MODEL_OUTPUTS_DIR) if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, f))]

# Add a dropdown to select the desired folder
selected_folder = st.sidebar.selectbox("Select Model Folder", available_folders)

if selected_folder:
    # Dynamically construct paths for Segmentation_output and Grad_CAM_output
    folder_base_path = os.path.join(MODEL_OUTPUTS_DIR, selected_folder, "Segmentation_output")
    grad_cam_base_path = os.path.join(MODEL_OUTPUTS_DIR, selected_folder, "Grad_CAM_output")

    # List subfolders in both Segmentation_output and Grad_CAM_output directories
    segmentation_subfolders = [
        subfolder for subfolder in os.listdir(folder_base_path)
        if os.path.isdir(os.path.join(folder_base_path, subfolder))
    ]
    grad_cam_subfolders = [
        subfolder for subfolder in os.listdir(grad_cam_base_path)
        if os.path.isdir(os.path.join(grad_cam_base_path, subfolder))
    ]

    # Identify CLEAN and No CLEAN folders
    clean_segmentation_folder = next((sub for sub in segmentation_subfolders if sub[:5] == "CLEAN"), None)
    non_clean_segmentation_folder = next((sub for sub in segmentation_subfolders if sub[:5] != "CLEAN"), None)

    clean_grad_cam_folder = next((sub for sub in grad_cam_subfolders if sub[:5] == "CLEAN"), None)
    non_clean_grad_cam_folder = next((sub for sub in grad_cam_subfolders if sub[:5] != "CLEAN"), None)

    # Radio button to toggle between Clean and No Clean
    clean_option = st.sidebar.radio("Select Dataset Type", ["No Clean", "Clean"])

    # Set paths based on the selected option
    if clean_option == "Clean" and clean_segmentation_folder and clean_grad_cam_folder:
        OUTPUT_MASK_DIR = os.path.join(folder_base_path, clean_segmentation_folder)
        GRAD_CAM_DIR = os.path.join(grad_cam_base_path, clean_grad_cam_folder)
        prefix = "CN"
    elif clean_option == "No Clean" and non_clean_segmentation_folder and non_clean_grad_cam_folder:
        OUTPUT_MASK_DIR = os.path.join(folder_base_path, non_clean_segmentation_folder)
        GRAD_CAM_DIR = os.path.join(grad_cam_base_path, non_clean_grad_cam_folder)
        prefix = "PD"
    else:
        st.error("The selected dataset type does not exist in the chosen folder.")
        st.stop()

# Collect `.npy` files dynamically
output_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('.npy')]
grad_cam_files = [f for f in os.listdir(GRAD_CAM_DIR) if f.endswith('.npy')]

# Parse and sort filenames
output_masks_by_patient = sort_patients(parse_filenames(output_mask_files, prefix=prefix))
grad_cams_by_patient = sort_patients(parse_filenames(grad_cam_files, prefix="GC"))

# Sidebar for selecting patient, region, slice, overlay type, and zoom factor
st.sidebar.title("Available Files")

if output_masks_by_patient:
    # Searchable dropdown for patients
    selected_patient = st.sidebar.multiselect(
        "Search Patient",
        sorted(output_masks_by_patient.keys(), key=lambda x: int(x)),
        default=[sorted(output_masks_by_patient.keys(), key=lambda x: int(x))[0]],
        max_selections=1,  # Only one patient can be selected
    )
else:
    st.sidebar.warning("No patients available in the selected folder.")
    st.stop()

if selected_patient:
    selected_patient = selected_patient[0]
    # Strip non-numeric characters from region keys for sorting
    sorted_regions = sorted(
    output_masks_by_patient[selected_patient].keys(),
    key=lambda x: int(''.join(filter(str.isdigit, x)))  # Extract only numeric parts for sorting
    )

    selected_region = st.sidebar.selectbox(
    "Select Region", sorted_regions, format_func=lambda x: ''.join(filter(str.isdigit, x))
    )

    sorted_slices = sorted(
        output_masks_by_patient[selected_patient][selected_region],
        key=lambda x: int(x[1].split()[-1])
    )
    selected_slice = st.sidebar.selectbox("Select Slice", sorted_slices, format_func=lambda x: x[1])

    overlay_type = st.sidebar.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"])
    zoom_factor = st.sidebar.slider("Zoom Factor", min_value=1.0, max_value=10.0, step=0.1, value=2.0)

    slice_index = selected_slice[1].split()[-1]
    st.header(f"Patient {int(selected_patient)} | Region {int(''.join(filter(str.isdigit, selected_region)))} | Slice {slice_index}")


    overlay_code = "GC" if overlay_type == "Grad-CAM Heatmap" else "MA" if overlay_type == "Ground Truth Mask" else prefix if overlay_type == "Predicted Mask" else None
    if overlay_type != "None":
        display_overlay(selected_patient, selected_region, slice_index, overlay_code, zoom_factor)
    else:
        file_name = f"{selected_patient}_NI{selected_region}_slice{slice_index}.npy"
        original_path = find_file_in_subfolder(IMAGE_DIR, int(selected_patient), file_name)
        original_image = load_npy(original_path) if original_path else None
        if original_image is not None:
            display_zoomable_image_with_annotation(original_image, file_name=file_name)
else:
    st.sidebar.warning("Please select a patient to proceed.")

# After displaying the overlay or original image, add the table
if selected_patient:
    slice_index = selected_slice[1].split()[-1]

    # Load Predicted Mask and Ground Truth Mask
    predicted_file_name = f"{selected_patient}_{prefix}{selected_region}_slice{slice_index}.npy"
    ground_truth_file_name = f"{selected_patient}_MA{selected_region}_slice{slice_index}.npy"

    # Locate predicted mask directly in OUTPUT_MASK_DIR
    predicted_path = find_file_in_dir(OUTPUT_MASK_DIR, predicted_file_name)

    # Locate ground truth mask in patient-specific subfolder of MASK_DIR
    ground_truth_path = find_file_in_subfolder(MASK_DIR, int(selected_patient), ground_truth_file_name)

    if predicted_path and ground_truth_path:
        predicted_mask = load_npy(predicted_path)
        ground_truth_mask = load_npy(ground_truth_path)

        if predicted_mask is not None and ground_truth_mask is not None:
            # Display calculated metrics
            display_scores_table(predicted_mask, ground_truth_mask)
        else:
            st.warning("Failed to load predicted or ground truth mask from file.")
    else:
        st.warning("Predicted or Ground Truth Mask file path not found.")
