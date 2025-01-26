import streamlit as st
import os
import numpy as np
from components import *

# Static Header Title
st.title("Explainable XAI for Lung Nodule Segmentation - Slice Results")

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

if selected_patient:
    selected_patient = selected_patient[0]
    sorted_regions = sorted(output_masks_by_patient[selected_patient].keys(), key=lambda x: int(x))
    selected_region = st.sidebar.selectbox("Select Region", sorted_regions)
    sorted_slices = sorted(output_masks_by_patient[selected_patient][selected_region], key=lambda x: int(x[1].split()[-1]))
    selected_slice = st.sidebar.selectbox("Select Slice", sorted_slices, format_func=lambda x: x[1])

    overlay_type = st.sidebar.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"])
    zoom_factor = st.sidebar.slider("Zoom Factor", min_value=1.0, max_value=10.0, step=0.1, value=2.0)

    slice_index = selected_slice[1].split()[-1]
    st.header(f"Patient {int(selected_patient)} | Region {int(selected_region)} | Slice {slice_index}")

    overlay_code = "GC" if overlay_type == "Grad-CAM Heatmap" else "MA" if overlay_type == "Ground Truth Mask" else "PD" if overlay_type == "Predicted Mask" else None
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
    predicted_file_name = f"{selected_patient}_PD{selected_region}_slice{slice_index}.npy"
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


