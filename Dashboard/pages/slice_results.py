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

    # Placeholder values for Dice Score and IoU Score
    dice_score = 0.0  # Replace with actual computation
    iou_score = 0.0  # Replace with actual computation

    # Display the table below the image section
    display_scores_table(dice_score, iou_score)
