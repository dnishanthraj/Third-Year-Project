import streamlit as st
import os
import numpy as np
from components import (
    find_file_in_subfolder, load_npy, display_overlay, display_zoomable_image_with_annotation,
    display_scores_table, parse_filenames, sort_patients, find_file_in_dir
)
from components.constants import IMAGE_DIR, MASK_DIR, OUTPUT_MASK_DIR, GRAD_CAM_DIR

st.set_page_config(page_title="Slice Viewer", layout="wide")

################################################################################
# 0) PAGE TITLE & INTRO
################################################################################
st.title("Slice Viewer")
st.markdown("---")

st.markdown("""
**Instructions:**  
1. Choose a **Model Folder** on the left to load its predictions and Grad-CAM data.  
2. Select **No Clean** or **Clean** to switch datasets.  
3. Pick a **Patient**, **Region**, and **Slice** to visualize.  
4. Choose **Overlay** type (e.g., Grad-CAM, Ground Truth, or Predicted Mask).  
5. Adjust the **Zoom Factor** if the slice is too large or small.

Below, your chosen slice will be displayed, along with a small table 
of slice-level metrics (Dice, IoU, etc.) for that slice's predicted vs. ground-truth mask.
""")

################################################################################
# 1) PATH SETUP
################################################################################
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_OUTPUTS_DIR = os.path.join(ROOT_DIR, "Project", "Segmentation", "model_outputs")

################################################################################
# 2) SIDEBAR: MODEL FOLDER SELECTION
################################################################################
with st.sidebar.expander("Select Model Folder", expanded=True):
    st.write("Pick the model folder whose slices/Grad-CAM data you want to view.")
    available_folders = [
        f for f in os.listdir(MODEL_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(MODEL_OUTPUTS_DIR, f))
    ]
    selected_folder = st.selectbox("Model Folder:", available_folders)

if not selected_folder:
    st.warning("No folder selected. Please pick one from the sidebar.")
    st.stop()

folder_base_path = os.path.join(MODEL_OUTPUTS_DIR, selected_folder, "Segmentation_output")
grad_cam_base_path = os.path.join(MODEL_OUTPUTS_DIR, selected_folder, "Grad_CAM_output")

segmentation_subfolders = [
    subfolder for subfolder in os.listdir(folder_base_path)
    if os.path.isdir(os.path.join(folder_base_path, subfolder))
]
grad_cam_subfolders = [
    subfolder for subfolder in os.listdir(grad_cam_base_path)
    if os.path.isdir(os.path.join(grad_cam_base_path, subfolder))
]

# Identify CLEAN or Non-CLEAN subfolders
clean_segmentation_folder = next((sub for sub in segmentation_subfolders if sub.startswith("CLEAN")), None)
non_clean_segmentation_folder = next((sub for sub in segmentation_subfolders if not sub.startswith("CLEAN")), None)

clean_grad_cam_folder = next((sub for sub in grad_cam_subfolders if sub.startswith("CLEAN")), None)
non_clean_grad_cam_folder = next((sub for sub in grad_cam_subfolders if not sub.startswith("CLEAN")), None)

################################################################################
# 3) SIDEBAR: DATASET TYPE SELECTION (NO CLEAN / CLEAN)
################################################################################
with st.sidebar.expander("Select Dataset Type", expanded=False):
    st.write("Choose whether to view **No Clean** or **Clean** slices.")
    clean_option = st.radio("Dataset Type:", ["No Clean", "Clean"])

# Assign prefix + directories based on user choice
if clean_option == "Clean" and clean_segmentation_folder and clean_grad_cam_folder:
    OUTPUT_MASK_DIR = os.path.join(folder_base_path, clean_segmentation_folder)
    GRAD_CAM_DIR = os.path.join(grad_cam_base_path, clean_grad_cam_folder)
    prefix = "CN"
elif clean_option == "No Clean" and non_clean_segmentation_folder and non_clean_grad_cam_folder:
    OUTPUT_MASK_DIR = os.path.join(folder_base_path, non_clean_segmentation_folder)
    GRAD_CAM_DIR = os.path.join(grad_cam_base_path, non_clean_grad_cam_folder)
    prefix = "PD"
else:
    st.error("The selected dataset type does not exist for this model folder.")
    st.stop()

################################################################################
# 4) GATHER .NPY FILES
################################################################################
output_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('.npy')]
grad_cam_files = [f for f in os.listdir(GRAD_CAM_DIR) if f.endswith('.npy')]

output_masks_by_patient = sort_patients(parse_filenames(output_mask_files, prefix=prefix))
grad_cams_by_patient = sort_patients(parse_filenames(grad_cam_files, prefix="GC"))

if not output_masks_by_patient:
    st.warning("No patient data found in this dataset/folder.")
    st.stop()

################################################################################
# 5) SIDEBAR: EXPLORE SLICES (PATIENT, REGION, SLICE, OVERLAY, ZOOM)
################################################################################
with st.sidebar.expander("Explore Slices", expanded=False):
    st.write("Pick your **Patient**, **Region**, and **Slice**, then choose an **Overlay**.")

    # Single patient pick
    all_patients = sorted(output_masks_by_patient.keys(), key=lambda x: int(x))
    selected_patient_list = st.multiselect(
        "Search or Pick Patient:",
        all_patients,
        default=[all_patients[0]] if all_patients else [],
        max_selections=1
    )

    if not selected_patient_list:
        st.warning("Please select at least one patient.")
        st.stop()
    selected_patient = selected_patient_list[0]

    # Sort region
    all_regions = output_masks_by_patient[selected_patient].keys()
    sorted_regions = sorted(all_regions, key=lambda x: int(''.join(filter(str.isdigit, x))))
    selected_region = st.selectbox(
        "Select Region",
        sorted_regions,
        format_func=lambda x: ''.join(filter(str.isdigit, x))
    )

    # Sort slices
    sorted_slices = sorted(
        output_masks_by_patient[selected_patient][selected_region],
        key=lambda x: int(x[1].split()[-1])
    )
    selected_slice = st.selectbox("Select Slice", sorted_slices, format_func=lambda x: x[1])

    overlay_type = st.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask"])
    zoom_factor = st.slider("Zoom Factor", 1.0, 10.0, 2.0, 0.1)

################################################################################
# 6) MAIN PAGE LAYOUT -> 2 COLUMNS (SLICE + ANNOTATION, METRICS)
################################################################################
col_left, col_right = st.columns([2,1], gap="large")

slice_index = selected_slice[1].split()[-1]
col_left.header("1) View Zoomable Overlay")
col_left.markdown("""
Below is the final merged slice (or the original slice if no overlay was chosen).
You can zoom in/out here.
""")
# Provide heading in the left column
col_left.subheader(f"Patient {int(selected_patient)} | Region {int(''.join(filter(str.isdigit, selected_region)))} | Slice {slice_index}")

# SECTION 1: Display the chosen slice (with or without overlay) in left column
overlay_code = None
if overlay_type == "Grad-CAM Heatmap":
    overlay_code = "GC"
elif overlay_type == "Ground Truth Mask":
    overlay_code = "MA"
elif overlay_type == "Predicted Mask":
    overlay_code = prefix

with col_left:
    if overlay_type != "None":
        # Use the overlay-based display
        display_overlay(selected_patient, selected_region, slice_index, overlay_code, zoom_factor)
    else:
        # Show original slice with annotation
        file_name = f"{selected_patient}_NI{selected_region}_slice{slice_index}.npy"
        original_path = find_file_in_subfolder(IMAGE_DIR, int(selected_patient), file_name)
        original_image = load_npy(original_path) if original_path else None
        if original_image is not None:
            display_zoomable_image_with_annotation(original_image, file_name=file_name)
        else:
            st.warning("Original slice not found for this combination.")

# SECTION 2: Display the slice-level metrics in right column
with col_right:
    # --- (Optional) mention "2) Metrics" if you handle them here, or just omit ---
    st.markdown("## 2) Slice-Level Metrics")
    st.markdown("""
    Typically, metrics (e.g. Dice, IoU) are computed outside this file. 
    If you want to display any slice-level metrics here, you can add them 
    before or after the annotation step.
    """)

    predicted_file_name = f"{selected_patient}_{prefix}{selected_region}_slice{slice_index}.npy"
    ground_truth_file_name = f"{selected_patient}_MA{selected_region}_slice{slice_index}.npy"

    predicted_path = find_file_in_dir(OUTPUT_MASK_DIR, predicted_file_name)
    ground_truth_path = find_file_in_subfolder(MASK_DIR, int(selected_patient), ground_truth_file_name)

    if predicted_path and ground_truth_path:
        predicted_mask = load_npy(predicted_path)
        ground_truth_mask = load_npy(ground_truth_path)

        if predicted_mask is not None and ground_truth_mask is not None:
            display_scores_table(predicted_mask, ground_truth_mask)
        else:
            st.warning("Failed to load the predicted or ground truth mask from file.")
    else:
        st.warning("Could not find predicted or ground truth mask file path for this slice.")

# Final tips
st.markdown("---")
st.markdown("""
**Tip:**  
- Switch between **No Clean** and **Clean** to see how the slice data differs.  
- Use **Zoom Factor** in the sidebar to enlarge or shrink the slice view.  
- Toggle **Overlay** to compare Grad-CAM, ground truth, and predicted masks at the slice level.
""")
