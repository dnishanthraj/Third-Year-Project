# ---------------------------------------------------
# IMPORTS AND PAGE SETUP
# ---------------------------------------------------
import streamlit as st
import os
import numpy as np
from components import (
    find_file_in_subfolder, load_npy, display_overlay, display_zoomable_image_with_annotation,
    display_scores_table, parse_filenames, sort_patients, find_file_in_dir, display_nodule_classification_overlay
)
import components.constants as const
from components.constants import IMAGE_DIR, MASK_DIR, MODEL_OUTPUTS_BASE_DIR

# Streamlit page configuration
st.set_page_config(page_title="Slice Viewer", layout="wide")

# Custom fonts and layout styling
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
[data-testid="stAppViewContainer"] *, [data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
}
.block-container {
    max-width: 1400px !important;
    padding: 3rem 2rem !important;
}
[data-testid="stSidebar"] .streamlit-expanderHeader {
    margin: 0.2rem 0 !important;
}
[data-testid="stSidebarNav"] ul li {
    margin-bottom: 0.2rem;
}
[data-testid="stSidebarNav"] ul li a {
    padding: 0.3rem 1rem;
    border-radius: 4px;
}
[data-testid="stSidebar"] h2 {
    margin-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------
# 0) PAGE TITLE & INTRO
# ---------------------------------------------------
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

st.markdown("""
**Note:**  
The FP classifier has been applied to the clean dataset, so false positives have been removed.  
As a result, only the **No Clean** (raw) dataset is displayed for overlays and nodule classification, 
ensuring more meaningful visualization.
""")


# ---------------------------------------------------
# 1) PATH SETUP
# ---------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Sidebar heading
st.sidebar.markdown("## Sidebar Settings")


# ---------------------------------------------------
# 2) SIDEBAR: MODEL FOLDER SELECTION
# ---------------------------------------------------
with st.sidebar.expander("Select Model Folder", expanded=True):
    st.write("Pick the model folder whose slices/Grad-CAM data you want to view.")
    
    available_folders = [
        f for f in os.listdir(MODEL_OUTPUTS_BASE_DIR)
        if os.path.isdir(os.path.join(MODEL_OUTPUTS_BASE_DIR, f))
    ]
    
    selected_folder = st.selectbox("Model Folder:", available_folders)

# Early exit if no folder is selected
if not selected_folder:
    st.warning("No folder selected. Please pick one from the sidebar.")
    st.stop()

# Build paths dynamically based on the selected model folder
folder_base_path = os.path.join(const.MODEL_OUTPUTS_BASE_DIR, selected_folder, "Segmentation_output")
grad_cam_base_path = os.path.join(const.MODEL_OUTPUTS_BASE_DIR, selected_folder, "Grad_CAM_output")
METRICS_DIR = os.path.join(const.MODEL_OUTPUTS_BASE_DIR, selected_folder, "metrics")  # If needed later

# Find subfolders (distinguish Clean vs. No Clean datasets)
segmentation_subfolders = [sub for sub in os.listdir(folder_base_path) if os.path.isdir(os.path.join(folder_base_path, sub))]
grad_cam_subfolders = [sub for sub in os.listdir(grad_cam_base_path) if os.path.isdir(os.path.join(grad_cam_base_path, sub))]

# Select the non-clean dataset subfolders
non_clean_segmentation_folder = next((sub for sub in segmentation_subfolders if not sub.startswith("CLEAN")), None)
non_clean_grad_cam_folder = next((sub for sub in grad_cam_subfolders if not sub.startswith("CLEAN")), None)

# Final output paths
OUTPUT_MASK_DIR = os.path.join(folder_base_path, non_clean_segmentation_folder)
GRAD_CAM_DIR = os.path.join(grad_cam_base_path, non_clean_grad_cam_folder)
prefix = "PD"

# Patch constants dynamically
const.OUTPUT_MASK_DIR = OUTPUT_MASK_DIR
const.GRAD_CAM_DIR = GRAD_CAM_DIR
const.METRICS_DIR = METRICS_DIR

# ---------------------------------------------------
# 4) GATHER .NPY FILES
# ---------------------------------------------------
output_mask_files = [f for f in os.listdir(OUTPUT_MASK_DIR) if f.endswith('.npy')]
grad_cam_files = [f for f in os.listdir(GRAD_CAM_DIR) if f.endswith('.npy')]

# Parse filenames into structured dictionaries: {patient_id: {region_id: [(file_path, slice_name), ...]}}
output_masks_by_patient = sort_patients(parse_filenames(output_mask_files, prefix="PD"))
grad_cams_by_patient = sort_patients(parse_filenames(grad_cam_files, prefix="GC"))

# If no output mask files found, stop the app
if not output_masks_by_patient:
    st.warning("No patient data found in this dataset/folder.")
    st.stop()


# ---------------------------------------------------
# 5) SIDEBAR: EXPLORE SLICES (PATIENT, REGION, SLICE, OVERLAY, ZOOM)
# ---------------------------------------------------
with st.sidebar.expander("Explore Slices", expanded=False):
    st.write("Pick your **Patient**, **Region**, and **Slice**, then choose an **Overlay**.")

    # Patient selection
    all_patients = sorted(output_masks_by_patient.keys(), key=lambda x: int(x))
    selected_patient_list = st.multiselect(
        "Search or Pick Patient:",
        all_patients,
        default=[all_patients[0]] if all_patients else [],
        max_selections=1
    )

    # If no patient selected, stop
    if not selected_patient_list:
        st.warning("Please select at least one patient.")
        st.stop()

    selected_patient = selected_patient_list[0]

    # Region selection
    all_regions = output_masks_by_patient[selected_patient].keys()
    sorted_regions = sorted(all_regions, key=lambda x: int(''.join(filter(str.isdigit, x))))
    selected_region = st.selectbox(
        "Select Region",
        sorted_regions,
        format_func=lambda x: ''.join(filter(str.isdigit, x))  # display just the number
    )

    # Slice selection
    sorted_slices = sorted(
        output_masks_by_patient[selected_patient][selected_region],
        key=lambda x: int(x[1].split()[-1])  # sort by slice number
    )
    selected_slice = st.selectbox(
        "Select Slice",
        sorted_slices,
        format_func=lambda x: x[1]  # show slice name
    )

    # Overlay and Zoom settings
    overlay_type = st.radio("Select Overlay", ["None", "Grad-CAM Heatmap", "Ground Truth Mask", "Predicted Mask", "Nodule Classification"])
    zoom_factor = st.slider("Zoom Factor", 1.0, 10.0, 2.0, 0.1)


# ---------------------------------------------------
# 6) MAIN PAGE LAYOUT -> 2 COLUMNS (SLICE + ANNOTATION, METRICS)
# ---------------------------------------------------
col_left, col_right = st.columns([2, 1], gap="large")

slice_index = selected_slice[1].split()[-1]

# Left column - Slice and overlay viewer
col_left.header("View Zoomable Overlay")
col_left.markdown("""
Below is the final merged slice (or the original slice if no overlay was chosen).
You can zoom in/out here.
""")
col_left.subheader(f"Patient {int(selected_patient)} | Region {int(''.join(filter(str.isdigit, selected_region)))} | Slice {slice_index}")

overlay_code = None
if overlay_type == "Grad-CAM Heatmap":
    overlay_code = "GC"
elif overlay_type == "Ground Truth Mask":
    overlay_code = "MA"
elif overlay_type == "Predicted Mask":
    overlay_code = "PD"

# --- Display image depending on selected overlay ---
with col_left:
    if overlay_type == "Nodule Classification":
        # New function for bounding box overlay: TP/FP/FN annotations
        display_nodule_classification_overlay(selected_patient, selected_region, slice_index, zoom_factor)

    elif overlay_type != "None":
        # Grad-CAM / Ground Truth / Predicted Mask overlays
        if overlay_type == "Grad-CAM Heatmap":
            st.markdown("""
            **Grad-CAM Heatmap Overlay**  
            Highlights important areas the model used for prediction (warmer colours mean higher importance).
            """)
        elif overlay_type == "Ground Truth Mask":
            st.markdown("""
            **Ground Truth Mask Overlay**  
            Shows the true annotated regions for the slice (the "correct" segmentation).
            """)
        elif overlay_type == "Predicted Mask":
            st.markdown("""
            **Predicted Mask Overlay**  
            Displays the model's segmentation prediction on top of the original slice.
            """)
        display_overlay(selected_patient, selected_region, slice_index, overlay_code, zoom_factor)

    else:
        # No overlay: Show the original CT slice with zoom/annotation options
        st.markdown("""
        **No Overlay Selected**  
        Viewing the original CT slice without any overlays.
        Switch between **Zoom Mode** (pan/zoom) and **Annotate Mode** (draw).
        """)
        file_name = f"{selected_patient}_NI{selected_region}_slice{slice_index}.npy"
        original_path = find_file_in_subfolder(IMAGE_DIR, int(selected_patient), file_name)
        original_image = load_npy(original_path) if original_path else None

        if original_image is not None:
            display_zoomable_image_with_annotation(original_image, zoom_factor=zoom_factor, file_name=file_name)
        else:
            st.warning("Original slice not found for this combination.")


# ---------------------------------------------------
# 7) RIGHT COLUMN: SLICE-LEVEL METRICS
# ---------------------------------------------------
with col_right:
    st.markdown("## Slice-Level Metrics")
    st.markdown("""
    Compare the model's predicted mask against ground truth for this slice.
    Displays metrics like Dice, IoU, Precision, and Recall.
    """)

    predicted_file_name = f"{selected_patient}_PD{selected_region}_slice{slice_index}.npy"
    ground_truth_file_name = f"{selected_patient}_MA{selected_region}_slice{slice_index}.npy"

    predicted_path = find_file_in_dir(OUTPUT_MASK_DIR, predicted_file_name)
    ground_truth_path = find_file_in_subfolder(MASK_DIR, int(selected_patient), ground_truth_file_name)

    if predicted_path and ground_truth_path:
        predicted_mask = load_npy(predicted_path)
        ground_truth_mask = load_npy(ground_truth_path)

        if predicted_mask is not None and ground_truth_mask is not None:
            display_scores_table(predicted_mask, ground_truth_mask)
        else:
            st.warning("Failed to load the predicted or ground truth mask.")
    else:
        st.warning("Predicted or ground truth file path could not be found.")


# ---------------------------------------------------
# FINAL TIPS
# ---------------------------------------------------
st.markdown("---")
st.markdown("""
**Tip:**  
- Use **No Clean** vs **Clean** toggle (if supported) to inspect dataset variations.  
- Adjust the **Zoom Factor** in sidebar for better inspection.  
- Switch overlays to explore different layers: Grad-CAM vs Ground Truth vs Predictions.
""")
