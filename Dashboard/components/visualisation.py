# -------------------------------------------------------
# Visualization Utilities for Lung Nodule Dashboard
# -------------------------------------------------------

import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import os
import streamlit as st
from streamlit_image_zoom import image_zoom
import components.constants as const  # Import project constants
from skimage.measure import label, regionprops
from .file_utils import find_file_in_subfolder, export_file, find_file_in_dir
from .image_utils import load_npy, combine_images

# -------------------------------------------------------
# Annotate nodules as TP / FP / FN on a slice
# -------------------------------------------------------
def annotate_nodules(base_image, pred_mask, gt_mask, distance_threshold=80):
    """
    Annotates the base image with bounding boxes and labels for each predicted nodule.
    A predicted region is labeled as TP if its center is near any ground truth region,
    otherwise labeled as FP. Missed ground truth nodules are labeled FN.
    Returns the annotated image as a numpy array.
    """

    # Convert base image to RGB if necessary
    if len(base_image.shape) == 2:
        base_rgb = np.stack([base_image] * 3, axis=-1)
    else:
        base_rgb = base_image.copy()

    # Normalize and convert to uint8
    if base_rgb.dtype != np.uint8:
        base_rgb = ((base_rgb - base_rgb.min()) / (base_rgb.max() - base_rgb.min()) * 255).astype(np.uint8)
    
    annotated_img = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(annotated_img)

    # Label predicted regions
    pred_labeled = label(pred_mask)
    pred_regions = regionprops(pred_labeled)

    # Label ground truth regions
    gt_labeled = label(gt_mask)
    gt_regions = regionprops(gt_labeled)
    gt_centroids = [region.centroid for region in gt_regions]

    # Annotate predictions
    for region in pred_regions:
        bbox = region.bbox
        pred_centroid = region.centroid
        is_tp = any(np.linalg.norm(np.array(pred_centroid) - np.array(gt_centroid)) < distance_threshold
                    for gt_centroid in gt_centroids)
        label_text = "TP" if is_tp else "FP"
        color = (0, 255, 0) if is_tp else (255, 0, 0)
        draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline=color, width=2)
        draw.text((bbox[1], max(bbox[0]-10, 0)), label_text, fill=color)

    # Annotate missed ground truth nodules (FN)
    for region in gt_regions:
        bbox = region.bbox
        gt_centroid = region.centroid
        matched = any(np.linalg.norm(np.array(region_pred.centroid) - np.array(gt_centroid)) < distance_threshold
                      for region_pred in pred_regions)
        if not matched:
            label_text = "FN"
            color = (0, 0, 255)
            draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline=color, width=2)
            draw.text((bbox[1], max(bbox[0]-10, 0)), label_text, fill=color)

    return np.array(annotated_img)

# -------------------------------------------------------
# Display annotated slice with TP / FP / FN classification
# -------------------------------------------------------
def display_nodule_classification_overlay(patient_id, region_id, slice_name, zoom_factor, distance_threshold=80):
    """
    Displays the nodule classification (TP, FP, FN) overlay on a given slice
    with zooming and annotation capabilities.
    """

    # Show legend
    st.markdown("""
    **Nodule Classification Legend**  
    - <span style="color:green;font-weight:bold;">Green Box (TP)</span>: Correct prediction  
    - <span style="color:red;font-weight:bold;">Red Box (FP)</span>: False positive  
    - <span style="color:blue;font-weight:bold;">Blue Box (FN)</span>: Missed nodule
    """, unsafe_allow_html=True)

    # Load file paths
    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    predicted_file_name = f"{patient_id}_PD{region_id}_slice{slice_name}.npy"
    ground_truth_file_name = f"{patient_id}_MA{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(const.IMAGE_DIR, int(patient_id), original_file_name)
    predicted_path = os.path.join(const.OUTPUT_MASK_DIR, predicted_file_name)
    ground_truth_path = find_file_in_subfolder(const.MASK_DIR, int(patient_id), ground_truth_file_name)

    base_image = load_npy(original_path)
    pred_mask = load_npy(predicted_path) if predicted_path and os.path.exists(predicted_path) else None
    gt_mask = load_npy(ground_truth_path) if ground_truth_path and os.path.exists(ground_truth_path) else None

    if base_image is None or pred_mask is None or gt_mask is None:
        st.warning("One or more required files not found.")
        return

    # Annotate and display
    annotated_image = annotate_nodules(base_image, pred_mask, gt_mask, distance_threshold)
    display_zoomable_image_with_annotation(
        base_image=annotated_image,
        overlay=None,
        overlay_type="Nodule Classification",
        zoom_factor=zoom_factor,
        file_name=f"{patient_id}_region{region_id}_slice{slice_name}_nodule_classification"
    )

# -------------------------------------------------------
# Display selected overlay (Grad-CAM, Mask, Predicted Mask)
# -------------------------------------------------------
def display_overlay(patient_id, region_id, slice_name, overlay_type, zoom_factor):
    """
    Display the specified overlay (Ground Truth, Predicted Mask, Grad-CAM) 
    combined with the original image for better visualization.
    """

    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(const.IMAGE_DIR, int(patient_id), original_file_name)
    if overlay_type == "MA":
        overlay_path = find_file_in_subfolder(const.MASK_DIR, int(patient_id), overlay_file_name)
    elif overlay_type == "GC":
        overlay_path = find_file_in_dir(const.GRAD_CAM_DIR, overlay_file_name)
    else:
        overlay_path = find_file_in_dir(const.OUTPUT_MASK_DIR, overlay_file_name)

    original_image = load_npy(original_path)
    overlay_image = load_npy(overlay_path) if overlay_path and os.path.exists(overlay_path) else None

    if original_image is not None:
        display_zoomable_image_with_annotation(
            base_image=original_image,
            overlay=overlay_image,
            overlay_type=overlay_type,
            zoom_factor=zoom_factor,
            file_name=f"{patient_id}_region{region_id}_slice{slice_name}"
        )
    else:
        st.warning("Original image not found.")

# -------------------------------------------------------
# Core Function: Display image with zoom + annotation + export
# -------------------------------------------------------
def display_zoomable_image_with_annotation(base_image, overlay=None, overlay_type=None, zoom_factor=1.0, file_name="exported_image"):
    """
    Zoom, annotate, and export images interactively inside Streamlit.
    """

    # Normalize and scale image
    base_image_min = base_image.min()
    base_image_max = base_image.max() if base_image.max() != base_image_min else (base_image_min + 1)
    base_image_normalized = (base_image - base_image_min) / (base_image_max - base_image_min)
    base_image_uint8 = (base_image_normalized * 255).astype(np.uint8)

    # Merge overlay if provided
    combined_image = combine_images(base_image_uint8, overlay, overlay_type) if overlay is not None else base_image_uint8

    # Convert grayscale to RGB if needed
    if len(combined_image.shape) == 2:
        combined_image = np.stack([combined_image]*3, axis=-1)

    combined_image_float = combined_image.astype(np.float32) / 255.0

    # Display zoomable image
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)
    try:
        image_zoom(combined_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
    except Exception as e:
        st.error(f"Error in image zoom functionality: {e}")
    st.write("</div>", unsafe_allow_html=True)

    # Annotation tools sidebar
    with st.sidebar.expander("Annotation Tools", expanded=False):
        drawing_mode = st.selectbox("Drawing Tool:", (
            "freedraw", "line", "rect", "circle", "polygon", "point", "transform"
        ))
        stroke_width = st.slider("Stroke Width:", 1, 25, 3)
        stroke_color = st.color_picker("Stroke Color:", "#FF0000")
        realtime_update = st.checkbox("Realtime Update", True)

    # Allow user to annotate
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=Image.fromarray(combined_image),
        update_streamlit=realtime_update,
        height=combined_image.shape[0],
        width=combined_image.shape[1],
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="annotation_canvas",
    )

    # Alpha blend the drawn annotations if present
    blended_rgb = combined_image_float
    if canvas_result.image_data is not None:
        annotated_image = np.array(canvas_result.image_data, dtype=np.uint8)
        st.image(annotated_image, caption="Annotated Image")

        annotated_image_resized = np.array(
            Image.fromarray(annotated_image).resize((combined_image.shape[1], combined_image.shape[0]))
        )
        annot_resized_float = annotated_image_resized.astype(np.float32) / 255.0
        alpha = annot_resized_float[..., 3:4]
        annot_rgb = annot_resized_float[..., :3]
        blended_rgb = alpha * annot_rgb + (1.0 - alpha) * combined_image_float

    # Export final annotated result
    export_file(blended_rgb, "npy", file_name)
    export_file((blended_rgb * 255).astype(np.uint8), "png", file_name)

# -------------------------------------------------------
# Utility: Lighten a hex color by a factor
# -------------------------------------------------------
def lighten_color(hex_color, factor=0.5):
    """
    Lightens the given hex color toward white.
    """
    import re
    factor = max(min(factor, 1.0), 0.0)
    hex_color = hex_color.strip("#")
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r)*factor)
    g = int(g + (255 - g)*factor)
    b = int(b + (255 - b)*factor)

    return "#{:02x}{:02x}{:02x}".format(r, g, b)

# -------------------------------------------------------
# Utility: Simple Zoom or Annotate Toggle
# -------------------------------------------------------
def display_zoom_and_annotate(base_image, zoom_factor=1.0, file_name="exported_image"):
    """
    Display base image in either Zoom mode or Annotate mode.
    """

    if base_image.dtype != np.uint8:
        base_image = ((base_image - base_image.min()) / (base_image.max() - base_image.min()) * 255).astype(np.uint8)

    mode = st.sidebar.radio("Select View Mode:", ["Zoom", "Annotate"], key="zoom_annotate_mode")

    if mode == "Zoom":
        st.markdown("### Zoom Mode")
        try:
            image_zoom(base_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
        except Exception as e:
            st.error(f"Error in zoom functionality: {e}")
    else:
        st.markdown("### Annotate Mode")
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=3,
            stroke_color="#FF0000",
            background_image=Image.fromarray(base_image),
            update_streamlit=True,
            height=base_image.shape[0],
            width=base_image.shape[1],
            drawing_mode="freedraw",
            display_toolbar=True,
            key="annotation_canvas_toggle"
        )
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data, caption="Annotated Image", use_column_width=True)
