# components/visualisation.py
import numpy as np
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
import os
import streamlit as st
from streamlit_image_zoom import image_zoom
# from .constants import MASK_DIR, GRAD_CAM_DIR, OUTPUT_MASK_DIR, IMAGE_DIR
import components.constants as const  # Import the entire module
from skimage.measure import label, regionprops
import os
from .file_utils import find_file_in_subfolder, export_file, find_file_in_dir
from .image_utils import load_npy, combine_images




def annotate_nodules(base_image, pred_mask, gt_mask, distance_threshold=80):
    """
    Annotates the base image with bounding boxes and labels for each predicted nodule.
    A predicted region is labeled as TP if its center is within the threshold
    distance of any ground truth region's center; otherwise, FP.
    Additionally, any ground truth region that has no nearby predicted candidate is labeled FN.
    Returns the annotated image as a numpy array.
    """
    # Ensure base image is in RGB
    if len(base_image.shape) == 2:
        base_rgb = np.stack([base_image] * 3, axis=-1)
    else:
        base_rgb = base_image.copy()
    
    # Convert to uint8 if necessary
    if base_rgb.dtype != np.uint8:
        base_rgb = ((base_rgb - base_rgb.min()) / (base_rgb.max() - base_rgb.min()) * 255).astype(np.uint8)
    
    annotated_img = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(annotated_img)

    # Label predicted mask and extract regions
    pred_labeled = label(pred_mask)
    pred_regions = regionprops(pred_labeled)

    # Label ground truth mask and get centroids
    gt_labeled = label(gt_mask)
    gt_regions = regionprops(gt_labeled)
    gt_centroids = [region.centroid for region in gt_regions]

    # For each predicted region, classify as TP or FP based on distance
    for region in pred_regions:
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        pred_centroid = region.centroid
        is_tp = False
        for gt_centroid in gt_centroids:
            dist = np.linalg.norm(np.array(pred_centroid) - np.array(gt_centroid))
            if dist < distance_threshold:
                is_tp = True
                break
        label_text = "TP" if is_tp else "FP"
        color = (0, 255, 0) if is_tp else (255, 0, 0)  # green for TP, red for FP
        draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline=color, width=2)
        draw.text((bbox[1], max(bbox[0]-10, 0)), label_text, fill=color)

    # For each ground truth region, if no predicted candidate is nearby, mark it FN.
    for region in gt_regions:
        bbox = region.bbox
        gt_centroid = region.centroid
        matched = False
        for region_pred in pred_regions:
            pred_centroid = region_pred.centroid
            dist = np.linalg.norm(np.array(pred_centroid) - np.array(gt_centroid))
            if dist < distance_threshold:
                matched = True
                break
        if not matched:
            label_text = "FN"
            color = (0, 0, 255)  # blue for FN
            draw.rectangle([bbox[1], bbox[0], bbox[3], bbox[2]], outline=color, width=2)
            draw.text((bbox[1], max(bbox[0]-10, 0)), label_text, fill=color)
    
    return np.array(annotated_img)


def display_nodule_classification_overlay(patient_id, region_id, slice_name, zoom_factor, distance_threshold=80):
    """
    Loads the original image, predicted mask, and ground truth mask for the given slice,
    annotates detected nodules as TP/FP/FN, and displays the resulting image in the
    zoomable and annotatable viewer.
    """

    # 1) Show a quick explanation/legend for the user
    st.markdown("""
    **Nodule Classification Legend**  
    - <span style="color:green;font-weight:bold;">Green Box (TP)</span>: Model correctly predicted a real nodule  
    - <span style="color:red;font-weight:bold;">Red Box (FP)</span>: Model predicted a nodule that does **not** exist  
    - <span style="color:blue;font-weight:bold;">Blue Box (FN)</span>: A real nodule was **missed** by the model  

    *(Note: True Negatives do not appear because there is no box to draw where neither ground truth nor prediction has a nodule.)*
    """, unsafe_allow_html=True)

    # 2) Build file names (adjust prefix if needed)
    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    predicted_file_name = f"{patient_id}_PD{region_id}_slice{slice_name}.npy"  # assumed predicted prefix "PD"
    ground_truth_file_name = f"{patient_id}_MA{region_id}_slice{slice_name}.npy"

    # 3) Load images
    original_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), original_file_name)
    base_image = load_npy(original_path)
    if base_image is None:
        st.warning("Original image not found.")
        return

    predicted_path = os.path.join(OUTPUT_MASK_DIR, predicted_file_name)
    ground_truth_path = find_file_in_subfolder(MASK_DIR, int(patient_id), ground_truth_file_name)

    pred_mask = load_npy(predicted_path) if predicted_path and os.path.exists(predicted_path) else None
    gt_mask = load_npy(ground_truth_path) if ground_truth_path and os.path.exists(ground_truth_path) else None

    if pred_mask is None or gt_mask is None:
        st.warning("Predicted or Ground Truth mask not found.")
        return

    # 4) Annotate nodules on the original image
    annotated_image = annotate_nodules(base_image, pred_mask, gt_mask, distance_threshold)

    # 5) Display the annotated image in the zoomable viewer
    display_zoomable_image_with_annotation(
        base_image=annotated_image,
        overlay=None,
        overlay_type="Nodule Classification",
        zoom_factor=zoom_factor,
        file_name=f"{patient_id}_region{region_id}_slice{slice_name}_nodule_classification"
    )


def display_overlay(patient_id, region_id, slice_name, overlay_type, zoom_factor):
    """
    Display an overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask) on the Original Image.
    This function:
      1) Loads the base slice
      2) Loads the overlay (depending on overlay_type)
      3) Merges them and calls display_zoomable_image_with_annotation
    """

    # # --- SECTION 1) VIEW OVERLAY ---
    # st.markdown("## 1) View Overlay")
    # st.markdown("""
    # Here, we load the **original slice** from disk, then combine it with the 
    # chosen overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask).  
    # This gives a quick visual comparison to see how the overlay aligns with the slice.
    # """)

    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(const.IMAGE_DIR, int(patient_id), original_file_name)
    if overlay_type == "MA":  # ground truth
        overlay_path = find_file_in_subfolder(const.MASK_DIR, int(patient_id), overlay_file_name)
    elif overlay_type == "GC":  # Grad-CAM
        overlay_path = find_file_in_dir(const.GRAD_CAM_DIR, overlay_file_name)
    else:  # Predicted Mask (PD) case
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

def display_zoomable_image_with_annotation(base_image,
                                           overlay=None,
                                           overlay_type=None,
                                           zoom_factor=1.0,
                                           file_name="exported_image"):
    """
    Displays the merged (or original) image in a zoomable viewer, plus an
    annotation canvas so the user can draw and then export the final annotated image.
    """

    # Normalize base image to [0..1], then scale to [0..255]
    base_image_min = base_image.min()
    base_image_max = base_image.max() if base_image.max() != base_image_min else (base_image_min + 1)  
    base_image_normalized = (base_image - base_image_min) / (base_image_max - base_image_min)

    base_image_uint8 = (base_image_normalized * 255).astype(np.uint8)

    # Combine with overlay if present
    # This yields a uint8 (H, W[, 3]) array
    combined_image = combine_images(base_image_uint8, overlay, overlay_type) if overlay is not None else base_image_uint8

    # We'll make a float version in [0..1], so we can alpha-blend later
    # If combined_image is grayscale => shape (H, W), convert to (H,W,3) for consistency
    if len(combined_image.shape) == 2:
        combined_image = np.stack([combined_image]*3, axis=-1)  # (H, W) -> (H, W, 3)
    combined_image_float = combined_image.astype(np.float32) / 255.0  # [0..1]

    # Show the combined image in a zoomable widget
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)
    try:
        image_zoom(combined_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
    except Exception as e:
        st.error(f"Error in image zoom functionality: {e}")
    st.write("</div>", unsafe_allow_html=True)

    # Sidebar annotation tools
    with st.sidebar.expander("Annotation Tools", expanded=False):
        drawing_mode = st.selectbox("Drawing Tool:", (
            "freedraw", "line", "rect", "circle", "polygon", "point", "transform"
        ))
        stroke_width = st.slider("Stroke Width:", 1, 25, 3)
        stroke_color = st.color_picker("Stroke Color:", "#FF0000")
        realtime_update = st.checkbox("Realtime Update", True)

    # --- SECTION 3) EXPORT & ANNOTATE ---
    st.markdown("## Export & Annotate")
    st.markdown("""
    Use the annotation canvas to draw lines, boxes, polygons, etc.  
    Once you're done, you can export the annotated result as both `.npy` and `.png`.
    """)

    # Create the annotation canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=Image.fromarray(combined_image),  # the background in color
        update_streamlit=realtime_update,
        height=combined_image.shape[0],
        width=combined_image.shape[1],
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="annotation_canvas",
    )

    # Start with a fallback for the final image => no annotation
    # so we can always export a final image
    blended_rgb = combined_image_float  # shape (H, W, 3), float in [0..1]

    # If user drew something
    if canvas_result.image_data is not None:
        annotated_image = np.array(canvas_result.image_data, dtype=np.uint8)
        st.image(annotated_image, caption="Annotated Image")

        # 1) Resize the annotated image to match the original dimension
        annotated_image_resized = np.array(
            Image.fromarray(annotated_image).resize((combined_image.shape[1], combined_image.shape[0]))
        )  # shape (H, W, 4)

        # 2) Convert combined_image_float to (H, W, 4) by adding alpha=1 if needed
        # or we can just assume it's (H, W, 3)
        # We'll do alpha blending manually with the annotation's alpha channel
        # combined_image_float => shape (H, W, 3)
        # annotated_image => shape (H, W, 4)
        annot_resized_float = annotated_image_resized.astype(np.float32) / 255.0  # shape (H, W, 4)

        alpha = annot_resized_float[..., 3:4]  # shape (H, W, 1) => annotation alpha
        annot_rgb = annot_resized_float[..., :3]  # shape (H, W, 3)

        # 3) Blend: final = alpha*annotation + (1-alpha)*combined_image
        blended_rgb = alpha * annot_rgb + (1.0 - alpha) * combined_image_float

    # Finally, export the resulting blended image (with or without annotation).
    # Save as float => .npy
    export_file(blended_rgb, "npy", file_name)
    # Save as .png => scale back to [0..255]
    export_file((blended_rgb * 255).astype(np.uint8), "png", file_name)

def lighten_color(hex_color, factor=0.5):
    """
    Lightens the given color by mixing it with white.
    factor=0.0 -> returns hex_color unmodified
    factor=1.0 -> returns white
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

def display_zoom_and_annotate(base_image, zoom_factor=1.0, file_name="exported_image"):
    """
    Displays the given base image in a toggleable mode:
    - **Zoom Mode:** The user can pan/zoom the image.
    - **Annotate Mode:** The user can draw annotations directly on the image.
    
    The toggle appears in the sidebar.
    """
    # Ensure the image is uint8 for proper display.
    if base_image.dtype != np.uint8:
        base_image = ((base_image - base_image.min()) / (base_image.max() - base_image.min()) * 255).astype(np.uint8)
    
    # Provide a sidebar toggle for mode selection.
    mode = st.sidebar.radio("Select View Mode:", ["Zoom", "Annotate"], key="zoom_annotate_mode")
    
    if mode == "Zoom":
        st.markdown("### Zoom Mode")
        try:
            image_zoom(base_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
        except Exception as e:
            st.error(f"Error in zoom functionality: {e}")
    else:
        st.markdown("### Annotate Mode")
        # Ensure st_canvas is available (it is imported at the top of this file).
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",  # transparent background
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

