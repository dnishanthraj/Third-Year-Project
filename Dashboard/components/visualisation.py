# components/visualisation.py
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import streamlit as st
from streamlit_image_zoom import image_zoom
from .file_utils import find_file_in_subfolder, export_file
from .image_utils import load_npy, combine_images
from .constants import MASK_DIR, GRAD_CAM_DIR, OUTPUT_MASK_DIR, IMAGE_DIR

def display_overlay(patient_id, region_id, slice_name, overlay_type, zoom_factor):
    """
    Display an overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask) on the Original Image.
    This function:
      1) Loads the base slice
      2) Loads the overlay (depending on overlay_type)
      3) Merges them and calls display_zoomable_image_with_annotation
    """

    # --- SECTION 1) VIEW OVERLAY ---
    st.markdown("## 1) View Overlay")
    st.markdown("""
    Here, we load the **original slice** from disk, then combine it with the 
    chosen overlay (Grad-CAM, Ground Truth Mask, or Predicted Mask).  
    This gives a quick visual comparison to see how the overlay aligns with the slice.
    """)

    original_file_name = f"{patient_id}_NI{region_id}_slice{slice_name}.npy"
    overlay_file_name = f"{patient_id}_{overlay_type}{region_id}_slice{slice_name}.npy"

    original_path = find_file_in_subfolder(IMAGE_DIR, int(patient_id), original_file_name)
    if overlay_type == "MA":  # ground truth
        overlay_path = find_file_in_subfolder(MASK_DIR, int(patient_id), overlay_file_name)
    else:
        overlay_dir = GRAD_CAM_DIR if overlay_type == "GC" else OUTPUT_MASK_DIR
        overlay_path = os.path.join(overlay_dir, overlay_file_name)

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
    st.markdown("## 3) Export & Annotate")
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
