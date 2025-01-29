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
        display_zoomable_image_with_annotation(
            original_image, overlay=overlay_image, overlay_type=overlay_type, zoom_factor=zoom_factor, file_name=f"{patient_id}_region{region_id}_slice{slice_name}"
        )
    else:
        st.warning("Original image not found.")


def display_zoomable_image_with_annotation(base_image, overlay=None, overlay_type=None, zoom_factor=1.0, file_name="exported_image"):
    """Display an annotation canvas and export merged annotations with the original image."""
    # Normalize and scale the base image for compatibility
    base_image_normalized = (base_image - base_image.min()) / (base_image.max() - base_image.min())
    base_image_uint8 = (base_image_normalized * 255).astype(np.uint8)

    # Combine images with the overlay
    combined_image = combine_images(base_image_uint8, overlay, overlay_type) if overlay is not None else base_image_uint8

    # # Debugging Zoom Factor
    # st.write(f"Zoom Factor: {zoom_factor}")  # Add this to verify zoom_factor value

    # Display zoomable image
    st.write("<div style='text-align: center;'>", unsafe_allow_html=True)  # Center the image
    try:
        # Pass the zoom factor to the image_zoom function
        image_zoom(combined_image, mode="dragmove", size=750, zoom_factor=zoom_factor)
    except Exception as e:
        st.error(f"Error in image zoom functionality: {e}")
    st.write("</div>", unsafe_allow_html=True)

    # Annotation Canvas
    st.subheader("Annotation Tool")
    drawing_mode = st.sidebar.selectbox(
        "Drawing Tool:",
        ("freedraw", "line", "rect", "circle", "polygon", "point", "transform")
    )
    stroke_width = st.sidebar.slider("Stroke Width:", 1, 25, 3)
    stroke_color = st.sidebar.color_picker("Stroke Color:", "#FF0000")
    realtime_update = st.sidebar.checkbox("Realtime Update", True)

    # Use expanded dimensions for canvas
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Transparent fill
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_image=Image.fromarray(combined_image),  # Use combined image as the background
        update_streamlit=realtime_update,
        height=combined_image.shape[0],
        width=combined_image.shape[1],
        drawing_mode=drawing_mode,
        display_toolbar=True,
        key="annotation_canvas",
    )

    if canvas_result.image_data is not None:
        # 1) Display the annotated image from canvas
        annotated_image = np.array(canvas_result.image_data, dtype=np.uint8)
        st.image(annotated_image, caption="Annotated Image")

        # 2) Resize to match the original dimension
        annotated_image_resized = np.array(
            Image.fromarray(annotated_image).resize((base_image.shape[1], base_image.shape[0]))
        )  # shape (H, W, 4)

        # 3) Convert base image to (H, W, 3) float
        base_image_normalized = (base_image - base_image.min()) / (base_image.max() - base_image.min())
        base_image_rgb = np.stack([base_image_normalized]*3, axis=-1)  # shape (H, W, 3), float 0..1

        # 4) Alpha blend
        annot_resized_float = annotated_image_resized.astype(np.float32) / 255.0  # shape (H, W, 4)
        alpha = annot_resized_float[..., 3:4]  # shape (H, W, 1)
        annot_rgb = annot_resized_float[..., :3]  # shape (H, W, 3)

        blended_rgb = alpha * annot_rgb + (1.0 - alpha) * base_image_rgb  # shape (H, W, 3), float 0..1

        # 5) Export the combined color image
        export_file(blended_rgb, "npy", file_name)  # Save as float in .npy
    export_file((blended_rgb * 255).astype(np.uint8), "png", file_name)  # Save as PNG


def lighten_color(hex_color, factor=0.5):
    """
    Lightens the given color by mixing it with white.
    factor=0.0 -> returns hex_color unmodified
    factor=1.0 -> returns white
    """
    import re
    # Clamp factor to [0..1]
    factor = max(min(factor, 1.0), 0.0)

    # Parse the hex string
    hex_color = hex_color.strip("#")
    # If shorthand like #abc, expand to #aabbcc
    if len(hex_color) == 3:
        hex_color = "".join([c*2 for c in hex_color])

    # Convert to ints
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    # Lighten by blending with white
    r = int(r + (255 - r)*factor)
    g = int(g + (255 - g)*factor)
    b = int(b + (255 - b)*factor)

    return "#{:02x}{:02x}{:02x}".format(r, g, b)
